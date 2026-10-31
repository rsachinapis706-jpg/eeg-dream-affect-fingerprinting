"""
Is participant fingerprinting specific to EEG?

Reviewer 1 asks for a comparison with ECG. Neither corpus used here records ECG:
DEED distributes six EEG channels and nothing else, and Sleep-EDF Expanded
records EEG, EOG, submental EMG, oro-nasal respiration and rectal temperature.
The literal comparison is therefore not available on these data.

The question behind it is answerable, and is more general than the ECG case.
If the confound reported in this work were peculiar to EEG, the argument would
have limited reach. If instead participant identity is recoverable from any
biosignal channel recorded on the same person, then window-level partitioning is
unsafe for every modality, including ECG, and the scope of the finding widens
rather than narrows.

Design
------
Identity is decoded from each modality SEPARATELY, on the same participants, the
same epochs and the same classifier. Because the channels are sampled at very
different rates -- 100 Hz for EEG and EOG, 1 Hz for EMG, respiration and
temperature -- band-power features cannot be computed for all of them. A
rate-agnostic statistical feature set is therefore used for every modality, so
the comparison is on equal terms:

    log-variance, mean, standard deviation, skewness, kurtosis,
    5/25/50/75/95th percentiles, inter-percentile range,
    Hjorth mobility and complexity, mean absolute first difference

Channels recorded at 1 Hz are decimated back to their native samples before
features are computed, since the EDF reader returns every channel at the file's
maximum rate and the interpolated points carry no independent information.

Epochs are 30 s, the standard scoring epoch, so that the 1 Hz channels contribute
30 real samples each. This differs from the 20 s window used elsewhere in the
paper; the analysis is self-contained and no number here is compared against the
20 s results.

Two partitionings are reported. The first is window-shuffled, which is the
protocol under examination. The second holds out one whole night per participant
and trains on the remaining nights, so identity must transfer across recording
sessions.

The held-out-night design replaces grouped k-fold here for a specific reason.
Participants in sleep-cassette contribute about two recordings each, so a grouped
k-fold can place both of a participant's recordings in the same test fold,
leaving that participant with no training data and making their epochs
unclassifiable for a mechanical reason rather than a physiological one. Holding
out one night at a time keeps every participant represented in training and
measures what the column is supposed to measure.
"""
import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import mne
import pyedflib
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
EXP = os.path.join(ROOT, "Sleep-EDF Expanded",
                   "sleep-edf-database-expanded-1.0.0", "sleep-cassette")

EPOCH_S = 30
N_SUBJECTS = 20
EPOCHS_PER_REC = 150
SEEDS = [20260727, 20260728, 20260729]
N_SPLITS = 5

MODALITIES = {
    "EEG (2 ch)":        ["EEG Fpz-Cz", "EEG Pz-Oz"],
    "EOG (1 ch)":        ["EOG horizontal"],
    "EMG (1 ch)":        ["EMG submental"],
    "Respiration (1 ch)": ["Resp oro-nasal"],
    "Temperature (1 ch)": ["Temp rectal"],
}

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def hjorth(x):
    dx = np.diff(x); ddx = np.diff(dx)
    v0 = np.var(x) + 1e-12; v1 = np.var(dx) + 1e-12; v2 = np.var(ddx) + 1e-12
    mob = np.sqrt(v1 / v0)
    return mob, np.sqrt(v2 / v1) / (mob + 1e-12)


def stat_features(x, tag):
    """Rate-agnostic descriptors; identical definitions for every modality."""
    q = np.percentile(x, [5, 25, 50, 75, 95])
    mob, comp = hjorth(x)
    return {
        f"{tag}_logvar": float(np.log10(np.var(x) + 1e-12)),
        f"{tag}_mean": float(np.mean(x)),
        f"{tag}_sd": float(np.std(x)),
        f"{tag}_skew": float(stats.skew(x)),
        f"{tag}_kurt": float(stats.kurtosis(x)),
        f"{tag}_p05": float(q[0]), f"{tag}_p25": float(q[1]),
        f"{tag}_p50": float(q[2]), f"{tag}_p75": float(q[3]),
        f"{tag}_p95": float(q[4]), f"{tag}_iqr": float(q[3] - q[1]),
        f"{tag}_range": float(q[4] - q[0]),
        f"{tag}_hjmob": float(mob), f"{tag}_hjcomp": float(comp),
        f"{tag}_mad1": float(np.mean(np.abs(np.diff(x)))),
    }


def native_rates(path):
    e = pyedflib.EdfReader(path)
    r = {e.getLabel(i): e.getSampleFrequency(i)
         for i in range(e.signals_in_file)}
    e.close()
    return r


def process(path, rng):
    stem = os.path.basename(path).split("-")[0]
    m = re.match(r"(S[CT])\d(\d{2})(\d)", stem)
    subject = m.group(1) + m.group(2) if m else stem

    rates = native_rates(path)
    wanted = [c for chans in MODALITIES.values() for c in chans]
    try:
        raw = mne.io.read_raw_edf(path, preload=False)
        have = [c for c in wanted if c in raw.ch_names]
        if len(have) < len(wanted):
            return []
        raw.pick(have)
        # keep a central 4 h span: avoids the long pre-sleep wake block and
        # bounds memory, without selecting on anything participant-specific
        total = raw.times[-1]
        mid = total / 2.0
        raw.crop(tmin=max(0, mid - 7200), tmax=min(total, mid + 7200))
        raw.load_data()
        fs = raw.info["sfreq"]
        data = raw.get_data()
        del raw
    except Exception:
        return []

    n = int(EPOCH_S * fs)
    n_ep = data.shape[1] // n
    if n_ep < 10:
        return []
    idx = rng.choice(n_ep, size=min(EPOCHS_PER_REC, n_ep), replace=False)

    rows = []
    for e in idx:
        seg = data[:, e * n:(e + 1) * n]
        if not np.isfinite(seg).all():
            continue
        f = {}
        ok = True
        for ci, ch in enumerate(have):
            x = seg[ci]
            step = int(round(fs / rates.get(ch, fs)))
            if step > 1:                     # recover native samples
                x = x[::step]
            if len(x) < 8 or np.std(x) == 0:
                ok = False
                break
            f.update(stat_features(x, ch.replace(" ", "_")))
        if not ok:
            continue
        f.update(subject=subject, recording=stem, epoch=int(e))
        rows.append(f)
    return rows


def decode(X, y, groups, mode, seed):
    if mode == "random":
        sp = StratifiedKFold(N_SPLITS, shuffle=True,
                             random_state=seed).split(X, y)
    else:
        sp = StratifiedGroupKFold(N_SPLITS, shuffle=True,
                                  random_state=seed).split(X, y, groups)
    yt, yp = [], []
    for tr, te in sp:
        if len(np.unique(y[tr])) < 2:
            continue
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   n_jobs=-1, random_state=seed).fit(X[tr], y[tr])
        yp.append(m.predict(X[te])); yt.append(y[te])
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return accuracy_score(yt, yp), balanced_accuracy_score(yt, yp)


def decode_night_out(X, y, rec, seed):
    """Hold out one whole night per participant; train on their other nights.

    Only participants with at least two recordings contribute, since a
    participant with a single night cannot have one held out while remaining in
    the training set. The number of folds is the largest number of recordings
    any participant has.
    """
    order = {}
    for s in np.unique(y):
        rs = sorted(np.unique(rec[y == s]))
        for k, r in enumerate(rs):
            order[r] = k
    slot = np.array([order[r] for r in rec])
    counts = pd.Series(y).groupby(pd.Series(y)).size()
    multi = {s for s in np.unique(y)
             if len(np.unique(rec[y == s])) >= 2}
    keep = np.array([s in multi for s in y])
    if keep.sum() == 0:
        return np.nan, np.nan, 0

    Xk, yk, sk = X[keep], y[keep], slot[keep]
    yt, yp = [], []
    for k in np.unique(sk):
        te = sk == k
        tr = ~te
        if te.sum() == 0 or len(np.unique(yk[tr])) < 2:
            continue
        # score only participants that are present in the training partition
        seen = np.isin(yk[te], np.unique(yk[tr]))
        if seen.sum() == 0:
            continue
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   n_jobs=-1, random_state=seed).fit(Xk[tr],
                                                                     yk[tr])
        idx = np.where(te)[0][seen]
        yp.append(m.predict(Xk[idx])); yt.append(yk[idx])
    if not yt:
        return np.nan, np.nan, len(multi)
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return (accuracy_score(yt, yp), balanced_accuracy_score(yt, yp),
            len(multi))


def main():
    rng = np.random.default_rng(20260727)
    psgs = sorted(glob.glob(os.path.join(EXP, "*PSG.edf")))

    subs, keep = set(), []
    for p in psgs:
        stem = os.path.basename(p).split("-")[0]
        m = re.match(r"(S[CT])\d(\d{2})(\d)", stem)
        s = m.group(1) + m.group(2) if m else stem
        if len(subs) < N_SUBJECTS or s in subs:
            subs.add(s); keep.append(p)

    log("=" * 78)
    log("IS PARTICIPANT FINGERPRINTING SPECIFIC TO EEG?")
    log("=" * 78)
    log("Sleep-EDF Expanded, sleep-cassette. Identity decoded from each")
    log("modality separately with an identical rate-agnostic feature set.")
    log(f"target participants {N_SUBJECTS} | recordings {len(keep)} | "
        f"epoch {EPOCH_S}s")
    log("")

    cache = os.path.join(OUT, "crossmodal_features.parquet")
    if os.path.exists(cache):
        d = pd.read_parquet(cache)
        log(f"  loaded cached features: {len(d)} epochs")
    else:
        rows = []
        for i, p in enumerate(keep, 1):
            r = process(p, rng)
            rows += r
            if i % 10 == 0 or i == len(keep):
                log(f"  processed {i}/{len(keep)} recordings, {len(rows)} epochs")
        d = pd.DataFrame(rows)
        d = d.replace([np.inf, -np.inf], np.nan).dropna()
        d.to_parquet(cache, index=False)

    nsub = d.subject.nunique()
    chance = 1.0 / nsub
    log("")
    log(f"epochs {len(d)} | participants {nsub} | recordings "
        f"{d.recording.nunique()} | chance {chance:.4f}")
    log("")

    y = d.subject.to_numpy()
    rec = d.recording.to_numpy()

    log(f"{'modality':<22}{'feat':>5}{'shuffled':>11}{'night held out':>16}"
        f"{'x chance':>10}")
    log("-" * 78)

    groups = list(MODALITIES.items()) + [("ALL non-EEG", None)]
    res = []
    for name, chans in groups:
        if chans is None:
            cols = [c for c in d.columns
                    if not c.startswith("EEG_")
                    and c not in ("subject", "recording", "epoch")]
            nch = 4
        else:
            cols = [c for c in d.columns
                    if any(c.startswith(ch.replace(" ", "_") + "_")
                           for ch in chans)]
            nch = len(chans)
        if not cols:
            continue
        X = d[cols].to_numpy(float)
        a_r = np.mean([decode(X, y, None, "random", s)[0] for s in SEEDS])
        outs = [decode_night_out(X, y, rec, s) for s in SEEDS]
        a_n = float(np.mean([o[0] for o in outs]))
        n_multi = outs[0][2]
        log(f"{name:<22}{len(cols):>5}{a_r:>11.4f}{a_n:>16.4f}"
            f"{a_r / chance:>9.0f}x")
        res.append(dict(modality=name, channels=nch, features=len(cols),
                        acc_shuffled=a_r, acc_night_held_out=a_n,
                        chance=chance, participants=nsub,
                        participants_multinight=n_multi,
                        epochs=len(d), n_seeds=len(SEEDS)))

    r = pd.DataFrame(res)
    r.round(4).to_csv(os.path.join(OUT, "crossmodal_identity.csv"), index=False)

    log("")
    log("=" * 78)
    log("INTERPRETATION")
    log("=" * 78)
    best_non = r[r.modality != "EEG (2 ch)"].acc_shuffled.max()
    eeg = float(r[r.modality == "EEG (2 ch)"].acc_shuffled.iloc[0])
    log(f"EEG identity accuracy            : {eeg:.4f}")
    log(f"best non-EEG modality            : {best_non:.4f}")
    log(f"chance                           : {chance:.4f}")
    log("")
    if best_non > 5 * chance:
        log("Participant identity is recoverable from non-EEG channels at")
        log("many times chance, including channels carrying no cognitive")
        log("content. Fingerprinting is a property of repeated within-person")
        log("recording, not of electroencephalography. Window-level")
        log("partitioning is therefore unsafe for any biosignal modality,")
        log("ECG included, and the scope of this paper's finding is wider")
        log("than EEG.")
    else:
        log("Non-EEG channels do not identify participants under this feature")
        log("set. Report the EEG result as modality-specific.")

    with open(os.path.join(OUT, "test30_crossmodal.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"\nWrote {OUT}\\crossmodal_identity.csv")


if __name__ == "__main__":
    main()
