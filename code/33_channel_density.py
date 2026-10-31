"""
Does spatial resolution limit decodability? A channel-density sweep on Tononi.

Every negative result so far used 6 channels: DEED has only 6, and Tononi was
restricted to 6 for comparability. That leaves a legitimate objection open --
that dream-related information is spatially distributed and six electrodes
cannot capture it. Tononi records 256 channels, so the objection is directly
testable.

Design
------
Features are extracted once for a 64-electrode geometric subset, then the sweep
evaluates nested subsets of 6, 12, 24, 48 and 64 channels. Electrodes are chosen
by farthest-point sampling on the montage sphere, so each subset covers the scalp
as evenly as its size allows and smaller subsets are contained in larger ones.
This isolates channel count from placement.

Interpretation, fixed in advance
--------------------------------
  accuracy rises with channel count  -> spatial resolution was a real limit and
                                        the negative result is qualified
  accuracy flat across counts        -> the limit is not spatial; six channels
                                        were sufficient to detect what is there

A caveat that must be reported either way: 64 channels yields 1152 features
against 668 windows, so p >> n. Regularisation is fixed across all subsets and
never tuned, and the subject-wise protocol is unchanged, but the high-dimensional
settings are more prone to variance and should be read with that in mind.
"""
import os
import re
import glob
import zipfile
import tempfile
import warnings
import numpy as np
import pandas as pd
import mne
from scipy import signal as sps
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
SRC = os.path.join(ROOT, "tononi")
CACHE = os.path.join(OUT, "tononi_64ch.parquet")

FS = 200.0
EPOCH_S = 20
EPOCH_N = int(FS * EPOCH_S)
ARTIFACT_UV = 300.0
N_MAX = 64
SEED = 20260727
SUBSETS = [6, 12, 24, 48, 64]

_trapz = getattr(np, "trapezoid", None) or np.trapz
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
         "sigma": (12.0, 16.0), "beta": (16.0, 30.0), "gamma": (30.0, 45.0)}
_b, _a = sps.butter(4, [0.5 / (FS / 2), 45.0 / (FS / 2)], btype="band")

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def read_montage():
    z = zipfile.ZipFile(os.path.join(SRC, "part1.zip"))
    name = [n for n in z.namelist() if n.endswith(".sfp")][0]
    toks = z.read(name).decode("utf8", "ignore").split()
    pos, i = {}, 0
    while i + 3 < len(toks) + 1:
        lab = toks[i]
        try:
            v = np.array([float(toks[i + 1]), float(toks[i + 2]),
                          float(toks[i + 3])], float)
        except (ValueError, IndexError):
            break
        n = np.linalg.norm(v)
        if n > 0:
            pos[lab] = v / n
        i += 4
    return pos


def farthest_point(pos, n):
    """Greedy maximin sampling: each new electrode maximises its distance to
    those already chosen, giving nested, evenly spread subsets."""
    keys = list(pos.keys())
    V = np.stack([pos[k] for k in keys])
    start = int(np.argmax(V[:, 2]))          # begin at the vertex
    chosen = [start]
    d = np.linalg.norm(V - V[start], axis=1)
    while len(chosen) < n:
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(V - V[nxt], axis=1))
    return [keys[i] for i in chosen]


def to_channel(label):
    return f"Chan {label}" if str(label).isdigit() else "Chan 257"


def hjorth(x):
    dx = np.diff(x); ddx = np.diff(dx)
    v0 = np.var(x) + 1e-12; v1 = np.var(dx) + 1e-12; v2 = np.var(ddx) + 1e-12
    mob = np.sqrt(v1 / v0)
    return mob, np.sqrt(v2 / v1) / (mob + 1e-12)


def epoch_features(ep):
    feats = {}
    nper = int(FS * 4)
    freqs, psd = sps.welch(ep, fs=FS, nperseg=nper, noverlap=nper // 2, axis=-1)
    tm = (freqs >= 0.5) & (freqs <= 45.0)
    total = _trapz(psd[:, tm], freqs[tm], axis=-1) + 1e-12
    ftm = freqs[tm]
    am = (freqs >= 2.0) & (freqs <= 40.0)
    for ci in range(ep.shape[0]):
        p = psd[ci]
        for bn, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            bp = _trapz(p[mask], freqs[mask]) + 1e-12
            feats[f"ch{ci}_{bn}_abs"] = np.log10(bp)
            feats[f"ch{ci}_{bn}_rel"] = bp / total[ci]
        pn = p[tm] / (p[tm].sum() + 1e-12)
        feats[f"ch{ci}_specent"] = float(
            -(pn * np.log2(pn + 1e-12)).sum() / np.log2(len(pn)))
        c = np.cumsum(p[tm]); c /= c[-1] + 1e-12
        feats[f"ch{ci}_sef95"] = float(ftm[min(np.searchsorted(c, 0.95),
                                              len(ftm) - 1)])
        feats[f"ch{ci}_apslope"] = float(
            np.polyfit(np.log10(freqs[am]), np.log10(p[am] + 1e-20), 1)[0])
        mob, comp = hjorth(ep[ci])
        feats[f"ch{ci}_hj_mob"] = float(mob)
        feats[f"ch{ci}_hj_comp"] = float(comp)
        feats[f"ch{ci}_logvar"] = float(np.log10(np.var(ep[ci]) + 1e-12))
    return feats


def build(order):
    """Extract 64-channel features once, in the farthest-point order."""
    import io as _io
    z0 = zipfile.ZipFile(os.path.join(SRC, "part1.zip"))
    reps = []
    for n in z0.namelist():
        if n.endswith(".xlsx") and "Reports_" in n:
            sid = n.split("Reports_")[1].replace(".xlsx", "").lower()
            try:
                d = pd.read_excel(_io.BytesIO(z0.read(n)))
            except Exception:
                continue
            d["subject"] = sid
            reps.append(d)
    rep = pd.concat(reps, ignore_index=True)
    rep.columns = [str(c).strip() for c in rep.columns]

    want = [to_channel(k) for k in order]
    rows = []
    for arc in sorted(glob.glob(os.path.join(SRC, "Part*-s*.zip"))):
        try:
            z = zipfile.ZipFile(arc)
        except Exception:
            continue
        for n in sorted(x for x in z.namelist() if x.endswith(".edf")):
            base = os.path.basename(n)[:-4]
            m = re.match(r"s(\d+)_ep(\d+)", base)
            if not m:
                continue
            subj, ep_no = f"s{m.group(1)}", int(m.group(2))
            lab = rep[(rep.subject == subj) & (rep["Awakening"] == ep_no)]
            if not len(lab) or lab["Dreaming"].iloc[0] not in (1, 2, 3):
                continue
            with tempfile.TemporaryDirectory() as td:
                try:
                    raw = mne.io.read_raw_edf(z.extract(n, td), preload=True)
                    have = [c for c in want if c in raw.ch_names]
                    if len(have) < N_MAX:
                        continue
                    raw.pick(have)          # preserves `want` order
                    raw.reorder_channels(have)
                    raw.resample(FS)
                    data = sps.filtfilt(_b, _a, raw.get_data() * 1e6, axis=-1)
                    del raw
                except Exception:
                    continue
            for e in range(data.shape[1] // EPOCH_N):
                seg = data[:, e * EPOCH_N:(e + 1) * EPOCH_N]
                # Window inclusion is decided on the first six electrodes only,
                # so every density condition is evaluated on an IDENTICAL set of
                # windows. Rejecting a window whenever any of 64 channels
                # exceeds threshold would discard far more windows at high
                # density than at low, and the sweep would then compare
                # different samples rather than different channel counts.
                core = seg[:6]
                if np.abs(core).max() > ARTIFACT_UV or np.allclose(
                        core.std(axis=1), 0):
                    continue
                # Additional channels are clipped rather than allowed to
                # dominate the spectra of a retained window.
                seg = np.clip(seg, -ARTIFACT_UV, ARTIFACT_UV)
                f = epoch_features(seg)
                f.update(subject=subj, recording=base,
                         dreaming=int(lab["Dreaming"].iloc[0]))
                rows.append(f)
        log(f"  {os.path.basename(arc):<18} windows {len(rows)}")
    df = pd.DataFrame(rows)
    df.to_parquet(CACHE, index=False)
    return df


def cv(X, y, groups, mode, model):
    if model == "logreg":
        mk = lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced"))
    else:
        mk = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                            n_jobs=-1, random_state=SEED)
    if mode == "random":
        sp = StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)
    else:
        sp = StratifiedGroupKFold(5, shuffle=True,
                                  random_state=SEED).split(X, y, groups)
    yt, yp = [], []
    for tr, te in sp:
        if len(np.unique(y[tr])) < 2:
            continue
        yp.append(mk().fit(X[tr], y[tr]).predict(X[te]))
        yt.append(y[te])
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return accuracy_score(yt, yp), balanced_accuracy_score(yt, yp)


def main():
    pos = read_montage()
    order = farthest_point(pos, N_MAX)
    log("=" * 76)
    log("CHANNEL-DENSITY SWEEP -- DOES SPATIAL RESOLUTION LIMIT DECODABILITY?")
    log("=" * 76)
    log(f"montage electrodes {len(pos)} | farthest-point subset {N_MAX}")
    log(f"first 8 selected: {order[:8]}")
    log("")

    if os.path.exists(CACHE):
        df = pd.read_parquet(CACHE)
        log(f"[cache] {len(df)} windows")
    else:
        log("extracting 64-channel features (one pass)...")
        df = build(order)

    fc_all = [c for c in df.columns if c.startswith("ch")]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=fc_all)
    y = df.dreaming.astype(str).to_numpy()
    subj = df.subject.astype(str).to_numpy()
    rec = df.recording.astype(str).to_numpy()

    log(f"windows {len(y)} | participants {len(np.unique(subj))}")
    log(f"chance {1/len(np.unique(y)):.4f} | majority "
        f"{pd.Series(y).value_counts().max()/len(y):.4f}")
    log("")

    log(f"{'channels':>9}{'features':>10}{'model':>9}"
        f"{'random':>9}{'subject':>9}{'bal.subj':>10}")
    log("-" * 76)
    rows = []
    for n in SUBSETS:
        keep = [c for c in fc_all
                if int(re.match(r"ch(\d+)_", c).group(1)) < n]
        X = df[keep].to_numpy(float)
        for model in ["logreg", "rf"]:
            ar, _ = cv(X, y, None, "random", model)
            asu, bsu = cv(X, y, subj, "grouped", model)
            log(f"{n:>9}{len(keep):>10}{model:>9}{ar:>9.4f}{asu:>9.4f}"
                f"{bsu:>10.4f}")
            rows.append(dict(channels=n, features=len(keep), model=model,
                             acc_random=ar, acc_subject=asu,
                             bal_subject=bsu))
    log("")

    r = pd.DataFrame(rows)
    best = r.loc[r.bal_subject.idxmax()]
    six = r[r.channels == 6].bal_subject.max()
    log("=" * 76)
    log("VERDICT")
    log("=" * 76)
    log(f"best subject-wise balanced accuracy : {best.bal_subject:.4f} "
        f"({int(best.channels)} channels, {best.model})")
    log(f"same at 6 channels                  : {six:.4f}")
    log(f"gain from added channels            : {best.bal_subject - six:+.4f}")
    log(f"chance                              : {1/3:.4f}")
    log("")
    if best.bal_subject > 0.40:
        log("Added channels recover signal. The negative result is limited to")
        log("low-density montages and must be qualified accordingly.")
    elif best.bal_subject - six > 0.03:
        log("A modest gain with density, but performance remains near chance.")
        log("Report the trend without claiming recovered signal.")
    else:
        log("No gain from spatial resolution. Performance is flat from 6 to 64")
        log("channels and remains at chance throughout, so channel count is")
        log("not the limiting factor and the six-channel result is not an")
        log("artefact of low density.")

    r.round(4).to_csv(os.path.join(OUT, "channel_density.csv"), index=False)
    with open(os.path.join(OUT, "test26_channel_density.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"\nWrote {OUT}\\test26_channel_density.txt")


if __name__ == "__main__":
    main()
