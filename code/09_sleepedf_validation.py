"""
External validation of the feature pipeline on Sleep-EDF (expert-scored).

Why this exists
---------------
On DEED the sleep-stage positive control reaches only ~0.37-0.45 balanced
accuracy. That leaves two incompatible explanations, and every downstream null
result is uninterpretable until we know which one holds:

    (A) the FEATURES are too weak       -> the DEED nulls mean nothing
    (B) the DEED stage LABELS are noisy -> the features are fine, and the DEED
                                           nulls are real findings about DEED

Sleep-EDF 1.0.0 settles it. It carries expert-scored hypnograms, so the labels
are known-good. Running the IDENTICAL feature code and the IDENTICAL subject-wise
protocol there gives a clean read:

    high accuracy on Sleep-EDF -> (B): pipeline validated, DEED nulls stand
    low accuracy on Sleep-EDF  -> (A): pipeline is the problem, fix features first

This is a harder test than DEED, not an easier one: Sleep-EDF gives us 2 EEG
channels where DEED gives 6.

Hypnogram coding (Sleep-EDF 1.0.0): 0=W 1=N1 2=N2 3=N3 4=N4 5=REM 6=movement 9=unscored
"""
import os
import glob
import warnings
import numpy as np
import pandas as pd
import mne
from scipy import signal as sps
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
EDF_DIR = os.path.join(ROOT, "sleepedf", "sleep-edf-database-1.0.0")

FS = 100.0                 # Sleep-EDF native
EPOCH_S = 20               # matched to the DEED pipeline
EPOCH_N = int(FS * EPOCH_S)
ARTIFACT_UV = 300.0
SEED = 20260727

_trapz = getattr(np, "trapezoid", None) or np.trapz
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
         "sigma": (12.0, 16.0), "beta": (16.0, 30.0), "gamma": (30.0, 45.0)}

_b, _a = sps.butter(4, [0.5 / (FS / 2), 45.0 / (FS / 2)], btype="band")
STAGE = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "N3", 5: "REM"}

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def hjorth(x):
    dx = np.diff(x); ddx = np.diff(dx)
    v0 = np.var(x) + 1e-12; v1 = np.var(dx) + 1e-12; v2 = np.var(ddx) + 1e-12
    mob = np.sqrt(v1 / v0)
    return mob, np.sqrt(v2 / v1) / (mob + 1e-12)


def epoch_features(ep):
    """Identical feature definitions to the DEED pipeline (06_features_filtered)."""
    feats = {}
    nper = int(FS * 4)
    freqs, psd = sps.welch(ep, fs=FS, nperseg=nper, noverlap=nper // 2, axis=-1)
    tm = (freqs >= 0.5) & (freqs <= 45.0)
    total = _trapz(psd[:, tm], freqs[tm], axis=-1) + 1e-12
    for ci in range(ep.shape[0]):
        p = psd[ci]
        for bn, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            bp = _trapz(p[mask], freqs[mask]) + 1e-12
            feats[f"ch{ci}_{bn}_abs"] = np.log10(bp)
            feats[f"ch{ci}_{bn}_rel"] = bp / total[ci]
        pn = p[tm] / (p[tm].sum() + 1e-12)
        feats[f"ch{ci}_specent"] = float(-(pn * np.log2(pn + 1e-12)).sum() / np.log2(len(pn)))
        c = np.cumsum(p[tm]); c /= c[-1] + 1e-12
        ftm = freqs[tm]
        # searchsorted can return len(c) when the cumulative sum never exceeds
        # 0.95 due to float rounding; clip to the last valid bin.
        feats[f"ch{ci}_sef95"] = float(ftm[min(np.searchsorted(c, 0.95), len(ftm) - 1)])
        am = (freqs >= 2.0) & (freqs <= 40.0)
        feats[f"ch{ci}_apslope"] = float(
            np.polyfit(np.log10(freqs[am]), np.log10(p[am] + 1e-20), 1)[0])
        mob, comp = hjorth(ep[ci])
        feats[f"ch{ci}_hj_mob"] = float(mob)
        feats[f"ch{ci}_hj_comp"] = float(comp)
        feats[f"ch{ci}_logvar"] = float(np.log10(np.var(ep[ci]) + 1e-12))
    return feats


def load_recording(rec):
    sid = os.path.basename(rec)[:-4]
    hyp = os.path.join(EDF_DIR, f"{sid}_hyp.edf")
    if not os.path.exists(hyp):
        return []
    raw = mne.io.read_raw_edf(rec, preload=True)
    eeg = [c for c in raw.ch_names if c.startswith("EEG")]
    if not eeg:
        return []
    raw.pick(eeg)
    if raw.info["sfreq"] != FS:
        raw.resample(FS)
    data = raw.get_data() * 1e6            # mne returns volts -> microvolts
    data = sps.filtfilt(_b, _a, data, axis=-1)

    codes = np.round(mne.io.read_raw_edf(hyp, preload=True).get_data()[0]).astype(int)

    rows = []
    for i, code in enumerate(codes):
        if code not in STAGE:
            continue
        # each hypnogram sample scores a 30 s epoch; take the first EPOCH_S of it
        s0 = int(i * 30 * FS)
        ep = data[:, s0:s0 + EPOCH_N]
        if ep.shape[1] < EPOCH_N or np.abs(ep).max() > ARTIFACT_UV:
            continue
        if np.allclose(ep.std(axis=1), 0):
            continue
        f = epoch_features(ep)
        f.update(subject=sid, stage=STAGE[code], epoch=i)
        rows.append(f)
    return rows


def loso(X, y, groups, clf_fn):
    yt, yp = [], []
    for s in np.unique(groups):
        te = groups == s
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        yp.append(clf_fn().fit(X[tr], y[tr]).predict(X[te]))
        yt.append(y[te])
    return np.concatenate(yt), np.concatenate(yp)


def main():
    cache = os.path.join(OUT, "sleepedf_features.parquet")
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        log(f"[cache] {len(df)} epochs")
    else:
        recs = sorted(f for f in glob.glob(os.path.join(EDF_DIR, "*.edf"))
                      if not f.endswith("_hyp.edf"))
        log(f"Loading {len(recs)} Sleep-EDF recordings...")
        rows = []
        for r in recs:
            got = load_recording(r)
            log(f"  {os.path.basename(r)}: {len(got)} epochs")
            rows += got
        df = pd.DataFrame(rows)
        df.to_parquet(cache, index=False)

    fc = [c for c in df.columns if c.startswith("ch")]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)

    log("")
    log("=" * 70)
    log("SLEEP-EDF EXTERNAL VALIDATION (expert-scored labels)")
    log("=" * 70)
    log(f"epochs {len(df)} | subjects {df.subject.nunique()} | features {len(fc)}")
    log(f"EEG channels: 2 (DEED has 6 -- this is the harder test)")
    log("")
    log(df.stage.value_counts().to_string())
    log("")

    results = []
    for name, sel in [("W/N1/N2 (matches DEED control)", ["W", "N1", "N2"]),
                      ("5-class (W/N1/N2/N3/REM)", ["W", "N1", "N2", "N3", "REM"])]:
        d = df[df.stage.isin(sel)]
        X = d[fc].to_numpy(float)
        y = d.stage.to_numpy()
        g = d.subject.to_numpy()
        maj = pd.Series(y).value_counts().max() / len(y)

        for clf_name, fn in [
            ("LogReg", lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced"))),
            ("RF", lambda: RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                random_state=SEED, class_weight="balanced")),
        ]:
            yt, yp = loso(X, y, g, fn)
            acc = accuracy_score(yt, yp)
            bacc = balanced_accuracy_score(yt, yp)
            log(f"{name:<34} {clf_name:<7} acc={acc:.4f}  bacc={bacc:.4f}  "
                f"(majority {maj:.4f}, chance {1/len(sel):.4f})")
            results.append(dict(task=name, clf=clf_name, accuracy=acc,
                                balanced_accuracy=bacc, majority=maj,
                                chance=1 / len(sel), n=len(y)))
        log("")

    best = max(r["balanced_accuracy"] for r in results
               if r["task"].startswith("W/N1/N2"))
    log("-" * 70)
    log(f"DEED  W/N1/N2 positive control : bacc 0.3730 - 0.4459")
    log(f"Sleep-EDF same task, same code : bacc {best:.4f}")
    log("")
    if best >= 0.60:
        log(">> PIPELINE VALIDATED. The features decode sleep stage well on")
        log(">> expert-scored data. The weak DEED control is therefore a property")
        log(">> of DEED (label noise / segment heterogeneity), NOT of the method.")
        log(">> The DEED null results are interpretable and reportable.")
    else:
        log(">> PIPELINE NOT VALIDATED. The features underperform even on")
        log(">> expert-scored data. The DEED nulls cannot be attributed to DEED;")
        log(">> the feature set must be improved before any claim is made.")

    pd.DataFrame(results).round(4).to_csv(
        os.path.join(OUT, "sleepedf_validation.csv"), index=False)
    with open(os.path.join(OUT, "test6_sleepedf.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"Wrote {OUT}\\test6_sleepedf.txt")


if __name__ == "__main__":
    main()
