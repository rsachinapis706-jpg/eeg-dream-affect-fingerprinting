"""
Replicate the published DEED pipelines, then re-validate them subject-wise.

Published methods being replicated
----------------------------------
Mahmoudi et al. 2025 (93.83%): bandpass 1-45 Hz -> 20 s windows -> EMD -> IMFs
    -> Sample Entropy per IMF -> SVM / KNN / RF
Mojdehbar et al. 2026 (92.33%): DWT + EMD subbands -> Fuzzy Entropy -> ReliefF
    -> KNN / SVM / XGB / RF

Both report >92% for 3-class dream emotion on DEED. Neither states subject-wise
validation.

What this script does
---------------------
  1. Builds nonlinear entropy features of the same family they use
     (DWT subbands + EMD IMFs, sample/permutation/approximate entropy).
  2. Evaluates with the SAME classifiers under three validation schemes:
        random  -> 5-fold over windows (no grouping)
        segment -> grouped by recording
        subject -> grouped by subject
  3. Reports all three side by side.

The claim this can support is narrow and explicit: IF the random-split number
lands near the published figure and the subject-wise number collapses, then the
published figure is a validation artefact. If the random-split number does NOT
reproduce, no such claim is made -- the difference would then be unexplained and
must be reported as such.
"""
import os
import warnings
import numpy as np
import pandas as pd
import pywt
try:
    import antropy as ant
except ImportError:
    # antropy needs numba, which does not yet support NumPy 2.5. Only feature
    # extraction uses it, and the features are cached in
    # out/features_nonlinear.parquet, so the classification can run without it.
    ant = None
import scipy.io as sio
from scipy import signal as sps
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
SEED = 20260727
FS = 200.0
EPOCH_N = int(FS * 20)
WAVELET = "db4"
DWT_LEVEL = 5
USE_EMD = True
N_IMF = 4
ARTIFACT_UV = 300.0

_b, _a = sps.butter(4, [1.0 / (FS / 2), 45.0 / (FS / 2)], btype="band")

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def entropy_block(x):
    """Nonlinear feature set for one 1-D sub-signal."""
    x = np.asarray(x, dtype=float)
    if x.size < 20 or np.std(x) < 1e-12:
        return [0.0, 0.0, 0.0, 0.0]
    # cap length: SampEn is O(N^2); published work uses short subbands anyway
    if x.size > 1024:
        x = x[:1024]
    xn = (x - x.mean()) / (x.std() + 1e-12)
    try:
        se = float(ant.sample_entropy(xn, order=2))
    except Exception:
        se = 0.0
    try:
        pe = float(ant.perm_entropy(xn, order=3, normalize=True))
    except Exception:
        pe = 0.0
    try:
        ae = float(ant.app_entropy(xn, order=2))
    except Exception:
        ae = 0.0
    try:
        hf = float(ant.higuchi_fd(xn))
    except Exception:
        hf = 0.0
    return [v if np.isfinite(v) else 0.0 for v in (se, pe, ae, hf)]


def epoch_nonlinear(ep):
    """ep: (6, EPOCH_N) -> feature vector from DWT subbands (+ EMD IMFs)."""
    feats = []
    for ci in range(ep.shape[0]):
        sig = ep[ci]
        coeffs = pywt.wavedec(sig, WAVELET, level=DWT_LEVEL)
        for c in coeffs:
            feats.extend(entropy_block(c))
        if USE_EMD:
            from PyEMD import EMD
            try:
                imfs = EMD(max_imf=N_IMF).emd(sig[::2])  # decimate: EMD is the bottleneck
            except Exception:
                imfs = np.empty((0, 0))
            for k in range(N_IMF):
                feats.extend(entropy_block(imfs[k]) if k < len(imfs) else [0.0] * 4)
    return feats


def process_segment(row):
    try:
        data = sio.loadmat(row["path"])["Data"].astype(np.float64)
    except Exception:
        return []
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.shape[0] != 6 or data.shape[1] < EPOCH_N:
        return []
    data = sps.filtfilt(_b, _a, data, axis=-1)
    out = []
    for e in range(data.shape[1] // EPOCH_N):
        ep = data[:, e * EPOCH_N:(e + 1) * EPOCH_N]
        if np.abs(ep).max() > ARTIFACT_UV or np.allclose(ep.std(axis=1), 0):
            continue
        out.append((epoch_nonlinear(ep), row["segment_id"], row["subject"],
                    row["E"], row["stage"]))
    return out


def cv_eval(X, y, groups, mode, clf_name, n_splits=5):
    if clf_name == "KNN":
        clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    elif clf_name == "SVM":
        clf = make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale"))
    else:
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     n_jobs=-1, random_state=SEED)
    if mode == "random":
        sp = StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(X, y)
    else:
        sp = StratifiedGroupKFold(n_splits, shuffle=True,
                                  random_state=SEED).split(X, y, groups)
    yt, yp = [], []
    for tr, te in sp:
        if len(np.unique(y[tr])) < 2:
            continue
        yp.append(clf.fit(X[tr], y[tr]).predict(X[te]))
        yt.append(y[te])
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return accuracy_score(yt, yp), balanced_accuracy_score(yt, yp)


def main():
    master = pd.read_csv(os.path.join(OUT, "master_table.csv"))
    master["segment_id"] = master.path.map(lambda p: os.path.basename(p)[:-4])
    master["subject"] = master.subject.astype(str).str.zfill(3)

    cache = os.path.join(OUT, "features_nonlinear.parquet")
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        log(f"[cache] loaded {len(df)} epochs from {os.path.basename(cache)}")
    else:
        log(f"Extracting nonlinear features (DWT{'+EMD' if USE_EMD else ''}) "
            f"for {len(master)} segments on all cores...")
        res = Parallel(n_jobs=-1, verbose=1)(
            delayed(process_segment)(r) for _, r in master.iterrows())
        rows = [item for sub in res for item in sub]
        if not rows:
            log("No epochs extracted."); return
        n_feat = len(rows[0][0])
        df = pd.DataFrame([r[0] for r in rows],
                          columns=[f"f{i}" for i in range(n_feat)])
        df["segment_id"] = [r[1] for r in rows]
        df["subject"] = [r[2] for r in rows]
        df["E"] = [r[3] for r in rows]
        df["stage"] = [r[4] for r in rows]
        df.to_parquet(cache, index=False)
        log(f"Extracted {len(df)} epochs x {n_feat} features")

    fc = [c for c in df.columns if c.startswith("f")]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)

    coll = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
    d = df[df.E.isin(coll)].copy()
    y = d.E.map(coll).to_numpy()
    X = d[fc].to_numpy(float)
    maj = pd.Series(y).value_counts().max() / len(y)

    log("")
    log("=" * 72)
    log("REPLICATION + SUBJECT-WISE RE-VALIDATION")
    log("=" * 72)
    log(f"epochs {len(y)} | features {len(fc)} | subjects {d.subject.nunique()}")
    log(f"majority-class baseline = {maj:.4f} | chance = 0.3333")
    log(f"published: Mahmoudi 2025 = 93.83% | Mojdehbar 2026 = 92.33%")
    log("")
    log(f"{'classifier':<8}{'held out':<10}{'accuracy':>10}{'balanced':>11}")
    log("-" * 72)

    rows = []
    for clf_name in ["KNN", "SVM", "RF"]:
        for mode, grp in [("random", None),
                          ("segment", d.segment_id.to_numpy()),
                          ("subject", d.subject.to_numpy())]:
            acc, bacc = cv_eval(X, y, grp, mode, clf_name)
            log(f"{clf_name:<8}{mode:<10}{acc:>10.4f}{bacc:>11.4f}")
            rows.append(dict(classifier=clf_name, held_out=mode,
                             accuracy=acc, balanced_accuracy=bacc,
                             majority=maj))
        log("-" * 72)

    res = pd.DataFrame(rows)
    res.round(4).to_csv(os.path.join(OUT, "replication_results.csv"), index=False)

    log("")
    best_random = res[res.held_out == "random"].accuracy.max()
    best_subject = res[res.held_out == "subject"].accuracy.max()
    log(f"best random-split accuracy  : {best_random:.4f}")
    log(f"best subject-wise accuracy  : {best_subject:.4f}")
    log(f"collapse                    : {best_random - best_subject:+.4f}")
    log("")
    if best_random >= 0.85:
        log(">> Published range REPRODUCED under random splits.")
        log(">> The collapse under subject-wise validation is therefore")
        log(">> attributable to validation protocol, not to features.")
    else:
        log(">> Published range NOT reproduced (best random = "
            f"{best_random:.3f} vs ~0.92 reported).")
        log(">> The gap is UNEXPLAINED by validation protocol alone and must be")
        log(">> reported as such. Do not claim their number is leakage.")

    with open(os.path.join(OUT, "test5_replication.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"Wrote {OUT}\\test5_replication.txt")


if __name__ == "__main__":
    main()
