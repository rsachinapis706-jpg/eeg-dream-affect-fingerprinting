"""
Rebuild DEED windows at several overlap fractions, from the source recordings.

Why this exists
---------------
Reviewer 1 asks why the published DEED analyses report 92-94% while our most
permissive protocol reaches 0.4711. Leakage as measured so far (window-shuffled
partitioning, selection on all data, oversampling before the split) accounts for
a large share of that gap but not all of it, so an honest answer needs the
remainder identified rather than asserted.

The most likely unmeasured multiplier is WINDOW OVERLAP. Every analysis in this
project used 20 s NON-OVERLAPPING windows. Neither published description states
an overlap. If windows overlap, adjacent windows are near-duplicates of one
another: at 90% overlap two neighbouring windows share 18 of their 20 seconds.
Under window-shuffled cross-validation those near-duplicates land on both sides
of the partition, so the classifier is scored partly on data it has already
seen. That is a different mechanism from participant fingerprinting and it
multiplies with it.

This script only builds the data. It cuts each recording at overlap fractions
0, 0.5, 0.75 and 0.9 and extracts the SAME 108 spectral features used everywhere
else in this project, so the sweep is comparable with every existing result.
36_overlap_ladder.py does the evaluation.

What is and is not claimed
--------------------------
Nothing here asserts that any published study used overlapping windows. Their
descriptions do not say. What the sweep establishes is the accuracy REACHABLE on
this dataset using choices that are common in the literature and absent from the
published method descriptions, together with the participant-separated value for
the identical configuration. The gap between those two columns is the claim.

Preprocessing, feature definitions, the artifact threshold and the label mapping
are copied unchanged from 06_features_filtered.py and 07_leakage_demo.py.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal as sps
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")

FS = 200.0
EPOCH_S = 20
EPOCH_N = int(FS * EPOCH_S)
ARTIFACT_UV = 300.0
OVERLAPS = [0.0, 0.5, 0.75, 0.9]
N_JOBS = 6

_trapz = getattr(np, "trapezoid", None) or np.trapz

BANDS = {
    "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0), "beta": (16.0, 30.0), "gamma": (30.0, 45.0),
}

_bp_b, _bp_a = sps.butter(4, [0.5 / (FS / 2), 45.0 / (FS / 2)], btype="band")
_nt_b, _nt_a = sps.iirnotch(50.0, Q=30.0, fs=FS)


def preprocess(x):
    x = sps.filtfilt(_bp_b, _bp_a, x, axis=-1)
    x = sps.filtfilt(_nt_b, _nt_a, x, axis=-1)
    return x


def hjorth(x):
    dx = np.diff(x); ddx = np.diff(dx)
    v0 = np.var(x) + 1e-12; v1 = np.var(dx) + 1e-12; v2 = np.var(ddx) + 1e-12
    mob = np.sqrt(v1 / v0)
    return mob, np.sqrt(v2 / v1) / (mob + 1e-12)


def epoch_features(ep):
    """Identical to 06_features_filtered.epoch_features, with the SEF95 index
    clamped (the original could index one past the end on a degenerate PSD)."""
    feats = {}
    nper = int(FS * 4)
    freqs, psd = sps.welch(ep, fs=FS, nperseg=nper, noverlap=nper // 2, axis=-1)
    tm = (freqs >= 0.5) & (freqs <= 45.0)
    total = _trapz(psd[:, tm], freqs[tm], axis=-1) + 1e-12
    ftm = freqs[tm]
    am = (freqs >= 2.0) & (freqs <= 40.0)
    lg_am = np.log10(freqs[am])
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
        feats[f"ch{ci}_sef95"] = float(
            ftm[min(np.searchsorted(c, 0.95), len(ftm) - 1)])
        feats[f"ch{ci}_apslope"] = float(
            np.polyfit(lg_am, np.log10(p[am] + 1e-20), 1)[0])
        mob, comp = hjorth(ep[ci])
        feats[f"ch{ci}_hj_mob"] = float(mob)
        feats[f"ch{ci}_hj_comp"] = float(comp)
        feats[f"ch{ci}_logvar"] = float(np.log10(np.var(ep[ci]) + 1e-12))
    return feats


def one_recording(rec, hop):
    """Cut one recording into 20 s windows advanced by `hop` samples.

    Artifact rejection is applied per window exactly as in the non-overlapping
    pipeline. A window is dropped if any channel exceeds the threshold, so a
    denser sampling of the same recording does not relax the criterion.
    """
    try:
        data = sio.loadmat(rec["path"])["Data"].astype(np.float64)
    except Exception:
        return []
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.shape[0] != 6 or data.shape[1] < EPOCH_N:
        return []
    data = preprocess(data)

    rows = []
    starts = range(0, data.shape[1] - EPOCH_N + 1, hop)
    for wi, s0 in enumerate(starts):
        ep = data[:, s0:s0 + EPOCH_N]
        if np.abs(ep).max() > ARTIFACT_UV:
            continue
        if np.allclose(ep.std(axis=1), 0):
            continue
        f = epoch_features(ep)
        f.update(segment_id=os.path.basename(rec["path"])[:-4],
                 subject=str(rec["subject"]).zfill(3), oid=rec["oid"],
                 window_idx=wi, start_sample=int(s0),
                 E=rec["E"], film=rec["film"], stage=rec["stage"])
        rows.append(f)
    return rows


def build(master, ov):
    hop = max(1, int(round(EPOCH_N * (1.0 - ov))))
    t0 = time.time()
    print(f"  overlap {ov:>5.0%} | hop {hop} samples ({hop / FS:.1f} s)",
          flush=True)
    out = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(one_recording)(r, hop) for _, r in master.iterrows())
    rows = [x for chunk in out for x in chunk]
    df = pd.DataFrame(rows)
    dt = time.time() - t0
    print(f"    windows {len(df):>7d} | recordings {df.segment_id.nunique():>4d}"
          f" | subjects {df.subject.nunique():>3d} | {dt / 60:.1f} min",
          flush=True)
    return df


def main():
    master = pd.read_csv(os.path.join(OUT, "master_table.csv"))
    print(f"source recordings in master table: {len(master)}")

    only = [float(a) for a in sys.argv[1:]] or OVERLAPS
    for ov in only:
        path = os.path.join(OUT, f"deed_ov{int(ov * 100):02d}.parquet")
        if os.path.exists(path):
            n = len(pd.read_parquet(path, columns=["subject"]))
            print(f"  overlap {ov:>5.0%} | cached ({n} windows) -> skip",
                  flush=True)
            continue
        df = build(master, ov)
        df.to_parquet(path, index=False)
        print(f"    wrote {os.path.basename(path)}", flush=True)


if __name__ == "__main__":
    main()
