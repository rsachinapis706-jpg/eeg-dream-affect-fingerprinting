"""
Feature extraction WITH standard preprocessing.

Run 1 (03_features.py) applied Welch directly to raw data: no bandpass, no notch,
no artifact rejection. That is very likely why the sleep-stage positive control
stalled at 0.446 balanced accuracy.

This version adds the preprocessing both published DEED analyses use, and writes
to a separate file so the two can be compared directly (the comparison is itself
a reportable ablation).

Preprocessing
-------------
  1. Zero-phase Butterworth bandpass 0.5-45 Hz (order 4, filtfilt -> no phase shift)
  2. 50 Hz notch  (DEED was recorded in China; mains sits inside the gamma band)
  3. Per-epoch artifact rejection: drop an epoch if any channel exceeds
     ARTIFACT_UV in absolute amplitude. Rejection counts are logged per stage so
     any bias in what gets dropped is visible rather than hidden.
"""
import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal as sps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")

FS = 200.0
EPOCH_S = 20
EPOCH_N = int(FS * EPOCH_S)
ARTIFACT_UV = 300.0

_trapz = getattr(np, "trapezoid", None) or np.trapz

BANDS = {
    "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0), "beta": (16.0, 30.0), "gamma": (30.0, 45.0),
}

_bp_b, _bp_a = sps.butter(4, [0.5 / (FS / 2), 45.0 / (FS / 2)], btype="band")
_nt_b, _nt_a = sps.iirnotch(50.0, Q=30.0, fs=FS)


def preprocess(x):
    """x: (channels, samples) -> filtered copy."""
    x = sps.filtfilt(_bp_b, _bp_a, x, axis=-1)
    x = sps.filtfilt(_nt_b, _nt_a, x, axis=-1)
    return x


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
        feats[f"ch{ci}_sef95"] = float(freqs[tm][np.searchsorted(c, 0.95)])
        am = (freqs >= 2.0) & (freqs <= 40.0)
        feats[f"ch{ci}_apslope"] = float(
            np.polyfit(np.log10(freqs[am]), np.log10(p[am] + 1e-20), 1)[0])
        mob, comp = hjorth(ep[ci])
        feats[f"ch{ci}_hj_mob"] = float(mob)
        feats[f"ch{ci}_hj_comp"] = float(comp)
        feats[f"ch{ci}_logvar"] = float(np.log10(np.var(ep[ci]) + 1e-12))
    return feats


def main():
    master = pd.read_csv(os.path.join(OUT, "master_table.csv"))
    rows, skipped = [], []
    rejected = {}

    for i, r in master.iterrows():
        try:
            data = sio.loadmat(r["path"])["Data"].astype(np.float64)
        except Exception as e:
            skipped.append((r["path"], f"load: {e}")); continue
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        if data.shape[0] != 6 or data.shape[1] < EPOCH_N:
            skipped.append((r["path"], f"shape {data.shape}")); continue

        data = preprocess(data)
        n_ep = data.shape[1] // EPOCH_N
        kept = 0
        for e in range(n_ep):
            ep = data[:, e * EPOCH_N:(e + 1) * EPOCH_N]
            if np.abs(ep).max() > ARTIFACT_UV:
                rejected[r["stage"]] = rejected.get(r["stage"], 0) + 1
                continue
            if np.allclose(ep.std(axis=1), 0):
                continue
            f = epoch_features(ep)
            f.update(segment_id=os.path.basename(r["path"])[:-4],
                     subject=str(r["subject"]).zfill(3), oid=r["oid"],
                     epoch_idx=e, n_epochs=n_ep, epochs_from_end=n_ep - 1 - e,
                     E=r["E"], film=r["film"], stage=r["stage"],
                     has_text=pd.notna(r.get("Dream_content")))
            rows.append(f); kept += 1
        if kept == 0:
            skipped.append((r["path"], "all epochs rejected"))
        if (i + 1) % 150 == 0:
            print(f"  {i + 1}/{len(master)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"), index=False)

    txt = "\n".join([
        "=" * 68, "FEATURE EXTRACTION (FILTERED)", "=" * 68,
        f"bandpass 0.5-45 Hz | notch 50 Hz | artifact threshold {ARTIFACT_UV} uV",
        f"epochs kept      : {len(df)}",
        f"epochs rejected  : {sum(rejected.values())}  by stage: {rejected}",
        f"segments kept    : {df.segment_id.nunique()} / {len(master)}",
        f"subjects         : {df.subject.nunique()}",
        "", "epochs by stage:", df.stage.value_counts().to_string(),
        "", f"skipped segments : {len(skipped)}",
    ])
    print(txt)
    with open(os.path.join(OUT, "features_filtered_audit.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)


if __name__ == "__main__":
    main()
