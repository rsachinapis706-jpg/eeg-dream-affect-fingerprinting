"""
Feature extraction for DEED segments.

Sampling rate
-------------
fs = 200 Hz, determined empirically, not assumed: for each segment with a usable
REM-period start/end timestamp, samples / duration was computed. The 25th, 50th
and 75th percentiles are all exactly 200.00. This is recorded in the audit output
so the manuscript can state how fs was established.

Epoching
--------
Segments vary from ~120 s to ~930 s. Each is cut into non-overlapping 20 s epochs,
matching the epoch length used by the published DEED analyses so results are
directly comparable. Features are computed per epoch and stored per epoch; any
aggregation to segment level happens downstream, never here.

Features (per channel, 6 channels)
----------------------------------
  - absolute + relative band power: delta, theta, alpha, sigma, beta, gamma
  - spectral entropy, spectral edge frequency (95%)
  - aperiodic (1/f) slope from a log-log fit, 2-40 Hz
  - Hjorth mobility and complexity
  - signal variance (log)

No channel-pair asymmetry features yet: the 6 channel identities are not
documented in the .mat files and asymmetry requires knowing left/right montage.
Flagged for the DEED paper.
"""
import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal as sps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
os.makedirs(OUT, exist_ok=True)

FS = 200.0
EPOCH_S = 20
EPOCH_N = int(FS * EPOCH_S)

# np.trapezoid is numpy>=2.0; np.trapz is the <2.0 spelling. Same function.
_trapz = getattr(np, "trapezoid", None) or np.trapz

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
    "gamma": (30.0, 45.0),
}


def hjorth(x):
    """Hjorth mobility and complexity (activity == variance, kept separately)."""
    dx = np.diff(x)
    ddx = np.diff(dx)
    v0 = np.var(x) + 1e-12
    v1 = np.var(dx) + 1e-12
    v2 = np.var(ddx) + 1e-12
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / (mob + 1e-12)
    return mob, comp


def epoch_features(ep):
    """ep: (n_channels, EPOCH_N) -> flat dict of features."""
    feats = {}
    nper = int(FS * 4)  # 4 s Welch window -> 0.25 Hz resolution
    freqs, psd = sps.welch(ep, fs=FS, nperseg=nper, noverlap=nper // 2, axis=-1)

    total_mask = (freqs >= 0.5) & (freqs <= 45.0)
    total = _trapz(psd[:, total_mask], freqs[total_mask], axis=-1) + 1e-12

    for ci in range(ep.shape[0]):
        p = psd[ci]

        for bname, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            bp = _trapz(p[mask], freqs[mask]) + 1e-12
            feats[f"ch{ci}_{bname}_abs"] = np.log10(bp)
            feats[f"ch{ci}_{bname}_rel"] = bp / total[ci]

        # spectral entropy over the analysis band
        pn = p[total_mask] / (p[total_mask].sum() + 1e-12)
        feats[f"ch{ci}_specent"] = float(-(pn * np.log2(pn + 1e-12)).sum()
                                         / np.log2(len(pn)))

        # 95% spectral edge frequency
        c = np.cumsum(p[total_mask])
        c /= c[-1] + 1e-12
        feats[f"ch{ci}_sef95"] = float(freqs[total_mask][np.searchsorted(c, 0.95)])

        # aperiodic slope: linear fit of log10(psd) on log10(f), 2-40 Hz
        am = (freqs >= 2.0) & (freqs <= 40.0)
        slope = np.polyfit(np.log10(freqs[am]), np.log10(p[am] + 1e-20), 1)[0]
        feats[f"ch{ci}_apslope"] = float(slope)

        mob, comp = hjorth(ep[ci])
        feats[f"ch{ci}_hj_mob"] = float(mob)
        feats[f"ch{ci}_hj_comp"] = float(comp)
        feats[f"ch{ci}_logvar"] = float(np.log10(np.var(ep[ci]) + 1e-12))

    return feats


def main():
    master = pd.read_csv(os.path.join(OUT, "master_table.csv"))
    rows = []
    skipped = []

    for i, r in master.iterrows():
        try:
            data = sio.loadmat(r["path"])["Data"].astype(np.float64)
        except Exception as e:
            skipped.append((r["path"], f"load: {e}"))
            continue

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        if data.shape[0] != 6:
            skipped.append((r["path"], f"channels={data.shape[0]}"))
            continue

        n_ep = data.shape[1] // EPOCH_N
        if n_ep < 1:
            skipped.append((r["path"], f"too short: {data.shape[1]} samples"))
            continue

        for e in range(n_ep):
            ep = data[:, e * EPOCH_N:(e + 1) * EPOCH_N]
            if np.allclose(ep.std(axis=1), 0):
                continue  # flat/dead epoch
            f = epoch_features(ep)
            f.update(
                segment_id=os.path.basename(r["path"])[:-4],
                subject=str(r["subject"]).zfill(3),
                oid=r["oid"],
                epoch_idx=e,
                n_epochs=n_ep,
                epochs_from_end=n_ep - 1 - e,   # 0 == last epoch before awakening
                E=r["E"],
                film=r["film"],
                stage=r["stage"],
                has_text=pd.notna(r.get("Dream_content")),
            )
            rows.append(f)

        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(master)} segments...")

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(OUT, "features_epoch.parquet"), index=False)

    meta_cols = ["segment_id", "subject", "oid", "epoch_idx", "n_epochs",
                 "epochs_from_end", "E", "film", "stage", "has_text"]
    feat_cols = [c for c in df.columns if c not in meta_cols]

    lines = [
        "=" * 68,
        "FEATURE EXTRACTION",
        "=" * 68,
        f"fs                 : {FS} Hz (empirically determined)",
        f"epoch length       : {EPOCH_S} s ({EPOCH_N} samples)",
        f"segments processed : {master.shape[0] - len(skipped)} / {len(master)}",
        f"epochs extracted   : {len(df)}",
        f"features per epoch : {len(feat_cols)}",
        f"subjects           : {df.subject.nunique()}",
        "",
        "epochs per segment:",
        df.groupby('segment_id').size().describe().round(2).to_string(),
        "",
        "epochs by stage:",
        df.stage.value_counts().to_string(),
        "",
        f"skipped segments   : {len(skipped)}",
    ]
    for p, why in skipped[:20]:
        lines.append(f"   {os.path.basename(p)}: {why}")

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "features_audit.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)


if __name__ == "__main__":
    main()
