"""
Raw-EEG tensors for DreamFM.

Everything so far used handcrafted features. The remaining objection to the
negative result is that a learned representation might find what spectral and
entropy features miss. Testing that needs raw signal, not features.

Outputs two memory-mapped arrays so the transformer can stream them without
loading everything into RAM:

  deed_raw.npy      (N, 6, 4000)   20 s windows @ 200 Hz, DEED
  sleepedf_raw.npy  (M, 2, 4000)   20 s windows @ 200 Hz, Sleep-EDF (upsampled
                                   from 100 Hz so both datasets share a rate)

Preprocessing is deliberately identical to the feature pipeline (0.5-45 Hz
bandpass, 50 Hz notch, 300 uV artifact rejection) so that any difference in
result is attributable to the model, not to the data handling.

Per-window z-scoring is applied per channel. This is label-free and window-local,
so it leaks nothing across splits, and it stops the network keying on absolute
amplitude, which is dominated by subject-level electrode differences.
"""
import os
import glob
import warnings
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal as sps

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
EDF_DIR = os.path.join(ROOT, "sleepedf", "sleep-edf-database-1.0.0")

FS = 200.0
EPOCH_S = 20
EPOCH_N = int(FS * EPOCH_S)
ARTIFACT_UV = 300.0

_bp_b, _bp_a = sps.butter(4, [0.5 / (FS / 2), 45.0 / (FS / 2)], btype="band")
_nt_b, _nt_a = sps.iirnotch(50.0, Q=30.0, fs=FS)

STAGE = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "N3", 5: "REM"}

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def clean(x):
    x = sps.filtfilt(_bp_b, _bp_a, x, axis=-1)
    return sps.filtfilt(_nt_b, _nt_a, x, axis=-1)


def znorm(ep):
    mu = ep.mean(axis=-1, keepdims=True)
    sd = ep.std(axis=-1, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (ep - mu) / sd


def build_deed():
    master = pd.read_csv(os.path.join(OUT, "master_table.csv"))
    master["segment_id"] = master.path.map(lambda p: os.path.basename(p)[:-4])
    master["subject"] = master.subject.astype(str).str.zfill(3)

    X, meta = [], []
    for _, r in master.iterrows():
        try:
            d = sio.loadmat(r["path"])["Data"].astype(np.float64)
        except Exception:
            continue
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        if d.shape[0] != 6 or d.shape[1] < EPOCH_N:
            continue
        d = clean(d)
        for e in range(d.shape[1] // EPOCH_N):
            ep = d[:, e * EPOCH_N:(e + 1) * EPOCH_N]
            if np.abs(ep).max() > ARTIFACT_UV or np.allclose(ep.std(axis=1), 0):
                continue
            X.append(znorm(ep).astype(np.float32))
            meta.append(dict(segment_id=r["segment_id"], subject=r["subject"],
                             oid=r["oid"], E=r["E"], film=r["film"],
                             stage=r["stage"], epoch_idx=e))
    X = np.stack(X)
    np.save(os.path.join(OUT, "deed_raw.npy"), X)
    pd.DataFrame(meta).to_parquet(os.path.join(OUT, "deed_raw_meta.parquet"),
                                  index=False)
    return X, pd.DataFrame(meta)


def build_sleepedf():
    import mne
    mne.set_log_level("ERROR")
    recs = sorted(f for f in glob.glob(os.path.join(EDF_DIR, "*.edf"))
                  if not f.endswith("_hyp.edf"))
    X, meta = [], []
    for rec in recs:
        sid = os.path.basename(rec)[:-4]
        hyp = os.path.join(EDF_DIR, f"{sid}_hyp.edf")
        if not os.path.exists(hyp):
            continue
        raw = mne.io.read_raw_edf(rec, preload=True)
        eeg = [c for c in raw.ch_names if c.startswith("EEG")]
        if not eeg:
            continue
        raw.pick(eeg)
        raw.resample(FS)                     # 100 -> 200 Hz, matches DEED
        d = raw.get_data() * 1e6
        d = clean(d)
        codes = np.round(
            mne.io.read_raw_edf(hyp, preload=True).get_data()[0]).astype(int)
        for i, c in enumerate(codes):
            if c not in STAGE:
                continue
            s0 = int(i * 30 * FS)
            ep = d[:, s0:s0 + EPOCH_N]
            if ep.shape[1] < EPOCH_N or np.abs(ep).max() > ARTIFACT_UV:
                continue
            if np.allclose(ep.std(axis=1), 0):
                continue
            X.append(znorm(ep).astype(np.float32))
            meta.append(dict(subject=sid, stage=STAGE[c], epoch=i))
        log(f"  {sid}: {sum(1 for m in meta if m['subject'] == sid)} windows")
    X = np.stack(X)
    np.save(os.path.join(OUT, "sleepedf_raw.npy"), X)
    pd.DataFrame(meta).to_parquet(os.path.join(OUT, "sleepedf_raw_meta.parquet"),
                                  index=False)
    return X, pd.DataFrame(meta)


def main():
    log("=" * 66)
    log("RAW EEG TENSORS FOR DreamFM")
    log("=" * 66)
    log(f"{EPOCH_S}s windows @ {FS} Hz = {EPOCH_N} samples")
    log("preprocessing identical to the feature pipeline")
    log("")

    log("Sleep-EDF (pretraining + positive control):")
    Xs, ms = build_sleepedf()
    log(f"  -> {Xs.shape}  subjects {ms.subject.nunique()}")
    log(ms.stage.value_counts().to_string())
    log("")

    log("DEED (target task):")
    Xd, md = build_deed()
    log(f"  -> {Xd.shape}  subjects {md.subject.nunique()} "
        f"segments {md.segment_id.nunique()}")
    log("")
    log(f"amplitude check (should be ~0 mean, ~1 sd after z-norm):")
    log(f"  DEED      mean {Xd.mean():+.4f}  sd {Xd.std():.4f}")
    log(f"  Sleep-EDF mean {Xs.mean():+.4f}  sd {Xs.std():.4f}")
    log("")
    log(f"pretraining pool: {len(Xs)} windows (unlabelled use)")
    log(f"target pool     : {len(Xd)} windows")

    with open(os.path.join(OUT, "raw_dataset_audit.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
