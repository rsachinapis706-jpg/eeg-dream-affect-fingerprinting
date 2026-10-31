"""
CBraMod input scaling (round 3, Reviewer 2, comment 9).

CBraMod was pretrained on EEG in units of 100 uV. The fine-tuning analyses
fed it other scales: the DEED runs (76) per-window, per-channel z-scores
(deed_raw.npy, from 12_raw_dataset.py), and the Sleep-EDF control (49)
microvolts. This script builds the DEED windows in the pretraining
convention, so that both analyses can be repeated with the input CBraMod
expects (CBRAMOD_SCALE=uv100 in 76 and 49; 49 divides on the fly).

The windows are those of deed_raw.npy, rebuilt by the same code path
(12_raw_dataset.clean, the same epoching and artifact rule) with z-scoring
replaced by division by 100 uV. Checked: the window metadata are identical to
deed_raw_meta.parquet, and z-scoring the new array reproduces deed_raw.npy.

Output: out/deed_raw_uv100.npy, out/deed_raw_uv100_meta.parquet
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
spec = importlib.util.spec_from_file_location(
    "raw12", os.path.join(HERE, "12_raw_dataset.py"))
raw12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw12)


def main():
    master = pd.read_csv(os.path.join(OUT, "master_table.csv"))
    master["segment_id"] = master.path.map(lambda p: os.path.basename(p)[:-4])
    master["subject"] = master.subject.astype(str).str.zfill(3)
    n_ep = raw12.EPOCH_N
    X, meta = [], []
    for _, r in master.iterrows():
        try:
            d = sio.loadmat(r["path"])["Data"].astype(np.float64)
        except Exception:
            continue
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        if d.shape[0] != 6 or d.shape[1] < n_ep:
            continue
        d = raw12.clean(d)
        for e in range(d.shape[1] // n_ep):
            ep = d[:, e * n_ep:(e + 1) * n_ep]
            if (np.abs(ep).max() > raw12.ARTIFACT_UV
                    or np.allclose(ep.std(axis=1), 0)):
                continue
            X.append((ep / 100.0).astype(np.float32))
            meta.append(dict(segment_id=r["segment_id"], subject=r["subject"],
                             oid=r["oid"], E=r["E"], film=r["film"],
                             stage=r["stage"], epoch_idx=e))
    X = np.stack(X)
    meta = pd.DataFrame(meta)
    ref = pd.read_parquet(os.path.join(OUT, "deed_raw_meta.parquet"))
    pd.testing.assert_frame_equal(meta.reset_index(drop=True),
                                  ref.reset_index(drop=True))
    Z = np.load(os.path.join(OUT, "deed_raw.npy"), mmap_mode="r")
    assert Z.shape == X.shape
    idx = np.linspace(0, len(X) - 1, 200).astype(int)
    rz = np.stack([raw12.znorm(X[i].astype(np.float64)) for i in idx])
    err = float(np.abs(rz - np.asarray(Z[idx])).max())
    assert err < 1e-3, err
    np.save(os.path.join(OUT, "deed_raw_uv100.npy"), X)
    meta.to_parquet(os.path.join(OUT, "deed_raw_uv100_meta.parquet"),
                    index=False)
    s = np.abs(X[::25])
    print(f"deed_raw_uv100.npy {X.shape}; windows identical to deed_raw.npy; "
          f"z-scored max |diff| {err:.2e}; |x| median {np.median(s):.3f}, "
          f"p99 {np.percentile(s, 99):.3f} (units of 100 uV)")


if __name__ == "__main__":
    main()
