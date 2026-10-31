"""
Independent check on the randomised-label ladder: is the high window-shuffled
accuracy recording memorisation?

If window-shuffled accuracy arises because test windows have near-copies from
the same recording in the training folds, then -- regardless of what the
labels mean -- a test window's nearest training window should usually come
from the same recording, and a nearest-neighbour rule should score about that
rate plus chance on the rest. This is measured without any label at all, so it
cannot share a bug with the ladder: same windows, same StratifiedKFold folds
(fold assignment depends on the labels only through stratification, so the
real labels are used for it), standardised features, Euclidean distance.

Reported for 0% and 90% overlap:
  same_recording   share of test windows whose nearest training window is from
                   the same recording
  same_participant share whose nearest training window is from the same
                   participant
  implied ceiling  same_recording + (1 - same_recording) / 3, the accuracy of a
                   rule that is right whenever it finds the recording and at
                   chance otherwise
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
VALENCE = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
META = {"segment_id", "subject", "oid", "window_idx", "start_sample",
        "E", "film", "stage"}
SEED = 20260727


def main():
    rows = []
    for ov in (0, 90):
        d = pd.read_parquet(os.path.join(OUT, f"deed_ov{ov:02d}.parquet"))
        fc = [c for c in d.columns if c not in META]
        d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
        d = d[d.E.isin(VALENCE)].reset_index(drop=True)
        X = d[fc].to_numpy(float)
        y = d.E.map(VALENCE).to_numpy()
        rec, subj = d.segment_id.to_numpy(), d.subject.to_numpy()
        same_r, same_s, n = 0, 0, 0
        for tr, te in StratifiedKFold(5, shuffle=True,
                                      random_state=SEED).split(X, y):
            sc = StandardScaler().fit(X[tr])
            nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(
                sc.transform(X[tr]))
            j = nn.kneighbors(sc.transform(X[te]), return_distance=False)[:, 0]
            same_r += int((rec[tr][j] == rec[te]).sum())
            same_s += int((subj[tr][j] == subj[te]).sum())
            n += len(te)
        r = dict(overlap=ov, windows=len(y), same_recording=same_r / n,
                 same_participant=same_s / n,
                 implied_ceiling=same_r / n + (1 - same_r / n) / 3)
        rows.append(r)
        print(f"overlap {ov:>2}%: {len(y)} windows | nearest training window "
              f"from the same recording {r['same_recording']:.4f}, same "
              f"participant {r['same_participant']:.4f} | implied accuracy "
              f"ceiling {r['implied_ceiling']:.4f}", flush=True)
    pd.DataFrame(rows).round(4).to_csv(
        os.path.join(OUT, "memorisation_check.csv"), index=False)


if __name__ == "__main__":
    main()
