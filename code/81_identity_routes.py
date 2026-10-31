"""
Two routes from shuffled windows to above-chance accuracy (Reviewer 1,
comment 6, follow-up to 74_identity_controls.py).

74's balanced control removes the association between participant and label
by construction (24 participants with all three classes, windows subsampled to
equal class counts within each participant), yet the window-shuffled forest
still scores 0.4834 against 0.3333. If that residual comes from recognising the
recording -- windows of one recording share a label and fall on both sides of a
window-level partition -- then grouping the same subsamples by recording should
return it to chance. This script:

  1. regenerates 74's ten balanced subsamples exactly (same random streams) and
     checks the first against 74's stored window-shuffled result;
  2. scores the same forest on each subsample with recording-grouped folds;
  3. measures, without labels, how often a test window's nearest training
     window (standardised features, Euclidean) comes from the same recording
     and from the same participant, on the 6,144 windows of Table 8 under
     window-shuffled folds (first three fold assignments) and on the balanced
     subsamples.

Random forest, 300 trees, min_samples_leaf = 2, as in 74.
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "ic", os.path.join(HERE, "74_identity_controls.py"))
ic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ic)
OUT = ic.OUT
THREADS = 6


def balanced_subsample(d, three, i):
    """Identical to 74.balanced_task's sampling."""
    rng = np.random.default_rng(800_000 + i)
    parts = []
    for s in three:
        g = d[d.subject == s]
        n = g.y.value_counts().min()
        parts.append(g.groupby("y", group_keys=False).apply(
            lambda x: x.sample(n, random_state=int(rng.integers(1e9)))))
    return pd.concat(parts).reset_index(drop=True)


def nn_rates(X, rec, subj, sp):
    same_r = same_s = n = 0
    for tr, te in sp:
        sc = StandardScaler().fit(X[tr])
        j = NearestNeighbors(n_neighbors=1, n_jobs=THREADS).fit(
            sc.transform(X[tr])).kneighbors(sc.transform(X[te]),
                                            return_distance=False)[:, 0]
        same_r += int((rec[tr][j] == rec[te]).sum())
        same_s += int((subj[tr][j] == subj[te]).sum())
        n += len(te)
    return same_r / n, same_s / n


def main():
    d, fc = ic.load()
    rows = []

    # 3. label-free nearest-neighbour rates on Table 8's windows
    X, y = d[fc].to_numpy(float), d.y.to_numpy()
    rec, subj = d.segment_id.to_numpy(), d.subject.to_numpy()
    for seed in ic.SEEDS:
        for part, grp in (("window-shuffled", None),
                          ("recording-grouped", rec)):
            r_, s_ = nn_rates(X, rec, subj, ic.folds(y, grp, seed))
            rows.append(dict(data="6,144 windows, 34 participants",
                             partition=part, seed=seed, nn_same_recording=r_,
                             nn_same_participant=s_))
            print(f"6144 {part:<18} seed {seed}: nearest training window "
                  f"same recording {r_:.4f}, same participant {s_:.4f}",
                  flush=True)

    # 1-2. balanced subsamples, both partitions
    three = sorted(g for g, v in d.groupby("subject").y.nunique().items()
                   if v == 3)
    stored = pd.read_csv(os.path.join(OUT, "identity_controls_balanced.csv"))
    for i in range(ic.N_BAL):
        dd = balanced_subsample(d, three, i)
        assert len(dd) == stored.windows[i]
        Xb, yb = dd[fc].to_numpy(float), dd.y.to_numpy()
        rb, sb = dd.segment_id.to_numpy(), dd.subject.to_numpy()
        for part, grp in (("window-shuffled", None),
                          ("recording-grouped", rb)):
            if part == "window-shuffled" and i > 0:
                acc = bal = np.nan          # stored in 74's output
            else:
                sp = ic.folds(yb, grp, ic.SEEDS[0])
                p = ic.rf_pred(Xb, yb, sp, ic.SEEDS[0], THREADS)
                acc, bal = accuracy_score(yb, p), balanced_accuracy_score(yb,
                                                                          p)
            sp = ic.folds(yb, grp, ic.SEEDS[0])
            r_, s_ = nn_rates(Xb, rb, sb, sp)
            rows.append(dict(data="balanced subsample", subsample=i,
                             partition=part, seed=ic.SEEDS[0],
                             windows=len(dd), recordings=len(set(rb)),
                             acc=acc, bal=bal, nn_same_recording=r_,
                             nn_same_participant=s_))
            print(f"balanced {i} {part:<18} acc {acc:.4f} bal {bal:.4f} | NN "
                  f"same recording {r_:.4f}, same participant {s_:.4f}",
                  flush=True)
        if i == 0:
            print(f"   check: window-shuffled acc {rows[-2]['acc']:.4f} vs 74 "
                  f"{stored.acc[0]:.4f}", flush=True)
            assert abs(rows[-2]["acc"] - stored.acc[0]) < 0.005

    out = pd.DataFrame(rows)
    out.round(4).to_csv(os.path.join(OUT, "identity_routes.csv"), index=False)
    b = out[out.data == "balanced subsample"]
    print("\nsummary")
    print(out[out.data != "balanced subsample"].groupby("partition")[
        ["nn_same_recording", "nn_same_participant"]].mean().round(4))
    print(b.groupby("partition")[["acc", "bal", "nn_same_recording",
                                  "nn_same_participant"]].agg(
        ["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
