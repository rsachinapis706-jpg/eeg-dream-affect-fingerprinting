"""
How many distinct partitions the "fold assignments" are, and how much the
participant-separated results depend on the partition.

Finding that prompted this
  StratifiedGroupKFold(shuffle=True) shuffles groups and then sorts them by
  the spread of their class proportions with a stable sort, so a new seed only
  reorders groups whose spread ties. With participants as groups the ten seeds
  used throughout give two distinct partitions on DEED (five seeds each); for
  a model without a seed of its own, such as the logistic regression, the ten
  participant-separated values are therefore two. Recording-grouped and
  window-level splits differ across all ten seeds.

What is measured
  1. For each splitter seed and each grouping (participant, recording) on DEED
     and on the Tononi corpus, which distinct partition it produces.
  2. The four candidate configurations of Table 5 on R random participant
     partitions: GroupKFold(5, shuffle=True), which assigns participants to
     folds at random, balancing fold sizes. The forest's own seed is held at
     the first seed, so the spread reflects the partition alone. Balanced
     accuracy on pooled out-of-fold predictions, as everywhere.

Outputs (out/)
  fold_assignment_partitions.csv          seed -> partition id per grouping
  fold_assignment_partitions_summary.csv  distinct partitions per grouping
  partition_sensitivity.csv               per random partition and config
  partition_sensitivity_summary.csv       mean, SD, range, share above chance
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from threadpoolctl import threadpool_limits

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "t5", os.path.join(HERE, "65_config_table.py"))
t5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t5)
OUT = t5.OUT
R = int(os.environ.get("PART_R", 20))
PART_BASE = 30_000_000
N_JOBS = int(os.environ.get("PART_JOBS", 3))


def partition_ids(groups, y, seeds):
    keys, ids = [], []
    for s in seeds:
        k = frozenset(frozenset(np.unique(groups[te])) for _, te in
                      StratifiedGroupKFold(t5.N_SPLITS, shuffle=True,
                                           random_state=s)
                      .split(np.zeros(len(y)), y, groups))
        if k not in keys:
            keys.append(k)
        ids.append(keys.index(k))
    return ids


def evaluate_random(X, y, subj, cfg, r):
    kind, scale, weight = t5.CONFIGS[cfg]
    sp = GroupKFold(t5.N_SPLITS, shuffle=True,
                    random_state=PART_BASE + r).split(X, y, subj)
    yt, yp, parts = [], [], []
    with threadpool_limits(t5.BLAS_THREADS, user_api="blas"):
        for tr, te in sp:
            m = t5.build(kind, scale, weight, t5.SEEDS[0]).fit(X[tr], y[tr])
            yp.append(m.predict(X[te]))
            yt.append(y[te])
            parts.append(frozenset(np.unique(subj[te])))
    return dict(partition=r, config=cfg,
                bal=balanced_accuracy_score(np.concatenate(yt), np.concatenate(yp)),
                key=hash(frozenset(parts)))


def random_stratified_folds(y, groups, n_splits, seed):
    """StratifiedGroupKFold's assignment rule -- each group goes to the fold
    that keeps the per-class fold proportions most even, ties to the smaller
    fold -- applied to the groups in a random order instead of sorted by the
    spread of their class proportions, so that every seed gives a different,
    still class-balanced, participant partition."""
    rng = np.random.default_rng(seed)
    cls, yc = np.unique(y, return_inverse=True)
    ug, gi = np.unique(groups, return_inverse=True)
    counts = np.zeros((len(ug), len(cls)))
    np.add.at(counts, (gi, yc), 1)
    tot = counts.sum(0)
    fold_counts = np.zeros((n_splits, len(cls)))
    fold_of = np.empty(len(ug), int)
    for g in rng.permutation(len(ug)):
        best, best_eval, best_n = None, np.inf, np.inf
        for f in range(n_splits):
            fold_counts[f] += counts[g]
            ev = np.mean(np.std(fold_counts / tot, axis=0))
            fold_counts[f] -= counts[g]
            n_f = fold_counts[f].sum()
            if ev < best_eval - 1e-12 or (abs(ev - best_eval) <= 1e-12 and n_f < best_n):
                best, best_eval, best_n = f, ev, n_f
        fold_of[g] = best
        fold_counts[best] += counts[g]
    for f in range(n_splits):
        te = np.where(fold_of[gi] == f)[0]
        yield np.where(fold_of[gi] != f)[0], te


def evaluate_random_stratified(X, y, subj, cfg, r):
    kind, scale, weight = t5.CONFIGS[cfg]
    yt, yp, parts = [], [], []
    with threadpool_limits(t5.BLAS_THREADS, user_api="blas"):
        for tr, te in random_stratified_folds(y, subj, t5.N_SPLITS,
                                              PART_BASE + 1000 + r):
            m = t5.build(kind, scale, weight, t5.SEEDS[0]).fit(X[tr], y[tr])
            yp.append(m.predict(X[te]))
            yt.append(y[te])
            parts.append(frozenset(np.unique(subj[te])))
    return dict(partition=r, config=cfg,
                bal=balanced_accuracy_score(np.concatenate(yt), np.concatenate(yp)),
                key=hash(frozenset(parts)))


def summarise_runs(t, tag):
    n_distinct = t[t.config == t5.CANDIDATES[0]].key.nunique()
    t.drop(columns="key").round(6).to_csv(
        os.path.join(OUT, f"partition_sensitivity{tag}.csv"), index=False)
    s = t.groupby("config", sort=False).bal.agg(
        mean="mean", sd=lambda v: v.std(ddof=0), min="min", max="max",
        above_chance=lambda v: int((v > 1 / 3).sum()))
    s["partitions"] = R
    s["distinct_partitions"] = n_distinct
    s.round(4).to_csv(os.path.join(OUT, f"partition_sensitivity{tag}_summary.csv"))
    print(s.round(4).to_string())


def stratified_only():
    X, y, subj, rec = t5.load()
    res = Parallel(n_jobs=N_JOBS)(
        delayed(evaluate_random_stratified)(X, y, subj, c, r)
        for r in range(R) for c in t5.CANDIDATES)
    summarise_runs(pd.DataFrame(res), "_stratified")


def main():
    import sys
    if "stratified" in sys.argv[1:]:
        return stratified_only()
    X, y, subj, rec = t5.load()
    tn = pd.read_parquet(os.path.join(OUT, "tononi_features.parquet"))
    ty = tn.dreaming.astype(str).to_numpy()
    rows, summ = [], []
    for corpus, g_by, yy in (("DEED", {"participant": subj, "recording": rec}, y),
                             ("Tononi", {"participant": tn.subject.to_numpy(),
                                         "recording": tn.recording.to_numpy()}, ty)):
        for grouping, g in g_by.items():
            ids = partition_ids(g, yy, t5.SEEDS)
            rows += [dict(corpus=corpus, grouping=grouping, seed=s, partition=i)
                     for s, i in zip(t5.SEEDS, ids)]
            summ.append(dict(corpus=corpus, grouping=grouping, seeds=len(ids),
                             distinct_partitions=len(set(ids))))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "fold_assignment_partitions.csv"),
                              index=False)
    sm = pd.DataFrame(summ)
    sm.to_csv(os.path.join(OUT, "fold_assignment_partitions_summary.csv"),
              index=False)
    print(sm.to_string(index=False))

    res = Parallel(n_jobs=N_JOBS)(delayed(evaluate_random)(X, y, subj, c, r)
                                  for r in range(R) for c in t5.CANDIDATES)
    t = pd.DataFrame(res)
    n_distinct = t[t.config == t5.CANDIDATES[0]].key.nunique()
    t.drop(columns="key").round(6).to_csv(
        os.path.join(OUT, "partition_sensitivity.csv"), index=False)
    s = t.groupby("config", sort=False).bal.agg(
        mean="mean", sd=lambda v: v.std(ddof=0), min="min", max="max",
        above_chance=lambda v: int((v > 1 / 3).sum()))
    s["partitions"] = R
    s["distinct_partitions"] = n_distinct
    s.round(4).to_csv(os.path.join(OUT, "partition_sensitivity_summary.csv"))
    print(s.round(4).to_string())


if __name__ == "__main__":
    main()
