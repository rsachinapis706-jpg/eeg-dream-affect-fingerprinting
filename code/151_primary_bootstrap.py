"""
Participant uncertainty of the primary evaluation (round 3, Reviewer 2,
comment 2: keep participant, partition and refit uncertainty separate).

120_primary_random.py records, for each candidate configuration, balanced
accuracy under each of the 20 random participant partitions, but not the
predictions. This script refits the observed-label evaluation of 120 with the
identical folds, models and seeds, keeps the out-of-fold predictions, checks
that every partition's balanced accuracy reproduces 120's to 1e-9, and
bootstraps participants: one draw of participants per resample, shared by
all 20 partitions (predictions held fixed), so the interval describes which
participants were evaluated, not the partition or the refit.

Outputs: out/primary_random_preds.parquet, out/primary_random_boot.csv
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t5 = _load("t5", "65_config_table.py")
ps = _load("ps", "114_partition_sensitivity.py")
pr = _load("pr", "120_primary_random.py")
OUT = t5.OUT
N_BOOT = 2000
N_JOBS = int(os.environ.get("BOOT_JOBS", 4))


def one(X, y, subj, cfg, r):
    """As 120.partition_mean for a single partition, keeping predictions."""
    kind, scale, weight = t5.CONFIGS[cfg]
    pred = np.empty(len(y), dtype=y.dtype)
    with threadpool_limits(t5.BLAS_THREADS, user_api="blas"):
        for tr, te in ps.random_stratified_folds(y, subj, t5.N_SPLITS,
                                                 pr.PART_SEED + r):
            m = t5.build(kind, scale, weight, t5.SEEDS[0]).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
    return cfg, r, pred


def main():
    X, y, subj, rec = t5.load()
    raw = pd.read_csv(pr.RAW)
    obs = raw[raw["mode"] == "observed"].set_index("config")
    res = Parallel(n_jobs=N_JOBS)(delayed(one)(X, y, subj, c, r)
                                  for c in t5.CANDIDATES for r in range(pr.R))
    rows = []
    for c, r, p in res:
        b = balanced_accuracy_score(y, p)
        assert abs(b - obs.loc[c, f"p{r}"]) < 1e-9, (c, r, b, obs.loc[c, f"p{r}"])
        rows.append(pd.DataFrame(dict(config=c, partition=r, idx=np.arange(len(y)),
                                      subj=subj, y=y, pred=p)))
    pd.concat(rows, ignore_index=True).to_parquet(
        os.path.join(OUT, "primary_random_preds.parquet"))
    # per (config, partition, participant) confusion counts; a resample sums
    # the counts of the drawn participants, with multiplicity
    classes = np.unique(y)
    K = len(classes)
    uq = np.unique(subj)
    si = {s: i for i, s in enumerate(uq)}
    yi = np.searchsorted(classes, y)
    rng = np.random.default_rng(20261021)
    draws = np.stack([np.ones(len(uq), int)]       # row 0: the cohort itself
                     + [np.bincount(rng.integers(0, len(uq), len(uq)),
                                    minlength=len(uq)) for _ in range(N_BOOT)])
    out = []
    for c in t5.CANDIDATES:
        boots = np.zeros((N_BOOT + 1, pr.R))
        for cc, r, p in res:
            if cc != c:
                continue
            pi = np.searchsorted(classes, p)
            C = np.zeros((len(uq), K, K))
            np.add.at(C, (np.vectorize(si.get)(subj), yi, pi), 1)
            S = np.tensordot(draws, C, axes=(1, 0))          # N_BOOT x K x K
            rec_ = np.diagonal(S, axis1=1, axis2=2) / S.sum(2).clip(min=1e-12)
            have = S.sum(2) > 0
            boots[:, r] = (rec_ * have).sum(1) / have.sum(1)
        assert abs(boots[0].mean() - obs.loc[c, "stat"]) < 1e-9, c
        m = boots[1:].mean(1)
        out.append(dict(config=c, observed_mean=obs.loc[c, "stat"],
                        participant_boot_lo=np.percentile(m, 2.5),
                        participant_boot_hi=np.percentile(m, 97.5),
                        n_boot=N_BOOT))
    t = pd.DataFrame(out)
    t.round(4).to_csv(os.path.join(OUT, "primary_random_boot.csv"), index=False)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
