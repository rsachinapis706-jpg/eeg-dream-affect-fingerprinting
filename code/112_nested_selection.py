"""
Selection-aware estimate for the configuration search of Table 5
(Reviewer 1, comment 7; Reviewer 2, comment 3).

Question
  0.3768 is the best of four candidate configurations, chosen after their
  participant-separated results were seen. How much of it survives when the
  choice is made inside the procedure, without the test participants?

Design (nested cross-validation)
  Everything is imported from 65_config_table.py: data, the four candidate
  configurations, model settings, the ten fold assignments and thread pins.
  For each fold assignment s, the outer folds are exactly those of Table 5.
  Within each outer training fold, every candidate is scored by
  65.evaluate -- five-fold participant-separated cross-validation over the
  training participants only, balanced accuracy on pooled out-of-fold
  predictions -- and the best is refitted on the whole training fold and
  applied to the held-out participants. The selected pipeline is scored on the
  pooled outer predictions, exactly as each fixed configuration is in Table 5.
  Ties in the inner score go to the first candidate in Table 5's order.

Uncertainty
  Participant bootstrap (2,000 resamples) on the fixed out-of-fold
  predictions, averaging the ten fold assignments inside each resample, and
  at the first fold assignment alone. No permutation test of the nested
  procedure is run: each relabelling would cost the whole nested search.

Outputs (out/)
  nested_selection_folds.csv     per assignment and outer fold: inner score of
                                 every candidate, the one chosen
  nested_selection_preds.parquet out-of-fold predictions of the selected pipeline
  nested_selection_summary.csv   per assignment and overall, with intervals
  run_nested_selection.log
"""
import importlib.util
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "t5", os.path.join(HERE, "65_config_table.py"))
t5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t5)
OUT = t5.OUT
N_JOBS = int(os.environ.get("NEST_JOBS", 3))
N_BOOT = 2000
LOG = os.path.join(OUT, "run_nested_selection.log")
T0 = time.time()


def log(m=""):
    m = f"[{(time.time() - T0) / 60:6.1f} min] {m}"
    print(m, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(m + "\n")


def one_assignment(X, y, subj, s):
    """Nested selection for fold assignment s (outer folds as in Table 5)."""
    sp = StratifiedGroupKFold(t5.N_SPLITS, shuffle=True,
                              random_state=s).split(X, y, subj)
    folds, preds = [], []
    for k, (tr, te) in enumerate(sp):
        if len(np.unique(y[tr])) < 2:
            continue
        inner = {c: t5.evaluate(X[tr], y[tr], subj[tr], c, s)
                 for c in t5.CANDIDATES}
        best = max(t5.CANDIDATES, key=lambda c: inner[c])   # first on ties
        kind, scale, weight = t5.CONFIGS[best]
        with threadpool_limits(t5.BLAS_THREADS, user_api="blas"):
            m = t5.build(kind, scale, weight, s).fit(X[tr], y[tr])
            p = m.predict(X[te])
        folds.append(dict(seed=s, fold=k, chosen=best,
                          **{f"inner {c}": v for c, v in inner.items()}))
        preds.append(pd.DataFrame(dict(seed=s, fold=k, idx=te, subj=subj[te],
                                       y=y[te], pred=p, chosen=best)))
    return folds, pd.concat(preds, ignore_index=True)


def boot(pr, seeds, rng):
    """Participant bootstrap of balanced accuracy, one draw of participants
    per resample shared by every fold assignment."""
    uq = np.unique(pr.subj)
    P = len(uq)
    W = np.zeros((N_BOOT, P))
    for b in range(N_BOOT):
        np.add.at(W[b], rng.integers(0, P, P), 1)
    cls = sorted(pr.y.unique())
    out = []
    for s in seeds:
        g = pr[pr.seed == s]
        sidx = np.searchsorted(uq, g.subj.to_numpy())
        tot = np.zeros((P, len(cls)))
        tp = np.zeros((P, len(cls)))
        for c, lab in enumerate(cls):
            m = (g.y == lab).to_numpy()
            np.add.at(tot[:, c], sidx[m], 1)
            np.add.at(tp[:, c], sidx[m & (g.pred == lab).to_numpy()], 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.nanmean((W @ tp) / (W @ tot), axis=1))
    return np.array(out)                                   # seeds x N_BOOT


def main():
    X, y, subj, rec = t5.load()
    log("=" * 70)
    log(f"NESTED SELECTION over {len(t5.CANDIDATES)} candidates, "
        f"{len(t5.SEEDS)} fold assignments, {N_JOBS} workers")
    res = Parallel(n_jobs=N_JOBS)(delayed(one_assignment)(X, y, subj, s)
                                  for s in t5.SEEDS)
    folds = pd.DataFrame([f for r in res for f in r[0]])
    pr = pd.concat([r[1] for r in res], ignore_index=True)
    folds.to_csv(os.path.join(OUT, "nested_selection_folds.csv"), index=False)
    pr.to_parquet(os.path.join(OUT, "nested_selection_preds.parquet"))

    per = pr.groupby("seed").apply(
        lambda g: balanced_accuracy_score(g.y, g.pred), include_groups=False)
    B = boot(pr, t5.SEEDS, np.random.default_rng(20261002))
    mean_b = B.mean(axis=0)
    rows = [dict(statistic=f"assignment {s}", bal=per[s],
                 bal_lo=np.nanpercentile(B[i], 2.5),
                 bal_hi=np.nanpercentile(B[i], 97.5))
            for i, s in enumerate(t5.SEEDS)]
    rows.append(dict(statistic="mean over ten assignments", bal=per.mean(),
                     sd=per.std(ddof=0), bal_lo=np.nanpercentile(mean_b, 2.5),
                     bal_hi=np.nanpercentile(mean_b, 97.5)))
    freq = folds.chosen.value_counts()
    for c in t5.CANDIDATES:
        rows.append(dict(statistic=f"chosen: {c}", count=int(freq.get(c, 0)),
                         share=freq.get(c, 0) / len(folds)))
    sm = pd.DataFrame(rows)
    sm.round(4).to_csv(os.path.join(OUT, "nested_selection_summary.csv"),
                       index=False)
    log("")
    with pd.option_context("display.width", 200):
        log("\n" + sm.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
