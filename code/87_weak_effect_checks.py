"""
Checks on the weak logistic-regression effect that the manuscript reports but
whose values were not all saved (Reviewer 1, comments 2 and 7).

  per-assignment significance
      For each of the ten fold assignments, the observed balanced accuracy
      against that assignment's own null (the first 100 permutations of
      65_config_table.py, which carry all ten assignments), under both nulls.
      Replaces "seeds p < 0.05" from 44_seed_stability.py, whose output no
      longer contains the balanced configuration.
  jackknife
      Leave-one-participant-out refit of the balanced logistic regression,
      for the first fold assignment (the published range 0.3362-0.3905 came
      from 45 without per-participant values) and for the mean over the ten
      assignments, the statistic the manuscript reports (0.3768). Every
      participant's value is saved.

Uses 65's load() and evaluate() unchanged (BLAS pinned to three threads).
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "t5", os.path.join(HERE, "65_config_table.py"))
t5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t5)
OUT = t5.OUT
CFG = "LogReg, scaled, balanced"
LR_CFGS = ["LogReg, scaled, unweighted", "LogReg, scaled, balanced"]


def per_assignment():
    r = pd.read_csv(os.path.join(OUT, "config_table_raw.csv"))
    rows = []
    for cfg in LR_CFGS:
        obs = r[(r.config == cfg) & (r["mode"] == "observed")].set_index("seed")
        for mode in ("within", "free"):
            n = r[(r.config == cfg) & (r["mode"] == mode)
                  & (r.b < t5.b_mean(cfg))]
            for sd, g in n.groupby("seed"):
                o = float(obs.loc[sd, "bacc"])
                assert len(g) == t5.b_mean(cfg)
                rows.append(dict(config=cfg, null=mode, seed=sd, observed=o,
                                 B=len(g), p=(1 + (g.bacc >= o).sum())
                                 / (1 + len(g))))
    a = pd.DataFrame(rows)
    a.round(4).to_csv(os.path.join(OUT, "weak_effect_per_assignment.csv"),
                      index=False)
    s = a.groupby(["config", "null"]).agg(
        n_assignments=("seed", "nunique"),
        n_p_below_005=("p", lambda v: int((v < 0.05).sum())),
        p_min=("p", "min"), p_max=("p", "max"))
    print(s.to_string())


def jack_one(X, y, subj, drop, seeds):
    keep = subj != drop
    return [t5.evaluate(X[keep], y[keep], subj[keep], CFG, sd) for sd in seeds]


def jackknife():
    X, y, subj, _ = t5.load()
    uq = sorted(np.unique(subj))
    full = [t5.evaluate(X, y, subj, CFG, sd) for sd in t5.SEEDS]
    res = Parallel(n_jobs=8)(delayed(jack_one)(X, y, subj, p, t5.SEEDS)
                             for p in uq)
    rows = [dict(left_out=p, first_assignment=v[0], mean10=float(np.mean(v)))
            for p, v in zip(uq, res)]
    j = pd.DataFrame(rows)
    j.round(4).to_csv(os.path.join(OUT, "weak_effect_jackknife.csv"),
                      index=False)
    summ = []
    for col, f in (("first_assignment", full[0]), ("mean10", np.mean(full))):
        v = j[col]
        summ.append(dict(statistic=col, full=f, min=v.min(),
                         min_left_out=j.loc[v.idxmin(), "left_out"],
                         median=v.median(), max=v.max(),
                         max_left_out=j.loc[v.idxmax(), "left_out"],
                         n_above_chance=int((v > 1 / 3).sum()),
                         n=len(v)))
    s = pd.DataFrame(summ)
    s.round(4).to_csv(os.path.join(OUT, "weak_effect_jackknife_summary.csv"),
                      index=False)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    per_assignment()
    jackknife()
