"""
Primary evaluation over random participant partitions (Reviewer 2, comment 2;
round 3, Access-2026-50879).

Why
  The ten seeded fold assignments give only two distinct DEED participant
  partitions, and on random partitions the best configuration falls from
  0.3768 to 0.3393. Reviewer 2 asks for one primary evaluation procedure
  defined independently of its performance, the selection pipeline assessed
  under it, a permutation budget adequate near 0.05 (Table 5's averaged
  statistic used 30 shared permutations for the forests), and the sources of
  uncertainty kept separate.

Primary procedure (chosen because the splitter gives only two participant
partitions, not for its outcome; its observed values were known from
114_partition_sensitivity.py, which uses the same partitions, but its
permutation tests, nested selection and refit analysis were run after it was
fixed)
  For every candidate configuration of Table 5 (data, features, models and
  thread pins imported from 65_config_table.py): balanced accuracy on pooled
  out-of-fold predictions under R = 20 random class-balanced participant
  partitions (114.random_stratified_folds, seeds 30_001_000 + r), averaged
  over the 20. The forest seed is held at the first seed, so the average
  reflects partitions, not refits. Folds are recomputed from the labelling
  being evaluated, for the observed labels and for every permuted labelling
  alike, so the procedure that is permuted is the procedure that is reported.

Permutation test
  Recording labels permuted within participant ("within") or across all
  recordings ("free"), with the permutations of Table 5 (65.permuted, seeds
  500_000 + b), shared by every configuration. Each configuration's p-value is
  a sequential Monte Carlo p-value (Besag and Clifford, 1991): permutations
  are taken in order b = 0, 1, ... and stopped when H = 20 null statistics
  reach the observed one (p = H / k) or at B_MAX = 1,000 (p = (1 + count) /
  (1 + B_MAX)). This is a valid p-value at any stopping point. Holm's
  procedure over the four candidates uses these p-values; the single-step
  max-statistic adjustment is reported over the permutations every
  configuration completed, with that number stated.

Other parts
  nested   the selection pipeline under the primary procedure: in each outer
           fold of each random partition, every candidate is scored by an
           inner random class-balanced participant partition of the training
           participants only; the best is refitted and applied to the held-out
           participants.
  refit    variability from model refitting, kept separate from partitions:
           both forest configurations under partitions 0-4 with ten forest
           seeds each.
  summary  writes the tables: primary_random_summary.csv,
           primary_random_nested_summary.csv, primary_random_refit.csv.

Usage: python 120_primary_random.py perm|nested|refit|summary
"""
import importlib.util
import os
import sys
import time

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
OUT = t5.OUT
R = 20
PART_SEED = ps.PART_BASE + 1000          # identical to 114's stratified mode
H, B_MAX = 20, 1000
MODES = ("within", "free")
N_JOBS = int(os.environ.get("PRIM_JOBS", 8))
RAW = os.path.join(OUT, "primary_random_raw.csv")
LOG = os.path.join(OUT, "run_primary_random.log")
T0 = time.time()


def log(m=""):
    m = f"[{(time.time() - T0) / 60:7.1f} min] {m}"
    print(m, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(m + "\n")


def partition_mean(X, yl, subj, cfg, model_seed=None, parts=range(R)):
    """Mean over random class-balanced participant partitions of pooled
    balanced accuracy; folds recomputed from the labelling yl."""
    kind, scale, weight = t5.CONFIGS[cfg]
    seed = t5.SEEDS[0] if model_seed is None else model_seed
    vals = []
    with threadpool_limits(t5.BLAS_THREADS, user_api="blas"):
        for r in parts:
            yt, yp = [], []
            for tr, te in ps.random_stratified_folds(yl, subj, t5.N_SPLITS,
                                                     PART_SEED + r):
                m = t5.build(kind, scale, weight, seed).fit(X[tr], yl[tr])
                yp.append(m.predict(X[te]))
                yt.append(yl[te])
            vals.append(balanced_accuracy_score(np.concatenate(yt),
                                                np.concatenate(yp)))
    return vals


def perm_task(X, y, subj, rec, cfg, mode, b):
    yl = y if mode == "observed" else t5.permuted(y, subj, rec, b, mode)
    vals = partition_mean(X, yl, subj, cfg)
    return dict(config=cfg, mode=mode, b=b, stat=float(np.mean(vals)),
                **{f"p{r}": v for r, v in enumerate(vals)})


def seq_state(raw, cfg, mode, obs):
    """Sequential p-value from permutations taken in index order."""
    n = raw[(raw.config == cfg) & (raw["mode"] == mode)].sort_values("b")
    b = n.b.to_numpy()
    k = 0
    while k < len(b) and b[k] == k:          # contiguous prefix 0..k-1 only
        k += 1
    s = n.stat.to_numpy()[:k]
    hits = np.cumsum(s >= obs - 1e-12)
    stop = np.flatnonzero(hits >= H)
    if len(stop):
        k_stop = int(stop[0]) + 1
        return dict(done=True, k=k_stop, count=H, p=H / k_stop, rule="H reached")
    if k >= B_MAX:
        c = int(hits[B_MAX - 1])
        return dict(done=True, k=B_MAX, count=c, p=(1 + c) / (1 + B_MAX),
                    rule="B_MAX")
    c = int(hits[-1]) if k else 0
    return dict(done=False, k=k, count=c, p=np.nan, rule="running")


def run_perm():
    X, y, subj, rec = t5.load()
    raw = pd.read_csv(RAW) if os.path.exists(RAW) else pd.DataFrame(
        columns=["config", "mode", "b", "stat"])
    log(f"PRIMARY over {R} random partitions; H={H}, B_MAX={B_MAX}; "
        f"{N_JOBS} workers; {len(raw)} rows already done")
    need = [c for c in t5.CANDIDATES
            if raw[(raw.config == c) & (raw["mode"] == "observed")].empty]
    if need:
        res = Parallel(n_jobs=N_JOBS)(delayed(perm_task)(X, y, subj, rec, c,
                                                         "observed", -1)
                                      for c in need)
        raw = pd.concat([raw, pd.DataFrame(res)], ignore_index=True)
        raw.to_csv(RAW, index=False)
        for r_ in res:
            log(f"observed {r_['config']:<28} mean {r_['stat']:.4f}")
    obs = {c: float(raw[(raw.config == c) & (raw["mode"] == "observed")]
                    .stat.iloc[0]) for c in t5.CANDIDATES}
    step = int(os.environ.get("PRIM_STEP", 16))
    while True:
        todo = []
        for mode in MODES:
            for c in t5.CANDIDATES:
                st = seq_state(raw, c, mode, obs[c])
                if st["done"]:
                    continue
                have = set(raw[(raw.config == c) & (raw["mode"] == mode)].b)
                # it cannot stop in fewer permutations than it lacks
                # exceedances, so none in the batch is surplus to the rule
                nxt = [b for b in range(B_MAX)
                       if b not in have][:min(step, H - st["count"])]
                todo += [(c, mode, b) for b in nxt]
        if not todo:
            break
        res = Parallel(n_jobs=N_JOBS)(delayed(perm_task)(X, y, subj, rec, c,
                                                         m, b)
                                      for c, m, b in todo)
        raw = pd.concat([raw, pd.DataFrame(res)], ignore_index=True)
        raw.to_csv(RAW, index=False)
        msg = []
        for mode in MODES:
            for c in t5.CANDIDATES:
                st = seq_state(raw, c, mode, obs[c])
                msg.append(f"{mode[:1]}:{c[:12]} k={st['k']} n>={st['count']}"
                           + (f" p={st['p']:.4f}" if st["done"] else ""))
        log(" | ".join(msg))
    log("permutation part finished")


def summary():
    raw = pd.read_csv(RAW)
    obs = raw[raw["mode"] == "observed"].set_index("config")
    rows = []
    for c in t5.CANDIDATES:
        pv = [obs.loc[c, f"p{r}"] for r in range(R)]
        row = dict(config=c, observed_mean=obs.loc[c, "stat"],
                   partition_sd=float(np.std(pv, ddof=1)),
                   partition_min=min(pv), partition_max=max(pv),
                   above_chance=int(sum(v > 1 / 3 for v in pv)), partitions=R)
        for mode in MODES:
            st = seq_state(raw, c, mode, obs.loc[c, "stat"])
            n = raw[(raw.config == c) & (raw["mode"] == mode)]
            row.update({f"{mode}_k": st["k"], f"{mode}_count": st["count"],
                        f"{mode}_p": st["p"], f"{mode}_rule": st["rule"],
                        f"{mode}_null_mean": n.stat.mean(),
                        f"{mode}_null_sd": n.stat.std(ddof=1)})
        rows.append(row)
    t = pd.DataFrame(rows).set_index("config")
    for mode in MODES:
        p = t[f"{mode}_p"].sort_values()
        k, run, holm = len(p), 0.0, {}
        for i, (c, v) in enumerate(p.items()):
            run = max(run, min(1.0, (k - i) * v))
            holm[c] = run
        t[f"{mode}_p_holm"] = pd.Series(holm)
        piv = raw[raw["mode"] == mode].pivot_table(index="b", columns="config",
                                                   values="stat")
        piv = piv[t5.CANDIDATES].dropna()
        mx = piv.max(axis=1).to_numpy()
        t[f"{mode}_maxT_B"] = len(mx)
        for c in t5.CANDIDATES:
            t.loc[c, f"{mode}_p_maxT"] = ((1 + (mx >= t.loc[c, "observed_mean"]
                                                - 1e-12).sum()) / (1 + len(mx)))
    t.round(4).to_csv(os.path.join(OUT, "primary_random_summary.csv"))
    with pd.option_context("display.width", 250):
        print(t.round(4).to_string())


# ------------------------------------------------------------------ nested
def nested_one(X, y, subj, r):
    folds, preds = [], []
    for k, (tr, te) in enumerate(ps.random_stratified_folds(
            y, subj, t5.N_SPLITS, PART_SEED + r)):
        inner = {}
        for c in t5.CANDIDATES:
            kind, scale, weight = t5.CONFIGS[c]
            yt, yp = [], []
            with threadpool_limits(t5.BLAS_THREADS, user_api="blas"):
                for itr, ite in ps.random_stratified_folds(
                        y[tr], subj[tr], t5.N_SPLITS, PART_SEED + 50_000 + 100 * r + k):
                    m = t5.build(kind, scale, weight, t5.SEEDS[0]).fit(
                        X[tr][itr], y[tr][itr])
                    yp.append(m.predict(X[tr][ite]))
                    yt.append(y[tr][ite])
            inner[c] = balanced_accuracy_score(np.concatenate(yt),
                                               np.concatenate(yp))
        best = max(t5.CANDIDATES, key=lambda c: inner[c])     # first on ties
        kind, scale, weight = t5.CONFIGS[best]
        with threadpool_limits(t5.BLAS_THREADS, user_api="blas"):
            p = t5.build(kind, scale, weight, t5.SEEDS[0]).fit(
                X[tr], y[tr]).predict(X[te])
        folds.append(dict(partition=r, fold=k, chosen=best,
                          **{f"inner {c}": v for c, v in inner.items()}))
        preds.append(pd.DataFrame(dict(partition=r, fold=k, idx=te,
                                       subj=subj[te], y=y[te], pred=p,
                                       chosen=best)))
    return folds, pd.concat(preds, ignore_index=True)


def run_nested():
    X, y, subj, rec = t5.load()
    log(f"NESTED selection on {R} random partitions, {N_JOBS} workers")
    res = Parallel(n_jobs=N_JOBS)(delayed(nested_one)(X, y, subj, r)
                                  for r in range(R))
    folds = pd.DataFrame([f for x in res for f in x[0]])
    pr = pd.concat([x[1] for x in res], ignore_index=True)
    folds.to_csv(os.path.join(OUT, "primary_random_nested_folds.csv"), index=False)
    pr.to_parquet(os.path.join(OUT, "primary_random_nested_preds.parquet"))
    per = pr.groupby("partition").apply(
        lambda g: balanced_accuracy_score(g.y, g.pred), include_groups=False)
    # participant bootstrap of the partition-averaged value, one participant
    # draw per resample shared by all partitions (fixed predictions)
    rng = np.random.default_rng(20261021)
    uq = np.unique(pr.subj)
    boots = []
    by = {(r, s): g for (r, s), g in pr.groupby(["partition", "subj"])}
    for _ in range(2000):
        draw = rng.choice(uq, len(uq))
        vals = []
        for r in range(R):
            g = pd.concat([by[(r, s)] for s in draw if (r, s) in by])
            vals.append(balanced_accuracy_score(g.y, g.pred))
        boots.append(np.mean(vals))
    s = pd.DataFrame([dict(statistic="nested selection, mean over partitions",
                           bal=per.mean(), partition_sd=per.std(ddof=1),
                           partition_min=per.min(), partition_max=per.max(),
                           above_chance=int((per > 1 / 3).sum()), partitions=R,
                           participant_boot_lo=np.percentile(boots, 2.5),
                           participant_boot_hi=np.percentile(boots, 97.5))])
    s.round(4).to_csv(os.path.join(OUT, "primary_random_nested_summary.csv"),
                      index=False)
    ch = folds.chosen.value_counts()
    ch.to_csv(os.path.join(OUT, "primary_random_nested_choices.csv"))
    log(s.round(4).to_string(index=False))
    log("chosen: " + ", ".join(f"{k} {v}" for k, v in ch.items()))


# ------------------------------------------------------------------- refit
def run_refit():
    X, y, subj, rec = t5.load()
    rf = [c for c in t5.CANDIDATES if t5.CONFIGS[c][0] == "rf"]
    jobs = [(c, r, s) for c in rf for r in range(5) for s in t5.SEEDS]
    log(f"REFIT variability: {len(jobs)} forest evaluations, {N_JOBS} workers")
    res = Parallel(n_jobs=N_JOBS)(
        delayed(lambda c, r, s: dict(config=c, partition=r, forest_seed=s,
                                     bal=partition_mean(X, y, subj, c, s,
                                                        parts=[r])[0]))(c, r, s)
        for c, r, s in jobs)
    d = pd.DataFrame(res)
    d.round(6).to_csv(os.path.join(OUT, "primary_random_refit.csv"), index=False)
    within = d.groupby(["config", "partition"]).bal.std(ddof=1)
    across = d.groupby(["config", "forest_seed"]).bal.std(ddof=1)
    sm = pd.DataFrame(dict(
        refit_sd_within_partition=within.groupby("config").mean(),
        partition_sd_within_seed=across.groupby("config").mean()))
    sm.round(4).to_csv(os.path.join(OUT, "primary_random_refit_summary.csv"))
    log(sm.round(4).to_string())


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "summary"
    {"perm": run_perm, "nested": run_nested, "refit": run_refit,
     "summary": summary}[part]()
