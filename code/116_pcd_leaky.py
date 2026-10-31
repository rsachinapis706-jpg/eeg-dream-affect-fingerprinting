"""
PCD under a leaky partition: does its guarantee survive the leakage that
inflates accuracy?

Question
  Partitioning DEED over windows rather than participants inflates accuracy,
  and does so even when the labels carry no information: with labels
  randomised across recordings the leaky protocol of Section IV-G reaches
  0.9403. A confidence threshold has no way to tell that confidence apart from
  signal. PCD's null decoders are fitted through the same partition, so they
  inherit the same leakage; Proposition 1 requires only that the fold
  assignment be fixed without reference to the labels. If it holds, PCD
  predicts on at most alpha of windows under a known null even when the
  partition is leaky.

Design
  Features, classifier (standardised, class-balanced logistic regression,
  C = 0.1), margin, permutation scheme (recording labels permuted within
  participant), B = 100 null decoders per fold, and the pooled p-value are
  imported unchanged from 77_pcd_revised.py. Only the partition differs, and
  both partitions are drawn once, without labels:
     window-level           KFold over windows (the leaky protocol)
     participant-separated  participants shuffled into five folds
  Labellings:
     observed               the DEED valence labels
     known null r           recording labels re-randomised within participant,
                            r = 0..R-1, with the generator seeds of 77's DEED
                            null, so the labellings are identical to 77's
  For every partition and labelling: balanced accuracy of all windows, and
  the coverage of PCD (alpha 0.10, 0.05, 0.01), max-softmax > 0.5 and
  normalised entropy < 0.9, with retained balanced accuracy.
  Observed-label rows get 95% participant-bootstrap intervals (2,000
  resamples), as in Table PCD.
  R = 100 null labellings by default (PCD_LEAKY_R overrides), the standard of
  the stratified-PCD table; labellings 0-19 are those of 77's DEED null.

Resuming
  Every finished labelling is appended to pcd_leaky_conditions_partial.csv
  and skipped on a restart. Labellings run in batches, all their folds in one
  parallel pool; each fold's result depends only on its own seed and runs
  single-threaded, so batching and restarts do not change any number.

Outputs (out/)
  pcd_leaky_conditions.csv   one row per partition x labelling x rule
  pcd_leaky_windows.parquet  per-window scores, observed labels
  pcd_leaky_summary.csv      null labellings summarised (mean, range)
  pcd_leaky_boot.csv         observed labels with participant-bootstrap CIs
  run_pcd_leaky.log

Usage: python 116_pcd_leaky.py [--smoke]
"""
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "pcd", os.path.join(HERE, "77_pcd_revised.py"))
pcd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pcd)

OUT = pcd.OUT
SEED = pcd.SEED
N_SPLITS = pcd.N_SPLITS
ALPHAS = (0.10, 0.05, 0.01)
MSP_THR, ENT_THR = pcd.MSP_THR, pcd.ENT_THR
SMOKE = "--smoke" in sys.argv
B = 10 if SMOKE else pcd.B
R_NULL = 2 if SMOKE else int(os.environ.get("PCD_LEAKY_R", 100))
N_BOOT = 200 if SMOKE else 2000
N_JOBS = int(os.environ.get("PCD_JOBS", 12))
BATCH = int(os.environ.get("PCD_LEAKY_BATCH", 4))   # labellings per pool
TAG = "_smoke" if SMOKE else ""
LOG = os.path.join(OUT, f"run_pcd_leaky{TAG}.log")
PARTIAL = os.path.join(OUT, f"pcd_leaky_conditions_partial{TAG}.csv")
T0 = time.time()


def log(m=""):
    m = f"[{(time.time() - T0) / 60:6.1f} min] {m}"
    print(m, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(m + "\n")


def load_deed():
    d = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    fc = [c for c in d.columns if c not in pcd.META]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = d[d.E.isin(pcd.VALENCE)].reset_index(drop=True)
    X, y = d[fc].to_numpy(float), d.E.map(pcd.VALENCE).to_numpy()
    subj, unit = d.subject.astype(str).to_numpy(), d.segment_id.to_numpy()
    return X, y, subj, unit


def partitions(subj):
    """Both partitions are fixed once, without reference to any labels."""
    n = len(subj)
    win = [te for _, te in KFold(N_SPLITS, shuffle=True,
                                 random_state=SEED).split(np.zeros(n))]
    rng = np.random.default_rng(SEED)
    people = rng.permutation(np.unique(subj))
    groups = np.array_split(people, N_SPLITS)
    par = [np.flatnonzero(np.isin(subj, g)) for g in groups]
    return {"window-level": win, "participant-separated": par}


def one_fold(X, y, subj, unit, te, seed):
    """Observed decoder and B null decoders for one fold; one worker."""
    with threadpool_limits(1):
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(np.unique(y[tr])) < 2:
            return None
        model = pcd.clf().fit(X[tr], y[tr])
        pt = model.predict_proba(X[te])
        S = pcd.null_margins(X[tr], y[tr], unit[tr], subj[tr], X[te], seed, B)
        return dict(idx=te, y=y[te], subj=subj[te],
                    pred=model.classes_[pt.argmax(1)],
                    p_pcd=pcd.pcd_p(pcd.margin(pt), S),
                    msp=pt.max(1), ent=pcd.entropy(pt), n_null=len(S))


def rules(g):
    """Coverage and retained balanced accuracy of every rule on one run."""
    out = [dict(rule="all inputs", coverage=1.0,
                bal_retained=balanced_accuracy_score(g.y, g.pred))]
    keeps = [(f"PCD alpha={a:.2f}", g.p_pcd <= a) for a in ALPHAS]
    keeps += [(f"Max-softmax>{MSP_THR}", g.msp > MSP_THR),
              (f"Entropy<{ENT_THR}", g.ent < ENT_THR)]
    for rule, kk in keeps:
        kk = kk.to_numpy()
        out.append(dict(rule=rule, coverage=kk.mean(),
                        bal_retained=(balanced_accuracy_score(
                            g.y[kk], g.pred[kk]) if kk.sum() > 10 else np.nan)))
    return out


def run(X, subj, unit, parts, labellings):
    """labellings: list of (name, y, seed). All folds of all labellings go
    to one pool; returns {name: (windows, rule rows)}."""
    jobs = [(name, y, seed, pname, k, te) for name, y, seed in labellings
            for pname, folds in parts.items() for k, te in enumerate(folds)]
    res = Parallel(n_jobs=N_JOBS)(
        delayed(one_fold)(X, y, subj, unit, te, seed + 1000 * k)
        for _, y, seed, _, k, te in jobs)
    out = {}
    for name, _, _ in labellings:
        wins = [pd.DataFrame(r).assign(partition=pname, fold=k)
                for (n_, _, _, pname, k, _), r in zip(jobs, res)
                if n_ == name and r is not None]
        w = pd.concat(wins, ignore_index=True)
        rows = [dict(partition=pname, labelling=name, **x)
                for pname, g in w.groupby("partition") for x in rules(g)]
        out[name] = (w, rows)
    return out


def bootstrap(w):
    """95% participant-bootstrap intervals for the observed-label rows."""
    rng = np.random.default_rng(SEED)
    out = []
    for pname, g in w.groupby("partition"):
        people = g.subj.unique()
        by = {s: h for s, h in g.groupby("subj")}
        point = {x["rule"]: x for x in rules(g)}
        draws = {r: {"coverage": [], "bal_retained": []} for r in point}
        for _ in range(N_BOOT):
            h = pd.concat([by[s] for s in rng.choice(people, len(people))],
                          ignore_index=True)
            for x in rules(h):
                draws[x["rule"]]["coverage"].append(x["coverage"])
                draws[x["rule"]]["bal_retained"].append(x["bal_retained"])
        for r, x in point.items():
            row = dict(partition=pname, rule=r)
            for m in ("coverage", "bal_retained"):
                v = np.array(draws[r][m], float)
                row[m] = x[m]
                row[m + "_lo"], row[m + "_hi"] = (
                    np.nanpercentile(v, [2.5, 97.5]) if np.isfinite(v).any()
                    else (np.nan, np.nan))
            out.append(row)
    return pd.DataFrame(out)


def null_labels(y, unit, subj, i):
    """Labelling i of the known null: identical to 77's DEED null (same
    generator seed), recording labels re-randomised within participant."""
    return pcd.permute_within(y, unit, subj,
                              np.random.default_rng(4_000_000 + i))


def main():
    X, y, subj, unit = load_deed()
    parts = partitions(subj)
    for pname, folds in parts.items():
        same = sum(np.isin(unit[te], unit[np.setdiff1d(np.arange(len(y)), te)])
                   .mean() for te in folds) / len(folds)
        log(f"{pname:<22} test windows whose recording is also in training: "
            f"{same:.4f}")
    log(f"DEED {len(y)} windows, {len(np.unique(unit))} recordings, "
        f"{len(np.unique(subj))} participants; B = {B}, R = {R_NULL}")

    done = (pd.read_csv(PARTIAL) if os.path.exists(PARTIAL)
            else pd.DataFrame(columns=["partition", "labelling"]))
    have = set(done.labelling)
    wfile = os.path.join(OUT, f"pcd_leaky_windows{TAG}.parquet")
    if "observed" not in have or not os.path.exists(wfile):
        w_obs, rows = run(X, subj, unit, parts,
                          [("observed", y, SEED)])["observed"]
        w_obs.to_parquet(wfile)
        done = pd.concat([done[done.labelling != "observed"],
                          pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(PARTIAL, index=False)
        log("observed labels done")
    w_obs = pd.read_parquet(wfile)

    todo = [i for i in range(R_NULL) if f"null {i}" not in have]
    log(f"known null: {R_NULL - len(todo)} labellings already done, "
        f"{len(todo)} to run in batches of {BATCH}")
    for b0 in range(0, len(todo), BATCH):
        batch = [(f"null {i}", null_labels(y, unit, subj, i), SEED + i)
                 for i in todo[b0:b0 + BATCH]]
        res = run(X, subj, unit, parts, batch)
        new = [r for name, _, _ in batch for r in res[name][1]]
        done = pd.concat([done, pd.DataFrame(new)], ignore_index=True)
        done.to_csv(PARTIAL, index=False)
        log(f"known null: {done.labelling.nunique() - 1}/{R_NULL} done")

    keep = {"observed"} | {f"null {i}" for i in range(R_NULL)}
    cond = done[done.labelling.isin(keep)].reset_index(drop=True).round(6)
    assert cond.labelling.nunique() == R_NULL + 1
    cond.round(6).to_csv(os.path.join(OUT, f"pcd_leaky_conditions{TAG}.csv"),
                         index=False)

    nul = cond[cond.labelling != "observed"]
    summ = (nul.groupby(["partition", "rule"])
            .agg(coverage_mean=("coverage", "mean"),
                 coverage_min=("coverage", "min"),
                 coverage_max=("coverage", "max"),
                 bal_mean=("bal_retained", "mean"),
                 bal_min=("bal_retained", "min"),
                 bal_max=("bal_retained", "max"),
                 n=("coverage", "size")).reset_index())
    summ.round(4).to_csv(os.path.join(OUT, f"pcd_leaky_summary{TAG}.csv"),
                         index=False)
    boot = bootstrap(w_obs)
    boot.round(4).to_csv(os.path.join(OUT, f"pcd_leaky_boot{TAG}.csv"),
                         index=False)
    with pd.option_context("display.width", 200):
        log("\nobserved labels (95% participant-bootstrap intervals)\n"
            + boot.round(4).to_string(index=False))
        log(f"\nknown null, {R_NULL} labellings\n"
            + summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
