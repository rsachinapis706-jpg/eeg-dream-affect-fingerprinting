"""
Factorial separation of the protocol choices in Table 4(b) (Reviewer 1,
comment 4).

The problem
-----------
Table 4(b) came from 10_leakage_mechanism.py and compared five orderings, each
reported with the best of three classifiers, at one fold assignment. The
manuscript's "feature selection outside the fold adds 15 points" compared
0.3831 (participant-separated, nested) with 0.5282 (window-shuffled, selection
before the split): two factors changed at once, and a third -- which of the
three classifiers was best -- changed too. The like-for-like step within the
window-shuffled rows is 0.5120 -> 0.5282.

Design
------
One classifier, fixed in advance: random forest, 300 trees,
min_samples_leaf=2, the headline model. Three factors, fully crossed:

  partition      participant   participants separated (StratifiedGroupKFold)
                 window        windows shuffled (StratifiedKFold)
  selection      none | inside the training fold | before the split
                 (ANOVA F, k = 40, as in 10_ and 37_)
  oversampling   none | inside the training fold | before the split (SMOTE)

2 x 3 x 3 = 18 cells, ten fold assignments each. When any operation is
applied before the split, the scaler is also fitted on all data first, as it
would be in that practice; otherwise scaling is fitted inside each fold.

Synthetic points
----------------
When oversampling precedes the split, synthetic points can land in test folds.
Every cell is therefore scored on the untouched REAL test windows, which is
the primary result; the all-points and synthetic-only scores are kept beside
it so the published-style figure can be reconciled. Under participant
separation a synthetic point has no participant of its own; it is assigned to
the participant of its nearest real window in the space SMOTE worked in, which
is how a point interpolated from a test participant ends up on the training
side of another fold.

Outputs
-------
  out/protocol_factorial_raw.csv    one row per cell and fold assignment
  out/protocol_factorial_preds.parquet  real-window predictions, for the
                                    participant-level bootstrap of effects
  out/protocol_factorial.csv        cell means, and paired effects of each
                                    factor with the others held fixed

Time-limited run (30 Sep 2026): 100 trees; three fold assignments; the twelve
cells that isolate each factor -- per partition: nothing, selection inside,
selection before, oversampling inside, oversampling before, and both before.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
RAW = os.path.join(OUT, "protocol_factorial_raw.csv")
PREDS = os.path.join(OUT, "protocol_factorial_preds.parquet")
SUMMARY = os.path.join(OUT, "protocol_factorial.csv")
LOG = os.path.join(OUT, "run_protocol_factorial.log")
if "--full" in sys.argv:
    RAW, PREDS, SUMMARY, LOG = (x.replace(".csv", "_full.csv")
                                .replace(".parquet", "_full.parquet")
                                .replace(".log", "_full.log")
                                for x in (RAW, PREDS, SUMMARY, LOG))

FULL = "--full" in sys.argv
# Time-limited run (30 Sep 2026): 3 assignments, 100 trees, 12 cells.
# --full (round 3, Reviewer 2, comment 9): the design as specified above,
# all 18 cells, ten assignments, 300 trees, written to *_full files.
SEEDS = [20260727 + i for i in range(10 if FULL else 3)]
N_TREES = 300 if FULL else 100
N_SPLITS = 5
K = 40
N_BOOT = 2000
N_WORKERS = int(os.environ.get("FACT_JOBS", 8)) if FULL else 1
RF_THREADS = 3 if FULL else 22
VALENCE = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
META = {"segment_id", "subject", "oid", "epoch_idx", "n_epochs",
        "epochs_from_end", "E", "film", "stage", "has_text"}
LEVELS = ("none", "inside", "before")
CELLS = ([(p, s, o) for p in ("participant", "window")
          for s in ("none", "inside", "before")
          for o in ("none", "inside", "before")] if FULL else
         [(p, s, o) for p in ("participant", "window")
          for s, o in (("none", "none"), ("inside", "none"),
                       ("before", "none"), ("none", "inside"),
                       ("none", "before"), ("before", "before"))])

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def load(smoke=False):
    d = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    fc = [c for c in d.columns if c not in META]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = d[d.E.isin(VALENCE)].reset_index(drop=True)
    if smoke:
        d = d[d.subject.isin(sorted(d.subject.unique())[:10])].reset_index(
            drop=True)
    return (d[fc].to_numpy(float), d.E.map(VALENCE).to_numpy(),
            d.subject.to_numpy())


def run_cell(X, y, subj, cell, seed, threads, n_trees=300):
    part, sel, ovs = cell
    n0 = len(y)
    Xw, yw, gw = X, y, subj
    real = np.ones(n0, bool)
    origin = np.arange(n0)                    # real index of every row
    pre = "before" in (sel, ovs)
    if pre:
        Xw = StandardScaler().fit_transform(Xw)
    if sel == "before":
        Xw = SelectKBest(f_classif, k=K).fit_transform(Xw, yw)
    if ovs == "before":
        Xs, ys = SMOTE(random_state=seed).fit_resample(Xw, yw)
        assert np.array_equal(Xs[:n0], Xw)
        nn = NearestNeighbors(n_neighbors=1).fit(Xw)
        near = nn.kneighbors(Xs[n0:], return_distance=False)[:, 0]
        gw = np.concatenate([subj, subj[near]])
        origin = np.concatenate([origin, np.full(len(ys) - n0, -1)])
        real = np.arange(len(ys)) < n0
        Xw, yw = Xs, ys
    if part == "participant":
        sp = StratifiedGroupKFold(N_SPLITS, shuffle=True,
                                  random_state=seed).split(Xw, yw, gw)
    else:
        sp = StratifiedKFold(N_SPLITS, shuffle=True,
                             random_state=seed).split(Xw, yw)

    yt, yp, rt, oi = [], [], [], []
    for tr, te in sp:
        if len(np.unique(yw[tr])) < 2:
            continue
        Xtr, Xte, ytr = Xw[tr], Xw[te], yw[tr]
        if not pre:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        if sel == "inside":
            sk = SelectKBest(f_classif, k=K).fit(Xtr, ytr)
            Xtr, Xte = sk.transform(Xtr), sk.transform(Xte)
        if ovs == "inside":
            Xtr, ytr = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
        m = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=2,
                                   n_jobs=threads,
                                   random_state=seed).fit(Xtr, ytr)
        yp.append(m.predict(Xte)); yt.append(yw[te])
        rt.append(real[te]); oi.append(origin[te])
    yt, yp, rt, oi = map(np.concatenate, (yt, yp, rt, oi))
    res = dict(partition=part, selection=sel, oversampling=ovs, seed=seed,
               acc_real=accuracy_score(yt[rt], yp[rt]),
               bal_real=balanced_accuracy_score(yt[rt], yp[rt]),
               acc_all=accuracy_score(yt, yp),
               bal_all=balanced_accuracy_score(yt, yp),
               synth_frac_test=float((~rt).mean()))
    # always present, so every appended CSV row has the same columns
    res["acc_synth"] = (accuracy_score(yt[~rt], yp[~rt]) if (~rt).any()
                        else np.nan)
    preds = pd.DataFrame(dict(idx=oi[rt], y=yt[rt], pred=yp[rt]))
    for k in ("partition", "selection", "oversampling", "seed"):
        preds[k] = res[k]
    return res, preds


def bootstrap_diff(pa, pb, subj, rng):
    """Participant-level bootstrap of a balanced-accuracy difference between
    two cells, on the same real windows."""
    m = pa.merge(pb, on="idx", suffixes=("_a", "_b"))
    m["s"] = subj[m.idx.to_numpy()]
    groups = {g: np.where(m.s.to_numpy() == g)[0] for g in np.unique(m.s)}
    uq = np.array(list(groups))
    ya, pa_, pb_ = m.y_a.to_numpy(), m.pred_a.to_numpy(), m.pred_b.to_numpy()
    out = []
    for _ in range(N_BOOT):
        sel = np.concatenate([groups[g] for g in rng.choice(uq, len(uq))])
        if len(np.unique(ya[sel])) < 2:
            continue
        out.append(balanced_accuracy_score(ya[sel], pa_[sel])
                   - balanced_accuracy_score(ya[sel], pb_[sel]))
    return np.percentile(out, [2.5, 97.5])


def effects(raw, preds, subj):
    """Paired effect of each factor with the other two held fixed."""
    rng = np.random.default_rng(20260930)
    rows = []
    specs = []
    for p in ("participant", "window"):
        for o in LEVELS:
            specs.append(("selection before vs inside", (p, "before", o),
                          (p, "inside", o)))
        for s in LEVELS:
            specs.append(("oversampling before vs inside", (p, s, "before"),
                          (p, s, "inside")))
    for s in LEVELS:
        for o in LEVELS:
            specs.append(("window vs participant partition", ("window", s, o),
                          ("participant", s, o)))
    key = ["partition", "selection", "oversampling"]
    have = set(map(tuple, raw[key].drop_duplicates().to_numpy()))
    specs = [x for x in specs if x[1] in have and x[2] in have]
    for name, a, b in specs:
        ra = raw.set_index(key).loc[a].set_index("seed")
        rb = raw.set_index(key).loc[b].set_index("seed")
        da = (ra.bal_real - rb.bal_real).dropna()
        dac = (ra.acc_real - rb.acc_real).dropna()
        s0 = SEEDS[0]
        pa = preds[(preds.partition == a[0]) & (preds.selection == a[1])
                   & (preds.oversampling == a[2]) & (preds.seed == s0)]
        pb = preds[(preds.partition == b[0]) & (preds.selection == b[1])
                   & (preds.oversampling == b[2]) & (preds.seed == s0)]
        lo, hi = bootstrap_diff(pa[["idx", "y", "pred"]],
                                pb[["idx", "y", "pred"]], subj, rng)
        rows.append(dict(effect=name, a="/".join(a), b="/".join(b),
                         d_bal_mean=da.mean(), d_bal_sd=da.std(),
                         d_acc_mean=dac.mean(), d_acc_sd=dac.std(),
                         d_bal_ci_lo=lo, d_bal_ci_hi=hi, n_seeds=len(da)))
    return pd.DataFrame(rows)


def main():
    smoke = "--smoke" in sys.argv
    X, y, subj = load(smoke)
    raw_path = RAW.replace(".csv", "_smoke.csv") if smoke else RAW
    pred_path = PREDS.replace(".parquet", "_smoke.parquet") if smoke else PREDS
    seeds = SEEDS[:1] if smoke else SEEDS
    log("=" * 78)
    log("PROTOCOL FACTORIAL: partition x selection x oversampling, RF fixed")
    log("=" * 78)
    log(f"windows {len(y)} | participants {len(np.unique(subj))} | cells "
        f"{len(CELLS)} | fold assignments {len(seeds)}")

    done = set()
    if os.path.exists(raw_path):
        p = pd.read_csv(raw_path)
        done = set(zip(p.partition, p.selection, p.oversampling, p.seed))
    todo = [(c, s) for c in CELLS for s in seeds if (*c, s) not in done]
    log(f"evaluations to run: {len(todo)}")
    t0 = time.time()
    gen = Parallel(n_jobs=1 if smoke else N_WORKERS,
                   return_as="generator_unordered")(
        delayed(run_cell)(X, y, subj, c, s, 2 if smoke else RF_THREADS,
                          50 if smoke else N_TREES) for c, s in todo)
    pending = []

    def flush():
        if not pending:
            return
        old = pd.read_parquet(pred_path) if os.path.exists(pred_path) else None
        new = pd.concat(pending)
        (pd.concat([old, new]) if old is not None else new).to_parquet(
            pred_path)
        pending.clear()

    for i, (res, pr) in enumerate(gen, 1):
        # predictions are written before the summary row, so a completed row
        # always has its predictions on disk when the run is resumed
        pending.append(pr)
        flush()
        pd.DataFrame([res]).to_csv(raw_path, mode="a",
                                   header=not os.path.exists(raw_path),
                                   index=False)
        if i % 20 == 0 or i == len(todo):
            log(f"  {i}/{len(todo)}  ({(time.time() - t0) / 60:.1f} min)")

    raw = pd.read_csv(raw_path)
    cells = (raw.groupby(["partition", "selection", "oversampling"])
             [["acc_real", "bal_real", "acc_all", "bal_all",
               "synth_frac_test"]].agg(["mean", "std"]))
    log("")
    with pd.option_context("display.width", 220):
        log(cells.round(4).to_string())
    if smoke:
        return
    eff = effects(raw, pd.read_parquet(pred_path), subj)
    eff.round(4).to_csv(SUMMARY, index=False)
    log("")
    with pd.option_context("display.width", 220, "display.max_colwidth", 40):
        log(eff.round(4).to_string(index=False))
    with open(LOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
