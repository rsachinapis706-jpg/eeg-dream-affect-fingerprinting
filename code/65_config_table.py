"""
Rebuild Table 5 -- configuration sensitivity and fold-assignment stability --
from one procedure, so that every row is computed the same way.

Why this replaces 43_null_diagnostic.py and 44_seed_stability.py for Table 5
---------------------------------------------------------------------------
Reviewer 1 found that the table named five configurations but displayed four,
and asked for the multiplicity correction to be reconciled. Checking the two
source scripts showed more than that:

  * the fifth configuration (RF, scaled, balanced weight) was specified but its
    run never completed;
  * configurations were tested against different permutation counts
    (logistic regression 300, forest 100, dummy 100);
  * the dummy row reported its within-participant p beside other rows'
    unrestricted p;
  * the "pooled p" compared the MEAN over ten fold assignments against a null
    built from SINGLE-assignment statistics. Fold assignments reuse the same
    participants, so they are not independent replications, and the averaged
    statistic needs its own null.

Here every configuration is evaluated on the same data, the same fold
assignments and the SAME permuted label vectors, against both nulls:

  within   labels permuted across recordings WITHIN each participant; keeps each
           participant's label multiset, so it tests for information beyond
           participant-level label composition
  free     labels permuted across ALL recordings; tests for any association
           between features and labels, including through composition

Labels are always permuted at the recording level, the unit at which dream
affect is reported, so all windows of a recording keep a common label.

Statistics
  single   balanced accuracy at the first fold assignment (Table 5a)
  mean10   balanced accuracy averaged over ten fold assignments (Table 5b); its
           null applies ONE permutation to all ten assignments and averages, so
           the null has the same dependence structure as the observed mean

p = (1 + #{null >= observed}) / (1 + B).

Multiplicity
  The candidate family is the four decoders. The stratified dummy is a
  calibration control -- it cannot carry signal by construction -- and is
  reported but not counted. Holm across the four, and a single-step
  max-statistic (Westfall-Young) correction, which is exact for this family
  because every configuration shares each permutation.

Balanced accuracy is computed on out-of-fold predictions pooled across the five
folds, so a fold lacking a class does not undefine the metric.

Results stream to out/config_table_raw.csv and the run resumes where it stopped.

Permutation counts (final, 1 Oct 2026): every configuration uses 1,000
permutations for the single-assignment statistic. For the averaged statistic
(each permutation costs ten evaluations) the forests use 30 and logistic
regression and the dummy 100. The max-statistic correction needs permutations
shared by all four candidates, so it uses 1,000 (single) and the first 30
(averaged); its smallest attainable p is 1/1001 and 1/31. The run was extended
in stages (10, 100, 200, 1,000 single; 10, 30 averaged); permutation b always
uses the same relabelling, so the stages are one design. Counts are stated in
the table footnote.

Standard deviations over fold assignments use the population form (ddof=0),
as in 44_seed_stability.py and 51_protocol_table.py.

Fix (1 Oct 2026): summarise() now restricts each null to its configured count
BEFORE building the single-assignment null. Previously rows for b = 10 left by
an earlier run entered the forest nulls (11 instead of 10).
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
RAW = os.path.join(OUT, "config_table_raw.csv")
SUMMARY = os.path.join(OUT, "config_table.csv")
LOG = os.path.join(OUT, "run_config_table.log")

SEEDS = [20260727 + i for i in range(10)]      # identical to 44_seed_stability
N_SPLITS = 5
B_SINGLE = 1000          # permutations for the single-assignment statistic
B_MEAN = 100             # of those, the first B_MEAN also get all ten seeds
# Time-limited run: forests get fewer permutations. Stated in the output.
B_SINGLE_BY_KIND = {"rf": 1000, "lr": 1000, "dummy": 1000}
B_MEAN_BY_KIND = {"rf": 30, "lr": 100, "dummy": 100}


def b_single(cfg):
    return B_SINGLE_BY_KIND[CONFIGS[cfg][0]]


def b_mean(cfg):
    return B_MEAN_BY_KIND[CONFIGS[cfg][0]]
PERM_BASE = 500_000
N_JOBS = int(os.environ.get("T5_JOBS", 12))  # worker processes; the count does not affect results
RF_THREADS = 3           # threads per forest; results do not depend on this
BLAS_THREADS = 3         # pinned: OpenBLAS results differ between 1 and >1 threads

VALENCE = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
META = {"segment_id", "subject", "oid", "epoch_idx", "n_epochs",
        "epochs_from_end", "E", "film", "stage", "has_text"}

CONFIGS = {
    "RF, unweighted (headline)": ("rf", False, None),
    "RF, scaled, balanced": ("rf", True, "balanced"),
    "LogReg, scaled, unweighted": ("lr", True, None),
    "LogReg, scaled, balanced": ("lr", True, "balanced"),
    "Dummy, stratified (control)": ("dummy", False, None),
}
CANDIDATES = [c for c in CONFIGS if not c.startswith("Dummy")]

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def load():
    d = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    fc = [c for c in d.columns if c not in META]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = d[d.E.isin(VALENCE)]
    return (d[fc].to_numpy(float), d.E.map(VALENCE).to_numpy(),
            d.subject.to_numpy(), d.segment_id.to_numpy())


def build(kind, scale, weight, seed):
    if kind == "rf":
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   class_weight=weight, n_jobs=RF_THREADS,
                                   random_state=seed)
    elif kind == "lr":
        m = LogisticRegression(C=0.1, max_iter=2000, class_weight=weight)
    else:
        m = DummyClassifier(strategy="stratified", random_state=seed)
    return make_pipeline(StandardScaler(), m) if scale else m


def evaluate(X, y, subj, cfg, fold_seed):
    """Fold assignment and model seed both follow fold_seed, as in 44.

    BLAS threads are pinned. OpenBLAS uses a different kernel on one thread
    than on several, and the logistic-regression solution then differs in the
    last digits: 0.378490 on one thread against 0.378319 on three or more for
    the balanced configuration at the first fold assignment. Leaving the count
    to the process pool made results depend on how many workers ran. Three
    threads reproduce the values previously published.
    """
    kind, scale, weight = CONFIGS[cfg]
    sp = StratifiedGroupKFold(N_SPLITS, shuffle=True,
                              random_state=fold_seed).split(X, y, subj)
    yt, yp = [], []
    with threadpool_limits(BLAS_THREADS, user_api="blas"):
        for tr, te in sp:
            if len(np.unique(y[tr])) < 2:
                continue
            m = build(kind, scale, weight, fold_seed).fit(X[tr], y[tr])
            yp.append(m.predict(X[te])); yt.append(y[te])
    return balanced_accuracy_score(np.concatenate(yt), np.concatenate(yp))


def rec_labels(y, rec):
    recs, first = np.unique(rec, return_index=True)
    return recs, y[first]


def permuted(y, subj, rec, b, mode):
    """Recording-level permutation; identical for every configuration."""
    rng = np.random.default_rng(PERM_BASE + b)
    recs, lab = rec_labels(y, rec)
    rec_subj = subj[np.unique(rec, return_index=True)[1]]
    new = lab.copy()
    if mode == "free":
        new = rng.permutation(lab)
    else:
        for g in np.unique(rec_subj):
            m = np.where(rec_subj == g)[0]
            if len(m) > 1:
                new[m] = rng.permutation(lab[m])
    lut = dict(zip(recs, new))
    return np.array([lut[r] for r in rec])


def task(X, y, subj, rec, cfg, mode, b):
    """One permutation of one configuration under one null."""
    if mode == "observed":
        yl = y
        seeds = SEEDS
    else:
        yl = permuted(y, subj, rec, b, mode)
        seeds = SEEDS if b < b_mean(cfg) else SEEDS[:1]
    return [(cfg, mode, b, s, evaluate(X, yl, subj, cfg, s)) for s in seeds]


def timing(X, y, subj):
    """Single-core cost of one five-fold evaluation per configuration."""
    log("timing one five-fold evaluation per configuration (single core):")
    est = 0.0
    for cfg in CONFIGS:
        t = time.time()
        evaluate(X, y, subj, cfg, SEEDS[0])
        dt = time.time() - t
        evals = 2 * (B_MEAN * len(SEEDS) + (B_SINGLE - B_MEAN)) + len(SEEDS)
        est += dt * evals
        log(f"  {cfg:<30}{dt:7.2f} s  -> {evals} evaluations")
    log(f"estimated wall time on {N_JOBS} workers: "
        f"{est / N_JOBS / 3600:.1f} h")


def summarise():
    r = pd.read_csv(RAW)
    # Extending the averaged statistic re-runs a permutation under all ten
    # assignments, which writes the first assignment again; keep the original
    # row so the single-assignment null is unchanged and every (b, seed) is
    # counted once.
    r = r.drop_duplicates(["config", "mode", "b", "seed"], keep="first")
    obs = (r[r["mode"] == "observed"].groupby("config").bacc)
    rows = []
    null_single, null_mean = {}, {}
    for mode in ("within", "free"):
        n = r[r["mode"] == mode]
        n = n[n.b < n.config.map(b_single)]
        s0 = n[n.seed == SEEDS[0]].pivot_table(index="b", columns="config",
                                                values="bacc")
        full = n[n.b < n.config.map(b_mean)].groupby(["config", "b"]).bacc
        cnt = full.count().unstack(0)
        mm = full.mean().unstack(0).where(cnt == len(SEEDS))
        null_single[mode], null_mean[mode] = s0, mm
    for cfg in CONFIGS:
        o_all = r[(r["mode"] == "observed") & (r.config == cfg)]
        o1 = float(o_all[o_all.seed == SEEDS[0]].bacc.iloc[0])
        om = float(o_all.bacc.mean())
        row = dict(config=cfg, observed_single=o1, observed_mean10=om,
                   sd_mean10=float(o_all.bacc.std(ddof=0)))
        for mode in ("within", "free"):
            ns = null_single[mode][cfg].dropna().to_numpy()
            nm = null_mean[mode][cfg].dropna().to_numpy()
            row[f"{mode}_B_single"] = len(ns)
            row[f"{mode}_null_mean_single"] = ns.mean()
            row[f"{mode}_p_single"] = (1 + (ns >= o1).sum()) / (1 + len(ns))
            row[f"{mode}_B_mean10"] = len(nm)
            row[f"{mode}_null_mean_mean10"] = nm.mean() if len(nm) else np.nan
            row[f"{mode}_p_mean10"] = ((1 + (nm >= om).sum()) / (1 + len(nm))
                                       if len(nm) else np.nan)
        rows.append(row)
    t = pd.DataFrame(rows).set_index("config")

    # Multiplicity over the four candidate decoders only.
    for mode in ("within", "free"):
        for stat, nulls, obs_col in (("single", null_single, "observed_single"),
                                     ("mean10", null_mean, "observed_mean10")):
            pc = f"{mode}_p_{stat}"
            p = t.loc[CANDIDATES, pc].sort_values()
            k = len(p)
            holm, run = {}, 0.0
            for i, (c, v) in enumerate(p.items()):
                run = max(run, min(1.0, (k - i) * v))
                holm[c] = run
            t.loc[CANDIDATES, f"{pc}_holm"] = pd.Series(holm)
            N = nulls[mode][CANDIDATES].dropna()
            mx = N.max(axis=1).to_numpy()
            for c in CANDIDATES:
                t.loc[c, f"{pc}_maxT"] = ((1 + (mx >= t.loc[c, obs_col]).sum())
                                          / (1 + len(mx)))
    t.round(4).to_csv(SUMMARY)
    return t


def main():
    X, y, subj, rec = load()
    log("=" * 78)
    log("TABLE 5 REBUILT: five configurations, both nulls, shared permutations")
    log("=" * 78)
    log(f"windows {len(y)} | participants {len(np.unique(subj))} | "
        f"recordings {len(np.unique(rec))}")
    log(f"single-assignment permutations {B_SINGLE}; averaged-statistic "
        f"permutations {B_MEAN} over {len(SEEDS)} fold assignments")
    if "--time" in sys.argv:
        timing(X, y, subj)
        return

    done = set()
    if os.path.exists(RAW):
        prev = pd.read_csv(RAW)
        need = prev.groupby(["config", "mode", "b"]).seed.count()
        for (c, m, b), n in need.items():
            want = len(SEEDS) if (m == "observed" or b < b_mean(c)) else 1
            if n >= want:
                done.add((c, m, int(b)))
        log(f"resuming: {len(done)} tasks already complete")
    else:
        pd.DataFrame(columns=["config", "mode", "b", "seed", "bacc"]).to_csv(
            RAW, index=False)

    # Observed first, then permutations interleaved across configurations so a
    # partial run still gives every configuration a usable null.
    todo = [(c, "observed", -1) for c in CONFIGS]
    for b in range(B_SINGLE):
        for mode in ("within", "free"):
            for c in CONFIGS:
                if b < b_single(c):
                    todo.append((c, mode, b))
    todo = [t for t in todo if t not in done]
    log(f"tasks to run: {len(todo)}")

    t0 = time.time()
    gen = Parallel(n_jobs=N_JOBS, return_as="generator_unordered",
                   batch_size=1)(delayed(task)(X, y, subj, rec, c, m, b)
                                 for c, m, b in todo)
    buf = []
    for i, res in enumerate(gen, 1):
        buf.extend(res)
        if len(buf) >= 20 or i == len(todo):     # small batches: little lost if stopped
            pd.DataFrame(buf, columns=["config", "mode", "b", "seed",
                                       "bacc"]).to_csv(RAW, mode="a",
                                                       header=False,
                                                       index=False)
            buf = []
        if i % 250 == 0 or i == len(todo):
            el = time.time() - t0
            log(f"  {i}/{len(todo)} tasks  {el / 60:.1f} min elapsed  "
                f"~{el / i * (len(todo) - i) / 60:.0f} min left")

    t = summarise()
    log("")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        log(t.round(4).to_string())
    with open(LOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
