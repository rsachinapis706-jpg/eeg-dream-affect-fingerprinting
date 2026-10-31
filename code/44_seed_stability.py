"""
Is the weak above-chance effect stable across fold assignments? (R2-4, and the
decisive test for this paper's central claim.)

Background
----------
43_null_diagnostic.py found that one configuration -- logistic regression with
standardisation and balanced class weights -- reaches 0.3783 subject-wise
balanced accuracy on DEED, exceeding an unrestricted permutation null at
p = 0.023. The paper's headline configuration, an unweighted random forest
without standardisation, reaches 0.3204, below the 0.3333 chance level.

That result rests on a single fold assignment. StratifiedGroupKFold's
assignment changed between scikit-learn 1.4 and 1.9 and moved the grouped
results by more than a point, so a single-split p-value is not sufficient
evidence for a claim this consequential. Two outcomes are possible and both are
useful:

  A. The effect persists across seeds. A weak participant-generalizable signal
     exists. It is far too small to account for published performance, but the
     manuscript's "no detectable signal" wording must be narrowed.

  B. The effect does not persist. The single-seed result was a property of one
     fold assignment, and the near-chance conclusion is reinstated -- now with
     multi-seed evidence rather than a single split.

Design
------
For every seed the ENTIRE procedure is repeated: the fold assignment, the
observed statistic, and the permutation null are all recomputed with that seed,
and a per-seed p-value is obtained. Reusing one null across seeds would
understate the variability being measured.

Labels are permuted across recordings without restriction, which destroys all
label structure including the between-participant component. This is the
stricter of the two nulls in 43_null_diagnostic.py and the appropriate
reference for a claim about decodable affect.

Reported: per-seed observed values and p-values, the mean and standard
deviation across seeds, the fraction of seeds reaching p < 0.05, and a
Bonferroni adjustment for the five configurations that were examined before
this one was singled out. The selection history is recorded because the
configuration was chosen after inspecting results, not in advance.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")

SEEDS = [20260727 + i for i in range(10)]
N_SPLITS = 5
N_JOBS = 20
N_CONFIGS_EXAMINED = 5          # for the Bonferroni note; see module docstring

VALENCE = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
META = {"segment_id", "subject", "oid", "epoch_idx", "n_epochs",
        "epochs_from_end", "E", "film", "stage", "has_text"}

PLAN = [
    ("LogReg, scaled, balanced weight", "lr", True, "balanced", 200),
    ("LogReg, scaled, no weight",       "lr", True, None,       100),
    ("RF, no scaling, no weight (paper)", "rf", False, None,     40),
]

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


def build(kind, scale, weight, seed, threads=1):
    if kind == "rf":
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   class_weight=weight, n_jobs=threads,
                                   random_state=seed)
    else:
        m = LogisticRegression(C=0.1, max_iter=2000, class_weight=weight)
    return make_pipeline(StandardScaler(), m) if scale else m


def evaluate(X, y, subj, kind, scale, weight, fold_seed, threads=1):
    """Fold assignment AND model seed both follow fold_seed."""
    sp = StratifiedGroupKFold(N_SPLITS, shuffle=True,
                              random_state=fold_seed).split(X, y, subj)
    yt, yp = [], []
    for tr, te in sp:
        if len(np.unique(y[tr])) < 2:
            continue
        m = build(kind, scale, weight, fold_seed, threads).fit(X[tr], y[tr])
        yp.append(m.predict(X[te])); yt.append(y[te])
    return balanced_accuracy_score(np.concatenate(yt), np.concatenate(yp))


def perm_free(y, rec, rng):
    """Shuffle labels across ALL recordings: destroys every label structure."""
    out = y.copy()
    recs = np.unique(rec)
    labs = rng.permutation(np.array([y[rec == r][0] for r in recs]))
    for r, lab in zip(recs, labs):
        out[rec == r] = lab
    return out


def one_perm(X, y, subj, rec, kind, scale, weight, fold_seed, k):
    rng = np.random.default_rng(int(fold_seed) * 1000 + k)
    yl = perm_free(y, rec, rng)
    if len(np.unique(yl)) < 2:
        return np.nan
    return evaluate(X, yl, subj, kind, scale, weight, fold_seed)


def main():
    X, y, subj, rec = load()
    only = sys.argv[1] if len(sys.argv) > 1 else None

    log("=" * 78)
    log("SEED STABILITY OF THE SUBJECT-WISE EFFECT")
    log("=" * 78)
    log(f"windows {len(y)} | participants {len(np.unique(subj))} | "
        f"recordings {len(np.unique(rec))}")
    log(f"seeds {len(SEEDS)} | null: labels shuffled across all recordings")
    log("chance 0.3333")
    log("")

    rows = []
    for name, kind, scale, weight, nperm in PLAN:
        if only and only.lower() not in name.lower():
            continue
        log(f"### {name}   ({nperm} permutations per seed)")
        log(f"{'seed':>12}{'observed':>11}{'null mean':>11}{'null p95':>11}"
            f"{'p':>8}")
        log("-" * 78)
        t0 = time.time()
        per_seed = []
        pooled_null = []
        for sd in SEEDS:
            obs = evaluate(X, y, subj, kind, scale, weight, sd,
                           threads=N_JOBS if kind == "rf" else 1)
            vals = Parallel(n_jobs=N_JOBS)(
                delayed(one_perm)(X, y, subj, rec, kind, scale, weight, sd, k)
                for k in range(nperm))
            v = np.array([a for a in vals if np.isfinite(a)])
            p = (np.sum(v >= obs) + 1) / (len(v) + 1)
            pooled_null.append(v)
            per_seed.append((sd, obs, v.mean(), float(np.percentile(v, 95)), p))
            log(f"{sd:>12}{obs:>11.4f}{v.mean():>11.4f}"
                f"{float(np.percentile(v, 95)):>11.4f}{p:>8.3f}")
            rows.append(dict(config=name, seed=sd, observed=obs,
                             null_mean=v.mean(),
                             null_p95=float(np.percentile(v, 95)),
                             p=float(p), n_perm=len(v)))
            pd.DataFrame(rows).round(4).to_csv(
                os.path.join(OUT, "seed_stability.csv"), index=False)

        obsv = np.array([r[1] for r in per_seed])
        ps = np.array([r[4] for r in per_seed])
        allnull = np.concatenate(pooled_null)
        p_pooled = (np.sum(allnull >= obsv.mean()) + 1) / (len(allnull) + 1)

        log("-" * 78)
        log(f"observed  mean {obsv.mean():.4f}  sd {obsv.std():.4f}  "
            f"range {obsv.min():.4f}-{obsv.max():.4f}")
        log(f"p-values  median {np.median(ps):.3f}  "
            f"range {ps.min():.3f}-{ps.max():.3f}")
        log(f"seeds with p < 0.05                : {int((ps < 0.05).sum())}"
            f" of {len(ps)}")
        log(f"mean observed vs pooled null       : p = {p_pooled:.4f}")
        log(f"Bonferroni over {N_CONFIGS_EXAMINED} configurations examined"
            f" : p = {min(1.0, p_pooled * N_CONFIGS_EXAMINED):.4f}")
        log(f"[{time.time() - t0:.0f}s]")
        log("")

        rows.append(dict(config=name, seed=-1, observed=obsv.mean(),
                         null_mean=allnull.mean(),
                         null_p95=float(np.percentile(allnull, 95)),
                         p=float(p_pooled), n_perm=len(allnull),
                         obs_sd=float(obsv.std()),
                         seeds_significant=int((ps < 0.05).sum()),
                         p_bonferroni=float(min(1.0,
                                                p_pooled * N_CONFIGS_EXAMINED)))
                    )
        pd.DataFrame(rows).round(4).to_csv(
            os.path.join(OUT, "seed_stability.csv"), index=False)

    with open(os.path.join(OUT, "test34_seed_stability.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"Wrote {OUT}\\seed_stability.csv")


if __name__ == "__main__":
    main()
