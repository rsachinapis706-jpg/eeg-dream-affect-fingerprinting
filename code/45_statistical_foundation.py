"""
The statistical foundation: uncertainty, effect sizes, sensitivity, power.

Supersedes 42_uncertainty_power.py, which contained two defects that invalidated
its output and are corrected here.

  1. The matched-pairs rank-biserial correlation was computed from scipy's
     returned Wilcoxon statistic using a formula that does not correspond to it.
     It reported +0.5419 where the correct value is -0.0837 -- a moderate
     positive effect where the truth is a negligible negative one. It is now
     computed from the signed ranks directly, which removes the dependence on
     what `alternative` causes scipy to return.

  2. The power analysis injected a signal scaled by the MEAN of the per-feature
     standard deviations on unstandardised features, so the realised separation
     bore no dependable relation to the nominal effect size. A nominal 0.05
     produced 0.66 balanced accuracy, which is not a small effect. Features are
     now standardised before injection, and -- more importantly -- power is
     reported against the balanced accuracy the injected signal ACTUALLY
     achieves, not against the nominal parameter. That makes the calibration
     question moot: the reported quantity is "power to detect an effect that
     produces balanced accuracy X", which is directly interpretable and needs no
     assumption about how the injection maps onto an effect-size scale.

Configurations
--------------
Both configurations that matter are carried through every analysis: the
unweighted random forest used for the manuscript's headline numbers, and the
standardised balanced-weight logistic regression that 43/44 showed recovers a
weak but seed-stable effect. Reporting only one of them would misrepresent the
result in one direction or the other.

Null
----
Labels are shuffled across all recordings without restriction, destroying every
label structure including the between-participant component. This is the
stricter of the two nulls examined in 43_null_diagnostic.py.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")

SEED = 20260727
N_SPLITS = 5
N_JOBS = 20
N_BOOT = 2000
N_PERM = 200
N_SIM = 30
DELTAS = [0.0, 0.005, 0.01, 0.02, 0.04, 0.07, 0.12, 0.20]
COHORTS = [10, 20, 34]

VALENCE = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
META = {"segment_id", "subject", "oid", "epoch_idx", "n_epochs",
        "epochs_from_end", "E", "film", "stage", "has_text"}

CONFIGS = {
    "RF, unweighted (headline)": ("rf", False, None),
    "LogReg, scaled, balanced": ("lr", True, "balanced"),
}

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


def cv_predict(X, y, subj, kind, scale, weight, seed=SEED, threads=1):
    sp = StratifiedGroupKFold(N_SPLITS, shuffle=True,
                              random_state=seed).split(X, y, subj)
    yt, yp, gs = [], [], []
    for tr, te in sp:
        if len(np.unique(y[tr])) < 2:
            continue
        m = build(kind, scale, weight, seed, threads).fit(X[tr], y[tr])
        yp.append(m.predict(X[te])); yt.append(y[te]); gs.append(subj[te])
    return np.concatenate(yt), np.concatenate(yp), np.concatenate(gs)


def cluster_boot(yt, yp, groups, n=N_BOOT, seed=SEED):
    """Resample PARTICIPANTS, not windows. Windows within a participant are
    dependent, so resampling them would give an interval far too narrow."""
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uq}
    acc, bal = [], []
    for _ in range(n):
        pick = rng.choice(uq, size=len(uq), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        if len(np.unique(yt[sel])) < 2:
            continue
        acc.append(accuracy_score(yt[sel], yp[sel]))
        bal.append(balanced_accuracy_score(yt[sel], yp[sel]))
    pc = lambda a: (float(np.percentile(a, 2.5)),
                    float(np.percentile(a, 97.5)))
    return pc(acc), pc(bal)


def rank_biserial(diff):
    """Matched-pairs effect size r = (W+ - W-) / (W+ + W-), from signed ranks."""
    d = np.asarray(diff)
    d = d[d != 0]
    if len(d) == 0:
        return np.nan, 0
    r = stats.rankdata(np.abs(d))
    wp, wn = float(r[d > 0].sum()), float(r[d < 0].sum())
    return ((wp - wn) / (wp + wn) if (wp + wn) else np.nan), len(d)


def inject(Xs, y, delta, rng):
    """Class-dependent shift of size `delta` along one random unit direction,
    applied to STANDARDISED features so the shift is in SD units."""
    if delta == 0:
        return Xs
    u = rng.normal(size=Xs.shape[1])
    u /= np.linalg.norm(u)
    classes = np.unique(y)
    code = {c: i - (len(classes) - 1) / 2 for i, c in enumerate(classes)}
    return Xs + np.array([code[v] for v in y])[:, None] * delta * u[None, :]


def sim(Xs, y, subj, kind, scale, weight, delta, s):
    rng = np.random.default_rng(700_000 + s)
    return balanced_accuracy_score(
        *cv_predict(inject(Xs, y, delta, rng), y, subj,
                    kind, scale, weight)[:2])


def perm(Xs, y, subj, rec, kind, scale, weight, k):
    rng = np.random.default_rng(800_000 + k)
    out = y.copy()
    recs = np.unique(rec)
    labs = rng.permutation(np.array([y[rec == r][0] for r in recs]))
    for r, lab in zip(recs, labs):
        out[rec == r] = lab
    if len(np.unique(out)) < 2:
        return np.nan
    return balanced_accuracy_score(
        *cv_predict(Xs, out, subj, kind, scale, weight)[:2])


# Each representation's within-participant results live in their own file.
# An earlier version read the CBraMod file for both and labelled the result
# "spectral", so the spectral effect sizes printed in the manuscript were in
# fact the CBraMod ones. Reviewer 1 caught the inconsistency (Section IV-J).
WITHIN_SOURCES = {
    "within-participant, spectral": "within_subject.csv",           # 11_
    "within-participant, CBraMod frozen": "cbramod_within_subject.csv",  # 15_
}


def within_effects():
    """Paired effect sizes of each participant's observed balanced accuracy
    over that participant's own permutation-null mean."""
    log("-- B. within-participant effect sizes --")
    out = []
    for label, fname in WITHIN_SOURCES.items():
        w = pd.read_csv(os.path.join(OUT, fname)).dropna(subset=["bacc",
                                                                  "null"])
        diff = (w["bacc"] - w["null"]).to_numpy()
        rb, n_eff = rank_biserial(diff)
        dz = diff.mean() / (diff.std(ddof=1) + 1e-12)
        nz = diff[diff != 0]
        p_one = stats.wilcoxon(nz, alternative="greater").pvalue
        p_two = stats.wilcoxon(nz).pvalue
        log(f"{label}   [{fname}]   (n = {len(w)} participants)")
        log(f"  mean observed             : {w['bacc'].mean():.4f}")
        log(f"  mean null                 : {w['null'].mean():.4f}")
        log(f"  mean observed - null      : {diff.mean():+.4f}")
        log(f"  Cohen's d_z               : {dz:+.4f}")
        log(f"  rank-biserial correlation : {rb:+.4f}")
        log(f"  above / below own null    : {(diff > 0).sum()} / "
            f"{(diff < 0).sum()}")
        log(f"  Wilcoxon p, one-sided (>) : {p_one:.4f}")
        log(f"  Wilcoxon p, two-sided     : {p_two:.4f}")
        out.append(dict(part="effect_size", config=label,
                        observed=float(w["bacc"].mean()),
                        null=float(w["null"].mean()),
                        balanced=float(diff.mean()), cohens_dz=float(dz),
                        rank_biserial=float(rb), wilcoxon_p=float(p_one),
                        wilcoxon_p_two_sided=float(p_two),
                        n_above=int((diff > 0).sum()),
                        n_below=int((diff < 0).sum()), n=len(w)))
    log("")
    return out


def effects_only():
    """Recompute only section B and replace its rows in the results file,
    leaving the expensive sections A, C and D untouched."""
    path = os.path.join(OUT, "statistical_foundation.csv")
    new = pd.DataFrame(within_effects())
    old = pd.read_csv(path)
    keep = old[~((old["part"] == "effect_size")
                 & old["config"].astype(str).str.startswith("within"))]
    pd.concat([keep, new], ignore_index=True).round(4).to_csv(path,
                                                              index=False)
    with open(os.path.join(OUT, "run_statfound_effects.log"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"updated {path}")


def main():
    if "--effects-only" in sys.argv:
        effects_only()
        return
    X, y, subj, rec = load()
    Xs = StandardScaler().fit_transform(X)     # for the injection only
    uq = np.unique(subj)
    rows = []

    log("=" * 78)
    log("STATISTICAL FOUNDATION")
    log("=" * 78)
    log(f"windows {len(y)} | participants {len(uq)} | recordings "
        f"{len(np.unique(rec))} | chance 0.3333")
    log(f"cluster bootstrap {N_BOOT} over participants | "
        f"null permutations {N_PERM} | simulations per point {N_SIM}")
    log("")

    # ---------- A. uncertainty --------------------------------------------
    log("-- A. participant-wise performance, cluster-bootstrap 95% CI --")
    log(f"{'configuration':<28}{'accuracy':>24}{'balanced accuracy':>26}")
    log("-" * 78)
    preds = {}
    for name, (kind, scale, weight) in CONFIGS.items():
        yt, yp, gs = cv_predict(X, y, subj, kind, scale, weight,
                                threads=N_JOBS if kind == "rf" else 1)
        preds[name] = (yt, yp, gs)
        (alo, ahi), (blo, bhi) = cluster_boot(yt, yp, gs)
        a, b = accuracy_score(yt, yp), balanced_accuracy_score(yt, yp)
        log(f"{name:<28}{a:>8.4f} [{alo:.4f}, {ahi:.4f}]"
            f"{b:>10.4f} [{blo:.4f}, {bhi:.4f}]")
        rows.append(dict(part="uncertainty", config=name, accuracy=a,
                         acc_lo=alo, acc_hi=ahi, balanced=b,
                         bal_lo=blo, bal_hi=bhi, n_boot=N_BOOT))
    log("")

    # ---------- B. effect sizes -------------------------------------------
    rows.extend(within_effects())

    for name, (yt, yp, gs) in preds.items():
        b = balanced_accuracy_score(yt, yp)
        (_, _), (blo, bhi) = cluster_boot(yt, yp, gs)
        log(f"{name}: balanced accuracy minus chance = {b - 1/3:+.4f} "
            f"[{blo - 1/3:+.4f}, {bhi - 1/3:+.4f}]")
        rows.append(dict(part="effect_size", config=name,
                         balanced=b - 1 / 3, bal_lo=blo - 1 / 3,
                         bal_hi=bhi - 1 / 3))
    log("")

    # ---------- C. sensitivity: jackknife ---------------------------------
    log("-- C. leave-one-participant-out jackknife --")
    for name, (kind, scale, weight) in CONFIGS.items():
        if kind == "rf":
            log(f"{name:<28} skipped (cost); classifier sensitivity is "
                f"reported in null_diagnostic.csv")
            continue

        def drop(g):
            m = subj != g
            return balanced_accuracy_score(
                *cv_predict(X[m], y[m], subj[m], kind, scale, weight)[:2])

        j = np.array(Parallel(n_jobs=N_JOBS)(delayed(drop)(g) for g in uq))
        full = balanced_accuracy_score(*preds[name][:2])
        log(f"{name:<28} full {full:.4f} | jackknife {j.min():.4f}-"
            f"{j.max():.4f} | sd {j.std():.4f} | max shift "
            f"{np.abs(j - full).max():.4f}")
        rows.append(dict(part="jackknife", config=name, balanced=full,
                         bal_lo=float(j.min()), bal_hi=float(j.max()),
                         jack_sd=float(j.std())))
    log("")

    # ---------- D. power ---------------------------------------------------
    log("-- D. power, reported against ACHIEVED balanced accuracy --")
    log("Signal is injected into standardised features; what is reported is")
    log("the balanced accuracy the injection actually produces and the")
    log("probability of detecting it, so no effect-size calibration is")
    log("assumed. Detection = exceeding the 95th percentile of the")
    log("unrestricted permutation null for that cohort.")
    log("")
    kind, scale, weight = CONFIGS["LogReg, scaled, balanced"]
    rng0 = np.random.default_rng(SEED)

    for nsub in COHORTS:
        ids = uq if nsub >= len(uq) else rng0.choice(uq, nsub, replace=False)
        m = np.isin(subj, ids)
        Xc, Xsc, yc, sc, rc = X[m], Xs[m], y[m], subj[m], rec[m]
        t0 = time.time()
        nulls = Parallel(n_jobs=N_JOBS)(
            delayed(perm)(Xsc, yc, sc, rc, kind, scale, weight, k)
            for k in range(N_PERM))
        nv = np.array([a for a in nulls if np.isfinite(a)])
        thr = float(np.percentile(nv, 95))
        log(f"cohort {nsub:>2}: null mean {nv.mean():.4f}, "
            f"95th percentile {thr:.4f}   [{time.time() - t0:.0f}s]")
        log(f"{'':>10}{'delta':>8}{'achieved bal acc':>19}{'power':>8}")
        for dlt in DELTAS:
            vals = np.array(Parallel(n_jobs=N_JOBS)(
                delayed(sim)(Xsc, yc, sc, kind, scale, weight, dlt, s)
                for s in range(N_SIM)))
            pw_ = float((vals > thr).mean())
            log(f"{'':>10}{dlt:>8.3f}{vals.mean():>19.4f}{pw_:>8.2f}")
            rows.append(dict(part="power", config=name, cohort=nsub,
                             delta=dlt, balanced=float(vals.mean()),
                             power=pw_, threshold=thr, n_sim=N_SIM))
            pd.DataFrame(rows).round(4).to_csv(
                os.path.join(OUT, "statistical_foundation.csv"), index=False)
        log("")

    r = pd.DataFrame(rows)
    p34 = r[(r.part == "power") & (r.cohort == 34)].sort_values("balanced")
    hit = p34[p34.power >= 0.80]
    log("=" * 78)
    log("INTERPRETATION")
    log("=" * 78)
    if len(hit):
        mde = float(hit.iloc[0].balanced)
        log(f"At 34 participants the protocol has 80% power against effects")
        log(f"producing balanced accuracy of {mde:.4f} or above.")
        log(f"The observed value for this configuration is "
            f"{balanced_accuracy_score(*preds['LogReg, scaled, balanced'][:2]):.4f}.")
        log("The null is therefore bounded rather than merely asserted:")
        log("anything at or above that level would have been detected.")
    else:
        log("80% power is not reached at any injected level tested, which")
        log("bounds the null weakly and must be reported as such.")

    r.round(4).to_csv(os.path.join(OUT, "statistical_foundation.csv"),
                      index=False)
    with open(os.path.join(OUT, "test35_statistical_foundation.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"\nWrote {OUT}\\statistical_foundation.csv")


if __name__ == "__main__":
    main()
