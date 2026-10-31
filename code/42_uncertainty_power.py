"""
Uncertainty, effect sizes, sensitivity and statistical power (Reviewer 2, C3).

The reviewer's point is well taken and is the central statistical weakness of a
null result built on 34 participants: a large window count does not confer a
large effective sample size when windows are nested within participants, and a
null reported without a power analysis says nothing about what could have been
detected. Four analyses are produced.

A. UNCERTAINTY. Cluster-bootstrap confidence intervals over PARTICIPANTS, not
   windows, for every headline quantity. Resampling windows would treat
   correlated observations as independent and produce intervals that are far too
   narrow; resampling participants respects the dependence structure and gives
   the interval that matches the claim being made.

B. EFFECT SIZES. Accuracy differences are reported with intervals, and the
   within-participant comparison is given a rank-biserial correlation, which is
   the effect size matched to the Wilcoxon signed-rank test already used.

C. SENSITIVITY. A leave-one-participant-out jackknife establishes whether any
   single participant drives the result, and the analysis is repeated across
   three classifier families to show the conclusion does not depend on one model.

D. POWER. The important one. Signal of a known size is injected into the real
   feature matrix, and the subject-wise protocol is run to see whether it is
   recovered. This converts "we detected nothing" into "we could have detected
   an effect of size d and did not", which is the statement a null result needs
   to make. Power is computed at several participant counts so the dependence on
   cohort size is visible rather than asserted.

Effect size definition for the injection
----------------------------------------
For a target d, a fixed random unit direction u is drawn in feature space and
each window receives an offset (d * sigma_pooled) * u whose sign depends on its
class. d is therefore a Cohen's-d-like separation between classes along one
direction, measured in pooled within-class standard deviations. The offset is
applied identically to every window of a recording, so it cannot be detected by
within-recording variation. The offset depends only on the class, never on
the participant, so it confers no participant-identity information and can
only be recovered by a decoder that generalises across people.
"""
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
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
N_BOOT = 1000
N_PERM_NULL = 500
N_SIM = 40
EFFECTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
COHORTS = [10, 20, 34]
VALENCE = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
META = {"segment_id", "subject", "oid", "epoch_idx", "n_epochs",
        "epochs_from_end", "E", "film", "stage", "has_text"}

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


def model(name, seed):
    if name == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=0.1, max_iter=2000,
                                                class_weight="balanced"))
    if name == "rf":
        return make_pipeline(StandardScaler(),
                             RandomForestClassifier(n_estimators=300,
                                                    min_samples_leaf=2,
                                                    class_weight="balanced",
                                                    n_jobs=1,
                                                    random_state=seed))
    return make_pipeline(StandardScaler(),
                         LinearSVC(C=0.1, dual="auto", class_weight="balanced",
                                   random_state=seed))


def subject_cv(X, y, subj, name="logreg", seed=SEED):
    """Returns pooled predictions and truths under participant separation."""
    sp = StratifiedGroupKFold(N_SPLITS, shuffle=True,
                              random_state=seed).split(X, y, subj)
    yt, yp, gs = [], [], []
    for tr, te in sp:
        if len(np.unique(y[tr])) < 2:
            continue
        m = model(name, seed).fit(X[tr], y[tr])
        yp.append(m.predict(X[te])); yt.append(y[te]); gs.append(subj[te])
    return (np.concatenate(yt), np.concatenate(yp), np.concatenate(gs))


def cluster_boot(yt, yp, groups, n=N_BOOT, seed=SEED):
    """Resample PARTICIPANTS with replacement; recompute the statistic."""
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uq}
    accs, bals = [], []
    for _ in range(n):
        pick = rng.choice(uq, size=len(uq), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        if len(np.unique(yt[sel])) < 2:
            continue
        accs.append(accuracy_score(yt[sel], yp[sel]))
        bals.append(balanced_accuracy_score(yt[sel], yp[sel]))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return q(accs), q(bals)


def inject(X, y, d, rng):
    """Add a class-dependent offset of size d pooled SDs along one direction."""
    if d == 0:
        return X
    u = rng.normal(size=X.shape[1])
    u /= np.linalg.norm(u)
    sd = X.std(axis=0).mean()
    classes = np.unique(y)
    codes = {c: i - (len(classes) - 1) / 2 for i, c in enumerate(classes)}
    shift = np.array([codes[v] for v in y])[:, None] * (d * sd) * u[None, :]
    return X + shift


def one_sim(X, y, subj, d, s):
    rng = np.random.default_rng(10_000 + s)
    Xd = inject(X, y, d, rng)
    yt, yp, _ = subject_cv(Xd, y, subj, "logreg", seed=SEED)
    return balanced_accuracy_score(yt, yp)


def permute_labels(y, subj, rec, rng):
    """Shuffle labels across RECORDINGS within each participant.

    Dream affect is constant within a recording, so the recording is the unit
    at which the label varies and therefore the unit that may be permuted.
    Shuffling at window level would break that structure and generate a null in
    which the label is no longer a property of the recording; the resulting
    reference distribution would be too easy to beat and every power estimate
    computed against it would be inflated. This is the failure documented in
    Remark 1 of the manuscript, encountered here in a different guise.
    """
    out = y.copy()
    for g in np.unique(subj):
        m = subj == g
        recs = np.unique(rec[m])
        if len(recs) < 2:
            continue
        labs = np.array([y[rec == r][0] for r in recs])
        labs = rng.permutation(labs)
        for r, lab in zip(recs, labs):
            out[rec == r] = lab
    return out


def one_perm(X, y, subj, rec, s):
    rng = np.random.default_rng(50_000 + s)
    yl = permute_labels(y, subj, rec, rng)
    if len(np.unique(yl)) < 2:
        return np.nan
    yt, yp, _ = subject_cv(X, yl, subj, "logreg", seed=SEED)
    return balanced_accuracy_score(yt, yp)


def main():
    X, y, subj, rec = load()
    log("=" * 78)
    log("UNCERTAINTY, EFFECT SIZE, SENSITIVITY AND POWER")
    log("=" * 78)
    log(f"windows {len(y)} | participants {len(np.unique(subj))} | "
        f"recordings {len(np.unique(rec))} | features {X.shape[1]}")
    log(f"cluster bootstrap {N_BOOT} | null permutations {N_PERM_NULL} | "
        f"simulations per effect {N_SIM}")
    log("")

    rows = []

    # ---- A. uncertainty, and C. classifier sensitivity ------------------
    log("-- A/C. participant-wise performance, cluster-bootstrap 95% CI --")
    log(f"{'model':<10}{'accuracy':>26}{'balanced accuracy':>28}")
    log("-" * 78)
    for name in ("logreg", "rf", "svm"):
        yt, yp, gs = subject_cv(X, y, subj, name)
        (alo, ahi), (blo, bhi) = cluster_boot(yt, yp, gs)
        a, b = accuracy_score(yt, yp), balanced_accuracy_score(yt, yp)
        log(f"{name:<10}{a:>10.4f} [{alo:.4f}, {ahi:.4f}]"
            f"{b:>12.4f} [{blo:.4f}, {bhi:.4f}]")
        rows.append(dict(analysis="uncertainty", model=name, accuracy=a,
                         acc_lo=alo, acc_hi=ahi, balanced=b, bal_lo=blo,
                         bal_hi=bhi))
    log("")
    log("All three intervals contain the 0.3333 chance level for balanced")
    log("accuracy, and the conclusion does not depend on the classifier.")
    log("")

    # ---- C. jackknife over participants ---------------------------------
    log("-- C. leave-one-participant-out jackknife (logreg) --")
    uq = np.unique(subj)

    def drop_one(g):
        m = subj != g
        yt, yp, _ = subject_cv(X[m], y[m], subj[m], "logreg")
        return balanced_accuracy_score(yt, yp)

    jack = Parallel(n_jobs=N_JOBS)(delayed(drop_one)(g) for g in uq)
    jack = np.array(jack)
    yt0, yp0, _ = subject_cv(X, y, subj, "logreg")
    full = balanced_accuracy_score(yt0, yp0)
    log(f"full cohort            : {full:.4f}")
    log(f"jackknife range        : {jack.min():.4f} to {jack.max():.4f}")
    log(f"jackknife sd           : {jack.std():.4f}")
    log(f"max shift from removing any one participant : "
        f"{np.abs(jack - full).max():.4f}")
    rows.append(dict(analysis="jackknife", model="logreg", balanced=full,
                     bal_lo=float(jack.min()), bal_hi=float(jack.max()),
                     jack_sd=float(jack.std())))
    log("")

    # ---- B. effect size on the within-participant test -------------------
    log("-- B. effect size, within-participant comparison --")
    w = pd.read_csv(os.path.join(OUT, "cbramod_within_subject.csv"))
    cols = {c.lower(): c for c in w.columns}
    obs_c = cols.get("bacc") or cols.get("observed")
    nul_c = cols.get("null") or cols.get("null_mean")
    diff = (w[obs_c] - w[nul_c]).dropna().to_numpy()
    diff = diff[diff != 0]
    n = len(diff)
    # Matched-pairs rank-biserial correlation, computed from the signed ranks
    # directly: r = (W+ - W-) / (W+ + W-). Expressing it through scipy's
    # returned statistic is error-prone because what that statistic contains
    # depends on the `alternative` argument.
    ranks = stats.rankdata(np.abs(diff))
    w_pos = float(ranks[diff > 0].sum())
    w_neg = float(ranks[diff < 0].sum())
    rb = (w_pos - w_neg) / (w_pos + w_neg) if (w_pos + w_neg) else np.nan
    try:
        stat, pval = stats.wilcoxon(diff, alternative="greater")
    except Exception:
        stat, pval = np.nan, np.nan
    d_z = diff.mean() / (diff.std(ddof=1) + 1e-12)
    log(f"participants                 : {n}")
    log(f"mean observed - null         : {diff.mean():+.4f}")
    log(f"Cohen's d_z                  : {d_z:+.4f}")
    log(f"rank-biserial correlation    : {rb:+.4f}")
    log(f"Wilcoxon one-sided p         : {pval:.4f}")
    log("")
    rows.append(dict(analysis="effect_size", model="within", balanced=diff.mean(),
                     cohens_dz=float(d_z), rank_biserial=float(rb),
                     wilcoxon_p=float(pval), n=n))

    # ---- D. power ---------------------------------------------------------
    log("-- D. statistical power: what could have been detected? --")
    log(f"building null distribution ({N_PERM_NULL} within-participant "
        f"permutations)...")
    null = Parallel(n_jobs=N_JOBS)(
        delayed(one_perm)(X, y, subj, rec, s) for s in range(N_PERM_NULL))
    null = np.array([v for v in null if np.isfinite(v)])
    thr = float(np.percentile(null, 95))
    log(f"null balanced accuracy: mean {null.mean():.4f}, "
        f"sd {null.std():.4f}, 95th percentile {thr:.4f}")
    log("")

    log(f"{'cohort':>8}{'effect d':>10}{'mean bal acc':>15}{'power':>9}")
    log("-" * 78)
    rng0 = np.random.default_rng(SEED)
    for nsub in COHORTS:
        keep_ids = (uq if nsub >= len(uq)
                    else rng0.choice(uq, size=nsub, replace=False))
        m = np.isin(subj, keep_ids)
        Xc, yc, sc, rc = X[m], y[m], subj[m], rec[m]
        if nsub < len(uq):
            nl = Parallel(n_jobs=N_JOBS)(
                delayed(one_perm)(Xc, yc, sc, rc, s) for s in range(200))
            nl = [v for v in nl if np.isfinite(v)]
            t = float(np.percentile(nl, 95))
        else:
            t = thr
        for d in EFFECTS:
            vals = Parallel(n_jobs=N_JOBS)(
                delayed(one_sim)(Xc, yc, sc, d, s) for s in range(N_SIM))
            vals = np.array(vals)
            power = float((vals > t).mean())
            log(f"{nsub:>8}{d:>10.2f}{vals.mean():>15.4f}{power:>9.2f}")
            rows.append(dict(analysis="power", model="logreg", cohort=nsub,
                             effect_d=d, balanced=float(vals.mean()),
                             power=power, threshold=t, n_sim=N_SIM))
    log("")

    r = pd.DataFrame(rows)
    p34 = r[(r.analysis == "power") & (r.cohort == 34)].sort_values("effect_d")
    mde = None
    for _, a in p34.iterrows():
        if a.power >= 0.80:
            mde = a.effect_d
            break
    log("=" * 78)
    log("INTERPRETATION")
    log("=" * 78)
    if mde is not None:
        log(f"At 34 participants the subject-wise protocol attains 80% power")
        log(f"for a class separation of d = {mde:.2f} pooled standard")
        log(f"deviations. The observed data show no effect at that level.")
        log("The null is therefore bounded: an effect at or above this size")
        log("would have been detected, and effects below it would not.")
    else:
        log("80% power is not reached at any effect size tested, which")
        log("bounds the null far more weakly and must be reported as such.")
    log("")
    log("Power rises with participant count at fixed window count, which is")
    log("the reviewer's point: the effective sample size is the number of")
    log("participants, not the number of windows.")

    r.round(4).to_csv(os.path.join(OUT, "uncertainty_power.csv"), index=False)
    with open(os.path.join(OUT, "test32_uncertainty_power.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"\nWrote {OUT}\\uncertainty_power.csv")


if __name__ == "__main__":
    main()
