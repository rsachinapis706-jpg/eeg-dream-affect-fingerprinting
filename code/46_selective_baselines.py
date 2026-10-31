"""
What does PCD add over established selective-prediction methods? (R2 C1, C8)

Reviewer 2 is right that PCD's components are established, and the manuscript
already concedes this. A concession is weak evidence, though. This script
replaces it with a measurement: four methods for declining to predict are run on
identical folds, identical models and identical data, across conditions whose
ground truth is known, and what each one does when the task carries no
information is recorded rather than argued.

Methods compared
----------------
  PCD (proposed)        abstain when the observed margin does not exceed the
                        permutation null at level alpha
  Max-softmax           abstain when the top class probability is below a fixed
                        threshold -- the standard selective-classification rule
  Predictive entropy    abstain when the entropy of the predicted distribution
                        exceeds a fixed threshold
  Split conformal       construct a prediction SET with guaranteed marginal
                        coverage 1-alpha; report the mean set size and the
                        fraction of inputs receiving a singleton set

Conditions
----------
  Sleep-EDF staging     signal present and strong (verified positive control)
  DEED dream affect     signal weak at best (0.3783 balanced accuracy for the
                        best configuration, 0.3224 for the manuscript's)

The claim under test is narrow and falsifiable. PCD should retain coverage where
signal exists and collapse toward alpha where it does not. The confidence-based
rules should retain substantial coverage in BOTH, because a softmax is
unconstrained by whether the mapping it approximates exists. Conformal should
remain valid in both by construction, achieving that validity through wider sets
rather than by signalling absence -- so its coverage guarantee is met while its
set size, not its coverage, carries the information.

If PCD does not separate the conditions more sharply than the alternatives, the
method adds nothing and that should be reported.
"""
import os
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")

SEED = 20260727
N_SPLITS = 5
B_NULL = 30
ALPHA = 0.05
MSP_THR = 0.50
ENT_THR = 0.80             # fixed normalised-entropy threshold
# A threshold set from a quantile of the test distribution is degenerate:
# thresholding at the median keeps exactly half the inputs whatever the
# data contain, so it cannot distinguish the conditions by construction.
# A FIXED threshold is the honest analogue of the fixed max-softmax rule.
N_JOBS = 20

VALENCE = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
META = {"segment_id", "subject", "oid", "epoch_idx", "n_epochs",
        "epochs_from_end", "E", "film", "stage", "has_text"}

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def clf():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced"))


def margin(p):
    s = np.sort(p, axis=1)
    return s[:, -1] - s[:, -2]


def entropy(p):
    q = np.clip(p, 1e-12, 1.0)
    return -(q * np.log(q)).sum(axis=1) / np.log(p.shape[1])


def fit_null(Xtr, ytr, grp, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"g": grp, "y": ytr}).drop_duplicates("g")
    mp = dict(zip(df.g.to_numpy(), rng.permutation(df.y.to_numpy())))
    yp = pd.Series(grp).map(mp).to_numpy()
    if len(np.unique(yp)) < 2:
        return None
    return clf().fit(Xtr, yp)


def conformal_sets(model, Xcal, ycal, Xte, alpha):
    """Split conformal with the softmax nonconformity score 1 - p(true class).

    Returns the mean prediction-set size and the fraction of singleton sets.
    Coverage is guaranteed by construction, so the informative quantities are
    how large the sets have to be to achieve it.
    """
    pc = model.predict_proba(Xcal)
    classes = list(model.classes_)
    idx = np.array([classes.index(v) for v in ycal])
    scores = 1.0 - pc[np.arange(len(ycal)), idx]
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    qhat = np.sort(scores)[min(k, n) - 1]
    pt = model.predict_proba(Xte)
    inset = (1.0 - pt) <= qhat
    sizes = inset.sum(axis=1)
    return float(sizes.mean()), float((sizes == 1).mean()), qhat


def run(X, y, subj, grp, name):
    """One condition. grp is the permutation unit (recording or window)."""
    sp = StratifiedGroupKFold(N_SPLITS, shuffle=True,
                              random_state=SEED).split(X, y, subj)
    D_all, msp_all, ent_all, ok_all = [], [], [], []
    setsz, singles = [], []
    rng = np.random.default_rng(SEED)

    for tr, te in sp:
        if len(np.unique(y[tr])) < 2:
            continue
        # split the training fold into a fitting part and a conformal
        # calibration part; participants are not split across the two
        gtr = subj[tr]
        uq = np.unique(gtr)
        cal_ids = rng.choice(uq, size=max(1, len(uq) // 4), replace=False)
        cal = np.isin(gtr, cal_ids)
        fit = ~cal
        if len(np.unique(y[tr][fit])) < 2 or cal.sum() == 0:
            fit = np.ones(len(tr), bool); cal = fit

        model = clf().fit(X[tr][fit], y[tr][fit])
        pt = model.predict_proba(X[te])
        pred = model.classes_[pt.argmax(axis=1)]

        nulls = Parallel(n_jobs=N_JOBS)(
            delayed(fit_null)(X[tr][fit], y[tr][fit], grp[tr][fit],
                              SEED + 7919 * b) for b in range(B_NULL))
        nulls = [m for m in nulls if m is not None]
        S = np.concatenate([margin(m.predict_proba(X[te])) for m in nulls])
        D = np.searchsorted(np.sort(S), margin(pt), side="left") / len(S)

        ms, sg, _ = conformal_sets(model, X[tr][cal], y[tr][cal], X[te], ALPHA)
        setsz.append(ms); singles.append(sg)

        D_all.append(D)
        msp_all.append(pt.max(axis=1))
        ent_all.append(entropy(pt))
        ok_all.append(pred == y[te])

    D = np.concatenate(D_all); msp = np.concatenate(msp_all)
    ent = np.concatenate(ent_all); ok = np.concatenate(ok_all)

    rows = []
    for rule, keep in [
            (f"PCD (alpha={ALPHA})", D >= 1 - ALPHA),
            (f"Max-softmax > {MSP_THR}", msp > MSP_THR),
            (f"Entropy < {ENT_THR}", ent < ENT_THR)]:
        cov = float(keep.mean())
        acc = float(ok[keep].mean()) if keep.sum() else float("nan")
        rows.append(dict(condition=name, rule=rule, coverage=cov,
                         accuracy_on_kept=acc))
        log(f"  {rule:<24} coverage {cov:>7.4f}   accuracy on kept "
            f"{acc:>7.4f}")

    ms, sg = float(np.mean(setsz)), float(np.mean(singles))
    rows.append(dict(condition=name, rule=f"Split conformal (alpha={ALPHA})",
                     coverage=sg, accuracy_on_kept=np.nan,
                     mean_set_size=ms, singleton_rate=sg))
    log(f"  {'Split conformal':<24} mean set size {ms:>5.3f} of "
        f"{len(np.unique(y))}   singleton rate {sg:.4f}")
    return rows


def main():
    log("=" * 78)
    log("PCD AGAINST ESTABLISHED SELECTIVE-PREDICTION METHODS")
    log("=" * 78)
    log(f"identical folds, model and data for every rule; alpha = {ALPHA}")
    log("")

    rows = []

    d = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    fc = [c for c in d.columns if c not in META]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = d[d.E.isin(VALENCE)]
    log("DEED dream affect  (weak signal at best; label constant per recording)")
    rows += run(d[fc].to_numpy(float), d.E.map(VALENCE).to_numpy(),
                d.subject.to_numpy(), d.segment_id.to_numpy(),
                "DEED dream affect")
    log("")

    sp = os.path.join(OUT, "sleepedf_expanded_features.parquet")
    s = pd.read_parquet(sp)
    sfc = [c for c in s.columns if c.startswith("ch")]
    s = s.replace([np.inf, -np.inf], np.nan).dropna(subset=sfc)
    s = s[s.stage.isin(["W", "N1", "N2"])]
    # subsample for tractability; participants are kept whole
    keep = np.random.default_rng(SEED).choice(s.subject.unique(),
                                              size=40, replace=False)
    s = s[s.subject.isin(keep)]
    s = s.groupby("subject", group_keys=False).head(1500)
    log(f"Sleep-EDF staging  (signal present; label varies per window) "
        f"[{len(s)} windows, {s.subject.nunique()} participants]")
    rows += run(s[sfc].to_numpy(float), s.stage.to_numpy(),
                s.subject.to_numpy(), np.arange(len(s)).astype(str),
                "Sleep-EDF staging")
    log("")

    r = pd.DataFrame(rows)
    r.round(4).to_csv(os.path.join(OUT, "selective_baselines.csv"), index=False)

    log("=" * 78)
    log("SEPARATION BETWEEN CONDITIONS")
    log("=" * 78)
    log(f"{'rule':<28}{'signal':>10}{'no/weak signal':>17}{'ratio':>9}")
    log("-" * 78)
    for rule in r.rule.unique():
        a = r[(r.rule == rule) & (r.condition == "Sleep-EDF staging")]
        b = r[(r.rule == rule) & (r.condition == "DEED dream affect")]
        if not len(a) or not len(b):
            continue
        av, bv = float(a.coverage.iloc[0]), float(b.coverage.iloc[0])
        log(f"{rule:<28}{av:>10.4f}{bv:>17.4f}"
            f"{(av / bv if bv else float('inf')):>9.2f}")
    log("")
    log("For conformal the reported figure is the singleton rate, since its")
    log("coverage is fixed at 1-alpha by construction and cannot distinguish")
    log("the conditions; the set size is what changes.")

    with open(os.path.join(OUT, "test36_selective_baselines.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"\nWrote {OUT}\\selective_baselines.csv")


if __name__ == "__main__":
    main()
