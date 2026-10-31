"""
PCD strong-signal condition on the 100-subject Sleep-EDF Expanded cohort.

PCD's separation result currently anchors its strong condition on 8 subjects
(coverage 0.9662). That cohort also turned out to be optimistic: the spectral
control fell from 0.8831 to 0.7748 balanced accuracy when measured on 100
subjects. The strong condition must be re-established on the larger cohort before
the separation claim goes in the paper.

Compute note (stated because it is a deviation, not a detail)
------------------------------------------------------------
Full LOSO over 100 subjects with 30 null models per fold is ~3,000 fits on
183,775 x 108 data, which is hours of compute. Windows are therefore subsampled
to at most CAP_PER_SUBJECT per subject, stratified by stage so class proportions
are preserved. Subsampling reduces precision; it does not bias coverage, because
coverage is a per-window rate and every retained window is scored the same way.
The number of SUBJECTS -- the quantity under scrutiny -- is not reduced.

Reported alongside:
  * PCD coverage and accuracy with cluster-bootstrap CIs
  * max-softmax at fixed thresholds (what a practitioner would actually set)
  * leave-one-null-out calibration, i.e. the true false-positive rate
"""
import os
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
SEED = 20260727
B_NULL = 25
N_BOOT = 1000
CAP_PER_SUBJECT = 250

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def clf():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced"))


def margin(model, X):
    p = model.predict_proba(X)
    s = np.sort(p, axis=1)
    return s[:, -1] - s[:, -2]


def fit_null(Xtr, ytr, grp_tr, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"g": grp_tr, "y": ytr}).drop_duplicates("g")
    mp = dict(zip(df.g.to_numpy(), rng.permutation(df.y.to_numpy())))
    yp = pd.Series(grp_tr).map(mp).to_numpy()
    if len(np.unique(yp)) < 2:
        return None
    return clf().fit(Xtr, yp)


def boot_ci(yt, yp, keep, subj, stat):
    us = np.unique(subj)
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(us, size=len(us), replace=True)
        idx = np.concatenate([np.where(subj == u)[0] for u in pick])
        if stat == "cov":
            vals.append(keep[idx].mean())
        else:
            k = idx[keep[idx]]
            vals.append(accuracy_score(yt[k], yp[k]) if len(k) > 5 else np.nan)
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    return (np.nan, np.nan) if len(vals) < 50 else tuple(
        np.percentile(vals, [2.5, 97.5]))


def main():
    d = pd.read_parquet(os.path.join(OUT, "sleepedf_expanded_features.parquet"))
    fc = [c for c in d.columns if c.startswith("ch")]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = d[d.stage.isin(["W", "N1", "N2"])]

    # Stratified subsample per subject, done on the INDEX.
    # groupby(...).apply() drops the grouping column under pandas 3.x, which
    # silently removed 'stage' here; selecting indices avoids touching columns
    # at all.
    rng = np.random.default_rng(SEED)
    keep_idx = []
    for s, g in d.groupby("subject"):
        if len(g) <= CAP_PER_SUBJECT:
            keep_idx.append(g.index.to_numpy())
            continue
        frac = CAP_PER_SUBJECT / len(g)
        for st, gg in g.groupby("stage"):
            n = max(1, int(round(len(gg) * frac)))
            keep_idx.append(rng.choice(gg.index.to_numpy(), size=n,
                                       replace=False))
    d = d.loc[np.concatenate(keep_idx)]
    assert "stage" in d.columns and "subject" in d.columns

    X = d[fc].to_numpy(float)
    y = d.stage.to_numpy()
    subj = d.subject.to_numpy()
    grp = np.arange(len(d)).astype(str)     # stage varies per window

    log("=" * 76)
    log("PCD STRONG-SIGNAL CONDITION -- Sleep-EDF Expanded (100 subjects)")
    log("=" * 76)
    log(f"windows {len(y)} (capped at {CAP_PER_SUBJECT}/subject, "
        f"stage-stratified) | subjects {len(np.unique(subj))}")
    log(f"null models per fold: {B_NULL}")
    log(f"prior 8-subject result: coverage 0.9662 at alpha=0.05")
    log("")

    yt, yp, Ds, msps, ss, Dloo = [], [], [], [], [], []
    us = np.unique(subj)
    for i, s in enumerate(us):
        te = subj == s
        tr = ~te
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            continue
        model = clf().fit(X[tr], y[tr])
        s_obs = margin(model, X[te])
        nulls = Parallel(n_jobs=-1)(
            delayed(fit_null)(X[tr], y[tr], grp[tr], SEED + 7919 * b)
            for b in range(B_NULL))
        nulls = [m for m in nulls if m is not None]
        if len(nulls) < 5:
            continue
        M = np.stack([margin(m, X[te]) for m in nulls])
        pooled = np.sort(M.reshape(-1))
        Ds.append(np.searchsorted(pooled, s_obs, side="left") / len(pooled))
        rest = np.sort(M[1:].reshape(-1))
        Dloo.append(np.searchsorted(rest, M[0], side="left") / len(rest))
        yt.append(y[te]); yp.append(model.predict(X[te]))
        msps.append(model.predict_proba(X[te]).max(axis=1))
        ss.append(np.full(te.sum(), s))
        if (i + 1) % 20 == 0:
            log(f"  ...{i+1}/{len(us)} subjects")

    yt = np.concatenate(yt); yp = np.concatenate(yp)
    D = np.concatenate(Ds); msp = np.concatenate(msps)
    ss = np.concatenate(ss); Dloo = np.concatenate(Dloo)

    base = accuracy_score(yt, yp)
    log("")
    log(f"no-abstention accuracy {base:.4f} | "
        f"balanced {balanced_accuracy_score(yt, yp):.4f}")
    log("")

    rows = []
    log("-- PCD --")
    log(f"{'alpha':>7}{'coverage':>11}{'95% CI':>20}{'accuracy':>11}{'95% CI':>20}")
    for alpha in [0.20, 0.10, 0.05, 0.01]:
        keep = D >= (1 - alpha)
        cov = keep.mean()
        cl, ch = boot_ci(yt, yp, keep, ss, "cov")
        acc = accuracy_score(yt[keep], yp[keep]) if keep.sum() > 5 else np.nan
        al, ah = boot_ci(yt, yp, keep, ss, "acc")
        log(f"{alpha:>7.2f}{cov:>11.4f}   [{cl:.4f}, {ch:.4f}]"
            f"{acc:>11.4f}   [{al:.4f}, {ah:.4f}]")
        rows.append(dict(rule="PCD", param=alpha, coverage=cov, cov_lo=cl,
                         cov_hi=ch, accuracy=acc, acc_lo=al, acc_hi=ah))
    log("")

    log("-- max-softmax at fixed thresholds --")
    log(f"{'thresh':>7}{'coverage':>11}{'95% CI':>20}{'accuracy':>11}{'95% CI':>20}")
    for thr in [0.50, 0.70, 0.90]:
        keep = msp >= thr
        cov = keep.mean()
        cl, ch = boot_ci(yt, yp, keep, ss, "cov")
        acc = accuracy_score(yt[keep], yp[keep]) if keep.sum() > 5 else np.nan
        al, ah = boot_ci(yt, yp, keep, ss, "acc")
        log(f"{thr:>7.2f}{cov:>11.4f}   [{cl:.4f}, {ch:.4f}]"
            f"{acc:>11.4f}   [{al:.4f}, {ah:.4f}]")
        rows.append(dict(rule="MSP_fixed", param=thr, coverage=cov, cov_lo=cl,
                         cov_hi=ch, accuracy=acc, acc_lo=al, acc_hi=ah))
    log("")

    log("-- calibration --")
    log(f"mean D (observed)            : {D.mean():.4f}")
    log(f"mean D (leave-one-null-out)  : {Dloo.mean():.4f}   <- ~0.5 if calibrated")
    log(f"P(D>=0.95) observed          : {(D >= 0.95).mean():.4f}")
    log(f"P(D>=0.95) null-vs-null      : {(Dloo >= 0.95).mean():.4f}   "
        f"<- true false-positive rate")
    log(f"KS vs uniform (null-vs-null) : p={stats.kstest(Dloo,'uniform').pvalue:.3g}")
    log("")

    log("=" * 76)
    log("SEPARATION: 100-subject strong condition vs DEED dream affect")
    log("=" * 76)
    cov05 = float([r for r in rows if r["rule"] == "PCD"
                   and r["param"] == 0.05][0]["coverage"])
    log(f"{'condition':<34}{'subjects':>10}{'PCD coverage @0.05':>22}")
    log(f"{'Sleep-EDF Expanded (strong)':<34}{len(np.unique(ss)):>10}{cov05:>22.4f}")
    log(f"{'Sleep-EDF 1.0.0 (strong, prior)':<34}{8:>10}{0.9662:>22.4f}")
    log(f"{'DEED dream affect (absent)':<34}{34:>10}{0.1071:>22.4f}")
    log(f"{'DEED affect, CBraMod (absent)':<34}{34:>10}{0.0828:>22.4f}")
    log("")
    log(f"separation ratio (strong/absent): {cov05/0.1071:.2f}x")

    pd.DataFrame(rows).round(4).to_csv(
        os.path.join(OUT, "pcd_expanded.csv"), index=False)
    with open(os.path.join(OUT, "test19_pcd_expanded.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"Wrote {OUT}\\test19_pcd_expanded.txt")


if __name__ == "__main__":
    main()
