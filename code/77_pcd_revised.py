"""
Permutation-Calibrated Decodability, reformulated and re-evaluated
(Reviewer 1, comments 1 and 10; Reviewer 2 on scope).

1. The score (R1-1)
   The previous score D(x) was the share of pooled null margins strictly
   below the observed margin: a lower-tail percentile, undefined in meaning
   under ties, and pooled across test inputs, so no per-input guarantee
   follows. It is replaced by an upper-tail permutation p-value whose
   reference also contains the observed model's own margins:

       p(x_i) = #{ (b, j) : b = 0..B, j = 1..n,  s_b(x_j) >= s_0(x_i) }
                -------------------------------------------------------
                                  (B + 1) n

   with b = 0 the model fitted to the observed labels. Ties count against
   prediction. Predict when p(x_i) <= alpha, abstain otherwise. If the
   observed labelling is exchangeable with the permuted labellings (the null),
   the B + 1 fitted models are exchangeable, and for a test input drawn at
   random P(p <= alpha) <= alpha exactly, for any B. The guarantee is
   MARGINAL over inputs; it is not per input and not per participant, and it
   says nothing about whether a retained prediction is correct.

2. Permutation scheme (stated once, used throughout)
   Labels are permuted at the unit at which they vary -- recordings for dream
   affect, windows for sleep stage -- within participant, as Algorithm 1
   states. (46_selective_baselines.py permuted across the whole training fold;
   that is corrected here.)

3. What is measured (R1-10)
   * PCD, max-softmax, predictive entropy and split conformal on identical
     participant-separated folds, models (logistic regression, C = 0.1,
     class-balanced) and data, for Sleep-EDF staging and DEED affect.
     Conformal calibration participants are drawn from the training fold and
     are disjoint from both the fitting and the test participants.
   * Risk-coverage curves: balanced accuracy and error of retained
     predictions as the level (PCD) or threshold (others) varies, and a
     comparison at matched coverage.
   * Known-null calibration on real data: DEED with its recording labels
     re-randomised within participant, R times. Overall and per-participant
     false-positive rates, with the spread across randomisations.
   * Simulation study: clustered data (participant and recording random
     effects), class imbalance and heterogeneous per-input noise, under the
     null. Marginal false-positive rate, per-participant rates, and per-input
     heterogeneity -- the case Reviewer 1 raised, where the pooled reference
     is a mixture of input-specific null distributions.

Outputs in out/pcd_revised_*.csv and out/pcd_revised_windows.parquet.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
LOG = os.path.join(OUT, "run_pcd_revised.log")

SEED = 20260727
N_SPLITS = 5
B = 100                       # null models per fold
ALPHA = 0.05
MSP_THR, ENT_THR = 0.50, 0.90
R_NULL = 20                   # re-randomisations of DEED labels
N_SIM = 100                   # simulated datasets
N_JOBS = 12
VALENCE = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
META = {"segment_id", "subject", "oid", "epoch_idx", "n_epochs",
        "epochs_from_end", "E", "film", "stage", "has_text"}

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=1000, C=0.1, class_weight="balanced"))


def margin(p):
    s = np.sort(p, axis=1)
    return s[:, -1] - s[:, -2]


def entropy(p):
    q = np.clip(p, 1e-12, 1.0)
    return -(q * np.log(q)).sum(axis=1) / np.log(p.shape[1])


def pcd_p(s_obs, s_null):
    """Upper-tail p-value with the observed model in the reference.
    s_obs: (n,) observed margins on the test inputs; s_null: (B, n)."""
    ref = np.sort(np.concatenate([s_null.ravel(), s_obs]))
    n_ge = len(ref) - np.searchsorted(ref, s_obs, side="left")
    return n_ge / len(ref)


def permute_within(y, unit, subj, rng):
    """Permute unit-level labels among the units of each participant."""
    df = pd.DataFrame(dict(u=unit, s=subj, y=y)).drop_duplicates("u")
    mp = {}
    for _, g in df.groupby("s"):
        mp.update(zip(g.u.to_numpy(), rng.permutation(g.y.to_numpy())))
    return np.array([mp[k] for k in unit])


def null_margins(Xtr, ytr, utr, str_, Xte, seed, b_count):
    out = []
    for b in range(b_count):
        rng = np.random.default_rng(seed + 7919 * b)
        yb = permute_within(ytr, utr, str_, rng)
        if len(np.unique(yb)) < 2:
            continue
        out.append(margin(clf().fit(Xtr, yb).predict_proba(Xte)))
    return np.array(out)


def conformal(model, Xcal, ycal, Xte, yte, alpha):
    pc = model.predict_proba(Xcal)
    cls = list(model.classes_)
    sc = 1.0 - pc[np.arange(len(ycal)), [cls.index(v) for v in ycal]]
    k = int(np.ceil((len(sc) + 1) * (1 - alpha)))
    qhat = np.sort(sc)[min(k, len(sc)) - 1]
    inset = (1.0 - model.predict_proba(Xte)) <= qhat
    covered = inset[np.arange(len(yte)), [cls.index(v) for v in yte]]
    return inset.sum(axis=1), covered


def label_free_folds(subj, seed, n_splits=N_SPLITS):
    """Participant-separated folds drawn without reference to any labels:
    participants shuffled with the given seed and split into n_splits groups
    (the participant partition of 116_pcd_leaky.py). Round 3 (Reviewer 2,
    comment 1): the folds previously came from StratifiedGroupKFold on the
    observed labels, so they depended on the observed labelling and were not
    recomputed for the permuted ones, against the condition of Propositions 1
    and 2. Folds that use no labels satisfy it for every labelling."""
    rng = np.random.default_rng(seed)
    people = rng.permutation(np.unique(subj))
    for g in np.array_split(people, n_splits):
        te = np.flatnonzero(np.isin(subj, g))
        yield np.setdiff1d(np.arange(len(subj)), te), te


def run_condition(X, y, subj, unit, name, b_count=B, keep_windows=True,
                  seed=SEED, n_jobs=N_JOBS):
    """Participant-separated folds drawn without labels; every method on the
    same fitted model."""
    sp = label_free_folds(subj, seed)
    rng = np.random.default_rng(seed)
    rows = []
    for k, (tr, te) in enumerate(sp):
        if len(np.unique(y[tr])) < 2:
            continue
        uq = np.unique(subj[tr])
        cal_ids = rng.choice(uq, size=max(1, len(uq) // 4), replace=False)
        cal = np.isin(subj[tr], cal_ids)
        fit = tr[~cal]
        calx = tr[cal]
        model = clf().fit(X[fit], y[fit])
        pt = model.predict_proba(X[te])
        pred = model.classes_[pt.argmax(1)]
        chunks = np.array_split(np.arange(b_count), max(1, n_jobs))
        parts = Parallel(n_jobs=n_jobs)(
            delayed(null_margins)(X[fit], y[fit], unit[fit], subj[fit], X[te],
                                  seed + 1000 * k + int(c[0]) * 7919, len(c))
            for c in chunks if len(c))
        S = np.vstack([p for p in parts if len(p)])
        p = pcd_p(margin(pt), S)
        sz, covd = conformal(model, X[calx], y[calx], X[te], y[te], ALPHA)
        rows.append(pd.DataFrame(dict(
            condition=name, fold=k, idx=te, subj=subj[te], y=y[te], pred=pred,
            p_pcd=p, msp=pt.max(1), ent=entropy(pt), margin=margin(pt),
            conf_size=sz, conf_covered=covd, n_null=len(S))))
    w = pd.concat(rows, ignore_index=True)
    return w


def summarise(w):
    out = []
    for name, g in w.groupby("condition"):
        ok = g.y == g.pred
        for rule, keep in ((f"PCD p<={ALPHA}", g.p_pcd <= ALPHA),
                           (f"Max-softmax>{MSP_THR}", g.msp > MSP_THR),
                           (f"Entropy<{ENT_THR}", g.ent < ENT_THR)):
            kk = keep.to_numpy()
            out.append(dict(condition=name, rule=rule, coverage=kk.mean(),
                            bal_retained=(balanced_accuracy_score(
                                g.y[kk], g.pred[kk]) if kk.sum() > 10
                                else np.nan),
                            err_retained=(1 - ok[kk].mean()) if kk.sum()
                            else np.nan))
        out.append(dict(condition=name, rule=f"Split conformal {ALPHA}",
                        coverage=(g.conf_size == 1).mean(),
                        set_size=g.conf_size.mean(),
                        set_coverage=g.conf_covered.mean()))
        out.append(dict(condition=name, rule="all inputs", coverage=1.0,
                        bal_retained=balanced_accuracy_score(g.y, g.pred),
                        err_retained=1 - ok.mean()))
    return pd.DataFrame(out)


def risk_coverage(w):
    rows = []
    for name, g in w.groupby("condition"):
        y, pr = g.y.to_numpy(), g.pred.to_numpy()
        scores = {"PCD": -g.p_pcd.to_numpy(), "Max-softmax": g.msp.to_numpy(),
                  "Entropy": -g.ent.to_numpy()}
        for rule, sc in scores.items():
            order = np.argsort(-sc, kind="stable")
            for cov in np.round(np.arange(0.05, 1.0001, 0.05), 2):
                k = max(10, int(round(cov * len(y))))
                sel = order[:k]
                rows.append(dict(condition=name, rule=rule, coverage=cov,
                                 bal_retained=balanced_accuracy_score(
                                     y[sel], pr[sel]),
                                 err_retained=float((y[sel] != pr[sel]).mean())))
    return pd.DataFrame(rows)


def participant_fpr(g):
    return g.groupby("subj").apply(lambda h: (h.p_pcd <= ALPHA).mean())


def simulate(i):
    """One dataset under the null: clustered, imbalanced, heterogeneous."""
    rng = np.random.default_rng(5_000_000 + i)
    P, d = 30, 20
    X, y, subj, unit = [], [], [], []
    for s in range(P):
        mu_s = rng.normal(0, 1.0, d)                        # participant effect
        pri = rng.dirichlet([0.6, 0.6, 0.6])                # composition
        for r in range(rng.integers(4, 16)):
            mu_r = mu_s + rng.normal(0, 0.5, d)             # recording effect
            lab = rng.choice(3, p=pri)                      # independent of X
            n = rng.integers(3, 25)
            scale = rng.lognormal(0, 0.6)                   # per-input noise
            X.append(mu_r + rng.normal(0, scale, (n, d)))
            y += [lab] * n
            subj += [s] * n
            unit += [f"{s}_{r}"] * n
    X = np.vstack(X)
    w = run_condition(X, np.array(y), np.array(subj), np.array(unit),
                      f"sim{i}", b_count=50, seed=6_000_000 + i, n_jobs=1)
    pf = participant_fpr(w)
    return dict(sim=i, n=len(w), fpr=(w.p_pcd <= ALPHA).mean(),
                part_fpr_mean=pf.mean(), part_fpr_p90=pf.quantile(.9),
                part_frac_above_2alpha=(pf > 2 * ALPHA).mean(),
                p_uniform_ks=float(np.abs(np.sort(w.p_pcd) - np.linspace(
                    0, 1, len(w))).max()))


def main():
    smoke = "--smoke" in sys.argv
    global B, R_NULL, N_SIM
    if smoke:
        B, R_NULL, N_SIM = 10, 2, 2
    tag = "_smoke" if smoke else ""
    t0 = time.time()

    d = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    fc = [c for c in d.columns if c not in META]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = d[d.E.isin(VALENCE)].reset_index(drop=True)
    Xd, yd = d[fc].to_numpy(float), d.E.map(VALENCE).to_numpy()
    sd, ud = d.subject.astype(str).to_numpy(), d.segment_id.to_numpy()

    s = pd.read_parquet(os.path.join(OUT, "sleepedf_expanded_features.parquet"))
    # Spectral features only, selected by prefix as in every published script.
    # The table also holds `epoch`, the window's position in the night, which
    # predicts sleep stage by itself and must never enter as a feature.
    sfc = [c for c in s.columns if c.startswith("ch")]
    assert "epoch" not in sfc and len(sfc) == 36, sfc
    s = s.replace([np.inf, -np.inf], np.nan).dropna(subset=sfc)
    s = s[s.stage.isin(["W", "N1", "N2"])].reset_index(drop=True)
    if smoke:
        s = s[s.subject.isin(sorted(s.subject.unique())[:12])]
        s = s.reset_index(drop=True)
    Xs, ys = s[sfc].to_numpy(float), s.stage.to_numpy()
    ss = s.subject.astype(str).to_numpy()
    us = np.arange(len(ys)).astype(str)          # sleep stage varies by window

    log("=" * 78)
    log("PCD REVISED: upper-tail p-value, stated permutation scheme")
    log("=" * 78)
    log(f"DEED {len(yd)} windows | Sleep-EDF {len(ys)} windows, "
        f"{len(sfc)} features | B = {B} null models per fold")

    w = pd.concat([run_condition(Xs, ys, ss, us, "Sleep-EDF staging"),
                   run_condition(Xd, yd, sd, ud, "DEED affect")])
    w.to_parquet(os.path.join(OUT, f"pcd_revised_windows{tag}.parquet"))
    sm = summarise(w)
    rc = risk_coverage(w)
    sm.round(4).to_csv(os.path.join(OUT, f"pcd_revised_summary{tag}.csv"),
                       index=False)
    rc.round(4).to_csv(os.path.join(OUT, f"pcd_revised_riskcov{tag}.csv"),
                       index=False)
    log("")
    with pd.option_context("display.width", 200):
        log(sm.round(4).to_string(index=False))
    log(f"  [{(time.time() - t0) / 60:.1f} min]")

    # matched coverage: other rules at PCD's coverage
    log("")
    log("-- at the coverage PCD attains at alpha --")
    for name, g in w.groupby("condition"):
        cov = (g.p_pcd <= ALPHA).mean()
        r = rc[(rc.condition == name)]
        near = r.iloc[(r.coverage - cov).abs().argsort()[:3]]
        log(f"  {name}: PCD coverage {cov:.4f}; nearest grid point "
            f"{near.coverage.iloc[0]:.2f}:")
        for _, x in r[r.coverage == near.coverage.iloc[0]].iterrows():
            log(f"     {x.rule:<12} retained bal {x.bal_retained:.4f} "
                f"err {x.err_retained:.4f}")

    # known null on real data
    nulls = []
    for i in range(R_NULL):
        rng = np.random.default_rng(4_000_000 + i)
        yr = permute_within(yd, ud, sd, rng)
        g = run_condition(Xd, yr, sd, ud, f"DEED null {i}", seed=SEED + i)
        pf = participant_fpr(g)
        nulls.append(dict(rep=i, fpr=(g.p_pcd <= ALPHA).mean(),
                          part_fpr_mean=pf.mean(),
                          part_fpr_max=pf.max(),
                          part_frac_above_2alpha=(pf > 2 * ALPHA).mean()))
    nl = pd.DataFrame(nulls)
    nl.to_csv(os.path.join(OUT, f"pcd_revised_deednull{tag}.csv"), index=False)
    log("")
    log(f"-- DEED with labels re-randomised within participant, {R_NULL} "
        f"times (alpha {ALPHA}) --")
    log(f"  overall FPR mean {nl.fpr.mean():.4f}, range {nl.fpr.min():.4f}-"
        f"{nl.fpr.max():.4f}, 2.5-97.5% {nl.fpr.quantile(.025):.4f}-"
        f"{nl.fpr.quantile(.975):.4f}")
    log(f"  per-participant FPR: mean {nl.part_fpr_mean.mean():.4f}; worst "
        f"participant median over reps {nl.part_fpr_max.median():.4f}; share "
        f"of participants above {2 * ALPHA} {nl.part_frac_above_2alpha.mean():.3f}")
    log(f"  [{(time.time() - t0) / 60:.1f} min]")

    sims = pd.DataFrame(Parallel(n_jobs=N_JOBS)(
        delayed(simulate)(i) for i in range(N_SIM)))
    sims.to_csv(os.path.join(OUT, f"pcd_revised_sims{tag}.csv"), index=False)
    log("")
    log(f"-- simulation under the null, {N_SIM} datasets (clustered, "
        f"imbalanced, heteroscedastic) --")
    log(f"  marginal FPR mean {sims.fpr.mean():.4f} (2.5-97.5% "
        f"{sims.fpr.quantile(.025):.4f}-{sims.fpr.quantile(.975):.4f}); "
        f"share of datasets above alpha {(sims.fpr > ALPHA).mean():.3f}")
    log(f"  per-participant FPR 90th percentile, median over datasets "
        f"{sims.part_fpr_p90.median():.4f}; share of participants above "
        f"{2 * ALPHA}: {sims.part_frac_above_2alpha.mean():.3f}")
    log(f"  [{(time.time() - t0) / 60:.1f} min]")
    with open(LOG.replace(".log", f"{tag}.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
