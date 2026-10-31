"""
Stratified Permutation-Calibrated Decodability: calibration at a chosen unit.

Question
  77_pcd_revised.py ranks each observed margin against the null margins of
  ALL test inputs pooled. Its guarantee is marginal over inputs, and on DEED
  with re-randomised labels individual participants are flagged far more
  often than alpha. Does ranking each margin only against the null margins of
  the same participant (or recording, or input) give the guarantee at that
  level, and what does it cost in power?

Theory (Proposition 2 of the revision)
  Under H0 the B + 1 rows of the margin array M[b, j] = s_b(x_j) are
  exchangeable given the inputs. Restricting M to any fixed set J of test
  columns -- the windows of one participant, which are known without labels --
  keeps its rows exchangeable, so the proof of Proposition 1 applies within J:
  for I drawn uniformly from J, P[p_J(x_I) <= alpha] <= alpha. Hence the
  expected share of a participant's windows on which the rule predicts is at
  most alpha for EVERY participant. With J = {i} this is the classical
  per-input permutation p-value, (1 + #{b : s_b(x_i) >= s_0(x_i)}) / (B + 1),
  valid for every input but resolved only in steps of 1 / (B + 1).

Design
  Every fold, seed, calibration split and null model is that of 77, imported
  from it, so the pooled p-values must reproduce 77's exactly (checked). Only
  the reference against which each margin is ranked changes:
     pooled       all test inputs of the fold (77)
     recording    the inputs of the same recording
     participant  the inputs of the same participant
     input        the input alone
  Pieces (each resumable, run in this order):
     deed      DEED affect, real labels: coverage and retained balanced
               accuracy per stratum (the power cost)
     deednull  DEED with recording labels re-randomised within participant,
               reps 0-19 identical to 77, extended to R_NULL reps so that each
               participant's expected rate can be estimated
     sims      the 100 simulated null datasets of 77 (clustered, imbalanced,
               heteroscedastic), every stratum
     sleepedf  Sleep-EDF staging, real labels: power cost where signal is strong

Outputs (out/)
  pcd_strat_windows.parquet         per-window p-values, real-label conditions
  pcd_strat_summary.csv             coverage / retained bal. acc. per stratum
  pcd_strat_deednull.csv            per rep and stratum: marginal rate
  pcd_strat_deednull_participants.csv  per rep, stratum, participant: rate
  pcd_strat_deednull_summary.csv    per stratum: participants' expected rates
  pcd_strat_sims.csv                per simulated dataset and stratum
  run_pcd_strat.log
"""
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "pcd", os.path.join(HERE, "77_pcd_revised.py"))
pcd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pcd)
OUT = pcd.OUT
ALPHAS = (0.10, 0.05, 0.01)
ALPHA = pcd.ALPHA
STRATA = ("pooled", "recording", "participant", "input")
R_NULL = int(os.environ.get("PCD_R_NULL", 100))
N_JOBS = int(os.environ.get("PCD_JOBS", 4))
LOG = os.path.join(OUT, "run_pcd_strat.log")
T0 = time.time()


def log(m=""):
    m = f"[{(time.time() - T0) / 60:6.1f} min] {m}"
    print(m, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(m + "\n")


# 77 ran its null models in 12 loky workers, which loky limits to
# cpu_count // 12 = 2 linear-algebra threads each, and its observed model in
# the main process with the library default. The same thread counts are used
# here: logistic-regression solutions differ in the last digits with the
# thread count, and the pooled p-values must reproduce 77 exactly.
NULL_THREADS = max(1, (os.cpu_count() or 24) // 12)


def null_worker(Xtr, ytr, utr, str_, Xte, seed, count):
    with threadpool_limits(NULL_THREADS):
        return pcd.null_margins(Xtr, ytr, utr, str_, Xte, seed, count)


def strat_p(s_obs, S, groups):
    """pcd_p computed separately within each group of test inputs."""
    out = np.empty(len(s_obs))
    for g in np.unique(groups):
        J = groups == g
        out[J] = pcd.pcd_p(s_obs[J], S[:, J])
    return out


def run_strat(X, y, subj, unit, rec, name, b_count=pcd.B, seed=pcd.SEED,
              n_jobs=N_JOBS, fold_y=None):
    """77's run_condition, keeping the null margins to rank them by stratum.
    Folds, calibration split, observed model and null-model seeds are
    identical (each null model's seed is seed + 1000 k + 7919 b, whatever
    the chunking).

    Round 3: folds are drawn without labels (77.label_free_folds), so they
    are the same for every labelling and satisfy Propositions 1 and 2.
    fold_y is kept for the call signature and no longer used."""
    sp = pcd.label_free_folds(subj, seed)
    rng = np.random.default_rng(seed)
    rows = []
    for k, (tr, te) in enumerate(sp):
        if len(np.unique(y[tr])) < 2:
            continue
        uq = np.unique(subj[tr])
        cal_ids = rng.choice(uq, size=max(1, len(uq) // 4), replace=False)
        fit = tr[~np.isin(subj[tr], cal_ids)]
        model = pcd.clf().fit(X[fit], y[fit])
        pt = model.predict_proba(X[te])
        pred = model.classes_[pt.argmax(1)]
        chunks = [c for c in np.array_split(np.arange(b_count), max(1, n_jobs))
                  if len(c)]
        if n_jobs > 1:
            parts = Parallel(n_jobs=n_jobs)(
                delayed(null_worker)(X[fit], y[fit], unit[fit], subj[fit],
                                     X[te], seed + 1000 * k + int(c[0]) * 7919,
                                     len(c)) for c in chunks)
        else:
            parts = [null_worker(X[fit], y[fit], unit[fit], subj[fit], X[te],
                                 seed + 1000 * k + int(c[0]) * 7919, len(c))
                     for c in chunks]
        S = np.vstack([p for p in parts if len(p)])
        s0 = pcd.margin(pt)
        rows.append(pd.DataFrame(dict(
            condition=name, fold=k, idx=te, subj=subj[te], rec=rec[te],
            y=y[te], pred=pred, margin=s0, n_null=len(S),
            p_pooled=pcd.pcd_p(s0, S),
            p_recording=strat_p(s0, S, rec[te]),
            p_participant=strat_p(s0, S, subj[te]),
            p_input=(1 + (S >= s0[None, :]).sum(0)) / (len(S) + 1))))
    return pd.concat(rows, ignore_index=True)


def summary(w):
    out = []
    for st in STRATA:
        p = w[f"p_{st}"].to_numpy()
        for a in ALPHAS:
            keep = p <= a
            pc = pd.Series(keep).groupby(w.subj.to_numpy()).mean()
            out.append(dict(
                condition=w.condition.iloc[0], stratum=st, alpha=a,
                coverage=keep.mean(),
                bal_retained=(balanced_accuracy_score(w.y[keep], w.pred[keep])
                              if keep.sum() > 10 else np.nan),
                part_cov_min=pc.min(), part_cov_median=pc.median(),
                part_cov_max=pc.max()))
    return out


# ------------------------------------------------------------------ data
def deed():
    d = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    fc = [c for c in d.columns if c not in pcd.META]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = d[d.E.isin(pcd.VALENCE)].reset_index(drop=True)
    X, y = d[fc].to_numpy(float), d.E.map(pcd.VALENCE).to_numpy()
    s, u = d.subject.astype(str).to_numpy(), d.segment_id.to_numpy()
    return X, y, s, u


def sleepedf():
    s = pd.read_parquet(os.path.join(OUT, "sleepedf_expanded_features.parquet"))
    sfc = [c for c in s.columns if c.startswith("ch")]
    assert "epoch" not in sfc and len(sfc) == 36
    s = s.replace([np.inf, -np.inf], np.nan).dropna(subset=sfc)
    s = s[s.stage.isin(["W", "N1", "N2"])].reset_index(drop=True)
    X, y = s[sfc].to_numpy(float), s.stage.to_numpy()
    sj = s.subject.astype(str).to_numpy()
    u = np.arange(len(y)).astype(str)                  # label varies by window
    rec = s.recording.astype(str).to_numpy()
    return X, y, sj, u, rec


def check_against_77(w, name):
    ref = pd.read_parquet(os.path.join(OUT, "pcd_revised_windows.parquet"))
    ref = ref[ref.condition == name].set_index("idx").p_pcd
    got = w.set_index("idx").p_pooled.reindex(ref.index)
    d = float(np.abs(got - ref).max())
    c77, cnew = (ref <= ALPHA).mean(), (got <= ALPHA).mean()
    log(f"  check vs 77 ({name}): max |p diff| {d:.2e}; coverage 77 "
        f"{c77:.4f}, here {cnew:.4f}")
    assert round(c77, 4) == round(cnew, 4), "pooled PCD does not reproduce 77"


# ------------------------------------------------------------------ pieces
def piece_real(name, data, path_tag):
    f = os.path.join(OUT, f"pcd_strat_windows_{path_tag}.parquet")
    if os.path.exists(f):
        log(f"{name}: already done")
        return pd.read_parquet(f)
    log(f"{name}: running")
    if len(data) == 4:
        X, y, s, u = data
        w = run_strat(X, y, s, u, u, name)
    else:
        X, y, s, u, rec = data
        w = run_strat(X, y, s, u, rec, name)
    w.to_parquet(f)
    check_against_77(w, name)
    return w


def piece_deednull(X, y, s, u, fixed=False):
    """fixed=False: folds stratified on the re-randomised labels, as in 77.
    fixed=True: folds stratified on the real labels, so the partition does
    not depend on the labelling under test and the propositions' condition
    holds exactly; same re-randomisations, seeds and null models otherwise."""
    tag = "_fixedfolds" if fixed else ""
    f = os.path.join(OUT, f"pcd_strat_deednull{tag}_participants.csv")
    done = set(pd.read_csv(f).rep.unique()) if os.path.exists(f) else set()
    ref = pd.read_csv(os.path.join(OUT, "pcd_revised_deednull.csv"))
    for i in range(R_NULL):
        if i in done:
            continue
        rng = np.random.default_rng(4_000_000 + i)        # as in 77
        yr = pcd.permute_within(y, u, s, rng)
        w = run_strat(X, yr, s, u, u, f"DEED null {i}", seed=pcd.SEED + i,
                      fold_y=(y if fixed else None))
        rows = []
        for st in STRATA:
            k = (w[f"p_{st}"] <= ALPHA).to_numpy()
            g = pd.Series(k).groupby(w.subj.to_numpy())
            for sj, r in zip(g.mean().index, g.mean().to_numpy()):
                rows.append(dict(rep=i, stratum=st, subj=sj,
                                 n=int(g.size()[sj]), rate=r))
        pd.DataFrame(rows).to_csv(f, mode="a", header=not os.path.exists(f),
                                  index=False)
        fpr = (w.p_pooled <= ALPHA).mean()
        if i < len(ref) and not fixed:
            assert round(fpr, 4) == round(ref.fpr.iloc[i], 4), (i, fpr)
        log(f"  null{tag} rep {i}: pooled {fpr:.4f}, participant "
            f"{(w.p_participant <= ALPHA).mean():.4f}, input "
            f"{(w.p_input <= ALPHA).mean():.4f}")


def sim_one(i):
    """77's simulate(): same data, same seeds, all strata."""
    rng = np.random.default_rng(5_000_000 + i)
    P, d = 30, 20
    X, y, subj, unit = [], [], [], []
    for s in range(P):
        mu_s = rng.normal(0, 1.0, d)
        pri = rng.dirichlet([0.6, 0.6, 0.6])
        for r in range(rng.integers(4, 16)):
            mu_r = mu_s + rng.normal(0, 0.5, d)
            lab = rng.choice(3, p=pri)
            n = rng.integers(3, 25)
            scale = rng.lognormal(0, 0.6)
            X.append(mu_r + rng.normal(0, scale, (n, d)))
            y += [lab] * n
            subj += [s] * n
            unit += [f"{s}_{r}"] * n
    X = np.vstack(X)
    unit = np.array(unit)
    with threadpool_limits(NULL_THREADS):            # 77 ran sims in workers
        w = run_strat(X, np.array(y), np.array(subj), unit, unit, f"sim{i}",
                      b_count=50, seed=6_000_000 + i, n_jobs=1)
    out = []
    for st in STRATA:
        k = (w[f"p_{st}"] <= ALPHA)
        pf = k.groupby(w.subj).mean()
        out.append(dict(sim=i, stratum=st, n=len(w), fpr=k.mean(),
                        part_fpr_mean=pf.mean(), part_fpr_p90=pf.quantile(.9),
                        part_frac_above_2alpha=(pf > 2 * ALPHA).mean()))
    return out


def piece_sims():
    f = os.path.join(OUT, "pcd_strat_sims.csv")
    if os.path.exists(f):
        log("sims: already done")
        return
    log("sims: running")
    res = Parallel(n_jobs=N_JOBS)(delayed(sim_one)(i) for i in range(100))
    t = pd.DataFrame([r for rr in res for r in rr])
    t.to_csv(f, index=False)
    ref = pd.read_csv(os.path.join(OUT, "pcd_revised_sims.csv"))
    got = t[t.stratum == "pooled"].set_index("sim").fpr
    d = float((got - ref.set_index("sim").fpr).abs().max())
    log(f"  check vs 77 (sims): max |pooled rate diff| {d:.2e}")
    assert d < 5e-5, "pooled PCD does not reproduce 77 in the simulations"


def summarise_null(tag=""):
    f = os.path.join(OUT, f"pcd_strat_deednull{tag}_participants.csv")
    if not os.path.exists(f):
        return
    p = pd.read_csv(f)
    R = p.rep.nunique()
    rows = []
    for st, g in p.groupby("stratum", sort=False):
        e = g.groupby("subj").rate.agg(["mean", "std", "count"])
        se = e["std"] / np.sqrt(e["count"])
        per_rep = g.groupby("rep")[["rate", "n"]].apply(
            lambda h: np.average(h.rate, weights=h.n))
        marg = per_rep.mean()
        rows.append(dict(
            stratum=st, reps=R, marginal_rate=marg,
            marginal_se=per_rep.std(ddof=1) / np.sqrt(len(per_rep)),
            participant_expected_max=e["mean"].max(),
            participant_expected_p90=e["mean"].quantile(.9),
            participant_expected_median=e["mean"].median(),
            n_participants_above_2alpha=int((e["mean"] > 2 * ALPHA).sum()),
            n_participants_signif_above_alpha=int(
                ((e["mean"] - ALPHA) / se > 2).sum())))
    t = pd.DataFrame(rows)
    t.round(4).to_csv(os.path.join(OUT, f"pcd_strat_deednull{tag}_summary.csv"),
                      index=False)
    log(f"DEED null{tag}, expected rate per participant over {R} reps:")
    with pd.option_context("display.width", 220):
        log("\n" + t.round(4).to_string(index=False))


def main():
    pieces = sys.argv[1:] or ["deed", "deednull", "sims", "sleepedf"]
    log("=" * 70)
    log(f"STRATIFIED PCD  pieces {pieces}  R_NULL {R_NULL}  jobs {N_JOBS}")
    Xd, yd, sd, ud = deed()
    summ = []
    if "deed" in pieces:
        summ += summary(piece_real("DEED affect", (Xd, yd, sd, ud), "deed"))
    if "deednull" in pieces:
        piece_deednull(Xd, yd, sd, ud)
        summarise_null()
    if "deednull_fixed" in pieces:
        piece_deednull(Xd, yd, sd, ud, fixed=True)
        summarise_null("_fixedfolds")
    if "sims" in pieces:
        piece_sims()
        t = pd.read_csv(os.path.join(OUT, "pcd_strat_sims.csv"))
        log("simulations, median over datasets:")
        log("\n" + t.groupby("stratum", sort=False)[
            ["fpr", "part_fpr_mean", "part_fpr_p90",
             "part_frac_above_2alpha"]].median().round(4).to_string())
    if "sleepedf" in pieces:
        summ += summary(piece_real("Sleep-EDF staging", sleepedf(), "sleepedf"))
    if summ:
        f = os.path.join(OUT, "pcd_strat_summary.csv")
        new = pd.DataFrame(summ)
        if os.path.exists(f):
            old = pd.read_csv(f)
            new = pd.concat([old[~old.condition.isin(new.condition)], new])
        new.round(4).to_csv(f, index=False)
        with pd.option_context("display.width", 220):
            log("\n" + new[new.alpha == ALPHA].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
