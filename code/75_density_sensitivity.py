"""
Channel-density sweep with nested model selection and uncertainty
(Reviewer 1, comment 8).

33_channel_density.py evaluated nested electrode subsets (6-64 channels, chosen
by farthest-point sampling) on the Tononi corpus with the regularisation fixed
at C = 0.1 and a single fold assignment, and the manuscript concluded that
spatial resolution does not limit the result. Reviewer 1 notes that 64
channels give 1,152 features for 598 windows from 18 participants, that fixed
regularisation does not rule out an unfavourable dimensionality-to-sample
ratio, and that no uncertainty was reported.

Models, all inside participant-separated five-fold cross-validation:
  lr_fixed     logistic regression, C = 0.1, class-balanced (as published)
  lr_nested    C chosen from {0.001, 0.01, 0.1, 1, 10} by inner three-fold
               participant-grouped CV on the training participants
  pca_lr       PCA fitted on the training fold, number of components chosen
               from {5, 10, 20, 40} jointly with C in {0.01, 0.1, 1} by the
               same inner CV -- dimensionality reduction nested
  rf           random forest, 300 trees, min_samples_leaf = 2 (as published)

Five fold assignments. For each model and channel count: balanced accuracy
(mean and SD over assignments), a participant-cluster bootstrap 95% interval,
and a recording-level permutation p against chance for the nested models.
Trend: the paired change from 6 to 64 channels and the slope against log2
channel count, both with participant-bootstrap intervals, on identical windows.

Fix (1 Oct 2026): the participant bootstrap now averages balanced accuracy
over the five fold assignments inside every resample, exactly as the point
estimate does. It previously resampled only the first assignment, so an
interval could exclude its own estimate. Out-of-fold predictions are saved
to density_sensitivity_preds.parquet. The permutation p-values remain on the
first fold assignment and are labelled so.

Window-shuffled side (Reviewer 1, comment 4: the same metric on both sides of
the density figure). The published figure set window-shuffled ACCURACY against
participant-separated BALANCED accuracy, and its table took the better of the
two classifiers in each cell. Here lr_fixed and rf are also evaluated under
window-shuffled five-fold cross-validation (StratifiedKFold, same seeds), and
both partitions are reported in balanced accuracy (accuracy alongside), per
model, never maximised over models.

Standard deviations across fold assignments use the population form (ddof=0),
as elsewhere in the paper.

The Tononi label is the experience category of the awakening (experience,
experience without recall, no experience), not affect; conclusions are about
this task only.
"""
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (GroupKFold, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
CACHE = os.path.join(OUT, "tononi_64ch.parquet")
LOG = os.path.join(OUT, "run_density_sensitivity.log")

SUBSETS = [6, 12, 24, 48, 64]
SEEDS = [20260727 + i for i in range(5)]
C_GRID = [0.001, 0.01, 0.1, 1, 10]
PCA_GRID = [(k, c) for k in (5, 10, 20, 40) for c in (0.01, 0.1, 1)]
N_BOOT = 2000
N_PERM = 100
N_WORKERS = 12
MODELS = ["lr_fixed", "lr_nested", "pca_lr", "rf"]

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def lr(c):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=c, max_iter=3000, class_weight="balanced"))


def pca_lr(k, c):
    return make_pipeline(StandardScaler(), PCA(n_components=k,
                                               random_state=0),
                         LogisticRegression(C=c, max_iter=3000,
                                            class_weight="balanced"))


def inner_pick(X, y, g, candidates, build):
    """Choose a candidate by inner participant-grouped CV on training data."""
    best, best_s = None, -1
    inner = GroupKFold(n_splits=3)
    for cand in candidates:
        yt, yp = [], []
        for tr, te in inner.split(X, y, g):
            if len(np.unique(y[tr])) < 2:
                continue
            k = cand[0] if isinstance(cand, tuple) else None
            if k is not None and k >= min(len(tr), X.shape[1]):
                continue
            m = build(cand).fit(X[tr], y[tr])
            yt.append(y[te]); yp.append(m.predict(X[te]))
        if not yt:
            continue
        s = balanced_accuracy_score(np.concatenate(yt), np.concatenate(yp))
        if s > best_s:
            best, best_s = cand, s
    return best


def fit_predict(model, Xtr, ytr, gtr, Xte, seed):
    if model == "lr_fixed":
        m = lr(0.1)
    elif model == "lr_nested":
        m = lr(inner_pick(Xtr, ytr, gtr, C_GRID, lr))
    elif model == "pca_lr":
        k, c = inner_pick(Xtr, ytr, gtr, PCA_GRID, lambda kc: pca_lr(*kc))
        m = pca_lr(k, c)
    else:
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   n_jobs=1, random_state=seed)
    return m.fit(Xtr, ytr).predict(Xte)


def evaluate(X, y, g, model, seed):
    yp = np.empty(len(y), dtype=object)
    for tr, te in StratifiedGroupKFold(5, shuffle=True,
                                      random_state=seed).split(X, y, g):
        yp[te] = fit_predict(model, X[tr], y[tr], g[tr], X[te], seed)
    return yp


def evaluate_ws(X, y, model, seed):
    """Window-shuffled five-fold; fixed-hyperparameter models only."""
    assert model in ("lr_fixed", "rf")
    yp = np.empty(len(y), dtype=object)
    for tr, te in StratifiedKFold(5, shuffle=True,
                                  random_state=seed).split(X, y):
        yp[te] = fit_predict(model, X[tr], y[tr], None, X[te], seed)
    return yp


def perm_labels(y, rec, rng):
    r = pd.DataFrame(dict(rec=rec, y=y)).drop_duplicates("rec")
    mp = dict(zip(r.rec, rng.permutation(r.y.to_numpy())))
    return np.array([mp[k] for k in rec])


def boot(y, preds_by_n, g, rng):
    """Participant bootstrap of per-subset BA, the 6->64 change and the slope,
    all on the same resampled participants. preds_by_n[n] is a list of
    prediction arrays, one per fold assignment; each resample averages BA over
    the assignments, as the point estimate does."""
    uq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uq}
    x = np.log2(SUBSETS)
    out = []
    for _ in range(N_BOOT):
        sel = np.concatenate([idx[u] for u in rng.choice(uq, len(uq))])
        if len(np.unique(y[sel])) < 2:
            continue
        b = np.array([np.mean([balanced_accuracy_score(y[sel], p[sel])
                               for p in preds_by_n[n]]) for n in SUBSETS])
        out.append(np.concatenate([b, [b[-1] - b[0], np.polyfit(x, b, 1)[0]]]))
    return np.array(out)


def main():
    smoke = "--smoke" in sys.argv
    df = pd.read_parquet(CACHE)
    fc_all = [c for c in df.columns if c.startswith("ch")]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=fc_all)
    y = df.dreaming.astype(str).to_numpy()
    g = df.subject.astype(str).to_numpy()
    rec = df.recording.astype(str).to_numpy()
    seeds = SEEDS[:1] if smoke else SEEDS
    nperm = 5 if smoke else N_PERM
    models = ["lr_fixed", "lr_nested"] if smoke else MODELS
    subsets = {n: [c for c in fc_all
                   if int(re.match(r"ch(\d+)_", c).group(1)) < n]
               for n in SUBSETS}
    t0 = time.time()
    log("=" * 78)
    log("CHANNEL DENSITY WITH NESTED MODEL SELECTION (Tononi, experience label)")
    log("=" * 78)
    log(f"windows {len(y)} | participants {len(np.unique(g))} | recordings "
        f"{len(np.unique(rec))} | classes {sorted(np.unique(y))}")
    for n in SUBSETS:
        log(f"  {n:>2} channels: {len(subsets[n])} features "
            f"({len(subsets[n]) / len(y):.2f} per window)")

    jobs = [(n, m, s) for n in SUBSETS for m in models for s in seeds]
    res = Parallel(n_jobs=N_WORKERS)(
        delayed(evaluate)(df[subsets[n]].to_numpy(float), y, g, m, s)
        for n, m, s in jobs)
    preds = {j: p for j, p in zip(jobs, res)}
    pd.DataFrame([dict(channels=n, model=m, seed=sd, idx=i, pred=v)
                  for (n, m, sd), p in preds.items() for i, v in enumerate(p)]
                 ).to_parquet(os.path.join(
                     OUT, f"density_sensitivity_preds"
                          f"{'_smoke' if smoke else ''}.parquet"))
    rows = []
    for (n, m, s), p in preds.items():
        rows.append(dict(channels=n, model=m, seed=s,
                         bal=balanced_accuracy_score(y, p),
                         acc=accuracy_score(y, p)))
    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(OUT, f"density_sensitivity_raw"
                                 f"{'_smoke' if smoke else ''}.csv"),
               index=False)
    log(f"  [{(time.time() - t0) / 60:.1f} min]")

    rng = np.random.default_rng(20260930)
    summ = []
    for m in models:
        pb = {n: [preds[(n, m, sd)] for sd in seeds] for n in SUBSETS}
        bt = boot(y, pb, g, rng)
        for i, n in enumerate(SUBSETS):
            v = raw[(raw.model == m) & (raw.channels == n)].bal
            summ.append(dict(model=m, channels=n, bal_mean=v.mean(),
                             bal_sd=v.std(ddof=0),
                             ci_lo=np.percentile(bt[:, i], 2.5),
                             ci_hi=np.percentile(bt[:, i], 97.5)))
        d = raw[raw.model == m].pivot(index="seed", columns="channels",
                                      values="bal")
        summ.append(dict(model=m, channels="64 minus 6",
                         bal_mean=(d[64] - d[6]).mean(),
                         bal_sd=(d[64] - d[6]).std(ddof=0),
                         ci_lo=np.percentile(bt[:, -2], 2.5),
                         ci_hi=np.percentile(bt[:, -2], 97.5)))
        slope = [np.polyfit(np.log2(SUBSETS), d.loc[s, SUBSETS].to_numpy(),
                            1)[0] for s in d.index]
        summ.append(dict(model=m, channels="slope per doubling",
                         bal_mean=np.mean(slope), bal_sd=np.std(slope),
                         ci_lo=np.percentile(bt[:, -1], 2.5),
                         ci_hi=np.percentile(bt[:, -1], 97.5)))
    s = pd.DataFrame(summ)

    # window-shuffled side, same metric, per model
    ws_models = [x for x in ("lr_fixed", "rf") if x in models]
    ws_jobs = [(n, m, sd) for n in SUBSETS for m in ws_models for sd in seeds]
    ws_res = Parallel(n_jobs=N_WORKERS)(
        delayed(evaluate_ws)(df[subsets[n]].to_numpy(float), y, m, sd)
        for n, m, sd in ws_jobs)
    ws = pd.DataFrame([dict(channels=n, model=m, seed=sd,
                            bal=balanced_accuracy_score(y, p),
                            acc=accuracy_score(y, p))
                       for (n, m, sd), p in zip(ws_jobs, ws_res)])
    ws.to_csv(os.path.join(OUT, f"density_window_shuffled_raw"
                                f"{'_smoke' if smoke else ''}.csv"),
              index=False)
    both = pd.concat([ws.assign(partition="window-shuffled"),
                      raw[raw.model.isin(ws_models)].assign(
                          partition="participant-separated")])
    wsum = both.groupby(["model", "channels", "partition"]).agg(
        bal_mean=("bal", "mean"), bal_sd=("bal", lambda v: v.std(ddof=0)),
        acc_mean=("acc", "mean"), acc_sd=("acc", lambda v: v.std(ddof=0)),
        n_assignments=("seed", "nunique")).reset_index()
    log("")
    log("-- both partitions, balanced accuracy (accuracy alongside), per model --")
    with pd.option_context("display.width", 200):
        log(wsum.round(4).to_string(index=False))
    log(f"  [{(time.time() - t0) / 60:.1f} min]")

    # permutation p against chance for the nested linear models
    pr = []
    for m in [x for x in ("lr_nested", "pca_lr") if x in models]:
        for n in SUBSETS:
            X = df[subsets[n]].to_numpy(float)
            obs = balanced_accuracy_score(y, preds[(n, m, seeds[0])])
            nulls = Parallel(n_jobs=N_WORKERS)(
                delayed(lambda k: balanced_accuracy_score(
                    yl := perm_labels(y, rec, np.random.default_rng(930_000 + k)),
                    evaluate(X, yl, g, m, seeds[0])))(k)
                for k in range(nperm))
            nulls = np.array(nulls)
            pr.append(dict(model=m, channels=n, observed=obs,
                           null_mean=nulls.mean(),
                           p=(1 + (nulls >= obs).sum()) / (1 + nperm),
                           n_perm=nperm))
    pr = pd.DataFrame(pr)
    tag = "_smoke" if smoke else ""
    s.round(4).to_csv(os.path.join(OUT, f"density_sensitivity{tag}.csv"),
                      index=False)
    pr.round(4).to_csv(os.path.join(OUT, f"density_permutation{tag}.csv"),
                       index=False)
    wsum.round(4).to_csv(os.path.join(OUT, f"density_both_partitions{tag}.csv"),
                         index=False)
    log("")
    with pd.option_context("display.width", 200):
        log(s.round(4).to_string(index=False))
        log("")
        log(pr.round(4).to_string(index=False))
    log(f"  [{(time.time() - t0) / 60:.1f} min]")
    with open(LOG.replace(".log", f"{tag}.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
