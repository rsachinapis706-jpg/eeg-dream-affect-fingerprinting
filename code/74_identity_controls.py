"""
Participant fingerprinting: predictive identity baseline and balanced controls
(Reviewer 1, comment 6).

Part A -- a cross-validated identity rule
  The identity-only oracle in Table 8 is computed from the full label counts,
  without cross-validation: descriptive, not an unbiased predictive baseline.
  Here each participant's majority label is estimated from TRAINING windows
  only and applied to the held-out windows of the same folds used for the EEG
  classifier (the Table 4(a) folds and seeds). A participant absent from the
  training fold receives the training majority class. Reported against the
  majority-class baseline estimated the same way, and against the EEG random
  forest on identical folds, in accuracy and balanced accuracy, per fold
  assignment, so every comparison is paired.

Part B -- does performance fall because identity stops predicting the label?
  The manuscript's control restricts to the 21 participants with no class above
  70% of their recordings. Removing participants also changes sample size and
  composition, so two controls are added:
    residual association  normalised mutual information between participant
                          and label in the subgroup, with a 2,000-permutation
                          test at the recording level
    size-matched draws    random subsets of 21 of the 34 participants: the
                          same windowed classifier and the same statistics.
                          If the subgroup's accuracy is low only because it is
                          smaller, it should sit inside this distribution; if
                          it is low because identity no longer carries label
                          information, it should sit in the lower tail, with
                          its association.
    balanced control      the 24 participants who contribute all three classes,
                          with windows subsampled within each participant to
                          equal class counts, so that identity carries no
                          label information by construction. Ten subsamples.

Window-shuffled five-fold cross-validation, random forest (300 trees,
min_samples_leaf=2), throughout Part B, as in the manuscript's control.

Time-limited run (30 Sep 2026): Part A uses the first three of the ten Table
4(a) fold assignments; Part B uses 20 size-matched draws and 10 balanced
subsamples.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             mutual_info_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
LOG = os.path.join(OUT, "run_identity_controls.log")

SEEDS = [20260727 + i for i in range(3)]      # first three Table 4(a) assignments
N_SPLITS = 5
N_DRAWS = 20
N_BAL = 10
N_PERM = 2000
N_WORKERS = 1
RF_THREADS = 22
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
    d = d[d.E.isin(VALENCE)].reset_index(drop=True)
    d["y"] = d.E.map(VALENCE)
    return d, fc


def folds(y, groups, seed):
    if groups is None:
        return list(StratifiedKFold(N_SPLITS, shuffle=True,
                                    random_state=seed).split(y, y))
    return list(StratifiedGroupKFold(N_SPLITS, shuffle=True,
                                     random_state=seed).split(y, y, groups))


def rf_pred(X, y, sp, seed, threads):
    yp = np.empty(len(y), dtype=object)
    for tr, te in sp:
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   n_jobs=threads,
                                   random_state=seed).fit(X[tr], y[tr])
        yp[te] = m.predict(X[te])
    return yp


def identity_pred(y, subj, sp):
    yp = np.empty(len(y), dtype=object)
    base = np.empty(len(y), dtype=object)
    for tr, te in sp:
        tr_df = pd.DataFrame(dict(s=subj[tr], y=y[tr]))
        glob = tr_df.y.value_counts().idxmax()
        maj = tr_df.groupby("s").y.agg(lambda v: v.value_counts().idxmax())
        yp[te] = pd.Series(subj[te]).map(maj).fillna(glob).to_numpy()
        base[te] = glob
    return yp, base


def part_a_task(X, y, subj, rec, partition, seed):
    grp = {"window-shuffled": None, "recording-grouped": rec}[partition]
    sp = folds(y, grp, seed)
    ide, base = identity_pred(y, subj, sp)
    eeg = rf_pred(X, y, sp, seed, RF_THREADS)
    out = []
    for name, p in (("majority class", base), ("identity rule (CV)", ide),
                    ("EEG random forest", eeg)):
        out.append(dict(partition=partition, seed=seed, predictor=name,
                        acc=accuracy_score(y, p),
                        bal=balanced_accuracy_score(y, p)))
    return out


def nmi_rec(dd):
    r = dd.drop_duplicates("segment_id")
    h = mutual_info_score(r.y, r.y)
    return mutual_info_score(r.subject, r.y) / h if h > 0 else 0.0


def assoc(dd, seed):
    """Recording-level NMI with a permutation p (labels shuffled across the
    subset's recordings)."""
    r = dd.drop_duplicates("segment_id")
    obs = nmi_rec(dd)
    rng = np.random.default_rng(seed)
    h = mutual_info_score(r.y, r.y)
    s, yy = r.subject.to_numpy(), r.y.to_numpy()
    null = np.array([mutual_info_score(s, rng.permutation(yy)) / h
                     for _ in range(N_PERM)])
    return obs, obs - null.mean(), (1 + (null >= obs).sum()) / (1 + N_PERM)


def windowed(dd, fc, seed):
    X, y, s = dd[fc].to_numpy(float), dd.y.to_numpy(), dd.subject.to_numpy()
    sp = folds(y, None, seed)
    p = rf_pred(X, y, sp, seed, RF_THREADS)
    ide, _ = identity_pred(y, s, sp)
    return dict(acc=accuracy_score(y, p), bal=balanced_accuracy_score(y, p),
                id_acc=accuracy_score(y, ide),
                id_bal=balanced_accuracy_score(y, ide))


def draw_task(d, fc, pool, k, i):
    rng = np.random.default_rng(700_000 + i)
    pick = rng.choice(pool, k, replace=False)
    dd = d[d.subject.isin(pick)].reset_index(drop=True)
    w = windowed(dd, fc, SEEDS[0])
    return dict(draw=i, nmi=nmi_rec(dd), windows=len(dd), **w)


def balanced_task(d, fc, three, i):
    rng = np.random.default_rng(800_000 + i)
    parts = []
    for s in three:
        g = d[d.subject == s]
        n = g.y.value_counts().min()
        parts.append(g.groupby("y", group_keys=False).apply(
            lambda x: x.sample(n, random_state=int(rng.integers(1e9)))))
    dd = pd.concat(parts).reset_index(drop=True)
    w = windowed(dd, fc, SEEDS[0])
    win_nmi = mutual_info_score(dd.subject, dd.y) / mutual_info_score(dd.y,
                                                                       dd.y)
    return dict(subsample=i, windows=len(dd), window_nmi=win_nmi, **w)


def main():
    smoke = "--smoke" in sys.argv
    d, fc = load()
    X, y = d[fc].to_numpy(float), d.y.to_numpy()
    subj, rec = d.subject.to_numpy(), d.segment_id.to_numpy()
    seeds = SEEDS[:1] if smoke else SEEDS
    draws = 3 if smoke else N_DRAWS
    nbal = 2 if smoke else N_BAL
    global N_PERM
    if smoke:
        N_PERM = 100
    tag = "_smoke" if smoke else ""
    t0 = time.time()

    log("=" * 78)
    log("IDENTITY CONTROLS")
    log("=" * 78)
    log(f"windows {len(y)} | participants {d.subject.nunique()} | "
        f"recordings {d.segment_id.nunique()}")

    # ---------------- Part A ----------------
    res = Parallel(n_jobs=1 if smoke else N_WORKERS)(
        delayed(part_a_task)(X, y, subj, rec, p, s)
        for p in ("window-shuffled", "recording-grouped") for s in seeds)
    a = pd.DataFrame([r for rr in res for r in rr])
    a.to_csv(os.path.join(OUT, f"identity_controls_A{tag}.csv"), index=False)
    log("")
    log("-- A. identity rule estimated from training folds only --")
    sa = a.groupby(["partition", "predictor"])[["acc", "bal"]].agg(
        ["mean", "std"])
    log(sa.round(4).to_string())
    for p in ("window-shuffled", "recording-grouped"):
        w = a[a.partition == p].pivot(index="seed", columns="predictor",
                                      values=["acc", "bal"])
        for met in ("acc", "bal"):
            e, i_, m = (w[(met, "EEG random forest")],
                        w[(met, "identity rule (CV)")],
                        w[(met, "majority class")])
            log(f"  {p:<18} {met}: identity - majority {(i_ - m).mean():+.4f}"
                f" | EEG - majority {(e - m).mean():+.4f} | EEG - identity "
                f"{(e - i_).mean():+.4f} (sd {(e - i_).std():.4f})")
    log(f"  [{(time.time() - t0) / 60:.1f} min]")

    # ---------------- Part B ----------------
    r = d.drop_duplicates("segment_id")
    dom = r.groupby("subject").y.agg(lambda v: v.value_counts(
        normalize=True).max())
    low = sorted(dom[dom <= 0.70].index)
    sub = d[d.subject.isin(low)].reset_index(drop=True)
    nm, ex, p = assoc(sub, 1)
    nm_all, ex_all, p_all = assoc(d, 2)
    w_sub = windowed(sub, fc, SEEDS[0])
    w_all = windowed(d, fc, SEEDS[0])
    log("")
    log("-- B1. subgroup with no class above 70% of recordings --")
    log(f"all 34: NMI {nm_all:.4f} (excess {ex_all:.4f}, p {p_all:.4f}) | "
        f"RF acc {w_all['acc']:.4f} bal {w_all['bal']:.4f} | identity-CV acc "
        f"{w_all['id_acc']:.4f}")
    log(f"subgroup {len(low)}: NMI {nm:.4f} (excess {ex:.4f}, p {p:.4f}) | "
        f"RF acc {w_sub['acc']:.4f} bal {w_sub['bal']:.4f} | identity-CV acc "
        f"{w_sub['id_acc']:.4f}")

    dr = pd.DataFrame(Parallel(n_jobs=1 if smoke else N_WORKERS)(
        delayed(draw_task)(d, fc, sorted(d.subject.unique()), len(low), i)
        for i in range(draws)))
    dr.to_csv(os.path.join(OUT, f"identity_controls_draws{tag}.csv"),
              index=False)
    log("")
    log(f"-- B2. {draws} random subsets of {len(low)} participants --")
    for c in ("nmi", "acc", "bal", "id_acc"):
        v = {"nmi": nm, "acc": w_sub["acc"], "bal": w_sub["bal"],
             "id_acc": w_sub["id_acc"]}[c]
        log(f"  {c:<7} draws median {dr[c].median():.4f} "
            f"[{dr[c].quantile(.025):.4f}, {dr[c].quantile(.975):.4f}] | "
            f"subgroup {v:.4f} | share of draws at or below the subgroup "
            f"{(dr[c] <= v).mean():.3f}")
    log(f"  correlation across draws, NMI vs RF accuracy: "
        f"{dr[['nmi', 'acc']].corr().iloc[0, 1]:.3f}")

    three = sorted(g for g, v in d.groupby("subject").y.nunique().items()
                   if v == 3)
    bl = pd.DataFrame(Parallel(n_jobs=1 if smoke else N_WORKERS)(
        delayed(balanced_task)(d, fc, three, i) for i in range(nbal)))
    bl.to_csv(os.path.join(OUT, f"identity_controls_balanced{tag}.csv"),
              index=False)
    log("")
    log(f"-- B3. {len(three)} participants with all three classes, windows "
        f"balanced within each participant ({nbal} subsamples) --")
    log(f"  window-level NMI {bl.window_nmi.mean():.4f} | windows "
        f"{bl.windows.mean():.0f}")
    log(f"  RF acc {bl.acc.mean():.4f} ± {bl.acc.std():.4f} | bal "
        f"{bl.bal.mean():.4f} ± {bl.bal.std():.4f} | identity-CV acc "
        f"{bl.id_acc.mean():.4f} (chance 0.3333, classes equal)")
    log(f"  [{(time.time() - t0) / 60:.1f} min]")
    with open(LOG.replace(".log", f"{tag}.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
