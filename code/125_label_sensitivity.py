"""
Label provenance and evaluation target (round 3, Reviewer 2, comment 3).

Questions
  1. Corroborated subset. DEED's ratings workbook can be matched to recordings
     only by position within a night; for 358 of 533 recordings it matches and
     the recording holds exactly the listed interval at 200 Hz
     (90_deed_alignment.py). Does the principal analysis hold on the
     recordings so corroborated (351 of them carry valence labels)?
  2. Pre-awakening windows. A report is assigned to a whole recording of
     variable length. Restricting each recording to its last 60 s (the three
     windows ending at the awakening; prespecified, epochs_from_end < 3) tests
     dilution by earlier parts of the interval.
  3. Evaluation target. A window-pooled score weights recordings and
     participants by their duration. Two further statistics, prespecified:
       recording level   per recording, the class probabilities of its
                         out-of-fold windows are averaged and the most
                         probable class is the recording's prediction;
                         balanced accuracy over recordings;
       participant-weighted  window-level balanced accuracy with each window
                         weighted by 1 / (windows of its participant), so every
                         participant counts equally.

Procedure (the primary procedure of 120_primary_random.py)
  The four Table 5 candidates; 20 random class-balanced participant
  partitions (seeds 30_001_000 + r), folds recomputed from the labelling
  evaluated; forest seed fixed; each statistic averaged over the partitions.
  Within-participant recording-level permutation null (65.permuted, mode
  "within"), sequential Monte Carlo p-values (H = 20, at most 1,000), per
  statistic; a configuration stops when all its statistics are decided.
  Participant-bootstrap intervals (2,000 resamples) on the fixed observed
  predictions, one participant draw per resample shared by the partitions.

Outputs (out/)
  label_sensitivity_raw.csv       observed and permuted statistics
  label_sensitivity_preds.parquet observed out-of-fold probabilities
  label_sensitivity_summary.csv   per data version, configuration, statistic

Usage: python 125_label_sensitivity.py run|summary
"""
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t5 = _load("t5", "65_config_table.py")
ps = _load("ps", "114_partition_sensitivity.py")
pr = _load("pr", "120_primary_random.py")
OUT = t5.OUT
R, PART_SEED, H, B_MAX = pr.R, pr.PART_SEED, pr.H, pr.B_MAX
MODE = "within"
STATS = ("window", "recording", "participant_weighted")
N_JOBS = int(os.environ.get("LABEL_JOBS", 8))
RAW = os.path.join(OUT, "label_sensitivity_raw.csv")
LOG = os.path.join(OUT, "run_label_sensitivity.log")
T0 = time.time()


def log(m=""):
    m = f"[{(time.time() - T0) / 60:7.1f} min] {m}"
    print(m, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(m + "\n")


def variants():
    d = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    fc = [c for c in d.columns if c not in t5.META]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = d[d.E.isin(t5.VALENCE)]
    X, y, subj, rec = t5.load()
    assert (d.segment_id.to_numpy() == rec).all() and len(d) == len(y)
    a = pd.read_csv(os.path.join(OUT, "deed_alignment.csv"))
    corr = set(a[a.join_ok & np.isclose(a.ratio, 200)].segment)
    masks = {"all windows": np.ones(len(y), bool),
             "corroborated recordings": d.segment_id.isin(corr).to_numpy(),
             "last 60 s before awakening": (d.epochs_from_end < 3).to_numpy()}
    return {k: (X[m], y[m], subj[m], rec[m]) for k, m in masks.items()}


def stats_one(X, yl, subj, rec, cfg, keep=False):
    """The three statistics averaged over the random partitions."""
    kind, scale, weight = t5.CONFIGS[cfg]
    cnt = pd.Series(subj).map(pd.Series(subj).value_counts()).to_numpy()
    w = 1.0 / cnt
    out = {s: [] for s in STATS}
    rows = []
    with threadpool_limits(t5.BLAS_THREADS, user_api="blas"):
        for r in range(R):
            P = np.zeros((len(yl), 3))
            classes = None
            for tr, te in ps.random_stratified_folds(yl, subj, t5.N_SPLITS,
                                                     PART_SEED + r):
                m = t5.build(kind, scale, weight, t5.SEEDS[0]).fit(X[tr], yl[tr])
                pp = m.predict_proba(X[te])
                classes = list(m.classes_) if classes is None else classes
                cols = [classes.index(c) for c in m.classes_]
                P[np.ix_(te, cols)] = pp
            cls = np.array(classes)
            pred = cls[P.argmax(1)]
            out["window"].append(balanced_accuracy_score(yl, pred))
            out["participant_weighted"].append(
                balanced_accuracy_score(yl, pred, sample_weight=w))
            g = pd.DataFrame(P, columns=cls).assign(rec=rec, y=yl).groupby("rec")
            mp = g[list(cls)].mean()
            ry = g.y.first().loc[mp.index]
            out["recording"].append(balanced_accuracy_score(
                ry, cls[mp.to_numpy().argmax(1)]))
            if keep:
                rows.append(pd.DataFrame(P, columns=[f"p_{c}" for c in cls])
                            .assign(partition=r, y=yl, pred=pred, subj=subj,
                                    rec=rec))
    res = {s: float(np.mean(v)) for s, v in out.items()}
    return res, (pd.concat(rows, ignore_index=True) if keep else None)


def task(V, var, cfg, b):
    X, y, subj, rec = V[var]
    yl = y if b < 0 else t5.permuted(y, subj, rec, b, MODE)
    res, preds = stats_one(X, yl, subj, rec, cfg, keep=b < 0)
    row = dict(variant=var, config=cfg, b=b, **res)
    return row, preds


def seq(raw, var, cfg, stat, obs):
    n = raw[(raw.variant == var) & (raw.config == cfg) & (raw.b >= 0)]
    n = n.sort_values("b")
    b = n.b.to_numpy()
    k = 0
    while k < len(b) and b[k] == k:
        k += 1
    hits = np.cumsum(n[stat].to_numpy()[:k] >= obs - 1e-12)
    stop = np.flatnonzero(hits >= H)
    if len(stop):
        return True, int(stop[0]) + 1, H / (int(stop[0]) + 1)
    if k >= B_MAX:
        c = int(hits[B_MAX - 1])
        return True, B_MAX, (1 + c) / (1 + B_MAX)
    return False, k, np.nan


def short(raw, var, cfg, stat, obs):
    """Exceedances still missing before the stopping count H can be reached,
    counted over the contiguous prefix of permutations as in seq()."""
    n = raw[(raw.variant == var) & (raw.config == cfg) & (raw.b >= 0)]
    n = n.sort_values("b")
    b = n.b.to_numpy()
    k = 0
    while k < len(b) and b[k] == k:
        k += 1
    return H - int((n[stat].to_numpy()[:k] >= obs - 1e-12).sum())


def run():
    V = variants()
    for k, (X, y, s, r) in V.items():
        log(f"{k}: {len(y)} windows, {len(np.unique(r))} recordings, "
            f"{len(np.unique(s))} participants")
    raw = pd.read_csv(RAW) if os.path.exists(RAW) else pd.DataFrame(
        columns=["variant", "config", "b", *STATS])
    pf = os.path.join(OUT, "label_sensitivity_preds.parquet")
    preds = [pd.read_parquet(pf)] if os.path.exists(pf) else []
    need = [(v, c) for v in V for c in t5.CANDIDATES
            if raw[(raw.variant == v) & (raw.config == c) & (raw.b < 0)].empty]
    if need:
        res = Parallel(n_jobs=N_JOBS)(delayed(task)(V, v, c, -1) for v, c in need)
        raw = pd.concat([raw, pd.DataFrame([x[0] for x in res])],
                        ignore_index=True)
        preds += [x[1].assign(variant=x[0]["variant"], config=x[0]["config"])
                  for x in res]
        pd.concat(preds, ignore_index=True).to_parquet(pf)
        raw.to_csv(RAW, index=False)
        for x in res:
            r_ = x[0]
            log(f"observed {r_['variant'][:22]:<22} {r_['config'][:26]:<26} "
                + " ".join(f"{s} {r_[s]:.4f}" for s in STATS))
    step = int(os.environ.get("LABEL_STEP", 16))
    while True:
        todo = []
        for v in V:
            for c in t5.CANDIDATES:
                o = raw[(raw.variant == v) & (raw.config == c) & (raw.b < 0)]
                live = [s for s in STATS
                        if not seq(raw, v, c, s, float(o[s].iloc[0]))[0]]
                if not live:
                    continue
                # no statistic can stop in fewer permutations than it lacks
                # exceedances, so the batch never computes one that the
                # stopping rule would not have needed
                n = max(short(raw, v, c, s, float(o[s].iloc[0])) for s in live)
                have = set(raw[(raw.variant == v) & (raw.config == c)].b)
                todo += [(v, c, b) for b in [x for x in range(B_MAX)
                                             if x not in have][:min(step, n)]]
        if not todo:
            break
        res = Parallel(n_jobs=N_JOBS)(delayed(task)(V, v, c, b)
                                      for v, c, b in todo)
        raw = pd.concat([raw, pd.DataFrame([x[0] for x in res])],
                        ignore_index=True)
        raw.to_csv(RAW, index=False)
        log(f"permutations: {int((raw.b >= 0).sum())} done; "
            f"{len(todo)} in last batch")
    log("permutation part finished")


def boot_ci(p, stat, n_boot=2000):
    """Participant bootstrap of the partition-averaged statistic, predictions
    held fixed. A participant drawn m times counts m times: each draw sums m
    copies of that participant's confusion counts -- over windows ("window"),
    over windows weighted 1/(the participant's window count)
    ("participant_weighted"), or over the participant's recordings, each
    predicted by its mean class probabilities ("recording"). Draw 0 is the
    cohort itself and must reproduce the observed statistic."""
    cls = [c[2:] for c in p.columns if c.startswith("p_")]
    K = len(cls)
    uq = np.unique(p.subj)
    si = pd.Series(np.arange(len(uq)), index=uq)
    rng = np.random.default_rng(20261021)
    draws = np.stack([np.ones(len(uq))] + [
        np.bincount(rng.integers(0, len(uq), len(uq)), minlength=len(uq))
        for _ in range(n_boot)])
    per = np.zeros((n_boot + 1, R))
    for r in range(R):
        g = p[p.partition == r]
        if stat == "recording":
            gg = g.groupby("rec")
            mp = gg[[f"p_{c}" for c in cls]].mean()
            first = gg[["y", "subj"]].first().loc[mp.index]
            yt = first.y.to_numpy()
            yp = np.array(cls)[mp.to_numpy().argmax(1)]
            s_ = first.subj.to_numpy()
            w = np.ones(len(yt))
        else:
            yt, yp, s_ = g.y.to_numpy(), g.pred.to_numpy(), g.subj.to_numpy()
            w = (1.0 / pd.Series(s_).map(pd.Series(s_).value_counts()).to_numpy()
                 if stat == "participant_weighted" else np.ones(len(yt)))
        C = np.zeros((len(uq), K, K))
        np.add.at(C, (si.loc[s_].to_numpy(), np.searchsorted(cls, yt),
                      np.searchsorted(cls, yp)), w)
        S = np.tensordot(draws, C, axes=(1, 0))
        tot = S.sum(2)
        have = tot > 0
        rec_ = np.where(have, np.diagonal(S, axis1=1, axis2=2)
                        / np.where(have, tot, 1), 0)
        per[:, r] = rec_.sum(1) / have.sum(1)
    m = per.mean(1)
    return m[0], np.percentile(m[1:], 2.5), np.percentile(m[1:], 97.5)


def summary():
    raw = pd.read_csv(RAW)
    P = pd.read_parquet(os.path.join(OUT, "label_sensitivity_preds.parquet"))
    rows = []
    for (v, c), g in raw.groupby(["variant", "config"], sort=False):
        o = g[g.b < 0].iloc[0]
        for s in STATS:
            done, k, p = seq(raw, v, c, s, float(o[s]))
            o0, lo, hi = boot_ci(P[(P.variant == v) & (P.config == c)], s)
            assert abs(o0 - float(o[s])) < 1e-9, (v, c, s, o0, o[s])
            rows.append(dict(variant=v, config=c, statistic=s,
                             observed=o[s], boot_lo=lo, boot_hi=hi,
                             null_mean=g[g.b >= 0][s].mean(), perms=k,
                             p_within=p, decided=done))
    t = pd.DataFrame(rows)
    t.round(4).to_csv(os.path.join(OUT, "label_sensitivity_summary.csv"),
                      index=False)
    with pd.option_context("display.width", 220):
        print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    {"run": run, "summary": summary}[sys.argv[1] if len(sys.argv) > 1
                                     else "summary"]()
