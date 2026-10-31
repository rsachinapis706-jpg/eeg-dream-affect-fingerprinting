"""
Paired effects for the protocol factorial, with intervals that describe the
same quantity as the estimates.

73_protocol_factorial.py reported each effect as the mean over three fold
assignments but bootstrapped only the first assignment, so an interval could
exclude its own estimate. Here every participant-bootstrap resample recomputes
the effect for each assignment on the resampled real windows and averages over
the assignments, exactly as the point estimate does.

Balanced accuracy depends only on per-class true-positive and total counts, so
counts are tabulated once per participant and a bootstrap resample, a vector
of participant multiplicities, becomes a matrix product.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
N_BOOT = 2000
META = {"segment_id", "subject", "oid", "epoch_idx", "n_epochs",
        "epochs_from_end", "E", "film", "stage", "has_text"}
VALENCE = {"E1", "E2", "E3", "E4", "E5"}
KEY = ["partition", "selection", "oversampling"]
CLASSES = ["neg", "neu", "pos"]
# --full (round 3): the complete 2 x 3 x 3 design of 73 --full, with every
# contrast at every level of the other factors.
FULL = "--full" in sys.argv
SFX = "_full" if FULL else ""


def counts(subj_idx, y, pred, n_part):
    """Per participant and class: number of windows, and correct ones."""
    tot = np.zeros((n_part, 3)); tp = np.zeros((n_part, 3))
    for c, name in enumerate(CLASSES):
        m = y == name
        np.add.at(tot[:, c], subj_idx[m], 1)
        np.add.at(tp[:, c], subj_idx[m & (pred == name)], 1)
    return tot, tp


def bal(W, tot, tp):
    """Balanced accuracy for each row of participant weights W."""
    T, P = W @ tot, W @ tp
    with np.errstate(invalid="ignore", divide="ignore"):
        r = P / T
    return np.nanmean(r, axis=1)


def acc(W, tot, tp):
    return (W @ tp).sum(1) / (W @ tot).sum(1)


def main():
    d = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    fc = [c for c in d.columns if c not in META]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = d[d.E.isin(VALENCE)].reset_index(drop=True)
    uq, sidx = np.unique(d.subject.to_numpy(), return_inverse=True)
    n_part = len(uq)

    pr = pd.read_parquet(os.path.join(OUT, f"protocol_factorial_preds{SFX}.parquet"))
    seeds = sorted(pr.seed.unique())
    tabs = {}
    for (p, s_, o, seed), g in pr.groupby(KEY + ["seed"]):
        tabs[((p, s_, o), seed)] = counts(sidx[g.idx.to_numpy()],
                                          g.y.to_numpy(), g.pred.to_numpy(),
                                          n_part)

    rng = np.random.default_rng(20260930)
    W = np.zeros((N_BOOT, n_part))
    for b in range(N_BOOT):
        np.add.at(W[b], rng.integers(0, n_part, n_part), 1)
    ones = np.ones((1, n_part))

    contrasts = []
    L = ("none", "inside", "before")
    if FULL:
        for p in ("participant", "window"):
            for o in L:
                contrasts += [("selection before vs inside", (p, "before", o),
                               (p, "inside", o)),
                              ("selection inside vs none", (p, "inside", o),
                               (p, "none", o))]
            for s_ in L:
                contrasts += [("oversampling before vs inside", (p, s_, "before"),
                               (p, s_, "inside")),
                              ("oversampling inside vs none", (p, s_, "inside"),
                               (p, s_, "none"))]
        for s_ in L:
            for o in L:
                contrasts.append(("window vs participant partition",
                                  ("window", s_, o), ("participant", s_, o)))
    for p in (() if FULL else ("participant", "window")):
        contrasts += [
            ("selection before vs inside", (p, "before", "none"),
             (p, "inside", "none")),
            ("oversampling before vs inside", (p, "none", "before"),
             (p, "none", "inside")),
            ("selection inside vs none", (p, "inside", "none"),
             (p, "none", "none")),
            ("oversampling inside vs none", (p, "none", "inside"),
             (p, "none", "none"))]
    for so in (() if FULL else (("none", "none"), ("before", "none"),
                                ("none", "before"), ("before", "before"))):
        contrasts.append(("window vs participant partition",
                          ("window",) + so, ("participant",) + so))

    rows, by_seed = [], []
    for name, a, b in contrasts:
        if (a, seeds[0]) not in tabs or (b, seeds[0]) not in tabs:
            continue
        est_b = np.mean([bal(ones, *tabs[(a, s)])[0] - bal(ones, *tabs[(b, s)])[0]
                         for s in seeds])
        est_a = np.mean([acc(ones, *tabs[(a, s)])[0] - acc(ones, *tabs[(b, s)])[0]
                         for s in seeds])
        boot = np.mean([bal(W, *tabs[(a, s)]) - bal(W, *tabs[(b, s)])
                        for s in seeds], axis=0)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        for s in seeds:
            by_seed.append(dict(effect=name, a="/".join(a), seed=s,
                                d_bal=bal(ones, *tabs[(a, s)])[0]
                                - bal(ones, *tabs[(b, s)])[0]))
        rows.append(dict(effect=name, a="/".join(a), b="/".join(b),
                         d_acc=est_a, d_bal=est_b, ci_lo=lo, ci_hi=hi,
                         n_seeds=len(seeds), n_boot=N_BOOT))
        print(f"{name:<34}{'/'.join(a):<27} d_acc {est_a:+.4f}  d_bal "
              f"{est_b:+.4f}  [{lo:+.4f}, {hi:+.4f}]", flush=True)
    pd.DataFrame(rows).round(4).to_csv(
        os.path.join(OUT, f"protocol_factorial_effects{SFX}.csv"), index=False)
    # The intervals resample participants with the fold assignments fixed, so
    # they omit variation between assignments; that is reported separately.
    pd.DataFrame(by_seed).round(6).to_csv(
        os.path.join(OUT, f"protocol_factorial_effects_by_seed{SFX}.csv"),
        index=False)


if __name__ == "__main__":
    main()
