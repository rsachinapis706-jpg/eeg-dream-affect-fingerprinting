"""
Every PCD result the manuscript reports, recomputed with the score and the
permutation scheme now defined in Section III (upper-tail p-value with the
observed model in the reference; labels permuted within participant at the unit
at which they vary). Everything is imported from 77_pcd_revised.py, so there is
one implementation of the method. The earlier results (18, 21, 24, 46, 47) used
the withdrawn lower-tail score and are superseded.

  grid      coverage and retained balanced accuracy at alpha = 0.10, 0.05, 0.01
            and max-softmax > 0.5, 0.7, 0.9, with participant-bootstrap 95%
            intervals, from the per-window output of 77 (no refitting)
  cbramod   DEED affect on the frozen CBraMod embeddings (pretrained weights
            only, never fine-tuned on these labels), same folds and model
  noise     Sleep-EDF labels replaced by a uniformly random class with
            probability eta in {0, .2, .4, .6, .8, 1}; full cohort, same folds.
            eta = 0 must reproduce 77's Sleep-EDF condition exactly
  nulldep   the reference null under changes to the ANALYSIS with the task
            fixed (design of 47): features 9/18/36, training participants
            10/20/40, regularisation C 0.01/0.1/1, on the 47 subset (40
            participants, at most 1,200 windows each). Accuracy and balanced
            accuracy are both reported (47 stored accuracy in a column named
            "balanced").

Outputs: out/pcd_extra_{grid,cbramod,noise,nulldep}.csv and
out/pcd_extra_{cbramod,noise}_windows.parquet.
"""
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "pcd", os.path.join(HERE, "77_pcd_revised.py"))
pcd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pcd)
OUT = pcd.OUT
N_BOOT = 2000
ETAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
LOG = os.path.join(OUT, "run_pcd_extra.log")
lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def boot_rules(w, rules, rng):
    """Participant bootstrap of coverage and retained balanced accuracy."""
    uq, sidx = np.unique(w.subj.to_numpy(), return_inverse=True)
    P = len(uq)
    cls = sorted(w.y.unique())
    y, pr = w.y.to_numpy(), w.pred.to_numpy()
    W = np.zeros((N_BOOT, P))
    for b in range(N_BOOT):
        np.add.at(W[b], rng.integers(0, P, P), 1)
    n_all = np.bincount(sidx, minlength=P).astype(float)
    out = []
    for name, keep in rules:
        kept = np.bincount(sidx, weights=keep.astype(float), minlength=P)
        tot = np.zeros((P, len(cls))); tp = np.zeros((P, len(cls)))
        for c, lab in enumerate(cls):
            m = keep & (y == lab)
            np.add.at(tot[:, c], sidx[m], 1)
            np.add.at(tp[:, c], sidx[m & (pr == lab)], 1)
        cov_b = (W @ kept) / (W @ n_all)
        with np.errstate(invalid="ignore", divide="ignore"):
            bal_b = np.nanmean((W @ tp) / (W @ tot), axis=1)
        cov = keep.mean()
        bal = (balanced_accuracy_score(y[keep], pr[keep]) if keep.sum() > 10
               else np.nan)
        out.append(dict(rule=name, coverage=cov,
                        cov_lo=np.percentile(cov_b, 2.5),
                        cov_hi=np.percentile(cov_b, 97.5),
                        bal_retained=bal,
                        bal_lo=np.nanpercentile(bal_b, 2.5),
                        bal_hi=np.nanpercentile(bal_b, 97.5)))
    return out


def grid():
    w = pd.read_parquet(os.path.join(OUT, "pcd_revised_windows.parquet"))
    rng = np.random.default_rng(20261001)
    rows = []
    for cond, g in w.groupby("condition"):
        rules = [(f"PCD alpha={a}", (g.p_pcd <= a).to_numpy())
                 for a in (0.10, 0.05, 0.01)]
        rules += [(f"Max-softmax>{t}", (g.msp > t).to_numpy())
                  for t in (0.5, 0.7, 0.9)]
        rules += [("all inputs", np.ones(len(g), bool))]
        for r in boot_rules(g.reset_index(drop=True), rules, rng):
            rows.append(dict(condition=cond, participants=g.subj.nunique(),
                             windows=len(g), **r))
    t = pd.DataFrame(rows)
    t.round(4).to_csv(os.path.join(OUT, "pcd_extra_grid.csv"), index=False)
    with pd.option_context("display.width", 200):
        log(t.round(4).to_string(index=False))


def cbramod():
    # CBRAMOD_SCALE=uv100: the embeddings of the windows in units of 100 uV
    # (15_cbramod_within_subject.py with the same setting); outputs then
    # carry the suffix _uv100
    scale = os.environ.get("CBRAMOD_SCALE", "")
    assert scale in ("", "uv100"), scale
    sfx = f"_{scale}" if scale else ""
    Z = np.load(os.path.join(OUT, f"cbramod_embeddings{sfx}.npy"))
    m = pd.read_parquet(os.path.join(OUT, "deed_raw_meta.parquet"))
    assert len(m) == len(Z) and (m.index == np.arange(len(m))).all()
    d = m[m.E.isin(pcd.VALENCE)]
    X = np.asarray(Z[d.index.to_numpy()], dtype=float)
    y = d.E.map(pcd.VALENCE).to_numpy()
    s = d.subject.astype(str).to_numpy()
    u = d.segment_id.to_numpy()
    log(f"CBraMod embeddings: {X.shape[0]} windows x {X.shape[1]} dims, "
        f"{len(set(s))} participants, {len(set(u))} recordings")
    w = pcd.run_condition(X, y, s, u, "DEED affect, CBraMod embeddings")
    w.to_parquet(os.path.join(OUT, f"pcd_extra_cbramod{sfx}_windows.parquet"))
    sm = pcd.summarise(w)
    sm.round(4).to_csv(os.path.join(OUT, f"pcd_extra_cbramod{sfx}.csv"),
                       index=False)
    with pd.option_context("display.width", 200):
        log(sm.round(4).to_string(index=False))


def sleepedf():
    s = pd.read_parquet(os.path.join(OUT, "sleepedf_expanded_features.parquet"))
    sfc = [c for c in s.columns if c.startswith("ch")]
    assert "epoch" not in sfc and len(sfc) == 36
    s = s.replace([np.inf, -np.inf], np.nan).dropna(subset=sfc)
    s = s[s.stage.isin(["W", "N1", "N2"])].reset_index(drop=True)
    return s, sfc


def noise():
    s, sfc = sleepedf()
    X, y0 = s[sfc].to_numpy(float), s.stage.to_numpy()
    sj = s.subject.astype(str).to_numpy()
    u = np.arange(len(y0)).astype(str)
    classes = np.unique(y0)
    ref = pd.read_parquet(os.path.join(OUT, "pcd_revised_windows.parquet"))
    ref = ref[ref.condition == "Sleep-EDF staging"]
    rows, ws = [], []
    for eta in ETAS:
        r = np.random.default_rng(20261001 + int(eta * 1000))
        y = y0.copy()
        flip = r.random(len(y)) < eta
        y[flip] = r.choice(classes, size=flip.sum())
        w = pcd.run_condition(X, y, sj, u, f"eta={eta}")
        if eta == 0.0:
            same = (np.sort(w.p_pcd.to_numpy())
                    == np.sort(ref.p_pcd.to_numpy())).all()
            log(f"  eta 0 reproduces 77's Sleep-EDF p-values exactly: {same}")
        rows.append(dict(eta=eta, windows=len(w),
                         accuracy=accuracy_score(w.y, w.pred),
                         balanced=balanced_accuracy_score(w.y, w.pred),
                         coverage=(w.p_pcd <= 0.05).mean(),
                         msp_coverage=(w.msp > 0.5).mean(),
                         mean_p=w.p_pcd.mean()))
        log(f"  eta {eta:.1f}: bal {rows[-1]['balanced']:.4f} | PCD coverage "
            f"{rows[-1]['coverage']:.4f} | max-softmax>0.5 "
            f"{rows[-1]['msp_coverage']:.4f}")
        ws.append(w.assign(eta=eta)[["eta", "subj", "y", "pred", "p_pcd",
                                     "msp"]])
    pd.concat(ws).to_parquet(os.path.join(OUT, "pcd_extra_noise_windows.parquet"))
    t = pd.DataFrame(rows)
    t.round(4).to_csv(os.path.join(OUT, "pcd_extra_noise.csv"), index=False)
    v = t.dropna()
    log(f"  Spearman(balanced, coverage) "
        f"{v[['balanced', 'coverage']].corr('spearman').iloc[0, 1]:.4f}; "
        f"Pearson {v[['balanced', 'coverage']].corr().iloc[0, 1]:.4f}")


def nulldep():
    s, sfc = sleepedf()
    allsub = s.subject.unique()
    base_c = 0.1

    def subset(nsub, nfeat):
        rng2 = np.random.default_rng(pcd.SEED)          # as in 47
        ids = rng2.choice(allsub, size=min(nsub, len(allsub)), replace=False)
        d = s[s.subject.isin(ids)].groupby("subject",
                                          group_keys=False).head(1200)
        cols = list(rng2.choice(sfc, size=nfeat, replace=False))
        return d, cols

    def run(nsub, nfeat, C):
        d, cols = subset(nsub, nfeat)
        orig, orig_p, null_means = pcd.clf, pcd.pcd_p, []

        def wrapped_p(s_obs, s_null):          # records the null margins' mean
            null_means.append(float(s_null.mean()))
            return orig_p(s_obs, s_null)

        pcd.clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(
            C=C, max_iter=2000, class_weight="balanced"))
        pcd.pcd_p = wrapped_p
        try:
            w = pcd.run_condition(d[cols].to_numpy(float), d.stage.to_numpy(),
                                  d.subject.astype(str).to_numpy(),
                                  np.arange(len(d)).astype(str), "nulldep")
        finally:
            pcd.clf, pcd.pcd_p = orig, orig_p
        return w, float(np.mean(null_means))

    settings = ([("features", nf, 40, nf, base_c) for nf in (9, 18, 36)]
                + [("participants", ns, ns, 36, base_c) for ns in (10, 20, 40)]
                + [("capacity", C, 40, 36, C) for C in (0.01, 0.1, 1.0)])
    cache, rows = {}, []
    for knob, val, ns, nf, C in settings:
        key = (ns, nf, C)
        if key not in cache:
            cache[key] = run(ns, nf, C)
        w, nm = cache[key]
        rows.append(dict(knob=knob, value=val, windows=len(w),
                         accuracy=accuracy_score(w.y, w.pred),
                         balanced=balanced_accuracy_score(w.y, w.pred),
                         null_mean_margin=nm,
                         coverage=(w.p_pcd <= 0.05).mean()))
        log(f"  {knob:<13}{val:>6}: windows {len(w)} | bal "
            f"{rows[-1]['balanced']:.4f} | null mean margin {nm:.4f} | "
            f"coverage {rows[-1]['coverage']:.4f}")
    t = pd.DataFrame(rows)
    t.round(4).to_csv(os.path.join(OUT, "pcd_extra_nulldep.csv"), index=False)


def main():
    parts = sys.argv[1:] or ["grid", "cbramod", "nulldep", "noise"]
    t0 = time.time()
    for p in parts:
        log(f"-- {p} --")
        {"grid": grid, "cbramod": cbramod, "noise": noise,
         "nulldep": nulldep}[p]()
        log(f"  [{(time.time() - t0) / 60:.1f} min]")
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
