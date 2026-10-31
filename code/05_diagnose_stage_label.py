"""
Diagnostic: what does the segment's stage tag actually describe?

The positive control (decode W/N1/N2) performs barely above its permutation null
when features are averaged over a whole segment. Segments run 2-15 minutes and
certainly span several sleep stages, yet each carries ONE stage tag.

Hypothesis: the tag describes the state at (or near) the awakening, not the whole
segment. If so, features from the LAST epoch should decode stage substantially
better than features averaged over the whole segment.

This is a diagnostic on the positive control only. No dream-affect claim depends
on it, and no affect target is touched here.

Aggregations compared:
    seg_mean   - mean over all epochs                     (what run 1/2 used)
    last1      - final 20 s epoch before awakening
    last3      - mean of final three 20 s epochs
    first3     - mean of first three epochs (negative control: should be WORST
                 if the tag is about the awakening)
"""
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
SEED = 20260727
N_PERM = 200
N_JOBS = -1

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def make_clf():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced"),
    )


def loso_predict(X, y, groups):
    pred = np.empty(len(y), dtype=object)
    for s in np.unique(groups):
        te = groups == s
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            pred[te] = pd.Series(y[tr]).mode()[0]
            continue
        pred[te] = make_clf().fit(X[tr], y[tr]).predict(X[te])
    return pred


def run(X, y, groups, name):
    bacc = balanced_accuracy_score(y, loso_predict(X, y, groups))
    df = pd.DataFrame({"s": groups, "y": y})
    by_subj = [(g.index.to_numpy(), g.y.to_numpy()) for _, g in df.groupby("s")]

    def one(seed):
        rng = np.random.default_rng(seed)
        yp = np.empty(len(y), dtype=object)
        for idx, ys in by_subj:
            yp[idx] = rng.permutation(ys)
        return balanced_accuracy_score(yp, loso_predict(X, yp, groups))

    null = np.asarray(Parallel(n_jobs=N_JOBS)(
        delayed(one)(SEED + 977 * i) for i in range(N_PERM)))
    p = (np.sum(null >= bacc) + 1) / (N_PERM + 1)
    log(f"{name:<12} n={len(y):<5} bacc={bacc:.4f}  null={null.mean():.4f}"
        f"  delta={bacc - null.mean():+.4f}  p={p:.4f}")
    return dict(agg=name, n=len(y), bacc=bacc, null=null.mean(), p=p)


def main():
    ep = pd.read_parquet(os.path.join(OUT, "features_epoch.parquet"))
    meta = ["segment_id", "subject", "oid", "epoch_idx", "n_epochs",
            "epochs_from_end", "E", "film", "stage", "has_text"]
    fc = [c for c in ep.columns if c not in meta]

    ep = ep[ep.stage.isin(["W", "N1", "N2"])].copy()

    log("=" * 74)
    log("DIAGNOSTIC: does the stage tag describe the whole segment or the awakening?")
    log("=" * 74)
    log(f"epochs available: {len(ep)} | segments: {ep.segment_id.nunique()} "
        f"| subjects: {ep.subject.nunique()}")
    log(f"permutations: {N_PERM}  (labels shuffled within subject)")
    log("")

    aggs = {
        "seg_mean": lambda g: g[fc].mean(),
        "last1":    lambda g: g.nsmallest(1, "epochs_from_end")[fc].mean(),
        "last3":    lambda g: g.nsmallest(3, "epochs_from_end")[fc].mean(),
        "first3":   lambda g: g.nsmallest(3, "epoch_idx")[fc].mean(),
    }

    results = []
    for name, fn in aggs.items():
        rows = []
        for sid, g in ep.groupby("segment_id"):
            v = fn(g)
            v["segment_id"] = sid
            v["subject"] = g.subject.iloc[0]
            v["stage"] = g.stage.iloc[0]
            rows.append(v)
        d = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
        results.append(run(d[fc].to_numpy(float), d.stage.to_numpy(),
                           d.subject.to_numpy(), name))

    log("")
    log("Interpretation:")
    log("  last1/last3 >> seg_mean  -> tag describes the AWAKENING; segment")
    log("                              averaging is invalid and must be dropped.")
    log("  all roughly equal        -> tag is segment-wide; the weak control is a")
    log("                              feature problem, not a labelling problem.")
    log("")
    pd.DataFrame(results).round(4).to_csv(
        os.path.join(OUT, "stage_label_diagnostic.csv"), index=False)
    with open(os.path.join(OUT, "test3_stage_diagnostic.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"Wrote {OUT}\\test3_stage_diagnostic.txt")


if __name__ == "__main__":
    main()
