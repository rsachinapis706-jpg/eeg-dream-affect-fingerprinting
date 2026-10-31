"""
PCD coverage depends on the reference null, not on decodability alone (R2 C8).

Reviewer 2 asks that the manuscript state explicitly that PCD's coverage is a
property of the reference distribution as well as of the task, and must not be
read as a universal measure of neural decodability. Section III-D already argues
this from the definition. This script measures it, so the limitation is
demonstrated rather than asserted.

The argument in Section III-D is that the pooled null margin distribution
F_null depends on quantities that are properties of the ANALYSIS rather than of
the task: how many features the null decoders are given, how much training data
they see, and how much capacity they have. All three govern how well a model
fits randomly permuted labels. If that is right, then holding the task fixed and
varying only those quantities should move coverage at a fixed alpha.

Design
------
One dataset, one task, one label set: Sleep-EDF staging, where signal is present
and verified. Three knobs are varied one at a time:

  features      a random subset of the available spectral features
  training size the number of participants contributing training data
  capacity      regularisation strength C of the decoder

For each setting the observed margins and the pooled null margins are recomputed
from scratch and coverage at alpha = 0.05 is recorded, alongside the balanced
accuracy actually achieved. If coverage moves while achieved accuracy stays
roughly constant, coverage is not measuring decodability.

Reading the result
------------------
The manuscript's claim is comparative, not absolute: PCD tells you whether an
input carries label information RELATIVE to what an uninformed model produces
under the same conditions. That is exactly what makes it useful within a dataset
and unusable as a cross-dataset scale, and this experiment is the evidence for
the second half of that sentence.
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
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")

SEED = 20260727
N_SPLITS = 5
B_NULL = 30
ALPHA = 0.05
N_JOBS = 20

N_FEATURES = [9, 18, 36]
N_SUBJECTS = [10, 20, 40]
CAPACITY = [0.01, 0.1, 1.0]

BASE_FEATURES = 36        # Sleep-EDF carries 2 EEG channels x 18 features
BASE_SUBJECTS = 40
BASE_C = 0.1

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def clf(C):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, C=C,
                                            class_weight="balanced"))


def margin(p):
    s = np.sort(p, axis=1)
    return s[:, -1] - s[:, -2]


def fit_null(Xtr, ytr, C, seed):
    rng = np.random.default_rng(seed)
    yp = rng.permutation(ytr)          # stage varies per window: window unit
    if len(np.unique(yp)) < 2:
        return None
    return clf(C).fit(Xtr, yp)


def evaluate(X, y, subj, C):
    """Returns coverage at ALPHA, achieved balanced accuracy, and the mean of
    the pooled null margin distribution."""
    sp = StratifiedGroupKFold(N_SPLITS, shuffle=True,
                              random_state=SEED).split(X, y, subj)
    D_all, ok_all, nullmean = [], [], []
    for tr, te in sp:
        if len(np.unique(y[tr])) < 2:
            continue
        m = clf(C).fit(X[tr], y[tr])
        pt = m.predict_proba(X[te])
        nulls = Parallel(n_jobs=N_JOBS)(
            delayed(fit_null)(X[tr], y[tr], C, SEED + 7919 * b)
            for b in range(B_NULL))
        nulls = [q for q in nulls if q is not None]
        S = np.concatenate([margin(q.predict_proba(X[te])) for q in nulls])
        D_all.append(np.searchsorted(np.sort(S), margin(pt), side="left")
                     / len(S))
        ok_all.append(m.classes_[pt.argmax(axis=1)] == y[te])
        nullmean.append(S.mean())
    D = np.concatenate(D_all)
    ok = np.concatenate(ok_all)
    yt = y
    return (float((D >= 1 - ALPHA).mean()), float(ok.mean()),
            float(np.mean(nullmean)))


def main():
    s = pd.read_parquet(os.path.join(OUT, "sleepedf_expanded_features.parquet"))
    fc = [c for c in s.columns if c.startswith("ch")]
    s = s.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    s = s[s.stage.isin(["W", "N1", "N2"])]
    rng = np.random.default_rng(SEED)
    allsub = s.subject.unique()

    log("=" * 78)
    log("PCD COVERAGE DEPENDS ON THE REFERENCE NULL, NOT ONLY ON THE TASK")
    log("=" * 78)
    log("Sleep-EDF staging throughout: one dataset, one task, one label set.")
    log(f"alpha = {ALPHA}, {B_NULL} null decoders per fold, "
        f"{N_SPLITS}-fold participant-separated")
    log("")

    def subset(nsub, nfeat, seed=SEED):
        ids = rng2.choice(allsub, size=min(nsub, len(allsub)), replace=False)
        d = s[s.subject.isin(ids)].groupby("subject",
                                          group_keys=False).head(1200)
        cols = list(rng2.choice(fc, size=nfeat, replace=False))
        return (d[cols].to_numpy(float), d.stage.to_numpy(),
                d.subject.to_numpy(), len(d))

    rows = []

    log("-- varying the number of FEATURES (participants and capacity fixed) --")
    log(f"{'features':>10}{'windows':>10}{'bal acc':>10}"
        f"{'null mean margin':>19}{'coverage':>11}")
    log("-" * 78)
    for nf in N_FEATURES:
        rng2 = np.random.default_rng(SEED)
        X, y, g, n = subset(BASE_SUBJECTS, nf)
        cov, acc, nm = evaluate(X, y, g, BASE_C)
        log(f"{nf:>10}{n:>10}{acc:>10.4f}{nm:>19.4f}{cov:>11.4f}")
        rows.append(dict(knob="features", value=nf, windows=n,
                         balanced=acc, null_mean_margin=nm, coverage=cov))
    log("")

    log("-- varying TRAINING SIZE (features and capacity fixed) --")
    log(f"{'participants':>13}{'windows':>10}{'bal acc':>10}"
        f"{'null mean margin':>19}{'coverage':>11}")
    log("-" * 78)
    for ns in N_SUBJECTS:
        rng2 = np.random.default_rng(SEED)
        X, y, g, n = subset(ns, BASE_FEATURES)
        cov, acc, nm = evaluate(X, y, g, BASE_C)
        log(f"{ns:>13}{n:>10}{acc:>10.4f}{nm:>19.4f}{cov:>11.4f}")
        rows.append(dict(knob="participants", value=ns, windows=n,
                         balanced=acc, null_mean_margin=nm, coverage=cov))
    log("")

    log("-- varying CAPACITY (regularisation C; data fixed) --")
    log(f"{'C':>10}{'windows':>10}{'bal acc':>10}"
        f"{'null mean margin':>19}{'coverage':>11}")
    log("-" * 78)
    for C in CAPACITY:
        rng2 = np.random.default_rng(SEED)
        X, y, g, n = subset(BASE_SUBJECTS, BASE_FEATURES)
        cov, acc, nm = evaluate(X, y, g, C)
        log(f"{C:>10.2f}{n:>10}{acc:>10.4f}{nm:>19.4f}{cov:>11.4f}")
        rows.append(dict(knob="capacity", value=C, windows=n,
                         balanced=acc, null_mean_margin=nm, coverage=cov))
    log("")

    r = pd.DataFrame(rows)
    r.round(4).to_csv(os.path.join(OUT, "pcd_null_dependence.csv"), index=False)

    log("=" * 78)
    log("INTERPRETATION")
    log("=" * 78)
    for knob in ("features", "participants", "capacity"):
        k = r[r.knob == knob]
        log(f"{knob:<14} coverage {k.coverage.min():.4f}-{k.coverage.max():.4f}"
            f"  (spread {k.coverage.max() - k.coverage.min():.4f})"
            f"   balanced accuracy {k.balanced.min():.4f}-{k.balanced.max():.4f}"
            f"  (spread {k.balanced.max() - k.balanced.min():.4f})")
    log("")
    cap = r[r.knob == "capacity"]
    if (cap.coverage.max() - cap.coverage.min()) > 0.05 and \
            (cap.balanced.max() - cap.balanced.min()) < 0.05:
        log("Capacity moves coverage while leaving achieved accuracy nearly")
        log("unchanged, which is the cleanest demonstration: the same task,")
        log("the same data and the same achieved performance give different")
        log("coverage because the null decoders fit permuted labels")
        log("differently. Coverage is therefore not a measure of decodability")
        log("on an absolute scale, and must not be compared across datasets")
        log("or across analysis settings.")
    else:
        log("Report the measured spreads as they stand; where coverage tracks")
        log("achieved accuracy the two cannot be separated by this design.")

    with open(os.path.join(OUT, "test37_pcd_null_dependence.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"\nWrote {OUT}\\pcd_null_dependence.csv")


if __name__ == "__main__":
    main()
