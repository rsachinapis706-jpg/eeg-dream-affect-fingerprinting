"""
Is dream affect subject-SPECIFIC rather than absent?

Everything so far tested CROSS-subject decoding and found nothing (0.30-0.39 vs
0.333 chance), using a pipeline validated at 0.88 on Sleep-EDF. But "no shared
code across people" is not the same as "no signal in anyone". If each person's
dream-affect signature is idiosyncratic, a within-subject model would find it
while a cross-subject model never could.

This is the cheap version of the few-shot personalisation experiment: if there is
no within-subject signal detectable with these features, a meta-learning model has
nothing to adapt to, and we learn that in minutes instead of days.

Protocol
--------
  within-subject : for each eligible subject, grouped k-fold over that subject's
                   own recordings. Folds are grouped by SEGMENT so epochs from one
                   recording never straddle the split -- the same leakage rule that
                   governs the cross-subject analysis.
  cross-subject  : leave-one-subject-out, for reference.
  null           : labels permuted at SEGMENT level within subject, whole
                   within-subject procedure re-run.

A subject is eligible with >= 2 classes and >= 6 recordings.

Also computes a personalisation curve: accuracy as a function of how many of the
subject's own recordings are available for training.
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
N_PERM = 200
MIN_SEGS = 6
MIN_CLASSES = 2
COLL = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def clf():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced"),
    )


def within_subject_bacc(X, y, seg, n_splits=3):
    """Grouped CV inside one subject. Returns balanced accuracy or None."""
    if len(np.unique(y)) < 2:
        return None
    n_groups = len(np.unique(seg))
    k = min(n_splits, n_groups)
    if k < 2:
        return None
    yt, yp = [], []
    try:
        splits = list(StratifiedGroupKFold(k, shuffle=True,
                                           random_state=SEED).split(X, y, seg))
    except ValueError:
        return None
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2 or len(te) == 0:
            continue
        yp.append(clf().fit(X[tr], y[tr]).predict(X[te]))
        yt.append(y[te])
    if not yt:
        return None
    return balanced_accuracy_score(np.concatenate(yt), np.concatenate(yp))


def permuted_bacc(X, y, seg, rng):
    """Shuffle labels across this subject's recordings, then repeat the CV."""
    seg_lab = pd.DataFrame({"s": seg, "y": y}).drop_duplicates("s")
    mapping = dict(zip(seg_lab.s.to_numpy(), rng.permutation(seg_lab.y.to_numpy())))
    yp = pd.Series(seg).map(mapping).to_numpy()
    return within_subject_bacc(X, yp, seg)


def main():
    ep = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    meta = ["segment_id", "subject", "oid", "epoch_idx", "n_epochs",
            "epochs_from_end", "E", "film", "stage", "has_text"]
    fc = [c for c in ep.columns if c not in meta]
    ep = ep.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = ep[ep.E.isin(COLL)].copy()
    d["y"] = d.E.map(COLL)

    log("=" * 72)
    log("WITHIN-SUBJECT vs CROSS-SUBJECT DREAM-AFFECT DECODING")
    log("=" * 72)
    log(f"epochs {len(d)} | subjects {d.subject.nunique()} | features {len(fc)}")
    log(f"eligibility: >= {MIN_CLASSES} classes and >= {MIN_SEGS} recordings")
    log(f"permutations per subject: {N_PERM} (labels shuffled across that")
    log("subject's own recordings)")
    log("")

    rows = []
    for subj, g in d.groupby("subject"):
        if g.segment_id.nunique() < MIN_SEGS or g.y.nunique() < MIN_CLASSES:
            continue
        X = g[fc].to_numpy(float)
        y = g.y.to_numpy()
        seg = g.segment_id.to_numpy()

        obs = within_subject_bacc(X, y, seg)
        if obs is None:
            continue

        nulls = Parallel(n_jobs=-1)(
            delayed(permuted_bacc)(X, y, seg, np.random.default_rng(SEED + 31 * i))
            for i in range(N_PERM))
        nulls = np.array([v for v in nulls if v is not None])
        if len(nulls) < 20:
            continue
        p = (np.sum(nulls >= obs) + 1) / (len(nulls) + 1)

        rows.append(dict(subject=subj, n_seg=g.segment_id.nunique(),
                         n_epoch=len(g), n_class=g.y.nunique(),
                         bacc=obs, null=nulls.mean(), p=p))

    res = pd.DataFrame(rows).sort_values("p")
    log(f"{'subj':<7}{'segs':>6}{'eps':>7}{'cls':>5}{'bacc':>9}{'null':>9}{'p':>9}")
    log("-" * 72)
    for _, r in res.iterrows():
        log(f"{r.subject:<7}{r.n_seg:>6}{r.n_epoch:>7}{r.n_class:>5}"
            f"{r.bacc:>9.4f}{r.null:>9.4f}{r.p:>9.4f}")
    log("")

    n = len(res)
    sig = int((res.p < 0.05).sum())
    expected = 0.05 * n
    log("=" * 72)
    log("SUMMARY")
    log("=" * 72)
    log(f"subjects tested                    : {n}")
    log(f"mean within-subject balanced acc.  : {res.bacc.mean():.4f}")
    log(f"mean permutation null              : {res.null.mean():.4f}")
    log(f"mean difference                    : {res.bacc.mean() - res.null.mean():+.4f}")
    log(f"cross-subject reference (LOSO)     : 0.3430")
    log("")
    log(f"subjects with p < 0.05             : {sig} / {n}")
    log(f"expected by chance at alpha = 0.05 : {expected:.1f}")
    log("")

    # group-level test: are within-subject scores above their own nulls overall?
    from scipy import stats
    w = stats.wilcoxon(res.bacc, res.null)
    log(f"Wilcoxon (bacc vs own null) across subjects: "
        f"W = {w.statistic:.1f}, p = {w.pvalue:.4f}")
    log("")

    if sig > expected * 2 and w.pvalue < 0.05:
        log(">> WITHIN-SUBJECT SIGNAL PRESENT. Dream affect appears subject-")
        log(">> specific rather than absent. Personalised / few-shot modelling")
        log(">> is justified and is the correct next step.")
    else:
        log(">> NO WITHIN-SUBJECT SIGNAL DETECTED. Significant subjects are at")
        log(">> or near the false-positive rate. Personalisation has nothing to")
        log(">> adapt to with these features; the negative result strengthens.")

    res.round(4).to_csv(os.path.join(OUT, "within_subject.csv"), index=False)
    with open(os.path.join(OUT, "test8_within_subject.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"Wrote {OUT}\\test8_within_subject.txt")


if __name__ == "__main__":
    main()
