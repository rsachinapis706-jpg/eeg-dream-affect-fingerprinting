"""
Subject-wise decoding: does the EEG track the FILM (objective) or the REPORT
(subjective)?

Protocol
--------
  - Leave-one-subject-out over the 34 three-digit subject IDs. `S0391` and `S0392`
    are the SAME person on different nights and must never straddle a fold.
  - Primary unit of analysis is the SEGMENT (epoch features averaged within
    segment). Epoch-level fitting would inflate n by ~14x through
    pseudoreplication, which is the error that inflates the published DEED numbers.
  - Predictions from all folds are pooled, then scored once. Balanced accuracy is
    primary because both label sets are imbalanced.
  - Every headline number gets a permutation null with labels shuffled at the
    level the label actually varies: film is assigned per NIGHT, report per
    SEGMENT. Shuffling is always done within subject.

Positive control
----------------
Sleep stage is decoded first. If these features cannot separate W/N1/N2, they are
broken and no null result below is interpretable.
"""
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
SEED = 20260727
RNG = np.random.default_rng(SEED)
N_PERM = 200          # p resolution 1/201 ~ 0.005; enough to separate p<.05 from null
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


def subject_normalize(X, groups):
    """Z-score each feature within subject.

    Cross-subject EEG amplitude varies by large factors (impedance, skull, montage),
    so under LOSO an unnormalised model spends capacity on between-subject offsets
    instead of within-subject state differences.

    This uses only each subject's own feature distribution and NO label information,
    so it leaks nothing across the LOSO boundary -- the held-out subject is
    normalised using only their own data.
    """
    Xn = np.empty_like(X, dtype=float)
    for s in np.unique(groups):
        m = groups == s
        blk = X[m]
        mu = blk.mean(axis=0)
        sd = blk.std(axis=0)
        sd[sd < 1e-12] = 1.0
        Xn[m] = (blk - mu) / sd
    return Xn


def loso_predict(X, y, groups):
    """Pooled leave-one-subject-out predictions."""
    pred = np.empty(len(y), dtype=object)
    for s in np.unique(groups):
        te = groups == s
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            pred[te] = pd.Series(y[tr]).mode()[0]
            continue
        clf = make_clf().fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return pred


def evaluate(X, y, groups, perm_unit, name, n_perm=N_PERM):
    """perm_unit: array parallel to y giving the unit within which the label is constant."""
    pred = loso_predict(X, y, groups)
    bacc = balanced_accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")

    vc = pd.Series(y).value_counts()
    majority = vc.max() / len(y)

    # permutation null: shuffle labels across units, within subject
    units = pd.DataFrame({"u": perm_unit, "s": groups, "y": y}).drop_duplicates("u")
    by_subj = [(g.u.to_numpy(), g.y.to_numpy()) for _, g in units.groupby("s")]
    pu = pd.Series(perm_unit)

    def one_perm(seed):
        rng = np.random.default_rng(seed)
        mapping = {}
        for us, ys in by_subj:
            mapping.update(dict(zip(us, rng.permutation(ys))))
        yp = pu.map(mapping).to_numpy()
        return balanced_accuracy_score(yp, loso_predict(X, yp, groups))

    null = np.asarray(
        Parallel(n_jobs=N_JOBS, verbose=0)(
            delayed(one_perm)(SEED + 1000 * i) for i in range(n_perm)
        )
    )
    p = (np.sum(null >= bacc) + 1) / (n_perm + 1)

    log(f"-- {name} --")
    log(f"n = {len(y)} segments | {len(np.unique(groups))} subjects | "
        f"{len(vc)} classes {dict(vc)}")
    log(f"balanced accuracy : {bacc:.4f}")
    log(f"macro F1          : {f1:.4f}")
    log(f"majority baseline : {majority:.4f}   chance (1/k): {1/len(vc):.4f}")
    log(f"permutation null  : {null.mean():.4f} +/- {null.std():.4f}  "
        f"(95th pct {np.percentile(null, 95):.4f})")
    log(f"permutation p     : {p:.4f}   [{n_perm} permutations]")
    log("confusion matrix (rows=true, cols=pred), labels " + str(list(np.unique(y))))
    log(str(confusion_matrix(y, pred, labels=np.unique(y))))
    log("")
    return dict(name=name, n=len(y), bacc=bacc, f1=f1, majority=majority,
                null_mean=null.mean(), p=p)


def main():
    ep = pd.read_parquet(os.path.join(OUT, "features_epoch.parquet"))
    meta = ["segment_id", "subject", "oid", "epoch_idx", "n_epochs",
            "epochs_from_end", "E", "film", "stage", "has_text"]
    feat_cols = [c for c in ep.columns if c not in meta]

    # segment level = mean of its epochs
    seg = ep.groupby("segment_id", as_index=False).agg(
        {**{c: "mean" for c in feat_cols},
         "subject": "first", "oid": "first", "E": "first",
         "film": "first", "stage": "first", "has_text": "first"}
    )
    seg = seg.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols)

    log("=" * 68)
    log("SUBJECT-WISE DECODING (LOSO, segment level)")
    log("=" * 68)
    log(f"segments: {len(seg)} | subjects: {seg.subject.nunique()} | "
        f"features: {len(feat_cols)}")
    log(f"permutations per test: {N_PERM}")
    log("")

    X_raw = seg[feat_cols].to_numpy(dtype=float)
    g_all = seg.subject.to_numpy()

    # Within-subject z-scoring. Applied to every test below, including the
    # positive control, so all results are directly comparable.
    X_all = subject_normalize(X_raw, g_all)
    log("Feature normalisation: within-subject z-score (label-free, no LOSO leakage)")
    log("")

    results = []

    # ---- positive control: sleep stage -------------------------------
    ctrl = seg[seg.stage.isin(["W", "N1", "N2"])]
    m = seg.stage.isin(["W", "N1", "N2"]).to_numpy()
    results.append(evaluate(
        X_all[m], ctrl.stage.to_numpy(), g_all[m],
        perm_unit=ctrl.segment_id.to_numpy(),
        name="POSITIVE CONTROL: sleep stage (W/N1/N2)",
    ))

    # ---- target A: film condition (objective, night-level) -----------
    a = seg.film.notna().to_numpy()
    sa = seg[a]
    results.append(evaluate(
        X_all[a], sa.film.to_numpy(), g_all[a],
        perm_unit=sa.oid.to_numpy(),          # film is constant within night
        name="TARGET A: pre-sleep FILM valence (objective)",
    ))

    # ---- target B: reported dream valence (subjective, segment-level)-
    coll = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
    b = seg.E.isin(coll).to_numpy()
    sb = seg[b]
    yb = sb.E.map(coll).to_numpy()
    results.append(evaluate(
        X_all[b], yb, g_all[b],
        perm_unit=sb.segment_id.to_numpy(),
        name="TARGET B: reported DREAM valence (subjective)",
    ))

    # ---- target B controlled for stage -------------------------------
    bs = seg[b & seg.stage.isin(["N1", "N2"]).to_numpy()]
    mbs = (b & seg.stage.isin(["N1", "N2"]).to_numpy())
    if len(bs) > 60:
        results.append(evaluate(
            X_all[mbs], bs.E.map(coll).to_numpy(), g_all[mbs],
            perm_unit=bs.segment_id.to_numpy(),
            name="TARGET B (stage-controlled: N1+N2 only)",
        ))

    log("=" * 68)
    log("SUMMARY")
    log("=" * 68)
    summary = pd.DataFrame(results)[
        ["name", "n", "bacc", "majority", "null_mean", "p"]
    ].round(4)
    log(summary.to_string(index=False))

    summary.to_csv(os.path.join(OUT, "decoding_summary.csv"), index=False)
    with open(os.path.join(OUT, "test2_decoding.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log("")
    log(f"Wrote {OUT}\\test2_decoding.txt")


if __name__ == "__main__":
    main()
