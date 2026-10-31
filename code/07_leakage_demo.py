"""
The leakage demonstration.

Published DEED analyses report 92-94% accuracy for 3-class dream emotion. They cut
each recording into 20 s windows and then cross-validate over windows. Windows from
the same recording -- and the same subject -- therefore appear in both train and
test.

This script holds EVERYTHING constant (same data, same features, same classifier,
same folds count) and varies ONLY what is held out:

    random    - 5-fold over epochs, shuffled            <- what the papers effectively do
    segment   - 5-fold, epochs grouped by recording     <- no within-recording leakage
    subject   - 5-fold, epochs grouped by subject       <- no subject leakage

If accuracy collapses from `random` to `subject`, the published figures are
measuring recording/subject identity, not dream affect. The size of that collapse
is the result.

Run on both targets:
  - reported dream valence (the published task)
  - sleep stage (positive control; same leakage logic applies)
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
SEED = 20260727
N_SPLITS = 5

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def make_clf():
    # A flexible model is used deliberately: leakage is most visible with a model
    # that CAN memorise recording identity. A linear model would understate it.
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED
    )


def cv_score(X, y, groups, mode):
    if mode == "random":
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        split = splitter.split(X, y)
    else:
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        split = splitter.split(X, y, groups)

    yt, yp = [], []
    for tr, te in split:
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_clf().fit(X[tr], y[tr])
        yp.append(clf.predict(X[te]))
        yt.append(y[te])
    yt = np.concatenate(yt)
    yp = np.concatenate(yp)
    return accuracy_score(yt, yp), balanced_accuracy_score(yt, yp)


def run_target(ep, fc, y, name, majority):
    log(f"### {name}")
    log(f"epochs = {len(y)} | majority-class baseline = {majority:.4f}")
    log("")
    log(f"{'held out':<12}{'accuracy':>10}{'balanced':>11}   interpretation")
    log("-" * 68)

    X = ep[fc].to_numpy(float)
    rows = []
    for mode, grp, desc in [
        ("random", None, "epochs shuffled  <- published protocol"),
        ("segment", ep.segment_id.to_numpy(), "no within-recording leakage"),
        ("subject", ep.subject.to_numpy(), "no subject leakage  <- honest"),
    ]:
        acc, bacc = cv_score(X, y, grp, mode)
        log(f"{mode:<12}{acc:>10.4f}{bacc:>11.4f}   {desc}")
        rows.append(dict(target=name, held_out=mode, accuracy=acc,
                         balanced_accuracy=bacc, majority=majority))
    log("")
    drop = rows[0]["accuracy"] - rows[2]["accuracy"]
    log(f">> accuracy drop, random -> subject : {drop:+.4f} "
        f"({drop / max(rows[0]['accuracy'], 1e-9):.1%} of the reported figure)")
    log("")
    return rows


def main():
    path = os.path.join(OUT, "features_epoch_filtered.parquet")
    if not os.path.exists(path):
        path = os.path.join(OUT, "features_epoch.parquet")
        log("[note] filtered features not found; using unfiltered set")
    ep = pd.read_parquet(path)

    meta = ["segment_id", "subject", "oid", "epoch_idx", "n_epochs",
            "epochs_from_end", "E", "film", "stage", "has_text"]
    fc = [c for c in ep.columns if c not in meta]
    ep = ep.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)

    log("=" * 68)
    log("LEAKAGE DEMONSTRATION")
    log("=" * 68)
    log(f"features file : {os.path.basename(path)}")
    log(f"epochs        : {len(ep)} | segments: {ep.segment_id.nunique()} "
        f"| subjects: {ep.subject.nunique()}")
    log(f"classifier    : RandomForest(300) | {N_SPLITS}-fold | seed {SEED}")
    log("")

    all_rows = []

    coll = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
    d = ep[ep.E.isin(coll)]
    y = d.E.map(coll).to_numpy()
    maj = pd.Series(y).value_counts().max() / len(y)
    all_rows += run_target(d, fc, y, "Reported dream valence (3-class)", maj)

    d2 = ep[ep.stage.isin(["W", "N1", "N2"])]
    y2 = d2.stage.to_numpy()
    maj2 = pd.Series(y2).value_counts().max() / len(y2)
    all_rows += run_target(d2, fc, y2, "Sleep stage (W/N1/N2)", maj2)

    pd.DataFrame(all_rows).round(4).to_csv(
        os.path.join(OUT, "leakage_results.csv"), index=False)
    with open(os.path.join(OUT, "test4_leakage.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"Wrote {OUT}\\test4_leakage.txt and leakage_results.csv")


if __name__ == "__main__":
    main()
