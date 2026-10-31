"""
Which pipeline ordering produces the published accuracy?

Our faithful replication of the published feature families reached only ~0.41-0.47
under random splits, far from the reported 92-94%. So either the features differ in
some detail we cannot see, or the reported figures arise from the ORDER in which
standard steps are applied.

Two orderings are extremely common in the applied EEG literature and both leak:

  * feature selection (e.g. ReliefF) fitted on the FULL dataset, then cross-validate
      -> the selector has already seen the test labels
  * SMOTE oversampling applied BEFORE splitting
      -> synthetic training points are interpolated from test samples

This script holds data, features and classifier fixed and varies ONLY the ordering:

  A  nested_subject     selection+SMOTE inside each fold, subject-wise split   <- correct
  B  nested_random      selection+SMOTE inside each fold, random split
  C  preselect_random   selection on ALL data, then random split
  D  presmote_random    SMOTE on ALL data, then random split
  E  both_random        selection AND SMOTE on all data, then random split

If E approaches the published range while A sits near chance, the mechanism is
identified: the reported performance is a property of pipeline ordering, not of
dream-related neural signal. This is a claim about METHOD, not about authors --
these orderings are widespread and usually applied in good faith.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
SEED = 20260727
K_BEST = 40
N_SPLITS = 5

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def get_clf(name):
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=5)
    if name == "SVM":
        return SVC(C=1.0, gamma="scale")
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                  n_jobs=-1, random_state=SEED)


def evaluate(X, y, groups, mode, clf_name):
    """mode in {nested_subject, nested_random, preselect_random,
                presmote_random, both_random}"""
    Xw, yw, gw = X.copy(), y.copy(), groups.copy()

    # ---- leaky preprocessing applied to the WHOLE dataset ------------
    if mode in ("preselect_random", "both_random"):
        sel = SelectKBest(f_classif, k=min(K_BEST, Xw.shape[1])).fit(
            StandardScaler().fit_transform(Xw), yw)
        Xw = Xw[:, sel.get_support()]
    if mode in ("presmote_random", "both_random"):
        Xw, yw = SMOTE(random_state=SEED).fit_resample(Xw, yw)
        gw = np.arange(len(yw))          # groups meaningless after synthesis

    subject_split = mode == "nested_subject"
    if subject_split:
        splits = StratifiedGroupKFold(N_SPLITS, shuffle=True,
                                      random_state=SEED).split(Xw, yw, gw)
    else:
        splits = StratifiedKFold(N_SPLITS, shuffle=True,
                                 random_state=SEED).split(Xw, yw)

    yt, yp = [], []
    for tr, te in splits:
        Xtr, ytr = Xw[tr], yw[tr]
        Xte = Xw[te]
        if len(np.unique(ytr)) < 2:
            continue

        # ---- correct ordering: everything fitted on TRAIN only ------
        if mode in ("nested_subject", "nested_random"):
            sc = StandardScaler().fit(Xtr)
            sel = SelectKBest(f_classif, k=min(K_BEST, Xtr.shape[1])).fit(
                sc.transform(Xtr), ytr)
            Xtr = sel.transform(sc.transform(Xtr))
            Xte = sel.transform(sc.transform(Xte))
            Xtr, ytr = SMOTE(random_state=SEED).fit_resample(Xtr, ytr)
        else:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

        yp.append(get_clf(clf_name).fit(Xtr, ytr).predict(Xte))
        yt.append(yw[te])

    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return accuracy_score(yt, yp), balanced_accuracy_score(yt, yp)


def main():
    f_non = os.path.join(OUT, "features_nonlinear.parquet")
    f_spec = os.path.join(OUT, "features_epoch_filtered.parquet")

    sets = {}
    if os.path.exists(f_non):
        d = pd.read_parquet(f_non)
        sets["nonlinear (DWT+EMD entropy, 240)"] = (
            d, [c for c in d.columns if c.startswith("f")])
    if os.path.exists(f_spec):
        d = pd.read_parquet(f_spec)
        meta = ["segment_id", "subject", "oid", "epoch_idx", "n_epochs",
                "epochs_from_end", "E", "film", "stage", "has_text"]
        sets["spectral (108)"] = (d, [c for c in d.columns if c not in meta])

    log("=" * 76)
    log("LEAKAGE MECHANISM: WHICH PIPELINE ORDERING REPRODUCES ~92%?")
    log("=" * 76)
    log(f"feature selection: SelectKBest(f_classif, k={K_BEST}) | "
        f"oversampling: SMOTE | folds: {N_SPLITS}")
    log("published: Mahmoudi 2025 = 0.9383 | Mojdehbar 2026 = 0.9233")
    log("")

    coll = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
    rows = []

    for set_name, (df, fc) in sets.items():
        d = df.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
        d = d[d.E.isin(coll)]
        X = d[fc].to_numpy(float)
        y = d.E.map(coll).to_numpy()
        g = d.subject.astype(str).to_numpy()
        maj = pd.Series(y).value_counts().max() / len(y)

        log(f"### {set_name}   n={len(y)} epochs, {len(np.unique(g))} subjects, "
            f"majority={maj:.4f}")
        log(f"{'ordering':<20}{'clf':<6}{'accuracy':>10}{'balanced':>11}   note")
        log("-" * 76)
        for mode, note in [
            ("nested_subject",   "correct: nothing leaks"),
            ("nested_random",    "random split, no preprocessing leak"),
            ("preselect_random", "feature selection on all data"),
            ("presmote_random",  "SMOTE before split"),
            ("both_random",      "selection + SMOTE on all data"),
        ]:
            for clf_name in ["KNN", "SVM", "RF"]:
                acc, bacc = evaluate(X, y, g, mode, clf_name)
                log(f"{mode:<20}{clf_name:<6}{acc:>10.4f}{bacc:>11.4f}   {note}")
                rows.append(dict(features=set_name, ordering=mode, clf=clf_name,
                                 accuracy=acc, balanced_accuracy=bacc,
                                 majority=maj))
            log("")
        log("")

    res = pd.DataFrame(rows)
    res.round(4).to_csv(os.path.join(OUT, "leakage_mechanism.csv"), index=False)

    correct = res[res.ordering == "nested_subject"].accuracy.max()
    leaky = res[res.ordering == "both_random"].accuracy.max()
    log("=" * 76)
    log(f"best CORRECT (nested, subject-wise) : {correct:.4f}")
    log(f"best LEAKY   (selection+SMOTE, random): {leaky:.4f}")
    log(f"inflation                            : {leaky - correct:+.4f}")
    log("")
    if leaky >= 0.85:
        log(">> MECHANISM IDENTIFIED. Applying feature selection and/or")
        log(">> oversampling before cross-validation reproduces the published")
        log(">> range from data that carries no subject-generalisable signal.")
    else:
        log(f">> Leaky ordering reaches {leaky:.3f}, still short of ~0.92.")
        log(">> Ordering alone does not account for the published figures;")
        log(">> report as an unexplained reproduction failure.")

    with open(os.path.join(OUT, "test7_mechanism.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"Wrote {OUT}\\test7_mechanism.txt")


if __name__ == "__main__":
    main()
