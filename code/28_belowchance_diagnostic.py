"""
Why is subject-wise balanced accuracy 0.3070, below the 0.3333 chance level?

Two explanations are possible and they have different consequences.

  (a) The classifier is misconfigured for this setting. Class weights are
      computed on the training participants, whose label distribution differs
      from the held-out participant's. A model with nothing to learn then
      systematically over-predicts whichever class the training set weighted
      up, which can push balanced accuracy below chance. This is a reporting
      artefact and should be corrected.

  (b) The data genuinely produce below-chance transfer. This would be a real,
      if uninteresting, property.

The distinction matters because (a) is fixable and (b) is not. Reference
strategies are included so the observed value can be placed against what a model
with no information can achieve by construction:

  uniform random          expected balanced accuracy 1/3
  constant (majority)     balanced accuracy exactly 1/3
  stratified random       expected balanced accuracy 1/3

If an unweighted classifier lands at ~1/3 while the weighted one sits below it,
explanation (a) holds and the unweighted figure is the honest one to report.
No configuration is selected on the basis of which gives a higher number; both
are reported.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (balanced_accuracy_score, accuracy_score,
                             confusion_matrix)

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
SEED = 20260727
COLL = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def loso(X, y, subj, make):
    yt, yp = [], []
    for s in np.unique(subj):
        te = subj == s
        tr = ~te
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            continue
        m = make().fit(X[tr], y[tr])
        yp.append(m.predict(X[te]))
        yt.append(y[te])
    return np.concatenate(yt), np.concatenate(yp)


def main():
    ep = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    meta = ["segment_id", "subject", "oid", "epoch_idx", "n_epochs",
            "epochs_from_end", "E", "film", "stage", "has_text"]
    fc = [c for c in ep.columns if c not in meta]
    ep = ep.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = ep[ep.E.isin(COLL)]
    X = d[fc].to_numpy(float)
    y = d.E.map(COLL).to_numpy()
    subj = d.subject.astype(str).to_numpy()

    log("=" * 74)
    log("WHY IS SUBJECT-WISE BALANCED ACCURACY BELOW CHANCE?")
    log("=" * 74)
    log(f"windows {len(y)} | participants {len(np.unique(subj))}")
    log(f"class counts {dict(pd.Series(y).value_counts())}")
    log(f"chance 0.3333 | majority baseline "
        f"{pd.Series(y).value_counts().max()/len(y):.4f}")
    log("")

    configs = [
        ("RF, class_weight=balanced",
         lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                        n_jobs=-1, random_state=SEED,
                                        class_weight="balanced")),
        ("RF, no class weighting",
         lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                        n_jobs=-1, random_state=SEED)),
        ("LogReg, class_weight=balanced",
         lambda: make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=1000, C=0.1,
                                                  class_weight="balanced"))),
        ("LogReg, no class weighting",
         lambda: make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=1000, C=0.1))),
        ("Dummy: most frequent",
         lambda: DummyClassifier(strategy="most_frequent")),
        ("Dummy: stratified",
         lambda: DummyClassifier(strategy="stratified", random_state=SEED)),
        ("Dummy: uniform",
         lambda: DummyClassifier(strategy="uniform", random_state=SEED)),
    ]

    log(f"{'configuration':<32}{'accuracy':>10}{'balanced':>11}")
    log("-" * 74)
    rows = []
    for name, mk in configs:
        yt, yp = loso(X, y, subj, mk)
        acc = accuracy_score(yt, yp)
        bacc = balanced_accuracy_score(yt, yp)
        log(f"{name:<32}{acc:>10.4f}{bacc:>11.4f}")
        rows.append(dict(config=name, accuracy=acc, balanced=bacc))
        if name.startswith("RF, class_weight"):
            cm_w = confusion_matrix(yt, yp, labels=["neg", "neu", "pos"])
        if name.startswith("RF, no class"):
            cm_u = confusion_matrix(yt, yp, labels=["neg", "neu", "pos"])
    log("")

    log("confusion, RF with class weighting (rows=true, order neg/neu/pos):")
    log(str(cm_w))
    log("")
    log("confusion, RF without class weighting:")
    log(str(cm_u))
    log("")

    res = pd.DataFrame(rows)
    w = float(res[res.config == "RF, class_weight=balanced"].balanced.iloc[0])
    u = float(res[res.config == "RF, no class weighting"].balanced.iloc[0])
    dummies = res[res.config.str.startswith("Dummy")].balanced

    log("=" * 74)
    log("INTERPRETATION")
    log("=" * 74)
    log(f"weighted   : {w:.4f}")
    log(f"unweighted : {u:.4f}")
    log(f"dummy range: {dummies.min():.4f} to {dummies.max():.4f}")
    log("")
    if u >= 0.3333 > w:
        log("Explanation (a). Class weighting computed on the training")
        log("participants transfers poorly to a held-out participant with a")
        log("different label distribution, pushing balanced accuracy below")
        log("chance. Without weighting the classifier sits at chance, which is")
        log("the correct description of a model that has learned nothing.")
        log("Report both; the substantive conclusion is unchanged.")
    elif abs(u - 1 / 3) < 0.02 and abs(w - 1 / 3) < 0.02:
        log("Both configurations sit at chance. The below-chance value is")
        log("sampling noise around 1/3 and needs no separate explanation.")
    else:
        log("Neither configuration reaches chance. This is a property of the")
        log("data under this protocol, not of the class weighting. Report as")
        log("observed.")

    res.round(4).to_csv(os.path.join(OUT, "belowchance_diagnostic.csv"),
                        index=False)
    with open(os.path.join(OUT, "test23_belowchance.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"\nWrote {OUT}\\test23_belowchance.txt")


if __name__ == "__main__":
    main()
