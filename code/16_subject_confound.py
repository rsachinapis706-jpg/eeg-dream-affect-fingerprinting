"""
Is "dream-emotion decoding" partly subject fingerprinting?

We measured that shuffling windows instead of grouping by subject inflates
accuracy by 17 points (0.4025 -> 0.5732). That is an effect. This script asks for
the MECHANISM.

Hypothesis
----------
EEG is known to carry strong individual identity. Under a window-shuffled split,
windows from the same recording appear in train and test, so a classifier can
recognise WHO a window came from. And because each subject has a skewed dream-
affect label distribution, knowing the subject already predicts the label. The
model never needs affect-related neural information at all.

If true, three things must hold:
  1. subject identity is highly decodable from these features
  2. an oracle that knows only the subject id -- no EEG -- reaches roughly the
     inflated accuracy
  3. subject and label share substantial mutual information

Test 3 measures the confound directly; test 2 is the key quantity, because it
puts a number on how much of the reported performance needs no neural signal
whatsoever.

Note on test 2: it is deliberately an ORACLE, not a model. It receives the true
subject id and predicts that subject's majority class. It therefore reports the
ceiling attainable by pure fingerprinting.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             mutual_info_score)

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
SEED = 20260727
COLL = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def cv_score(X, y, groups, mode, model="rf", n_splits=5):
    if model == "rf":
        mk = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                            n_jobs=-1, random_state=SEED)
    else:
        mk = lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced"))

    if mode == "random":
        sp = StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(X, y)
    else:
        sp = StratifiedGroupKFold(n_splits, shuffle=True,
                                  random_state=SEED).split(X, y, groups)
    yt, yp = [], []
    for tr, te in sp:
        if len(np.unique(y[tr])) < 2:
            continue
        yp.append(mk().fit(X[tr], y[tr]).predict(X[te]))
        yt.append(y[te])
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return accuracy_score(yt, yp), balanced_accuracy_score(yt, yp)


def main():
    ep = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    meta = ["segment_id", "subject", "oid", "epoch_idx", "n_epochs",
            "epochs_from_end", "E", "film", "stage", "has_text"]
    fc = [c for c in ep.columns if c not in meta]
    ep = ep.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = ep[ep.E.isin(COLL)].copy()
    d["y"] = d.E.map(COLL)

    X = d[fc].to_numpy(float)
    y = d.y.to_numpy()
    subj = d.subject.astype(str).to_numpy()
    seg = d.segment_id.to_numpy()

    log("=" * 74)
    log("IS DREAM-EMOTION DECODING PARTLY SUBJECT FINGERPRINTING?")
    log("=" * 74)
    log(f"windows {len(y)} | subjects {len(np.unique(subj))} | features {len(fc)}")
    log(f"reference: affect accuracy was 0.5732 (window-shuffled) vs "
        f"0.4025 (subject-wise)")
    log("")

    # ---- 1. how identifiable are subjects from these features? --------
    log("-- 1. Subject identity decodability (34-way) --")
    acc_r, bacc_r = cv_score(X, subj, None, "random")
    log(f"window-shuffled split : accuracy {acc_r:.4f}  balanced {bacc_r:.4f}")
    acc_s, bacc_s = cv_score(X, subj, seg, "grouped")
    log(f"recording-grouped     : accuracy {acc_s:.4f}  balanced {bacc_s:.4f}")
    log(f"chance (1/{len(np.unique(subj))})            : "
        f"{1/len(np.unique(subj)):.4f}")
    log("")

    # ---- 2. oracle: predict label from subject id ALONE, no EEG ------
    log("-- 2. Oracle using ONLY subject id (no EEG at all) --")
    maj = d.groupby("subject").y.agg(lambda s: s.value_counts().idxmax())
    y_oracle = d.subject.map(maj).to_numpy()
    o_acc = accuracy_score(y, y_oracle)
    o_bacc = balanced_accuracy_score(y, y_oracle)
    log(f"accuracy from subject id alone : {o_acc:.4f}")
    log(f"balanced accuracy              : {o_bacc:.4f}")
    log(f"observed window-shuffled affect : 0.5732")
    log(f"observed subject-wise affect    : 0.4025")
    log("")
    log(f">> pure fingerprinting explains {o_acc/0.5732:.1%} of the "
        f"window-shuffled accuracy")
    log("")

    # ---- 3. mutual information between subject and label -------------
    log("-- 3. Subject/label confounding --")
    mi = mutual_info_score(subj, y)
    h_y = mutual_info_score(y, y)
    log(f"I(subject; label)  = {mi:.4f} nats")
    log(f"H(label)           = {h_y:.4f} nats")
    log(f"normalised         = {mi/h_y:.4f}  "
        f"(fraction of label entropy explained by subject identity)")
    log("")

    per_subj = d.groupby("subject").y.value_counts(normalize=True).unstack().fillna(0)
    dominance = per_subj.max(axis=1)
    log(f"per-subject majority-class share: mean {dominance.mean():.3f}, "
        f"median {dominance.median():.3f}, max {dominance.max():.3f}")
    log(f"subjects whose windows are >70% one class: "
        f"{int((dominance > 0.7).sum())}/{len(dominance)}")
    log("")

    # ---- 4. does label survive once subject is known? ----------------
    log("-- 4. Affect decoding WITHIN the most-balanced subjects only --")
    balanced_subj = dominance[dominance < 0.6].index.tolist()
    if len(balanced_subj) >= 5:
        mask = d.subject.isin(balanced_subj).to_numpy()
        a, b = cv_score(X[mask], y[mask], subj[mask], "grouped")
        log(f"subjects with no dominant class ({len(balanced_subj)}): "
            f"accuracy {a:.4f}  balanced {b:.4f}")
        log("(if fingerprinting drove the earlier numbers, this should sit at chance)")
    else:
        log(f"only {len(balanced_subj)} subjects are class-balanced -- not run")
    log("")

    log("=" * 74)
    log("INTERPRETATION")
    log("=" * 74)
    if o_acc >= 0.50 and acc_r >= 0.80:
        log("Subjects are highly identifiable from these features, and knowing")
        log("the subject alone -- with no EEG -- recovers most of the accuracy")
        log("reported under window-shuffled validation. The mechanism behind the")
        log("inflation is subject fingerprinting combined with per-subject label")
        log("imbalance, not dream-related neural information.")
    else:
        log("Fingerprinting does not fully account for the inflation; report the")
        log("measured components without over-attributing the mechanism.")

    pd.DataFrame([dict(subject_id_acc_random=acc_r, subject_id_acc_grouped=acc_s,
                       oracle_acc=o_acc, oracle_bacc=o_bacc,
                       mi_subject_label=mi, mi_normalised=mi / h_y)]).round(4)\
        .to_csv(os.path.join(OUT, "subject_confound.csv"), index=False)
    with open(os.path.join(OUT, "test15_subject_confound.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"Wrote {OUT}\\test15_subject_confound.txt")


if __name__ == "__main__":
    main()
