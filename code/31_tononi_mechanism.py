"""
Independent test of the fingerprinting mechanism on Tononi Serial Awakenings.

Registered prediction, fixed before this script was run
------------------------------------------------------
The mechanism holds that window-level partitioning inflates accuracy because it
allows a classifier to recognise the participant, and that participant identity
predicts the label only where labels are concentrated within participants.

    DEED    within-participant label dominance 0.750, 38% of participants
            above 70%      ->  observed inflation +17.1 accuracy points
    Tononi  dominance 0.536, 16.7% above 70%
                           ->  prediction: SUBSTANTIALLY LESS inflation

Inflation in Tononi comparable to DEED's would falsify the mechanism.

The protocol mirrors 07_leakage_demo.py exactly -- same classifier, same fold
count, same three partitioning rules -- so the two datasets are directly
comparable. Identity decoding and the identity-only oracle are also repeated,
since those are the quantities the mechanism is built on.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             mutual_info_score)

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
SEED = 20260727
N_SPLITS = 5

# DEED reference values, from leakage_results.csv and subject_confound.csv
DEED = dict(random=0.5732, recording=0.4243, subject=0.4025,
            dominance=0.750, over70=0.38, id_random=0.8037,
            id_grouped=0.7313, oracle=0.5465, mi_norm=0.1624)

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def mk():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                  n_jobs=-1, random_state=SEED)


def cv(X, y, groups, mode):
    if mode == "random":
        sp = StratifiedKFold(N_SPLITS, shuffle=True,
                             random_state=SEED).split(X, y)
    else:
        sp = StratifiedGroupKFold(N_SPLITS, shuffle=True,
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
    d = pd.read_parquet(os.path.join(OUT, "tononi_features.parquet"))
    fc = [c for c in d.columns if c.startswith("ch")]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)

    X = d[fc].to_numpy(float)
    y = d.dreaming.astype(str).to_numpy()
    subj = d.subject.astype(str).to_numpy()
    rec = d.recording.astype(str).to_numpy()

    n_sub = len(np.unique(subj))
    maj = pd.Series(y).value_counts().max() / len(y)

    log("=" * 76)
    log("TONONI SERIAL AWAKENINGS -- MECHANISM TEST")
    log("=" * 76)
    log(f"windows {len(y)} | participants {n_sub} | recordings "
        f"{len(np.unique(rec))} | features {len(fc)}")
    log(f"classes {dict(pd.Series(y).value_counts())}")
    log(f"majority baseline {maj:.4f} | chance {1/len(np.unique(y)):.4f}")
    log("")

    # ---- label concentration, the mechanism's precondition -----------
    rl = d.groupby(["subject", "recording"]).dreaming.first()
    dom = rl.groupby("subject").agg(
        lambda x: x.value_counts(normalize=True).max())
    log("-- precondition --")
    log(f"{'':<34}{'Tononi':>10}{'DEED':>10}")
    log(f"{'median label dominance':<34}{dom.median():>10.3f}"
        f"{DEED['dominance']:>10.3f}")
    log(f"{'participants >70% one class':<34}{dom.gt(0.7).mean():>10.3f}"
        f"{DEED['over70']:>10.3f}")
    log("")

    # ---- the leakage ladder -----------------------------------------
    log("-- partitioning rule (RandomForest, 5-fold; mirrors DEED) --")
    log(f"{'held out':<14}{'accuracy':>10}{'balanced':>11}{'DEED acc':>11}")
    log("-" * 76)
    res = {}
    for mode, grp, dref in [("random", None, DEED["random"]),
                            ("recording", rec, DEED["recording"]),
                            ("subject", subj, DEED["subject"])]:
        a, b = cv(X, y, grp, mode if mode == "random" else "grouped")
        res[mode] = a
        log(f"{mode:<14}{a:>10.4f}{b:>11.4f}{dref:>11.4f}")
    log("")

    infl_t = res["random"] - res["subject"]
    infl_d = DEED["random"] - DEED["subject"]
    log(f"{'inflation, random - subject':<34}{infl_t:>+10.4f}{infl_d:>+10.4f}")
    log("")

    # ---- identity decodability and the oracle ------------------------
    log("-- fingerprinting components --")
    ida, _ = cv(X, subj, None, "random")
    idg, _ = cv(X, subj, rec, "grouped")
    majcls = d.groupby("subject").dreaming.agg(
        lambda s: s.value_counts().idxmax())
    oracle = accuracy_score(y, d.subject.map(majcls).astype(str).to_numpy())
    mi = mutual_info_score(subj, y)
    hy = mutual_info_score(y, y)

    log(f"{'':<34}{'Tononi':>10}{'DEED':>10}")
    log(f"{'identity acc, window-shuffled':<34}{ida:>10.4f}"
        f"{DEED['id_random']:>10.4f}")
    log(f"{'identity acc, recording-grouped':<34}{idg:>10.4f}"
        f"{DEED['id_grouped']:>10.4f}")
    log(f"{'chance for identity':<34}{1/n_sub:>10.4f}{1/34:>10.4f}")
    log(f"{'oracle from identity alone':<34}{oracle:>10.4f}"
        f"{DEED['oracle']:>10.4f}")
    log(f"{'I(subject;label)/H(label)':<34}{mi/hy:>10.4f}"
        f"{DEED['mi_norm']:>10.4f}")
    log("")

    # ---- verdict -----------------------------------------------------
    log("=" * 76)
    log("VERDICT")
    log("=" * 76)
    ratio = infl_t / infl_d if infl_d else float("nan")
    log(f"Tononi inflation {infl_t:+.4f} vs DEED {infl_d:+.4f} "
        f"({ratio:.2f}x DEED)")
    log("")
    if infl_t < 0.6 * infl_d:
        log("PREDICTION SUPPORTED. The corpus with more balanced")
        log("within-participant labels shows markedly less inflation under")
        log("window-level partitioning, as the mechanism requires.")
    elif infl_t > 0.9 * infl_d:
        log("PREDICTION NOT SUPPORTED. Inflation is comparable despite far")
        log("more balanced labels. Label imbalance is therefore not necessary")
        log("for the effect, and the mechanism as stated is incomplete.")
        log("Section 6.2 must be revised rather than reinforced.")
    else:
        log("PARTIALLY SUPPORTED. Inflation is reduced but not in proportion")
        log("to the difference in label concentration. Report the observed")
        log("values without asserting a quantitative correspondence.")

    pd.DataFrame([dict(
        dataset="Tononi", windows=len(y), participants=n_sub,
        dominance=float(dom.median()), over70=float(dom.gt(0.7).mean()),
        acc_random=res["random"], acc_recording=res["recording"],
        acc_subject=res["subject"], inflation=infl_t,
        identity_random=ida, identity_grouped=idg, oracle=oracle,
        mi_normalised=mi / hy)]).round(4).to_csv(
        os.path.join(OUT, "tononi_mechanism.csv"), index=False)
    with open(os.path.join(OUT, "test24_tononi_mechanism.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"\nWrote {OUT}\\test24_tononi_mechanism.txt")


if __name__ == "__main__":
    main()
