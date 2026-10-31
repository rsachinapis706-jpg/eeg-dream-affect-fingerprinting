"""
Re-establish the positive control on Sleep-EDF Expanded (100 subjects).

The positive control is the load-bearing result of this project: it is what makes
every null interpretable. Until now it rested on 8 recordings from the 1.0.0
release, which is a fair target for reviewer criticism. This repeats it on 197
recordings from 100 subjects using identical feature definitions and the identical
leave-one-subject-out protocol.

Two tasks are run:

  W/N1/N2   the 3-class task used throughout the project, so the new number is
            directly comparable with the 8-subject control (0.8831) and with
            CBraMod (0.8217)

  5-class   W/N1/N2/N3/REM, the standard sleep-staging benchmark, so the pipeline
            can be compared against the published literature -- e.g. SIMSleepSM
            reports 0.862 accuracy on Sleep-EDF-78

Grouping is on the cohort-prefixed subject id (SC00, ST01, ...). Sleep-cassette
and sleep-telemetry number participants independently, so the prefix is required:
without it, 21 pairs of unrelated people collapse into single groups.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, cohen_kappa_score, f1_score)

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
SEED = 20260727
N_BOOT = 1000

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def loso(X, y, subj, model="logreg"):
    yt, yp, ss = [], [], []
    for i, s in enumerate(np.unique(subj)):
        te = subj == s
        tr = ~te
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            continue
        if model == "logreg":
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, C=0.1,
                                   class_weight="balanced"))
        else:
            clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                         n_jobs=-1, random_state=SEED,
                                         class_weight="balanced")
        clf.fit(X[tr], y[tr])
        yp.append(clf.predict(X[te])); yt.append(y[te])
        ss.append(np.full(te.sum(), s))
        if (i + 1) % 20 == 0:
            log(f"    ...{i+1}/{len(np.unique(subj))} subjects")
    return np.concatenate(yt), np.concatenate(yp), np.concatenate(ss)


def boot_ci(yt, yp, ss, fn):
    us = np.unique(ss)
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(us, size=len(us), replace=True)
        idx = np.concatenate([np.where(ss == u)[0] for u in pick])
        vals.append(fn(yt[idx], yp[idx]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def run(d, fc, classes, name, ref):
    dd = d[d.stage.isin(classes)]
    X = dd[fc].to_numpy(float)
    y = dd.stage.to_numpy()
    subj = dd.subject.to_numpy()

    log("=" * 72)
    log(f"{name}")
    log("=" * 72)
    log(f"windows {len(y)} | subjects {len(np.unique(subj))} | "
        f"classes {classes}")
    log(f"majority baseline {pd.Series(y).value_counts().max()/len(y):.4f} | "
        f"chance {1/len(classes):.4f}")
    log("")

    out = []
    for model in ["logreg", "rf"]:
        yt, yp, ss = loso(X, y, subj, model)
        acc = accuracy_score(yt, yp)
        bacc = balanced_accuracy_score(yt, yp)
        kap = cohen_kappa_score(yt, yp)
        mf1 = f1_score(yt, yp, average="macro")
        al, ah = boot_ci(yt, yp, ss, accuracy_score)
        bl, bh = boot_ci(yt, yp, ss, balanced_accuracy_score)
        log(f"{model:<8} accuracy {acc:.4f} [{al:.4f}, {ah:.4f}] | "
            f"balanced {bacc:.4f} [{bl:.4f}, {bh:.4f}] | "
            f"kappa {kap:.4f} | macroF1 {mf1:.4f}")
        out.append(dict(task=name, model=model, n=len(y),
                        subjects=len(np.unique(subj)), accuracy=acc,
                        acc_lo=al, acc_hi=ah, balanced=bacc, bal_lo=bl,
                        bal_hi=bh, kappa=kap, macro_f1=mf1))
        if model == "logreg":
            log("confusion (rows=true), labels " + str(sorted(np.unique(y))))
            log(str(confusion_matrix(yt, yp, labels=sorted(np.unique(y)))))
    log("")
    log(f"reference: {ref}")
    log("")
    return out


def main():
    d = pd.read_parquet(os.path.join(OUT, "sleepedf_expanded_features.parquet"))
    fc = [c for c in d.columns if c.startswith("ch")]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)

    rows = []
    rows += run(d, fc, ["W", "N1", "N2"],
                "W/N1/N2 -- comparable with all prior results",
                "8-subject control 0.8831 balanced | CBraMod 0.8217")
    rows += run(d, fc, ["W", "N1", "N2", "N3", "REM"],
                "5-class -- standard sleep-staging benchmark",
                "SIMSleepSM (Sleep Research 2026) reports 0.862 accuracy "
                "on Sleep-EDF-78")

    res = pd.DataFrame(rows)
    res.round(4).to_csv(os.path.join(OUT, "expanded_control.csv"), index=False)

    log("=" * 72)
    log("8 SUBJECTS vs 100 SUBJECTS")
    log("=" * 72)
    r3 = res[(res.task.str.startswith("W/N1/N2")) & (res.model == "logreg")].iloc[0]
    log(f"{'':<26}{'8 subj':>12}{'100 subj':>12}")
    log(f"{'recordings':<26}{8:>12}{197:>12}")
    log(f"{'windows':<26}{15140:>12}{int(r3.n):>12}")
    log(f"{'balanced accuracy':<26}{0.8831:>12.4f}{r3.balanced:>12.4f}")
    log(f"{'95% CI':<26}{'n/a':>12}{f'[{r3.bal_lo:.3f},{r3.bal_hi:.3f}]':>12}")

    with open(os.path.join(OUT, "test18_expanded_control.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"Wrote {OUT}\\test18_expanded_control.txt")


if __name__ == "__main__":
    main()
