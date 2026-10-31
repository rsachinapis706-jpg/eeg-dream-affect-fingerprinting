"""
Does window overlap explain the residual gap to the published 92-94%?

Background
----------
08_replicate_published.py implemented both published DEED pipelines and reached
0.4711 under the most permissive protocol, against their reported 93.83% and
92.33%. The leakage sources measured so far account for a large share of that gap
but not all of it. This script tests the one remaining candidate that is both
common in the literature and unstated in both method descriptions: overlapping
analysis windows.

Mechanism under test
--------------------
At 90% overlap two neighbouring 20 s windows share 18 s of signal. Under
window-shuffled cross-validation such near-duplicates are split across the
partition, so part of the test set is a near-copy of something in the training
set. This is distinct from participant fingerprinting -- it operates within a
recording, not across recordings -- and the two multiply.

Design
------
Everything is held fixed except the two variables of interest:

    overlap        0%, 50%, 75%, 90%     (same recordings, same features)
    partitioning   window-shuffled  vs  participant-separated

Preprocessing is nested correctly in BOTH columns (scaling fitted inside each
training fold, no selection, no resampling), so this isolates overlap alone. The
other leakage sources are added on top in 37_leakage_stack.py.

Every cell is repeated over several seeds, because the fold assignment of
StratifiedGroupKFold changed between scikit-learn 1.4 and 1.9 and the earlier
single-seed numbers were sensitive to it. Mean and standard deviation across
seeds are reported so that sensitivity is a measured quantity rather than an
unstated one.

Reading the output
------------------
If the window-shuffled column rises steeply with overlap while the
participant-separated column stays at chance, then overlap is a large and
independent contributor to reported accuracy, and the residual gap is explained
without attributing any specific choice to any specific paper.
"""
import os
import time
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")

OVERLAPS = [0, 50, 75, 90]
SEEDS = [20260727, 20260728, 20260729, 20260730, 20260731]
N_SPLITS = 5
VALENCE = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}

META = {"segment_id", "subject", "oid", "window_idx", "start_sample",
        "E", "film", "stage"}

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def load(ov):
    df = pd.read_parquet(os.path.join(OUT, f"deed_ov{ov:02d}.parquet"))
    fc = [c for c in df.columns if c not in META]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    df = df[df.E.isin(VALENCE)]
    return df, fc


def evaluate(X, y, groups, mode, seed):
    """Scaling is fitted inside each training fold in both modes."""
    if mode == "random":
        sp = StratifiedKFold(N_SPLITS, shuffle=True,
                             random_state=seed).split(X, y)
    else:
        sp = StratifiedGroupKFold(N_SPLITS, shuffle=True,
                                  random_state=seed).split(X, y, groups)
    yt, yp = [], []
    for tr, te in sp:
        if len(np.unique(y[tr])) < 2:
            continue
        pipe = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   n_jobs=-1, random_state=seed))
        pipe.fit(X[tr], y[tr])
        yp.append(pipe.predict(X[te]))
        yt.append(y[te])
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return accuracy_score(yt, yp), balanced_accuracy_score(yt, yp)


def main():
    log("=" * 78)
    log("WINDOW OVERLAP AS A LEAKAGE MULTIPLIER")
    log("=" * 78)
    log("RandomForest(300), 5-fold, scaling nested in both columns.")
    log(f"Seeds: {len(SEEDS)}. Values are mean +/- sd across seeds.")
    log("")

    rows = []
    for ov in OVERLAPS:
        df, fc = load(ov)
        X = df[fc].to_numpy(float)
        y = df.E.map(VALENCE).to_numpy()
        subj = df.subject.to_numpy()
        maj = pd.Series(y).value_counts().max() / len(y)

        log(f"-- overlap {ov}% | windows {len(y)} | recordings "
            f"{df.segment_id.nunique()} | subjects {df.subject.nunique()} "
            f"| majority {maj:.4f} --")

        for mode, grp in [("random", None), ("subject", subj)]:
            t0 = time.time()
            accs, bals = [], []
            for sd in SEEDS:
                a, b = evaluate(X, y, grp, mode, sd)
                accs.append(a); bals.append(b)
            accs, bals = np.array(accs), np.array(bals)
            log(f"   {mode:<8} acc {accs.mean():.4f} +/- {accs.std():.4f}"
                f"   bal {bals.mean():.4f} +/- {bals.std():.4f}"
                f"   [{time.time() - t0:.0f}s]")
            rows.append(dict(
                overlap=ov, windows=len(y), partition=mode, majority=maj,
                acc_mean=accs.mean(), acc_sd=accs.std(),
                bal_mean=bals.mean(), bal_sd=bals.std(),
                acc_min=accs.min(), acc_max=accs.max(),
                n_seeds=len(SEEDS)))
            pd.DataFrame(rows).round(4).to_csv(
                os.path.join(OUT, "overlap_sweep.csv"), index=False)
        log("")

    r = pd.DataFrame(rows)
    rnd = r[r.partition == "random"].set_index("overlap")
    sub = r[r.partition == "subject"].set_index("overlap")

    log("=" * 78)
    log("SUMMARY")
    log("=" * 78)
    log(f"{'overlap':>8}{'windows':>10}{'shuffled':>11}{'subject-wise':>14}"
        f"{'inflation':>12}")
    log("-" * 78)
    for ov in OVERLAPS:
        log(f"{ov:>7}%{int(rnd.loc[ov, 'windows']):>10}"
            f"{rnd.loc[ov, 'acc_mean']:>11.4f}{sub.loc[ov, 'acc_mean']:>14.4f}"
            f"{rnd.loc[ov, 'acc_mean'] - sub.loc[ov, 'acc_mean']:>+12.4f}")
    log("")
    g = rnd.loc[OVERLAPS[-1], "acc_mean"] - rnd.loc[OVERLAPS[0], "acc_mean"]
    s = sub.loc[OVERLAPS[-1], "acc_mean"] - sub.loc[OVERLAPS[0], "acc_mean"]
    log(f"window-shuffled, 0% -> {OVERLAPS[-1]}% overlap : {g:+.4f}")
    log(f"participant-wise, same change             : {s:+.4f}")
    log("")
    if g > 0.15 and abs(s) < 0.06:
        log("Overlap inflates the window-shuffled figure substantially while")
        log("leaving the participant-separated figure unchanged. It is an")
        log("independent leakage source, additional to fingerprinting.")
    elif g <= 0.05:
        log("Overlap does not materially change either column. It is not the")
        log("explanation for the residual gap; report the gap as unexplained.")
    else:
        log("Overlap has a partial effect. Report the measured magnitude")
        log("without claiming it closes the gap.")

    with open(os.path.join(OUT, "test27_overlap_sweep.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"\nWrote {OUT}\\overlap_sweep.csv and test27_overlap_sweep.txt")


if __name__ == "__main__":
    main()
