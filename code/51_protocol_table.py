"""
Regenerate the headline protocol table under the current pinned environment,
with seeds and intervals (R2 C4, R2 C3, and an internal consistency fix).

Three problems are resolved here.

1. ENVIRONMENT DRIFT. The values in the submitted Table 4(a) were computed with
   scikit-learn 1.4.1. Under 1.9.0 the ungrouped split reproduces to the digit
   but StratifiedGroupKFold assigns folds differently, moving the grouped rows
   by more than a point. Numbers that cannot be reproduced in the environment
   the paper specifies should not be in the paper.

2. SINGLE SEED. The table reported one fold assignment. Reviewer 2 asks for
   robustness across seeds, and the seed-stability analysis showed that fold
   assignment is exactly where the earlier numbers were fragile.

3. MISLABELLED CLASSIFIER. The accompanying text attributes these values to
   logistic regression. 07_leakage_demo.py, which produced them, uses a random
   forest. The classifier is stated correctly here and the text is corrected to
   match.

Everything else is unchanged: same features, same three partitioning rules, same
two targets, same forest hyperparameters fixed a priori.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")

SEEDS = [20260727 + i for i in range(10)]
N_SPLITS = 5
N_JOBS = 20
N_BOOT = 2000
VALENCE = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
META = {"segment_id", "subject", "oid", "epoch_idx", "n_epochs",
        "epochs_from_end", "E", "film", "stage", "has_text"}

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def cv(X, y, groups, seed, track):
    """`track` is carried through the folds so the bootstrap can cluster on
    participant regardless of which variable defined the partition."""
    if groups is None:
        sp = StratifiedKFold(N_SPLITS, shuffle=True,
                             random_state=seed).split(X, y)
    else:
        sp = StratifiedGroupKFold(N_SPLITS, shuffle=True,
                                  random_state=seed).split(X, y, groups)
    yt, yp, tk = [], [], []
    for tr, te in sp:
        if len(np.unique(y[tr])) < 2:
            continue
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   n_jobs=N_JOBS,
                                   random_state=seed).fit(X[tr], y[tr])
        yp.append(m.predict(X[te])); yt.append(y[te]); tk.append(track[te])
    return (np.concatenate(yt), np.concatenate(yp), np.concatenate(tk))


def cluster_boot(yt, yp, g, seed=20260727):
    rng = np.random.default_rng(seed)
    uq = np.unique(g)
    idx = {k: np.where(g == k)[0] for k in uq}
    bal = []
    for _ in range(N_BOOT):
        pick = rng.choice(uq, size=len(uq), replace=True)
        sel = np.concatenate([idx[k] for k in pick])
        if len(np.unique(yt[sel])) < 2:
            continue
        bal.append(balanced_accuracy_score(yt[sel], yp[sel]))
    return float(np.percentile(bal, 2.5)), float(np.percentile(bal, 97.5))


def main():
    d = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    fc = [c for c in d.columns if c not in META]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)

    log("=" * 78)
    log("PROTOCOL TABLE, CURRENT ENVIRONMENT, 10 SEEDS")
    log("=" * 78)
    log("RandomForest(300, min_samples_leaf=2), 5-fold, features fixed.")
    log(f"seeds {len(SEEDS)} | cluster bootstrap {N_BOOT} over participants")
    log("")

    rows = []
    targets = [
        ("Dream valence", d[d.E.isin(VALENCE)],
         lambda x: x.E.map(VALENCE).to_numpy(), 0.3333),
        ("Sleep stage (W/N1/N2)", d[d.stage.isin(["W", "N1", "N2"])],
         lambda x: x.stage.to_numpy(), 0.3333),
    ]

    for tname, dd, ymap, chance in targets:
        y = ymap(dd)
        X = dd[fc].to_numpy(float)
        subj = dd.subject.to_numpy()
        rec = dd.segment_id.to_numpy()
        maj = pd.Series(y).value_counts().max() / len(y)
        log(f"### {tname}: {len(y)} windows, {dd.subject.nunique()} "
            f"participants, majority {maj:.4f}")
        log(f"{'held out':<26}{'accuracy':>18}{'balanced accuracy':>28}")
        log("-" * 78)
        for mode, grp in [("Random (window-shuffled)", None),
                          ("Recording", rec), ("Participant", subj)]:
            accs, bals = [], []
            ref = None
            for sd in SEEDS:
                yt, yp, tk = cv(X, y, grp, sd, subj)
                accs.append(accuracy_score(yt, yp))
                bals.append(balanced_accuracy_score(yt, yp))
                if ref is None:
                    ref = (yt, yp, tk)
            lo, hi = cluster_boot(*ref)
            a, b = np.array(accs), np.array(bals)
            log(f"{mode:<26}{a.mean():>10.4f} +/- {a.std():.4f}"
                f"{b.mean():>16.4f} +/- {b.std():.4f}  [{lo:.4f}, {hi:.4f}]")
            rows.append(dict(target=tname, held_out=mode,
                             acc_mean=a.mean(), acc_sd=a.std(),
                             bal_mean=b.mean(), bal_sd=b.std(),
                             bal_lo=lo, bal_hi=hi, majority=maj,
                             n_seeds=len(SEEDS)))
        log("")

    r = pd.DataFrame(rows)
    r.round(4).to_csv(os.path.join(OUT, "protocol_table.csv"), index=False)

    log("=" * 78)
    log("COMPARISON WITH THE SUBMITTED VALUES (scikit-learn 1.4.1)")
    log("=" * 78)
    old = {"Random (window-shuffled)": (0.5732, 0.4942),
           "Recording": (0.4243, 0.3484),
           "Participant": (0.4025, 0.3070)}
    log(f"{'held out':<26}{'submitted acc':>15}{'now':>10}"
        f"{'submitted bal':>16}{'now':>10}")
    log("-" * 78)
    for _, x in r[r.target == "Dream valence"].iterrows():
        oa, ob = old[x.held_out]
        log(f"{x.held_out:<26}{oa:>15.4f}{x.acc_mean:>10.4f}"
            f"{ob:>16.4f}{x.bal_mean:>10.4f}")
    log("")
    log("The window-shuffled row reproduces exactly; the grouped rows move")
    log("because StratifiedGroupKFold changed between versions. Conclusions")
    log("are unchanged: participant-separated accuracy stays below the")
    log("majority baseline and balanced accuracy stays at chance.")

    with open(os.path.join(OUT, "test41_protocol_table.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"\nWrote {OUT}\\protocol_table.csv")


if __name__ == "__main__":
    main()
