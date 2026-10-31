"""
How high can reported accuracy be driven using only common, undocumented choices?

Purpose
-------
36_overlap_sweep.py showed that overlapping windows alone lift window-shuffled
accuracy to 0.8825 while leaving the participant-separated figure at 0.4198.
This script adds the remaining pipeline-ordering choices on top, one at a time,
to establish the accuracy REACHABLE on DEED when no participant-generalizable
dream-affect signal is present.

The ladder
----------
Each rung adds one choice to the rung above it:

    0  honest baseline        20 s non-overlapping, participant-separated,
                              scaling nested, no selection, no resampling,
                              classifier fixed in advance
    1  + window-shuffled cross-validation
    2  + 90% window overlap
    3  + scaler fitted on all data before splitting
    4  + feature selection fitted on all data before splitting
    5  + oversampling applied before splitting
    6  + hyperparameters chosen on all data
    7  + best of three classifiers reported

Two columns are produced for every rung. LEAKY applies each operation to the
complete dataset before partitioning, and partitions over windows. SOUND uses
the same data configuration and the same set of operations, but fits every one
of them inside each training fold and separates participants. The contrast is
therefore "these operations done wrong" against "the same operations done
right", not "with" against "without".

What this does and does not claim
---------------------------------
It does NOT claim that any published study made these choices. Their method
descriptions state neither the window overlap nor the placement of feature
selection and resampling relative to the split. What the ladder establishes is
that a pipeline assembled entirely from choices that are common in the applied
EEG literature, and individually easy to defend, reaches the published range on
data whose participant-separated result never leaves chance. That is an
existence proof about the protocol, not a claim about any author.

Two implementation notes
------------------------
An ANOVA F filter stands in for ReliefF. The quantity under test is the
PLACEMENT of selection relative to the partition, not the filter statistic, and
the F filter is cheap enough to also run nested for the sound column.

At rung 7 the leaky column reports the maximum over three classifiers, which is
selection over configurations using the test score. Its sound counterpart is the
single classifier declared in advance, since choosing the model by inner
cross-validation on the training fold is what correct practice requires.
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     GridSearchCV)
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")

SEEDS = [20260727, 20260728, 20260729]
N_SPLITS = 5
K_FEATURES = 40
VALENCE = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
META = {"segment_id", "subject", "oid", "window_idx", "start_sample",
        "E", "film", "stage"}
GRID = {"max_depth": [None, 20], "min_samples_leaf": [2, 5]}

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def load(ov):
    df = pd.read_parquet(os.path.join(OUT, f"deed_ov{ov:02d}.parquet"))
    fc = [c for c in df.columns if c not in META]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    df = df[df.E.isin(VALENCE)]
    return (df[fc].to_numpy(float), df.E.map(VALENCE).to_numpy(),
            df.subject.to_numpy(), len(fc))


def rf(seed, **kw):
    p = dict(n_estimators=300, min_samples_leaf=2, n_jobs=4,
             random_state=seed)
    p.update(kw)
    return RandomForestClassifier(**p)


def make_models(seed, best_of):
    m = {"rf": rf(seed)}
    if best_of:
        m["knn"] = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        m["svm"] = LinearSVC(C=0.1, dual="auto", random_state=seed)
    return m


def evaluate(X, y, groups, seed, cfg, leaky):
    """Evaluate one (rung, column).

    cfg names which operations are part of the pipeline at this rung:
        select, smote, tune, best_of  (scaling is always present)

    leaky=True  -> every operation is fitted on the complete dataset and the
                   partition is over windows.
    leaky=False -> every operation is fitted inside the training fold and the
                   partition separates participants.
    """
    Xw, yw, gw = X, y, groups

    if leaky:
        Xw = StandardScaler().fit_transform(Xw)
        if cfg.get("select"):
            Xw = SelectKBest(f_classif, k=K_FEATURES).fit_transform(Xw, yw)
        if cfg.get("smote"):
            # Group labels cannot follow synthetic points. That is exactly why
            # the ordering is invalid; the leaky column is window-partitioned.
            Xw, yw = SMOTE(random_state=seed).fit_resample(Xw, yw)
            gw = None
        params = {}
        if cfg.get("tune"):
            params = GridSearchCV(rf(seed), GRID, cv=3, n_jobs=1,
                                  scoring="accuracy").fit(Xw, yw).best_params_
        protos = make_models(seed, cfg.get("best_of", False))
        if params:
            protos["rf"] = rf(seed, **params)
        sp = StratifiedKFold(N_SPLITS, shuffle=True,
                             random_state=seed).split(np.zeros(len(yw)), yw)
    else:
        protos = {"rf": rf(seed)}          # declared in advance
        sp = StratifiedGroupKFold(N_SPLITS, shuffle=True,
                                  random_state=seed).split(np.zeros(len(yw)),
                                                           yw, gw)

    folds = list(sp)
    scores = {}
    for name, proto in protos.items():
        yt, yp = [], []
        for tr, te in folds:
            if len(np.unique(yw[tr])) < 2:
                continue
            Xtr, Xte, ytr = Xw[tr], Xw[te], yw[tr]

            if not leaky:
                sc = StandardScaler().fit(Xtr)
                Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
                if cfg.get("select"):
                    sk = SelectKBest(
                        f_classif, k=min(K_FEATURES, Xtr.shape[1])).fit(Xtr, ytr)
                    Xtr, Xte = sk.transform(Xtr), sk.transform(Xte)
                if cfg.get("smote"):
                    Xtr, ytr = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
                if cfg.get("tune"):
                    best = GridSearchCV(rf(seed), GRID, cv=3, n_jobs=1,
                                        scoring="accuracy"
                                        ).fit(Xtr, ytr).best_params_
                    proto = rf(seed, **best)

            m = clone(proto).fit(Xtr, ytr)
            yp.append(m.predict(Xte))
            yt.append(yw[te])
        yt, yp = np.concatenate(yt), np.concatenate(yp)
        scores[name] = (accuracy_score(yt, yp),
                        balanced_accuracy_score(yt, yp))

    return max(scores.values(), key=lambda t: t[0])


RUNGS = [
    ("0  honest baseline",             0,  dict()),
    ("1  + window-shuffled CV",        0,  dict()),
    ("2  + 90% window overlap",        90, dict()),
    ("3  + scaling on all data",       90, dict()),
    ("4  + selection on all data",     90, dict(select=True)),
    ("5  + oversampling before split", 90, dict(select=True, smote=True)),
    ("6  + tuning on all data",        90, dict(select=True, smote=True,
                                                tune=True)),
    ("7  + best of three models",      90, dict(select=True, smote=True,
                                                tune=True, best_of=True)),
]


def main():
    log("=" * 84)
    log("HOW HIGH CAN REPORTED ACCURACY BE DRIVEN WITHOUT ANY SIGNAL?")
    log("=" * 84)
    log("LEAKY: operation fitted on all data, partition over windows.")
    log("SOUND: same operations fitted inside the fold, participants separated.")
    log(f"RandomForest(300) unless stated. {N_SPLITS}-fold. "
        f"Mean over {len(SEEDS)} seeds.")
    log("")

    cache = {ov: load(ov) for ov in (0, 90)}
    for ov, (Xc, yc, _, nf) in cache.items():
        log(f"overlap {ov:>2}% : {len(yc)} windows, {nf} features, "
            f"majority {pd.Series(yc).value_counts().max() / len(yc):.4f}")
    log("")

    log(f"{'rung':<32}{'LEAKY acc':>11}{'bal':>9}{'SOUND acc':>12}{'bal':>9}"
        f"{'gap':>9}")
    log("-" * 84)

    # Resume: completed rungs are read back rather than recomputed. The high
    # overlap conditions are expensive and an interrupted run should not have
    # to repeat them.
    csv_path = os.path.join(OUT, "leakage_stack.csv")
    rows = []
    done = {}
    if os.path.exists(csv_path):
        prev = pd.read_csv(csv_path)
        prev = prev[prev.n_seeds == len(SEEDS)]
        done = {int(r.rung): r.to_dict() for _, r in prev.iterrows()}
        if done:
            log(f"resuming: rungs {sorted(done)} already complete")
            log("")

    # The sound column declares its classifier in advance, so `best_of` never
    # reaches it. Two rungs differing only by `best_of` therefore request a
    # bit-identical sound-column computation -- including the nested grid
    # search, which is the most expensive step in the ladder. Memoising on the
    # keys the sound path actually reads avoids repeating it.
    def sound_key(ov, cfg):
        return (ov, bool(cfg.get("select")), bool(cfg.get("smote")),
                bool(cfg.get("tune")))

    sound_cache = {}
    for i, (name, ov, cfg) in enumerate(RUNGS):
        if i in done:
            d = done[i]
            sound_cache.setdefault(sound_key(ov, cfg),
                                   dict(acc=d["sound_acc"],
                                        acc_sd=d["sound_acc_sd"],
                                        bal=d["sound_bal"], reused=True))
            log(f"{name:<32}{d['leaky_acc']:>11.4f}{d['leaky_bal']:>9.4f}"
                f"{d['sound_acc']:>12.4f}{d['sound_bal']:>9.4f}"
                f"{d['gap']:>+9.4f}   [cached]")
            rows.append(d)
            continue

        X, y, g, _ = cache[ov]
        t0 = time.time()
        la, lb = [], []
        for sd in SEEDS:
            a, b = evaluate(X, y, g, sd, cfg, leaky=(i != 0))
            la.append(a); lb.append(b)
        la, lb = np.array(la), np.array(lb)

        sk = sound_key(ov, cfg)
        if sk in sound_cache:
            s = sound_cache[sk]
            note = " (sound reused)"
        else:
            sa, sb = [], []
            for sd in SEEDS:
                a2, b2 = evaluate(X, y, g, sd, cfg, leaky=False)
                sa.append(a2); sb.append(b2)
            sa, sb = np.array(sa), np.array(sb)
            s = dict(acc=sa.mean(), acc_sd=sa.std(), bal=sb.mean())
            sound_cache[sk] = s
            note = ""

        log(f"{name:<32}{la.mean():>11.4f}{lb.mean():>9.4f}"
            f"{s['acc']:>12.4f}{s['bal']:>9.4f}"
            f"{la.mean() - s['acc']:>+9.4f}   [{time.time() - t0:.0f}s]{note}")
        rows.append(dict(
            rung=i, description=name.strip(), overlap=ov,
            leaky_acc=la.mean(), leaky_acc_sd=la.std(), leaky_bal=lb.mean(),
            sound_acc=s["acc"], sound_acc_sd=s["acc_sd"], sound_bal=s["bal"],
            gap=la.mean() - s["acc"], n_seeds=len(SEEDS)))
        pd.DataFrame(rows).round(4).to_csv(
            os.path.join(OUT, "leakage_stack.csv"), index=False)

    r = pd.DataFrame(rows)
    top = float(r.iloc[-1].leaky_acc)
    log("")
    log("=" * 84)
    log("SUMMARY")
    log("=" * 84)
    log(f"honest baseline                 : {r.iloc[0].leaky_acc:.4f}")
    log(f"top of the ladder               : {top:.4f}")
    log(f"participant-separated, top rung : {r.iloc[-1].sound_acc:.4f} "
        f"(balanced {r.iloc[-1].sound_bal:.4f})")
    log(f"total inflation                 : "
        f"{top - r.iloc[0].leaky_acc:+.4f}")
    log("")
    log("published reference points: Mahmoudi et al. 0.9383, "
        "Mojdehbar et al. 0.9233")
    log(f"top of ladder vs Mahmoudi       : {top - 0.9383:+.4f}")
    log(f"top of ladder vs Mojdehbar      : {top - 0.9233:+.4f}")
    log("")
    if top >= 0.90:
        log("The published range is reachable on this dataset using only")
        log("common, individually defensible choices, while the")
        log("participant-separated value never leaves chance.")
    elif top >= 0.80:
        log("Most of the published range is reachable. Report the measured")
        log("ceiling and leave the residual difference open.")
    else:
        log("A substantial gap remains unexplained. Report the ceiling")
        log("reached and state the remainder as open.")

    with open(os.path.join(OUT, "test28_leakage_stack.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"\nWrote {OUT}\\leakage_stack.csv and test28_leakage_stack.txt")


if __name__ == "__main__":
    main()
