"""
The leakage ladder repeated with randomised recording-level labels
(Reviewer 1, comment 5).

The question
------------
Section IV-G builds a pipeline from common, individually defensible choices and
reaches 0.9055 window-shuffled accuracy while the participant-separated value
stays at chance. With the ORIGINAL labels that shows the pipeline is
susceptible to inflation; it does not by itself show that high window-level
accuracy needs no task-relevant information. Reviewer 1 asks for the ladder to
be repeated with labels randomised at the recording level, before segmentation
and resampling, under a null that keeps the repeated-measure structure.

Label conditions
----------------
  real     the reported labels; reproduces the published ladder
  within   recording labels permuted WITHIN each participant. Every window of
           a recording keeps a common label, and each participant keeps their
           own label multiset, so participant identity still predicts the label
           exactly as much as it does in the data. What is destroyed is any
           relation between a recording's EEG and its reported affect beyond
           the participant's overall composition.
  free     recording labels permuted across ALL recordings. Windows within a
           recording still share a label, but the participant-label
           association is destroyed as well.

Randomising at the recording level and propagating the label to the recording's
windows is equivalent to assigning it before segmentation, because segmentation
does not depend on the label; resampling (SMOTE) and selection then operate on
the randomised labels exactly as they did on the real ones. The same
recording-to-label mapping is used at both overlap levels within a repetition.

If the leaky column still reaches the published range under `within` and `free`,
the high number is attainable when the labels carry no information about the
EEG of the recording they are attached to -- which is the claim the manuscript
may then make. If it does not, the manuscript may claim only susceptibility.

Rungs
-----
0-5 of 37_leakage_stack.py except rung 3 (scaling fitted on all data), which is
identical to rung 2 for a random forest: splits are invariant to per-feature
monotone rescaling, and with the real labels both rungs return 0.8827. Rungs 6 (tuning on all data) and 7 (best of three
models) are not repeated: with the real labels both return exactly the rung-5
value, 0.9055, and rung 6 alone took more than two hours per run.

Synthetic observations (Reviewer 1, comment 4)
----------------------------------------------
At rung 5 the leaky column oversamples before partitioning, so synthetic points
enter the test folds. Every leaky rung-5 evaluation is scored three ways: on all
test points (the published figure), on the untouched real test windows only,
and on the synthetic test points only. imblearn's SMOTE returns the original
samples first and the synthetic ones after; this is checked, not assumed.

Output: out/random_label_ladder_raw.csv (resumable), summary in
out/random_label_ladder.csv.

Time-limited run (30 Sep 2026)
------------------------------
Each 300-tree forest on the 90%-overlap data takes about 25 minutes per
five-fold evaluation on this machine when the labels carry no signal, so the
design is reduced and the reduction is reported: 50 trees; rungs 0, 1, 2 and 5
(the published ladder already shows rung 4 between rungs 2 and 5 and rung 3
equal to rung 2); participant-separated references at rungs 0 and 1 only;
two repetitions of each randomised condition and one real-label run. The
real-label run at 50 trees is compared with the published 300-tree ladder, so
the effect of the tree count on these values is measured rather than assumed.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
RAW = os.path.join(OUT, "random_label_ladder_raw.csv")
SUMMARY = os.path.join(OUT, "random_label_ladder.csv")
LOG = os.path.join(OUT, "run_random_label_ladder.log")

N_SPLITS = 5
K_FEATURES = 40                       # as 37_leakage_stack.py
VALENCE = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
META = {"segment_id", "subject", "oid", "window_idx", "start_sample",
        "E", "film", "stage"}

REAL_SEEDS = [20260727]                          # one real-label run
N_RAND = 2                                       # repetitions per null (time-limited run)
RAND_BASE = 910_000
N_WORKERS = 1                                    # sequential: one forest in memory at a time
N_TREES = 50                                     # time-limited run; 30 in --smoke
SOUND_RUNGS = {0, 1}                             # participant-separated references
RF_THREADS = 22                                  # threads per forest

RUNGS = [
    (0, "honest baseline", 0, dict(), False),
    (1, "+ window-shuffled CV", 0, dict(), True),
    (2, "+ 90% window overlap", 90, dict(), True),
    (5, "+ oversampling before split", 90, dict(select=True, smote=True), True),
]

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def load(ov, smoke=False):
    df = pd.read_parquet(os.path.join(OUT, f"deed_ov{ov:02d}.parquet"))
    fc = [c for c in df.columns if c not in META]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    df = df[df.E.isin(VALENCE)].reset_index(drop=True)
    if smoke:
        keep = sorted(df.subject.unique())[:8]
        df = df[df.subject.isin(keep)].reset_index(drop=True)
    return dict(X=df[fc].to_numpy(float), y=df.E.map(VALENCE).to_numpy(),
                subj=df.subject.to_numpy(), rec=df.segment_id.to_numpy())


def label_map(base, mode, r):
    """Recording -> label for one repetition. `base` is the real mapping."""
    recs = np.array(sorted(base))
    lab = np.array([base[k][0] for k in recs])
    subj = np.array([base[k][1] for k in recs])
    if mode == "real":
        return dict(zip(recs, lab))
    rng = np.random.default_rng(RAND_BASE + 1000 * (mode == "free") + r)
    new = lab.copy()
    if mode == "free":
        new = rng.permutation(lab)
    else:
        for g in np.unique(subj):
            m = np.where(subj == g)[0]
            new[m] = rng.permutation(lab[m])
    return dict(zip(recs, new))


def rf(seed, threads):
    return RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=2,
                                  n_jobs=threads, random_state=seed)


def evaluate(X, y, subj, seed, cfg, leaky, threads):
    """One rung, one column; mirrors 37_leakage_stack.evaluate for the
    single-classifier rungs, and additionally scores real vs synthetic."""
    n0 = len(y)
    Xw, yw, gw = X, y, subj
    is_real = np.ones(n0, bool)
    if leaky:
        Xw = StandardScaler().fit_transform(Xw)
        if cfg.get("select"):
            Xw = SelectKBest(f_classif, k=K_FEATURES).fit_transform(Xw, yw)
        if cfg.get("smote"):
            Xs, ys = SMOTE(random_state=seed).fit_resample(Xw, yw)
            # originals first, synthetic after -- verify rather than assume
            assert np.array_equal(Xs[:n0], Xw) and np.array_equal(ys[:n0], yw)
            Xw, yw = Xs, ys
            is_real = np.arange(len(yw)) < n0
            gw = None
        sp = StratifiedKFold(N_SPLITS, shuffle=True,
                             random_state=seed).split(np.zeros(len(yw)), yw)
    else:
        sp = StratifiedGroupKFold(N_SPLITS, shuffle=True,
                                  random_state=seed).split(np.zeros(len(yw)),
                                                           yw, gw)
    yt, yp, rt = [], [], []
    for tr, te in sp:
        if len(np.unique(yw[tr])) < 2:
            continue
        Xtr, Xte, ytr = Xw[tr], Xw[te], yw[tr]
        if not leaky:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            if cfg.get("select"):
                sk = SelectKBest(f_classif,
                                 k=min(K_FEATURES, Xtr.shape[1])).fit(Xtr, ytr)
                Xtr, Xte = sk.transform(Xtr), sk.transform(Xte)
            if cfg.get("smote"):
                Xtr, ytr = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
        m = clone(rf(seed, threads)).fit(Xtr, ytr)
        yp.append(m.predict(Xte)); yt.append(yw[te]); rt.append(is_real[te])
    yt, yp, rt = map(np.concatenate, (yt, yp, rt))
    out = dict(acc=accuracy_score(yt, yp), bal=balanced_accuracy_score(yt, yp),
               n_test=len(yt), synth_frac=float((~rt).mean()))
    out["acc_real"] = accuracy_score(yt[rt], yp[rt])
    out["bal_real"] = balanced_accuracy_score(yt[rt], yp[rt])
    # always present, so every appended CSV row has the same columns
    has = (~rt).any()
    out["acc_synth"] = accuracy_score(yt[~rt], yp[~rt]) if has else np.nan
    out["bal_synth"] = (balanced_accuracy_score(yt[~rt], yp[~rt]) if has
                        else np.nan)
    return out


def repetition(mode, r, seed, data, base, done, threads):
    """Whole ladder for one label condition and one repetition."""
    mp = label_map(base, mode, r)
    rows = []
    sound_memo = {}
    for rung, name, ov, cfg, leaky in RUNGS:
        d = data[ov]
        y = np.array([mp[k] for k in d["rec"]])
        for col in ("leaky", "sound"):
            if col == "sound" and rung not in SOUND_RUNGS:
                continue
            if (mode, r, rung, col) in done:
                continue
            if col == "leaky" and not leaky:
                # rung 0 is the honest baseline: its "leaky" entry is the
                # sound protocol itself, as in 37_leakage_stack.py
                res = evaluate(d["X"], y, d["subj"], seed, cfg, False, threads)
            elif col == "leaky":
                res = evaluate(d["X"], y, d["subj"], seed, cfg, True, threads)
            else:
                key = (ov, bool(cfg.get("select")), bool(cfg.get("smote")))
                if key not in sound_memo:
                    sound_memo[key] = evaluate(d["X"], y, d["subj"], seed, cfg,
                                               False, threads)
                res = sound_memo[key]
            rows.append(dict(mode=mode, r=r, seed=seed, rung=rung,
                             description=name, overlap=ov, column=col, **res))
            print(f"    {mode} r{r} rung {rung} {col}: acc {res['acc']:.4f} "
                  f"bal {res['bal']:.4f} [{time.strftime('%H:%M:%S')}]", flush=True)
    return rows


def summarise():
    r = pd.read_csv(RAW)
    g = r.groupby(["mode", "rung", "column"])
    s = g.agg(n=("acc", "size"), acc=("acc", "mean"), acc_sd=("acc", "std"),
              bal=("bal", "mean"), bal_sd=("bal", "std"),
              acc_real=("acc_real", "mean"), bal_real=("bal_real", "mean"),
              acc_synth=("acc_synth", "mean"),
              synth_frac=("synth_frac", "mean")).reset_index()
    s.round(4).to_csv(SUMMARY, index=False)
    return s


def main():
    smoke = "--smoke" in sys.argv
    global N_RAND, N_TREES, REAL_SEEDS
    if smoke:
        N_RAND, N_TREES, REAL_SEEDS = 1, 30, REAL_SEEDS[:1]
    data = {ov: load(ov, smoke) for ov in (0, 90)}
    d0 = data[0]
    base = {}
    for k, lab, s in zip(d0["rec"], d0["y"], d0["subj"]):
        base.setdefault(k, (lab, s))
    for ov in (90,):                       # same recordings at both overlaps
        missing = set(data[ov]["rec"]) - set(base)
        assert not missing, f"{len(missing)} recordings only at {ov}% overlap"

    log("=" * 84)
    log("LEAKAGE LADDER WITH RANDOMISED RECORDING-LEVEL LABELS")
    log("=" * 84)
    for ov, d in data.items():
        log(f"overlap {ov:>2}%: {len(d['y'])} windows, "
            f"{len(np.unique(d['rec']))} recordings, "
            f"{len(np.unique(d['subj']))} participants")
    log(f"conditions: real ({len(REAL_SEEDS)} seeds), within and free "
        f"({N_RAND} repetitions each)")
    log("")

    raw = RAW.replace(".csv", "_smoke.csv") if smoke else RAW
    done = set()
    if os.path.exists(raw):
        p = pd.read_csv(raw)
        done = set(zip(p["mode"], p.r, p.rung, p.column))
    jobs = [("real", i, s) for i, s in enumerate(REAL_SEEDS)]
    jobs += [(m, i, 20260727 + i) for i in range(N_RAND)
             for m in ("within", "free")]
    need = len(RUNGS) + len(SOUND_RUNGS)
    jobs = [j for j in jobs if sum((j[0], j[1], k, c) in done for k, *_ in RUNGS
                                   for c in ("leaky", "sound")) < need]
    log(f"repetitions to run: {len(jobs)}")

    t0 = time.time()
    gen = Parallel(n_jobs=1 if smoke else N_WORKERS,
                   return_as="generator_unordered")(
        delayed(repetition)(m, i, s, data, base, done,
                            2 if smoke else RF_THREADS) for m, i, s in jobs)
    for k, rows in enumerate(gen, 1):
        df = pd.DataFrame(rows)
        df.to_csv(raw, mode="a", header=not os.path.exists(raw), index=False)
        log(f"  {k}/{len(jobs)} repetitions  ({(time.time() - t0) / 60:.1f} min)")

    if smoke:
        log(pd.read_csv(raw).round(3).to_string())
        return
    s = summarise()
    log("")
    with pd.option_context("display.width", 200):
        log(s.round(4).to_string(index=False))
    with open(LOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
