"""
Regenerate the fingerprinting and replication numbers in the pinned environment.

51_protocol_table.py moved the DEED protocol rows because StratifiedGroupKFold
changed between scikit-learn 1.4 and 1.9. Tables 5 and 6 of the manuscript still
carried values from the earlier environment, so the paper contained two
generations of numbers side by side: Table 4 said the participant-separated
figure was 0.3224 while Tables 5 and 6 said 0.4025 for the same quantity.

Rather than caveat that, everything grouped is recomputed here in one
environment, with ten fold assignments so the sensitivity that caused the
problem is reported rather than hidden.

What is recomputed, and what is not
-----------------------------------
  recomputed   identity decoding (window-shuffled and recording-grouped) on
               DEED and on Tononi, and the three partitioning rules on Tononi
  unchanged    the identity-only oracle and the mutual information, which are
               computed by counting and involve no cross-validation, so no
               version of any library can move them

The oracle is the quantity the mechanism argument rests on, and it is
reassuring that it is exactly the one that cannot drift.
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

SEEDS = [20260727 + i for i in range(10)]
N_SPLITS = 5
N_JOBS = 20
VALENCE = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
META = {"segment_id", "subject", "oid", "epoch_idx", "n_epochs",
        "epochs_from_end", "E", "film", "stage", "has_text"}

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def cv(X, y, groups, seed):
    if groups is None:
        sp = StratifiedKFold(N_SPLITS, shuffle=True,
                             random_state=seed).split(X, y)
    else:
        sp = StratifiedGroupKFold(N_SPLITS, shuffle=True,
                                  random_state=seed).split(X, y, groups)
    yt, yp = [], []
    for tr, te in sp:
        if len(np.unique(y[tr])) < 2:
            continue
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   n_jobs=N_JOBS,
                                   random_state=seed).fit(X[tr], y[tr])
        seen = np.isin(y[te], np.unique(y[tr]))
        if not seen.any():
            continue
        yp.append(m.predict(X[te][seen])); yt.append(y[te][seen])
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return accuracy_score(yt, yp), balanced_accuracy_score(yt, yp)


def over_seeds(X, y, groups):
    a = np.array([cv(X, y, groups, s)[0] for s in SEEDS])
    b = np.array([cv(X, y, groups, s)[1] for s in SEEDS])
    return a.mean(), a.std(), b.mean(), b.std()


def oracle_and_mi(y, subj):
    """Counting only: no cross-validation, so no library version can move it."""
    maj = pd.Series(y).groupby(pd.Series(subj)).agg(
        lambda s: s.value_counts().idxmax())
    pred = pd.Series(subj).map(maj).to_numpy()
    acc = accuracy_score(y, pred)
    bacc = balanced_accuracy_score(y, pred)
    mi = mutual_info_score(subj, y)
    hy = mutual_info_score(y, y)
    return acc, bacc, mi, mi / hy if hy else np.nan


def main():
    rows = []
    log("=" * 78)
    log("FINGERPRINTING AND REPLICATION, PINNED ENVIRONMENT, 10 SEEDS")
    log("=" * 78)
    log("")

    # ---------------- DEED ------------------------------------------------
    d = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    fc = [c for c in d.columns if c not in META]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = d[d.E.isin(VALENCE)]
    X = d[fc].to_numpy(float)
    y = d.E.map(VALENCE).to_numpy()
    subj = d.subject.to_numpy()
    rec = d.segment_id.to_numpy()

    log(f"### DEED: {len(y)} windows, {d.subject.nunique()} participants, "
        f"{d.segment_id.nunique()} recordings")
    log("")
    log("-- identity decoding (34-way, chance 0.0294) --")
    for name, grp in [("window-shuffled", None), ("recording-grouped", rec)]:
        am, asd, _, _ = over_seeds(X, subj, grp)
        log(f"  {name:<22}{am:.4f} +/- {asd:.4f}")
        rows.append(dict(dataset="DEED", quantity=f"identity {name}",
                         mean=am, sd=asd, n_seeds=len(SEEDS)))

    oa, ob, mi, min_ = oracle_and_mi(y, subj)
    log("")
    log("-- identity-only oracle and mutual information (no CV) --")
    log(f"  oracle accuracy        {oa:.4f}")
    log(f"  oracle balanced        {ob:.4f}")
    log(f"  I(subject;label)/H     {min_:.4f}")
    for q, v in [("oracle accuracy", oa), ("oracle balanced", ob),
                 ("mi normalised", min_)]:
        rows.append(dict(dataset="DEED", quantity=q, mean=v, sd=0.0,
                         n_seeds=0))
    log("")

    # ---------------- Tononi ---------------------------------------------
    tp = os.path.join(OUT, "tononi_features.parquet")
    if not os.path.exists(tp):
        log("tononi_features.parquet absent; skipping replication refresh")
    else:
        t = pd.read_parquet(tp)
        tfc = [c for c in t.columns if c.startswith("ch")]
        t = t.replace([np.inf, -np.inf], np.nan).dropna(subset=tfc)
        Xt = t[tfc].to_numpy(float)
        yt = t.dreaming.astype(str).to_numpy()
        st = t.subject.astype(str).to_numpy()
        rt = t.recording.astype(str).to_numpy()

        log(f"### Tononi: {len(yt)} windows, {len(np.unique(st))} "
            f"participants, {len(np.unique(rt))} recordings")
        log("")
        log("-- partitioning rule --")
        for name, grp in [("window-shuffled", None), ("recording-grouped", rt),
                          ("participant-wise", st)]:
            am, asd, bm, bsd = over_seeds(Xt, yt, grp)
            log(f"  {name:<22}acc {am:.4f} +/- {asd:.4f}   "
                f"bal {bm:.4f} +/- {bsd:.4f}")
            rows.append(dict(dataset="Tononi", quantity=f"acc {name}",
                             mean=am, sd=asd, bal_mean=bm, bal_sd=bsd,
                             n_seeds=len(SEEDS)))

        log("")
        log("-- identity decoding (18-way, chance 0.0556) --")
        for name, grp in [("window-shuffled", None),
                          ("recording-grouped", rt)]:
            am, asd, _, _ = over_seeds(Xt, st, grp)
            log(f"  {name:<22}{am:.4f} +/- {asd:.4f}")
            rows.append(dict(dataset="Tononi", quantity=f"identity {name}",
                             mean=am, sd=asd, n_seeds=len(SEEDS)))

        toa, tob, tmi, tmin = oracle_and_mi(yt, st)
        log("")
        log(f"  oracle accuracy        {toa:.4f}")
        log(f"  I(subject;label)/H     {tmin:.4f}")
        rows.append(dict(dataset="Tononi", quantity="oracle accuracy",
                         mean=toa, sd=0.0, n_seeds=0))
        rows.append(dict(dataset="Tononi", quantity="mi normalised",
                         mean=tmin, sd=0.0, n_seeds=0))

    r = pd.DataFrame(rows)
    r.round(4).to_csv(os.path.join(OUT, "mechanism_refresh.csv"), index=False)

    log("")
    log("=" * 78)
    log("DRIFT AGAINST THE SUBMITTED VALUES")
    log("=" * 78)
    old = {("DEED", "identity window-shuffled"): 0.8037,
           ("DEED", "identity recording-grouped"): 0.7313,
           ("DEED", "oracle accuracy"): 0.5465,
           ("DEED", "mi normalised"): 0.1624,
           ("Tononi", "acc window-shuffled"): 0.6183,
           ("Tononi", "acc recording-grouped"): 0.3757,
           ("Tononi", "acc participant-wise"): 0.3892,
           ("Tononi", "identity window-shuffled"): 0.8668,
           ("Tononi", "identity recording-grouped"): 0.7380,
           ("Tononi", "oracle accuracy"): 0.5778,
           ("Tononi", "mi normalised"): 0.1902}
    log(f"{'dataset':<9}{'quantity':<30}{'submitted':>11}{'now':>9}{'delta':>9}")
    log("-" * 78)
    for _, x in r.iterrows():
        k = (x.dataset, x.quantity)
        if k in old:
            log(f"{x.dataset:<9}{x.quantity:<30}{old[k]:>11.4f}"
                f"{x['mean']:>9.4f}{x['mean'] - old[k]:>+9.4f}")

    with open(os.path.join(OUT, "test42_mechanism_refresh.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"\nWrote {OUT}\\mechanism_refresh.csv")


if __name__ == "__main__":
    main()
