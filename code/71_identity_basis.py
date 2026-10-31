"""
Table 9 recomputed: participant identity under three partitionings, on the same
windows, grouping unit, classifier and seeds as Table 8.

What was wrong (Reviewer 1, comment 2)
  Table 8 reports recording-grouped identity accuracy 0.7254; Table 9 reported
  0.4862 under the same label. 48_identity_basis.py grouped by `oid`, which
  encodes participant and NIGHT (78 groups), not by `segment_id`, the
  recording (532 groups). Its "recording-grouped" column was therefore a
  night-grouped column, and it used all 7,418 windows and one seed where
  Table 8 used the 6,144-window valence subset and ten seeds. The paragraph
  explaining why the session-held-out column exceeded it was accordingly wrong.

What is done here
  windows      the 6,144-window valence subset of Table 8 (identity decoding
               uses no affect label, but the same windows keep the tables
               commensurate)
  shuffled     StratifiedKFold over windows, ten seeds
  recording    StratifiedGroupKFold over segment_id, ten seeds -- the Table 8
               procedure, so the all-features row must reproduce 0.7254
  session      leave-one-session-slot-out among participants with more than
               one night; forest seed varied over the same ten seeds
  night        StratifiedGroupKFold over oid, ten seeds, all features only --
               kept to document what the mislabelled column measured

Accuracy is scored on test windows whose participant appears in training, as
in 52_mechanism_refresh.py. Random forest, 300 trees, min_samples_leaf=2.
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
RAW = os.path.join(OUT, "identity_basis_v2_raw.csv")
SUMMARY = os.path.join(OUT, "identity_basis_v2.csv")
LOG = os.path.join(OUT, "run_identity_basis_v2.log")

SEEDS = [20260727 + i for i in range(10)]       # as in 52_mechanism_refresh
N_SPLITS = 5
RF_THREADS = 22
VALENCE = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
META = {"segment_id", "subject", "oid", "epoch_idx", "n_epochs",
        "epochs_from_end", "E", "film", "stage", "has_text"}
ORDER = ["all features", "amplitude-dependent", "absolute power",
         "log-variance", "amplitude-normalised", "relative power",
         "spectral shape", "aperiodic slope", "Hjorth"]

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def families(cols):
    f = {
        "absolute power": [c for c in cols if c.endswith("_abs")],
        "log-variance": [c for c in cols if c.endswith("_logvar")],
        "relative power": [c for c in cols if c.endswith("_rel")],
        "aperiodic slope": [c for c in cols if c.endswith("_apslope")],
        "spectral shape": [c for c in cols
                           if c.endswith("_specent") or c.endswith("_sef95")],
        "Hjorth": [c for c in cols
                   if c.endswith("_hj_mob") or c.endswith("_hj_comp")],
    }
    f["amplitude-dependent"] = f["absolute power"] + f["log-variance"]
    f["amplitude-normalised"] = f["relative power"] + f["spectral shape"]
    f["all features"] = list(cols)
    return f


def rf(seed):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                  n_jobs=RF_THREADS, random_state=seed)


def score(splits, X, y, seed):
    yt, yp = [], []
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2:
            continue
        m = rf(seed).fit(X[tr], y[tr])
        seen = np.isin(y[te], np.unique(y[tr]))
        if not seen.any():
            continue
        yp.append(m.predict(X[te][seen])); yt.append(y[te][seen])
    return accuracy_score(np.concatenate(yt), np.concatenate(yp))


def splits_for(mode, X, y, groups, seed, slot=None):
    if mode == "shuffled":
        return list(StratifiedKFold(N_SPLITS, shuffle=True,
                                    random_state=seed).split(X, y))
    if mode in ("recording", "night"):
        return list(StratifiedGroupKFold(N_SPLITS, shuffle=True,
                                         random_state=seed).split(X, y, groups))
    idx = np.arange(len(y))
    return [(idx[slot != k], idx[slot == k]) for k in np.unique(slot)]


def main():
    d = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    fc = [c for c in d.columns if c not in META]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = d[d.E.isin(VALENCE)].reset_index(drop=True)

    order = (d.drop_duplicates("oid")[["subject", "oid"]]
             .sort_values(["subject", "oid"]))
    order["slot"] = order.groupby("subject").cumcount()
    d = d.merge(order[["oid", "slot"]], on="oid", how="left")
    nights = d.groupby("subject").oid.nunique()
    multi = sorted(nights[nights >= 2].index)
    dm = d[d.subject.isin(multi)].reset_index(drop=True)

    log("=" * 78)
    log("PARTICIPANT IDENTITY UNDER THREE PARTITIONINGS (Table 9, recomputed)")
    log("=" * 78)
    log(f"windows {len(d)} | participants {d.subject.nunique()} | recordings "
        f"{d.segment_id.nunique()} | nights {d.oid.nunique()}")
    log(f"multi-night participants {len(multi)} ({len(dm)} windows); chance "
        f"{1 / d.subject.nunique():.4f} (all) and {1 / len(multi):.4f} "
        f"(multi-night)")
    log("")

    fam = families(fc)
    done = set()
    if os.path.exists(RAW):
        prev = pd.read_csv(RAW)
        done = set(zip(prev.family, prev["mode"], prev.seed))
    else:
        pd.DataFrame(columns=["family", "mode", "seed", "acc", "n_features"]
                     ).to_csv(RAW, index=False)

    jobs = [(f, m) for f in ORDER for m in ("shuffled", "recording",
                                            "session")]
    jobs.append(("all features", "night"))
    t0 = time.time()
    for fname, mode in jobs:
        cols = fam[fname]
        src = dm if mode == "session" else d
        X = src[cols].to_numpy(float)
        y = src.subject.to_numpy()
        grp = {"recording": src.segment_id.to_numpy(),
               "night": src.oid.to_numpy()}.get(mode)
        use = SEEDS if fname == "all features" else SEEDS[:5]
        for s in use:
            if (fname, mode, s) in done:
                continue
            acc = score(splits_for(mode, X, y, grp, s,
                                   src.slot.to_numpy()), X, y, s)
            pd.DataFrame([dict(family=fname, mode=mode, seed=s, acc=acc,
                               n_features=len(cols))]).to_csv(
                RAW, mode="a", header=False, index=False)
        log(f"  {fname:<22}{mode:<10} done  ({(time.time() - t0) / 60:.1f} min)")

    r = pd.read_csv(RAW)
    t = (r.groupby(["family", "mode"]).acc.agg(["mean", "std", "count"])
         .unstack("mode"))
    rows = []
    for f in ORDER:
        row = dict(family=f, n_features=len(fam[f]))
        for m in ("shuffled", "recording", "session", "night"):
            if ("mean", m) in t.columns and pd.notna(t.loc[f, ("mean", m)]):
                row[f"{m}_mean"] = t.loc[f, ("mean", m)]
                row[f"{m}_sd"] = t.loc[f, ("std", m)]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(SUMMARY, index=False)

    log("")
    log(f"{'family':<22}{'n':>4}{'shuffled':>16}{'recording':>16}"
        f"{'session out':>16}")
    for _, x in out.iterrows():
        log(f"{x.family:<22}{int(x.n_features):>4}"
            f"{x.shuffled_mean:>9.4f}±{x.shuffled_sd:.4f}"
            f"{x.recording_mean:>9.4f}±{x.recording_sd:.4f}"
            f"{x.session_mean:>9.4f}±{x.session_sd:.4f}")
    a = out[out.family == "all features"].iloc[0]
    log("")
    log(f"night-grouped, all features (documents the old column): "
        f"{a.night_mean:.4f} ± {a.night_sd:.4f}")
    log("Table 8 recording-grouped reference: 0.7254 ± 0.0028")
    with open(LOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
