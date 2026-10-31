"""
The cross-validated identity rule over all ten fold assignments, on DEED and
on Tononi (Reviewer 1, comment 6).

The identity-only "oracle" of 52_mechanism_refresh.py reads each participant's
majority label from all of their labels, including the ones it then scores:
descriptive, not a predictive baseline. Here the rule is estimated like any
other predictor: in each fold, each participant's majority label is taken from
that participant's training windows and applied to their test windows; a
participant absent from the training fold receives the training majority
class. Folds are exactly those of 52 and of Table 4(a) (StratifiedKFold or
StratifiedGroupKFold by recording, five folds, seeds 20260727..36), so the rule
and the random forest are compared on identical held-out windows.

Under window-shuffled folds the rule can also draw on the held-out
recording itself, because most of that recording's windows are in training and
count towards the participant's majority. A second variant removes that route:
the participant's majority is estimated from training windows of their OTHER
recordings only (ties broken towards the class listed first in CLASSES; a
participant with no other training recording receives the training majority
class). The difference between the two variants on identical folds is the part
of the rule's score that comes from the held-out recording.

Standard deviations across fold assignments use the population form (ddof=0),
as in 51_protocol_table.py and 52_mechanism_refresh.py.

Also reported for Tononi, as 74 does for DEED: recording-level
I(participant; label) / H(label) with a 2,000-permutation test (labels shuffled
across recordings).

No model is fitted; the script runs in seconds.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             mutual_info_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
SEEDS = [20260727 + i for i in range(10)]
N_SPLITS = 5
N_PERM = 2000
VALENCE = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
META = {"segment_id", "subject", "oid", "epoch_idx", "n_epochs",
        "epochs_from_end", "E", "film", "stage", "has_text"}


def splits(y, groups, seed):
    if groups is None:
        return StratifiedKFold(N_SPLITS, shuffle=True,
                               random_state=seed).split(y, y)
    return StratifiedGroupKFold(N_SPLITS, shuffle=True,
                                random_state=seed).split(y, y, groups)


def rule(y, subj, sp):
    ide = np.empty(len(y), dtype=object)
    base = np.empty(len(y), dtype=object)
    for tr, te in sp:
        t = pd.DataFrame(dict(s=subj[tr], y=y[tr]))
        glob = t.y.value_counts().idxmax()
        maj = t.groupby("s").y.agg(lambda v: v.value_counts().idxmax())
        ide[te] = pd.Series(subj[te]).map(maj).fillna(glob).to_numpy()
        base[te] = glob
    return ide, base


def rule_other_recordings(y, subj, rec, sp, classes):
    """Participant majority from training windows of other recordings only."""
    ci = {c: k for k, c in enumerate(classes)}
    yi = np.array([ci[v] for v in y])
    out = np.empty(len(y), dtype=object)
    for tr, te in sp:
        glob = classes[int(np.argmax(np.bincount(yi[tr], minlength=len(classes))))]
        ps = pd.crosstab(subj[tr], yi[tr]).reindex(columns=range(len(classes)),
                                                   fill_value=0)
        rs = pd.crosstab(rec[tr], yi[tr]).reindex(columns=range(len(classes)),
                                                  fill_value=0)
        for i in te:
            if subj[i] not in ps.index:
                out[i] = glob
                continue
            c = ps.loc[subj[i]].to_numpy().copy()
            if rec[i] in rs.index:
                c = c - rs.loc[rec[i]].to_numpy()
            out[i] = classes[int(np.argmax(c))] if c.sum() > 0 else glob
    return out


def rec_assoc(y, subj, rec, seed):
    r = pd.DataFrame(dict(y=y, s=subj, r=rec))
    assert (r.groupby("r").y.nunique() == 1).all(), "label varies in recording"
    r = r.drop_duplicates("r")
    h = mutual_info_score(r.y, r.y)
    obs = mutual_info_score(r.s, r.y) / h
    rng = np.random.default_rng(seed)
    null = np.array([mutual_info_score(r.s, rng.permutation(r.y.to_numpy())) / h
                     for _ in range(N_PERM)])
    return obs, obs - null.mean(), (1 + (null >= obs).sum()) / (1 + N_PERM)


def run(name, y, subj, rec, classes):
    rows = []
    for seed in SEEDS:
        for part, grp in (("window-shuffled", None),
                          ("recording-grouped", rec)):
            sp = list(splits(y, grp, seed))
            ide, base = rule(y, subj, sp)
            preds = [("identity rule (CV)", ide), ("majority class", base)]
            if part == "window-shuffled":
                preds.append(("identity rule (CV), other recordings only",
                              rule_other_recordings(y, subj, rec, sp,
                                                    classes)))
            for pred, p in preds:
                rows.append(dict(dataset=name, partition=part, seed=seed,
                                 predictor=pred, acc=accuracy_score(y, p),
                                 bal=balanced_accuracy_score(y, p)))
    return rows


def main():
    d = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    fc = [c for c in d.columns if c not in META]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = d[d.E.isin(VALENCE)].reset_index(drop=True)
    y, s, r = (d.E.map(VALENCE).to_numpy(), d.subject.to_numpy(),
               d.segment_id.to_numpy())
    print(f"DEED: {len(y)} windows, {len(set(s))} participants, "
          f"{len(set(r))} recordings")
    rows = run("DEED", y, s, r, ["neg", "neu", "pos"])

    t = pd.read_parquet(os.path.join(OUT, "tononi_features.parquet"))
    tfc = [c for c in t.columns if c.startswith("ch")]
    t = t.replace([np.inf, -np.inf], np.nan).dropna(subset=tfc)
    yt = t.dreaming.astype(str).to_numpy()
    st = t.subject.astype(str).to_numpy()
    rt = t.recording.astype(str).to_numpy()
    print(f"Tononi: {len(yt)} windows, {len(set(st))} participants, "
          f"{len(set(rt))} recordings")
    rows += run("Tononi", yt, st, rt, sorted(set(yt)))

    a = pd.DataFrame(rows)
    a.to_csv(os.path.join(OUT, "identity_rule_cv10.csv"), index=False)
    summ = a.groupby(["dataset", "partition", "predictor"]).agg(
        acc_mean=("acc", "mean"), acc_sd=("acc", lambda v: v.std(ddof=0)),
        bal_mean=("bal", "mean"), bal_sd=("bal", lambda v: v.std(ddof=0)),
        n_assignments=("seed", "nunique")).reset_index()
    print(summ.round(4).to_string(), flush=True)

    assoc = []
    for name, yy, ss, rr, seed in (("DEED", y, s, r, 2),
                                   ("Tononi", yt, st, rt, 3)):
        obs, ex, p = rec_assoc(yy, ss, rr, seed)
        hw = mutual_info_score(ss, yy) / mutual_info_score(yy, yy)
        assoc.append(dict(dataset=name, window_level_I_over_H=hw,
                          recording_level_I_over_H=obs,
                          excess_over_permutation_mean=ex, p=p,
                          n_perm=N_PERM))
        print(f"{name}: window-level I/H {hw:.4f} | recording-level I/H "
              f"{obs:.4f}, excess over permutation mean {ex:.4f}, p {p:.4f}",
              flush=True)
    summ.round(4).to_csv(os.path.join(OUT, "identity_rule_cv10_summary.csv"),
                         index=False)
    pd.DataFrame(assoc).round(4).to_csv(
        os.path.join(OUT, "identity_association.csv"), index=False)


if __name__ == "__main__":
    main()
