"""
Recording route on Tononi versus DEED, measured without labels (Reviewer 1,
comment 6; companion to 79 and 81).

83_identity_rule_cv.py found that Tononi's recording-level association between
participant and label does not exceed its permutation null (p = 0.23), unlike
DEED's (p = 0.0005), although Tononi's window-shuffled inflation is larger. The
other route is recognising the recording. For each test window under the
window-shuffled folds of 52 (seeds 20260727..36), this reports how often the
nearest training window (standardised features, Euclidean) comes from the
same recording and from the same participant, together with the structure
that governs it: windows per recording and recordings per participant.
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "ir", os.path.join(HERE, "81_identity_routes.py"))
ir = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ir)
OUT = ir.OUT
SEEDS = [20260727 + i for i in range(10)]


def main():
    d, fc = ir.ic.load()
    t = pd.read_parquet(os.path.join(OUT, "tononi_features.parquet"))
    tfc = [c for c in t.columns if c.startswith("ch")]
    t = t.replace([np.inf, -np.inf], np.nan).dropna(subset=tfc)
    sets = {
        "DEED": (d[fc].to_numpy(float), d.y.to_numpy(),
                 d.segment_id.to_numpy(), d.subject.to_numpy()),
        "Tononi": (t[tfc].to_numpy(float), t.dreaming.astype(str).to_numpy(),
                   t.recording.astype(str).to_numpy(),
                   t.subject.astype(str).to_numpy()),
    }
    rows = []
    for name, (X, y, rec, subj) in sets.items():
        wpr = pd.Series(rec).value_counts()
        rpp = pd.DataFrame(dict(r=rec, s=subj)).drop_duplicates().s.value_counts()
        for seed in SEEDS:
            sp = StratifiedKFold(5, shuffle=True,
                                 random_state=seed).split(y, y)
            r_, s_ = ir.nn_rates(X, rec, subj, sp)
            rows.append(dict(dataset=name, seed=seed, features=X.shape[1],
                             windows=len(y), recordings=len(wpr),
                             participants=len(rpp),
                             windows_per_recording_median=wpr.median(),
                             recordings_per_participant_median=rpp.median(),
                             nn_same_recording=r_, nn_same_participant=s_))
    a = pd.DataFrame(rows)
    a.round(4).to_csv(os.path.join(OUT, "tononi_routes.csv"), index=False)
    print(a.groupby("dataset").agg(
        features=("features", "first"), windows=("windows", "first"),
        recordings=("recordings", "first"),
        participants=("participants", "first"),
        wpr=("windows_per_recording_median", "first"),
        rpp=("recordings_per_participant_median", "first"),
        same_rec=("nn_same_recording", "mean"),
        same_rec_sd=("nn_same_recording", "std"),
        same_part=("nn_same_participant", "mean")).round(4).to_string())


if __name__ == "__main__":
    main()
