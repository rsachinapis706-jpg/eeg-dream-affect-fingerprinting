"""
Table 9's "recording-grouped" column, identified and corrected (Reviewer 1,
comment 2). Time-limited replacement for 71_identity_basis.py.

48_identity_basis.py produced Table 9 on all 7,418 windows with one fold
assignment (seed 20260727), grouping its middle column by `oid` -- participant
and NIGHT -- not by `segment_id`, the recording. This script, with the same
windows, seed, classifier and scoring as 48:

  1. reproduces the published middle-column value with night grouping, which
     identifies what the column measured;
  2. computes the value with true recording grouping on the same windows;
  3. computes the recording-grouped value on Table 8's 6,144 valence windows
     at the same seed, for the like-for-like comparison with Table 8 (whose
     0.7254 is a mean over ten fold assignments).

All features (108). Random forest, 300 trees, min_samples_leaf = 2; accuracy on
test windows whose participant appears in training, as in 48 and 52.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
SEED = 20260727
META = {"segment_id", "subject", "oid", "epoch_idx", "n_epochs",
        "epochs_from_end", "E", "film", "stage", "has_text"}
VALENCE = {"E1", "E2", "E3", "E4", "E5"}


def decode(X, y, groups):
    yt, yp = [], []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED
                                       ).split(X, y, groups):
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   n_jobs=22, random_state=SEED).fit(X[tr],
                                                                     y[tr])
        seen = np.isin(y[te], np.unique(y[tr]))
        yp.append(m.predict(X[te][seen])); yt.append(y[te][seen])
    return accuracy_score(np.concatenate(yt), np.concatenate(yp))


def main():
    d = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    fc = [c for c in d.columns if c not in META]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    rows = []
    for wname, dd in (("all 7,418 windows (Table 9)", d),
                      ("6,144 valence windows (Table 8)",
                       d[d.E.isin(VALENCE)])):
        X, y = dd[fc].to_numpy(float), dd.subject.to_numpy()
        for gname, col in (("night (oid)", "oid"),
                           ("recording (segment_id)", "segment_id")):
            if wname.startswith("6,144") and col == "oid":
                continue
            a = decode(X, y, dd[col].to_numpy())
            rows.append(dict(windows=wname, grouping=gname, n_groups=dd[col]
                             .nunique(), accuracy=a))
            print(f"{wname:<34}{gname:<24}{a:.4f}", flush=True)
    pd.DataFrame(rows).round(4).to_csv(
        os.path.join(OUT, "identity_grouping_fix.csv"), index=False)
    print("published Table 9 middle column, all features: 0.4862")
    print("published Table 8 recording-grouped (10-seed mean): 0.7254")


if __name__ == "__main__":
    main()
