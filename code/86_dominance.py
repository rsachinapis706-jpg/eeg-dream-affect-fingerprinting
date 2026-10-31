"""
Within-participant label dominance on DEED and Tononi, computed the same way
for both (replaces the DEED constants typed into 31_tononi_mechanism.py).

31 computed Tononi's dominance over RECORDINGS (share of a participant's
recordings in their most frequent class) but compared it with a DEED value of
0.750 copied from an earlier output, which is not the recording-level quantity:
Section IV-C reports DEED's recording-level median as 0.5762. Both levels are
reported here for both corpora, on the windows used in the replication table
(DEED: 6,144 valence windows; Tononi: 668 windows).
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
VALENCE = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
META = {"segment_id", "subject", "oid", "epoch_idx", "n_epochs",
        "epochs_from_end", "E", "film", "stage", "has_text"}


def dom(df, s, r, y):
    rec = df.drop_duplicates(r)
    assert (df.groupby(r)[y].nunique() == 1).all()
    by_rec = rec.groupby(s)[y].agg(lambda v: v.value_counts(normalize=True)
                                   .max())
    by_win = df.groupby(s)[y].agg(lambda v: v.value_counts(normalize=True)
                                  .max())
    return by_rec, by_win


def main():
    d = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    fc = [c for c in d.columns if c not in META]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = d[d.E.isin(VALENCE)].assign(y=lambda x: x.E.map(VALENCE))
    t = pd.read_parquet(os.path.join(OUT, "tononi_features.parquet"))
    tfc = [c for c in t.columns if c.startswith("ch")]
    t = t.replace([np.inf, -np.inf], np.nan).dropna(subset=tfc)
    t = t.assign(y=t.dreaming.astype(str), s=t.subject.astype(str),
                 r=t.recording.astype(str))
    rows = []
    for name, df, s, r in (("DEED", d, "subject", "segment_id"),
                           ("Tononi", t, "s", "r")):
        br, bw = dom(df, s, r, "y")
        for level, v in (("recordings", br), ("windows", bw)):
            rows.append(dict(dataset=name, level=level, windows=len(df),
                             participants=len(v), median=v.median(),
                             share_above_0_70=(v > 0.70).mean(),
                             n_above_0_70=int((v > 0.70).sum())))
    out = pd.DataFrame(rows)
    out.round(4).to_csv(os.path.join(OUT, "dominance.csv"), index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
