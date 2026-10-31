"""
DEED accounting quoted in Section IV-C (Reviewer 1, comment 3): windows,
recordings and participants at each exclusion step, class counts and shares at
the window and recording level, and participant contributions. Written to
out/deed_accounting.csv so every count and share in that paragraph has a file.
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
VALENCE = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
META = {"segment_id", "subject", "oid", "epoch_idx", "n_epochs",
        "epochs_from_end", "E", "film", "stage", "has_text"}


def main():
    rows = []

    def put(q, v):
        rows.append(dict(quantity=q, value=v))

    d = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    fc = [c for c in d.columns if c not in META]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    put("windows after rejection", len(d))
    put("recordings with windows", d.segment_id.nunique())
    put("participants", d.subject.nunique())
    e0 = d[d.E == "E0"]
    put("E0 windows", len(e0)); put("E0 recordings", e0.segment_id.nunique())
    put("E0 participants", e0.subject.nunique())
    v = d[d.E.isin(VALENCE)].assign(y=lambda x: x.E.map(VALENCE))
    put("valence windows", len(v)); put("valence recordings", v.segment_id.nunique())
    put("valence participants", v.subject.nunique())
    for c, n in v.y.value_counts().items():
        put(f"windows {c}", n); put(f"windows {c} share", round(n / len(v), 4))
    r = v.drop_duplicates("segment_id")
    for c, n in r.y.value_counts().items():
        put(f"recordings {c}", n); put(f"recordings {c} share",
                                       round(n / len(r), 4))
    rec_per = r.groupby("subject").size()
    win_per = v.groupby("subject").size()
    put("recordings per participant median", float(rec_per.median()))
    put("recordings per participant min", int(rec_per.min()))
    put("recordings per participant max", int(rec_per.max()))
    put("windows per participant median", float(win_per.median()))
    put("windows per participant min", int(win_per.min()))
    put("windows per participant max", int(win_per.max()))
    ncls = r.groupby("subject").y.nunique().value_counts()
    for k in (3, 2, 1):
        put(f"participants with {k} classes", int(ncls.get(k, 0)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "deed_accounting.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
