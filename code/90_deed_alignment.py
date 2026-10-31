"""
How DEED's EEG segments line up with the reports (Reviewer 1, comment 3).

DEED distributes one .mat file per awakening; the reported-emotion code (E0-E5)
and a sleep-stage tag are fields of the filename, so every label used in this
paper is attached to its EEG by the dataset itself, not by any join made here.
The accompanying ratings workbook lists, per report, a start and an end time.
It can be matched to segments only by position within the night, and only
where a night has as many reports as segments (01_build_master_table.py).

This script measures, for every positionally matched segment with both times,
the ratio of the segment's sample count to the listed interval. A ratio of the
sampling rate (200 Hz) means the file holds exactly the listed interval. It
also reports the stage tags, the segment lengths, and the coverage of each
join, so the data description can state what is known and what is not.
"""
import os

import numpy as np
import pandas as pd
import scipy.io as sio

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def secs(t):
    h, m, s = (int(x) for x in str(t).split(":"))
    return 3600 * h + 60 * m + s


def main():
    m = pd.read_csv(os.path.join(OUT, "master_table.csv"))
    rows = []
    for _, r in m.iterrows():
        n = sio.loadmat(r.path, variable_names=["Data"])["Data"].shape[1]
        dur = np.nan
        if (r.join_ok and pd.notna(r.REM_period_start_time)
                and pd.notna(r.End_time_of_REM_period)):
            d = secs(r.End_time_of_REM_period) - secs(r.REM_period_start_time)
            dur = d + 86400 if d < 0 else d
        rows.append(dict(segment=os.path.basename(r.path)[:-4], oid=r.oid,
                         E=r.E, stage=r.stage, join_ok=bool(r.join_ok),
                         samples=n, seconds_at_200hz=n / 200.0,
                         listed_seconds=dur,
                         ratio=(n / dur) if dur and dur > 0 else np.nan))
    a = pd.DataFrame(rows)
    a.to_csv(os.path.join(OUT, "deed_alignment.csv"), index=False)
    v = a.dropna(subset=["ratio"])
    print(f"segments {len(a)} | positionally matched to a report "
          f"{int(a.join_ok.sum())} | with both listed times {len(v)}")
    print("samples / listed interval: "
          f"median {v.ratio.median():.2f}, quartiles "
          f"{v.ratio.quantile(.25):.2f}-{v.ratio.quantile(.75):.2f}, "
          f"within 1% of 200 Hz: {int((abs(v.ratio - 200) <= 2).sum())} "
          f"of {len(v)}")
    print("segment length (s): median "
          f"{a.seconds_at_200hz.median():.0f}, range "
          f"{a.seconds_at_200hz.min():.0f}-{a.seconds_at_200hz.max():.0f}")
    print("stage tag in filename:")
    print(a.stage.fillna("none").value_counts().to_string())
    print("stage tag x valence code (E1-E5 only):")
    b = a[a.E.isin(["E1", "E2", "E3", "E4", "E5"])]
    print(pd.crosstab(b.stage.fillna("none"), b.E).to_string())


if __name__ == "__main__":
    main()
