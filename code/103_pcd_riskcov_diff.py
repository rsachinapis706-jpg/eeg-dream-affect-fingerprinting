"""
Differences in retained balanced accuracy between PCD and the confidence rules
at each coverage level (from pcd_revised_riskcov.csv, 77), quoted in Section
IV-K. Point estimates on the first fold assignment.
"""
import os

import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
r = pd.read_csv(os.path.join(OUT, "pcd_revised_riskcov.csv"))
rows = []
for cond, g in r.groupby("condition"):
    p = g.pivot(index="coverage", columns="rule", values="bal_retained")
    for cov, row in p.iterrows():
        rows.append(dict(condition=cond, coverage=cov,
                         pcd_minus_msp=row["PCD"] - row["Max-softmax"],
                         pcd_minus_entropy=row["PCD"] - row["Entropy"]))
d = pd.DataFrame(rows)
d.round(4).to_csv(os.path.join(OUT, "pcd_riskcov_differences.csv"), index=False)
for cond, g in d.groupby("condition"):
    print(f"{cond}: PCD-MSP min {g.pcd_minus_msp.min():+.4f} max "
          f"{g.pcd_minus_msp.max():+.4f} (at {g.loc[g.pcd_minus_msp.idxmax(), 'coverage']:.2f}); "
          f"PCD-entropy max {g.pcd_minus_entropy.max():+.4f}; "
          f"|PCD-MSP| max {g.pcd_minus_msp.abs().max():.4f}")
