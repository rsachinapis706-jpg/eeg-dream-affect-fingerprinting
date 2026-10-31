"""
Summary of the PCD calibration check on DEED with labels re-randomised within
participant (pcd_revised_deednull.csv, written by 77_pcd_revised.py, one row
per re-randomisation). Writes the statistics quoted in Section IV-K so that
each traces to a file: mean, minimum and maximum marginal rate, the median
over re-randomisations of the worst participant's rate, and the mean share of
participants flagged at more than twice the nominal rate.

Output: out/pcd_revised_deednull_summary.csv
"""
import os

import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def main():
    d = pd.read_csv(os.path.join(OUT, "pcd_revised_deednull.csv"))
    s = pd.DataFrame([
        ("re-randomisations", len(d)),
        ("marginal rate mean", d.fpr.mean()),
        ("marginal rate min", d.fpr.min()),
        ("marginal rate max", d.fpr.max()),
        ("worst participant rate, median over re-randomisations",
         d.part_fpr_max.median()),
        ("share of participants above twice nominal, mean",
         d.part_frac_above_2alpha.mean()),
    ], columns=["quantity", "value"])
    s["value"] = s.value.round(4)
    s.to_csv(os.path.join(OUT, "pcd_revised_deednull_summary.csv"), index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
