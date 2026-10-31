"""
Figure 7 redrawn from the stored attribution results with neutral headings.

23_fingerprint_xai.py titled the right panel "Anatomy-dependent features
dominate" and the figure "OUR FINDING: ...". Attribution identifies which
inputs a model relies on, not why they differ between people (Reviewer 1,
comment 6), so the headings now describe what is plotted. Values are read
from fingerprint_families.csv, written by 23; nothing is refitted.
"""
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIG = os.path.join(HERE, "figures")
MANUSCRIPT = os.path.join("E:" + os.sep, "Dream_Analysis")


def main():
    fam = pd.read_csv(os.path.join(OUT, "fingerprint_families.csv"),
                      index_col=0)
    a = fam.loc["absolute band power", "impurity_share"]
    ap = fam.loc["aperiodic (1/f) slope", "impurity_share"]
    r_ = fam.loc["relative band power", "impurity_share"]
    assert (round(a, 4), round(ap, 4), round(r_, 4)) == (0.4910, 0.0596,
                                                         0.2682)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.4))
    f2 = fam.sort_values("impurity_share")
    cols = ["#C1352C" if i in ("absolute band power", "aperiodic (1/f) slope")
            else "#8D99AE" for i in f2.index]
    a1.barh(range(len(f2)), f2.impurity_share, color=cols, edgecolor="black",
            lw=.5)
    a1.set_yticks(range(len(f2)))
    a1.set_yticklabels(f2.index, fontsize=7.4)
    a1.set_xlabel("Share of importance for participant identity")
    a1.set_title("Importance by feature family", fontsize=9.5)

    names = ["Absolute\npower", "Aperiodic\n1/f slope", "Relative\npower"]
    vals = [a, ap, r_]
    a2.bar(names, vals, color=["#C1352C", "#C1352C", "#2A6F97"],
           edgecolor="black", lw=.6)
    for i, v in enumerate(vals):
        a2.text(i, v + .004, f"{v:.3f}", ha="center", fontsize=8.5,
                fontweight="bold")
    a2.set_ylabel("Share of importance")
    a2.set_title("Amplitude-dependent and amplitude-normalised families",
                 fontsize=9.5)
    fig.suptitle("Feature importance for the 34-way identity classifier "
                 "(random forest, window-shuffled)", fontsize=10, y=1.03)
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(FIG, f"fig7_fingerprint_xai.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        shutil.copyfile(p, os.path.join(MANUSCRIPT,
                                        f"fig7_fingerprint_xai.{ext}"))
    plt.close(fig)
    print("figure 7 redrawn")


if __name__ == "__main__":
    main()
