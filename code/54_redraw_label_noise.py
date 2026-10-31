"""
Redraw the label-noise figure from the stored sweep.

24_label_noise_sweep.py computes the sweep and draws the figure in one pass.
The sweep is expensive and its output is already on disk, so regenerating the
figure should not require recomputing it. This script reads
label_noise_sweep.csv and nothing else, which is also the property the Data
Availability Statement claims for every figure in the paper: the figure is a
rendering of a result file, not a separate computation that could silently
disagree with it.

Two corrections relative to the earlier version of this figure:

  1. The title asserted that "coverage does not track signal strength", which
     contradicts the plot beneath it. Within a dataset the relationship is
     monotonic, with Spearman correlation 1.0000. What does not transfer is the
     comparison ACROSS datasets, which is a different claim and belongs in the
     caption rather than on the axes.

  2. The figure did not use the serif style shared by every other figure, so it
     appeared in a different typeface in the compiled manuscript.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
FIG = os.path.join(ROOT, "analysis", "figures")
PAPER = os.path.join(ROOT, "paper")
TEXDIR = os.path.join("E:" + os.sep, "Dream_Analysis")

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "figure.dpi": 600, "savefig.dpi": 600,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25,
                     "axes.axisbelow": True})


def main():
    res = pd.read_csv(os.path.join(OUT, "label_noise_sweep.csv"))
    v = res.dropna().sort_values("balanced")

    rho = pd.Series(v.balanced).corr(pd.Series(v.pcd_coverage),
                                     method="spearman")
    pear = float(np.corrcoef(v.balanced, v.pcd_coverage)[0, 1])

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(v.balanced, v.pcd_coverage, "o-", color="#2A6F97", ms=6, lw=1.8)
    for _, r in v.iterrows():
        ax.annotate(f"{r.noise:.0%}", (r.balanced, r.pcd_coverage),
                    textcoords="offset points", xytext=(6, -10), fontsize=7.5)
    ax.axhline(0.05, ls=":", color="#C1352C", lw=1.2)
    ax.text(v.balanced.max(), 0.075, "nominal $\\alpha$ = 0.05", fontsize=7.5,
            color="#C1352C", ha="right")
    ax.set_xlabel("True decodability (balanced accuracy, "
                  "participant-separated)")
    ax.set_ylabel("PCD coverage at $\\alpha$=0.05")
    ax.set_ylim(-.03, 1.03)
    ax.set_title("OUR FINDING: within a dataset, PCD coverage tracks\n"
                 f"true decodability (Spearman {rho:.2f})", fontsize=9.5)

    fig.tight_layout()
    for d in (FIG, PAPER, TEXDIR):
        if os.path.isdir(d):
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(d, f"fig8_label_noise.{ext}"),
                            bbox_inches="tight")
    plt.close(fig)

    print("redrew fig8_label_noise from label_noise_sweep.csv")
    print(f"  noise levels : {len(v)}")
    print(f"  coverage     : {v.pcd_coverage.min():.4f} to "
          f"{v.pcd_coverage.max():.4f}")
    print(f"  Spearman     : {rho:.4f}   Pearson: {pear:.4f}")


if __name__ == "__main__":
    main()
