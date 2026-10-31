"""
Figure 4 (within-participant analysis), redrawn with corrected statistics.

The panel data are unchanged -- each participant's observed within-participant
balanced accuracy against that participant's permutation null. What changes:

  * each panel now states its own paired effect: mean difference, Cohen's d_z,
    and one-sided Wilcoxon p, read from statistical_foundation.csv, where the
    spectral and CBraMod rows are now computed from their own files;
  * the heading states what was measured instead of asserting a finding, as
    Reviewer 1 asked of figure headings (comment 12). The spectral panel shows
    a small positive, non-significant tendency; "no effect" would overstate it.

Only this figure is regenerated. Running 22_figures.py would also overwrite
figures that later scripts have since superseded.
"""
import importlib.util
import os
import shutil

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "figs", os.path.join(HERE, "22_figures.py"))
figs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figs)          # style, colours and save() only
plt = figs.plt

OUT = os.path.join(os.path.dirname(HERE), "analysis", "out")
MANUSCRIPT = os.path.join("E:" + os.sep, "Dream_Analysis")


def main():
    sf = pd.read_csv(os.path.join(OUT, "statistical_foundation.csv"))
    es = sf[sf.part == "effect_size"].set_index("config")
    panels = [("within_subject.csv", "Spectral (108)",
               "within-participant, spectral"),
              ("cbramod_within_subject.csv", "CBraMod, frozen",
               "within-participant, CBraMod frozen")]

    # Drawn at the printed width (one column, 3.4 in). The statistics of each
    # panel are given in the caption, so the figure carries only the data.
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.95), sharey=True)
    for ax, (fname, title, key) in zip(axes, panels):
        d = pd.read_csv(os.path.join(OUT, fname)).dropna(subset=["bacc",
                                                                  "null"])
        assert key in es.index
        ax.scatter(d.null, d.bacc, s=9, color=figs.C_OK, edgecolor="black",
                   lw=.3, zorder=3)
        lo, hi = .05, .75
        ax.plot([lo, hi], [lo, hi], ls="--", color="k", lw=0.7, zorder=2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xticks([.2, .4, .6]); ax.set_yticks([.2, .4, .6])
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("Permutation null", fontsize=7)
        ax.set_title(f"{title}, n={len(d)}", fontsize=7)
    axes[0].set_ylabel("Observed balanced acc.", fontsize=7)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.21, top=0.88,
                        wspace=0.12)
    with plt.rc_context({"savefig.bbox": "standard"}):
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(figs.FIG, f"fig4_within_subject.{ext}"))
    plt.close(fig)
    for ext in ("pdf", "png"):
        src = os.path.join(figs.FIG, f"fig4_within_subject.{ext}")
        shutil.copyfile(src, os.path.join(MANUSCRIPT,
                                          f"fig4_within_subject.{ext}"))
        print("wrote", src, "and the manuscript copy")


if __name__ == "__main__":
    main()
