"""
Figure 9: participant-label association across the DREAM catalogue, with DEED
on the same scale.

Replaces the figure drawn by 32_catalogue_mi.py, which set DEED's RAW
normalised mutual information (0.1624, window level) as a dashed line among the
catalogue datasets' permutation-ADJUSTED excess values. Reviewer 1 noted that
the two are not commensurate. Here DEED is a bar computed by the identical
procedure as every catalogue dataset: one label per awakening, the same
inclusion rule, 10,000 label shuffles (66_catalogue_audit.py).

Bars are shaded by the Holm-adjusted p across the twelve catalogue datasets;
DEED is outside that family and drawn in its own colour.
"""
import os
import shutil

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.patches import Patch                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
DESTS = [os.path.join(ROOT, "analysis", "figures"),
         os.path.join(ROOT, "paper"),
         os.path.join("E:" + os.sep, "Dream_Analysis")]
NAME = "fig9b_catalogue_datasets"

plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "figure.dpi": 600, "savefig.dpi": 600,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25,
                     "axes.axisbelow": True})

HOLM, RAW, NS, DEED = "#C1352C", "#E8A09A", "#8D99AE", "#2A6F97"


def main():
    a = pd.read_csv(os.path.join(OUT, "catalogue_audit.csv"))
    a = a.sort_values("nmi_excess").reset_index(drop=True)

    def colour(x):
        if str(x.set_id) == "DEED":
            return DEED
        if x.p_holm < 0.05:
            return HOLM
        return RAW if x.p < 0.05 else NS

    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    ax.barh(range(len(a)), a.nmi_excess, color=[colour(x) for _, x in
                                                a.iterrows()],
            edgecolor="black", lw=.5, height=.72)
    ax.axvline(0, color="k", lw=1)
    labels = [("DEED (valence)" if str(x.set_id) == "DEED" else x.dataset)
              + f"  [{int(x.participants)}]" for _, x in a.iterrows()]
    ax.set_yticks(range(len(a)))
    ax.set_yticklabels(labels, fontsize=7.5)
    for i, x in a.iterrows():
        if str(x.set_id) == "DEED":
            ax.get_yticklabels()[i].set_fontweight("bold")
    ax.set_xlabel("Excess $I(\\mathrm{participant};\\mathrm{label})/H"
                  "(\\mathrm{label})$ over permutation null")
    ax.set_title("Participant-label association in catalogued dream datasets",
                 fontsize=10)
    ax.legend(handles=[Patch(fc=HOLM, ec="k", lw=.5,
                             label="Holm-adjusted $p<0.05$"),
                       Patch(fc=RAW, ec="k", lw=.5,
                             label="$p<0.05$ uncorrected only"),
                       Patch(fc=NS, ec="k", lw=.5, label="not significant"),
                       Patch(fc=DEED, ec="k", lw=.5,
                             label="DEED, same procedure")],
              fontsize=7, loc="lower right", frameon=True)
    fig.tight_layout()

    first = None
    for d in DESTS:
        os.makedirs(d, exist_ok=True)
        for ext in ("pdf", "png"):
            p = os.path.join(d, f"{NAME}.{ext}")
            if first is None or not first.get(ext):
                fig.savefig(p, bbox_inches="tight")
                first = (first or {}) | {ext: p}
            else:
                shutil.copyfile(first[ext], p)
            print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
