"""
Figures 7 and 8 for PCD as now defined (upper-tail p-value, within-participant
permutation), from the per-window outputs of 77_pcd_revised.py and
85_pcd_extra.py. Replaces the panels drawn by 22_figures.py and
24_label_noise_sweep.py from the withdrawn lower-tail score; headings describe
what is plotted.

  Figure 7  left: coverage C(alpha) against alpha for Sleep-EDF staging, DEED
            affect and DEED affect on frozen CBraMod embeddings, with the line
            C = alpha that bounds coverage under the null (Proposition 1).
            right: balanced accuracy of retained predictions against coverage
            for PCD, max-softmax and predictive entropy (risk-coverage).
  Figure 8  label-noise sweep: PCD coverage at alpha = 0.05, and max-softmax
            > 0.5 coverage, against the balanced accuracy actually achieved.
"""
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIG = os.path.join(HERE, "figures")
MANUSCRIPT = os.path.join("E:" + os.sep, "Dream_Analysis")
plt.rcParams.update({"font.family": "serif", "font.size": 7.5,
                     "figure.dpi": 300, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25,
                     "axes.axisbelow": True})
COL = {"Sleep-EDF staging": "#2A6F97", "DEED affect": "#C1352C",
       "DEED affect, CBraMod embeddings": "#E07A5F"}
RULE_LS = {"PCD": "-", "Max-softmax": "--", "Entropy": ":"}


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(FIG, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight")
        shutil.copyfile(p, os.path.join(MANUSCRIPT, f"{name}.{ext}"))
    plt.close(fig)


def fig7():
    w = pd.read_parquet(os.path.join(OUT, "pcd_revised_windows.parquet"))
    wc = pd.read_parquet(os.path.join(OUT, "pcd_extra_cbramod_windows.parquet"))
    w = pd.concat([w, wc], ignore_index=True)
    rc = pd.read_csv(os.path.join(OUT, "pcd_revised_riskcov.csv"))
    alphas = np.logspace(-3, np.log10(0.3), 40)
    # drawn at the printed width (full page, 7.0 in)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for cond, g in w.groupby("condition"):
        cov = [(g.p_pcd <= a).mean() for a in alphas]
        a1.plot(alphas, cov, color=COL[cond], lw=1.6, label=cond)
        a1.plot([0.05], [(g.p_pcd <= 0.05).mean()], "o", color=COL[cond],
                ms=4)
    a1.plot(alphas, alphas, color="black", lw=.9, ls="--",
            label=r"$C(\alpha)=\alpha$ (null bound)")
    a1.set_xscale("log")
    a1.set_xlabel(r"$\alpha$")
    a1.set_ylabel(r"Coverage $C(\alpha)$")
    a1.set_ylim(0, 1)
    a1.legend(fontsize=6.5, frameon=False, loc="center right")
    a1.set_title("(a) Coverage against the level", fontsize=7.5)

    for cond in ("Sleep-EDF staging", "DEED affect"):
        for rule, ls in RULE_LS.items():
            r = rc[(rc.condition == cond) & (rc.rule == rule)
                   ].sort_values("coverage")
            a2.plot(r.coverage, r.bal_retained, ls=ls, color=COL[cond],
                    lw=1.3, label=f"{cond}, {rule}")
    a2.axhline(1 / 3, color="black", lw=.8, ls=":")
    a2.set_xlabel("Coverage (share of windows retained)")
    a2.set_ylabel("Balanced accuracy of retained predictions")
    a2.set_ylim(0.2, 1.0)
    a2.legend(fontsize=6.5, frameon=False, loc="center",
              bbox_to_anchor=(0.66, 0.5), ncol=1)
    a2.set_title("(b) Risk-coverage, participant-separated", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig5_pcd")
    print("figure 7 (PCD) redrawn")


def fig8():
    t = pd.read_csv(os.path.join(OUT, "pcd_extra_noise.csv"))
    # drawn at the printed width (one column, 3.4 in); title in the caption
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(t.balanced, t.coverage, "o-", color="#2A6F97", lw=1.6,
            label=r"PCD, $\alpha=0.05$")
    ax.plot(t.balanced, t.msp_coverage, "s--", color="#8D99AE", lw=1.2,
            label="Max-softmax $>0.5$")
    for _, r in t.iterrows():
        ax.annotate(fr"$\eta={r.eta:.1f}$", (r.balanced, r.coverage),
                    textcoords="offset points", xytext=(4, -9), fontsize=6)
    ax.axhline(0.05, color="black", lw=.8, ls=":")
    ax.set_xlabel("Balanced accuracy achieved")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig8_label_noise")
    print("figure 8 (label noise) redrawn")


if __name__ == "__main__":
    import sys
    parts = sys.argv[1:] or ["fig7", "fig8"]
    for p in parts:
        {"fig7": fig7, "fig8": fig8}[p]()
