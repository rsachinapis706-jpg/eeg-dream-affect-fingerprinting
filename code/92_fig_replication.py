"""
Figure 5 (replication and channel density), redrawn (Reviewer 1, comments 4,
6 and 8).

34_replication_figure.py plotted window-shuffled ACCURACY against
participant-separated BALANCED accuracy and, at every electrode count, took
the better of two classifiers for each curve; panel (a) showed an identity
"oracle" fitted to the labels it scored. Here:

  (a) balanced accuracy under the three partitions, DEED and Tononi, random
      forest, mean and SD over ten fold assignments; beside them the identity
      rule estimated on training folds only, scored on the same held-out
      windows, with and without the held-out recording's own windows
  (b) balanced accuracy under window-shuffled and participant-separated
      five-fold partitioning against electrode count, one line per model
      (logistic regression C = 0.1 and random forest, as published), mean and
      SD over five fold assignments; nothing is maximised over models

Sources: protocol_table.csv (51), mechanism_refresh.csv (52),
identity_rule_cv10_summary.csv (83), density_both_partitions.csv (75).
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

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "figure.dpi": 300, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25,
                     "axes.axisbelow": True})
C_DEED, C_TON = "#2A6F97", "#C1352C"
C_WS, C_PS = "#C1352C", "#4B7F52"


def main():
    pt = pd.read_csv(os.path.join(OUT, "protocol_table.csv"))
    pt = pt[pt.target == "Dream valence"].set_index("held_out")
    mr = pd.read_csv(os.path.join(OUT, "mechanism_refresh.csv"))
    mr = mr[mr.dataset == "Tononi"].set_index("quantity")
    ir = pd.read_csv(os.path.join(OUT, "identity_rule_cv10_summary.csv"))
    den = pd.read_csv(os.path.join(OUT, "density_both_partitions.csv"))

    def rule(ds, part, pred):
        r = ir[(ir.dataset == ds) & (ir.partition == part)
               & (ir.predictor == pred)].iloc[0]
        return r.bal_mean, r.bal_sd

    deed = [(pt.loc["Random (window-shuffled)", "bal_mean"],
             pt.loc["Random (window-shuffled)", "bal_sd"]),
            (pt.loc["Recording", "bal_mean"], pt.loc["Recording", "bal_sd"]),
            (pt.loc["Participant", "bal_mean"], pt.loc["Participant", "bal_sd"]),
            rule("DEED", "window-shuffled", "identity rule (CV)"),
            rule("DEED", "window-shuffled",
                 "identity rule (CV), other recordings only")]
    ton = [(mr.loc["acc window-shuffled", "bal_mean"],
            mr.loc["acc window-shuffled", "bal_sd"]),
           (mr.loc["acc recording-grouped", "bal_mean"],
            mr.loc["acc recording-grouped", "bal_sd"]),
           (mr.loc["acc participant-wise", "bal_mean"],
            mr.loc["acc participant-wise", "bal_sd"]),
           rule("Tononi", "window-shuffled", "identity rule (CV)"),
           rule("Tononi", "window-shuffled",
                "identity rule (CV), other recordings only")]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4),
                             gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    labels = ["Window-\nshuffled", "Recording-\ngrouped", "Participant-\nseparated",
              "Identity\nrule", "Identity rule,\nother rec."]
    x = np.arange(len(labels))
    w = 0.38
    for off, vals, col, lab in ((-w / 2, deed, C_DEED, "DEED (34 part.)"),
                                (w / 2, ton, C_TON, "Tononi (18 part.)")):
        m = [v[0] for v in vals]
        e = [v[1] for v in vals]
        ax.bar(x + off, m, w, yerr=e, color=col, edgecolor="black", lw=.5,
               capsize=2, error_kw=dict(lw=.7), label=lab)
        for xi, v, ee in zip(x + off, m, e):
            ax.text(xi, v + ee + .015, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=6, rotation=90)
    ax.axvline(2.5, color="0.6", lw=.7, ls=":")
    ax.axhline(1 / 3, ls="--", color="black", lw=.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title("(a) Random forest (left) and no-EEG rule (right)",
                 fontsize=8.5)

    ax = axes[1]
    ch = sorted(den.channels.unique())
    xi = np.arange(len(ch))
    for model, mk, name in (("lr_fixed", "o", "logistic regression"),
                            ("rf", "s", "random forest")):
        for part, col, ls in (("window-shuffled", C_WS, "-"),
                              ("participant-separated", C_PS, "--")):
            s = den[(den.model == model) & (den.partition == part)
                    ].set_index("channels").loc[ch]
            assert (s.n_assignments == 5).all()
            ax.errorbar(xi, s.bal_mean, yerr=s.bal_sd, fmt=mk + ls, color=col,
                        ms=4, lw=1.2, capsize=2,
                        mfc=col if model == "rf" else "white",
                        label=f"{part}, {name}")
    ax.axhline(1 / 3, ls=":", color="black", lw=.9)
    ax.set_xticks(xi)
    ax.set_xticklabels([str(c) for c in ch])
    ax.set_xlabel("Electrodes (nested farthest-point subsets)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.2, 0.85)
    ax.legend(fontsize=6, frameon=False, loc="center right",
              bbox_to_anchor=(1.0, 0.40))
    ax.set_title("(b) Tononi, both partitions, per model", fontsize=8.5)

    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(FIG, f"fig10_replication_density.{ext}")
        fig.savefig(p, bbox_inches="tight")
        shutil.copyfile(p, os.path.join(MANUSCRIPT,
                                        f"fig10_replication_density.{ext}"))
    plt.close(fig)
    print("figure 5 (replication and density) redrawn")


if __name__ == "__main__":
    main()
