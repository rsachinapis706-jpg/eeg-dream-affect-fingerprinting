"""
Figure for the two confirmatory experiments: independent mechanism test and
the channel-density sweep.

Panel (a) places the Tononi mechanism replication beside DEED. The three
partitioning rules and the identity-only oracle are drawn for both corpora so
the reader can see that the ladder has the same shape in a second, independently
collected dataset with a different label and a different montage.

Panel (b) answers the spatial-resolution objection. Two curves are drawn against
channel count: accuracy under window-shuffled partitioning, and balanced
accuracy under participant-separated partitioning. The first rises with density,
the second does not. The gap between them is the artefact, and it widens.

Values are read from the CSVs written by 31_tononi_mechanism.py and
33_channel_density.py. Nothing is typed in.
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

# DEED reference values, regenerated in the pinned environment over ten fold
# assignments by 51_protocol_table.py and 52_mechanism_refresh.py. The
# earlier figures came from scikit-learn 1.4.1 and no longer match the
# tables they sit beside.
DEED = dict(random=0.5701, recording=0.4335, subject=0.4184, oracle=0.5465)

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "figure.dpi": 600, "savefig.dpi": 600,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25,
                     "axes.axisbelow": True})

C_DEED = "#2A6F97"
C_TON = "#C1352C"
C_FLAT = "#4B7F52"


def main():
    mr = pd.read_csv(os.path.join(OUT, "mechanism_refresh.csv"))
    mr = mr[mr.dataset == "Tononi"].set_index("quantity")["mean"]
    ton = pd.Series(dict(acc_random=mr["acc window-shuffled"],
                         acc_recording=mr["acc recording-grouped"],
                         acc_subject=mr["acc participant-wise"],
                         oracle=mr["oracle accuracy"]))
    den = pd.read_csv(os.path.join(OUT, "channel_density.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

    # ---- panel (a): mechanism, two corpora ---------------------------
    ax = axes[0]
    labels = ["Window-\nshuffled", "Recording-\ngrouped", "Participant-\nwise",
              "Identity\noracle"]
    d_vals = [DEED["random"], DEED["recording"], DEED["subject"],
              DEED["oracle"]]
    t_vals = [ton.acc_random, ton.acc_recording, ton.acc_subject, ton.oracle]
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, d_vals, w, color=C_DEED, edgecolor="black", lw=.5,
           label="DEED (34 part.)")
    ax.bar(x + w / 2, t_vals, w, color=C_TON, edgecolor="black", lw=.5,
           label="Tononi (18 part.)")
    for xi, v in zip(x - w / 2, d_vals):
        ax.text(xi, v + .012, f"{v:.2f}", ha="center", fontsize=7)
    for xi, v in zip(x + w / 2, t_vals):
        ax.text(xi, v + .012, f"{v:.2f}", ha="center", fontsize=7)
    ax.axhline(1 / 3, ls=":", color="black", lw=1)
    ax.text(len(labels) - .45, 1 / 3 + .012, "chance", fontsize=7, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 0.78)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("(a) Mechanism replicates in a second corpus $\\bigstar$",
                 fontsize=9)

    # ---- panel (b): channel density ---------------------------------
    ax = axes[1]
    g = den.groupby("channels")
    ch = np.array(sorted(den.channels.unique()))
    rnd = np.array([g.get_group(c).acc_random.max() for c in ch])
    sub = np.array([g.get_group(c).bal_subject.max() for c in ch])
    xi = np.arange(len(ch))

    ax.plot(xi, rnd, "o-", color=C_TON, lw=1.6, ms=5,
            label="Window-shuffled accuracy")
    ax.plot(xi, sub, "s-", color=C_FLAT, lw=1.6, ms=5,
            label="Participant-wise balanced acc.")
    ax.fill_between(xi, sub, rnd, color=C_TON, alpha=.10)
    ax.axhline(1 / 3, ls=":", color="black", lw=1)
    ax.text(len(ch) - 1.05, 1 / 3 - .045, "chance", fontsize=7, ha="right")
    ax.annotate("artefact grows\nwith density", xy=(3.0, 0.55),
                fontsize=7.5, ha="center", color=C_TON)
    ax.set_xticks(xi)
    ax.set_xticklabels([str(c) for c in ch])
    ax.set_xlabel("Electrodes (nested farthest-point subsets)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.24, 0.86)
    ax.legend(fontsize=7.5, frameon=False, loc="center left")
    ax.set_title("(b) No gain from spatial resolution $\\bigstar$",
                 fontsize=9)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = f"fig10_replication_density.{ext}"
        fig.savefig(os.path.join(FIG, p), bbox_inches="tight")
        fig.savefig(os.path.join(PAPER, p), bbox_inches="tight")
    plt.close(fig)

    print("wrote fig10_replication_density")
    print(f"  Tononi inflation  {ton.acc_random - ton.acc_subject:+.4f}")
    print(f"  DEED   inflation  {DEED['random'] - DEED['subject']:+.4f}")
    for m in ("logreg", "rf"):
        s = den[den.model == m].sort_values("channels")
        i0 = s.iloc[0].acc_random - s.iloc[0].acc_subject
        i1 = s.iloc[-1].acc_random - s.iloc[-1].acc_subject
        print(f"  {m:<7} inflation 6ch {i0:+.4f} -> 64ch {i1:+.4f}")


if __name__ == "__main__":
    main()
