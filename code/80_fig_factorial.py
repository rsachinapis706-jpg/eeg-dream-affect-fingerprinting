"""
Figure 1: the effect of each protocol choice with the others held fixed.

Replaces the bar chart drawn by 22_figures.py from 10_leakage_mechanism.py,
whose bars used the best of three classifiers per ordering and compared rows
that differed in more than one factor (Reviewer 1, comment 4). Plotted here:
paired differences in balanced accuracy on real test windows from
73b_factorial_effects.py, with 95% participant-bootstrap intervals.
"""
import importlib.util
import os
import shutil

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "figs", os.path.join(HERE, "22_figures.py"))
figs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figs)
plt = figs.plt
OUT = os.path.join(os.path.dirname(HERE), "analysis", "out")
MANUSCRIPT = os.path.join("E:" + os.sep, "Dream_Analysis")

ROWS = [
    ("selection before vs inside", "participant/before/none",
     "Selection before split\n(participant-separated)"),
    ("selection before vs inside", "window/before/none",
     "Selection before split\n(window-shuffled)"),
    ("oversampling before vs inside", "participant/none/before",
     "Oversampling before split\n(participant-separated)"),
    ("oversampling before vs inside", "window/none/before",
     "Oversampling before split\n(window-shuffled)"),
    ("window vs participant partition", "window/none/none",
     "Window-level partition\n(nothing before split)"),
    ("window vs participant partition", "window/none/before",
     "Window-level partition\n(oversampling before split)"),
]


def fmt(v):
    # an effect that rounds to zero at four decimals is printed at five, so
    # the figure never shows a signed zero
    return f"{v:+.4f}" if abs(v) >= 0.00005 else f"{v:+.5f}"


def main():
    # the complete factorial (73 --full, 73b --full) when it exists
    sfx = ("_full" if os.path.exists(os.path.join(
        OUT, "protocol_factorial_effects_full.csv")) else "")
    e = pd.read_csv(os.path.join(OUT, f"protocol_factorial_effects{sfx}.csv"))
    per = pd.read_csv(os.path.join(OUT, f"protocol_factorial_effects_by_seed{sfx}.csv"))
    n_seed = per.seed.nunique()
    # drawn at the printed width (one column, 3.4 in), so 7 pt stays 7 pt
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ys, labels = [], []
    for i, (eff, a, lab) in enumerate(ROWS):
        r = e[(e.effect == eff) & (e.a == a)].iloc[0]
        y = len(ROWS) - 1 - i
        col = figs.C_BAD if r.ci_lo > 0 else figs.C_NEU
        ax.errorbar(r.d_bal, y, xerr=[[r.d_bal - r.ci_lo], [r.ci_hi - r.d_bal]],
                    fmt="o", color=col, ecolor=col, capsize=2, ms=4, lw=1.1)
        v = per[(per.effect == eff) & (per.a == a)].d_bal
        assert len(v) == n_seed
        ax.plot(v, [y + 0.25] * len(v), "x", color="0.45", ms=3, mew=0.8)
        top = max(r.ci_hi, v.max(), 0)
        ax.text(top + 0.012, y, fmt(v.mean()), va="center", fontsize=6.5)
        ys.append(y); labels.append(lab)
    ax.axvline(0, color="k", lw=0.8)
    ax.axhline(1.5, color="0.7", lw=.5, ls=":")
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlabel("Change in balanced accuracy\non real test windows",
                  fontsize=7)
    ax.set_xlim(-0.06, 0.53)
    # the caption explains dots, bars and crosses; no title inside the figure
    # fixed margins, no tight crop: the saved figure is exactly 3.4 in wide
    fig.subplots_adjust(left=0.40, right=0.98, bottom=0.18, top=0.98)
    with plt.rc_context({"savefig.bbox": "standard"}):
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(figs.FIG, f"fig1_leakage_ladder.{ext}"))
    plt.close(fig)
    for ext in ("pdf", "png"):
        shutil.copyfile(os.path.join(figs.FIG, f"fig1_leakage_ladder.{ext}"),
                        os.path.join(MANUSCRIPT, f"fig1_leakage_ladder.{ext}"))
    print("figure 1 redrawn")


if __name__ == "__main__":
    main()
