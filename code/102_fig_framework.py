"""
Figure 1 (four-phase framework), redrawn with every number read from the
result files and with PCD as now defined; replaces 25_framework_figure.py,
whose boxes carried withdrawn values (identity oracle, collapsed CBraMod run,
the lower-tail score D(x) and its calibration figures). Also scopes the
caption and the introductory sentence on the foundation model (Reviewer 1,
comment 9; Reviewer 2 on scope).
"""
import io
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIG = os.path.join(HERE, "figures")
MANUSCRIPT = os.path.join("E:" + os.sep, "Dream_Analysis")
TEX = os.path.join(MANUSCRIPT, "accesstemp.tex")
C_HDR, C_BOX, C_EDGE, C_OUT, C_OUTE = ("#1F3864", "#EDF2FA", "#2A6F97",
                                       "#E8F1E4", "#4C8C3F")


def rd(n, **k):
    return pd.read_csv(os.path.join(OUT, n), **k)


def main():
    c = rd("config_table.csv").set_index("config")
    fp = rd("fingerprint_table.csv").set_index("quantity")["value"]
    lad = rd("random_label_ladder.csv")
    free5 = lad[(lad["mode"] == "free") & (lad.rung == 5)
                & (lad.column == "leaky")].iloc[0]
    st = rd("leakage_stack.csv").set_index("rung")
    cb = rd("cbramod_diag_summary.csv").set_index("config")
    pe = rd("cbramod_diag_pretraining_effect.csv").set_index("pretrained")
    dn = rd("pcd_revised_deednull.csv")
    si = rd("pcd_revised_sims.csv")
    ec = rd("expanded_control.csv")
    ctrl = ec[ec.task.str.startswith("W/N1/N2")
              & (ec.model == "logreg")].balanced.iloc[0]   # three-class, LR
    fm_ctrl = rd("cbramod_expanded.csv").set_index("model").loc[
        "cbramod", "balanced"]                          # matched CBraMod control
    fm_100 = rd("cbramod_expanded_uv100.csv").set_index("model").loc[
        "cbramod", "balanced"]                          # input in units of 100 uV
    # the primary procedure (120): 20 random participant partitions
    pri = rd("primary_random_summary.csv").set_index("config")
    pri_best = pri.observed_mean.max()
    pri_pmin = pri[["within_p", "free_p"]].min().min()
    # Three seeds when available: their range, not the first seed alone.
    if os.path.exists(os.path.join(OUT, "cbramod_seeds_summary.csv")):
        sr = rd("cbramod_seeds_summary.csv").set_index("config").loc["revised"]
        sp = rd("cbramod_seeds_pretraining.csv")
        fm_txt = (f"Control {fm_ctrl:.4f} (µV) / {fm_100:.4f} (100 µV)\n"
                  f"Affect {sr.lo:.4f} to {sr.hi:.4f} (3 seeds)\n"
                  f"Pretrained minus scratch {sp['diff'].min():+.4f} "
                  f"to {sp['diff'].max():+.4f}").replace("-0.", "−0.")
    else:
        fm_txt = (f"Control {fm_ctrl:.4f}; affect "
                  f"{cb.loc['revised', 'pooled_bal']:.4f}\nPretrained minus "
                  f"scratch {pe.loc['revised', 'diff']:+.4f}")
    phases = [
        ("Phase 1\nEvaluation Protocol",
         ["Participant-separated splits", "Recording-level permutation nulls",
          "External positive control"],
         f"Control {ctrl:.4f} balanced, 100 participants\n"
         f"DEED affect {c.loc['RF, unweighted (headline)', 'observed_mean10']:.4f} (headline)\n"
         f"Primary, 20 random partitions: best of\n"
         f"four {pri_best:.4f}, every p ≥ {pri_pmin:.2f}"),
        ("Phase 2\nMechanism Analysis",
         ["Pipeline ladder with reported and randomised labels",
          "Protocol factorial", "Identity rule and identity controls"],
         f"Window-level {st.loc[5, 'leaky_acc']:.4f} (reported labels), "
         f"{free5.acc:.4f} (random labels)\n"
         f"Recording route {fp['derived DEED recording route bal (ws - rg)']:.4f} "
         f"of {fp['derived DEED inflation bal (ws - ps)']:.4f}"),
        ("Phase 3\nFoundation-Model Check",
         ["CBraMod, four fine-tuning configurations",
          "Same architecture without pretraining", "Within-participant test"],
         fm_txt),
        ("Phase 4\nProposed Method (PCD)",
         ["Null decoders on labels permuted within participant",
          "Upper-tail permutation p-value p(x)", "Predict when p(x) ≤ α"],
         f"False-positive rate {dn.fpr.mean():.4f} (re-randomised DEED)\n"
         f"{si.fpr.mean():.4f} (100 simulated datasets), α = 0.05"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 9.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(-2.6, 21.5)
    ax.axis("off")
    y = 19.6
    for i, (title, bullets, result) in enumerate(phases):
        ax.add_patch(FancyBboxPatch((0.5, y - 1.32), 9.0, 1.32,
                                    boxstyle="round,pad=0.06", fc=C_HDR,
                                    ec=C_HDR))
        ax.text(5.0, y - 0.66, title, ha="center", va="center", color="white",
                fontsize=10, fontweight="bold", linespacing=1.35)
        h = 0.62 * len(bullets) + 0.42
        ax.add_patch(FancyBboxPatch((0.5, y - 1.32 - h), 9.0, h,
                                    boxstyle="round,pad=0.06", fc=C_BOX,
                                    ec=C_EDGE, lw=1.1))
        for j, b in enumerate(bullets):
            ax.text(1.0, y - 1.82 - 0.62 * j, "•  " + b, ha="left",
                    va="center", fontsize=8.6)
        # the result box grows with its number of lines, so no text spills
        top = y - 1.32 - h - 0.18          # a gap below the bullet box
        hgt = 0.44 * (result.count("\n") + 1) + 0.30
        ax.add_patch(FancyBboxPatch((1.1, top - hgt), 7.8, hgt,
                                    boxstyle="round,pad=0.05", fc=C_OUT,
                                    ec=C_OUTE, lw=1.0))
        ax.text(5.0, top - hgt / 2, result, ha="center", va="center",
                fontsize=8.0, color="#1D4D14", linespacing=1.3)
        y = top - hgt - 0.46
        if i < len(phases) - 1:
            ax.add_patch(FancyArrowPatch((5.0, y + 0.06), (5.0, y - 0.52),
                                         arrowstyle="-|>", mutation_scale=14,
                                         lw=1.5, color=C_HDR))
            y -= 0.78
    # the axis (and the figure, at constant scale) end below the last box,
    # so it is never clipped however many result lines there are
    low = y - 0.1
    ax.set_ylim(low, 21.5)
    fig.set_size_inches(7.2, 9.3 * (21.5 - low) / 24.1)
    ax.text(5.0, 21.0, "Four-phase analysis", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=C_HDR)
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(FIG, f"fig0_framework.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        shutil.copyfile(p, os.path.join(MANUSCRIPT, f"fig0_framework.{ext}"))
    plt.close(fig)

    t = io.open(TEX, encoding="utf-8", newline="").read()
    for old, new in (
        ("Phase~3 repeats the central evaluation on a transformer pretrained on 9{,}000 hours of EEG, so that the finding cannot be attributed to weak handcrafted features. Phase~4 introduces Permutation-Calibrated Decodability, which converts the dataset-level permutation test of Phase~1 into a per-input abstention rule.",
         "Phase~3 repeats the central evaluation on a transformer pretrained on 9{,}000 hours of EEG, with and without its pretraining, to test whether the finding depends on handcrafted features for this model and task. Phase~4 introduces Permutation-Calibrated Decodability, which localises the dataset-level permutation test of Phase~1 to individual inputs as an abstention rule."),
        ("Phase~3 repeats the central evaluation using a pretrained foundation model, so the conclusion does not depend on the choice of handcrafted features.",
         "Phase~3 repeats the central evaluation using a pretrained foundation model, to test whether the conclusion depends on the choice of handcrafted features."),
    ):
        if old in t:
            assert t.count(old) == 1
            t = t.replace(old, new)
    io.open(TEX, "w", encoding="utf-8", newline="").write(t)
    print("figure 1 (framework) redrawn; caption and introduction scoped")


if __name__ == "__main__":
    main()
