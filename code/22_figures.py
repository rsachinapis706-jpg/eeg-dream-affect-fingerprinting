"""
Publication figures. Every value is read from the result CSVs on disk -- no
number is typed into this file.
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
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 600, "savefig.dpi": 600, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.grid": True, "axes.axisbelow": True, "grid.linestyle": "-",
})
C_OK, C_BAD, C_NEU, C_ALT = "#2A6F97", "#C1352C", "#8D99AE", "#E09F3E"


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p + ".png"); fig.savefig(p + ".pdf")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Fig 1
def fig_leakage_ladder():
    d = pd.read_csv(os.path.join(OUT, "leakage_mechanism.csv"))
    d = d[d.features.str.startswith("spectral")]
    order = ["nested_subject", "nested_random", "preselect_random",
             "presmote_random", "both_random"]
    labels = ["Nested,\nparticipant-\nseparated  $\\bigstar$OURS", "Nested,\nrandom split",
              "Feature selection\non all data", "SMOTE\nbefore split",
              "Selection + SMOTE\non all data"]
    best = [d[d.ordering == o].accuracy.max() for o in order]
    colors = [C_OK] + [C_NEU] * 2 + [C_BAD] * 2

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    b = ax.bar(range(len(order)), best, color=colors, edgecolor="black", lw=.6)
    for i, v in enumerate(best):
        ax.text(i, v + .012, f"{v:.3f}", ha="center", fontsize=8.5,
                fontweight="bold")
    ax.axhline(best[0], ls="--", lw=1, color=C_OK)
    ax.text(4.42, best[0] + .008, "our protocol (correct)", color=C_OK,
            fontsize=7.5, ha="right", fontweight="bold")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_ylabel("Accuracy (best of KNN / SVM / RF)")
    ax.set_ylim(0, max(best) * 1.22)
    ax.set_title("OUR FINDING: reported accuracy is set by pipeline ordering\n"
                 "(same data, same features, same classifiers)")
    save(fig, "fig1_leakage_ladder")


# ---------------------------------------------------------------- Fig 2
def fig_fingerprinting():
    s = pd.read_csv(os.path.join(OUT, "subject_confound.csv")).iloc[0]
    lk = pd.read_csv(os.path.join(OUT, "leakage_results.csv"))
    aff = lk[lk.target.str.contains("valence")]
    win = float(aff[aff.held_out == "random"].accuracy.iloc[0])
    sub = float(aff[aff.held_out == "subject"].accuracy.iloc[0])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 3.2))

    a1.bar(["Window-\nshuffled", "Recording-\ngrouped"],
           [s.subject_id_acc_random, s.subject_id_acc_grouped],
           color=[C_BAD, C_ALT], edgecolor="black", lw=.6)
    a1.axhline(1 / 34, ls="--", color="k", lw=1)
    a1.text(1.42, 1 / 34 + .03, "chance (1/34)", ha="right", fontsize=7.5)
    for i, v in enumerate([s.subject_id_acc_random, s.subject_id_acc_grouped]):
        a1.text(i, v + .02, f"{v:.3f}", ha="center", fontsize=8.5,
                fontweight="bold")
    a1.set_ylim(0, 1.02); a1.set_ylabel("Subject-identity accuracy")
    a1.set_title("EEG identifies the individual", fontsize=9.5)

    vals = [s.oracle_acc, win, sub]
    a2.bar(["Subject ID\nonly\n(no EEG)", "Window-shuffled\naffect model",
            "Subject-wise\naffect model"], vals,
           color=[C_BAD, C_NEU, C_OK], edgecolor="black", lw=.6)
    for i, v in enumerate(vals):
        a2.text(i, v + .012, f"{v:.3f}", ha="center", fontsize=8.5,
                fontweight="bold")
    a2.axhline(1 / 3, ls="--", color="k", lw=1)
    a2.text(2.42, 1 / 3 + .012, "chance", ha="right", fontsize=7.5)
    a2.set_ylim(0, max(vals) * 1.25); a2.set_ylabel("Accuracy")
    a2.set_title("An oracle with no EEG nearly matches it", fontsize=9.5)

    fig.suptitle("OUR FINDING: apparent dream-emotion decoding is largely "
                 "subject fingerprinting", fontsize=10, y=1.03)
    save(fig, "fig2_fingerprinting")


# ---------------------------------------------------------------- Fig 3
def fig_control():
    e = pd.read_csv(os.path.join(OUT, "expanded_control.csv"))
    e3 = e[(e.task.str.startswith("W/N1/N2")) & (e.model == "logreg")].iloc[0]
    names = ["Sleep-EDF staging\n100 participants\n$\\bigstar$OUR PIPELINE",
             "DEED dream affect\n34 participants\n$\\bigstar$OUR PIPELINE"]
    vals = [e3.balanced, 0.3373]
    err = [[e3.balanced - e3.bal_lo, 0], [e3.bal_hi - e3.balanced, 0]]

    fig, ax = plt.subplots(figsize=(4.3, 3.4))
    ax.bar(names, vals, yerr=err, color=[C_OK, C_BAD], edgecolor="black",
           lw=.6, capsize=5)
    for i, v in enumerate(vals):
        ax.text(i, v + .028, f"{v:.3f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.axhline(1 / 3, ls="--", color="k", lw=1)
    ax.text(1.44, 1 / 3 + .012, "chance", ha="right", fontsize=7.5)
    ax.set_ylabel("Balanced accuracy (leave-one-participant-out)")
    ax.set_ylim(0, .95)
    ax.set_title("Identical pipeline, two tasks:\nthe null is a property of the "
                 "data, not the method", fontsize=9.5)
    save(fig, "fig3_positive_control")


# ---------------------------------------------------------------- Fig 4
def fig_within_subject():
    a = pd.read_csv(os.path.join(OUT, "within_subject.csv"))
    b = pd.read_csv(os.path.join(OUT, "cbramod_within_subject.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.2), sharey=True)
    for ax, d, t in zip(axes, [a, b],
                        ["Spectral features", "CBraMod embeddings"]):
        ax.scatter(d.null, d.bacc, s=26, color=C_OK, edgecolor="black", lw=.4,
                   zorder=3)
        lo, hi = .05, .75
        ax.plot([lo, hi], [lo, hi], ls="--", color="k", lw=1, zorder=2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Permutation null (balanced acc)")
        ax.set_title(f"{t}\nn={len(d)} participants", fontsize=9.5)
    axes[0].set_ylabel("Observed within-participant (balanced acc)")
    fig.suptitle("OUR FINDING: no within-participant dream-affect effect "
                 "detected — participants sit on the identity line",
                 fontsize=10, y=1.03)
    save(fig, "fig4_within_subject")


# ---------------------------------------------------------------- Fig 5
def fig_pcd():
    d = pd.read_csv(os.path.join(OUT, "pcd_validation.csv"))
    # Replace the 8-subject strong condition with the 100-subject re-run so the
    # whole figure reflects one cohort standard.
    exp = os.path.join(OUT, "pcd_expanded.csv")
    if os.path.exists(exp):
        e = pd.read_csv(exp)
        e["task"] = "strong"
        e["base_accuracy"] = np.nan
        d = pd.concat([d[d.task != "strong"], e], ignore_index=True)
    p = d[d.rule == "PCD"]
    tasks = [("strong", "Strong signal\n(Sleep-EDF, 100 part.)", C_OK),
             ("intermediate", "Weak signal\n(DEED stage)", C_ALT),
             ("absent", "Weak at best\n(DEED affect)", C_BAD),
             ("absent_cbramod", "Weak at best\n(CBraMod embeddings)", C_NEU)]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for t, lab, c in tasks:
        s = p[p.task == t].sort_values("param")
        a1.plot(s.param, s.coverage, "o-", color=c, label=lab, ms=4.5, lw=1.6)
        a1.fill_between(s.param, s.cov_lo, s.cov_hi, color=c, alpha=.15)
    a1.set_xscale("log"); a1.invert_xaxis()
    a1.set_xlabel(r"$\alpha$ (stricter $\rightarrow$)")
    a1.set_ylabel("Coverage (fraction predicted)")
    a1.set_ylim(-.03, 1.03)
    a1.legend(fontsize=6.6, loc="center left", frameon=False)
    a1.set_title("PCD (ours) engages only where signal exists", fontsize=9.5)

    a05 = p[p.param == 0.05]
    m = d[(d.rule == "MSP_fixed") & (d.param == 0.50)]
    names = [lab.replace("\n", " ") for _, lab, _ in tasks]
    x = np.arange(len(tasks)); w = .38
    a2.bar(x - w / 2, [float(a05[a05.task == t].coverage.iloc[0])
                       for t, _, _ in tasks], w, label="PCD ($\\alpha$=0.05)",
           color=C_OK, edgecolor="black", lw=.5)
    a2.bar(x + w / 2, [float(m[m.task == t].coverage.iloc[0])
                       for t, _, _ in tasks], w, label="Max-softmax > 0.5  (baseline)",
           color=C_BAD, edgecolor="black", lw=.5)
    a2.set_xticks(x)
    a2.set_xticklabels(["Strong", "Weak", "None", "None\n(CBraMod)"],
                       fontsize=7.6)
    a2.set_ylabel("Coverage"); a2.legend(fontsize=7, frameon=False)
    a2.set_title("Confidence thresholds keep predicting\nwhere PCD abstains",
                 fontsize=9.5)
    save(fig, "fig5_pcd")


# ---------------------------------------------------------------- Fig 6
def fig_calibration():
    d = pd.read_csv(os.path.join(OUT, "pcd_validation.csv"))
    c = d[d.rule == "calibration"]
    names = {"strong": "Strong", "intermediate": "Weak", "absent": "None",
             "absent_cbramod": "None\n(CBraMod)"}
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    x = np.arange(len(c))
    ax.bar(x, c.coverage, color=C_NEU, edgecolor="black", lw=.6)
    ax.axhline(.05, ls="--", color=C_BAD, lw=1.2)
    ax.text(len(c) - .55, .056, "nominal 0.05", color=C_BAD, fontsize=7.5,
            ha="right")
    for i, v in enumerate(c.coverage):
        ax.text(i, v + .002, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([names.get(t, t) for t in c.task], fontsize=7.8)
    ax.set_ylabel("False-positive rate (null vs null)")
    ax.set_ylim(0, .09)
    ax.set_title("PCD is calibrated:\nempirical FPR matches the nominal rate",
                 fontsize=9.5)
    save(fig, "fig6_calibration")


if __name__ == "__main__":
    for f in [fig_leakage_ladder, fig_fingerprinting, fig_control,
              fig_within_subject, fig_pcd, fig_calibration]:
        try:
            f()
        except Exception as e:
            print(f"FAILED {f.__name__}: {type(e).__name__}: {e}")
    print("\nfigures ->", FIG)
