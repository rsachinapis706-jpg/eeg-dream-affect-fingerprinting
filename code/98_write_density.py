"""
Writes the channel-density results (Section IV-I), a density table, the
replication-figure caption and the discussion statements that depend on them,
from the result files of 75_density_sensitivity.py (Reviewer 1, comments 4
and 8). Also corrects the Tononi window accounting (30 vs 33: the density
sweep requires all 64 electrodes).
"""
import io
import shutil

import pandas as pd

OUT = r"E:\Dream GAN\analysis\out"
TEX = r"E:\Dream_Analysis\accesstemp.tex"
MODELS = [("lr_fixed", "LR, fixed $C$"), ("lr_nested", "LR, nested $C$"),
          ("pca_lr", "PCA--LR, nested"), ("rf", "RF")]
READABLE = {"lr_fixed": "logistic regression with fixed regularisation",
            "lr_nested": "logistic regression with nested regularisation",
            "pca_lr": "principal-component logistic regression",
            "rf": "the random forest"}


def main():
    b = pd.read_csv(OUT + r"\density_both_partitions.csv")
    s = pd.read_csv(OUT + r"\density_sensitivity.csv")
    p = pd.read_csv(OUT + r"\density_permutation.csv")
    ch = [6, 12, 24, 48, 64]
    feats = {6: "108", 12: "216", 24: "432", 48: "864", 64: "1{,}152"}

    def ps(model, n):
        r = s[(s.model == model) & (s.channels == str(n))].iloc[0]
        return r

    def ws(model, n):
        return b[(b.model == model) & (b.channels == n)
                 & (b.partition == "window-shuffled")].iloc[0]

    rows = []
    for n in ch:
        cells = [f"{ws(m, n).bal_mean:.4f} $\\pm$ {ws(m, n).bal_sd:.4f}"
                 for m in ("lr_fixed", "rf")]
        cells += [f"{ps(m, n).bal_mean:.4f} $\\pm$ {ps(m, n).bal_sd:.4f}"
                  for m, _ in MODELS]
        rows.append(f"{n} ({feats[n]}) & " + " & ".join(cells) + r" \\")
    diff = ["\\multicolumn{3}{@{}l}{64 minus 6 [95\\% CI]}"] + [
        f"${ps(m, '64 minus 6').bal_mean:+.4f}$ [{ps(m, '64 minus 6').ci_lo:+.4f}, "
        f"{ps(m, '64 minus 6').ci_hi:+.4f}]" for m, _ in MODELS]
    slope = ["\\multicolumn{3}{@{}l}{Slope per doubling [95\\% CI]}"] + [
        f"${ps(m, 'slope per doubling').bal_mean:+.4f}$ "
        f"[{ps(m, 'slope per doubling').ci_lo:+.4f}, "
        f"{ps(m, 'slope per doubling').ci_hi:+.4f}]" for m, _ in MODELS]
    pmin = p.groupby("model").p.min()
    table = "\n".join([
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Montage density on the Tononi corpus (dream recall; 598 windows, 106 recordings, 18 participants, identical windows at every density). Balanced accuracy, mean $\pm$ SD over five fold assignments; nothing is maximised over models. Participant-bootstrap intervals average over the five assignments within each resample.}",
        r"\label{tab:density}",
        r"\scriptsize",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Window-shuffled} & \multicolumn{4}{c}{Participant-separated} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}",
        r"Electrodes (features) & LR, fixed $C$ & RF & " + " & ".join(n for _, n in MODELS) + r" \\",
        r"\midrule",
    ] + rows + [r"\midrule", " & ".join(diff) + r" \\",
                " & ".join(slope) + r" \\", r"\botrule", r"\end{tabular}",
                r"\normalsize", r"\end{table*}"])

    lo = {m: ps(m, "64 minus 6") for m, _ in MODELS}
    upper = max(r.ci_hi for r in lo.values())
    txt830 = (
        "Table~\\ref{tab:density} and Figure~\\ref{fig:replication}(b) give the result, in balanced accuracy on both sides and separately for each model. Under participant separation no model improves with density. From 6 to 64 electrodes the change is "
        + ", ".join(f"${lo[m].bal_mean:+.4f}$ ({lo[m].ci_lo:+.4f} to {lo[m].ci_hi:+.4f}) for {READABLE[m]}"
                    for m, _ in MODELS)
        + (f", and neither nested model exceeds its permutation null at any electrode count (smallest $p$ {pmin.min():.4f}, 100 permutations, first fold assignment). "
           if pmin.min() >= 0.05 else
           f"; a nested model exceeds its permutation null at some electrode count (smallest $p$ {pmin.min():.4f}). ")
        + f"Under window-shuffled partitioning balanced accuracy rises from {ws('lr_fixed', 6).bal_mean:.4f} to {ws('lr_fixed', 64).bal_mean:.4f} for logistic regression and from {ws('rf', 6).bal_mean:.4f} to {ws('rf', 64).bal_mean:.4f} for the random forest.")
    txt832 = (
        "The window-level advantage therefore widens with density: more electrodes give a classifier more by which to recognise a recording or a participant, while adding nothing measurable about the label under participant separation, so a study reporting window-level cross-validation on a high-density montage is more exposed to the problem, not less. Two limits apply. At 64 electrodes there are 1{,}152 features for 598 windows from 18 participants; nested selection of the regularisation and of the number of principal components leaves the pattern unchanged, but the intervals are wide, and a gain of up to "
        f"{upper:.2f} in balanced accuracy from 6 to 64 electrodes cannot be excluded. And the label is dream recall: the sweep shows that density does not help this task with this feature pipeline, not whether spatial resolution limits affect decoding in DEED.")

    t = io.open(TEX, encoding="utf-8", newline="").read()
    shutil.copyfile(TEX, TEX + ".pre_density")

    def line_swap(prefix, new):
        nonlocal t
        assert t.count(prefix) == 1, (t.count(prefix), prefix[:60])
        i = t.index(prefix)
        j = t.index("\n", i)
        t = t[:i] + new + t[j:]

    def swap(old, new):
        nonlocal t
        assert t.count(old) == 1, (t.count(old), old[:60])
        t = t.replace(old, new)

    swap("Window inclusion is decided on the first six electrodes in every condition, so all densities are evaluated on an identical set of 598 windows and a denser montage cannot change which data enter the comparison.",
         "Window inclusion is decided on the first six electrodes in every condition, so all densities are evaluated on an identical set of windows and a denser montage cannot change which data enter the comparison. Recordings lacking any of the 64 electrodes are excluded, which leaves 598 windows from 106 recordings, against 668 windows from 113 recordings in panel (a), where only six electrodes are required. Besides logistic regression with fixed regularisation and the random forest, logistic regression with its regularisation chosen by inner participant-grouped cross-validation, and a principal-component logistic regression with the number of components and the regularisation chosen the same way, are evaluated over five fold assignments.")
    line_swap("Panel~(b) of Table~\\ref{tab:replication} and Figure~\\ref{fig:replication}(b) give the result", txt830 + "\n\n" + table)
    line_swap("This inverts a natural expectation.", txt832)

    # replication table: panel (b) moves to the density table
    i = t.index("\\multicolumn{3}{@{}l}{\\textit{(b) Montage density, Tononi, 598 windows, 18 participants}} \\\\")
    i = t.rindex("\\midrule", 0, i)
    j = t.index("\\botrule", i)
    t = t[:i] + t[j:]
    swap("Panel (a) repeats the DEED protocol exactly, on six electrodes; the label is dream recall, so this tests the mechanism and not the affect null. Panel (b) varies montage density on nested electrode subsets.}",
         "The DEED protocol is repeated exactly, on six electrodes; the label is dream recall, so this tests the mechanism and not the affect null. Random-forest and identity-rule values are means over ten fold assignments.}")
    swap("\\multicolumn{3}{@{}l}{\\textit{(a) Mechanism, DEED protocol applied unchanged}} \\\\\n", "")

    # figure caption
    i = t.index("\\caption{\\textbf{Two confirmatory experiments on the Tononi corpus.}")
    j = t.index("\\label{fig:replication}", i)
    t = t[:i] + ("\\caption{\\textbf{Replication and channel density on the Tononi corpus.} (a) Balanced accuracy under three partitions, DEED and Tononi, random forest, mean $\\pm$ SD over ten fold assignments; and the identity rule, which receives only the participant identifier and is scored on the same held-out windows, with its majority taken from all of the participant's training windows or from their other recordings only. Grouping recordings removes the window-level advantage on Tononi. Without the held-out recording's own windows the rule falls to chance on DEED (0.3314) and from 0.5592 to 0.4354 on Tononi. (b) Balanced accuracy under window-shuffled (solid) and participant-separated (dashed) partitioning against the number of electrodes, per model, mean $\\pm$ SD over five fold assignments; values in Table~\\ref{tab:density}. The Tononi label is dream recall, not affect.}\n") + t[j:]

    # discussion
    swap("First, the positive control demonstrates that the analysis pipeline is capable of recovering a real effect from the same recording modality and window length.",
         "First, the positive control shows that the pipeline recovers sleep-stage information from the same recording modality and window length; it does not show sensitivity to weak or differently represented affect information, which the simulation of Section~\\ref{sec:howmuch} bounds only for one signal structure.")
    swap("Fourth, the one instrumental explanation that could be tested was tested and did not hold: raising montage density tenfold, from 6 to 64 electrodes, left participant-separated balanced accuracy flat within four points of chance. Spatial resolution is therefore not what limits the result, which narrows the space of remaining explanations considerably.",
         "Fourth, on a different task, dream recall in the Tononi corpus, raising montage density from 6 to 64 electrodes did not raise participant-separated balanced accuracy for any model tried, including models with nested regularisation; the intervals are wide, and a different decoding target cannot settle whether spatial resolution limits affect decoding in DEED.")
    swap("Eighteen participants and 668 analysable windows survived the requirement that all 64 selected electrodes be present and the amplitude-rejection criterion, against 34 participants and 7{,}418 windows in DEED.",
         "Eighteen participants and 668 analysable windows from 113 recordings survived the amplitude-rejection criterion on the six electrodes of the replication, and requiring all 64 electrodes of the density sweep leaves 598 windows from 106 recordings, against 34 participants and 6{,}144 valence windows in DEED.")
    t = t.replace("Table~\\ref{tab:replication}b, Fig.~\\ref{fig:replication}b", "Table~\\ref{tab:density}, Fig.~\\ref{fig:replication}b")
    io.open(TEX, "w", encoding="utf-8", newline="").write(t)
    print("density section, table, caption and discussion written")


if __name__ == "__main__":
    main()
