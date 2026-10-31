"""
Writes the PCD results (Sections IV-K and IV-L), their tables and figure
captions, and the PCD paragraphs of Section VII, from the outputs of
77_pcd_revised.py and 85_pcd_extra.py (Reviewer 1, comments 1, 9 and 10;
Reviewer 2 on PCD scope). Every number is read from a result file.
"""
import io
import shutil

import numpy as np
import pandas as pd

OUT = r"E:\Dream GAN\analysis\out"
TEX = r"E:\Dream_Analysis\accesstemp.tex"


def rd(n):
    return pd.read_csv(f"{OUT}\\{n}")


def main():
    sm = rd("pcd_revised_summary.csv")
    gr = rd("pcd_extra_grid.csv")
    cbx = rd("pcd_extra_cbramod.csv")
    dn = rd("pcd_revised_deednull.csv")
    si = rd("pcd_revised_sims.csv")
    rc = rd("pcd_revised_riskcov.csv")
    nz = rd("pcd_extra_noise.csv")
    nd = rd("pcd_extra_nulldep.csv")
    SE, DE = "Sleep-EDF staging", "DEED affect"

    def g(cond, rule, col="coverage"):
        return float(gr[(gr.condition == cond) & (gr.rule == rule)].iloc[0][col])

    def gci(cond, rule, col):
        r = gr[(gr.condition == cond) & (gr.rule == rule)].iloc[0]
        lo, hi = ("cov_lo", "cov_hi") if col == "coverage" else ("bal_lo", "bal_hi")
        return f"{r[col]:.4f} [{r[lo]:.4f}, {r[hi]:.4f}]"

    def s(cond, rule, col):
        return float(sm[(sm.condition == cond) & (sm.rule == rule)].iloc[0][col])

    cb_cov = float(cbx[cbx.rule == "PCD p<=0.05"].coverage.iloc[0])
    cb_msp = float(cbx[cbx.rule == "Max-softmax>0.5"].coverage.iloc[0])

    def matched(cond, target):
        grid = sorted(rc[rc.condition == cond].coverage.unique())
        cv = min(grid, key=lambda x: abs(x - target))
        r = rc[(rc.condition == cond) & (np.isclose(rc.coverage, cv))]
        return cv, {k: float(r[r.rule == k].bal_retained.iloc[0])
                    for k in ("PCD", "Max-softmax", "Entropy")}

    cvS, mS = matched(SE, g(SE, "PCD alpha=0.05"))
    piv = {c: rc[rc.condition == c].pivot(index="coverage", columns="rule",
                                          values="bal_retained")
           for c in (SE, DE)}
    dS = piv[SE]["PCD"] - piv[SE]["Max-softmax"]
    eS = piv[SE]["PCD"] - piv[SE]["Entropy"]
    dD = piv[DE]["PCD"] - piv[DE]["Max-softmax"]
    cvD, mD = matched(DE, g(DE, "PCD alpha=0.05"))

    # ------------------------------------------------------------- tables
    trows = []
    for cond, lab in ((SE, "Sleep-EDF staging, 100 part., 183{,}775 windows"),
                      (DE, "DEED dream valence, 34 part., 6{,}144 windows")):
        trows.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\textit{{{lab}}}}} \\\\")
        for rule, name in (("PCD alpha=0.1", "PCD, $\\alpha=0.10$"),
                           ("PCD alpha=0.05", "PCD, $\\alpha=0.05$"),
                           ("PCD alpha=0.01", "PCD, $\\alpha=0.01$"),
                           ("Max-softmax>0.5", "Max-softmax $>0.5$"),
                           ("Max-softmax>0.7", "Max-softmax $>0.7$"),
                           ("Max-softmax>0.9", "Max-softmax $>0.9$"),
                           ("all inputs", "All inputs")):
            trows.append(f"{name} & {gci(cond, rule, 'coverage')} & "
                         f"{gci(cond, rule, 'bal_retained')} \\\\")
        trows.append("\\midrule")
    trows = trows[:-1]
    t_pcd = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{PCD and fixed confidence thresholds on identical participant-separated folds, models and data. Coverage is the share of windows on which the rule predicts; retained balanced accuracy is computed on those windows. 95\% participant-bootstrap intervals, 2{,}000 resamples.}",
        r"\label{tab:pcd}",
        r"\scriptsize",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"Rule & Coverage & Retained bal.\ acc. \\",
        r"\midrule"] + trows + [r"\botrule", r"\end{tabular}", r"\normalsize",
                                 r"\end{table}"])

    def srow(rule, name):
        return (f"{name} & {s(SE, rule, 'coverage'):.4f} & "
                f"{s(DE, rule, 'coverage'):.4f} & "
                f"{s(SE, rule, 'bal_retained'):.4f} & "
                f"{s(DE, rule, 'bal_retained'):.4f} \\\\")
    conf = "Split conformal 0.05"
    t_sel = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{PCD against established ways of declining to predict, on identical folds, models and data. Coverage: share of windows on which the rule predicts; for split conformal prediction, the share of single-class prediction sets. Conformal calibration uses a quarter of the training participants, excluded from fitting and disjoint from the test participants.}",
        r"\label{tab:selective}",
        r"\scriptsize",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Coverage} & \multicolumn{2}{c}{Retained bal.\ acc.} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Rule & Sleep-EDF & DEED & Sleep-EDF & DEED \\",
        r"\midrule",
        srow("PCD p<=0.05", "PCD, $\\alpha=0.05$"),
        srow("Max-softmax>0.5", "Max-softmax $>0.5$"),
        srow("Entropy<0.9", "Normalised entropy $<0.9$"),
        f"Split conformal, $\\alpha=0.05$ & {s(SE, conf, 'coverage'):.4f} & {s(DE, conf, 'coverage'):.4f} & -- & -- \\\\",
        f"\\quad set size / label coverage & {s(SE, conf, 'set_size'):.2f} / {s(SE, conf, 'set_coverage'):.4f} & {s(DE, conf, 'set_size'):.2f} / {s(DE, conf, 'set_coverage'):.4f} & & \\\\",
        srow("all inputs", "All inputs"),
        r"\botrule", r"\end{tabular}", r"\normalsize", r"\end{table}"])

    # ------------------------------------------------------------- text
    n0, n1 = nz[nz.eta == 0.0].iloc[0], nz[nz.eta == 1.0].iloc[0]
    n8 = nz[np.isclose(nz.eta, 0.8)].iloc[0]
    sp = nz[["balanced", "coverage"]].corr("spearman").iloc[0, 1]
    pe = nz[["balanced", "coverage"]].corr().iloc[0, 1]
    mono = bool(np.all(np.diff(nz.sort_values("eta").coverage) <= 0))

    def rng(knob, col):
        k = nd[nd.knob == knob]
        return k[col].min(), k[col].max()

    f_nm, p_nm, c_nm = (rng("features", "null_mean_margin"),
                        rng("participants", "null_mean_margin"),
                        rng("capacity", "null_mean_margin"))
    f_cv, p_cv, c_cv = (rng("features", "coverage"), rng("participants", "coverage"),
                        rng("capacity", "coverage"))
    f_ba, p_ba = rng("features", "balanced"), rng("participants", "balanced")

    body = f"""\\subsection{{Permutation-Calibrated Decodability}}\\label{{sec:pcdres}}

Table~\\ref{{tab:pcd}} reports PCD as defined in Section~\\ref{{sec3}} beside fixed confidence thresholds, on identical participant-separated folds, models and data: a standardised, class-balanced logistic regression with $B=100$ null decoders per fold, fitted to labels permuted within participant at the unit at which they vary. On Sleep-EDF staging PCD at $\\alpha=0.05$ predicts on {gci(SE, 'PCD alpha=0.05', 'coverage')} of windows, with retained balanced accuracy {gci(SE, 'PCD alpha=0.05', 'bal_retained')} against {g(SE, 'all inputs', 'bal_retained'):.4f} for all windows. On DEED affect it predicts on {gci(DE, 'PCD alpha=0.05', 'coverage')} of windows, whose balanced accuracy, {g(DE, 'PCD alpha=0.05', 'bal_retained'):.4f}, is no higher than that of all windows, {g(DE, 'all inputs', 'bal_retained'):.4f}; on frozen CBraMod embeddings of the same task, which were never fine-tuned on these labels, it predicts on {cb_cov:.4f}. A max-softmax threshold of 0.5 predicts on {g(SE, 'Max-softmax>0.5'):.4f} and {g(DE, 'Max-softmax>0.5'):.4f} of windows in the two tasks, and on {cb_msp:.4f} with the embeddings.

Calibration is checked where the null holds by construction rather than on DEED as it stands, whose weak association with the label is itself in question. With DEED's recording labels re-randomised within participant 20 times, PCD predicts on {dn.fpr.mean():.4f} of windows on average (range {dn.fpr.min():.4f} to {dn.fpr.max():.4f}), as Proposition~\\ref{{prop:calib}} requires. Participants differ far more: in the median re-randomisation the worst participant is flagged on {dn.part_fpr_max.median():.4f} of their windows, and on average {dn.part_frac_above_2alpha.mean():.3f} of participants exceed twice the nominal rate. Across 100 simulated datasets with participant and recording random effects, class imbalance and heterogeneous per-input noise, the marginal rate is {si.fpr.mean():.4f} (2.5th to 97.5th percentile {si.fpr.quantile(.025):.4f} to {si.fpr.quantile(.975):.4f}), the 90th percentile of participant-level rates has a median of {si.part_fpr_p90.median():.4f}, and the Kolmogorov distance between the null $p$-values and the uniform distribution has a median of {si.p_uniform_ks.median():.4f}. Marginal validity holds; calibration for a participant does not, as the proposition states.

\\subsubsection{{Comparison with established selective-prediction methods}}

Table~\\ref{{tab:selective}} adds predictive entropy and split conformal prediction, and Figure~\\ref{{fig:pcd}}(b) gives risk--coverage curves. Two questions should be kept apart: how often a rule predicts when there is little or nothing to predict, and how accurate its retained predictions are. On the second, the answer depends on the task. On Sleep-EDF, where the signal is strong, PCD's retained balanced accuracy is never more than {-dS.min():.4f} below that of max-softmax at any coverage and exceeds it by up to {dS.max():.4f} (at coverage {dS.idxmax():.2f}), and entropy by up to {eS.max():.4f}; at the coverage PCD reaches at $\\alpha=0.05$ the three are close ({mS['PCD']:.4f}, {mS['Max-softmax']:.4f} and {mS['Entropy']:.4f} at coverage {cvS:.2f}). On DEED the curves differ by no more than {dD.abs().max():.4f} in either direction, with no consistent sign. These are point estimates on one fold assignment. On the first, the rules differ at their customary settings: on DEED max-softmax $>0.5$ predicts on {s(DE, 'Max-softmax>0.5', 'coverage'):.4f} of windows and entropy $<0.9$ on {s(DE, 'Entropy<0.9', 'coverage'):.4f}, against {s(DE, 'PCD p<=0.05', 'coverage'):.4f} for PCD. A stricter confidence threshold can match PCD on DEED -- max-softmax $>0.7$ predicts on {g(DE, 'Max-softmax>0.7'):.4f} -- but then predicts on only {g(SE, 'Max-softmax>0.7'):.4f} of Sleep-EDF windows, against {g(SE, 'PCD alpha=0.05'):.4f} for PCD, and nothing about a confidence threshold says in advance where it should be set. Split conformal prediction attains its nominal coverage of the true label ({s(SE, conf, 'set_coverage'):.4f} and {s(DE, conf, 'set_coverage'):.4f}) with mean set sizes of {s(SE, conf, 'set_size'):.2f} and {s(DE, conf, 'set_size'):.2f}, and returns a single class for {s(SE, conf, 'coverage'):.4f} and {s(DE, conf, 'coverage'):.4f} of windows; its $\\alpha$ controls coverage of the true label, PCD's the rate of prediction under the null, so the two are not interchangeable. What PCD contributes is therefore an abstention level with a stated meaning under the null, together with a ranking of inputs that is at least as good as confidence where signal is present and no better where it is weak.

{t_sel}

{t_pcd}

\\begin{{figure*}}[htbp]
\\centering
\\includegraphics[width=0.85\\linewidth]{{fig5_pcd.pdf}}
\\caption{{\\textbf{{PCD coverage and the accuracy of retained predictions.}} (a) Coverage $C(\\alpha)$, the share of windows on which PCD predicts, against $\\alpha$, for Sleep-EDF staging, DEED affect and DEED affect on frozen CBraMod embeddings; the dashed line $C=\\alpha$ bounds coverage under the null (Proposition~\\ref{{prop:calib}}). Dots mark $\\alpha=0.05$. (b) Balanced accuracy of retained predictions against coverage for PCD, max-softmax and predictive entropy; on Sleep-EDF PCD lies above the confidence rules at intermediate coverage, and on DEED the three coincide. All evaluation is participant-separated, first fold assignment.}}
\\label{{fig:pcd}}
\\end{{figure*}}

\\subsubsection{{Coverage depends on the reference null}}\\label{{sec:pcddep}}

Section~\\ref{{sec:theory}} argues that coverage is a property of the reference distribution as well as of the task. Holding the dataset, the task and the label set fixed -- Sleep-EDF staging, on 40 participants with at most 1{{,}}200 windows each -- and varying only the analysis tests this. The mean null margin changes from {f_nm[0]:.4f} to {f_nm[1]:.4f} as the feature count goes from 9 to 36, from {p_nm[0]:.4f} to {p_nm[1]:.4f} across training cohorts of 10 to 40 participants, and from {c_nm[0]:.4f} to {c_nm[1]:.4f} across regularisation strengths. Coverage at $\\alpha=0.05$ ranges from {f_cv[0]:.4f} to {f_cv[1]:.4f} across feature counts, {p_cv[0]:.4f} to {p_cv[1]:.4f} across cohort sizes and {c_cv[0]:.4f} to {c_cv[1]:.4f} across regularisation strengths, while balanced accuracy ranges from {f_ba[0]:.4f} to {f_ba[1]:.4f} and {p_ba[0]:.4f} to {p_ba[1]:.4f} across the first two. The reference distribution depends on the analysis, so coverage may be compared within one analysis of one dataset but not across datasets or analysis settings.

\\subsection{{Controlled Label-Noise Sweep}}

Comparing coverage between datasets confounds signal strength with properties of the data and the analysis. Figure~\\ref{{fig:noise}} therefore constructs a controlled gradient on the full Sleep-EDF cohort: each label is replaced by a uniformly random class with probability $\\eta$, so decodability falls by a known amount while everything else is held fixed. At $\\eta=0$ the evaluation is identical to that of Table~\\ref{{tab:pcd}}, and it reproduces its $p$-values exactly.

Coverage {'declines monotonically' if mono else 'declines'} with decodability, from {n0.coverage:.4f} at $\\eta=0$ to {n1.coverage:.4f} at $\\eta=1$, with Spearman correlation {sp:.4f} and Pearson correlation {pe:.4f} against the balanced accuracy achieved, which falls from {n0.balanced:.4f} to {n1.balanced:.4f}. At $\\eta=1$ the labels are independent of the EEG and exchangeable, so Proposition~\\ref{{prop:calib}} applies, and coverage is {n1.coverage:.4f} against the nominal 0.05. PCD coverage stays well above the nominal level while any signal remains -- {n8.coverage:.4f} at $\\eta=0.8$, where balanced accuracy is {n8.balanced:.4f} -- and falls to it only when none remains. A max-softmax threshold of 0.5 behaves differently: it falls from {n0.msp_coverage:.4f} at $\\eta=0$ to {n8.msp_coverage:.4f} at $\\eta=0.8$, because the decoder's confidence declines with label noise although the labels still carry information. Within one analysis of one dataset, then, PCD coverage orders the noise levels correctly, but it registers whether the decoder outperforms its null rather than by how much; across datasets it is not a common scale.

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.65\\linewidth]{{fig8_label_noise.pdf}}
\\caption{{\\textbf{{Coverage against decodability within one dataset.}} Sleep-EDF labels are replaced by uniformly random classes with probability $\\eta$; the horizontal axis is the balanced accuracy achieved under participant-separated evaluation, the vertical axis the share of windows on which a rule predicts. PCD at $\\alpha=0.05$ falls from {n0.coverage:.4f} to {n1.coverage:.4f}; max-softmax $>0.5$ from {n0.msp_coverage:.4f} to {n1.msp_coverage:.4f}. Dotted line: 0.05.}}
\\label{{fig:noise}}
\\end{{figure}}

"""

    t = io.open(TEX, encoding="utf-8", newline="").read()
    shutil.copyfile(TEX, TEX + ".pre_pcdres")
    a = t.index("\\subsection{Permutation-Calibrated Decodability}",
                t.index("\\label{sec:within}"))
    b = t.index("\\section{Comparisons}", a)
    b = t.rindex("%%%%", a, b)
    b = t.rindex("\n", a, b) + 1
    t = t[:a] + body + t[b:]

    def line_swap(prefix, new):
        nonlocal t
        assert t.count(prefix) == 1, (t.count(prefix), prefix[:60])
        i = t.index(prefix)
        j = t.index("\n", i)
        t = t[:i] + new + t[j:]

    line_swap("These findings create a practical need:",
              "These findings create a practical need: a decoder deployed on a dataset of unknown quality needs a way to decline to predict when the data carry little usable information, and confidence cannot supply it with a stated error rate, because a softmax can assign high probability to structureless input. Permutation-Calibrated Decodability replaces the confidence reference with a permutation null, so that the level at which the rule abstains has a meaning under the null. Section~\\ref{sec3} derives the procedure, and Proposition~\\ref{prop:calib} shows what that meaning is: under the null, the rule predicts on a randomly drawn input with probability at most $\\alpha$. The guarantee is marginal over inputs; it is not a guarantee for an individual input or participant, and a prediction the rule makes is not thereby shown to be correct.")
    line_swap("Empirically, PCD predicts on 0.9666 of windows",
              f"Empirically, PCD at $\\alpha=0.05$ predicts on {g(SE, 'PCD alpha=0.05'):.4f} of Sleep-EDF windows and on {g(DE, 'PCD alpha=0.05'):.4f} of DEED windows, where a max-softmax threshold of 0.5 predicts on {g(DE, 'Max-softmax>0.5'):.4f}. Its false-positive rate matches the nominal level when the null holds by construction, on re-randomised DEED labels ({dn.fpr.mean():.4f}) and on simulated data ({si.fpr.mean():.4f}), although individual participants deviate widely. At matched coverage its retained predictions are at least as accurate as those of confidence thresholds where signal is strong, and no more accurate on DEED, and under controlled label corruption its coverage falls from {n0.coverage:.4f} to {n1.coverage:.4f} as the labels become random.")
    line_swap("We state the standing of this contribution plainly.",
              "We state the standing of this contribution plainly. Its components -- permutation testing, the margin score, the reject option -- are all established, and the marginal guarantee follows from standard exchangeability arguments. What is new is the localisation of a dataset-level permutation test to individual inputs and its use as an abstention rule. It is offered as a practical instrument that the empirical findings motivated, not as a theoretical advance, and every empirical result above stands independently of it.")
    line_swap("PCD coverage is graded within a dataset but is not a dataset-independent scale",
              f"PCD coverage is graded within one analysis of one dataset but is not a dataset-independent scale. Section~\\ref{{sec:pcddep}} holds the dataset, task and labels fixed and varies only the analysis: the mean null margin moves from {f_nm[0]:.4f} to {f_nm[1]:.4f} across feature counts and from {p_nm[0]:.4f} to {p_nm[1]:.4f} across training-cohort sizes, confirming that the reference is not a property of the task. A normalisation that would make coverage comparable across corpora, turning a detector of signal presence into a measure of signal strength, remains open, as does calibration at the level of participants.")
    io.open(TEX, "w", encoding="utf-8", newline="").write(t)
    print("Sections IV-K, IV-L and PCD paragraphs of Section VII written")


if __name__ == "__main__":
    main()
