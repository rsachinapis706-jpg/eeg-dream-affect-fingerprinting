"""
Writes the stratified-PCD extension (110_pcd_stratified.py) into the
manuscript: Proposition 2 and its scope in Section III-E, a paragraph and
table in Section IV-K, and matching one-line changes in the Introduction,
Table 3, Table 1, Limitations and Conclusion. Every number comes from the
result files. The text is inserted only if the results support it: the
participant-stratified rule must stay within chance of alpha for every
participant on the DEED null and must reduce participant-level excess in the
simulations. Re-runnable: each insertion replaces its own earlier version.
"""
import io
import os
import shutil

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
TEX = os.path.join("E:" + os.sep, "Dream_Analysis", "accesstemp.tex")
ALPHA = 0.05


def f4(x):
    return f"{x:.4f}"


def load():
    sm = pd.read_csv(os.path.join(OUT, "pcd_strat_summary.csv"))
    sm = sm[sm.alpha == ALPHA].set_index(["condition", "stratum"])
    nl = pd.read_csv(os.path.join(OUT, "pcd_strat_deednull_summary.csv"))
    nl = nl.set_index("stratum")
    si = pd.read_csv(os.path.join(OUT, "pcd_strat_sims.csv"))
    si = si.groupby("stratum").agg(share=("part_frac_above_2alpha", "mean"),
                                   p90=("part_fpr_p90", "median"),
                                   fpr=("fpr", "mean"))
    si.round(4).to_csv(os.path.join(OUT, "pcd_strat_sims_summary.csv"))
    pp = pd.read_csv(os.path.join(OUT, "pcd_strat_deednull_participants.csv"))
    # Round 3: every PCD run draws its folds without labels
    # (77.label_free_folds), so the null runs satisfy the propositions
    # directly and no separate fixed-partition run is needed.
    nlf = nl
    return sm, nl, nlf, si, pp


def support(nlf, si, pp):
    """The extension goes in only if the results bear it out, judged on the
    fixed-partition runs, where the propositions' conditions hold."""
    R = int(nlf.loc["participant", "reps"])
    n_part = pp.subj.nunique()
    ok = (nlf.loc["participant", "n_participants_signif_above_alpha"]
          <= max(2, int(np.ceil(0.05 * n_part)))
          and nlf.loc["participant", "participant_expected_max"]
          < nlf.loc["pooled", "participant_expected_max"]
          and si.loc["participant", "share"] < si.loc["pooled", "share"]
          and nlf.loc["participant", "marginal_rate"]
          <= ALPHA + 2 * nlf.loc["participant", "marginal_se"])
    return ok, R, n_part


FOLD_COND_OLD = ("Let every step of fitting that uses labels -- scaling fitted with "
                 "labels, feature selection, resampling, fine-tuning -- be repeated "
                 "identically for the observed and for each permuted labelling.")
FOLD_COND_NEW = ("Let every step that uses labels -- the assignment of folds, scaling "
                 "fitted with labels, feature selection, resampling, fine-tuning -- be "
                 "repeated identically for the observed and for each permuted "
                 "labelling, or fixed without reference to the labels.")
CAL_ANCHOR = ("in agreement with Proposition~\\ref{prop:calib} within the variation "
              "across re-randomisations.")
CAL_ADD = (" The folds are drawn without reference to any labels, so the condition of "
           "Proposition~\\ref{prop:calib} holds for every labelling; over @R@ "
           "re-randomisations the rate is @SP@ (standard error @SPSE@).")


PROP = r"""\begin{proposition}[Validity within a stratum]\label{prop:strat}
Let $J$ be a set of test inputs fixed without reference to their labels, such as the windows of one participant, and let $p_J(\mathbf{x}_i)$, for $i \in J$, be Equation~\eqref{eq:dscore} with $j$ restricted to $J$. Under the conditions of Proposition~\ref{prop:calib}, if $I$ is drawn uniformly from $J$, independently of the data and the permutations, then $\Pr[p_J(\mathbf{x}_I) \le \alpha] \le \alpha$ for every $\alpha \in [0,1]$ and every $B \ge 1$.
\end{proposition}

\noindent\emph{Proof.} Under $H_0$ the rows of the margin array are exchangeable, and so are the rows of its restriction to the columns in $J$, which are fixed given the inputs. The argument of Proposition~\ref{prop:calib}, applied to the restricted array, gives the bound. $\square$

With $J$ the windows of one participant, the expected share of that participant's windows on which the rule predicts under $H_0$ is at most $\alpha$, for every participant; with $J$ a single input, the score is the classical permutation $p$-value of that input, $(1 + |\{b \ge 1 : s_b(\mathbf{x}_i) \ge s_0(\mathbf{x}_i)\}|)/(B+1)$, valid for the input but resolved only in steps of $1/(B+1)$. The stratum is therefore a choice of the unit at which calibration is required, as the partition is a choice of the unit at which generalisation is claimed; its counterpart in conformal prediction is the group-conditional, or Mondrian, construction \cite{vovk2005algorithmic}. A smaller stratum gives a stronger guarantee from a smaller reference, and Section~\ref{sec:pcdres} measures how that changes which inputs are retained."""

RES = r"""Ranking each margin only against the null margins of the same participant's windows (Proposition~\ref{prop:strat}) restores calibration at that level (Table~\ref{tab:strat}). Over @R@ re-randomisations of DEED, with folds drawn without reference to any labels, each participant's expected rate of prediction is estimated by its mean over re-randomisations. With the pooled reference the highest is @POOLMAX@, and @POOLSIGTXT@; ranked within participant, the highest is @PARTMAX@ and @PARTSIGTXT@, about what chance produces among @NP@. In the simulations the share of participants flagged at more than twice the nominal rate falls from @SIMPOOL@ to @SIMPART@; that share reflects variability within a single dataset as well as bias, and it is lowest for the participant reference, whose stratum is the unit being assessed. @COVLEAD@: at $\alpha=0.05$ PCD predicts on @SPOOL@ of Sleep-EDF windows with the pooled reference, @SPART@ within participant and @SINPUT@ with each input ranked against its own null margins alone, and on @DPOOL@, @DPART@ and @DINPUT@ of DEED windows. What changes is which windows are retained. A window is retained whenever its margin exceeds the null margins of its stratum, however small that margin is, so the smaller the stratum, the less confident the retained predictions: on Sleep-EDF the 10th percentile of retained margins is @M10POOL@ pooled, @M10PART@ within participant and @M10INPUT@ for the input alone, and retained balanced accuracy is @BPOOL@, @BPART@ and @BINPUT@."""

TABLE = r"""\begin{table*}[htbp]
\centering
\caption{PCD with the reference pooled over all test inputs (Proposition~\ref{prop:calib}) or restricted to a stratum (Proposition~\ref{prop:strat}). DEED null: recording labels re-randomised within participant @R@ times, folds drawn without reference to any labels; a participant's expected rate is the mean of their rate over re-randomisations. Simulations: the 100 null datasets of Section~\ref{sec:pcdres}, folds as in the other analyses, mean over datasets. Coverage and retained balanced accuracy at $\alpha=0.05$ on the real labels. Same folds, models and null models in every row.}
\label{tab:strat}
\small
\begin{tabular}{@{}lccccccc@{}}
\toprule
& \multicolumn{3}{c}{DEED null} & Simulations & \multicolumn{2}{c}{Sleep-EDF staging} & DEED affect \\
\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-7}\cmidrule(lr){8-8}
Reference & Rate & Highest participant & Above $\alpha$\footnotemark[1] & Share $>2\alpha$ & Coverage & Retained bal.\ acc. & Coverage \\
\midrule
@ROWS@
\botrule
\end{tabular}
\footnotetext[1]{Participants whose mean rate exceeds $\alpha$ by more than two standard errors, of @NP@.}
\normalsize
\end{table*}"""

NAMES = [("pooled", "Pooled (all test inputs)"), ("recording", "Recording"),
         ("participant", "Participant"), ("input", "Input alone")]


def main():
    import re
    sm, nl, nlf, si, pp = load()
    t = io.open(TEX, encoding="utf-8", newline="").read()
    shutil.copyfile(TEX, TEX + ".pre_strat")

    # 1. The fold condition of Proposition 1 and its measured consequence.
    #    A correction, made whether or not the extension goes in.
    if FOLD_COND_OLD in t:
        t = t.replace(FOLD_COND_OLD, FOLD_COND_NEW)
    assert t.count(FOLD_COND_NEW) == 1
    t = re.sub(r" The folds used here are stratified on the re-randomised labels and are "
               r"not recomputed.*?labels it is 0\.\d{4} \(0\.\d{4}\)\.", "", t)
    t = re.sub(r" The folds are drawn without reference to any labels, so the "
               r"condition of .*?\(standard error 0\.\d{4}\)\.", "", t)
    cal = (CAL_ADD.replace("@R@", str(int(nl.loc["pooled", "reps"])))
           .replace("@SP@", f4(nl.loc["pooled", "marginal_rate"]))
           .replace("@SPSE@", f4(nl.loc["pooled", "marginal_se"])))
    assert t.count(CAL_ANCHOR) == 1
    t = t.replace(CAL_ANCHOR, CAL_ANCHOR + cal)

    # 2. The stratified extension, only if the results bear it out.
    ok, R, NP = support(nlf, si, pp)
    if not ok:
        io.open(TEX, "w", encoding="utf-8", newline="").write(t)
        print("fold condition written; results do not support the stratified "
              "extension, which is NOT added")
        print(nlf.round(4).to_string())
        print(si.round(4).to_string())
        return
    S, D = "Sleep-EDF staging", "DEED affect"
    # margins of the retained Sleep-EDF windows, written to a result file
    w = pd.read_parquet(os.path.join(OUT, "pcd_strat_windows_sleepedf.parquet"))
    mr = pd.DataFrame([dict(stratum=st, retained=(w[f"p_{st}"] <= ALPHA).mean(),
                            margin_p10_retained=w.margin[w[f"p_{st}"] <= ALPHA]
                            .quantile(.1),
                            margin_mean_retained=w.margin[w[f"p_{st}"] <= ALPHA]
                            .mean()) for st, _ in NAMES]).set_index("stratum")
    mr.round(4).to_csv(os.path.join(OUT, "pcd_strat_retained_margins.csv"))
    cov_up = sm.loc[(S, "participant"), "coverage"] >= sm.loc[(S, "pooled"), "coverage"]
    acc_down = (sm.loc[(S, "participant"), "bal_retained"]
                < sm.loc[(S, "pooled"), "bal_retained"])
    rows = []
    for st, nm in NAMES:
        rows.append(" & ".join([
            nm, f4(nlf.loc[st, "marginal_rate"]),
            f4(nlf.loc[st, "participant_expected_max"]),
            str(int(nlf.loc[st, "n_participants_signif_above_alpha"])),
            f4(si.loc[st, "share"]), f4(sm.loc[(S, st), "coverage"]),
            f4(sm.loc[(S, st), "bal_retained"]),
            f4(sm.loc[(D, st), "coverage"])]) + r" \\")
    W = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
         6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    n_pool = int(nlf.loc["pooled", "n_participants_signif_above_alpha"])
    n_part = int(nlf.loc["participant", "n_participants_signif_above_alpha"])
    pool_txt = (f"{W.get(n_pool, n_pool)} of the {NP} participants "
                f"{'lies' if n_pool == 1 else 'lie'} more than two standard "
                "errors above $\\alpha$")
    part_txt = ("none lies above $\\alpha$ by that margin" if n_part == 0 else
                f"{W.get(n_part, n_part)} "
                f"{'lies' if n_part == 1 else 'lie'} above $\\alpha$ by that margin")
    assert si.loc["participant", "share"] == si.share.min()
    sub = {"@R@": str(R), "@NP@": str(NP),
           "@POOLSIGTXT@": pool_txt, "@PARTSIGTXT@": part_txt,
           "@POOLMAX@": f4(nlf.loc["pooled", "participant_expected_max"]),
           "@PARTMAX@": f4(nlf.loc["participant", "participant_expected_max"]),
           "@SIMPOOL@": f4(si.loc["pooled", "share"]),
           "@SIMPART@": f4(si.loc["participant", "share"]),
           "@SPOOL@": f4(sm.loc[(S, "pooled"), "coverage"]),
           "@SPART@": f4(sm.loc[(S, "participant"), "coverage"]),
           "@SINPUT@": f4(sm.loc[(S, "input"), "coverage"]),
           "@DPOOL@": f4(sm.loc[(D, "pooled"), "coverage"]),
           "@DPART@": f4(sm.loc[(D, "participant"), "coverage"]),
           "@DINPUT@": f4(sm.loc[(D, "input"), "coverage"]),
           "@COVLEAD@": ("Stratifying does not reduce coverage where signal is present"
                         if cov_up else "Stratifying reduces coverage where signal is present"),
           "@M10POOL@": f4(mr.loc["pooled", "margin_p10_retained"]),
           "@M10PART@": f4(mr.loc["participant", "margin_p10_retained"]),
           "@M10INPUT@": f4(mr.loc["input", "margin_p10_retained"]),
           "@BPOOL@": f4(sm.loc[(S, "pooled"), "bal_retained"]),
           "@BPART@": f4(sm.loc[(S, "participant"), "bal_retained"]),
           "@BINPUT@": f4(sm.loc[(S, "input"), "bal_retained"]),
           "@ROWS@": "\n".join(rows)}
    res, table = RES, TABLE
    for k, v in sub.items():
        res, table = res.replace(k, v), table.replace(k, v)
    assert not re.search(r"@[A-Z0-9]+@", res + table), "unfilled placeholder"

    def remove_block(start, end):
        nonlocal t
        if start in t:
            i = t.index(start)
            j = t.index(end, i) + len(end)
            t = t[:i] + t[j:]

    # re-run: remove earlier versions
    remove_block("\n\n\\begin{proposition}[Validity within a stratum]",
                 "Section~\\ref{sec:pcdres} measures how that changes which inputs are retained.")
    # earlier paragraph and table: from the paragraph to the \end{table*}
    # that closes the table labelled tab:strat
    p0 = "\n\nRanking each margin only against the null margins of the same participant's windows"
    if p0 in t:
        i = t.index(p0)
        lab = t.index("\\label{tab:strat}", i)
        j = t.index("\\end{table*}", lab) + len("\\end{table*}")
        t = t[:i] + t[j:]
    assert "\\label{tab:strat}" not in t, "stray copy of the stratified table"

    # Section III-E: Proposition 2 after the scope paragraph of Proposition 1
    a = "Section~\\ref{sec4} therefore reports it across re-randomisations and simulated datasets rather than from a single run."
    assert t.count(a) == 1
    i = t.index(a) + len(a)
    t = t[:i] + "\n\n" + PROP + t[i:]
    # Section IV-K: paragraph and table after the calibration paragraph
    b = "Marginal validity holds; calibration for a participant does not, as the proposition states."
    b2 = "Marginal validity holds; calibration for a participant does not, as Proposition~\\ref{prop:calib} states."
    if b in t:
        t = t.replace(b, b2)
    assert t.count(b2) == 1
    i = t.index(b2) + len(b2)
    t = t[:i] + "\n\n" + res + "\n\n" + table + t[i:]

    def swap(old, new):
        nonlocal t
        if old in t:
            assert t.count(old) == 1, old[:60]
            t = t.replace(old, new)
        assert t.count(new) == 1, new[:60]

    swap("It does not establish calibration for a participant, since the marginal rate averages over inputs and a participant contributing many atypical inputs can exceed it.",
         "It does not establish calibration for a participant, since the marginal rate averages over inputs and a participant contributing many atypical inputs can exceed it; Proposition~\\ref{prop:strat} below shows how to obtain it.")
    swap("& \\textbf{Coverage at most $\\alpha$ on average over inputs} \\\\",
         "& \\textbf{Coverage at most $\\alpha$ on average over inputs, or for every participant when stratified} \\\\")
    swap("and the marginal guarantee it carries follows from standard exchangeability arguments \\cite{hemerik2018exact}.",
         "and the guarantee it carries, on average over inputs or, with a stratified reference, for every participant, follows from standard exchangeability arguments \\cite{hemerik2018exact}.")
    swap("its marginal guarantee follows from standard exchangeability arguments.",
         "its guarantees, marginal or within a stratum, follow from standard exchangeability arguments.")
    swap("The guarantee is marginal over inputs; it is not a guarantee for an individual input or participant, and a prediction the rule makes is not thereby shown to be correct.",
         "With the reference pooled over inputs the guarantee is marginal; ranking each margin within its participant's windows extends it to every participant (Proposition~\\ref{prop:strat}) "
         + ("without reducing coverage" if cov_up else "at some cost in coverage")
         + (", although the predictions it retains are slightly less accurate"
            if acc_down else "")
         + ", and a prediction the rule makes is not thereby shown to be correct.")
    swap("; a normalisation that would make it comparable across corpora remains open, as does calibration at the level of participants.",
         "; a normalisation that would make it comparable across corpora remains open.")
    swap("unit-level within participant, $B=100$ per fold ($B=50$ in simulations) \\\\",
         "unit-level within participant, $B=100$ per fold ($B=50$ in simulations); reference pooled or by recording, participant or input \\\\")
    swap("DEED re-randomised 20 times;",
         f"DEED re-randomised 20 times ({R} for Table~\\ref{{tab:strat}});")
    io.open(TEX, "w", encoding="utf-8", newline="").write(t)
    print("stratified PCD written:", res[:200], "...")


if __name__ == "__main__":
    main()
