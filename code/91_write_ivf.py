"""
Writes Section IV-F ("How Much Signal Is Actually There?"), Table 5 and the
new power table into the manuscript, taking every number from the result
files, so that no value is transcribed by hand. Also updates the headline
p-values of Section IV-D.

Sources
  config_table.csv                 65  observed values, both nulls, Holm, max-T
  weak_effect_per_assignment.csv   87  per-assignment p for the logistic models
  weak_effect_jackknife*.csv       87  leave-one-participant-out refits
  statistical_foundation.csv       45  participant-bootstrap intervals, first
                                       fold assignment
  power_revised.csv                78  sensitivity from a true-null baseline
"""
import io
import re
import os
import shutil

import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
TEX = r"E:\Dream_Analysis\accesstemp.tex"


def f4(x):
    return f"{x:.4f}"


def N(k):
    """Counts in the manuscript's style: 1{,}000."""
    return f"{int(k):,}".replace(",", "{,}")


def main():
    c = pd.read_csv(os.path.join(OUT, "config_table.csv")).set_index("config")
    rf, rfb = c.loc["RF, unweighted (headline)"], c.loc["RF, scaled, balanced"]
    lu, lb = c.loc["LogReg, scaled, unweighted"], c.loc["LogReg, scaled, balanced"]
    du = c.loc["Dummy, stratified (control)"]
    BS = int(lb.within_B_single)                      # single-assignment perms
    for r in (rf, rfb, lu, lb, du):
        assert r.within_B_single == BS and r.free_B_single == BS, r.name
    BM_RF, BM_LR = int(rf.within_B_mean10), int(lb.within_B_mean10)
    BM_SH = min(int(c.loc[k, "within_B_mean10"]) for k in
                ("RF, unweighted (headline)", "RF, scaled, balanced",
                 "LogReg, scaled, unweighted", "LogReg, scaled, balanced"))
    floor_s, floor_m = 1 / (BS + 1), 1 / (BM_SH + 1)
    pa = pd.read_csv(os.path.join(OUT, "weak_effect_per_assignment.csv"))
    pb = pa[pa.config == "LogReg, scaled, balanced"]
    pbf, pbw = pb[pb.null == "free"].p, pb[pb.null == "within"].p
    assert pb.B.nunique() == 1
    B_PA = int(pb.B.iloc[0])                          # per-assignment perms
    pu = pa[pa.config == "LogReg, scaled, unweighted"]
    jk = pd.read_csv(os.path.join(OUT, "weak_effect_jackknife_summary.csv"),
                     dtype={"min_left_out": str}).set_index("statistic")
    jr = pd.read_csv(os.path.join(OUT, "weak_effect_jackknife.csv"),
                     dtype={"left_out": str})
    low = jr.sort_values("mean10").head(4)
    sf = pd.read_csv(os.path.join(OUT, "statistical_foundation.csv"))
    ci = sf[sf.part == "uncertainty"].set_index("config")
    ci_rf, ci_lb = ci.loc["RF, unweighted (headline)"], ci.loc["LogReg, scaled, balanced"]
    assert round(ci_lb.balanced, 4) == round(lb.observed_single, 4)
    assert round(ci_rf.balanced, 4) == round(rf.observed_single, 4)
    pw = pd.read_csv(os.path.join(OUT, "power_revised.csv"))
    NUM = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
           6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "all ten"}

    def ALLOF(n):
        return "all ten assignments" if n == 10 else f"{NUM[n]} of the ten assignments"
    import math
    mc_se = max(math.sqrt(q * (1 - q) / BS) for q in
                (lb.within_p_single_maxT, lb.free_p_single_maxT))
    # Does any corrected p-value for the selected configuration fall below
    # 0.05? The wording below follows the result rather than presuming it.
    DEP = min(lb[f"{n}_p_{s}_{c}"] for n in ("within", "free")
              for s in ("single", "mean10") for c in ("holm", "maxT")) < 0.05
    opening = ("Whether it survives correction for the search that found it "
               "depends on the correction." if DEP else
               "It does not survive correction for the search that found it, "
               "under either procedure.")
    signif = ("whether the effect counts as significant depends on the choice "
              "of statistic and correction" if DEP else
              "it is not significant after correction for that search")
    # Selection made inside the procedure (112_nested_selection.py), if run.
    nested = ""
    nf = os.path.join(OUT, "nested_selection_summary.csv")
    if os.path.exists(nf):
        ns = pd.read_csv(nf).set_index("statistic")
        nm_ = ns.loc["mean over ten assignments"]
        ch = ns[ns.index.str.startswith("chosen: ")]["count"].astype(int)
        n_lb = int(ch["chosen: LogReg, scaled, balanced"])
        nested = (
            " Selection can also be made part of the procedure. When the choice "
            "among the four candidates is made inside each training fold, by "
            "participant-separated cross-validation over the training "
            "participants alone, the selected pipeline reaches "
            f"{nm_.bal:.4f} balanced accuracy averaged over the ten fold "
            "assignments on the splitter's partitions "
            f"(participant-bootstrap interval {nm_.bal_lo:.4f} to "
            f"{nm_.bal_hi:.4f}), choosing the balanced logistic regression in "
            f"{n_lb} of {int(ch.sum())} outer folds. This estimate already "
            "accounts for the search, and its interval "
            + ("excludes" if nm_.bal_lo > 1 / 3 else "includes")
            + " the 0.3333 chance level.")

    # Distinct partitions behind the ten fold assignments, and sensitivity to
    # the partition (114_partition_sensitivity.py).
    W = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    fap = pd.read_csv(os.path.join(OUT, "fold_assignment_partitions.csv"))
    fas = pd.read_csv(os.path.join(OUT, "fold_assignment_partitions_summary.csv"))
    fas = fas.set_index(["corpus", "grouping"]).distinct_partitions
    n_dp = int(fas[("DEED", "participant")])
    pmap = fap[(fap.corpus == "DEED") & (fap.grouping == "participant")]
    pmap = dict(zip(pmap.seed, pmap.partition))
    pbp = pb.assign(part=pb.seed.map(pmap))
    by_part = pbp.groupby(["part", "null"]).agg(obs=("observed", "first"),
                                                p=("p", "first"),
                                                n=("p", "nunique"))
    assert (by_part.n == 1).all(), "one value per partition expected"
    parts = sorted(pbp.part.unique())
    assert len(parts) == n_dp == 2, "wording below assumes two partitions"
    o_txt = " and ".join(f4(by_part.loc[(k, "free"), "obs"]) for k in parts)

    def sig_txt(null):
        ps = [by_part.loc[(k, null), "p"] for k in parts]
        k = sum(p < 0.05 for p in ps)
        lead = {2: "both give", 1: "one of the two gives", 0: "neither gives"}[k]
        return f"{lead} $p<0.05$ ({' and '.join(f4(p) for p in ps)})"
    # random participant partitions: class-stratified (as the splitter is)
    # and, as a check, unstratified
    ps_ = pd.read_csv(os.path.join(OUT, "partition_sensitivity_stratified_summary.csv"))
    ps_ = ps_.set_index("config")
    pu_ = pd.read_csv(os.path.join(OUT, "partition_sensitivity_summary.csv"))
    pu_ = pu_.set_index("config")
    plb, prf = ps_.loc["LogReg, scaled, balanced"], ps_.loc["RF, unweighted (headline)"]
    pulb = pu_.loc["LogReg, scaled, balanced"]
    RP = int(plb.partitions)
    assert int(plb.distinct_partitions) == RP == int(pulb.distinct_partitions)
    all_above = int(plb.above_chance) == RP
    above_txt = ("in all " + str(RP) if all_above
                 else f"in {int(plb.above_chance)} of the {RP}")
    separated = plb["min"] > prf["max"]
    split_vals = [by_part.loc[(k, "free"), "obs"] for k in parts]
    above_all_random = min(split_vals) > max(plb["max"], pulb["max"])
    STAB = (
        ("It is not an artefact of one partition, although the ten fold "
         "assignments test that less than their number suggests" if all_above else
         "Whether it survives a change of partition must be tested directly, "
         "because the ten fold assignments test that less than their number "
         "suggests")
        + f": with participants as groups, the ten seeds of the splitter produce "
        f"only {W[n_dp]} distinct partitions of DEED "
        "(Section~\\ref{sec:reporting}), on which the balanced configuration gives "
        f"{o_txt} balanced accuracy. Averaged over the ten assignments it gives "
        f"{f4(lb.observed_mean10)}, and tested as a single statistic against the "
        f"null of the averaged statistic, $p={f4(lb.free_p_mean10)}$ under the "
        f"unrestricted null and {f4(lb.within_p_mean10)} under the "
        "within-participant null. Taken one partition at a time, with "
        f"{N(B_PA)} relabellings each, {sig_txt('free')} under the unrestricted "
        f"null and {sig_txt('within')} under the within-participant null. "
        f"Under the {RP} random participant partitions of the primary "
        f"evaluation, balanced for class in the same way, it gives {f4(plb['mean'])} on average, from {f4(plb['min'])} "
        f"to {f4(plb['max'])}, above chance {above_txt}, and across {RP} drawn "
        f"without class balancing {f4(pulb['mean'])} ({f4(pulb['min'])} to "
        f"{f4(pulb['max'])}); the headline forest gives {f4(prf['mean'])}, from "
        f"{f4(prf['min'])} to {f4(prf['max'])}. "
        + ("The partition is not what separates them. " if separated else
           "Across partitions the two ranges overlap. ")
        + ("Both partitions the splitter produces lie above every random one: "
           f"the value of {f4(lb.observed_mean10)} owes much to the partition, "
           "and the average across random partitions is smaller, "
           f"{f4(plb['mean'])}, and not distinguishable from the permutation "
           "null (Table~\\ref{tab:primary})." if above_all_random else
           f"The value of {f4(lb.observed_mean10)} lies within the range of "
           "random partitions."))
    DEF = (" A fold assignment is one seed of the grouped, stratified splitter, "
           "which for the random forest also seeds the forest. The splitter's "
           "shuffling reorders only groups whose class proportions are equally "
           "spread, so with participants as groups the ten seeds give "
           f"{W[n_dp]} distinct partitions of DEED and "
           f"{W[int(fas[('Tononi', 'participant')])]} of the Tononi corpus, while "
           "recording-grouped and window-level splits differ across all ten. On "
           "DEED a participant-separated model without a seed of its own, such "
           f"as the logistic regression, therefore takes only {W[n_dp]} values "
           "over the ten assignments, and sensitivity to the partition is "
           "measured separately, over random participant partitions, which "
           "also define the primary evaluation (Section~\\ref{sec:howmuch}).")
    assert int(fas[("DEED", "recording")]) == 10
    assert int(fas[("Tononi", "recording")]) == 10

    # ------------------------------------------------ the primary evaluation
    # 120_primary_random.py (perm, summary, nested, refit) and the participant
    # bootstrap of its partition-averaged values (151_primary_bootstrap.py)
    pm = pd.read_csv(os.path.join(OUT, "primary_random_summary.csv")
                     ).set_index("config")
    pbt = pd.read_csv(os.path.join(OUT, "primary_random_boot.csv")
                      ).set_index("config")
    pns = pd.read_csv(os.path.join(OUT, "primary_random_nested_summary.csv")
                      ).iloc[0]
    pch = pd.read_csv(os.path.join(OUT, "primary_random_nested_choices.csv"),
                      index_col=0).iloc[:, 0]
    prf_ = pd.read_csv(os.path.join(OUT, "primary_random_refit_summary.csv")
                       ).set_index("config")
    CANDS = ["RF, unweighted (headline)", "RF, scaled, balanced",
             "LogReg, scaled, unweighted", "LogReg, scaled, balanced"]
    PROSE = {CANDS[0]: "the headline forest",
             CANDS[1]: "the standardised, class-weighted forest",
             CANDS[2]: "the unweighted logistic regression",
             CANDS[3]: "the balanced logistic regression"}
    RPm = int(pm.partitions.iloc[0])
    assert (pm.partitions == RPm).all()
    # the primary procedure reproduces 114's stratified partitions
    assert abs(pm.loc["LogReg, scaled, balanced", "observed_mean"]
               - plb["mean"]) < 5e-5
    pmin = float(pm[["within_p", "free_p"]].min().min())
    any_sig = bool((pm[["within_p_holm", "free_p_holm"]] < 0.05).any().any())
    best = pm.observed_mean.idxmax()
    pb_ = pm.loc[best]
    n_ch_lb = int(pch.get("LogReg, scaled, balanced", 0))
    rf_ratio = (prf_.refit_sd_within_partition
                / prf_.partition_sd_within_seed).max()
    maxT_B = int(pm.within_maxT_B.iloc[0])

    def prow(name, r, bold=False):
        b = pbt.loc[name]
        cells = [f"{f4(r.observed_mean)}",
                 f"[{b.participant_boot_lo:.4f}, {b.participant_boot_hi:.4f}]",
                 f"{r.partition_sd:.4f}",
                 f"{f4(r.partition_min)}--{f4(r.partition_max)}",
                 f"{int(r.above_chance)}",
                 f"{r.within_p:.4f} ({int(r.within_k)})",
                 f"{r.free_p:.4f} ({int(r.free_k)})",
                 f"{r.within_p_holm:.4f} / {r.free_p_holm:.4f}"]
        nm = f"\\textbf{{{name}}}" if bold else name
        return nm + " & " + " & ".join(cells) + r" \\"

    TPRI = "\n".join([
        r"\begin{table*}[htbp]",
        r"\centering",
        (f"\\caption{{The primary evaluation. Balanced accuracy on pooled out-of-fold predictions, "
         f"averaged over {RPm} random class-balanced participant partitions "
         f"(five folds each), with folds recomputed from every labelling. "
         f"Participant interval: 95\\% participant bootstrap of the "
         f"partition-averaged value, predictions held fixed. Partition SD and "
         f"range: across the {RPm} partitions. Permutation $p$: sequential "
         f"Monte Carlo \\cite{{besag1991sequential}}, stopping when 20 null "
         f"statistics reach the observed one or at 1{{,}}000, with the number "
         f"of relabellings used in parentheses; Holm's correction over the "
         f"four configurations. Chance is 0.3333.}}"),
        r"\label{tab:primary}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lcccccccc@{}}",
        r"\toprule",
        r"& & Participant & Partition & & Above chance & \multicolumn{2}{c}{Permutation $p$ (relabellings)} & Holm \\",
        r"\cmidrule(lr){7-8}",
        r"Configuration & Bal.\ acc. & 95\% interval & SD & Range & (of " + str(RPm) + r") & within & unrestricted & within / unrestr. \\",
        r"\midrule",
    ] + [prow(n, pm.loc[n], n == best) for n in CANDS] + [
        r"\midrule",
        (f"Selected among the four in each training fold & {f4(pns.bal)} & "
         f"[{pns.participant_boot_lo:.4f}, {pns.participant_boot_hi:.4f}] & "
         f"{pns.partition_sd:.4f} & {f4(pns.partition_min)}--"
         f"{f4(pns.partition_max)} & {int(pns.above_chance)} & -- & -- & -- \\\\"),
        r"\botrule",
        r"\end{tabular}%",
        r"}",
        r"\normalsize",
        r"\end{table*}",
    ])
    refit_txt = (
        "Refitting the two forests with ten seeds on five of the partitions "
        "moves balanced accuracy by a standard deviation of "
        + " and ".join(f"{prf_.loc[k, 'refit_sd_within_partition']:.4f}"
                       for k in CANDS[:2])
        + " within a partition, against "
        + " and ".join(f"{prf_.loc[k, 'partition_sd_within_seed']:.4f}"
                       for k in CANDS[:2])
        + " across partitions at a fixed seed"
        + (", so the partition contributes more to the spread than "
           "refitting does, though refitting is not negligible."
           if rf_ratio < 1 else
           ", so refitting contributes to the spread across partitions as "
           "much as the partition does."))
    PRIMARY = (
        "\\subsubsection{The primary evaluation}\n\n"
        "Because the splitter used elsewhere in this paper produces only two "
        "distinct participant partitions of DEED (Section~\\ref{sec:reporting}), "
        "one procedure is designated primary. It is chosen for that design "
        "reason, not for its outcome: its observed values were already known "
        "from the partition analysis below, where they are less favourable "
        "than those of the splitter's partitions, and its permutation tests, "
        "the selection pipeline evaluated under it and the refit analysis "
        "were computed after it was fixed. For each of the four candidate "
        "configurations of "
        "Table~\\ref{tab:weak}, balanced accuracy is computed on pooled "
        f"out-of-fold predictions under {RPm} random participant partitions "
        "balanced for class, five folds each, and averaged over them; the "
        "forest seed is held fixed, so the average reflects partitions, not "
        "refits. Folds are recomputed from each labelling, observed or "
        "permuted, so the procedure that is tested is the procedure that is "
        "reported. Each configuration is tested under both nulls of "
        "Section~\\ref{sec:reporting} with a sequential Monte Carlo $p$-value "
        "\\cite{besag1991sequential}: relabellings are taken in a fixed order "
        "and stopped once 20 null statistics reach the observed one, giving "
        "$p = 20/k$ after $k$ relabellings, or at 1{,}000; this is a valid "
        "$p$-value whenever it stops, and it spends relabellings where they "
        "are needed, near small $p$. Holm's procedure corrects across the "
        "four configurations.\n\n"
        "Table~\\ref{tab:primary} gives the result"
        + (". No configuration is distinguishable from its permutation null. "
           if not any_sig and pmin >= 0.05 else
           ". ")
        + f"The highest, {PROSE[best]}, averages {f4(pb_.observed_mean)} "
        f"(participant interval {pbt.loc[best, 'participant_boot_lo']:.4f} to "
        f"{pbt.loc[best, 'participant_boot_hi']:.4f}; "
        f"{f4(pb_.partition_min)} to {f4(pb_.partition_max)} across "
        f"partitions, above chance in {int(pb_.above_chance)} of {RPm}), with "
        f"$p={pb_.within_p:.4f}$ under the within-participant null and "
        f"${pb_.free_p:.4f}$ under the unrestricted null, "
        f"{pb_.within_p_holm:.4f} and {pb_.free_p_holm:.4f} after Holm's "
        "correction. The headline forest averages "
        f"{f4(pm.loc[CANDS[0], 'observed_mean'])}, with $p="
        f"{pm.loc[CANDS[0], 'within_p']:.4f}$ and "
        f"${pm.loc[CANDS[0], 'free_p']:.4f}$. "
        "The max-statistic correction is not used here: the configurations "
        "stop at different numbers of relabellings, and only the "
        f"{maxT_B} computed for all four can enter it, whose smallest "
        f"attainable value is {1 / (maxT_B + 1):.4f}; it gives "
        f"{pb_.within_p_maxT:.4f} and {pb_.free_p_maxT:.4f} for the "
        "highest configuration.\n\n"
        "Selection can also be made part of the procedure. When the choice "
        "among the four is made inside each training fold of each random "
        "partition, by a random class-balanced participant partition of the "
        "training participants alone, the selected pipeline averages "
        f"{f4(pns.bal)} (participant interval {pns.participant_boot_lo:.4f} to "
        f"{pns.participant_boot_hi:.4f}; {f4(pns.partition_min)} to "
        f"{f4(pns.partition_max)} across partitions), choosing the balanced "
        f"logistic regression in {n_ch_lb} of {int(pch.sum())} outer folds. "
        "This estimate already accounts for the search, and its participant "
        "interval "
        + ("excludes" if pns.participant_boot_lo > 1 / 3 else "includes")
        + " the 0.3333 chance level. " + refit_txt + " The three sources of "
        "uncertainty are reported separately because they answer different "
        "questions: which participants were evaluated, how they were "
        "partitioned, and how the forests were initialised.")
    # The claim follows the result; a result that changes it is written by
    # hand, not generated.
    if any_sig or pmin < 0.05 or pns.participant_boot_lo > 1 / 3:
        raise SystemExit("primary evaluation detects an effect: rewrite the "
                         "claim of Section IV-F by hand")
    holm_min = float(pm[["within_p_holm", "free_p_holm"]].min().min())
    CLAIM_PRI = (
        "Under the primary evaluation no configuration of the four is "
        f"distinguishable from its permutation null (smallest $p$ {pmin:.4f}; "
        + ("1.0000 for every configuration after Holm's correction"
           if holm_min >= 1 else
           f"after Holm's correction {holm_min:.4f} or more")
        + "), and the selection "
        f"pipeline evaluated under the same procedure averages {f4(pns.bal)}, "
        "with a participant interval that includes chance.")
    CLAIM_BOLD = ("the data show no participant-generalizable dream-affect "
                  "effect under the primary evaluation, and at most a weak, "
                  "partition-dependent one under the configurations tested")

    def P(n, d):
        r = pw[(pw.cohort == n) & (abs(pw.delta - d) < 1e-9)].iloc[0]
        return r

    # --------------------------------------------------------------- table 5
    def row(name, r, bold=False):
        cells = [f4(r.observed_single), f4(r.within_p_single),
                 f4(r.free_p_single)]
        if name.startswith("Dummy"):
            cells += ["--", "--"]
        else:
            cells += [f"{r.within_p_single_holm:.4f} / {r.free_p_single_holm:.4f}",
                      f"{r.within_p_single_maxT:.4f} / {r.free_p_single_maxT:.4f}"]
        s = f"{name} & " + " & ".join(cells) + r" \\"
        return (r"\textbf{" + name + "}" + s[len(name):]) if bold else s

    def row10(name, r, bold=False):
        cells = [f"{r.observed_mean10:.4f} $\\pm$ {r.sd_mean10:.4f}",
                 f4(r.within_p_mean10), f4(r.free_p_mean10)]
        if name.startswith("Dummy"):
            cells += ["--", "--"]
        else:
            cells += [f"{r.within_p_mean10_holm:.4f} / {r.free_p_mean10_holm:.4f}",
                      f"{r.within_p_mean10_maxT:.4f} / {r.free_p_mean10_maxT:.4f}"]
        s = f"{name} & " + " & ".join(cells) + r" \\"
        return (r"\textbf{" + name + "}" + s[len(name):]) if bold else s

    names = [("RF, unweighted (headline)", rf), ("RF, scaled, balanced", rfb),
             ("LogReg, scaled, unweighted", lu),
             ("LogReg, scaled, balanced", lb),
             ("Dummy, stratified (control)", du)]
    t5 = "\n".join([
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{How much dream-affect signal is present under participant separation. Five configurations on identical data and folds, each tested against the same relabellings under both null schemes of Section~\ref{sec:reporting}. Panel (a): the first fold assignment. Panel (b): the mean over ten fold assignments (two distinct participant partitions; Section~\ref{sec:reporting}), tested as a single statistic. Holm and max-statistic corrections run over the four candidate classifiers; the stratified dummy is a calibration control and is not counted. Chance is 0.3333.}",
        r"\label{tab:weak}",
        r"\small",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{Permutation $p$} & \multicolumn{2}{c}{Corrected $p$ (within / unrestricted)} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Configuration & Bal.\ acc. & within & unrestricted & Holm & max-statistic \\",
        r"\midrule",
        r"\multicolumn{6}{@{}l}{\textit{(a) First fold assignment}} \\",
    ] + [row(n, r, n.startswith("LogReg, scaled, balanced")) for n, r in names] + [
        r"\midrule",
        r"\multicolumn{6}{@{}l}{\textit{(b) Mean $\pm$ SD over ten fold assignments, null of the averaged statistic}} \\",
    ] + [row10(n, r, n.startswith("LogReg, scaled, balanced")) for n, r in names] + [
        r"\botrule",
        r"\end{tabular}",
        (f"\\footnotetext{{Permutations: {N(BS)} per null in panel (a) for every configuration; in panel (b), {N(BM_LR)} for logistic regression and the dummy and {N(BM_RF)} for the forests, whose averaged statistic costs ten forest fits per permutation. The max-statistic correction uses the permutations shared by all four candidates, {N(BS)} in panel (a) and {N(BM_SH)} in panel (b), so its smallest attainable value is {floor_s:.4f} and {floor_m:.4f}. Participant-bootstrap 95\\% intervals at the first fold assignment: headline forest [{ci_rf.bal_lo:.4f}, {ci_rf.bal_hi:.4f}], balanced logistic regression [{ci_lb.bal_lo:.4f}, {ci_lb.bal_hi:.4f}].}}"),
        r"\normalsize",
        r"\end{table*}",
    ])

    # --------------------------------------------------------------- power table
    deltas = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20]
    prow = []
    for d in deltas:
        cells = []
        for n in (10, 20, 34):
            r = P(n, d)
            cells += [f4(r.achieved_bal), f"{r.power:.2f}"]
        prow.append(f"{d:.2f} & " + " & ".join(cells) + r" \\")
    tp = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Sensitivity of the participant-separated protocol, from a true-null baseline. Balanced accuracy achieved by the injected shift $\delta$ (mean over simulations) and the proportion of simulations exceeding the 95th percentile of the $\delta=0$ simulations for that cohort size. 100 simulations per cell; Monte Carlo standard error at most 0.05. Pre-specified configuration: logistic regression, standardised, class-balanced.}",
        r"\label{tab:power}",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{10 participants} & \multicolumn{2}{c}{20 participants} & \multicolumn{2}{c}{34 participants} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"$\delta$ & Bal.\ acc. & Power & Bal.\ acc. & Power & Bal.\ acc. & Power \\",
        r"\midrule",
    ] + prow + [
        r"\botrule",
        r"\end{tabular}",
        r"\normalsize",
        r"\end{table}",
    ])

    # the value that holds across random partitions, against the power bound
    POWER_ADD = (f" The value that holds across random partitions, "
                 f"{f4(plb['mean'])}, is "
                 + ("below" if plb["mean"] < P(34, 0.02).achieved_bal else "above")
                 + " that bound.")

    # --------------------------------------------------------------- text
    txt = f"""\\subsection{{How Much Signal Is Actually There?}}\\label{{sec:howmuch}}

A null result is only as strong as the search behind it. Three questions decide what the preceding subsection is entitled to claim: what a single evaluation procedure fixed in advance shows, whether the result depends on the classifier and the partition that produced it, and how large an effect the protocol could have detected had one been present.

{PRIMARY}

{TPRI}

\\subsubsection{{At the splitter's partitions: dependence on the classifier}}

The analyses in the rest of this subsection use the splitter's fold assignments, as the other sections of the paper do. They are secondary to the primary evaluation and show how the result depends on the classifier, the partition and the participants. Table~\\ref{{tab:weak}} repeats the participant-separated evaluation across five configurations, holding the data and the folds fixed and testing every configuration against the same relabellings, under both null schemes of Section~\\ref{{sec:reporting}}. The headline configuration -- an unweighted random forest on unstandardised features -- reaches {f4(rf.observed_single)} balanced accuracy at the first fold assignment and {f4(rf.observed_mean10)} averaged over ten, below the 0.3333 chance level, with permutation $p$-values of {f4(rf.within_p_single)} and {f4(rf.free_p_single)} under the within-participant and unrestricted nulls. A stratified dummy classifier, run as a calibration control, reaches {f4(du.observed_single)} with $p={f4(du.within_p_single)}$ and ${f4(du.free_p_single)}$, confirming that the testing machinery does not manufacture significance.

One configuration behaves differently. A logistic regression on standardised features with balanced class weights reaches \\textbf{{{f4(lb.observed_single)}}} balanced accuracy at the first fold assignment, and its participant-bootstrap 95\\% interval, $[{ci_lb.bal_lo:.4f}, {ci_lb.bal_hi:.4f}]$, excludes chance. Its permutation $p$ is {f4(lb.free_p_single)} under the unrestricted null and {f4(lb.within_p_single)} under the within-participant null. Removing the class weighting gives {f4(lu.observed_single)}, with $p={f4(lu.within_p_single)}$ under the within-participant null and {f4(lu.free_p_single)} under the unrestricted null, so the effect is specific to the weighted variant.

This result is reported because it bears directly on how a null should be read. A configuration that cannot recover a weak effect even when one is present will report a null whatever the data contain, so a null obtained from a single configuration is a statement about that configuration as much as about the data. Reporting the configuration sweep alongside the headline figure is what makes the two separable.

\\subsubsection{{The effect across partitions and participants}}

Three properties of that effect bear on how much weight it can take, and they do not all point the same way.

{STAB}

{opening} The four candidate configurations were examined together, and this one was singled out after inspecting their results rather than nominated in advance. Holm's procedure, which does not use the correlation between configurations, gives {lb.within_p_single_holm:.4f} and {lb.free_p_single_holm:.4f} at the first fold assignment (within-participant and unrestricted nulls) and {lb.within_p_mean10_holm:.4f} and {lb.free_p_mean10_holm:.4f} for the ten-assignment mean. The single-step max-statistic correction, which accounts for that correlation because every configuration is scored on the same relabellings, gives {lb.within_p_single_maxT:.4f} and {lb.free_p_single_maxT:.4f} at the first fold assignment, each with a Monte Carlo standard error of about {mc_se:.3f} at {N(BS)} relabellings; for the ten-assignment mean only {N(BM_SH)} relabellings are shared by all four configurations, so it cannot fall below {floor_m:.4f} and gives {lb.within_p_mean10_maxT:.4f} and {lb.free_p_mean10_maxT:.4f}.{nested}

It rests on a minority of participants. A leave-one-participant-out jackknife, which refits the whole procedure without each participant in turn, leaves the ten-assignment mean above chance in all {int(jk.loc['mean10', 'n_above_chance'])} cases, but the spread is wide: from {jk.loc['mean10', 'min']:.4f}, with participant {jk.loc['mean10', 'min_left_out']} removed, to {jk.loc['mean10', 'max']:.4f}, with a median of {jk.loc['mean10', 'median']:.4f}, below the full-cohort {f4(lb.observed_mean10)}. Removing any one of four participants lowers it to between {low.mean10.min():.4f} and {low.mean10.max():.4f}. A handful of individuals carry much of the effect, which 34 participants cannot resolve.

\\subsubsection{{What could have been detected}}

A null reported without a power analysis says nothing about what was detectable. The sensitivity of the protocol was therefore measured from a true null. Labels were permuted across all recordings, which removes any association between EEG and label while keeping one label per recording and the class proportions; a class-dependent mean shift of $\\delta$ standard deviations along a random direction of the standardised feature space was then injected, and the participant-separated evaluation of the balanced logistic regression was run on the result. Detection means exceeding the 95th percentile of the $\\delta=0$ simulations for the same cohort size, which are draws from the null, so the false-positive rate is 0.05 by construction. Every simulation draws a new participant subset, permutation and direction, 100 per cell, and power is reported against the balanced accuracy the injection achieves, so no calibration between $\\delta$ and effect size need be assumed (Table~\\ref{{tab:power}}).

At 34 participants power is {P(34,0.02).power:.2f} for injections achieving {f4(P(34,0.02).achieved_bal)} balanced accuracy, {P(34,0.04).power:.2f} at {f4(P(34,0.04).achieved_bal)} and {P(34,0.06).power:.2f} at {f4(P(34,0.06).achieved_bal)}. At 20 participants it is {P(20,0.04).power:.2f} at {f4(P(20,0.04).achieved_bal)} and {P(20,0.08).power:.2f} at {f4(P(20,0.08).achieved_bal)}; at 10 participants it reaches {P(10,0.14).power:.2f} only at {f4(P(10,0.14).achieved_bal)}.

The observed ten-assignment mean, {f4(lb.observed_mean10)}, lies between the achieved values at which power at 34 participants is {P(34,0.02).power:.2f} and {P(34,0.04).power:.2f}, so an effect of that size is detected by a cohort of this size in some replications and missed in others; this is why the partition, multiplicity and jackknife checks above carry more weight than the point estimate. The simulation also bounds what a negative result can mean: effects producing balanced accuracy below about {f4(P(34,0.02).achieved_bal)} would usually be missed by 34 participants.{POWER_ADD} The injected signal is a linear mean shift shared by all participants, so these figures describe the protocol's sensitivity to that structure only; signal that is nonlinear, participant-specific or confined to a subset of features would be detected differently.

It also gives the sample-size objection a concrete form. The effective sample is the 34 participants, not the 6{{,}}144 windows, and an effect of the size seen here sits near the edge of what 34 participants can resolve.

{tp}

\\subsubsection{{The claim this supports}}

{CLAIM_PRI}

At the two partitions the splitter produces, the balanced logistic regression reaches {f4(lb.observed_mean10)} averaged over the ten fold assignments, against a chance level of 0.3333, with a participant-bootstrap interval of $[{ci_lb.bal_lo:.4f}, {ci_lb.bal_hi:.4f}]$ at the first fold assignment. That configuration was selected after inspecting four candidates, {signif}, and the value is sensitive to the partition and to participant composition, so it is reported as exploratory.

The claim we therefore make is that \\textbf{{{CLAIM_BOLD}}}. This is not a demonstration that no effect exists: at 34 participants an effect giving balanced accuracy of {f4(P(34,0.02).achieved_bal)} is detected in {P(34,0.02).power*100:.0f}\\% of simulations (Table~\\ref{{tab:power}})."""

    s = io.open(TEX, encoding="utf-8", newline="").read()
    shutil.copyfile(TEX, TEX + ".pre_ivf")
    a = s.index(r"\subsection{How Much Signal Is Actually There?}")
    b = s.index("Three quantities appear in this paper and are easily conflated", a)
    s = s[:a] + txt + "\n\n" + s[b:]
    ta = s.index(r"\begin{table*}[htbp]", s.index("Three quantities appear in this paper"))
    assert "\\label{tab:weak}" in s[ta:ta + 800]
    tb = s.index(r"\end{table*}", ta) + len(r"\end{table*}")
    s = s[:ta] + t5 + s[tb:]

    new = (f"At the first fold assignment the observed {f4(rf.observed_single)} "
           f"gives a one-sided $p={f4(rf.within_p_single)}$ against {N(BS)} "
           f"within-participant relabellings (null mean "
           f"{f4(rf.within_null_mean_single)}) and ${f4(rf.free_p_single)}$ "
           f"against {N(BS)} unrestricted ones (null mean "
           f"{f4(rf.free_null_mean_single)}): under this configuration there is "
           f"no statistically significant decodability (Table~\\ref{{tab:weak}}).")
    pat = re.compile(r"At the first fold assignment the observed 0\.\d{4} gives a "
                     r"one-sided .*?no statistically significant decodability "
                     r"\(Table~\\ref\{tab:weak\}\)\.", re.S)
    old_v1 = ("A permutation test over 200 within-participant label reassignments "
              "places the observed value against a null mean of 0.3346 and yields "
              "a one-sided $p = 0.3582$: under this configuration there is no "
              "statistically significant decodability.")
    if old_v1 in s:
        s = s.replace(old_v1, new)
    else:
        assert len(pat.findall(s)) == 1
        s = pat.sub(lambda m: new, s)
    # what a fold assignment is, in the statistical procedures (re-runnable)
    anchor = "so the null distribution carries the same dependence as the observed mean."
    s = re.sub(r" A fold assignment is one seed of the grouped, stratified "
               r"splitter.*?over random participant partitions(?:, which also "
               r"define the primary evaluation)? "
               r"\(Section~\\ref\{sec:howmuch\}\)\.", "", s)
    assert s.count(anchor) == 1
    s = s.replace(anchor, anchor + DEF)
    io.open(TEX, "w", encoding="utf-8", newline="").write(s)
    print("Section IV-F, Table 5, power table and IV-D p-values written")


if __name__ == "__main__":
    main()
