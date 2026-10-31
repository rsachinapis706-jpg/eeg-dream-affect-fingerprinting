"""
Writes the Abstract, Section VII-A, the weak-effect paragraph of Section VII-C
and the comparison-table footnote from the result files (Reviewer 1,
comments 5, 8 and 9; Reviewer 2 on the exploratory 0.3768 and on scope).
The abstract is kept within 250 words.
"""
import io
import os
import shutil

import pandas as pd

OUT = r"E:\Dream GAN\analysis\out"
TEX = r"E:\Dream_Analysis\accesstemp.tex"


def rd(n, **k):
    return pd.read_csv(f"{OUT}\\{n}", **k)


def main():
    # The PCD sentence of the abstract mentions the stratified guarantee only
    # if 111_write_pcd_strat.py has put Proposition 2 into the manuscript.
    tex_now = io.open(TEX, encoding="utf-8").read()
    strat = "\\label{prop:strat}" in tex_now
    # ... and that it holds under a leaky partition only if
    # 117_write_pcd_leaky.py has written that result.
    leaky = "\\label{sec:pcdleak}" in tex_now
    # What Propositions 1 and 2 establish, and no more (round 3, Reviewer 2,
    # comment 5): a bound on how often the rule predicts under the null.
    if strat and leaky:
        pcd_tail = ("under a permutation null it predicts no more often than "
                    "its nominal level, per participant when stratified, "
                    "even under a leaky partition.")
    elif strat:
        pcd_tail = ("under a permutation null it predicts no more often than "
                    "its nominal level, for each participant when stratified.")
    else:
        pcd_tail = ("under a permutation null it predicts no more often than "
                    "its nominal level.")
    we_propose = "We propose" if leaky else "We also propose"
    c = rd("config_table.csv").set_index("config")
    rf, lb = c.loc["RF, unweighted (headline)"], c.loc["LogReg, scaled, balanced"]
    du = c.loc["Dummy, stratified (control)"]
    jr = rd("weak_effect_jackknife.csv", dtype={"left_out": str})
    low = jr.sort_values("mean10").head(4).mean10
    lad = rd("random_label_ladder.csv")
    free5 = lad[(lad["mode"] == "free") & (lad.rung == 5) & (lad.column == "leaky")].iloc[0]
    fp = rd("fingerprint_table.csv").set_index("quantity")["value"]
    cb = rd("cbramod_diag_summary.csv").set_index("config")
    pe = rd("cbramod_diag_pretraining_effect.csv").set_index("pretrained")
    den = rd("density_sensitivity.csv")
    st = rd("leakage_stack.csv").set_index("rung")
    ov = rd("overlap_sweep.csv")

    # Wording that depends on the corrected p-values follows the result.
    mxs = (lb.within_p_single_maxT, lb.free_p_single_maxT)
    DEP = min([lb[f"{n}_p_{s}_{c}"] for n in ("within", "free")
               for s in ("single", "mean10") for c in ("holm", "maxT")]) < 0.05
    sig_abs = ("an exploratory result whose significance depends on the "
               "correction for that search" if DEP else
               "an exploratory result that is not significant after "
               "correction for that search")
    n_mx = sum(q < 0.05 for q in mxs)
    # sensitivity to the partition (114_partition_sensitivity.py): random
    # participant partitions balanced for class as the splitter's are
    ps_ = rd("partition_sensitivity_stratified_summary.csv").set_index("config")
    plb = ps_.loc["LogReg, scaled, balanced"]
    RP = int(plb.partitions)
    part_above = (f"in all {RP}" if int(plb.above_chance) == RP
                  else f"in {int(plb.above_chance)} of {RP}")
    part_lo, part_hi = plb["min"], plb["max"]
    # abstract: the partition-averaged value beside the splitter's value
    sig_abs = (f"but {plb['mean']:.4f} averaged over {RP} random participant "
               "partitions, an exploratory result"
               + ("" if DEP else " not significant after correction"))
    # the weak-effect paragraph states that the mean fails Holm's correction
    assert min(lb.within_p_mean10_holm, lb.free_p_mean10_holm) >= 0.05
    mx_pro = {2: ", and stays below 0.05 after the max-statistic correction "
                 "at a single fold assignment",
              1: ", and stays below 0.05 after the max-statistic correction "
                 "at a single fold assignment under the "
                 + ("within-participant" if mxs[0] < 0.05 else "unrestricted")
                 + " null",
              0: ""}[n_mx]
    mx_con = ("" if n_mx == 2 else
              ", that the max-statistic correction at a single fold assignment "
              "leaves it at or above 0.05 under "
              + ("either null" if n_mx == 0 else "the other null"))
    # the primary evaluation (120, 151): 20 random participant partitions
    pm = rd("primary_random_summary.csv").set_index("config")
    pns = rd("primary_random_nested_summary.csv").iloc[0]
    RPm = int(pm.partitions.iloc[0])
    p_best = pm.observed_mean.max()
    p_min = float(pm[["within_p", "free_p"]].min().min())
    p_sig = bool((pm[["within_p_holm", "free_p_holm"]] < 0.05).any().any())
    if p_sig or p_min < 0.05 or pns.participant_boot_lo > 1 / 3:
        raise SystemExit("primary evaluation detects an effect: rewrite the "
                         "abstract and Section VII by hand")
    p_lo = f"{int(p_min * 100) / 100:.2f}"          # floor, so "p >= p_lo"
    pw = rd("power_revised.csv")
    pw34 = pw[(pw.cohort == 34) & ((pw.delta - 0.02).abs() < 1e-9)].iloc[0]
    rev = cb.loc["revised"]
    rev_ci = f"{rev.pooled_bal:.4f} [{rev.bal_lo:.4f}, {rev.bal_hi:.4f}]"
    # With three seeds the abstract reports their range, not the first seed,
    # which happens to be the highest.
    seeds_f = os.path.join(OUT, "cbramod_seeds_summary.csv")
    if os.path.exists(seeds_f):
        sd_ = pd.read_csv(seeds_f).set_index("config").loc["revised"]
        fm_abs = f"{sd_.lo:.4f} to {sd_.hi:.4f}"
        fm_seeds = " across three seeds"
    else:
        fm_abs, fm_seeds = f"{rev.pooled_bal:.4f}", ""
    dr = pe.loc["revised"]
    dr_ci = f"{dr['diff']:+.4f} [{dr.lo:+.4f}, {dr.hi:+.4f}]"
    ws_rule = fp["identity rule (CV) window-shuffled bal_mean"]
    ws_rf = fp["forest window-shuffled bal_mean"]

    abstract = (
        "Two studies report 92--94\\% accuracy for classifying dream emotion in the Dream Emotion Evaluation Dataset (DEED) without stating whether validation separated participants. We show that window-level evaluation on DEED can reach that range without information about affect: combining overlapping windows, partitioning over windows rather than participants and oversampling before the split gives "
        f"{st.loc[5, 'leaky_acc']:.4f} accuracy with the reported labels and {free5.acc:.4f} ({free5.acc_real:.4f} on real test windows) with labels randomised across recordings, while participant-separated balanced accuracy stays near chance. This shows susceptibility, not which pipeline the studies used. "
        f"Under participant separation the headline configuration gives {rf.observed_mean10:.4f} balanced accuracy, and in a primary evaluation over {RPm} random participant partitions none of four configurations exceeds its permutation null (highest {p_best:.4f}, $p \\ge {p_lo}$). "
        f"Of the {fp['derived DEED inflation bal (ws - ps)']:.4f} balanced-accuracy gain from window-level partitioning, keeping each recording's windows together removes {fp['derived DEED recording route bal (ws - rg)']:.4f} and separating participants a further {fp['derived DEED participant route bal (rg - ps)']:.4f}. A rule that uses no EEG and knows only the participant reaches a similar balanced accuracy to the window-shuffled classifier ({ws_rule:.4f} against {ws_rf:.4f}) and falls to chance when it cannot use the held-out recording. The pattern reproduces on an independent corpus with a different label. "
        f"Fine-tuned, a transformer pretrained on 9{{,}}000 hours of EEG reaches {fm_abs} balanced accuracy on affect{fm_seeds}, with no detectable benefit from pretraining. "
        f"{we_propose} Permutation-Calibrated Decodability, which localises a permutation test to individual inputs as null-referenced abstention: " + pcd_tail)
    words = len(abstract.replace("--", " ").split())
    assert words <= 250, words

    first = (
        "\\emph{First, the data show no participant-generalizable dream-affect effect under the primary evaluation, and at most a weak, partition-dependent one under the configurations tested.} "
        f"Under the configuration used for the headline analysis, participant-separated balanced accuracy is {rf.observed_mean10:.4f} against a chance level of 0.3333, with $p={rf.within_p_single:.4f}$ and ${rf.free_p_single:.4f}$ at the first fold assignment under the two nulls. "
        f"Under the primary evaluation, chosen because the splitter gives only two partitions and not for its outcome -- {RPm} random class-balanced participant partitions, with folds recomputed for every labelling -- the four candidate configurations average {pm.observed_mean.min():.4f} to {p_best:.4f}, every one with $p \\ge {p_lo}$ under both nulls, and selecting among them inside each training fold gives {pns.bal:.4f}, with a participant interval of {pns.participant_boot_lo:.4f} to {pns.participant_boot_hi:.4f}. "
        f"On the two partitions the splitter produces, the balanced logistic regression reaches {lb.observed_mean10:.4f}. That value is exploratory: it lies above every random partition, "
        + ("its significance depends on the correction for the search that found it, " if DEP else
           "it is not significant after correction for the search that found it, ")
        + f"and it falls to between {low.min():.4f} and {low.max():.4f} when any one of four participants is removed. "
        f"We therefore claim neither that a signal has been established nor that none exists: at 34 participants an effect reaching {pw34.achieved_bal:.4f} balanced accuracy is detected in {pw34.power * 100:.0f}\\% of simulations.")
    o0, o90 = ov[ov.overlap == 0].iloc[0], ov[ov.overlap == 90].iloc[0]
    second = (
        "\\emph{Second, window-level evaluation on DEED can reach the published range without information about affect.} "
        f"Choices that are individually ordinary and not stated in the published method descriptions -- overlapping analysis windows, partitioning over windows rather than participants and oversampling before the split -- raise measured accuracy from {st.loc[0, 'leaky_acc']:.4f} to {st.loc[5, 'leaky_acc']:.4f} with the reported labels, and with labels randomised across recordings the same construction reaches {free5.acc:.4f} counting synthetic test windows and {free5.acc_real:.4f} on real test windows alone, within the published range, while participant-separated balanced accuracy stays near chance. "
        "Window overlap accounts for the largest single step, because at 90\\% overlap most test windows have a near-copy from the same recording in training. "
        "What this shows is the susceptibility of window-level evaluation, not which pipeline the published studies used, about which no claim is made.")
    lr_n = den[(den.model == "lr_nested") & (den.channels == "64 minus 6")].iloc[0]
    const = [int(rd(f"cbramod_diag_revised_{s_}_folds.csv").constant_pred.sum())
             for s_ in ("20260727", "20260728", "20260729")]
    const_txt = (f"{min(const)} to {max(const)}" if min(const) != max(const)
                 else f"{const[0]}")
    # the control with the windows in microvolts and in units of 100 uV, the
    # pretraining convention (49; 144), and the DEED runs in that unit (76)
    cx = {k: rd(f"cbramod_expanded{k}.csv").set_index("model")
          for k in ("", "_uv100")}
    d_cx = {k: v.loc["cbramod", "balanced"] - v.loc["spectral_logreg", "balanced"]
            for k, v in cx.items()}
    # the one fine-tuning run in that unit (145_cbramod_input.py)
    u100 = rd("cbramod_uv100_finetune.csv").set_index("input").loc[
        "units of 100 uV"]
    fourth = (
        "\\emph{Fourth, two instrumental explanations were tested, within limits.} "
        f"A representation pretrained on 9{{,}}000 hours of EEG differs from the spectral baseline on the sleep-staging control by {d_cx['']:+.4f} with the windows in microvolts and by {d_cx['_uv100']:+.4f} with the windows in units of 100\\,$\\mu$V, the convention of its pretraining, on matched data and without retuning, so the control does not show which representation is better. On dream affect, with a revised fine-tuning schedule under which {const_txt} of 34 held-out participants still receive a single predicted class, it reaches {rev_ci} with standardised inputs, with no detectable advantage over the same architecture trained from scratch ({dr_ci}), and {u100.pooled_bal:.4f} [{u100.bal_lo:.4f}, {u100.bal_hi:.4f}] at the first seed with inputs scaled as in pretraining; this concerns CBraMod on this task, not EEG foundation models as a class. "
        f"On the Tononi corpus, raising montage density from 6 to 64 electrodes did not raise participant-separated balanced accuracy for dream recall with any model tried ({lr_n.bal_mean:+.4f}, {lr_n.ci_lo:+.4f} to {lr_n.ci_hi:+.4f}, with nested regularisation), while the window-level advantage widened; the intervals do not exclude modest gains, and the result bears on that task and pipeline, not on the spatial requirements of affect decoding in DEED.")
    weak = (
        "Whether a weak effect exists is not settled, and 34 participants cannot settle it. "
        f"The primary evaluation finds none: over {RPm} random participant partitions no configuration exceeds its permutation null ($p \\ge {p_lo}$ for every configuration under both nulls), and the selection pipeline evaluated in the same way averages {pns.bal:.4f}, with a participant interval of {pns.participant_boot_lo:.4f} to {pns.participant_boot_hi:.4f}. "
        f"On the two partitions the splitter produces, the ten-assignment mean of the balanced logistic regression, {lb.observed_mean10:.4f}, reaches $p={lb.free_p_mean10:.4f}$ against a null that removes every association between EEG and label ({lb.within_p_mean10:.4f} within participants), against {du.free_p_mean10:.4f} and {du.within_p_mean10:.4f} for a stratified dummy classifier, and at the first fold assignment the max-statistic correction gives {lb.within_p_single_maxT:.4f} and {lb.free_p_single_maxT:.4f}; "
        f"but that value was selected after inspecting four configurations and does not survive Holm's correction for the search{mx_con}, both partitions lie above every random one, and removing any one of four participants lowers it to between {low.min():.4f} and {low.max():.4f}. "
        f"What the null can mean is bounded by power: at 34 participants an effect reaching {pw34.achieved_bal:.4f} balanced accuracy is detected in {pw34.power * 100:.0f}\\% of simulations. "
        "A larger cohort with the configuration and the evaluation procedure fixed in advance would resolve in a single experiment what this one can only bound, and its design is fully determined by what is reported here. We regard it as the most useful follow-up available.")

    t = io.open(TEX, encoding="utf-8", newline="").read()
    shutil.copyfile(TEX, TEX + ".pre_summary")
    a = t.index("\\begin{abstract}") + len("\\begin{abstract}\n")
    b = t.index("\\end{abstract}")
    t = t[:a] + abstract + "\n" + t[b:]

    def line_swap(prefixes, new):
        """Replace the line starting with the first prefix present (the
        original wording, or the wording this script wrote on a previous run)."""
        nonlocal t
        import re
        new = re.sub(r"(?<![\w$])([+-]\d\.\d{4})", r"$\1$", new)
        hits = [p for p in prefixes if t.count(p) == 1]
        assert hits, prefixes[0][:60]
        i = t.index(hits[0])
        j = t.index("\n", i)
        t = t[:i] + new + t[j:]

    line_swap(["\\emph{First, the data support at most a weak participant-generalizable dream-affect effect",
               "\\emph{First, the data show no participant-generalizable dream-affect effect"], first)
    line_swap(["\\emph{Second, the reported performance is reconstructible from protocol alone.}",
               "\\emph{Second, window-level evaluation on DEED can reach the published range without information about affect.}"], second)
    line_swap(["\\emph{Fourth, two obvious instrumental explanations are excluded.}",
               "\\emph{Fourth, two instrumental explanations were tested, within limits.}"], fourth)
    line_swap(["Whether the weak effect is real is not settled",
               "Whether a weak effect exists is not settled"], weak)
    for old, new in (
        ("Four findings are supported by direct measurement and, in our view, settled for the corpora examined.",
         "Four findings are supported by direct measurement for the corpora examined."),
        ("A standardised, class-weighted logistic regression reaches 0.3768 on the same data; see Section~\\ref{sec4}.",
         "A standardised, class-weighted logistic regression reaches 0.3768 on the same data, an exploratory result selected among four configurations (Section~\\ref{sec:howmuch}).")):
        if old in t:
            t = t.replace(old, new)
        elif "an exploratory result selected among four configurations" not in new:
            assert new in t, new[:60]
    import re
    # the three quantities, in Section IV-F and in the Conclusion: 0.3768 is
    # named with the value under the primary evaluation beside it
    tq = re.compile(r"(\\textbf\{0\.3768\} is the highest value obtained by any "
                    r"configuration tested)(?:, on the two partitions the splitter "
                    r"produces \([^)]*\))?(, and is the subject of this subsection\.)")
    assert len(tq.findall(t)) == 1
    t = tq.sub(lambda m: f"{m.group(1)}, on the two partitions the splitter "
               f"produces ({p_best:.4f} under the primary evaluation over "
               f"{RPm} random participant partitions, where no configuration "
               f"exceeds its permutation null){m.group(2)}", t)
    cq = re.compile(r"(\\textbf\{0\.3768\}, what the most favourable of the "
                    r"configurations tested finds)(?: on the splitter's partitions "
                    r"\([^)]*\))?;")
    assert len(cq.findall(t)) == 1
    t = cq.sub(lambda m: f"{m.group(1)} on the splitter's partitions "
               f"({p_best:.4f} across random ones, not significant);", t)
    # comparison-table footnote: the value under the primary evaluation
    fn = re.compile(r"(A standardised, class-weighted logistic regression reaches "
                    r"0\.\d{4} on the same data, an exploratory result selected "
                    r"among four configurations)(?:, and 0\.\d{4} [^(]*?)? "
                    r"(\(Section~\\ref\{sec:howmuch\}\)\.)")
    assert len(fn.findall(t)) == 1
    t = fn.sub(lambda m: f"{m.group(1)}, and {p_best:.4f} under the primary "
               f"evaluation over {RPm} random participant partitions, where no "
               f"configuration exceeds its permutation null {m.group(2)}", t)
    # Limitations: what the affect result is
    lim = re.compile(r"the \\emph\{affect result\} -- .*? -- was established on "
                     r"DEED alone\.")
    assert len(lim.findall(t)) == 1
    t = lim.sub(lambda m: "the \\emph{affect result} -- no participant-"
                "generalizable effect under the primary evaluation, and at most "
                "a weak, partition-dependent one under the configurations "
                "tested -- was established on DEED alone.", t)
    io.open(TEX, "w", encoding="utf-8", newline="").write(t)
    print(f"abstract ({words} words), Section VII-A, VII-C weak-effect paragraph, footnote written")


if __name__ == "__main__":
    main()
