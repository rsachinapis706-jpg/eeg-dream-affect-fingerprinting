"""
PCD prose after the label-free rerun (round 3, Reviewer 2, comment 1).

Every PCD analysis was re-run with folds drawn without labels
(77.label_free_folds). The tables and the stratified-reference paragraph are
rebuilt by 111, 115 and 117; this script updates the remaining PCD sentences
in place, each located by its wording, with every number read from the new
result files. Statements that depend on the numbers (where an interval
excludes zero, whether a threshold predicts on any window) are derived from
them rather than kept. 99_write_pcd.py is not used: it regenerates the whole
section and would discard the round-3 text.

Each pattern must match exactly once; the script stops otherwise. Idempotent.
"""
import io
import re

import numpy as np
import pandas as pd

OUT = r"E:\Dream GAN\analysis\out"
TEX = r"E:\Dream_Analysis\accesstemp.tex"
N = r"[-+]?\d\.\d{4}"          # a printed 4-decimal number
SE, DE = "Sleep-EDF staging", "DEED affect"


def rd(n):
    return pd.read_csv(f"{OUT}\\{n}")


gr = rd("pcd_extra_grid.csv")
sm = rd("pcd_revised_summary.csv")
cbx = rd("pcd_extra_cbramod.csv")
dn = rd("pcd_revised_deednull.csv")
si = rd("pcd_revised_sims.csv")
ci = rd("pcd_riskcov_ci.csv")
nz = rd("pcd_extra_noise.csv").sort_values("eta")
nd = rd("pcd_extra_nulldep.csv")


def g(cond, rule, col="coverage"):
    return float(gr[(gr.condition == cond) & (gr.rule == rule)].iloc[0][col])


def gci(cond, rule, col):
    r = gr[(gr.condition == cond) & (gr.rule == rule)].iloc[0]
    lo, hi = ("cov_lo", "cov_hi") if col == "coverage" else ("bal_lo", "bal_hi")
    return f"{r[col]:.4f} [{r[lo]:.4f}, {r[hi]:.4f}]"


def s(cond, rule, col="coverage"):
    return float(sm[(sm.condition == cond) & (sm.rule == rule)].iloc[0][col])


def f4(x):
    return f"{x:.4f}"


def runs(cov):
    """'0.25 to 0.65 and at 0.80 and 0.85' for a sorted list of grid points."""
    cov = sorted(cov)
    parts, start = [], cov[0]
    for a, b in zip(cov, cov[1:] + [None]):
        if b is None or not np.isclose(b - a, 0.05):
            parts.append((start, a))
            start = b
    txt = [f"{a:.2f} to {b:.2f}" if a != b else f"{a:.2f}" for a, b in parts]
    return txt[0] if len(txt) == 1 else ", ".join(txt[:-1]) + " and " + txt[-1]


# ------------------------------------------------------------ derived values
cS = ci[(ci.condition == SE) & (ci.coverage < 1)]
cD = ci[(ci.condition == DE) & (ci.coverage < 1)]
iS = cS.pcd_minus_msp.idxmax()
iE = cS.pcd_minus_entropy.idxmax()
excl_S = cS[cS.msp_lo > 0].coverage.tolist()
excl_D_msp = cD[(cD.msp_lo > 0) | (cD.msp_hi < 0)]
excl_D_ent = cD[(cD.ent_lo > 0) | (cD.ent_hi < 0)]
pcd_cov_S = g(SE, "PCD alpha=0.05")
at = ci[ci.condition == SE].iloc[
    (ci[ci.condition == SE].coverage - pcd_cov_S).abs().argmin()]
assert len(excl_D_msp) == 0, "DEED max-softmax interval now excludes zero"

if len(excl_D_ent) == 0:
    deed_ent = "as does the interval for the difference from entropy"
else:
    r = excl_D_ent.iloc[0]
    deed_ent = (f"against entropy the interval excludes zero at "
                f"{len(excl_D_ent)} of {len(cD)} coverage levels")

n0, n1 = nz.iloc[0], nz.iloc[-1]
n8 = nz[np.isclose(nz.eta, 0.8)].iloc[0]
assert n0.eta == 0 and n1.eta == 1
sp = nz[["balanced", "coverage"]].corr("spearman").iloc[0, 1]
pe = nz[["balanced", "coverage"]].corr().iloc[0, 1]
mono = bool(np.all(np.diff(nz.coverage) <= 0))
msp1 = ("on none of the Sleep-EDF windows" if n1.msp_coverage == 0 else
        f"on {f4(n1.msp_coverage)} of Sleep-EDF windows")


def rng(knob, col):
    k = nd[nd.knob == knob]
    return f4(k[col].min()), f4(k[col].max())


# (pattern, replacement); pattern must match exactly once
EDITS = [
    # IV-K, first paragraph
    (rf"On Sleep-EDF staging PCD at \$\\alpha=0\.05\$ predicts on {N} \[{N}, {N}\] "
     rf"of windows, with retained balanced accuracy {N} \[{N}, {N}\] against {N} "
     r"for all windows\.",
     f"On Sleep-EDF staging PCD at $\\alpha=0.05$ predicts on "
     f"{gci(SE, 'PCD alpha=0.05', 'coverage')} of windows, with retained "
     f"balanced accuracy {gci(SE, 'PCD alpha=0.05', 'bal_retained')} against "
     f"{f4(g(SE, 'all inputs', 'bal_retained'))} for all windows."),
    (rf"On DEED affect it predicts on {N} \[{N}, {N}\] of windows, whose balanced "
     rf"accuracy, {N}(?: \[{N}, {N}\])?, is no higher than that of all "
     rf"windows, {N}; on frozen "
     r"CBraMod embeddings of the same task, which were never fine-tuned on these "
     rf"labels, it predicts on {N}\. A max-softmax threshold of 0\.5 predicts on "
     rf"{N} and {N} of windows in the two tasks, and on {N} with the embeddings\.",
     f"On DEED affect it predicts on {gci(DE, 'PCD alpha=0.05', 'coverage')} of "
     f"windows, whose balanced accuracy, "
     f"{gci(DE, 'PCD alpha=0.05', 'bal_retained')}, is no higher than that of "
     f"all windows, {f4(g(DE, 'all inputs', 'bal_retained'))}; on frozen "
     "CBraMod embeddings of the same task, which were never fine-tuned on these "
     "labels, it predicts on "
     f"{f4(float(cbx[cbx.rule == 'PCD p<=0.05'].coverage.iloc[0]))}. A "
     "max-softmax threshold of 0.5 predicts on "
     f"{f4(g(SE, 'Max-softmax>0.5'))} and {f4(g(DE, 'Max-softmax>0.5'))} of "
     "windows in the two tasks, and on "
     f"{f4(float(cbx[cbx.rule == 'Max-softmax>0.5'].coverage.iloc[0]))} with "
     "the embeddings."),
    # IV-K, calibration paragraph
    (rf"PCD predicts on {N} of windows on average \(range {N} to {N}\), in "
     r"agreement",
     f"PCD predicts on {f4(dn.fpr.mean())} of windows on average (range "
     f"{f4(dn.fpr.min())} to {f4(dn.fpr.max())}), in agreement"),
    (rf"the worst participant is flagged on {N} of their windows, and on average "
     rf"{N} of participants exceed",
     f"the worst participant is flagged on {f4(dn.part_fpr_max.median())} of "
     f"their windows, and on average {f4(dn.part_frac_above_2alpha.mean())} of "
     "participants exceed"),
    (rf"the marginal rate is {N} \(2\.5th to 97\.5th percentile {N} to {N}\), the "
     rf"90th percentile of participant-level rates has a median of {N}, and the "
     r"Kolmogorov distance between the null \$p\$-values and the uniform "
     rf"distribution has a median of {N}\.",
     f"the marginal rate is {f4(si.fpr.mean())} (2.5th to 97.5th percentile "
     f"{f4(si.fpr.quantile(.025))} to {f4(si.fpr.quantile(.975))}), the 90th "
     "percentile of participant-level rates has a median of "
     f"{f4(si.part_fpr_p90.median())}, and the Kolmogorov distance between the "
     "null $p$-values and the uniform distribution has a median of "
     f"{f4(si.p_uniform_ks.median())}."),
    # IV-K-1, risk--coverage comparison
    (r"PCD's retained balanced accuracy exceeds that of max-softmax by up to .*?"
     r"Intervals resample participants on the first fold assignment\.",
     "PCD's retained balanced accuracy exceeds that of max-softmax by up to "
     f"{f4(cS.loc[iS, 'pcd_minus_msp'])} (95\\% participant-bootstrap interval "
     f"{f4(cS.loc[iS, 'msp_lo'])} to {f4(cS.loc[iS, 'msp_hi'])}, at coverage "
     f"{cS.loc[iS, 'coverage']:.2f}), the interval excluding zero at coverages "
     f"{runs(excl_S)}, and that of entropy by up to "
     f"{f4(cS.loc[iE, 'pcd_minus_entropy'])} ({f4(cS.loc[iE, 'ent_lo'])} to "
     f"{f4(cS.loc[iE, 'ent_hi'])}); at the coverage PCD reaches at "
     f"$\\alpha=0.05$ the three are close ({f4(at.pcd)}, {f4(at.msp)} and "
     f"{f4(at.entropy)} at coverage {at.coverage:.2f}). On DEED the difference "
     f"from max-softmax is no larger than {f4(cD.pcd_minus_msp.abs().max())} in "
     f"either direction and its interval includes zero at every coverage, "
     f"{deed_ent}. Intervals resample participants on the first fold "
     "assignment."),
    (rf"on DEED max-softmax \$>0\.5\$ predicts on {N} of windows and entropy "
     rf"\$<0\.9\$ on {N}, against {N} for PCD\. A stricter confidence threshold "
     r"can (?:match|come close to) PCD on DEED -- max-softmax \$>0\.7\$ predicts "
     rf"on {N} -- but then predicts on only {N} of Sleep-EDF windows, against "
     rf"{N} for PCD",
     f"on DEED max-softmax $>0.5$ predicts on {f4(s(DE, 'Max-softmax>0.5'))} of "
     f"windows and entropy $<0.9$ on {f4(s(DE, 'Entropy<0.9'))}, against "
     f"{f4(s(DE, 'PCD p<=0.05'))} for PCD. A stricter confidence threshold can "
     f"come close to PCD on DEED -- max-softmax $>0.7$ predicts on "
     f"{f4(g(DE, 'Max-softmax>0.7'))} -- but then predicts on only "
     f"{f4(g(SE, 'Max-softmax>0.7'))} of Sleep-EDF windows, against "
     f"{f4(g(SE, 'PCD alpha=0.05'))} for PCD"),
    # IV-L, dependence on the reference
    (rf"The mean null margin changes from {N} to {N} as the feature count goes "
     rf"from 9 to 36, (?:from|lies between) {N} (?:to|and) {N} across training "
     rf"cohorts of 10 to 40 participants, and (?:from|between) {N} (?:to|and) "
     rf"{N} across regularisation strengths\. "
     rf"Coverage at \$\\alpha=0\.05\$ ranges from {N} to {N} across feature "
     rf"counts, {N} to {N} across cohort sizes and {N} to {N} across "
     rf"regularisation strengths, while balanced accuracy ranges from {N} to {N} "
     rf"and {N} to {N} across the first two\.",
     "The mean null margin changes from {} to {} as the feature count goes from "
     "9 to 36, lies between {} and {} across training cohorts of 10 to 40 "
     "participants, and between {} and {} across regularisation strengths. "
     "Coverage at "
     "$\\alpha=0.05$ ranges from {} to {} across feature counts, {} to {} across "
     "cohort sizes and {} to {} across regularisation strengths, while balanced "
     "accuracy ranges from {} to {} and {} to {} across the first two.".format(
         *rng("features", "null_mean_margin"),
         *rng("participants", "null_mean_margin"),
         *rng("capacity", "null_mean_margin"),
         *rng("features", "coverage"), *rng("participants", "coverage"),
         *rng("capacity", "coverage"), *rng("features", "balanced"),
         *rng("participants", "balanced"))),
    # label-noise sweep
    (rf"Coverage declines(?: monotonically)? with decodability, from {N} at "
     rf"\$\\eta=0\$ to {N} at \$\\eta=1\$, with Spearman correlation {N} and "
     rf"Pearson correlation {N} against the balanced accuracy achieved, which "
     rf"falls from {N} to {N}\. At \$\\eta=1\$ the labels are independent of the "
     r"EEG and exchangeable, so Proposition~\\ref\{prop:calib\} applies, and "
     rf"coverage is {N} against the nominal 0\.05\. PCD coverage stays well above "
     rf"the nominal level while any signal remains -- {N} at \$\\eta=0\.8\$, "
     rf"where balanced accuracy is {N} -- and falls to it only when none remains\. "
     rf"A max-softmax threshold of 0\.5 behaves differently: it falls from {N} at "
     rf"\$\\eta=0\$ to {N} at \$\\eta=0\.8\$",
     f"Coverage {'declines monotonically' if mono else 'declines'} with "
     f"decodability, from {f4(n0.coverage)} at $\\eta=0$ to {f4(n1.coverage)} "
     f"at $\\eta=1$, with Spearman correlation {f4(sp)} and Pearson correlation "
     f"{f4(pe)} against the balanced accuracy achieved, which falls from "
     f"{f4(n0.balanced)} to {f4(n1.balanced)}. At $\\eta=1$ the labels are "
     "independent of the EEG and exchangeable, so "
     "Proposition~\\ref{prop:calib} applies, and coverage is "
     f"{f4(n1.coverage)} against the nominal 0.05. PCD coverage stays well "
     f"above the nominal level while any signal remains -- {f4(n8.coverage)} "
     f"at $\\eta=0.8$, where balanced accuracy is {f4(n8.balanced)} -- and "
     "falls to it only when none remains. A max-softmax threshold of 0.5 "
     f"behaves differently: it falls from {f4(n0.msp_coverage)} at $\\eta=0$ "
     f"to {f4(n8.msp_coverage)} at $\\eta=0.8$"),
    (rf"PCD at \$\\alpha=0\.05\$ falls from {N} to {N}; max-softmax \$>0\.5\$ from "
     rf"{N} to {N}\. Dotted line",
     f"PCD at $\\alpha=0.05$ falls from {f4(n0.coverage)} to {f4(n1.coverage)}; "
     f"max-softmax $>0.5$ from {f4(n0.msp_coverage)} to "
     f"{f4(n1.msp_coverage)}. Dotted line"),
    # Section III-D, what each method can express
    (rf"a threshold of 0\.5 predicts on {N} of DEED windows, on (?:none of the "
     rf"Sleep-EDF windows|{N} of Sleep-EDF windows) once their labels are fully "
     rf"randomised, and on {N} at a noise level",
     f"a threshold of 0.5 predicts on {f4(g(DE, 'Max-softmax>0.5'))} of DEED "
     f"windows, {msp1} once their labels are fully randomised, and on "
     f"{f4(n8.msp_coverage)} at a noise level"),
    # Section VII
    (rf"Empirically, PCD at \$\\alpha=0\.05\$ predicts on {N} of Sleep-EDF "
     rf"windows and on {N} of DEED windows, where a max-softmax threshold of 0\.5 "
     rf"predicts on {N}\. Its false-positive rate is close to the nominal level "
     rf"when the null holds by construction, on re-randomised DEED labels "
     rf"\({N}\) and on simulated data \({N}\)",
     f"Empirically, PCD at $\\alpha=0.05$ predicts on "
     f"{f4(g(SE, 'PCD alpha=0.05'))} of Sleep-EDF windows and on "
     f"{f4(g(DE, 'PCD alpha=0.05'))} of DEED windows, where a max-softmax "
     f"threshold of 0.5 predicts on {f4(g(DE, 'Max-softmax>0.5'))}. Its "
     "false-positive rate is close to the nominal level when the null holds by "
     f"construction, on re-randomised DEED labels ({f4(dn.fpr.mean())}) and on "
     f"simulated data ({f4(si.fpr.mean())})"),
    (rf"under controlled label corruption its coverage falls from {N} to {N} as "
     r"the labels become random",
     f"under controlled label corruption its coverage falls from "
     f"{f4(n0.coverage)} to {f4(n1.coverage)} as the labels become random"),
]


def main():
    t = io.open(TEX, encoding="utf-8", newline="").read()
    for pat, new in EDITS:
        hits = re.findall(pat, t, flags=re.S)
        assert len(hits) == 1, (len(hits), pat[:90])
        t = re.sub(pat, lambda _m: new, t, flags=re.S)
    io.open(TEX, "w", encoding="utf-8", newline="").write(t)
    print(f"{len(EDITS)} PCD passages updated")
    print(f"  Sleep-EDF coverage {f4(pcd_cov_S)}; DEED "
          f"{f4(g(DE, 'PCD alpha=0.05'))}, retained "
          f"{f4(g(DE, 'PCD alpha=0.05', 'bal_retained'))} vs all "
          f"{f4(g(DE, 'all inputs', 'bal_retained'))}")
    print(f"  max-softmax interval excludes zero at {runs(excl_S)}; {deed_ent}")


if __name__ == "__main__":
    main()
