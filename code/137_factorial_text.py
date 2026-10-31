"""
The complete protocol factorial in the manuscript (round 3, Reviewer 2,
comments 8 and 9).

Source: 73_protocol_factorial.py --full (2 partitions x 3 selection x 3
oversampling placements = 18 cells, ten fold assignments, 300 trees) and
73b_factorial_effects.py --full (every paired contrast at every level of the
other factors, participant-bootstrap intervals averaged over assignments).

* Table 4(b), the time-limited 12-cell run, is replaced by Table
  tab:factorial with all 18 cells.
* The paragraph on the effects, the SMOTE paragraph (with the implementation
  specification Reviewer 2 asks for), the figure caption, the experiment
  table row and every other sentence quoting the old factorial are rewritten
  from the full results.
* The F-test experiment is described as a sensitivity experiment, not a
  reproduction of the ReliefF / fuzzy-entropy pipeline.
Idempotent.
"""
import io
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
TEX = r"E:\Dream_Analysis\accesstemp.tex"

raw = pd.read_csv(os.path.join(OUT, "protocol_factorial_raw_full.csv"))
eff = pd.read_csv(os.path.join(OUT, "protocol_factorial_effects_full.csv"))
bys = pd.read_csv(os.path.join(OUT, "protocol_factorial_effects_by_seed_full.csv"))
NSEED = raw.seed.nunique()
assert NSEED == 10 and len(raw) == 180, (NSEED, len(raw))
cell = raw.groupby(["partition", "selection", "oversampling"])
# the cell means and SDs the table prints, kept as a result file so that the
# provenance table (132) can key them
_m = cell[["acc_real", "bal_real", "acc_all", "bal_all", "synth_frac_test"]]
pd.concat([_m.mean().add_suffix("_mean"), _m.std().add_suffix("_sd")],
          axis=1).round(4).reset_index().to_csv(
    os.path.join(OUT, "protocol_factorial_cells_full.csv"), index=False)


def C(p, s, o, col):
    return cell[col].mean().loc[(p, s, o)], cell[col].std().loc[(p, s, o)]


def E(effect, a):
    r = eff[(eff.effect == effect) & (eff.a == a)].iloc[0]
    return r.d_bal, r.ci_lo, r.ci_hi


def sg(v):
    return (f"$-${abs(v):.4f}" if v < 0 else f"$+${v:.4f}") if abs(v) >= 5e-5 \
        else (f"$-${abs(v):.5f}" if v < 0 else f"$+${v:.5f}")


def ci(lo, hi):
    f = lambda v: f"$-${abs(v):.4f}" if v < 0 else f"{v:.4f}"
    return f"{f(lo)} to {f(hi)}"


def eci(effect, a):
    d, lo, hi = E(effect, a)
    return f"{sg(d)} ({ci(lo, hi)})"


def seed_range(effect, a):
    v = bys[(bys.effect == effect) & (bys.a == a)].d_bal
    assert len(v) == NSEED
    return f"{v.min():.4f}", f"{v.max():.4f}"


L = ("none", "inside", "before")
part = E("window vs participant partition", "window/none/none")
part_other = [E("window vs participant partition", f"window/{s}/{o}")[0]
              for s in L for o in L if (s, o) != ("none", "none")]
sel = eff[eff.effect == "selection before vs inside"]
sel_all_zero = bool(((sel.ci_lo <= 0) & (sel.ci_hi >= 0)).all())
sel_max = float(sel.d_bal.abs().max())
synth = C("window", "none", "before", "synth_frac_test")[0]
acc_real, _ = C("window", "none", "before", "acc_real")
acc_all, _ = C("window", "none", "before", "acc_all")
max_all = float(cell["acc_all"].mean().max())
pr_lo, pr_hi = seed_range("window vs participant partition", "window/none/none")
ov_lo, ov_hi = seed_range("oversampling before vs inside", "window/none/before")
ovp = E("oversampling before vs inside", "participant/none/before")
ovp2 = E("oversampling before vs inside", "participant/before/before")

# ------------------------------------------------------------------ table
rows = []
NAME = {"none": "none", "inside": "inside the fold", "before": "before the split"}
for s in L:
    for o in L:
        pm, psd = C("participant", s, o, "bal_real")
        wm, wsd = C("window", s, o, "bal_real")
        wa = (f"{C('window', s, o, 'bal_all')[0]:.4f}" if o == "before" else "--")
        rows.append(f"{NAME[s]} & {NAME[o]} & {pm:.4f} ({psd:.4f}) & "
                    f"{wm:.4f} ({wsd:.4f}) & {wa} \\\\")
TABLE = (
    "\\begin{table*}[htbp]\n\\centering\n"
    "\\caption{The complete protocol factorial on DEED (dream valence, 34 "
    "participants, 6{,}144 windows): partition $\\times$ placement of feature "
    "selection ($F$-test, 40 of 108 features) $\\times$ placement of SMOTE "
    "oversampling, random forest fixed (300 trees). Balanced accuracy on the "
    "untouched real test windows, mean (SD) over ten fold assignments; the "
    "last column also counts the synthetic points that pre-split oversampling "
    "places in the test folds. Paired effects of every factor at every level "
    "of the others are in the supplementary material.}\n"
    "\\label{tab:factorial}\n\\small\n"
    "\\begin{tabular}{@{}llccc@{}}\n\\toprule\n"
    "& & Participant-separated & \\multicolumn{2}{c}{Window-shuffled} \\\\\n"
    "\\cmidrule(lr){3-3}\\cmidrule(lr){4-5}\n"
    "Feature selection & Oversampling & real windows & real windows & all "
    "test points \\\\\n\\midrule\n" + "\n".join(rows) + "\n"
    "\\botrule\n\\end{tabular}\n\\normalsize\n\\end{table*}\n")

PARA = (
    "Table~\\ref{tab:factorial} and Figure~\\ref{fig:ladder} separate the "
    "choices. The classifier is fixed (random forest, 300 trees) and the three "
    "factors -- the partition, where feature selection sits relative to the "
    "split, and where oversampling sits -- are fully crossed, giving 18 "
    f"configurations, each evaluated under {NSEED} fold assignments and scored "
    "on the untouched real test windows, so synthetic points never count "
    "towards a score. Each effect is a paired difference with the other two "
    "factors held fixed; every contrast at every level of the other factors is "
    "in the supplementary material. Partitioning over windows rather than "
    "participants raises balanced accuracy by "
    f"{eci('window vs participant partition', 'window/none/none')} with "
    "neither selection nor oversampling, and by between "
    f"{min(part_other):.4f} and {max(part_other):.4f} at the other eight "
    "combinations. Moving oversampling from inside the fold to before the "
    "split adds "
    f"{eci('oversampling before vs inside', 'window/none/before')} when "
    "windows are shuffled. When participants are separated it adds "
    f"{eci('oversampling before vs inside', 'participant/none/before')}, and "
    f"{eci('oversampling before vs inside', 'participant/before/before')} "
    "when feature selection is also fitted before the split: small, but "
    + ("not shown to be zero" if ovp2[1] > 0 else "not excluded") +
    ", so a pipeline that oversamples before the split is not leakage-free "
    "under participant separation merely because its effect there is small. "
    "Moving feature selection before the split changes balanced accuracy by at "
    f"most {sel_max:.4f} in either partition"
    + (", every interval including zero" if sel_all_zero else "")
    + "; selecting 40 of 108 features at all lowers balanced accuracy when "
    "windows are shuffled, by "
    f"{eci('selection inside vs none', 'window/inside/none').replace('$-$', '', 1)}"
    " inside the fold. Comparing a participant-separated row with a "
    "window-shuffled one, or reporting the best of several classifiers for "
    "each row, would attribute to feature selection what belongs to the "
    "partition; both confounds are removed here. The intervals resample "
    "participants with the predictions held fixed and average over the "
    "assignments, so they omit variation between assignments, which is "
    "reported separately: across assignments the partition effect ranges from "
    f"{pr_lo} to {pr_hi}, and the effect of pre-split oversampling with "
    f"shuffled windows from {ov_lo} to {ov_hi}.")

SMOTE = (
    "How the oversampling is implemented matters for what these numbers mean. "
    "SMOTE \\cite{chawla2002smote} creates each synthetic point between a "
    "window and one of its five nearest neighbours of the same class, "
    "whichever participants they belong to. Applied before the split, every "
    "synthetic point is assigned the participant of its nearest real window "
    "in the feature space SMOTE used, the folds are then formed on real and "
    "synthetic points together, and a synthetic point carries no recording, "
    "since recordings are not a grouping unit here. A synthetic point in a "
    "training fold can therefore have one or both parents among that fold's "
    "test windows, which is the leak being measured. Every real window is "
    "scored exactly once in every configuration, so all configurations are "
    "compared on the same real test windows, although the fold a given real "
    "window falls in can differ between configurations. "
    "Pre-split oversampling also places synthetic points in the test folds: "
    f"they make up {100 * synth:.1f}\\% of test points here. Scored on all "
    f"test points, the window-shuffled configuration gives {acc_all:.4f} "
    f"accuracy; scored on the real windows alone, {acc_real:.4f}. ")


def main():
    s = io.open(TEX, encoding="utf-8", newline="").read()
    # 1. Table 4: drop the old panel (b); insert the complete factorial
    lab = s.index("\\label{tab:protocol}")
    b0 = s.find("\\midrule\n\\multicolumn{3}{@{}l}{\\textit{(b) Pipeline ordering", lab)
    if b0 > 0:
        b1 = s.index("\\botrule", b0)
        s = s[:b0] + s[b1:]
    s = s.replace("\\caption{Effect of validation protocol and pipeline ordering "
                  "on measured performance in DEED",
                  "\\caption{Effect of validation protocol on measured "
                  "performance in DEED")
    if "\\label{tab:factorial}" in s:
        a = s.rindex("\\begin{table*}", 0, s.index("\\label{tab:factorial}"))
        if s[a - 2:a] == "\n\n":        # the newline inserted with the table
            a -= 1
        e = s.index("\\end{table*}", a) + len("\\end{table*}\n")
        s = s[:a] + s[e:]
    end = s.index("\\end{table*}", lab) + len("\\end{table*}\n")
    s = s[:end] + "\n" + TABLE + s[end:]

    # 2. the effects paragraph, replaced whole
    for start in ("Panel~(b) of Table~\\ref{tab:protocol} and Figure~"
                  "\\ref{fig:ladder} separate the choices.",
                  "Table~\\ref{tab:factorial} and Figure~\\ref{fig:ladder} "
                  "separate the choices."):
        if start in s:
            i = s.index(start)
            j = s.index("\n", i)
            s = s[:i] + PARA + s[j:]
            break
    else:
        raise SystemExit("effects paragraph not found")

    # 3. the oversampling paragraph: specification + synthetic share
    for start in ("When synthetic samples are generated before the split, "
                  "points interpolated from test observations enter the "
                  "training set, and synthetic points also enter the test "
                  "folds:", "How the oversampling is implemented matters"):
        if start in s:
            i = s.index(start)
            j = s.index("Scoring synthetic points also balances the test "
                        "classes", i)
            s = s[:i] + SMOTE + s[j:]
            break
    else:
        raise SystemExit("oversampling paragraph not found")

    # 4. sentences elsewhere
    sel_d, sel_lo, sel_hi = E("selection before vs inside", "window/before/none")
    edits = [
        ("\\caption{\\textbf{The effect of each protocol choice with the others "
         "held fixed.} Change in balanced accuracy on untouched real test "
         "windows, random forest fixed, mean of three fold assignments, with "
         "95\\% participant-bootstrap intervals that hold the fold assignments "
         "fixed; crosses mark the three assignments individually.",
         "\\caption{\\textbf{The effect of each protocol choice with the others "
         "held fixed.} Change in balanced accuracy on untouched real test "
         f"windows, random forest fixed (300 trees), mean of {NSEED} fold "
         "assignments, with 95\\% participant-bootstrap intervals that hold the "
         "predictions fixed; crosses mark the assignments individually."),
        ("Only two choices matter: the partition, and oversampling before the "
         "split when windows are shuffled. Feature-selection placement has no "
         "measurable effect in either partition.}",
         "The partition and, with shuffled windows, oversampling before the "
         "split have large effects; oversampling before the split under "
         "participant separation has a small one (text); feature-selection "
         "placement has none that is detectable.}"),
        ("to at most 0.7169 on all test points,",
         f"to at most {max_all:.4f} on all test points,"),
        ("adds 0.1145 in balanced accuracy on real test windows "
         "(Table~\\ref{tab:protocol}b).",
         f"adds {E('oversampling before vs inside', 'window/none/before')[0]:.4f}"
         " in balanced accuracy on real test windows "
         "(Table~\\ref{tab:factorial})."),
        ("as noted for Table~\\ref{tab:protocol}(b), the coincidence",
         "as noted for Table~\\ref{tab:factorial}, the coincidence"),
        ("for the protocol factorial of Table~\\ref{tab:protocol}(b),",
         "for the protocol factorial of Table~\\ref{tab:factorial},"),
        ("With the classifier and the partition held fixed, and an $F$-test "
         "selector standing in for ReliefF, moving selection from within each "
         "fold to the complete dataset changes balanced accuracy by "
         "$-0.00001$ under window-level partitioning ($+0.0112$, $-0.0087$ and "
         "$-0.0025$ in the three fold assignments) "
         "(Table~\\ref{tab:protocol}b);",
         "With the classifier and the partition held fixed, and an $F$-test "
         "selector standing in for ReliefF, moving selection from within each "
         "fold to the complete dataset changes balanced accuracy by "
         f"{sg(sel_d)} ({ci(sel_lo, sel_hi)}) under window-level partitioning "
         "(Table~\\ref{tab:factorial}). This is a sensitivity experiment, not a "
         "reproduction of their ReliefF and fuzzy-entropy pipeline;"),
        ("Pipeline ordering (Table~\\ref{tab:protocol}b) & as above & 108; 40 "
         "by $F$-test when selecting & RF, 100 trees & window-shuffled and "
         "part.-sep. $\\times$ selection and SMOTE placement; 3 assignments; "
         "scored on real windows",
         "Pipeline ordering (Table~\\ref{tab:factorial}) & as above & 108; 40 "
         "by $F$-test when selecting & RF, 300 trees & window-shuffled and "
         "part.-sep. $\\times$ selection $\\times$ SMOTE placement, all 18 "
         f"cells; {NSEED} assignments; scored on real windows"),
    ]
    for old, new in edits:
        if old in s:
            s = s.replace(old, new)
        assert new in s, f"not applied: {new[:80]}"
    assert "tab:protocol}b" not in s and "tab:protocol}(b)" not in s
    io.open(TEX, "w", encoding="utf-8", newline="").write(s)
    print(f"factorial written: partition {part[0]:+.4f}, oversampling "
          f"(windows) {E('oversampling before vs inside', 'window/none/before')[0]:+.4f}, "
          f"(participants) {ovp[0]:+.4f} / with selection {ovp2[0]:+.4f}; "
          f"selection max |d| {sel_max:.4f}, all CIs include 0: {sel_all_zero}")


if __name__ == "__main__":
    main()
