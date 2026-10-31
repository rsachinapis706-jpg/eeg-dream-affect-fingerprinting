"""
One table for PCD and the established ways of declining to predict, replacing
the two tables that overlapped (Table "pcd": PCD and max-softmax thresholds
with intervals; Table "selective": entropy and split conformal without).

Every row comes from out/pcd_revised_windows.parquet (77_pcd_revised.py) with
the participant bootstrap of 85_pcd_extra.py: the same resampling (seed
20261001, conditions in the same order, 2,000 resamples), so the intervals
already reported for PCD and max-softmax are reproduced exactly (checked), and
entropy and split conformal get intervals from the same resamples. Split
conformal coverage is the share of single-class prediction sets; its mean set
size and coverage of the true label come from pcd_revised_summary.csv.

Outputs: out/pcd_table_merged.csv; the merged table written into the
manuscript in place of the two, with the references updated. Re-runnable.
"""
import importlib.util
import io
import os
import re
import shutil

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "extra", os.path.join(HERE, "85_pcd_extra.py"))
extra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extra)
OUT = extra.OUT
TEX = os.path.join("E:" + os.sep, "Dream_Analysis", "accesstemp.tex")

ORDER = ["PCD alpha=0.1", "PCD alpha=0.05", "PCD alpha=0.01",
         "Max-softmax>0.5", "Max-softmax>0.7", "Max-softmax>0.9",
         "Entropy<0.9", "Split conformal", "all inputs"]
LABEL = {"PCD alpha=0.1": r"PCD, $\alpha=0.10$",
         "PCD alpha=0.05": r"PCD, $\alpha=0.05$",
         "PCD alpha=0.01": r"PCD, $\alpha=0.01$",
         "Max-softmax>0.5": r"Max-softmax $>0.5$",
         "Max-softmax>0.7": r"Max-softmax $>0.7$",
         "Max-softmax>0.9": r"Max-softmax $>0.9$",
         "Entropy<0.9": r"Normalised entropy $<0.9$",
         "Split conformal": r"Split conformal, $\alpha=0.05$\footnotemark[1]",
         "all inputs": "All inputs"}


def compute():
    w = pd.read_parquet(os.path.join(OUT, "pcd_revised_windows.parquet"))
    rng = np.random.default_rng(20261001)          # as in 85_pcd_extra.grid
    rows = []
    for cond, g in w.groupby("condition"):          # same order as 85
        rules = [(f"PCD alpha={a}", (g.p_pcd <= a).to_numpy())
                 for a in (0.10, 0.05, 0.01)]
        rules += [(f"Max-softmax>{t}", (g.msp > t).to_numpy())
                  for t in (0.5, 0.7, 0.9)]
        rules += [("all inputs", np.ones(len(g), bool))]
        rules += [("Entropy<0.9", (g.ent < 0.9).to_numpy()),
                  ("Split conformal", (g.conf_size == 1).to_numpy())]
        for r in extra.boot_rules(g.reset_index(drop=True), rules, rng):
            rows.append(dict(condition=cond, participants=g.subj.nunique(),
                             windows=len(g), **r))
    t = pd.DataFrame(rows)
    t.loc[t.rule == "Split conformal", ["bal_retained", "bal_lo", "bal_hi"]] = np.nan
    # the intervals already reported must be reproduced exactly
    old = pd.read_csv(os.path.join(OUT, "pcd_extra_grid.csv"))
    m = t.merge(old, on=["condition", "rule"], suffixes=("", "_old"))
    assert len(m) == 14, len(m)
    for c in ("coverage", "cov_lo", "cov_hi", "bal_retained", "bal_lo", "bal_hi"):
        assert (m[c].round(4) == m[c + "_old"].round(4)).all(), c
    # entropy and conformal point values must match pcd_revised_summary.csv
    sm = pd.read_csv(os.path.join(OUT, "pcd_revised_summary.csv"))
    for cond in t.condition.unique():
        e = sm[(sm.condition == cond) & (sm.rule == "Entropy<0.9")].iloc[0]
        r = t[(t.condition == cond) & (t.rule == "Entropy<0.9")].iloc[0]
        assert round(e.coverage, 4) == round(r.coverage, 4)
        assert round(e.bal_retained, 4) == round(r.bal_retained, 4)
        c = sm[(sm.condition == cond) & sm.rule.str.startswith("Split conformal")].iloc[0]
        r = t[(t.condition == cond) & (t.rule == "Split conformal")].iloc[0]
        assert round(c.coverage, 4) == round(r.coverage, 4)
    t.round(4).to_csv(os.path.join(OUT, "pcd_table_merged.csv"), index=False)
    return t, sm


def fmt(v, lo, hi):
    return "--" if pd.isna(v) else f"{v:.4f} [{lo:.4f}, {hi:.4f}]"


def main():
    t, sm = compute()
    s = io.open(TEX, encoding="utf-8", newline="").read()
    shutil.copyfile(TEX, TEX + ".pre_pcdmerge")

    def table_span(label):
        i = s.index("\\label{" + label + "}")
        a = s.rindex("\\begin{table", 0, i)
        b = s.index("\\end{table", i)
        b = s.index("}", b) + 1
        return a, b

    a, b = table_span("tab:pcd")
    old_tab = s[a:b]
    heads = re.findall(r"\\textit\{((?:[^{}]|\{[^{}]*\})*)\}", old_tab)
    head = {"Sleep-EDF staging": next(h for h in heads if h.startswith("Sleep-EDF")),
            "DEED affect": next(h for h in heads if h.startswith("DEED"))}
    blocks = []
    for cond in ("Sleep-EDF staging", "DEED affect"):
        g = t[t.condition == cond].set_index("rule")
        rows = [f"\\multicolumn{{3}}{{@{{}}l}}{{\\textit{{{head[cond]}}}}} \\\\"]
        for r in ORDER:
            x = g.loc[r]
            rows.append(f"{LABEL[r]} & {fmt(x.coverage, x.cov_lo, x.cov_hi)} & "
                        f"{fmt(x.bal_retained, x.bal_lo, x.bal_hi)} \\\\")
        blocks.append("\n".join(rows))
    cs = sm[sm.rule.str.startswith("Split conformal")].set_index("condition")
    new_tab = "\n".join([
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{PCD against established ways of declining to predict, on the "
        "same participant-separated folds and data; in each fold all rules share "
        "one decoder, fitted on three quarters of the training participants. "
        "Coverage is the "
        "share of windows on which the rule predicts; retained balanced accuracy "
        "is computed on those windows. 95\\% participant-bootstrap intervals, "
        "2{,}000 resamples.}",
        "\\label{tab:pcd}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4.5pt}",
        "\\begin{tabular}{@{}lcc@{}}",
        "\\toprule",
        "Rule & Coverage & Retained bal.\\ acc. \\\\",
        "\\midrule",
        blocks[0],
        "\\midrule",
        blocks[1],
        "\\botrule",
        "\\end{tabular}",
        "\\footnotetext[1]{Coverage of split conformal prediction is the share of "
        "single-class prediction sets. Its mean set size is "
        f"{cs.loc['Sleep-EDF staging', 'set_size']:.2f} on Sleep-EDF and "
        f"{cs.loc['DEED affect', 'set_size']:.2f} on DEED, and its coverage of the "
        f"true label {cs.loc['Sleep-EDF staging', 'set_coverage']:.4f} and "
        f"{cs.loc['DEED affect', 'set_coverage']:.4f}; it is the window-level "
        "form, an empirical comparator (Section~\\ref{sec:pcdres}). Its calibration "
        "uses the remaining quarter of the training participants, excluded from "
        "fitting and disjoint from the test participants; PCD's null decoders are "
        "fitted on the same three quarters as the shared decoder.}",
        "\\normalsize",
        "\\end{table}"])
    s = s[:a] + new_tab + s[b:]
    if "\\label{tab:selective}" in s:                  # remove the second table
        a2, b2 = table_span("tab:selective")
        s = s[:a2].rstrip("\n") + "\n" + s[b2:].lstrip("\n")
    for old, new in (
            ("PCD (Tables~\\ref{tab:selective}, \\ref{tab:pcd};", "PCD (Table~\\ref{tab:pcd};"),
            ("Table~\\ref{tab:selective} adds predictive entropy and split conformal prediction",
             "Table~\\ref{tab:pcd} also gives predictive entropy and split conformal prediction")):
        s = s.replace(old, new)
    assert "tab:selective" not in s, "a reference to the removed table remains"
    io.open(TEX, "w", encoding="utf-8", newline="").write(s)
    print("merged PCD table written; existing intervals reproduced exactly")


if __name__ == "__main__":
    main()
