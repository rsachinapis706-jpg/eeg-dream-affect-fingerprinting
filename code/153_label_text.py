"""
Label provenance and the unit of prediction in the manuscript (round 3,
Reviewer 2, comment 3), from 125_label_sensitivity.py.

Adds to Section IV-F a subsection and a table: the primary evaluation
repeated on the recordings whose interval the ratings workbook corroborates,
and on the last minute before each awakening, with balanced accuracy over
windows, over recordings and weighted per participant, each tested against
the within-participant null (sequential, as the primary evaluation) with
Holm's correction across the four configurations. One sentence in the
Datasets section points to it.

The prose was written by hand for the result obtained: no configuration
significant after correction, and an uncorrected tendency in the last
minute before awakening for the two logistic regressions only. The script
checks that this pattern holds and stops otherwise. Idempotent.
"""
import io
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
TEX = os.environ.get("LABEL_TEX", r"E:\Dream_Analysis\accesstemp.tex")
VAR = ["all windows", "corroborated recordings", "last 60 s before awakening"]
CFG = ["RF, unweighted (headline)", "RF, scaled, balanced",
       "LogReg, scaled, unweighted", "LogReg, scaled, balanced"]
STATS = ["window", "recording", "participant_weighted"]
SUB = "\\subsubsection{Corroborated labels, the interval before awakening and the unit of prediction}"
ANCHOR = "\\subsubsection{At the splitter's partitions: dependence on the classifier}"
DS_OLD = ("it is used as distributed and is not re-segmented.")
DS_NEW = ("it is used as distributed and is not re-segmented. "
          "Section~\\ref{sec:howmuch} repeats the primary evaluation on the "
          "corroborated recordings alone.")


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


def f4(x):
    return f"{x:.4f}"


def main():
    t = pd.read_csv(os.path.join(OUT, "label_sensitivity_summary.csv"))
    assert t.decided.all(), "a sequential test has not stopped"
    t = t.set_index(["variant", "config", "statistic"])
    for v in VAR:
        for s in STATS:
            h = holm([t.loc[(v, c, s), "p_within"] for c in CFG])
            for c, x in zip(CFG, h):
                t.loc[(v, c, s), "p_holm"] = x
    # the pattern the prose describes
    lr = CFG[2:]
    last = "last 60 s before awakening"
    other = t.drop(index=last, level="variant")
    assert other.p_within.min() >= 0.05, "all windows / corroborated: p < 0.05"
    assert t.p_holm.min() >= 0.05, "a result survives Holm's correction"
    assert t.loc[(last, CFG[0])].p_within.min() >= 0.05
    assert t.loc[(last, CFG[1])].p_within.min() >= 0.05
    assert min(t.loc[(last, c)].p_within.min() for c in lr) < 0.05
    lo_ = min(t.loc[(last, c)].p_within.min() for c in lr)
    hi_ = max(t.loc[(last, c)].p_within.max() for c in lr)
    rf_min = min(t.loc[(last, c)].p_within.min() for c in CFG[:2])
    a = {s: [t.loc[(last, c, s), "observed"] for c in (CFG[3], CFG[2])]
         for s in STATS}
    # subset sizes as 125 logged them
    import re
    log = io.open(os.path.join(OUT, "run_label_sensitivity.log"),
                  encoding="utf-8").read()

    def size(v):
        m = re.search(re.escape(v) + r": (\d+) windows, (\d+) recordings, "
                      r"(\d+) participants", log)
        return dict(windows=int(m.group(1)), recordings=int(m.group(2)),
                    participants=int(m.group(3)))
    n_corr, n_l = size("corroborated recordings"), size(last)
    assert size("all windows")["recordings"] == n_l["recordings"]

    def N(k):
        return f"{k:,}".replace(",", "{,}")
    rows = []
    for v in VAR:
        for k, c in enumerate(CFG):
            cells = [f"{f4(t.loc[(v, c, s), 'observed'])} "
                     f"({t.loc[(v, c, s), 'p_within']:.4f})" for s in STATS]
            lab = ({"all windows": "All windows",
                    "corroborated recordings": "Corroborated recordings",
                    last: "Last 60\\,s before awakening"}[v] if k == 0 else "")
            rows.append(f"{lab} & {c} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule" if v != VAR[-1] else r"\botrule")
    table = "\n".join([
        r"\begin{table*}[htbp]",
        r"\centering",
        ("\\caption{The primary evaluation on corroborated labels and on the "
         "last minute before awakening, at three units. Balanced accuracy "
         "averaged over the 20 random participant partitions, with its "
         "sequential within-participant permutation $p$ in parentheses. "
         "Window: over windows. Recording: over recordings, each predicted "
         "from the mean of its windows' class probabilities. "
         "Participant-weighted: over windows weighted inversely to each "
         "participant's window count. Holm's correction across the four "
         "configurations leaves no $p$ below 0.05 (smallest "
         f"{t.p_holm.min():.4f}). Chance is 0.3333.}}"),
        r"\label{tab:labelsens}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"Data & Configuration & Window & Recording & Participant-weighted \\",
        r"\midrule",
    ] + rows[:-1] + [r"\botrule", r"\end{tabular}", r"\normalsize",
                     r"\end{table*}"])
    par = (
        f"{SUB}\n\n"
        "Three further versions of the primary evaluation address how the "
        "labels were attached and what a prediction is of. First, it was "
        f"repeated on the {n_corr['recordings']} valence recordings whose "
        "interval the ratings workbook corroborates "
        f"(Section~\\ref{{sec:reporting}}; {n_corr['participants']} "
        f"participants, {N(n_corr['windows'])} windows). Second, because one "
        "report labels a whole recording of variable length, it was repeated "
        "on the last minute before each awakening, the final three 20\\,s "
        f"windows of every recording ({N(n_l['windows'])} windows from all "
        f"{n_l['recordings']} recordings). Third, for all windows and for both subsets, balanced "
        "accuracy is reported at three units: over windows, which weights "
        "each recording by its length; over recordings, each predicted from "
        "the mean class probabilities of its windows, which scores the "
        "prediction of an awakening report; and over windows weighted "
        "inversely to each participant's window count, which gives every "
        "participant equal weight. Each is tested against the "
        "within-participant null with the sequential procedure of the "
        "primary evaluation, and Holm's procedure corrects across the four "
        "configurations (Table~\\ref{tab:labelsens}).\n\n"
        "Neither the restriction to corroborated recordings nor the unit of "
        "prediction changes the result: on all windows and on the "
        "corroborated recordings no configuration is distinguishable from its "
        f"null at any unit (smallest $p$ {other.p_within.min():.4f}). In the "
        "last minute before awakening the two logistic regressions score "
        f"higher, {f4(a['window'][0])} and {f4(a['window'][1])} over windows, "
        f"{f4(a['recording'][0])} and {f4(a['recording'][1])} over "
        f"recordings and {f4(a['participant_weighted'][0])} and "
        f"{f4(a['participant_weighted'][1])} with participants weighted "
        f"equally, with uncorrected $p$ from {lo_:.4f} to {hi_:.4f}; after "
        "Holm's correction across the four configurations the smallest is "
        f"{t.p_holm.min():.4f}, and the 36 tests of the table would call for "
        "a stronger correction still. The forests show no such tendency "
        f"($p \\ge {int(rf_min * 100) / 100:.2f}$). The pattern is the one "
        "this analysis was designed to detect, an association concentrated "
        "near the awakening and diluted when one report labels a whole "
        "recording, but at this cohort size it is not distinguishable from "
        "chance after correction, and it is reported as a hypothesis for a "
        "larger cohort, not as a finding.\n\n" + table + "\n\n")
    s = io.open(TEX, encoding="utf-8", newline="").read()
    if SUB in s:                                   # re-run: replace it whole
        i = s.index(SUB)
        j = s.index(ANCHOR, i)
        s = s[:i] + s[j:]
    assert s.count(ANCHOR) == 1
    s = s.replace(ANCHOR, par + ANCHOR)
    if DS_NEW not in s:
        assert s.count(DS_OLD) == 1
        s = s.replace(DS_OLD, DS_NEW)
    io.open(TEX, "w", encoding="utf-8", newline="").write(s)
    print(par)


if __name__ == "__main__":
    main()
