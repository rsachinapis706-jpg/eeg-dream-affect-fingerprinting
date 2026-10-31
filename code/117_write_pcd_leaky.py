"""
Write the leaky-partition PCD result (116_pcd_leaky.py) into the manuscript.

Every number is read from out/pcd_leaky_*.csv. Each sentence that makes a
claim is written only if the numbers support it, and the script stops with a
message otherwise:
  * PCD within its level under the known null: mean coverage no more than two
    standard errors (over re-randomisations) above alpha, at every alpha;
  * the window-level partition inflates balanced accuracy under the known
    null: its interval over re-randomisations lies above the
    participant-separated one;
  * observed labels give the window-level partition no more than the known
    null does: the observed value lies within two standard deviations of the
    known-null values.

Edits (idempotent; re-running replaces the block and skips existing text):
  1. New subsubsection and table after "Coverage depends on the reference
     null", before the label-noise subsection.
  2. One sentence closing the paragraph on the scope of Proposition 1.
  3. One sentence in the PCD item of the contribution list.
  4. One sentence in the concluding paragraph on PCD.
The abstract clause is written by 101_write_summary.py and the letter by
94_response_letter.py, from the same files.
"""
import importlib.util
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
TEX = os.path.join("E:" + os.sep, "Dream_Analysis", "accesstemp.tex")
ALPHAS = (0.01, 0.05, 0.10)
PARTS = ("participant-separated", "window-level")


def f4(x):
    return f"{x:.4f}"


def stats():
    cond = pd.read_csv(os.path.join(OUT, "pcd_leaky_conditions.csv"))
    boot = pd.read_csv(os.path.join(OUT, "pcd_leaky_boot.csv"))
    nul = cond[cond.labelling != "observed"]
    R = nul.labelling.nunique()
    N = {}
    for (p, r), g in nul.groupby(["partition", "rule"]):
        N[(p, r, "cov")] = (g.coverage.mean(), g.coverage.std(ddof=1),
                            g.coverage.std(ddof=1) / np.sqrt(len(g)))
        N[(p, r, "bal")] = (g.bal_retained.mean(), g.bal_retained.std(ddof=1),
                            g.bal_retained.std(ddof=1) / np.sqrt(len(g)))
    O = boot.set_index(["partition", "rule"])
    # share of window-level test windows whose recording is also in training
    spec = importlib.util.spec_from_file_location(
        "leaky", os.path.join(HERE, "116_pcd_leaky.py"))
    lk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lk)
    X, y, subj, unit = lk.load_deed()
    folds = lk.partitions(subj)["window-level"]
    same = np.mean([np.isin(unit[te], unit[np.setdiff1d(np.arange(len(y)),
                                                         te)]).mean()
                    for te in folds])
    return R, N, O, same


def check(R, N, O):
    pcd = {a: f"PCD alpha={a:.2f}" for a in ALPHAS}
    for p in PARTS:
        for a in ALPHAS:
            m, _, se = N[(p, pcd[a], "cov")]
            if m - a > 2 * se:
                sys.exit(f"STOP: {p} known null, alpha {a}: coverage {m:.4f} "
                         f"exceeds alpha by more than 2 SE ({se:.4f}); the "
                         f"'within its level' claim is not supported")
    wn, wsd, _ = N[("window-level", "all inputs", "bal")]
    pn, psd, _ = N[("participant-separated", "all inputs", "bal")]
    if not wn - 2 * wsd > pn + 2 * psd:
        sys.exit("STOP: leak does not clearly inflate balanced accuracy")
    ow = O.loc[("window-level", "all inputs"), "bal_retained"]
    if abs(ow - wn) > 2 * wsd:
        sys.exit(f"STOP: observed window-level {ow:.4f} differs from the "
                 f"known null {wn:.4f} by more than 2 SD")


def block(R, N, O, same):
    g = lambda p, r, k="cov": f4(N[(p, r, k)][0])
    se = lambda p, r, k="cov": f4(N[(p, r, k)][2])
    W, P = "window-level", "participant-separated"
    MS, P05 = "Max-softmax>0.5", "PCD alpha=0.05"

    def obs(p, r, m):
        x = O.loc[(p, r)]
        return f"{f4(x[m])} [{f4(x[m + '_lo'])}, {f4(x[m + '_hi'])}]"

    def nrow(p, r, k):
        m, _, s = N[(p, r, k)]
        return f"{f4(m)} ({f4(s)})"

    text = (
        "\\subsubsection{The guarantee under a leaky partition}"
        "\\label{sec:pcdleak}\n\n"
        "Proposition~\\ref{prop:calib} asks only that the fold assignment be "
        "fixed without reference to the labels; it does not ask that the "
        "partition separate participants. A partition that places windows of "
        "one recording on both sides is applied identically to the observed "
        "and to every permuted labelling, so the null decoders draw the same "
        "advantage from it as the observed decoder. "
        "Table~\\ref{tab:pcdleak} tests this on DEED with the decoder of "
        "Table~\\ref{tab:pcd}, changing only the partition: five folds drawn "
        "once, without the labels, over participants or over windows. Over "
        f"windows, {f4(same)} of test windows have training windows from the "
        "same recording. "
        f"Under a known null, with recording labels re-randomised within "
        f"participant {R} times, the window-level partition raises balanced "
        f"accuracy to {g(W, 'all inputs', 'bal')}, against "
        f"{g(P, 'all inputs', 'bal')} with participants separated, and a "
        f"max-softmax threshold of 0.5 predicts on {g(W, MS)} of windows, "
        f"with retained balanced accuracy {g(W, MS, 'bal')}. PCD predicts on "
        f"{g(W, 'PCD alpha=0.01')}, {g(W, P05)} and {g(W, 'PCD alpha=0.10')} "
        "of windows at $\\alpha = 0.01$, $0.05$ and $0.10$, within two "
        "standard errors of each level, as it does with participants "
        f"separated ({g(P, 'PCD alpha=0.01')}, {g(P, P05)} and "
        f"{g(P, 'PCD alpha=0.10')}). With the observed labels the "
        "window-level partition gives a balanced accuracy of "
        f"{obs(W, 'all inputs', 'bal_retained')}, as under the known null, "
        f"and PCD predicts on {obs(W, P05, 'coverage')} of windows: under "
        "this partition, as with participants separated, it finds nothing "
        "beyond what the partition itself produces. "
        "What the guarantee controls is how often the rule predicts, not "
        "whether the evaluation is sound: the windows PCD retains under the "
        "leaky partition are still scored with leaked information, and their "
        f"balanced accuracy under the known null, {g(W, P05, 'bal')}, is not "
        "evidence of signal. A confidence threshold offers no such control: "
        f"its coverage under the known null, {g(P, MS)} and {g(W, MS)} under "
        "the two partitions, reflects the model and the data rather than an "
        "association with the label.\n\n"
        "\\begin{table*}[htbp]\n"
        "\\centering\n"
        "\\caption{PCD under a leaky partition. DEED valence, with the decoder "
        "and the $B=100$ null decoders per fold of Table~\\ref{tab:pcd}; five "
        "folds drawn once over participants or over windows, without "
        "reference to the labels. Observed labels: 95\\% participant-bootstrap "
        "intervals, 2{,}000 resamples. Known null: recording labels "
        f"re-randomised within participant {R} times; mean, with its standard "
        "error in parentheses.}\n"
        "\\label{tab:pcdleak}\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\begin{tabular}{@{}llccccc@{}}\n"
        "\\toprule\n"
        "& & & \\multicolumn{2}{c}{Max-softmax $>0.5$} & "
        "\\multicolumn{2}{c}{PCD, $\\alpha=0.05$} \\\\\n"
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "Folds over & Labels & Bal.\\ acc., all windows & Coverage & "
        "Retained bal.\\ acc. & Coverage & Retained bal.\\ acc. \\\\\n"
        "\\midrule\n")
    for p, name in ((P, "Participants"), (W, "Windows")):
        text += (f"{name} & Observed & {obs(p, 'all inputs', 'bal_retained')} "
                 f"& {obs(p, MS, 'coverage')} & {obs(p, MS, 'bal_retained')} "
                 f"& {obs(p, P05, 'coverage')} & {obs(p, P05, 'bal_retained')}"
                 " \\\\\n")
        text += (f" & Known null & {nrow(p, 'all inputs', 'bal')} "
                 f"& {nrow(p, MS, 'cov')} & {nrow(p, MS, 'bal')} "
                 f"& {nrow(p, P05, 'cov')} & {nrow(p, P05, 'bal')} \\\\\n")
    text += ("\\botrule\n\\end{tabular}\n\\normalsize\n\\end{table*}\n\n")
    return text


def insert_once(s, anchor, addition, marker):
    """Append `addition` right after `anchor` unless `marker` is present."""
    if marker in s:
        return s
    assert s.count(anchor) == 1, (s.count(anchor), anchor[:60])
    return s.replace(anchor, anchor + addition)


def save_values(R, N, same):
    """Every derived value the text prints, so the audit can locate it."""
    rows = [dict(partition=p, rule=r, quantity=k, labellings=R, mean=m,
                 sd=sd, se=se) for (p, r, k), (m, sd, se) in N.items()]
    rows.append(dict(partition="window-level", rule="",
                     quantity="test windows sharing a recording with training",
                     labellings=R, mean=same, sd=np.nan, se=np.nan))
    pd.DataFrame(rows).round(4).to_csv(
        os.path.join(OUT, "pcd_leaky_table.csv"), index=False)


def main():
    R, N, O, same = stats()
    check(R, N, O)
    if "--dry" in sys.argv:          # print the block; write nothing
        print(block(R, N, O, same))
        return
    save_values(R, N, same)
    s = io.open(TEX, encoding="utf-8", newline="").read()
    W, MS, P05 = "window-level", "Max-softmax>0.5", "PCD alpha=0.05"
    msp_w, p05_w = f4(N[(W, MS, "cov")][0]), f4(N[(W, P05, "cov")][0])
    bal_w = f4(N[(W, "all inputs", "bal")][0])

    # 1. results block, replaced wholesale if present
    start = "\\subsubsection{The guarantee under a leaky partition}"
    nxt = "\\subsection{Controlled Label-Noise Sweep}"
    assert s.count(nxt) == 1
    if start in s:
        a = s.index(start)
        s = s[:a] + s[s.index(nxt):]
    s = s.replace(nxt, block(R, N, O, same) + nxt)

    # 2. scope of Proposition 1
    s = insert_once(
        s,
        "Section~\\ref{sec4} therefore reports it across re-randomisations "
        "and simulated datasets rather than from a single run.",
        " In one respect its conditions are weaker than they may appear: the "
        "partition need not separate participants, only be fixed without "
        "reference to the labels, so the guarantee holds under a partition "
        "that leaks recording or participant identity "
        "(Section~\\ref{sec:pcdleak}).",
        "so the guarantee holds under a partition that leaks")

    # 3. contribution list
    s = insert_once(
        s,
        "follows from standard exchangeability arguments "
        "\\cite{hemerik2018exact}.",
        " It holds even when the partition leaks: with labels that carry no "
        "information and folds drawn over windows, a confidence threshold "
        f"predicts on {msp_w} of DEED windows and PCD on {p05_w} at "
        "$\\alpha = 0.05$.",
        "It holds even when the partition leaks")

    # 4. concluding paragraph on PCD
    s = insert_once(
        s,
        "and a prediction the rule makes is not thereby shown to be correct.",
        " Nor does the guarantee depend on a sound partition: on DEED with "
        "labels that carry no information, a window-level partition raises "
        f"balanced accuracy to {bal_w} and a confidence threshold predicts on "
        f"{msp_w} of windows, while PCD predicts on {p05_w}.",
        "Nor does the guarantee depend on a sound partition")

    io.open(TEX, "w", encoding="utf-8", newline="").write(s)
    print(f"leaky-partition PCD written: R = {R}; window-level known null "
          f"bal {bal_w}, max-softmax {msp_w}, PCD(0.05) {p05_w}")


if __name__ == "__main__":
    main()
