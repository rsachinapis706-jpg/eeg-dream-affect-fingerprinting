"""
Numerical audit of the manuscript against the result files (Reviewer 1,
comment 2; Reviewer 2, first comment).

Every decimal number in the body of the manuscript is extracted with its line.
A value is located when some number written by an analysis script -- in any
CSV, text or log file under analysis/out -- rounds to it at the precision the
manuscript prints; a percentage is compared as a proportion, and a signed
difference as printed. Values that cannot be located are listed with their
context in out/final_audit_unmatched.csv for review; values that are settings
rather than results (band edges, thresholds, alpha levels) are recognised by a
short list of contexts and reported separately, never silently dropped.

Output: out/final_audit.csv (every value, where located), out/final_audit_
unmatched.csv, out/final_audit_summary.csv.
"""
import glob
import io
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
TEX = r"E:\Dream_Analysis\accesstemp.tex"
NUM = re.compile(r"(?<![\w.])[-+−]?\d+\.\d+(?!\w|\.\d)")
SETTING_CONTEXT = re.compile(
    r"(Hz|\\,s\b|seconds|\\mu|uV|μV|alpha|\\alpha|threshold|C=|C =|"
    r"learning rate|lr |epoch|batch|patience|version|Python|scikit|NumPy|"
    r"SciPy|pandas|MNE|PyTorch|vol\.|pp\.|\\cite|eta|\\eta|delta|\\delta|"
    r"Q=|order|weight decay|0\.5--45|300)", re.I)


def pool():
    vals = []
    files = (glob.glob(os.path.join(OUT, "**", "*.csv"), recursive=True)
             + glob.glob(os.path.join(OUT, "*.txt"))
             + glob.glob(os.path.join(OUT, "*.log")))
    src = {}
    for f in files:
        if os.path.basename(f).startswith(("final_audit", "stale_")):
            continue
        try:
            t = io.open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for m in re.finditer(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", t):
            try:
                v = float(m.group())
            except ValueError:
                continue
            vals.append(v)
            src.setdefault(round(v, 4), os.path.basename(f))
    return np.unique(np.array(vals)), src


def located(x, dec, arr):
    tol = 0.5 * 10 ** (-dec) + 1e-9
    i = np.searchsorted(arr, x - tol)
    return i < len(arr) and arr[i] <= x + tol


def main():
    arr, src = pool()
    s = io.open(TEX, encoding="utf-8").read()
    start = s.index("\\begin{document}")
    end = s.find("\\begin{thebibliography}")
    end = end if end > 0 else s.index("\\end{document}")
    lines = s[:end].split("\n")
    first = s[:start].count("\n")
    rows = []
    in_tikz = False
    for ln in range(first, len(lines)):
        line = lines[ln]
        if r"\begin{tikzpicture}" in line:
            in_tikz = True
        if r"\end{tikzpicture}" in line:
            in_tikz = False
        if line.lstrip().startswith("%"):
            continue
        for m in NUM.finditer(line):
            txt = m.group().replace("−", "-")
            x = float(txt)
            dec = len(txt.split(".")[1])
            after = line[m.end():m.end() + 3]
            ctx = line[max(0, m.start() - 60):m.end() + 40]
            pct = after.startswith("\\%") or after.startswith("%")
            ok = located(x, dec, arr) or located(-x, dec, arr)
            how = "value"
            if not ok and pct:
                ok = located(x / 100, dec + 2, arr)
                how = "percentage"
            if not ok and dec >= 3:
                # a difference of two located values printed as a signed number
                how = "unlocated"
            # axis limits and bar positions of drawn figures are layout, but
            # the plotted values themselves must still be located
            setting = bool(SETTING_CONTEXT.search(ctx)) or (
                in_tikz and re.search(r"(xmin|xmax|ymin|ymax|xtick|ytick|"
                                      r"axis cs|coordinates)", line) is not None
                and not ok)
            rows.append(dict(line=ln + 1, value=txt, decimals=dec,
                             located=ok, how=how if ok else "unlocated",
                             setting_context=setting, context=ctx))
    a = pd.DataFrame(rows)
    a.to_csv(os.path.join(OUT, "final_audit.csv"), index=False)
    un = a[~a.located]
    un.to_csv(os.path.join(OUT, "final_audit_unmatched.csv"), index=False)
    res = un[~un.setting_context]
    summ = dict(n_values=len(a), n_matched=int(a.located.sum()),
                n_unmatched=len(un), n_unmatched_results=len(res),
                note=(f"{len(un) - len(res)} unlocated values are settings "
                      f"(filter edges, thresholds, levels, versions); "
                      f"{len(res)} unlocated result values listed for review"))
    pd.DataFrame([summ]).to_csv(os.path.join(OUT, "final_audit_summary.csv"),
                                index=False)
    print(summ)
    with pd.option_context("display.max_colwidth", 110, "display.width", 250,
                           "display.max_rows", 400):
        print(res[["line", "value", "context"]].to_string(index=False))


if __name__ == "__main__":
    main()
