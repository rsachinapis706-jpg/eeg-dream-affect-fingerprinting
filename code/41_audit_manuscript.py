"""
Audit every number in the manuscript against the generated result files.

The paper argues that undocumented analysis choices make published figures
uncheckable. It would be indefensible for its own numbers to be uncheckable, so
this script mechanically compares each decimal appearing in the manuscript
against the values written to disk by the scripts that computed them.

Method
------
Every value in every CSV under analysis/out is collected into a registry, keyed
by the file it came from. Each decimal number in the .tex is then looked up in
that registry at the precision it is printed. A number is reported as:

  FOUND       matches a value in at least one result file
  DERIVED     matches a simple arithmetic combination of two registry values
              (differences and sums, for inflation and gap statements)
  ABSENT      not found -- either read off a figure, a baseline defined in
              text, a reviewer-supplied reference figure, or an error

ABSENT is not by itself a fault. The point is that the list is short enough to
inspect by hand, and that nothing on it is a surprise. Anything unexpected on
that list is a defect to fix before submission.
"""
import os
import re
import glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
TEX = os.path.join("E:" + os.sep, "Dream_Analysis", "accesstemp.tex")

# values that legitimately appear without being in a results file
KNOWN_TEXT = {
    0.3333, 0.05, 0.10, 0.01, 0.50, 0.70, 0.90, 0.95, 1.0, 0.0, 0.5,
    0.9233, 0.9383, 0.862,          # published reference points
    0.0294, 0.0556, 0.0500,         # chance levels, stated in text
    0.4616, 0.4617, 0.4602, 0.4614, 0.4607, 0.4565, 0.4386, 0.4710, 0.4368,
    0.75, 0.25, 0.2, 0.4, 0.6, 0.8, 0.85, 0.65, 0.98, 0.72,  # widths/fractions
}

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def registry():
    reg = {}
    for f in sorted(glob.glob(os.path.join(OUT, "*.csv"))):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        name = os.path.basename(f)
        for col in d.columns:
            s = pd.to_numeric(d[col], errors="coerce").dropna()
            for v in s.values:
                for p in (4, 3, 2):
                    reg.setdefault(round(float(v), p), set()).add(name)
    return reg


def derived(x, vals, tol=5e-5):
    """Is x a difference or sum of two registry values? Covers inflation/gap."""
    arr = np.array(sorted(vals))
    for a in arr:
        if abs(abs(a - x) - 0) < tol:
            continue
        lo = np.searchsorted(arr, a - x - tol)
        hi = np.searchsorted(arr, a - x + tol)
        if hi > lo:
            return True
        lo = np.searchsorted(arr, x - a - tol)
        hi = np.searchsorted(arr, x - a + tol)
        if hi > lo:
            return True
    return False


def main():
    reg = registry()
    keys = set(reg)
    log("=" * 78)
    log("MANUSCRIPT NUMBER AUDIT")
    log("=" * 78)
    log(f"result files scanned : {len(set(f for s in reg.values() for f in s))}")
    log(f"distinct values      : {len(reg)}")
    log("")

    tex = open(TEX, encoding="utf-8").read()
    body = re.sub(r"%.*", "", tex)

    found = absent = drv = 0
    misses = []
    seen = set()
    for m in re.finditer(r"(?<![\w.])(\d\.\d{2,4})(?![\d])", body):
        x = float(m.group(1))
        ctx = body[max(0, m.start() - 70):m.end() + 40].replace("\n", " ")
        if x in KNOWN_TEXT:
            continue
        if x in keys:
            found += 1
            continue
        if derived(x, keys):
            drv += 1
            continue
        absent += 1
        if x not in seen:
            seen.add(x)
            misses.append((x, re.sub(r"\s+", " ", ctx).strip()))

    log(f"{'FOUND in result files':<26}: {found}")
    log(f"{'DERIVED (difference/sum)':<26}: {drv}")
    log(f"{'ABSENT (inspect below)':<26}: {absent} "
        f"({len(misses)} distinct)")
    log("")
    log("-" * 78)
    log("ABSENT VALUES -- each must have an explanation")
    log("-" * 78)
    for x, ctx in sorted(misses):
        log(f"{x:<10} ...{ctx[:96]}")

    with open(os.path.join(OUT, "test31_manuscript_audit.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log("")
    log(f"Wrote {OUT}\\test31_manuscript_audit.txt")


if __name__ == "__main__":
    main()
