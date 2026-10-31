"""
Keyed update of PCD numbers after the label-free rerun (round 3, Reviewer 2,
comments 1 and 7).

Every PCD analysis was re-run with folds drawn without labels; the previous
outputs are kept as out/stale_labelstrat_<name>. For each number printed in
the manuscript, the cells of the OLD PCD result files that match it at the
printed precision are found. A number is updated only if all those cells,
looked up by the same key (file, identifying row fields, column) in the NEW
files, give one and the same new value at the printed precision; it is then
replaced in place, with the same number of decimals. Numbers that match no
old PCD cell are left alone (they are not PCD results); numbers whose keys
disagree are listed and not changed.

Blocks rebuilt by the writers (111, 115, 117) are rewritten from the new files
when those run afterwards; updating them here first is harmless.

Usage: python 136_pcd_keyed_update.py [--dry] [--allow LINE:VALUE,...]
(--allow: numbers reviewed by hand as PCD results whose value also happens to
occur in a non-PCD output file.)
Output: out/pcd_keyed_update.csv (every decision)
"""
import glob
import io
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
TEX = r"E:\Dream_Analysis\accesstemp.tex"
PFX = "stale_labelstrat_"
NUM = re.compile(r"(?<![\w.])[-+]?\d+\.\d+(?![\w.])")


def cells(path, tag):
    d = pd.read_csv(path)
    keycols = [c for c in d.columns if d[c].dtype == object]
    out = {}
    for i in range(len(d)):
        key = tuple(str(d.iloc[i][c]) for c in keycols)
        for c in d.select_dtypes("number").columns:
            v = d.iloc[i][c]
            if pd.notna(v) and np.isfinite(v):
                out[(tag, key, c)] = float(v)
    return out


def main():
    dry = "--dry" in sys.argv
    # One-time migration, applied on 21 Oct 2026. Re-run on a manuscript that
    # already holds the new values, it would map any of them that equals some
    # other old cell to that cell's new value (found by the double run of the
    # final pipeline). Refuse unless the update is explicitly forced.
    if (os.path.exists(os.path.join(OUT, "pcd_keyed_update.csv"))
            and "--force" not in sys.argv and not dry):
        raise SystemExit("136 was already applied; re-running it corrupts "
                         "values (see the comment above). Use --dry to inspect.")
    # line:value pairs reviewed by hand as PCD numbers although the same
    # value also occurs in some non-PCD output (a coincidence at 4 decimals)
    allow = set()
    if "--allow" in sys.argv:
        allow = set(sys.argv[sys.argv.index("--allow") + 1].split(","))
    old, new = {}, {}
    for f in sorted(glob.glob(os.path.join(OUT, PFX + "*.csv"))):
        name = os.path.basename(f)[len(PFX):]
        nf = os.path.join(OUT, name)
        if not os.path.exists(nf):
            continue
        old.update(cells(f, name))
        new.update(cells(nf, name))
    ov = np.array(list(old.values()))
    ok_keys = list(old.keys())
    # every other (non-PCD, non-stale) result value: a printed number that
    # also matches one of these may not be a PCD number and is not touched
    other = []
    for f in glob.glob(os.path.join(OUT, "**", "*.csv"), recursive=True):
        b = os.path.basename(f)
        if b.startswith(("stale_", "pcd_", "final_audit", "provenance_",
                         "letter_audit", "pcd_keyed")):
            continue
        try:
            d = pd.read_csv(f).select_dtypes("number").to_numpy().ravel()
        except Exception:
            continue
        other.append(d[np.isfinite(d)])
    other = np.sort(np.concatenate(other)) if other else np.array([])

    def in_other(x, tol):
        i = np.searchsorted(other, x - tol)
        return i < len(other) and other[i] <= x + tol
    s = io.open(TEX, encoding="utf-8", newline="").read()
    body0 = s.index("\\begin{abstract}")
    rows, out, last = [], [], 0
    for m in NUM.finditer(s):
        if m.start() < body0:
            continue
        txt = m.group()
        dec = len(txt.split(".")[1])
        x = float(txt)
        tol = 0.5 * 10 ** (-dec) + 1e-9
        hit = np.flatnonzero(np.abs(ov - x) <= tol)
        if dec < 3 or not len(hit):
            continue
        keys = [ok_keys[i] for i in hit]
        newv = {round(new[k], dec) for k in keys if k in new}
        missing = [k for k in keys if k not in new]
        line = s.count("\n", 0, m.start()) + 1
        ctx = s[max(0, m.start() - 90):m.end() + 40].replace("\n", " ")
        cand = (f"{next(iter(newv)):.{dec}f}" if len(newv) == 1 and not missing
                else "")
        if ((in_other(x, tol) or in_other(-x, tol))
                and f"{line}:{txt}" not in allow):
            rows.append(dict(line=line, old=txt, new="", candidate=cand,
                             decision="also matches a non-PCD result; left "
                             "unchanged", n_keys=len(keys), example_key=" | "
                             .join(map(str, keys[0])), context=ctx))
            continue
        if len(newv) == 1 and not missing:
            nv = f"{newv.pop():.{dec}f}"
            if txt.startswith("+") and not nv.startswith("-"):
                nv = "+" + nv
            decision = "updated" if nv != txt else "unchanged"
            if decision == "updated":
                out.append(s[last:m.start()] + nv)
                last = m.end()
        else:
            nv, decision = "", ("ambiguous: keys disagree" if len(newv) > 1
                                else "key missing in new output")
        if f"{line}:{txt}" in allow:
            decision += " (allowed after review)"
        rows.append(dict(line=line, old=txt, new=nv, candidate=cand,
                         decision=decision,
                         n_keys=len(keys), example_key=" | ".join(
                             map(str, keys[0])), context=ctx))
    out.append(s[last:])
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "pcd_keyed_update.csv"), index=False)
    print(t.decision.value_counts().to_string())
    for _, r in t[~t.decision.isin(["updated", "unchanged"])].iterrows():
        print(f"  line {r.line} {r.old}: {r.decision} ... {r.context[-110:]}")
    if not dry:
        io.open(TEX, "w", encoding="utf-8", newline="").write("".join(out))
        print("manuscript updated")


if __name__ == "__main__":
    main()
