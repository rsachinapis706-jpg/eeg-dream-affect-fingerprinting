"""
Deep structural audit: tables, equations, floats, cross-references, prose.

53_final_audit.py checks the things that break a build or contradict a claim.
This checks the things that survive a build and embarrass you in print: a table
whose rows do not match its column specification, an equation that is numbered
but never referred to, a float that is never cited so LaTeX drifts it to the end
of the document, a sentence duplicated by an editing pass, or the same quantity
written to a different precision in two places.

Nothing here is clever. It is the set of checks that would have caught each of
the defects actually found during this revision, applied to everything at once.
"""
import io
import os
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP

TEXDIR = os.path.join("E:" + os.sep, "Dream_Analysis")
TEX = os.path.join(TEXDIR, "accesstemp.tex")

lines_out = []
problems = []
notes = []


def log(m=""):
    print(m, flush=True)
    lines_out.append(str(m))


def fail(m):
    problems.append(m)
    log(f"  FAIL  {m}")


def note(m):
    notes.append(m)
    log(f"  note  {m}")


def ok(m):
    log(f"  ok    {m}")


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)



def read_spec_and_body(blk, i):
    """Read a tabular's column spec by matching braces, then its body.

    A naive [^}]* capture breaks on specs containing @{} directives, which is
    every table in this paper.
    """
    while i < len(blk) and blk[i] in " \n":
        i += 1
    if i >= len(blk) or blk[i] != "{":
        return None, None
    depth, j = 1, i + 1
    while j < len(blk) and depth:
        if blk[j] == "{":
            depth += 1
        elif blk[j] == "}":
            depth -= 1
        j += 1
    spec = blk[i + 1:j - 1]
    end = blk.find("\\end{tabular}", j)
    return spec, blk[j:end if end > 0 else len(blk)]


def float_blocks(s, kind):
    pat = r"\\begin\{" + kind + r"\*?\}(.*?)\\end\{" + kind + r"\*?\}"
    return list(re.finditer(pat, s, re.S))


def main():
    raw = io.open(TEX, encoding="utf-8").read()
    s = strip_comments(raw)

    log("=" * 78)
    log("DEEP STRUCTURAL AUDIT")
    log("=" * 78)
    log("")

    # ---------- 1. tables: column arithmetic -------------------------
    log("1. Table column arithmetic")
    for m in float_blocks(s, "table"):
        blk = m.group(1)
        lab = re.search(r"label\{([^}]+)\}", blk)
        name = lab.group(1) if lab else "(unlabelled)"
        for tm in re.finditer(r"\\begin\{tabular\}", blk):
            spec, body = read_spec_and_body(blk, tm.end())
            if spec is None:
                continue
            ncol = len(re.findall(r"p\{[^}]*\}|[lcr]",
                                  re.sub(r"@\{[^}]*\}", "", spec)))
            bad = []
            for row in body.split("\\\\"):
                r = row.strip()
                if not r or r.startswith("\\") and "&" not in r:
                    continue
                if any(k in r for k in ("toprule", "midrule", "botrule",
                                        "cmidrule", "addlinespace")) and "&" not in r:
                    continue
                # An escaped ampersand is literal text, not a column
                # separator; counting it made every table containing "\&"
                # look malformed.
                r2 = r.replace("\\&", "")
                r2 = re.sub(r"\\multicolumn\{(\d+)\}",
                            lambda x: "&" * (int(x.group(1)) - 1), r2)
                n = r2.count("&") + 1
                if n != ncol and "&" in r:
                    bad.append((n, r[:52]))
            if bad:
                fail(f"{name}: spec has {ncol} columns; "
                     f"{len(bad)} row(s) differ, e.g. {bad[0][0]} in '{bad[0][1]}'")
            else:
                ok(f"{name}: all rows match {ncol} columns")
    log("")

    # ---------- 2. floats referenced ---------------------------------
    log("2. Every float referenced in the text")
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", s))
    for kind, tag in (("table", "tab"), ("figure", "fig")):
        for m in float_blocks(s, kind):
            lab = re.search(r"label\{([^}]+)\}", m.group(1))
            if not lab:
                fail(f"a {kind} has no \\label")
                continue
            if lab.group(1) not in refs:
                fail(f"{kind} '{lab.group(1)}' is never referenced "
                     f"(LaTeX will float it to the end)")
    n_t = len(float_blocks(s, "table"))
    n_f = len(float_blocks(s, "figure"))
    ok(f"{n_t} tables and {n_f} figures examined")
    log("")

    # ---------- 3. equations -----------------------------------------
    log("3. Equations")
    eqlabs = re.findall(r"\\begin\{equation\}(.*?)\\end\{equation\}", s, re.S)
    labelled = [re.search(r"label\{([^}]+)\}", e) for e in eqlabs]
    named = [x.group(1) for x in labelled if x]
    log(f"        {len(eqlabs)} numbered equations, {len(named)} labelled")
    unref = [n for n in named if n not in refs]
    if unref:
        note(f"labelled but never referenced: {unref} "
             f"(harmless, but the number is then decorative)")
    else:
        ok("every labelled equation is referenced")
    for e in eqlabs:
        if e.count("{") != e.count("}"):
            fail(f"unbalanced braces in equation: {e.strip()[:60]}")
    ok("equation braces balanced")
    log("")

    # ---------- 4. section cross-references --------------------------
    log("4. Cross-references")
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", s))
    labs = set(re.findall(r"\\label\{([^}]+)\}", s))
    missing = sorted(refs - labs)
    (ok(f"all {len(refs)} \\ref targets exist") if not missing
     else fail(f"dangling: {missing}"))
    dup = [k for k, v in Counter(re.findall(r"\\label\{([^}]+)\}", s)).items()
           if v > 1]
    (ok("no duplicate labels") if not dup else fail(f"duplicate labels: {dup}"))
    log("")

    # ---------- 5. duplicated prose ----------------------------------
    log("5. Duplicated sentences (editing artefacts)")
    body = re.sub(r"\\begin\{(table|figure|equation|algorithmic|tabular)\*?\}.*?"
                  r"\\end\{\1\*?\}", " ", s, flags=re.S)
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", body)
             if len(x.strip()) > 90]
    seen = defaultdict(int)
    for x in sents:
        seen[re.sub(r"\s+", " ", x)] += 1
    dups = {k: v for k, v in seen.items() if v > 1}
    if dups:
        for k, v in list(dups.items())[:5]:
            fail(f"sentence appears {v}x: '{k[:70]}...'")
    else:
        ok(f"{len(sents)} long sentences, none duplicated")
    log("")

    # ---------- 6. number consistency --------------------------------
    log("6. Key quantities stated consistently")
    CANON = {
        "participant-separated balanced accuracy (headline)": ["0.3224"],
        "best configuration": ["0.3768", "0.3783"],
        "leakage ceiling": ["0.9055", "0.906"],
        "identity, window-shuffled": ["0.8040", "0.804"],
        "identity-only oracle": ["0.5465", "0.547"],
        "CBraMod matched control": ["0.7906", "0.791"],
        "spectral matched control": ["0.7799", "0.780"],
    }
    for what, forms in CANON.items():
        counts = {f: len(re.findall(re.escape(f), s)) for f in forms}
        present = {f: c for f, c in counts.items() if c}
        if not present:
            fail(f"{what}: none of {forms} appears")
        else:
            ok(f"{what}: " + ", ".join(f"{f} x{c}" for f, c in present.items()))
    # Many three-decimal values in this paper are standalone quantities --
    # p-values, mutual-information excesses, effect differences -- and are not
    # roundings of anything. Flagging those produced noise. The defect worth
    # catching is narrower: one of the CANONICAL quantities above appearing at
    # three decimals as an INCORRECT rounding of its four-decimal form.
    round_problems = []
    for what, forms in CANON.items():
        longest = [f for f in forms if len(f.split(".")[1]) == 4]
        shorter = [f for f in forms if len(f.split(".")[1]) == 3]
        if not longest or not shorter:
            continue
        ref = float(longest[0])
        # Python's round() is banker's rounding, so round(0.9055, 3) gives
        # 0.905. Scientific writing rounds half away from zero, giving 0.906.
        # Using round() here would flag correct prose as wrong.
        exp = Decimal(longest[0]).quantize(Decimal("0.001"),
                                           rounding=ROUND_HALF_UP)
        for sh in shorter:
            if Decimal(sh) != exp:
                round_problems.append(
                    f"{what}: '{sh}' is not the correct rounding of "
                    f"'{longest[0]}' (expected {exp})")
    if round_problems:
        for r in round_problems:
            fail(r)
    else:
        ok("every shortened form of a canonical quantity rounds correctly")

    log("")

    # ---------- 7. leftovers -----------------------------------------
    log("7. Draft leftovers")
    for pat, desc in [(r"\bTODO\b", "TODO"), (r"\bFIXME\b", "FIXME"),
                      (r"\bXXX\b", "XXX"), (r"\?\?", "?? (broken ref)"),
                      (r"\\todo", "\\todo"), (r"lorem", "lorem ipsum")]:
        n = len(re.findall(pat, s, re.I))
        (ok(f"no {desc}") if n == 0 else fail(f"{desc} appears {n}x"))
    dbl = re.findall(r"\b(\w+)\s+\1\b", body)
    dbl = [w for w in dbl if w.lower() not in ("that", "had", "s")]
    (ok("no doubled words") if not dbl
     else note(f"doubled words: {sorted(set(dbl))[:8]}"))
    log("")

    log("=" * 78)
    if problems:
        log(f"VERDICT: {len(problems)} problem(s)")
        for p in problems:
            log(f"  - {p}")
    else:
        log("VERDICT: structurally clean.")
    if notes:
        log("")
        log(f"{len(notes)} note(s) for human judgement:")
        for n in notes:
            log(f"  - {n}")
    log("=" * 78)

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "analysis", "out", "test44_deep_audit.txt")
    io.open(out, "w", encoding="utf-8").write("\n".join(lines_out))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
