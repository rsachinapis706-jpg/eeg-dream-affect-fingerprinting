"""
Final consistency audit of the revised manuscript.

Everything a reviewer could mechanically check is checked here, so that nothing
mechanical is left for them to find. Seven classes of defect are tested, each of
which has actually occurred at some point during this revision:

  1. LaTeX structure      unbalanced braces or environments, undefined
                          references, footnote markers with no text
  2. Bibliography         citations with no entry, entries never cited
  3. Graphics             a figure referenced but absent from the directory
  4. Number provenance    a decimal in the manuscript that no result file
                          contains and that is not a documented constant
  5. Superseded values    numbers from the earlier software environment or from
                          the pre-correction CBraMod cohort, surviving anywhere
                          they are asserted rather than disclosed
  6. Claim consistency    absolute-null language that the revised evidence no
                          longer supports
  7. Terminology          the protocol named inconsistently

Number provenance is delegated to 41_audit_manuscript.py, which is the tool the
Data Availability Statement promises readers; it is invoked here rather than
duplicated.
"""
import io
import os
import re
import subprocess
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXDIR = os.path.join("E:" + os.sep, "Dream_Analysis")
TEX = os.path.join(TEXDIR, "accesstemp.tex")
BIB = os.path.join(TEXDIR, "references.bib")

lines = []
problems = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def fail(m):
    problems.append(m)
    log(f"  FAIL  {m}")


def ok(m):
    log(f"  ok    {m}")


def main():
    s = io.open(TEX, encoding="utf-8").read()

    log("=" * 78)
    log("FINAL MANUSCRIPT AUDIT")
    log("=" * 78)
    log(f"file: {TEX}")
    log(f"words: {len(s.split())}")
    log("")

    # ---- 1. structure -------------------------------------------------
    log("1. LaTeX structure")
    if s.count("{") == s.count("}"):
        ok(f"braces balanced ({s.count('{')})")
    else:
        fail(f"braces: {s.count('{')} open, {s.count('}')} close")

    b = Counter(re.findall(r"\\begin\{(\w+\*?)\}", s))
    e = Counter(re.findall(r"\\end\{(\w+\*?)\}", s))
    bad = [(k, b[k], e[k]) for k in set(b) | set(e) if b[k] != e[k]]
    ok("environments balanced") if not bad else fail(f"environments {bad}")

    labs = set(re.findall(r"\\label\{([^}]+)\}", s))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", s))
    (ok(f"all {len(refs)} references defined") if not refs - labs
     else fail(f"undefined refs: {sorted(refs - labs)}"))

    for m in re.finditer(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", s, re.S):
        blk = m.group(0)
        lab = re.search(r"label\{([^}]+)\}", blk)
        name = lab.group(1) if lab else "(unlabelled)"
        mk = sorted(set(re.findall(r"footnotemark\[(\d)\]", blk)))
        tx = sorted(set(re.findall(r"footnotetext\[(\d)\]", blk)))
        if mk or tx:
            (ok(f"footnotes paired in {name}") if mk == tx
             else fail(f"{name}: markers {mk} vs texts {tx}"))
    log("")

    # ---- 2. bibliography ----------------------------------------------
    log("2. Bibliography")
    cites = set()
    for x in re.findall(r"\\cite\{([^}]+)\}", s):
        cites.update(c.strip() for c in x.split(","))
    entries = set(re.findall(r"@\w+\{([^,]+),",
                             io.open(BIB, encoding="utf-8").read()))
    (ok(f"all {len(cites)} citations resolve") if not cites - entries
     else fail(f"cited but absent: {sorted(cites - entries)}"))
    unc = entries - cites
    (ok(f"all {len(entries)} entries cited") if not unc
     else fail(f"uncited entries: {sorted(unc)}"))
    log("")

    # ---- 3. graphics ---------------------------------------------------
    log("3. Graphics")
    gfx = set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", s))
    missing = [g for g in gfx if not os.path.exists(os.path.join(TEXDIR, g))]
    # Author photographs are supplied by the authors at submission and are
    # not produced by any script here, so their absence is reported
    # separately from a missing analysis figure, which would be a defect.
    photos = [g for g in missing
              if not os.path.basename(g).startswith('fig')]
    figs = [g for g in missing if g not in photos]
    (ok(f'{len(gfx) - len(missing)} of {len(gfx)} graphics present')
     if not figs else fail(f'missing analysis figures: {figs}'))
    if photos:
        log(f'  note  author photographs absent: {sorted(photos)}')
        log('        author-supplied assets, required before compiling')
    log("")

    # ---- 4. number provenance ------------------------------------------
    log("4. Number provenance (delegated to 41_audit_manuscript.py)")
    r = subprocess.run([sys.executable,
                        os.path.join(ROOT, "analysis",
                                     "41_audit_manuscript.py")],
                       capture_output=True, text=True)
    out = r.stdout
    m = re.search(r"ABSENT \(inspect below\)\s*:\s*(\d+)", out)
    f = re.search(r"FOUND in result files\s*:\s*(\d+)", out)
    d = re.search(r"DERIVED \(difference/sum\)\s*:\s*(\d+)", out)
    if m and int(m.group(1)) == 0:
        ok(f"{f.group(1)} matched to result files, {d.group(1)} derived, "
           f"0 unaccounted")
    elif m:
        fail(f"{m.group(1)} numbers not traceable to any result file")
    else:
        fail("provenance audit did not report a result")
    log("")

    # ---- 5. superseded values ------------------------------------------
    log("5. Superseded values still asserted")
    # each: (value, what it was, phrases that legitimately disclose it)
    SUPERSEDED = [
        # No disclosure phrases are permitted any more: the manuscript is a
        # standalone document, so superseded values must be absent outright
        # rather than present-but-explained.
        ("0.8217", "CBraMod on the 8-recording cohort", []),
        ("0.3070", "participant-separated bal. acc., scikit-learn 1.4.1", []),
        ("0.4025", "participant-separated accuracy, scikit-learn 1.4.1", []),
    ]
    for val, what, allowed in SUPERSEDED:
        hits = []
        for ln_no, ln in enumerate(s.split("\n"), 1):
            if val in ln and not any(a in ln for a in allowed):
                hits.append(ln_no)
        (ok(f"{val} ({what}) only appears where disclosed") if not hits
         else fail(f"{val} asserted as current at line(s) {hits}"))
    log("")

    # ---- 6. claim consistency ------------------------------------------
    log("6. Claim consistency")
    OVERCLAIM = [
        "no detectable dream-affect signal",
        "no statistically detectable",
        "no participant-generalizable affect signal is detectable",
        "no dream-affect signal was detected",
        # "absence of signal" is deliberately NOT listed. It occurs once,
        # in "absence of evidence ... is weaker than absence of signal",
        # which is a correct statement about epistemics rather than a
        # claim about the data. Flagging it would push the prose towards
        # being vaguer than the evidence, which is the opposite of the aim.
    ]
    for phrase in OVERCLAIM:
        n = len(re.findall(re.escape(phrase), s, re.I))
        (ok(f"absent: '{phrase}'") if n == 0
         else fail(f"'{phrase}' appears {n} time(s)"))
    for anchor in ("at most a weak", "0.3224", "0.3768", "0.9055"):
        n = len(re.findall(re.escape(anchor), s))
        (ok(f"'{anchor}' present ({n})") if n
         else fail(f"'{anchor}' missing from the manuscript"))
    log("")

    # ---- 7. terminology -------------------------------------------------
    log("7. Terminology")
    n_sep = len(re.findall(r"participant-separated", s, re.I))
    n_pw = len(re.findall(r"participant-wise", s, re.I))
    ok(f"participant-separated used {n_sep} times") if n_sep else \
        fail("primary term absent")
    (ok("participant-wise eliminated") if n_pw == 0
     else fail(f"participant-wise still appears {n_pw} time(s)"))
    own = []
    for ln_no, ln in enumerate(s.split("\n"), 1):
        if re.search(r"subject-wise", ln, re.I):
            if any(k in ln for k in ("record-wise", "saeb", "reserve",
                                     "Subject-wise Validation",
                                     "eq:subjectwise")):
                continue
            own.append(ln_no)
    (ok("subject-wise only in citations, keywords and the definition")
     if not own else fail(f"subject-wise used as our own term at {own}"))
    log("")

    # ---- verdict --------------------------------------------------------
    log("=" * 78)
    if problems:
        log(f"VERDICT: {len(problems)} problem(s) require attention")
        for p in problems:
            log(f"  - {p}")
    else:
        log("VERDICT: no mechanical defects found.")
        log("")
        log("This covers what can be checked automatically. It does not")
        log("cover whether the arguments are sound, whether the reviewers")
        log("will accept them, or whether the prose reads well.")
    log("=" * 78)

    with open(os.path.join(ROOT, "analysis", "out",
                           "test43_final_audit.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
