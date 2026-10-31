"""
Static checks on the manuscript source, standing in for a compile where no
LaTeX installation is available: environment nesting, brace balance, labels
and references, citation keys against the .bib, and, for every tabular, that
each row has as many cells as the column specification declares (counting
\\multicolumn spans). Prints problems; exits non-zero if any are found.

Usage: python 113_tex_static_check.py [path/to/file.tex]
"""
import io
import re
import sys

TEX = sys.argv[1] if len(sys.argv) > 1 else r"E:\Dream_Analysis\accesstemp.tex"
BIB = r"E:\Dream_Analysis\references.bib"


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


def colspec_count(spec):
    spec = re.sub(r"[><]\{(?:[^{}]|\{[^{}]*\})*\}", "", spec)   # >{...} <{...}
    spec = re.sub(r"@\{[^{}]*\}", "", spec)
    spec = re.sub(r"[pmb]\{[^{}]*\}", "X", spec)
    spec = re.sub(r"\*\{(\d+)\}\{([^{}]*)\}",
                  lambda m: m.group(2) * int(m.group(1)), spec)
    return len(re.findall(r"[lcrX]", spec))


def balanced_arg(s, i):
    """Return (content, end) of the {...} group starting at s[i] == '{'."""
    assert s[i] == "{"
    d, j = 1, i + 1
    while d:
        d += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return s[i + 1:j - 1], j


def main():
    s = strip_comments(io.open(TEX, encoding="utf-8").read())
    probs = []
    # braces
    if s.count("{") - s.count("\\{") != s.count("}") - s.count("\\}"):
        probs.append(f"brace imbalance {s.count('{') - s.count('}')}")
    # environments
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]*)\}", s):
        kind, env = m.groups()
        line = s.count("\n", 0, m.start()) + 1
        if kind == "begin":
            stack.append((env, line))
        elif not stack or stack[-1][0] != env:
            probs.append(f"line {line}: \\end{{{env}}} does not close "
                         f"{stack[-1] if stack else 'anything'}")
        else:
            stack.pop()
    if stack:
        probs.append(f"unclosed environments: {stack}")
    # labels, refs, cites
    labels = re.findall(r"\\label\{([^}]+)\}", s)
    dup = {x for x in labels if labels.count(x) > 1}
    if dup:
        probs.append(f"duplicate labels {sorted(dup)}")
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", s))
    if refs - set(labels):
        probs.append(f"undefined refs {sorted(refs - set(labels))}")
    keys = {k.strip() for m in re.findall(r"\\cite\{([^}]+)\}", s)
            for k in m.split(",")}
    bib = set(re.findall(r"@\w+\{([^,\s]+),", io.open(BIB, encoding="utf-8").read()))
    if keys - bib:
        probs.append(f"citations missing from .bib {sorted(keys - bib)}")
    # tabular cell counts
    n_tab = 0
    for m in re.finditer(r"\\begin\{tabular\}", s):
        n_tab += 1
        spec, j = balanced_arg(s, m.end())
        ncol = colspec_count(spec)
        end = s.index("\\end{tabular}", j)
        body = s[j:end]
        line0 = s.count("\n", 0, m.start()) + 1
        for row in re.split(r"\\\\", body):
            r = re.sub(r"\\(toprule|midrule|botrule|bottomrule|hline)\b", "", row)
            r = re.sub(r"\\(cmidrule|addlinespace)(\([^)]*\))?(\[[^]]*\])?(\{[^}]*\})?", "", r)
            if not r.strip():
                continue
            cells = 0
            depth = 0
            parts = [""]
            for ch in r:                      # split on & at brace depth 0
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                if ch == "&" and depth == 0 and not parts[-1].endswith("\\"):
                    parts.append("")
                else:
                    parts[-1] += ch
            for p in parts:
                mc = re.search(r"\\multicolumn\{(\d+)\}", p)
                cells += int(mc.group(1)) if mc else 1
            if cells != ncol:
                probs.append(f"table at line {line0}: row has {cells} cells, "
                             f"spec has {ncol}: {r.strip()[:70]!r}")
    print(f"{TEX}: {len(labels)} labels, {len(refs)} refs, {len(keys)} cite "
          f"keys, {n_tab} tabulars checked")
    for p in probs:
        print("  PROBLEM:", p)
    print("no problems found" if not probs else f"{len(probs)} problem(s)")
    sys.exit(1 if probs else 0)


if __name__ == "__main__":
    main()
