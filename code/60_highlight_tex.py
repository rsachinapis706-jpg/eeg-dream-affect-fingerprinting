"""
Build a highlighted copy of the manuscript for the resubmission upload.

IEEE Access asks for a version in which the changes are visible. The revision
added roughly fourteen thousand words, so marking every altered line would turn
most of the document one colour and tell a reader nothing. This script marks
prose that is new or rewritten, and leaves structural material -- tabular rows,
equation bodies, float preambles -- untouched, because colour commands inside
those break compilation for no gain. New tables and figures are signalled by
colouring their captions, which is where a reader looks first anyway.

Output: accesstemp_highlighted.tex, compilable with the same command as the
clean file.
"""
import io
import os
import re
import difflib

TEXDIR = os.path.join("E:" + os.sep, "Dream_Analysis")
OLD = os.path.join(TEXDIR, "access.tex")
NEW = os.path.join(TEXDIR, "accesstemp.tex")
OUT = os.path.join(TEXDIR, "accesstemp_highlighted.tex")

# Environments whose interior must not be wrapped in a colour group.
UNSAFE_ENV = re.compile(
    r"\\begin\{(tabular|tabularx|equation|align|gather|array|"
    r"eqnarray|displaymath|multline|split|cases)\*?\}")
UNSAFE_END = re.compile(
    r"\\end\{(tabular|tabularx|equation|align|gather|array|"
    r"eqnarray|displaymath|multline|split|cases)\*?\}")

# Lines that are pure structure: highlighting them is meaningless or unsafe.
STRUCTURAL = re.compile(
    r"^\s*(\\(begin|end|centering|small|normalsize|footnotesize|toprule|"
    r"midrule|bottomrule|botrule|cmidrule|addlinespace|includegraphics|"
    r"label|vspace|hspace|usepackage|documentclass|markboth|maketitle|"
    r"tableofcontents|bibliography|EOD|newpage|clearpage|hline|"
    r"footnotetext|multicolumn|columnbreak|appendices|IEEE|def|newcommand|"
    r"setlength|renewcommand|history|doi|corresp|received|title|author|"
    r"address|thanks|editor|tfootnote)|%|\{\\)")

# Marking starts at the abstract. The title block above it is a run of
# multi-line command arguments where a colour group cannot be closed on the
# same line; the title change is called out in the covering note instead.
BODY_HOOK = "\\begin{abstract}"
PREAMBLE_HOOK = "\\begin{document}"

MACRO = r"""
% ---- revision highlighting (resubmission copy only) -------------------------
\usepackage{xcolor}
\definecolor{revcolour}{RGB}{0,70,190}
\newcommand{\rev}[1]{{\color{revcolour}#1}}
% -----------------------------------------------------------------------------
"""


ARG_CMD = re.compile(
    r"\\(caption|section|subsection|subsubsection|paragraph)\*?\{")


def arg_wrap(line):
    """Colour the argument of a sectioning or caption command, in place.

    These commands must not be swallowed into a macro argument -- \\caption is
    fragile and sectioning commands write to the table of contents -- so the
    colour goes inside the braces instead. Returns None if the argument does
    not close on this line.
    """
    m = ARG_CMD.search(line)
    if not m:
        return None
    start = m.end()
    j, depth = start, 1
    while j < len(line) and depth:
        if line[j] == "{":
            depth += 1
        elif line[j] == "}":
            depth -= 1
        j += 1
    if depth:
        return None
    return line[:start] + "\\rev{" + line[start:j - 1] + "}" + line[j - 1:]


def item_wrap(line):
    """Colour a list item's text, leaving \\item itself outside the group."""
    m = re.match(r"^(\s*\\item\s*)(.*)$", line)
    if not m or not m.group(2).strip():
        return None
    return m.group(1) + "\\rev{" + m.group(2) + "}"


def main():
    old = io.open(OLD, encoding="utf-8").read().split("\n")
    new = io.open(NEW, encoding="utf-8").read().split("\n")

    changed = set()
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            changed.update(range(j1, j2))

    depth = 0
    out = []
    n_prose = n_cap = n_skip = 0
    in_body = False

    for k, line in enumerate(new):
        if line.strip().startswith(BODY_HOOK):
            in_body = True
        if UNSAFE_ENV.search(line):
            depth += 1
        opening = depth

        emit = line
        if k in changed and in_body and line.strip():
            if ARG_CMD.search(line) or re.match(r"^\s*\\item\b", line):
                # These commands take the colour inside their argument rather
                # than being swallowed into one.
                w = arg_wrap(line) or item_wrap(line)
                if w:
                    emit = w
                    n_cap += 1
                else:
                    n_skip += 1
            elif depth == 0 and not STRUCTURAL.match(line):
                # A whole paragraph sits on one line in this source, so a
                # single colour group per line is both safe and complete.
                emit = "\\rev{" + line + "}"
                n_prose += 1
            else:
                n_skip += 1

        out.append(emit)
        if UNSAFE_END.search(line) and opening:
            depth -= 1

    text = "\n".join(out)
    if "\\definecolor{revcolour}" not in text:
        idx = text.find(PREAMBLE_HOOK)
        assert idx > 0, "\\begin{document} not found"
        text = text[:idx] + MACRO + "\n" + text[idx:]

    io.open(OUT, "w", encoding="utf-8", newline="").write(text)

    print(f"changed lines in diff : {len(changed)}")
    print(f"prose paragraphs marked: {n_prose}")
    print(f"captions marked        : {n_cap}")
    print(f"structural, left plain : {n_skip}")
    print(f"written                : {OUT}")


if __name__ == "__main__":
    main()
