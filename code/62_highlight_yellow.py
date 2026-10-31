"""
Build the highlighted manuscript with a yellow background, as IEEE Access asks.

Yellow highlighting in LaTeX means soul's \\hl, which sets text character by
character and therefore breaks on anything it cannot walk through: inline math,
citation and reference macros, and the moving arguments of captions and
sectioning commands. Each of those is handled here rather than left to fail at
compile time:

  inline math      wrapped in \\mbox so soul steps over it as one box
  -- en dashes     wrapped likewise, which also preserves the ligature
  \\cite, \\ref etc  registered with soul so it walks through them safely
  captions         given an empty optional argument, so the fragile \\hl is
                   never written to the .lof/.lot auxiliary files
  section titles   given a plain optional argument for the table of contents
                   and running heads, with only the printed title highlighted

The blue-text version from 60_highlight_tex.py remains available; it uses only
\\color and cannot fail, and is the fallback if soul misbehaves on this class.
"""
import io
import os
import re
import difflib

TEXDIR = os.path.join("E:" + os.sep, "Dream_Analysis")
OLD = os.path.join(TEXDIR, "submitted_50879", "accesstemp.tex")  # as submitted, Access-2026-50879 (Final/ source zip, 5 Oct 2026)
NEW = os.path.join(TEXDIR, "accesstemp.tex")
OUT = os.path.join(TEXDIR, "accesstemp_highlighted.tex")

# Environments whose lines are code, not prose: TikZ/pgfplots and algorithms
# are included because \rev{} around their lines breaks them (pgfmath errors).
UNSAFE_ENV = re.compile(
    r"\\begin\{(tabular|tabularx|equation|align|gather|array|"
    r"eqnarray|displaymath|multline|split|cases|tikzpicture|axis|"
    r"algorithmic)\*?\}")
UNSAFE_END = re.compile(
    r"\\end\{(tabular|tabularx|equation|align|gather|array|"
    r"eqnarray|displaymath|multline|split|cases|tikzpicture|axis|"
    r"algorithmic)\*?\}")

STRUCTURAL = re.compile(
    r"^\s*(\\(begin|end|centering|small|normalsize|footnotesize|scriptsize|"
    r"tiny|toprule|"
    r"midrule|bottomrule|botrule|cmidrule|addlinespace|includegraphics|"
    r"label|vspace|hspace|usepackage|documentclass|markboth|maketitle|"
    r"tableofcontents|bibliography|EOD|newpage|clearpage|hline|"
    r"footnotetext|multicolumn|columnbreak|appendices|IEEE|def|newcommand|"
    r"setlength|renewcommand|history|doi|corresp|received|title|author|"
    r"address|thanks|editor|tfootnote|resizebox|raggedbottom|parbox)|%|\{\\|\}\s*$)")

SECT_CMD = re.compile(r"\\(section|subsection|subsubsection|paragraph)(\*?)\{")
CAPTION = re.compile(r"\\caption\{")

BODY_HOOK = "\\begin{abstract}"
PREAMBLE_HOOK = "\\begin{document}"

MACRO = r"""
% ---- revision highlighting (resubmission copy only) -------------------------
\usepackage{soul}
\usepackage{xcolor}
\sethlcolor{yellow}
% soul walks through text one character at a time and stops at macros it does
% not know. Registering the ones that appear inside highlighted prose lets it
% step over them intact.
% The second argument is a number: 7 passes the command and its argument
% through unchanged (citations, references), 1 marks a one-argument text
% command whose argument is highlighted, 0 a command without arguments.
\soulregister{\cite}{7}
\soulregister{\ref}{7}
\soulregister{\eqref}{7}
\soulregister{\textbf}{1}
\soulregister{\emph}{1}
\soulregister{\textit}{1}
\soulregister{\textsc}{1}
\soulregister{\texttt}{1}
\soulregister{\headeretal}{0}
\newcommand{\rev}[1]{\hl{#1}}
% -----------------------------------------------------------------------------
"""


def brace_arg(line, start):
    """Return (inner, end_index) for the braced group beginning at start."""
    j, depth = start, 1
    while j < len(line) and depth:
        if line[j] == "{":
            depth += 1
        elif line[j] == "}":
            depth -= 1
        j += 1
    return (None, None) if depth else (line[start:j - 1], j)


def soul_safe(text):
    """Box the constructs soul cannot walk through."""
    # inline math, longest-first so $$ pairs are not split
    text = re.sub(r"(?<!\\)\$([^$]+?)(?<!\\)\$", r"\\mbox{$\1$}", text)
    # en dash: boxing keeps the ligature that soul would otherwise break apart
    text = text.replace("--", "\\mbox{--}")
    return text


def strip_markup(text):
    """Plain text for a table-of-contents or list-of-figures entry."""
    t = re.sub(r"\\(textbf|emph|textit|textsc|texttt)\{([^{}]*)\}", r"\2", text)
    t = re.sub(r"\\label\{[^}]*\}", "", t)
    return t.strip()


def main():
    old = io.open(OLD, encoding="utf-8").read().split("\n")
    new = io.open(NEW, encoding="utf-8").read().split("\n")

    changed = set()
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(
            None, old, new, autojunk=False).get_opcodes():
        if tag in ("replace", "insert"):
            changed.update(range(j1, j2))

    depth = 0
    out = []
    n_prose = n_cap = n_sec = n_skip = 0
    in_body = False

    for k, line in enumerate(new):
        if line.strip().startswith(BODY_HOOK):
            in_body = True
        if UNSAFE_ENV.search(line):
            depth += 1
        opening = depth
        emit = line

        if k in changed and in_body and line.strip():
            m = SECT_CMD.search(line)
            c = CAPTION.search(line)
            if m:
                inner, end = brace_arg(line, m.end())
                if inner is not None:
                    # The optional argument carries a plain title to the table
                    # of contents and the running head; only what is printed on
                    # the page is highlighted.
                    plain = strip_markup(inner)
                    emit = (line[:m.start()]
                            + f"\\{m.group(1)}{m.group(2)}[{plain}]"
                            + "{\\rev{" + soul_safe(inner) + "}}"
                            + line[end:])
                    n_sec += 1
                else:
                    n_skip += 1
            elif c:
                inner, end = brace_arg(line, c.end())
                if inner is not None:
                    # An empty short caption keeps the fragile \hl out of the
                    # .lof and .lot files, where it would break the build.
                    emit = (line[:c.start()] + "\\caption[]{\\rev{"
                            + soul_safe(inner) + "}}" + line[end:])
                    n_cap += 1
                else:
                    n_skip += 1
            elif re.match(r"^\s*\\item\b", line):
                mm = re.match(r"^(\s*\\item\s*)(.*)$", line)
                if mm and mm.group(2).strip():
                    emit = mm.group(1) + "\\rev{" + soul_safe(mm.group(2)) + "}"
                    n_prose += 1
                else:
                    n_skip += 1
            elif depth == 0 and not STRUCTURAL.match(line):
                # \noindent is a paragraph command; it stays outside the
                # highlight, which holds text only
                lead = "\\noindent" if line.startswith("\\noindent") else ""
                emit = lead + "\\rev{" + soul_safe(line[len(lead):]) + "}"
                n_prose += 1
            else:
                n_skip += 1

        out.append(emit)
        if UNSAFE_END.search(line) and opening:
            depth -= 1

    text = "\n".join(out)
    idx = text.find(PREAMBLE_HOOK)
    assert idx > 0, "\\begin{document} not found"
    text = text[:idx] + MACRO + "\n" + text[idx:]

    io.open(OUT, "w", encoding="utf-8", newline="").write(text)
    print(f"prose paragraphs highlighted : {n_prose}")
    print(f"captions highlighted         : {n_cap}")
    print(f"section titles highlighted   : {n_sec}")
    print(f"structural, left plain       : {n_skip}")
    print(f"written                      : {OUT}")


if __name__ == "__main__":
    main()
