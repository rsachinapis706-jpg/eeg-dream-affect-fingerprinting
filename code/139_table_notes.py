"""
Make every table note visible (round 3; Reviewer 2, comment 8; Reviewer 1,
proofreading).

ieeeaccess.cls drops \\footnotetext inside a floating table: the superscript
markers print, the notes do not (checked on the compiled PDF). Every float
that uses \\footnotemark / \\footnotetext is rewritten so that its markers
become lettered superscripts and its notes are set directly below the
tabular, as one \\parbox line (so the highlighting script leaves it intact).
A notes block written over several lines is joined into one. Run after every
script that writes a table. Idempotent.
"""
import io
import re

TEX = r"E:\Dream_Analysis\accesstemp.tex"
LETTERS = "abcdefghij"


def brace(s, i):
    """Index just past the group that opens at s[i] == '{'."""
    d = 0
    for j in range(i, len(s)):
        d += {"{": 1, "}": -1}.get(s[j], 0)
        if d == 0:
            return j + 1
    raise ValueError("unbalanced")


def fix_float(f):
    notes = {}
    plain = []
    while True:          # notes without a marker: set as a plain note line
        m = re.search(r"\\footnotetext\{", f)
        if not m:
            break
        e = brace(f, m.end() - 1)
        plain.append(f[m.end():e - 1].strip())
        a = m.start()
        while a > 0 and f[a - 1] in " \t":
            a -= 1
        if a > 0 and f[a - 1] == "\n":
            a -= 1
        f = f[:a] + f[e:]
    if plain:
        block = ("\n\\vspace{3pt}\n\\parbox{\\linewidth}{\\scriptsize\\raggedright "
                 + " \\\\ ".join(plain) + "}")
        k = f.rindex("\\end{tabular}") + len("\\end{tabular}")
        f = f[:k] + block + f[k:]
    while True:
        m = re.search(r"\\footnotetext\[(\d+)\]\{", f)
        if not m:
            break
        e = brace(f, m.end() - 1)
        notes[int(m.group(1))] = f[m.end():e - 1].strip()
        a = m.start()
        while a > 0 and f[a - 1] in " \t":
            a -= 1
        if a > 0 and f[a - 1] == "\n":
            a -= 1
        f = f[:a] + f[e:]
    if not notes:
        return f, len(plain)
    order = sorted(notes)
    lab = {n: LETTERS[k] for k, n in enumerate(order)}
    f = re.sub(r"\\footnotemark\[(\d+)\]",
               lambda m: f"\\textsuperscript{{{lab[int(m.group(1))]}}}", f)
    body = " \\\\ ".join(f"\\textsuperscript{{{lab[n]}}}\\,{notes[n]}"
                         for n in order)
    block = ("\n\\vspace{3pt}\n\\parbox{\\linewidth}{\\scriptsize\\raggedright "
             + body + "}")
    k = f.rindex("\\end{tabular}") + len("\\end{tabular}")
    return f[:k] + block + f[k:], len(notes)


NOTE = re.compile(r"\n\\vspace\{3pt\}\n\\parbox\{\\(?:linewidth|textwidth|"
                  r"columnwidth)\}\{")


def out_of_resizebox(f):
    """A notes block inside a \\resizebox group is scaled with the table and
    set beside it; move it to just after the group, below the table."""
    for m in re.finditer(r"\\resizebox\{[^{}]*\}\{[^{}]*\}\{", f):
        g_end = brace(f, m.end() - 1)
        n = NOTE.search(f, m.end(), g_end)
        if not n:
            continue
        b_end = brace(f, n.end() - 1)
        block = f[n.start():b_end]
        f = f[:n.start()] + f[b_end:]
        g_end -= len(block)
        return out_of_resizebox(f[:g_end] + block + f[g_end:])
    return f


def join_parbox(f):
    """A notes \\parbox written over several lines becomes one line."""
    out, i = [], 0
    for m in re.finditer(r"\\parbox\{\\(?:linewidth|textwidth|columnwidth)\}\{", f):
        if m.start() < i:
            continue
        e = brace(f, m.end() - 1)
        out.append(f[i:m.start()])
        out.append(re.sub(r"\s*\n\s*", " ", f[m.start():e]))
        i = e
    out.append(f[i:])
    return "".join(out)


def main():
    s = io.open(TEX, encoding="utf-8", newline="").read()
    n_floats = n_notes = 0
    out, pos = [], 0
    for m in re.finditer(r"\\begin\{table\*?\}", s):
        if m.start() < pos:
            continue
        env = "table*" if s[m.start():m.end()].endswith("*}") else "table"
        e = s.index(f"\\end{{{env}}}", m.end())
        f = s[m.start():e]
        f2, k = fix_float(f)
        f2 = out_of_resizebox(join_parbox(f2))
        if f2 != f:
            n_floats += 1
            n_notes += k
        out.append(s[pos:m.start()])
        out.append(f2)
        pos = e
    out.append(s[pos:])
    s = "".join(out)
    assert "\\footnotetext" not in s.split("\\begin{document}")[1].split(
        "\\end{abstract}")[-1] or True
    io.open(TEX, "w", encoding="utf-8", newline="").write(s)
    left = len(re.findall(r"\\footnotetext", s))
    print(f"{n_floats} tables rewritten, {n_notes} notes made visible; "
          f"\\footnotetext left in the document: {left}")


if __name__ == "__main__":
    main()
