"""
Keyed provenance table for every reported number (round 3, Reviewer 2,
comment 7).

95_audit.py checks that each decimal in the manuscript equals, at the printed
precision, some value written by an analysis script. That shows the number
exists in the results, not that it belongs to the right dataset, task,
model, partition, metric and aggregation. This script assigns each number a
key -- result file, the identifying (non-numeric) fields of its row, and its
column -- and records how specific that assignment is:

  unique      exactly one summary-level result cell matches the printed value
  keyed       several cells match; the key is the one whose file name, row
              fields and column share the most (stemmed) words with the
              section title and the sentence the number appears in, with a
              boost for files and rows keyed for other numbers on the same
              line, and for files keyed in the same table or paragraph; it
              is asserted only if it shares at least two words and scores
              strictly above the runner-up
  ambiguous   several cells match and none is clearly best: the best and
              the next three candidates are listed, and no key is claimed
  setting or threshold, plot coordinate, chance level
              values that are not results, recognised from the words around
              them (alpha = 0.05, max-softmax > 0.5, an axis tick, 1/3)
  quoted      a figure reported by a cited study, not computed here
  unlocated   no result cell matches (listed for review)

Every manuscript number was then read in context, and where the automatic
key was wrong, missing or not a single cell, provenance_overrides.py gives
the decision; each is recomputed here and must reproduce the printed value:

  keyed (checked by hand)   file, row query and column
  derived                   an expression over result cells (a difference,
                            ratio, mean or bound computed in the text)
  from analysis log         a figure an analysis script printed to its log

The same is done for the response letter (automatic matching only). Outputs:
  out/provenance_manuscript.csv, out/provenance_letter.csv,
  out/provenance_summary.csv
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
LETTER = os.environ.get("PROV_LETTER", "")
NUM = re.compile(r"(?<![\w.])[-+\u2212]?\d+\.\d+(?!\w|\.\d)")
SKIP = ("final_audit", "stale_", "provenance_", "letter_audit",
        "pcd_keyed_update")     # the last is 136's log of edits, not a result
# files of individual draws (permutations, simulations, partitions, folds,
# training epochs, per-recording rows): thousands of values, so a printed
# number matches some of them by coincidence. They are candidates only when
# no summary-level file holds the number.
DRAWS = re.compile(r"(_raw(_full)?|_curves|_sims|_draws|_folds|_participants)"
                   r"\.csv$|^(pcd_leaky_conditions|deed_alignment)\.csv$")
# superseded or partial versions of a result file; never a source
SUPERSEDED = (".pre_fix", ".prev", ".B100", ".B200", "_smoke", "_partial",
              "_discarded")
# figures quoted from the cited studies, not results of this work
QUOTED = {"0.9233", "0.9383", "93.83", "92.33", "0.862", "92", "94"}
STOP = set("the of and a an in on at to by for with from is are was were as "
           "or that this these its it be than over per each all".split())
# abbreviations used in file and column names
ABBR = {"bal": "balanced", "acc": "accuracy", "cov": "coverage",
        "msp": "softmax", "perm": "permutation", "lr": "logistic",
        "logreg": "logistic", "rf": "forest", "sd": "deviation",
        "se": "error", "ci": "interval", "lo": "interval", "hi": "interval",
        "nmi": "information", "fpr": "rate", "deed": "deed"}
# a key is asserted only when its match is at least this strong and strictly
# stronger than the runner-up's; otherwise the candidates are listed
MIN_WORDS = 2
LOCAL_BOOST = 3          # same source file as a keyed number on the same line
BLOCK_BOOST = 2          # ... in the same table or paragraph
ROW_BOOST = 3            # same result row as a keyed number on the same line
sys.path.insert(0, HERE)


def words(s):
    """Content words, abbreviations expanded, stemmed to five letters."""
    out = set()
    for w in re.findall(r"[a-z]+", s.lower().replace("_", " ")):
        w = ABBR.get(w, w)
        if (len(w) > 2 or w in ABBR.values()) and w not in STOP:
            out.add(w[:5])
    return out


def index_results():
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "**", "*.csv"), recursive=True)):
        b = os.path.basename(f)
        if b.startswith(SKIP) or any(x in b for x in SUPERSEDED):
            continue
        # a versioned copy (name.<tag>.csv) of a file that exists is an
        # earlier version of that file, not a source
        if b.count(".") > 1 and os.path.exists(
                os.path.join(os.path.dirname(f), b.split(".")[0] + ".csv")):
            continue
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if d.empty:
            continue
        keycols = [c for c in d.columns if d[c].dtype == object][:4]
        num = d.select_dtypes("number")
        for i in range(len(d)):
            key = "; ".join(f"{c}={d.iloc[i][c]}" for c in keycols)
            for c in num.columns:
                v = num.iloc[i][c]
                if pd.notna(v) and np.isfinite(v):
                    rows.append((float(v), b, key, str(c)))
    t = pd.DataFrame(rows, columns=["value", "file", "row_key", "column"])
    t["words"] = [words(f"{a} {b} {c}") for a, b, c in
                  zip(t.file, t.row_key, t.column)]
    return t.sort_values("value").reset_index(drop=True)


def candidates(t, x, dec):
    tol = 0.5 * 10 ** (-dec) + 1e-9
    v = t.value.to_numpy()
    out = []
    for target in (x, -x):
        i, j = np.searchsorted(v, target - tol), np.searchsorted(v, target + tol,
                                                                  side="right")
        out.append(t.iloc[i:j])
    return pd.concat(out).drop_duplicates(["file", "row_key", "column"])


SOURCE = re.compile(r"source=([\w.\-]+\.csv)")
PLOT = re.compile(r"\\(addplot|draw|includegraphics)|\b[xy](min|max|tick)\s*="
                  r"|legend style")


def local_kind(text, m):
    """Numbers that are not results, recognised from the words just before
    or after them: a setting or threshold (alpha = 0.05, max-softmax > 0.5,
    p < 0.05, C = 0.1, a band edge in Hz, a coverage level of a grid), a
    plot coordinate, or the chance level 1/3. Empty for anything else."""
    pre, post = text[max(0, m.start() - 60):m.start()], text[m.end():m.end() + 20]
    p = pre[-30:]
    if re.search(r"(\\alpha|\\eta|\bC)\s*=\s*\$?\s*$", p) \
            or re.search(r"(>|<|\\ge|\\le|\\geq|\\leq)\s*\$?\s*$", p) \
            or re.search(r"(threshold of|tolerance|coverages?)\s*$", p, re.I):
        return "setting or threshold"
    if (post.startswith("--") or re.search(r"--\s*$", p)) and re.search(
            r"Hz|delta|theta|alpha|sigma|beta|gamma|bandpass|band-pass",
            pre + post):
        return "setting or threshold"
    if PLOT.search(text) and re.search(
            r"(\(\s*|axis cs:\s*|(min|max)=\s*|tick=\{[^}]*|width=\s*)$",
            pre):
        return "plot coordinate"
    # balanced accuracy 1/3 is chance for three classes, and what any
    # constant (e.g. majority-class) prediction scores, by construction
    if m.group() == "0.3333":
        return "chance level"
    # chance for a k-way identity task (1/34, 1/27, 1/18 ...)
    x = float(m.group().replace("−", "-"))
    dec = len(m.group().split(".")[1])
    if re.search(r"chance", pre + post, re.I) and any(
            abs(round(1 / k, dec) - x) < 1e-12 for k in range(2, 101)):
        return "chance level"
    return ""


def collapse(c):
    """One candidate per quantity. A row that names its source file
    (source=<file>) is a copy of a cell of that file and is dropped when that
    file is also a candidate; cells with the same non-empty row key and
    column in several files (a table merged from its parts) are one
    quantity, kept once with the other files listed under `copies`."""
    if c.empty:
        return c.assign(copies="")
    draw = c.file.map(lambda f: DRAWS.search(f) is not None)
    if (~draw).any():
        c = c[~draw.to_numpy()]
    src = c.row_key.str.extract(SOURCE)[0]
    c = c[~src.isin(set(c.file)).to_numpy()]
    q = np.where(c.row_key.str.strip() != "", c.row_key + "\x00" + c.column,
                 c.file + "\x00" + c.row_key + "\x00" + c.column)
    c = c.assign(q=q)
    copies = c.groupby("q").file.agg(lambda s: "; ".join(sorted(s)[1:]))
    c = (c.sort_values("file").drop_duplicates("q")
         .assign(copies=lambda d: d.q.map(copies)).drop(columns="q"))
    # columns of one row holding the same value (e.g. accuracy on all and on
    # real test points when nothing is synthetic) identify one row: kept
    # once, with the equal columns joined
    keyed = c.row_key.str.strip().ne("")
    if keyed.any():
        k = c[keyed.to_numpy()]
        cols = k.groupby(["file", "row_key"]).column.agg(
            lambda s: " = ".join(sorted(s)))
        k = k.drop_duplicates(["file", "row_key"]).copy()
        k["column"] = [cols[(f, r)] for f, r in zip(k.file, k.row_key)]
        k["words"] = [words(f"{f} {r} {col}") for f, r, col in
                      zip(k.file, k.row_key, k.column)]
        c = pd.concat([k, c[~keyed.to_numpy()]])
    return c


def assign(t, items, settings):
    """Numbers with a unique or clearly best candidate are keyed first; then,
    until nothing changes, the files keyed for other numbers on the same line
    (a table row or sentence draws on one result file) and, more weakly, in
    the same table or paragraph, favour those files' cells."""
    cand = []
    for where, section, txt, ctx, block, kind in items:
        x, dec = float(txt.replace("\u2212", "-")), len(txt.split(".")[1])
        c = collapse(candidates(t, x, dec))
        if c.empty and ctx[ctx.find(txt) + len(txt):][:2].startswith(("\\%", "%")):
            c = collapse(candidates(t, x / 100, dec + 2))
        cw = words(f"{section} {ctx}")
        sc = (c.words.apply(lambda w: len(w & cw)).to_numpy() if len(c)
              else np.array([]))
        cand.append((c, sc))

    def pick(c, sc):
        order = np.argsort(-sc, kind="stable")
        best = c.iloc[order[0]]
        run = c.iloc[order[1]] if len(c) > 1 else None
        clear = len(c) == 1 or (sc[order[0]] >= MIN_WORDS
                                and sc[order[0]] > sc[order[1]])
        return best, run, order, clear

    def files_of(c):
        """Each candidate's file and the files holding identical copies."""
        return [set([f] + [x for x in cp.split("; ") if x])
                for f, cp in zip(c.file, c.copies)]

    def boosted(c, sc, line_files, block_files, line_rows=frozenset()):
        fs = files_of(c)
        return (sc + LOCAL_BOOST * np.array([bool(f & line_files) for f in fs])
                + BLOCK_BOOST * np.array([bool(f & block_files) for f in fs])
                + ROW_BOOST * np.array([(f, r) in line_rows for f, r in
                                        zip(c.file, c.row_key)]))

    by_line, by_block, by_row = {}, {}, {}
    for _ in range(6):
        new_line, new_block, new_row = {}, {}, {}
        for (where, _s, _t, _c, block, kind), (c, sc) in zip(items, cand):
            if not len(c) or kind:
                continue
            b = boosted(c, sc, by_line.get(where, set()),
                        by_block.get(block, set()), by_row.get(where, set()))
            best, _r, _o, clear = pick(c, b)
            if clear:
                new_line.setdefault(where, set()).add(best.file)
                new_block.setdefault(block, set()).add(best.file)
                if best.row_key.strip():
                    new_row.setdefault(where, set()).add(
                        (best.file, best.row_key))
        if (new_line, new_block, new_row) == (by_line, by_block, by_row):
            break
        by_line, by_block, by_row = new_line, new_block, new_row
    keyed = by_line
    rows = []
    for (where, section, txt, ctx, block, kind), (c, sc) in zip(items, cand):
        base = dict(location=where, section=section, value=txt,
                    context=ctx.strip())
        if txt in QUOTED:
            rows.append(dict(base, status="quoted from cited work",
                             n_candidates=0))
            continue
        if kind:
            rows.append(dict(base, status=kind, n_candidates=len(c)))
            continue
        if c.empty:
            st = "setting" if settings.search(ctx) else "unlocated"
            rows.append(dict(base, status=st, n_candidates=0))
            continue
        sc = boosted(c, sc, keyed.get(where, set()), by_block.get(block, set()),
                     by_row.get(where, set()))
        best, run, order, clear = pick(c, sc)
        if len(c) == 1:
            st = "unique"
        elif clear:
            st = "keyed"
        else:
            st = "ambiguous"
        alts = "; ".join(f"{r.file} | {r.row_key} | {r.column}"
                         for r in (c.iloc[k] for k in order[1:4]))
        rows.append(dict(base, status=st, n_candidates=len(c),
                         source_file=best.file, row_key=best.row_key,
                         column=best.column, identical_copies=best.copies,
                         match_score=int(sc[order[0]]),
                         runner_up_score=(int(sc[order[1]]) if len(c) > 1
                                          else ""),
                         other_candidates=alts))
    return pd.DataFrame(rows)


def manuscript_items():
    s = io.open(TEX, encoding="utf-8").read()
    start = s.index("\\begin{document}")
    end = s.find("\\bibliography{")
    lines = s[:end].split("\n")
    section, items = "", []
    # block: a table or figure environment, otherwise a paragraph (lines
    # between blank lines)
    block, in_float = 0, False
    for ln in range(s[:start].count("\n"), len(lines)):
        line = lines[ln]
        if re.match(r"\s*\\begin\{(table|figure)\*?\}", line):
            block, in_float = block + 1, True
        elif not in_float and not line.strip():
            block += 1
        if re.match(r"\s*\\end\{(table|figure)\*?\}", line):
            in_float = False
        m = re.search(r"\\(sub)*section\*?\{([^}]*)\}", line)
        if m:
            section = m.group(2)
        if line.lstrip().startswith("%"):
            continue
        for m in NUM.finditer(line):
            ctx = line[max(0, m.start() - 160):m.end() + 80]
            items.append((f"line {ln + 1}", section, m.group(), ctx,
                          f"block {block}", local_kind(line, m)))
    return items


def letter_items(path):
    import docx
    d = docx.Document(path)
    paras = [p.text for p in d.paragraphs]
    for tb in d.tables:
        for r in tb.rows:
            paras += [c.text for c in r.cells]
    items, concern, quoting = [], "", False
    for k, p in enumerate(paras):
        if p.startswith("Reviewer#") or p.startswith("Reviewer "):
            concern, quoting = p[:60], True
        elif p.startswith("Author response:"):
            quoting = False
        for m in NUM.finditer(p):
            # the reviewer's own words (heading and comment paragraphs) quote
            # numbers from our previous submission; they are not claims here
            kind = ("quoted from the reviewer's comment" if quoting
                    else local_kind(p, m))
            items.append((f"paragraph {k + 1}", concern, m.group(),
                          p[max(0, m.start() - 160):m.end() + 80],
                          f"paragraph {k + 1}", kind))
    return items


def apply_overrides(ms, letter=False):
    """The decisions of provenance_overrides.py, each recomputed: a key must
    select one value equal to the printed number at its printed precision
    (sign aside: a contrast may be printed as a magnitude), a derived check
    must reproduce it, a log must contain the quoted line."""
    import provenance_overrides as po
    from scipy import stats
    ns = dict(vars(po), np=np, stats=stats)
    ms = ms.copy()
    ms["occ"] = ms.groupby(["location", "value"]).cumcount()
    for col in ("source_file", "row_key", "column", "identical_copies"):
        if col not in ms:
            ms[col] = ""
        ms[col] = ms[col].astype(object)
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    entries = po.letter_overrides() if letter else po.all_overrides()
    for o in entries:
        if letter:
            # by value and surrounding text, outside the reviewer's quotes
            sel = ((ms.value == o["value"])
                   & ms.context.str.contains(o["ctx"], regex=False)
                   & (ms.status != "quoted from the reviewer's comment"))
            assert sel.sum() >= 1, f"letter override not found: {o}"
        else:
            sel = ((ms.location == f"line {o['line']}")
                   & (ms.value == o["value"]) & (ms.occ == o["occ"]))
            assert sel.sum() == 1, f"override not found: {o}"
        txt = o["value"]
        dec = len(txt.split(".")[1])
        x = abs(float(txt.replace("−", "-")))
        tol = 0.5 * 10 ** (-dec) + 1e-9
        if o["kind"] in ("key", "derived"):
            v = (po.cell(o["file"], o["query"], o["column"]) * o["scale"]
                 if o["kind"] == "key" else float(eval(o["check"], ns)))
            assert abs(abs(v) - x) <= tol, f"{o}: computes {v}"
            new = (dict(status="keyed (checked by hand)", source_file=o["file"],
                        row_key=o["query"], column=o["column"])
                   if o["kind"] == "key" else
                   dict(status="derived", source_file="(computed in the text)",
                        row_key=o["check"], column=o["note"]))
        elif o["kind"] == "log":
            body = io.open(os.path.join(OUT, o["file"]), encoding="utf-8",
                           errors="replace").read()
            assert norm(o["text"]) in norm(body), f"{o}: not in log"
            new = dict(status="from analysis log", source_file=o["file"],
                       row_key=o["text"], column="")
        else:
            new = dict(status=o["status"], source_file="", row_key="",
                       column=o["note"])
        for k, v in new.items():
            ms.loc[sel, k] = v
        for k in ("identical_copies", "other_candidates"):
            if k in ms:
                ms.loc[sel, k] = ""
    return ms.drop(columns="occ")


def main():
    au_spec = __import__("importlib.util").util.spec_from_file_location(
        "au", os.path.join(HERE, "95_audit.py"))
    au = __import__("importlib.util").util.module_from_spec(au_spec)
    au_spec.loader.exec_module(au)
    t = index_results()
    ms = apply_overrides(assign(t, manuscript_items(), au.SETTING_CONTEXT))
    ms.to_csv(os.path.join(OUT, "provenance_manuscript.csv"), index=False)
    summ = [dict(document="manuscript", **ms.status.value_counts().to_dict())]
    if LETTER and os.path.exists(LETTER):
        lt = apply_overrides(assign(t, letter_items(LETTER),
                                    au.SETTING_CONTEXT), letter=True)
        lt.to_csv(os.path.join(OUT, "provenance_letter.csv"), index=False)
        summ.append(dict(document="response letter",
                         **lt.status.value_counts().to_dict()))
    sm = pd.DataFrame(summ).fillna(0)
    sm.to_csv(os.path.join(OUT, "provenance_summary.csv"), index=False)
    print(sm.to_string(index=False))
    un = ms[ms.status == "unlocated"]
    for _, r in un.iterrows():
        print(f"  UNLOCATED {r.location} {r.value}: ...{r.context[-120:]}")


if __name__ == "__main__":
    main()
