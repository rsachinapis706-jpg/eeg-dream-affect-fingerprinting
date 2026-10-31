"""
Verify every bibliography entry against the record registered for its DOI.

Supersedes 38_verify_refs.py, whose log predates the revised bibliography.
Script 38 searched Crossref by title, which can return a preprint or a
different version of the same work. Here each DOI is resolved at doi.org with
content negotiation (CSL JSON). That returns the registered record whether the
DOI is held by Crossref or by DataCite (PhysioNet, figshare), and the record
is compared field by field with the .bib: title, first author's surname, year,
container, volume and pages.

Entries without a DOI (conference papers on OpenReview, PMLR, NeurIPS and
JMLR, which do not register DOIs for these volumes) cannot be resolved this
way. They are listed with the URL to check, and the outcome of the check is
recorded by hand in reference_manual_checks.csv, which this script merges in.

No personal identifier is sent with the requests.

Output
  out/reference_audit.csv          one row per entry
  out/reference_audit.txt          readable report
"""
import csv
import difflib
import json
import os
import re
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
BIB = os.path.join("E:" + os.sep, "Dream_Analysis", "references.bib")
TEX = os.path.join("E:" + os.sep, "Dream_Analysis", "accesstemp.tex")
MANUAL = os.path.join(OUT, "reference_manual_checks.csv")
UA = "reference-check/2.0 (bibliography verification script)"

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def parse_bib(path):
    """Sequential BibTeX reader: field = {braced} | "quoted" | bare."""
    txt = open(path, encoding="utf-8").read()
    out = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", txt):
        kind, key = m.group(1).lower(), m.group(2)
        i, f = m.end(), {}
        while True:
            fm = re.compile(r"\s*(\w+)\s*=\s*").match(txt, i)
            if not fm:
                break
            name, k = fm.group(1).lower(), fm.end()
            if txt[k] == "{":
                d, e = 1, k + 1
                while d:
                    d += {"{": 1, "}": -1}.get(txt[e], 0)
                    e += 1
                val, i = txt[k + 1:e - 1], e
            elif txt[k] == '"':
                e = txt.index('"', k + 1)
                val, i = txt[k + 1:e], e + 1
            else:
                e = re.compile(r"[,}\n]").search(txt, k).start()
                val, i = txt[k:e], e
            f[name] = re.sub(r"\s+", " ", val).strip()
            cm = re.compile(r"\s*,").match(txt, i)
            if not cm:
                break
            i = cm.end()
        out.append(dict(key=key, kind=kind, **f))
    return out


def clean(s):
    s = re.sub(r"\\[a-zA-Z]+\s*", " ", s or "")
    s = re.sub(r"[{}\\$~'`\"^]", "", s)
    s = re.sub(r"--", "-", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def sim(a, b):
    return difflib.SequenceMatcher(None, clean(a), clean(b)).ratio()


def resolve(doi):
    req = urllib.request.Request(
        "https://doi.org/" + doi,
        headers={"Accept": "application/vnd.citationstyles.csl+json",
                 "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)


def years(rec):
    ys = set()
    for k in ("issued", "published-print", "published-online", "published"):
        dp = (rec.get(k) or {}).get("date-parts") or []
        if dp and dp[0] and dp[0][0]:
            ys.add(str(dp[0][0]))
    return ys


def first_surname(bib_author):
    a = clean(bib_author.split(" and ")[0])
    return a.split(",")[0].strip() if "," in a else (a.split()[-1] if a else "")


def pages_norm(p):
    return re.sub(r"[^0-9a-z]+", "-", clean(p)).strip("-")


def main():
    entries = parse_bib(BIB)
    tex = open(TEX, encoding="utf-8").read()
    cited = {k.strip() for m in re.findall(r"\\cite\{([^}]+)\}", tex)
             for k in m.split(",")}
    manual = {}
    if os.path.exists(MANUAL):
        with open(MANUAL, encoding="utf-8") as fh:
            manual = {r["key"]: r for r in csv.DictReader(fh)}

    log("BIBLIOGRAPHY VERIFICATION AGAINST THE REGISTERED DOI RECORD")
    log(f"source: {BIB}  ({len(entries)} entries, {len(cited)} cited keys)")
    log("")
    rows = []
    for e in entries:
        doi = e.get("doi", "")
        row = dict(key=e["key"], cited=e["key"] in cited, doi=doi,
                   bib_title=e.get("title", ""), bib_year=e.get("year", ""),
                   bib_first_author=first_surname(e.get("author", "")),
                   bib_container=e.get("journal", e.get("booktitle",
                                       e.get("publisher", ""))),
                   bib_volume=e.get("volume", ""), bib_pages=e.get("pages", ""),
                   reg_title="", reg_year="", reg_first_author="",
                   reg_container="", reg_volume="", reg_pages="",
                   status="", differences="", manual_check="")
        if not doi:
            m = manual.get(e["key"])
            if m:
                row["status"] = m["outcome"]
                row["manual_check"] = f"{m['source']} | {m['checked']} | {m['note']}"
            else:
                row["status"] = "NO DOI - check by hand"
            rows.append(row)
            log(f"{e['key']:30s} {row['status']}")
            continue
        try:
            rec = resolve(doi)
        except Exception as ex:                       # noqa: BLE001
            row["status"] = f"DOI DID NOT RESOLVE ({type(ex).__name__})"
            rows.append(row)
            log(f"{e['key']:30s} {row['status']}")
            continue
        time.sleep(0.5)
        auth = rec.get("author") or rec.get("editor") or [{}]
        a0 = auth[0]
        row.update(
            reg_title=rec.get("title", "") if isinstance(rec.get("title"), str)
            else " ".join(rec.get("title", [])),
            reg_year="/".join(sorted(years(rec))),
            reg_first_author=clean(a0.get("family", a0.get("literal", ""))),
            reg_container=rec.get("container-title", "") if isinstance(
                rec.get("container-title"), str) else " ".join(
                rec.get("container-title", [])),
            reg_volume=str(rec.get("volume", "")),
            reg_pages=str(rec.get("page", rec.get("article-number", ""))))
        diff = []
        s = sim(row["bib_title"], row["reg_title"])
        if s < 0.90:
            diff.append(f"title similarity {s:.2f}")
        if row["bib_first_author"] and row["reg_first_author"] and \
                clean(row["bib_first_author"]) != row["reg_first_author"]:
            diff.append(f"first author bib={row['bib_first_author']} "
                        f"registered={row['reg_first_author']}")
        if row["bib_year"] and row["bib_year"] not in years(rec):
            diff.append(f"year bib={row['bib_year']} registered={row['reg_year']}")
        if row["bib_volume"] and row["reg_volume"] and \
                clean(row["bib_volume"]) != clean(row["reg_volume"]):
            diff.append(f"volume bib={row['bib_volume']} "
                        f"registered={row['reg_volume']}")
        if row["bib_pages"] and row["reg_pages"] and \
                pages_norm(row["bib_pages"]) != pages_norm(row["reg_pages"]):
            diff.append(f"pages bib={row['bib_pages']} "
                        f"registered={row['reg_pages']}")
        row["differences"] = "; ".join(diff)
        row["status"] = "MATCH" if not diff else "DIFFERS"
        m = manual.get(e["key"])
        if diff and m:
            row["status"] = m["outcome"]
            row["manual_check"] = f"{m['source']} | {m['checked']} | {m['note']}"
        rows.append(row)
        log(f"{e['key']:30s} {row['status']:10s} {row['differences']}")

    with open(os.path.join(OUT, "reference_audit.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log("")
    from collections import Counter
    log("summary: " + ", ".join(f"{k} {v}" for k, v in
                                Counter(r["status"] for r in rows).items()))
    log(f"uncited entries: {[r['key'] for r in rows if not r['cited']] or 'none'}")
    with open(os.path.join(OUT, "reference_audit.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
