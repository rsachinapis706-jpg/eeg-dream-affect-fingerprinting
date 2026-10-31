"""
Verify every bibliography entry against the Crossref record.

Reviewer 2, minor comment 11, asks that all recent references be verified for
publication status, DOI and bibliographic detail. Doing this by eye is exactly
how errors survive, so it is done programmatically here and the raw comparison
is written to disk alongside the other result files.

Method
------
For each entry the title is queried against the Crossref works API. The
best-scoring returned record is compared field by field against what the .bib
claims: title similarity, first author surname, container (journal or
proceedings), volume, pages and year. Anything that disagrees is flagged for
manual checking rather than silently corrected -- an automated rewrite of a
bibliography is its own source of error.

Entries that Crossref does not index are reported as UNVERIFIED, not as wrong.
Preprint servers, some IEEE conference volumes and dataset DOIs registered with
DataCite rather than Crossref fall in this category and need a manual look.

Output
------
  out/reference_audit.csv   one row per entry, with the Crossref record
  out/test29_reference_audit.txt   readable report
"""
import os
import re
import csv
import json
import time
import difflib
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
BIB = os.path.join("E:" + os.sep, "Dream_Analysis", "references.bib")
API = "https://api.crossref.org/works"
MAILTO = __import__("os").environ.get("CROSSREF_MAILTO", "")  # optional contact for the Crossref polite pool
TIMEOUT = 30
PAUSE = 0.7

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def parse_bib(path):
    """Minimal BibTeX reader: enough for well-formed entries, no @string."""
    txt = open(path, encoding="utf-8").read()
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),", txt):
        kind, key = m.group(1).lower(), m.group(2).strip()
        start = m.end()
        depth, i = 1, m.start()
        i = txt.index("{", m.start())
        depth, j = 1, i + 1
        while j < len(txt) and depth:
            if txt[j] == "{":
                depth += 1
            elif txt[j] == "}":
                depth -= 1
            j += 1
        body = txt[start:j - 1]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            name = fm.group(1).lower()
            k = fm.end()
            while k < len(body) and body[k] in " \t\n":
                k += 1
            if k >= len(body):
                continue
            if body[k] == "{":
                d, e = 1, k + 1
                while e < len(body) and d:
                    if body[e] == "{":
                        d += 1
                    elif body[e] == "}":
                        d -= 1
                    e += 1
                fields[name] = body[k + 1:e - 1].strip()
            else:
                e = body.find(",", k)
                fields[name] = body[k:(e if e > 0 else len(body))].strip()
        entries.append(dict(key=key, kind=kind, **fields))
    return entries


def clean(s):
    s = re.sub(r"[{}\\]", "", s or "")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def crossref(title):
    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": 3,
                                "mailto": MAILTO})
    req = urllib.request.Request(f"{API}?{q}",
                                 headers={"User-Agent": f"refcheck/1.0 ({MAILTO})"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)["message"].get("items", [])


def sim(a, b):
    return difflib.SequenceMatcher(None, clean(a), clean(b)).ratio()


def first_surname(author_field):
    if not author_field:
        return ""
    first = author_field.split(" and ")[0]
    if "," in first:
        return clean(first.split(",")[0])
    return clean(first.split()[-1]) if first.split() else ""


def main():
    entries = parse_bib(BIB)
    log("=" * 78)
    log("BIBLIOGRAPHY VERIFICATION AGAINST CROSSREF")
    log("=" * 78)
    log(f"source: {BIB}")
    log(f"entries: {len(entries)}")
    log("")

    rows = []
    for i, e in enumerate(entries, 1):
        title = e.get("title", "")
        got, status, note = None, "UNVERIFIED", ""
        try:
            items = crossref(title)
            if items:
                best = max(items, key=lambda it: sim(title,
                                                     (it.get("title") or [""])[0]))
                s = sim(title, (best.get("title") or [""])[0])
                if s >= 0.85:
                    got, status = best, "MATCH"
                else:
                    note = f"best title similarity {s:.2f}"
        except Exception as ex:
            note = f"query failed: {type(ex).__name__}"
        time.sleep(PAUSE)

        row = dict(key=e["key"], bib_year=e.get("year", ""),
                   bib_doi=e.get("doi", ""), status=status, note=note,
                   cr_doi="", cr_year="", cr_container="", cr_volume="",
                   cr_page="", cr_type="", mismatches="")

        if got:
            cy = ""
            for f in ("published-print", "published-online", "issued"):
                if got.get(f, {}).get("date-parts", [[None]])[0][0]:
                    cy = str(got[f]["date-parts"][0][0]); break
            row.update(cr_doi=got.get("DOI", ""), cr_year=cy,
                       cr_container=(got.get("container-title") or [""])[0],
                       cr_volume=got.get("volume", ""),
                       cr_page=got.get("page", ""), cr_type=got.get("type", ""))

            bad = []
            if e.get("year") and cy and e["year"] != cy:
                bad.append(f"year bib={e['year']} crossref={cy}")
            if e.get("volume") and row["cr_volume"] and \
                    e["volume"] != row["cr_volume"]:
                bad.append(f"volume bib={e['volume']} crossref={row['cr_volume']}")
            if e.get("pages") and row["cr_page"]:
                if clean(e["pages"]).replace("--", "-") != clean(row["cr_page"]):
                    bad.append(f"pages bib={e['pages']} crossref={row['cr_page']}")
            if e.get("doi") and row["cr_doi"] and \
                    clean(e["doi"]) != clean(row["cr_doi"]):
                bad.append(f"doi bib={e['doi']} crossref={row['cr_doi']}")
            cont = e.get("journal") or e.get("booktitle") or ""
            if cont and row["cr_container"] and sim(cont, row["cr_container"]) < 0.6:
                bad.append(f"venue bib='{cont}' crossref='{row['cr_container']}'")
            sn = first_surname(e.get("author", ""))
            crs = ""
            if got.get("author"):
                crs = clean(got["author"][0].get("family", ""))
            if sn and crs and sn != crs:
                bad.append(f"first author bib={sn} crossref={crs}")
            row["mismatches"] = " | ".join(bad)
            if bad:
                row["status"] = "CHECK"

        rows.append(row)
        flag = {"MATCH": "  ok", "CHECK": "CHECK", "UNVERIFIED": " ??? "}[row["status"]]
        log(f"[{i:>2}/{len(entries)}] {flag}  {e['key']:<28} {row['cr_doi']}")
        if row["mismatches"]:
            for b in row["mismatches"].split(" | "):
                log(f"           - {b}")
        elif row["status"] == "UNVERIFIED" and note:
            log(f"           - {note}")

    log("")
    log("=" * 78)
    n_ok = sum(r["status"] == "MATCH" for r in rows)
    n_ck = sum(r["status"] == "CHECK" for r in rows)
    n_un = sum(r["status"] == "UNVERIFIED" for r in rows)
    log(f"verified clean          : {n_ok}")
    log(f"needs manual check      : {n_ck}")
    log(f"not indexed by Crossref : {n_un}")
    log("")
    if n_ck:
        log("CHECK entries (field disagreements):")
        for r in rows:
            if r["status"] == "CHECK":
                log(f"  {r['key']}: {r['mismatches']}")
    if n_un:
        log("")
        log("UNVERIFIED entries (confirm by hand -- preprints, some IEEE")
        log("proceedings and DataCite dataset DOIs are not in Crossref):")
        for r in rows:
            if r["status"] == "UNVERIFIED":
                log(f"  {r['key']}")

    with open(os.path.join(OUT, "reference_audit.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, "test29_reference_audit.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"\nWrote {OUT}\\reference_audit.csv")


if __name__ == "__main__":
    main()
