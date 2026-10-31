"""
Resolve the bibliography entries that 38_verify_refs.py could not confirm, and
fetch the records for the Sleep-EDF citations Reviewer 1 asked for.

Each entry is looked up by DOI where one is known, otherwise by title, and the
record's own metadata is printed. Nothing is written to references.bib here;
the corrections are applied by hand from this output so every change is seen.
"""
import json
import re
import time
import urllib.parse
import urllib.request

MAILTO = __import__("os").environ.get("CROSSREF_MAILTO", "")  # optional contact for the Crossref polite pool
H = {"User-Agent": f"refcheck/1.0 ({MAILTO})"}
API = "https://api.crossref.org/works"


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=H),
                                timeout=40) as r:
        return json.load(r)["message"]


def show(it, tag):
    auth = "; ".join(f"{a.get('given', '')} {a.get('family', '')}".strip()
                     for a in it.get("author", []))
    yr = (it.get("issued", {}).get("date-parts") or [[None]])[0]
    print(f"[{tag}]")
    print(f"   title  : {(it.get('title') or [''])[0]}")
    print(f"   authors: {auth[:300]}")
    print(f"   venue  : {(it.get('container-title') or [''])[0]} | "
          f"{it.get('publisher')} | issued {yr}")
    print(f"   vol {it.get('volume')} no {it.get('issue')} pp "
          f"{it.get('page') or it.get('article-number')} | type "
          f"{it.get('type')} | DOI {it.get('DOI')}")
    rel = it.get("relation", {})
    for k in ("is-preprint-of", "has-preprint"):
        if k in rel:
            print(f"   {k}: {[x.get('id') for x in rel[k]]}")


def by_doi(doi, tag):
    try:
        show(get(f"{API}/{urllib.parse.quote(doi)}"), tag)
    except Exception as e:                                # noqa: BLE001
        print(f"[{tag}] DOI {doi} did not resolve on Crossref: {e}")
    time.sleep(0.7)


def by_title(title, tag, author=None, rows=4):
    q = {"query.bibliographic": title, "rows": rows, "mailto": MAILTO}
    if author:
        q["query.author"] = author
    try:
        items = get(f"{API}?{urllib.parse.urlencode(q)}").get("items", [])
    except Exception as e:                                # noqa: BLE001
        print(f"[{tag}] lookup failed: {e}")
        return
    want = re.sub(r"[^a-z0-9]", "", title.lower())[:60]
    hit = [it for it in items
           if re.sub(r"[^a-z0-9]", "", (it.get("title") or [""])[0].lower())
           [:60] == want]
    if not hit:
        print(f"[{tag}] no exact title match; nearest:")
        for it in items[:rows]:
            print(f"     - {(it.get('title') or [''])[0][:100]} "
                  f"[{it.get('DOI')}] {it.get('type')}")
    for it in hit:
        show(it, tag)
    time.sleep(0.7)


# ---- entries flagged or unindexed ------------------------------------------
by_title("DEED: a dataset for dream-related emotion research", "liu2022deed",
         "Liu")
by_title("BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals",
         "xiao2026brainomni")
by_title("A hybrid deep learning scheme for multi-channel sleep stage "
         "classification", "pei2021hybrid")
by_title("Automatic EEG-based dream-related emotion recognition using fuzzy "
         "entropy and efficient signal decomposition methods",
         "mojdehbar2026automatic")
by_title("Emotion state detection using EEG signals - A machine learning "
         "perspective", "naidu2024emotion")
by_title("Permutation tests for studying classifier performance",
         "ojala2010permutation", "Ojala")
by_doi("10.1561/2200000101", "angelopoulos2023gentle")
by_title("The DREAM database", "wong2023dream", "Wong")
by_title("A multi-dataset characterization of dream reports and EEG "
         "DREAM database", "wong2023dream (paper?)", "Andrillon")

# ---- Sleep-EDF, requested by Reviewer 1 -------------------------------------
by_doi("10.1109/10.867928", "kemp2000analysis (Sleep-EDF)")
by_doi("10.1161/01.CIR.101.23.e215", "goldberger2000physiobank (PhysioNet)")
