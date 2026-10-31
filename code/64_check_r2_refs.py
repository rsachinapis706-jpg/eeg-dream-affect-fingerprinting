"""
Verify the eight references suggested by Reviewer 2 in round two.

Each is looked up on Crossref by title and first author. A match is accepted
only if the returned title is close to the one the reviewer gave, and the
record's own metadata -- authors, venue, year, volume, pages, DOI -- is what
would go into the bibliography. Where Crossref carries an abstract it is printed,
because relevance is judged from what the paper actually does, not its title.

Nothing is added to references.bib here; this only establishes the facts.
"""
import difflib
import json
import os
import re
import time
import urllib.parse
import urllib.request

API = "https://api.crossref.org/works"
MAILTO = __import__("os").environ.get("CROSSREF_MAILTO", "")  # optional contact for the Crossref polite pool
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out",
                   "r2_suggested_refs.json")

SUGGESTED = [
    ("Numerical methods and computational algorithms for ECG signal processing "
     "and cardiac abnormality detection: A review", "Siddiqui"),
    ("Advanced signal processing algorithm-based state estimation and denoising "
     "of ECG signals for biomedical applications", "Arif"),
    ("Advanced AI-based early warning framework for ICU patient monitoring "
     "using temporal clinical data", "Siddiqui"),
    ("The extraction of sleep stages using EEG signals from different channel",
     "Siddiqui"),
    ("Smart diagnostic system for cardiovascular, neural, and muscular health "
     "using AI-based ECG, EEG, and EMG processing", "Siddiqui"),
    ("AI-integrated diagnostic models for neurological disorders: Current "
     "trends, technologies, and future directions", "Siddiqui"),
    ("AI-driven hybrid detection and classification framework for secure sleep "
     "health IoT networks", "Valsalan"),
    ("3D diffusion modeling for volumetric medical image super-resolution",
     "Qadir"),
]


def clean(s):
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def query(title, author):
    q = urllib.parse.urlencode({"query.bibliographic": title,
                                "query.author": author, "rows": 5,
                                "mailto": MAILTO})
    req = urllib.request.Request(
        f"{API}?{q}", headers={"User-Agent": f"refcheck/1.0 ({MAILTO})"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["message"].get("items", [])


def main():
    found = []
    for i, (title, author) in enumerate(SUGGESTED, 1):
        print("=" * 78)
        print(f"[{i}] {title}")
        try:
            items = query(title, author)
        except Exception as e:                          # noqa: BLE001
            print(f"    lookup failed: {e}")
            found.append(dict(n=i, title=title, status="lookup failed"))
            continue
        best, score = None, 0.0
        for it in items:
            t = (it.get("title") or [""])[0]
            s = difflib.SequenceMatcher(None, clean(t), clean(title)).ratio()
            if s > score:
                best, score = it, s
        if not best or score < 0.85:
            print(f"    NOT FOUND on Crossref (best similarity {score:.2f})")
            found.append(dict(n=i, title=title, status="not on Crossref"))
            time.sleep(0.7)
            continue
        auth = "; ".join(f"{a.get('given', '')} {a.get('family', '')}".strip()
                         for a in best.get("author", []))
        yr = (best.get("issued", {}).get("date-parts") or [[None]])[0][0]
        rec = dict(n=i, status="found", similarity=round(score, 3),
                   title=(best.get("title") or [""])[0], authors=auth,
                   venue=(best.get("container-title") or [""])[0],
                   year=yr, volume=best.get("volume"), issue=best.get("issue"),
                   pages=best.get("page") or best.get("article-number"),
                   doi=best.get("DOI"), type=best.get("type"),
                   abstract=clean(best.get("abstract", ""))[:900])
        found.append(rec)
        for k in ("title", "authors", "venue", "year", "volume", "issue",
                  "pages", "doi", "type"):
            print(f"    {k:<9}: {rec[k]}")
        print(f"    abstract : {rec['abstract'] or '(none on Crossref)'}")
        time.sleep(0.7)

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(found, fh, indent=1, ensure_ascii=False)
    print()
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
