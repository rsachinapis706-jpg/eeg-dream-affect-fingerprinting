"""
Find and verify candidate references for the expanded literature review.

Reviewer 1 comment 3 and Reviewer 2 major comment 5 both ask for a broader
survey: subject-dependent versus subject-independent EEG evaluation, participant
identification from biosignals, and data leakage in biomedical machine learning
including preprocessing, feature-selection and resampling leakage.

Nothing is cited from memory. Each candidate is looked up in Crossref and only
the fields Crossref returns are recorded, so every entry that reaches the
bibliography has a verifiable record behind it. Queries that return no
sufficiently similar title are reported as NOT FOUND rather than guessed at.

Output: out/candidate_refs.csv and a ready-to-paste BibTeX block built only
from returned metadata.
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
API = "https://api.crossref.org/works"
MAILTO = __import__("os").environ.get("CROSSREF_MAILTO", "")  # optional contact for the Crossref polite pool

# (search string, topic tag) -- the topic tag records why it is wanted
QUERIES = [
    ("Leakage and the reproducibility crisis in machine-learning-based science",
     "leakage"),
    ("The need to approximate the use-case in clinical machine learning",
     "subject-wise"),
    ("Cross-validation failure: small sample sizes lead to large error bars",
     "cross-validation"),
    ("Leakage in data mining formulation detection and avoidance",
     "leakage"),
    ("Assessing and tuning brain decoders cross-validation caveats and guidelines",
     "cross-validation"),
    ("Brain waves based user identification system a pilot study",
     "eeg-biometrics"),
    ("EEG biometrics for individual recognition in resting state with closed eyes",
     "eeg-biometrics"),
    ("EEG fingerprints subject specific signature based on the aperiodic component "
     "of power spectrum", "eeg-biometrics"),
    ("On the vulnerability of machine learning models to biased data splits",
     "leakage"),
    ("Deep learning based electroencephalography analysis a systematic review",
     "eeg-review"),
    ("Machine learning for neuroimaging with scikit-learn", "cross-validation"),
    ("Individual differences in the resting state EEG a functional connectome "
     "fingerprint", "eeg-biometrics"),
    ("Subject independent emotion recognition from EEG signals using deep "
     "learning", "subject-wise"),
    ("Reproducibility of machine learning in health research", "leakage"),
    ("SMOTE synthetic minority over-sampling technique", "resampling"),
    ("The dangers of oversampling before cross validation", "resampling"),
    ("Selection bias in gene extraction on the basis of microarray gene "
     "expression data", "feature-selection"),
    ("Ten quick tips for machine learning in computational biology",
     "leakage"),
]

TIMEOUT = 30
PAUSE = 0.8


def clean(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def sim(a, b):
    return difflib.SequenceMatcher(None, clean(a).lower(),
                                   clean(b).lower()).ratio()


def search(q, rows=5):
    url = f"{API}?" + urllib.parse.urlencode(
        {"query.bibliographic": q, "rows": rows, "mailto": MAILTO})
    req = urllib.request.Request(
        url, headers={"User-Agent": f"reffind/1.0 ({MAILTO})"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)["message"].get("items", [])


def year_of(it):
    for f in ("published-print", "published-online", "issued"):
        dp = it.get(f, {}).get("date-parts", [[None]])
        if dp and dp[0] and dp[0][0]:
            return dp[0][0]
    return None


def bibkey(it):
    fam = "anon"
    if it.get("author"):
        fam = re.sub(r"[^a-z]", "", (it["author"][0].get("family") or "").lower())
    yr = year_of(it) or "0000"
    word = re.sub(r"[^a-z]", "",
                  clean((it.get("title") or [""])[0]).split(" ")[0].lower())
    return f"{fam}{yr}{word}"


def to_bibtex(it):
    t = it.get("type", "")
    kind = {"journal-article": "article",
            "proceedings-article": "inproceedings",
            "book": "book", "book-chapter": "incollection"}.get(t, "misc")
    authors = " and ".join(
        f"{a.get('family','')}, {a.get('given','')}".strip(", ")
        for a in it.get("author", []) if a.get("family"))
    fields = {
        "title": clean((it.get("title") or [""])[0]),
        "author": authors,
        "year": year_of(it),
        "doi": it.get("DOI"),
    }
    cont = (it.get("container-title") or [""])[0]
    if kind == "inproceedings":
        fields["booktitle"] = clean(cont)
    elif cont:
        fields["journal"] = clean(cont)
    for k in ("volume", "page", "publisher"):
        if it.get(k):
            fields["pages" if k == "page" else k] = it[k]
    body = ",\n  ".join(f"{k}={{{v}}}" for k, v in fields.items() if v)
    return f"@{kind}{{{bibkey(it)},\n  {body}\n}}"


def main():
    rows, bibs = [], []
    print("=" * 78)
    print("CANDIDATE REFERENCES -- CROSSREF VERIFIED")
    print("=" * 78)
    for q, tag in QUERIES:
        try:
            items = search(q)
        except Exception as e:
            print(f"[fail ] {q[:58]:<58} {type(e).__name__}")
            time.sleep(PAUSE)
            continue
        best, s = None, 0.0
        for it in items:
            ss = sim(q, (it.get("title") or [""])[0])
            if ss > s:
                best, s = it, ss
        time.sleep(PAUSE)
        if not best or s < 0.72:
            print(f"[  -  ] {q[:58]:<58} best sim {s:.2f}  NOT FOUND")
            rows.append(dict(query=q, topic=tag, status="NOT FOUND",
                             similarity=round(s, 2), doi="", title="",
                             year="", venue=""))
            continue
        title = clean((best.get("title") or [""])[0])
        print(f"[ ok  ] {title[:58]:<58} {year_of(best)}")
        print(f"         {best.get('DOI')}  |  "
              f"{clean((best.get('container-title') or [''])[0])[:50]}")
        rows.append(dict(query=q, topic=tag, status="FOUND",
                         similarity=round(s, 2), doi=best.get("DOI", ""),
                         title=title, year=year_of(best),
                         venue=clean((best.get("container-title") or [""])[0])))
        bibs.append((tag, to_bibtex(best)))

    with open(os.path.join(OUT, "candidate_refs.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    with open(os.path.join(OUT, "candidate_refs.bib"), "w",
              encoding="utf-8") as fh:
        fh.write("% Crossref-verified candidates for the expanded literature\n"
                 "% review. Every field below came from the Crossref record.\n\n")
        for tag, b in sorted(bibs):
            fh.write(f"% topic: {tag}\n{b}\n\n")

    n = sum(r["status"] == "FOUND" for r in rows)
    print("")
    print(f"found {n} / {len(rows)}")
    print(f"wrote {OUT}\\candidate_refs.csv and candidate_refs.bib")


if __name__ == "__main__":
    main()
