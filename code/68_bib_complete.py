"""
Complete and correct references.bib (Reviewer 2: "ensure that all references,
particularly recent 2025-2026 publications, have complete bibliographic
information and DOI details where available"; Reviewer 1: cite the Sleep-EDF
resource explicitly).

Sources of every change, all checked on 30 Sep 2026:
  * DOIs, volumes and pages for entries whose title, first author, year and
    venue already agree with Crossref (38_verify_refs.py status MATCH).
    Existing values are never overwritten automatically; only missing fields
    are filled, and any disagreement is printed for a decision.
  * Hand corrections, each from the primary record named beside it.
  * New entries for the Sleep-EDF resource, as PhysioNet's page requires, and
    for the peer-reviewed DREAM database paper.

A unified diff is printed; the original is kept as references.bib.pre_68.
"""
import difflib
import io
import os
import re
import shutil

import pandas as pd

BIB = os.path.join("E:" + os.sep, "Dream_Analysis", "references.bib")
TEX = os.path.join("E:" + os.sep, "Dream_Analysis", "accesstemp.tex")
AUDIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out",
                     "reference_audit.csv")


def span(b, key):
    m = re.search(r"@\w+\{" + re.escape(key) + r",.*?\n\}", b, re.S)
    if not m:
        raise KeyError(key)
    return m.start(), m.end()


def has(entry, field):
    return re.search(r"^\s*" + field + r"\s*=", entry, re.M) is not None


def set_field(entry, field, value):
    line = f"  {field}={{{value}}}"
    if has(entry, field):
        # keep the separator the replaced line carried, or the next field
        # runs into this one and BibTeX cannot parse the entry
        return re.sub(r"^\s*" + field + r"\s*=.*?(,?)\s*$",
                      lambda m: line + m.group(1), entry, count=1, flags=re.M)
    body = entry.rstrip()
    assert body.endswith("}")
    body = body[:-1].rstrip()
    if not body.endswith(","):
        body += ","
    return body + "\n" + line + "\n}"


def del_field(entry, field):
    return re.sub(r"^\s*" + field + r"\s*=.*\n", "", entry, flags=re.M)


def set_type(entry, typ):
    return re.sub(r"^@\w+\{", f"@{typ}{{", entry)


def edit(b, key, fn):
    s, e = span(b, key)
    return b[:s] + fn(b[s:e]) + b[e:]


def main():
    b0 = io.open(BIB, encoding="utf-8").read()
    b = b0

    # ---- 1. DOIs and missing fields for entries already matching Crossref --
    a = pd.read_csv(AUDIT)
    m = a[(a.status == "MATCH") & (a.bib_doi.isna()) & a.cr_doi.notna()]
    skip = {"liu2022deed"}                     # handled by hand below
    for _, r in m.iterrows():
        if r.key in skip:
            continue

        def fill(ent, r=r):
            ent = set_field(ent, "doi", r.cr_doi)
            if not has(ent, "volume") and pd.notna(r.cr_volume):
                ent = set_field(ent, "volume", str(int(r.cr_volume)))
            if not has(ent, "pages") and pd.notna(r.cr_page):
                ent = set_field(ent, "pages", str(r.cr_page).replace("-", "--"))
            return ent
        b = edit(b, r.key, fill)

    # ---- 2. hand corrections, each from its primary record ----------------
    fixes = {
        # bioRxiv record 10.1101/2022.09.19.508475; "pages={2022--09}" was a
        # Google Scholar artefact
        "liu2022deed": lambda e: set_field(del_field(e, "pages"), "doi",
                                           "10.1101/2022.09.19.508475"),
        # NeurIPS 38 (2025), Crossref 10.52202/085713-1375
        "xiao2026brainomni": lambda e: set_field(set_field(set_field(set_field(
            set_field(set_type(del_field(e, "journal"), "inproceedings"),
                      "booktitle", "Advances in Neural Information Processing "
                                   "Systems"),
            "year", "2025"), "pages", "46081--46114"), "volume", "38"),
            "doi", "10.52202/085713-1375"),
        # publisher page techscience.com/cmc/v71n1/45428: 2022, 71(1) 889-905.
        # Authors confirmed there; Crossref's author metadata for this DOI is
        # wrong, so the bib author list is kept.
        "pei2021hybrid": lambda e: set_field(set_field(set_field(
            e, "year", "2022"), "journal", "Computers, Materials \\& Continua"),
            "doi", "10.32604/cmc.2022.021830"),
        # Crossref 10.1016/j.compbiomed.2025.111425: family name is
        # "Sayad Mojdehbar"
        "mojdehbar2026automatic": lambda e: set_field(set_field(
            e, "author", "Sayad Mojdehbar, Nazanin and Mohammadzadeh Asl, Babak "
                         "and Zarei, Asghar"),
            "doi", "10.1016/j.compbiomed.2025.111425"),
        # Springer LNNS 947, Crossref 10.1007/978-981-97-1326-4_38
        "naidu2024emotion": lambda e: set_field(set_field(set_field(set_field(
            set_field(del_field(e, "organization"), "booktitle",
                      "Smart Trends in Computing and Communications "
                      "(SmartCom 2024)"),
            "series", "Lecture Notes in Networks and Systems"),
            "volume", "947"), "publisher", "Springer Nature Singapore"),
            "doi", "10.1007/978-981-97-1326-4_38"),
        # jmlr.org/papers/v11/ojala10a.html: 11(62) 1833-1863
        "ojala2010permutation": lambda e: set_field(set_field(set_field(
            set_field(e, "title", "Permutation Tests for Studying Classifier "
                                  "Performance"),
            "journal", "Journal of Machine Learning Research"),
            "number", "62"), "pages", "1833--1863"),
        # published version, Crossref 10.1561/2200000101
        "angelopoulos2023gentle": lambda e: set_field(set_field(set_field(
            set_field(set_field(set_field(set_field(
                e, "title", "Conformal Prediction: A Gentle Introduction"),
                "journal", "Foundations and Trends in Machine Learning"),
                "volume", "16"), "number", "4"), "pages", "494--591"),
                "year", "2023"), "doi", "10.1561/2200000101"),
        # papers.neurips.cc/paper/7073 and dblp: NIPS 30, pp. 4878-4887
        "geifman2017selective": lambda e: set_field(set_field(set_type(
            set_field(del_field(e, "journal"), "booktitle",
                      "Advances in Neural Information Processing Systems"),
            "inproceedings"), "pages", "4878--4887"), "volume", "30"),
        # proceedings.mlr.press/v70/guo17a.html
        "guo2017calibration": lambda e: set_field(set_field(set_field(
            del_field(e, "organization"), "booktitle",
            "Proceedings of the 34th International Conference on Machine "
            "Learning"), "series", "Proceedings of Machine Learning Research"),
            "volume", "70"),
        # ICLR has no pagination; the volume and page fields were artefacts
        "jiang2024large": lambda e: set_field(set_field(del_field(del_field(
            e, "volume"), "pages"), "booktitle",
            "International Conference on Learning Representations (ICLR)"),
            "url", "https://openreview.net/forum?id=QzTpTRVtrP"),
        "wang2025cbramod": lambda e: set_field(set_field(del_field(del_field(
            e, "volume"), "pages"), "booktitle",
            "International Conference on Learning Representations (ICLR)"),
            "url", "https://proceedings.iclr.cc/paper_files/paper/2025/hash/"
                   "bbbd6d915cb90be21c1254a82d45cedd-Abstract-Conference.html"),
    }
    for key, fn in fixes.items():
        b = edit(b, key, fn)

    # ---- 3. DREAM: exact dataset version analysed, plus the journal paper --
    # Local files match figshare version 9 byte for byte (People 1685,
    # Data records 357539, Datasets 170822 bytes).
    s, e = span(b, "wong2023dream")
    b = b[:s] + """@misc{wong2025dreamdata,
  title={The {DREAM} Database},
  author={Wong, William and Andrillon, Thomas and Decat, Nicolas and Herzog, Rub{\\'e}n and Noreika, Valdas and Valli, Katja and Windt, Jennifer and Tsuchiya, Naotsugu},
  howpublished={Monash University, dataset, version 9},
  year={2025},
  doi={10.26180/22133105.v9}
}""" + b[e:]

    b += """
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Added in revision 2 for dataset attribution (Reviewer 1, comment 12).
%% Sleep-EDF: the three citations PhysioNet's resource page requires.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@article{wong2025dream,
  title={A Dream {EEG} and Mentation Database},
  author={Wong, William and Herzog, Rub{\\'e}n and Andrade, K{\\'a}tia Cristine and Andrillon, Thomas and de Araujo, Draulio Barros and Arnulf, Isabelle and others},
  journal={Nature Communications},
  volume={16},
  pages={7495},
  year={2025},
  doi={10.1038/s41467-025-61945-1}
}

@article{kemp2000analysis,
  title={Analysis of a Sleep-Dependent Neuronal Feedback Loop: The Slow-Wave Microcontinuity of the {EEG}},
  author={Kemp, Bob and Zwinderman, Aeilko H. and Tuk, Bert and Kamphuisen, Hilbert A. C. and Obery{\\'e}, Josefien J. L.},
  journal={IEEE Transactions on Biomedical Engineering},
  volume={47},
  number={9},
  pages={1185--1194},
  year={2000},
  doi={10.1109/10.867928}
}

@misc{kemp2013sleepedf,
  title={Sleep-{EDF} Database Expanded (Version 1.0.0)},
  author={Kemp, Bob},
  howpublished={PhysioNet},
  year={2013},
  doi={10.13026/C2X676}
}

@article{pollard2026physionet,
  title={{PhysioNet} as a Global Platform for Biomedical Research},
  author={Pollard, Tom and Moody, Benjamin E. and Lehman, Li-wei H. and Gow, Brian J. and Fernandes, Chrystinne and Xie, Chen and Johnson, Alistair and Mark, Roger G. and Heldt, Thomas},
  journal={Nature Health},
  volume={1},
  number={8},
  pages={792--795},
  year={2026},
  doi={10.1038/s44360-026-00096-z}
}
"""

    shutil.copyfile(BIB, BIB + ".pre_68")
    io.open(BIB, "w", encoding="utf-8", newline="").write(b)

    for l in difflib.unified_diff(b0.splitlines(), b.splitlines(),
                                  "references.bib (before)",
                                  "references.bib (after)", n=0,
                                  lineterm=""):
        print(l)


if __name__ == "__main__":
    main()
