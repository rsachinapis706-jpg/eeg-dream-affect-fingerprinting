"""
The primary evaluation (120_primary_random.py) in the Methods, the
specification table and the bibliography (round 3, Reviewer 2, comment 2).

Section IV-F itself, its table and the claim are written by 91_write_ivf.py;
the abstract, the conclusions and table note c by 101_write_summary.py. This
script adds what neither owns:

  * the statement of how p-values are computed (Section III-B), which said
    every permutation p-value uses Equation (perm) -- the primary evaluation
    uses sequential Monte Carlo p-values (Besag and Clifford, 1991);
  * the multiplicity paragraph (Section IV-A): Holm's procedure for the
    primary evaluation, whose configurations stop at different numbers of
    relabellings;
  * a row of the specification table and the validation row of the
    datasets table;
  * the Besag and Clifford entry of references.bib (checked against its
    Crossref record, doi 10.1093/biomet/78.2.301).
Idempotent.
"""
import io
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
TEX = r"E:\Dream_Analysis\accesstemp.tex"
BIB = r"E:\Dream_Analysis\references.bib"

BESAG = """@article{besag1991sequential,
  title={Sequential {M}onte {C}arlo $p$-values},
  author={Besag, Julian and Clifford, Peter},
  journal={Biometrika},
  volume={78},
  number={2},
  pages={301--304},
  year={1991},
  doi={10.1093/biomet/78.2.301},
  note={doi: \\nolinkurl{10.1093/biomet/78.2.301}}
}
"""


def main():
    pr = pd.read_csv(os.path.join(OUT, "primary_random_summary.csv"))
    R = int(pr.partitions.iloc[0])
    edits = [
        ("Every permutation $p$-value reported in this paper is computed by "
         "Equation~\\eqref{eq:perm};",
         "Every permutation $p$-value computed from a fixed number of "
         "permutations is given by Equation~\\eqref{eq:perm}, and those of the "
         "primary evaluation by its sequential counterpart "
         "(Section~\\ref{sec:howmuch}) \\cite{besag1991sequential};"),
        ("Both Holm's procedure and a single-step max-statistic correction are "
         "given, the latter using the permutations shared by all "
         "configurations so that their correlation is accounted for.",
         "Both Holm's procedure and a single-step max-statistic correction are "
         "given, the latter using the permutations shared by all "
         "configurations so that their correlation is accounted for. In the "
         "primary evaluation the configurations stop at different numbers of "
         "relabellings, too few of which are shared for the max-statistic "
         "correction to resolve values near 0.05, so Holm's procedure is the "
         "correction used there."),
        ("Validation & Participant-separated throughout the affect analyses: "
         "five-fold, grouped on participant, over ten fold assignments on "
         "DEED;",
         "Validation & Participant-separated throughout the affect analyses: "
         f"five-fold, grouped on participant; on DEED the primary evaluation "
         f"averages over {R} random class-balanced participant partitions and "
         "the other analyses over ten fold assignments of the splitter;"),
    ]
    row_anchor = "Overlap and ladder (Tables~\\ref{tab:overlap}"
    row = ("Primary evaluation (Table~\\ref{tab:primary}) & as above & 108 & "
           "RF $\\times$2, LR $\\times$2 & part.-sep.; "
           f"{R} random class-balanced partitions, folds recomputed for every "
           "labelling & none; selection among the four nested in each "
           "training fold as a separate estimate & recording level, "
           "within-participant and unrestricted; sequential, stopping at 20 "
           "exceedances or 1{,}000 \\\\\n")
    s = io.open(TEX, encoding="utf-8", newline="").read()
    for old, new in edits:
        # a new passage may extend its old one, so test for the new first
        if new not in s and old in s:
            assert s.count(old) == 1, old[:60]
            s = s.replace(old, new)
        assert s.count(new) == 1, new[:60]
    if row not in s:
        assert s.count(row_anchor) == 1
        s = s.replace(row_anchor, row + row_anchor)
    assert s.count("Primary evaluation (Table~\\ref{tab:primary})") == 1
    io.open(TEX, "w", encoding="utf-8", newline="").write(s)
    b = io.open(BIB, encoding="utf-8", newline="").read()
    if "@article{besag1991sequential," not in b:
        io.open(BIB, "w", encoding="utf-8", newline="").write(
            b.rstrip("\n") + "\n\n" + BESAG)
    print("primary evaluation: Methods, specification table, bibliography")


if __name__ == "__main__":
    main()
