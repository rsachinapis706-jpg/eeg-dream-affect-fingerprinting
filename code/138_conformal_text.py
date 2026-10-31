"""
The conformal comparator in the manuscript (round 3, Reviewer 2, comment 6).

* Removes the suggestion that conformal prediction presupposes a useful
  signal: its validity rests on exchangeability of calibration and test
  points, whatever the strength of the association.
* States the specification (score, quantile, weighting, effective
  calibration size, dependence), reports the window-level construction as an
  empirical comparator -- on DEED it covers less than its nominal level -- and
  adds the group-aware construction of 130_group_conformal.py, which is valid
  for a window of a new participant.
* States the training / calibration allocation of every comparator instead
  of "identical folds, models and data".
Numbers from out/conformal_group.csv. Idempotent.
"""
import io
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = r"E:\Dream_Analysis\accesstemp.tex"
g = pd.read_csv(os.path.join(HERE, "out", "conformal_group.csv"))
S, D = "Sleep-EDF staging", "DEED affect"


def r(cond, units):
    return g[(g.condition == cond) & (g.calibration_units == units)].iloc[0]


wS, wD = r(S, "windows"), r(D, "windows")
gS, gD = r(S, "participants"), r(D, "participants")
assert gD.share_finite == 0.0 and gS.share_finite == 1.0, "case changed"

ALLOC = ("all rules share one decoder per fold, fitted on three quarters of "
         "that fold's training participants; the remaining quarter is used "
         "only to calibrate conformal prediction, and PCD's null decoders are "
         "fitted on the same three quarters")
COMP = (
    "Split conformal prediction is reported in two forms. In both, the "
    "conformity score is one minus the predicted probability of the true "
    "class and the threshold is the $\\lceil (n+1)(1-\\alpha) \\rceil$-th "
    "smallest calibration score, unweighted. The window-level form, used in "
    "Table~\\ref{tab:pcd}, takes every window of the calibration participants "
    "as a calibration point; windows of one participant are not exchangeable "
    "with a window of a new participant, so its guarantee does not formally "
    "apply here, and it is an empirical comparator: it covers the true label "
    f"for {wS.true_label_coverage:.4f} of Sleep-EDF windows but only "
    f"{wD.true_label_coverage:.4f} of DEED windows against the nominal "
    f"0.95, with mean set sizes {wS.set_size:.2f} and {wD.set_size:.2f}, and "
    f"returns a single class for {wS.singleton:.4f} and {wD.singleton:.4f} of "
    "windows. The group-aware form draws one window per calibration "
    "participant, the single-subsampling construction of Dunn et "
    "al.~\\cite{dunn2023distribution} in split form, so that the $n$ "
    "calibration scores are exchangeable with "
    "the score of a randomly drawn window of a new participant and the "
    "guarantee holds whatever the dependence within a participant; its "
    "effective calibration size is the number of calibration participants. "
    f"On Sleep-EDF, with {int(gS.calibration_size_median)} per fold, it "
    f"covers {gS.true_label_coverage:.4f} of windows with mean set size "
    f"{gS.set_size:.2f} and a single class for {gS.singleton:.4f}; on DEED, "
    f"with {int(gD.calibration_size_median)}, the required rank exceeds the "
    "number of calibration scores at $\\alpha=0.05$, so every set contains "
    "all three classes: the construction is informative only when $n$ "
    "exceeds $1/\\alpha-1$, here 19. Its $\\alpha$ controls coverage of the true label, "
    "PCD's the rate of prediction under the null, so the two are not")

EDITS = [
    ("Conformal prediction is valid in the formal sense, since its marginal "
     "coverage guarantee holds by construction, but it achieves that coverage "
     "by widening prediction sets, and a wide set is equally consistent with a "
     "hard input and with an unlearnable task. Each answers a question that "
     "presupposes the signal is present.",
     "Conformal prediction is valid under its own assumption, exchangeability "
     "of calibration and test points, whatever the strength of the "
     "association; when the association is weak it keeps that validity by "
     "returning wide sets, and a wide set is equally consistent with a hard "
     "input and with an unlearnable task. It asks which labels are plausible "
     "for an input, not whether the decoder's output on that input differs "
     "from that of decoders that learned nothing."),
    ("A conformal predictor remains formally valid in that setting, since its "
     "coverage guarantee holds by construction,",
     "A conformal predictor remains valid in that setting, since its coverage "
     "guarantee rests on exchangeability of calibration and test data rather "
     "than on the strength of the association,"),
    ("beside fixed confidence thresholds, on identical participant-separated "
     "folds, models and data:",
     f"beside fixed confidence thresholds, on the same participant-separated "
     f"folds and data, where {ALLOC}:"),
]
for prefix, rest in (("Split conformal prediction attains its nominal "
                      "coverage of the true label", " compared"),):
    pass


BIB = r"E:\Dream_Analysis\references.bib"
# Checked against Crossref (doi 10.1080/01621459.2022.2060112) and the
# paper's Section 4.2, "Subsampling Once" (arXiv:1809.07441).
DUNN = """
@article{dunn2023distribution,
  title={Distribution-Free Prediction Sets for Two-Layer Hierarchical Models},
  author={Dunn, Robin and Wasserman, Larry and Ramdas, Aaditya},
  journal={Journal of the American Statistical Association},
  volume={118},
  number={544},
  pages={2491--2502},
  year={2023},
  doi={10.1080/01621459.2022.2060112},
  note={doi: \\nolinkurl{10.1080/01621459.2022.2060112}}
}
"""


def main():
    b = io.open(BIB, encoding="utf-8", newline="").read()
    if "{dunn2023distribution," not in b:
        io.open(BIB, "w", encoding="utf-8", newline="").write(
            b.rstrip("\n") + "\n" + DUNN)
    s = io.open(TEX, encoding="utf-8", newline="").read()
    for old, new in EDITS:
        if old in s:
            s = s.replace(old, new)
        assert new in s, f"not applied: {new[:80]}"
    # the comparison sentence(s), from their start to "so the two are not"
    for start in ("Split conformal prediction attains its nominal coverage of "
                  "the true label", "Split conformal prediction is reported "
                  "in two forms."):
        if start in s:
            i = s.index(start)
            j = s.index("so the two are not", i) + len("so the two are not")
            s = s[:i] + COMP + s[j:]
            break
    else:
        raise SystemExit("comparison sentence not found")
    io.open(TEX, "w", encoding="utf-8", newline="").write(s)
    print("conformal comparator rewritten; DEED window-level coverage "
          f"{wD.true_label_coverage:.4f}, group-aware all-class sets")


if __name__ == "__main__":
    main()
