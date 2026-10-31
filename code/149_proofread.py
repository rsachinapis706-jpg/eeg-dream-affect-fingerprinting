"""
Proofreading corrections to text that no other writer owns (round 3,
Reviewer 1: proofreading and consistency; Reviewer 2, comment 10: remove
overstatements). Each entry replaces one exact passage; re-running leaves the
manuscript unchanged. Text owned by a writer is corrected in that writer.
"""
import io

TEX = r"E:\Dream_Analysis\accesstemp.tex"

PROV = ("it cannot locate; and a provenance table that assigns each reported "
        "number to its result file, row and column or, for a number computed "
        "in the text, to the computation that reproduces it from the result "
        "files, each assignment checked against the sentence in which the "
        "number appears.")

FIXES = [
    # Literature review: the published description does not say where
    # ReliefF was fitted (Table 21, note b says the same)
    ("The feature-selection stage improves discriminability relative to "
     "using all features. Its placement is consequential, however: ReliefF is "
     "described as applied to the feature matrix prior to classification "
     "rather than within each cross-validation fold. Selection fitted on the "
     "complete dataset has access to test labels. Section~\\ref{sec4} "
     "quantifies how much this ordering alone contributes to measured "
     "accuracy.",
     "The authors report that the feature-selection stage improves "
     "discriminability relative to using all features. Its placement matters: "
     "ReliefF is described as applied to the feature matrix prior to "
     "classification, and whether it was fitted before the split or within "
     "each cross-validation fold is not stated. Selection fitted on the "
     "complete dataset has access to test labels. Section~\\ref{sec4} "
     "measures how much this ordering alone changes balanced accuracy."),
    # Demuru and Fraschini (2020), abstract: the aperiodic component
    # "outperforms the canonically-defined frequency bands"
    ("reporting that this non-oscillatory background identifies individuals "
     "as reliably as the oscillatory features usually studied.",
     "reporting that this non-oscillatory background identifies individuals "
     "better than the canonically defined frequency bands, consistently "
     "across eyes-open and eyes-closed recordings."),
    # Section III-C: an empirical ordering is consistent with, not explained
    # by, a definition (Reviewer 1: do not overgeneralise the PCD maths)
    ("Coverage therefore increases monotonically with the extent of that "
     "dominance, which explains the ordering observed in "
     "Figure~\\ref{fig:noise}.",
     "Coverage therefore rises as that dominance grows, which is consistent "
     "with the ordering observed in Figure~\\ref{fig:noise}."),
    # terminology of Section III-B: "participant", not "subject"
    ("All figures are leave-one-subject-out cross-validation scores.}",
     "All figures are leave-one-participant-out cross-validation scores.}"),
    # Table tab:overlap: name the metric (Reviewer 2, comment 8)
    ("so no other leakage source is active. Mean $\\pm$ standard deviation "
     "over five seeds.}",
     "so no other leakage source is active. Accuracy, mean $\\pm$ standard "
     "deviation over five seeds.}"),
    # neutral wording for the participant-separated estimate (Reviewer 2,
    # comment 10)
    ("Window overlap is therefore a large and independent contributor to "
     "reported accuracy that leaves the honest estimate untouched, and it is "
     "not reported in either study.",
     "Window overlap therefore adds substantially to window-level accuracy "
     "while leaving the participant-separated estimate unchanged, and neither "
     "study reports it."),
    ("0 \\ Honest baseline                & ",
     "0 \\ Participant-separated baseline & "),
    ("The session-transfer result supports the stable-trait reading and rules "
     "out the possibility that the confound is purely an artefact of one "
     "recording setup.",
     "The session-transfer result supports the stable-trait reading and is "
     "inconsistent with a confound that arises purely from one recording "
     "setup."),
    ("reaches the same conclusion about the aperiodic component "
     "\\cite{demuru2020eeg} strengthens the reading without converting it "
     "into a demonstration of mechanism.",
     "reaches the same conclusion about the aperiodic component "
     "\\cite{demuru2020eeg} agrees with this reading without converting it "
     "into a demonstration of mechanism."),
    # Section IV-I: scope the generalisation to what was tested
    ("Every modality tested here carries it, and window-level partitioning is "
     "therefore unsafe wherever a participant contributes more than one "
     "window.",
     "Every modality tested here carries it, so window-level partitioning "
     "carries the same risk wherever a participant contributes more than one "
     "window and the label is associated with the participant or the "
     "recording."),
    ("It also follows that the confound cannot be avoided by changing "
     "modality; it is avoided only by changing the partition.",
     "It also follows that the confound is avoided by changing the "
     "partition, not the modality."),
    ("The median excess normalised mutual information is 0.117, ranging from "
     "0.185 to 0.019.",
     "The median excess normalised mutual information is 0.117, ranging from "
     "0.019 to 0.185."),
    # Ethics: four resources are analysed (Table tab:datasets), not three
    ("This study is a secondary analysis of three publicly available "
     "resources. DEED, Sleep-EDF Expanded and the DREAM database catalogue "
     "were collected",
     "This study is a secondary analysis of four publicly available "
     "resources. DEED, Sleep-EDF Expanded, the Tononi serial-awakenings "
     "corpus and the DREAM database catalogue were collected"),
    # Data availability: the keyed provenance table (Reviewer 2, comment 7),
    # first as added, then as it stands after every number was read in
    # context (132_provenance.py, provenance_overrides.py)
    ("it cannot locate; and a provenance table that assigns each reported "
     "number to its result file, row and column, stating where several result "
     "cells share its value.",
     PROV),
    ("and an audit script that re-extracts every decimal number appearing in "
     "the manuscript and matches it against the result files, reporting any "
     "value it cannot locate.",
     "an audit script that re-extracts every decimal number appearing in the "
     "manuscript and matches it against the result files, reporting any value "
     + PROV),
    # paired-test power: recomputed from the stored per-participant results
    # (155_within_checks.py), the signed-rank power is 0.2352, not 0.23
    ("about 0.39 for a $t$-test, and 0.23 for the signed-rank test",
     "about 0.39 for a $t$-test, and 0.24 for the signed-rank test"),
]


def main():
    s = io.open(TEX, encoding="utf-8", newline="").read()
    for old, new in FIXES:
        if old in s:
            assert s.count(old) == 1, old[:60]
            s = s.replace(old, new)
        assert s.count(new) == 1, new[:60]
    io.open(TEX, "w", encoding="utf-8", newline="").write(s)
    print(f"{len(FIXES)} proofreading corrections in place")


if __name__ == "__main__":
    main()
