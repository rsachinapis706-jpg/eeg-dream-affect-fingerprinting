"""
The catalogue result stated as a testable, qualified hypothesis (round 3,
Reviewer 2, comment 10).

The categorical reading of the non-significant catalogue associations was
removed earlier; Reviewer 2 asks that it be replaced by a hypothesis that can
be tested. The sentence added states what the participant route predicts where
signals are available, and how weak that prediction is where no association
was detected. Idempotent.
"""
import io

TEX = r"E:\Dream_Analysis\accesstemp.tex"
ANCHOR = ("and leakage through overlapping windows or shared recordings does not "
          "require a participant-level association at all.")
ADD = (" What the participant route predicts can be tested wherever the signals "
       "are available: other things being equal, where participant identity "
       "carries label information, separating participants between training "
       "and test should lower accuracy relative to partitioning by window or "
       "record, by more the stronger that association; where no association "
       "was detected, the route predicts no such loss, a weak prediction for "
       "the small cohorts concerned and one that says nothing about loss "
       "through shared recordings.")


def main():
    s = io.open(TEX, encoding="utf-8", newline="").read()
    if ADD not in s:
        assert s.count(ANCHOR) == 1
        s = s.replace(ANCHOR, ANCHOR + ADD)
    io.open(TEX, "w", encoding="utf-8", newline="").write(s)
    print("testable hypothesis added after the catalogue result")


if __name__ == "__main__":
    main()
