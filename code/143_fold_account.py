"""
Experiment-level account of label-dependent folds (round 3, Reviewer 2,
comment 1).

Checked in the code: every permutation test passes the permuted labelling to
the fold constructor (65 evaluate(X, yl, ...), 75 evaluate(X, yl, ...),
11/15 within_bacc on the permuted labels, 120/125/129 partition_mean and
random_stratified_folds on yl), so the class-stratified participant folds are
recomputed for each labelling with the same seed. PCD (77, 85, 110) and its
leaky-partition analysis (116) use folds drawn without labels, as
Proposition 1 requires. The Methods state this once. Idempotent.
"""
import io

TEX = r"E:\Dream_Analysis\accesstemp.tex"
ANCHOR = "Neither is uniformly stricter, and each result names the scheme it uses."
ADD = (" Participant folds are stratified by class and so depend on the labels; "
       "in every permutation test they are recomputed from each permuted "
       "labelling with the same seed, so that the observed statistic is "
       "compared with statistics of the identical procedure. PCD is the "
       "exception by design: Proposition~\\ref{prop:calib} requires a fold "
       "assignment fixed without reference to the labels, and every PCD result "
       "uses folds drawn by shuffling participants without labels.")


def main():
    s = io.open(TEX, encoding="utf-8", newline="").read()
    if ADD not in s:
        assert s.count(ANCHOR) == 1
        s = s.replace(ANCHOR, ANCHOR + ADD)
    io.open(TEX, "w", encoding="utf-8", newline="").write(s)
    print("fold account added to the permutation-test methods")


if __name__ == "__main__":
    main()
