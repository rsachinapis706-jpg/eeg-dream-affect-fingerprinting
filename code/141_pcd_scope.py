"""
PCD described as null-referenced abstention throughout (round 3, Reviewer 2,
comment 5). Propositions 1 and 2 bound how often the rule predicts under a
global permutation null; they do not bound the error of retained predictions
or identify the informative inputs among a mixture of informative and
uninformative ones. This is stated where the proposition's scope is set out,
and the contributions item names what the guarantee is. Idempotent.
"""
import io

TEX = r"E:\Dream_Analysis\accesstemp.tex"

EDITS = [
    ("nor that the predicted label is correct.",
     "nor that the predicted label is correct. Nor does it bound the error "
     "among retained predictions, or identify which inputs are informative "
     "when informative and uninformative inputs are mixed: it bounds how often "
     "the rule predicts under a global null. PCD is therefore described here "
     "as null-referenced abstention, not as a test of whether an individual "
     "prediction is reliable."),
    ("Its components are standard; what is new is the localisation and its use "
     "as a decision rule, and the guarantee it carries, on average over inputs "
     "or, with a stratified reference, for every participant,",
     "Its components are standard; what is new is the localisation and its use "
     "as a decision rule, and the guarantee it carries -- a bound on how often "
     "it predicts under a permutation null, on average over inputs or, with a "
     "stratified reference, for every participant --"),
]


def main():
    s = io.open(TEX, encoding="utf-8", newline="").read()
    for old, new in EDITS:
        if new not in s:
            assert s.count(old) == 1, old[:70]
            s = s.replace(old, new)
    io.open(TEX, "w", encoding="utf-8", newline="").write(s)
    print("PCD scope stated as null-referenced abstention")


if __name__ == "__main__":
    main()
