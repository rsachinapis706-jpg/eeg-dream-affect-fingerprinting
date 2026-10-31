"""
Classifier against identity rule in Section IV-H, with paired uncertainty and
agreement (round 3, Reviewer 2, comment 4).

The Introduction and Conclusion already give the agreement (61% of windows);
the results section stated only that the two balanced accuracies match, and
that "what the forest achieves is also achieved" by the rule. Reviewer 2 asks
for paired uncertainty for the classifier-versus-identity differences and
notes that similar balanced accuracy does not establish the same predictions
or mechanism. Numbers from out/route_contrasts.csv (124_route_contrasts.py).
Idempotent.
"""
import io
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = r"E:\Dream_Analysis\accesstemp.tex"


def num(x, sign=False):
    s = f"{abs(x):.4f}"
    if x < 0:
        return "$-$" + s
    return ("+" if sign else "") + s


def main():
    t = pd.read_csv(os.path.join(HERE, "out", "route_contrasts.csv")).set_index(
        ["dataset", "contrast"])
    fi = "forest minus identity rule, window-shuffled"
    ag = "agreement forest vs identity rule, window-shuffled"
    d, tn = t.loc[("DEED", fi)], t.loc[("Tononi", fi)]
    da, ta = t.loc[("DEED", ag)], t.loc[("Tononi", ag)]
    old1 = ("the rule reaches 0.4915 balanced accuracy, matching the forest's "
            "0.4909, where the majority class scores 0.3333.")
    new1 = ("the rule reaches 0.4915 balanced accuracy against the forest's "
            "0.4909, where the majority class scores 0.3333; the paired "
            f"difference, forest minus rule, is {num(d.estimate, True)} "
            f"(95\\% participant-bootstrap interval {num(d.lo)} to "
            f"{num(d.hi)}). Similar balanced accuracy does not mean the same "
            f"predictions: the two predict the same class on {da.estimate:.4f} "
            f"of windows, against {da.expected_if_independent:.4f} expected if "
            "each predicted independently with its own class frequencies. On "
            f"the Tononi corpus the difference is {num(tn.estimate, True)} "
            f"({num(tn.lo)} to {num(tn.hi)}) and the agreement "
            f"{ta.estimate:.4f} against {ta.expected_if_independent:.4f}.")
    old2 = ("once recordings are kept together, what the forest achieves is "
            "also achieved by a rule that knows only who the participant is.")
    new2 = ("once recordings are kept together, a rule that knows only who the "
            "participant is reaches the forest's level, which shows that this "
            "level does not require the EEG to carry information about "
            "affect, not that the forest relies on identity.")
    # wording of the first version of this script
    old2b = new2.replace("does not require the EEG to carry information "
                         "about affect", "does not require information about "
                         "affect")
    s = io.open(TEX, encoding="utf-8", newline="").read()
    for old, new in ((old1, new1), (old2, new2), (old2b, new2)):
        if old in s:
            assert s.count(old) == 1
            s = s.replace(old, new)
    for new in (new1, new2):
        assert s.count(new) == 1, new[:60]
    io.open(TEX, "w", encoding="utf-8", newline="").write(s)
    print(new1)
    print(new2)


if __name__ == "__main__":
    main()
