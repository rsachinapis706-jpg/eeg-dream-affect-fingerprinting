"""
Single-class CBraMod folds, read against their label distributions and
training diagnostics (round 3, Reviewer 2, comment 10).

A fold that predicts one class is not by itself a failure of training. From
out/cbramod_diag_revised_<seed>_folds.csv (per-fold test class counts,
prediction counts, chosen epoch, best validation balanced accuracy) this
reports, over the three seeds of the second settings: how many folds predict
one class, how many of those held-out participants contribute all three
classes, and the chosen epoch and validation balanced accuracy in those folds
against the others. The sentence says only what the diagnostics show.
Idempotent: the sentence is replaced if present.
"""
import glob
import io
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = r"E:\Dream_Analysis\accesstemp.tex"
ANCHOR = ("every interval for balanced accuracy includes the 0.3333 chance "
          "level.")
START = "Across the three seeds, "

rows = []
for f in sorted(glob.glob(os.path.join(HERE, "out",
                                       "cbramod_diag_revised_2026*_folds.csv"))):
    d = pd.read_csv(f, dtype={"held_out": str})
    c, o = d[d.constant_pred], d[~d.constant_pred]
    rows.append(dict(n=len(c), folds=len(d), three=int((c.test_classes == 3).sum()),
                     ep_c=c.chosen_epoch.median(), ep_o=o.chosen_epoch.median(),
                     va_c=c.best_val_bal.median(), va_o=o.best_val_bal.median()))
t = pd.DataFrame(rows)
assert len(t) == 3 and (t.folds == 34).all()
counts = ", ".join(str(n) for n in t.n[:-1]) + f" and {t.n.iloc[-1]}"


def rng(a, b, fmt):
    lo, hi = min(a), max(b)
    return f"{lo:{fmt}}" if lo == hi else f"{lo:{fmt}} to {hi:{fmt}}"


SENT = (
    f"{START}{counts} of 34 folds predict a single class. In {t.three.sum()} "
    f"of these {t.n.sum()} folds the held-out participant contributes all "
    "three classes, so the constant prediction is not explained by the test "
    "labels; early stopping kept an early model (median chosen epoch "
    f"{rng(t.ep_c, t.ep_c, 'g')}, against {rng(t.ep_o, t.ep_o, 'g')} in the "
    "other folds), whose balanced accuracy on the validation participants was "
    f"at chance (median {rng(t.va_c, t.va_c, '.4f')}, against "
    f"{rng(t.va_o, t.va_o, '.4f')}). These diagnostics show that no validation "
    "signal above chance was found in those folds; they do not distinguish a "
    "failure of optimisation from the absence of a learnable association.")


def main():
    s = io.open(TEX, encoding="utf-8", newline="").read()
    s = re.sub(re.escape(" " + START) + r".*?learnable association\.", "", s,
               flags=re.S)
    assert s.count(ANCHOR) == 1
    s = s.replace(ANCHOR, ANCHOR + " " + SENT)
    io.open(TEX, "w", encoding="utf-8", newline="").write(s)
    print(SENT)


if __name__ == "__main__":
    main()
