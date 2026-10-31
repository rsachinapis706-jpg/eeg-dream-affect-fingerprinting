"""
Adds the CBraMod seed-sensitivity result (105_cbramod_seeds.py) to Section IV-J,
the foundation-model table footnote and Section VII-A. Re-runnable: each
insertion replaces its own earlier version.
"""
import io
import re
import shutil

import pandas as pd

OUT = r"E:\Dream GAN\analysis\out"
TEX = r"E:\Dream_Analysis\accesstemp.tex"


def m(x):
    return f"${x:+.4f}$"


def main():
    r = pd.read_csv(OUT + r"\cbramod_seeds.csv")
    p = pd.read_csv(OUT + r"\cbramod_seeds_pretraining.csv")
    sm = pd.read_csv(OUT + r"\cbramod_seeds_summary.csv").set_index("config")
    pre, scr = sm.loc["revised"], sm.loc["scratch_revised"]
    all_zero = bool(((p.lo <= 0) & (p.hi >= 0)).all())
    chance_all = bool(((r.bal_lo <= 1 / 3) & (r.bal_hi >= 1 / 3)).all())
    sent = (
        "<<SEEDS>>Sensitivity to initialisation was measured by repeating the second settings with "
        f"three seeds, which change the head initialisation, the order of the training data and "
        f"the participants held out for validation, and for the randomly initialised model the "
        f"backbone as well. Pooled balanced accuracy "
        f"ranges from {pre.lo:.4f} to {pre.hi:.4f} with pretraining (mean {pre['mean']:.4f}, "
        f"standard deviation {pre.sd:.4f}) and from {scr.lo:.4f} to {scr.hi:.4f} without "
        f"(mean {scr['mean']:.4f}, standard deviation {scr.sd:.4f}); the paired difference "
        f"ranges from {m(p['diff'].min())} to {m(p['diff'].max())}"
        + (", every interval including zero" if all_zero else "")
        + (", and every interval for balanced accuracy includes the 0.3333 chance level."
           if chance_all else ".") + "<</SEEDS>>")
    t = io.open(TEX, encoding="utf-8", newline="").read()
    shutil.copyfile(TEX, TEX + ".pre_seeds")
    first = "Sensitivity to initialisation was measured by repeating the second settings"
    if first in t:                      # re-run: remove the earlier version,
        i = t.index(first) - 1          # which always ends its paragraph line
        j = t.index("\n", i + 1)
        t = t[:i] + t[j:]
    sent = sent.replace("<<SEEDS>>", "").replace("<</SEEDS>>", "")
    anchor = "Every interval includes the 0.3333 chance level."
    assert t.count(anchor) == 1, t.count(anchor)
    i = t.index(anchor) + len(anchor)
    assert t[i] == "\n", "anchor must end its paragraph"
    t = t[:i] + " " + sent + t[i:]
    assert t.count(first) == 1
    # Section VII-A
    v = (f" Across three seeds the pretrained model ranges from {pre.lo:.4f} to "
         f"{pre.hi:.4f} and the model trained from scratch from {scr.lo:.4f} to "
         f"{scr.hi:.4f}.")
    t = re.sub(r" Across three seeds the pretrained model ranges from 0\.\d{4} to "
               r"0\.\d{4} and the model trained from scratch from 0\.\d{4} to 0\.\d{4}\.",
               "", t)
    a2 = "this concerns CBraMod on this task, not EEG foundation models as a class."
    assert t.count(a2) == 1
    t = t.replace(a2, a2 + v)
    # Section V: the first-seed value alone would read as the best of three
    pat = re.compile(r"On dream affect, fine-tuned with a procedure of the same "
                     r"kind, it reaches (0\.\d{4} \[0\.\d{4}, 0\.\d{4}\])"
                     r"(?: with the first seed and 0\.\d{4} to 0\.\d{4} across "
                     r"three)? \(Section~\\ref\{sec:within\}\)\.")
    assert len(pat.findall(t)) == 1
    t = pat.sub(lambda mm: (
        "On dream affect, fine-tuned with a procedure of the same kind, it "
        f"reaches {mm.group(1)} with the first seed and {pre.lo:.4f} to "
        f"{pre.hi:.4f} across three (Section~\\ref{{sec:within}})."), t)
    io.open(TEX, "w", encoding="utf-8", newline="").write(t)
    print(sent.replace("<<SEEDS>>", "").replace("<</SEEDS>>", ""))
    print(v)


if __name__ == "__main__":
    main()
