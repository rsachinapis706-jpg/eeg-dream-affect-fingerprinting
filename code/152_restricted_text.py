"""
Exchangeability of recordings and the restricted nulls in the manuscript
(round 3, Reviewer 2, comment 1).

The statistical procedures said that both permutation schemes treat a
participant's recordings as exchangeable and that "no test here conditions
on that order". This replaces that sentence with the null under which the
exchanges are justified and with the result of 129_restricted_nulls.py: the
primary evaluation tested with labels exchanged only within a night, and
only within consecutive blocks of three recordings of a night. The result
sentence is generated only if neither restriction changes the conclusion;
otherwise the script stops and the passage is written by hand. Idempotent.
"""
import importlib.util
import io
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
TEX = r"E:\Dream_Analysis\accesstemp.tex"
OLD = ("Both treat a participant's recordings as exchangeable, which ignores "
       "their order within a night; no test here conditions on that order.")
START = "Both treat a participant's recordings as exchangeable across nights"
END = ("the conclusions of Section~\\ref{sec:howmuch} do not depend on "
       "exchanges across nights or positions within a night.")
OLD_END = "the conclusions of Section~\\ref{sec:howmuch} do not depend on them."


def main():
    spec = importlib.util.spec_from_file_location(
        "rn", os.path.join(HERE, "129_restricted_nulls.py"))
    rn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rn)
    t = pd.read_csv(os.path.join(OUT, "restricted_nulls_summary.csv")
                    ).set_index("config")
    assert len(t) == 4
    ps = t[["night_p", "night_block_p"]]
    holm = t[["night_p_holm", "night_block_p_holm"]]
    if (ps < 0.05).any().any() or (holm < 0.05).any().any():
        raise SystemExit("a restricted null gives p < 0.05: write the passage "
                         "by hand\n" + t.round(4).to_string())
    tab = rn.recording_table()
    n_rec = len(tab)
    night_n = tab.groupby("night").rec.transform("size")
    block_n = tab.groupby(["night", "block"]).rec.transform("size")
    in_night = int((night_n > 1).sum())
    in_block = int((block_n > 1).sum())
    new = (
        f"{START} and across positions within a night. That is justified "
        "under the null of Equation~\\eqref{eq:hypothesis}, conditional on "
        "the EEG and on the "
        "assignment of windows to recordings and participants, if a "
        "participant's labels do not depend on the night or the time of night; "
        "if labels and EEG both drifted across nights or through a night, "
        "exchanging labels across them could create or remove an association. "
        "The primary evaluation (Section~\\ref{sec:howmuch}) was therefore also "
        "tested under two restricted schemes: labels exchanged only among the "
        "recordings of one night, and only within consecutive blocks of three "
        "recordings of a night ordered by their position in it, which also "
        "preserves slow drifts through the night. Of the "
        f"{n_rec} recordings, {in_night} share a night with at least one other "
        f"and {in_block} share a block, and only those can exchange labels. "
        "Under the night restriction the four configurations give $p$ from "
        f"{t.night_p.min():.4f} to {t.night_p.max():.4f}, and under the block "
        f"restriction from {t.night_block_p.min():.4f} to "
        f"{t.night_block_p.max():.4f} (sequential, as in "
        "Section~\\ref{sec:howmuch}; after Holm's correction "
        f"{float(holm.min().min()):.4f} or more), so {END}")
    s = io.open(TEX, encoding="utf-8", newline="").read()
    if OLD in s:
        assert s.count(OLD) == 1
        s = s.replace(OLD, new)
    else:
        i = s.index(START)
        end = END if s.find(END, i) >= 0 else OLD_END
        j = s.index(end, i) + len(end)
        s = s[:i] + new + s[j:]
    assert s.count(START) == 1
    io.open(TEX, "w", encoding="utf-8", newline="").write(s)
    print(new)


if __name__ == "__main__":
    main()
