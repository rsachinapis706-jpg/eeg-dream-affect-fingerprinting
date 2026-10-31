"""
Randomised-label ladder against the published figures (round 3, Reviewer 2,
comment 7).

0.9403 counts synthetic test observations; on real test windows alone the
same rung gives 0.9245, which lies within the published range (0.9233,
0.9383), not above it. Every statement now keeps the two apart. The two
ladder values are read from out/random_label_ladder.csv; the published
figures are quoted. Idempotent.
"""
import io
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = r"E:\Dream_Analysis\accesstemp.tex"


def main():
    s = io.open(TEX, encoding="utf-8", newline="").read()
    # rung 5 with labels permuted across all recordings, from the ladder's
    # result file; the published accuracies (Mojdehbar et al., Mahmoudi et
    # al.) are quoted from those papers
    lad = pd.read_csv(os.path.join(HERE, "out", "random_label_ladder.csv"))
    r5 = lad[(lad["mode"] == "free") & (lad.rung == 5)
             & (lad.column == "leaky")].iloc[0]
    allp, real = f"{r5.acc:.4f}", f"{r5.acc_real:.4f}"
    lo, hi = "0.9233", "0.9383"
    for v in (allp, real, lo, hi):
        assert v in s, v
    assert float(lo) < float(real) < float(hi) < float(allp)
    edits = [
        (f"With labels randomised across recordings the same construction "
         f"reaches {allp}, and {real} on real test windows alone, above both "
         f"published figures of {lo} and {hi}.",
         f"With labels randomised across recordings the same construction "
         f"reaches {allp} counting synthetic test windows and {real} on real "
         f"test windows alone, which lies within the published range of {lo} "
         f"to {hi}."),
        (f"0.9123 at rung 2 and {allp} at rung 5 when labels are permuted "
         f"across all recordings, the latter above both published figures, and "
         f"{real} on the real test windows alone.",
         f"0.9123 at rung 2 and {allp} at rung 5 when labels are permuted "
         f"across all recordings, a figure that counts synthetic test windows; "
         f"on the real test windows alone rung 5 gives {real}, within the "
         f"published range of {lo} to {hi}."),
        (f"reach 0.9055 on the same data with the reported labels and {allp}, "
         f"above {hi}, with labels randomised across recordings.",
         f"reach 0.9055 on the same data with the reported labels and, with "
         f"labels randomised across recordings, {allp} counting synthetic test "
         f"windows and {real} on real test windows alone, below the {hi} that "
         f"study reports."),
    ]
    for old, new in edits:
        if old in s:
            s = s.replace(old, new)
        assert new in s, new[:80]
    io.open(TEX, "w", encoding="utf-8", newline="").write(s)
    print(f"{len(edits)} statements now separate {allp} (with synthetic test "
          f"windows) from {real} (real windows only)")


if __name__ == "__main__":
    main()
