"""
Participant-bootstrap intervals for the risk-coverage comparison of PCD with
max-softmax and predictive entropy (Section IV-K), from the per-window output
of 77_pcd_revised.py.

For each rule the windows are ranked once by its score (PCD by p-value
ascending, max-softmax descending, entropy ascending), exactly as in 77's
risk_coverage(). A bootstrap resample draws participants with replacement;
each window then carries its participant's multiplicity as a weight, and the
retained set at coverage c is the top-ranked windows whose cumulative weight
reaches c times the total. Balanced accuracy of the retained set is computed
from weighted per-class counts. Because the ranking does not depend on the
resample, every resample is a cumulative sum over a fixed order.

Reported per condition and coverage level: the point estimate (no resampling,
identical to pcd_revised_riskcov.csv up to the k >= 10 floor of 77) and 95%
intervals for PCD minus max-softmax and PCD minus entropy. 1,000 resamples.
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
N_BOOT = 1000
COVS = np.round(np.arange(0.05, 1.0001, 0.05), 2)


def bal_at(order, yi, correct, w, ncls):
    """Weighted balanced accuracy of the top windows, for every coverage."""
    wo, yo, co = w[order], yi[order], correct[order]
    cw = np.cumsum(wo)
    total = cw[-1]
    out = []
    tot_c = np.zeros((len(order), ncls))
    tp_c = np.zeros((len(order), ncls))
    for c in range(ncls):
        m = yo == c
        tot_c[:, c] = np.cumsum(wo * m)
        tp_c[:, c] = np.cumsum(wo * m * co)
    for cov in COVS:
        k = int(np.searchsorted(cw, cov * total - 1e-9))
        k = min(max(k, 9), len(order) - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = tp_c[k] / tot_c[k]
        out.append(np.nanmean(r))
    return np.array(out)


def main():
    w = pd.read_parquet(os.path.join(OUT, "pcd_revised_windows.parquet"))
    rows = []
    rng = np.random.default_rng(20261001)
    for cond, g in w.groupby("condition"):
        g = g.reset_index(drop=True)
        cls = sorted(g.y.unique())
        yi = g.y.map({c: i for i, c in enumerate(cls)}).to_numpy()
        correct = (g.y == g.pred).to_numpy().astype(float)
        orders = {"PCD": np.argsort(g.p_pcd.to_numpy(), kind="stable"),
                  "Max-softmax": np.argsort(-g.msp.to_numpy(), kind="stable"),
                  "Entropy": np.argsort(g.ent.to_numpy(), kind="stable")}
        uq, sidx = np.unique(g.subj.to_numpy(), return_inverse=True)
        ones = np.ones(len(g))
        est = {k: bal_at(o, yi, correct, ones, len(cls)) for k, o in orders.items()}
        dm, de = [], []
        for _ in range(N_BOOT):
            mult = np.bincount(rng.integers(0, len(uq), len(uq)),
                               minlength=len(uq)).astype(float)
            wt = mult[sidx]
            b = {k: bal_at(o, yi, correct, wt, len(cls)) for k, o in orders.items()}
            dm.append(b["PCD"] - b["Max-softmax"])
            de.append(b["PCD"] - b["Entropy"])
        dm, de = np.array(dm), np.array(de)
        for i, cov in enumerate(COVS):
            rows.append(dict(
                condition=cond, coverage=cov,
                pcd=est["PCD"][i], msp=est["Max-softmax"][i],
                entropy=est["Entropy"][i],
                pcd_minus_msp=est["PCD"][i] - est["Max-softmax"][i],
                msp_lo=np.nanpercentile(dm[:, i], 2.5),
                msp_hi=np.nanpercentile(dm[:, i], 97.5),
                pcd_minus_entropy=est["PCD"][i] - est["Entropy"][i],
                ent_lo=np.nanpercentile(de[:, i], 2.5),
                ent_hi=np.nanpercentile(de[:, i], 97.5)))
        print(cond, "done", flush=True)
    t = pd.DataFrame(rows)
    t.round(4).to_csv(os.path.join(OUT, "pcd_riskcov_ci.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 60):
        print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
