"""
Result files for within-participant figures that the manuscript prints but no
result file held (round 3, Reviewer 2, comment 7: keyed provenance).

  1. Effect sizes for the frozen CBraMod embeddings with windows in units of
     100 uV (Section IV-J), from cbramod_within_subject_uv100.csv, with the
     formulas of 45_statistical_foundation.py (Cohen's d_z, matched-pairs
     rank-biserial r, one-sided Wilcoxon signed-rank p).
  2. Power of a paired test at the observed spectral effect with the
     observed 28 participants: a one-sided one-sample t-test (noncentral t),
     the one-sided Wilcoxon signed-rank test with the observed differences
     resampled, and the number of participants a t-test would need for
     power 0.80. Nothing is fitted; only the stored per-participant results
     are used.

Outputs: out/within_effects_uv100.csv, out/within_paired_power.csv
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ALPHA = 0.05
B = 10000
SEED = 20260727


def rank_biserial(d):
    """As in 45_statistical_foundation.py: (W+ - W-) / (W+ + W-)."""
    d = d[d != 0]
    r = stats.rankdata(np.abs(d))
    wp, wn = r[d > 0].sum(), r[d < 0].sum()
    return (wp - wn) / (wp + wn)


def effects(fname):
    w = pd.read_csv(os.path.join(OUT, fname)).dropna(subset=["bacc", "null"])
    d = (w.bacc - w.null).to_numpy()
    nz = d[d != 0]
    return dict(source=fname, n=len(w), observed=w.bacc.mean(),
                null=w.null.mean(), difference=d.mean(),
                cohens_dz=d.mean() / d.std(ddof=1), rank_biserial=rank_biserial(d),
                wilcoxon_p=stats.wilcoxon(nz, alternative="greater").pvalue,
                n_above=int((d > 0).sum()), n_below=int((d < 0).sum())), d


def t_power(dz, n):
    """One-sided one-sample t-test power at standardised effect dz."""
    crit = stats.t.ppf(1 - ALPHA, n - 1)
    return 1 - stats.nct.cdf(crit, n - 1, dz * np.sqrt(n))


def main():
    e, _ = effects("cbramod_within_subject_uv100.csv")
    pd.DataFrame([e]).to_csv(os.path.join(OUT, "within_effects_uv100.csv"),
                             index=False)
    print(pd.DataFrame([e]).round(4).to_string(index=False))

    s, d = effects("within_subject.csv")
    n, dz = s["n"], s["cohens_dz"]
    rng = np.random.default_rng(SEED)
    hits = 0
    for _ in range(B):
        x = rng.choice(d, size=n, replace=True)
        x = x[x != 0]
        hits += stats.wilcoxon(x, alternative="greater").pvalue < ALPHA
    n80 = next(m for m in range(5, 1000) if t_power(dz, m) >= 0.80)
    row = dict(source="within_subject.csv", n=n, cohens_dz=dz, alpha=ALPHA,
               power_t=t_power(dz, n), power_signed_rank_resampled=hits / B,
               resamples=B, seed=SEED, n_for_power_080_t=n80)
    pd.DataFrame([row]).to_csv(os.path.join(OUT, "within_paired_power.csv"),
                               index=False)
    print(pd.DataFrame([row]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
