"""
Test 1 (behavioural, no EEG): does pre-sleep film valence predict reported dream valence?

This is the gate for the mediation design. If film valence does not shift dream
affect at all, the "film -> EEG -> dream" path has no effect to mediate and the
study must be rescoped.

Design facts that constrain the test:
  - Film valence is assigned at the NIGHT level (constant within night, verified).
  - Multiple awakenings per night are not independent observations.
  - Subjects contribute multiple nights.
So the null distribution is built by permuting film labels across nights *within
subject*, which preserves both the nesting and each subject's own night count.

E0 ("no dream recall") is excluded from the valence analysis: it is an absence of
report, not a point on the valence scale. It is analysed separately as a recall
outcome.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")

RNG = np.random.default_rng(20260727)
N_PERM = 10000

# Verified against Dream_emotion counts in the ratings workbook:
# extremely unhappy=11 -> E1, relatively unhappy=79 -> E2, calm=219 -> E3,
# relatively happy=122 -> E4, extremely happy=24 -> E5.
E_TO_VALENCE = {"E1": 1, "E2": 2, "E3": 3, "E4": 4, "E5": 5}
FILM_TO_SCORE = {"Negative": -1, "Neutral": 0, "Positive": 1}

lines = []


def log(m=""):
    print(m)
    lines.append(str(m))


def night_level_permutation(df, stat_fn, n_perm=N_PERM):
    """Permute film labels across nights within subject; recompute stat each time."""
    nights = df[["subject", "oid", "film_score"]].drop_duplicates("oid")
    observed = stat_fn(df)
    null = np.empty(n_perm)

    by_subj = {s: g.oid.to_numpy() for s, g in nights.groupby("subject")}
    labels = {s: g.film_score.to_numpy() for s, g in nights.groupby("subject")}

    for i in range(n_perm):
        mapping = {}
        for s, oids in by_subj.items():
            mapping.update(dict(zip(oids, RNG.permutation(labels[s]))))
        shuffled = df.assign(film_score=df.oid.map(mapping))
        null[i] = stat_fn(shuffled)

    # two-sided, +1 correction (Phipson & Smyth)
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, p, null


def main():
    m = pd.read_csv(os.path.join(OUT, "master_table.csv"))

    log("=" * 68)
    log("TEST 1: PRE-SLEEP FILM VALENCE -> REPORTED DREAM VALENCE")
    log("=" * 68)
    log(f"Permutations: {N_PERM} (film labels shuffled across nights within subject)")
    log("")

    d = m[m.E.isin(E_TO_VALENCE)].copy()
    d["valence"] = d.E.map(E_TO_VALENCE)
    d["film_score"] = d.film.map(FILM_TO_SCORE)
    d = d.dropna(subset=["valence", "film_score", "subject", "oid"])

    log(f"Segments analysed : {len(d)}  (E0 excluded: {int((m.E == 'E0').sum())})")
    log(f"Nights            : {d.oid.nunique()}")
    log(f"Subjects          : {d.subject.nunique()}")
    log("")

    log("-- Mean dream valence by film condition --")
    summ = d.groupby("film").valence.agg(["count", "mean", "std"]).round(3)
    log(summ.to_string())
    log("")

    # --- segment-level correlation -------------------------------------
    def stat_seg(x):
        return stats.spearmanr(x.film_score, x.valence).statistic

    r_seg, p_seg, _ = night_level_permutation(d, stat_seg)
    log("-- Segment-level Spearman (film valence vs dream valence) --")
    log(f"rho = {r_seg:+.4f}   permutation p = {p_seg:.4f}   n = {len(d)}")
    log("")

    # --- night-level (one value per night; removes within-night pseudoreplication)
    nights = d.groupby(["subject", "oid", "film_score"], as_index=False).valence.mean()

    def stat_night(x):
        return stats.spearmanr(x.film_score, x.valence).statistic

    r_n, p_n, _ = night_level_permutation(nights, stat_night)
    log("-- Night-level Spearman (night-mean dream valence) --")
    log(f"rho = {r_n:+.4f}   permutation p = {p_n:.4f}   n = {len(nights)} nights")
    log("")

    # --- within-subject contrast: does the SAME person dream more negatively
    #     after a negative film than after their own positive-film night?
    piv = nights.pivot_table(index="subject", columns="film_score", values="valence")
    both = piv.dropna(subset=[-1, 1]) if {-1, 1}.issubset(piv.columns) else pd.DataFrame()
    log("-- Within-subject contrast (subjects with both Negative and Positive nights) --")
    if len(both) >= 5:
        diff = both[1] - both[-1]
        w = stats.wilcoxon(diff)
        log(f"n = {len(both)} subjects | mean(pos - neg) = {diff.mean():+.3f}")
        log(f"Wilcoxon signed-rank W = {w.statistic:.1f}, p = {w.pvalue:.4f}")
    else:
        log(f"Only {len(both)} subjects have both conditions -- test not run.")
    log("")

    # --- recall rate (E0) as a separate outcome ------------------------
    r = m.dropna(subset=["film"]).copy()
    r["no_recall"] = (r.E == "E0").astype(int)
    log("-- No-recall rate (E0) by film condition --")
    log((r.groupby("film").no_recall.agg(["count", "mean"]).round(3)).to_string())
    ct = pd.crosstab(r.film, r.no_recall)
    chi = stats.chi2_contingency(ct)
    log(f"chi2 = {chi.statistic:.3f}, p = {chi.pvalue:.4f} "
        f"(segment-level, not night-corrected -- indicative only)")
    log("")

    with open(os.path.join(OUT, "test1_induction.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"Wrote {OUT}\\test1_induction.txt")


if __name__ == "__main__":
    main()
