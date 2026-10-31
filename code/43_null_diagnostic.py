"""
Does the DEED null depend on the classifier configuration?

While building the power analysis, one configuration -- logistic regression with
standardisation and balanced class weights -- produced a subject-wise balanced
accuracy of 0.3785 against a permutation null whose 95th percentile was 0.3675.
Taken at face value that would contradict the central claim of this paper, which
is that no participant-generalizable dream-affect signal is detectable. It must
therefore be checked directly rather than left implicit in a power curve.

Two possibilities are distinguished here.

  (i)  There is weak signal that the classifier used for the headline result
       (an unweighted random forest, no standardisation) fails to find. If so
       the manuscript's claim needs qualifying and the finding is real.

  (ii) The permutation null is mis-specified for this configuration. Labels are
       shuffled across recordings WITHIN participant, which preserves each
       participant's label multiset and therefore preserves the between-
       participant component of the identity-label association. A classifier
       that exploits the global class prior differently under permuted labels
       than under real ones would then be compared against a null that is too
       low, and would appear significant without any within-participant signal.

The diagnostic runs every configuration against BOTH nulls:

  within-participant  labels shuffled across a participant's own recordings
                      (tests for within-participant recording-level signal)
  unrestricted        labels shuffled across all recordings in the cohort
                      (tests for any label structure at all, including the
                      between-participant component)

If a configuration is significant against the within-participant null but not
against the unrestricted one, the effect is a property of the null construction
rather than evidence of decodable affect.
"""
import os
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.dummy import DummyClassifier

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")

SEED = 20260727
N_SPLITS = 5
PERMS = {"lr": 300, "rf": 100, "dummy": 100}
N_JOBS = 20
VALENCE = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
META = {"segment_id", "subject", "oid", "epoch_idx", "n_epochs",
        "epochs_from_end", "E", "film", "stage", "has_text"}

# Ordered cheapest-and-most-suspect first, so the decisive result lands
# early and is written to disk before the expensive forests run.
CONFIGS = {
    "LogReg, scaled, balanced weight": ("lr", True, "balanced"),
    "LogReg, scaled, no weight": ("lr", True, None),
    "Dummy: stratified": ("dummy", False, None),
    "RF, no scaling, no weight  (paper)": ("rf", False, None),
    "RF, scaled, balanced weight": ("rf", True, "balanced"),
}

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def load():
    d = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    fc = [c for c in d.columns if c not in META]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = d[d.E.isin(VALENCE)]
    return (d[fc].to_numpy(float), d.E.map(VALENCE).to_numpy(),
            d.subject.to_numpy(), d.segment_id.to_numpy())


def build(kind, scale, weight, seed):
    if kind == "rf":
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   class_weight=weight, n_jobs=1,
                                   random_state=seed)
    elif kind == "lr":
        m = LogisticRegression(C=0.1, max_iter=2000, class_weight=weight)
    else:
        m = DummyClassifier(strategy="stratified", random_state=seed)
    return make_pipeline(StandardScaler(), m) if scale else m


def run(X, y, subj, kind, scale, weight, seed=SEED):
    sp = StratifiedGroupKFold(N_SPLITS, shuffle=True,
                              random_state=seed).split(X, y, subj)
    yt, yp = [], []
    for tr, te in sp:
        if len(np.unique(y[tr])) < 2:
            continue
        m = build(kind, scale, weight, seed).fit(X[tr], y[tr])
        yp.append(m.predict(X[te])); yt.append(y[te])
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return accuracy_score(yt, yp), balanced_accuracy_score(yt, yp)


def perm_within(y, subj, rec, rng):
    out = y.copy()
    for g in np.unique(subj):
        m = subj == g
        recs = np.unique(rec[m])
        if len(recs) < 2:
            continue
        labs = rng.permutation(np.array([y[rec == r][0] for r in recs]))
        for r, lab in zip(recs, labs):
            out[rec == r] = lab
    return out


def perm_free(y, subj, rec, rng):
    out = y.copy()
    recs = np.unique(rec)
    labs = rng.permutation(np.array([y[rec == r][0] for r in recs]))
    for r, lab in zip(recs, labs):
        out[rec == r] = lab
    return out


def one(X, y, subj, rec, kind, scale, weight, mode, s):
    rng = np.random.default_rng(90_000 + s)
    yl = (perm_within if mode == "within" else perm_free)(y, subj, rec, rng)
    if len(np.unique(yl)) < 2:
        return np.nan
    return run(X, yl, subj, kind, scale, weight)[1]


def main():
    X, y, subj, rec = load()
    log("=" * 78)
    log("IS THE DEED NULL CONFIGURATION-DEPENDENT?")
    log("=" * 78)
    log(f"windows {len(y)} | participants {len(np.unique(subj))} | "
        f"recordings {len(np.unique(rec))}")
    log(f"permutations per null: {PERMS}   chance 0.3333")
    log("")
    log(f"{'configuration':<36}{'observed':>10}"
        f"{'within-p null':>16}{'p':>7}{'free null':>12}{'p':>7}")
    log("-" * 96)

    rows = []
    for name, (kind, scale, weight) in CONFIGS.items():
        _, obs = run(X, y, subj, kind, scale, weight)
        res = {}
        nperm = PERMS[kind]
        for mode in ("within", "free"):
            vals = Parallel(n_jobs=N_JOBS)(
                delayed(one)(X, y, subj, rec, kind, scale, weight, mode, s)
                for s in range(nperm))
            v = np.array([a for a in vals if np.isfinite(a)])
            p = (np.sum(v >= obs) + 1) / (len(v) + 1)
            res[mode] = (v.mean(), float(np.percentile(v, 95)), p)
        log(f"{name:<36}{obs:>10.4f}"
            f"{res['within'][0]:>10.4f}/{res['within'][1]:.4f}"
            f"{res['within'][2]:>7.3f}"
            f"{res['free'][0]:>7.4f}/{res['free'][1]:.4f}"
            f"{res['free'][2]:>7.3f}")
        rows.append(dict(config=name, observed=obs,
                         within_null_mean=res["within"][0],
                         within_null_p95=res["within"][1],
                         within_p=res["within"][2],
                         free_null_mean=res["free"][0],
                         free_null_p95=res["free"][1],
                         free_p=res["free"][2], n_perm=nperm))
        # write after every configuration so partial results survive
        pd.DataFrame(rows).round(4).to_csv(
            os.path.join(OUT, "null_diagnostic.csv"), index=False)

    r = pd.DataFrame(rows)
    r.round(4).to_csv(os.path.join(OUT, "null_diagnostic.csv"), index=False)

    log("")
    log("=" * 78)
    log("READING")
    log("=" * 78)
    log("Columns give the null mean / 95th percentile and the permutation p.")
    log("")
    sig_w = r[(r.within_p < 0.05) & (~r.config.str.startswith("Dummy"))]
    sig_f = r[(r.free_p < 0.05) & (~r.config.str.startswith("Dummy"))]
    log(f"configurations significant against within-participant null : "
        f"{len(sig_w)} of {len(r) - 1}")
    log(f"configurations significant against unrestricted null       : "
        f"{len(sig_f)} of {len(r) - 1}")
    log("")
    if len(sig_f) == 0 and len(sig_w) > 0:
        log("Significance appears only against the within-participant null.")
        log("That null preserves each participant's label multiset and so")
        log("preserves the between-participant identity-label association,")
        log("which makes it the wrong reference for a claim about decodable")
        log("affect. The unrestricted null, which destroys all label")
        log("structure, shows no configuration beating chance. The paper's")
        log("null stands; the power analysis must use the unrestricted null.")
    elif len(sig_f) > 0:
        log("At least one configuration beats the unrestricted null. This")
        log("would be genuine evidence of decodable structure and the")
        log("manuscript's claim must be qualified accordingly. Report it.")
    else:
        log("No configuration beats either null. The paper's claim is")
        log("unchanged and is robust to classifier configuration.")

    with open(os.path.join(OUT, "test33_null_diagnostic.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"\nWrote {OUT}\\null_diagnostic.csv")


if __name__ == "__main__":
    main()
