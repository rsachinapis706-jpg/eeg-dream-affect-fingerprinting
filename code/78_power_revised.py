"""
Sensitivity of the participant-separated protocol, from a true null baseline
(Reviewer 1, comment 7).

What was wrong
--------------
45_statistical_foundation.py injected signal into the REAL labels, which
already carry the weak logistic-regression association (0.378 balanced
accuracy). At 34 participants the un-injected data (delta = 0) exceeded the
null's 95th percentile in all 30 simulations, so the reported "80% power near
0.377" measured the existing effect, not the protocol's sensitivity. The
simulations also varied only the injection direction, and 30 per level leave a
Monte Carlo standard error of up to 0.09.

Design
------
  baseline     labels permuted across ALL recordings (the unrestricted null of
               Table 5): every recording keeps a single label and the class
               proportions are kept, but no association between EEG and label
               survives, within or between participants
  injection    a class-dependent mean shift of delta standard deviations along
               a random unit direction in the standardised feature space,
               codes -1, 0, +1 for negative, neutral, positive
  statistic    pooled out-of-fold balanced accuracy of the pre-specified
               configuration (logistic regression, C = 0.1, scaled,
               class-balanced), participant-separated five-fold CV
  threshold    95th percentile of the statistic across the delta = 0
               simulations of that cohort size, which ARE draws from the null
  cohorts      10, 20 and 34 participants; every simulation draws a new
               participant subset, a new permutation and a new direction
  simulations  100 per cohort and delta; Monte Carlo SE = sqrt(p(1-p)/100)
  multiplicity one pre-specified configuration, so none is applied

This gives the sensitivity of the protocol to this particular signal structure
-- a linear mean shift shared by all participants. It is not a general bound:
signal that is nonlinear, participant-specific or confined to a subset of
features would be detected differently.
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
LOG = os.path.join(OUT, "run_power_revised.log")

SEED = 20260727
N_SIM = 100
DELTAS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20]
COHORTS = [10, 20, 34]
N_JOBS = 12
VALENCE = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
CODE = {"neg": -1.0, "neu": 0.0, "pos": 1.0}
META = {"segment_id", "subject", "oid", "epoch_idx", "n_epochs",
        "epochs_from_end", "E", "film", "stage", "has_text"}

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def one(Xs, y, subj, rec, n_sub, delta, s):
    rng = np.random.default_rng(10_000_000 + 1000 * n_sub + 100_003 * s
                                + int(delta * 1e4))
    ids = np.unique(subj)
    pick = ids if n_sub >= len(ids) else rng.choice(ids, n_sub, replace=False)
    m = np.isin(subj, pick)
    X, yy, g, r = Xs[m], y[m], subj[m], rec[m]
    recs, first = np.unique(r, return_index=True)
    lut = dict(zip(recs, rng.permutation(yy[first])))     # unrestricted null
    yn = np.array([lut[k] for k in r])
    u = rng.normal(size=X.shape[1]); u /= np.linalg.norm(u)
    Xi = X + np.array([CODE[v] for v in yn])[:, None] * delta * u[None, :]
    yt, yp = [], []
    with threadpool_limits(1, user_api="blas"):
        for tr, te in StratifiedGroupKFold(5, shuffle=True,
                                           random_state=SEED).split(Xi, yn, g):
            if len(np.unique(yn[tr])) < 2:
                continue
            mdl = make_pipeline(StandardScaler(), LogisticRegression(
                C=0.1, max_iter=2000, class_weight="balanced"))
            yp.append(mdl.fit(Xi[tr], yn[tr]).predict(Xi[te]))
            yt.append(yn[te])
    return dict(cohort=n_sub, delta=delta, sim=s,
                bal=balanced_accuracy_score(np.concatenate(yt),
                                            np.concatenate(yp)))


def main():
    d = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    fc = [c for c in d.columns if c not in META]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = d[d.E.isin(VALENCE)].reset_index(drop=True)
    Xs = StandardScaler().fit_transform(d[fc].to_numpy(float))
    y = d.E.map(VALENCE).to_numpy()
    subj, rec = d.subject.to_numpy(), d.segment_id.to_numpy()
    t0 = time.time()
    log("=" * 78)
    log("SENSITIVITY FROM A TRUE NULL BASELINE")
    log("=" * 78)
    jobs = [(n, dl, s) for n in COHORTS for dl in DELTAS for s in range(N_SIM)]
    res = pd.DataFrame(Parallel(n_jobs=N_JOBS)(
        delayed(one)(Xs, y, subj, rec, n, dl, s) for n, dl, s in jobs))
    res.to_csv(os.path.join(OUT, "power_revised_raw.csv"), index=False)
    rows = []
    for n in COHORTS:
        null = res[(res.cohort == n) & (res.delta == 0)].bal
        thr = float(np.percentile(null, 95))
        log(f"cohort {n:>2}: null mean {null.mean():.4f}, 95th percentile "
            f"{thr:.4f}")
        for dl in DELTAS:
            v = res[(res.cohort == n) & (res.delta == dl)].bal
            pw = float((v > thr).mean())
            se = float(np.sqrt(pw * (1 - pw) / len(v)))
            rows.append(dict(cohort=n, delta=dl, achieved_bal=v.mean(),
                             achieved_sd=v.std(), power=pw, mc_se=se,
                             threshold=thr, n_sim=len(v)))
            log(f"   delta {dl:.2f}: achieved bal {v.mean():.4f} "
                f"(sd {v.std():.4f})  power {pw:.2f} +/- {se:.2f}")
    out = pd.DataFrame(rows)
    out.round(4).to_csv(os.path.join(OUT, "power_revised.csv"), index=False)
    log(f"[{(time.time() - t0) / 60:.1f} min]")
    with open(LOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
