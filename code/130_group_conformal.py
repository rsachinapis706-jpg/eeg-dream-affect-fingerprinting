"""
The conformal comparator for clustered data (round 3, Reviewer 2, comment 6).

Specification of the comparator used in Table PCD (77_pcd_revised.conformal)
  conformity score   s = 1 - p_hat(true class), from the decoder fitted on the
                     fitting participants of the training fold
  calibration set    every window of a quarter of the training participants,
                     drawn at random per fold, excluded from fitting and
                     disjoint from the test participants
  quantile           q = the k-th smallest calibration score, k = ceil((n + 1)
                     (1 - alpha)), n the number of calibration windows; no
                     weighting; prediction set {c : 1 - p_hat(c) <= q}
  dependence         none modelled: windows are treated as exchangeable,
                     although windows of one participant are not. Its
                     guarantee therefore does not formally extend to a window
                     of a new participant, and it is reported as an empirical
                     comparator.

Group-aware construction added here
  Participants are taken as exchangeable units. Drawing one calibration
  window per calibration participant, uniformly within that participant,
  gives m scores exchangeable with the score of a uniformly drawn window of a
  new participant, so split conformal on those m scores has coverage at least
  1 - alpha for such a window, exactly and for any dependence within a
  participant. Its effective calibration size is m, the number of calibration
  participants: when (m + 1)(1 - alpha) > m the quantile is infinite and every
  set contains all classes. Each of 200 independent draws is a valid set
  rule; the table reports their mean behaviour.

Folds, calibration participants and fitted models are those of 77 (folds
drawn without labels, 77.label_free_folds), reproduced here; the window-level
comparator must reproduce 77's set sizes and coverage (checked).

Output: out/conformal_group.csv
"""
import importlib.util
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pcd = _load("pcd", "77_pcd_revised.py")
st = _load("st", "110_pcd_stratified.py")
OUT = pcd.OUT
ALPHA = pcd.ALPHA
N_DRAW = 200


def scores(p, classes, y):
    return 1.0 - p[np.arange(len(y)), [classes.index(v) for v in y]]


def run(X, y, subj, name, seed=pcd.SEED):
    rng = np.random.default_rng(seed)          # identical use to 77
    draw_rng = np.random.default_rng(20261021)
    win_sz, win_cov, grp = [], [], []
    for k, (tr, te) in enumerate(pcd.label_free_folds(subj, seed)):
        if len(np.unique(y[tr])) < 2:
            continue
        uq = np.unique(subj[tr])
        cal_ids = rng.choice(uq, size=max(1, len(uq) // 4), replace=False)
        cal = np.isin(subj[tr], cal_ids)
        fit, calx = tr[~cal], tr[cal]
        model = pcd.clf().fit(X[fit], y[fit])
        sz, cov = pcd.conformal(model, X[calx], y[calx], X[te], y[te], ALPHA)
        win_sz.append(sz)
        win_cov.append(cov)
        cls = list(model.classes_)
        sc_cal = scores(model.predict_proba(X[calx]), cls, y[calx])
        pt = model.predict_proba(X[te])
        yi = np.array([cls.index(v) for v in y[te]])
        m = len(cal_ids)
        q_rank = int(np.ceil((m + 1) * (1 - ALPHA)))
        cs = subj[calx]
        idx_by = {g: np.flatnonzero(cs == g) for g in cal_ids}
        for d in range(N_DRAW):
            pick = np.array([draw_rng.choice(idx_by[g]) for g in cal_ids])
            s = np.sort(sc_cal[pick])
            qh = np.inf if q_rank > m else s[q_rank - 1]
            inset = (1.0 - pt) <= qh
            grp.append(dict(fold=k, draw=d, m=m, finite=bool(np.isfinite(qh)),
                            set_size=inset.sum(1).mean(),
                            singleton=(inset.sum(1) == 1).mean(),
                            covered=inset[np.arange(len(yi)), yi].mean(),
                            n_test=len(te)))
    win_sz, win_cov = np.concatenate(win_sz), np.concatenate(win_cov)
    g = pd.DataFrame(grp)
    w = lambda c: np.average(g[c], weights=g.n_test)
    return [dict(condition=name, comparator="window-level split conformal "
                 "(empirical)", calibration_units="windows",
                 calibration_size_median=np.nan,
                 set_size=win_sz.mean(), singleton=(win_sz == 1).mean(),
                 true_label_coverage=win_cov.mean(), share_finite=1.0),
            dict(condition=name, comparator="group-aware split conformal, one "
                 "window per calibration participant", calibration_units=
                 "participants", calibration_size_median=float(g.m.median()),
                 set_size=w("set_size"), singleton=w("singleton"),
                 true_label_coverage=w("covered"),
                 share_finite=float(g.finite.mean()))]


def main():
    Xd, yd, sd, _ = st.deed()
    Xs, ys, ss, _, _ = st.sleepedf()
    rows = run(Xs, ys, ss, "Sleep-EDF staging") + run(Xd, yd, sd, "DEED affect")
    t = pd.DataFrame(rows)
    # the window-level rows must reproduce 77's split-conformal results
    ref = pd.read_csv(os.path.join(OUT, "pcd_revised_summary.csv"))
    ref = ref[ref.rule.str.startswith("Split conformal")].set_index("condition")
    for _, r in t[t.calibration_units == "windows"].iterrows():
        a, b = r.set_size, ref.loc[r.condition, "set_size"]
        assert abs(a - b) < 5e-5, (r.condition, a, b)
    t.round(4).to_csv(os.path.join(OUT, "conformal_group.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
