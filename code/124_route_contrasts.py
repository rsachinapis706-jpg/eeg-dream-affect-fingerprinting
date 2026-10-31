"""
Protocol contrasts with paired uncertainty (round 3, Reviewer 2, comment 4).

The two-route figures (window-shuffled minus recording-grouped, recording-
grouped minus participant-separated) are differences between evaluation
protocols. Reviewer 2 asks for paired uncertainty for these contrasts and for
the classifier-versus-identity-rule comparison, and for evidence on whether the
two predictors make the same predictions rather than only reaching similar
balanced accuracy.

Design
  Forest predictions are regenerated with exactly the settings of
  51_protocol_table.py (DEED) and 52_mechanism_refresh.py (Tononi):
  RandomForest(300, min_samples_leaf=2, random_state=seed), five folds,
  StratifiedKFold (window-shuffled) or StratifiedGroupKFold by recording or by
  participant, seeds 20260727..36. On Tononi, as in 52, test windows of a
  class absent from the training fold are skipped. The stored means must
  reproduce (checked, tolerance 5e-5). The identity rule is that of
  83_identity_rule_cv.py on the same window-shuffled folds.

  Paired participant bootstrap, 2,000 resamples: one draw of participants per
  resample, shared by every protocol, predictor and seed, so each contrast is
  computed on the same resampled participants; the statistic is the mean over
  the ten fold assignments, as reported. Predictions are held fixed, so the
  intervals reflect participant sampling, not refitting (stated).

  Agreement: share of window-shuffled test windows on which the forest and
  the identity rule predict the same class, against the share expected if
  they predicted independently with their own marginal frequencies.

Outputs: out/route_contrasts.csv, out/route_contrasts_preds.parquet,
         out/run_route_contrasts.log
"""
import importlib.util
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
spec = importlib.util.spec_from_file_location(
    "ir", os.path.join(HERE, "83_identity_rule_cv.py"))
ir = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ir)
SEEDS = ir.SEEDS
N_BOOT = 2000
N_JOBS = int(os.environ.get("ROUTE_JOBS", 4))
RF_THREADS = 3
LOG = os.path.join(OUT, "run_route_contrasts.log")
T0 = time.time()
PARTS = ("window-shuffled", "recording-grouped", "participant-separated")


def log(m=""):
    m = f"[{(time.time() - T0) / 60:6.1f} min] {m}"
    print(m, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(m + "\n")


def load():
    d = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    fc = [c for c in d.columns if c not in ir.META]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = d[d.E.isin(ir.VALENCE)]
    deed = (d[fc].to_numpy(float), d.E.map(ir.VALENCE).to_numpy(),
            d.subject.astype(str).to_numpy(), d.segment_id.astype(str).to_numpy())
    t = pd.read_parquet(os.path.join(OUT, "tononi_features.parquet"))
    tfc = [c for c in t.columns if c.startswith("ch")]
    t = t.replace([np.inf, -np.inf], np.nan).dropna(subset=tfc)
    ton = (t[tfc].to_numpy(float), t.dreaming.astype(str).to_numpy(),
           t.subject.astype(str).to_numpy(), t.recording.astype(str).to_numpy())
    return {"DEED": deed, "Tononi": ton}


def forest_run(name, X, y, subj, rec, part, seed):
    grp = {"window-shuffled": None, "recording-grouped": rec,
           "participant-separated": subj}[part]
    rows = []
    with threadpool_limits(RF_THREADS):
        for tr, te in ir.splits(y, grp, seed):
            if len(np.unique(y[tr])) < 2:
                continue
            if name == "Tononi":          # as 52: skip classes unseen in training
                te = te[np.isin(y[te], np.unique(y[tr]))]
                if not len(te):
                    continue
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                       n_jobs=RF_THREADS,
                                       random_state=seed).fit(X[tr], y[tr])
            rows.append(pd.DataFrame(dict(idx=te, pred=m.predict(X[te]))))
    p = pd.concat(rows, ignore_index=True)
    return pd.DataFrame(dict(dataset=name, predictor="forest", partition=part,
                             seed=seed, idx=p.idx, subj=subj[p.idx],
                             y=y[p.idx], pred=p.pred))


def rule_run(name, y, subj, seed):
    sp = list(ir.splits(y, None, seed))
    ide, _ = ir.rule(y, subj, sp)
    idx = np.arange(len(y))
    return pd.DataFrame(dict(dataset=name, predictor="identity rule",
                             partition="window-shuffled", seed=seed, idx=idx,
                             subj=subj, y=y, pred=ide))


def bal_matrix(g, uq, W):
    """Balanced accuracy of one run under each bootstrap weight vector."""
    cls = np.unique(g.y)
    si = np.searchsorted(uq, g.subj.to_numpy())
    tot = np.zeros((len(uq), len(cls)))
    tp = np.zeros((len(uq), len(cls)))
    yy, pp = g.y.to_numpy(), g.pred.to_numpy()
    for c, lab in enumerate(cls):
        m = yy == lab
        np.add.at(tot[:, c], si[m], 1)
        np.add.at(tp[:, c], si[m & (pp == lab)], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean((W @ tp) / (W @ tot), axis=1)


def main():
    data = load()
    jobs = [(n, p, s) for n in data for p in PARTS for s in SEEDS]
    log(f"forest runs: {len(jobs)}, {N_JOBS} workers x {RF_THREADS} threads")
    res = Parallel(n_jobs=N_JOBS)(delayed(forest_run)(n, *data[n], p, s)
                                  for n, p, s in jobs)
    res += [rule_run(n, data[n][1], data[n][2], s) for n in data for s in SEEDS]
    pr = pd.concat(res, ignore_index=True)
    pr.to_parquet(os.path.join(OUT, "route_contrasts_preds.parquet"))

    # reproduction of the stored means
    pt = pd.read_csv(os.path.join(OUT, "protocol_table.csv"))
    pt = pt[pt.target == "Dream valence"].set_index("held_out").bal_mean
    mr = pd.read_csv(os.path.join(OUT, "mechanism_refresh.csv"))
    mr = mr[mr.dataset == "Tononi"].set_index("quantity").bal_mean
    stored = {("DEED", "window-shuffled"): pt["Random (window-shuffled)"],
              ("DEED", "recording-grouped"): pt["Recording"],
              ("DEED", "participant-separated"): pt["Participant"],
              ("Tononi", "window-shuffled"): mr["acc window-shuffled"],
              ("Tononi", "recording-grouped"): mr["acc recording-grouped"],
              ("Tononi", "participant-separated"): mr["acc participant-wise"]}
    point = {}
    for (n, pred, part), g in pr.groupby(["dataset", "predictor", "partition"]):
        point[(n, pred, part)] = np.mean([balanced_accuracy_score(h.y, h.pred)
                                          for _, h in g.groupby("seed")])
    for k, v in stored.items():
        got = point[(k[0], "forest", k[1])]
        log(f"reproduce {k}: stored {v:.4f} regenerated {got:.4f}")
        assert abs(got - v) < 5e-5, f"no reproduction for {k}"

    rows = []
    for n in data:
        uq = np.unique(data[n][2])
        rng = np.random.default_rng(20261021)
        W = np.zeros((N_BOOT, len(uq)))
        for b in range(N_BOOT):
            np.add.at(W[b], rng.integers(0, len(uq), len(uq)), 1)
        boot = {}
        for (pred, part), g in pr[pr.dataset == n].groupby(["predictor",
                                                             "partition"]):
            boot[(pred, part)] = np.mean(
                [bal_matrix(h, uq, W) for _, h in g.groupby("seed")], axis=0)
        F = lambda p: ("forest", p)
        contrasts = [
            ("window-shuffled minus recording-grouped (forest)",
             F("window-shuffled"), F("recording-grouped")),
            ("recording-grouped minus participant-separated (forest)",
             F("recording-grouped"), F("participant-separated")),
            ("window-shuffled minus participant-separated (forest)",
             F("window-shuffled"), F("participant-separated")),
            ("forest minus identity rule, window-shuffled",
             F("window-shuffled"), ("identity rule", "window-shuffled")),
        ]
        for label, a, b in contrasts:
            d = boot[a] - boot[b]
            est = point[(n, *a)] - point[(n, *b)]
            rows.append(dict(dataset=n, contrast=label, estimate=est,
                             lo=np.nanpercentile(d, 2.5),
                             hi=np.nanpercentile(d, 97.5),
                             share_boot_below_zero=float(np.mean(d < 0))))
        # agreement between forest and identity rule on identical windows
        ag, ex = [], []
        for s in SEEDS:
            f = pr[(pr.dataset == n) & (pr.predictor == "forest")
                   & (pr.partition == "window-shuffled") & (pr.seed == s)]
            r = pr[(pr.dataset == n) & (pr.predictor == "identity rule")
                   & (pr.seed == s)].set_index("idx").loc[f.idx]
            ag.append(float(np.mean(f.pred.to_numpy() == r.pred.to_numpy())))
            pf = f.pred.value_counts(normalize=True)
            pq = r.pred.value_counts(normalize=True)
            ex.append(float(sum(pf.get(c, 0) * pq.get(c, 0)
                                for c in set(pf.index) | set(pq.index))))
        rows.append(dict(dataset=n, contrast="agreement forest vs identity "
                         "rule, window-shuffled", estimate=np.mean(ag),
                         lo=min(ag), hi=max(ag),
                         share_boot_below_zero=np.nan,
                         expected_if_independent=np.mean(ex)))
    out = pd.DataFrame(rows)
    out.round(4).to_csv(os.path.join(OUT, "route_contrasts.csv"), index=False)
    with pd.option_context("display.width", 220):
        log("\n" + out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
