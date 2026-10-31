"""
Session- and temporal-block-restricted permutation nulls (round 3, Reviewer
2, comment 1).

Why
  The within-participant null exchanges a participant's recording labels
  across nights and across positions within a night. Under the null stated in
  the manuscript -- conditional on the EEG and on the assignment of windows to
  recordings and participants, the labels are invariant to those exchanges --
  that is exact; but if labels depend on the night or on the time of night,
  and the EEG does too, exchanging across nights or positions could create or
  remove an association. Two restricted nulls test how much the conclusions
  depend on that:
    night        labels exchanged only among recordings of the same night;
    night_block  recordings of a night ordered by their position (the R index
                 of the file name) and cut into consecutive blocks of three;
                 labels exchanged only within a block.
  Both preserve every night's label composition; the second also preserves
  slow drifts across the night.

Procedure
  The primary procedure of 120_primary_random.py (the four Table 5
  candidates, 20 random class-balanced participant partitions, folds
  recomputed from each labelling, mean over partitions) with sequential Monte
  Carlo p-values (H = 20, at most 1,000). Observed values are 120's.

Outputs: out/restricted_nulls_raw.csv, out/restricted_nulls_summary.csv
"""
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t5 = _load("t5", "65_config_table.py")
pr = _load("pr", "120_primary_random.py")
OUT = t5.OUT
MODES = ("night", "night_block")
BLOCK = 3
PERM_BASE = 700_000
N_JOBS = int(os.environ.get("RESTR_JOBS", 4))
RAW = os.path.join(OUT, "restricted_nulls_raw.csv")
LOG = os.path.join(OUT, "run_restricted_nulls.log")
T0 = time.time()


def log(m=""):
    m = f"[{(time.time() - T0) / 60:7.1f} min] {m}"
    print(m, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(m + "\n")


def recording_table():
    d = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"),
                        columns=["segment_id", "oid", "E"])
    d = d[d.E.isin(t5.VALENCE)].drop_duplicates("segment_id")
    pos = d.segment_id.str.extract(r"_R(\d+)_")[0].astype(int)
    r = pd.DataFrame(dict(rec=d.segment_id.to_numpy(), night=d.oid.to_numpy(),
                          pos=pos.to_numpy()))
    r = r.sort_values(["night", "pos", "rec"]).reset_index(drop=True)
    r["block"] = r.groupby("night").cumcount() // BLOCK
    return r


def permuted(y, rec, tab, b, mode):
    rng = np.random.default_rng(PERM_BASE + b)
    lab = pd.Series(y).groupby(rec).first()
    t = tab.assign(y=lab.loc[tab.rec].to_numpy())
    keys = ["night"] if mode == "night" else ["night", "block"]
    new = t.y.to_numpy().copy()
    for _, idx in t.groupby(keys).indices.items():
        if len(idx) > 1:
            new[idx] = rng.permutation(new[idx])
    lut = dict(zip(t.rec, new))
    return np.array([lut[r] for r in rec])


def task(X, y, subj, rec, tab, cfg, mode, b):
    yl = permuted(y, rec, tab, b, mode)
    return dict(config=cfg, mode=mode, b=b,
                stat=float(np.mean(pr.partition_mean(X, yl, subj, cfg))))


def main():
    X, y, subj, rec = t5.load()
    tab = recording_table()
    p_raw = pd.read_csv(pr.RAW)
    obs = {c: float(p_raw[(p_raw.config == c) & (p_raw["mode"] == "observed")]
                    .stat.iloc[0]) for c in t5.CANDIDATES}
    raw = pd.read_csv(RAW) if os.path.exists(RAW) else pd.DataFrame(
        columns=["config", "mode", "b", "stat"])
    log(f"RESTRICTED NULLS {MODES}, block {BLOCK}; observed from 120: "
        + ", ".join(f"{c[:12]} {v:.4f}" for c, v in obs.items()))
    step = int(os.environ.get("RESTR_STEP", 16))
    while True:
        todo = []
        for mode in MODES:
            for c in t5.CANDIDATES:
                st = pr.seq_state(raw, c, mode, obs[c])
                if st["done"]:
                    continue
                have = set(raw[(raw.config == c) & (raw["mode"] == mode)].b)
                # it cannot stop in fewer permutations than it lacks
                # exceedances, so none in the batch is surplus to the rule
                todo += [(c, mode, b) for b in
                         [x for x in range(pr.B_MAX) if x not in have]
                         [:min(step, pr.H - st["count"])]]
        if not todo:
            break
        res = Parallel(n_jobs=N_JOBS)(delayed(task)(X, y, subj, rec, tab, c,
                                                    m, b) for c, m, b in todo)
        raw = pd.concat([raw, pd.DataFrame(res)], ignore_index=True)
        raw.to_csv(RAW, index=False)
        log(" | ".join(f"{m[:5]}:{c[:10]} " + (
            lambda st: f"k={st['k']}" + (f" p={st['p']:.4f}" if st["done"]
                                          else ""))(pr.seq_state(raw, c, m, obs[c]))
            for m in MODES for c in t5.CANDIDATES))
    rows = []
    for c in t5.CANDIDATES:
        row = dict(config=c, observed_mean=obs[c])
        for m in MODES:
            st = pr.seq_state(raw, c, m, obs[c])
            n = raw[(raw.config == c) & (raw["mode"] == m)]
            row.update({f"{m}_k": st["k"], f"{m}_p": st["p"],
                        f"{m}_null_mean": n.stat.mean()})
        rows.append(row)
    t = pd.DataFrame(rows).set_index("config")
    for m in MODES:
        p = t[f"{m}_p"].sort_values()
        k, run, holm = len(p), 0.0, {}
        for i, (c, v) in enumerate(p.items()):
            run = max(run, min(1.0, (k - i) * v))
            holm[c] = run
        t[f"{m}_p_holm"] = pd.Series(holm)
    t.round(4).to_csv(os.path.join(OUT, "restricted_nulls_summary.csv"))
    log("\n" + t.round(4).to_string())


if __name__ == "__main__":
    main()
