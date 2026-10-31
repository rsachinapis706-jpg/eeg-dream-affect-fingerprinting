"""
Seed sensitivity of CBraMod fine-tuning under the revised settings (Reviewer 1,
comment 9: sensitivity to initialisation). Seeds 20260727 (from 76's main
runs), 20260728 and 20260729; pretrained and randomly initialised.

Per seed: pooled balanced accuracy over all held-out windows with a 2,000-
resample participant bootstrap interval, constant folds, median chosen epoch,
and the paired pretrained-minus-scratch difference on the same resampled
participants. Across seeds: mean and SD (ddof=0) of pooled balanced accuracy.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SEEDS = [20260727, 20260728, 20260729]
N_BOOT = 2000
# CBRAMOD_SCALE=uv100: the runs with inputs in units of 100 uV (144, 76)
SCALE = os.environ.get("CBRAMOD_SCALE", "")
assert SCALE in ("", "uv100"), SCALE
SFX = f"_{SCALE}" if SCALE else ""


def load(cfg, seed):
    f = pd.read_csv(os.path.join(OUT, f"cbramod_diag_{cfg}{SFX}_{seed}_folds.csv"),
                    dtype={"held_out": str})
    assert len(f) == 34 and not f.held_out.duplicated().any(), (cfg, seed)
    per = {}
    for s in f.held_out:
        a = np.load(os.path.join(OUT, f"cbramod_diag_{cfg}{SFX}_{seed}_{s}_pred.npy"))
        per[s] = (a[0], a[1])
    return f, per


def pooled(per, ids):
    return balanced_accuracy_score(np.concatenate([per[i][0] for i in ids]),
                                   np.concatenate([per[i][1] for i in ids]))


def main():
    rows, pairs = [], []
    for seed in SEEDS:
        data = {c: load(c, seed) for c in ("revised", "scratch_revised")}
        ids = sorted(data["revised"][1])
        assert ids == sorted(data["scratch_revised"][1])
        rng = np.random.default_rng(seed)
        draws = [rng.choice(ids, len(ids)) for _ in range(N_BOOT)]
        boots = {}
        for c, (f, per) in data.items():
            bb = np.array([pooled(per, d) for d in draws])
            boots[c] = bb
            rows.append(dict(config=c, seed=seed, pooled_bal=pooled(per, ids),
                             bal_lo=np.percentile(bb, 2.5),
                             bal_hi=np.percentile(bb, 97.5),
                             constant_folds=int(f.constant_pred.sum()),
                             chosen_epoch_median=float(f.chosen_epoch.median())))
        diff = boots["revised"] - boots["scratch_revised"]
        est = rows[-2]["pooled_bal"] - rows[-1]["pooled_bal"]
        pairs.append(dict(seed=seed, diff=est, lo=np.percentile(diff, 2.5),
                          hi=np.percentile(diff, 97.5)))
    r = pd.DataFrame(rows)
    p = pd.DataFrame(pairs)
    r.round(4).to_csv(os.path.join(OUT, f"cbramod_seeds{SFX}.csv"), index=False)
    p.round(4).to_csv(os.path.join(OUT, f"cbramod_seeds_pretraining{SFX}.csv"),
                      index=False)
    summ = r.groupby("config").pooled_bal.agg(
        mean="mean", sd=lambda v: v.std(ddof=0), lo="min", hi="max")
    summ.round(4).to_csv(os.path.join(OUT, f"cbramod_seeds_summary{SFX}.csv"))
    print(r.round(4).to_string(index=False))
    print(p.round(4).to_string(index=False))
    print(summ.round(4))


if __name__ == "__main__":
    main()
