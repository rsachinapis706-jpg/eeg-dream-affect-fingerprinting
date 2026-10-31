"""
Pooled results of the CBraMod diagnostics (76_cbramod_diagnostics.py) for each
configuration, from the per-fold files it saves (Reviewer 1, comment 9).

76's own pooling step read participant identifiers as integers ("002" -> 2)
and could not find the prediction files; that is fixed in 76, and this script
does the pooling independently with identifiers read as strings.

Per configuration: pooled balanced accuracy and accuracy over all held-out
windows, the confusion matrix, folds with a constant prediction, folds at zero
balanced accuracy (and how many of those have a single-class test fold), the
distribution of the epoch chosen by early stopping, and the best validation
balanced accuracy. Per epoch (from the curves): mean training loss and mean
validation balanced accuracy across folds.

Uncertainty: every fold is one held-out participant, so a participant
bootstrap resamples folds (2,000 resamples) and recomputes pooled balanced
accuracy. Matched comparisons (pretrained minus scratch, under the published
and under the revised settings) are paired on the same resampled participants.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SEED = 20260727
N_BOOT = 2000
CFGS = ("published", "scratch", "revised", "scratch_revised")
PAIRS = (("published", "scratch"), ("revised", "scratch_revised"))


def load_cfg(cfg):
    f = pd.read_csv(os.path.join(OUT, f"cbramod_diag_{cfg}_{SEED}_folds.csv"),
                    dtype={"held_out": str})
    assert not f.held_out.duplicated().any()
    per = {}
    for s in f.held_out:
        a = np.load(os.path.join(OUT, f"cbramod_diag_{cfg}_{SEED}_{s}"
                                      f"_pred.npy"))
        per[s] = (a[0], a[1])
    return f, per


def pooled(per, ids):
    yt = np.concatenate([per[i][0] for i in ids])
    yp = np.concatenate([per[i][1] for i in ids])
    return balanced_accuracy_score(yt, yp)


def main():
    rows, cms, curves = [], [], []
    boots = {}
    for cfg in CFGS:
        fp = os.path.join(OUT, f"cbramod_diag_{cfg}_{SEED}_folds.csv")
        if not os.path.exists(fp):
            continue
        f = pd.read_csv(fp, dtype={"held_out": str})
        assert not f.held_out.duplicated().any()
        yt, yp = [], []
        for s in f.held_out:
            a = np.load(os.path.join(OUT, f"cbramod_diag_{cfg}_{SEED}_{s}"
                                          f"_pred.npy"))
            yt.append(a[0]); yp.append(a[1])
        yt, yp = np.concatenate(yt), np.concatenate(yp)
        cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
        ce = f.chosen_epoch.value_counts().sort_index()
        rows.append(dict(
            config=cfg, folds=len(f), windows=len(yt),
            pooled_bal=balanced_accuracy_score(yt, yp),
            pooled_acc=accuracy_score(yt, yp),
            constant_folds=int(f.constant_pred.sum()),
            zero_bal_folds=int((f.bal == 0).sum()),
            zero_bal_single_class_test=int(((f.bal == 0)
                                            & (f.test_classes == 1)).sum()),
            chosen_epoch_1=int((f.chosen_epoch == 1).sum()),
            chosen_epoch_median=float(f.chosen_epoch.median()),
            best_val_bal_median=float(f.best_val_bal.median()),
            best_val_bal_max=float(f.best_val_bal.max()),
            val_windows_median=float(f.n_val.median()),
            pred_share_neg=float((yp == 0).mean()),
            pred_share_neu=float((yp == 1).mean()),
            pred_share_pos=float((yp == 2).mean())))
        for i, lab in enumerate(("neg", "neu", "pos")):
            cms.append(dict(config=cfg, true=lab, pred_neg=cm[i, 0],
                            pred_neu=cm[i, 1], pred_pos=cm[i, 2]))
        _, per = load_cfg(cfg)
        ids = sorted(per)
        rng = np.random.default_rng(20261001)
        draws = [rng.choice(ids, len(ids)) for _ in range(N_BOOT)]
        boots[cfg] = (ids, draws, per)
        bb = np.array([pooled(per, d) for d in draws])
        rows[-1]["bal_lo"] = float(np.percentile(bb, 2.5))
        rows[-1]["bal_hi"] = float(np.percentile(bb, 97.5))
        cp = os.path.join(OUT, f"cbramod_diag_{cfg}_{SEED}_curves.csv")
        cv = pd.read_csv(cp, dtype={"held_out": str})
        num = [c for c in cv.columns if c not in ("config", "seed",
                                                    "held_out", "epoch")
               and pd.api.types.is_numeric_dtype(cv[c])]
        g = cv.groupby("epoch")[num].mean().reset_index()
        g.insert(0, "config", cfg)
        g["folds_reaching_epoch"] = cv.groupby("epoch").held_out.nunique(
        ).to_numpy()
        curves.append(g)
        print(f"{cfg}: {len(f)} folds, pooled bal {rows[-1]['pooled_bal']:.4f}"
              f" acc {rows[-1]['pooled_acc']:.4f} | constant folds "
              f"{rows[-1]['constant_folds']} | chosen epoch 1 in "
              f"{rows[-1]['chosen_epoch_1']} | best val bal median "
              f"{rows[-1]['best_val_bal_median']:.4f}")
        print(cm)
    pairs = []
    for a, b in PAIRS:
        if a not in boots or b not in boots:
            continue
        ids_a, _, pa = boots[a]
        ids_b, _, pb = boots[b]
        if ids_a != ids_b:
            print(f"{a} vs {b}: participant sets differ, pairing skipped")
            continue
        rng = np.random.default_rng(20261002)
        diffs = []
        for _ in range(N_BOOT):
            d = rng.choice(ids_a, len(ids_a))
            diffs.append(pooled(pa, d) - pooled(pb, d))
        est = pooled(pa, ids_a) - pooled(pb, ids_a)
        pairs.append(dict(pretrained=a, scratch=b, diff=est,
                          lo=np.percentile(diffs, 2.5),
                          hi=np.percentile(diffs, 97.5)))
        print(f"{a} - {b}: {est:+.4f} [{pairs[-1]['lo']:+.4f}, "
              f"{pairs[-1]['hi']:+.4f}]")
    pd.DataFrame(pairs).round(4).to_csv(
        os.path.join(OUT, "cbramod_diag_pretraining_effect.csv"), index=False)
    pd.DataFrame(rows).round(4).to_csv(
        os.path.join(OUT, "cbramod_diag_summary.csv"), index=False)
    pd.DataFrame(cms).to_csv(os.path.join(OUT, "cbramod_diag_confusion.csv"),
                             index=False)
    pd.concat(curves).round(4).to_csv(
        os.path.join(OUT, "cbramod_diag_curves_summary.csv"), index=False)


if __name__ == "__main__":
    main()
