"""
CBraMod on DEED dream affect: training diagnostics (Reviewer 1, comment 9).

The published fine-tuning run (14_cbramod_finetune.py --stage deed) returned
0.3147 pooled balanced accuracy, with 24 of 34 leave-one-participant-out folds
predicting a single class and 5 more at zero balanced accuracy. Reviewer 1 asks
whether that reflects the data or the training configuration, and for training
and validation diagnostics, fold-level class distributions, confusion matrices,
complete optimisation settings, sensitivity to initialisation and training
choices, and the same architecture without pretraining.

Configurations (all leave-one-participant-out on the identical 6,144 windows):
  --config published   the settings of the published run: pretrained backbone,
                       full fine-tuning, AdamW, lr 1e-3 (backbone 1e-4), weight
                       decay 0.01, batch 32, 8 epochs, class-weighted loss, early
                       stopping on ONE held-out training participant, patience 3
  --config scratch     identical, but the backbone is randomly initialised:
                       the same architecture without pretraining
  --config revised     pretrained; early stopping on a fifth of the training
                       participants rather than one; lr 1e-4 (backbone 1e-5);
                       batch 64; up to 15 epochs, patience 4
  --config scratch_revised  the revised settings with a randomly initialised
                       backbone: the matched no-pretraining control for revised
  --seed               initialisation / data-order seed (default 20260727)

Recorded for every fold: held-out participant, class counts in the test fold,
predicted-class counts, balanced accuracy, whether the prediction was constant,
validation participants, the epoch chosen by early stopping, and the per-epoch
training loss and validation balanced accuracy. The input amplitude scale is
logged once, against CBraMod's pretraining convention (EEG in units of 100 uV).

Outputs out/cbramod_diag_<config>_<seed>_folds.csv and _curves.csv; resumable
at the fold level. The seed also draws, fold by fold, the training participants
held out for validation. A run resumed after an interruption draws them afresh
from the point of resumption, so they can differ from an uninterrupted run with
the same seed; the participants actually used are recorded for every fold
(val_ids), and GPU training is not bit-reproducible in any case.
"""
import argparse
import importlib.util
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "cbft", os.path.join(HERE, "14_cbramod_finetune.py"))
cbft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cbft)             # model, loaders, predict, seed_all
DEV = cbft.DEV
OUT = cbft.OUT

CONFIGS = {
    "published": dict(pretrained=True, lr=1e-3, epochs=8, patience=3,
                      val="one"),
    "scratch": dict(pretrained=False, lr=1e-3, epochs=8, patience=3,
                    val="one"),
    "revised": dict(pretrained=True, lr=1e-4, epochs=15, patience=4,
                    val="fifth", bs=64),
    # the same architecture without pretraining, under the revised settings,
    # so that the pretrained/random comparison is matched in every setting
    "scratch_revised": dict(pretrained=False, lr=1e-4, epochs=15, patience=4,
                            val="fifth", bs=64),
}
BS = 32


class GpuLoader:
    """Batches drawn by index from a tensor already on the GPU."""

    def __init__(self, xg, yg, idx, bs, shuffle, gen):
        self.xg, self.yg = xg, yg
        self.idx = torch.as_tensor(np.asarray(idx), device=DEV)
        self.bs, self.shuffle, self.gen = bs, shuffle, gen

    def __iter__(self):
        order = self.idx
        if self.shuffle:
            order = self.idx[torch.randperm(len(self.idx), device=DEV,
                                            generator=self.gen)]
        for k in range(0, len(order), self.bs):
            b = order[k:k + self.bs]
            yield self.xg[b], self.yg[b]


@torch.no_grad()
def predict_gpu(model, dl):
    model.eval()
    yt, yp = [], []
    for xb, yb in dl:
        with torch.amp.autocast("cuda", enabled=DEV.type == "cuda"):
            yp.append(model(xb).argmax(1).cpu().numpy())
        yt.append(yb.cpu().numpy())
    return np.concatenate(yt), np.concatenate(yp)


def train_fold(model, dl_tr, dl_va, cw, cfg):
    params = [{"params": model.backbone.parameters(), "lr": cfg["lr"] * 0.1},
              {"params": model.head.parameters(), "lr": cfg["lr"]}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32, device=DEV))
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")
    best, best_state, bad, best_ep = -1.0, None, 0, -1
    curve = []
    for ep in range(cfg["epochs"]):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in dl_tr:
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEV.type == "cuda"):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * len(yb); n += len(yb)
        yt, yp = predict_gpu(model, dl_va)
        vb = balanced_accuracy_score(yt, yp)
        curve.append(dict(epoch=ep + 1, train_loss=tot / n, val_bal=vb,
                          val_pred_classes=int(len(np.unique(yp)))))
        if vb > best:
            best, bad, best_ep = vb, 0, ep + 1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if best_state:
        model.load_state_dict(best_state)
    return best, best_ep, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, default=20260727)
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    # CBRAMOD_SCALE=uv100: inputs in units of 100 uV, CBraMod's pretraining
    # convention (144_cbramod_scaling.py), instead of the default per-window
    # z-scores of deed_raw.npy; outputs then carry the suffix _uv100.
    scale = os.environ.get("CBRAMOD_SCALE", "")
    sfx = f"_{scale}" if scale else ""
    tag = f"{args.config}{sfx}_{args.seed}"
    fpath = os.path.join(OUT, f"cbramod_diag_{tag}_folds.csv")
    cpath = os.path.join(OUT, f"cbramod_diag_{tag}_curves.csv")

    X = np.load(os.path.join(OUT, f"deed_raw{sfx}.npy"), mmap_mode="r")
    m = pd.read_parquet(os.path.join(OUT, "deed_raw_meta.parquet"))
    sel = m.E.isin(cbft.COLL).to_numpy()
    m2 = m[sel].reset_index(drop=True)
    idx_all = np.where(sel)[0]
    lab = m2.E.map(cbft.COLL)
    classes = sorted(lab.unique())
    y = lab.map({c: i for i, c in enumerate(classes)}).to_numpy()
    subj = m2.subject.to_numpy()

    xall = np.ascontiguousarray(X[idx_all]).astype(np.float32)
    samp = np.abs(xall[::50])
    xg = torch.from_numpy(cbft.to_patches(xall)).to(DEV)
    yg = torch.from_numpy(y).long().to(DEV)
    del xall
    gen = torch.Generator(device=DEV)
    print(f"[{tag}] windows {len(y)} | participants {len(np.unique(subj))} | "
          f"classes {classes} | input |x| median {np.median(samp):.3f}, "
          f"p99 {np.percentile(samp, 99):.3f} (pretraining convention: "
          f"EEG / 100 uV)", flush=True)

    done = set()
    if os.path.exists(fpath):
        done = set(pd.read_csv(fpath, dtype={"held_out": str}).held_out)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for s in np.unique(subj):
        if str(s) in done:
            continue
        te = subj == s
        others = np.unique(subj[~te])
        if cfg["val"] == "one":
            val_ids = others[:1]               # as in the published run
        else:
            k = max(1, int(round(len(others) / 5)))
            val_ids = np.sort(rng.choice(others, k, replace=False))
        va = np.isin(subj, val_ids)
        tr = ~te & ~va
        cw = len(y[tr]) / (len(classes) *
                           np.bincount(y[tr], minlength=len(classes)).clip(min=1))
        cbft.seed_all(args.seed)
        model = cbft.CBraModClassifier(n_classes=len(classes),
                                       pretrained=cfg["pretrained"]).to(DEV)
        gen.manual_seed(args.seed)
        bs = cfg.get("bs", BS)
        dl_tr = GpuLoader(xg, yg, np.where(tr)[0], bs, True, gen)
        dl_va = GpuLoader(xg, yg, np.where(va)[0], bs * 2, False, gen)
        dl_te = GpuLoader(xg, yg, np.where(te)[0], bs * 2, False, gen)
        vbest, ep, curve = train_fold(model, dl_tr, dl_va, cw, cfg)
        yt, yp = predict_gpu(model, dl_te)
        tc = np.bincount(yt, minlength=3)
        pc = np.bincount(yp, minlength=3)
        row = dict(config=args.config, seed=args.seed, held_out=str(s),
                   n_test=len(yt), test_neg=tc[0], test_neu=tc[1],
                   test_pos=tc[2], pred_neg=pc[0], pred_neu=pc[1],
                   pred_pos=pc[2], test_classes=int((tc > 0).sum()),
                   constant_pred=bool((pc > 0).sum() == 1),
                   bal=balanced_accuracy_score(yt, yp),
                   acc=accuracy_score(yt, yp), val_ids=";".join(map(str,
                                                                    val_ids)),
                   n_train=int(tr.sum()), n_val=int(va.sum()),
                   chosen_epoch=ep, best_val_bal=vbest,
                   epochs_run=len(curve))
        pd.DataFrame([row]).to_csv(fpath, mode="a",
                                   header=not os.path.exists(fpath),
                                   index=False)
        cv = pd.DataFrame(curve).assign(config=args.config, seed=args.seed,
                                        held_out=str(s))
        cv.to_csv(cpath, mode="a", header=not os.path.exists(cpath),
                  index=False)
        np.save(os.path.join(OUT, f"cbramod_diag_{tag}_{s}_pred.npy"),
                np.stack([yt, yp]))
        print(f"[{tag}] {s}: bal {row['bal']:.4f} constant "
              f"{row['constant_pred']} epoch {ep}/{len(curve)} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        del model
        torch.cuda.empty_cache()

    f = pd.read_csv(fpath, dtype={"held_out": str})
    yts, yps = [], []
    for s in f.held_out.astype(str):
        a = np.load(os.path.join(OUT, f"cbramod_diag_{tag}_{s}_pred.npy"))
        yts.append(a[0]); yps.append(a[1])
    yt, yp = np.concatenate(yts), np.concatenate(yps)
    print(f"[{tag}] POOLED balanced {balanced_accuracy_score(yt, yp):.4f} | "
          f"accuracy {accuracy_score(yt, yp):.4f} | constant folds "
          f"{int(f.constant_pred.sum())}/{len(f)} | zero-bal folds "
          f"{int((f.bal == 0).sum())} (of which single-class test folds "
          f"{int(((f.bal == 0) & (f.test_classes == 1)).sum())})")
    print(f"[{tag}] confusion (rows = true neg/neu/pos):\n"
          f"{confusion_matrix(yt, yp, labels=[0, 1, 2])}")


if __name__ == "__main__":
    main()
