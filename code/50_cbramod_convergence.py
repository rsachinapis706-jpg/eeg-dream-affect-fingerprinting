"""
Did CBraMod converge within its training budget, or was it cut short?

49_cbramod_expanded.py trained for at most 8 epochs with early stopping at
patience 3, and reached 0.7844 balanced accuracy against a spectral baseline of
0.7799. Before that number is written into the manuscript it has to be
established that the model was given a fair chance to fit. Reporting an
under-trained foundation model would misrepresent it, and the manuscript uses
CBraMod specifically to close off the objection that a stronger model would have
found the dream-affect signal. That argument only holds if the model was
actually trained to convergence.

This is a diagnostic, not a hyperparameter search. One fold is trained for a
long budget and the validation curve is recorded at every epoch. The question it
answers is binary:

  the curve plateaus well before epoch 8   -> the budget was sufficient, the
                                              reported number stands, and more
                                              epochs would not change it

  the curve is still rising at epoch 8     -> the model was cut short and the
                                              full comparison must be re-run
                                              with a longer budget

Whichever way it comes out is reported. Choosing the budget after seeing which
value gives the preferred answer would be the same error this paper documents in
others, so the decision rule is fixed here in advance: re-run the full
comparison if and only if validation balanced accuracy at the budget ceiling is
still more than 0.005 above its value three epochs earlier.
"""
import os
import sys
import importlib.util
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "foundation", "CBraMod"))

SEED = 20260727
MAX_EPOCHS = 24
BS = 64
LR = 1e-4
STAGES = ["W", "N1", "N2"]
PLATEAU_TOL = 0.005            # decision rule, fixed before running

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def load_cb():
    spec = importlib.util.spec_from_file_location(
        "cbf", os.path.join(HERE, "14_cbramod_finetune.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    cb = load_cb()
    X = np.load(os.path.join(OUT, "sleepedf_exp_raw.npy"), mmap_mode="r")
    m = pd.read_parquet(os.path.join(OUT, "sleepedf_exp_raw_meta.parquet"))

    cls = {c: i for i, c in enumerate(STAGES)}
    y = np.array([cls[v] for v in m.stage])
    g = m.subject.to_numpy()
    idx_all = np.arange(len(y))

    log("=" * 78)
    log("CBRAMOD CONVERGENCE DIAGNOSTIC")
    log("=" * 78)
    log(f"windows {len(y)} | participants {len(np.unique(g))} | "
        f"device {cb.DEV}")
    log(f"budget {MAX_EPOCHS} epochs, no early stopping, one fold")
    log(f"decision rule fixed in advance: re-run the full comparison if")
    log(f"validation balanced accuracy at epoch 8 is still more than")
    log(f"{PLATEAU_TOL} above its value at epoch 5.")
    log("")

    tr, te = next(iter(StratifiedGroupKFold(
        5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y, g)))
    gtr = g[tr]
    hold = np.unique(gtr)[0]
    va = tr[gtr == hold]
    tr2 = tr[gtr != hold]

    sel = lambda ix: np.isin(idx_all, ix)
    cw = len(y[tr2]) / (len(STAGES) *
                        np.bincount(y[tr2], minlength=len(STAGES)).clip(min=1))

    cb.seed_all(SEED)
    model = cb.CBraModClassifier(n_classes=len(STAGES), pretrained=True,
                                 freeze=False).to(cb.DEV)
    dl_tr = cb.make_loader(X, idx_all, sel(tr2), y, BS, True)
    dl_va = cb.make_loader(X, idx_all, sel(va), y, BS * 2, False)
    dl_te = cb.make_loader(X, idx_all, sel(te), y, BS * 2, False)

    opt = torch.optim.AdamW(
        [{"params": model.backbone.parameters(), "lr": LR * 0.1},
         {"params": model.head.parameters(), "lr": LR}], weight_decay=0.01)
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32, device=cb.DEV))
    scaler = torch.amp.GradScaler("cuda", enabled=cb.DEV.type == "cuda")

    log(f"{'epoch':>6}{'val balanced':>15}{'test balanced':>16}")
    log("-" * 78)
    rows = []
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(cb.DEV, non_blocking=True), yb.to(cb.DEV)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=cb.DEV.type == "cuda"):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

        yt, yp = cb.predict(model, dl_va)
        vb = balanced_accuracy_score(yt, yp)
        yt2, yp2 = cb.predict(model, dl_te)
        tb = balanced_accuracy_score(yt2, yp2)
        log(f"{ep:>6}{vb:>15.4f}{tb:>16.4f}")
        rows.append(dict(epoch=ep, val_balanced=vb, test_balanced=tb))
        pd.DataFrame(rows).round(4).to_csv(
            os.path.join(OUT, "cbramod_convergence.csv"), index=False)

    r = pd.DataFrame(rows)
    log("")
    log("=" * 78)
    log("VERDICT")
    log("=" * 78)
    v5 = float(r[r.epoch == 5].val_balanced.iloc[0])
    v8 = float(r[r.epoch == 8].val_balanced.iloc[0])
    best_ep = int(r.loc[r.val_balanced.idxmax(), "epoch"])
    log(f"validation balanced accuracy: epoch 5 {v5:.4f}, epoch 8 {v8:.4f}")
    log(f"best validation epoch overall: {best_ep} "
        f"({r.val_balanced.max():.4f})")
    log(f"test balanced accuracy at the best validation epoch: "
        f"{float(r.loc[r.val_balanced.idxmax(), 'test_balanced']):.4f}")
    log(f"final epoch {MAX_EPOCHS}: val {float(r.val_balanced.iloc[-1]):.4f}, "
        f"test {float(r.test_balanced.iloc[-1]):.4f}")
    log("")
    if v8 - v5 > PLATEAU_TOL:
        log("STILL RISING at the 8-epoch budget. The reported comparison used")
        log("an under-trained model and must be re-run with a longer budget.")
    else:
        log("PLATEAUED within the 8-epoch budget. The reported comparison")
        log("stands; a longer budget would not change it, and extending the")
        log("budget now would be tuning against an observed result.")

    with open(os.path.join(OUT, "test40_cbramod_convergence.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"\nWrote {OUT}\\cbramod_convergence.csv")


if __name__ == "__main__":
    main()
