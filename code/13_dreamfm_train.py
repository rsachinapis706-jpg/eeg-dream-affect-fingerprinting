"""
DreamFM training harness: SSL pretraining, supervised fine-tuning, subject-wise
evaluation, permutation nulls.

Stages (select with --stage)
----------------------------
  sanity   Train DreamFM supervised on Sleep-EDF sleep staging, subject-wise.
           GATE: must beat the 0.88 balanced accuracy the spectral features
           reached. If it cannot learn a task with known signal, running it on
           dream affect proves nothing.

  ssl      Masked-EEG-modelling pretraining on Sleep-EDF (labels unused).

  deed     Fine-tune on DEED dream affect, leave-one-subject-out, with a
           permutation null. Optionally initialised from the SSL checkpoint.

Every split is grouped by subject. For DEED the 3-digit subject id is the group,
so the same person never appears in train and test. Within a training fold a
small validation slice is held out BY SUBJECT for early stopping, never by
window.
"""
import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

from dreamfm import DreamFM

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
CKPT = os.path.join(OUT, "ckpt")
os.makedirs(CKPT, exist_ok=True)

SEED = 20260727
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLL = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def seed_all(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def save_log(name):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ------------------------------------------------------------------ data
def load_sleepedf():
    X = np.load(os.path.join(OUT, "sleepedf_raw.npy"), mmap_mode="r")
    m = pd.read_parquet(os.path.join(OUT, "sleepedf_raw_meta.parquet"))
    return X, m


def load_deed():
    X = np.load(os.path.join(OUT, "deed_raw.npy"), mmap_mode="r")
    m = pd.read_parquet(os.path.join(OUT, "deed_raw_meta.parquet"))
    return X, m


def loaders(X, y, idx_all, sel_tr, sel_te, bs=64):
    """X is the FULL raw array; y is already restricted to the selected rows.

    idx_all maps selected-row position -> row in X, so X must be indexed with
    idx_all[sel] while y is indexed with sel directly. Conflating the two
    coordinate systems is an easy and silent mistake, hence this one entry point.
    """
    xt = torch.from_numpy(np.ascontiguousarray(X[idx_all[sel_tr]])).float()
    xe = torch.from_numpy(np.ascontiguousarray(X[idx_all[sel_te]])).float()
    yt = torch.from_numpy(y[sel_tr]).long()
    ye = torch.from_numpy(y[sel_te]).long()
    assert len(xt) == len(yt) and len(xe) == len(ye), "index mismatch"
    return (DataLoader(TensorDataset(xt, yt), batch_size=bs, shuffle=True,
                       drop_last=False),
            DataLoader(TensorDataset(xe, ye), batch_size=bs * 2, shuffle=False))


# ------------------------------------------------------------------ train
def train_supervised(model, tr, va, n_classes, epochs=15, lr=3e-4,
                     class_weights=None, patience=4, quiet=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * len(tr)))
    crit = nn.CrossEntropyLoss(
        weight=None if class_weights is None
        else torch.tensor(class_weights, dtype=torch.float32, device=DEV))

    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(yb)

        yt, yp = predict(model, va)
        b = balanced_accuracy_score(yt, yp)
        if not quiet:
            log(f"    epoch {ep+1:02d}  loss {tot/len(va.dataset):.4f}  val_bacc {b:.4f}")
        if b > best:
            best, bad = b, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


@torch.no_grad()
def predict(model, dl):
    model.eval()
    yt, yp = [], []
    for xb, yb in dl:
        yp.append(model(xb.to(DEV)).argmax(1).cpu().numpy())
        yt.append(yb.numpy())
    return np.concatenate(yt), np.concatenate(yp)


# ------------------------------------------------------------------ stages
def stage_sanity(args):
    X, m = load_sleepedf()
    sel = m.stage.isin(["W", "N1", "N2"]).to_numpy()
    m2 = m[sel].reset_index(drop=True)
    idx_all = np.where(sel)[0]
    classes = sorted(m2.stage.unique())
    y = m2.stage.map({c: i for i, c in enumerate(classes)}).to_numpy()
    subj = m2.subject.to_numpy()

    log("=" * 70)
    log("STAGE 1 (GATE): DreamFM supervised on Sleep-EDF sleep staging")
    log("=" * 70)
    log(f"device {DEV} | windows {len(y)} | subjects {len(np.unique(subj))}")
    log(f"classes {classes} | spectral-feature reference bacc = 0.8831")
    log("")

    yts, yps = [], []
    for s in np.unique(subj):
        te = subj == s
        tr = ~te
        # validation = one training subject, held out whole
        tr_subj = [u for u in np.unique(subj[tr])]
        va_s = tr_subj[0]
        va = (subj == va_s)
        tr = tr & ~va

        cw = len(y[tr]) / (len(classes) * np.bincount(y[tr], minlength=len(classes))
                           .clip(min=1))
        seed_all(SEED)
        model = DreamFM(n_classes=len(classes)).to(DEV)
        dl_tr, dl_va = loaders(X, y, idx_all, tr, va, bs=args.bs)
        train_supervised(model, dl_tr, dl_va, len(classes), epochs=args.epochs,
                         class_weights=cw, quiet=True)
        _, dl_te = loaders(X, y, idx_all, va, te, bs=args.bs)
        yt, yp = predict(model, dl_te)
        log(f"  held-out {s}: bacc {balanced_accuracy_score(yt, yp):.4f} "
            f"(n={len(yt)})")
        yts.append(yt); yps.append(yp)

    yt, yp = np.concatenate(yts), np.concatenate(yps)
    bacc = balanced_accuracy_score(yt, yp)
    log("")
    log(f"POOLED accuracy {accuracy_score(yt, yp):.4f} | balanced {bacc:.4f}")
    log(f"spectral features reference : 0.8831")
    log("")
    log(">> GATE PASSED" if bacc >= 0.80 else
        ">> GATE NOT PASSED -- DreamFM underperforms handcrafted features.")
    log("   A null on dream affect is only meaningful once this passes.")
    save_log("test9_dreamfm_sanity.txt")


def stage_ssl(args):
    X, m = load_sleepedf()
    log("=" * 70)
    log("STAGE 2: self-supervised masked-EEG pretraining on Sleep-EDF")
    log("=" * 70)
    log(f"device {DEV} | windows {len(m)} | labels NOT used")
    log("")

    seed_all(SEED)
    model = DreamFM(n_classes=3).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    n = len(m)
    order = np.random.permutation(n)

    for ep in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for i in range(0, n, args.bs):
            b = order[i:i + args.bs]
            xb = torch.from_numpy(np.ascontiguousarray(X[np.sort(b)])).float().to(DEV)
            opt.zero_grad()
            loss = model.ssl_loss(xb, mask_ratio=args.mask_ratio)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss); nb += 1
        log(f"  epoch {ep+1:02d}  masked-recon loss {tot/max(nb,1):.5f}")

    torch.save(model.state_dict(), os.path.join(CKPT, "dreamfm_ssl.pt"))
    log("")
    log(f"saved {CKPT}\\dreamfm_ssl.pt")
    save_log("test10_dreamfm_ssl.txt")


def stage_deed(args):
    X, m = load_deed()
    sel = m.E.isin(COLL).to_numpy()
    m2 = m[sel].reset_index(drop=True)
    idx_all = np.where(sel)[0]
    y_lab = m2.E.map(COLL)
    classes = sorted(y_lab.unique())
    y = y_lab.map({c: i for i, c in enumerate(classes)}).to_numpy()
    subj = m2.subject.to_numpy()
    seg = m2.segment_id.to_numpy()

    log("=" * 70)
    log("STAGE 3: DreamFM on DEED dream affect (leave-one-subject-out)")
    log("=" * 70)
    log(f"device {DEV} | windows {len(y)} | subjects {len(np.unique(subj))}")
    log(f"classes {classes} | init: "
        f"{'SSL checkpoint' if args.init_ssl else 'random'}")
    log(f"spectral-feature reference bacc = 0.3430 (chance 0.3333)")
    log("")

    def run_once(y_use, quiet=True):
        yts, yps = [], []
        for s in np.unique(subj):
            te = subj == s
            tr = ~te
            tr_subj = [u for u in np.unique(subj[tr])]
            va = subj == tr_subj[0]
            tr = tr & ~va
            if len(np.unique(y_use[tr])) < 2:
                continue
            cw = len(y_use[tr]) / (len(classes) *
                                   np.bincount(y_use[tr], minlength=len(classes)).clip(min=1))
            seed_all(SEED)
            model = DreamFM(n_classes=len(classes)).to(DEV)
            if args.init_ssl:
                sd = torch.load(os.path.join(CKPT, "dreamfm_ssl.pt"),
                                map_location=DEV)
                model.load_state_dict(sd, strict=False)
            dl_tr, dl_va = loaders(X, y_use, idx_all, tr, va, bs=args.bs)
            train_supervised(model, dl_tr, dl_va, len(classes),
                             epochs=args.epochs, class_weights=cw, quiet=True)
            _, dl_te = loaders(X, y_use, idx_all, va, te, bs=args.bs)
            yt, yp = predict(model, dl_te)
            if not quiet:
                log(f"  held-out {s}: bacc {balanced_accuracy_score(yt, yp):.4f}")
            yts.append(yt); yps.append(yp)
        return np.concatenate(yts), np.concatenate(yps)

    yt, yp = run_once(y, quiet=False)
    bacc = balanced_accuracy_score(yt, yp)
    log("")
    log(f"POOLED accuracy {accuracy_score(yt, yp):.4f} | balanced {bacc:.4f}")
    log("confusion (rows=true):")
    log(str(confusion_matrix(yt, yp)))
    log("")

    if args.perm > 0:
        log(f"permutation null ({args.perm} runs, labels shuffled across "
            f"recordings within subject)...")
        nulls = []
        for i in range(args.perm):
            rng = np.random.default_rng(SEED + 101 * i)
            sl = pd.DataFrame({"s": seg, "u": subj, "y": y}).drop_duplicates("s")
            mp = {}
            for u, g in sl.groupby("u"):
                mp.update(dict(zip(g.s.to_numpy(), rng.permutation(g.y.to_numpy()))))
            yperm = pd.Series(seg).map(mp).to_numpy()
            ytp, ypp = run_once(yperm)
            nulls.append(balanced_accuracy_score(ytp, ypp))
            log(f"  perm {i+1}/{args.perm}: {nulls[-1]:.4f}")
        nulls = np.array(nulls)
        p = (np.sum(nulls >= bacc) + 1) / (len(nulls) + 1)
        log("")
        log(f"observed {bacc:.4f} | null {nulls.mean():.4f} +/- {nulls.std():.4f} "
            f"| p = {p:.4f}")

    save_log("test11_dreamfm_deed.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["sanity", "ssl", "deed"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--init-ssl", action="store_true")
    ap.add_argument("--perm", type=int, default=0)
    args = ap.parse_args()

    {"sanity": stage_sanity, "ssl": stage_ssl, "deed": stage_deed}[args.stage](args)
