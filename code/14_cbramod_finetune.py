"""
Phase 1: fine-tune CBraMod (ICLR 2025 EEG foundation model) under our protocol.

Why a published foundation model rather than our own architecture
-----------------------------------------------------------------
The result we are heading toward is negative. The standard objection to a
negative result is "your model was not good enough". Using OUR model leaves that
objection open forever. Using CBraMod does not: it is peer-reviewed, pretrained on
9,000 hours of TUEG, and reports state-of-the-art results on multiple BCI
benchmarks including 6-channel sleep staging on ISRUC -- the closest published
analogue to DEED.

So the claim becomes: a state-of-the-art EEG foundation model, validated
elsewhere, cannot decode dream affect here.

Protocol is unchanged from the rest of the project
--------------------------------------------------
  * leave-one-subject-out on the 3-digit DEED subject id
  * a whole training subject held out for early stopping, never windows
  * permutation null with labels shuffled across recordings within subject
  * sleep stage as the positive control, run through the identical code path

Input format
------------
CBraMod expects (batch, channels, patches, points_per_patch). Our 20 s windows at
200 Hz give 20 patches of 200 samples, so (B, C, 20, 200).

The head mean-pools over channels, which keeps it channel-agnostic: the same head
shape works for 6-channel DEED and 2-channel Sleep-EDF, so the positive control
and the target task are genuinely the same model.

Modes
-----
  --probe   freeze the backbone, train the head only (linear probe). Fast, and it
            asks whether the pretrained representation ALREADY separates the
            classes.
  default   full fine-tuning at a lower backbone learning rate.
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
CB = os.path.join(ROOT, "foundation", "CBraMod")
sys.path.insert(0, CB)
from models.cbramod import CBraMod  # noqa: E402

SEED = 20260727
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLL = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}
WEIGHTS = os.path.join(CB, "pretrained_weights", "pretrained_weights.pth")

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def seed_all(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class CBraModClassifier(nn.Module):
    """Pretrained CBraMod backbone + channel-agnostic classification head."""

    def __init__(self, n_classes=3, pretrained=True, freeze=False, dropout=0.3,
                 n_patches=20):
        super().__init__()
        self.backbone = CBraMod()
        if pretrained:
            sd = torch.load(WEIGHTS, map_location="cpu")
            self.backbone.load_state_dict(sd, strict=True)
        self.backbone.proj_out = nn.Identity()
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.Flatten(),                              # (B, P*200)
            nn.LayerNorm(n_patches * 200),
            nn.Dropout(dropout),
            nn.Linear(n_patches * 200, 256), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):                              # (B, C, P, 200)
        f = self.backbone(x)                           # (B, C, P, 200)
        f = f.mean(dim=1)                              # channel-agnostic pooling
        return self.head(f)


def to_patches(x):
    """(N, C, 4000) -> (N, C, 20, 200)"""
    n, c, t = x.shape
    return x.reshape(n, c, t // 200, 200)


def make_loader(X, idx_all, sel, y, bs, shuffle):
    xb = torch.from_numpy(to_patches(np.ascontiguousarray(X[idx_all[sel]]))).float()
    yb = torch.from_numpy(y[sel]).long()
    assert len(xb) == len(yb)
    return DataLoader(TensorDataset(xb, yb), batch_size=bs, shuffle=shuffle)


@torch.no_grad()
def predict(model, dl):
    model.eval()
    yt, yp = [], []
    for xb, yb in dl:
        yp.append(model(xb.to(DEV)).argmax(1).cpu().numpy())
        yt.append(yb.numpy())
    return np.concatenate(yt), np.concatenate(yp)


def train_fold(model, dl_tr, dl_va, cw, epochs, lr, probe, patience=3):
    if probe:
        params = [{"params": model.head.parameters(), "lr": lr}]
    else:
        params = [{"params": model.backbone.parameters(), "lr": lr * 0.1},
                  {"params": model.head.parameters(), "lr": lr}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32, device=DEV))
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")

    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEV, non_blocking=True), yb.to(DEV)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEV.type == "cuda"):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        yt, yp = predict(model, dl_va)
        b = balanced_accuracy_score(yt, yp)
        if b > best:
            best, bad = b, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return best


def run_loso(X, idx_all, y, subj, classes, args, quiet=False):
    yts, yps = [], []
    for s in np.unique(subj):
        te = subj == s
        tr = ~te
        others = [u for u in np.unique(subj[tr])]
        if not others:
            continue
        va = subj == others[0]
        tr = tr & ~va
        if len(np.unique(y[tr])) < 2 or te.sum() == 0 or va.sum() == 0:
            continue
        cw = len(y[tr]) / (len(classes) *
                           np.bincount(y[tr], minlength=len(classes)).clip(min=1))
        seed_all(SEED)
        model = CBraModClassifier(n_classes=len(classes), pretrained=True,
                                  freeze=args.probe).to(DEV)
        dl_tr = make_loader(X, idx_all, tr, y, args.bs, True)
        dl_va = make_loader(X, idx_all, va, y, args.bs * 2, False)
        dl_te = make_loader(X, idx_all, te, y, args.bs * 2, False)
        train_fold(model, dl_tr, dl_va, cw, args.epochs, args.lr, args.probe)
        yt, yp = predict(model, dl_te)
        if not quiet:
            log(f"  held-out {s}: bacc {balanced_accuracy_score(yt, yp):.4f} "
                f"(n={len(yt)})")
        yts.append(yt); yps.append(yp)
        del model
        torch.cuda.empty_cache()
    return np.concatenate(yts), np.concatenate(yps)


def stage_sanity(args):
    X = np.load(os.path.join(OUT, "sleepedf_raw.npy"), mmap_mode="r")
    m = pd.read_parquet(os.path.join(OUT, "sleepedf_raw_meta.parquet"))
    sel = m.stage.isin(["W", "N1", "N2"]).to_numpy()
    m2 = m[sel].reset_index(drop=True)
    idx_all = np.where(sel)[0]
    classes = sorted(m2.stage.unique())
    y = m2.stage.map({c: i for i, c in enumerate(classes)}).to_numpy()
    subj = m2.subject.to_numpy()

    log("=" * 72)
    log("CBraMod POSITIVE CONTROL: Sleep-EDF sleep staging (W/N1/N2)")
    log("=" * 72)
    log(f"device {DEV} | windows {len(y)} | subjects {len(np.unique(subj))}")
    log(f"mode: {'linear probe (backbone frozen)' if args.probe else 'full fine-tune'}")
    log(f"reference: spectral features = 0.8831 balanced accuracy")
    log("")

    yt, yp = run_loso(X, idx_all, y, subj, classes, args)
    bacc = balanced_accuracy_score(yt, yp)
    log("")
    log(f"POOLED accuracy {accuracy_score(yt, yp):.4f} | balanced {bacc:.4f}")
    log(f"spectral-feature reference     : 0.8831")
    log("")
    log(">> CONTROL PASSED" if bacc >= 0.75 else ">> CONTROL WEAK")
    save("test12_cbramod_sanity.txt")


def stage_deed(args):
    X = np.load(os.path.join(OUT, "deed_raw.npy"), mmap_mode="r")
    m = pd.read_parquet(os.path.join(OUT, "deed_raw_meta.parquet"))
    sel = m.E.isin(COLL).to_numpy()
    m2 = m[sel].reset_index(drop=True)
    idx_all = np.where(sel)[0]
    lab = m2.E.map(COLL)
    classes = sorted(lab.unique())
    y = lab.map({c: i for i, c in enumerate(classes)}).to_numpy()
    subj = m2.subject.to_numpy()
    seg = m2.segment_id.to_numpy()

    log("=" * 72)
    log("CBraMod on DEED dream affect (leave-one-subject-out)")
    log("=" * 72)
    log(f"device {DEV} | windows {len(y)} | subjects {len(np.unique(subj))}")
    log(f"mode: {'linear probe' if args.probe else 'full fine-tune'} | "
        f"classes {classes}")
    log(f"chance 0.3333 | spectral-feature reference 0.3430")
    log("")

    yt, yp = run_loso(X, idx_all, y, subj, classes, args)
    bacc = balanced_accuracy_score(yt, yp)
    log("")
    log(f"POOLED accuracy {accuracy_score(yt, yp):.4f} | balanced {bacc:.4f}")
    log("confusion (rows=true):")
    log(str(confusion_matrix(yt, yp)))
    log("")

    if args.perm > 0:
        log(f"permutation null: {args.perm} runs, labels shuffled across "
            f"recordings within subject")
        nulls = []
        for i in range(args.perm):
            rng = np.random.default_rng(SEED + 977 * i)
            sl = pd.DataFrame({"s": seg, "u": subj, "y": y}).drop_duplicates("s")
            mp = {}
            for u, g in sl.groupby("u"):
                mp.update(dict(zip(g.s.to_numpy(), rng.permutation(g.y.to_numpy()))))
            yperm = pd.Series(seg).map(mp).to_numpy()
            ytp, ypp = run_loso(X, idx_all, yperm, subj, classes, args, quiet=True)
            nulls.append(balanced_accuracy_score(ytp, ypp))
            log(f"  perm {i+1}/{args.perm}: {nulls[-1]:.4f}")
        nulls = np.array(nulls)
        p = (np.sum(nulls >= bacc) + 1) / (len(nulls) + 1)
        log("")
        log(f"observed {bacc:.4f} | null {nulls.mean():.4f} +/- {nulls.std():.4f}"
            f" | p = {p:.4f}")

    save("test13_cbramod_deed.txt")


def save(name):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"Wrote {OUT}\\{name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["sanity", "deed"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--perm", type=int, default=0)
    args = ap.parse_args()
    {"sanity": stage_sanity, "deed": stage_deed}[args.stage](args)
