"""
Does a foundation-model representation expose within-subject dream-affect signal
that handcrafted features miss?

This is the decision point for the "subject-adaptive decoder" paper.

Script 11 found no within-subject signal using 108 spectral/nonlinear features
(28 subjects, mean bacc 0.3653 vs null 0.3365, Wilcoxon p = 0.362). Every
personalisation method -- few-shot, test-time adaptation, subject adaptation --
is upper-bounded by within-subject signal: adaptation reaches a ceiling faster,
it does not raise the ceiling. So if no within-subject structure exists, none of
those methods can work.

But that test used band powers and entropies. A representation learned from
9,000 hours of EEG might encode subject-specific structure those features cannot
express. That is a real possibility and it deserves a direct test.

    within-subject signal in CBraMod embeddings  -> subject-adaptive paper is live
    no within-subject signal                     -> adaptation has no ceiling to
                                                    reach, and we report that

Method
------
CBraMod is used FROZEN. The question is whether the pretrained representation
already contains the structure, not whether we can fit it into existence -- and a
frozen extractor cannot overfit the within-subject folds.

Embeddings: backbone output (B, C, P, 200) pooled to (B, 400) by concatenating
the mean and standard deviation over channels and patches. The std term keeps
within-window temporal variability, which the mean alone discards.

Protocol is identical to script 11 so the two are directly comparable:
grouped CV by recording inside each subject, permutation null by shuffling labels
across that subject's own recordings.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

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
# CBRAMOD_SCALE=uv100: embed the windows in units of 100 uV, CBraMod's
# pretraining convention (144_cbramod_scaling.py), instead of the default
# per-window z-scores of deed_raw.npy; outputs then carry the suffix _uv100.
SCALE = os.environ.get("CBRAMOD_SCALE", "")
assert SCALE in ("", "uv100"), SCALE
SFX = f"_{SCALE}" if SCALE else ""
CACHE = os.path.join(OUT, f"cbramod_embeddings{SFX}.npy")
N_PERM = 200
MIN_SEGS = 6
MIN_CLASSES = 2

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


@torch.no_grad()
def extract_embeddings(bs=64):
    if os.path.exists(CACHE):
        Z = np.load(CACHE)
        log(f"[cache] embeddings {Z.shape}")
        return Z

    X = np.load(os.path.join(OUT, f"deed_raw{SFX}.npy"), mmap_mode="r")
    s = np.abs(np.asarray(X[::50]))
    log(f"input |x| median {np.median(s):.3f}, p99 {np.percentile(s, 99):.3f} "
        f"({'units of 100 uV' if SCALE else 'per-window z-scores'})")
    model = CBraMod().to(DEV)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEV), strict=True)
    model.proj_out = nn.Identity()
    model.eval()

    out = []
    for i in range(0, len(X), bs):
        xb = np.ascontiguousarray(X[i:i + bs])
        n, c, t = xb.shape
        xb = torch.from_numpy(xb.reshape(n, c, t // 200, 200)).float().to(DEV)
        with torch.amp.autocast("cuda", enabled=DEV.type == "cuda"):
            f = model(xb)                       # (n, C, P, 200)
        f = f.float()
        mu = f.mean(dim=(1, 2))                 # (n, 200)
        sd = f.std(dim=(1, 2))                  # (n, 200)
        out.append(torch.cat([mu, sd], dim=1).cpu().numpy())
        if (i // bs) % 20 == 0:
            log(f"  embedded {min(i + bs, len(X))}/{len(X)}")
    Z = np.concatenate(out).astype(np.float32)
    np.save(CACHE, Z)
    log(f"embeddings {Z.shape} -> {CACHE}")
    return Z


def clf():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced"))


def within_bacc(X, y, seg, n_splits=3):
    if len(np.unique(y)) < 2:
        return None
    k = min(n_splits, len(np.unique(seg)))
    if k < 2:
        return None
    try:
        splits = list(StratifiedGroupKFold(k, shuffle=True,
                                           random_state=SEED).split(X, y, seg))
    except ValueError:
        return None
    yt, yp = [], []
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2 or len(te) == 0:
            continue
        yp.append(clf().fit(X[tr], y[tr]).predict(X[te]))
        yt.append(y[te])
    if not yt:
        return None
    return balanced_accuracy_score(np.concatenate(yt), np.concatenate(yp))


def perm_bacc(X, y, seg, rng):
    sl = pd.DataFrame({"s": seg, "y": y}).drop_duplicates("s")
    mp = dict(zip(sl.s.to_numpy(), rng.permutation(sl.y.to_numpy())))
    return within_bacc(X, pd.Series(seg).map(mp).to_numpy(), seg)


def main():
    log("=" * 74)
    log("WITHIN-SUBJECT DREAM AFFECT IN CBraMod EMBEDDINGS")
    log("=" * 74)
    log("CBraMod frozen (pretrained on 9,000 h TUEG). Decision point for the")
    log("subject-adaptive decoder: adaptation is bounded by within-subject signal.")
    log("")

    Z = extract_embeddings()
    m = pd.read_parquet(os.path.join(OUT, "deed_raw_meta.parquet"))
    assert len(Z) == len(m), f"embedding/meta mismatch {len(Z)} vs {len(m)}"

    d = m[m.E.isin(COLL)].copy()
    d["y"] = d.E.map(COLL)
    Zs = Z[d.index.to_numpy()]

    log(f"windows {len(d)} | subjects {d.subject.nunique()} | dim {Z.shape[1]}")
    log(f"eligibility: >= {MIN_CLASSES} classes, >= {MIN_SEGS} recordings")
    log(f"permutations per subject: {N_PERM}")
    log("")

    rows = []
    for subj, g in d.groupby("subject"):
        if g.segment_id.nunique() < MIN_SEGS or g.y.nunique() < MIN_CLASSES:
            continue
        pos = d.index.get_indexer(g.index)
        X = Zs[pos]
        y = g.y.to_numpy()
        seg = g.segment_id.to_numpy()
        obs = within_bacc(X, y, seg)
        if obs is None:
            continue
        nulls = Parallel(n_jobs=-1)(
            delayed(perm_bacc)(X, y, seg, np.random.default_rng(SEED + 31 * i))
            for i in range(N_PERM))
        nulls = np.array([v for v in nulls if v is not None])
        if len(nulls) < 20:
            continue
        p = (np.sum(nulls >= obs) + 1) / (len(nulls) + 1)
        rows.append(dict(subject=subj, n_seg=g.segment_id.nunique(),
                         n_epoch=len(g), bacc=obs, null=nulls.mean(), p=p))

    res = pd.DataFrame(rows).sort_values("p")
    log(f"{'subj':<7}{'segs':>6}{'eps':>7}{'bacc':>9}{'null':>9}{'p':>9}")
    log("-" * 74)
    for _, r in res.iterrows():
        log(f"{r.subject:<7}{r.n_seg:>6}{r.n_epoch:>7}{r.bacc:>9.4f}"
            f"{r.null:>9.4f}{r.p:>9.4f}")

    n = len(res)
    sig = int((res.p < 0.05).sum())
    w = stats.wilcoxon(res.bacc, res.null)

    log("")
    log("=" * 74)
    log("SUMMARY -- CBraMod embeddings vs handcrafted features")
    log("=" * 74)
    log(f"{'':<34}{'CBraMod':>12}{'spectral':>12}")
    log(f"{'subjects tested':<34}{n:>12}{28:>12}")
    log(f"{'mean within-subject bacc':<34}{res.bacc.mean():>12.4f}{0.3653:>12.4f}")
    log(f"{'mean permutation null':<34}{res.null.mean():>12.4f}{0.3365:>12.4f}")
    log(f"{'difference':<34}{res.bacc.mean()-res.null.mean():>+12.4f}{0.0288:>+12.4f}")
    log(f"{'subjects p<0.05':<34}{sig:>12}{2:>12}")
    log(f"{'expected by chance':<34}{0.05*n:>12.1f}{1.4:>12.1f}")
    log(f"{'Wilcoxon p':<34}{w.pvalue:>12.4f}{0.3620:>12.4f}")
    log("")

    if sig > 2 * 0.05 * n and w.pvalue < 0.05:
        log(">> WITHIN-SUBJECT SIGNAL PRESENT in the foundation-model space.")
        log(">> The subject-adaptive decoder is justified: build test-time")
        log(">> adaptation and few-shot personalisation on these embeddings.")
    else:
        log(">> NO WITHIN-SUBJECT SIGNAL, even in a 9,000-hour pretrained space.")
        log(">> Adaptation methods have no ceiling to reach. Report this as the")
        log(">> bound on personalisation rather than proposing a method that")
        log(">> cannot work.")

    res.round(4).to_csv(os.path.join(OUT, f"cbramod_within_subject{SFX}.csv"),
                        index=False)
    with open(os.path.join(OUT, f"test14_cbramod_within{SFX}.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"Wrote {OUT}\\test14_cbramod_within{SFX}.txt")


if __name__ == "__main__":
    main()
