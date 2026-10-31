"""
Re-run the CBraMod control on the EXPANDED Sleep-EDF cohort (Reviewer 2, C9).

Why this is necessary
---------------------
The manuscript reports CBraMod at 0.8217 balanced accuracy for sleep staging and
places it beside the spectral pipeline's 0.7748. Those two numbers were measured
on different cohorts. 14_cbramod_finetune.py reads sleepedf_raw.npy, which
12_raw_dataset.py built from the sleep-edf-database-1.0.0 release: 8 recordings,
12,252 windows. The 0.7748 figure comes from Sleep-EDF Expanded: 197 recordings,
100 participants. Presenting them side by side implies a comparison that was
never made.

On those same 8 recordings the spectral pipeline reaches 0.8831, so CBraMod was
in fact six points BELOW the handcrafted baseline on matched data, while the
manuscript describes it only as confirming transfer.

This script fixes the comparison rather than caveating it. Both models are run on
the identical cohort, the identical windows and the identical folds; only the
representation changes.

Design
------
  cohort      all participants in Sleep-EDF Expanded that yield usable windows
  windows     up to N_PER_REC per recording, sampled evenly across W/N1/N2 so
              the class mix does not depend on how long a recording is
  budget      20 epochs, early stopping on a validation split of a fifth
              of the TRAINING participants (patience 5). The original code
              validated on one participant, whose score swings several
              points per epoch, making checkpoint selection unreliable;
              50_cbramod_convergence.py measured that directly.
  protocol    5-fold cross-validation grouped on participant. Leave-one-subject-
              out over 100 participants would mean 100 transformer fine-tunes
              and is not affordable; 5-fold grouped preserves the property that
              matters, which is that no participant appears on both sides.
  baseline    the same spectral feature pipeline used throughout the paper,
              fitted on exactly the same folds and windows

Reporting the spectral baseline on these folds is the point. Without it, a
different CBraMod number could be attributed to the change of cohort rather than
to the model, and the same ambiguity would be reintroduced in a new place.
"""
import os
import re
import sys
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import mne
from scipy import signal as sps

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
EXP = os.path.join(ROOT, "Sleep-EDF Expanded",
                   "sleep-edf-database-expanded-1.0.0")
RAW = os.path.join(OUT, "sleepedf_exp_raw.npy")
METAP = os.path.join(OUT, "sleepedf_exp_raw_meta.parquet")

SEED = 20260727
FS_SRC = 100.0
FS = 200.0                     # CBraMod expects 200 Hz
EPOCH_S = 20
EPOCH_N = int(FS * EPOCH_S)    # 4000
SCORING_S = 30
CROP_MIN = 30
ARTIFACT_UV = 300.0
N_PER_REC = 180                # per recording, balanced across the 3 stages
STAGES = ["W", "N1", "N2"]

STAGE_MAP = {"Sleep stage W": "W", "Sleep stage 1": "N1",
             "Sleep stage 2": "N2", "Sleep stage 3": "N3",
             "Sleep stage 4": "N3", "Sleep stage R": "REM"}

_b, _a = sps.butter(4, [0.5 / (FS / 2), 45.0 / (FS / 2)], btype="band")

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def parse_id(stem):
    m = re.match(r"(S[CT])\d(\d{2})(\d)", stem)
    return (m.group(1) + m.group(2)) if m else stem


def stage_series(hyp, n):
    ann = mne.read_annotations(hyp)
    lab = np.full(n, None, dtype=object)
    for on, dur, desc in zip(ann.onset, ann.duration, ann.description):
        st = STAGE_MAP.get(desc)
        if st is None:
            continue
        i0, i1 = int(round(on / SCORING_S)), int(round((on + dur) / SCORING_S))
        lab[max(0, i0):min(n, i1)] = st
    return lab


def one_recording(psg, hyp, rng):
    stem = os.path.basename(psg).split("-")[0]
    subject = parse_id(stem)
    try:
        raw = mne.io.read_raw_edf(psg, preload=False)
        eeg = [c for c in raw.ch_names if c.startswith("EEG")]
        if len(eeg) < 2:
            return None
        raw.pick(eeg[:2])
        total = raw.times[-1]
        n_sc = int(total / SCORING_S)
        labs = stage_series(hyp, n_sc)
        asleep = np.where([l is not None and l != "W" for l in labs])[0]
        if not len(asleep):
            return None
        pad = int(CROP_MIN * 60 / SCORING_S)
        lo = max(0, asleep[0] - pad)
        hi = min(n_sc, asleep[-1] + pad + 1)
        raw.crop(tmin=lo * SCORING_S, tmax=min(total, hi * SCORING_S))
        raw.load_data()
        raw.resample(FS)
        data = sps.filtfilt(_b, _a, raw.get_data() * 1e6, axis=-1)
        del raw
    except Exception:
        return None

    by_stage = {s: [] for s in STAGES}
    for i in range(lo, hi):
        st = labs[i]
        if st not in STAGES:
            continue
        s0 = int((i - lo) * SCORING_S * FS)
        ep = data[:, s0:s0 + EPOCH_N]
        if ep.shape[1] < EPOCH_N or np.abs(ep).max() > ARTIFACT_UV:
            continue
        if np.allclose(ep.std(axis=1), 0):
            continue
        by_stage[st].append(ep.astype(np.float32))

    per = max(1, N_PER_REC // len(STAGES))
    segs, meta = [], []
    for st in STAGES:
        pool = by_stage[st]
        if not pool:
            continue
        take = rng.choice(len(pool), size=min(per, len(pool)), replace=False)
        for t in take:
            segs.append(pool[t])
            meta.append((subject, stem, st))
    del data
    if not segs:
        return None
    return np.stack(segs), meta


def build():
    rng = np.random.default_rng(SEED)
    pairs = []
    for sub in ("sleep-cassette", "sleep-telemetry"):
        psgs = sorted(glob.glob(os.path.join(EXP, sub, "*PSG.edf")))
        hyps = sorted(glob.glob(os.path.join(EXP, sub, "*Hypnogram.edf")))
        hm = {os.path.basename(h).split("-")[0][:6]: h for h in hyps}
        for p in psgs:
            k = os.path.basename(p).split("-")[0][:6]
            if k in hm:
                pairs.append((p, hm[k]))
    log(f"recordings found: {len(pairs)}")

    chunks, metas = [], []
    for i, (p, h) in enumerate(pairs, 1):
        r = one_recording(p, h, rng)
        if r is not None:
            chunks.append(r[0]); metas += r[1]
        if i % 25 == 0 or i == len(pairs):
            n = sum(c.shape[0] for c in chunks)
            log(f"  {i}/{len(pairs)} recordings -> {n} windows")

    X = np.concatenate(chunks, axis=0)
    m = pd.DataFrame(metas, columns=["subject", "recording", "stage"])
    np.save(RAW, X)
    m.to_parquet(METAP, index=False)
    log(f"wrote {RAW}  shape {X.shape}  "
        f"({X.nbytes / 1e9:.2f} GB)")
    log(f"participants {m.subject.nunique()} | "
        f"stages {dict(m.stage.value_counts())}")
    return X, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    log("=" * 78)
    log("CBRAMOD CONTROL ON SLEEP-EDF EXPANDED")
    log("=" * 78)

    if os.path.exists(RAW) and os.path.exists(METAP):
        X = np.load(RAW, mmap_mode="r")
        m = pd.read_parquet(METAP)
        log(f"cached raw array: {X.shape}, "
            f"{m.subject.nunique()} participants")
    else:
        X, m = build()
    if args.build_only:
        return

    # ---- spectral baseline on the identical windows and folds -----------
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import balanced_accuracy_score, accuracy_score

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    feat_mod = import_module("35_overlap_features") if False else None

    log("")
    log("computing spectral features on the identical windows...")
    from joblib import Parallel, delayed
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ovf", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "35_overlap_features.py"))
    ovf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ovf)

    def feats(i):
        return ovf.epoch_features(np.asarray(X[i], dtype=np.float64))

    rows = Parallel(n_jobs=20)(delayed(feats)(i) for i in range(len(m)))
    F = pd.DataFrame(rows)
    F = F.replace([np.inf, -np.inf], np.nan)
    ok = F.notna().all(axis=1).to_numpy()
    log(f"usable windows: {int(ok.sum())} of {len(m)}")

    Xs = F[ok].to_numpy(float)
    ys = m.stage.to_numpy()[ok]
    gs = m.subject.to_numpy()[ok]

    folds = list(StratifiedGroupKFold(5, shuffle=True,
                                      random_state=SEED).split(Xs, ys, gs))
    yt, yp = [], []
    for tr, te in folds:
        mdl = make_pipeline(StandardScaler(),
                            LogisticRegression(C=0.1, max_iter=2000,
                                               class_weight="balanced"))
        mdl.fit(Xs[tr], ys[tr])
        yp.append(mdl.predict(Xs[te])); yt.append(ys[te])
    yt_s, yp_s = np.concatenate(yt), np.concatenate(yp)
    sp_bal = balanced_accuracy_score(yt_s, yp_s)
    sp_acc = accuracy_score(yt_s, yp_s)
    log(f"spectral (LogReg) : accuracy {sp_acc:.4f} | balanced {sp_bal:.4f}")

    yt, yp = [], []
    for tr, te in folds:
        mdl = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     n_jobs=20, random_state=SEED)
        mdl.fit(Xs[tr], ys[tr])
        yp.append(mdl.predict(Xs[te])); yt.append(ys[te])
    rf_bal = balanced_accuracy_score(np.concatenate(yt), np.concatenate(yp))
    log(f"spectral (RF)     : balanced {rf_bal:.4f}")

    # ---- CBraMod on the same folds ---------------------------------------
    import torch
    sys.path.insert(0, os.path.join(ROOT, "foundation", "CBraMod"))
    cbm = importlib.util.spec_from_file_location(
        "cbf", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "14_cbramod_finetune.py"))
    cb = importlib.util.module_from_spec(cbm)
    cbm.loader.exec_module(cb)

    classes = STAGES
    # CBRAMOD_SCALE=uv100: CBraMod receives the windows in units of 100 uV,
    # its pretraining convention (144_cbramod_scaling.py); by default it
    # receives microvolts. The spectral baseline always uses microvolts.
    scale = os.environ.get("CBRAMOD_SCALE", "")
    assert scale in ("", "uv100"), scale
    sfx = f"_{scale}" if scale else ""

    class _Scaled:
        def __init__(self, a, f):
            self.a, self.f = a, f

        def __getitem__(self, k):
            return np.asarray(self.a[k], dtype=np.float32) / self.f

    Xcb = _Scaled(X, 100.0) if scale else X
    log(f"CBraMod input scale: {'EEG / 100 uV' if scale else 'microvolts'}")
    cls_idx = {c: i for i, c in enumerate(classes)}
    yc = np.array([cls_idx[v] for v in ys])
    idx_all = np.where(ok)[0]

    log("")
    log(f"CBraMod: device {cb.DEV}, "
        f"{'linear probe' if args.probe else 'full fine-tune'}, "
        f"{args.epochs} epochs max, batch {args.bs}")
    yts, yps = [], []
    for k, (tr, te) in enumerate(folds, 1):
        gtr = gs[tr]
        tr_ids = np.unique(gtr)
        n_val = max(2, len(tr_ids) // 5)
        val_ids = np.random.default_rng(SEED + k).choice(
            tr_ids, size=n_val, replace=False)
        is_val = np.isin(gtr, val_ids)
        va = tr[is_val]
        tr2 = tr[~is_val]
        cw = len(yc[tr2]) / (len(classes) *
                             np.bincount(yc[tr2],
                                         minlength=len(classes)).clip(min=1))
        cb.seed_all(SEED)
        model = cb.CBraModClassifier(n_classes=len(classes), pretrained=True,
                                     freeze=args.probe).to(cb.DEV)
        sel_tr = np.zeros(len(yc), bool); sel_tr[tr2] = True
        sel_va = np.zeros(len(yc), bool); sel_va[va] = True
        sel_te = np.zeros(len(yc), bool); sel_te[te] = True
        dl_tr = cb.make_loader(Xcb, idx_all, sel_tr, yc, args.bs, True)
        dl_va = cb.make_loader(Xcb, idx_all, sel_va, yc, args.bs * 2, False)
        dl_te = cb.make_loader(Xcb, idx_all, sel_te, yc, args.bs * 2, False)
        cb.train_fold(model, dl_tr, dl_va, cw, args.epochs, args.lr,
                      args.probe, patience=5)
        a, b = cb.predict(model, dl_te)
        log(f"  fold {k}: balanced {balanced_accuracy_score(a, b):.4f} "
            f"(n={len(a)})")
        yts.append(a); yps.append(b)
        del model
        torch.cuda.empty_cache()

    yt_c, yp_c = np.concatenate(yts), np.concatenate(yps)
    cb_bal = balanced_accuracy_score(yt_c, yp_c)
    cb_acc = accuracy_score(yt_c, yp_c)

    log("")
    log("=" * 78)
    log("MATCHED COMPARISON -- same cohort, same windows, same folds")
    log("=" * 78)
    log(f"{'model':<34}{'accuracy':>11}{'balanced':>11}")
    log("-" * 78)
    log(f"{'Spectral features, LogReg':<34}{sp_acc:>11.4f}{sp_bal:>11.4f}")
    log(f"{'Spectral features, RandomForest':<34}{'':>11}{rf_bal:>11.4f}")
    log(f"{'CBraMod (9,000 h pretraining)':<34}{cb_acc:>11.4f}{cb_bal:>11.4f}")
    log("")
    log(f"difference, CBraMod minus best spectral: "
        f"{cb_bal - max(sp_bal, rf_bal):+.4f}")
    log("")
    log("For reference, the 8-recording Sleep-EDF 1.0.0 cohort used in the")
    log("submitted manuscript gave CBraMod 0.8217 and spectral 0.8831.")

    pd.DataFrame([
        dict(model="spectral_logreg", accuracy=sp_acc, balanced=sp_bal),
        dict(model="spectral_rf", accuracy=np.nan, balanced=rf_bal),
        dict(model="cbramod", accuracy=cb_acc, balanced=cb_bal),
    ]).round(4).to_csv(os.path.join(OUT, f"cbramod_expanded{sfx}.csv"), index=False)
    with open(os.path.join(OUT, f"test39_cbramod_expanded{sfx}.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"\nWrote {OUT}\\cbramod_expanded{sfx}.csv")


if __name__ == "__main__":
    main()
