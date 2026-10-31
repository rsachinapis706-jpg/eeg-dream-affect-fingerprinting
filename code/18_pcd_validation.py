"""
PCD validation suite: graded conditions, fixed-threshold baseline, bootstrap CIs,
calibration diagnosis, and representation independence.

Addresses five gaps in the first PCD run.

1. GRADED CONDITIONS. The first run tested two extremes. A detector that fires on
   0.88-accuracy data and not on 0.33-accuracy data might only be a binary switch.
   Adding an intermediate task tests whether coverage TRACKS signal strength:

       strong        Sleep-EDF sleep stage    0.8831
       intermediate  DEED sleep stage         0.4459   <- new
       absent        DEED dream affect        0.3373

2. FIXED-THRESHOLD BASELINE. The first run matched max-softmax to PCD's coverage
   by construction, so it could not show that confidence-based abstention FAILS to
   abstain. Here MSP is run at its natural thresholds (0.5/0.7/0.9), which is what
   a practitioner would actually do.

3. BOOTSTRAP CONFIDENCE INTERVALS, resampling SUBJECTS rather than windows, since
   windows within a subject are not independent.

4. CALIBRATION. Observed coverage on no-signal data was 0.107 where 0.05 was
   expected. Hypothesis: null models fit a harder task (permuted labels), so their
   margins are systematically smaller, biasing D upward for every input. The
   leave-one-null-out check tests this directly -- score one null model against the
   remaining ones. Under sound calibration that yields mean D ~ 0.5.

5. REPRESENTATION INDEPENDENCE. Repeat on frozen CBraMod embeddings. If PCD works
   on both handcrafted and learned representations it is a property of the method,
   not of the features.
"""
import os
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
SEED = 20260727
B_NULL = 30
N_BOOT = 1000
COLL = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def clf():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced"))


def margin(model, X):
    p = model.predict_proba(X)
    s = np.sort(p, axis=1)
    return s[:, -1] - s[:, -2]


def fit_null(Xtr, ytr, grp_tr, seed):
    """Labels shuffled at the level where the label varies (caller sets grp_tr)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"g": grp_tr, "y": ytr}).drop_duplicates("g")
    mp = dict(zip(df.g.to_numpy(), rng.permutation(df.y.to_numpy())))
    yp = pd.Series(grp_tr).map(mp).to_numpy()
    if len(np.unique(yp)) < 2:
        return None
    return clf().fit(Xtr, yp)


def collect(X, y, subj, grp):
    """LOSO. Returns per-window truth, prediction, D, max-softmax, subject,
    plus the leave-one-null-out D used for the calibration check."""
    yt, yp, Ds, msps, ss, Dloo = [], [], [], [], [], []
    for s in np.unique(subj):
        te = subj == s
        tr = ~te
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            continue
        model = clf().fit(X[tr], y[tr])
        s_obs = margin(model, X[te])
        nulls = Parallel(n_jobs=-1)(
            delayed(fit_null)(X[tr], y[tr], grp[tr], SEED + 7919 * b)
            for b in range(B_NULL))
        nulls = [m for m in nulls if m is not None]
        if len(nulls) < 5:
            continue
        M = np.stack([margin(m, X[te]) for m in nulls])      # (B, n_te)

        pooled = np.sort(M.reshape(-1))
        D = np.searchsorted(pooled, s_obs, side="left") / len(pooled)

        # calibration: treat null model 0 as if it were the observed model and
        # score it against the remaining nulls. Should be ~Uniform(0,1).
        rest = np.sort(M[1:].reshape(-1))
        Dl = np.searchsorted(rest, M[0], side="left") / len(rest)

        yt.append(y[te]); yp.append(model.predict(X[te]))
        Ds.append(D); Dloo.append(Dl)
        msps.append(model.predict_proba(X[te]).max(axis=1))
        ss.append(np.full(te.sum(), s))
    return (np.concatenate(yt), np.concatenate(yp), np.concatenate(Ds),
            np.concatenate(msps), np.concatenate(ss), np.concatenate(Dloo))


def boot_ci(yt, yp, keep, subj, stat="acc"):
    """Cluster bootstrap over subjects."""
    us = np.unique(subj)
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(us, size=len(us), replace=True)
        idx = np.concatenate([np.where(subj == u)[0] for u in pick])
        if stat == "cov":
            vals.append(keep[idx].mean())
        else:
            k = idx[keep[idx]]
            vals.append(accuracy_score(yt[k], yp[k]) if len(k) > 5 else np.nan)
    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 50:
        return (np.nan, np.nan)
    return tuple(np.percentile(vals, [2.5, 97.5]))


def run(X, y, subj, grp, name, tag, expected):
    log("=" * 78)
    log(f"{name}")
    log(f"   verified reference accuracy: {expected}")
    log("=" * 78)
    yt, yp, D, msp, ss, Dloo = collect(X, y, subj, grp)
    base = accuracy_score(yt, yp)
    log(f"windows {len(yt)} | subjects {len(np.unique(ss))} | "
        f"no-abstention accuracy {base:.4f} (balanced "
        f"{balanced_accuracy_score(yt, yp):.4f})")
    log("")

    rows = []
    log("-- PCD (permutation-calibrated) --")
    log(f"{'alpha':>7}{'coverage':>11}{'95% CI':>20}{'accuracy':>11}{'95% CI':>20}")
    for alpha in [0.20, 0.10, 0.05, 0.01]:
        keep = D >= (1 - alpha)
        cov = keep.mean()
        cl, ch = boot_ci(yt, yp, keep, ss, "cov")
        if keep.sum() > 5:
            acc = accuracy_score(yt[keep], yp[keep])
            al, ah = boot_ci(yt, yp, keep, ss, "acc")
        else:
            acc, al, ah = np.nan, np.nan, np.nan
        log(f"{alpha:>7.2f}{cov:>11.4f}   [{cl:.4f}, {ch:.4f}]"
            f"{acc:>11.4f}   [{al:.4f}, {ah:.4f}]")
        rows.append(dict(task=tag, rule="PCD", param=alpha, coverage=cov,
                         cov_lo=cl, cov_hi=ch, accuracy=acc, acc_lo=al,
                         acc_hi=ah, base_accuracy=base))
    log("")

    log("-- Max-softmax at FIXED thresholds (what a practitioner would use) --")
    log(f"{'thresh':>7}{'coverage':>11}{'95% CI':>20}{'accuracy':>11}{'95% CI':>20}")
    for thr in [0.50, 0.70, 0.90]:
        keep = msp >= thr
        cov = keep.mean()
        cl, ch = boot_ci(yt, yp, keep, ss, "cov")
        if keep.sum() > 5:
            acc = accuracy_score(yt[keep], yp[keep])
            al, ah = boot_ci(yt, yp, keep, ss, "acc")
        else:
            acc, al, ah = np.nan, np.nan, np.nan
        log(f"{thr:>7.2f}{cov:>11.4f}   [{cl:.4f}, {ch:.4f}]"
            f"{acc:>11.4f}   [{al:.4f}, {ah:.4f}]")
        rows.append(dict(task=tag, rule="MSP_fixed", param=thr, coverage=cov,
                         cov_lo=cl, cov_hi=ch, accuracy=acc, acc_lo=al,
                         acc_hi=ah, base_accuracy=base))
    log("")

    ks = stats.kstest(D, "uniform")
    ksl = stats.kstest(Dloo, "uniform")
    log("-- Calibration --")
    log(f"mean D (observed model)      : {D.mean():.4f}")
    log(f"mean D (leave-one-null-out)  : {Dloo.mean():.4f}   "
        f"<- should be ~0.5 if calibrated")
    log(f"P(D>=0.95) observed          : {(D >= 0.95).mean():.4f}")
    log(f"P(D>=0.95) null-vs-null      : {(Dloo >= 0.95).mean():.4f}   "
        f"<- the true false-positive rate")
    log(f"KS vs uniform: observed p={ks.pvalue:.3g} | null-vs-null p={ksl.pvalue:.3g}")
    log("")
    rows.append(dict(task=tag, rule="calibration", param=np.nan,
                     coverage=(Dloo >= 0.95).mean(), cov_lo=np.nan, cov_hi=np.nan,
                     accuracy=Dloo.mean(), acc_lo=np.nan, acc_hi=np.nan,
                     base_accuracy=base))
    return rows


def main():
    all_rows = []

    ep = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    meta = ["segment_id", "subject", "oid", "epoch_idx", "n_epochs",
            "epochs_from_end", "E", "film", "stage", "has_text"]
    fc = [c for c in ep.columns if c not in meta]
    ep = ep.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)

    se = pd.read_parquet(os.path.join(OUT, "sleepedf_features.parquet"))
    fcs = [c for c in se.columns if c.startswith("ch")]
    se = se.replace([np.inf, -np.inf], np.nan).dropna(subset=fcs)
    se = se[se.stage.isin(["W", "N1", "N2"])]

    # ---- condition 1: STRONG signal --------------------------------
    all_rows += run(se[fcs].to_numpy(float), se.stage.to_numpy(),
                    se.subject.to_numpy(), np.arange(len(se)).astype(str),
                    "STRONG SIGNAL -- Sleep-EDF sleep stage", "strong", "0.8831")

    # ---- condition 2: INTERMEDIATE signal --------------------------
    ds = ep[ep.stage.isin(["W", "N1", "N2"])]
    all_rows += run(ds[fc].to_numpy(float), ds.stage.to_numpy(),
                    ds.subject.astype(str).to_numpy(),
                    ds.segment_id.to_numpy(),
                    "INTERMEDIATE SIGNAL -- DEED sleep stage", "intermediate",
                    "0.4459")

    # ---- condition 3: NO signal ------------------------------------
    da = ep[ep.E.isin(COLL)]
    all_rows += run(da[fc].to_numpy(float), da.E.map(COLL).to_numpy(),
                    da.subject.astype(str).to_numpy(), da.segment_id.to_numpy(),
                    "NO SIGNAL -- DEED dream affect", "absent", "0.3373")

    # ---- condition 4: representation independence ------------------
    emb_path = os.path.join(OUT, "cbramod_embeddings.npy")
    if os.path.exists(emb_path):
        Z = np.load(emb_path)
        m = pd.read_parquet(os.path.join(OUT, "deed_raw_meta.parquet"))
        d = m[m.E.isin(COLL)]
        all_rows += run(Z[d.index.to_numpy()], d.E.map(COLL).to_numpy(),
                        d.subject.astype(str).to_numpy(),
                        d.segment_id.to_numpy(),
                        "NO SIGNAL -- DEED affect on CBraMod embeddings",
                        "absent_cbramod", "0.3147")

    res = pd.DataFrame(all_rows)
    res.round(4).to_csv(os.path.join(OUT, "pcd_validation.csv"), index=False)

    log("=" * 78)
    log("IS PCD GRADED? (coverage at alpha = 0.05)")
    log("=" * 78)
    p = res[(res.rule == "PCD") & (res.param == 0.05)]
    log(f"{'condition':<20}{'reference acc':>15}{'PCD coverage':>15}"
        f"{'MSP@0.5 cov':>15}")
    for _, r in p.iterrows():
        mspr = res[(res.task == r.task) & (res.rule == "MSP_fixed")
                   & (res.param == 0.50)]
        mc = float(mspr.coverage.iloc[0]) if len(mspr) else np.nan
        log(f"{r.task:<20}{r.base_accuracy:>15.4f}{r.coverage:>15.4f}{mc:>15.4f}")
    log("")

    with open(os.path.join(OUT, "test17_pcd_validation.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"Wrote {OUT}\\test17_pcd_validation.txt")


if __name__ == "__main__":
    main()
