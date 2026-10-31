"""
Phase 3: Permutation-Calibrated Decodability (PCD).

Problem
-------
Every EEG decoder assumes its input contains task-relevant information. Given a
window it always emits a class. On data with no signal it emits a confident class
anyway -- a softmax still returns ~0.6 for something. Confidence estimates are
miscalibrated precisely where they matter most.

We reformulate: before predicting WHICH class, estimate WHETHER the window carries
decodable information at all.

Method
------
Calibrate abstention against the permutation null rather than against model
confidence.

  1. train decoder f on (X_tr, y_tr)
  2. train B null decoders on (X_tr, permuted y_tr), labels shuffled across
     recordings within subject -- the same null used throughout this project
  3. score s(x) = decision margin (top logit minus runner-up)
  4. D(x) = fraction of null-model scores below s(x)   [empirical p-value]
  5. predict when D(x) >= 1 - alpha, else abstain

D(x) asks: is this window's evidence stronger than what a model trained on
shuffled labels produces on the same input?

Validation
----------
We hold ground truth for BOTH conditions, which is what makes this testable:

  signal present : Sleep-EDF sleep staging   (0.8831 spectral, 0.8217 CBraMod)
  signal absent  : DEED dream affect         (chance; verified cross-subject,
                                              within-subject, and with a
                                              9,000-hour foundation model)

Prediction, stated before running:
  PCD keeps high coverage on Sleep-EDF and collapses toward zero coverage on
  DEED, whereas max-softmax abstention keeps predicting confidently on DEED.

If PCD abstains on Sleep-EDF, or engages on DEED, the method has failed and we
report that.
"""
import os
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
SEED = 20260727
B_NULL = 30
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
    """Decision margin: top probability minus runner-up. Scale-free, works for
    any K, and unlike max-softmax it reflects how DISTINGUISHED the top class is."""
    p = model.predict_proba(X)
    s = np.sort(p, axis=1)
    return s[:, -1] - s[:, -2]


def fit_null(Xtr, ytr, grp_tr, seed):
    """One null decoder, labels shuffled at the level where the label VARIES.

    This is the critical detail. Permuting at the wrong level does not just weaken
    the null, it can invert it:

      * DEED dream affect is constant within a recording, so the unit is the
        recording. Shuffling windows instead would leave most windows with their
        true label.
      * Sleep stage changes from window to window, so the unit is the window.
        Shuffling at subject level would assign every window of a subject one
        stage, and the null model would then learn subject identity (80%
        decodable in these features) and become MORE confident than the real
        model -- driving D toward 0 on data that genuinely has signal.

    Callers pass grp_tr already set to the correct unit; per-window units are
    simply a unique id per row.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"g": grp_tr, "y": ytr}).drop_duplicates("g")
    mp = dict(zip(df.g.to_numpy(), rng.permutation(df.y.to_numpy())))
    yp = pd.Series(grp_tr).map(mp).to_numpy()
    if len(np.unique(yp)) < 2:
        return None
    return clf().fit(Xtr, yp)


def pcd_fold(Xtr, ytr, grp_tr, Xte):
    """Returns (predictions, D scores, max-softmax baseline scores)."""
    model = clf().fit(Xtr, ytr)
    pred = model.predict(Xte)
    s_obs = margin(model, Xte)
    msp = model.predict_proba(Xte).max(axis=1)

    nulls = Parallel(n_jobs=-1)(
        delayed(fit_null)(Xtr, ytr, grp_tr, SEED + 7919 * b) for b in range(B_NULL))
    nulls = [m for m in nulls if m is not None]
    if not nulls:
        return pred, np.zeros(len(Xte)), msp

    # pooled null margin distribution, evaluated on the same held-out windows
    S = np.concatenate([margin(m, Xte) for m in nulls])
    D = np.searchsorted(np.sort(S), s_obs, side="left") / len(S)
    return pred, D, msp


def run_dataset(X, y, subj, grp, name, task):
    log("=" * 76)
    log(f"{name}")
    log("=" * 76)
    log(f"windows {len(y)} | subjects {len(np.unique(subj))} | "
        f"classes {len(np.unique(y))} | null models per fold: {B_NULL}")
    log("")

    yt, yp, Ds, msps = [], [], [], []
    for s in np.unique(subj):
        te = subj == s
        tr = ~te
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            continue
        pred, D, msp = pcd_fold(X[tr], y[tr], grp[tr], X[te])
        yt.append(y[te]); yp.append(pred); Ds.append(D); msps.append(msp)

    yt = np.concatenate(yt); yp = np.concatenate(yp)
    D = np.concatenate(Ds); msp = np.concatenate(msps)

    base_acc = accuracy_score(yt, yp)
    base_bacc = balanced_accuracy_score(yt, yp)
    log(f"no abstention: accuracy {base_acc:.4f} | balanced {base_bacc:.4f}")
    log("")

    rows = []
    log(f"{'alpha':>7}{'PCD cov':>10}{'PCD acc':>10}"
        f"{'MSP cov':>10}{'MSP acc':>10}")
    log("-" * 76)
    for alpha in [0.50, 0.20, 0.10, 0.05, 0.01]:
        keep = D >= (1 - alpha)
        cov = keep.mean()
        acc = accuracy_score(yt[keep], yp[keep]) if keep.sum() > 5 else np.nan

        # max-softmax baseline matched to the SAME coverage, so the comparison is
        # about which windows each rule selects, not how many
        if keep.sum() > 5:
            thr = np.quantile(msp, 1 - cov) if cov < 1 else -np.inf
            keep_m = msp >= thr
            cov_m = keep_m.mean()
            acc_m = (accuracy_score(yt[keep_m], yp[keep_m])
                     if keep_m.sum() > 5 else np.nan)
        else:
            cov_m, acc_m = np.nan, np.nan

        log(f"{alpha:>7.2f}{cov:>10.4f}{acc:>10.4f}{cov_m:>10.4f}{acc_m:>10.4f}")
        rows.append(dict(task=task, alpha=alpha, pcd_coverage=cov,
                         pcd_accuracy=acc, msp_coverage=cov_m,
                         msp_accuracy=acc_m, base_accuracy=base_acc))
    log("")
    log(f"mean D score: {D.mean():.4f}   (0.5 = indistinguishable from null)")
    log(f"fraction of windows with D >= 0.95: {(D >= 0.95).mean():.4f}")
    log("")
    return rows


def main():
    all_rows = []

    # ---------- signal ABSENT: DEED dream affect ----------
    ep = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    meta = ["segment_id", "subject", "oid", "epoch_idx", "n_epochs",
            "epochs_from_end", "E", "film", "stage", "has_text"]
    fc = [c for c in ep.columns if c not in meta]
    ep = ep.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = ep[ep.E.isin(COLL)]
    all_rows += run_dataset(
        d[fc].to_numpy(float), d.E.map(COLL).to_numpy(),
        d.subject.astype(str).to_numpy(), d.segment_id.to_numpy(),
        "DEED dream affect  --  SIGNAL ABSENT (verified chance)", "deed_affect")

    # ---------- signal PRESENT: Sleep-EDF staging ----------
    se = pd.read_parquet(os.path.join(OUT, "sleepedf_features.parquet"))
    fcs = [c for c in se.columns if c.startswith("ch")]
    se = se.replace([np.inf, -np.inf], np.nan).dropna(subset=fcs)
    se = se[se.stage.isin(["W", "N1", "N2"])]
    # Sleep stage varies window to window, so the permutation unit is the window
    # (a unique id per row), NOT the subject.
    all_rows += run_dataset(
        se[fcs].to_numpy(float), se.stage.to_numpy(),
        se.subject.to_numpy(), np.arange(len(se)).astype(str),
        "Sleep-EDF staging  --  SIGNAL PRESENT (verified 0.8831)", "sleepedf_stage")

    res = pd.DataFrame(all_rows)
    res.round(4).to_csv(os.path.join(OUT, "decodability.csv"), index=False)

    log("=" * 76)
    log("DOES THE ESTIMATOR SEPARATE THE TWO CONDITIONS?")
    log("=" * 76)
    piv = res.pivot_table(index="alpha", columns="task",
                          values="pcd_coverage").round(4)
    log(piv.to_string())
    log("")
    a05 = res[res.alpha == 0.05]
    cov_deed = float(a05[a05.task == "deed_affect"].pcd_coverage.iloc[0])
    cov_sedf = float(a05[a05.task == "sleepedf_stage"].pcd_coverage.iloc[0])
    log(f"at alpha=0.05:  DEED coverage {cov_deed:.4f} | "
        f"Sleep-EDF coverage {cov_sedf:.4f}")
    log("")
    if cov_sedf > 0.5 and cov_deed < 0.2:
        log(">> METHOD WORKS. High coverage where signal exists, collapse where it")
        log(">> does not. Compare the MSP columns: confidence-based abstention")
        log(">> keeps predicting on data with no signal.")
    else:
        log(">> METHOD DOES NOT SEPARATE THE CONDITIONS AS PREDICTED.")
        log(">> Report as stated; do not retune the threshold to force it.")

    with open(os.path.join(OUT, "test16_decodability.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"Wrote {OUT}\\test16_decodability.txt")


if __name__ == "__main__":
    main()
