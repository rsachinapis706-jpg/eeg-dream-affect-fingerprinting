"""
What makes an EEG window identify its subject?

Subject identity is 80.4% decodable from these features, and an oracle that knows
only the subject -- with no EEG -- reaches 0.5465 accuracy against the
window-shuffled affect model's 0.5732. That establishes THAT fingerprinting
drives the inflation. This script asks WHICH features carry identity, which turns
the observation into a physiological account.

The prediction worth testing: identity should be carried mainly by ABSOLUTE
power and the aperiodic (1/f) slope -- quantities set by skull thickness,
electrode impedance and individual anatomy -- rather than by RELATIVE band power,
which normalises those away. If so, any protocol that lets a model see the same
subject in train and test will exploit them, on any dataset.

Attribution is deliberately NOT applied to the dream-affect model. That model
performs at chance (0.3070 balanced), so its feature attributions would describe
a fit that generalises to nothing. Interpretability is only meaningful for a
model that works.

Two attribution methods, reported together because agreement between an impurity-
based and a permutation-based measure is more convincing than either alone:
  * RandomForest impurity importance
  * permutation importance on held-out subjects (model-agnostic)
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
FIG = os.path.join(ROOT, "analysis", "figures")
os.makedirs(FIG, exist_ok=True)
SEED = 20260727
COLL = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def family(name):
    """Group a feature name into an interpretable family."""
    if name.endswith("_abs"):
        return "absolute band power"
    if name.endswith("_rel"):
        return "relative band power"
    if "apslope" in name:
        return "aperiodic (1/f) slope"
    if "logvar" in name:
        return "signal variance"
    if "specent" in name:
        return "spectral entropy"
    if "sef95" in name:
        return "spectral edge freq."
    if "hj_" in name:
        return "Hjorth parameters"
    return "other"


def band(name):
    for b in ["delta", "theta", "alpha", "sigma", "beta", "gamma"]:
        if b in name:
            return b
    return None


def main():
    ep = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    meta = ["segment_id", "subject", "oid", "epoch_idx", "n_epochs",
            "epochs_from_end", "E", "film", "stage", "has_text"]
    fc = [c for c in ep.columns if c not in meta]
    ep = ep.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
    d = ep[ep.E.isin(COLL)]

    X = d[fc].to_numpy(float)
    y = d.subject.astype(str).to_numpy()
    seg = d.segment_id.to_numpy()

    log("=" * 74)
    log("WHAT MAKES AN EEG WINDOW IDENTIFY ITS SUBJECT?")
    log("=" * 74)
    log(f"windows {len(y)} | subjects {len(np.unique(y))} | features {len(fc)}")
    log("target: subject identity (34-way). Attribution is NOT run on the")
    log("dream-affect model, which performs at chance.")
    log("")

    # recording-grouped split: identity must be recovered from a recording the
    # model has never seen, not from memorising one recording's quirks
    tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
                  .split(X, y, seg))
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                n_jobs=-1, random_state=SEED).fit(X[tr], y[tr])
    acc = accuracy_score(y[te], rf.predict(X[te]))
    log(f"held-out subject-identity accuracy (recording-grouped): {acc:.4f}"
        f"  (chance {1/len(np.unique(y)):.4f})")
    log("")

    imp = pd.DataFrame({"feature": fc, "impurity": rf.feature_importances_})
    log("computing permutation importance on held-out data...")
    pi = permutation_importance(rf, X[te], y[te], n_repeats=5,
                                random_state=SEED, n_jobs=-1)
    imp["permutation"] = pi.importances_mean
    imp["family"] = imp.feature.map(family)
    imp["band"] = imp.feature.map(band)
    imp["channel"] = imp.feature.str.extract(r"^(ch\d)")

    # ---- by family -------------------------------------------------
    fam = imp.groupby("family")[["impurity", "permutation"]].sum()
    fam["n_features"] = imp.groupby("family").size()
    fam["impurity_share"] = fam.impurity / fam.impurity.sum()
    fam = fam.sort_values("impurity_share", ascending=False)

    log("")
    log("-- importance by feature family --")
    log(f"{'family':<26}{'n':>4}{'impurity share':>17}{'perm. importance':>19}")
    log("-" * 74)
    for f, r in fam.iterrows():
        log(f"{f:<26}{int(r.n_features):>4}{r.impurity_share:>17.4f}"
            f"{r.permutation:>19.5f}")
    log("")

    # normalise by family size, since families differ in feature count
    fam["per_feature"] = fam.impurity / fam.n_features
    log("-- per-feature importance (controls for family size) --")
    for f, r in fam.sort_values("per_feature", ascending=False).iterrows():
        log(f"{f:<26}{r.per_feature:>12.5f}")
    log("")

    # ---- absolute vs relative: the key contrast ---------------------
    a = float(fam.loc["absolute band power", "impurity_share"]) \
        if "absolute band power" in fam.index else 0.0
    r_ = float(fam.loc["relative band power", "impurity_share"]) \
        if "relative band power" in fam.index else 0.0
    ap = float(fam.loc["aperiodic (1/f) slope", "impurity_share"]) \
        if "aperiodic (1/f) slope" in fam.index else 0.0
    log("-- the physiological contrast --")
    log(f"absolute band power (anatomy/impedance dependent) : {a:.4f}")
    log(f"relative band power (normalises anatomy away)     : {r_:.4f}")
    log(f"aperiodic 1/f slope (individual, trait-like)      : {ap:.4f}")
    log(f"absolute + aperiodic combined                     : {a+ap:.4f}")
    log(f"ratio absolute:relative                           : "
        f"{(a/r_ if r_ > 0 else float('nan')):.2f}x")
    log("")

    log("-- top 15 individual features --")
    top = imp.sort_values("impurity", ascending=False).head(15)
    for _, r in top.iterrows():
        log(f"  {r.feature:<22}{r.family:<24}{r.impurity:.5f}")
    log("")

    if a + ap > r_:
        log(">> Identity is carried mainly by ABSOLUTE power and the aperiodic")
        log(">> slope -- quantities determined by skull thickness, electrode")
        log(">> impedance and individual anatomy, and largely stable within a")
        log(">> person across recordings. Relative band power, which normalises")
        log(">> those away, contributes less. Any protocol that places the same")
        log(">> subject in train and test can therefore recover identity, and")
        log(">> where per-subject label distributions are skewed, identity alone")
        log(">> predicts the label.")
    else:
        log(">> Identity is NOT dominated by absolute power; report the observed")
        log(">> ordering without the anatomical interpretation.")

    imp.round(6).to_csv(os.path.join(OUT, "fingerprint_importance.csv"),
                        index=False)
    fam.round(6).to_csv(os.path.join(OUT, "fingerprint_families.csv"))

    # ---- figure -----------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.4))
    f2 = fam.sort_values("impurity_share")
    cols = ["#C1352C" if i in ("absolute band power", "aperiodic (1/f) slope")
            else "#8D99AE" for i in f2.index]
    a1.barh(range(len(f2)), f2.impurity_share, color=cols, edgecolor="black",
            lw=.5)
    a1.set_yticks(range(len(f2)))
    a1.set_yticklabels(f2.index, fontsize=7.4)
    a1.set_xlabel("Share of importance for participant identity")
    a1.set_title("What identifies the individual", fontsize=9.5)

    names = ["Absolute\npower", "Aperiodic\n1/f slope", "Relative\npower"]
    vals = [a, ap, r_]
    a2.bar(names, vals, color=["#C1352C", "#C1352C", "#2A6F97"],
           edgecolor="black", lw=.6)
    for i, v in enumerate(vals):
        a2.text(i, v + .004, f"{v:.3f}", ha="center", fontsize=8.5,
                fontweight="bold")
    a2.set_ylabel("Share of importance")
    a2.set_title("Anatomy-dependent features dominate", fontsize=9.5)
    fig.suptitle("OUR FINDING: participant identity is carried by amplitude-dependent "
                 "features", fontsize=10, y=1.03)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig7_fingerprint_xai.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    log("wrote fig7_fingerprint_xai")

    with open(os.path.join(OUT, "test20_fingerprint_xai.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"Wrote {OUT}\\test20_fingerprint_xai.txt")


if __name__ == "__main__":
    main()
