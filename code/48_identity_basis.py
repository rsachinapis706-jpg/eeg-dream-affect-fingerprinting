"""
What actually carries participant identity, and does it behave like anatomy?
(Reviewer 2, C7.)

The reviewer objects that a feature-attribution plot showing absolute band power
and the aperiodic slope to be important does not establish that those features
are CAUSED by individual anatomy and electrode impedance. That objection is
correct: attribution identifies which inputs a model relies on, not why those
inputs differ between people.

Rather than soften the wording and leave the claim unsupported, this script runs
the experiment that discriminates between the two explanations the manuscript
conflates.

The discriminating prediction
-----------------------------
  anatomy          Skull thickness, cortical geometry and head size are stable
                   traits. Identity carried by them should transfer ACROSS
                   RECORDING SESSIONS: a model trained on a participant's first
                   night should recognise them on a later night.

  session factors  Electrode impedance, exact cap placement, skin preparation
                   and reference quality are re-established at every setup.
                   Identity carried by them should NOT transfer across nights,
                   because the nuisance variable is redrawn each session.

DEED supports the test: 27 of 34 participants recorded more than one night.
Identity is therefore decoded under three partitionings of increasing severity:

  window-shuffled   windows from the same recording on both sides (upper bound)
  recording-grouped whole recordings held out, nights may still be shared
  night held out    an entire session held out; the model has never seen that
                    participant under that electrode setup

A feature-family ablation is run alongside, separating amplitude-dependent
features (absolute band power, log-variance) from amplitude-normalised ones
(relative band power, spectral entropy, edge frequency) and from the aperiodic
slope, so the physiological reading is tied to what survives the session change
rather than to what an importance plot ranks highly.

Whatever the outcome, it constrains the manuscript's wording. High cross-session
identity would support the anatomical account; a collapse would show that the
manuscript's stated mechanism is wrong even though its empirical consequence --
that identity is decodable whenever a participant appears on both sides of a
partition -- is unaffected.
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")

SEED = 20260727
N_SPLITS = 5
N_JOBS = 20
META = {"segment_id", "subject", "oid", "epoch_idx", "n_epochs",
        "epochs_from_end", "E", "film", "stage", "has_text"}

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def families(cols):
    """Split the 108 spectral features into interpretable families."""
    f = {
        "absolute power": [c for c in cols if c.endswith("_abs")],
        "log-variance": [c for c in cols if c.endswith("_logvar")],
        "relative power": [c for c in cols if c.endswith("_rel")],
        "aperiodic slope": [c for c in cols if c.endswith("_apslope")],
        "spectral shape": [c for c in cols
                           if c.endswith("_specent") or c.endswith("_sef95")],
        "Hjorth": [c for c in cols
                   if c.endswith("_hj_mob") or c.endswith("_hj_comp")],
    }
    f["amplitude-dependent"] = f["absolute power"] + f["log-variance"]
    f["amplitude-normalised"] = f["relative power"] + f["spectral shape"]
    f["all features"] = list(cols)
    return f


def rf(seed=SEED):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                  n_jobs=N_JOBS, random_state=seed)


def decode(X, y, groups, mode):
    if mode == "shuffled":
        sp = StratifiedKFold(N_SPLITS, shuffle=True,
                             random_state=SEED).split(X, y)
    else:
        sp = StratifiedGroupKFold(N_SPLITS, shuffle=True,
                                  random_state=SEED).split(X, y, groups)
    yt, yp = [], []
    for tr, te in sp:
        if len(np.unique(y[tr])) < 2:
            continue
        m = rf().fit(X[tr], y[tr])
        seen = np.isin(y[te], np.unique(y[tr]))
        if not seen.any():
            continue
        idx = np.where(te if te.dtype == bool else np.isin(
            np.arange(len(y)), te))[0]
        yp.append(m.predict(X[te][seen])); yt.append(y[te][seen])
    if not yt:
        return np.nan
    return accuracy_score(np.concatenate(yt), np.concatenate(yp))


def decode_night_out(X, y, slot):
    """Hold out an entire session. Only participants that still appear in the
    training partition are scored, so a missing participant cannot be counted
    as an error."""
    yt, yp = [], []
    for k in np.unique(slot):
        te = slot == k
        tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        seen = np.isin(y[te], np.unique(y[tr]))
        if not seen.any():
            continue
        m = rf().fit(X[tr], y[tr])
        Xte = X[te][seen]
        yp.append(m.predict(Xte)); yt.append(y[te][seen])
    if not yt:
        return np.nan
    return accuracy_score(np.concatenate(yt), np.concatenate(yp))


def main():
    d = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    fc = [c for c in d.columns if c not in META]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=fc)

    # oid encodes participant and session: S0021 -> participant 002, night 1
    night = d.oid.astype(str).str.extract(r"S(\d{3})(\d)")
    d = d.assign(pid=night[0].to_numpy(), night=night[1].to_numpy())
    d = d[d.pid.notna()]

    # session slot: 1st, 2nd, 3rd session of that participant, in order
    order = (d.drop_duplicates("oid")[["pid", "night", "oid"]]
             .sort_values(["pid", "night"]))
    order["slot"] = order.groupby("pid").cumcount()
    d = d.merge(order[["oid", "slot"]], on="oid", how="left")

    npn = d.groupby("pid").night.nunique()
    multi = set(npn[npn >= 2].index)

    log("=" * 78)
    log("DOES PARTICIPANT IDENTITY BEHAVE LIKE A STABLE TRAIT?")
    log("=" * 78)
    log(f"windows {len(d)} | participants {d.pid.nunique()} | "
        f"recordings {d.oid.nunique()}")
    log(f"participants with more than one session: {len(multi)} of "
        f"{d.pid.nunique()}")
    log(f"chance for identity: {1 / d.pid.nunique():.4f}")
    log("")

    dm = d[d.pid.isin(multi)].reset_index(drop=True)
    log(f"cross-session analysis restricted to the {len(multi)} "
        f"multi-session participants ({len(dm)} windows, "
        f"chance {1 / len(multi):.4f})")
    log("")

    fam = families(fc)
    y_all = d.pid.to_numpy()
    rec_all = d.oid.to_numpy()
    y_m = dm.pid.to_numpy()
    rec_m = dm.oid.to_numpy()
    slot_m = dm.slot.to_numpy()

    log(f"{'feature family':<24}{'n':>5}{'shuffled':>11}"
        f"{'rec-grouped':>14}{'session held out':>19}")
    log("-" * 78)

    rows = []
    for name in ["all features", "amplitude-dependent", "absolute power",
                 "log-variance", "amplitude-normalised", "relative power",
                 "spectral shape", "aperiodic slope", "Hjorth"]:
        cols = fam[name]
        if not cols:
            continue
        a = decode(d[cols].to_numpy(float), y_all, rec_all, "shuffled")
        b = decode(d[cols].to_numpy(float), y_all, rec_all, "grouped")
        c = decode_night_out(dm[cols].to_numpy(float), y_m, slot_m)
        log(f"{name:<24}{len(cols):>5}{a:>11.4f}{b:>14.4f}{c:>19.4f}")
        rows.append(dict(family=name, n_features=len(cols),
                         acc_shuffled=a, acc_recording_grouped=b,
                         acc_session_held_out=c,
                         chance_all=1 / d.pid.nunique(),
                         chance_multi=1 / len(multi)))

    r = pd.DataFrame(rows)
    r.round(4).to_csv(os.path.join(OUT, "identity_basis.csv"), index=False)

    log("")
    log("=" * 78)
    log("INTERPRETATION")
    log("=" * 78)
    allrow = r[r.family == "all features"].iloc[0]
    ch_m = 1 / len(multi)
    ratio = allrow.acc_session_held_out / ch_m
    log(f"all features: {allrow.acc_shuffled:.4f} shuffled -> "
        f"{allrow.acc_recording_grouped:.4f} recording-grouped -> "
        f"{allrow.acc_session_held_out:.4f} session held out")
    log(f"cross-session identity is {ratio:.1f} times chance "
        f"({ch_m:.4f})")
    log("")
    if allrow.acc_session_held_out > 0.5:
        log("Identity transfers strongly across recording sessions. That is")
        log("consistent with a stable individual trait and supports, without")
        log("proving, the anatomical reading.")
    elif ratio > 3:
        log("Identity transfers across sessions well above chance but far")
        log("below its within-session level. Both a stable trait component")
        log("and a session-specific component are present, and attribution")
        log("alone cannot apportion them. The manuscript should claim only")
        log("the part that is measured.")
    else:
        log("Identity barely transfers across sessions. The decodability")
        log("exploited under window-level partitioning is therefore largely")
        log("SESSION-specific -- impedance, placement and setup -- rather")
        log("than a stable anatomical trait. The empirical consequence for")
        log("this paper is unchanged, but the stated mechanism is wrong and")
        log("must be corrected rather than softened.")

    with open(os.path.join(OUT, "test38_identity_basis.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"\nWrote {OUT}\\identity_basis.csv")


if __name__ == "__main__":
    main()
