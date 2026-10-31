"""
Participant-label association across the DREAM catalogue, recomputed so the
numbers can be audited (Reviewer 1, comments 2 and 11).

What this corrects in 27_dream_catalogue.py and 32_catalogue_mi.py
  * Participant count. "263 participants" was d['Subject ID'].nunique() over
    all datasets. DREAM subject identifiers are local to each dataset, so an ID
    such as "1" occurring in two datasets was counted once. The unit here is the
    dataset-specific participant entry, (Set ID, Subject ID). Whether the same
    person recurs across datasets cannot be determined from the catalogue's
    identifiers, and the manuscript now says so.
  * Permutation resolution. 200 permutations floors p at 0.005, which cannot
    pass a Holm correction over twelve datasets (0.05 / 12 = 0.0042) whatever
    the data. 10,000 are used here, with a per-dataset seed so each result is
    reproducible independently of the others.
  * Multiplicity. Holm (family-wise) and Benjamini-Hochberg (false discovery
    rate) across the datasets analysed are reported beside each raw p.
  * The DEED comparison. The earlier figure set DEED's RAW normalised mutual
    information beside the catalogue's permutation-ADJUSTED excess. DEED is
    now passed through the identical procedure, at the same unit -- one label
    per awakening -- and the same inclusion rule.

Inclusion rule, unchanged: the three standard experience categories; at least
three records per participant; at least three such participants; at least two
label classes present.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
CAT = os.path.join(ROOT, "Dream Dataset")

VALID = ["Experience", "No experience", "Without recall"]
MIN_REC, MIN_SUBJ = 3, 3
N_PERM = 10_000
SEED = 20260930
VALENCE = {"E1": "neg", "E2": "neg", "E3": "neu", "E4": "pos", "E5": "pos"}

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def nmi(subj, lab):
    """I(S;Y) / H(Y): share of label entropy accounted for by participant."""
    h = mutual_info_score(lab, lab)
    return mutual_info_score(subj, lab) / h if h > 0 else 0.0


def oracle(subj, lab):
    """Descriptive only: each participant's majority label, from all records."""
    df = pd.DataFrame(dict(s=subj, y=lab))
    maj = df.groupby("s").y.agg(lambda v: v.value_counts().idxmax())
    return float((df.y == df.s.map(maj)).mean())


def test(subj, lab, seed):
    rng = np.random.default_rng(seed)
    obs = nmi(subj, lab)
    null = np.array([nmi(subj, rng.permutation(lab)) for _ in range(N_PERM)])
    return dict(nmi=obs, nmi_null=null.mean(), nmi_excess=obs - null.mean(),
                null_p95=float(np.percentile(null, 95)),
                p=(1 + np.sum(null >= obs)) / (1 + N_PERM),
                oracle=oracle(subj, lab))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, k in enumerate(o):
        run = max(run, min(1.0, (len(p) - i) * p[k]))
        adj[k] = run
    return adj


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)[::-1]
    adj = np.empty_like(p)
    run = 1.0
    for i, k in enumerate(o):
        run = min(run, p[k] * n / (n - i))
        adj[k] = run
    return adj


def include(g):
    """Apply the inclusion rule to one dataset's records; return the kept
    records and the reason if the dataset is excluded."""
    counts = g.groupby("subj").size()
    g2 = g[g.subj.isin(counts[counts >= MIN_REC].index)]
    if g2.subj.nunique() < MIN_SUBJ:
        return g2, f"fewer than {MIN_SUBJ} participants with >= {MIN_REC} records"
    if g2.lab.nunique() < 2:
        return g2, "fewer than two label classes"
    return g2, ""


def main():
    raw = pd.read_csv(os.path.join(CAT, "Data records.csv"))
    ds = pd.read_csv(os.path.join(CAT, "Datasets.csv"))
    name = dict(zip(ds["Set ID"], ds.get("Common name", ds["Set ID"])))

    log("=" * 78)
    log("DREAM CATALOGUE: PARTICIPANT-LABEL ASSOCIATION, AUDITABLE")
    log("=" * 78)
    ids_str = raw["Subject ID"].nunique()
    entries = raw[["Set ID", "Subject ID"]].drop_duplicates().shape[0]
    log(f"records                                : {len(raw)}")
    log(f"datasets                               : {raw['Set ID'].nunique()}")
    log(f"distinct Subject ID strings (old count): {ids_str}")
    log(f"dataset-specific participant entries   : {entries}")
    log("Subject IDs are local to each dataset, so the first figure merges")
    log("different people who share an ID string. The second is the correct")
    log("count of participant entries. Recurrence of the same individual")
    log("across datasets cannot be established from these identifiers.")
    log("")

    d = raw[raw.Experience.isin(VALID)].rename(
        columns={"Set ID": "set", "Subject ID": "subj", "Experience": "lab"})
    d["subj"] = d["set"].astype(str) + ":" + d["subj"].astype(str)

    inc, rows = [], []
    for sid in sorted(raw["Set ID"].unique()):
        g_all = raw[raw["Set ID"] == sid]
        g = d[d["set"] == sid]
        kept, why = include(g) if len(g) else (g, "no records in the three "
                                                  "standard categories")
        inc.append(dict(set_id=sid, dataset=str(name.get(sid, sid)),
                        records_all=len(g_all),
                        participants_all=g_all["Subject ID"].nunique(),
                        records_valid=len(g),
                        participants_valid=g.subj.nunique(),
                        records_kept=len(kept),
                        participants_kept=kept.subj.nunique(),
                        included=(why == ""), reason=why))
        if why:
            continue
        r = test(kept.subj.to_numpy(), kept.lab.to_numpy(), SEED + int(sid))
        rows.append(dict(set_id=sid, dataset=str(name.get(sid, sid)),
                         participants=kept.subj.nunique(),
                         records=len(kept), **r))

    inc = pd.DataFrame(inc)
    inc.to_csv(os.path.join(OUT, "catalogue_inclusion.csv"), index=False)
    log("-- inclusion accounting, all datasets --")
    with pd.option_context("display.width", 200, "display.max_colwidth", 30):
        log(inc.drop(columns="dataset").to_string(index=False))
    log("")
    log(f"datasets included          : {int(inc.included.sum())} of {len(inc)}")
    log(f"participant entries        : {int(inc.participants_kept[inc.included].sum())}")
    log(f"records                    : {int(inc.records_kept[inc.included].sum())}")
    log("")

    r = pd.DataFrame(rows)
    r["p_holm"] = holm(r.p)
    r["p_bh"] = bh(r.p)

    # ---- DEED through the identical procedure, one label per awakening ----
    f = pd.read_parquet(os.path.join(OUT, "features_epoch_filtered.parquet"))
    f = f[f.E.isin(VALENCE)]
    rec = f.drop_duplicates("segment_id")
    g = pd.DataFrame(dict(subj=rec.subject.astype(str),
                          lab=rec.E.map(VALENCE)))
    kept, why = include(g)
    deed = dict(set_id="DEED", dataset="DEED (valence, per awakening)",
                participants=kept.subj.nunique(), records=len(kept),
                **test(kept.subj.to_numpy(), kept.lab.to_numpy(), SEED))
    # window level, for reference against the number used in Section IV-H
    wl = test(f.subject.astype(str).to_numpy(), f.E.map(VALENCE).to_numpy(),
              SEED + 1)
    log("-- DEED, same procedure --")
    log(f"awakening level: {deed['participants']} participants, "
        f"{deed['records']} records, NMI {deed['nmi']:.4f}, null "
        f"{deed['nmi_null']:.4f}, excess {deed['nmi_excess']:.4f}, "
        f"p {deed['p']:.4f}, oracle {deed['oracle']:.4f}"
        + (f"  [excluded: {why}]" if why else ""))
    log(f"window level   : NMI {wl['nmi']:.4f}, null {wl['nmi_null']:.4f}, "
        f"excess {wl['nmi_excess']:.4f}, p {wl['p']:.4f}, "
        f"oracle {wl['oracle']:.4f}   (reference only)")
    log("")

    out = pd.concat([r, pd.DataFrame([deed])], ignore_index=True)
    out = out.sort_values("nmi_excess", ascending=False)
    out.round(5).to_csv(os.path.join(OUT, "catalogue_audit.csv"), index=False)

    log(f"-- per dataset, {N_PERM} permutations each --")
    log(f"{'dataset':<32}{'part':>5}{'rec':>6}{'NMI':>7}{'null':>7}"
        f"{'excess':>8}{'p':>9}{'Holm':>8}{'BH':>8}")
    for _, x in out.iterrows():
        hp = "" if pd.isna(x.get("p_holm")) else f"{x.p_holm:>8.4f}"
        bp = "" if pd.isna(x.get("p_bh")) else f"{x.p_bh:>8.4f}"
        log(f"{str(x.dataset)[:31]:<32}{int(x.participants):>5}"
            f"{int(x.records):>6}{x.nmi:>7.3f}{x.nmi_null:>7.3f}"
            f"{x.nmi_excess:>8.3f}{x.p:>9.4f}{hp}{bp}")
    log("")
    log(f"catalogue datasets analysed       : {len(r)}")
    log(f"raw p < 0.05                      : {int((r.p < 0.05).sum())}")
    log(f"Holm-adjusted p < 0.05            : {int((r.p_holm < 0.05).sum())}")
    log(f"Benjamini-Hochberg q < 0.05       : {int((r.p_bh < 0.05).sum())}")
    log("A non-significant dataset is not evidence that the mechanism cannot")
    log("operate there; with few participants or records the test has little")
    log("power, and window-overlap leakage does not require this association.")

    with open(os.path.join(OUT, "run_catalogue_audit.log"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
