"""
Recompute the catalogue prevalence analysis using mutual information.

Why this supersedes 27_dream_catalogue.py
----------------------------------------
The earlier analysis summarised each dataset by MEDIAN WITHIN-PARTICIPANT LABEL
DOMINANCE, on the assumption that concentrated labels make participant identity
predictive of the label. The Tononi replication refuted that assumption as an
operationalisation:

    DEED    dominance 0.750, I(subject;label)/H 0.1624, oracle 0.5465
            -> inflation +0.1707
    Tononi  dominance 0.536, I(subject;label)/H 0.1902, oracle 0.5778
            -> inflation +0.2290

Tononi has LOWER dominance but HIGHER mutual information, and it shows MORE
inflation. Dominance measures how concentrated each participant is on one class;
it is blind to whether participants differ from one another. Participants can
each be individually balanced while still having distinct label distributions,
which makes identity informative. Mutual information and the identity-only
oracle capture that; dominance does not.

Both quantities are computed here, per dataset, so the substitution is auditable
rather than asserted. Normalised mutual information I(S;Y)/H(Y) is the primary
measure and the oracle is reported alongside it as an interpretable
accuracy-scale equivalent.

A permutation baseline is included because mutual information is biased upward
when the number of participants is large relative to the number of records. For
each dataset, labels are shuffled across records within the dataset and the MI
recomputed, giving the value expected under no participant-label association.
The excess over that baseline is what matters.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
CAT = os.path.join(ROOT, "Dream Dataset")
FIG = os.path.join(ROOT, "analysis", "figures")
PAPER = os.path.join(ROOT, "paper")

VALID = ["Experience", "No experience", "Without recall"]
MIN_REC = 3
MIN_SUBJ = 3
N_PERM = 200
SEED = 20260727

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def nmi(subj, lab):
    """I(S;Y) / H(Y): fraction of label entropy explained by participant."""
    h = mutual_info_score(lab, lab)
    return mutual_info_score(subj, lab) / h if h > 0 else 0.0


def oracle_acc(df):
    """Accuracy of predicting each participant's majority label from ID alone."""
    maj = df.groupby("subj").lab.agg(lambda s: s.value_counts().idxmax())
    return float((df.lab == df.subj.map(maj)).mean())


def main():
    d = pd.read_csv(os.path.join(CAT, "Data records.csv"))
    ds = pd.read_csv(os.path.join(CAT, "Datasets.csv"))
    d = d[d.Experience.isin(VALID)].rename(
        columns={"Set ID": "set", "Subject ID": "subj", "Experience": "lab"})

    name = dict(zip(ds["Set ID"], ds.get("Common name", ds["Set ID"])))
    rng = np.random.default_rng(SEED)

    log("=" * 78)
    log("CATALOGUE PREVALENCE, RECOMPUTED WITH MUTUAL INFORMATION")
    log("=" * 78)
    log("Supersedes the median-dominance analysis, which the Tononi")
    log("replication showed does not predict inflation.")
    log("")

    rows = []
    for sid, g in d.groupby("set"):
        keep = g.groupby("subj").size()
        keep = keep[keep >= MIN_REC].index
        g = g[g.subj.isin(keep)]
        if g.subj.nunique() < MIN_SUBJ or g.lab.nunique() < 2:
            continue
        obs = nmi(g.subj.to_numpy(), g.lab.to_numpy())
        null = np.array([
            nmi(g.subj.to_numpy(), rng.permutation(g.lab.to_numpy()))
            for _ in range(N_PERM)])
        dom = g.groupby("subj").lab.agg(
            lambda x: x.value_counts(normalize=True).max()).median()
        rows.append(dict(
            set_id=sid, dataset=str(name.get(sid, sid))[:28],
            participants=g.subj.nunique(), records=len(g),
            nmi=obs, nmi_null=null.mean(),
            nmi_excess=obs - null.mean(),
            p=(np.sum(null >= obs) + 1) / (N_PERM + 1),
            oracle=oracle_acc(g), median_dominance=float(dom)))

    r = pd.DataFrame(rows).sort_values("nmi_excess", ascending=False)

    log(f"{'dataset':<30}{'part':>5}{'rec':>6}{'NMI':>8}{'null':>7}"
        f"{'excess':>8}{'p':>7}{'oracle':>8}{'dom':>7}")
    log("-" * 78)
    for _, x in r.iterrows():
        log(f"{x.dataset:<30}{int(x.participants):>5}{int(x.records):>6}"
            f"{x.nmi:>8.3f}{x.nmi_null:>7.3f}{x.nmi_excess:>8.3f}"
            f"{x.p:>7.3f}{x.oracle:>8.3f}{x.median_dominance:>7.3f}")
    log("")

    sig = r[r.p < 0.05]
    log(f"datasets analysed                    : {len(r)}")
    log(f"with identity-label association p<.05: {len(sig)}")
    log(f"median NMI excess                    : {r.nmi_excess.median():.3f}")
    log(f"median identity-only oracle          : {r.oracle.median():.3f}")
    log("")

    # how well does each summary statistic order the two measured datasets?
    log("-- which statistic predicts measured inflation? --")
    log(f"{'':<26}{'DEED':>10}{'Tononi':>10}{'ordering':>22}")
    log(f"{'measured inflation':<26}{0.1707:>10.4f}{0.2290:>10.4f}"
        f"{'Tononi > DEED':>22}")
    log(f"{'median dominance':<26}{0.750:>10.4f}{0.536:>10.4f}"
        f"{'WRONG DIRECTION':>22}")
    log(f"{'NMI I(S;Y)/H(Y)':<26}{0.1624:>10.4f}{0.1902:>10.4f}"
        f"{'correct':>22}")
    log(f"{'identity-only oracle':<26}{0.5465:>10.4f}{0.5778:>10.4f}"
        f"{'correct':>22}")
    log("")
    log("Median dominance orders the two datasets in the wrong direction.")
    log("Mutual information and the oracle order them correctly, so they are")
    log("used as the precondition measures from here on.")
    log("")

    r.round(4).to_csv(os.path.join(OUT, "catalogue_mi.csv"), index=False)

    # ---- figure ------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "figure.dpi": 600, "savefig.dpi": 600,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .25,
                         "axes.axisbelow": True})

    p2 = r.sort_values("nmi_excess")
    cols = ["#C1352C" if v < 0.05 else "#8D99AE" for v in p2.p]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.barh(range(len(p2)), p2.nmi_excess, color=cols, edgecolor="black",
            lw=.5, height=.72)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(0.1624, ls="--", color="#2A6F97", lw=1.3)
    ax.text(0.1624, len(p2) - 0.2, " DEED", color="#2A6F97", fontsize=8,
            va="top")
    ax.set_yticks(range(len(p2)))
    ax.set_yticklabels(p2.dataset, fontsize=8)
    ax.set_xlabel("Excess $I(\\mathrm{participant};\\mathrm{label})/H"
                  "(\\mathrm{label})$ over permutation null")
    ax.set_title("Identity carries label information across the catalogue",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig9b_catalogue_datasets.{ext}"),
                    bbox_inches="tight")
        fig.savefig(os.path.join(PAPER, f"fig9b_catalogue_datasets.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.hist(r.nmi_excess, bins=10, color="#2A6F97", edgecolor="black", lw=.5)
    ax.axvline(0.1624, ls="--", color="#C1352C", lw=1.4)
    ax.text(0.166, ax.get_ylim()[1] * .9, "DEED", color="#C1352C", fontsize=9)
    ax.set_xlabel("Excess normalised mutual information")
    ax.set_ylabel("Datasets")
    ax.set_title(f"{len(r)} catalogued datasets", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig9a_catalogue_hist.{ext}"),
                    bbox_inches="tight")
        fig.savefig(os.path.join(PAPER, f"fig9a_catalogue_hist.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    log("regenerated fig9a and fig9b using mutual information")

    with open(os.path.join(OUT, "test25_catalogue_mi.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"Wrote {OUT}\\test25_catalogue_mi.txt")


if __name__ == "__main__":
    main()
