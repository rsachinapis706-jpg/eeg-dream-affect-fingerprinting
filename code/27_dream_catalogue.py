"""
How widely do the preconditions for the fingerprinting mechanism hold?

The mechanism identified in this work requires two ingredients:

  (i)  participant identity must be recoverable from the signal. This is a
       general property of EEG and is measured directly on DEED (0.8037).
  (ii) the label distribution must differ between participants, so that
       identity is predictive of the label.

Ingredient (ii) is a property of study design rather than of EEG, so it need not
generalise. The DREAM database catalogue (Wong et al., 2023) records, for 3,191
awakenings across 20 contributed datasets and 263 participants, both the
participant identifier and the reported experience category. That is exactly
what is required to measure ingredient (ii) across the field -- no signals are
needed.

This script quantifies, per dataset:
  * how often a participant contributes more than one record, which is what
    makes window- or record-level partitioning able to recognise them
  * how concentrated each participant's labels are

Only the catalogue metadata is used. No claim is made about the decodability of
any dataset other than DEED.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
CAT = os.path.join(ROOT, "Dream Dataset")
FIG = os.path.join(ROOT, "analysis", "figures")
PAPER = os.path.join(ROOT, "paper")

VALID = ["Experience", "No experience", "Without recall"]
MIN_REC = 3

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def main():
    d = pd.read_csv(os.path.join(CAT, "Data records.csv"))
    ds = pd.read_csv(os.path.join(CAT, "Datasets.csv"))
    d = d[d.Experience.isin(VALID)]

    log("=" * 74)
    log("PREVALENCE OF THE MECHANISM'S PRECONDITIONS ACROSS THE DREAM DATABASE")
    log("=" * 74)
    log(f"records {len(d)} | datasets {d['Set ID'].nunique()} | "
        f"participants {d['Subject ID'].nunique()}")
    log("")

    # ---- (a) do participants contribute multiple records? -------------
    per_sub = d.groupby(["Set ID", "Subject ID"]).size()
    log("-- Repeated records per participant --")
    log(f"median records per participant : {int(per_sub.median())}")
    log(f"maximum                        : {int(per_sub.max())}")
    log(f"participants with >1 record    : {per_sub.gt(1).mean():.3f}")
    log("")
    log("Where a participant contributes more than one record, a partition")
    log("formed over windows or records can place that participant on both")
    log("sides, which is the condition under which identity becomes usable.")
    log("")

    # ---- (b) how concentrated are each participant's labels? ----------
    rows = []
    for (sid, subj), g in d.groupby(["Set ID", "Subject ID"]):
        if len(g) < MIN_REC:
            continue
        share = g.Experience.value_counts(normalize=True)
        rows.append(dict(set_id=sid, subject=subj, n=len(g),
                         dominance=share.max(),
                         n_classes=g.Experience.nunique()))
    r = pd.DataFrame(rows)

    log(f"-- Label concentration (participants with >= {MIN_REC} records) --")
    log(f"participants analysed          : {len(r)}")
    log(f"median majority-class share    : {r.dominance.median():.3f}")
    log(f"share of participants >70\% one class : {r.dominance.gt(0.7).mean():.3f}")
    log(f"share of participants at 100\% one class: {r.dominance.eq(1.0).mean():.3f}")
    log(f"participants with only one class observed: "
        f"{int(r.n_classes.eq(1).sum())} of {len(r)}")
    log("")

    # ---- per dataset --------------------------------------------------
    name = dict(zip(ds["Set ID"], ds.get("Common name", ds["Set ID"])))
    per = r.groupby("set_id").agg(
        participants=("subject", "size"),
        median_records=("n", "median"),
        median_dominance=("dominance", "median"),
        frac_over_70=("dominance", lambda x: (x > 0.7).mean()))
    per["dataset"] = [str(name.get(i, i))[:28] for i in per.index]
    per = per.sort_values("median_dominance", ascending=False)

    log("-- Per dataset --")
    log(f"{'dataset':<30}{'part.':>7}{'med.rec':>9}{'med.dom':>9}{'>70%':>8}")
    log("-" * 74)
    for _, x in per.iterrows():
        log(f"{x.dataset:<30}{int(x.participants):>7}{x.median_records:>9.0f}"
            f"{x.median_dominance:>9.3f}{x.frac_over_70:>8.2f}")
    log("")
    n_hi = int((per.median_dominance > 0.7).sum())
    log(f"datasets whose median participant exceeds 70% in one class: "
        f"{n_hi} of {len(per)}")
    log("")
    log("Interpretation: ingredient (ii) of the mechanism is not specific to")
    log("DEED. In most catalogued dream datasets the typical participant's")
    log("reports are concentrated in a single category, so participant identity")
    log("would be predictive of the label wherever identity is recoverable.")
    log("This does not establish that those datasets yield inflated results,")
    log("which would require access to the signals and a replication of each")
    log("study's protocol. It establishes that the precondition is common.")

    per.round(4).to_csv(os.path.join(OUT, "dream_catalogue.csv"))
    with open(os.path.join(OUT, "test22_dream_catalogue.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    # ---- figure -------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "figure.dpi": 600, "savefig.dpi": 600,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .25,
                         "axes.axisbelow": True})
    # Two panels drawn as separate files. Combining them forced the long
    # dataset names of the right panel to overrun the left, and the figure-level
    # title to collide with the panel titles.
    fig, a1 = plt.subplots(figsize=(4.6, 3.4))
    a1.hist(r.dominance, bins=np.linspace(0.3, 1.0, 15), color="#2A6F97",
            edgecolor="black", lw=.5)
    a1.axvline(0.7, ls="--", color="#C1352C", lw=1.4)
    a1.text(0.715, a1.get_ylim()[1] * .90, "70\\%", color="#C1352C",
            fontsize=9)
    a1.set_xlabel("Majority-class share within participant")
    a1.set_ylabel("Participants")
    a1.set_title(f"{len(r)} participants, {per.shape[0]} datasets",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig9a_catalogue_hist.{ext}"),
                    bbox_inches="tight")
        fig.savefig(os.path.join(PAPER, f"fig9a_catalogue_hist.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)

    p2 = per.sort_values("median_dominance")
    cols = ["#C1352C" if v > 0.7 else "#8D99AE" for v in p2.median_dominance]
    fig, a2 = plt.subplots(figsize=(5.4, 3.6))
    a2.barh(range(len(p2)), p2.median_dominance, color=cols,
            edgecolor="black", lw=.5, height=.72)
    a2.axvline(0.7, ls="--", color="#C1352C", lw=1.4)
    a2.set_yticks(range(len(p2)))
    a2.set_yticklabels(p2.dataset, fontsize=8)
    a2.set_xlabel("Median majority-class share")
    a2.set_xlim(0, 1.05)
    a2.set_ylim(-0.6, len(p2) - 0.4)
    a2.set_title("Per dataset", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig9b_catalogue_datasets.{ext}"),
                    bbox_inches="tight")
        fig.savefig(os.path.join(PAPER, f"fig9b_catalogue_datasets.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    log("wrote fig9a_catalogue_hist and fig9b_catalogue_datasets")


if __name__ == "__main__":
    main()
