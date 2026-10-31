"""
Every number in the mechanism section (Tables 8 and 8b, Figure 3), gathered
from the output files that produced it, with its source, so the section can be
audited line by line. Nothing is computed here except means, SDs (ddof=0) and
ranges of stored per-run values.

Sources
  protocol_table.csv            51  random forest, three partitions, 10 assignments
  identity_rule_cv10.csv        83  identity rule and majority class, same folds
  identity_association.csv      83  window- and recording-level I/H, permutation p
  mechanism_refresh.csv         52  identity decoding, 10 assignments
  tononi_routes.csv             83b nearest-neighbour rates, 10 assignments
  identity_controls_balanced.csv 74 balanced control, window-shuffled forest
  identity_routes.csv           81  balanced control, recording-grouped forest
  identity_routes_rule.csv      81b identity rule on the balanced subsamples
  identity_controls_draws.csv   74  size-matched draws
  run_identity_controls.log     74  subgroup and all-34 rows (logged only)
"""
import os
import re

import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
rows = []


def put(q, v, src):
    rows.append(dict(quantity=q, value=round(float(v), 4), source=src))


def main():
    pt = pd.read_csv(os.path.join(OUT, "protocol_table.csv"))
    pt = pt[pt.target == "Dream valence"]
    for held, name in (("Random (window-shuffled)", "window-shuffled"),
                       ("Recording", "recording-grouped"),
                       ("Participant", "participant-separated")):
        r = pt[pt.held_out == held].iloc[0]
        for k in ("acc_mean", "acc_sd", "bal_mean", "bal_sd"):
            put(f"forest {name} {k}", r[k], "protocol_table.csv")

    ir = pd.read_csv(os.path.join(OUT, "identity_rule_cv10.csv"))
    ir = ir[ir.dataset == "DEED"]
    for (part, pred), g in ir.groupby(["partition", "predictor"]):
        assert g.seed.nunique() == 10
        put(f"{pred} {part} acc_mean", g.acc.mean(), "identity_rule_cv10.csv")
        put(f"{pred} {part} acc_sd", g.acc.std(ddof=0), "identity_rule_cv10.csv")
        put(f"{pred} {part} bal_mean", g.bal.mean(), "identity_rule_cv10.csv")
        put(f"{pred} {part} bal_sd", g.bal.std(ddof=0), "identity_rule_cv10.csv")

    a = pd.read_csv(os.path.join(OUT, "identity_association.csv"))
    for _, r in a.iterrows():
        for k in ("window_level_I_over_H", "recording_level_I_over_H",
                  "excess_over_permutation_mean", "p"):
            put(f"{r.dataset} {k}", r[k], "identity_association.csv")

    m = pd.read_csv(os.path.join(OUT, "mechanism_refresh.csv"))
    for q in ("identity window-shuffled", "identity recording-grouped"):
        r = m[(m.dataset == "DEED") & (m.quantity == q)].iloc[0]
        put(f"DEED {q} mean", r["mean"], "mechanism_refresh.csv")
        put(f"DEED {q} sd", r["sd"], "mechanism_refresh.csv")

    t = pd.read_csv(os.path.join(OUT, "tononi_routes.csv"))
    for ds, g in t.groupby("dataset"):
        put(f"{ds} nn same recording", g.nn_same_recording.mean(),
            "tononi_routes.csv")
        put(f"{ds} nn same participant", g.nn_same_participant.mean(),
            "tononi_routes.csv")

    b = pd.read_csv(os.path.join(OUT, "identity_controls_balanced.csv"))
    put("balanced window-shuffled forest mean", b.acc.mean(),
        "identity_controls_balanced.csv")
    put("balanced window-shuffled forest min", b.acc.min(),
        "identity_controls_balanced.csv")
    put("balanced window-shuffled forest max", b.acc.max(),
        "identity_controls_balanced.csv")
    put("balanced windows", b.windows.iloc[0], "identity_controls_balanced.csv")
    put("balanced window-level association", b.window_nmi.max(),
        "identity_controls_balanced.csv")
    r81 = pd.read_csv(os.path.join(OUT, "identity_routes.csv"))
    rg = r81[(r81.data == "balanced subsample")
             & (r81.partition == "recording-grouped")]
    assert len(rg) == 10
    put("balanced recording-grouped forest mean", rg.acc.mean(),
        "identity_routes.csv")
    put("balanced recording-grouped forest min", rg.acc.min(),
        "identity_routes.csv")
    put("balanced recording-grouped forest max", rg.acc.max(),
        "identity_routes.csv")
    rr = pd.read_csv(os.path.join(OUT, "identity_routes_rule.csv"))
    for part, g in rr.groupby("partition"):
        put(f"balanced {part} rule mean", g.id_acc.mean(),
            "identity_routes_rule.csv")

    d = pd.read_csv(os.path.join(OUT, "identity_controls_draws.csv"))
    for c in ("nmi", "acc", "bal", "id_acc"):
        put(f"draws {c} median", d[c].median(), "identity_controls_draws.csv")
        put(f"draws {c} q2.5", d[c].quantile(.025),
            "identity_controls_draws.csv")
        put(f"draws {c} q97.5", d[c].quantile(.975),
            "identity_controls_draws.csv")
    put("draws corr nmi acc", d[["nmi", "acc"]].corr().iloc[0, 1],
        "identity_controls_draws.csv")

    log = open(os.path.join(OUT, "run_identity_controls.log"),
               encoding="utf-8").read()
    for tag, pat in (("all34", r"all 34: NMI ([\d.]+) \(excess ([\d.]+), p "
                               r"([\d.]+)\) \| RF acc ([\d.]+) bal ([\d.]+) \| "
                               r"identity-CV acc ([\d.]+)"),
                     ("subgroup21", r"subgroup 21: NMI ([\d.]+) \(excess "
                                    r"([\d.]+), p ([\d.]+)\) \| RF acc "
                                    r"([\d.]+) bal ([\d.]+) \| identity-CV "
                                    r"acc ([\d.]+)")):
        v = re.search(pat, log).groups()
        for k, x in zip(("assoc", "excess", "p", "forest_acc", "forest_bal",
                         "rule_acc"), v):
            put(f"{tag} {k}", x, "run_identity_controls.log")
    for c, share in re.findall(r"  (nmi|acc|bal|id_acc)\s+draws median .*?"
                               r"share of draws at or below the subgroup "
                               r"([\d.]+)", log):
        put(f"draws share at or below subgroup {c}", share,
            "run_identity_controls.log")

    # derived quantities quoted in the text: differences of the values above
    v = {r["quantity"]: r["value"] for r in rows}
    ws, rg, ps = (v["forest window-shuffled bal_mean"],
                  v["forest recording-grouped bal_mean"],
                  v["forest participant-separated bal_mean"])
    put("derived DEED inflation bal (ws - ps)", ws - ps, "derived")
    put("derived DEED recording route bal (ws - rg)", ws - rg, "derived")
    put("derived DEED participant route bal (rg - ps)", rg - ps, "derived")
    t = m[m.dataset == "Tononi"].set_index("quantity")
    tws, trg, tps = (t.loc["acc window-shuffled", "bal_mean"],
                     t.loc["acc recording-grouped", "bal_mean"],
                     t.loc["acc participant-wise", "bal_mean"])
    put("derived Tononi inflation bal (ws - ps)", tws - tps, "derived")
    put("derived Tononi recording route bal (ws - rg)", tws - trg, "derived")
    put("derived Tononi inflation acc (ws - ps)",
        t.loc["acc window-shuffled", "mean"] - t.loc["acc participant-wise", "mean"],
        "derived")
    put("derived DEED inflation acc (ws - ps)",
        v["forest window-shuffled acc_mean"]
        - v["forest participant-separated acc_mean"], "derived")
    bs = pd.read_csv(os.path.join(OUT, "protocol_factorial_effects_by_seed.csv"))
    sel = bs[(bs.effect == "selection before vs inside")
             & (bs.a == "window/before/none")].d_bal.mean()
    rows.append(dict(quantity="derived factorial selection window mean d_bal",
                     value=round(float(sel), 6), source="derived"))
    st = pd.read_csv(os.path.join(OUT, "leakage_stack.csv")).set_index("rung")
    put("derived ladder inflation acc points (rung 5 - rung 0) x100",
        100 * (st.loc[5, "leaky_acc"] - st.loc[0, "leaky_acc"]), "derived")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "fingerprint_table.csv"), index=False)
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
