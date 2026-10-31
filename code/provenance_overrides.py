"""
Provenance decisions made by reading the manuscript (round 3, Reviewer 2,
comment 7), applied by 132_provenance.py after its automatic matching.

Each entry names a number by its line in accesstemp.tex, its printed text and
its occurrence among equal numbers on that line (0 = first), and gives one of

  key      the result cell it is: file, a pandas query selecting the row and
           the column; `scale` converts a share to a printed percentage
  derived  a number computed in the text from result cells (a difference, a
           ratio, a mean, a bound): `check` is a Python expression over the
           result files that must reproduce the printed value
  log      a figure printed by an analysis script in its log: the log file
           and the line it appears on
  status   not a result (a setting, threshold, quoted figure, ...): the
           status and a note

132 recomputes every key and check and stops if one does not reproduce the
printed value at its printed precision, or if the line does not hold the
number. Nothing here is taken on trust.

The automatic matching assigns a number to the cell whose file, row and
column share most words with the sentence. Reading all of them showed where
that goes wrong: a value that also occurs, by coincidence, in an unrelated
file; a difference or ratio computed in the text, which no file holds; and
figures written to a log rather than a table. These entries correct those.
"""
import glob
import io
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
TEX = r"E:\Dream_Analysis\accesstemp.tex"

_cache = {}


def D(f):
    """A result file as a data frame."""
    if f not in _cache:
        _cache[f] = pd.read_csv(os.path.join(OUT, f))
    return _cache[f]


def cell(f, query, col):
    """The single cell of file f selected by query, column col."""
    r = D(f).query(query)
    assert len(r) >= 1, (f, query)
    v = r[col].astype(float).unique()
    assert len(v) == 1, (f, query, col, v)
    return float(v[0])


# ---- row selections used repeatedly
PT = "protocol_table.csv"
P_PART = "target == 'Dream valence' and held_out == 'Participant'"
P_REC = "target == 'Dream valence' and held_out == 'Recording'"
P_WIN = "target == 'Dream valence' and held_out == 'Random (window-shuffled)'"
PRS = "primary_random_summary.csv"
CT = "config_table.csv"
LS = "leakage_stack.csv"
LB = "config == 'LogReg, scaled, balanced'"
LU = "config == 'LogReg, scaled, unweighted'"
RFH = "config == 'RF, unweighted (headline)'"
RFB = "config == 'RF, scaled, balanced'"
LSS = "label_sensitivity_summary.csv"
L60 = "variant == 'last 60 s before awakening'"
WEA = "weak_effect_per_assignment.csv"
PW = "power_revised.csv"
CDS = "cbramod_diag_summary.csv"
SEEDS = "cbramod_seeds.csv"
NOISE = "pcd_extra_noise.csv"
LEAKT = "pcd_leaky_table.csv"
RC = "route_contrasts.csv"
JK = "weak_effect_jackknife.csv"


def K(line, value, file, query, column, occ=0, scale=1):
    return dict(line=line, value=value, occ=occ, kind="key", file=file,
                query=query, column=column, scale=scale)


def DV(line, value, check, note, occ=0):
    return dict(line=line, value=value, occ=occ, kind="derived", check=check,
                note=note)


def LG(line, value, file, text, occ=0):
    return dict(line=line, value=value, occ=occ, kind="log", file=file,
                text=text)


def ST(line, value, status, note, occ=0):
    return dict(line=line, value=value, occ=occ, kind="status",
                status=status, note=note)


def cbramod_constant():
    """The per-seed medians 142_cbramod_constant.py prints."""
    rows = []
    for f in sorted(glob.glob(os.path.join(
            OUT, "cbramod_diag_revised_2026*_folds.csv"))):
        d = pd.read_csv(f, dtype={"held_out": str})
        c, o = d[d.constant_pred], d[~d.constant_pred]
        rows.append(dict(ep_o=o.chosen_epoch.median(),
                         va_c=c.best_val_bal.median(),
                         va_o=o.best_val_bal.median()))
    return pd.DataFrame(rows)


def seeds(config):
    return D(SEEDS).query(f"config == '{config}'").pooled_bal


DIFF_UV = ("cell('cbramod_expanded.csv', \"model == 'cbramod'\", 'balanced')"
           " - cell('cbramod_expanded.csv', \"model == 'spectral_logreg'\","
           " 'balanced')")
DIFF_100 = ("cell('cbramod_expanded_uv100.csv', \"model == 'cbramod'\","
            " 'balanced') - cell('cbramod_expanded.csv',"
            " \"model == 'spectral_logreg'\", 'balanced')")
SPEAR = "stats.spearmanr(D(NOISE).coverage, D(NOISE).balanced)[0]"
PEARS = "stats.pearsonr(D(NOISE).coverage, D(NOISE).balanced)[0]"
HOLM60 = (f"4 * D(LSS).query(\"{L60}\").p_within.min()")
WITHIN_UV = "within_effects_uv100.csv"
POWER = "within_paired_power.csv"

OVERRIDES = [
    # ---------------------------------------------------------- abstract
    K(83, "0.3224", PT, P_PART, "bal_mean"),
    K(83, "0.3393", PRS, LB, "observed_mean"),
    K(83, "0.3294", SEEDS, "config == 'revised' and seed == 20260728",
      "pooled_bal"),
    K(83, "0.3517", SEEDS, "config == 'revised' and seed == 20260727",
      "pooled_bal"),
    # ---------------------------------------------------------- introduction
    K(111, "0.4208", LS, "rung == 0", "leaky_acc"),
    K(112, "0.8040", "mechanism_refresh.csv",
      "dataset == 'DEED' and quantity == 'identity window-shuffled'", "mean"),
    K(116, "0.3001", "pcd_leaky_summary.csv",
      "partition == 'window-level' and rule == 'Max-softmax>0.5'",
      "coverage_mean"),
    K(116, "0.0526", "pcd_leaky_summary.csv",
      "partition == 'window-level' and rule == 'PCD alpha=0.05'",
      "coverage_mean"),
    ST(136, "0.85", "quoted from cited work", "Moctezuma et al."),
    ST(146, "86.2", "quoted from cited work", "Xu et al."),
    K(241, "0.0001", NOISE, "eta == 1.0", "msp_coverage"),
    K(241, "0.0130", NOISE, "eta == 0.8", "msp_coverage"),
    # ---------------------------------------------------------- data
    DV(396, "200.00",
       "D('deed_alignment.csv').ratio.quantile([.25, .5, .75]).round(2)"
       ".nunique() == 1 and D('deed_alignment.csv').ratio.median()",
       "25th, 50th and 75th percentiles of the samples-per-second ratio in "
       "deed_alignment.csv, all equal"),
    K(437, "0.3110", "deed_accounting.csv",
      "quantity == 'recordings pos share'", "value"),
    ST(441, "0.70", "setting or threshold", "share threshold"),
    ST(453, "0.05", "setting or threshold", "significance level"),
    LG(461, "0.4835", "test18_expanded_control.txt",
       "majority baseline 0.4835"),
    K(488, "0.0017", PT, P_PART, "bal_sd"),
    K(488, "0.3329", CT, RFH, "free_null_mean_single"),
    ST(490, "1.4", "setting", "scikit-learn version"),
    ST(490, "1.9", "setting", "scikit-learn version"),
    K(505, "0.0057", PT, P_REC, "bal_sd"),
    K(506, "0.0017", PT, P_PART, "bal_sd"),
    # ---------------------------------------------------------- factorial
    K(544, "0.1084", "protocol_factorial_effects_full.csv",
      "effect == 'window vs participant partition' and "
      "a == 'window/inside/none'", "d_bal"),
    K(544, "0.3295", "protocol_factorial_effects_full.csv",
      "effect == 'window vs participant partition' and "
      "a == 'window/none/before'", "d_bal"),
    K(544, "0.0082", "protocol_factorial_effects_full.csv",
      "effect == 'selection before vs inside' and "
      "a == 'participant/before/before'", "d_bal"),
    K(546, "0.7282", "protocol_factorial_cells_full.csv",
      "partition == 'window' and selection == 'none' and "
      "oversampling == 'before'", "acc_all_mean"),
    # ---------------------------------------------------------- primary
    K(556, "0.3393", PRS, LB, "observed_mean"),
    K(556, "1.0000", PRS, LB, "within_p_holm", occ=0),
    K(556, "1.0000", PRS, LB, "free_p_holm", occ=1),
    K(556, "1.0000", PRS, RFH, "within_p", occ=2),
    DV(556, "0.0303", f"1 / (1 + cell(PRS, \"{LB}\", 'within_maxT_B'))",
       "smallest attainable max-statistic p with the shared relabellings"),
    K(573, "0.3222", PRS, RFH, "observed_mean"),
    K(574, "0.3242", PRS, RFB, "observed_mean"),
    K(575, "0.3238", PRS, LU, "observed_mean"),
    K(576, "0.3393", PRS, LB, "observed_mean"),
    # ---------------------------------------------------------- labels
    K(589, "0.0170", LSS, f"{L60} and {LB} and statistic == 'window'",
      "p_within"),
    DV(589, "0.0680", HOLM60,
       "smallest Holm-adjusted p across the four configurations: 4 x the "
       "smallest p (153_label_text.py)"),
    ST(593, "0.05", "setting or threshold", "significance level"),
    DV(593, "0.0680", HOLM60, "as line 589"),
    K(607, "0.8696", LSS, "variant == 'corroborated recordings' and "
      f"{RFB} and statistic == 'window'", "p_within", occ=0),
    K(612, "0.3398", LSS, f"{L60} and {RFB} and statistic == 'window'",
      "observed"),
    # ---------------------------------------------------------- splitter
    K(624, "0.4092", "statistical_foundation.csv",
      f"part == 'uncertainty' and {LB}", "bal_hi"),
    K(624, "0.0320", CT, LB, "free_p_single", occ=0),
    K(624, "0.0320", CT, LB, "within_p_single", occ=1),
    K(624, "0.3537", CT, LU, "observed_single"),
    K(624, "0.0849", CT, LU, "within_p_single"),
    K(624, "0.0899", CT, LU, "free_p_single"),
    K(632, "0.3783", WEA, f"{LB} and null == 'within' and seed == 20260727",
      "observed"),
    K(632, "0.3754", WEA, f"{LB} and null == 'within' and seed == 20260729",
      "observed"),
    K(632, "0.0396", CT, LB, "free_p_mean10", occ=0),
    K(632, "0.0495", CT, LB, "within_p_mean10", occ=0),
    K(632, "0.0396", WEA, f"{LB} and null == 'free' and seed == 20260727",
      "p", occ=1),
    K(632, "0.0495", WEA, f"{LB} and null == 'free' and seed == 20260729",
      "p", occ=1),
    K(632, "0.0495", WEA, f"{LB} and null == 'within' and seed == 20260727",
      "p", occ=2),
    K(632, "0.0594", WEA, f"{LB} and null == 'within' and seed == 20260729",
      "p"),
    DV(632, "0.3357",
       f"D('partition_sensitivity.csv').query(\"{LB}\").bal.mean()",
       "mean over the 20 partitions drawn without class balancing"),
    K(632, "0.3222", PRS, RFH, "observed_mean"),
    K(632, "0.3114", PRS, RFH, "partition_min"),
    K(632, "0.3351", PRS, RFH, "partition_max"),
    K(632, "0.3768", CT, LB, "observed_mean10", occ=1),
    DV(634, "0.006",
       f"(lambda p: (p * (1 - p) / 1000) ** 0.5)(cell(CT, \"{LB}\", "
       "'within_p_single_maxT'))",
       "Monte Carlo standard error of a p near 0.038 at 1,000 relabellings"),
    DV(634, "0.0323", "1 / (1 + 30)",
       "smallest attainable p with 30 shared relabellings", occ=0),
    K(636, "0.3457", JK, "left_out == 17", "mean10"),
    ST(640, "0.05", "setting or threshold", "false-positive rate by "
       "construction of the 95th-percentile threshold"),
    K(642, "0.73", PW, "cohort == 34 and delta == 0.04", "power"),
    K(642, "1.00", PW, "cohort == 34 and delta == 0.06", "power"),
    K(642, "0.26", PW, "cohort == 20 and delta == 0.04", "power"),
    K(644, "0.18", PW, "cohort == 34 and delta == 0.02", "power"),
    K(644, "0.73", PW, "cohort == 34 and delta == 0.04", "power"),
    K(644, "0.3487", PW, "cohort == 34 and delta == 0.02", "achieved_bal"),
    K(644, "0.3393", PRS, LB, "observed_mean"),
    DV(650, "0.05", "(0.25 / 100) ** 0.5",
       "largest binomial standard error at 100 simulations"),
    K(681, "0.3224", PT, P_PART, "bal_mean"),
    K(681, "0.3393", PRS, LB, "observed_mean"),
    K(681, "0.9055", LS, "rung == 5", "leaky_acc"),
    DV(710, "0.0010", "1 / (1 + 1000)",
       "smallest attainable p with 1,000 shared relabellings"),
    DV(710, "0.0323", "1 / (1 + 30)",
       "smallest attainable p with 30 shared relabellings"),
    # ---------------------------------------------------------- overlap, ladder
    LG(738, "+0.3150", "test27_overlap_sweep.txt",
       "window-shuffled, 0% -> 90% overlap : +0.3150"),
    LG(738, "-0.0011", "test27_overlap_sweep.txt",
       "participant-wise, same change : -0.0011"),
    LG(744, "-0.0011", "test27_overlap_sweep.txt",
       "participant-wise, same change : -0.0011"),
    DV(771, "+0.0089", "cell(LS, 'rung == 5', 'sound_bal') - "
       "cell(LS, 'rung == 0', 'sound_bal')", "rung 5 minus rung 0"),
    K(777, "0.4208", LS, "rung == 0", "leaky_acc"),
    K(777, "0.9055", LS, "rung == 5", "leaky_acc"),
    K(777, "0.3246", LS, "rung == 0", "sound_bal"),
    K(777, "0.3335", LS, "rung == 5", "sound_bal"),
    DV(781, "+0.32", "cell(LS, 'rung == 2', 'leaky_acc') - "
       "cell(LS, 'rung == 1', 'leaky_acc')", "rung 2 minus rung 1"),
    DV(781, "+0.02", "cell(LS, 'rung == 5', 'leaky_acc') - "
       "cell(LS, 'rung == 2', 'leaky_acc')", "rung 5 minus rung 2"),
    K(785, "0.8827", LS, "rung == 2", "leaky_acc"),
    K(804, "0.3492", "random_label_ladder.csv",
      "mode == 'within' and column == 'sound'", "bal"),
    K(811, "0.3492", "random_label_ladder.csv",
      "mode == 'within' and column == 'sound'", "bal"),
    K(813, "26.8", "memorisation_check.csv", "overlap == 0",
      "same_recording", scale=100),
    K(813, "73.0", "memorisation_check.csv", "overlap == 0",
      "same_participant", scale=100),
    DV(813, "0.093", "1 - cell('memorisation_check.csv', 'overlap == 90', "
       "'same_recording')", "share of test windows whose nearest training "
       "window is from another recording"),
    DV(817, "1.8", "(0.9233 - cell(LS, 'rung == 5', 'leaky_acc')) * 100",
       "points below the lower published figure"),
    DV(817, "3.3", "(0.9383 - cell(LS, 'rung == 5', 'leaky_acc')) * 100",
       "points below the higher published figure"),
    # ---------------------------------------------------------- mechanism
    K(838, "0.0028", PT, P_WIN, "acc_sd"),
    K(841, "0.0017", PT, P_PART, "bal_sd"),
    K(857, "0.5509", "fingerprint_table.csv",
      "quantity == 'subgroup21 forest_acc'", "value"),
    K(857, "0.07", "fingerprint_table.csv",
      "quantity == 'draws corr nmi acc'", "value"),
    K(870, "0.5509", "fingerprint_table.csv",
      "quantity == 'subgroup21 forest_acc'", "value"),
    K(880, "0.5480", "fingerprint_table.csv",
      "quantity == 'draws acc q2.5'", "value"),
    ST(904, "0.5", "plot coordinate", "axis tick"),
    ST(904, "0.6", "plot coordinate", "axis tick"),
    ST(909, "0.99", "plot coordinate", "legend position", occ=0),
    ST(909, "0.99", "plot coordinate", "legend position", occ=1),
    DV(925, "1.83",
       "cell('fingerprint_families.csv', \"family == 'absolute band power'\","
       " 'impurity') / cell('fingerprint_families.csv', "
       "\"family == 'relative band power'\", 'impurity')",
       "absolute over relative band-power importance"),
    DV(929, "14.3", "cell('identity_basis.csv', \"family == 'all features'\","
       " 'acc_session_held_out') * 27", "accuracy over 27-way chance"),
    K(972, "0.1685", RC, "dataset == 'DEED' and contrast == "
      "'window-shuffled minus participant-separated (forest)'", "estimate"),
    K(976, "0.1685", RC, "dataset == 'DEED' and contrast == "
      "'window-shuffled minus participant-separated (forest)'", "estimate"),
    K(1001, "0.0120", "density_both_partitions.csv",
      "model == 'lr_fixed' and partition == 'window-shuffled' and "
      "channels == 48", "bal_sd"),
    K(1001, "0.0126", "density_both_partitions.csv",
      "model == 'rf' and partition == 'window-shuffled' and channels == 48",
      "bal_sd"),
    DV(1013, "0.10", "D('density_sensitivity.csv').query("
       "\"channels == '64 minus 6'\").ci_hi.max()",
       "largest upper interval bound of the 64-minus-6 change"),
    # ---------------------------------------------------------- within, CBraMod
    DV(1093, "1.4", "28 * 0.05", "participants expected below 0.05 by chance"),
    K(1093, "0.39", POWER, "n == 28", "power_t"),
    K(1093, "0.24", POWER, "n == 28", "power_signed_rank_resampled"),
    ST(1093, "0.80", "setting or threshold", "target power"),
    K(1095, "0.3246", CDS, "config == 'scratch'", "bal_lo"),
    K(1095, "0.3009", CDS, "config == 'scratch_revised'", "bal_lo"),
    K(1095, "+0.0249", "cbramod_diag_pretraining_effect.csv",
      "pretrained == 'revised'", "diff", occ=0),
    K(1095, "-0.0238", "cbramod_diag_pretraining_effect.csv",
      "pretrained == 'revised'", "lo"),
    K(1095, "0.3294", SEEDS, "config == 'revised' and seed == 20260728",
      "pooled_bal"),
    K(1095, "0.3517", SEEDS, "config == 'revised' and seed == 20260727",
      "pooled_bal", occ=1),
    K(1095, "0.3397", "cbramod_seeds_summary.csv", "config == 'revised'",
      "mean"),
    K(1095, "0.0091", "cbramod_seeds_summary.csv", "config == 'revised'",
      "sd"),
    K(1095, "0.3368", "cbramod_seeds_summary.csv",
      "config == 'scratch_revised'", "mean"),
    K(1095, "0.0185", "cbramod_seeds_summary.csv",
      "config == 'scratch_revised'", "sd"),
    K(1095, "+0.0249", "cbramod_seeds_pretraining.csv", "seed == 20260727",
      "diff", occ=1),
    DV(1095, "3.5", "cbramod_constant().ep_o.min()",
       "median chosen epoch in non-constant folds, lowest seed "
       "(142_cbramod_constant.py)"),
    DV(1095, "0.3337", "cbramod_constant().va_c.max()",
       "median validation balanced accuracy in constant folds, highest seed"),
    DV(1095, "0.3462", "cbramod_constant().va_o.min()",
       "median validation balanced accuracy in other folds, lowest seed"),
    DV(1095, "0.3493", "cbramod_constant().va_o.max()",
       "median validation balanced accuracy in other folds, highest seed"),
    K(1097, "0.3507", WITHIN_UV, "n == 28", "observed"),
    K(1097, "0.3410", WITHIN_UV, "n == 28", "null"),
    K(1097, "+0.0097", WITHIN_UV, "n == 28", "difference"),
    K(1097, "0.111", WITHIN_UV, "n == 28", "cohens_dz"),
    K(1097, "0.103", WITHIN_UV, "n == 28", "rank_biserial"),
    K(1097, "0.323", WITHIN_UV, "n == 28", "wilcoxon_p"),
    K(1097, "0.0594", "pcd_extra_cbramod.csv",
      "condition == 'DEED affect, CBraMod embeddings' and "
      "rule == 'PCD p<=0.05'", "coverage"),
    DV(1099, "+0.0107", DIFF_UV, "CBraMod minus spectral, microvolts"),
    DV(1099, "-0.0462", DIFF_100, "CBraMod minus spectral, 100 uV"),
    K(1115, "0.3143", CDS, "config == 'revised'", "pooled_acc"),
    ST(1121, "0.01", "setting", "weight decay"),
    ST(1121, "1.0", "setting", "gradient-norm clip"),
    K(1121, "-0.1114", "cbramod_diag_pretraining_effect.csv",
      "pretrained == 'published'", "lo"),
    DV(1128, "1.4", "28 * 0.05", "participants expected below 0.05 by chance"),
    K(1128, "0.3507", WITHIN_UV, "n == 28", "observed"),
    K(1128, "0.3410", WITHIN_UV, "n == 28", "null"),
    K(1128, "0.323", WITHIN_UV, "n == 28", "wilcoxon_p"),
    # ---------------------------------------------------------- PCD
    K(1136, "0.0864", "pcd_revised_deednull_summary.csv",
      "quantity == 'marginal rate max'", "value"),
    DV(1136, "0.0355", "D('pcd_revised_sims.csv').fpr.quantile(0.025)",
       "2.5th percentile over simulated datasets"),
    DV(1136, "0.0664", "D('pcd_revised_sims.csv').fpr.quantile(0.975)",
       "97.5th percentile over simulated datasets"),
    DV(1136, "0.0251", "D('pcd_revised_sims.csv').p_uniform_ks.median()",
       "median Kolmogorov distance (column p_uniform_ks, 77_pcd_revised.py)"),
    ST(1164, "0.65", "setting or threshold", "coverage level of the grid"),
    ST(1164, "0.80", "setting or threshold", "coverage level of the grid"),
    ST(1164, "0.85", "setting or threshold", "coverage level of the grid",
       occ=0),
    ST(1164, "0.95", "setting or threshold", "nominal coverage"),
    K(1212, "0.1174", "pcd_extra_nulldep.csv",
      "knob == 'features' and value == 36", "null_mean_margin"),
    K(1212, "0.7943", "pcd_extra_nulldep.csv",
      "knob == 'participants' and value == 40", "coverage"),
    K(1212, "0.7577", "pcd_extra_nulldep.csv",
      "knob == 'participants' and value == 40", "balanced", occ=1),
    ST(1216, "0.05", "setting or threshold", "PCD level alpha"),
    ST(1216, "0.10", "setting or threshold", "PCD level alpha"),
    DV(1243, "1.0000", SPEAR, "Spearman correlation over the noise levels"),
    DV(1243, "0.7943", PEARS, "Pearson correlation over the noise levels"),
    K(1243, "0.3325", NOISE, "eta == 1.0", "balanced"),
    ST(1243, "0.05", "setting or threshold", "nominal level"),
    ST(1248, "0.05", "setting or threshold", "nominal level (dotted line)",
       occ=1),
    # ---------------------------------------------------------- comparisons
    K(1302, "0.0022", "protocol_factorial_effects_full.csv",
      "effect == 'selection before vs inside' and a == 'window/before/none'",
      "d_bal"),
    K(1302, "0.0009", "protocol_factorial_effects_full.csv",
      "effect == 'selection before vs inside' and a == 'window/before/none'",
      "ci_lo"),
    K(1302, "0.0060", "protocol_factorial_effects_full.csv",
      "effect == 'selection before vs inside' and a == 'window/before/none'",
      "ci_hi"),
    DV(1308, "+0.0107", DIFF_UV, "CBraMod minus spectral, microvolts"),
    DV(1308, "-0.0462", DIFF_100, "CBraMod minus spectral, 100 uV"),
    K(1308, "0.3294", SEEDS, "config == 'revised' and seed == 20260728",
      "pooled_bal"),
    # ---------------------------------------------------------- discussion
    K(1325, "0.80", "mechanism_refresh.csv",
      "dataset == 'DEED' and quantity == 'identity window-shuffled'", "mean"),
    K(1325, "0.87", "mechanism_refresh.csv",
      "dataset == 'Tononi' and quantity == 'identity window-shuffled'",
      "mean"),
    LG(1329, "61.4", "test22_dream_catalogue.txt",
       "participants with >1 record : 0.614"),
    DV(1329, "0.117", "D('catalogue_audit.csv').query(\"set_id != 'DEED'\")"
       ".nmi_excess.median()", "median over the twelve catalogue datasets"),
    DV(1350, "1.0000", SPEAR, "Spearman correlation over the noise levels"),
    K(1369, "0.3224", PT, P_PART, "bal_mean"),
    K(1369, "0.3393", PRS, LB, "observed_mean"),
    K(1369, "0.9055", LS, "rung == 5", "leaky_acc"),
    K(1371, "0.3224", PT, P_PART, "bal_mean"),
    K(1371, "0.3222", PRS, RFH, "observed_mean"),
    K(1371, "0.3393", PRS, LB, "observed_mean"),
    K(1371, "0.3409", JK, "left_out == 15", "mean10"),
    K(1371, "0.3457", JK, "left_out == 17", "mean10"),
    K(1371, "0.3487", PW, "cohort == 34 and delta == 0.02", "achieved_bal"),
    K(1373, "0.4208", LS, "rung == 0", "leaky_acc"),
    K(1375, "0.1374", RC, "dataset == 'DEED' and contrast == "
      "'window-shuffled minus recording-grouped (forest)'", "estimate"),
    DV(1377, "+0.0107", DIFF_UV, "CBraMod minus spectral, microvolts"),
    DV(1377, "-0.0462", DIFF_100, "CBraMod minus spectral, 100 uV"),
    K(1381, "0.4614", LEAKT, "partition == 'window-level' and "
      "rule == 'all inputs' and quantity == 'bal'", "mean"),
    K(1381, "0.3001", LEAKT, "partition == 'window-level' and "
      "rule == 'Max-softmax>0.5' and quantity == 'cov'", "mean"),
    K(1381, "0.0526", LEAKT, "partition == 'window-level' and "
      "rule == 'PCD alpha=0.05' and quantity == 'cov'", "mean"),
    K(1383, "0.0498", "pcd_revised_deednull_summary.csv",
      "quantity == 'marginal rate mean'", "value"),
    K(1391, "0.0396", CT, LB, "free_p_mean10"),
    K(1391, "0.0495", CT, LB, "within_p_mean10"),
    K(1391, "0.3409", JK, "left_out == 15", "mean10"),
    K(1391, "0.3457", JK, "left_out == 17", "mean10"),
    K(1391, "0.3487", PW, "cohort == 34 and delta == 0.02", "achieved_bal"),
    K(1402, "0.3785", "config_table_raw_1thread_discarded.csv",
      f"{LB} and mode == 'observed' and seed == 20260727", "bacc"),
    K(1402, "0.3783", CT, LB, "observed_single"),
    ST(1402, "0.001", "approximate bound",
       "observed run-to-run difference of multithreaded forests, not a "
       "stored result"),
    # ---------------------------------------------------------- the same
    # result held in more than one file: the file the sentence reports
    K(488, "0.4616", PT, P_PART, "majority"),
    K(607, "0.8696", LSS, "variant == 'corroborated recordings' and "
      f"{RFB} and statistic == 'recording'", "p_within", occ=1),
    K(632, "0.3768", CT, LB, "observed_mean10", occ=0),
    K(736, "0.0008", "overlap_sweep.csv",
      "overlap == 90 and partition == 'random'", "acc_sd"),
    DV(770, "+0.4847", "cell(LS, 'rung == 5', 'leaky_acc') - "
       "cell(LS, 'rung == 0', 'leaky_acc')", "rung 5 minus rung 0"),
    K(781, "0.1206", "protocol_factorial_effects_full.csv",
      "effect == 'oversampling before vs inside' and "
      "a == 'window/none/before'", "d_bal"),
    K(817, "0.9055", LS, "rung == 5", "leaky_acc"),
    K(1095, "0.3517", CDS, "config == 'revised'", "pooled_bal", occ=0),
    K(1097, "0.0516", "pcd_extra_cbramod_uv100.csv",
      "condition == 'DEED affect, CBraMod embeddings' and "
      "rule == 'PCD p<=0.05'", "coverage"),
    K(1164, "0.0618", "pcd_extra_grid.csv",
      "condition == 'DEED affect' and rule == 'PCD alpha=0.05'", "coverage"),
    K(1164, "0.8601", "pcd_extra_grid.csv",
      "condition == 'Sleep-EDF staging' and rule == 'PCD alpha=0.05'",
      "coverage"),
    K(1348, "0.9055", LS, "rung == 5", "leaky_acc"),
    K(1369, "0.3768", CT, LB, "observed_mean10"),
]


def power_table():
    """Table tab:power: each row is delta, then balanced accuracy and power
    for 10, 20 and 34 participants (power_revised.csv)."""
    lines = io.open(TEX, encoding="utf-8").read().split("\n")
    i = next(k for k, s in enumerate(lines) if "\\label{tab:power}" in s)
    out = []
    for k in range(i, len(lines)):
        if "\\end{tabular}" in lines[k]:
            break
        m = re.match(r"\s*(\d\.\d\d) & (.*)\\\\", lines[k])
        if not m:
            continue
        delta, rest = m.group(1), [x.strip() for x in m.group(2).split("&")]
        seen = {}
        cells = [(delta, None, "delta")] + [
            (rest[2 * j + c], n, col) for j, n in enumerate((10, 20, 34))
            for c, col in enumerate(("achieved_bal", "power"))]
        for v, n, col in cells:
            occ = seen.get(v, 0)
            seen[v] = occ + 1
            q = (f"delta == {float(delta)}" if n is None
                 else f"cohort == {n} and delta == {float(delta)}")
            out.append(K(k + 1, v, PW, q if n else
                         f"cohort == 34 and {q}", col, occ=occ))
    assert len(out) == 8 * 7, len(out)
    return out


def all_overrides():
    return OVERRIDES + power_table()


# ---------------------------------------------------------------- letter
# The response letter is regenerated by 134, so its numbers are named by
# their value and a piece of the text around them rather than a paragraph
# number; every matching number outside the reviewer's quoted comments takes
# the decision. Previous values come from the result files kept, before the
# label-free rerun of Concern 1, as stale_labelstrat_<name>.
GRID0 = "stale_labelstrat_pcd_extra_grid.csv"
FIX0 = "stale_labelstrat_pcd_strat_deednull_fixedfolds_summary.csv"
RED0 = "stale_labelstrat_pcd_strat_deednull_summary.csv"
DEED_ALL = "condition == 'DEED affect' and rule == 'all inputs'"
DEED_PCD = "condition == 'DEED affect' and rule == 'PCD alpha=0.05'"
SEDF_PCD = "condition == 'Sleep-EDF staging' and rule == 'PCD alpha=0.05'"
P_LO = ("np.floor(min(D(PRS).within_p.min(), D(PRS).free_p.min()) * 100) "
        "/ 100")


def LK(value, ctx, file, query, column, scale=1):
    return dict(value=value, ctx=ctx, occ=None, kind="key", file=file,
                query=query, column=column, scale=scale)


def LD(value, ctx, check, note):
    return dict(value=value, ctx=ctx, occ=None, kind="derived", check=check,
                note=note)


def LS_(value, ctx, status, note):
    return dict(value=value, ctx=ctx, occ=None, kind="status", status=status,
                note=note)


LETTER = [
    # previous values (superseded by the label-free rerun)
    LK("0.3638", "0.3638", GRID0, DEED_ALL, "bal_retained"),
    LK("0.0701", "0.0701", GRID0, DEED_PCD, "coverage"),
    LK("0.3558", "0.3558", GRID0, DEED_PCD, "bal_retained"),
    LK("0.8588", "0.8588", GRID0, SEDF_PCD, "coverage"),
    LK("0.0584", "0.0584", FIX0, "stratum == 'participant'",
       "participant_expected_max"),
    LK("0.0955", "0.0955", FIX0, "stratum == 'pooled'",
       "participant_expected_max"),
    LK("0.0619", "0.0619", RED0, "stratum == 'participant'",
       "participant_expected_max"),
    LK("0.1123", "0.1123", RED0, "stratum == 'pooled'",
       "participant_expected_max"),
    # current values whose automatic match was a coincidence
    LK("0.3263", "0.3263", "pcd_extra_grid.csv", DEED_ALL, "bal_retained"),
    LK("0.3177", "0.3177", "pcd_extra_grid.csv", DEED_PCD, "bal_retained"),
    LK("0.0618", "0.0618", "pcd_extra_grid.csv", DEED_PCD, "coverage"),
    LK("0.8601", "0.8601", "pcd_extra_grid.csv", SEDF_PCD, "coverage"),
    LK("0.3222", "average 0.3222", PRS, RFH, "observed_mean"),
    LK("0.3393", "0.3393", PRS, LB, "observed_mean"),
    LK("0.3768", "0.3768", CT, LB, "observed_mean10"),
    LK("0.0170", "0.0170", LSS, f"{L60} and {LB} and statistic == 'window'",
       "p_within"),
    LK("0.3507", "0.3507", WITHIN_UV, "n == 28", "observed"),
    LK("0.3410", "0.3410", WITHIN_UV, "n == 28", "null"),
    LK("0.323", "0.323", WITHIN_UV, "n == 28", "wilcoxon_p"),
    LK("0.0849", "0.0849", CT, LU, "within_p_single"),
    LK("0.0899", "0.0899", CT, LU, "free_p_single"),
    LK("0.24", "0.24", POWER, "n == 28", "power_signed_rank_resampled"),
    LK("0.2352", "0.2352", POWER, "n == 28", "power_signed_rank_resampled"),
    # computed in the text
    LD("0.27", "0.27", P_LO, "smallest p of the primary evaluation, rounded "
       "down"),
    LD("0.0303", "0.0303", f"1 / (1 + cell(PRS, \"{LB}\", 'within_maxT_B'))",
       "smallest attainable max-statistic p"),
    LD("0.0680", "0.0680", HOLM60, "smallest Holm-adjusted p"),
    LD("0.0107", "0.0107", DIFF_UV, "CBraMod minus spectral, microvolts"),
    LD("+0.0107", "+0.0107", DIFF_UV, "CBraMod minus spectral, microvolts"),
    LD("−0.0462", "−0.0462", DIFF_100,
       "CBraMod minus spectral, 100 uV"),
    # not results
    LS_("0.05", "0.05", "setting or threshold", "PCD level alpha"),
    LS_("0.23", "0.23", "previously reported value (corrected)",
        "power of the signed-rank test as previously printed"),
    LS_("10.1101", "10.1101", "not a number", "part of a DOI"),
]


def letter_overrides():
    return LETTER
