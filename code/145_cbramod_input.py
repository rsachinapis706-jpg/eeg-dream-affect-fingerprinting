"""
CBraMod input scaling in the manuscript (round 3, Reviewer 2, comment 9).

The note to Table tab:cbramod said that CBraMod's input "follows the
pretraining convention (median absolute value 0.624 in units of 100 uV)".
That was wrong: the DEED windows given to CBraMod (fine-tuning, 76, and
frozen embeddings, 15) were standardised per window and channel, and the
Sleep-EDF control (49) received microvolts, whereas CBraMod was pretrained on
EEG in units of 100 uV. The analyses were repeated in that unit (144 builds
the DEED windows; CBRAMOD_SCALE=uv100 in 15, 85 and 49, and in 76 for the
second settings, pretrained, first seed only: the reviewer asked that the
scaling be documented, not rerun, so the further seeds and the
randomly initialised model were not run). This script

  * computes the median absolute input on the 6,144 fine-tuning windows for
    both scalings, exactly (two-pass selection on the memory-mapped arrays),
    and writes out/cbramod_input_amplitude.csv;
  * corrects the table note;
  * adds a paragraph to Section IV-J with the frozen-embedding, PCD and
    fine-tuning results in the pretraining unit;
  * reports the Sleep-EDF control under both scalings in IV-J and Section V.

Table 21 (122) and Section VII (101) read the same result files. If the
results in the pretraining unit change a conclusion of IV-J, the script stops
instead of writing text: that case must be written by hand. Idempotent.
"""
import importlib.util
import io
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
TEX = r"E:\Dream_Analysis\accesstemp.tex"
AMP = os.path.join(OUT, "cbramod_input_amplitude.csv")
INPUTS = {"standardised per window and channel": "deed_raw.npy",
          "units of 100 uV (pretraining convention)": "deed_raw_uv100.npy"}


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def abs_median(path, idx, chunk=256, bins=1 << 16):
    """Exact median of |x| over the windows idx: a histogram locates the
    order statistics, a second pass collects the values in their bin."""
    X = np.load(path, mmap_mode="r")
    n = len(idx) * int(np.prod(X.shape[1:]))
    parts = [idx[i:i + chunk] for i in range(0, len(idx), chunk)]
    mx = max(float(np.abs(X[p]).max()) for p in parts)
    edges = np.linspace(0.0, mx, bins + 1)
    cnt = np.zeros(bins, np.int64)
    for p in parts:
        cnt += np.histogram(np.abs(X[p]), edges)[0]
    k1, k2 = (n - 1) // 2, n // 2
    cum = np.cumsum(cnt)
    b1 = int(np.searchsorted(cum, k1, side="right"))
    b2 = int(np.searchsorted(cum, k2, side="right"))
    lo, hi = edges[b1], edges[min(b2 + 1, bins)]
    below, vals = 0, []
    for p in parts:
        a = np.abs(X[p]).ravel()
        below += int((a < lo).sum())
        vals.append(a[(a >= lo) & (a <= hi)])
    v = np.sort(np.concatenate(vals))
    return 0.5 * (float(v[k1 - below]) + float(v[k2 - below]))


def amplitudes():
    if os.path.exists(AMP):
        return pd.read_csv(AMP).set_index("input").median_abs
    cbft = _load("cbft", "14_cbramod_finetune.py")
    m = pd.read_parquet(os.path.join(OUT, "deed_raw_meta.parquet"))
    idx = np.where(m.E.isin(cbft.COLL).to_numpy())[0]
    assert len(idx) == 6144, len(idx)
    rows = [dict(input=k, file=f, windows=len(idx),
                 median_abs=abs_median(os.path.join(OUT, f), idx))
            for k, f in INPUTS.items()]
    pd.DataFrame(rows).to_csv(AMP, index=False)
    return pd.DataFrame(rows).set_index("input").median_abs


def within(fname):
    """As 45_statistical_foundation.within_effects."""
    w = pd.read_csv(os.path.join(OUT, fname)).dropna(subset=["bacc", "null"])
    d = (w.bacc - w.null).to_numpy()
    nz = d[d != 0]
    r = stats.rankdata(np.abs(nz))
    wp, wn = r[nz > 0].sum(), r[nz < 0].sum()
    return dict(obs=w.bacc.mean(), null=w.null.mean(), diff=d.mean(),
                dz=d.mean() / d.std(ddof=1), r=(wp - wn) / (wp + wn),
                above=int((d > 0).sum()), below=int((d < 0).sum()),
                p=stats.wilcoxon(nz, alternative="greater").pvalue)


def pcd_cov(fname):
    t = pd.read_csv(os.path.join(OUT, fname))
    return float(t[t.rule == "PCD p<=0.05"].coverage.iloc[0])


FT = os.path.join(OUT, "cbramod_uv100_finetune.csv")


def finetune_uv100(seed=20260727, n_boot=2000):
    """The one fine-tuning run in the pretraining unit (second settings,
    pretrained, first seed; 76 with CBRAMOD_SCALE=uv100), beside the run with
    standardised inputs at the same seed: pooled balanced accuracy, 95%
    participant bootstrap, and the paired difference on the same resamples
    (as 105_cbramod_seeds.py)."""
    sd = _load("sd105", "105_cbramod_seeds.py")
    runs = {}
    for tag, cfg in (("units of 100 uV", "revised_uv100"),
                     ("standardised", "revised")):
        f = pd.read_csv(os.path.join(OUT, f"cbramod_diag_{cfg}_{seed}_folds.csv"),
                        dtype={"held_out": str})
        assert len(f) == 34 and not f.held_out.duplicated().any(), (cfg, len(f))
        per = {s: tuple(np.load(os.path.join(
            OUT, f"cbramod_diag_{cfg}_{seed}_{s}_pred.npy"))) for s in f.held_out}
        runs[tag] = (f, per)
    ids = sorted(runs["units of 100 uV"][1])
    assert ids == sorted(runs["standardised"][1])
    rng = np.random.default_rng(seed)
    draws = [rng.choice(ids, len(ids)) for _ in range(n_boot)]
    rows, boots = [], {}
    for tag, (f, per) in runs.items():
        boots[tag] = np.array([sd.pooled(per, d) for d in draws])
        rows.append(dict(input=tag, seed=seed, pooled_bal=sd.pooled(per, ids),
                         bal_lo=np.percentile(boots[tag], 2.5),
                         bal_hi=np.percentile(boots[tag], 97.5),
                         constant_folds=int(f.constant_pred.sum()),
                         chosen_epoch_median=float(f.chosen_epoch.median())))
    d = boots["units of 100 uV"] - boots["standardised"]
    rows.append(dict(input="difference (100 uV minus standardised)", seed=seed,
                     pooled_bal=rows[0]["pooled_bal"] - rows[1]["pooled_bal"],
                     bal_lo=np.percentile(d, 2.5), bal_hi=np.percentile(d, 97.5)))
    t = pd.DataFrame(rows)
    t.round(4).to_csv(FT, index=False)
    return t.set_index("input")


def main():
    amp = amplitudes()
    z = amp["standardised per window and channel"]
    u = amp["units of 100 uV (pretraining convention)"]
    fz = within("cbramod_within_subject.csv")
    fu = within("cbramod_within_subject_uv100.csv")
    cov0, cov1 = pcd_cov("pcd_extra_cbramod.csv"), pcd_cov("pcd_extra_cbramod_uv100.csv")
    ft = finetune_uv100()
    u1, s1 = ft.loc["units of 100 uV"], ft.loc["standardised"]
    dd = ft.loc["difference (100 uV minus standardised)"]
    cx = {k: pd.read_csv(os.path.join(OUT, f"cbramod_expanded{k}.csv")).set_index("model")
          for k in ("", "_uv100")}
    sp = cx[""].loc["spectral_logreg", "balanced"]
    assert sp == cx["_uv100"].loc["spectral_logreg", "balanced"]
    c_uv, c_100 = (cx[k].loc["cbramod", "balanced"] for k in ("", "_uv100"))

    chance = u1.bal_lo <= 1 / 3 <= u1.bal_hi
    if not (chance and fu["p"] >= 0.05 and fz["p"] >= 0.05):
        raise SystemExit("results in the pretraining unit change a conclusion "
                         "of IV-J; write that paragraph by hand:\n"
                         f"{ft}\nfrozen p {fu['p']:.3f}")

    # 1. the table note
    old_note = ("input amplitude follows the pretraining convention (median "
                "absolute value 0.624 in units of 100\\,$\\mu$V)")
    new_note = (f"each window and channel is standardised (median absolute "
                f"value {z:.3f}), not expressed in units of 100\\,$\\mu$V as "
                f"in pretraining (median absolute value {u:.3f} on these "
                f"windows); the text reports the second settings in that unit")
    # 2. the paragraph in IV-J
    start = "The windows given to CBraMod in Table~\\ref{tab:cbramod}"
    par = (
        f"{start} and for the frozen embeddings above were standardised per "
        f"window and channel (median absolute value {z:.3f}), whereas the "
        f"model was pretrained on EEG in units of 100\\,$\\mu$V, in which "
        f"these windows have median absolute value {u:.3f}. Both analyses "
        f"were therefore repeated with the windows in that unit, all other "
        f"settings unchanged. The frozen embeddings then give "
        f"{fu['obs']:.4f} against a null of {fu['null']:.4f}, a difference "
        f"of ${fu['diff']:+.4f}$ ($d_z={fu['dz']:.3f}$, $r={fu['r']:.3f}$; "
        f"{fu['above']} participants above their null and {fu['below']} "
        f"below; one-sided $p={fu['p']:.3f}$), and PCD on them predicts on "
        f"{cov1:.4f} of windows, against {cov0:.4f} with standardised "
        f"inputs. Fine-tuned with pretraining under the second settings, at "
        f"the first seed, pooled balanced accuracy is {u1.pooled_bal:.4f} "
        f"[{u1.bal_lo:.4f}, {u1.bal_hi:.4f}], against {s1.pooled_bal:.4f} "
        f"with standardised inputs at the same seed "
        f"(Table~\\ref{{tab:cbramod}}), a paired difference of "
        f"${dd.pooled_bal:+.4f}$ "
        f"[${dd.bal_lo:+.4f}$, ${dd.bal_hi:+.4f}$], and "
        f"{int(u1.constant_folds)} of 34 folds predict a single class, "
        f"against {int(s1.constant_folds)}. The interval includes the 0.3333 "
        f"chance level; this one run was not repeated over seeds or without "
        f"pretraining. Expressing the input in the unit of pretraining "
        f"therefore leaves the conclusions of this section unchanged.")
    # 3. the positive control, in IV-J and in Section V
    pc_old = ("The corresponding positive control is reported in "
              "Section~\\ref{sec5}, where the model is compared with the "
              "spectral pipeline on the identical cohort, windows and folds: "
              "0.7906 against 0.7799, a difference of $+0.0107$.")
    pc_new = ("The corresponding positive control is reported in "
              "Section~\\ref{sec5}, where the model is compared with the "
              "spectral pipeline on the identical cohort, windows and folds: "
              f"{c_uv:.4f} against {sp:.4f} with the windows in microvolts, "
              f"a difference of ${c_uv - sp:+.4f}$, and {c_100:.4f} with the "
              f"windows in units of 100\\,$\\mu$V, a difference of "
              f"${c_100 - sp:+.4f}$.")
    v_old = ("CBraMod reaches 0.7906 balanced accuracy and the spectral "
             "pipeline 0.7799, a difference of $+0.0107$.")
    v_new = (f"With the windows in microvolts CBraMod reaches {c_uv:.4f} "
             f"balanced accuracy and the spectral pipeline {sp:.4f}, a "
             f"difference of ${c_uv - sp:+.4f}$; with the windows in units "
             f"of 100\\,$\\mu$V, the convention of its pretraining, CBraMod "
             f"reaches {c_100:.4f}, a difference of ${c_100 - sp:+.4f}$. "
             f"The training schedule was not retuned for either scaling. "
             f"CBraMod detects the sleep-stage signal under both, but the "
             f"comparison does not show which representation is better on "
             f"this task.")
    # 4. the specification table (Table tab:spec)
    t_old = ("DEED valence raw, 6 ch $\\times$ 4{,}000 samples, 6{,}144 / 492 "
             "/ 34 & learned")
    t_new = ("DEED valence raw, 6 ch $\\times$ 4{,}000 samples, 6{,}144 / 492 "
             "/ 34; standardised per window and channel, repeated in units "
             "of 100\\,$\\mu$V & learned")
    c_old = ("& RF, LR, CBraMod & leave-one-participant-out; matched 5-fold "
             "for CBraMod &")
    c_new = ("& RF, LR, CBraMod (input in $\\mu$V and in units of "
             "100\\,$\\mu$V) & leave-one-participant-out; matched 5-fold for "
             "CBraMod &")
    # 5. the caption of Fig. fig:within (the figure shows standardised inputs)
    f_old = (f"Right, frozen CBraMod embeddings: mean {fz['obs']:.4f} against "
             f"{fz['null']:.4f} (difference ${fz['diff']:+.4f}$), one-sided "
             f"$p={fz['p']:.3f}$.")
    f_new = (f"Right, frozen CBraMod embeddings of windows standardised per "
             f"window and channel: mean {fz['obs']:.4f} against "
             f"{fz['null']:.4f} (difference ${fz['diff']:+.4f}$), one-sided "
             f"$p={fz['p']:.3f}$; with the windows in units of "
             f"100\\,$\\mu$V, as in pretraining, {fu['obs']:.4f} against "
             f"{fu['null']:.4f}, $p={fu['p']:.3f}$ (text).")
    m_old = ("The one-point margin therefore compares a pretrained "
             "transformer with spectral features under unequal tuning;")
    m_new = ("Either margin therefore compares a pretrained transformer with "
             "spectral features under unequal tuning;")

    s = io.open(TEX, encoding="utf-8", newline="").read()
    for old, new in ((old_note, new_note), (pc_old, pc_new),
                     (v_old, v_new), (m_old, m_new), (t_old, t_new),
                     (c_old, c_new), (f_old, f_new)):
        if old in s:
            assert s.count(old) == 1, old[:60]
            s = s.replace(old, new)
    for key in ("each window and channel is standardised (median absolute",
                "with the windows in microvolts, a difference of",
                "the convention of its pretraining, CBraMod reaches",
                "Either margin therefore compares", t_new, c_new, f_new):
        assert s.count(key) == 1, key
    # regenerate the note and the two control sentences from current values
    for key, new, end in (
            ("each window and channel is standardised (median absolute", new_note,
             "the text reports the second settings in that unit"),
            ("The corresponding positive control is reported in", pc_new,
             "\n"),
            ("With the windows in microvolts CBraMod reaches", v_new,
             "on this task.")):
        i = s.index(key)
        j = s.index(end, i) + (0 if end == "\n" else len(end))
        s = s[:i] + new + s[j:]
    if start in s:
        i = s.index(start)
        j = s.index("\n\n", i) + 2
        s = s[:i] + s[j:]
    anchor = "The corresponding positive control is reported in Section~\\ref{sec5}"
    assert s.count(anchor) == 1
    i = s.index(anchor)
    s = s[:i] + par + "\n\n" + s[i:]
    assert s.count(start) == 1
    io.open(TEX, "w", encoding="utf-8", newline="").write(s)
    print(f"input median |x|: standardised {z:.3f}, units of 100 uV {u:.3f}")
    print(par)


if __name__ == "__main__":
    main()
