"""
Process Tononi Serial Awakenings: an independent test of the fingerprinting
mechanism.

What this tests, and what it does not
-------------------------------------
Tononi labels each awakening as dreaming / dreaming-without-recall / no
experience. It carries no affective ratings, so it CANNOT replicate the
dream-affect null. What it can do is test the MECHANISM on independent data.

The mechanism claims that window-level partitioning inflates accuracy because it
lets a classifier recognise the participant, and identity predicts the label only
where labels are concentrated within participants. The two corpora differ
sharply on that precondition:

    DEED    median within-participant label dominance 0.750, 38% of
            participants above 70%   -> observed inflation +17.1 points
    Tononi  median 0.571, 17.9% above 70%   -> mechanism predicts
            SUBSTANTIALLY LESS inflation

This prediction is registered before the analysis is run. Comparable inflation
in Tononi would falsify the mechanism.

Channel selection
-----------------
Tononi is 256-channel HydroCel GSN at 500 Hz; DEED is 6 channels at 200 Hz.
Using all 256 would alter the feature-to-sample ratio and confound the very
comparison being made, so six electrodes are selected to match DEED's channel
count. Selection is geometric rather than arbitrary: the shipped .sfp montage
gives 3-D coordinates, and the electrode nearest each canonical 10-20 position
(Fz, Cz, Pz, Oz, C3, C4) is chosen. Signals are resampled to 200 Hz so the
feature definitions are identical to those used on DEED.
"""
import os
import re
import io
import glob
import zipfile
import tempfile
import warnings
import numpy as np
import pandas as pd
import mne
from scipy import signal as sps

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
SRC = os.path.join(ROOT, "tononi")

FS = 200.0                  # resampled to match DEED
EPOCH_S = 20
EPOCH_N = int(FS * EPOCH_S)
ARTIFACT_UV = 300.0

# canonical 10-20 directions on the unit sphere: x right, y anterior, z up
TARGETS = {
    "Fz": (0.000,  0.707,  0.707),
    "Cz": (0.000,  0.000,  1.000),
    "Pz": (0.000, -0.707,  0.707),
    "Oz": (0.000, -0.951,  0.309),
    "C3": (-0.707, 0.000,  0.707),
    "C4": (0.707,  0.000,  0.707),
}

_trapz = getattr(np, "trapezoid", None) or np.trapz
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
         "sigma": (12.0, 16.0), "beta": (16.0, 30.0), "gamma": (30.0, 45.0)}
_b, _a = sps.butter(4, [0.5 / (FS / 2), 45.0 / (FS / 2)], btype="band")

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


# ------------------------------------------------------------------ montage
def read_montage():
    """Parse the single-line .sfp into {electrode_number: unit vector}."""
    z = zipfile.ZipFile(os.path.join(SRC, "part1.zip"))
    name = [n for n in z.namelist() if n.endswith(".sfp")][0]
    toks = z.read(name).decode("utf8", "ignore").split()
    pos = {}
    i = 0
    while i + 3 < len(toks) + 1:
        lab = toks[i]
        try:
            x, y, zc = (float(toks[i + 1]), float(toks[i + 2]),
                        float(toks[i + 3]))
        except (ValueError, IndexError):
            break
        v = np.array([x, y, zc], float)
        n = np.linalg.norm(v)
        if n > 0:
            pos[lab] = v / n
        i += 4
    return pos


def select_channels(pos):
    """Electrode nearest each canonical position, by angular distance."""
    chosen, used = {}, set()
    for lab, t in TARGETS.items():
        t = np.array(t, float)
        t /= np.linalg.norm(t)
        best, bd = None, -2
        for k, v in pos.items():
            if k in used:
                continue
            d = float(np.dot(v, t))
            if d > bd:
                bd, best = d, k
        chosen[lab] = best
        used.add(best)
    return chosen


# ------------------------------------------------------------------ features
def hjorth(x):
    dx = np.diff(x); ddx = np.diff(dx)
    v0 = np.var(x) + 1e-12; v1 = np.var(dx) + 1e-12; v2 = np.var(ddx) + 1e-12
    mob = np.sqrt(v1 / v0)
    return mob, np.sqrt(v2 / v1) / (mob + 1e-12)


def epoch_features(ep):
    """Identical definitions to the DEED pipeline."""
    feats = {}
    nper = int(FS * 4)
    freqs, psd = sps.welch(ep, fs=FS, nperseg=nper, noverlap=nper // 2, axis=-1)
    tm = (freqs >= 0.5) & (freqs <= 45.0)
    total = _trapz(psd[:, tm], freqs[tm], axis=-1) + 1e-12
    ftm = freqs[tm]
    for ci in range(ep.shape[0]):
        p = psd[ci]
        for bn, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            bp = _trapz(p[mask], freqs[mask]) + 1e-12
            feats[f"ch{ci}_{bn}_abs"] = np.log10(bp)
            feats[f"ch{ci}_{bn}_rel"] = bp / total[ci]
        pn = p[tm] / (p[tm].sum() + 1e-12)
        feats[f"ch{ci}_specent"] = float(
            -(pn * np.log2(pn + 1e-12)).sum() / np.log2(len(pn)))
        c = np.cumsum(p[tm]); c /= c[-1] + 1e-12
        feats[f"ch{ci}_sef95"] = float(ftm[min(np.searchsorted(c, 0.95),
                                              len(ftm) - 1)])
        am = (freqs >= 2.0) & (freqs <= 40.0)
        feats[f"ch{ci}_apslope"] = float(
            np.polyfit(np.log10(freqs[am]), np.log10(p[am] + 1e-20), 1)[0])
        mob, comp = hjorth(ep[ci])
        feats[f"ch{ci}_hj_mob"] = float(mob)
        feats[f"ch{ci}_hj_comp"] = float(comp)
        feats[f"ch{ci}_logvar"] = float(np.log10(np.var(ep[ci]) + 1e-12))
    return feats


# ------------------------------------------------------------------ labels
def read_reports():
    z = zipfile.ZipFile(os.path.join(SRC, "part1.zip"))
    rows = []
    for n in z.namelist():
        if not (n.endswith(".xlsx") and "Reports_" in n):
            continue
        sid = n.split("Reports_")[1].replace(".xlsx", "").lower()
        try:
            d = pd.read_excel(io.BytesIO(z.read(n)))
        except Exception:
            continue
        d["subject"] = sid
        rows.append(d)
    r = pd.concat(rows, ignore_index=True)
    r.columns = [str(c).strip() for c in r.columns]
    return r


def main():
    pos = read_montage()
    sel = select_channels(pos)
    rep = read_reports()

    log("=" * 74)
    log("TONONI SERIAL AWAKENINGS -- INDEPENDENT TEST OF THE MECHANISM")
    log("=" * 74)
    log(f"montage electrodes parsed : {len(pos)}")
    log(f"selected channels         : "
        + ", ".join(f"{k}=Chan {v}" for k, v in sel.items()))
    log(f"report rows               : {len(rep)}  "
        f"subjects {rep.subject.nunique()}")
    log("")

    # The .sfp labels electrodes 1..256 numerically and names the 257th "Cz";
    # the EDF calls them "Chan 1".."Chan 257". Map the named vertex electrode
    # onto its channel index rather than emitting "Chan Cz".
    def to_channel(label):
        return f"Chan {label}" if str(label).isdigit() else "Chan 257"

    want = [to_channel(sel[k]) for k in ["Fz", "Cz", "Pz", "Oz", "C3", "C4"]]
    log(f"EDF channels requested    : {want}")

    archives = sorted(glob.glob(os.path.join(SRC, "Part*-s*.zip")))
    log(f"archives available        : {len(archives)}")
    log("")

    rows, skipped = [], []
    for arc in archives:
        try:
            z = zipfile.ZipFile(arc)
        except Exception as e:
            skipped.append((os.path.basename(arc), f"unreadable: {e}"))
            continue
        edfs = [n for n in z.namelist() if n.endswith(".edf")]
        for n in sorted(edfs):
            base = os.path.basename(n).replace(".edf", "")     # s01_ep01
            m = re.match(r"s(\d+)_ep(\d+)", base)
            if not m:
                continue
            subj, ep_no = f"s{m.group(1)}", int(m.group(2))
            with tempfile.TemporaryDirectory() as td:
                p = z.extract(n, td)
                try:
                    raw = mne.io.read_raw_edf(p, preload=True)
                    have = [c for c in want if c in raw.ch_names]
                    if len(have) < 6:
                        skipped.append((base, f"channels {len(have)}/6"))
                        continue
                    raw.pick(have)
                    raw.resample(FS)
                    data = sps.filtfilt(_b, _a, raw.get_data() * 1e6, axis=-1)
                    del raw
                except Exception as e:
                    skipped.append((base, f"{type(e).__name__}"))
                    continue

            lab = rep[(rep.subject == subj)
                      & (rep["Awakening"] == ep_no)]
            if not len(lab):
                skipped.append((base, "no matching report row"))
                continue
            dreaming = lab["Dreaming"].iloc[0]
            if dreaming not in (1, 2, 3):
                continue

            n_ep = data.shape[1] // EPOCH_N
            for e in range(n_ep):
                seg = data[:, e * EPOCH_N:(e + 1) * EPOCH_N]
                if np.abs(seg).max() > ARTIFACT_UV or np.allclose(
                        seg.std(axis=1), 0):
                    continue
                f = epoch_features(seg)
                f.update(subject=subj, awakening=ep_no,
                         recording=base, dreaming=int(dreaming),
                         window=e)
                rows.append(f)
        log(f"  {os.path.basename(arc):<18} cumulative windows {len(rows)}")

    if not rows:
        log("\nNo windows extracted.")
        return

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(OUT, "tononi_features.parquet"), index=False)

    log("")
    log(f"windows      : {len(df)}")
    log(f"subjects     : {df.subject.nunique()}")
    log(f"recordings   : {df.recording.nunique()}")
    log(f"windows/rec  : {len(df)/max(df.recording.nunique(),1):.1f}")
    log("")
    log("label distribution (1=dreaming, 2=without recall, 3=no experience):")
    log(df.dreaming.value_counts().sort_index().to_string())
    log("")
    dom = df.groupby(["subject", "recording"]).dreaming.first()
    dom = dom.groupby("subject").agg(
        lambda x: x.value_counts(normalize=True).max())
    log(f"within-subject label dominance: median {dom.median():.3f} "
        f"(DEED 0.750)")
    log(f"subjects >70% one class      : {dom.gt(0.7).mean():.3f} "
        f"(DEED 0.38)")
    log("")
    if skipped:
        log(f"skipped: {len(skipped)}")
        for b, w in skipped[:10]:
            log(f"   {b}: {w}")

    with open(os.path.join(OUT, "tononi_audit.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
