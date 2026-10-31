"""
Process Sleep-EDF Expanded (197 recordings) with the existing feature pipeline.

Every Sleep-EDF result so far rests on 8 recordings from the 1.0.0 release. The
expanded release has 197 recordings from ~78 subjects and was already downloaded
but never opened. Reprocessing upgrades the project's most load-bearing result --
the positive control that makes every null interpretable.

Three format differences from the 1.0.0 release
-----------------------------------------------
1. Hypnograms are EDF+ ANNOTATIONS with variable-duration blocks, not stage codes
   stored as channel data. They must be expanded to fixed 30 s epochs.

2. Recordings run ~22 h and begin with many hours of pre-sleep Wake (the first
   annotation in SC4001E0 is 30,630 s of Wake). Leaving that in would make Wake
   ~80% of the data and inflate accuracy trivially. Standard practice, applied
   here: crop to the sleep period +/- 30 min.

3. Filenames encode subject AND night: SC4001E0 -> subject 00, night 1;
   SC4002E0 -> subject 00, night 2. Grouping LOSO on the filename would put the
   same person in train and test. The grouping key is the subject digits.
   This is the same trap as DEED's S0392.

Feature definitions are unchanged from 09_sleepedf_validation.py so the new
results are directly comparable with everything already computed.
"""
import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import mne
from scipy import signal as sps
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
EXP = os.path.join(ROOT, "Sleep-EDF Expanded",
                   "sleep-edf-database-expanded-1.0.0")

FS = 100.0          # native Sleep-EDF rate; matches 09_sleepedf_validation
EPOCH_S = 20
EPOCH_N = int(FS * EPOCH_S)
SCORING_S = 30          # hypnogram scoring resolution
CROP_MIN = 30           # minutes of wake retained either side of the sleep period
ARTIFACT_UV = 300.0

_trapz = getattr(np, "trapezoid", None) or np.trapz
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
         "sigma": (12.0, 16.0), "beta": (16.0, 30.0), "gamma": (30.0, 45.0)}
_b, _a = sps.butter(4, [0.5 / (FS / 2), 45.0 / (FS / 2)], btype="band")

STAGE = {
    "Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
    "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM",
}


def hjorth(x):
    dx = np.diff(x); ddx = np.diff(dx)
    v0 = np.var(x) + 1e-12; v1 = np.var(dx) + 1e-12; v2 = np.var(ddx) + 1e-12
    mob = np.sqrt(v1 / v0)
    return mob, np.sqrt(v2 / v1) / (mob + 1e-12)


def epoch_features(ep):
    """Identical definitions to the 8-recording pipeline."""
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


def parse_id(stem):
    """SC4001E0 -> ('SC00', '1');  ST7022J0 -> ('ST02', '2').

    The cohort prefix is part of the subject key. Sleep-cassette and
    sleep-telemetry number their participants independently, so SC4011E0 and
    ST7011J0 are DIFFERENT people who both carry digits '01'. Dropping the
    prefix merges 21 pairs of unrelated participants into single LOSO groups
    and understates the cohort as 78 subjects instead of 100.
    """
    m = re.match(r"(S[CT])\d(\d{2})(\d)", stem)
    if not m:
        return stem, "0"
    return m.group(1) + m.group(2), m.group(3)


def stage_series(hyp_path, n_scoring):
    """Expand variable-duration EDF+ annotations to one label per 30 s epoch."""
    ann = mne.read_annotations(hyp_path)
    labels = np.full(n_scoring, None, dtype=object)
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        st = STAGE.get(desc)
        if st is None:
            continue                       # 'Sleep stage ?' and movement
        i0 = int(round(onset / SCORING_S))
        i1 = int(round((onset + dur) / SCORING_S))
        labels[max(0, i0):min(n_scoring, i1)] = st
    return labels


def process(psg_path, hyp_path):
    """Crop to the sleep period BEFORE loading samples.

    A full recording is ~22 h at 100 Hz. Loading all of it in every worker
    exhausted system memory. Annotations are read first so only the sleep
    period is ever materialised, which also discards the many hours of
    pre-sleep Wake we would drop anyway.
    """
    stem = os.path.basename(psg_path).split("-")[0]
    subject, night = parse_id(stem)
    try:
        raw = mne.io.read_raw_edf(psg_path, preload=False)
        eeg = [c for c in raw.ch_names if c.startswith("EEG")]
        if not eeg:
            return []
        raw.pick(eeg)

        total_s = raw.times[-1]
        n_scoring = int(total_s / SCORING_S)
        labels = stage_series(hyp_path, n_scoring)

        asleep = np.where([l is not None and l != "W" for l in labels])[0]
        if len(asleep) == 0:
            return []
        pad = int(CROP_MIN * 60 / SCORING_S)
        lo = max(0, asleep[0] - pad)
        hi = min(n_scoring, asleep[-1] + pad + 1)

        raw.crop(tmin=lo * SCORING_S,
                 tmax=min(total_s, hi * SCORING_S))
        raw.load_data()
        if raw.info["sfreq"] != FS:
            raw.resample(FS)
        data = sps.filtfilt(_b, _a, raw.get_data() * 1e6, axis=-1)
        del raw
    except Exception:
        return []

    rows = []
    for i in range(lo, hi):
        st = labels[i]
        if st is None:
            continue
        s0 = int((i - lo) * SCORING_S * FS)      # index within the cropped span
        ep = data[:, s0:s0 + EPOCH_N]
        if ep.shape[1] < EPOCH_N or np.abs(ep).max() > ARTIFACT_UV:
            continue
        if np.allclose(ep.std(axis=1), 0):
            continue
        f = epoch_features(ep)
        f.update(recording=stem, subject=subject, night=night,
                 stage=st, epoch=i)
        rows.append(f)
    del data
    return rows


def main():
    pairs = []
    for sub in ["sleep-cassette", "sleep-telemetry"]:
        psgs = sorted(glob.glob(os.path.join(EXP, sub, "*PSG.edf")))
        hyps = sorted(glob.glob(os.path.join(EXP, sub, "*Hypnogram.edf")))
        hmap = {os.path.basename(h).split("-")[0][:6]: h for h in hyps}
        for p in psgs:
            k = os.path.basename(p).split("-")[0][:6]
            if k in hmap:
                pairs.append((p, hmap[k]))

    print(f"matched {len(pairs)} PSG/hypnogram pairs", flush=True)
    res = Parallel(n_jobs=6, verbose=5, max_nbytes="50M")(
        delayed(process)(p, h) for p, h in pairs)
    rows = [r for sub in res for r in sub]

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(OUT, "sleepedf_expanded_features.parquet"),
                  index=False)

    txt = "\n".join([
        "=" * 70, "SLEEP-EDF EXPANDED", "=" * 70,
        f"recordings matched : {len(pairs)}",
        f"recordings yielding data : {df.recording.nunique()}",
        f"SUBJECTS (LOSO groups)   : {df.subject.nunique()}",
        f"windows                  : {len(df)}",
        f"cropped to sleep period +/- {CROP_MIN} min",
        "", "stage distribution:", df.stage.value_counts().to_string(),
        "", "recordings per subject:",
        df.groupby('subject').recording.nunique().value_counts().to_string(),
        "", "-- comparison with the 8-recording release --",
        f"{'':<26}{'expanded':>12}{'1.0.0':>12}",
        f"{'recordings':<26}{df.recording.nunique():>12}{8:>12}",
        f"{'subjects':<26}{df.subject.nunique():>12}{8:>12}",
        f"{'windows':<26}{len(df):>12}{15140:>12}",
    ])
    print(txt, flush=True)
    with open(os.path.join(OUT, "sleepedf_expanded_audit.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)


if __name__ == "__main__":
    main()
