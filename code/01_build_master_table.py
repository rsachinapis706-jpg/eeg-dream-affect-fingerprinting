"""
Build the DEED master table by joining EEG segments to dream reports,
film condition, and session timing.

Principles:
  - Nothing is invented. Unjoinable rows keep NaN and are counted, not dropped silently.
  - Every join is confidence-flagged so downstream analysis can require `join_ok`.
  - All coverage numbers are written to audit.txt for the paper's data section.

Filename schema (verified empirically, 533/533 files, 8 underscore fields):
    G _ <Online_id> _ <M1|M2|M3> _ <E0..E5> _ <R#> _ <stage> _ raw _ ref.mat
    Online_id = S + 3-digit subject + 1-digit night   (34 subjects, 78 nights)
    M         = night-level condition, constant within night (candidate: pre-sleep film)
    E         = reported dream-emotion code
    R         = awakening index within the night
    stage     = sleep stage tag (N1/N2/N3/REM/W/nan)
"""
import os
import re
import glob
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAT_DIR = os.path.join(ROOT, "dream_eeg", "Dream EEG with emotion labels")
OUT_DIR = os.path.join(ROOT, "analysis", "out")
os.makedirs(OUT_DIR, exist_ok=True)

log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


# ---------------------------------------------------------------- 1. segments
def load_segments():
    paths = sorted(glob.glob(os.path.join(MAT_DIR, "*.mat")))
    if not paths:
        raise FileNotFoundError(f"No .mat files under {MAT_DIR}")

    rows = []
    for p in paths:
        stem = os.path.basename(p)[:-4]
        f = stem.split("_")
        if len(f) != 8:
            raise ValueError(f"Unexpected filename schema: {stem}")
        rows.append(
            {
                "path": p,
                "oid": f[1],
                "subject": f[1][1:4],          # 3-digit subject -> LOSO grouping key
                "night": f[1][4],
                "M": f[2],
                "E": f[3],
                "awakening": int(re.sub(r"\D", "", f[4])),
                "stage": f[5],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. reports
def load_reports():
    er = pd.read_excel(os.path.join(ROOT, "dream_eeg", "Emotional_ratings_excel_filesfiles.xlsx"))
    # Online_id/Date/PSG_number are merged cells in the source workbook: only the
    # first row of each night carries a value. Forward-fill restores night identity.
    for c in ["Online_id", "Date", "PSG_number"]:
        er[c] = er[c].ffill()
    er = er.rename(columns={"Online_id": "oid"})
    # Order within night = awakening order. Stable sort preserves sheet order,
    # which follows chronological REM period.
    er["report_seq"] = er.groupby("oid").cumcount() + 1
    return er


# ---------------------------------------------------------------- 3. context
def load_context():
    vl = pd.read_excel(os.path.join(ROOT, "dream_eeg", "Video_list.xlsx")).rename(
        columns={"Online_id": "oid"}
    )
    st = pd.read_excel(
        os.path.join(ROOT, "dream_eeg", "Status_identification_of_each_stage_of_EEG.xlsx")
    ).rename(columns={"Online_id": "oid"})
    return vl, st


def main():
    seg = load_segments()
    rep = load_reports()
    vid, sta = load_context()

    log("=" * 68)
    log("DEED MASTER TABLE AUDIT")
    log("=" * 68)
    log(f"EEG segments            : {len(seg)}")
    log(f"Nights (Online_id)      : {seg.oid.nunique()}")
    log(f"Subjects (LOSO groups)  : {seg.subject.nunique()}")
    log(f"Dream report rows       : {len(rep)}  ({rep.oid.nunique()} nights)")
    log("")

    log("-- Sleep stage tag distribution --")
    log(seg.stage.value_counts().to_string())
    log("")
    log("-- Reported emotion distribution --")
    log(seg.E.value_counts().sort_index().to_string())
    log("")
    log("-- Night-level condition M --")
    log(seg.M.value_counts().to_string())
    within = seg.groupby("oid").M.nunique()
    log(f"Nights where M varies within night: {(within > 1).sum()} of {len(within)}")
    log("")

    # ---- segment <-> report alignment, per night -------------------------
    seg["seg_seq"] = seg.sort_values("awakening").groupby("oid").cumcount() + 1
    nseg = seg.oid.value_counts().rename("n_seg")
    nrep = rep.oid.value_counts().rename("n_rep")
    align = pd.concat([nseg, nrep], axis=1).fillna(0).astype(int)
    align["match"] = align.n_seg == align.n_rep

    ok_nights = set(align[align.match & (align.n_seg > 0)].index)
    log("-- Segment/report alignment by night --")
    log(f"Nights with equal counts (joinable) : {len(ok_nights)}")
    log(f"Nights with unequal counts          : {int((~align.match).sum())}")
    log(f"Nights with segments but no reports : {int((align.n_rep == 0).sum())}")
    log("")

    # Positional join is only defensible where per-night counts agree.
    seg["join_ok"] = seg.oid.isin(ok_nights)
    master = seg.merge(
        rep[["oid", "report_seq", "Dream_content", "Dream_emotion",
             "REM_period_start_time", "End_time_of_REM_period"]],
        left_on=["oid", "seg_seq"],
        right_on=["oid", "report_seq"],
        how="left",
    )
    master.loc[~master.join_ok, ["Dream_content", "Dream_emotion"]] = pd.NA

    # Film condition comes from the filename field M, not from Video_list.
    # Rationale (verified in this script, see audit output): M is constant within
    # night, and on the 47 nights where Video_list is unambiguous (exactly one film
    # row) M maps to film valence at 95.7% purity -- M1=Negative, M2=Neutral,
    # M3=Positive. Video_list itself carries 10 nights with conflicting duplicate
    # rows, which would silently multiply segments if merged directly.
    M_TO_FILM = {"M1": "Negative", "M2": "Neutral", "M3": "Positive"}
    master["film"] = master.M.map(M_TO_FILM)

    # Report the purity check rather than asserting the mapping is exact.
    vid_counts = vid.oid.value_counts()
    vid_unamb = vid[vid.oid.isin(vid_counts[vid_counts == 1].index)]
    chk = (
        seg[["oid", "M"]]
        .drop_duplicates()
        .merge(vid_unamb[["oid", "Video_type"]], on="oid", how="inner")
    )
    if len(chk):
        ct = pd.crosstab(chk.M, chk.Video_type)
        agree = sum(ct.loc[m, M_TO_FILM[m]] for m in ct.index if M_TO_FILM[m] in ct.columns)
        log("-- M -> film valence verification (unambiguous nights only) --")
        log(ct.to_string())
        log(f"Nights checked: {len(chk)} | agreement: {agree}/{len(chk)} = {agree/len(chk):.1%}")
        log(f"Nights with conflicting Video_list rows (excluded): "
            f"{int((vid_counts > 1).sum())}")
        log("")

    master = master.merge(
        vid_unamb[["oid", "Video_name"]].drop_duplicates("oid"), on="oid", how="left"
    )
    # Status sheet contains 2 exact duplicate rows (S0392, S0373); dedupe or the
    # merge silently multiplies those nights' segments.
    sta_cols = ["oid", "Start_measuring_dream", "Resting_state_before_movie",
                "Resting_state_after_movie"]
    n_dup_sta = int(sta.oid.duplicated().sum())
    master = master.merge(sta[sta_cols].drop_duplicates("oid"), on="oid", how="left")
    log(f"Duplicate rows dropped from status sheet: {n_dup_sta}")
    log("")

    # The master table must stay one row per EEG segment. A join that inflates
    # this is a bug, not a finding -- fail loudly rather than analyse phantom rows.
    assert len(master) == len(seg), (
        f"Row inflation: {len(master)} rows from {len(seg)} segments"
    )

    has_text = master.Dream_content.notna()
    has_film = master.film.notna()

    log("-- Coverage of the joined master table --")
    log(f"Rows (== segments)              : {len(master)}")
    log(f"Segments with dream narrative   : {int(has_text.sum())} / {len(master)}")
    log(f"Segments with film condition    : {int(has_film.sum())} / {len(master)}")
    log(f"Segments with BOTH              : {int((has_text & has_film).sum())}")
    log(f"Subjects retained (text+film)   : {master[has_text & has_film].subject.nunique()}")
    log("")

    log("-- Film condition x reported emotion (ALL segments) --")
    log(pd.crosstab(master.film, master.E).to_string())
    log("")
    log("-- Film condition x reported emotion (row %) --")
    log((pd.crosstab(master.film, master.E, normalize="index") * 100).round(1).to_string())
    log("")
    log("-- Sleep stage x reported emotion (confound check) --")
    log(pd.crosstab(master.stage, master.E).to_string())
    log("")

    master.to_csv(os.path.join(OUT_DIR, "master_table.csv"), index=False)
    with open(os.path.join(OUT_DIR, "audit.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines))
    log(f"Wrote {OUT_DIR}\\master_table.csv and audit.txt")


if __name__ == "__main__":
    main()
