"""
Builds the supplementary archive uploaded with the resubmission so that the
analyses can be checked during review (Reviewer 1, comment 12; Reviewer 2,
reproducibility).

Contents
  code/        every numbered analysis script and its helpers
  results/     every CSV, text and log file written by those scripts
  supplement/  cohort manifests, fold assignments, window-level predictions
  predictions/ per-window outputs of the factorial, PCD, density and
               foundation-model analyses
  features/    the DEED and Tononi feature tables the analyses start from
  environment/ the pinned package list
  README.txt   what is where, how to run the numerical audit
Raw signals are not redistributed; they are public (sources in the README),
and the scripts that read them are included.
"""
import glob
import hashlib
import io
import os
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DEST = os.path.join("E:" + os.sep, "Dream_Analysis",
                    "Access-2026-50879_supplementary_code_and_results.zip")

README = """CONFIDENTIAL - provided for the peer review of Access-2026-50879 only.
Please do not redistribute. A public deposit with a DOI will follow acceptance.

Supplementary material for Access-2026-50879 (revised)
"High Window-Level Accuracy Without Affect Information: Evaluation Protocol,
Recording and Participant Fingerprinting in EEG Dream-Emotion Decoding"

code/         Numbered analysis scripts. Each begins with a header stating the
              question it answers, its design and its outputs. Scripts 65-152
              were written or revised in the two revisions; 120-152 for
              this one.
results/      Every result file (CSV, text, log) the scripts wrote. Every
              number in the manuscript is located in one of these files.
supplement/   manifest_deed.csv (every analysed DEED window with participant,
              recording, night and label), folds_deed.csv (the test fold of
              every window under each partition and fold assignment of
              Table 4a), predictions_deed.parquet (out-of-fold predictions of
              the headline and balanced logistic-regression configurations),
              manifest_tononi.csv, manifest_sleepedf.csv.
predictions/  Window-level outputs: protocol factorial, PCD (p-values,
              confidence, conformal sets; pcd_strat_* with the pooled,
              recording, participant and input references; pcd_leaky_* under
              participant-separated and window-level folds), density sweep,
              and every CBraMod fold ([true; predicted] per held-out
              participant). CBraMod runs record, per fold, the validation
              participants actually used (val_ids in
              results/cbramod_diag_*_folds.csv): runs resumed after an
              interruption draw them afresh from that point, and GPU training
              is not bit-reproducible.
features/     The DEED (6-channel, 108 features) and Tononi feature tables.
environment/  pip freeze of the environment used (Python 3.12.5,
              scikit-learn 1.9.0, NumPy 2.5.2, SciPy 1.16.3, pandas 2.3.3,
              MNE 1.11.0, PyTorch 2.6.0).

MANIFEST.csv lists every file in this archive with its size in bytes and
its SHA-256 checksum, so that the contents can be verified.

Provenance: results/provenance_manuscript.csv and
results/provenance_letter.csv assign every number in the manuscript and in
the response letter to its result file, row and column (code/132_provenance.py);
results/provenance_summary.csv counts how each was assigned. Assignments
made by reading a number in its sentence, and numbers computed in the text,
are listed in code/provenance_overrides.py with the computation that
reproduces them; 132_provenance.py re-evaluates every one against the result
files. Figures that no result table held are written by
code/155_within_checks.py (within-participant effect sizes and paired-test
power).

Numerical audit: results/final_audit.csv lists every decimal value in the
manuscript with its line and whether it was located in the result files;
code/95_audit.py regenerates it.
Reference check: results/reference_audit.csv and results/reference_audit.txt
(code/108_verify_refs_doi.py) compare every bibliography entry with the record
registered for its DOI at Crossref or DataCite. Entries without a DOI, and
entries whose registered record differs from the publisher's own page, were
checked against that page; results/reference_manual_checks.csv records the
URL, date and outcome of each such check.

Data sources (not redistributed): DEED (Liu et al.); Sleep-EDF Expanded
(PhysioNet); the Tononi serial-awakenings corpus (figshare); the DREAM
database catalogue, version 9. See the manuscript's Data Availability
Statement.
"""


# superseded or partial versions of a result file, never packed
SUPERSEDED = (".pre_fix", ".prev", ".B100", ".B200", "_smoke", "_partial")
MANIFEST = []


def add(z, path, arc):
    if os.path.exists(path):
        z.write(path, arc)
        with open(path, "rb") as fh:
            MANIFEST.append((arc, os.path.getsize(path),
                             hashlib.sha256(fh.read()).hexdigest()))
        return 1
    return 0


def main():
    n = 0
    MANIFEST.clear()
    with zipfile.ZipFile(DEST, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", README)
        for f in sorted(glob.glob(os.path.join(HERE, "*.py"))):
            n += add(z, f, "code/" + os.path.basename(f))
        for f in sorted(glob.glob(os.path.join(HERE, "*.sh"))):
            n += add(z, f, "code/" + os.path.basename(f))
        for pat in ("*.csv", "*.txt", "*.log"):
            for f in sorted(glob.glob(os.path.join(OUT, pat))):
                b = os.path.basename(f)
                if b.startswith("stale_") or any(x in b for x in SUPERSEDED):
                    continue
                n += add(z, f, "results/" + b)
        for f in sorted(glob.glob(os.path.join(OUT, "supplement", "*"))):
            n += add(z, f, "supplement/" + os.path.basename(f))
        for b in ("protocol_factorial_preds.parquet",
                  "pcd_revised_windows.parquet",
                  "pcd_extra_cbramod_windows.parquet",
                  "pcd_extra_noise_windows.parquet",
                  "density_sensitivity_preds.parquet",
                  "pcd_strat_windows_deed.parquet",
                  "pcd_strat_windows_sleepedf.parquet",
                  "pcd_leaky_windows.parquet",
                  # round 3
                  "protocol_factorial_preds_full.parquet",
                  "primary_random_preds.parquet",
                  "primary_random_nested_preds.parquet",
                  "label_sensitivity_preds.parquet",
                  "route_contrasts_preds.parquet",
                  "pcd_extra_cbramod_uv100_windows.parquet"):
            n += add(z, os.path.join(OUT, b), "predictions/" + b)
        for f in sorted(glob.glob(os.path.join(OUT, "cbramod_diag_*_pred.npy"))):
            n += add(z, f, "predictions/cbramod/" + os.path.basename(f))
        for b in ("features_epoch_filtered.parquet", "tononi_features.parquet"):
            n += add(z, os.path.join(OUT, b), "features/" + b)
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True).stdout
        z.writestr("environment/requirements_pinned.txt", freeze)
        z.writestr("MANIFEST.csv", "path,bytes,sha256\n" + "".join(
            f"{a},{b},{h}\n" for a, b, h in MANIFEST))
    size = os.path.getsize(DEST) / 1e6
    print(f"wrote {DEST}: {n} files, {size:.1f} MB")


if __name__ == "__main__":
    main()
