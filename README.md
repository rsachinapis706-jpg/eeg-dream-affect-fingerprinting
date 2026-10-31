# High Window-Level Accuracy Without Affect Information

Code and results for:

> Sachin R, Jaganath R, S T Yeshwanth and Abhishek S, "High Window-Level Accuracy Without Affect Information: Evaluation Protocol, Recording and Participant Fingerprinting in EEG Dream-Emotion Decoding," *IEEE Access*, 2026 (accepted; manuscript Access-2026-50879).

Two published studies report 92–94% accuracy for classifying dream emotion in the Dream Emotion Evaluation Dataset (DEED) without stating whether validation separated participants. This repository holds every analysis script and result file behind the paper, which shows that window-level evaluation on DEED can reach that range without information about affect, measures the protocol contrasts that account for the window-level advantage, and proposes Permutation-Calibrated Decodability (PCD), a null-referenced abstention rule.

## Contents

| Folder | What it holds |
|---|---|
| `code/` | Numbered analysis scripts. Each begins with a header stating the question it answers, its design and its outputs. |
| `results/` | Every result file (CSV, text, log) the scripts wrote. Every number in the manuscript is located in one of these files. |
| `supplement/` | `manifest_deed.csv` (every analysed DEED window with participant, recording, night and label), `folds_deed.csv` (the test fold of every window under each partition and fold assignment of Table 5(a) of the paper), `predictions_deed.parquet` (out-of-fold predictions of the headline and balanced logistic-regression configurations), `manifest_tononi.csv`, `manifest_sleepedf.csv`. |
| `predictions/` | Window-level outputs: protocol factorial, PCD (p-values, confidence, conformal sets; `pcd_strat_*` with the pooled, recording, participant and input references; `pcd_leaky_*` under participant-separated and window-level folds), density sweep, and every CBraMod fold (`[true; predicted]` per held-out participant). |
| `features/` | The DEED (6-channel, 108 features) and Tononi feature tables. |
| `environment/` | `requirements_pinned.txt`, the pip freeze of the environment used. |
| `MANIFEST.csv` | Every file with its size in bytes and SHA-256 checksum. |

## Environment

Python 3.12.5, scikit-learn 1.9.0, NumPy 2.5.2, SciPy 1.16.3, pandas 2.3.3, MNE 1.11.0, PyTorch 2.6.0.

```bash
pip install -r environment/requirements_pinned.txt
```

Library versions matter here: `StratifiedGroupKFold` changed its fold-assignment algorithm between scikit-learn 1.4 and 1.9, which moves the two grouped rows of the protocol table by roughly one accuracy point. Linear-algebra threads are fixed at three throughout, because OpenBLAS uses a different kernel on one thread than on several and that moves logistic-regression solutions in the fourth decimal place. Multithreaded random-forest prediction can differ between runs by up to about 0.001 in balanced accuracy. CBraMod runs record, per fold, the validation participants actually used (`val_ids` in `results/cbramod_diag_*_folds.csv`); runs resumed after an interruption draw them afresh from that point, and GPU training is not bit-reproducible.

The scripts were run on Windows from `E:\Dream_Analysis`; data and output paths are set near the top of each script and need to be changed to your own locations.

## Data

The raw recordings are not redistributed here. Obtain them from their sources:

- **DEED** — Liu et al., *DEED: a dataset for dream-related emotion research*, doi:10.1101/2022.09.19.508475.
- **Sleep-EDF Expanded** — PhysioNet (Kemp et al., 2000).
- **Tononi serial-awakenings corpus** — Siclari et al., *The neural correlates of dreaming*, Nature Neuroscience, 2017, doi:10.1038/nn.4545; data on figshare, version 3, doi:10.26180/23306054.v3 (CC BY 4.0).
- **DREAM database**, version 9 — doi:10.26180/22133105.v9.

## Provenance and audit

- `results/provenance_manuscript.csv` assigns every number in the manuscript to its result file, row and column (`code/132_provenance.py`); `results/provenance_summary.csv` counts how each was assigned. Numbers read from their sentence, and numbers computed in the text, are listed in `code/provenance_overrides.py` with the computation that reproduces them. Figures that no result table held are written by `code/155_within_checks.py`.
- `results/final_audit.csv` lists every decimal value in the manuscript with its line and whether it was located in the result files; `code/95_audit.py` regenerates it.
- `results/reference_audit.csv` and `results/reference_audit.txt` (`code/108_verify_refs_doi.py`) compare every bibliography entry with the record registered for its DOI at Crossref or DataCite; `results/reference_manual_checks.csv` records the URL, date and outcome of each manual check.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

MIT; see `LICENSE`. The license covers the code and the result files in this repository, not the source datasets, which remain under their own terms.
