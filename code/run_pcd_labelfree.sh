#!/bin/bash
# Round 3, Reviewer 2 comment 1: every PCD analysis re-run with folds drawn
# without labels (77.label_free_folds). Previous outputs are kept as
# out/stale_labelstrat_*. Order matters: 85 and 110 read 77's outputs, and 110
# asserts that its pooled p-values reproduce 77's.
cd "/e/Dream GAN/analysis"
PY=/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
L=out/pcd_labelfree_stdout.txt
echo "start $(date)" > $L
step() { echo "== $* $(date)" >> $L; "$@" >> $L 2>&1 || { echo "EXIT 1 at $*" >> $L; exit 1; }; }
step $PY 77_pcd_revised.py
step $PY 85_pcd_extra.py grid cbramod nulldep noise
PCD_R_NULL=100 PCD_JOBS=8 step $PY 110_pcd_stratified.py deed deednull sims sleepedf
step $PY 103_pcd_riskcov_diff.py
step $PY 104_pcd_riskcov_ci.py
step $PY 109_pcd_deednull_summary.py
echo "EXIT 0 $(date)" >> $L
