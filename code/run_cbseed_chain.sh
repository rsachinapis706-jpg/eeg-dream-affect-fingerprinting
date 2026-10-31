#!/bin/bash
# Seed sensitivity for CBraMod fine-tuning (Reviewer 1, comment 9): two more
# initialisation / data-order seeds for the revised settings, pretrained and
# randomly initialised. Two runs share the GPU at a time.
cd "E:/Dream GAN"
PY=/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python
# Limit CPU threads so the GPU runs do not starve CPU jobs running alongside;
# the computation is on the GPU and the thread count does not change results.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for SEED in 20260728 20260729; do
  $PY analysis/76_cbramod_diagnostics.py --config revised --seed $SEED >> analysis/out/run_cbseed_revised_$SEED.log 2>&1 &
  $PY analysis/76_cbramod_diagnostics.py --config scratch_revised --seed $SEED >> analysis/out/run_cbseed_scratch_revised_$SEED.log 2>&1 &
  wait
  echo "seed $SEED done $(date)"
done
