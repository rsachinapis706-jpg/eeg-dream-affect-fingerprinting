#!/bin/bash
# CBraMod with its pretraining input convention (EEG / 100 uV), round 3,
# Reviewer 2, comment 9. GPU queue, one run at a time, in order of priority:
# the Sleep-EDF control, then DEED pretrained and without pretraining at the
# first seed, then the other two seeds.
cd "/e/Dream GAN/analysis"
PY=/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CBRAMOD_SCALE=uv100
L=out/cbramod_scaling_stdout.txt
echo "start $(date)" > $L
run() { echo "== $* $(date)" >> $L; "$@" >> $L 2>&1 || echo "FAILED $* $(date)" >> $L; }
run $PY 49_cbramod_expanded.py
for s in 20260727 20260728 20260729; do
  run $PY 76_cbramod_diagnostics.py --config revised --seed $s
  run $PY 76_cbramod_diagnostics.py --config scratch_revised --seed $s
done
echo "EXIT 0 $(date)" >> $L
