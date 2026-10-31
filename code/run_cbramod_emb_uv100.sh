#!/bin/bash
# Frozen CBraMod embeddings with the pretraining input convention (EEG / 100 uV):
# extraction and within-participant test (15), then PCD on the embeddings (85).
cd "/e/Dream GAN/analysis"
PY=/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CBRAMOD_SCALE=uv100
L=out/cbramod_emb_uv100_stdout.txt
echo "start $(date)" > $L
run() { echo "== $* $(date)" >> $L; "$@" >> $L 2>&1 || echo "FAILED $* $(date)" >> $L; }
run $PY 15_cbramod_within_subject.py
run $PY 85_pcd_extra.py cbramod
echo "EXIT 0 $(date)" >> $L
