#!/bin/bash
# Round 3 queue 2: restricted nulls (129) once 120's permutations are done.
cd "/e/Dream GAN/analysis"
PY=/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
until grep -q "^EXIT" out/primary_random_stdout.txt 2>/dev/null; do sleep 30; done
echo "restricted nulls start $(date)" >> out/round3_queue.log
RESTR_JOBS=4 $PY 129_restricted_nulls.py > out/restricted_nulls_stdout.txt 2>&1
echo "EXIT $?" >> out/restricted_nulls_stdout.txt
echo "restricted nulls done $(date)" >> out/round3_queue.log
