#!/bin/bash
# Round 3 queue 4: label sensitivity (125) and restricted nulls (129) start as
# soon as the PCD chain frees its cores, alongside 120's permutations. Neither
# needs 120 to finish: 129 reads only 120's observed values, already written.
# Replaces the waiting halves of run_round3_queue.sh and run_round3_queue2.sh.
cd "/e/Dream GAN/analysis"
PY=/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
until grep -q "^EXIT" out/pcd_labelfree_stdout.txt 2>/dev/null; do sleep 30; done
echo "queue4 start $(date)" >> out/round3_queue.log
(
  LABEL_JOBS=5 $PY 125_label_sensitivity.py run > out/label_sensitivity_stdout.txt 2>&1 \
    && $PY 125_label_sensitivity.py summary >> out/label_sensitivity_stdout.txt 2>&1
  echo "EXIT $?" >> out/label_sensitivity_stdout.txt
  echo "label sensitivity done $(date)" >> out/round3_queue.log
) &
(
  RESTR_JOBS=4 $PY 129_restricted_nulls.py > out/restricted_nulls_stdout.txt 2>&1
  echo "EXIT $?" >> out/restricted_nulls_stdout.txt
  echo "restricted nulls done $(date)" >> out/round3_queue.log
) &
wait
echo "queue4 done $(date)" >> out/round3_queue.log
