#!/bin/bash
# Round 3 compute queue.
#   73 full factorial: now, 4 workers (shares the CPU with 120)
#   125 label sensitivity: when the 120 chain has finished
cd "/e/Dream GAN/analysis"
PY=/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
echo "queue started $(date)" > out/round3_queue.log
wait_exit() { until grep -q "^EXIT" "$1" 2>/dev/null; do sleep 30; done; }

(
  FACT_JOBS=4 $PY 73_protocol_factorial.py --full > out/protocol_factorial_full_stdout.txt 2>&1
  echo "EXIT $?" >> out/protocol_factorial_full_stdout.txt
  echo "factorial done $(date)" >> out/round3_queue.log
) &

(
  wait_exit out/primary_random_stdout.txt
  echo "label sensitivity start $(date)" >> out/round3_queue.log
  LABEL_JOBS=8 $PY 125_label_sensitivity.py run > out/label_sensitivity_stdout.txt 2>&1 \
    && $PY 125_label_sensitivity.py summary >> out/label_sensitivity_stdout.txt 2>&1
  echo "EXIT $?" >> out/label_sensitivity_stdout.txt
  echo "label sensitivity done $(date)" >> out/round3_queue.log
) &
wait
echo "queue done $(date)" >> out/round3_queue.log
