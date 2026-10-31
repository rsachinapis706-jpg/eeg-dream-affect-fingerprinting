#!/bin/bash
# Round 3: once the first permutation batch of 125 (label sensitivity) and 129
# (restricted nulls) is on disk, restart each so that later batches are sized
# by the exceedances each statistic still lacks (no permutation beyond what
# the sequential stopping rule needs). Both resume from their raw CSV, so
# nothing already computed is lost; at most the first minute of the second
# batch is recomputed. p-values do not depend on batch size.
cd "/e/Dream GAN/analysis"
PY=/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

kill_script() {  # kill the python running script $1 and its workers
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { \$_.CommandLine -match '$1' } | % { taskkill /PID \$_.ProcessId /T /F }"
}

restart() {  # $1 raw csv, $2 script, $3 stdout, $4 command to relaunch
  local t0
  t0=$(stat -c %Y "$1" 2>/dev/null || echo 0)
  until [ "$(stat -c %Y "$1" 2>/dev/null || echo 0)" -gt "$t0" ]; do sleep 30; done
  echo "restart $2 after first batch $(date)" >> out/round3_queue.log
  kill_script "$2"
  until grep -q "^EXIT" "$3"; do sleep 5; done
  sed -i '/^EXIT/d' "$3"
  echo "--- restarted $(date) with needs-sized batches ---" >> "$3"
  eval "$4"
  echo "EXIT $?" >> "$3"
  echo "$2 done $(date)" >> out/round3_queue.log
}

restart out/label_sensitivity_raw.csv 125_label_sensitivity.py \
  out/label_sensitivity_stdout.txt \
  'LABEL_JOBS=5 $PY 125_label_sensitivity.py run >> out/label_sensitivity_stdout.txt 2>&1 && $PY 125_label_sensitivity.py summary >> out/label_sensitivity_stdout.txt 2>&1' &
restart out/restricted_nulls_raw.csv 129_restricted_nulls.py \
  out/restricted_nulls_stdout.txt \
  'RESTR_JOBS=4 $PY 129_restricted_nulls.py >> out/restricted_nulls_stdout.txt 2>&1' &
wait
