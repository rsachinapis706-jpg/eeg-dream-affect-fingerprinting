#!/bin/bash
# Second CPU queue: starts when the first queue has finished. The process
# filter is restricted to bash.exe so the polling command cannot match itself.
cd "E:/Dream GAN"
PY=/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python
while powershell.exe -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | Where-Object { \$_.CommandLine -like '*run_phase2_chain.sh*' }) { exit 0 } else { exit 1 }"; do sleep 30; done
echo "queue A done $(date)"
$PY analysis/78_power_revised.py > analysis/out/run_power_revised_stdout.log 2>&1; echo "power done $(date) exit $?"
