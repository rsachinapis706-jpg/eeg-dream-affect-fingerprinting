#!/bin/bash
# CPU queue, time-limited design. One job at a time, each using the whole CPU.
cd "E:/Dream GAN"
PY=/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python
while powershell.exe -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*72_random_label*' }) { exit 0 } else { exit 1 }"; do sleep 30; done
echo "ladder done $(date)"
$PY analysis/73_protocol_factorial.py      > analysis/out/run_protocol_factorial_stdout.log 2>&1; echo "factorial done $(date) exit $?"
$PY analysis/74_identity_controls.py       > analysis/out/run_identity_controls_stdout.log 2>&1; echo "identity controls done $(date) exit $?"
$PY analysis/71b_identity_grouping_fix.py  > analysis/out/run_identity_grouping_fix.log 2>&1; echo "table 9 fix done $(date) exit $?"
$PY analysis/65_config_table.py            > analysis/out/run_config_table_stdout.log 2>&1; echo "table 5 done $(date) exit $?"
$PY analysis/75_density_sensitivity.py     > analysis/out/run_density_sensitivity_stdout.log 2>&1; echo "density done $(date) exit $?"
$PY analysis/77_pcd_revised.py             > analysis/out/run_pcd_revised_stdout.log 2>&1; echo "pcd done $(date) exit $?"
