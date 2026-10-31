#!/bin/bash
# Start the revised CBraMod configuration once the other two GPU runs finish.
cd "E:/Dream GAN"
PY=/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python
while powershell.exe -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*76_cbramod*' }) { exit 0 } else { exit 1 }"; do sleep 60; done
echo "published+scratch done $(date)"
$PY analysis/76_cbramod_diagnostics.py --config revised > analysis/out/run_cbdiag_revised.log 2>&1
echo "revised done $(date) exit $?"
