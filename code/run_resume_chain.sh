#!/bin/bash
# Resume the CPU jobs stopped by the session shutdown (1 Oct 2026), one at a
# time so memory is not exhausted: Table 5 forest nulls to 200, PCD
# recomputation, density re-run with the corrected bootstrap, manifests.
cd "E:/Dream GAN/analysis"
PY=/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python
T5_JOBS=8 $PY -u 65_config_table.py > out/run_config_table_ext200b_stdout.log 2>&1
echo "table 5 done $(date) exit $?"
$PY -u 85_pcd_extra.py cbramod nulldep noise > out/run_pcd_extra_stdout.log 2>&1
echo "pcd extra done $(date) exit $?"
$PY -u 75_density_sensitivity.py > out/run_density_rerun_stdout.log 2>&1
echo "density done $(date) exit $?"
$PY -u 89_manifests.py > out/run_manifests.log 2>&1
echo "manifests done $(date) exit $?"
