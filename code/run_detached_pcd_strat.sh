#!/bin/bash
# Stratified PCD validation (110_pcd_stratified.py): simulations, Sleep-EDF,
# then the DEED null extended to 100 re-randomisations. Each piece resumes.
cd "E:/Dream GAN/analysis"
PY=/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python
export PCD_JOBS=3
L=out/run_pcd_strat_stdout.log
$PY -W ignore 110_pcd_stratified.py sims >> $L 2>&1 && \
$PY -W ignore 110_pcd_stratified.py sleepedf >> $L 2>&1 && \
PCD_R_NULL=100 $PY -W ignore 110_pcd_stratified.py deednull >> $L 2>&1
echo "strat chain exit $? $(date)" >> $L
