#!/bin/bash
# DEED null re-randomisations with the fold partition fixed independently of
# the labelling under test (110_pcd_stratified.py deednull_fixed). Resumable.
cd "E:/Dream GAN/analysis"
export PCD_JOBS=3 PCD_R_NULL=100
/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python -W ignore \
  110_pcd_stratified.py deednull_fixed >> out/run_pcd_strat_stdout.log 2>&1
echo "fixed-folds chain exit $? $(date)" >> out/run_pcd_strat_stdout.log
