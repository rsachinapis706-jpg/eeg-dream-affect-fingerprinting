#!/bin/bash
# Partition sensitivity of the participant-separated results (114).
cd "E:/Dream GAN/analysis"
export PART_JOBS=3 PART_R=20
/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python -u -W ignore \
  114_partition_sensitivity.py > out/run_partition_sensitivity_stdout.log 2>&1
echo "partition exit $? $(date)" >> out/run_partition_sensitivity_stdout.log
