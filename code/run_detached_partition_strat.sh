#!/bin/bash
# Partition sensitivity with class-stratified random participant partitions.
cd "E:/Dream GAN/analysis"
export PART_JOBS=3 PART_R=20
/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python -u -W ignore \
  114_partition_sensitivity.py stratified > out/run_partition_sensitivity_strat_stdout.log 2>&1
echo "partition-stratified exit $? $(date)" >> out/run_partition_sensitivity_strat_stdout.log
