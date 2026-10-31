#!/bin/bash
# Table 5 extension to 1000 permutations, resumable; launched as an
# independent process so that it is not tied to an interactive session.
cd "E:/Dream GAN/analysis"
export T5_JOBS=7
/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python -u -W ignore \
  65_config_table.py >> out/run_config_table_B1000_stdout.log 2>&1
echo "table5 exit $? $(date)" >> out/run_config_table_B1000_stdout.log
