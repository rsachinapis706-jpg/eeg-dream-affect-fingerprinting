#!/bin/bash
# Selection-aware (nested) estimate for the Table 5 configuration search.
cd "E:/Dream GAN/analysis"
export NEST_JOBS=3
/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python -u -W ignore \
  112_nested_selection.py > out/run_nested_selection_stdout.log 2>&1
echo "nested exit $? $(date)" >> out/run_nested_selection_stdout.log
