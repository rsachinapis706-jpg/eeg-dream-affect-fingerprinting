#!/bin/bash
# Round 3 queue 3: group-aware conformal (130) once the label-free 77 run is
# complete (marked by the start of 85 in the PCD chain log).
cd "/e/Dream GAN/analysis"
PY=/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
until grep -q "85_pcd_extra.py" out/pcd_labelfree_stdout.txt 2>/dev/null; do sleep 30; done
echo "group conformal start $(date)" >> out/round3_queue.log
$PY 130_group_conformal.py > out/conformal_group_stdout.txt 2>&1
echo "EXIT $?" >> out/conformal_group_stdout.txt
echo "group conformal done $(date)" >> out/round3_queue.log
