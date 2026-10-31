#!/bin/bash
# Final integration, run after the CBraMod seed runs (run_cbseed_chain.sh),
# the Table 5 extension (65_config_table.py), the stratified-PCD runs
# (110_pcd_stratified.py), the nested selection (112_nested_selection.py)
# and the leaky-partition PCD (116_pcd_leaky.py)
# have finished. Each step stops the pipeline on error. Every manuscript
# writer is re-runnable.
set -e
cd "E:/Dream GAN/analysis"
PY=/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python
echo "== seeds";        $PY 105_cbramod_seeds.py > /dev/null
echo "== seeds -> tex"; $PY 107_write_seeds.py > /dev/null
echo "== Table 5 -> IV-F, IV-D"; $PY 91_write_ivf.py
echo "== stratified PCD"; $PY 111_write_pcd_strat.py | tail -1
echo "== merged PCD table"; $PY -W ignore 115_pcd_table_merged.py
echo "== leaky-partition PCD"; $PY 117_write_pcd_leaky.py
echo "== abstract, VII"; $PY 101_write_summary.py
echo "== framework fig"; $PY 102_fig_framework.py
echo "== DEED-null summary"; $PY 109_pcd_deednull_summary.py > /dev/null
echo "== static check (clean)"; $PY 113_tex_static_check.py
echo "== audit";        $PY 95_audit.py | head -5
echo "== letter";       $PY 94_response_letter.py
echo "== letter audit"; $PY 118_letter_audit.py
echo "== highlight";    $PY 62_highlight_yellow.py | tail -2
echo "== static check (highlighted)"; $PY 113_tex_static_check.py "E:/Dream_Analysis/accesstemp_highlighted.tex"
echo "== bundle";       $PY 63_submission_bundle.py | grep archive
echo "== supplement";   $PY 100_supplement_zip.py
echo "final pipeline done $(date)"
