#!/bin/bash
# Final integration for the round-3 resubmission (Access-2026-50879).
# Run after every analysis has finished (120, 125, 129, 151, the CBraMod
# runs). Every writer is idempotent; the pipeline is run twice and the
# manuscript must not change on the second pass. Stops on the first error.
set -e
cd "E:/Dream GAN/analysis"
PY=/c/Users/Sachin.R/AppData/Local/Programs/Python/Python312/python
run() { echo "== $1"; $PY "$@" > "out/pipeline_$(basename "$1" .py).txt" 2>&1 \
          || { echo "FAILED $1"; tail -5 "out/pipeline_$(basename "$1" .py).txt"; exit 1; }; }

# summaries that writers read
run 105_cbramod_seeds.py
run 109_pcd_deednull_summary.py
# writers predating round 3, in their established order
run 107_write_seeds.py
run 91_write_ivf.py
# 111, 115 and 117 (PCD, round 2) are not re-run: round-3 writers revised
# their wording (123, 141), and 136 and 140 now set their numbers from the
# label-free PCD results, so re-running them would restore superseded text.
# 136 (keyed old-to-new PCD update) is a one-time migration, applied once on
# 21 Oct 2026: re-run, it maps current values that equal some other old cell
# to that cell's new value, so it is never part of the pipeline.
# round-3 writers, in number order (142 follows 107)
for s in 121_refs_round3.py 122_table_comparison.py 123_pcd_wording.py \
         126_entropy_inventory.py 128_published_range.py 131_route_text.py \
         133_wording_round3.py 135_primary_text.py \
         137_factorial_text.py 138_conformal_text.py 140_pcd_text_update.py \
         141_pcd_scope.py 142_cbramod_constant.py 143_fold_account.py \
         145_cbramod_input.py 146_catalogue_hypothesis.py \
         147_identity_contrast.py 148_deed_stage_field.py 149_proofread.py \
         152_restricted_text.py 153_label_text.py; do
  run $s
done
# abstract and conclusions after Table 21 (122); the framework figure
run 101_write_summary.py
run 102_fig_framework.py
# table notes last, so every note any writer produced is made visible
run 139_table_notes.py
echo "manuscript writers done $(date)"
