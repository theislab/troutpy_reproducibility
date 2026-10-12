#!/bin/bash
# Submits, per segger-processed sample, a formatting job (stage 1: segger ->
# troutpy-ready SpatialData) followed by the troutpy analysis pipeline (stage 2),
# chained with --dependency=afterok so stage 2 only runs if stage 1 succeeded.
#
# Only samples that actually have a segger_segmentation.parquet are included;
# the segger runs for cosmx_*/merscope_*/xenium_msbrain_300 never produced one,
# so those are skipped rather than submitted to fail.

set -euo pipefail

SEGGER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEGGER_MODELS_DIR="/lustre/groups/ml01/workspace/mostafa.shahhosseini/data/troutpy_rois/segger_out/models"

SAMPLES="atera_hsbreast xenium_hsfemur xenium_hsheart xenium_hsprostate xenium_hsputamen xenium_hsskin_5k xenium_msbrain_5k xenium_mscolon_5k xenium_opsc_5k xenium_opsc_v1 xenium_rcc xenium_rcc_oldzarr"

printf "%-20s %-12s %-12s\n" "sample" "format_job" "pipeline_job"
for s in $SAMPLES; do
  if [ ! -f "${SEGGER_MODELS_DIR}/${s}/segger_segmentation.parquet" ]; then
    printf "%-20s %-12s %-12s\n" "$s" "SKIPPED" "SKIPPED"
    continue
  fi

  fmt_jobid=$(sbatch --parsable "${SEGGER_DIR}/format_segger_${s}.sbatch")
  pipe_jobid=$(sbatch --parsable --dependency=afterok:${fmt_jobid} "${SEGGER_DIR}/urna_analysis_pipeline_segger_${s}.sbatch")

  printf "%-20s %-12s %-12s\n" "$s" "$fmt_jobid" "$pipe_jobid"
done
