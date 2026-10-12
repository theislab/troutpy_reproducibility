# 2b. Proseg resegmentation

1. `run_proseg/`: run Proseg (v3.1.1) on each ROI (`1_run_proseg.sbatch` →
   `proseg_pipeline.py`) and rebuild a SpatialData object from its output
   (`2_run_reconstruction*.sbatch` → `reconstruct*.py`). All submitted commands are in
   `run_proseg/all_runs_commands.sh` (submit from inside `run_proseg/`).
2. `urna_<dataset>_proseg.sbatch` runs `urna_analysis_pipeline.py` on
   `rois_proseg/<dataset>`, giving the same outputs as the default-segmentation pipeline.
