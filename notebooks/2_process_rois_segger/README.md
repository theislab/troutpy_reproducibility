# 2c. Segger resegmentation

1. `fix_cellid_*.sbatch` → `fix_cell_id_for_segger.py`: make transcript cell ids
   compatible with Segger (CosMx, MERSCOPE, Xenium 300).
2. `run_segger_*.sbatch` (`submit_all_segger.sh`): train/predict Segger (env `envs/segger.yml`).
3. `format_segger_*.sbatch` → `format_segger_to_sdata.py`: turn Segger output into a
   troutpy-ready `rois_segger/<dataset>/data.zarr` (`reformat_xenium_rcc_zarr.py` for RCC).
4. `urna_analysis_pipeline_segger_*.sbatch` → `urna_analysis_pipeline.py`: the uRNA pipeline.
