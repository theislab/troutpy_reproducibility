# 2. uRNA pipeline (default segmentation)

`urna_analysis_pipeline*.py` runs the full troutpy workflow on one ROI folder
(`rois/<dataset>/data.zarr`) and writes `data_with_source_score.zarr` and `figures/` next to
it. Run `python <script> --help` for the options; each step is described in the script's
docstring. One `urna_analysis_pipeline_<dataset>.sbatch` per dataset submits the job.

| Script | Datasets |
|---|---|
| `urna_analysis_pipeline.py` | Xenium, Atera, CosMx mouse brain and stomach |
| `urna_analysis_pipeline_cosmx_merscope.py` | CosMx hgsoc / melanoma, MERSCOPE brain / uterine |
| `urna_analysis_pipeline_xenium_rcc.py` | Multimodal Xenium RCC |

`convert_zarr_v3_to_v2.py` converts the RCC ROI (written with the newer spatialdata/zarr v3)
back to zarr v2 for the main environment.
