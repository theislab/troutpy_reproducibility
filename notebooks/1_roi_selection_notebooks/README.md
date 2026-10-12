# 1. ROI selection

One notebook per dataset: read the raw vendor output, choose the region of interest used
in the study, convert it to troutpy's SpatialData layout (`troutpy.pp.xenium_converter` /
`cosmx_converter`, `troutpy_reproducibility.roi_processing`) and write
`rois/<dataset>/data.zarr` + `metadata.csv`. The resulting ROIs are on Zenodo
(10.5281/zenodo.17446348), so this step can be skipped.

| Notebook | Dataset folder |
|---|---|
| `Atera.ipynb`, `atera_processing.py` (+ `.sbatch`) | `atera_hsbreast` |
| `CosMx_hgsoc.ipynb` / `CosMx_melanoma.ipynb` / `CosMx_msbrain.ipynb` / `CosMx_pancreas.ipynb` / `CosMx_stomach.ipynb` | `cosmx_hgsoc` / `cosmx_melanoma` / `cosmx_mousebrain` / `cosmx_pancreas` / `cosmx_stomach` |
| `MERSCOPE_human_brain.ipynb` / `MERSCOPE_hs_uterine.ipynb` | `merscope_hsbrain` / `merscope_hsuterine` |
| `Xenium_hs_femur.ipynb` / `Xenium_hs_heart.ipynb` / `Xenium_5khs_prostate.ipynb` / `Xenium_hs_putamen.ipynb` / `Xenium_5khs_skin.ipynb` | `xenium_hsfemur` / `xenium_hsheart` / `xenium_hsprostate` / `xenium_hsputamen` / `xenium_hsskin_5k` |
| `Xenium_300msbrain.ipynb` / `Xenium_5kmsbrain.ipynb` / `Xenium_5kmscolon.ipynb` | `xenium_msbrain_300` / `xenium_msbrain_5k` / `xenium_mscolon_5k` |
| `Xenium_opsc_v1.ipynb` / `Xenium_opsc_5k.ipynb` | `xenium_opsc_v1` / `xenium_opsc_5k` |
| `Part1_Xenium_5k_rcc_multiomics.ipynb`, `xenium_5k_multiomics.py` | `xenium_rcc` (multimodal RNA + protein, Fig. 4) |
