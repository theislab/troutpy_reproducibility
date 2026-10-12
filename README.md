# troutpy reproducibility

Code to reproduce the analyses of

> **Exploration of RNA outside segmented cells in spatial transcriptomics reveals extrasomatic RNA organization**
> Marco Salas S., Ries C., Dammann M., Shahhoseini M., Kfuri-Rubens R., Drummer F., Halle L., Becker S., Castelo-Branco G., Theis F. J.

The methods themselves are implemented in **troutpy**
([github.com/theislab/troutpy](https://github.com/theislab/troutpy),
[documentation](https://troutpy.readthedocs.io/en/latest/)). This repository holds the
notebooks, batch scripts and small helper package (`troutpy_reproducibility`) used to
apply troutpy to the 17 + 1 datasets of the study and to make the figures.

The in situ sequencing and sparse-labelling (Sox10-CreERT2) analyses (Fig. 2I, ED Fig. 7F,
ISS part) are in [github.com/sergiomarco25/regiss](https://github.com/sergiomarco25/regiss).

## Installation

The analyses were run on a SLURM cluster with conda environments. Exported environments
are in [`envs/`](envs):

| Environment | Used for |
|---|---|
| `envs/troutpy_drvi.yml` | Almost everything: ROI selection, uRNA pipelines, figures, DRVI |
| `envs/troutpy_updated.yml` | Decontamination (notebooks 8-9) and the multimodal RCC analysis (SOM / Spatiomic) |
| `envs/spatialdata.yml` | zarr v3 → v2 conversion and the multimodal RCC ROI (newer spatialdata) |
| `envs/segger.yml` | Running Segger |

```bash
conda env create -f envs/troutpy_drvi.yml
conda activate troutpy_drvi
pip install -e /path/to/troutpy --no-deps    # troutpy (see version below)
pip install -e . --no-deps                   # this repository's helper package
```

**troutpy version.** All results use the troutpy version that renames
`tl.adaptative_source_score_optimized` to `tl.urna_source_score` (λ = 0.5 default) and
adds `tl.subtract_local_urna_background`. Use troutpy ≥ that release.

Notebooks and scripts import the code as normal packages:

```python
import troutpy
from troutpy import pp, tl, pl
from troutpy_reproducibility.batch_running import clustering_pipeline
```

Proseg v3.1.1 is needed for the Proseg resegmentation.

## Data

- Public datasets: see Extended Data Table 1 for sources.
- Regions of interest used in the study (`rois/<dataset>/data.zarr`): Zenodo
  [10.5281/zenodo.17446348](https://doi.org/10.5281/zenodo.17446348).
- ISS + sparse-labelling data: Zenodo [10.5281/zenodo.21900575](https://doi.org/10.5281/zenodo.21900575).

Paths in the code point to the authors' cluster
(`/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/...`). After downloading the
ROIs, replace that root with your own data folder. The layout below it is
`rois/<dataset>/` for default segmentation and `rois_proseg/<dataset>/`,
`rois_segger/<dataset>/` for the resegmentations.

## Workflow

```
notebooks/
├── 1_roi_selection_notebooks/   1. crop each raw dataset to its ROI → rois/<dataset>/data.zarr
├── 2_process_rois/              2. uRNA pipeline per dataset (default segmentation)
├── 2_process_rois_proseg/       2b. Proseg resegmentation + the same pipeline
├── 2_process_rois_segger/       2c. Segger resegmentation + the same pipeline
├── 3_analysis_of_results/       3. cross-dataset figures (Fig. 1-3, ED Fig. 1, 2, 5)
├── 4_detailed_analysis/         4. case studies: mouse brain, breast, RCC, ... (Fig. 2, 4, 5, ED Fig. 3-4, 6-8)
├── 6_multi_dataset_lambda05_annotation/   λ = 0.5 source score across all datasets
├── SUPP1_sainsc_parameter_tuning/         ED Fig. 2A-C, density hyperparameters
├── SUPP2_rayleigh_distribution_simulation/ ED Fig. 5C-E
└── GEO_download_notebooks/      data download
src/troutpy_reproducibility/     helper functions used by the notebooks
envs/                            conda environments
```

Steps 2-2c run on SLURM: each dataset has an `.sbatch` file that calls the pipeline
script with `rois*/<dataset>` as input. The pipeline writes
`data_with_source_score.zarr` and `figures/` in that folder; step 3 and 4 notebooks read
those. Every notebook starts with a short header giving its aim, the manuscript panels it
produces, its inputs/outputs and its kernel.

## Figure → code

| Panel(s) | Notebook / script |
|---|---|
| Fig. 1A, ED Table 1 | `3_analysis_of_results/0_metadata_roi_analysis.ipynb` |
| Fig. 1B | `3_analysis_of_results/1_summary_metrics_roi_analysis_revisions.ipynb` |
| Fig. 1C | Schematic |
| Fig. 1D-E | `3_analysis_of_results/8_cosim_staining_density_comparison_and_spatial.ipynb` |
| Fig. 2A-B | `3_analysis_of_results/2_summary_cell_like_roi_analysis.ipynb` |
| Fig. 2C | `3_analysis_of_results/7_spatial_map_segmentation_comparison.ipynb` |
| Fig. 2D-H | `4_detailed_analysis/2_drvi_analysis_msbrain.ipynb` |
| Fig. 2I | regiss repository |
| Fig. 3A | `3_analysis_of_results/9_other_urna_noise_proportion.ipynb` |
| Fig. 3B | `3_analysis_of_results/10_other_urna_diffusion_proportion.ipynb` |
| Fig. 3C | `3_analysis_of_results/11_full_rna_classification.ipynb` |
| Fig. 3D, 3G (left) | `3_analysis_of_results/5_overall_per_noise_gene_scatters_revisions.ipynb` |
| Fig. 3E | `4_detailed_analysis/detailed_analysis_melanoma_revisions.ipynb`, `3_analysis_of_results/3_overall_scatters.ipynb` |
| Fig. 3F | `4_detailed_analysis/detailed_analysis_mscolon_revisions.ipynb` |
| Fig. 3H | `3_analysis_of_results/4_cross_technology_comparison.ipynb` |
| Fig. 3I | `3_analysis_of_results/4_cross_technology_comparison_opsc.ipynb` |
| Fig. 4A-B | `4_detailed_analysis/4_image_analysis_msbrain_5k.ipynb` |
| Fig. 4C-D | `4_detailed_analysis/rcc_som_multimodal_part1.ipynb` |
| Fig. 4E-G | `4_detailed_analysis/rcc_som_multimodal_part2.ipynb`, `rcc_som_multimodal_part2_batch/` |
| Fig. 5B-E, 5G | `4_detailed_analysis/5_computing_cell_scores_msbrain.ipynb` (source score from `recompute_source_score_final_lambda05.py`) |
| Fig. 5H | `4_detailed_analysis/9_urna_proportion_corrected_vs_uncorrected.ipynb` |
| ED Fig. 1B | `3_analysis_of_results/13_morphology_urna_association.ipynb` (+ `morphology/compute_missing_morphology.py`) |
| ED Fig. 1C | `3_analysis_of_results/12_segmentation_method_consistency.ipynb` |
| ED Fig. 2A-C | `SUPP1_sainsc_parameter_tuning/1_sainsc_hyperparameter.py` + `2_sainsc_parameter_analysis_figures.ipynb` |
| ED Fig. 2D-F | `3_analysis_of_results/2_summary_cell_like_roi_analysis.ipynb` |
| ED Fig. 2G | `3_analysis_of_results/7_spatial_map_segmentation_comparison.ipynb` |
| ED Fig. 3A-B, 3E | `4_detailed_analysis/0_annotation_msbrain_5k.ipynb` |
| ED Fig. 3C-G | `4_detailed_analysis/2_drvi_analysis_msbrain.ipynb` |
| ED Fig. 4 | `4_detailed_analysis/0_annotation_atera_breast.ipynb`, `2_drvi_analysis_atera_hsbreast.ipynb` |
| ED Fig. 5A | `3_analysis_of_results/control_probes/build_preservation_plot_v2.py` |
| ED Fig. 5C-E | `SUPP2_rayleigh_distribution_simulation/Simulating_data_for_diffussion_complex_3d.ipynb` |
| ED Fig. 5F | `3_analysis_of_results/11_full_rna_classification.ipynb` |
| ED Fig. 5G | `3_analysis_of_results/4_cross_technology_comparison.ipynb` |
| ED Fig. 6 | `4_detailed_analysis/rcc_som_multimodal_part1.ipynb`, `rcc_som_multimodal_part2.ipynb` |
| ED Fig. 7A | `4_detailed_analysis/recompute_lambda_scores_msbrain.py` |
| ED Fig. 7B, 7D-F | `4_detailed_analysis/5_computing_cell_scores_msbrain.ipynb` (7F: Xenium side) |
| ED Fig. 7C | `4_detailed_analysis/7_morphology_urna_celltype_association.ipynb` |
| ED Fig. 7G-I | `4_detailed_analysis/8_local_urna_background_decontamination.ipynb` (`run_local_bg_msbrain.py`) |
| ED Fig. 7J | `4_detailed_analysis/6_computing_protrusion_proportions.ipynb`, `9_urna_proportion_corrected_vs_uncorrected.ipynb` |
| ED Fig. 7K | `4_detailed_analysis/nuclei_cyto_validation.py` |
| ED Fig. 8 | `4_detailed_analysis/purity_evaluation_decontamination.py` |

## Key parameters

| Step | Function | Setting used |
|---|---|---|
| uRNA source score | `tl.urna_source_score` | `lmbda=0.5, max_dist=200, max_k=10, signal_threshold=10, residual=0.1` |
| uRNA structures | `tl.segment_protrusions` | `cluster_eps=2.0, min_samples=1, connectivity_threshold=4.0, n_neighbors=10` |
| uRNA bins | `pp.aggregate_urna` | 10 µm squares |
| Local background subtraction | `tl.subtract_local_urna_background` | inner radius 10 µm, outer radius 100 µm |

## License

See [LICENSE](LICENSE).
