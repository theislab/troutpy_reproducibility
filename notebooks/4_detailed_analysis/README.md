# 4. Detailed analyses

Case studies on single datasets (mainly the Xenium 5k mouse brain). Order for the mouse
brain:

1. `0_annotation_msbrain_5k.ipynb`: cell type annotation.
2. `2_drvi_analysis_msbrain.ipynb`: uRNA structures (DRVI, structure types, parent cells).
3. `4_image_analysis_msbrain_5k.ipynb`: staining-based validation.
4. `recompute_lambda_scores_msbrain.py`: λ sweep (ED Fig. 7A).
   `recompute_source_score_final_lambda05.py`: production source score at λ = 0.5.
5. `5_computing_cell_scores_msbrain.ipynb`: source/target scores, cell-cell contacts.
6. `6_computing_protrusion_proportions.ipynb`, `7_morphology_urna_celltype_association.ipynb`.
7. Decontamination: `run_local_bg_msbrain.py` → `8_local_urna_background_decontamination.ipynb`,
   `recompute_source_score_final_lambda05_decontaminated.py` → `9_urna_proportion_corrected_vs_uncorrected.ipynb`,
   `purity_evaluation_decontamination.py` (ED Fig. 8), `nuclei_cyto_validation.py` (ED Fig. 7K).
8. `source_score_neighborhood_comparison.py` → notebooks 10 and 11 (reviewer analyses).

Other datasets: `0_annotation_atera_breast.ipynb` + `2_drvi_analysis_atera_hsbreast.ipynb`
(breast), `rcc_som_multimodal_part1/2.ipynb` + `rcc_som_multimodal_part2_batch/` (multimodal
RCC, Fig. 4), `detailed_analysis_{melanoma,mscolon,bone}_revisions.ipynb`.
Scripts are submitted with the `.sbatch` file of the same name.
