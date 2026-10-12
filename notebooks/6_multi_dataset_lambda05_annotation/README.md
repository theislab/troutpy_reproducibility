# Source score at λ = 0.5 across all datasets

Coarse, comparable cell type annotation (`annotate_cell_type_general.py`, marker sets in
`coarse_marker_sets.py`, mapping of existing fine labels in `fine_to_coarse_mapping.py`) and
the λ = 0.5 uRNA source score for every dataset (`recompute_source_score.py`). Datasets are
listed in `dataset_registry.py`; `submit_annotate_all.py` / `submit_source_score_all.py`
generate and submit one job per dataset into `sbatch_generated/`.
`urna_burden_by_celltype.ipynb` and `urna_by_celltype_summary.py` summarise the results.
