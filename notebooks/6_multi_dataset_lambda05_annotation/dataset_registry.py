"""Registry of datasets for the multi-dataset lambda=0.5 source-score recompute and
coarse cell-type-general annotation.

Populated from a discovery scan of /lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/
on 2026-08-27: for each dataset folder, the "best" (most processed) zarr was picked by
priority (an annotated variant with a real `cell_type` column first, else the pipeline's
final `data_with_source_score.zarr`), and its table.obs / transcripts columns were checked.

xenium_msbrain_5k is excluded: it already has its own dedicated lambda=0.5 recompute
(see notebooks/4_detailed_analysis/recompute_source_score_final_lambda05.py).

cosmx_pancreas is excluded: never finished the base processing pipeline (no
density_similarity_score / enrichment_over_random columns yet).

xenium_rcc is excluded for now: its processed zarr is Zarr v3 format, unreadable by this
environment's zarr/spatialdata stack. Needs data_with_source_score.zarr converted the
same way rois/convert_zarr_v3_to_v2.py converted data.zarr before it can be included.
"""
from dataclasses import dataclass

BASE = "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois"


@dataclass(frozen=True)
class Dataset:
    name: str
    zarr_name: str  # best available zarr, relative to BASE/<name>/
    species: str  # "human" or "mouse"
    has_cell_type: bool  # True if a real (fine) cell_type column already exists


DATASETS = [
    Dataset("atera_hsbreast", "data_annotated.zarr", "human", True),
    Dataset("cosmx_hgsoc", "data_with_source_score.zarr", "human", False),
    Dataset("cosmx_melanoma", "data_with_source_score.zarr", "human", False),
    Dataset("cosmx_mousebrain", "data_with_source_score.zarr", "mouse", False),
    Dataset("cosmx_stomach", "data_with_source_score.zarr", "human", False),
    Dataset("merscope_hsbrain", "data_with_source_score.zarr", "human", False),
    Dataset("merscope_hsuterine", "data_with_source_score.zarr", "human", False),
    Dataset("xenium_hsfemur", "data_with_source_score_annotated.zarr", "human", True),
    Dataset("xenium_hsheart", "data_with_source_score.zarr", "human", False),
    Dataset("xenium_hsprostate", "data_with_source_score.zarr", "human", False),
    Dataset("xenium_hsputamen", "data_with_source_score.zarr", "human", False),
    Dataset("xenium_hsskin_5k", "data_with_source_score.zarr", "human", False),
    Dataset("xenium_msbrain_300", "data_with_source_score.zarr", "mouse", False),
    Dataset("xenium_mscolon_5k", "data_with_source_score.zarr", "mouse", False),
    Dataset("xenium_opsc_5k", "data_with_source_score.zarr", "human", False),
    Dataset("xenium_opsc_v1", "data_with_source_score.zarr", "human", False),
]

NEEDS_GENERAL_ANNOTATION = [d for d in DATASETS if not d.has_cell_type]

BY_NAME = {d.name: d for d in DATASETS}


def zarr_path(d: Dataset) -> str:
    return f"{BASE}/{d.name}/{d.zarr_name}"


def general_annotated_zarr_path(d: Dataset) -> str:
    """Path to the coarse cell_type_general-annotated zarr produced by
    annotate_cell_type_general.py (only exists for datasets with has_cell_type=False)."""
    return zarr_path(d)[: -len(".zarr")] + "_general_annotated.zarr"


def source_score_input(d: Dataset) -> tuple[str, str]:
    """Returns (input_zarr_path, cell_type_col) to use for the lambda=0.5 source-score
    recompute: the dataset's own fine cell_type on its best zarr if already annotated,
    else cell_type_general on the coarse-annotated zarr produced by
    annotate_cell_type_general.py."""
    if d.has_cell_type:
        return zarr_path(d), "cell_type"
    return general_annotated_zarr_path(d), "cell_type_general"


def source_score_output(d: Dataset) -> str:
    in_path, _ = source_score_input(d)
    return in_path[: -len(".zarr")] + "_source_score_lambda0.5.zarr"
