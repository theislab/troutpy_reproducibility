#!/usr/bin/env python3
"""Final cross-dataset check: combine the recomputed lambda=0.5 urna_source_score with
the unified cell_type_general label, per dataset, and visualize how much uRNA signal
lands on each coarse cell type across all 17 datasets.

Reads table.obs['urna_source_score'] and table.obs['cell_type_general'] directly via
the zarr array API (bypassing a full spatialdata parse / image loading, which is not
needed here and is the slow part of sd.read_zarr).

Outputs:
- urna_by_celltype_summary.csv: per (dataset, cell_type_general) row with n_cells,
  mean_score, total_score, pct_of_dataset_total_urna.
- urna_by_celltype_heatmap.pdf: heatmap of % of each dataset's total assigned uRNA
  landing on each coarse cell type (editable vector PDF).
"""
import zarr
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import seaborn as sns

# (dataset_name, path to its final zarr carrying both urna_source_score and cell_type_general)
FINAL_ZARRS = [
    ("xenium_msbrain_5k", "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_msbrain_5k/data_annotated_lambda0.5_unified.zarr"),
    ("atera_hsbreast", "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/atera_hsbreast/data_annotated_source_score_lambda0.5_unified.zarr"),
    ("xenium_hsfemur", "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_hsfemur/data_with_source_score_annotated_source_score_lambda0.5_unified.zarr"),
    ("cosmx_hgsoc", "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/cosmx_hgsoc/data_with_source_score_general_annotated_source_score_lambda0.5.zarr"),
    ("cosmx_melanoma", "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/cosmx_melanoma/data_with_source_score_general_annotated_source_score_lambda0.5.zarr"),
    ("cosmx_mousebrain", "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/cosmx_mousebrain/data_with_source_score_general_annotated_source_score_lambda0.5.zarr"),
    ("cosmx_stomach", "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/cosmx_stomach/data_with_source_score_general_annotated_source_score_lambda0.5.zarr"),
    ("merscope_hsbrain", "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/merscope_hsbrain/data_with_source_score_general_annotated_source_score_lambda0.5.zarr"),
    ("merscope_hsuterine", "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/merscope_hsuterine/data_with_source_score_general_annotated_source_score_lambda0.5.zarr"),
    ("xenium_hsheart", "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_hsheart/data_with_source_score_general_annotated_source_score_lambda0.5.zarr"),
    ("xenium_hsprostate", "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_hsprostate/data_with_source_score_general_annotated_source_score_lambda0.5.zarr"),
    ("xenium_hsputamen", "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_hsputamen/data_with_source_score_general_annotated_source_score_lambda0.5.zarr"),
    ("xenium_hsskin_5k", "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_hsskin_5k/data_with_source_score_general_annotated_source_score_lambda0.5.zarr"),
    ("xenium_msbrain_300", "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_msbrain_300/data_with_source_score_general_annotated_source_score_lambda0.5.zarr"),
    ("xenium_mscolon_5k", "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_mscolon_5k/data_with_source_score_general_annotated_source_score_lambda0.5.zarr"),
    ("xenium_opsc_5k", "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_opsc_5k/data_with_source_score_general_annotated_source_score_lambda0.5.zarr"),
    ("xenium_opsc_v1", "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_opsc_v1/data_with_source_score_general_annotated_source_score_lambda0.5.zarr"),
]

HERE_FIGURES = "/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/6_multi_dataset_lambda05_annotation/figures"


def load_obs_fields(path):
    z = zarr.open(f"{path}/tables/table/obs", mode="r")
    score = z["urna_source_score"][:]
    ct = z["cell_type_general"]
    categories = np.array(ct["categories"][:])
    codes = ct["codes"][:]
    labels = np.where(codes >= 0, categories[np.clip(codes, 0, len(categories) - 1)], "NA")
    return pd.DataFrame({"urna_source_score": score, "cell_type_general": labels})


rows = []
for name, path in FINAL_ZARRS:
    print(f"[{name}] reading {path}", flush=True)
    df = load_obs_fields(path)
    dataset_total = df["urna_source_score"].sum()
    agg = df.groupby("cell_type_general")["urna_source_score"].agg(n_cells="size", mean_score="mean", total_score="sum")
    agg["pct_of_dataset_total_urna"] = 100 * agg["total_score"] / dataset_total
    agg["dataset"] = name
    agg = agg.reset_index()
    rows.append(agg)
    print(agg[["cell_type_general", "n_cells", "mean_score", "pct_of_dataset_total_urna"]], flush=True)
    print(flush=True)

summary = pd.concat(rows, ignore_index=True)
summary.to_csv(f"{HERE_FIGURES}/urna_by_celltype_summary.csv", index=False)
print(f"Saved {HERE_FIGURES}/urna_by_celltype_summary.csv", flush=True)

# --- Heatmap: % of each dataset's total assigned uRNA per coarse cell type ---
pivot = summary.pivot(index="dataset", columns="cell_type_general", values="pct_of_dataset_total_urna").fillna(0)
category_order = ["Epithelial", "Malignant/Tumor", "Endothelial", "Fibroblast/Stromal", "Muscle", "Neural", "Immune", "Other/Unclear"]
pivot = pivot[[c for c in category_order if c in pivot.columns]]

dataset_order = [n for n, _ in FINAL_ZARRS]
pivot = pivot.loc[dataset_order]

sns.set_theme(style="white", font_scale=1.1)
fig, ax = plt.subplots(figsize=(1.4 * pivot.shape[1] + 2, 0.5 * pivot.shape[0] + 2))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "% of dataset's total assigned uRNA"}, linewidths=0.5, linecolor="white", ax=ax)
ax.set_title("uRNA source score (λ=0.5) by coarse cell type, per dataset", fontsize=14)
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(f"{HERE_FIGURES}/urna_by_celltype_heatmap.pdf", bbox_inches="tight")
plt.close()
print(f"Saved {HERE_FIGURES}/urna_by_celltype_heatmap.pdf", flush=True)

print("Done.", flush=True)
