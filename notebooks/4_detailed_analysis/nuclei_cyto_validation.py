#!/usr/bin/env python3
"""Nuclei-vs-cytoplasm validation of the source-scoring method, corrected vs. uncorrected.

Modernized reproduction of
notebooks/4_detailed_analysis/old/computing_extrasomatic_RNA_analysis_nucleiVScyto.ipynb,
run on the same underlying dataset (data_processed_with_scores.zarr) it originally used.

Ground truth: real segmentation already tells us, per (gene, cell type), what fraction
of a cell's transcripts are cytoplasmic vs. nuclear. We treat nuclear-only counts as
"cell expression" and cytoplasmic transcripts as if they were unassigned (uRNA) -- a
ground-truth stand-in for the real intracellular/extracellular split -- then ask
whether troutpy's source-scoring correctly recovers each cell type's true cytoplasmic
proportion.

Two changes from the original notebook, per project discussion:
1. Uses tl.urna_source_score (lmbda=0.5, the value used for every source-score result in
   the manuscript; the original notebook used 0.1) instead of the older
   tl.compute_source_score, for tractability at full dataset scale.
2. Cytoplasmic ("fake uRNA") transcripts are spatially diffused (Brownian, scale =
   mean protrusion_length in this dataset = 5.8um) away from their true position before
   scoring, so they resemble genuine uRNA (which has physically traveled) rather than
   sitting exactly where they started -- a more honest test of the distance-decay
   scoring logic than zero displacement would be.

--decontaminate runs troutpy.tl.subtract_local_urna_background (radius=100,
inner_radius=10) on the nuclear-only counts, using the diffused cytoplasmic pool as the
local background reference, before scoring -- testing whether decontamination itself
improves the validated accuracy of the recovered proportions.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import spatialdata as sd
from scipy.sparse import issparse
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from spatialdata.models import PointsModel

from troutpy import tl

from troutpy_reproducibility.diffusion_simulation import get_displacement

parser = argparse.ArgumentParser()
parser.add_argument("--decontaminate", action="store_true")
args = parser.parse_args()
label = "corrected" if args.decontaminate else "uncorrected"

DIFFUSION_SCALE = 5.8  # microns; mean protrusion_length in this dataset

data_dir = Path("/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_msbrain_5k")
saving_path = Path("/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/ms_figures/xenium_msbrain_5k/nuclei_vs_cyto_validation")
saving_path.mkdir(parents=True, exist_ok=True)

print(f"=== Nuclei-vs-cyto validation ({label}) ===", flush=True)
sdata = sd.read_zarr(data_dir / "data_processed_with_scores.zarr")

print("Loading transcripts and applying the nuclear/cytoplasm ground-truth split...", flush=True)
transcripts = sdata.points["transcripts"].compute()
trans_filt = transcripts[transcripts["extracellular"] != True].copy()  # noqa: E712 -- pandas bool column
trans_filt = trans_filt[trans_filt["cell_id"] != "UNASSIGNED"]

adata = sdata["table"]
id2ct = dict(zip(adata.obs["cell_id"], adata.obs["cell_type"], strict=False))
trans_filt["cell_type"] = trans_filt["cell_id"].map(id2ct)
trans_filt = trans_filt.loc[~trans_filt["cell_type"].isna(), :]

# Redefine "extracellular" = cytoplasmic (the fake-uRNA ground-truth swap)
trans_filt["extracellular"] = ~trans_filt["overlaps_nucleus"]

# --- Ground truth: true cytoplasmic proportion per (gene, cell_type), from real segmentation ---
grouped = trans_filt.groupby(["gene", "cell_type", "extracellular"]).size().unstack(fill_value=0)
if True not in grouped.columns:
    grouped[True] = 0
if False not in grouped.columns:
    grouped[False] = 0
grouped["total"] = grouped[True] + grouped[False]
missing = (grouped[True] == 0) | (grouped[False] == 0)
grouped["percent_extracellular"] = np.where(missing, np.nan, grouped[True] / grouped["total"])
ground_truth = grouped.reset_index()
ground_truth.to_csv(saving_path / f"ground_truth_cytoplasmic_proportion_{label}.csv", index=False)

# --- Diffuse the cytoplasmic ("fake uRNA") transcripts away from their true position ---
cyto_mask = trans_filt["extracellular"].to_numpy()
n_cyto = int(cyto_mask.sum())
dx, dy, _ = get_displacement(n_cyto, DIFFUSION_SCALE, mode="brownian_2d")
trans_filt.loc[cyto_mask, "x"] = trans_filt.loc[cyto_mask, "x"].to_numpy() + dx
trans_filt.loc[cyto_mask, "y"] = trans_filt.loc[cyto_mask, "y"].to_numpy() + dy
print(f"Diffused {n_cyto} cytoplasmic transcripts by Brownian(scale={DIFFUSION_SCALE}um) per axis.", flush=True)

# --- Sentinel cell_id for the fake-uRNA pool: urna_source_score decides
# its "unassigned pool to score" purely by cell_id membership, not by the 'extracellular'
# column, so this is required for it to pick these up (matches the convention already
# used elsewhere in this pipeline for genuine uRNA). ---
trans_filt.loc[cyto_mask, "cell_id"] = "UNASSIGNED"
trans_filt["overlaps_cell_new"] = trans_filt["cell_id"] != "UNASSIGNED"

# --- Redefine cell expression as nuclear-only ---
nuclear_only = trans_filt[~trans_filt["extracellular"]]
expr = pd.crosstab(nuclear_only["cell_id"], nuclear_only["gene"])
expr = expr.reindex(index=adata.obs["cell_id"], columns=adata.var_names, fill_value=0)
adata.X = expr.to_numpy()
adata.layers["raw"] = adata.X.copy()
sdata["table"] = adata
sdata.points["transcripts"] = PointsModel.parse(trans_filt)

if args.decontaminate:
    print("Decontaminating nuclear-only counts against the diffused cytoplasmic 'uRNA' pool...", flush=True)
    tl.subtract_local_urna_background(
        sdata, radius=100.0, inner_radius=10.0, update_X=True, credit_to_urna_pool=False
    )

# --- enrichment_over_random, needed by urna_source_score's k_adaptive sizing ---
# segmentation_key points at our redefined "is this cell_id valid" column, so density_similarity's
# is_extra mask matches exactly the pool urna_source_score will score.
print("Computing local density / enrichment_over_random...", flush=True)
tl.density_similarity(sdata, radius=10.0, segmentation_key="overlaps_cell_new")

transcripts2 = sdata.points["transcripts"].compute()
min_enrichment = transcripts2[transcripts2["density_similarity_score"] > 0.5]["enrichment_over_random"].min()
print(f"ambient_floor (min_enrichment) = {min_enrichment}", flush=True)

print("Running urna_source_score (lambda=0.5)...", flush=True)
tl.urna_source_score(
    sdata,
    chunk_size=100000,
    cell_type_col="cell_type",
    max_dist=200,
    lmbda=0.5,
    max_k=10,
    ambient_floor=min_enrichment,
    signal_threshold=10,
    residual=0.1,
)

# --- Predicted cytoplasmic proportion from the source-score contribution matrix ---
adata = sdata["table"]
X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
expr_df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
cell_types = adata.obs["cell_type"]

contrib_df = sdata["source_score"].to_df()
contrib_df["gene"] = sdata["source_score"].obs["gene"]
contrib_matrix = contrib_df.groupby("gene").sum()
contrib_matrix = contrib_matrix.T.groupby(level=0).sum().T

intra_matrix = expr_df.groupby(cell_types.to_numpy()).sum().T
intra_mean = expr_df.groupby(cell_types.to_numpy()).mean().T

genes_final = contrib_matrix.index.intersection(intra_matrix.index)
ct_final = contrib_matrix.columns.intersection(intra_matrix.columns)
extra_matrix = contrib_matrix.loc[genes_final, ct_final]
intra_matrix_f = intra_matrix.loc[genes_final, ct_final]
intra_mean_f = intra_mean.loc[genes_final, ct_final]

prop = extra_matrix / (extra_matrix + intra_matrix_f)
filt = (intra_mean_f > 0.1) & (intra_mean_f.div(intra_mean_f.sum(axis=1), axis=0) > 0.1)

rows = []
for ct in prop.columns:
    p = prop[ct][filt[ct]].fillna(0)
    for g, v in p.items():
        rows.append({"gene": g, "cell_type": ct, "extracellular_proportion": v})
predicted_df = pd.DataFrame(rows)
predicted_df.to_csv(saving_path / f"predicted_cytoplasmic_proportion_{label}.csv", index=False)

# --- Merge with ground truth, compute PCC + cosine similarity ---
gt_filt = ground_truth[~ground_truth["percent_extracellular"].isna()]
merged = pd.merge(gt_filt, predicted_df, on=["gene", "cell_type"], how="inner")
merged.to_csv(saving_path / f"merged_comparison_{label}.csv", index=False)

r, _ = pearsonr(merged["percent_extracellular"], merged["extracellular_proportion"])
cos_sim = cosine_similarity(
    merged["percent_extracellular"].to_numpy().reshape(1, -1),
    merged["extracellular_proportion"].to_numpy().reshape(1, -1),
)[0, 0]
print(f"[{label}] n_pairs={len(merged)}  PCC={r:.4f}  Cosine similarity={cos_sim:.4f}", flush=True)

with open(saving_path / f"metrics_{label}.txt", "w") as f:
    f.write(f"label={label}\nn_pairs={len(merged)}\nPCC={r}\ncosine_similarity={cos_sim}\n")

# --- Plots ---
order_ct = predicted_df.groupby("cell_type")["extracellular_proportion"].mean().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(7, 7))
sns.violinplot(data=predicted_df, y="cell_type", x="extracellular_proportion", order=order_ct, inner=None, ax=ax)
sns.stripplot(data=predicted_df, y="cell_type", x="extracellular_proportion", color="gray", s=1.2, order=order_ct, jitter=0.3, ax=ax)
ax.set_title(f"Predicted cytoplasmic proportion ({label})")
plt.tight_layout()
fig.savefig(saving_path / f"violin_predicted_{label}.pdf", format="pdf", bbox_inches="tight")

order_ct2 = gt_filt.groupby("cell_type")["percent_extracellular"].mean().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(7, 7))
sns.violinplot(data=gt_filt, y="cell_type", x="percent_extracellular", order=order_ct2, inner=None, ax=ax)
sns.stripplot(data=gt_filt, y="cell_type", x="percent_extracellular", color="gray", s=1.2, order=order_ct2, jitter=0.3, ax=ax)
ax.set_title(f"True cytoplasmic proportion, ground truth ({label} run)")
plt.tight_layout()
fig.savefig(saving_path / f"violin_true_{label}.pdf", format="pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(5, 5))
sns.regplot(
    x="percent_extracellular", y="extracellular_proportion", data=merged, ax=ax,
    scatter_kws={"s": 3, "alpha": 0.5}, line_kws={"color": "#a6054b"},
)
ax.text(
    0.05, 0.95, f"PCC={r:.2f}\nCos sim={cos_sim:.2f}", transform=ax.transAxes, va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)
ax.set_xlabel("True cytoplasm proportion")
ax.set_ylabel("Predicted cytoplasm proportion")
ax.set_title(label)
plt.tight_layout()
fig.savefig(saving_path / f"scatter_regression_{label}.pdf", format="pdf", bbox_inches="tight")

print("Done.", flush=True)
