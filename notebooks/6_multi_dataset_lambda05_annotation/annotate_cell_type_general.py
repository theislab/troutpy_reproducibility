#!/usr/bin/env python3
"""Annotate a dataset with a coarse, cross-dataset-comparable 'cell_type_general' column.

Hybrid method, identical for every dataset (no per-dataset tuning):

1. Load the dataset's best-available zarr (see dataset_registry.py).
2. Run CellTypist's Immune_All_High.pkl model on sdata['table'], with majority voting
   over the existing 'leiden' clusters (over_clustering=leiden) -- this gives one
   immune-vs-not call + confidence per leiden cluster rather than a noisy per-cell
   call, and needs no tissue-matched reference (immune signatures are broadly
   conserved, unlike organ-specific structural cell types).
3. A cluster is labelled 'Immune' if its mean majority-voting confidence exceeds
   IMMUNE_CONF_THRESHOLD.
4. All other clusters are scored against local marker gene sets (Epithelial,
   Endothelial, Fibroblast/Stromal, Muscle, Neural -- see coarse_marker_sets.py) via
   sc.tl.score_genes; each cluster gets the argmax category by mean score. A cluster
   where every category scores <= 0 is labelled 'Other/Unclear' rather than forced.
5. cell_type_general is written to sdata['table'].obs and the object is saved to a
   NEW zarr (<original>_general_annotated.zarr) -- the source zarr is never touched.

Usage: python annotate_cell_type_general.py <dataset_name>
"""
import os
import sys

import celltypist
import pandas as pd
import scanpy as sc
import spatialdata as sd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coarse_marker_sets import markers_for_species
from dataset_registry import BY_NAME, zarr_path

IMMUNE_CONF_THRESHOLD = 0.5


def run(dataset_name: str):
    ds = BY_NAME[dataset_name]
    in_path = zarr_path(ds)
    print(f"[{dataset_name}] loading {in_path}", flush=True)
    sdata = sd.read_zarr(in_path)

    adata = sdata["table"]
    if "leiden" not in adata.obs.columns:
        raise KeyError(f"{dataset_name}: 'leiden' not found in table.obs")

    # --- 1-3. CellTypist immune call, majority-voted over existing leiden clusters ---
    print(f"[{dataset_name}] running CellTypist Immune_All_High.pkl", flush=True)
    model = celltypist.models.Model.load(model="Immune_All_High.pkl")

    # CellTypist's models expect log1p(CP10K) input; the pipeline's own normalize_total
    # (batch_running.clustering_pipeline) uses the scanpy default target_sum=None (median
    # library size) for adata.X, which is a different scale and triggers CellTypist's
    # "invalid expression matrix" warning with deflated confidence scores. Rebuild the
    # CellTypist input from raw counts instead; adata.X itself is left untouched.
    if "raw" not in adata.layers:
        raise KeyError(f"{dataset_name}: table.layers['raw'] not found, needed to renormalize for CellTypist")
    ct_adata = adata.copy()
    ct_adata.X = ct_adata.layers["raw"].copy()
    sc.pp.normalize_total(ct_adata, target_sum=1e4)
    sc.pp.log1p(ct_adata)
    if ds.species == "mouse":
        # Immune_All_High.pkl is trained on human gene symbols; uppercase mouse
        # symbols so 1:1 orthologs match by name (standard practical workaround,
        # not a curated ortholog map).
        ct_adata.var_names = ct_adata.var_names.str.upper()

    pred = celltypist.annotate(
        ct_adata,
        model=model,
        majority_voting=True,
        over_clustering=adata.obs["leiden"].astype(str).values,
    )
    ct_out = pred.to_adata(insert_conf=True, insert_conf_by="majority_voting")

    cluster_conf = (
        pd.DataFrame({"leiden": adata.obs["leiden"].astype(str).values, "conf_score": ct_out.obs["conf_score"].values})
        .groupby("leiden")["conf_score"]
        .mean()
    )
    immune_clusters = set(cluster_conf.index[cluster_conf > IMMUNE_CONF_THRESHOLD])
    print(
        f"[{dataset_name}] {len(immune_clusters)}/{cluster_conf.shape[0]} leiden clusters called Immune "
        f"(mean conf > {IMMUNE_CONF_THRESHOLD})",
        flush=True,
    )

    # --- 4. Local marker scoring for the non-immune remainder ---
    markers = markers_for_species(ds.species)
    present_markers = {cat: [g for g in genes if g in adata.var_names] for cat, genes in markers.items()}
    present_markers = {cat: genes for cat, genes in present_markers.items() if genes}
    print(
        f"[{dataset_name}] marker genes present per category: { {k: len(v) for k, v in present_markers.items()} }",
        flush=True,
    )

    for cat, genes in present_markers.items():
        sc.tl.score_genes(adata, gene_list=genes, score_name=f"score_{cat}", use_raw=False)

    score_cols = [f"score_{cat}" for cat in present_markers]
    cluster_scores = adata.obs.groupby("leiden")[score_cols].mean()

    def assign_local(leiden_id):
        row = cluster_scores.loc[leiden_id]
        if (row <= 0).all():
            return "Other/Unclear"
        return row.idxmax().replace("score_", "")

    # --- 5. Combine into cell_type_general, save to a new zarr ---
    general = [
        "Immune" if cid in immune_clusters else assign_local(cid)
        for cid in adata.obs["leiden"].astype(str)
    ]
    adata.obs["cell_type_general"] = pd.Categorical(general)
    adata.obs.drop(columns=score_cols, inplace=True)

    print(f"[{dataset_name}] cell_type_general value counts:", flush=True)
    print(adata.obs["cell_type_general"].value_counts(), flush=True)

    sdata["table"] = adata

    out_path = in_path[: -len(".zarr")] + "_general_annotated.zarr"
    print(f"[{dataset_name}] writing {out_path}", flush=True)
    sdata.write(out_path, overwrite=False)
    print(f"[{dataset_name}] done.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python annotate_cell_type_general.py <dataset_name>")
        sys.exit(1)
    run(sys.argv[1])
