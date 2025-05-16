import scanpy as sc
import os
import os
from typing import Sequence, Callable, Optional
import pandas as pd
import scanpy as sc
import troutpy


def clustering_pipeline(
    sdata,
    clustering_resolution=0.8,
    minimum_genes=10,
    minimum_counts=30,
    minimum_cells=3,
    n_neighbors=10,
    saving_path="",
):
    """
    Runs a standard preprocessing and clustering pipeline on spatial or single-cell AnnData.

    Parameters:
        sdata: SpatialData or dict-like object containing 'table' key for AnnData.
        clustering_resolution: float, resolution for Leiden clustering.
        minimum_genes: int, minimum number of genes for cell filtering.
        minimum_counts: int, minimum total counts for cell filtering.
        minimum_cells: int, minimum number of cells a gene must be expressed in.
        n_neighbors: int, number of neighbors for neighborhood graph.
        saving_path: str, directory to save plots. If empty, plots won't be saved.

    Returns:
        adata: Processed AnnData object with clustering results.
    """
    adata = sdata["table"]
    adata.raw = adata.copy()

    all_cells = adata.shape[0]
    sc.pp.filter_cells(adata, min_genes=minimum_genes)
    sc.pp.filter_cells(adata, min_counts=minimum_counts)
    sc.pp.filter_genes(adata, min_cells=minimum_cells)

    all_cells_filt = adata.shape[0]
    prop_cells = all_cells_filt / all_cells
    print(f"Proportion of cells retained after filtering: {prop_cells:.2%}")

    adata.layers["raw"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.umap(adata)

    print(f"Final adata shape: {adata.shape}")

    sc.tl.leiden(adata, key_added="leiden", resolution=clustering_resolution)
    sc.tl.dendrogram(adata, groupby="leiden")
    sc.tl.rank_genes_groups(adata, groupby="leiden")

    if saving_path:
        os.makedirs(saving_path, exist_ok=True)
        sc.pl.umap(adata, color="leiden", save=os.path.join(saving_path, "umap_leiden.png"))
        sc.pl.spatial(adata, color="leiden", spot_size=40, save=os.path.join(saving_path, "spatial_leiden.png"))
        sc.pl.rank_genes_groups_dotplot(
            adata, groupby="leiden", n_genes=3, save=os.path.join(saving_path, "rank_genes_dotplot.png")
        )
    else:
        sc.pl.umap(adata, color="leiden")
        sc.pl.spatial(adata, color="leiden", spot_size=40)
        sc.pl.rank_genes_groups_dotplot(adata, groupby="leiden", n_genes=3)

    return adata


def batch_spatial_inout_maps(
    sdata,
    metadata_layer: str,
    variables: Sequence[str],
    output_dir: str,
    n_top: int = 3,
    n_bottom: int = 3,
    spot_size_cells: float = 7,
    spot_size_extra: float = 12,
    save: bool = True,
    custom_fn_template: Optional[Callable[[str, str, str], str]] = None,
    **plot_kwargs,
) -> None:
    """
    Generate and (optionally) save spatial intra-/extra-cellular maps
    for top and bottom features of each variable.

    Parameters
    ----------
    sdata
        Your SpatialData / AnnData object, with `metadata_layer` in .var.
    metadata_layer
        Key in `sdata` where per-gene metadata (DataFrame) lives, e.g.
        "xrna_metadata".
    variables
        List of column names in `sdata[metadata_layer].var` to rank genes by.
    output_dir
        Directory in which to save all the plots. Will be created if missing.
    n_top
        Number of highest-scoring genes per variable to plot.
    n_bottom
        Number of lowest-scoring genes per variable to plot.
    spot_size_cells
        Passed to `spatial_inout_expression(..., spot_size_cells=...)`.
    spot_size_extra
        Passed to `spatial_inout_expression(..., spot_size_extra=...)`.
    save
        If True, each plot will be written to PDF in `output_dir`.
    custom_fn_template
        Optional function `(variable, rank, gene) -> filename` to
        name each PDF. By default:
            "{variable}_{rank}_{gene}.pdf"
        where `rank` is "top" or "bottom".
    **plot_kwargs
        Any extra keyword args forwarded to
        `troutpy.pl.spatial_inout_expression`.

    """
    # 1) Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 2) Build default filename template if none provided
    if custom_fn_template is None:

        def custom_fn_template(var, rank, gene):
            return f"{var}_{rank}_{gene}.png"

    # 3) Pull out the per-gene DataFrame
    var_df = sdata[metadata_layer].var.copy()

    # 4) Loop through each variable
    for var in variables:
        if var not in var_df.columns:
            raise KeyError(f"Variable '{var}' not found in {metadata_layer}.var")

        # rank genes
        ranked = var_df[var].sort_values(ascending=False)
        top_genes = ranked.head(n_top).index
        bottom_genes = ranked.tail(n_bottom).index

        # plot top
        for gene in top_genes:
            fname = custom_fn_template(var, "top", gene)
            troutpy.pl.spatial_inout_expression(
                sdata,
                gene,
                spot_size_cells=spot_size_cells,
                spot_size_extra=spot_size_extra,
                save=save,
                figures_path=output_dir,
                custom_plot_filename=fname,
                **plot_kwargs,
            )

        # plot bottom
        for gene in bottom_genes:
            fname = custom_fn_template(var, "bottom", gene)
            troutpy.pl.spatial_inout_expression(
                sdata,
                gene,
                spot_size_cells=spot_size_cells,
                spot_size_extra=spot_size_extra,
                save=save,
                figures_path=output_dir,
                custom_plot_filename=fname,
                **plot_kwargs,
            )


import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc


def estimate_environment_neighbors(
    sdata, cell_key="table", env_key="segmentation_free_table", min_counts=5, plot=True
):
    """
    Estimate the number of neighbors from the environment to approximate the expression level of cells.

    Parameters
    ----------
    sdata : SpatialData object or dict-like
        A data container with AnnData objects under `cell_key` and `env_key`.
    cell_key : str
        Key for the segmented cell-level AnnData object.
    env_key : str
        Key for the segmentation-free (environment-level) AnnData object.
    min_counts : int
        Minimum number of counts required to keep a cell/environment spot.
    plot : bool
        If True, plots histograms and spatial plots.

    Returns
    -------
    int
        Estimated number of neighbors needed from the environment to approximate cell-level expression.
    """
    # Filter environment table
    sc.pp.filter_cells(sdata[env_key], min_counts=min_counts)

    if plot:
        # Spatial plot of environment
        sc.pl.spatial(sdata[env_key], color="n_counts", spot_size=20, title="Environment n_counts")

        # Histograms
        plt.figure(figsize=(8, 5))
        plt.hist(sdata[cell_key].obs["n_counts"], bins=100, alpha=0.5, label="Cells")
        plt.hist(sdata[env_key].obs["n_counts"], bins=100, alpha=0.5, label="Environment")
        plt.xlabel("n_counts")
        plt.ylabel("Frequency")
        plt.legend()
        plt.title("n_counts distributions")
        plt.tight_layout()
        plt.show()

    # Median counts
    median_cell = np.median(sdata[cell_key].obs["n_counts"])
    median_env = np.median(sdata[env_key].obs["n_counts"])

    # Estimate neighbors
    nneighbors = int(np.round(median_cell / median_env))
    return nneighbors
