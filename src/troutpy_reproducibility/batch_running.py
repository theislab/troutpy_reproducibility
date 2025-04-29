import scanpy as sc
import os

def clustering_pipeline(sdata,
                                  clustering_resolution=0.8,
                                  minimum_genes=10,
                                  minimum_counts=30,
                                  minimum_cells=3,
                                  n_neighbors=10,
                                  saving_path=''):
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
    sc.tl.dendrogram(adata, groupby='leiden')
    sc.tl.rank_genes_groups(adata, groupby='leiden')

    if saving_path:
        os.makedirs(saving_path, exist_ok=True)
        sc.pl.umap(adata, color='leiden', save=os.path.join(saving_path, 'umap_leiden.png'))
        sc.pl.spatial(adata, color='leiden', spot_size=40, save=os.path.join(saving_path, 'spatial_leiden.png'))
        sc.pl.rank_genes_groups_dotplot(adata, groupby='leiden', n_genes=3, save=os.path.join(saving_path, 'rank_genes_dotplot.png'))
    else:
        sc.pl.umap(adata, color='leiden')
        sc.pl.spatial(adata, color='leiden', spot_size=40)
        sc.pl.rank_genes_groups_dotplot(adata, groupby='leiden', n_genes=3)

    return adata
