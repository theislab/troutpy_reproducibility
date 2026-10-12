#!/usr/bin/env python3
"""uRNA analysis pipeline, variant used for the CosMx pancreas dataset.

Identical to urna_analysis_pipeline.py plus sanitize_spatialdata_tables(), which casts
pyarrow-backed strings and de-duplicates case-variant obs columns so the large CosMx
pancreas tables pass SpatialData schema validation when written.

Steps (see Methods):
  1. Platform-specific harmonisation (CosMx / MERSCOPE cell-id conventions; nuclei
     segmentation used as "overlaps_cell" when metadata.csv says so).
  2. Scanpy clustering of segmented cells and cell morphology metrics.
  3. Local transcript density: Bayesian high/moderate-density/background classes
     (tl.density_similarity, radius from tl.calculate_heuristic_radius_by_cells, k=10).
  4. Cell-like regions with Sainsc (pp.segmentation_free_sainsc, bin size from
     pp.find_optimal_segmentation_free_bin_size) and the Bayesian uRNA probability
     (pp.define_urna_probability / pp.define_urna).
  5. Per-gene uRNA metrics: noise over control probes (tl.quantify_overexpression),
     uRNA proportion (tl.extracellular_enrichment), uRNA bins (pp.aggregate_urna),
     Moran's I (tl.spatial_variability), intracellular-uRNA correlation
     (tl.in_out_correlation) and colocalization.
  6. Staining intensity per transcript and staining-based cell-likeness (if images exist).
  7. uRNA source score (tl.urna_source_score, lambda = --lambda_decay, default 0.5).
  8. uRNA structures (tl.segment_protrusions: DBSCAN eps 2, no minimum size,
     connectivity 4 um, 10 neighbouring cells for the neighborhood score).
  9. Rayleigh-based isotropic diffusion test (tl.assess_diffusion) on non-high-density uRNA.

Input : <inpath>/data.zarr (+ optional <inpath>/metadata.csv)
Output: <inpath>/data_with_segfree_2.zarr, data_processed_full.zarr,
        data_with_source_score.zarr (final), and <inpath>/figures/
        (pipeline_stats.csv, transcript_categories.csv, QC plots).

Usage : python urna_analysis_pipeline_cosmx_pancreas.py <inpath> [options]   (see the *.sbatch files next to this script)
Env   : troutpy_drvi (see envs/), with troutpy and troutpy_reproducibility installed.
"""
import os
import sys
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scanpy as sc
import spatialdata as sd
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation
import spatialdata_io
import scipy.sparse as sp

# --- CONFIGURE LOCAL DEVELOPER MODULE PATHS ---
# Local troutpy reproducibility path
from troutpy_reproducibility.batch_running import *

    
# Import local pipeline submodules directly
from troutpy import pl
from troutpy import tl
from troutpy import pp


def sanitize_spatialdata_tables(sdata):
    """
    Sanitizes all AnnData tables within a SpatialData object to ensure strict compatibility
    with SpatialData schema validation constraints (resolves PyArrow strings & case-variant/duplicate columns).
    """
    if hasattr(sdata, "tables") and sdata.tables:
        for table_name, adata in sdata.tables.items():
            print(f"Sanitizing table '{table_name}' for validation compatibility...")
            
            # --- 1. CLEAN OBS COLUMNS ---
            for col in adata.obs.columns:
                if isinstance(adata.obs[col].array, pd.arrays.ArrowStringArray) or str(adata.obs[col].dtype).startswith("string[pyarrow]"):
                    adata.obs[col] = adata.obs[col].astype(str)
            
            obs_groups = defaultdict(list)
            for col in adata.obs.columns:
                obs_groups[col.lower()].append(col)
                
            new_cols = list(adata.obs.columns)
            seen_lower = set()
            for i, col in enumerate(new_cols):
                col_lower = col.lower()
                variants = obs_groups[col_lower]
                if len(variants) > 1:
                    keep_col = col_lower if col_lower in variants else variants[0]
                    if col != keep_col or col_lower in seen_lower:
                        suffix = 1
                        candidate = f"{col}_conflicting_{suffix}"
                        while candidate.lower() in seen_lower:
                            suffix += 1
                            candidate = f"{col}_conflicting_{suffix}"
                        new_cols[i] = candidate
                        col_lower = candidate.lower()
                seen_lower.add(col_lower)
            adata.obs.columns = new_cols
                
            # Clean obs index names
            if isinstance(adata.obs.index.array, pd.arrays.ArrowStringArray) or str(adata.obs.index.dtype).startswith("string[pyarrow]"):
                adata.obs.index = adata.obs.index.astype(str)
            if adata.obs.index.name is not None and adata.obs.index.name.lower() in [c.lower() for c in adata.obs.columns]:
                adata.obs.index.name = None

            # --- 2. CLEAN VAR COLUMNS ---
            for col in adata.var.columns:
                if isinstance(adata.var[col].array, pd.arrays.ArrowStringArray) or str(adata.var[col].dtype).startswith("string[pyarrow]"):
                    adata.var[col] = adata.var[col].astype(str)
            
            var_groups = defaultdict(list)
            for col in adata.var.columns:
                var_groups[col.lower()].append(col)
                
            new_var_cols = list(adata.var.columns)
            seen_var_lower = set()
            for i, col in enumerate(new_var_cols):
                col_lower = col.lower()
                variants = var_groups[col_lower]
                if len(variants) > 1:
                    keep_col = col_lower if col_lower in variants else variants[0]
                    if col != keep_col or col_lower in seen_var_lower:
                        suffix = 1
                        candidate = f"{col}_conflicting_{suffix}"
                        while candidate.lower() in seen_var_lower:
                            suffix += 1
                            candidate = f"{col}_conflicting_{suffix}"
                        new_var_cols[i] = candidate
                        col_lower = candidate.lower()
                seen_var_lower.add(col_lower)
            adata.var.columns = new_var_cols
                
            # Clean var index names
            if isinstance(adata.var.index.array, pd.arrays.ArrowStringArray) or str(adata.var.index.dtype).startswith("string[pyarrow]"):
                adata.var.index = adata.var.index.astype(str)
            if adata.var.index.name is not None and adata.var.index.name.lower() in [c.lower() for c in adata.var.columns]:
                adata.var.index.name = None


def run_troutpy_pipeline(inpath, args):
    """
    Generalized troutpy analysis pipeline integrating platform-specific preprocessing,
    expanded hyperparameter options, and robust error handling for staining steps.
    """
    # --- PATH PREPARATION ---
    figures_path = os.path.join(inpath, "figures")
    os.makedirs(figures_path, exist_ok=True)
    
    # Input file is dynamically loaded from the provided directory as 'data.zarr'
    zarr_path = os.path.join(inpath, "data.zarr")
    meta_path = os.path.join(inpath, "metadata.csv")

    print(f"Loading spatial data from: {zarr_path}")
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Critical Error: 'data.zarr' not found in {inpath}")
        
    sdata = sd.read_zarr(zarr_path) if hasattr(sd, "read_zarr") else SpatialData.read(zarr_path)
    
    # --- METADATA & PLATFORM SPECIFIC PREPROCESSING ---
    if os.path.exists(meta_path):
        metadata = pd.read_csv(meta_path)
        
        # Technology-specific coordinate/identifier preprocessing
        if len(metadata) > 0 and "technology" in metadata.columns:
            tech = metadata.loc[0, 'technology']
            
            if tech == 'CosMx':
                print("Applying platform-specific adjustments for CosMx data...")
                transcripts = sdata['transcripts'].compute().rename(columns={'cell_ID': 'cell_id'})
                transcripts['cell_id'] = transcripts['cell_id'].astype(str).replace(['0', '0.0'], 'UNASSIGNED')
                
                sdata['table'].obs = sdata['table'].obs.rename(columns={'cell_ID': 'cell_id'})
                sdata['table'].obs['cell_id'] = sdata['table'].obs['cell_id'].astype(str)
                sdata["transcripts"] = sd.models.PointsModel.parse(transcripts)
                sdata['table'].uns['spatialdata_attrs']['instance_key'] = 'cell_id'
                
            elif tech == 'MERSCOPE':
                print("Applying platform-specific adjustments for MERSCOPE data...")
                transcripts = sdata['transcripts'].compute()
                transcripts['cell_id'] = transcripts['cell_id'].astype(str).replace(['-1'], 'UNASSIGNED')
                sdata["transcripts"] = sd.models.PointsModel.parse(transcripts)
                sdata['table'].obs['cell_id'] = sdata['table'].obs['cell_id'].astype(str)
        
        # Segmentation fallback logic
        if "segmentation_method" in metadata.columns and len(metadata) > 0:
            if "nuclei" in str(metadata["segmentation_method"][0]):
                transcripts = sdata["transcripts"].compute()
                transcripts["overlaps_cell"] = transcripts["overlaps_nucleus"].astype(bool)
                sdata["transcripts"] = sd.models.PointsModel.parse(transcripts)
                print("Redefined overlaps_cell in transcripts using nuclei segmentation.")
    else:
        print(f"No metadata.csv found at {meta_path}. Skipping preprocessing & fallback logic.")

    # --- DENSE TO SPARSE MATRIX CONVERSION ---
    if isinstance(sdata.table.X, np.ndarray):
        print("Converting dense matrix to CSR sparse matrix...")
        sdata.table.X = sp.csr_matrix(sdata.table.X)

    # --- PIPELINE EXECUTION ---
    clustering_pipeline(sdata)
    try:
        pp.add_morphological_metrics(sdata)
    except Exception as e: 
        print(f"Warning: Morphological metrics calculation failed ({e}). Check data integrity.")
    
    opt_r = tl.calculate_heuristic_radius_by_cells(sdata, k=10, n_cells=500)
    tl.density_similarity(sdata, radius=opt_r, process_all=True, segmentation_key='overlaps_cell')

    # Find optimal bin size using an expanded range of bins and save visual distributions
    results, met_df, binsize_selected = pp.find_optimal_segmentation_free_bin_size(
        sdata, bin_sizes=[0.5, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20], roi_size=400
    )
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=results, x="bin_size", y="cosine_sim", hue="is_cellular", split=True)
    plt.title("Cosine Similarity Distribution: Intra vs Extra-cellular")
    
    violin_out_path = os.path.join(figures_path, "cosine_similarity_distribution.png")
    plt.savefig(violin_out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved optimal bin size violin plot to: {violin_out_path}")

    # Segmentation-free SAiNSC using local modules
    pp.segmentation_free_sainsc(
        sdata, binsize=binsize_selected, celltype_key="leiden", 
        background_filter=args.background_filter, gaussian_kernel_key=args.gaussian_kernel_key, 
        n_threads=args.sainsc_threads
    )
    pp.define_urna_probability(sdata)
    pp.define_urna(sdata, method='sainsc')
    
    # Histogram of scores
    pl.histogram(sdata, x="prob_is_urna", hue="overlaps_cell", palette="troutpy", save=True, figures_path=figures_path)
    pl.histogram(sdata, x="cosine_similarity", hue="overlaps_cell", palette="troutpy", save=True, figures_path=figures_path)
    
    # Crosstabs and charts
    pl.crosstab(
        sdata, yvar="extracellular", xvar="overlaps_cell", normalize=False, 
        cmap="troutpy", kind="barh", stacked=True, save=True, figures_path=figures_path, figsize=(5, 2)
    )
    pl.crosstab(
        sdata, xvar="extracellular", yvar="overlaps_cell", normalize=True, 
        cmap="Aquas", kind="heatmap", stacked=True, save=True, figures_path=figures_path
    )
    
    # Sanitize and write step 1
    sanitize_spatialdata_tables(sdata)
    zarr_path_segfree = os.path.join(inpath, "data_with_segfree_2.zarr")
    sdata.write(zarr_path_segfree, overwrite=True) 

    # Single-value metrics computation
    proportion_of_segmented = (sdata["transcripts"]["overlaps_cell"].compute() == True).sum() / len(sdata["transcripts"]["extracellular"].compute())
    proportion_of_extracellular = (sdata["transcripts"]["extracellular"].compute() == True).sum() / len(sdata["transcripts"]["extracellular"].compute())
    
    tl.quantify_overexpression(sdata, codeword_key="control_probe", control_codewords=[True])
    tl.extracellular_enrichment(sdata, gene_key="gene")

    # Aggregate extracellular metrics
    pp.aggregate_urna(sdata, gene_key="gene", square_size=args.square_size)
    
    # Environment Neighbor Estimation
    nneighbors = estimate_environment_neighbors(
        sdata, cell_key="table", env_key="segmentation_free_table", min_counts=args.min_counts_env, plot=False
    )
    
    tl.spatial_variability(sdata, gene_key="gene", n_neighbors=args.spatial_var_neighbors, square_size=2 * args.square_size)
    tl.in_out_correlation(sdata, n_neighbors=nneighbors)
    tl.spatial_colocalization(sdata, threshold_colocalized=args.colocalization_threshold, copy=False)
    
    sdata['xrna_metadata'].var['control_probe'] = ~sdata['xrna_metadata'].var['is_control']
    sdata["xrna_metadata"].var["control_probe"] = sdata["xrna_metadata"].var["control_probe"].astype(bool)
    
    # Calculate Noise & Gene Metadata profiles
    gene_info = sdata["xrna_metadata"].var[sdata["xrna_metadata"].var['is_control'] == False]
    genes_prop_over_noise = (gene_info['p_val_noise'] < args.max_p_val_noise).sum() / gene_info.shape[0]
    non_noise_genes = gene_info[gene_info['p_val_noise'] < args.max_p_val_noise].index
    all_gene_ids = sdata['transcripts']['gene'].compute()
    noise_RNA_proportion = np.sum(all_gene_ids.isin(non_noise_genes)) / len(all_gene_ids)
    
    # Sanitize and write step 2
    sanitize_spatialdata_tables(sdata)
    sdata.write(os.path.join(inpath, "data_processed_full.zarr"), overwrite=True)

    # --- QUANTIFY STAININGS (WITH FALLBACK) ---
    try:
        tl.image_intensities_per_transcript(
            sdata, image_key="morphology_focus", scale="scale0", 
            transcript_key="transcripts", extracellular=False, copy=False
        )
        pp.define_urna_probability_stainings(sdata)
    except Exception as e:
        print("Not possible to define probabilities based on stainings.")

    # Source score modelling optimization
    min_enrichment = sdata.points['transcripts'][
        sdata.points['transcripts']['density_similarity_score'] > 0.5
    ]['enrichment_over_random'].min().compute()

    tl.urna_source_score(
        sdata, chunk_size=100000, cell_type_col='leiden', max_dist=200, 
        lmbda=args.lambda_decay, max_k=10, ambient_floor=min_enrichment, 
        signal_threshold=10, residual=0.1, copy=False
    )

    try:
        table_name = "image_intensity_per_transcript"
        sdata.tables[table_name].obs['gene'] = (
            sdata.tables[table_name].obs['gene']
            .astype(str)
            .replace('nan', 'unknown') 
        )
    except Exception as e:
        print("Could not process image intensity per transcript table for source score modelling.")

    # Segment protrusions with adjusted parameters
    tl.segment_protrusions(
        sdata, n_neighbors=10, cluster_eps=2.0, 
        connectivity_threshold=4.0, min_samples=1
    )

    # Assess diffusion
    tl.assess_diffusion(
        sdata, gene_key="gene", distance_key="distance_to_source", 
        filters={'enrichment_class': ('High Density', False)}
    )

    # Calculate categories and generate final CSV stats
    categories = pp.get_transcript_categories(sdata)

    # Final Sanitize and write step 3
    sanitize_spatialdata_tables(sdata)
    sdata.write(os.path.join(inpath, "data_with_source_score.zarr"), overwrite=True)

    stats_df = pd.DataFrame({
        "Metric": [
            "proportion_of_segmented", 
            "proportion_of_extracellular", 
            "genes_prop_over_noise", 
            "noise_RNA_proportion"
        ],
        "Value": [
            proportion_of_segmented, 
            proportion_of_extracellular, 
            genes_prop_over_noise, 
            noise_RNA_proportion
        ]
    })
    
    stats_df.to_csv(os.path.join(figures_path, "pipeline_stats.csv"), index=False)
    categories.to_csv(os.path.join(figures_path, "transcript_categories.csv"), index=False)
    print(f"Pipeline completed smoothly. Figures and metrics saved in {figures_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run troutpy spatial analysis pipeline on a targeted data folder.")
    
    # Positional required argument (The folder containing data.zarr)
    parser.add_argument("inpath", type=str, help="Path to the directory containing 'data.zarr'")
    
    # Optional hyperparameter flags with defaults
    parser.add_argument("--background_filter", type=float, default=0.4, help="Sainsc background filter (fraction of max KDE).")
    parser.add_argument("--gaussian_kernel_key", type=float, default=2.0, help="Sainsc Gaussian kernel bandwidth (um).")
    parser.add_argument("--sainsc_threads", type=int, default=16, help="Threads used by Sainsc.")
    parser.add_argument("--square_size", type=int, default=10, help="Side (um) of the bins used to aggregate uRNA (pp.aggregate_urna).")
    parser.add_argument("--min_counts_env", type=int, default=5, help="Minimum counts per uRNA bin when estimating the neighbourhood size for in/out correlation.")
    parser.add_argument("--spatial_var_neighbors", type=int, default=10, help="Number of spatial neighbours for Moran's I.")
    parser.add_argument("--colocalization_threshold", type=int, default=1, help="Threshold used by tl.spatial_colocalization.")
    parser.add_argument("--lambda_decay", type=float, default=0.5, help="Distance-decay rate of the uRNA source score (manuscript: 0.5).")
    parser.add_argument("--max_p_val_noise", type=float, default=0.05, help="P-value threshold for a gene to be called above the control-probe noise.")

    parsed_args = parser.parse_args()
    
    # Execute pipeline
    run_troutpy_pipeline(inpath=parsed_args.inpath, args=parsed_args)