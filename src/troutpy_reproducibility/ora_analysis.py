"""Enrichment of uRNA structure types for extracellular-vesicle and synaptic gene sets.

Implements the Methods section "Overrepresentation and enrichment analysis against
existing databases" (ED Fig. 3H-I, 4F): a hypergeometric over-representation test
against a curated Vesiclepedia mRNA set (:func:`calculate_ev_ora`) and an
over-representation test against SynGO cellular-component terms
(:func:`calculate_syngo_ora`).
"""
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
from spatialdata import SpatialData

def calculate_ev_ora(
    sdata: SpatialData,
    vesiclepedia_path: str,
    table_key: str = "structure_table",
    cluster_key: str = "leiden",
    species: str = "Homo sapiens",
    min_experiments: int = 2,
    case_normalization: str = "upper",
    alpha_marker: float = 0.05,
    min_logfc: float = 0.0,
) -> pd.DataFrame:
    """Performs Over-Representation Analysis (ORA) via a Hypergeometric Test

    to evaluate EV mRNA enrichment per cluster, bypassing GSEApy completely.

    Parameters:
    -----------
    sdata : SpatialData
        The SpatialData object.
    vesiclepedia_path : str
        Path to the VESICLEPEDIA text database file.
    table_key : str, default 'structure_table'
        The key in sdata.tables where the AnnData object is located.
    cluster_key : str, default 'leiden'
        The obs column inside the table defining cluster groups.
    species : str, default 'Homo sapiens'
        Target species to filter from Vesiclepedia (e.g., 'Homo sapiens', 'Mus musculus').
    min_experiments : int, default 2
        Minimum number of unique experiments an EV gene must appear in to be high-confidence.
    case_normalization : str, default 'upper'
        Case modifier for database strings ('upper', 'capitalize', 'lower', or None).
    alpha_marker : float, default 0.05
        The adjusted p-value threshold used to define significant cluster marker genes.
    min_logfc : float, default 0.0
        The minimum log-fold change to define active, upregulated marker genes.

    Returns:
    --------
    pd.DataFrame: A sorted summary table containing enrichment statistics, p-values, 
                  FDR corrections, and overlapping gene lists for each cluster.
    """
    # --- STEP 1: VALIDATE INPUTS ---
    if table_key not in sdata.tables:
        raise KeyError(f"'{table_key}' not found in sdata.tables.")
    
    adata = sdata.tables[table_key]

    if cluster_key not in adata.obs.columns:
        raise KeyError(f"Cluster key '{cluster_key}' not found in sdata.tables['{table_key}'].obs.")

    # --- STEP 2: CURATE HIGH-CONFIDENCE EV GENE SET ---
    print(f"Loading Vesiclepedia and filtering for species: {species}...")
    vesicle_df = pd.read_csv(vesiclepedia_path, sep="\t")

    quality_methods = ["RNA-Seq", "Microarray", "Next-generation sequencing"]
    filtered_df = vesicle_df[
        (vesicle_df["CONTENT TYPE"] == "mRNA")
        & (vesicle_df["SPECIES"] == species)
        & (
            vesicle_df["METHODS"].str.contains(
                "|".join(quality_methods), case=False, na=False
            )
        )
    ].copy()

    filtered_df = filtered_df.dropna(subset=["GENE SYMBOL"])
    filtered_df["GENE SYMBOL"] = filtered_df["GENE SYMBOL"].astype(str)

    if case_normalization == "upper":
        filtered_df["GENE SYMBOL"] = filtered_df["GENE SYMBOL"].str.upper()
    elif case_normalization == "capitalize":
        filtered_df["GENE SYMBOL"] = filtered_df["GENE SYMBOL"].str.capitalize()
    elif case_normalization == "lower":
        filtered_df["GENE SYMBOL"] = filtered_df["GENE SYMBOL"].str.lower()

    gene_counts = filtered_df.groupby("GENE SYMBOL")["EXPERIMENT ID"].nunique()
    high_confidence_genes = gene_counts[
        gene_counts >= min_experiments
    ].index.tolist()

    print(f"Loaded {len(high_confidence_genes)} high-confidence database genes.")

    # --- STEP 3: ESTABLISH THE HYPERGEOMETRIC BACKGROUND ---
    # Convert dataset features to matching case string array
    if case_normalization == "upper":
        sdata_features = adata.var_names.str.upper()
    elif case_normalization == "capitalize":
        sdata_features = adata.var_names.str.capitalize()
    elif case_normalization == "lower":
        sdata_features = adata.var_names.str.lower()
    else:
        sdata_features = adata.var_names.copy()

    # M = Total background universe of genes available in your spatial data profile
    all_background_genes = set(sdata_features)
    M = len(all_background_genes)

    # n = Number of true high-confidence EV genes mapped inside your dataset's genome room
    ev_set = set(high_confidence_genes).intersection(all_background_genes)
    n = len(ev_set)

    print(f"Dataset Background Universe (M): {M} unique genes.")
    print(f"Intersected EV Target Genes (n): {n} genes mapped to dataset features.")
    
    if n == 0:
        print("🛑 ERROR: 0 genes overlap between the database and your dataset array.")
        print(f"Sample data features: {list(adata.var_names[:5])}")
        print(f"Sample DB features:   {high_confidence_genes[:5]}")
        return pd.DataFrame()

    # --- STEP 4: RANK GENES VIA WILCOXON ---
    print(f"Calculating marker profiles using Wilcoxon grouping by '{cluster_key}'...")
    sc.tl.rank_genes_groups(adata, groupby=cluster_key, method="wilcoxon")

    # --- STEP 5: COMPUTE ENRICHMENT PER CLUSTER ---
    ora_results = []
    unique_clusters = [c for c in adata.obs[cluster_key].unique() if pd.notna(c)]

    for cluster in unique_clusters:
        dedf = sc.get.rank_genes_groups_df(adata, group=str(cluster))
        
        # Casing matching adjustment
        if case_normalization == "upper":
            dedf["names"] = dedf["names"].astype(str).str.upper()
        elif case_normalization == "capitalize":
            dedf["names"] = dedf["names"].astype(str).str.capitalize()
        elif case_normalization == "lower":
            dedf["names"] = dedf["names"].astype(str).str.lower()

        # Isolate significantly upregulated marker genes for this specific cluster 
        sig_markers = dedf[(dedf["pvals_adj"] < alpha_marker) & (dedf["logfoldchanges"] > min_logfc)]
        cluster_genes = set(sig_markers["names"])
        
        # N = Total count of passing marker genes representing this single cluster profile
        N = len(cluster_genes)
        
        # x = Count of matching cluster markers that belong to the active EV set
        overlap_genes = cluster_genes.intersection(ev_set)
        x = len(overlap_genes)
        
        if N == 0:
            print(f"⚠️ Cluster '{cluster}' has 0 significant upregulated markers. Skipping.")
            continue

        # Hypergeometric test: Survival function (sf) yields P[X >= x]
        # We supply x - 1 because the calculation bounds are strictly greater than exclusive
        p_val = hypergeom.sf(x - 1, M, n, N)

        ora_results.append({
            "cluster": cluster,
            "cluster_marker_count": N,
            "ev_overlap_count": x,
            "expected_overlap": round((n * N) / M, 2),
            "p_value": p_val,
            "overlapping_genes": ", ".join(list(overlap_genes))
        })

    if not ora_results:
        print("No valid cluster results calculated.")
        return pd.DataFrame()

    # --- STEP 6: COMPILE RESULTS AND APPLY FDR MULTI-TEST CORRECTION ---
    ora_df = pd.DataFrame(ora_results)
    
    # Benjamini-Hochberg FDR correction across all parsed cluster tests
    ora_df["fdr_bh"] = multipletests(ora_df["p_value"], method="fdr_bh")[1]
    ora_df["minus_log10_fdr"] = -np.log10(ora_df["fdr_bh"] + 1e-300)
    
    # Sort output dataframe prioritizing statistical confidence
    ora_df = ora_df.sort_values("p_value", ascending=True).reset_index(drop=True)
    
    print("\nEnrichment analysis complete.")
    return ora_df


import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import hypergeom
from spatialdata import SpatialData
from statsmodels.stats.multitest import multipletests


def calculate_syngo_ora(
    sdata: SpatialData,
    syngo_anno_path: str,
    table_key: str = "structure_table",
    cluster_key: str = "structure_annotation",
    go_domain: str = "CC",
    case_normalization: str = "upper",
    alpha_marker: float = 0.05,
    min_logfc: float = 0.0,
    min_term_size: int = 3,
) -> pd.DataFrame:
    """Performs Over-Representation Analysis (ORA) via Hypergeometric Tests

    to evaluate SynGO pathway enrichment per cluster, bypassing GSEApy.

    Parameters:
    -----------
    sdata : SpatialData
        The SpatialData object.
    syngo_anno_path : str
        Path to the SynGO 'annotations.xlsx' file.
    table_key : str, default 'structure_table'
        The key in sdata.tables where the AnnData object is located.
    cluster_key : str, default 'structure_annotation'
        The obs column inside the table defining cluster groups.
    go_domain : str, default 'CC'
        SynGO domain filter ('CC' for Cellular Component, 'BP' for Biological Process).
    case_normalization : str, default 'upper'
        Case modifier for database strings ('upper', 'capitalize', 'lower', or None).
    alpha_marker : float, default 0.05
        The adjusted p-value threshold used to define significant cluster marker genes.
    min_logfc : float, default 0.0
        The minimum log-fold change to define active, upregulated marker genes.
    min_term_size : int, default 3
        Minimum number of background-intersected genes a SynGO term must have to be tested.

    Returns:
    --------
    pd.DataFrame: A sorted table containing enrichment stats, p-values, FDR corrections,
                  and overlapping gene lists for each cluster-term combination.
    """
    # --- STEP 1: VALIDATE INPUTS ---
    if table_key not in sdata.tables:
        raise KeyError(f"'{table_key}' not found in sdata.tables.")

    adata = sdata.tables[table_key]

    if cluster_key not in adata.obs.columns:
        raise KeyError(
            f"Cluster key '{cluster_key}' not found in sdata.tables['{table_key}'].obs."
        )

    # --- STEP 2: CURATE SYNGO GENE SETS ---
    print(f"Loading SynGO annotations from: {syngo_anno_path}...")
    df_anno = pd.read_excel(syngo_anno_path)

    if go_domain:
        df_anno = df_anno[df_anno["go_domain"] == go_domain]

    # Group into dictionary of terms
    raw_syngo_gs = (
        df_anno.groupby("go_name")["hgnc_symbol"]
        .apply(lambda x: list(set(x.dropna().unique())))
        .to_dict()
    )

    # Apply case transformations to match target datasets
    syngo_gs = {}
    for term, genes in raw_syngo_gs.items():
        if case_normalization == "upper":
            syngo_gs[term] = [str(g).upper() for g in genes]
        elif case_normalization == "capitalize":
            syngo_gs[term] = [str(g).capitalize() for g in genes]
        elif case_normalization == "lower":
            syngo_gs[term] = [str(g).lower() for g in genes]
        else:
            syngo_gs[term] = [str(g) for g in genes]

    print(f"Loaded {len(syngo_gs)} unique SynGO terms for evaluation.")

    # --- STEP 3: ESTABLISH THE HYPERGEOMETRIC BACKGROUND ---
    if case_normalization == "upper":
        sdata_features = adata.var_names.str.upper()
    elif case_normalization == "capitalize":
        sdata_features = adata.var_names.str.capitalize()
    elif case_normalization == "lower":
        sdata_features = adata.var_names.str.lower()
    else:
        sdata_features = adata.var_names.copy()

    all_background_genes = set(sdata_features)
    M = len(all_background_genes)
    print(f"Dataset Background Universe (M): {M} unique genes.")

    # Filter gene sets to preserve background-relevant features
    filtered_syngo_gs = {}
    for term, genes in syngo_gs.items():
        intersected = set(genes).intersection(all_background_genes)
        if len(intersected) >= min_term_size:
            filtered_syngo_gs[term] = intersected

    print(
        f"Retained {len(filtered_syngo_gs)} SynGO terms passing the min_term_size={min_term_size} threshold."
    )

    if not filtered_syngo_gs:
        print("🛑 ERROR: No SynGO terms overlap with your dataset features.")
        return pd.DataFrame()

    # --- STEP 4: RANK GENES VIA WILCOXON ---
    print(
        f"Calculating marker profiles using Wilcoxon grouping by '{cluster_key}'..."
    )
    sc.tl.rank_genes_groups(adata, groupby=cluster_key, method="wilcoxon")

    # --- STEP 5: COMPUTE MULTI-TERM ENRICHMENT PER CLUSTER ---
    ora_results = []
    unique_clusters = [
        c for c in adata.obs[cluster_key].unique() if pd.notna(c)
    ]

    for cluster in unique_clusters:
        dedf = sc.get.rank_genes_groups_df(adata, group=str(cluster))

        if case_normalization == "upper":
            dedf["names"] = dedf["names"].astype(str).str.upper()
        elif case_normalization == "capitalize":
            dedf["names"] = dedf["names"].astype(str).str.capitalize()
        elif case_normalization == "lower":
            dedf["names"] = dedf["names"].astype(str).str.lower()

        # Extract significant markers for this single cluster
        sig_markers = dedf[
            (dedf["pvals_adj"] < alpha_marker) & (dedf["logfoldchanges"] > min_logfc)
        ]
        cluster_genes = set(sig_markers["names"])
        N = len(cluster_genes)

        if N == 0:
            print(
                f"⚠️ Cluster '{cluster}' has 0 significant upregulated markers. Skipping."
            )
            continue

        # Test every active SynGO pathway against this specific cluster
        for term, term_genes in filtered_syngo_gs.items():
            n = len(term_genes)  # Number of successes in universe for this term
            overlap_genes = cluster_genes.intersection(term_genes)
            x = len(overlap_genes)  # Observed overlap

            # Hypergeometric survival function
            p_val = hypergeom.sf(x - 1, M, n, N)

            ora_results.append(
                {
                    "cluster": cluster,
                    "syn_go_term": term,
                    "cluster_marker_count": N,
                    "term_background_count": n,
                    "observed_overlap_count": x,
                    "expected_overlap": round((n * N) / M, 2),
                    "enrichment_ratio": (
                        (x / ((n * N) / M)) if (n * N) / M > 0 else 0.0
                    ),
                    "p_value": p_val,
                    "overlapping_genes": ", ".join(list(overlap_genes)),
                }
            )

    if not ora_results:
        print("No valid enrichment tracks calculated.")
        return pd.DataFrame()

    # --- STEP 6: COMPILE RESULTS AND APPLY GLOBAL FDR MULTI-TEST CORRECTION ---
    ora_df = pd.DataFrame(ora_results)

    # Correction across all combinations tested
    ora_df["fdr_bh"] = multipletests(ora_df["p_value"], method="fdr_bh")[1]
    ora_df["minus_log10_fdr"] = -np.log10(ora_df["fdr_bh"] + 1e-300)

    # Sort prioritizing statistical confidence
    ora_df = ora_df.sort_values("p_value", ascending=True).reset_index(
        drop=True
    )

    print("\nSynGO ORA enrichment analysis complete.")
    return ora_df
