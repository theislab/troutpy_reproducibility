"""Transcript-category summaries shared by the cross-dataset analysis notebooks.

Used by ``notebooks/3_analysis_of_results`` to classify every transcript by the two
uRNA criteria (local density and transcriptomic similarity to cells) and to compute
transcript-count distributions of cells versus uRNA structures.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spatialdata as sd

def process_transcript_classifications(sdata, dataset_name):
    """Classify transcripts by both uRNA criteria and plot their agreement.

    Criterion 1 is the local-density class (``enrichment_class``: High / Moderate
    density, Background); criterion 2 is transcriptomic similarity to segmented
    cells (cell-like vs. non-cell-like, from the ``extracellular`` flag).
    Intracellular transcripts (``overlaps_cell``) are kept as their own class. A
    heatmap of the uRNA cross-classification (fraction of all uRNA) is shown.

    Parameters
    ----------
    sdata
        SpatialData whose ``"transcripts"`` have ``overlaps_cell``,
        ``extracellular`` and ``enrichment_class`` columns.
    dataset_name
        Used in the plot title.

    Returns
    -------
    ``(c1_counts, c2_counts, combined_counts)``: dictionaries of transcript counts
    per criterion-1 class, per criterion-2 class and per combined class
    (e.g. ``"High Density (Cell-like)"``).
    """
    # 1. Load transcripts into memory
    transcripts = sdata['transcripts'].compute()
    
    # 2. Vectorized Classifications with rigorous NaN handling
    # Handle Criterion 1
    c1_raw = np.where(transcripts['overlaps_cell'] == True, 'Intracellular', transcripts['enrichment_class'])
    crit1_series = pd.Series(c1_raw, index=transcripts.index).fillna('Unknown').astype(str)
    
    # Handle Criterion 2
    c2_raw = np.select(
        condlist=[
            transcripts['overlaps_cell'] == True,
            (transcripts['overlaps_cell'] == False) & (transcripts['extracellular'] == True),
            (transcripts['overlaps_cell'] == False) & (transcripts['extracellular'] != True)
        ],
        choicelist=['Intracellular', 'Non-cell-like', 'Cell-like'],
        default='Unclassified'
    )
    crit2_series = pd.Series(c2_raw, index=transcripts.index).fillna('Unknown').astype(str)
    
    # Handle Combined via Pandas native categorical string concatenation (Bypasses ufunc bugs)
    is_cell = (transcripts['overlaps_cell'] == True).values
    
    # Initialize combined series
    combined_series = pd.Series('Intracellular', index=transcripts.index)
    
    # For uRNA rows, concatenate dynamically
    uRNA_mask = ~is_cell
    if uRNA_mask.any():
        uRNA_combined = crit1_series[uRNA_mask].str.cat(crit2_series[uRNA_mask], sep=" (") + ")"
        combined_series.loc[uRNA_mask] = uRNA_combined

    # 3. Extract Headcounts
    c1_counts = crit1_series.value_counts().to_dict()
    c2_counts = crit2_series.value_counts().to_dict()
    comb_counts = combined_series.value_counts().to_dict()
    
    # 4. Generate Heatmap for uRNA
    is_test_urna = ~is_cell
    if is_test_urna.any():
        confusion_matrix_pct = pd.crosstab(
            index=crit1_series[is_test_urna], 
            columns=crit2_series[is_test_urna], 
            normalize='all'
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(confusion_matrix_pct, annot=True, fmt=".1%", cmap="YlGnBu", linewidths=0.5)
        ax.set_title(f'uRNA Cross-Classification Breakdown\nDataset: {dataset_name}', fontsize=11, weight='bold', pad=15)
        ax.set_ylabel('Transcript Density-Based (Criterion 1)', fontsize=10)
        ax.set_xlabel('RNA Local Similarity-Based (Criterion 2)', fontsize=10)
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
    return c1_counts, c2_counts, comb_counts

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from spatialdata import SpatialData


def compute_transcript_distributions(
    sdata: SpatialData,
    cell_table_key: str = "table",
    fragment_table_key: str = "structure_table",
) -> dict:
    """Extracts raw transcript matrices from SpatialData tables, calculates row sums,

    computes frequency binned intervals, and applies a Kolmogorov-Smirnov test.

    Returns:
    --------
    dict: Contains 'cell_counts', 'fragment_counts', 'df_cells', 'df_fragments',
          'ks_stat', and 'ks_p'.
    """
    # 1. Validation Checks
    for key in [cell_table_key, fragment_table_key]:
        if key not in sdata.tables:
            raise KeyError(f"Table key '{key}' not found in sdata.tables.")

    # 2. Total transcript counting function (handles dense & sparse matrices)
    def _get_total_counts(adata):
        X = adata.X
        if hasattr(X, "toarray"):  # Sparse matrix conversion safe-guard
            return np.asarray(X.sum(axis=1)).ravel()
        return np.asarray(X.sum(axis=1)).ravel()

    cell_counts = _get_total_counts(sdata.tables[cell_table_key])
    fragment_counts = _get_total_counts(sdata.tables[fragment_table_key])

    # 3. Frequency Bin Allocation
    edges = [0, 10, 25, 50, 100, 250, 500, 1000, np.inf]
    range_labels = [
        "0–9",
        "10–24",
        "25–49",
        "50–99",
        "100–249",
        "250–499",
        "500–999",
        "≥1000",
    ]

    def _bin_data(arr):
        counts, _ = np.histogram(arr, bins=edges)
        pct = (counts / counts.sum()) * 100
        return pd.DataFrame({"range": range_labels, "n": counts, "pct": pct})

    df_cells = _bin_data(cell_counts)
    df_fragments = _bin_data(fragment_counts)

    # 4. Statistical Tests (KS-test)
    ks_stat, ks_p = stats.ks_2samp(fragment_counts, cell_counts)

    # 5. Print Descriptive Summary to Terminal
    for name, df, arr in [
        ("CELLS", df_cells, cell_counts),
        ("FRAGMENTS", df_fragments, fragment_counts),
    ]:
        print("═" * 55)
        print(f"  {name}  (n = {len(arr):,})")
        print("═" * 55)
        print(df.to_string(index=False))
        print(
            f"  median={np.median(arr):.0f}  mean={arr.mean():.1f}"
            f"  p10={np.percentile(arr, 10):.0f}  p90={np.percentile(arr, 90):.0f}\n"
        )

    print(f"KS two-sample test: D = {ks_stat:.3f}, p = {ks_p:.2e}\n")

    return {
        "cell_counts": cell_counts,
        "fragment_counts": fragment_counts,
        "df_cells": df_cells,
        "df_fragments": df_fragments,
        "ks_stat": ks_stat,
        "ks_p": ks_p}


