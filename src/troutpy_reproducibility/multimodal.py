"""Univariate staining analysis used for the multimodal validation (Fig. 4A-D, ED Fig. 6A).

Workflow (see Methods, "Stainings' quantification and multimodal analysis",
"Univariate analysis"):

1. :func:`analyze_channels_and_get_thresholds` derives one positivity threshold per
   staining channel from the channel's cumulative spatial autocorrelation curve.
2. :func:`compute_transcript_pixel_positivity` flags, per transcript and channel,
   whether the transcript sits on a positive pixel.
3. :func:`assign_transcript_categories` labels every transcript as intracellular,
   cell-like, high-density or other uRNA.
4. :func:`compute_gene_category_pixel_enrichment` tests, per gene, category and
   channel, whether the gene's transcripts land on positive pixels more often than
   expected by chance (label permutation within the category).
"""
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import correlate

def compute_transcript_pixel_positivity(
    sdata,
    df_summary: pd.DataFrame,
    intensity_key: str = "image_intensity_per_transcript",
    extra_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Flag, for each transcript and staining channel, whether it lies on a positive pixel.

    Parameters
    ----------
    sdata
        SpatialData with a ``"transcripts"`` points layer and a per-transcript
        intensity table ``sdata[intensity_key]`` (AnnData whose ``obs`` match the
        transcripts 1:1 and whose ``var`` are the channels), as produced by
        ``troutpy.tl.image_intensities_per_transcript``.
    df_summary
        Per-channel thresholds, as returned by :func:`analyze_channels_and_get_thresholds`
        (columns ``"Channel Name"`` and ``"Intensity Threshold"``).
    intensity_key
        Key of the per-transcript intensity table.
    extra_columns
        Additional transcript columns to carry over.

    Returns
    -------
    ``(trans_df, used_channel_names)``: the transcript table joined with the
    intensities plus one boolean ``positive_<channel>`` column per thresholded
    channel, and the list of those channels.
    """
    base_columns = ["gene", "overlaps_cell", "extracellular", "enrichment_class"]
    if extra_columns:
        base_columns = list(dict.fromkeys(base_columns + extra_columns))

    trans_df = sdata["transcripts"][base_columns].compute()
    trans_df.index = trans_df.index.astype(str)

    intensity_adata = sdata[intensity_key]
    channels = intensity_adata.var_names.tolist()
    intensities_matrix = (
        intensity_adata.X.toarray() if hasattr(intensity_adata.X, "toarray") else intensity_adata.X
    )
    intensities_df = pd.DataFrame(
        intensities_matrix, index=intensity_adata.obs_names.astype(str), columns=channels
    )

    trans_df = trans_df.join(intensities_df, how="inner")

    threshold_dict = dict(zip(df_summary["Channel Name"], df_summary["Intensity Threshold"]))
    used_channel_names = []

    print("=== Flagging per-transcript pixel positivity per channel ===")
    for channel in channels:
        if channel not in threshold_dict:
            continue
        trans_df[f"positive_{channel}"] = trans_df[channel] > threshold_dict[channel]
        used_channel_names.append(channel)
        print(f"-> {channel}: {trans_df[f'positive_{channel}'].mean() * 100:.2f}% of transcripts on positive pixels")

    return trans_df, used_channel_names



def assign_transcript_categories(
    trans_df: pd.DataFrame,
    overlaps_cell_key: str = "overlaps_cell",
    extracellular_key: str = "extracellular",
    enrichment_class_key: str = "enrichment_class",
    high_density_label: str = "High Density",
    category_col: str = "transcript_category",
) -> pd.DataFrame:
    """Assign each transcript to exactly one uRNA category.

    Categories: ``intracellular`` (overlaps a segmented cell), ``cell_like`` (outside
    cells but not extracellular, i.e. in a cell-like region), ``high_density``
    (extracellular and in a High Density region) or ``other``.

    Parameters
    ----------
    trans_df
        Transcript table with the three flag columns below.
    overlaps_cell_key, extracellular_key, enrichment_class_key
        Column names (booleans may be stored as strings).
    high_density_label
        Value of ``enrichment_class_key`` marking high-density transcripts.
    category_col
        Name of the output column.

    Returns
    -------
    A copy of ``trans_df`` with a categorical ``category_col`` column.
    """
    overlaps_cell = trans_df[overlaps_cell_key].astype(str).str.lower() == "true"
    extracellular = trans_df[extracellular_key].astype(str).str.lower() == "true"
    is_high_density = trans_df[enrichment_class_key] == high_density_label

    category = np.full(len(trans_df), "other", dtype=object)
    category[(~overlaps_cell & ~extracellular).to_numpy()] = "cell_like"
    category[(~overlaps_cell & extracellular & is_high_density).to_numpy()] = "high_density"
    category[overlaps_cell.to_numpy()] = "intracellular"

    trans_df = trans_df.copy()
    trans_df[category_col] = pd.Categorical(
        category, categories=["intracellular", "cell_like", "high_density", "other"]
    )
    return trans_df


def compute_gene_category_pixel_enrichment(
    trans_df: pd.DataFrame,
    channel_names: list[str],
    category_col: str = "transcript_category",
    gene_key: str = "gene",
    n_perm: int = 200,
    min_transcripts: int = 20,
    chunk_size: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    """Permutation test of gene enrichment on staining-positive pixels, per category.

    For every (gene, category, channel) the observed percentage of the gene's
    transcripts on positive pixels is compared with a null distribution obtained
    by permuting the positive/negative labels within the category (which keeps the
    category's overall positive rate). Permutations are processed in chunks to keep
    memory bounded.

    Parameters
    ----------
    trans_df
        Output of :func:`compute_transcript_pixel_positivity` +
        :func:`assign_transcript_categories`.
    channel_names
        Channels to test (each needs a ``positive_<channel>`` column).
    category_col, gene_key
        Column names.
    n_perm
        Number of permutations (200 in the manuscript).
    min_transcripts
        Minimum transcripts of a gene within a category to be tested (100 for the
        mouse-brain analysis in the manuscript).
    chunk_size
        Permutations processed at once.
    seed
        Random seed.

    Returns
    -------
    Long DataFrame with ``gene``, ``category``, ``channel``, ``n_transcripts``,
    ``pct_positive_observed``, ``pct_positive_expected_random``,
    ``pct_positive_expected_std``, ``enrichment_ratio`` (observed / expected),
    empirical ``pvalue`` and ``pvalue_adj`` (Benjamini-Hochberg within each
    category and channel).
    """
    # Filter out rows with missing gene names to prevent pd.factorize from yielding -1
    trans_df = trans_df.dropna(subset=[gene_key])

    rng = np.random.default_rng(seed)
    results = []

    for category, cat_df in trans_df.groupby(category_col, observed=True):
        n_cat = len(cat_df)
        if n_cat == 0:
            continue

        gene_codes, gene_index = pd.factorize(cat_df[gene_key].to_numpy())
        n_genes = len(gene_index)
        counts_per_gene = np.bincount(gene_codes, minlength=n_genes)
        keep_genes = np.where(counts_per_gene >= min_transcripts)[0]
        if keep_genes.size == 0:
            continue

        gene_design = sparse.csr_matrix(
            (np.ones(n_cat), (gene_codes, np.arange(n_cat))), shape=(n_genes, n_cat)
        )

        print(f"=== Category '{category}' ({n_cat} transcripts, {keep_genes.size} genes passing min_transcripts) ===")
        for channel in tqdm(channel_names):
            labels = cat_df[f"positive_{channel}"].to_numpy(dtype=np.float32)

            observed_sum = gene_design @ labels
            observed_pct = observed_sum / counts_per_gene * 100

            sum_null = np.zeros(n_genes)
            sumsq_null = np.zeros(n_genes)
            exceed_count = np.zeros(n_genes)
            remaining = n_perm
            while remaining > 0:
                cur = min(chunk_size, remaining)
                null_labels = rng.permuted(np.tile(labels, (cur, 1)), axis=1)
                null_sum = gene_design @ null_labels.T
                null_pct = null_sum / counts_per_gene[:, None] * 100
                sum_null += null_pct.sum(axis=1)
                sumsq_null += (null_pct ** 2).sum(axis=1)
                exceed_count += (null_pct >= observed_pct[:, None]).sum(axis=1)
                remaining -= cur

            expected_pct = sum_null / n_perm
            expected_var = np.maximum(sumsq_null / n_perm - expected_pct ** 2, 0)
            expected_std = np.sqrt(expected_var)
            pvalue = (exceed_count + 1) / (n_perm + 1)

            for gi in keep_genes:
                results.append({
                    "gene": gene_index[gi],
                    "category": category,
                    "channel": channel,
                    "n_transcripts": int(counts_per_gene[gi]),
                    "pct_positive_observed": observed_pct[gi],
                    "pct_positive_expected_random": expected_pct[gi],
                    "pct_positive_expected_std": expected_std[gi],
                    "enrichment_ratio": (observed_pct[gi] / expected_pct[gi]) if expected_pct[gi] > 0 else np.nan,
                    "pvalue": pvalue[gi],
                })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        from statsmodels.stats.multitest import multipletests
        result_df["pvalue_adj"] = np.nan
        for _, idx in result_df.groupby(["category", "channel"]).groups.items():
            result_df.loc[idx, "pvalue_adj"] = multipletests(result_df.loc[idx, "pvalue"], method="fdr_bh")[1]

    return result_df

def analyze_channels_and_get_thresholds(
    sdata,
    image_key: str = "morphology_focus",
    scale_key: str = "scale0",
    roi: tuple[float, float, float, float] | None = None,
    saving_path: Path | str | None = None,
    step_size: int = 50,
    cmap: str = "gray",
    mode: str = "stringent",  # Options: "permissive", "standard", "stringent"
    threshold: float | dict[str, float] | None = None  # <-- Accepts target correlation values (e.g., 0.3)
) -> pd.DataFrame:
    """Derive a positivity threshold for every staining channel from its spatial autocorrelation.

    For increasing upper bounds ``v`` (step ``step_size``), the Pearson correlation
    between each pixel's intensity and the mean of its 8 neighbours is computed over
    all pixels with intensity in ``[0, v]``. Background is spatially incoherent, so
    the curve is low while only background is included and rises once true signal
    enters; the threshold is taken at that inflection (or at a user-given target
    correlation). A three-panel diagnostic figure (image, curve, mask) is drawn per
    channel.

    Parameters
    ----------
    sdata
        SpatialData with a ``"transcripts"`` points layer (used to map ROI
        coordinates to pixels) and the image ``sdata.images[image_key][scale_key]``.
    image_key, scale_key
        Image and resolution level to analyse.
    roi
        Optional ``(xmin, xmax, ymin, ymax)`` crop, in transcript coordinates.
    saving_path
        Folder where the per-channel diagnostic PNGs are written (optional).
    step_size
        Intensity step between evaluated upper bounds.
    cmap
        Colormap for the raw image panel.
    mode
        Automatic rule when ``threshold`` is not given: ``"permissive"`` (first
        point where the curve stops dropping), ``"standard"`` (curve minimum) or
        ``"stringent"`` (steepest rise after the minimum).
    threshold
        Target autocorrelation value (float, or dict per channel); the intensity
        whose cumulative correlation is closest to it is used as threshold.

    Returns
    -------
    DataFrame with one row per channel: ``"Channel Name"``,
    ``"Intensity Threshold"`` and ``"% Positive Pixels (Signal)"``.
    """
    print("\n=== STEP 1: Extracting XY Transcript Data ===")
    if 'transcripts' not in sdata.points:
        raise KeyError("The key 'transcripts' was not found in sdata.points.")
    
    trans_df = sdata.points['transcripts'][['x', 'y']].compute()
    t_xmin, t_xmax = trans_df['x'].min(), trans_df['x'].max()
    t_ymin, t_ymax = trans_df['y'].min(), trans_df['y'].max()
    t_dx = t_xmax - t_xmin
    t_dy = t_ymax - t_ymin

    print("\n=== STEP 2: Raw Image Matrix Dimensions ===")
    node = sdata.images[image_key][scale_key]
    img_data = node.image if hasattr(node, "image") else node
    img_shape = img_data.shape  
    img_w, img_h = img_shape[2], img_shape[1]

    print("\n=== STEP 3: Scaling Factor Derivation ===")
    scale_factor = ((img_w / t_dx) + (img_h / t_dy)) / 2

    if roi is not None:
        print("\n=== STEP 4: Target ROI Transformation ===")
        roi_xmin, roi_xmax, roi_ymin, roi_ymax = roi
        
        pixel_xmin = max(0, min(int(np.floor((roi_xmin - t_xmin) * scale_factor)), img_w))
        pixel_xmax = max(0, min(int(np.ceil((roi_xmax - t_xmin) * scale_factor)), img_w))
        pixel_ymin = max(0, min(int(np.floor((roi_ymin - t_ymin) * scale_factor)), img_h))
        pixel_ymax = max(0, min(int(np.ceil((roi_ymax - t_ymin) * scale_factor)), img_h))
        
        cropped_img = img_data[:, pixel_ymin:pixel_ymax, pixel_xmin:pixel_xmax]
    else:
        cropped_img = img_data
        roi_xmin, roi_xmax, roi_ymin, roi_ymax = t_xmin, t_xmax, t_ymin, t_ymax

    # Safety check for empty crops
    if cropped_img.size == 0 or cropped_img.shape[1] == 0 or cropped_img.shape[2] == 0:
        raise ValueError(f"The provided ROI {roi} resulted in an empty image slice. Check your coordinates!")

    if "c" in cropped_img.coords:
        channel_names = [str(ch) for ch in cropped_img.coords["c"].values]
    else:
        channel_names = [f"Channel_{i}" for i in range(cropped_img.shape[0])]

    neighbor_kernel = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]]) / 8.0
    total_pixels = cropped_img.shape[1] * cropped_img.shape[2]

    summary_data = []

    print("\n=== STEP 5: Cumulative Autocorrelation Engine ===")
    for idx, ch_name in enumerate(channel_names):
        channel_slice = cropped_img[idx, :, :].compute() if hasattr(cropped_img, "compute") else cropped_img[idx, :, :]
        img_np = np.asarray(channel_slice, dtype=np.float32)
        
        local_mean = correlate(img_np, neighbor_kernel, mode='constant', cval=0.0)
        flat_img = img_np.ravel()
        flat_mean = local_mean.ravel()
        
        max_intensity = int(np.ceil(flat_img.max()))
        upper_bounds = list(range(step_size, max_intensity + step_size, step_size))
        
        cumulative_correlations = []
        cumulative_pixel_percentages = []
        
        for max_val in upper_bounds:
            mask = (flat_img >= 0) & (flat_img <= max_val)
            pixel_count = np.sum(mask)
            cumulative_pixel_percentages.append((pixel_count / total_pixels) * 100)
            
            if pixel_count > 30:
                pixel_vals = flat_img[mask]
                neighbor_vals = flat_mean[mask]
                if np.std(pixel_vals) == 0 or np.std(neighbor_vals) == 0:
                    cumulative_correlations.append(0.0)
                else:
                    cumulative_correlations.append(np.corrcoef(pixel_vals, neighbor_vals)[0, 1])
            else:
                cumulative_correlations.append(np.nan)
        
        # Convert correlations to numpy array for indexing
        cumulative_correlations = np.array(cumulative_correlations)
        valid_idx = np.where(~np.isnan(cumulative_correlations))[0]
        
        # ----------------------------------------------------
        # MULTI-MODE THRESHOLD FINDER WITH AUTOCORRELATION TARGETING
        # ----------------------------------------------------
        target_correlation = None
        if threshold is not None:
            if isinstance(threshold, dict):
                target_correlation = threshold.get(ch_name)
            else:
                target_correlation = threshold

        if target_correlation is not None and len(valid_idx) > 0:
            # Map the user's requested correlation value to the closest matching intensity step
            closest_valid_idx = np.argmin(np.abs(cumulative_correlations[valid_idx] - target_correlation))
            inflection_idx = valid_idx[closest_valid_idx]
            detected_threshold = upper_bounds[inflection_idx]
            plot_label = f'Target Autocorr ({target_correlation}): Cutoff {detected_threshold}'
        else:
            # Fall back to automated curve shape rules if no manual correlation target is specified
            if len(valid_idx) > 2:
                valley_idx = valid_idx[np.nanargmin(cumulative_correlations[valid_idx])]
                
                if mode == "permissive":
                    diffs = np.diff(cumulative_correlations)
                    elbow_candidates = np.where(diffs > -0.04)[0]
                    inflection_idx = valid_idx[elbow_candidates[0] + 1] if len(elbow_candidates) > 0 else valley_idx
                    
                elif mode == "stringent":
                    post_valley_idx = valid_idx[valid_idx >= valley_idx]
                    if len(post_valley_idx) > 1:
                        diffs_post = np.diff(cumulative_correlations[post_valley_idx])
                        if len(diffs_post) > 0:
                            inflection_idx = post_valley_idx[np.argmax(diffs_post) + 1]
                        else:
                            inflection_idx = post_valley_idx[-1]
                    else:
                        inflection_idx = min(valley_idx + 2, valid_idx[-1])
                
                else: # "standard" mode
                    inflection_idx = valley_idx
                    if inflection_idx == 0 and len(valid_idx) > 1:
                        diffs = np.diff(cumulative_correlations)
                        inflection_idx = np.where(diffs > -0.02)[0][0] + 1 if any(diffs > -0.02) else 1
                
                detected_threshold = upper_bounds[inflection_idx]
            else:
                detected_threshold = step_size
            
            plot_label = f'{mode.capitalize()} Autocorr Cutoff: {detected_threshold}'
            
        # Calculate % of positive pixels
        positive_pixels_count = np.sum(img_np > detected_threshold)
        pct_positive_pixels = (positive_pixels_count / total_pixels) * 100
        
        summary_data.append({
            "Channel Name": ch_name,
            "Intensity Threshold": detected_threshold,
            "% Positive Pixels (Signal)": round(pct_positive_pixels, 2)
        })
        
        # ----------------------------------------------------
        # VISUALIZATION TRIPLE PANEL
        # ----------------------------------------------------
        fig, (ax_img, ax_corr, ax_mask) = plt.subplots(1, 3, figsize=(18, 5))
        extent = [roi_xmin, roi_xmax, roi_ymin, roi_ymax]
        
        ax_img.imshow(img_np, cmap=cmap, origin="lower", extent=extent)
        ax_img.set_title(f"{ch_name} - Raw Intensity", fontweight='bold', fontsize=11)
        ax_img.axis("off")
        
        color_corr = 'crimson'
        ax_corr.plot(upper_bounds, cumulative_correlations, marker='o', color=color_corr, linewidth=2, markersize=4, zorder=3)
        ax_corr.axvline(detected_threshold, color='darkgreen', linestyle='--', linewidth=2, label=plot_label)
        ax_corr.set_ylabel("Cumulative Spatial Correlation", color=color_corr, fontweight='bold')
        ax_corr.tick_params(axis='y', labelcolor=color_corr)
        ax_corr.set_xlabel("Upper Threshold Limit (X)")
        ax_corr.set_ylim(-0.05, 1.05)
        
        ax_perc = ax_corr.twinx()
        color_perc = 'navy'
        ax_perc.fill_between(upper_bounds, 0, cumulative_pixel_percentages, color=color_perc, alpha=0.08, zorder=1)
        ax_perc.set_ylabel("Total Pixels Included (%)", color=color_perc, fontweight='bold')
        ax_perc.tick_params(axis='y', labelcolor=color_perc)
        ax_corr.legend(loc='lower right')
        ax_corr.set_title("Autocorrelation Curve", fontweight='bold', fontsize=11)
        ax_corr.grid(True, linestyle=':', alpha=0.4)
        
        ax_mask.imshow(img_np > detected_threshold, cmap="gray", origin="lower", extent=extent)
        ax_mask.set_title(f"Mask (> {detected_threshold})\nCoverage: {pct_positive_pixels:.2f}%", fontweight='bold', fontsize=11, color='darkgreen')
        ax_mask.axis("off")
        
        plt.tight_layout()
        
        if saving_path is not None:
            save_dir = Path(saving_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            roi_suffix = f"_roi_{int(roi_xmin)}_{int(roi_xmax)}"
            clean_ch_name = ch_name.replace(" ", "_").replace("/", "-")
            filename = f"analysis_{clean_ch_name}{roi_suffix}.png"
            fig.savefig(save_dir / filename, bbox_inches="tight", dpi=150)
            
        plt.show()
        plt.close(fig)

    df_summary = pd.DataFrame(summary_data)
    return df_summary