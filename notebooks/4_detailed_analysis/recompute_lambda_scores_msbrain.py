#!/usr/bin/env python3
"""Lambda sensitivity sweep for the annotated xenium_msbrain_5k object.

Sweeps lmbda over a few values with the production `tl.urna_source_score`
(not the throwaway prototype from the SUPP_3 notebook) on the FULL dataset (~8.4M
extracellular transcripts, no subsampling) to produce a robustness sensitivity
analysis for the reviewer response, saved as editable vector PDFs.

This is deliberately sweep-only: it does NOT write a new production zarr. Once a
lambda is chosen from these results, rerun urna_source_score once
at that value and write data_annotated_lambda<X>.zarr as a separate step.
"""
import os
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spatialdata as sd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as shannon_entropy

from troutpy import tl

data_dir = Path("/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_msbrain_5k")
saving_path = Path("/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/ms_figures/xenium_msbrain_5k")
saving_path.mkdir(parents=True, exist_ok=True)

print(f"Loading {data_dir / 'data_annotated.zarr'} ...", flush=True)
sdata = sd.read_zarr(data_dir / "data_annotated.zarr")

obs_cols = list(sdata["table"].obs.columns)
transcript_cols = list(sdata.points["transcripts"].columns)
print("table.obs columns:", obs_cols, flush=True)
print("transcripts columns:", transcript_cols, flush=True)

for required, where in [
    ("cell_type", "table.obs"),
    ("density_similarity_score", "transcripts"),
    ("enrichment_over_random", "transcripts"),
]:
    cols = obs_cols if where == "table.obs" else transcript_cols
    if required not in cols:
        raise KeyError(f"Required column '{required}' missing from {where}. Found: {cols}")

transcripts = sdata.points["transcripts"]
min_enrichment = (
    transcripts[transcripts["density_similarity_score"] > 0.5]["enrichment_over_random"].min().compute()
)
print(f"ambient_floor (min_enrichment) = {min_enrichment}", flush=True)

COMMON_KW = dict(
    chunk_size=100000,
    cell_type_col="cell_type",
    max_dist=200,
    max_k=10,
    ambient_floor=min_enrichment,
    signal_threshold=10,
    residual=0.1,
)

# --- Lambda sensitivity sweep with the production scorer, full dataset ---
print("=== Running lambda sensitivity sweep ===", flush=True)
lambdas = [0.1, 0.25, 0.5, 1.0]
snapshots = {}
for lam in lambdas:
    print(f"  lambda = {lam}", flush=True)
    tl.urna_source_score(sdata, lmbda=lam, copy=False, **COMMON_KW)
    ss = sdata.tables["source_score"]
    snapshots[lam] = pd.DataFrame(ss.X, index=ss.obs_names, columns=ss.var_names)

# Pairwise concordance / JSD (symmetric, no ground truth needed)
comparison_metrics = []
for l1, l2 in combinations(lambdas, 2):
    p1, p2 = snapshots[l1], snapshots[l2]
    calls1, calls2 = p1.idxmax(axis=1), p2.idxmax(axis=1)
    js = jensenshannon(p1.values + 1e-9, p2.values + 1e-9, axis=1)
    concordance = (calls1.values == calls2.values).mean() * 100
    comparison_metrics.append(
        {"Lambda_A": l1, "Lambda_B": l2, "Mean_JSD": np.nanmean(js), "Concordance_%": concordance}
    )
comparison_df = pd.DataFrame(comparison_metrics)

# Entropy per lambda, and concordance against the most stringent lambda tested
# (used as a stability reference, mirroring the approach in the SUPP_3 prototype).
ref_lambda = max(lambdas)
ref_calls = snapshots[ref_lambda].idxmax(axis=1)
entropy_rows = []
for lam in lambdas:
    probs = snapshots[lam]
    ent = shannon_entropy(probs.values + 1e-9, axis=1)
    calls = probs.idxmax(axis=1)
    ref_concordance = (calls.values == ref_calls.values).mean() * 100
    entropy_rows.append(
        {
            "Lambda": lam,
            "Mean_Entropy": np.nanmean(ent),
            "Median_Entropy": np.nanmedian(ent),
            f"Concordance_vs_lambda{ref_lambda:g}": ref_concordance,
        }
    )
entropy_df = pd.DataFrame(entropy_rows)

comparison_df.to_csv(saving_path / "supp3_lambda_sweep_comparison_metrics.csv", index=False)
entropy_df.to_csv(saving_path / "supp3_lambda_sweep_entropy_metrics.csv", index=False)
print(comparison_df, flush=True)
print(entropy_df, flush=True)

# --- Figures (vector PDF, editable in Illustrator: fonttype 42) ---
sns.set_theme(style="white", font_scale=1.2)


def plot_full_all_vs_all_matrices(comparison_df, entropy_df):
    flipped = comparison_df.rename(columns={"Lambda_A": "Lambda_B", "Lambda_B": "Lambda_A"})
    all_lams = sorted(set(comparison_df["Lambda_A"]) | set(comparison_df["Lambda_B"]))
    diagonal = pd.DataFrame(
        {"Lambda_A": all_lams, "Lambda_B": all_lams, "Mean_JSD": 0.0, "Concordance_%": 100.0}
    )
    full_df = pd.concat([comparison_df, flipped, diagonal]).drop_duplicates(subset=["Lambda_A", "Lambda_B"])
    pivot_conc = full_df.pivot(index="Lambda_A", columns="Lambda_B", values="Concordance_%")
    pivot_jsd = full_df.pivot(index="Lambda_A", columns="Lambda_B", values="Mean_JSD")

    fig, axes = plt.subplots(1, 3, figsize=(30, 8))
    sns.heatmap(
        pivot_conc, ax=axes[0], cmap="viridis", annot=True, fmt=".1f",
        cbar_kws={"label": "Concordance (%)"}, linewidths=1, linecolor="white",
    )
    axes[0].set_title("Full Assignment Concordance\n(urna_source_score)", fontsize=14)

    sns.heatmap(
        pivot_jsd, ax=axes[1], cmap="viridis_r", annot=True, fmt=".3f",
        cbar_kws={"label": "Mean JSD"}, linewidths=1, linecolor="white",
    )
    axes[1].set_title("Full Jensen-Shannon Distance", fontsize=14)

    ent_pivot = entropy_df.set_index("Lambda")[["Mean_Entropy"]]
    sns.heatmap(
        ent_pivot, ax=axes[2], cmap="magma", annot=True, fmt=".3f",
        cbar_kws={"label": "Bits (Shannon Entropy)"}, linewidths=1, linecolor="white",
    )
    axes[2].set_title("Mean Assignment Entropy\n(System Uncertainty)", fontsize=14)
    plt.tight_layout()
    return fig


fig1 = plot_full_all_vs_all_matrices(comparison_df, entropy_df)
fig1.savefig(saving_path / "supp3_lambda_sensitivity_heatmaps.pdf", bbox_inches="tight")
plt.close(fig1)


def plot_sweet_spot(entropy_df, ref_lambda):
    ref_col = f"Concordance_vs_lambda{ref_lambda:g}"
    x = entropy_df["Lambda"].values
    y_conc = entropy_df[ref_col].values
    y_ent = entropy_df["Mean_Entropy"].values
    min_ent_idx = entropy_df["Mean_Entropy"].idxmin()
    sweet_x, sweet_y = x[min_ent_idx], y_conc[min_ent_idx]

    fig, ax1 = plt.subplots(figsize=(12, 7))
    c1 = "tab:orange"
    ax1.set_xlabel("Lambda (λ) [Stringency]", fontsize=12)
    ax1.set_ylabel(f"Concordance vs. λ={ref_lambda:g} (%)", color=c1, fontsize=12, fontweight="bold")
    ax1.plot(x, y_conc, marker="o", color=c1, linewidth=4, zorder=3)
    ax1.tick_params(axis="y", labelcolor=c1)

    ax2 = ax1.twinx()
    c2 = "tab:blue"
    ax2.set_ylabel("Mean Assignment Entropy", color=c2, fontsize=12, fontweight="bold")
    ax2.plot(x, y_ent, marker="s", color=c2, linestyle="--", alpha=0.6)
    ax2.tick_params(axis="y", labelcolor=c2)

    plt.axvline(x=sweet_x, color="black", linestyle=":", alpha=0.5)
    ax1.scatter(sweet_x, sweet_y, color="crimson", s=250, edgecolors="black", zorder=5)
    ax1.annotate(
        f"Entropy minimum: λ={sweet_x:g}\nConcordance: {sweet_y:.1f}%",
        xy=(sweet_x, sweet_y),
        xytext=(sweet_x + (x[-1] * 0.05), sweet_y - 15),
        arrowprops=dict(facecolor="black", shrink=0.05, width=2),
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="crimson", alpha=0.9),
    )
    plt.title("Lambda sensitivity: urna_source_score (xenium_msbrain_5k)", fontsize=15, pad=20)
    plt.tight_layout()
    return fig


fig2 = plot_sweet_spot(entropy_df, ref_lambda)
fig2.savefig(saving_path / "supp3_lambda_sweet_spot.pdf", bbox_inches="tight")
plt.close(fig2)

print("All done.", flush=True)
