"""Run subtract_local_urna_background on the real xenium_msbrain_5k dataset,
then identify and plot the genes whose intracellular expression changed most.

Not part of the troutpy package -- a one-off analysis script.
"""

import sys


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spatialdata as sd

import troutpy.tl as tl

data_dir = Path("/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_msbrain_5k")
out_dir = Path("/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/ms_figures/xenium_msbrain_5k/local_urna_background_decontamination")
out_dir.mkdir(parents=True, exist_ok=True)

print("Loading zarr...", flush=True)
sdata = sd.read_zarr(data_dir / "data_annotated.zarr")

print("Running subtract_local_urna_background...", flush=True)
tl.subtract_local_urna_background(sdata, radius=100.0, inner_radius=10.0)

adata = sdata["table"]
raw = adata.layers["raw"]
corr = adata.layers["decontaminated_density"]
raw = raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw)
corr = corr.toarray() if hasattr(corr, "toarray") else np.asarray(corr)

genes = adata.var_names.to_numpy()
raw_total = raw.sum(axis=0)
corr_total = corr.sum(axis=0)
removed_total = raw_total - corr_total

df = pd.DataFrame(
    {
        "gene": genes,
        "raw_total": raw_total,
        "corrected_total": corr_total,
        "removed_total": removed_total,
        "fraction_removed": np.divide(removed_total, raw_total, out=np.zeros_like(raw_total), where=raw_total > 0),
    }
)

df.to_csv(out_dir / "local_background_gene_changes.csv", index=False)
print(f"Saved full per-gene table to {out_dir / 'local_background_gene_changes.csv'}")

# Focus on genes with enough total signal to make "fraction removed" meaningful
MIN_RAW_TOTAL = 50
df_filtered = df[df["raw_total"] >= MIN_RAW_TOTAL].copy()
top_changed = df_filtered.sort_values("fraction_removed", ascending=False).head(20)

print("\nTop 20 most-changed genes (by fraction of intracellular signal removed):")
print(top_changed[["gene", "raw_total", "corrected_total", "fraction_removed"]].to_string(index=False))

# --- Plot 1: bar chart of top changed genes ---
fig, ax = plt.subplots(figsize=(8, 7))
order = top_changed.iloc[::-1]
ax.barh(order["gene"], order["fraction_removed"] * 100, color="#bf1324")
ax.set_xlabel("% of intracellular signal removed as local uRNA background")
ax.set_title(f"Top 20 most-changed genes (min raw total = {MIN_RAW_TOTAL})")
plt.tight_layout()
fig.savefig(out_dir / "top_changed_genes_barh.png", dpi=150)
print(f"Saved bar chart to {out_dir / 'top_changed_genes_barh.png'}")

# --- Plot 2: scatter raw vs corrected total, all genes, top-changed labeled ---
fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.scatter(df_filtered["raw_total"], df_filtered["corrected_total"], s=8, alpha=0.4, color="gray")
lims = [1, max(df_filtered["raw_total"].max(), df_filtered["corrected_total"].max()) * 1.1]
ax2.plot(lims, lims, "--", color="black", linewidth=0.8, label="y = x (no change)")
top_labeled = top_changed.head(15)
ax2.scatter(top_labeled["raw_total"], top_labeled["corrected_total"], s=20, color="#bf1324", zorder=3)
for _, row in top_labeled.iterrows():
    ax2.annotate(row["gene"], (row["raw_total"], row["corrected_total"]), fontsize=8, color="#bf1324")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlim(lims)
ax2.set_ylim(lims)
ax2.set_xlabel("Raw total intracellular count (all cells)")
ax2.set_ylabel("Corrected total intracellular count (all cells)")
ax2.set_title("Effect of local uRNA background subtraction, per gene")
ax2.legend()
plt.tight_layout()
fig2.savefig(out_dir / "raw_vs_corrected_scatter.png", dpi=150)
print(f"Saved scatter plot to {out_dir / 'raw_vs_corrected_scatter.png'}")

print("\nDone.")
