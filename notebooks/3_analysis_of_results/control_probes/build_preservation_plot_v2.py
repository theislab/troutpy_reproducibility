"""
Reviewer-response analysis v2: does sample preservation (FFPE vs fresh-frozen) or
panel/chemistry design associate with negative-control signal - now correctly
accounting for two facts pointed out on review of v1:

1. "Control probe" is not one thing. Two conceptually different QC features get
   lumped under that name across platforms:
     - PROBE-level negative controls (Xenium NegControlProbe_, CosMx NegPrb,
       Atera NegControlProbe_): synthetic oligos with no complementary target in
       the transcriptome. A signal here means non-specific probe
       hybridization/amplification - a *chemistry* background.
     - CODEWORD/barcode-level negative controls (Xenium NegControlCodeword_/
       BLANK_, CosMx SystemControl, Atera NegControlCodeword_): valid barcode
       identities in the decoding space that are deliberately left unassigned to
       any probe. A signal here means a barcode got read/decoded/optically
       assigned when nothing should have been there - a *decoding/optical* noise
       source, unrelated to probe hybridization specificity.
   These are analyzed and plotted SEPARATELY below rather than summed together.
   (MERSCOPE is still excluded entirely - see v1 docstring.)

2. A panel with more designed negative-control identities will rack up a larger
   RAW SUM of control transcripts even at an identical true background rate per
   probe - e.g. Xenium codeword-control feature counts alone range 41-555 across
   panels in this dataset (a >13x difference), and CosMx's two panel designs don't
   even share a control type (1k = probe-only, WTx = codeword-only). So the metric
   here is the MEAN transcripts PER DESIGNED CONTROL FEATURE (not the sum), further
   normalized per mm^2 of profiled tissue area. This asks "how much signal does a
   single negative-control identity accumulate per unit tissue", which is
   comparable across panels regardless of how many control features they carry.

Feature counts used as the per-feature denominator are the number of *unique
observed* control names (count > 0) for that subtype in that sample - a slight
underestimate of the true designed count for very sparse panels/small ROIs, since
a designed control with zero observed transcripts isn't counted; noted as a caveat.
"""
import os
import re
import warnings

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

warnings.filterwarnings("ignore")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

scratch_dir = "/ictstr01/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/scratch_claude"
rois_dir = "/ictstr01/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois"
fig_dir = "/ictstr01/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/figures/preservation_panel_control_probe"
os.makedirs(fig_dir, exist_ok=True)

# CosMx WTx labels its negative-control PROBES "Negative##" (not "NegPrb##" like the
# CosMx 1k panel) - confirmed against NanoString's own README for this exact WTx
# pancreas dataset (smi-public.objects.liquidweb.services/cosmx-wtx/Pancreas-CosMx-ReadMe.html),
# which describes 50 "Negative#" probes distinct from the "SystemControl#" codewords.
# An earlier version of this script only matched NegPrb/NegControlProbe and so missed
# these entirely, incorrectly showing CosMx WTx as having zero probe-type controls.
PROBE_CONTROL_RE = re.compile(r"^(NegControlProbe|NegPrb|Negative\d)", re.IGNORECASE)
CODEWORD_CONTROL_RE = re.compile(r"^(NegControlCodeword|BLANK|Blank|SystemControl)", re.IGNORECASE)

panel_design = {
    "atera_hsbreast":     "WTA (~18k)",
    "cosmx_hgsoc":        "CosMx 1k",
    "cosmx_melanoma":     "CosMx 1k",
    "cosmx_mousebrain":   "CosMx 1k",
    "cosmx_stomach":      "CosMx 1k",
    "cosmx_pancreas":     "CosMx WTx",
    "xenium_hsfemur":     "Xenium standard",
    "xenium_hsheart":     "Xenium standard",
    "xenium_hsprostate":  "Xenium 5K (Prime)",
    "xenium_hsputamen":   "Xenium standard",
    "xenium_hsskin_5k":   "Xenium 5K (Prime)",
    "xenium_msbrain_300": "Xenium standard",
    "xenium_msbrain_5k":  "Xenium 5K (Prime)",
    "xenium_mscolon_5k":  "Xenium standard",
    "xenium_opsc_5k":     "Xenium 5K (Prime)",
    "xenium_opsc_v1":     "Xenium standard",
    "xenium_rcc":         "Xenium standard",
}

# See build_preservation_plot.py for the per-sample sourcing of these labels.
preservation = {
    "atera_hsbreast":     "FFPE",
    "cosmx_hgsoc":        "Fresh Frozen",
    "cosmx_melanoma":     "FFPE",
    "cosmx_mousebrain":   "Fresh Frozen",
    "cosmx_stomach":      "FFPE",
    "cosmx_pancreas":     "FFPE",
    "xenium_hsfemur":     "FFPE",
    "xenium_hsheart":     "FFPE",
    "xenium_hsprostate":  "FFPE",
    "xenium_hsputamen":   "Fresh Frozen",
    "xenium_hsskin_5k":   "FFPE",
    "xenium_msbrain_300": "Fresh Frozen",
    "xenium_msbrain_5k":  "Fresh Frozen",
    "xenium_mscolon_5k":  "Fresh Frozen",
    "xenium_opsc_5k":     "Fresh Frozen",
    "xenium_opsc_v1":     "Fresh Frozen",
    "xenium_rcc":         "FFPE",
}

records = []
for sample in sorted(panel_design):
    ppath = os.path.join(rois_dir, sample, "data.zarr", "points", "transcripts", "points.parquet")
    ok = False
    for attempt in range(3):
        try:
            t = pq.read_table(ppath, columns=["gene"])
            ok = True
            break
        except Exception as e:
            print(f"[{sample}] attempt {attempt} failed: {e}")
    if not ok:
        continue
    genes = t.column("gene").to_pandas().astype(str)

    mpath = os.path.join(rois_dir, sample, "metadata.csv")
    m = pd.read_csv(mpath)
    area_mm2 = float(m["area_profiled_um2"].iloc[0]) / 1e6

    for subtype_name, regex in [("probe_control", PROBE_CONTROL_RE), ("codeword_control", CODEWORD_CONTROL_RE)]:
        mask = genes.str.match(regex)
        sub = genes[mask]
        n_transcripts = int(mask.sum())
        n_features = sub.nunique()
        mean_per_feature = n_transcripts / n_features if n_features > 0 else float("nan")
        records.append({
            "sample": sample,
            "technology": m["technology"].iloc[0],
            "panel_design": panel_design[sample],
            "preservation": preservation[sample],
            "control_subtype": subtype_name,
            "n_control_transcripts": n_transcripts,
            "n_control_features": n_features,
            "area_mm2": area_mm2,
            "mean_transcripts_per_feature": mean_per_feature,
            "mean_transcripts_per_feature_per_mm2": mean_per_feature / area_mm2 if n_features > 0 else float("nan"),
        })
    print(f"[{sample}] done")

df = pd.DataFrame.from_records(records)
out_table = os.path.join(scratch_dir, "combined_control_probe_by_subtype.csv")
df.to_csv(out_table, index=False)
print("\nSaved table to", out_table)
print(df.to_string())

# --- Plot: 2 subtypes (rows) x 2 groupings (columns) --------------------------------
tech_palette = {"Xenium": "#2a78d6", "CosMx": "#e34948", "Atera": "#9b59b6"}
subtype_labels = {
    "probe_control": "Negative control PROBES\n(non-specific hybridization/amplification)",
    "codeword_control": "Negative control CODEWORDS / blanks\n(optical decoding & spatial noise)",
}
preservation_order = ["Fresh Frozen", "FFPE"]
design_order_full = ["CosMx 1k", "CosMx WTx", "Xenium standard", "Xenium 5K (Prime)", "WTA (~18k)"]

sns.set_style("ticks")
fig, axes = plt.subplots(2, 2, figsize=(11, 9), gridspec_kw={"width_ratios": [1, 1.5]})

for row, subtype in enumerate(["probe_control", "codeword_control"]):
    sub_df = df[(df["control_subtype"] == subtype) & df["mean_transcripts_per_feature_per_mm2"].notna()].copy()

    # column 0: preservation
    ax = axes[row, 0]
    order = [g for g in preservation_order if g in sub_df["preservation"].unique()]
    sns.violinplot(
        data=sub_df, x="preservation", y="mean_transcripts_per_feature_per_mm2", order=order,
        inner=None, color="0.92", linewidth=1, cut=0, ax=ax, log_scale=True,
    )
    sns.stripplot(
        data=sub_df, x="preservation", y="mean_transcripts_per_feature_per_mm2", order=order,
        hue="technology", hue_order=["Xenium", "CosMx", "Atera"], palette=tech_palette,
        size=8, jitter=0.15, linewidth=0.6, edgecolor="white", ax=ax, legend=(row == 0),
    )
    n_per_group = sub_df.groupby("preservation").size()
    ax.set_ylim(top=ax.get_ylim()[1] * 3)
    for i, grp in enumerate(order):
        ax.text(
            (i + 0.5) / max(len(order), 1), 0.97, f"n={n_per_group.get(grp, 0)}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.3",
        )
    ax.set_xlabel("")
    ax.set_ylabel(f"{subtype_labels[subtype]}\nmean transcripts / feature / mm$^2$", fontsize=9)
    if row == 0:
        ax.set_title("Preservation method", fontsize=11)
        ax.legend(title="Technology", loc="upper left", bbox_to_anchor=(0, -0.28), frameon=False, fontsize=8, title_fontsize=8, ncol=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # column 1: panel design
    ax = axes[row, 1]
    order = [d for d in design_order_full if d in sub_df["panel_design"].unique()]
    sns.violinplot(
        data=sub_df, x="panel_design", y="mean_transcripts_per_feature_per_mm2", order=order,
        inner=None, color="0.92", linewidth=1, cut=0, ax=ax, log_scale=True,
    )
    sns.stripplot(
        data=sub_df, x="panel_design", y="mean_transcripts_per_feature_per_mm2", order=order,
        hue="technology", hue_order=["Xenium", "CosMx", "Atera"], palette=tech_palette,
        size=8, jitter=0.15, linewidth=0.6, edgecolor="white", ax=ax, legend=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    if row == 0:
        ax.set_title("Panel / chemistry design", fontsize=11)
    # Each row keeps its own x labels - the two subtypes don't share a category set
    # (e.g. CosMx 1k has no codeword-type controls at all, so it's absent from that row).
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "Negative-control signal per designed control feature, per mm$^2$ of tissue\n"
    "(probe-level vs. codeword-level controls, shown separately)",
    fontsize=12, y=1.01,
)
fig.text(
    0.5, -0.02,
    "CosMx 1k is absent from the codeword-control row (bottom) because that panel's exported data has no "
    "codeword/blank-type control category at all - this is unknown, not measured as zero.",
    ha="center", va="top", fontsize=8, color="0.3", style="italic",
)
plt.tight_layout()
out_pdf = os.path.join(fig_dir, "control_probe_by_subtype_v2.pdf")
out_png = os.path.join(fig_dir, "control_probe_by_subtype_v2.png")
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, bbox_inches="tight", dpi=200)
print("\nSaved:", out_pdf, "and", out_png)
