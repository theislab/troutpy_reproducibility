"""Manual mapping from each already-annotated dataset's real (fine) `cell_type` labels
down to the same coarse cell_type_general vocabulary used by the 14 datasets annotated
via annotate_cell_type_general.py (Epithelial, Endothelial, Fibroblast/Stromal, Muscle,
Neural, Immune).

Tumor cells are deliberately NOT split out into a separate Malignant/Tumor bucket, even
for atera_hsbreast (the only dataset whose fine annotation could support that split):
reliably distinguishing malignant from normal epithelial cells is only possible where a
fine annotation already did it, so keeping the distinction for one dataset out of many
cancer-containing datasets (cosmx_hgsoc, cosmx_melanoma, cosmx_stomach, xenium_opsc_5k/v1,
xenium_hsprostate all lack it, having gone through the generic marker-score method) would
make cell_type_general inconsistent across datasets rather than more informative. All
tumor/malignant cells are folded into Epithelial everywhere.

Notable non-obvious calls, made explicitly rather than left implicit:
- Tumor cells (all breast subtypes) -> Epithelial (see above; not split out).
- Osteoclast -> Immune (myeloid/macrophage-lineage despite the bone-cell name).
- Myoepithelial/Basal cells -> Epithelial (standard convention despite contractile function).
- Adipocyte, Osteoblast, Pre-osteoblast, SSPC_MPC -> Fibroblast/Stromal (mesenchymal
  lineage; no dedicated bone/fat bucket since only xenium_hsfemur has these).
- Immune-rel. fibroblast -> Fibroblast/Stromal, not Immune (still a fibroblast subtype).
- Microglia -> Immune (CNS-resident macrophages).
- SMC/Pericytes, Mural Cells (Pericytes/SMCs) -> Muscle.
"""

FINE_TO_COARSE = {
    "atera_hsbreast": {
        "Angiogenic Endothelial Cells": "Endothelial",
        "Endothelial Cells (Blood)": "Endothelial",
        "Lymphatic Endothelial Cells (LECs)": "Endothelial",
        "Cancer-Associated Fibroblasts (CAFs)": "Fibroblast/Stromal",
        "Mural Cells (Pericytes/SMCs)": "Muscle",
        "Myeloid Cells / Macrophages": "Immune",
        "Plasma B Cells": "Immune",
        "T Cells": "Immune",
        "Myoepithelial / Basal cells": "Epithelial",
        "Tumor cells (Basal-like/Progenitor)": "Epithelial",
        "Tumor cells (Cycling/Proliferating)": "Epithelial",
        "Tumor cells (Luminal A)": "Epithelial",
        "Tumor cells (Luminal Differentiated)": "Epithelial",
        "Tumor cells (Luminal HR+)": "Epithelial",
        "Tumor cells (Luminal Progenitor-like)": "Epithelial",
        "Tumor cells (Luminal TFF1/3 High)": "Epithelial",
        "Tumor cells (MDM2+ Amplified)": "Epithelial",
        "Tumor cells (Secretory/XBP1+)": "Epithelial",
    },
    "xenium_hsfemur": {
        "Endothelial": "Endothelial",
        "Fibroblast 1": "Fibroblast/Stromal",
        "Fibroblast 2": "Fibroblast/Stromal",
        "Immune-rel. fibroblast": "Fibroblast/Stromal",
        "Osteoblast": "Fibroblast/Stromal",
        "Pre-osteoblast": "Fibroblast/Stromal",
        "SSPC_MPC": "Fibroblast/Stromal",
        "Adipocyte": "Fibroblast/Stromal",
        "Osteoclast": "Immune",
        "B cell": "Immune",
        "Myeloid": "Immune",
        "NK_Cells_T_cells": "Immune",
    },
    "xenium_msbrain_5k": {
        "Astrocytes": "Neural",
        "Astrocytes Thal.": "Neural",
        "Ependymal cells": "Neural",
        "Choroid plexus": "Neural",
        "OPC": "Neural",
        "Oligodendrocytes": "Neural",
        "Cajal-Retzius": "Neural",
        "CA1-2": "Neural",
        "CA3": "Neural",
        "DG": "Neural",
        "EXC neurons Thal.": "Neural",
        "EXC. neurons CTX": "Neural",
        "INH neurons": "Neural",
        "INH neurons CP": "Neural",
        "SGZ Neuroblasts": "Neural",
        "Endothelial": "Endothelial",
        "Microglia": "Immune",
        "SMC/Pericytes": "Muscle",
    },
}
