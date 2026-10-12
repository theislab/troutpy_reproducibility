"""Canonical coarse cell-type marker gene sets, used to score the non-immune
compartment after CellTypist has called immune clusters (see
annotate_cell_type_general.py).

Kept deliberately small and broadly-conserved so a reasonable fraction survives
intersection with limited CosMx/MERSCOPE panels (300-1000 genes) as well as the
larger Xenium panels. Human gene symbols are canonical; mouse symbols are derived
by title-casing, which holds for the large majority of 1:1 orthologs and is a
standard practical approximation (not a curated ortholog mapping).
"""

HUMAN_MARKERS = {
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "KRT5", "KRT14"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "FLT1", "KDR"],
    "Fibroblast/Stromal": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA", "PDGFRB"],
    "Muscle": ["ACTA2", "MYH11", "TAGLN", "DES", "MYL9", "CNN1"],
    "Neural": ["RBFOX3", "SNAP25", "SYT1", "MAP2", "GFAP", "MBP", "PLP1", "SOX10", "S100B"],
}


def markers_for_species(species: str) -> dict:
    if species == "human":
        return HUMAN_MARKERS
    if species == "mouse":
        return {cat: [g.capitalize() for g in genes] for cat, genes in HUMAN_MARKERS.items()}
    raise ValueError(f"Unknown species: {species!r}, expected 'human' or 'mouse'")
