"""Compute morphology metrics for the CosMx/MERSCOPE datasets (default 'rois' segmentation)
that are missing them because their labels raster is empty (CosMx) or absent (MERSCOPE).

Rasterizes each dataset's sdata.shapes["shapes"] cell-boundary polygons into a fresh labels
layer (skimage-based, validated against CosMx's own native Area column: Spearman r=0.99),
computes the same regionprops-derived metrics as troutpy.pp.add_morphological_metrics (merged
onto obs via .loc rather than that function's index-based join, to avoid mutating obs.index),
and writes the result to a NEW zarr file (data_with_source_score_morphology.zarr) alongside
the original -- the original data_with_source_score.zarr is never modified.

Usage: python compute_missing_morphology.py <sample>
"""
import os
import sys
import numpy as np
import pandas as pd
import shapely.affinity as saffinity
import spatialdata as sd
from spatialdata.models import Labels2DModel
from spatialdata.transformations import get_transformation, set_transformation
from skimage.draw import polygon as sk_polygon
from scipy.stats import spearmanr, pearsonr

from troutpy import pp

base_dir = "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois"


def clean_id_str(series):
    """Stringify an id column/index without ever round-tripping through float64 (which would
    silently lose precision on MERSCOPE's 19-digit EntityIDs, since they exceed float64's
    2**53 exact-integer range) -- just strip a trailing '.0' left over from float-dtype
    storage of small integer ids (e.g. CosMx's shapes['label'])."""
    s = pd.Series(series).astype(str)
    return s.str.replace(r"\.0$", "", regex=True)


def get_canvas(sdata, target_unit_to_pixels=1.0):
    """Return (label_transform, height, width) for the rasterization canvas.

    Reuses the existing (empty) labels raster's shape/transform when present (CosMx), else
    derives a fresh canvas from the shapes bounding box at target_unit_to_pixels resolution
    (MERSCOPE, which has no raster layer at all).
    """
    if "labels" in sdata.labels:
        label_transform = get_transformation(sdata.labels["labels"], "global")
        label_element = sdata.labels["labels"]
        arr_shape = label_element["scale0"].image.shape if hasattr(label_element, "scale0") else label_element.shape
        height, width = (int(s) for s in arr_shape)
        return label_transform, height, width

    from spatialdata.transformations import Affine
    gdf = sdata.shapes["shapes"]
    bounds = gdf.total_bounds  # in shapes' own local frame
    shapes_transform = get_transformation(gdf, "global")
    shapes_affine = shapes_transform.to_affine_matrix(("x", "y"), ("x", "y"))
    # Use all 4 corners (not just 2 diagonal ones) so this stays correct if the transform ever
    # has rotation/shear, not just the pure scale+translation seen so far.
    corners_local = np.array([
        [bounds[0], bounds[1], 1], [bounds[2], bounds[1], 1],
        [bounds[2], bounds[3], 1], [bounds[0], bounds[3], 1],
    ]).T
    corners_global = shapes_affine @ corners_local
    xmin, xmax = corners_global[0, :].min(), corners_global[0, :].max()
    ymin, ymax = corners_global[1, :].min(), corners_global[1, :].max()
    xmin, ymin = np.floor([xmin, ymin]) - 1
    xmax, ymax = np.ceil([xmax, ymax]) + 1
    width = int(np.ceil((xmax - xmin) * target_unit_to_pixels))
    height = int(np.ceil((ymax - ymin) * target_unit_to_pixels))
    # Forward (pixel -> global) transform: pixel 0 must map to global xmin/ymin, and the scale
    # factor is *units per pixel* (1/target_unit_to_pixels), not target_unit_to_pixels itself --
    # both the sign of the translation and the scale direction were backwards in an earlier
    # version of this function, which silently shifted every rasterized polygon off-canvas.
    inv_scale = 1.0 / target_unit_to_pixels
    new_affine = Affine(
        np.array([[inv_scale, 0, xmin], [0, inv_scale, ymin], [0, 0, 1]]),
        input_axes=("x", "y"), output_axes=("x", "y"),
    )
    return new_affine, height, width


def rasterize_shapes(sdata, id_col, label_transform, height, width):
    """Rasterize sdata.shapes['shapes'] into a (height, width) uint32 label array, burning
    sequential integer codes (not the raw id_col values, which may exceed uint32 range for
    MERSCOPE's 19-digit EntityIDs). Returns (canvas, code_to_origid: dict[int, str])."""
    gdf = sdata.shapes["shapes"].copy()
    shapes_transform = get_transformation(gdf, "global")
    shapes_affine = shapes_transform.to_affine_matrix(("x", "y"), ("x", "y"))
    a, b, xoff = shapes_affine[0]
    d, e, yoff = shapes_affine[1]
    shapely_params = [a, b, d, e, xoff, yoff]

    canvas_affine = label_transform.to_affine_matrix(("x", "y"), ("x", "y"))
    cx_off, cy_off = float(canvas_affine[0, 2]), float(canvas_affine[1, 2])
    cx_scale, cy_scale = float(canvas_affine[0, 0]), float(canvas_affine[1, 1])

    gdf["geometry"] = gdf["geometry"].apply(lambda g: saffinity.affine_transform(g, shapely_params))
    gdf["geometry"] = gdf["geometry"].apply(
        lambda g: saffinity.affine_transform(g, [1 / cx_scale, 0, 0, 1 / cy_scale, -cx_off / cx_scale, -cy_off / cy_scale])
    )

    orig_ids = clean_id_str(gdf[id_col]).tolist()
    codes = np.arange(1, len(orig_ids) + 1, dtype=np.uint32)
    code_to_origid = dict(zip(codes.tolist(), orig_ids))

    canvas = np.zeros((height, width), dtype=np.uint32)
    n_drawn, n_skipped = 0, 0
    for geom, code in zip(gdf["geometry"], codes):
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for p in polys:
            if p.is_empty or not p.exterior.coords:
                n_skipped += 1
                continue
            xs, ys = p.exterior.coords.xy
            rr, cc = sk_polygon(np.array(ys), np.array(xs), shape=canvas.shape)
            if len(rr) == 0:
                n_skipped += 1
                continue
            canvas[rr, cc] = code
            n_drawn += 1
    print(f"  drew {n_drawn} polygon parts, skipped {n_skipped}")
    return canvas, code_to_origid


def process(seg, sample, id_col_candidates=("label", "EntityID")):
    zarr_path = os.path.join(base_dir, sample, "data_with_source_score.zarr")
    out_zarr = os.path.join(base_dir, sample, "data_with_source_score_morphology.zarr")
    print(f"\n=== {seg}/{sample} ===")
    print(f"Loading {zarr_path} ...")
    sdata = sd.read_zarr(zarr_path)

    gdf = sdata.shapes["shapes"]
    id_col = next((c for c in id_col_candidates if c in gdf.columns), None)
    if id_col is None:
        raise RuntimeError(f"No known id column found in shapes for {sample}: {gdf.columns.tolist()}")
    print(f"  using shapes id column: {id_col}")

    label_transform, height, width = get_canvas(sdata)
    print(f"  canvas: height={height}, width={width}")

    canvas, code_to_origid = rasterize_shapes(sdata, id_col, label_transform, height, width)
    n_unique = len(np.unique(canvas)) - (1 if 0 in canvas else 0)
    print(f"  canvas n_unique_labels={n_unique} / {len(code_to_origid)} polygons")

    new_label_key = "cell_labels_rasterized"
    labels_element = Labels2DModel.parse(canvas)
    set_transformation(labels_element, label_transform, "global")
    sdata.labels[new_label_key] = labels_element

    table_obs = sdata.tables["table"].obs

    # spatialdata enforces case-insensitive uniqueness on obs column names at write time.
    # CosMx's native obs already has e.g. "Area"/"AspectRatio" columns, which collide with our
    # new "area" column even though the case differs. Rename any pre-existing case-insensitive
    # collisions (keeping the data, just renamed) rather than renaming our own new columns, so
    # downstream code can rely on the standard lowercase metric names across every dataset.
    morph_metric_names = {"area", "perimeter", "solidity", "major_axis_length",
                           "equivalent_diameter", "circularity", "protrusion_length",
                           "morphological_complexity"}
    rename_map = {c: f"{c}_native" for c in table_obs.columns
                  if c.lower() in morph_metric_names and c not in morph_metric_names}
    if rename_map:
        print(f"  renaming pre-existing columns to avoid case-insensitive collision: {rename_map}")
        table_obs.rename(columns=rename_map, inplace=True)

    # Match each rasterized code back to its obs row via whichever key actually works: try
    # obs.index directly first (works for MERSCOPE, where index == EntityID), else obs["cell_id"]
    # (works for CosMx, where index is a composite "row_fov" string).
    origid_to_code = {v: k for k, v in code_to_origid.items()}
    idx_match = pd.Series(clean_id_str(table_obs.index).map(origid_to_code).values, index=table_obs.index)
    if idx_match.notna().mean() > 0.5:
        print("  matching via obs.index")
        code_for_row = idx_match
    elif "cell_id" in table_obs.columns:
        print("  matching via obs['cell_id']")
        code_for_row = pd.Series(clean_id_str(table_obs["cell_id"]).map(origid_to_code).values, index=table_obs.index)
    else:
        raise RuntimeError(f"Could not find a usable id match for {sample} (index match rate {idx_match.notna().mean():.2%}, no cell_id column)")

    match_rate = code_for_row.notna().mean()
    print(f"  obs<->raster match rate: {match_rate:.2%} ({code_for_row.notna().sum()}/{len(code_for_row)})")
    if match_rate < 0.5:
        raise RuntimeError(f"Low match rate ({match_rate:.2%}) for {sample} -- aborting, needs investigation")

    # Compute regionprops directly (same logic as troutpy.pp.add_morphological_metrics) and
    # merge onto obs via .loc keyed by each row's ORIGINAL index label -- never mutating
    # obs.index in place, which was found to desync AnnData's internal aligned-mapping
    # consistency check (obsm/etc. vs. obs.index) and break the zarr write validation.
    from skimage.measure import regionprops_table
    props = regionprops_table(canvas, properties=["label", "area", "perimeter", "solidity",
                                                   "major_axis_length", "equivalent_diameter"])
    df_morph = pd.DataFrame(props).set_index("label")
    df_morph["circularity"] = (4 * np.pi * df_morph["area"]) / (df_morph["perimeter"] ** 2)
    df_morph["protrusion_length"] = (df_morph["major_axis_length"] - df_morph["equivalent_diameter"]) / 2
    df_morph["morphological_complexity"] = 1 - df_morph["solidity"]

    morph_cols = ["area", "perimeter", "solidity", "major_axis_length", "equivalent_diameter",
                  "circularity", "protrusion_length", "morphological_complexity"]
    for c in morph_cols:
        table_obs[c] = np.nan

    valid = code_for_row.dropna().astype(int)
    valid = valid[valid.isin(df_morph.index)]
    table_obs.loc[valid.index, morph_cols] = df_morph.loc[valid.values, morph_cols].values

    obs = sdata.tables["table"].obs
    n_with_morph = obs["area"].notna().sum()
    print(f"  morphology present: True, n_cells with area: {n_with_morph}/{len(obs)}")

    if "Area_native" in obs.columns:
        sanity = obs[["area", "Area_native"]].dropna()
        if len(sanity) > 5:
            r_s, _ = spearmanr(sanity["area"], sanity["Area_native"])
            r_p, _ = pearsonr(sanity["area"], sanity["Area_native"])
            print(f"  sanity check vs native Area: n={len(sanity)}, spearman r={r_s:.4f}, pearson r={r_p:.4f}")

    if os.path.isdir(out_zarr):
        import shutil
        shutil.rmtree(out_zarr)
    print(f"  writing {out_zarr} ...")
    sdata.write(out_zarr, overwrite=False)
    print(f"  done: {sample}")
    return n_with_morph, len(obs)


if __name__ == "__main__":
    targets = [
        ("rois", "cosmx_hgsoc"),
        ("rois", "cosmx_melanoma"),
        ("rois", "cosmx_mousebrain"),
        ("rois", "cosmx_stomach"),
        ("rois", "merscope_hsbrain"),
        ("rois", "merscope_hsuterine"),
    ]
    results = []
    for seg, sample in targets:
        try:
            n_ok, n_total = process(seg, sample)
            results.append((seg, sample, "OK", n_ok, n_total))
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            results.append((seg, sample, f"FAILED: {e}", None, None))

    print("\n=== SUMMARY ===")
    for seg, sample, status, n_ok, n_total in results:
        print(f"{seg}/{sample}: {status} ({n_ok}/{n_total})" if n_ok is not None else f"{seg}/{sample}: {status}")
