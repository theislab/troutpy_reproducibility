#!/usr/bin/env python3
"""Fix the transcripts `cell_id` column of a raw ROI data.zarr so segger's
generic spatialdata loader can read it, and write the result to a new
staging copy (the original data.zarr in rois/ is left untouched).

Root causes fixed here (see segger_out/models/<sample>/segger.log for the
original failures):

- CosMx samples (hgsoc, melanoma, mousebrain, pancreas, stomach): the
  transcripts table has no column literally named `cell_id` (segger's
  StandardTranscriptFields default), only `cell_ID` (int, 0 = unassigned).
  Segger's spatialdata-mode reader silently treats the missing `cell_id`
  column as all-null, so every transcript gets filtered out ("No transcripts
  with assigned cell IDs remain").

- MERSCOPE (hsbrain, hsuterine) and xenium_msbrain_300: `cell_id` exists but
  is a numeric dtype. `anndata_from_transcripts` builds one index by casting
  cell ids to str (`segment_idx...astype(str)`) and another by leaving them
  in their native dtype (`centroids...set_index(cell_id_column)`), so the
  `.loc[]` alignment between the two never matches ("None of [...] are in
  the index").

Fix: derive a single, string-typed `cell_id` column that is a true null (not
a sentinel like 0/-1/"UNASSIGNED") wherever `overlaps_cell` is False, and
overwrite the `cell_id` column with it.

Important: `cell_ID` (CosMx) is already globally unique across the whole ROI
(verified against shapes/shapes.parquet's `label` column: zero collisions
across FOVs, and the label range matches `cell_ID`'s range exactly) - do NOT
scope it by `fov`, and do NOT prefer the composite `cell` string column that
exists for cosmx_pancreas: segger's reference-cell step intersects transcript
cell ids against these boundary polygon labels, so the transcript-side id
must match that plain-integer convention exactly, or every transcript gets
filtered out again (empty intersection instead of empty null-filter).

- CosMx hgsoc/melanoma/mousebrain/stomach (but not pancreas): even with the
  cell_id fix above, segger trained but predicted exactly zero assigned
  transcripts (segger_similarity uniformly 0.0, loss_bd stuck at 0.000 for
  every epoch). Root cause: the boundary polygons in shapes/shapes.parquet
  are in raw CosMx pixel coordinates, while the transcripts' x/y are already
  in microns - the two occupy physically disjoint regions of coordinate
  space (verified: boundary bounds * 0.12028 um/px, the standard CosMx pixel
  size, align almost exactly with the transcripts' bounds). Segger's
  cell_id-string matching (used to build the reference AnnData) doesn't
  care about geometry so it still worked, but whatever spatial/geometric
  step produces the boundary training loss found zero valid transcript-in-
  polygon pairs. `stomach` additionally has boundaries from other FOVs
  that were never cropped down to this ROI's transcripts.

Fix: rescale (if needed - detected per-sample, not assumed) the boundary
geometries by the CosMx mpp and crop to the transcripts' spatial extent.
"""
import argparse
import json
import os
import shutil
import sys

import geopandas as gpd
import pandas as pd
from shapely import affinity

DEFAULT_ROIS_DIR = "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois"
DEFAULT_OUTPUT_DIR = "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois_segger_input_fixed"
COSMX_MPP = 0.12028  # microns per pixel, matches segger's CosMxBoundaryFields.mpp


def build_cell_id(df: pd.DataFrame) -> pd.Series:
    if "cell_id" in df.columns:
        # MERSCOPE / xenium_msbrain_300: numeric but already globally unique.
        cell_id = df["cell_id"].astype(str)
    elif "cell_ID" in df.columns:
        # CosMx: already globally unique across the whole ROI, matches the
        # boundary polygon labels directly. Segger's own boundary-id
        # normalization does `bd['label'].astype(str)` on a float64 column
        # (e.g. "21580.0"), so the transcript-side id must be formatted the
        # same way (float-then-str), not a plain int-then-str ("21580") -
        # otherwise the two never match despite being the same cell.
        cell_id = df["cell_ID"].astype(float).astype(str)
    else:
        raise ValueError(f"No recognizable cell-id column found among: {df.columns.tolist()}")

    if "overlaps_cell" not in df.columns:
        raise ValueError("Expected an 'overlaps_cell' column to identify unassigned transcripts.")

    return cell_id.where(df["overlaps_cell"].astype(bool), other=None)


def _bbox_iou(bounds_a, bounds_b):
    ax0, ay0, ax1, ay1 = bounds_a
    bx0, by0, bx1, by1 = bounds_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _synthesize_boundaries_from_transcripts(transcripts_df):
    """Build convex-hull cell boundaries directly from assigned transcripts.

    Fallback for when shapes.parquet's polygons carry no real positional
    relationship to the transcripts at all (verified for cosmx_stomach: an
    iterative RANSAC-style affine refit between per-cell_id boundary
    centroids and per-cell_id transcript centroids never plateaus at a
    stable "good" subset - residuals keep shrinking all the way down to a
    handful of cells, the signature of fitting noise, not a recoverable
    offset/rotation. The boundary/transcript cell_id match is coincidental,
    likely from an id_map collision during the original whole-slide FOV
    stitching, not a real correspondence). The transcripts' own x/y are
    trustworthy (that's what's used throughout this script and what segger
    actually predicts against), so hulls built from them are guaranteed to
    sit exactly where the cells really are, unlike the shipped shapes file.
    """
    from shapely.geometry import MultiPoint

    assigned = transcripts_df[transcripts_df["cell_id"].notna()]
    labels, polygons = [], []
    dropped = []
    for cid, group in assigned.groupby("cell_id"):
        if len(group) >= 3:
            polygons.append(MultiPoint(group[["x", "y"]].values).convex_hull)
            labels.append(cid)
        else:
            # <3 points can't form a polygon. These must also be nulled back
            # out of the transcripts' cell_id (by the caller) - segger's own
            # embedding indexing assumes every cell_id referenced by
            # transcripts has a matching boundary entry, and silently
            # leaving a gap here causes an out-of-bounds index crash deep in
            # its GPU training code, not a clean data-loading error.
            dropped.append(cid)
    gdf = gpd.GeoDataFrame({"label": labels, "geometry": polygons})
    return gdf, dropped


def fix_boundary_geometry(shapes_path, tx_bounds, transcripts_df=None, iou_floor=0.3):
    """Rescale (if needed) and crop the boundary polygons to match the
    transcripts' coordinate system and spatial extent. Only rescales if
    doing so demonstrably improves alignment with tx_bounds (pancreas
    already lines up raw and must not be touched). If even the best-case
    alignment is far too poor to be a coordinate/unit issue, replace the
    boundaries outright with hulls synthesized from the transcripts
    (requires transcripts_df with 'x', 'y', 'cell_id' columns).
    """
    gdf = gpd.read_parquet(shapes_path)
    raw_bounds = tuple(gdf.total_bounds)
    scaled_geom = gdf.geometry.apply(
        lambda g: affinity.scale(g, xfact=COSMX_MPP, yfact=COSMX_MPP, origin=(0, 0))
    )
    scaled_bounds = tuple(gpd.GeoSeries(scaled_geom).total_bounds)

    iou_raw = _bbox_iou(raw_bounds, tx_bounds)
    iou_scaled = _bbox_iou(scaled_bounds, tx_bounds)
    print(f"  boundary/transcript bbox IoU: raw={iou_raw:.3f} scaled_by_mpp={iou_scaled:.3f}")

    if max(iou_raw, iou_scaled) < iou_floor:
        if transcripts_df is None:
            raise ValueError(
                f"Boundary geometry bbox IoU ({max(iou_raw, iou_scaled):.3f}) is too low to "
                "be a simple unit/coordinate issue, and no transcripts_df was given to "
                "synthesize a replacement from."
            )
        print(
            f"  Boundary/transcript IoU ({max(iou_raw, iou_scaled):.3f}) is too low even "
            "after rescaling to be a coordinate issue - synthesizing boundaries from "
            "assigned transcripts instead of trusting shapes.parquet."
        )
        gdf, dropped_cell_ids = _synthesize_boundaries_from_transcripts(transcripts_df)
        print(
            f"  Synthesized {len(gdf)} convex-hull boundaries from transcripts "
            f"({len(dropped_cell_ids)} cell_ids with <3 transcripts could not form a "
            "polygon and must be nulled back out of the transcripts table)."
        )
        gdf.to_parquet(shapes_path)
        return dropped_cell_ids

    if iou_scaled > iou_raw:
        print(f"  Boundary geometry is in pixels; rescaling by {COSMX_MPP} (CosMx mpp).")
        gdf["geometry"] = scaled_geom
    else:
        print("  Boundary geometry already aligned with transcripts; no rescale applied.")

    # Margin needs to be generous enough to absorb small stitching/rounding
    # offsets between the transcripts and boundaries coordinate pipelines.
    # A fixed absolute margin (not proportional to the ROI's own extent,
    # which can be a few hundred um) comfortably covers realistic offsets.
    minx, miny, maxx, maxy = tx_bounds
    margin = 300.0
    padded = (minx - margin, miny - margin, maxx + margin, maxy + margin)
    centroids = gdf.geometry.centroid
    keep = (
        (centroids.x >= padded[0]) & (centroids.x <= padded[2])
        & (centroids.y >= padded[1]) & (centroids.y <= padded[3])
    )
    n_before = len(gdf)
    gdf = gdf.loc[keep]
    print(f"  Cropped boundaries to transcript region: kept {len(gdf)}/{n_before}")

    gdf.to_parquet(shapes_path)
    return []


def fix_table_obs_case_collisions(dst_zarr):
    """Rename obs columns that only differ by case (e.g. cosmx_pancreas has
    both 'cell_id' and 'cell_ID', and both 'x' and 'X') so spatialdata's
    strict validation (which rejects case-variant duplicates) can read the
    table. Renaming (not dropping) since these turned out NOT to be
    redundant copies: 'cell_id' is a composite FOV-scoped string, 'cell_ID'
    a plain numeric label (and is the one actually wired up as
    uns/spatialdata_attrs/instance_key - renaming THAT one breaks the
    table/shapes link with a different, more confusing error); 'x' is a
    real coordinate, 'X' an unrelated flag column. Pure filesystem rename +
    .zattrs edit - no full anndata rewrite.
    """
    import zarr

    obs_dir = os.path.join(dst_zarr, "tables", "table", "obs")
    zattrs_path = os.path.join(obs_dir, ".zattrs")
    if not os.path.exists(zattrs_path):
        return

    protected = set()
    attrs_dir = os.path.join(dst_zarr, "tables", "table", "uns", "spatialdata_attrs")
    if os.path.exists(attrs_dir):
        g = zarr.open(attrs_dir, mode="r")
        for key in ("instance_key", "region_key"):
            if key in g:
                protected.add(str(g[key][...].item() if hasattr(g[key][...], "item") else g[key][0]))

    with open(zattrs_path) as f:
        attrs = json.load(f)
    cols = attrs.get("column-order", [])
    seen_lower = {}
    renames = {}
    for c in cols:
        key = c.lower()
        if key in seen_lower:
            # prefer renaming whichever of the pair isn't protected
            other = seen_lower[key]
            if other not in protected:
                renames[other] = f"{other}_dup"
                seen_lower[key] = c
            elif c not in protected:
                renames[c] = f"{c}_dup"
            else:
                raise ValueError(f"Both case-variants of '{key}' are protected (instance_key/region_key): {c!r}, {other!r}")
        else:
            seen_lower[key] = c
    if not renames:
        print("  No obs case-collisions found in table.")
        return
    print(f"  Renaming case-colliding obs columns: {renames}")
    attrs["column-order"] = [renames.get(c, c) for c in cols]
    with open(zattrs_path, "w") as f:
        json.dump(attrs, f)
    for old, new in renames.items():
        os.rename(os.path.join(obs_dir, old), os.path.join(obs_dir, new))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=str)
    parser.add_argument("--rois-dir", type=str, default=DEFAULT_ROIS_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    src_zarr = os.path.join(args.rois_dir, args.sample, "data.zarr")
    dst_zarr = os.path.join(args.output_dir, args.sample, "data.zarr")

    if not os.path.exists(src_zarr):
        print(f"SUMMARY sample={args.sample} status=failed error='source data.zarr not found: {src_zarr}'")
        sys.exit(1)

    try:
        if os.path.exists(dst_zarr):
            print(f"Removing existing staging copy: {dst_zarr}")
            shutil.rmtree(dst_zarr)
        os.makedirs(os.path.dirname(dst_zarr), exist_ok=True)

        print(f"[{args.sample}] Copying {src_zarr} -> {dst_zarr}")
        shutil.copytree(src_zarr, dst_zarr)

        parquet_dir = os.path.join(dst_zarr, "points", "transcripts", "points.parquet")
        part_files = sorted(
            os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith(".parquet")
        )
        if not part_files:
            raise FileNotFoundError(f"No parquet part files found under {parquet_dir}")

        is_cosmx = False
        tx_bounds = None
        tx_for_synthesis = []
        for part in part_files:
            print(f"[{args.sample}] Fixing cell_id in {part}")
            df = pd.read_parquet(part)
            is_cosmx = is_cosmx or ("cell_ID" in df.columns and "cell_id" not in df.columns)
            part_bounds = (df["x"].min(), df["y"].min(), df["x"].max(), df["y"].max())
            tx_bounds = part_bounds if tx_bounds is None else (
                min(tx_bounds[0], part_bounds[0]), min(tx_bounds[1], part_bounds[1]),
                max(tx_bounds[2], part_bounds[2]), max(tx_bounds[3], part_bounds[3]),
            )
            new_cell_id = build_cell_id(df)
            n_assigned_before = (df["overlaps_cell"]).sum()
            df["cell_id"] = new_cell_id
            print(
                f"  rows={len(df)} assigned={n_assigned_before} "
                f"null_cell_id={df['cell_id'].isna().sum()} "
                f"unique_cell_id={df['cell_id'].nunique()}"
            )
            df.to_parquet(part)
            tx_for_synthesis.append(df[["x", "y", "cell_id"]])

        if is_cosmx:
            shapes_path = os.path.join(dst_zarr, "shapes", "shapes", "shapes.parquet")
            if os.path.exists(shapes_path):
                print(f"[{args.sample}] Fixing boundary geometry in {shapes_path}")
                transcripts_df = pd.concat(tx_for_synthesis, ignore_index=True)
                dropped_cell_ids = fix_boundary_geometry(shapes_path, tx_bounds, transcripts_df=transcripts_df)
                if dropped_cell_ids:
                    print(
                        f"[{args.sample}] Nulling {len(dropped_cell_ids)} cell_ids with no "
                        "synthesized boundary back out of the transcripts (segger's own "
                        "embedding indexing requires every referenced cell_id to have a "
                        "matching boundary entry)."
                    )
                    dropped_set = set(dropped_cell_ids)
                    for part in part_files:
                        df = pd.read_parquet(part)
                        df.loc[df["cell_id"].isin(dropped_set), "cell_id"] = None
                        df.to_parquet(part)
            else:
                print(f"[{args.sample}] No shapes.parquet found at {shapes_path}; skipping boundary fix.")

        print(f"[{args.sample}] Checking table obs for case-variant column collisions")
        fix_table_obs_case_collisions(dst_zarr)

        print(f"SUMMARY sample={args.sample} status=success output={dst_zarr}")
    except Exception as e:
        print(f"SUMMARY sample={args.sample} status=failed error={e!r}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
