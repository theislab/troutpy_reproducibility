#!/usr/bin/env python3
"""
Run SOM (self-organizing map) clustering on the morphology image of a
SpatialData ROI and save the resulting community labels.

Ported from the interactive SOM section of
notebooks/4_detailed_analysis/rcc_som_multimodal_part2.ipynb for batch
execution on the cluster (see the paired .sbatch file in this folder).

Steps:
    1. Load the multi-channel morphology image and cluster its foreground
       pixels into SOM communities (clip -> normalize -> subsample -> SOM
       fit -> kNN graph -> Leiden clustering).
    2. Save a QC image of the community map and the per-community mean
       channel intensities.
    3. Map every transcript to its overlapping SOM community and persist
       that as a new 'som_community' column on the transcripts table.
    4. Add the community map itself as a labels layer ('som_communities')
       and write everything to a new zarr store.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spatialdata as sd
import spatiomic as so
from scipy.ndimage import map_coordinates
from spatialdata.models import Labels2DModel, PointsModel
from spatialdata.transformations import Identity
from tqdm import tqdm


def load_image_array(sdata, image_key, scale):
    image_element = sdata.images[image_key][scale]
    data = np.array(image_element.image)  # (C, Y, X)
    channels = np.array(image_element.c)
    data = data.transpose(1, 2, 0)  # (Y, X, C)
    return data, channels


def run_som_clustering(
    data,
    foreground_threshold,
    clip_min,
    clip_max,
    subsample_count,
    som_grid,
    iterations,
    knn_neighbors,
    leiden_resolution,
    seed,
):
    max_projection = data.max(axis=-1)
    binary_mask = max_projection > foreground_threshold

    clipper = so.process.clip(percentile_min=clip_min, percentile_max=clip_max)
    normalizer = so.process.normalize()
    data_clipped = clipper.fit_transform(data)
    data_normalized = normalizer.fit_transform(data_clipped)

    subsample = so.data.subsample().fit_transform(
        data_normalized, method="count", count=subsample_count, seed=seed
    )

    som = so.dimension.som(node_count=som_grid, dimension_count=subsample.shape[1], seed=seed)
    som.fit(subsample, iteration_count=iterations)

    knn_graph = so.neighbor.knn_graph().create(som, neighbor_count=knn_neighbors)
    communities, modularity = so.cluster.leiden().predict(knn_graph, resolution=leiden_resolution)
    communities = np.array(communities) + 1  # reserve 0 for background/non-foreground pixels

    labels = som.label(data_normalized, communities)

    background_image = np.zeros_like(max_projection)
    background_image[binary_mask] = labels[binary_mask].flatten()

    print(
        f"SOM clustering found {len(np.unique(communities))} communities "
        f"(modularity={modularity:.3f}) covering "
        f"{binary_mask.sum():,}/{binary_mask.size:,} foreground pixels."
    )
    return background_image, communities


def save_cluster_image(background_image, communities, out_path):
    fig = so.plot.cluster_image(
        background_image,
        colormap=so.plot.colormap(
            color_count=len(np.unique(communities)) + 1,
            flavor="glasbey",
            color_override={0: "#000000"},
        ),
        figsize=(12, 12),
        show_colorbar=False,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def compute_community_means(data, background_image, channel_names):
    """Mean intensity per channel for each SOM community (excludes background/0)."""
    unique_communities = np.unique(background_image)
    unique_communities = unique_communities[unique_communities != 0]

    results = []
    for community_id in tqdm(unique_communities, desc="Calculating mean intensities"):
        mask = background_image == community_id
        means = data[mask].mean(axis=0)
        row = {"community_id": int(community_id)}
        row.update({ch: means[i] for i, ch in enumerate(channel_names)})
        results.append(row)

    return pd.DataFrame(results).set_index("community_id")


def map_transcripts_to_som(transcripts_df, background_image):
    """Nearest-neighbor lookup of each transcript's pixel into the SOM community map."""
    minx, maxx = transcripts_df["x"].min(), transcripts_df["x"].max()
    miny, maxy = transcripts_df["y"].min(), transcripts_df["y"].max()

    img_h, img_w = background_image.shape
    mfx = img_w / (maxx - minx)
    mfy = img_h / (maxy - miny)

    x_scaled = (transcripts_df["x"] - minx) * mfx
    y_scaled = (transcripts_df["y"] - miny) * mfy

    y_coords = np.clip(y_scaled.values, 0, img_h - 1)
    x_coords = np.clip(x_scaled.values, 0, img_w - 1)

    return map_coordinates(background_image, [y_coords, x_coords], order=0, mode="nearest")


def main(args):
    figures_path = os.path.join(args.inpath, args.figures_dirname)
    os.makedirs(figures_path, exist_ok=True)

    zarr_path = os.path.join(args.inpath, args.zarr_name)
    print(f"Loading spatial data from: {zarr_path}")
    sdata = sd.read_zarr(zarr_path)

    data, channels = load_image_array(sdata, args.image_key, args.scale)

    if args.crop:
        y1, y2, x1, x2 = (int(v) for v in args.crop.split(","))
        data = data[y1:y2, x1:x2, :]
        print(
            f"Cropped image to y=({y1},{y2}) x=({x1},{x2}); note the saved "
            "labels layer keeps an Identity transform, so downstream spatial "
            "alignment will be off by the crop offset."
        )

    som_grid = tuple(int(v) for v in args.som_grid.split(","))

    background_image, communities = run_som_clustering(
        data,
        foreground_threshold=args.foreground_threshold,
        clip_min=args.clip_percentile_min,
        clip_max=args.clip_percentile_max,
        subsample_count=args.subsample_count,
        som_grid=som_grid,
        iterations=args.iterations,
        knn_neighbors=args.knn_neighbors,
        leiden_resolution=args.leiden_resolution,
        seed=args.seed,
    )

    save_cluster_image(background_image, communities, os.path.join(figures_path, "som_cluster_image.png"))

    df_intensities = compute_community_means(data, background_image, channels)
    df_intensities.to_csv(os.path.join(figures_path, "som_community_mean_intensities.csv"))

    print("Mapping transcripts to SOM communities...")
    transcripts_df = sdata.points["transcripts"].compute()
    transcripts_df["som_community"] = map_transcripts_to_som(transcripts_df, background_image)

    counts = transcripts_df["som_community"].value_counts().sort_index()
    counts.to_csv(os.path.join(figures_path, "som_community_transcript_counts.csv"), header=["n_transcripts"])
    print(counts)

    sdata.points["transcripts"] = PointsModel.parse(transcripts_df)
    sdata.labels["som_communities"] = Labels2DModel.parse(
        background_image, dims=("y", "x"), transformations={"global": Identity()}
    )

    out_zarr = os.path.join(args.inpath, args.output_zarr_name)
    print(f"Writing output to: {out_zarr}")
    sdata.write(out_zarr, overwrite=True)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SOM clustering on the morphology image of a SpatialData ROI "
        "and save the resulting community labels/assignments."
    )
    parser.add_argument("inpath", type=str, help="Path to the directory containing the input zarr store")
    parser.add_argument("--zarr_name", type=str, default="data_with_source_score.zarr")
    parser.add_argument("--output_zarr_name", type=str, default="data_with_som.zarr")
    parser.add_argument("--image_key", type=str, default="morphology_focus")
    parser.add_argument("--scale", type=str, default="scale0")
    parser.add_argument("--crop", type=str, default=None, help="Optional 'y1,y2,x1,x2' pixel crop of the image")
    parser.add_argument("--foreground_threshold", type=float, default=100.0)
    parser.add_argument("--clip_percentile_min", type=float, default=1.0)
    parser.add_argument("--clip_percentile_max", type=float, default=99.9)
    parser.add_argument("--subsample_count", type=int, default=100000)
    parser.add_argument("--som_grid", type=str, default="20,20", help="SOM node grid as 'rows,cols'")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--knn_neighbors", type=int, default=10)
    parser.add_argument("--leiden_resolution", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--figures_dirname", type=str, default="figures/som")

    main(parser.parse_args())
