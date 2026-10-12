#!/usr/bin/env python3
"""uRNA source score with vs. without the extracellular neighborhood context (xenium_msbrain_5k).

The production source score (recompute_source_score_final_lambda05.py) builds, for every
extracellular transcript, a *context gene set* from its k nearest extracellular neighbors,
where k grows from 1 (enrichment_over_random <= ambient_floor) to max_k=10
(enrichment >= signal_threshold). Candidate parent cells are then weighted by
(fraction of context genes they express) * exp(-lambda * distance).

Two modes, everything else identical (lambda=0.5, max_dist=200, residual=0.1, same shell,
same expression matrix, same ambient_floor):

  * with_neighbors  : production settings, max_k=10 (adaptive k).
  * no_neighbors    : max_k=1, i.e. the context is ONLY the transcript's own gene, so a
                      candidate's weight is exp(-lambda*d) if it expresses the gene, else 0.
                      The highest-weighted parent is the closest expressing cell.

Outputs (parquet / csv, no zarr is modified) in
  <data_dir>/source_score_neighborhood_comparison/
    real_per_transcript.parquet      per extracellular transcript, both modes
    real_per_cell.csv                urna_source_score per cell, both modes
    real_gene_by_type_<mode>.csv     summed type mass per gene (extrasomatic attribution)
    benchmark_seed<s>_dens<d>_<expr>.parquet  peripheral hold-out benchmark (ground truth known)
    run_info.json

Benchmark (ground truth): for a random 10% of cells, the outer 30% of their transcripts
(by distance to the cell's transcript centroid) are removed from the cell (expression and
shell are rebuilt without them) and inserted into the extracellular pool. They are then
scored like any uRNA and we check whether the true parent cell / cell type is recovered.
`density` < 1 keeps only that fraction of the held-out ring in the pool (the rest is
dropped entirely) to mimic sparser uRNA clouds. A sweep over max_k is run on these queries.
expr_mode="truncated" removes the held-out ring from the cell's expression (realistic: genes only
detected outside the mask are not "expressed" by the cell); expr_mode="intact" keeps the full
expression profile, isolating the effect of the spatial neighborhood itself.

Usage: python source_score_neighborhood_comparison.py [--test]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import dask

dask.config.set({"dataframe.query-planning": False})

import numpy as np
import pandas as pd
import spatialdata as sd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

from troutpy.tl.source_cell import _core_scoring_engine_chunk  # noqa: E402

DATA_DIR = Path("/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_msbrain_5k")

# Production parameters (recompute_source_score_final_lambda05.py)
PARAMS = dict(max_dist=200.0, lmbda=0.5, max_k=10, signal_threshold=10.0, residual=0.1, chunk_size=100_000)
CELL_TYPE_COL = "cell_type"
MODES = {"with_neighbors": 10, "no_neighbors": 1}

BENCH_SEEDS = [0, 1, 2]
BENCH_DENSITIES = [1.0, 0.2]
BENCH_CELL_FRAC = 0.10
BENCH_PERIPH_FRAC = 0.30
BENCH_MAX_K = [1, 2, 5, 10, 20]

TX_COLS = ["x", "y", "gene", "cell_id", "enrichment_over_random", "density_similarity_score"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def k_from_enrichment(enrich, max_k, ambient_floor, signal_threshold):
    """Adaptive k exactly as in tl.urna_source_score."""
    return np.clip(1 + (max_k - 1) * (enrich - ambient_floor) / (signal_threshold - ambient_floor), 1, max_k).astype(int)


def build_shell(assigned_df, cell_id_to_row, row_type_idx):
    """Outermost (min/max x, min/max y) transcript per cell, as in production."""
    g = assigned_df.groupby("cell_id")
    shell_idx = pd.concat([g["x"].idxmin(), g["x"].idxmax(), g["y"].idxmin(), g["y"].idxmax()]).unique()
    shell_df = assigned_df.loc[shell_idx]
    coords = shell_df[["x", "y"]].values.astype(np.float32)
    rows = np.array([cell_id_to_row[c] for c in shell_df["cell_id"]], dtype=np.int32)
    return coords, rows, row_type_idx[rows].astype(np.int32), cKDTree(coords)


def score(query_coords, query_pool_idx, pool_tree, pool_gene_idxs, k_vals, max_k, shell, csr, n_types, p=PARAMS):
    """Chunked call of the production numba kernel.

    max_k == 1 -> the context is the query's own gene (self index), never a duplicate-coordinate
    neighbor; max_k > 1 -> k nearest pool transcripts, as in production.
    """
    shell_coords, shell_rows, shell_types, shell_tree = shell
    n = len(query_coords)
    prob = np.zeros((n, n_types), dtype=np.float32)
    best_rows = np.full(n, -1, dtype=np.int32)
    best_scores = np.zeros(n, dtype=np.float32)
    best_d = np.zeros(n, dtype=np.float32)
    cs = p["chunk_size"]
    for start in range(0, n, cs):
        end = min(start + cs, n)
        cc = query_coords[start:end]
        if max_k == 1:
            neighs = query_pool_idx[start:end, None].astype(np.int64)
        else:
            _, neighs = pool_tree.query(cc, k=max_k)
            neighs = np.asarray(neighs).reshape(len(cc), max_k)
        sh = shell_tree.query_ball_point(cc, r=p["max_dist"])
        lens = np.fromiter((len(s) for s in sh), dtype=np.int64, count=len(sh))
        s_ptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
        s_flat = np.fromiter((j for s in sh for j in s), dtype=np.int32, count=int(lens.sum()))
        pm, br, bs, bd = _core_scoring_engine_chunk(
            cc, pool_gene_idxs, neighs, k_vals[start:end], shell_coords, shell_rows, shell_types,
            s_flat, s_ptr, csr.data, csr.indices, csr.indptr, p["lmbda"], p["residual"], n_types,
        )
        prob[start:end], best_rows[start:end], best_scores[start:end], best_d[start:end] = pm, br, bs, bd
    return prob, best_rows, best_scores, best_d


def type_summary(prob):
    """argmax type, fraction of type mass on argmax, normalized entropy, total mass."""
    mass = prob.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pn = prob / mass[:, None]
        h = -np.nansum(np.where(pn > 0, pn * np.log(pn), 0), axis=1) / np.log(prob.shape[1])
        maxfrac = np.nanmax(pn, axis=1)
    arg = np.where(mass > 0, prob.argmax(1), -1).astype(np.int16)
    h[mass == 0] = np.nan
    maxfrac[mass == 0] = np.nan
    return arg, maxfrac.astype(np.float32), h.astype(np.float32), mass.astype(np.float32)


def gene_by_type(gene_codes, n_genes, prob):
    ind = coo_matrix((np.ones(len(gene_codes), np.float32), (gene_codes, np.arange(len(gene_codes)))), shape=(n_genes, len(gene_codes))).tocsr()
    return np.asarray(ind @ prob)


def main(test=False):
    out_dir = DATA_DIR / ("source_score_neighborhood_comparison" + ("_test" if test else ""))
    out_dir.mkdir(exist_ok=True)
    t0 = time.time()

    log("Loading data_annotated.zarr (input of the production lambda=0.5 run)")
    sdata = sd.read_zarr(DATA_DIR / "data_annotated.zarr")
    cells = sdata["table"]
    obs = cells.obs.copy()
    obs["cell_id"] = obs["cell_id"].astype(str).str.strip()
    tx = sdata.points["transcripts"][TX_COLS].compute()
    tx["cell_id"] = tx["cell_id"].astype(str).str.strip()
    log(f"{len(tx)} transcripts, {len(obs)} cells, index unique={tx.index.is_unique}")

    # ambient floor exactly as production (computed on the full ROI, also in test mode)
    ambient_floor = float(tx.loc[tx["density_similarity_score"] > 0.5, "enrichment_over_random"].min())
    log(f"ambient_floor = {ambient_floor}")

    if test:  # small crop for a quick end-to-end check
        x0, y0 = tx["x"].quantile(0.45), tx["y"].quantile(0.45)
        crop = (tx["x"].between(x0, x0 + 600)) & (tx["y"].between(y0, y0 + 600))
        tx = tx[crop]
        log(f"TEST crop: {len(tx)} transcripts")

    unique_types = np.sort(obs[CELL_TYPE_COL].unique())
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    row_type_idx = np.array([type_to_idx[t] for t in obs[CELL_TYPE_COL]], dtype=np.int32)
    cell_id_to_row = {c: i for i, c in enumerate(obs["cell_id"])}
    gene_to_idx = {g: i for i, g in enumerate(cells.var_names)}
    n_types = len(unique_types)

    is_assigned = tx["cell_id"].isin(set(obs["cell_id"]))
    et_df = tx[~is_assigned]
    assigned_df = tx[is_assigned]

    csr_prod = csr_matrix(cells.X)  # production: sparsity pattern of X = "expressed"
    log(f"X nnz={csr_prod.nnz}, true nonzeros={csr_prod.count_nonzero()}")
    shell = build_shell(assigned_df, cell_id_to_row, row_type_idx)

    # ------------------------------------------------------------------ real data
    et_coords = et_df[["x", "y"]].values.astype(np.float32)
    et_gene_idxs = np.array([gene_to_idx.get(g, -1) for g in et_df["gene"]], dtype=np.int32)
    enrich = et_df["enrichment_over_random"].values
    log(f"Building pool tree over {len(et_coords)} extracellular transcripts")
    et_tree = cKDTree(et_coords)
    self_idx = np.arange(len(et_coords))

    gene_cat = pd.Categorical(et_df["gene"].astype(str))
    real = pd.DataFrame({
        "tx_index": et_df.index.values, "gene": gene_cat, "x": et_coords[:, 0], "y": et_coords[:, 1],
        "enrichment": enrich.astype(np.float32),
        "k_prod": k_from_enrichment(enrich, PARAMS["max_k"], ambient_floor, PARAMS["signal_threshold"]).astype(np.int16),
    })
    P_dense = csr_prod.toarray() != 0  # 'expressed' lookup, same pattern the kernel searches
    per_cell = obs[["cell_id", CELL_TYPE_COL]].copy()
    for mode, mk in MODES.items():
        log(f"Scoring real data: {mode} (max_k={mk})")
        k_vals = k_from_enrichment(enrich, mk, ambient_floor, PARAMS["signal_threshold"])
        prob, br, bs, bd = score(et_coords, self_idx, et_tree, et_gene_idxs, k_vals, mk, shell, csr_prod, n_types)
        arg, maxfrac, h, mass = type_summary(prob)
        valid = br != -1
        expr_in_parent = np.zeros(len(br), dtype=bool)
        ok = valid & (et_gene_idxs >= 0)
        expr_in_parent[ok] = P_dense[br[ok], et_gene_idxs[ok]]
        real[f"{mode}__parent_row"] = br
        real[f"{mode}__score"] = bs
        real[f"{mode}__dist"] = bd
        real[f"{mode}__type"] = arg
        real[f"{mode}__type_maxfrac"] = maxfrac
        real[f"{mode}__entropy"] = h
        real[f"{mode}__mass"] = mass
        real[f"{mode}__gene_expr_in_parent"] = expr_in_parent
        cs = np.zeros(len(obs))
        np.add.at(cs, br[valid], bs[valid])
        per_cell[f"{mode}__urna_source_score"] = cs
        per_cell[f"{mode}__n_assigned"] = np.bincount(br[valid], minlength=len(obs))
        gbt = gene_by_type(gene_cat.codes, len(gene_cat.categories), prob)
        pd.DataFrame(gbt, index=gene_cat.categories, columns=unique_types).to_csv(out_dir / f"real_gene_by_type_{mode}.csv")
        log(f"  assigned {valid.sum()} / {len(br)}")
        del prob

    # sanity check: with_neighbors must reproduce the stored production result
    if not test:
        import anndata as ad
        prod = ad.read_zarr(DATA_DIR / "data_annotated_lambda0.5.zarr" / "tables" / "source_score").obs
        assert len(prod) == len(real), (len(prod), len(real))  # same et_df construction -> same order
        mine = np.where(real["with_neighbors__parent_row"] >= 0, obs["cell_id"].values[real["with_neighbors__parent_row"].clip(0)], "unassigned")
        agree = float(np.mean(prod["predicted_parent"].astype(str).values == mine))
        dscore = float(np.nanmax(np.abs(prod["assignment_score"].values - real["with_neighbors__score"].values)))
        log(f"Reproduction of production run: parent agreement={agree:.6f}, max |score diff|={dscore:.2e}")
    else:
        agree, dscore = None, None

    real.to_parquet(out_dir / "real_per_transcript.parquet", index=False)
    per_cell.to_csv(out_dir / "real_per_cell.csv", index=False)
    del et_tree

    # ------------------------------------------------------------------ benchmark
    raw = cells.layers["raw"].tocsr()
    a = assigned_df.reset_index(drop=True)
    a_row = np.array([cell_id_to_row[c] for c in a["cell_id"]], dtype=np.int32)
    a_gene = np.array([gene_to_idx.get(g, -1) for g in a["gene"]], dtype=np.int32)
    present_rows = np.unique(a_row)
    for seed in BENCH_SEEDS:
        rng = np.random.default_rng(seed)
        sampled = rng.choice(present_rows, size=max(1, int(BENCH_CELL_FRAC * len(present_rows))), replace=False)
        in_s = np.isin(a_row, sampled)
        sub = a.loc[in_s, ["x", "y"]].copy()
        sub["row"] = a_row[in_s]
        cen = sub.groupby("row")[["x", "y"]].transform("mean")
        sub["dc"] = np.hypot(sub["x"] - cen["x"], sub["y"] - cen["y"])
        pct = sub.groupby("row")["dc"].rank(pct=True, method="first")
        periph_pos = sub.index[pct > 1 - BENCH_PERIPH_FRAC].values  # positions in `a`

        for dens in BENCH_DENSITIES:
            keep = rng.random(len(periph_pos)) < dens
            q_pos, removed_pos = periph_pos[keep], periph_pos

            # cell expression without the whole held-out ring (kept + dropped)
            rm_ok = a_gene[removed_pos] >= 0
            rm = coo_matrix((np.ones(rm_ok.sum(), np.float32), (a_row[removed_pos][rm_ok], a_gene[removed_pos][rm_ok])), shape=raw.shape).tocsr()
            new_raw = (raw - rm).tocsr()
            Pb = P_dense.copy()
            Pb[sampled] = new_raw[sampled].toarray() > 0
            csr_b = csr_matrix(Pb.astype(np.float32))
            csr_b.sort_indices()

            remaining = a.drop(index=removed_pos)
            shell_b = build_shell(remaining, cell_id_to_row, row_type_idx)

            q = a.loc[q_pos]
            pool_coords = np.vstack([et_coords, q[["x", "y"]].values.astype(np.float32)])
            pool_genes = np.concatenate([et_gene_idxs, a_gene[q_pos]])
            pool_tree = cKDTree(pool_coords)
            q_pool_idx = np.arange(len(et_coords), len(pool_coords))
            q_coords = pool_coords[q_pool_idx]
            q_enrich = q["enrichment_over_random"].values
            true_row = a_row[q_pos]

            d_near, i_near = shell_b[3].query(q_coords, k=1)
            bench = pd.DataFrame({
                "seed": seed, "density": dens, "true_row": true_row, "true_type": row_type_idx[true_row].astype(np.int16),
                "gene_idx": a_gene[q_pos], "enrichment": q_enrich.astype(np.float32),
                "k_prod": k_from_enrichment(q_enrich, PARAMS["max_k"], ambient_floor, PARAMS["signal_threshold"]).astype(np.int16),
                "nearest_row": shell_b[1][i_near], "dist_nearest": d_near.astype(np.float32),
                "gene_expr_in_true_after_holdout": Pb[true_row, np.maximum(a_gene[q_pos], 0)] & (a_gene[q_pos] >= 0),
            })
            # truncated: true cell's expression lacks the held-out ring (realistic);
            # intact: expression from the full cell, only positions/shell change (isolates the spatial-context effect)
            csr_intact = csr_matrix(P_dense.astype(np.float32))
            csr_intact.sort_indices()
            for expr_mode, csr_use in [("truncated", csr_b), ("intact", csr_intact)]:
                bench_e = bench.copy()
                bench_e["expr_mode"] = expr_mode
                for mk in BENCH_MAX_K:
                    log(f"Benchmark seed={seed} density={dens} expr={expr_mode} max_k={mk}: {len(q_coords)} queries")
                    k_vals = k_from_enrichment(q_enrich, mk, ambient_floor, PARAMS["signal_threshold"])
                    prob, br, bs, bd = score(q_coords, q_pool_idx, pool_tree, pool_genes, k_vals, mk, shell_b, csr_use, n_types)
                    arg, maxfrac, h, mass = type_summary(prob)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        true_frac = prob[np.arange(len(prob)), bench_e["true_type"].values] / mass
                    bench_e[f"k{mk}__parent_row"] = br
                    bench_e[f"k{mk}__score"] = bs
                    bench_e[f"k{mk}__type"] = arg
                    bench_e[f"k{mk}__true_type_frac"] = true_frac.astype(np.float32)
                    bench_e[f"k{mk}__entropy"] = h
                bench_e.to_parquet(out_dir / f"benchmark_seed{seed}_dens{dens}_{expr_mode}.parquet", index=False)
            del pool_tree

    info = dict(params=PARAMS, modes=MODES, ambient_floor=ambient_floor, cell_type_col=CELL_TYPE_COL,
                cell_types=list(map(str, unique_types)), cell_ids=list(obs["cell_id"]),
                bench=dict(seeds=BENCH_SEEDS, densities=BENCH_DENSITIES, cell_frac=BENCH_CELL_FRAC,
                           periph_frac=BENCH_PERIPH_FRAC, max_k=BENCH_MAX_K),
                reproduction_parent_agreement=agree, reproduction_max_score_diff=dscore,
                test=test, runtime_min=(time.time() - t0) / 60)
    json.dump(info, open(out_dir / "run_info.json", "w"))
    log(f"Done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    main(ap.parse_args().test)
