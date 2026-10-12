"""
Does local-uRNA background subtraction make the expression profiles "purer"?

Metrics (all computed on held-out cells, markers defined on the other half using UNCORRECTED data):
  1. Negative-marker fraction : share of a cell's counts that come from marker genes of OTHER lineages / cell types.
  2. Own-marker retention     : share of the cell's own-lineage marker signal that is kept (sensitivity check).
  3. MECR                     : mutually exclusive co-expression rate (Hartman & Satija) over marker pairs from
                                different lineages: n(both) / n(either). Lower = purer.

Because any removal of signal lowers co-expression, every metric is compared against two controls that remove the
same amount of signal WITHOUT using uRNA information:
  - uniform : every cell is scaled by (1 - its own removed fraction)               (cell-wise matched, gene-blind)
  - permuted: each gene's removed fractions are shuffled across the cells that
              express it                                                           (gene-wise matched, cell-blind)
A real gain = corrected beats both controls.
"""
import argparse
import os
import warnings
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
warnings.filterwarnings("ignore")

DATA = Path("/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_msbrain_5k/data_annotated_lambda0.5_decontaminated.zarr/tables/table")
OUT = Path("/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/ms_figures/xenium_msbrain_5k/purity_evaluation")

LINEAGE = {
    "DG": "Neurons", "CA1-2": "Neurons", "CA3": "Neurons", "EXC neurons Thal.": "Neurons",
    "EXC. neurons CTX": "Neurons", "INH neurons": "Neurons", "INH neurons CP": "Neurons",
    "SGZ Neuroblasts": "Neurons", "Cajal-Retzius": "Neurons",
    "Astrocytes": "Astrocytes", "Astrocytes Thal.": "Astrocytes",
    "Oligodendrocytes": "Oligodendrocytes", "OPC": "OPC", "Microglia": "Microglia",
    "Endothelial": "Endothelial", "SMC/Pericytes": "SMC/Pericytes",
    "Ependymal cells": "Ependymal/CP", "Choroid plexus": "Ependymal/CP",
}
COND = ["uncorrected", "corrected", "uniform", "permuted"]
COL = {"uncorrected": "#9e9e9e", "corrected": "#d95f02", "uniform": "#7fb3d5", "permuted": "#82c785"}
TAUS = [0.25, 0.5, 0.75, 1.0]
TAU0 = 0.5

p = argparse.ArgumentParser()
p.add_argument("--n_markers", type=int, default=15)
p.add_argument("--n_boot", type=int, default=200)
p.add_argument("--n_perm", type=int, default=5)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--levels", default="lineage,cell_type")
args = p.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)

# ---------------------------------------------------------------- load
print("loading", flush=True)
a = ad.read_zarr(DATA)
dense = lambda m: (m.toarray() if sp.issparse(m) else np.asarray(m)).astype(np.float32)
R = dense(a.layers["raw"])
D = np.minimum(dense(a.layers["decontaminated_density"]), R)
genes = np.array(a.var_names)
ct = a.obs["cell_type"].astype(str).values
ct_colors = dict(zip(a.obs["cell_type"].astype("category").cat.categories, a.uns.get("cell_type_colors", [])))
n_cells = R.shape[0]
print(R.shape, "removed fraction overall: %.3f" % (1 - D.sum() / R.sum()), flush=True)

# ---------------------------------------------------------------- controls (use the same removal budget, no uRNA info)
f_cell = 1 - D.sum(1) / np.maximum(R.sum(1), 1e-9)
U = R * (1 - f_cell)[:, None]


def perm_control(seed):
    r = np.random.default_rng(seed)
    C = R.copy()
    for j in range(R.shape[1]):
        nz = np.flatnonzero(R[:, j] > 0)
        if nz.size < 2:
            continue
        frac = np.clip(1 - D[nz, j] / R[nz, j], 0, 1)
        C[nz, j] = R[nz, j] * (1 - r.permutation(frac))
    return C


print("building permuted controls", flush=True)
PERMS = [perm_control(args.seed + 1 + k) for k in range(args.n_perm)]
MATS = {"uncorrected": [R], "corrected": [D], "uniform": [U], "permuted": PERMS}

# ---------------------------------------------------------------- split cells: markers from A (raw), evaluation on B
perm_idx = rng.permutation(n_cells)
idxA, idxB_all = np.sort(perm_idx[: n_cells // 2]), np.sort(perm_idx[n_cells // 2:])
norm = R / np.maximum(R.sum(1, keepdims=True), 1) * np.median(R.sum(1))


def select_markers(labels, K, min_det=0.10, min_mean=0.05, min_l2fc=1.0, eps=0.02):
    groups = np.unique(labels[idxA])
    mean = np.stack([norm[idxA][labels[idxA] == g].mean(0) for g in groups])
    det = np.stack([(R[idxA][labels[idxA] == g] > 0).mean(0) for g in groups])
    rows = []
    for gi, g in enumerate(groups):
        omax = np.delete(mean, gi, 0).max(0)
        l2fc = np.log2((mean[gi] + eps) / (omax + eps))
        ok = (mean[gi] >= omax) & (det[gi] >= min_det) & (mean[gi] >= min_mean) & (l2fc >= min_l2fc)
        for j in np.flatnonzero(ok)[np.argsort(-l2fc[ok])][:K]:
            rows.append((g, genes[j], j, l2fc[j], mean[gi, j], det[gi, j]))
    return pd.DataFrame(rows, columns=["group", "gene", "col", "log2fc", "mean_in", "det_in"])


def weighted_jaccard(Xm):
    # continuous co-expression of marker pairs: sum_i min(x_ia, x_ib) / sum_i max(x_ia, x_ib); suits soft (non-integer) counts
    X64 = Xm.astype(np.float64)
    L1 = cdist(X64.T, X64.T, "cityblock")
    sm = X64.sum(0)
    S = sm[:, None] + sm[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (S - L1) / (S + L1)


def mecr_pairs(B, ml, min_either=20):
    n = B.sum(0).astype(float)
    co = B.T.astype(np.float32) @ B.astype(np.float32)
    either = n[:, None] + n[None, :] - co
    with np.errstate(divide="ignore", invalid="ignore"):
        m = co / either
    return m, either


def run_level(level, labels):
    print(f"\n===== level: {level} =====", flush=True)
    mk = select_markers(labels, args.n_markers)
    cnt = mk.group.value_counts()
    keep = cnt[cnt >= 3].index
    mk = mk[mk.group.isin(keep)].reset_index(drop=True)
    groups = [g for g in pd.unique(mk.group)]
    print(cnt.to_dict(), flush=True)
    mk.to_csv(OUT / f"markers_{level}.csv", index=False)

    idxB = idxB_all[np.isin(labels[idxB_all], groups)]
    gid = {g: i for i, g in enumerate(groups)}
    lab = np.array([gid[x] for x in labels[idxB]])
    ml = np.array([gid[g] for g in mk.group])
    mcols = mk.col.values
    F = ml[None, :] != lab[:, None]  # foreign-marker mask, cells x markers
    nB, nM = len(idxB), len(mk)
    pair_mask = ml[:, None] != ml[None, :]
    triu = np.triu(np.ones((nM, nM), bool), 1)

    # per-condition, per-replicate cached quantities (full matrices only needed for row totals and marker columns)
    cache = {}
    for c, mats in MATS.items():
        cache[c] = []
        for X in mats:
            Xm = X[idxB][:, mcols]
            cache[c].append({"Xm": Xm, "tot": X[idxB].sum(1), "bin": {t: Xm >= t for t in TAUS}})
    Rm, Dm = R[idxB][:, mcols], D[idxB][:, mcols]
    Rtot, Dtot = R[idxB].sum(1), D[idxB].sum(1)
    R_for, D_for = (Rm * F).sum(1), (Dm * F).sum(1)
    R_own, D_own = (Rm * ~F).sum(1), (Dm * ~F).sum(1)
    REM = {"own": (R_own, D_own), "foreign": (R_for, D_for), "other": (Rtot - R_for - R_own, Dtot - D_for - D_own)}
    # valid MECR pairs defined once on the uncorrected data
    valid = {}
    for t in TAUS:
        _, either = mecr_pairs(cache["uncorrected"][0]["bin"][t], ml)
        valid[t] = pair_mask & triu & (either >= 20)
    print("valid MECR pairs per tau:", {t: int(v.sum()) for t, v in valid.items()}, flush=True)

    def summarize(ii):
        out = {}
        l = lab[ii]
        for c in COND:
            reps = cache[c]
            vals = {}
            for r in reps:
                Xm, tot = r["Xm"][ii], r["tot"][ii]
                fmask = F[ii]
                fo, ow = (Xm * fmask).sum(1), (Xm * ~fmask).sum(1)
                v = {"foreign_frac_all": np.mean(fo / np.maximum(tot, 1e-9)), "own_sum_all": ow.sum()}
                for gi, g in enumerate(groups):
                    s = l == gi
                    v[f"foreign_frac|{g}"] = np.mean(fo[s] / np.maximum(tot[s], 1e-9)) if s.any() else np.nan
                    v[f"own_sum|{g}"] = ow[s].sum()
                for t in TAUS:
                    m, _ = mecr_pairs(r["bin"][t][ii], ml)
                    v[f"mecr|{t}"] = np.nanmean(m[valid[t]])
                v["wj"] = np.nanmean(weighted_jaccard(Xm[ii])[valid[TAU0]])
                for k, x in v.items():
                    vals.setdefault(k, []).append(x)
            for k, x in vals.items():
                out[(k, c)] = float(np.mean(x))
        for gi, g in [(None, "all")] + list(enumerate(groups)):
            sel = np.ones(len(ii), bool) if gi is None else (l == gi)
            for nm, (Rv, Dv) in REM.items():
                out[(f"rem_{nm}|{g}", "corrected")] = float(1 - Dv[ii][sel].sum() / max(Rv[ii][sel].sum(), 1e-9))
        return out

    print("point estimate + bootstrap", flush=True)
    full = summarize(np.arange(nB))
    boots = [summarize(rng.integers(0, nB, nB)) for _ in range(args.n_boot)]
    keys = list(full.keys())
    arr = {k: np.array([full[k]] + [b[k] for b in boots]) for k in keys}  # [0]=point

    rows = []
    for (k, c), v in arr.items():
        met, _, grp = k.partition("|")
        lo, hi = np.nanpercentile(v[1:], [2.5, 97.5])
        rows.append((level, met, grp or "all", c, v[0], lo, hi))
    tidy = pd.DataFrame(rows, columns=["level", "metric", "group", "condition", "value", "ci_lo", "ci_hi"])
    tidy.to_csv(OUT / f"metrics_{level}.csv", index=False)

    def pct(metric_key, c, base="uncorrected"):  # % change vs uncorrected, paired by bootstrap draw
        d = 100 * (arr[(metric_key, c)] / arr[(metric_key, base)] - 1)
        return d[0], *np.nanpercentile(d[1:], [2.5, 97.5])

    prow = []
    for k in sorted({k for (k, c) in arr if k.split("|")[0] in ("foreign_frac", "foreign_frac_all", "own_sum", "own_sum_all", "wj", "mecr")}):
        for c in ["corrected", "uniform", "permuted"]:
            d0, dlo, dhi = pct(k, c)
            prow.append((level, k, c, d0, dlo, dhi))
    pd.DataFrame(prow, columns=["level", "metric_key", "condition", "pct_change_vs_uncorrected", "ci_lo", "ci_hi"]).to_csv(OUT / f"paired_pct_change_{level}.csv", index=False)

    # per-cell scatter (corrected only)
    Xc, Xr = D[idxB][:, mcols], R[idxB][:, mcols]
    ff = lambda Xm, tot: (Xm * F).sum(1) / np.maximum(tot, 1e-9)
    d_cell = pd.DataFrame({
        "group": labels[idxB], "frac_removed": f_cell[idxB],
        "foreign_frac_uncorrected": ff(Xr, R[idxB].sum(1)), "foreign_frac_corrected": ff(Xc, D[idxB].sum(1)),
    })
    d_cell.to_csv(OUT / f"per_cell_{level}.csv", index=False)

    # lineage-pair MECR heatmap at tau0 (point estimates)
    def pair_mat(c):
        m = np.mean([weighted_jaccard(r["Xm"]) for r in cache[c]], 0)
        out = np.full((len(groups), len(groups)), np.nan)
        for i in range(len(groups)):
            for j in range(len(groups)):
                lo_, hi_ = min(i, j), max(i, j)
                sel = (ml[:, None] == lo_) & (ml[None, :] == hi_) & valid[TAU0]
                if i != j and sel.any():
                    out[i, j] = np.nanmean(m[sel])
        return out

    Mr, Mc, Mu = pair_mat("uncorrected"), pair_mat("corrected"), pair_mat("uniform")
    pd.DataFrame(Mc - Mu, index=groups, columns=groups).to_csv(OUT / f"weightedjaccard_delta_corrected_minus_uniform_{level}.csv")

    return dict(level=level, groups=groups, arr=arr, pct=pct, d_cell=d_cell, Mr=Mr, Mc=Mc, Mu=Mu, nB=nB, nM=nM)


def make_figure(res):
    level, groups, arr, pct = res["level"], res["groups"], res["arr"], res["pct"]
    fig = plt.figure(figsize=(26, 11.5))
    gs = fig.add_gridspec(2, 4, hspace=0.75, wspace=0.6)
    ctrl = ["corrected", "uniform", "permuted"]
    xs = ["all"] + groups
    xl = ["All cells"] + groups

    def ci(key, c):
        v = arr[(key, c)]
        return v[0], *np.nanpercentile(v[1:], [2.5, 97.5])

    def err(y, lo, hi):
        return [np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)]

    def grouped(ax, series, ylab, title, hline=None):
        n = len(series)
        wid = 0.8 / n
        for si, (name, color, fn) in enumerate(series):
            vals = [fn(g) for g in xs]
            y, lo, hi = (np.array(z) for z in zip(*vals))
            ax.bar(np.arange(len(xs)) + (si - (n - 1) / 2) * wid, y, wid, color=color, label=name, yerr=err(y, lo, hi),
                   error_kw=dict(lw=0.8, capsize=1.5))
        ax.axhline(0 if hline is None else hline, color="k", lw=0.6)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xl, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylab, fontsize=9); ax.set_title(title, fontsize=10, loc="left")

    def cond_bars(ax, key, ylab, title, fmt):
        for ci_, c in enumerate(COND):
            y, lo, hi = ci(key, c)
            ax.bar(ci_, y, 0.7, color=COL[c], yerr=err(np.array([y]), np.array([lo]), np.array([hi])), error_kw=dict(lw=0.9, capsize=3))
            ax.text(ci_, hi, fmt % y, ha="center", va="bottom", fontsize=8)
            if c != "uncorrected":
                ax.text(ci_, y / 2, "%+.1f%%" % pct(key, c)[0], ha="center", va="center", fontsize=9)
        ax.set_xticks(range(4)); ax.set_xticklabels(COND, rotation=20)
        lo_, hi_ = ax.get_ylim(); ax.set_ylim(0, hi_ * 1.08)
        ax.set_ylabel(ylab, fontsize=9); ax.set_title(title, fontsize=10, loc="left")

    ax = fig.add_subplot(gs[0, 0])
    grouped(ax, [(c, COL[c], (lambda g, c=c: pct("foreign_frac_all" if g == "all" else f"foreign_frac|{g}", c))) for c in ctrl],
            "% change in negative-marker fraction\nvs uncorrected (lower = purer)", "a  Negative-marker fraction\n(foreign-lineage markers / cell counts)")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    grouped(ax, [(c, COL[c], (lambda g, c=c: tuple(100 + np.array(pct("own_sum_all" if g == "all" else f"own_sum|{g}", c))))) for c in ctrl],
            "% own-marker signal retained\n(100 = nothing lost)", "b  Retention of own-lineage markers\n(sensitivity)", hline=100)

    ax = fig.add_subplot(gs[0, 2])
    RC = {"own": "#4c78a8", "foreign": "#e45756", "other": "#bab0ac"}
    grouped(ax, [(f"{nm} markers" if nm != "other" else "other genes", RC[nm],
                  (lambda g, nm=nm: tuple(100 * np.array(ci(f"rem_{nm}|{g}", "corrected"))))) for nm in ["own", "foreign", "other"]],
            "% of counts removed by the correction", "c  What gets removed?\n(removal rate by gene class)")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[0, 3])
    cond_bars(ax, "wj", "mean weighted Jaccard of marker pairs\nfrom different lineages (lower = purer)",
              "d  Continuous co-expression (soft counts)", "%.4f")

    ax = fig.add_subplot(gs[1, 0])
    cond_bars(ax, f"mecr|{TAU0}", "mean MECR (lower = purer)", f"e  Binary MECR (expressed >= {TAU0} count)", "%.4f")

    ax = fig.add_subplot(gs[1, 1])
    for c in COND:
        y = np.array([arr[(f"mecr|{t}", c)][0] for t in TAUS])
        lo = np.array([np.nanpercentile(arr[(f"mecr|{t}", c)][1:], 2.5) for t in TAUS])
        hi = np.array([np.nanpercentile(arr[(f"mecr|{t}", c)][1:], 97.5) for t in TAUS])
        ax.plot(TAUS, y, "-o", color=COL[c], label=c, ms=4)
        ax.fill_between(TAUS, lo, hi, color=COL[c], alpha=0.2, lw=0)
    ax.set_xlabel("counts needed to call a gene 'expressed'", fontsize=9)
    ax.set_ylabel("mean MECR (lower = purer)", fontsize=9)
    ax.set_title("f  MECR vs. detection threshold", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    M = res["Mc"] - res["Mu"]
    M = np.where(np.isnan(M), M.T, M)
    lim = np.nanmax(np.abs(M)) or 1
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups, fontsize=8)
    if len(groups) <= 9:
        for i in range(len(groups)):
            for j in range(len(groups)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("weighted Jaccard, corrected - uniform\n(blue = purer than matched control)", fontsize=8)
    ax.set_title("g  Gain per lineage pair beyond\nthe matched control", fontsize=10, loc="left")

    ax = fig.add_subplot(gs[1, 3])
    d = res["d_cell"]
    d["delta"] = d.foreign_frac_corrected - d.foreign_frac_uncorrected
    for g in groups:
        sub = d[d.group == g]
        ax.scatter(sub.frac_removed * 100, sub.delta * 100, s=5, alpha=0.5, label=g, color=ct_colors.get(g))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("% of the cell's counts removed", fontsize=9)
    ax.set_ylabel("change in negative-marker fraction\n(percentage points; < 0 = purer)", fontsize=9)
    ax.set_title("h  Per cell: more correction -> purer?", fontsize=10, loc="left")
    if level == "lineage":
        ax.legend(frameon=False, fontsize=7, markerscale=2, ncol=2)
    fig.suptitle(f"Expression purity after local-uRNA background correction ({level} level; markers selected on a held-out half; "
                 f"{res['nB']} evaluation cells, {res['nM']} markers; error bars = 95% bootstrap CI)", fontsize=11, y=0.99)
    fig.savefig(OUT / f"purity_evaluation_{level}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"purity_evaluation_{level}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_validation_figure(res):
    """Compact 2x2 figure with the four most informative validation panels (paired bootstrap CIs vs uncorrected)."""
    groups, arr, pct = res["groups"], res["arr"], res["pct"]
    xs = ["all"] + groups
    xl = ["All cells"] + groups
    ctrl = ["corrected", "uniform", "permuted"]
    err = lambda y, lo, hi: [np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)]
    ci = lambda key, c: (arr[(key, c)][0], *np.nanpercentile(arr[(key, c)][1:], [2.5, 97.5]))

    def grouped(ax, series, ylab, title, hline=0.0):
        n = len(series); wid = 0.8 / n
        for si, (name, color, fn) in enumerate(series):
            y, lo, hi = (np.array(z) for z in zip(*[fn(g) for g in xs]))
            ax.bar(np.arange(len(xs)) + (si - (n - 1) / 2) * wid, y, wid, color=color, label=name, yerr=err(y, lo, hi),
                   error_kw=dict(lw=0.8, capsize=1.5))
        ax.axhline(hline, color="k", lw=0.6)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xl, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(ylab, fontsize=10); ax.set_title(title, fontsize=11, loc="left")
        ax.spines[["top", "right"]].set_visible(False)

    fig, axs = plt.subplots(2, 2, figsize=(13, 10.5))
    plt.subplots_adjust(hspace=0.75, wspace=0.28)
    RC = {"own": "#4c78a8", "foreign": "#e45756", "other": "#bab0ac"}
    ax = axs[0, 0]
    grouped(ax, [(f"{nm} markers" if nm != "other" else "other genes", RC[nm],
                  (lambda g, nm=nm: tuple(100 * np.array(ci(f"rem_{nm}|{g}", "corrected"))))) for nm in ["own", "foreign", "other"]],
            "% of counts removed by the correction", "a  What the correction removes\n(by gene class; markers from a held-out half)")
    ax.legend(frameon=False, fontsize=9)
    ax = axs[0, 1]
    grouped(ax, [(c, COL[c], (lambda g, c=c: pct("foreign_frac_all" if g == "all" else f"foreign_frac|{g}", c))) for c in ctrl],
            "% change in negative-marker fraction\nvs uncorrected (lower = purer)",
            "b  Negative-marker fraction\n(counts from other lineages' markers / cell counts)")
    ax.legend(frameon=False, fontsize=9)
    ax = axs[1, 0]
    for i, c in enumerate(ctrl):
        y, lo, hi = pct("wj", c)
        ax.bar(i, y, 0.6, color=COL[c], yerr=err(np.array([y]), np.array([lo]), np.array([hi])), error_kw=dict(lw=1, capsize=4))
        ax.text(i, 0.15 if y >= 0 else y - 0.05, "%+.2f%%" % y, ha="center", va="top" if y < 0 else "bottom", fontsize=9)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(ctrl); ax.set_ylabel("% change in cross-lineage marker co-expression\nvs uncorrected (lower = purer)", fontsize=10)
    ax.set_title("c  Continuous co-expression of marker pairs\nfrom different lineages (weighted Jaccard)", fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax = axs[1, 1]
    grouped(ax, [(c, COL[c], (lambda g, c=c: tuple(100 + np.array(pct("own_sum_all" if g == "all" else f"own_sum|{g}", c))))) for c in ctrl],
            "% own-lineage marker signal retained\n(100 = nothing lost)", "d  Retention of own-lineage markers\n(sensitivity check)", hline=100)
    fig.suptitle("Validation of the local-uRNA background correction on msbrain 5k  (%d held-out cells, %d markers; "
                 "controls remove the same signal without uRNA information; error bars: 95%% paired bootstrap CI)" % (res["nB"], res["nM"]),
                 fontsize=10, y=0.99)
    fig.savefig(OUT / "purity_validation_main_panels.pdf", bbox_inches="tight")
    fig.savefig(OUT / "purity_validation_main_panels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


for level, labels in [("lineage", np.array([LINEAGE[x] for x in ct])), ("cell_type", ct)]:
    if level not in args.levels.split(","):
        continue
    res = run_level(level, labels)
    make_figure(res)
    if level == "lineage":
        make_validation_figure(res)
    t = pd.read_csv(OUT / f"metrics_{level}.csv")
    print(t[(t.group == "all") | (t.metric == "mecr")].to_string(), flush=True)
print("done", flush=True)
