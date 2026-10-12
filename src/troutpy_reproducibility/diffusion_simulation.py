"""Simulations used to validate the Rayleigh-based diffusion test (ED Fig. 5C-E).

Cells are placed in a 3D volume, a fraction of each cell's transcripts is displaced
according to a movement model (``brownian``, ``brownian_2d``, ``correlated``,
``levy`` or ``uniform``), a thin optical z-slice is taken, and each gene's
transcript-to-nearest-cell distances are tested against a fitted Rayleigh
distribution. See the Methods section "Simulation of RNA movement models and
evaluation of Rayleigh-based diffusion classification" and
``notebooks/SUPP2_rayleigh_distribution_simulation/``.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d import Axes3D

# =====================================================================
# 1. CORE SIMULATION ENGINES
# =====================================================================

def get_displacement(count, scale, mode='brownian', space_size=(800, 800, 50), cell_pos=(0,0,0), **kwargs):
    """Sample displacements for ``count`` transcripts released by one cell.

    Parameters
    ----------
    count
        Number of transcripts to displace.
    scale
        Displacement scale (sigma), in the same units as ``space_size``.
    mode
        Movement model:

        - ``"brownian"``: isotropic 3D Gaussian jump, N(0, scale^2) per axis.
        - ``"brownian_2d"``: Gaussian jump in x/y only (dz = 0).
        - ``"correlated"``: persistent random walk of 5 sub-steps, persistence
          ``kwargs["persistence"]`` (default 0.8), each step of length scale/sqrt(5).
        - ``"levy"``: Levy flight with step length ``scale * U**(-1/alpha)``
          (``kwargs["alpha"]``, default 1.5) in a random 3D direction.
        - ``"uniform"``: no localized movement; positions drawn uniformly in the
          whole volume (null model).
    space_size
        ``(width, height, depth)`` of the simulated volume.
    cell_pos
        Source cell position ``(x, y, z)`` (not used by the displacement models,
        kept for a uniform call signature).
    **kwargs
        Model-specific parameters (``persistence``, ``alpha``).

    Returns
    -------
    Three arrays ``(dx, dy, dz)`` of length ``count``. For ``mode="uniform"`` these
    are absolute coordinates rather than offsets (the caller handles this).
    Unknown modes return zero displacement.
    """
    width, height, depth = space_size
    cx, cy, cz = cell_pos

    if mode == 'uniform':
        # TRULY UNIFORM: Pick a random coordinate anywhere in the ACTUAL 3D block
        tx = np.random.uniform(0, width, count)
        ty = np.random.uniform(0, height, count)
        tz = np.random.uniform(0, depth, count)
        
        # Return the DELTA. When the main function does (cx + dx), 
        # the result is guaranteed to be exactly (tx), which is inside the box.
        return (tx, ty, tz)

    elif mode == 'brownian':
        return [np.random.normal(0, scale, count) for _ in range(3)]

    elif mode == 'brownian_2d':
        # 2D Diffusion: Random X and Y, but NO Z displacement
        dx = np.random.normal(0, scale, count)
        dy = np.random.normal(0, scale, count)
        dz = np.zeros(count)
        return (dx, dy, dz)

    elif mode == 'correlated':
        rho = kwargs.get('persistence', 0.8)
        steps = 5
        dx, dy, dz = np.zeros(count), np.zeros(count), np.zeros(count)
        for i in range(count):
            curr_dir = np.random.normal(0, 1, 3); curr_dir /= np.linalg.norm(curr_dir)
            pos = np.zeros(3)
            for _ in range(steps):
                new_dir = np.random.normal(0, 1, 3); new_dir /= np.linalg.norm(new_dir)
                curr_dir = rho * curr_dir + (1 - rho) * new_dir
                curr_dir /= np.linalg.norm(curr_dir)
                pos += curr_dir * (scale / np.sqrt(steps))
            dx[i], dy[i], dz[i] = pos
        return dx, dy, dz

    elif mode == 'levy':
        alpha = kwargs.get('alpha', 1.5)
        lengths = scale * (np.random.random(count)**(-1/alpha))
        dirs = np.random.normal(0, 1, (count, 3))
        dirs /= np.linalg.norm(dirs, axis=1)[:, np.newaxis]
        return (dirs[:, 0] * lengths, dirs[:, 1] * lengths, dirs[:, 2] * lengths)

    return [np.zeros(count) for _ in range(3)]


# =====================================================================
# 2. STATISTICAL ASSESSMENT FUNCTIONS
# =====================================================================

def assess_diffusion_nearest_neighbor(
    exRNA_df: pd.DataFrame,
    cells_df: pd.DataFrame,
    gene_col: str = "gene",
    x_col: str = "x",
    y_col: str = "y",
    ks_thresh: float = 0.05
) -> pd.DataFrame:
    """Test, per gene, whether transcript-to-nearest-cell distances follow a Rayleigh law.

    Each transcript is assigned to its nearest cell in 2D; for every gene with at
    least 15 transcripts a Rayleigh distribution (location fixed at 0) is fitted by
    maximum likelihood and compared with the observed distances using a
    Kolmogorov-Smirnov test. An Anderson-Darling test on ``d^2 / (2 sigma^2)``
    (exponential under the Rayleigh hypothesis) is reported as well.

    Parameters
    ----------
    exRNA_df
        Unassigned transcripts with ``gene_col``, ``x_col`` and ``y_col`` columns.
    cells_df
        Cell positions with ``x_col``, ``y_col`` and ``"cell_id"`` columns.
    gene_col, x_col, y_col
        Column names.
    ks_thresh
        Not used here (kept for API symmetry); the caller thresholds ``ks_pval``.

    Returns
    -------
    DataFrame indexed by gene with ``ks_stat``, ``ks_pval``, ``-log_ks_pval``,
    ``ad_stat``, ``ad_critical_values``, ``ad_is_significant`` (5% level),
    ``log_likelihood``, ``mean_displacement``, ``estimated_sigma`` and
    ``n_transcripts``. Empty if no gene has enough transcripts.
    """
    if exRNA_df.empty or cells_df.empty:
        return pd.DataFrame()

    # 1. 2D Spatial Tree for Nearest Neighbor Assignment
    cell_coords = cells_df[[x_col, y_col]].to_numpy()
    tree = cKDTree(cell_coords)

    # 2. Query nearest cell
    rna_coords = exRNA_df[[x_col, y_col]].to_numpy()
    distances, indices = tree.query(rna_coords, k=1) 

    merged = exRNA_df.copy()
    merged["assigned_cell"] = cells_df.iloc[indices]["cell_id"].values
    merged["distance"] = distances 

    results = []
    for gene, group in merged.groupby(gene_col):
        dist_vals = group["distance"].dropna().values
        
        if len(dist_vals) < 15: 
            continue

        # --- THE RAYLEIGH FIT ---
        param = stats.rayleigh.fit(dist_vals, floc=0) 
        sigma_est = param[1]
        
        # --- KS TEST ---
        ks_stat, ks_pval = stats.kstest(dist_vals, "rayleigh", args=param)
        
        # --- ANDERSON-DARLING TEST ---
        transformed_dist = (dist_vals**2) / (2 * (sigma_est**2))
        ad_result = stats.anderson(transformed_dist, dist='expon')
        
        # --- LIKELIHOOD ---
        log_lik = np.sum(stats.rayleigh.logpdf(dist_vals, *param))
        
        results.append({
            "gene": gene,
            "ks_stat": ks_stat,
            "ks_pval": ks_pval,
            "ad_stat": ad_result.statistic,
            "ad_critical_values": ad_result.critical_values, 
            "log_likelihood": log_lik,
            "mean_displacement": np.mean(dist_vals),
            "estimated_sigma": sigma_est,
            "n_transcripts": len(dist_vals)
        })

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results).set_index("gene")
    results_df["-log_ks_pval"] = -np.log(results_df["ks_pval"].clip(lower=1e-300))
    
    # Flag if AD statistic exceeds the 5% critical value (index 2)
    results_df["ad_is_significant"] = results_df.apply(
        lambda x: x["ad_stat"] > x["ad_critical_values"][2], axis=1
    )
    
    return results_df


def assess_diffusion_by_ratio(
    exRNA_df: pd.DataFrame, 
    cells_df: pd.DataFrame, 
    gene_col: str = "gene", 
    x_col: str = "x", 
    y_col: str = "y",
    target_cv: float = 0.5227  # Theoretical CV of Rayleigh
) -> pd.DataFrame:
    """Score how close each gene's distance distribution is to a Rayleigh shape via its CV.

    Alternative, faster diagnostic to :func:`assess_diffusion_nearest_neighbor`:
    the coefficient of variation of a Rayleigh distribution is constant
    (``sqrt(4/pi - 1)`` ~ 0.5227), so genes whose transcript-to-nearest-cell
    distances have a CV close to it are scored as diffusion-like.

    Parameters
    ----------
    exRNA_df, cells_df, gene_col, x_col, y_col
        As in :func:`assess_diffusion_nearest_neighbor`. Genes with fewer than 20
        transcripts are skipped.
    target_cv
        Theoretical CV of the Rayleigh distribution.

    Returns
    -------
    DataFrame indexed by gene with ``mean_dist``, ``cv_observed``,
    ``brownian_score`` (``exp(-|cv - target_cv| / 0.1)``, 1 = perfect match) and
    ``n_transcripts``.
    """
    if exRNA_df.empty or cells_df.empty:
        return pd.DataFrame()

    # 1. 2D Nearest Neighbor Assignment
    cell_coords = cells_df[[x_col, y_col]].to_numpy()
    rna_coords = exRNA_df[[x_col, y_col]].to_numpy()
    tree = cKDTree(cell_coords)
    distances, _ = tree.query(rna_coords, k=1)

    merged = exRNA_df.copy()
    merged["distance"] = distances

    results = []
    for gene, group in merged.groupby(gene_col):
        dist_vals = group["distance"].values
        if len(dist_vals) < 20: continue

        # Calculate empirical stats
        mu = np.mean(dist_vals)
        std = np.std(dist_vals)
        cv = std / mu if mu != 0 else 0
        
        # Brownian Score: How close is the CV to 0.523?
        brownian_score = np.exp(-abs(cv - target_cv) / 0.1)
        
        results.append({
            "gene": gene,
            "mean_dist": mu,
            "cv_observed": cv,
            "brownian_score": brownian_score,
            "n_transcripts": len(dist_vals)
        })

    return pd.DataFrame(results).set_index("gene")


# =====================================================================
# 3. MAIN WORKFLOW PIPELINE
# =====================================================================

def run_diffusion_simulation_3d(n_cells=200, n_genes=50, 
                                space_size=(800, 800, 50), 
                                scale_diffusion=10, 
                                z_slice_thickness=10,
                                mode='brownian',
                                spatial_structure='random',
                                exRNA_factor=0.1, ks_thresh=0.05,
                                return_full=False, plot=False, max_points=5000,
                                asessment_method='ratio',save_path=None,
                                **mode_kwargs):
    """Run one simulation: place cells, release and move transcripts, slice, and test.

    Parameters
    ----------
    n_cells, n_genes
        Number of simulated cells and genes.
    space_size
        ``(width, height, depth)`` of the volume.
    scale_diffusion
        Displacement scale passed to :func:`get_displacement`.
    z_slice_thickness
        Thickness of the imaged optical section, centred at mid-depth.
    mode
        Movement model (see :func:`get_displacement`).
    spatial_structure
        ``"random"`` (uniform cell positions, Poisson(20) counts) or
        ``"clustered"`` (5 niches, each with its own Gamma-distributed mean
        expression profile).
    exRNA_factor
        Fraction of each cell's counts released as unassigned RNA.
    ks_thresh
        Significance level; a gene is "Rayleigh-consistent" if ``ks_pval > ks_thresh``.
    return_full
        If True, return the full result dictionary; otherwise only ``prop_global``.
    plot, max_points, save_path
        Plot the simulation (see :func:`plot_simulation_results`), subsampling to
        ``max_points`` transcripts, optionally saving it as PDF.
    asessment_method
        ``"ratio"`` (:func:`assess_diffusion_by_ratio`, consistent if
        ``brownian_score > 0.8``) or anything else for the KS-based
        :func:`assess_diffusion_nearest_neighbor` used in the manuscript.
    **mode_kwargs
        Extra arguments for the movement model.

    Returns
    -------
    The proportion of Rayleigh-consistent genes among transcripts inside the slice
    (``prop_global``), or, with ``return_full=True``, a dict with ``prop_global``,
    ``stats_global`` (all transcripts in the slice) and ``prop_in_plane``,
    ``stats_in_plane`` (only transcripts whose source cell is also in the slice).
    """
    genes = [f"Gene_{i}" for i in range(n_genes)]
    width, height, depth = space_size
    z_center = depth / 2
    z_min, z_max = z_center - (z_slice_thickness/2), z_center + (z_slice_thickness/2)

    # --- 1. Simulate Cells ---
    if spatial_structure == 'clustered':
        n_niches = 5
        niche_centers = np.random.uniform(low=[100,100,10], high=[width-100, height-100, depth-10], size=(n_niches, 3))
        cell_niche_indices = np.random.choice(range(n_niches), size=n_cells)
        offsets = np.random.normal(0, 50, size=(n_cells, 3))
        cell_pos = niche_centers[cell_niche_indices] + offsets
        cell_pos = np.clip(cell_pos, [0,0,0], [width, height, depth])
        niche_profiles = np.random.gamma(2, 10, size=(n_niches, n_genes))
        expression = np.zeros((n_cells, n_genes), dtype=int)
        for i in range(n_cells):
            expression[i, :] = np.random.poisson(lam=niche_profiles[cell_niche_indices[i]])
    else:
        cell_pos = np.random.uniform(low=[0,0,0], high=[width, height, depth], size=(n_cells, 3))
        expression = np.random.poisson(lam=20, size=(n_cells, n_genes))

    cells_df = pd.DataFrame(cell_pos, columns=["x", "y", "z"])
    cells_df["cell_id"] = [f"cell_{i}" for i in range(n_cells)]
    for i, gene in enumerate(genes): 
        cells_df[gene] = expression[:, i]

    # --- 2. Simulate Displacement ---
    exRNA_data = []
    for idx, cell in cells_df.iterrows():
        cx, cy, cz = cell["x"], cell["y"], cell["z"]
        for gene in genes:
            count = int(cell[gene] * exRNA_factor)
            if count <= 0: continue
            cells_df.at[idx, gene] -= count # Subtraction
            
            dx, dy, dz = get_displacement(count, scale_diffusion, mode=mode, 
                                          space_size=space_size, cell_pos=(cx, cy, cz), 
                                          **mode_kwargs)
            
            # Record transcript with its TRUE source_cell ID and its TRUE source_z
            for x_off, y_off, z_off in zip(dx, dy, dz):
                tx = (x_off if mode in ['uniform', 'uniform_2d'] else cx + x_off)
                ty = (y_off if mode in ['uniform', 'uniform_2d'] else cy + y_off)
                tz = (z_off if mode in ['uniform', 'uniform_2d'] else cz + z_off)
                
                exRNA_data.append({
                    "x": tx, "y": ty, "z": tz, 
                    "gene": gene, 
                    "source_cell": cell["cell_id"],
                    "source_z": cz  # Keep track of where the parent was
                })
            
    exRNA_df = pd.DataFrame(exRNA_data)

    # --- 3. Dual Slicing and Assessment ---
    # Global Slice: All transcripts found in the Z-plane
    sliced_exRNA_global = exRNA_df[(exRNA_df['z'] >= z_min) & (exRNA_df['z'] <= z_max)].copy()
    
    # In-Plane Slice: Transcripts in the Z-plane WHOSE SOURCE was also in the Z-plane
    sliced_exRNA_in_plane = sliced_exRNA_global[
        (sliced_exRNA_global['source_z'] >= z_min) & (sliced_exRNA_global['source_z'] <= z_max)
    ].copy()

    sliced_cells = cells_df[(cells_df['z'] >= z_min) & (cells_df['z'] <= z_max)].copy()

    def run_eval(df, method):
        if method == 'ratio':
            stats_df = assess_diffusion_by_ratio(df, sliced_cells)
            prop = (stats_df['brownian_score'] > 0.8).mean() if not stats_df.empty else 0.0
        else:
            stats_df = assess_diffusion_nearest_neighbor(df, sliced_cells, ks_thresh=ks_thresh)
            prop = (stats_df['ks_pval'] > ks_thresh).mean() if not stats_df.empty else 0.0
        return prop, stats_df

    print('Assessing Global Transcripts...')
    prop_global, stats_global = run_eval(sliced_exRNA_global, asessment_method)
    
    print('Assessing In-Plane Only Transcripts...')
    prop_in_plane, stats_in_plane = run_eval(sliced_exRNA_in_plane, asessment_method)

    if plot:
        # Pass the save_path down to the plotting module
        plot_simulation_results(cells_df, exRNA_df, sliced_cells, sliced_exRNA_global, 
                                space_size, (z_min, z_max), mode, prop_global, max_points,
                                save_path=save_path) 
        
    # Return results for both categories
    result = {
        "prop_global": prop_global,
        "stats_global": stats_global,
        "prop_in_plane": prop_in_plane,
        "stats_in_plane": stats_in_plane
    }
    
    return result if return_full else prop_global


# =====================================================================
# 4. VISUALIZATION MODULE
# =====================================================================

def plot_simulation_results(cells_df, exRNA_df, sliced_cells, sliced_exRNA, 
                            space_size, z_range, mode, fail_rate, max_points=5000,
                            save_path=None): # <-- Added parameter
    """Plot a simulation: 3D volume, x-z cross-section and the final 2D projection.

    Parameters
    ----------
    cells_df, exRNA_df
        All simulated cells and released transcripts (with ``x``, ``y``, ``z``).
    sliced_cells, sliced_exRNA
        Cells and transcripts inside the optical slice.
    space_size
        ``(width, height, depth)`` of the volume.
    z_range
        ``(z_min, z_max)`` of the slice.
    mode
        Movement model name, used in the title.
    fail_rate
        Proportion shown in the third panel's title.
    max_points
        Maximum number of transcripts drawn per panel.
    save_path
        If given, the figure is saved there as PDF.
    """
    width, height, depth = space_size
    z_min, z_max = z_range
    
    fig = plt.figure(figsize=(20, 7))
    
    # --- Panel 1: 3D Global Volume ---
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(cells_df["x"], cells_df["y"], cells_df["z"], c='black', s=15, alpha=0.6)
    
    if not exRNA_df.empty:
        samp_rna = exRNA_df.sample(min(len(exRNA_df), max_points))
        ax1.scatter(samp_rna["x"], samp_rna["y"], samp_rna["z"], c='blue', s=4, alpha=0.05)
    
    xx, yy = np.meshgrid([0, width], [0, height])
    ax1.plot_surface(xx, yy, np.full_like(xx, z_min), alpha=0.2, color='red')
    ax1.plot_surface(xx, yy, np.full_like(xx, z_max), alpha=0.2, color='red')
    
    ax1.set_xlim(0, width)
    ax1.set_ylim(0, height)
    ax1.set_zlim(0, depth)
    ax1.set_title(f"1. 3D Volume ({mode})")

    # --- Panel 2: X-Z Side View ---
    ax2 = fig.add_subplot(132)
    ax2.scatter(exRNA_df["x"], exRNA_df["z"], c='blue', s=4, alpha=0.1)
    ax2.scatter(cells_df["x"], cells_df["z"], c='black', s=10)
    ax2.axhspan(z_min, z_max, color='red', alpha=0.3, label="Microscope Slice")
    
    ax2.set_xlim(0, width)
    ax2.set_ylim(0, depth)
    ax2.set_title("2. Z-Axis Cross Section")
    ax2.set_ylabel("Z (microns)")

    # --- Panel 3: Final 2D Projection ---
    ax3 = fig.add_subplot(133)
    ax3.scatter(sliced_cells["x"], sliced_cells["y"], c="black", s=30, label="Visible Cells")
    
    if not sliced_exRNA.empty:
        plot_rna = sliced_exRNA.sample(min(len(sliced_exRNA), max_points))
        gene_colors = pd.factorize(plot_rna["gene"])[0]
        ax3.scatter(plot_rna["x"], plot_rna["y"], c=gene_colors, 
                    cmap="tab20", s=10, alpha=0.8, label="Captured exRNA")
    
    ax3.set_xlim(0, width)
    ax3.set_ylim(0, height)
    ax3.set_title(f"3. Final 2D Projection\nRayleigh Fail Rate: {fail_rate:.2%}")
    
    plt.tight_layout()
    
    # --- SAVE LOGIC BLOCK ---
    if save_path is not None:
        # Convert to string path if it's a Path object
        plt.savefig(str(save_path), format='pdf', bbox_inches='tight', dpi=300)
        print(f"Simulation visualization successfully saved to: {save_path}")
        
    plt.show()