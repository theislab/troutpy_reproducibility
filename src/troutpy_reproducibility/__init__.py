"""Helper code used to reproduce the analyses of the troutpy manuscript.

This package only holds analysis-specific helpers; the uRNA methods themselves live
in `troutpy <https://github.com/theislab/troutpy>`_. Submodules are imported
explicitly by the notebooks and scripts (e.g.
``from troutpy_reproducibility.batch_running import clustering_pipeline``) so that a
missing optional dependency of one module does not break the others.

Submodules
----------
roi_processing
    Region-of-interest selection, plotting and per-ROI metadata.
batch_running
    Clustering and environment-estimation helpers used by the uRNA pipelines.
downstream
    Transcript-category summaries shared by the cross-dataset analysis notebooks.
ora_analysis
    Vesiclepedia over-representation and SynGO enrichment of uRNA structure types.
plotting
    Plotting helpers shared by several notebooks.
multimodal
    Staining-positivity thresholds and gene/pixel enrichment for the multimodal analysis.
diffusion_simulation
    Simulations used to validate the Rayleigh-based diffusion test.
utils
    Small utilities.
"""

__version__ = "1.0.0"
