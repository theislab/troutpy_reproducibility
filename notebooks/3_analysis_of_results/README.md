# 3. Cross-dataset analysis

Notebooks that combine the pipeline outputs of all datasets (and segmentations) into the
summary figures. Each notebook's header lists the panels it makes; see the figure table in
the main README. Notebooks 9 → 10 → 11 → 12 build on each other (noise, diffusion, full
classification, cross-segmentation consistency). Long notebooks were executed on SLURM with
`execute_nb11.sbatch` / `12_run_notebook.sbatch`.

- `control_probes/`: control-probe signal by preservation method and chemistry (ED Fig. 5A).
- `morphology/`: morphology metrics for the CosMx/MERSCOPE datasets whose segmentation is
  stored only as polygons (input to notebook 13).
