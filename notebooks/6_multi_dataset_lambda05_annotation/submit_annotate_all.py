#!/usr/bin/env python3
"""Generate a per-dataset .sbatch (from run_annotate.sbatch.template) and submit it for
every dataset in dataset_registry.NEEDS_GENERAL_ANNOTATION, skipping any already listed
in SKIP (e.g. ones already run interactively as a validation test).
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dataset_registry import NEEDS_GENERAL_ANNOTATION

HERE = Path(__file__).parent
TEMPLATE = (HERE / "run_annotate.sbatch.template").read_text()
SKIP = {"cosmx_hgsoc"}  # already run interactively as a validation test

(HERE / "logs").mkdir(exist_ok=True)
(HERE / "sbatch_generated").mkdir(exist_ok=True)

submitted = {}
for ds in NEEDS_GENERAL_ANNOTATION:
    if ds.name in SKIP:
        print(f"skipping {ds.name} (already done)")
        continue
    sbatch_text = TEMPLATE.replace("{{DATASET}}", ds.name)
    sbatch_path = HERE / "sbatch_generated" / f"annot_{ds.name}.sbatch"
    sbatch_path.write_text(sbatch_text)
    result = subprocess.run(["sbatch", "--parsable", str(sbatch_path)], capture_output=True, text=True, check=True)
    job_id = result.stdout.strip()
    submitted[ds.name] = job_id
    print(f"{ds.name}: submitted as job {job_id}")

print()
print("All job IDs:", " ".join(submitted.values()))
