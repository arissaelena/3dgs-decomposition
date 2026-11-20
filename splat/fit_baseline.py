# fit albedo, freeze, then fit shading and residual sets
import subprocess
import tempfile
import os
import numpy as np
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
intr_decomp_path = repo_root / "intr_decomp_3dgs"
if str(intr_decomp_path) not in sys.path:
    sys.path.insert(0, str(intr_decomp_path))

def run_baseline_train(
        images_folder: str, # path to /images folder inside source_path
        source_path: str, # parent directory to /sparse/0/ and /images folder
        albedos_path: str, # path to 2D albedo maps for the predictor used
        depth_paths: str, # path to folder of 16-bit depth pngs
        output_dir: str, # path to where we want the output
        eval: bool=True,
        iterations: int=10_000,
):
    pass

def render_test_views():
    pass

def align_depth_scale(scene_dir: str, depths_dir: str):
    args = [sys.executable, os.path.join(intr_decomp_path, 'scripts/make_depth_scale.py'), '--base_dir', scene_dir, '--depths_dir', depths_dir]
    subprocess.run(args)
