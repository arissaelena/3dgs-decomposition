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
        use_color_correction: bool=False,
        use_down_weight: bool=False,
        use_gaussian_average: bool=False
):
    args = [sys.executable, os.path.join(repo_root, 'splat/train_baseline_copy.py'), '-s', source_path, '-m', output_dir, '-a', albedos_path, '--depths', depth_paths, '--images', images_folder, '--albedo_iterations', str(iterations), '--shading_iterations', str(iterations), '--residual_iterations', str(iterations)]
    if use_color_correction:
        args.append('--use_color_correction')
    if use_down_weight:
        args.append('--use_down_weight')
    if use_gaussian_average:
        args.append('--use_gaussian_average')
    if eval:
        args.append('--eval')
    subprocess.run(args, check=True)

def render_test_views(source_path: str, output_dir: str, iterations: int=10_000):
    args = [sys.executable, os.path.join(intr_decomp_path, 'evaluate.py'), '-s', source_path, '-m', output_dir, '--iteration', str(iterations)]
    subprocess.run(args, check=True)

def align_depth_scale(scene_dir: str, depths_dir: str):
    args = [sys.executable, os.path.join(intr_decomp_path, 'scripts/make_depth_scale.py'), '--base_dir', scene_dir, '--depths_dir', depths_dir]
    subprocess.run(args, check=True)
