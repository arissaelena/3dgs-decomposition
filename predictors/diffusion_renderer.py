# function for processing the image sets with diffusion renderer to get normalised predicted values

import subprocess
import tempfile
import os
import numpy as np

def diffusion_renderer_predict(image_paths: list[str], dataset: str, scene: str):

    save_dir = os.path.abspath(f'diffusion_renderer/{dataset}/{scene}/')
    os.makedirs(save_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dr_input_") as input_dir:
        frames_dir = os.path.join(input_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        for frame_idx, img_path in enumerate(image_paths):
            ext = os.path.splitext(img_path)[1]
            os.symlink(
                os.path.abspath(img_path),
                os.path.join(frames_dir, f"frame_{frame_idx:04d}{ext}")
            )
    
        args = ['python', 'inference_svd_rgbx.py', '--config', 'configs/rgbx_inference.yaml', f'inference_input_dir={input_dir}', f'inference_save_dir={save_dir}', 'inference_n_frames=12', 'inference_n_steps=20', "model_passes=['diffuse_albedo']", 'inference_res=[512,512]', 'chunk_mode=all']
        subprocess.run(args, cwd="diffusion-renderer", check=True)
