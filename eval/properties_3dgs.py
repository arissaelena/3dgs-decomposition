from argparse import ArgumentParser
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root, "intr_decomp_3dgs"))

import numpy as np
import torch
import cv2
from arguments import ModelParams, PipelineParams
from intr_decomp_3dgs.gaussian_renderer import render
from intr_decomp_3dgs.scene import GaussianModel, Scene
from scipy.ndimage import distance_transform_edt


@torch.no_grad()
def export_3dgs_depth(src_path: str, model_path: str, out_depth_dir: str, iter: int = 10000) -> None:
    os.makedirs(out_depth_dir, exist_ok=True)

    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    raw_args = parser.parse_args([])
    args = lp.extract(raw_args)
    pipe = pp.extract(raw_args)

    args.source_path = src_path
    args.model_path = model_path
    args.eval = False

    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians)
    gaussians.load_ply(os.path.join(model_path, "point_cloud", f"iteration_{iter}", "albedo.ply"))

    # because otherwise they dont match
    if gaussians._features_rest.ndim >= 2 and gaussians._features_rest.shape[1] > 0:
        n_rest = gaussians._features_rest.shape[1]
        gaussians.active_sh_degree = int(round((n_rest + 1) ** 0.5)) - 1
    else:
        gaussians.active_sh_degree = 0
        gaussians.max_sh_degree = 0

    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    for cam in scene.getTrainCameras():
        # trying to make them less noisy and patchy
        out = render(cam, gaussians, pipe, bg)
        inv_d = out["depth"].squeeze().detach().cpu().numpy().astype(np.float32)
        alpha = out["alpha_mask"].squeeze().detach().cpu().numpy().astype(np.float32)

        valid = (inv_d > 1e-6) & (alpha > 0.2)
        inv_d_norm = np.divide(inv_d, alpha, out=np.zeros_like(inv_d), where=valid)
        depth = np.divide(1.0, inv_d_norm, out=np.zeros_like(inv_d), where=valid)

        if np.any(valid):
            # maybe invalid holes bleed into valid surfaces
            nearest_idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
            depth_filled = depth[tuple(nearest_idx)]

            # try removing splat spikes
            depth_filled = cv2.medianBlur(depth_filled, 5)
            log_d = np.log(np.maximum(depth_filled, 1e-4)).astype(np.float32)
            # try smoothing jitter
            log_d = cv2.bilateralFilter(log_d, d=7, sigmaColor=0.06, sigmaSpace=7)
            depth = np.where(valid, np.exp(log_d), 0.0).astype(np.float32)

        stem = os.path.splitext(cam.image_name)[0]
        np.save(os.path.join(out_depth_dir, f"{stem}.npy"), depth)

    del scene, gaussians
    torch.cuda.empty_cache()