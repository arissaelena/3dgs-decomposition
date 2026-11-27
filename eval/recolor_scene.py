# show a recolouring demo on a real scene
import os
import numpy as np
import cv2
import torch
import sys
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from pathlib import Path
from shutil import copy
from tqdm import tqdm

repo_root = Path(__file__).resolve().parent.parent
intr_decomp_path = repo_root / "intr_decomp_3dgs"
if str(intr_decomp_path) not in sys.path:
    sys.path.insert(0, str(intr_decomp_path))

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from scene.dataset_readers import get_dataset_reader
from utils.general_utils import inverse_sigmoid

class Dummy(object):
    pass

# experiment_folder is the path to the scene in the baseline_train folder (the trained splats for our datasets)
def create_recoloring_demo(model_path: str, source_path: str, iteration, background, experiment_folder, mode, run_name: str='test', ext: str='.png'):
    render_path = os.path.join(model_path, experiment_folder, run_name, "ours_{}".format(iteration), "full", "renders")
    gts_path = os.path.join(model_path, experiment_folder, run_name, "ours_{}".format(iteration), "full", "gt")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    pipe = Dummy()
    pipe.debug = False
    pipe.antialiasing = False
    pipe.compute_cov3D_python = False
    pipe.convert_SHs_python = False

    albedo_ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", experiment_folder, "albedo.ply")
    shading_ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", experiment_folder, "shading.ply")
    residual_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", experiment_folder, "residual.safetensors")
    
    albedo_gaussians = GaussianModel(0)
    albedo_gaussians.use_color_activation = True
    albedo_gaussians.load_ply(albedo_ply_path)
    albedo_gaussians = apply_recoloring(albedo_gaussians, mode)
    
    shading_gaussians = GaussianModel(0)
    shading_gaussians.load_ply(shading_ply_path)

    residual_gaussians = GaussianModel(sh_degree=3, grayscale_dc=True)
    residual_gaussians.load_safetensors(residual_path)

    convert_images_to_linear = ext == ".png"

    dataset_reader = get_dataset_reader(source_path, images_dir="images", depths_dir=None, eval=True, llffhold=8, convert_images_to_linear=convert_images_to_linear)
    scene_info = dataset_reader.read()

    views = scene_info.test_cameras
    gt_source_dir = os.path.join(source_path, "images")

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rend_albedo = render(view, albedo_gaussians, pipe, background, colors_activation=True)["render"]
        rend_shading = render(view, shading_gaussians, pipe, background, colors_activation=False)["render"]
        rend_residual = render(view, residual_gaussians, pipe, background, colors_activation=False, enable_shs = True)["render"]

        rendering = (rend_albedo * rend_shading) + rend_residual

        # gt = view.original_image[0:3, :, :] 
        gt_source_path = os.path.join(gt_source_dir, view.image_name + ext)
        render_out_path = os.path.join(render_path, '{0:05d}'.format(idx) + ext)
        
        rendering = rendering.detach().permute(1, 2, 0).cpu().numpy()
        if ext == ".png":
            rendering = (rendering * 255.0).astype(np.uint8)

        cv2.imwrite(render_out_path, rendering[:, :, ::-1]) # cv2 BGR convention but our rendering is in RGB

        copy(gt_source_path, os.path.join(gts_path, '{0:05d}'.format(idx) + ext))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ext))
    
def apply_recoloring(albedo_gaussians, mode): # mode = ("colour_shift", (initial, final)) | ("hue_shift", shift_amount)
    rgb = albedo_gaussians.get_features_dc # (values in [0, 1], shaped as (N, 1, 3))
    new_rgb = rgb.clone()
    if mode[0] == "colour_shift":
        initial, final = mode[1]
        start_val_r, start_val_g, start_val_b = max(0.0, (initial[0] - 40) / 255), max(0.0, (initial[1] - 40) / 255), max(0.0, (initial[2] - 40) / 255)
        end_val_r, end_val_g, end_val_b = min(1.0, (initial[0] + 40) / 255), min(1.0, (initial[1] + 40) / 255), min(1.0, (initial[2] + 40) / 255)
        mask = (rgb[:, 0, 0] > start_val_r) & (rgb[:, 0, 0] < end_val_r) & (rgb[:, 0, 1] > start_val_g) & (rgb[:, 0, 1] < end_val_g) & (rgb[:, 0, 2] > start_val_b) & (rgb[:, 0, 2] < end_val_b)

        target_r, target_g, target_b = final[0] / 255, final[1] / 255, final[2] / 255
        new_rgb[mask, 0, :] = torch.tensor([target_r, target_g, target_b], device=rgb.device)
    else:
        shift_amount = mode[1]
        rgb_np = rgb[:, 0, :].detach().cpu().numpy()
        hsv = rgb_to_hsv(rgb_np)
        hsv[:, 0] = (hsv[:, 0] + shift_amount) % 1.0
        shifted_rgb_np = hsv_to_rgb(hsv)
        new_rgb = torch.from_numpy(shifted_rgb_np).to(device=rgb.device, dtype=rgb.dtype).unsqueeze(1)
    new_vals = inverse_sigmoid(new_rgb.clamp(1e-4, 1.0 - 1e-4))
    albedo_gaussians._features_dc = torch.nn.Parameter(new_vals)
    return albedo_gaussians
