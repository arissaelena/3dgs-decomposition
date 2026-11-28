#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
# os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
intr_decomp_path = repo_root / "intr_decomp_3dgs"
if str(intr_decomp_path) not in sys.path:
    sys.path.insert(0, str(intr_decomp_path))

# import cv2
import numpy as np
from random import randint
from utils.loss_utils import l1_loss
from gaussian_renderer import render, investigate
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, linear_to_inverse
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_viewer import GaussianViewer
from fusion.gaussian_average import compute_gaussian_albedo_targets

from scene import Camera
from enum import Enum
from threading import Thread
from gs_viewer.viewer_types import ViewerMode
from simple_knn._C import distCUDA2
from utils.graphics_utils import depths_to_points

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from fused_ssim import fused_ssim

# EXTENSION 2 
import fusion.down_weight as e2

class Layer(Enum):
    albedo = 0
    shading = 1
    residual = 2

def init_residual(albedo_gaussians : GaussianModel, shading_gaussians : GaussianModel, scene : Scene, pipe : PipelineParams) -> GaussianModel:
    residual_xyz_cpu = []
    for cam in tqdm(scene.getTrainCameras(), desc="Generating residual gaussians"):
        bg = torch.rand((3), device="cuda")

        shading_render = render(cam, shading_gaussians, pipe, bg, colors_activation=False)["render"]
        albedo_render = render(cam, albedo_gaussians, pipe, bg, colors_activation=True)["render"]

        gt_image = cam.gt_active
        diffuse_render = albedo_render * shading_render

        diff_image = (gt_image - diffuse_render)
        pixel_candidates_mask = torch.any(diff_image > 0.1, dim=0)
        # pixel_candidates_mask = torch.any(diff_image > 1.0, dim=0)

        # Randomly sub-sample candidate pixels, we dont need a gaussian per pixel
        avail_pixels = torch.nonzero(pixel_candidates_mask, as_tuple=False)
        nb_samples = int(0.01 * avail_pixels.shape[0])
        
        # Parts of the scene do not need any residual gaussians because they are mostly diffuse.
        if nb_samples == 0:
            continue

        selected_indices = torch.multinomial(torch.ones(avail_pixels.shape[0], device="cuda"), nb_samples, replacement=False)
        selected_pixels = avail_pixels[selected_indices]
        
        # Reuse already initialized mask
        pixel_candidates_mask[:, :] = False
        pixel_candidates_mask[selected_pixels[:, 0], selected_pixels[:, 1]] = True

        # residual_xyz_cpu.append(depths_to_points(cam, 1.0 / albedo_depth, pixel_candidates_mask).detach().cpu())
        residual_xyz_cpu.append(depths_to_points(cam, 1.0 / cam.gt_depth.unsqueeze(0), pixel_candidates_mask).detach().cpu())

    residual_xyz = torch.cat(residual_xyz_cpu, dim=0).to("cuda")    
    dist2 = torch.clamp_min(distCUDA2(residual_xyz), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    rots = torch.zeros((residual_xyz.shape[0], 4), device="cuda")
    rots[:, 0] = 1
    
    residual_gaussians = GaussianModel(sh_degree=3, grayscale_dc=True)
    residual_gaussians.use_color_activation = False
    residual_gaussians.max_radii2D = torch.zeros((residual_xyz.shape[0]), device="cuda")
    residual_gaussians.setup_functions()
    residual_gaussians.from_params({
        "_xyz": residual_xyz.requires_grad_(True),
        # "_features_dc":  self.edition_gaussians.inverse_features_dc_activation(0.1 * torch.ones((pointmap.shape[0], 1, 3), dtype=torch.float, device="cuda")),
        "_features_dc": (torch.ones((residual_xyz.shape[0], 1, 3), device="cuda") * 0.001).requires_grad_(True),
        "_features_rest": (torch.ones((residual_xyz.shape[0], 15, 3), device="cuda") * 0.0001).requires_grad_(True), #self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        "_scaling": scales.requires_grad_(True),
        "_rotation": rots.requires_grad_(True),
        "_opacity": residual_gaussians.inverse_opacity_activation(0.1 * torch.ones((residual_xyz.shape[0], 1), dtype=torch.float, device="cuda")),
        # "_opacity": torch.ones((residual_xyz.shape[0], 1), device="cuda").requires_grad_(True)
    })

    return residual_gaussians


def albedo_loss(albedo_gaussians : GaussianModel, viewpoint : Camera, depth_weight : float, pipe : PipelineParams, background : torch.Tensor, lambda_dssim : float, clamp_albedo : bool = True, use_weight_map: bool = False, gaussian_average=False, target_albedo=None, scene: Scene | None = None):
    render_pkg = render(viewpoint, albedo_gaussians, pipe, background)
    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

    gt_image = viewpoint.gt_active
    if clamp_albedo:
        gt_image = torch.clamp(gt_image, 0, 1)

    # for extension 2, if we warp predicted albedos via depth and calculate a confidence weight map based on 
    # when pixels disagree, we can use the weight map to down weight pixels where the views disagree

    if use_weight_map and scene:
        weight_map = e2.generate_confidence_maps(viewpoint, scene)
        Ll1 = (torch.abs(image - gt_image) * weight_map).mean()
    elif use_weight_map and not scene:
        print("Pass scene to get a weight map")
        Ll1 = l1_loss(image, gt_image)
    else:
        Ll1 = l1_loss(image, gt_image)


    ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))

    loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_value) + 0.1 * torch.mean(albedo_gaussians.get_scaling)

    # integrate extension 3
    if gaussian_average and target_albedo is not None:
        current_rgb = torch.sigmoid(albedo_gaussians._features_dc)[:, 0, :]
        loss += 0.1 * torch.abs(current_rgb - target_albedo).mean()

    # depth regularization
    Ll1depth_pure = 0.0
    if viewpoint.depth_reliable and depth_weight > 0:
        invDepth = render_pkg["depth"]
        mono_invdepth = viewpoint.gt_depth
        
        # BASIC L1 LOSS
        Ll1depth_pure = torch.abs(invDepth  - mono_invdepth).mean()

        Ll1depth = depth_weight * Ll1depth_pure 
        loss += Ll1depth
        Ll1depth = Ll1depth.item()
    else:
        Ll1depth = 0

    return loss, Ll1, Ll1depth, viewspace_point_tensor, visibility_filter, radii

def shading_loss(albedo_gaussians : GaussianModel, shading_gaussians : GaussianModel, viewpoint : Camera, depth_weight : float, pipe : PipelineParams, background : torch.Tensor, lambda_dssim : float, lambda_col : float, inverse_space : bool = True, color_reg : bool = True):
    render_pkg = render(viewpoint, shading_gaussians, pipe, background, colors_activation=False)
    rend_shading, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    rend_albedo = render(viewpoint, albedo_gaussians, pipe, background)["render"]

    diffuse_render = rend_shading * rend_albedo
    
    gt_image = viewpoint.gt_active

    if inverse_space:
        gt_image = linear_to_inverse(gt_image)
        diffuse_render = linear_to_inverse(diffuse_render)

    
    Ll1 = l1_loss(diffuse_render, gt_image)
    ssim_value = fused_ssim(diffuse_render.unsqueeze(0), gt_image.unsqueeze(0))

    loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_value) + 0.1 * torch.mean(shading_gaussians.get_scaling)  # mean of range per gaussian
    
    if color_reg:
        mean_range_gaussians = torch.mean(torch.max(shading_gaussians.get_features_dc, dim=2).values - torch.min(shading_gaussians.get_features_dc, dim=2).values)
        loss += lambda_col * mean_range_gaussians

    # Depth regularization
    Ll1depth_pure = 0.0
    if depth_weight > 0:
        invDepth = render_pkg["depth"]
        mono_invdepth = viewpoint.gt_depth

        Ll1depth_pure = torch.abs(invDepth  - mono_invdepth).mean()
        Ll1depth = depth_weight * Ll1depth_pure 
        loss += Ll1depth
        Ll1depth = Ll1depth.item()
    else:
        Ll1depth = 0

    return loss, Ll1, Ll1depth, viewspace_point_tensor, visibility_filter, radii

def residual_loss(albedo_gaussians : GaussianModel, shading_gaussians : GaussianModel, residual_gaussians : GaussianModel, viewpoint : Camera, depth_weight : float, pipe : PipelineParams, background : torch.Tensor, lambda_dssim : float, lambda_col : float, inverse_space : bool = True, color_reg : bool = True):
    residual_pkg = render(viewpoint, residual_gaussians, pipe, background, colors_activation=False, enable_shs=True)
    rend_albedo = render(viewpoint, albedo_gaussians, pipe, background)["render"]
    rend_shading = render(viewpoint, shading_gaussians, pipe, background, colors_activation=False)["render"]
    rend_residual, viewspace_point_tensor, visibility_filter, radii = residual_pkg["render"], residual_pkg["viewspace_points"], residual_pkg["visibility_filter"], residual_pkg["radii"]

    glossy_render = rend_shading * rend_albedo + rend_residual
    
    gt_image = viewpoint.gt_active
    if inverse_space:
        gt_image = linear_to_inverse(gt_image)
        glossy_render = linear_to_inverse(glossy_render)

    
    Ll1 = l1_loss(glossy_render, gt_image)
    ssim_value = fused_ssim(glossy_render.unsqueeze(0), gt_image.unsqueeze(0))

    loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_value) + 0.1 * torch.mean(residual_gaussians.get_scaling)  # mean of range per gaussian
    
    if color_reg:
        mean_range_gaussians = torch.mean(torch.max(residual_gaussians.get_features_dc, dim=2).values - torch.min(residual_gaussians.get_features_dc, dim=2).values)
        loss += lambda_col * mean_range_gaussians

    # Depth regularization
    Ll1depth_pure = 0.0
    if depth_weight > 0:
        invDepth = residual_pkg["depth"]
        mono_invdepth = viewpoint.gt_depth

        Ll1depth_pure = torch.abs(invDepth  - mono_invdepth).mean()
        Ll1depth = depth_weight * Ll1depth_pure 
        loss += Ll1depth
        Ll1depth = Ll1depth.item()
    else:
        Ll1depth = 0

    return loss, Ll1, Ll1depth, viewspace_point_tensor, visibility_filter, radii

def training_loop(scene : Scene, iterations : int, dataset : ModelParams, opt : OptimizationParams, pipe : PipelineParams, testing_iterations: list[int], saving_iterations : list[int], gaussians_list : list[GaussianModel], tb_writer = None, gaussian_average : bool = False, down_weight: bool = False):

    first_iter = 0
    progress_bar = tqdm(range(first_iter, iterations), desc="Training progress")
    first_iter += 1
    viewpoint_stack = None
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=iterations)
    active_id = len(gaussians_list) - 1

    shading_gaussians, residual_gaussians = None, None
    albedo_gaussians = gaussians_list[0]
    
    if len(gaussians_list) > 1:
        shading_gaussians = gaussians_list[1]
    
    if len(gaussians_list) > 2:
        residual_gaussians = gaussians_list[2]

    target_albedo = None
    
    for iteration in range(first_iter, iterations + 1):

        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        gaussians_list[active_id].update_learning_rate(iteration)
        bg = torch.rand((3), device="cuda")

        if gaussian_average and (target_albedo is None or target_albedo.shape[0] != albedo_gaussians.get_xyz.shape[0] or iteration % 100 == 0):
            with torch.no_grad():
                target_albedo = compute_gaussian_albedo_targets(albedo_gaussians, scene.getTrainCameras())

        if residual_gaussians is not None:
            lambda_col = 0.3 if iteration < 3000 else 0.1
            loss, Ll1, depth_loss, viewspace_point_tensor, visibility_filter, radii = residual_loss(albedo_gaussians, shading_gaussians, residual_gaussians, viewpoint_cam, depth_l1_weight(iteration), pipe, bg, opt.lambda_dssim, lambda_col)
        elif shading_gaussians is not None:
            lambda_col = 0.3 if iteration < 3000 else 0.1
            loss, Ll1, depth_loss, viewspace_point_tensor, visibility_filter, radii = shading_loss(albedo_gaussians, shading_gaussians, viewpoint_cam, depth_l1_weight(iteration), pipe, bg, opt.lambda_dssim, lambda_col)
        else:
            if down_weight:
                loss, Ll1, depth_loss, viewspace_point_tensor, visibility_filter, radii = albedo_loss(albedo_gaussians, viewpoint_cam, depth_l1_weight(iteration), pipe, bg, opt.lambda_dssim, gaussian_average=gaussian_average, target_albedo=target_albedo, use_weight_map=down_weight, scene=scene)
            else:
                loss, Ll1, depth_loss, viewspace_point_tensor, visibility_filter, radii = albedo_loss(albedo_gaussians, viewpoint_cam, depth_l1_weight(iteration), pipe, bg, opt.lambda_dssim, gaussian_average=gaussian_average, target_albedo=target_albedo, use_weight_map=down_weight)

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * depth_loss + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == iterations:
                progress_bar.close()

            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, bg),
                albedo_gaussians=albedo_gaussians,
                shading_gaussians=shading_gaussians,
                residual_gaussians=residual_gaussians,
                should_clamp=dataset.clamp_albedo
            )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                if residual_gaussians is not None:
                    out_pc_path = os.path.join(dataset.model_path, "point_cloud/iteration_{}".format(iteration))
                    residual_gaussians.save_as_safetensor(os.path.join(out_pc_path, "residual.safetensors"))
                elif shading_gaussians is not None:
                    scene.save(iteration, pc_name="shading.ply")
                else:
                    scene.save(iteration, pc_name="albedo.ply")

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians_list[active_id].max_radii2D[visibility_filter] = torch.max(gaussians_list[active_id].max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians_list[active_id].add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians_list[active_id].densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                    
                    # remove poorly initialized scene.gaussians that would result in floaters? 
                    if dataset.clean_floaters:
                        suspect = investigate(viewpoint_cam, gaussians_list[active_id], pipe, bg, nstddev=0.3)
                        print(f"{(suspect > 0).sum()} gaussians_list[active_id] are suspected to be floaters ....")
                        gaussians_list[active_id].prune_points((suspect > 0).squeeze())
                        gaussians_list[active_id].tmp_radii = None
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians_list[active_id].reset_opacity() # by default opacity get resetted to 0.01

            # Optimizer step
            if iteration < iterations:
                visible = radii > 0
                gaussians_list[active_id].optimizer.step(visible, radii.shape[0])
                gaussians_list[active_id].optimizer.zero_grad(set_to_none = True)


def training(dataset : ModelParams, opt : OptimizationParams, pipe : PipelineParams, testing_iterations: list[int], saving_iterations : list[int], enable_viewer : bool = False, out_folder : str = None, albedo_checkpoint : str = "", colour_correction : bool = False, down_weight : bool = False, gaussian_average : bool = False):
    tb_writer = prepare_output_and_logger(dataset)

    albedo_gaussians = GaussianModel(0)
    shading_gaussians = GaussianModel(0)
    shading_gaussians.use_color_activation = False

    scene = Scene(dataset, albedo_gaussians, active_layer="albedo")

    albedo_gaussians.training_setup(opt)
    albedo_gaussians.use_color_activation = True

    print(f"starting with {albedo_gaussians.get_xyz.shape[0]} points.")

    viewer = None
    if enable_viewer:
        mode = ViewerMode.LOCAL
        viewer = GaussianViewer.from_gaussians(dataset=dataset, pipe=pipe, mode=mode, albedo_gaussians=albedo_gaussians)
        viewer_thd = Thread(target=viewer.run, daemon=True)
        viewer_thd.start()
    
    training_loop(scene, opt.albedo_iterations, dataset, opt, pipe, testing_iterations, saving_iterations, gaussians_list=[albedo_gaussians], tb_writer=tb_writer, gaussian_average=gaussian_average)
    albedo_gaussians.freeze_params()
    albedo_gaussians.optimizer = None

    if enable_viewer:
        viewer.set_shading_gaussians(shading_gaussians)

    # shading reconstruction based on pretrained albedo
    scene.gaussians = shading_gaussians
    scene.set_active_layer("input")
    scene.reset_gaussians()
    shading_gaussians.training_setup(opt)
    training_loop(scene, opt.shading_iterations, dataset, opt, pipe, testing_iterations, saving_iterations, gaussians_list=[albedo_gaussians, shading_gaussians], tb_writer=tb_writer)


    # residual reconstruction based on pretrained albedo and shading
    shading_gaussians.freeze_params()
    shading_gaussians.optimizer = None
    residual_gaussians = init_residual(albedo_gaussians, shading_gaussians, scene, pipe)
    residual_gaussians.training_setup(opt)

    if enable_viewer:
        viewer.set_residual_gaussians(residual_gaussians)
            
    scene.gaussians = residual_gaussians
    training_loop(scene, opt.residual_iterations, dataset, opt, pipe, testing_iterations, saving_iterations, gaussians_list=[albedo_gaussians, shading_gaussians, residual_gaussians], tb_writer=tb_writer, down_weight=down_weight)


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, albedo_gaussians : GaussianModel = None, shading_gaussians : GaussianModel = None, residual_gaussians : GaussianModel = None, should_clamp : bool = False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    layer = Layer.albedo
    if shading_gaussians is not None:
        layer = Layer.shading
    elif residual_gaussians is not None:
        layer = Layer.residual

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if layer == Layer.albedo:
                        active_albedo = albedo_gaussians if albedo_gaussians is not None else scene.gaussians
                        image = renderFunc(viewpoint, active_albedo, *renderArgs)["render"]
                        gt_image = viewpoint.gt_active
                        if should_clamp:
                            gt_image = gt_image.clamp(0, 1)
                    elif layer == Layer.shading:
                        if albedo_gaussians is None or shading_gaussians is None:
                            raise ValueError("Layer.shading evaluation requires both albedo_gaussians and shading_gaussians")

                        rend_albedo = renderFunc(viewpoint, albedo_gaussians, *renderArgs)["render"]
                        rend_shading = renderFunc(viewpoint, shading_gaussians, *renderArgs, colors_activation=False)["render"]

                        image = rend_albedo * rend_shading
                        gt_image = viewpoint.gt_active
                    elif layer == Layer.residual:
                        if albedo_gaussians is None or shading_gaussians is None:
                            raise ValueError("Layer.residual evaluation requires both albedo_gaussians and shading_gaussians")

                        active_residual = residual_gaussians if residual_gaussians is not None else scene.gaussians
                        rend_albedo = renderFunc(viewpoint, albedo_gaussians, *renderArgs)["render"]
                        rend_shading = renderFunc(viewpoint, shading_gaussians, *renderArgs, colors_activation=False)["render"]
                        rend_residual = renderFunc(viewpoint, active_residual, *renderArgs, colors_activation=False, enable_shs=True)["render"]

                        image = rend_albedo * rend_shading + rend_residual
                        gt_image = viewpoint.gt_active
                    else:
                        raise ValueError(f"Unsupported evaluation layer: {layer}")

                    if layer in (Layer.shading, Layer.residual):
                        # both shading and residual Losses are computed in inverse space when optimizing them, mimic this in evaluations.
                        image = linear_to_inverse(image)
                        gt_image = linear_to_inverse(gt_image)

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--out_folder', type=str, default="exp")
    parser.add_argument('--albedo_checkpoint', type=str, default="")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--enable_viewer", action="store_true")
    # enable extensions
    parser.add_argument("--use_color_correction", action="store_true")
    parser.add_argument("--use_down_weight", action="store_true")
    parser.add_argument("--use_gaussian_average", action="store_true")
    # may need to add other hyperparameters if needed for any extensions
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.albedo_iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.enable_viewer, out_folder = args.out_folder, albedo_checkpoint = args.albedo_checkpoint, colour_correction=args.use_color_correction, down_weight=args.use_down_weight, gaussian_average=args.use_gaussian_average)

    # All done
    print("\nTraining complete.")
