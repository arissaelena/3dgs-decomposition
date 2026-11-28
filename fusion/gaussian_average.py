# constrain each gaussian’s albedo to a weighted average of its per-view predictions
import torch
import numpy as np
import math
import os

def project_gaussians_to_camera(gaussians_xyz, camera):
    pos_2d = torch.cat([gaussians_xyz, torch.ones((len(gaussians_xyz), 1)).to("cuda")], dim=1) @ camera.full_proj_transform 
    ndc_pos = pos_2d[:, :2] / pos_2d[:, 3:4]
    u = (ndc_pos[:, 0] + 1) * 0.5 * camera.image_width
    v = (ndc_pos[:, 1] + 1) * 0.5 * camera.image_height

    z = (torch.cat([gaussians_xyz, torch.ones((len(gaussians_xyz), 1)).to("cuda")], dim=1) @ camera.world_view_transform)[..., 2]

    return u, v, z

def compute_gaussian_albedo_targets(albedo_gaussians, train_cameras):
    xyz = albedo_gaussians.get_xyz
    accum_colors = torch.zeros(xyz.shape, device="cuda")
    accum_weights = torch.zeros((len(xyz), 1), device="cuda")

    for camera in train_cameras:
        u, v, z = project_gaussians_to_camera(xyz, camera)
        H, W = camera.image_height, camera.image_width
        albedo = camera.gt_albedo.cuda().permute(1, 2, 0)
        visible = (v >= 0) & (v < H) & (u >= 0) & (u < W) & (z > 0.1)
        u_vis = u[visible].long()
        v_vis = v[visible].long()
        z_vis = z[visible]
        surface_depth = camera.gt_depth.squeeze()[v_vis, u_vis]

        vis_idx = torch.where(visible)[0]
        valid_depths = (abs((1 / z_vis) - surface_depth) < 1e-2)
        valid_idx = vis_idx[valid_depths]
        u_vis = u[valid_idx].long()
        v_vis = v[valid_idx].long()

        sampled_albedo = albedo[v_vis, u_vis]
        accum_colors[valid_idx] += sampled_albedo
        accum_weights[valid_idx] += 1.0

    return torch.where(accum_weights > 0, accum_colors / accum_weights.clamp(min=1e-6), torch.sigmoid(albedo_gaussians._features_dc)[:, 0, :])


