# measure how inconsistent each is across views by reprojecting its predictions between views using depth from the splat

from data.prepare_hypersim import ray_depth_to_planar_z
import numpy as np
import cv2
from colmap_reader import scale_intrinsics
from loaders import get_depth_path
import os

def reproject_view(cam_src: dict, cam_tgt: dict, depth_src: np.ndarray, depth_tgt: np.ndarray, albedo_tgt: np.ndarray, occ_thresh: float = 0.05, edge_thresh: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    # BACKWARD warp target albedo into source camera pov where center is 0.5 because colmap does 0-1
    
    h, w = depth_src.shape
    h_tgt, w_tgt = depth_tgt.shape

    fx_s, fy_s, cx_s, cy_s = scale_intrinsics(cam_src, w, h)
    fx_t, fy_t, cx_t, cy_t = scale_intrinsics(cam_tgt, w_tgt, h_tgt)

    # back project source pixel centers (+0.5) to camera coords in 3d world
    u, v = np.meshgrid(np.arange(w, dtype=np.float32) + 0.5, np.arange(h, dtype=np.float32) + 0.5)
    # pixel centers for every pixel in view
    # shift center
    u_cent = u - cx_s
    v_cent = v - cy_s
    # to normalised camera coords
    x_norm = u_cent / fx_s
    y_norm = v_cent / fy_s
    # scale ray by planar depth
    X_i = x_norm * depth_src
    Y_i = y_norm * depth_src

    pts_i = np.stack([X_i, Y_i, depth_src], axis=-1)
    # source camera -> target camera
    R_s, t_s = cam_src["R"], cam_src["t"]
    R_t, t_t = cam_tgt["R"], cam_tgt["t"]
    pts_world = (pts_i - t_s) @ R_s
    pts_j = pts_world @ R_t.T + t_t

    # project onto target pixel indices and switch to opencv indexing
    X_j, Y_j, Z_j = pts_j[..., 0], pts_j[..., 1], pts_j[..., 2]
    z_safe = np.maximum(Z_j, 1e-4) # stop div by 0
    u_j = (fx_t * (X_j / z_safe) + cx_t - 0.5).astype(np.float32)
    v_j = (fy_t * (Y_j / z_safe) + cy_t - 0.5).astype(np.float32)

    # sample target depth and albedo
    sampled_z_tgt = cv2.remap(depth_tgt, u_j, v_j, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_albedo = cv2.remap(albedo_tgt, u_j, v_j, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # in front of camera in image
    in_bounds = (Z_j > 1e-4) & (u_j >= 0) & (u_j <= w_tgt - 1) & (v_j >= 0) & (v_j <= h_tgt - 1)
    # difference in depth means its occluded
    not_occluded = (sampled_z_tgt > 1e-4) & (np.abs(Z_j - sampled_z_tgt) / z_safe < occ_thresh)

    # depth jumps to right and down
    gx = np.zeros_like(depth_src)
    gy = np.zeros_like(depth_src)
    gx[:, :-1] = depth_src[:, 1:] - depth_src[:, :-1]
    gy[:-1, :] = depth_src[1:, :] - depth_src[:-1, :]
    grad_mag = np.sqrt(gx**2 + gy**2)
    # keep jumps below threshold
    z_safe = np.maximum(depth_src, 1e-4)
    not_edge = (depth_src > 1e-4) & ((grad_mag / z_safe) < edge_thresh)

    M = in_bounds & not_occluded & not_edge
    return warped_albedo, M


def covisible_pairs(root: str, dataset: str, scene: str, cams: list[dict], change: list[int] = [2, 8], min_overlap: float = 0.1) -> dict [int, list[tuple[int, int]]]:
    depths = {}
    if dataset == "hypersim":
        thresh = 0.05 # can be stricter because the depths are synthetic
    else:
        thresh = 0.12

    for i, cam in enumerate(cams):
        p = get_depth_path(root, dataset, scene, cam)
        if os.path.exists(p):
            depths[i] = np.load(p).astype(np.float32).squeeze()
    
    pairs_by_diff = {}
    for d in change:
        pairs_by_diff[d] = []
        for i in range(len(cams) - d):
            j = i+d
            if i in depths and j in depths:
                h, w = depths[j].shape
                dummy = np.zeros((h, w, 3), dtype=np.float32) # placeholder in right shape
                _, mask = reproject_view(cams[i], cams[j], depths[i], depths[j], dummy, thresh, thresh)
                if mask.mean() >= min_overlap:
                    pairs_by_diff[d].append((i, j))

    return pairs_by_diff