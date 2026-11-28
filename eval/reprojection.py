# measure how inconsistent each is across views by reprojecting its predictions between views using depth from the splat

from data.prepare_hypersim import ray_depth_to_planar_z
import numpy as np
import cv2
from colmap_reader import scale_intrinsics
from loaders import get_depth_path, get_gt_depth_path, load_depth
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


def covisible_pairs(root: str, dataset: str, scene: str, cams: list[dict], predictor: str, change: list[int] = [2, 8], min_overlap: float = 0.1, use_gt_depth: bool = False, thresh: float = 0.12, scale: float = 0.25) -> dict[int, list[tuple[int, int]]]:
    depths = {}
    small_cams = [scale_cam(c, scale) for c in cams]
    depth_fn = get_gt_depth_path if use_gt_depth else get_depth_path

    if not use_gt_depth:
        occ_thresh = 0.15
        edge_thresh = 0.30 * (1.0 / scale) ** 0.5 # adjust pixel step grad for downsampled grid
    else:
        occ_thresh = thresh
        edge_thresh = thresh * (1.0 / scale) ** 0.5
            
    pairs_by_diff = {d: [] for d in change}
    for i in range(len(cams)):
        if i not in depths:
            p = depth_fn(root, dataset, scene, cams[i], predictor)
            if os.path.exists(p):
                depths[i] = load_depth(p, (small_cams[i]["h"], small_cams[i]["w"]))
        
        if i in depths:
            for d in change:
                j = i + d
                if j < len(cams):
                    if j not in depths:
                        p = depth_fn(root, dataset, scene, cams[j], predictor)
                        if os.path.exists(p):
                            depths[j] = load_depth(p, (small_cams[j]["h"], small_cams[j]["w"])) # downsampled size
                    if j in depths:
                        h, w = depths[j].shape
                        dummy = np.zeros((h, w, 3), dtype=np.float32) # placeholder in right shape
                    
                        _, mask = reproject_view(small_cams[i], small_cams[j], depths[i], depths[j], dummy, occ_thresh, edge_thresh)
                        if mask.mean() >= min_overlap:
                            pairs_by_diff[d].append((i, j))
        depths.pop(i, None)

    return pairs_by_diff


def scale_cam(cam: dict, scale: float) -> dict:
    sc = dict(cam)
    sc["h"] = int(round(cam["h"] * scale))
    sc["w"] = int(round(cam["w"] * scale))
    if "K" in sc:
        K = sc["K"].copy()
        K[0, :] *= scale
        K[1, :] *= scale
        sc["K"] = K
    for k in ["fc", "fy", "cx", "cy"]:
        if k in sc:
            sc[k] = sc[k] * scale
    return sc