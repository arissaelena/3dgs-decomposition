import os
import cv2
import numpy as np


def get_albedo_path(root: str, method: str, dataset: str, scene: str, cam: dict) -> str:
    # get the path to the albedo npy
    # hypersim - frame.0000.tonemap.npy
    # mipnerf 360 - cam['colmap_stem'] + ".npy"
    albedo_dir = os.path.join(root, method, dataset, scene, "albedos")
    if dataset == "hypersim":
        stems = [f"frame.{cam['colmap_stem'].split('_')[-1]}.tonemap", cam["colmap_stem"]]
    else:
        stems = [cam["colmap_stem"]]

    for stem in stems:
        for ext in (".npy", ".png", ".exr"):
            albedo_path = os.path.join(albedo_dir, f"{stem}{ext}")
            if os.path.exists(albedo_path):
                return albedo_path

    return os.path.join(albedo_dir, f"{stems[0]}.npy")


def get_depth_path(root: str, dataset: str, scene: str, cam: dict) -> str:
    if dataset == "hypersim":
        base = os.path.join(root, "data", "hypersim", "colmap_format", scene, "cam_00")
        depth_dirs = [os.path.join(base, "ground_truth", "depth_z"), os.path.join(base, "depths"), os.path.join(base, "depth")]
    else:
        depth_dirs = [os.path.join(root, "eval", "depth_cache", "mipnerf360", scene), os.path.join(root, "data", "mipnerf360", scene, "depths")]

    stems = [cam["colmap_stem"]]
    if cam["colmap_stem"].endswith(".color"):
        stems.append(cam["colmap_stem"][: -len(".color")])

    for depth_dir in depth_dirs:
        for stem in stems:
            for ext in (".npy", ".png", ".hdf5", ".h5"):
                depth_path = os.path.join(depth_dir, f"{stem}{ext}")
                if os.path.exists(depth_path):
                    return depth_path

    return os.path.join(depth_dirs[0], f"{cam['colmap_stem']}.npy")


def load_albedo(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        arr = np.load(path).astype(np.float32)
    else:
        arr = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)

    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.max() > 1.5:
        arr /= 255.0  # get to all between 0 and 1
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def load_depth(path: str, target_hws: tuple[int, int]) -> np.ndarray:
    d = np.load(path).astype(np.float32).squeeze()
    h, w = target_hws
    if d.shape != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    return d