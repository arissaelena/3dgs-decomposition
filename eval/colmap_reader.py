# reads the colmap style mip-360 data and converted hypersim data

import os
import re
import struct
import numpy as np

def load_scene_cameras(colmap_scene_dir: str, split: str = "all", llffhold: int = 8) -> list[dict]:
    sparse_dir = os.path.join(colmap_scene_dir, "sparse", "0")
    intrinsics = read_cameras_bin(os.path.join(sparse_dir, "cameras.bin"))
    frames = []

    with open(os.path.join(sparse_dir, "images.bin"), "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]): # total number of registered images
            struct.unpack("<i", f.read(4)) # image id
            qvec = np.array(struct.unpack("<4d", f.read(32)), dtype=np.float32) # w2c rotation as quaternion
            tvec = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float32) # w2c translation vec
            cid = struct.unpack("<i", f.read(4))[0] # camera id can get intrinsics from intrinsi s

            raw = bytearray()
            while (c := f.read(1)) != b"\x00":
                raw.extend(c) # decode image file name
            name = raw.decode("utf-8")
            f.seek(struct.unpack("<Q", f.read(8))[0] * 24, 1) # skip to next image

            colmap_stem = os.path.splitext(os.path.basename(name))[0]
            frames.append({
                "colmap_name": name,
                "colmap_stem": colmap_stem,
                "frame_num": int(re.findall(r"\d+", colmap_stem)[-1]), # get numbers from file name
                "R": qvec_to_rotmat(qvec),
                "t": tvec,
                **intrinsics[cid],
            })

    frames.sort(key=lambda x: x["frame_num"])
    return frames

def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    rotmat = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz),     2 * (qz * qx + qw * qy)],
        [2 * (qx * qy + qw * qz),     1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
        [2 * (qz * qx - qw * qy),     2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
    ], dtype=np.float32)
    return rotmat


def read_cameras_bin(path: str) -> dict[int, dict]:
    cam_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8} # colmap model id to number of params map
    intrinsics= {}
    with open(path, "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            cid, mid, w, h = struct.unpack("<iiQQ", f.read(24))
            p = struct.unpack(f"<{cam_params[mid]}d", f.read(8 * cam_params[mid]))
            if mid in (0, 2, 3):
                fx, fy = (p[0], p[0])
                cx, cy = (p[1], p[2])
            else:
                fx, fy = (p[0], p[1])
                cx, cy = (p[2], p[3])
            intrinsics[cid] = {"w": w, "h": h, "fx": fx, "fy": fy, "cx": cx, "cy": cy}
    return intrinsics

def scale_intrinsics(cam: dict, target_w: int, target_h: int) -> tuple[float, float, float, float]:
    sx = target_w / float(cam["w"])
    sy = target_h / float(cam["h"])
    scaled_fx = cam["fx"] * sx
    scaled_fy = cam["fy"] * sy
    scaled_cx = cam["cx"] * sx
    scaled_cy = cam["cy"] * sy
    return scaled_fx, scaled_fy, scaled_cx, scaled_cy