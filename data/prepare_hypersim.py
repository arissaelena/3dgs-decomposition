# functions to convert hypersim data to COLMAP sparse/0/ and ground truths for comparison

# hypersim scenes are made up of HDR renderers and lighting in .hdf5 files
# tone-mapped RGP visual previews in .jpg files
# depth maps in .hdf5 files where the values give ray distance from camera center
# geometry and normals (per-pixel) in .hdf5
# per pixel semantic labels and segmentation information as hdf5 or png
# camera and scene metadata in csvs
# detail contains the csv files that link node ids to semantic labels, camera trajectories and 3d bbs
# images have folders per trajectory which are seperated into different renders 

import numpy as np
import os
import csv
import h5py
from PIL import Image
import struct
import json

# hypersim uses rotation matrices to store orientations / camera rotations, COLMAP uses quaternions
def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    a, b, c, d, e, f, g, h, i = rotation_matrix.flat
    k = np.array([
        [a - e - i, d + b,     g + c,     h - f],
        [d + b,     e - a - i, h + f,     c - g],
        [g + c,     h + f,     i - a - e, d - b],
        [h - f,     c - g,     d - b,     a + e + i],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    leigvec = eigvecs[:, np.argmax(eigvals)]  
    quaternion = np.array([leigvec[3], leigvec[0], leigvec[1], leigvec[2]])
    return -quaternion if quaternion[0] < 0 else quaternion


# get the meters per asset conversion from _detail/metadata_scene.csb
def get_scale(scene_dir_path: str) -> float:
    metadata_csv = os.path.join(scene_dir_path, "_detail", "metadata_scene.csv")
    with open(metadata_csv, newline="") as f:
        # will close itself when its done
        reader = csv.DictReader(f)
        if reader.fieldnames and "parameter_name" in reader.fieldnames:
            for row in reader:
                if row["parameter_name"] == "meters_per_asset_unit":
                    return float(row["parameter_value"])
        else:
            return float(next(reader)["meters_per_asset_unit"])
    raise KeyError(f"meters_per_asset_unit not found in {metadata_csv}")
# hypersim measures everything in assets, and COLMAP does meters

def get_intrinsics(s_name: str, c_name: str, w: int = 1024, h: int = 768) -> tuple[float, float, float, float]:
    # intrinsics of the pinhole camera in pixels fx, fy, cx, cy which are focal lengths and principal point coordinates
    detail_dir = os.path.join(s_name, "_detail", c_name)

    with h5py.File(os.path.join(detail_dir, "camera_keyframe_positions.hdf5"), "r") as f:
        positions_asset = f["dataset"][:]
    with h5py.File(os.path.join(detail_dir, "camera_keyframe_orientations.hdf5"), "r") as f:
        orientations_opengl = f["dataset"][:]

    geometry_dir = os.path.join(s_name, "images", f"scene_{c_name}_geometry_hdf5")

    for i in range(len(positions_asset)):
        # use first frame with everything we need
        translation_c2w_asset = positions_asset[i]
        rotation_c2w_opengl = orientations_opengl[i]
        pose_path = os.path.join(geometry_dir, f"frame.{i:04d}.position.hdf5")

        if (np.all(np.isfinite(translation_c2w_asset)) and np.all(np.isfinite(rotation_c2w_opengl)) and os.path.exists(pose_path)):
            with h5py.File(pose_path, "r") as f:
                points_world = f["dataset"][:]  # (h, w, 3) in asset units
    
            h, w, _ = points_world.shape
            rotation_c2w_opencv = rotation_c2w_opengl @ np.diag([1.0, -1.0, -1.0]) # convert between opengl to opencv standards
            points_camera = (points_world - translation_c2w_asset) @ rotation_c2w_opencv # world into opencv standard camera

            # build pixel center grid
            u, v = np.meshgrid(np.arange(w, dtype=np.float64) + 0.5, np.arange(h, dtype=np.float64) + 0.5)

            # rays hit scene geometry and depth more than 0.1
            valid = np.all(np.isfinite(points_camera), axis=-1) & (points_camera[..., 2] > 0.1)
            x_norm = points_camera[valid, 0] / points_camera[valid, 2]
            y_norm = points_camera[valid, 1] / points_camera[valid, 2]


            ax, bx = np.polyfit(u[valid], x_norm, 1)
            ay, by = np.polyfit(v[valid], y_norm, 1)

            fx = float(1.0 / ax)
            fy = float(1.0 / ay)
            cx = float(-bx / ax)
            cy = float(-by / ay)
            
            return fx, fy, cx, cy
    # fallback
    fx = w / (2.0 * np.tan(np.deg2rad(60.0) / 2.0))
    return float(fx), float(fx), w / 2.0, h / 2.0

def load_trajectory_poses(scene_dir: str, cam_name: str, scale: float) -> tuple[np.ndarray, np.ndarray]:
    # get trajectory poses from hypersim datat
    cam_detail_dir = os.path.join(scene_dir, "_detail", cam_name)
    with h5py.File(os.path.join(cam_detail_dir, "camera_keyframe_positions.hdf5"), "r") as f:
        positions_meters = f["dataset"][:] * scale
    with h5py.File(os.path.join(cam_detail_dir, "camera_keyframe_orientations.hdf5"), "r") as f:
        orientations_gl = f["dataset"][:]
    return positions_meters, orientations_gl

def ray_depth_to_planar_z(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    # hypersim ray distance to planar z-depth used in COLMAP
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64) + 0.5, np.arange(h, dtype=np.float64) + 0.5)
    ray_norm = np.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2 + 1.0)
    z_depth = (depth / ray_norm).astype(np.float32)
    return z_depth

def backproject_frame(rgb: np.ndarray, z_depth: np.ndarray, rotation_c2w_cv: np.ndarray, translation_c2w: np.ndarray, fx: float, fy: float, cx: float, cy: float, stride: int = 16) -> tuple[np.ndarray, np.ndarray]:
    # backproject into world coordinates in meters like colmap uses
    h, w = z_depth.shape
    u_coords = np.arange(0, w, stride, dtype=np.float64) + 0.5
    v_coords = np.arange(0, h, stride, dtype=np.float64) + 0.5
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)
    z = z_depth[::stride, ::stride].astype(np.float64)
    colors = rgb[::stride, ::stride]

    valid = np.isfinite(z) & (z > 0.05) & (z < 100.0) # mask
    u = u_grid[valid]
    v = v_grid[valid]
    z = z[valid]
    colors = colors[valid]

    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    z_cam = z
    pts_cam = np.column_stack((x_cam, y_cam, z_cam))
    
    pts_world = np.matmul(pts_cam, rotation_c2w_cv.T) + translation_c2w
    
    return pts_world, colors


# poses_bounds.npy stores a 2d floating-point array of shape (N, 17) where N is then number of camera views / images in the scene
# each row of 17 numbers corresponds to a single image and its sorted alphabetically by filename
# the first 15 elements are the camera pose and intrinsics as a flattened 3x5 matrix
# last two represent the near and far depth bounds from the cameras perspective
# columns 0-2have the 3x3 camera2world rotation matrix
# third column has a translation vector rpresenting the camera's position in world space
# last column has the height, width and focal length in pixels
# columns need to be reordered in the 3x5 matrix because mipnerf is in a llff format

def make_poses_bounds_row(rotation_c2w: np.ndarray, translation_c2w: np.ndarray, h: int, w: int, f: float, near: float, far: float) -> np.ndarray:
    row = np.zeros(17, dtype=np.float64)
    # swap first two columns and negate third column
    r_llff = np.column_stack([rotation_c2w[:, 1], rotation_c2w[:, 0], -rotation_c2w[:, 2]])

    row[:3] = r_llff[0]
    row[3] = translation_c2w[0]
    row[4] = h
    row[5:8] = r_llff[1]
    row[8] = translation_c2w[1]
    row[9] = w
    row[10:13] = r_llff[2]
    row[13] = translation_c2w[2]
    row[14] = f
    row[15] = near
    row[16] = far
    return row


def write_comlap_bins(sparse_dir: str, w: int, h: int, intrinsics: tuple[float, float, float, float], images_metadata: list[tuple[int, np.ndarray, np.ndarray, str]], points: np.ndarray, colours: np.ndarray) -> None:
    # write COLMAP bin sparse/0/ files
    # images_metadata is a list of per image tupes - image id, w2c rotation quaternion, w2c translation vector, image file name
    fx, fy, cx, cy = intrinsics

    # write cameras.bin which has camera intrinsics
    with open(os.path.join(sparse_dir, "cameras.bin"), "wb") as f:
        # capital letter means unsigned 
        # letter indicates size 64, 32 eg
        # 4d is four floats
        # write 64 bit int that says theres 1 camera
        # < means little endian
        f.write(struct.pack("<Q", 1))
        # write camera parameters
        f.write(struct.pack("<IiQQ4d", 1, 1, w, h, fx, fy, cx, cy))
    
    # write images.bin per-image camera poses/orientation and position
    with open(os.path.join(sparse_dir, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(images_metadata))) # number of images
        for image_id, qvec, tvec, name in images_metadata:
            f.write(struct.pack("<I4d3dI", image_id, *qvec, *tvec, 1))
            f.write(name.encode("utf-8") + b"\x00") # null terminated filename, variable bytes
            f.write(struct.pack("<Q", 0))  # 0 2D keypoint tracks, just read next record
    
    # points3D.bin
    dt = np.dtype([
        ("id", "<u8"), # 8 byes of unsigned for the id
        ("xyz", "<f8", (3,)), # 3 x 8 bytes of floats for x y and z
        ("rgb", "<u1", (3,)), # 4 x 8 bytes of unsigned ints for r, g and b
        ("error", "<f8"), # 8 bytes of float for error
        ("track_len", "<u8"), # 8 bytes of uint64 for number of image tracks
    ])
    records = np.zeros(len(points), dtype=dt)
    if len(points) > 0:
        records["id"] = np.arange(1, len(points) + 1, dtype=np.uint64)
        records["xyz"] = points
        records["rgb"] = colours
    with open(os.path.join(sparse_dir, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        f.write(records.tobytes())


import json

def convert_trajectory(scene_dir: str, cam_name: str, out_dir: str, point_stride: int = 16, max_points: int = 200_000) -> int:
    # one trajectory to colmap mipnerf format
    scale = get_scale(scene_dir)
    positions_meters, orientations_gl = load_trajectory_poses(scene_dir, cam_name, scale)

    preview_dir = os.path.join(scene_dir, "images", f"scene_{cam_name}_final_preview")
    geometry_dir = os.path.join(scene_dir, "images", f"scene_{cam_name}_geometry_hdf5")
    img_names = sorted(fn for fn in os.listdir(preview_dir) if fn.endswith(".tonemap.jpg"))
    if not img_names:
        raise FileNotFoundError(f"no tonemap.jpg frames in {preview_dir}")

    for sub in ("images", "depths", "sparse/0", "ground_truth/depth_z", "ground_truth/normals_cv"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    with Image.open(os.path.join(preview_dir, img_names[0])) as img:
        w, h = img.size

    intrinsics = get_intrinsics(scene_dir, cam_name, w, h)
    fx, fy, cx, cy = intrinsics
    focal = 0.5 * (fx + fy)
    images_meta, frame_names, depth_frame_names = [], [], []
    c2w_list, bounds_rows = [], []
    all_pts, all_cols = [], []

    for fn in img_names:
        frame_str = fn.split(".")[1]
        i = int(frame_str)
        if i >= len(positions_meters):
            continue

        rotation_gl, translation_c2w = orientations_gl[i], positions_meters[i]
        if not (np.all(np.isfinite(rotation_gl)) and np.all(np.isfinite(translation_c2w))):
            continue

        out_name = f"frame_{frame_str}.jpg"
        src_img = os.path.join(preview_dir, fn)
        with open(src_img, "rb") as src, open(os.path.join(out_dir, "images", out_name), "wb") as dst:
            dst.write(src.read())

        rotation_c2w = rotation_gl @ np.diag([1.0, -1.0, -1.0])
        rotation_w2c = rotation_c2w.T
        translation_w2c = -rotation_w2c @ translation_c2w
        qvec = rotation_matrix_to_quaternion(rotation_w2c)

        pose_c2w = np.eye(4, dtype=np.float64)
        pose_c2w[:3, :3], pose_c2w[:3, 3] = rotation_c2w, translation_c2w

        frame_names.append(out_name)
        c2w_list.append(pose_c2w)
        images_meta.append((len(frame_names), qvec, translation_w2c, out_name))

        near, far = 0.1, 100.0
        depth_h5 = os.path.join(geometry_dir, f"frame.{frame_str}.depth_meters.hdf5")
        if os.path.exists(depth_h5):
            with h5py.File(depth_h5, "r") as f:
                z_depth = ray_depth_to_planar_z(f["dataset"][:], fx, fy, cx, cy)
            np.save(os.path.join(out_dir, "ground_truth", "depth_z", f"frame_{frame_str}.npy"), z_depth)

            z_clean = np.nan_to_num(z_depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth_mm = np.clip(z_clean * 1000.0, 0, 65535).astype(np.uint16)
            Image.fromarray(depth_mm).save(os.path.join(out_dir, "depths", f"frame_{frame_str}.png"))
            depth_frame_names.append(out_name)

            valid_z = z_depth[np.isfinite(z_depth) & (z_depth > 0.05)]
            if len(valid_z) > 0:
                near = float(np.percentile(valid_z, 0.1)) * 0.9
                far = float(np.percentile(valid_z, 99.9)) * 1.1

            with Image.open(src_img) as im:
                rgb_np = np.asarray(im.convert("RGB"))
            pts, cols = backproject_frame(rgb_np, z_depth, rotation_c2w, translation_c2w, fx, fy, cx, cy, point_stride)
            all_pts.append(pts)
            all_cols.append(cols)

        bounds_rows.append(make_poses_bounds_row(rotation_c2w, translation_c2w, h, w, focal, near, far))

        norm_h5 = os.path.join(geometry_dir, f"frame.{frame_str}.normal_cam.hdf5")
        if os.path.exists(norm_h5):
            with h5py.File(norm_h5, "r") as f:
                norm_cv = (f["dataset"][:] @ np.diag([1.0, -1.0, -1.0])).astype(np.float32)
            np.save(os.path.join(out_dir, "ground_truth", "normals_cv", f"frame_{frame_str}.npy"), norm_cv)

    pts = np.concatenate(all_pts) if all_pts else np.empty((0, 3), dtype=np.float64)
    cols = np.concatenate(all_cols) if all_cols else np.empty((0, 3), dtype=np.uint8)
    if len(pts) > max_points:
        keep = np.random.default_rng(42).choice(len(pts), max_points, replace=False)
        pts, cols = pts[keep], cols[keep]

    sparse_dir = os.path.join(out_dir, "sparse", "0")
    write_comlap_bins(sparse_dir, w, h, intrinsics, images_meta, pts, cols)

    depth_params = {
        os.path.splitext(name)[0]: {"scale": 0.001, "offset": 0.0}
        for name in depth_frame_names
    }
    with open(os.path.join(sparse_dir, "depth_params.json"), "w") as f:
        json.dump(depth_params, f, indent=2)

    poses_bounds = np.stack(bounds_rows) if bounds_rows else np.empty((0, 17), dtype=np.float64)
    np.save(os.path.join(out_dir, "poses_bounds.npy"), poses_bounds)

    gt_dir = os.path.join(out_dir, "ground_truth")
    c2w_poses = np.stack(c2w_list) if c2w_list else np.empty((0, 4, 4), dtype=np.float64)
    w2c_poses = np.linalg.inv(c2w_poses) if len(c2w_poses) > 0 else np.empty((0, 4, 4), dtype=np.float64)
    np.save(os.path.join(gt_dir, "c2w_cv.npy"), c2w_poses)
    np.save(os.path.join(gt_dir, "w2c_cv.npy"), w2c_poses)
    np.save(os.path.join(gt_dir, "K.npy"), np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]))
    np.save(os.path.join(gt_dir, "frame_names.npy"), np.array(frame_names))
    np.save(os.path.join(gt_dir, "gt_points.npy"), np.hstack([pts, cols]))

    return len(frame_names)


def prepare_hypersim_scenes(scene_dirs: list[str], out_root_dir: str, point_stride: int = 16, max_points: int = 200_000) -> None:
    # convert list of hypersim scene dirs to mip-nerf360 style
    summary = {}
    for scene_dir in map(os.path.normpath, scene_dirs):
        scene_name = os.path.basename(scene_dir)
        detail_dir = os.path.join(scene_dir, "_detail")
        cams = sorted(d for d in os.listdir(detail_dir) if d.startswith("cam_"))

        for cam_name in cams:
            if not os.path.isdir(os.path.join(detail_dir, cam_name)):
                continue
            out_dir = os.path.join(out_root_dir, scene_name, cam_name)
            key = f"{scene_name}/{cam_name}"
            summary[key] = convert_trajectory(
                scene_dir, cam_name, out_dir, point_stride, max_points
            )
            print(f"[{key}] converted {summary[key]} frames -> {out_dir}")
