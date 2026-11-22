# PSNR, SI-MSE, etc. that can be used since we have hypersim ground truths

import torchmetrics # if using in colab, make sure to run !pip install torchmetrics
import cv2
import os
import h5py
from PIL import Image
import numpy as np
import torch
import lpips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

psnr_loss_fn = torchmetrics.image.PeakSignalNoiseRatio(data_range=1).to(device)
lpips_loss_fn = lpips.LPIPS(net='vgg').to(device).eval()
ssim_loss_fn = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1).to(device)

def predictor_evaluation_from_data(albedo_gt_hdf5_paths: list[str], predicted_albedo_paths: list[str], save_dir: str):
    os.makedirs(os.path.join(save_dir, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'predicted'), exist_ok=True)

    gt_by_stem = {
        ".".join(os.path.basename(p).split(".")[:2]): p
        for p in albedo_gt_hdf5_paths
    }

    pred_by_stem = {
        ".".join(os.path.basename(p).split(".")[:2]): p
        for p in predicted_albedo_paths
    }

    psnrs, lpipss, ssims = [], [], []

    for stem in sorted(gt_by_stem.keys() & pred_by_stem.keys()):
        gt_path = gt_by_stem[stem]
        pred_path = pred_by_stem[stem]
        gt_out = os.path.join(save_dir, 'gt', f'{stem}.png')
        pred_out = os.path.join(save_dir, 'predicted', f'{stem}.png')
        with h5py.File(gt_path, 'r') as file:
            gt = file['dataset'][:].astype(np.float32)
            gt = np.nan_to_num(gt, nan=0.0, posinf=1.0, neginf=0.0)
            gt = np.clip(gt, 0.0, 1.0) ** (1.0 / 2.2)
            gt = cv2.cvtColor((gt * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(gt_out, gt)

            pred = np.load(pred_path).astype(np.float32)
            if pred.ndim == 3 and pred.shape[0] == 3: 
                pred = np.transpose(pred, (1, 2, 0))
            if pred.min() < 0: 
                pred = (pred + 1.0) / 2.0
            if pred.shape[:2] != gt.shape[:2]: 
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
            pred = np.clip(pred, 0.0, 1.0)
            pred = cv2.cvtColor((pred * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(pred_out, pred)

            psnrs.append(compute_psnr(pred_out, gt_out, True))
            lpipss.append(compute_lpips(pred_out, gt_out, True))
            ssims.append(compute_ssim(pred_out, gt_out, True))

    return psnrs, lpipss, ssims

# the input images must have matching shapes, and be in the range [0, 1]
def compute_psnr(result: str, original: str, scale_invariant: bool=False):
    result, original = load_images(result, original, scale_invariant)
    
    with torch.no_grad():
        return psnr_loss_fn(result, original).item()

def compute_lpips(result: str, original: str, scale_invariant: bool=False):
    result, original = load_images(result, original, scale_invariant)

    with torch.no_grad():
        return lpips_loss_fn(result, original, normalize=True).item()

def compute_ssim(result: str, original: str, scale_invariant: bool=False):
    result, original = load_images(result, original, scale_invariant)

    with torch.no_grad():
        return ssim_loss_fn(result, original).item()

def load_images(result_path: str, original_path: str, scale_invariant: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
    result = cv2.cvtColor(cv2.imread(result_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    result = torch.from_numpy(result).permute(2, 0, 1).unsqueeze(0).to(device)
    original = cv2.cvtColor(cv2.imread(original_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    original = torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0).to(device)

    if scale_invariant:
        alpha = (result * original).sum() / ((result ** 2).sum() + 1e-8)
        result = torch.clamp(result * alpha, 0.0, 1.0)

    return result, original