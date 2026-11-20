# PSNR, SI-MSE, etc. that can be used since we have hypersim ground truths

import torchmetrics # if using in colab, make sure to run !pip install torchmetrics
import cv2
import numpy as np
import torch
import lpips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

psnr_loss_fn = torchmetrics.image.PeakSignalNoiseRatio(data_range=1).to(device)
lpips_loss_fn = lpips.LPIPS(net='vgg').to(device).eval()
ssim_loss_fn = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1).to(device)

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