import numpy as np
import lpips
import torch

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_LPIPS_FN = lpips.LPIPS(net="vgg", spatial=True).to(_DEVICE).eval()


def compute_mv_metrics(albedo_src: np.ndarray, warped_tgt: np.ndarray, m: np.ndarray) -> dict[str, float | int]:
    mod_m = int(m.sum())
    if mod_m == 0:
        return {"overlap": 0.0, "mod_m": 0}

    a_1 = albedo_src[m]
    a_2_1 = warped_tgt[m]

    # compare raw pred srgb vals across warped views
    mse = float(np.mean((a_1 - a_2_1) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(a_1 - a_2_1)))
    psnr = float(-10.0 * np.log10(max(mse, 1e-10)))
    lpips_val = masked_lpips(albedo_src, warped_tgt, m)

    return {
        "overlap": float(m.mean()),
        "mod_m": mod_m,
        "rmse": rmse,
        "mae": mae,
        "psnr": psnr,
        "lpips": lpips_val,
    }


def masked_lpips(albedo_src: np.ndarray, full_warped: np.ndarray, m: np.ndarray) -> float:
    # lpips over just the area that m says is valid in both
    min_crop = 64
    h, w = m.shape
    ys, xs = np.where(m)
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1

    if (y1 - y0) < min_crop:
        pad_y = (min_crop - (y1 - y0) + 1) // 2
        y0 = max(0, y0 - pad_y)
        y1 = min(h, y0 + min_crop)
    if (x1 - x0) < min_crop:
        pad_x = (min_crop - (x1 - x0) + 1) // 2
        x0 = max(0, x0 - pad_x)
        x1 = min(w, x0 + min_crop)

    src_np = albedo_src[y0:y1, x0:x1]
    tgt_np = full_warped[y0:y1, x0:x1]
    src_trch = torch.from_numpy(src_np).permute(2, 0, 1).unsqueeze(0).to(_DEVICE)
    tgt_trch = torch.from_numpy(tgt_np).permute(2, 0, 1).unsqueeze(0).to(_DEVICE)

    mask_crop = m[y0:y1, x0:x1]
    with torch.no_grad():
        pd_map = _LPIPS_FN(tgt_trch, src_trch, normalize=True).squeeze().cpu().numpy()
    return float(pd_map[mask_crop].mean())