# function for processing the image sets with marigold and get normalised predicted values

import diffusers
import torch
import numpy as np
from PIL import Image

def marigold_predict(image_paths: list[str]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    mi_pipeline = diffusers.MarigoldIntrinsicsPipeline.from_pretrained(
        "prs-eth/marigold-iid-lighting-v1-1", variant="fp16", torch_dtype=torch.bfloat16
        ).to("cuda")
    
    albedos: list[np.ndrray[np.uint8]] = []
    diffuse_shading: list[np.ndrray[np.uint8]] = []
    residuals: list[np.ndrray[np.uint8]] = []

    for path in image_paths:
        image = diffusers.utils.load_image(path)
        intrinsics = mi_pipeline(image)

        vis: list[dict[str, Image.Image]] = mi_pipeline.image_processor.visualize_intrinsics(intrinsics.prediction, mi_pipeline.target_properties)

        albedos.append(np.array(vis[0]["albedo"]))
        diffuse_shading.append(np.array(vis[0]["shading"]))
        residuals.append(np.array(vis[0]["residual"]))
    

    # normalise to [0,1] to use the same printing function as colourful iid
    norm_albedos = [x.astype(np.float32) / 255.0 for x in albedos]
    norm_ds = [x.astype(np.float32) / 255.0 for x in diffuse_shading]
    norm_residuals = [x.astype(np.float32) / 255.0 for x in residuals]

    return norm_albedos, norm_ds, norm_residuals