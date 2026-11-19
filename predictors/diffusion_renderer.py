# function for processing the image sets with diffusion renderer to get normalised predicted values

import diffusers
import torch
import numpy as np
from PIL import Image

def diffusion_renderer_predict(image_paths: list[str]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    
    albedos = []
    diffuse_shading = []
    residuals = []

    for path in image_paths:
        pass
        # finish this

    return albedos, diffuse_shading, residuals