# function for processing the image sets with the rgbx predictor

import os
import cv2
import sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"   # must be set before cv2 is imported

import torch, torchvision
from PIL import Image
from diffusers import DDIMScheduler
import numpy as np

here = os.getcwd()
rgb2x_path = os.path.join(here, "rgbx", "rgb2x")
sys.path.append(rgb2x_path)

from .pipeline_rgb2x import StableDiffusionAOVMatEstPipeline

PROMPTS = {
    "albedo": "Albedo (diffuse basecolor)",
    "irradiance": "Irradiance (diffuse lighting)",
}


def rgbx_predict(image_paths: list[str], cache_dir: str) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
  max_side = 1000
  seed = 2
  steps = 50

  pipeline = StableDiffusionAOVMatEstPipeline.from_pretrained(
      "zheng95z/rgb-to-x", torch_dtype=torch.float16, cache_dir=cache_dir
  ).to("cuda")
  pipeline.scheduler = DDIMScheduler.from_config(
      pipeline.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
  )
  pipeline.set_progress_bar_config(disable=True)

  albedos = []
  diffuse_shading = []
  residuals = []

  for path in image_paths:
    photo = load_ldr_image(path, from_srgb=True).to("cuda")
    old_height, old_width = photo.shape[1], photo.shape[2]

    ratio = old_height / old_width
    if old_height > old_width:
      new_height = max_side
      new_width = int(new_height / ratio)
    else:
      new_width = max_side
      new_height = int(new_width * ratio)

    new_width = new_width // 8 * 8
    new_height = new_height // 8 * 8

    photo = torchvision.transforms.Resize((new_height, new_width))(photo)

    results = {}
    for aov, prompt in PROMPTS.items():
      generator = torch.Generator(device="cuda").manual_seed(seed)   
      output = pipeline(
          prompt=prompt, photo=photo, num_inference_steps=steps,
          height=new_height, width=new_width,
          generator=generator, required_aovs=[aov],
      ).images[0][0]

      results[aov] = np.array(
          output.resize((old_width, old_height), Image.BILINEAR).convert("RGB")
      )

    albedos.append(results["albedo"])
    diffuse_shading.append(results["irradiance"])
    residuals.append(np.zeros_like(results["albedo"])) 

  return albedos, diffuse_shading, residuals

# helper code from rgbx repo

def load_ldr_image(image_path, from_srgb=False, clamp=False, normalize=False):
    # Load png or jpg image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image.astype(np.float32) / 255.0)  # (h, w, c)
    image[~torch.isfinite(image)] = 0
    if from_srgb:
        # Convert from sRGB to linear RGB
        image = image**2.2
    if clamp:
        image = torch.clamp(image, min=0.0, max=1.0)
    if normalize:
        # Normalize to [-1, 1]
        image = image * 2.0 - 1.0
        image = torch.nn.functional.normalize(image, dim=-1, eps=1e-6)
    return image.permute(2, 0, 1)  # returns (c, h, w)