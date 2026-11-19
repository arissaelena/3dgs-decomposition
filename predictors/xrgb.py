import os
import sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"   
import argparse, glob, sys
import torch, torchvision
from PIL import Image
from diffusers import DDIMScheduler
import numpy as np

here = os.getcwd()   
rgb2x_path = os.path.join(here, "rgbx", "rgb2x")
sys.path.append(rgb2x_path) # TODO: change this to wherever rgb2x's path is 

sys.path.append(rgb2x_path)
from load_image import load_ldr_image
from pipeline_rgb2x import StableDiffusionAOVMatEstPipeline

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
    photo =  load_ldr_image(path, from_srgb=True).to("cuda")
    old_height = photo.shape[1]
    old_width = photo.shape[2]
    new_height = old_height
    new_width = old_width
    radio = old_height / old_width
    max_side = 1000
    if old_height > old_width:
        new_height = max_side
        new_width = int(new_height / radio)
    else:
        new_width = max_side
        new_height = int(new_width * radio)

    if new_width % 8 != 0 or new_height % 8 != 0:
        new_width = new_width // 8 * 8
        new_height = new_height // 8 * 8


    photo = torchvision.transform.Resize(new_h, new_w)(photo)

  required_aovs = ["albedo", "irradiance"]
  prompts = {"albedo": "Albedo (diffuse basecolor)",
             "irradiance": "Irradiance (diffuse lighting)"
             }


  results = {}
  for aov, prompt in prompts.items():
    generator = torch.Generator(device="cuda").manual_seed(seed)
    output = pipeline(prompt=prompt, photo=photo, num_inference_steps=steps,
                height=new_height,
                width=new_width,
                generator=generator,
                required_aovs=[aov]
            ).images[0][0]

    results[aov] = np.array(
                output.resize((w, h), Image.BILINEAR).convert("RGB")
            )

    albedos.append(results["albedo"])
    diffuse_shading.append(results["irradiance"])
    residuals.append(np.zeros_like(results["albedo"]))

  return albedos, diffuse_shading, residuals