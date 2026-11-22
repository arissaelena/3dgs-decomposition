# Intrinsic Decomposition of 3D Gaussian Splats with Imperfect 2D Priors
## Repository Structure

Our repository is structured as follows:
```
3dgs_decomposition/
├── data/                           # (gitignored) hypersim/ & mipnerf360/
│   ├── data_urls.txt               # urls for fetching scene zips
│   ├── download_script.sh          # bash script for downloading zips and extracting data
│   └── get_data.py                 # script to get a list of image paths for a dataset and scene
├── diffusion-renderer/             # DiffusionRenderer submodule
├── eval/                           # metrics, cross-view reprojection, recolouring
│   ├── hypersim_metrics.py         # PSNR, SI-MSE (scale-invariant) vs hypersim ground truth
│   ├── reprojection.py             # depth-warped cross-view albedo variance
│   └── recolor_scene.py            # scene recoloring script for mipnerf360 scenes
├── fusion/                         # extensions - multi-view consistency methods
│   ├── color_correction.py         # (a) per-view affine/scale color shift
│   ├── down_weight.py              # (b) pixel down-weighting via reprojection variance
│   └── gaussian_average.py         # (c) 3D gaussian weighted average target
├── intr_decomp_3dgs/               # baseline submodule
├── predictors/                     # 2D prior generation
│   ├── rgbx/ 
|   |   ├── load_image.py           # KRISTY
|   │   ├── pipeline_rgb2x.py       # KRISTY
|   │   └── rgbx.py                 # KRISTY
│   ├── colourful_iid.py            # colourful iid (paper [4])
│   ├── depth_anything.py           # depth anything V2
│   ├── diffusion_renderer.py       # diffusion renderer video model (paper [7])
│   ├── marigold.py                 # marigold (paper [6])
│   └── rgbx.py                     # rgbx (paper [5])
├── splat/                          # 3-layer gsplat / 3DGS model
│   ├── fit_baseline_gsplat.ipynb   # [KRISTY TO FILL IN]
│   ├── fit_baseline.py             # wrapper for fiting albedo, freeze, then fit shading and residual sets
│   └── train_baseline_copy.py      # training: albedo -> freeze -> shading -> residual from baseline repo
├── predictors.ipynb                # colab notebook for getting the albedo predictions
├── prepare_hypersim_data.ipynb     # colab notebook for converting the hypersim dataset into colmap mipnerf360 format
└── run_baseline.ipynb              # colab notebook for reproducing the baseline paper

```

## Pipeline

// add description of the stages in our pipeline //
1. Fetch data and run albedo predictors

## Notes on Scene Data Chosen
Will probably need to move this elsewhere and write a proper README later on but it's convenient to have any useful notes here for now. We'll be using 2 Hypersim scenes (I chose the first ones from volumes 1 and 2 as can be seen in the download script), and 2 outdoor (bicycle and garden) and 1 indoor (room) scenes from the Mip-NeRF 360 set.

For the Mip-NeRF set, there's the images, but also images_2, images_4, and images_8, we should probably try using the _4 or _8 sets just in terms of memory utilisation and time processing would take.

## Running Colourful Diffuse IID Albedo Predictor

You should create a fresh virtual environment (`python3 -m venv intrinsic_env`) and start it (`source intrinsic_env/bin/activate`)

You can install the repository as a package using pip:

```
git clone https://github.com/compphoto/Intrinsic
cd Intrinsic
pip install .
```

This will allow you to import the repository as a Python package, and use the pipeline as part of our codebase.
