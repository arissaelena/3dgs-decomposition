# Intrinsic Decomposition of 3D Gaussian Splats with Imperfect 2D Priors
## Repository Structure

// explain repo structure //

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
