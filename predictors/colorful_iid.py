# function for processing the image sets with the predictor and get normalised predicted values

from chrislib.data_util import load_image
from intrinsic.pipeline import load_models, run_pipeline
import numpy as np

def colorful_iid_predict(images: list[str]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    models = load_models('v2')

    albedos = []
    diffuse_shading = []
    residuals = []

    # loop over the images in the scene and run the predictor
    for image in images:
        results = run_pipeline(models, load_image(image))

        # we only need the values computed for the three priors used for 3DGS
        albedos.append(results['hr_alb'])
        diffuse_shading.append(results['dif_shd'])
        residuals.append(results['residual'])

    return albedos, diffuse_shading, residuals