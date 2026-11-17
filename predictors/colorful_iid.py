# TODO: write function for processing the image sets with the predictor and get normalised predicted values
from chrislib.data_util import load_image
from intrinsic.pipeline import load_models, run_pipeline

def colorful_iid_predict(images):
    models = load_models('v2')
    albedos = []
    diffuse_shading = []
    residuals = []
    for image in images:
        results = run_pipeline(models, load_image(image))
        albedos.append(results['hr_alb'])
        diffuse_shading.append(results['dif_shd'])
        residuals.append(results['residual'])
    return albedos, diffuse_shading, residuals