import os

# select images used by the albedo predictors
def get_data_for_predictors(dataset: str ='hypersim', scene: str ='ai_001_001', root_dir: str ='') -> list[str]:
    image_paths = []

    # need to specify the path to the project root from wherever the function is called
    path = root_dir + 'data/'

    if dataset == 'hypersim':
        path = path + dataset + '/' + scene + '/images/scene_cam_00_final_preview/'
        # sorting the frames so they are in sequential order for the sliding window effect
        image_paths = sorted([path + image for image in os.listdir(path) if image.endswith(('.tonemap.jpg'))])
    else:
        # for mipnerf360, we'll experiment with the lowest resolution first for better memory and compute speed 
        # keep in mind there is the option to use one of the higher resolution sets
        path = path + dataset + '/' + scene + '/images_8/'
        image_paths = sorted([path + image for image in os.listdir(path) if image.endswith(('.JPG'))])

    return image_paths

def get_paths_for_predictor_evaluation(predictor: str, scene: str ='ai_001_001', root_dir: str ='') -> tuple[list[str], list[str]]:
    gt_paths = []
    pred_paths = []

    path = os.path.join(root_dir, 'data', 'hypersim', scene, 'images', 'scene_cam_00_final_hdf5')
    gt_paths = sorted([os.path.join(path, image) for image in os.listdir(path) if image.endswith(('.diffuse_reflectance.hdf5'))])
    path = os.path.join(root_dir, predictor, 'hypersim', scene, 'albedos')
    pred_paths = sorted([os.path.join(path, image) for image in os.listdir(path) if image.endswith(('.npy'))])

    return gt_paths, pred_paths