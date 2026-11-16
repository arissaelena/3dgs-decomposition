import os

# select images used by the albedo predictors
def get_data_for_predictors(dataset='hypersim', scene='ai_001_001', root_dir=''):
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
