# down-weight pixels where predictions disagree across views
# add baseline repo to path
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root, "intr_decomp_3dgs"))

from intr_decomp_3dgs.scene import Camera
import torch



def generate_confidence_maps(vp: Camera) -> torch.Tensor:
    # take scene (for all the cameras), and the camera/viewpoint
    # get cameras within certain distance
    # warp them to origina;
    # find differences between them


    # weight map needs to be a torch.Tensor with same: 
    # device and dtype of image and gt_image
    # shape (1, H, W)
    # generated inside torch no grad block
    return weight_map

    pass