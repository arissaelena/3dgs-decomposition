# produce depth maps for the inputs using depth anything v2 via hugging face

import io
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def get_depth_map(images: list[str]) -> list[torch.Tensor]:
    # images: list of str which are paths to image
    # returns predicted depth tensor
    # uses depth anything v2 via huggingface
    predicted_depths = []
    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", device_map="auto")

    for img_path in images:
        with open(img_path, "rb") as f:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")

            input = image_processor(images=img, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(**input)

            # interpolate to resize to original size
            post_processed_output = image_processor.post_process_depth_estimation(
                outputs,
                target_sizes=[(img.height, img.width)],
            )
            
            
            predicted_depth = post_processed_output[0]["predicted_depth"]
            predicted_depths.append(predicted_depth)
    
    return predicted_depths