# produce depth maps for the inputs using depth anything v2 via hugging face
import io
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def get_depth_map(img_path: str)-> Image:
    # uses depth anything v2 via huggingface

    with open(img_path, "rb") as f:
        img = Image.open(io.BytesIO(f.read()))

        image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", device_map="auto")

        input = image_processor(images=img, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**input)

        # interpolate to resize to original size
        post_processed_output = image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(img.height, img.width)],
        )
        
        # convert float depths into colours to be displayed in an image
        predicted_depth = post_processed_output[0]["predicted_depth"]
        return predicted_depth