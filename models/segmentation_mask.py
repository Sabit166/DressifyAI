from ultralytics import SAM
import numpy as np
# Load a model
model = SAM("sam2.1_b.pt")

# Display model information (optional)
# model.info()

def segmentation_mask(image, box):
    segment_result = model(image,bboxes=box)
    masks = segment_result[0].masks.data.cpu().numpy()  # (N, H, W)
    binary_mask = (masks > 0.5).astype(np.uint8) * 255  # convert to 0/255
    # Return only the first mask for display
    return binary_mask[0]