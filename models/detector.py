from networkx import draw
import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import ImageDraw, ImageFont

# Load a larger font (fallback to default if arial.ttf not found)
try:
    font = ImageFont.truetype("arial.ttf", 80)  # 80px font size
except:
    font = ImageFont.load_default()


def YOLOObjectDetection(image):
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    ckpt = "yainage90/fashion-object-detection-yolos-tiny"
    image_processor = YolosImageProcessor.from_pretrained(ckpt)
    model = YolosForObjectDetection.from_pretrained(ckpt).to(device)

    with torch.no_grad():
        inputs = image_processor(images=[image], return_tensors="pt")
        outputs = model(**inputs.to(device))
        target_sizes = torch.tensor([[image.size[1], image.size[0]]])
        results = image_processor.post_process_object_detection(
            outputs, threshold=0.85, target_sizes=target_sizes
        )[0]

    # Draw boxes on a copy of the image
    draw = ImageDraw.Draw(image)
    boxes = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        score = round(score.item(), 3)
        label = model.config.id2label[label.item()]
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1] - 10), f"{label} {score}", fill="black", font=font)
        boxes.append(box)

    return image, boxes[0]  # Returns image and first bounding box

