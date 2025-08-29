import requests
import base64
import os
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image

load_dotenv()
api_key = os.getenv("API_KEY")
url = os.getenv("URL")


def image_generation(image, masked_image, prompt):
  # Request payload
  data = {
    "image": pil_image_to_base64(image),  # Or use image_file_to_base64("IMAGE_PATH")
    "mask": pil_image_to_base64(Image.fromarray(masked_image)),  # Or use image_file_to_base64("IMAGE_PATH")
    "prompt": prompt,
    "negative_prompt": "blur",
    "samples": 1,
    "scheduler": "DDIM",
    "num_inference_steps": 25,
    "guidance_scale": 7.5,
    "seed": 12467,
    "strength": 0.9,
    "base64": False
  }

  headers = {'x-api-key': api_key}

  response = requests.post(url, json=data, headers=headers)
  return response.content  # The response is the generated image

def pil_image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


