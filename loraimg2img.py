import adapt_sd
import requests
import torch
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
import sys

LORA_WEIGHTS = "results/house/lora/final_lora.safetensors"
def load_lora():
    model = adapt_sd.load_3DFuse_no_control(LORA_WEIGHTS, 0.3)
    # Disable safety checker
    model.safety_checker = None
    return model

pipe = load_lora()

device = "cuda"
prompt = sys.argv[1]

negative_prompt = "blurry. corrupted. distorted. glitch"

dir = sys.argv[2]
# Load each image in dir
import os
for filename in os.listdir(dir):
    if filename.endswith(".png"):
        print(filename)
        init_image = Image.open(os.path.join(dir, filename))
        print(filename)
        images = pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_image, strength=0.35, guidance_scale=12.5).images
        images[0].save(os.path.join("denoise-test", filename))

# Load each image in "denoise-test" and stitch them in to a video
import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob('denoise-test/*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
# Write images to video
for i in range(len(img_array)):
    out.write(img_array[i])
# Save video
out.release()
