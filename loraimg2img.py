import adapt_sd
import requests
import torch
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
import sys

LORA_WEIGHTS = sys.argv[1]
def load_lora():
    model = adapt_sd.load_3DFuse_no_control(LORA_WEIGHTS, 0.4)
    # Disable safety checker
    model.safety_checker = None
    return model

pipe = load_lora()

device = "cuda"
prompt = sys.argv[2]

dir = sys.argv[3]
# Load each image in dir
negative_prompt = "glitch. pixelated. distorted. ugly"
import os
for filename in os.listdir(dir):
    if filename.endswith(".png"):
        print(filename)
        init_image = Image.open(os.path.join(dir, filename))
        print(filename)
        images = pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_image, strength=0.8, guidance_scale=8, num_inference_steps=50).images
        output_image = images[0]
        # Blend input and output image
        #output_image = Image.blend(init_image, output_image, 0.5)
        images[0].save(os.path.join("denoise-test", filename))
        #output_image.save(os.path.join("denoise-test", "blend-" + filename))


# Load each image in "denoise-test" and stitch them in to a video
import cv2
import numpy as np
import glob

img_array = []
filenames = []
for filename in glob.glob('denoise-test/*.png'):
    filenames.append(filename)
# Sort filenames
filenames.sort()
for filename in filenames:
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
