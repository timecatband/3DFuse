import PIL
from PIL import Image
import os
import sys
import cv2
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
import torch
import numpy as np
# Import PIL image ops
from PIL import ImageOps

mask_color = (1.0, 0.0, 0.0)

text="deep space. beautiful digital painting. 8k. ultra detailed. abstract. spaceships. high resolution"
class Inpainter():
    def __init__(self):
        self.pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",
            revision="fp16",
            torch_dtype=torch.float16,
        ).to("cuda")
        #self.pipe.safety_checker = lambda images, a: images, False

    def dilate_mask(mask):
        # mask is np.array convert to image
        mask = Image.fromarray(mask)

        mask = mask.convert("L")
        mask = np.array(mask)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Create a square structuring element for dilation
        kernel = np.ones((5, 5), np.uint8)
        
        # Dilate the mask
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        
        return dilated_mask

    def inpaint_image(self,image):
        img_array = np.array(image)
        mask_array = np.zeros_like(img_array)

        # Identify all pure red pixels (255, 0, 0)
        red_pixels = (img_array[:, :, 0] == 255) & (img_array[:, :, 1] == 0) & (img_array[:, :, 2] == 0)

        # Set the mask pixels to white (255, 255, 255) for the identified red pixels
        mask_array[red_pixels] = (255, 255, 255)

        # Convert the NumPy array back to a PIL image
        mask = Image.fromarray(mask_array)


        image = image.convert("RGB")
        
        # Run diffusion inpainting on image with mask
        inpainted_image = self.pipe(prompt=text,
                        negative_prompt="watermark. logo. text. people. land. corrupted. glitchy. flat. painting, blotchy pixelated. warped. distorted. ugly. photo",
                        guidance_scale=8.0,
                        image=image, mask_image=mask)

        output_image = inpainted_image.images[0]
        # Correct output image colorspace
        output_image = output_image.convert("RGB")
        # Replace output image with original image where mask is empty
        mask=mask.convert("L")
        # Invert mask
        mask = ImageOps.invert(mask)
        output_image.paste(image, mask=mask)
        return output_image
    
def inpaint_patch(inpainter, image, x, y):
    # Crop a 512x512 patch from the image
    patch = image.crop((x - 256, y - 256, x + 256, y + 256))
    # Inpaint the patch
    patch = inpainter.inpaint_image(patch)
    # Paste the inpainted patch back into the image
    image.paste(patch, (x - 256, y - 256))
    return image

CANVAS_SIZE = 2048
# Create an empty RGBA image of size 8192
canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (255, 0, 0))
x = 768
y = 768
x_diff = 128
y_diff = 128
inpainter = Inpainter()
def inpaint_direction(x_diff, y_diff):
    for j in range(0,3):
        for i in range(0, 3):
            x = min(CANVAS_SIZE-256, int(CANVAS_SIZE/2)+x_diff*i)
            y = min(CANVAS_SIZE-256, int(CANVAS_SIZE/2)+y_diff*j)
            image = inpaint_patch(inpainter, canvas, x, y)
        image.save("inpainted.png")
    return image

def dummy(images, **kwargs):
    return images, False
inpainter.pipe.safety_checker = dummy

image = inpaint_direction(128,128)
image = inpaint_direction(128,-128)
image = inpaint_direction(-128,128)
image = inpaint_direction(-128,-128)

# Save image
image.save("inpainting.png")

