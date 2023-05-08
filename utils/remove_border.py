import os
from PIL import Image

def remove_border(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            img = Image.open(image_path)

            width, height = img.size

            left = int(width * 0.25)
            upper = int(height * 0.25)
            right = int(width * 0.75)
            lower = int(height * 0.75)

            # Convert the image to RGBA if it isn't
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            # Create a transparent image of the same size
            transparent_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))

            # Paste the cropped image onto the transparent image
            transparent_img.paste(img.crop((left, upper, right, lower)), (left, upper))

            output_path = os.path.join(output_folder, filename)
            transparent_img.save(output_path)

import sys
input_folder = sys.argv[1]
output_folder = sys.argv[2]

remove_border(input_folder, output_folder)
