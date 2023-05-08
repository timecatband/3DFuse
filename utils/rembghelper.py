from rembg import remove
import sys
input_dir = sys.argv[1]
output_dir = sys.argv[2]

def rem_bg(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)

import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Load each file in input_dir and remove the background
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        rem_bg(input_path, output_path)
    
