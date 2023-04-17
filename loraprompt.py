import adapt_sd
import torch
import sys


LORA_WEIGHTS = sys.argv[1]
def load_lora():
    model = adapt_sd.load_3DFuse_no_control(LORA_WEIGHTS, 0.3, img2img=false)
    # Disable safety checker
    model.safety_checker = None
    return model

pipe = load_lora()

device = "cuda"
prompt = sys.argv[2]
images = pipe(prompt=prompt, strength=0.2, guidance_scale=7.5).images
images[0].save("out.png")