import adapt_sd
import torch
import sys


LORA_WEIGHTS = sys.argv[1]
def load_lora():
    model = adapt_sd.load_3DFuse_no_control(LORA_WEIGHTS, 0.3, img2img=False)
    # Disable safety checker
    model.safety_checker = None
    return model

pipe = load_lora()

device = "cuda"
prompt = sys.argv[2]
for i in range(5):
    images = pipe(prompt=prompt, negative_prompt="blurry").images
    images[0].save(f"lora/instance{i}.png")