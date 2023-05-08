import torch
import sys
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def load_sd():
    model_id = "stabilityai/stable-diffusion-2"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    return pipe

pipe = load_sd()

image = pipe(sys.argv[1]).images[0]
image.save(sys.argv[2])
