from lora_diffusion.cli_lora_pti import train
import os
import argparse
from argparse import ArgumentParser

def train_lora(args):
    exp_dir = args.exp_dir
    ti_step = args.ti_step
    pt_step = args.pt_step
    initial = args.initial
    prompt = args.prompt

    instance_dir=os.path.join(exp_dir,'initial_image')
    weight_dir=os.path.join(exp_dir,'lora')
    if initial=="":
        initial=None
    modelfile = os.path.join(weight_dir, "final_lora.safetensors")
    if os.path.exists(modelfile):
        # If the model exists, we don't need to train it again
        print("Model exists, skipping training")
        return

    # Load the weights if they already exist
    print("Weight dir" + weight_dir)
    
    train(pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5',\
          instance_data_dir=instance_dir,output_dir=weight_dir,gradient_checkpointing=True,\
          scale_lr=True,lora_rank=1,cached_latents=False,save_steps=ti_step,\
          max_train_steps_ti=ti_step,max_train_steps_tuning=pt_step, use_template="object",\
          lr_warmup_steps=0, lr_warmup_steps_lora=100, placeholder_tokens="<0>", initializer_tokens=initial,\
          continue_inversion=True, continue_inversion_lr=1e-4,device="cuda:0",
          enable_xformers_memory_efficient_attention=True
          )



if __name__ == "__main__":
    parser = ArgumentParser(description="Semantic Coding")
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory")
    parser.add_argument("--ti_step", type=int, required=False, default=800, help="TI step")
    parser.add_argument("--pt_step", type=int, required=False, default=800, help="PT step")
    parser.add_argument("--initial", type=str, required=False, default="", help="Initial image")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt")

    args = parser.parse_args()
    train_lora(args)
