
#GAN模型
from PIL import Image
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)
device="cuda"
model_id = "./Taiyi-Stable-Diffusion-1B-Chinese-v0.1"

pipe_text2img = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
pipe_img2img = StableDiffusionImg2ImgPipeline(**pipe_text2img.components).to(device)
def infer_text2img(prompt, guide, steps, width, height, image_in, strength):
    if image_in is not None:
        init_image = image_in.convert("RGB").resize((width, height))
        output = pipe_img2img(prompt, image=init_image, strength=strength, guidance_scale=guide, num_inference_steps=steps)
    else:
        output = pipe_text2img(prompt, width=width, height=height, guidance_scale=guide, num_inference_steps=steps,)
    image = output.images[0]
    return image