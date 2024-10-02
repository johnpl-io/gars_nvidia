import torch

from diffusers import StableDiffusionXLImg2ImgPipeline,  UniPCMultistepScheduler

from diffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(

    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16

)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.enable_model_cpu_offload()

url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"

init_image = load_image(url).convert("RGB")

prompt = "a photo of an astronaut riding a horse on mars"

image = pipe(prompt, image=init_image).images[0]