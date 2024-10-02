import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler, StableDiffusionPipeline
from IPython.display import display
from PIL import Image

# Load the pipeline
pipe = StableDiffusionXLPipeline.from_single_file(
    "https://huggingface.co/ByteDance/SDXL-Lightning/blob/main/sdxl_lightning_8step.safetensors",
    torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
)
pipe.enable_model_cpu_offload()

# Set the scheduler
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

# Define the callback function
def callback(iter, t, latents):
    # Convert latents to image
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = pipe.vae.decode(latents).sample
        image.save
# Generate the image with the callback
pipe("test", num_inference_steps=8, guidance_scale=0, callback=callback, callback_steps=8)