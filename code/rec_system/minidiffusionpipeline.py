import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler, StableDiffusionPipeline

class MiniDiffusionPipeline:
    def __init__(self, model_steps: int):
        self.inference_steps = model_steps
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            f"https://huggingface.co/ByteDance/SDXL-Lightning/blob/main/sdxl_lightning_{self.inference_steps}step.safetensors",
            torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        self.pipe.enable_model_cpu_offload()
        # scheduler for sdxl lightning
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
    def txt2img(self, prompt):
        return self.pipe(prompt, num_inference_steps=self.inference_steps, guidance_scale=0).images[0]
MiniDiffusionPipeline(4)