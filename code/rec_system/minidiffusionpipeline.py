import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

class MiniDiffusionPipeline:
    def __init__(self, model_steps: int, mock=False):
        if not mock:
            self.inference_steps = model_steps
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                f"https://huggingface.co/ByteDance/SDXL-Lightning/blob/main/sdxl_lightning_{self.inference_steps}step.safetensors",
                torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
            self.pipe.enable_model_cpu_offload()
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config,
                                                                     timestep_spacing="trailing")
            self.text2img = self.txt2imgreal
        else:
            self.text2img = self.txt2imgmock
        # scheduler for sdxl lightning
        
    def txt2imgreal(self, prompt):
        return self.pipe(prompt, num_inference_steps=self.inference_steps, guidance_scale=0).images[0]

    def txt2imgmock(self, prompt):
        return "https://fal-cdn.batuhan-941.workers.dev/files/koala/-CQBCeIxrvPqrvt4FDY5n.jpeg"