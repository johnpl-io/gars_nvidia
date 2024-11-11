import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler


class MiniDiffusionPipeline:
    """
    A simplified wrapper for the Stable Diffusion XL pipeline for generating images from text prompts.

    This class offers both real and mock functionality, depending on the `mock` parameter.
    When `mock` is False, it initializes the Stable Diffusion pipeline with specified model steps
    for actual image generation. When `mock` is True, it uses a mock function that returns a placeholder URL.

    Attributes:
        inference_steps (int): The number of steps for model inference in real mode.
        pipe (StableDiffusionXLPipeline): The pipeline for generating images with Stable Diffusion.
        text2img (function): The method for text-to-image generation, either real or mock.
    """

    def __init__(self, model_steps: int, mock=False):
        """
        Initializes the MiniDiffusionPipeline with specified model steps and mode (real or mock).

        Args:
            model_steps (int): The number of inference steps for the diffusion model.
            mock (bool): If True, uses a mock function for text-to-image generation, else uses the real model.
        """
        if not mock:
            # Set inference steps and initialize the Stable Diffusion pipeline with specified steps
            self.inference_steps = model_steps
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                f"https://huggingface.co/ByteDance/SDXL-Lightning/blob/main/sdxl_lightning_{self.inference_steps}step.safetensors",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )

            # Enable CPU offload to optimize memory usage during inference
            self.pipe.enable_model_cpu_offload()

            # Set up the scheduler for handling model timesteps with Euler discretization
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config, timestep_spacing="trailing"
            )

            # Set real text-to-image function
            self.text2img = self.txt2imgreal
        else:
            # Use mock function for text-to-image when mock mode is enabled
            self.text2img = self.txt2imgmock

    def txt2imgreal(self, prompt: str):
        """
        Generates an image from the provided text prompt using the real Stable Diffusion pipeline.

        Args:
            prompt (str): The text prompt for generating the image.

        Returns:
            PIL.Image.Image: The generated image.
        """
        return self.pipe(
            prompt, num_inference_steps=self.inference_steps, guidance_scale=0
        ).images[0]

    def txt2imgmock(self, prompt: str):
        """
        Returns a placeholder image URL for mock functionality.

        Args:
            prompt (str): The text prompt for generating the mock image.

        Returns:
            str: A URL pointing to a placeholder image.
        """
        return "https://fal-cdn.batuhan-941.workers.dev/files/koala/-CQBCeIxrvPqrvt4FDY5n.jpeg"
