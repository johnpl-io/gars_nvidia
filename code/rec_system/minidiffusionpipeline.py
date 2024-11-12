import queue
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from PIL import Image
import cv2

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

    def __init__(self, model_steps: int, mock: bool=False):
        """
        Initializes the MiniDiffusionPipeline with specified model steps and mode (real or mock).

        Args:
            model_steps (int): The number of inference steps for the diffusion model.
            mock (bool): If True, uses a mock function for text-to-image generation, else uses the real model.
        """
        if not mock:
            # Set inference steps and initialize the Stable Diffusion pipeline with specified steps
            self._inference_steps = model_steps
            self._pipe = StableDiffusionXLPipeline.from_single_file(
                f"https://huggingface.co/ByteDance/SDXL-Lightning/blob/main/sdxl_lightning_{self._inference_steps}step.safetensors",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )

            # Enable CPU offload to optimize memory usage during inference
            self._pipe.enable_model_cpu_offload()

            # Set up the scheduler for handling model timesteps with Euler discretization
            self._pipe.scheduler = EulerDiscreteScheduler.from_config(
                self._pipe.scheduler.config, timestep_spacing="trailing"
            )

            # Set real text-to-image function
            self._text2img = self._txt2imgreal
            self._latent_img = []
            self.queue = queue.Queue()
            self._current_step = 0
        else:
            # Use mock function for text-to-image when mock mode is enabled
            self._text2img = self._txt2imgmock

    def _txt2imgreal(self, prompt: str) -> Image.Image:
        """
        Generates an image from the provided text prompt using the real Stable Diffusion pipeline.

        Args:
            prompt (str): The text prompt for generating the image.

        Returns:
            PIL.Image.Image: The generated image.
        """
        return self._pipe(
            prompt, num_inference_steps=self._inference_steps, guidance_scale=0,
        callback_on_step_end = self._decode_tensors,
        callback_on_step_end_tensor_inputs = ["latents"],
        ).images[0]

    def _latents_to_rgb(self, latents):
        weights = (
            (60, -60, 25, -70),
            (60, -5, 15, -50),
            (60, 10, -5, -35)
        )
        weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
        biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
        rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(
            -1).unsqueeze(-1)
        image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
        image_array = image_array.transpose(1, 2, 0)
        return cv2.resize(image_array, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)

    def _decode_tensors(self, pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        image = self._latents_to_rgb(latents)
        self.queue.put(image)
        self._current_step = step
        return callback_kwargs

    def _txt2imgmock(self, prompt: str) -> str:
        """
        Returns a placeholder image URL for mock functionality.

        Args:
            prompt (str): The text prompt for generating the mock image.

        Returns:
            str: A URL pointing to a placeholder image.
        """
        return "https://fal-cdn.batuhan-941.workers.dev/files/koala/-CQBCeIxrvPqrvt4FDY5n.jpeg"

