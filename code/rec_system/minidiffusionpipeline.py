import queue
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from PIL import Image


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

    def __init__(self, model_steps: int, mock: bool = False):
        """
        Initializes the MiniDiffusionPipeline with specified model steps and mode (real or mock).

        Args:
            model_steps (int): The number of inference steps for the diffusion model.
            mock (bool): If True, uses a mock function for text-to-image generation, else uses the real model.
        """
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
        self._latent_img = []
        self.is_gen_queue = queue.Queue()
        self.latent_queue = queue.Queue()
        self._current_step = 0

    def text2img(self, prompt: str) -> Image.Image:
        """
        Generates an image from the provided text prompt using the real Stable Diffusion pipeline.

        Args:
            prompt (str): The text prompt for generating the image.

        Returns:
            PIL.Image.Image: The generated image.
        """

        return self._pipe(
            prompt,
            num_inference_steps=self._inference_steps,
            guidance_scale=0,
            callback_on_step_end=self._decode_tensors,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]

    # Thanks to https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space
    # which we use here to help decode latents

    def _latents_to_rgb(self, latents: torch.Tensor) -> Image.Image:
        weights = ((60, -60, 25, -70), (60, -5, 15, -50), (60, 10, -5, -35))
        weights_tensor = torch.t(
            torch.tensor(weights, dtype=latents.dtype).to(latents.device)
        )
        biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(
            latents.device
        )
        rgb_tensor = torch.einsum(
            "...lxy,lr -> ...rxy", latents, weights_tensor
        ) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
        image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
        # Assuming image_array is a numpy array with shape (C, H, W)
        image_array = image_array.transpose(1, 2, 0)  # Transpose to (H, W, C)

        # Convert numpy array to PIL Image
        image = Image.fromarray(image_array)

        # Resize image to 1024x1024 using PIL
        resized_image = image.resize((1024, 1024), Image.BICUBIC)

        return resized_image

    def _decode_tensors(
        self, pipe: StableDiffusionXLPipeline, step: int, timestep: int, callback_kwargs
    ):
        latents = callback_kwargs["latents"]
        image = self._latents_to_rgb(latents)
        self.latent_queue.put(image)
        self._current_step = step
        return callback_kwargs
