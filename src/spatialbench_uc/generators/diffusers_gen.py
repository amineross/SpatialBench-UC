"""
Diffusers-based generator implementation.

This generator handles Stable Diffusion models via the HuggingFace diffusers library.
It supports both prompt-only generation and ControlNet-guided generation.

Supported modes:
- prompt_only: Standard text-to-image generation
- controlnet: ControlNet-guided generation with edge maps
"""

from PIL import Image

from spatialbench_uc.generators.base import BaseGenerator, GeneratorConfig
from spatialbench_uc.generators.registry import register_generator


@register_generator("diffusers")
class DiffusersGenerator(BaseGenerator):
    """
    Generator for diffusers-compatible models (SD 1.5, SDXL, etc.).

    This generator supports:
    - Stable Diffusion 1.5 prompt-only
    - Stable Diffusion 1.5 + ControlNet Canny

    The mode is controlled via config.mode:
    - "prompt_only": Standard text-to-image
    - "controlnet": Requires control_image in generate()

    Example config:
        ```yaml
        generator:
          type: diffusers
          model_id: stable-diffusion-v1-5/stable-diffusion-v1-5
          mode: prompt_only
          params:
            height: 512
            width: 512
            num_inference_steps: 30
            guidance_scale: 7.5
        ```
    """

    def __init__(self, config: GeneratorConfig) -> None:
        """
        Initialize the diffusers generator.

        Args:
            config: Generator configuration with model_id, mode, and params.

        Note:
            This is a stub implementation. Full implementation in Phase 3.
        """
        self.config = config
        self.model_id = config.model_id
        self.mode = config.mode
        self.controlnet_id = config.controlnet_id
        self.params = config.params or {}

        # Pipeline will be loaded lazily or in warmup()
        self.pipeline = None
        self.device = None
        self.dtype = None

    def generate(
        self,
        prompt: str,
        seed: int,
        control_image: Image.Image | None = None,
    ) -> Image.Image:
        """
        Generate an image from a text prompt.

        Args:
            prompt: The text prompt describing the desired image.
            seed: Random seed for reproducibility.
            control_image: Control image for ControlNet mode (required if mode="controlnet").

        Returns:
            The generated image as a PIL Image.

        Note:
            This is a stub implementation. Full implementation in Phase 3.
        """
        # Stub: Return a placeholder image
        # Full implementation will use diffusers pipelines
        raise NotImplementedError(
            "DiffusersGenerator.generate() is a stub. "
            "Full implementation will be added in Phase 3."
        )

    def warmup(self) -> None:
        """
        Preload the diffusers pipeline.

        This loads the model to GPU and optionally runs a dummy inference
        to trigger JIT compilation.

        Note:
            This is a stub implementation. Full implementation in Phase 3.
        """
        pass

    def cleanup(self) -> None:
        """Release GPU memory by deleting the pipeline."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

