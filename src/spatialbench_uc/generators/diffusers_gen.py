"""
Diffusers-based generator implementation.

This generator handles Stable Diffusion models via the HuggingFace diffusers library.
It supports both prompt-only generation and ControlNet-guided generation.

Supported modes:
- prompt_only: Standard text-to-image generation
- controlnet: ControlNet-guided generation with edge maps

Reference: PROJECT.md Section 5 (Génération d'images)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from PIL import Image

from spatialbench_uc.generators.base import BaseGenerator, GeneratorConfig
from spatialbench_uc.generators.registry import register_generator
from spatialbench_uc.utils.device import (
    enable_memory_optimizations,
    get_device,
    get_torch_dtype,
    set_seed,
)

if TYPE_CHECKING:
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionPipeline

logger = logging.getLogger(__name__)


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
            The pipeline is not loaded until warmup() or first generate() call.
            This allows for lazy loading and better memory management.
        """
        self.config = config
        self.model_id = config.model_id
        self.mode = config.mode
        self.controlnet_id = config.controlnet_id
        self.params = config.params or {}

        # Pipeline will be loaded lazily
        self.pipeline: StableDiffusionPipeline | StableDiffusionControlNetPipeline | None = None
        self.device: torch.device | None = None
        self.dtype: torch.dtype | None = None
        self._is_loaded = False

    def _load_pipeline(self) -> None:
        """Load the diffusers pipeline based on mode."""
        if self._is_loaded:
            return

        # Import diffusers here to avoid import errors if not installed
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            StableDiffusionPipeline,
            UniPCMultistepScheduler,
        )

        self.device = get_device()
        self.dtype = get_torch_dtype(self.device)

        logger.info(f"Loading pipeline on {self.device} with dtype {self.dtype}")
        logger.info(f"Mode: {self.mode}, Model: {self.model_id}")

        if self.mode == "prompt_only":
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                use_safetensors=True,
            )

        elif self.mode == "controlnet":
            if not self.controlnet_id:
                raise ValueError(
                    "ControlNet mode requires controlnet_id in config. "
                    "Example: controlnet_id: 'lllyasviel/sd-controlnet-canny'"
                )

            logger.info(f"Loading ControlNet: {self.controlnet_id}")
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_id,
                torch_dtype=self.dtype,
            )

            self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id,
                controlnet=controlnet,
                torch_dtype=self.dtype,
                use_safetensors=True,
            )

        else:
            raise ValueError(
                f"Unknown mode: {self.mode}. "
                f"Supported modes: 'prompt_only', 'controlnet'"
            )

        # Move to device
        self.pipeline = self.pipeline.to(self.device)

        # Set scheduler if specified
        scheduler_name = self.params.get("scheduler")
        if scheduler_name == "UniPCMultistepScheduler":
            self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            logger.info("Using UniPCMultistepScheduler")

        # Apply memory optimizations
        enable_memory_optimizations(self.pipeline)

        # Disable progress bar for batch processing
        self.pipeline.set_progress_bar_config(disable=True)

        self._is_loaded = True
        logger.info("Pipeline loaded successfully")

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

        Raises:
            ValueError: If ControlNet mode is used without a control image.
        """
        # Lazy load pipeline
        if not self._is_loaded:
            self._load_pipeline()

        # Create seeded generator
        generator = set_seed(seed, self.device)

        # Build generation kwargs
        gen_kwargs = {
            "prompt": prompt,
            "generator": generator,
            "height": self.params.get("height", 512),
            "width": self.params.get("width", 512),
            "num_inference_steps": self.params.get("num_inference_steps", 30),
            "guidance_scale": self.params.get("guidance_scale", 7.5),
        }

        # Add negative prompt if specified
        negative_prompt = self.params.get("negative_prompt")
        if negative_prompt:
            gen_kwargs["negative_prompt"] = negative_prompt

        # Handle ControlNet mode
        if self.mode == "controlnet":
            if control_image is None:
                raise ValueError(
                    "ControlNet mode requires a control_image. "
                    "Pass a PIL Image with edge map or spatial guidance."
                )

            # Ensure control image matches generation size
            target_size = (gen_kwargs["width"], gen_kwargs["height"])
            if control_image.size != target_size:
                control_image = control_image.resize(target_size, Image.Resampling.LANCZOS)

            gen_kwargs["image"] = control_image

            # Add ControlNet conditioning scale if specified
            conditioning_scale = self.params.get("controlnet_conditioning_scale", 1.0)
            gen_kwargs["controlnet_conditioning_scale"] = conditioning_scale

        # Generate image
        result = self.pipeline(**gen_kwargs)
        return result.images[0]

    def warmup(self) -> None:
        """
        Preload the diffusers pipeline.

        This loads the model to GPU/MPS and optionally runs a dummy inference
        to trigger any lazy initialization.
        """
        if not self._is_loaded:
            self._load_pipeline()
            logger.info("Warmup complete")

    def cleanup(self) -> None:
        """Release GPU/MPS memory by deleting the pipeline."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self._is_loaded = False

            # Force garbage collection
            import gc
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Pipeline cleaned up")
