"""
BoxDiff-based generator implementation.

This generator handles BoxDiff (Training-Free Box-Constrained Diffusion) models.
BoxDiff consumes (phrase, box) pairs like GLIGEN but uses attention constraints
during denoising, requiring no extra training.

Key differences from GLIGEN:
- GLIGEN: Requires specially trained model with grounding capability
- BoxDiff: Works with ANY Stable Diffusion model via attention manipulation

The generator is self-contained: it computes bounding boxes internally from
prompt_data, keeping the harness model-agnostic.

Reference:
- BoxDiff Paper (ICCV 2023): https://arxiv.org/abs/2307.10816
- Official Repo: https://github.com/showlab/BoxDiff
- JOURNAL.md architecture requirements
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image

from spatialbench_uc.generators.base import BaseGenerator, GeneratorConfig, PromptData
from spatialbench_uc.generators.registry import register_generator
from spatialbench_uc.utils.device import (
    enable_memory_optimizations,
    get_device,
    get_torch_dtype,
    set_seed,
)

if TYPE_CHECKING:
    from spatialbench_uc.generators.pipelines import StableDiffusionBoxDiffPipeline

logger = logging.getLogger(__name__)


# =============================================================================
# Box Coordinate Utilities
# =============================================================================

# Default placements matching gen_sd15_gligen.yaml and gen_sd15_boxdiff.yaml
# BoxDiff uses the same coordinate system as GLIGEN: normalized [0, 1] coordinates
DEFAULT_BOX_PLACEMENTS = {
    "left_of": {
        "box_a": {"x_range": (0.05, 0.35), "y_range": (0.25, 0.75)},
        "box_b": {"x_range": (0.65, 0.95), "y_range": (0.25, 0.75)},
    },
    "right_of": {
        "box_a": {"x_range": (0.65, 0.95), "y_range": (0.25, 0.75)},
        "box_b": {"x_range": (0.05, 0.35), "y_range": (0.25, 0.75)},
    },
    "above": {
        "box_a": {"x_range": (0.25, 0.75), "y_range": (0.05, 0.35)},
        "box_b": {"x_range": (0.25, 0.75), "y_range": (0.65, 0.95)},
    },
    "below": {
        "box_a": {"x_range": (0.25, 0.75), "y_range": (0.65, 0.95)},
        "box_b": {"x_range": (0.25, 0.75), "y_range": (0.05, 0.35)},
    },
}


def _extract_words_from_object(obj: str) -> list[str]:
    """
    Extract individual words from an object name for BoxDiff token matching.
    
    BoxDiff requires single tokens that match exactly in the tokenized prompt.
    For multi-word objects like "teddy bear", we return each word separately
    so they can each be constrained to the same bounding box.
    
    Args:
        obj: Object name (e.g., "cake", "teddy bear", "ice cream")
        
    Returns:
        List of words with articles removed (e.g., ["teddy", "bear"])
    """
    # Strip and lowercase for processing
    obj = obj.strip()
    
    # Remove leading articles
    for article in ["a ", "an ", "the "]:
        if obj.lower().startswith(article):
            obj = obj[len(article):]
            break
    
    # Split into individual words
    words = obj.split()
    
    # Filter out empty strings and very short words that might not tokenize well
    words = [w.strip() for w in words if w.strip()]
    
    return words


def compute_boxdiff_boxes(
    relation: str,
    object_a: str,
    object_b: str,
    placement_config: dict[str, Any] | None = None,
) -> tuple[list[list[float]], list[str]]:
    """
    Compute BoxDiff bounding boxes and phrases for a spatial relation.

    BoxDiff uses the same coordinate format as GLIGEN:
    normalized [0, 1] coordinates in [x0, y0, x1, y1] format.

    IMPORTANT: BoxDiff phrases must be SINGLE TOKENS that appear in the prompt.
    For multi-word objects like "teddy bear", we split them and assign the same
    bounding box to each word token (e.g., both "teddy" and "bear" get box_a).

    Args:
        relation: Spatial relation. One of: 'left_of', 'right_of', 'above', 'below'
        object_a: Name of object A (e.g., "cake", "teddy bear")
        object_b: Name of object B (e.g., "bed", "ice cream")
        placement_config: Optional dict with custom box placement ranges.

    Returns:
        Tuple of (boxes, phrases) where:
        - boxes: List of [x0, y0, x1, y1] normalized coordinates
        - phrases: List of single-word tokens matching the boxes

    Example:
        >>> boxes, phrases = compute_boxdiff_boxes("left_of", "teddy bear", "bed")
        >>> boxes
        [[0.05, 0.25, 0.35, 0.75], [0.05, 0.25, 0.35, 0.75], [0.65, 0.25, 0.95, 0.75]]
        >>> phrases
        ['teddy', 'bear', 'bed']
    """
    # Get placement for this relation
    if placement_config and relation in placement_config:
        placement = placement_config[relation]
    elif relation in DEFAULT_BOX_PLACEMENTS:
        placement = DEFAULT_BOX_PLACEMENTS[relation]
    else:
        raise ValueError(
            f"Unknown relation: {relation}. "
            f"Supported: {list(DEFAULT_BOX_PLACEMENTS.keys())}"
        )

    # Extract box coordinates (normalized [0, 1])
    box_a_cfg = placement["box_a"]
    box_a = [
        box_a_cfg["x_range"][0],  # x0
        box_a_cfg["y_range"][0],  # y0
        box_a_cfg["x_range"][1],  # x1
        box_a_cfg["y_range"][1],  # y1
    ]

    box_b_cfg = placement["box_b"]
    box_b = [
        box_b_cfg["x_range"][0],  # x0
        box_b_cfg["y_range"][0],  # y0
        box_b_cfg["x_range"][1],  # x1
        box_b_cfg["y_range"][1],  # y1
    ]

    # Extract words from each object (handles multi-word like "teddy bear")
    words_a = _extract_words_from_object(object_a)
    words_b = _extract_words_from_object(object_b)
    
    # Build boxes and phrases lists
    # Each word in object_a gets box_a, each word in object_b gets box_b
    boxes = []
    phrases = []
    
    for word in words_a:
        boxes.append(box_a)
        phrases.append(word)
    
    for word in words_b:
        boxes.append(box_b)
        phrases.append(word)

    return boxes, phrases


def load_box_placement_config(config: dict[str, Any]) -> dict[str, Any] | None:
    """
    Extract box placement configuration from a generation config dict.

    Args:
        config: Full generation config (from YAML)

    Returns:
        Box placement config dict or None if not specified
    """
    return config.get("box_placement")


# =============================================================================
# BoxDiff Generator
# =============================================================================


@register_generator("boxdiff")
class BoxDiffGenerator(BaseGenerator):
    """
    Generator for BoxDiff (Training-Free Box-Constrained Diffusion).

    BoxDiff enables spatial control in ANY Stable Diffusion model without
    requiring additional training. It works by manipulating cross-attention
    maps during the denoising process to concentrate attention for each
    phrase within its corresponding bounding box.

    Key differences from GLIGEN:
    - Works with any SD model (SD 1.5, SD 2.1, etc.)
    - No additional training required
    - Uses attention manipulation instead of grounding tokens

    This generator is self-contained: it computes bounding boxes internally
    from prompt_data.relation, keeping the harness model-agnostic.

    Example config:
        ```yaml
        generator:
          type: boxdiff
          model_id: runwayml/stable-diffusion-v1-5
          params:
            height: 512
            width: 512
            num_inference_steps: 30
            guidance_scale: 7.5
            boxdiff_kwargs:
              attention_res: 16
              P: 0.2
              max_iter_to_alter: 25

        box_placement:
          left_of:
            box_a: {x_range: [0.05, 0.35], y_range: [0.25, 0.75]}
            box_b: {x_range: [0.65, 0.95], y_range: [0.25, 0.75]}
        ```
    """

    def __init__(self, config: GeneratorConfig) -> None:
        """
        Initialize the BoxDiff generator.

        Args:
            config: Generator configuration with model_id, params,
                   and full_config for box placement and boxdiff_kwargs settings.

        Note:
            The pipeline is not loaded until warmup() or first generate() call.
            This allows for lazy loading and better memory management.
        """
        self.config = config
        self.model_id = config.model_id
        self.params = config.params or {}

        # Extract box placement config from full config
        self._box_placement = config.full_config.get("box_placement")

        # Extract BoxDiff-specific kwargs
        self._boxdiff_kwargs = self._prepare_boxdiff_kwargs(
            self.params.get("boxdiff_kwargs", {})
        )

        # Pipeline will be loaded lazily
        self.pipeline: StableDiffusionBoxDiffPipeline | None = None
        self.device: torch.device | None = None
        self.dtype: torch.dtype | None = None
        self._is_loaded = False

    def _prepare_boxdiff_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare BoxDiff kwargs, converting types as needed.

        Some YAML types don't map directly to what BoxDiff expects:
        - scale_range: YAML list -> Python tuple
        - loss_thresholds: YAML dict with string keys -> dict with int keys
        """
        result = kwargs.copy()

        # Convert scale_range from list to tuple
        if "scale_range" in result and isinstance(result["scale_range"], list):
            result["scale_range"] = tuple(result["scale_range"])

        # Convert loss_thresholds keys from strings to ints
        if "loss_thresholds" in result:
            thresholds = result["loss_thresholds"]
            if isinstance(thresholds, dict):
                result["loss_thresholds"] = {
                    int(k): v for k, v in thresholds.items()
                }

        return result

    def _load_pipeline(self) -> None:
        """Load the BoxDiff diffusers pipeline."""
        if self._is_loaded:
            return

        # Import the vendored pipeline
        from diffusers import StableDiffusionPipeline
        from spatialbench_uc.generators.pipelines import StableDiffusionBoxDiffPipeline

        self.device = get_device()
        self.dtype = get_torch_dtype(self.device)

        logger.info(f"Loading BoxDiff pipeline on {self.device} with dtype {self.dtype}")
        logger.info(f"Model: {self.model_id}")

        # Load base SD pipeline first, then construct BoxDiff from components
        # This avoids compatibility issues with diffusers' from_pretrained
        base_pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
        )

        # Construct BoxDiff pipeline from base pipeline components
        self.pipeline = StableDiffusionBoxDiffPipeline(
            vae=base_pipe.vae,
            text_encoder=base_pipe.text_encoder,
            tokenizer=base_pipe.tokenizer,
            unet=base_pipe.unet,
            scheduler=base_pipe.scheduler,
            safety_checker=None,
            feature_extractor=getattr(base_pipe, 'feature_extractor', None),
            requires_safety_checker=False,
            image_encoder=getattr(base_pipe, 'image_encoder', None),
        )
        
        # Clean up base pipeline to free memory
        del base_pipe

        # Move to device
        self.pipeline = self.pipeline.to(self.device)

        # Set scheduler if specified
        scheduler_name = self.params.get("scheduler")
        if scheduler_name:
            self._set_scheduler(scheduler_name)

        # Apply memory optimizations
        enable_memory_optimizations(self.pipeline)

        # Disable progress bar for batch processing
        self.pipeline.set_progress_bar_config(disable=True)

        self._is_loaded = True
        logger.info("BoxDiff pipeline loaded successfully")

    def _set_scheduler(self, scheduler_name: str) -> None:
        """Set the scheduler based on config name."""
        from diffusers import (
            DDIMScheduler,
            DPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            PNDMScheduler,
            UniPCMultistepScheduler,
        )

        scheduler_map = {
            "UniPCMultistepScheduler": UniPCMultistepScheduler,
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
            "EulerDiscreteScheduler": EulerDiscreteScheduler,
            "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
            "DDIMScheduler": DDIMScheduler,
            "PNDMScheduler": PNDMScheduler,
        }

        if scheduler_name in scheduler_map:
            scheduler_cls = scheduler_map[scheduler_name]
            self.pipeline.scheduler = scheduler_cls.from_config(
                self.pipeline.scheduler.config
            )
            logger.info(f"Using {scheduler_name}")
        else:
            logger.warning(
                f"Unknown scheduler: {scheduler_name}. Using default. "
                f"Available: {list(scheduler_map.keys())}"
            )

    def _compute_boxes(
        self, prompt_data: PromptData
    ) -> tuple[list[list[float]], list[str]]:
        """
        Compute bounding boxes and phrases from prompt data.

        This is handled internally by the generator, keeping the harness
        model-agnostic.
        """
        return compute_boxdiff_boxes(
            relation=prompt_data.relation,
            object_a=prompt_data.object_a,
            object_b=prompt_data.object_b,
            placement_config=self._box_placement,
        )

    def generate(
        self,
        prompt_data: PromptData,
        seed: int,
    ) -> Image.Image:
        """
        Generate an image from prompt data with BoxDiff spatial constraints.

        Bounding boxes are computed internally from prompt_data.relation
        and the configured box_placement settings.

        Args:
            prompt_data: Structured prompt information including the text prompt,
                        spatial relation, and object names.
            seed: Random seed for reproducibility.

        Returns:
            The generated image as a PIL Image.
        """
        # Lazy load pipeline
        if not self._is_loaded:
            self._load_pipeline()

        # Compute boxes from prompt data
        boxdiff_boxes, boxdiff_phrases = self._compute_boxes(prompt_data)

        # Create seeded generator (MPS-safe via set_seed)
        generator = set_seed(seed, self.device)

        # Build generation kwargs
        gen_kwargs = {
            "prompt": prompt_data.prompt,
            "generator": generator,
            "height": self.params.get("height", 512),
            "width": self.params.get("width", 512),
            "num_inference_steps": self.params.get("num_inference_steps", 30),
            "guidance_scale": self.params.get("guidance_scale", 7.5),
            # BoxDiff-specific parameters
            "boxdiff_boxes": boxdiff_boxes,
            "boxdiff_phrases": boxdiff_phrases,
            "boxdiff_kwargs": self._boxdiff_kwargs,
        }

        # Add negative prompt if specified
        negative_prompt = self.params.get("negative_prompt")
        if negative_prompt:
            gen_kwargs["negative_prompt"] = negative_prompt

        # Generate image
        logger.debug(
            f"Generating with BoxDiff: boxes={boxdiff_boxes}, phrases={boxdiff_phrases}"
        )
        result = self.pipeline(**gen_kwargs)
        return result.images[0]

    def warmup(self) -> None:
        """
        Preload the BoxDiff pipeline.

        This loads the model to GPU/MPS and optionally runs a dummy inference
        to trigger any lazy initialization.
        """
        if not self._is_loaded:
            self._load_pipeline()
            logger.info("BoxDiff warmup complete")

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

            logger.info("BoxDiff pipeline cleaned up")
