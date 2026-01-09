"""
GLIGEN-based generator implementation.

This generator handles GLIGEN (Grounded-Language-to-Image Generation) models
via the HuggingFace diffusers library. GLIGEN enables controllable image
generation through bounding box grounding - the model receives explicit
"put object X in region [x0, y0, x1, y1]" instructions.

Key difference from ControlNet:
- ControlNet: Visual conditioning via edge maps (model interprets pixels)
- GLIGEN: Semantic conditioning via (box, phrase) pairs (explicit grounding)

The generator is self-contained: it computes bounding boxes internally from
prompt_data, keeping the harness model-agnostic.

Reference: 
- PROJECT.md Section 5
- JOURNAL.md "Pre-Phase 6: Study Design Refinement"
- https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/gligen
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
    from diffusers import StableDiffusionGLIGENPipeline

logger = logging.getLogger(__name__)


# =============================================================================
# Box Coordinate Utilities
# =============================================================================

# Default placements matching gen_sd15_gligen.yaml
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


def compute_gligen_boxes(
    relation: str,
    object_a: str,
    object_b: str,
    placement_config: dict[str, Any] | None = None,
) -> tuple[list[list[float]], list[str]]:
    """
    Compute GLIGEN bounding boxes and phrases for a spatial relation.

    Unlike ControlNet edge maps, GLIGEN takes box coordinates directly as
    normalized [0, 1] coordinates in [x0, y0, x1, y1] format.

    Args:
        relation: Spatial relation. One of: 'left_of', 'right_of', 'above', 'below'
        object_a: Name of object A (e.g., "cake")
        object_b: Name of object B (e.g., "bed")
        placement_config: Optional dict with custom box placement ranges.

    Returns:
        Tuple of (boxes, phrases) where:
        - boxes: List of [x0, y0, x1, y1] normalized coordinates
        - phrases: List of object phrases matching the boxes

    Example:
        >>> boxes, phrases = compute_gligen_boxes("left_of", "cake", "bed")
        >>> boxes
        [[0.05, 0.25, 0.35, 0.75], [0.65, 0.25, 0.95, 0.75]]
        >>> phrases
        ['a cake', 'a bed']
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

    # Format phrases (add article if not present)
    phrase_a = object_a if object_a.startswith(("a ", "an ", "the ")) else f"a {object_a}"
    phrase_b = object_b if object_b.startswith(("a ", "an ", "the ")) else f"a {object_b}"

    return [box_a, box_b], [phrase_a, phrase_b]


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
# GLIGEN Generator
# =============================================================================


@register_generator("gligen")
class GLIGENGenerator(BaseGenerator):
    """
    Generator for GLIGEN (Grounded-Language-to-Image Generation).

    GLIGEN extends Stable Diffusion with bounding box grounding capability.
    Instead of interpreting edge maps like ControlNet, GLIGEN receives
    explicit (box, phrase) pairs that specify where objects should be placed.

    This generator is self-contained: it computes bounding boxes internally
    from prompt_data.relation, keeping the harness model-agnostic.

    Example config:
        ```yaml
        generator:
          type: gligen
          model_id: masterful/gligen-1-4-generation-text-box
          mode: gligen
          params:
            height: 512
            width: 512
            num_inference_steps: 30
            guidance_scale: 7.5
            gligen_scheduled_sampling_beta: 0.3
        
        box_placement:
          left_of:
            box_a: {x_range: [0.05, 0.35], y_range: [0.25, 0.75]}
            box_b: {x_range: [0.65, 0.95], y_range: [0.25, 0.75]}
        ```
    """

    def __init__(self, config: GeneratorConfig) -> None:
        """
        Initialize the GLIGEN generator.

        Args:
            config: Generator configuration with model_id, params,
                   and full_config for box placement settings.

        Note:
            The pipeline is not loaded until warmup() or first generate() call.
            This allows for lazy loading and better memory management.
        """
        self.config = config
        self.model_id = config.model_id
        self.params = config.params or {}
        
        # Model revision for reproducibility (FIX_PLAN.md 1D.1)
        self._revision = config.revision
        self._actual_revision: str | None = None

        # Extract box placement config from full config
        self._box_placement = config.full_config.get("box_placement")

        # Pipeline will be loaded lazily
        self.pipeline: StableDiffusionGLIGENPipeline | None = None
        self.device: torch.device | None = None
        self.dtype: torch.dtype | None = None
        self._is_loaded = False

    def _load_pipeline(self) -> None:
        """Load the GLIGEN diffusers pipeline."""
        if self._is_loaded:
            return

        # Import diffusers here to avoid import errors if not installed
        from diffusers import StableDiffusionGLIGENPipeline

        self.device = get_device()
        self.dtype = get_torch_dtype(self.device)

        logger.info(f"Loading GLIGEN pipeline on {self.device} with dtype {self.dtype}")
        logger.info(f"Model: {self.model_id}")

        # FIX_PLAN.md 1D.1: Use explicit revision if provided
        revision_kwargs = {}
        if self._revision:
            revision_kwargs["revision"] = self._revision
            logger.info(f"Using revision: {self._revision}")

        self.pipeline = StableDiffusionGLIGENPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            **revision_kwargs,
        )
        
        # Record actual revision used
        self._actual_revision = self._revision or "main"

        # Move to device
        self.pipeline = self.pipeline.to(self.device)

        # Apply memory optimizations
        enable_memory_optimizations(self.pipeline)

        # Disable progress bar for batch processing
        self.pipeline.set_progress_bar_config(disable=True)

        self._is_loaded = True
        logger.info("GLIGEN pipeline loaded successfully")

    def _compute_boxes(
        self, prompt_data: PromptData
    ) -> tuple[list[list[float]], list[str]]:
        """
        Compute bounding boxes and phrases from prompt data.
        
        This is handled internally by the generator, keeping the harness
        model-agnostic.
        """
        return compute_gligen_boxes(
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
        Generate an image from prompt data with GLIGEN spatial grounding.

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
        gligen_boxes, gligen_phrases = self._compute_boxes(prompt_data)

        # Create seeded generator
        generator = set_seed(seed, self.device)

        # Build generation kwargs
        gen_kwargs = {
            "prompt": prompt_data.prompt,
            "generator": generator,
            "height": self.params.get("height", 512),
            "width": self.params.get("width", 512),
            "num_inference_steps": self.params.get("num_inference_steps", 30),
            "guidance_scale": self.params.get("guidance_scale", 7.5),
            # GLIGEN-specific parameters
            "gligen_boxes": gligen_boxes,
            "gligen_phrases": gligen_phrases,
            "gligen_scheduled_sampling_beta": self.params.get(
                "gligen_scheduled_sampling_beta", 0.3
            ),
        }

        # Add negative prompt if specified
        negative_prompt = self.params.get("negative_prompt")
        if negative_prompt:
            gen_kwargs["negative_prompt"] = negative_prompt

        # Generate image
        logger.debug(
            f"Generating with GLIGEN: boxes={gligen_boxes}, phrases={gligen_phrases}"
        )
        result = self.pipeline(**gen_kwargs)
        return result.images[0]

    def warmup(self) -> None:
        """
        Preload the GLIGEN pipeline.

        This loads the model to GPU/MPS and optionally runs a dummy inference
        to trigger any lazy initialization.
        """
        if not self._is_loaded:
            self._load_pipeline()
            logger.info("GLIGEN warmup complete")

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

            logger.info("GLIGEN pipeline cleaned up")

    def get_model_info(self) -> dict[str, Any]:
        """
        Get model information for manifest recording.
        
        FIX_PLAN.md 1D.1: Record actual HF model revisions.
        """
        return {
            "model_id": self.model_id,
            "revision": self._actual_revision or self._revision or "main",
        }
