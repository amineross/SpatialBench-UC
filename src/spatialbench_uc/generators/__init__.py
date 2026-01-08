"""
Generator module for text-to-image model abstraction.

This module provides:
- BaseGenerator: Abstract base class defining the generator interface
- PromptData: Structured prompt information for generators
- register_generator: Decorator to register new generator implementations
- get_generator: Factory function to instantiate generators from config

Design Philosophy:
- Generators are self-contained: they extract their own config and context
- The harness (generate.py) is model-agnostic: it passes unified inputs
- New models can be added without modifying the harness
"""

from spatialbench_uc.generators.base import BaseGenerator, GeneratorConfig, PromptData
from spatialbench_uc.generators.registry import (
    GENERATOR_REGISTRY,
    get_generator,
    list_generators,
    register_generator,
)

# Import implementations to trigger registration
from spatialbench_uc.generators import diffusers_gen  # noqa: F401
from spatialbench_uc.generators import gligen_gen  # noqa: F401
from spatialbench_uc.generators import boxdiff_gen  # noqa: F401

# Import control image utilities (for external use, e.g., saving debug images)
from spatialbench_uc.generators.control_image import create_spatial_edge_map

# Import GLIGEN box utilities (for external use, e.g., visualization)
from spatialbench_uc.generators.gligen_gen import compute_gligen_boxes

# Import BoxDiff box utilities (for external use, e.g., visualization)
from spatialbench_uc.generators.boxdiff_gen import compute_boxdiff_boxes

__all__ = [
    # Base classes
    "BaseGenerator",
    "GeneratorConfig",
    "PromptData",
    # Registry
    "GENERATOR_REGISTRY",
    "get_generator",
    "list_generators",
    "register_generator",
    # Utilities (for external use)
    "create_spatial_edge_map",
    "compute_gligen_boxes",
    "compute_boxdiff_boxes",
]
