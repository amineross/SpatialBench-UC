"""
Generator module for text-to-image model abstraction.

This module provides:
- BaseGenerator: Abstract base class defining the generator interface
- register_generator: Decorator to register new generator implementations
- get_generator: Factory function to instantiate generators from config
"""

from spatialbench_uc.generators.base import BaseGenerator
from spatialbench_uc.generators.registry import (
    GENERATOR_REGISTRY,
    get_generator,
    register_generator,
)

# Import implementations to trigger registration
from spatialbench_uc.generators import diffusers_gen  # noqa: F401

__all__ = [
    "BaseGenerator",
    "GENERATOR_REGISTRY",
    "get_generator",
    "register_generator",
]

