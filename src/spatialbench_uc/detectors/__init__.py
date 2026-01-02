"""
Detector module for object detection abstraction.

This module provides:
- BaseDetector: Abstract base class defining the detector interface
- Detection: Dataclass representing a detected object
- register_detector: Decorator to register new detector implementations
- get_detector: Factory function to instantiate detectors from config
"""

from spatialbench_uc.detectors.base import BaseDetector, Detection
from spatialbench_uc.detectors.registry import (
    DETECTOR_REGISTRY,
    get_detector,
    register_detector,
)

# Import implementations to trigger registration
from spatialbench_uc.detectors import fasterrcnn  # noqa: F401
from spatialbench_uc.detectors import grounding_dino  # noqa: F401

__all__ = [
    "BaseDetector",
    "Detection",
    "DETECTOR_REGISTRY",
    "get_detector",
    "register_detector",
]

