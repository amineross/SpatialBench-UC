"""
SpatialBench-UC: Uncertainty-aware spatial relationship benchmark for text-to-image models.

This package provides tools to:
1. Generate benchmark prompts with counterfactual pairs
2. Produce images using pluggable generators (SD 1.5, ControlNet, etc.)
3. Evaluate spatial correctness with uncertainty quantification
4. Report metrics including consistency and risk-coverage curves
"""

__version__ = "0.1.0"

from spatialbench_uc.generators import BaseGenerator, get_generator, register_generator
from spatialbench_uc.detectors import BaseDetector, Detection, get_detector, register_detector

__all__ = [
    "__version__",
    "BaseGenerator",
    "get_generator",
    "register_generator",
    "BaseDetector",
    "Detection",
    "get_detector",
    "register_detector",
]

