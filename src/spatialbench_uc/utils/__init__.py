"""
Utility modules for SpatialBench-UC.

Provides cross-platform device detection, image perturbations,
geometric calculations, and overlay generation.
"""

from spatialbench_uc.utils.device import (
    device_info,
    get_device,
    get_torch_dtype,
)

__all__ = [
    "get_device",
    "get_torch_dtype",
    "device_info",
]

