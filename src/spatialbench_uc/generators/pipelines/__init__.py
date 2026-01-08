"""
Vendored community pipelines.

This module contains community pipelines vendored from external sources
for reproducibility and stability. Each pipeline credits its source.
"""

from spatialbench_uc.generators.pipelines.pipeline_stable_diffusion_boxdiff import (
    StableDiffusionBoxDiffPipeline,
)

__all__ = ["StableDiffusionBoxDiffPipeline"]
