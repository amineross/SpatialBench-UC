"""
Cross-platform device utilities.

This module provides device abstraction for PyTorch, automatically detecting
the best available device (CUDA, MPS, or CPU) and appropriate dtype.

Supports:
- NVIDIA GPUs via CUDA
- Apple Silicon via MPS (Metal Performance Shaders)
- CPU fallback for any platform
"""

from __future__ import annotations

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """
    Get the best available PyTorch device.

    Args:
        prefer: Device preference. Options:
            - "auto": Automatically select best available (CUDA > MPS > CPU)
            - "cuda": Force CUDA (raises error if unavailable)
            - "mps": Force MPS (raises error if unavailable)
            - "cpu": Force CPU

    Returns:
        torch.device: The selected device.

    Raises:
        RuntimeError: If the preferred device is not available.

    Examples:
        ```python
        device = get_device()  # Auto-detect
        device = get_device("cuda")  # Force CUDA
        device = get_device("cpu")  # Force CPU
        ```
    """
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    elif prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but not available. "
                "Check that you have a CUDA-capable GPU and appropriate drivers."
            )
        return torch.device("cuda")

    elif prefer == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS device requested but not available. "
                "MPS requires macOS 12.3+ and Apple Silicon or AMD GPU."
            )
        return torch.device("mps")

    elif prefer == "cpu":
        return torch.device("cpu")

    else:
        raise ValueError(
            f"Unknown device preference: '{prefer}'. "
            f"Must be one of: 'auto', 'cuda', 'mps', 'cpu'"
        )


def get_torch_dtype(device: torch.device | None = None) -> torch.dtype:
    """
    Get the appropriate dtype for the given device.

    CUDA devices work well with float16 for memory efficiency.
    MPS has improving float16 support but float32 is more stable.
    CPU uses float32 as it doesn't benefit from float16.

    Args:
        device: The target device. If None, auto-detects.

    Returns:
        torch.dtype: The recommended dtype for the device.

    Examples:
        ```python
        device = get_device()
        dtype = get_torch_dtype(device)  # float16 for CUDA, float32 for MPS/CPU
        model = Model.from_pretrained(..., torch_dtype=dtype)
        ```
    """
    if device is None:
        device = get_device()

    if device.type == "cuda":
        return torch.float16
    elif device.type == "mps":
        # MPS float16 support is improving but float32 is more stable
        # Can be changed to float16 if testing shows stability
        return torch.float32
    else:
        return torch.float32


def device_info() -> dict:
    """
    Get detailed information about available devices.

    This is useful for logging and reproducibility - include this
    in run manifests to track the hardware configuration.

    Returns:
        dict: Device information including:
            - device: The auto-selected device name
            - cuda_available: Whether CUDA is available
            - mps_available: Whether MPS is available
            - cuda_device_name: Name of CUDA device (if available)
            - cuda_memory_gb: Total CUDA memory in GB (if available)
            - torch_version: PyTorch version string

    Examples:
        ```python
        info = device_info()
        # {'device': 'cuda', 'cuda_available': True, 'mps_available': False,
        #  'cuda_device_name': 'NVIDIA RTX 4090', 'cuda_memory_gb': 24.0,
        #  'torch_version': '2.2.0'}
        ```
    """
    info = {
        "device": str(get_device()),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "torch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        info["cuda_memory_gb"] = round(props.total_memory / (1024**3), 2)
        info["cuda_capability"] = f"{props.major}.{props.minor}"

    if torch.backends.mps.is_available():
        # MPS doesn't provide detailed device info like CUDA
        info["mps_device"] = "Apple Silicon"

    return info


def set_seed(seed: int, device: torch.device | None = None) -> torch.Generator:
    """
    Create a seeded random generator for reproducible generation.

    Args:
        seed: The random seed.
        device: Target device for the generator. If None, auto-detects.

    Returns:
        torch.Generator: A seeded generator for the target device.

    Examples:
        ```python
        generator = set_seed(42)
        image = pipeline(prompt, generator=generator).images[0]
        ```
    """
    if device is None:
        device = get_device()

    # For MPS and CPU, we create the generator on CPU
    # (MPS generators have some quirks)
    if device.type == "mps":
        generator = torch.Generator("cpu").manual_seed(seed)
    else:
        generator = torch.Generator(device).manual_seed(seed)

    return generator


def enable_memory_optimizations(pipeline) -> None:
    """
    Enable memory optimizations on a diffusers pipeline.

    This function applies platform-appropriate optimizations:
    - CUDA: xformers memory-efficient attention (if available)
    - MPS: Attention slicing
    - CPU: CPU offload

    Args:
        pipeline: A diffusers pipeline object.

    Note:
        This function modifies the pipeline in-place and fails silently
        if optimizations are not available.
    """
    device = get_device()

    if device.type == "cuda":
        # Try xformers first (most memory efficient)
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            return
        except (ImportError, ModuleNotFoundError):
            pass

        # Fall back to attention slicing
        try:
            pipeline.enable_attention_slicing()
        except AttributeError:
            pass

    elif device.type == "mps":
        # MPS works best with attention slicing
        try:
            pipeline.enable_attention_slicing()
        except AttributeError:
            pass

    else:
        # CPU: enable sequential offload if available
        try:
            pipeline.enable_sequential_cpu_offload()
        except AttributeError:
            pass

