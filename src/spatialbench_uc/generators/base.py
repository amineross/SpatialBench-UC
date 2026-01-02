"""
Base class for text-to-image generators.

All generator implementations must inherit from BaseGenerator and implement
the generate() method. This ensures a consistent interface across different
models (Stable Diffusion, DALL-E, Flux, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass
class GeneratorConfig:
    """Configuration for a generator instance."""

    type: str
    model_id: str
    mode: str = "prompt_only"
    controlnet_id: str | None = None
    params: dict[str, Any] | None = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


class BaseGenerator(ABC):
    """
    Abstract base class for text-to-image generators.

    All generator implementations must inherit from this class and implement
    the generate() method. The interface is intentionally simple to accommodate
    different model architectures and APIs.

    Example:
        ```python
        @register_generator("my_model")
        class MyGenerator(BaseGenerator):
            def __init__(self, config: GeneratorConfig):
                self.model = load_my_model(config.model_id)

            def generate(self, prompt: str, seed: int) -> Image.Image:
                return self.model.generate(prompt, seed=seed)
        ```
    """

    @abstractmethod
    def __init__(self, config: GeneratorConfig) -> None:
        """
        Initialize the generator with configuration.

        Args:
            config: Generator configuration including model_id, mode, and params.
        """
        pass

    @abstractmethod
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
            control_image: Optional control image for guided generation
                          (e.g., ControlNet edge map).

        Returns:
            The generated image as a PIL Image.
        """
        pass

    def warmup(self) -> None:
        """
        Optional warmup method to preload models or run initial inference.

        Override this method if your generator benefits from warmup
        (e.g., to trigger JIT compilation or load models to GPU).
        """
        pass

    def cleanup(self) -> None:
        """
        Optional cleanup method to release resources.

        Override this method if your generator needs explicit cleanup
        (e.g., to free GPU memory or close API connections).
        """
        pass

