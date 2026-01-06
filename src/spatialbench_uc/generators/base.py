"""
Base class for text-to-image generators.

All generator implementations must inherit from BaseGenerator and implement
the generate() method. This ensures a consistent interface across different
models (Stable Diffusion, DALL-E, Flux, etc.).

Design Philosophy:
- Generators are self-contained: they extract their own config and context
- The harness (generate.py) is model-agnostic: it passes unified inputs
- New models can be added without modifying the harness
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    # Full config from YAML (set by harness, allows generators to access any section)
    full_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class PromptData:
    """
    Structured prompt data passed to generators.
    
    This provides a consistent interface for all generators to access
    prompt information without coupling to the JSONL schema.
    """
    
    prompt_id: str
    prompt: str
    relation: str
    object_a: str
    object_b: str
    # Full prompt record for any additional fields
    raw: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptData":
        """Create PromptData from a prompt JSONL record."""
        return cls(
            prompt_id=data["prompt_id"],
            prompt=data["prompt"],
            relation=data["relation"],
            object_a=data["object_a"]["name"],
            object_b=data["object_b"]["name"],
            raw=data,
        )


class BaseGenerator(ABC):
    """
    Abstract base class for text-to-image generators.

    All generator implementations must inherit from this class and implement
    the generate() method. Generators are responsible for:
    
    1. Extracting their configuration from `self.config.full_config`
    2. Processing prompt_data to create any model-specific inputs
    3. Generating images via their specific API/pipeline
    
    This design ensures the generation harness remains model-agnostic.

    Example:
        ```python
        @register_generator("my_model")
        class MyGenerator(BaseGenerator):
            def __init__(self, config: GeneratorConfig):
                self.config = config
                self.model = load_my_model(config.model_id)
                # Extract model-specific config
                self.my_config = config.full_config.get("my_model_settings", {})

            def generate(
                self, 
                prompt_data: PromptData, 
                seed: int,
            ) -> Image.Image:
                # Generator handles its own context extraction
                spatial_info = self._compute_spatial_context(prompt_data)
                return self.model.generate(
                    prompt_data.prompt, 
                    seed=seed,
                    **spatial_info,
                )
        ```
    """

    @abstractmethod
    def __init__(self, config: GeneratorConfig) -> None:
        """
        Initialize the generator with configuration.

        Args:
            config: Generator configuration including model_id, mode, params,
                   and full_config (the complete YAML config for extracting
                   model-specific sections).
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt_data: PromptData,
        seed: int,
    ) -> Image.Image:
        """
        Generate an image from prompt data.

        Each generator is responsible for extracting what it needs from
        prompt_data and its configuration. This keeps the generation
        harness model-agnostic.

        Args:
            prompt_data: Structured prompt information including the text prompt,
                        spatial relation, and object names.
            seed: Random seed for reproducibility.

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
