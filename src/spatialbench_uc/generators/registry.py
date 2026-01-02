"""
Generator registry for dynamic model loading.

The registry pattern allows new generator implementations to be added
without modifying the core pipeline code. Generators register themselves
using the @register_generator decorator.
"""

from typing import Callable, Type

from spatialbench_uc.generators.base import BaseGenerator, GeneratorConfig

# Global registry mapping type names to generator classes
GENERATOR_REGISTRY: dict[str, Type[BaseGenerator]] = {}


def register_generator(name: str) -> Callable[[Type[BaseGenerator]], Type[BaseGenerator]]:
    """
    Decorator to register a generator implementation.

    Use this decorator on any class that inherits from BaseGenerator
    to make it available for instantiation via get_generator().

    Args:
        name: The type name to register (used in config files).

    Returns:
        Decorator function that registers the class.

    Example:
        ```python
        @register_generator("diffusers")
        class DiffusersGenerator(BaseGenerator):
            def generate(self, prompt: str, seed: int) -> Image.Image:
                ...
        ```

        Then in config.yaml:
        ```yaml
        generator:
          type: diffusers
          model_id: stable-diffusion-v1-5
        ```
    """

    def decorator(cls: Type[BaseGenerator]) -> Type[BaseGenerator]:
        if name in GENERATOR_REGISTRY:
            raise ValueError(
                f"Generator '{name}' is already registered. "
                f"Existing: {GENERATOR_REGISTRY[name]}, New: {cls}"
            )
        if not issubclass(cls, BaseGenerator):
            raise TypeError(
                f"Cannot register {cls}: must be a subclass of BaseGenerator"
            )
        GENERATOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_generator(config: GeneratorConfig | dict) -> BaseGenerator:
    """
    Instantiate a generator from configuration.

    This factory function looks up the generator class in the registry
    based on the config type and instantiates it.

    Args:
        config: Generator configuration. Can be a GeneratorConfig object
               or a dict with at least a 'type' key.

    Returns:
        An instance of the registered generator class.

    Raises:
        KeyError: If the generator type is not registered.
        TypeError: If config is invalid.

    Example:
        ```python
        config = GeneratorConfig(type="diffusers", model_id="sd-v1-5")
        generator = get_generator(config)
        image = generator.generate("a cat", seed=42)
        ```
    """
    # Convert dict to GeneratorConfig if needed
    if isinstance(config, dict):
        config = GeneratorConfig(**config)

    if config.type not in GENERATOR_REGISTRY:
        available = list(GENERATOR_REGISTRY.keys())
        raise KeyError(
            f"Generator type '{config.type}' not found. "
            f"Available types: {available}. "
            f"Did you forget to import the generator module?"
        )

    generator_cls = GENERATOR_REGISTRY[config.type]
    return generator_cls(config)


def list_generators() -> list[str]:
    """
    List all registered generator types.

    Returns:
        List of registered generator type names.
    """
    return list(GENERATOR_REGISTRY.keys())

