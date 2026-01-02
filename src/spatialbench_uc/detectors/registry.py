"""
Detector registry for dynamic model loading.

The registry pattern allows new detector implementations to be added
without modifying the core pipeline code. Detectors register themselves
using the @register_detector decorator.
"""

from typing import Callable, Type

from spatialbench_uc.detectors.base import BaseDetector, DetectorConfig

# Global registry mapping type names to detector classes
DETECTOR_REGISTRY: dict[str, Type[BaseDetector]] = {}


def register_detector(name: str) -> Callable[[Type[BaseDetector]], Type[BaseDetector]]:
    """
    Decorator to register a detector implementation.

    Use this decorator on any class that inherits from BaseDetector
    to make it available for instantiation via get_detector().

    Args:
        name: The type name to register (used in config files).

    Returns:
        Decorator function that registers the class.

    Example:
        ```python
        @register_detector("fasterrcnn")
        class FasterRCNNDetector(BaseDetector):
            def detect(self, image, labels) -> list[Detection]:
                ...
        ```

        Then in config.yaml:
        ```yaml
        detector:
          type: fasterrcnn
        ```
    """

    def decorator(cls: Type[BaseDetector]) -> Type[BaseDetector]:
        if name in DETECTOR_REGISTRY:
            raise ValueError(
                f"Detector '{name}' is already registered. "
                f"Existing: {DETECTOR_REGISTRY[name]}, New: {cls}"
            )
        if not issubclass(cls, BaseDetector):
            raise TypeError(
                f"Cannot register {cls}: must be a subclass of BaseDetector"
            )
        DETECTOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_detector(config: DetectorConfig | dict) -> BaseDetector:
    """
    Instantiate a detector from configuration.

    This factory function looks up the detector class in the registry
    based on the config type and instantiates it.

    Args:
        config: Detector configuration. Can be a DetectorConfig object
               or a dict with at least a 'type' key.

    Returns:
        An instance of the registered detector class.

    Raises:
        KeyError: If the detector type is not registered.
        TypeError: If config is invalid.

    Example:
        ```python
        config = DetectorConfig(type="fasterrcnn")
        detector = get_detector(config)
        detections = detector.detect(image, ["cat", "dog"])
        ```
    """
    # Convert dict to DetectorConfig if needed
    if isinstance(config, dict):
        config = DetectorConfig(**config)

    if config.type not in DETECTOR_REGISTRY:
        available = list(DETECTOR_REGISTRY.keys())
        raise KeyError(
            f"Detector type '{config.type}' not found. "
            f"Available types: {available}. "
            f"Did you forget to import the detector module?"
        )

    detector_cls = DETECTOR_REGISTRY[config.type]
    return detector_cls(config)


def list_detectors() -> list[str]:
    """
    List all registered detector types.

    Returns:
        List of registered detector type names.
    """
    return list(DETECTOR_REGISTRY.keys())

