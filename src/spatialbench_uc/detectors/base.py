"""
Base class for object detectors.

All detector implementations must inherit from BaseDetector and implement
the detect() method. This ensures a consistent interface across different
detection models (Faster R-CNN, GroundingDINO, YOLO, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass
class Detection:
    """
    Represents a detected object in an image.

    Attributes:
        box_xyxy: Bounding box coordinates as (x1, y1, x2, y2) in pixels.
                  (x1, y1) is top-left, (x2, y2) is bottom-right.
        score: Confidence score in range [0, 1].
        label: The detected object label/class name.
    """

    box_xyxy: tuple[float, float, float, float]
    score: float
    label: str

    @property
    def center(self) -> tuple[float, float]:
        """Get the center point of the bounding box."""
        x1, y1, x2, y2 = self.box_xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        """Get the area of the bounding box in pixels."""
        x1, y1, x2, y2 = self.box_xyxy
        return (x2 - x1) * (y2 - y1)

    @property
    def width(self) -> float:
        """Get the width of the bounding box."""
        return self.box_xyxy[2] - self.box_xyxy[0]

    @property
    def height(self) -> float:
        """Get the height of the bounding box."""
        return self.box_xyxy[3] - self.box_xyxy[1]


@dataclass
class DetectorConfig:
    """Configuration for a detector instance."""

    type: str
    model_id: str | None = None
    revision: str | None = None  # HuggingFace model revision/commit hash
    params: dict[str, Any] | None = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


class BaseDetector(ABC):
    """
    Abstract base class for object detectors.

    All detector implementations must inherit from this class and implement
    the detect() method. The interface supports both closed-vocabulary
    (COCO classes) and open-vocabulary (text query) detection.

    Example:
        ```python
        @register_detector("my_detector")
        class MyDetector(BaseDetector):
            def __init__(self, config: DetectorConfig):
                self.model = load_my_model(config.model_id)

            def detect(self, image: Image.Image, labels: list[str]) -> list[Detection]:
                results = self.model.detect(image, classes=labels)
                return [Detection(r.box, r.score, r.label) for r in results]
        ```
    """

    @abstractmethod
    def __init__(self, config: DetectorConfig) -> None:
        """
        Initialize the detector with configuration.

        Args:
            config: Detector configuration including model_id and params.
        """
        pass

    @abstractmethod
    def detect(self, image: Image.Image, labels: list[str]) -> list[Detection]:
        """
        Detect objects matching the given labels in an image.

        Args:
            image: The input image as a PIL Image.
            labels: List of object labels/classes to detect.
                   For closed-vocab detectors, these should be valid class names.
                   For open-vocab detectors, these can be any text descriptions.

        Returns:
            List of Detection objects for all detected instances.
            May include multiple detections per label if multiple instances exist.
        """
        pass

    def warmup(self) -> None:
        """
        Optional warmup method to preload models.

        Override this method if your detector benefits from warmup.
        """
        pass

    def cleanup(self) -> None:
        """
        Optional cleanup method to release resources.

        Override this method if your detector needs explicit cleanup.
        """
        pass

