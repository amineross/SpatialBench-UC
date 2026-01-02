"""
GroundingDINO detector implementation using HuggingFace transformers.

This detector uses the GroundingDINO model for open-vocabulary object detection.
It can detect any object described by text, not limited to a fixed class set.

Important: Queries must be lowercase and end with a period (e.g., "a cat. a dog.").
"""

from PIL import Image

from spatialbench_uc.detectors.base import BaseDetector, Detection, DetectorConfig
from spatialbench_uc.detectors.registry import register_detector


@register_detector("grounding_dino")
class GroundingDINODetector(BaseDetector):
    """
    GroundingDINO open-vocabulary detector.

    This detector uses the IDEA-Research/grounding-dino-base model from
    HuggingFace. It supports open-vocabulary detection with text queries.

    Note: Query format is critical:
    - Labels must be lowercase
    - Labels must be separated by periods
    - Example: "a cat. a dog." (NOT "cat, dog")

    Example config:
        ```yaml
        detector:
          type: grounding_dino
          model_id: IDEA-Research/grounding-dino-base
          params:
            box_threshold: 0.35
            text_threshold: 0.25
        ```
    """

    def __init__(self, config: DetectorConfig) -> None:
        """
        Initialize the GroundingDINO detector.

        Args:
            config: Detector configuration with model_id and params.

        Note:
            This is a stub implementation. Full implementation in Phase 4.
        """
        self.config = config
        self.model_id = config.model_id or "IDEA-Research/grounding-dino-base"
        self.params = config.params or {}
        self.box_threshold = self.params.get("box_threshold", 0.35)
        self.text_threshold = self.params.get("text_threshold", 0.25)

        # Model will be loaded lazily or in warmup()
        self.model = None
        self.processor = None
        self.device = None

    def detect(self, image: Image.Image, labels: list[str]) -> list[Detection]:
        """
        Detect objects matching the given labels.

        Args:
            image: The input image as a PIL Image.
            labels: List of object descriptions to detect.
                   Will be formatted as "a {label}. " for each label.

        Returns:
            List of Detection objects for matching instances.

        Note:
            This is a stub implementation. Full implementation in Phase 4.
        """
        raise NotImplementedError(
            "GroundingDINODetector.detect() is a stub. "
            "Full implementation will be added in Phase 4."
        )

    def warmup(self) -> None:
        """
        Preload the GroundingDINO model.

        Note:
            This is a stub implementation. Full implementation in Phase 4.
        """
        pass

    @staticmethod
    def format_query(labels: list[str]) -> str:
        """
        Format labels into a GroundingDINO query string.

        Args:
            labels: List of object labels (e.g., ["cat", "dog"]).

        Returns:
            Formatted query string (e.g., "a cat. a dog.").
        """
        return " ".join(f"a {label.lower()}." for label in labels)

