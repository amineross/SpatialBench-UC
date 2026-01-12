"""
GroundingDINO detector implementation using HuggingFace transformers.

This detector uses the GroundingDINO model for open-vocabulary object detection.
It can detect any object described by text, not limited to a fixed class set.

Important: Queries must be lowercase and end with a period (e.g., "a cat. a dog.").
"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from spatialbench_uc.detectors.base import BaseDetector, Detection, DetectorConfig
from spatialbench_uc.detectors.registry import register_detector
from spatialbench_uc.utils.device import get_device


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
        """
        self.config = config
        self.model_id = config.model_id or "IDEA-Research/grounding-dino-base"
        self.revision = config.revision  # HuggingFace model revision
        self.params = config.params or {}
        self.box_threshold = self.params.get("box_threshold", 0.35)
        self.text_threshold = self.params.get("text_threshold", 0.25)

        # Model will be loaded lazily or in warmup()
        self._model = None
        self._processor = None
        self._device = None

    def _load_model(self) -> None:
        """Load the GroundingDINO model if not already loaded."""
        if self._model is not None:
            return
            
        # Get device from config or auto-detect
        device_pref = self.params.get("device", "auto")
        self._device = get_device(device_pref)
        
        # Load processor and model (with optional revision for reproducibility)
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, revision=self.revision
        )
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id, revision=self.revision
        ).to(self._device)
        self._model.eval()

    def detect(self, image: Image.Image, labels: list[str]) -> list[Detection]:
        """
        Detect objects matching the given labels.

        Args:
            image: The input image as a PIL Image.
            labels: List of object descriptions to detect.
                   Will be formatted as "a {label}. " for each label.

        Returns:
            List of Detection objects for matching instances.
        """
        self._load_model()
        
        if not labels:
            return []
        
        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Format query string (lowercase + periods)
        # VERY important: text queries need to be lowercased + end with a dot
        text = self.format_query(labels)
        
        # Create normalized label mapping for result matching
        # Map "a cat." -> "cat", "a dog." -> "dog"
        label_map = {f"a {label.lower()}.": label.lower() for label in labels}
        
        # Process inputs
        inputs = self._processor(
            images=image, 
            text=text, 
            return_tensors="pt"
        ).to(self._device)
        
        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)
        
        # Post-process results
        # target_sizes is (height, width), but PIL.Image.size is (width, height)
        # Note: API changed in transformers 4.40+ (box_threshold -> threshold)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]]  # Convert (W, H) to (H, W)
            )
        
        # Convert to Detection objects
        detections = []
        result = results[0]  # Single image
        
        boxes = result["boxes"].cpu()
        scores = result["scores"].cpu()
        # Use text_labels if available (new API), otherwise labels (deprecated)
        result_labels = result.get("text_labels", result.get("labels", []))
        
        for box, score, label_text in zip(boxes, scores, result_labels):
            score_val = float(score)
            
            # Filter by box_threshold manually (in case API doesn't support it)
            if score_val < self.box_threshold:
                continue
            
            # Normalize the detected label
            # GroundingDINO may return labels like "a cat" or "cat"
            label_normalized = label_text.lower().strip()
            if label_normalized.startswith("a "):
                label_normalized = label_normalized[2:]
            if label_normalized.endswith("."):
                label_normalized = label_normalized[:-1]
            
            detections.append(Detection(
                box_xyxy=tuple(float(x) for x in box.tolist()),
                score=score_val,
                label=label_normalized,
            ))
        
        return detections

    def warmup(self) -> None:
        """Preload the GroundingDINO model."""
        self._load_model()

    def cleanup(self) -> None:
        """Release model resources."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

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
