"""
Evaluation module for spatial relationship checking.

This module implements the checker/evaluator that:
1. Loads generated images from a manifest
2. Runs object detection (Faster R-CNN + GroundingDINO)
3. Applies geometric rules to determine PASS/FAIL/UNDECIDABLE
4. Computes confidence scores based on detection, geometry, stability, agreement
5. Outputs per-sample results and aggregated metrics

Usage:
    python -m spatialbench_uc.evaluate \
        --manifest runs/<run_id>/manifest.jsonl \
        --config configs/checker_v1.yaml \
        --out runs/<run_id>/eval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from tqdm import tqdm

from spatialbench_uc.detectors import get_detector, Detection
from spatialbench_uc.detectors.base import DetectorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Code Version Tracking (for reproducibility)
# =============================================================================

def get_code_versions() -> dict[str, str]:
    """Get versions of key libraries for reproducibility."""
    import subprocess
    
    versions = {}

    try:
        import torch
        versions["torch"] = torch.__version__
    except ImportError:
        pass

    try:
        import torchvision
        versions["torchvision"] = torchvision.__version__
    except ImportError:
        pass

    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except ImportError:
        pass

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        if result.returncode == 0:
            versions["git_commit"] = result.stdout.strip()[:8]
    except Exception:
        pass

    return versions


# =============================================================================
# Verdict Types
# =============================================================================

class Verdict(str, Enum):
    """Possible verdict outcomes for spatial relation evaluation."""
    PASS = "PASS"
    FAIL = "FAIL"
    UNDECIDABLE = "UNDECIDABLE"


@dataclass
class VerdictResult:
    """Result of evaluating a single image."""
    verdict: Verdict
    reason: str | None = None
    
    # Detection results
    detection_a: Detection | None = None
    detection_b: Detection | None = None
    
    # Geometry metrics
    delta: float | None = None  # Normalized center difference
    margin: float = 0.05
    iou_ab: float | None = None
    
    # Confidence components
    conf_detection: float = 0.0
    conf_geometry: float = 0.0
    conf_stability: float = 1.0
    conf_agreement: float = 1.0
    
    # Overall confidence
    confidence: float = 0.0


# =============================================================================
# Geometry Utilities
# =============================================================================

def compute_iou(box1: tuple, box2: tuple) -> float:
    """Compute Intersection over Union of two bounding boxes.
    
    Args:
        box1, box2: Bounding boxes as (x1, y1, x2, y2).
        
    Returns:
        IoU value in [0, 1].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def get_center(box: tuple) -> tuple[float, float]:
    """Get center point of a bounding box.
    
    Args:
        box: Bounding box as (x1, y1, x2, y2).
        
    Returns:
        Center as (cx, cy).
    """
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def compute_geometric_delta(
    box_a: tuple, 
    box_b: tuple, 
    relation: str, 
    img_width: int, 
    img_height: int
) -> float:
    """Compute normalized geometric delta for a spatial relation.
    
    For left_of/right_of: delta = (cx_A - cx_B) / width
    For above/below: delta = (cy_A - cy_B) / height
    
    Args:
        box_a, box_b: Bounding boxes as (x1, y1, x2, y2).
        relation: One of 'left_of', 'right_of', 'above', 'below'.
        img_width, img_height: Image dimensions for normalization.
        
    Returns:
        Normalized delta. Negative means A is left/above B.
    """
    cx_a, cy_a = get_center(box_a)
    cx_b, cy_b = get_center(box_b)
    
    if relation in ("left_of", "right_of"):
        return (cx_a - cx_b) / img_width
    else:  # above, below
        return (cy_a - cy_b) / img_height


def evaluate_geometry(
    delta: float, 
    relation: str, 
    margin: float
) -> tuple[Verdict, str | None]:
    """Apply geometric rules to determine verdict.
    
    Rules (per PROJECT.md Section 6.2):
    - left_of: PASS if delta < -margin, FAIL if delta > +margin
    - right_of: PASS if delta > +margin, FAIL if delta < -margin
    - above: PASS if delta < -margin, FAIL if delta > +margin
    - below: PASS if delta > +margin, FAIL if delta < -margin
    - UNDECIDABLE if |delta| <= margin
    
    Args:
        delta: Normalized center difference.
        relation: The expected spatial relation.
        margin: UNDECIDABLE zone boundary.
        
    Returns:
        Tuple of (verdict, reason or None).
    """
    abs_delta = abs(delta)
    
    if abs_delta <= margin:
        return Verdict.UNDECIDABLE, f"near_boundary (|delta|={abs_delta:.3f} <= margin={margin})"
    
    if relation == "left_of":
        # A should be to the left of B → cx_A < cx_B → delta < 0
        if delta < -margin:
            return Verdict.PASS, None
        else:
            return Verdict.FAIL, None
            
    elif relation == "right_of":
        # A should be to the right of B → cx_A > cx_B → delta > 0
        if delta > margin:
            return Verdict.PASS, None
        else:
            return Verdict.FAIL, None
            
    elif relation == "above":
        # A should be above B → cy_A < cy_B → delta < 0
        if delta < -margin:
            return Verdict.PASS, None
        else:
            return Verdict.FAIL, None
            
    elif relation == "below":
        # A should be below B → cy_A > cy_B → delta > 0
        if delta > margin:
            return Verdict.PASS, None
        else:
            return Verdict.FAIL, None
    
    return Verdict.UNDECIDABLE, f"unknown_relation: {relation}"


# =============================================================================
# Confidence Score Computation
# =============================================================================

def compute_confidence(
    det_a_score: float,
    det_b_score: float,
    delta: float,
    margin: float,
    geom_slope: float,
    stability: float,
    agreement: float,
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Compute overall confidence score.
    
    Formula (per PROJECT.md Section 6.5):
        Conf = det^w_det * geom^w_geom * stab^w_stab * agree^w_agree
    
    Components:
    - det = sqrt(score_A * score_B)
    - geom = clip((|delta| - margin) / slope, 0, 1)
    - stab = stability score from perturbation testing
    - agree = agreement score between detectors
    
    Args:
        det_a_score, det_b_score: Detection confidence scores.
        delta: Geometric delta (normalized).
        margin: Geometric margin.
        geom_slope: Slope for geometry confidence.
        stability: Stability score in [0, 1].
        agreement: Agreement score in [0, 1].
        weights: Dict with 'detection', 'geometry', 'stability', 'agreement' weights.
        
    Returns:
        Tuple of (overall confidence, component dict).
    """
    import math
    
    # Detection confidence
    conf_det = math.sqrt(det_a_score * det_b_score)
    
    # Geometry confidence (distance from boundary)
    d = abs(delta)
    conf_geom = max(0.0, min(1.0, (d - margin) / geom_slope))
    
    # Clamp all to valid range
    conf_det = max(0.0, min(1.0, conf_det))
    stability = max(0.0, min(1.0, stability))
    agreement = max(0.0, min(1.0, agreement))
    
    components = {
        "detection": conf_det,
        "geometry": conf_geom,
        "stability": stability,
        "agreement": agreement,
    }
    
    # Weighted geometric mean (product of powers)
    w_det = weights.get("detection", 0.4)
    w_geom = weights.get("geometry", 0.3)
    w_stab = weights.get("stability", 0.2)
    w_agree = weights.get("agreement", 0.1)
    
    # Avoid log(0) by using small epsilon
    eps = 1e-10
    conf = (
        (conf_det + eps) ** w_det *
        (conf_geom + eps) ** w_geom *
        (stability + eps) ** w_stab *
        (agreement + eps) ** w_agree
    )
    
    return conf, components


# =============================================================================
# Perturbation Testing
# =============================================================================

def apply_perturbation(image: Image.Image, perturb_config: dict) -> Image.Image:
    """Apply a perturbation to an image.
    
    Supported perturbations:
    - brightness: Adjust brightness by factor
    - gaussian_blur: Apply Gaussian blur
    - resize: Resize by scale then back
    
    Args:
        image: Input PIL Image.
        perturb_config: Dict with 'type' and parameters.
        
    Returns:
        Perturbed image.
    """
    from PIL import ImageEnhance, ImageFilter
    
    perturb_type = perturb_config["type"]
    
    if perturb_type == "brightness":
        factor = perturb_config.get("factor", 1.0)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
        
    elif perturb_type == "gaussian_blur":
        sigma = perturb_config.get("sigma", 1.0)
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))
        
    elif perturb_type == "resize":
        scale = perturb_config.get("scale", 0.9)
        orig_size = image.size
        new_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
        # Resize down then back up
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
        return resized.resize(orig_size, Image.Resampling.LANCZOS)
    
    # Unknown perturbation, return unchanged
    return image


def test_stability(
    image: Image.Image,
    detector,
    labels: list[str],
    relation: str,
    base_verdict: Verdict,
    perturbations: list[dict],
    thresholds: dict,
    geometry_config: dict,
) -> tuple[float, float]:
    """Test verdict stability under perturbations.
    
    FIX (per FIX_PLAN.md 1C.3): If perturbations produce no detections, 
    stability now decreases properly instead of defaulting to 1.0.
    
    Args:
        image: Original image.
        detector: Detector instance.
        labels: [object_a, object_b] labels.
        relation: Spatial relation.
        base_verdict: Verdict from unperturbed image.
        perturbations: List of perturbation configs.
        thresholds: Detection thresholds config.
        geometry_config: Geometry rules config.
        
    Returns:
        Tuple of (combined_stability, verdict_stability).
        - combined_stability: 0.5 * verdict_stability + 0.5 * iou_stability
        - verdict_stability: Fraction of perturbations with same verdict
    """
    if not perturbations:
        return 1.0, 1.0
    
    margin = geometry_config.get("margin", 0.05)
    img_width, img_height = image.size
    
    # Get base detections for IoU comparison
    base_detections = detector.detect(image, labels)
    base_det_a = select_best_detection(base_detections, labels[0], thresholds, img_width, img_height)
    base_det_b = select_best_detection(base_detections, labels[1], thresholds, img_width, img_height)
    
    verdict_matches = 0
    iou_scores = []
    successful_perturbs = 0
    missing_perturbs = 0  # FIX 1C.3: Track perturbations with missing detections
    
    for perturb in perturbations:
        try:
            perturbed = apply_perturbation(image, perturb)
            detections = detector.detect(perturbed, labels)
            
            det_a = select_best_detection(detections, labels[0], thresholds, img_width, img_height)
            det_b = select_best_detection(detections, labels[1], thresholds, img_width, img_height)
            
            if det_a is None or det_b is None:
                # FIX 1C.3: Missing detections under perturbation count as instability
                # (verdict would be UNDECIDABLE which doesn't match base PASS/FAIL)
                missing_perturbs += 1
                continue
            
            successful_perturbs += 1
            
            # Compute verdict on perturbed image
            delta = compute_geometric_delta(
                det_a.box_xyxy, det_b.box_xyxy, relation, img_width, img_height
            )
            verdict, _ = evaluate_geometry(delta, relation, margin)
            
            if verdict == base_verdict:
                verdict_matches += 1
            
            # Compute IoU stability
            if base_det_a is not None:
                iou_scores.append(compute_iou(base_det_a.box_xyxy, det_a.box_xyxy))
            if base_det_b is not None:
                iou_scores.append(compute_iou(base_det_b.box_xyxy, det_b.box_xyxy))
                
        except Exception as e:
            logger.warning(f"Perturbation failed: {perturb['type']}: {e}")
            missing_perturbs += 1
            continue
    
    n_perturbs = len(perturbations)
    
    # FIX 1C.3: Properly handle cases where perturbations fail to detect objects
    # If all perturbations produce missing detections, stability should be near 0, not 0.5
    if successful_perturbs == 0:
        # All perturbations failed → very unstable (was incorrectly 0.5 before)
        return 0.0, 0.0
    
    # Verdict stability: fraction of all perturbations with matching verdict
    # Missing detections count as non-matches (base was PASS/FAIL, perturbed would be UNDECIDABLE)
    verdict_stability = verdict_matches / n_perturbs if n_perturbs > 0 else 0.0
    
    # IoU stability: average IoU for successful detections only
    iou_stability = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
    
    # Combined stability (per PROJECT.md Section 6.3)
    stability = 0.5 * verdict_stability + 0.5 * iou_stability
    
    return stability, verdict_stability


# =============================================================================
# Detection Selection
# =============================================================================

def select_best_detection(
    detections: list[Detection], 
    target_label: str,
    thresholds: dict,
    image_width: int | None = None,
    image_height: int | None = None,
) -> Detection | None:
    """Select the best detection for a target label.
    
    FIX (per FIX_PLAN.md 1C.1): Now applies min_area_fraction filtering.
    
    Args:
        detections: List of Detection objects.
        target_label: The label to match.
        thresholds: Dict with detection thresholds.
        image_width: Image width for area fraction calculation.
        image_height: Image height for area fraction calculation.
        
    Returns:
        Best matching Detection or None.
    """
    target = target_label.lower()
    min_score = thresholds.get("detection_score", 0.3)
    min_area_frac = thresholds.get("min_area_fraction", 0.005)
    
    # Filter to matching labels
    matching = [d for d in detections if d.label.lower() == target]
    
    if not matching:
        return None
    
    # Filter by score threshold
    matching = [d for d in matching if d.score >= min_score]
    
    if not matching:
        return None
    
    # FIX: Apply min_area_fraction filter (per FIX_PLAN.md 1C.1)
    # Filter out tiny spurious detections that are likely noise
    if image_width is not None and image_height is not None:
        image_area = image_width * image_height
        min_area = min_area_frac * image_area
        
        def get_box_area(d: Detection) -> float:
            box = d.box_xyxy
            return (box[2] - box[0]) * (box[3] - box[1])
        
        matching = [d for d in matching if get_box_area(d) >= min_area]
        
        if not matching:
            return None
    
    # Sort by score descending
    matching.sort(key=lambda d: d.score, reverse=True)
    
    return matching[0]


def check_ambiguity(
    detections: list[Detection],
    target_label: str,
    thresholds: dict,
) -> bool:
    """Check if there's ambiguous multi-instance detection.
    
    Args:
        detections: List of Detection objects.
        target_label: The label to check.
        thresholds: Detection thresholds including ambiguity_delta.
        
    Returns:
        True if ambiguous (multiple high-confidence instances).
    """
    target = target_label.lower()
    min_score = thresholds.get("detection_score", 0.3)
    ambiguity_delta = thresholds.get("ambiguity_delta", 0.1)
    
    # Filter to matching labels with sufficient score
    matching = [
        d for d in detections 
        if d.label.lower() == target and d.score >= min_score
    ]
    
    if len(matching) < 2:
        return False
    
    # Sort by score descending
    matching.sort(key=lambda d: d.score, reverse=True)
    
    # Check if top two are too close
    top_score = matching[0].score
    second_score = matching[1].score
    
    return (top_score - second_score) < ambiguity_delta


# =============================================================================
# Main Evaluation Logic
# =============================================================================

@dataclass
class EvaluationResult:
    """Result of evaluating a single sample."""
    sample_id: str
    prompt_id: str
    relation: str
    object_a: str
    object_b: str
    verdict_raw: str
    verdict_reason: str | None
    conf: float
    conf_components: dict
    detectors: dict
    selected: dict
    geometry: dict
    overlay_path: str | None = None


def evaluate_sample(
    sample: dict,
    image: Image.Image,
    primary_detector,
    secondary_detector,
    config: dict,
) -> EvaluationResult:
    """Evaluate a single sample.
    
    Args:
        sample: Manifest sample dict.
        image: The generated image.
        primary_detector: Primary detector (e.g., Faster R-CNN).
        secondary_detector: Secondary detector (e.g., GroundingDINO) or None.
        config: Checker configuration.
        
    Returns:
        EvaluationResult with verdict and metrics.
    """
    sample_id = sample["sample_id"]
    prompt_id = sample["prompt_id"]
    relation = sample["relation"]
    object_a = sample["object_a"]
    object_b = sample["object_b"]
    
    thresholds = config.get("thresholds", {})
    geometry_config = config.get("geometry", {})
    stability_config = config.get("stability", {})
    confidence_config = config.get("confidence", {})
    
    margin = geometry_config.get("margin", 0.05)
    weights = confidence_config.get("weights", {})
    geom_slope = confidence_config.get("geom_slope", 0.15)
    
    labels = [object_a, object_b]
    img_width, img_height = image.size
    
    # Run primary detector
    primary_detections = primary_detector.detect(image, labels)
    det_a = select_best_detection(primary_detections, object_a, thresholds, img_width, img_height)
    det_b = select_best_detection(primary_detections, object_b, thresholds, img_width, img_height)
    
    # Store detector results
    detector_results = {
        "primary": {
            "type": config.get("detector", {}).get("primary", {}).get("type", "unknown"),
            "detections": [
                {
                    "label": d.label,
                    "score": d.score,
                    "box_xyxy": list(d.box_xyxy),
                }
                for d in primary_detections
            ],
        }
    }
    
    # Check for missing detections
    if det_a is None or det_b is None:
        missing = []
        if det_a is None:
            missing.append(object_a)
        if det_b is None:
            missing.append(object_b)
        
        return EvaluationResult(
            sample_id=sample_id,
            prompt_id=prompt_id,
            relation=relation,
            object_a=object_a,
            object_b=object_b,
            verdict_raw=Verdict.UNDECIDABLE.value,
            verdict_reason=f"missing: {', '.join(missing)}",
            conf=0.0,
            conf_components={"detection": 0.0, "geometry": 0.0, "stability": 0.0, "agreement": 0.0},
            detectors=detector_results,
            selected={"box_a_xyxy": None, "box_b_xyxy": None, "score_a": None, "score_b": None},
            geometry={"delta": None, "margin": margin, "iou_ab": None},
        )
    
    # Check for ambiguity
    if check_ambiguity(primary_detections, object_a, thresholds):
        return EvaluationResult(
            sample_id=sample_id,
            prompt_id=prompt_id,
            relation=relation,
            object_a=object_a,
            object_b=object_b,
            verdict_raw=Verdict.UNDECIDABLE.value,
            verdict_reason=f"ambiguous: multiple {object_a}",
            conf=0.0,
            conf_components={"detection": 0.0, "geometry": 0.0, "stability": 0.0, "agreement": 0.0},
            detectors=detector_results,
            selected={
                "box_a_xyxy": list(det_a.box_xyxy),
                "box_b_xyxy": list(det_b.box_xyxy),
                "score_a": det_a.score,
                "score_b": det_b.score,
            },
            geometry={"delta": None, "margin": margin, "iou_ab": None},
        )
    
    if check_ambiguity(primary_detections, object_b, thresholds):
        return EvaluationResult(
            sample_id=sample_id,
            prompt_id=prompt_id,
            relation=relation,
            object_a=object_a,
            object_b=object_b,
            verdict_raw=Verdict.UNDECIDABLE.value,
            verdict_reason=f"ambiguous: multiple {object_b}",
            conf=0.0,
            conf_components={"detection": 0.0, "geometry": 0.0, "stability": 0.0, "agreement": 0.0},
            detectors=detector_results,
            selected={
                "box_a_xyxy": list(det_a.box_xyxy),
                "box_b_xyxy": list(det_b.box_xyxy),
                "score_a": det_a.score,
                "score_b": det_b.score,
            },
            geometry={"delta": None, "margin": margin, "iou_ab": None},
        )
    
    # Compute geometry
    delta = compute_geometric_delta(
        det_a.box_xyxy, det_b.box_xyxy, relation, img_width, img_height
    )
    iou_ab = compute_iou(det_a.box_xyxy, det_b.box_xyxy)
    
    # Check for high overlap (unreliable for left/right relations)
    max_overlap = thresholds.get("max_overlap_iou", 0.5)
    if relation in ("left_of", "right_of") and iou_ab > max_overlap:
        return EvaluationResult(
            sample_id=sample_id,
            prompt_id=prompt_id,
            relation=relation,
            object_a=object_a,
            object_b=object_b,
            verdict_raw=Verdict.UNDECIDABLE.value,
            verdict_reason=f"high_overlap: IoU={iou_ab:.2f} > {max_overlap}",
            conf=0.0,
            conf_components={"detection": 0.0, "geometry": 0.0, "stability": 0.0, "agreement": 0.0},
            detectors=detector_results,
            selected={
                "box_a_xyxy": list(det_a.box_xyxy),
                "box_b_xyxy": list(det_b.box_xyxy),
                "score_a": det_a.score,
                "score_b": det_b.score,
            },
            geometry={"delta": delta, "margin": margin, "iou_ab": iou_ab},
        )
    
    # Evaluate geometric relation
    verdict, verdict_reason = evaluate_geometry(delta, relation, margin)
    
    # Test stability if enabled
    stability = 1.0
    if stability_config.get("enabled", True) and verdict != Verdict.UNDECIDABLE:
        perturbations = stability_config.get("perturbations", [])
        stability, _ = test_stability(
            image, primary_detector, labels, relation, verdict,
            perturbations, thresholds, geometry_config
        )
        
        # Check consistency threshold
        consistency_threshold = stability_config.get("consistency_threshold", 0.5)
        if stability < consistency_threshold:
            verdict = Verdict.UNDECIDABLE
            verdict_reason = f"unstable: stability={stability:.2f} < {consistency_threshold}"
    
    # Compute agreement with secondary detector
    agreement = 1.0
    if secondary_detector is not None:
        try:
            secondary_detections = secondary_detector.detect(image, labels)
            detector_results["secondary"] = {
                "type": config.get("detector", {}).get("secondary", {}).get("type", "unknown"),
                "detections": [
                    {
                        "label": d.label,
                        "score": d.score,
                        "box_xyxy": list(d.box_xyxy),
                    }
                    for d in secondary_detections
                ],
            }
            
            sec_det_a = select_best_detection(secondary_detections, object_a, thresholds, img_width, img_height)
            sec_det_b = select_best_detection(secondary_detections, object_b, thresholds, img_width, img_height)
            
            if sec_det_a is not None and sec_det_b is not None:
                # Compute agreement on geometry
                sec_delta = compute_geometric_delta(
                    sec_det_a.box_xyxy, sec_det_b.box_xyxy, relation, img_width, img_height
                )
                sec_verdict, _ = evaluate_geometry(sec_delta, relation, margin)
                
                # Agreement: same verdict and same sign of delta
                agreement_verdict = 1.0 if sec_verdict == verdict else 0.0
                agreement_sign = 1.0 if (delta * sec_delta >= 0) else 0.0
                agreement = 0.5 * agreement_verdict + 0.5 * agreement_sign
                
                # Compute IoU between detector boxes for additional agreement metric
                iou_a = compute_iou(det_a.box_xyxy, sec_det_a.box_xyxy)
                iou_b = compute_iou(det_b.box_xyxy, sec_det_b.box_xyxy)
                agreement = 0.5 * agreement + 0.25 * iou_a + 0.25 * iou_b
            else:
                # Secondary detector didn't find both objects
                agreement = 0.5  # Neutral
                
        except Exception as e:
            logger.warning(f"Secondary detector failed for {sample_id}: {e}")
            agreement = 0.5  # Neutral on failure
    
    # Compute overall confidence
    conf, conf_components = compute_confidence(
        det_a.score, det_b.score, delta, margin, geom_slope,
        stability, agreement, weights
    )
    
    return EvaluationResult(
        sample_id=sample_id,
        prompt_id=prompt_id,
        relation=relation,
        object_a=object_a,
        object_b=object_b,
        verdict_raw=verdict.value,
        verdict_reason=verdict_reason,
        conf=conf,
        conf_components=conf_components,
        detectors=detector_results,
        selected={
            "box_a_xyxy": list(det_a.box_xyxy),
            "box_b_xyxy": list(det_b.box_xyxy),
            "score_a": det_a.score,
            "score_b": det_b.score,
        },
        geometry={"delta": delta, "margin": margin, "iou_ab": iou_ab},
    )


# =============================================================================
# Overlay Generation
# =============================================================================

def generate_overlay(
    image: Image.Image,
    result: EvaluationResult,
    style: dict,
) -> Image.Image:
    """Generate a debug overlay image.
    
    Args:
        image: Original image.
        result: Evaluation result.
        style: Overlay style configuration.
        
    Returns:
        Image with boxes and annotations.
    """
    from PIL import ImageDraw, ImageFont
    
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    
    # Get colors
    box_a_color = tuple(style.get("box_a_color", [255, 0, 0]))
    box_b_color = tuple(style.get("box_b_color", [0, 0, 255]))
    text_color = tuple(style.get("text_color", [255, 255, 255]))
    
    # Draw boxes
    if result.selected.get("box_a_xyxy"):
        box_a = result.selected["box_a_xyxy"]
        draw.rectangle(box_a, outline=box_a_color, width=3)
        # Label
        label_a = f"A: {result.object_a} ({result.selected.get('score_a', 0):.2f})"
        draw.text((box_a[0], box_a[1] - 20), label_a, fill=box_a_color)
    
    if result.selected.get("box_b_xyxy"):
        box_b = result.selected["box_b_xyxy"]
        draw.rectangle(box_b, outline=box_b_color, width=3)
        # Label
        label_b = f"B: {result.object_b} ({result.selected.get('score_b', 0):.2f})"
        draw.text((box_b[0], box_b[1] - 20), label_b, fill=box_b_color)
    
    # Draw verdict info at bottom
    verdict_text = f"{result.verdict_raw}"
    if result.verdict_reason:
        verdict_text += f" ({result.verdict_reason})"
    verdict_text += f" | Conf: {result.conf:.2f}"
    
    # Add relation info
    info_text = f"Relation: {result.relation}"
    if result.geometry.get("delta") is not None:
        info_text += f" | Delta: {result.geometry['delta']:.3f}"
    
    draw.text((10, image.height - 40), verdict_text, fill=text_color)
    draw.text((10, image.height - 20), info_text, fill=text_color)
    
    return overlay


# =============================================================================
# Metrics Aggregation
# =============================================================================

def compute_metrics(results: list[EvaluationResult]) -> dict:
    """Compute aggregated metrics from evaluation results.
    
    Metrics (per PROJECT.md Section 7):
    - coverage = (PASS + FAIL) / N
    - pass_rate_overall = PASS / N
    - pass_rate_cond = PASS / (PASS + FAIL)
    - undecidable_rate = UNDECIDABLE / N
    
    Args:
        results: List of evaluation results.
        
    Returns:
        Dict of aggregated metrics.
    """
    n_total = len(results)
    if n_total == 0:
        return {"error": "no_samples"}
    
    n_pass = sum(1 for r in results if r.verdict_raw == "PASS")
    n_fail = sum(1 for r in results if r.verdict_raw == "FAIL")
    n_undecidable = sum(1 for r in results if r.verdict_raw == "UNDECIDABLE")
    
    n_decided = n_pass + n_fail
    
    # Overall metrics
    metrics = {
        "total_samples": n_total,
        "pass_count": n_pass,
        "fail_count": n_fail,
        "undecidable_count": n_undecidable,
        "coverage": n_decided / n_total if n_total > 0 else 0.0,
        "pass_rate_overall": n_pass / n_total if n_total > 0 else 0.0,
        "pass_rate_conditional": n_pass / n_decided if n_decided > 0 else 0.0,
        "undecidable_rate": n_undecidable / n_total if n_total > 0 else 0.0,
        "mean_confidence": sum(r.conf for r in results) / n_total if n_total > 0 else 0.0,
    }
    
    # Breakdown by relation
    relations = ["left_of", "right_of", "above", "below"]
    metrics["by_relation"] = {}
    
    for rel in relations:
        rel_results = [r for r in results if r.relation == rel]
        n_rel = len(rel_results)
        if n_rel == 0:
            continue
        
        n_rel_pass = sum(1 for r in rel_results if r.verdict_raw == "PASS")
        n_rel_fail = sum(1 for r in rel_results if r.verdict_raw == "FAIL")
        n_rel_decided = n_rel_pass + n_rel_fail
        
        metrics["by_relation"][rel] = {
            "count": n_rel,
            "pass_count": n_rel_pass,
            "fail_count": n_rel_fail,
            "undecidable_count": n_rel - n_rel_decided,
            "coverage": n_rel_decided / n_rel if n_rel > 0 else 0.0,
            "pass_rate_overall": n_rel_pass / n_rel if n_rel > 0 else 0.0,
            "pass_rate_conditional": n_rel_pass / n_rel_decided if n_rel_decided > 0 else 0.0,
        }
    
    # Undecidable reason breakdown
    reasons = {}
    for r in results:
        if r.verdict_raw == "UNDECIDABLE" and r.verdict_reason:
            # Extract reason category
            reason_cat = r.verdict_reason.split(":")[0] if ":" in r.verdict_reason else r.verdict_reason
            reasons[reason_cat] = reasons.get(reason_cat, 0) + 1
    metrics["undecidable_reasons"] = reasons
    
    return metrics


# =============================================================================
# CLI Entry Point
# =============================================================================

def load_manifest(manifest_path: Path) -> list[dict]:
    """Load manifest JSONL file."""
    samples = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate spatial relationships in generated images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to generation manifest.jsonl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/checker_v1.yaml"),
        help="Path to checker configuration YAML",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip generating overlay images",
    )
    parser.add_argument(
        "--no-secondary",
        action="store_true",
        help="Skip secondary detector (faster but no agreement score)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to evaluate (for testing)",
    )
    
    args = parser.parse_args()
    
    # Load config and manifest
    if not args.config.exists():
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)
    
    if not args.manifest.exists():
        logger.error(f"Manifest not found: {args.manifest}")
        sys.exit(1)
    
    config = load_config(args.config)
    samples = load_manifest(args.manifest)
    
    if args.limit:
        samples = samples[:args.limit]
    
    logger.info(f"Loaded {len(samples)} samples from manifest")
    logger.info(f"Config: {args.config}")
    
    # Log code versions for reproducibility
    code_versions = get_code_versions()
    logger.info(f"Code versions: {code_versions}")
    
    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)
    
    # Copy config to output
    with open(args.out / "checker_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Get run directory (manifest parent)
    run_dir = args.manifest.parent
    
    # Initialize detectors
    logger.info("Loading primary detector...")
    primary_config = config.get("detector", {}).get("primary", {"type": "fasterrcnn"})
    primary_detector = get_detector(primary_config)
    primary_detector.warmup()
    
    secondary_detector = None
    if not args.no_secondary:
        secondary_config = config.get("detector", {}).get("secondary")
        if secondary_config:
            logger.info("Loading secondary detector...")
            try:
                secondary_detector = get_detector(secondary_config)
                secondary_detector.warmup()
            except Exception as e:
                logger.warning(f"Failed to load secondary detector: {e}")
                logger.warning("Continuing without secondary detector")
    
    # Create overlays directory if needed
    output_config = config.get("output", {})
    generate_overlays = output_config.get("generate_overlays", True) and not args.no_overlays
    overlays_dir = None
    if generate_overlays:
        overlays_dir = args.out / output_config.get("overlays_dir", "overlays")
        overlays_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate samples
    results = []
    results_file = args.out / output_config.get("results_file", "per_sample.jsonl")
    
    with open(results_file, "w") as f_out:
        for sample in tqdm(samples, desc="Evaluating"):
            # Load image
            image_path = run_dir / sample["image_path"]
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")
                continue
            
            # Evaluate
            try:
                result = evaluate_sample(
                    sample, image, primary_detector, secondary_detector, config
                )
            except Exception as e:
                logger.error(f"Evaluation failed for {sample['sample_id']}: {e}")
                continue
            
            # Generate overlay if enabled
            if generate_overlays:
                try:
                    overlay_style = output_config.get("overlay_style", {})
                    overlay = generate_overlay(image, result, overlay_style)
                    overlay_path = overlays_dir / f"{result.sample_id}.png"
                    overlay.save(overlay_path)
                    result.overlay_path = str(overlay_path.relative_to(args.out))
                except Exception as e:
                    logger.warning(f"Overlay generation failed: {e}")
            
            results.append(result)
            
            # Write result line
            result_dict = asdict(result)
            f_out.write(json.dumps(result_dict) + "\n")
    
    logger.info(f"Evaluated {len(results)} samples")
    logger.info(f"Results written to: {results_file}")
    
    # Compute and save metrics
    metrics = compute_metrics(results)
    metrics["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    metrics["manifest"] = str(args.manifest)
    metrics["config"] = str(args.config)
    metrics["code_version"] = get_code_versions()
    
    metrics_file = args.out / output_config.get("metrics_file", "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics written to: {metrics_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total samples:        {metrics['total_samples']}")
    print(f"Coverage:             {metrics['coverage']:.1%}")
    print(f"Pass rate (overall):  {metrics['pass_rate_overall']:.1%}")
    print(f"Pass rate (cond.):    {metrics['pass_rate_conditional']:.1%}")
    print(f"Undecidable rate:     {metrics['undecidable_rate']:.1%}")
    print(f"Mean confidence:      {metrics['mean_confidence']:.3f}")
    print("=" * 60)
    
    if metrics.get("by_relation"):
        print("\nBy Relation:")
        for rel, rel_metrics in metrics["by_relation"].items():
            print(f"  {rel:10s}: coverage={rel_metrics['coverage']:.1%}, "
                  f"pass_rate={rel_metrics['pass_rate_conditional']:.1%} "
                  f"(n={rel_metrics['count']})")
    
    if metrics.get("undecidable_reasons"):
        print("\nUndecidable Reasons:")
        for reason, count in sorted(metrics["undecidable_reasons"].items(), 
                                   key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    
    # Cleanup
    primary_detector.cleanup()
    if secondary_detector:
        secondary_detector.cleanup()


if __name__ == "__main__":
    main()

