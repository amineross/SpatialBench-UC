#!/usr/bin/env python3
"""
Human audit analysis and calibration.

Compares human labels to checker verdicts, generates risk-coverage curves,
and performs automatic parameter calibration via grid search.

Usage:
    python -m spatialbench_uc.audit.analyze \
        --labels audits/v1/labels_filled.json \
        --sample audits/v1/sample.csv \
        --out audits/v1/audit_metrics.json \
        [--update-config]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Import evaluation functions for re-running checker logic
from spatialbench_uc.evaluate import (
    evaluate_sample,
    load_config as load_checker_config,
)
from spatialbench_uc.detectors import get_detector

# Optional matplotlib for plots
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_labels(labels_path: Path) -> dict[str, dict]:
    """Load human labels from JSON or CSV.
    
    Returns:
        Dict mapping sample_id -> {human_verdict, notes, timestamp}
    """
    labels = {}
    
    if labels_path.suffix == ".json":
        with open(labels_path) as f:
            label_list = json.load(f)
            for label in label_list:
                sample_id = label.get("sample_id")
                if sample_id:
                    labels[sample_id] = {
                        "human_verdict": label.get("human_verdict", ""),
                        "notes": label.get("notes", ""),
                        "timestamp": label.get("timestamp", ""),
                    }
    else:
        # CSV format
        with open(labels_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = row.get("sample_id")
                if sample_id:
                    labels[sample_id] = {
                        "human_verdict": row.get("human_verdict", ""),
                        "notes": row.get("notes", ""),
                        "timestamp": row.get("timestamp", ""),
                    }
    
    return labels


def load_sample_metadata(sample_path: Path) -> dict[str, dict]:
    """Load sample metadata from CSV.
    
    Returns:
        Dict mapping sample_id -> sample metadata
    """
    samples = {}
    with open(sample_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row.get("sample_id")
            if sample_id:
                samples[sample_id] = dict(row)
    
    return samples


def load_original_results(run_paths: list[Path], eval_subdir: str = "eval") -> dict[str, dict]:
    """Load original checker results from all runs.
    
    Returns:
        Dict mapping sample_id -> evaluation result
    """
    all_results = {}
    
    for run_path in run_paths:
        eval_dir = run_path / eval_subdir
        per_sample_file = eval_dir / "per_sample.jsonl"
        
        if not per_sample_file.exists():
            continue
        
        with open(per_sample_file) as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    sample_id = result.get("sample_id")
                    if sample_id:
                        all_results[sample_id] = result
    
    return all_results


def make_tau_values(
    sample_metadata: dict[str, dict],
    original_results: dict[str, dict],
    *,
    strategy: str,
    grid_step: float,
) -> list[float]:
    """
    Build the list of confidence thresholds (tau) to sweep for risk--coverage.

    - strategy="grid": tau in {0, grid_step, ..., 1.0}
    - strategy="unique": sweep over all unique confidence values observed on the audited subset
    """
    if strategy == "grid":
        step = float(grid_step)
        if step <= 0 or step > 1:
            raise ValueError(f"grid_step must be in (0, 1], got {grid_step}")

        n = int(round(1.0 / step))
        taus = [i * step for i in range(n + 1)]
        taus[-1] = 1.0  # guard against floating rounding
        return taus

    if strategy == "unique":
        confs: list[float] = []
        for sample_id in sample_metadata.keys():
            r = original_results.get(sample_id)
            if not r:
                continue
            verdict = r.get("verdict_raw", "")
            if verdict not in ("PASS", "FAIL"):
                continue
            try:
                conf = float(r.get("conf", 0.0))
            except Exception:
                continue
            if 0.0 <= conf <= 1.0:
                confs.append(conf)

        unique = sorted(set(confs))
        # Always include endpoints for interpretability and to ensure a 0-coverage point exists.
        taus = sorted(set([0.0, 1.0, *unique]))
        return taus

    raise ValueError(f"Unknown tau sweep strategy: {strategy!r}")


def compute_confusion_matrix(
    checker_verdicts: dict[str, str],
    human_verdicts: dict[str, str],
) -> dict[str, Any]:
    """Compute confusion matrix between checker and human verdicts.
    
    Returns:
        Dict with confusion matrix and metrics
    """
    # Filter to samples with both verdicts
    common_samples = set(checker_verdicts.keys()) & set(human_verdicts.keys())
    
    # Separate UNDECIDABLE from PASS/FAIL
    decided_checker = {}
    decided_human = {}
    undecidable_checker = 0
    undecidable_human = 0
    
    for sample_id in common_samples:
        c_verdict = checker_verdicts[sample_id]
        h_verdict = human_verdicts[sample_id]
        
        if c_verdict == "UNDECIDABLE":
            undecidable_checker += 1
        else:
            decided_checker[sample_id] = c_verdict
        
        if h_verdict == "UNDECIDABLE":
            undecidable_human += 1
        else:
            decided_human[sample_id] = h_verdict
    
    # Confusion matrix for decided cases
    matrix = {
        "PASS": {"PASS": 0, "FAIL": 0},
        "FAIL": {"PASS": 0, "FAIL": 0},
    }
    
    for sample_id in set(decided_checker.keys()) & set(decided_human.keys()):
        c_verdict = decided_checker[sample_id]
        h_verdict = decided_human[sample_id]
        if c_verdict in matrix and h_verdict in matrix[c_verdict]:
            matrix[c_verdict][h_verdict] += 1
    
    # Compute metrics
    tp = matrix["PASS"]["PASS"]  # True positive: checker PASS, human PASS
    fp = matrix["PASS"]["FAIL"]  # False positive: checker PASS, human FAIL
    fn = matrix["FAIL"]["PASS"]  # False negative: checker FAIL, human PASS
    tn = matrix["FAIL"]["FAIL"]  # True negative: checker FAIL, human FAIL
    
    total_decided = tp + fp + fn + tn
    accuracy = (tp + tn) / total_decided if total_decided > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "confusion_matrix": matrix,
        "undecidable_checker": undecidable_checker,
        "undecidable_human": undecidable_human,
        "total_samples": len(common_samples),
        "decided_samples": total_decided,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_risk_coverage_curve(
    samples: dict[str, dict],
    labels: dict[str, dict],
    original_results: dict[str, dict],
    tau_values: list[float],
) -> list[dict]:
    """Compute risk-coverage curve for different confidence thresholds.
    
    Args:
        samples: Sample metadata dict
        labels: Human labels dict
        original_results: Original checker results dict
        tau_values: List of confidence thresholds to test
    
    Returns:
        List of {tau, coverage, risk, accuracy} dicts
    """
    curve = []
    
    for tau in tau_values:
        # Filter to samples with verdict AND conf >= tau
        covered_samples = []
        for sample_id in samples.keys():
            if sample_id not in original_results:
                continue
            
            result = original_results[sample_id]
            verdict = result.get("verdict_raw", "")
            conf = result.get("conf", 0.0)
            
            if verdict in ("PASS", "FAIL") and conf >= tau:
                covered_samples.append(sample_id)
        
        if not covered_samples:
            curve.append({
                "tau": tau,
                "coverage": 0.0,
                # At zero coverage the risk is undefined; for plotting/reporting we set
                # risk=0 (no decisions => no observed errors) and accuracy=1 by convention.
                "risk": 0.0,
                "accuracy": 1.0,
                "n_covered": 0,
            })
            continue
        
        # Compute accuracy on covered samples
        correct = 0
        total = 0
        
        for sample_id in covered_samples:
            if sample_id not in labels:
                continue
            
            checker_verdict = original_results[sample_id].get("verdict_raw", "")
            human_verdict = labels[sample_id].get("human_verdict", "")
            
            if human_verdict == "UNDECIDABLE":
                continue  # Skip human UNDECIDABLE for accuracy
            
            if checker_verdict == human_verdict:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        risk = 1.0 - accuracy
        coverage = len(covered_samples) / len(samples) if len(samples) > 0 else 0.0
        
        curve.append({
            "tau": tau,
            "coverage": coverage,
            "risk": risk,
            "accuracy": accuracy,
            "n_covered": len(covered_samples),
            "n_evaluated": total,
        })
    
    return curve


def find_image_path(sample_id: str, run_paths: list[Path]) -> Path | None:
    """Find image path for a sample_id across run directories."""
    for run_path in run_paths:
        candidate = run_path / "images" / f"{sample_id}.png"
        if candidate.exists():
            return candidate
    return None


def _normalize_rel_path(path_str: str) -> str:
    """Normalize CSV-stored paths (Windows/backslashes) to forward slashes."""
    return (path_str or "").replace("\\", "/")


def infer_run_dir_from_media_path(media_path: str) -> Path | None:
    """
    Infer a run directory from a sample.csv media path.

    Expected patterns (relative to sample.csv folder):
      - ../../runs/<...>/<run_name>/images/<sample_id>.png
      - ../../runs/<...>/<run_name>/eval/overlays/<sample_id>.png

    This supports nested runs such as: runs/final_YYYYMMDD_HHMMSS/<method>/
    """
    media_path = _normalize_rel_path(media_path)
    if not media_path:
        return None

    parts = [p for p in media_path.split("/") if p and p not in (".", "..")]
    if "runs" not in parts:
        return None

    i = parts.index("runs")
    # Find the first segment after "runs" that indicates we're inside a run folder
    for j in range(i + 1, len(parts)):
        if parts[j] in ("images", "eval"):
            return Path(*parts[i:j])
    return None


def resolve_media_path(sample_csv_dir: Path, media_path: str) -> Path | None:
    """Resolve a media path from sample.csv to an absolute path."""
    media_path = _normalize_rel_path(media_path)
    if not media_path:
        return None
    p = (sample_csv_dir / Path(media_path)).resolve()
    return p if p.exists() else None


def load_manifests(run_paths: list[Path]) -> dict[str, dict]:
    """Load manifest files to get sample metadata.
    
    Returns:
        Dict mapping sample_id -> manifest record
    """
    manifests = {}
    for run_path in run_paths:
        manifest_file = run_path / "manifest.jsonl"
        if not manifest_file.exists():
            continue
        
        with open(manifest_file) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    sample_id = record.get("sample_id")
                    if sample_id:
                        manifests[sample_id] = record
    
    return manifests


def re_evaluate_with_params(
    sample_metadata: dict,
    original_results: dict[str, dict],
    run_paths: list[Path],
    sample_csv_dir: Path,
    checker_config: dict,
    modified_params: dict,
) -> dict[str, dict]:
    """Re-run checker logic with modified parameters.
    
    This is expensive - only call for calibration grid search.
    
    Args:
        sample_metadata: Sample metadata dict
        original_results: Original checker results (for image paths)
        run_paths: List of run directories
        checker_config: Base checker config
        modified_params: Dict with {margin, detection_score, ...} to override
    
    Returns:
        Dict mapping sample_id -> new evaluation result
    """
    # Create modified config
    modified_config = json.loads(json.dumps(checker_config))  # Deep copy
    
    if "margin" in modified_params:
        modified_config.setdefault("geometry", {})["margin"] = modified_params["margin"]
    
    if "detection_score" in modified_params:
        det_score = modified_params["detection_score"]
        modified_config.setdefault("thresholds", {})["detection_score"] = det_score

        # IMPORTANT: Make detection_score actually affect detector outputs.
        # Otherwise, detector-level filtering (e.g., Faster R-CNN score_threshold=0.5)
        # can make detection_score ineffective for calibration ranges below 0.5.
        detector_cfg = modified_config.setdefault("detector", {})
        primary_cfg = detector_cfg.setdefault("primary", {})
        primary_params = primary_cfg.setdefault("params", {})
        primary_params["score_threshold"] = det_score

        secondary_cfg = detector_cfg.get("secondary")
        if isinstance(secondary_cfg, dict):
            secondary_params = secondary_cfg.setdefault("params", {})
            # Align GroundingDINO box threshold with detection_score for calibration
            if "box_threshold" in secondary_params:
                secondary_params["box_threshold"] = det_score
    
    # Initialize detectors (reuse if possible, but for now create new)
    primary_config = modified_config.get("detector", {}).get("primary", {"type": "fasterrcnn"})
    primary_detector = get_detector(primary_config)
    primary_detector.warmup()
    
    secondary_config = modified_config.get("detector", {}).get("secondary")
    secondary_detector = None
    if secondary_config:
        try:
            secondary_detector = get_detector(secondary_config)
            secondary_detector.warmup()
        except Exception:
            pass
    
    new_results = {}
    
    # Re-evaluate each sample
    from PIL import Image
    
    for sample_id, metadata in sample_metadata.items():
        if sample_id not in original_results:
            continue
        
        # Find image path
        # Prefer resolving the explicit image_path from sample.csv (robust to nested run dirs)
        image_path = resolve_media_path(sample_csv_dir, metadata.get("image_path", "")) or find_image_path(sample_id, run_paths)
        if not image_path or not image_path.exists():
            continue
        
        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue
        
        # Build sample dict for evaluate_sample
        # Use metadata from sample.csv, fallback to original result
        sample_dict = {
            "sample_id": sample_id,
            "prompt_id": metadata.get("prompt_id", original_results[sample_id].get("prompt_id", "")),
            "relation": metadata.get("relation", original_results[sample_id].get("relation", "")),
            "object_a": metadata.get("object_a", original_results[sample_id].get("object_a", "")),
            "object_b": metadata.get("object_b", original_results[sample_id].get("object_b", "")),
        }
        
        # Re-evaluate
        try:
            result = evaluate_sample(
                sample_dict,
                image,
                primary_detector,
                secondary_detector,
                modified_config,
            )
            
            # Convert to dict
            from dataclasses import asdict
            new_results[sample_id] = asdict(result)
        except Exception as e:
            print(f"Warning: Re-evaluation failed for {sample_id}: {e}")
            continue
    
    # Cleanup
    primary_detector.cleanup()
    if secondary_detector:
        secondary_detector.cleanup()
    
    return new_results


def grid_search_calibration(
    sample_metadata: dict,
    labels: dict[str, dict],
    original_results: dict[str, dict],
    run_paths: list[Path],
    sample_csv_dir: Path,
    checker_config: dict,
) -> dict[str, Any]:
    """Perform grid search over parameter space.
    
    Tests combinations of:
    - margin: [0.03, 0.05, 0.07, 0.10]
    - detection_score: [0.2, 0.3, 0.4]
    - tau (confidence threshold): [0.3, 0.5, 0.7]
    
    Returns:
        Dict with best parameters and comparison metrics
    """
    margins = [0.03, 0.05, 0.07, 0.10]
    det_thresholds = [0.2, 0.3, 0.4]
    tau_values = [0.3, 0.5, 0.7]
    
    best_score = float('inf')
    best_params = None
    best_metrics = None
    
    print("Grid search calibration (this may take a while)...")
    total_combinations = len(margins) * len(det_thresholds)
    combination_idx = 0
    
    for margin in margins:
        for det_threshold in det_thresholds:
            combination_idx += 1
            print(f"  Testing margin={margin}, det_threshold={det_threshold} "
                  f"({combination_idx}/{total_combinations})...")
            
            # Re-evaluate with these parameters
            modified_params = {
                "margin": margin,
                "detection_score": det_threshold,
            }
            
            try:
                new_results = re_evaluate_with_params(
                    sample_metadata,
                    original_results,
                    run_paths,
                    sample_csv_dir,
                    checker_config,
                    modified_params,
                )
            except Exception as e:
                print(f"    Error: {e}")
                continue
            
            # Evaluate each tau
            for tau in tau_values:
                # Compute false PASS rate at high confidence
                false_pass_high_conf = 0
                total_high_conf = 0
                
                for sample_id, result in new_results.items():
                    if sample_id not in labels:
                        continue
                    
                    verdict = result.get("verdict_raw", "")
                    conf = result.get("conf", 0.0)
                    human_verdict = labels[sample_id].get("human_verdict", "")
                    
                    if verdict == "PASS" and conf >= tau:
                        total_high_conf += 1
                        if human_verdict == "FAIL":
                            false_pass_high_conf += 1
                
                false_pass_rate = false_pass_high_conf / total_high_conf if total_high_conf > 0 else 1.0
                
                # Compute risk-coverage curve
                curve = compute_risk_coverage_curve(
                    sample_metadata,
                    labels,
                    new_results,
                    [tau],
                )
                
                if not curve:
                    continue
                
                curve_point = curve[0]
                risk = curve_point.get("risk", 1.0)
                coverage = curve_point.get("coverage", 0.0)
                
                # Score: prioritize low false PASS rate, then low risk, then high coverage
                # Lower score is better
                score = (
                    false_pass_rate * 10.0 +  # Weight false PASS heavily
                    risk * 2.0 +
                    (1.0 - coverage) * 0.5
                )
                
                if score < best_score:
                    best_score = score
                    best_params = {
                        "margin": margin,
                        "detection_score": det_threshold,
                        "confidence_threshold": tau,
                    }
                    best_metrics = {
                        "false_pass_rate_high_conf": false_pass_rate,
                        "risk": risk,
                        "coverage": coverage,
                        "score": score,
                    }
    
    return {
        "best_parameters": best_params,
        "best_metrics": best_metrics,
    }


def plot_risk_coverage(
    curve: list[dict],
    output_path: Path,
):
    """Plot risk-coverage curve."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plot")
        return
    
    # Sort by coverage so the curve renders left-to-right.
    curve_sorted = sorted(curve, key=lambda p: p.get("coverage", 0.0))
    taus = [p["tau"] for p in curve_sorted]
    coverages = [p["coverage"] for p in curve_sorted]
    risks = [p["risk"] for p in curve_sorted]
    
    plt.figure(figsize=(8, 6))
    # Risk--coverage under a threshold sweep is step-like when confidences are discrete.
    plt.step(coverages, risks, where="post", linewidth=2, color="C0")
    
    # Annotate a small number of tau values (avoid clutter for dense sweeps)
    if len(taus) <= 25:
        label_every = 5
        label_indices = list(range(0, len(taus), label_every))
    else:
        # label quantiles + endpoints
        q = [0.0, 0.25, 0.5, 0.75, 1.0]
        label_indices = sorted({min(len(taus) - 1, int(round(p * (len(taus) - 1)))) for p in q})

    for i in label_indices:
        tau = taus[i]
        plt.annotate(
            f'τ={tau:.2f}',
            (coverages[i], risks[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
        )
    
    plt.xlabel('Coverage', fontsize=12)
    plt.ylabel('Risk (1 - Accuracy)', fontsize=12)
    plt.title('Risk-Coverage Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze human audit labels and calibrate checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to labels_filled.json or labels_filled.csv",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        required=True,
        help="Path to sample.csv (from audit.sample)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for audit metrics",
    )
    parser.add_argument(
        "--checker-config",
        type=Path,
        default=Path("configs/checker_v1.yaml"),
        help="Path to checker config (for re-evaluation)",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Update checker_v1.yaml with calibrated parameters",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip expensive grid search calibration",
    )
    parser.add_argument(
        "--eval-subdir",
        type=str,
        default="eval",
        help="Evaluation subdirectory to load per-sample outputs from (e.g., eval or eval_precal_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--tau-sweep",
        type=str,
        choices=("unique", "grid"),
        default="unique",
        help="How to sweep tau for the risk--coverage curve",
    )
    parser.add_argument(
        "--tau-grid-step",
        type=float,
        default=0.05,
        help="Step size for tau sweep when --tau-sweep=grid",
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading labels...")
    labels = load_labels(args.labels)
    print(f"Loaded {len(labels)} human labels")
    
    print("Loading sample metadata...")
    sample_metadata = load_sample_metadata(args.sample)
    print(f"Loaded {len(sample_metadata)} samples")
    
    sample_csv_dir = args.sample.parent.resolve()

    # Find run paths from sample metadata (robust to nested run directories)
    run_paths_set: set[Path] = set()
    for metadata in sample_metadata.values():
        # New (recommended): explicit run_path column from audit.sample
        run_path_str = _normalize_rel_path(metadata.get("run_path", ""))
        if run_path_str:
            candidate = Path(run_path_str)
            if candidate.exists():
                run_paths_set.add(candidate.resolve())

        # Backward-compat: infer from image_path / overlay_path patterns
        for col in ("image_path", "overlay_path"):
            inferred = infer_run_dir_from_media_path(metadata.get(col, ""))
            if inferred and inferred.exists():
                run_paths_set.add(inferred.resolve())

        # Legacy fallback: try run_id as runs/<run_id>
        run_id = metadata.get("run_id", "")
        if run_id:
            for candidate in (Path("runs") / run_id, Path(run_id)):
                if candidate.exists():
                    run_paths_set.add(candidate.resolve())
                    break

    run_paths = sorted(run_paths_set)
    print(f"Found {len(run_paths)} run directories")
    
    # Load original checker results
    print("Loading original checker results...")
    original_results = load_original_results(run_paths, eval_subdir=args.eval_subdir)
    print(f"Loaded {len(original_results)} original results")
    
    # Load checker config
    checker_config = load_checker_config(args.checker_config)
    
    # Extract checker and human verdicts
    checker_verdicts = {
        sample_id: result.get("verdict_raw", "")
        for sample_id, result in original_results.items()
    }
    
    human_verdicts = {
        sample_id: label.get("human_verdict", "")
        for sample_id, label in labels.items()
    }
    
    # Compute confusion matrix
    print("\nComputing confusion matrix...")
    confusion = compute_confusion_matrix(checker_verdicts, human_verdicts)
    print(f"Accuracy: {confusion['accuracy']:.3f}")
    print(f"Precision: {confusion['precision']:.3f}")
    print(f"Recall: {confusion['recall']:.3f}")
    print(f"F1: {confusion['f1']:.3f}")
    
    # Compute risk-coverage curve
    print("\nComputing risk-coverage curve...")
    tau_values = make_tau_values(
        sample_metadata,
        original_results,
        strategy=args.tau_sweep,
        grid_step=args.tau_grid_step,
    )
    risk_coverage_curve = compute_risk_coverage_curve(
        sample_metadata,
        labels,
        original_results,
        tau_values,
    )
    
    # Grid search calibration (expensive)
    calibration_results = None
    if not args.skip_calibration:
        print("\nPerforming grid search calibration...")
        calibration_results = grid_search_calibration(
            sample_metadata,
            labels,
            original_results,
            run_paths,
            sample_csv_dir,
            checker_config,
        )
        
        if calibration_results and calibration_results.get("best_parameters"):
            best = calibration_results["best_parameters"]
            print(f"\nBest parameters:")
            print(f"  margin: {best['margin']}")
            print(f"  detection_score: {best['detection_score']}")
            print(f"  confidence_threshold: {best['confidence_threshold']}")
    
    # Compile metrics
    metrics = {
        "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "eval_subdir": args.eval_subdir,
        "tau_sweep": {
            "strategy": args.tau_sweep,
            "grid_step": args.tau_grid_step if args.tau_sweep == "grid" else None,
            "n_taus": len(tau_values),
        },
        "confusion_matrix": confusion,
        "risk_coverage_curve": risk_coverage_curve,
    }
    
    if calibration_results:
        metrics["calibration"] = calibration_results
    
    # Save metrics
    args.out.mkdir(parents=True, exist_ok=True)
    metrics_file = args.out / "audit_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_file}")
    
    # Plot risk-coverage curve
    if HAS_MATPLOTLIB:
        plot_path = args.out / "risk_coverage.png"
        plot_risk_coverage(risk_coverage_curve, plot_path)
        print(f"Saved plot to: {plot_path}")
    
    # Update config if requested
    if args.update_config and calibration_results:
        best_params = calibration_results.get("best_parameters")
        if best_params:
            # Create backup of original config
            backup_path = args.checker_config.with_suffix(
                f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            )
            shutil.copy2(args.checker_config, backup_path)
            print(f"\nCreated backup: {backup_path}")
            
            # Update checker config
            checker_config.setdefault("geometry", {})["margin"] = best_params["margin"]
            checker_config.setdefault("thresholds", {})["detection_score"] = best_params["detection_score"]
            
            # Note: confidence_threshold is not stored in config (it's used for filtering, not evaluation)
            # It's documented in audit_metrics.json instead
            
            # Save updated config
            with open(args.checker_config, "w") as f:
                yaml.dump(checker_config, f, default_flow_style=False)
            print(f"Updated checker config: {args.checker_config}")
            print(f"  margin: {best_params['margin']}")
            print(f"  detection_score: {best_params['detection_score']}")
            print(f"  (confidence_threshold={best_params['confidence_threshold']} is for filtering, not stored in config)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

