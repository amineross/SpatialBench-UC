"""
Image generation harness for SpatialBench-UC.

This module generates images from benchmark prompts using pluggable generators.
It produces a manifest file tracking all generated images for reproducibility.

Usage:
    python -m spatialbench_uc.generate \
        --config configs/gen_sd15_promptonly.yaml \
        --prompts data/prompts/v1.0.0/prompts.jsonl \
        --out runs/2026-01-02_sd15_promptonly

Reference: PROJECT.md Section 5 (Génération d'images)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from spatialbench_uc.generators.base import GeneratorConfig
from spatialbench_uc.generators.control_image import (
    create_spatial_edge_map,
    load_control_config,
)
from spatialbench_uc.generators.registry import get_generator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load generation configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_prompts(prompts_path: Path) -> list[dict[str, Any]]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def get_code_versions() -> dict[str, str]:
    """Get versions of key libraries for reproducibility."""
    versions = {}

    try:
        import torch
        versions["torch"] = torch.__version__
    except ImportError:
        pass

    try:
        import diffusers
        versions["diffusers"] = diffusers.__version__
    except ImportError:
        pass

    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except ImportError:
        pass

    try:
        import subprocess
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


def build_sample_id(run_id: str, prompt_id: str, seed: int) -> str:
    """
    Build a unique sample ID for an image.

    Format: {run_id}_{prompt_id}_seed{seed:04d}
    """
    return f"{run_id}_{prompt_id}_seed{seed:04d}"


def build_manifest_record(
    sample_id: str,
    run_id: str,
    prompt_data: dict[str, Any],
    seed: int,
    image_path: Path,
    config: dict[str, Any],
    code_versions: dict[str, str],
) -> dict[str, Any]:
    """
    Build a manifest record for a generated image.

    Follows the format specified in PROJECT.md Section 2.2.
    """
    gen_config = config.get("generator", {})

    return {
        "sample_id": sample_id,
        "run_id": run_id,
        "prompt_id": prompt_data["prompt_id"],
        "image_path": str(image_path),
        "model": {
            "pipeline": "StableDiffusionPipeline" if gen_config.get("mode") == "prompt_only"
                       else "StableDiffusionControlNetPipeline",
            "model_id": gen_config.get("model_id", ""),
            "revision": "main",
        },
        "gen_params": {
            "seed": seed,
            "height": gen_config.get("params", {}).get("height", 512),
            "width": gen_config.get("params", {}).get("width", 512),
            "num_inference_steps": gen_config.get("params", {}).get("num_inference_steps", 30),
            "guidance_scale": gen_config.get("params", {}).get("guidance_scale", 7.5),
            "scheduler": gen_config.get("params", {}).get("scheduler"),
            "negative_prompt": gen_config.get("params", {}).get("negative_prompt"),
        },
        "prompt": prompt_data["prompt"],
        "relation": prompt_data["relation"],
        "object_a": prompt_data["object_a"]["name"],
        "object_b": prompt_data["object_b"]["name"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_version": code_versions,
    }


def main(args: argparse.Namespace | None = None) -> int:
    """Main entry point for image generation."""
    parser = argparse.ArgumentParser(
        description="Generate images for SpatialBench-UC benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate with prompt-only SD 1.5
    python -m spatialbench_uc.generate \\
        --config configs/gen_sd15_promptonly.yaml \\
        --prompts data/prompts/v1.0.0/prompts.jsonl \\
        --out runs/2026-01-02_sd15_promptonly

    # Generate with ControlNet
    python -m spatialbench_uc.generate \\
        --config configs/gen_sd15_controlnet.yaml \\
        --prompts data/prompts/v1.0.0/prompts.jsonl \\
        --out runs/2026-01-02_sd15_controlnet

    # Limit to first N prompts for testing
    python -m spatialbench_uc.generate \\
        --config configs/gen_sd15_promptonly.yaml \\
        --prompts data/prompts/v1.0.0/prompts.jsonl \\
        --out runs/test_run \\
        --limit 2
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to generation config YAML",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="Path to prompts JSONL file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for run artifacts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N prompts (for testing)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Regenerate all images even if they exist",
    )

    if args is None:
        args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Load prompts
    logger.info(f"Loading prompts from {args.prompts}")
    prompts = load_prompts(args.prompts)
    if args.limit:
        prompts = prompts[:args.limit]
    logger.info(f"Loaded {len(prompts)} prompts")

    # Get seeds from config
    seeds = config.get("seeds", [0, 1, 2, 3])
    logger.info(f"Seeds: {seeds}")

    # Calculate total images
    total_images = len(prompts) * len(seeds)
    logger.info(f"Total images to generate: {total_images}")

    # Setup output directory
    out_dir = args.out
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Save control images if ControlNet mode
    gen_config = config.get("generator", {})
    save_control_images = config.get("output", {}).get("save_control_images", False)
    if gen_config.get("mode") == "controlnet" and save_control_images:
        control_images_dir = out_dir / "control_images"
        control_images_dir.mkdir(parents=True, exist_ok=True)
    else:
        control_images_dir = None

    # Copy config to output directory
    config_copy_path = out_dir / "gen_config.yaml"
    if not config_copy_path.exists():
        shutil.copy(args.config, config_copy_path)
        logger.info(f"Saved config to {config_copy_path}")

    # Get code versions
    code_versions = get_code_versions()
    logger.info(f"Code versions: {code_versions}")

    # Extract run_id from output directory name
    run_id = out_dir.name

    # Check for resume
    resume = config.get("resume", True) and not args.no_resume
    manifest_path = out_dir / "manifest.jsonl"

    # Load existing manifest for resume
    existing_samples = set()
    if resume and manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    existing_samples.add(record["sample_id"])
        logger.info(f"Resume mode: found {len(existing_samples)} existing samples")

    # Create generator
    logger.info("Initializing generator...")
    generator_config = GeneratorConfig(
        type=gen_config.get("type", "diffusers"),
        model_id=gen_config.get("model_id", ""),
        mode=gen_config.get("mode", "prompt_only"),
        controlnet_id=gen_config.get("controlnet_id"),
        params=gen_config.get("params", {}),
    )
    generator = get_generator(generator_config)

    # Warmup (loads model)
    logger.info("Loading model (this may take a minute on first run)...")
    generator.warmup()

    # Get control image config for ControlNet
    placement_config, edge_method, edge_style = load_control_config(config)
    logger.info(f"Edge map method: {edge_method}")

    # Open manifest for appending
    manifest_file = open(manifest_path, "a")

    try:
        generated_count = 0
        skipped_count = 0

        for prompt_idx, prompt_data in enumerate(prompts):
            prompt_id = prompt_data["prompt_id"]
            prompt_text = prompt_data["prompt"]
            relation = prompt_data["relation"]

            # Create control image for ControlNet mode
            control_image = None
            if gen_config.get("mode") == "controlnet":
                width = gen_config.get("params", {}).get("width", 512)
                height = gen_config.get("params", {}).get("height", 512)
                control_image = create_spatial_edge_map(
                    relation,
                    width,
                    height,
                    placement_config=placement_config,
                    method=edge_method,
                    style=edge_style,
                )

                # Save control image if configured
                if control_images_dir:
                    control_path = control_images_dir / f"{prompt_id}_control.png"
                    if not control_path.exists():
                        control_image.save(control_path)

            for seed in seeds:
                sample_id = build_sample_id(run_id, prompt_id, seed)

                # Skip if already generated
                if sample_id in existing_samples:
                    skipped_count += 1
                    continue

                # Generate image
                image = generator.generate(
                    prompt=prompt_text,
                    seed=seed,
                    control_image=control_image,
                )

                # Save image
                image_filename = f"{sample_id}.png"
                image_path = images_dir / image_filename
                image.save(image_path)

                # Build and write manifest record
                record = build_manifest_record(
                    sample_id=sample_id,
                    run_id=run_id,
                    prompt_data=prompt_data,
                    seed=seed,
                    image_path=image_path.relative_to(out_dir),
                    config=config,
                    code_versions=code_versions,
                )
                manifest_file.write(json.dumps(record) + "\n")
                manifest_file.flush()

                generated_count += 1
                total_done = generated_count + skipped_count
                logger.info(
                    f"[{total_done}/{total_images}] Generated {sample_id}"
                )

        logger.info(f"Generation complete!")
        logger.info(f"  Generated: {generated_count}")
        logger.info(f"  Skipped (resume): {skipped_count}")
        logger.info(f"  Output: {out_dir}")

    finally:
        manifest_file.close()
        generator.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())

