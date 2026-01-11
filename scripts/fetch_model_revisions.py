#!/usr/bin/env python3
"""
Fetch exact HuggingFace model revisions for reproducibility.

This script queries the HuggingFace Hub API to get the current commit SHA
for each model used in the project. Use these SHAs in your config files
to pin models to specific versions.

Usage:
    python scripts/fetch_model_revisions.py

Output:
    Prints the model IDs and their current commit SHAs, formatted for
    easy copy-paste into config YAML files.

Reference: FIX_PLAN.md 1D.1 (Record HF model revisions)
"""

from __future__ import annotations

import sys

# Models used in SpatialBench-UC
# Format: (model_id, description, config_files)
MODELS = [
    # Generators
    (
        "runwayml/stable-diffusion-v1-5",
        "Stable Diffusion 1.5 (prompt-only, ControlNet, BoxDiff)",
        ["gen_sd15_promptonly.yaml", "gen_sd15_controlnet.yaml", "gen_sd15_boxdiff.yaml"],
    ),
    (
        "lllyasviel/sd-controlnet-canny",
        "ControlNet Canny",
        ["gen_sd15_controlnet.yaml"],
    ),
    (
        "masterful/gligen-1-4-generation-text-box",
        "GLIGEN (text+box grounding)",
        ["gen_sd15_gligen.yaml"],
    ),
    # Detectors
    (
        "IDEA-Research/grounding-dino-base",
        "GroundingDINO (open-vocab detector)",
        ["checker_v1.yaml"],
    ),
]


def fetch_revision(model_id: str) -> str | None:
    """
    Fetch the latest commit SHA for a HuggingFace model.
    
    Returns the commit SHA or None if the model is not found.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed.", file=sys.stderr)
        print("Install it with: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)
    
    api = HfApi()
    
    try:
        # Get model info from HuggingFace Hub
        model_info = api.model_info(model_id)
        return model_info.sha
    except Exception as e:
        print(f"  WARNING: Could not fetch {model_id}: {e}", file=sys.stderr)
        return None


def main() -> None:
    """Fetch and print revisions for all models."""
    print("=" * 70)
    print("SpatialBench-UC Model Revisions")
    print("=" * 70)
    print()
    print("Fetching current commit SHAs from HuggingFace Hub...")
    print()
    
    results: list[tuple[str, str, str | None, list[str]]] = []
    
    for model_id, description, configs in MODELS:
        print(f"  Fetching: {model_id}...")
        sha = fetch_revision(model_id)
        results.append((model_id, description, sha, configs))
    
    print()
    print("-" * 70)
    print("RESULTS")
    print("-" * 70)
    print()
    
    for model_id, description, sha, configs in results:
        print(f"Model: {model_id}")
        print(f"  Description: {description}")
        print(f"  Used in: {', '.join(configs)}")
        if sha:
            print(f"  Revision: {sha}")
        else:
            print(f"  Revision: (failed to fetch)")
        print()
    
    print("-" * 70)
    print("YAML SNIPPETS (copy-paste into your configs)")
    print("-" * 70)
    print()
    
    # Group by config file for easy copy-paste
    config_snippets: dict[str, list[str]] = {}
    
    for model_id, description, sha, configs in results:
        if not sha:
            continue
            
        for config in configs:
            if config not in config_snippets:
                config_snippets[config] = []
            
            # Determine the key name based on the model type
            if "controlnet" in model_id.lower() and "canny" in model_id.lower():
                key = "controlnet_revision"
            elif "grounding-dino" in model_id.lower():
                key = "# Add to secondary detector params"
                config_snippets[config].append(f"  model_revision: \"{sha}\"  # {model_id}")
                continue
            else:
                key = "revision"
            
            config_snippets[config].append(f"  {key}: \"{sha}\"  # {model_id}")
    
    for config, snippets in sorted(config_snippets.items()):
        print(f"# {config}")
        print("generator:")
        for snippet in snippets:
            print(snippet)
        print()
    
    print("-" * 70)
    print("NOTES")
    print("-" * 70)
    print("""
1. These SHAs represent the CURRENT state of each model on HuggingFace.
   Re-run this script if you want to update to newer versions.

2. Add the 'revision' field to your generator config:
   
   generator:
     model_id: "runwayml/stable-diffusion-v1-5"
     revision: "<sha-from-above>"
     
3. For ControlNet, also add 'controlnet_revision':
   
   generator:
     model_id: "runwayml/stable-diffusion-v1-5"
     revision: "<sha>"
     controlnet_id: "lllyasviel/sd-controlnet-canny"
     controlnet_revision: "<sha>"

4. For GroundingDINO detector, the revision support needs to be added
   to the detector implementation if not already present.

5. Faster R-CNN uses torchvision weights (not HuggingFace), so no
   revision pinning is needed - the torchvision version pins the weights.
""")


if __name__ == "__main__":
    main()
