#!/usr/bin/env python3
"""
Generate COCO vocabulary JSON from torchvision pretrained weights.

This script extracts the official COCO class names from torchvision's
FasterRCNN pretrained weights metadata, ensuring we use the exact
vocabulary that the model was trained on.

Usage:
    python scripts/generate_coco_vocab.py

Output:
    data/objects/coco_classes.json - Full COCO vocabulary from torchvision
"""

import json
import sys
from pathlib import Path


def get_coco_classes_from_torchvision() -> list[dict]:
    """
    Extract COCO class information from torchvision pretrained weights.
    
    Returns:
        List of dicts with 'id' and 'name' for each COCO class.
    """
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
    except ImportError:
        print("ERROR: torchvision is not installed.")
        print("Please install PyTorch first:")
        print("  pip install -r requirements/torch-mps.txt  # For Mac")
        print("  pip install -r requirements/torch-cuda121.txt  # For Linux/CUDA")
        sys.exit(1)
    
    # Get the default weights (trained on COCO)
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    
    # Extract categories from weights metadata
    categories = weights.meta["categories"]
    
    # Build structured list with index as ID
    # Note: Index 0 is typically "__background__" in COCO detection
    coco_classes = []
    for idx, name in enumerate(categories):
        coco_classes.append({
            "id": idx,
            "name": name,
            "is_background": (name == "__background__" or idx == 0)
        })
    
    return coco_classes


def main():
    # Determine output path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "data" / "objects"
    output_file = output_dir / "coco_classes.json"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Extracting COCO classes from torchvision...")
    print(f"  Source: FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta['categories']")
    
    # Extract classes
    coco_classes = get_coco_classes_from_torchvision()
    
    # Build output structure
    output_data = {
        "_metadata": {
            "source": "torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT",
            "description": "Official COCO class names from torchvision pretrained weights",
            "generated_by": "scripts/generate_coco_vocab.py",
            "total_classes": len(coco_classes),
            "note": "This file is auto-generated. Do not edit manually."
        },
        "classes": coco_classes
    }
    
    # Write to file
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nGenerated: {output_file}")
    print(f"Total classes: {len(coco_classes)}")
    
    # Print summary
    print("\nClasses extracted:")
    for cls in coco_classes:
        marker = " (background)" if cls["is_background"] else ""
        print(f"  [{cls['id']:2d}] {cls['name']}{marker}")
    
    print("\n✓ COCO vocabulary generated successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

