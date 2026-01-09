"""
Build benchmark prompts with counterfactual pairs.

This module generates the SpatialBench-UC prompt dataset:
- 50 object pairs × 4 relations = 200 prompts
- Each prompt has a linked counterfactual (left_of ↔ right_of, above ↔ below)
- Output: prompts.jsonl, dataset_meta.json, sha256.txt

Usage:
    python -m spatialbench_uc.build_prompts --config configs/prompts_v1.yaml

Reference: PROJECT.md Section 4 (Construction du benchmark)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ObjectInfo:
    """Represents an object from the COCO subset."""

    name: str
    coco_id: int
    category: str
    notes: str = ""


@dataclass
class PromptRecord:
    """A single prompt record following PROJECT.md Section 2.2 format."""

    prompt_id: str
    pair_id: str
    version: str
    language: str
    relation: str
    object_a: dict[str, str]
    object_b: dict[str, str]
    prompt: str
    counterfactual: dict[str, str]
    template: str
    difficulty_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt_id": self.prompt_id,
            "pair_id": self.pair_id,
            "version": self.version,
            "language": self.language,
            "relation": self.relation,
            "object_a": self.object_a,
            "object_b": self.object_b,
            "prompt": self.prompt,
            "counterfactual": self.counterfactual,
            "template": self.template,
            "difficulty_tags": self.difficulty_tags,
        }


# =============================================================================
# Counterfactual Mapping
# =============================================================================

# Maps base relations to their counterfactuals
COUNTERFACTUAL_MAP = {
    "left_of": "right_of",
    "right_of": "left_of",
    "above": "below",
    "below": "above",
}


# =============================================================================
# Core Functions
# =============================================================================


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate configuration YAML."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required = ["version", "objects_file", "generation", "output"]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    return config


def load_objects(objects_file: Path) -> list[ObjectInfo]:
    """Load objects from the COCO subset JSON file."""
    with open(objects_file) as f:
        data = json.load(f)

    objects = []
    for obj in data["objects"]:
        objects.append(
            ObjectInfo(
                name=obj["name"],
                coco_id=obj["coco_id"],
                category=obj["category"],
                notes=obj.get("notes", ""),
            )
        )

    return objects


def sample_object_pairs(
    objects: list[ObjectInfo], num_pairs: int, seed: int
) -> list[tuple[ObjectInfo, ObjectInfo]]:
    """
    Sample unique object pairs (A, B) where A != B.

    Uses seeded random for reproducibility.
    Samples without replacement from all possible pairs.
    """
    rng = random.Random(seed)

    # Generate all possible pairs (A, B) where A != B
    all_pairs = []
    for i, obj_a in enumerate(objects):
        for j, obj_b in enumerate(objects):
            if i != j:
                all_pairs.append((obj_a, obj_b))

    # Check we have enough pairs
    if len(all_pairs) < num_pairs:
        raise ValueError(
            f"Not enough unique pairs. Have {len(all_pairs)}, need {num_pairs}. "
            f"With {len(objects)} objects, max pairs = {len(objects) * (len(objects) - 1)}"
        )

    # Sample without replacement
    sampled = rng.sample(all_pairs, num_pairs)
    return sampled


def generate_prompt_text(template: str, obj_a: ObjectInfo, obj_b: ObjectInfo) -> str:
    """Generate prompt text from template and objects."""
    return template.format(A=obj_a.name, B=obj_b.name)


def build_prompts(
    pairs: list[tuple[ObjectInfo, ObjectInfo]],
    relations: list[str],
    templates: dict[str, str],
    version: str,
    language: str,
) -> list[PromptRecord]:
    """
    Build all prompts with counterfactual linking.

    For each pair (A, B) and each base relation, we create:
    - A prompt with the base relation (e.g., "A left of B")
    - A linked counterfactual (e.g., "B right of A")

    CANONICAL CONTRACT (per FIX_PLAN.md D1):
    - Every prompt record must satisfy: Text describes `object_a relation object_b`
    - For counterfactuals (right_of, below): object_a/object_b are SWAPPED in metadata
      to match the text semantics and checker's evaluation logic.
    
    Counterfactual definition (per FIX_PLAN.md D2):
    - `A left_of B` ↔ `B right_of A`
    - `A above B` ↔ `B below A`
    - cf.relation == inverse(base.relation)
    - cf.object_a == base.object_b and cf.object_b == base.object_a

    Returns a flat list of all prompts.
    """
    prompts = []
    prompt_counter = 0
    pair_counter = 0

    for obj_a, obj_b in pairs:
        for base_relation in relations:
            cf_relation = COUNTERFACTUAL_MAP[base_relation]

            # Generate prompt IDs
            base_prompt_id = f"v1_{prompt_counter:06d}"
            cf_prompt_id = f"v1_{prompt_counter + 1:06d}"
            pair_id = f"pair_{pair_counter:06d}"

            # Generate prompt texts
            base_text = generate_prompt_text(templates[base_relation], obj_a, obj_b)
            cf_text = generate_prompt_text(templates[cf_relation], obj_a, obj_b)

            # Determine template name
            template_name = "photo_simple_v1"

            # Difficulty tags
            difficulty_tags = ["two_objects", "spatial"]
            # Add multi-word tag if either object has spaces
            if " " in obj_a.name or " " in obj_b.name:
                difficulty_tags.append("multi_word_object")

            # Create base prompt record
            # For base relations (left_of, above): object_a is A, object_b is B
            # Text: "A photo of a {A} to the left of a {B}" → A left_of B
            base_record = PromptRecord(
                prompt_id=base_prompt_id,
                pair_id=pair_id,
                version=version,
                language=language,
                relation=base_relation,
                object_a={"name": obj_a.name, "coco_label": obj_a.name},
                object_b={"name": obj_b.name, "coco_label": obj_b.name},
                prompt=base_text,
                counterfactual={
                    "prompt_id": cf_prompt_id,
                    "relation": cf_relation,
                    "prompt": cf_text,
                },
                template=template_name,
                difficulty_tags=difficulty_tags.copy(),
            )

            # Create counterfactual prompt record
            # CRITICAL FIX (per FIX_PLAN.md D1 Option B):
            # For counterfactual (right_of, below), SWAP object_a and object_b
            # so that the metadata matches the text and checker semantics.
            # Text: "A photo of a {B} to the right of a {A}" → B right_of A
            # Therefore: object_a = B, object_b = A
            cf_record = PromptRecord(
                prompt_id=cf_prompt_id,
                pair_id=pair_id,
                version=version,
                language=language,
                relation=cf_relation,
                object_a={"name": obj_b.name, "coco_label": obj_b.name},  # SWAPPED: B is now object_a
                object_b={"name": obj_a.name, "coco_label": obj_a.name},  # SWAPPED: A is now object_b
                prompt=cf_text,
                counterfactual={
                    "prompt_id": base_prompt_id,
                    "relation": base_relation,
                    "prompt": base_text,
                },
                template=template_name,
                difficulty_tags=difficulty_tags.copy(),
            )

            prompts.extend([base_record, cf_record])
            prompt_counter += 2
            pair_counter += 1

    return prompts


# =============================================================================
# Validation
# =============================================================================


def validate_prompts(prompts: list[PromptRecord], config: dict[str, Any]) -> None:
    """
    Validate the generated prompts according to PROJECT.md Section 4.6 and FIX_PLAN.md.

    Checks:
    - Unique prompt_id
    - Symmetric counterfactual linking
    - Balanced relations
    - No self-pairs (A == B)
    - PROMPT CONTRACT: Text matches (object_a, relation, object_b) metadata
    - COUNTERFACTUAL CONTRACT: cf.object_a == base.object_b and cf.object_b == base.object_a
    """
    validation_config = config.get("validation", {})

    # Build lookup for validation
    prompt_by_id = {p.prompt_id: p for p in prompts}

    # Check 1: Unique prompt_id
    if len(prompt_by_id) != len(prompts):
        raise ValueError("Duplicate prompt_id found!")

    # Check 2: No self-pairs
    if validation_config.get("no_self_pairs", True):
        for p in prompts:
            if p.object_a["name"] == p.object_b["name"]:
                raise ValueError(f"Self-pair found: {p.prompt_id} has A == B == {p.object_a['name']}")

    # Check 3: Counterfactual links are symmetric
    if validation_config.get("verify_counterfactuals", True):
        for p in prompts:
            cf_id = p.counterfactual["prompt_id"]
            if cf_id not in prompt_by_id:
                raise ValueError(f"Counterfactual {cf_id} not found for prompt {p.prompt_id}")

            cf_prompt = prompt_by_id[cf_id]
            if cf_prompt.counterfactual["prompt_id"] != p.prompt_id:
                raise ValueError(
                    f"Asymmetric counterfactual link: {p.prompt_id} -> {cf_id} "
                    f"but {cf_id} -> {cf_prompt.counterfactual['prompt_id']}"
                )

    # Check 4: Balanced relations
    if validation_config.get("check_balance", True):
        relation_counts = Counter(p.relation for p in prompts)
        counts = list(relation_counts.values())
        if len(set(counts)) > 1:
            print(f"Warning: Unbalanced relations: {dict(relation_counts)}")
        else:
            print(f"Relations balanced: {dict(relation_counts)}")

    # Check 5: PROMPT CONTRACT (per FIX_PLAN.md D1)
    # Verify that prompt text matches (object_a, relation, object_b) metadata
    # The text should describe "object_a relation object_b"
    print("Validating prompt contract...")
    contract_violations = []
    for p in prompts:
        obj_a_name = p.object_a["name"]
        obj_b_name = p.object_b["name"]
        prompt_lower = p.prompt.lower()
        
        # Check that object_a appears before object_b in the prompt text
        # (this validates the text matches the metadata semantics)
        pos_a = prompt_lower.find(obj_a_name.lower())
        pos_b = prompt_lower.find(obj_b_name.lower())
        
        if pos_a == -1:
            contract_violations.append(f"{p.prompt_id}: object_a '{obj_a_name}' not found in prompt")
        elif pos_b == -1:
            contract_violations.append(f"{p.prompt_id}: object_b '{obj_b_name}' not found in prompt")
        elif pos_a > pos_b:
            # object_a should appear before object_b in the text for all our relations
            contract_violations.append(
                f"{p.prompt_id}: object_a '{obj_a_name}' appears after object_b '{obj_b_name}' "
                f"in prompt (violates 'object_a {p.relation} object_b' contract)"
            )
    
    if contract_violations:
        for v in contract_violations[:10]:  # Show first 10
            print(f"  CONTRACT VIOLATION: {v}")
        raise ValueError(f"Found {len(contract_violations)} prompt contract violations! See above.")
    print(f"  ✓ All {len(prompts)} prompts satisfy text↔metadata contract")

    # Check 6: COUNTERFACTUAL CONTRACT (per FIX_PLAN.md D2)
    # Verify: cf.relation == inverse(base.relation) and cf.object_a == base.object_b
    print("Validating counterfactual contract...")
    cf_violations = []
    processed_pairs = set()
    
    for p in prompts:
        cf_id = p.counterfactual["prompt_id"]
        pair_key = tuple(sorted([p.prompt_id, cf_id]))
        
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)
        
        cf_prompt = prompt_by_id[cf_id]
        
        # Verify relation is inverse
        expected_cf_relation = COUNTERFACTUAL_MAP.get(p.relation)
        if cf_prompt.relation != expected_cf_relation:
            cf_violations.append(
                f"{p.prompt_id}↔{cf_id}: relation mismatch. "
                f"Expected cf.relation={expected_cf_relation}, got {cf_prompt.relation}"
            )
        
        # Verify object swap: cf.object_a == base.object_b and cf.object_b == base.object_a
        if cf_prompt.object_a["name"] != p.object_b["name"]:
            cf_violations.append(
                f"{p.prompt_id}↔{cf_id}: object swap violation. "
                f"Expected cf.object_a='{p.object_b['name']}', got '{cf_prompt.object_a['name']}'"
            )
        if cf_prompt.object_b["name"] != p.object_a["name"]:
            cf_violations.append(
                f"{p.prompt_id}↔{cf_id}: object swap violation. "
                f"Expected cf.object_b='{p.object_a['name']}', got '{cf_prompt.object_b['name']}'"
            )
    
    if cf_violations:
        for v in cf_violations[:10]:  # Show first 10
            print(f"  CF VIOLATION: {v}")
        raise ValueError(f"Found {len(cf_violations)} counterfactual contract violations! See above.")
    print(f"  ✓ All {len(processed_pairs)} counterfactual pairs satisfy D2 contract")

    print(f"Validation passed: {len(prompts)} prompts OK")


# =============================================================================
# Output Writers
# =============================================================================


def write_prompts_jsonl(prompts: list[PromptRecord], output_path: Path) -> str:
    """
    Write prompts to JSONL file.

    Returns the SHA256 hash of the file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt.to_dict(), ensure_ascii=False) + "\n")

    # Compute SHA256
    sha256_hash = hashlib.sha256()
    with open(output_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


def write_dataset_meta(
    prompts: list[PromptRecord],
    pairs: list[tuple[ObjectInfo, ObjectInfo]],
    config: dict[str, Any],
    output_path: Path,
) -> None:
    """Write dataset metadata JSON."""
    # Compute statistics
    relation_counts = Counter(p.relation for p in prompts)
    object_counts = Counter()
    for p in prompts:
        object_counts[p.object_a["name"]] += 1
        object_counts[p.object_b["name"]] += 1

    category_counts = Counter()
    for obj_a, obj_b in pairs:
        category_counts[obj_a.category] += 1
        category_counts[obj_b.category] += 1

    meta = {
        "version": config["version"],
        "name": config["dataset"]["name"],
        "description": config["dataset"]["description"],
        "language": config["dataset"]["language"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "statistics": {
            "total_prompts": len(prompts),
            "total_pairs": len(pairs),
            "prompts_per_pair": len(prompts) // len(pairs) if pairs else 0,
            "relations": dict(relation_counts),
            "unique_objects": len(object_counts),
            "object_distribution": dict(object_counts.most_common()),
            "category_distribution": dict(category_counts),
        },
        "generation_config": {
            "seed": config["generation"]["seed"],
            "num_pairs": config["generation"]["num_pairs"],
            "relations": config["generation"]["relations"],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def write_sha256(sha256_hash: str, output_path: Path) -> None:
    """Write SHA256 hash to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"{sha256_hash}  prompts.jsonl\n")


# =============================================================================
# Main Entry Point
# =============================================================================


def main(args: argparse.Namespace | None = None) -> int:
    """Main entry point for prompt generation."""
    parser = argparse.ArgumentParser(
        description="Generate SpatialBench-UC benchmark prompts with counterfactual pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m spatialbench_uc.build_prompts --config configs/prompts_v1.yaml
    python -m spatialbench_uc.build_prompts --config configs/prompts_v1.yaml --dry-run
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to prompts configuration YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without writing output files",
    )

    if args is None:
        args = parser.parse_args()

    # Load configuration
    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Resolve paths relative to config file location
    config_dir = args.config.parent
    objects_path = Path(config["objects_file"])
    if not objects_path.is_absolute():
        # Try relative to current working directory first
        if not objects_path.exists():
            objects_path = config_dir.parent / config["objects_file"]

    # Load objects
    print(f"Loading objects from {objects_path}")
    objects = load_objects(objects_path)
    print(f"Loaded {len(objects)} objects")

    # Sample pairs
    num_pairs = config["generation"]["num_pairs"]
    seed = config["generation"]["seed"]
    print(f"Sampling {num_pairs} object pairs with seed {seed}")
    pairs = sample_object_pairs(objects, num_pairs, seed)
    print(f"Sampled {len(pairs)} pairs")

    # Get templates and relations
    relations = config["generation"]["relations"]
    templates = config["generation"]["templates"]
    version = config["version"]
    language = config["dataset"]["language"]

    # Build prompts
    print(f"Building prompts for relations: {relations}")
    prompts = build_prompts(pairs, relations, templates, version, language)
    print(f"Generated {len(prompts)} prompts")

    # Validate
    print("Validating prompts...")
    validate_prompts(prompts, config)

    if args.dry_run:
        print("Dry run complete. No files written.")
        return 0

    # Write output files
    output_dir = Path(config["output"]["directory"])
    prompts_path = output_dir / config["output"]["prompts_file"]
    meta_path = output_dir / config["output"]["meta_file"]
    hash_path = output_dir / config["output"]["hash_file"]

    print(f"Writing prompts to {prompts_path}")
    sha256_hash = write_prompts_jsonl(prompts, prompts_path)
    print(f"SHA256: {sha256_hash}")

    print(f"Writing metadata to {meta_path}")
    write_dataset_meta(prompts, pairs, config, meta_path)

    print(f"Writing hash to {hash_path}")
    write_sha256(sha256_hash, hash_path)

    print(f"\nDone! Generated {len(prompts)} prompts in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

