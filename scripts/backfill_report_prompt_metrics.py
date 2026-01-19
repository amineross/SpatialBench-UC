#!/usr/bin/env python3
"""
Backfill prompt-level metrics (best-of-K and all-of-K) into an existing report directory.

This is useful for older reports that predate the `prompt_metrics.csv` export.

Usage:
  python3 scripts/backfill_report_prompt_metrics.py \
    --report-dir runs/.../reports/v1_calibrated_.../
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_per_sample(per_sample_path: Path) -> list[dict]:
    out: list[dict] = []
    with open(per_sample_path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def compute_prompt_verdicts(samples: list[dict], k: int) -> tuple[dict[str, str], dict[str, str]]:
    """
    Returns:
      best_of: prompt_id -> verdict
      all_of:  prompt_id -> verdict
    """
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_prompt[s["prompt_id"]].append(s)

    best_of: dict[str, str] = {}
    all_of: dict[str, str] = {}

    for pid, lst in by_prompt.items():
        verdicts = [x.get("verdict_raw", "") for x in lst]

        # best-of-K
        if any(v == "PASS" for v in verdicts):
            best_of[pid] = "PASS"
        elif all(v == "FAIL" for v in verdicts):
            best_of[pid] = "FAIL"
        else:
            best_of[pid] = "UNDECIDABLE"

        # all-of-K (strict)
        if len(verdicts) != k:
            all_of[pid] = "UNDECIDABLE"
        elif all(v == "PASS" for v in verdicts):
            all_of[pid] = "PASS"
        elif any(v == "FAIL" for v in verdicts):
            all_of[pid] = "FAIL"
        else:
            all_of[pid] = "UNDECIDABLE"

    return best_of, all_of


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill prompt_metrics.csv into an existing report directory.")
    parser.add_argument("--report-dir", type=Path, required=True, help="Path to an existing report directory")
    args = parser.parse_args()

    report_dir = args.report_dir.resolve()
    meta_path = report_dir / "report_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"Missing {meta_path}")

    meta = json.loads(meta_path.read_text())
    eval_subdir = meta.get("eval_subdir") or "eval"
    k = int(meta.get("k") or 4)
    run_paths = [Path(p) for p in (meta.get("run_paths") or [])]
    if not run_paths:
        raise SystemExit("report_meta.json is missing run_paths")

    rows: list[tuple[str, int, float, float]] = []
    for run_path in run_paths:
        per_sample = run_path / eval_subdir / "per_sample.jsonl"
        if not per_sample.exists():
            raise SystemExit(f"Missing {per_sample}")

        samples = load_per_sample(per_sample)
        best_of, all_of = compute_prompt_verdicts(samples, k=k)

        total_prompts = len(best_of)
        best_pass = sum(1 for v in best_of.values() if v == "PASS")
        all_pass = sum(1 for v in all_of.values() if v == "PASS")
        best_rate = best_pass / total_prompts if total_prompts > 0 else 0.0
        all_rate = all_pass / total_prompts if total_prompts > 0 else 0.0

        rows.append((run_path.name, total_prompts, best_rate, all_rate))

    tables_dir = report_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "prompt_metrics.csv"
    with open(out_path, "w") as f:
        f.write("run,total_prompts,best_of_k_pass_rate,all_of_k_pass_rate\n")
        for run_name, total_prompts, best_rate, all_rate in rows:
            f.write(f"\"{run_name}\",{total_prompts},{best_rate:.4f},{all_rate:.4f}\n")

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

