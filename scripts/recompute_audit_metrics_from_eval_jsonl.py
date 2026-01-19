#!/usr/bin/env python3
"""
Recompute audit metrics (confusion matrix + risk--coverage) from existing per-sample
evaluation JSONL files and human labels.

This script is intentionally **lightweight**: it does NOT import spatialbench_uc
or require torch/detectors. It only reads:
  - audits/v1/sample.csv (sample_id + run_path)
  - audits/v1/labels_filled.json|csv (human verdicts)
  - runs/.../<eval_subdir>/per_sample.jsonl (checker verdict + confidence)

Typical usage:
  python3 scripts/recompute_audit_metrics_from_eval_jsonl.py \
    --sample audits/v1/sample.csv \
    --labels audits/v1/labels_filled.json \
    --out audits/v1/analysis_calibrated \
    --eval-subdir eval \
    --tau-sweep unique

  python3 scripts/recompute_audit_metrics_from_eval_jsonl.py \
    --sample audits/v1/sample.csv \
    --labels audits/v1/labels_filled.json \
    --out audits/v1/analysis_uncalibrated \
    --eval-subdir eval_precal_20260116_113552 \
    --tau-sweep unique
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


def load_labels(labels_path: Path) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    if labels_path.suffix.lower() == ".json":
        with open(labels_path) as f:
            label_list = json.load(f)
        for label in label_list:
            sample_id = label.get("sample_id")
            if not sample_id:
                continue
            labels[sample_id] = {
                "human_verdict": label.get("human_verdict", ""),
                "notes": label.get("notes", ""),
                "timestamp": label.get("timestamp", ""),
            }
        return labels

    # CSV
    with open(labels_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row.get("sample_id")
            if not sample_id:
                continue
            labels[sample_id] = {
                "human_verdict": row.get("human_verdict", ""),
                "notes": row.get("notes", ""),
                "timestamp": row.get("timestamp", ""),
            }
    return labels


def load_sample_metadata(sample_path: Path) -> dict[str, dict[str, str]]:
    samples: dict[str, dict[str, str]] = {}
    with open(sample_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row.get("sample_id")
            if not sample_id:
                continue
            samples[sample_id] = {k: (v or "") for k, v in row.items()}
    return samples


def _as_repo_path(p: str, repo_root: Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def load_results_for_samples(
    *,
    repo_root: Path,
    sample_metadata: dict[str, dict[str, str]],
    eval_subdir: str,
) -> dict[str, dict[str, Any]]:
    wanted = set(sample_metadata.keys())

    run_paths = sorted(
        {
            _as_repo_path(meta.get("run_path", ""), repo_root)
            for meta in sample_metadata.values()
            if meta.get("run_path", "").strip()
        }
    )

    results: dict[str, dict[str, Any]] = {}
    for run_path in run_paths:
        per_sample = run_path / eval_subdir / "per_sample.jsonl"
        if not per_sample.exists():
            continue
        with open(per_sample) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                sid = obj.get("sample_id")
                if sid in wanted:
                    results[sid] = obj
    return results


def compute_confusion_matrix(
    checker_verdicts: dict[str, str],
    human_verdicts: dict[str, str],
) -> dict[str, Any]:
    common = set(checker_verdicts.keys()) & set(human_verdicts.keys())

    decided_checker: dict[str, str] = {}
    decided_human: dict[str, str] = {}
    undecidable_checker = 0
    undecidable_human = 0

    for sid in common:
        c = checker_verdicts[sid]
        h = human_verdicts[sid]

        if c == "UNDECIDABLE":
            undecidable_checker += 1
        else:
            decided_checker[sid] = c

        if h == "UNDECIDABLE":
            undecidable_human += 1
        else:
            decided_human[sid] = h

    matrix = {"PASS": {"PASS": 0, "FAIL": 0}, "FAIL": {"PASS": 0, "FAIL": 0}}
    for sid in set(decided_checker.keys()) & set(decided_human.keys()):
        c = decided_checker[sid]
        h = decided_human[sid]
        if c in matrix and h in matrix[c]:
            matrix[c][h] += 1

    tp = matrix["PASS"]["PASS"]
    fp = matrix["PASS"]["FAIL"]
    fn = matrix["FAIL"]["PASS"]
    tn = matrix["FAIL"]["FAIL"]

    total_decided = tp + fp + fn + tn
    accuracy = (tp + tn) / total_decided if total_decided > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "confusion_matrix": matrix,
        "undecidable_checker": undecidable_checker,
        "undecidable_human": undecidable_human,
        "total_samples": len(common),
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


def make_tau_values(
    *,
    sample_metadata: dict[str, dict[str, str]],
    results: dict[str, dict[str, Any]],
    strategy: Literal["unique", "grid"],
    grid_step: float,
) -> list[float]:
    if strategy == "grid":
        step = float(grid_step)
        if step <= 0 or step > 1:
            raise ValueError(f"grid_step must be in (0, 1], got {grid_step}")
        n = int(round(1.0 / step))
        taus = [i * step for i in range(n + 1)]
        taus[-1] = 1.0
        return taus

    confs: list[float] = []
    for sid in sample_metadata.keys():
        r = results.get(sid)
        if not r:
            continue
        if r.get("verdict_raw") not in ("PASS", "FAIL"):
            continue
        try:
            conf = float(r.get("conf", 0.0))
        except Exception:
            continue
        if 0.0 <= conf <= 1.0:
            confs.append(conf)

    unique = sorted(set(confs))
    return sorted(set([0.0, 1.0, *unique]))


def compute_risk_coverage_curve(
    *,
    sample_metadata: dict[str, dict[str, str]],
    labels: dict[str, dict[str, str]],
    results: dict[str, dict[str, Any]],
    tau_values: list[float],
) -> list[dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    n_total = len(sample_metadata)

    for tau in tau_values:
        covered: list[str] = []
        for sid in sample_metadata.keys():
            r = results.get(sid)
            if not r:
                continue
            verdict = r.get("verdict_raw", "")
            conf = float(r.get("conf", 0.0) or 0.0)
            if verdict in ("PASS", "FAIL") and conf >= tau:
                covered.append(sid)

        if not covered:
            # At zero coverage the risk is undefined; for plotting/reporting we set
            # risk=0 (no decisions => no observed errors) and accuracy=1 by convention.
            curve.append(
                {"tau": tau, "coverage": 0.0, "risk": 0.0, "accuracy": 1.0, "n_covered": 0, "n_evaluated": 0}
            )
            continue

        correct = 0
        denom = 0
        for sid in covered:
            lab = labels.get(sid)
            if not lab:
                continue
            human = lab.get("human_verdict", "")
            if human == "UNDECIDABLE":
                continue
            denom += 1
            if results[sid].get("verdict_raw", "") == human:
                correct += 1

        accuracy = correct / denom if denom > 0 else 0.0
        risk = 1.0 - accuracy
        coverage = len(covered) / n_total if n_total > 0 else 0.0
        curve.append(
            {
                "tau": tau,
                "coverage": coverage,
                "risk": risk,
                "accuracy": accuracy,
                "n_covered": len(covered),
                "n_evaluated": denom,
            }
        )

    return curve


def plot_risk_coverage(curve: list[dict[str, Any]], out_png: Path) -> None:
    # Optional dependency
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve_sorted = sorted(curve, key=lambda p: p.get("coverage", 0.0))
    xs = [p["coverage"] for p in curve_sorted]
    ys = [p["risk"] for p in curve_sorted]
    taus = [p["tau"] for p in curve_sorted]

    plt.figure(figsize=(7.0, 5.0))
    # Risk--coverage under a unique-threshold sweep is naturally step-like.
    plt.step(xs, ys, where="post", linewidth=2, color="C0")
    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 − Accuracy)")
    plt.title("Risk–coverage (audited subset)")
    plt.grid(True, alpha=0.3)

    # Label a few thresholds to give intuition without clutter.
    if len(taus) <= 25:
        idxs = list(range(0, len(taus), 5))
    else:
        qs = [0.0, 0.25, 0.5, 0.75, 1.0]
        idxs = sorted({min(len(taus) - 1, int(round(q * (len(taus) - 1)))) for q in qs})

    for i in idxs:
        plt.annotate(f"τ={taus[i]:.2f}", (xs[i], ys[i]), xytext=(6, 6), textcoords="offset points", fontsize=9)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute audit metrics from per_sample.jsonl (no torch).")
    parser.add_argument("--sample", type=Path, required=True, help="Path to audits/v1/sample.csv")
    parser.add_argument("--labels", type=Path, required=True, help="Path to labels_filled.json or labels_filled.csv")
    parser.add_argument("--out", type=Path, required=True, help="Output directory (e.g., audits/v1/analysis_calibrated)")
    parser.add_argument("--eval-subdir", type=str, default="eval", help="Eval subdir inside each run_path")
    parser.add_argument("--tau-sweep", choices=("unique", "grid"), default="unique")
    parser.add_argument("--tau-grid-step", type=float, default=0.05)
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Repo root for resolving relative run_path values")

    args = parser.parse_args()
    repo_root = args.repo_root.resolve()

    sample_metadata = load_sample_metadata(args.sample)
    labels = load_labels(args.labels)
    results = load_results_for_samples(repo_root=repo_root, sample_metadata=sample_metadata, eval_subdir=args.eval_subdir)

    checker_verdicts = {sid: r.get("verdict_raw", "") for sid, r in results.items()}
    human_verdicts = {sid: lab.get("human_verdict", "") for sid, lab in labels.items()}

    confusion = compute_confusion_matrix(checker_verdicts, human_verdicts)
    tau_values = make_tau_values(
        sample_metadata=sample_metadata,
        results=results,
        strategy=args.tau_sweep,
        grid_step=args.tau_grid_step,
    )
    curve = compute_risk_coverage_curve(
        sample_metadata=sample_metadata,
        labels=labels,
        results=results,
        tau_values=tau_values,
    )

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "eval_subdir": args.eval_subdir,
        "tau_sweep": {
            "strategy": args.tau_sweep,
            "grid_step": args.tau_grid_step if args.tau_sweep == "grid" else None,
            "n_taus": len(tau_values),
        },
        "confusion_matrix": confusion,
        "risk_coverage_curve": curve,
        "n_samples_in_sample_csv": len(sample_metadata),
        "n_samples_with_eval_results": len(results),
        "n_samples_with_human_labels": len(labels),
    }
    with open(out_dir / "audit_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_risk_coverage(curve, out_dir / "risk_coverage.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

