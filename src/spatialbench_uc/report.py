#!/usr/bin/env python3
"""
Report Generation Module for SpatialBench-UC.

Generates minimalistic academic-style HTML reports with:
- Per-image and per-prompt (Best-of-K) metrics
- Counterfactual consistency analysis
- Comparison tables across multiple runs
- Matplotlib visualizations

Usage:
    python -m spatialbench_uc.report \
        --runs runs/sd15_promptonly runs/sd15_controlnet \
        --config configs/report_v1.yaml \
        --out reports/v1
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Optional imports for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Sample:
    """Single evaluated sample."""
    sample_id: str
    prompt_id: str
    relation: str
    object_a: str
    object_b: str
    verdict: str  # PASS, FAIL, UNDECIDABLE
    verdict_reason: str | None
    confidence: float
    overlay_path: str | None = None
    
    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        return cls(
            sample_id=d["sample_id"],
            prompt_id=d["prompt_id"],
            relation=d["relation"],
            object_a=d["object_a"],
            object_b=d["object_b"],
            verdict=d["verdict_raw"],
            verdict_reason=d.get("verdict_reason"),
            confidence=d.get("conf", 0.0),
            overlay_path=d.get("overlay_path"),
        )


@dataclass
class PromptInfo:
    """Prompt metadata from prompts.jsonl."""
    prompt_id: str
    pair_id: str
    relation: str
    object_a: str
    object_b: str
    prompt_text: str
    counterfactual_id: str | None = None


@dataclass
class RunData:
    """Data from a single evaluation run."""
    name: str
    path: Path
    samples: list[Sample] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    color: str = "#333333"


@dataclass
class ReportMetrics:
    """Aggregated metrics for a run."""
    # Per-image metrics
    total_samples: int = 0
    pass_count: int = 0
    fail_count: int = 0
    undecidable_count: int = 0
    coverage: float = 0.0
    pass_rate_overall: float = 0.0
    pass_rate_conditional: float = 0.0
    undecidable_rate: float = 0.0
    mean_confidence: float = 0.0
    
    # Per-prompt (Best-of-K) metrics
    total_prompts: int = 0
    prompt_pass_count: int = 0
    prompt_fail_count: int = 0
    prompt_undecidable_count: int = 0
    prompt_coverage: float = 0.0
    prompt_pass_rate: float = 0.0
    
    # By relation breakdown
    by_relation: dict = field(default_factory=dict)
    
    # Counterfactual consistency
    cf_total_pairs: int = 0
    cf_both_pass: int = 0
    cf_both_fail: int = 0
    cf_one_sided: int = 0
    cf_undecidable: int = 0
    cf_both_pass_rate: float = 0.0
    cf_one_sided_rate: float = 0.0
    cf_both_fail_rate: float = 0.0


# =============================================================================
# Data Loading
# =============================================================================

def load_samples(eval_dir: Path) -> list[Sample]:
    """Load per_sample.jsonl from evaluation directory."""
    per_sample_path = eval_dir / "per_sample.jsonl"
    if not per_sample_path.exists():
        raise FileNotFoundError(f"Missing per_sample.jsonl in {eval_dir}")
    
    samples = []
    with open(per_sample_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(Sample.from_dict(json.loads(line)))
    return samples


def load_metrics(eval_dir: Path) -> dict:
    """Load metrics.json from evaluation directory."""
    metrics_path = eval_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def load_prompts(prompts_path: Path) -> dict[str, PromptInfo]:
    """Load prompts.jsonl and return dict keyed by prompt_id."""
    prompts = {}
    if not prompts_path.exists():
        return prompts
    
    with open(prompts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompts[d["prompt_id"]] = PromptInfo(
                prompt_id=d["prompt_id"],
                pair_id=d["pair_id"],
                relation=d["relation"],
                object_a=d["object_a"]["name"],
                object_b=d["object_b"]["name"],
                prompt_text=d["prompt"],
                counterfactual_id=d.get("counterfactual", {}).get("prompt_id"),
            )
    return prompts


def load_run(run_path: Path, name: str, color: str = "#333333") -> RunData:
    """Load all data for a single run."""
    run = RunData(name=name, path=run_path, color=color)
    
    eval_dir = run_path / "eval"
    if eval_dir.exists():
        run.samples = load_samples(eval_dir)
        run.metrics = load_metrics(eval_dir)
    
    return run


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_best_of_k(samples: list[Sample], k: int = 4) -> dict[str, str]:
    """
    Compute best-of-K verdict per prompt.
    
    Rules:
    - PASS if any image is PASS
    - FAIL if all images are FAIL
    - UNDECIDABLE otherwise (no PASS, at least one UNDECIDABLE)
    """
    by_prompt: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_prompt[s.prompt_id].append(s)
    
    verdicts = {}
    for prompt_id, prompt_samples in by_prompt.items():
        has_pass = any(s.verdict == "PASS" for s in prompt_samples)
        all_fail = all(s.verdict == "FAIL" for s in prompt_samples)
        
        if has_pass:
            verdicts[prompt_id] = "PASS"
        elif all_fail:
            verdicts[prompt_id] = "FAIL"
        else:
            verdicts[prompt_id] = "UNDECIDABLE"
    
    return verdicts


def compute_counterfactual_consistency(
    prompt_verdicts: dict[str, str],
    prompts: dict[str, PromptInfo],
) -> dict[str, Any]:
    """
    Compute counterfactual consistency metrics.
    
    For each pair (p, p_cf):
    - both_pass: both PASS
    - both_fail: both FAIL  
    - one_sided: one PASS, one FAIL
    - undecidable: at least one UNDECIDABLE
    """
    # Group prompts by pair_id
    pairs: dict[str, list[str]] = defaultdict(list)
    for pid, info in prompts.items():
        pairs[info.pair_id].append(pid)
    
    stats = {
        "total_pairs": 0,
        "both_pass": 0,
        "both_fail": 0,
        "one_sided": 0,
        "undecidable": 0,
        "pair_details": [],  # For debugging
    }
    
    seen_pairs = set()
    for pair_id, prompt_ids in pairs.items():
        # Skip if we've already processed this pair
        if pair_id in seen_pairs:
            continue
        seen_pairs.add(pair_id)
        
        # Get verdicts for both prompts in the pair
        # Note: pair_id groups all 4 prompts (left_of, right_of, above, below)
        # We want to compare counterfactual pairs specifically
        for pid in prompt_ids:
            if pid not in prompts:
                continue
            info = prompts[pid]
            cf_id = info.counterfactual_id
            if not cf_id or cf_id not in prompt_verdicts:
                continue
            if pid not in prompt_verdicts:
                continue
            
            # Only process each pair once (skip the reverse)
            if f"{cf_id}_{pid}" in seen_pairs:
                continue
            seen_pairs.add(f"{pid}_{cf_id}")
            
            v1 = prompt_verdicts[pid]
            v2 = prompt_verdicts[cf_id]
            
            stats["total_pairs"] += 1
            
            if v1 == "UNDECIDABLE" or v2 == "UNDECIDABLE":
                stats["undecidable"] += 1
            elif v1 == "PASS" and v2 == "PASS":
                stats["both_pass"] += 1
            elif v1 == "FAIL" and v2 == "FAIL":
                stats["both_fail"] += 1
            else:
                stats["one_sided"] += 1
                stats["pair_details"].append({
                    "prompt_1": pid,
                    "verdict_1": v1,
                    "prompt_2": cf_id,
                    "verdict_2": v2,
                })
    
    # Compute rates
    total = stats["total_pairs"]
    if total > 0:
        stats["both_pass_rate"] = stats["both_pass"] / total
        stats["both_fail_rate"] = stats["both_fail"] / total
        stats["one_sided_rate"] = stats["one_sided"] / total
        stats["undecidable_rate"] = stats["undecidable"] / total
    else:
        stats["both_pass_rate"] = 0.0
        stats["both_fail_rate"] = 0.0
        stats["one_sided_rate"] = 0.0
        stats["undecidable_rate"] = 0.0
    
    return stats


def compute_metrics(
    run: RunData,
    prompts: dict[str, PromptInfo],
    k: int = 4,
) -> ReportMetrics:
    """Compute all metrics for a run."""
    m = ReportMetrics()
    samples = run.samples
    
    if not samples:
        return m
    
    # Per-image metrics
    m.total_samples = len(samples)
    m.pass_count = sum(1 for s in samples if s.verdict == "PASS")
    m.fail_count = sum(1 for s in samples if s.verdict == "FAIL")
    m.undecidable_count = sum(1 for s in samples if s.verdict == "UNDECIDABLE")
    
    decided = m.pass_count + m.fail_count
    m.coverage = decided / m.total_samples if m.total_samples > 0 else 0.0
    m.pass_rate_overall = m.pass_count / m.total_samples if m.total_samples > 0 else 0.0
    m.pass_rate_conditional = m.pass_count / decided if decided > 0 else 0.0
    m.undecidable_rate = m.undecidable_count / m.total_samples if m.total_samples > 0 else 0.0
    m.mean_confidence = sum(s.confidence for s in samples) / len(samples)
    
    # By relation breakdown
    by_rel: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_rel[s.relation].append(s)
    
    for rel, rel_samples in by_rel.items():
        n = len(rel_samples)
        p = sum(1 for s in rel_samples if s.verdict == "PASS")
        f = sum(1 for s in rel_samples if s.verdict == "FAIL")
        m.by_relation[rel] = {
            "count": n,
            "pass_count": p,
            "fail_count": f,
            "undecidable_count": n - p - f,
            "pass_rate": p / n if n > 0 else 0.0,
            "coverage": (p + f) / n if n > 0 else 0.0,
        }
    
    # Best-of-K metrics
    prompt_verdicts = compute_best_of_k(samples, k)
    m.total_prompts = len(prompt_verdicts)
    m.prompt_pass_count = sum(1 for v in prompt_verdicts.values() if v == "PASS")
    m.prompt_fail_count = sum(1 for v in prompt_verdicts.values() if v == "FAIL")
    m.prompt_undecidable_count = sum(1 for v in prompt_verdicts.values() if v == "UNDECIDABLE")
    
    prompt_decided = m.prompt_pass_count + m.prompt_fail_count
    m.prompt_coverage = prompt_decided / m.total_prompts if m.total_prompts > 0 else 0.0
    m.prompt_pass_rate = m.prompt_pass_count / m.total_prompts if m.total_prompts > 0 else 0.0
    
    # Counterfactual consistency
    if prompts:
        cf_stats = compute_counterfactual_consistency(prompt_verdicts, prompts)
        m.cf_total_pairs = cf_stats["total_pairs"]
        m.cf_both_pass = cf_stats["both_pass"]
        m.cf_both_fail = cf_stats["both_fail"]
        m.cf_one_sided = cf_stats["one_sided"]
        m.cf_undecidable = cf_stats["undecidable"]
        m.cf_both_pass_rate = cf_stats["both_pass_rate"]
        m.cf_one_sided_rate = cf_stats["one_sided_rate"]
        m.cf_both_fail_rate = cf_stats["both_fail_rate"]
    
    return m


# =============================================================================
# Visualization
# =============================================================================

def setup_plot_style():
    """Configure matplotlib for academic-style plots."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


def plot_pass_rate_comparison(
    runs: list[tuple[str, ReportMetrics, str]],
    output_path: Path,
    title: str = "Pass Rate by Method",
):
    """Bar chart comparing pass rates across runs."""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    names = [r[0] for r in runs]
    pass_rates = [r[1].pass_rate_overall * 100 for r in runs]
    colors = [r[2] for r in runs]
    
    x = range(len(names))
    bars = ax.bar(x, pass_rates, color=colors, edgecolor='#333', linewidth=0.5)
    
    ax.set_title(title)
    ax.set_ylabel('Pass Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar, val in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_pass_rate_by_relation(
    runs: list[tuple[str, ReportMetrics, str]],
    output_path: Path,
    title: str = "Pass Rate by Spatial Relation",
):
    """Grouped bar chart: pass rate by relation for each run."""
    setup_plot_style()
    
    # Collect all relations
    all_relations = set()
    for _, metrics, _ in runs:
        all_relations.update(metrics.by_relation.keys())
    relations = sorted(all_relations)
    
    if not relations:
        return
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    n_runs = len(runs)
    n_rels = len(relations)
    width = 0.8 / n_runs
    x = range(n_rels)
    
    for i, (name, metrics, color) in enumerate(runs):
        rates = []
        for rel in relations:
            if rel in metrics.by_relation:
                rates.append(metrics.by_relation[rel]["pass_rate"] * 100)
            else:
                rates.append(0)
        
        offset = (i - n_runs/2 + 0.5) * width
        ax.bar([xi + offset for xi in x], rates, width, 
               label=name, color=color, edgecolor='#333', linewidth=0.5)
    
    ax.set_title(title)
    ax.set_ylabel('Pass Rate (%)')
    ax.set_xlabel('Relation')
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace('_', ' ') for r in relations])
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_confidence_distribution(
    runs: list[tuple[str, list[Sample], str]],
    output_path: Path,
    title: str = "Confidence Score Distribution",
    bins: int = 20,
):
    """Histogram of confidence scores."""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for name, samples, color in runs:
        confs = [s.confidence for s in samples if s.verdict != "UNDECIDABLE"]
        if confs:
            ax.hist(confs, bins=bins, alpha=0.6, label=name, 
                   color=color, edgecolor='#333', linewidth=0.5)
    
    ax.set_title(title)
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 1)
    ax.legend(loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_counterfactual_consistency(
    runs: list[tuple[str, ReportMetrics, str]],
    output_path: Path,
    title: str = "Counterfactual Consistency",
):
    """Stacked bar chart for counterfactual consistency."""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    names = [r[0] for r in runs]
    n = len(names)
    x = range(n)
    
    both_pass = [r[1].cf_both_pass_rate * 100 for r in runs]
    one_sided = [r[1].cf_one_sided_rate * 100 for r in runs]
    both_fail = [r[1].cf_both_fail_rate * 100 for r in runs]
    
    # Stacked bars
    ax.bar(x, both_pass, label='Both Pass', color='#4CAF50', edgecolor='#333', linewidth=0.5)
    ax.bar(x, one_sided, bottom=both_pass, label='One-Sided', color='#FFC107', edgecolor='#333', linewidth=0.5)
    bottom2 = [bp + os for bp, os in zip(both_pass, one_sided)]
    ax.bar(x, both_fail, bottom=bottom2, label='Both Fail', color='#F44336', edgecolor='#333', linewidth=0.5)
    
    ax.set_title(title)
    ax.set_ylabel('Percentage of Pairs (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# HTML Report Generation
# =============================================================================

MINIMAL_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        :root {
            --text: #1a1a2e;
            --bg: #fafafa;
            --accent: #2d3436;
            --border: #ddd;
            --pass: #27ae60;
            --fail: #c0392b;
            --undecidable: #7f8c8d;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Charter', 'Georgia', serif;
            line-height: 1.6;
            color: var(--text);
            background: var(--bg);
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem 1.5rem;
        }
        
        h1 {
            font-size: 1.75rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }
        
        h2 {
            font-size: 1.25rem;
            font-weight: 600;
            margin: 2rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }
        
        h3 {
            font-size: 1rem;
            font-weight: 600;
            margin: 1.5rem 0 0.75rem;
        }
        
        .meta {
            color: #666;
            font-size: 0.875rem;
            margin-bottom: 2rem;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.875rem;
        }
        
        th, td {
            padding: 0.5rem 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        
        th {
            font-weight: 600;
            background: #f5f5f5;
        }
        
        tr:hover { background: #fafafa; }
        
        .num { text-align: right; font-variant-numeric: tabular-nums; }
        
        .pass { color: var(--pass); }
        .fail { color: var(--fail); }
        .undecidable { color: var(--undecidable); }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .metric-card {
            padding: 1rem;
            border: 1px solid var(--border);
            border-radius: 4px;
            background: #fff;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
        }
        
        .metric-label {
            font-size: 0.75rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        figure {
            margin: 1.5rem 0;
            text-align: center;
        }
        
        figure img {
            max-width: 100%;
            height: auto;
            border: 1px solid var(--border);
        }
        
        figcaption {
            font-size: 0.8125rem;
            color: #666;
            margin-top: 0.5rem;
            font-style: italic;
        }
        
        .section { margin-bottom: 2.5rem; }
        
        footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
            font-size: 0.75rem;
            color: #999;
            text-align: center;
        }
        
        @media (max-width: 600px) {
            body { padding: 1rem; }
            h1 { font-size: 1.5rem; }
            .metric-grid { grid-template-columns: 1fr 1fr; }
        }
    </style>
</head>
<body>
    <header>
        <h1>{{ title }}</h1>
        <p class="meta">
            Generated: {{ timestamp }}<br>
            Runs: {{ runs|length }}
        </p>
    </header>
    
    <section class="section">
        <h2>Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Run</th>
                    <th class="num">Samples</th>
                    <th class="num">Pass Rate</th>
                    <th class="num">Coverage</th>
                    <th class="num">Best-of-K Pass</th>
                    <th class="num">Mean Conf.</th>
                </tr>
            </thead>
            <tbody>
            {% for run in runs %}
                <tr>
                    <td><strong>{{ run.name }}</strong></td>
                    <td class="num">{{ run.metrics.total_samples }}</td>
                    <td class="num">{{ "%.1f"|format(run.metrics.pass_rate_overall * 100) }}%</td>
                    <td class="num">{{ "%.1f"|format(run.metrics.coverage * 100) }}%</td>
                    <td class="num">{{ "%.1f"|format(run.metrics.prompt_pass_rate * 100) }}%</td>
                    <td class="num">{{ "%.3f"|format(run.metrics.mean_confidence) }}</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
    </section>
    
    {% if figures.pass_rate_comparison %}
    <section class="section">
        <h2>Pass Rate Comparison</h2>
        <figure>
            <img src="{{ figures.pass_rate_comparison }}" alt="Pass rate comparison">
            <figcaption>Figure 1. Overall pass rate comparison across runs.</figcaption>
        </figure>
    </section>
    {% endif %}
    
    <section class="section">
        <h2>Verdict Distribution</h2>
        {% for run in runs %}
        <h3>{{ run.name }}</h3>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value pass">{{ run.metrics.pass_count }}</div>
                <div class="metric-label">Pass</div>
            </div>
            <div class="metric-card">
                <div class="metric-value fail">{{ run.metrics.fail_count }}</div>
                <div class="metric-label">Fail</div>
            </div>
            <div class="metric-card">
                <div class="metric-value undecidable">{{ run.metrics.undecidable_count }}</div>
                <div class="metric-label">Undecidable</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.1f"|format(run.metrics.pass_rate_conditional * 100) }}%</div>
                <div class="metric-label">Pass | Decided</div>
            </div>
        </div>
        {% endfor %}
    </section>
    
    {% if figures.pass_rate_by_relation %}
    <section class="section">
        <h2>Pass Rate by Relation</h2>
        <figure>
            <img src="{{ figures.pass_rate_by_relation }}" alt="Pass rate by relation">
            <figcaption>Figure 2. Pass rate breakdown by spatial relation.</figcaption>
        </figure>
        
        <table>
            <thead>
                <tr>
                    <th>Relation</th>
                    {% for run in runs %}
                    <th class="num">{{ run.name }}</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
            {% for rel in relations %}
                <tr>
                    <td>{{ rel|replace('_', ' ') }}</td>
                    {% for run in runs %}
                    <td class="num">
                        {% if rel in run.metrics.by_relation %}
                        {{ "%.1f"|format(run.metrics.by_relation[rel].pass_rate * 100) }}%
                        {% else %}
                        —
                        {% endif %}
                    </td>
                    {% endfor %}
                </tr>
            {% endfor %}
            </tbody>
        </table>
    </section>
    {% endif %}
    
    <section class="section">
        <h2>Best-of-K Analysis (K={{ k }})</h2>
        <p>A prompt passes if <em>any</em> of its K generated images pass.</p>
        <table>
            <thead>
                <tr>
                    <th>Run</th>
                    <th class="num">Prompts</th>
                    <th class="num">Pass</th>
                    <th class="num">Fail</th>
                    <th class="num">Undecidable</th>
                    <th class="num">Coverage</th>
                </tr>
            </thead>
            <tbody>
            {% for run in runs %}
                <tr>
                    <td>{{ run.name }}</td>
                    <td class="num">{{ run.metrics.total_prompts }}</td>
                    <td class="num pass">{{ run.metrics.prompt_pass_count }}</td>
                    <td class="num fail">{{ run.metrics.prompt_fail_count }}</td>
                    <td class="num undecidable">{{ run.metrics.prompt_undecidable_count }}</td>
                    <td class="num">{{ "%.1f"|format(run.metrics.prompt_coverage * 100) }}%</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
    </section>
    
    <section class="section">
        <h2>Counterfactual Consistency</h2>
        <p>Pairs of logically equivalent prompts (e.g., "A left of B" ↔ "B right of A") should produce consistent results.</p>
        
        {% if figures.counterfactual_consistency %}
        <figure>
            <img src="{{ figures.counterfactual_consistency }}" alt="Counterfactual consistency">
            <figcaption>Figure 3. Counterfactual pair consistency breakdown.</figcaption>
        </figure>
        {% endif %}
        
        <table>
            <thead>
                <tr>
                    <th>Run</th>
                    <th class="num">Pairs</th>
                    <th class="num">Both Pass</th>
                    <th class="num">One-Sided</th>
                    <th class="num">Both Fail</th>
                    <th class="num">Undecidable</th>
                </tr>
            </thead>
            <tbody>
            {% for run in runs %}
                <tr>
                    <td>{{ run.name }}</td>
                    <td class="num">{{ run.metrics.cf_total_pairs }}</td>
                    <td class="num pass">{{ "%.1f"|format(run.metrics.cf_both_pass_rate * 100) }}%</td>
                    <td class="num" style="color: #f39c12;">{{ "%.1f"|format(run.metrics.cf_one_sided_rate * 100) }}%</td>
                    <td class="num fail">{{ "%.1f"|format(run.metrics.cf_both_fail_rate * 100) }}%</td>
                    <td class="num undecidable">{{ run.metrics.cf_undecidable }}</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
    </section>
    
    {% if figures.confidence_distribution %}
    <section class="section">
        <h2>Confidence Distribution</h2>
        <figure>
            <img src="{{ figures.confidence_distribution }}" alt="Confidence distribution">
            <figcaption>Figure 4. Distribution of confidence scores for decided samples.</figcaption>
        </figure>
    </section>
    {% endif %}
    
    <footer>
        SpatialBench-UC Report • {{ timestamp }}
    </footer>
</body>
</html>
"""


def generate_html_report(
    runs: list[tuple[RunData, ReportMetrics]],
    figures: dict[str, str],
    output_path: Path,
    config: dict,
    k: int = 4,
):
    """Generate HTML report using Jinja2 or fallback template."""
    # Collect all relations
    all_relations = set()
    for _, metrics in runs:
        all_relations.update(metrics.by_relation.keys())
    relations = sorted(all_relations)
    
    # Prepare template data
    template_data = {
        "title": config.get("report", {}).get("title", "SpatialBench-UC Report"),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "runs": [
            {"name": run.name, "metrics": metrics, "color": run.color}
            for run, metrics in runs
        ],
        "figures": figures,
        "relations": relations,
        "k": k,
    }
    
    # Try Jinja2 with custom template, fall back to embedded
    if HAS_JINJA2:
        template_path = Path(config.get("output", {}).get("template", ""))
        if template_path.exists():
            env = Environment(
                loader=FileSystemLoader(template_path.parent),
                autoescape=select_autoescape(['html']),
            )
            template = env.get_template(template_path.name)
        else:
            from jinja2 import Template
            template = Template(MINIMAL_TEMPLATE)
        
        html = template.render(**template_data)
    else:
        # Basic string replacement fallback
        print("Warning: Jinja2 not installed, using simplified template")
        html = "<html><body><h1>Report</h1><p>Install jinja2 for full report.</p></body></html>"
    
    output_path.write_text(html)


# =============================================================================
# CSV Export
# =============================================================================

def export_csv_tables(
    runs: list[tuple[RunData, ReportMetrics]],
    output_dir: Path,
    tables_dir_name: str = "tables",
):
    """Export metrics as CSV files."""
    tables_dir = output_dir / tables_dir_name
    tables_dir.mkdir(exist_ok=True)
    
    # Main results table
    with open(tables_dir / "main_results.csv", "w") as f:
        f.write("run,samples,pass_rate,coverage,pass_rate_cond,prompt_pass_rate,mean_conf\n")
        for run, m in runs:
            f.write(f'"{run.name}",{m.total_samples},{m.pass_rate_overall:.4f},'
                    f'{m.coverage:.4f},{m.pass_rate_conditional:.4f},'
                    f'{m.prompt_pass_rate:.4f},{m.mean_confidence:.4f}\n')
    
    # Counterfactual table
    with open(tables_dir / "counterfactual.csv", "w") as f:
        f.write("run,pairs,both_pass,one_sided,both_fail,undecidable\n")
        for run, m in runs:
            f.write(f'"{run.name}",{m.cf_total_pairs},{m.cf_both_pass_rate:.4f},'
                    f'{m.cf_one_sided_rate:.4f},{m.cf_both_fail_rate:.4f},'
                    f'{m.cf_undecidable}\n')
    
    # By relation breakdown
    all_relations = set()
    for _, m in runs:
        all_relations.update(m.by_relation.keys())
    relations = sorted(all_relations)
    
    with open(tables_dir / "by_relation.csv", "w") as f:
        header = "run," + ",".join(f"{r}_pass_rate" for r in relations)
        f.write(header + "\n")
        for run, m in runs:
            values = []
            for rel in relations:
                if rel in m.by_relation:
                    values.append(f"{m.by_relation[rel]['pass_rate']:.4f}")
                else:
                    values.append("")
            f.write(f'"{run.name}",' + ",".join(values) + "\n")


# =============================================================================
# Main CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate HTML report from SpatialBench-UC evaluation runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Paths to run directories (e.g., runs/sd15_promptonly runs/sd15_controlnet)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/report_v1.yaml"),
        help="Report configuration file",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("data/prompts/v1.0.0/prompts.jsonl"),
        help="Path to prompts.jsonl for counterfactual linking",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for report",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="K value for best-of-K aggregation (default: 4)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating matplotlib plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = {}
    if args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    
    # Get K from config, CLI overrides
    k = args.k
    if args.k == 4:  # Default value, check config
        k = config.get("metrics", {}).get("best_of_k", {}).get("k", 4)
    
    # Get run colors and names from config
    run_configs = {}
    for run_cfg in config.get("runs", []):
        run_configs[run_cfg.get("path", "")] = {
            "name": run_cfg.get("name"),
            "color": run_cfg.get("color", "#333333"),
        }
    
    # Default colors if not in config
    default_colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c"]
    
    # Load prompts for counterfactual linking
    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")
    
    # Load runs
    runs_data: list[tuple[RunData, ReportMetrics]] = []
    for i, run_path_str in enumerate(args.runs):
        run_path = Path(run_path_str)
        if not run_path.exists():
            print(f"Warning: Run path does not exist: {run_path}")
            continue
        
        # Get name and color from config or defaults
        run_cfg = run_configs.get(str(run_path), {})
        name = run_cfg.get("name") or run_path.name
        color = run_cfg.get("color") or default_colors[i % len(default_colors)]
        
        run = load_run(run_path, name, color)
        if not run.samples:
            print(f"Warning: No samples found in {run_path}")
            continue
        
        metrics = compute_metrics(run, prompts, k)
        runs_data.append((run, metrics))
        print(f"Loaded {len(run.samples)} samples from {run_path}")
    
    if not runs_data:
        print("Error: No valid runs found")
        return 1
    
    # Get output config
    output_cfg = config.get("output", {})
    assets_dir_name = output_cfg.get("assets_dir", "assets")
    tables_dir_name = output_cfg.get("tables_dir", "tables")
    
    # Create output directories
    args.out.mkdir(parents=True, exist_ok=True)
    assets_dir = args.out / assets_dir_name
    assets_dir.mkdir(exist_ok=True)
    
    # Generate plots based on config
    figures = {}
    viz_config = config.get("visualizations", {})
    
    if HAS_MATPLOTLIB and not args.no_plots and viz_config:
        print("Generating visualizations...")
        
        plot_data = [(run.name, metrics, run.color) for run, metrics in runs_data]
        sample_data = [(run.name, run.samples, run.color) for run, _ in runs_data]
        
        # Generate each visualization defined in config
        for viz_name, viz_cfg in viz_config.items():
            viz_type = viz_cfg.get("type", "bar")
            filename = viz_cfg.get("filename", f"{viz_name}.png")
            title = viz_cfg.get("title", viz_name.replace("_", " ").title())
            fig_path = assets_dir / filename
            
            if viz_name == "pass_rate_comparison":
                plot_pass_rate_comparison(plot_data, fig_path, title=title)
                figures[viz_name] = f"{assets_dir_name}/{filename}"
            
            elif viz_name == "pass_rate_by_relation":
                plot_pass_rate_by_relation(plot_data, fig_path, title=title)
                figures[viz_name] = f"{assets_dir_name}/{filename}"
            
            elif viz_name == "confidence_distribution":
                bins = viz_cfg.get("bins", 20)
                plot_confidence_distribution(sample_data, fig_path, title=title, bins=bins)
                figures[viz_name] = f"{assets_dir_name}/{filename}"
            
            elif viz_name == "counterfactual_consistency":
                plot_counterfactual_consistency(plot_data, fig_path, title=title)
                figures[viz_name] = f"{assets_dir_name}/{filename}"
            
            # coverage_accuracy plot can be added later
    
    elif not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plots")
    elif not viz_config:
        print("Warning: No visualizations defined in config")
    
    # Export CSV tables
    print("Exporting CSV tables...")
    export_csv_tables(runs_data, args.out, tables_dir_name)
    
    # Generate HTML report
    print("Generating HTML report...")
    html_path = args.out / output_cfg.get("index_file", "index.html")
    generate_html_report(runs_data, figures, html_path, config, k)
    
    print(f"\nReport generated: {html_path}")
    print(f"Tables exported to: {args.out / tables_dir_name}")
    if figures:
        print(f"Figures saved to: {assets_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())

