#!/usr/bin/env python3
"""
Stratified sampling for human audit.

Selects images for human annotation using stratified sampling across:
- Relation type (LR vs UD)
- Generation method (prompt-only, controlnet, gligen)
- Confidence bins (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)

Also generates a web-based annotation interface (audit_interface.html).

Usage:
    python -m spatialbench_uc.audit.sample \
        --runs runs/smoke_promptonly runs/smoke_controlnet runs/smoke_gligen \
        --n 200 \
        --out audits/v1/sample.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AuditSample:
    """Single sample selected for human audit."""
    sample_id: str
    image_path: str  # Relative to audit root
    overlay_path: str | None  # Relative to audit root
    prompt: str
    relation: str
    object_a: str
    object_b: str
    checker_verdict_raw: str
    conf: float
    run_id: str
    method: str  # promptonly, controlnet, gligen, etc.


def extract_method_from_run(run_path: Path) -> str:
    """Extract method name from run directory name.
    
    Examples:
        smoke_promptonly -> promptonly
        smoke_controlnet -> controlnet
        smoke_gligen -> gligen
        sd15_promptonly -> promptonly
    """
    name = run_path.name.lower()
    
    # Remove common prefixes
    for prefix in ["smoke_", "sd15_", "exp_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    
    # Remove suffixes like _0.5_conditionning_scale
    if "_" in name:
        parts = name.split("_")
        # Take first meaningful part
        name = parts[0]
    
    return name


def classify_relation(relation: str) -> str:
    """Classify relation into LR (left/right) or UD (up/down)."""
    if relation in ("left_of", "right_of"):
        return "LR"
    elif relation in ("above", "below"):
        return "UD"
    else:
        return "UNKNOWN"


def get_confidence_bin(conf: float) -> str:
    """Get confidence bin label."""
    if conf < 0.2:
        return "0.0-0.2"
    elif conf < 0.4:
        return "0.2-0.4"
    elif conf < 0.6:
        return "0.4-0.6"
    elif conf < 0.8:
        return "0.6-0.8"
    else:
        return "0.8-1.0"


def load_samples_from_run(run_path: Path) -> list[dict]:
    """Load per_sample.jsonl from a run directory."""
    eval_dir = run_path / "eval"
    per_sample_file = eval_dir / "per_sample.jsonl"
    
    if not per_sample_file.exists():
        return []
    
    samples = []
    with open(per_sample_file) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    
    return samples


def load_prompt_text(prompt_id: str, prompts_path: Path | None) -> str:
    """Load prompt text from prompts.jsonl if available."""
    if prompts_path is None or not prompts_path.exists():
        return f"[Prompt {prompt_id}]"
    
    with open(prompts_path) as f:
        for line in f:
            if line.strip():
                prompt_data = json.loads(line)
                if prompt_data.get("prompt_id") == prompt_id:
                    return prompt_data.get("prompt", f"[Prompt {prompt_id}]")
    
    return f"[Prompt {prompt_id}]"


def stratified_sample(
    all_samples: list[dict],
    run_paths: list[Path],
    target_n: int,
    prompts_path: Path | None = None,
) -> list[AuditSample]:
    """Perform stratified sampling across relation, method, and confidence bins."""
    
    # Build stratification key: (relation_class, method, conf_bin) -> list of samples
    strata: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    
    for run_path, samples in zip(run_paths, all_samples):
        method = extract_method_from_run(run_path)
        run_id = run_path.name
        
        for sample in samples:
            relation = sample.get("relation", "unknown")
            relation_class = classify_relation(relation)
            conf = sample.get("conf", 0.0)
            conf_bin = get_confidence_bin(conf)
            
            key = (relation_class, method, conf_bin)
            # Store with run info
            sample_with_meta = {
                **sample,
                "_run_path": run_path,
                "_run_id": run_id,
                "_method": method,
            }
            strata[key].append(sample_with_meta)
    
    # Compute target per stratum (proportional allocation)
    total_available = sum(len(samples) for samples in strata.values())
    if total_available == 0:
        return []
    
    # Allocate samples proportionally, with minimum 1 per stratum
    stratum_targets = {}
    allocated = 0
    
    for key, samples in strata.items():
        if len(samples) > 0:
            # Proportional allocation
            target = max(1, int(len(samples) * target_n / total_available))
            # Don't exceed available
            target = min(target, len(samples))
            stratum_targets[key] = target
            allocated += target
    
    # Adjust if we're under target
    if allocated < target_n:
        # Sort strata by size (descending) and add more
        sorted_strata = sorted(
            strata.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        for key, samples in sorted_strata:
            current = stratum_targets.get(key, 0)
            if current < len(samples) and allocated < target_n:
                add = min(len(samples) - current, target_n - allocated)
                stratum_targets[key] = current + add
                allocated += add
    
    # Sample from each stratum
    audit_samples = []
    for key, samples in strata.items():
        target = stratum_targets.get(key, 0)
        if target == 0:
            continue
        
        # Random sample without replacement
        selected = random.sample(samples, min(target, len(samples)))
        
        for sample in selected:
            run_path = sample["_run_path"]
            run_id = sample["_run_id"]
            method = sample["_method"]
            
            # Get prompt text
            prompt_id = sample.get("prompt_id", "")
            prompt_text = load_prompt_text(prompt_id, prompts_path)
            
            # Build paths relative to audit root
            # Audit root will be the parent of sample.csv
            image_path_rel = f"../../{run_path}/images/{sample['sample_id']}.png"
            overlay_path_rel = None
            if sample.get("overlay_path"):
                overlay_path_rel = f"../../{run_path}/eval/{sample['overlay_path']}"
            
            audit_sample = AuditSample(
                sample_id=sample["sample_id"],
                image_path=image_path_rel,
                overlay_path=overlay_path_rel,
                prompt=prompt_text,
                relation=sample.get("relation", "unknown"),
                object_a=sample.get("object_a", ""),
                object_b=sample.get("object_b", ""),
                checker_verdict_raw=sample.get("verdict_raw", "UNDECIDABLE"),
                conf=sample.get("conf", 0.0),
                run_id=run_id,
                method=method,
            )
            audit_samples.append(audit_sample)
    
    return audit_samples


def write_csv(samples: list[AuditSample], csv_path: Path):
    """Write samples to CSV file."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "image_path",
            "overlay_path",
            "prompt",
            "relation",
            "object_a",
            "object_b",
            "checker_verdict_raw",
            "conf",
            "run_id",
            "method",
        ])
        
        for sample in samples:
            writer.writerow([
                sample.sample_id,
                sample.image_path,
                sample.overlay_path or "",
                sample.prompt,
                sample.relation,
                sample.object_a,
                sample.object_b,
                sample.checker_verdict_raw,
                f"{sample.conf:.4f}",
                sample.run_id,
                sample.method,
            ])


def generate_html_interface(samples: list[AuditSample], html_path: Path):
    """Generate web-based annotation interface."""
    
    # Embed sample data as JSON
    samples_json = json.dumps([
        {
            "sample_id": s.sample_id,
            "image_path": s.image_path,
            "overlay_path": s.overlay_path,
            "prompt": s.prompt,
            "relation": s.relation,
            "object_a": s.object_a,
            "object_b": s.object_b,
            "checker_verdict_raw": s.checker_verdict_raw,
            "conf": s.conf,
            "run_id": s.run_id,
            "method": s.method,
        }
        for s in samples
    ], indent=2)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SpatialBench-UC Human Audit</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Charter', 'Georgia', serif;
            background: #fafafa;
            color: #1a1a2e;
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #ddd;
        }}
        
        h1 {{
            font-size: 2em;
            margin-bottom: 10px;
        }}
        
        .progress {{
            margin-top: 15px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .progress-bar {{
            flex: 1;
            height: 25px;
            background: #e0e0e0;
            border-radius: 5px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            background: #2563eb;
            transition: width 0.3s;
        }}
        
        .main-content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .image-container {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            text-align: center;
        }}
        
        .image-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ccc;
        }}
        
        .metadata {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 20px;
        }}
        
        .metadata h3 {{
            margin-bottom: 15px;
            color: #2563eb;
        }}
        
        .metadata-item {{
            margin-bottom: 10px;
        }}
        
        .metadata-item strong {{
            display: inline-block;
            width: 120px;
        }}
        
        .verdict-buttons {{
            display: flex;
            gap: 15px;
            margin-top: 30px;
            flex-wrap: wrap;
        }}
        
        .verdict-btn {{
            flex: 1;
            min-width: 120px;
            padding: 15px 25px;
            font-size: 1.1em;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .verdict-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        .verdict-btn.pass {{
            background: #16a34a;
            color: white;
        }}
        
        .verdict-btn.fail {{
            background: #dc2626;
            color: white;
        }}
        
        .verdict-btn.undecidable {{
            background: #6b7280;
            color: white;
        }}
        
        .verdict-btn.selected {{
            border: 3px solid #1a1a2e;
            box-shadow: 0 0 0 2px rgba(26, 26, 46, 0.2);
        }}
        
        .navigation {{
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }}
        
        .nav-btn {{
            padding: 10px 20px;
            font-size: 1em;
            border: 1px solid #ddd;
            background: white;
            border-radius: 5px;
            cursor: pointer;
        }}
        
        .nav-btn:hover {{
            background: #f0f0f0;
        }}
        
        .notes {{
            margin-top: 20px;
        }}
        
        .notes textarea {{
            width: 100%;
            min-height: 100px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-family: inherit;
            font-size: 0.95em;
        }}
        
        .export-section {{
            margin-top: 30px;
            padding: 20px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        
        .export-btn {{
            padding: 12px 24px;
            font-size: 1em;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }}
        
        .export-btn:hover {{
            background: #1d4ed8;
        }}
        
        .keyboard-hint {{
            font-size: 0.85em;
            color: #666;
            margin-top: 5px;
        }}
        
        @media (max-width: 768px) {{
            .main-content {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>SpatialBench-UC Human Audit</h1>
            <div class="progress">
                <span id="progress-text">Sample 0/0</span>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
                </div>
            </div>
        </header>
        
        <div class="main-content">
            <div class="image-container">
                <h3>Original Image</h3>
                <img id="image" src="" alt="Generated image">
            </div>
            <div class="image-container">
                <h3>Overlay (with detections)</h3>
                <img id="overlay" src="" alt="Overlay image">
            </div>
        </div>
        
        <div class="metadata">
            <h3>Sample Information</h3>
            <div class="metadata-item">
                <strong>Prompt:</strong>
                <span id="prompt-text"></span>
            </div>
            <div class="metadata-item">
                <strong>Relation:</strong>
                <span id="relation-text"></span>
            </div>
            <div class="metadata-item">
                <strong>Objects:</strong>
                <span id="objects-text"></span>
            </div>
            <div class="metadata-item">
                <strong>Checker Verdict:</strong>
                <span id="checker-verdict"></span>
            </div>
            <div class="metadata-item">
                <strong>Confidence:</strong>
                <span id="confidence-text"></span>
            </div>
            <div class="metadata-item">
                <strong>Method:</strong>
                <span id="method-text"></span>
            </div>
            
            <div class="verdict-buttons">
                <button class="verdict-btn pass" id="btn-pass" onclick="setVerdict('PASS')">
                    PASS
                    <div class="keyboard-hint">(Press P)</div>
                </button>
                <button class="verdict-btn fail" id="btn-fail" onclick="setVerdict('FAIL')">
                    FAIL
                    <div class="keyboard-hint">(Press F)</div>
                </button>
                <button class="verdict-btn undecidable" id="btn-undecidable" onclick="setVerdict('UNDECIDABLE')">
                    UNDECIDABLE
                    <div class="keyboard-hint">(Press U)</div>
                </button>
            </div>
            
            <div class="notes">
                <label for="notes-textarea"><strong>Notes (optional):</strong></label>
                <textarea id="notes-textarea" placeholder="Add any comments about this sample..."></textarea>
            </div>
            
            <div class="navigation">
                <button class="nav-btn" onclick="previousSample()">← Previous</button>
                <button class="nav-btn" onclick="nextSample()">Next →</button>
                <button class="nav-btn" onclick="jumpToSample()">Jump to sample ID...</button>
            </div>
        </div>
        
        <div class="export-section">
            <h3>Export Labels</h3>
            <p>Labels are auto-saved to browser localStorage. Export when complete:</p>
            <button class="export-btn" onclick="exportLabels()">Export to labels_filled.json</button>
            <button class="export-btn" onclick="exportCSV()">Export to labels_filled.csv</button>
        </div>
    </div>
    
    <script>
        // Sample data embedded from Python
        const samples = {samples_json};
        
        // Load saved labels from localStorage
        let labels = JSON.parse(localStorage.getItem('audit_labels') || '{{}}');
        
        let currentIndex = parseInt(localStorage.getItem('audit_current_index') || '0');
        
        function updateDisplay() {{
            if (currentIndex < 0 || currentIndex >= samples.length) {{
                return;
            }}
            
            const sample = samples[currentIndex];
            
            // Update images
            document.getElementById('image').src = sample.image_path;
            if (sample.overlay_path) {{
                document.getElementById('overlay').src = sample.overlay_path;
            }} else {{
                document.getElementById('overlay').src = sample.image_path;
            }}
            
            // Update metadata
            document.getElementById('prompt-text').textContent = sample.prompt;
            document.getElementById('relation-text').textContent = sample.relation;
            document.getElementById('objects-text').textContent = `${{sample.object_a}} / ${{sample.object_b}}`;
            document.getElementById('checker-verdict').textContent = sample.checker_verdict_raw;
            document.getElementById('confidence-text').textContent = sample.conf.toFixed(3);
            document.getElementById('method-text').textContent = sample.method;
            
            // Update progress
            const progress = ((currentIndex + 1) / samples.length) * 100;
            document.getElementById('progress-fill').style.width = progress + '%';
            document.getElementById('progress-text').textContent = `Sample ${{currentIndex + 1}}/${{samples.length}}`;
            
            // Update verdict buttons
            const savedLabel = labels[sample.sample_id];
            document.querySelectorAll('.verdict-btn').forEach(btn => {{
                btn.classList.remove('selected');
            }});
            
            if (savedLabel) {{
                const btnId = 'btn-' + savedLabel.verdict.toLowerCase();
                const btn = document.getElementById(btnId);
                if (btn) {{
                    btn.classList.add('selected');
                }}
                document.getElementById('notes-textarea').value = savedLabel.notes || '';
            }} else {{
                document.getElementById('notes-textarea').value = '';
            }}
            
            // Save current index
            localStorage.setItem('audit_current_index', currentIndex.toString());
        }}
        
        function setVerdict(verdict) {{
            const sample = samples[currentIndex];
            const notes = document.getElementById('notes-textarea').value;
            
            labels[sample.sample_id] = {{
                verdict: verdict,
                notes: notes,
                timestamp: new Date().toISOString(),
            }};
            
            localStorage.setItem('audit_labels', JSON.stringify(labels));
            
            // Update button selection
            document.querySelectorAll('.verdict-btn').forEach(btn => {{
                btn.classList.remove('selected');
            }});
            document.getElementById('btn-' + verdict.toLowerCase()).classList.add('selected');
        }}
        
        function nextSample() {{
            if (currentIndex < samples.length - 1) {{
                // Save notes before moving
                const sample = samples[currentIndex];
                const notes = document.getElementById('notes-textarea').value;
                if (labels[sample.sample_id]) {{
                    labels[sample.sample_id].notes = notes;
                    localStorage.setItem('audit_labels', JSON.stringify(labels));
                }}
                
                currentIndex++;
                updateDisplay();
            }}
        }}
        
        function previousSample() {{
            if (currentIndex > 0) {{
                // Save notes before moving
                const sample = samples[currentIndex];
                const notes = document.getElementById('notes-textarea').value;
                if (labels[sample.sample_id]) {{
                    labels[sample.sample_id].notes = notes;
                    localStorage.setItem('audit_labels', JSON.stringify(labels));
                }}
                
                currentIndex--;
                updateDisplay();
            }}
        }}
        
        function jumpToSample() {{
            const sampleId = prompt('Enter sample ID:');
            if (!sampleId) return;
            
            const index = samples.findIndex(s => s.sample_id === sampleId);
            if (index >= 0) {{
                currentIndex = index;
                updateDisplay();
            }} else {{
                alert('Sample ID not found');
            }}
        }}
        
        function exportLabels() {{
            const exportData = [];
            for (const sample of samples) {{
                const label = labels[sample.sample_id];
                if (label) {{
                    exportData.push({{
                        sample_id: sample.sample_id,
                        human_verdict: label.verdict,
                        notes: label.notes || '',
                        timestamp: label.timestamp,
                    }});
                }}
            }}
            
            const blob = new Blob([JSON.stringify(exportData, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'labels_filled.json';
            a.click();
            URL.revokeObjectURL(url);
        }}
        
        function exportCSV() {{
            const rows = [['sample_id', 'human_verdict', 'notes', 'timestamp']];
            for (const sample of samples) {{
                const label = labels[sample.sample_id];
                if (label) {{
                    rows.push([
                        sample.sample_id,
                        label.verdict,
                        (label.notes || '').replace(/"/g, '""'),
                        label.timestamp,
                    ]);
                }}
            }}
            
            const csv = rows.map(row => 
                row.map(cell => `"${{cell}}"`).join(',')
            ).join('\\n');
            
            const blob = new Blob([csv], {{type: 'text/csv'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'labels_filled.csv';
            a.click();
            URL.revokeObjectURL(url);
        }}
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.target.tagName === 'TEXTAREA') return;
            
            if (e.key === 'p' || e.key === 'P') {{
                e.preventDefault();
                setVerdict('PASS');
            }} else if (e.key === 'f' || e.key === 'F') {{
                e.preventDefault();
                setVerdict('FAIL');
            }} else if (e.key === 'u' || e.key === 'U') {{
                e.preventDefault();
                setVerdict('UNDECIDABLE');
            }} else if (e.key === 'ArrowLeft') {{
                e.preventDefault();
                previousSample();
            }} else if (e.key === 'ArrowRight') {{
                e.preventDefault();
                nextSample();
            }}
        }});
        
        // Initialize
        updateDisplay();
    </script>
</body>
</html>
"""
    
    with open(html_path, "w") as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser(
        description="Stratified sampling for human audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help="Run directories to sample from",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Target number of samples (default: 200)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV file path (will also generate HTML in same directory)",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Path to prompts.jsonl (for prompt text)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load samples from all runs
    all_samples_by_run = []
    for run_path in args.runs:
        if not run_path.exists():
            print(f"Warning: Run path does not exist: {run_path}")
            continue
        
        samples = load_samples_from_run(run_path)
        if not samples:
            print(f"Warning: No samples found in {run_path}")
            continue
        
        all_samples_by_run.append(samples)
        print(f"Loaded {len(samples)} samples from {run_path}")
    
    if not all_samples_by_run:
        print("Error: No samples found in any run")
        return 1
    
    # Perform stratified sampling
    print(f"\nPerforming stratified sampling (target: {args.n} samples)...")
    audit_samples = stratified_sample(
        all_samples_by_run,
        args.runs,
        args.n,
        prompts_path=args.prompts,
    )
    
    print(f"Selected {len(audit_samples)} samples for audit")
    
    # Create output directory
    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    # Write CSV
    write_csv(audit_samples, args.out)
    print(f"Wrote CSV to: {args.out}")
    
    # Generate HTML interface
    html_path = args.out.parent / "audit_interface.html"
    generate_html_interface(audit_samples, html_path)
    print(f"Generated HTML interface: {html_path}")
    
    # Print stratification summary
    print("\nStratification Summary:")
    summary = defaultdict(int)
    for sample in audit_samples:
        rel_class = classify_relation(sample.relation)
        conf_bin = get_confidence_bin(sample.conf)
        key = f"{rel_class}/{sample.method}/{conf_bin}"
        summary[key] += 1
    
    for key, count in sorted(summary.items()):
        print(f"  {key}: {count}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

