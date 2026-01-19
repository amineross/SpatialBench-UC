# Frozen Run Bundle: final_20260112_084335

This directory contains the complete evaluation outputs for the SpatialBench-UC paper. All results can be reproduced from these artifacts **without regenerating images**.

## Contents

### Per-method outputs

Each method directory (`sd15_promptonly/`, `sd15_boxdiff/`, `sd14_gligen/`) contains:

| File | Description |
|------|-------------|
| `manifest.jsonl` | Prompt-to-image mapping (image paths excluded from git) |
| `gen_config.yaml` | Generation configuration used |
| `eval/per_sample.jsonl` | Per-image checker outputs (calibrated) |
| `eval/metrics.json` | Aggregate metrics |
| `eval/checker_config.yaml` | Checker configuration |
| `eval/provenance.json` | Environment and version info |
| `eval_precal_*/` | Uncalibrated checker outputs (before calibration) |

**Note**: `images/` and `eval/overlays/` directories are excluded from git due to size (~2.4k images per method). The per-sample JSONL files contain all information needed to reproduce tables and plots.

### Reports

```
reports/
├── v1_calibrated_20260116_113552/    # Calibrated report
│   ├── tables/*.csv                   # Exported CSV tables
│   ├── assets/*.png                   # Plots
│   ├── index.html                     # HTML report
│   └── report_meta.json               # Provenance
└── v1_finalfix_20260114_143137/      # Pre-calibration report
```

### Human audit

```
audits/v1/
├── sample.csv              # 200 stratified samples for audit
├── labels_filled.json      # Human verdicts (PASS/FAIL/UNDECIDABLE)
├── labels_filled.csv       # Same, in CSV format
├── analysis_calibrated/    # Metrics computed on calibrated outputs
├── analysis_uncalibrated/  # Metrics computed on uncalibrated outputs
└── audit_interface.html    # HTML interface used for labeling
```

### Data snapshot

```
data/
├── prompts/v1.0.1/         # Versioned prompts (with SHA256)
└── objects/                # COCO vocabulary subset
```

## Reproducing paper tables

From the repository root:

```bash
# Regenerate report
python -m spatialbench_uc.report \
  --config configs/report_v1.yaml \
  --eval-subdir eval

# Recompute audit metrics
python scripts/recompute_audit_metrics_from_eval_jsonl.py \
  --sample runs/final_20260112_084335/audits/v1/sample.csv \
  --labels runs/final_20260112_084335/audits/v1/labels_filled.json \
  --out runs/final_20260112_084335/audits/v1/analysis_calibrated \
  --eval-subdir eval \
  --tau-sweep unique
```

## Methods evaluated

| Method | Model | Description |
|--------|-------|-------------|
| `sd15_promptonly` | Stable Diffusion 1.5 | Text-only generation |
| `sd15_boxdiff` | Stable Diffusion 1.5 + BoxDiff | Box-constrained attention |
| `sd14_gligen` | GLIGEN (SD 1.4 base) | Grounded generation |

All methods: 200 prompts x 4 seeds = 800 images per method.
