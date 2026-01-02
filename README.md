# SpatialBench-UC

**Uncertainty-aware spatial relationship benchmark for text-to-image models.**

SpatialBench-UC evaluates whether text-to-image models correctly follow spatial constraints (e.g., "A is to the left of B") with:

- **Uncertainty quantification**: PASS / FAIL / UNDECIDABLE verdicts with calibrated confidence scores
- **Counterfactual consistency**: Tests logically equivalent prompt pairs ("A left of B" ↔ "B right of A")
- **Reproducible evaluation**: Detector-based geometric verification, not learned scorers

## Installation

### Step 1: Create Environment

```bash
# Using venv (recommended)
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Or using conda
conda create -n spatialbench python=3.10 -y
conda activate spatialbench
```

### Step 2: Install PyTorch (Platform-Specific)

Choose ONE based on your system:

**Apple Silicon Mac (M1/M2/M3):**
```bash
pip install -r requirements/torch-mps.txt
```

**Linux with NVIDIA GPU (CUDA 12.1):**
```bash
pip install -r requirements/torch-cuda121.txt
```

**Linux with NVIDIA GPU (CUDA 11.8):**
```bash
pip install -r requirements/torch-cuda118.txt
```

**CPU only (any platform):**
```bash
pip install -r requirements/torch-cpu.txt
```

### Step 3: Install SpatialBench-UC

```bash
# Install the package in editable mode
pip install -e .

# Or with development tools
pip install -e ".[dev]"
```

### Step 4: Generate COCO Vocabulary (One-Time Setup)

```bash
python scripts/generate_coco_vocab.py
```

This extracts the official COCO class names from torchvision's pretrained weights.

### Step 5: Verify Installation

```bash
python -c "import spatialbench_uc; print('SpatialBench-UC installed successfully!')"
python -c "from spatialbench_uc.detectors.fasterrcnn import get_coco_class_names; print(f'COCO classes: {len(get_coco_class_names())}')"
```

## Quick Start

### 1. Build Benchmark Prompts

```bash
python -m spatialbench_uc.build_prompts --config configs/prompts_v1.yaml
```

### 2. Generate Images

**Baseline 1: Prompt-Only (SD 1.5)**
```bash
python -m spatialbench_uc.generate \
  --config configs/gen_sd15_promptonly.yaml \
  --prompts data/prompts/v1.0.0/prompts.jsonl \
  --out runs/sd15_promptonly
```

**Baseline 2: ControlNet (SD 1.5 + Canny)**
```bash
python -m spatialbench_uc.generate \
  --config configs/gen_sd15_controlnet.yaml \
  --prompts data/prompts/v1.0.0/prompts.jsonl \
  --out runs/sd15_controlnet
```

### 3. Evaluate

```bash
python -m spatialbench_uc.evaluate \
  --manifest runs/sd15_promptonly/manifest.jsonl \
  --config configs/checker_v1.yaml \
  --out runs/sd15_promptonly/eval
```

### 4. Generate Report

```bash
python -m spatialbench_uc.report \
  --runs runs/sd15_promptonly runs/sd15_controlnet \
  --out reports/v1
```

## Project Structure

```
spatialbench-uc/
├── configs/                    # Configuration files
│   ├── prompts_v1.yaml        # Prompt generation config
│   ├── gen_sd15_promptonly.yaml
│   ├── gen_sd15_controlnet.yaml
│   └── checker_v1.yaml
├── data/
│   ├── objects/               # COCO vocabulary
│   │   ├── coco_classes.json  # Auto-generated from torchvision
│   │   └── coco_subset_v1.json
│   └── prompts/               # Generated prompts
├── src/spatialbench_uc/       # Main package
│   ├── generators/            # Text-to-image generators
│   ├── detectors/             # Object detectors
│   └── utils/                 # Utilities
├── scripts/                   # Utility scripts
├── runs/                      # Generated images and evaluations
└── reports/                   # HTML reports
```

## Extending SpatialBench-UC

### Adding a New Generator

```python
from spatialbench_uc.generators import BaseGenerator, register_generator

@register_generator("my_model")
class MyGenerator(BaseGenerator):
    def __init__(self, config):
        # Load your model
        pass
    
    def generate(self, prompt: str, seed: int, control_image=None):
        # Generate and return PIL.Image
        pass
```

Then create a config file:

```yaml
generator:
  type: my_model
  model_id: your-model-id
  params:
    height: 512
    width: 512
```

### Adding a New Detector

```python
from spatialbench_uc.detectors import BaseDetector, Detection, register_detector

@register_detector("my_detector")
class MyDetector(BaseDetector):
    def __init__(self, config):
        # Load your model
        pass
    
    def detect(self, image, labels: list[str]) -> list[Detection]:
        # Return list of Detection(box_xyxy, score, label)
        pass
```

## v1 Scope

| Baseline | Model | Method | Research Question |
|----------|-------|--------|-------------------|
| 1 | SD 1.5 | Prompt-Only | Can text alone convey spatial relations? |
| 2 | SD 1.5 | + ControlNet Canny | Does structural guidance help? |

## License

MIT License

## Citation

If you use SpatialBench-UC in your research, please cite:

```bibtex
@software{spatialbench_uc,
  title = {SpatialBench-UC: Uncertainty-aware Spatial Relationship Benchmark},
  year = {2026},
  url = {https://github.com/yourusername/spatialbench-uc}
}
```

