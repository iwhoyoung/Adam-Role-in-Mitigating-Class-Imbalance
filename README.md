# From Element-Wise to Cross-Iteration Normalization: Revisiting Adam’s Role in Mitigating Class Imbalance

This repository contains code for the submission **"From Element-Wise to Cross-Iteration Normalization: Revisiting Adam’s Role in Mitigating Class Imbalance"**.

The code is organized in two layers:

- `src/adam_imbalance/`: clean, reusable optimizer implementations that match the paper algorithms.
- `py/`, `data/*.py`, and top-level `draw_*.py`: legacy experiment scripts used for GPT-2/WikiText-103, CIFAR-100-LT, heavy-tailed ImageNet, gradient orthogonality analysis, and figure generation.

Large datasets, checkpoints, generated logs, and plot outputs are intentionally excluded from the repository.

## Optimizers

The paper-facing implementations are:

- `AdamLDN`: Adam-LDN, replacing element-wise second-moment normalization with layer-wise dynamics normalization.
- `AdamS`: Adam with layer-specific scaling based on initial parameter magnitude.
- `AdamSLDN`: Adam-S-LDN, combining layer-specific scaling with layer-wise dynamics normalization.

Example:

```python
from adam_imbalance import AdamLDN, AdamS, AdamSLDN

optimizer = AdamSLDN(model.parameters(), lr=5e-2)
```

Run the smoke test:

```bash
PYTHONPATH=src python examples/optimizer_smoke_test.py
```

For Windows PowerShell:

```powershell
$env:PYTHONPATH='src'; python examples/optimizer_smoke_test.py
```

## Experiment Coverage

The included code covers the paper's main experiment groups:

- GPT-2 on WikiText-103, including Adam, SGD, Adam-LDN, Adam-S, Adam-S-LDN, and RMSProp comparisons.
- CIFAR-100-LT experiments with VGG16-BN, ResNet18, ResNet50, ViT-S, and ViT-B.
- Heavy-tailed ImageNet experiments.
- Gradient orthogonality and mean gradient orthogonality analysis.
- Figure and training-log processing scripts.

See `docs/code_map.md` for a detailed mapping from paper sections and algorithms to code files.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Notes

- Full training was run on 2 V100 GPUs in the paper.
- Some legacy scripts still contain absolute paths from the original experiment machine. Update dataset and output paths before launching full runs.
- The repository keeps source code and reproducibility scripts, not raw datasets or generated checkpoints.
