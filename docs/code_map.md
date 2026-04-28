# Code Map and Paper Match

This repository supports the ICLR 2026 submission **"Adam Can Mitigate Class Imbalance Without Element-Wise Gradient Normalization"**.

## Paper-facing implementation

| Paper item | Clean implementation | Legacy experiment source |
| --- | --- | --- |
| Algorithm 3, Adam-LDN | `src/adam_imbalance/optimizers.py::AdamLDN` | `py/gpt/knn-transformers/adam_sldn.py`, `py/adapoly_optimizer/adassd_gamma.py`, `py/imagenet/adassd_gamma.py` |
| Algorithm 4, Adam-S | `src/adam_imbalance/optimizers.py::AdamS` | `py/adapoly_optimizer/adassd_gamma.py`, `py/imagenet/adassd_gamma.py` |
| Algorithm 5, Adam-S-LDN | `src/adam_imbalance/optimizers.py::AdamSLDN` | `py/gpt/knn-transformers/adam_sldn.py`, `py/adapoly_optimizer/adassd_gamma.py` |
| GPT-2 on WikiText-103 | `py/gpt/knn-transformers/run_clm.py` and related files | `py/gpt/knn-transformers/` |
| CIFAR-100-LT visual experiments | `py/adapoly_optimizer/main_sup_base.py`, `py/adapoly_optimizer/grid_search_cifar.py` | `py/adapoly_optimizer/` |
| Heavy-tailed ImageNet experiments | `py/imagenet/main.py`, `py/imagenet/main_10class.py`, `py/imagenet/grid_search_imagenet.py` | `py/imagenet/` |
| Gradient orthogonality analysis | `py/adapoly_optimizer/grad_utils.py`, `py/adapoly_optimizer/layer_grad_utils.py`, `py/gpt2/cal_grad.py`, `py/gpt2/cal_layer_grad.py` | `py/adapoly_optimizer/`, `py/gpt2/` |
| Figures and logs processing | top-level `draw_*.py`, `py/gpt/knn-transformers/draw.py`, `py/gpt2/draw_gpt.py` | top-level scripts and `py/*/draw*.py` |

## Correctness notes

- The clean optimizer classes are CPU/GPU agnostic and pass a local PyTorch smoke test.
- The legacy scripts were copied without datasets, model checkpoints, generated outputs, or large logs. They are intended as experiment entry points, not a packaged library.
- Several legacy scripts contain absolute paths from the original training machine. Before rerunning full experiments, set dataset and output paths for your environment.

## Readability notes

- Use `src/adam_imbalance/optimizers.py` as the readable reference implementation for optimizer behavior.
- Use `py/` for full experiment reproduction because it preserves the original script organization.
- Use this file and the main README to map paper figures and appendix settings to code.
