# Code Audit Report

This report records the repository audit against `iclr2026_conference.pdf`.

## Paper-to-code coverage

The paper requires code for the following areas:

- Adam-LDN, Adam-S, and Adam-S-LDN optimizer variants.
- GPT-2 / WikiText-103 language modeling experiments.
- CIFAR-100-LT experiments.
- Heavy-tailed ImageNet experiments.
- Gradient orthogonality analysis.
- RMSProp comparison.
- Computational overhead analysis.

The repository contains corresponding code paths for all of these categories. The primary map is in `docs/code_map.md`, and the clean optimizer implementations are in `src/adam_imbalance/optimizers.py`.

Raw datasets, model checkpoints, logs, and generated experiment artifacts are intentionally not committed. They are either too large for a source repository or environment-specific.

## Push status

The local `main` branch was checked against `origin/main` before the audit fixes. The previous code bundle was already pushed. This audit adds fixes and documentation on top of that pushed state.

## Checks performed

- Verified that all paper-required experiment categories have a repository code path.
- Checked for oversized committed files; no file larger than 5 MB was found.
- Compiled every Python source file with `py_compile`.
- Ran `ruff` for syntax-level and undefined-name checks with `F821,E9`.
- Ran the packaged optimizer smoke test.
- Checked key training/evaluation entrypoints with `--help`:
  - `py/imagenet/main.py`
  - `py/adapoly_optimizer/main_sup_base.py`
  - `py/gpt/knn-transformers/run_clm.py`

## Issues fixed

- Added missing runtime dependencies for GPT, ViT, ImageNet-LT LMDB, and evaluation tooling.
- Fixed undefined variables in plotting scripts.
- Fixed undefined variables in ImageNet and adaptive optimizer training scripts.
- Fixed undefined gradient references in optimizer and gradient-analysis utilities.
- Added missing imports for pandas, ImageNet-LT dataset support, and ViT constructors.
- Made the GPT `run_clm.py` telemetry import compatible with newer `transformers` versions.
- Removed an invalid scheduler reference from the GPT evaluation-only script.

## Remaining execution constraints

Full reproduction still requires the real datasets, GPU hardware, and the experiment-specific paths used by the original scripts. Many legacy experiment scripts assume CUDA and original filesystem layouts such as `/home/wangjzh`; those paths must be adjusted or provided through the corresponding script arguments before running full training.

The audit verifies that the repository is pushed, structurally complete for the paper, free of the checked undefined-name/syntax errors, and that representative entrypoints start correctly. It does not rerun full paper-scale training because that requires datasets, checkpoints, and long GPU jobs.
