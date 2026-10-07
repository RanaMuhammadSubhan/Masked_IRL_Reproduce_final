# Masked IRL reproduction

This repository contains my reproduction of **Masked IRL: LLM-Guided Reward Disambiguation from Demonstrations and Language** and a small follow-up study on one limitation I found during the reproduction.

The short version is: the public code was usable, but the simulation trajectories and environment assets used for the paper's main experiment were not included. I therefore built a fixed replacement benchmark, kept the original code path separate, and then made a second implementation that follows the recoverable paper settings more closely.

## Main result

The core comparison uses three prespecified seeds (`12345`, `23456`, `34567`). Values are mean +/- SE of pairwise trajectory-ordering accuracy.

| Method        |  Before adaptation |   Few-shot adapted |   Strict zero-shot |
| ------------- | -----------------: | -----------------: | -----------------: |
| LC-RL         |     62.93 +/- 1.44 |     65.07 +/- 0.81 |     62.09 +/- 1.09 |
| Explicit Mask |     67.66 +/- 3.15 | **70.38 +/- 1.48** |     65.44 +/- 2.21 |
| Masked IRL    | **67.97 +/- 2.02** |     69.41 +/- 1.02 | **66.26 +/- 1.70** |

Explicit Mask has the best adapted mean, while Masked IRL has the best strict zero-shot mean. I do not treat this as evidence of a universal winner; the ranking depends on the evaluation endpoint and the study uses one fixed replacement dataset.

## What I changed to follow the paper more closely

Compared with the released behavior, the V3 implementation uses:

- a frozen `t5-base` encoder;
- Uniform(0,1) perturbations for the masking objective;
- the local state/coordinate masking loss described in the manuscript;
- `lambda = 10` for Masked IRL;
- 1000 pretraining epochs + 100 adaptation epochs;
- validation on training preference identities only;
- a separate strict zero-shot set that never supplies demonstrations, gradients, or checkpoint-selection scores.

I intentionally retained the released cross-sample importance estimator because the paper does not specify that part clearly enough to justify silently replacing it.

## Limitation I investigated

The most important practical issue I found was **mask reliability**. When relevance masks are wrong, Masked IRL can learn the wrong invariances.

I tested this by flipping mask bits at increasing rates and running the downstream reward-learning experiment, rather than only measuring mask F1. Under 30% corruption, adapted accuracy was:

- Explicit Mask: **58.74 +/- 1.71%**
- Original Masked IRL: **51.18 +/- 0.75%**

So the original Masked IRL was not automatically more robust to mask errors.

## Exploratory improvement

I tested an uncertainty-aware version of the masking loss. A binary "irrelevant" decision is softened when the available demonstrations show that the coordinate has a noticeably different state distribution. This reduces the strength of a potentially incorrect invariance constraint without changing the clean baseline protocol.

At 30% mask corruption, the adapted result was:

- Original Masked IRL: **51.18 +/- 0.75%**
- Explicit Mask: **58.74 +/- 1.71%**
- Uncertainty-aware variant: **65.47 +/- 0.42%**

The paired gain over original Masked IRL is **+14.29 percentage points** at this corruption level.

This result is **exploratory**. The candidate was designed after the baseline stress test, performs slightly worse in the clean condition, and may benefit partly from lower total regularization. A fresh benchmark and mass-matched controls are the main next steps.

## What is in the repository

- `report/` - short professor report and supporting documentation
- `figures/` - main comparison, adaptation, corruption, and exploratory-method figures
- `code/upstream/` - released model/utils used by the reproduction
- `code/reproduction/` - V3 training and evaluation adapters
- `code/extensions/` - robustness and uncertainty-aware experiments
- `reconstruction/datasets/simulation_v2/` - fixed replacement benchmark
- `results/` - frozen runs, predictions, checkpoints, and summary tables
- `verification/` - independent checks for metrics, splits, masks, schedules, and hashes
- `evidence/` - experiment registry and implementation decisions
- `environment/` - environment specification

## Quick verification

From the repository root:

```bash
python verification/verify_submission.py --package .
python scripts/summarize_results.py --package . --output work/recomputed_tables
```

For a new training run, use a new output directory so the frozen evidence is not overwritten:

```bash
python scripts/check_environment.py --training --fetch-model
python scripts/run_core_v3.py --output work/core --gpus 0
```

The core scripts also support the corruption and extension experiments:

```bash
python scripts/run_corruption.py --output work/core --gpus 0
python scripts/run_extensions.py --base-root work/core --output work/extensions --gpus 0
```

## Important limitations

- The original simulation trajectories and environment/resource package were unavailable, so the paper's exact numerical simulation result cannot be claimed.
- The study uses one replacement dataset and three training seeds.
- The released cross-sample importance estimator is retained.
- Synthetic independent bit flips are a stress test, not a full model of real LLM mask errors.
- The uncertainty-aware variant has not been confirmed on a fresh benchmark.

## Source

- Paper: **Masked IRL: LLM-Guided Reward Disambiguation from Demonstrations and Language**, arXiv:2511.14565
- Official repository: **MIT-CLEAR-Lab/Masked-IRL**
