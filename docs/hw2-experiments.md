# HW2 experiments

All runs: git SHA in each `metrics.json`, PyTorch 2.14.1+cu130, CPU (8 cores, UVA `standard` partition),
dataset cache `modelnet10_1024pts_seed0_v1.npz` (train 3,193 / val 798 / test 908), 60 epochs, batch 32, Adam,
weight decay 1e-4, cosine schedule, dropout 0.3 (Mini-PointNet), hidden 64, ReLU, training-only augmentation.

## hw2-e01 — learning-rate selection on validation only (`--skip-test`, seed 0)

| Model | Learning rate | Best val accuracy | Selected epoch |
| --- | ---: | ---: | ---: |
| shallow | 0.0005 | 76.69% | 57 |
| shallow | 0.001 | 82.96% | 48 |
| shallow | 0.003 | 87.97% | 44 |
| shallow | 0.01 | 90.48% | 49 |
| shallow | 0.03 **(chosen)** | 90.60% | 59 |
| mini_pointnet | 0.0005 | 97.49% | 42 |
| mini_pointnet | 0.001 **(chosen)** | 97.74% | 45 |
| mini_pointnet | 0.003 | 97.62% | 49 |

No test metric exists for these runs; the test split was not evaluated.

## hw2-e02 — final protocol, seeds 0–2, test split scored once per run

| Model | Seed | Selected epoch | Best val acc | Test accuracy | Test error |
| --- | ---: | ---: | ---: | ---: | ---: |
| shallow | 0 | 59 | 90.60% | 83.81% | 16.19% |
| shallow | 1 | 52 | 90.60% | 84.03% | 15.97% |
| shallow | 2 | 57 | 90.85% | 83.70% | 16.30% |
| mini_pointnet | 0 | 45 | 97.74% | 93.61% | 6.39% |
| mini_pointnet | 1 | 40 | 97.74% | 92.29% | 7.71% |
| mini_pointnet | 2 | 49 | 97.62% | 92.40% | 7.60% |

Mean ± std test accuracy: shallow 83.85% ± 0.14% (83.8%, 84.0%, 83.7%); Mini-PointNet 92.77% ± 0.60% (93.6%, 92.3%, 92.4%).
Seed 0 of each model is bundled in `backend/models/modelnet10/`.
## Notes on the protocol

- The shallow baseline's chosen rate, 0.03, is the largest one tried; the
  validation gain from 0.01 to 0.03 was only 0.12 points (90.48% → 90.60%), so the
  baseline is close to its plateau and not handicapped by an under-tuned rate.
- The final seed-0 runs reproduce the sweep's validation curves exactly
  (same selected epoch and validation accuracy), confirming CPU determinism.
- A CUDA smoke run (job 21252496, NVIDIA RTX A6000, driver 595.71.05) trained
  Mini-PointNet for 3 epochs at 0.4 s/epoch with `--device auto`; it verifies the
  GPU path only and is not part of the reported results.

## Exactly how the numbers were produced (UVA HPC)

| Item | Value |
| --- | --- |
| Code | Public submission repository (`main`); exact run metadata is bundled with the released metrics |
| Environment | Python 3.12.9 venv, torch 2.14.1+cu130, numpy 2.5.4, matplotlib 3.11.2 (pins in `backend/requirements-dev.txt`) |
| Dataset preparation | Slurm job 21251186, `standard` partition, 16 CPUs, 39 s; archive SHA-256 `9d867943…07d51` |
| hw2-e01 sweep | job arrays 21252014 (0–5) and 21252500 (0–1), 8 CPUs each, `--skip-test` |
| hw2-e02 final | job array 21253296 (0–5): shallow lr 0.03, Mini-PointNet lr 0.001, seeds 0–2 |
| Bundling | `python -m app.classification.bundle --baseline …/shallow_seed0 --final …/mini_pointnet_seed0 --baseline-seeds … --final-seeds …` |

Per run, from `backend/`:

```bash
export MODELNET10_DATA_DIR=/path/to/modelnet10
python -m app.classification.dataset prepare
# hw2-e01, for each model and rate
python -m app.classification.training train --model mini_pointnet --learning-rate 0.001 --epochs 60 --seed 0 --skip-test --out data/experiments/e01/mini_pointnet_lr0.001
# hw2-e02, for each model and seed
python -m app.classification.training train --model shallow --learning-rate 0.03 --epochs 60 --seed 0 --out data/experiments/e02/shallow_seed0
python -m app.classification.training train --model mini_pointnet --learning-rate 0.001 --epochs 60 --seed 0 --out data/experiments/e02/mini_pointnet_seed0
```

Generated run directories and scheduler logs are intentionally not tracked. Released checkpoints, metrics, histories, and plots are stored under `backend/models/modelnet10/`.

## Differences from the original PointNet

No input or feature transform networks (T-Nets), a smaller shared MLP
(64-128-256 instead of 64-128-1024), a two-layer head (256→128→10 instead of
512→256→k), 1,024 points, Adam with cosine decay for 60 epochs instead of step
decay for 250, and no rotation augmentation (ModelNet10 is aligned). These keep
the network small and the comparison with the shallow baseline easy to read.
