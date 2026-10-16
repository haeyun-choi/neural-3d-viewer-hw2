"""Training and evaluation for the ModelNet10 classifiers.

    python -m app.classification.training train --model mini_pointnet --out DIR [options]
    python -m app.classification.training evaluate --checkpoint DIR/best.pt
    python -m app.classification.training compare --baseline DIR --final DIR --out FILE

One optimisation step, spelled out in ``train_one_epoch``:

1. forward propagation  ``logits = model(points)``
2. loss                 ``loss = cross_entropy(logits, labels)``
3. backpropagation      ``loss.backward()`` fills ``p.grad = dloss/dp`` for every weight
4. update               ``optimizer.step()`` moves each weight using its gradient (Adam)

Checkpoints are selected on validation accuracy only. The official test split
is evaluated once, with the selected checkpoint, after training has finished.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import dataset
from .geometry import PREPROCESSING
from .models import build_model, parameter_count

CHECKPOINT_FORMAT = "neural3d-modelnet10-classifier"


@dataclass
class TrainConfig:
    model: str = "mini_pointnet"
    epochs: int = 60
    learning_rate: float = 1e-3
    batch_size: int = 32
    hidden_dim: int = 64
    activation: str = "relu"
    dropout: float = 0.3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    augment: bool = True
    seed: int = 0
    device: str = "auto"

    @classmethod
    def from_dict(cls, values: dict) -> "TrainConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)


class TrainingCancelled(Exception):
    pass


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def augment(points: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Training-only augmentation: anisotropic scale, small shift, clipped jitter.

    No rotation: ModelNet10 objects are consistently upright, and so is the test split.
    """
    b = points.shape[0]
    device = points.device
    scale = torch.empty(b, 1, 3).uniform_(0.8, 1.25, generator=generator).to(device)
    shift = torch.empty(b, 1, 3).uniform_(-0.1, 0.1, generator=generator).to(device)
    jitter = (torch.randn(points.shape, generator=generator) * 0.01).clamp_(-0.03, 0.03).to(device)
    return points * scale + shift + jitter


def gradient_norm(model: nn.Module) -> float:
    """L2 norm of all gradients after ``loss.backward()`` — the size of the step direction."""
    squares = [p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def train_one_epoch(model, optimizer, points, labels, config: TrainConfig, generator, should_stop) -> dict:
    model.train()  # BatchNorm uses batch statistics, Dropout is active
    order = torch.randperm(len(labels), generator=generator)
    total_loss, correct, seen, norms = 0.0, 0, 0, []
    for start in range(0, len(order), config.batch_size):
        if should_stop():
            raise TrainingCancelled()
        index = order[start:start + config.batch_size]
        if len(index) < 2:  # BatchNorm needs more than one sample per batch
            continue
        x = points[index.to(points.device)]
        y = labels[index.to(labels.device)]
        if config.augment:
            x = augment(x, generator)
        logits = model(x)                          # 1. forward propagation
        loss = F.cross_entropy(logits, y)          # 2. cross-entropy loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()                            # 3. backpropagation: p.grad = dloss/dp
        norms.append(gradient_norm(model))         #    gradient magnitude, recorded per step
        optimizer.step()                           # 4. optimizer update of every weight
        total_loss += loss.item() * len(index)
        correct += int((logits.argmax(1) == y).sum())
        seen += len(index)
    return {"loss": total_loss / seen, "accuracy": correct / seen, "grad_norm": float(np.mean(norms))}


@torch.no_grad()
def evaluate(model, points, labels, num_classes: int, batch_size: int = 256) -> dict:
    model.eval()  # BatchNorm uses running statistics, Dropout is off
    total_loss, predictions = 0.0, []
    for start in range(0, len(labels), batch_size):
        x, y = points[start:start + batch_size], labels[start:start + batch_size]
        logits = model(x)
        total_loss += float(F.cross_entropy(logits, y, reduction="sum"))
        predictions.append(logits.argmax(1))
    predicted = torch.cat(predictions).cpu().numpy()
    truth = labels.cpu().numpy()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    accuracy = float((predicted == truth).mean())
    return {"loss": total_loss / len(truth), "accuracy": accuracy, "error": 1 - accuracy, "confusion": confusion}


def per_class(confusion: np.ndarray, classes: list[str]) -> list[dict]:
    rows = []
    for i, name in enumerate(classes):
        support = int(confusion[i].sum())
        predicted = int(confusion[:, i].sum())
        hit = int(confusion[i, i])
        rows.append({"class": name, "support": support, "correct": hit,
                     "recall": hit / support if support else 0.0,
                     "precision": hit / predicted if predicted else 0.0})
    return rows


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, timeout=5, check=True).stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance only
        return None


def make_checkpoint(model: nn.Module, config: TrainConfig, classes: list[str], extra: dict) -> dict:
    return {
        "format": CHECKPOINT_FORMAT, "version": 1, "model": config.model,
        "model_config": {"hidden_dim": config.hidden_dim, "activation": config.activation, "dropout": config.dropout},
        "train_config": config.to_dict(), "classes": list(classes), "preprocessing": PREPROCESSING,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, **extra,
    }


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    # weights_only: the file holds tensors and plain values, never pickled code.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    model = build_model(checkpoint["model"], len(checkpoint["classes"]), **checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def train(config: TrainConfig, data: dataset.PreparedData, out_dir: Path | None = None,
          progress: Callable[[dict], None] | None = None, should_stop: Callable[[], bool] = lambda: False,
          phase: Callable[[str], None] | None = None, evaluate_test: bool = True) -> dict:
    """Train with validation-based checkpoint selection, then evaluate the test split once.

    ``evaluate_test=False`` is for hyperparameter selection: the test split is not touched.
    """
    started = time.monotonic()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    device = resolve_device(config.device)
    classes = data.classes

    def tensors(points, labels):
        return torch.from_numpy(points).float().to(device), torch.from_numpy(labels).long().to(device)

    train_x, train_y = tensors(data.train_points, data.train_labels)
    val_x, val_y = tensors(data.val_points, data.val_labels)
    model = build_model(config.model, len(classes), config.hidden_dim, config.activation, config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01)
                 if config.scheduler == "cosine" else None)

    history, best = [], {"epoch": 0, "val_accuracy": -1.0, "val_loss": math.inf, "state": None}
    if phase:
        phase("training")
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.monotonic()
        lr = optimizer.param_groups[0]["lr"]
        train_stats = train_one_epoch(model, optimizer, train_x, train_y, config, generator, should_stop)
        val_stats = evaluate(model, val_x, val_y, len(classes))
        if scheduler:
            scheduler.step()
        row = {"epoch": epoch, "train_loss": train_stats["loss"], "train_accuracy": train_stats["accuracy"],
               "val_loss": val_stats["loss"], "val_accuracy": val_stats["accuracy"],
               "grad_norm": train_stats["grad_norm"], "learning_rate": lr,
               "seconds": time.monotonic() - epoch_started}
        history.append(row)
        if (val_stats["accuracy"], -val_stats["loss"]) > (best["val_accuracy"], -best["val_loss"]):
            best = {"epoch": epoch, "val_accuracy": val_stats["accuracy"], "val_loss": val_stats["loss"],
                    "state": copy.deepcopy(model.state_dict())}
        if progress:
            progress(row)

    model.load_state_dict(best["state"])
    metrics = {
        "model": config.model, "parameters": parameter_count(model), "device": str(device),
        "best_epoch": best["epoch"], "best_val_accuracy": best["val_accuracy"], "best_val_loss": best["val_loss"],
        "test_accuracy": None, "test_error": None, "test_loss": None, "confusion_matrix": None, "per_class": None,
        "classes": classes, "split_sizes": data.summary(),
        "train_seconds": time.monotonic() - started, "torch": torch.__version__, "git_sha": git_sha(),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if evaluate_test:
        if phase:
            phase("evaluating test split")
        test_x, test_y = tensors(data.test_points, data.test_labels)
        test = evaluate(model, test_x, test_y, len(classes))
        metrics.update(test_accuracy=test["accuracy"], test_error=test["error"], test_loss=test["loss"],
                       confusion_matrix=test["confusion"].tolist(), per_class=per_class(test["confusion"], classes))
    if out_dir is not None:
        save_run(out_dir, model, config, classes, history, metrics)
    return {"history": history, "metrics": metrics, "model": model}


def save_run(out_dir: Path, model: nn.Module, config: TrainConfig, classes: list[str], history: list, metrics: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = {key: metrics[key] for key in ("best_epoch", "best_val_accuracy", "test_accuracy", "parameters", "git_sha")}
    torch.save(make_checkpoint(model, config, classes, extra), out_dir / "best.pt")
    (out_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    try:
        plot_run(out_dir, history, metrics)
    except ImportError:
        pass  # matplotlib is a development dependency; the JSON is the record


def plot_run(out_dir: Path, history: list, metrics: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    ax_loss.plot(epochs, [r["train_loss"] for r in history], label="train", color="#2a6fdb")
    ax_loss.plot(epochs, [r["val_loss"] for r in history], label="validation", color="#e07b00")
    ax_loss.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    ax_acc.plot(epochs, [r["train_accuracy"] for r in history], label="train", color="#2a6fdb")
    ax_acc.plot(epochs, [r["val_accuracy"] for r in history], label="validation", color="#e07b00")
    ax_acc.axvline(metrics["best_epoch"], color="#888", linestyle=":", label=f"selected epoch {metrics['best_epoch']}")
    test = f" · test {metrics['test_accuracy']:.3f}" if metrics.get("test_accuracy") is not None else ""
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title=f"Accuracy{test}", ylim=(0, 1))
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle(f"{metrics['model']} on ModelNet10")
    fig.savefig(out_dir / "curves.png", dpi=110)
    plt.close(fig)
    if metrics.get("confusion_matrix") is None:
        return

    confusion = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.6, 5), constrained_layout=True)
    ax.imshow(confusion, cmap="Blues")
    classes = metrics["classes"]
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    for (i, j), value in np.ndenumerate(confusion):
        if value:
            ax.text(j, i, value, ha="center", va="center", fontsize=7,
                    color="white" if value > confusion.max() / 2 else "black")
    ax.set(xlabel="predicted", ylabel="true", title=f"{metrics['model']} · test confusion matrix")
    fig.savefig(out_dir / "confusion.png", dpi=110)
    plt.close(fig)


def compare(baseline_dir: Path, final_dir: Path) -> dict:
    base = json.loads((baseline_dir / "metrics.json").read_text())
    final = json.loads((final_dir / "metrics.json").read_text())
    return {
        "baseline": {k: base[k] for k in ("model", "parameters", "best_epoch", "best_val_accuracy", "test_accuracy", "test_error")},
        "final": {k: final[k] for k in ("model", "parameters", "best_epoch", "best_val_accuracy", "test_accuracy", "test_error")},
        "test_error_reduction": base["test_error"] - final["test_error"],
        "relative_error_reduction": (base["test_error"] - final["test_error"]) / base["test_error"],
        "final_beats_baseline": final["test_error"] < base["test_error"],
        "test_size": final["split_sizes"]["test"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("train")
    defaults = TrainConfig()
    for f in fields(TrainConfig):
        value = getattr(defaults, f.name)
        if isinstance(value, bool):
            run.add_argument(f"--{f.name.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "yes"), default=value)
        else:
            run.add_argument(f"--{f.name.replace('_', '-')}", type=type(value), default=value)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--skip-test", action="store_true", help="hyperparameter selection: do not touch the test split")
    run.add_argument("--data-dir", type=Path, default=None)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--data-dir", type=Path, default=None)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--baseline", type=Path, required=True)
    cmp_.add_argument("--final", type=Path, required=True)
    cmp_.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "compare":
        result = compare(args.baseline, args.final)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0
    data = dataset.load(args.data_dir)
    if args.command == "evaluate":
        model, checkpoint = load_checkpoint(args.checkpoint)
        x = torch.from_numpy(data.test_points).float()
        y = torch.from_numpy(data.test_labels).long()
        result = evaluate(model, x, y, len(data.classes))
        print(json.dumps({"checkpoint": str(args.checkpoint), "model": checkpoint["model"],
                          "test_accuracy": result["accuracy"], "test_error": result["error"],
                          "recorded_test_accuracy": checkpoint.get("test_accuracy")}, indent=2))
        return 0
    config = TrainConfig.from_dict({k: v for k, v in vars(args).items() if k != "skip_test"})
    print(f"config {json.dumps(config.to_dict())} · data {data.summary()} · device {resolve_device(config.device)}")

    def report(row):
        print(f"epoch {row['epoch']:3d}  train loss {row['train_loss']:.4f} acc {row['train_accuracy']:.4f}  "
              f"val loss {row['val_loss']:.4f} acc {row['val_accuracy']:.4f}  |g| {row['grad_norm']:.3f}  "
              f"lr {row['learning_rate']:.2e}  {row['seconds']:.1f}s", flush=True)

    result = train(config, data, args.out, progress=report, evaluate_test=not args.skip_test)
    m = result["metrics"]
    test = "test split not evaluated" if args.skip_test else f"test acc {m['test_accuracy']:.4f} · test error {m['test_error']:.4f}"
    print(f"selected epoch {m['best_epoch']} (val acc {m['best_val_accuracy']:.4f}) · {test} · {m['parameters']:,} parameters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
