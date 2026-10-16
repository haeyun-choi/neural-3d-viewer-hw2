"""Small generated stand-ins for ModelNet10, so the API and training tests need no download."""

import json
from pathlib import Path

import numpy as np

from app.classification import dataset
from app.classification.geometry import normalize_points
from app.classification.inference import write_ply
from app.classification.models import build_model
from app.classification.training import TrainConfig, save_run


def toy_object(label: int, rng: np.random.Generator, count: int = 1024) -> np.ndarray:
    """Ten distinguishable shapes: ellipsoids whose axes depend on the class."""
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    axes = np.array([1 + (label % 3), 1 + (label // 3) % 3, 1 + label / 4])
    return normalize_points(direction * axes + rng.normal(scale=0.01, size=(count, 3)))


def write_fixture_dataset(data_dir: Path, per_class: int = 6, test_per_class: int = 2) -> Path:
    rng = np.random.default_rng(0)
    classes = list(dataset.CLASSES)
    arrays = {}
    for split, n in (("train", per_class), ("test", test_per_class)):
        labels = np.repeat(np.arange(len(classes)), n)
        arrays[f"{split}_points"] = np.stack([toy_object(int(k), rng) for k in labels]).astype(np.float32)
        arrays[f"{split}_labels"] = labels.astype(np.int64)
        arrays[f"{split}_files"] = np.asarray([f"{classes[k]}/{split}/{classes[k]}_{i:04d}.off" for i, k in enumerate(labels)])
    train_index, val_index = dataset.stratified_split(arrays["train_labels"])
    meta = {"classes": classes, "counts": {"train": len(train_index), "val": len(val_index), "test": len(arrays["test_labels"])}}
    target = dataset.cache_path(data_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays, train_index=train_index, val_index=val_index, meta=json.dumps(meta))
    target.with_suffix(".json").write_text(json.dumps(meta))
    return target


def write_fixture_models(models_dir: Path) -> None:
    """Untrained but structurally real checkpoints, written by the same save_run as real training."""
    for name in ("shallow", "mini_pointnet"):
        config = TrainConfig(model=name, epochs=1)
        model = build_model(name, 10, config.hidden_dim, config.activation, config.dropout).eval()
        metrics = {"model": name, "parameters": 1, "best_epoch": 1, "best_val_accuracy": 0.1, "test_accuracy": 0.1,
                   "test_error": 0.9, "git_sha": None, "split_sizes": {"test": 20}, "classes": list(dataset.CLASSES),
                   "confusion_matrix": np.eye(10, dtype=int).tolist()}
        history = [{"epoch": 1, "train_loss": 2.3, "val_loss": 2.3, "train_accuracy": 0.1, "val_accuracy": 0.1}]
        save_run(models_dir / name, model, config, list(dataset.CLASSES), history, metrics)


def write_fixture_samples(samples_dir: Path) -> list[dict]:
    rng = np.random.default_rng(1)
    samples_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for label in (1, 2):
        name = dataset.CLASSES[label]
        entry = {"id": f"{name}_0001", "class": name, "label": f"{name} 0001", "file": f"{name}_0001.ply",
                 "source_file": f"{name}/test/{name}_0001.off", "split": "test"}
        write_ply(samples_dir / entry["file"], toy_object(label, rng), ["fixture"])
        entries.append(entry)
    (samples_dir / "index.json").write_text(json.dumps({"samples": entries}))
    return entries
