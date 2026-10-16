"""Copy the selected training runs into the repository and write the bundled samples.

    python -m app.classification.bundle --baseline RUN_DIR --final RUN_DIR \\
        [--baseline-seeds DIR ...] [--final-seeds DIR ...]

Writes ``backend/models/modelnet10/{shallow,mini_pointnet}/`` (checkpoint,
config, history, metrics, plots), ``backend/models/modelnet10/comparison.json``
and ``backend/samples/modelnet10/`` — three official test-split objects per
class, stored exactly as the networks saw them (1,024 normalised points), with
each bundled model's prediction recorded for regression testing.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from . import dataset
from .inference import ModelRegistry, parse_ply, preprocess, write_ply
from .routes import ClassificationSettings
from .training import compare

ARTIFACTS = ("best.pt", "config.json", "history.json", "metrics.json", "curves.png", "confusion.png")
SAMPLES_PER_CLASS = 3
SAMPLE_SEED = 0


def seed_summary(folders: list[Path]) -> dict:
    accuracies = [json.loads((f / "metrics.json").read_text())["test_accuracy"] for f in folders]
    return {"runs": [str(f) for f in folders], "test_accuracy": accuracies,
            "mean": float(np.mean(accuracies)), "std": float(np.std(accuracies))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--baseline-seeds", type=Path, nargs="*", default=[])
    parser.add_argument("--final-seeds", type=Path, nargs="*", default=[])
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    settings = ClassificationSettings()

    for name, source in (("shallow", args.baseline), ("mini_pointnet", args.final)):
        target = settings.models_dir / name
        target.mkdir(parents=True, exist_ok=True)
        for artifact in ARTIFACTS:
            shutil.copy2(source / artifact, target / artifact)
    comparison = compare(settings.models_dir / "shallow", settings.models_dir / "mini_pointnet")
    if args.baseline_seeds and args.final_seeds:
        comparison["seeds"] = {"shallow": seed_summary(args.baseline_seeds), "mini_pointnet": seed_summary(args.final_seeds)}
    (settings.models_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")

    data = dataset.load(args.data_dir, download=False)
    rng = np.random.default_rng(SAMPLE_SEED)
    settings.samples_dir.mkdir(parents=True, exist_ok=True)
    for old in settings.samples_dir.glob("*.ply"):
        old.unlink()
    registry = ModelRegistry(settings.models_dir, settings.runs_dir)
    entries = []
    for label, name in enumerate(data.classes):
        candidates = np.flatnonzero(data.test_labels == label)
        for index in np.sort(rng.choice(candidates, SAMPLES_PER_CLASS, replace=False)):
            source = str(data.test_files[index])
            stem = Path(source).stem
            entry = {"id": stem, "class": name, "label": stem.replace(f"{name}_", f"{name.replace('_', ' ')} #"),
                     "file": f"{stem}.ply", "source_file": source, "split": "test"}
            write_ply(settings.samples_dir / entry["file"], data.test_points[index],
                      [f"ModelNet10 {source} (official test split)", "1024 surface points, centred, unit sphere",
                       "Source: Princeton ModelNet, https://modelnet.cs.princeton.edu/ (academic research use)"])
            points, _ = preprocess(parse_ply((settings.samples_dir / entry["file"]).read_bytes()))
            entry["predictions"] = {model: registry.predict(model, points)["predicted_class"]
                                    for model in ("shallow", "mini_pointnet")}
            entries.append(entry)
    (settings.samples_dir / "index.json").write_text(json.dumps(
        {"source": dataset.LICENSE_NOTE, "samples_per_class": SAMPLES_PER_CLASS, "sample_seed": SAMPLE_SEED,
         "samples": entries}, indent=2) + "\n")
    hits = {m: sum(e["predictions"][m] == e["class"] for e in entries) for m in ("shallow", "mini_pointnet")}
    print(json.dumps({"comparison": comparison, "bundled_samples": len(entries), "sample_hits": hits}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
