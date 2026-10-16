"""Prediction: PLY parsing, the shared preprocessing, and checkpoint loading.

Uploaded bytes are parsed in memory and discarded after the response; nothing
is written to disk or to the database.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .geometry import NUM_POINTS, points_from_geometry
from .training import load_checkpoint

MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_VERTICES = 5_000_000
INFERENCE_SEED = 0
BUNDLED_MODELS = {
    "mini_pointnet": "Mini-PointNet (final)",
    "shallow": "Shallow baseline",
}
DEFAULT_MODEL = "mini_pointnet"

PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


class InputError(ValueError):
    """The input cannot be classified; the message is safe to show the user."""


@dataclass
class PlyGeometry:
    vertices: np.ndarray
    faces: np.ndarray | None
    format: str


def _parse_header(data: bytes):
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise InputError("Not a PLY file: it must start with 'ply' and contain 'end_header'.")
    newline = data.find(b"\n", end)
    body_start = len(data) if newline < 0 else newline + 1
    lines = data[:end].decode("ascii", errors="replace").splitlines()
    fmt, elements = None, []
    for line in lines[1:]:
        words = line.split()
        if not words or words[0] in ("comment", "obj_info"):
            continue
        if words[0] == "format":
            fmt = words[1] if len(words) > 1 else None
        elif words[0] == "element" and len(words) == 3:
            elements.append({"name": words[1], "count": int(words[2]), "properties": []})
        elif words[0] == "property" and elements:
            if words[1] == "list" and len(words) == 5:
                elements[-1]["properties"].append((words[4], "list", words[2], words[3]))
            elif len(words) == 3:
                elements[-1]["properties"].append((words[2], "scalar", words[1], None))
            else:
                raise InputError(f"Malformed PLY property line: {line.strip()!r}")
        else:
            raise InputError(f"Malformed PLY header line: {line.strip()!r}")
    if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise InputError("Unsupported PLY format; use ascii, binary_little_endian or binary_big_endian.")
    for element in elements:
        for _, kind, t1, t2 in element["properties"]:
            if t1 not in PLY_TYPES or (kind == "list" and t2 not in PLY_TYPES):
                raise InputError(f"Unsupported PLY property type in element '{element['name']}'.")
    return fmt, elements, body_start


def _faces_from_lists(lists: list[np.ndarray]) -> np.ndarray:
    triangles = [(f[0], f[i], f[i + 1]) for f in lists for i in range(1, len(f) - 1)]
    return np.asarray(triangles, dtype=np.int64).reshape(-1, 3)


def parse_ply(data: bytes) -> PlyGeometry:
    fmt, elements, offset = _parse_header(data)
    names = [e["name"] for e in elements]
    if "vertex" not in names:
        raise InputError("The PLY has no vertex element.")
    vertex = elements[names.index("vertex")]
    props = [p[0] for p in vertex["properties"]]
    if not all(axis in props for axis in ("x", "y", "z")):
        raise InputError("The PLY vertex element needs x, y and z properties.")
    if any(p[1] == "list" for p in vertex["properties"]):
        raise InputError("List properties on vertices are not supported.")
    if vertex["count"] < 1:
        raise InputError("The PLY contains no points.")
    if vertex["count"] > MAX_VERTICES:
        raise InputError(f"The PLY has {vertex['count']:,} vertices; the limit for classification is {MAX_VERTICES:,}.")
    vertices, faces = None, None
    if fmt == "ascii":
        rows = iter(data[offset:].decode("ascii", errors="replace").splitlines())
        for element in elements:
            parsed = []
            for _ in range(element["count"]):
                line = next(rows, None)
                while line is not None and not line.strip():
                    line = next(rows, None)
                if line is None:
                    raise InputError(f"The PLY ends before all {element['count']:,} '{element['name']}' rows.")
                parsed.append(line.split())
            if element is vertex:
                try:
                    index = [props.index(a) for a in ("x", "y", "z")]
                    vertices = np.array([[float(r[i]) for i in index] for r in parsed], dtype=np.float64)
                except (ValueError, IndexError) as error:
                    raise InputError("A vertex row is not a list of numbers.") from error
            elif element["name"] == "face" and element["properties"] and element["properties"][0][1] == "list":
                try:
                    faces = _faces_from_lists([np.array(r[1:1 + int(r[0])], dtype=np.int64) for r in parsed])
                except (ValueError, IndexError) as error:
                    raise InputError("A face row is malformed.") from error
    else:
        endian = "<" if fmt == "binary_little_endian" else ">"
        for element in elements:
            if all(p[1] == "scalar" for p in element["properties"]):
                dtype = np.dtype([(p[0], endian + PLY_TYPES[p[2]]) for p in element["properties"]])
                size = dtype.itemsize * element["count"]
                if offset + size > len(data):
                    raise InputError(f"The PLY ends before all {element['count']:,} '{element['name']}' rows.")
                rows = np.frombuffer(data, dtype=dtype, count=element["count"], offset=offset)
                offset += size
                if element is vertex:
                    vertices = np.stack([rows[a].astype(np.float64) for a in ("x", "y", "z")], axis=1)
                continue
            lists = []
            for _ in range(element["count"]):
                for name, kind, t1, t2 in element["properties"]:
                    if kind == "scalar":
                        offset += np.dtype(PLY_TYPES[t1]).itemsize
                        continue
                    count_type = np.dtype(endian + PLY_TYPES[t1])
                    item_type = np.dtype(endian + PLY_TYPES[t2])
                    if offset + count_type.itemsize > len(data):
                        raise InputError(f"The PLY ends inside the '{element['name']}' element.")
                    n = int(np.frombuffer(data, count_type, 1, offset)[0])
                    offset += count_type.itemsize
                    if n < 0 or offset + n * item_type.itemsize > len(data):
                        raise InputError(f"The PLY ends inside the '{element['name']}' element.")
                    values = np.frombuffer(data, item_type, n, offset)
                    offset += n * item_type.itemsize
                    if element["name"] == "face" and name in ("vertex_indices", "vertex_index"):
                        lists.append(values.astype(np.int64))
            if element["name"] == "face":
                faces = _faces_from_lists(lists)
    label = {"ascii": "PLY · ASCII", "binary_little_endian": "PLY · binary LE", "binary_big_endian": "PLY · binary BE"}[fmt]
    return PlyGeometry(vertices=vertices, faces=faces if faces is not None and len(faces) else None, format=label)


def write_ply(path: Path, points: np.ndarray, comments: list[str] = ()) -> None:
    """Binary little-endian point-cloud PLY (x, y, z float32) for the bundled samples."""
    header = ["ply", "format binary_little_endian 1.0", *(f"comment {c}" for c in comments),
              f"element vertex {len(points)}", "property float x", "property float y", "property float z", "end_header"]
    path.write_bytes(("\n".join(header) + "\n").encode() + np.asarray(points, dtype="<f4").tobytes())


def preprocess(geometry: PlyGeometry) -> tuple[np.ndarray, dict]:
    try:
        points, mode = points_from_geometry(geometry.vertices, geometry.faces, np.random.default_rng(INFERENCE_SEED))
    except ValueError as error:
        raise InputError(str(error)) from error
    return points, {
        "input_points": int(len(geometry.vertices)),
        "input_faces": int(len(geometry.faces)) if geometry.faces is not None else 0,
        "model_points": NUM_POINTS,
        "sampling": "surface sampling of the mesh" if mode == "surface" else "subsampled from the point cloud",
        "format": geometry.format,
    }


class ModelRegistry:
    """Bundled checkpoints plus completed GUI training runs, loaded lazily."""

    def __init__(self, models_dir: Path, runs_dir: Path):
        self.models_dir, self.runs_dir = models_dir, runs_dir
        self._cache: dict[str, tuple[float, torch.nn.Module, dict]] = {}
        self._lock = threading.Lock()

    def checkpoint_path(self, model_id: str) -> Path:
        if model_id in BUNDLED_MODELS:
            return self.models_dir / model_id / "best.pt"
        match = re.fullmatch(r"run-([0-9a-f]{12})", model_id)
        if match:
            return self.runs_dir / match.group(1) / "best.pt"
        raise KeyError(model_id)

    def available(self) -> list[dict]:
        models = []
        for model_id, label in BUNDLED_MODELS.items():
            folder = self.models_dir / model_id
            if (folder / "best.pt").exists():
                metrics = _read_json(folder / "metrics.json")
                config = _read_json(folder / "config.json")
                models.append({"id": model_id, "label": label, "source": "bundled", "architecture": config.get("model", model_id),
                               "config": config, "parameters": metrics.get("parameters"),
                               "test_accuracy": metrics.get("test_accuracy"), "best_val_accuracy": metrics.get("best_val_accuracy")})
        if self.runs_dir.exists():
            for folder in sorted(self.runs_dir.iterdir()):
                status = _read_json(folder / "status.json")
                if status.get("status") == "completed" and (folder / "best.pt").exists():
                    metrics = status.get("metrics") or {}
                    config = status.get("config") or {}
                    models.append({"id": f"run-{folder.name}", "label": f"Run {folder.name[:6]} · {config.get('model')}",
                                   "source": "run", "architecture": config.get("model"), "config": config,
                                   "parameters": metrics.get("parameters"), "test_accuracy": metrics.get("test_accuracy"),
                                   "best_val_accuracy": metrics.get("best_val_accuracy")})
        return models

    def get(self, model_id: str) -> tuple[torch.nn.Module, dict]:
        try:
            path = self.checkpoint_path(model_id)
        except KeyError:
            raise KeyError(f"Unknown model '{model_id}'.") from None
        if not path.exists():
            raise KeyError(f"Model '{model_id}' has no checkpoint at {path.name}.")
        mtime = path.stat().st_mtime
        with self._lock:
            cached = self._cache.get(model_id)
            if cached is None or cached[0] != mtime:
                model, checkpoint = load_checkpoint(path, "cpu")
                checkpoint.pop("state_dict", None)
                self._cache[model_id] = cached = (mtime, model, checkpoint)
            return cached[1], cached[2]

    def predict(self, model_id: str, points: np.ndarray, top_k: int = 3) -> dict:
        model, checkpoint = self.get(model_id)
        classes = checkpoint["classes"]
        started = time.perf_counter()
        with self._lock, torch.no_grad():
            model.eval()
            logits = model(torch.from_numpy(points).float().unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)[0].numpy()
        elapsed = (time.perf_counter() - started) * 1000
        order = np.argsort(-probabilities)
        top = [{"class": classes[i], "probability": float(probabilities[i])} for i in order[:max(1, min(top_k, len(classes)))]]
        return {
            "predicted_class": top[0]["class"], "confidence": top[0]["probability"], "top_k": top,
            "probabilities": {classes[i]: float(p) for i, p in enumerate(probabilities)},
            "model": {"id": model_id, "architecture": checkpoint["model"], "config": checkpoint["model_config"],
                      "parameters": checkpoint.get("parameters"), "test_accuracy": checkpoint.get("test_accuracy")},
            "inference_ms": elapsed,
        }


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
