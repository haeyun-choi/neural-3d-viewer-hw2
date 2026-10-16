"""FastAPI routes for ModelNet10 classification, mounted under /api/classification."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from . import dataset
from .inference import (BUNDLED_MODELS, DEFAULT_MODEL, MAX_UPLOAD_BYTES, InputError, ModelRegistry, parse_ply,
                        preprocess)
from .jobs import QueueFull, TrainingManager
from .models import ACTIVATIONS
from .training import TrainConfig

BACKEND_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ClassificationSettings:
    models_dir: Path = field(default_factory=lambda: Path(os.environ.get("CLASSIFICATION_MODELS_DIR", BACKEND_ROOT / "models" / "modelnet10")))
    samples_dir: Path = field(default_factory=lambda: Path(os.environ.get("CLASSIFICATION_SAMPLES_DIR", BACKEND_ROOT / "samples" / "modelnet10")))
    runs_dir: Path = field(default_factory=lambda: Path(os.environ.get("CLASSIFICATION_RUNS_DIR", BACKEND_ROOT / "data" / "runs")))
    data_dir: Path = field(default_factory=dataset.data_dir_from_env)
    # A GUI run downloads and preprocesses ModelNet10 on first use unless disabled.
    allow_download: bool = True


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["shallow", "mini_pointnet"] = "mini_pointnet"
    epochs: int = Field(20, strict=True, ge=1, le=200)
    learning_rate: float = Field(1e-3, gt=0, le=0.1, allow_inf_nan=False)
    hidden_dim: int = Field(64, strict=True, ge=8, le=256)
    activation: Literal[tuple(ACTIVATIONS)] = "relu"  # type: ignore[valid-type]
    batch_size: int = Field(32, strict=True, ge=4, le=256)
    dropout: float = Field(0.3, ge=0, le=0.8, allow_inf_nan=False)
    seed: int = Field(0, strict=True, ge=0, le=2**31 - 1)


class SamplePredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_id: str = Field(min_length=1, max_length=80)
    model_id: str = Field(DEFAULT_MODEL, min_length=1, max_length=40)
    top_k: int = Field(3, strict=True, ge=1, le=10)


def create_router(settings: ClassificationSettings) -> tuple[APIRouter, TrainingManager]:
    router = APIRouter(prefix="/api/classification", tags=["classification"])
    registry = ModelRegistry(settings.models_dir, settings.runs_dir)
    manager = TrainingManager(settings.runs_dir, lambda: dataset.load(settings.data_dir, download=settings.allow_download))

    def samples() -> list[dict]:
        try:
            return json.loads((settings.samples_dir / "index.json").read_text())["samples"]
        except (OSError, ValueError, KeyError):
            return []

    def sample(sample_id: str) -> dict:
        for entry in samples():
            if entry["id"] == sample_id:
                return entry
        raise HTTPException(404, f"Unknown sample '{sample_id}'.")

    def classify(data: bytes, model_id: str, top_k: int) -> dict:
        started = time.perf_counter()
        try:
            geometry = parse_ply(data)
            points, info = preprocess(geometry)
        except InputError as error:
            raise HTTPException(422, str(error)) from None
        preprocess_ms = (time.perf_counter() - started) * 1000
        try:
            result = registry.predict(model_id, points, top_k)
        except KeyError as error:
            raise HTTPException(404, error.args[0]) from None
        return {**result, "input": info, "preprocess_ms": preprocess_ms}

    @router.get("/models")
    def models():
        return {"classes": list(dataset.CLASSES), "default_model": DEFAULT_MODEL, "models": registry.available(),
                "activations": list(ACTIVATIONS), "architectures": ["shallow", "mini_pointnet"]}

    @router.get("/dataset")
    def dataset_status():
        return dataset.status(settings.data_dir)

    @router.get("/samples")
    def list_samples():
        return {"samples": samples(), "source": dataset.LICENSE_NOTE}

    @router.get("/samples/{sample_id}/ply")
    def sample_ply(sample_id: str):
        entry = sample(sample_id)
        return FileResponse(settings.samples_dir / entry["file"], media_type="application/octet-stream",
                            filename=entry["file"])

    @router.post("/predict/sample")
    def predict_sample(payload: SamplePredictRequest):
        entry = sample(payload.sample_id)
        result = classify((settings.samples_dir / entry["file"]).read_bytes(), payload.model_id, payload.top_k)
        return {**result, "sample": entry}

    @router.post("/predict/upload")
    async def predict_upload(request: Request, model_id: str = Query(DEFAULT_MODEL, max_length=40),
                             top_k: int = Query(3, ge=1, le=10), filename: str = Query("upload.ply", max_length=255)):
        if not filename.lower().endswith(".ply"):
            raise HTTPException(415, "Only .ply files can be classified.")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"The file is larger than the {MAX_UPLOAD_BYTES // 2**20} MiB classification limit.")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"The file is larger than the {MAX_UPLOAD_BYTES // 2**20} MiB classification limit.")
        if not body:
            raise HTTPException(422, "The upload is empty.")
        result = await run_in_threadpool(classify, bytes(body), model_id, top_k)
        return {**result, "filename": filename, "size_bytes": len(body)}

    @router.post("/runs", status_code=202)
    def start_run(payload: TrainRequest):
        try:
            return manager.submit(TrainConfig.from_dict(payload.model_dump()))
        except QueueFull as error:
            raise HTTPException(429, str(error)) from None

    @router.get("/runs")
    def list_runs():
        return {"runs": manager.list()}

    @router.get("/runs/{run_id}")
    def get_run(run_id: str):
        try:
            return manager.get(run_id)
        except KeyError:
            raise HTTPException(404, "Training run not found.") from None

    @router.post("/runs/{run_id}/cancel")
    def cancel_run(run_id: str):
        try:
            return manager.cancel(run_id)
        except KeyError:
            raise HTTPException(404, "Training run not found.") from None

    @router.get("/results")
    def results():
        out = {}
        for model_id, label in BUNDLED_MODELS.items():
            folder = settings.models_dir / model_id
            try:
                out[model_id] = {"label": label, "metrics": json.loads((folder / "metrics.json").read_text()),
                                 "history": json.loads((folder / "history.json").read_text()),
                                 "config": json.loads((folder / "config.json").read_text())}
            except (OSError, ValueError):
                continue
        try:
            comparison = json.loads((settings.models_dir / "comparison.json").read_text())
        except (OSError, ValueError):
            comparison = None
        return {"models": out, "comparison": comparison}

    return router, manager
