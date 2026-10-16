"""Background training runs launched from the web application.

One worker thread trains one run at a time; further runs wait as ``queued``.
States: queued → running → completed | cancelled | failed. Each run's status,
including the per-epoch history, is written to ``runs_dir/<id>/status.json``
after every change, so a finished run survives a backend restart and a run that
was interrupted by one is reported as failed rather than left "running".
"""

from __future__ import annotations

import json
import logging
import os
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

from . import dataset
from .training import TrainConfig, TrainingCancelled, resolve_device, save_run, train

ACTIVE = ("queued", "running")
MAX_QUEUED = 3
logger = logging.getLogger(__name__)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class QueueFull(RuntimeError):
    pass


class TrainingManager:
    def __init__(self, runs_dir: Path, data_loader: Callable[[], dataset.PreparedData]):
        self.runs_dir = runs_dir
        self.load_data = data_loader
        self._runs: dict[str, dict] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="training")
        self._restore()

    def _restore(self) -> None:
        if not self.runs_dir.exists():
            return
        for folder in sorted(self.runs_dir.iterdir()):
            path = folder / "status.json"
            try:
                run = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if run.get("status") in ACTIVE:
                run.update(status="failed", error="Interrupted: the backend stopped before this run finished.",
                           finished_at=run.get("finished_at") or now())
                path.write_text(json.dumps(run, indent=2))
            self._runs[run["id"]] = run

    def _save(self, run: dict) -> None:
        folder = self.runs_dir / run["id"]
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / "status.json.tmp"
        temporary.write_text(json.dumps(run, indent=2))
        temporary.replace(folder / "status.json")

    def _update(self, run_id: str, **changes) -> None:
        with self._lock:
            run = self._runs[run_id]
            run.update(changes)
            self._save(run)

    def submit(self, config: TrainConfig) -> dict:
        with self._lock:
            if sum(r["status"] in ACTIVE for r in self._runs.values()) >= MAX_QUEUED + 1:
                raise QueueFull("Too many training runs are waiting. Cancel one or wait for it to finish.")
            run_id = uuid.uuid4().hex[:12]
            run = {"id": run_id, "status": "queued", "phase": "waiting for the trainer", "config": config.to_dict(),
                   "created_at": now(), "started_at": None, "finished_at": None, "epoch": 0,
                   "total_epochs": config.epochs, "history": [], "metrics": None, "error": None, "device": None}
            self._runs[run_id] = run
            self._cancel[run_id] = threading.Event()
            self._save(run)
        self._executor.submit(self._execute, run_id)
        return self.get(run_id)

    def _execute(self, run_id: str) -> None:
        stop = self._cancel[run_id]
        if stop.is_set():
            return
        config = TrainConfig.from_dict(self._runs[run_id]["config"])
        threads = int(os.environ.get("CLASSIFICATION_TRAIN_THREADS", "0"))
        if threads > 0:
            torch.set_num_threads(threads)
        self._update(run_id, status="running", started_at=now(), phase="loading ModelNet10",
                     device=str(resolve_device(config.device)))
        try:
            data = self.load_data()
            if stop.is_set():
                raise TrainingCancelled()

            def progress(row: dict) -> None:
                with self._lock:
                    run = self._runs[run_id]
                    run["history"].append(row)
                    run["epoch"] = row["epoch"]
                    self._save(run)

            result = train(config, data, progress=progress, should_stop=stop.is_set,
                           phase=lambda name: self._update(run_id, phase=name))
            save_run(self.runs_dir / run_id, result["model"], config, data.classes, result["history"], result["metrics"])
            self._update(run_id, status="completed", phase="completed", finished_at=now(), metrics=result["metrics"])
        except TrainingCancelled:
            self._update(run_id, status="cancelled", phase="cancelled", finished_at=now())
        except Exception as error:  # noqa: BLE001 - a failed run reports why, the server keeps running
            logger.error("Training run %s failed:\n%s", run_id, traceback.format_exc())
            self._update(run_id, status="failed", phase="failed", finished_at=now(),
                         error=f"{type(error).__name__}: {error}")

    def cancel(self, run_id: str) -> dict:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(run_id)
            if run["status"] in ACTIVE:
                self._cancel[run_id].set()
                if run["status"] == "queued":
                    run.update(status="cancelled", phase="cancelled", finished_at=now())
                else:
                    run["phase"] = "cancelling after the current batch"
                self._save(run)
            return json.loads(json.dumps(run))

    def get(self, run_id: str) -> dict:
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(run_id)
            return json.loads(json.dumps(self._runs[run_id]))

    def list(self) -> list[dict]:
        with self._lock:
            runs = sorted(self._runs.values(), key=lambda r: r["created_at"], reverse=True)
            return [{k: v for k, v in r.items() if k != "history"} | {"last": r["history"][-1] if r["history"] else None}
                    for r in json.loads(json.dumps(runs))]

    def shutdown(self) -> None:
        for event in self._cancel.values():
            event.set()
        self._executor.shutdown(wait=True, cancel_futures=True)
