#!/usr/bin/env python3
"""Start a real, short-lived Uvicorn process with a temporary SQLite database."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="hw1-startup-") as directory:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = {**os.environ, "DATABASE_PATH": str(Path(directory) / "smoke.sqlite3"),
               "CLASSIFICATION_RUNS_DIR": str(Path(directory) / "runs")}
        imported = subprocess.run(
            [sys.executable, "-c", "from app.main import app; assert app.title == 'Neural 3D Data Viewer'"],
            cwd=ROOT / "backend", env=env, check=True,
        )
        assert imported.returncode == 0
        assert not Path(env["DATABASE_PATH"]).exists(), "Import must not create a database"
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT / "backend", env=env,
        )
        try:
            deadline = time.monotonic() + 60  # PyTorch is imported at startup
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Backend exited before startup")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as response:
                        assert json.load(response) == {"status": "ok", "database": "sqlite"}
                    with urlopen(f"http://127.0.0.1:{port}/api/scenes", timeout=1) as response:
                        assert json.load(response)[0]["id"] == "spectrum-garden"
                    assert Path(env["DATABASE_PATH"]).exists()
                    print("PASS: import without side effects, Uvicorn startup, HTTP health/scenes, SQLite creation")
                    classify(port)
                    break
                except URLError:
                    time.sleep(0.1)
            else:
                raise RuntimeError("Backend startup timed out")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def classify(port: int) -> None:
    """Real inference over HTTP with the bundled checkpoints and samples."""
    index = json.loads((ROOT / "backend" / "samples" / "modelnet10" / "index.json").read_text())
    entry = index["samples"][0]
    for model in ("shallow", "mini_pointnet"):
        body = json.dumps({"sample_id": entry["id"], "model_id": model}).encode()
        request = Request(f"http://127.0.0.1:{port}/api/classification/predict/sample", data=body,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=30) as response:
            result = json.load(response)
        assert result["predicted_class"] == entry["predictions"][model], result
        assert len(result["top_k"]) == 3 and result["model"]["id"] == model
    data = (ROOT / "backend" / "samples" / "modelnet10" / entry["file"]).read_bytes()
    request = Request(f"http://127.0.0.1:{port}/api/classification/predict/upload?filename={entry['file']}",
                      data=data, headers={"Content-Type": "application/octet-stream"})
    with urlopen(request, timeout=30) as response:
        uploaded = json.load(response)
    assert uploaded["predicted_class"] == entry["predictions"]["mini_pointnet"], uploaded
    print(f"PASS: HTTP classification of {entry['id']} ({entry['class']}) by both bundled models; "
          f"upload path agrees ({uploaded['predicted_class']}, {uploaded['confidence']:.3f})")


if __name__ == "__main__":
    main()
