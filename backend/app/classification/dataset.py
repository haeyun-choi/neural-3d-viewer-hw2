"""ModelNet10: download, verify, extract, validate, sample, normalise and cache.

    python -m app.classification.dataset prepare [--data-dir DIR] [--workers N]

ModelNet10 (Wu et al., CVPR 2015) is distributed by the Princeton Vision & Robotics
group at https://modelnet.cs.princeton.edu/ for academic research. It is
downloaded on demand and never committed: the archive, the extracted meshes and
the processed cache all live under ``data_dir`` (default ``backend/data/modelnet10``,
which is Git-ignored).

The processed cache keeps the official split: 3,991 training and 908 test
meshes. The official training set is further split, stratified by class, into
training and validation subsets with a fixed seed. Model selection only ever
looks at validation; the test split is read once, after selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import NUM_POINTS, PREPROCESSING, points_from_geometry

CLASSES = ("bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet")
# Official ModelNet10 file counts per class: (train, test). 3,991 / 908 in total.
EXPECTED_COUNTS = {
    "bathtub": (106, 50), "bed": (515, 100), "chair": (889, 100), "desk": (200, 86), "dresser": (200, 86),
    "monitor": (465, 100), "night_stand": (200, 86), "sofa": (680, 100), "table": (392, 100), "toilet": (344, 100),
}
SOURCE_URLS = (
    "https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
)
ARCHIVE_NAME = "ModelNet10.zip"
ARCHIVE_BYTES = 473_402_300
ARCHIVE_SHA256 = "9d8679435fc07d1d26f13009878db164a7aa8ea5e7ea3c8880e42794b7307d51"
LICENSE_NOTE = ("ModelNet10, Princeton Vision & Robotics (Wu et al., '3D ShapeNets', CVPR 2015), "
                "https://modelnet.cs.princeton.edu/ — provided for academic research use.")
SPLIT_SEED = 0
SAMPLING_SEED = 0
VALIDATION_FRACTION = 0.2
CACHE_VERSION = 1
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "modelnet10"


def data_dir_from_env() -> Path:
    return Path(os.environ.get("MODELNET10_DATA_DIR", DEFAULT_DATA_DIR)).expanduser().resolve()


def cache_path(data_dir: Path) -> Path:
    return data_dir / "processed" / f"modelnet10_{NUM_POINTS}pts_seed{SAMPLING_SEED}_v{CACHE_VERSION}.npz"


class DatasetError(RuntimeError):
    pass


# --------------------------------------------------------------------------- download

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_urllib(url: str, target: Path, log) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "neural-3d-viewer-hw2/1.0"})
    # urllib follows the http -> https redirect the Princeton server issues.
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done, last = 0, 0.0
        while block := response.read(1 << 20):
            out.write(block)
            done += len(block)
            if time.monotonic() - last > 10:
                last = time.monotonic()
                log(f"  {done / 2**20:,.0f} / {total / 2**20:,.0f} MiB from {response.geturl()}")


def _download_cli(url: str, target: Path, log) -> None:
    for tool in (["curl", "-fL", "--retry", "3", "-sS", "-o", str(target), url],
                 ["wget", "-q", "-O", str(target), url]):
        if shutil.which(tool[0]):
            log(f"  retrying with {tool[0]}")
            subprocess.run(tool, check=True, timeout=3600)
            return
    raise DatasetError("Neither curl nor wget is available for a fallback download.")


def verify_archive(path: Path) -> str:
    """Size, checksum and ZIP integrity. Returns the SHA-256."""
    size = path.stat().st_size
    if size != ARCHIVE_BYTES:
        raise DatasetError(f"{path.name} is {size:,} bytes; the published archive is {ARCHIVE_BYTES:,} bytes.")
    digest = sha256_file(path)
    if digest != ARCHIVE_SHA256:
        raise DatasetError(f"{path.name} SHA-256 {digest} does not match the expected {ARCHIVE_SHA256}.")
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
    if bad is not None:
        raise DatasetError(f"{path.name} is corrupt: CRC check failed for {bad}.")
    return digest


def download_archive(data_dir: Path, log=print) -> Path:
    target = data_dir / ARCHIVE_NAME
    if target.exists():
        try:
            verify_archive(target)
            return target
        except (DatasetError, zipfile.BadZipFile) as error:
            log(f"Existing archive rejected ({error}); downloading again.")
            target.unlink()
    data_dir.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".zip.part")
    urls = [os.environ["MODELNET10_URL"]] if os.environ.get("MODELNET10_URL") else list(SOURCE_URLS)
    failures = []
    for url in urls:
        for method in (_download_urllib, _download_cli):
            partial.unlink(missing_ok=True)
            try:
                log(f"Downloading {url} ({method.__name__.strip('_')})")
                method(url, partial, log)
                verify_archive(partial)
                partial.replace(target)
                log(f"Verified {target} ({ARCHIVE_BYTES:,} bytes, SHA-256 {sha256_file(target)})")
                return target
            except Exception as error:  # noqa: BLE001 - every failure is reported below
                failures.append(f"{url} via {method.__name__.strip('_')}: {type(error).__name__}: {error}")
                log(f"  failed: {failures[-1]}")
    partial.unlink(missing_ok=True)
    raise DatasetError("ModelNet10 could not be downloaded:\n  " + "\n  ".join(failures))


def extract_archive(archive: Path, data_dir: Path) -> Path:
    root = data_dir / "ModelNet10"
    marker = root / ".extracted"
    if marker.exists():
        return root
    with zipfile.ZipFile(archive) as bundle:
        members = [m for m in bundle.infolist()
                   if m.filename.startswith("ModelNet10/") and m.filename.endswith(".off") and "__MACOSX" not in m.filename]
        for member in members:
            # Only plain relative paths inside ModelNet10/ are written.
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise DatasetError(f"Unsafe path in archive: {member.filename}")
            bundle.extract(member, data_dir)
    marker.write_text(str(len(members)))
    return root


def list_meshes(root: Path) -> dict[str, list[tuple[str, int]]]:
    """Validate the extracted tree against the official class list and counts."""
    found = {"train": [], "test": []}
    problems = []
    present = sorted(p.name for p in root.iterdir() if p.is_dir())
    if sorted(present) != sorted(CLASSES):
        problems.append(f"classes {present} != expected {sorted(CLASSES)}")
    for label, name in enumerate(CLASSES):
        for split_index, split in enumerate(("train", "test")):
            files = sorted((root / name / split).glob("*.off"))
            expected = EXPECTED_COUNTS[name][split_index]
            if len(files) != expected:
                problems.append(f"{name}/{split}: {len(files)} files, expected {expected}")
            found[split].extend((str(f.relative_to(root)), label) for f in files)
    if problems:
        raise DatasetError("ModelNet10 does not match the official layout: " + "; ".join(problems))
    return found


# --------------------------------------------------------------------------- meshes

def parse_off(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse an OFF mesh. Handles ModelNet's 'OFF490 518 0' merged header lines."""
    lines = [line for line in (raw.split("#", 1)[0].strip() for raw in text.splitlines()) if line]
    if not lines or not lines[0].upper().startswith("OFF"):
        raise ValueError("not an OFF file")
    rest = lines[0][3:].strip()
    if rest:
        counts, body = rest.split(), lines[1:]
    else:
        counts, body = lines[1].split(), lines[2:]
    n_vertices, n_faces = int(counts[0]), int(counts[1])
    vertices = np.array(" ".join(body[:n_vertices]).split(), dtype=np.float64).reshape(n_vertices, 3)
    face_lines = body[n_vertices:n_vertices + n_faces]
    triangles = []
    for line in face_lines:
        values = [int(v) for v in line.split()]
        k, index = values[0], values[1:1 + values[0]]
        triangles.extend((index[0], index[i], index[i + 1]) for i in range(1, k - 1))
    return vertices, np.asarray(triangles, dtype=np.int64).reshape(-1, 3)


def object_seed(relative_path: str) -> int:
    """A per-object seed that does not depend on processing order or worker count."""
    return SAMPLING_SEED * 1_000_003 + zlib.crc32(relative_path.encode())


def process_mesh(args: tuple[str, str]) -> np.ndarray:
    root, relative = args
    vertices, faces = parse_off(Path(root, relative).read_text())
    points, _ = points_from_geometry(vertices, faces, np.random.default_rng(object_seed(relative)))
    return points


# --------------------------------------------------------------------------- cache

def stratified_split(labels: np.ndarray, fraction: float = VALIDATION_FRACTION, seed: int = SPLIT_SEED):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in np.unique(labels):
        index = rng.permutation(np.flatnonzero(labels == label))
        cut = int(round(len(index) * fraction))
        val.extend(index[:cut])
        train.extend(index[cut:])
    return np.sort(np.asarray(train)), np.sort(np.asarray(val))


def prepare(data_dir: Path | None = None, workers: int | None = None, log=print) -> Path:
    data_dir = (data_dir or data_dir_from_env()).resolve()
    target = cache_path(data_dir)
    if target.exists():
        return target
    archive = download_archive(data_dir, log)
    archive_digest = sha256_file(archive)
    root = extract_archive(archive, data_dir)
    meshes = list_meshes(root)
    workers = workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    arrays = {}
    for split, items in meshes.items():
        started = time.monotonic()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            points = list(pool.map(process_mesh, [(str(root), rel) for rel, _ in items], chunksize=8))
        arrays[f"{split}_points"] = np.stack(points).astype(np.float32)
        arrays[f"{split}_labels"] = np.asarray([label for _, label in items], dtype=np.int64)
        arrays[f"{split}_files"] = np.asarray([rel for rel, _ in items])
        log(f"Sampled {len(items):,} {split} meshes in {time.monotonic() - started:.0f}s with {workers} workers")
    train_index, val_index = stratified_split(arrays["train_labels"])
    meta = {
        "dataset": "ModelNet10", "cache_version": CACHE_VERSION, "classes": list(CLASSES),
        "source_urls": list(SOURCE_URLS), "archive_sha256": archive_digest, "archive_bytes": ARCHIVE_BYTES,
        "license": LICENSE_NOTE, "preprocessing": PREPROCESSING, "sampling_seed": SAMPLING_SEED,
        "split_seed": SPLIT_SEED, "validation_fraction": VALIDATION_FRACTION,
        "counts": {"train": int(len(train_index)), "val": int(len(val_index)), "test": int(len(arrays["test_labels"]))},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.npz")
    np.savez_compressed(partial, **arrays, train_index=train_index, val_index=val_index, meta=json.dumps(meta))
    partial.replace(target)
    target.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    log(f"Cached {target} — train {len(train_index)}, val {len(val_index)}, test {len(arrays['test_labels'])}")
    return target


@dataclass
class PreparedData:
    classes: list[str]
    train_points: np.ndarray
    train_labels: np.ndarray
    val_points: np.ndarray
    val_labels: np.ndarray
    test_points: np.ndarray
    test_labels: np.ndarray
    test_files: np.ndarray
    meta: dict

    def summary(self) -> dict:
        return {"train": len(self.train_labels), "val": len(self.val_labels), "test": len(self.test_labels)}


def load(data_dir: Path | None = None, download: bool = True, log=print) -> PreparedData:
    data_dir = (data_dir or data_dir_from_env()).resolve()
    target = cache_path(data_dir)
    if not target.exists():
        if not download:
            raise DatasetError(f"No processed ModelNet10 cache at {target}. Run: python -m app.classification.dataset prepare")
        prepare(data_dir, log=log)
    with np.load(target, allow_pickle=False) as data:
        train_index, val_index = data["train_index"], data["val_index"]
        return PreparedData(
            classes=json.loads(str(data["meta"]))["classes"],
            train_points=data["train_points"][train_index], train_labels=data["train_labels"][train_index],
            val_points=data["train_points"][val_index], val_labels=data["train_labels"][val_index],
            test_points=data["test_points"], test_labels=data["test_labels"], test_files=data["test_files"],
            meta=json.loads(str(data["meta"])),
        )


def status(data_dir: Path | None = None) -> dict:
    data_dir = (data_dir or data_dir_from_env()).resolve()
    target = cache_path(data_dir)
    info = {"prepared": target.exists(), "archive_present": (data_dir / ARCHIVE_NAME).exists(),
            "download_bytes": ARCHIVE_BYTES, "source": SOURCE_URLS[0], "license": LICENSE_NOTE}
    meta_file = target.with_suffix(".json")
    if meta_file.exists():
        info["counts"] = json.loads(meta_file.read_text())["counts"]
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "status"])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    if args.command == "status":
        print(json.dumps(status(args.data_dir), indent=2))
        return 0
    try:
        prepare(args.data_dir, args.workers)
    except DatasetError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
