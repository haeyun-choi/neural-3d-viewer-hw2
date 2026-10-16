"""ModelNet10 classification: preprocessing, models, training, checkpoints and the API."""

import json
import struct
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app.classification import dataset, routes
from app.classification.geometry import NUM_POINTS, normalize_points, points_from_geometry, sample_surface
from app.classification.inference import InputError, ModelRegistry, parse_ply, preprocess, write_ply
from app.classification.models import ACTIVATIONS, MiniPointNet, ShallowPointNet, build_model, parameter_count
from app.classification.routes import ClassificationSettings
from app.classification.training import TrainConfig, load_checkpoint, train
from app.main import create_app

from classification_fixtures import toy_object, write_fixture_dataset, write_fixture_models, write_fixture_samples

BACKEND = Path(__file__).resolve().parents[1]
API = "/api/classification"


# ----------------------------------------------------------------------------- preprocessing

def test_normalisation_centres_and_scales_into_unit_sphere():
    points = np.random.default_rng(0).normal(size=(500, 3)) * [3, 1, 0.5] + [10, -4, 2]
    out = normalize_points(points)
    assert out.dtype == np.float32
    assert np.allclose(out.mean(axis=0), 0, atol=1e-6)
    assert np.isclose(np.linalg.norm(out, axis=1).max(), 1, atol=1e-6)
    assert np.allclose(normalize_points(out), out, atol=1e-6)  # idempotent
    with pytest.raises(ValueError):
        normalize_points(np.ones((10, 3)))


def test_surface_sampling_is_on_the_mesh_area_weighted_and_deterministic():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 5], [3, 0, 5], [0, 1, 5]], float)
    faces = np.array([[0, 1, 2], [3, 4, 5]])  # areas 0.5 and 1.5
    points = sample_surface(vertices, faces, 20_000, np.random.default_rng(0))
    on_upper = np.isclose(points[:, 2], 5)
    assert (on_upper | np.isclose(points[:, 2], 0)).all()  # every point on one of the two planes
    assert (points[:, :2] >= -1e-12).all()
    upper = points[on_upper]
    assert (upper[:, 0] / 3 + upper[:, 1] <= 1 + 1e-9).all()  # inside the triangle
    assert abs(on_upper.mean() - 0.75) < 0.02
    again = sample_surface(vertices, faces, 20_000, np.random.default_rng(0))
    assert np.array_equal(points, again)


def test_points_from_geometry_meshes_point_clouds_and_bad_input():
    rng = np.random.default_rng(0)
    cube = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5]])
    points, mode = points_from_geometry(cube, faces, rng)
    assert mode == "surface" and points.shape == (NUM_POINTS, 3)
    points, mode = points_from_geometry(cube, None, rng)
    assert mode == "vertices" and points.shape == (NUM_POINTS, 3)
    many = np.random.default_rng(1).normal(size=(5000, 3))
    points, _ = points_from_geometry(many, None, np.random.default_rng(0))
    assert len(np.unique(points, axis=0)) == NUM_POINTS  # without replacement
    degenerate = np.array([[0, 1, 2]])
    assert points_from_geometry(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], float), degenerate, rng)[1] == "vertices"
    with pytest.raises(ValueError):
        points_from_geometry(np.array([[0, 0, np.nan], [1, 1, 1]]), None, rng)
    with pytest.raises(ValueError):
        points_from_geometry(cube, np.array([[0, 1, 99]]), rng)


def test_off_parser_handles_modelnet_merged_header_and_polygons():
    text = "OFF4 2 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n3 0 1 2\n"
    vertices, faces = dataset.parse_off(text)
    assert vertices.shape == (4, 3)
    assert faces.tolist() == [[0, 1, 2], [0, 2, 3], [0, 1, 2]]
    vertices, faces = dataset.parse_off("OFF\n# comment\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")
    assert faces.tolist() == [[0, 1, 2]]
    with pytest.raises(ValueError):
        dataset.parse_off("PLY\n")


def test_stratified_split_is_deterministic_disjoint_and_per_class():
    labels = np.repeat(np.arange(10), [106, 515, 889, 200, 200, 465, 200, 680, 392, 344])
    train_index, val_index = dataset.stratified_split(labels)
    again = dataset.stratified_split(labels)
    assert np.array_equal(train_index, again[0]) and np.array_equal(val_index, again[1])
    assert not set(train_index) & set(val_index)
    assert len(train_index) + len(val_index) == 3991
    for k in range(10):
        n = (labels == k).sum()
        assert (labels[val_index] == k).sum() == round(n * 0.2)


def test_dataset_layout_validation_and_archive_size_check(tmp_path):
    root = tmp_path / "ModelNet10"
    for name in dataset.CLASSES:
        for split in ("train", "test"):
            (root / name / split).mkdir(parents=True)
    (root / "chair" / "train" / "chair_0001.off").write_text("OFF\n")
    with pytest.raises(dataset.DatasetError, match="chair/train: 1 files, expected 889"):
        dataset.list_meshes(root)
    archive = tmp_path / "ModelNet10.zip"
    archive.write_bytes(b"not the archive")
    with pytest.raises(dataset.DatasetError, match="published archive"):
        dataset.verify_archive(archive)


# ----------------------------------------------------------------------------- PLY input

def ascii_ply(points, faces=()):
    lines = ["ply", "format ascii 1.0", f"element vertex {len(points)}", "property float x", "property float y",
             "property float z", "property uchar red", f"element face {len(faces)}", "property list uchar int vertex_indices",
             "end_header"]
    lines += [f"{x} {y} {z} 255" for x, y, z in points] + [f"{len(f)} " + " ".join(map(str, f)) for f in faces]
    return ("\n".join(lines) + "\n").encode()


def binary_ply(points, faces, endian):
    tag = {"<": "binary_little_endian", ">": "binary_big_endian"}[endian]
    header = ["ply", f"format {tag} 1.0", f"element vertex {len(points)}", "property double x", "property double y",
              "property double z", f"element face {len(faces)}", "property list uchar uint vertex_indices", "end_header"]
    body = b"".join(struct.pack(f"{endian}3d", *p) for p in points)
    body += b"".join(struct.pack(f"{endian}B{len(f)}I", len(f), *f) for f in faces)
    return ("\n".join(header) + "\n").encode() + body


def test_ply_parser_reads_ascii_and_both_binary_layouts_with_faces():
    points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    for data, fmt in ((ascii_ply(points, [(0, 1, 2, 3)]), "PLY · ASCII"),
                      (binary_ply(points, [(0, 1, 2, 3)], "<"), "PLY · binary LE"),
                      (binary_ply(points, [(0, 1, 2, 3)], ">"), "PLY · binary BE")):
        geometry = parse_ply(data)
        assert geometry.format == fmt
        assert np.allclose(geometry.vertices, points)
        assert geometry.faces.tolist() == [[0, 1, 2], [0, 2, 3]]
    cloud = parse_ply(ascii_ply(points))
    assert cloud.faces is None


@pytest.mark.parametrize("data, message", [
    (b"", "Not a PLY"),
    (b"hello world", "Not a PLY"),
    (b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n1\n", "x, y and z"),
    (b"ply\nformat binary_little_endian 1.0\nelement vertex 10\nproperty float x\nproperty float y\nproperty float z\nend_header\n\0\0", "ends before"),
    (b"ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 2 3\n", "ends before"),
    (b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\na b c\n", "not a list of numbers"),
    (b"ply\nformat foo 1.0\nelement vertex 1\nend_header\n", "Unsupported PLY format"),
    (b"ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n", "no points"),
])
def test_ply_parser_rejects_malformed_input(data, message):
    with pytest.raises(InputError, match=message):
        parse_ply(data)


def test_bundled_sample_preprocessing_reproduces_the_cached_points(tmp_path):
    original = toy_object(3, np.random.default_rng(0))
    write_ply(tmp_path / "s.ply", original)
    points, info = preprocess(parse_ply((tmp_path / "s.ply").read_bytes()))
    assert info["input_points"] == NUM_POINTS and info["model_points"] == NUM_POINTS
    # Same 1,024 points (a permutation), same normalisation.
    order = lambda a: a[np.lexsort(a.T)]
    assert np.allclose(order(points), order(original), atol=1e-5)


# ----------------------------------------------------------------------------- models

def test_model_shapes_architecture_and_permutation_invariance():
    x = torch.randn(4, NUM_POINTS, 3)
    shallow, mini = build_model("shallow"), build_model("mini_pointnet")
    assert isinstance(shallow, ShallowPointNet) and isinstance(mini, MiniPointNet)
    for model in (shallow, mini):
        model.eval()
        logits = model(x)
        assert logits.shape == (4, 10)
        assert torch.allclose(model(x[:, torch.randperm(NUM_POINTS)]), logits, atol=1e-5)
    assert parameter_count(shallow) == 3 * 64 + 64 + 64 * 10 + 10
    widths = [m.out_channels for m in mini.point_mlp if isinstance(m, torch.nn.Conv1d)]
    assert widths == [64, 128, 256]
    linears = [(m.in_features, m.out_features) for m in mini.head if isinstance(m, torch.nn.Linear)]
    assert linears == [(256, 128), (128, 10)]
    for name in ACTIVATIONS:
        assert build_model("mini_pointnet", hidden_dim=16, activation=name)(x).shape == (4, 10)
    with pytest.raises(ValueError):
        build_model("mini_pointnet", activation="swishy")


def test_backpropagation_fills_every_gradient_and_the_optimizer_changes_weights():
    model = build_model("mini_pointnet", hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x, y = torch.randn(8, 256, 3), torch.randint(0, 10, (8,))
    before = [p.detach().clone() for p in model.parameters()]
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert all(not torch.equal(b, p) for b, p in zip(before, model.parameters()))


# ----------------------------------------------------------------------------- training

@pytest.fixture
def fixture_data(tmp_path):
    write_fixture_dataset(tmp_path / "data")
    return tmp_path / "data"


def test_training_smoke_learns_saves_and_reloads(fixture_data, tmp_path):
    data = dataset.load(fixture_data, download=False)
    assert data.summary() == {"train": 50, "val": 10, "test": 20}
    rows = []
    config = TrainConfig(model="mini_pointnet", epochs=6, learning_rate=5e-3, batch_size=10, hidden_dim=16, device="cpu")
    result = train(config, data, tmp_path / "run", progress=rows.append)
    assert [r["epoch"] for r in rows] == list(range(1, 7))
    assert rows[-1]["train_loss"] < rows[0]["train_loss"]
    assert all(r["grad_norm"] > 0 for r in rows)
    metrics = result["metrics"]
    assert 1 <= metrics["best_epoch"] <= 6
    assert np.array(metrics["confusion_matrix"]).sum() == 20
    assert metrics["test_error"] == pytest.approx(1 - metrics["test_accuracy"])
    for name in ("best.pt", "history.json", "metrics.json", "config.json"):
        assert (tmp_path / "run" / name).exists()
    model, checkpoint = load_checkpoint(tmp_path / "run" / "best.pt")
    assert not model.training and checkpoint["classes"] == list(dataset.CLASSES)
    x = torch.from_numpy(data.test_points).float()
    result["model"].eval()
    assert torch.allclose(model(x), result["model"](x), atol=1e-5)


def test_missing_dataset_without_download_is_a_clear_error(tmp_path):
    with pytest.raises(dataset.DatasetError, match="prepare"):
        dataset.load(tmp_path / "empty", download=False)


# ----------------------------------------------------------------------------- API

@pytest.fixture
def settings(tmp_path, fixture_data):
    write_fixture_models(tmp_path / "models")
    write_fixture_samples(tmp_path / "samples")
    return ClassificationSettings(models_dir=tmp_path / "models", samples_dir=tmp_path / "samples",
                                  runs_dir=tmp_path / "runs", data_dir=fixture_data, allow_download=False)


@pytest.fixture
def client(tmp_path, settings):
    with TestClient(create_app(tmp_path / "db.sqlite3", settings)) as instance:
        yield instance


def wait_for(client, run_id, states=("completed", "failed", "cancelled"), timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = client.get(f"{API}/runs/{run_id}").json()
        if run["status"] in states:
            return run
        time.sleep(0.1)
    raise AssertionError(f"run {run_id} still {run['status']}")


def test_models_samples_and_sample_prediction(client):
    listing = client.get(f"{API}/models").json()
    assert listing["classes"] == list(dataset.CLASSES)
    assert [m["id"] for m in listing["models"]] == ["mini_pointnet", "shallow"]
    samples = client.get(f"{API}/samples").json()["samples"]
    assert [s["class"] for s in samples] == ["bed", "chair"]
    ply = client.get(f"{API}/samples/{samples[0]['id']}/ply")
    assert ply.status_code == 200 and ply.content.startswith(b"ply\n")
    for model_id in ("mini_pointnet", "shallow"):
        result = client.post(f"{API}/predict/sample", json={"sample_id": samples[0]["id"], "model_id": model_id}).json()
        assert result["predicted_class"] in dataset.CLASSES
        assert len(result["top_k"]) == 3
        assert result["top_k"][0]["probability"] == pytest.approx(result["confidence"])
        assert sum(result["probabilities"].values()) == pytest.approx(1, abs=1e-5)
        assert result["model"]["id"] == model_id
        assert result["input"]["input_points"] == NUM_POINTS
        assert result["inference_ms"] >= 0 and result["sample"]["class"] == "bed"
    assert client.post(f"{API}/predict/sample", json={"sample_id": "nope"}).status_code == 404
    assert client.post(f"{API}/predict/sample", json={"sample_id": samples[0]["id"], "model_id": "nope"}).status_code == 404
    assert client.get(f"{API}/samples/nope/ply").status_code == 404


def test_upload_prediction_and_invalid_uploads(client, monkeypatch):
    cube = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    mesh = ascii_ply(cube, [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
    response = client.post(f"{API}/predict/upload?filename=cube.ply&top_k=5", content=mesh,
                           headers={"Content-Type": "application/octet-stream"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["input"] == {"input_points": 8, "input_faces": 12, "model_points": NUM_POINTS,
                               "sampling": "surface sampling of the mesh", "format": "PLY · ASCII"}
    assert len(result["top_k"]) == 5 and result["filename"] == "cube.ply"
    cloud = binary_ply(np.random.default_rng(0).normal(size=(3000, 3)).tolist(), [], "<")
    result = client.post(f"{API}/predict/upload?filename=c.ply&model_id=shallow", content=cloud).json()
    assert result["input"]["sampling"] == "subsampled from the point cloud" and result["model"]["id"] == "shallow"

    assert client.post(f"{API}/predict/upload?filename=x.obj", content=mesh).status_code == 415
    assert client.post(f"{API}/predict/upload?filename=x.ply", content=b"").status_code == 422
    bad = client.post(f"{API}/predict/upload?filename=x.ply", content=b"not a ply at all")
    assert bad.status_code == 422 and "Not a PLY" in bad.json()["detail"]
    assert client.post(f"{API}/predict/upload?filename=x.ply&model_id=nope", content=mesh).status_code == 404
    monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 100)
    assert client.post(f"{API}/predict/upload?filename=x.ply", content=mesh).status_code == 413


def test_training_run_lifecycle_history_results_and_new_model(client, tmp_path):
    config = {"model": "mini_pointnet", "epochs": 3, "learning_rate": 0.005, "hidden_dim": 16,
              "activation": "gelu", "batch_size": 10}
    response = client.post(f"{API}/runs", json=config)
    assert response.status_code == 202
    run_id = response.json()["id"]
    assert response.json()["status"] in ("queued", "running")
    run = wait_for(client, run_id)
    assert run["status"] == "completed", run
    assert run["epoch"] == run["total_epochs"] == 3
    assert [row["epoch"] for row in run["history"]] == [1, 2, 3]
    for key in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"):
        assert all(np.isfinite(row[key]) for row in run["history"])
    assert 0 <= run["metrics"]["test_accuracy"] <= 1 and run["config"]["activation"] == "gelu"
    assert run["metrics"]["split_sizes"]["test"] == 20
    assert (tmp_path / "runs" / run_id / "best.pt").exists()
    assert any(r["id"] == run_id for r in client.get(f"{API}/runs").json()["runs"])
    models = [m["id"] for m in client.get(f"{API}/models").json()["models"]]
    assert f"run-{run_id}" in models
    sample = client.get(f"{API}/samples").json()["samples"][0]["id"]
    predicted = client.post(f"{API}/predict/sample", json={"sample_id": sample, "model_id": f"run-{run_id}"})
    assert predicted.status_code == 200 and predicted.json()["model"]["config"]["activation"] == "gelu"


def test_training_run_cancel_and_queue(client):
    slow = {"model": "mini_pointnet", "epochs": 200, "hidden_dim": 16, "batch_size": 4}
    first = client.post(f"{API}/runs", json=slow).json()["id"]
    second = client.post(f"{API}/runs", json=slow).json()["id"]
    assert client.get(f"{API}/runs/{second}").json()["status"] == "queued"
    assert client.post(f"{API}/runs/{second}/cancel").json()["status"] == "cancelled"
    wait_for(client, first, states=("running",))
    client.post(f"{API}/runs/{first}/cancel")
    assert wait_for(client, first)["status"] == "cancelled"
    assert client.get(f"{API}/runs/{second}").json()["status"] == "cancelled"
    assert client.post(f"{API}/runs/missing/cancel").status_code == 404
    assert client.get(f"{API}/runs/missing").status_code == 404


@pytest.mark.parametrize("change", [
    {"epochs": 0}, {"epochs": 1000}, {"learning_rate": 0}, {"learning_rate": 5}, {"hidden_dim": 2},
    {"activation": "swish"}, {"model": "pointnet++"}, {"batch_size": 1}, {"epochs": "5"}, {"unexpected": 1},
])
def test_training_config_is_validated(client, change):
    response = client.post(f"{API}/runs", json={"epochs": 1, **change})
    assert response.status_code == 422
    assert client.get(f"{API}/runs").json()["runs"] == []


def test_failed_run_reports_the_error_and_runs_survive_restart(tmp_path, settings):
    settings.data_dir = tmp_path / "missing"
    with TestClient(create_app(tmp_path / "db.sqlite3", settings)) as client:
        run = wait_for(client, client.post(f"{API}/runs", json={"epochs": 1}).json()["id"])
        assert run["status"] == "failed" and "DatasetError" in run["error"] and "prepare" in run["error"]
    with TestClient(create_app(tmp_path / "db.sqlite3", settings)) as client:
        assert client.get(f"{API}/runs/{run['id']}").json()["status"] == "failed"


def test_dataset_status_and_results_endpoints(client):
    status = client.get(f"{API}/dataset").json()
    assert status["prepared"] is True and status["counts"]["test"] == 20
    results = client.get(f"{API}/results").json()
    assert set(results["models"]) == {"shallow", "mini_pointnet"}


# ----------------------------------------------------------------------------- bundled artifacts

BUNDLED = BACKEND / "models" / "modelnet10"


@pytest.mark.skipif(not (BUNDLED / "mini_pointnet" / "best.pt").exists(), reason="bundled checkpoints not trained yet")
def test_bundled_checkpoints_load_and_classify_bundled_samples():
    registry = ModelRegistry(BUNDLED, BACKEND / "data" / "no-runs")
    samples = json.loads((BACKEND / "samples" / "modelnet10" / "index.json").read_text())["samples"]
    assert len(samples) >= 10
    comparison = json.loads((BUNDLED / "comparison.json").read_text())
    assert comparison["final_beats_baseline"] is True
    for model_id in ("shallow", "mini_pointnet"):
        model, checkpoint = registry.get(model_id)
        assert not model.training and checkpoint["model"] == model_id
        for entry in samples:
            points, _ = preprocess(parse_ply((BACKEND / "samples" / "modelnet10" / entry["file"]).read_bytes()))
            result = registry.predict(model_id, points)
            assert result["predicted_class"] == entry["predictions"][model_id]
