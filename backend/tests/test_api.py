import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import SCENE, create_app

BASE = f"/api/scenes/{SCENE['id']}/annotations"
PAYLOAD = {"category": "Structure", "label": "Torus rim", "note": "Synthetic landmark", "x": -1.2, "y": 1.4, "z": 0.1}


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "test.sqlite3")) as instance:
        yield instance


def test_health_scenes_and_cors(client):
    assert client.get("/api/health").json() == {"status": "ok", "database": "sqlite"}
    assert client.get("/api/scenes").json() == [SCENE]
    assert client.get(f"/api/scenes/{SCENE['id']}").json()["point_count"] > 0
    response = client.options(BASE, headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-origin" not in client.get("/api/health", headers={"Origin": "https://untrusted.example"}).headers


def test_crud_and_persistence_across_restart(tmp_path):
    db_path = tmp_path / "persistent.sqlite3"
    with TestClient(create_app(db_path)) as client:
        assert client.get(BASE).json() == []
        response = client.post(BASE, json=PAYLOAD)
        assert response.status_code == 201
        original = response.json()
        endpoint = f"{BASE}/{original['id']}"
        assert original["scene_id"] == SCENE["id"]
        assert original["created_at"].endswith("Z")
        assert client.get(endpoint).json() == original
    with TestClient(create_app(db_path)) as client:
        assert client.get(BASE).json() == [original]
        edited = {**PAYLOAD, "label": "Edited rim", "note": "", "x": 0.25}
        response = client.put(endpoint, json=edited)
        assert response.status_code == 200
        assert response.json()["label"] == "Edited rim"
        assert response.json()["created_at"] == original["created_at"]
    with TestClient(create_app(db_path)) as client:
        assert client.get(endpoint).json()["x"] == 0.25
        assert client.delete(endpoint).status_code == 204
        assert client.get(endpoint).status_code == 404
    with TestClient(create_app(db_path)) as client:
        assert client.get(BASE).json() == []


@pytest.mark.parametrize("change", [
    {"label": "   "}, {"category": ""}, {"label": "a" * 81}, {"category": "a" * 41},
    {"note": "a" * 1001}, {"x": "NaN"}, {"y": "Infinity"}, {"z": "-Infinity"},
    {"x": None}, {"x": True}, {"x": "0.5"}, {"unexpected": True},
])
def test_rejects_invalid_input_without_mutation(client, change):
    response = client.post(BASE, json={**PAYLOAD, **change})
    assert response.status_code == 422
    assert response.json()["detail"]
    assert client.get(BASE).json() == []


def test_update_validation_and_whitespace(client):
    row = client.post(BASE, json={**PAYLOAD, "label": "  Rim  "}).json()
    assert row["label"] == "Rim"
    endpoint = f"{BASE}/{row['id']}"
    assert client.put(endpoint, json={**PAYLOAD, "label": " "}).status_code == 422
    assert client.get(endpoint).json()["label"] == "Rim"


@pytest.mark.parametrize("number", ["1e999", "NaN"])
def test_nonfinite_json_produces_useful_validation_error(client, number):
    body = json.dumps(PAYLOAD).replace("-1.2", number)
    response = client.post(BASE, content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "x"]
    assert client.get(BASE).json() == []


def test_missing_resources_and_uuid_validation(client):
    missing = f"{BASE}/{uuid4()}"
    assert client.get(missing).status_code == 404
    assert client.put(missing, json=PAYLOAD).status_code == 404
    assert client.delete(missing).status_code == 404
    assert client.get(f"{BASE}/not-a-uuid").status_code == 422
    assert client.get("/api/scenes/missing").status_code == 404
    assert client.get("/api/scenes/missing/annotations").status_code == 404
    assert client.post("/api/scenes/missing/annotations", json=PAYLOAD).status_code == 404


def local_scene(fingerprint="a" * 64):
    return {"sha256": fingerprint, "name": "Local demo", "filename": "demo.ply",
            "format": "PLY · ASCII", "point_count": 5, "size_bytes": 240,
            "coordinates": "Native coordinates; units unspecified"}


def test_local_scenes_are_isolated_and_reopen_across_restart(tmp_path):
    path = tmp_path / "scenes.sqlite3"
    first, second = local_scene(), local_scene("b" * 64)
    scene_a, scene_b = f"/api/scenes/ply-{first['sha256']}", f"/api/scenes/ply-{second['sha256']}"
    with TestClient(create_app(path)) as client:
        assert client.put(scene_a, json=first).status_code == 200
        assert client.put(scene_b, json=second).status_code == 200
        row = client.post(f"{scene_a}/annotations", json=PAYLOAD).json()
        assert client.get(f"{scene_b}/annotations").json() == []
        assert client.get(f"{scene_b}/annotations/{row['id']}").status_code == 404
        assert client.put(f"{scene_b}/annotations/{row['id']}", json=PAYLOAD).status_code == 404
        assert client.delete(f"{scene_b}/annotations/{row['id']}").status_code == 404
        assert client.get(BASE).json() == []
    with TestClient(create_app(path)) as client:
        assert client.put(scene_a, json={**first, "filename": "renamed.ply"}).status_code == 200
        assert client.get(f"{scene_a}/annotations").json() == [row]
        assert client.get(scene_a).json()["filename"] == "renamed.ply"
        assert len(client.get("/api/scenes").json()) == 3
        assert client.put(scene_a, json={**first, "point_count": 6}).status_code == 409
        assert client.get(f"{scene_a}/annotations").json() == [row]


@pytest.mark.parametrize("change", [
    {"ply_bytes": "secret local bytes"}, {"filename": "../private.ply"},
    {"filename": "bad.txt"}, {"point_count": 0}, {"size_bytes": -1}, {"face_count": -1}, {"color_source": "face"},
    {"sha256": "invalid"}, {"format": "OBJ"}, {"name": ""},
])
def test_metadata_registration_rejects_invalid_or_file_payloads(client, change):
    assert client.put(f"/api/scenes/ply-{'a' * 64}", json={**local_scene(), **change}).status_code == 422
    assert client.get("/api/scenes").json() == [SCENE]


def test_additive_schema_preserves_legacy_annotations(tmp_path):
    from sqlalchemy import create_engine, inspect
    from app.database import Annotation

    path = tmp_path / "legacy.sqlite3"
    engine = create_engine(f"sqlite:///{path}")
    # This is the original release schema, without a scenes table.
    Annotation.__table__.create(engine)
    with engine.begin() as connection:
        connection.execute(Annotation.__table__.insert().values(scene_id="spectrum-garden", **PAYLOAD))
    assert inspect(engine).get_table_names() == ["annotations"]
    engine.dispose()
    with TestClient(create_app(path)) as client:
        records = client.get(BASE).json()
        assert len(records) == 1 and records[0]["label"] == PAYLOAD["label"]
        assert client.put(f"/api/scenes/ply-{'a' * 64}", json=local_scene()).status_code == 200
        assert client.get(BASE).json() == records


def test_scene_id_cannot_spoof_sample_or_another_hash(client):
    assert client.put("/api/scenes/spectrum-garden", json=local_scene()).status_code == 422
    assert client.put(f"/api/scenes/ply-{'b' * 64}", json=local_scene()).status_code == 422


def test_large_scene_metadata_and_finite_coordinates_have_no_default_ceiling(client):
    metadata = {**local_scene(), "format": "PLY · binary LE", "point_count": 25_000_001,
                "size_bytes": 1024 ** 3 + 1, "face_count": 900_000_000, "color_source": "fallback"}
    endpoint = f"/api/scenes/ply-{metadata['sha256']}"
    assert client.put(endpoint, json=metadata).status_code == 200
    assert client.get(endpoint).json()["face_count"] == 900_000_000
    annotation = {**PAYLOAD, "x": 1_000_001.0}
    assert client.post(f"{endpoint}/annotations", json=annotation).status_code == 201
    assert client.get(f"{endpoint}/annotations").json()[0]["x"] == 1_000_001.0


def test_legacy_expanded_mesh_count_refresh_preserves_annotations(tmp_path):
    from sqlalchemy.orm import Session
    from app.database import Scene, open_database

    path = tmp_path / "expanded-mesh.sqlite3"
    legacy = {**local_scene(), "format": "PLY · binary LE", "point_count": 24}
    scene_id = f"ply-{legacy['sha256']}"
    engine, _ = open_database(path)
    with Session(engine) as db:
        db.add(Scene(id=scene_id, details=legacy))  # metadata predates face_count
        db.commit()
    engine.dispose()
    endpoint = f"/api/scenes/{scene_id}"
    with TestClient(create_app(path)) as client:
        record = client.post(f"{endpoint}/annotations", json=PAYLOAD).json()
        corrected = {**legacy, "point_count": 5, "face_count": 8, "color_source": "vertex"}
        assert client.put(endpoint, json={**corrected, "size_bytes": 999}).status_code == 409
        assert client.put(endpoint, json=corrected).status_code == 200
        assert client.get(endpoint).json()["point_count"] == 5
        assert client.get(f"{endpoint}/annotations").json() == [record]
        assert client.put(endpoint, json={**corrected, "point_count": 6}).status_code == 409
    with TestClient(create_app(path)) as client:
        assert client.get(f"{endpoint}/annotations").json() == [record]
