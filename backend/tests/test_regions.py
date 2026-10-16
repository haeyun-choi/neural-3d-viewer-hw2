"""Region records are compact, validated, scene-isolated, and migration-safe."""
import copy
import json
import sqlite3
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import SCENE, create_app

BASE = f"/api/scenes/{SCENE['id']}/annotations"
OP = {"mode": "replace", "shape": "rectangle", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
      "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]}
REGION = {"version": 1, "semantics": "through", "vertex_count": SCENE["point_count"], "selected_count": 6000, "operations": [OP]}
PAYLOAD = {"kind": "region", "region": REGION, "category": "Part", "label": "Pedestal", "note": "One region", "x": 0, "y": 0, "z": 0}


def test_region_crud_persistence_and_scene_isolation(tmp_path):
    path = tmp_path / "regions.sqlite3"
    with TestClient(create_app(path)) as client:
        response = client.post(BASE, json=PAYLOAD)
        assert response.status_code == 201
        row = response.json()
        assert len(client.get(BASE).json()) == 1
        assert row["region"] == REGION
        assert len(json.dumps(row)) < 1000  # 6000 points, only operations in JSON
    with TestClient(create_app(path)) as client:
        endpoint = f"{BASE}/{row['id']}"
        assert client.get(endpoint).json()["region"] == REGION
        assert client.get(f"/api/scenes/ply-{'b' * 64}/annotations/{row['id']}").status_code == 404
        changed = client.put(endpoint, json={**PAYLOAD, "category": "Wing", "label": "Changed"})
        assert changed.status_code == 200
        assert changed.json()["region"] == REGION
        assert client.delete(endpoint).status_code == 204
        assert client.get(BASE).json() == []


@pytest.mark.parametrize("case", ["empty", "too_many", "layout", "nan", "matrix", "polygon", "history", "first_add", "visible", "xyz_array", "point_region", "missing", "kind", "string_count"])
def test_invalid_region_does_not_mutate_database(tmp_path, case):
    payload = copy.deepcopy(PAYLOAD)
    region = payload["region"]
    if case == "empty": region["selected_count"] = 0
    if case == "too_many": region["selected_count"] = SCENE["point_count"] + 1
    if case == "layout": region["vertex_count"] += 1
    if case == "nan": region["operations"][0]["matrix"][0] = "NaN"
    if case == "matrix": region["operations"][0]["matrix"].pop()
    if case == "polygon": region["operations"][0]["polygon"][0] = [-1, 0]
    if case == "history": region["operations"] *= 65
    if case == "first_add": region["operations"][0]["mode"] = "add"
    if case == "visible": region["semantics"] = "visible-only"
    if case == "xyz_array": region["positions"] = [[0, 0, 0]]
    if case == "point_region": payload["kind"] = "point"
    if case == "missing": payload["region"] = None
    if case == "kind": payload["kind"] = "mesh"
    if case == "string_count": region["selected_count"] = "6000"
    with TestClient(create_app(tmp_path / "invalid.sqlite3")) as client:
        assert client.post(BASE, json=payload).status_code == 422
        assert client.get(BASE).json() == []


def test_additive_legacy_sqlite_migration_preserves_point_row(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    row_id = str(uuid4())
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE annotations (id VARCHAR(36) PRIMARY KEY, scene_id VARCHAR(80), category VARCHAR(40), label VARCHAR(80), note VARCHAR(1000), x FLOAT, y FLOAT, z FLOAT, created_at DATETIME, updated_at DATETIME)")
        db.execute("INSERT INTO annotations VALUES (?, ?, 'Landmark', 'Old point', 'Keep me', 1, 2, 3, '2026-09-01 12:00:00', '2026-09-01 12:00:00')", (row_id, SCENE["id"]))
    for _ in range(2):
        with TestClient(create_app(path)) as client:
            rows = client.get(BASE).json()
            assert len(rows) == 1
            assert rows[0]["id"] == row_id and rows[0]["note"] == "Keep me"
            assert rows[0]["kind"] == "point" and rows[0]["region"] is None
            assert rows[0]["x"] == 1 and rows[0]["y"] == 2 and rows[0]["z"] == 3
