import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .classification.routes import ClassificationSettings, create_router
from .database import Annotation, Scene, open_database, utcnow
from .schemas import AnnotationInput, AnnotationRead, SceneInput

SCENE = json.loads(Path(__file__).with_name("scene.json").read_text())
DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "annotations.sqlite3"
DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173"
logger = logging.getLogger(__name__)


def create_app(database_path: str | Path | None = None, classification: ClassificationSettings | None = None) -> FastAPI:
    db_path = Path(database_path or os.environ.get("DATABASE_PATH", DEFAULT_DB)).expanduser().resolve()
    classification_router, training = create_router(classification or ClassificationSettings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine, factory = open_database(db_path)
        app.state.session_factory = factory
        try:
            yield
        finally:
            training.shutdown()
            engine.dispose()

    app = FastAPI(title="Neural 3D Data Viewer", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[s.strip() for s in os.environ.get("CORS_ORIGINS", DEFAULT_ORIGINS).split(",") if s.strip()],
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type"],
    )

    def session(request: Request):
        with request.app.state.session_factory() as db:
            yield db

    Database = Annotated[Session, Depends(session)]

    def require_scene(db: Session, scene_id: str):
        if scene_id == SCENE["id"]:
            return SCENE
        row = db.get(Scene, scene_id)
        if row is None:
            raise HTTPException(404, "Scene not found")
        return {"id": row.id, **row.details}

    def require_annotation(db: Session, scene_id: str, annotation_id: UUID):
        require_scene(db, scene_id)
        row = db.get(Annotation, str(annotation_id))
        if row is None or row.scene_id != scene_id:
            raise HTTPException(404, "Annotation not found in this scene")
        return row

    @app.exception_handler(SQLAlchemyError)
    async def database_error(_request: Request, exc: SQLAlchemyError):
        logger.error("Database operation failed: %s", type(exc).__name__)
        return JSONResponse(status_code=503, content={"detail": "Database unavailable. Please retry; your form has not been cleared."})

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        # Do not echo raw inputs: JSON numbers such as 1e999 become infinity in
        # Python, which cannot itself be encoded in a standards-compliant response.
        details = [{"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
                   for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": details})

    @app.get("/api/health")
    def health(db: Database):
        db.execute(text("SELECT 1"))
        return {"status": "ok", "database": "sqlite"}

    @app.get("/api/scenes")
    def scenes(db: Database):
        return [SCENE, *[{"id": row.id, **row.details} for row in db.scalars(select(Scene).order_by(Scene.id))]]

    @app.get("/api/scenes/{scene_id}")
    def scene(scene_id: str, db: Database):
        return require_scene(db, scene_id)

    @app.put("/api/scenes/{scene_id}")
    def register_scene(scene_id: str, payload: SceneInput, db: Database):
        if scene_id != f"ply-{payload.sha256}":
            raise HTTPException(422, "Scene ID must match the PLY SHA-256 fingerprint")
        details = payload.model_dump()
        row = db.get(Scene, scene_id)
        if row:
            # Older PLYLoader imports with face colors/UVs could register an
            # expanded triangle count. Refresh that derived count once, without
            # changing the file identity or any annotation rows.
            legacy_mesh = "face_count" not in row.details and payload.face_count > 0
            identity_conflict = any(row.details[key] != details[key] for key in ("sha256", "format", "size_bytes"))
            count_conflict = row.details["point_count"] != details["point_count"] and not legacy_mesh
            if identity_conflict or count_conflict:
                raise HTTPException(409, "Content metadata conflicts with this scene fingerprint")
            # Renaming the same file does not change its identity or annotations.
            row.details = details
        else:
            db.add(Scene(id=scene_id, details=details))
        db.commit()
        return {"id": scene_id, **details}

    @app.get("/api/scenes/{scene_id}/annotations", response_model=list[AnnotationRead])
    def list_annotations(scene_id: str, db: Database):
        require_scene(db, scene_id)
        return db.scalars(select(Annotation).where(Annotation.scene_id == scene_id)
                          .order_by(Annotation.created_at, Annotation.id)).all()

    def validate_region(scene_id: str, payload: AnnotationInput, db: Session):
        metadata = require_scene(db, scene_id)
        if payload.region and payload.region.vertex_count != metadata["point_count"]:
            raise HTTPException(422, "Region vertex count must match this scene")

    @app.post("/api/scenes/{scene_id}/annotations", response_model=AnnotationRead, status_code=201)
    def create_annotation(scene_id: str, payload: AnnotationInput, db: Database):
        validate_region(scene_id, payload, db)
        row = Annotation(scene_id=scene_id, **payload.model_dump())
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    @app.get("/api/scenes/{scene_id}/annotations/{annotation_id}", response_model=AnnotationRead)
    def read_annotation(scene_id: str, annotation_id: UUID, db: Database):
        return require_annotation(db, scene_id, annotation_id)

    @app.put("/api/scenes/{scene_id}/annotations/{annotation_id}", response_model=AnnotationRead)
    def update_annotation(scene_id: str, annotation_id: UUID, payload: AnnotationInput, db: Database):
        row = require_annotation(db, scene_id, annotation_id)
        validate_region(scene_id, payload, db)
        for key, value in payload.model_dump().items():
            setattr(row, key, value)
        row.updated_at = utcnow()
        db.commit()
        db.refresh(row)
        return row

    @app.delete("/api/scenes/{scene_id}/annotations/{annotation_id}", status_code=204)
    def delete_annotation(scene_id: str, annotation_id: UUID, db: Database):
        row = require_annotation(db, scene_id, annotation_id)
        db.delete(row)
        db.commit()
        return Response(status_code=204)

    app.include_router(classification_router)
    return app


app = create_app()
