"""Small SQLAlchemy layer; SQLite files are created at application startup only."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Float, String, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Scene(Base):
    """Only small validated metadata. PLY bytes never enter the API or DB."""
    __tablename__ = "scenes"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    details: Mapped[dict] = mapped_column(JSON)


class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    scene_id: Mapped[str] = mapped_column(String(80), index=True)
    category: Mapped[str] = mapped_column(String(40))
    label: Mapped[str] = mapped_column(String(80))
    kind: Mapped[str] = mapped_column(String(12), default="point", server_default="point")
    region: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    note: Mapped[str] = mapped_column(String(1000), default="")
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    z: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def open_database(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False, "timeout": 15})
    Base.metadata.create_all(engine)
    # Additive, idempotent upgrade of existing point-only SQLite databases.
    # No table rebuild, row rewrite, deletion, or external migration framework.
    with engine.begin() as connection:
        columns = {column["name"] for column in inspect(connection).get_columns("annotations")}
        if "kind" not in columns:
            connection.execute(text("ALTER TABLE annotations ADD COLUMN kind VARCHAR(12) NOT NULL DEFAULT 'point'"))
        if "region" not in columns:
            connection.execute(text("ALTER TABLE annotations ADD COLUMN region JSON"))
    return engine, sessionmaker(engine, expire_on_commit=False)
