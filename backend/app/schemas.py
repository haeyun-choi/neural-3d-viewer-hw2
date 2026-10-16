from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Coordinate = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class SceneInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    name: str = Field(min_length=1, max_length=255)
    filename: str = Field(min_length=1, max_length=255)
    format: Literal["PLY · ASCII", "PLY · binary LE", "PLY · binary BE"]
    point_count: int = Field(strict=True, ge=1)
    size_bytes: int = Field(strict=True, ge=1)
    face_count: int = Field(default=0, strict=True, ge=0)
    color_source: Literal["vertex", "fallback"] | None = None
    coordinates: str = Field(min_length=1, max_length=500)

    @field_validator("filename")
    @classmethod
    def basename_only(cls, value: str) -> str:
        if "/" in value or "\\" in value or any(ord(c) < 32 for c in value) or not value.lower().endswith(".ply"):
            raise ValueError("Use a PLY filename without directory components")
        return value


UnitCoordinate = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0, le=1)]


class SelectionOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["replace", "add", "subtract"]
    shape: Literal["rectangle", "lasso"]
    polygon: list[tuple[UnitCoordinate, UnitCoordinate]] = Field(min_length=3, max_length=512)
    matrix: list[Coordinate] = Field(min_length=16, max_length=16)

    @model_validator(mode="after")
    def rectangle_corners(self):
        if self.shape == "rectangle" and len(self.polygon) != 4:
            raise ValueError("Rectangle requires four corners")
        return self


class RegionData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1]
    semantics: Literal["through"]
    vertex_count: int = Field(strict=True, ge=1)
    selected_count: int = Field(strict=True, ge=1)
    operations: list[SelectionOperation] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def reproducible_selection(self):
        if self.selected_count > self.vertex_count:
            raise ValueError("Selected count exceeds the scene vertex count")
        if self.operations[0].mode != "replace":
            raise ValueError("Selection history must begin with replace")
        return self


class AnnotationInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    kind: Literal["point", "region"] = "point"
    region: RegionData | None = None
    category: str = Field(min_length=1, max_length=40)
    label: str = Field(min_length=1, max_length=80)
    note: str = Field(default="", max_length=1000)
    x: Coordinate
    y: Coordinate
    z: Coordinate

    @model_validator(mode="after")
    def annotation_geometry(self):
        if (self.kind == "region") != (self.region is not None):
            raise ValueError("Region annotations require selection data; point annotations cannot contain it")
        return self


class AnnotationRead(AnnotationInput):
    model_config = ConfigDict(from_attributes=True)

    id: str
    scene_id: str
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at")
    @classmethod
    def explicit_utc(cls, value: datetime) -> datetime:
        # SQLite drops timezone metadata. All writes in this app are UTC.
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
