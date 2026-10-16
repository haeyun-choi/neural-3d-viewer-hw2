"""Point sampling and normalisation shared by training, evaluation and inference.

Every point cloud the networks see — training, validation, test, a bundled
sample or an uploaded PLY — goes through ``points_from_geometry``, so the web
application applies exactly the preprocessing the checkpoints were trained on.
"""

from __future__ import annotations

import numpy as np

NUM_POINTS = 1024
PREPROCESSING = {
    "num_points": NUM_POINTS,
    "sampling": "area-weighted surface sampling of triangle meshes; vertex subsampling of point clouds",
    "normalization": "subtract the centroid of the sampled points, then divide by the largest distance to it (unit sphere)",
}


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def sample_surface(vertices: np.ndarray, faces: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly sample ``count`` points on a triangle mesh surface.

    A triangle is chosen with probability proportional to its area, then a point
    is drawn uniformly inside it with the square-root barycentric construction.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    areas = triangle_areas(vertices, faces)
    total = areas.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("The mesh has no surface area to sample.")
    chosen = rng.choice(len(faces), size=count, p=areas / total)
    r1 = np.sqrt(rng.random(count))[:, None]
    r2 = rng.random(count)[:, None]
    a, b, c = (vertices[faces[chosen, i]] for i in range(3))
    return (1 - r1) * a + r1 * (1 - r2) * b + r1 * r2 * c


def subsample_points(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Pick ``count`` points; without replacement when there are enough."""
    points = np.asarray(points, dtype=np.float64)
    index = rng.choice(len(points), size=count, replace=len(points) < count)
    return points[index]


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Centre on the centroid and scale into the unit sphere."""
    points = np.asarray(points, dtype=np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(centered, axis=1).max()
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("All points coincide; the object has no extent to normalise.")
    return (centered / radius).astype(np.float32)


def points_from_geometry(vertices: np.ndarray, faces: np.ndarray | None, rng: np.random.Generator,
                         count: int = NUM_POINTS) -> tuple[np.ndarray, str]:
    """The single preprocessing entry point: sample, then normalise.

    A mesh with positive surface area is sampled on its surface, exactly as the
    ModelNet10 training meshes are. Anything else is treated as a point cloud
    and its vertices are subsampled.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("Expected at least one 3D point.")
    if not np.isfinite(vertices).all():
        raise ValueError("Point coordinates must be finite numbers.")
    if faces is not None and len(faces):
        faces = np.asarray(faces, dtype=np.int64)
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("A face refers to a vertex that does not exist.")
        if triangle_areas(vertices, faces).sum() > 0:
            return normalize_points(sample_surface(vertices, faces, count, rng)), "surface"
    return normalize_points(subsample_points(vertices, count, rng)), "vertices"
