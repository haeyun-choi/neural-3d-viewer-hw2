// Training-camera trajectories: parsing, coordinate conversion and spatial anchors.
//
// This module is renderer-independent on purpose. It turns one input format
// into a canonical list of poses; visualization lives in CameraTrajectoryView.ts
// and exploration state lives in useTrajectory.ts, so a second input format later
// only has to produce TrajectoryCamera[]. Movement lives in trajectoryTube.ts
// and FreeLookController.ts; neither of them parses anything.
//
// ---------------------------------------------------------------------------
// Coordinate semantics, read out of the official Graphdeco implementation
// (github.com/graphdeco-inria/gaussian-splatting), not inferred from field names
// ---------------------------------------------------------------------------
//
// utils/camera_utils.py::camera_to_JSON
//
//     Rt = np.zeros((4, 4))
//     Rt[:3, :3] = camera.R.transpose()
//     Rt[:3, 3]  = camera.T
//     Rt[3, 3]   = 1.0
//     W2C = np.linalg.inv(Rt)          # <- misleading name; see below
//     pos = W2C[:3, 3]
//     rot = W2C[:3, :3]
//
// 1. `camera.R` is the camera-to-world rotation and `camera.T` the
//    world-to-camera translation: scene/dataset_readers.py builds them as
//    `R = np.transpose(qvec2rotmat(extr.qvec))` and `T = np.array(extr.tvec)`,
//    with the comment "R is stored transposed due to 'glm' in CUDA code".
//    So `Rt` above is the world-to-camera matrix.
// 2. utils/graphics_utils.py::getWorld2View2 performs the *identical*
//    construction and names `np.linalg.inv(Rt)` **`C2W`**. The variable called
//    `W2C` inside camera_to_JSON is therefore the camera-to-world matrix, which
//    makes `position` the camera centre in world space and `rotation` the
//    camera-to-world rotation with the camera axes as its **columns**.
// 3. Those axes are COLMAP/OpenCV: +x right, +y **down**, +z **forward**.
//    scene/dataset_readers.py states the relationship to OpenGL explicitly when
//    it reads NeRF transforms: "change from OpenGL/Blender camera axes
//    (Y up, Z back) to COLMAP (Y down, Z forward)" followed by
//    `c2w[:3, 1:3] *= -1`. Three.js uses the OpenGL convention, so this module
//    applies exactly the inverse of that line: negate columns 1 and 2.
// 4. utils/graphics_utils.py::focal2fov is `2 * atan(pixels / (2 * focal))`, so
//    the vertical field of view is `focal2fov(fy, height)`.
//
// No sign is chosen by looking at a render. tests/camera-trajectory.spec.ts
// pins position, forward, up and right numerically, including an asymmetric
// pose where a transpose, an inverse or a flipped sign cannot pass by accident.

import * as THREE from "three";
import type { QuaternionTuple, Vector3Tuple } from "./types";

/** One training camera, in this application's canonical form. */
export interface TrajectoryCamera {
  /** Stable identifier from the file. Not assumed unique; see `sourceIndex`. */
  id: number;
  imgName: string;
  /** Camera centre in world space, the same space as the splats. */
  position: Vector3Tuple;
  /** Orientation in the Three.js convention: -z forward, +y up. */
  quaternion: QuaternionTuple;
  width: number;
  height: number;
  fx: number;
  fy: number;
  /** Vertical field of view in degrees, from fy and height. */
  fovYDeg: number;
  /** Index in the source array. The stable internal key, whatever `id` says. */
  sourceIndex: number;
  /** Reserved for formats that carry capture time; null for cameras.json. */
  timestampSeconds: number | null;
}

export interface ParsedTrajectory {
  cameras: TrajectoryCamera[];
  /** Non-fatal observations worth telling the user about. */
  warnings: string[];
}

export class TrajectoryError extends Error {}

/** cameras.json is pose metadata; anything larger is not one. */
export const TRAJECTORY_MAX_BYTES = 64 * 1024 * 1024;
export const TRAJECTORY_ACCEPT = ".json";
/** A rotation matrix that is this far from orthonormal is not a rotation. */
const ROTATION_TOLERANCE = 1e-3;

const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

function readVector3(value: unknown, what: string, index: number): Vector3Tuple {
  if (!Array.isArray(value) || value.length !== 3 || !value.every(finite))
    throw new TrajectoryError(`Camera ${index}: ${what} must be three finite numbers.`);
  return [value[0], value[1], value[2]] as Vector3Tuple;
}

function readRotation(value: unknown, index: number): number[][] {
  if (!Array.isArray(value) || value.length !== 3)
    throw new TrajectoryError(`Camera ${index}: rotation must be a 3x3 matrix.`);
  const rows = value.map((row) => {
    if (!Array.isArray(row) || row.length !== 3 || !row.every(finite))
      throw new TrajectoryError(`Camera ${index}: rotation must be a 3x3 matrix of finite numbers.`);
    return [row[0], row[1], row[2]] as number[];
  });
  // Reject anything that is not a proper rotation. This is what catches a
  // matrix that was scaled, sheared, or written in some other convention,
  // before it turns into a silently wrong viewpoint.
  const column = (c: number) => [rows[0][c], rows[1][c], rows[2][c]];
  const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const [x, y, z] = [column(0), column(1), column(2)];
  const orthonormal =
    Math.abs(dot(x, x) - 1) < ROTATION_TOLERANCE &&
    Math.abs(dot(y, y) - 1) < ROTATION_TOLERANCE &&
    Math.abs(dot(z, z) - 1) < ROTATION_TOLERANCE &&
    Math.abs(dot(x, y)) < ROTATION_TOLERANCE &&
    Math.abs(dot(x, z)) < ROTATION_TOLERANCE &&
    Math.abs(dot(y, z)) < ROTATION_TOLERANCE;
  const determinant =
    rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1]) -
    rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0]) +
    rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0]);
  if (!orthonormal || Math.abs(determinant - 1) > ROTATION_TOLERANCE)
    throw new TrajectoryError(
      `Camera ${index}: rotation is not a right-handed rotation matrix (determinant ${determinant.toFixed(4)}).`,
    );
  return rows;
}

/**
 * Camera-to-world rotation in OpenCV axes (columns: right, down, forward) to a
 * Three.js camera orientation (-z forward, +y up). Negating columns 1 and 2 is
 * the inverse of Graphdeco's own OpenGL-to-COLMAP line, `c2w[:3, 1:3] *= -1`.
 */
export function openCvRotationToThreeQuaternion(rotation: number[][]): QuaternionTuple {
  const right = new THREE.Vector3(rotation[0][0], rotation[1][0], rotation[2][0]);
  const down = new THREE.Vector3(rotation[0][1], rotation[1][1], rotation[2][1]);
  const forward = new THREE.Vector3(rotation[0][2], rotation[1][2], rotation[2][2]);
  const basis = new THREE.Matrix4().makeBasis(right, down.negate(), forward.negate());
  const quaternion = new THREE.Quaternion().setFromRotationMatrix(basis).normalize();
  return [quaternion.x, quaternion.y, quaternion.z, quaternion.w];
}

/** Graphdeco's focal2fov(focal, pixels) = 2*atan(pixels / (2*focal)), in degrees. */
export function verticalFovDegrees(fy: number, height: number): number {
  return THREE.MathUtils.radToDeg(2 * Math.atan(height / (2 * fy)));
}

/** The world-space axes a Three.js camera with this orientation looks along. */
export function cameraAxes(quaternion: QuaternionTuple) {
  const q = new THREE.Quaternion(...quaternion);
  return {
    forward: new THREE.Vector3(0, 0, -1).applyQuaternion(q).toArray() as Vector3Tuple,
    up: new THREE.Vector3(0, 1, 0).applyQuaternion(q).toArray() as Vector3Tuple,
    right: new THREE.Vector3(1, 0, 0).applyQuaternion(q).toArray() as Vector3Tuple,
  };
}

/**
 * Parse an original INRIA/Graphdeco cameras.json. Every camera is validated
 * before any of them is returned, so a malformed file replaces nothing.
 */
export function parseGraphdecoCameras(text: string): ParsedTrajectory {
  if (text.length > TRAJECTORY_MAX_BYTES)
    throw new TrajectoryError("This camera file is too large to be a cameras.json.");
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch (error) {
    throw new TrajectoryError(`This file is not valid JSON: ${error instanceof Error ? error.message : String(error)}`);
  }
  if (!Array.isArray(data))
    throw new TrajectoryError("A Graphdeco cameras.json is a JSON array of camera objects.");
  if (!data.length) throw new TrajectoryError("This camera file contains no cameras.");

  const warnings: string[] = [];
  const seen = new Map<number, number>();
  const cameras: TrajectoryCamera[] = data.map((entry, index) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry))
      throw new TrajectoryError(`Camera ${index}: each entry must be an object.`);
    const record = entry as Record<string, unknown>;
    if (!finite(record.id)) throw new TrajectoryError(`Camera ${index}: id must be a finite number.`);
    if (typeof record.img_name !== "string" || !record.img_name.trim())
      throw new TrajectoryError(`Camera ${index}: img_name must be a non-empty string.`);
    for (const key of ["width", "height", "fx", "fy"] as const)
      if (!finite(record[key]) || (record[key] as number) <= 0)
        throw new TrajectoryError(`Camera ${index}: ${key} must be a positive finite number.`);
    const position = readVector3(record.position, "position", index);
    const rotation = readRotation(record.rotation, index);
    const previous = seen.get(record.id);
    if (previous !== undefined && warnings.length < 4)
      warnings.push(`id ${record.id} appears at positions ${previous} and ${index}; source order is used instead.`);
    seen.set(record.id, index);
    const height = record.height as number;
    const fy = record.fy as number;
    return {
      id: record.id,
      imgName: (record.img_name as string).trim(),
      position,
      quaternion: openCvRotationToThreeQuaternion(rotation),
      width: record.width as number,
      height,
      fx: record.fx as number,
      fy,
      fovYDeg: verticalFovDegrees(fy, height),
      sourceIndex: index,
      timestampSeconds: null,
    };
  });
  if (seen.size !== cameras.length)
    warnings.unshift(`${cameras.length - seen.size} duplicate camera id(s); cameras are keyed by source position.`);
  return { cameras, warnings };
}

// ---------------------------------------------------------------------------
// Spatial anchors
// ---------------------------------------------------------------------------
//
// A capture rig does not produce one pose per place. The real trajectory this
// was built against is 948 cameras at 158 stations: six cube faces
// (front/back/left/right/up/down) share one centre, and their positions are
// *not* bit-identical — the largest spread inside a station is 2.3e-14, which is
// float noise, not geometry. Replaying the raw poses therefore spins on the spot
// six times per station, which is not a trajectory.
//
// So the raw cameras are collapsed to the distinct places they were captured
// from. Those anchors, in capture order, are the polyline the viewer is allowed
// to explore around; the raw poses stay available for debug visualization only.

/** One distinct place on the capture path, with the raw cameras taken there. */
export interface SpatialAnchor {
  /** Position along the collapsed path, 0-based. */
  index: number;
  /** Mean of the member camera centres; every member is within `tolerance`. */
  position: Vector3Tuple;
  /** `sourceIndex` of every raw camera at this place, in source order. */
  cameras: number[];
  /** `station_039` if the image names carry one, else the first image name. */
  label: string;
  /** Station number parsed from the image names, when they all agree. */
  station: number | null;
}

export interface SpatialTrajectory {
  /** The raw cameras, untouched and in source order. */
  cameras: TrajectoryCamera[];
  /** The distinct capture places, in the order they first appear. */
  anchors: SpatialAnchor[];
  /** Positions closer than this were treated as the same place. */
  tolerance: number;
  /** Diagonal of the camera-centre bounding box. The scene's working scale. */
  extent: number;
  /** Median gap between consecutive anchors. Zero when there is only one. */
  medianSpacing: number;
  /** Suggested tube radius, and the range the UI may set it to. */
  defaultRadius: number;
  minRadius: number;
  maxRadius: number;
  /** Suggested walking speed in world units per second, and its range. */
  defaultSpeed: number;
  minSpeed: number;
  maxSpeed: number;
  warnings: string[];
}

/**
 * Clustering tolerance as a fraction of the scene extent. On the real data this
 * is 0.0168 world units: about 10^12 above the intra-station float noise and
 * about 7.7x below the smallest genuine gap between stations (0.129), so it
 * cannot merge two real places or split one.
 */
export const ANCHOR_TOLERANCE_OF_EXTENT = 1e-3;
/** Radius bounds, as fractions of the scene extent. */
export const RADIUS_MIN_OF_EXTENT = 1 / 400;
export const RADIUS_MAX_OF_EXTENT = 1 / 10;
/** Half a station gap: close enough to the path to stay in reconstructed space. */
export const RADIUS_OF_SPACING = 0.5;
/** Walking speed, and its range, as multiples of the median station gap. */
export const SPEED_OF_SPACING = 2;
export const SPEED_MIN_OF_SPACING = 0.25;
export const SPEED_MAX_OF_SPACING = 12;
/**
 * Floor on the clustering tolerance, relative to how far from the origin the
 * capture sits. A scene extent is not always a usable scale: a capture from a
 * single place has an extent of only its own float noise, and a tolerance
 * derived from that would split one place into six. Two coordinates that agree
 * to one part in a billion are the same point at any scene scale.
 */
const COORDINATE_EPSILON = 1e-9;
/** Used only when the capture sits exactly on the origin with no extent. */
const DEGENERATE_TOLERANCE = 1e-9;

/**
 * The station a training image belongs to, or null. Graphdeco itself imposes no
 * naming scheme, so this is used to *check* the collapsed order, never to build
 * it — a file whose names carry no station still works.
 */
export function parseStationToken(imgName: string): number | null {
  const match = /(?:^|[^a-z0-9])station[_-]?(\d+)/i.exec(imgName);
  if (!match) return null;
  const value = Number.parseInt(match[1], 10);
  return Number.isSafeInteger(value) ? value : null;
}

const distance = (a: Vector3Tuple, b: Vector3Tuple) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Collapse raw training cameras to the distinct places they were captured from.
 *
 * Matching is by proximity, not by equality or by name: the six faces of one
 * station differ in the last bits of their coordinates. A uniform grid of
 * `tolerance`-sized cells keeps this linear — each camera looks only in the 27
 * cells around it, so 948 cameras cost 948 lookups, not 948^2 comparisons.
 */
export function buildSpatialTrajectory(cameras: TrajectoryCamera[]): SpatialTrajectory {
  if (!cameras.length) throw new TrajectoryError("A trajectory needs at least one camera.");
  const warnings: string[] = [];

  const min: Vector3Tuple = [Infinity, Infinity, Infinity];
  const max: Vector3Tuple = [-Infinity, -Infinity, -Infinity];
  for (const camera of cameras)
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], camera.position[axis]);
      max[axis] = Math.max(max[axis], camera.position[axis]);
    }
  const extent = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
  const magnitude = Math.max(...min.map(Math.abs), ...max.map(Math.abs));
  const tolerance = Math.max(
    extent * ANCHOR_TOLERANCE_OF_EXTENT,
    magnitude * COORDINATE_EPSILON,
    DEGENERATE_TOLERANCE,
  );

  // Seed position per anchor: what membership is measured against. The reported
  // position is the mean of the members, which is within `tolerance` of it.
  const seeds: Vector3Tuple[] = [];
  const members: number[][] = [];
  const grid = new Map<string, number[]>();
  const cell = (p: Vector3Tuple) => [Math.floor(p[0] / tolerance), Math.floor(p[1] / tolerance), Math.floor(p[2] / tolerance)];

  for (const camera of cameras) {
    const [cx, cy, cz] = cell(camera.position);
    let found = -1;
    let best = Infinity;
    for (let dx = -1; dx <= 1; dx += 1)
      for (let dy = -1; dy <= 1; dy += 1)
        for (let dz = -1; dz <= 1; dz += 1)
          for (const candidate of grid.get(`${cx + dx},${cy + dy},${cz + dz}`) ?? []) {
            const d = distance(seeds[candidate], camera.position);
            if (d <= tolerance && d < best) {
              best = d;
              found = candidate;
            }
          }
    if (found >= 0) {
      members[found].push(camera.sourceIndex);
      continue;
    }
    const index = seeds.length;
    seeds.push(camera.position);
    members.push([camera.sourceIndex]);
    const key = `${cx},${cy},${cz}`;
    const bucket = grid.get(key);
    if (bucket) bucket.push(index);
    else grid.set(key, [index]);
  }

  const byIndex = new Map(cameras.map((camera) => [camera.sourceIndex, camera]));
  const anchors: SpatialAnchor[] = members.map((group, index) => {
    const sum: Vector3Tuple = [0, 0, 0];
    for (const sourceIndex of group) {
      const position = byIndex.get(sourceIndex)!.position;
      sum[0] += position[0];
      sum[1] += position[1];
      sum[2] += position[2];
    }
    const first = byIndex.get(group[0])!;
    const stations = new Set(group.map((sourceIndex) => parseStationToken(byIndex.get(sourceIndex)!.imgName)));
    const station = stations.size === 1 ? [...stations][0] : null;
    return {
      index,
      position: [sum[0] / group.length, sum[1] / group.length, sum[2] / group.length] as Vector3Tuple,
      cameras: group,
      label: station === null ? first.imgName : `station_${String(station).padStart(3, "0")}`,
      station,
    };
  });

  // Order check. Source order is what the path uses; station numbers only get to
  // report that it disagrees with them, because no schema guarantees they exist.
  const numbered = anchors.filter((anchor) => anchor.station !== null);
  if (numbered.length === anchors.length && anchors.length > 1) {
    let inversions = 0;
    for (let i = 1; i < numbered.length; i += 1) if (numbered[i].station! < numbered[i - 1].station!) inversions += 1;
    if (inversions)
      warnings.push(
        `Station numbers are not increasing in file order (${inversions} step(s) go backwards); the path follows file order.`,
      );
  }
  const mixed = anchors.filter((anchor) => anchor.station === null && anchor.cameras.length > 1).length;
  if (mixed) warnings.push(`${mixed} place(s) hold cameras from more than one station name.`);

  const gaps: number[] = [];
  for (let i = 1; i < anchors.length; i += 1) {
    const gap = distance(anchors[i - 1].position, anchors[i].position);
    if (gap > 0) gaps.push(gap);
  }
  const medianSpacing = median(gaps);

  // A capture from one place still has to be walkable, so when the extent gives
  // no scale the tolerance does: it is the only length this capture defines.
  const scale = extent > tolerance ? extent : tolerance * 1000;
  const minRadius = scale * RADIUS_MIN_OF_EXTENT;
  const maxRadius = scale * RADIUS_MAX_OF_EXTENT;
  const defaultRadius = THREE.MathUtils.clamp(
    medianSpacing > 0 ? medianSpacing * RADIUS_OF_SPACING : maxRadius,
    minRadius,
    maxRadius,
  );
  // With no spacing to go by (one place), walking is bounded by the tube anyway.
  const spacing = medianSpacing > 0 ? medianSpacing : defaultRadius;
  return {
    cameras,
    anchors,
    tolerance,
    extent,
    medianSpacing,
    defaultRadius,
    minRadius,
    maxRadius,
    defaultSpeed: spacing * SPEED_OF_SPACING,
    minSpeed: spacing * SPEED_MIN_OF_SPACING,
    maxSpeed: spacing * SPEED_MAX_OF_SPACING,
    warnings,
  };
}

/** Index of the anchor whose centre is closest to `position`. */
export function nearestAnchor(anchors: SpatialAnchor[], position: Vector3Tuple): number {
  let best = -1;
  let bestDistance = Infinity;
  for (const anchor of anchors) {
    const d = distance(anchor.position, position);
    if (d < bestDistance) {
      bestDistance = d;
      best = anchor.index;
    }
  }
  return best;
}
