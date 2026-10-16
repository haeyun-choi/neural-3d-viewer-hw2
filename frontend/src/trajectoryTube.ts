// The region the viewer is allowed into: a tube around the capture path.
//
// A Gaussian reconstruction is only trustworthy where the training cameras
// actually looked. A metre off the path the same scene is a smear of
// view-dependent artefacts, so free flight through it looks broken even though
// nothing is broken. This module defines the allowed set as
//
//     { p : distance(p, capture polyline) <= radius }
//
// and projects anything outside it back onto the boundary. Two properties come
// out of that definition rather than out of special cases:
//
// * **Sliding.** Only the component of the motion that leaves the tube is
//   removed, so walking into the wall at an angle keeps the along-path part of
//   the motion instead of stopping dead.
// * **Rounded ends.** The closest point on a segment is clamped to its
//   endpoints, so past the last anchor the allowed set is already a ball of the
//   same radius. No end cap is coded; it is what the distance function says.
//
// This module is pure: no renderer, no DOM, no React. FreeLookController.ts
// proposes a position, this decides where the camera actually lands.

import * as THREE from "three";
import type { SpatialAnchor } from "./cameraTrajectory";

export interface TubeSegment {
  start: THREE.Vector3;
  /** end - start. Zero for the single-anchor case. */
  delta: THREE.Vector3;
  lengthSquared: number;
  length: number;
}

export interface TubePath {
  points: THREE.Vector3[];
  segments: TubeSegment[];
  totalLength: number;
}

export interface ClosestPoint {
  point: THREE.Vector3;
  /** Index into `path.segments`. */
  segment: number;
  /** Position along that segment, 0..1. */
  t: number;
  distance: number;
}

export interface TubeConstraint {
  position: THREE.Vector3;
  /** The segment to search around next frame. Feed it back in. */
  segment: number;
  /** Distance from the path *after* constraining. */
  distance: number;
  /** True when the move was blocked by the boundary. */
  clamped: boolean;
}

/**
 * How many segments either side of the active one are searched.
 *
 * A capture path crosses itself. A global nearest-point search would hand the
 * camera to whichever branch happens to be nearest, which teleports the walker
 * onto a different part of the route at every crossing. Searching a window
 * around the segment the walker is *on* keeps them on their own branch. Eight
 * segments is about 2.4 world units on the real data, far more than one frame
 * of walking, so the window can never be outrun.
 */
export const ACTIVE_SEGMENT_WINDOW = 8;

/** The polyline through the capture places, in capture order. */
export function buildTubePath(anchors: SpatialAnchor[]): TubePath {
  const points = anchors.map((anchor) => new THREE.Vector3(...anchor.position));
  const segments: TubeSegment[] = [];
  if (points.length === 1) {
    // One place: a degenerate segment, so the allowed set is a ball around it
    // and every caller below stays on its ordinary path.
    segments.push({ start: points[0].clone(), delta: new THREE.Vector3(), lengthSquared: 0, length: 0 });
  }
  for (let i = 1; i < points.length; i += 1) {
    const delta = points[i].clone().sub(points[i - 1]);
    const length = delta.length();
    segments.push({ start: points[i - 1].clone(), delta, lengthSquared: length * length, length });
  }
  return { points, segments, totalLength: segments.reduce((sum, segment) => sum + segment.length, 0) };
}

const scratch = new THREE.Vector3();

/**
 * Closest point on the path. With `near >= 0` only the segments within `window`
 * of it are considered; see ACTIVE_SEGMENT_WINDOW for why that is the default
 * for live movement and why a global search is only for entering or re-homing.
 */
export function closestOnPath(
  path: TubePath,
  position: THREE.Vector3,
  options: { near?: number; window?: number } = {},
): ClosestPoint {
  const count = path.segments.length;
  if (!count) throw new Error("A tube path needs at least one anchor.");
  const near = options.near ?? -1;
  const window = options.window ?? ACTIVE_SEGMENT_WINDOW;
  const first = near >= 0 ? Math.max(0, Math.min(count - 1, near) - window) : 0;
  const last = near >= 0 ? Math.min(count - 1, Math.min(count - 1, near) + window) : count - 1;

  let bestDistance = Infinity;
  let bestSegment = first;
  let bestT = 0;
  const bestPoint = new THREE.Vector3();
  for (let index = first; index <= last; index += 1) {
    const segment = path.segments[index];
    const t = segment.lengthSquared > 0
      ? THREE.MathUtils.clamp(scratch.copy(position).sub(segment.start).dot(segment.delta) / segment.lengthSquared, 0, 1)
      : 0;
    scratch.copy(segment.start).addScaledVector(segment.delta, t);
    const distance = scratch.distanceTo(position);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestSegment = index;
      bestT = t;
      bestPoint.copy(scratch);
    }
  }
  return { point: bestPoint, segment: bestSegment, t: bestT, distance: bestDistance };
}

/**
 * Keep a proposed camera position inside the tube.
 *
 * `activeSegment` is the segment the walker was on last frame, or -1 to search
 * the whole path (entering the mode, or returning to an anchor).
 */
export function constrainToTube(
  path: TubePath,
  proposed: THREE.Vector3,
  radius: number,
  activeSegment: number,
): TubeConstraint {
  const closest = closestOnPath(path, proposed, { near: activeSegment });
  if (!(closest.distance > radius))
    return { position: proposed.clone(), segment: closest.segment, distance: closest.distance, clamped: false };
  const outward = scratch.copy(proposed).sub(closest.point).divideScalar(closest.distance);
  return {
    position: closest.point.clone().addScaledVector(outward, radius),
    segment: closest.segment,
    distance: radius,
    clamped: true,
  };
}
