// Deterministic training-camera fixtures in the original INRIA/Graphdeco
// cameras.json schema. Every value is generated here; no camera file from
// private research data is committed.

export interface GraphdecoCamera {
  id: number;
  img_name: string;
  width: number;
  height: number;
  position: [number, number, number];
  rotation: number[][];
  fy: number;
  fx: number;
}

/**
 * An asymmetric known pose. Its camera-to-world rotation has every axis on a
 * different world axis and is not symmetric, so a transpose, an inverse or a
 * flipped sign produces a different answer instead of accidentally passing:
 *
 *   columns (OpenCV camera axes in world space)
 *     right   = col 0 = (0, 1, 0)
 *     down    = col 1 = (0, 0, 1)
 *     forward = col 2 = (1, 0, 0)     right x down = forward, so right-handed
 *
 *   rows (what a transposed reading would give)
 *     row 0 = (0, 0, 1)   row 1 = (1, 0, 0)   row 2 = (0, 1, 0)
 *
 * fy is exactly half the image height, so the vertical field of view is exactly
 * 90 degrees; fx is different, so reading fx, or pairing fy with width, is
 * visibly wrong rather than merely imprecise.
 */
export const ASYMMETRIC_CAMERA: GraphdecoCamera = {
  id: 7,
  img_name: "asymmetric_007.png",
  width: 1600,
  height: 900,
  position: [1, 2, 3],
  rotation: [
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
  ],
  fy: 450,
  fx: 1385.6406460551018,
};

/** Expected Three.js basis for ASYMMETRIC_CAMERA: -z forward, +y up. */
export const ASYMMETRIC_EXPECTED = {
  position: [1, 2, 3] as [number, number, number],
  forward: [1, 0, 0] as [number, number, number], //  cv forward
  up: [0, 0, -1] as [number, number, number], // -cv down
  right: [0, 1, 0] as [number, number, number], //  cv right
  fovYDeg: 90,
};

/** Identity camera-to-world: OpenCV looks along world +z with world -y up. */
export const IDENTITY_CAMERA: GraphdecoCamera = {
  id: 0,
  img_name: "identity_000.png",
  width: 800,
  height: 800,
  position: [0, 0, 0],
  rotation: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
  fy: 400,
  fx: 400,
};

/** Rotation about the world Y axis by `degrees`, as a camera-to-world matrix. */
export function yawedCamera(id: number, degrees: number, position: [number, number, number]): GraphdecoCamera {
  const a = (degrees * Math.PI) / 180;
  const c = Math.cos(a);
  const s = Math.sin(a);
  // Ry(a) applied to the identity OpenCV basis; columns stay right/down/forward.
  return {
    id,
    img_name: `yaw_${String(id).padStart(3, "0")}.png`,
    width: 1024,
    height: 768,
    position,
    rotation: [
      [c, 0, s],
      [0, 1, 0],
      [-s, 0, c],
    ],
    fy: 768,
    fx: 1024,
  };
}

/** A short, valid path: four cameras yawing around a circle. */
export function circleCameras(count = 4): GraphdecoCamera[] {
  return Array.from({ length: count }, (_, i) => {
    const degrees = (360 / count) * i;
    const a = (degrees * Math.PI) / 180;
    return yawedCamera(i, degrees, [Math.sin(a) * 5, 0, Math.cos(a) * 5]);
  });
}

export const cameraJson = (cameras: unknown) => JSON.stringify(cameras);

/**
 * Cameras on a circle in the world xz-plane, all looking at `target`.
 *
 * Built in OpenCV axes so the matrix is a genuine Graphdeco camera-to-world
 * rotation with right/down/forward as its columns:
 *
 *   forward = normalize(target - position)
 *   right   = normalize(forward x worldUp)
 *   down    = forward x right            so right x down = forward
 */
export function lookAtCameras(count = 8, radius = 4, target: [number, number, number] = [0, 0, 0], height = 1.2): GraphdecoCamera[] {
  const sub = (a: number[], b: number[]) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a: number[], b: number[]) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const unit = (a: number[]) => {
    const n = Math.hypot(a[0], a[1], a[2]);
    return [a[0] / n, a[1] / n, a[2] / n];
  };
  return Array.from({ length: count }, (_, i) => {
    const angle = (i / count) * Math.PI * 2;
    const position: [number, number, number] = [
      target[0] + Math.sin(angle) * radius,
      target[1] + height,
      target[2] + Math.cos(angle) * radius,
    ];
    const forward = unit(sub(target, position));
    const right = unit(cross(forward, [0, 1, 0]));
    const down = cross(forward, right);
    return {
      id: i,
      img_name: `station_${String(i).padStart(3, "0")}.png`,
      width: 1600,
      height: 900,
      position,
      rotation: [
        [right[0], down[0], forward[0]],
        [right[1], down[1], forward[1]],
        [right[2], down[2], forward[2]],
      ],
      fy: 450,
      fx: 800,
    };
  });
}

/** The six cube faces the real capture rig writes for every place it stands. */
export const CUBE_FACES = ["front", "back", "left", "right", "up", "down"] as const;

const FACE_ROTATIONS: Record<(typeof CUBE_FACES)[number], number[][]> = {
  // Right-handed rotations in camera space, composed onto the identity OpenCV
  // basis (columns right / down / forward), so each face is a proper rotation.
  front: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  back: [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], // Ry(180)
  left: [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], // Ry(90)
  right: [[0, 0, -1], [0, 1, 0], [1, 0, 0]], // Ry(-90)
  up: [[1, 0, 0], [0, 0, -1], [0, 1, 0]], // Rx(-90)
  down: [[1, 0, 0], [0, 0, 1], [0, -1, 0]], // Rx(90)
};

/**
 * One place, six cube faces, the way the real 948-camera export does it.
 *
 * The six positions at a place are deliberately *not* bit-identical: on the real
 * data the largest spread inside a station is 2.3e-14, which is float noise from
 * the reconstruction, and a fixture with identical tuples would let an
 * equality-based grouping pass a test that proximity grouping exists to solve.
 */
export function cubeStationCameras(
  positions: [number, number, number][],
  jitter = 2e-14,
): GraphdecoCamera[] {
  return positions.flatMap((position, station) =>
    CUBE_FACES.map((face, f) => ({
      id: station * CUBE_FACES.length + f,
      img_name: `station_${String(station).padStart(3, "0")}_t${String(station).padStart(3, "0")}_0s_${face}.png`,
      width: 2048,
      height: 2048,
      position: [
        position[0] + jitter * (f - 2.5),
        position[1] - jitter * (f - 2.5),
        position[2] + jitter * ((f % 2) - 0.5),
      ] as [number, number, number],
      rotation: FACE_ROTATIONS[face],
      fy: 1024,
      fx: 1024,
    })),
  );
}

/** `count` places along the world x axis, `spacing` apart. */
export function stationLine(count: number, spacing = 0.3): [number, number, number][] {
  return Array.from({ length: count }, (_, i) => [i * spacing, 0, 0] as [number, number, number]);
}
