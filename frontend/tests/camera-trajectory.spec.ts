import { expect, test } from "@playwright/test";
import * as THREE from "three";
import {
  ANCHOR_TOLERANCE_OF_EXTENT,
  buildSpatialTrajectory,
  cameraAxes,
  nearestAnchor,
  openCvRotationToThreeQuaternion,
  parseGraphdecoCameras,
  parseStationToken,
  RADIUS_MAX_OF_EXTENT,
  RADIUS_MIN_OF_EXTENT,
  RADIUS_OF_SPACING,
  TrajectoryError,
  verticalFovDegrees,
} from "../src/cameraTrajectory";
import type { SpatialAnchor } from "../src/cameraTrajectory";
import { ACTIVE_SEGMENT_WINDOW, buildTubePath, closestOnPath, constrainToTube } from "../src/trajectoryTube";
import { FreeLookController, KEY_LOOK_RATE, MAX_PITCH, SPRINT_MULTIPLIER } from "../src/FreeLookController";
import { parseSceneManifest, resolveTrajectoryUrl, HostedSceneError } from "../src/hostedScene";
import {
  ASYMMETRIC_CAMERA,
  ASYMMETRIC_EXPECTED,
  cameraJson,
  circleCameras,
  cubeStationCameras,
  IDENTITY_CAMERA,
  stationLine,
  yawedCamera,
} from "./fixtures/cameras";
import type { GraphdecoCamera } from "./fixtures/cameras";

const close = (actual: number[], expected: number[], tolerance = 1e-9) => {
  expect(actual).toHaveLength(expected.length);
  actual.forEach((value, i) => expect(Math.abs(value - expected[i])).toBeLessThan(tolerance));
};
const reason = (cameras: unknown) => {
  try {
    parseGraphdecoCameras(cameraJson(cameras));
    return "";
  } catch (error) {
    expect(error).toBeInstanceOf(TrajectoryError);
    return (error as TrajectoryError).message;
  }
};

// --- Coordinate semantics ------------------------------------------------
//
// The reference is the official Graphdeco implementation, not the field names:
// utils/camera_utils.py::camera_to_JSON writes inv(Rt) where Rt is built the
// same way as in utils/graphics_utils.py::getWorld2View2, which names that
// inverse C2W. So `position` is the camera centre and `rotation` is the
// camera-to-world rotation with OpenCV axes (right, down, forward) as columns.

test("a Graphdeco rotation is read as camera-to-world with OpenCV axes in its columns", () => {
  const { cameras } = parseGraphdecoCameras(cameraJson([ASYMMETRIC_CAMERA]));
  expect(cameras).toHaveLength(1);
  const axes = cameraAxes(cameras[0].quaternion);
  close(cameras[0].position, ASYMMETRIC_EXPECTED.position);
  close(axes.forward, ASYMMETRIC_EXPECTED.forward, 1e-12);
  close(axes.up, ASYMMETRIC_EXPECTED.up, 1e-12);
  close(axes.right, ASYMMETRIC_EXPECTED.right, 1e-12);

  // The same matrix read transposed, inverted or sign-flipped must not agree,
  // which is what makes the assertions above meaningful.
  const rows = ASYMMETRIC_CAMERA.rotation;
  const transposed = [0, 1, 2].map((r) => [rows[0][r], rows[1][r], rows[2][r]]);
  const transposedAxes = cameraAxes(openCvRotationToThreeQuaternion(transposed));
  expect(transposedAxes.forward).not.toEqual(axes.forward);
  expect(transposedAxes.up).not.toEqual(axes.up);
});

test("an identity camera-to-world rotation looks along world +z with world -y up", () => {
  // OpenCV: +x right, +y down, +z forward. With the identity matrix those are
  // the world axes, so a Three.js camera must look at +z and have up = -y.
  const { cameras } = parseGraphdecoCameras(cameraJson([IDENTITY_CAMERA]));
  const axes = cameraAxes(cameras[0].quaternion);
  close(axes.forward, [0, 0, 1], 1e-12);
  close(axes.up, [0, -1, 0], 1e-12);
  close(axes.right, [1, 0, 0], 1e-12);
});

test("a yawed camera rotates about the world up axis in the expected direction", () => {
  // Yaw by +90 degrees about world +y takes OpenCV forward from +z to +x.
  const { cameras } = parseGraphdecoCameras(cameraJson([yawedCamera(1, 90, [0, 0, 0])]));
  const axes = cameraAxes(cameras[0].quaternion);
  close(axes.forward, [1, 0, 0], 1e-9);
  close(axes.up, [0, -1, 0], 1e-9);
  close(axes.right, [0, 0, -1], 1e-9);
});

test("the Three.js basis stays right-handed and orthonormal for every camera", () => {
  const { cameras } = parseGraphdecoCameras(cameraJson(circleCameras(8)));
  for (const camera of cameras) {
    const { forward, up, right } = cameraAxes(camera.quaternion);
    const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    for (const axis of [forward, up, right]) expect(Math.abs(dot(axis, axis) - 1)).toBeLessThan(1e-9);
    expect(Math.abs(dot(forward, up))).toBeLessThan(1e-9);
    expect(Math.abs(dot(forward, right))).toBeLessThan(1e-9);
    expect(Math.abs(dot(up, right))).toBeLessThan(1e-9);
    // right x up = -forward for a camera that looks down its own -z.
    const cross = [
      right[1] * up[2] - right[2] * up[1],
      right[2] * up[0] - right[0] * up[2],
      right[0] * up[1] - right[1] * up[0],
    ];
    close(cross, forward.map((v) => -v), 1e-9);
  }
});

test("vertical field of view comes from fy and image height, never fx or width", () => {
  // focal2fov(focal, pixels) = 2*atan(pixels / (2*focal)).
  expect(verticalFovDegrees(450, 900)).toBeCloseTo(90, 12);
  expect(verticalFovDegrees((900 * Math.sqrt(3)) / 2, 900)).toBeCloseTo(60, 10);
  const { cameras } = parseGraphdecoCameras(cameraJson([ASYMMETRIC_CAMERA]));
  expect(cameras[0].fovYDeg).toBeCloseTo(ASYMMETRIC_EXPECTED.fovYDeg, 12);
  // Reading fx, or pairing fy with width, would give these instead — both far
  // from the 90 degrees the fixture's fy and height actually mean.
  expect(verticalFovDegrees(ASYMMETRIC_CAMERA.fx, ASYMMETRIC_CAMERA.height)).toBeCloseTo(35.9834, 4);
  expect(verticalFovDegrees(ASYMMETRIC_CAMERA.fy, ASYMMETRIC_CAMERA.width)).toBeCloseTo(121.2845, 4);
});

// --- Parsing and validation ----------------------------------------------

test("a valid cameras.json parses into the canonical model", () => {
  const { cameras, warnings } = parseGraphdecoCameras(cameraJson(circleCameras(4)));
  expect(warnings).toEqual([]);
  expect(cameras).toHaveLength(4);
  expect(cameras.map((camera) => camera.id)).toEqual([0, 1, 2, 3]);
  expect(cameras.map((camera) => camera.sourceIndex)).toEqual([0, 1, 2, 3]);
  expect(cameras[0].imgName).toBe("yaw_000.png");
  expect(cameras[0].width).toBe(1024);
  expect(cameras[0].height).toBe(768);
  expect(cameras[0].fx).toBe(1024);
  expect(cameras[0].fy).toBe(768);
  expect(cameras[0].timestampSeconds).toBeNull();
  expect(cameras[0].quaternion).toHaveLength(4);
  const length = Math.hypot(...cameras[0].quaternion);
  expect(Math.abs(length - 1)).toBeLessThan(1e-12);
});

test("malformed, missing and non-finite fields are refused with a specific reason", () => {
  const valid = circleCameras(2);
  const broken = (mutate: (camera: GraphdecoCamera) => void) => {
    const copy = JSON.parse(JSON.stringify(valid)) as GraphdecoCamera[];
    mutate(copy[1]);
    return reason(copy);
  };
  expect(reason("not an array")).toContain("JSON array");
  expect(reason({ cameras: [] })).toContain("JSON array");
  expect(parseGraphdecoCameras.bind(null, "{oops")).toThrow(/not valid JSON/);
  expect(reason([1, 2, 3])).toContain("must be an object");
  expect(broken((c) => { delete (c as Partial<GraphdecoCamera>).id; })).toContain("id must be a finite number");
  expect(broken((c) => { (c as unknown as { id: unknown }).id = Number.NaN; })).toContain("id must be a finite number");
  expect(broken((c) => { c.img_name = "  "; })).toContain("img_name");
  expect(broken((c) => { c.width = 0; })).toContain("width");
  expect(broken((c) => { c.height = -8; })).toContain("height");
  expect(broken((c) => { c.fy = Number.POSITIVE_INFINITY; })).toContain("fy");
  expect(broken((c) => { c.position = [1, 2] as unknown as [number, number, number]; })).toContain("position");
  expect(broken((c) => { c.position = [1, 2, Number.NaN]; })).toContain("position");
  expect(broken((c) => { c.rotation = [[1, 0, 0], [0, 1, 0]]; })).toContain("3x3");
  expect(broken((c) => { c.rotation[0][0] = Number.NaN; })).toContain("3x3");
  // A scaled matrix is not a rotation, and neither is a reflection.
  expect(broken((c) => { c.rotation = c.rotation.map((row) => row.map((v) => v * 2)); })).toContain("rotation matrix");
  expect(broken((c) => { c.rotation = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]; })).toContain("rotation matrix");
});

test("an empty camera list is refused, and nothing is returned from a partly bad file", () => {
  expect(reason([])).toContain("no cameras");
  const mixed = circleCameras(3) as unknown[];
  (mixed[2] as GraphdecoCamera).rotation = [[2, 0, 0], [0, 2, 0], [0, 0, 2]];
  expect(reason(mixed)).toContain("Camera 2");
});

test("duplicate ids are kept but reported, and cameras stay keyed by source position", () => {
  const cameras = circleCameras(3);
  cameras[1].id = cameras[0].id;
  cameras[2].id = cameras[0].id;
  const parsed = parseGraphdecoCameras(cameraJson(cameras));
  expect(parsed.cameras).toHaveLength(3);
  expect(parsed.cameras.map((camera) => camera.sourceIndex)).toEqual([0, 1, 2]);
  expect(parsed.warnings.join(" ")).toContain("duplicate camera id");
  expect(parsed.warnings.join(" ")).toContain("source order is used instead");
});

// --- Spatial anchors ------------------------------------------------------

const parse = (cameras: unknown) => parseGraphdecoCameras(cameraJson(cameras)).cameras;
const collapse = (cameras: unknown) => buildSpatialTrajectory(parse(cameras));

test("a station token is read from the image name, and only when it is really there", () => {
  expect(parseStationToken("station_039_t019_5s_left.png")).toBe(39);
  expect(parseStationToken("STATION-7.png")).toBe(7);
  expect(parseStationToken("capture/station000_up.png")).toBe(0);
  // Not a station: the word has to stand on its own, and there has to be a number.
  expect(parseStationToken("constellation_012.png")).toBeNull();
  expect(parseStationToken("station_left.png")).toBeNull();
  expect(parseStationToken("frame_0012.png")).toBeNull();
});

test("six cube faces per place collapse to one anchor each, by proximity not equality", () => {
  const raw = cubeStationCameras(stationLine(12, 0.3));
  expect(raw).toHaveLength(72);
  // The fixture, like the real export, writes six positions that differ in the
  // last bits. Equality-based grouping would find 72 distinct places.
  expect(new Set(raw.map((camera) => camera.position.join(","))).size).toBe(72);

  const trajectory = collapse(raw);
  expect(trajectory.cameras).toHaveLength(72);
  expect(trajectory.anchors).toHaveLength(12);
  expect(trajectory.anchors.map((anchor) => anchor.index)).toEqual([...Array(12).keys()]);
  for (const [i, anchor] of trajectory.anchors.entries()) {
    expect(anchor.cameras).toHaveLength(6);
    // Anchors keep capture order, and each carries the raw cameras taken there.
    expect(anchor.cameras).toEqual([0, 1, 2, 3, 4, 5].map((f) => i * 6 + f));
    expect(anchor.station).toBe(i);
    expect(anchor.label).toBe(`station_${String(i).padStart(3, "0")}`);
    close(anchor.position, [i * 0.3, 0, 0], 1e-12);
  }
  expect(trajectory.warnings).toEqual([]);
});

test("the clustering tolerance is proportional to the scene, and sits between the noise and the real gaps", () => {
  const spacing = 0.3;
  const trajectory = collapse(cubeStationCameras(stationLine(12, spacing)));
  const extent = 11 * spacing;
  expect(trajectory.extent).toBeCloseTo(extent, 12);
  expect(trajectory.tolerance).toBeCloseTo(extent * ANCHOR_TOLERANCE_OF_EXTENT, 15);
  // Orders of magnitude above the intra-place float noise...
  expect(trajectory.tolerance).toBeGreaterThan(1e-10);
  // ...and well below the smallest genuine gap between places.
  expect(trajectory.tolerance).toBeLessThan(spacing / 5);

  // The same capture measured in millimetres gets a tolerance a thousand times
  // larger and still finds exactly the same twelve places.
  const scaled = cubeStationCameras(stationLine(12, spacing * 1000), 2e-11);
  expect(collapse(scaled).anchors).toHaveLength(12);
  expect(collapse(scaled).tolerance).toBeCloseTo(trajectory.tolerance * 1000, 9);
});

test("places closer together than the smallest real gap are still kept apart", () => {
  // Two places one tolerance-and-a-bit apart must not merge; two places inside
  // one tolerance must.
  const line = stationLine(10, 1);
  const extent = 9;
  const tolerance = extent * ANCHOR_TOLERANCE_OF_EXTENT;
  const apart = collapse(cubeStationCameras([...line, [line[9][0] + tolerance * 3, 0, 0]]));
  expect(apart.anchors).toHaveLength(11);
  const together = collapse(cubeStationCameras([...line, [line[9][0] + tolerance * 0.4, 0, 0]]));
  expect(together.anchors).toHaveLength(10);
  expect(together.anchors[9].cameras).toHaveLength(12);
});

test("spacing, the default radius and the radius bounds all come from the capture's own scale", () => {
  // Uneven gaps so the median is a real choice rather than the only value.
  const positions: [number, number, number][] = [[0, 0, 0], [0.2, 0, 0], [0.6, 0, 0], [0.8, 0, 0], [2, 0, 0]];
  const trajectory = collapse(cubeStationCameras(positions));
  expect(trajectory.medianSpacing).toBeCloseTo(0.3, 12); // gaps 0.2 0.4 0.2 1.2
  expect(trajectory.minRadius).toBeCloseTo(trajectory.extent * RADIUS_MIN_OF_EXTENT, 12);
  expect(trajectory.maxRadius).toBeCloseTo(trajectory.extent * RADIUS_MAX_OF_EXTENT, 12);
  expect(trajectory.defaultRadius).toBeCloseTo(trajectory.medianSpacing * RADIUS_OF_SPACING, 12);
  expect(trajectory.defaultRadius).toBeGreaterThanOrEqual(trajectory.minRadius);
  expect(trajectory.defaultRadius).toBeLessThanOrEqual(trajectory.maxRadius);
  expect(trajectory.defaultSpeed).toBeGreaterThan(0);
  expect(trajectory.minSpeed).toBeLessThan(trajectory.defaultSpeed);
  expect(trajectory.maxSpeed).toBeGreaterThan(trajectory.defaultSpeed);

  // A capture whose places are far apart relative to its extent has its radius
  // clamped rather than being allowed to swallow the scene.
  const sparse = collapse(cubeStationCameras([[0, 0, 0], [1, 0, 0]]));
  expect(sparse.defaultRadius).toBeCloseTo(sparse.extent * RADIUS_MAX_OF_EXTENT, 12);
});

test("degenerate captures are paths too: one place, or every camera in one spot", () => {
  const single = collapse(cubeStationCameras([[3, 4, 5]]));
  expect(single.anchors).toHaveLength(1);
  expect(single.extent).toBeLessThan(1e-12);
  expect(single.medianSpacing).toBe(0);
  expect(Number.isFinite(single.defaultRadius)).toBe(true);
  expect(single.defaultRadius).toBeGreaterThan(0);
  expect(Number.isFinite(single.defaultSpeed)).toBe(true);
  // A single anchor is still a tube: a ball around the one place.
  const tube = buildTubePath(single.anchors);
  expect(tube.segments).toHaveLength(1);
  expect(tube.totalLength).toBe(0);
  const held = constrainToTube(tube, new THREE.Vector3(3 + 10, 4, 5), 0.25, -1);
  expect(held.clamped).toBe(true);
  close(held.position.toArray(), [3.25, 4, 5], 1e-9);

  expect(() => buildSpatialTrajectory([])).toThrow(TrajectoryError);
});

test("station numbers check the collapsed order but never impose it", () => {
  const scrambled = cubeStationCameras(stationLine(4, 0.5)).map((camera, i) => ({
    ...camera,
    // Rename so the station numbers run backwards against the file order.
    img_name: `station_${String(3 - Math.floor(i / 6)).padStart(3, "0")}_face${i % 6}.png`,
  }));
  const trajectory = collapse(scrambled);
  // File order still wins: anchor 0 is the first place in the file.
  close(trajectory.anchors[0].position, [0, 0, 0], 1e-12);
  expect(trajectory.warnings.join(" ")).toContain("not increasing in file order");

  // A capture with no station names at all is fine and says nothing.
  const unnamed = collapse(circleCameras(4));
  expect(unnamed.anchors).toHaveLength(4);
  expect(unnamed.anchors[0].station).toBeNull();
  expect(unnamed.anchors[0].label).toBe("yaw_000.png");
  expect(unnamed.warnings).toEqual([]);
});

test("nearestAnchor finds the closest place, on and off the path", () => {
  const { anchors } = collapse(cubeStationCameras(stationLine(5, 1)));
  expect(nearestAnchor(anchors, [0, 0, 0])).toBe(0);
  expect(nearestAnchor(anchors, [2.4, 9, 9])).toBe(2);
  expect(nearestAnchor(anchors, [100, 0, 0])).toBe(4);
});

// --- The tube -------------------------------------------------------------

const lineAnchors = (count: number, spacing = 1): SpatialAnchor[] =>
  Array.from({ length: count }, (_, i) => ({
    index: i, position: [i * spacing, 0, 0], cameras: [i], label: `a${i}`, station: i,
  }));

test("the closest point on the path is clamped to each segment, which is what caps the ends", () => {
  const path = buildTubePath(lineAnchors(4));
  expect(path.segments).toHaveLength(3);
  expect(path.totalLength).toBeCloseTo(3, 12);

  const middle = closestOnPath(path, new THREE.Vector3(1.5, 2, 0));
  expect(middle.segment).toBe(1);
  expect(middle.t).toBeCloseTo(0.5, 12);
  expect(middle.distance).toBeCloseTo(2, 12);
  close(middle.point.toArray(), [1.5, 0, 0], 1e-12);

  // Past the last anchor the nearest point is the anchor itself, so the allowed
  // region there is a ball and no end cap has to be written.
  const beyond = closestOnPath(path, new THREE.Vector3(9, 0, 0));
  close(beyond.point.toArray(), [3, 0, 0], 1e-12);
  expect(beyond.distance).toBeCloseTo(6, 12);
  const before = closestOnPath(path, new THREE.Vector3(-4, 3, 0));
  close(before.point.toArray(), [0, 0, 0], 1e-12);
  expect(before.distance).toBeCloseTo(5, 12);
});

test("inside the tube nothing is touched; outside it the camera lands exactly on the boundary", () => {
  const path = buildTubePath(lineAnchors(4));
  const radius = 0.5;

  const inside = constrainToTube(path, new THREE.Vector3(1.2, 0.3, 0.1), radius, -1);
  expect(inside.clamped).toBe(false);
  close(inside.position.toArray(), [1.2, 0.3, 0.1], 1e-12);
  expect(inside.distance).toBeCloseTo(Math.hypot(0.3, 0.1), 12);

  const outside = constrainToTube(path, new THREE.Vector3(1.5, 4, 0), radius, -1);
  expect(outside.clamped).toBe(true);
  expect(outside.distance).toBeCloseTo(radius, 12);
  close(outside.position.toArray(), [1.5, radius, 0], 1e-12);

  // Sliding: a diagonal push into the wall keeps the along-path component and
  // loses only the part that left the tube. Walking the wall still makes ground.
  let position = new THREE.Vector3(0, radius, 0);
  let segment = 0;
  const step = new THREE.Vector3(0.1, 0.1, 0);
  for (let i = 0; i < 20; i += 1) {
    const result = constrainToTube(path, position.clone().add(step), radius, segment);
    position = result.position;
    segment = result.segment;
  }
  expect(position.x).toBeCloseTo(2, 6);         // all 20 steps of along-path motion kept
  expect(position.y).toBeCloseTo(radius, 9);    // and none of the outward motion
});

test("the end cap is a sphere: you may go one radius past the last anchor, in any direction", () => {
  const path = buildTubePath(lineAnchors(3));
  const radius = 0.4;
  for (const direction of [[1, 0, 0], [1, 1, 0], [0, 0, 1], [-1, 0, 0]] as const) {
    const far = new THREE.Vector3(...direction).normalize().multiplyScalar(50).add(new THREE.Vector3(2, 0, 0));
    const held = constrainToTube(path, far, radius, -1);
    // Every direction is allowed exactly `radius` from the nearest path point.
    expect(closestOnPath(path, held.position).distance).toBeCloseTo(radius, 9);
  }
  // And the path itself is never left: the far end stays within radius of it.
  const outward = constrainToTube(path, new THREE.Vector3(80, 0, 0), radius, -1);
  close(outward.position.toArray(), [2 + radius, 0, 0], 1e-9);
});

test("the active segment keeps the walker on their own branch where the path crosses itself", () => {
  // A capture that comes back along itself a few centimetres away: anchors 0..19
  // run out along +x at y = 0, anchors 20..39 come back at y = 0.05. The two
  // branches are adjacent in space and twenty segments apart in the walk, which
  // is exactly the situation a global nearest-point search gets wrong.
  const out: SpatialAnchor[] = Array.from({ length: 20 }, (_, i) => ({
    index: i, position: [i * 0.3, 0, 0], cameras: [i], label: `o${i}`, station: i,
  }));
  const back: SpatialAnchor[] = Array.from({ length: 20 }, (_, j) => ({
    index: 20 + j, position: [(19 - j) * 0.3, 0.05, 0], cameras: [20 + j], label: `r${j}`, station: 20 + j,
  }));
  const path = buildTubePath([...out, ...back]);
  const probe = new THREE.Vector3(3.15, 0.026, 0);

  // Walking the outbound branch keeps segment 10, not the return leg 28 that a
  // global search would pick because it happens to be a hair closer.
  expect(constrainToTube(path, probe.clone(), 0.5, 10).segment).toBe(10);
  // Walking the return branch, from exactly the same point, stays on it.
  expect(constrainToTube(path, probe.clone(), 0.5, 28).segment).toBe(28);
  // The global search really does prefer the other branch, so the window is
  // doing the work rather than the geometry being unambiguous.
  expect(closestOnPath(path, probe.clone()).segment).toBe(28);

  // The window is finite and centred on the active segment.
  expect(closestOnPath(path, probe.clone(), { near: 10 }).segment).toBe(10);
  expect(Math.abs(closestOnPath(path, probe.clone(), { near: 0, window: 2 }).segment)).toBeLessThanOrEqual(2);
  expect(ACTIVE_SEGMENT_WINDOW).toBeGreaterThan(1);
  expect(ACTIVE_SEGMENT_WINDOW).toBeLessThan(20);
});

test("a whole walk along the path never leaves the tube and never jumps a branch", () => {
  const path = buildTubePath(lineAnchors(20, 0.3));
  const radius = 0.15;
  let position = new THREE.Vector3(0, 0, 0);
  let segment = 0;
  const visited: number[] = [];
  for (let i = 0; i < 400; i += 1) {
    // Push forward, and also sideways and upward hard enough to hit the wall.
    const step = new THREE.Vector3(0.02, Math.sin(i / 7) * 0.05, Math.cos(i / 11) * 0.05);
    const result = constrainToTube(path, position.clone().add(step), radius, segment);
    expect(result.distance).toBeLessThanOrEqual(radius + 1e-9);
    // The segment only ever moves by a little, never across the path.
    expect(Math.abs(result.segment - segment)).toBeLessThanOrEqual(ACTIVE_SEGMENT_WINDOW);
    position = result.position;
    segment = result.segment;
    visited.push(segment);
  }
  // The walk really got somewhere: it reached the far end and stopped there.
  expect(Math.max(...visited)).toBe(path.segments.length - 1);
  expect(closestOnPath(path, position).distance).toBeLessThanOrEqual(radius + 1e-9);
  expect(position.x).toBeLessThanOrEqual(0.3 * 19 + radius + 1e-9);
});

// --- Walking input --------------------------------------------------------
//
// FreeLookController touches only four DOM members, so the movement maths can be
// exercised without a browser. What a real browser does with pointer lock, focus
// and key repeat is covered in tests/camera-path.spec.ts instead.

/** A frame is never longer than this; the controller clamps to it. */
const FRAME = 0.25;

function harness(lock = true) {
  const listeners = new Map<string, Set<(event: unknown) => void>>();
  const on = (type: string, fn: (event: unknown) => void) => {
    if (!listeners.has(type)) listeners.set(type, new Set());
    listeners.get(type)!.add(fn);
  };
  const off = (type: string, fn: (event: unknown) => void) => { listeners.get(type)?.delete(fn); };
  const document = {
    pointerLockElement: null as unknown,
    activeElement: null as unknown,
    defaultView: { addEventListener: on, removeEventListener: off },
    addEventListener: on,
    removeEventListener: off,
    exitPointerLock() { document.pointerLockElement = null; },
  };
  const canvas = {
    ownerDocument: document,
    addEventListener: on,
    removeEventListener: off,
    focus() { document.activeElement = canvas; },
    requestPointerLock() { document.pointerLockElement = canvas; },
  };
  const camera = new THREE.PerspectiveCamera(50, 1.6, 0.1, 100);
  const controller = new FreeLookController(
    camera, canvas as unknown as HTMLCanvasElement,
  );
  controller.enable();
  if (lock) canvas.requestPointerLock();
  const fire = (type: string, event: Record<string, unknown>) => {
    for (const fn of [...(listeners.get(type) ?? [])]) fn(event);
  };
  const key = (code: string, down = true) => fire(down ? "keydown" : "keyup", { code, preventDefault() {} });
  const mouse = (movementX: number, movementY: number) => fire("mousemove", { movementX, movementY });
  /** One frame's movement at `speed` world units per second, as a vector. */
  const walk = (speed: number) => controller.update(FRAME, speed / FRAME);
  return { camera, canvas, document, controller, key, mouse, fire, walk };
}

test("WASD moves along the flattened look direction, and only while the keys are held", () => {
  const h = harness();
  h.camera.quaternion.identity(); // looking down -z, up +y
  h.controller.adoptCamera();

  close(h.walk(2).delta.toArray(), [0, 0, 0], 1e-12); // nothing held
  h.key("KeyW");
  close(h.walk(2).delta.toArray(), [0, 0, -2], 1e-9); // forward is -z
  h.key("KeyD");
  // Diagonal movement is normalised: two keys are not faster than one.
  const diagonal = h.walk(2).delta;
  expect(diagonal.length()).toBeCloseTo(2, 9);
  close(diagonal.toArray(), [Math.SQRT1_2 * 2, 0, -Math.SQRT1_2 * 2], 1e-9);
  h.key("KeyW", false);
  close(h.walk(2).delta.toArray(), [2, 0, 0], 1e-9); // right only
  h.key("KeyD", false);
  h.key("KeyE");
  close(h.walk(2).delta.toArray(), [0, 2, 0], 1e-9); // up is scene up
  h.key("KeyQ");
  close(h.walk(2).delta.toArray(), [0, 0, 0], 1e-12); // and cancels
  h.key("KeyQ", false);

  // Shift is a multiplier on the speed, not a different direction.
  h.key("ShiftLeft");
  expect(h.walk(2).delta.length()).toBeCloseTo(2 * SPRINT_MULTIPLIER, 9);
  h.key("ShiftLeft", false);
  // Speed is world units per second, and a long stall is clamped to one frame
  // rather than teleporting the walker across the scene.
  expect(h.controller.update(FRAME / 2, 8).delta.length()).toBeCloseTo(1, 9);
  expect(h.controller.update(30, 8).delta.length()).toBeCloseTo(8 * FRAME, 9);
});

test("looking up does not fly the walker upwards", () => {
  const h = harness();
  h.controller.adoptCamera();
  h.mouse(0, -400); // look well up
  h.controller.update(1 / 60, 1);
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(h.controller.update(1 / 60, 1).quaternion);
  expect(forward.y).toBeGreaterThan(0.5);
  h.key("KeyW");
  const delta = h.walk(1).delta;
  expect(Math.abs(delta.y)).toBeLessThan(1e-9);
  expect(delta.length()).toBeCloseTo(1, 9);
});

test("pitch is clamped short of the poles and yaw never accumulates roll", () => {
  const h = harness();
  h.controller.adoptCamera();
  for (let i = 0; i < 40; i += 1) {
    h.mouse(137, -311); // large, uneven mouse movement in both axes
    h.controller.update(1 / 60, 1);
  }
  const quaternion = h.controller.update(1 / 60, 1).quaternion;
  const up = new THREE.Vector3(0, 1, 0).applyQuaternion(quaternion);
  const right = new THREE.Vector3(1, 0, 0).applyQuaternion(quaternion);
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(quaternion);
  // No roll: the camera's right axis stays in the scene's horizontal plane.
  expect(Math.abs(right.y)).toBeLessThan(1e-9);
  expect(up.y).toBeGreaterThan(0);
  // Pitch stopped short of straight up rather than flipping over the pole.
  expect(forward.y).toBeLessThan(Math.sin(MAX_PITCH) + 1e-9);
  expect(forward.y).toBeCloseTo(Math.sin(MAX_PITCH), 9);

  // And the other way.
  for (let i = 0; i < 80; i += 1) {
    h.mouse(-97, 411);
    h.controller.update(1 / 60, 1);
  }
  const down = new THREE.Vector3(0, 0, -1).applyQuaternion(h.controller.update(1 / 60, 1).quaternion);
  expect(down.y).toBeCloseTo(-Math.sin(MAX_PITCH), 9);
  expect(Math.abs(new THREE.Vector3(1, 0, 0).applyQuaternion(h.controller.update(1 / 60, 1).quaternion).y)).toBeLessThan(1e-9);
});

test("entering keeps the direction the camera already had, and yaws about the scene's up axis", () => {
  const h = harness();
  // A scene whose up axis is +z, as the orientation controls can make it, and a
  // camera looking along +x.
  h.camera.up.set(0, 0, 1);
  h.camera.position.set(0, 0, 0);
  h.camera.lookAt(new THREE.Vector3(1, 0, 0));
  const before = new THREE.Vector3(0, 0, -1).applyQuaternion(h.camera.quaternion);
  h.controller.adoptCamera();
  const after = new THREE.Vector3(0, 0, -1).applyQuaternion(h.controller.update(0, 1).quaternion);
  expect(after.dot(before)).toBeGreaterThan(1 - 1e-9);

  // Yaw turns about +z now, so the look direction stays in the xy-plane.
  h.mouse(300, 0);
  const turned = new THREE.Vector3(0, 0, -1).applyQuaternion(h.controller.update(1 / 60, 1).quaternion);
  expect(Math.abs(turned.z)).toBeLessThan(1e-9);
  expect(turned.dot(before)).toBeLessThan(1 - 1e-6);
  // Moving forward stays in that plane too.
  h.key("KeyW");
  expect(Math.abs(h.walk(1).delta.z)).toBeLessThan(1e-9);
});

test("arrow and page keys turn the view at a fixed rate, the way they point", () => {
  const h = harness(false);
  h.canvas.focus();
  h.camera.quaternion.identity(); // forward -z, up +y, right +x
  h.controller.adoptCamera();
  const forward = () => new THREE.Vector3(0, 0, -1).applyQuaternion(h.controller.update(0, 0).quaternion);

  // A quarter turn to the right at KEY_LOOK_RATE, taken in four frames.
  h.key("ArrowRight");
  const quarter = (Math.PI / 2) / KEY_LOOK_RATE;
  for (let i = 0; i < 4; i += 1) h.controller.update(quarter / 4, 0);
  h.key("ArrowRight", false);
  close(forward().toArray(), [1, 0, 0], 1e-9); // was -z, now +x: right
  // And back again, so the rate is symmetric rather than drifting.
  h.key("ArrowLeft");
  for (let i = 0; i < 4; i += 1) h.controller.update(quarter / 4, 0);
  h.key("ArrowLeft", false);
  close(forward().toArray(), [0, 0, -1], 1e-9);

  // Page Up looks up, Page Down looks down, and neither rolls the horizon.
  h.key("PageUp");
  for (let i = 0; i < 4; i += 1) h.controller.update(quarter / 8, 0);
  h.key("PageUp", false);
  const up = forward();
  expect(up.y).toBeCloseTo(Math.sin(Math.PI / 4), 9);
  expect(Math.abs(new THREE.Vector3(1, 0, 0).applyQuaternion(h.controller.update(0, 0).quaternion).y)).toBeLessThan(1e-9);
  h.key("PageDown");
  for (let i = 0; i < 8; i += 1) h.controller.update(quarter / 8, 0);
  h.key("PageDown", false);
  expect(forward().y).toBeCloseTo(-Math.sin(Math.PI / 4), 9);
});

test("keys are ignored when the canvas has neither focus nor the pointer", () => {
  const h = harness(false);
  h.controller.adoptCamera();
  h.document.activeElement = null;
  h.document.pointerLockElement = null;
  h.key("KeyW");
  close(h.walk(5).delta.toArray(), [0, 0, 0], 1e-12);
  // Focus alone is enough: pointer lock can be refused outright, and a walk that
  // needs a captured mouse is unusable from the keyboard.
  h.canvas.focus();
  h.key("KeyW");
  expect(h.walk(5).delta.length()).toBeCloseTo(5, 9);
  // Mouse look only applies while the pointer really is locked.
  const still = h.controller.update(1 / 60, 0).quaternion.clone();
  h.mouse(500, 0);
  expect(h.controller.update(1 / 60, 0).quaternion.angleTo(still)).toBeLessThan(1e-9);
  h.controller.requestPointerLock();
  h.mouse(500, 0);
  expect(h.controller.update(1 / 60, 0).quaternion.angleTo(still)).toBeGreaterThan(0.1);

  // Disabling releases everything and stops accepting input.
  h.controller.disable();
  h.key("KeyW");
  close(h.walk(5).delta.toArray(), [0, 0, 0], 1e-12);
});

// --- Hosted trajectories --------------------------------------------------

const BASE = "https://viewer.example.edu/hw1-demo/";

test("a hosted scene may declare a cameras.json, and only an http(s) .json one", () => {
  expect(resolveTrajectoryUrl("cameras.json", BASE)).toBe("https://viewer.example.edu/hw1-demo/cameras.json");
  for (const bad of ["", "javascript:alert(1)", "data:application/json,[]", "file:///tmp/cameras.json", "https://x.example/cameras.ply"])
    expect(() => resolveTrajectoryUrl(bad, BASE)).toThrow(HostedSceneError);

  const scenes = parseSceneManifest(JSON.stringify({
    version: 1,
    scenes: [
      { id: "with", name: "With path", url: "a-lod.rad", trajectoryUrl: "paths/cameras.json" },
      { id: "without", name: "No path", url: "b-lod.rad" },
      { id: "bad", name: "Bad path", url: "c-lod.rad", trajectoryUrl: "javascript:alert(1)" },
    ],
  }), BASE);
  expect(scenes.map((scene) => scene.trajectoryUrl)).toEqual([
    "https://viewer.example.edu/hw1-demo/paths/cameras.json",
    null,
    // A bad trajectory URL drops the trajectory, never the scene.
    null,
  ]);
  expect(scenes).toHaveLength(3);
});
