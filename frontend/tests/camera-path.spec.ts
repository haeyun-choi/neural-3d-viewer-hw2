import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { gaussianPly, plainPointCloudPly } from "./fixtures/gaussianSplat";
import { cameraJson, cubeStationCameras } from "./fixtures/cameras";
import type { SplatSnapshot } from "../src/viewerDiagnostics";

// Spark decodes and sorts under software WebGL2 here, which is slow but works.
test.setTimeout(240_000);
const READY = { timeout: 120_000 };

const snapshot = (page: Page): Promise<SplatSnapshot> => page.evaluate(() => window.__NEURAL3D_SPLAT__!());
const button = (page: Page, name: string) => page.getByRole("button", { name, exact: true });

/**
 * Nine places along a line, six cube faces at each: the shape of the real
 * capture, scaled to the test scene. 54 raw cameras, 9 anchors, 0.75 between
 * them, so the default radius is 0.375 and the extent is 6.
 */
const STATIONS: [number, number, number][] = Array.from({ length: 9 }, (_, i) => [i * 0.75 - 3, 0, 0]);
const CAMERAS = cubeStationCameras(STATIONS);
const ANCHORS = STATIONS.length;
const RAW = CAMERAS.length;
const DEFAULT_RADIUS = 0.375;

async function ready(page: Page) {
  await page.goto("/");
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready", READY);
}

async function openSplat(page: Page, name = "ring.ply", buffer = gaussianPly({ count: 16 })) {
  await page.getByLabel("Open Gaussian Splat file").setInputFiles({ name, mimeType: "application/octet-stream", buffer });
  await expect(page.getByTestId("splat-viewport")).toHaveAttribute("data-state", "ready", READY);
}

async function openPath(page: Page, cameras: unknown, name = "cameras.json") {
  await page.getByLabel("Open camera path file").setInputFiles({
    name, mimeType: "application/json", buffer: Buffer.from(cameraJson(cameras)),
  });
}

async function withPath(page: Page, cameras: unknown = CAMERAS) {
  await ready(page);
  await openSplat(page);
  await openPath(page, cameras);
  await expect(page.getByTestId("camera-path-panel")).toBeVisible(READY);
}

/**
 * Scroll the canvas into view and return a point on it that is inside the
 * browser viewport. The path panel makes the page taller than the window, so
 * the canvas centre can sit below it and a click there would land outside the
 * page entirely.
 */
async function canvasView(page: Page) {
  const canvas = page.locator("canvas");
  await canvas.scrollIntoViewIfNeeded();
  const box = (await canvas.boundingBox())!;
  const viewport = page.viewportSize()!;
  const top = Math.max(box.y, 0);
  const bottom = Math.min(box.y + box.height, viewport.height);
  return { box, viewport, centre: { x: box.x + box.width / 2, y: (top + bottom) / 2 } };
}

/** An anchor whose marker is on screen and reachable by the mouse. */
async function pickableMarker(page: Page) {
  const { box, viewport } = await canvasView(page);
  const projected = (await snapshot(page)).trajectory.projected;
  // Anchors behind the viewer, or off the canvas, are not clickable by anyone.
  const index = projected.findIndex((marker) =>
    marker.onScreen && box.y + marker.y > 4 && box.y + marker.y < viewport.height - 4);
  expect(index, "at least one anchor marker must be on screen").toBeGreaterThanOrEqual(0);
  return { index, x: box.x + projected[index].x, y: box.y + projected[index].y };
}

/** Orbit the scene with the mouse, from a point guaranteed to be clickable. */
async function orbit(page: Page, dx: number, dy: number) {
  const { centre } = await canvasView(page);
  await page.mouse.move(centre.x, centre.y);
  await page.mouse.down();
  await page.mouse.move(centre.x + dx, centre.y + dy, { steps: 10 });
  await page.mouse.up();
}

/**
 * Enter the walk and hold keys for a while.
 *
 * The keyboard, not the mouse, drives these tests on purpose: pointer lock needs
 * a real user gesture and a focused window, is refused outright in some headless
 * configurations, and would swallow Playwright's synthetic mouse moves. The
 * controller accepts keys whenever the canvas has focus precisely so that the
 * walk is usable — and testable — without it.
 */
async function hold(page: Page, keys: string[], ms: number, samples = 0) {
  await page.locator("canvas").focus();
  for (const key of keys) await page.keyboard.down(key);
  const seen: SplatSnapshot[] = [];
  if (samples) for (let i = 0; i < samples; i++) {
    await page.waitForTimeout(Math.round(ms / samples));
    seen.push(await snapshot(page));
  } else await page.waitForTimeout(ms);
  for (const key of keys) await page.keyboard.up(key);
  return seen;
}

const unit = (a: number[]) => {
  const n = Math.hypot(...a);
  return a.map((v) => v / n);
};
const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
/** Where the camera is actually pointing. OrbitControls' target is stale while
 *  the walker drives, so the camera's own -z is the only honest reading. */
const look = (state: SplatSnapshot) => unit(state.forward);
const apart = (a: number[], b: number[]) => Math.hypot(...a.map((v, i) => v - b[i]));

test("948-style cube-face captures collapse to distinct places, and that is what is drawn", async ({ page }) => {
  const uploads: string[] = [];
  page.on("request", (request) => { if (request.postData()) uploads.push(request.url() + request.postData()); });
  await ready(page);
  await openSplat(page);
  // The control only exists for a Gaussian scene, and only after a path loads.
  await expect(page.getByTestId("camera-path-panel")).toHaveCount(0);
  await openPath(page, CAMERAS);

  const panel = page.getByTestId("camera-path-panel");
  await expect(panel).toBeVisible(READY);
  await expect(page.getByTestId("camera-path-summary"))
    .toContainText(`${RAW} cameras · ${ANCHORS} trajectory anchors`);
  await expect(panel).toContainText("cameras.json");
  await expect(page.getByTestId("anchor-status")).toContainText(`${ANCHORS} anchors`);

  await expect.poll(async () => (await snapshot(page)).trajectoryAnchors, READY).toBe(ANCHORS);
  const state = await snapshot(page);
  expect(state.trajectoryCameras).toBe(RAW);
  const drawn = state.trajectory;
  // The overlay is the collapsed path: one polyline vertex and one marker per
  // place, not one per raw pose.
  expect(drawn.anchors).toBe(ANCHORS);
  expect(drawn.vertices.path).toBe(ANCHORS);
  expect(drawn.vertices.markers).toBe(ANCHORS);
  expect(drawn.visible.path).toBe(true);
  expect(drawn.visible.markers).toBe(true);
  expect(drawn.projected).toHaveLength(ANCHORS);
  // The raw frustums are a debug aid: off, and not even built, until asked for.
  expect(drawn.visible.frustums).toBe(false);
  expect(drawn.frustumsBuilt).toBe(false);
  expect(drawn.vertices.frustums).toBe(0);
  // Loading a path uploads nothing.
  expect(uploads.some((body) => body.includes("station_000") || body.includes("rotation"))).toBe(false);
});

test("there is no pose playback: no Play, no speed multiplier, no loop, no ordering", async ({ page }) => {
  await withPath(page);
  // The old interaction replayed all 54 poses, which spun through six cube faces
  // on the spot at every place. Nothing in the panel offers that any more.
  for (const name of ["▶ Play", "⏸ Pause", "Return to free navigation"])
    await expect(button(page, name)).toHaveCount(0);
  await expect(page.getByRole("checkbox", { name: "Loop" })).toHaveCount(0);
  await expect(page.getByLabel("Order")).toHaveCount(0);
  await expect(page.getByLabel("Speed", { exact: true })).toHaveCount(0);
  await expect(page.getByLabel("Camera", { exact: true })).toHaveCount(0);
  await expect(page.getByTestId("camera-path-panel")).not.toContainText("cameras per second");
});

test("the raw per-image frustums are available behind the debug toggle", async ({ page }) => {
  await withPath(page);
  await page.getByTestId("show-raw-frustums").check();
  await expect.poll(async () => (await snapshot(page)).trajectory.visible.frustums, READY).toBe(true);
  const shown = (await snapshot(page)).trajectory;
  // Ten line segments per frustum, two vertices each, for every raw camera —
  // one batched LineSegments, not 54 objects.
  expect(shown.vertices.frustums).toBe(RAW * 20);
  expect(shown.frustumsBuilt).toBe(true);
  expect(shown.anchors).toBe(ANCHORS);

  const before = shown.vertices.frustums;
  await page.getByLabel("Marker size").fill("3");
  await expect.poll(async () => (await snapshot(page)).trajectory.scale, READY).toBe(3);
  // Rebuilt at the new size rather than transforming per-camera objects.
  expect((await snapshot(page)).trajectory.vertices.frustums).toBe(before);

  await page.getByTestId("show-raw-frustums").uncheck();
  await expect.poll(async () => (await snapshot(page)).trajectory.visible.frustums, READY).toBe(false);
  await page.getByRole("checkbox", { name: "Show path and anchors" }).uncheck();
  await expect.poll(async () => (await snapshot(page)).trajectory.visible.path, READY).toBe(false);
  expect((await snapshot(page)).trajectory.visible.markers).toBe(false);
});

test("choosing an anchor moves where you stand and never turns you", async ({ page }) => {
  await withPath(page);
  const status = page.getByTestId("anchor-status");
  const before = await snapshot(page);
  expect(before.trajectory.selected).toBe(-1);

  await button(page, "Next anchor ▶").click();
  await expect(status).toContainText(`1 / ${ANCHORS} · station_000 · 6 cameras`);
  const first = await snapshot(page);
  // Exactly the place, and exactly the direction the user was already looking:
  // the six poses here are cube faces, and adopting one of them would spin the
  // view for no reason the user asked for.
  first.camera.forEach((value, i) => expect(Math.abs(value - STATIONS[0][i])).toBeLessThan(1e-4));
  expect(dot(look(first), look(before))).toBeGreaterThan(0.99999);
  // The field of view is the viewer's own; no training camera imposes one.
  expect(first.fov).toBeCloseTo(before.fov, 6);
  // Selection is marked by shape and size, not colour alone: a cross of three
  // world-space axes plus a larger dot.
  expect(first.trajectory.selected).toBe(0);
  expect(first.trajectory.visible.selected).toBe(true);
  expect(first.trajectory.vertices.selected).toBe(6);

  await button(page, "Next anchor ▶").click();
  await expect(status).toContainText(`2 / ${ANCHORS} · station_001`);
  await button(page, "◀ Previous anchor").click();
  await expect(status).toContainText(`1 / ${ANCHORS}`);
  // The ends clamp; there is no wrap-around to surprise anyone.
  await button(page, "◀ Previous anchor").click();
  await expect(status).toContainText(`1 / ${ANCHORS}`);

  const slider = page.getByLabel("Anchor", { exact: true });
  await slider.fill(String(ANCHORS - 3));
  await expect(status).toContainText(`${ANCHORS - 2} / ${ANCHORS} · station_00${ANCHORS - 3}`);
  await slider.focus();
  await page.keyboard.press("ArrowRight");
  await expect(status).toContainText(`${ANCHORS - 1} / ${ANCHORS}`);
  const late = await snapshot(page);
  expect(dot(look(late), look(before))).toBeGreaterThan(0.99999);
});

test("clicking an anchor marker in the scene goes to that place", async ({ page }) => {
  await withPath(page);
  await expect.poll(async () => (await snapshot(page)).trajectory.projected.length, READY).toBe(ANCHORS);

  const target = await pickableMarker(page);
  await page.mouse.click(target.x, target.y);
  await expect(page.getByTestId("anchor-status"))
    .toContainText(`${target.index + 1} / ${ANCHORS} · station_${String(target.index).padStart(3, "0")}`, READY);
  const at = await snapshot(page);
  at.camera.forEach((value, i) => expect(Math.abs(value - STATIONS[target.index][i])).toBeLessThan(1e-3));

  // A click on an empty corner of the canvas changes nothing.
  const { box } = await canvasView(page);
  await page.mouse.click(box.x + 8, Math.max(box.y, 0) + 8);
  await expect(page.getByTestId("anchor-status")).toContainText(`${target.index + 1} / ${ANCHORS}`);
});

test("Explore trajectory enters at the nearest point on the path, facing the same way", async ({ page }) => {
  await withPath(page);
  const before = await snapshot(page);
  // The overview camera starts well outside the tube.
  expect(before.controlsEnabled).toBe(true);
  expect(before.exploring).toBe(false);

  await page.getByTestId("explore-toggle").click();
  await expect.poll(async () => (await snapshot(page)).exploring, READY).toBe(true);
  const entered = await snapshot(page);
  // Inside the region, straight away, and still looking where the user was.
  expect(entered.tubeDistance).toBeLessThanOrEqual(entered.tubeRadius + 1e-6);
  expect(entered.tubeRadius).toBeCloseTo(DEFAULT_RADIUS, 6);
  expect(dot(look(entered), look(before))).toBeGreaterThan(0.999);
  // OrbitControls is off only while the walker owns the camera.
  expect(entered.controlsEnabled).toBe(false);
  expect(entered.activeSegment).toBeGreaterThanOrEqual(0);
  await expect(page.getByTestId("explore-status")).toContainText("Inside the trajectory tube");

  await page.getByTestId("explore-toggle").click();
  await expect.poll(async () => (await snapshot(page)).exploring, READY).toBe(false);
  await expect(page.getByTestId("explore-status")).toContainText("Not exploring");
  const left = await snapshot(page);
  expect(left.controlsEnabled).toBe(true);
  expect(left.activeSegment).toBe(-1);
  // Orbiting works again straight away.
  await orbit(page, -160, 40);
  await expect.poll(async () => (await snapshot(page)).camera.join(), READY).not.toBe(left.camera.join());
});

test("WASD walks the scene and the tube never lets the camera off the path", async ({ page }) => {
  await withPath(page);
  await page.getByTestId("explore-toggle").click();
  await expect.poll(async () => (await snapshot(page)).exploring, READY).toBe(true);
  const start = await snapshot(page);

  // Walk forward while turning, so the walk is not at the mercy of whichever
  // direction the overview camera happened to be facing. Turning means the
  // route curves back on itself, so how far it *got* is the furthest it reached,
  // not where it happened to stop.
  const samples = await hold(page, ["w", "ArrowLeft"], 6000, 6);
  const walked = await snapshot(page);
  expect(Math.max(...samples.map((s) => apart(s.camera, start.camera)))).toBeGreaterThan(DEFAULT_RADIUS * 2);
  for (const sample of samples) expect(sample.tubeDistance).toBeLessThanOrEqual(sample.tubeRadius + 1e-6);
  expect(walked.tubeDistance).toBeLessThanOrEqual(walked.tubeRadius + 1e-6);
  // Releasing the keys stops the walk rather than coasting.
  const stopped = await snapshot(page);
  await page.waitForTimeout(700);
  expect(apart((await snapshot(page)).camera, stopped.camera)).toBeLessThan(1e-6);

  // Strafing and rising are bounded by the same surface.
  await hold(page, ["d", "e"], 2500);
  const pressed = await snapshot(page);
  expect(pressed.tubeDistance).toBeLessThanOrEqual(pressed.tubeRadius + 1e-6);
  expect(pressed.tubeAtBoundary).toBe(true);
  await expect(page.getByTestId("explore-status")).toContainText("Trajectory boundary");
  // Held, not frozen: the along-path component of the push is still delivered.
  expect(apart(pressed.camera, walked.camera)).toBeGreaterThan(0);
});

test("the allowed radius is adjustable and takes effect under a standing camera", async ({ page }) => {
  await withPath(page);
  await page.getByTestId("explore-toggle").click();
  await expect.poll(async () => (await snapshot(page)).exploring, READY).toBe(true);

  // Press into the boundary at the default radius.
  await hold(page, ["e"], 2000);
  await expect.poll(async () => (await snapshot(page)).tubeAtBoundary, READY).toBe(true);
  const wide = await snapshot(page);
  expect(wide.tubeDistance).toBeCloseTo(DEFAULT_RADIUS, 3);

  // Shrinking the radius pulls the camera back in without it moving a key.
  await page.getByTestId("explore-radius").fill("0.1");
  await expect.poll(async () => (await snapshot(page)).tubeRadius, READY).toBeCloseTo(0.1, 6);
  await expect.poll(async () => (await snapshot(page)).tubeDistance, READY).toBeLessThanOrEqual(0.100001);
  expect((await snapshot(page)).exploring).toBe(true);

  // Widening it again lets the walker back out to the new surface.
  await page.getByTestId("explore-radius").fill("0.6");
  await hold(page, ["e"], 2000);
  await expect.poll(async () => (await snapshot(page)).tubeDistance, READY).toBeGreaterThan(0.3);
  expect((await snapshot(page)).tubeDistance).toBeLessThanOrEqual(0.600001);
});

test("Return to nearest anchor puts the walker back on the path", async ({ page }) => {
  await withPath(page);
  await page.getByTestId("explore-toggle").click();
  await expect.poll(async () => (await snapshot(page)).exploring, READY).toBe(true);
  await hold(page, ["e", "d"], 2500);
  await expect.poll(async () => (await snapshot(page)).tubeAtBoundary, READY).toBe(true);

  await page.getByTestId("return-to-path").click();
  await expect.poll(async () => (await snapshot(page)).tubeDistance, READY).toBeLessThan(1e-6);
  const home = await snapshot(page);
  expect(home.exploring).toBe(true);
  // It lands on a real place, not merely on the line between two of them.
  const nearest = Math.min(...STATIONS.map((station) => apart(home.camera, station)));
  expect(nearest).toBeLessThan(1e-4);

  // It works outside the walk too, and leaves OrbitControls in charge.
  await page.getByTestId("explore-toggle").click();
  await expect.poll(async () => (await snapshot(page)).exploring, READY).toBe(false);
  await orbit(page, 200, 60);
  await page.getByTestId("return-to-path").click();
  await expect.poll(async () => (await snapshot(page)).controlsEnabled, READY).toBe(true);
  const orbited = await snapshot(page);
  expect(Math.min(...STATIONS.map((station) => apart(orbited.camera, station)))).toBeLessThan(1e-3);
});

test("Escape leaves the walk, and so does loading another scene", async ({ page }) => {
  await withPath(page);
  await page.getByTestId("explore-toggle").click();
  await expect.poll(async () => (await snapshot(page)).exploring, READY).toBe(true);

  // The browser releases the pointer on the first Escape; with the pointer
  // already free, Escape leaves the mode.
  await page.locator("canvas").focus();
  await page.keyboard.press("Escape");
  await expect.poll(async () => (await snapshot(page)).exploring, READY).toBe(false);
  await expect(button(page, "Explore trajectory")).toBeVisible();
  expect((await snapshot(page)).controlsEnabled).toBe(true);

  await page.getByTestId("explore-toggle").click();
  await expect.poll(async () => (await snapshot(page)).exploring, READY).toBe(true);
  await openSplat(page, "other.ply", gaussianPly({ count: 12, phase: 0.6 }));
  await expect(page.getByTestId("camera-path-panel")).toHaveCount(0, READY);
  await expect.poll(async () => (await snapshot(page)).trajectoryAnchors, READY).toBe(0);
  expect((await snapshot(page)).exploring).toBe(false);
  expect((await snapshot(page)).controlsEnabled).toBe(true);
});

test("a malformed camera file is refused and the loaded path is preserved", async ({ page }) => {
  await withPath(page);
  await expect(page.getByTestId("camera-path-summary")).toContainText(`${ANCHORS} trajectory anchors`, READY);
  await button(page, "Next anchor ▶").click();
  await expect(page.getByTestId("anchor-status")).toContainText(`1 / ${ANCHORS}`);

  const broken = JSON.parse(JSON.stringify(CAMERAS)) as typeof CAMERAS;
  broken[2].rotation = [[2, 0, 0], [0, 2, 0], [0, 0, 2]];
  await openPath(page, broken, "broken.json");
  await expect(page.getByTestId("camera-path-error")).toContainText("Camera 2", READY);
  // Still the original path, still on the same place, scene untouched.
  await expect(page.getByTestId("camera-path-panel")).toContainText("cameras.json");
  await expect(page.getByTestId("anchor-status")).toContainText(`1 / ${ANCHORS}`);
  await expect(page.getByTestId("splat-viewport")).toHaveAttribute("data-state", "ready");
  expect((await snapshot(page)).trajectoryAnchors).toBe(ANCHORS);

  // A file that is not JSON at all, and an empty list, are refused the same way.
  await page.getByLabel("Open camera path file").setInputFiles({
    name: "notjson.json", mimeType: "application/json", buffer: Buffer.from("{nope"),
  });
  await expect(page.getByTestId("camera-path-error")).toContainText("not valid JSON", READY);
  await openPath(page, [], "empty.json");
  await expect(page.getByTestId("camera-path-error")).toContainText("no cameras", READY);
  expect((await snapshot(page)).trajectoryAnchors).toBe(ANCHORS);
});

test("a point cloud has no path control at all", async ({ page }) => {
  await withPath(page);
  await page.getByLabel("Open PLY file").setInputFiles({
    name: "cloud.ply", mimeType: "application/octet-stream", buffer: plainPointCloudPly({ count: 8 }),
  });
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready", READY);
  await expect(page.getByTestId("camera-path-panel")).toHaveCount(0);
  await expect(button(page, "Open camera path")).toHaveCount(0);
  expect(await page.evaluate(() => typeof window.__NEURAL3D_SPLAT__)).toBe("undefined");
});

test("the trajectory panel is usable at 390 px with no horizontal overflow", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await withPath(page);
  await button(page, "Next anchor ▶").click();
  await expect(page.getByTestId("anchor-status")).toContainText(`1 / ${ANCHORS}`);

  expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBe(0);
  // It collapses rather than covering the viewport. A closed <details> keeps its
  // children in the DOM, so visibility is what to assert here.
  await page.locator('[data-testid="camera-path-panel"] > summary').click();
  await expect(page.getByTestId("anchor-status")).not.toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBe(0);
  await expect(page.getByTestId("splat-viewport")).toHaveAttribute("data-state", "ready");
});
