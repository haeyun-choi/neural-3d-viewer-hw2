import { createHash } from "node:crypto";
import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { asymmetricPly, asymmetricPoints } from "./fixtures/asymmetric";
import { presetBasis } from "../src/CameraNavigation";
import type { Annotation, ViewPreferences, ViewPreset } from "../src/types";
import type { ViewerSnapshot } from "../src/viewerDiagnostics";

const hash = (file: Buffer) => createHash("sha256").update(file).digest("hex");
const snapshot = (page: Page): Promise<ViewerSnapshot> => page.evaluate(() => window.__NEURAL3D__!());
const button = (page: Page, name: string) => page.getByRole("button", { name, exact: true });
async function ready(page: Page) {
  await page.goto("/"); await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready");
}
async function open(page: Page, buffer: Buffer, name = "Asymmetric.ply") {
  await page.getByLabel("Open PLY file").setInputFiles({ name, mimeType: "application/octet-stream", buffer });
  await expect.poll(async () => (await snapshot(page)).fingerprint).toBe(hash(buffer));
  await expect(button(page, "+ Pick point")).toBeEnabled();
}
async function orientation(page: Page) {
  if (!(await page.locator(".orientation-controls").getAttribute("open")) && !(await button(page, "Apply roll").isVisible()))
    await page.locator(".orientation-controls summary").click();
}
async function preset(page: Page, name = "Front") { await orientation(page); await button(page, name).click(); }
async function rectangle(page: Page, box = [0.05, 0.05, 0.95, 0.5], modifier?: "Shift" | "Alt") {
  await button(page, "Rectangle").click();
  await page.locator("canvas").scrollIntoViewIfNeeded();
  const canvas = (await page.locator("canvas").boundingBox())!;
  if (modifier) await page.keyboard.down(modifier);
  await page.mouse.move(canvas.x + canvas.width * box[0], canvas.y + canvas.height * box[1]);
  await page.mouse.down();
  await page.mouse.move(canvas.x + canvas.width * box[2], canvas.y + canvas.height * box[3], { steps: 4 });
  await expect(page.getByLabel("Selection outline")).toBeVisible();
  expect((await snapshot(page)).controlsEnabled).toBe(false);
  await page.mouse.up();
  if (modifier) await page.keyboard.up(modifier);
  await expect(page.getByLabel("Selection outline")).toHaveCount(0);
  expect((await snapshot(page)).controlsEnabled).toBe(true);
}
async function selectedIndices(page: Page, expected: number[]) {
  await expect.poll(async () => (await snapshot(page)).highlights[0]?.indices ?? []).toEqual(expected);
}
function closeArray(actual: number[], expected: number[]) { actual.forEach((n, i) => expect(n).toBeCloseTo(expected[i], 6)); }
async function save(page: Page, label = "Upper part") {
  await page.getByLabel("Category", { exact: true }).fill("Part");
  await page.getByLabel("Label required").fill(label);
  await page.getByLabel("Note optional").fill("Keep the entire selected region together.");
  await button(page, "Save annotation").click();
  await expect(page.getByText(/Annotation saved/)).toBeVisible();
}

test("roll, upright, complete home basis, six presets, original coordinates, scene isolation and restore", async ({ page }) => {
  await page.route("**/api/**", (route) => route.abort());
  await ready(page);
  const first = asymmetricPly("orientation"), second = asymmetricPly("different");
  await open(page, first);
  const original = await snapshot(page);
  await orientation(page);
  await page.getByLabel("Roll angle (degrees)").fill("37");
  await button(page, "Apply roll").click();
  const rolled = await snapshot(page);
  expect(rolled.cameraUp).not.toEqual(original.cameraUp);
  closeArray(rolled.camera, original.camera);
  await button(page, "Save upright").click();
  await expect(page.locator(".orientation-controls summary")).toContainText("Upright saved");
  await button(page, "Set as front").click();
  const home = await snapshot(page);
  const key = `neural3d:view:v1:ply-${hash(first)}`;
  const saved = await page.evaluate((key) => JSON.parse(localStorage.getItem(key)!) as ViewPreferences, key);
  closeArray(saved.home!.up, rolled.cameraUp);
  for (const name of ["Front", "Back", "Left", "Right", "Top", "Bottom"] as ViewPreset[]) {
    await button(page, name).click();
    const pose = await snapshot(page), basis = presetBasis(saved, name);
    const offset = pose.camera.map((n, i) => n - pose.target[i]), length = Math.hypot(...offset);
    closeArray(offset.map((n) => n / length), basis.direction.toArray());
    closeArray(pose.cameraUp, basis.up.toArray());
  }
  await button(page, "Reset view").click();
  closeArray((await snapshot(page)).camera, home.camera);
  closeArray((await snapshot(page)).cameraUp, home.cameraUp);
  await button(page, "+ Pick point").click();
  await page.locator("canvas").scrollIntoViewIfNeeded();
  const box = (await page.locator("canvas").boundingBox())!, projected = (await snapshot(page)).projected[0];
  await page.mouse.click(box.x + box.width * projected.x, box.y + box.height * projected.y);
  await expect(page.locator(".selected-count")).toHaveText("1 point selected");
  await save(page, "Attached point");
  await button(page, "Invert up").click();
  closeArray((await snapshot(page)).markers[0].position, asymmetricPoints[0]);
  await button(page, "Focus point 1: Attached point").click();
  closeArray((await snapshot(page)).target, asymmetricPoints[0]);
  await open(page, second);
  closeArray((await snapshot(page)).cameraUp, [0, 1, 0]);
  await expect(page.locator(".orientation-controls summary")).toContainText("Imported orientation");
  await open(page, first);
  closeArray((await snapshot(page)).camera, home.camera);
  closeArray((await snapshot(page)).cameraUp, home.cameraUp);
  await page.reload(); await open(page, first);
  closeArray((await snapshot(page)).camera, home.camera);
  await orientation(page); await button(page, "Restore imported orientation").click();
  closeArray((await snapshot(page)).camera, original.camera);
  closeArray((await snapshot(page)).cameraUp, [0, 1, 0]);
  expect(await page.evaluate((key) => localStorage.getItem(key), key)).toBeNull();
  await expect(button(page, "Focus point 1: Attached point")).toBeVisible();
});

for (const local of [false, true]) test(`region rectangle/operations/grouping/CRUD/replay/scene isolation (${local ? "local" : "database"})`, async ({ page }, info) => {
  if (local) await page.route("**/api/**", (route) => route.abort());
  await ready(page);
  const file = asymmetricPly(`region-${local}`);
  await open(page, file); await preset(page);
  const original = await snapshot(page);
  await rectangle(page); await selectedIndices(page, [0, 1, 2]);
  closeArray((await snapshot(page)).camera, original.camera); // drag never orbited
  expect((await snapshot(page)).colors).toEqual(original.colors);
  await rectangle(page, [0.05, 0.5, 0.5, 0.95], "Shift"); await selectedIndices(page, [0, 1, 2, 3]);
  await rectangle(page, [0.5, 0.05, 0.95, 0.5], "Alt"); await selectedIndices(page, [0, 1, 3]);
  await save(page);
  const records = local
    ? await page.evaluate((key) => JSON.parse(localStorage.getItem(key)!) as Annotation[], `neural3d:annotations:v1:ply-${hash(file)}`)
    : await (await page.request.get(`/api/scenes/ply-${hash(file)}/annotations`)).json() as Annotation[];
  expect(records).toHaveLength(1); expect(records[0].kind).toBe("region");
  expect(records[0].region!.selected_count).toBe(3);
  expect(records[0].region!.operations).toHaveLength(3);
  expect(JSON.stringify(records).length).toBeLessThan(3000);
  await button(page, "Navigate").click();
  await expect.poll(async () => (await snapshot(page)).highlights.length).toBe(0);
  expect((await snapshot(page)).colors).toEqual(original.colors);
  await button(page, "Focus region 1: Upper part").click(); await selectedIndices(page, [0, 1, 3]);
  const focused = await snapshot(page);
  expect(Math.hypot(...focused.camera.map((n, i) => n - focused.target[i]))).toBeLessThan(Math.hypot(...original.camera.map((n, i) => n - original.target[i])));
  await button(page, "Apply roll").click(); await selectedIndices(page, [0, 1, 3]);
  await page.getByLabel("Label required").fill("Edited region"); await button(page, "Save changes").click();
  await button(page, "Navigate").click(); await button(page, "+ Pick point").click();
  await page.locator("canvas").scrollIntoViewIfNeeded();
  const box = (await page.locator("canvas").boundingBox())!, point = (await snapshot(page)).projected[0];
  await page.mouse.click(box.x + point.x * box.width, box.y + point.y * box.height);
  await save(page, " edited REGION ");
  const group = page.getByTestId("annotation-group");
  await expect(group).toHaveCount(1); await expect(group).toContainText("1 point · 1 region");
  await group.getByRole("button", { name: /Highlight group/ }).click();
  await selectedIndices(page, [0, 1, 3]);
  const grouped = await snapshot(page);
  expect(grouped.markers.every((m) => m.selected && m.color === grouped.highlights[0].color)).toBe(true);
  await button(page, "Focus region 1: Edited region").click();
  await page.getByLabel("Category", { exact: true }).fill("Wing"); await button(page, "Save changes").click();
  await expect(group).toHaveCount(2);
  await page.screenshot({ path: info.outputPath(`region-${local}-desktop.png`), fullPage: true });
  await open(page, asymmetricPly("different-region")); await expect(page.getByText("Your observations start here")).toBeVisible();
  expect((await snapshot(page)).highlights).toHaveLength(0);
  await open(page, file); await button(page, "Focus region 1: Edited region").click(); await selectedIndices(page, [0, 1, 3]);
  await page.reload(); await open(page, file);
  await button(page, "Focus region 1: Edited region").click(); await selectedIndices(page, [0, 1, 3]);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: info.outputPath(`region-${local}-mobile.png`), fullPage: true });
  await button(page, "Delete Edited region").click(); await button(page, "Confirm delete").click();
  await expect(button(page, "Focus region 1: Edited region")).toHaveCount(0);
  expect((await snapshot(page)).highlights).toHaveLength(0);
  expect((await snapshot(page)).colors).toEqual(original.colors);
});

test("lasso, empty selection, Escape cancellation and explicit selection operation", async ({ page }) => {
  await page.route("**/api/**", (route) => route.abort());
  await ready(page); await open(page, asymmetricPly("lasso")); await preset(page);
  await rectangle(page, [0.01, 0.01, 0.08, 0.08]);
  await expect(page.getByText("No points selected. Drag over the cloud to select a region.")).toBeVisible();
  await expect(button(page, "Save annotation")).toHaveCount(0);
  await button(page, "Lasso").click(); await page.locator("canvas").scrollIntoViewIfNeeded();
  const box = (await page.locator("canvas").boundingBox())!;
  const move = async (x: number, y: number) => page.mouse.move(box.x + box.width * x, box.y + box.height * y, { steps: 3 });
  await move(0.05, 0.05); await page.mouse.down(); await move(0.95, 0.05); await move(0.95, 0.5); await move(0.05, 0.5); await page.mouse.up();
  await selectedIndices(page, [0, 1, 2]);
  await move(0.2, 0.2); await page.mouse.down(); await move(0.8, 0.8);
  await page.keyboard.press("Escape"); await page.mouse.up();
  await expect(page.getByLabel("Selection outline")).toHaveCount(0);
  expect((await snapshot(page)).controlsEnabled).toBe(true); await selectedIndices(page, [0, 1, 2]);
  await page.getByLabel("Selection operation").selectOption("subtract");
  await rectangle(page, [0.5, 0.05, 0.95, 0.5]); await selectedIndices(page, [0, 1]);
});

test("selection failure, cancellation and scene changes discard stale asynchronous results", async ({ page }) => {
  await page.route("**/api/**", (route) => route.abort());
  await page.route("**/src/regionSelection.ts", async (route) => {
    const response = await route.fetch(); const body = await response.text();
    const changed = body.replace("const count = positions.length / 3;", 'if (window.pauseSelection) await new Promise(r => setTimeout(r, 1500)); if (window.failSelection) throw new Error("Injected allocation failure"); const count = positions.length / 3;');
    expect(changed).not.toBe(body); await route.fulfill({ response, body: changed });
  });
  await ready(page); const file = asymmetricPly("cancel-region"); await open(page, file); await preset(page);
  await rectangle(page); await selectedIndices(page, [0, 1, 2]);
  await page.evaluate(() => Object.assign(window, { failSelection: true }));
  await rectangle(page, [0.01, 0.01, 0.99, 0.99]); await expect(page.getByRole("alert")).toContainText("Previous selection preserved");
  await selectedIndices(page, [0, 1, 2]);
  await page.evaluate(() => Object.assign(window, { failSelection: false, pauseSelection: true }));
  await rectangle(page, [0.01, 0.01, 0.99, 0.99]); await button(page, "Cancel selection").click();
  await selectedIndices(page, [0, 1, 2]);
  await rectangle(page, [0.01, 0.01, 0.99, 0.99]); await open(page, asymmetricPly("newer-scene"));
  await expect(button(page, "Cancel selection")).toHaveCount(0);
  // An explicit timeout here allows the injected delayed result to arrive.
  await page.waitForTimeout(1700);
  expect((await snapshot(page)).highlights).toHaveLength(0);
  await expect(page.locator(".selected-count")).toHaveText("0 points selected");
});

test("corrupt region data and view preferences recover only their own scene keys", async ({ page }) => {
  await page.route("**/api/**", (route) => route.abort());
  const file = asymmetricPly("corrupt-region"), scene = `ply-${hash(file)}`;
  await page.addInitScript(({ scene }) => {
    localStorage.setItem(`neural3d:view:v1:${scene}`, "broken");
    localStorage.setItem(`neural3d:annotations:v1:${scene}`, '[{"kind":"region","region":{}}]');
    localStorage.setItem("unrelated", "keep");
  }, { scene });
  await ready(page);
  await page.getByLabel("Open PLY file").setInputFiles({ name: "Corrupt.ply", mimeType: "application/octet-stream", buffer: file });
  await expect.poll(async () => (await snapshot(page)).fingerprint).toBe(hash(file));
  await expect(page.locator(".orientation-controls").getByRole("alert")).toContainText("preserved");
  await button(page, "Restore imported orientation").click();
  await button(page, "Reset this scene's local notes").click(); await button(page, "Confirm reset local notes").click();
  await expect(page.getByText("Your observations start here")).toBeVisible();
  expect(await page.evaluate(() => localStorage.getItem("unrelated"))).toBe("keep");
  await preset(page); await rectangle(page); await selectedIndices(page, [0, 1, 2]); await save(page);
});

test("axis choices, saved-home resize, repeated orientation/selection disposal and unmount", async ({ page }) => {
  test.setTimeout(120_000);
  await page.route("**/api/**", (route) => route.abort());
  await page.addInitScript(() => {
    const tracked = new Map<EventTarget, Set<EventListenerOrEventListenerObject>>(), buffers = new Set<WebGLBuffer>();
    const add = EventTarget.prototype.addEventListener, remove = EventTarget.prototype.removeEventListener;
    EventTarget.prototype.addEventListener = function (type, listener, options) {
      if (listener && (this instanceof HTMLCanvasElement || this === window || this === document)) {
        if (!tracked.has(this)) tracked.set(this, new Set()); tracked.get(this)!.add(listener);
      }
      return add.call(this, type, listener, options);
    };
    EventTarget.prototype.removeEventListener = function (type, listener, options) {
      if (listener) tracked.get(this)?.delete(listener); return remove.call(this, type, listener, options);
    };
    const create = WebGL2RenderingContext.prototype.createBuffer, dispose = WebGL2RenderingContext.prototype.deleteBuffer;
    WebGL2RenderingContext.prototype.createBuffer = function () { const b = create.call(this); if (b) buffers.add(b); return b; };
    WebGL2RenderingContext.prototype.deleteBuffer = function (b) { if (b) buffers.delete(b); dispose.call(this, b); };
    Object.assign(window, { regionResources: () => ({ buffers: buffers.size, listeners: [...tracked.values()].reduce((n, set) => n + set.size, 0) }) });
  });
  await page.route("**/src/main.tsx", async (route) => {
    const response = await route.fetch(), body = await response.text();
    await route.fulfill({ response, body: body.replace(/createRoot\((document.getElementById\(["']root["']\))\)/, "(window.testRoot = createRoot($1))") });
  });
  await ready(page); const file = asymmetricPly("lifecycle"); await open(page, file); await orientation(page);
  for (const [axis, up] of [["X", [1, 0, 0]], ["Z", [0, 0, 1]], ["Y", [0, 1, 0]]] as const) {
    await button(page, `${axis} up`).click(); closeArray((await snapshot(page)).cameraUp, [...up]);
  }
  await page.getByLabel("Roll angle (degrees)").fill("-123.5"); await button(page, "Apply roll").click();
  await button(page, "Set as front").click(); await button(page, "Reset view").click();
  const home = await snapshot(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect.poll(async () => (await snapshot(page)).camera).not.toEqual(home.camera);
  closeArray((await snapshot(page)).cameraUp, home.cameraUp);
  await page.setViewportSize({ width: 1440, height: 1080 });
  await expect.poll(async () => (await snapshot(page)).camera.map((n) => Number(n.toFixed(6)))).toEqual(home.camera.map((n) => Number(n.toFixed(6))));
  await button(page, "Apply roll").click(); const manual = await snapshot(page);
  await page.setViewportSize({ width: 1000, height: 1080 });
  closeArray((await snapshot(page)).camera, manual.camera);
  await page.setViewportSize({ width: 1440, height: 1080 });
  await button(page, "Restore imported orientation").click(); await preset(page);
  const counts = () => page.evaluate(() => (window as unknown as { regionResources: () => { buffers: number; listeners: number } }).regionResources());
  // Warm the shared Sprite quad and React's lazily registered form listeners.
  await rectangle(page); await selectedIndices(page, [0, 1, 2]); await button(page, "Navigate").click();
  const baseline = await counts(), baselineGeometries = (await snapshot(page)).geometries;
  const canvas = await page.locator("canvas").elementHandle();
  for (let i = 0; i < 4; i++) {
    await rectangle(page); await selectedIndices(page, [0, 1, 2]);
    await button(page, "Navigate").click();
    await button(page, "Apply roll").click(); await button(page, "Reset view").click();
    await button(page, "Use sample").click(); await expect(page.getByText("6,453 points · Ready")).toBeVisible();
    await open(page, file); await preset(page);
    await expect.poll(counts).toEqual(baseline);
    expect((await snapshot(page)).geometries).toBe(baselineGeometries);
  }
  await expect(page.locator("canvas")).toHaveCount(1);
  expect(await canvas!.evaluate((element) => element === document.querySelector("canvas"))).toBe(true);
  await rectangle(page); await selectedIndices(page, [0, 1, 2]);
  await page.evaluate(() => (window as unknown as { testRoot: { unmount: () => void } }).testRoot.unmount());
  await expect(page.locator("canvas")).toHaveCount(0);
  await expect.poll(async () => (await counts()).buffers).toBe(0);
  // React keeps its global document listener; all viewer/control listeners go.
  expect((await counts()).listeners).toBeLessThan(baseline.listeners - 10);
});

test("home preserves tight single-vertex region focus and up without distance clamping", async ({ page }) => {
  await page.route("**/api/**", (route) => route.abort());
  await ready(page); await open(page, asymmetricPly("tight-home")); await preset(page);
  const p = (await snapshot(page)).projected[0];
  await rectangle(page, [p.x - 0.015, p.y - 0.015, p.x + 0.015, p.y + 0.015]); await selectedIndices(page, [0]);
  await save(page, "Tiny part"); await button(page, "Focus region 1: Tiny part").click();
  await button(page, "Apply roll").click(); await button(page, "Set as front").click();
  const home = await snapshot(page);
  await button(page, "Back").click(); await button(page, "Reset view").click();
  closeArray((await snapshot(page)).camera, home.camera);
  closeArray((await snapshot(page)).cameraUp, home.cameraUp);
  const center = (await snapshot(page)).projected[0];
  expect(center.x).toBeCloseTo(0.5); expect(center.y).toBeCloseTo(0.5);
});

test("highlight GPU allocation failure preserves the previous region and cloud", async ({ page }) => {
  await page.route("**/api/**", (route) => route.abort());
  await page.addInitScript(() => {
    const upload = WebGL2RenderingContext.prototype.bufferData, error = WebGL2RenderingContext.prototype.getError;
    let pending = false;
    const state = { fail: false }; Object.assign(window, { highlightGpu: state });
    WebGL2RenderingContext.prototype.bufferData = new Proxy(upload, {
      apply(target, context, args) { Reflect.apply(target, context, args); if (state.fail) pending = true; },
    });
    WebGL2RenderingContext.prototype.getError = function () {
      if (pending) { pending = false; state.fail = false; return this.OUT_OF_MEMORY; }
      return error.call(this);
    };
  });
  await ready(page); await open(page, asymmetricPly("highlight-gpu")); await preset(page);
  await rectangle(page); await selectedIndices(page, [0, 1, 2]); const original = await snapshot(page);
  await page.evaluate(() => { (window as unknown as { highlightGpu: { fail: boolean } }).highlightGpu.fail = true; });
  await rectangle(page, [0.01, 0.01, 0.99, 0.99]);
  await expect(page.getByRole("alert")).toContainText("GPU memory");
  await selectedIndices(page, [0, 1, 2]);
  const after = await snapshot(page);
  expect(after.fingerprint).toBe(original.fingerprint); expect(after.colors).toEqual(original.colors);
  await rectangle(page, [0.01, 0.01, 0.99, 0.99]); await selectedIndices(page, [0, 1, 2, 3, 4, 5]);
});
