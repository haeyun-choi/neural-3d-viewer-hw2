import { meshPly } from "./fixtures/ply";
import { createHash } from "node:crypto";
import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import type { ViewerSnapshot } from "../src/viewerDiagnostics";

const coordinates = [[0, 0, 0], [-1, -1, -1], [1, -1, -1], [-1, 1, 1], [1, 1, 1]];
function ascii(colored = true, comment = "fixture") {
  return Buffer.from(`ply\nformat ascii 1.0\ncomment ${comment}\nelement vertex 5\nproperty float x\nproperty float y\nproperty float z\n${colored ? "property uchar red\nproperty uchar green\nproperty uchar blue\n" : ""}end_header\n${coordinates.map((p, i) => `${p.join(" ")}${colored ? ` ${40 + i * 40} 160 210` : ""}`).join("\n")}\n`);
}
function binary(little: boolean) {
  const header = Buffer.from(`ply\nformat binary_${little ? "little" : "big"}_endian 1.0\nelement vertex 5\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n`);
  const body = Buffer.alloc(5 * 15);
  coordinates.forEach((point, i) => {
    point.forEach((value, axis) => little ? body.writeFloatLE(value, i * 15 + axis * 4) : body.writeFloatBE(value, i * 15 + axis * 4));
    body.set([100, 160, 210], i * 15 + 12);
  });
  return Buffer.concat([header, body]);
}
const fingerprint = (buffer: Buffer) => createHash("sha256").update(buffer).digest("hex");
const snapshot = (page: Page): Promise<ViewerSnapshot> => page.evaluate(() => window.__NEURAL3D__!());
async function ready(page: Page) {
  await page.goto("/");
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready");
}
async function open(page: Page, buffer: Buffer, name = "local.ply") {
  await page.getByLabel("Open PLY file").setInputFiles({ name, mimeType: "application/octet-stream", buffer });
  await expect.poll(async () => (await snapshot(page)).fingerprint).toBe(fingerprint(buffer));
  await expect(page.getByRole("button", { name: "Open PLY", exact: true })).toBeEnabled();
  await expect(page.getByRole("button", { name: "+ Pick point", exact: true })).toBeEnabled();
}
async function addNote(page: Page, label: string, category = "Structure", index = 0) {
  await page.getByRole("button", { name: "+ Pick point", exact: true }).click();
  await page.locator("canvas").scrollIntoViewIfNeeded();
  const box = (await page.locator("canvas").boundingBox())!;
  const point = (await snapshot(page)).projected[index];
  await page.mouse.click(box.x + point.x * box.width, box.y + point.y * box.height);
  await page.getByLabel("Category", { exact: true }).fill(category);
  await page.getByLabel("Label required").fill(label);
  await page.getByLabel("Note optional").fill(`Note for ${label}`);
  await page.getByRole("button", { name: "Save annotation", exact: true }).click();
  await expect(page.getByText(/Annotation saved/)).toBeVisible();
}

test("colored ASCII, metadata, fallback color, binary endianness, drag/drop and sample return", async ({ page }) => {
  const requests: string[] = [];
  page.on("request", (request) => { if (request.postData()) requests.push(request.postData()!); });
  await ready(page);
  const sampleHash = (await snapshot(page)).fingerprint;
  const colored = ascii(true, "never-send-this-file-content");
  await open(page, colored, "Colored garden.PLY");
  await expect(page.getByRole("heading", { name: "Colored garden", exact: true })).toBeVisible();
  await expect(page.getByText("Local file - not uploaded", { exact: true })).toBeVisible();
  await expect(page.locator(".scene-index")).toHaveText("LOCAL PLY");
  const metadata = page.locator(".metadata-grid");
  await expect(metadata).toContainText("Colored garden.PLY");
  await expect(metadata).toContainText("PLY · ASCII");
  await expect(metadata).toContainText("units unspecified");
  await expect(metadata).toContainText(`${(colored.byteLength / 1024).toFixed(1)} KiB`);
  await expect(page.getByText("5 points · Ready")).toBeVisible();
  expect((await snapshot(page)).vertexColors).toBe(true);
  expect(requests.every((body) => !body.includes("never-send-this-file-content") && !body.includes("positions"))).toBe(true);
  await open(page, ascii(false), "Uncolored.ply");
  expect((await snapshot(page)).vertexColors).toBe(false);
  expect((await snapshot(page)).fallbackColor).toBe("7dd3fc");
  for (const little of [true, false]) {
    await open(page, binary(little), "Binary.ply");
    await expect(metadata).toContainText(`binary ${little ? "LE" : "BE"}`);
    expect((await snapshot(page)).vertexColors).toBe(true);
  }
  const data = await page.evaluateHandle((text) => {
    const transfer = new DataTransfer();
    transfer.items.add(new File([text], "Dropped.ply"));
    return transfer;
  }, ascii().toString());
  await page.getByRole("region", { name: "Scene workspace" }).dispatchEvent("drop", { dataTransfer: data });
  await expect(page.getByRole("heading", { name: "Dropped", exact: true })).toBeVisible();
  await data.dispose();
  await page.getByRole("button", { name: "Use sample", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Spectrum Garden", exact: true })).toBeVisible();
  await expect.poll(async () => (await snapshot(page)).fingerprint).toBe(sampleHash);
  await expect(page.getByText("6,453 points · Ready")).toBeVisible();
  await expect(page.getByText("Local file - not uploaded", { exact: true })).toHaveCount(0);
});

test("invalid imports preserve the last valid geometry, metadata and notes", async ({ page }) => {
  await page.route("**/api/**", (route) => route.abort());
  await ready(page);
  await open(page, ascii(), "Retained.ply");
  await addNote(page, "Retained note");
  const original = await snapshot(page);
  const failures = [
    { name: "wrong.obj", buffer: ascii(), error: "Only .ply" },
    { name: "fake.ply", buffer: Buffer.from("not a PLY file"), error: "not a PLY" },
    { name: "truncated.ply", buffer: ascii().subarray(0, -30), error: "Malformed PLY" },
    { name: "empty.ply", buffer: Buffer.from(ascii().toString().replace("vertex 5", "vertex 0")), error: "no points" },
    { name: "counts.ply", buffer: Buffer.from(ascii().toString().replace("vertex 5", "vertex 6")), error: "Malformed PLY" },
  ];
  for (const file of failures) {
    await page.getByLabel("Open PLY file").setInputFiles({ name: file.name, mimeType: "application/octet-stream", buffer: file.buffer });
    await expect(page.getByRole("alert")).toContainText(file.error);
    await expect(page.getByRole("heading", { name: "Retained", exact: true })).toBeVisible();
    await expect(page.getByText("Retained note", { exact: true })).toBeVisible();
    await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready");
    expect((await snapshot(page)).fingerprint).toBe(original.fingerprint);
  }
  await open(page, ascii(false));
  await expect(page.getByRole("alert")).toHaveCount(0);
});

for (const local of [false, true]) test(`scene isolation and identical-file restoration (${local ? "localStorage" : "SQLite"})`, async ({ page }) => {
  if (local) await page.route("**/api/**", (route) => route.abort());
  await ready(page);
  const first = ascii(true, `isolation-${local}`);
  const second = ascii(false, `isolation-${local}`);
  await open(page, first, "First.ply");
  await addNote(page, "First scene note");
  await page.getByRole("button", { name: "Focus point 1: First scene note", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Edit annotation", exact: true })).toBeVisible();
  await open(page, second, "Second.ply");
  await expect(page.getByRole("heading", { name: "Edit annotation", exact: true })).toHaveCount(0);
  expect((await snapshot(page)).markers).toHaveLength(0);
  await expect(page.getByText("Your observations start here")).toBeVisible();
  await expect(page.getByText("First scene note", { exact: true })).toHaveCount(0);
  await addNote(page, "Second scene note");
  await open(page, first, "Renamed but identical.ply");
  await expect(page.getByText("First scene note", { exact: true })).toBeVisible();
  await expect(page.getByText("Second scene note", { exact: true })).toHaveCount(0);
  await page.reload();
  await expect(page.getByText("6,453 points · Ready")).toBeVisible();
  await expect(page.getByText("First scene note", { exact: true })).toHaveCount(0);
  await open(page, second, "Second.ply");
  await expect(page.getByText("Second scene note", { exact: true })).toBeVisible();
  if (local) {
    const keys = await page.evaluate(() => Object.keys(localStorage));
    expect(keys).toContain(`neural3d:annotations:v1:ply-${fingerprint(first)}`);
    expect(keys).toContain(`neural3d:annotations:v1:ply-${fingerprint(second)}`);
  }
});

test("the Modules section separates active work from TBD and has no interactive controls", async ({ page }) => {
  await ready(page);
  const section = page.getByRole("region", { name: "Modules" });
  // Point-cloud and Gaussian Splat viewing are implemented; the rest stay TBD.
  await expect(section.getByText("ACTIVE", { exact: true })).toHaveCount(2);
  await expect(section.getByText("3D Gaussian Splat Viewer", { exact: true })).toBeVisible();
  await expect(section.getByText("TBD · PLANNED")).toHaveCount(3);
  await expect(section.locator("button, a, input, [tabindex]")).toHaveCount(0);
});

test("normalized groups retain independent notes, expand, highlight, focus and regroup on edit", async ({ page }, testInfo) => {
  await page.route("**/api/**", (route) => route.abort());
  await ready(page);
  await open(page, ascii(), "Semantic groups.ply");
  await addNote(page, "Torus", "Structure", 1);
  await addNote(page, " torus ", " structure ", 2);
  await addNote(page, "TORUS", "Surface", 3);
  const groups = page.getByTestId("annotation-group");
  await expect(groups).toHaveCount(2);
  await expect(groups.first()).toContainText("2 points");
  await page.getByRole("button", { name: "Collapse Structure: Torus", exact: true }).click();
  await expect(page.getByRole("button", { name: "Focus point 1: Torus", exact: true })).toBeHidden();
  await page.getByRole("button", { name: "Expand Structure: Torus", exact: true }).click();
  await page.getByRole("button", { name: "Highlight group Structure: Torus", exact: true }).click();
  await expect.poll(async () => (await snapshot(page)).markers.filter((m) => m.selected).length).toBe(2);
  const markers = (await snapshot(page)).markers;
  expect(markers[0].color).toBe(markers[1].color);
  expect(await groups.first().evaluate((element) => (element as HTMLElement).style.getPropertyValue("--group-color"))).toBe(markers[0].color);
  await page.getByRole("button", { name: "Focus point 2: torus", exact: true }).click();
  await expect.poll(async () => (await snapshot(page)).markers.filter((m) => m.selected).length).toBe(1);
  expect((await snapshot(page)).target).toEqual(coordinates[2]);
  await expect(page.getByLabel("Note optional")).toHaveValue("Note for  torus");
  await expect(page.locator("#label-options option")).toHaveCount(1);
  await page.getByLabel("Category", { exact: true }).fill("Surface");
  await page.getByRole("button", { name: "Save changes", exact: true }).click();
  await expect(groups.first()).toContainText("1 point");
  await expect(groups.last()).toContainText("2 points");
  expect((await snapshot(page)).markers).toHaveLength(3);
  const records = await page.evaluate(() => JSON.parse(localStorage.getItem(Object.keys(localStorage)[0])!) as { note: string }[]);
  expect(records.map((record) => record.note)).toEqual(["Note for Torus", "Note for  torus", "Note for TORUS"]);
  await page.screenshot({ path: testInfo.outputPath("groups-desktop.png"), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath("groups-mobile.png"), fullPage: true });
});

test("repeated scene switching reuses one canvas, listeners, observer and WebGL resources", async ({ page }) => {
  test.setTimeout(120_000); // 24 complete imports with software WebGL and resource instrumentation.
  await page.route("**/api/**", (route) => route.abort());
  await page.addInitScript(() => {
    const registrations = new Map<EventTarget, Set<EventListenerOrEventListenerObject>>();
    const add = EventTarget.prototype.addEventListener;
    const remove = EventTarget.prototype.removeEventListener;
    EventTarget.prototype.addEventListener = function (type, callback, options) {
      if (callback && (this instanceof HTMLCanvasElement || this === document || this === window)) {
        if (!registrations.has(this)) registrations.set(this, new Set());
        registrations.get(this)!.add(callback);
      }
      return add.call(this, type, callback, options);
    };
    EventTarget.prototype.removeEventListener = function (type, callback, options) {
      if (callback) registrations.get(this)?.delete(callback);
      return remove.call(this, type, callback, options);
    };
    const buffers = new Set<WebGLBuffer>();
    const createBuffer = WebGL2RenderingContext.prototype.createBuffer;
    const deleteBuffer = WebGL2RenderingContext.prototype.deleteBuffer;
    WebGL2RenderingContext.prototype.createBuffer = function () {
      const buffer = createBuffer.call(this);
      if (buffer) buffers.add(buffer);
      return buffer;
    };
    WebGL2RenderingContext.prototype.deleteBuffer = function (buffer) {
      if (buffer) buffers.delete(buffer);
      return deleteBuffer.call(this, buffer);
    };
    let workers = 0, observers = 0;
    const NativeWorker = Worker, NativeObserver = ResizeObserver;
    window.Worker = class extends NativeWorker {
      active = true;
      constructor(url: string | URL, options?: WorkerOptions) { super(url, options); workers++; }
      terminate() { if (this.active) { workers--; this.active = false; } super.terminate(); }
    };
    window.ResizeObserver = class extends NativeObserver {
      constructor(callback: ResizeObserverCallback) { super(callback); observers++; }
      disconnect() { observers--; super.disconnect(); }
    };
    Object.assign(window, { resourceCounts: () => ({ listeners: [...registrations.values()].reduce((sum, set) => sum + set.size, 0), workers, observers, buffers: buffers.size }) });
  });
  await ready(page);
  await open(page, ascii());
  await addNote(page, "Repeated note");
  const canvas = await page.locator("canvas").elementHandle();
  const counts = () => page.evaluate(() => (window as unknown as { resourceCounts: () => unknown }).resourceCounts());
  const baseline = await counts();
  await expect.poll(async () => (await snapshot(page)).textures).toBe(1);
  const gpu = await snapshot(page);
  for (let i = 0; i < 6; i++) {
    await open(page, ascii(false), "Other.ply");
    await open(page, meshPly(), "Mesh.ply");
    expect((await snapshot(page)).displayedPoints).toBe(5);
    expect((await snapshot(page)).indexCount).toBe(0);
    await page.getByRole("button", { name: "Use sample", exact: true }).click();
    await expect(page.getByText("6,453 points · Ready")).toBeVisible();
    await open(page, ascii());
    await expect(page.getByText("Repeated note", { exact: true })).toBeVisible();
  }
  await expect(page.locator("canvas")).toHaveCount(1);
  expect(await canvas!.evaluate((element) => element === document.querySelector("canvas"))).toBe(true);
  await expect.poll(counts).toEqual(baseline);
  await expect.poll(async () => (await snapshot(page)).textures).toBe(gpu.textures);
  const final = await snapshot(page);
  expect(final.geometries).toBe(gpu.geometries);
  expect(final.programs).toBe(gpu.programs);
});
