import { createHash } from "node:crypto";
import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import type { ViewerSnapshot } from "../src/viewerDiagnostics";
import { meshPly, vertices } from "./fixtures/ply";

const hash = (data: Buffer) => createHash("sha256").update(data).digest("hex");
const snapshot = (page: Page): Promise<ViewerSnapshot> => page.evaluate(() => window.__NEURAL3D__!());
const pickFile = (page: Page, buffer: Buffer, name = "Synthetic mesh.ply") => page.getByLabel("Open PLY file").setInputFiles({ name, mimeType: "application/octet-stream", buffer });
async function ready(page: Page) {
  await page.goto("/");
  await expect(page.getByText("6,453 points · Ready")).toBeVisible();
}
async function opened(page: Page, data: Buffer) {
  await expect.poll(async () => (await snapshot(page)).fingerprint).toBe(hash(data));
  await expect(page.getByRole("button", { name: "+ Pick point", exact: true })).toBeEnabled();
}

// Only header reports are enlarged; the real parser always receives tiny data.
// A captured, delayed worker result exercises cancellation even if queued before
// terminate(). No large File/ArrayBuffer or production test override is needed.
async function instrumentWorker(page: Page) {
  await page.addInitScript(() => {
    const observations = { parses: 0, workers: 0, delayed: false, delivered: false };
    Object.assign(window, { importObservations: observations });
    const NativeWorker = Worker;
    window.Worker = class extends NativeWorker {
      filename = "";
      active = true;
      constructor(url: string | URL, options?: WorkerOptions) {
        super(url, options);
        observations.workers++;
        this.addEventListener("message", (event) => {
          if (event.data.kind === "header" && /^(Large|Very large)/.test(this.filename)) {
            Object.defineProperty(event, "data", { value: { ...event.data, summary: {
              ...event.data.summary, sizeBytes: this.filename.startsWith("Very") ? 1024 ** 3 + 1 : 256 * 1024 ** 2 + 1,
            } } });
          }
          if (event.data.kind === "result" && this.filename === "Delayed.ply") {
            event.stopImmediatePropagation();
            observations.delayed = true;
            const callback = this.onmessage;
            setTimeout(() => { callback?.call(this, event); observations.delivered = true; }, 2000);
          }
        });
      }
      override postMessage(message: unknown, options?: StructuredSerializeOptions | Transferable[]) {
        const request = message as { kind: string; file?: File };
        if (request.kind === "inspect") this.filename = request.file?.name ?? "";
        if (request.kind === "parse") observations.parses++;
        if (Array.isArray(options)) super.postMessage(message, options);
        else super.postMessage(message, options);
      }
      override terminate() {
        if (this.active) { observations.workers--; this.active = false; }
        super.terminate();
      }
    };
  });
}
const observations = (page: Page) => page.evaluate(() => (window as unknown as { importObservations: { parses: number; workers: number; delayed: boolean; delivered: boolean } }).importObservations);

test("binary mesh shows only original vertices; picking, note restoration, fit and failed imports remain correct", async ({ page }, testInfo) => {
  await page.route("**/api/**", (route) => route.abort());
  await ready(page);
  const mesh = meshPly();
  await pickFile(page, mesh);
  await opened(page, mesh);
  const state = await snapshot(page);
  expect(state.displayedPoints).toBe(5);
  expect(state.faceCount).toBe(8);
  expect(state.sourceIndexCount).toBe(24);
  expect(state.indexCount).toBe(0);
  expect(state.fallbackColor).toBe("7dd3fc");
  expect(state.projected.every((p) => p.x > 0 && p.x < 1 && p.y > 0 && p.y < 1)).toBe(true);
  await expect(page.getByText("Mesh PLY detected. Displaying 5 unique vertices as a point cloud; 8 faces are ignored.")).toBeVisible();
  await expect(page.locator(".metadata-grid")).toContainText("PLY · binary LE");
  await expect(page.locator(".metadata-grid")).toContainText("Fallback · sky blue");
  await page.getByRole("button", { name: "+ Pick point", exact: true }).click();
  const box = (await page.locator("canvas").boundingBox())!;
  await page.mouse.click(box.x + state.projected[0].x * box.width, box.y + state.projected[0].y * box.height);
  await page.getByLabel("Category", { exact: true }).fill("Structure");
  await page.getByLabel("Label required").fill("Mesh landmark");
  await page.getByRole("button", { name: "Save annotation", exact: true }).click();
  await expect(page.getByText("Annotation saved in this browser only.")).toBeVisible();
  const key = `neural3d:annotations:v1:ply-${hash(mesh)}`;
  const records = await page.evaluate((key) => JSON.parse(localStorage.getItem(key)!) as { x: number; y: number; z: number }[], key);
  expect(vertices).toContainEqual([records[0].x, records[0].y, records[0].z]);
  await page.getByRole("button", { name: "Highlight group Structure: Mesh landmark", exact: true }).click();
  expect((await snapshot(page)).markers[0].selected).toBe(true);
  await page.getByRole("button", { name: "Focus point 1: Mesh landmark", exact: true }).click();
  expect((await snapshot(page)).target).toEqual([records[0].x, records[0].y, records[0].z]);
  await page.getByRole("button", { name: "Reset camera" }).click();
  expect((await snapshot(page)).manual).toBe(false);
  await page.screenshot({ path: testInfo.outputPath("mesh-desktop.png"), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect.poll(async () => (await snapshot(page)).projected.every((p) => p.x > 0 && p.x < 1 && p.y > 0 && p.y < 1)).toBe(true);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath("mesh-mobile.png"), fullPage: true });
  await pickFile(page, mesh.subarray(0, -1), "Truncated faces.ply");
  await expect(page.getByRole("alert")).toContainText("Malformed PLY");
  expect((await snapshot(page)).fingerprint).toBe(hash(mesh));
  await expect(page.getByText("Mesh landmark", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Use sample", exact: true }).click();
  await expect(page.getByText("6,453 points · Ready")).toBeVisible();
  await expect(page.getByText("Mesh landmark", { exact: true })).toHaveCount(0);
  await pickFile(page, mesh, "Renamed mesh.ply");
  await opened(page, mesh);
  await expect(page.getByText("Mesh landmark", { exact: true })).toBeVisible();
});

for (const tier of ["Large", "Very large"]) test(`${tier} warning: no full read before consent, Cancel preserves view, Continue loads`, async ({ page }, testInfo) => {
  await instrumentWorker(page);
  await ready(page);
  const original = await snapshot(page), before = await observations(page);
  const mesh = meshPly();
  await pickFile(page, mesh, `${tier} mesh.ply`);
  const warning = page.getByRole("alertdialog");
  await expect(warning).toContainText(`${tier} PLY`);
  await expect(warning).toContainText("substantially more memory");
  if (tier === "Very large") await expect(warning).toContainText("terminated");
  expect((await observations(page)).parses).toBe(before.parses);
  expect((await snapshot(page)).fingerprint).toBe(original.fingerprint);
  await page.screenshot({ path: testInfo.outputPath("warning-desktop.png"), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await warning.scrollIntoViewIfNeeded();
  await page.screenshot({ path: testInfo.outputPath("warning-mobile.png"), fullPage: true });
  await warning.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(warning).toHaveCount(0);
  expect((await snapshot(page)).fingerprint).toBe(original.fingerprint);
  expect((await observations(page)).workers).toBe(0);
  await pickFile(page, mesh, `${tier} mesh.ply`);
  await warning.getByRole("button", { name: "Load full resolution", exact: true }).click();
  await opened(page, mesh);
  expect((await observations(page)).parses).toBe(before.parses + 1);
  expect((await observations(page)).workers).toBe(0);
});

test("cancel during parsing and newer import ignore stale worker results and release workers", async ({ page }) => {
  await instrumentWorker(page);
  await ready(page);
  const original = (await snapshot(page)).fingerprint;
  await pickFile(page, meshPly(), "Delayed.ply");
  await expect.poll(async () => (await observations(page)).delayed).toBe(true);
  await page.getByRole("button", { name: "Cancel import", exact: true }).click();
  expect((await observations(page)).workers).toBe(0);
  expect((await snapshot(page)).fingerprint).toBe(original);
  const replacement = meshPly({ colors: true });
  await pickFile(page, replacement, "Newer.ply");
  await opened(page, replacement);
  await expect.poll(async () => (await observations(page)).delivered).toBe(true);
  expect((await snapshot(page)).fingerprint).toBe(hash(replacement));
  expect((await snapshot(page)).vertexColors).toBe(true);
  await expect(page.locator("canvas")).toHaveCount(1);
  expect((await observations(page)).workers).toBe(0);
});

test("GPU allocation failure keeps the previous initialized scene and allows retry", async ({ page }) => {
  await ready(page);
  const original = await snapshot(page);
  await page.evaluate(() => {
    const original = WebGL2RenderingContext.prototype.bufferData;
    WebGL2RenderingContext.prototype.bufferData = function () {
      WebGL2RenderingContext.prototype.bufferData = original;
      throw new RangeError("Simulated allocation failure");
    };
  });
  const mesh = meshPly();
  await pickFile(page, mesh);
  await expect(page.getByRole("alert")).toContainText("could not load this PLY at full resolution");
  await expect(page.getByText("6,453 points · Ready")).toBeVisible();
  expect((await snapshot(page)).fingerprint).toBe(original.fingerprint);
  expect((await snapshot(page)).geometries).toBe(original.geometries);
  await pickFile(page, mesh);
  await opened(page, mesh);
});

test("cancelling before any valid scene restores usable sample controls", async ({ page }) => {
  await instrumentWorker(page);
  await page.route("**/data/spectrum-garden.ply", (route) => route.abort());
  await page.goto("/");
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "error");
  await pickFile(page, meshPly(), "Large initial.ply");
  await page.getByRole("alertdialog").getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(page.getByText("Import cancelled. Open a file or reload the sample.")).toBeVisible();
  expect((await observations(page)).workers).toBe(0);
  await page.unroute("**/data/spectrum-garden.ply");
  await page.getByRole("button", { name: "Use sample", exact: true }).click();
  await expect(page.getByText("6,453 points · Ready")).toBeVisible();
});

test("unmount releases a pending worker, controls listeners, observer, canvas and GPU buffers", async ({ page }) => {
  await instrumentWorker(page);
  await page.addInitScript(() => {
    const buffers = new Set<WebGLBuffer>();
    let observers = 0;
    const tracked = new Map<EventTarget, Set<EventListenerOrEventListenerObject>>();
    const add = EventTarget.prototype.addEventListener, remove = EventTarget.prototype.removeEventListener;
    EventTarget.prototype.addEventListener = function (type, listener, options) {
      if (this instanceof HTMLCanvasElement && listener) {
        if (!tracked.has(this)) tracked.set(this, new Set());
        tracked.get(this)!.add(listener);
      }
      return add.call(this, type, listener, options);
    };
    EventTarget.prototype.removeEventListener = function (type, listener, options) {
      if (listener) tracked.get(this)?.delete(listener);
      return remove.call(this, type, listener, options);
    };
    const create = WebGL2RenderingContext.prototype.createBuffer, dispose = WebGL2RenderingContext.prototype.deleteBuffer;
    WebGL2RenderingContext.prototype.createBuffer = function () { const buffer = create.call(this); if (buffer) buffers.add(buffer); return buffer; };
    WebGL2RenderingContext.prototype.deleteBuffer = function (buffer) { if (buffer) buffers.delete(buffer); dispose.call(this, buffer); };
    const NativeObserver = ResizeObserver;
    window.ResizeObserver = class extends NativeObserver {
      constructor(callback: ResizeObserverCallback) { super(callback); observers++; }
      disconnect() { observers--; super.disconnect(); }
    };
    Object.assign(window, { lifecycleCounts: () => ({ buffers: buffers.size, observers, listeners: [...tracked.values()].reduce((n, set) => n + set.size, 0) }) });
  });
  // Capture the real React root in test-only served code, without adding an
  // application debug action or relying on React's private internals.
  await page.route("**/src/main.tsx", async (route) => {
    const response = await route.fetch();
    const original = await response.text();
    const body = original.replace(/createRoot\((document.getElementById\(["']root["']\))\)/, "(window.testRoot = createRoot($1))");
    expect(body).not.toBe(original);
    await route.fulfill({ response, body });
  });
  await ready(page);
  await pickFile(page, meshPly(), "Large unmount.ply");
  await expect(page.getByRole("alertdialog")).toBeVisible();
  expect((await observations(page)).workers).toBe(1);
  await page.evaluate(() => (window as unknown as { testRoot: { unmount: () => void } }).testRoot.unmount());
  expect((await observations(page)).workers).toBe(0);
  await expect(page.locator("canvas")).toHaveCount(0);
  await expect.poll(() => page.evaluate(() => (window as unknown as { lifecycleCounts: () => unknown }).lifecycleCounts())).toEqual({ buffers: 0, observers: 0, listeners: 0 });
});
