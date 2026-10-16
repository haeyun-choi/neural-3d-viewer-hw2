import { createHash } from "node:crypto";
import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { gaussianPly, plainPointCloudPly } from "./fixtures/gaussianSplat";
import type { SplatSnapshot, ViewerSnapshot } from "../src/viewerDiagnostics";

// Decoding runs Spark's WebAssembly parser and its splat sort worker under
// software WebGL2, which is far slower than a GPU: allow for it explicitly.
test.setTimeout(240_000);
const READY = { timeout: 120_000 };
const poll = { timeout: 120_000 };

declare global {
  interface Window { __OBJECT_URLS__?: { created: number; revoked: number } }
}

const fingerprint = (buffer: Buffer) => createHash("sha256").update(buffer).digest("hex");
const splatSnapshot = (page: Page): Promise<SplatSnapshot> => page.evaluate(() => window.__NEURAL3D_SPLAT__!());
const pointSnapshot = (page: Page): Promise<ViewerSnapshot> => page.evaluate(() => window.__NEURAL3D__!());

async function ready(page: Page) {
  await page.goto("/");
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready", READY);
}
async function openSplat(page: Page, buffer: Buffer, name = "ring.ply") {
  await page.getByLabel("Open Gaussian Splat file").setInputFiles({ name, mimeType: "application/octet-stream", buffer });
  await expect(page.getByTestId("splat-viewport")).toHaveAttribute("data-state", "ready", READY);
  await expect.poll(async () => (await splatSnapshot(page)).fingerprint, READY).toBe(fingerprint(buffer));
}
const button = (page: Page, name: string) => page.getByRole("button", { name, exact: true });
/** The orientation controls live in a <details> that starts collapsed. */
async function orientation(page: Page) {
  if (!(await page.locator(".orientation-controls").getAttribute("open")) && !(await button(page, "Apply roll").isVisible()))
    await page.locator(".orientation-controls summary").click();
}
async function openPointCloud(page: Page, buffer: Buffer, name = "cloud.ply") {
  await page.getByLabel("Open PLY file").setInputFiles({ name, mimeType: "application/octet-stream", buffer });
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready", READY);
  await expect.poll(async () => (await pointSnapshot(page)).fingerprint, READY).toBe(fingerprint(buffer));
}

test("a Gaussian Splat PLY opens as a splat scene, with its bytes kept in the browser", async ({ page }) => {
  const uploads: string[] = [];
  page.on("request", (request) => { if (request.postData()) uploads.push(`${request.url()} ${request.postData()}`); });
  await ready(page);
  const splat = gaussianPly({ count: 12, shDegree: 1 });
  await openSplat(page, splat, "Campus ring.PLY");

  await expect(page.getByTestId("representation-badge")).toContainText("Gaussian Splat");
  await expect(page.getByTestId("representation-badge")).toContainText("SPARK");
  await expect(page.getByText("12 splats · Ready")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Campus ring", exact: true })).toBeVisible();
  await expect(page.locator(".scene-index")).toHaveText("LOCAL SPLAT");
  await expect(page.getByText("Local file - not uploaded", { exact: true })).toBeVisible();
  await expect(page.getByText("Annotations paused", { exact: true })).toBeVisible();

  const metadata = page.locator(".metadata-grid");
  await expect(metadata).toContainText("Gaussian Splat · 3DGS PLY binary LE · SH degree 1");
  await expect(metadata).toContainText("Gaussian Splat · Spark");
  await expect(metadata).toContainText("Campus ring.PLY");
  await expect(metadata).toContainText(`${(splat.byteLength / 1024).toFixed(1)} KiB`);
  await expect(metadata).toContainText("Not available for Gaussian Splats yet");

  const snapshot = await splatSnapshot(page);
  expect(snapshot.splats).toBe(12);
  expect(snapshot.fileType).toBe("ply");
  // Exactly one renderer canvas is mounted, and the point viewer is gone.
  expect(snapshot.canvases).toBe(1);
  await expect(page.getByTestId("viewport")).toHaveCount(0);

  // No splat bytes, identity or metadata reach the API.
  expect(uploads.some((body) => body.includes(snapshot.fingerprint) || body.includes("packed_position") || body.includes("f_dc_0"))).toBe(false);
  expect(uploads.some((body) => body.includes("/scenes/splat-"))).toBe(false);
});

test("annotation is clearly disabled for splats and untouched for point clouds", async ({ page }) => {
  await ready(page);
  await openSplat(page, gaussianPly());

  await expect(page.getByTestId("annotation-unsupported")).toContainText("not available for Gaussian Splats yet");
  await expect(page.getByTestId("gaussian-annotation-notice")).toContainText("not available for Gaussian Splats yet");
  await expect(page.getByRole("button", { name: "+ Pick point", exact: true })).toBeDisabled();
  for (const mode of ["Navigate", "Point", "Rectangle", "Lasso"])
    await expect(page.getByRole("button", { name: mode, exact: true })).toHaveCount(0);
  await expect(page.getByRole("group", { name: "Navigation and annotation mode" })).toHaveCount(0);
  await expect(page.locator(".point-size-control")).toHaveCount(0);

  // Returning to a point cloud restores the full annotation workflow.
  await page.getByRole("button", { name: "Use sample", exact: true }).click();
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready", READY);
  await expect(page.getByTestId("annotation-unsupported")).toHaveCount(0, READY);
  await expect(page.getByTestId("gaussian-annotation-notice")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "+ Pick point", exact: true })).toBeEnabled();
  await expect(page.getByRole("button", { name: "Rectangle", exact: true })).toBeEnabled();
  await expect(page.locator(".point-size-control")).toBeVisible();
});

test("orbit, reset, presets, resize and background all work in a Gaussian scene", async ({ page }) => {
  await ready(page);
  await openSplat(page, gaussianPly({ count: 16 }));
  const fitted = await splatSnapshot(page);
  expect(fitted.manual).toBe(false);
  expect(fitted.camera.every(Number.isFinite)).toBe(true);
  expect(Math.hypot(...fitted.camera.map((v, i) => v - fitted.target[i]))).toBeGreaterThan(0);

  const canvas = page.locator("canvas");
  const box = (await canvas.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 160, box.y + box.height / 2 + 60, { steps: 12 });
  await page.mouse.up();
  await expect.poll(async () => (await splatSnapshot(page)).manual, poll).toBe(true);
  const orbited = await splatSnapshot(page);
  expect(orbited.camera).not.toEqual(fitted.camera);

  await page.getByRole("button", { name: "↺ Reset camera" }).click();
  await expect.poll(async () => (await splatSnapshot(page)).camera.map((v) => v.toFixed(3)).join(), poll)
    .toBe(fitted.camera.map((v) => v.toFixed(3)).join());

  await orientation(page);
  await button(page, "Top").click();
  await expect.poll(async () => (await splatSnapshot(page)).camera.join(), poll).not.toBe(fitted.camera.join());
  await button(page, "Z up").click();
  await expect.poll(async () => (await splatSnapshot(page)).cameraUp.map(Math.round).join()).toBe("0,0,1");

  await page.getByLabel("Background").fill("#204030");
  await expect.poll(async () => (await splatSnapshot(page)).background, poll).toBe("204030");

  await page.setViewportSize({ width: 800, height: 900 });
  await expect.poll(async () => (await splatSnapshot(page)).camera.every(Number.isFinite), poll).toBe(true);
  await expect(page.getByTestId("splat-viewport")).toHaveAttribute("data-state", "ready", READY);
});

test("switching representations repeatedly keeps one canvas and releases the previous renderer", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  await ready(page);
  const splat = gaussianPly({ count: 10 });
  const cloud = plainPointCloudPly({ count: 8 });

  for (let round = 0; round < 3; round++) {
    await openSplat(page, splat);
    expect(await page.locator("canvas").count()).toBe(1);
    const inSplat = await splatSnapshot(page);
    expect(inSplat.canvases).toBe(1);
    expect(await page.evaluate(() => typeof window.__NEURAL3D__)).toBe("undefined");

    await openPointCloud(page, cloud);
    expect(await page.locator("canvas").count()).toBe(1);
    expect(await page.evaluate(() => typeof window.__NEURAL3D_SPLAT__)).toBe("undefined");
    const inCloud = await pointSnapshot(page);
    // Only the current point cloud's own buffers remain allocated.
    expect(inCloud.geometries).toBeLessThanOrEqual(3);
    expect(inCloud.displayedPoints).toBe(8);
  }
  expect(errors).toEqual([]);
});

test("object URLs created while importing and rendering splats are always revoked", async ({ page }) => {
  await page.addInitScript(() => {
    window.__OBJECT_URLS__ = { created: 0, revoked: 0 };
    const create = URL.createObjectURL.bind(URL);
    const revoke = URL.revokeObjectURL.bind(URL);
    URL.createObjectURL = (object: Blob | MediaSource) => { window.__OBJECT_URLS__!.created++; return create(object as Blob); };
    URL.revokeObjectURL = (url: string) => { window.__OBJECT_URLS__!.revoked++; revoke(url); };
  });
  await ready(page);
  await openSplat(page, gaussianPly());
  await openSplat(page, gaussianPly({ count: 14 }), "second.ply");
  await page.getByRole("button", { name: "Use sample", exact: true }).click();
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready", READY);
  const urls = await page.evaluate(() => window.__OBJECT_URLS__!);
  expect(urls.revoked).toBe(urls.created);
});

test("a failed import never disturbs the scene that is already displayed", async ({ page }) => {
  await ready(page);
  const cloud = plainPointCloudPly({ count: 8 });
  await openPointCloud(page, cloud, "Retained.ply");

  // A Gaussian Splat PLY must not be parsed as a point cloud.
  await page.getByLabel("Open PLY file").setInputFiles({ name: "splat.ply", mimeType: "application/octet-stream", buffer: gaussianPly() });
  await expect(page.getByRole("alert")).toContainText("Use Open Gaussian", READY);
  await expect(page.getByRole("heading", { name: "Retained", exact: true })).toBeVisible();
  expect((await pointSnapshot(page)).fingerprint).toBe(fingerprint(cloud));

  // A point cloud must not be opened through the Gaussian importer.
  await page.getByLabel("Open Gaussian Splat file").setInputFiles({ name: "cloud.ply", mimeType: "application/octet-stream", buffer: cloud });
  await expect(page.getByRole("alert")).toContainText("Use Open PLY", READY);
  expect((await pointSnapshot(page)).fingerprint).toBe(fingerprint(cloud));
  await expect(page.getByTestId("splat-viewport")).toHaveCount(0);

  // The same rule applies once a splat scene is on screen.
  const splat = gaussianPly({ count: 9 });
  await openSplat(page, splat);
  await page.getByLabel("Open Gaussian Splat file").setInputFiles({ name: "cut.ply", mimeType: "application/octet-stream", buffer: gaussianPly({ truncate: 12 }) });
  await expect(page.getByRole("alert")).toContainText("truncated", READY);
  await expect(page.getByTestId("splat-viewport")).toHaveAttribute("data-state", "ready", READY);
  expect((await splatSnapshot(page)).fingerprint).toBe(fingerprint(splat));
  await page.getByLabel("Open Gaussian Splat file").setInputFiles({ name: "scene.obj", mimeType: "application/octet-stream", buffer: splat });
  await expect(page.getByRole("alert")).toContainText("Choose a Gaussian Splat file", READY);
  expect((await splatSnapshot(page)).fingerprint).toBe(fingerprint(splat));
});

test("drag and drop routes each .ply to the representation its contents belong to", async ({ page }) => {
  await ready(page);
  const drop = async (buffer: Buffer, name: string) => {
    const data = await page.evaluateHandle(
      ([bytes, filename]) => {
        const transfer = new DataTransfer();
        transfer.items.add(new File([new Uint8Array(bytes as number[])], filename as string));
        return transfer;
      },
      [Array.from(buffer), name] as const,
    );
    await page.getByRole("region", { name: "Scene workspace" }).dispatchEvent("drop", { dataTransfer: data });
    await data.dispose();
  };
  const splat = gaussianPly({ count: 11 });
  await drop(splat, "Dropped splat.ply");
  await expect(page.getByTestId("splat-viewport")).toHaveAttribute("data-state", "ready", READY);
  expect((await splatSnapshot(page)).fingerprint).toBe(fingerprint(splat));

  const cloud = plainPointCloudPly({ count: 7 });
  await drop(cloud, "Dropped cloud.ply");
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready", READY);
  expect((await pointSnapshot(page)).fingerprint).toBe(fingerprint(cloud));
});

test("the same splat file always resolves to the same scene, with per-scene camera settings", async ({ page }) => {
  await ready(page);
  const splat = gaussianPly({ count: 13 });
  await openSplat(page, splat);
  await orientation(page);
  await button(page, "Z up").click();
  await button(page, "Set as front").click();
  await expect(page.getByText("Front/home saved in this browser.")).toBeVisible();

  const other = gaussianPly({ count: 13, phase: 0.75 });
  await openSplat(page, other, "other.ply");
  expect((await splatSnapshot(page)).fingerprint).not.toBe(fingerprint(splat));
  // A different file must not inherit the first file's saved view.
  await expect(page.locator(".orientation-controls summary")).toContainText("Imported orientation");

  await openSplat(page, splat, "renamed.ply");
  await expect(page.locator(".orientation-controls summary")).toContainText("Front/home saved");
  await expect.poll(async () => (await splatSnapshot(page)).cameraUp.map(Math.round).join(), poll).toBe("0,0,1");
});
