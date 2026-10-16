import { basename, dirname } from "node:path";
import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { startStaticHost } from "./fixtures/staticServer";
import type { StaticHost } from "./fixtures/staticServer";
import { cameraJson, lookAtCameras } from "./fixtures/cameras";
import type { SplatSnapshot } from "../src/viewerDiagnostics";

// Paging a real prebuilt level-of-detail asset needs a real asset, and a
// multi-gigabyte scene is deliberately not in this repository. Point these at a
// .rad built with Spark's build-lod tool (see docs/hosted-scenes.md):
//
//   HW1_RAD_SINGLE   single-file  …-lod.rad
//   HW1_RAD_CHUNKED  chunked      …-lod.rad beside …-lod-<n>.radc
//
// Without them the suite still runs; these cases are skipped and say so.
const SINGLE = process.env.HW1_RAD_SINGLE;
const CHUNKED = process.env.HW1_RAD_CHUNKED;

test.setTimeout(300_000);
const READY = { timeout: 180_000 };

const splatSnapshot = (page: Page): Promise<SplatSnapshot> => page.evaluate(() => window.__NEURAL3D_SPLAT__!());

async function ready(page: Page) {
  await page.goto("/");
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready", READY);
}

async function openHosted(page: Page, url: string) {
  await page.getByTestId("hosted-scenes").locator("summary").click();
  await page.getByLabel("RAD asset URL").fill(url);
  await page.getByRole("button", { name: "Open RAD URL", exact: true }).click();
}

let host: StaticHost | null = null;
test.afterEach(async () => {
  await host?.close();
  host = null;
});

test("a single-file RAD asset is paged over HTTP range requests", async ({ page }) => {
  test.skip(!SINGLE, "set HW1_RAD_SINGLE to a prebuilt single-file Spark .rad asset");
  host = await startStaticHost(dirname(SINGLE!));
  await ready(page);
  await openHosted(page, `${host.origin}/${basename(SINGLE!)}`);

  await expect(page.getByTestId("splat-viewport")).toHaveAttribute("data-state", "ready", READY);
  const snapshot = await splatSnapshot(page);
  expect(snapshot.paged).toBe(true);
  expect(snapshot.lod).toBe(true);
  expect(snapshot.origin).toBe("hosted-rad");
  expect(snapshot.identity).toBe("hosted-url");
  expect(snapshot.splats).toBeGreaterThan(0);
  // Pages are resident in the renderer's pool, not decoded up front.
  expect(snapshot.pagedSplats).toBeGreaterThan(0);
  expect(snapshot.camera.every(Number.isFinite)).toBe(true);

  const asset = `/${basename(SINGLE!)}`;
  const assetRequests = host.requests.filter((request) => request.path === asset);
  expect(assetRequests.length).toBeGreaterThan(1);
  expect(assetRequests.every((request) => request.range?.startsWith("bytes="))).toBe(true);
  // Every range is a slice; nothing asked for the whole asset in one go.
  expect(assetRequests.some((request) => request.range !== "bytes=0-65535")).toBe(true);

  await expect(page.getByTestId("gaussian-delivery")).toContainText("paged over HTTP");
  await expect(page.locator(".metadata-grid")).toContainText("HTTP-range chunks");
  // A hosted scene must not be labelled as a local file anywhere.
  await expect(page.locator(".scene-index")).toHaveText("HOSTED SPLAT");
  await expect(page.getByText("Hosted asset - paged, never uploaded", { exact: true })).toBeVisible();
  await expect(page.getByText("Local file - not uploaded", { exact: true })).toHaveCount(0);
  // The preflight learns the asset's length from Content-Range, so the size shown
  // is the real one; "Delivery" is what says it is paged rather than downloaded.
  await expect(page.locator(".metadata-grid")).toContainText(/File size\s*\d/);
});

test("a chunked RAD asset is paged from its .radc files and needs no range support", async ({ page }) => {
  test.skip(!CHUNKED, "set HW1_RAD_CHUNKED to a prebuilt chunked Spark .rad asset");
  host = await startStaticHost(dirname(CHUNKED!), { ranges: false });
  await ready(page);
  await openHosted(page, `${host.origin}/${basename(CHUNKED!)}`);

  await expect(page.getByTestId("splat-viewport")).toHaveAttribute("data-state", "ready", READY);
  const snapshot = await splatSnapshot(page);
  expect(snapshot.paged).toBe(true);
  expect(snapshot.pagedSplats).toBeGreaterThan(0);

  expect(host.requests.some((request) => request.path.endsWith(".radc"))).toBe(true);
  await expect(page.locator(".metadata-grid")).toContainText(".radc files");
});

test("a single-file RAD asset on a host without range support is refused before the scene changes", async ({ page }) => {
  test.skip(!SINGLE, "set HW1_RAD_SINGLE to a prebuilt single-file Spark .rad asset");
  host = await startStaticHost(dirname(SINGLE!), { ranges: false });
  await ready(page);
  await openHosted(page, `${host.origin}/${basename(SINGLE!)}`);

  await expect(page.getByRole("alert")).toContainText("Range", READY);
  await expect(page.getByTestId("splat-viewport")).toHaveCount(0);
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready");
});

test("a hosted scene manifest may declare a cameras.json, and the viewer loads it with the scene", async ({ page }) => {
  test.skip(!SINGLE, "set HW1_RAD_SINGLE to a prebuilt single-file Spark .rad asset");
  host = await startStaticHost(dirname(SINGLE!));
  const trajectoryUrl = "https://cameras.test/paths/cameras.json";
  const cameras = lookAtCameras(6, 4);
  // The manifest and the camera file are both served to the page directly, so
  // the test exercises the real manifest -> hosted scene -> trajectory path
  // without writing anything next to the RAD asset.
  await page.route("**/data/scenes.json", (route) => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({
      version: 1,
      scenes: [{
        id: "hosted-with-path",
        name: "Hosted scene with a camera path",
        url: `${host!.origin}/${basename(SINGLE!)}`,
        trajectoryUrl,
      }],
    }),
  }));
  await page.route(trajectoryUrl, (route) => route.fulfill({
    contentType: "application/json",
    body: cameraJson(cameras),
    headers: { "Access-Control-Allow-Origin": "*" },
  }));

  await ready(page);
  await expect(page.getByLabel("Published scene")).toHaveCount(1);
  await page.getByTestId("hosted-scenes").locator("summary").click();
  await page.getByRole("button", { name: "Open hosted scene", exact: true }).click();
  await expect(page.getByTestId("splat-viewport")).toHaveAttribute("data-state", "ready", READY);

  // The trajectory arrived on its own, with no "Open camera path" click.
  const panel = page.getByTestId("camera-path-panel");
  await expect(panel).toBeVisible(READY);
  // Six cameras from six distinct places, so every one is its own anchor.
  await expect(page.getByTestId("camera-path-summary")).toContainText("6 cameras · 6 trajectory anchors");
  await expect(panel).toContainText("cameras.json");
  await expect.poll(async () => (await splatSnapshot(page)).trajectoryAnchors, READY).toBe(6);
  await page.getByRole("button", { name: "Next anchor ▶", exact: true }).click();
  await expect(page.getByTestId("anchor-status")).toContainText("1 / 6 · station_000 · 1 camera");
  const at = await splatSnapshot(page);
  expect(at.paged).toBe(true);
  at.camera.forEach((value, i) => expect(Math.abs(value - cameras[0].position[i])).toBeLessThan(1e-4));
});

test("a URL that is not a RAD asset is refused without any network access", async ({ page }) => {
  await ready(page);
  await openHosted(page, "https://example.invalid/scene.spz");
  await expect(page.getByRole("alert")).toContainText("prebuilt Spark .rad");
  await expect(page.getByTestId("splat-viewport")).toHaveCount(0);
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready");
});
