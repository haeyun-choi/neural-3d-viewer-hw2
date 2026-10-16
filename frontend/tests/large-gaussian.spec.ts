import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { plainPointCloudPly, writeLargeCompressedGaussianPly } from "./fixtures/gaussianSplat";
import { GAUSSIAN_LARGE_SCENE } from "../src/gaussianFormat";
import type { SplatSnapshot, ViewerSnapshot } from "../src/viewerDiagnostics";

// Decoding two million splats and building a level-of-detail tree runs Spark's
// WebAssembly parser and sorter under software WebGL2, which is far slower than
// a GPU. These budgets are generous on purpose.
test.setTimeout(600_000);
const READY = { timeout: 300_000 };

declare global {
  interface Window {
    __BLOB_READS__?: { arrayBuffer: number[]; stream: number[] };
  }
}

/** Just over the large-scene splat threshold, so the real large path is taken. */
const LARGE_SPLATS = GAUSSIAN_LARGE_SCENE.splats + 1;
let fixture: { path: string; sizeBytes: number; count: number };

test.beforeAll(() => {
  const directory = mkdtempSync(join(process.env.TMPDIR ?? tmpdir(), "neural3d-large-"));
  fixture = writeLargeCompressedGaussianPly(join(directory, "campus-walk.ply"), LARGE_SPLATS);
  // Well under the byte threshold, so splat count alone must trigger the large path.
  expect(fixture.sizeBytes).toBeLessThan(GAUSSIAN_LARGE_SCENE.bytes);
});

const splatSnapshot = (page: Page): Promise<SplatSnapshot> => page.evaluate(() => window.__NEURAL3D_SPLAT__!());
const pointSnapshot = (page: Page): Promise<ViewerSnapshot> => page.evaluate(() => window.__NEURAL3D__!());
const reads = (page: Page) => page.evaluate(() => window.__BLOB_READS__!);

/**
 * Record every Blob read the page makes. Spark is fed through Blob.stream(), so
 * a whole-file Blob.arrayBuffer() call is exactly the failure this milestone
 * fixes: it is what made the real 3.13 GB point_cloud.ply crash the tab.
 */
async function watchBlobReads(page: Page) {
  await page.addInitScript(() => {
    window.__BLOB_READS__ = { arrayBuffer: [], stream: [] };
    const arrayBuffer = Blob.prototype.arrayBuffer;
    Blob.prototype.arrayBuffer = function patched(this: Blob) {
      window.__BLOB_READS__!.arrayBuffer.push(this.size);
      return arrayBuffer.call(this);
    };
    const stream = Blob.prototype.stream;
    Blob.prototype.stream = function patched(this: Blob) {
      window.__BLOB_READS__!.stream.push(this.size);
      return stream.call(this);
    };
  });
}

async function ready(page: Page) {
  await page.goto("/");
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready", READY);
}

test("a large Gaussian import is streamed: no whole-file arrayBuffer, on-load LoD, bounded identity", async ({ page }) => {
  await watchBlobReads(page);
  await ready(page);

  await page.getByLabel("Open Gaussian Splat file").setInputFiles(fixture.path);

  // The three import phases are reported separately while the import runs.
  const phases = page.getByTestId("gaussian-progress");
  await expect(phases).toBeVisible({ timeout: 60_000 });
  await expect(phases.locator('li[data-phase="streaming"]')).toContainText("Streaming");
  await expect(phases.locator('li[data-phase="decoding"]')).toContainText("Decoding");
  await expect(phases.locator('li[data-phase="lod"]')).toContainText("Level of detail");
  // The previous scene stays on screen for the whole import.
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready");

  await expect(page.getByTestId("splat-viewport")).toHaveAttribute("data-state", "ready", READY);

  const blobReads = await reads(page);
  // The file itself is only ever streamed.
  expect(blobReads.stream).toContain(fixture.sizeBytes);
  expect(blobReads.arrayBuffer).not.toContain(fixture.sizeBytes);
  // Nothing anywhere in the import read a blob as large as the file.
  expect(Math.max(0, ...blobReads.arrayBuffer)).toBeLessThan(fixture.sizeBytes);

  const snapshot = await splatSnapshot(page);
  expect(snapshot.splats).toBe(LARGE_SPLATS);
  expect(snapshot.identity).toBe("bounded-sample");
  expect(snapshot.origin).toBe("local-file");
  expect(snapshot.lod).toBe(true);
  expect(snapshot.paged).toBe(false);
  // An on-load LoD build keeps the tree instead of the original splats, and a
  // tree over N leaves has more nodes than N because of its interior levels:
  // Spark logs "Tiny LoD: 2000001 -> 2966702" for this fixture. What is reported
  // to the user stays the source count; what Spark decoded is the tree.
  expect(snapshot.decodedSplats).toBeGreaterThan(LARGE_SPLATS);
  expect(snapshot.camera.every(Number.isFinite)).toBe(true);

  await expect(page.getByTestId("gaussian-identity")).toContainText("Bounded fingerprint");
  await expect(page.getByTestId("gaussian-delivery")).toContainText("level-of-detail tree built on load");
  await expect(page.locator(".viewport-badge")).toContainText("level of detail");
});

test("cancelling a large Gaussian import keeps the scene that is already displayed", async ({ page }) => {
  await watchBlobReads(page);
  await ready(page);
  const cloud = plainPointCloudPly({ count: 8 });
  await page.getByLabel("Open PLY file").setInputFiles({ name: "Retained.ply", mimeType: "application/octet-stream", buffer: cloud });
  // Wait for this scene, not merely for a ready viewport: the bundled sample is
  // already ready, so a bare state check would pass before the import lands.
  await expect(page.getByRole("heading", { name: "Retained", exact: true })).toBeVisible(READY);
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready", READY);
  const before = await pointSnapshot(page);
  expect(before.displayedPoints).toBe(8);

  await page.getByLabel("Open Gaussian Splat file").setInputFiles(fixture.path);
  await expect(page.getByTestId("gaussian-progress")).toBeVisible({ timeout: 60_000 });
  await page.getByRole("button", { name: "Cancel import", exact: true }).click();

  await expect(page.getByTestId("gaussian-progress")).toHaveCount(0, READY);
  // The point cloud is still the scene, unchanged, and no splat viewer appeared.
  await expect(page.getByTestId("splat-viewport")).toHaveCount(0);
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready");
  await expect(page.getByRole("heading", { name: "Retained", exact: true })).toBeVisible();
  expect((await pointSnapshot(page)).fingerprint).toBe(before.fingerprint);
  expect((await pointSnapshot(page)).displayedPoints).toBe(8);

  // Cancelling never turned into a whole-file read either.
  const blobReads = await reads(page);
  expect(Math.max(0, ...blobReads.arrayBuffer)).toBeLessThan(fixture.sizeBytes);
});
