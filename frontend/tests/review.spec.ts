import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import type { ViewerSnapshot } from "../src/viewerDiagnostics";

const snapshot = (page: Page): Promise<ViewerSnapshot> => page.evaluate(() => window.__NEURAL3D__!());
const distance = (a: number[], b: number[]) => Math.hypot(...a.map((v, i) => v - b[i]));

test("resize fits untouched/reset cameras and preserves intentional orbit, keyboard and focus views", async ({ page }) => {
  await page.route("**/api/**", (route) => route.abort());
  await page.addInitScript(() => localStorage.setItem("neural3d:annotations:v1:spectrum-garden", JSON.stringify([{
    id: "resize-note", scene_id: "spectrum-garden", category: "Landmark", label: "Resize landmark", note: "",
    x: 1, y: 0, z: 0, created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z",
  }])));
  await page.goto("/");
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready");
  const wide = await snapshot(page);
  expect(wide.manual).toBe(false);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect.poll(async () => distance((await snapshot(page)).camera, wide.camera)).toBeGreaterThan(0.1);
  const narrow = await snapshot(page);
  expect(narrow.manual).toBe(false);
  expect(narrow.projected.every((p) => p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1)).toBe(true);
  await page.locator("canvas").scrollIntoViewIfNeeded();
  const box = (await page.locator("canvas").boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.7, box.y + box.height * 0.6, { steps: 8 });
  await page.mouse.up();
  await expect.poll(async () => (await snapshot(page)).manual).toBe(true);
  await expect(async () => {
    const before = await snapshot(page);
    await page.waitForTimeout(200); // Observe damping settling, not a load timing assumption.
    expect(distance((await snapshot(page)).camera, before.camera)).toBeLessThan(0.0001);
  }).toPass();
  const manual = await snapshot(page);
  await page.setViewportSize({ width: 1440, height: 1080 });
  await expect.poll(async () => page.locator("canvas").evaluate((canvas) => canvas.clientWidth)).toBeGreaterThan(700);
  expect(distance((await snapshot(page)).camera, manual.camera)).toBeLessThan(0.001);
  expect((await snapshot(page)).target).toEqual(manual.target);
  await page.getByRole("button", { name: "Reset camera" }).click();
  await expect.poll(async () => (await snapshot(page)).manual).toBe(false);
  expect(distance((await snapshot(page)).camera, wide.camera)).toBeLessThan(0.001);
  await page.locator("canvas").focus();
  await page.keyboard.press("ArrowLeft");
  expect((await snapshot(page)).manual).toBe(true);
  await page.getByRole("button", { name: "Reset camera" }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect.poll(async () => distance((await snapshot(page)).camera, narrow.camera)).toBeLessThan(0.001);
  await page.getByRole("button", { name: "Focus point 1: Resize landmark", exact: true }).click();
  const focused = await snapshot(page);
  expect(focused.manual).toBe(true);
  expect(focused.target).toEqual([1, 0, 0]);
  await page.setViewportSize({ width: 1440, height: 1080 });
  await expect.poll(async () => page.locator("canvas").evaluate((canvas) => canvas.clientWidth)).toBeGreaterThan(700);
  expect(distance((await snapshot(page)).camera, focused.camera)).toBeLessThan(0.001);
  expect((await snapshot(page)).target).toEqual(focused.target);
});

for (const damaged of ["{broken", "", "{}", '[{"id":"wrong-shape"}]']) test(`explicit storage recovery is scoped to current scene (${JSON.stringify(damaged)})`, async ({ page }) => {
  await page.route("**/api/**", (route) => route.abort());
  await page.addInitScript((value) => {
    if (!sessionStorage.getItem("fixture-set")) {
      localStorage.setItem("neural3d:annotations:v1:spectrum-garden", value);
      localStorage.setItem("neural3d:annotations:v1:ply-other-scene", "[]");
      localStorage.setItem("unrelated-application", "keep me");
      sessionStorage.setItem("fixture-set", "yes");
    }
  }, damaged);
  await page.goto("/");
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready");
  await expect(page.getByRole("alert")).toContainText("Local annotations could not be read");
  await expect(page.getByRole("button", { name: "+ Pick point", exact: true })).toBeDisabled();
  expect(await page.evaluate(() => localStorage.getItem("neural3d:annotations:v1:spectrum-garden"))).toBe(damaged);
  await page.getByRole("button", { name: "Reset this scene's local notes", exact: true }).click();
  await page.getByRole("button", { name: "Keep data", exact: true }).click();
  expect(await page.evaluate(() => localStorage.getItem("neural3d:annotations:v1:spectrum-garden"))).toBe(damaged);
  await page.getByRole("button", { name: "Reset this scene's local notes", exact: true }).click();
  await page.getByRole("button", { name: "Confirm reset local notes", exact: true }).click();
  await expect(page.getByRole("button", { name: "+ Pick point", exact: true })).toBeEnabled();
  await expect(page.getByText("Your observations start here")).toBeVisible();
  const remaining = await page.evaluate(() => ({ ...localStorage }));
  expect(remaining).toEqual({ "neural3d:annotations:v1:ply-other-scene": "[]", "unrelated-application": "keep me" });
  await expect(page.getByRole("button", { name: "Reset this scene's local notes", exact: true })).toHaveCount(0);
  await page.reload();
  await expect(page.getByRole("button", { name: "+ Pick point", exact: true })).toBeEnabled();
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("healthy or disabled storage does not offer destructive corruption recovery", async ({ page }) => {
  await page.route("**/api/**", (route) => route.abort());
  await page.goto("/");
  await expect(page.getByRole("button", { name: "+ Pick point", exact: true })).toBeEnabled();
  await expect(page.getByRole("button", { name: "Reset this scene's local notes", exact: true })).toHaveCount(0);
  await page.addInitScript(() => {
    Storage.prototype.getItem = () => { throw new DOMException("Disabled", "SecurityError"); };
  });
  await page.reload();
  await expect(page.getByRole("alert")).toContainText("Browser storage is unavailable");
  await expect(page.getByRole("button", { name: "Reset this scene's local notes", exact: true })).toHaveCount(0);
});
