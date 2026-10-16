import { expect, test } from "@playwright/test";
import { createHash } from "node:crypto";

test("a delayed previous scene response cannot replace current annotations or picking state", async ({ page }) => {
  const file = (name: string) => ({
    name: `${name}.ply`, mimeType: "application/octet-stream",
    buffer: Buffer.from(`ply\nformat ascii 1.0\ncomment ${name}\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n-1 -1 -1\n1 1 1\n`),
  });
  const first = file("Earlier"), second = file("Current");
  const oldId = `ply-${createHash("sha256").update(first.buffer).digest("hex")}`;
  let release: () => void = () => undefined;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  const endpoint = `**/api/scenes/${oldId}/annotations`;
  await page.route(endpoint, async (route) => {
    await gate;
    await route.fulfill({ json: [{
      id: "late-note", scene_id: oldId, category: "Landmark", label: "Previous scene only", note: "",
      x: 0, y: 0, z: 0, created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z",
    }] });
  });
  await page.goto("/");
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready");
  const pending = page.waitForRequest((request) => request.url().endsWith(`/scenes/${oldId}/annotations`));
  await page.getByLabel("Open PLY file").setInputFiles(first);
  await pending;
  await expect(page.getByRole("button", { name: "Open PLY", exact: true })).toBeEnabled();
  await page.getByLabel("Open PLY file").setInputFiles(second);
  await expect(page.getByRole("heading", { name: "Current", exact: true })).toBeVisible();
  await expect(page.getByText("Database connected", { exact: true })).toBeVisible();
  const response = page.waitForResponse((item) => item.url().endsWith(`/scenes/${oldId}/annotations`));
  release();
  await (await response).finished();
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => resolve())));
  await expect(page.getByText("Previous scene only", { exact: true })).toHaveCount(0);
  await expect(page.getByText("Your observations start here")).toBeVisible();
  await expect(page.getByRole("button", { name: "+ Pick point", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: "+ Pick point", exact: true }).click();
  await expect(page.getByRole("button", { name: "◎ Picking… click a point", exact: true })).toHaveAttribute("aria-pressed", "true");
});
