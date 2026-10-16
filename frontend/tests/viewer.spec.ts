import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";

async function ready(page: Page) {
  await page.goto("/");
  await expect(page.getByTestId("viewport")).toHaveAttribute(
    "data-state",
    "ready",
  );
  await expect(page.locator("canvas")).toBeVisible();
}

async function pickPoint(page: Page) {
  await page.getByRole("button", { name: "+ Pick point", exact: true }).click();
  await page.locator("canvas").scrollIntoViewIfNeeded();
  const box = (await page.locator("canvas").boundingBox())!;
  for (const [x, y] of [
    [0.6, 0.5],
    [0.4, 0.5],
    [0.5, 0.6],
    [0.6, 0.65],
    [0.4, 0.65],
  ]) {
    await page.mouse.click(box.x + box.width * x, box.y + box.height * y);
    if (await page.getByLabel("Label required").isVisible()) break;
  }
  await expect(page.getByLabel("Label required")).toBeVisible();
  await expect(page.getByLabel("Selected coordinates")).toBeVisible();
}

test("WebGL navigation, display controls, database CRUD and reload persistence", async ({
  page,
}, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await ready(page);
  await expect(page.getByText("Database connected")).toBeVisible();
  await expect(page.getByText("6,453 points · Ready")).toBeVisible();
  const canvas = page.locator("canvas");
  expect(
    await canvas.evaluate(
      (element: HTMLCanvasElement) =>
        !!element.getContext("webgl2") &&
        !element.getContext("webgl2")!.isContextLost(),
    ),
  ).toBe(true);
  const original = await canvas.screenshot();
  const box = (await canvas.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(
    box.x + box.width / 2 + 110,
    box.y + box.height / 2 + 30,
    { steps: 8 },
  );
  await page.mouse.up();
  await expect(async () =>
    expect((await canvas.screenshot()).equals(original)).toBe(false),
  ).toPass();
  await page.mouse.wheel(0, -180);
  await page.mouse.down({ button: "right" });
  await page.mouse.move(box.x + box.width / 2 + 160, box.y + box.height / 2, {
    steps: 5,
  });
  await page.mouse.up({ button: "right" });
  await page.getByRole("button", { name: "Reset camera" }).click();
  await page.getByLabel("Point size", { exact: true }).fill("5");
  await expect(page.locator("output")).toHaveText("5.0 px");
  await page.getByLabel("Background", { exact: true }).fill("#182134");
  await page.screenshot({
    path: testInfo.outputPath("viewer-desktop.png"),
    fullPage: true,
  });
  await pickPoint(page);
  await page.getByLabel("Label required").fill("Surface observation");
  await page.getByLabel("Category", { exact: true }).fill("Surface");
  await page.getByLabel("Note optional").fill("Saved through the API.");
  await page
    .getByRole("button", { name: "Save annotation", exact: true })
    .click();
  await expect(
    page.getByText("Annotation saved to the database."),
  ).toBeVisible();
  await page.reload();
  await expect(
    page.getByRole("button", { name: /Focus point \d+: Surface observation/ }).first(),
  ).toBeVisible();
  await page
    .getByRole("button", { name: /Focus point \d+: Surface observation/ })
    .first()
    .click();
  await expect(
    page.getByRole("heading", { name: "Edit annotation" }),
  ).toBeVisible();
  await page.getByLabel("Label required").fill("Edited observation");
  await page.getByRole("button", { name: "Save changes" }).click();
  await expect(
    page.getByText("Edited observation", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Delete Edited observation", exact: true })
    .click();
  await page.getByRole("button", { name: "Confirm delete" }).click();
  await expect(
    page.getByText("Edited observation", { exact: true }),
  ).toHaveCount(0);
  await page.reload();
  await expect(page.getByText("Your observations start here")).toBeVisible();
  expect(errors).toEqual([]);
});

test("backend unavailable: local CRUD survives reload, responsive viewport", async ({
  page,
}, testInfo) => {
  await page.route("**/api/**", (route) => route.abort());
  await page.setViewportSize({ width: 390, height: 844 });
  await ready(page);
  await expect(
    page.getByText("Local demo mode", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Static demo · saved in this browser only"),
  ).toBeVisible();
  await pickPoint(page);
  await page.getByLabel("Label required").fill("Local landmark");
  await page
    .getByRole("button", { name: "Save annotation", exact: true })
    .click();
  await expect(
    page.getByText("Annotation saved in this browser only."),
  ).toBeVisible();
  await page.reload();
  await expect(page.getByText("Local landmark", { exact: true })).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: testInfo.outputPath("viewer-mobile.png"),
    fullPage: true,
  });
  await page
    .getByRole("button", { name: /Focus point \d+: Local landmark/ })
    .first()
    .click();
  await page.getByLabel("Label required").fill("Local edited");
  await page.getByRole("button", { name: "Save changes" }).click();
  await expect(page.getByText("Local edited", { exact: true })).toBeVisible();
  await page
    .getByRole("button", { name: "Delete Local edited", exact: true })
    .click();
  await page.getByRole("button", { name: "Confirm delete" }).click();
  await page.reload();
  await expect(page.getByText("Your observations start here")).toBeVisible();
});

test("a failed database write preserves the form and database mode", async ({
  page,
}) => {
  await ready(page);
  await expect(page.getByText("Database connected")).toBeVisible();
  await pickPoint(page);
  await page.getByLabel("Label required").fill("Keep this draft");
  await page.route("**/api/**", (route) => route.abort());
  await page
    .getByRole("button", { name: "Save annotation", exact: true })
    .click();
  await expect(page.getByRole("alert")).toContainText(
    "Cannot reach the annotation service",
  );
  await expect(page.getByLabel("Label required")).toHaveValue(
    "Keep this draft",
  );
  await expect(page.getByText("Database connected")).toBeVisible();
  expect(await page.evaluate(() => localStorage.length)).toBe(0);
});

test("loading, error recovery and empty PLY states", async ({ page }) => {
  let release: () => void = () => undefined;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route("**/*.ply", async (route) => {
    await gate;
    await route.fulfill({ status: 404, body: "Missing scene" });
  });
  await page.goto("/");
  await expect(page.getByTestId("viewport")).toHaveAttribute(
    "data-state",
    "loading",
  );
  release();
  await expect(page.getByTestId("viewport")).toHaveAttribute(
    "data-state",
    "error",
  );
  await expect(page.getByRole("alert")).toContainText("HTTP 404");
  await page.unroute("**/*.ply");
  await page.getByRole("button", { name: "Reload scene" }).click();
  await expect(page.getByTestId("viewport")).toHaveAttribute(
    "data-state",
    "ready",
  );
  await page.route("**/*.ply", (route) =>
    route.fulfill({
      status: 200,
      body: "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n",
    }),
  );
  await page.reload();
  await expect(page.getByTestId("viewport")).toHaveAttribute(
    "data-state",
    "empty",
  );
  await expect(
    page.getByText("This PLY contains no points.", { exact: false }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "+ Pick point", exact: true }),
  ).toBeDisabled();
});

test("corrupt local storage is reported without overwriting records", async ({
  page,
}) => {
  await page.route("**/api/**", (route) => route.abort());
  await page.addInitScript(() =>
    localStorage.setItem("neural3d:annotations:v1:spectrum-garden", "{broken"),
  );
  await ready(page);
  await expect(page.getByRole("alert")).toContainText(
    "Local annotations could not be read",
  );
  await expect(
    page.getByRole("button", { name: "+ Pick point", exact: true }),
  ).toBeDisabled();
  expect(
    await page.evaluate(() =>
      localStorage.getItem("neural3d:annotations:v1:spectrum-garden"),
    ),
  ).toBe("{broken");
});
