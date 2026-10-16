import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import type { ViewerSnapshot } from "../src/viewerDiagnostics";

// End-to-end HW2 flow against the real FastAPI backend and the bundled
// ModelNet10 checkpoints and samples: show an object in the existing viewer,
// classify it, upload a PLY, and drive a real training run.

const snapshot = (page: Page): Promise<ViewerSnapshot> => page.evaluate(() => window.__NEURAL3D__!());
const panel = (page: Page) => page.getByTestId("classification-panel");

async function ready(page: Page) {
  await page.goto("/");
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-state", "ready");
  await expect(panel(page).getByLabel("ModelNet10 sample").locator("option")).not.toHaveCount(0);
}

function cubeMesh() {
  const vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
  const faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]];
  return Buffer.from(`ply\nformat ascii 1.0\nelement vertex 8\nproperty float x\nproperty float y\nproperty float z\nelement face 6\nproperty list uchar int vertex_indices\nend_header\n${vertices.map((v) => v.join(" ")).join("\n")}\n${faces.map((f) => `4 ${f.join(" ")}`).join("\n")}\n`);
}

test("a ModelNet10 sample is shown in the viewer and classified by both models", async ({ page, request }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await ready(page);
  await expect(page.getByTestId("classification-scope")).toContainText("one of the ten ModelNet10 classes");
  await expect(page.getByTestId("classification-scope")).toContainText("night stand");

  const { samples } = await (await request.get("/api/classification/samples")).json();
  const sample = samples.find((s: { class: string }) => s.class === "chair") ?? samples[0];
  await panel(page).getByLabel("ModelNet10 sample").selectOption(sample.id);
  await panel(page).getByRole("button", { name: "Show sample" }).click();
  await expect(page.getByRole("heading", { level: 2, name: `ModelNet10 · ${sample.label}` })).toBeVisible();
  await expect(page.getByText("1,024 points · Ready")).toBeVisible();
  // ModelNet10 is Z-up; the viewer turns the object upright.
  await expect.poll(async () => (await snapshot(page)).cameraUp.map((v) => Math.round(v))).toEqual([0, 0, 1]);
  await expect(page.getByTestId("classification-input")).toContainText(`true class ${sample.class.replace("_", " ")}`);

  for (const modelId of ["mini_pointnet", "shallow"]) {
    await panel(page).getByLabel("Model", { exact: true }).selectOption(modelId);
    await expect(page.getByTestId("prediction")).toHaveCount(0);
    await page.getByTestId("predict-button").click();
    const prediction = page.getByTestId("prediction");
    await expect(prediction).toBeVisible();
    const expected = await (await request.post("/api/classification/predict/sample", { data: { sample_id: sample.id, model_id: modelId } })).json();
    await expect(page.getByTestId("predicted-class")).toHaveText(expected.predicted_class.replace("_", " "));
    await expect(page.getByTestId("prediction-confidence")).toHaveText(`${(expected.confidence * 100).toFixed(1)}% confidence`);
    await expect(prediction.getByRole("list", { name: "Top-3 probabilities" }).locator("li")).toHaveCount(3);
    await expect(prediction).toContainText(expected.model.architecture);
    await expect(prediction).toContainText("1,024 points, subsampled from the point cloud, unit sphere");
    await expect(prediction).toContainText("ms inference");
  }
  expect(errors).toEqual([]);
});

test("an uploaded PLY is shown, classified into the ten classes, and bad files are reported", async ({ page }) => {
  await ready(page);
  const uploads: string[] = [];
  page.on("request", (r) => { if (r.url().includes("/predict/upload")) uploads.push(r.url()); });
  await panel(page).getByLabel("Choose PLY to classify").setInputFiles({ name: "cube.ply", mimeType: "application/octet-stream", buffer: cubeMesh() });
  await expect(page.getByRole("heading", { level: 2, name: "cube" })).toBeVisible();
  await expect(page.getByTestId("classification-input")).toContainText("uploaded only when you press Predict");
  expect(uploads).toEqual([]);
  await page.getByTestId("predict-button").click();
  await expect(page.getByTestId("prediction")).toContainText("PLY · ASCII · 8 points · 12 faces");
  await expect(page.getByTestId("prediction")).toContainText("surface sampling of the mesh");
  expect(uploads).toHaveLength(1);
  expect(uploads[0]).toContain("filename=cube.ply");

  await panel(page).getByLabel("Choose PLY to classify").setInputFiles({ name: "broken.ply", mimeType: "application/octet-stream", buffer: Buffer.from("definitely not a ply") });
  await page.getByTestId("predict-button").click();
  await expect(page.getByTestId("classification-error")).toContainText("Not a PLY");
  await expect(page.getByTestId("prediction")).toHaveCount(0);
});

test("the comparison shows the bundled shallow baseline and Mini-PointNet test results", async ({ page, request }) => {
  await ready(page);
  const { comparison } = await (await request.get("/api/classification/results")).json();
  const table = page.getByTestId("model-comparison");
  await expect(table.getByRole("row")).toHaveCount(3);
  await expect(table).toContainText(`${(comparison.baseline.test_accuracy * 100).toFixed(1)}%`);
  await expect(table).toContainText(`${(comparison.final.test_accuracy * 100).toFixed(1)}%`);
  await expect(page.getByTestId("comparison-summary")).toContainText(`on ${comparison.test_size} test objects`);
});

test("interactive training runs on real ModelNet10 data, is monitored live and can be cancelled", async ({ page, request }) => {
  test.setTimeout(300_000);
  const data = await (await request.get("/api/classification/dataset")).json();
  test.skip(!data.prepared, "ModelNet10 is not prepared for this backend (set MODELNET10_DATA_DIR); a GUI run would download it");
  await ready(page);
  const form = panel(page);
  await form.getByLabel("Architecture").selectOption("shallow");
  await form.getByLabel("Learning rate").fill("0.003");
  await form.getByLabel("Epochs").fill("3");
  await form.getByLabel("Hidden dimension").selectOption("32");
  await form.getByLabel("Activation").selectOption("gelu");
  await form.getByLabel("Batch size").selectOption("64");
  await page.getByTestId("start-training").click();
  await expect(page.getByTestId("run-status")).toHaveText(/queued|running/);
  await expect(page.getByTestId("run-epoch")).toContainText("/ 3");
  await expect(page.getByTestId("run-status")).toHaveText("completed", { timeout: 240_000 });
  await expect(page.getByTestId("run-epoch")).toHaveText("epoch 3 / 3");
  const metrics = page.getByTestId("live-metrics");
  await expect(metrics).toContainText("Train loss");
  await expect(metrics.getByTestId("run-test-accuracy")).toHaveText(/\d+\.\d%/);
  await expect(page.getByTestId("loss-chart").locator("polyline")).toHaveCount(2);
  await expect(page.getByTestId("model-comparison")).toContainText("This session’s run · shallow");
  // The finished run can be used for prediction straight away.
  const option = form.getByLabel("Model", { exact: true }).locator("option", { hasText: "· shallow" }).last();
  await expect(option).toHaveText(/^Run [0-9a-f]{6} · shallow/);

  await form.getByLabel("Architecture").selectOption("mini_pointnet");
  await form.getByLabel("Epochs").fill("100");
  await page.getByTestId("start-training").click();
  await expect(page.getByTestId("run-status")).toHaveText("running", { timeout: 60_000 });
  await page.getByTestId("cancel-training").click();
  await expect(page.getByTestId("run-status")).toHaveText("cancelled", { timeout: 60_000 });
  await expect(page.getByTestId("start-training")).toBeEnabled();
});

test("the classification panel fits a phone-width layout", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 900 });
  await ready(page);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});
