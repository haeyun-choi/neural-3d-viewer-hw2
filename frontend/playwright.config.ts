import { defineConfig } from "@playwright/test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const testDatabase =
  process.env.HW1_TEST_DB ??
  join(mkdtempSync(join(tmpdir(), "neural3d-hw1-")), "annotations.sqlite3");
// GUI-launched training runs write checkpoints here, never into the repository.
const runsDirectory = process.env.CLASSIFICATION_RUNS_DIR ?? mkdtempSync(join(tmpdir(), "neural3d-runs-"));
const browserName =
  process.env.HW1_BROWSER === "firefox" ? "firefox" : "chromium";
// The dev-server port is overridable so a suite run does not collide with a dev
// server the developer already has open. Defaults to Vite's usual 5173.
const port = Number(process.env.HW1_PORT ?? 5173);
const apiPort = Number(process.env.HW1_API_PORT ?? 8000);
const apiUrl = `http://127.0.0.1:${apiPort}`;

export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  timeout: 60_000,
  // A cold dev server may still compile worker imports on the first load.
  // Tests use small fixtures; allow compilation and software WebGL setup time.
  expect: { timeout: 15_000 },
  use: {
    browserName,
    headless: process.env.HW1_HEADED !== "1",
    baseURL: `http://127.0.0.1:${port}`,
    viewport: { width: 1440, height: 1080 },
    launchOptions: {
      executablePath: browserName === "chromium" ? process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE : undefined,
      args:
        browserName === "chromium"
          ? ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
          : [],
      firefoxUserPrefs: { "webgl.force-enabled": true },
    },
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command: `${process.env.HW1_PYTHON ?? "python3"} -m uvicorn app.main:app --host 127.0.0.1 --port ${apiPort}`,
      cwd: "../backend",
      env: { DATABASE_PATH: testDatabase, CLASSIFICATION_RUNS_DIR: runsDirectory },
      url: `${apiUrl}/api/health`,
      reuseExistingServer: false,
    },
    {
      command: `npm run dev -- --port ${port} --strictPort`,
      env: { VITE_VIEWER_DIAGNOSTICS: "1", HW1_API_URL: apiUrl },
      url: `http://127.0.0.1:${port}`,
      reuseExistingServer: false,
    },
  ],
});
