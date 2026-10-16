import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to the FastAPI backend. The target is overridable
// so a test run can use its own backend port without disturbing a backend the
// developer already has running on the default one.
const apiTarget = process.env.HW1_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  base: "./",
  server: {
    proxy: { "/api": apiTarget },
  },
  build: {
    rollupOptions: {
      output: { manualChunks: { three: ["three"] } },
    },
  },
});
