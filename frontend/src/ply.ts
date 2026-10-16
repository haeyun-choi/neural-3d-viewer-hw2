import type { PointCloudData } from "./types";
import type { PlySummary } from "./plyHeader";
import { FULL_RESOLUTION_ERROR, PlyError } from "./plyPolicy";
export { PLY_IMPORT_TEXT, PlyError } from "./plyPolicy";

type Reply = { kind: "header"; summary: PlySummary } | { kind: "progress"; message: string } |
  { kind: "result"; cloud: PointCloudData } | { kind: "error"; error: string; empty?: boolean };

// A Blob is an immutable handle, not a second main-thread ArrayBuffer. The
// worker inspects a small header first and waits for consent before the full read.
export function parsePly(blob: Blob, signal: AbortSignal,
  approve: (summary: PlySummary) => Promise<boolean>, progress: (message: string) => void): Promise<PointCloudData> {
  if (signal.aborted) return Promise.reject(new DOMException("Cancelled", "AbortError"));
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./ply.worker.ts", import.meta.url), { type: "module" });
    let finished = false;
    const finish = () => {
      finished = true;
      signal.removeEventListener("abort", cancel);
      worker.onmessage = null; worker.onerror = null;
      worker.terminate();
    };
    const cancel = () => { finish(); reject(new DOMException("Cancelled", "AbortError")); };
    signal.addEventListener("abort", cancel, { once: true });
    worker.onerror = (event) => { event.preventDefault(); finish(); reject(new PlyError(FULL_RESOLUTION_ERROR)); };
    worker.onmessage = async (event: MessageEvent<Reply>) => {
      if (finished || signal.aborted) return;
      const reply = event.data;
      if (reply.kind === "header") {
        try {
          const accepted = await approve(reply.summary);
          if (finished || signal.aborted) return;
          if (accepted) worker.postMessage({ kind: "parse" }); else cancel();
        } catch { if (!finished) { finish(); reject(new PlyError(FULL_RESOLUTION_ERROR)); } }
      } else if (reply.kind === "progress") progress(reply.message);
      else {
        finish();
        if (reply.kind === "result") resolve(reply.cloud);
        else reject(new PlyError(reply.error, reply.empty));
      }
    };
    try { worker.postMessage({ kind: "inspect", file: blob }); }
    catch { finish(); reject(new PlyError(FULL_RESOLUTION_ERROR)); }
  });
}
