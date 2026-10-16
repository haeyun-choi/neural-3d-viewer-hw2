import { inspectHeader } from "./plyHeader";
import type { PlyHeader } from "./plyHeader";
import { parsePlyBytes } from "./plyParser";
import { FULL_RESOLUTION_ERROR, PlyError } from "./plyPolicy";

let file: Blob | null = null;
let header: PlyHeader | null = null;
const progress = (message: string) => self.postMessage({ kind: "progress", message });
self.onmessage = async (event: MessageEvent<{ kind: "inspect"; file: Blob } | { kind: "parse" }>) => {
  try {
    if (event.data.kind === "inspect") {
      file = event.data.file;
      header = await inspectHeader(file);
      self.postMessage({ kind: "header", summary: {
        encoding: header.encoding, vertexCount: header.vertexCount, faceCount: header.faceCount, sizeBytes: file.size,
      } });
    } else if (file && header) {
      progress("Reading the full PLY file…");
      const bytes = await file.arrayBuffer(); // one full read, exclusively in this worker
      file = null;
      const cloud = await parsePlyBytes(bytes, header, progress);
      header = null;
      self.postMessage({ kind: "result", cloud }, { transfer: [cloud.positions.buffer, ...(cloud.colors ? [cloud.colors.buffer] : [])] });
    }
  } catch (error) {
    file = null; header = null;
    self.postMessage({ kind: "error", error: error instanceof PlyError ? error.message : FULL_RESOLUTION_ERROR,
      empty: error instanceof PlyError && error.empty });
  }
};
