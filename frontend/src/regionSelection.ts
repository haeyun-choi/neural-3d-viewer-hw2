import type { RegionData, RegionSelection, SelectionOperation } from "./types";

// Limits apply to annotation description complexity, never PLY/selected point count.
export const MAX_SELECTION_OPERATIONS = 64;
export const MAX_POLYGON_VERTICES = 512;
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
export function validRegion(value: unknown): value is RegionData {
  if (!value || typeof value !== "object") return false;
  const r = value as RegionData;
  return r.version === 1 && r.semantics === "through" &&
    Number.isSafeInteger(r.vertex_count) && r.vertex_count > 0 &&
    Number.isSafeInteger(r.selected_count) && r.selected_count > 0 && r.selected_count <= r.vertex_count &&
    Array.isArray(r.operations) && r.operations.length > 0 && r.operations.length <= MAX_SELECTION_OPERATIONS &&
    r.operations[0]?.mode === "replace" && r.operations.every((op) => op &&
      ["replace", "add", "subtract"].includes(op.mode) && ["rectangle", "lasso"].includes(op.shape) &&
      Array.isArray(op.matrix) && op.matrix.length === 16 && op.matrix.every(finite) &&
      Array.isArray(op.polygon) && op.polygon.length >= 3 && op.polygon.length <= MAX_POLYGON_VERTICES &&
      (op.shape !== "rectangle" || op.polygon.length === 4) &&
      op.polygon.every((p) => Array.isArray(p) && p.length === 2 && p.every((v) => finite(v) && v >= 0 && v <= 1)));
}

export function insidePolygon(x: number, y: number, polygon: [number, number][]) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [ax, ay] = polygon[j], [bx, by] = polygon[i];
    // Include boundary points deterministically. The same predicate is replayed.
    if (Math.abs((x - ax) * (by - ay) - (y - ay) * (bx - ax)) < 1e-12 &&
        x >= Math.min(ax, bx) - 1e-12 && x <= Math.max(ax, bx) + 1e-12 &&
        y >= Math.min(ay, by) - 1e-12 && y <= Math.max(ay, by) + 1e-12) return true;
    if ((ay > y) !== (by > y) && x < (bx - ax) * (y - ay) / (by - ay) + ax) inside = !inside;
  }
  return inside;
}

const cancelled = () => new DOMException("Selection cancelled", "AbortError");
// Yield before work and every ~8 ms, checking the budget every 64 vertices.
// This avoids duplicating the entire source cloud in a persistent worker.
export async function selectRegion(positions: Float32Array, operations: SelectionOperation[], signal: AbortSignal,
  progress: (fraction: number) => void = () => {}): Promise<RegionSelection> {
  const count = positions.length / 3;
  if (!operations.length || operations.length > MAX_SELECTION_OPERATIONS) throw new Error("Selection history allows 64 operations. Choose Replace to start a new region.");
  const prepared = operations.map((op) => ({ ...op,
    minX: Math.min(...op.polygon.map((p) => p[0])), maxX: Math.max(...op.polygon.map((p) => p[0])),
    minY: Math.min(...op.polygon.map((p) => p[1])), maxY: Math.max(...op.polygon.map((p) => p[1])),
  }));
  const pause = async () => {
    if (signal.aborted) throw cancelled();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    if (signal.aborted) throw cancelled();
  };
  await pause();
  const mask = new Uint8Array(count);
  let selectedCount = 0, deadline = performance.now() + 8;
  for (let i = 0; i < count; i++) {
    const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
    let selected = false;
    for (const op of prepared) {
      const m = op.matrix;
      const w = m[3] * x + m[7] * y + m[11] * z + m[15];
      let inside = false;
      if (w > 0) {
        const depth = (m[2] * x + m[6] * y + m[10] * z + m[14]) / w;
        const px = ((m[0] * x + m[4] * y + m[8] * z + m[12]) / w + 1) / 2;
        const py = (1 - (m[1] * x + m[5] * y + m[9] * z + m[13]) / w) / 2;
        inside = depth >= -1 && depth <= 1 && px >= op.minX - 1e-12 && px <= op.maxX + 1e-12 &&
          py >= op.minY - 1e-12 && py <= op.maxY + 1e-12 && insidePolygon(px, py, op.polygon);
      }
      if (op.mode === "replace") selected = inside;
      else if (inside) selected = op.mode === "add";
    }
    if (selected) { mask[i] = 1; selectedCount++; }
    if ((i & 63) === 0 && performance.now() >= deadline) {
      progress(i / count * 0.85); await pause(); deadline = performance.now() + 8;
    }
  }
  const indices = new Uint32Array(selectedCount);
  const min = { x: Infinity, y: Infinity, z: Infinity }, max = { x: -Infinity, y: -Infinity, z: -Infinity };
  let output = 0;
  for (let i = 0; i < count; i++) {
    if (mask[i]) {
      indices[output++] = i;
      min.x = Math.min(min.x, positions[i * 3]); max.x = Math.max(max.x, positions[i * 3]);
      min.y = Math.min(min.y, positions[i * 3 + 1]); max.y = Math.max(max.y, positions[i * 3 + 1]);
      min.z = Math.min(min.z, positions[i * 3 + 2]); max.z = Math.max(max.z, positions[i * 3 + 2]);
    }
    if ((i & 4095) === 0 && performance.now() >= deadline) {
      progress(0.85 + i / count * 0.15); await pause(); deadline = performance.now() + 8;
    }
  }
  if (signal.aborted) throw cancelled();
  progress(1);
  return { indices, bounds: output ? { min, max } : null };
}
