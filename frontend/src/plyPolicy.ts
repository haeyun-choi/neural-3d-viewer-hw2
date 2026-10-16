// Advisory thresholds only. There is no administrator/default hard limit.
export const PLY_IMPORT_TEXT = "ASCII or binary PLY. Large files can be loaded when device memory permits.";
export const FULL_RESOLUTION_ERROR = "The browser could not load this PLY at full resolution. Try a downsampled file or the future streaming viewer.";
export const WARNING_THRESHOLDS = {
  large: { bytes: 256 * 1024 ** 2, vertices: 5_000_000 },
  veryLarge: { bytes: 1024 ** 3, vertices: 25_000_000 },
} as const;
export type WarningTier = "large" | "very-large" | null;
export function sizeWarning(sizeBytes: number, vertexCount: number): WarningTier {
  if (sizeBytes > WARNING_THRESHOLDS.veryLarge.bytes || vertexCount > WARNING_THRESHOLDS.veryLarge.vertices) return "very-large";
  if (sizeBytes > WARNING_THRESHOLDS.large.bytes || vertexCount > WARNING_THRESHOLDS.large.vertices) return "large";
  return null;
}
export class PlyError extends Error {
  constructor(message: string, public empty = false) { super(message); }
}
