// Preflight for a hosted Spark RAD asset.
//
// Spark's PagedSplats does its own ranged fetches and its own RAD header
// decoding, and this module deliberately does not duplicate either: the binary
// format stays Spark's business. What it does is make one ranged request before
// the displayed scene changes, so the three ways a static host breaks paged
// streaming are reported with an actionable message instead of a console error
// inside Spark:
//
//   1. the URL is unreachable or blocked by CORS,
//   2. the host ignores Range and answers 200 with the whole asset,
//   3. the asset is not a RAD file at all.
//
// Only (2) needs explaining: a single-file RAD is paged purely with Range
// requests, so a host that ignores Range turns every page fetch into a download
// of the entire multi-gigabyte asset. A chunked RAD layout (a small .rad header
// beside .radc chunk files) does not need Range and is the portable option.

import { HostedSceneError } from "./hostedScene";

/** Spark tries 64 KiB, 256 KiB then 1 MiB for the header; match its first try. */
const PROBE_BYTES = 65536;
/** "RAD0" little-endian, the magic Spark's decode_rad_header() requires. */
const RAD_MAGIC = 0x30444152;

export interface RadProbe {
  /** The host answered 206 and honoured the byte range. */
  rangeSupported: boolean;
  /** Full asset length when the host reported one, else null. */
  totalBytes: number | null;
  /** Length of the JSON metadata block, from the RAD header. */
  metaBytes: number;
}

function totalFromHeaders(response: Response): number | null {
  const range = response.headers.get("Content-Range");
  const declared = range ? Number(range.split("/").pop()) : Number(response.headers.get("Content-Length"));
  return Number.isFinite(declared) && declared > 0 ? declared : null;
}

export async function probeRadAsset(url: string, signal?: AbortSignal): Promise<RadProbe> {
  let response: Response;
  try {
    response = await fetch(url, { headers: { Range: `bytes=0-${PROBE_BYTES - 1}` }, signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new HostedSceneError(
      `The hosted scene could not be reached: ${error instanceof Error ? error.message : String(error)}. ` +
        "Check the URL, and that the host sends Access-Control-Allow-Origin and allows the Range request header.",
    );
  }
  if (!response.ok)
    throw new HostedSceneError(`The hosted scene returned HTTP ${response.status} ${response.statusText}.`);

  const rangeSupported = response.status === 206;
  const totalBytes = totalFromHeaders(response);
  const head = new Uint8Array(await response.arrayBuffer());
  if (head.byteLength < 8)
    throw new HostedSceneError("The hosted scene is too short to be a Spark RAD asset.");
  const view = new DataView(head.buffer, head.byteOffset, 8);
  if (view.getUint32(0, true) !== RAD_MAGIC)
    throw new HostedSceneError(
      "That URL is not a Spark RAD asset: its first bytes are not the RAD0 signature. Build one with Spark's build-lod tool; see docs/hosted-scenes.md.",
    );
  const metaBytes = view.getUint32(4, true);

  if (!rangeSupported && totalBytes !== null && totalBytes > head.byteLength)
    throw new HostedSceneError(
      `This host ignored the Range request and answered with the whole ${(totalBytes / 1024 ** 3).toFixed(2)} GB asset. ` +
        "Paging a single-file .rad needs HTTP Range (206) support; otherwise publish a chunked .rad/.radc layout. See docs/hosted-scenes.md.",
    );

  return { rangeSupported, totalBytes, metaBytes };
}
