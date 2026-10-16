import { readHeaderText } from "./plyHeader";
import { PlyError } from "./plyPolicy";
import {
  detectGaussianFileType,
  GaussianError,
  gaussianExtension,
  gaussianFormatLabel,
  GAUSSIAN_EXTENSIONS,
  GAUSSIAN_HEADER_PROBE_BYTES,
  isLargeGaussianScene,
  parseGaussianPly,
  tryGaussianPly,
} from "./gaussianFormat";
import type { GaussianFileType, GaussianPlyInfo } from "./gaussianFormat";
import { boundedStream } from "./gaussianStream";
import { loadSpark } from "./sparkModule";
import type { SparkModule } from "./sparkModule";
import { probeRadAsset } from "./radAsset";
import { HostedSceneError, radFilename } from "./hostedScene";
import type { HostedScene } from "./hostedScene";
import type { GaussianIdentityKind, GaussianSplatData, PreparedSplatMesh } from "./types";

type SplatMeshInstance = InstanceType<SparkModule["SplatMesh"]>;

/** Map this app's container names onto Spark's SplatFileType members. */
export const sparkFileType = (spark: SparkModule, fileType: GaussianFileType) => ({
  ply: spark.SplatFileType.PLY,
  spz: spark.SplatFileType.SPZ,
  splat: spark.SplatFileType.SPLAT,
  ksplat: spark.SplatFileType.KSPLAT,
  pcsogszip: spark.SplatFileType.PCSOGSZIP,
  rad: spark.SplatFileType.RAD,
}[fileType]);

export type GaussianPhase = "inspecting" | "streaming" | "decoding" | "lod" | "paging";

export interface GaussianProgress {
  phase: GaussianPhase;
  /** Bytes pulled out of the local file so far. */
  streamedBytes: number;
  /** Bytes Spark's WebAssembly decoder has been handed so far. */
  decodedBytes: number;
  /** Local file size, or 0 for a hosted scene, whose size is never downloaded. */
  totalBytes: number;
  /** Spark is building (or has been asked to build) a level-of-detail tree. */
  lod: boolean;
  filename: string;
}

/**
 * How one local Gaussian file will be imported. Pure, so the decision that
 * matters most — a large file is streamed and fingerprinted without ever being
 * held in one ArrayBuffer — is testable without a browser or a 3 GB fixture.
 */
export interface GaussianImportPlan {
  fileType: GaussianFileType;
  /** Always true: Spark is fed from file.stream(), never from fileBytes. */
  streaming: true;
  /** Ask Spark to build a level-of-detail tree while it decodes. */
  lod: boolean;
  identity: GaussianIdentityKind;
  large: boolean;
}

export function planGaussianImport(
  fileType: GaussianFileType,
  sizeBytes: number,
  declaredSplatCount: number | null,
): GaussianImportPlan {
  const large = isLargeGaussianScene(sizeBytes, declaredSplatCount);
  return {
    fileType,
    streaming: true,
    // Spark's own RAD assets already carry an LoD tree; everything else has one
    // built on load, but only when the scene is large enough to need it. Small
    // scenes stay on the plain decode path, so their splat counts stay exact.
    lod: large && fileType !== "rad",
    identity: large ? "bounded-sample" : "file-sha256",
    large,
  };
}

const hex = (digest: ArrayBuffer) =>
  [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");

/**
 * Bounded local-scene identity for a file too large to hold in one ArrayBuffer.
 *
 * Web Crypto has no incremental digest, so hashing every byte of a 3 GB file
 * would mean materialising all of it. This fingerprint instead hashes a short,
 * fully specified description of the file: its name, byte length and
 * last-modified time, its complete PLY header when it has one, and three 1 MiB
 * windows taken from the start, the middle and the end of the payload.
 *
 * What it guarantees: the same file always produces the same scene id, and a
 * file of a different length, name, modification time, header or sampled window
 * produces a different one.
 *
 * What it cannot do: distinguish two files that differ only in payload bytes
 * outside the sampled windows. That is acceptable here and nowhere else in the
 * application — Gaussian annotation is disabled, no Gaussian identity reaches
 * the backend, and the id only selects the camera orientation saved for a scene
 * in this browser. Point-cloud PLY identity is unchanged and still hashes every
 * byte. Files at or below GAUSSIAN_LARGE_SCENE.bytes also still hash every byte.
 */
const IDENTITY_WINDOW_BYTES = 1024 * 1024;

async function boundedFingerprint(file: File, headerText: string): Promise<string> {
  const described = JSON.stringify({
    v: 1,
    kind: "bounded-sample",
    name: file.name,
    size: file.size,
    modified: file.lastModified,
    window: IDENTITY_WINDOW_BYTES,
    header: headerText,
  });
  const offsets = [
    0,
    Math.max(0, Math.floor(file.size / 2) - IDENTITY_WINDOW_BYTES / 2),
    Math.max(0, file.size - IDENTITY_WINDOW_BYTES),
  ];
  const parts: BlobPart[] = [new TextEncoder().encode(described)];
  for (const offset of offsets) parts.push(file.slice(offset, Math.min(offset + IDENTITY_WINDOW_BYTES, file.size)));
  // At most the description plus three windows: a few MiB, never the file.
  return hex(await crypto.subtle.digest("SHA-256", await new Blob(parts).arrayBuffer()));
}

export interface GaussianInspection {
  fileType: GaussianFileType;
  format: string;
  sha256: string;
  identity: GaussianIdentityKind;
  declaredSplatCount: number | null;
  plan: GaussianImportPlan;
}

/**
 * Inspect a local Gaussian Splat file and fingerprint it, reading only a bounded
 * prefix for classification. The bytes never leave the browser and the whole
 * file is never materialised: the container is recognised from four bytes, a
 * PLY header from at most GAUSSIAN_HEADER_PROBE_BYTES, and identity from either
 * the whole file (small) or the bounded fingerprint above (large).
 */
export async function inspectGaussianFile(file: File, signal: AbortSignal): Promise<GaussianInspection> {
  if (!GAUSSIAN_EXTENSIONS.includes(gaussianExtension(file.name) as (typeof GAUSSIAN_EXTENSIONS)[number]))
    throw new GaussianError(`Choose a Gaussian Splat file (${GAUSSIAN_EXTENSIONS.join(", ")}).`);
  if (file.name.length > 255 || /[\\/]/.test(file.name) || [...file.name].some((c) => c.charCodeAt(0) < 32))
    throw new GaussianError("Choose a filename of at most 255 characters without directory or control characters.");
  if (!file.size) throw new GaussianError("This Gaussian Splat file is empty.", true);

  const head = new Uint8Array(await file.slice(0, 4).arrayBuffer());
  if (signal.aborted) throw new DOMException("Cancelled", "AbortError");
  const fileType = detectGaussianFileType(head, file.name);

  let ply: GaussianPlyInfo | null = null;
  let headerText = "";
  if (fileType === "ply") {
    const header = await readHeaderText(file, GAUSSIAN_HEADER_PROBE_BYTES).catch((error: unknown) => {
      throw error instanceof PlyError ? new GaussianError(error.message, error.empty) : error;
    });
    if (signal.aborted) throw new DOMException("Cancelled", "AbortError");
    ply = parseGaussianPly(header.text, header.length, file.size);
    headerText = header.text;
  }

  const plan = planGaussianImport(fileType, file.size, ply?.splatCount ?? null);
  const sha256 =
    plan.identity === "bounded-sample"
      ? await boundedFingerprint(file, headerText)
      : hex(await crypto.subtle.digest("SHA-256", await file.arrayBuffer()));
  if (signal.aborted) throw new DOMException("Cancelled", "AbortError");

  return {
    fileType,
    format: gaussianFormatLabel(fileType, ply),
    sha256,
    identity: plan.identity,
    declaredSplatCount: ply?.splatCount ?? null,
    plan,
  };
}

/** How often the import progress is published; Spark reports every chunk. */
const PROGRESS_INTERVAL_MS = 150;

function meshPhase(streamed: number, decoded: number, total: number, lod: boolean): GaussianPhase {
  if (streamed < total) return "streaming";
  if (decoded < total) return "decoding";
  return lod ? "lod" : "decoding";
}

/**
 * Decode a local Gaussian file into a Spark mesh **before** the displayed scene
 * changes, so a cancelled or failed import leaves the previous scene alone.
 *
 * The file is handed to Spark as `stream` + `streamLength` + `fileName`. Spark's
 * SplatLoader pulls one chunk at a time and forwards it to its WebAssembly
 * ChunkDecoder in a worker, so neither this module nor Spark ever holds the
 * whole file: `file.arrayBuffer()` is not called at any size.
 */
export async function prepareLocalGaussianScene(
  file: File,
  signal: AbortSignal,
  report: (progress: GaussianProgress) => void,
): Promise<GaussianSplatData> {
  report({ phase: "inspecting", streamedBytes: 0, decodedBytes: 0, totalBytes: file.size, lod: false, filename: file.name });
  const inspection = await inspectGaussianFile(file, signal);
  const { plan } = inspection;
  const spark = await loadSpark();
  if (signal.aborted) throw new DOMException("Cancelled", "AbortError");

  const source = boundedStream(file);
  let decodedBytes = 0;
  const mesh: SplatMeshInstance = new spark.SplatMesh({
    stream: source.stream,
    streamLength: file.size,
    fileName: file.name,
    fileType: sparkFileType(spark, plan.fileType),
    ...(plan.lod ? { lod: true as const } : {}),
    onProgress: (event: ProgressEvent) => {
      decodedBytes = event.loaded;
    },
  });

  const publish = () =>
    report({
      phase: meshPhase(source.readBytes(), decodedBytes, file.size, plan.lod),
      streamedBytes: source.readBytes(),
      decodedBytes,
      totalBytes: file.size,
      lod: plan.lod,
      filename: file.name,
    });
  const ticker = window.setInterval(publish, PROGRESS_INTERVAL_MS);
  const stopFeeding = () => source.cancel();
  signal.addEventListener("abort", stopFeeding, { once: true });

  try {
    await mesh.initialized;
  } catch (error) {
    // A cancelled import closes the stream early, so Spark's decoder usually
    // fails on the truncated input. Either way the partial mesh is released here.
    mesh.dispose();
    if (signal.aborted) throw new DOMException("Cancelled", "AbortError");
    throw new GaussianError(
      `Spark could not decode this Gaussian Splat file: ${error instanceof Error ? error.message : String(error)}`,
    );
  } finally {
    window.clearInterval(ticker);
    signal.removeEventListener("abort", stopFeeding);
    source.cancel();
  }
  if (signal.aborted) {
    mesh.dispose();
    throw new DOMException("Cancelled", "AbortError");
  }

  const decoded = decodedSplatCount(mesh);
  if (!decoded) {
    mesh.dispose();
    throw new GaussianError("This Gaussian Splat file decoded to zero splats.", true);
  }

  return {
    origin: "local-file",
    file,
    url: null,
    filename: file.name,
    fileType: plan.fileType,
    format: inspection.format,
    sha256: inspection.sha256,
    identity: inspection.identity,
    sizeBytes: file.size,
    declaredSplatCount: inspection.declaredSplatCount ?? decoded,
    lod: plan.lod,
    paged: false,
    view: null,
    trajectoryUrl: null,
    mesh: mesh as unknown as PreparedSplatMesh,
  };
}

/**
 * Splats a decoded mesh actually holds. With an on-load LoD build Spark keeps
 * only the level-of-detail tree (packedSplats.numSplats is 0 by design), so the
 * tree's node count is what proves the decode produced something.
 */
export function decodedSplatCount(mesh: SplatMeshInstance): number {
  return (
    mesh.packedSplats?.lodSplats?.numSplats ||
    mesh.packedSplats?.numSplats ||
    mesh.extSplats?.numSplats ||
    mesh.splats?.getNumSplats() ||
    0
  );
}

/**
 * Attach a hosted, prebuilt Spark RAD level-of-detail asset.
 *
 * Nothing is downloaded up front. Spark reads the RAD header with an HTTP range
 * request and then pages 64K-splat chunks as the camera moves, either out of the
 * single .rad file with further range requests or, when the asset was built with
 * `build-lod --rad-chunked`, out of sibling .radc files named by the header. Both
 * layouts are Spark's; this function only chooses the asset and validates that
 * the host can serve it before the displayed scene changes.
 */
export async function prepareHostedRadScene(
  scene: HostedScene,
  signal: AbortSignal,
  report: (progress: GaussianProgress) => void,
): Promise<GaussianSplatData> {
  const filename = radFilename(scene.url);
  const progress = (phase: GaussianPhase, totalBytes: number): GaussianProgress => ({
    phase, streamedBytes: 0, decodedBytes: 0, totalBytes, lod: true, filename,
  });
  report(progress("inspecting", scene.sizeBytes ?? 0));

  const probe = await probeRadAsset(scene.url, signal);
  if (signal.aborted) throw new DOMException("Cancelled", "AbortError");
  const spark = await loadSpark();
  if (signal.aborted) throw new DOMException("Cancelled", "AbortError");

  report(progress("paging", probe.totalBytes ?? scene.sizeBytes ?? 0));
  const mesh: SplatMeshInstance = new spark.SplatMesh({
    url: scene.url,
    paged: true,
    fileType: spark.SplatFileType.RAD,
  });

  // Spark's own RAD decoder parses the header; this is where an asset that
  // passed the byte-level probe but is not a usable LoD tree is caught.
  let meta: Awaited<ReturnType<NonNullable<SplatMeshInstance["paged"]>["getRadMeta"]>>;
  try {
    meta = await mesh.paged!.getRadMeta();
  } catch (error) {
    mesh.dispose();
    if (signal.aborted) throw new DOMException("Cancelled", "AbortError");
    throw new HostedSceneError(
      `Spark could not read this RAD asset's header: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (signal.aborted) {
    mesh.dispose();
    throw new DOMException("Cancelled", "AbortError");
  }

  const chunked = !!meta.meta.chunks[0]?.filename;
  const layout = chunked
    ? `chunked · ${meta.meta.chunks.length} .radc files`
    : `single file · ${meta.meta.chunks.length} HTTP-range chunks`;
  if (!chunked && !probe.rangeSupported) {
    mesh.dispose();
    throw new HostedSceneError(
      "This single-file RAD asset needs HTTP Range support, which the host did not provide. See docs/hosted-scenes.md.",
    );
  }

  return {
    origin: "hosted-rad",
    file: null,
    url: scene.url,
    filename,
    fileType: "rad",
    format: `Gaussian Splat · Spark RAD (paged LoD) · ${layout}`,
    sha256: hex(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(scene.url))),
    identity: "hosted-url",
    sizeBytes: probe.totalBytes ?? scene.sizeBytes ?? 0,
    declaredSplatCount: meta.meta.count || scene.splatCount,
    lod: true,
    paged: true,
    view: scene.view,
    trajectoryUrl: scene.trajectoryUrl,
    mesh: mesh as unknown as PreparedSplatMesh,
  };
}

/**
 * Would Spark render this .ply as Gaussian splats? Used to keep a splat file out
 * of the point-cloud parser, which would otherwise accept its x/y/z properties
 * and draw a meaningless grey cloud. Reads a bounded prefix only.
 */
export async function isGaussianSplatPly(blob: Blob): Promise<boolean> {
  try {
    const header = await readHeaderText(blob, GAUSSIAN_HEADER_PROBE_BYTES);
    return tryGaussianPly(header.text, header.length, blob.size) !== null;
  } catch {
    return false;
  }
}

/** Route a dragged-and-dropped file to the representation its content belongs to. */
export async function classifyDroppedFile(file: File): Promise<"point-cloud" | "gaussian-splat"> {
  const extension = gaussianExtension(file.name);
  if (extension === ".ply") return (await isGaussianSplatPly(file)) ? "gaussian-splat" : "point-cloud";
  return GAUSSIAN_EXTENSIONS.includes(extension as (typeof GAUSSIAN_EXTENSIONS)[number])
    ? "gaussian-splat"
    : "point-cloud";
}
