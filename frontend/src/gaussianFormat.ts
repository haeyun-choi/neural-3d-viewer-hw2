// Gaussian Splat file recognition and structural validation.
//
// Spark decodes splat files inside its own WebAssembly parser, and its exported
// getSplatFileType() only reads the container magic: an ordinary colored
// point-cloud PLY is reported as "ply" and is then decoded as splats with
// default scale/opacity. The checks below therefore mirror the property sets
// that Spark's PLY parser actually requires, so a point cloud is never opened
// as a splat scene and a splat scene is never opened by the point-cloud parser.
// No byte-size or splat-count ceiling is imposed anywhere; device memory is the
// only limit, exactly as for PLY point clouds.

// Spark's SplatFileType values for the container formats this viewer accepts.
// "rad" is Spark's prebuilt level-of-detail container. It is reachable only
// through the hosted-scene path (by URL, paged), never through the local file
// picker, because a local RAD gains nothing over the hosted paged path and
// Spark cannot page a chunked RAD out of local bytes.
export type GaussianFileType = "ply" | "spz" | "splat" | "ksplat" | "pcsogszip" | "rad";

export const GAUSSIAN_EXTENSIONS = [".ply", ".spz", ".splat", ".ksplat", ".sog"] as const;
export const GAUSSIAN_ACCEPT = GAUSSIAN_EXTENSIONS.join(",");
export const GAUSSIAN_IMPORT_TEXT =
  "Gaussian Splat .ply (3DGS or compressed), .spz, .splat, .ksplat or .sog. Files stay in this browser.";

/**
 * Bounded prefix read to classify a local Gaussian file. A 3DGS PLY header with
 * 45 f_rest properties is under 2 KiB, so 1 MiB is far beyond any real header
 * and keeps classification O(1) instead of O(file size).
 */
export const GAUSSIAN_HEADER_PROBE_BYTES = 1024 * 1024;

/**
 * Above either threshold a local import is treated as a large scene: Spark
 * builds a level-of-detail tree while decoding, and scene identity switches
 * from a whole-file SHA-256 to the bounded fingerprint documented in
 * gaussianSplat.ts. Both are advisory, strictly greater-than, and impose no
 * ceiling: nothing is refused for being large.
 */
export const GAUSSIAN_LARGE_SCENE = { bytes: 256 * 1024 ** 2, splats: 2_000_000 } as const;

export function isLargeGaussianScene(sizeBytes: number, declaredSplatCount: number | null): boolean {
  return sizeBytes > GAUSSIAN_LARGE_SCENE.bytes || (declaredSplatCount ?? 0) > GAUSSIAN_LARGE_SCENE.splats;
}

export class GaussianError extends Error {
  constructor(message: string, public empty = false) {
    super(message);
  }
}

export type GaussianPlyVariant = "3dgs" | "compressed";
export interface GaussianPlyInfo {
  variant: GaussianPlyVariant;
  splatCount: number;
  /** Spherical-harmonic bands beyond the diffuse term (0-3). */
  shBands: number;
  headerBytes: number;
  payloadBytes: number;
}

const scalarWidths: Record<string, number> = {
  char: 1, int8: 1, uchar: 1, uint8: 1, short: 2, int16: 2,
  ushort: 2, uint16: 2, int: 4, int32: 4, uint: 4, uint32: 4,
  float: 4, float32: 4, double: 8, float64: 8,
};

interface RawElement { name: string; count: number; properties: { type: string; name: string }[] }

const SPLAT_3DGS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
  "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"];
const SPLAT_COMPRESSED = ["packed_position", "packed_rotation", "packed_scale", "packed_color"];
const CHUNK_REQUIRED = ["min_x", "min_y", "min_z", "max_x", "max_y", "max_z",
  "min_scale_x", "min_scale_y", "min_scale_z", "max_scale_x", "max_scale_y", "max_scale_z"];
// Spherical harmonics: 0, 9, 24 or 45 extra coefficients for degrees 0-3.
const SH_COEFFICIENTS: Record<number, number> = { 0: 0, 9: 1, 24: 2, 45: 3 };

function elementsOf(text: string): RawElement[] {
  const lines = text.trim().split(/\r\n|\r|\n/);
  if (lines.shift() !== "ply") throw new GaussianError("This is not a PLY file. Choose a supported Gaussian Splat file.");
  const elements: RawElement[] = [];
  let current: RawElement | undefined;
  let format: string | undefined;
  for (const line of lines) {
    const fields = line.trim().split(/\s+/);
    if (!line.trim() || ["comment", "obj_info", "end_header"].includes(fields[0])) continue;
    if (fields[0] === "format" && !format && fields.length === 3) {
      if (fields[2] !== "1.0") throw new GaussianError(`Unsupported PLY version ${fields[2]}. Spark reads PLY 1.0 only.`);
      format = fields[1];
    } else if (fields[0] === "element" && fields.length === 3 && /^\d+$/.test(fields[2])) {
      const count = Number(fields[2]);
      if (!Number.isSafeInteger(count) || elements.some((e) => e.name === fields[1]))
        throw new GaussianError("Malformed PLY header: check the element declarations.");
      current = { name: fields[1], count, properties: [] };
      elements.push(current);
    } else if (fields[0] === "property" && current) {
      if (fields[1] === "list")
        throw new GaussianError("This PLY uses list properties (for example mesh faces). Spark's Gaussian PLY parser does not support them.");
      if (fields.length !== 3 || !Object.hasOwn(scalarWidths, fields[1]))
        throw new GaussianError("Malformed PLY header: check the property declarations.");
      current.properties.push({ type: fields[1], name: fields[2] });
    } else {
      throw new GaussianError("Malformed PLY header: an unsupported declaration was found.");
    }
  }
  if (!format) throw new GaussianError("Malformed PLY header: the format line is missing.");
  if (format !== "binary_little_endian")
    throw new GaussianError(`Spark reads binary little-endian Gaussian PLY only; this file is ${format.replace(/_/g, " ")}.`);
  return elements;
}

const has = (element: RawElement, names: string[]) =>
  names.every((name) => element.properties.some((property) => property.name === name));

function rowBytes(element: RawElement) {
  return element.properties.reduce((sum, property) => sum + scalarWidths[property.type], 0);
}

/**
 * Validate a Gaussian Splat PLY header the way Spark's parser does, and report a
 * specific reason when the file is a PLY that Spark cannot render as splats.
 */
export function parseGaussianPly(text: string, headerBytes: number, fileSize: number): GaussianPlyInfo {
  const elements = elementsOf(text);
  const vertex = elements.find((element) => element.name === "vertex");
  if (!vertex) throw new GaussianError("This PLY has no vertex element, so it contains no Gaussian splats.");

  let variant: GaussianPlyVariant;
  let shBands = 0;
  if (has(vertex, SPLAT_COMPRESSED)) {
    variant = "compressed";
    const chunk = elements.find((element) => element.name === "chunk");
    if (!chunk || !has(chunk, CHUNK_REQUIRED))
      throw new GaussianError("This compressed Gaussian PLY is missing its chunk element, so it cannot be decoded.");
    if (chunk.count < Math.ceil(vertex.count / 256))
      throw new GaussianError("This compressed Gaussian PLY declares too few chunk records for its splat count.");
  } else if (has(vertex, SPLAT_3DGS)) {
    variant = "3dgs";
    const rest = vertex.properties.filter((property) => /^f_rest_\d+$/.test(property.name)).length;
    if (!Object.hasOwn(SH_COEFFICIENTS, rest))
      throw new GaussianError(`This Gaussian PLY declares ${rest} f_rest properties; 0, 9, 24 or 45 are expected.`);
    shBands = SH_COEFFICIENTS[rest];
  } else if (has(vertex, ["x", "y", "z"])) {
    throw new GaussianError(
      "This is an ordinary point-cloud PLY: it has no Gaussian splat properties (f_dc, opacity, scale, rot). Use Open PLY to view it as a point cloud.",
    );
  } else {
    throw new GaussianError("This PLY has neither point coordinates nor Gaussian splat properties.");
  }

  if (!vertex.count) throw new GaussianError("This Gaussian Splat file contains no splats.", true);
  // Every accepted element is fixed-width, so the payload length is exact.
  const payloadBytes = elements.reduce((sum, element) => sum + element.count * rowBytes(element), 0);
  if (!Number.isSafeInteger(payloadBytes)) throw new GaussianError("This Gaussian PLY declares an unrepresentable payload size.");
  if (fileSize < headerBytes + payloadBytes)
    throw new GaussianError("This Gaussian PLY is truncated: its data is shorter than the header declares.");
  return { variant, splatCount: vertex.count, shBands, headerBytes, payloadBytes };
}

/** Non-throwing routing check: is this PLY one that Spark can render as splats? */
export function tryGaussianPly(text: string, headerBytes: number, fileSize: number): GaussianPlyInfo | null {
  try {
    return parseGaussianPly(text, headerBytes, fileSize);
  } catch {
    return null;
  }
}

export function gaussianExtension(filename: string): string {
  const dot = filename.lastIndexOf(".");
  return dot <= 0 ? "" : filename.slice(dot).toLowerCase();
}

/**
 * Container recognition. Magic bytes decide wherever a format has them; only the
 * headerless .splat/.ksplat formats fall back to the filename extension, which
 * is what Spark's own getSplatFileTypeFromPath does.
 */
export function detectGaussianFileType(head: Uint8Array, filename: string): GaussianFileType {
  const extension = gaussianExtension(filename);
  const magic = (a: number, b: number, c: number, d?: number) =>
    head.length >= (d === undefined ? 3 : 4) && head[0] === a && head[1] === b && head[2] === c &&
    (d === undefined || head[3] === d);
  if (magic(0x70, 0x6c, 0x79)) return "ply"; // "ply"
  if (magic(0x1f, 0x8b, 0x08)) {
    if (extension !== ".spz") throw new GaussianError(`This is a gzip container, which Spark reads as .spz; the file is named ${extension || "without an extension"}.`);
    return "spz";
  }
  if (magic(0x50, 0x4b, 0x03, 0x04)) {
    if (extension !== ".sog") throw new GaussianError(`This is a ZIP container, which Spark reads as .sog; the file is named ${extension || "without an extension"}.`);
    return "pcsogszip";
  }
  if (extension === ".splat" || extension === ".ksplat") return extension.slice(1) as GaussianFileType;
  throw new GaussianError(
    `This file is not a supported Gaussian Splat container. Supported: ${GAUSSIAN_EXTENSIONS.join(", ")}.`,
  );
}

export function gaussianFormatLabel(fileType: GaussianFileType, ply: GaussianPlyInfo | null): string {
  if (fileType === "ply")
    return ply?.variant === "compressed"
      ? "Gaussian Splat · compressed PLY"
      : `Gaussian Splat · 3DGS PLY binary LE${ply ? ` · SH degree ${ply.shBands}` : ""}`;
  const names: Record<Exclude<GaussianFileType, "ply">, string> = {
    spz: "Gaussian Splat · SPZ",
    splat: "Gaussian Splat · SPLAT",
    ksplat: "Gaussian Splat · KSPLAT",
    pcsogszip: "Gaussian Splat · SOG",
    rad: "Gaussian Splat · Spark RAD (paged LoD)",
  };
  return names[fileType];
}
