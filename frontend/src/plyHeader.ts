import { PlyError } from "./plyPolicy";

export interface PlyProperty { type: string; countType?: string; name: string }
export interface PlyElement { name: string; count: number; properties: PlyProperty[] }
export interface PlyHeader {
  encoding: "ascii" | "binary_little_endian" | "binary_big_endian";
  byteLength: number;
  text: string;
  elements: PlyElement[];
  vertexCount: number;
  faceCount: number;
}
export interface PlySummary {
  encoding: PlyHeader["encoding"];
  vertexCount: number;
  faceCount: number;
  sizeBytes: number;
}
export const scalarWidths: Record<string, number> = {
  char: 1, int8: 1, uchar: 1, uint8: 1, short: 2, int16: 2,
  ushort: 2, uint16: 2, int: 4, int32: 4, uint: 4, uint32: 4,
  float: 4, float32: 4, double: 8, float64: 8,
};
export const invalidPly = () => new PlyError("Malformed PLY: check the header, element counts and numeric data.");
const knownType = (type: string) => Object.hasOwn(scalarWidths, type);

export function parseHeader(text: string, byteLength = text.length): PlyHeader {
  const lines = text.trim().split(/\r\n|\r|\n/);
  if (lines.shift() !== "ply") throw new PlyError("This is not a PLY file. Choose a valid .ply file.");
  const elements: PlyElement[] = [];
  let current: PlyElement | undefined;
  let encoding: PlyHeader["encoding"] | undefined;
  for (const line of lines) {
    const fields = line.trim().split(/\s+/);
    if (!line.trim() || ["comment", "obj_info", "end_header"].includes(fields[0])) continue;
    if (fields[0] === "format" && !encoding && fields.length === 3 && fields[2] === "1.0" &&
        ["ascii", "binary_little_endian", "binary_big_endian"].includes(fields[1])) {
      encoding = fields[1] as PlyHeader["encoding"];
    } else if (fields[0] === "element" && fields.length === 3) {
      const count = Number(fields[2]);
      if (!/^\d+$/.test(fields[2]) || !Number.isSafeInteger(count) || count < 0 || elements.some((e) => e.name === fields[1])) throw invalidPly();
      current = { name: fields[1], count, properties: [] };
      elements.push(current);
    } else if (fields[0] === "property" && current) {
      const list = fields[1] === "list";
      const type = fields[list ? 3 : 1], countType = list ? fields[2] : undefined, name = fields[list ? 4 : 2];
      if (fields.length !== (list ? 5 : 3) || !knownType(type) ||
          (countType && (!knownType(countType) || /float|double/.test(countType))) || current.properties.some((p) => p.name === name)) throw invalidPly();
      current.properties.push({ type, countType, name });
    } else {
      // Reject unknown header directives before PLYLoader could log their text.
      throw invalidPly();
    }
  }
  const vertex = elements.find((element) => element.name === "vertex");
  if (!encoding || !vertex || !["x", "y", "z"].every((axis) => vertex.properties.some((p) => p.name === axis && !p.countType))) throw invalidPly();
  if (!vertex.count) throw new PlyError("This PLY contains no points. Choose a non-empty point cloud.", true);
  return { encoding, byteLength, text, elements, vertexCount: vertex.count, faceCount: elements.find((e) => e.name === "face")?.count ?? 0 };
}

// Header-only inspection uses small slices; neither a byte/count ceiling nor a
// full binary-to-string conversion is needed. Cancellation terminates the worker.
// The Gaussian Splat importer reuses this reader for its own header checks, and
// passes maxBytes so that classifying a multi-gigabyte file reads a bounded
// prefix instead of walking the whole blob when end_header is missing.
export async function readHeaderText(blob: Blob, maxBytes = Number.POSITIVE_INFINITY): Promise<{ text: string; length: number }> {
  if (!blob.size) throw new PlyError("This PLY file is empty.", true);
  const limit = Math.min(blob.size, Math.max(maxBytes, 64 * 1024));
  const decoder = new TextDecoder("latin1"); // one code unit per header byte
  let text = "";
  for (let offset = 0; offset < limit; offset += 64 * 1024) {
    const sliceEnd = Math.min(offset + 64 * 1024, limit);
    text += decoder.decode(await blob.slice(offset, sliceEnd).arrayBuffer());
    if (!/^ply(?:\r\n|\r|\n)/.test(text)) throw new PlyError("This is not a PLY file. Choose a valid .ply file.");
    const end = /(?:^|[\r\n])end_header(\r\n|\n|\r)/.exec(text);
    if (end) {
      // At a slice boundary a CR might be the first half of CRLF.
      if (end[1] === "\r" && end.index + end[0].length === text.length && sliceEnd < limit) continue;
      const length = end.index + end[0].length;
      return { text: text.slice(0, length), length };
    }
    if (text.includes("\0")) throw invalidPly();
  }
  throw new PlyError(
    limit < blob.size
      ? `Malformed PLY: no end_header declaration in the first ${Math.round(limit / 1024)} KiB of the file.`
      : "Malformed PLY: the end_header declaration is missing.",
  );
}

export async function inspectHeader(blob: Blob): Promise<PlyHeader> {
  const { text, length } = await readHeaderText(blob);
  return parseHeader(text, length);
}
