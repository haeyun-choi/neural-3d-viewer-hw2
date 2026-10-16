import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { FULL_RESOLUTION_ERROR, PlyError } from "./plyPolicy";
import { invalidPly, scalarWidths } from "./plyHeader";
import type { PlyHeader } from "./plyHeader";
import type { PointCloudData } from "./types";

// Validate payload bounds rather than imposing arbitrary face/list/point limits.
// No numeric arrays are allocated during this pass.
export function validatePayload(bytes: ArrayBuffer, header: PlyHeader) {
  const { encoding, elements, vertexCount } = header;
  const body = encoding === "ascii" ? new TextDecoder().decode(new Uint8Array(bytes, header.byteLength)) : "";
  const tokens = /\S+/g;
  const view = new DataView(bytes);
  let cursor = header.byteLength;
  const little = encoding === "binary_little_endian";
  const read = (type: string) => {
    let value: number;
    if (encoding === "ascii") {
      const token = tokens.exec(body)?.[0];
      if (!token || !/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(token)) throw invalidPly();
      value = Number(token);
    } else {
      if (cursor + scalarWidths[type] > bytes.byteLength) throw invalidPly();
      switch (type) {
        case "char": case "int8": value = view.getInt8(cursor); break;
        case "uchar": case "uint8": value = view.getUint8(cursor); break;
        case "short": case "int16": value = view.getInt16(cursor, little); break;
        case "ushort": case "uint16": value = view.getUint16(cursor, little); break;
        case "int": case "int32": value = view.getInt32(cursor, little); break;
        case "uint": case "uint32": value = view.getUint32(cursor, little); break;
        case "float": case "float32": value = view.getFloat32(cursor, little); break;
        default: value = view.getFloat64(cursor, little);
      }
      cursor += scalarWidths[type];
    }
    if (!Number.isFinite(value) || (!/float|double/.test(type) && !Number.isInteger(value))) throw invalidPly();
    return value;
  };
  for (const element of elements) {
    if (!element.properties.length) continue; // zero-byte unknown elements
    const minimumRowBytes = element.properties.reduce((sum, p) => sum + scalarWidths[p.countType ?? p.type], 0);
    if (encoding !== "ascii" && element.count > Math.floor((bytes.byteLength - cursor) / minimumRowBytes)) throw invalidPly();
    for (let i = 0; i < element.count; i++) {
      for (const property of element.properties) {
        const count = property.countType ? read(property.countType) : 1;
        if (!Number.isSafeInteger(count) || count < 0 ||
            (encoding !== "ascii" && count > Math.floor((bytes.byteLength - cursor) / scalarWidths[property.type]))) throw invalidPly();
        for (let j = 0; j < count; j++) {
          const value = read(property.type);
          if (element.name === "vertex" && ["x", "y", "z"].includes(property.name) && !Number.isFinite(Math.fround(value)))
            throw new PlyError("These coordinates cannot be represented by the browser's point geometry.");
          if (element.name === "face" && ["vertex_indices", "vertex_index"].includes(property.name) &&
              (!Number.isInteger(value) || value < 0 || value >= vertexCount)) throw invalidPly();
        }
      }
    }
  }
  if (encoding === "ascii" ? tokens.exec(body) !== null : cursor !== bytes.byteLength) throw invalidPly();
}

// Some mesh PLYs have per-face colors/UVs. PLYLoader expands triangles for those
// attributes. Rename only that ignored element in the already-hashed buffer,
// in place and at the same byte length, so vertex data and its colors stay intact.
function preventFaceExpansion(bytes: ArrayBuffer, header: PlyHeader) {
  const face = header.elements.find((element) => element.name === "face");
  if (!face) return;
  const indices = face.properties.some((p) => ["vertex_indices", "vertex_index"].includes(p.name));
  const expands = face.properties.some((p) => ["texcoord", "red", "green", "blue", "diffuse_red", "diffuse_green", "diffuse_blue", "r", "g", "b", "diffuse_r", "diffuse_g", "diffuse_b"].includes(p.name));
  if (!indices || expands) {
    const match = /(?:^|[\r\n])[ \t]*element[ \t]+(face)(?=[ \t])/.exec(header.text)!;
    const offset = match.index + match[0].length - 4;
    new Uint8Array(bytes, offset, 4).set([115, 107, 105, 112]); // face -> skip
  }
}

export function parsePointGeometry(bytes: ArrayBuffer, header: PlyHeader) {
  preventFaceExpansion(bytes, header);
  const geometry = new PLYLoader().parse(bytes);
  try {
    const sourceIndexCount = geometry.index?.count ?? 0;
    if (geometry.index) geometry.setIndex(null); // retain original vertices; never expand triangles
    const position = geometry.getAttribute("position"), color = geometry.getAttribute("color");
    if (!position || position.count !== header.vertexCount || !position.array.every(Number.isFinite) ||
        (color && (color.count !== position.count || !color.array.every(Number.isFinite)))) throw new PlyError(FULL_RESOLUTION_ERROR);
    return { geometry, sourceIndexCount };
  } catch (error) { geometry.dispose(); throw error; }
}

export async function parsePlyBytes(bytes: ArrayBuffer, header: PlyHeader, progress: (message: string) => void): Promise<PointCloudData> {
  progress("Validating PLY data…");
  validatePayload(bytes, header);
  progress("Computing scene fingerprint…");
  // Web Crypto snapshots input internally. Await it before allocating geometry,
  // reuse the same bytes, and hash BEFORE any face-header compatibility edit.
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  const sha256 = [...new Uint8Array(digest)].map((v) => v.toString(16).padStart(2, "0")).join("");
  progress("Preparing unique vertices…");
  const { geometry, sourceIndexCount } = parsePointGeometry(bytes, header);
  try {
    geometry.computeBoundingBox();
    const box = geometry.boundingBox!;
    const position = geometry.getAttribute("position"), color = geometry.getAttribute("color");
    return {
      positions: position.array as Float32Array, colors: color?.array as Float32Array | undefined,
      sha256, sourceIndexCount, face_count: header.faceCount,
      color_source: color ? "vertex" : "fallback",
      format: `PLY · ${header.encoding === "ascii" ? "ASCII" : header.encoding === "binary_little_endian" ? "binary LE" : "binary BE"}`,
      bounds: { min: { x: box.min.x, y: box.min.y, z: box.min.z }, max: { x: box.max.x, y: box.max.y, z: box.max.z } },
    };
  } finally { geometry.dispose(); }
}
