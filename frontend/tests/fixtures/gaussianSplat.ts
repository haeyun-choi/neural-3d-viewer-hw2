import { closeSync, openSync, statSync, writeSync } from "node:fs";

// Deterministic, license-safe Gaussian Splat fixtures.
//
// Provenance: every byte below is generated from the analytic ring formula in
// this file. No downloaded, trained or third-party splat asset is used or
// committed, so the fixtures stay tiny and reproducible. The property layout is
// the original 3D Gaussian Splatting (INRIA) PLY layout that Spark reads, and
// the compressed layout is the PlayCanvas/SuperSplat "chunk" layout.

export interface GaussianPlyOptions {
  count?: number;
  /** 0-3; writes 0, 9, 24 or 45 f_rest properties. */
  shDegree?: 0 | 1 | 2 | 3;
  /** Write "ascii" or "binary_big_endian" instead; Spark reads neither. */
  format?: string;
  /** Drop these vertex properties to simulate an incomplete splat file. */
  omit?: string[];
  /** Cut this many bytes off the payload to simulate truncation. */
  truncate?: number;
  /** Append a mesh-style list property, which Spark's splat parser rejects. */
  listProperty?: boolean;
  /** Vary the generated values so two fixtures differ byte for byte. */
  phase?: number;
}

const REST_FOR_DEGREE = { 0: 0, 1: 9, 2: 24, 3: 45 } as const;

export function gaussianPly(options: GaussianPlyOptions = {}) {
  const { count = 12, shDegree = 0, format = "binary_little_endian", omit = [],
    truncate = 0, listProperty = false, phase = 0 } = options;
  const rest = REST_FOR_DEGREE[shDegree];
  const names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
    ...Array.from({ length: rest }, (_, i) => `f_rest_${i}`),
    "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
  ].filter((name) => !omit.includes(name));
  const header = Buffer.from(
    `ply\nformat ${format} 1.0\ncomment Synthetic Gaussian Splat regression fixture\n` +
    `element vertex ${count}\n` + names.map((name) => `property float ${name}\n`).join("") +
    (listProperty ? "property list uchar int extra_indices\n" : "") + "end_header\n", "latin1");

  const value = (name: string, i: number) => {
    const angle = (i / Math.max(count, 1)) * Math.PI * 2 + phase;
    switch (name) {
      case "x": return Math.cos(angle);
      case "y": return i % 2 ? 0.5 : -0.5;
      case "z": return Math.sin(angle);
      case "f_dc_0": return 0.6;
      case "f_dc_1": return -0.2;
      case "f_dc_2": return 0.9;
      case "opacity": return 4; // sigmoid(4) ≈ 0.98, clearly visible
      case "scale_0": case "scale_1": case "scale_2": return Math.log(0.12);
      case "rot_0": return 1;
      default: return 0; // normals, spherical harmonics and the other quaternion terms
    }
  };
  const body = Buffer.alloc(count * names.length * 4);
  let offset = 0;
  for (let i = 0; i < count; i++)
    for (const name of names) { body.writeFloatLE(value(name, i), offset); offset += 4; }
  return Buffer.concat([header, truncate ? body.subarray(0, Math.max(body.length - truncate, 0)) : body]);
}

/** PlayCanvas/SuperSplat compressed splat PLY. Header-shaped for routing and
 *  validation tests; its quantised payload is not a decoding fixture. */
export function compressedGaussianPly({ count = 512, chunks = Math.ceil(512 / 256), omitChunk = false } = {}) {
  const chunkProperties = ["min_x", "min_y", "min_z", "max_x", "max_y", "max_z",
    "min_scale_x", "min_scale_y", "min_scale_z", "max_scale_x", "max_scale_y", "max_scale_z",
    "min_r", "min_g", "min_b", "max_r", "max_g", "max_b"];
  const vertexProperties = ["packed_position", "packed_rotation", "packed_scale", "packed_color"];
  const header = Buffer.from(
    "ply\nformat binary_little_endian 1.0\n" +
    (omitChunk ? "" : `element chunk ${chunks}\n` + chunkProperties.map((n) => `property float ${n}\n`).join("")) +
    `element vertex ${count}\n` + vertexProperties.map((n) => `property uint ${n}\n`).join("") +
    "end_header\n", "latin1");
  const body = Buffer.alloc((omitChunk ? 0 : chunks * chunkProperties.length * 4) + count * vertexProperties.length * 4);
  return Buffer.concat([header, body]);
}

/** An ordinary colored point cloud: valid PLY, no Gaussian splat properties. */
export function plainPointCloudPly({ count = 6 } = {}) {
  const header = Buffer.from(
    `ply\nformat binary_little_endian 1.0\nelement vertex ${count}\nproperty float x\nproperty float y\n` +
    "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n", "latin1");
  const body = Buffer.alloc(count * 15);
  for (let i = 0; i < count; i++) {
    [Math.cos(i), i % 2 ? 0.5 : -0.5, Math.sin(i)].forEach((v, axis) => body.writeFloatLE(v, i * 15 + axis * 4));
    body.set([100, 160, 210], i * 15 + 12);
  }
  return Buffer.concat([header, body]);
}

/** A gzip container, which Spark recognises as .spz by magic bytes. */
export const gzipHeaderBytes = () => Buffer.from([0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0, 3]);
/** A ZIP container, which Spark recognises as .sog by magic bytes. */
export const zipHeaderBytes = () => Buffer.from([0x50, 0x4b, 0x03, 0x04, 0, 0, 0, 0]);

/**
 * A large compressed (PlayCanvas/SuperSplat) Gaussian PLY, written straight to
 * disk so the test never holds a second copy in memory.
 *
 * The compressed layout costs 16 bytes per splat, so it is the cheapest way to
 * build a file that genuinely crosses the viewer's large-scene splat-count
 * threshold: two million splats in ~32 MiB rather than the ~500 MiB the
 * uncompressed 3DGS layout would need. Chunk bounds and packed fields are filled
 * deterministically from the index, so the scene has real extent and the file is
 * byte-for-byte reproducible.
 */
export function writeLargeCompressedGaussianPly(path: string, count: number) {
  const chunkProperties = ["min_x", "min_y", "min_z", "max_x", "max_y", "max_z",
    "min_scale_x", "min_scale_y", "min_scale_z", "max_scale_x", "max_scale_y", "max_scale_z"];
  const vertexProperties = ["packed_position", "packed_rotation", "packed_scale", "packed_color"];
  const chunks = Math.ceil(count / 256);
  const handle = openSync(path, "w");
  try {
    writeSync(handle, Buffer.from(
      "ply\nformat binary_little_endian 1.0\ncomment Synthetic large compressed Gaussian Splat fixture\n" +
      `element chunk ${chunks}\n` + chunkProperties.map((n) => `property float ${n}\n`).join("") +
      `element vertex ${count}\n` + vertexProperties.map((n) => `property uint ${n}\n`).join("") +
      "end_header\n", "latin1"));

    const chunkRow = Buffer.alloc(chunkProperties.length * 4);
    for (let chunk = 0; chunk < chunks; chunk++) {
      const span = 1 + (chunk % 8);
      const bounds = [-span, -span, -span, span, span, span, Math.log(0.01), Math.log(0.01), Math.log(0.01),
        Math.log(0.05), Math.log(0.05), Math.log(0.05)];
      bounds.forEach((value, i) => chunkRow.writeFloatLE(value, i * 4));
      writeSync(handle, chunkRow);
    }

    // 1 MiB of vertices at a time: 65,536 rows of four uint32 fields.
    const batch = 65_536;
    const rows = Buffer.alloc(batch * 16);
    for (let base = 0; base < count; base += batch) {
      const rowsHere = Math.min(batch, count - base);
      for (let i = 0; i < rowsHere; i++) {
        const index = base + i;
        // 11/10/11-bit packed position, a unit-ish quaternion code, scale and RGBA.
        rows.writeUInt32LE(Math.imul(index, 2654435761) >>> 0, i * 16);
        rows.writeUInt32LE((Math.imul(index, 40503) >>> 0) & 0x3fffffff, i * 16 + 4);
        rows.writeUInt32LE((Math.imul(index, 2246822519) >>> 0) & 0x3fffffff, i * 16 + 8);
        rows.writeUInt32LE((0xff000000 | (Math.imul(index, 7919) & 0x00ffffff)) >>> 0, i * 16 + 12);
      }
      writeSync(handle, rows.subarray(0, rowsHere * 16));
    }
  } finally {
    closeSync(handle);
  }
  return { path, sizeBytes: statSync(path).size, count };
}
