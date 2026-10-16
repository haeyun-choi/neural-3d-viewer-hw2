// Deterministic, license-safe test geometry. No external scan or asset is used.
export const vertices = [[0, 0, 0], [-1, -1, -1], [1, -1, -1], [-1, 1, 1], [1, 1, 1]];
const triangles = [[0, 1, 2], [0, 2, 4], [0, 4, 3], [0, 3, 1], [1, 3, 4], [1, 4, 2], [2, 4, 3], [2, 3, 1]];
export function meshPly(options: { ascii?: boolean; little?: boolean; colors?: boolean; faceAttributes?: boolean; faces?: boolean; extras?: boolean } = {}) {
  const { ascii = false, little = true, colors = false, faceAttributes = false, faces = true, extras = false } = options;
  const header = Buffer.from(`ply\nformat ${ascii ? "ascii" : `binary_${little ? "little" : "big"}_endian`} 1.0\ncomment Synthetic regression fixture\nelement vertex ${vertices.length}\nproperty float x\nproperty float y\nproperty float z\n${colors ? "property uchar red\nproperty uchar green\nproperty uchar blue\n" : ""}${faces ? `element face ${triangles.length}\nproperty list uchar int vertex_indices\n${faceAttributes ? "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty list uchar float texcoord\n" : ""}` : ""}${extras ? "element edge 1\nproperty int vertex1\nproperty int vertex2\nelement confidence 1\nproperty float value\n" : ""}end_header\n`);
  const rows: { value: number; type: "float" | "int" | "uchar" }[][] = vertices.map((v) => [
    ...v.map((value) => ({ value, type: "float" as const })),
    ...(colors ? [90, 180, 240].map((value) => ({ value, type: "uchar" as const })) : []),
  ]);
  if (faces) for (const indices of triangles) rows.push([
    { value: 3, type: "uchar" }, ...indices.map((value) => ({ value, type: "int" as const })),
    ...(faceAttributes ? [255, 0, 0].map((value) => ({ value, type: "uchar" as const })) : []),
    ...(faceAttributes ? [{ value: 6, type: "uchar" as const }, ...[0, 0, 1, 0, 1, 1].map((value) => ({ value, type: "float" as const }))] : []),
  ]);
  if (extras) rows.push([{ value: 0, type: "int" }, { value: 1, type: "int" }], [{ value: 0.5, type: "float" }]);
  if (ascii) return Buffer.concat([header, Buffer.from(rows.map((r) => r.map((v) => v.value).join(" ")).join("\n") + "\n")]);
  const body = Buffer.alloc(rows.flat().reduce((size, p) => size + (p.type === "uchar" ? 1 : 4), 0));
  let offset = 0;
  for (const row of rows) for (const { value, type } of row) {
    if (type === "uchar") body.writeUInt8(value, offset);
    else if (type === "int") { if (little) body.writeInt32LE(value, offset); else body.writeInt32BE(value, offset); }
    else { if (little) body.writeFloatLE(value, offset); else body.writeFloatBE(value, offset); }
    offset += type === "uchar" ? 1 : 4;
  }
  return Buffer.concat([header, body]);
}
