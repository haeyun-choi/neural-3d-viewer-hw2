import { createHash } from "node:crypto";
import { expect, test } from "@playwright/test";
import { inspectHeader, parseHeader } from "../src/plyHeader";
import { parsePlyBytes, parsePointGeometry, validatePayload } from "../src/plyParser";
import { sizeWarning, WARNING_THRESHOLDS as tiers } from "../src/plyPolicy";
import { meshPly, vertices } from "./fixtures/ply";

const bytes = (buffer: Buffer) => Uint8Array.from(buffer).buffer;
for (const options of [
  { ascii: true, faces: false, colors: true },
  { ascii: true, faces: false },
  { faces: false, colors: true },
  {}, { little: false }, { ascii: true, colors: true, extras: true },
  { colors: true, faceAttributes: true }, { faceAttributes: true },
]) test(`PLY unique vertices, original colors and original-byte hash: ${JSON.stringify(options)}`, async () => {
  const buffer = meshPly(options), data = bytes(buffer);
  const header = await inspectHeader(new Blob([data]));
  validatePayload(data, header);
  const { geometry, sourceIndexCount } = parsePointGeometry(data, header);
  expect(geometry.index).toBeNull();
  expect(geometry.getAttribute("position").count).toBe(vertices.length);
  expect(Array.from(geometry.getAttribute("position").array)).toEqual(vertices.flat());
  if (options.faces !== false && !options.faceAttributes) expect(sourceIndexCount).toBe(24);
  expect(!!geometry.getAttribute("color")).toBe(!!options.colors);
  geometry.dispose();
  const cloud = await parsePlyBytes(bytes(buffer), header, () => {});
  expect(cloud.positions.length / 3).toBe(5);
  expect(cloud.face_count).toBe(options.faces === false ? 0 : 8);
  expect(cloud.color_source).toBe(options.colors ? "vertex" : "fallback");
  expect(cloud.sha256).toBe(createHash("sha256").update(buffer).digest("hex"));
  expect(cloud.bounds).toEqual({ min: { x: -1, y: -1, z: -1 }, max: { x: 1, y: 1, z: 1 } });
});

test("Spanner header and larger counts have no default element/point/size ceiling", async () => {
  const text = "ply\nformat binary_little_endian 1.0\nelement vertex 150000\nproperty float x\nproperty float y\nproperty float z\nelement face 300000\nproperty list uchar int vertex_indices\nend_header\n";
  const header = parseHeader(text);
  expect(header.vertexCount).toBe(150000);
  expect(header.faceCount).toBe(300000);
  expect(sizeWarning(5700266, header.vertexCount)).toBeNull();
  expect(sizeWarning(10 * 1024 ** 2 + 1, 200001)).toBeNull();
  expect(parseHeader(text.replace("150000", "25000001").replace("300000", "900000000")).faceCount).toBe(900000000);
  expect(sizeWarning(Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER)).toBe("very-large");
});

for (const [size, count, warning] of [
  [tiers.large.bytes - 1, tiers.large.vertices - 1, null],
  [tiers.large.bytes, tiers.large.vertices, null],
  [tiers.large.bytes + 1, 1, "large"], [1, tiers.large.vertices + 1, "large"],
  [tiers.veryLarge.bytes - 1, tiers.veryLarge.vertices - 1, "large"],
  [tiers.veryLarge.bytes, tiers.veryLarge.vertices, "large"],
  [tiers.veryLarge.bytes + 1, 1, "very-large"], [1, tiers.veryLarge.vertices + 1, "very-large"],
] as const) test(`advisory threshold: ${size} bytes, ${count} vertices => ${warning}`, () => {
  expect(sizeWarning(size, count)).toBe(warning);
});

test("header inspection reads slices only and supports long headers without a hard cap", async () => {
  let fullReads = 0, slices = 0;
  class HeaderBlob extends Blob {
    override async arrayBuffer(): Promise<ArrayBuffer> { fullReads++; throw new Error("Full read during inspection"); }
    override slice(start?: number, end?: number) { slices++; return super.slice(start, end); }
  }
  const fixture = meshPly({ ascii: true }).toString().replace("comment Synthetic regression fixture", `comment ${"x".repeat(70_000)}`);
  const header = await inspectHeader(new HeaderBlob([fixture]));
  expect(header.vertexCount).toBe(5);
  expect(fullReads).toBe(0);
  expect(slices).toBe(2);
});

test("truncated lists, invalid indices, missing header and non-PLY fail safely", async () => {
  const original = meshPly();
  const header = await inspectHeader(new Blob([bytes(original)]));
  for (const missing of [1, 7, 13]) expect(() => validatePayload(bytes(original.subarray(0, -missing)), header)).toThrow("Malformed PLY");
  const invalidIndex = Buffer.from(original);
  invalidIndex.writeInt32LE(5, header.byteLength + vertices.length * 12 + 1);
  expect(() => validatePayload(bytes(invalidIndex), header)).toThrow("Malformed PLY");
  const invalidList = Buffer.from(original);
  invalidList[header.byteLength + vertices.length * 12] = 255;
  expect(() => validatePayload(bytes(invalidList), header)).toThrow("Malformed PLY");
  await expect(inspectHeader(new Blob(["not a PLY"]))).rejects.toThrow("not a PLY");
  await expect(inspectHeader(new Blob(["ply\nformat ascii 1.0\n"]))).rejects.toThrow("end_header");
});


for (const indexed of [true, false]) test(`indented ignored face declarations preserve vertex rows (indexed=${indexed})`, async () => {
  const original = meshPly({ ascii: true, colors: true, faceAttributes: indexed }).toString();
  const text = original.replace("element face", "  element face").replace("vertex_indices", indexed ? "vertex_indices" : "connections");
  const data = new TextEncoder().encode(text).buffer;
  const header = await inspectHeader(new Blob([data]));
  const cloud = await parsePlyBytes(data, header, () => {});
  expect(Array.from(cloud.positions)).toEqual(vertices.flat());
  expect(cloud.face_count).toBe(8);
  expect(cloud.color_source).toBe("vertex");
});
