import { createHash } from "node:crypto";
import { expect, test } from "@playwright/test";
import {
  detectGaussianFileType,
  GaussianError,
  gaussianFormatLabel,
  GAUSSIAN_EXTENSIONS,
  parseGaussianPly,
  tryGaussianPly,
} from "../src/gaussianFormat";
import {
  classifyDroppedFile,
  inspectGaussianFile,
  isGaussianSplatPly,
  planGaussianImport,
} from "../src/gaussianSplat";
import { GAUSSIAN_HEADER_PROBE_BYTES, GAUSSIAN_LARGE_SCENE, isLargeGaussianScene } from "../src/gaussianFormat";
import { parseSceneManifest, resolveRadUrl, sceneFromUrl, HostedSceneError } from "../src/hostedScene";
import { inspectHeader, readHeaderText } from "../src/plyHeader";
import { compressedGaussianPly, gaussianPly, gzipHeaderBytes, plainPointCloudPly, zipHeaderBytes } from "./fixtures/gaussianSplat";
import { meshPly } from "./fixtures/ply";

const file = (buffer: Buffer, name: string) =>
  new File([Uint8Array.from(buffer)], name, { type: "application/octet-stream" });
const blob = (buffer: Buffer) => new Blob([Uint8Array.from(buffer)]);
const live = () => new AbortController().signal;
const parse = async (buffer: Buffer) => {
  const header = await readHeaderText(blob(buffer));
  return parseGaussianPly(header.text, header.length, buffer.byteLength);
};
const reason = async (buffer: Buffer) => {
  try {
    await parse(buffer);
    return "";
  } catch (error) {
    expect(error).toBeInstanceOf(GaussianError);
    return (error as GaussianError).message;
  }
};

test("valid 3DGS PLY headers are accepted at every spherical-harmonic degree", async () => {
  for (const shDegree of [0, 1, 2, 3] as const) {
    const buffer = gaussianPly({ count: 12, shDegree });
    const info = await parse(buffer);
    expect(info.variant).toBe("3dgs");
    expect(info.splatCount).toBe(12);
    expect(info.shBands).toBe(shDegree);
    expect(info.headerBytes + info.payloadBytes).toBe(buffer.byteLength);
    expect(gaussianFormatLabel("ply", info)).toContain(`SH degree ${shDegree}`);
  }
});

test("compressed chunk PLY headers are accepted and validated against their chunk count", async () => {
  const info = await parse(compressedGaussianPly({ count: 512, chunks: 2 }));
  expect(info.variant).toBe("compressed");
  expect(info.splatCount).toBe(512);
  expect(gaussianFormatLabel("ply", info)).toBe("Gaussian Splat · compressed PLY");
  expect(await reason(compressedGaussianPly({ count: 512, chunks: 1 }))).toContain("too few chunk records");
  expect(await reason(compressedGaussianPly({ count: 512, omitChunk: true }))).toContain("missing its chunk element");
});

test("a plain point cloud is rejected with an instruction to use the point-cloud importer", async () => {
  expect(await reason(plainPointCloudPly())).toContain("Use Open PLY");
  expect(await reason(meshPly({ colors: true }))).toContain("list properties");
});

test("incomplete, mis-encoded, empty and truncated splat files each report their own reason", async () => {
  expect(await reason(gaussianPly({ format: "ascii" }))).toContain("binary little-endian");
  expect(await reason(gaussianPly({ format: "binary_big_endian" }))).toContain("binary little-endian");
  expect(await reason(gaussianPly({ omit: ["scale_1"] }))).toContain("Use Open PLY");
  expect(await reason(gaussianPly({ omit: ["x", "y", "z"] }))).toContain("neither point coordinates");
  expect(await reason(gaussianPly({ listProperty: true }))).toContain("list properties");
  expect(await reason(gaussianPly({ count: 0 }))).toContain("contains no splats");
  expect(await reason(gaussianPly({ truncate: 4 }))).toContain("truncated");
  // A stray f_rest count is not a valid spherical-harmonic degree.
  const stray = gaussianPly({ shDegree: 1 }).toString("latin1").replace("property float f_rest_8\n", "");
  expect(await reason(Buffer.from(stray, "latin1"))).toContain("f_rest properties");
});

test("container recognition prefers magic bytes and falls back to the extension only for headerless formats", () => {
  expect(detectGaussianFileType(Uint8Array.from(gaussianPly()).slice(0, 4), "scene.ply")).toBe("ply");
  // A .ply magic wins even when the name claims another container.
  expect(detectGaussianFileType(Uint8Array.from(gaussianPly()).slice(0, 4), "scene.spz")).toBe("ply");
  expect(detectGaussianFileType(Uint8Array.from(gzipHeaderBytes()), "scene.spz")).toBe("spz");
  expect(detectGaussianFileType(Uint8Array.from(zipHeaderBytes()), "scene.sog")).toBe("pcsogszip");
  expect(detectGaussianFileType(new Uint8Array([1, 2, 3, 4]), "scene.splat")).toBe("splat");
  expect(detectGaussianFileType(new Uint8Array([1, 2, 3, 4]), "scene.ksplat")).toBe("ksplat");
  expect(() => detectGaussianFileType(Uint8Array.from(gzipHeaderBytes()), "scene.sog")).toThrow(/gzip/);
  expect(() => detectGaussianFileType(Uint8Array.from(zipHeaderBytes()), "scene.spz")).toThrow(/ZIP/);
  expect(() => detectGaussianFileType(new Uint8Array([1, 2, 3, 4]), "scene.obj")).toThrow(/not a supported/);
  for (const extension of GAUSSIAN_EXTENSIONS) expect(extension.startsWith(".")).toBe(true);
});

test("point-cloud and Gaussian import paths never receive each other's PLY files", async () => {
  expect(await isGaussianSplatPly(blob(gaussianPly()))).toBe(true);
  expect(await isGaussianSplatPly(blob(compressedGaussianPly()))).toBe(true);
  expect(await isGaussianSplatPly(blob(plainPointCloudPly()))).toBe(false);
  expect(await isGaussianSplatPly(blob(meshPly({ colors: true })))).toBe(false);
  // An ASCII 3DGS PLY is not renderable by Spark, so it keeps its old route.
  expect(await isGaussianSplatPly(blob(gaussianPly({ format: "ascii" })))).toBe(false);
  expect(await isGaussianSplatPly(blob(Buffer.from("not a ply")))).toBe(false);

  expect(await classifyDroppedFile(file(gaussianPly(), "scene.ply"))).toBe("gaussian-splat");
  expect(await classifyDroppedFile(file(plainPointCloudPly(), "cloud.ply"))).toBe("point-cloud");
  expect(await classifyDroppedFile(file(gzipHeaderBytes(), "scene.spz"))).toBe("gaussian-splat");
  expect(await classifyDroppedFile(file(Buffer.from("x"), "scene.SPLAT"))).toBe("gaussian-splat");
  expect(await classifyDroppedFile(file(meshPly(), "mesh.ply"))).toBe("point-cloud");
  expect(await classifyDroppedFile(file(Buffer.from("x"), "notes.txt"))).toBe("point-cloud");

  // The point-cloud header parser still accepts an ordinary cloud unchanged.
  expect((await inspectHeader(blob(plainPointCloudPly()))).vertexCount).toBe(6);
});

test("Gaussian scene identity is the SHA-256 of the exact file bytes and is deterministic", async () => {
  const buffer = gaussianPly({ count: 20, shDegree: 1 });
  const expected = createHash("sha256").update(buffer).digest("hex");
  const first = await inspectGaussianFile(file(buffer, "ring.ply"), live());
  const second = await inspectGaussianFile(file(buffer, "renamed-copy.ply"), live());
  expect(first.sha256).toBe(expected);
  expect(second.sha256).toBe(expected); // identity follows the bytes, not the name
  expect(first.declaredSplatCount).toBe(20);
  expect(first.fileType).toBe("ply");
  expect(first.identity).toBe("file-sha256");
  expect(first.plan).toEqual({ fileType: "ply", streaming: true, lod: false, identity: "file-sha256", large: false });
  expect(first.format).toContain("3DGS PLY binary LE");

  const other = await inspectGaussianFile(file(gaussianPly({ count: 20, shDegree: 1, phase: 0.5 }), "ring.ply"), live());
  expect(other.sha256).not.toBe(expected);
});

test("Gaussian import rejects unsupported names, empty files and corrupt input before any scene changes", async () => {
  const rejected = async (buffer: Buffer, name: string) => {
    try {
      await inspectGaussianFile(file(buffer, name), live());
      return "";
    } catch (error) {
      return error instanceof Error ? error.message : String(error);
    }
  };
  expect(await rejected(gaussianPly(), "scene.obj")).toContain("Choose a Gaussian Splat file");
  expect(await rejected(gaussianPly(), `${"n".repeat(260)}.ply`)).toContain("255 characters");
  expect(await rejected(Buffer.alloc(0), "scene.ply")).toContain("empty");
  expect(await rejected(Buffer.from("definitely not a splat"), "scene.ply")).toContain("not a supported Gaussian Splat container");
  expect(await rejected(plainPointCloudPly(), "cloud.ply")).toContain("Use Open PLY");
  expect(await rejected(gaussianPly({ truncate: 8 }), "cut.ply")).toContain("truncated");
});

test("tryGaussianPly never throws and reports null for files Spark cannot render", async () => {
  const header = await readHeaderText(blob(plainPointCloudPly()));
  expect(tryGaussianPly(header.text, header.length, 1)).toBeNull();
  expect(tryGaussianPly("not a header", 12, 12)).toBeNull();
});

// --- Large local scenes --------------------------------------------------
//
// The decision that fixes the 3.13 GB blocker is pure, so it is checked here
// without a browser or a multi-gigabyte fixture: a large file is always
// streamed, never buffered, gets an on-load level-of-detail build and switches
// to the bounded fingerprint. tests/large-gaussian.spec.ts proves the same
// property against a running import by instrumenting Blob.prototype.arrayBuffer.

const THREE_GIGABYTE_PLY = 3_128_668_597; // the real final_3dgs/point_cloud.ply
const TWELVE_MILLION_SPLATS = 12_615_593; // its declared vertex count

test("a multi-gigabyte Gaussian PLY is planned as a streamed, level-of-detail import", () => {
  const plan = planGaussianImport("ply", THREE_GIGABYTE_PLY, TWELVE_MILLION_SPLATS);
  expect(plan).toEqual({
    fileType: "ply",
    streaming: true,
    lod: true,
    identity: "bounded-sample",
    large: true,
  });
  // No plan, at any size, asks for a whole-file buffer.
  expect(Object.keys(plan)).not.toContain("fileBytes");
});

test("the large-scene thresholds are advisory, strictly greater-than, and never a ceiling", () => {
  expect(isLargeGaussianScene(GAUSSIAN_LARGE_SCENE.bytes, null)).toBe(false);
  expect(isLargeGaussianScene(GAUSSIAN_LARGE_SCENE.bytes + 1, null)).toBe(true);
  expect(isLargeGaussianScene(1024, GAUSSIAN_LARGE_SCENE.splats)).toBe(false);
  expect(isLargeGaussianScene(1024, GAUSSIAN_LARGE_SCENE.splats + 1)).toBe(true);
  // Either trigger is enough: a compressed container is small but splat-heavy.
  expect(planGaussianImport("spz", 1024, GAUSSIAN_LARGE_SCENE.splats + 1).lod).toBe(true);
  expect(planGaussianImport("ply", 1024, 10).lod).toBe(false);
  // A prebuilt RAD already carries its LoD tree, so nothing is rebuilt.
  expect(planGaussianImport("rad", THREE_GIGABYTE_PLY, TWELVE_MILLION_SPLATS).lod).toBe(false);
});

test("classifying a Gaussian PLY reads a bounded prefix, not the whole blob", async () => {
  // A blob that claims to be a PLY but never declares end_header must not be
  // walked to its end; the reader stops at the probe limit and says so.
  const runaway = new Blob([
    new TextEncoder().encode("ply\nformat binary_little_endian 1.0\n"),
    // Printable filler: NUL bytes would trip the malformed-header check first.
    new Uint8Array(GAUSSIAN_HEADER_PROBE_BYTES + 512 * 1024).fill(0x41),
  ]);
  await expect(readHeaderText(runaway, GAUSSIAN_HEADER_PROBE_BYTES)).rejects.toThrow(/no end_header declaration in the first/);
  expect(await isGaussianSplatPly(runaway)).toBe(false);
  // A real header is still found and still bounds the payload check.
  const real = blob(gaussianPly({ count: 4, shDegree: 3 }));
  expect((await readHeaderText(real, GAUSSIAN_HEADER_PROBE_BYTES)).text).toContain("end_header");
});

// --- Hosted RAD scenes ---------------------------------------------------

const BASE = "https://viewer.example.edu/hw1-demo/";

test("only http(s) .rad URLs are accepted as hosted scenes", () => {
  expect(resolveRadUrl("https://cdn.example.edu/a/scene-lod.rad", BASE)).toBe("https://cdn.example.edu/a/scene-lod.rad");
  expect(resolveRadUrl("scenes/campus-lod.RAD", BASE)).toBe("https://viewer.example.edu/hw1-demo/scenes/campus-lod.RAD");
  for (const bad of ["", "   ", "javascript:alert(1)", "data:text/plain,rad", "file:///tmp/a.rad", "https://cdn.example.edu/a.spz", "https://cdn.example.edu/a.ply"])
    expect(() => resolveRadUrl(bad, BASE)).toThrow(HostedSceneError);
  expect(sceneFromUrl("https://cdn.example.edu/a/campus-lod.rad", BASE)).toMatchObject({
    id: "https://cdn.example.edu/a/campus-lod.rad",
    name: "campus-lod",
    url: "https://cdn.example.edu/a/campus-lod.rad",
    splatCount: null,
    view: null,
  });
});

test("the scene manifest keeps valid entries, drops bad ones and never throws", () => {
  const manifest = JSON.stringify({
    version: 1,
    scenes: [
      { id: "campus", name: "Campus walk", description: "12.6M splats.", url: "scenes/campus-lod.rad",
        attribution: "Internal use only.", splatCount: 12615593, sizeBytes: 1280517688,
        view: { center: [1, 2, 3], radius: 12.5 } },
      { id: "chunked", name: "Chunked", url: "https://cdn.example.edu/c/scene-lod.rad" },
      { id: "campus", name: "Duplicate id", url: "scenes/other-lod.rad" },
      { id: "", name: "No id", url: "scenes/x-lod.rad" },
      { id: "bad-scheme", name: "Bad scheme", url: "javascript:alert(1)" },
      { id: "not-rad", name: "Not RAD", url: "scenes/x.spz" },
      "nonsense",
    ],
  });
  const scenes = parseSceneManifest(manifest, BASE);
  expect(scenes.map((scene) => scene.id)).toEqual(["campus", "chunked"]);
  expect(scenes[0]).toMatchObject({
    name: "Campus walk",
    url: "https://viewer.example.edu/hw1-demo/scenes/campus-lod.rad",
    splatCount: 12615593,
    view: { center: [1, 2, 3], radius: 12.5 },
  });
  expect(scenes[1].view).toBeNull();

  for (const broken of ["", "{", "[]", "null", JSON.stringify({ version: 2, scenes: [] }), JSON.stringify({ version: 1 })])
    expect(parseSceneManifest(broken, BASE)).toEqual([]);
  // A manifest is metadata; an oversized document is refused outright.
  expect(parseSceneManifest(JSON.stringify({ version: 1, scenes: [], pad: "x".repeat(300_000) }), BASE)).toEqual([]);
  // A malformed camera hint is dropped without dropping the scene.
  expect(parseSceneManifest(JSON.stringify({ version: 1, scenes: [
    { id: "a", name: "A", url: "a-lod.rad", view: { center: [1, 2], radius: 3 } }] }), BASE)[0].view).toBeNull();
});
