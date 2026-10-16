import * as THREE from "three";
import type { SparkModule } from "./sparkModule";

type SplatMeshInstance = InstanceType<SparkModule["SplatMesh"]>;

/**
 * Camera framing for a Spark mesh, at bounded cost.
 *
 * SplatMesh.getBoundingBox() walks every splat, which is fine for a small scene
 * and wrong for a large one: a 12M-splat scene would block the main thread, and
 * after an on-load LoD build SplatMesh.splats is an empty PackedSplats (Spark
 * keeps only the LoD tree), so the walk would return an empty box anyway.
 *
 * So centres are read directly out of the packed buffer that actually holds the
 * scene, with a stride that caps the work at MAX_SAMPLES regardless of size.
 * unpackSplat() is Spark's own decoder for its 16-byte packed layout. Sampling
 * can under-estimate the extent of a scene whose outliers fall between samples;
 * the radius is therefore padded slightly, and a manifest may override the
 * framing outright.
 */
const MAX_SAMPLES = 100_000;
const RADIUS_MARGIN = 1.02;

function sampledSphere(
  spark: SparkModule,
  packed: Uint32Array,
  count: number,
  encoding: Parameters<SparkModule["unpackSplat"]>[2],
): THREE.Sphere | null {
  const stride = Math.max(1, Math.ceil(count / MAX_SAMPLES));
  const box = new THREE.Box3();
  let seen = 0;
  for (let index = 0; index < count; index += stride) {
    const { center } = spark.unpackSplat(packed, index, encoding);
    if (!Number.isFinite(center.x) || !Number.isFinite(center.y) || !Number.isFinite(center.z)) continue;
    box.expandByPoint(center);
    seen += 1;
  }
  // The last splat is always included so a strided sample still spans the file.
  if (count > 0) {
    const { center } = spark.unpackSplat(packed, count - 1, encoding);
    if (Number.isFinite(center.x)) box.expandByPoint(center);
  }
  if (!seen) return null;
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  sphere.radius = Math.max(sphere.radius * RADIUS_MARGIN, 0.01);
  return sphere;
}

/** Bounding sphere of whatever splat data the mesh currently holds, or null. */
export function splatBoundingSphere(spark: SparkModule, mesh: SplatMeshInstance): THREE.Sphere | null {
  const lod = mesh.packedSplats?.lodSplats;
  if (lod?.packedArray && lod.numSplats > 0)
    return sampledSphere(spark, lod.packedArray, lod.numSplats, lod.splatEncoding);

  const packed = mesh.packedSplats;
  if (packed?.packedArray && packed.numSplats > 0)
    return sampledSphere(spark, packed.packedArray, packed.numSplats, packed.splatEncoding);

  // Paged scenes keep their splats in the renderer's shared page pool, and
  // extended-encoding scenes use a different layout; both are small enough at
  // this point (one root page, or a non-LoD ExtSplats scene) for Spark's walk.
  const resident = mesh.paged?.numSplats ?? mesh.extSplats?.numSplats ?? 0;
  if (resident > 0) {
    const sphere = mesh.getBoundingBox().getBoundingSphere(new THREE.Sphere());
    if (Number.isFinite(sphere.radius)) {
      sphere.radius = Math.max(sphere.radius * RADIUS_MARGIN, 0.01);
      return sphere;
    }
  }
  return null;
}
