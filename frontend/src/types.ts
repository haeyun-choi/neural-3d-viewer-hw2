import type { GaussianFileType } from "./gaussianFormat";

export interface Position {
  x: number;
  y: number;
  z: number;
}

/** Every scene declares which renderer owns it; there is no implicit default. */
export type RepresentationType = "point-cloud" | "gaussian-splat";

export interface SceneMetadata {
  id: string;
  name: string;
  description: string;
  /** Representation the scene is displayed with, independent of its file format. */
  representation: RepresentationType;
  /** Actual on-disk format, e.g. "PLY · ASCII" or "Gaussian Splat · SPZ". */
  format: string;
  point_count: number;
  asset: string;
  size_bytes: number;
  sha256: string;
  coordinates: string;
  filename?: string;
  /** bundled: the repository's sample. local: a file on this device. hosted: a
   *  URL this viewer pages over HTTP and never downloads in full. */
  source?: "bundled" | "local" | "hosted";
  face_count?: number;
  color_source?: "vertex" | "fallback";
}

export interface PointCloudData {
  positions: Float32Array;
  colors?: Float32Array;
  sha256: string;
  format: string;
  face_count: number;
  color_source: "vertex" | "fallback";
  sourceIndexCount: number;
  bounds: { min: Position; max: Position };
}

/** Where a Gaussian scene's bytes come from. */
export type GaussianOrigin = "local-file" | "hosted-rad";

/**
 * How a Gaussian scene is identified.
 * - `file-sha256`: SHA-256 of every byte of the local file, as for PLY point clouds.
 * - `bounded-sample`: the bounded local-scene fingerprint used above
 *   GAUSSIAN_LARGE_SCENE.bytes, where hashing every byte would need a whole-file
 *   ArrayBuffer. See fingerprintLargeFile() in gaussianSplat.ts for exactly what
 *   it covers and what it cannot distinguish.
 * - `hosted-url`: SHA-256 of the resolved asset URL. A hosted scene is public
 *   and immutable by convention, so its URL is its identity; no bytes are hashed.
 */
export type GaussianIdentityKind = "file-sha256" | "bounded-sample" | "hosted-url";

/** The Spark mesh a prepared Gaussian scene owns, kept opaque to non-Spark code. */
export interface PreparedSplatMesh {
  /** THREE.Object3D the viewer adds to its scene. */
  readonly isObject3D: true;
  dispose: () => void;
}

/**
 * A Gaussian scene never holds a decoded copy of its file on the main thread.
 *
 * - A local scene keeps the browser File handle and is streamed into Spark's
 *   WebAssembly decoder chunk by chunk; `file.arrayBuffer()` is never called on
 *   the whole file at any size.
 * - A hosted scene keeps only a URL; Spark pages the prebuilt RAD level-of-detail
 *   tree over HTTP as the camera moves.
 *
 * `mesh` is decoded *before* the displayed scene changes, so a cancelled or
 * failed import leaves the previous scene exactly as it was. Whoever created the
 * mesh (useScene) disposes it; the viewer only adds and removes it.
 */
export interface GaussianSplatData {
  origin: GaussianOrigin;
  /** Local import only. Never read in full into an ArrayBuffer. */
  file: Blob | null;
  /** Hosted import only: the resolved .rad URL. */
  url: string | null;
  filename: string;
  fileType: GaussianFileType;
  format: string;
  /** Scene-identity value; `identity` says how it was derived. */
  sha256: string;
  identity: GaussianIdentityKind;
  /** 0 when a hosted asset does not report a length. */
  sizeBytes: number;
  /** Splat count declared by a PLY or RAD header; null when only Spark can tell. */
  declaredSplatCount: number | null;
  /** Spark built a level-of-detail tree while decoding this scene. */
  lod: boolean;
  /** Spark pages this scene's chunks over HTTP instead of decoding it up front. */
  paged: boolean;
  /** Camera framing supplied by a scene manifest, when it has one. */
  view: { center: Vector3Tuple; radius: number } | null;
  /** Graphdeco cameras.json a hosted scene manifest points at, if any. */
  trajectoryUrl: string | null;
  /** Decoded Spark mesh, prepared before the scene changed. */
  mesh: PreparedSplatMesh;
}

export type LoadedScene =
  | { representation: "point-cloud"; metadata: SceneMetadata; cloud: PointCloudData }
  | { representation: "gaussian-splat"; metadata: SceneMetadata; splat: GaussianSplatData };

export interface AnnotationInput extends Position {
  kind?: "point" | "region";
  region?: RegionData | null;
  category: string;
  label: string;
  note: string;
}

export interface Annotation extends AnnotationInput {
  id: string;
  scene_id: string;
  created_at: string;
  updated_at: string;
}

export type ViewerStatus =
  | { state: "loading"; message: string }
  /** `note` qualifies the count, e.g. a level-of-detail or paged scene. */
  | { state: "ready"; count: number; note?: string }
  | { state: "empty" | "error"; message: string };

// Renderer boundary: scene coordinates and plain data, with no storage knowledge.
// Camera control is representation-independent and lives in the shared handle;
// point-cloud picking, highlights and region focus stay in the point handle.
export interface CameraHandle {
  reset: () => void;
  roll: (degrees: number) => void;
  setUp: (axis: "X" | "Y" | "Z" | "invert") => void;
  captureView: (home: boolean) => ViewPreferences;
  preset: (preset: ViewPreset) => void;
  restoreImported: () => void;
}

export interface ViewerHandle extends CameraHandle {
  prepare: (cloud: PointCloudData) => void;
  prepareHighlights: (highlights: Highlight[]) => void;
  focus: (position: Position) => void;
  focusRegion: (bounds: NonNullable<RegionSelection["bounds"]>) => void;
}

export type Vector3Tuple = [number, number, number];
export type QuaternionTuple = [number, number, number, number];
export interface CameraPose { position: Vector3Tuple; target: Vector3Tuple; up: Vector3Tuple; aspect: number; near: number; far: number }
export interface ViewPreferences { version: 1; up: Vector3Tuple; front: Vector3Tuple; home: CameraPose | null }
export type ViewPreset = "Front" | "Back" | "Left" | "Right" | "Top" | "Bottom";
export type InteractionMode = "navigate" | "point" | "rectangle" | "lasso";
export type SelectionOperationMode = "replace" | "add" | "subtract";
export interface SelectionOperation {
  mode: SelectionOperationMode;
  shape: "rectangle" | "lasso";
  polygon: [number, number][];
  matrix: number[]; // column-major original-local to clip; original buffers stay immutable
}
export interface RegionData {
  version: 1;
  semantics: "through";
  vertex_count: number;
  selected_count: number;
  operations: SelectionOperation[];
}
export interface RegionSelection {
  indices: Uint32Array;
  bounds: { min: Position; max: Position } | null;
}
export interface Highlight { key: string; color: string; indices: Uint32Array }
