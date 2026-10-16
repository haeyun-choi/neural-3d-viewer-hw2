// Read-only browser-test observations. Installed only in the Vite development
// server when explicitly enabled; production builds remove the guarded code.
export interface ViewerSnapshot {
  camera: number[];
  target: number[];
  cameraUp: number[];
  controlsEnabled: boolean;
  highlights: { key: string; count: number; indices: number[]; color: string }[];
  colors: number[];
  fingerprint: string;
  manual: boolean;
  displayedPoints: number;
  indexCount: number;
  sourceIndexCount: number;
  faceCount: number;
  geometries: number;
  textures: number;
  programs: number;
  vertexColors: boolean;
  fallbackColor: string;
  markers: { id?: string; selected: boolean; color: string; position: number[] }[];
  projected: { x: number; y: number }[];
}

export interface SplatSnapshot {
  fingerprint: string;
  /** How `fingerprint` was derived: file-sha256, bounded-sample or hosted-url. */
  identity: string;
  fileType: string;
  origin: string;
  /** Source splats reported to the user. */
  splats: number;
  /** Splats (or level-of-detail nodes) Spark actually decoded. */
  decodedSplats: number;
  /** Splats currently resident in the renderer's page pool, for paged scenes. */
  pagedSplats: number;
  lod: boolean;
  paged: boolean;
  /** Splats Spark's renderer actually sorted and drew in the last frame. */
  activeSplats: number;
  /** Raw training cameras behind the loaded path. */
  trajectoryCameras: number;
  /** Distinct places those cameras were taken from. */
  trajectoryAnchors: number;
  /** Anchor the camera was last placed on, or -1. */
  trajectoryAnchor: number;
  /** True while the walker owns the camera and the tube constrains it. */
  exploring: boolean;
  pointerLocked: boolean;
  /** Tube radius in world units, and the live distance from the path. */
  tubeRadius: number;
  tubeDistance: number;
  /** True while the camera is resting against the boundary. */
  tubeAtBoundary: boolean;
  /** Path segment the walker is being held against, or -1. */
  activeSegment: number;
  /** OrbitControls.enabled: false only while the walker drives the camera. */
  controlsEnabled: boolean;
  /** Vertical field of view in degrees. */
  fov: number;
  /** What the path overlay is drawing, and where its markers landed. */
  trajectory: {
    cameras: number;
    anchors: number;
    selected: number;
    activeSegment: number;
    scale: number;
    /** Marker and frustum size chosen from the path's spacing and extent. */
    baseDepth: number;
    frustumsBuilt: boolean;
    visible: { path: boolean; markers: boolean; frustums: boolean; selected: boolean; active: boolean };
    vertices: { path: number; markers: number; frustums: number; selected: number; active: number };
    projected: { x: number; y: number; onScreen: boolean }[];
  };
  camera: number[];
  target: number[];
  /** The camera's own -z in world space. The only honest reading of "looking
   *  where?" while the walker drives, because OrbitControls' target is stale. */
  forward: number[];
  cameraUp: number[];
  manual: boolean;
  background: string;
  geometries: number;
  textures: number;
  programs: number;
  canvases: number;
}

declare global {
  interface Window {
    __NEURAL3D__?: () => ViewerSnapshot;
    __NEURAL3D_SPLAT__?: () => SplatSnapshot;
  }
}
