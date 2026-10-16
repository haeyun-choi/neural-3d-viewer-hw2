import * as THREE from "three";
import type { SpatialTrajectory, TrajectoryCamera } from "./cameraTrajectory";

/**
 * The capture path drawn inside the scene the splats live in.
 *
 * What is drawn is the **collapsed** path: one marker per place the rig stood,
 * joined in capture order. The raw poses are a different thing — the real data
 * is 948 cameras at 158 places, six cube faces per place, so drawing them all
 * stacks six frustums on every marker and says nothing about where the walker
 * may go. They stay available behind a debug toggle and are built only if it is
 * ever switched on.
 *
 * Everything is batched, whatever the count: one Line for the path, one Points
 * for the anchors, one Line for the segment the walker is on, one small Points
 * plus cross for the selected anchor, and — only on demand — one LineSegments
 * holding every raw frustum. A THREE.CameraHelper or a DOM label per camera
 * would mean thousands of objects and thousands of draw calls paid by Spark on
 * every frame.
 *
 * Nothing here applies a transform of its own. The object is added straight to
 * the viewer's scene, so the path, the frustums and the splats share exactly one
 * world space; the orientation controls move only the camera, never the scene,
 * so an upright/front choice moves all three together by construction.
 */

/** Frustum and marker size as a fraction of the median gap between anchors. */
const DEPTH_FROM_SPACING = 0.9;
/** Kept inside this fraction of the path's own extent, whatever the spacing. */
const DEPTH_MIN_OF_EXTENT = 1 / 200;
const DEPTH_MAX_OF_EXTENT = 1 / 6;
/** Fallback when every camera sits at exactly the same place. */
const MIN_DEPTH = 1e-3;
const PATH_COLOR = 0x5fa8ff;
const FRUSTUM_COLOR = 0x8bb8e8;
const SELECTED_COLOR = 0xffc857;
const ACTIVE_COLOR = 0x7ef0c0;
const MARKER_COLOR = 0xbcd8f5;

/** Unit frustum in camera space (-z forward), scaled per camera when written. */
function frustumEdges(aspect: number, halfFovY: number): number[] {
  const y = Math.tan(halfFovY);
  const x = y * aspect;
  const apex: [number, number, number] = [0, 0, 0];
  const corners: [number, number, number][] = [
    [x, y, -1],
    [-x, y, -1],
    [-x, -y, -1],
    [x, -y, -1],
  ];
  // Four rays to the image plane, the image rectangle, and an up marker so the
  // roll of the camera is visible without relying on colour.
  const up: [number, number, number] = [0, y * 1.45, -1];
  const pairs: [number, number, number][][] = [
    [apex, corners[0]], [apex, corners[1]], [apex, corners[2]], [apex, corners[3]],
    [corners[0], corners[1]], [corners[1], corners[2]], [corners[2], corners[3]], [corners[3], corners[0]],
    [corners[0], up], [corners[1], up],
  ];
  return pairs.flat().flat();
}

/**
 * A drawing size that suits the capture, for any shape of capture.
 *
 * The median gap between consecutive *anchors* is the natural scale for a walk.
 * It is measured on the collapsed path on purpose: on the raw poses five of
 * every six gaps are zero, the plain median is zero, and frustums came out too
 * small to see. The result is clamped to a fraction of the path's extent so a
 * sparse ring does not get markers that swallow the scene and a dense walk does
 * not get invisible ones.
 */
function baseSize(trajectory: SpatialTrajectory): number {
  const { extent, medianSpacing } = trajectory;
  if (extent <= 0) return MIN_DEPTH;
  const wanted = medianSpacing > 0 ? medianSpacing * DEPTH_FROM_SPACING : extent * DEPTH_MAX_OF_EXTENT;
  return Math.max(Math.min(wanted, extent * DEPTH_MAX_OF_EXTENT), extent * DEPTH_MIN_OF_EXTENT, MIN_DEPTH);
}

const EMPTY = () => new THREE.BufferAttribute(new Float32Array(0), 3);

export class CameraTrajectoryView {
  readonly group = new THREE.Group();
  private trajectory: SpatialTrajectory | null = null;
  private selected = -1;
  private activeSegment = -1;
  private scale = 1;
  private baseDepth = 1;
  private pathVisible = true;
  private frustumsVisible = false;
  /** Raw frustums are expensive and usually off; built the first time asked. */
  private frustumsBuilt = false;

  private readonly pathGeometry = new THREE.BufferGeometry();
  private readonly path: THREE.Line;
  private readonly markerGeometry = new THREE.BufferGeometry();
  private readonly markers: THREE.Points;
  private readonly activeGeometry = new THREE.BufferGeometry();
  private readonly active: THREE.LineSegments;
  private readonly selectedGeometry = new THREE.BufferGeometry();
  private readonly selectedCross: THREE.LineSegments;
  private readonly selectedPointGeometry = new THREE.BufferGeometry();
  private readonly selectedPoint: THREE.Points;
  private readonly frustumGeometry = new THREE.BufferGeometry();
  private readonly frustums: THREE.LineSegments;
  private readonly materials: THREE.Material[] = [];

  constructor() {
    this.group.name = "camera-trajectory";
    // Drawn after the splats and never occluding them: the path is an overlay on
    // a volumetric scene, and depth-testing thin lines against splats mostly
    // hides them.
    const line = (color: number, opacity: number, width = 1) => {
      const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthTest: false, linewidth: width });
      this.materials.push(material);
      return material;
    };
    const points = (color: number, size: number) => {
      const material = new THREE.PointsMaterial({ color, size, sizeAttenuation: false, transparent: true, opacity: 0.9, depthTest: false });
      this.materials.push(material);
      return material;
    };
    this.path = new THREE.Line(this.pathGeometry, line(PATH_COLOR, 0.9));
    this.markers = new THREE.Points(this.markerGeometry, points(MARKER_COLOR, 7));
    this.active = new THREE.LineSegments(this.activeGeometry, line(ACTIVE_COLOR, 1, 3));
    this.selectedCross = new THREE.LineSegments(this.selectedGeometry, line(SELECTED_COLOR, 1));
    this.selectedPoint = new THREE.Points(this.selectedPointGeometry, points(SELECTED_COLOR, 13));
    this.frustums = new THREE.LineSegments(this.frustumGeometry, line(FRUSTUM_COLOR, 0.45));
    for (const object of [this.path, this.frustums, this.active, this.selectedCross, this.markers, this.selectedPoint]) {
      object.renderOrder = 10;
      object.frustumCulled = false;
      this.group.add(object);
    }
    this.group.visible = false;
  }

  get anchorCount() { return this.trajectory?.anchors.length ?? 0; }
  get cameraCount() { return this.trajectory?.cameras.length ?? 0; }

  setTrajectory(trajectory: SpatialTrajectory | null) {
    this.trajectory = trajectory;
    this.selected = -1;
    this.activeSegment = -1;
    this.frustumsBuilt = false;
    this.group.visible = !!trajectory;
    for (const geometry of [this.pathGeometry, this.markerGeometry, this.activeGeometry, this.selectedGeometry, this.selectedPointGeometry, this.frustumGeometry])
      geometry.setAttribute("position", EMPTY());
    if (!trajectory) return;
    const centres = new Float32Array(trajectory.anchors.length * 3);
    trajectory.anchors.forEach((anchor, i) => centres.set(anchor.position, i * 3));
    this.pathGeometry.setAttribute("position", new THREE.BufferAttribute(centres, 3));
    this.markerGeometry.setAttribute("position", new THREE.BufferAttribute(centres.slice(), 3));
    this.pathGeometry.computeBoundingSphere();
    this.markerGeometry.computeBoundingSphere();
    this.baseDepth = baseSize(trajectory);
    this.rebuildFrustums();
    this.applyVisibility();
  }

  setScale(scale: number) {
    if (!Number.isFinite(scale) || scale <= 0 || scale === this.scale) return;
    this.scale = scale;
    this.rebuildFrustums();
    this.rebuildSelected();
  }

  setVisibility(path: boolean, frustums: boolean) {
    this.pathVisible = path;
    const changed = frustums !== this.frustumsVisible;
    this.frustumsVisible = frustums;
    if (changed && frustums) this.rebuildFrustums();
    this.applyVisibility();
  }

  private applyVisibility() {
    const anchors = this.anchorCount;
    this.path.visible = this.pathVisible && anchors > 1;
    this.markers.visible = this.pathVisible && anchors > 0;
    this.frustums.visible = this.frustumsVisible && this.cameraCount > 0;
    const marked = this.selected >= 0 && (this.pathVisible || this.frustumsVisible);
    this.selectedCross.visible = marked;
    this.selectedPoint.visible = marked;
    this.active.visible = this.pathVisible && this.activeSegment >= 0 && anchors > 1;
  }

  setSelected(index: number) {
    const next = index >= 0 && index < this.anchorCount ? index : -1;
    if (next === this.selected) return;
    this.selected = next;
    this.rebuildSelected();
    this.applyVisibility();
  }

  /** Highlight the path segment the walker is currently being held against. */
  setActiveSegment(index: number) {
    const anchors = this.trajectory?.anchors ?? [];
    const next = index >= 0 && index + 1 < anchors.length ? index : -1;
    if (next === this.activeSegment) return;
    this.activeSegment = next;
    if (next < 0) this.activeGeometry.setAttribute("position", EMPTY());
    else {
      const data = new Float32Array(6);
      data.set(anchors[next].position, 0);
      data.set(anchors[next + 1].position, 3);
      this.activeGeometry.setAttribute("position", new THREE.BufferAttribute(data, 3));
      this.activeGeometry.computeBoundingSphere();
    }
    this.applyVisibility();
  }

  /** Vertices of one camera's frustum, written into `target` at `offset`. */
  private writeFrustum(camera: TrajectoryCamera, depth: number, target: Float32Array, offset: number) {
    const unit = frustumEdges(camera.width / camera.height, THREE.MathUtils.degToRad(camera.fovYDeg) / 2);
    const matrix = new THREE.Matrix4().compose(
      new THREE.Vector3(...camera.position),
      new THREE.Quaternion(...camera.quaternion),
      new THREE.Vector3(depth, depth, depth),
    );
    const vertex = new THREE.Vector3();
    for (let i = 0; i < unit.length; i += 3) {
      vertex.set(unit[i], unit[i + 1], unit[i + 2]).applyMatrix4(matrix);
      target[offset + i] = vertex.x;
      target[offset + i + 1] = vertex.y;
      target[offset + i + 2] = vertex.z;
    }
    return unit.length;
  }

  private rebuildFrustums() {
    if (!this.trajectory || !this.frustumsVisible) return;
    const depth = this.baseDepth * this.scale * 0.6;
    const perCamera = frustumEdges(1, 0.5).length;
    const data = new Float32Array(this.trajectory.cameras.length * perCamera);
    let offset = 0;
    for (const camera of this.trajectory.cameras) offset += this.writeFrustum(camera, depth, data, offset);
    this.frustumGeometry.setAttribute("position", new THREE.BufferAttribute(data, 3));
    this.frustumGeometry.computeBoundingSphere();
    this.frustumsBuilt = true;
  }

  private rebuildSelected() {
    const anchors = this.trajectory?.anchors ?? [];
    if (this.selected < 0 || this.selected >= anchors.length) {
      this.selectedGeometry.setAttribute("position", EMPTY());
      this.selectedPointGeometry.setAttribute("position", EMPTY());
      return;
    }
    const [x, y, z] = anchors[this.selected].position;
    // A larger dot *and* an axis cross, so selection is never signalled by
    // colour alone.
    const r = this.baseDepth * this.scale * 0.7;
    this.selectedGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
      x - r, y, z, x + r, y, z,
      x, y - r, z, x, y + r, z,
      x, y, z - r, x, y, z + r,
    ]), 3));
    this.selectedPointGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array([x, y, z]), 3));
    this.selectedGeometry.computeBoundingSphere();
    this.selectedPointGeometry.computeBoundingSphere();
  }

  /** Project one anchor to canvas pixels, or null if it is behind the camera. */
  private project(index: number, camera: THREE.Camera, width: number, height: number) {
    const anchors = this.trajectory?.anchors ?? [];
    const point = new THREE.Vector3(...anchors[index].position).project(camera);
    if (!Number.isFinite(point.x) || point.z < -1 || point.z > 1) return null;
    return { x: ((point.x + 1) / 2) * width, y: ((1 - point.y) / 2) * height, depth: point.z };
  }

  /**
   * The anchor marker under the pointer, or null.
   *
   * Picking is done in screen space rather than by raycasting the Points: a
   * Points raycast needs a world-space threshold, which is a different number of
   * pixels at every distance and every zoom level, so it either misses markers
   * the user can clearly see or grabs ones far behind them. Projecting the
   * anchors costs one matrix multiply each per click — nothing at 158 — and hits
   * exactly what is drawn.
   */
  pick(pointerX: number, pointerY: number, camera: THREE.Camera, width: number, height: number, pixelRadius = 16): number | null {
    if (!this.markers.visible || !this.anchorCount || !width || !height) return null;
    camera.updateMatrixWorld();
    let best: number | null = null;
    let bestDistance = pixelRadius;
    let bestDepth = Number.POSITIVE_INFINITY;
    for (let i = 0; i < this.anchorCount; i++) {
      const screen = this.project(i, camera, width, height);
      if (!screen) continue;
      const distance = Math.hypot(screen.x - pointerX, screen.y - pointerY);
      if (distance > pixelRadius) continue;
      // Nearest to the pointer wins; equally close markers resolve to the one
      // in front, which is the one the user is looking at.
      if (distance < bestDistance - 0.5 || (distance <= bestDistance + 0.5 && screen.depth < bestDepth)) {
        best = i;
        bestDistance = Math.min(bestDistance, distance);
        bestDepth = screen.depth;
      }
    }
    return best;
  }

  /**
   * Read-only observations for browser tests: what is drawn, and where each
   * anchor marker lands on screen so a test can click one. Installed only in the
   * development diagnostics build, so the projection cost is never shipped.
   */
  snapshot(camera: THREE.Camera, width: number, height: number, maxProjected = 1024) {
    const count = (geometry: THREE.BufferGeometry) => geometry.getAttribute("position")?.count ?? 0;
    const projected: { x: number; y: number; onScreen: boolean }[] = [];
    if (this.anchorCount <= maxProjected)
      for (let i = 0; i < this.anchorCount; i++) {
        const screen = this.project(i, camera, width, height);
        projected.push(screen
          ? {
              x: Math.round(screen.x),
              y: Math.round(screen.y),
              onScreen: screen.x >= 0 && screen.x <= width && screen.y >= 0 && screen.y <= height,
            }
          : { x: -1, y: -1, onScreen: false });
      }
    return {
      cameras: this.cameraCount,
      anchors: this.anchorCount,
      selected: this.selected,
      activeSegment: this.activeSegment,
      scale: this.scale,
      baseDepth: this.baseDepth,
      frustumsBuilt: this.frustumsBuilt,
      visible: {
        path: this.path.visible,
        markers: this.markers.visible,
        frustums: this.frustums.visible,
        selected: this.selectedCross.visible,
        active: this.active.visible,
      },
      vertices: {
        path: count(this.pathGeometry),
        markers: count(this.markerGeometry),
        frustums: count(this.frustumGeometry),
        selected: count(this.selectedGeometry),
        active: count(this.activeGeometry),
      },
      projected,
    };
  }

  dispose() {
    this.group.removeFromParent();
    this.group.clear();
    for (const geometry of [this.pathGeometry, this.markerGeometry, this.activeGeometry, this.selectedGeometry, this.selectedPointGeometry, this.frustumGeometry])
      geometry.dispose();
    for (const material of this.materials) material.dispose();
    this.materials.length = 0;
    this.trajectory = null;
  }
}
