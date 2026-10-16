import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { CameraPose, Position, Vector3Tuple, ViewPreferences, ViewPreset } from "./types";

const importedDirection = new THREE.Vector3(1.25, 0.9, 1.7).normalize();
export function viewBasis(view: ViewPreferences | null) {
  const front = new THREE.Vector3(...(view?.front ?? [0, 0, 1])).normalize();
  const up = new THREE.Vector3(...(view?.up ?? [0, 1, 0])).normalize();
  const right = new THREE.Vector3().crossVectors(up, front).normalize();
  return { front, right, up: new THREE.Vector3().crossVectors(front, right).normalize() };
}
export function presetBasis(view: ViewPreferences | null, preset: ViewPreset) {
  const { front, right, up } = viewBasis(view);
  switch (preset) {
    case "Front": return { direction: front, up };
    case "Back": return { direction: front.negate(), up };
    case "Left": return { direction: right.negate(), up };
    case "Right": return { direction: right, up };
    case "Top": return { direction: up, up: front.negate() };
    case "Bottom": return { direction: up.negate(), up: front };
  }
}

// A single live controls instance. Recreate it only when up changes: OrbitControls
// caches its up-to-Y quaternion in the constructor. Always dispose the old one.
export class CameraNavigation {
  controls: OrbitControls;
  manual = false;
  center = new THREE.Vector3();
  radius = 1;
  preferences: ViewPreferences | null = null;
  private interacting = false;
  private programmatic = false;
  private start = () => { this.interacting = true; };
  private end = () => { this.interacting = false; };
  private change = () => { if (this.interacting && !this.programmatic) this.manual = true; };
  private key = (event: KeyboardEvent) => { if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) this.manual = true; };
  /** The viewer's own field of view, restored whenever a training FOV is left. */
  readonly defaultFov: number;
  constructor(readonly camera: THREE.PerspectiveCamera, readonly canvas: HTMLCanvasElement) {
    this.defaultFov = camera.fov;
    this.controls = this.makeControls();
    canvas.addEventListener("keydown", this.key);
  }
  private makeControls() {
    const controls = new OrbitControls(this.camera, this.canvas);
    controls.enableDamping = true; controls.dampingFactor = 0.12;
    controls.listenToKeyEvents(this.canvas);
    controls.addEventListener("start", this.start);
    controls.addEventListener("end", this.end);
    controls.addEventListener("change", this.change);
    return controls;
  }
  private discardControls() {
    this.controls.removeEventListener("start", this.start);
    this.controls.removeEventListener("end", this.end);
    this.controls.removeEventListener("change", this.change);
    this.controls.dispose();
  }
  private angle(aspect = this.camera.aspect) {
    const half = THREE.MathUtils.degToRad(this.camera.fov / 2);
    return Math.min(half, Math.atan(Math.tan(half) * aspect));
  }
  private distance(radius = this.radius) { return Math.max(radius, 0.01) / Math.sin(this.angle()) * 1.12; }
  private pose(position: THREE.Vector3, target: THREE.Vector3, up: THREE.Vector3, manual: boolean, radius = this.radius, clipping?: { near: number; far: number }) {
    this.programmatic = true;
    this.controls.enableDamping = false; this.controls.update(); // consume old motion
    const enabled = this.controls.enabled;
    if (this.camera.up.distanceToSquared(up) > 1e-20) {
      this.discardControls();
      this.camera.up.copy(up).normalize();
      this.controls = this.makeControls();
    }
    this.camera.position.copy(position);
    this.controls.target.copy(target);
    this.camera.near = clipping?.near ?? Math.max(Math.min(radius, this.radius) / 1000, 0.00001);
    this.camera.far = clipping?.far ?? Math.max(this.radius * 150, position.distanceTo(this.center) + this.radius * 2);
    this.camera.updateProjectionMatrix();
    // A saved tightly focused region can be much smaller than the whole cloud.
    // Never let OrbitControls clamp an explicitly requested camera pose outward.
    this.controls.minDistance = Math.min(Math.max(Math.min(radius, this.radius) * 0.008, 0.0001), position.distanceTo(target) * 0.5);
    this.controls.maxDistance = this.radius * 100;
    this.controls.enableDamping = false; this.controls.update();
    this.controls.enableDamping = true; this.controls.enabled = enabled;
    this.controls.saveState();
    this.programmatic = false; this.interacting = false; this.manual = manual;
  }
  reset() {
    this.restoreDefaultFov();
    const home = this.preferences?.home;
    if (home) {
      const target = new THREE.Vector3(...home.target);
      const offset = new THREE.Vector3(...home.position).sub(target);
      // A narrow untouched viewport needs more distance, keeping saved framing.
      offset.multiplyScalar(Math.max(1, Math.sin(this.angle(home.aspect)) / Math.sin(this.angle())));
      this.pose(target.clone().add(offset), target, new THREE.Vector3(...home.up), false, this.radius, home);
    } else {
      const direction = this.preferences ? new THREE.Vector3(...this.preferences.front) : importedDirection.clone();
      this.pose(this.center.clone().addScaledVector(direction, this.distance()), this.center,
        new THREE.Vector3(...(this.preferences?.up ?? [0, 1, 0])), false);
    }
  }
  resize(aspect: number) {
    this.camera.aspect = aspect; this.camera.updateProjectionMatrix();
    if (!this.manual) this.reset();
  }
  focus(position: Position) {
    const target = new THREE.Vector3(position.x, position.y, position.z);
    const offset = this.camera.position.clone().sub(this.controls.target);
    this.pose(target.clone().add(offset), target, this.camera.up.clone(), true);
  }
  focusRegion(bounds: { min: Position; max: Position }) {
    const box = new THREE.Box3(new THREE.Vector3(bounds.min.x, bounds.min.y, bounds.min.z), new THREE.Vector3(bounds.max.x, bounds.max.y, bounds.max.z));
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const direction = this.camera.position.clone().sub(this.controls.target).normalize();
    this.pose(sphere.center.clone().addScaledVector(direction, this.distance(sphere.radius)), sphere.center, this.camera.up.clone(), true, sphere.radius);
  }
  roll(degrees: number) {
    if (!Number.isFinite(degrees)) return;
    const forward = this.controls.target.clone().sub(this.camera.position).normalize();
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion).applyAxisAngle(forward, THREE.MathUtils.degToRad(degrees)).normalize();
    this.pose(this.camera.position.clone(), this.controls.target.clone(), up, true);
  }
  setUp(axis: "X" | "Y" | "Z" | "invert") {
    const up = axis === "invert" ? this.camera.up.clone().negate() : new THREE.Vector3(axis === "X" ? 1 : 0, axis === "Y" ? 1 : 0, axis === "Z" ? 1 : 0);
    let offset = this.camera.position.clone().sub(this.controls.target);
    // Avoid a pole when choosing an axis aligned with the current viewing ray.
    if (Math.abs(offset.clone().normalize().dot(up)) > 0.9999) {
      const tangent = new THREE.Vector3(1, 0, 0);
      if (Math.abs(tangent.dot(up)) > 0.9) tangent.set(0, 0, 1);
      offset = tangent.cross(up).normalize().multiplyScalar(offset.length());
    }
    this.pose(this.controls.target.clone().add(offset), this.controls.target.clone(), up, true);
  }
  capture(home: boolean): ViewPreferences {
    const front = this.camera.position.clone().sub(this.controls.target).normalize().toArray() as Vector3Tuple;
    const up = this.camera.up.clone().normalize().toArray() as Vector3Tuple;
    const pose: CameraPose = { position: this.camera.position.toArray() as Vector3Tuple, target: this.controls.target.toArray() as Vector3Tuple, up, aspect: this.camera.aspect, near: this.camera.near, far: this.camera.far };
    return { version: 1, up, front, home: home ? pose : null };
  }
  preset(name: ViewPreset) {
    const basis = presetBasis(this.preferences, name);
    this.pose(this.center.clone().addScaledVector(basis.direction, this.distance()), this.center, basis.up, true);
  }
  restoreImported() { this.preferences = null; this.reset(); }

  // --- Capture path -------------------------------------------------------
  //
  // Two entry points on purpose. jumpToPoint() lands on an anchor and then
  // hands control back to OrbitControls, which may mean rebuilding it because
  // OrbitControls caches its up axis at construction. driveFreeLook() is the
  // per-frame path while the user walks the capture path: rebuilding controls
  // sixty times a second would be absurd, so it moves the camera directly and
  // the viewer skips controls.update() while it is driving.
  // syncControlsToCamera() puts the two back in agreement on the way out.

  /** Orbit distance that keeps the scene, not the camera's nose, at the centre. */
  private orbitDistance(position: THREE.Vector3, forward: THREE.Vector3) {
    const ahead = this.center.clone().sub(position).dot(forward);
    return ahead > this.radius * 0.05 ? ahead : Math.max(this.radius * 0.5, 0.01);
  }

  /** Put the camera at a point on the path, keeping the direction it faces. */
  jumpToPoint(position: THREE.Vector3) {
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion).normalize();
    this.pose(position.clone(), position.clone().addScaledVector(forward, this.orbitDistance(position, forward)),
      this.camera.up.clone(), true);
  }

  /**
   * Move the camera while walking, without disturbing OrbitControls.
   *
   * `camera.up` is deliberately left alone: it is the scene's up axis, the one
   * the orientation controls set and the one FreeLookController yaws about, so
   * writing the walker's own up into it every frame would let the horizon drift.
   */
  driveFreeLook(position: THREE.Vector3, quaternion: THREE.Quaternion) {
    this.camera.position.copy(position);
    this.camera.quaternion.copy(quaternion);
    this.camera.updateMatrixWorld();
    this.manual = true;
  }

  /** Re-seat OrbitControls on wherever the camera ended up. */
  syncControlsToCamera() {
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion).normalize();
    const eye = this.camera.position.clone();
    this.pose(eye, eye.clone().addScaledVector(forward, this.orbitDistance(eye, forward)), this.camera.up.clone(), true);
  }

  /** Leave a training field of view behind; the fitted views assume the default. */
  restoreDefaultFov() {
    if (this.camera.fov === this.defaultFov) return;
    this.camera.fov = this.defaultFov;
    this.camera.updateProjectionMatrix();
  }

  dispose() { this.discardControls(); this.canvas.removeEventListener("keydown", this.key); }
}
