// Walking input: mouse look, WASD, and nothing else.
//
// This owns *intent*, not the camera. Each frame it turns whatever the user did
// into a look direction and a world-space step; GaussianSplatViewer asks
// trajectoryTube.ts where that step is actually allowed to end, and only then is
// the camera moved. Keeping the two apart is what makes the boundary testable
// without a browser: the tube maths has no input in it, and this has no tube in
// it.
//
// Roll is never accumulated. Yaw and pitch are stored as angles against the
// scene's own up axis and the orientation is rebuilt from them every frame, so
// the horizon cannot drift the way it does when you multiply rotations onto a
// quaternion frame after frame.

import * as THREE from "three";

/** Radians of look per pixel of mouse movement. About 0.13 degrees. */
export const LOOK_SENSITIVITY = 0.0022;
/** Arrow-key look rate, radians per second. Mouse-free operation. */
export const KEY_LOOK_RATE = Math.PI / 2;
/** How far from straight up or down the pitch may get. Avoids a degenerate basis. */
export const MAX_PITCH = Math.PI / 2 - 0.03;
/** Multiplier while Shift is held. */
export const SPRINT_MULTIPLIER = 3;

const MOVE_KEYS: Record<string, [forward: number, right: number, up: number]> = {
  KeyW: [1, 0, 0],
  KeyS: [-1, 0, 0],
  KeyA: [0, -1, 0],
  KeyD: [0, 1, 0],
  KeyE: [0, 0, 1],
  KeyQ: [0, 0, -1],
  ArrowUp: [1, 0, 0],
  ArrowDown: [-1, 0, 0],
};
const LOOK_KEYS: Record<string, [yaw: number, pitch: number]> = {
  ArrowLeft: [1, 0],
  ArrowRight: [-1, 0],
  PageUp: [0, 1],
  PageDown: [0, -1],
};
const ORIGIN = /*@__PURE__*/ new THREE.Vector3();
/** Keys that would otherwise scroll the page while the canvas has focus. */
const SCROLLING_KEYS = new Set(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "PageUp", "PageDown"]);

export interface FreeLookStep {
  /** World-space movement requested this frame, before the tube is consulted. */
  delta: THREE.Vector3;
  /** Where the camera should be pointing, roll-free. */
  quaternion: THREE.Quaternion;
  moved: boolean;
  looked: boolean;
}

/**
 * Pointer-lock mouse look plus keyboard movement, active only while the mode is
 * on. Keys are read whenever the pointer is locked *or* the canvas has focus:
 * pointer lock cannot be granted without a user gesture and is refused outright
 * in some environments, and a walk that only works with a captured mouse is not
 * usable with a keyboard alone.
 */
export class FreeLookController {
  private enabled = false;
  private readonly keys = new Set<string>();
  private pendingYaw = 0;
  private pendingPitch = 0;
  private yaw = 0;
  private pitch = 0;
  private readonly up = new THREE.Vector3(0, 1, 0);
  private readonly refA = new THREE.Vector3(1, 0, 0);
  private readonly refB = new THREE.Vector3(0, 0, 1);
  private readonly forward = new THREE.Vector3(0, 0, -1);
  private readonly quaternion = new THREE.Quaternion();
  private readonly basis = new THREE.Matrix4();

  constructor(
    private readonly camera: THREE.PerspectiveCamera,
    private readonly canvas: HTMLCanvasElement,
    private readonly onPointerLockChange: (locked: boolean) => void = () => {},
    private readonly onEscape: () => void = () => {},
  ) {}

  get pointerLocked() {
    return this.canvas.ownerDocument.pointerLockElement === this.canvas;
  }

  /** True when keys should be read: the pointer is ours, or the canvas has focus. */
  get accepting() {
    return this.enabled && (this.pointerLocked || this.canvas.ownerDocument.activeElement === this.canvas);
  }

  /**
   * Start walking from wherever the camera is now, keeping the current view
   * direction. Entering the mode must not swing the view somewhere else.
   */
  enable() {
    if (this.enabled) return;
    this.enabled = true;
    this.adoptCamera();
    // Keys are listened for on the document, not the canvas: while the pointer
    // is locked the mouse events go to the locked element but the keyboard still
    // goes to whatever has focus, and `accepting` is the single gate that
    // decides whether this controller is entitled to the keystroke.
    const document = this.canvas.ownerDocument;
    document.addEventListener("mousemove", this.mousemove);
    document.addEventListener("keydown", this.keydown);
    document.addEventListener("keyup", this.keyup);
    this.canvas.addEventListener("blur", this.clear);
    document.addEventListener("pointerlockchange", this.pointerlockchange);
    // defaultView, not the global window: it is the right window for a canvas
    // in an iframe, and it lets the movement maths be exercised without a DOM.
    document.defaultView?.addEventListener("blur", this.clear);
    this.canvas.focus();
  }

  disable() {
    if (!this.enabled) return;
    this.enabled = false;
    this.clear();
    const document = this.canvas.ownerDocument;
    document.removeEventListener("mousemove", this.mousemove);
    document.removeEventListener("keydown", this.keydown);
    document.removeEventListener("keyup", this.keyup);
    this.canvas.removeEventListener("blur", this.clear);
    document.removeEventListener("pointerlockchange", this.pointerlockchange);
    document.defaultView?.removeEventListener("blur", this.clear);
    if (this.pointerLocked) document.exitPointerLock();
  }

  dispose() { this.disable(); }

  /** Read yaw and pitch back off the live camera. Also the up-axis handshake. */
  adoptCamera() {
    this.up.copy(this.camera.up).normalize();
    if (!Number.isFinite(this.up.lengthSq()) || this.up.lengthSq() < 1e-12) this.up.set(0, 1, 0);
    // Any fixed horizontal reference will do; yaw is measured against it and the
    // absolute value never reaches the user.
    this.refA.set(1, 0, 0);
    if (Math.abs(this.refA.dot(this.up)) > 0.9) this.refA.set(0, 0, 1);
    this.refA.addScaledVector(this.up, -this.refA.dot(this.up)).normalize();
    this.refB.crossVectors(this.up, this.refA).normalize();

    const look = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion).normalize();
    this.pitch = THREE.MathUtils.clamp(Math.asin(THREE.MathUtils.clamp(look.dot(this.up), -1, 1)), -MAX_PITCH, MAX_PITCH);
    const horizontal = look.clone().addScaledVector(this.up, -look.dot(this.up));
    this.yaw = horizontal.lengthSq() < 1e-12 ? 0 : Math.atan2(horizontal.dot(this.refB), horizontal.dot(this.refA));
    this.pendingYaw = 0;
    this.pendingPitch = 0;
    this.rebuild();
  }

  /** Ask for the mouse. Must be called from a user gesture; refusal is normal. */
  requestPointerLock() {
    if (!this.enabled || this.pointerLocked) return;
    const request = this.canvas.requestPointerLock() as unknown;
    if (request instanceof Promise) request.catch(() => {});
  }

  /**
   * Advance one frame. `speed` is world units per second; the caller derives it
   * from the scene scale, because a metre means nothing without one.
   */
  update(deltaSeconds: number, speed: number): FreeLookStep {
    const dt = Number.isFinite(deltaSeconds) ? THREE.MathUtils.clamp(deltaSeconds, 0, 0.25) : 0;
    let looked = this.pendingYaw !== 0 || this.pendingPitch !== 0;
    this.yaw -= this.pendingYaw;
    this.pitch -= this.pendingPitch;
    this.pendingYaw = 0;
    this.pendingPitch = 0;
    if (this.accepting)
      for (const [code, [yaw, pitch]] of Object.entries(LOOK_KEYS))
        if (this.keys.has(code)) {
          this.yaw += yaw * KEY_LOOK_RATE * dt;
          this.pitch += pitch * KEY_LOOK_RATE * dt;
          looked = true;
        }
    this.pitch = THREE.MathUtils.clamp(this.pitch, -MAX_PITCH, MAX_PITCH);
    this.rebuild();

    const move = new THREE.Vector3();
    if (this.accepting)
      for (const [code, axes] of Object.entries(MOVE_KEYS))
        if (this.keys.has(code)) move.set(move.x + axes[0], move.y + axes[1], move.z + axes[2]);
    const delta = new THREE.Vector3();
    if (move.lengthSq() > 0) {
      // Forward is the look direction flattened onto the scene's horizontal
      // plane, so looking up does not lift the walker off the path.
      const flat = this.forward.clone().addScaledVector(this.up, -this.forward.dot(this.up));
      if (flat.lengthSq() < 1e-12) flat.copy(this.refA);
      flat.normalize();
      const right = new THREE.Vector3().crossVectors(flat, this.up).normalize();
      delta.addScaledVector(flat, move.x).addScaledVector(right, move.y).addScaledVector(this.up, move.z);
      if (delta.lengthSq() > 0)
        delta.normalize().multiplyScalar(speed * dt * (this.keys.has("ShiftLeft") || this.keys.has("ShiftRight") ? SPRINT_MULTIPLIER : 1));
    }
    return { delta, quaternion: this.quaternion.clone(), moved: delta.lengthSq() > 0, looked };
  }

  /** The roll-free orientation for the current yaw and pitch. */
  private rebuild() {
    const cos = Math.cos(this.pitch);
    this.forward
      .copy(this.refA).multiplyScalar(cos * Math.cos(this.yaw))
      .addScaledVector(this.refB, cos * Math.sin(this.yaw))
      .addScaledVector(this.up, Math.sin(this.pitch))
      .normalize();
    // Matrix4.lookAt(eye, target, up) puts -z on target and derives
    // right = up x z, up' = z x right -- which is exactly "no roll": the
    // camera's right axis stays perpendicular to the scene's up axis.
    this.basis.lookAt(ORIGIN, this.forward, this.up);
    this.quaternion.setFromRotationMatrix(this.basis).normalize();
  }

  private readonly clear = () => { this.keys.clear(); this.pendingYaw = 0; this.pendingPitch = 0; };

  private readonly mousemove = (event: MouseEvent) => {
    if (!this.enabled || !this.pointerLocked) return;
    this.pendingYaw += event.movementX * LOOK_SENSITIVITY;
    this.pendingPitch += event.movementY * LOOK_SENSITIVITY;
  };

  private readonly keydown = (event: KeyboardEvent) => {
    if (!this.enabled) return;
    if (event.code === "Escape") { this.onEscape(); return; }
    if (!this.accepting || event.ctrlKey || event.metaKey || event.altKey) return;
    if (!Object.hasOwn(MOVE_KEYS, event.code) && !Object.hasOwn(LOOK_KEYS, event.code) && !event.code.startsWith("Shift")) return;
    // Movement keys are plain letters and do nothing else in a browser; the
    // arrows and page keys would scroll the page out from under the canvas.
    if (SCROLLING_KEYS.has(event.code)) event.preventDefault();
    this.keys.add(event.code);
  };

  private readonly keyup = (event: KeyboardEvent) => { this.keys.delete(event.code); };

  /** Focus the canvas so keys are accepted even if pointer lock is refused. */
  focusCanvas() { if (this.enabled) this.canvas.focus(); }

  private readonly pointerlockchange = () => {
    if (!this.pointerLocked) this.clear();
    this.onPointerLockChange(this.pointerLocked);
  };
}
