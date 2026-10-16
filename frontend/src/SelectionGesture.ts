import * as THREE from "three";
import { MAX_POLYGON_VERTICES } from "./regionSelection";
import type { CameraNavigation } from "./CameraNavigation";
import type { InteractionMode, SelectionOperation, SelectionOperationMode } from "./types";

type Polygon = [number, number][];
interface Options {
  mode: InteractionMode; operation: SelectionOperationMode; busy: boolean;
  onSelection: (operation: SelectionOperation) => void;
  onError: (message: string) => void;
}
// Capture before OrbitControls receives a region drag. Projection happens once
// on release; pointermove only updates this inexpensive SVG outline.
export class SelectionGesture {
  private drag: { id: number; points: Polygon; shape: "rectangle" | "lasso"; mode: SelectionOperationMode; enabled: boolean } | null = null;
  constructor(private canvas: HTMLCanvasElement, private navigation: CameraNavigation,
    private options: () => Options, private overlay: (polygon: Polygon | null) => void) {
    canvas.addEventListener("pointerdown", this.down, true);
    canvas.addEventListener("pointermove", this.move, true);
    canvas.addEventListener("pointerup", this.up, true);
    canvas.addEventListener("pointercancel", this.abort, true);
    canvas.addEventListener("lostpointercapture", this.abort, true);
    window.addEventListener("keydown", this.key);
  }
  private point(event: PointerEvent): [number, number] {
    const box = this.canvas.getBoundingClientRect();
    return [Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)), Math.max(0, Math.min(1, (event.clientY - box.top) / box.height))];
  }
  private consume(event: PointerEvent) { event.preventDefault(); event.stopImmediatePropagation(); }
  private polygon(): Polygon {
    const drag = this.drag!;
    if (drag.shape === "lasso") return [...drag.points];
    const [a, b = a] = drag.points;
    return [a, [b[0], a[1]], b, [a[0], b[1]]];
  }
  private down = (event: PointerEvent) => {
    const options = this.options();
    if (options.mode !== "rectangle" && options.mode !== "lasso") return;
    if (event.button !== 0) return;
    this.consume(event);
    if (!event.isPrimary || options.busy) return;
    this.cancel();
    const controls = this.navigation.controls;
    controls.enableDamping = false; controls.update(); controls.enableDamping = true;
    this.drag = { id: event.pointerId, points: [this.point(event)], shape: options.mode,
      mode: event.altKey ? "subtract" : event.shiftKey ? "add" : options.operation, enabled: controls.enabled };
    controls.enabled = false;
    this.canvas.focus({ preventScroll: true });
    this.canvas.setPointerCapture(event.pointerId);
    this.overlay(this.polygon());
  };
  private move = (event: PointerEvent) => {
    if (!this.drag || this.drag.id !== event.pointerId) return;
    this.consume(event);
    const point = this.point(event);
    if (this.drag.shape === "rectangle") this.drag.points[1] = point;
    else {
      const last = this.drag.points.at(-1)!;
      const box = this.canvas.getBoundingClientRect();
      if (Math.hypot((last[0] - point[0]) * box.width, (last[1] - point[1]) * box.height) < 3) return;
      if (this.drag.points.length >= MAX_POLYGON_VERTICES) {
        this.cancel();
        this.options().onError("This lasso is too detailed (512 corners). Draw a shorter outline; your previous selection is preserved.");
        return;
      }
      this.drag.points.push(point);
    }
    this.overlay(this.polygon());
  };
  private up = (event: PointerEvent) => {
    if (!this.drag || this.drag.id !== event.pointerId) return;
    this.move(event);
    if (!this.drag) return;
    const polygon = this.polygon(), { mode, shape } = this.drag;
    this.cancel();
    const box = this.canvas.getBoundingClientRect();
    const xs = polygon.map((p) => p[0]), ys = polygon.map((p) => p[1]);
    if (polygon.length < 3 || (Math.max(...xs) - Math.min(...xs)) * box.width < 3 ||
        (Math.max(...ys) - Math.min(...ys)) * box.height < 3) return;
    const camera = this.navigation.camera;
    camera.updateMatrixWorld();
    const matrix = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).toArray();
    this.options().onSelection({ mode, shape, polygon, matrix });
  };
  private abort = () => { this.cancel(); };
  private key = (event: KeyboardEvent) => { if (event.key === "Escape" && this.drag) { event.preventDefault(); this.cancel(); } };
  cancel(notify = true) {
    const drag = this.drag;
    this.drag = null;
    if (!drag) return;
    this.navigation.controls.enabled = drag.enabled;
    if (this.canvas.hasPointerCapture(drag.id)) this.canvas.releasePointerCapture(drag.id);
    if (notify) this.overlay(null);
  }
  dispose() {
    this.cancel(false);
    this.canvas.removeEventListener("pointerdown", this.down, true);
    this.canvas.removeEventListener("pointermove", this.move, true);
    this.canvas.removeEventListener("pointerup", this.up, true);
    this.canvas.removeEventListener("pointercancel", this.abort, true);
    this.canvas.removeEventListener("lostpointercapture", this.abort, true);
    window.removeEventListener("keydown", this.key);
  }
}
