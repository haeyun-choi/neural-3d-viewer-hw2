import * as THREE from "three";
import { CameraNavigation } from "./CameraNavigation";

export class WebGLUnavailableError extends Error {}

const AXES = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
const inverse = new THREE.Quaternion();
const axis = new THREE.Vector3();

/** Source axes projected into the current camera basis, including roll. */
export function drawAxisIndicator(svg: SVGSVGElement | null, camera: THREE.Camera) {
  if (!svg) return;
  inverse.copy(camera.quaternion).invert();
  AXES.forEach((source, i) => {
    axis.copy(source).applyQuaternion(inverse);
    const group = svg.children[i];
    group.children[0].setAttribute("x2", String(42 + axis.x * 26));
    group.children[0].setAttribute("y2", String(42 - axis.y * 26));
    group.children[1].setAttribute("x", String(42 + axis.x * 34));
    group.children[1].setAttribute("y", String(46 - axis.y * 34));
  });
}

export const WEBGL_UNAVAILABLE =
  "WebGL is unavailable. Enable hardware acceleration or try a WebGL-capable browser.";

// The DOM lib in use does not declare HTMLCanvasElementEventMap, so the WebGL
// context events the viewers listen for are named explicitly.
type CanvasEvents = HTMLElementEventMap & { webglcontextlost: Event; webglcontextrestored: Event };

interface RuntimeOptions {
  /** Spark's splat shader gains nothing from MSAA and loses performance, so the
   *  Gaussian renderer opts out while the point renderer keeps antialiasing. */
  antialias: boolean;
  label: string;
}

/**
 * The lifecycle both representations share: one WebGL canvas, one perspective
 * camera, one CameraNavigation (orbit/pan/zoom, roll, up axis, presets, home),
 * one ResizeObserver and one animation loop, all released together.
 *
 * Each representation owns its own renderer objects on top of this: the point
 * renderer owns THREE.Points, markers and highlight layers; the Gaussian
 * renderer owns a SparkRenderer and its SplatMesh. Exactly one runtime exists at
 * a time because App mounts exactly one viewer component.
 */
export class ViewerRuntime {
  readonly renderer: THREE.WebGLRenderer;
  readonly canvas: HTMLCanvasElement;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly navigation: CameraNavigation;
  private readonly observer: ResizeObserver;
  private readonly listeners: { type: string; handler: EventListener }[] = [];
  private resizeHandler: ((width: number, height: number) => void) | null = null;

  constructor(private readonly container: HTMLElement, options: RuntimeOptions) {
    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: options.antialias, alpha: false });
    } catch (error) {
      throw new WebGLUnavailableError(error instanceof Error ? error.message : WEBGL_UNAVAILABLE);
    }
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.canvas = this.renderer.domElement;
    this.canvas.tabIndex = 0;
    this.canvas.setAttribute("role", "img");
    this.canvas.setAttribute("aria-label", options.label);
    container.appendChild(this.canvas);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
    this.camera.position.set(5, 3, 6);
    this.navigation = new CameraNavigation(this.camera, this.canvas);
    this.observer = new ResizeObserver(() => this.applyResize());
    this.observer.observe(container);
  }

  private applyResize() {
    const width = Math.max(this.container.clientWidth, 1);
    const height = Math.max(this.container.clientHeight, 1);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    this.resizeHandler?.(width, height);
  }

  /** Register the representation-specific part of a resize and apply one now. */
  onResize(handler: (width: number, height: number) => void) {
    this.resizeHandler = handler;
    this.applyResize();
  }

  /** Canvas listeners registered here are removed by dispose(). */
  on<K extends keyof CanvasEvents>(type: K, handler: (event: CanvasEvents[K]) => void) {
    this.canvas.addEventListener(type, handler as EventListener);
    this.listeners.push({ type, handler: handler as EventListener });
  }

  setAnimationLoop(loop: (() => void) | null) {
    this.renderer.setAnimationLoop(loop);
  }

  render() {
    if (!this.renderer.getContext().isContextLost()) this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.observer.disconnect();
    this.resizeHandler = null;
    this.renderer.setAnimationLoop(null);
    for (const { type, handler } of this.listeners) this.canvas.removeEventListener(type, handler);
    this.listeners.length = 0;
    this.navigation.dispose();
    this.renderer.dispose();
    this.renderer.forceContextLoss();
    this.canvas.remove();
  }
}
