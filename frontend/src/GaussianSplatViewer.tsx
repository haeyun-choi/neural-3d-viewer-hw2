import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import * as THREE from "three";
import { loadSpark } from "./sparkModule";
import { splatBoundingSphere } from "./gaussianFit";
import { decodedSplatCount } from "./gaussianSplat";
import { drawAxisIndicator, ViewerRuntime, WebGLUnavailableError, WEBGL_UNAVAILABLE } from "./viewerRuntime";
import { CameraTrajectoryView } from "./CameraTrajectoryView";
import { FreeLookController } from "./FreeLookController";
import { buildTubePath, closestOnPath, constrainToTube } from "./trajectoryTube";
import { nearestAnchor } from "./cameraTrajectory";
import type { SpatialTrajectory } from "./cameraTrajectory";
import type { TubePath } from "./trajectoryTube";
import type { ExploreStatus, TrajectoryDisplay } from "./useTrajectory";
import type { SparkModule } from "./sparkModule";
import type { CameraHandle, GaussianSplatData, ViewerStatus, ViewPreferences } from "./types";
import type { SplatSnapshot } from "./viewerDiagnostics";

interface Props {
  splat: GaussianSplatData | null;
  view: ViewPreferences | null;
  background: string;
  loadStatus: ViewerStatus;
  onReload: () => void;
  onStatus: (status: ViewerStatus) => void;
  /** The collapsed capture path, or null when none is loaded. */
  trajectory: SpatialTrajectory | null;
  /** Anchor the user last chose; -1 means none. */
  selectedAnchor: number;
  /** True while the walker owns the camera and the tube constrains it. */
  exploring: boolean;
  /** Tube radius, world units. How far off the path the viewer may stray. */
  radius: number;
  /** Walking speed, world units per second. */
  speed: number;
  trajectoryDisplay: TrajectoryDisplay;
  /** Bumped by the panel to ask for a jump back onto the path. */
  recenter: number;
  /** The user clicked an anchor marker in the scene. */
  onAnchorChange: (index: number) => void;
  /** Throttled report of where the walker is relative to the tube. */
  onExploreStatus: (status: ExploreStatus) => void;
  /** Escape outside pointer lock, or a path that went away underneath us. */
  onExitExplore: () => void;
}

type SparkRendererInstance = InstanceType<SparkModule["SparkRenderer"]>;
type SplatMeshInstance = InstanceType<SparkModule["SplatMesh"]>;
interface Runtime extends CameraHandle { update: () => void }

/**
 * A paged scene has no splats until the renderer has fetched its first level-of-
 * detail page, which needs the SparkRenderer that only exists here. If no page
 * arrives within this budget the host is reachable (the import already probed
 * it) but not serving chunks, and saying so beats spinning forever.
 */
const FIRST_PAGE_TIMEOUT_MS = 90_000;

export const GaussianSplatViewer = forwardRef<CameraHandle, Props>(
  function GaussianSplatViewer(props, ref) {
    const axes = useRef<SVGSVGElement>(null);
    const host = useRef<HTMLDivElement>(null);
    const runtime = useRef<Runtime | null>(null);
    const latest = useRef(props);
    const [attempt, setAttempt] = useState(0);
    const [status, setStatus] = useState<ViewerStatus>({ state: "loading", message: "Preparing the splat renderer…" });

    useEffect(() => {
      latest.current = props;
      runtime.current?.update();
    });
    useImperativeHandle(ref, () => ({
      reset: () => runtime.current?.reset(),
      roll: (degrees) => runtime.current?.roll(degrees),
      setUp: (axis) => runtime.current?.setUp(axis),
      captureView: (home) => {
        if (!runtime.current) throw new Error("Viewer is not ready.");
        return runtime.current.captureView(home);
      },
      preset: (name) => runtime.current?.preset(name),
      restoreImported: () => runtime.current?.restoreImported(),
    }), []);

    useEffect(() => {
      const container = host.current!;
      let disposed = false;
      const publish = (next: ViewerStatus) => {
        if (!disposed) {
          setStatus(next);
          latest.current.onStatus(next);
        }
      };
      publish({ state: "loading", message: "Preparing the splat renderer…" });
      let view: ViewerRuntime;
      try {
        view = new ViewerRuntime(container, {
          antialias: false,
          label: "Interactive 3D Gaussian Splat scene. Drag to orbit, right-drag to pan, scroll to zoom.",
        });
      } catch (error) {
        publish({
          state: "error",
          message: error instanceof WebGLUnavailableError
            ? `${WEBGL_UNAVAILABLE} Spark requires WebGL2.`
            : "The splat renderer could not be created.",
        });
        return () => { disposed = true; };
      }
      if (!view.renderer.capabilities.isWebGL2) {
        view.dispose();
        publish({ state: "error", message: "Gaussian Splat rendering requires WebGL2, which this browser did not provide." });
        return () => { disposed = true; };
      }

      const { navigation, scene, camera } = view;
      let sparkModule: SparkModule | null = null;
      let spark: SparkRendererInstance | null = null;
      let mesh: SplatMeshInstance | null = null;
      let activeSplat: GaussianSplatData | null = null;
      let requested: GaussianSplatData | null = null;
      let splatCount = 0;
      let pagedSince = 0;
      let pagedFitted = false;
      // Capture-path state. Walking is driven by the render clock, so the live
      // position, the active segment and the boundary flag all live here; React
      // sees only what a render needs, and only when it changes.
      const trajectoryView = new CameraTrajectoryView();
      scene.add(trajectoryView.group);
      const freeLook = new FreeLookController(
        camera, view.canvas,
        () => reportStatus(true),
        // Escape releases the mouse first (the browser does that itself); a
        // second Escape, with the mouse already free, leaves the mode.
        () => { if (exploring && !freeLook.pointerLocked) latest.current.onExitExplore(); },
      );
      let spatial: SpatialTrajectory | null = null;
      let tube: TubePath | null = null;
      let exploring = false;
      let activeSegment = -1;
      let lastDistance = 0;
      let lastClamped = false;
      let lastStatusAt = 0;
      let reported: ExploreStatus | null = null;
      let appliedAnchor = -1;
      let appliedRecenter = 0;
      let lastFrameTime = 0;
      // Every adoption is tagged; a superseded or unmounted one puts back the
      // mesh it took out of the scene instead of touching a newer one.
      let generation = 0;

      /**
       * Detach a mesh without disposing it. The scene that owns the mesh
       * (useScene) disposes it when it is replaced, so React Strict Mode's
       * mount/unmount/mount cycle cannot destroy the scene being displayed.
       */
      const detach = (target: SplatMeshInstance | null) => {
        if (!target) return;
        scene.remove(target);
        if (target.paged && spark?.pager) spark.pager.removeSplats(target.paged);
      };

      const frame = (sphere: THREE.Sphere | null) => {
        if (sphere && Number.isFinite(sphere.radius)) {
          navigation.center.copy(sphere.center);
          navigation.radius = Math.max(sphere.radius, 0.01);
        }
        navigation.reset();
      };

      const readyStatus = (data: GaussianSplatData, target: SplatMeshInstance): ViewerStatus => {
        const decoded = decodedSplatCount(target);
        const count = data.declaredSplatCount ?? decoded;
        splatCount = count;
        const note = data.paged
          ? "paged level of detail"
          : data.lod
            ? `level of detail · ${decoded.toLocaleString()} nodes`
            : undefined;
        return { state: "ready", count, note };
      };

      const ensureSpark = async () => {
        const module = sparkModule ?? (sparkModule = await loadSpark());
        spark ??= (() => {
          const created = new module.SparkRenderer({ renderer: view.renderer });
          scene.add(created);
          return created;
        })();
        return module;
      };

      /**
       * Take over a mesh the import already decoded. The scene-graph swap is
       * synchronous so that it always happens before useScene's effect disposes
       * the mesh being replaced; only fitting and the ready report wait for the
       * Spark module.
       */
      const adopt = (data: GaussianSplatData) => {
        const token = ++generation;
        const next = data.mesh as unknown as SplatMeshInstance;
        detach(mesh);
        mesh = next;
        scene.add(next);
        activeSplat = data;
        splatCount = data.declaredSplatCount ?? 0;
        pagedFitted = false;
        pagedSince = performance.now();
        publish({ state: "loading", message: data.paged ? "Streaming level-of-detail pages…" : "Placing the decoded splats…" });
        void ensureSpark().then((module) => {
          if (disposed || token !== generation) return;
          // A paged scene has nothing resident yet: the animation loop fits it
          // and reports ready once the renderer's pager delivers the root page.
          if (data.paged) return;
          frame(data.view
            ? new THREE.Sphere(new THREE.Vector3(...data.view.center), data.view.radius)
            : splatBoundingSphere(module, next));
          publish(readyStatus(data, next));
        }).catch((error: unknown) => {
          if (disposed || token !== generation) return;
          publish({
            state: "error",
            message: `The splat renderer could not be started: ${error instanceof Error ? error.message : String(error)}`,
          });
        });
      };

      /**
       * Put the camera somewhere valid, without turning it.
       *
       * Somewhere the walker can move *from*: the nearest point of the path
       * itself when they were outside the tube, and exactly where they are when
       * they were already inside. Landing on the boundary — which is the literal
       * nearest valid point from an overview camera — would start every walk
       * pinned against the wall with the readout already saying so.
       */
      const enterTube = (target?: THREE.Vector3) => {
        if (!tube) return;
        const wanted = target ?? camera.position;
        // A global search, not the windowed one: there is no active segment to
        // carry over, so the honest answer is the nearest branch of the path.
        const home = closestOnPath(tube, wanted);
        activeSegment = home.segment;
        if (home.distance > latest.current.radius) camera.position.copy(home.point);
        else camera.position.copy(wanted);
        camera.updateMatrixWorld();
        lastDistance = home.distance > latest.current.radius ? 0 : home.distance;
        lastClamped = false;
      };

      const beginExploring = () => {
        exploring = true;
        navigation.controls.enabled = false;
        navigation.restoreDefaultFov();
        enterTube();
        freeLook.enable();
        // adoptCamera reads yaw and pitch off the camera as it is now, which is
        // what keeps the view direction the user already had.
        freeLook.adoptCamera();
        trajectoryView.setActiveSegment(activeSegment);
        lastFrameTime = performance.now();
        reportStatus(true);
      };

      const endExploring = () => {
        exploring = false;
        freeLook.disable();
        reported = null;
        activeSegment = -1;
        trajectoryView.setActiveSegment(-1);
        navigation.controls.enabled = true;
        navigation.syncControlsToCamera();
      };

      const update = () => {
        const current = latest.current;
        navigation.preferences = current.view;
        scene.background = new THREE.Color(current.background);
        if (current.splat && current.splat !== requested) {
          requested = current.splat;
          adopt(current.splat);
        }
        if (current.trajectory !== spatial) {
          if (exploring) endExploring();
          spatial = current.trajectory;
          tube = spatial ? buildTubePath(spatial.anchors) : null;
          trajectoryView.setTrajectory(spatial);
          appliedAnchor = -1;
          appliedRecenter = current.recenter;
        }
        trajectoryView.setVisibility(current.trajectoryDisplay.showPath, current.trajectoryDisplay.showFrustums);
        trajectoryView.setScale(current.trajectoryDisplay.frustumScale);
        trajectoryView.setSelected(spatial ? current.selectedAnchor : -1);

        if (current.exploring && !exploring && tube) beginExploring();
        else if (!current.exploring && exploring) endExploring();

        // Choosing an anchor moves *where* the camera is, never where it looks:
        // the raw poses at one anchor are six cube faces, and replaying their
        // orientations is exactly the spinning this mode exists to replace.
        const anchors = spatial?.anchors ?? [];
        if (anchors.length && current.selectedAnchor >= 0 && current.selectedAnchor !== appliedAnchor) {
          appliedAnchor = Math.min(current.selectedAnchor, anchors.length - 1);
          const point = new THREE.Vector3(...anchors[appliedAnchor].position);
          if (exploring) enterTube(point);
          else navigation.jumpToPoint(point);
        } else if (current.selectedAnchor < 0) {
          appliedAnchor = -1;
        }
        if (current.recenter !== appliedRecenter) {
          appliedRecenter = current.recenter;
          if (tube) {
            const anchor = anchors.length ? nearestAnchor(anchors, camera.position.toArray() as [number, number, number]) : -1;
            const point = anchor >= 0 ? new THREE.Vector3(...anchors[anchor].position) : camera.position.clone();
            if (exploring) { enterTube(point); freeLook.focusCanvas(); }
            else navigation.jumpToPoint(point);
            reportStatus(true);
          }
        }
      };

      /**
       * Tell React where the walker is, at a rate React can live with.
       *
       * The numbers change every frame; a setState every frame would re-render
       * the whole panel sixty times a second for a readout that nobody can read
       * that fast. Crossing the boundary is reported immediately because it is
       * a state change the user needs to see the instant it happens; the
       * distance follows at about six hertz.
       */
      const reportStatus = (force = false) => {
        const now = performance.now();
        if (!force && now - lastStatusAt < 250) return;
        const anchors = spatial?.anchors ?? [];
        const next = {
          distance: lastDistance,
          atBoundary: lastClamped,
          anchor: anchors.length ? nearestAnchor(anchors, camera.position.toArray() as [number, number, number]) : -1,
          pointerLocked: freeLook.pointerLocked,
        };
        // Every report is a React render of the whole panel, so send one only
        // when the reading actually changed by something the readout can show.
        if (!force && reported
          && reported.atBoundary === next.atBoundary
          && reported.anchor === next.anchor
          && reported.pointerLocked === next.pointerLocked
          && Math.abs(reported.distance - next.distance) < latest.current.radius / 200) return;
        lastStatusAt = now;
        reported = next;
        latest.current.onExploreStatus(next);
      };

      /**
       * One frame of walking. Returns true while it owns the camera.
       *
       * The tube is applied every frame, not only when the user moves: the
       * radius control can shrink the allowed region under a standing camera,
       * and the boundary has to pull it back in rather than wait for a keypress.
       */
      const walk = (now: number) => {
        if (!exploring || !tube) {
          lastFrameTime = now;
          return false;
        }
        const delta = (now - lastFrameTime) / 1000;
        lastFrameTime = now;
        const step = freeLook.update(delta, latest.current.speed);
        const result = constrainToTube(tube, camera.position.clone().add(step.delta), latest.current.radius, activeSegment);
        activeSegment = result.segment;
        navigation.driveFreeLook(result.position, step.quaternion);
        trajectoryView.setActiveSegment(activeSegment);
        // "At the boundary", not "was pushed back this frame". Once the walker
        // is resting against the surface no single frame is a clamp, and a
        // readout that flickered off the instant they stopped pressing would be
        // telling them they are free to go when they are not.
        const atBoundary = result.clamped || result.distance >= latest.current.radius * (1 - 1e-6);
        const crossed = atBoundary !== lastClamped;
        lastClamped = atBoundary;
        lastDistance = result.distance;
        if (crossed || step.moved || step.looked) reportStatus(crossed);
        return true;
      };

      /** While walking, a click is a request for the mouse, not a selection. */
      const pickAnchor = (event: PointerEvent) => {
        if (!spatial?.anchors.length) return;
        if (exploring) {
          freeLook.requestPointerLock();
          freeLook.focusCanvas();
          return;
        }
        const bounds = view.canvas.getBoundingClientRect();
        if (!bounds.width || !bounds.height) return;
        const hit = trajectoryView.pick(
          event.clientX - bounds.left, event.clientY - bounds.top, camera, bounds.width, bounds.height,
        );
        if (hit !== null) latest.current.onAnchorChange(hit);
      };
      // Click, not drag: a pointerup that barely moved from its pointerdown.
      let pressedAt: { x: number; y: number } | null = null;
      view.on("pointerdown", (event) => { pressedAt = { x: event.clientX, y: event.clientY }; });
      view.on("pointerup", (event) => {
        const start = pressedAt;
        pressedAt = null;
        if (start && Math.hypot(event.clientX - start.x, event.clientY - start.y) < 5) pickAnchor(event);
      });

      let previousWidth = 0, previousHeight = 0;
      view.onResize((width, height) => {
        if ((Math.abs(width - previousWidth) > 2 || Math.abs(height - previousHeight) > 2) && mesh)
          navigation.resize(width / height);
        previousWidth = width;
        previousHeight = height;
      });
      view.on("webglcontextlost", (event) => {
        event.preventDefault();
        publish({ state: "error", message: "The graphics context was lost. Reload the scene to reconnect the splat renderer." });
      });

      runtime.current = {
        update,
        reset: () => navigation.reset(),
        roll: (degrees) => navigation.roll(degrees),
        setUp: (axis) => navigation.setUp(axis),
        captureView: (home) => navigation.capture(home),
        preset: (name) => navigation.preset(name),
        restoreImported: () => navigation.restoreImported(),
      };
      update();

      // Three.js reschedules the next frame only after the callback returns, so
      // an exception inside it would stop rendering for good and leave the
      // viewer spinning. Report it instead of disappearing.
      view.setAnimationLoop(() => {
        try {
          // The walker owns the camera while the mode is on, so OrbitControls
          // must not also update it: its damping would drag the camera back to
          // its own target every frame.
          if (!walk(performance.now())) navigation.controls.update();
          drawAxisIndicator(axes.current, camera);
          view.renderer.render(scene, camera);
          if (activeSplat?.paged && mesh && !pagedFitted && sparkModule) {
            if (mesh.paged?.numSplats) {
              pagedFitted = true;
              frame(activeSplat.view
                ? new THREE.Sphere(new THREE.Vector3(...activeSplat.view.center), activeSplat.view.radius)
                : splatBoundingSphere(sparkModule, mesh));
              publish(readyStatus(activeSplat, mesh));
            } else if (performance.now() - pagedSince > FIRST_PAGE_TIMEOUT_MS) {
              pagedFitted = true;
              publish({
                state: "error",
                message: "No level-of-detail pages arrived from the hosted scene. Check the host's Range and CORS behaviour, then reload the scene.",
              });
            }
          }
        } catch (error) {
          view.setAnimationLoop(null);
          // Never leave navigation disabled because a frame threw.
          exploring = false;
          freeLook.disable();
          navigation.controls.enabled = true;
          publish({
            state: "error",
            message: `The splat renderer stopped: ${error instanceof Error ? error.message : String(error)} Reload the scene to restart it.`,
          });
        }
      });

      if (import.meta.env.DEV && import.meta.env.VITE_VIEWER_DIAGNOSTICS === "1") {
        window.__NEURAL3D_SPLAT__ = (): SplatSnapshot => {
          camera.updateMatrixWorld();
          return {
            fingerprint: activeSplat?.sha256 ?? "",
            identity: activeSplat?.identity ?? "",
            fileType: activeSplat?.fileType ?? "",
            origin: activeSplat?.origin ?? "",
            splats: splatCount,
            decodedSplats: mesh ? decodedSplatCount(mesh) : 0,
            pagedSplats: mesh?.paged?.numSplats ?? 0,
            lod: !!activeSplat?.lod,
            paged: !!activeSplat?.paged,
            activeSplats: spark?.activeSplats ?? 0,
            trajectoryCameras: trajectoryView.cameraCount,
            trajectoryAnchors: trajectoryView.anchorCount,
            trajectoryAnchor: spatial ? appliedAnchor : -1,
            exploring,
            pointerLocked: freeLook.pointerLocked,
            tubeRadius: latest.current.radius,
            tubeDistance: lastDistance,
            tubeAtBoundary: lastClamped,
            activeSegment,
            controlsEnabled: navigation.controls.enabled,
            fov: camera.fov,
            trajectory: trajectoryView.snapshot(camera, view.canvas.clientWidth, view.canvas.clientHeight),
            camera: camera.position.toArray(),
            target: navigation.controls.target.toArray(),
            forward: new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion).toArray(),
            cameraUp: camera.up.toArray(),
            manual: navigation.manual,
            background: (scene.background as THREE.Color | null)?.getHexString() ?? "",
            geometries: view.renderer.info.memory.geometries,
            textures: view.renderer.info.memory.textures,
            programs: view.renderer.info.programs?.length ?? 0,
            canvases: document.querySelectorAll("canvas").length,
          };
        };
      }

      return () => {
        disposed = true;
        generation += 1;
        view.setAnimationLoop(null);
        if (import.meta.env.DEV) delete window.__NEURAL3D_SPLAT__;
        runtime.current = null;
        exploring = false;
        freeLook.dispose();
        navigation.controls.enabled = true;
        trajectoryView.dispose();
        spatial = null;
        tube = null;
        detach(mesh);
        mesh = null;
        if (spark) {
          // Quiesce Spark before releasing its GPU resources and workers.
          //
          // SparkRenderer.onBeforeRender schedules its async update through a
          // 1 ms setTimeout whose promise nothing owns, and dispose() does not
          // clear that timer: a pending one fires against the disposed renderer
          // and surfaces as an unhandled rejection. updateTimeoutId is public,
          // so cancel it. Turning off auto-update and the LoD driving/fetching
          // first stops new sort and level-of-detail worker calls being queued,
          // which is what dispose() would otherwise reject. (All public fields
          // of @sparkjsdev/spark 2.2.0; a call already in flight still rejects,
          // which is handled by the cleanup guard below.)
          spark.autoUpdate = false;
          spark.enableLodFetching = false;
          spark.enableDriveLod = false;
          spark.enableLod = false;
          if (spark.updateTimeoutId !== -1) {
            clearTimeout(spark.updateTimeoutId);
            spark.updateTimeoutId = -1;
          }
          scene.remove(spark);
          spark.dispose();
          spark = null;
        }
        sparkModule = null;
        view.dispose();
      };
    }, [attempt]);

    const visibleStatus = props.splat ? status : props.loadStatus;

    return (
      <div className="viewport" data-testid="splat-viewport" data-state={visibleStatus.state}>
        <div ref={host} className="canvas-host" />
        {visibleStatus.state === "ready" ? (
          <>
            <div className="viewport-badge">
              <span className="live-dot" />
              {visibleStatus.count.toLocaleString()} splats · Ready
              {visibleStatus.note ? ` · ${visibleStatus.note}` : ""}
            </div>
            <svg ref={axes} className="axis-key" viewBox="0 0 84 84" aria-label="Source axes in the current camera orientation">
              {["X", "Y", "Z"].map((axis, i) => <g key={axis} fill={["#fb7185", "#86efac", "#93c5fd"][i]} stroke={["#fb7185", "#86efac", "#93c5fd"][i]}>
                <line x1="42" y1="42" x2="42" y2="42" strokeWidth="2" />
                <text x="42" y="42" stroke="none" textAnchor="middle">{axis}</text>
              </g>)}
            </svg>
            <div className="viewport-hint" role="status">
              {props.exploring
                ? "Click to capture the mouse · WASD to walk · Q/E down and up · Shift to hurry · Esc to release"
                : "Drag to orbit · Right-drag to pan · Scroll to zoom · Annotation is not available for Gaussian Splats yet"}
            </div>
          </>
        ) : (
          <div className="viewer-state" role={visibleStatus.state === "error" ? "alert" : "status"}>
            <span className={visibleStatus.state === "loading" ? "spinner" : "state-symbol"}>
              {visibleStatus.state === "loading" ? "" : "◇"}
            </span>
            <h3>
              {visibleStatus.state === "loading"
                ? "Preparing your splat scene"
                : visibleStatus.state === "empty"
                  ? "An empty scene"
                  : "Unable to display scene"}
            </h3>
            <p>{visibleStatus.message}</p>
            {visibleStatus.state !== "loading" && (
              <button onClick={() => { if (props.splat) setAttempt((n) => n + 1); else props.onReload(); }}>
                Reload scene
              </button>
            )}
          </div>
        )}
      </div>
    );
  },
);
