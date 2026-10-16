import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import * as THREE from "three";
import { drawAxisIndicator, ViewerRuntime, WEBGL_UNAVAILABLE } from "./viewerRuntime";
import { SelectionGesture } from "./SelectionGesture";
import { groupKey, groupColor } from "./annotationGroups";
import type { Highlight, InteractionMode, SelectionOperation, SelectionOperationMode, ViewPreferences, PointCloudData } from "./types";
import type { ViewerSnapshot } from "./viewerDiagnostics";
import type { Annotation, Position, ViewerHandle, ViewerStatus } from "./types";

interface Props {
  view: ViewPreferences | null;
  interactionMode: InteractionMode;
  operationMode: SelectionOperationMode;
  selectionBusy: boolean;
  highlights: Highlight[];
  onSelection: (operation: SelectionOperation) => void;
  onSelectionError: (message: string) => void;
  cloud: PointCloudData | null;
  loadStatus: ViewerStatus;
  onReload: () => void;
  selectedGroup: string | null;
  pointSize: number;
  background: string;
  picking: boolean;
  annotations: Annotation[];
  selectedId: string | null;
  draft: Position | null;
  onPick: (position: Position) => void;
  onSelect: (id: string) => void;
  onStatus: (status: ViewerStatus) => void;
}

interface Runtime extends ViewerHandle { update: () => void }

function markerTexture(color: string, text: string, selected: boolean): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 96;
  const context = canvas.getContext("2d")!;
  context.beginPath();
  context.arc(48, 48, 36, 0, Math.PI * 2);
  context.fillStyle = color;
  context.fill();
  context.lineWidth = selected ? 9 : 6;
  context.strokeStyle = selected ? "#fcd34d" : "#ffffff";
  context.stroke();
  context.fillStyle = "#ffffff";
  context.font = "bold 38px system-ui";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, 48, 50);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

export const PointCloudViewer = forwardRef<ViewerHandle, Props>(
  function PointCloudViewer(props, ref) {
    const axes = useRef<SVGSVGElement>(null);
    const [overlay, setOverlay] = useState<[number, number][] | null>(null);
    const host = useRef<HTMLDivElement>(null);
    const runtime = useRef<Runtime | null>(null);
    const latest = useRef(props);
    const [hint, setHint] = useState("");
    const [attempt, setAttempt] = useState(0);
    const [status, setStatus] = useState<ViewerStatus>({
      state: "loading",
      message: "Loading point cloud…",
    });

    useEffect(() => {
      latest.current = props;
      runtime.current?.update();
    });
    useImperativeHandle(
      ref,
      () => ({
        prepare: (cloud) => {
          if (!runtime.current) throw new Error("WebGL is unavailable.");
          runtime.current.prepare(cloud);
        },
        prepareHighlights: (highlights) => {
          if (!runtime.current) throw new Error("WebGL is unavailable.");
          runtime.current.prepareHighlights(highlights);
        },
        reset: () => runtime.current?.reset(),
        focus: (position) => runtime.current?.focus(position),
        focusRegion: (bounds) => runtime.current?.focusRegion(bounds),
        roll: (angle) => runtime.current?.roll(angle),
        setUp: (axis) => runtime.current?.setUp(axis),
        captureView: (home) => {
          if (!runtime.current) throw new Error("Viewer is not ready.");
          return runtime.current.captureView(home);
        },
        preset: (name) => runtime.current?.preset(name),
        restoreImported: () => runtime.current?.restoreImported(),
      }),
      [],
    );

    useEffect(() => {
      const container = host.current!;
      let disposed = false;
      const publish = (next: ViewerStatus) => {
        if (!disposed) {
          setStatus(next);
          latest.current.onStatus(next);
        }
      };
      publish({ state: "loading", message: "Loading point cloud…" });
      setHint("");
      let view: ViewerRuntime;
      try {
        view = new ViewerRuntime(container, {
          antialias: true,
          label: "Interactive 3D point cloud. Drag to orbit, right-drag to pan, scroll to zoom. Use Pick point to annotate.",
        });
      } catch {
        publish({ state: "error", message: WEBGL_UNAVAILABLE });
        return () => {
          disposed = true;
        };
      }
      const { renderer, canvas, scene, camera, navigation } = view;
      const gesture = new SelectionGesture(canvas, navigation, () => ({
        mode: latest.current.interactionMode, operation: latest.current.operationMode,
        busy: latest.current.selectionBusy, onSelection: latest.current.onSelection,
        onError: latest.current.onSelectionError,
      }), setOverlay);
      let previousMode = latest.current.interactionMode;
      let points: THREE.Points<
        THREE.BufferGeometry,
        THREE.PointsMaterial
      > | null = null;
      let activeCloud: PointCloudData | null = null;
      let prepared: { cloud: PointCloudData; points: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial> } | null = null;
      const discardPrepared = () => {
        prepared?.points.geometry.dispose();
        prepared?.points.material.dispose();
        prepared = null;
      };
      let down: { x: number; y: number; pointerId: number } | null = null;
      let markers: THREE.Sprite[] = [];
      let markerGeometry: THREE.BufferGeometry | null = null;
      let markerProps: unknown[] = [];
      let highlightProps: Highlight[] | null = null;
      type Layer = THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial>;
      let highlights: Layer[] = [];
      let preparedHighlights: { descriptors: Highlight[]; layers: Layer[] } | null = null;
      const releaseLayers = (layers: Layer[]) => {
        layers.forEach((layer) => {
          layer.removeFromParent();
          // Borrowed position: do not delete the original cloud's GPU buffer.
          layer.geometry.deleteAttribute("position");
          layer.geometry.dispose(); layer.material.dispose();
        });
      };
      const clearHighlights = () => {
        releaseLayers(highlights); highlights = []; highlightProps = null;
      };
      const discardHighlights = () => {
        if (preparedHighlights) releaseLayers(preparedHighlights.layers);
        preparedHighlights = null;
      };
      const sameIndices = (a: Highlight[] | null, b: Highlight[]) => !!a && a.length === b.length &&
        a.every((value, i) => value.indices === b[i].indices && value.key === b[i].key);
      const prepareHighlights = (descriptors: Highlight[]) => {
        if (sameIndices(highlightProps, descriptors) || sameIndices(preparedHighlights?.descriptors ?? null, descriptors)) return;
        discardHighlights();
        const layers: Layer[] = [];
        const staging = new THREE.Scene();
        try {
          if (!points) throw new Error("Point cloud is not ready.");
          for (const descriptor of descriptors) {
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute("position", points.geometry.getAttribute("position"));
            geometry.setIndex(new THREE.BufferAttribute(descriptor.indices, 1));
            geometry.boundingSphere = points.geometry.boundingSphere;
            const material = new THREE.PointsMaterial({ color: descriptor.color, size: latest.current.pointSize + 3,
              sizeAttenuation: false, depthTest: false, depthWrite: false });
            const layer = new THREE.Points(geometry, material);
            layer.renderOrder = 5; layer.frustumCulled = false;
            staging.add(layer); layers.push(layer);
          }
          if (layers.length) {
            const gl = renderer.getContext();
            if (gl.isContextLost()) throw new Error("WebGL context lost.");
            while (gl.getError() !== gl.NO_ERROR) { /* clear earlier unrelated errors */ }
            renderer.render(staging, camera);
            if (gl.isContextLost() || gl.getError() !== gl.NO_ERROR) throw new Error("Not enough GPU memory for this selection.");
          }
          layers.forEach((layer) => { layer.removeFromParent(); layer.frustumCulled = true; });
          preparedHighlights = { descriptors, layers };
        } catch (error) { releaseLayers(layers); throw error; }
        finally { if (!renderer.getContext().isContextLost()) renderer.render(scene, camera); }
      };
      let previousWidth = 0, previousHeight = 0;
      view.onResize((width, height) => {
        const materialResize = Math.abs(width - previousWidth) > 2 || Math.abs(height - previousHeight) > 2;
        if (materialResize) { gesture.cancel(); if (points) navigation.resize(width / height); }
        previousWidth = width;
        previousHeight = height;
      });

      const clearMarkers = () => {
        markers.forEach((marker) => {
          scene.remove(marker);
          marker.material.map?.dispose();
          marker.material.dispose();
        });
        markers = [];
      };
      const prepare = (cloud: PointCloudData) => {
        if (prepared?.cloud === cloud) return;
        discardPrepared();
        const geometry = new THREE.BufferGeometry();
        const material = new THREE.PointsMaterial({
          size: latest.current.pointSize, sizeAttenuation: false,
          vertexColors: !!cloud.colors, color: cloud.colors ? "#ffffff" : "#7dd3fc",
        });
        const replacement = new THREE.Points(geometry, material);
        try {
          geometry.setAttribute("position", new THREE.BufferAttribute(cloud.positions, 3));
          if (cloud.colors) geometry.setAttribute("color", new THREE.BufferAttribute(cloud.colors, 3));
          geometry.computeBoundingSphere();
          const sphere = geometry.boundingSphere!;
          if (!Number.isFinite(sphere.radius)) throw new Error("Unrepresentable bounds.");
          const staging = new THREE.Scene();
          staging.background = new THREE.Color(latest.current.background);
          staging.add(replacement);
          replacement.frustumCulled = false; // force buffer upload before the swap
          const probe = camera.clone();
          const scale = Math.max(sphere.radius, 0.1);
          probe.near = Math.max(scale / 1000, 0.001);
          probe.far = scale * 150;
          probe.position.copy(sphere.center).add(new THREE.Vector3(0, 0, scale * 4));
          probe.lookAt(sphere.center);
          probe.updateProjectionMatrix();
          const gl = renderer.getContext();
          if (gl.isContextLost()) throw new Error("WebGL context lost.");
          // Existing errors are unrelated to this candidate's allocation.
          while (gl.getError() !== gl.NO_ERROR) { /* drain pending errors */ }
          renderer.render(staging, probe);
          if (gl.isContextLost() || gl.getError() !== gl.NO_ERROR) throw new Error("WebGL allocation failed.");
          replacement.frustumCulled = true;
          staging.remove(replacement);
          prepared = { cloud, points: replacement };
        } catch (error) {
          geometry.dispose();
          material.dispose();
          throw error;
        } finally {
          // Restore the visible scene while React commits the validated swap.
          if (!renderer.getContext().isContextLost()) renderer.render(scene, camera);
        }
      };
      const update = () => {
        const current = latest.current;
        navigation.preferences = current.view;
        if (previousMode !== current.interactionMode) { gesture.cancel(); previousMode = current.interactionMode; }
        if (current.cloud && current.cloud !== activeCloud) {
          try { prepare(current.cloud); }
          catch {
            publish({ state: "error", message: "The browser could not initialize this PLY at full resolution. Reload the scene to retry." });
            return;
          }
          gesture.cancel();
          clearHighlights();
          discardHighlights();
          if (points) {
            scene.remove(points);
            points.geometry.dispose();
            points.material.dispose();
          }
          points = prepared!.points;
          const geometry = points.geometry;
          prepared = null;
          scene.add(points);
          activeCloud = current.cloud;
          down = null;
          setHint("");
          clearMarkers();
          markerProps = [];
          navigation.center.copy(geometry.boundingSphere!.center);
          navigation.radius = Math.max(geometry.boundingSphere!.radius, 0.1);
          navigation.reset();
          publish({ state: "ready", count: current.cloud.positions.length / 3 });
        }
        scene.background = new THREE.Color(current.background);
        if (points) points.material.size = current.pointSize;
        canvas.style.cursor = current.interactionMode === "navigate" ? "grab" : "crosshair";
        if (points && highlightProps !== current.highlights) {
          if (!sameIndices(highlightProps, current.highlights)) {
            try { prepareHighlights(current.highlights); }
            catch (reason) {
              current.onSelectionError(reason instanceof Error ? reason.message : "Selection display could not be initialized.");
              return;
            }
            clearHighlights();
            highlights = preparedHighlights!.layers; preparedHighlights = null;
            highlights.forEach((layer) => scene.add(layer));
          } else discardHighlights();
          highlightProps = current.highlights;
        }
        highlights.forEach((layer, i) => {
          layer.material.size = current.pointSize + 3;
          layer.material.color.set(current.highlights[i].color);
        });
        const signature = [current.annotations, current.selectedId, current.selectedGroup, current.draft];
        if (signature.every((value, i) => value === markerProps[i])) return;
        markerProps = signature;
        clearMarkers();
        const addMarker = (
          position: Position,
          color: string,
          text: string,
          id?: string,
          selected = false,
        ) => {
          const marker = new THREE.Sprite(
            new THREE.SpriteMaterial({
              map: markerTexture(color, text, selected),
              depthTest: false,
              depthWrite: false,
            }),
          );
          // Own the tiny shared marker quad so unmount explicitly releases it.
          markerGeometry ??= marker.geometry.clone();
          marker.geometry = markerGeometry;
          marker.position.set(position.x, position.y, position.z);
          marker.renderOrder = 10;
          marker.userData.annotationId = id;
          marker.userData.selected = selected;
          marker.userData.color = color;
          scene.add(marker);
          markers.push(marker);
        };
        current.annotations.forEach((annotation, index) => {
          addMarker(
            annotation,
            groupColor(groupKey(annotation.category, annotation.label)),
            String(index + 1),
            annotation.id,
            annotation.id === current.selectedId || groupKey(annotation.category, annotation.label) === current.selectedGroup,
          );
        });
        if (current.draft && !current.selectedId)
          addMarker(current.draft, "#26734d", "+", undefined, true);
      };
      runtime.current = { update, prepare, prepareHighlights,
        reset: () => navigation.reset(), focus: (p) => navigation.focus(p),
        focusRegion: (b) => navigation.focusRegion(b), roll: (d) => navigation.roll(d),
        setUp: (a) => navigation.setUp(a), captureView: (h) => navigation.capture(h),
        preset: (p) => navigation.preset(p), restoreImported: () => navigation.restoreImported(),
      };
      update();

      const pointerDown = (event: PointerEvent) => {
        if (event.isPrimary && event.button === 0)
          down = {
            x: event.clientX,
            y: event.clientY,
            pointerId: event.pointerId,
          };
        else down = null;
      };
      const cancelPointer = () => {
        down = null;
      };
      const raycaster = new THREE.Raycaster();
      const pointerUp = (event: PointerEvent) => {
        const start = down;
        down = null;
        if (
          !start ||
          start.pointerId !== event.pointerId ||
          Math.hypot(start.x - event.clientX, start.y - event.clientY) > 5 ||
          !points
        )
          return;
        const bounds = canvas.getBoundingClientRect();
        const mouse = new THREE.Vector2(
          ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
          (-(event.clientY - bounds.top) / bounds.height) * 2 + 1,
        );
        raycaster.setFromCamera(mouse, camera);
        if (!latest.current.picking) {
          const hit = raycaster
            .intersectObjects(markers)
            .find((item) => item.object.userData.annotationId);
          if (hit)
            latest.current.onSelect(hit.object.userData.annotationId as string);
          return;
        }
        // Project original unique vertices to pixels: selection remains
        // equally forgiving at any zoom, and stores a vertex rather than a ray point.
        const positions = points.geometry.getAttribute("position");
        const candidate = new THREE.Vector3();
        let selected: THREE.Vector3 | null = null;
        let bestDepth = Infinity;
        for (let i = 0; i < positions.count; i++) {
          candidate.fromBufferAttribute(positions, i).project(camera);
          if (candidate.z < -1 || candidate.z > 1) continue;
          const distance = Math.hypot(
            ((candidate.x - mouse.x) * bounds.width) / 2,
            ((candidate.y - mouse.y) * bounds.height) / 2,
          );
          if (
            distance <= Math.max(10, latest.current.pointSize / 2 + 4) &&
            candidate.z < bestDepth
          ) {
            bestDepth = candidate.z;
            selected = new THREE.Vector3().fromBufferAttribute(positions, i);
          }
        }
        if (selected) {
          setHint("Point selected. Add a label in the annotation panel.");
          latest.current.onPick({
            x: selected.x,
            y: selected.y,
            z: selected.z,
          });
        } else
          setHint(
            "No point nearby. Click a colored point, or zoom in and try again.",
          );
      };
      const contextLost = (event: Event) => {
        event.preventDefault();
        publish({
          state: "error",
          message:
            "The graphics context was lost. Reload the scene to reconnect the viewer.",
        });
      };
      view.on("pointerdown", pointerDown);
      view.on("pointerup", pointerUp);
      view.on("pointercancel", cancelPointer);
      view.on("webglcontextlost", contextLost);
      view.setAnimationLoop(() => {
        navigation.controls.update();
        drawAxisIndicator(axes.current, camera);
        markers.forEach((marker) => {
          const size =
            camera.position.distanceTo(marker.position) *
            (marker.userData.selected ? 0.043 : 0.034);
          marker.scale.setScalar(size);
        });
        renderer.render(scene, camera);
      });

      if (import.meta.env.DEV && import.meta.env.VITE_VIEWER_DIAGNOSTICS === "1") {
        window.__NEURAL3D__ = (): ViewerSnapshot => {
          const positions = points?.geometry.getAttribute("position");
          camera.updateMatrixWorld();
          return {
            camera: camera.position.toArray(), target: navigation.controls.target.toArray(),
            fingerprint: activeCloud?.sha256 ?? "", manual: navigation.manual,
            cameraUp: camera.up.toArray(), controlsEnabled: navigation.controls.enabled,
            highlights: (highlightProps ?? []).map((h) => ({ key: h.key, count: h.indices.length, indices: Array.from(h.indices.slice(0, 64)), color: h.color })),
            colors: Array.from(activeCloud?.colors?.slice(0, 48) ?? []),
            displayedPoints: positions?.count ?? 0, indexCount: points?.geometry.index?.count ?? 0,
            sourceIndexCount: activeCloud?.sourceIndexCount ?? 0, faceCount: activeCloud?.face_count ?? 0,
            geometries: renderer.info.memory.geometries, textures: renderer.info.memory.textures,
            programs: renderer.info.programs?.length ?? 0,
            vertexColors: points?.material.vertexColors ?? false,
            fallbackColor: points?.material.color.getHexString() ?? "",
            markers: markers.map((marker) => ({ id: marker.userData.annotationId, selected: marker.userData.selected, color: marker.userData.color, position: marker.position.toArray() })),
            projected: Array.from({ length: Math.min(positions?.count ?? 0, 16) }, (_, i) => {
              const point = new THREE.Vector3().fromBufferAttribute(positions!, i).project(camera);
              return { x: (point.x + 1) / 2, y: (1 - point.y) / 2 };
            }),
          };
        };
      }

      return () => {
        disposed = true;
        if (import.meta.env.DEV) delete window.__NEURAL3D__;
        runtime.current = null;
        gesture.dispose();
        clearHighlights();
        discardHighlights();
        clearMarkers();
        markerGeometry?.dispose();
        discardPrepared();
        points?.geometry.dispose();
        points?.material.dispose();
        view.dispose();
      };
    }, [attempt]);

    const visibleStatus = props.cloud ? status : props.loadStatus;

    return (
      <div
        className="viewport"
        data-testid="viewport"
        data-state={visibleStatus.state}
      >
        <div ref={host} className="canvas-host" />
        {visibleStatus.state === "ready" ? (
          <>
            <div className="viewport-badge">
              <span className="live-dot" />
              {visibleStatus.count.toLocaleString()} points · Ready
            </div>
            <svg ref={axes} className="axis-key" viewBox="0 0 84 84" aria-label="Source axes in the current camera orientation">
              {["X", "Y", "Z"].map((axis, i) => <g key={axis} fill={["#fb7185", "#86efac", "#93c5fd"][i]} stroke={["#fb7185", "#86efac", "#93c5fd"][i]}>
                <line x1="42" y1="42" x2="42" y2="42" strokeWidth="2" />
                <text x="42" y="42" stroke="none" textAnchor="middle">{axis}</text>
              </g>)}
            </svg>
            {overlay && <svg className="selection-overlay" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Selection outline">
              <polygon points={overlay.map((p) => p.join(",")).join(" ")} vectorEffect="non-scaling-stroke" />
            </svg>}
            <div className="viewport-hint" role="status">
              {props.interactionMode === "rectangle" || props.interactionMode === "lasso"
                ? "Drag to select through · Shift adds · Alt/Option subtracts · Esc cancels drag"
                : props.picking
                ? hint.startsWith("No point")
                  ? hint
                  : "Click a colored point to place a marker · Drag still orbits"
                : "Drag to orbit · Right-drag to pan · Scroll to zoom"}
            </div>
          </>
        ) : (
          <div
            className="viewer-state"
            role={visibleStatus.state === "error" ? "alert" : "status"}
          >
            <span
              className={
                visibleStatus.state === "loading" ? "spinner" : "state-symbol"
              }
            >
              {visibleStatus.state === "loading" ? "" : "◇"}
            </span>
            <h3>
              {visibleStatus.state === "loading"
                ? "Preparing your scene"
                : visibleStatus.state === "empty"
                  ? "An empty scene"
                  : "Unable to display scene"}
            </h3>
            <p>{visibleStatus.message}</p>
            {visibleStatus.state !== "loading" && (
              <button onClick={() => { if (props.cloud) setAttempt((n) => n + 1); else props.onReload(); }}>
                Reload scene
              </button>
            )}
          </div>
        )}
      </div>
    );
  },
);
