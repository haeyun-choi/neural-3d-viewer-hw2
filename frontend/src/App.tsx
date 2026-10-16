import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { OrientationControls } from "./OrientationControls";
import { readView, saveView, clearView } from "./viewPreferences";
import { useRegionSelection } from "./useRegionSelection";
import { groupKey, groupColor } from "./annotationGroups";
import type { InteractionMode, SelectionOperationMode } from "./types";
import { AnnotationPanel } from "./AnnotationPanel";
import { PLY_IMPORT_TEXT } from "./ply";
import { GAUSSIAN_ACCEPT, GAUSSIAN_IMPORT_TEXT } from "./gaussianFormat";
import { classifyDroppedFile } from "./gaussianSplat";
import { HostedSceneError, sceneFromUrl } from "./sceneManifest";
import { formatBytes, useScene } from "./useScene";
import { useTrajectory } from "./useTrajectory";
import { TRAJECTORY_ACCEPT } from "./cameraTrajectory";
import { PointCloudViewer } from "./PointCloudViewer";
import { GaussianSplatViewer } from "./GaussianSplatViewer";
import { ClassificationPanel } from "./ClassificationPanel";
import { connectStorage, CorruptStorageError, errorMessage, recoverLocalStorage } from "./storage";
import type { AnnotationStore } from "./storage";
import type {
  Annotation,
  AnnotationInput,
  CameraHandle,
  Position,
  PointCloudData,
  RepresentationType,
  ViewerHandle,
  ViewerStatus,
} from "./types";

// Import phases in the order Spark goes through them, so each row can say
// whether it is done, running or still to come. Streaming and decoding are
// genuinely separate measurements — bytes pulled out of the file versus bytes
// handed to Spark's WebAssembly decoder — but Spark decodes each chunk as it
// arrives, so in practice they track each other closely; the level-of-detail
// build is the phase that runs on its own after the last byte.
const PHASE_ORDER: Record<string, number> = { inspecting: 0, streaming: 1, paging: 1, decoding: 2, lod: 3 };
const phaseState = (row: string, current: string) =>
  (PHASE_ORDER[current] ?? 0) > PHASE_ORDER[row] ? "done" : (PHASE_ORDER[current] ?? 0) === PHASE_ORDER[row] ? "active" : "pending";

const GAUSSIAN_ANNOTATION_NOTICE =
  "Annotation is not available for Gaussian Splats yet. Splat picking and region selection need a different, splat-accurate hit test; the point-cloud annotation workflow is unchanged. Open a PLY point cloud to annotate.";

/**
 * Slider bounds a person can read, for a range whose size comes from the scene.
 *
 * The radius and speed limits are derived from the capture, so they are numbers
 * like 0.015000000000000251. A step of one two-hundredth of that is unreadable
 * in the tooltip and awkward from the keyboard, so the step is rounded to 1, 2
 * or 5 times a power of ten and the ends are pulled *inward* onto that grid —
 * inward, so a slider can never ask for a value the model would clamp.
 */
function sliderRange(min: number, max: number) {
  const span = max - min;
  if (!(span > 0)) return { min, max, step: "any" as const };
  const rough = span / 150;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const scaled = rough / magnitude;
  const step = (scaled >= 5 ? 5 : scaled >= 2 ? 2 : 1) * magnitude;
  return { min: Math.ceil(min / step) * step, max: Math.floor(max / step) * step, step };
}

export default function App() {
  const [renderStatus, setRenderStatus] = useState<ViewerStatus>({ state: "loading", message: "Preparing viewer…" });
  const [selectedGroup, setSelectedGroup] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [hostedId, setHostedId] = useState("");
  const [hostedUrl, setHostedUrl] = useState("");
  const [hostedError, setHostedError] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const splatInput = useRef<HTMLInputElement>(null);
  const pathInput = useRef<HTMLInputElement>(null);
  const [pointSize, setPointSize] = useState(3);
  const [background, setBackground] = useState("#101827");
  const [store, setStore] = useState<AnnotationStore | null>(null);
  const [records, setRecords] = useState<Annotation[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [position, setPosition] = useState<Position | null>(null);
  const [interactionMode, setInteractionMode] = useState<InteractionMode>("navigate");
  const picking = interactionMode === "point";
  const setPicking = (value: boolean) => setInteractionMode(value ? "point" : "navigate");
  const [operationMode, setOperationMode] = useState<SelectionOperationMode>("replace");
  const clearSelection = useRef<() => void>(() => {});
  const [viewRevision, setViewRevision] = useState(0);
  const [viewMessage, setViewMessage] = useState({ scene: "", error: "", notice: "" });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [corruption, setCorruption] = useState<CorruptStorageError | null>(null);
  const viewer = useRef<ViewerHandle>(null);
  const splatViewer = useRef<CameraHandle>(null);
  const viewerArea = useRef<HTMLElement>(null);
  const storageGeneration = useRef(0);
  const selected = records.find((record) => record.id === selectedId) ?? null;

  const activate = useCallback(() => {
    // Invalidate old promises immediately, before React commits the new scene
    // and runs the previous storage effect's cleanup.
    storageGeneration.current += 1;
    clearSelection.current();
    setStore(null);
    setRecords([]);
    setSelectedId(null);
    setSelectedGroup(null);
    setPosition(null);
    setInteractionMode("navigate");
    setError("");
    setNotice("");
    setCorruption(null);
  }, []);
  const representationRef = useRef<RepresentationType>("point-cloud");
  const prepare = useCallback((cloud: PointCloudData) => {
    // While a Gaussian scene is displayed the point renderer is unmounted, so
    // there is nothing to stage into: it allocates its buffers when it mounts
    // and reports its own failure there. When it is mounted, a failed
    // allocation must abort the import before the visible scene changes.
    if (representationRef.current === "point-cloud" && !viewer.current)
      throw new Error("Viewer is not ready.");
    viewer.current?.prepare(cloud);
  }, []);
  const {
    loaded, loading, importError, loadStatus, warning, gaussianProgress, hostedScenes,
    continueImport, cancelImport, openFile, openHosted, restoreSample,
  } = useScene(activate, prepare);
  const scene = loaded?.metadata;
  const status = loaded ? renderStatus : loadStatus;
  const sceneId = scene?.id ?? "";
  // Exactly one representation is active, so exactly one renderer is mounted.
  const representation = loaded?.representation ?? "point-cloud";
  const isGaussian = representation === "gaussian-splat";
  const annotationsSupported = !isGaussian;
  // Camera control is shared; the active viewer supplies it.
  const camera = useCallback((): CameraHandle | null => (isGaussian ? splatViewer.current : viewer.current), [isGaussian]);
  useEffect(() => { representationRef.current = representation; }, [representation]);
  useEffect(() => {
    if (hostedScenes.length && !hostedScenes.some((scene) => scene.id === hostedId)) setHostedId(hostedScenes[0].id);
  }, [hostedScenes, hostedId]);
  const openHostedScene = (id: string) => {
    const scene = hostedScenes.find((entry) => entry.id === id);
    if (!scene || busy || loading) return;
    setHostedError("");
    void openHosted(scene);
  };
  const openHostedByUrl = () => {
    if (busy || loading) return;
    try {
      setHostedError("");
      void openHosted(sceneFromUrl(hostedUrl));
    } catch (reason) {
      setHostedError(reason instanceof HostedSceneError ? reason.message : errorMessage(reason));
    }
  };
  const storedView = useMemo(() => { void viewRevision; return readView(sceneId); }, [sceneId, viewRevision]);
  // ModelNet10 is Z-up. A classification sample is turned upright once it is
  // displayed, unless this browser already saved a view for that object.
  const zUpFile = useRef<string | null>(null);
  useEffect(() => {
    if (!zUpFile.current || scene?.filename !== zUpFile.current || status.state !== "ready") return;
    zUpFile.current = null;
    if (!storedView.view) camera()?.setUp("Z");
  }, [scene, status, storedView.view, camera]);
  const activeSplat = loaded?.representation === "gaussian-splat" ? loaded.splat : null;
  const splatDelivery = !activeSplat
    ? "—"
    : activeSplat.paged
      ? "Hosted RAD, paged over HTTP by Spark"
      : `Streamed from your device into Spark's decoder${activeSplat.lod ? ", level-of-detail tree built on load" : ""}`;
  // The capture path belongs to the Gaussian scene on screen; a new scene
  // clears it. Multi-scene browsing is a later milestone.
  const path = useTrajectory(activeSplat ? `splat:${activeSplat.sha256}` : `other:${sceneId}`, activeSplat?.trajectoryUrl ?? null);
  const pathAnchor = path.selected >= 0 ? path.spatial?.anchors[path.selected] ?? null : null;
  const splatIdentity = !activeSplat
    ? "—"
    : activeSplat.identity === "file-sha256"
      ? "SHA-256 of every byte of the file"
      : activeSplat.identity === "bounded-sample"
        ? "Bounded fingerprint: name, size, modified time, PLY header and three 1 MiB payload windows"
        : "SHA-256 of the hosted asset URL";
  const selection = useRegionSelection(loaded?.representation === "point-cloud" ? loaded.cloud : null, setError, (layers) => {
    if (!viewer.current) throw new Error("Viewer is not ready.");
    viewer.current.prepareHighlights(layers);
  });
  const selectedCount = selection.region?.selected_count ?? (position ? 1 : 0);
  useEffect(() => { clearSelection.current = selection.clear; }, [selection.clear]);
  const highlights = useMemo(() => selection.layers.map((layer) => {
    const annotation = records.find((a) => a.id === (layer.key === "draft" ? selectedId : layer.key));
    return annotation ? { ...layer, color: groupColor(groupKey(annotation.category, annotation.label)) } : layer;
  }), [selection.layers, records, selectedId]);
  function saveOrientation(home: boolean) {
    const active = camera();
    if (!scene || !active) return;
    try {
      saveView(scene.id, active.captureView(home));
      setViewRevision((n) => n + 1);
      setViewMessage({ scene: scene.id, error: "", notice: home ? "Front/home saved in this browser." : "Upright saved. Previous home cleared; Set as front to save exact framing." });
    } catch (reason) { setViewMessage({ scene: scene.id, error: errorMessage(reason), notice: "" }); }
  }
  function restoreOrientation() {
    if (!scene) return;
    try {
      clearView(scene.id); camera()?.restoreImported(); setViewRevision((n) => n + 1);
      setViewMessage({ scene: scene.id, error: "", notice: "Imported orientation restored; saved front/home cleared." });
    } catch (reason) { setViewMessage({ scene: scene.id, error: errorMessage(reason), notice: "" }); }
  }

  useEffect(() => {
    // Gaussian scenes do not register with the annotation backend at all: no
    // splat metadata, identity or geometry reaches the API in this milestone.
    if (!scene || scene.representation !== "point-cloud") return;
    let cancelled = false;
    const generation = storageGeneration.current;
    void connectStorage(scene)
      .then(({ store: connected, records: saved }) => {
        if (!cancelled && generation === storageGeneration.current) {
          setStore(connected);
          setRecords(saved);
        }
      })
      .catch((reason: unknown) => {
        if (!cancelled && generation === storageGeneration.current) {
          setError(errorMessage(reason));
          setCorruption(reason instanceof CorruptStorageError ? reason : null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [scene]);

  const cancel = useCallback(() => {
    clearSelection.current();
    setSelectedId(null);
    setSelectedGroup(null);
    setPosition(null);
    setInteractionMode("navigate");
    setError("");
    setNotice("");
  }, []);
  const select = (annotation: Annotation) => {
    const ready = () => {
      setSelectedId(annotation.id); setSelectedGroup(null);
      setPosition({ x: annotation.x, y: annotation.y, z: annotation.z });
      setInteractionMode("navigate"); setError(""); setNotice("");
    };
    if (annotation.kind === "region") {
      void selection.show([annotation], true, (result) => {
        ready(); if (result.result?.bounds) viewer.current?.focusRegion(result.result.bounds);
      });
    } else { selection.clear(); ready(); viewer.current?.focus(annotation); }
  };
  const pick = useCallback((point: Position) => {
    clearSelection.current();
    setSelectedId(null);
    setSelectedGroup(null);
    setPosition(point);
    setInteractionMode("navigate");
    setError("");
    setNotice("");
  }, []);

  async function save(input: AnnotationInput) {
    if (!store || busy || loading || selection.busy) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const record = selectedId
        ? await store.update(selectedId, input)
        : await store.create(input);
      setRecords((items) =>
        selectedId
          ? items.map((item) => (item.id === record.id ? record : item))
          : [...items, record],
      );
      if (input.kind === "region") setSelectedId(record.id);
      else { setSelectedId(null); setPosition(null); }
      setInteractionMode("navigate");
      setNotice(
        store.mode === "database"
          ? "Annotation saved to the database."
          : "Annotation saved in this browser only.",
      );
    } catch (reason) {
      setError(errorMessage(reason));
      if (reason instanceof CorruptStorageError) setCorruption(reason);
    } finally {
      setBusy(false);
    }
  }

  async function remove(id: string) {
    if (!store || busy || loading || selection.busy) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await store.remove(id);
      setRecords((items) => items.filter((item) => item.id !== id));
      cancel();
      setNotice("Annotation deleted.");
    } catch (reason) {
      setError(errorMessage(reason));
      if (reason instanceof CorruptStorageError) setCorruption(reason);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <header className="app-header">
        <a
          className="brand"
          href={import.meta.env.BASE_URL}
          aria-label="Neural 3D home"
        >
          <span className="brand-symbol">
            N<span>3</span>
          </span>
          <span>
            Neural <b>3D</b>
            <small>DATA VIEWER</small>
          </span>
        </a>
        <div className="header-right">
          <span className="release-tag">{isGaussian ? "GAUSSIAN · SPARK" : "POINT CLOUD · PLY"}</span>
          <span className={`storage-badge ${isGaussian ? "paused" : store?.mode ?? "connecting"}`}>
            <span className="live-dot" />
            {isGaussian
              ? "Annotations paused"
              : store?.mode === "database"
                ? "Database connected"
                : store?.mode === "local"
                  ? "Local demo mode"
                  : error
                    ? "Storage unavailable"
                    : "Connecting storage…"}
          </span>
        </div>
      </header>
      <main>
        <section className="intro">
          <div>
            <p className="eyebrow">EXPLORE / OBSERVE / ANNOTATE</p>
            <h1>A new perspective on your data.</h1>
            <p>{isGaussian ? "Navigate a Gaussian Splat scene rendered by Spark." : "Navigate a point cloud and turn details into spatial notes."}</p>
          </div>
          <span className="scene-index">
            {isGaussian
              ? (scene?.source === "hosted" ? <>HOSTED <b>SPLAT</b></> : <>LOCAL <b>SPLAT</b></>)
              : scene?.source === "local" ? <>LOCAL <b>PLY</b></> : <>SAMPLE <b>01</b> / 01</>}
          </span>
        </section>
        {!isGaussian && store?.mode === "local" && (
          <div className="mode-banner" role="status">
            <span>◉</span>
            <div>
              <strong>Static demo · saved in this browser only</strong>
              <p>
                The backend is unavailable or local mode is configured. Notes
                are separate from the database, are not synced, and disappear if
                browser data is cleared. Start the backend and reload to
                connect.
              </p>
            </div>
          </div>
        )}
        <div className="workspace">
          <section
            ref={viewerArea}
            className={`viewer-column ${dragging ? "drop-active" : ""}`}
            onDragOver={(event) => { event.preventDefault(); if (!busy) setDragging(true); }}
            onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false); }}
            onDrop={(event) => {
              event.preventDefault(); setDragging(false);
              if (busy) return;
              const file = event.dataTransfer.files[0];
              // A dropped .ply may be either representation, so the content decides.
              if (file) void classifyDroppedFile(file).then((kind) => openFile(file, kind));
            }}
            aria-label="Scene workspace"
          >
            <div className="scene-heading">
              <div>
                <span className="eyebrow">{isGaussian
                  ? (scene?.source === "hosted" ? "HOSTED GAUSSIAN SPLAT" : "LOCAL GAUSSIAN SPLAT")
                  : scene?.source === "local" ? "LOCAL POINT CLOUD" : "SYNTHETIC COLLECTION"}</span>
                <h2>{scene?.name ?? "Loading scene…"}</h2>
              </div>
              <span className="public-tag">{scene?.source === "hosted"
                ? "Hosted asset - paged, never uploaded"
                : scene?.source === "local" ? "Local file - not uploaded" : "Public-safe sample"}</span>
            </div>
            <div className="import-bar">
              <input ref={fileInput} type="file" accept=".ply" aria-label="Open PLY file" hidden
                onChange={(event) => {
                  const file = event.currentTarget.files?.[0];
                  event.currentTarget.value = "";
                  if (file && !busy) void openFile(file, "point-cloud");
                }} />
              <input ref={splatInput} type="file" accept={GAUSSIAN_ACCEPT} aria-label="Open Gaussian Splat file" hidden
                onChange={(event) => {
                  const file = event.currentTarget.files?.[0];
                  event.currentTarget.value = "";
                  if (file && !busy) void openFile(file, "gaussian-splat");
                }} />
              <button className="primary" onClick={() => fileInput.current?.click()} disabled={busy}>Open PLY</button>
              <button onClick={() => splatInput.current?.click()} disabled={busy}>Open Gaussian</button>
              <button onClick={() => void restoreSample()} disabled={busy || (!loading && scene?.source === "bundled")}>Use sample</button>
              {isGaussian && <>
                <input ref={pathInput} type="file" accept={TRAJECTORY_ACCEPT} aria-label="Open camera path file" hidden
                  onChange={(event) => {
                    const file = event.currentTarget.files?.[0];
                    event.currentTarget.value = "";
                    if (file && !busy) void path.openFile(file);
                  }} />
                <button onClick={() => pathInput.current?.click()} disabled={busy || path.busy}>Open camera path</button>
              </>}
              <span>or drop a file here. Point cloud: {PLY_IMPORT_TEXT} {GAUSSIAN_IMPORT_TEXT}</span>
            </div>
            <details className="hosted-scenes" data-testid="hosted-scenes">
              <summary>Hosted large scene · prebuilt Spark RAD</summary>
              <div className="hosted-body">
                {hostedScenes.length > 0 && (
                  <div className="hosted-row">
                    <label htmlFor="hosted-scene">Published scene</label>
                    <select id="hosted-scene" value={hostedId} disabled={busy || loading}
                      onChange={(event) => setHostedId(event.target.value)}>
                      {hostedScenes.map((scene) => <option key={scene.id} value={scene.id}>{scene.name}</option>)}
                    </select>
                    <button onClick={() => openHostedScene(hostedId)} disabled={busy || loading}>Open hosted scene</button>
                  </div>
                )}
                <div className="hosted-row">
                  <label htmlFor="hosted-url">RAD asset URL</label>
                  <input id="hosted-url" type="url" value={hostedUrl} placeholder="https://host/scene-lod.rad"
                    disabled={busy || loading} onChange={(event) => setHostedUrl(event.target.value)} />
                  <button onClick={openHostedByUrl} disabled={busy || loading}>Open RAD URL</button>
                </div>
                {hostedError && <p className="alert error" role="alert">{hostedError}</p>}
                <p>
                  A prebuilt Spark <code>.rad</code> level-of-detail asset is paged over HTTP as the camera
                  moves: nothing is downloaded in full, so a multi-gigabyte scene never enters this repository
                  or your browser&rsquo;s memory. A single <code>.rad</code> needs HTTP Range support on the host;
                  a chunked <code>.rad</code>&nbsp;+&nbsp;<code>.radc</code> layout does not. See
                  {" "}<code>docs/hosted-scenes.md</code>.
                </p>
              </div>
            </details>
            {warning && <section className="import-warning alert" role="alertdialog" aria-labelledby="import-warning-title" aria-describedby="import-warning-description">
              <h3 id="import-warning-title">{warning.tier === "very-large" ? "Very large PLY" : "Large PLY"} · {warning.filename}</h3>
              <p>{(warning.summary.sizeBytes / 1024 ** 2).toLocaleString(undefined, { maximumFractionDigits: 1 })} MiB · {warning.summary.vertexCount.toLocaleString()} unique vertices</p>
              <p id="import-warning-description">Parsing may use substantially more memory than the file size. Performance depends on your browser, RAM, and GPU.
                {warning.tier === "very-large" && " The tab may become unresponsive or be terminated."}
                {" No downsampling is applied. Your current scene stays visible while loading."}</p>
              <div className="import-actions">
                <button className="primary" onClick={continueImport}>Load full resolution</button>
                <button autoFocus onClick={cancelImport}>Cancel</button>
              </div>
            </section>}
            {loading && !warning && <div className="alert import-progress" role="status">
              <span>{loadStatus.state === "loading" ? loadStatus.message : "Loading PLY…"} {loaded ? "Your current scene stays visible." : ""}</span>
              {gaussianProgress && <ul className="import-phases" data-testid="gaussian-progress" data-phase={gaussianProgress.phase}>
                <li data-phase="streaming" data-state={phaseState("streaming", gaussianProgress.phase)}>
                  <span>{gaussianProgress.totalBytes ? "Streaming" : "Opening"}</span>
                  <progress value={gaussianProgress.streamedBytes} max={Math.max(gaussianProgress.totalBytes, 1)} />
                  <output>{gaussianProgress.totalBytes
                    ? `${formatBytes(gaussianProgress.streamedBytes)} of ${formatBytes(gaussianProgress.totalBytes)}`
                    : "paged over HTTP — nothing downloaded in full"}</output>
                </li>
                <li data-phase="decoding" data-state={phaseState("decoding", gaussianProgress.phase)}>
                  <span>Decoding</span>
                  <progress value={gaussianProgress.decodedBytes} max={Math.max(gaussianProgress.totalBytes, 1)} />
                  <output>{gaussianProgress.totalBytes
                    ? `${formatBytes(gaussianProgress.decodedBytes)} of ${formatBytes(gaussianProgress.totalBytes)}`
                    : "decoded by Spark as pages arrive"}</output>
                </li>
                <li data-phase="lod" data-state={gaussianProgress.lod ? phaseState("lod", gaussianProgress.phase) : "skipped"}>
                  <span>Level of detail</span>
                  <output>{gaussianProgress.lod
                    ? (gaussianProgress.phase === "lod" ? "building the LoD tree…" : gaussianProgress.phase === "paging" ? "prebuilt in the RAD asset" : "after the last byte")
                    : "not needed for this scene"}</output>
                </li>
              </ul>}
              <button onClick={cancelImport}>Cancel import</button>
            </div>}
            {importError && loaded && <p className="alert error" role="alert">{importError} Your previous scene is unchanged.</p>}
            {!!scene?.face_count && <p className="alert mesh-notice" role="status">Mesh PLY detected. Displaying {scene.point_count.toLocaleString()} unique vertices as a point cloud; {scene.face_count.toLocaleString()} faces are ignored.</p>}
            <div className="viewer-panel panel">
              <div className="toolbar">
                <div className="view-label" data-testid="representation-badge">
                  <span>◇</span> {isGaussian ? "Gaussian Splat" : "Point cloud"}{" "}
                  <span className="tag">{isGaussian ? "SPARK" : "PLY"}</span>
                </div>
                <button
                  onClick={() => camera()?.reset()}
                  disabled={status.state !== "ready" || selection.busy}
                  title={isGaussian ? "Return to saved home, or fit the splat scene" : "Return to saved home, or fit the point cloud"}
                >
                  ↺ Reset camera
                </button>
              </div>
              {isGaussian ? (
                <p className="alert splat-notice" role="status" data-testid="gaussian-annotation-notice">
                  {GAUSSIAN_ANNOTATION_NOTICE}
                </p>
              ) : (
              <div className="selection-controls">
                <div className="mode-buttons" role="group" aria-label="Navigation and annotation mode">
                  {(["navigate", "point", "rectangle", "lasso"] as const).map((mode) => <button key={mode}
                    aria-pressed={interactionMode === mode} disabled={busy || loading || selection.busy || status.state !== "ready"}
                    onClick={() => {
                      if (mode === "navigate" || mode === "point") cancel();
                      setInteractionMode(mode);
                    }}>{mode === "navigate" ? "Navigate" : mode[0].toUpperCase() + mode.slice(1)}</button>)}
                </div>
                {(interactionMode === "rectangle" || interactionMode === "lasso") && <div className="selection-options">
                  <label>Selection operation <select value={operationMode} onChange={(e) => setOperationMode(e.target.value as SelectionOperationMode)} disabled={selection.busy}>
                    <option value="replace">Replace selection</option><option value="add">Add selection</option><option value="subtract">Subtract selection</option>
                  </select></label>
                  <p>Select through depth · Shift adds · Alt/Option subtracts · Esc cancels drag</p>
                </div>}
                <span className="selected-count" role="status">{selectedGroup ? "Group highlighted" : `${selectedCount.toLocaleString()} ${selectedCount === 1 ? "point" : "points"} selected`}</span>
                {selection.busy && <div className="selection-progress" role="status">
                  <label>Processing selection <progress value={selection.progress} max="1" /></label>
                  <button onClick={selection.abort}>Cancel selection</button>
                </div>}
              </div>
              )}
              <OrientationControls camera={camera} view={storedView.view} disabled={busy || loading || selection.busy || status.state !== "ready"}
                error={(viewMessage.scene === sceneId ? viewMessage.error : "") || storedView.error}
                notice={(viewMessage.scene === sceneId ? viewMessage.notice : "") || storedView.notice || ""} onSave={saveOrientation} onRestore={restoreOrientation} />
              {isGaussian && (path.trajectory || path.error || path.busy) && (
                <details className="camera-path" data-testid="camera-path-panel" open>
                  <summary>
                    Trajectory exploration
                    <span data-testid="camera-path-summary">{path.trajectory
                      ? `${path.summary} · ${path.trajectory.filename}`
                      : path.busy ? "Loading…" : "Not loaded"}</span>
                  </summary>
                  <div className="camera-path-body">
                    {path.error && <p className="alert error" role="alert" data-testid="camera-path-error">{path.error}</p>}
                    {path.trajectory?.warnings.map((warning) => (
                      <p className="alert" role="status" key={warning}>{warning}</p>
                    ))}
                    {path.trajectory && path.spatial && <>
                      <div className="camera-path-row" role="group" aria-label="Trajectory exploration">
                        <button className="primary" data-testid="explore-toggle" disabled={!path.count}
                          onClick={() => (path.exploring ? path.stopExploring() : path.startExploring())}>
                          {path.exploring ? "Stop exploring" : "Explore trajectory"}
                        </button>
                        <button data-testid="return-to-path" onClick={path.returnToNearest} disabled={!path.count}>
                          Return to nearest anchor
                        </button>
                      </div>
                      <p className="camera-path-status" data-testid="explore-status">
                        {/* Only the words are a live region. The distance changes
                            several times a second, and announcing every reading
                            would make the panel unusable with a screen reader. */}
                        <span role="status">
                          {!path.exploring
                            ? "Not exploring · orbit, pan and zoom are yours"
                            : path.status?.atBoundary
                              ? "Trajectory boundary · at the edge of the reliable region"
                              : "Inside the trajectory tube"}
                          {path.exploring && !path.status?.pointerLocked && " · click the scene to capture the mouse"}
                        </span>
                        {path.exploring && (
                          <output aria-live="off">{" · "}{(path.status?.distance ?? 0).toFixed(3)} from the path
                            {path.status && path.status.anchor >= 0 && ` · nearest ${path.status.anchor + 1} / ${path.count}`}
                          </output>
                        )}
                      </p>
                      <div className="camera-path-row">
                        <button onClick={() => path.step(-1)} disabled={!path.count}>◀ Previous anchor</button>
                        <button onClick={() => path.step(1)} disabled={!path.count}>Next anchor ▶</button>
                        <label htmlFor="anchor-index">Anchor</label>
                        <input id="anchor-index" type="range" min="0" max={Math.max(path.count - 1, 0)} step="1"
                          value={Math.max(path.selected, 0)} disabled={path.count < 2}
                          aria-valuetext={pathAnchor ? `${path.selected + 1} of ${path.count}, ${pathAnchor.label}` : `${path.count} anchors`}
                          onChange={(event) => path.seek(Number(event.target.value))} />
                        <output htmlFor="anchor-index" data-testid="anchor-status">
                          {pathAnchor
                            ? `${path.selected + 1} / ${path.count} · ${pathAnchor.label} · ${pathAnchor.cameras.length} camera${pathAnchor.cameras.length === 1 ? "" : "s"}`
                            : `${path.count} anchors`}
                        </output>
                      </div>
                      <div className="camera-path-row">
                        <label htmlFor="explore-radius">Allowed radius</label>
                        <input id="explore-radius" type="range" data-testid="explore-radius"
                          {...sliderRange(path.spatial.minRadius, path.spatial.maxRadius)}
                          value={path.radius} onChange={(event) => path.setRadius(Number(event.target.value))} />
                        <output htmlFor="explore-radius">{path.radius.toFixed(3)}</output>
                        <label htmlFor="explore-speed">Walking speed</label>
                        <input id="explore-speed" type="range" data-testid="explore-speed"
                          {...sliderRange(path.spatial.minSpeed, path.spatial.maxSpeed)}
                          value={path.speed} onChange={(event) => path.setSpeed(Number(event.target.value))} />
                        <output htmlFor="explore-speed">{path.speed.toFixed(3)}/s</output>
                      </div>
                      <div className="camera-path-row">
                        <label className="camera-path-check">
                          <input type="checkbox" checked={path.display.showPath}
                            onChange={(event) => path.setDisplay({ ...path.display, showPath: event.target.checked })} /> Show path and anchors
                        </label>
                        <label className="camera-path-check">
                          <input type="checkbox" checked={path.display.showFrustums} data-testid="show-raw-frustums"
                            onChange={(event) => path.setDisplay({ ...path.display, showFrustums: event.target.checked })} /> Raw camera frustums (debug)
                        </label>
                        <label htmlFor="frustum-scale">Marker size</label>
                        <input id="frustum-scale" type="range" min="0.25" max="4" step="0.25" value={path.display.frustumScale}
                          onChange={(event) => path.setDisplay({ ...path.display, frustumScale: Number(event.target.value) })} />
                        <output htmlFor="frustum-scale">{path.display.frustumScale.toFixed(2)}x</output>
                      </div>
                      <p className="camera-path-note">
                        <strong>Explore trajectory</strong> walks the scene from inside. Click the canvas to capture the
                        mouse and look around, <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> to move,
                        <kbd>Q</kbd>/<kbd>E</kbd> down and up, <kbd>Shift</kbd> to go faster, arrow keys and
                        <kbd>Page Up</kbd>/<kbd>Page Down</kbd> to move and look without a mouse.
                        <kbd>Esc</kbd> releases the mouse; a second <kbd>Esc</kbd> leaves the mode.
                        The camera is held within <em>Allowed radius</em> of the line joining
                        the {path.count.toLocaleString()} anchors, because the reconstruction is only trustworthy
                        where the training cameras looked. Choosing an anchor moves where you stand and never turns
                        you: the {path.spatial.cameras.length.toLocaleString()} raw poses are cube faces shot from
                        those {path.count.toLocaleString()} places, and replaying their orientations spins on the spot.
                      </p>
                    </>}
                  </div>
                </details>
              )}
              {loaded?.representation === "gaussian-splat" ? (
                <GaussianSplatViewer
                  ref={splatViewer}
                  splat={loaded.splat}
                  view={storedView.view}
                  background={background}
                  loadStatus={loadStatus}
                  onReload={() => void restoreSample()}
                  onStatus={setRenderStatus}
                  trajectory={path.spatial}
                  selectedAnchor={path.selected}
                  exploring={path.exploring}
                  radius={path.radius}
                  speed={path.speed}
                  recenter={path.recenter}
                  trajectoryDisplay={path.display}
                  onAnchorChange={path.setSelected}
                  onExploreStatus={path.setStatus}
                  onExitExplore={path.stopExploring}
                />
              ) : (
              <PointCloudViewer
                ref={viewer}
                cloud={loaded?.representation === "point-cloud" ? loaded.cloud : null}
                view={storedView.view}
                interactionMode={interactionMode}
                operationMode={operationMode}
                selectionBusy={selection.busy || busy || loading}
                highlights={highlights}
                onSelectionError={setError}
                onSelection={(operation) => {
                  setError("");
                  void selection.draw(operation, (result) => {
                    setSelectedId(null); setSelectedGroup(null); setNotice("");
                    const bounds = result.result?.bounds;
                    setPosition(bounds ? { x: (bounds.min.x + bounds.max.x) / 2, y: (bounds.min.y + bounds.max.y) / 2, z: (bounds.min.z + bounds.max.z) / 2 } : null);
                    if (!bounds) setNotice("No points selected. Drag over the cloud to select a region.");
                  });
                }}
                loadStatus={loadStatus}
                onReload={() => void restoreSample()}
                pointSize={pointSize}
                background={background}
                picking={picking && !loading}
                annotations={records}
                selectedId={selectedId}
                selectedGroup={selectedGroup}
                draft={position}
                onPick={pick}
                onSelect={(id) => {
                  const record = records.find((a) => a.id === id);
                  if (record && !busy && !loading && !selection.busy) select(record);
                }}
                onStatus={setRenderStatus}
              />
              )}
              <div className="display-controls">
                {!isGaussian && (
                <div className="point-size-control">
                  <label htmlFor="point-size">Point size</label>
                  <input
                    id="point-size"
                    type="range"
                    min="1"
                    max="8"
                    step="0.5"
                    value={pointSize}
                    onChange={(event) =>
                      setPointSize(Number(event.target.value))
                    }
                  />
                  <output htmlFor="point-size">
                    {pointSize.toFixed(1)} px
                  </output>
                </div>
                )}
                <label className="background-control" htmlFor="background">
                  Background{" "}
                  <input
                    id="background"
                    type="color"
                    value={background}
                    onChange={(event) => setBackground(event.target.value)}
                  />
                </label>
              </div>
            </div>
            <div className="scene-details panel">
              <div className="detail-heading">
                <h3>About this scene</h3>
                <span className="tag">{isGaussian
                  ? (scene?.source === "hosted" ? "HOSTED SPLAT" : "GAUSSIAN SPLAT")
                  : scene?.source === "local" ? "LOCAL FILE" : "100% SYNTHETIC"}</span>
              </div>
              <p>
                {scene?.description ?? "Preparing a small, public-safe sample."}
              </p>
              <dl className="metadata-grid">
                <div>
                  <dt>{isGaussian ? "Splats" : "Points"}</dt>
                  <dd>
                    {(status.state === "ready"
                      ? status.count
                      : scene?.point_count
                    )?.toLocaleString() ?? "—"}
                  </dd>
                </div>
                <div>
                  <dt>Representation</dt>
                  <dd>{isGaussian ? "Gaussian Splat · Spark" : "Point cloud · Three.js"}</dd>
                </div>
                <div>
                  <dt>Format</dt>
                  <dd>{scene?.format ?? "—"}</dd>
                </div>
                <div>
                  <dt>File size</dt>
                  <dd>{scene ? (scene.size_bytes ? formatBytes(scene.size_bytes) : "Hosted · not downloaded") : "—"}</dd>
                </div>
                <div>
                  <dt>Coordinates</dt>
                  <dd>{scene?.coordinates ?? "—"}</dd>
                </div>
                <div><dt>Filename</dt><dd>{scene?.filename ?? "—"}</dd></div>
                {isGaussian ? (
                  <>
                    <div><dt>Annotation</dt><dd>Not available for Gaussian Splats yet</dd></div>
                    <div><dt>Splat color</dt><dd>Spherical harmonics from the splat file</dd></div>
                    <div data-testid="gaussian-delivery"><dt>Delivery</dt><dd>{splatDelivery}</dd></div>
                    <div data-testid="gaussian-identity"><dt>Scene identity</dt><dd>{splatIdentity}</dd></div>
                  </>
                ) : (
                  <>
                    <div><dt>Faces (ignored)</dt><dd>{scene?.face_count?.toLocaleString() ?? "—"}</dd></div>
                    <div><dt>Point color</dt><dd>{scene ? (scene.color_source === "vertex" ? "Original vertex colors" : "Fallback · sky blue") : "—"}</dd></div>
                  </>
                )}
              </dl>
            </div>
            <div className="navigation-guide">
              <span>
                ⌘ <b>Navigate</b> Drag to orbit · Right-drag to pan · Scroll to
                zoom
              </span>
              <span>
                <b>Touch</b> One finger orbits · Two fingers pan / pinch
              </span>
            </div>
          </section>
          <AnnotationPanel
            records={records}
            region={selection.region}
            position={position}
            selected={selected}
            picking={picking}
            unsupported={annotationsSupported ? null : GAUSSIAN_ANNOTATION_NOTICE}
            enabled={!!store && !corruption && annotationsSupported}
            busy={busy || loading || selection.busy}
            canPick={status.state === "ready"}
            error={error}
            notice={notice}
            canRecover={corruption?.sceneId === scene?.id && !!corruption}
            onRecover={async () => {
              if (!corruption || busy || loading || corruption.sceneId !== scene?.id) return;
              setBusy(true);
              try {
                const restored = await recoverLocalStorage(corruption);
                setStore(restored.store); setRecords(restored.records);
                cancel(); setCorruption(null);
                setNotice("Local notes are ready. Other scenes and browser data were kept.");
              } catch (reason) { setError(errorMessage(reason)); }
              finally { setBusy(false); }
            }}
            onNew={() => {
              if (picking) cancel();
              else {
                cancel();
                setPicking(true);
                if (window.innerWidth <= 800)
                  viewerArea.current?.scrollIntoView({ block: "start" });
              }
            }}
            onCancel={cancel}
            onSelect={select}
            selectedGroup={selectedGroup}
            onSelectGroup={(key) => {
              const members = records.filter((a) => groupKey(a.category, a.label) === key);
              void selection.show(members, false, () => {
                setSelectedGroup(key); setSelectedId(null); setPosition(null);
                setInteractionMode("navigate"); setError(""); setNotice("");
              });
            }}
            onSave={save}
            onDelete={remove}
          />
        </div>
        <ClassificationPanel
          viewerBusy={busy || loading}
          onShow={async (file, details, zUp) => {
            zUpFile.current = zUp ? file.name : null;
            await openFile(file, "point-cloud", details);
            viewerArea.current?.scrollIntoView({ block: "start", behavior: "smooth" });
          }}
        />
        <section className="future-modules" aria-labelledby="future-heading">
          <div className="future-heading">
            <h2 id="future-heading">Modules</h2>
            <p>Point-cloud viewing, Gaussian Splat viewing and ModelNet10 classification are active. Everything marked TBD is planned only and is not implemented.</p>
          </div>
          <div className="future-grid">
            <article><span className="tag">ACTIVE</span><h3>Point Cloud Viewer</h3><p>Three.js PLY viewing and annotation</p></article>
            <article><span className="tag">ACTIVE</span><h3>3D Gaussian Splat Viewer</h3><p>Spark.js splat viewing, local files only</p></article>
            <article><span className="tag">ACTIVE · HW2</span><h3>ModelNet10 Classification</h3><p>Shallow baseline and Mini-PointNet, interactive training</p></article>
            <article><span className="tag">TBD · PLANNED</span><h3>Training-camera trajectories</h3><p>Multi-scene exploration</p></article>
            <article><span className="tag">TBD · PLANNED</span><h3>Point Cloud to 3DGS</h3><p>Basic training pipeline</p></article>
            <article><span className="tag">TBD · PLANNED</span><h3>Multiview Reconstruction</h3><p>COLMAP preprocessing/reconstruction</p></article>
          </div>
        </section>
      </main>
      <footer className="app-footer">
        <span>
          NEURAL 3D <span className="footer-divider">/</span> Deep Learning ·
          HW1 · HW2
        </span>
        <span>Synthetic data. Real exploration.</span>
      </footer>
    </>
  );
}
