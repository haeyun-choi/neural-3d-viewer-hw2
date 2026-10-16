import { useCallback, useEffect, useRef, useState } from "react";
import { parsePly, PlyError } from "./ply";
import { FULL_RESOLUTION_ERROR, sizeWarning } from "./plyPolicy";
import type { WarningTier } from "./plyPolicy";
import type { PlySummary } from "./plyHeader";
import { GaussianError } from "./gaussianFormat";
import { isGaussianSplatPly, prepareHostedRadScene, prepareLocalGaussianScene } from "./gaussianSplat";
import type { GaussianProgress } from "./gaussianSplat";
import { loadSceneManifest } from "./sceneManifest";
import type { HostedScene } from "./hostedScene";
import { errorMessage } from "./storage";
import type {
  LoadedScene,
  PointCloudData,
  PreparedSplatMesh,
  RepresentationType,
  SceneMetadata,
  ViewerStatus,
} from "./types";

const assetUrl = (path: string) => `${import.meta.env.BASE_URL}${path}`;
interface ImportWarning { tier: Exclude<WarningTier, null>; summary: PlySummary; filename: string }
const importEmpty = (error: unknown) =>
  (error instanceof PlyError || error instanceof GaussianError) && error.empty;
const vector = (p: { x: number; y: number; z: number }) =>
  [p.x, p.y, p.z].map((n) => Number(n.toPrecision(6))).join(", ");

/** Optional wording for a file opened by another panel, such as a classification sample. */
export interface FileDetails { name: string; description: string }

export type LoadRequest =
  | { kind: "file"; file: File; representation: RepresentationType; details?: FileDetails }
  | { kind: "hosted"; scene: HostedScene };

/** Human-readable byte size; KiB below a mebibyte so small scenes read exactly. */
export function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return `${(bytes / 1024).toFixed(1)} KiB`;
}

function gaussianMessage(progress: GaussianProgress): string {
  switch (progress.phase) {
    case "inspecting":
      return `Inspecting ${progress.filename}…`;
    case "streaming":
      return `Streaming ${progress.filename} into the splat decoder…`;
    case "decoding":
      return "Decoding Gaussian splats…";
    case "lod":
      return "Building the level-of-detail tree…";
    case "paging":
      return "Opening the hosted level-of-detail asset…";
  }
}

export function useScene(onActivate: () => void, onPrepare: (cloud: PointCloudData) => void) {
  const [loaded, setLoaded] = useState<LoadedScene | null>(null);
  const [loading, setLoading] = useState(true);
  const [importError, setImportError] = useState("");
  const [warning, setWarning] = useState<ImportWarning | null>(null);
  const [gaussianProgress, setGaussianProgress] = useState<GaussianProgress | null>(null);
  const [hostedScenes, setHostedScenes] = useState<HostedScene[]>([]);
  const [loadStatus, setLoadStatus] = useState<ViewerStatus>({ state: "loading", message: "Loading point cloud…" });
  const pending = useRef<AbortController | null>(null);
  const sample = useRef<SceneMetadata | null>(null);
  const decision = useRef<((accepted: boolean) => void) | null>(null);

  // The Spark mesh belonging to the scene on screen. It is created by an import
  // before the scene changes and disposed here, after React has committed the
  // replacement, so neither a Strict Mode remount of the viewer nor a frame
  // rendered mid-swap can touch a disposed mesh.
  const displayedMesh = useRef<PreparedSplatMesh | null>(null);
  useEffect(() => {
    const current = loaded?.representation === "gaussian-splat" ? loaded.splat.mesh : null;
    const previous = displayedMesh.current;
    displayedMesh.current = current;
    if (previous && previous !== current) previous.dispose();
  }, [loaded]);
  useEffect(() => () => {
    displayedMesh.current?.dispose();
    displayedMesh.current = null;
  }, []);

  const cancelImport = useCallback(() => {
    pending.current?.abort();
    decision.current?.(false);
    decision.current = null;
    setWarning(null);
    setGaussianProgress(null);
    setLoading(false);
    setLoadStatus({ state: "empty", message: "Import cancelled. Open a file or reload the sample." });
  }, []);

  const load = useCallback(async (request?: LoadRequest) => {
    pending.current?.abort();
    decision.current?.(false);
    decision.current = null;
    setWarning(null);
    setGaussianProgress(null);
    const controller = new AbortController();
    pending.current = controller;
    setLoading(true);
    setImportError("");
    const gaussian = request?.kind === "hosted" || (request?.kind === "file" && request.representation === "gaussian-splat");
    setLoadStatus({ state: "loading", message: gaussian ? "Inspecting Gaussian Splat scene…" : "Inspecting PLY header…" });
    try {
      if (request?.kind === "hosted" || (request?.kind === "file" && request.representation === "gaussian-splat")) {
        const report = (progress: GaussianProgress) => {
          if (controller.signal.aborted) return;
          setGaussianProgress(progress);
          setLoadStatus({ state: "loading", message: gaussianMessage(progress) });
        };
        const splat = request.kind === "hosted"
          ? await prepareHostedRadScene(request.scene, controller.signal, report)
          : await prepareLocalGaussianScene(request.file, controller.signal, report);
        if (controller.signal.aborted) {
          splat.mesh.dispose();
          return;
        }
        const hosted = request.kind === "hosted" ? request.scene : null;
        const metadata: SceneMetadata = {
          id: `splat-${splat.sha256}`,
          name: hosted?.name ?? (splat.filename.replace(/\.[^.]+$/, "") || splat.filename),
          filename: splat.filename,
          representation: "gaussian-splat",
          source: hosted ? "hosted" : "local",
          asset: splat.url ?? "",
          description: hosted
            ? `${hosted.description || "Hosted Gaussian Splat scene."} Spark pages its level-of-detail chunks over HTTP as the camera moves; the asset is never downloaded in full and nothing is uploaded. Annotation is not available for Gaussian Splats yet.${hosted.attribution ? ` ${hosted.attribution}` : ""}`
            : "Gaussian Splat scene opened from your device. The splat file is streamed into the decoder in this browser session and is never uploaded to the backend. Annotation is not available for Gaussian Splats yet.",
          format: splat.format,
          sha256: splat.sha256,
          point_count: splat.declaredSplatCount ?? 0,
          size_bytes: splat.sizeBytes,
          coordinates: "Native file coordinates · adjustable view orientation · units unspecified",
        };
        onActivate();
        setLoaded({ representation: "gaussian-splat", metadata, splat });
        return;
      }

      let metadata: SceneMetadata;
      let blob: Blob;
      const file = request?.kind === "file" ? request.file : undefined;
      if (file) {
        if (!/\.ply$/i.test(file.name)) throw new PlyError("Only .ply files are supported. Choose a PLY file.");
        if (file.name.length > 255 || (/[\\/]/.test(file.name) || [...file.name].some((c) => c.charCodeAt(0) < 32))) throw new PlyError("Choose a PLY filename of at most 255 characters without directory or control characters.");
        // A 3DGS PLY also declares x/y/z, so the point-cloud parser would accept
        // it and draw a meaningless cloud. Route it by content, not by extension.
        if (await isGaussianSplatPly(file))
          throw new PlyError("This is a Gaussian Splat PLY, not a point cloud. Use Open Gaussian to view it.");
        if (controller.signal.aborted) return;
        blob = file;
      } else {
        if (!sample.current) {
          const response = await fetch(assetUrl("data/scene.json"), { signal: controller.signal });
          if (!response.ok) throw new Error(`Scene metadata unavailable (HTTP ${response.status}).`);
          const data: SceneMetadata = await response.json();
          if (!data.asset || data.id !== "spectrum-garden" || !data.sha256) throw new Error("Invalid sample metadata.");
          sample.current = data;
        }
        const response = await fetch(assetUrl(sample.current.asset), { signal: controller.signal });
        if (!response.ok) throw new Error(`Could not load the PLY (HTTP ${response.status}).`);
        blob = await response.blob();
      }
      if (controller.signal.aborted) return;
      const cloud = await parsePly(blob, controller.signal, async (summary) => {
        const tier = sizeWarning(summary.sizeBytes, summary.vertexCount);
        if (!tier) return true;
        setWarning({ tier, summary, filename: file?.name ?? "Spectrum Garden" });
        return new Promise<boolean>((resolve) => { decision.current = resolve; });
      }, (message) => {
        if (!controller.signal.aborted) setLoadStatus({ state: "loading", message });
      });
      if (controller.signal.aborted) return;
      if (file) {
        const details = request?.kind === "file" ? request.details : undefined;
        metadata = {
          id: cloud.sha256 === sample.current?.sha256 ? "spectrum-garden" : `ply-${cloud.sha256}`,
          name: details?.name ?? (file.name.replace(/\.ply$/i, "") || file.name), filename: file.name, source: "local", asset: "",
          representation: "point-cloud",
          description: details?.description ?? "Opened from your device. PLY bytes stay in this browser session. Only scene metadata and annotations may be saved to the connected database.",
          format: cloud.format, sha256: cloud.sha256, point_count: cloud.positions.length / 3, size_bytes: file.size,
          coordinates: `Native file coordinates · adjustable view orientation · units unspecified. Bounds [${vector(cloud.bounds.min)}] → [${vector(cloud.bounds.max)}]`,
        };
      } else {
        metadata = { ...sample.current!, representation: "point-cloud", source: "bundled", filename: sample.current!.asset.split("/").pop() };
        if (cloud.sha256 !== metadata.sha256) throw new PlyError("The sample checksum does not match its metadata. Restore the bundled sample and retry.");
      }
      metadata.face_count = cloud.face_count;
      metadata.color_source = cloud.color_source;
      // Upload replacement buffers successfully before changing the active scene
      // or its annotation store. A failed allocation leaves the old view intact.
      try { onPrepare(cloud); } catch { throw new PlyError(FULL_RESOLUTION_ERROR); }
      onActivate();
      setLoaded({ representation: "point-cloud", metadata, cloud });
    } catch (error) {
      if (!controller.signal.aborted && !(error instanceof DOMException && error.name === "AbortError")) {
        const message = error instanceof RangeError ? FULL_RESOLUTION_ERROR : errorMessage(error);
        setImportError(message);
        setLoadStatus({ state: importEmpty(error) ? "empty" : "error", message });
      }
    } finally {
      if (pending.current === controller) {
        setLoading(false);
        setWarning(null);
        setGaussianProgress(null);
        decision.current = null;
      }
    }
  }, [onActivate, onPrepare]);

  useEffect(() => {
    void load();
    return () => { pending.current?.abort(); pending.current = null; decision.current?.(false); decision.current = null; };
  }, [load]);

  // The hosted-scene manifest is optional metadata; a deployment without one
  // simply has no hosted scenes, and a failure here never blocks the importers.
  useEffect(() => {
    const controller = new AbortController();
    void loadSceneManifest(controller.signal).then((scenes) => {
      if (!controller.signal.aborted) setHostedScenes(scenes);
    });
    return () => controller.abort();
  }, []);

  const continueImport = () => { setWarning(null); decision.current?.(true); decision.current = null; };
  return {
    loaded, loading, importError, loadStatus, warning, gaussianProgress, hostedScenes,
    continueImport, cancelImport,
    openFile: (file: File, representation: RepresentationType = "point-cloud", details?: FileDetails) =>
      load({ kind: "file", file, representation, details }),
    openHosted: (scene: HostedScene) => load({ kind: "hosted", scene }),
    restoreSample: () => load(),
  };
}
