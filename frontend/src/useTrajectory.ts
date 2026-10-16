import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  buildSpatialTrajectory,
  parseGraphdecoCameras,
  TRAJECTORY_MAX_BYTES,
  TrajectoryError,
} from "./cameraTrajectory";
import type { SpatialTrajectory } from "./cameraTrajectory";

/**
 * One loaded capture path, plus the UI state for exploring it.
 *
 * The path belongs to the scene on screen: opening another Gaussian scene clears
 * it, because a path from one reconstruction means nothing in another. Local
 * files stay in the browser — the bytes are read, parsed and dropped, and
 * nothing is uploaded, written to SQLite or kept in localStorage.
 *
 * Nothing here runs per frame. Walking happens in the viewer's animation loop;
 * this hook holds only what a React render needs, and takes a throttled status
 * back from the loop so the readout can show where the walker is without a
 * re-render at sixty hertz.
 */
export interface LoadedTrajectory {
  spatial: SpatialTrajectory;
  origin: "local-file" | "hosted";
  filename: string;
  warnings: string[];
}

export interface TrajectoryDisplay {
  showPath: boolean;
  /** The raw per-image frustums. A debug aid; off by default. */
  showFrustums: boolean;
  frustumScale: number;
}

/** Live position of the walker relative to the tube, reported by the viewer. */
export interface ExploreStatus {
  /** Distance from the capture path, world units. */
  distance: number;
  /** True while the camera is resting against the boundary. */
  atBoundary: boolean;
  /** Nearest anchor index, for the readout and for "Return to nearest". */
  anchor: number;
  pointerLocked: boolean;
}

const DEFAULT_DISPLAY: TrajectoryDisplay = { showPath: true, showFrustums: false, frustumScale: 1 };

const message = (error: unknown) =>
  error instanceof TrajectoryError || error instanceof Error ? error.message : String(error);

/**
 * @param sceneKey  identity of the displayed scene; a change clears the path.
 * @param hostedUrl optional cameras.json declared by a hosted scene manifest.
 */
export function useTrajectory(sceneKey: string, hostedUrl: string | null) {
  const [trajectory, setTrajectory] = useState<LoadedTrajectory | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  // -1 is "no anchor chosen": a loaded path does not yank the camera off the
  // overview until the user asks for somewhere to be.
  const [selected, setSelected] = useState(-1);
  const [exploring, setExploring] = useState(false);
  const [radius, setRadiusState] = useState(0);
  const [speed, setSpeedState] = useState(0);
  const [display, setDisplay] = useState<TrajectoryDisplay>(DEFAULT_DISPLAY);
  const [status, setStatus] = useState<ExploreStatus | null>(null);
  /** Bumped to ask the viewer for a one-shot move; see returnToNearest. */
  const [recenter, setRecenter] = useState(0);
  const pending = useRef<AbortController | null>(null);

  const clear = useCallback(() => {
    pending.current?.abort();
    pending.current = null;
    setTrajectory(null);
    setError("");
    setBusy(false);
    setSelected(-1);
    setExploring(false);
    setStatus(null);
  }, []);

  // A new scene invalidates the path. This runs on the scene key rather than on
  // the scene object so a re-render of the same scene keeps the trajectory.
  const loadedFor = useRef(sceneKey);
  useEffect(() => {
    if (loadedFor.current === sceneKey) return;
    loadedFor.current = sceneKey;
    clear();
  }, [sceneKey, clear]);

  const accept = useCallback((spatial: SpatialTrajectory, warnings: string[], origin: LoadedTrajectory["origin"], filename: string) => {
    setTrajectory({ spatial, origin, filename, warnings: [...warnings, ...spatial.warnings] });
    // Both defaults are derived from the capture's own spacing, so a scene in
    // millimetres and a scene in metres both start out walkable.
    setRadiusState(spatial.defaultRadius);
    setSpeedState(spatial.defaultSpeed);
    setSelected(-1);
    setExploring(false);
    setStatus(null);
    setError("");
  }, []);

  /** Open a local cameras.json. The current path survives a malformed file. */
  const openFile = useCallback(async (file: File) => {
    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;
    setBusy(true);
    setError("");
    try {
      if (!/\.json$/i.test(file.name))
        throw new TrajectoryError("Choose a .json camera file in the Graphdeco cameras.json schema.");
      if (!file.size) throw new TrajectoryError("This camera file is empty.");
      if (file.size > TRAJECTORY_MAX_BYTES)
        throw new TrajectoryError("This camera file is too large to be a cameras.json.");
      const parsed = parseGraphdecoCameras(await file.text());
      if (controller.signal.aborted) return;
      accept(buildSpatialTrajectory(parsed.cameras), parsed.warnings, "local-file", file.name);
    } catch (reason) {
      if (!controller.signal.aborted) setError(message(reason));
    } finally {
      if (pending.current === controller) {
        setBusy(false);
        pending.current = null;
      }
    }
  }, [accept]);

  // A hosted scene may publish its cameras.json beside the RAD asset. A failure
  // here is reported but never blocks the scene, which is already on screen.
  useEffect(() => {
    if (!hostedUrl) return;
    const controller = new AbortController();
    setBusy(true);
    setError("");
    void (async () => {
      try {
        const response = await fetch(hostedUrl, { signal: controller.signal });
        if (!response.ok) throw new TrajectoryError(`The camera path returned HTTP ${response.status} ${response.statusText}.`);
        const length = Number(response.headers.get("Content-Length") ?? 0);
        if (length > TRAJECTORY_MAX_BYTES) throw new TrajectoryError("This camera file is too large to be a cameras.json.");
        const parsed = parseGraphdecoCameras(await response.text());
        if (controller.signal.aborted) return;
        accept(
          buildSpatialTrajectory(parsed.cameras),
          parsed.warnings,
          "hosted",
          decodeURIComponent(new URL(hostedUrl).pathname.split("/").pop() || "cameras.json"),
        );
      } catch (reason) {
        if (!controller.signal.aborted && !(reason instanceof DOMException && reason.name === "AbortError"))
          setError(`The scene's camera path could not be loaded: ${message(reason)}`);
      } finally {
        if (!controller.signal.aborted) setBusy(false);
      }
    })();
    return () => controller.abort();
    // sceneKey is a dependency so that reopening a scene refetches its path:
    // the effect above clears on a scene change, and it is declared first, so
    // the clear always runs before this fetch starts.
  }, [hostedUrl, sceneKey, accept]);

  const spatial = trajectory?.spatial ?? null;
  const count = spatial?.anchors.length ?? 0;

  /** Step between anchors. Position only — the view direction is the user's. */
  const step = useCallback((delta: number) => {
    if (!count) return;
    setSelected((index) => {
      if (index < 0) return delta > 0 ? 0 : count - 1;
      return Math.min(Math.max(index + delta, 0), count - 1);
    });
  }, [count]);

  const seek = useCallback((index: number) => {
    if (!count) return;
    setSelected(Math.min(Math.max(Math.round(index), 0), count - 1));
  }, [count]);

  const setRadius = useCallback((value: number) => {
    if (!spatial || !Number.isFinite(value)) return;
    setRadiusState(Math.min(Math.max(value, spatial.minRadius), spatial.maxRadius));
  }, [spatial]);

  const setSpeed = useCallback((value: number) => {
    if (!spatial || !Number.isFinite(value)) return;
    setSpeedState(Math.min(Math.max(value, spatial.minSpeed), spatial.maxSpeed));
  }, [spatial]);

  /**
   * Put the walker back on the path. Also the way out of a boundary the user has
   * pressed themselves into, and the reason the tube never needs an escape
   * hatch: the allowed set is convex enough locally that the nearest point is
   * always reachable, and this jumps straight to it.
   */
  const returnToNearest = useCallback(() => {
    if (!count) return;
    setRecenter((value) => value + 1);
  }, [count]);

  const startExploring = useCallback(() => { if (count) setExploring(true); }, [count]);
  const stopExploring = useCallback(() => { setExploring(false); setStatus(null); }, []);

  useEffect(() => { if (!count) setExploring(false); }, [count]);
  useEffect(() => () => pending.current?.abort(), []);

  /** "948 cameras · 158 trajectory anchors", the way the panel says it. */
  const summary = useMemo(() => {
    if (!spatial) return "";
    const plural = (n: number, word: string) => `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;
    return `${plural(spatial.cameras.length, "camera")} · ${plural(spatial.anchors.length, "trajectory anchor")}`;
  }, [spatial]);

  return {
    trajectory, spatial, error, busy, selected, exploring, radius, speed, display, status, recenter, count, summary,
    openFile, clear, step, seek, setRadius, setSpeed, returnToNearest, startExploring, stopExploring,
    setDisplay, setSelected, setStatus,
  };
}
