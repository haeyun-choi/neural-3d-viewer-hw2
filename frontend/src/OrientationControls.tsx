import { useState } from "react";
import type { CameraHandle, ViewPreferences, ViewPreset } from "./types";

export function OrientationControls({ camera, view, disabled, error, notice, onSave, onRestore }: {
  camera: () => CameraHandle | null; view: ViewPreferences | null; disabled: boolean;
  error: string; notice: string; onSave: (home: boolean) => void; onRestore: () => void;
}) {
  const [angle, setAngle] = useState("15");
  return <details className="orientation-controls" open={error ? true : undefined}>
    <summary>Orientation & views <span>{view?.home ? "Front/home saved" : view ? "Upright saved" : "Imported orientation"}</span></summary>
    <p>Orbit to face the object, then roll or choose its up axis. Save upright; Set as front saves this exact camera as home. Settings stay in this browser, per file. These controls move the camera only; point and splat coordinates are never rewritten.</p>
    <fieldset disabled={disabled}>
      <legend className="sr-only">Orientation controls</legend>
      <div className="orientation-row">
        <label>Roll angle (degrees)<input type="number" step="any" value={angle} onChange={(e) => setAngle(e.target.value)} /></label>
        <button onClick={() => camera()?.roll(Number(angle))} disabled={!angle || !Number.isFinite(Number(angle))}>Apply roll</button>
        {(["X", "Y", "Z"] as const).map((axis) => <button key={axis} onClick={() => camera()?.setUp(axis)}>{axis} up</button>)}
        <button onClick={() => camera()?.setUp("invert")}>Invert up</button>
      </div>
      <div className="orientation-row">
        <button onClick={() => onSave(false)}>Save upright</button>
        <button onClick={() => onSave(true)}>Set as front</button>
        <button onClick={() => camera()?.reset()}>Reset view</button>
        <button onClick={onRestore}>Restore imported orientation</button>
      </div>
      <div className="orientation-row" aria-label="Camera presets">
        {(["Front", "Back", "Left", "Right", "Top", "Bottom"] as ViewPreset[]).map((name) => <button key={name} onClick={() => camera()?.preset(name)}>{name}</button>)}
      </div>
    </fieldset>
    {error && <p className="alert error" role="alert">{error}</p>}
    {notice && <p role="status">{notice}</p>}
  </details>;
}
