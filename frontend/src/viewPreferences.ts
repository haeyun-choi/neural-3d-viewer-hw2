import type { ViewPreferences } from "./types";
const key = (scene: string) => `neural3d:view:v1:${scene}`;
const vector = (v: unknown): v is [number, number, number] => Array.isArray(v) && v.length === 3 && v.every((n) => typeof n === "number" && Number.isFinite(n));
export function validView(value: unknown): value is ViewPreferences {
  if (!value || typeof value !== "object") return false;
  const v = value as ViewPreferences;
  if (v.version !== 1 || !vector(v.up) || !vector(v.front) || Math.abs(Math.hypot(...v.up) - 1) > 1e-5 || Math.abs(Math.hypot(...v.front) - 1) > 1e-5) return false;
  // OrbitControls permits views within 1e-6 radians of a pole. Use the cross
  // product instead of a rounded dot product so those valid poses can be saved.
  const cross = [v.up[1] * v.front[2] - v.up[2] * v.front[1],
    v.up[2] * v.front[0] - v.up[0] * v.front[2], v.up[0] * v.front[1] - v.up[1] * v.front[0]];
  if (Math.hypot(...cross) < 1e-12) return false;
  return v.home === null || (!!v.home && vector(v.home.position) && vector(v.home.target) && vector(v.home.up) &&
    Math.abs(Math.hypot(...v.home.up) - 1) < 1e-5 && Number.isFinite(v.home.aspect) && v.home.aspect > 0 &&
    Number.isFinite(v.home.near) && v.home.near > 0 && Number.isFinite(v.home.far) && v.home.far > v.home.near &&
    Math.hypot(...v.home.position.map((n, i) => n - v.home!.target[i])) > 0 &&
    v.home.up.every((n, i) => Math.abs(n - v.up[i]) < 1e-8));
}
export function readView(scene: string): { view: ViewPreferences | null; error: string; notice?: string } {
  let raw: string | null;
  try { raw = localStorage.getItem(key(scene)); }
  catch {
    // Unavailable storage is not evidence of corrupt saved data. Navigation
    // remains usable; a later explicit save reports a write failure if needed.
    return { view: null, error: "", notice: "Browser storage is unavailable. View settings cannot be restored or saved; navigation still works." };
  }
  try {
    if (raw === null) return { view: null, error: "" };
    const view: unknown = JSON.parse(raw);
    if (!validView(view)) throw new Error("invalid");
    return { view, error: "" };
  } catch {
    return { view: null, error: "Saved view settings could not be read. They were preserved. Restore imported orientation to reset only this scene's view settings." };
  }
}
export function saveView(scene: string, view: ViewPreferences) {
  if (!validView(view)) throw new Error("The current camera basis cannot be saved.");
  try { localStorage.setItem(key(scene), JSON.stringify(view)); }
  catch { throw new Error("View settings were not saved: browser storage is full or disabled. The current view is still usable."); }
}
export function clearView(scene: string) {
  try { localStorage.removeItem(key(scene)); }
  catch { throw new Error("Could not reset saved view settings. Browser storage is disabled."); }
}
