// Browser side of the hosted-scene manifest: where the application is served
// from, and fetching frontend/public/data/scenes.json relative to it. The
// schema, validation and URL rules live in hostedScene.ts so they can be tested
// without a browser. See docs/hosted-scenes.md.
import { MANIFEST_MAX_BYTES, parseSceneManifest, sceneFromUrl as sceneFromUrlAt } from "./hostedScene";
import type { HostedScene } from "./hostedScene";

export { HostedSceneError, radFilename } from "./hostedScene";
export type { HostedScene } from "./hostedScene";

const MANIFEST_PATH = "data/scenes.json";

/** Absolute base the application is served from, honouring a non-root --base. */
export const appBase = () => new URL(import.meta.env.BASE_URL, window.location.href).toString();

/**
 * Read the optional hosted-scene manifest. Never throws: a missing, oversized,
 * malformed or unreachable manifest simply means no hosted scenes.
 */
export async function loadSceneManifest(signal?: AbortSignal): Promise<HostedScene[]> {
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}${MANIFEST_PATH}`, { signal });
    if (!response.ok) return [];
    const length = Number(response.headers.get("Content-Length") ?? 0);
    if (length > MANIFEST_MAX_BYTES) return [];
    return parseSceneManifest(await response.text(), appBase());
  } catch {
    return [];
  }
}

/** A hosted scene entered by URL in the import bar. */
export const sceneFromUrl = (raw: string): HostedScene => sceneFromUrlAt(raw, appBase());
