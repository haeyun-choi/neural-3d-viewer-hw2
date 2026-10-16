// Hosted large-scene descriptions and manifest parsing.
//
// A multi-gigabyte scene is never bundled with this repository. Instead a
// deployment may publish a small JSON manifest next to the application that
// points at prebuilt Spark RAD level-of-detail assets on static hosting:
//
//   frontend/public/data/scenes.json
//   {
//     "version": 1,
//     "scenes": [
//       {
//         "id": "campus-walk",
//         "name": "Campus walk",
//         "description": "12.6M splats, prebuilt RAD LoD tree.",
//         "url": "https://static.example.edu/scenes/campus-lod.rad",
//         "attribution": "Captured 2026-09; internal use only.",
//         "splatCount": 12615593,
//         "sizeBytes": 1280517688,
//         "trajectoryUrl": "https://static.example.edu/scenes/cameras.json",
//         "view": { "center": [0, 0, 0], "radius": 12.5 }
//       }
//     ]
//   }
//
// Only `id`, `name` and `url` are required; everything else is presentation or
// camera framing. The manifest carries no splat data, so it stays a few hundred
// bytes in Git while the RAD asset lives on a static host. Everything here is
// pure and base-URL agnostic so it can be tested without a browser; the fetch
// and the application base live in sceneManifest.ts.

export interface HostedScene {
  id: string;
  name: string;
  description: string;
  /** Absolute http(s) URL, already resolved against the application base. */
  url: string;
  attribution: string;
  /** Splat count declared by whoever built the asset; Spark's RAD header wins. */
  splatCount: number | null;
  sizeBytes: number | null;
  /** Optional Graphdeco cameras.json published beside the asset. */
  trajectoryUrl: string | null;
  view: { center: [number, number, number]; radius: number } | null;
}

export class HostedSceneError extends Error {}

/** A manifest is metadata only; anything larger is not one. */
export const MANIFEST_MAX_BYTES = 256 * 1024;

const finite3 = (value: unknown): value is [number, number, number] =>
  Array.isArray(value) && value.length === 3 && value.every((n) => typeof n === "number" && Number.isFinite(n));

const positive = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;

/**
 * Resolve a manifest or user-entered URL against the application base, and
 * refuse anything that is not a plain http(s) RAD asset. Rejecting other schemes
 * keeps a manifest from pointing the application at `javascript:` or `data:`.
 */
export function resolveRadUrl(raw: string, base: string): string {
  const text = raw.trim();
  if (!text) throw new HostedSceneError("Enter the URL of a Spark .rad asset.");
  let url: URL;
  try {
    url = new URL(text, base);
  } catch {
    throw new HostedSceneError("That is not a valid URL.");
  }
  if (url.protocol !== "http:" && url.protocol !== "https:")
    throw new HostedSceneError(`Hosted scenes must be served over http or https; this URL uses ${url.protocol}`);
  if (!/\.rad$/i.test(url.pathname))
    throw new HostedSceneError(
      "A hosted scene must be a prebuilt Spark .rad level-of-detail asset. Build one with Spark's build-lod tool; see docs/hosted-scenes.md.",
    );
  return url.toString();
}

/**
 * A hosted trajectory is an ordinary JSON document, so the same http(s)-only
 * rule applies as for the RAD asset: a manifest must not be able to point the
 * application at `javascript:` or `data:`.
 */
export function resolveTrajectoryUrl(raw: string, base: string): string {
  const text = raw.trim();
  if (!text) throw new HostedSceneError("Enter the URL of a cameras.json file.");
  let url: URL;
  try {
    url = new URL(text, base);
  } catch {
    throw new HostedSceneError("That is not a valid URL.");
  }
  if (url.protocol !== "http:" && url.protocol !== "https:")
    throw new HostedSceneError(`A hosted camera path must be served over http or https; this URL uses ${url.protocol}`);
  if (!/\.json$/i.test(url.pathname))
    throw new HostedSceneError("A hosted camera path must be a .json file in the Graphdeco cameras.json schema.");
  return url.toString();
}

export function radFilename(url: string): string {
  return decodeURIComponent(new URL(url).pathname.split("/").pop() || "scene.rad");
}

function readScene(entry: unknown, base: string): HostedScene | null {
  if (!entry || typeof entry !== "object") return null;
  const record = entry as Record<string, unknown>;
  const { id, name, url } = record;
  if (typeof id !== "string" || !id.trim()) return null;
  if (typeof name !== "string" || !name.trim()) return null;
  if (typeof url !== "string") return null;
  let resolved: string;
  try {
    resolved = resolveRadUrl(url, base);
  } catch {
    return null;
  }
  const view = record.view as Record<string, unknown> | undefined;
  const radius = view ? positive(view.radius) : null;
  let trajectoryUrl: string | null = null;
  if (typeof record.trajectoryUrl === "string" && record.trajectoryUrl.trim()) {
    // A bad trajectory URL drops only the trajectory, never the scene.
    try {
      trajectoryUrl = resolveTrajectoryUrl(record.trajectoryUrl, base);
    } catch {
      trajectoryUrl = null;
    }
  }
  return {
    id: id.trim(),
    name: name.trim(),
    description: typeof record.description === "string" ? record.description : "",
    url: resolved,
    attribution: typeof record.attribution === "string" ? record.attribution : "",
    splatCount: positive(record.splatCount),
    sizeBytes: positive(record.sizeBytes),
    trajectoryUrl,
    view: view && finite3(view.center) && radius !== null ? { center: view.center, radius } : null,
  };
}

/**
 * Parse a manifest document. Never throws: any JSON or schema problem yields an
 * empty list, and an individual malformed entry is dropped rather than taking
 * the rest of the manifest with it. Hosted scenes are an addition to the local
 * importers and must never break them.
 */
export function parseSceneManifest(text: string, base: string): HostedScene[] {
  try {
    if (text.length > MANIFEST_MAX_BYTES) return [];
    const data: unknown = JSON.parse(text);
    if (!data || typeof data !== "object") return [];
    const { version, scenes } = data as { version?: unknown; scenes?: unknown };
    if (version !== 1 || !Array.isArray(scenes)) return [];
    const seen = new Set<string>();
    const unique: HostedScene[] = [];
    for (const entry of scenes) {
      const scene = readScene(entry, base);
      if (!scene || seen.has(scene.id)) continue;
      seen.add(scene.id);
      unique.push(scene);
    }
    return unique;
  } catch {
    return [];
  }
}

/** A hosted scene entered by URL rather than declared in the manifest. */
export function sceneFromUrl(raw: string, base: string): HostedScene {
  const url = resolveRadUrl(raw, base);
  const filename = radFilename(url);
  return {
    id: url,
    name: filename.replace(/\.rad$/i, "") || filename,
    description:
      "Prebuilt Spark RAD level-of-detail asset opened by URL. Its chunks are paged over HTTP as the camera moves; nothing is downloaded in full.",
    url,
    attribution: "",
    splatCount: null,
    sizeBytes: null,
    trajectoryUrl: null,
    view: null,
  };
}
