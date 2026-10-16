import { validRegion } from "./regionSelection";
import type { Annotation, AnnotationInput, SceneMetadata } from "./types";

const apiBase = (import.meta.env.VITE_API_BASE_URL || "/api").replace(
  /\/$/,
  "",
);

export interface AnnotationStore {
  mode: "database" | "local";
  list: () => Promise<Annotation[]>;
  create: (input: AnnotationInput) => Promise<Annotation>;
  update: (id: string, input: AnnotationInput) => Promise<Annotation>;
  remove: (id: string) => Promise<void>;
}

const localKey = (sceneId: string) => `neural3d:annotations:v1:${sceneId}`;

export class CorruptStorageError extends Error {
  constructor(public readonly sceneId: string) {
    super("Local annotations could not be read. The saved data was preserved. You can reset this scene's local notes to recover; other scenes and browser data will be kept.");
  }
}

export function errorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "An unexpected error occurred. Please retry.";
}

function isAnnotation(value: unknown, sceneId: string): value is Annotation {
  if (!value || typeof value !== "object") return false;
  const a = value as Record<string, unknown>;
  return (
    ((a.kind === undefined || a.kind === "point") ? a.region == null : a.kind === "region" && validRegion(a.region)) &&
    typeof a.id === "string" &&
    a.id.length > 0 &&
    a.scene_id === sceneId &&
    typeof a.category === "string" &&
    a.category.trim().length > 0 &&
    a.category.length <= 40 &&
    typeof a.label === "string" &&
    a.label.trim().length > 0 &&
    a.label.length <= 80 &&
    typeof a.note === "string" &&
    a.note.length <= 1000 &&
    ["x", "y", "z"].every(
      (key) =>
        typeof a[key] === "number" &&
        Number.isFinite(a[key]),
    ) &&
    typeof a.created_at === "string" &&
    typeof a.updated_at === "string"
  );
}

function readRecords(value: unknown, sceneId: string): Annotation[] {
  if (
    !Array.isArray(value) ||
    !value.every((row) => isAnnotation(row, sceneId))
  ) {
    throw new Error(
      "Stored annotations are invalid. Existing data has been left untouched.",
    );
  }
  if (new Set(value.map((row) => row.id)).size !== value.length) {
    throw new Error(
      "Stored annotations contain duplicate IDs. Existing data has been left untouched.",
    );
  }
  return value;
}

async function request(
  path: string,
  method = "GET",
  body?: AnnotationInput | Record<string, string | number>,
): Promise<unknown> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 4000);
  try {
    const response = await fetch(`${apiBase}${path}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      const detail =
        typeof payload?.detail === "string"
          ? payload.detail
          : response.status === 422
            ? "Check the category, label, note, and coordinates."
            : "The annotation service could not complete the request.";
      throw new Error(`${detail} (HTTP ${response.status})`);
    }
    return response.status === 204 ? undefined : await response.json();
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(
        "The annotation service timed out. Reload to check whether the change was saved before retrying.",
        { cause: error },
      );
    }
    if (error instanceof TypeError)
      throw new Error(
        "Cannot reach the annotation service. Your form is still here; reconnect and retry.",
        { cause: error },
      );
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

function databaseStore(sceneId: string): AnnotationStore {
  const path = `/scenes/${encodeURIComponent(sceneId)}/annotations`;
  const row = (value: unknown): Annotation => {
    if (!isAnnotation(value, sceneId))
      throw new Error("The annotation service returned an invalid record.");
    return value;
  };
  return {
    mode: "database",
    list: async () => readRecords(await request(path), sceneId),
    create: async (input) => row(await request(path, "POST", input)),
    update: async (id, input) =>
      row(await request(`${path}/${encodeURIComponent(id)}`, "PUT", input)),
    remove: async (id) => {
      await request(`${path}/${encodeURIComponent(id)}`, "DELETE");
    },
  };
}

function localStore(sceneId: string): AnnotationStore {
  const key = localKey(sceneId);
  const read = (): Annotation[] => {
    let raw: string | null;
    try {
      raw = localStorage.getItem(key);
    } catch {
      throw new Error(
        "Browser storage is unavailable. Enable local storage or connect the backend to save annotations.",
      );
    }
    if (raw === null) return [];
    try {
      return readRecords(JSON.parse(raw), sceneId);
    } catch {
      throw new CorruptStorageError(sceneId);
    }
  };
  const write = (records: Annotation[]) => {
    try {
      localStorage.setItem(key, JSON.stringify(records));
    } catch {
      throw new Error(
        "Browser storage is full or disabled. This change was not saved.",
      );
    }
  };
  return {
    mode: "local",
    list: async () => read(),
    create: async (input) => {
      const now = new Date().toISOString();
      const record = {
        ...input,
        id: crypto.randomUUID(),
        scene_id: sceneId,
        created_at: now,
        updated_at: now,
      };
      write([...read(), record]);
      return record;
    },
    update: async (id, input) => {
      const records = read();
      const current = records.find((a) => a.id === id);
      if (!current)
        throw new Error(
          "This annotation no longer exists. Reload to refresh the list.",
        );
      const record = {
        ...current,
        ...input,
        updated_at: new Date().toISOString(),
      };
      write(records.map((a) => (a.id === id ? record : a)));
      return record;
    },
    remove: async (id) => {
      write(read().filter((a) => a.id !== id));
    },
  };
}

export async function recoverLocalStorage(error: CorruptStorageError) {
  const store = localStore(error.sceneId);
  try {
    // Another tab may already have repaired the value. Never discard valid notes.
    return { store, records: await store.list() };
  } catch (reason) {
    if (!(reason instanceof CorruptStorageError)) throw reason;
    try { localStorage.removeItem(localKey(error.sceneId)); }
    catch { throw new Error("Browser storage is unavailable. Recovery could not remove this scene's damaged notes."); }
    return { store, records: await store.list() };
  }
}

export async function connectStorage(
  scene: SceneMetadata,
): Promise<{ store: AnnotationStore; records: Annotation[] }> {
  const sceneId = scene.id;
  if (import.meta.env.VITE_STORAGE_MODE !== "local") {
    const store = databaseStore(sceneId);
    try {
      if (scene.source === "local" && sceneId !== "spectrum-garden") {
        await request(`/scenes/${encodeURIComponent(sceneId)}`, "PUT", {
          sha256: scene.sha256, name: scene.name, filename: scene.filename!,
          format: scene.format, point_count: scene.point_count,
          size_bytes: scene.size_bytes, coordinates: scene.coordinates,
          face_count: scene.face_count ?? 0, color_source: scene.color_source ?? "fallback",
        });
      }
      return { store, records: await store.list() };
    } catch {
      /* Explicit, visible local demo fallback on initial connection only. */
    }
  }
  const store = localStore(sceneId);
  return { store, records: await store.list() };
}
