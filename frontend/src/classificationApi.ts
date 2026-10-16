// Client for the FastAPI ModelNet10 classification routes (/api/classification).
const apiBase = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");
const base = `${apiBase}/classification`;

export interface ModelInfo {
  id: string;
  label: string;
  source: "bundled" | "run";
  architecture: string;
  config: Record<string, unknown>;
  parameters: number | null;
  test_accuracy: number | null;
  best_val_accuracy: number | null;
}

export interface ModelListing {
  classes: string[];
  default_model: string;
  models: ModelInfo[];
  activations: string[];
  architectures: string[];
}

export interface ClassificationSample {
  id: string;
  class: string;
  label: string;
  file: string;
  source_file: string;
  split: string;
}

export interface Prediction {
  predicted_class: string;
  confidence: number;
  top_k: { class: string; probability: number }[];
  model: { id: string; architecture: string; config: Record<string, unknown>; parameters: number | null; test_accuracy: number | null };
  input: { input_points: number; input_faces: number; model_points: number; sampling: string; format: string };
  inference_ms: number;
  preprocess_ms: number;
  sample?: ClassificationSample;
  filename?: string;
}

export interface EpochRow {
  epoch: number;
  train_loss: number;
  val_loss: number;
  train_accuracy: number;
  val_accuracy: number;
  grad_norm?: number;
  learning_rate?: number;
  seconds?: number;
}

export interface RunMetrics {
  model: string;
  parameters: number;
  best_epoch: number;
  best_val_accuracy: number;
  test_accuracy: number | null;
  test_error: number | null;
  device?: string;
  per_class?: { class: string; support: number; correct: number; recall: number; precision: number }[] | null;
}

export type RunStatus = "queued" | "running" | "completed" | "cancelled" | "failed";

export interface TrainingRun {
  id: string;
  status: RunStatus;
  phase: string;
  config: TrainingConfig;
  epoch: number;
  total_epochs: number;
  history: EpochRow[];
  metrics: RunMetrics | null;
  error: string | null;
  device: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface TrainingConfig {
  model: "shallow" | "mini_pointnet";
  epochs: number;
  learning_rate: number;
  hidden_dim: number;
  activation: string;
  batch_size: number;
}

export interface BundledResult {
  label: string;
  metrics: RunMetrics;
  history: EpochRow[];
  config: Record<string, unknown>;
}

export interface Results {
  models: Partial<Record<"shallow" | "mini_pointnet", BundledResult>>;
  comparison: {
    baseline: { model: string; test_accuracy: number; test_error: number; parameters: number };
    final: { model: string; test_accuracy: number; test_error: number; parameters: number };
    test_error_reduction: number;
    final_beats_baseline: boolean;
    test_size: number;
    seeds?: Record<string, { test_accuracy: number[]; mean: number; std: number }>;
  } | null;
}

export interface DatasetStatus {
  prepared: boolean;
  download_bytes: number;
  source: string;
  license: string;
  counts?: { train: number; val: number; test: number };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${base}${path}`, init);
  } catch {
    throw new Error("The classification backend is unreachable. Start the FastAPI server and retry.");
  }
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { loc: unknown[]; msg: string }) => `${d.loc.at(-1)}: ${d.msg}`).join("; ");
    } catch { /* keep the status text */ }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

export const classificationApi = {
  models: () => request<ModelListing>("/models"),
  samples: () => request<{ samples: ClassificationSample[]; source: string }>("/samples"),
  dataset: () => request<DatasetStatus>("/dataset"),
  results: () => request<Results>("/results"),
  samplePly: async (id: string): Promise<File> => {
    const response = await fetch(`${base}/samples/${encodeURIComponent(id)}/ply`).catch(() => null);
    if (!response?.ok) throw new Error("Could not download the sample point cloud from the backend.");
    return new File([await response.blob()], `${id}.ply`, { type: "application/octet-stream" });
  },
  predictSample: (sampleId: string, modelId: string) =>
    request<Prediction>("/predict/sample", json({ sample_id: sampleId, model_id: modelId, top_k: 3 })),
  predictUpload: (file: File, modelId: string) =>
    request<Prediction>(`/predict/upload?${new URLSearchParams({ model_id: modelId, top_k: "3", filename: file.name })}`, {
      method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file,
    }),
  startRun: (config: TrainingConfig) => request<TrainingRun>("/runs", json(config)),
  runs: () => request<{ runs: Pick<TrainingRun, "id" | "status">[] }>("/runs"),
  run: (id: string) => request<TrainingRun>(`/runs/${id}`),
  cancelRun: (id: string) => request<TrainingRun>(`/runs/${id}/cancel`, { method: "POST" }),
};
