import { useCallback, useEffect, useRef, useState } from "react";
import { classificationApi } from "./classificationApi";
import type {
  ClassificationSample, DatasetStatus, ModelInfo, ModelListing, Prediction, Results, TrainingConfig, TrainingRun,
} from "./classificationApi";
import { MetricChart } from "./MetricChart";
import { errorMessage } from "./storage";

const POLL_MS = 1000;
const ACTIVE = new Set(["queued", "running"]);
const TRAIN_COLOR = "#2a6fdb";
const VAL_COLOR = "#e07b00";
const percent = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined ? "—" : `${(value * 100).toFixed(digits)}%`;
const className = (name: string) => name.replace("_", " ");

type Input =
  | { kind: "sample"; sample: ClassificationSample }
  | { kind: "upload"; file: File };

const DEFAULT_CONFIG: TrainingConfig = {
  model: "mini_pointnet", epochs: 20, learning_rate: 0.001, hidden_dim: 64, activation: "relu", batch_size: 32,
};

/**
 * ModelNet10 classification and interactive training, docked under the viewer.
 * Objects are shown in the existing point-cloud viewer through `onShow`.
 */
export function ClassificationPanel({ onShow, viewerBusy }: {
  onShow: (file: File, details: { name: string; description: string }, zUp: boolean) => Promise<void>;
  viewerBusy: boolean;
}) {
  const [listing, setListing] = useState<ModelListing | null>(null);
  const [samples, setSamples] = useState<ClassificationSample[]>([]);
  const [results, setResults] = useState<Results | null>(null);
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus | null>(null);
  const [modelId, setModelId] = useState("mini_pointnet");
  const [sampleId, setSampleId] = useState("");
  const [input, setInput] = useState<Input | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState("");
  // Not being able to reach the backend on load is a state, not an error the
  // user caused, so it is announced politely rather than as an alert.
  const [unavailable, setUnavailable] = useState("");
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG);
  const [run, setRun] = useState<TrainingRun | null>(null);
  const [runError, setRunError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const uploadInput = useRef<HTMLInputElement>(null);

  const refreshModels = useCallback(async () => {
    const models = await classificationApi.models();
    setListing(models);
    return models;
  }, []);

  useEffect(() => {
    let cancelled = false;
    void Promise.all([refreshModels(), classificationApi.samples(), classificationApi.results(), classificationApi.dataset()])
      .then(([, sampleList, bundled, data]) => {
        if (cancelled) return;
        setSamples(sampleList.samples);
        setSampleId(sampleList.samples[0]?.id ?? "");
        setResults(bundled);
        setDatasetStatus(data);
      })
      .catch((reason: unknown) => { if (!cancelled) setUnavailable(errorMessage(reason)); });
    // After a reload, keep monitoring a run that is still queued or running.
    void classificationApi.runs()
      .then(({ runs }) => runs.find((r) => ACTIVE.has(r.status)))
      .then((active) => (active ? classificationApi.run(active.id) : null))
      .then((latest) => { if (!cancelled && latest) setRun((current) => current ?? latest); })
      .catch(() => { /* the monitor simply starts empty */ });
    return () => { cancelled = true; };
  }, [refreshModels]);

  // Poll the active run until it reaches a final state.
  const runId = run?.id;
  const runActive = run ? ACTIVE.has(run.status) : false;
  useEffect(() => {
    if (!runId || !runActive) return;
    let stopped = false;
    const timer = window.setInterval(() => {
      void classificationApi.run(runId).then((latest) => {
        if (stopped) return;
        setRun(latest);
        if (latest.status === "completed") void refreshModels();
      }).catch((reason: unknown) => { if (!stopped) setRunError(errorMessage(reason)); });
    }, POLL_MS);
    return () => { stopped = true; window.clearInterval(timer); };
  }, [runId, runActive, refreshModels]);

  async function show(next: Input) {
    setError("");
    setPrediction(null);
    try {
      if (next.kind === "sample") {
        const file = await classificationApi.samplePly(next.sample.id);
        await onShow(file, {
          name: `ModelNet10 · ${next.sample.label}`,
          description: `ModelNet10 official test-split object (${next.sample.source_file}), preprocessed exactly as for training: 1,024 surface points, centred and scaled into the unit sphere. Served by the classification backend from the repository's bundled samples. Z is up in ModelNet10.`,
        }, true);
      } else {
        await onShow(next.file, {
          name: next.file.name.replace(/\.ply$/i, "") || next.file.name,
          description: "Opened from your device for classification. Viewing it keeps the bytes in this browser; pressing Predict sends this PLY to the classification backend, which samples and normalises it in memory and does not store it.",
        }, false);
      }
      setInput(next);
    } catch (reason) { setError(errorMessage(reason)); }
  }

  async function predict() {
    if (!input) return;
    setPredicting(true);
    setError("");
    try {
      setPrediction(input.kind === "sample"
        ? await classificationApi.predictSample(input.sample.id, modelId)
        : await classificationApi.predictUpload(input.file, modelId));
    } catch (reason) {
      setPrediction(null);
      setError(errorMessage(reason));
    } finally { setPredicting(false); }
  }

  async function startTraining() {
    setSubmitting(true);
    setRunError("");
    try { setRun(await classificationApi.startRun(config)); }
    catch (reason) { setRunError(errorMessage(reason)); }
    finally { setSubmitting(false); }
  }

  async function cancelTraining() {
    if (!run) return;
    try { setRun(await classificationApi.cancelRun(run.id)); }
    catch (reason) { setRunError(errorMessage(reason)); }
  }

  const classes = listing?.classes ?? [];
  const models: ModelInfo[] = listing?.models ?? [];
  const history = run?.history ?? [];
  const last = history.at(-1);
  const baseline = results?.models.shallow?.metrics;
  const final = results?.models.mini_pointnet?.metrics;
  const truth = input?.kind === "sample" ? input.sample.class : null;
  const set = <K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) => setConfig((c) => ({ ...c, [key]: value }));
  const progress = run ? (run.status === "completed" ? 1 : run.epoch / Math.max(run.total_epochs, 1)) : 0;

  return (
    <section className="classification" aria-labelledby="classification-heading" data-testid="classification-panel">
      <div className="classification-heading">
        <div>
          <p className="eyebrow">HW2 · POINT-CLOUD OBJECT CLASSIFICATION</p>
          <h2 id="classification-heading">ModelNet10 classifier</h2>
        </div>
        <span className="tag">SHALLOW BASELINE · MINI-POINTNET</span>
      </div>
      <p className="classification-note" data-testid="classification-scope">
        Every object is assigned <strong>one of the ten ModelNet10 classes</strong>
        {classes.length ? <> — {classes.map(className).join(", ")}</> : null}. An uploaded PLY that is not one of these
        furniture categories is still forced into the closest of the ten, so treat its confidence accordingly.
      </p>

      {unavailable && <p className="alert" role="status" data-testid="classification-unavailable">
        {unavailable} Viewing still works; classification and training need the FastAPI backend.</p>}
      <div className="classification-grid">
        <article className="panel classification-card" aria-labelledby="classify-heading">
          <h3 id="classify-heading">Classify an object</h3>
          <div className="form-row">
            <label htmlFor="cls-model">Model</label>
            <select id="cls-model" value={modelId} onChange={(e) => { setModelId(e.target.value); setPrediction(null); }}>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}{m.test_accuracy !== null && m.test_accuracy !== undefined ? ` · test ${percent(m.test_accuracy)}` : ""}
                </option>
              ))}
            </select>
          </div>
          <div className="form-row">
            <label htmlFor="cls-sample">ModelNet10 sample</label>
            <select id="cls-sample" value={sampleId} onChange={(e) => setSampleId(e.target.value)}>
              {samples.map((s) => <option key={s.id} value={s.id}>{className(s.class)} · {s.label}</option>)}
            </select>
            <button disabled={!sampleId || viewerBusy}
              onClick={() => { const s = samples.find((x) => x.id === sampleId); if (s) void show({ kind: "sample", sample: s }); }}>
              Show sample
            </button>
          </div>
          <div className="form-row">
            <input ref={uploadInput} type="file" accept=".ply" hidden aria-label="Choose PLY to classify"
              onChange={(event) => {
                const file = event.currentTarget.files?.[0];
                event.currentTarget.value = "";
                if (file) void show({ kind: "upload", file });
              }} />
            <button disabled={viewerBusy} onClick={() => uploadInput.current?.click()}>Upload PLY to classify</button>
            <span className="hint">Mesh PLY: surface-sampled. Point-cloud PLY: subsampled. 64 MiB limit.</span>
          </div>
          <p className="current-input" data-testid="classification-input">
            {input
              ? input.kind === "sample"
                ? <>Input: <b>{input.sample.label}</b> · true class <b>{className(input.sample.class)}</b> · shown in the viewer</>
                : <>Input: <b>{input.file.name}</b> · uploaded only when you press Predict</>
              : "Choose a sample or upload a PLY; it opens in the viewer above."}
          </p>
          <button className="primary" data-testid="predict-button" disabled={!input || predicting || !models.length} onClick={() => void predict()}>
            {predicting ? "Predicting…" : "Predict"}
          </button>
          {error && <p className="alert error" role="alert" data-testid="classification-error">{error}</p>}
          {prediction && (
            <div className="prediction" data-testid="prediction" role="status">
              <div className="prediction-headline">
                <span className="prediction-class" data-testid="predicted-class">{className(prediction.predicted_class)}</span>
                <span className="prediction-confidence" data-testid="prediction-confidence">{percent(prediction.confidence)} confidence</span>
                {truth && (
                  <span className={`tag ${truth === prediction.predicted_class ? "ok" : "miss"}`}>
                    {truth === prediction.predicted_class ? "✓ matches label" : `✗ label is ${className(truth)}`}
                  </span>
                )}
              </div>
              <ol className="topk" aria-label="Top-3 probabilities">
                {prediction.top_k.map((row) => (
                  <li key={row.class}>
                    <span>{className(row.class)}</span>
                    <span className="bar"><i style={{ width: `${row.probability * 100}%` }} /></span>
                    <output>{percent(row.probability)}</output>
                  </li>
                ))}
              </ol>
              <dl className="metadata-grid compact">
                <div><dt>Model</dt><dd>{prediction.model.architecture} · {prediction.model.parameters?.toLocaleString() ?? "?"} params</dd></div>
                <div><dt>Model config</dt><dd>{Object.entries(prediction.model.config).map(([k, v]) => `${k} ${v}`).join(" · ")}</dd></div>
                <div><dt>Input</dt><dd>{prediction.input.format} · {prediction.input.input_points.toLocaleString()} points{prediction.input.input_faces ? ` · ${prediction.input.input_faces.toLocaleString()} faces` : ""}</dd></div>
                <div><dt>Model input</dt><dd>{prediction.input.model_points.toLocaleString()} points, {prediction.input.sampling}, unit sphere</dd></div>
                <div><dt>Timing</dt><dd>{prediction.preprocess_ms.toFixed(1)} ms preprocessing · {prediction.inference_ms.toFixed(1)} ms inference</dd></div>
              </dl>
            </div>
          )}
        </article>

        <article className="panel classification-card" aria-labelledby="train-heading">
          <h3 id="train-heading">Train a model</h3>
          <div className="train-form">
            <label>Architecture
              <select value={config.model} onChange={(e) => set("model", e.target.value as TrainingConfig["model"])}>
                <option value="mini_pointnet">Mini-PointNet</option>
                <option value="shallow">Shallow baseline</option>
              </select>
            </label>
            <label>Learning rate
              <input type="number" min="0.00001" max="0.1" step="0.0001" value={config.learning_rate}
                onChange={(e) => set("learning_rate", Number(e.target.value))} />
            </label>
            <label>Epochs
              <input type="number" min="1" max="200" step="1" value={config.epochs}
                onChange={(e) => set("epochs", Math.round(Number(e.target.value)))} />
            </label>
            <label>Hidden dimension
              <select value={config.hidden_dim} onChange={(e) => set("hidden_dim", Number(e.target.value))}>
                {[16, 32, 64, 128].map((h) => <option key={h} value={h}>{h}</option>)}
              </select>
            </label>
            <label>Activation
              <select value={config.activation} onChange={(e) => set("activation", e.target.value)}>
                {(listing?.activations ?? ["relu"]).map((a) => <option key={a} value={a}>{a}</option>)}
              </select>
            </label>
            <label>Batch size
              <select value={config.batch_size} onChange={(e) => set("batch_size", Number(e.target.value))}>
                {[16, 32, 64, 128].map((b) => <option key={b} value={b}>{b}</option>)}
              </select>
            </label>
          </div>
          <p className="hint">
            {config.model === "mini_pointnet"
              ? `Shared MLP 3→${config.hidden_dim}→${config.hidden_dim * 2}→${config.hidden_dim * 4}, max pool, head ${config.hidden_dim * 4}→${config.hidden_dim * 2}→10.`
              : `Shared 3→${config.hidden_dim} layer, max pool, linear ${config.hidden_dim}→10.`}
            {" "}Adam with cosine decay; checkpoint chosen on validation accuracy; the test split is scored once at the end.
            {datasetStatus && !datasetStatus.prepared && ` The first run downloads ModelNet10 (${(datasetStatus.download_bytes / 2 ** 20).toFixed(0)} MiB) and preprocesses it.`}
          </p>
          <div className="form-row">
            <button className="primary" data-testid="start-training" disabled={submitting || runActive} onClick={() => void startTraining()}>
              {runActive ? "Training…" : "Start training"}
            </button>
            <button data-testid="cancel-training" disabled={!runActive} onClick={() => void cancelTraining()}>Cancel</button>
          </div>
          {runError && <p className="alert error" role="alert">{runError}</p>}
          {run && (
            <div className="monitor" data-testid="training-monitor">
              <p className="run-status">
                <span className={`run-badge ${run.status}`} data-testid="run-status">{run.status}</span>
                <span data-testid="run-epoch">epoch {run.epoch} / {run.total_epochs}</span>
                <span>{run.phase}{run.device ? ` · ${run.device}` : ""}</span>
              </p>
              <progress value={progress} max={1} aria-label="Training progress" />
              {run.error && <p className="alert error" role="alert" data-testid="run-error">{run.error}</p>}
              <dl className="metadata-grid compact live-metrics" data-testid="live-metrics">
                <div><dt>Train loss</dt><dd>{last ? last.train_loss.toFixed(4) : "—"}</dd></div>
                <div><dt>Val loss</dt><dd>{last ? last.val_loss.toFixed(4) : "—"}</dd></div>
                <div><dt>Train accuracy</dt><dd>{percent(last?.train_accuracy)}</dd></div>
                <div><dt>Val accuracy</dt><dd>{percent(last?.val_accuracy)}</dd></div>
                {run.metrics && <>
                  <div><dt>Selected epoch</dt><dd>{run.metrics.best_epoch} · val {percent(run.metrics.best_val_accuracy)}</dd></div>
                  <div><dt>Test accuracy</dt><dd data-testid="run-test-accuracy">{percent(run.metrics.test_accuracy)}</dd></div>
                </>}
              </dl>
              <div className="charts">
                <MetricChart title="Loss" epochs={run.total_epochs} testId="loss-chart" format={(v) => v.toFixed(2)}
                  series={[{ label: "train", color: TRAIN_COLOR, values: history.map((r) => r.train_loss) },
                    { label: "validation", color: VAL_COLOR, values: history.map((r) => r.val_loss) }]} />
                <MetricChart title="Accuracy" epochs={run.total_epochs} yMax={1} testId="accuracy-chart" format={(v) => `${Math.round(v * 100)}%`}
                  series={[{ label: "train", color: TRAIN_COLOR, values: history.map((r) => r.train_accuracy) },
                    { label: "validation", color: VAL_COLOR, values: history.map((r) => r.val_accuracy) }]} />
              </div>
              {run.status === "completed" && <p className="hint">This run is now selectable in <b>Model</b> for prediction.</p>}
            </div>
          )}
        </article>
      </div>

      <article className="panel comparison" aria-labelledby="comparison-heading" data-testid="model-comparison">
        <h3 id="comparison-heading">Shallow baseline vs Mini-PointNet · official ModelNet10 test split</h3>
        {baseline && final ? (
          <>
            <table>
              <thead><tr><th>Model</th><th>Parameters</th><th>Selected epoch</th><th>Best val acc</th><th>Test accuracy</th><th>Test error</th></tr></thead>
              <tbody>
                {[["Shallow baseline", baseline], ["Mini-PointNet (final)", final]].map(([label, m]) => {
                  const metrics = m as typeof final;
                  return (
                    <tr key={label as string}>
                      <th scope="row">{label as string}</th>
                      <td>{metrics.parameters.toLocaleString()}</td>
                      <td>{metrics.best_epoch}</td>
                      <td>{percent(metrics.best_val_accuracy)}</td>
                      <td>{percent(metrics.test_accuracy)}</td>
                      <td>{percent(metrics.test_error)}</td>
                    </tr>
                  );
                })}
                {run?.status === "completed" && run.metrics && (
                  <tr className="current-run">
                    <th scope="row">This session&rsquo;s run · {run.config.model}</th>
                    <td>{run.metrics.parameters.toLocaleString()}</td>
                    <td>{run.metrics.best_epoch}</td>
                    <td>{percent(run.metrics.best_val_accuracy)}</td>
                    <td>{percent(run.metrics.test_accuracy)}</td>
                    <td>{percent(run.metrics.test_error)}</td>
                  </tr>
                )}
              </tbody>
            </table>
            {results?.comparison && (
              <p className="hint" data-testid="comparison-summary">
                Mini-PointNet lowers held-out test error from {percent(results.comparison.baseline.test_error)} to{" "}
                {percent(results.comparison.final.test_error)} on {results.comparison.test_size} test objects
                {results.comparison.seeds ? `; over seeds 0–2 the test accuracy is ${percent(results.comparison.seeds.shallow?.mean)} ± ${percent(results.comparison.seeds.shallow?.std)} vs ${percent(results.comparison.seeds.mini_pointnet?.mean)} ± ${percent(results.comparison.seeds.mini_pointnet?.std)}` : ""}.
              </p>
            )}
            <div className="charts">
              <MetricChart title="Bundled runs · validation accuracy" yMax={1} format={(v) => `${Math.round(v * 100)}%`}
                epochs={Math.max(results?.models.shallow?.history.length ?? 1, results?.models.mini_pointnet?.history.length ?? 1)}
                series={[{ label: "shallow", color: "#8a95a6", values: results?.models.shallow?.history.map((r) => r.val_accuracy) ?? [] },
                  { label: "mini-pointnet", color: VAL_COLOR, values: results?.models.mini_pointnet?.history.map((r) => r.val_accuracy) ?? [] }]} />
              <MetricChart title="Bundled runs · training loss"
                epochs={Math.max(results?.models.shallow?.history.length ?? 1, results?.models.mini_pointnet?.history.length ?? 1)}
                series={[{ label: "shallow", color: "#8a95a6", values: results?.models.shallow?.history.map((r) => r.train_loss) ?? [] },
                  { label: "mini-pointnet", color: TRAIN_COLOR, values: results?.models.mini_pointnet?.history.map((r) => r.train_loss) ?? [] }]} />
            </div>
          </>
        ) : <p className="hint">No bundled results found. Train both models with the commands in the README.</p>}
      </article>
    </section>
  );
}
