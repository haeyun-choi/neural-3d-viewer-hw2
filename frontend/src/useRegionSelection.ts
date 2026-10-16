import { useCallback, useEffect, useRef, useState } from "react";
import { selectRegion, MAX_SELECTION_OPERATIONS } from "./regionSelection";
import { groupColor, groupKey } from "./annotationGroups";
import type { Annotation, Highlight, PointCloudData, RegionData, RegionSelection, SelectionOperation } from "./types";

interface SelectionState { region: RegionData | null; result: RegionSelection | null; layers: Highlight[] }
const empty = (): SelectionState => ({ region: null, result: null, layers: [] });
export function useRegionSelection(cloud: PointCloudData | null, onError: (message: string) => void, prepare: (layers: Highlight[]) => void) {
  const [state, setState] = useState<SelectionState>(empty);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const pending = useRef<AbortController | null>(null);
  const abort = useCallback(() => { pending.current?.abort(); pending.current = null; setBusy(false); }, []);
  const clear = useCallback(() => { abort(); setState(empty()); }, [abort]);
  useEffect(() => () => { pending.current?.abort(); pending.current = null; }, [cloud]);

  async function calculate(work: (signal: AbortSignal, update: (n: number) => void) => Promise<SelectionState>, ready: (state: SelectionState) => void) {
    abort();
    const controller = new AbortController(); pending.current = controller;
    setBusy(true); setProgress(0);
    try {
      const result = await work(controller.signal, (n) => {
        if (pending.current === controller) setProgress(n);
      });
      if (controller.signal.aborted || pending.current !== controller) return;
      prepare(result.layers); // GPU allocation must succeed before replacing a valid selection.
      setState(result); ready(result);
    } catch (error) {
      if (!controller.signal.aborted && pending.current === controller)
        onError(error instanceof Error ? `Selection could not be completed: ${error.message} Previous selection preserved.` : "Selection could not be completed. Previous selection preserved.");
    } finally {
      if (pending.current === controller) { pending.current = null; setBusy(false); }
    }
  }
  async function draw(operation: SelectionOperation, ready: (state: SelectionState) => void) {
    if (!cloud) return;
    const previous = state.region?.operations ?? [];
    if (!previous.length && operation.mode === "subtract") return;
    const op = !previous.length ? { ...operation, mode: "replace" as const } : operation;
    const operations = op.mode === "replace" ? [op] : [...previous, op];
    if (operations.length > MAX_SELECTION_OPERATIONS) { onError("Selection history allows 64 operations. Choose Replace to start a new region."); return; }
    await calculate(async (signal, update) => {
      const result = await selectRegion(cloud.positions, operations, signal, update);
      return {
        region: result.indices.length ? { version: 1, semantics: "through", vertex_count: cloud.positions.length / 3, selected_count: result.indices.length, operations } : null,
        result, layers: [{ key: "draft", indices: result.indices, color: "#26734d" }],
      };
    }, ready);
  }
  async function show(records: Annotation[], individual: boolean, ready: (state: SelectionState) => void) {
    if (!cloud) return;
    await calculate(async (signal, update) => {
      const next = empty();
      const regions = records.filter((a) => a.kind === "region" && a.region);
      for (const [i, record] of regions.entries()) {
        const region = record.region!;
        if (region.vertex_count !== cloud.positions.length / 3) throw new Error("Region belongs to a different vertex layout.");
        const result = await selectRegion(cloud.positions, region.operations, signal, (n) => update((i + n) / regions.length));
        if (result.indices.length !== region.selected_count) throw new Error("Region replay count does not match its saved selection.");
        next.layers.push({ key: record.id, indices: result.indices, color: groupColor(groupKey(record.category, record.label)) });
        if (individual) { next.region = region; next.result = result; }
      }
      return next;
    }, ready);
  }
  return { ...state, busy, progress, abort, clear, draw, show };
}
