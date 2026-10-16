import { expect, test } from "@playwright/test";
import { selectRegion, validRegion } from "../src/regionSelection";
import { presetBasis } from "../src/CameraNavigation";
import { validView } from "../src/viewPreferences";
import type { RegionData, SelectionOperation, ViewPreferences } from "../src/types";

const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const positions = new Float32Array([-0.5, 0.5, 0, -0.5, 0.5, 0.7, 0.5, 0.5, 0, -0.5, -0.5, 0, 0.5, -0.5, 0, 0, 0, 2]);
const rectangle = (x0: number, y0: number, x1: number, y1: number, mode: SelectionOperation["mode"] = "replace"): SelectionOperation => ({
  mode, shape: "rectangle", polygon: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], matrix: identity,
});
const indices = async (ops: SelectionOperation[]) => Array.from((await selectRegion(positions, ops, new AbortController().signal)).indices);

test("rectangle selects exact unique indices through depth and excludes clipped points", async () => {
  expect(await indices([rectangle(0, 0, 0.5, 0.5)])).toEqual([0, 1]);
  expect(await indices([rectangle(0, 0, 1, 1)])).toEqual([0, 1, 2, 3, 4]);
});
test("replace, add, subtract and inclusive boundary semantics replay exactly", async () => {
  const top = rectangle(0, 0, 1, 0.5), bottom = rectangle(0, 0.5, 1, 1, "add");
  expect(await indices([top, bottom, rectangle(0.5, 0, 1, 1, "subtract")])).toEqual([0, 1, 3]);
  expect(await indices([top, rectangle(0.75, 0.75, 1, 1)])).toEqual([4]);
});
test("lasso polygon and native-coordinate bounds, no mutation of positions", async () => {
  const original = positions.slice();
  const result = await selectRegion(positions, [{ mode: "replace", shape: "lasso", polygon: [[0, 0], [0.6, 0], [0, 0.6]], matrix: identity }], new AbortController().signal);
  expect(Array.from(result.indices)).toEqual([0, 1]);
  expect(result.bounds?.min).toEqual({ x: -0.5, y: 0.5, z: 0 });
  expect(positions).toEqual(original);
});
test("empty selection cannot be persisted", async () => {
  const op = rectangle(0, 0, 0.1, 0.1);
  const result = await selectRegion(positions, [op], new AbortController().signal);
  expect(result.indices.length).toBe(0); expect(result.bounds).toBeNull();
  expect(validRegion({ version: 1, semantics: "through", vertex_count: 6, selected_count: 0, operations: [op] })).toBe(false);
});
test("cancellable multi-million-point scan yields progress without a vertex cap", async () => {
  const signal = new AbortController(), ticks: number[] = [];
  const many = new Float32Array(2_000_000 * 3);
  await expect(selectRegion(many, [rectangle(0, 0, 1, 1)], signal.signal, (n) => { ticks.push(n); signal.abort(); })).rejects.toThrow("cancelled");
  expect(ticks.length).toBeGreaterThan(0);
  const result = await selectRegion(many, [rectangle(0, 0, 1, 1)], new AbortController().signal);
  expect(result.indices.length).toBe(2_000_000); expect(result.indices.at(-1)).toBe(1_999_999);
});
test("region metadata is compact and rejects corrupt transforms or operation history", () => {
  const data: RegionData = { version: 1, semantics: "through", vertex_count: 5_000_001, selected_count: 4_000_000, operations: [rectangle(0, 0, 1, 1)] };
  expect(validRegion(data)).toBe(true); expect(JSON.stringify(data).length).toBeLessThan(400);
  for (const bad of [ { ...data, selected_count: data.vertex_count + 1 }, { ...data, operations: [{ ...data.operations[0], mode: "add" }] }, { ...data, operations: [{ ...data.operations[0], matrix: [NaN] }] }, { ...data, operations: Array(65).fill(data.operations[0]) } ]) expect(validRegion(bad)).toBe(false);
});
test("six presets form an orthogonal scene basis for an arbitrary saved up", () => {
  const view: ViewPreferences = { version: 1, up: [1, 0, 0], front: [0, 0, 1], home: null };
  const expected = { Front: [0, 0, 1], Back: [0, 0, -1], Left: [0, 1, 0], Right: [0, -1, 0], Top: [1, 0, 0], Bottom: [-1, 0, 0] };
  for (const name of Object.keys(expected) as (keyof typeof expected)[]) {
    const basis = presetBasis(view, name);
    basis.direction.toArray().forEach((n, i) => expect(n).toBeCloseTo(expected[name][i]));
    expect(basis.direction.dot(basis.up)).toBeCloseTo(0);
  }
  expect(validView(view)).toBe(true);
  expect(validView({ ...view, up: [0, 1, 0], front: [1e-6, Math.sqrt(1 - 1e-12), 0] })).toBe(true);
  expect(validView({ ...view, up: [0, 0, 1] })).toBe(false);
});
