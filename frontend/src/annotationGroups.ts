import type { Annotation } from "./types";

export const normalize = (value: string) => value.trim().toLowerCase();
export const groupKey = (category: string, label: string) =>
  JSON.stringify([normalize(category), normalize(label)]);

// Dark colors keep white marker numbers legible. Text, counts, outlines and
// marker numbers also identify groups; color is never the only signal.
const palette = ["#126782", "#925500", "#7545a0", "#26734d", "#a33855", "#3b5fa5"];
export function groupColor(key: string) {
  let hash = 2166136261;
  for (let i = 0; i < key.length; i++) hash = Math.imul(hash ^ key.charCodeAt(i), 16777619);
  return palette[(hash >>> 0) % palette.length];
}

export function groupAnnotations(records: Annotation[]) {
  const groups = new Map<string, {
    key: string; category: string; label: string; color: string;
    members: { record: Annotation; number: number }[];
  }>();
  records.forEach((record, index) => {
    const key = groupKey(record.category, record.label);
    if (!groups.has(key)) groups.set(key, {
      key, category: record.category.trim(), label: record.label.trim(),
      color: groupColor(key), members: [],
    });
    groups.get(key)!.members.push({ record, number: index + 1 });
  });
  return [...groups.values()];
}
