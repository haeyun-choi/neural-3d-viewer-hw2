// Spark is a real npm dependency (@sparkjsdev/spark) that Vite bundles into the
// production output; nothing is fetched from a CDN at runtime. It is loaded on
// demand so that point-cloud-only sessions never download the splat renderer and
// its embedded WebAssembly parser.
export type SparkModule = typeof import("@sparkjsdev/spark");

let pending: Promise<SparkModule> | null = null;

export function loadSpark(): Promise<SparkModule> {
  pending ??= import("@sparkjsdev/spark").catch((error: unknown) => {
    pending = null;
    throw error;
  });
  return pending;
}
