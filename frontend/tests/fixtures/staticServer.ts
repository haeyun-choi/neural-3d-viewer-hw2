import { createReadStream, statSync } from "node:fs";
import { createServer } from "node:http";
import type { Server } from "node:http";
import { join, normalize } from "node:path";

/**
 * A minimal static host for the paged-RAD tests, deliberately written to the
 * same contract docs/hosted-scenes.md asks a real host to meet: cross-origin
 * reads, an exposed Content-Range, and optional HTTP Range support so the
 * "this host ignores Range" path can be exercised on purpose.
 */
export interface StaticHost {
  origin: string;
  /** Every path requested, in order, for asserting the paging pattern. */
  requests: { path: string; range: string | null }[];
  close: () => Promise<void>;
}

export async function startStaticHost(root: string, options: { ranges?: boolean } = {}): Promise<StaticHost> {
  const ranges = options.ranges ?? true;
  const requests: StaticHost["requests"] = [];
  const server: Server = createServer((request, response) => {
    const url = new URL(request.url ?? "/", "http://127.0.0.1");
    const relative = normalize(decodeURIComponent(url.pathname)).replace(/^(\.\.[/\\])+/, "").replace(/^[/\\]+/, "");
    const file = join(root, relative);
    const cors = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "Range",
      "Access-Control-Expose-Headers": "Content-Range, Content-Length, Accept-Ranges",
    };
    if (request.method === "OPTIONS") {
      response.writeHead(204, cors);
      response.end();
      return;
    }
    requests.push({ path: `/${relative}`, range: request.headers.range ?? null });
    let size: number;
    try {
      size = statSync(file).size;
    } catch {
      response.writeHead(404, cors);
      response.end("not found");
      return;
    }
    const header = ranges ? request.headers.range : undefined;
    const match = header ? /^bytes=(\d+)-(\d*)$/.exec(header) : null;
    if (!match) {
      response.writeHead(200, { ...cors, "Accept-Ranges": ranges ? "bytes" : "none", "Content-Length": String(size) });
      createReadStream(file).pipe(response);
      return;
    }
    const start = Number(match[1]);
    const end = Math.min(match[2] ? Number(match[2]) : size - 1, size - 1);
    if (start >= size) {
      response.writeHead(416, { ...cors, "Content-Range": `bytes */${size}` });
      response.end();
      return;
    }
    response.writeHead(206, {
      ...cors,
      "Accept-Ranges": "bytes",
      "Content-Range": `bytes ${start}-${end}/${size}`,
      "Content-Length": String(end - start + 1),
    });
    createReadStream(file, { start, end }).pipe(response);
  });

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("static host did not bind a port");
  return {
    origin: `http://127.0.0.1:${address.port}`,
    requests,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}
