# Hosted large Gaussian scenes

Two paths open a Gaussian Splat scene in this viewer, and they solve different
problems.

| Path | Use it for | What crosses the network | LoD |
| --- | --- | --- | --- |
| **Open Gaussian** (local file) | a file on your own machine, any size | nothing — the file is streamed into Spark's decoder inside the browser | built on load above the large-scene threshold |
| **Hosted large scene** (`.rad` URL) | a published multi-gigabyte scene | only the chunks the camera needs | prebuilt offline by `build-lod` |

The local path has no size limit and never uploads anything, but it still has to
decode the whole file once per visit. For a scene you publish — a campus, a
building, a hospital wing — prebuild the level-of-detail tree once, host it as a
static asset, and let Spark page it. **No multi-gigabyte asset belongs in this
repository**; `.rad`, `.radc` and `.ply` payloads are ignored by Git on purpose.

## 1. Build the RAD asset offline

Spark ships a Rust command-line tool, `build-lod`, that turns any splat file it
can read (`.ply` including PlayCanvas-compressed, `.spz`, `.splat`, `.ksplat`,
`.sog`) into a `.rad` level-of-detail tree. It needs a Rust toolchain
(<https://rust-lang.org/tools/install/>) and a clone of the Spark repository;
the published npm package contains only `dist/`.

```bash
git clone https://github.com/sparkjsdev/spark
cd spark && npm install

# single-file layout: one scene-lod.rad, paged with HTTP range requests
npm run build-lod -- /path/to/final_3dgs/point_cloud.ply --quality

# chunked layout: point_cloud-lod.rad + point_cloud-lod-<n>.radc
npm run build-lod -- /path/to/final_3dgs/point_cloud.ply --quality --rad-chunked
```

Useful options: `--quality` (slower, better `bhatt-lod`; recommended offline),
`--quick` (default `tiny-lod`), `--max-sh=#` to cap spherical harmonics, and
`--rad-chunked` for the chunked layout. The output is named after the input with
a `-lod` suffix.

If you already have a single-file `.rad` and need the chunked layout — typically
because the host cannot serve byte ranges — `scripts/rad_chunk.py` rewrites the
header's chunk table and copies the chunk payloads out byte for byte, leaving
the splat data exactly as `build-lod` produced it:

```bash
python scripts/rad_chunk.py scene-lod.rad --output out/
```

## 2. Host it

Both layouts are plain static files. What the host must do:

* **CORS**, when the asset is not served from the application's own origin:
  `Access-Control-Allow-Origin` for that origin, `Access-Control-Allow-Headers:
  Range` (a `Range` request header triggers a preflight), and
  `Access-Control-Expose-Headers: Content-Range, Content-Length, Accept-Ranges`
  so the browser can read the reply's length.
* **HTTP Range (206)** — required by the single-file layout and by nothing else.
  Spark reads the RAD header with `Range: bytes=0-65535` and then fetches each
  64K-splat chunk as its own byte range. A host that ignores `Range` and answers
  `200` with the whole file turns every page fetch into a full download; the
  viewer detects this before the scene changes and says so instead.
* The **chunked layout needs no Range support at all**: the small `.rad` header
  is fetched whole and each `-<n>.radc` chunk is an ordinary `GET`. Chunk files
  are resolved relative to the `.rad` URL, so keep them in the same directory.
* Long-lived cache headers are safe: a RAD asset is immutable by convention, and
  the viewer identifies a hosted scene by its URL for exactly that reason.

## 3. Point the viewer at it

Either paste the URL into **Hosted large scene → RAD asset URL**, or publish a
manifest at `frontend/public/data/scenes.json` so the scene appears in the
picker. The manifest is metadata only — a few hundred bytes, safe to commit:

```json
{
  "version": 1,
  "scenes": [
    {
      "id": "campus-walk",
      "name": "Campus walk",
      "description": "12.6M splats from final_3dgs, prebuilt RAD LoD tree.",
      "url": "https://static.example.edu/scenes/point_cloud-lod.rad",
      "attribution": "Captured 2026-09. Internal use only.",
      "splatCount": 12615593,
      "sizeBytes": 1280517688,
      "view": { "center": [0, 1.4, 0], "radius": 12.5 }
    }
  ]
}
```

`id`, `name` and `url` are required; the rest is presentation and camera framing.
`url` may be relative to the application base. Only `http` and `https` URLs
ending in `.rad` are accepted, a malformed entry is dropped rather than breaking
the manifest, and a missing manifest simply means no hosted scenes. `view` skips
the sampled auto-fit and frames the scene exactly, which is worth setting for a
large scene whose extent is dominated by outliers.

## What the viewer does with it

1. One ranged request checks that the URL is reachable, that CORS is configured,
   that the first bytes are the `RAD0` signature, and — for a single-file asset —
   that the host answered `206`. Any failure is reported **before** the displayed
   scene changes.
2. Spark's own RAD decoder reads the header and reports the splat count, the
   chunk count and whether the layout is single-file or chunked.
3. `SplatMesh({ url, paged: true, fileType: SplatFileType.RAD })` hands paging to
   Spark. `SparkRenderer` creates the shared page pool, fetches the root chunk
   first and then chunks by distance to the camera, evicting by least-recent use.
4. The camera is framed once the first page is resident, or from the manifest's
   `view` when it has one.

## Limits worth knowing

* A paged scene reports the splat count from the RAD header; the number resident
  in the page pool at any instant is the renderer's LoD budget (2.5M on desktop,
  1–1.5M on mobile), not the whole scene.
* Paging cannot start until the renderer exists, so a hosted scene is attached
  after the preflight succeeds and shows "Streaming level-of-detail pages…" until
  its first page arrives. If none arrives within 90 s the viewer reports it.
* Annotation is not available for Gaussian scenes, hosted or local.
