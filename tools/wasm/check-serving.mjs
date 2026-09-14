import assert from "node:assert/strict";
import { readdir } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "../..");
const origin = new URL(process.argv[2] ?? "http://localhost:3000");
const assets = (await readdir(resolve(root, "dist/client/assets")))
  .filter(name => name.includes("worker") && name.endsWith(".js"))
  .map(name => `/assets/${name}`);
assert.ok(assets.some(path => path.includes("simulation.worker")), "Build the worker assets first");
const snippets = await readdir(resolve(root, "public/wasm/fluid-wasm/threaded/snippets"));
const rayon = snippets.find(name => name.startsWith("wasm-bindgen-rayon-"));
assert.ok(rayon, "Build the threaded artifact first");
const paths = ["/", ...assets,
  "/wasm/fluid-wasm/threaded/fluid_wasm.js",
  "/wasm/fluid-wasm/threaded/fluid_wasm_bg.wasm",
  `/wasm/fluid-wasm/threaded/snippets/${rayon}/src/workerHelpers.js`];
const results = await Promise.allSettled(paths.map(async path => {
  const response = await fetch(new URL(path, origin), { method: "HEAD", signal: AbortSignal.timeout(5000) });
  assert.equal(response.status, 200, `${path} status`);
  if (path.endsWith(".wasm")) assert.equal(response.headers.get("content-type"), "application/wasm", `${path}: streaming Wasm MIME type`);
  if (path.startsWith("/wasm/")) assert.equal(response.headers.get("cache-control"), "no-cache", `${path}: stable artifacts must revalidate`);
  for (const [name, value] of Object.entries({
    "cross-origin-opener-policy": "same-origin",
    "cross-origin-embedder-policy": "require-corp",
    "cross-origin-resource-policy": "same-origin",
  })) assert.equal(response.headers.get(name), value, `${path}: ${name}`);
}));
const failures = results.filter(result => result.status === "rejected");
if (failures.length) throw new AggregateError(failures.map(result => result.reason), "Physics worker serving policy failed");
console.log(`Verified isolation headers on ${paths.length} HTML, worker and Wasm responses from ${origin.origin}`);
