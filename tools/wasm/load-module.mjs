import { readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { NodeWebWorker } from "./node-web-worker.mjs";

/** Load one browser-targeted artifact under Node for ABI smoke tests. */
export async function loadFluidWasmForNode(repository = resolve(import.meta.dirname, "../.."), options = {}) {
  const artifact = options.artifact ?? "scalar";
  if (!new Set(["scalar", "simd", "threaded"]).has(artifact)) {
    throw new RangeError(`Unknown fluid Wasm artifact ${artifact}`);
  }
  if (artifact === "threaded") {
    globalThis.Worker = NodeWebWorker;
    globalThis.self = { addEventListener() {}, removeEventListener() {} };
  }
  const directory = resolve(repository, `public/wasm/fluid-wasm/${artifact}`);
  const wasmModule = await import(pathToFileURL(resolve(directory, "fluid_wasm.js")));
  const bytes = await readFile(resolve(directory, "fluid_wasm_bg.wasm"));
  await wasmModule.default({ module_or_path: bytes });
  if (artifact === "threaded") {
    if (typeof wasmModule.initThreadPool !== "function") {
      throw new TypeError("threaded fluid Wasm module has no initThreadPool");
    }
    await wasmModule.initThreadPool(options.threadCount ?? 4);
  }
  if (typeof wasmModule.FluidWorld?.from_scene !== "function") {
    throw new TypeError("fluid Wasm module has no FluidWorld.from_scene");
  }
  return wasmModule;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await loadFluidWasmForNode();
  process.stdout.write("loaded single-worker fluid Wasm module in Node\n");
}
