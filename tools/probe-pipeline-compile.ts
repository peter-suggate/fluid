/** Cold compile time of dumped compute pipelines, one at a time.
 *
 * tools/profile-solver-construction.ts --dump=<dir> writes every module's WGSL
 * and every compute pipeline's entry point and override constants. This
 * compiles the matching ones alone, in order, with nothing else in flight, and
 * prints each one's createComputePipelineAsync time. The Metal shader cache is
 * a fresh directory of this process's own unless FLUID_METAL_SHADER_CACHE
 * names one, so every compile is cold.
 *
 * env WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-pipeline-compile.ts
 *   --dump=<dir> [--module=<regex>] [--entry=<regex>] [--limit=N] [--source=<sha>:<file.wgsl>]
 * --source replaces the dumped WGSL of module <sha> with <file.wgsl> (an edited copy). */
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, nodeDawnOptions } from "../lib/harness/node-dawn-provider";

const options = process.argv.slice(2);
const option = (key: string) => options.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
for (const a of options) assert.ok(/^--(?:dump|module|entry|limit|source)=.+$/.test(a), `Unknown option: ${a}`);
const dump = resolve(option("dump") ?? assert.fail("--dump=<dir> is required"));
const moduleFilter = new RegExp(option("module") ?? ""), entryFilter = new RegExp(option("entry") ?? "");
const limit = Number(option("limit") ?? Infinity);
const sources = new Map(options.filter(a => a.startsWith("--source=")).map(a => { const [sha, file] = a.slice(9).split(":"); return [sha!, file!]; }));
process.env.FLUID_METAL_SHADER_CACHE ??= mkdtempSync(join(tmpdir(), "fluid-compile-probe-"));

interface Dumped { module?: string; sha?: string; entryPoint?: string; constants?: Record<string, number> }
const all: Dumped[] = JSON.parse(readFileSync(join(dump, "pipelines.json"), "utf8"));
const chosen = all.filter(p => moduleFilter.test(p.module ?? "") && entryFilter.test(p.entryPoint ?? "")).slice(0, limit);

const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const gpu = createProcessRetainedDawnGPU(dawn, nodeDawnOptions(["backend=metal"]));
const adapter = await gpu.requestAdapter(); assert.ok(adapter);
const device: GPUDevice = await adapter.requestDevice({ requiredFeatures: [...(["subgroups", "float32-filterable"] as const).filter(f => adapter.features.has(f))],
  requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
device.addEventListener("uncapturederror", e => console.error(e.error.message));
console.log(JSON.stringify({ dump, cache: process.env.FLUID_METAL_SHADER_CACHE, pipelines: chosen.length, of: all.length }));

const modules = new Map<string, GPUShaderModule>();
let total = 0;
for (const p of chosen) {
  let module = modules.get(p.sha!);
  if (!module) {
    const code = readFileSync(sources.get(p.sha!) ?? join(dump, `${p.sha}.wgsl`), "utf8");
    module = device.createShaderModule({ code, label: p.module });
    for (const m of (await module.getCompilationInfo()).messages) if (m.type === "error") console.log(`${m.lineNum}: ${m.message}`);
    modules.set(p.sha!, module);
  }
  const start = performance.now();
  try {
    await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: p.entryPoint!, constants: p.constants } });
    const ms = performance.now() - start; total += ms;
    console.log(JSON.stringify({ module: p.module, entryPoint: p.entryPoint, ms: Math.round(ms), constants: p.constants }));
  } catch (error) {
    console.log(JSON.stringify({ module: p.module, entryPoint: p.entryPoint, error: String((error as Error).message).slice(0, 400) }));
  }
}
console.log(JSON.stringify({ total_ms: Math.round(total) }));
device.destroy(); process.exit(0);
