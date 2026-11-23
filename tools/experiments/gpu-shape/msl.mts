// Compile one WGSL compute entry through Dawn (no dispatch) with dump_shaders: the MSL lands in this process's output.
//   node --import tsx tools/experiments/gpu-shape/msl.mts a.wgsl <entry>
// census.mts --caps runs one process a kernel and cuts the last "Dumped generated MSL" block out of its output.
import { readFileSync } from "node:fs";
const root = new URL("../../../", import.meta.url).pathname;
const { createProcessRetainedDawnGPU } = await import(`${root}lib/harness/node-dawn-provider.ts`);
const { requiredFluidDeviceLimits } = await import(`${root}lib/core/webgpu-device-limits.ts`);
const dawn = await import(`${root}node_modules/webgpu/index.js`);
Object.assign(globalThis, dawn.globals);
const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal", "enable-dawn-features=dump_shaders,disable_symbol_renaming"]);
const adapter = await gpu.requestAdapter();
const device = await adapter!.requestDevice({ requiredFeatures: ["subgroups", "float32-filterable"].filter(f => adapter!.features.has(f)) as any, requiredLimits: requiredFluidDeviceLimits(adapter!.limits) });
const f = process.argv[2]!, entry = process.argv[3]!;
const code = readFileSync(f, "utf8");
const module = device.createShaderModule({ code });
for (const m of (await module.getCompilationInfo()).messages) if (m.type === "error") console.error(`ERR ${m.lineNum}:${m.linePos} ${m.message.slice(0, 300)}`);
try { await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: entry } }); } catch (e) { console.error(`PIPE ERR ${String(e).slice(0, 300)}`); }
await new Promise(r => setTimeout(r, 500));
process.exit(0);
