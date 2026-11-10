// Compile one WGSL entry point through Dawn (no dispatch) and dump its MSL.
// usage: dump-msl.mts <wgsl> <entry> <constants.json> <out.metal>
import { readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { createProcessRetainedDawnGPU } from "/Users/petersuggate/code/me/fluid-scratch/lane-n/tree/lib/harness/node-dawn-provider";
import { requiredFluidDeviceLimits } from "/Users/petersuggate/code/me/fluid-scratch/lane-n/tree/lib/core/webgpu-device-limits";
const [wgsl, entry, constantsFile, out] = process.argv.slice(2) as [string, string, string, string];
const dawn = await import(pathToFileURL(resolve("/Users/petersuggate/code/me/fluid-scratch/lane-n/tree/node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal", "enable-dawn-features=skip_validation,dump_shaders,disable_symbol_renaming"]);
const adapter = await gpu.requestAdapter();
const device = await adapter!.requestDevice({ requiredFeatures: adapter!.features.has("subgroups") ? ["subgroups"] as any : [], requiredLimits: requiredFluidDeviceLimits(adapter!.limits) });
const code = readFileSync(wgsl, "utf8");
const module = device.createShaderModule({ code });
const info = await module.getCompilationInfo();
for (const m of info.messages) if (m.type === "error") console.log("ERR", m.message.slice(0, 400));
const t = performance.now();
await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: entry, constants: JSON.parse(readFileSync(constantsFile, "utf8")) } });
console.log("pipeline ms", (performance.now() - t).toFixed(0));
process.exit(0);
