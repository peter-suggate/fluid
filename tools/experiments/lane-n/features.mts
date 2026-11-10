// Prints the Metal adapter's features and the formats relevant to a filterable apron atlas. No device work.
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "/Users/petersuggate/code/me/fluid-scratch/lane-n/tree/lib/harness/node-dawn-provider";
const dawn = await import(pathToFileURL("/Users/petersuggate/code/me/fluid-scratch/lane-n/tree/node_modules/webgpu/index.js").href);
Object.assign(globalThis, dawn.globals);
const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
const adapter = await gpu.requestAdapter();
console.log("features", [...adapter!.features].sort().join(","));
console.log("info", JSON.stringify(adapter!.info ?? {}));
process.exit(0);
