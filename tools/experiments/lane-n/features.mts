// Prints the Metal adapter's features and the formats relevant to a filterable apron atlas. No device work.
import { createProcessRetainedDawnGPU } from "../../../lib/harness/node-dawn-provider";
const dawn = await import(new URL("../../../node_modules/webgpu/index.js", import.meta.url).href);
Object.assign(globalThis, dawn.globals);
const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
const adapter = await gpu.requestAdapter();
console.log("features", [...adapter!.features].sort().join(","));
console.log("info", JSON.stringify(adapter!.info ?? {}));
process.exit(0);
