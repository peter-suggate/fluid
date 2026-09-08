/** Node-only observer for generation preparation heap peaks. Load through
 * run-sparse-cm12-compilation-probe.mjs, not inherited NODE_OPTIONS. */
import { WebGPUSparseCM12Resident } from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import { observeMethodCalls } from "./sparse-cm12-call-observer";

const startedAt = performance.now();
const memory = (phase: string, extra?: unknown) => console.error(JSON.stringify({
  probe: "cm12-generation-host", phase, elapsedMilliseconds: performance.now() - startedAt,
  ...process.memoryUsage(), extra,
}));
const resident = WebGPUSparseCM12Resident as unknown as Record<string, any>;
const prototype = resident.prototype as Record<string, any>;
for (const name of ["captureGenerationTransferSourceWhileLeased", "prepareGenerationReplacement", "createReplacement"]) {
  observeMethodCalls(prototype, name, (phase, outcome) => memory(`${name}:${phase}`, { outcome }));
}
// Do not wrap a positional reporter argument or read lazy geometry merely to
// log a count. Named caller progress already records initialization phases.
observeMethodCalls(resident, "createConfigured", (phase, outcome) => memory(`createConfigured:${phase}`, { outcome }));
