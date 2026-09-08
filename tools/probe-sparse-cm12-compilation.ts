/** Node-only canonical-lane preload. Counts direct compilation separately
 * from manifest-cache statistics, with bounded output and weak device keys. */
import "./probe-sparse-cm12-generation-host";
import { GPUCompilationManager } from "../lib/core/gpu-compilation-manager";

interface CompilationTimes {
  manager: number; modules: number; sourceCodeUnits: number; moduleCallMs: number;
  pipelines: number; queueWaitMs: number; executionMs: number;
  slowest: { label: string; queueWaitMs: number; executionMs: number }[];
}
const started = performance.now();
const summaries: CompilationTimes[] = [];
const byManager = new WeakMap<object, CompilationTimes>();
const queued = new WeakMap<object, number>();
function times(manager: object) {
  let summary = byManager.get(manager);
  if (!summary) {
    summary = { manager: summaries.length, modules: 0, sourceCodeUnits: 0,
      moduleCallMs: 0, pipelines: 0, queueWaitMs: 0, executionMs: 0, slowest: [] };
    byManager.set(manager, summary); summaries.push(summary);
  }
  return summary;
}
const prototype = GPUCompilationManager.prototype as unknown as Record<string, any>;
const createModule = prototype.createShaderModule;
prototype.createShaderModule = function (descriptor: GPUShaderModuleDescriptor) {
  const summary = times(this), begin = performance.now();
  try { return createModule.call(this, descriptor); }
  finally { summary.modules++; summary.sourceCodeUnits += descriptor.code.length;
    summary.moduleCallMs += performance.now() - begin; }
};
const enqueue = prototype.enqueueDirect;
prototype.enqueueDirect = function (kind: string, descriptor: object, options: object) {
  queued.set(descriptor, performance.now()); return enqueue.call(this, kind, descriptor, options);
};
const run = prototype.run;
prototype.run = async function (job: { kind: string; label: string; descriptor?: object }) {
  if (job.kind === "manifest") return run.call(this, job);
  const summary = times(this), begin = performance.now();
  const queueWaitMs = begin - (queued.get(job.descriptor!) ?? begin);
  try { return await run.call(this, job); }
  finally {
    const executionMs = performance.now() - begin;
    summary.pipelines++; summary.queueWaitMs += queueWaitMs; summary.executionMs += executionMs;
    summary.slowest.push({ label: job.label, queueWaitMs, executionMs });
    summary.slowest.sort((a, b) => b.executionMs - a.executionMs); summary.slowest.length = Math.min(12, summary.slowest.length);
  }
};
process.once("beforeExit", () => console.error(JSON.stringify({ probe: "cm12-direct-compilation",
  elapsedMilliseconds: performance.now() - started, summaries })));
