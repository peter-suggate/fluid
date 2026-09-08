/** Node-only canonical-lane preload. Counts direct compilation separately
 * from manifest-cache statistics, with bounded output and weak device keys. */
import "./probe-sparse-cm12-generation-host";
import { appendFileSync } from "node:fs";
import { GPUCompilationManager } from "../lib/core/gpu-compilation-manager";

interface CompilationTimes {
  manager: number; modules: number; sourceCodeUnits: number; moduleCallMs: number;
  pipelines: number; queueWaitMs: number; executionMs: number;
  active: { label: string; started: number; queueWaitMs: number }[];
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
      moduleCallMs: 0, pipelines: 0, queueWaitMs: 0, executionMs: 0, active: [], slowest: [] };
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
  const active = { label: job.label, started: begin, queueWaitMs }; summary.active.push(active);
  try { return await run.call(this, job); }
  finally {
    const executionMs = performance.now() - begin;
    summary.active.splice(summary.active.indexOf(active), 1);
    summary.pipelines++; summary.queueWaitMs += queueWaitMs; summary.executionMs += executionMs;
    summary.slowest.push({ label: job.label, queueWaitMs, executionMs });
    summary.slowest.sort((a, b) => b.executionMs - a.executionMs); summary.slowest.length = Math.min(12, summary.slowest.length);
  }
};
const output = `/tmp/fluid-cm12-compilation-${process.pid}.jsonl`;
function report(phase: string) {
  if (summaries.length === 0) return;
  const now = performance.now();
  const line = JSON.stringify({ probe: "cm12-direct-compilation", phase, pid: process.pid,
    arguments: process.argv.slice(1), output, elapsedMilliseconds: now - started,
    summaries: summaries.map(({ active, ...summary }) => ({ ...summary,
      active: active.map(({ started: begin, ...entry }) => ({ ...entry, executionMs: now - begin })) })) });
  appendFileSync(output, `${line}\n`); console.error(line);
}
// A canonical timeout terminates the child before beforeExit. Persist a
// bounded snapshot while the driver runs, without extending its deadline or
// installing a signal handler that changes ordinary termination semantics.
let snapshots = 0;
const timer = setInterval(() => { report("running"); if (++snapshots >= 12) clearInterval(timer); }, 5000);
timer.unref();
process.once("beforeExit", () => report("complete"));
