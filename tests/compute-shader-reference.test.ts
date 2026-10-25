import assert from "node:assert/strict";
import test from "node:test";
import { computeShaderReference } from "../tools/compute-shader-reference";

test("compute comparison substitutes only selected pipelines and retains dispatches", async () => {
  const compiled: GPUComputePipelineDescriptor[] = [], calls: unknown[] = [];
  const device = {
    createShaderModule: (descriptor: GPUShaderModuleDescriptor) => ({ ...descriptor }),
    async createComputePipelineAsync(descriptor: GPUComputePipelineDescriptor) { compiled.push(descriptor);return { label: descriptor.label }; },
  } as unknown as GPUDevice;
  const comparison = computeShaderReference(device, { pipelines: ["AO"], replacements: [{ search: "candidate", replacement: "reference" }], omitPasses: ["candidate-only"] });
  const module = comparison.device.createShaderModule({ code: "candidate" });
  const ao = await comparison.device.createComputePipelineAsync({ label: "AO", layout: "auto", compute: { module, entryPoint: "main" } });
  const other = await comparison.device.createComputePipelineAsync({ label: "other", layout: "auto", compute: { module, entryPoint: "main" } });
  const encoder = {
    beginComputePass(descriptor: GPUComputePassDescriptor) { calls.push(descriptor);return {
      setPipeline(pipeline: GPUComputePipeline) { calls.push(pipeline.label); },
      dispatchWorkgroups(...counts: number[]) { calls.push(counts); }, end() { calls.push("end"); },
    }; },
    clearBuffer() { calls.push("clear"); },
  } as unknown as GPUCommandEncoder;
  const wrapped = comparison.encode(encoder);
  const pass = wrapped.beginComputePass({ label: "production pass" });
  pass.setPipeline(ao);pass.dispatchWorkgroups(3,4);pass.setPipeline(other);pass.end();wrapped.clearBuffer({} as GPUBuffer);
  const omitted = wrapped.beginComputePass({ label: "candidate-only" });
  omitted.setPipeline(ao);omitted.dispatchWorkgroups(100);omitted.end();
  assert.deepEqual(calls, [{ label: "production pass" }, "AO reference", [3,4], "other", "end", "clear"]);
  assert.equal((compiled[1]!.compute.module as unknown as GPUShaderModuleDescriptor).code, "reference");
  assert.equal((compiled[2]!.compute.module as unknown as GPUShaderModuleDescriptor).code, "candidate");
  assert.equal(ao.label, "AO", "production pipeline remains available for the candidate arm");
});

test("compute comparison rejects stale source patterns and missing pipeline labels", async () => {
  const device = {
    createShaderModule: (descriptor: GPUShaderModuleDescriptor) => ({ ...descriptor }),
    async createComputePipelineAsync(descriptor: GPUComputePipelineDescriptor) { return { label: descriptor.label }; },
  } as unknown as GPUDevice;
  const missing = computeShaderReference(device, { pipelines: ["AO"], replacements: [{ search: "old", replacement: "ref" }] });
  const module = missing.device.createShaderModule({ code: "new" });
  await assert.rejects(missing.device.createComputePipelineAsync({ label: "AO", layout: "auto", compute: { module, entryPoint: "main" } }), /No reference substitution/);
  const partial = computeShaderReference(device, { pipelines: [], replacements: [{ search: "new", replacement: "ref" }, { search: "missing", replacement: "ref" }] });
  partial.device.createShaderModule({ code: "new" });
  assert.throws(() => partial.encode({} as GPUCommandEncoder), /search did not match/);
  const absent = computeShaderReference(device, { pipelines: ["absent"], replacements: [] });
  assert.throws(() => absent.encode({} as GPUCommandEncoder), /not compiled/);
});
