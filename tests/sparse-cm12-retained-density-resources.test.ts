import assert from "node:assert/strict";
import test from "node:test";
import { RetainedDensityResourceBudget, WebGPURetainedDensityField } from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-retained-density";
import { compileBernsteinSupport, positiveBernsteinField } from "../lib/methods/adaptive-mass/sparse-cm12-positive-density-field";

Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { STORAGE: 128, COPY_DST: 8, COPY_SRC: 4 } });
function fakeDevice() {
  const buffers: { size: number; usage: number; destroyed: number; destroy(): void }[] = [];
  let creates = 0, writes = 0, failCreate = -1, failWrite = -1;
  const device = {
    limits: { maxStorageBufferBindingSize: 1e7, maxBufferSize: 1e7, maxComputeWorkgroupsPerDimension: 65535 },
    createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
    createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
    createBuffer: (descriptor: GPUBufferDescriptor) => {
      if (++creates === failCreate) throw new Error("injected allocation failure");
      const buffer = { size: descriptor.size, usage: descriptor.usage, destroyed: 0, destroy() { this.destroyed++; } };
      buffers.push(buffer); return buffer;
    },
    queue: { writeBuffer: () => { if (++writes === failWrite) throw new Error("injected upload failure"); } },
  } as unknown as GPUDevice;
  return { device, buffers, failNextCreate: (distance = 1) => { failCreate = creates + distance; }, failNextWrite: (distance = 1) => { failWrite = writes + distance; } };
}
function field(generation = 1) {
  const support = compileBernsteinSupport([{ lower: [0, 0, 0], width: 1 }]);
  return positiveBernsteinField(support, new Float64Array(27).fill(0.5), generation);
}
const queries = [{ cell: 0, point: [0.3, 0.4, 0.5] as const }];
test("leases, simultaneous generations and operation images share one bounded accounting owner", async () => {
  const fake = fakeDevice(), source = field(), budget = new RetainedDensityResourceBudget(1000);
  const gpu = await WebGPURetainedDensityField.create(fake.device, source, { resourceBudget: budget });
  assert.equal(budget.receipt.liveBytes, 236); assert.equal(budget.receipt.liveBuffers, 2);
  const lease = gpu.retain();
  assert.equal(budget.receipt.liveBytes, 236);
  const next = gpu.next(positiveBernsteinField(source.support, source.controls, 2));
  assert.equal(budget.receipt.liveBytes, 344);
  const operation = gpu.compileQueries(queries);
  assert.equal(budget.receipt.liveBytes, 380);
  gpu.release(); lease.release();
  assert.equal(budget.receipt.liveBytes, 380, "operation leases its original coefficient generation");
  operation.release(); operation.release();
  assert.equal(budget.receipt.liveBytes, 236);
  next.release(); next.release();
  assert.equal(budget.receipt.liveBytes, 0); assert.equal(budget.receipt.liveBuffers, 0);
  assert.equal(budget.receipt.peakBytes, 380);
  assert.ok(fake.buffers.every(buffer => buffer.destroyed === 1));
  assert.throws(() => gpu.retain(), /Released/);
});
test("budget exhaustion rolls back a partially reserved operation and preserves accepted field", async () => {
  const fake = fakeDevice(), source = field(), budget = new RetainedDensityResourceBudget(268);
  const gpu = await WebGPURetainedDensityField.create(fake.device, source, { resourceBudget: budget });
  assert.throws(() => gpu.compileQueries(queries), /budget exceeded/);
  assert.equal(budget.receipt.liveBytes, 236);
  assert.equal(fake.buffers[2]!.destroyed, 1);
  assert.throws(() => gpu.next(positiveBernsteinField(source.support, source.controls, 2)), /budget exceeded/);
  assert.equal(gpu.generation, 1);
  assert.equal(fake.buffers[0]!.destroyed, 0); assert.equal(fake.buffers[1]!.destroyed, 0);
  const empty = gpu.compileQueries([]); empty.release();
  gpu.release(); assert.equal(budget.receipt.liveBytes, 0);
});
test("allocation and upload exceptions release only newly allocated resources", async () => {
  for (const stage of ["create", "write"] as const) for (const distance of [1, 2]) {
    const fake = fakeDevice(), budget = new RetainedDensityResourceBudget(1000);
    const gpu = await WebGPURetainedDensityField.create(fake.device, field(), { resourceBudget: budget });
    if (stage === "create") fake.failNextCreate(distance); else fake.failNextWrite(distance);
    assert.throws(() => gpu.compileQueries(queries), /injected/);
    assert.equal(budget.receipt.liveBytes, 236); assert.equal(budget.receipt.liveBuffers, 2);
    assert.equal(fake.buffers[0]!.destroyed, 0); assert.equal(fake.buffers[1]!.destroyed, 0);
    assert.ok(fake.buffers.slice(2).every(buffer => buffer.destroyed === 1));
    const retry = gpu.compileQueries(queries); retry.release(); gpu.release();
    assert.equal(budget.receipt.liveBytes, 0);
  }
});
test("failed first generation and failed replacement support leave shared budget and old generation valid", async () => {
  for (const stage of ["create", "write"] as const) {
    const fake = fakeDevice(), budget = new RetainedDensityResourceBudget(1000);
    if (stage === "create") fake.failNextCreate(2); else fake.failNextWrite(2);
    await assert.rejects(WebGPURetainedDensityField.create(fake.device, field(), { resourceBudget: budget }), /injected/);
    assert.equal(budget.receipt.liveBytes, 0); assert.ok(fake.buffers.every(buffer => buffer.destroyed === 1));
  }
  const fake = fakeDevice(), budget = new RetainedDensityResourceBudget(300);
  const old = await WebGPURetainedDensityField.create(fake.device, field(), { resourceBudget: budget });
  await assert.rejects(WebGPURetainedDensityField.create(fake.device, field(2), { resourceBudget: budget }), /budget/);
  assert.equal(old.generation, 1); assert.equal(budget.receipt.liveBytes, 236);
  old.release(); assert.equal(budget.receipt.liveBytes, 0);
});
test("malformed support cardinality is rejected before any GPU allocation", async () => {
  const fake = fakeDevice(), original = field();
  await assert.rejects(WebGPURetainedDensityField.create(fake.device, { ...original,
    support: { ...original.support, cellControls: [] } }), /cardinality/);
  assert.equal(fake.buffers.length, 0);
});
test("failed coefficient generation upload preserves old leases and permits a retry", async () => {
  const fake = fakeDevice(), source = field(), budget = new RetainedDensityResourceBudget(1000);
  const old = await WebGPURetainedDensityField.create(fake.device, source, { resourceBudget: budget });
  const replacement = positiveBernsteinField(source.support, source.controls, 2);
  fake.failNextWrite();
  assert.throws(() => old.next(replacement), /upload failure/);
  assert.equal(budget.receipt.liveBytes, 236); assert.equal(old.generation, 1);
  assert.equal(fake.buffers[2]!.destroyed, 1);
  const next = old.next(replacement);
  old.release(); assert.equal(next.generation, 2);
  next.release(); assert.equal(budget.receipt.liveBytes, 0);
  assert.ok(fake.buffers.every(buffer => buffer.destroyed === 1));
});
