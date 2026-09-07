import assert from "node:assert/strict";
import test from "node:test";
import { RetainedDensityResourceBudget, WebGPURetainedDensityField } from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-retained-density";
import { compileBernsteinSupport, positiveBernsteinField } from "../lib/methods/adaptive-mass/sparse-cm12-positive-density-field";

Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { STORAGE: 128, COPY_DST: 8, COPY_SRC: 4 } });
function fakeDevice() {
  const buffers: { size: number; usage: number; destroyed: number; destroy(): void }[] = [];
  let creates = 0, writes = 0, failCreate = -1, failWrite = -1;
  let asyncFailureAt = -1, asyncKind: GPUErrorFilter = "validation", rejectScope = false;
  const scopes: { kind: GPUErrorFilter; error: GPUError | null; reject?: boolean }[] = [];
  const device = {
    limits: { maxStorageBufferBindingSize: 1e7, maxBufferSize: 1e7, maxComputeWorkgroupsPerDimension: 65535 },
    pushErrorScope: (kind: GPUErrorFilter) => { scopes.push({ kind, error: null }); },
    popErrorScope: () => {
      const scope = scopes.pop();
      if (!scope) throw new Error("Unbalanced fake error scope");
      return scope.reject ? Promise.reject(new Error("injected async scope rejection")) : Promise.resolve(scope.error);
    },
    createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
    createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
    createBuffer: (descriptor: GPUBufferDescriptor) => {
      if (++creates === failCreate) throw new Error("injected allocation failure");
      if (creates === asyncFailureAt) {
        const scope = [...scopes].reverse().find(scope => scope.kind === asyncKind)!;
        scope.error = { message: `injected async ${asyncKind}` } as GPUError;
        scope.reject = rejectScope;
      }
      const buffer = { size: descriptor.size, usage: descriptor.usage, destroyed: 0, destroy() { this.destroyed++; } };
      buffers.push(buffer); return buffer;
    },
    queue: { writeBuffer: () => { if (++writes === failWrite) throw new Error("injected upload failure"); } },
  } as unknown as GPUDevice;
  return { device, buffers, scopes,
    failNextAsync: (kind: GPUErrorFilter, distance = 1, reject = false) => { asyncKind = kind; asyncFailureAt = creates + distance; rejectScope = reject; },
    failNextCreate: (distance = 1) => { failCreate = creates + distance; }, failNextWrite: (distance = 1) => { failWrite = writes + distance; } };
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

test("asynchronous scoped creation failures reject readiness and retire invalid resources", async () => {
  for (const kind of ["validation", "out-of-memory"] as const) for (const distance of [1, 2]) {
    const fake = fakeDevice(), budget = new RetainedDensityResourceBudget(1000);
    fake.failNextAsync(kind, distance);
    await assert.rejects(WebGPURetainedDensityField.create(fake.device, field(), { resourceBudget: budget }), /injected async/);
    assert.equal(fake.scopes.length, 0); assert.equal(budget.receipt.liveBytes, 0);
    assert.ok(fake.buffers.every(buffer => buffer.destroyed === 1));
  }
});
test("failed async next generation preserves accepted field and releases reservation", async () => {
  const fake = fakeDevice(), source = field(), budget = new RetainedDensityResourceBudget(1000);
  const old = await WebGPURetainedDensityField.create(fake.device, source, { resourceBudget: budget });
  fake.failNextAsync("out-of-memory");
  const invalid = old.next(positiveBernsteinField(source.support, source.controls, 2));
  assert.equal(fake.scopes.length, 0, "no error scope survives a synchronous allocation call");
  await assert.rejects(invalid.ready(), /injected async/);
  assert.equal(budget.receipt.liveBytes, 236); assert.equal(old.generation, 1);
  assert.throws(() => invalid.compileQueries(queries), /injected async/);
  invalid.release(); old.release(); assert.equal(budget.receipt.liveBytes, 0);
});
test("operation readiness gates encoding and async error receipts rollback only operation buffers", async () => {
  for (const reject of [false, true]) for (const distance of [1, 2]) {
    const fake = fakeDevice(), budget = new RetainedDensityResourceBudget(1000);
    const old = await WebGPURetainedDensityField.create(fake.device, field(), { resourceBudget: budget });
    fake.failNextAsync("validation", distance, reject);
    const operation = old.compileQueries(queries);
    assert.equal(fake.scopes.length, 0);
    assert.throws(() => operation.encode({} as GPUCommandEncoder, {} as GPUBuffer), /not ready/);
    await assert.rejects(operation.ready(), /injected async/);
    assert.equal(budget.receipt.liveBytes, 236); assert.equal(old.generation, 1);
    const valid = old.compileQueries([]); await valid.ready();
    assert.doesNotThrow(() => valid.encode({} as GPUCommandEncoder, { size: 16, usage: 128 } as GPUBuffer));
    valid.release(); operation.release(); old.release();
    assert.equal(budget.receipt.liveBytes, 0); assert.ok(fake.buffers.every(buffer => buffer.destroyed === 1));
  }
});
test("releasing pending allocations is safe when later async receipts fail", async () => {
  const fake = fakeDevice(), budget = new RetainedDensityResourceBudget(1000), source = field();
  const old = await WebGPURetainedDensityField.create(fake.device, source, { resourceBudget: budget });
  fake.failNextAsync("validation");
  const next = old.next(positiveBernsteinField(source.support, source.controls, 2));
  next.release();
  await assert.rejects(next.ready(), /Released/);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(budget.receipt.liveBytes, 236);
  old.release(); assert.ok(fake.buffers.every(buffer => buffer.destroyed === 1));
});
