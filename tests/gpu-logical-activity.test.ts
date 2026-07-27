import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  GPULogicalActivityRecorder,
  GPU_LOGICAL_ACTIVITY_FLAGS,
  GPU_LOGICAL_ACTIVITY_HEADER_WORDS,
  GPU_LOGICAL_ACTIVITY_RECORD_WORDS,
  GPU_LOGICAL_ACTIVITY_STRATIFIED_SAMPLE_COUNT,
  GPU_LOGICAL_ACTIVITY_UNKNOWN_U32,
  binGPULogicalActivity1ms,
  buildGPULogicalActivityShaderVariant,
  createGPULogicalActivityBufferImage,
  decodeGPULogicalActivity,
  gpuLogicalActivityBufferByteSize,
  gpuLogicalActivitySubgroupCallWGSL,
  gpuLogicalActivityStratifiedSampleOrdinals,
  gpuLogicalActivityWGSL,
  gpuLogicalActivityWorkgroupCallWGSL,
  reconstructGPULogicalActivityTicks,
} from "../lib/gpu-logical-activity";

const disabled = { mode: "disabled" } as const;
const workgroup = { mode: "workgroup", group: 3, binding: 0 } as const;
const subgroup = { mode: "subgroup", group: 3, binding: 0 } as const;

test("disabled shader variant is byte-identical and emits zero profiling WGSL", () => {
  const source = "@compute @workgroup_size(1) fn main() {}\n";
  const variant = buildGPULogicalActivityShaderVariant({ source, baseKey: "main", config: disabled });
  assert.equal(variant.code, source);
  assert.equal(variant.instrumentationWGSL, "");
  assert.equal(gpuLogicalActivityWGSL(disabled), "");
  assert.equal(gpuLogicalActivityWorkgroupCallWGSL(disabled, {
    taskId: "1u", checkpointId: "2u", workgroupId: "wid",
    localInvocationIndex: "lane", workgroupLaneCount: "64u",
  }), "");
  assert.equal(gpuLogicalActivitySubgroupCallWGSL(disabled, {
    taskId: "1u", checkpointId: "2u", workgroupId: "wid", subgroupId: "sid",
    subgroupIdEvidence: "measured", subgroupLane: "slane", subgroupSize: "ssize", active: "true",
  }), "");
  assert.doesNotMatch(variant.code, /fluidGpu|LogicalActivity|atomic<|@binding/);
});

test("enabled variants emit bounded atomics, logical helpers, calls, and distinct cache keys", () => {
  const source = "@compute @workgroup_size(64) fn main() {}";
  const workgroupVariant = buildGPULogicalActivityShaderVariant({ source, baseKey: "main", config: workgroup });
  const subgroupVariant = buildGPULogicalActivityShaderVariant({ source, baseKey: "main", config: subgroup });
  assert.match(workgroupVariant.code, /@group\(3\) @binding\(0\)/);
  assert.match(workgroupVariant.code, /atomicAdd/);
  assert.match(workgroupVariant.code, /fluidGpuLogicalActivityStratifiedSampleIndex/);
  assert.match(workgroupVariant.code, /sequence >= fluidGpuLogicalActivity\.capacity/);
  assert.doesNotMatch(workgroupVariant.code, /subgroupBallot/);
  assert.match(subgroupVariant.code, /enable subgroups;/);
  assert.match(subgroupVariant.code, /subgroupBallot\(activePredicate\)/);
  assert.notEqual(workgroupVariant.cacheKey, subgroupVariant.cacheKey);
  assert.notEqual(workgroupVariant.cacheKey,
    buildGPULogicalActivityShaderVariant({ source, baseKey: "main", config: disabled }).cacheKey);
  assert.match(gpuLogicalActivityWorkgroupCallWGSL(workgroup, {
    taskId: "7u", checkpointId: "9u", tick: "tick", workgroupId: "wid",
    localInvocationIndex: "lane", workgroupLaneCount: "64u",
    numWorkgroups: "dispatchSize",
  }), /fluidGpuLogicalActivityWorkgroup\(7u, 9u, tick, wid, dispatchSize, lane, 64u\)/);
  assert.match(gpuLogicalActivitySubgroupCallWGSL(subgroup, {
    taskId: "7u", checkpointId: "9u", workgroupId: "wid", subgroupId: "sid",
    subgroupIdEvidence: "reconstructed", subgroupLane: "slane", subgroupSize: "ssize", active: "eligible",
    numWorkgroups: "dispatchSize",
  }), /wid, dispatchSize, sid, true, slane, ssize, eligible/);
});

test("sixteen midpoint strata cover the whole dispatch without prefix bias", () => {
  assert.deepEqual(gpuLogicalActivityStratifiedSampleOrdinals(1), [0]);
  assert.deepEqual(gpuLogicalActivityStratifiedSampleOrdinals(16),
    Array.from({ length: GPU_LOGICAL_ACTIVITY_STRATIFIED_SAMPLE_COUNT }, (_, index) => index));
  assert.deepEqual(gpuLogicalActivityStratifiedSampleOrdinals(17),
    Array.from({ length: 16 }, (_, index) => index + 1));
  const large = gpuLogicalActivityStratifiedSampleOrdinals(1_000);
  assert.equal(large.length, 16);
  assert.ok(large[0]! > 0, "sampling must not begin at dispatch prefix zero");
  assert.ok(large.at(-1)! > 900, "sampling must reach the final dispatch stratum");
  assert.equal(new Set(large).size, 16);
});

test("instrumentation declarations follow existing WGSL directives", () => {
  const source = `// module policy\nrequires readonly_and_readwrite_storage_textures;\nenable f16;\n\n@compute @workgroup_size(1) fn main() {}`;
  const variant = buildGPULogicalActivityShaderVariant({ source, baseKey: "directives", config: subgroup });
  const subgroupEnable = variant.code.indexOf("enable subgroups;");
  const requires = variant.code.indexOf("requires readonly_and_readwrite_storage_textures;");
  const f16 = variant.code.indexOf("enable f16;");
  const declaration = variant.code.indexOf("const FLUID_GPU_ACTIVITY_UNKNOWN");
  assert.ok(subgroupEnable >= 0 && requires > subgroupEnable && f16 > requires && declaration > f16);
});

function writeRecord(image: Uint32Array, index: number, values: readonly number[]) {
  assert.equal(values.length, GPU_LOGICAL_ACTIVITY_RECORD_WORDS);
  image.set(values, GPU_LOGICAL_ACTIVITY_HEADER_WORDS + index * GPU_LOGICAL_ACTIVITY_RECORD_WORDS);
}

test("CPU decoder preserves measured and reconstructed logical evidence", () => {
  const image = createGPULogicalActivityBufferImage(4, 91);
  image[4] = 2;
  writeRecord(image, 0, [
    0, 10, 20, 3, 1, 2, 3,
    0, 64,
    GPU_LOGICAL_ACTIVITY_UNKNOWN_U32, 0, 64, GPU_LOGICAL_ACTIVITY_UNKNOWN_U32,
    0, 0, 0, 0,
    GPU_LOGICAL_ACTIVITY_FLAGS.workgroupIdPresent
      | GPU_LOGICAL_ACTIVITY_FLAGS.laneIdPresent
      | GPU_LOGICAL_ACTIVITY_FLAGS.laneCountReconstructed
      | GPU_LOGICAL_ACTIVITY_FLAGS.stratifiedSamplePresent,
  ]);
  writeRecord(image, 1, [
    1, 10, 21, 4, 1, 2, 3,
    1, 64,
    2, 0, 32, 3,
    0b1011, 0, 0, 0,
    GPU_LOGICAL_ACTIVITY_FLAGS.workgroupIdPresent
      | GPU_LOGICAL_ACTIVITY_FLAGS.subgroupIdPresent
      | GPU_LOGICAL_ACTIVITY_FLAGS.laneIdPresent
      | GPU_LOGICAL_ACTIVITY_FLAGS.laneCountMeasured
      | GPU_LOGICAL_ACTIVITY_FLAGS.activeMaskMeasured
      | GPU_LOGICAL_ACTIVITY_FLAGS.stratifiedSamplePresent,
  ]);
  const decoded = decodeGPULogicalActivity(image);
  assert.equal(decoded.ok, true);
  if (!decoded.ok) return;
  assert.equal(decoded.capture.captureId, 91);
  assert.equal(decoded.capture.overflowed, false);
  assert.equal(decoded.capture.droppedEventCount, 0);
  assert.equal(decoded.capture.events[0].logicalLaneCountEvidence, "reconstructed");
  assert.equal(decoded.capture.events[0].activeLaneEvidence, "unknown");
  assert.equal(decoded.capture.events[1].subgroupEvidence, "measured");
  assert.equal(decoded.capture.events[1].activeLaneEvidence, "measured");
  assert.deepEqual(decoded.capture.events[1].activeLaneMask, [0b1011, 0, 0, 0]);
  assert.equal(decoded.capture.events[1].activeLaneCount, 3);
  assert.equal(decoded.capture.events[1].sampleCount, 16);
  assert.equal(decoded.capture.events[1].dispatchWorkgroupCount, 64);
  assert.equal(gpuLogicalActivityBufferByteSize(4), image.byteLength);
});

test("overflow preserves its complete prefix and malformed records still fail closed", () => {
  const overflow = createGPULogicalActivityBufferImage(1, 1);
  overflow[4] = 2;
  overflow[5] = 1;
  writeRecord(overflow, 0, [
    0, 1, 2, 0, 0, 0, 0,
    0, 1,
    GPU_LOGICAL_ACTIVITY_UNKNOWN_U32, 0, 32, GPU_LOGICAL_ACTIVITY_UNKNOWN_U32,
    0, 0, 0, 0,
    GPU_LOGICAL_ACTIVITY_FLAGS.workgroupIdPresent
      | GPU_LOGICAL_ACTIVITY_FLAGS.laneIdPresent
      | GPU_LOGICAL_ACTIVITY_FLAGS.laneCountReconstructed
      | GPU_LOGICAL_ACTIVITY_FLAGS.stratifiedSamplePresent,
  ]);
  const truncated = decodeGPULogicalActivity(overflow);
  assert.equal(truncated.ok, true);
  if (truncated.ok) {
    assert.equal(truncated.capture.events.length, 1);
    assert.equal(truncated.capture.overflowed, true);
    assert.equal(truncated.capture.droppedEventCount, 1);
  }

  const malformed = createGPULogicalActivityBufferImage(1, 2);
  malformed[4] = 1;
  writeRecord(malformed, 0, [
    0, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 32, 2,
    1, 0, 0, 0,
    GPU_LOGICAL_ACTIVITY_FLAGS.workgroupIdPresent
      | GPU_LOGICAL_ACTIVITY_FLAGS.subgroupIdPresent
      | GPU_LOGICAL_ACTIVITY_FLAGS.laneIdPresent
      | GPU_LOGICAL_ACTIVITY_FLAGS.laneCountMeasured
      | GPU_LOGICAL_ACTIVITY_FLAGS.activeMaskMeasured
      | GPU_LOGICAL_ACTIVITY_FLAGS.stratifiedSamplePresent,
  ]);
  const result = decodeGPULogicalActivity(malformed);
  assert.equal(result.ok, false);
  if (!result.ok) assert.equal(result.code, "malformed-record");

  const unaligned = decodeGPULogicalActivity(new Uint8Array(new ArrayBuffer(7), 1, 4));
  assert.equal(unaligned.ok, false);
  if (!unaligned.ok) assert.equal(unaligned.code, "bad-size");
});

test("1 ms bins never invent time and retain time/activity evidence", () => {
  const image = createGPULogicalActivityBufferImage(3, 5);
  image[4] = 2;
  const measuredFlags = GPU_LOGICAL_ACTIVITY_FLAGS.workgroupIdPresent
    | GPU_LOGICAL_ACTIVITY_FLAGS.subgroupIdPresent
    | GPU_LOGICAL_ACTIVITY_FLAGS.laneIdPresent
    | GPU_LOGICAL_ACTIVITY_FLAGS.laneCountMeasured
    | GPU_LOGICAL_ACTIVITY_FLAGS.activeMaskMeasured
    | GPU_LOGICAL_ACTIVITY_FLAGS.stratifiedSamplePresent;
  writeRecord(image, 0, [0, 1, 1, 10, 0, 0, 0, 0, 2, 0, 0, 4, 2, 3, 0, 0, 0, measuredFlags]);
  writeRecord(image, 1, [1, 1, 2, 11, 1, 0, 0, 1, 2, 0, 0, 4, 1, 1, 0, 0, 0, measuredFlags]);
  const decoded = decodeGPULogicalActivity(image);
  assert.equal(decoded.ok, true);
  if (!decoded.ok) return;

  const unknown = binGPULogicalActivity1ms(decoded.capture);
  assert.equal(unknown.timeEvidence, "unknown");
  assert.equal(unknown.unknownTimeEventCount, 2);
  assert.deepEqual(unknown.bins, []);

  const reconstructed = binGPULogicalActivity1ms(decoded.capture, reconstructGPULogicalActivityTicks({
    originTick: 10, origin_ms: 4.7, millisecondsPerTick: 0.6,
  }));
  assert.equal(reconstructed.timeEvidence, "reconstructed");
  assert.equal(reconstructed.unknownTimeEventCount, 0);
  assert.deepEqual(reconstructed.bins.map((bin) => [bin.start_ms, bin.eventCount]), [[4, 1], [5, 1]]);
  assert.deepEqual(reconstructed.bins.map((bin) => bin.activeLanes), [
    { sampleCount: 1, sum: 2, maximum: 2, evidence: "measured" },
    { sampleCount: 1, sum: 1, maximum: 1, evidence: "measured" },
  ]);

  const measured = binGPULogicalActivity1ms(decoded.capture, (event) => ({
    time_ms: 8.2 + event.sequence * 0.1, evidence: "measured",
  }));
  assert.equal(measured.timeEvidence, "measured");
  assert.equal(measured.bins[0].timeEvidence, "measured");
  assert.equal(measured.bins[0].eventCount, 2);
});

test("one-shot recorder owns initialization, binding, staging, decode, and destruction", async () => {
  const previousBufferUsage = globalThis.GPUBufferUsage;
  const previousMapMode = globalThis.GPUMapMode;
  Object.assign(globalThis, {
    GPUBufferUsage: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8 },
    GPUMapMode: { READ: 1 },
  });
  type FakeBuffer = GPUBuffer & { bytes: Uint8Array; destroyed: boolean; mapped: boolean };
  const buffers: FakeBuffer[] = [];
  const device = {
    queue: {
      writeBuffer(buffer: FakeBuffer, offset: number, data: ArrayBuffer | ArrayBufferView, dataOffset = 0, size?: number) {
        const bytes = data instanceof ArrayBuffer
          ? new Uint8Array(data, dataOffset, size)
          : new Uint8Array(data.buffer, data.byteOffset + dataOffset, size ?? data.byteLength - dataOffset);
        buffer.bytes.set(bytes, offset);
      },
    },
    createBuffer(descriptor: GPUBufferDescriptor) {
      const bytes = new Uint8Array(Number(descriptor.size));
      const buffer = {
        bytes, destroyed: false, mapped: false,
        get mapState() { return (this as FakeBuffer).mapped ? "mapped" : "unmapped"; },
        async mapAsync(this: FakeBuffer) { this.mapped = true; },
        getMappedRange(offset = 0, size = bytes.byteLength) {
          return bytes.buffer.slice(offset, offset + size);
        },
        unmap(this: FakeBuffer) { this.mapped = false; },
        destroy(this: FakeBuffer) { this.destroyed = true; },
      } as unknown as FakeBuffer;
      buffers.push(buffer);
      return buffer;
    },
    createBindGroup(descriptor: GPUBindGroupDescriptor) { return { descriptor } as unknown as GPUBindGroup; },
  } as unknown as GPUDevice;
  const encoder = {
    copyBufferToBuffer(source: FakeBuffer, sourceOffset: number, destination: FakeBuffer, destinationOffset: number, size: number) {
      destination.bytes.set(source.bytes.subarray(sourceOffset, sourceOffset + size), destinationOffset);
    },
  } as unknown as GPUCommandEncoder;
  try {
    assert.equal(GPULogicalActivityRecorder.create(device, { mode: "disabled", capacity: 2, captureId: 1 }), undefined);
    const recorder = GPULogicalActivityRecorder.create(device, {
      mode: "workgroup", capacity: 2, captureId: 77, label: "test heartbeat",
    });
    assert.ok(recorder);
    const group = recorder.createBindGroup({} as GPUBindGroupLayout, 0) as unknown as { descriptor: GPUBindGroupDescriptor };
    const entries = Array.from(group.descriptor.entries);
    assert.equal(entries[0].binding, 0);
    assert.equal(entries[0].resource instanceof Object, true);
    recorder.finish(encoder);
    assert.equal(recorder.state, "staged");
    const result = await recorder.read();
    assert.equal(result.ok, true);
    if (result.ok) {
      assert.equal(result.capture.captureId, 77);
      assert.deepEqual(result.capture.events, []);
    }
    assert.equal(recorder.state, "complete");
    assert.equal(buffers[1].destroyed, true, "readback is retired after decode");
    recorder.destroy();
    recorder.destroy();
    assert.equal(buffers[0].destroyed, true, "storage destruction is idempotent");
  } finally {
    Object.assign(globalThis, { GPUBufferUsage: previousBufferUsage, GPUMapMode: previousMapMode });
  }
});

const modulePath = process.env.WEBGPU_NODE_MODULE;
test("generated workgroup heartbeat round-trips through WebGPU", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU heartbeat validation",
}, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const config = { mode: "workgroup", group: 3, binding: 0 } as const;
  const source = `${gpuLogicalActivityWGSL(config)}
@compute @workgroup_size(4)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) lane: u32) {
  ${gpuLogicalActivityWorkgroupCallWGSL(config, {
    taskId: "7u", checkpointId: "9u", tick: "workgroupId.x",
    workgroupId: "workgroupId", localInvocationIndex: "lane", workgroupLaneCount: "4u",
  })}
}`;
  const recorder = GPULogicalActivityRecorder.create(device, { mode: "workgroup", capacity: 4, captureId: 123 });
  assert.ok(recorder);
  try {
    const shaderModule = device.createShaderModule({ code: source });
    const info = await shaderModule.getCompilationInfo();
    assert.deepEqual(info.messages.filter(({ type }) => type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: shaderModule, entryPoint: "main" } });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(config.group, recorder.createBindGroupForPipeline(pipeline, config));
    pass.dispatchWorkgroups(2);
    pass.end();
    recorder.finish(encoder);
    device.queue.submit([encoder.finish()]);
    const result = await recorder.read();
    assert.equal(result.ok, true);
    if (result.ok) {
      assert.equal(result.capture.captureId, 123);
      assert.deepEqual(result.capture.events.map((event) => event.workgroupId), [[0, 0, 0], [1, 0, 0]]);
      assert.deepEqual(result.capture.events.map((event) => event.tick), [0, 1]);
    }
  } finally {
    recorder.destroy();
    device.destroy();
  }
});
