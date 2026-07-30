import assert from "node:assert/strict";
import test from "node:test";
import {
  WebGPUOctreePressureHistoryRemap,
  octreePressureHistoryRemapShader,
} from "../lib/webgpu-octree-pressure-history";
import { PassBroker } from "../lib/webgpu-pass-broker";

test("pressure history remap uses the stable new-to-old row identity", () => {
  assert.match(octreePressureHistoryRemapShader,
    /let encoded = rowDelta\[params\.newToOldOffset \+ row\]/);
  assert.match(octreePressureHistoryRemapShader,
    /let value = encoded & 0x3fffffffu;[\s\S]*let oldRow = value - 1u/,
    "affected and structural row bits must not corrupt the predecessor index");
  assert.match(octreePressureHistoryRemapShader,
    /var history = 0\.0/,
    "new rows must make a zero secant rather than inventing old pressure");
  assert.match(octreePressureHistoryRemapShader,
    /history = current - carried/);
  assert.match(octreePressureHistoryRemapShader,
    /candidateHistory\[row\] = history/);
  assert.match(octreePressureHistoryRemapShader, /@workgroup_size\(256\)/,
    "the remap must match the accepted frontier dispatch quantum");
});

test("pressure history remap binds only its compact five-resource ABI", () => {
  Object.assign(globalThis, {
    GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, UNIFORM: 4 },
  });
  const buffer = (size = 4096) => ({ size, destroy() {} }) as GPUBuffer;
  let bindings: number[] = [], dispatch = 0;
  const device = {
    queue: { writeBuffer() {} },
    createBuffer: ({ size }: GPUBufferDescriptor) => buffer(Number(size)),
    createShaderModule: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroup: ({ entries }: GPUBindGroupDescriptor) => {
      bindings = Array.from(entries, (entry) => entry.binding); return {};
    },
  } as unknown as GPUDevice;
  const rowDispatch = buffer(48);
  const remap = new WebGPUOctreePressureHistoryRemap(device, {
    rowDelta: buffer(8192), rowDeltaControlOffsetWords: 16,
    rowDeltaNewToOldOffsetWords: 256, rowDispatch,
    rowDispatchOffsetBytes: 24, currentCandidatePressure: buffer(),
    candidateHistory: buffer(), rowCapacity: 128,
  });
  const encoder = {
    beginComputePass: () => ({
      setPipeline() {}, setBindGroup() {},
      dispatchWorkgroupsIndirect(buffer: GPUBuffer, offset: number) {
        assert.equal(buffer, rowDispatch);
        assert.equal(offset, 24);
        dispatch += 1;
      }, end() {},
    }),
  } as unknown as GPUCommandEncoder;
  remap.encode(new PassBroker(encoder), buffer());
  assert.deepEqual(bindings, [0, 1, 2, 3, 4]);
  assert.equal(dispatch, 1);
  remap.destroy();
});
