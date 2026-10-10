import assert from "node:assert/strict";
import test from "node:test";
import { createUniformMixedLayout } from "../lib/methods/uniform/uniform-mixed-layout";
import { UniformMixedPressureCycles } from "../lib/methods/uniform/uniform-mixed-pressure-cycles";

async function pressureFixture(n: number, listed: boolean) {
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8 }, GPUShaderStage: { COMPUTE: 4 } });
  const codes: string[] = [], pipelines: GPUComputePipelineDescriptor[] = [], offsets: number[] = [];
  let writes = 0;
  const device = {
    limits: { maxComputeInvocationsPerWorkgroup: 1024, maxComputeWorkgroupsPerDimension: 65535, minUniformBufferOffsetAlignment: 256 },
    createBuffer: (d: GPUBufferDescriptor) => ({ size: d.size, destroy() {} }),
    createBindGroupLayout: (d: unknown) => d, createBindGroup: (d: unknown) => d, createPipelineLayout: (d: unknown) => d,
    createShaderModule: (d: GPUShaderModuleDescriptor) => { codes.push(d.code); return { getCompilationInfo: async () => ({ messages: [] }) }; },
    createComputePipelineAsync: async (d: GPUComputePipelineDescriptor) => { pipelines.push(d); return d; },
    queue: { writeBuffer() { writes++; } },
  } as unknown as GPUDevice;
  const lattice = { dimensions: [n, n, n] as const, cellSize_m: [1 / n, 2 / n, 3 / n] as const, origin_m: { x: 0, y: 0, z: 0 } };
  const layout = createUniformMixedLayout(lattice, [], 4), arena = { size: 32 * 1024 * 1024 } as GPUBuffer;
  let at = 0;
  const field = (d: readonly number[], components = 1) => {
    const size = 4 * components * d.reduce((a, b) => a * b, 1), offset = at;
    at += Math.ceil(size / 256) * 256;
    return { dimensions: d, buffer: { buffer: arena, offset, size } };
  };
  const levels = [];
  for (let t = n / 4; ; t = Math.ceil(t / 2)) {
    const dimensions = [t + 2, t + 2, t + 2];
    levels.push({ dimensions, pressure: [field(dimensions), field(dimensions)], rhs: [field(dimensions), field(dimensions)],
      minimum: [field(dimensions), field(dimensions)], coefficients: field(dimensions, 4) });
    if (t === 1) break;
  }
  const root = levels[0]!, phi = field(root.dimensions), topology = field(root.dimensions, 4);
  const count = layout.tiles.length + 6 * (n / 4) ** 2;
  at = 16 * 1024 * 1024;
  const mixed = () => { const offset = at, size = 4 * count; at += Math.ceil(size / 256) * 256; return { buffer: arena, offset, size }; };
  const ownership = { layout, bindLayout: {}, bindGroup: {} };
  const cycle = listed ? { list: { size: 4096 }, tiles: root.dimensions.reduce((a, b) => a * Math.ceil(b / 4), 1), pressure: root.pressure[1] } : undefined;
  const native = { levels, pressure: root.pressure[0], rhs: root.rhs[0], minimum: root.minimum[0], phi, topology, cycle, encodeSetup() {} };
  const solver = new UniformMixedPressureCycles(device, { ownership, pressure: mixed(), rhs: [mixed(), mixed()],
    minimum: [mixed(), mixed()], frozen: mixed(), residual: mixed(), phi: mixed() } as any, mixed(), native as any, {} as GPUBindGroup, n > 32);
  await solver.initialize();
  const initialWrites = writes, initialPipelines = pipelines.length;
  const pass = { setPipeline() {}, setBindGroup(group: number, _binding: unknown, dynamic?: number[]) { if (group === 1) offsets.push(dynamic![0]!); }, dispatchWorkgroups() {}, end() {} };
  const encoder = { beginComputePass: () => pass, clearBuffer() {} } as unknown as GPUCommandEncoder;
  solver.encodeSetup(encoder); solver.encodeFullCycle(encoder); solver.encodeVCycle(encoder); solver.encodeFullCycle(encoder, true);
  assert.equal(writes, initialWrites, "encoded cycles cannot overwrite earlier dispatch parameters");
  assert.equal(pipelines.length, initialPipelines, "advancing pressure compiles no pipelines");
  assert.ok(offsets.length > 20 && new Set(offsets).size > 10, "traversals select immutable records within the shared kernels");
  assert.ok(offsets.every(offset => offset % 256 === 0));
  solver.destroy();
  return { code: codes[0], kernels: pipelines.map(p => [p.compute.entryPoint, p.compute.constants]).sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b))) };
}

test("pressure kernels are reusable across tank dimensions, hierarchy depths and list modes", async () => {
  const small = await pressureFixture(32, false), large = await pressureFixture(128, true);
  assert.equal(small.code, large.code, "tank geometry and hierarchy tables must not change shader text");
  assert.deepEqual(small.kernels, large.kernels);
  assert.equal(small.kernels.length, 8, "levels and traversal parities share eight installed kernels");
});
