import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { SPARSE_CM12_PRESSURE_ROW_GRADIENT_WGSL } from
  "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl";

const f = Math.fround;
const add = (a: number, b: number) => f(f(a) + f(b));
const mul = (a: number, b: number) => f(f(a) * f(b));
const bits = (value: number) => new Uint32Array(new Float32Array([value]).buffer)[0];
const fine = [-22783.525390625, -0.0034095989540219307,
  25331.087890625, 0.00018020442803390324];
const permutations = [[0, 1, 2, 3], [1, 0, 3, 2],
  [2, 3, 0, 1], [3, 2, 1, 0]];

const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;
dawnTest("production pressure gradient pairs a 2:1 fine quad under every reflection", async () => {
  const pressures: number[] = [], coefficients: number[] = [], members: number[] = [];
  const expected: number[] = [];
  // Each axis uses a different sparse membership pattern. Permuting values and
  // membership together models both tangential reflections around that axis.
  const masks = [[1, 1, 1, 1], [1, 0, 1, 1], [0, 1, 1, 0]];
  for (let axis = 0; axis < 3; axis++) for (const sign of [-1, 1]) {
    for (const permutation of permutations) {
      const side = permutation.map(i => fine[i]);
      const mask = permutation.map(i => masks[axis][i]);
      for (let i = 0; i < 4; i++) {
        pressures.push(side[i]); coefficients.push(f(-0.25 * sign)); members.push(mask[i]);
      }
      pressures.push(321.25); coefficients.push(sign); members.push(1);
      const admitted = side.map((value, i) => mask[i] ? value : 0);
      const quad = add(add(admitted[0], admitted[1]), add(admitted[2], admitted[3]));
      expected.push(add(mul(-0.25 * sign, quad), mul(sign, 321.25)));
    }
  }
  const oldSerial = permutations.map(permutation => {
    let sum = 0;
    for (const i of permutation) sum = add(sum, mul(-0.25, fine[i]));
    return add(sum, 321.25);
  });
  assert.ok(new Set(oldSerial.map(bits)).size > 1,
    "fixture must expose the former term-order dependence");

  await acquireWebGPUExclusiveLock("dawn-test", "sparse-cm12-pressure-gradient-symmetry");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  const buffers: GPUBuffer[] = [];
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    assert.ok(gpu);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const module = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read> state:array<f32>;
@group(0) @binding(1) var<storage,read> coefficients:array<f32>;
@group(0) @binding(2) var<storage,read> members:array<u32>;
@group(0) @binding(3) var<storage,read_write> output:array<f32>;
fn rowTermRange(row:u32)->vec2u{return vec2u(row*5u,row*5u+5u);}
fn termCoefficient(term:u32)->f32{return coefficients[term];}
fn termCell(term:u32)->u32{return term;}
fn peiPressureCellMember(cell:u32)->bool{return members[cell]!=0u;}
${SPARSE_CM12_PRESSURE_ROW_GRADIENT_WGSL}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3u){
  if(gid.x<${expected.length}u){output[gid.x]=pressureRowGradient(gid.x,0u);}
}` });
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto",
      compute: { module, entryPoint: "main" } });
    const make = (data: ArrayBufferView, usage: GPUBufferUsageFlags) => {
      const buffer = device!.createBuffer({ size: data.byteLength, usage, mappedAtCreation: true });
      new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      buffer.unmap(); buffers.push(buffer); return buffer;
    };
    const pressureBuffer = make(new Float32Array(pressures), GPUBufferUsage.STORAGE);
    const coefficientBuffer = make(new Float32Array(coefficients), GPUBufferUsage.STORAGE);
    const memberBuffer = make(new Uint32Array(members), GPUBufferUsage.STORAGE);
    const output = device.createBuffer({ size: expected.length * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }); buffers.push(output);
    const readback = device.createBuffer({ size: expected.length * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); buffers.push(readback);
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: pressureBuffer } },
      { binding: 1, resource: { buffer: coefficientBuffer } },
      { binding: 2, resource: { buffer: memberBuffer } },
      { binding: 3, resource: { buffer: output } },
    ] });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, expected.length * 4);
    device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
    const actual = Array.from(new Float32Array(readback.getMappedRange()));
    assert.deepEqual(actual.map(bits), expected.map(bits));
    for (let row = 0; row < expected.length; row += 4) {
      assert.equal(new Set(actual.slice(row, row + 4).map(bits)).size, 1);
    }
    readback.unmap();
  } finally {
    for (const buffer of buffers) buffer.destroy();
    device?.destroy(); await releaseWebGPUExclusiveLock(); assert.ok(gpu);
  }
});
