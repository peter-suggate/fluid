/** Compile the water shaders and evaluate the actual GGX WGSL on Dawn.
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/check-water-specular.ts
 */
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { unifiedLightingShaderLibrary, WATER_OPTICS } from "../lib/core/webgpu-lighting";
import { causticShader, compositeShader } from "../lib/core/webgpu-water-pipeline";

const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? `${process.cwd()}/node_modules/webgpu/index.js`).href);
Object.assign(globalThis, dawn.globals);
const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
const adapter = await gpu.requestAdapter();
assert.ok(adapter, "Dawn adapter required");
const device = await adapter.requestDevice();
const uncaptured: string[] = [];
device.addEventListener("uncapturederror", event => { event.preventDefault(); uncaptured.push(event.error.message); });
try {
  for (const [label, code] of [["Water caustic projection", causticShader], ["Water optical composite", compositeShader]]) {
    const info = await device.createShaderModule({ label, code }).getCompilationInfo();
    assert.deepEqual(info.messages.filter(message => message.type === "error").map(message => message.message), [], label);
  }
  const module = device.createShaderModule({ code: `${unifiedLightingShaderLibrary}
@group(0) @binding(0) var<storage,read_write> results: array<f32>;
@compute @workgroup_size(1) fn main() {
  let n = vec3f(0,1,0);
  let off = normalize(vec3f(0.1,1,0));
  let f0 = ${WATER_OPTICS.fresnelF0};
  results[0] = unifiedWaterSpecular(n,n,n,0.04,f0);
  results[1] = unifiedWaterSpecular(n,n,n,0.12,f0);
  results[2] = unifiedWaterSpecular(n,n,n,0.3,f0);
  results[3] = unifiedWaterSpecular(n,n,off,0.12,f0);
  results[4] = unifiedWaterSpecular(n,n,off,0.3,f0);
  results[5] = unifiedWaterSpecular(n,n,-n,0.12,f0);
  results[6] = unifiedWaterSpecular(n,-n,n,0.12,f0);
  let v = normalize(vec3f(0.5,1,0));
  let l = normalize(vec3f(-0.3,1,0));
  results[7] = unifiedWaterSpecular(n,v,l,0.12,f0) / dot(n,l);
  results[8] = unifiedWaterSpecular(n,l,v,0.12,f0) / dot(n,v);
  let grazingV = normalize(vec3f(1,0.0001,0));
  let grazingL = normalize(vec3f(-1,0.0001,0));
  results[9] = unifiedWaterSpecular(n,grazingV,grazingL,0.04,f0);
  results[10] = unifiedWaterSpecular(n,n,n,0.0,f0);
}` });
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
  const size = 11 * 4;
  const output = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, size);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const results = new Float32Array(readback.getMappedRange()).slice();
  readback.unmap(); output.destroy(); readback.destroy();
  assert.ok([...results].every(value => Number.isFinite(value) && value >= 0), "all outputs finite, including grazing and minimum roughness");
  for (const [index, roughness] of [0.04, 0.12, 0.3].entries()) {
    const expected = WATER_OPTICS.fresnelF0 / (4 * Math.PI * roughness ** 4);
    assert.ok(Math.abs(results[index] / expected - 1) < 1e-4, `roughness ${roughness}: normalized GGX peak`);
  }
  assert.ok(results[1] > results[2], "lower roughness concentrates energy into a higher peak");
  assert.ok(results[3] / results[1] < results[4] / results[2], "lower roughness has a narrower angular highlight");
  assert.equal(results[5], 0, "no highlight for light below the surface");
  assert.equal(results[6], 0, "no highlight for viewer below the surface");
  assert.ok(Math.abs(results[7] / results[8] - 1) < 1e-5, "BRDF is reciprocal");
  assert.equal(results[10], results[0], "zero roughness is safely bounded");
  assert.deepEqual(uncaptured, []);
  console.log("Water shaders compile; GGX normalization, angular width, reciprocity and finite limits pass.");
  console.log(JSON.stringify({ peakAtDefault: results[1], peakAtRoughness03: results[2] }));
} finally {
  device.destroy();
}
