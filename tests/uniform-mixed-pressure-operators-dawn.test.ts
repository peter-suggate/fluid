import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { uniformMixedPressureReconstructionWGSL } from "../lib/methods/uniform/uniform-mixed-pressure-reconstruction.wgsl";
import { uniformMixedPressureOperatorWGSL } from "../lib/methods/uniform/uniform-mixed-pressure-operator.wgsl";
import { uniformMixedPressureTransferWGSL } from "../lib/methods/uniform/uniform-mixed-pressure-transfer.wgsl";
import { mixedPressureLayouts } from "./helpers/uniform-mixed-pressure";
import { MixedMultigridOracle } from "./helpers/uniform-mixed-multigrid";
import { seamLayout } from "./helpers/uniform-geometric-seam";

const modulePath = process.env.WEBGPU_NODE_MODULE;
async function checkedModule(device: GPUDevice, code: string): Promise<GPUShaderModule> {
  const module = device.createShaderModule({ code }), info = await module.getCompilationInfo();
  assert.deepEqual(info.messages.filter(m => m.type === "error").map(m => `${m.lineNum}: ${m.message}`), []);
  return module;
}
const maxError = (actual: ArrayLike<number>, expected: ArrayLike<number>) => Math.max(...Array.from({ length: expected.length }, (_, i) => Math.abs(actual[i]! - expected[i]!)));

(modulePath ? test : test.skip)("mixed pressure residual, coloured sweep and hierarchy transfers match the independent oracle on GPU", { timeout: 180000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "Uniform mixed pressure operators");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter();
    assert.ok(adapter); device = await adapter.requestDevice();
    const errors: string[] = []; device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    for (const layout of [...mixedPressureLayouts(), seamLayout(0, "fine"), seamLayout(0, "coarse")]) {
      const oracle = new MixedMultigridOracle(layout);
      const ownership = oracle.layouts.map(l => new UniformMixedOwnership(device!, l));
      try {
        for (const level of [0, 1]) {
          const owner = ownership[level]!, fine = oracle.fixtures[level]!, n = fine.cells.length;
          const buffers: GPUBuffer[] = [];
          const buffer = (size: number, usage: GPUBufferUsageFlags) => {
            const b = device!.createBuffer({ size, usage }); buffers.push(b); return b;
          };
          const pressure = buffer(n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
          const slopes = buffer(n * 16, GPUBufferUsage.STORAGE);
          const rhs = buffer(n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
          const frozen = buffer(n * 4, GPUBufferUsage.STORAGE);
          const output = buffer(n * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          const readback: GPUBuffer = buffer(n * 20, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
          try {
            const p = Float32Array.from(fine.cells, (_, i) => Math.sin(i * 7));
            const b = Float32Array.from(fine.cells, (_, i) => Math.cos(i * 13));
            device.queue.writeBuffer(pressure, 0, p); device.queue.writeBuffer(rhs, 0, b);
            const module = await checkedModule(device, uniformMixedTopologyWGSL(oracle.layouts[level]!, 0) + /* wgsl */ `
const UM_H=vec3f(${layout.lattice.cellSize_m.map(n => n.toFixed(8)).join(",")});
override umPressureParity:u32=0u;
@group(1) @binding(0) var<storage,read_write> pressures:array<f32>;
@group(1) @binding(1) var<storage,read_write> slopes:array<vec4f>;
@group(1) @binding(2) var<storage,read> rhs:array<f32>;
@group(1) @binding(3) var<storage,read_write> frozen:array<f32>;
@group(1) @binding(4) var<storage,read_write> result:array<vec4f>;
fn umPressure(o:UMOwner)->f32{return pressures[o.index];}
fn umPressureSlope(o:UMOwner)->vec3f{return slopes[o.index].xyz;}
${uniformMixedPressureReconstructionWGSL}
${uniformMixedPressureOperatorWGSL}
@compute @workgroup_size(64) fn reconstruct(@builtin(global_invocation_id) gid:vec3u){
 let o=umOwner(gid);if(o.width==0u){return;}slopes[o.index]=vec4f(umReconstructPressureSlope(o),0.0);
}
@compute @workgroup_size(64) fn freeze(@builtin(global_invocation_id) gid:vec3u){
 let o=umOwner(gid);if(o.width==0u){return;}
 let core=umPressureCoreTerms(o);let corrected=umPressureCorrectedRhs(o,rhs[o.index]);
 result[o.index]=vec4f(umPressureApply(o),corrected,core.x,f32(umPressureColour(o)));
 frozen[o.index]=corrected;
}
@compute @workgroup_size(64) fn smoothOwners(@builtin(global_invocation_id) gid:vec3u){
 let o=umOwner(gid);if(o.width==0u||(umPressureColour(o)&1u)!=umPressureParity){return;}
 let core=umPressureCoreTerms(o);pressures[o.index]=select(0.0,(frozen[o.index]+core.y)/core.x,core.x>0.0);
}
`);
            const resources = device.createBindGroupLayout({ entries: [0, 1, 2, 3, 4].map(binding => ({ binding,
              visibility: GPUShaderStage.COMPUTE, buffer: { type: binding === 2 ? "read-only-storage" as const : "storage" as const } })) });
            const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [owner.bindLayout, resources] });
            const compile = (entryPoint: string, width: number, parity = 0) => device!.createComputePipelineAsync({ layout: pipelineLayout,
              compute: { module, entryPoint, constants: { umCellWidth: width, umDispatchX: owner.dispatchX, umPressureParity: parity } } });
            const [reconstruct, freeze, smooth] = await Promise.all([
              Promise.all([1, 2, 4].map(w => compile("reconstruct", w))), Promise.all([1, 2, 4].map(w => compile("freeze", w))),
              Promise.all([0, 1, 2, 3, 4, 5].map(c => compile("smoothOwners", 1 << (c >> 1), c & 1))),
            ]);
            const group = device.createBindGroup({ layout: resources, entries: [pressure, slopes, rhs, frozen, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
            const encoder = device.createCommandEncoder();
            for (const stage of [reconstruct, freeze]) {
              const pass = encoder.beginComputePass(); pass.setBindGroup(0, owner.bindGroup); pass.setBindGroup(1, group); owner.dispatch(pass, stage); pass.end();
            }
            for (let colour = 0; colour < 6; colour++) {
              const pass = encoder.beginComputePass(); pass.setBindGroup(0, owner.bindGroup); pass.setBindGroup(1, group);
              owner.dispatchTier(pass, smooth[colour]!, (colour >> 1) as 0 | 1 | 2); pass.end();
            }
            encoder.copyBufferToBuffer(output, 0, readback, 0, n * 16);
            encoder.copyBufferToBuffer(pressure, 0, readback, n * 16, n * 4);
            device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
            const values: Float32Array = new Float32Array(readback.getMappedRange());
            const applied = oracle.apply(level, p).map((v, i) => v / fine.cells[i]!.volume);
            assert.ok(maxError(Float32Array.from(fine.cells, (_, i) => values[4 * i]!), applied) < 1e-5);
            const colours = fine.cells.map((_, i) => values[4 * i + 3]!);
            for (const f of fine.faces) assert.notEqual(colours[f.left], colours[f.right]);
            const seams = new Set(fine.faces.filter(f => fine.cells[f.left]!.width !== fine.cells[f.right]!.width).flatMap(f => [f.left, f.right]));
            fine.cells.forEach((_, i) => { if (!seams.has(i)) assert.equal(values[4 * i + 1], b[i], "uniform rows retain their exact RHS"); });
            const swept = oracle.smoothSweep(level, p, Float64Array.from(b, (v, i) => v * fine.cells[i]!.volume));
            assert.ok(maxError(values.subarray(n * 4), swept) < 1e-5);
            readback.unmap();
          } finally { if (readback.mapState === "mapped") readback.unmap(); buffers.forEach(b => b.destroy()); }
          await checkTransfers(device, oracle, ownership, level);
        }
      } finally { ownership.forEach(o => o.destroy()); }
    }
    assert.deepEqual(errors, []);
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});

async function checkTransfers(device: GPUDevice, oracle: MixedMultigridOracle, ownership: readonly UniformMixedOwnership[], level: number) {
  const fine = ownership[level]!, coarse = ownership[level + 1]!, nf = fine.layout.cellCount, nc = coarse.layout.cellCount;
  const buffers: GPUBuffer[] = [];
  const buffer = (size: number, usage: GPUBufferUsageFlags) => { const b = device.createBuffer({ size, usage }); buffers.push(b); return b; };
  const fineIn = buffer(nf * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const coarseIn = buffer(nc * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const fineOut = buffer(nf * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const coarseOut = buffer(nc * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const readback: GPUBuffer = buffer((nf + nc) * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  try {
    const f = Float32Array.from({ length: nf }, (_, i) => Math.sin(i * 7)), c = Float32Array.from({ length: nc }, (_, i) => Math.cos(i * 13));
    device.queue.writeBuffer(fineIn, 0, f); device.queue.writeBuffer(coarseIn, 0, c);
    const module = await checkedModule(device, uniformMixedTopologyWGSL(fine.layout, 0, "fine") + uniformMixedTopologyWGSL(coarse.layout, 1, "coarse") + /* wgsl */ `
@group(2) @binding(0) var<storage,read> fineValues:array<f32>;
@group(2) @binding(1) var<storage,read> coarseValues:array<f32>;
@group(2) @binding(2) var<storage,read_write> prolonged:array<f32>;
@group(2) @binding(3) var<storage,read_write> restricted:array<f32>;
fn umFineResidual(o:fineUMOwner)->f32{return fineValues[o.index];}
fn umCoarsePressure(o:coarseUMOwner)->f32{return coarseValues[o.index];}
${uniformMixedPressureTransferWGSL}
@compute @workgroup_size(64) fn restrictValues(@builtin(global_invocation_id) gid:vec3u){
 let o=coarseumOwner(gid);if(o.width!=0u){restricted[o.index]=umRestrictPressureResidual(o);}
}
@compute @workgroup_size(64) fn prolongValues(@builtin(global_invocation_id) gid:vec3u){
 let o=fineumOwner(gid);if(o.width!=0u){prolonged[o.index]=umProlongPressureCorrection(o);}
}
`);
    const resources = device.createBindGroupLayout({ entries: [0, 1, 2, 3].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: binding < 2 ? "read-only-storage" as const : "storage" as const } })) });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [fine.bindLayout, coarse.bindLayout, resources] });
    const stages = await Promise.all(["restrictValues", "prolongValues"].map(entryPoint => Promise.all([1, 2, 4].map(width =>
      device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint, constants: {
        fineumCellWidth: width, coarseumCellWidth: width, fineumDispatchX: fine.dispatchX, coarseumDispatchX: coarse.dispatchX,
      } } })))));
    const group = device.createBindGroup({ layout: resources, entries: [fineIn, coarseIn, fineOut, coarseOut].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const encoder = device.createCommandEncoder();
    for (let stage = 0; stage < 2; stage++) {
      const pass = encoder.beginComputePass(); pass.setBindGroup(0, fine.bindGroup); pass.setBindGroup(1, coarse.bindGroup); pass.setBindGroup(2, group);
      (stage === 0 ? coarse : fine).dispatch(pass, stages[stage]!); pass.end();
    }
    encoder.copyBufferToBuffer(fineOut, 0, readback, 0, nf * 4); encoder.copyBufferToBuffer(coarseOut, 0, readback, nf * 4, nc * 4);
    device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
    const values: Float32Array = new Float32Array(readback.getMappedRange());
    const expected = oracle.restrict(level, Float64Array.from(f, (v, i) => v * oracle.fixtures[level]!.cells[i]!.volume))
      .map((v, i) => v / oracle.fixtures[level + 1]!.cells[i]!.volume);
    assert.ok(maxError(values.subarray(0, nf), oracle.prolong(level, c)) < 1e-6);
    assert.ok(maxError(values.subarray(nf), expected) < 1e-6);
    readback.unmap();
  } finally { if (readback.mapState === "mapped") readback.unmap(); buffers.forEach(b => b.destroy()); }
}
