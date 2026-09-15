import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createLevelSetVolumeWGSL } from "../lib/methods/adaptive-volume/levelset-volume-core.wgsl";
import { createLevelSetVolumeLayout, createLevelSetVolumeInitialWords } from "../lib/methods/adaptive-volume/levelset-volume-layout";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("released walls carve the actual MAC displacement on all six faces", { timeout: 60_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "levelset wall separation analytic displacement");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const layout = createLevelSetVolumeLayout({ activeCellCapacity: 8, vertexCapacity: 64 });
    const sampleBase = layout.totalWords;
    const code = `
struct Params { face:u32, releaseMode:u32, dt:f32, deep:u32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read_write> topologyArena:array<atomic<u32>>;
const INVALID:u32=0xffffffffu;
fn isFinite(v:f32)->bool{return abs(v)<=3.402823466e38;}
fn cellWidths(cell:u32)->vec3f{return vec3f(2.0);}
fn cellCenter(cell:u32)->vec3f{return vec3f(vec3u(cell&1u,(cell>>1u)&1u,(cell>>2u)&1u))*2.0+vec3f(1.0);}
fn ownerCellAt(q:vec3i)->u32{if(any(q<vec3i(0))||any(q>=vec3i(4))){return INVALID;}let c=vec3u(q)/2u;return c.x+2u*c.y+4u*c.z;}
fn released(p:vec3f)->vec2f {
  let axis=params.face/2u;
  let distance=select(p[axis],4.0-p[axis],(params.face&1u)!=0u);
  var phi=2.0*params.dt-distance;
  if(params.releaseMode==2u){phi=max(phi,2.0*params.dt-p[(axis+1u)%3u]);}
  return vec2f(phi,select(0.0,1.0,params.releaseMode!=0u));
}
` + createLevelSetVolumeWGSL({ layout,
      acceptedGenerationExpression: "1u", buildGenerationExpression: "1u",
      buildSlotExpression: "0u", buildCellCountExpression: "8u",
      buildCellAtOrdinal: n => n, acceptedCellOrdinal: c => c,
      acceptedOwnerCellAt: q => `ownerCellAt(${q})`, buildOwnerCellAt: q => `ownerCellAt(${q})`,
      authoredSample: () => "vec2f(select(-4.0,-100.0,params.deep!=0u),select(3.0,2.0,params.deep!=0u))",
      // The cell trace is deliberately stationary; the wall's MAC speed is 2.
      velocitySample: () => "vec4f(0.0,0.0,0.0,1.0)",
      releasedWallPhi: p => `released(${p})`, dtExpression: "params.dt", constraintWidthExpression: "0.0",
    }) + `
@compute @workgroup_size(32) fn sampleDestination(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=27u){return;}
 let p=vec3i(vec3u(i%3u,(i/3u)%3u,i/9u))*2;
 let slot=lsvAcceptedSlot();let vertex=lsvLookupVertex(slot,p);
 let bank=1u-lsvLoad(lsvHeader(slot,4u));
 atomicStore(&topologyArena[${sampleBase}u+i],bitcast<u32>(lsvVertexPhi(slot,bank,vertex)));
 atomicStore(&topologyArena[${sampleBase + 27}u+i],lsvVertexSupport(slot,bank,vertex));
}`;
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    assert.deepEqual(info.messages.filter(m => m.type === "error").map(m => `${m.lineNum}: ${m.message}`), []);
    const arena = device.createBuffer({ size: (sampleBase + 54) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const readback = device.createBuffer({ size: 216, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const bindLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindLayout] });
    const bindings = device.createBindGroup({ layout: bindLayout, entries: [{ binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: arena } }] });
    const names = ["lsvBeginTopology", "lsvClearTopology", "lsvCatalogCellCorners", "lsvInsertVertexHash", "lsvResolveCellCorners", "lsvCompileConstraints", "lsvInitializeAuthoredPhi", "lsvBeginBuildConstraintProjection", "lsvApplyBuildConstraints", "lsvAdvanceBuildConstraintProjection", "lsvValidateTopology", "lsvSealTopology", "lsvPublishTopology", "lsvAdvectPhi", "sampleDestination"];
    const pipelines = new Map<string, GPUComputePipeline>();
    for (const entryPoint of names) pipelines.set(entryPoint, await device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint } }));
    for (const deep of [false, true]) for (const dt of [1.5, 0.25, 0.125, 0]) for (const active of [0, 1, 2]) for (let face = 0; face < 6; face++) {
      device.queue.writeBuffer(arena, 0, new Uint32Array(sampleBase + 54));
      device.queue.writeBuffer(arena, layout.headerBaseWords * 4, new Uint32Array(createLevelSetVolumeInitialWords(layout)));
      const uniform = new ArrayBuffer(16); const u = new Uint32Array(uniform);
      u[0] = face; u[1] = active; new Float32Array(uniform)[2] = dt; u[3] = Number(deep);
      device.queue.writeBuffer(params, 0, uniform);
      const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass(); pass.setBindGroup(0, bindings);
      const dispatch = (name: string, count = 1) => { pass.setPipeline(pipelines.get(name)!); pass.dispatchWorkgroups(count); };
      dispatch("lsvBeginTopology"); dispatch("lsvClearTopology", 8);
      for (const name of ["lsvCatalogCellCorners", "lsvInsertVertexHash", "lsvResolveCellCorners", "lsvCompileConstraints", "lsvInitializeAuthoredPhi", "lsvBeginBuildConstraintProjection"]) dispatch(name);
      for (let level = 0; level < 3; level++) { dispatch("lsvApplyBuildConstraints"); dispatch("lsvAdvanceBuildConstraintProjection"); }
      for (const name of ["lsvValidateTopology", "lsvSealTopology", "lsvPublishTopology", "lsvAdvectPhi", "sampleDestination"]) dispatch(name);
      pass.end(); encoder.copyBufferToBuffer(arena, sampleBase * 4, readback, 0, 216); device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(readback.getMappedRange()); const words = new Uint32Array(values.buffer);
      for (let i = 0; i < 27; i++) {
        const p = [2 * (i % 3), 2 * (Math.floor(i / 3) % 3), 2 * Math.floor(i / 9)];
        const axis = Math.floor(face / 2); const distance = face & 1 ? 4 - p[axis]! : p[axis]!;
        let expected = deep ? -100 : -4;
        if (active && dt > 0) {
          expected = Math.max(expected, 2 * dt - distance);
          if (active === 2) expected = Math.max(expected, 2 * dt - p[(axis + 1) % 3]!);
        }
        const context = JSON.stringify({ deep, dt, active, face, i });
        assert.ok(Math.abs(values[i]! - expected) < 1e-5, `${context}: phi ${values[i]} expected ${expected}`);
        if (active && dt > 0) assert.equal(words[27 + i], 3, `${context}: released contour needs metric support`);
      }
      readback.unmap();
    }
    arena.destroy(); params.destroy(); readback.destroy();
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
