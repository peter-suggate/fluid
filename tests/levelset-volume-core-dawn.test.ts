import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createLevelSetVolumeWGSL } from "../lib/methods/adaptive-volume/levelset-volume-core.wgsl";
import { createLevelSetVolumeRedistanceWGSL, LEVELSET_VOLUME_REDISTANCE_ENTRY_POINTS,
  LEVELSET_VOLUME_REDISTANCE_RECEIPT_BASE } from
  "../lib/methods/adaptive-volume/levelset-volume-redistance.wgsl";
import { createLevelSetVolumeLayout, createLevelSetVolumeInitialWords, LEVELSET_VOLUME_GLOBAL_HEADER as G, LEVELSET_VOLUME_SLOT_HEADER as H } from "../lib/methods/adaptive-volume/levelset-volume-layout";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("adaptive vertex phi preserves a plane across hanging vertices and repeated topology transfer", { timeout: 60_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "adaptive vertex phi seam and transfer");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const layout = createLevelSetVolumeLayout({ activeCellCapacity: 32, vertexCapacity: 128 });
    const sampleBase = layout.totalWords;
    const code = `
struct Params { generation:u32, slot:u32, coarse:u32, pad:u32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read_write> topologyArena:array<atomic<u32>>;
const INVALID:u32=0xffffffffu;
fn isFinite(v:f32)->bool{return abs(v)<=3.402823466e38;}
fn cellCount()->u32{if(params.pad!=0u){return 1u;}return select(9u,2u,params.coarse!=0u);}
fn cellWidths(cell:u32)->vec3f{if(params.pad!=0u){return vec3f(16.0);}
 return vec3f(select(1.0,2.0,cell==0u||params.coarse!=0u));}
fn cellCenter(cell:u32)->vec3f{
 if(params.pad!=0u){return vec3f(8.0);}
 if(cell==0u){return vec3f(1.0);}
 if(params.coarse!=0u){return vec3f(3.0,1.0,1.0);}
 let q=cell-1u;return vec3f(2.5+f32(q&1u),0.5+f32((q>>1u)&1u),0.5+f32((q>>2u)&1u));
}
fn ownerCellAt(q:vec3i)->u32{
 if(params.pad!=0u){return select(INVALID,0u,all(q>=vec3i(0))&&all(q<vec3i(16)));}
 if(any(q<vec3i(0))||any(q>=vec3i(4,2,2))){return INVALID;}
 if(q.x<2){return 0u;}if(params.coarse!=0u){return 1u;}
 return 1u+u32(q.x-2)+2u*u32(q.y)+4u*u32(q.z);
}
fn authored(p:vec3f)->vec2f{if(params.pad!=0u){return vec2f(p.x-8.0,3.0);}
 return vec2f(select((p.x+0.3*p.y+0.2*p.z-2.0)/sqrt(1.13),1000.0,params.generation>1u),3.0);}
` + createLevelSetVolumeWGSL({ layout,
      acceptedGenerationExpression: "params.generation", buildGenerationExpression: "params.generation",
      buildSlotExpression: "params.slot", buildCellCountExpression: "cellCount()",
      buildCellAtOrdinal: n => n, acceptedCellOrdinal: c => c,
      acceptedOwnerCellAt: q => `ownerCellAt(${q})`, buildOwnerCellAt: q => `ownerCellAt(${q})`,
      authoredSample: p => `authored(${p})`, velocitySample: () => "vec4f(0.0,0.0,0.0,1.0)",
      dtExpression: "0.0", constraintWidthExpression: "0.0",
    }) + createLevelSetVolumeRedistanceWGSL({ layout, bandWidthExpression: "4.0" }) + `
@compute @workgroup_size(64) fn shiftPlane(@builtin(global_invocation_id) gid:vec3u){
 let vertex=gid.x;let slot=lsvAcceptedSlot();if(vertex>=lsvLoad(lsvHeader(slot,3u))){return;}
 for(var bank=0u;bank<2u;bank+=1u){let at=lsvPhiBase(slot,bank)+vertex;lsvStoreFloat(at,lsvFloat(at)+0.125);}
}
@compute @workgroup_size(64) fn scaleCurrentPhi(@builtin(global_invocation_id) gid:vec3u){
 let vertex=gid.x;let slot=lsvAcceptedSlot();if(vertex>=lsvLoad(lsvHeader(slot,3u))){return;}
 let bank=lsvLoad(lsvHeader(slot,4u));let at=lsvPhiBase(slot,bank)+vertex;lsvStoreFloat(at,0.25*lsvFloat(at));
}
@compute @workgroup_size(64) fn samplePlane(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=48u){return;}
 let p=vec3f(1.99+0.01*f32(i%3u),0.25+0.5*f32((i/3u)%4u),0.25+0.5*f32(i/12u));
 atomicStore(&topologyArena[${sampleBase}u+i],bitcast<u32>(lsvPhiAt(p)));
}
@compute @workgroup_size(64) fn samplePlaneUnder(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=48u){return;}
 let p=vec3f(1.99+0.01*f32(i%3u),0.25+0.5*f32((i/3u)%4u),0.25+0.5*f32(i/12u));
 atomicStore(&topologyArena[${sampleBase + 48}u+i],bitcast<u32>(lsvPhiAt(p)));
}
@compute @workgroup_size(64) fn auditMacroMetric(@builtin(global_invocation_id) gid:vec3u){
 let vertex=gid.x;let slot=lsvAcceptedSlot();if(vertex>=lsvLoad(lsvHeader(slot,3u))){return;}
 let bank=lsvLoad(lsvHeader(slot,4u));if(lsvVertexSupport(slot,bank,vertex)!=LSV_SUPPORT_METRIC){
  atomicAdd(&topologyArena[${sampleBase + 96}u],1u);}
}`;
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    assert.deepEqual(info.messages.filter(m => m.type === "error").map(m => `${m.lineNum}: ${m.message}`), []);
    const arena = device.createBuffer({ size: (sampleBase + 128) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const readback = device.createBuffer({ size: arena.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    device.queue.writeBuffer(arena, layout.headerBaseWords * 4, new Uint32Array(createLevelSetVolumeInitialWords(layout)));
    const bindLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindLayout] });
    const bindings = device.createBindGroup({ layout: bindLayout, entries: [{ binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: arena } }] });
    const names = ["lsvBeginTopology", "lsvClearTopology", "lsvCatalogCellCorners", "lsvInsertVertexHash", "lsvResolveCellCorners", "lsvCompileConstraints", "lsvInitializeAuthoredPhi", "lsvTransferPhi", "lsvBeginBuildConstraintProjection", "lsvApplyBuildConstraints", "lsvAdvanceBuildConstraintProjection", "lsvValidateTopology", "lsvSealTopology", "lsvPublishTopology", "lsvAdvectPhi", "lsvBeginConstraintProjection", "lsvApplyConstraints", "lsvAdvanceConstraintProjection", "lsvCommitPhi", ...LEVELSET_VOLUME_REDISTANCE_ENTRY_POINTS, "shiftPlane", "scaleCurrentPhi", "samplePlane", "samplePlaneUnder", "auditMacroMetric"];
    const pipelines = new Map<string, GPUComputePipeline>();
    for (const entryPoint of names) pipelines.set(entryPoint, await device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint } }));
    for (let generation = 1; generation <= 6; generation++) {
      const slot = (generation - 1) % 2;
      const macro = generation === 6;
      device.queue.writeBuffer(params, 0, new Uint32Array([generation, slot,
        generation % 2 === 0 ? 1 : 0, macro ? 1 : 0]));
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass(); pass.setBindGroup(0, bindings);
      const dispatch = (name: string, count = 2) => { pass.setPipeline(pipelines.get(name)!); pass.dispatchWorkgroups(count); };
      const projectConstraints = (building: boolean) => {
        dispatch(building ? "lsvBeginBuildConstraintProjection" : "lsvBeginConstraintProjection", 1);
        for (let level = 0; level < 3; level++) {
          dispatch(building ? "lsvApplyBuildConstraints" : "lsvApplyConstraints");
          dispatch(building ? "lsvAdvanceBuildConstraintProjection" : "lsvAdvanceConstraintProjection", 1);
        }
      };
      const redistance = () => {
        dispatch("lsvrBegin", 1); dispatch("lsvrCaptureOriginal"); dispatch("lsvrSeedClosestPoints"); dispatch("lsvrAdvance", 1);
        for (let iteration = 0; iteration < 16; iteration++) {
          dispatch("lsvrRelaxClosestPoints"); dispatch("lsvrAdvance", 1);
        }
        dispatch("lsvrResolve"); projectConstraints(false); dispatch("lsvrAuditContour");
      };
      dispatch("lsvBeginTopology", 1); dispatch("lsvClearTopology", 4);
      dispatch("lsvCatalogCellCorners", 1); dispatch("lsvInsertVertexHash"); dispatch("lsvResolveCellCorners", 1);
      dispatch("lsvCompileConstraints"); dispatch(generation === 1 || macro
        ? "lsvInitializeAuthoredPhi" : "lsvTransferPhi");
      projectConstraints(true); dispatch("lsvValidateTopology"); dispatch("lsvSealTopology", 1); dispatch("lsvPublishTopology", 1);
      if(generation === 1) dispatch("shiftPlane");
      dispatch("lsvAdvectPhi"); projectConstraints(false); dispatch("lsvCommitPhi", 1);
      redistance();
      if (macro) dispatch("auditMacroMetric"); else dispatch("samplePlane", 1);
      if (generation === 5) {
        dispatch("scaleCurrentPhi"); redistance();
        dispatch("samplePlaneUnder", 1);
      }
      pass.end(); encoder.copyBufferToBuffer(arena, 0, readback, 0, arena.size); device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange()); const values = new Float32Array(words.buffer);
      assert.equal(words[layout.slots[slot]!.headerBaseWords + H.fault], 0, `generation ${generation} fault`);
      assert.equal(words[layout.headerBaseWords + G.acceptedGeneration], generation);
      for (let i = 0; i < (macro ? 0 : 48); i++) {
        const expected = (1.99 + .01 * (i % 3) + .3 * (.25 + .5 * (Math.floor(i / 3) % 4)) + .2 * (.25 + .5 * Math.floor(i / 12)) - 2) / Math.sqrt(1.13) + .125;
        assert.ok(Math.abs(values[sampleBase + i]! - expected) < 2e-5, `generation ${generation} sample ${i}: ${values[sampleBase + i]} vs ${expected}`);
      }
      if (macro) assert.equal(words[sampleBase + 96], 0,
        "all H16 plane-crossing corners remain metric beyond the base band");
      if (generation === 5) {
        let recovered = 0;
        const underSamples: number[] = [];
        for (let i = 0; i < 48; i++) {
          const value = values[sampleBase + 48 + i]!;
          underSamples.push(value);
          assert.ok(Number.isFinite(value), `under-distance sample ${i} is finite`);
          if (Math.abs(value) > .25 * Math.abs(values[sampleBase + i]!) + 1e-4) recovered++;
        }
        const receipt = layout.slots[slot]!.headerBaseWords + LEVELSET_VOLUME_REDISTANCE_RECEIPT_BASE;
        assert.ok(recovered > 0,
          `closest-point propagation increases an under-distance field; samples=${JSON.stringify(underSamples)}, prior=${JSON.stringify(Array.from(values.slice(sampleBase, sampleBase + 48)))}, receipts=${JSON.stringify(Array.from(words.slice(receipt, receipt + 12)))}`);
        assert.ok(Number.isFinite(values[receipt + 3]!), "contour drift receipt is finite");
        assert.ok(values[receipt + 3]! < 1.0, `contour drift remains locally bounded: ${values[receipt + 3]}`);
      }
      readback.unmap();
    }
    arena.destroy(); params.destroy(); readback.destroy();
  } finally {
    device?.destroy(); await new Promise<void>(resolve => setImmediate(resolve)); gpu = undefined;
    await releaseWebGPUExclusiveLock();
  }
});
