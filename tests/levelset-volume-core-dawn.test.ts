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
    const layout = createLevelSetVolumeLayout({ activeCellCapacity: 64, vertexCapacity: 256 });
    const sampleBase = layout.totalWords;
    const code = `
struct Params { generation:u32, slot:u32, coarse:u32, pad:u32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read_write> topologyArena:array<atomic<u32>>;
const INVALID:u32=0xffffffffu;
fn isFinite(v:f32)->bool{return abs(v)<=3.402823466e38;}
fn boxMode()->bool{return params.pad>=2u;}
fn cellCount()->u32{if(boxMode()){return select(64u,36u,params.generation==7u);}
 if(params.pad!=0u){return 1u;}return select(9u,2u,params.coarse!=0u);}
fn cellWidths(cell:u32)->vec3f{if(boxMode()){return vec3f(1.0);}if(params.pad!=0u){return vec3f(16.0);}
 return vec3f(select(1.0,2.0,cell==0u||params.coarse!=0u));}
fn cellCenter(cell:u32)->vec3f{
 if(boxMode()){if(params.generation==7u){return vec3f(1.5+f32(cell%6u),0.5,1.5+f32(cell/6u));}
  return vec3f(0.5+f32(cell&7u),0.5,0.5+f32(cell>>3u));}
 if(params.pad!=0u){return vec3f(8.0);}
 if(cell==0u){return vec3f(1.0);}
 if(params.coarse!=0u){return vec3f(3.0,1.0,1.0);}
 let q=cell-1u;return vec3f(2.5+f32(q&1u),0.5+f32((q>>1u)&1u),0.5+f32((q>>2u)&1u));
}
fn ownerCellAt(q:vec3i)->u32{
 if(boxMode()){if(q.y!=0||q.x<0||q.z<0||q.x>=8||q.z>=8){return INVALID;}
  if(params.generation==7u){if(q.x<1||q.x>=7||q.z<1||q.z>=7){return INVALID;}
   return u32(q.x-1)+6u*u32(q.z-1);}
  return u32(q.x)+8u*u32(q.z);}
 if(params.pad!=0u){return select(INVALID,0u,all(q>=vec3i(0))&&all(q<vec3i(16)));}
 if(any(q<vec3i(0))||any(q>=vec3i(4,2,2))){return INVALID;}
 if(q.x<2){return 0u;}if(params.coarse!=0u){return 1u;}
 return 1u+u32(q.x-2)+2u*u32(q.y)+4u*u32(q.z);
}
fn authored(p:vec3f)->vec2f{if(boxMode()){
 let q=vec3f(abs(p.x-4.0)-2.0,p.y-1.0,abs(p.z-4.0)-2.0);
 return vec2f(length(max(q,vec3f(0.0)))+min(max(q.x,max(q.y,q.z)),0.0),3.0);}
 if(params.pad!=0u){return vec2f(p.x-8.0,3.0);}
 return vec2f(select((p.x+0.3*p.y+0.2*p.z-2.0)/sqrt(1.13),1000.0,params.generation>1u),3.0);}
` + createLevelSetVolumeWGSL({ layout,
      acceptedGenerationExpression: "params.generation", buildGenerationExpression: "params.generation",
      buildSlotExpression: "params.slot", buildCellCountExpression: "cellCount()",
      buildCellAtOrdinal: n => n, acceptedCellOrdinal: c => c,
      acceptedOwnerCellAt: q => `ownerCellAt(${q})`, buildOwnerCellAt: q => `ownerCellAt(${q})`,
      authoredSample: p => `authored(${p})`, velocitySample: p =>
        `select(vec4f(0.0,0.0,0.0,1.0),vec4f(1.0,0.0,1.0,1.0),params.pad==3u&&${p}.x<=0.0&&${p}.y==0.0&&${p}.z<=0.0)`,
      dtExpression: "select(0.0,1.0,params.pad==3u)", constraintWidthExpression: "0.0",
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
}
@compute @workgroup_size(8) fn sampleFractionalExtension(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=6u){return;}var p=vec3f(1.7,0.65,0.75);
 switch i {case 0u:{p.x=-0.25;}case 1u:{p.x=4.25;}case 2u:{p.y=-0.25;}
  case 3u:{p.y=2.25;}case 4u:{p.z=-0.25;}default:{p.z=2.25;}}
 atomicStore(&topologyArena[${sampleBase + 100}u+i],bitcast<u32>(lsvExtendFromSlot(lsvAcceptedSlot(),p).phi));
}
@compute @workgroup_size(1) fn sampleCertifiedAdvection(){
 let slot=lsvAcceptedSlot();let vertex=lsvLookupVertex(slot,vec3i(0));let destination=1u-lsvLoad(lsvHeader(slot,4u));
 atomicStore(&topologyArena[${sampleBase + 106}u],bitcast<u32>(lsvVertexPhi(slot,destination,vertex)));
 atomicStore(&topologyArena[${sampleBase + 107}u],lsvVertexSupport(slot,destination,vertex));
}
@compute @workgroup_size(256) fn sampleTransferredBox(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=162u){return;}let x=i%9u;let z=(i/9u)%9u;let y=i/81u;
 atomicStore(&topologyArena[${sampleBase + 128}u+i],bitcast<u32>(lsvPhiAt(vec3f(f32(x),f32(y),f32(z)))));
}
@compute @workgroup_size(256) fn sampleBox(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=162u){return;}let x=i%9u;let z=(i/9u)%9u;let y=i/81u;
 atomicStore(&topologyArena[${sampleBase + 290}u+i],bitcast<u32>(lsvPhiAt(vec3f(f32(x),f32(y),f32(z)))));
}`;
    const shaderModule = device.createShaderModule({ code });
    const info = await shaderModule.getCompilationInfo();
    assert.deepEqual(info.messages.filter(m => m.type === "error").map(m => `${m.lineNum}: ${m.message}`), []);
    const arena = device.createBuffer({ size: (sampleBase + 512) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const readback = device.createBuffer({ size: arena.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    device.queue.writeBuffer(arena, layout.headerBaseWords * 4, new Uint32Array(createLevelSetVolumeInitialWords(layout)));
    const bindLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindLayout] });
    const bindings = device.createBindGroup({ layout: bindLayout, entries: [{ binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: arena } }] });
    const names = ["lsvBeginTopology", "lsvClearTopology", "lsvCatalogCellCorners", "lsvInsertVertexHash", "lsvResolveCellCorners", "lsvCompileConstraints", "lsvInitializeAuthoredPhi", "lsvTransferPhi", "lsvBeginBuildConstraintProjection", "lsvApplyBuildConstraints", "lsvAdvanceBuildConstraintProjection", "lsvValidateTopology", "lsvSealTopology", "lsvPublishTopology", "lsvAdvectPhi", "lsvBeginConstraintProjection", "lsvApplyConstraints", "lsvAdvanceConstraintProjection", "lsvCommitPhi", ...LEVELSET_VOLUME_REDISTANCE_ENTRY_POINTS, "shiftPlane", "scaleCurrentPhi", "samplePlane", "samplePlaneUnder", "auditMacroMetric", "sampleFractionalExtension", "sampleCertifiedAdvection", "sampleTransferredBox", "sampleBox"];
    const pipelines = new Map<string, GPUComputePipeline>();
    for (const entryPoint of names) pipelines.set(entryPoint, await device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module: shaderModule, entryPoint } }));
    for (let generation = 1; generation <= 9; generation++) {
      const slot = (generation - 1) % 2;
      const macro = generation === 6;
      const box = generation >= 7;
      device.queue.writeBuffer(params, 0, new Uint32Array([generation, slot,
        generation % 2 === 0 ? 1 : 0, generation === 9 ? 3 : box ? 2 : macro ? 1 : 0]));
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass(); pass.setBindGroup(0, bindings);
      const dispatch = (name: string, count = 2) => { pass.setPipeline(pipelines.get(name)!); pass.dispatchWorkgroups(count); };
      const vertexWorkgroups = box ? 4 : 2;
      const projectConstraints = (building: boolean) => {
        dispatch(building ? "lsvBeginBuildConstraintProjection" : "lsvBeginConstraintProjection", 1);
        for (let level = 0; level < 3; level++) {
          dispatch(building ? "lsvApplyBuildConstraints" : "lsvApplyConstraints", vertexWorkgroups);
          dispatch(building ? "lsvAdvanceBuildConstraintProjection" : "lsvAdvanceConstraintProjection", 1);
        }
      };
      const redistance = () => {
        dispatch("lsvrBegin", 1); dispatch("lsvrCaptureOriginal", vertexWorkgroups);
        dispatch("lsvrSeedClosestPoints", vertexWorkgroups); dispatch("lsvrAdvance", 1);
        for (let iteration = 0; iteration < 16; iteration++) {
          dispatch("lsvrRelaxClosestPoints", vertexWorkgroups); dispatch("lsvrAdvance", 1);
        }
        dispatch("lsvrResolve", vertexWorkgroups); projectConstraints(false);
        dispatch("lsvrAuditContour", vertexWorkgroups);
      };
      dispatch("lsvBeginTopology", 1); dispatch("lsvClearTopology", 8);
      dispatch("lsvCatalogCellCorners", 1); dispatch("lsvInsertVertexHash", 4); dispatch("lsvResolveCellCorners", 1);
      dispatch("lsvCompileConstraints", 4); dispatch(generation === 1 || macro || generation === 7
        ? "lsvInitializeAuthoredPhi" : "lsvTransferPhi", 4);
      projectConstraints(true); dispatch("lsvValidateTopology", vertexWorkgroups); dispatch("lsvSealTopology", 1); dispatch("lsvPublishTopology", 1);
      if(generation === 1) dispatch("shiftPlane");
      dispatch("lsvAdvectPhi", vertexWorkgroups);
      if (generation === 9) dispatch("sampleCertifiedAdvection", 1);
      projectConstraints(false); dispatch("lsvCommitPhi", 1);
      if (generation === 8) dispatch("sampleTransferredBox", 1);
      redistance();
      if (generation === 8) dispatch("sampleBox", 1);
      else if (macro) dispatch("auditMacroMetric"); else if (!box) dispatch("samplePlane", 1);
      if (generation === 1) dispatch("sampleFractionalExtension", 1);
      if (generation === 5) {
        dispatch("scaleCurrentPhi"); redistance();
        dispatch("samplePlaneUnder", 1);
      }
      pass.end(); encoder.copyBufferToBuffer(arena, 0, readback, 0, arena.size); device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange()); const values = new Float32Array(words.buffer);
      assert.equal(words[layout.slots[slot]!.headerBaseWords + H.fault], 0, `generation ${generation} fault`);
      assert.equal(words[layout.headerBaseWords + G.acceptedGeneration], generation);
      for (let i = 0; i < (macro || box ? 0 : 48); i++) {
        const expected = (1.99 + .01 * (i % 3) + .3 * (.25 + .5 * (Math.floor(i / 3) % 4)) + .2 * (.25 + .5 * Math.floor(i / 12)) - 2) / Math.sqrt(1.13) + .125;
        assert.ok(Math.abs(values[sampleBase + i]! - expected) < 2e-5, `generation ${generation} sample ${i}: ${values[sampleBase + i]} vs ${expected}`);
      }
      if (macro) assert.equal(words[sampleBase + 96], 0,
        "all H16 plane-crossing corners remain metric beyond the base band");
      if (generation === 1) {
        const points = [[-.25,.65,.75], [4.25,.65,.75], [1.7,-.25,.75],
          [1.7,2.25,.75], [1.7,.65,-.25], [1.7,.65,2.25]];
        for (let i = 0; i < points.length; i++) {
          const [x, y, z] = points[i]!;
          const expected = (x + .3 * y + .2 * z - 2) / Math.sqrt(1.13) + .125;
          assert.ok(Math.abs(values[sampleBase + 100 + i]! - expected) < 2e-5,
            `fractional extension ${i}: ${values[sampleBase + 100 + i]} vs ${expected}`);
        }
      }
      if (generation === 9) {
        assert.ok(Math.abs(values[sampleBase + 106]! - Math.SQRT2) < 2e-5,
          `certified diagonal advection consumes travel: ${values[sampleBase + 106]}`);
        assert.equal(words[sampleBase + 107], 1,
          "an unsampled but phase-certified departure is downgraded to deep air");
      }
      if (generation === 8) for (const [stage, offset] of [["transfer", 128], ["redistance", 290]] as const) {
        const boxPhi = values.slice(sampleBase + offset, sampleBase + offset + 162);
        const at = (x: number, y: number, z: number) => boxPhi[x + 9 * (z + 9 * y)]!;
        for (let y = 0; y <= 1; y++) for (let z = 0; z <= 8; z++) for (let x = 0; x <= 8; x++) {
          const value = at(x, y, z);
          assert.ok(Number.isFinite(value), `box generation ${generation} ${stage} ${x},${y},${z} finite`);
          if (stage === "redistance") {
            const q = [Math.abs(x - 4) - 2, y - 1, Math.abs(z - 4) - 2];
            const expected = Math.hypot(...q.map(component => Math.max(component, 0)))
              + Math.min(Math.max(...q), 0);
            assert.ok(Math.abs(value - expected) < 2e-5,
              `box generation ${generation} ${stage} analytic SDF at ${x},${y},${z}: ${value} vs ${expected}`);
          }
          const inside = x >= 2 && x <= 6 && z >= 2 && z <= 6;
          if (!inside) assert.ok(value >= -1e-6,
            `box generation ${generation} ${stage} must not extend liquid to ${x},${y},${z}: ${value}`);
          assert.ok(Math.abs(value - at(8 - x, y, z)) < 2e-5,
            `box generation ${generation} ${stage} x reflection at ${x},${y},${z}: ${value} vs ${at(8 - x, y, z)}`);
          assert.ok(Math.abs(value - at(x, y, 8 - z)) < 2e-5,
            `box generation ${generation} ${stage} z reflection at ${x},${y},${z}: ${value} vs ${at(x, y, 8 - z)}`);
          assert.ok(Math.abs(value - at(z, y, x)) < 2e-5,
            `box generation ${generation} ${stage} transpose at ${x},${y},${z}: ${value} vs ${at(z, y, x)}`);
          if (inside && (y === 1 || x === 2 || x === 6 || z === 2 || z === 6)) {
            assert.ok(Math.abs(value) < 2e-5,
              `box generation ${generation} ${stage} retains boundary zero at ${x},${y},${z}: ${value}`);
          }
        }
      }
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
