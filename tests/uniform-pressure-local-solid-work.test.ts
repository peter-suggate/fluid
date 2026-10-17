import assert from "node:assert/strict";
import test from "node:test";
import {createRigidCouplingOracleScene} from "../lib/core/scenes";
import {initializeRigidBodies} from "../lib/core/rigid-body";
import {gpuCompilationManagerFor} from "../lib/core/gpu-compilation-manager";
import {UniformPressureBand} from "../lib/methods/uniform/uniform-pressure-band";
import {createUniformSolver,withUniformDevice} from "./helpers/uniform-geometric";

// Compare actual baked pressure rows at the frame head, before any body
// integration or row initialization. Both variants consume identical inputs.
test("local solid work preserves current moving-body band rows",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("pressure local solid work",async device=>{
  const scene=createRigidCouplingOracleScene("rigid-hydrostatic");
  scene.voxelDomain.finestCellSize_m=.0125;scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
  scene.fluid.gravity_m_s2={x:0,y:0,z:0};scene.fluid.dynamicViscosity_Pa_s=0;
  const bodies=initializeRigidBodies(scene.rigidBodies),solver=await createUniformSolver(device,scene);
  const frame=(solver as any).mixedFrame,band:UniformPressureBand=frame.band,original=band as any;
  const reference=new UniformPressureBand(original.device,original.simulation,original.pressure,{...original.fields,rows:undefined},original.schedule,original.solid,original.spatialOrder);
  const manager=gpuCompilationManagerFor(original.device),modules=new WeakSet<GPUShaderModule>();
  const create=manager.createShaderModule.bind(manager),compile=manager.compileComputePipeline.bind(manager);let overridden=0;
  manager.createShaderModule=descriptor=>{const module=create(descriptor);if(/override\s+bLocalSolidWork\s*:/.test(descriptor.code))modules.add(module);return module;};
  manager.compileComputePipeline=(descriptor,options)=>{if(modules.has(descriptor.compute.module)){overridden++;descriptor={...descriptor,compute:{...descriptor.compute,constants:{...descriptor.compute.constants,bLocalSolidWork:0}}};}return compile(descriptor,options);};
  try{await reference.initialize();assert.ok(overridden>0);}finally{manager.createShaderModule=create;manager.compileComputePipeline=compile;}
  const bytes=band.bytesAt(band.capacity).rows;
  const candidateRead=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const referenceRead=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const countRead=device.createBuffer({size:band.index.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const solid=original.solid.coarse;const tiles=original.simulation.capacity.tiles;
  const flagsRead=device.createBuffer({size:16*tiles,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const prepare=band.encodePrepare.bind(band);let captured=false;let clearBand=0,nearBand=0;
  band.encodePrepare=encoder=>{prepare(encoder);encoder.copyBufferToBuffer(original.rows,0,candidateRead,0,bytes);encoder.copyBufferToBuffer(band.index,0,countRead,0,countRead.size);encoder.copyBufferToBuffer(solid.record,16*solid.count,flagsRead,0,flagsRead.size);reference.encodePrepare(encoder);encoder.copyBufferToBuffer((reference as any).rows,0,referenceRead,0,bytes);captured=true;};
  try{
   for(let step=1;step<=3;step++){
    // Move the current body between frames, exercising entered/vacated
    // record rebuilds as well as the many remote open pressure rows.
    if(step>1){bodies[0]!.position_m.x+=.025;bodies[0]!.linearVelocity_m_s.x=0;}
    captured=false;assert.ok(solver.advanceTo(step/60,bodies));await solver.awaitFrameCompletion();assert.ok(captured);assert.equal(solver.info.simulationPipelineError,undefined);
    await Promise.all([candidateRead.mapAsync(GPUMapMode.READ),referenceRead.mapAsync(GPUMapMode.READ),countRead.mapAsync(GPUMapMode.READ),flagsRead.mapAsync(GPUMapMode.READ)]);
    const a=new Uint32Array(candidateRead.getMappedRange()),b=new Uint32Array(referenceRead.getMappedRange()),index=new Uint32Array(countRead.getMappedRange()),count=index[0]!;
    const flags=new Float32Array(flagsRead.getMappedRange());const list=band.slotMapOffset/4-tiles;const clearRegions=new Set<number>(),nearRegions=new Set<number>();
    for(let slot=0;slot<count;slot++){const tile=index[list+slot]!,local=tile&511;const region=8*(tile>>>9)+((local&7)>>>2)+2*(((local>>>3)&7)>>>2)+4*((local>>>6)>>>2);(flags[4*tile+3]!>.5?clearRegions:nearRegions).add(region);}
    clearBand+=clearRegions.size;nearBand+=nearRegions.size;console.log(JSON.stringify({step,activeBandTiles:count,clearBandRegions:clearRegions.size,nearBandRegions:nearRegions.size}));
    assert.ok(count>0);const N=64*band.capacity;
    for(let field=0;field<9;field++)assert.deepEqual(a.subarray(field*N,field*N+64*count),b.subarray(field*N,field*N+64*count),`frame ${step} row field ${field}`);
    for(const offset of [10*N,10*N+240*band.capacity])assert.deepEqual(a.subarray(offset,offset+240*count),b.subarray(offset,offset+240*count),`frame ${step} face records`);
    candidateRead.unmap();referenceRead.unmap();countRead.unmap();flagsRead.unmap();
   }
   assert.ok(clearBand>0,"current16h clear certificate must actually skip active band solid work");assert.ok(nearBand>0,"near-body band rows must retain full solid work");
  }finally{band.encodePrepare=prepare;for(const buffer of [candidateRead,referenceRead,countRead,flagsRead]){if(buffer.mapState==="mapped")buffer.unmap();buffer.destroy();}reference.destroy();solver.destroy();}
 });
});
