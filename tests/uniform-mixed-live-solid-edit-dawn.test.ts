import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import type {FluidRefinementRegion,SceneDescription} from "../lib/core/model";
import {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {uniformGeometricSolverOptions} from "../lib/methods/uniform/uniform-geometric-options";
import {mixedExtent,readMixedTexture} from "./helpers/uniform-mixed-native-fields";
const modulePath=process.env.WEBGPU_NODE_MODULE;

/** Live solid voxel edits on the mixed frame: a block filled into the
 * collapsing dam column must displace its liquid (none left inside, mass
 * conserved), converge, and survive being cleared again, under dynamic
 * coarsening (GPU-built layouts, the builder's drop re-run), under a
 * coarse region, in a scene built with no solids, and under Requested at
 * zero detail with solid contact off and on: the edit's tiles are joined to
 * one census, so the displacement meets them at h, and a scene that asks
 * for nothing is back at zero detail afterwards. */
const base=():SceneDescription=>{
 const s=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
 s.container.width_m=s.container.height_m=s.container.depth_m=.8;s.voxelDomain.finestCellSize_m=.025;
 s.rigidBodies=[];s.fluid.inflow=undefined;s.fluid.initialDamBreakDimensions_m={x:.3,y:.4,z:.8};
 // A static block downstream: the frame compiles its solid library.
 s.solidVoxels=[{operation:"fill",minimum:[16,0,12],maximumExclusive:[20,4,20]}];
 return s;
};
const coarse:FluidRefinementRegion={id:"coarse",rule:"minimum-cell-size",minimumCellSize_cells:4,maximumCellSize_cells:4,min_m:{x:-.4,y:0,z:-.4},max_m:{x:.4,y:.8,z:.4}};
const block={minimum:[3,3,20] as [number,number,number],maximumExclusive:[9,9,26] as [number,number,number]};

async function ownerFields(device:GPUDevice,solver:WebGPUUniformReferenceSolver){
 const texture=solver.volumeTexture,volume=await readMixedTexture(device,texture);
 const [nx,ny,nz]=mixedExtent(texture);
 const words=(solver as unknown as {mixedFrame:{ownership:{presentation:{buffer:GPUBuffer}}}}).mixedFrame.ownership.presentation.buffer;
 const bytes=4*(nx>>2)*(ny>>2)*(nz>>2),staging=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(words,0,staging,0,bytes);device.queue.submit([encoder.finish()]);
 await staging.mapAsync(GPUMapMode.READ);const tiles=new Uint32Array(staging.getMappedRange().slice(0));staging.destroy();
 let mass=0,inBlock=0,blockCoarse=0,sealed=0,fine=0;tiles.forEach(word=>{fine+=word>>>31;});
 for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
  const word=tiles[(x>>2)+(nx>>2)*((y>>2)+(ny>>2)*(z>>2))]!,width=word&0x80000000?1:4;
  if(x%width||y%width||z%width)continue;
  const v=volume[x+nx*(y+ny*z)]!;mass+=v*width**3;
  if([x,y,z].every((c,a)=>c>=block.minimum[a]!&&c<block.maximumExclusive[a]!)){inBlock=Math.max(inBlock,v);if(width!==1)blockCoarse++;}
  // Owners wholly inside the block, whatever their width.
  if([x,y,z].every((c,a)=>c>=block.minimum[a]!&&c+width<=block.maximumExclusive[a]!))sealed=Math.max(sealed,v);
 }
 return {mass,inBlock,blockCoarse,sealed,fine};
}

(modulePath?test:test.skip)("live solid voxel edits displace liquid on the mixed frame",{timeout:1800000},async()=>{
 let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const failures:string[]=[];
  // Compile by need: the solid-free scene builds no solid-gated pipeline, so
  // its first voxel waits for them. upfront: the same scene with every
  // pipeline built before the first step (warmPipelines), whose edit lands
  // at once; the two runs must hold the same fields.
  const landed:Record<string,{after:Float32Array;end:Float32Array}>={};
  // zero: Requested with no request, the named solid-contact setting.
  for(const [mode,empty,upfront,zero] of [["dynamic",false,false,undefined],["requested",false,false,undefined],["dynamic",true,false,undefined],["dynamic",true,true,undefined],
    ["requested",false,false,"off"],["requested",true,false,"on"]] as const){
   const scene=base();if(mode==="requested"&&!zero)scene.fluid.refinementRegions=[coarse];if(empty)scene.solidVoxels=[];
   const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,uniformGeometricSolverOptions({detailPolicy:mode,...(zero?{detailSolidContact:zero}:{})},scene),()=>{});
   const capacity=()=>(solver as unknown as {mixedFrame:{ownership:{capacity:{fineTiles:number}}}}).mixedFrame.ownership.capacity.fineTiles;
   const label=`${mode}${zero?` zero detail, contact ${zero}`:""}`;
   try{
    const deferred=solver.deferredPipelines;
    if(upfront){await solver.warmPipelines();if(solver.deferredPipelines)failures.push(`upfront: ${solver.deferredPipelines} pipelines left unbuilt by the warm-up`);}
    else if(!deferred)failures.push(`${mode}: setup deferred no pipeline`);
    let frame=0;const residuals:number[]=[];
    const step=async(n:number)=>{for(let i=0;i<n;i++){frame++;assert.ok(solver.advanceTo(frame/30),`${mode} advance ${frame}`);await solver.awaitFrameCompletion();residuals.push((await solver.readStats()).uniformPressureAcceptedResidual!);}};
    await step(8);
    const before=await ownerFields(device,solver);
    if(zero&&(before.fine||capacity()))failures.push(`${label}: ${before.fine} h tiles, capacity ${capacity()} before the edit`);
    const filled=structuredClone(scene);filled.solidVoxels.push({operation:"fill",...block});
    solver.applySceneUniforms(filled);
    // The first live edit waits for pipelines no earlier state needed
    // (displacement; in the solid-free scene every solid-gated variant).
    const waited=solver.info.uniformDetail?.preparing;
    if(zero&&!waited?.includes("dynamic"))failures.push(`${label}: the edit did not wait for the census pipelines (${waited})`);
    if(upfront?waited!==undefined:!waited?.includes("displace")||empty!==waited.includes("solids"))failures.push(`${mode}${empty?" (solid-free)":""}${upfront?" upfront":""}: the edit waited for ${waited??"nothing"}`);
    // Nothing may advance a state the caller did not ask for.
    if(!upfront)assert.throws(()=>solver.advanceTo((frame+1)/30),/await pipelinesPrepared/);
    await solver.pipelinesPrepared();
    if(solver.info.uniformDetail?.preparing)failures.push(`${mode}: still preparing ${solver.info.uniformDetail.preparing} after the wait`);
    await step(1);
    const after=await ownerFields(device,solver);
    const afterVolume=empty?await readMixedTexture(device,solver.volumeTexture):undefined;
    await step(5);
    // Requests only: the join was one build. With contact off the block's
    // tiles are 4h again and its sealed owners hold no liquid.
    const settled=zero?await ownerFields(device,solver):undefined;
    if(settled&&!(settled.sealed<=1e-4))failures.push(`${label}: ${settled.sealed} of liquid in an owner sealed by the solid, five frames on`);
    if(settled&&!(Math.abs(settled.mass-before.mass)<=1e-3*before.mass))failures.push(`${label}: mass ${settled.mass} five frames on vs ${before.mass}`);
    if(settled&&zero==="off"&&(settled.fine||capacity()))failures.push(`${label}: ${settled.fine} h tiles, capacity ${capacity()} five frames after the edit`);
    const cleared=structuredClone(filled);cleared.solidVoxels.push({operation:"clear",...block});
    solver.applySceneUniforms(cleared);
    await step(5);
    const end=await ownerFields(device,solver);
    if(empty&&!zero)landed[upfront?"upfront":"waited"]={after:afterVolume!,end:await readMixedTexture(device,solver.volumeTexture)};
    if(zero&&(end.fine||capacity()))failures.push(`${label}: ${end.fine} h tiles, capacity ${capacity()} after the block was cleared`);
    const report={mode,zero,empty,upfront,waited,deferred,before,after,settled,end,residuals:residuals.slice(7)};
    console.log(JSON.stringify(report));
    if(!(before.inBlock>0.5))failures.push(`${mode}: the block region was not submerged (${before.inBlock})`);
    if(!(after.inBlock<=1e-4))failures.push(`${mode}: ${after.inBlock} of liquid left inside the new solid`);
    if(after.blockCoarse)failures.push(`${mode}: ${after.blockCoarse} 4h owners inside the new solid`);
    if(!(Math.abs(after.mass-before.mass)<=1e-3*before.mass))failures.push(`${mode}: edit frame mass ${after.mass} vs ${before.mass}`);
    if(!(Math.abs(end.mass-before.mass)<=2e-3*before.mass))failures.push(`${mode}: mass after clear ${end.mass} vs ${before.mass}`);
    if(!residuals.every(Number.isFinite))failures.push(`${mode}: non-finite residuals ${residuals}`);
   }finally{solver.destroy();}
  }
  // The edit that waited landed as the one whose pipelines were built up front.
  for(const at of ["after","end"] as const){
   const a=landed.waited![at],b=landed.upfront![at];let worst=0,cells=0;
   for(let i=0;i<a.length;i++){const d=Math.abs(a[i]!-b[i]!);if(d>0){cells++;worst=Math.max(worst,d);}}
   console.log(JSON.stringify({compare:at,cells:a.length,differing:cells,worst}));
   if(a.length!==b.length||cells)failures.push(`solid-free edit: ${cells} cells of volume differ from the solver compiled up front at ${at} (worst ${worst})`);
  }
  assert.deepEqual(errors,[]);assert.deepEqual(failures,[]);
 }finally{device?.destroy();}
});
