import assert from "node:assert/strict";
import test from "node:test";
import {writeFileSync} from "node:fs";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import type {FluidRefinementRegion,SceneDescription} from "../lib/core/model";
import {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {uniformGeometricSolverOptions} from "../lib/methods/uniform/uniform-geometric-options";
import {mixedExtent,readMixedTexture} from "./helpers/uniform-mixed-native-fields";
const modulePath=process.env.WEBGPU_NODE_MODULE;

/** The detail policy lifecycle on one running solver, no rebuild and no time
 * reset: Requested with no region is all 4h; a Fine region drawn live is h
 * exactly on its tiles; Full is every tile h; removing both returns to all
 * 4h; Dynamic holds h on the focus box and follows it when it moves. */
const base=():SceneDescription=>{
 const s=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
 s.container.width_m=s.container.height_m=s.container.depth_m=.8;s.voxelDomain.finestCellSize_m=.025;
 s.rigidBodies=[];s.fluid.inflow=undefined;s.fluid.initialDamBreakDimensions_m={x:.3,y:.4,z:.8};s.solidVoxels=[];
 return s;
};
// Tiles 4..7 in x, 0..3 in y, all of z (32³ cells, 8³ tiles, 0.1 m per tile).
const fine:FluidRefinementRegion={id:"fine",rule:"minimum-cell-size",minimumCellSize_cells:1,maximumCellSize_cells:1,min_m:{x:0,y:0,z:-.4},max_m:{x:.4,y:.4,z:.4}};

async function owners(device:GPUDevice,solver:WebGPUUniformReferenceSolver){
 const texture=solver.volumeTexture,volume=await readMixedTexture(device,texture);
 const [nx,ny,nz]=mixedExtent(texture);
 const words=(solver as unknown as {mixedFrame:{ownership:{presentation:{buffer:GPUBuffer}}}}).mixedFrame.ownership.presentation.buffer;
 const bytes=4*(nx>>2)*(ny>>2)*(nz>>2),staging=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(words,0,staging,0,bytes);device.queue.submit([encoder.finish()]);
 await staging.mapAsync(GPUMapMode.READ);const tiles=new Uint32Array(staging.getMappedRange().slice(0));staging.destroy();
 let mass=0;const fineTiles:number[]=[];
 tiles.forEach((word,t)=>{if(word&0x80000000)fineTiles.push(t);});
 for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
  const width=tiles[(x>>2)+(nx>>2)*((y>>2)+(ny>>2)*(z>>2))]!&0x80000000?1:4;
  if(x%width||y%width||z%width)continue;
  const v=volume[x+nx*(y+ny*z)]!;assert.ok(Number.isFinite(v),`volume at ${x},${y},${z}`);mass+=v*width**3;
 }
 return {mass,fineTiles,tileDims:[nx>>2,ny>>2,nz>>2] as const};
}

/** Vertex phi stays a bounded distance (drifting to about -1.2 m deep in the
 * 0.8 m dam break); ten container widths separates that from a blow-up. The
 * surface-volume shift under an every-tile-h layout wrote -1e4 m on frame 1
 * at 32³ and still passed every ownership, mass and residual check here. */
async function phiFault(device:GPUDevice,solver:WebGPUUniformReferenceSolver,width_m:number):Promise<string|undefined>{
 const phi=await readMixedTexture(device,solver.vertexPhiTexture!);let min=Infinity,max=-Infinity,bad=0;
 for(const v of phi){if(!Number.isFinite(v)){bad++;continue;}if(v<min)min=v;if(v>max)max=v;}
 return bad||Math.max(-min,max)>10*width_m?`phi ${min}..${max} m, ${bad} non-finite`:undefined;
}

/** GPU time of a frame as the sum over its compute passes (timestamp
 * queries). Wall time is no measure here: the binding's collections stall
 * the event pump for whole frames. */
function passClock(raw:GPUDevice){
 // 4096 is WebGPU's query set limit: 2,048 passes a frame.
 const CAP=4096,set=raw.createQuerySet({type:"timestamp",count:CAP});
 const resolved=raw.createBuffer({size:8*CAP,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),staging=raw.createBuffer({size:8*CAP,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const probe=raw.createCommandEncoder(),proto=Object.getPrototypeOf(probe) as {beginComputePass:GPUCommandEncoder["beginComputePass"]},begin=proto.beginComputePass;probe.finish();
 let next=0,on=false;
 proto.beginComputePass=function(this:GPUCommandEncoder,d?:GPUComputePassDescriptor){
  if(!on||d?.timestampWrites)return begin.call(this,d);
  assert.ok(next+2<=CAP,"pass clock: more passes in a frame than its query set holds");const i=next;next+=2;
  return begin.call(this,{...d,timestampWrites:{querySet:set,beginningOfPassWriteIndex:i,endOfPassWriteIndex:i+1}});
 };
 return {
  start(){next=0;on=true;},
  async stop():Promise<number>{
   on=false;const n=next;assert.ok(n>0,"pass clock: the frame encoded no compute pass");
   const e=raw.createCommandEncoder();e.resolveQuerySet(set,0,n,resolved,0);e.copyBufferToBuffer(resolved,0,staging,0,8*n);raw.queue.submit([e.finish()]);
   await staging.mapAsync(GPUMapMode.READ,0,8*n);const t=new BigUint64Array(staging.getMappedRange(0,8*n).slice(0));staging.unmap();
   let ms=0;for(let k=0;k<n;k+=2)ms+=Number(t[k+1]!-t[k]!)/1e6;return ms;
  },
  restore(){proto.beginComputePass=begin;set.destroy();resolved.destroy();staging.destroy();},
 };
}
/** Zero detail to Dynamic: the first two frames after the switch, in GPU pass
 * time, as a multiple of the steady Dynamic frame (the median of the twelve
 * focus-moved frames). The direct launch budgets decay while no tile is h
 * and the receipts that would regrow them are two frames late, so a switch
 * that kept them ran its first two frames at 9-10x here (13-16x after twelve
 * zero frames). With the budgets forgotten at the attach, frame 1 is
 * 2.5-2.8x (it carries the remap of every tile) and frame 2 1.4x: Apple
 * M-series, Metal, identity placement, 4 October 2026. Domain placement
 * (the default), same day: frame 1 1.97x (13.8 ms), frame 2 1.34x of a
 * 7.0 ms steady frame. */
const DYNAMIC_ENTRY_MULTIPLE=5;

(modulePath?test:test.skip)("detail policy changes ownership live: zero, partial, full, zero, moving focus",{timeout:1800000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform detail policy lifecycle");let device:GPUDevice|undefined,clock:ReturnType<typeof passClock>|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal","disable-dawn-features=timestamp_quantization"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  const raw=await adapter.requestDevice({requiredFeatures:["timestamp-query"],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
  clock=passClock(raw);device=managedGPUDevice(raw,{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const failures:string[]=[],report:Record<string,unknown>[]=[];
  const scene=base();
  const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,uniformGeometricSolverOptions({detailPolicy:"requested"},scene),()=>{});
  try{
   let frame=0;
   const step=async(n:number,label:string,passTimes?:number[])=>{for(let i=0;i<n;i++){frame++;if(passTimes)clock!.start();
    assert.ok(solver.advanceTo(frame/30),`${label} advance ${frame}`);await solver.awaitFrameCompletion();if(passTimes)passTimes.push(await clock!.stop());
    const residual=(await solver.readStats()).uniformPressureAcceptedResidual!;if(!Number.isFinite(residual))failures.push(`${label}: non-finite residual at frame ${frame}`);}};
   const check=async(label:string,expect:(t:number,d:readonly[number,number,number])=>boolean|undefined,reference?:number)=>{
    const o=await owners(device!,solver),[tx,ty]=o.tileDims,total=tx*ty*o.tileDims[2];
    const isFine=new Uint8Array(total);o.fineTiles.forEach(t=>isFine[t]=1);let wrong=0;
    for(let t=0;t<total;t++){const want=expect(t,o.tileDims);if(want!==undefined&&want!==!!isFine[t])wrong++;}
    report.push({label,frame,fineTiles:o.fineTiles.length,tiles:total,mass:o.mass,wrong});
    if(wrong)failures.push(`${label}: ${wrong} tiles at the wrong width (${o.fineTiles.length}/${total} h)`);
    if(reference!==undefined&&!(Math.abs(o.mass-reference)<=2e-3*reference))failures.push(`${label}: mass ${o.mass} vs ${reference}`);
    return o;
   };
   const inFine=(t:number,d:readonly[number,number,number])=>{const x=t%d[0],y=Math.floor(t/d[0])%d[1];return x>=4&&y<4;};
   await step(4,"requested");
   // Mass is checked phase to phase (the live-solid-edit lane's 2e-3 over a
   // comparable span); the report keeps the absolute values.
   let last=(await check("requested, no region",()=>false)).mass;
   // A Fine region drawn while running.
   const drawn=structuredClone(scene);drawn.fluid.refinementRegions=[fine];solver.applySceneUniforms(drawn);await solver.pipelinesPrepared();
   await step(4,"fine region");
   last=(await check("requested, fine region",inFine,last)).mass;
   solver.applyRuntimeValues({detailPolicy:"full"});await solver.pipelinesPrepared();
   await step(4,"full");
   last=(await check("full",()=>true,last)).mass;
   {const fault=await phiFault(device,solver,.8);if(fault)failures.push(`full: ${fault}`);}
   solver.applyRuntimeValues({detailPolicy:"requested"});solver.applySceneUniforms(scene);await solver.pipelinesPrepared();
   await step(4,"zero again");
   last=(await check("requested again, region removed",()=>false,last)).mass;
   // Dynamic: the focus box is h wherever the census would also allow 4h.
   const radius=.2*.8,focusTiles=(c:number)=>(t:number,d:readonly[number,number,number])=>{
    const x=t%d[0],y=Math.floor(t/d[0])%d[1],z=Math.floor(t/(d[0]*d[1]));
    // Tile centres strictly inside the focus box must be h; others are the census's choice.
    const inside=[x,y,z].every((v,a)=>{const centre=(v+.5)*.1+(a===1?0:-.4),target=a===0?c:a===1?.6:0;return Math.abs(centre-target)<radius-.05;});
    return inside?true:undefined;
   };
   solver.applyRuntimeValues({detailPolicy:"dynamic",detailNearFocus:"on",detailFocusRadiusPercent:20});await solver.pipelinesPrepared();
   solver.applyDetailInput({focus:{revision:1,position_m:{x:.2,y:.6,z:0}}});
   const entry:number[]=[],steady:number[]=[];
   await step(4,"dynamic focus",entry);
   last=(await check("dynamic, focus at +x in dry air",focusTiles(.2),last)).mass;
   solver.applyDetailInput({focus:{revision:2,position_m:{x:-.2,y:.6,z:0}}});
   await step(12,"dynamic focus moved",steady);
   await check("dynamic, focus moved to -x",focusTiles(-.2),last);
   {const median=[...steady].sort((a,b)=>a-b)[steady.length>>1]!,multiple=Math.max(entry[0]!,entry[1]!)/median,ms=(v:number)=>+v.toFixed(2);
    report.push({label:"dynamic entry pass time",entry_ms:entry.map(ms),steady_ms:ms(median),multiple:+multiple.toFixed(2),limit:DYNAMIC_ENTRY_MULTIPLE});
    if(!(multiple<=DYNAMIC_ENTRY_MULTIPLE))failures.push(`zero to Dynamic: frames 1 and 2 took ${entry.slice(0,2).map(ms).join(" and ")} ms of GPU passes, ${multiple.toFixed(1)}x the steady ${ms(median)} ms (limit ${DYNAMIC_ENTRY_MULTIPLE}x)`);}
   if(Math.abs(frame/30-(await solver.readStats()).simulatedTime_s!)>1e-9)failures.push(`simulation time ${(await solver.readStats()).simulatedTime_s} is not ${frame/30}`);
   // Last, because the failure is sticky: a GPU relayout that builds more
   // h tiles than the owner-indexed buffers hold fails loudly (the builder's
   // h-tile capacity word, lowered here to zero), never silently.
   (solver as unknown as {mixedBuilder:{setFineCapacity(tiles:number):void}}).mixedBuilder.setFineCapacity(0);
   let fatal="";
   try{for(let i=0;i<6;i++){frame++;if(!solver.advanceTo(frame/30))break;await solver.awaitFrameCompletion();await solver.readStats();}}catch(error){fatal=error instanceof Error?error.message:String(error);}
   report.push({label:"h-tile capacity fatal",frame,fatal});
   if(!/h tiles over the owner-indexed storage's capacity/.test(fatal)||!/capacity \d+/.test(fatal))failures.push(`a relayout over the h-tile capacity did not fail loudly: ${fatal||"no error"}`);
   console.log(JSON.stringify(report));
   // The lane runner prints a passing file's output nowhere; the measured
   // entry multiples are wanted on a pass too.
   if(process.env.UNIFORM_DETAIL_POLICY_REPORT)writeFileSync(process.env.UNIFORM_DETAIL_POLICY_REPORT,JSON.stringify(report));
  }finally{solver.destroy();}
  assert.deepEqual(errors,[]);assert.deepEqual(failures,[]);
 }finally{clock?.restore();device?.destroy();await releaseWebGPUExclusiveLock();}
});

/** Full above 32³: the 64³ dam break with every tile h from the first frame,
 * through the impact and the first return. The pressure gate throws on a
 * rejected or unconverged frame, so reaching frame 60 is 60 accepted solves. */
(modulePath?test:test.skip)("full detail runs the 64³ dam break: every tile h, accepted pressure, bounded phi",{timeout:900000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform detail policy full 64");let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const failures:string[]=[];
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-64")));
  const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,uniformGeometricSolverOptions({detailPolicy:"full"},scene),()=>{});
  try{
   let residualMax=0,bandMax=0,start:Awaited<ReturnType<typeof owners>>|undefined,tiles=0;
   for(let frame=1;frame<=60;frame++){
    assert.ok(solver.advanceTo(frame/30),`advance ${frame}`);await solver.awaitFrameCompletion();
    const stats=await solver.readStats(),residual=stats.uniformPressureAcceptedResidual!;
    if(!Number.isFinite(residual))failures.push(`non-finite residual at frame ${frame}`);
    residualMax=Math.max(residualMax,residual);bandMax=Math.max(bandMax,stats.uniformPressureBandTiles??0);
    if(frame<=3||frame%10===0){const fault=await phiFault(device,solver,scene.container.width_m);if(fault)failures.push(`frame ${frame}: ${fault}`);}
    if(frame===1){start=await owners(device,solver);tiles=start.tileDims[0]*start.tileDims[1]*start.tileDims[2];assert.equal(start.fineTiles.length,tiles,"full is every tile h from the first frame");}
   }
   assert.ok(start);const end=await owners(device,solver);
   if(end.fineTiles.length!==tiles)failures.push(`${end.fineTiles.length}/${tiles} tiles h after 60 frames`);
   if(!(Math.abs(end.mass-start.mass)<=2e-3*start.mass))failures.push(`mass ${end.mass} vs ${start.mass}`);
   const time=(await solver.readStats()).simulatedTime_s!;if(Math.abs(time-2)>1e-9)failures.push(`simulation time ${time} is not 2`);
   console.log(JSON.stringify({label:"full 64",tiles,residualMax,bandMax,mass:[start.mass,end.mass]}));
  }finally{solver.destroy();}
  assert.deepEqual(errors,[]);assert.deepEqual(failures,[]);
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});

/** A Fine region drawn live far from the liquid, with solid contact on (the
 * app's default) and a solid in the liquid, so the census and its residency
 * certificate run every frame. The region's tiles become h by the builder's
 * static mask, in a page the certificate of that census left absent as far
 * air: the next census found h tiles in an absent page and raised the
 * residency fatal ("invalid support (layout build N: fatal bits 8)") one
 * frame after an ordinary gesture (garden hose x10, 5 October 2026). The
 * certificate now seeds the static h tiles. 64³: 16³ tiles, 4³ pages. */
(modulePath?test:test.skip)("a Fine region drawn and moved far from the liquid under solid contact keeps its pages resident",{timeout:900000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform detail policy far region");let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const failures:string[]=[];
  // A resting pool one tile deep over the whole floor (the census's near set
  // ends at tile 4 in y); a block standing in it gives the census
  // liquid-conditional contact tiles, so it runs every frame.
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-64")));
  scene.fluid.initialDamBreakDimensions_m={x:.8,y:.05,z:.8};
  scene.solidVoxels=[...scene.solidVoxels,{operation:"fill",minimum:[28,0,28],maximumExclusive:[36,8,36],materialId:1}];
  // Tiles [x0, x0+2) x [14, 16) x [12, 14), 0.05 m per tile: the top page row (tiles 12..15 in y), eight tiles above the near set.
  const region=(x0:number):FluidRefinementRegion=>({id:"far",rule:"minimum-cell-size",minimumCellSize_cells:1,maximumCellSize_cells:1,
   min_m:{x:-.4+.05*x0,y:.7,z:.2},max_m:{x:-.4+.05*(x0+2),y:.8,z:.3}});
  const inRegion=(x0:number)=>(t:number,d:readonly[number,number,number])=>{const x=t%d[0],y=Math.floor(t/d[0])%d[1],z=Math.floor(t/(d[0]*d[1]));return x>=x0&&x<x0+2&&y>=14&&z>=12&&z<14;};
  const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,uniformGeometricSolverOptions({detailPolicy:"requested"},scene),()=>{});
  try{
   let frame=0;const report:Record<string,unknown>[]=[];
   const step=async(n:number,label:string)=>{for(let i=0;i<n;i++){frame++;
    assert.ok(solver.advanceTo(frame/30),`${label} advance ${frame}`);await solver.awaitFrameCompletion();
    const residual=(await solver.readStats()).uniformPressureAcceptedResidual!;if(!Number.isFinite(residual))failures.push(`${label}: non-finite residual at frame ${frame}`);}};
   const check=async(label:string,x0?:number)=>{
    const o=await owners(device!,solver),total=o.tileDims[0]*o.tileDims[1]*o.tileDims[2],isFine=new Uint8Array(total);o.fineTiles.forEach(t=>isFine[t]=1);
    let inside=0,far=0;
    for(let t=0;t<total;t++){const y=Math.floor(t/o.tileDims[0])%o.tileDims[1];if(x0!==undefined&&inRegion(x0)(t,o.tileDims)){if(isFine[t])inside++;}else if(isFine[t]&&y>=10)far++;}
    report.push({label,frame,fineTiles:o.fineTiles.length,inside,far,mass:o.mass});
    if(x0!==undefined&&inside!==8)failures.push(`${label}: ${inside} of the region's 8 tiles are h`);
    if(far)failures.push(`${label}: ${far} h tiles far from the liquid outside the region`);
    return o;
   };
   await step(3,"contact only");
   const before=await check("contact only");
   // The census must be the one deciding here: contact tiles exist and the region's page is far air.
   assert.ok(before.fineTiles.length>0,"solid contact refined no tile: the census does not run every frame in this scene");
   for(const x0 of [12,11,10]){
    const drawn=structuredClone(scene);drawn.fluid.refinementRegions=[region(x0)];solver.applySceneUniforms(drawn);await solver.pipelinesPrepared();
    await step(3,`region at ${x0}`);
    const o=await check(`region at tile ${x0}`,x0);
    if(!(Math.abs(o.mass-before.mass)<=2e-3*before.mass))failures.push(`region at tile ${x0}: mass ${o.mass} vs ${before.mass}`);
   }
   solver.applySceneUniforms(scene);await solver.pipelinesPrepared();
   await step(3,"region removed");await check("region removed");
   console.log(JSON.stringify(report));
  }finally{solver.destroy();}
  assert.deepEqual(errors,[]);assert.deepEqual(failures,[]);
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
