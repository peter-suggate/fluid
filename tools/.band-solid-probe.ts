// Scratch probe (delete after use): mixed band pressure with static solids.
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import type {SceneDescription} from "../lib/core/model";
import {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {uniformGeometricSolverOptions} from "../lib/methods/uniform/uniform-geometric-options";
const kind=process.argv[2]??"voxel",frames=Number(process.argv[3]??3);
const s=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32"))) as SceneDescription;
s.container.width_m=s.container.height_m=s.container.depth_m=.8;s.voxelDomain.finestCellSize_m=.025;
s.solidVoxels=[];s.rigidBodies=[];s.fluid.inflow=undefined;s.fluid.initialDamBreakDimensions_m={x:.3,y:.4,z:.8};
if(kind==="voxel")s.solidVoxels.push({operation:"fill",minimum:[3,5,10],maximumExclusive:[7,9,14]},{operation:"fill",minimum:[16,0,12],maximumExclusive:[20,4,20]});
if(kind==="terrain")s.terrain={baseHeight_m:.03,features:[{kind:"mound",center_m:{x:.05,z:0},radius_m:{x:.2,z:.25},amount_m:.11,flat:.2}]};
if(process.argv[5]==="coarse")s.fluid.refinementRegions=[{id:"coarse",rule:"minimum-cell-size",minimumCellSize_cells:4,maximumCellSize_cells:4,min_m:{x:-.4,y:0,z:-.4},max_m:{x:.4,y:.8,z:.4}}];
await acquireWebGPUExclusiveLock("dawn-test","band solid probe");let device:GPUDevice|undefined;
try{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();
 device=managedGPUDevice(await adapter!.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter!.limits)}),{requireWorkerRealm:false});
 device.addEventListener("uncapturederror",e=>{e.preventDefault();console.log("GPU ERROR",e.error.message.slice(0,2000));});
 const t0=performance.now();const solver=await WebGPUUniformReferenceSolver.createAsync(device,s,"balanced",undefined,{...uniformGeometricSolverOptions({},s),mixedOwnership:true},()=>{});
 try{
  console.log(JSON.stringify({created_s:(performance.now()-t0)/1000}));
  const fr=(solver as any).mixedFrame;let probeRhs:GPUBuffer|undefined;
  {const L=fr.levels[0],tiles=L.ownership.layout.tiles.length;probeRhs=device!.createBuffer({size:8*tiles,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
   var probeSol=device!.createBuffer({size:L.phi.size+4*tiles,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
   const origBand=fr.band.encodeSolve.bind(fr.band);fr.band.encodeSolve=(e:GPUCommandEncoder)=>{e.copyBufferToBuffer(L.pressure.buffer,L.pressure.offset??0,probeSol!,0,4*tiles);e.copyBufferToBuffer(L.phi.buffer,L.phi.offset??0,probeSol!,4*tiles,L.phi.size);origBand(e);};
   const orig=fr.cycles.encodeSurfaceRestriction.bind(fr.cycles);fr.cycles.encodeSurfaceRestriction=(e:GPUCommandEncoder)=>{e.copyBufferToBuffer(L.rhs[0].buffer,L.rhs[0].offset??0,probeRhs!,0,4*tiles);e.copyBufferToBuffer(L.minimum[0].buffer,L.minimum[0].offset??0,probeRhs!,4*tiles,4*tiles);orig(e);};}
  for(let i=1;i<=frames;i++){
   const t1=performance.now();solver.advanceTo(i/30);await solver.awaitFrameCompletion();const st=await solver.readStats();
   const frame=(solver as unknown as {mixedFrame:{bandTiles?:number;bandResidual?:number}}).mixedFrame;
   console.log(JSON.stringify({frame:i,residual:st.uniformPressureAcceptedResidual,bandTiles:frame.bandTiles,bandResidual:frame.bandResidual,fine:solver.info.uniformMixedFineTiles,coarse:solver.info.uniformMixedCoarseTiles,seconds:(performance.now()-t1)/1000}));
  }
  const frame=(solver as unknown as {mixedFrame:any}).mixedFrame;
  const read=async(buffer:GPUBuffer,offset:number,size:number)=>{const b=device!.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const e=device!.createCommandEncoder();e.copyBufferToBuffer(buffer,offset,b,0,size);device!.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);const f=new Float32Array(b.getMappedRange().slice(0));b.destroy();return f;};
  {const tiles=frame.levels[0].ownership.layout.tiles.length;const p=await read(frame.levels[0].pressure.buffer,frame.levels[0].pressure.offset??0,4*tiles);const phi=await read(frame.levels[0].phi.buffer,frame.levels[0].phi.offset??0,4*tiles);
   let pmax=0,liquid=0;for(let t=0;t<tiles;t++){pmax=Math.max(pmax,p[t]!);if(phi[t]!<0)liquid++;}console.log(JSON.stringify({pmax4h:pmax,liquid4h:liquid}));}
  {const L=frame.levels[0],tiles=L.ownership.layout.tiles.length;
   const rhs=await read(probeRhs!,0,probeRhs!.size);let nz=0,sum=0;for(let t=0;t<tiles;t++){if(rhs[t]!==0)nz++;sum+=Math.abs(rhs[t]!);}console.log(JSON.stringify({rhsNonzero:nz,rhsAbs:sum}));const sol=await read(probeSol!,0,probeSol!.size);const p=sol,phiAll=sol.subarray(tiles);
   const base=Math.ceil(tiles*4/256)*64;
   for(const x of [1,5])for(let y=0;y<8;y++){const t=x+8*(y+8*4);console.log(JSON.stringify({x,y,t,p:p[t],phi:phiAll[t],rhs:rhs[t],min:rhs[tiles+t],top:[...phiAll.slice(base+4*t,base+4*t+4)]}));}}
  if(frame.solid?.coarse){
   const c=frame.solid.coarse,tiles=frame.levels[0].ownership.layout.tiles.length;
   const r=await read(c.record,0,c.record.size);
   let partial=0,closed=0,cut=0,vlow=0;for(let t=0;t<tiles;t++){const o=r[4*t]!;if(o<1)partial++;if(o<=1e-5)closed++;if(r[4*(c.count+t)]!>0.5)cut++;for(let a=1;a<4;a++)if(r[4*t+a]!<0.99999)vlow++;}
   const phi=await read(frame.levels[0].phi.buffer,frame.levels[0].phi.offset??0,4*tiles);const p=await read(frame.levels[0].pressure.buffer,frame.levels[0].pressure.offset??0,4*tiles);
   let liquid=0,pmax=0;for(let t=0;t<tiles;t++){if(phi[t]!<0)liquid++;pmax=Math.max(pmax,p[t]!);}
   const topo=await read(frame.solidTopology.buffer,frame.solidTopology.offset,frame.solidTopology.size);
   let same=true;for(let i=0;i<topo.length;i++)if(topo[i]!==r[i])same=false;
   console.log(JSON.stringify({tiles,partial,closed,cut,vlow,liquid4h:liquid,pmax,topologyMatchesRecord:same,sample:[...r.slice(0,16)]}));
  }
 }finally{solver.destroy();}
}finally{device?.destroy();await releaseWebGPUExclusiveLock();}
