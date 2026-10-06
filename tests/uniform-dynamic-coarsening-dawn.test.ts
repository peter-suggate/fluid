import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {refinementRegionLattice} from "../lib/core/refinement-regions";
import {uniformVolumeMethod} from "../lib/methods/uniform/uniform-volume-method";
import type {GPUSolverInstance} from "../lib/core/method-contract";
import {UNIFORM_ADVANCE_PHASE} from "../lib/methods/uniform/uniform-stages";
import {uniformDetailField} from "../lib/methods/uniform/uniform-detail-fields";

// Phase 5 lane (docs/plans/uniform-dynamic-coarsening.md): the app's
// Uniform Geometric method with detailPolicy=dynamic on the 128³ dam break.
// The simulation layout is h/4h only, ownership
// must actually follow the flow, volume must be conserved, and a relayout
// must not compile anything or grow memory past the band's high-water mark.
// Surface detail stays at h; smooth surface, regular bulk and air can use 4h.
const STEPS=Number(process.env.UNIFORM_DYNAMIC_LANE_STEPS??30);
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("dynamic coarsening follows the 128³ dam break on h/4h ownership without drift or recompilation",{timeout:1200000},async t=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform dynamic coarsening lane");
 let device:GPUDevice|undefined,solver:GPUSolverInstance|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  const raw=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
  let watching=false;const allocations:string[]=[];
  for(const name of ["createBuffer","createTexture","createShaderModule","createComputePipeline","createComputePipelineAsync"] as const){
   const original=(raw[name] as Function).bind(raw);
   Object.defineProperty(raw,name,{configurable:true,writable:true,value:(...args:unknown[])=>{
    if(watching)allocations.push(`${name}: ${(args[0] as {label?:string})?.label} ${(args[0] as {size?:number})?.size??""}`);
    // The ownership words are read back below to find refined tiles.
    const d=args[0] as GPUBufferDescriptor;
    if(name==="createBuffer"&&d.label?.startsWith("Uniform mixed owners"))return original({...d,usage:d.usage|GPUBufferUsage.COPY_SRC});
    return original(...args);}});
  }
  device=managedGPUDevice(raw,{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const scene=structuredClone(sceneDocument(getSceneDefinition("high-resolution-dam-break")));scene.fluid.refinementRegions=[];
  assert.deepEqual(refinementRegionLattice(scene).dimensions,[128,128,128]);
  solver=await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",{pressureResidualTolerance:5,detailPolicy:"dynamic"},undefined,()=>{});
  const info=solver.info as unknown as Record<string,unknown>;
  // A refined tile must carry exactly the surface its 4h corners described:
  // every stored vertex of its closure is their trilinear interpolant.
  const findHost=(o:any,depth=0,seen=new Set<unknown>()):any=>{if(!o||typeof o!=="object"||depth>4||seen.has(o))return undefined;seen.add(o);if(o.mixedFrame&&o.vertexPhiField)return o;for(const k of Object.keys(o)){const r=findHost(o[k],depth+1,seen);if(r)return r;}};
  const host=findHost(solver);assert.ok(host,"mixed Uniform host");
  const T=32,N=128,h=refinementRegionLattice(scene).cellSize_m[0]!;
  const readback=async(copy:(e:GPUCommandEncoder,b:GPUBuffer)=>void,size:number)=>{
   const b=device!.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const e=device!.createCommandEncoder();copy(e,b);device!.queue.submit([e.finish()]);
   await b.mapAsync(GPUMapMode.READ);const out=b.getMappedRange().slice(0);b.unmap();b.destroy();return out;};
  const widths=async()=>Uint8Array.from(new Uint32Array(await readback((e,b)=>e.copyBufferToBuffer(host.mixedFrame.ownership.presentation.buffer,0,b,0,T*T*T*4),T*T*T*4)),w=>(w&0x80000000)?1:4);
  // Horizon one adopts and remaps at the frame head, and the same frame then
  // transports phi at h. The refine invariant is checked where it holds:
  // right after the remap (the head's census phase), before any transport.
  // Logical vertex phi: the detail storage places the field (base block + patch atlas).
  const vt=host.vertexPhiField as GPUTexture,detail=uniformDetailField(vt)!.storage;
  let headPhi:(()=>Promise<Float32Array>)|undefined;
  const headOwners=device.createBuffer({size:T*T*T*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  let headCaptured=false;
  host.mixedFrameTrace=()=>({instrument:(e:GPUCommandEncoder)=>e,submit(){},submitted(){},abort(){},
   phase(e:GPUCommandEncoder,phase:unknown){
    if(headCaptured||phase!==UNIFORM_ADVANCE_PHASE.resolutionCensus)return;headCaptured=true;
    headPhi=detail.capture(e,vt);
    e.copyBufferToBuffer(host.mixedFrame.ownership.presentation.buffer,0,headOwners,0,T*T*T*4);}});
  const headWidths=async()=>Uint8Array.from(new Uint32Array(await readback((e,b)=>e.copyBufferToBuffer(headOwners,0,b,0,T*T*T*4),T*T*T*4)),w=>(w&0x80000000)?1:4);
  const headPhiAt=async()=>{const raw=await headPhi!();return (x:number,y:number,z:number)=>raw[(z*(N+1)+y)*(N+1)+x]!;};
  let previous:Uint8Array|undefined,refinedTiles=0;
  let relayouts=0,coarsest=Infinity,finest=0;const wall:number[]=[],shapes:string[]=[];
  for(let step=1;step<=STEPS;step++){
   // Watch from the third frame: the first frames grow the band, the tap
   // cache and the pressure records to their working sizes.
   watching=step>2;
   headCaptured=false;
   const start=performance.now();assert.ok(solver.advanceTo(step/30,[]));await solver.awaitFrameCompletion?.();wall.push(performance.now()-start);
   await solver.readStats();
   const fine=Number(info.uniformMixedFineTiles),four=Number(info.uniformMixedCoarseTiles);
   assert.equal(fine+four,32768,`step ${step}: tile counts`);
   relayouts=Number(info.uniformMixedDynamicRelayouts);coarsest=Math.min(coarsest,fine);finest=Math.max(finest,fine);
   shapes.push(`${fine}/${four}`);
   assert.ok(Math.abs(Number(info.volumeDrift))<1e-4,`step ${step}: volume drift ${info.volumeDrift}`);
   const was=watching;watching=false;
   assert.ok(headCaptured,`step ${step}: no frame head census phase`);
   const current=await widths(),head=await headWidths(),P=await headPhiAt();
   for(let t=0;previous&&t<T*T*T;t++){
    if(previous[t]!==4||head[t]!==1)continue;refinedTiles++;
    const o=[t%T,(t>>5)%T,t>>10].map(c=>4*c) as [number,number,number];
    for(let z=0;z<=4;z++)for(let y=0;y<=4;y++)for(let x=0;x<=4;x++){
     const p=[o[0]+x,o[1]+y,o[2]+z] as [number,number,number];
     // Vertices shared with a 4h tile are derived there, not stored.
     let derived=false;
     for(let k=0;k<8;k++){const q=p.map((c,a)=>Math.floor((c-((k>>a)&1))/4));if(q.every(c=>c>=0&&c<T)&&head[q[0]!+T*(q[1]!+T*q[2]!)]===4)derived=true;}
     if(derived||p.some(c=>c>N))continue;
     let want=0;for(let k=0;k<8;k++){let w=1;const c=[0,1,2].map(a=>{const f=(p[a]!-o[a]!)/4,bit=(k>>a)&1;w*=bit?f:1-f;return o[a]!+4*bit;});want+=w*P(c[0]!,c[1]!,c[2]!);}
     assert.ok(Math.abs(P(...p)-want)<=1e-4*h,`step ${step}: refined tile ${t} vertex ${p} phi ${P(...p)/h} h, its 4h surface gives ${want/h} h`);
    }
   }
   previous=current;watching=was;
  }
  assert.ok(refinedTiles>0,"no tile refined from 4h to h");
  t.diagnostic(`${refinedTiles} refined tiles kept their 4h surface`);
  watching=false;
  t.diagnostic(`h/4h tiles per step: ${shapes.join(" ")}`);
  const median=(a:number[])=>[...a].sort((x,y)=>x-y)[a.length>>1]!;
  t.diagnostic(`${relayouts} relayouts; median wall ${median(wall).toFixed(1)} ms (steps 1-${STEPS})`);
  assert.deepEqual(errors,[]);
  // Ownership follows the flow: it starts mostly coarse and refines as the front spreads.
  assert.ok(relayouts>=STEPS/2,`relayouts ${relayouts}`);
  assert.ok(coarsest<32768/2&&finest>coarsest,`fine tiles range ${coarsest}..${finest}`);
  const compiled=allocations.filter(a=>!a.startsWith("createBuffer")&&!a.startsWith("createTexture"));
  assert.deepEqual(compiled,[],"relayout compiled shaders or pipelines");
  t.diagnostic(`allocations after frame 2: ${allocations.length ? allocations.join("; ") : "none"}`);
 }finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});

// A ball dropped into far 4h air: the host refines its footprint for the
// frame that seeds it, and the census then decides its ownership. Its mass
// must arrive whole.
(modulePath?test:test.skip)("dynamic coarsening takes a liquid ball dropped into 4h air",{timeout:600000},async t=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform dynamic coarsening drop");
 let device:GPUDevice|undefined,solver:GPUSolverInstance|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const scene=structuredClone(sceneDocument(getSceneDefinition("high-resolution-dam-break")));scene.fluid.refinementRegions=[];
  solver=await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",{pressureResidualTolerance:5,detailPolicy:"dynamic"},undefined,()=>{});
  const info=solver.info as unknown as Record<string,unknown>;
  for(let step=1;step<=3;step++){assert.ok(solver.advanceTo(step/30,[]));await solver.awaitFrameCompletion?.();}
  const before=Number((await solver.readStats()).volumeCellSum),fineBefore=Number(info.uniformMixedFineTiles);
  const c=scene.container,cell=c.width_m/128,radius=.06*c.width_m;
  solver.injectLiquidBall!({centre_m:{x:.35*c.width_m,y:.5*c.height_m,z:0},radius_m:radius});
  for(let step=4;step<=8;step++){
   assert.ok(solver.advanceTo(step/30,[]));await solver.awaitFrameCompletion?.();await solver.readStats();
   assert.equal(Number(info.uniformMixedFineTiles)+Number(info.uniformMixedCoarseTiles),32768,`step ${step}: tile counts`);
  }
  const added=Number(info.volumeCellSum)-before,expected=4/3*Math.PI*(radius/cell)**3;
  t.diagnostic(`fine tiles ${fineBefore} -> ${info.uniformMixedFineTiles}; added ${added.toFixed(1)} of ${expected.toFixed(1)} cells`);
  assert.ok(Math.abs(added-expected)<.02*expected,`drop added ${added} cells, expected ${expected}`);
  assert.deepEqual(errors,[]);
 }finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
