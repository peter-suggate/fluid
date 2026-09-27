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
import {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {uniformGeometricSolverOptions} from "../lib/methods/uniform/uniform-geometric-options";
import {UniformMixedFrame,type UniformMixedFrameFields} from "../lib/methods/uniform/uniform-mixed-frame";
import {createUniformMixedLayout,mixedCellWidth} from "../lib/methods/uniform/uniform-mixed-layout";
import type {WebGPUUniformPressureMultigrid} from "../lib/methods/uniform/webgpu-uniform-pressure-multigrid";
import type {WebGPUUniformVelocityExtrapolator} from "../lib/methods/uniform/webgpu-uniform-velocity-extrapolation";
import {readMixedTexture} from "./helpers/uniform-mixed-native-fields";
const modulePath=process.env.WEBGPU_NODE_MODULE;
const profile=process.env.FLUID_MIXED_FRAME_PROFILE==="1";
(modulePath?test:test.skip)("unified frame advances the native fields without additional dense allocations",{timeout:240000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform unified frame");let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined,frame:UniformMixedFrame|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits),requiredFeatures:profile?["timestamp-query"]:[]}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-64")));
  const perf=process.env.FLUID_MIXED_FRAME_PERF==="1";
  if(!perf){scene.container.width_m=scene.container.height_m=scene.container.depth_m=.8;scene.voxelDomain.finestCellSize_m=.025;scene.solidVoxels=[];scene.rigidBodies=[];scene.fluid.initialDamBreakDimensions_m={x:.2,y:.4,z:.4};}
  // Correctness exercises default cleanup; the historical performance
  // diagnostic keeps its matched cleanup-disabled settings.
  const options=uniformGeometricSolverOptions(perf?{volumeDustThreshold:0,orphanDustThreshold:0}:{},scene);
  if(!perf)assert.ok(options.volumeDustThreshold!>0&&options.orphanDustThreshold!>options.volumeDustThreshold!,"correctness fixture must exercise default cleanup");
  solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{...options,volumePages:16},()=>{});
  if(perf){
   for(let i=1;i<=4;i++){assert.ok(solver.advanceTo(i/30));await solver.awaitFrameCompletion();}
   const start=performance.now();for(let i=5;i<=12;i++){assert.ok(solver.advanceTo(i/30));await solver.awaitFrameCompletion();}
   console.log({nativeMsPerFrame:(performance.now()-start)/8});
   solver.destroy();
   solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{...options,volumePages:16},()=>{});
  }
  const s=solver as unknown as {scratchArena:UniformMixedFrameFields["arena"];conditioningScratch:GPUBuffer;volumeA:GPUTexture;volumeB:GPUTexture;velocityA:GPUTexture;velocityB:GPUTexture;velocityD:GPUTexture;boundaryVelocityA:GPUBuffer;boundaryVelocityB:GPUBuffer;boundaryVelocityD:GPUBuffer;vertexPhiField:GPUTexture;vertexPhiScratch:GPUTexture;surfaceA:GPUTexture;surfaceB:GPUTexture;gammaA:GPUTexture;gammaB:GPUTexture;pressureMultigrid:WebGPUUniformPressureMultigrid;velocityExtrapolator:WebGPUUniformVelocityExtrapolator;pressureMultigridGroup:GPUBindGroup;writeParams(dt:number,bodies:number,inflow:number):void;initializeVolumeAndTerrain():void};
  const scalar=()=>device!.createTexture({size:[s.gammaB.width,s.gammaB.height,s.gammaB.depthOrArrayLayers],dimension:"3d",format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});
  const fields:UniformMixedFrameFields={arena:s.scratchArena,conditioning:s.conditioningScratch,volume:s.volumeA,volumeScratch:s.volumeB,velocity:s.velocityA,velocityScratch:s.velocityB,departure:s.velocityD,negative:s.boundaryVelocityA,negativeScratch:s.boundaryVelocityB,negativeDeparture:s.boundaryVelocityD,phi:s.vertexPhiField,phiScratch:s.vertexPhiScratch,phase:s.surfaceA,centerPhi:s.surfaceB,target:s.gammaB,correction:s.gammaA,pressure:s.pressureMultigrid.prepareMixedContinuation(),extension:s.velocityExtrapolator.prepareMixedContinuation(),uniformGroup:s.pressureMultigridGroup,pressureGeometry:{target:scalar(),centerPhi:scalar()}};
  const labels:string[]=[];
  const query=profile?device.createQuerySet({type:"timestamp",count:4096}):undefined;
  const timing=profile?device.createBuffer({size:4096*8,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}):undefined;
  const timingRead=profile?device.createBuffer({size:4096*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}):undefined;
  const measured=profile?new Proxy(device,{get(target,key){if(key==="createCommandEncoder")return(desc?:GPUCommandEncoderDescriptor)=>{const encoder=target.createCommandEncoder(desc);return new Proxy(encoder,{get(e,k){if(k==="beginComputePass")return(desc:GPUComputePassDescriptor={})=>{const at=labels.length*2;assert.ok(at+1<4096,"timestamp capacity exceeded");labels.push(desc.label??"unlabelled");return e.beginComputePass({...desc,timestampWrites:{querySet:query!,beginningOfPassWriteIndex:at,endOfPassWriteIndex:at+1}});};const v=Reflect.get(e,k,e);return typeof v==="function"?v.bind(e):v;}});};const v=Reflect.get(target,key,target);return typeof v==="function"?v.bind(target):v;}}):device;
  frame=new UniformMixedFrame(measured,createUniformMixedLayout(refinementRegionLattice(scene),[]),fields,scene.container.top==="open");await frame.initialize();
  const before=await readMixedTexture(device,s.volumeA),mass=before.reduce((a,b)=>a+b,0);
  assert.ok(mass>0,"fixture contains liquid");
  const lattice=refinementRegionLattice(scene);
  const region={id:"coarse",rule:"minimum-cell-size" as const,minimumCellSize_cells:4 as const,maximumCellSize_cells:4 as const,min_m:{x:0,y:0,z:-.4},max_m:{x:.4,y:.8,z:.4}};
  let layouts=[frame.ownership.layout,createUniformMixedLayout(lattice,[region]),createUniformMixedLayout(lattice,[]),createUniformMixedLayout(lattice,[{...region,min_m:{x:-.4,y:0,z:-.4}}]),createUniformMixedLayout(lattice,[])];
  if(perf){
   const mode=process.env.FLUID_MIXED_FRAME_LAYOUT;
   const axes=["x","y","z"] as const;
   const airRegion={...region,id:"single-air-tile",
    min_m:Object.fromEntries(axes.map((a,i)=>[a,lattice.origin_m[a]+(lattice.dimensions[i]!-8)*lattice.cellSize_m[i]!])) as typeof region.min_m,
    max_m:Object.fromEntries(axes.map((a,i)=>[a,lattice.origin_m[a]+(lattice.dimensions[i]!-4)*lattice.cellSize_m[i]!])) as typeof region.max_m};
   const layout=createUniformMixedLayout(lattice,mode==="fine"?[]:[mode==="air"?airRegion:region]);
   if(mode==="air"){
    assert.equal(layout.coarseTiles.length,1);assert.equal(layout.transitionTiles.length,26);
    const d=lattice.dimensions;
    for(let z=d[2]-8;z<d[2]-4;z++)for(let y=d[1]-8;y<d[1]-4;y++)for(let x=d[0]-8;x<d[0]-4;x++)assert.equal(before[x+d[0]*(y+d[1]*z)],0,"coarse tile starts in air");
   }
   layouts=Array.from({length:12},()=>layout);
  }
  const physicalMass=(v:Float32Array)=>{let sum=0;const l=frame!.ownership.layout,d=l.lattice.dimensions,t=l.tileDimensions;l.tiles.forEach((word,tile)=>{const w=mixedCellWidth(word),ox=tile%t[0]*4,oy=Math.floor(tile/t[0])%t[1]*4,oz=Math.floor(tile/(t[0]*t[1]))*4;
   for(let z=0;z<4;z+=w)for(let y=0;y<4;y+=w)for(let x=0;x<4;x+=w)sum+=v[ox+x+d[0]*(oy+y+d[1]*(oz+z))]!*w*w*w;
  });return sum;};
  const maxCanonicalPositiveFaceSpeed=(values:Float32Array)=>{
   const l=frame!.ownership.layout,d=l.lattice.dimensions,t=l.tileDimensions;let maximum=0;
   l.tiles.forEach((word,tile)=>{const w=mixedCellWidth(word),base=[tile%t[0]*4,Math.floor(tile/t[0])%t[1]*4,Math.floor(tile/(t[0]*t[1]))*4];
    for(let z=0;z<4;z+=w)for(let y=0;y<4;y+=w)for(let x=0;x<4;x+=w){const origin=[base[0]!+x,base[1]!+y,base[2]!+z];
     for(let axis=0;axis<3;axis++){const next=[...origin];next[axis]!+=w;
      const nw=next[axis]!>=d[axis]!?w:mixedCellWidth(l.tiles[Math.floor(next[0]!/4)+t[0]*(Math.floor(next[1]!/4)+t[1]*Math.floor(next[2]!/4))]!);
      const patch=Math.min(w,nw),u=(axis+1)%3,v=(axis+2)%3;
      for(let j=0;j<w;j+=patch)for(let i=0;i<w;i+=patch){const p=[...origin];p[axis]!+=w-1;p[u]!+=i;p[v]!+=j;
       maximum=Math.max(maximum,Math.abs(values[4*(p[0]!+d[0]*(p[1]!+d[1]*p[2]!))+axis]!));}
     }
    }
   });return maximum;
  };
  const frameTimes:number[]=[];
  let expectedMass=mass;
  let lastParameters:Parameters<UniformMixedFrame["advance"]>[0]|undefined;
  for(let step=0;step<layouts.length;step++){
   if(step===0||!perf)frame.updateLayout(layouts[step]!);
   const remappedValues=await readMixedTexture(device,s.volumeA);const remapped=physicalMass(remappedValues);
   assert.deepEqual(errors,[]);
   assert.ok(Math.abs(remapped-expectedMass)/mass<1e-4,`remap mass drift ${remapped}/${expectedMass}`);
   s.writeParams(1/30,0,0);
   labels.length=0;const started=performance.now();
   const parameters={dt:1/30,gravity:scene.fluid.gravity_m_s2.y,density:scene.fluid.density_kg_m3,viscosity:scene.fluid.dynamicViscosity_Pa_s,surfaceTension:scene.fluid.surfaceTension_N_m,openTop:scene.container.top==="open",noSlip:scene.container.fluidWallMode==="no-slip",cubic:true,drain:true,dust:options.volumeDustThreshold!,orphanDust:options.orphanDustThreshold!,sharpeningStrength:options.sharpeningStrength!,sharpeningDistance:options.sharpeningDistance!,pressureTolerance:5};
   lastParameters=parameters;const receipt=await frame.advance(parameters);
   const frameMs=performance.now()-started;
   if(perf&&step>=4)frameTimes.push(frameMs);
   if(profile&&step===4){const e=device.createCommandEncoder();e.resolveQuerySet(query!,0,labels.length*2,timing!,0);e.copyBufferToBuffer(timing!,0,timingRead!,0,labels.length*16);device.queue.submit([e.finish()]);await timingRead!.mapAsync(GPUMapMode.READ);const times=new BigUint64Array(timingRead!.getMappedRange()),totals=new Map<string,number>();labels.forEach((label,i)=>totals.set(label,(totals.get(label)??0)+Number(times[i*2+1]!-times[i*2]!)/1e6));console.log({stages:[...totals].sort((a,b)=>b[1]-a[1]).slice(0,18)});timingRead!.unmap();}
   assert.deepEqual(errors,[]);assert.ok(Number.isFinite(receipt.residual));
   const volume=await readMixedTexture(device,s.volumeA),velocity=await readMixedTexture(device,s.velocityA),phi=await readMixedTexture(device,s.vertexPhiField);
   assert.ok(volume.every(Number.isFinite)&&velocity.every(Number.isFinite)&&phi.every(Number.isFinite),"finite complete frame");
   expectedMass-=receipt.dustMass_cells;
   const total=physicalMass(volume);assert.ok(Math.abs(total-expectedMass)/mass<1e-4,`unaccounted mass drift ${total}/${expectedMass}`);
   console.log({step,frameMs,...receipt,mass:total,maxCanonicalPositiveFaceSpeed:maxCanonicalPositiveFaceSpeed(velocity)});
  }
  if(!perf){
   const previousVelocity=await readMixedTexture(device,s.velocityA);
   await assert.rejects(frame.advance({...lastParameters!,pressureTolerance:1e-12}),/projection withheld/);
   assert.deepEqual(await readMixedTexture(device,s.velocityA),previousVelocity,"failed pressure must not publish velocity");
   await assert.rejects(frame.advance(lastParameters!),/not ready/,"a partially advanced failed frame requires reset");
  }
  if(perf)console.log({mixedMsPerFrame:frameTimes.reduce((a,b)=>a+b,0)/frameTimes.length,profile,cleanup:"disabled in both arms; diagnostic only",owners:frame.ownership.layout.cellCount,additionalFrameBytes:frame.allocatedBytes,nativeReportedBytes:solver.info.allocatedBytes});
  query?.destroy();timing?.destroy();timingRead?.destroy();
 }finally{frame?.destroy();solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
