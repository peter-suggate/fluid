import assert from "node:assert/strict";
import test, { before, after } from "node:test";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { solidVoxelShellForScene } from "../lib/core/scene-lattice";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { resolveUniformGeometricValues } from "../lib/methods/uniform/uniform-geometric-parameters";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";

async function read(device:GPUDevice,t:GPUTexture):Promise<Float32Array>{
 const components=t.format==="rgba32float"?4:1,row=Math.ceil(t.width*components*4/256)*256;
 const b=device.createBuffer({size:row*t.height*t.depthOrArrayLayers,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 try{const e=device.createCommandEncoder();e.copyTextureToBuffer({texture:t},{buffer:b,bytesPerRow:row,rowsPerImage:t.height},[t.width,t.height,t.depthOrArrayLayers]);device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);
 const mapped=new Float32Array(b.getMappedRange()),out=new Float32Array(t.width*t.height*t.depthOrArrayLayers*components);
 for(let z=0;z<t.depthOrArrayLayers;z++)for(let y=0;y<t.height;y++)out.set(mapped.subarray((z*t.height+y)*row/4,(z*t.height+y)*row/4+t.width*components),(z*t.height+y)*t.width*components);
 return out;}finally{if(b.mapState==="mapped")b.unmap();b.destroy();}
}
const sum=(a:Float32Array)=>a.reduce((s,v)=>s+v,0);
const modulePath=process.env.WEBGPU_NODE_MODULE;
let ownsLease=false;
before(async()=>{if(modulePath){await acquireWebGPUExclusiveLock("dawn-test","Uniform runtime coarse proof");ownsLease=true;}});
after(async()=>{if(ownsLease)await releaseWebGPUExclusiveLock();});
for(const pressureCycleBudget of ["fixed","lagged"] as const)
(modulePath?test:test.skip)(`live h -> 4h -> h preserves evolved liquid without allocation (${pressureCycleBudget} pressure)`,{timeout:1200000},async()=>{
 let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  const raw=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
  let watching=false,liveBytes=0,fineBytes=0;const allocations:string[]=[];
  for(const name of ["createBuffer","createTexture","createShaderModule","createComputePipeline","createComputePipelineAsync"] as const){
   const original=(raw[name] as Function).bind(raw);
   Object.defineProperty(raw,name,{configurable:true,writable:true,value:(...args:unknown[])=>{if(watching)allocations.push(`${name}: ${(args[0] as {label?:string})?.label}`);const result=original(...args);
    if(name==="createBuffer" || name==="createTexture") {
      const bytes=name==="createBuffer"?result.size:result.width*result.height*result.depthOrArrayLayers*({r32float:4,rg32float:8,rgba32float:16,r32uint:4,rgba32uint:16,rgba8unorm:4} as Record<string,number>)[result.format]!;
      assert.ok(Number.isFinite(bytes),`accounted resource format ${result.format}`);liveBytes+=bytes;
      const destroy=result.destroy.bind(result);let dead=false;
      Object.defineProperty(result,"destroy",{configurable:true,value:()=>{if(!dead){liveBytes-=bytes;dead=true;}destroy();}});
    }
    return result;}});
  }
  device=managedGPUDevice(raw,{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-64")));
  const n=Number(process.env.UNIFORM_COARSE_TEST_SIZE??64),cn=n/4;
  assert.ok(n===64 || n===128,"proof sizes are 64 or 128");
  scene.voxelDomain.finestCellSize_m*=64/n;scene.nominalResolution.length_m*=64/n;
  scene.solidVoxels=[...solidVoxelShellForScene(scene)];
  const options=uniformGeometricSolverOptions({},scene),pressureTolerance=options.pressureSchedule!.residualTolerance!;
  solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,
   {...options,prepareCoarseSimulation:true,pressureCycleBudget},snapshot=>{if(snapshot.taskId==="uniform.warmup")fineBytes=liveBytes;});
  const preparedBytes=liveBytes,actualAdditionalBytes=preparedBytes-fineBytes;
  assert.ok(actualAdditionalBytes/fineBytes<=.03,`actual GPU overhead ${actualAdditionalBytes/fineBytes}`);
  assert.equal(solver.coarseSimulationPrepared,true);assert.equal(solver.simulationCellScale,1);
  const fineTexture=solver.volumeTexture;
  const initial=await read(device,fineTexture);
  const fineTimes:number[]=[];
  for(let frame=1;frame<=6;frame++){const start=performance.now();assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();fineTimes.push(performance.now()-start);}
  const fine=await read(device,solver.volumeTexture);
  assert.ok(fine.some((v,i)=>Math.abs(v-initial[i]!)>1e-4),"fine simulation genuinely evolved before switching");
  const beforeTime=solver.info.simulatedTime_s,beforeSteps=solver.info.encodedSteps;
  watching=true;
  const transitionStart=performance.now();solver.requestCoarseSimulation();
  assert.equal(solver.advanceTo(beforeTime!),false,"request commits without advancing the clock");await solver.awaitFrameCompletion();
  const transitionMs=performance.now()-transitionStart;
  watching=false;
  assert.deepEqual(allocations,[],"transition does not allocate or compile");
  assert.equal(solver.simulationCellScale,4);assert.equal(solver.info.nx,cn);
  assert.equal(solver.info.simulatedTime_s,beforeTime);assert.equal(solver.info.encodedSteps,beforeSteps);
  assert.notEqual(solver.volumeTexture,fineTexture);assert.equal(solver.vertexPhiTexture!.width,cn+1);
  const coarse=await read(device,solver.volumeTexture);
  let maxRestrictionError=0;
  for(let z=0;z<cn;z++)for(let y=0;y<cn;y++)for(let x=0;x<cn;x++){
   let expected=0;for(let k=0;k<4;k++)for(let j=0;j<4;j++)for(let i=0;i<4;i++)expected+=fine[4*x+i+n*(4*y+j+n*(4*z+k))]!;
   maxRestrictionError=Math.max(maxRestrictionError,Math.abs(coarse[x+cn*(y+cn*z)]!-expected/64));
  }
  assert.ok(maxRestrictionError<2e-6,`per-cell conservative restriction: ${maxRestrictionError}`);
  assert.ok(Math.abs(sum(coarse)*64-sum(fine))/sum(fine)<2e-6,"physical volume conserved at switch");
  watching=true;const transitionStats=await solver.readStats();watching=false;
  assert.deepEqual(allocations,[],"coarse diagnostics were also prepared at startup");
  const transitionResidual=transitionStats.uniformPressureAcceptedResidual;
  assert.ok(Number.isFinite(transitionResidual) && transitionResidual!<=pressureTolerance,`transfer projection residual ${transitionResidual}`);
  const velocity=await read(device,solver.velocityTexture);assert.ok(velocity.every(Number.isFinite));assert.ok(velocity.some(v=>Math.abs(v)>1e-3),"moving water stays in motion");
  const coarseTimes:number[]=[];watching=true;
  for(let frame=7;frame<=12;frame++){const start=performance.now();assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();coarseTimes.push(performance.now()-start);}
  watching=false;assert.deepEqual(allocations,[],"first and subsequent coarse steps do not allocate or compile");
  assert.equal(solver.info.encodedSteps,12);assert.ok(Math.abs(solver.info.simulatedTime_s!-.4)<1e-6);
  const evolved=await read(device,solver.volumeTexture);
  assert.ok(evolved.every(Number.isFinite));assert.ok(Math.abs(sum(evolved)-sum(coarse))/sum(coarse)<1e-5,"coarse evolution retains liquid volume");assert.ok(evolved.some((v,i)=>Math.abs(v-coarse[i]!)>1e-5),"coarse physics evolves transferred state");
  const frozenFine=await read(device,fineTexture);assert.deepEqual(frozenFine,fine,"inactive fine state is not stepped or synchronized");
  const returnStart=performance.now();
  watching=true;solver.requestFineSimulation();
  assert.equal(solver.advanceTo(.4),false,"return to fine consumes no timestep");
  await solver.awaitFrameCompletion();watching=false;
  const returnMs=performance.now()-returnStart;
  assert.deepEqual(allocations,[],"return to fine does not allocate or compile");
  assert.equal(solver.simulationCellScale,1);assert.equal(solver.volumeTexture,fineTexture);
  assert.equal(solver.info.encodedSteps,12);assert.ok(Math.abs(solver.info.simulatedTime_s!-.4)<1e-6);
  const restored=await read(device,solver.volumeTexture);
  let maxProlongationError=0;
  for(let z=0;z<n;z++)for(let y=0;y<n;y++)for(let x=0;x<n;x++)
    maxProlongationError=Math.max(maxProlongationError,Math.abs(restored[x+n*(y+n*z)]!-evolved[Math.floor(x/4)+cn*(Math.floor(y/4)+cn*Math.floor(z/4))]!));
  assert.ok(maxProlongationError<2e-6,"every fine V cell comes from current coarse liquid");
  assert.ok(Math.abs(sum(restored)-64*sum(evolved))/sum(restored)<2e-6,"return conserves physical liquid volume");
  const restoredVelocity=await read(device,solver.velocityTexture);
  assert.ok(restoredVelocity.every(Number.isFinite));assert.ok(restoredVelocity.some(v=>Math.abs(v)>1e-3));
  watching=true;const restoredStats=await solver.readStats();watching=false;
  const returnResidual=restoredStats.uniformPressureAcceptedResidual;
  assert.ok(Number.isFinite(returnResidual) && returnResidual!<=pressureTolerance,`fine transfer projection residual ${returnResidual}`);
  watching=true;
  for(let frame=13;frame<=18;frame++){assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();}
  watching=false;
  const fineEvolved=await read(device,solver.volumeTexture);
  assert.ok(fineEvolved.some((v,i)=>Math.abs(v-restored[i]!)>1e-5),"reconstructed fine simulation continues evolving");
  const roundTripMass=sum(fineEvolved);
  const runtimeValues={...resolveUniformGeometricValues({prepareCoarseSimulation:"on"}),pressureCycleBudget};
  watching=true;
  for(let repeat=0;repeat<3;repeat++){
    solver.applyRuntimeValues({...runtimeValues,coarseSimulation:"on"});await solver.awaitFrameCompletion();
    assert.equal(solver.simulationCellScale,4);
    solver.applyRuntimeValues({...runtimeValues,coarseSimulation:"off"});await solver.awaitFrameCompletion();
    assert.equal(solver.simulationCellScale,1);
  }
  watching=false;
  assert.deepEqual(allocations,[],"fine advances and repeated toggles do not allocate or compile");
  const repeated=await read(device,solver.volumeTexture);
  assert.ok(Math.abs(sum(repeated)-roundTripMass)/roundTripMass<2e-6,"repeated toggles conserve volume");
  assert.equal(solver.info.encodedSteps,18);
  assert.ok(Math.abs(solver.info.simulatedTime_s!-.6)<1e-6,"paused UI toggles consume no simulation time");
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({size:n,pressureCycleBudget,pressureTolerance,transitionResidual,returnResidual,returnMs,maxProlongationError,fineBytes,preparedBytes,actualAdditionalBytes,actualMemoryOverhead:actualAdditionalBytes/fineBytes,fineTimes,coarseTimes,transitionMs,maxRestrictionError,restrictionMassRelative:Math.abs(sum(coarse)*64-sum(fine))/sum(fine),coarseEvolutionMassRelative:Math.abs(sum(evolved)-sum(coarse))/sum(coarse),allocatedBytes:solver.info.allocatedBytes,coarseAdditionalBytes:solver.coarseAdditionalBytes}));
 }finally{solver?.destroy();device?.destroy();}
});

(modulePath?test:test.skip)("restriction and prolongation preserve nonuniform volume, MAC face flux and physical phi units",{timeout:120000},async()=>{
 let device:GPUDevice|undefined;const resources:(GPUTexture|GPUBuffer)[]=[];
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const {UniformResolutionRestriction,UniformResolutionProlongation}=await import("../lib/methods/uniform/uniform-resolution-transfer");
  const fields=(n:number)=>{
   const texture=(size:number,format:GPUTextureFormat)=>{const t=device!.createTexture({size:[size,size,size],dimension:"3d",format,usage:GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});resources.push(t);return t;};
   const negativeFaces=device!.createBuffer({size:3*n*n*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});resources.push(negativeFaces);
   return {volume:texture(n,"r32float"),velocity:texture(n,"rgba32float"),phi:texture(n+1,"r32float"),negativeFaces};
  };
  const fine=fields(8),coarse=fields(2),v=new Float32Array(512),u=new Float32Array(2048),phi=new Float32Array(729),boundary=Float32Array.from({length:192},(_,i)=>(i-96)/128);
  for(let z=0;z<8;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++){
   const i=x+8*(y+8*z);v[i]=((x+3*y+5*z)%64)/64;
   for(let axis=0;axis<3;axis++)u[4*i+axis]=(3*x+5*y+7*z+11*axis)/128;
  }
  for(let z=0;z<9;z++)for(let y=0;y<9;y++)for(let x=0;x<9;x++)phi[x+9*(y+9*z)]=(x+2*y-3*z)/16;
  const write=(t:GPUTexture,data:Float32Array<ArrayBuffer>,c:number)=>device!.queue.writeTexture({texture:t},data,{bytesPerRow:t.width*4*c,rowsPerImage:t.height},[t.width,t.height,t.depthOrArrayLayers]);
  write(fine.volume,v,1);write(fine.velocity,u,4);write(fine.phi,phi,1);device.queue.writeBuffer(fine.negativeFaces,0,boundary);
  const restriction=await UniformResolutionRestriction.create(device,fine,coarse);
  const encoder=device.createCommandEncoder();restriction.encode(encoder);device.queue.submit([encoder.finish()]);
  const cv=await read(device,coarse.volume),cu=await read(device,coarse.velocity),cp=await read(device,coarse.phi);
  const staging=device.createBuffer({size:48,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});resources.push(staging);
  const copy=device.createCommandEncoder();copy.copyBufferToBuffer(coarse.negativeFaces,0,staging,0,48);device.queue.submit([copy.finish()]);await staging.mapAsync(GPUMapMode.READ);const cb=new Float32Array(staging.getMappedRange()).slice();staging.unmap();
  const index=(p:number[],axis:number,n:number)=>axis===0?p[1]!+n*p[2]!:axis===1?n*n+p[0]!+n*p[2]!:2*n*n+p[0]!+n*p[1]!;
  for(let z=0;z<2;z++)for(let y=0;y<2;y++)for(let x=0;x<2;x++){
   const id=[x,y,z],i=x+2*(y+2*z);let volume=0;
   for(let dz=0;dz<4;dz++)for(let dy=0;dy<4;dy++)for(let dx=0;dx<4;dx++)volume+=v[4*x+dx+8*(4*y+dy+8*(4*z+dz))]!;
   assert.equal(cv[i],volume/64);
   for(let axis=0;axis<3;axis++){
    let flux=0,negative=0;
    for(let b=0;b<4;b++)for(let a=0;a<4;a++){
     const p=id.map(n=>4*n);p[axis]!+=3;p[(axis+1)%3]!+=a;p[(axis+2)%3]!+=b;
     flux+=u[4*(p[0]!+8*(p[1]!+8*p[2]!))+axis]!;
     negative+=boundary[index(p,axis,8)]!;
    }
    assert.equal(cu[4*i+axis],flux/16);
    if(id[axis]===0)assert.equal(cb[index(id,axis,2)],negative/16);
   }
  }
  for(let z=0;z<3;z++)for(let y=0;y<3;y++)for(let x=0;x<3;x++)assert.equal(cp[x+3*(y+3*z)],phi[4*x+9*(4*y+9*4*z)]);
  const prolongation=await UniformResolutionProlongation.create(device,coarse,fine);
  const prolong=device.createCommandEncoder();prolongation.encode(prolong);device.queue.submit([prolong.finish()]);
  const fv=await read(device,fine.volume),fu=await read(device,fine.velocity),fp=await read(device,fine.phi);
  for(let z=0;z<8;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++){
    const id=[x,y,z],parent=id.map(n=>Math.floor(n/4)),ci=parent[0]!+2*(parent[1]!+2*parent[2]!),fi=x+8*(y+8*z);
    assert.equal(fv[fi],cv[ci]);
    for(let axis=0;axis<3;axis++){
      const left=parent[axis]===0?cb[index(parent,axis,2)]!:cu[4*(ci-2**axis)+axis]!;
      const right=cu[4*ci+axis]!,weight=(id[axis]!%4+1)/4;
      assert.equal(fu[4*fi+axis],left+(right-left)*weight,"normal MAC interpolation preserves parent face flux");
    }
  }
  assert.deepEqual(fp,phi,"trilinear prolongation reproduces a linear physical-distance field");
  const roundTrip=device.createCommandEncoder();restriction.encode(roundTrip);device.queue.submit([roundTrip.finish()]);
  assert.deepEqual(await read(device,coarse.volume),cv,"prolong/restrict preserves every parent V");
  assert.deepEqual(await read(device,coarse.velocity),cu,"prolong/restrict preserves each positive face flux");
  const boundaryCopy=device.createCommandEncoder();boundaryCopy.copyBufferToBuffer(coarse.negativeFaces,0,staging,0,48);device.queue.submit([boundaryCopy.finish()]);
  await staging.mapAsync(GPUMapMode.READ);assert.deepEqual(new Float32Array(staging.getMappedRange()).slice(),cb,"negative boundary flux round trip");staging.unmap();

 }finally{for(const r of resources)r.destroy();device?.destroy();}
});
