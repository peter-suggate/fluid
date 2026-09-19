import { createGridOverlayLevelSetVolumeWGSL, gridOverlayLevelSetVolumeUniform } from "../lib/core/grid-overlay-levelset-volume.wgsl";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { resolveMethodValues } from "../lib/core/method-contract";
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";

async function read(device:GPUDevice,texture:GPUTexture):Promise<Float32Array>{
  const components=texture.format==="rgba32float"?4:1;
  const row=Math.ceil(texture.width*components*4/256)*256;
  const buffer=device.createBuffer({size:row*texture.height*texture.depthOrArrayLayers,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try {const e=device.createCommandEncoder();e.copyTextureToBuffer({texture},{buffer,bytesPerRow:row,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);device.queue.submit([e.finish()]);await buffer.mapAsync(GPUMapMode.READ);
    const mapped=new Float32Array(buffer.getMappedRange());const result=new Float32Array(texture.width*texture.height*texture.depthOrArrayLayers*components);
    for(let z=0;z<texture.depthOrArrayLayers;z++)for(let y=0;y<texture.height;y++)result.set(mapped.subarray((z*texture.height+y)*row/4,(z*texture.height+y)*row/4+texture.width*components),(z*texture.height+y)*texture.width*components);
    return result;
  }finally{buffer.unmap();buffer.destroy();}
}
function write(device:GPUDevice,texture:GPUTexture,values:Float32Array){const components=texture.format==="rgba32float"?4:1;device.queue.writeTexture({texture},values as Float32Array<ArrayBuffer>,{bytesPerRow:texture.width*components*4,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);}
const sum=(a:Float32Array)=>a.reduce((s,v)=>s+v,0);
interface TestAccess {
  transportA:GPUTexture;volumeB:GPUTexture;conditioningScratch:GPUBuffer;
  writeParams(dt:number,bodies:number,inflow:number):void;
  encodeGeometricVolume(encoder:GPUCommandEncoder):void;
}
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("uniform geometric numerical invariants and hydrostatic stability",{timeout:180_000},async t=>{
  await acquireWebGPUExclusiveLock("dawn-test","uniform-volume numerical invariants");
  let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
  try{
    const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
    const gpu=createProcessRetainedDawnGPU(dawn as NodeDawnProvider,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
    const adapter=await gpu.requestAdapter();assert.ok(adapter);
    device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
    const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
    const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
    scene.voxelDomain.finestCellSize_m=scene.container.width_m/16;
    scene.fluid.initialCondition="tank-fill";scene.container.fillFraction=0.5;
    scene.fluid.gravity_m_s2={x:0,y:0,z:0};scene.fluid.initialLiquidVolumes=[];
    solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{geometricVolume:true,liquidCapacityBalancing:true,densitySharpening:false,solidExcessCorrection:false},()=>{});
    const access=solver as unknown as TestAccess;const {nx,ny,nz}=solver.info;
    const v0=await read(device,solver.volumeTexture);const phi0=await read(device,solver.vertexPhiTexture!);
    await t.test("combined slice reads independent vertex phi and V/open capacity",async()=>{
      const d=device!; const source=solver!.denseLevelSetVolumeSource!;
      const module=d.createShaderModule({code:`
@group(0) @binding(9) var densityField:texture_3d<f32>;
var<private> sparseTopologyArena:array<u32,1>;
var<private> sparseState:array<f32,1>;
fn sparseOwner(p:vec3i)->vec2u{return vec2u(0xffffffffu);}
fn sparseDensityOffset()->u32{return 0u;}
${createGridOverlayLevelSetVolumeWGSL(true)}
@group(0) @binding(23) var<storage,read_write> result:array<vec4f>;
@compute @workgroup_size(1) fn probe(){result[0]=vec4f(sliceLevelSetPhi(vec3f(3.25,3.5,3.75)),sliceVolumeFill(vec3i(3)));}
`});
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
      const pipeline=await d.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"probe"}});
      const params=d.createBuffer({size:96,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
      d.queue.writeBuffer(params,0,gridOverlayLevelSetVolumeUniform(undefined,source));
      const output=d.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
      const staging=d.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
      const group=d.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[
        {binding:9,resource:solver!.volumeTexture.createView()},
        {binding:20,resource:{buffer:params}}, {binding:21,resource:source.vertexPhi.createView()},
        {binding:22,resource:source.openFraction.createView()}, {binding:23,resource:{buffer:output}},
      ]});
      const e=d.createCommandEncoder();const pass=e.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();e.copyBufferToBuffer(output,0,staging,0,16);d.queue.submit([e.finish()]);
      await staging.mapAsync(GPUMapMode.READ);const result=new Float32Array(staging.getMappedRange());
      assert.ok(Math.abs(result[0]!-(3.5-ny/2))<1e-5);assert.equal(result[1],1);assert.equal(result[2],1);assert.equal(result[3],1);
      staging.unmap();for(const b of [params,output,staging])b.destroy();
    });
    await t.test("zero velocity preserves V and planar phi",async()=>{
      assert.ok(solver!.advanceTo(1/30));await device!.queue.onSubmittedWorkDone();
      assert.deepEqual(await read(device!,solver!.volumeTexture),v0);
      const diagnostics=(solver as unknown as {pressureMultigrid:{diagnostics:GPUBuffer}}).pressureMultigrid.diagnostics;
      const receipt=device!.createBuffer({size:12,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
      try{const e=device!.createCommandEncoder();e.copyBufferToBuffer(diagnostics,64,receipt,0,12);device!.queue.submit([e.finish()]);
        await receipt.mapAsync(GPUMapMode.READ);assert.deepEqual([...new Uint32Array(receipt.getMappedRange())],[1,1,0],"default tolerance skips remaining cycles in a converged pool");
      }finally{receipt.unmap();receipt.destroy();}
      const phi=await read(device!,solver!.vertexPhiTexture!);
      assert.ok(phi.every((p,i)=>Math.abs(p-phi0[i]!)<1e-6));
    });
    const volume=new Float32Array(nx*ny*nz);const phi=new Float32Array((nx+1)*(ny+1)*(nz+1));
    for(let z=2;z<nz-2;z++)for(let y=2;y<ny-2;y++)for(let x=3;x<7;x++)volume[x+nx*(y+ny*z)]=1;
    const hx=scene.container.width_m/nx;
    for(let z=0;z<=nz;z++)for(let y=0;y<=ny;y++)for(let x=0;x<=nx;x++)phi[x+(nx+1)*(y+(ny+1)*z)]=(x-7)*hx;
    const transport=new Float32Array(access.transportA.width*access.transportA.height*access.transportA.depthOrArrayLayers*4);
    const reset=(displacement:number)=>{write(device!,solver!.volumeTexture,volume);write(device!,solver!.vertexPhiTexture!,phi);
      transport.fill(0);for(let i=0;i<transport.length;i+=4)transport[i]=displacement*hx*30;write(device!,access.transportA,transport);access.writeParams(1/30,0,0);};
    const encode=async()=>{const e=device!.createCommandEncoder();access.encodeGeometricVolume(e);device!.queue.submit([e.finish()]);await device!.queue.onSubmittedWorkDone();return read(device!,access.volumeB);};
    await t.test("integer Courant-two translation is exact away from walls",async()=>{
      reset(2);const actual=await encode();
      for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++)assert.ok(Math.abs(actual[x+nx*(y+ny*z)]!-(x>=2?volume[x-2+nx*(y+ny*z)]!:0))<2e-6);
      const movedPhi=await read(device!,solver!.vertexPhiTexture!);
      for(let x=4;x<nx-2;x++)assert.ok(Math.abs(movedPhi[x+(nx+1)*(3+(ny+1)*3)]!-(x-9)*hx)<1e-5);
    });
    await t.test("fractional translation remains positive and conserves total V",async()=>{
      reset(0.375);const actual=await encode();assert.ok(actual.every(v=>Number.isFinite(v)&&v>=0));
      assert.ok(Math.abs(sum(actual)-sum(volume))<1e-5*sum(volume));
    });
    await t.test("capacity error gates GPU rounds and respects the round cap",async()=>{
      const index=4+nx*(4+ny*4);const original=volume[index]!;volume[index]=1.05;
      const receipt=async()=>{
        const b=device!.createBuffer({size:28,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        const e=device!.createCommandEncoder();e.copyBufferToBuffer(access.conditioningScratch,2*nx*ny*nz*4,b,0,28);device!.queue.submit([e.finish()]);
        await b.mapAsync(GPUMapMode.READ);const words=new Uint32Array(b.getMappedRange()).slice();b.unmap();b.destroy();return words;
      };
      for(const tolerance of [10,0.1]){
        solver!.applyRuntimeValues({liquidCapacityBalancing:"on",densitySharpening:"off",liquidCapacityBalancingRounds:3,liquidCapacityBalancingTolerance:tolerance});
        reset(0);const actual=await encode();const words=await receipt();
        assert.equal(words[5],tolerance===10?0:3,"only errors above tolerance schedule corrective rounds");
        assert.deepEqual([...words.slice(2,5)],tolerance===10?[0,0,0]:[Math.ceil(nx/4),Math.ceil(ny/4),Math.ceil(nz/4)]);
        assert.ok(Math.abs(sum(actual)-sum(volume))<1e-5*sum(volume));
      }
      volume[index]=original;
      solver!.applyRuntimeValues({liquidCapacityBalancing:"on",densitySharpening:"off"});reset(0);await encode();
      assert.equal((await receipt())[5],0,"a feasible identity map needs no corrective round");
    });
    await t.test("sharpening conserves V and leaves phi unchanged",async()=>{
      volume[6+nx*(3+ny*3)]=0.8; volume[7+nx*(3+ny*3)]=0.2;
      reset(0);const unsharpened=await encode();const before=await read(device!,solver!.vertexPhiTexture!);
      solver!.applyRuntimeValues({liquidCapacityBalancing:"on",densitySharpening:"on",sharpeningStrength:1});reset(0);const sharp=await encode();
      assert.deepEqual(await read(device!,solver!.vertexPhiTexture!),before);
      assert.ok(Math.abs(sum(sharp)-sum(unsharpened))<1e-5*sum(volume));assert.ok(sharp.every(v=>Number.isFinite(v)&&v>=-1e-7));
      assert.ok(sharp.some((v,i)=>Math.abs(v-unsharpened[i]!)>1e-5),"sharpening must actually move volume");
    });
    await t.test("arriving liquid continues onto a closed wall vertex",async()=>{
      reset(0);
      const approaching=Float32Array.from(phi,(_,i)=>((i%(nx+1))-(nx-0.5))*hx);
      write(device!,solver!.vertexPhiTexture!,approaching);
      // Constant interior +X velocity, zero normal velocity at the closed wall.
      for(let z=0;z<access.transportA.depthOrArrayLayers;z++)for(let y=0;y<access.transportA.height;y++)for(let x=0;x<access.transportA.width;x++){
        transport[4*(x+access.transportA.width*(y+access.transportA.height*z))]=x>=nx?0:2*hx*30;
      }
      write(device!,access.transportA,transport);await encode();
      const contacted=await read(device!,solver!.vertexPhiTexture!);
      assert.ok(contacted[nx+(nx+1)*(4+(ny+1)*4)]!<0,"closed-wall phi must accept arriving interior liquid despite zero normal wall velocity");
    });
    await t.test("planar pool stays hydrostatic under gravity",async()=>{
      write(device!,solver!.volumeTexture,v0);write(device!,solver!.vertexPhiTexture!,phi0);
      write(device!,solver!.velocityTexture,new Float32Array(nx*ny*nz*4));
      write(device!,access.transportA,new Float32Array(transport.length));
      scene.fluid.gravity_m_s2={x:0,y:-9.81,z:0};
      for(let frame=2;frame<=32;frame++)assert.ok(solver!.advanceTo(frame/30));
      await device!.queue.onSubmittedWorkDone();const values=await read(device!,solver!.volumeTexture);
      assert.ok(values.every(v=>Number.isFinite(v)&&v>=-1e-6));assert.ok(Math.abs(sum(values)-sum(v0))<1e-4*sum(v0));
      const velocity=await read(device!,solver!.velocityTexture);let max=0;for(let i=0;i<velocity.length;i+=4)max=Math.max(max,Math.hypot(velocity[i]!,velocity[i+1]!,velocity[i+2]!));
      assert.ok(max<0.01,`hydrostatic max speed ${max}`);
    });
    await t.test("pressure cycle convergence preserves parity, resets, and supports live tolerance",async()=>{
      const run=async(fullCycles:number,vCycles:number,tolerance:number)=>{
        const candidate=await WebGPUUniformReferenceSolver.createAsync(device!,scene,"balanced",undefined,{
          geometricVolume:true,densitySharpening:false,solidExcessCorrection:false,
          pressureSchedule:{fullCycles,vCycles,preSweeps:6,postSweeps:6,residualTolerance:tolerance},
        },()=>{});
        const mg=(candidate as unknown as {pressureMultigrid:{diagnostics:GPUBuffer;pressureTexture:GPUTexture}}).pressureMultigrid;
        const status=async()=>{
          const buffer=device!.createBuffer({size:76,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
          try{const e=device!.createCommandEncoder();e.copyBufferToBuffer(mg.diagnostics,0,buffer,0,76);device!.queue.submit([e.finish()]);
            await buffer.mapAsync(GPUMapMode.READ);return new Uint32Array(buffer.getMappedRange()).slice();
          }finally{buffer.unmap();buffer.destroy();}
        };
        try{
          assert.ok(candidate.advanceTo(1/30));
          const first=await status();const pressure=await read(device!,mg.pressureTexture);
          assert.ok(pressure.some(p=>p>0),"exercise nonzero pressure, not only an empty solve");
          if(tolerance>0){
            assert.equal(first[16],1);assert.equal(first[17],fullCycles>0?1:0);assert.equal(first[18],fullCycles>0?0:1);
            candidate.applyRuntimeValues({pressureResidualTolerance:0,densitySharpening:"off"});
            assert.ok(candidate.advanceTo(2/30));const second=await status();
            assert.equal(second[16],0);assert.equal(second[17],fullCycles);assert.equal(second[18],vCycles);
          }else{assert.equal(first[16],0);assert.equal(first[17],fullCycles);assert.equal(first[18],vCycles);}
          return pressure;
        }finally{candidate.destroy();}
      };
      assert.deepEqual(await run(3,4,1e6),await run(1,0,0),"Full-Cycle stop must preserve its canonical pressure");
      assert.deepEqual(await run(0,4,1e6),await run(0,1,0),"V-Cycle stop must preserve odd pressure parity");
    });
    await t.test("mini32 conserves liquid through separating far-wall impact",async()=>{
      solver!.destroy();
      solver=await uniformVolumeMethod.createSolverAsync!(device!,sceneDocument(getSceneDefinition("minimal-power-dam-break-32")),"balanced",resolveMethodValues(uniformVolumeMethod,"balanced",{liquidCapacityBalancing:"on",velocityTransport:"semi-lagrangian"}),undefined,()=>{}) as WebGPUUniformReferenceSolver;
      for(let frame=1;frame<=90;frame++){
        assert.ok(solver.advanceTo(frame/30));
        if(frame===30||frame===90){
          await device!.queue.onSubmittedWorkDone();const stats=await solver.readStats();
          // Track geometric drift separately from conserved V. The former
          // sticky-wall silhouette cutoff is not a mass-conservation oracle.
          const volume=await read(device!,solver.volumeTexture);
          const maximumVolume=volume.reduce((maximum,value)=>Math.max(maximum,value),0);
          console.log(JSON.stringify({case:"mini32-separating-wall",frame,representedVolumeDrift:stats.representedVolumeDrift,maximumVolume}));
          assert.ok(Math.abs(sum(volume)/stats.initialVolumeCellSum!-1)<1e-5);
          if(frame===90)assert.ok(maximumVolume<1.1,`impact capacity bound exceeded: maximum V=${maximumVolume}`);
        }
      }
    });
    assert.deepEqual(errors,[]);
  }finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
