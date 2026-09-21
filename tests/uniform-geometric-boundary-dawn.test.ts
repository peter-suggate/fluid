import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { solidWorldForScene, sampleSolidWorld } from "../lib/core/solid-world";
import { solidVoxelShellForScene } from "../lib/core/scene-lattice";
import { readBufferBinding } from "../lib/harness/webgpu-smoke-readbacks";
import { resolveMethodValues } from "../lib/core/method-contract";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
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


const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("uniform geometric separates from domain and embedded solids",{timeout:420_000},async t=>{
  await acquireWebGPUExclusiveLock("dawn-test","uniform geometric separating contact");
  let device:GPUDevice|undefined;
  try {
    const dawn=await import(pathToFileURL(modulePath!).href) as NodeDawnProvider;Object.assign(globalThis,dawn.globals);
    const gpu=createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
    const adapter=await gpu.requestAdapter();assert.ok(adapter);
    device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
    const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
    for(const kind of ["domain-ceiling","embedded-ceiling","side-wall-zero-gravity","negative-side-wall-zero-gravity"] as const){
      await t.test(kind,async()=>{
        const scene=structuredClone(sceneDocument(getSceneDefinition("ceiling-slab-drop")));
        const side=kind.includes("side-wall");const low=kind.startsWith("negative");
        if(kind!=="domain-ceiling")scene.container.height_m=24*0.05;
        scene.fluid.initialBrickSeeds_m=[{x:0,y:0.6,z:0}];
        scene.fluid.gravity_m_s2={x:0,y:side?0:-9.81,z:0};
        scene.solidVoxels=[...solidVoxelShellForScene(scene)];
        if(kind==="embedded-ceiling")scene.solidVoxels.push({operation:"fill",minimum:[0,16,0],maximumExclusive:[24,17,24]});
        if(side)scene.solidVoxels.push({operation:"fill",minimum:[low?7:16,0,0],maximumExclusive:[low?8:17,24,24]});
        const solver=await WebGPUUniformReferenceSolver.createAsync(device!,scene,"balanced",undefined,{
          geometricVolume:true,pageDomain:true,densitySharpening:false,solidExcessCorrection:false,velocityTransport:"semi-lagrangian",
          pressureSchedule:{fullCycles:3,vCycles:4,preSweeps:6,postSweeps:6,residualTolerance:1e-6},
        },()=>{});
        try {
          const {nx,ny,nz}=solver.info;
          const initial=await read(device!,solver.volumeTexture);const mass=initial.reduce((a,b)=>a+b,0);
          if(side){const velocity=new Float32Array(nx*ny*nz*4);for(let i=0;i<nx*ny*nz;i++)velocity[4*i]=low?0.5:-0.5;solver.initializeVelocityForQA(velocity);}
          let firstVelocity:Float32Array|undefined;
          for(let frame=1;frame<=3;frame++){
            assert.ok(solver.advanceTo(frame/30));
            if(frame===1)firstVelocity=await read(device!,solver.velocityTexture);
          }
          await device!.queue.onSubmittedWorkDone();
          const phi=await read(device!,solver.vertexPhiTexture!);const volume=await read(device!,solver.volumeTexture);
          const contacts:number[]=[];const speeds:number[]=[];
          for(let z=10;z<=14;z++)for(let q=10;q<=14;q++){
            const x=side?(low?8:16):q,y=side?q:16;
            contacts.push(phi[x+(nx+1)*(y+(ny+1)*z)]!);
            const cell=side?(low?7:15)+nx*(q+ny*z):q+nx*(15+ny*z);
            speeds.push((low?-1:1)*firstVelocity![4*cell+(side?0:1)]!);
          }
          console.log(JSON.stringify({kind,mass,finalMass:volume.reduce((a,b)=>a+b,0),minContactPhi:Math.min(...contacts),maxContactVelocity:Math.max(...speeds)}));
          assert.ok(Math.min(...contacts)>0,`${kind}: wall vertices must expose air`);
          assert.ok(Math.max(...speeds)<-0.01,`${kind}: projected faces must retain separating velocity`);
          assert.ok(volume.every(v=>Number.isFinite(v)&&v>=-1e-6));
          assert.ok(Math.abs(volume.reduce((a,b)=>a+b,0)/mass-1)<1e-5,"release must conserve liquid mass");
          // No volume may be transported through the closed slab.
          if(kind!=="domain-ceiling")for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
            if(low?x<8:(side?x:y)>=16)assert.ok(volume[x+nx*(y+ny*z)]!<1e-6,"no solid penetration");
          }
        }finally{solver.destroy();}
      });
    }
    await t.test("figure 8 releases its overhead voxel shell",async()=>{
      const scene=structuredClone(sceneDocument(getSceneDefinition("cm12-figure-8")));
      scene.voxelDomain.finestCellSize_m=scene.container.width_m/32;
      scene.solidVoxels=[...solidVoxelShellForScene(scene)];
      const solver=await WebGPUUniformReferenceSolver.createAsync(device!,scene,"balanced",undefined,{
        geometricVolume:true,pageDomain:true,densitySharpening:true,solidExcessCorrection:false,velocityTransport:"semi-lagrangian",
      },()=>{});
      try {
        const {nx,ny,nz}=solver.info;const world=solidWorldForScene(scene);
        const wallVertices:number[]=[];
        for(let z=1;z<nz;z++)for(let y=Math.ceil(ny*0.65);y<ny;y++)for(let x=1;x<nx;x++){
          let solid=false,open=false;
          for(let k=0;k<8;k++){
            const fraction=sampleSolidWorld(world,[x-1+(k&1),y-1+((k>>1)&1),z-1+((k>>2)&1)]).solidFraction;
            if(fraction>0)solid=true;else open=true;
          }
          if(solid&&open)wallVertices.push(x+(nx+1)*(y+(ny+1)*z));
        }
        const initialPhi=await read(device!,solver.vertexPhiTexture!);
        const initialV=await read(device!,solver.volumeTexture);const mass=initialV.reduce((a,b)=>a+b,0);
        const initialWet=wallVertices.filter(i=>initialPhi[i]!<0).length;
        for(let frame=1;frame<=30;frame++)assert.ok(solver.advanceTo(frame/30));
        const phi=await read(device!,solver.vertexPhiTexture!);const volume=await read(device!,solver.volumeTexture);
        const finalWet=wallVertices.filter(i=>initialPhi[i]!<0&&phi[i]!<0).length;
        console.log(JSON.stringify({kind:"figure-8",initialWet,finalWet,mass,finalMass:volume.reduce((a,b)=>a+b,0)}));
        assert.ok(initialWet>100,"exercise an initially wetted curved overhead shell");
        assert.ok(finalWet<initialWet/2,"initial overhead contact must peel from the sphere");
        assert.ok(phi.every(Number.isFinite));assert.ok(volume.every(v=>Number.isFinite(v)&&v>=-1e-6));
        assert.ok(Math.abs(volume.reduce((a,b)=>a+b,0)/mass-1)<1e-5);
        for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
          if(sampleSolidWorld(world,[x,y,z]).solidFraction>0)assert.ok(volume[x+nx*(y+ny*z)]!<1e-6,"sphere must remain impermeable");
        }
      }finally{solver.destroy();}
    });
    await t.test("default figure 12 keeps solid-covered domain faces closed through rebound",async()=>{
      const scene=sceneDocument(getSceneDefinition("cm12-figure-12"));
      const solver=await uniformVolumeMethod.createSolverAsync!(device!,scene,"balanced",
        resolveMethodValues(uniformVolumeMethod,"balanced",{}),undefined,()=>{}) as WebGPUUniformReferenceSolver;
      try {
        const {nx,ny,nz}=solver.info;assert.deepEqual([nx,ny,nz],[128,128,128]);
        const world=solidWorldForScene(scene);const blocked:number[]=[];
        for(let z=0;z<nz;z++)for(let x=0;x<nx;x++){
          if(sampleSolidWorld(world,[x,0,z]).solidFraction===1)blocked.push(ny*nz+x+nx*z);
        }
        assert.ok(blocked.length>1000,"exercise the vessel/domain overlap");
        for(let frame=1;frame<=60;frame++){
          assert.ok(solver.advanceTo(frame/30));
          if(frame<26)continue;
          const bytes=await readBufferBinding(device!,{buffer:solver.negativeBoundaryVelocityBuffer},solver.negativeBoundaryVelocityBytes);
          const boundary=new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4);
          for(const i of blocked)assert.equal(boundary[i],0,`frame ${frame}: solid-covered floor must carry no fluid velocity`);
          if(frame===34||frame===60){
            const velocity=await read(device!,solver.velocityTexture);let maximum=0;
            for(let i=0;i<velocity.length;i++)if(i%4!==3){assert.ok(Number.isFinite(velocity[i]));maximum=Math.max(maximum,Math.abs(velocity[i]!));}
            console.log(JSON.stringify({kind:"figure-12-default",frame,maximum,blockedFloorFaces:blocked.length}));
          }
        }
      }finally{solver.destroy();}
    });
    assert.deepEqual(errors,[]);
  }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
