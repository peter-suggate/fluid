import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {uniformGeometricSolverOptions} from "../lib/methods/uniform/uniform-geometric-options";
import {uniformDetailImportance,uniformDetailSettings} from "../lib/methods/uniform/uniform-detail-policy";
import type {UniformMixedDynamicClassifier} from "../lib/methods/uniform/uniform-mixed-dynamic";
import {mixedExtent,readMixedTexture} from "./helpers/uniform-mixed-native-fields";
const modulePath=process.env.WEBGPU_NODE_MODULE;

(modulePath?test:test.skip)("Surface intersects Dynamic selection with phi crossings and never seeds detail",{timeout:180000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform Surface filter");let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
  scene.container.width_m=scene.container.height_m=scene.container.depth_m=.8;scene.voxelDomain.finestCellSize_m=.025;
  scene.rigidBodies=[];scene.fluid.inflow=undefined;scene.fluid.initialDamBreakDimensions_m={x:.3,y:.4,z:.8};scene.solidVoxels=[];scene.fluid.refinementRegions=[];
  const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,uniformGeometricSolverOptions({detailPolicy:"dynamic"},scene),()=>{});
  try{
   const dynamic=(solver as unknown as {mixedDynamic:UniformMixedDynamicClassifier}).mixedDynamic;
   const texture=solver.vertexPhiTexture!;const phi=await readMixedTexture(device,texture),[nx,ny,nz]=mixedExtent(texture);
   const tx=(nx-1)/4,ty=(ny-1)/4,tz=(nz-1)/4,crossing=new Uint8Array(tx*ty*tz);
   for(let z=0;z<tz;z++)for(let y=0;y<ty;y++)for(let x=0;x<tx;x++){
    let lo=Infinity,hi=-Infinity;
    for(let k=0;k<=4;k++)for(let j=0;j<=4;j++)for(let i=0;i<=4;i++){
     const value=phi[4*x+i+nx*(4*y+j+ny*(4*z+k))]!;lo=Math.min(lo,value);hi=Math.max(hi,value);
    }
    crossing[x+tx*(y+ty*z)]=Number(lo<0&&hi>=0);
   }
   const census=async(surface:boolean,shape=true,join?:Uint8Array,distance=0)=>{
    if(join)dynamic.join(join);
    const importance=uniformDetailImportance(uniformDetailSettings({detailPolicy:"dynamic",detailSurface:surface?"on":"off",detailSurfaceDistance:distance,detailShape:shape?"on":"off",detailImpact:"off",detailHoldSteps:0}),crossing.length);
    const encoder=device!.createCommandEncoder();
    dynamic.encode(encoder,{dt:1/60,steps:1,gravity:[0,0,0],reach:2,hysteresis:0,fullTolerance:.01,emptyTolerance:1e-6,surfaceTolerance:0,fastTravel:0,boundaryTravel:0,closedWalls:0b101111,up:1,importance,reasons:true});
    device!.queue.submit([encoder.finish()]);return dynamic.read();
   };
   // Do not adopt either layout: both classifiers see exactly the same fields.
   const off=await census(false),on=await census(true);
   assert.ok(on.fineTiles>0);assert.ok(on.fineTiles<off.fineTiles,`${on.fineTiles} must remove some of ${off.fineTiles} tiles`);
   for(let t=0;t<crossing.length;t++){
    assert.ok(on.fine[t]!<=off.fine[t]!,`Surface added tile ${t}`);
    assert.equal(on.fine[t],crossing[t],`surface crossing at tile ${t}`);
   }
   let previous=on.fine;
   for(let distance=1;distance<=3;distance++){
    const band=await census(true,true,undefined,distance);
    for(let z=0;z<tz;z++)for(let y=0;y<ty;y++)for(let x=0;x<tx;x++){
     const t=x+tx*(y+ty*z);let near=false;
     for(let k=Math.max(0,z-distance);k<=Math.min(tz-1,z+distance);k++)
      for(let j=Math.max(0,y-distance);j<=Math.min(ty-1,y+distance);j++)
       for(let i=Math.max(0,x-distance);i<=Math.min(tx-1,x+distance);i++)near ||= crossing[i+tx*(j+ty*k)]===1;
     assert.equal(band.fine[t],Number(near&&!!off.fine[t]),`distance ${distance}, tile ${t}`);
     assert.ok(band.fine[t]!>=previous[t]!,`distance ${distance} shrank the allowed band`);
    }
    previous=band.fine;
   }
   for(let distance=0;distance<=3;distance++)assert.equal((await census(true,false,undefined,distance)).fineTiles,0,"Surface distance must not seed detail");
   const join=new Uint8Array(crossing.length),air=crossing.findIndex(v=>!v);assert.ok(air>=0);join[air]=1;
   const explicitOff=await census(false,true,join),explicitOn=await census(true,true,join);
   assert.equal(explicitOn.fine[air],1,"explicit requests retain priority");
   for(let t=0;t<crossing.length;t++)assert.ok(explicitOn.fine[t]!<=explicitOff.fine[t]!);
   console.log(JSON.stringify({surfaceOffTiles:off.fineTiles,surfaceOnTiles:on.fineTiles,crossingTiles:crossing.reduce((a,b)=>a+b,0)}));
   assert.deepEqual(errors,[]);
  }finally{solver.destroy();}
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
