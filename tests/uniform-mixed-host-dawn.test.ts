import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import type {GPUEulerianInfo} from "../lib/core/webgpu-eulerian";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {uniformGeometricSolverOptions} from "../lib/methods/uniform/uniform-geometric-options";
import {readMixedTexture} from "./helpers/uniform-mixed-native-fields";
import {createUniformMixedLayout,mixedCellWidth} from "../lib/methods/uniform/uniform-mixed-layout";
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("mixed host publishes one ownership generation and applies paused and pending region edits without restart",{timeout:180000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed host lifecycle");let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
  scene.container.width_m=scene.container.height_m=scene.container.depth_m=.8;scene.voxelDomain.finestCellSize_m=.025;
  scene.solidVoxels=[];scene.rigidBodies=[];scene.fluid.inflow=undefined;scene.fluid.initialDamBreakDimensions_m={x:.2,y:.4,z:.4};
  solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{...uniformGeometricSolverOptions({},scene),mixedOwnership:true},()=>{});
  const source=solver.denseLevelSetVolumeSource!;assert.ok(source.mixedOwnership);assert.equal(solver.info.uniformMixedCoarseTiles,0);
  const region={id:"coarse",rule:"minimum-cell-size" as const,minimumCellSize_cells:4 as const,maximumCellSize_cells:4 as const,min_m:{x:0,y:0,z:-.4},max_m:{x:.4,y:.8,z:.4}};
  const mixed=structuredClone(scene);mixed.fluid.refinementRegions=[region];
  const generation=solver.info.uniformMixedGeneration!;solver.applySceneUniforms(mixed);
  assert.ok(solver.info.uniformMixedCoarseTiles!>0);assert.equal(solver.info.uniformMixedGeneration,generation+1);
  assert.equal(solver.denseLevelSetVolumeSource,source,"edits reuse consumer bindings");
  assert.ok(solver.advanceTo(1/30));assert.ok(solver.framePending);assert.ok(solver.presentationPending);
  assert.equal(solver.advanceTo(2/30),false);
  solver.applySceneUniforms(scene);assert.equal(solver.info.uniformMixedGeneration,generation+1,"pending frame retains ownership");
  await solver.awaitFrameCompletion();assert.equal(solver.info.uniformMixedCoarseTiles,0);assert.equal(solver.info.uniformMixedGeneration,generation+2);
  assert.equal(solver.info.completedTime_s,1/30);assert.equal(solver.info.simulatedTime_s,1/30);assert.equal(solver.framePending,false);
  assert.ok(solver.advanceTo(2/30));await solver.awaitFrameCompletion();
  const info=await solver.readStats();assert.equal(info.completedTime_s,2/30);assert.equal(info.encodedSteps,2);assert.equal(info.uniformPressureCyclesConverged,true);
  // Add a sphere wholly in coarse air after evolved-state remapping. Count
  // physical volume at canonical addresses, never stale inactive fine texels.
  solver.applySceneUniforms(mixed);
  const layout=createUniformMixedLayout({dimensions:[32,32,32],cellSize_m:[.025,.025,.025],origin_m:{x:-.4,y:0,z:-.4}},[region],false);
  const mass=async()=>{
   const values=await readMixedTexture(device!,solver!.volumeTexture);let sum=0;
   for(let t=0;t<layout.tiles.length;t++){
    const w=mixedCellWidth(layout.tiles[t]!);const origin=[t%8*4,Math.floor(t/8)%8*4,Math.floor(t/64)*4];
    for(let z=0;z<4;z+=w)for(let y=0;y<4;y+=w)for(let x=0;x<4;x+=w){
     const value=values[origin[0]!+x+32*(origin[1]!+y+32*(origin[2]!+z))]!;
     assert.ok(Number.isFinite(value));sum+=value*w**3;
    }
   }return sum;
  };
  const before=await mass(),centre={x:.2,y:.6,z:0},radius=.06;
  let expected=0;
  for(let z=0;z<32;z++)for(let y=0;y<32;y++)for(let x=0;x<32;x++)for(let sample=0;sample<8;sample++){
   const dx=-.4+(x+.25+.5*(sample&1))*.025-centre.x;
   const dy=(y+.25+.5*((sample>>1)&1))*.025-centre.y;
   const dz=-.4+(z+.25+.5*((sample>>2)&1))*.025-centre.z;
   if(Math.hypot(dx,dy,dz)<=radius)expected+=.125;
  }
  solver.injectLiquidBall({centre_m:centre,radius_m:radius});
  assert.ok(solver.advanceTo(3/30));await solver.awaitFrameCompletion();
  const after=await mass(),dropInfo=await solver.readStats();
  assert.equal(dropInfo.completedTime_s,3/30);
  assert.ok(Math.abs(dropInfo.volumeCellSum!-after)<.1,`canonical diagnostic mass ${dropInfo.volumeCellSum} vs ${after}`);
  assert.ok(Number.isFinite(dropInfo.maxSpeed_m_s)&&dropInfo.maxSpeed_m_s!>0);
  assert.ok(Math.abs(after-before+Number(dropInfo.uniformVolumeDustMass_cells??0)-expected)<.02,
   `drop mass ${after-before}, dust ${dropInfo.uniformVolumeDustMass_cells}, expected ${expected}`);
  const hose=structuredClone(mixed);
  hose.fluid.inflow={center_m:{x:.125,y:.65,z:.2},radius_m:.05,length_m:.05,velocity_m_s:{x:2,y:0,z:0},start_s:0,end_s:10,ramp_s:0};
  solver.applySceneUniforms(hose);await solver.refreshSceneTopology();
  let hoseMass=await mass();const addedPerStep=Math.PI*.05**2*2/30/.025**3;
  for(let step=4;step<=6;step++){
   assert.ok(solver.advanceTo(step/30));await solver.awaitFrameCompletion();
   const current=await mass();const info:GPUEulerianInfo=await solver.readStats();
   assert.ok(Math.abs(info.completedTime_s!-step/30)<1e-12);
   assert.ok(Math.abs(current-hoseMass+Number(info.uniformVolumeDustMass_cells??0)-addedPerStep)<.05,
    `inlet mass ${current-hoseMass}, expected ${addedPerStep}`);
   hoseMass=current;
  }
  const stopped=structuredClone(hose);stopped.fluid.inflow!.enabled=false;solver.applySceneUniforms(stopped);
  assert.ok(solver.advanceTo(7/30));await solver.awaitFrameCompletion();
  const stoppedInfo=await solver.readStats();
  assert.ok(Math.abs(await mass()-hoseMass+Number(stoppedInfo.uniformVolumeDustMass_cells??0))<.05,"closed tap adds no liquid");
  assert.deepEqual(errors,[]);
 }finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
