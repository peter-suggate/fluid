import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {solidVoxelShellForScene} from "../lib/core/scene-lattice";
import type {FluidRefinementRegion,SceneDescription} from "../lib/core/model";
import {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {uniformGeometricSolverOptions} from "../lib/methods/uniform/uniform-geometric-options";
import {readMixedTexture} from "./helpers/uniform-mixed-native-fields";
const modulePath=process.env.WEBGPU_NODE_MODULE;
const only=process.env.FLUID_SOLID_PARITY_CASE;

/** Parity cases run all-fine mixed ownership against the native oracle for one
 * frame: the dam wets the solids at once, and frame 2 is already chaotic (the
 * plain box drifts to V 0.02 from last-bit pressure differences). `region`
 * cases coarsen the whole domain; promotion must keep every tile near a cut
 * cell fine, conserve mass and converge. `throws` cases must refuse loudly. */
type Case={id:string;frames:number;scene:(s:SceneDescription)=>void;region?:boolean;throws?:RegExp;pressureTolerance?:number};
const base=():SceneDescription=>{
 const s=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
 s.container.width_m=s.container.height_m=s.container.depth_m=.8;s.voxelDomain.finestCellSize_m=.025;
 s.solidVoxels=[];s.rigidBodies=[];s.fluid.inflow=undefined;s.fluid.initialDamBreakDimensions_m={x:.3,y:.4,z:.8};
 return s;
};
// Inside the dam column (submerged, off the floor) and downstream on the floor.
const blocks=(s:SceneDescription)=>{s.solidVoxels.push({operation:"fill",minimum:[3,5,10],maximumExclusive:[7,9,14]},{operation:"fill",minimum:[16,0,12],maximumExclusive:[20,4,20]});};
// Non-integer heights give partial open fractions and apertures.
const terrain=(s:SceneDescription)=>{s.terrain={baseHeight_m:.03,features:[{kind:"mound",center_m:{x:.05,z:0},radius_m:{x:.2,z:.25},amount_m:.11,flat:.2}]};};
const coarse:FluidRefinementRegion={id:"coarse",rule:"minimum-cell-size",minimumCellSize_cells:4,maximumCellSize_cells:4,min_m:{x:-.4,y:0,z:-.4},max_m:{x:.4,y:.8,z:.4}};
const cases:Case[]=[
 {id:"box",frames:1,scene:()=>{}},
 {id:"voxel",frames:1,scene:blocks},
 {id:"terrain",frames:1,scene:terrain},
 // Converged: at the default tolerance both accept after one Full-Cycle whose
 // iterates differ by 2.2e-3 m/s (residual 0.556 native, 0.460 mixed). The
 // restricted phi/V pyramid matches native exactly; the one-cycle gap is open.
 {id:"sphere",frames:1,pressureTolerance:1e-3,scene:s=>{s.container.shape="sphere";s.solidVoxels=[...solidVoxelShellForScene(s)];}},
 // Solid-free controls: an all-4h box conserves; a 4h floor band under h
 // liquid (y-normal seam) already fails conservation without any solid.
 {id:"box-coarse",frames:4,scene:()=>{},region:true},
 {id:"box-floor-coarse",frames:4,scene:s=>{s.fluid.refinementRegions=[{...coarse,max_m:{x:.4,y:.2,z:.4}}];},region:true},
 {id:"voxel-coarse",frames:4,scene:blocks,region:true},
 {id:"terrain-coarse",frames:4,scene:terrain,region:true},
 {id:"rigid",frames:1,throws:/rigid bodies/,scene:s=>{s.rigidBodies=[{id:"crate",name:"Crate",shape:"box",dimensions_m:{x:.1,y:.1,z:.1},density_kg_m3:500,
  position_m:{x:.1,y:.1,z:0},orientation:{x:0,y:0,z:0,w:1},linearVelocity_m_s:{x:0,y:0,z:0},angularVelocity_rad_s:{x:0,y:0,z:0},restitution:.2,friction:.5,motion:"static"}];}},
];
// FP32 reassociation only: owner-driven sums and the h/2h/4h traversal order
// differ from the dense native kernels. The plain box shows V 0, u 7e-7 m/s,
// phi 3e-8 m after one frame; an unported solid term is O(1e-2) or larger.
const TOLERANCE={volume:1e-4,velocity:1e-4,phi:1e-5};

async function run(device:GPUDevice,scene:SceneDescription,mixed:boolean,frames:number,pressureTolerance?:number){
 const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{...uniformGeometricSolverOptions({},scene),mixedOwnership:mixed},()=>{});
 try{
  if(pressureTolerance!==undefined)solver.applyRuntimeValues({pressureResidualTolerance:pressureTolerance});
  const residuals:number[]=[];
  for(let i=1;i<=frames;i++){assert.ok(solver.advanceTo(i/30),`advance ${i}`);await solver.awaitFrameCompletion();residuals.push((await solver.readStats()).uniformPressureAcceptedResidual!);}
  const info=solver.info;
  return {volume:await readMixedTexture(device,solver.volumeTexture),velocity:await readMixedTexture(device,solver.velocityTexture),phi:await readMixedTexture(device,solver.vertexPhiTexture!),residuals,
   tiles:{fine:info.uniformMixedFineTiles,transition:info.uniformMixedTransitionTiles,coarse:info.uniformMixedCoarseTiles}};
 }finally{solver.destroy();}
}
const maxDiff=(a:Float32Array,b:Float32Array,stride=1,components=stride)=>{let m=0,at=-1;for(let i=0;i<a.length;i++){if(i%stride>=components)continue;const d=Math.abs(a[i]!-b[i]!);if(!(d<=m)){m=d;at=i;}}return {max:m,at:Math.floor(at/stride)};};
const mass=(v:Float32Array)=>v.reduce((a,b)=>a+b,0);

const lane=(name:string,region:boolean)=>(modulePath?test:test.skip)(name,{timeout:1800000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test",`Uniform mixed solid ${region?"regions":"parity"}`);let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const failures:string[]=[];
  for(const c of cases.filter(c=>!!c.region===region&&(!only||only.split(",").includes(c.id)))){
   const scene=base();if(c.region)scene.fluid.refinementRegions=[coarse];c.scene(scene);
   const frames=Number(process.env.FLUID_SOLID_PARITY_FRAMES??c.frames);
   if(c.throws){
    await assert.rejects(run(device,scene,true,frames),c.throws);console.log(JSON.stringify({case:c.id,refused:true}));continue;
   }
   const native=await run(device,scene,false,frames,c.pressureTolerance),mixed=await run(device,scene,true,frames,c.pressureTolerance);
   const report={case:c.id,frames,massNative:mass(native.volume),massMixed:mass(mixed.volume),volume:maxDiff(native.volume,mixed.volume),
    velocity:maxDiff(native.velocity,mixed.velocity,4,3),phi:maxDiff(native.phi,mixed.phi),residualNative:native.residuals,residualMixed:mixed.residuals,tiles:mixed.tiles};
   console.log(JSON.stringify(report));
   if(c.region){
    // Promotion: coarse ownership exists, and the solid neighbourhood is fine.
    if(!(mixed.tiles.coarse!>0&&(c.id.startsWith("box")||mixed.tiles.fine!>0)))failures.push(`${c.id}: promotion produced ${JSON.stringify(mixed.tiles)}`);
    // Conservation against native (both remove only dust; 1e-3 of the dam).
    if(!(Math.abs(report.massMixed-report.massNative)<=1e-3*report.massNative))failures.push(`${c.id}: mass ${report.massMixed} vs native ${report.massNative}`);
   }else{
    for(const key of ["volume","velocity","phi"] as const)if(!(report[key].max<=TOLERANCE[key]))failures.push(`${c.id}: ${key} ${report[key].max} at ${report[key].at}`);
   }
  }
  assert.deepEqual(errors,[]);assert.deepEqual(failures,[]);
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
lane("mixed all-fine ownership reproduces native Uniform with embedded solids",false);
lane("coarse regions promote solid neighbourhoods to h and conserve mass",true);
