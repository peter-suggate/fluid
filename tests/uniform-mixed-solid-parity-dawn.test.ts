import assert from "node:assert/strict";
import test from "node:test";
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
import {readMixedTexture,readMixedTileWords} from "./helpers/uniform-mixed-native-fields";
const modulePath=process.env.WEBGPU_NODE_MODULE;
const only=process.env.FLUID_SOLID_PARITY_CASE;

/** Mixed ownership with embedded solids. `region` cases coarsen the whole
 * domain; promotion must keep every tile near a cut cell fine, conserve mass
 * and converge. `throws` cases must refuse loudly. */
type Case={id:string;frames:number;scene:(s:SceneDescription)=>void;throws?:RegExp;pressureTolerance?:number};
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
 {id:"box-coarse",frames:4,scene:()=>{}},
 {id:"box-floor-coarse",frames:4,scene:s=>{s.fluid.refinementRegions=[{...coarse,max_m:{x:.4,y:.2,z:.4}}];}},
 {id:"voxel-coarse",frames:4,scene:blocks},
 {id:"terrain-coarse",frames:4,scene:terrain},
];
async function run(device:GPUDevice,scene:SceneDescription,frames:number,pressureTolerance?:number){
 const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,uniformGeometricSolverOptions({},scene),()=>{});
 try{
  const initial=await ownerMass(device,solver);
  if(pressureTolerance!==undefined)solver.applyRuntimeValues({pressureResidualTolerance:pressureTolerance});
  const residuals:number[]=[];
  for(let i=1;i<=frames;i++){assert.ok(solver.advanceTo(i/30),`advance ${i}`);await solver.awaitFrameCompletion();residuals.push((await solver.readStats()).uniformPressureAcceptedResidual!);}
  const info=solver.info;
  return {initial,volume:await ownerMass(device,solver),residuals,
   tiles:{fine:info.uniformMixedFineTiles,coarse:info.uniformMixedCoarseTiles}};
 }finally{solver.destroy();}
}
/** Liquid volume in h cells. A mixed owner's V lives at its origin texel and
 * covers width^3 cells; its other texels are not state (transport, cleanup and
 * sharpening write the origin only), so a plain texture sum counts stale
 * values wherever a tile is 4h. This is the same owner sum as the solver's
 * volumeCellSum, unquantized. */
async function ownerMass(device:GPUDevice,solver:WebGPUUniformReferenceSolver):Promise<number>{
 const texture=solver.volumeTexture,volume=await readMixedTexture(device,texture);
 const tiles=await readMixedTileWords(device,solver);
 const [nx,ny,nz]=[texture.width,texture.height,texture.depthOrArrayLayers];let sum=0;
 for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
  const word=tiles[(x>>2)+(nx>>2)*((y>>2)+(ny>>2)*(z>>2))]!,width=word&0x80000000?1:4;
  if(x%width===0&&y%width===0&&z%width===0)sum+=volume[x+nx*(y+ny*z)]!*width**3;
 }
 return sum;
}

(modulePath?test:test.skip)("coarse regions promote solid neighbourhoods to h and conserve mass",{timeout:1800000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed solid regions");let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const failures:string[]=[];
  for(const c of cases.filter(c=>!only||only.split(",").includes(c.id))){
   const scene=base();scene.fluid.refinementRegions=[coarse];c.scene(scene);
   const frames=Number(process.env.FLUID_SOLID_PARITY_FRAMES??c.frames);
   if(c.throws){
    await assert.rejects(run(device,scene,frames),c.throws);console.log(JSON.stringify({case:c.id,refused:true}));continue;
   }
   const mixed=await run(device,scene,frames,c.pressureTolerance);
   const report={case:c.id,frames,massInitial:mixed.initial,massMixed:mixed.volume,residualMixed:mixed.residuals,tiles:mixed.tiles};
   console.log(JSON.stringify(report));
   // Promotion: coarse ownership exists, and the solid neighbourhood is fine.
   if(!(mixed.tiles.coarse!>0&&(c.id.startsWith("box")||mixed.tiles.fine!>0)))failures.push(`${c.id}: promotion produced ${JSON.stringify(mixed.tiles)}`);
   // Conservation (only dust is removed; 1e-3 of the dam).
   if(!(Math.abs(report.massMixed-report.massInitial)<=1e-3*report.massInitial))failures.push(`${c.id}: mass ${report.massMixed} vs initial ${report.massInitial}`);
  }
  assert.deepEqual(errors,[]);assert.deepEqual(failures,[]);
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
