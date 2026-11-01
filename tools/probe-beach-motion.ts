/** Production NB-FLIP trajectory and wave-gauge diagnostic; no rendered water substitute. */
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createBeachScene, beachPistonPosition } from '../lib/core/beach-scene';
import { initializeRigidBodies } from '../lib/core/rigid-body';
import { uniformNarrowBandMethod as method } from '../lib/methods/uniform/uniform-narrow-band-method';
import { resolveMethodValues } from '../lib/core/method-contract';
import type { WebGPUUniformReferenceSolver } from '../lib/methods/uniform/webgpu-uniform-reference';
import { managedGPUDevice } from '../lib/core/gpu-compilation-manager';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from '../lib/harness/node-dawn-provider';
import { readUniformLiveAddress } from '../tests/helpers/uniform-geometric';
import { readMixedTexture, readMixedTileWords } from '../tests/helpers/uniform-mixed-native-fields';
import { uniformTileKey } from '../lib/methods/uniform/uniform-lattice-address';
const args=new Map(process.argv.slice(2).map(arg=>{const at=arg.indexOf('=');return [arg.slice(2,at),arg.slice(at+1)];}));
const scene=createBeachScene();
const seconds=Number(args.get('time')??16), output=resolve(args.get('out')??'artifacts/beach/motion.json');
assert.ok(seconds>0&&seconds<=scene.duration_s);
if(args.has('density'))for(const body of scene.rigidBodies)if(body.appearance==='seaweed')body.density_kg_m3=Number(args.get('density'));
if(args.has('thickness'))for(const body of scene.rigidBodies)if(body.appearance==='seaweed'){
 body.dimensions_m.y=Number(args.get('thickness'));
 body.position_m.y=2.4+body.dimensions_m.y*(.5-body.density_kg_m3/scene.fluid.density_kg_m3);
}
const dt=scene.numerics.fixedDt_s,bodies=initializeRigidBodies(scene.rigidBodies);
paceDawnEventPolling();const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal']).requestAdapter();assert.ok(adapter);
const device=managedGPUDevice(await adapter.requestDevice({requiredFeatures:['float32-filterable','subgroups'].filter(f=>adapter.features.has(f)) as GPUFeatureName[],requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
const errors:string[]=[];device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});
let solver:WebGPUUniformReferenceSolver|undefined;
const trajectory:unknown[]=[],gauges:unknown[]=[];
const metrics={maxSeaweedY_m:-Infinity,minSeaweedY_m:Infinity,maxPistonError_m:0,firstOffshoreCrest_s:null as number|null,offshoreCrest_m:0,shoreMinimum_m:Infinity,shoreMaximum_m:-Infinity};
let verified=false;
try{
 const values=resolveMethodValues(method,'balanced',{...method.appDefaults,timeStep:'scene',rigidCoupling:'on'});
 solver=await method.createSolverAsync!(device,scene,'balanced',values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 console.log('solver ready');
 for(let step=1;step<=Math.round(seconds/dt);step++){
  assert.ok(solver.advanceTo(step*dt,bodies));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);
  if(step%6===0){
   const poses=await solver.readRigidBodyPoses();assert.ok(poses);trajectory.push({t:step*dt,poses});
   metrics.maxPistonError_m=Math.max(metrics.maxPistonError_m,Math.abs(poses[0]!.position_m.x-beachPistonPosition(step*dt)));
   for(const pose of poses.slice(1)){metrics.maxSeaweedY_m=Math.max(metrics.maxSeaweedY_m,pose.position_m.y);metrics.minSeaweedY_m=Math.min(metrics.minSeaweedY_m,pose.position_m.y);}
  }
  if(step%60===0){
   const address=await readUniformLiveAddress(device,solver);
   const [phi,tiles]=await Promise.all([readMixedTexture(device,solver.vertexPhiTexture!,address),readMixedTileWords(device,solver)]);
   const {nx,ny,nz}=solver.info,h=scene.voxelDomain!.finestCellSize_m;
   const vertex=(x:number,y:number,z:number)=>{
    let best=Infinity,origin:number[]|undefined;
    for(let k=0;k<8;k++){
     const c=[x-(k&1),y-((k>>1)&1),z-((k>>2)&1)];if(c.some((v,a)=>v<0||v>=[nx,ny,nz][a]!))continue;
     const word=tiles[uniformTileKey(address,c[0]!>>2,c[1]!>>2,c[2]!>>2)]!;
     if(!(word&0x80000000)&&(word&0x3fffffff)<best){best=word&0x3fffffff;origin=c.map(v=>v-v%4);}
    }
    const at=(a:number,b:number,c:number)=>phi[a+(nx+1)*(b+(ny+1)*c)]!;
    if(!origin)return at(x,y,z);const f=[(x-origin[0]!)/4,(y-origin[1]!)/4,(z-origin[2]!)/4];let value=0;
    for(let k=0;k<8;k++){const c=[k&1,(k>>1)&1,(k>>2)&1],w=c.reduce((s,v,a)=>s*(v?f[a]!:1-f[a]!),1);if(w)value+=w*at(origin[0]!+4*c[0]!,origin[1]!+4*c[1]!,origin[2]!+4*c[2]!);}
    return value;
   };
   const surface=(x:number,z:number)=>{const ix=Math.round((x+16)/h),iz=Math.round((z+12)/h);let top=0;
    for(let y=1;y<Math.min(solver!.info.ny-1,Math.ceil(6/h));y++){const a=vertex(ix,y,iz),b=vertex(ix,y+1,iz);if(a<=0&&b>0)top=(y-a/(b-a))*h;}
    return top;};
   const poses=(await solver.readRigidBodyPoses())!;
   const sample={t:step*dt,water:[-12,-9,-6,-3,0,2].map(x=>({x,y:surface(x,0)})),bodies:poses.map((p,i)=>({id:scene.rigidBodies[i]!.id,...p,waterY:surface(p.position_m.x,p.position_m.z)}))};
   gauges.push(sample);
   const crest=sample.water[0]!.y-2.4;metrics.offshoreCrest_m=Math.max(metrics.offshoreCrest_m,crest);
   if(crest>.08&&metrics.firstOffshoreCrest_s===null)metrics.firstOffshoreCrest_s=step*dt;
   if(step*dt>=3){metrics.shoreMinimum_m=Math.min(metrics.shoreMinimum_m,sample.water[4]!.y);metrics.shoreMaximum_m=Math.max(metrics.shoreMaximum_m,sample.water[4]!.y);}
   console.log(JSON.stringify(sample));
  }
 }
 assert.deepEqual(errors,[]);
 if(args.get('verify')==='true'){
  assert.ok(seconds>=12,'Verification must include startup and several arriving wave periods');
  assert.ok(metrics.maxPistonError_m<1e-4,'The GPU paddle must follow its authored track');
  assert.ok(metrics.maxSeaweedY_m<3.2&&metrics.minSeaweedY_m>1.5,`Seaweed must stay near the coastal surface: ${JSON.stringify(metrics)}`);
  assert.ok(metrics.firstOffshoreCrest_s!==null&&metrics.firstOffshoreCrest_s<=2,'An offshore wave must form within two seconds');
  assert.ok(metrics.shoreMaximum_m-metrics.shoreMinimum_m>.15,'Generated waves must propagate to the visible shoreline');
  verified=true;
 }
 console.log(JSON.stringify({metrics,verified}));
}finally{
 mkdirSync(dirname(output),{recursive:true});writeFileSync(output,JSON.stringify({sceneId:'beach',seconds,dt,rigidBodies:scene.rigidBodies,errors,metrics,verified,trajectory,gauges},null,2)+'\n');solver?.destroy();await device.queue.onSubmittedWorkDone();device.destroy();
}
