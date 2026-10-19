/** Paired production drift/impact/settled steps, with alternating arm order.
 * Run under lockf /private/tmp/claude-501/fluid-gpu.lock.
 * SCENARIO=drift STEPS=150 FROM=100 REPEATS=2 OUT=/tmp/paired.json
 * Defaults compare all retained fusions with their separate paths.
 * ARMS='[{"bandCycles":3},{"bandCycles":4}]'
 * To check band noise, use ARMS='[{"bandExit":"strict"},{"bandExit":"returns"}]'.
 * Timings include encoding and GPU completion, exclude all diagnostic reads.
 * A source fingerprint is checked at the end; use a frozen source copy when
 * other sessions edit the shared checkout. GPU contention remains external. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { createBackyardPoolDriftScene, createBackyardPoolPartyScene } from "../lib/core/backyard-pool-party-scene";
import type { MethodParamValues } from "../lib/core/method-contract";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { createUniformSolver, withUniformDevice } from "../tests/helpers/uniform-geometric";
import { readMixedBuffer } from "../tests/helpers/uniform-mixed-native-fields";
import { UNIFORM_DETAIL_CRITERIA, UNIFORM_STAGE_IMPORTANCE } from "../lib/methods/uniform/uniform-stage-grids";
import type { UniformMixedFrame } from "../lib/methods/uniform/uniform-mixed-frame";

const scenario=process.env.SCENARIO??"drift",steps=Number(process.env.STEPS??150),from=Number(process.env.FROM??100),repeats=Number(process.env.REPEATS??2);
assert.ok(["drift","pool-impact","settled"].includes(scenario));
assert.ok(Number.isInteger(steps)&&Number.isInteger(from)&&from>0&&steps>=from&&Number.isInteger(repeats)&&repeats>0);
const arms:MethodParamValues[]=JSON.parse(process.env.ARMS??'[{"rootReserve":"parallel","rootCoarse":"tiled","rootTiles":"octant","surfaceSplat":"separate"},{}]');
assert.equal(arms.length,2);
const out=resolve(process.env.OUT??"/tmp/near-settled-paired.json");
process.env.WEBGPU_NODE_MODULE??=resolve("node_modules/webgpu/index.js");
const fingerprint=()=>{const hash=createHash("sha256");for(const f of readdirSync("lib/methods/uniform").filter(f=>f.endsWith(".ts")).sort())hash.update(f).update(readFileSync(`lib/methods/uniform/${f}`));return hash.digest("hex");};
const sourceHash=fingerprint();
const scene=scenario==="drift"?createBackyardPoolDriftScene():createBackyardPoolPartyScene();
if(scenario==="pool-impact"){
 const box=scene.fluid.initialLiquidVolumes!.find(v=>v.shape==="box");assert.ok(box?.shape==="box");
 Object.assign(scene.rigidBodies[0]!,{position_m:{x:0,y:box.max_m.y+.4,z:0},linearVelocity_m_s:{x:0,y:-8,z:0}});
}
assert.ok(steps*scene.numerics.fixedDt_s<=scene.duration_s);
paceDawnEventPolling();
const mean=(a:number[])=>a.reduce((s,x)=>s+x,0)/a.length;
const median=(a:number[])=>{const b=[...a].sort((x,y)=>x-y);return b[b.length>>1]!;};
const runs:unknown[]=[];
await withUniformDevice("paired near-settled pool",async device=>{
 for(let repeat=0;repeat<repeats;repeat++){
  const solvers:Awaited<ReturnType<typeof createUniformSolver>>[]=[],bodies=arms.map(()=>initializeRigidBodies(scene.rigidBodies));
  const rows:{frame:number;first:number;ms:number[];state:unknown[]}[]=[];
  try{
   for(let arm=0;arm<2;arm++){
    solvers.push(await createUniformSolver(device,structuredClone(scene),arms[arm]));
    console.log(JSON.stringify({phase:"constructed",repeat,arm,sourceHash}));
   }
   for(let frame=1;frame<=steps;frame++){
    const first=(frame+repeat)&1,ms=[0,0],state:unknown[]=[];
    for(const arm of [first,1-first]){
     const solver=solvers[arm]!,start=performance.now();
     assert.ok(solver.advanceTo(frame*scene.numerics.fixedDt_s,bodies[arm]!));await solver.awaitFrameCompletion();ms[arm]=performance.now()-start;
     assert.equal(solver.info.simulationPipelineError,undefined);
     const inner=solver as unknown as {rigidSystem:{stateBuffer:GPUBuffer}};
     const rigid=await readMixedBuffer(device,inner.rigidSystem.stateBuffer);
     state[arm]={particles:solver.narrowBandFlipInfo!.particles,bandTiles:solver.info.uniformPressureBandTiles,
      bandCycles:solver.info.uniformPressureBandCycles,bandResidual:solver.info.uniformPressureBandResidual,
      rootResidual:solver.info.uniformPressureAcceptedResidual,bodyPosition:Array.from(rigid.slice(0,3)),bodyVelocity:Array.from(rigid.slice(12,15))};
    }
    rows.push({frame,first,ms,state});
    if(frame%30===0||frame===steps)console.log(JSON.stringify({repeat,...rows.at(-1)}));
   }
   const physics=[];for(const solver of solvers){
    const stats=await solver.readStats(),frame=(solver as unknown as {mixedFrame:UniformMixedFrame}).mixedFrame;
    const particles=frame.narrowBandFlip,tiles=frame.ownership.capacity.tiles;
    const read=device.createBuffer({size:8*tiles,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const triggers=Object.fromEntries(UNIFORM_DETAIL_CRITERIA.map(k=>[k,0]));
    try{
     const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(particles.airGateBuffer,4*(particles.activityHeatWord-2*tiles),read,0,read.size);
     device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const words=new Uint32Array(read.getMappedRange());
     for(let t=0;t<tiles;t++){
      const flags=words[2*t+1]!;if(flags&UNIFORM_STAGE_IMPORTANCE.dropped)continue;
      const mask=(flags>>>UNIFORM_STAGE_IMPORTANCE.triggeredShift)&63;
      UNIFORM_DETAIL_CRITERIA.forEach((k,i)=>{if(mask&(1<<i))triggers[k]!++;});
     }
    }finally{if(read.mapState==="mapped")read.unmap();read.destroy();}
    physics.push({maxSpeed:stats.maxSpeed_m_s,volumeDrift:stats.volumeDrift,finalAdmittedTriggers:triggers});
   }
   const measured=rows.filter(r=>r.frame>=from),summary={median_ms:arms.map((_,arm)=>median(measured.map(r=>r.ms[arm]!))),
    mean_ms:arms.map((_,arm)=>mean(measured.map(r=>r.ms[arm]!))),pairedMedianSaved_ms:median(measured.map(r=>r.ms[0]!-r.ms[1]!)),
    meanParticles:arms.map((_,arm)=>mean(measured.map(r=>(r.state[arm] as {particles:number}).particles)))};
   runs.push({repeat,summary,physics,rows});console.log(JSON.stringify({repeat,summary,physics}));
  }catch(error){mkdirSync(dirname(out),{recursive:true});writeFileSync(out+".failed.json",JSON.stringify({sourceHash,scenario,arms,repeat,error:String(error),rows},null,2));throw error;}
  finally{for(const solver of solvers)solver.destroy();}
 }
});
const sourceUnchanged=sourceHash===fingerprint();
mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify({sourceHash,sourceUnchanged,scenario,steps,from,arms,runs},null,2)+"\n");
assert.ok(sourceUnchanged,"sources changed during measurement; repeat from a frozen source copy");
console.log(JSON.stringify({out,sourceUnchanged}));
