import assert from "node:assert/strict";
import test from "node:test";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {uniformNarrowBandMethod} from "../lib/methods/uniform/uniform-narrow-band-method";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {advanceUniform,withUniformDevice} from "./helpers/uniform-geometric";

const gpuTest=process.env.WEBGPU_NODE_MODULE?test:test.skip;

// Exercise flight, impact, spreading and settling. A final-only measurement
// can hide early loss followed by reconstruction gain (the old no-reseed
// control lost 49% before returning to within 3% of its initial volume).
for(const dt of [0.017,0.05])gpuTest(`NB Figure 2 bounds geometric volume throughout 5.1s at dt=${dt}`,{timeout:180_000},async()=>{
 await withUniformDevice("NB Figure 2 volume",async device=>{
  const scene=structuredClone(sceneDocument(getSceneDefinition("cm12-figure-2")));
  scene.numerics={...scene.numerics,fixedDt_s:dt,maxDt_s:dt};
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",{...uniformNarrowBandMethod.appDefaults,timeStep:"scene"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   const samples:{time:number;drift:number}[]=[];
   for(let step=1;step<=Math.round(5.1/dt);step++){
    await advanceUniform(solver,step*dt);
    assert.ok(Math.abs((solver.info.completedTime_s??0)-step*dt)<1e-8,"retain the requested global timestep");
    if(step%10===0||step===Math.round(5.1/dt)){
     const stats=await solver.readStats();const drift=Number(stats.volumeDrift);
     samples.push({time:step*dt,drift});
     assert.ok(Number.isFinite(drift)&&Math.abs(drift)<0.1,`geometric volume at ${step*dt}s: ${(100*drift).toFixed(3)}%`);
    }
   }
   console.log(JSON.stringify({dt,volumeSamples:samples}));
  }finally{solver.destroy();}
 });
});
