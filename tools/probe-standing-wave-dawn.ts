/** Analytic seiche receipts for both UI presets. --fine runs the same solver at B8. */
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { mkdirSync,writeFileSync } from "node:fs";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { standingWaveOmega,STANDING_WAVE } from "../lib/core/analytic-motion-scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
const fine=process.argv.includes("--fine");
const mode=process.argv.find(a=>a.startsWith("--mode="))?.split("=")[1];
const stepsArg=process.argv.find(a=>a.startsWith("--steps="))?.split("=")[1];
const results:unknown[]=[];
await acquireWebGPUExclusiveLock("dawn-probe","standing-wave");
try {
  assert.ok(process.env.WEBGPU_NODE_MODULE);
  const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE).href);Object.assign(globalThis,dawn.globals);
  const gpu:GPU=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
  for(const live of mode==="live"?[true]:mode==="fixed"?[false]:[false,true]){
    const definition=getSceneDefinition(live?"coarse-surface-standing-wave-live":"coarse-surface-standing-wave");
    const scene=sceneDocument(definition);
    if(fine)scene.fluid.refinementRegions=[{id:"fine-control",rule:"minimum-cell-size",minimumCellSize_cells:1,maximumCellSize_cells:1,
      min_m:{x:-.8,y:0,z:-.2},max_m:{x:.8,y:1.2,z:.2}}];
    const adapter=await gpu.requestAdapter();assert.ok(adapter);
    const device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
    let solver:WebGPUAdaptiveMassSolver|undefined;
    const samples:unknown[]=[];let promoted=0,demoted=0,mixedMovingFrames=0;
    let previous=new Map<string,number>();
    try{
      const values=resolveMethodValues(adaptiveMassMethod,"balanced",definition.methodProfile!.overrides);
      solver=await adaptiveMassMethod.createSolverAsync!(device,scene,"balanced",values,undefined,()=>{}) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady();
      const dt=scene.numerics.fixedDt_s,steps=stepsArg?Number(stepsArg):Math.ceil(scene.duration_s/dt);
      for(let step=0;step<=steps;step++){
        if(step){while(!solver.advanceTo(step*dt,[]))await new Promise(setImmediate);
          await solver.waitForTopologyReady();await solver.assertSimulationHealthy();}
        assert.equal(solver.info.encodedSteps,step);
        const activity=await solver.readGPUActivityPolicy();
        const rungs=new Map(activity.bricks.filter(b=>b.active).map(b=>[b.coordinate.join("/"),b.acceptedResolution]));
        for(const [key,rung] of rungs){const old=previous.get(key);if(old!==undefined){if(rung>old)promoted++;if(rung<old)demoted++;}}
        previous=rungs;
        const fields=await solver.readDiagnosticFields(true);
        assert.ok(fields.density.every(Number.isFinite)&&fields.velocity.every(Number.isFinite));
        const heights=new Float64Array(32);const speeds=new Map<number,number>();
        let mass=0;
        for(let x=0;x<32;x++)for(let z=0;z<8;z++){
          let top=-1;
          for(let y=0;y<24;y++){
            const i=x+32*(y+24*z),rho=fields.density[i]!;
            heights[x]+=rho*.05/8;mass+=rho*.05**3;
            if(rho>.01)top=y;
          }
          if(top>=0){const i=x+32*(top+24*z),rung=rungs.get([Math.floor(x/8),Math.floor(top/8),0].join("/"));
            if(rung!==undefined)speeds.set(rung,Math.max(speeds.get(rung)??0,Math.hypot(...fields.velocity.slice(4*i,4*i+3))));}
        }
        // Integrated column heights, compared with cell-averaged cosine mode.
        const k=Math.PI/STANDING_WAVE.length_m,sinc=Math.sin(k*.025)/(k*.025);
        let numerator=0,denominator=0,rms=0;
        const t=step*dt,expectedAmplitude=STANDING_WAVE.amplitude_m*Math.cos(standingWaveOmega()*t);
        for(let x=0;x<32;x++){
          const basis=Math.cos(k*(x+.5)*.05)*sinc;
          numerator+=(heights[x]!-.6)*basis;denominator+=basis*basis;
          rms+=(heights[x]!-.6-expectedAmplitude*basis)**2/32;
        }
        const movingRungs=[...speeds].filter(([,speed])=>speed>1e-5).map(([rung])=>rung).sort();
        if(movingRungs.length>1)mixedMovingFrames++;
        const sample={step,t,mass_m3:mass,massRelativeError:Math.abs(mass/.384-1),
          amplitude_m:numerator/denominator,expectedAmplitude_m:expectedAmplitude,heightRmsError_m:Math.sqrt(rms),
          surfaceRungs:[...speeds.keys()].sort(),movingRungs,surfaceSpeedByRung:Object.fromEntries(speeds),promoted,demoted};
        samples.push(sample);
        if(step%15===0||step===steps)console.log(JSON.stringify({live,fine,...sample}));
      }
      if(!fine && steps>=60) assert.ok(mixedMovingFrames>0,"wave must exercise a moving mixed-resolution surface");
      if(live && !fine && steps>=180) {
        assert.ok(promoted>0 && demoted>0,"live wave must both refine and coarsen during motion");
      }
      results.push({live,fine,mixedMovingFrames,promoted,demoted,samples});
    }catch(error){results.push({live,fine,mixedMovingFrames,promoted,demoted,samples,error:String(error)});throw error;}
    finally{solver?.destroy();device.destroy();}
  }
}finally{
  mkdirSync("artifacts/analytic-motion",{recursive:true});
  writeFileSync(`artifacts/analytic-motion/wave-${fine?"fine":"mixed"}-${mode??"both"}.json`,JSON.stringify(results,null,2));
  await releaseWebGPUExclusiveLock();
}
