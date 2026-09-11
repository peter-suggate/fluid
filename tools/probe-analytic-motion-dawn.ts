/** Same scenes/profile as the UI. Reports analytic errors without hiding failing physics.
 * Run: WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-analytic-motion-dawn.ts
 * Add --verify to enforce the declared analytic error budgets.
 */
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { mkdirSync, writeFileSync } from "node:fs";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

import { measureAnalyticMotionPublishedSurface } from "./analytic-motion-published-surface";

const modulePath = process.env.WEBGPU_NODE_MODULE;
assert.ok(modulePath, "WEBGPU_NODE_MODULE is required");
const h=.05;
const control=process.argv.includes("--fine") ? "fine" : "mixed";
const seamWidthArg=process.argv.find(a=>a.startsWith("--seam-width="))?.split("=")[1];
const seamWidth=seamWidthArg?Number(seamWidthArg):8;
assert.ok([2,4,8].includes(seamWidth), "seam width must describe adjacent 2:1 rungs");
const conditioningOff=process.argv.includes("--conditioning-off");
const motions=process.argv.includes("--rerung") ? ["free-fall-rerung"] : ["translation","free-fall"];
const failures: string[]=[];
const results: unknown[]=[];
await acquireWebGPUExclusiveLock("dawn-test", "analytic-motion");
try {
  const dawn=await import(pathToFileURL(modulePath).href); Object.assign(globalThis,dawn.globals);
  const gpu: GPU=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  for(const motion of motions) {
    const adapter=await gpu.requestAdapter(); assert.ok(adapter);
    const device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const definition=getSceneDefinition(`coarse-surface-${motion}`), scene=sceneDocument(definition);
      const dims=[32,Math.round(scene.container.height_m/h),8] as const;
      if(control==="fine") {
        scene.fluid.refinementRegions=[{id:"fine-control",rule:"minimum-cell-size",minimumCellSize_cells:1,maximumCellSize_cells:1,
          min_m:{x:-.8,y:0,z:-.2},max_m:{x:.8,y:scene.container.height_m,z:.2}}];
        delete scene.fluid.refinementKeyframes;
      }
      if(control!=="fine"&&seamWidth!==8) {
        assert.notEqual(motion,"free-fall-rerung","seam-width control is fixed topology");
        scene.fluid.refinementRegions=scene.fluid.refinementRegions!.map((region,index)=>({...region,
          minimumCellSize_cells:index===0?seamWidth:seamWidth/2,
          maximumCellSize_cells:index===0?seamWidth:seamWidth/2}));
      }
      const values=resolveMethodValues(adaptiveMassMethod,"balanced", {...definition.methodProfile!.overrides,...(conditioningOff?{gammaDiffusion:"off",surfaceSharpening:"off"}:{})});
      solver=await adaptiveMassMethod.createSolverAsync!(device,scene,"balanced",values,undefined,()=>{}) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady();
      const dt=scene.numerics.fixedDt_s, v0=scene.fluid.initialVelocity_m_s!.y, g=scene.fluid.gravity_m_s2.y;
      const samples=[];
      for(let step=0;step<=Math.round(scene.duration_s/dt);step++) {
        if(step) {
          while(!solver.advanceTo(step*dt,[])) await new Promise(setImmediate);
          await solver.waitForTopologyReady(); await solver.assertSimulationHealthy();
        }
        const fields=await solver.readDiagnosticFields(true);
        assert.ok(fields.density.every(Number.isFinite) && fields.velocity.every(Number.isFinite));
        const activity: Awaited<ReturnType<WebGPUAdaptiveMassSolver["readGPUActivityPolicy"]>> = await solver.readGPUActivityPolicy();
        let mass=0, moment=0, momentumY=0, lateral2=0, l1=0, floorBandMass=0, maximumDensity=0, wetSeedMass=0;
        const rungAt=new Map(activity.bricks.filter(b=>b.active).map(b=>[b.coordinate.join("/"),b.acceptedResolution]));
        const t=step*dt, displacement=v0*t+.5*g*t*t;
        // CM12 transports with the previous velocity, then applies gravity.
        const discreteDisplacement=v0*t+.5*g*dt*dt*step*(step-1);
        const bottom=.8+discreteDisplacement, top=1.2+discreteDisplacement;
        const halfMass=[0,0], halfMoment=[0,0];
        for(let z=0;z<dims[2];z++) for(let y=0;y<dims[1];y++) for(let x=0;x<dims[0];x++) {
          const i=x+dims[0]*(y+dims[1]*z), rho=fields.density[i]!;
          // Compare analytic volume averages on the actual accepted cells,
          // not a sharp finest-cell raster against replicated coarse values.
          const rung=rungAt.get([Math.floor(x/8),Math.floor(y/8),Math.floor(z/8)].join("/")) ?? (control === "fine" ? 8 : x<16?8/seamWidth:16/seamWidth);
          const width=8/rung, y0=Math.floor(y/width)*width*h;
          const x0=Math.floor(x/width)*width*h;
          const expected=Math.max(0,Math.min(x0+width*h,1.2)-Math.max(x0,.4))/(width*h)
            * Math.max(0,Math.min(y0+width*h,top)-Math.max(y0,bottom))/(width*h);
          mass+=rho;maximumDensity=Math.max(maximumDensity,rho);if(rho>.5)wetSeedMass+=rho; if(y<8)floorBandMass+=rho; moment+=rho*(y+.5)*h; momentumY+=rho*fields.velocity[4*i+1]!;
          lateral2+=rho*(fields.velocity[4*i]!**2+fields.velocity[4*i+2]!**2);
          l1+=Math.abs(rho-expected);
          const half=x<16?0:1;halfMass[half]+=rho;halfMoment[half]+=rho*(y+.5)*h;
        }
        const observedSurfaceRungs=new Set<number>();
        const surfaceSpeedByRung=new Map<number,number>();
        for(let x=8;x<24;x++) for(let z=0;z<8;z++) {
          for(let y=dims[1]-1;y>=0;y--) if(fields.density[x+32*(y+dims[1]*z)]!>1e-6) {
            const rung=rungAt.get([Math.floor(x/8),Math.floor(y/8),0].join("/"));
            if(rung!==undefined) {
              observedSurfaceRungs.add(rung);
              const i=x+32*(y+dims[1]*z);
              const speed=Math.hypot(fields.velocity[4*i]!,fields.velocity[4*i+1]!,fields.velocity[4*i+2]!);
              surfaceSpeedByRung.set(rung,Math.max(surfaceSpeedByRung.get(rung)??0,speed));
            }
            break;
          }
        }
        const surfaceRungs=[...observedSurfaceRungs].sort();
        const sample={step,t,mass_m3:mass*h**3,massRelativeError:Math.abs(mass/1024-1),
          centerY_m:moment/mass,centerError_m:moment/mass-(1+displacement),
          discreteCenterError_m:moment/mass-(1+discreteDisplacement),
          velocityY_m_s:momentumY/mass,velocityError_m_s:momentumY/mass-(v0+g*t),
          lateralRms_m_s:Math.sqrt(Math.max(0,lateral2/mass)),cellAverageRelativeL1:l1/1024,
          halfCenterDifference_m:halfMoment[0]/halfMass[0]-halfMoment[1]/halfMass[1],surfaceRungs,
          surfaceSpeedByRung:Object.fromEntries(surfaceSpeedByRung),floorBandMassFraction:floorBandMass/1024,
          acceptedTopologyGeneration:activity.acceptedTopologyGeneration,maximumDensity,wetSeedMassFraction:wetSeedMass/1024,
          acceptedRungs:activity.bricks.filter(b=>b.active).map(b=>[b.coordinate,b.acceptedResolution])};
        assert.equal(solver.info.encodedSteps,step);
        if(step===0) {
          assert.ok(sample.massRelativeError<1e-7,`exact starting box volume: ${JSON.stringify({sample,dims,actual:[solver.info.nx,solver.info.ny,solver.info.nz]})}`);
          assert.ok(Math.abs(sample.velocityError_m_s)<1e-6,"starting velocity must reach accepted cells");
          assert.deepEqual(surfaceRungs,control==="fine"?[8]:[8/seamWidth,16/seamWidth],"both authored surface rungs must be accepted");
        }
        if(motion === "free-fall-rerung" && control !== "fine" && (step === 7 || step === 13)) {
          const halves=[1,2].map(x=>activity.bricks.find(b=>b.active&&b.coordinate[0]===x&&b.coordinate[1]===1&&b.coordinate[2]===0)?.acceptedResolution);
          assert.deepEqual(halves,step===7?[2,1]:[1,2],"timed region edit must reach accepted topology without resetting motion");
        }
        if(process.argv.includes("--surface") && (step===0||step===Math.round(scene.duration_s/dt))) {
          const surface=await measureAnalyticMotionPublishedSurface(device,solver,h,bottom,top);
          Object.assign(sample,{publishedSurface:surface});
          console.log(JSON.stringify({motion,step,publishedSurface:{...surface,columns:undefined}}));
        }
        samples.push(sample);
        if(step>0 && [...surfaceSpeedByRung.values()].some(speed=>speed<1e-5)) failures.push(`${motion} step ${step}: a surface rung is stationary`);
        if(control!=="fine" && surfaceRungs.length<2) failures.push(`${motion} step ${step}: surface is not mixed resolution`);
        if(sample.cellAverageRelativeL1>.1) failures.push(`${motion} step ${step}: cell-average shape error > 10%`);
        if(sample.massRelativeError>1e-4) failures.push(`${motion} step ${step}: mass error > 0.01%`);
        if(Math.abs(sample.discreteCenterError_m)>.025) failures.push(`${motion} step ${step}: centre error > half a finest cell`);
        if(Math.abs(sample.velocityError_m_s)>.02) failures.push(`${motion} step ${step}: vertical velocity error > 0.02 m/s`);
        if(sample.lateralRms_m_s>.01) failures.push(`${motion} step ${step}: lateral velocity > 0.01 m/s`);
        if(Math.abs(sample.halfCenterDifference_m)>.025) failures.push(`${motion} step ${step}: coarse/fine centres differ by > half a finest cell`);
        if(step%6===0) console.log(JSON.stringify({motion,...sample,acceptedRungs:undefined}));
      }
      results.push({motion,control,seamWidth,conditioningOff,samples});
    } finally {solver?.destroy();device.destroy();}
  }
} finally {await releaseWebGPUExclusiveLock();}
mkdirSync("artifacts/analytic-motion",{recursive:true});
writeFileSync(`artifacts/analytic-motion/${motions.length===1?"rerung-":""}${control}${control!=="fine"&&seamWidth!==8?`-width${seamWidth}`:""}${conditioningOff?"-conditioning-off":""}.json`,JSON.stringify({results,failures},null,2));
console.log(JSON.stringify({failures}));
if(process.argv.includes("--verify")) assert.deepEqual(failures,[]);
