import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { adaptiveMassSolverOptions } from "../lib/methods/adaptive-volume/method";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { readPublishedCM12Field } from "../tools/sparse-cm12-published-field";

const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;
const live = new Set<GPU>();
dawnTest("UI bowl initializes and publishes its curved surface directly at coarse and fine resolution", { timeout: 120_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "stationary-bowl-scene");
  let gpu: GPU | undefined, device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis,dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
    for (const width of [4,1]) {
      const scene = sceneDocument(getSceneDefinition("stationary-bowl"));
      scene.fluid.refinementRegions![0]!.minimumCellSize_cells = width;
      scene.fluid.refinementRegions![0]!.maximumCellSize_cells = width;
      solver = await WebGPUAdaptiveMassSolver.createAsync(device,scene,"balanced",undefined,
        // Isolate initialization/publication from conditioning during the stationary check.
        adaptiveMassSolverOptions({ ...getSceneDefinition("stationary-bowl").methodProfile!.overrides,
          gammaDiffusion: "off", surfaceSharpening: "off" }),()=>{});
      await solver.waitForSimulationReady();
      solver.setTopologyFrozen(true);
      const initial = await solver.readDiagnosticFields(true);
      const activity = await solver.readGPUActivityPolicy();
      assert.ok(activity.bricks.every(b=>!b.active || b.acceptedResolution === 8/width));
      const [nx,ny,nz]=[solver.info.nx,solver.info.ny,solver.info.nz];
      assert.deepEqual([nx,ny,nz],[48,32,40]);
      for(let z=8;z<nz-8;z+=width) for(let x=8;x<nx-8;x+=width) {
        let mass=0;for(let y=0;y<ny;y++) mass+=initial.density[x+nx*(y+ny*z)]!;
        const expected=17.3+.003*((x+width/2-24)**2+.7*(z+width/2-20)**2+1.7*(width**2/12-1/768));
        assert.ok(Math.abs(mass-expected)<1e-5,`width ${width}, column ${x},${z}: ${mass} != ${expected}`);
      }
      // A curved surface must already be present when the user opens the paused scene.
      const phi = (await readPublishedCM12Field(device,solver)).values;
      const crossing=(x:number,z:number)=>{
        for(let y=0;y<ny-1;y++) {
          const lo=phi[x+nx*(y+ny*z)]!,hi=phi[x+nx*(y+1+ny*z)]!;
          if(lo<=0&&hi>0)return (y+.5-lo/(hi-lo))*.05;
        }
        return NaN;
      };
      assert.ok(crossing(8,20)-crossing(23,20)>.025,"reset publication is the bowl, not a flat tank");
      while(!solver.advanceTo(1/60,[]))await new Promise(setImmediate);
      await solver.waitForTopologyReady();
      const after=await solver.readDiagnosticFields(true);
      assert.deepEqual(after.density,initial.density,"pressing play must leave the source stationary");
      const frame=await solver.readFrameControlQA();assert.ok(frame);assert.equal(frame.fault,0);
      assert.deepEqual(errors,[]);
      solver.destroy();solver=undefined;
    }
  } finally {
    solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();if(gpu)live.delete(gpu);
  }
});
