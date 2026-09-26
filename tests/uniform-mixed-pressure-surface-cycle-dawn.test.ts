import { cm12GhostFluidTheta } from "../lib/core/cm12-numerics";
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { MixedPressureCycleDawn } from "./helpers/uniform-mixed-pressure-cycle-dawn";
import { mixedPressureFixture, mixedPressureLayouts, geometricDivergence } from "./helpers/uniform-mixed-pressure";

const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("mixed ghost-fluid pressure cycles converge without analytic pressure slopes", {timeout:180000}, async t=>{
  await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed free-surface pressure cycles");
  let device:GPUDevice|undefined;
  try {
    const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
    const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
    device=await adapter.requestDevice();
    const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
    for(const [index,layout] of mixedPressureLayouts().entries()) {
      const cycle=new MixedPressureCycleDawn(device,layout,false,true);
      const bounded=new MixedPressureCycleDawn(device,layout,true,true);
      try {
        await cycle.initialize(); await bounded.initialize();
        const fine=mixedPressureFixture(layout);
        for(let vertical=0;vertical<4;vertical++) for(const direction of [-1,1]) for(const mode of (vertical===3 ? ["random"] as const : ["hydrostatic","random","bounded"] as const)) await t.test(`fixture ${index}, ${vertical===3?"curved":"axis "+vertical}, sign ${direction}, ${mode}`,async()=>{
          const height=vertical<3?(layout.lattice.dimensions[vertical]!*.5+.25)*layout.lattice.cellSize_m[vertical]!:0;
          // Supply only finest-owner geometry. The GPU restricts phi using
          // the native C=2 rule; no pressure or slopes are supplied.
          const phi=[Float32Array.from(fine.cells,c=>direction*(vertical===3
            ? Math.hypot(...c.center.map((v,a)=>v/(layout.lattice.dimensions[a]!*layout.lattice.cellSize_m[a]!)-.4))-.32
            : c.center[vertical]!-height))];
          const velocity=Float64Array.from(fine.faces,(f,k)=>(phi[0]![f.left]!<0||phi[0]![f.right]!<0) ? (mode === "random" ? Math.sin(k*13) : f.axis===vertical?-direction:0) : 0);
          const rhs=Float32Array.from(geometricDivergence(fine,velocity),(v,i)=>phi[0]![i]!<0?-v:0);
          const actual=mode==="bounded"?await bounded.solve(rhs,new Float32Array(fine.cells.length),phi[0]):await cycle.solve(rhs,undefined,phi[0]);
          assert.equal(actual.coarseExhausted,0);
          assert.ok(actual.residuals.at(-1)!<1e-3,`residuals ${actual.residuals}`);
          const error=Math.max(...actual.pressure.map((p,i)=>phi[0]![i]!<0?Math.abs(p+phi[0]![i]!):0));
          // Pressure itself has units and can accumulate a harmless long-wave
          // error under the native coarse tolerance. Match the existing gate:
          // actual velocity correction and physical divergence, both <1e-3.
          const gradient=Float64Array.from(fine.faces,f=>{
            const l=phi[0]![f.left]!<0,r=phi[0]![f.right]!<0;
            if(!l&&!r)return 0;
            if(l!==r){
              const liquid=l?f.left:f.right,air=l?f.right:f.left;
              let tangential=0;
              for(let a=0;a<3;a++)if(a!==f.axis)tangential+=actual.slopes[4*liquid+a]!*(fine.cells[f.right]!.center[a]!-fine.cells[f.left]!.center[a]!);
              return ((l?-1:1)*actual.pressure[liquid]!/cm12GhostFluidTheta(phi[0]![liquid]!,phi[0]![air]!,1e-9)-tangential)/f.distance;
            }
            let difference=actual.pressure[f.right]!-actual.pressure[f.left]!;
            for(const [i,other,sign] of [[f.left,f.right,-1],[f.right,f.left,1]]){
              if(fine.cells[i!]!.width<=fine.cells[other!]!.width)continue;
              for(let a=0;a<3;a++)if(a!==f.axis)difference+=sign!*actual.slopes[4*i!+a]!*(f.center[a]!-fine.cells[i!]!.center[a]!);
            }
            return difference/f.distance;
          });
          const projected=velocity.map((v,i)=>v-gradient[i]!);
          if(mode !== "random")assert.ok(Math.max(...projected.map(Math.abs))<1e-3,"hydrostatic parasitic velocity");
          const divergence=geometricDivergence(fine,projected);
          assert.ok(Math.max(...divergence.map((v,i)=>phi[0]![i]!<0?Math.abs(v):0))<1e-3,"independent liquid divergence");
          t.diagnostic(`${index}/${vertical}/${direction}/${mode}: residual ${actual.residuals.at(-1)}${mode!=="random"?`, pressure error ${error}`:""}`);
        });
      } finally {cycle.destroy();bounded.destroy();}
    }
    assert.deepEqual(errors,[]);
  } finally {device?.destroy();await releaseWebGPUExclusiveLock();}
});
