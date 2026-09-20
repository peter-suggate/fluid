/** Real owned-world + worker protocol + UI decoder scene acceptance; no mock physics. */
import assert from "node:assert/strict";
import {readFileSync,writeFileSync} from "node:fs";
import {findSceneDefinition} from "../../lib/core/scenes";
import {sceneDocument} from "../../lib/core/scene-definition";
import {UniformLabController,uniformLabSeed,UNIFORM_LAB_VALUES} from "../../lib/physics-wasm/uniform-controller";
import {UNIFORM_GEOMETRIC_DEFAULTS} from "../../lib/methods/uniform/uniform-geometric-parameters";
import {PhysicsWasmWorkerRuntime} from "../../lib/physics-wasm/worker-runtime";
import type {WorkerPort} from "../../lib/physics-wasm/client";
import type {PhysicsWorkerResponse,PhysicsWorkerRequest} from "../../lib/physics-wasm/protocol";
import type {FluidWasmModule} from "../../lib/physics-wasm/module";
const summaries:unknown[]=[];
const baseline=new Map<string,unknown>();
assert.deepEqual(UNIFORM_LAB_VALUES,{...UNIFORM_GEOMETRIC_DEFAULTS,activeRegion:"off"});
for(const artifact of ["scalar","simd"] as const){
  const root=new URL(`../../public/wasm/fluid-wasm/${artifact}/`,import.meta.url);
  const wasm=await import(new URL("fluid_wasm.js",root).href) as FluidWasmModule;
  await wasm.default({module_or_path:readFileSync(new URL("fluid_wasm_bg.wasm",root))});
  const factory=():WorkerPort=>{
    const listeners=new Set<(e:MessageEvent<PhysicsWorkerResponse>)=>void>();
    const runtime=new PhysicsWasmWorkerRuntime((message,transfer)=>{
      const data=structuredClone(message,{transfer:transfer??[]});
      queueMicrotask(()=>listeners.forEach(listener=>listener({data} as MessageEvent<PhysicsWorkerResponse>)));
    },async()=>({...wasm,default:async()=>{}}));
    return {postMessage(message:PhysicsWorkerRequest,transfer?:Transferable[]){runtime.receive(structuredClone(message,{transfer:transfer??[]}));},
      addEventListener(type:string,listener:unknown){if(type==="message")listeners.add(listener as (e:MessageEvent<PhysicsWorkerResponse>)=>void);},terminate(){listeners.clear();}};
  };
  const controller=await UniformLabController.create({artifact,workerFactory:factory});
  try{
    for(const sceneId of ["water-box-dam-break","minimal-power-dam-break-32","ceiling-slab-drop"]){
      const scene=sceneDocument(findSceneDefinition(sceneId)!);
      const seed=uniformLabSeed(scene);
      const initial=await controller.load(scene);
      assert.equal(initial.revision.frame,0);
      assert.deepEqual([...initial.volume],[...new Float32Array(seed.volume)]);
      assert.deepEqual([...initial.phi],[...new Float32Array(seed.phi)]);
      assert.equal(initial.receipt.method,"uniform-volume");
      const initialMass=initial.volume.reduce((a,b)=>a+b,0);
      let view=initial,dust=0;const costs:number[]=[];
      for(let frame=1;frame<=60;frame++){
        const start=performance.now();view=await controller.advance(1/30);costs.push(performance.now()-start);
        assert.equal(view.revision.frame,frame);assert.equal(view.revision.runEpoch,initial.revision.runEpoch);
        for(const field of [view.volume,view.phi,view.velocity,view.pressure])assert.ok(field.every(Number.isFinite),`${sceneId} ${frame}: finite fields`);
        const receipt=view.receipt.uniform as {transport:{dustVolume:number};sharpeningDust:number};dust+=receipt.transport.dustVolume+receipt.sharpeningDust;
      }
      const finalMass=view.volume.reduce((a,b)=>a+b,0);
      assert.ok(Math.abs(finalMass+dust-initialMass)<1e-5*Math.max(1,initialMass),`${sceneId}: conservative V with reported dust`);
      const final={volume:[...view.volume],phi:[...view.phi],velocity:[...view.velocity],pressure:[...view.pressure],released:[...view.released],tiles:[...view.tiles]};
      if(artifact==="scalar")baseline.set(sceneId,final);else assert.deepEqual(final,baseline.get(sceneId),`${sceneId}: scalar/SIMD owned-world parity`);
      const reset=await controller.load(scene);
      assert.equal(reset.revision.frame,0);assert.ok(reset.revision.runEpoch>initial.revision.runEpoch);
      assert.deepEqual(reset.volume,initial.volume);assert.deepEqual(reset.phi,initial.phi);
      const stepped=await controller.advance(1/30);assert.equal(stepped.revision.frame,1);
      assert.deepEqual(reset.volume,initial.volume,"publication stays owned after the slot is recycled");
      costs.sort((a,b)=>a-b);const summary={artifact,scene:sceneId,frames:60,dimensions:[view.nx,view.ny],initialMass,finalMass,dust,medianMs:costs[30],p95Ms:costs[57]};summaries.push(summary);console.log(JSON.stringify(summary));
    }
  }finally{await controller.destroy();}
}
writeFileSync("docs/research/uniform-geometric-2d-2026-09-20/ui-scene-acceptance.json",JSON.stringify({profile:UNIFORM_LAB_VALUES,scenes:summaries},null,2)+"\n");
