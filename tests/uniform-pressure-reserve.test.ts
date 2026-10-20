import assert from "node:assert/strict";
import test from "node:test";
import { createRegionTierScene } from "../tools/region-tier-scenes";
import type { UniformMixedPressureCycles } from "../lib/methods/uniform/uniform-mixed-pressure-cycles";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

async function read(device:GPUDevice,range:GPUBufferBinding):Promise<Uint32Array<ArrayBuffer>>{
 const size=range.size??range.buffer.size-(range.offset??0);
 const staging=device.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 try{
  const e=device.createCommandEncoder();e.copyBufferToBuffer(range.buffer,range.offset??0,staging,0,size);device.queue.submit([e.finish()]);
  await staging.mapAsync(GPUMapMode.READ);return new Uint32Array(staging.getMappedRange()).slice();
 }finally{if(staging.mapState==="mapped")staging.unmap();staging.destroy();}
}

test("page smoothing and fused safety traversals preserve open V/Full solves and closed slots",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("fused pressure safety slots",async device=>{
  const scene=createRegionTierScene("splash",128);scene.fluid.pagePool={blocks:34};
  const solver=await createUniformSolver(device,scene,{rootReserve:"parallel",rootCoarse:"tiled",surfaceSplat:"separate"});
  try{
   assert.ok(solver.advanceTo(1/60,[]));await solver.awaitFrameCompletion();
   const cycles=(solver as unknown as {mixedFrame:{cycles:UniformMixedPressureCycles}}).mixedFrame.cycles;
   type Launch={pipeline:GPUComputePipeline;tiles:number;listed:boolean};
   const internal=cycles as unknown as {fusedReserve:boolean;resourceEntries:GPUBindGroupEntry[];pipelines:Map<string,Launch>};
   const roots=[...internal.pipelines].filter(([name])=>name.startsWith("smooth.0.")).map(([,launch])=>({launch,page:{...launch}}));
   const tiled=[...internal.pipelines].find(([name])=>name.startsWith("smooth.1."))![1].pipeline;
   const root=cycles.level,gate=4*(9*root.ownership.layout.tiles.length+24);
   device.queue.writeBuffer(root.ownership.support,gate,new Uint32Array([0]));
   // Pressure scratch is borrowed after the root phase. Supply an explicit
   // bounded correction problem instead of replaying expired frame words.
   const count=896*root.ownership.layout.address.blocks;
   const pressure=new Float32Array(count),rhs=new Float32Array(count),minimum=new Float32Array(count).fill(-3.402823e38);
   for(let i=0;i<count;i++){pressure[i]=.01*Math.sin(i*.07);rhs[i]=.1*Math.sin(i*.13);if(i%257===0)minimum[i]=-.001;}
   for(const [range,values] of [[root.pressure,pressure],[root.rhs[0],rhs],[root.minimum[0],minimum]] as const)device.queue.writeBuffer(range.buffer,range.offset??0,values);
   const setup=device.createCommandEncoder();cycles.encodeSetup(setup);device.queue.submit([setup.finish()]);
   const ranges=[0,1,3,5].map(binding=>internal.resourceEntries.find(e=>e.binding===binding)!.resource as GPUBufferBinding);
   const saved=await Promise.all(ranges.map(r=>read(device,r)));
   const restore=()=>{for(let i=0;i<ranges.length;i++)device.queue.writeBuffer(ranges[i]!.buffer,ranges[i]!.offset??0,saved[i]!);};
   for(const full of [false,true]){
    const results=[],residuals=[];
    for(const arm of ["octant","page","reserve"]){
     restore();internal.fusedReserve=arm==="reserve";
     for(const {launch,page} of roots)Object.assign(launch,arm==="octant"?{pipeline:tiled,tiles:8*root.ownership.layout.address.blocks,listed:true}:{...page,tiles:2});
     const e=device.createCommandEncoder();
     if(full)cycles.encodeFullCycle(e,true);else cycles.encodeVCycle(e,true);
     cycles.encodeMeasure(e);device.queue.submit([e.finish()]);results.push(await read(device,root.pressure));residuals.push(new Float32Array((await read(device,root.residual)).buffer));
    }
    for(let arm=1;arm<3;arm++){
    const reference=new Float32Array(results[0]!.buffer),candidate=new Float32Array(results[arm]!.buffer);
    if(arm===1)assert.deepEqual(results[arm],results[0],"page and octant visits are bit-identical");
    let absolute=0,relative=0;
    for(let i=0;i<reference.length;i++){
     assert.ok(Number.isFinite(candidate[i]!));
     const error=Math.abs(candidate[i]!-reference[i]!);absolute=Math.max(absolute,error);relative=Math.max(relative,error/Math.max(1,Math.abs(reference[i]!)));
    }
    console.log({cycle:full?"Full":"V",arm,absolute,relative});
    // The storage traversal and the temporal-halo shader can contract f32
    // expressions differently. Bound that rounding and compare the actual
    // projected residual, which is the production acceptance quantity.
    for(let i=0;i<reference.length;i++)assert.ok(Math.abs(candidate[i]!-reference[i]!)<=8e-6+8e-7*Math.abs(reference[i]!),`pressure row ${i}`);
    let residualError=0;
    for(let i=0;i<residuals[0]!.length;i++)residualError=Math.max(residualError,Math.abs(residuals[0]![i]!-residuals[arm]![i]!));
    console.log({cycle:full?"Full":"V",arm,residualError});assert.ok(residualError<=1e-5,`projected residual difference ${residualError}`);
    }
   }
   restore();internal.fusedReserve=true;device.queue.writeBuffer(root.ownership.support,gate,new Uint32Array([1]));
   const e=device.createCommandEncoder();cycles.encodeVCycle(e,true);cycles.encodeFullCycle(e,true);device.queue.submit([e.finish()]);
   assert.deepEqual(await read(device,ranges[0]!),saved[0],"closed slots preserve all mixed fields");
  }finally{solver.destroy();}
 });
});
