import assert from 'node:assert/strict';
import test from 'node:test';
import {createRigidCouplingOracleScene} from '../lib/core/scenes';
import {uniformCompensatedSumWGSL} from '../lib/methods/uniform/uniform-compensated-sum.wgsl';
import {UniformSourceWater} from '../lib/methods/uniform/uniform-source-water';
import {NB_VOLUME_COMPENSATION_FLOAT} from '../lib/methods/uniform/uniform-mixed-surface-volume';
import type {UniformMixedFrame} from '../lib/methods/uniform/uniform-mixed-frame';
import type {WetClothSourceField} from '../lib/cloth/source-types';
import {readMixedBuffer} from './helpers/uniform-mixed-native-fields';
import {createUniformSolver,withUniformDevice} from './helpers/uniform-geometric';

// Budgets and ledgers accumulate many small f32 deltas onto a large total in a
// (high, low) pair. The two-sum form `low += (old - high) + delta` folds to 0
// under Metal's fast math, so low never moves and every rounding is lost.
// umCompensatedAdd splits through round() instead. Both forms run here as
// separate dispatches over storage, as the budget sites do, against a Float64
// sum of the same f32 deltas.
const gpu={skip:!process.env.WEBGPU_NODE_MODULE};

// The spacing of f32 values at |x|.
const ulpAt=(x:number)=>2**(Math.max(Math.floor(Math.log2(Math.abs(x)||2**-126)),-126)-23);
function random(seed:number):()=>number{let s=seed>>>0;return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s/2**32;};}

test('umCompensatedAdd keeps the pair at the Float64 sum of many small deltas onto a large budget',gpu,async t=>{
 await withUniformDevice('compensated sum',async device=>{
  const module=device.createShaderModule({label:'Compensated sum regression',code:uniformCompensatedSumWGSL+/* wgsl */`
@group(0) @binding(0) var<storage,read_write> pair:array<f32>;
@group(0) @binding(1) var<storage,read> deltas:array<f32>;
@group(0) @binding(2) var<storage,read_write> index:array<u32>;
@compute @workgroup_size(1) fn twoSum(){let delta=deltas[index[0]];index[0]+=1u;let old=pair[0];pair[0]=old+delta;pair[1]+=(old-pair[0])+delta;}
@compute @workgroup_size(1) fn helper(){let delta=deltas[index[0]];index[0]+=1u;let p=umCompensatedAdd(vec2f(pair[0],pair[1]),delta);pair[0]=p.x;pair[1]=p.y;}`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');assert.equal(errors.map(m=>m.message).join('\n'),'');
  const pipelines=Object.fromEntries(await Promise.all(['twoSum','helper'].map(async entryPoint=>[entryPoint,await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint}})] as const)));
  const run=async(entryPoint:'twoSum'|'helper',start:number,deltas:Float32Array<ArrayBuffer>):Promise<[number,number]>=>{
   const pair=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
   const input=device.createBuffer({size:deltas.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
   const index=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE});
   try{
    device.queue.writeBuffer(pair,0,new Float32Array([start,0,0,0]));device.queue.writeBuffer(input,0,deltas);
    const pipeline=pipelines[entryPoint]!,group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[pair,input,index].map((buffer,binding)=>({binding,resource:{buffer}}))});
    const e=device.createCommandEncoder(),pass=e.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);
    for(let i=0;i<deltas.length;i++)pass.dispatchWorkgroups(1);
    pass.end();device.queue.submit([e.finish()]);
    const words=await readMixedBuffer(device,pair);return [words[0]!,words[1]!];
   }finally{pair.destroy();input.destroy();index.destroy();}
  };
  const n=2000,cases:{name:string;start:number;delta:(r:()=>number)=>number}[]=[
   {name:'deltas under half an ulp onto 3e5',start:3e5,delta:r=>.01*r()},
   {name:'up across 2^24',start:2**24-64,delta:r=>r()},
   {name:'mixed signs down across 2^20',start:2**20+10,delta:r=>2*r()-1.5},
   {name:'small budget',start:1e-3,delta:r=>1e-9*(r()-.25)},
  ];
  for(const [k,c] of cases.entries()){
   const r=random(17+k),deltas=Float32Array.from({length:n},()=>c.delta(r));
   let exact=Math.fround(c.start);for(const d of deltas)exact+=d;
   const [high,low]=await run('helper',c.start,deltas),[plainHigh,plainLow]=await run('twoSum',c.start,deltas);
   const error=Math.abs(high+low-exact),plainError=Math.abs(plainHigh+plainLow-exact);
   // Low stays within 1.5 ulps of the sum, and its adds round at that scale.
   const ulp=ulpAt(exact),bound=n*2**-22*ulp;
   t.diagnostic(`${c.name}: helper error ${error.toExponential(2)} (low ${low}, bound ${bound.toExponential(2)}); two-sum error ${plainError.toExponential(2)} (low ${plainLow})`);
   assert.ok(error<=bound,`${c.name}: pair ${high}+${low} versus Float64 ${exact}`);
   assert.ok(Math.abs(low)<=2*ulp,`${c.name}: high ${high} carries the sum, low ${low}`);
  }
 });
});

test('source water commits its budget and ledger at the Float64 sum of its GPU deltas',gpu,async t=>{
 await withUniformDevice('source water compensated budget',async device=>{
  const scene=createRigidCouplingOracleScene('rigid-hydrostatic');scene.rigidBodies=[];scene.fluid.gravity_m_s2={x:0,y:0,z:0};scene.numerics.fixedDt_s=scene.numerics.maxDt_s=.001;
  scene.fluid.refinementRegions=[{id:'source-support',min_m:{x:-.35,y:.1,z:-.35},max_m:{x:.35,y:.8,z:.35},rule:'minimum-cell-size',minimumCellSize_cells:1,maximumCellSize_cells:1}];
  const solver=await createUniformSolver(device,scene),frame=(solver as unknown as {mixedFrame:UniformMixedFrame}).mixedFrame,l=frame.ownership.capacity.lattice;
  const fields=frame.clothCouplingFields([l.origin_m.x,l.origin_m.y,l.origin_m.z],l.cellSize_m),plugin=new UniformSourceWater(fields);
  const cellVolume=Math.fround(l.cellSize_m.reduce((a,b)=>a*b,1));
  try{
   await plugin.initialize();assert.ok(solver.advanceTo(.001,[]));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);
   const budget0=await readMixedBuffer(device,fields.waterBudget),ledger0=await readMixedBuffer(device,plugin.externalLedger);
   let budget=budget0[0]!+budget0[NB_VOLUME_COMPENSATION_FLOAT]!,ledger=ledger0[0]!+ledger0[1]!,moved=0;
   // A sphere emitted then terminated at a run of centres: each commit adds a
   // fractional cell count onto the tank's whole budget.
   for(let i=0;i<12;i++){
    const source:WetClothSourceField={index:0,kind:'sphere',usage:'source',group:0,inside:false,sampled:true,root:true,center_m:[-.2+.035*i,.55,.01*i],radius_m:.05+.002*i,sourceAttributes:{}};
    for(const kind of ['emit','terminate'] as const){
     plugin.prepare([source],[{fieldIndex:0,kind,velocity_m_s:[0,0,0]}]);
     const batch=(plugin as unknown as {batches:{buffers:GPUBuffer[];vertices:number}[]}).batches[0]!;
     const e=device.createCommandEncoder();plugin.encodeSupport(e);plugin.encode(e);device.queue.submit([e.finish()]);await device.queue.onSubmittedWorkDone();
     const delta=(await readMixedBuffer(device,batch.buffers[1]!))[4*batch.vertices]!;
     budget+=delta;ledger+=Math.fround(delta*cellVolume);if(delta!==0)moved++;
    }
   }
   const after=await readMixedBuffer(device,fields.waterBudget),afterLedger=await readMixedBuffer(device,plugin.externalLedger);
   const pair=after[0]!+after[NB_VOLUME_COMPENSATION_FLOAT]!,ledgerPair=afterLedger[0]!+afterLedger[1]!;
   t.diagnostic(`${moved} non-zero commits onto budget ${budget0[0]} cells: pair error ${(pair-budget).toExponential(2)} cells (low ${after[NB_VOLUME_COMPENSATION_FLOAT]}), ledger error ${(ledgerPair-ledger).toExponential(2)} m^3 (low ${afterLedger[1]})`);
   assert.ok(moved>=12,'the commits move the budget');
   assert.ok(Math.abs(pair-budget)<=24*2**-22*ulpAt(budget),`budget pair ${pair} versus Float64 ${budget}`);
   assert.ok(Math.abs(ledgerPair-ledger)<=24*2**-22*ulpAt(ledger),`ledger pair ${ledgerPair} versus Float64 ${ledger}`);
  }finally{plugin.destroy();solver.destroy();}
 });
});
