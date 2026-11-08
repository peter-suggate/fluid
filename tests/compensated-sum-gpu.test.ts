import assert from 'node:assert/strict';
import test from 'node:test';
import {uniformCompensatedSumWGSL} from '../lib/methods/uniform/uniform-compensated-sum.wgsl';
import {readMixedBuffer} from './helpers/uniform-mixed-native-fields';
import {withUniformDevice} from './helpers/uniform-geometric';

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
