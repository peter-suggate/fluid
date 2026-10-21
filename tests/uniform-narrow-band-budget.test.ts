import assert from 'node:assert/strict';
import test from 'node:test';
import { narrowBandBudgetWGSL, narrowBandBudgetWords } from '../lib/methods/uniform/uniform-narrow-band-budget.wgsl';
import { narrowBandControlValues } from '../lib/methods/uniform/uniform-narrow-band-controls';
import { uniformNarrowBandMethod } from '../lib/methods/uniform/uniform-narrow-band-method';
import { UNIFORM_STAGE_IMPORTANCE as I } from '../lib/methods/uniform/uniform-stage-grids';
import { withUniformDevice } from './helpers/uniform-geometric';

test('NB-FLIP defaults to unrestricted adaptive detail',()=>{
 assert.equal(narrowBandControlValues().adaptiveBudgetPercent,100);
 assert.equal(uniformNarrowBandMethod.appDefaults?.adaptiveBudgetPercent,100);
 assert.equal(narrowBandControlValues({adaptiveBudgetPercent:50}).adaptiveBudgetPercent,50);
});

test('GPU budget admits complete score groups independently of tile order',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('NB-FLIP budget ties',async device=>{
  const tiles=128,base=3*tiles,words=base+narrowBandBudgetWords(tiles);
  const buffer=device.createBuffer({size:4*words,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  const settings=device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const read=device.createBuffer({size:buffer.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const helpers=narrowBandBudgetWGSL.slice(0,narrowBandBudgetWGSL.indexOf('fn nbBudgetLiquid'));
  const entries=narrowBandBudgetWGSL.slice(narrowBandBudgetWGSL.indexOf('@compute @workgroup_size(1) fn activityCutoff'));
  const module=device.createShaderModule({code:`
const UM_TILES:u32=${tiles}u;const umDispatchX:u32=2u;
struct Policy{activity:vec4u,step:vec4f}
@group(0) @binding(0) var<storage,read_write> census:array<atomic<u32>>;
@group(0) @binding(1) var<uniform> policy:Policy;
fn importanceIndex(t:u32,k:u32)->u32{return 2u*t+k;}
fn holdIndex(t:u32)->u32{return 2u*UM_TILES+t;}
fn umTileLive(t:u32)->bool{return t<UM_TILES;}
fn publishImportance(t:u32,w:u32){}
${helpers}
${entries}`});
  const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}}]});
  const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
  const cutoff=await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint:'activityCutoff'}});
  const admit=await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint:'activityAdmit'}});
  const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer}},{binding:1,resource:{buffer:settings}}]});
  try{
   for(const scores of [Array(128).fill(255),Array.from({length:128},(_,i)=>i<32?255:i<96?180:64)]){
    for(const percent of [0,1,25,50,75,100])for(const reversed of [false,true]){
     const order=reversed?[...scores].reverse():scores,initial=new Uint32Array(words);
     for(let t=0;t<tiles;t++){initial[2*t+1]=I.required;initial[base+order[t]!]!++;initial[base+257+t]=order[t]!<<8;}
     device.queue.writeBuffer(buffer,0,initial);device.queue.writeBuffer(settings,0,new Uint32Array([percent,0,0,0,0,0,0,0]));
     const e=device.createCommandEncoder(),p=e.beginComputePass();p.setBindGroup(0,group);p.setPipeline(cutoff);p.dispatchWorkgroups(1);p.setPipeline(admit);p.dispatchWorkgroups(2);p.end();e.copyBufferToBuffer(buffer,0,read,0,buffer.size);device.queue.submit([e.finish()]);
     await read.mapAsync(GPUMapMode.READ);const result=new Uint32Array(read.getMappedRange()).slice();read.unmap();
     const target=Math.floor(tiles*percent/100),sorted=[...scores].sort((a,b)=>b-a),threshold=target?sorted[target-1]!:256;
     for(let t=0;t<tiles;t++)assert.equal(!!(result[2*t+1]!&I.required),order[t]!>=threshold,`${percent}%, tile ${t}, reversed=${reversed}`);
     assert.ok(result.subarray(base,base+256).every(v=>v===0),'histogram is cleared for the next census');
    }
   }
  }finally{read.destroy();settings.destroy();buffer.destroy();}
 });
});
