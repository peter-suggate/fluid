import assert from "node:assert/strict";
import test from "node:test";
import { narrowBandBudgetWGSL, narrowBandBudgetWords, NARROW_BAND_BUDGET_ENTRIES } from "../lib/methods/uniform/uniform-narrow-band-budget.wgsl";
import { UNIFORM_STAGE_IMPORTANCE as I } from "../lib/methods/uniform/uniform-stage-grids";
import { withUniformDevice } from "./helpers/uniform-geometric";

const gpuTest=process.env.WEBGPU_NODE_MODULE?test:test.skip;
gpuTest("adaptive quota ranks pooled scores and prior admissions with exact, stable saturated-score ties",async()=>{
 await withUniformDevice("NB candidate budget",async device=>{
  const dimensions=[7,7,4],n=196,mask=0b010101; // Shape, strain, impact; other scored criteria are disabled.
  const coordinates=Array.from({length:n},(_,t)=>[t%7,Math.floor(t/7)%7,Math.floor(t/49)]);
  const buffer=(size:number,usage:number)=>device.createBuffer({size,usage});
  const work=buffer(4*(3*n+64+narrowBandBudgetWords(n)),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
  const params=buffer(32,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
  const output=buffer(4*n,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
  const read=buffer(4*n,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
  const module=device.createShaderModule({code:/* wgsl */`
const UM_TILES=196u;const UM_T=vec3u(7,7,4);const umDispatchX=2u;const nbBudgetSupported=true;
struct Policy{activity:vec4u,enabled:vec4u}
@group(0) @binding(0) var<storage,read_write> census:array<atomic<u32>>;
@group(0) @binding(1) var<uniform> policy:Policy;
@group(0) @binding(2) var<storage,read_write> result:array<u32>;
fn importanceIndex(t:u32,k:u32)->u32{return 2u*t+k;}
fn holdIndex(t:u32)->u32{return 2u*UM_TILES+t;}
fn binIndex(b:u32)->u32{return 3u*UM_TILES+b;}
fn umTileCoord(t:u32)->vec3u{return vec3u(t%7u,(t/7u)%7u,t/49u);}
fn umTileAt(p:vec3u)->u32{return p.x+7u*(p.y+7u*p.z);}
fn criterionOn(k:u32)->bool{return (policy.enabled.x&(1u<<k))!=0u;}
fn umScoreByte(w0:u32,w1:u32,k:u32)->u32{if(k<4u){return (w0>>(8u*k))&255u;}return (w1>>(8u*(k-4u)))&255u;}
fn publishImportance(t:u32,w1:u32){result[t]=u32((w1&${I.required}u)!=0u);}
@compute @workgroup_size(64) fn prepare(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x+128u*gid.y;if(t>=UM_TILES){return;}
 if(!nbBudgetOn()){publishImportance(t,atomicLoad(&census[2u*t+1u]));}
}
${narrowBandBudgetWGSL}
`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message);assert.deepEqual(errors,[]);
  const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
  const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:work}},{binding:1,resource:{buffer:params}},{binding:2,resource:{buffer:output}}]});
  const pipelines=new Map(await Promise.all(["prepare",...NARROW_BAND_BUDGET_ENTRIES].map(async entryPoint=>[entryPoint,await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint}})] as const)));
  try{
   let admitted=new Set<number>();
   for(const saturated of [false,true])for(const [census,percent] of [0,1,25,33,50,50,99,100].entries()){
    const words=new Uint32Array(2*n),scoresByTile:number[]=[],wantedTiles:number[]=[];
    for(let t=0;t<n;t++){
     // Change the priorities between censuses to exercise admission history.
     const seed=t+17*census;
     const scores=saturated?[255,255,255,255,255,255]:[(seed*19)%256,255,(seed*73)%256,255,(seed*37)%256,255];
     words[2*t]=(scores[0]!|(scores[1]!<<8)|(scores[2]!<<16)|(scores[3]!<<24))>>>0;
     const wanted=t%7!==0;words[2*t+1]=scores[4]!|(scores[5]!<<8)|(wanted?I.required:0);
     scoresByTile.push(Math.max(scores[0]!,scores[2]!,scores[4]!));
     if(wanted)wantedTiles.push(t);
    }
    // Independent CPU ranking: collect requesting tiles within one tile on
    // every axis, then order by history, pooled score and stable tile index.
    const ranked=wantedTiles.map(tile=>({tile,score:Math.min(255,Math.floor(wantedTiles
     .filter(other=>dimensions.every((_,axis)=>Math.abs(coordinates[tile]![axis]!-coordinates[other]![axis]!)<=1))
     .reduce((sum,other)=>sum+scoresByTile[other]!,0)/16))}));
    if(saturated)assert.ok(ranked.filter(({score})=>score===255).length>64,"saturated ties span workgroups");
    ranked.sort((a,b)=>Number(admitted.has(b.tile))-Number(admitted.has(a.tile))||b.score-a.score||a.tile-b.tile);
    const expected=ranked.slice(0,Math.floor(ranked.length*percent/100)).map(r=>r.tile).sort((a,b)=>a-b);
    device.queue.writeBuffer(work,0,words);device.queue.writeBuffer(params,0,new Uint32Array([percent,0,0,0,mask,0,0,0]));
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setBindGroup(0,group);
    for(const entry of ["prepare",...(percent<100?NARROW_BAND_BUDGET_ENTRIES:[])]){
     pass.setPipeline(pipelines.get(entry)!);
     if(entry==="activityCutoff"||entry==="activityTiePrefix")pass.dispatchWorkgroups(1);else pass.dispatchWorkgroups(2,2);
    }
    pass.end();encoder.copyBufferToBuffer(output,0,read,0,4*n);device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);const selected=[...new Uint32Array(read.getMappedRange())].flatMap((v,t)=>v?[t]:[]);read.unmap();
    assert.deepEqual(selected,expected,`${percent}% of ${ranked.length} candidates, saturated=${saturated}`);
    if(percent<100)admitted=new Set(expected);
   }
  }finally{for(const b of [work,params,output,read])b.destroy();}
 });
});
