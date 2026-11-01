import assert from "node:assert/strict";
import test from "node:test";
import {readFileSync} from "node:fs";
import {withUniformDevice} from "./helpers/uniform-geometric";
import {pairedComputeTimings} from "./helpers/gpu-paired-timing";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";

const referenceProduction=/* wgsl */`
fn referenceProduction(at:u32,slot:u32,lb:vec3u,kinds:u32,black:bool)->f32{
 var off=pF(P_RHS,at);var walls=false;
 for(var f=0u;f<6u;f++){
  let kind=bFaceKind(kinds,f);
  if(kind==K_BAND){off+=pCoefficient(at,kinds,f)*pNeighbourX(at,slot,lb,f,black);}
  else if(kind==K_SEAM){off+=pF(P_C+f*PLANE,at)*qF(PN_X,nAcross(vec3i(ubaOrigin(ubaBrick(slot))+lb),f));}
  else if(kind==K_WALL){walls=true;}
 }
 let d0=pF(P_DIAG,at);var v=off/d0;
 if(walls||bInside(kinds)){
  let base=pF(P_BASE,at);
  if(walls){
   for(var it=0u;it<4u;it++){
    var d=d0;var o=off;
    for(var f=0u;f<6u;f++){
     if(bFaceKind(kinds,f)!=K_WALL){continue;}
     let w=pF(P_C+f*PLANE,at);let g=pLift(at,f);
     if(v+g>=-base){d-=w;o+=w*g;}else{o-=w*base;}
    }
    if(d>0.0){v=o/d;}
   }
  }
  if(bInside(kinds)){v=max(v,-base);}
 }
 return v;
}
`;

test("wall face masks preserve four-pass pressure bits across changing active sets",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("wall face mask iteration",async device=>{
  const count=65536,input=new Float32Array(count*16);let seed=0x17ca5e;
  const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};
  for(let i=0;i<count;i++){
   const at=16*i;let sum=0;const wallMask=(i<32768?i>>>6:i)&63;
   for(let f=0;f<6;f++){const w=((wallMask>>>f)&1)?random()*20:0;input[at+4+f]=w;sum+=w;input[at+10+f]=(random()-.5)*100;}
   input[at]=sum+random()*10+.001;input[at+1]=(random()-.5)*1000;input[at+2]=(random()-.5)*100;input[at+3]=wallMask|((i%2)<<6);
  }
  // Degenerate zero-diagonal rows retain v; signed zeros must retain bits.
  input.set([0,0,0,0],0);input.set([0,1,0,0],16);input.set([1,-0,0,0],32);
  const rows=device.createBuffer({size:input.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(rows,0,input);
  const output=device.createBuffer({size:count*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   const solve=(name:string,step:string)=>`fn ${name}(r:u32)->vec2u{
    let d0=rows[16u*r];let off=rows[16u*r+1u];let base=rows[16u*r+2u];var v=off/d0;var iterations=0u;let kinds=u32(rows[16u*r+3u]);
    for(var it=0u;it<4u;it++){
     iterations++;var d=d0;var o=off;
     for(var f=0u;f<6u;f++){
      if(((kinds>>f)&1u)==0u){continue;}let w=rows[16u*r+4u+f];let g=rows[16u*r+10u+f];
      if(v+g>=-base){d-=w;o+=w*g;}else{o-=w*base;}
     }
     ${step}
    }
    if((kinds&64u)!=0u){v=max(v,-base);}return vec2u(bitcast<u32>(v),iterations);
   }`;
   const source=readFileSync(new URL("../lib/methods/uniform/uniform-pressure-arena.ts",import.meta.url),"utf8");
   const start=source.indexOf("fn pRelaxRow(");assert.ok(start>=0);let depth=0,end=start;
   for(let i=source.indexOf("{",start);i<source.length;i++){if(source[i]==="{")depth++;if(source[i]==="}"&&--depth===0){end=i+1;break;}}
   const production=source.slice(start,end);
   const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> rows:array<f32>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4u>;
${solve("original","if(d>0.0){v=o/d;}")}
const P_RHS:u32=1u;const P_DIAG:u32=0u;const P_BASE:u32=2u;const P_C:u32=4u;const PLANE:u32=1u;const PN_X:u32=0u;
const K_BAND:u32=1u;const K_SEAM:u32=2u;const K_WALL:u32=3u;
fn bFaceKind(kinds:u32,f:u32)->u32{return select(0u,K_WALL,((kinds>>f)&1u)!=0u);}
fn bInside(kinds:u32)->bool{return (kinds&64u)!=0u;}
fn pF(field:u32,at:u32)->f32{return rows[16u*at+field];}
fn pLift(at:u32,f:u32)->f32{return rows[16u*at+10u+f];}
fn pCoefficient(at:u32,kinds:u32,f:u32)->f32{return 0.0;}
fn pNeighbourX(at:u32,slot:u32,lb:vec3u,f:u32,black:bool)->f32{return 0.0;}
fn qF(field:u32,at:u32)->f32{return 0.0;}
fn nAcross(p:vec3i,f:u32)->u32{return 0u;}
fn ubaBrick(slot:u32)->u32{return 0u;}
fn ubaOrigin(brick:u32)->vec3u{return vec3u(0u);}
${production}
${referenceProduction}
fn referenceTiming(r:u32)->vec2u{return vec2u(bitcast<u32>(referenceProduction(r,0u,vec3u(0u),u32(rows[16u*r+3u]),false)),4u);}
fn reused(r:u32)->vec2u{return vec2u(bitcast<u32>(pRelaxRow(r,0u,vec3u(0u),u32(rows[16u*r+3u]),false)),4u);}

@compute @workgroup_size(64) fn baseline(@builtin(global_invocation_id) g:vec3u){output[g.x]=vec4u(referenceTiming(g.x),vec2u(0));}
@compute @workgroup_size(64) fn candidate(@builtin(global_invocation_id) g:vec3u){output[g.x]=vec4u(reused(g.x),vec2u(0));}
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) g:vec3u){let a=original(g.x);let b=reused(g.x);output[g.x]=vec4u(a,b);}
`});
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"probe"}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:rows}},{binding:1,resource:{buffer:output}}]});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(count/64);p.end();device.queue.submit([e.finish()]);
   const values=new Uint32Array((await readMixedBuffer(device,output)).buffer);let saved=0;
   for(let i=0;i<count;i++){
    assert.equal(values[4*i+2],values[4*i],`pressure row ${i}`);assert.equal(values[4*i+1],4);
    saved+=4-values[4*i+3]!;
   }
   assert.equal(saved,0,"all four active-set passes remain");
   if(process.env.FLUID_EXACT_ALGORITHM_TIMING){
    const variants=await Promise.all(["baseline","candidate"].map(entryPoint=>device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint}})));
    const groups=variants.map(p=>device.createBindGroup({layout:p.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:rows}},{binding:1,resource:{buffer:output}}]}));
    console.log(JSON.stringify({algorithm:"pressure-wall-face-mask-production",savedPasses:saved,originalPasses:4*count,...await pairedComputeTimings(device,variants,groups,count/64,64)}));
   }
  }finally{rows.destroy();output.destroy();}
 },process.env.FLUID_EXACT_ALGORITHM_TIMING?["timestamp-query"]:[]);
});
