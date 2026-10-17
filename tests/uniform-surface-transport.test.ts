import assert from "node:assert/strict";
import test from "node:test";
import {uniformSurfaceTransportWGSL} from "../lib/methods/uniform/uniform-surface-transport.wgsl";
import {narrowBandTraceWGSL} from "../lib/methods/uniform/uniform-narrow-band-advection.wgsl";
import {withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";

test("whole surface model transports nonzero flow and rejects its local error budget",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("whole surface transport budgets",async device=>{
  const cases=[
   {rho:0,lo:.1,hi:.1,eps:1/32,accept:1},
   {rho:1e-5,lo:.099,hi:.101,eps:1/32,accept:1},
   {rho:.001,lo:.1,hi:.1,eps:1/32,accept:0},
   {rho:0,lo:-1,hi:1,eps:1/32,accept:0},
   {rho:0,lo:2,hi:2,eps:1/32,accept:0},
   {rho:NaN,lo:.1,hi:.1,eps:1/32,accept:0},
   // Curved far fields exceed the old absolute residual bound, but small
   // nonzero certified midpoint transport fits the displacement bound.
   {rho:.001,lo:.0003,hi:.0003,eps:0,accept:1},
   {rho:.001,lo:.0002,hi:.0004,eps:0,accept:1},
  ];
  const input=device.createBuffer({size:16*cases.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:16*cases.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   device.queue.writeBuffer(input,0,new Float32Array(cases.flatMap(c=>[c.rho,c.lo,c.hi,c.eps])));
   const module=device.createShaderModule({code:`${uniformSurfaceTransportWGSL}
@group(0) @binding(0) var<storage,read> input:array<vec4f>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4f>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){let a=input[id.x];
 var c=select(0.02,0.2,id.x>=6u);var result=vec4f(0.0);
 for(var frame=0u;frame<100u;frame++){
  result=umSurfaceTransport(c,vec3f(0.0,0.05,0.0),a.x,vec3f(0.0,a.y,0.0),vec3f(0.0,a.z,0.0),vec3f(0.05),1.0/60.0,a.w,vec3f(32.0),1.0);
  if(result.w==0.0){break;}c=result.x;
 }
 output[id.x]=result;}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(cases.length);p.end();device.queue.submit([e.finish()]);
   const result=await readMixedBuffer(device,output);
   cases.forEach((c,i)=>{assert.equal(result[4*i+3],c.accept,`case ${i}`);if(c.accept){const initial=i>=6?.2:.02,expected=initial-100*.5*(c.lo+c.hi)/60;
    const tolerance=128*2**-23*Math.max(Math.abs(initial),Math.abs(expected));
    assert.ok(Math.abs(result[4*i]!-expected)<=tolerance,`100 f32 transport steps case${i}: ${result[4*i]} vs ${expected}`);
    assert.ok(Math.abs(result[4*i]!-initial)>.9*Math.abs(expected-initial),"certified midpoint flow must keep moving over repeated frames");assert.ok(result[4*i+1]!<=.001,"accepted phi error stays within .02h");}});
  }finally{input.destroy();output.destroy();}
 });
});


test("endpoint intervals bound production RK across exponents and nonzero tiny flow",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 let seed=7193;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
 const coordinates=[0,1,31,32,1023,1024,1048575,1048576,8388607,8388608,16777215,16777216];
 const cases=Array.from({length:384},(_,i)=>{
  const g=Array.from({length:3},()=>Math.fround((random()-.5)*.1)),rho=[.02,.2,30][i%3]!;
  const scale=10**(-12+(i%11));const u=Array.from({length:3},()=>Math.fround((random()-.5)*scale));
  const lo=u.map(v=>Math.fround(v-.1*scale-1e-15)),hi=u.map(v=>Math.fround(v+.1*scale+1e-15));
  const stages=Array.from({length:9},(_,k)=>Math.fround(lo[k%3]!+random()*(hi[k%3]!-lo[k%3]!)));
  const p=coordinates[i%coordinates.length]!,foot=Array.from({length:3},()=>Math.fround(p+(random()-.5)*.8));
  const values=Array.from({length:125},(_,k)=>Math.fround(.02+g[0]!*(k%5-2)+g[1]!*(Math.floor(k/5)%5-2)+g[2]!*(Math.floor(k/25)-2)+(2*random()-1)*rho));
  const far=values.map(v=>Math.fround(v-100*rho-2));
  const sample=(q:number[],cubic:boolean,v:number[])=>{
   const base=q.map(Math.floor),f=q.map((x,k)=>x-base[k]!);let result=0,low=Infinity,high=-Infinity;
   const w=f.map(t=>cubic?[.5*(2*t*t-t*t*t-t),.5*(3*t*t*t-5*t*t+2),.5*(4*t*t-3*t*t*t+t),.5*(t*t*t-t*t)]:[1-t,t]);
   const n=cubic?4:2,offset=cubic?-1:0;
   for(let z=0;z<n;z++)for(let y=0;y<n;y++)for(let x=0;x<n;x++){
    const value=v[base[0]!+x+offset+2+5*(base[1]!+y+offset+2+5*(base[2]!+z+offset+2))]!;
    result+=value*w[0]![x]!*w[1]![y]!*w[2]![z]!;
    if(!cubic||(x>=1&&x<=2&&y>=1&&y<=2&&z>=1&&z<=2)){low=Math.min(low,value);high=Math.max(high,value);}
   }return cubic?Math.max(low,Math.min(high,result)):result;
  };
  return {g,rho,u,lo,hi,stages,p,foot,values,far,sample};
 });
 await withUniformDevice("rounded trace interval",async device=>{
  const input=device.createBuffer({size:128*cases.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:96*cases.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   const data=new Float32Array(32*cases.length);cases.forEach((c,i)=>data.set([...c.g,c.rho,...c.lo,...c.hi,...c.u,c.values[62]!,...c.stages,c.p,c.p,c.p,...c.foot,c.far[62]!],32*i));device.queue.writeBuffer(input,0,data);
   const module=device.createShaderModule({code:`${uniformSurfaceTransportWGSL}
struct FixtureParams{hDt:vec4f};var<private> params:FixtureParams;
fn fixtureVelocity(p:vec3f)->vec3f{return vec3f(0.0);}
${narrowBandTraceWGSL("fixtureVelocity")}
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read_write> result:array<vec4f>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){let b=32u*id.x;let r=6u*id.x;
 let g=vec3f(a[b],a[b+1u],a[b+2u]);let lo=vec3f(a[b+4u],a[b+5u],a[b+6u]);let hi=vec3f(a[b+7u],a[b+8u],a[b+9u]);let u=vec3f(a[b+10u],a[b+11u],a[b+12u]);
 let p=vec3f(a[b+23u],a[b+24u],a[b+25u]);let foot=vec3f(a[b+26u],a[b+27u],a[b+28u]);let shift=p-foot;let h=vec3f(.05);let dt=1.0/60.0;
 let interval=umTraceEndpointInterval(foot,shift,lo,hi,u,h,dt);
 var full=nbTraceEnd(foot,-dt,vec3f(a[b+14u],a[b+15u],a[b+16u])/h,vec3f(a[b+17u],a[b+18u],a[b+19u])/h,vec3f(a[b+20u],a[b+21u],a[b+22u])/h,false).xyz;
 if(any(shift!=vec3f(0.0))){full+=shift;}
 result[r]=umSurfaceEndpointBound(a[b+13u],g,a[b+3u],h,dt,p,1.0,0.0,interval,false);
 result[r+1u]=umSurfaceEndpointBound(a[b+29u],g,a[b+3u],h,dt,p,1.0,0.0,interval,true);
 result[r+2u]=vec4f(interval.euler,0.0);result[r+3u]=vec4f(full,0.0);result[r+4u]=vec4f(interval.lo,0.0);result[r+5u]=vec4f(interval.hi,0.0);
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}}),group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
   const e=device.createCommandEncoder(),pass=e.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(cases.length);pass.end();device.queue.submit([e.finish()]);
   const result=await readMixedBuffer(device,output);let accepted=0;
   cases.forEach((c,i)=>{const r=24*i,euler=Array.from(result.slice(r+8,r+11)),full=Array.from(result.slice(r+12,r+15));
    for(let k=0;k<3;k++)assert.ok(full[k]!>=result[r+16+k]!&&full[k]!<=result[r+20+k]!,`case ${i} axis ${k}: rounded RK endpoint outside interval`);
    const eq=euler.map(x=>x-c.p),fq=full.map(x=>x-c.p);
    if(result[r+3]){accepted++;const error=Math.abs(c.sample(fq,true,c.values)-c.sample(eq,false,c.values));assert.ok(error<=result[r]!,`case ${i}: cubic error ${error} exceeds ${result[r]}`);}
    if(result[r+7]){const error=Math.abs(c.sample(fq,false,c.far)-c.sample(eq,false,c.far));assert.ok(error<=result[r+4]!,`case ${i}: far linear error ${error} exceeds ${result[r+4]}`);}
   });assert.ok(accepted>100,`only ${accepted} current nonzero-flow intervals admitted`);
  }finally{input.destroy();output.destroy();}
 });
});

// These checks validate the admission criterion's monotonicity, rather
// than asserting that its conservative error estimate is a true-error floor.
test("home-only necessary rejection cannot discard an admitted enclosing criterion",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 let seed=819341;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
 const count=4096,data=new Float32Array(32*count);
 for(let i=0;i<count;i++){
  const p=[1,32,1024,65536,4194304][i%5]!,width=i%2?1:4;
  const lo=Array.from({length:3},()=>Math.fround((random()-.5)*10**(-10+i%11)));
  const hi=lo.map(v=>Math.fround(v+random()*10**(-10+i%11)));
  const outerLo=lo.map(v=>Math.fround(v-random()*10**(-10+i%11)));
  const outerHi=hi.map(v=>Math.fround(v+random()*10**(-10+i%11)));
  const u=outerLo.map((v,k)=>Math.fround(v+random()*(outerHi[k]!-v)));
  const rho=i%7===0?0:10**(-9+i%10)*random(),outerRho=rho+random()*10**(-9+i%10);
  const foot=Array.from({length:3},()=>Math.fround(p+(random()-.5)*4*width));
  data.set([...lo,...hi,...outerLo,...outerHi,...u,...foot,p,width,rho,outerRho,
   ...Array.from({length:3},()=>Math.fround((random()-.5)*10**(-3+i%5))),
   [0,.02,-.02,1,-50][i%5]!,[.0125,.05,.1][i%3]!,1/60,i%3===0?1:0],32*i);
 }
 await withUniformDevice("home criterion necessary rejection",async device=>{
  const input=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:32*count,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   device.queue.writeBuffer(input,0,data);
   const module=device.createShaderModule({code:`${uniformSurfaceTransportWGSL}
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read_write> out:array<vec4f>;
fn padded(lo:vec3f,hi:vec3f)->mat2x3f{let pad=0.00000762939453125*max(abs(lo),abs(hi));return mat2x3f(lo-pad,hi+pad);}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){let b=32u*id.x;
 let home=padded(vec3f(a[b],a[b+1u],a[b+2u]),vec3f(a[b+3u],a[b+4u],a[b+5u]));
 let outer=padded(vec3f(a[b+6u],a[b+7u],a[b+8u]),vec3f(a[b+9u],a[b+10u],a[b+11u]));
 let u=vec3f(a[b+12u],a[b+13u],a[b+14u]);let foot=vec3f(a[b+15u],a[b+16u],a[b+17u]);let p=vec3f(a[b+18u]);let shift=p-foot;
 let h=vec3f(a[b+26u]);let dt=a[b+27u];let g=vec3f(a[b+22u],a[b+23u],a[b+24u]);
 let innerI=umTraceEndpointInterval(foot,shift,home[0],home[1],u,h,dt);
 let outerI=umTraceEndpointInterval(foot,shift,outer[0],outer[1],u,h,dt);
 let innerB=umSurfaceEndpointBound(a[b+25u],g,a[b+20u],h,dt,p,a[b+19u],0.0,innerI,a[b+28u]>0.0);
 let outerB=umSurfaceEndpointBound(a[b+25u],g,a[b+21u],h,dt,p,a[b+19u],0.0,outerI,a[b+28u]>0.0);
 out[2u*id.x]=vec4f(innerB.w,outerB.w,f32(all(innerI.lo==p)&&all(innerI.hi==p)),f32(all(outerI.lo==p)&&all(outerI.hi==p)));
 out[2u*id.x+1u]=vec4f(innerB.x,outerB.x,f32(all(dt*max(abs(home[0]),abs(home[1]))/h<=vec3f(.5))),f32(all(dt*max(abs(outer[0]),abs(outer[1]))/h<=vec3f(.5))));
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(count);p.end();device.queue.submit([e.finish()]);
   const values=await readMixedBuffer(device,output);let admitted=0,collapsed=0;
   for(let i=0;i<count;i++){
    if(values[8*i+1]===1){admitted++;assert.equal(values[8*i],1,`home error rejection would discard admitted case ${i}`);}
    if(values[8*i+3]===1){collapsed++;assert.equal(values[8*i+2],1,`home endpoint rejection would discard collapsed case ${i}`);}
    if(values[8*i+7]===1)assert.equal(values[8*i+6],1,`home speed rejection would discard case ${i}`);
   }
   assert.ok(admitted>100&&collapsed>100,"stress fixture must exercise admitted and rounded endpoint cases");
  }finally{input.destroy();output.destroy();}
 });
});
