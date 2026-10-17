import {readFileSync} from "node:fs";
import assert from "node:assert/strict";
import test from "node:test";
import {NARROW_BAND_TRACE_LIMIT,narrowBandTraceWGSL} from "../lib/methods/uniform/uniform-narrow-band-advection.wgsl";
import {withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";

test("production trace seed preserves endpoints, sampler state and single-use invalidation",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const source=readFileSync(new URL("../lib/methods/uniform/uniform-mixed-surface.ts",import.meta.url),"utf8");
 const trace=source.slice(source.indexOf("override umSurfaceStationaryTrace"),source.indexOf("${uniformMixedSolidWGSL"))
  .replaceAll("${3*NARROW_BAND_TRACE_LIMIT+3}",String(3*NARROW_BAND_TRACE_LIMIT+3)).replaceAll("${NARROW_BAND_TRACE_LIMIT}",String(NARROW_BAND_TRACE_LIMIT));
 await withUniformDevice("surface trace seed",async device=>{
  const output=device.createBuffer({size:32*64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),state=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  try{
   const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> result:array<vec4f>;
@group(0) @binding(1) var<storage,read_write> nbState:array<atomic<u32>>;
struct FixtureParams{hDt:vec4f};var<private> params:FixtureParams;
struct umBounds{lo:vec3f,hi:vec3f};var<private> umHold:umBounds;
var<private> umSlow=false;var<private> calls=0u;var<private> strength=0.0;var<private> stationary=false;
override umFastSurface:bool=true;
fn umPointBounds(p:vec3f,reach:f32)->umBounds{return umBounds(p-vec3f(reach),p+vec3f(reach));}
fn umPointFrom(home:vec3f,p:vec3f)->vec3f{return clamp(p,vec3f(0),vec3f(10000));}
fn fixtureVelocity(p:vec3f)->vec3f{calls++;umHold=umPointBounds(p,8.0);if(stationary){return vec3f(0);}
 return vec3f(strength+0.01*p.y,-0.5*strength+0.02*p.z,0.3*strength-0.03*p.x);}
fn umFastVelocity(p:vec3f)->vec3f{let u=fixtureVelocity(p);if(p.x<0.0){umSlow=true;return vec3f(0);}return u;}
fn umSampleVelocity(p:vec3f)->vec3f{return fixtureVelocity(p);}
${narrowBandTraceWGSL("umSampleVelocity")}
${trace}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){let i=id.x;
 params.hDt=vec4f(vec3f(.05),1.0/60.0);stationary=i%8u==0u;strength=f32(i%9u)*.3;
 let p=vec3f(select(f32(i)+2.0,.01,i%16u==15u),3.0,4.0);
 umSlow=false;calls=0u;umTraceSeedValid=false;let reference=umSurfaceTrace(p);let referenceCalls=calls;let referenceSlow=umSlow;let referenceHold=umHold;
 umSlow=false;let u=umFastVelocity(p);let initialSlow=umSlow;
 umTraceSeedPoint=p+select(vec3f(0),vec3f(1,0,0),i%4u==1u);umTraceSeedVelocity=u;umTraceSeedValid=!initialSlow&&i%4u!=2u;
 let expected=umSurfaceTraceSeed&&umFastSurface&&umTraceSeedValid&&i%4u!=1u;
 umHold=umBounds(vec3f(-999),vec3f(-998));umSlow=false;calls=0u;let actual=umSurfaceTrace(p);
 let endpoint=all(bitcast<vec3u>(reference)==bitcast<vec3u>(actual));
 let hold=all(bitcast<vec3u>(referenceHold.lo)==bitcast<vec3u>(umHold.lo))&&all(bitcast<vec3u>(referenceHold.hi)==bitcast<vec3u>(umHold.hi));
 result[2u*i]=vec4f(f32(endpoint),f32(hold),f32(referenceCalls),f32(calls));
 result[2u*i+1u]=vec4f(f32(umSlow==referenceSlow),f32(!umTraceSeedValid),f32(expected),0.0);
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
   for(const fast of [0,1])for(const seed of [0,1]){
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main",constants:{umFastSurface:fast,umSurfaceTraceSeed:seed}}});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[output,state].map((buffer,binding)=>({binding,resource:{buffer}}))});
    const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(64);p.end();device.queue.submit([e.finish()]);
    const result=await readMixedBuffer(device,output);let reused=0;
    for(let i=0;i<64;i++){
     for(const j of [0,1,4,5])assert.equal(result[8*i+j],1,`fast${fast}/seed${seed}/case${i}/invariant${j}`);
     const expected=result[8*i+6]!;reused+=expected;
     assert.equal(result[8*i+3],result[8*i+2]!-expected,`exactly one sample saved fast${fast}/seed${seed}/case${i}`);
    }
    assert.equal(reused>0,fast===1&&seed===1);
   }
  }finally{output.destroy();state.destroy();}
 });
});
