import assert from "node:assert/strict";
import test from "node:test";
import { uniformSurfacePlaneWGSL, UNIFORM_SURFACE_PLANE_FILL_BUDGET } from "../lib/methods/uniform/uniform-surface-plane.wgsl";
import { uniformVolumeTargetWGSL } from "../lib/methods/uniform/uniform-volume.wgsl";
import { geometricPlaneBoxWGSL } from "../lib/core/geometric-plane-box.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

test("current-corner plane model bounds production geometric fill during motion",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const cases:{kind:string;v:number[]}[]=[];
 let seed=78361;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
 for(let i=0;i<7200;i++){
  const kind=["plane","small curve","curve","saddle","weak","zero corners"][i%6]!;
  const g=[random()*2-1,random()*2-1,random()*2-1];const offset=(random()-.5)*2;
  const amplitude=kind==="small curve"?1e-6:kind==="curve"?.03:kind==="saddle"?.4:0;
  const v=Array.from({length:8},(_,k)=>{
   const x=(k&1)-.5,y=((k>>1)&1)-.5,z=((k>>2)&1)-.5;
   let value=offset+g[0]!*x+g[1]!*y+g[2]!*z+amplitude*(x*y+y*z+z*x);
   if(kind==="weak")value*=1e-14;
   if(kind==="zero corners"&&(k===0||k===7))value=0;
   return Math.fround(value);
  });cases.push({kind,v});
 }
 // Scale invariance, near-corner crossings, anisotropic normals, and f32
 // cancellation exercise the projection/CDF allowances rather than a CPU fit.
 for(const scale of [1e-8,1e-3,1,1e6,1e20,1e35]){
  for(const g of [[1,0,0],[1,1e-7,0],[1,1,1],[1,-1,.00001]]){
   for(const offset of [-1.5000001,-.50000006,-.5,-1e-8,0,1e-8,.5,.50000006,1.5000001]){
    cases.push({kind:"f32 edge",v:Array.from({length:8},(_,k)=>Math.fround(scale*(offset+g[0]!*((k&1)-.5)+g[1]!*(((k>>1)&1)-.5)+g[2]!*(((k>>2)&1)-.5))))});
   }
  }
 }
 for(let i=0;i<2400;i++){
  const amplitude=.0001+random()*.002;const offset=(random()-.5)*1.5;
  const g=[.4+random()*.6,random()-.5,random()-.5];
  cases.push({kind:"gentle curve",v:Array.from({length:8},(_,k)=>{
   const x=(k&1)-.5,y=((k>>1)&1)-.5,z=((k>>2)&1)-.5;
   return Math.fround(offset+g[0]!*x+g[1]!*y+g[2]!*z+amplitude*(x*y+y*z+z*x));
  })});
 }
 cases.push({kind:"all zero",v:Array(8).fill(0)});
 for(const bad of [NaN,Infinity,-Infinity])cases.push({kind:"nonfinite",v:[bad,1,-1,1,-1,1,-1,1]});
 await withUniformDevice("bounded current-corner geometry",async device=>{
  const data=new Float32Array(cases.flatMap(c=>c.v));
  const input=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:16*cases.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  device.queue.writeBuffer(input,0,data);
  const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> fixture:array<f32>;
@group(0) @binding(1) var<storage,read_write> result:array<vec4f>;
struct UMOwner{width:u32};
fn uvCorner(k:u32)->vec3i{return vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));}
fn umOrigin(owner:UMOwner)->vec3u{return vec3u(0);}
fn umCellOpen(p:vec3i)->f32{return 1.0;}
fn d4Sum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
${geometricPlaneBoxWGSL}
${uniformVolumeTargetWGSL(true,true)}
${uniformSurfacePlaneWGSL}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=${cases.length}u){return;}
 var v:array<f32,8>;for(var k=0u;k<8u;k++){v[k]=fixture[8u*id.x+k];}
 let plane=gcCurrentPlane(v);let exact=umSurfaceTarget(UMOwner(1u),v);
 result[id.x]=vec4f(exact,plane.x,plane.y,f32(plane.x>=0.0));
}`});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
  try{
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(Math.ceil(cases.length/64));p.end();device.queue.submit([e.finish()]);
   const values=await readMixedBuffer(device,output);const accepted=new Map<string,number>();
   cases.forEach((c,i)=>{
    const [exact,model,bound,used]=Array.from(values.slice(4*i,4*i+4));
    if(c.kind==="nonfinite"||c.kind==="weak"||c.kind==="all zero")assert.equal(used,0,c.kind);
    if(!used)return;
    accepted.set(c.kind,(accepted.get(c.kind)??0)+1);
    const error=Math.abs(exact!-model!);
    assert.ok(bound!<=UNIFORM_SURFACE_PLANE_FILL_BUDGET,`${c.kind} bound ${bound}`);
    assert.ok(error<=bound!,`${c.kind}[${i}] error ${error}, bound ${bound}`);
    // h solid cells use the same model times open: absolute error contracts.
    for(const open of [0,.03,.5,1])assert.ok(error*open<=bound!*open+1e-12);
   });
   assert.ok((accepted.get("plane")??0)>1100,`plane acceptance ${accepted.get("plane")}`);
   assert.ok((accepted.get("small curve")??0)>1100,`moving curved acceptance ${accepted.get("small curve")}`);
   assert.ok((accepted.get("gentle curve")??0)>2000,`gentle curved acceptance ${accepted.get("gentle curve")}`);
   assert.equal(accepted.get("saddle")??0,0);
   assert.equal(accepted.get("curve")??0,0);
  }finally{input.destroy();output.destroy();}
 });
});
