import {readFileSync} from "node:fs";
import assert from "node:assert/strict";
import test from "node:test";
import {uniformSurfaceTransportWGSL} from "../lib/methods/uniform/uniform-surface-transport.wgsl";
import {uniformMixedVertexSamplingSource} from "../lib/methods/uniform/uniform-mixed-vertex-sampling.wgsl";
import {withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";
import {uniformSurfaceFarCellWGSL} from "../lib/methods/uniform/uniform-surface-far-cell.wgsl";

test("production far-cell bound covers nonzero trilinear transport and mixed canonical widths",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 let seed=168334;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
 const count=2048,data=new Float32Array(32*count),bad:number[]=[];
 for(let i=0;i<count;i++){
  const width=i%2?1:4,ownerWidth=i%3?1:4,h=.05,origin=[4,1024,1048576][i%3]!;
  const fraction=[1e-6,.01,.2,.5,.8,.99,1-1e-6][i%7]!;
  const euler=Array.from({length:3},(_,a)=>Math.fround(origin+width*(a===0?fraction:.2+.6*random())));
  const travel=width*[1e-6,1e-4,.01,.1][i%4]!;
  const lo=euler.map(q=>Math.fround(Math.max(origin,q-travel))),hi=euler.map(q=>Math.fround(Math.min(origin+width-1e-6,q+travel)));
  const q=lo.map((v,a)=>Math.fround(v+random()*(hi[a]!-v)));
  const magnitude=[.5,2,50][i%3]!,sign=i%2?1:-1;
  const corners=Array.from({length:8},(_,k)=>Math.fround(sign*(magnitude+(.01+random())*((k&1)+2*((k>>1)&1)+((k>>2)&1)))));
  const center=sign*(magnitude+.2),b=32*i;
  data.set([origin,origin,origin,width,ownerWidth,h,center,...lo,...hi,...euler,...q,...corners],b);
  if(i%31===0){data[b+3]=2;bad.push(i);}
  else if(i%31===1){data[b+10]=origin+2*width;bad.push(i);}
  else if(i%31===2){data[b+19]=-.01;data[b+20]=.01;bad.push(i);}
  else if(i%31===3){data[b+6]=0;bad.push(i);}
 }
 const surface=readFileSync(new URL("../lib/methods/uniform/uniform-mixed-surface.ts",import.meta.url),"utf8");
 // Compile the actual production cell loader/interpolator, with only its
 // backing texture/topology access replaced by this explicit current cell.
 const cellSource=surface.slice(surface.indexOf("struct UMVertexCell"),surface.indexOf("// The cubic at q"));
 // Eligible far-band feet return before gradient/smoothing. Compile that
 // exact production prefix; near-band cases are rejected by the new helper.
 const footSource=surface.slice(surface.indexOf("struct UMFoot"),surface.indexOf(" let page=umPointCell(p)",surface.indexOf("struct UMFoot")))+" return result; }";
 const sum=uniformMixedVertexSamplingSource().match(/fn umVertexSum8\([^\n]+/u)![0];
 await withUniformDevice("current far-cell bound",async device=>{
  const input=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),output=device.createBuffer({size:16*count,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   device.queue.writeBuffer(input,0,data);
   const module=device.createShaderModule({code:`${uniformSurfaceTransportWGSL}\n${uniformSurfaceFarCellWGSL}\n${sum}
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read_write> out:array<vec4f>;
var<private> fixtureCorners:array<f32,8>;var<private> fixtureWidth:u32;var<private> fixtureCenter:f32;
struct FixtureParams{hDt:vec4f};var<private> params:FixtureParams;
const umRegularFine=false;
fn umPointCell(q:vec3f)->vec3i{return vec3i(floor(q));}
fn umTileAt(q:vec3u)->u32{return 0u;}
fn umTileWidth(tile:u32)->u32{return fixtureWidth;}
fn umLoadCorner8(p:vec3u,width:u32)->array<f32,8>{return fixtureCorners;}
fn umLoadVertex8(p:vec3u,width:u32)->array<f32,8>{return fixtureCorners;}
fn umLatticeLoad(p:vec3i,width:u32)->f32{return fixtureCenter;}
fn umCorner(k:u32,side:u32)->vec3u{return vec3u(k%side,(k/side)%side,k/(side*side));}
${cellSource}
${footSource}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){let b=32u*id.x;
 let origin=vec3u(vec3f(a[b],a[b+1u],a[b+2u]));let width=u32(a[b+3u]);let lo=vec3f(a[b+7u],a[b+8u],a[b+9u]);let hi=vec3f(a[b+10u],a[b+11u],a[b+12u]);let euler=vec3f(a[b+13u],a[b+14u],a[b+15u]);let q=vec3f(a[b+16u],a[b+17u],a[b+18u]);var corners:array<f32,8>;
 for(var k=0u;k<8u;k++){corners[k]=a[b+19u+k];}
 fixtureCorners=corners;fixtureWidth=width;fixtureCenter=a[b+6u];params.hDt=vec4f(vec3f(a[b+5u]),1.0/60.0);
 let foot=umVertexFoot(origin,u32(a[b+4u]));let cell=umVertexCell(euler);let fullCell=umVertexCell(q);
 let bound=umSurfaceFarCellBound(foot.center,u32(a[b+4u]),vec3f(a[b+5u]),cell.origin,cell.width,cell.corners,UMTraceInterval(lo,hi,euler));
 out[id.x]=vec4f(bound,cell.value,fullCell.value);
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>`${m.lineNum}:${m.linePos} ${m.message}`),[]);
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}}),group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(count);p.end();device.queue.submit([e.finish()]);
   const values=await readMixedBuffer(device,output);let accepted=0,nonzero=0;
   for(let i=0;i<count;i++){
    if(bad.includes(i)){assert.equal(values[4*i+1],0,`invalid case ${i}`);continue;}
    if(values[4*i+1]===0)continue;
    accepted++;const difference=Math.abs(values[4*i+2]!-values[4*i+3]!);if(difference>0)nonzero++;
    assert.ok(difference<=values[4*i]!+1e-12,`case ${i}: ${difference} > ${values[4*i]}`);
    const band=2*data[32*i+4]!*data[32*i+5]!;
    assert.ok(Math.abs(values[4*i+3]!)>=band,`full trilinear enters cubic band ${i}`);
   }
   assert.ok(accepted>100&&nonzero>50,`${accepted} admitted cases, ${nonzero} with nonzero transport`);
  }finally{input.destroy();output.destroy();}
 });
});
