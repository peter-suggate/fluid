import {readFileSync} from "node:fs";
import assert from "node:assert/strict";
import test from "node:test";
import {uniformSurfaceTransportWGSL} from "../lib/methods/uniform/uniform-surface-transport.wgsl";
import {uniformMixedVertexSamplingSource} from "../lib/methods/uniform/uniform-mixed-vertex-sampling.wgsl";
import {withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";

test("native-edge bounds cover actual trilinear transport across canonical cell boundaries",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const surface=readFileSync(new URL("../lib/methods/uniform/uniform-mixed-surface.ts",import.meta.url),"utf8"),cellSource=surface.slice(surface.indexOf("struct UMVertexCell"),surface.indexOf("// The cubic at q")),sum=uniformMixedVertexSamplingSource().match(/fn umVertexSum8\([^\n]+/u)![0];
 let seed=912348;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
 const count=2048,stride=160,data=new Float32Array(stride*count);const bad=new Set<number>();
 for(let i=0;i<count;i++){
  const p=[32,1024,65536][i%3]!,native=i%2?1:4,owner=native===4?4:(i%4===1?4:1),h=.05,c=(i%2?1:-1)*[.5,1,50][i%3]!;
  const noise=[0,.001,.02][i%3]!,g=Array.from({length:3},()=>Math.fround((random()-.5)*.02)),field=Array.from({length:125},(_,k)=>Math.fround(c+native*(g[0]!*(k%5-2)+g[1]!*(Math.floor(k/5)%5-2)+g[2]!*(Math.floor(k/25)-2))+(random()-.5)*noise));
  const L=[0,0,0];for(let z=0;z<5;z++)for(let y=0;y<5;y++)for(let x=0;x<5;x++)for(let axis=0;axis<3;axis++){
   const point=[x,y,z];if(point[axis]===4)continue;const at=x+5*y+25*z;point[axis]++;const to=point[0]!+5*point[1]!+25*point[2]!;L[axis]=Math.max(L[axis]!,Math.abs(field[to]!-field[at]!)/native);
  }
  for(let axis=0;axis<3;axis++)L[axis]=Math.fround(L[axis]!*(1+4*2**-23)+1.17549435e-38);
  const d=[1e-5,1e-4,.001,.01,.1,.4][i%6]!,lo=[p-d,p-d,p-d].map(Math.fround),hi=[p+d,p+d,p+d].map(Math.fround),euler=Array.from({length:3},(_,a)=>Math.fround(p+(a%2?1:-1)*d*.5)),q=Array.from({length:3},(_,a)=>Math.fround(p+(a%2?-1:1)*d*.5));
  const b=stride*i;data.set([p,p,p,native,owner,h,...lo,...hi,...euler,...q,...L,field[62]!],b);data.set(field,b+32);
  if(i%37===0){data[b+18]=-1;bad.add(i);}else if(i%37===1){data[b+9]=p+1;bad.add(i);}
 }
 await withUniformDevice("native-edge far transport",async device=>{
  const input=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),output=device.createBuffer({size:16*count,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   device.queue.writeBuffer(input,0,data);
   const module=device.createShaderModule({code:`${uniformSurfaceTransportWGSL}\n${sum}
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read_write> out:array<vec4f>;
var<private> fixtureBase:u32;var<private> fixtureWidth:u32;var<private> fixtureOrigin:vec3i;
const umRegularFine=false;
fn umPointCell(q:vec3f)->vec3i{return vec3i(floor(q));}fn umTileAt(q:vec3u)->u32{return 0u;}fn umTileWidth(tile:u32)->u32{return fixtureWidth;}
fn umCorner(k:u32,side:u32)->vec3u{return vec3u(k%side,(k/side)%side,k/(side*side));}
fn umLoadVertex8(origin:vec3u,width:u32)->array<f32,8>{var result:array<f32,8>;let base=(vec3i(origin)-fixtureOrigin)/i32(fixtureWidth)+vec3i(2);
 for(var k=0u;k<8u;k++){let p=base+vec3i(umCorner(k,2u));result[k]=a[fixtureBase+32u+u32(p.x+5*p.y+25*p.z)];}return result;}
fn umLoadCorner8(origin:vec3u,width:u32)->array<f32,8>{return umLoadVertex8(origin,width);}
${cellSource}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){let b=${stride}u*id.x;fixtureBase=b;fixtureWidth=u32(a[b+3u]);fixtureOrigin=vec3i(i32(a[b]));
 let p=vec3f(a[b]);let lo=vec3f(a[b+6u],a[b+7u],a[b+8u]);let hi=vec3f(a[b+9u],a[b+10u],a[b+11u]);let euler=vec3f(a[b+12u],a[b+13u],a[b+14u]);let q=vec3f(a[b+15u],a[b+16u],a[b+17u]);let L=vec3f(a[b+18u],a[b+19u],a[b+20u]);
 let bound=umSurfaceNativeFarBound(a[b+21u],L,vec3f(a[b+5u]),1.0/60.0,p,a[b+4u],UMTraceInterval(lo,hi,euler));
 out[id.x]=vec4f(bound.x,bound.w,umVertexCell(euler).value,umVertexCell(q).value);
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}}),group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(count);p.end();device.queue.submit([e.finish()]);const values=await readMixedBuffer(device,output);let admitted=0,crossing=0,nonzero=0;
   for(let i=0;i<count;i++){
    if(bad.has(i)){assert.equal(values[4*i+1],0,`invalid native bound${i}`);continue;}if(values[4*i+1]!==1)continue;admitted++;
    const difference=Math.abs(values[4*i+2]!-values[4*i+3]!);if(difference>0)nonzero++;
    assert.ok(difference<=values[4*i]!+1e-12,`case${i}: ${difference} > ${values[4*i]}`);
    assert.ok(Math.abs(values[4*i+3]!)>=2*data[stride*i+4]!*data[stride*i+5]!,`cubic gate entered${i}`);
    const w=data[stride*i+3]!;if(Math.floor(data[stride*i+12]!/w)!==Math.floor(data[stride*i+15]!/w))crossing++;
   }
   assert.ok(admitted>100&&crossing>50&&nonzero>50,`${admitted} admitted,${crossing} crossing cells,${nonzero} nonzero`);
  }finally{input.destroy();output.destroy();}
 });
});
