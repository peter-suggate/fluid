import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";
import {uniformMixedWallFaceWGSL} from "../lib/methods/uniform/uniform-mixed-walls.wgsl";
import {withUniformDevice} from "./helpers/uniform-geometric";

// Exercise the production workgroup/barrier listing, including all six
// faces' shared edges/corners, mixed h/4h widths, cold holds and far stores.
test("surface wall listing retains each canonical boundary vertex exactly once",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const source=readFileSync(new URL("../lib/methods/uniform/uniform-mixed-surface.ts",import.meta.url),"utf8");
 const listing=source.slice(source.indexOf("var<workgroup> wallListed:"),source.indexOf("// Six samples through one sampler"));
 await withUniformDevice("surface wall canonical listing",async device=>{
  const vertexCount=33**3,listWords=8+6*1089;
  const deferred=device.createBuffer({size:4*listWords,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const stored=device.createBuffer({size:4*vertexCount,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const read=device.createBuffer({size:4*(listWords+vertexCount),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   const module=device.createShaderModule({code:`
struct List {header:array<atomic<u32>,8>,data:array<u32>}
@group(0) @binding(0) var<storage,read_write> deferred:List;
@group(0) @binding(1) var<storage,read_write> stored:array<atomic<u32>>;
struct Params {hDt:vec4f}
const params=Params(vec4f(1));
override umRegularFine:bool=false;
fn umWallFaceCount()->u32{return 6u;}
fn umWallFace(i:u32)->u32{return 6u+i;}
fn umBlockPage(b:u32)->vec3u{return vec3u(0);}
fn umTileAt(t:vec3u)->u32{return t.x+8u*t.y+64u*t.z;}
fn umCoarseIncident(tile:u32,local:vec3u)->u32{return u32(tile%3u!=0u);}
fn umWallBlock(p:vec3i,f:u32)->u32{return u32(p[f>>1u]==i32(32u*(f&1u)));}
fn umWallPlane(p:vec3i)->u32{for(var f=0u;f<6u;f++){if(umWallBlock(p,f)!=0u){return f;}}return 6u;}
fn umVertexAt(p:vec3u)->u32{return p.x+33u*p.y+1089u*p.z;}
fn umStoreSurface(p:vec3u,width:u32,value:f32){atomicAdd(&stored[umVertexAt(p)],1u);}
fn umHoldCold(p:vec3u,width:u32)->bool{if((p.x+p.y+p.z)%7u!=0u){return false;}umStoreSurface(p,width,0.0);return true;}
fn umFarAir(p:vec3u)->f32{return select(-1.0,1.0,(p.x+p.y+p.z)%5u==0u);}
fn umRecordRegion(p:vec3u,tier:u32){}
${uniformMixedWallFaceWGSL}
${listing}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
   for(const regular of [false,true]){
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"advectWalls",constants:{umRegularFine:Number(regular)}}});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[deferred,stored].map((buffer,binding)=>({binding,resource:{buffer}}))});
    const e=device.createCommandEncoder();e.clearBuffer(deferred);e.clearBuffer(stored);
    const pass=e.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(4);pass.end();
    e.copyBufferToBuffer(deferred,0,read,0,deferred.size);e.copyBufferToBuffer(stored,0,read,deferred.size,stored.size);device.queue.submit([e.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const words=new Uint32Array(read.getMappedRange().slice(0));read.unmap();
    const seen=words.slice(listWords),count=words[0]!;
    assert.ok(count<=6*1089);
    for(let i=0;i<count;i++){const word=words[8+i]!,key=word&0x3fffffff;assert.ok(key<vertexCount);seen[key]!++;}
    for(let z=0;z<=32;z++)for(let y=0;y<=32;y++)for(let x=0;x<=32;x++){
     const p=[x,y,z],wall=p.some(v=>v===0||v===32);
     const tile=p.map(v=>Math.min(v,31)>>2);
     const width=regular||((tile[0]!+8*tile[1]!+64*tile[2]!)%3===0)?1:4;
     const expected=wall&&p.every(v=>v%width===0)?1:0;
     assert.equal(seen[x+33*y+1089*z],expected,`${regular?"regular":"mixed"}: ${p}`);
    }
   }
  }finally{if(read.mapState==="mapped")read.unmap();read.destroy();stored.destroy();deferred.destroy();}
 });
});
