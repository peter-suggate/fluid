import assert from "node:assert/strict";
import test from "node:test";
import {uniformRegionTierWGSL} from "../lib/methods/uniform/uniform-region-heat";
import {narrowBandReuseLayout,narrowBandReuseWGSL} from "../lib/methods/uniform/uniform-narrow-band-reuse.wgsl";
import {withUniformDevice} from "./helpers/uniform-geometric";

test("vertex closure visits every incident stored region at interiors, faces, edges and corners",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("region closure walk",async device=>{
  const source=narrowBandReuseWGSL(narrowBandReuseLayout(0,64));
  const cold=source.slice(source.indexOf("fn nbVertexCold("),source.indexOf("fn nbReuseRegion("));
  const count=65**3,bytes=count*8;
  const tiers=device.createBuffer({size:256,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> tiers:array<u32>;
@group(0) @binding(1) var<storage,read_write> output:array<vec2u>;
fn umCorner(k:u32,n:u32)->vec3u{return vec3u(k%n,(k/n)%n,k/(n*n));}
fn umPageOfTile(tile:u32)->u32{let p=umCorner(tile,16u)/4u;return p.x+4u*(p.y+4u*p.z);}
fn umTileKey(p:vec3i)->u32{
 if(any(p<vec3i(0))||any(p>=vec3i(16))){return 0xffffffffu;}
 let tile=u32(p.x+16*(p.y+16*p.z));
 return select(tile,0xffffffffu,tiers[umPageOfTile(tile)]==3u);
}
fn umKeyStored(tile:u32)->bool{return tile<4096u;}
${uniformRegionTierWGSL(64,"tiers[region]")}
fn nbRegionCold(region:u32)->bool{return umRegionTier(region)==0u;}
${cold}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=${count}u){return;}
 let p=umCorner(id.x,65u);output[id.x]=vec2u(umVertexTier(p),u32(nbVertexCold(p)));
}`});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:tiers}},{binding:1,resource:{buffer:output}}]});
  try{
   for(let epoch=0;epoch<3;epoch++){
    // Include holes, every tier, and recycled region contents.
    const values=Uint32Array.from({length:64},(_,i)=>(Math.imul(i+3,13)+epoch*7+(i>>2))%4);
    device.queue.writeBuffer(tiers,0,values);
    const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(Math.ceil(count/64));p.end();e.copyBufferToBuffer(output,0,read,0,bytes);device.queue.submit([e.finish()]);
    await read.mapAsync(GPUMapMode.READ);const actual=new Uint32Array(read.getMappedRange());
    for(let z=0;z<=64;z++)for(let y=0;y<=64;y++)for(let x=0;x<=64;x++){
     let expected=2;
     // Independent geometric oracle: closed 16-cell region extents.
     for(let rz=Math.max(0,Math.ceil(z/16)-1);rz<=Math.min(3,Math.floor(z/16));rz++)
      for(let ry=Math.max(0,Math.ceil(y/16)-1);ry<=Math.min(3,Math.floor(y/16));ry++)
       for(let rx=Math.max(0,Math.ceil(x/16)-1);rx<=Math.min(3,Math.floor(x/16));rx++){
        const tier=values[rx+4*(ry+4*rz)]!;if(tier!==3)expected=Math.min(expected,tier);
       }
     const i=2*(x+65*(y+65*z));
     assert.equal(actual[i],expected,`epoch ${epoch}, vertex ${x},${y},${z}`);
     assert.equal(actual[i+1],Number(expected===0));
    }
    read.unmap();
   }
  }finally{if(read.mapState==="mapped")read.unmap();tiers.destroy();output.destroy();read.destroy();}
 });
});
