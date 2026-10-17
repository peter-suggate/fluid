import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";
import {UNIFORM_PAGE_POOL_KIND} from "../lib/methods/uniform/uniform-page-pool";
import {withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";

test("point-cell lookup preserves the first resident page for every incident AIR/resident/SOLID arrangement",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const source=readFileSync(new URL("../lib/methods/uniform/uniform-lattice-address.ts",import.meta.url),"utf8");
 const start=source.indexOf("fn umPointCell("),end=source.indexOf("\n}",start)+2;
 assert.ok(start>=0&&end>start);
 // All 3^8 page arrangements, all face/edge/corner masks, positive and
 // negative pages, and integer/fractional positions off the page faces.
 const arrangements=3**8,count=arrangements*8*2*2;
 await withUniformDevice("point-cell boundary enumeration",async device=>{
  const output=device.createBuffer({size:count*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   const module=device.createShaderModule({code:`
const UM_RESIDENT=${UNIFORM_PAGE_POOL_KIND.resident}u;
var<private> arrangement:u32;
var<private> high:vec3i;
fn umCellKind(c:vec3i)->u32{
 let upper=(c>>vec3u(5u))>=high;
 let i=u32(upper.x)|(u32(upper.y)<<1u)|(u32(upper.z)<<2u);
 let powers=array<u32,8>(1u,3u,9u,27u,81u,243u,729u,2187u);
 return (arrangement/powers[i])%3u;
}
${source.slice(start,end)}
@group(0) @binding(0) var<storage,read_write> result:array<vec4i>;
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=${count}u){return;}
 arrangement=i%${arrangements}u;let mode=i/${arrangements}u;let faces=mode&7u;
 let origin=select(32.0,-32.0,(mode&8u)!=0u);let offset=select(1.0,.25,(mode&16u)!=0u);
 let on=vec3<bool>((faces&1u)!=0u,(faces&2u)!=0u,(faces&4u)!=0u);
 let p=vec3f(origin)+select(vec3f(offset),vec3f(0),on);
 high=vec3i(floor(p/32.0));result[i]=vec4i(umPointCell(p),0);
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"probe"}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]});
   const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();device.queue.submit([encoder.finish()]);
   const result=new Int32Array((await readMixedBuffer(device,output)).buffer);
   for(let i=0;i<count;i++){
    const arrangement=i%arrangements,mode=Math.floor(i/arrangements),faces=mode&7;
    const origin=(mode&8)?-32:32,offset=(mode&16)?0.25:1;
    const c=[0,1,2].map(a=>Math.floor(origin+((faces&(1<<a))?0:offset)));
    const resident=(q:number[])=>{
     const page=q.reduce((mask,v,a)=>mask|((v>>5)>=(origin>>5)?1<<a:0),0);
     return Math.floor(arrangement/3**page)%3===UNIFORM_PAGE_POOL_KIND.resident;
    };
    let expected=c;
    // Independent specification: prefer the containing cell, then x-low,
    // y-low, xy-low, z-low, xz-low, yz-low, xyz-low, on eligible axes only.
    if(!resident(c))for(const mask of [1,2,3,4,5,6,7]){
     if((mask&faces)!==mask)continue;
     const q=c.map((v,a)=>v-((mask>>a)&1));if(resident(q)){expected=q;break;}
    }
    for(let a=0;a<3;a++)assert.equal(result[4*i+a],expected[a],`arrangement ${arrangement}, mode ${mode}, axis ${a}`);
   }
  }finally{output.destroy();}
 });
});
