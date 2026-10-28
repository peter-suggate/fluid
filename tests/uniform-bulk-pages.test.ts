import assert from "node:assert/strict";
import test from "node:test";
import {BULK16_PAGE,BULK16_PAGE_WORDS,bulk16PageLayout,bulk16PageSlot,bulk16PageWord,prolongBulk16Page,restrictBulk16Page} from "../lib/methods/uniform/uniform-bulk-pages-reference";
import {BULK16_STATUS,bulk16Failure,bulk16LifecycleWGSL,UniformBulkPages} from "../lib/methods/uniform/uniform-bulk-pages";
import {uniformLatticeAddress,uniformPageKind,uniformPageWord,uniformPageBlock,uniformBlockStored,UNIFORM_KIND} from "../lib/methods/uniform/uniform-lattice-address";
import {createUniformMixedLayoutFromWidths} from "../lib/methods/uniform/uniform-mixed-layout";
import {UniformMixedOwnership} from "../lib/methods/uniform/uniform-mixed-ownership";
import {uniformMixedTopologyWGSL} from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import {UNIFORM_DIRECTORY_TOP_WORDS,UNIFORM_DIRECTORY_WORDS,uniformDirectoryFlat} from "../lib/methods/uniform/uniform-window-address";
import {withUniformDevice} from "./helpers/uniform-geometric";

test("BULK16 words cannot alias resident blocks or the AIR and SOLID sentinels",()=>{
 for(const slot of [1,127,4096,100_000,0x3fffffff])assert.equal(bulk16PageSlot(bulk16PageWord(slot)),slot);
 for(const word of [0,1,512,0x7fffffff,0x80000000,0xc0000001,0xffffffff])assert.equal(bulk16PageSlot(word),0);
 for(const slot of [0,-1,.5,0x40000000])assert.throws(()=>bulk16PageWord(slot));
});

test("BULK16 directory entries are liquid pages without a resident 4h block",()=>{
 const address=uniformLatticeAddress([64,32,32]);
 const tail=address.seed.tail,child=UNIFORM_DIRECTORY_TOP_WORDS+64*tail[0]!;
 tail[child]=bulk16PageWord(37);
 assert.equal(uniformPageWord(address,0,0,0),bulk16PageWord(37));
 assert.equal(uniformPageKind(address,0,0,0),UNIFORM_KIND.bulk);
 assert.equal(uniformBlockStored(address,uniformPageBlock(address,0,0,0)),false);
 assert.equal(uniformPageKind(address,1,0,0),UNIFORM_KIND.resident);
 assert.equal(uniformPageKind(address,-1,0,0),UNIFORM_KIND.solid);
});

test("whole-page transfer conserves face flux and hydrostatic pressure means",()=>{
 const face=Float64Array.from({length:36},(_,i)=>Math.sin(i)*3),pressure=Float64Array.from({length:8},(_,i)=>100+5*i);
 for(const gradient of [[0,0,0],[.3,-9.8,.7]] as const){
  const fine=prolongBulk16Page(face,pressure,gradient),roundtrip=restrictBulk16Page(fine.faces,fine.pressure);
  for(let i=0;i<36;i++)assert.ok(Math.abs(roundtrip.faces[i]!-face[i]!)<1e-12,`face ${i} flux`);
  for(let i=0;i<8;i++)assert.ok(Math.abs(roundtrip.pressure[i]!-pressure[i]!)<1e-12,`cell ${i} mean`);
 }
});

test("BULK16 page storage includes seams and stays near four KiB per page at vast-scene scale",()=>{
 assert.ok(BULK16_PAGE.seamHeads+8*96<=BULK16_PAGE_WORDS);
 const plan=bulk16PageLayout(100_000);
 assert.ok(plan.records>=plan.pending+plan.capacity);
 assert.ok(plan.bytes<420_000_000);
 assert.equal(plan.words-plan.records,1024*100_001);
});

test("BULK16 GPU slots quarantine retired pages and fail explicitly at fixed capacity",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("BULK16 lifecycle",async device=>{
  const pages=new UniformBulkPages(device,3);
  const output=device.createBuffer({size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:pages.state.size+32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   await pages.initialize();
   const module=device.createShaderModule({code:bulk16LifecycleWGSL(pages.layout)+`
@group(0) @binding(1) var<storage,read_write> result:array<u32>;
@compute @workgroup_size(1) fn claimTwo(){
 result[0]=bpClaim(vec3i(4,5,6));result[1]=bpClaim(vec3i(7,8,9));
 if(result[0]!=0u){bpWrite(result[0],500u,42u);}
}
@compute @workgroup_size(1) fn retireOne(){bpRetire(result[0]);}
@compute @workgroup_size(1) fn reclaim(){result[2]=bpClaim(vec3i(-2,3,4));result[3]=bpRead(result[2],500u);}
`});
   const layout=device.createBindGroupLayout({entries:[0,1].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}}))});
   const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
   const pipes=await Promise.all(["claimTwo","retireOne","reclaim"].map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
   const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:pages.state}},{binding:1,resource:{buffer:output}}]});
   const frame=async(entries:number[])=>{
    const encoder=device.createCommandEncoder();pages.encodeBegin(encoder);
    for(const entry of entries){const pass=encoder.beginComputePass();pass.setPipeline(pipes[entry]!);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();}
    pages.encodeFinish(encoder);encoder.copyBufferToBuffer(pages.state,0,read,0,pages.state.size);encoder.copyBufferToBuffer(output,0,read,pages.state.size,32);
    device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const values=new Uint32Array(read.getMappedRange()).slice();read.unmap();return values;
   };
   const first=await frame([0]);assert.equal(first[BULK16_STATUS.live],2);assert.equal(first[BULK16_STATUS.free],1);assert.equal(bulk16Failure(first),undefined);
   const second=await frame([1,0]);assert.equal(second[BULK16_STATUS.live],2);assert.equal(second[BULK16_STATUS.free],0);assert.equal(second[BULK16_STATUS.pending],1);
   assert.match(bulk16Failure(second)!,/fixed slot capacity 3 exhausted/);
   const third=await frame([2]),out=third.subarray(pages.state.size/4);
   assert.equal(out[2],1,"the retired slot is available only in the next frame");
   assert.equal(out[3],0,"reused slots discard the previous pressure and seam state");
   assert.equal(third[BULK16_STATUS.live],3);assert.equal(third[BULK16_STATUS.pending],0);
   assert.ok(bulk16Failure(third),"capacity failures remain sticky");
   assert.deepEqual(Array.from(third.subarray(pages.layout.live,pages.layout.live+3)),[1,2,3]);
  }finally{read.destroy();output.destroy();pages.destroy();}
 });
});


test("BULK16 topology readers preserve tagged slots under the fixed world specialization",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("BULK16 topology",async device=>{
  const dimensions=[64,32,32] as const,address=uniformLatticeAddress(dimensions);
  const layout=createUniformMixedLayoutFromWidths({dimensions,cellSize_m:[1,1,1],origin_m:{x:0,y:0,z:0}},new Uint8Array(address.boxTiles).fill(4),[],address);
  const pages=new UniformBulkPages(device,2),tail=address.seed.tail;
  tail[UNIFORM_DIRECTORY_TOP_WORDS+64*tail[0]!]=bulk16PageWord(1);uniformDirectoryFlat(tail,[0,0,0]);
  const ownership=new UniformMixedOwnership(device,layout,false,0,true,pages);
  const output=device.createBuffer({size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   device.queue.writeBuffer(pages.state,4*(pages.layout.records+BULK16_PAGE_WORDS+BULK16_PAGE.pressure),new Float32Array([12.5]));
   const module=device.createShaderModule({code:uniformMixedTopologyWGSL(ownership.capacity,0)+`
@group(1) @binding(0) var<storage,read_write> out:array<u32>;
@compute @workgroup_size(1) fn main(){
 out[0]=umPageWord(vec3i(0));out[1]=umPageKind(vec3i(0));out[2]=u32(umCellPresent(vec3i(2)));
 out[3]=umPageBulk(vec3i(0));out[4]=bitcast<u32>(umBulkPressure(out[3],vec3u(0)));
 let p=umPointHeld(vec3f(8));out[5]=bitcast<u32>(p.x);out[6]=umPageKind(vec3i(-1,0,0));
}`});
   const resultLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
   const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,resultLayout]}),compute:{module,entryPoint:"main",constants:{umFixedWorldGeometry:1,...Object.fromEntries(Array.from({length:8},(_,i)=>[`umWindowWord${24+i}`,tail[UNIFORM_DIRECTORY_WORDS+24+i]!]))}}});
   const group=device.createBindGroup({layout:resultLayout,entries:[{binding:0,resource:{buffer:output}}]});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,ownership.bindGroup);p.setBindGroup(1,group);p.dispatchWorkgroups(1);p.end();e.copyBufferToBuffer(output,0,read,0,32);device.queue.submit([e.finish()]);
   await read.mapAsync(GPUMapMode.READ);const u=new Uint32Array(read.getMappedRange()),f=new Float32Array(u.buffer);
   assert.deepEqual(Array.from(u.subarray(0,4)),[bulk16PageWord(1),UNIFORM_KIND.bulk,0,1]);assert.equal(f[4],12.5);assert.equal(f[5],8);assert.equal(u[6],UNIFORM_KIND.solid);
  }finally{read.destroy();output.destroy();ownership.destroy();pages.destroy();}
 });
});
