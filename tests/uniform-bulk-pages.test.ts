import assert from "node:assert/strict";
import test from "node:test";
import {BULK16_PAGE,BULK16_PAGE_WORDS,BULK16_PHI_TEXELS,BULK16_TAIL_CHANNELS,bulk16FaceIndexWGSL,bulk16PageLayout,bulk16ReadWGSL,bulk16PageSlot,bulk16PageWord,bulk16TransferWGSL,prolongBulk16Page,restrictBulk16Page} from "../lib/methods/uniform/uniform-bulk-pages-reference";
import {BULK16_STATUS,bulk16Failure,bulk16LifecycleWGSL,bulk16LiveWGSL,bulk16SlotCapacity,UniformBulkPages} from "../lib/methods/uniform/uniform-bulk-pages";
import {uniformLatticeAddress,uniformPageKind,uniformPageWord,uniformPageBlock,uniformBlockStored,UNIFORM_KIND,UNIFORM_NEAR_SPAN,UNIFORM_PAGE_TABLE_WORDS,uniformBlockHeld,uniformBlockNeighbourLayout,uniformBlockNeighbourWords,uniformNearFrom} from "../lib/methods/uniform/uniform-lattice-address";
import {UniformBlockNeighbours} from "../lib/methods/uniform/uniform-block-neighbours";
import {uniformExtrapolationClosedFace} from "../lib/methods/uniform/uniform-velocity-extrapolation-twin";
import {narrowBandNear} from "../lib/methods/uniform/uniform-narrow-band-near.wgsl";
import {createUniformMixedLayoutFromWidths} from "../lib/methods/uniform/uniform-mixed-layout";
import {UniformMixedOwnership} from "../lib/methods/uniform/uniform-mixed-ownership";
import {uniformMixedTopologyNeighboursWGSL,uniformMixedTopologyWGSL} from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import {UNIFORM_DIRECTORY_TOP_WORDS,UNIFORM_DIRECTORY_WORDS,uniformDirectoryFlat} from "../lib/methods/uniform/uniform-window-address";
import {withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";

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
  // A 16h face's flux is its sixteen subfaces' (pairwise) sum: bit for bit.
  for(let i=0;i<36;i++)assert.equal(roundtrip.faces[i],face[i],`face ${i} flux`);
  // The cell mean: exact without a gradient; the gradient's offsets cancel to rounding.
  for(let i=0;i<8;i++){if(gradient[1]===0)assert.equal(roundtrip.pressure[i],pressure[i]);else assert.ok(Math.abs(roundtrip.pressure[i]!-pressure[i]!)<1e-12,`cell ${i} mean`);}
 }
});

// The GPU transfer (bulk16TransferWGSL) over plain buffers: a random 4h page
// demotes, its 16h state promotes, and that demotes again.
test("BULK16 GPU transfer: demote after promote returns every face flux bit for bit, and the CPU twin's means",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("BULK16 transfer",async device=>{
  const module=device.createShaderModule({code:bulk16FaceIndexWGSL+`
@group(0) @binding(0) var<storage,read_write> fine:array<f32>;
@group(0) @binding(1) var<storage,read_write> coarse:array<f32>;
@group(0) @binding(2) var<uniform> gradient:vec4f;
// Component-major 9x8x8 faces, the normal plane fastest; then 8^3 pressures.
fn bpFineFace(axis:u32,n:u32,i:u32,j:u32)->f32{return fine[576u*axis+n+9u*(i+8u*j)];}
fn bpFinePressure(c:vec3u)->f32{return fine[1728u+c.x+8u*(c.y+8u*c.z)];}
`+bulk16TransferWGSL+`
@compute @workgroup_size(64) fn restrictPage(@builtin(local_invocation_index) lane:u32){
 if(lane<36u){coarse[lane]=bpRestrictFace(lane);}else if(lane<44u){coarse[lane]=bpRestrictPressure(lane-36u);}
}
@compute @workgroup_size(64) fn prolongPage(@builtin(local_invocation_index) lane:u32){
 var face:array<f32,36>;var pressure:array<f32,8>;
 for(var k=0u;k<36u;k++){face[k]=coarse[k];}for(var r=0u;r<8u;r++){pressure[r]=coarse[36u+r];}
 for(var i=lane;i<1728u;i+=64u){let axis=i/576u;let n=i%9u;let t=(i%576u)/9u;fine[i]=bpProlongFace(&face,axis,n,t%8u,t/8u);}
 for(var i=lane;i<512u;i+=64u){fine[1728u+i]=bpProlongPressure(&pressure,vec3u(i%8u,(i/8u)%8u,i/64u),gradient.xyz);}
}`});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
  const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;
  const fine=device.createBuffer({size:4*2240,usage}),coarse=device.createBuffer({size:4*44,usage}),uniform=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const read=device.createBuffer({size:4*(2240+88),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   const layout=device.createBindGroupLayout({entries:[...[0,1].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform" as const}}]});
   const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
   const [demote,promote]=await Promise.all(["restrictPage","prolongPage"].map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
   const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:fine}},{binding:1,resource:{buffer:coarse}},{binding:2,resource:{buffer:uniform}}]});
   let seed=11;const random=()=>{seed=(seed*1103515245+12345)%2147483648;return seed/2147483648-0.5;};
   const field=Float32Array.from({length:2240},(_,i)=>i<1728?3*random():9810+400*random());
   for(const g of [[0,0,0],[0.3,-49.05,0.7]]){
    device.queue.writeBuffer(fine,0,field);device.queue.writeBuffer(uniform,0,new Float32Array([...g,0]));
    const e=device.createCommandEncoder();
    const run=(pipeline:GPUComputePipeline)=>{const p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(1);p.end();};
    run(demote);e.copyBufferToBuffer(coarse,0,read,4*2240,4*44);run(promote);e.copyBufferToBuffer(fine,0,read,0,4*2240);run(demote);e.copyBufferToBuffer(coarse,0,read,4*2284,4*44);
    device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);
    const all=new Float32Array(read.getMappedRange()).slice();read.unmap();
    const first=all.subarray(2240,2284),again=all.subarray(2284,2328),promoted=all.subarray(0,2240);
    // The twin's means of the same f32 page agree to f32 rounding.
    const twin=restrictBulk16Page(field.subarray(0,1728),field.subarray(1728));
    for(let k=0;k<36;k++)assert.ok(Math.abs(first[k]!-twin.faces[k]!)<=2e-7*Math.max(1,Math.abs(twin.faces[k]!)),`face ${k}: ${first[k]} against ${twin.faces[k]}`);
    for(let r=0;r<8;r++)assert.ok(Math.abs(first[36+r]!-twin.pressure[r]!)<=2e-7*Math.abs(twin.pressure[r]!),`pressure ${r}`);
    // Demote after promote: every face mean (so every 16h face's flux) bit for bit.
    for(let k=0;k<36;k++)assert.equal(again[k],first[k],`face ${k} after the round trip`);
    // The cell mean: exact with no gradient; with one, its offsets cancel to f32 rounding.
    for(let r=0;r<8;r++){if(g[1]===0)assert.equal(again[36+r],first[36+r]);else assert.ok(Math.abs(again[36+r]!-first[36+r]!)<=2e-7*Math.abs(first[36+r]!),`pressure ${r} after the round trip`);}
    // The promoted page is the CPU twin's prolongation of the same means.
    const cpu=prolongBulk16Page(first.subarray(0,36),first.subarray(36),g as [number,number,number]);
    for(let i=0;i<1728;i++)assert.ok(Math.abs(promoted[i]!-cpu.faces[i]!)<=2e-7*Math.max(1,Math.abs(cpu.faces[i]!)),`fine face ${i}`);
    for(let i=0;i<512;i++)assert.ok(Math.abs(promoted[1728+i]!-cpu.pressure[i]!)<=2e-7*Math.abs(cpu.pressure[i]!),`fine pressure ${i}`);
   }
  }finally{read.destroy();fine.destroy();coarse.destroy();uniform.destroy();}
 });
});

// The pool's BULK16 transitions: upDemote retires the block and leaves the
// slot word in the entry (which a request then leaves alone); upPromote
// gives the page a block again, its super-block count unchanged.
test("BULK16 pool: a demoted page gives its block back, keeps its word against requests, and a promote claims it",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const {UniformPagePool,uniformPagePoolWGSL,UNIFORM_PAGE_POOL_STATUS:S,UNIFORM_PAGE_POOL_RECORD:R,UNIFORM_PAGE_POOL_BLOCK:B}=await import("../lib/methods/uniform/uniform-page-pool");
 await withUniformDevice("BULK16 pool transitions",async device=>{
  const address=uniformLatticeAddress([128,32,32],false,{blocks:6});
  const pool=new UniformPagePool(device,address.pool);
  const control=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  try{
   pool.seed(address.seed);await pool.initialize();
   // x: 0 request, 1 demote, 2 promote; y the page (x coordinate); z the BULK16 word.
   const module=device.createShaderModule({code:uniformPagePoolWGSL(pool.layout,0,0,1)+`
@group(0) @binding(2) var<uniform> c:vec4u;
@compute @workgroup_size(1) fn act(){let q=vec3i(i32(c.y),0,0);
 if(c.x==0u){upRequest(q);}else if(c.x==1u){upDemote(upBlockOf(q),c.z);}else{upPromote(q);}}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"act"}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:pool.state}},{binding:1,resource:{buffer:pool.work}},{binding:2,resource:{buffer:control}}]});
   const commit=async(acts:[number,number,number][])=>{
    for(const [kind,x,word] of acts){
     device.queue.writeBuffer(control,0,new Uint32Array([kind,x,word,0]));
     const e=device.createCommandEncoder();const pass=e.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();device.queue.submit([e.finish()]);
    }
    const e=device.createCommandEncoder();pool.encodeCommit(e);device.queue.submit([e.finish()]);
    return new Uint32Array((await readMixedBuffer(device,pool.state)).buffer);
   };
   const word=bulk16PageWord(5);
   const entry=(state:Uint32Array,x:number)=>{const t=state.subarray(pool.layout.state.table);const top=t[x>>2]!;return t[UNIFORM_DIRECTORY_TOP_WORDS+64*top+(x&3)]!;};
   const first=await commit([[0,1,0],[0,2,0]]);
   assert.equal(first[S.fatal],0);assert.equal(first[S.resident],2);
   const block=entry(first,2);assert.ok(block>0&&block<6);
   const demoted=await commit([[0,1,0],[0,2,0],[1,2,word]]);
   assert.equal(demoted[S.fatal],0);assert.equal(demoted[S.resident],1,"the demoted page's block retires though it was requested");
   assert.equal(entry(demoted,2),word);assert.equal(demoted[pool.layout.state.records+R.words*block+R.state],B.pending);
   const held=await commit([[0,1,0],[0,2,0]]);
   assert.equal(held[S.fatal],0);assert.equal(held[S.resident],1);assert.equal(held[S.claimed],0,"a request leaves a BULK16 page alone");
   assert.equal(entry(held,2),word);
   const promoted=await commit([[0,1,0],[2,2,0]]);
   assert.equal(promoted[S.fatal],0);assert.equal(promoted[S.resident],2);assert.equal(promoted[S.claimed],1);
   const again=entry(promoted,2);assert.ok(again>0&&again<6,`page 2 holds block ${again}`);
   assert.equal(promoted[pool.layout.state.records+R.words*again]!,2);
   // Its super-block stayed counted throughout: a later idle retire still clears it cleanly.
   const kept=await commit([[0,1,0],[0,2,0]]);assert.equal(kept[S.fatal],0);assert.equal(kept[S.resident],2);
  }finally{pool.destroy();control.destroy();}
 });
});

test("BULK16 page storage holds its phi block and stays within four KiB per page at vast-scene scale",()=>{
 assert.ok(BULK16_PAGE.phi+BULK16_PHI_TEXELS<=BULK16_PAGE.tail3);
 assert.ok(BULK16_PAGE.tail5+BULK16_TAIL_CHANNELS<=BULK16_PAGE.promoteLevel&&BULK16_PAGE.promoteLevel<BULK16_PAGE_WORDS);
 const plan=bulk16PageLayout(100_000);
 assert.ok(plan.records>=plan.pending+plan.capacity);
 assert.ok(plan.bytes<420_000_000);
 assert.equal(plan.words-plan.records,BULK16_PAGE_WORDS*100_001);
 // A slot's record and its three list words.
 assert.ok(plan.bytes/plan.capacity<=4096);
});

test("BULK16 slot capacity is the whole pages of liquid the scene holds, at most the box's",()=>{
 // A BULK16 page is 32^3 h cells of liquid: 2.5 pages of liquid fill two.
 assert.equal(bulk16SlotCapacity(2.5*32768,400),2);
 assert.equal(bulk16SlotCapacity(32767,400),1,"a store keeps one slot");
 assert.equal(bulk16SlotCapacity(1e9,400),400,"never more than the box's pages");
 assert.throws(()=>bulk16SlotCapacity(Number.NaN,400));
});

test("BULK16 GPU slots quarantine retired pages and refuse claims past fixed capacity",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("BULK16 lifecycle",async device=>{
  const pages=new UniformBulkPages(device,3);
  const output=device.createBuffer({size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:pages.state.size+32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   await pages.initialize();
   const module=device.createShaderModule({code:bulk16LifecycleWGSL(pages.layout)+`
@group(0) @binding(1) var<storage,read_write> result:array<u32>;
@compute @workgroup_size(1) fn claimTwo(){
 result[0]=bpClaim(vec3i(4,5,6),255u);result[1]=bpClaim(vec3i(7,8,9),255u);
 if(result[0]!=0u){bpWrite(result[0],500u,42u);}
}
@compute @workgroup_size(1) fn retireOne(){bpRetire(result[0]);}
@compute @workgroup_size(1) fn reclaim(){result[2]=bpClaim(vec3i(-2,3,4),255u);result[3]=bpRead(result[2],500u);}
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
   // The claim past the free slots is refused (the page stays 4h), counted, not fatal.
   assert.equal(second.subarray(pages.state.size/4)[1],0,"the second claim found no free slot");
   assert.equal(second[BULK16_STATUS.refused],1);assert.equal(bulk16Failure(second),undefined);
   const third=await frame([2]),out=third.subarray(pages.state.size/4);
   assert.equal(out[2],1,"the retired slot is available only in the next frame");
   assert.equal(out[3],0,"reused slots discard the previous pressure and seam state");
   assert.equal(third[BULK16_STATUS.live],3);assert.equal(third[BULK16_STATUS.pending],0);
   assert.equal(third[BULK16_STATUS.refused],1,"refusals accumulate across frames");assert.equal(bulk16Failure(third),undefined);
   assert.deepEqual(Array.from(third.subarray(pages.layout.live,pages.layout.live+3)),[1,2,3]);
   // A reader compiled without the capacity (the renderer's) finds the live
   // slots' pages from the status words alone.
   const live=device.createBuffer({size:64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),liveRead=device.createBuffer({size:64,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   try{
    const reader=device.createShaderModule({code:`@group(0) @binding(0) var<storage,read> liveStore:array<u32>;
@group(0) @binding(1) var<storage,read_write> livePages:array<i32>;
${bulk16LiveWGSL("liveStore")}
@compute @workgroup_size(1) fn main(){let n=liveStoreLive();livePages[0]=i32(n);for(var i=0u;i<n;i++){let p=liveStoreLivePage(i);livePages[1u+3u*i]=p.x;livePages[2u+3u*i]=p.y;livePages[3u+3u*i]=p.z;}}`});
    const readerLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
    const pipe=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[readerLayout]}),compute:{module:reader,entryPoint:"main"}});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipe);
    pass.setBindGroup(0,device.createBindGroup({layout:readerLayout,entries:[{binding:0,resource:pages.binding},{binding:1,resource:{buffer:live}}]}));pass.dispatchWorkgroups(1);pass.end();
    encoder.copyBufferToBuffer(live,0,liveRead,0,64);device.queue.submit([encoder.finish()]);await liveRead.mapAsync(GPUMapMode.READ);
    const found=new Int32Array(liveRead.getMappedRange().slice(0));liveRead.unmap();
    const signed=new Int32Array(third.buffer,third.byteOffset,pages.state.size/4);
    const expected=[1,2,3].flatMap(slot=>Array.from(signed.subarray(pages.layout.records+BULK16_PAGE_WORDS*slot+BULK16_PAGE.page,pages.layout.records+BULK16_PAGE_WORDS*slot+BULK16_PAGE.page+3)));
    assert.deepEqual(Array.from(found.subarray(0,10)),[3,...expected]);
    assert.deepEqual(expected.slice(0,3),[-2,3,4]);
   }finally{live.destroy();liveRead.destroy();}
  }finally{read.destroy();output.destroy();pages.destroy();}
 });
});


test("BULK16 topology readers: the raw word keeps the slot, the page word folds to blocks-1 under the fixed world specialization",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("BULK16 topology",async device=>{
  const dimensions=[64,32,32] as const,address=uniformLatticeAddress(dimensions);
  const layout=createUniformMixedLayoutFromWidths({dimensions,cellSize_m:[1,1,1],origin_m:{x:0,y:0,z:0}},new Uint8Array(address.boxTiles).fill(4),[],address);
  const pages=new UniformBulkPages(device,2),tail=address.seed.tail;
  tail[UNIFORM_DIRECTORY_TOP_WORDS+64*tail[0]!]=bulk16PageWord(1);uniformDirectoryFlat(tail,[0,0,0]);
  const ownership=new UniformMixedOwnership(device,layout,false,0,true,pages);
  const output=device.createBuffer({size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   device.queue.writeBuffer(pages.state,4*(pages.layout.records+BULK16_PAGE_WORDS+BULK16_PAGE.pressure),new Float32Array([12.5]));
   // The slot store binds in its readers' own group: the shared topology group stays at three bindings.
   const module=device.createShaderModule({code:uniformMixedTopologyWGSL(ownership.capacity,0)+bulk16ReadWGSL(pages.layout,1,1)+`
@group(1) @binding(0) var<storage,read_write> out:array<u32>;
@compute @workgroup_size(1) fn main(){
 let raw=umPageRawWord(vec3i(0));out[0]=raw;out[1]=umPageKind(vec3i(0));out[2]=u32(umCellPresent(vec3i(2)));
 out[3]=select(0u,raw&0x3fffffffu,umBulkTag(raw));out[4]=bitcast<u32>(umBulkPressure(out[3],vec3u(0)));
 let p=umPointHeld(vec3f(8));out[5]=bitcast<u32>(p.x);out[6]=umPageKind(vec3i(-1,0,0));out[7]=umPageWord(vec3i(0));
}`});
   const resultLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}}]});
   const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,resultLayout]}),compute:{module,entryPoint:"main",constants:{umFixedWorldGeometry:1,umBulkSlots:1,...Object.fromEntries(Array.from({length:8},(_,i)=>[`umWindowWord${24+i}`,tail[UNIFORM_DIRECTORY_WORDS+24+i]!]))}}});
   const group=device.createBindGroup({layout:resultLayout,entries:[{binding:0,resource:{buffer:output}},{binding:1,resource:{buffer:pages.state}}]});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,ownership.bindGroup);p.setBindGroup(1,group);p.dispatchWorkgroups(1);p.end();e.copyBufferToBuffer(output,0,read,0,32);device.queue.submit([e.finish()]);
   await read.mapAsync(GPUMapMode.READ);const u=new Uint32Array(read.getMappedRange()),f=new Float32Array(u.buffer);
   assert.deepEqual(Array.from(u.subarray(0,4)),[bulk16PageWord(1),UNIFORM_KIND.bulk,0,1]);assert.equal(f[4],12.5);assert.equal(f[5],8);assert.equal(u[6],UNIFORM_KIND.solid);
   // umPageWord of the fixed geometry folds the tag to blocks-1, which no block
   // holds, so its range bound (kept words below SOLID) still stands.
   assert.equal(u[7],address.blocks-1);
  }finally{read.destroy();output.destroy();ownership.destroy();pages.destroy();}
 });
});

// A hand-placed BULK16 page between two resident ones: page (1,0,0) of a
// 3x1x1-page box takes slot 5, and its block's record is freed. Nothing 4h is
// stored there, so every neighbour path holds at it and its faces are wall
// slots (SOLID-like), while its kind and slot stay readable.
const bulkBox=()=>{
 const address=uniformLatticeAddress([96,32,32]),tail=address.seed.tail,records=address.seed.blocks;
 const block=uniformPageWord(address,1,0,0);
 tail[UNIFORM_DIRECTORY_TOP_WORDS+64*tail[0]!+1]=bulk16PageWord(5);uniformDirectoryFlat(tail,[0,0,0]);
 records[block]=0xffffffff;tail[UNIFORM_PAGE_TABLE_WORDS+block]=0xffffffff;
 return {address,low:uniformPageWord(address,0,0,0),high:uniformPageWord(address,2,0,0)};
};

test("BULK16 page: every host neighbour twin holds at it and lists its faces as walls",()=>{
 const {address,low,high}=bulkBox(),blocks=address.blocks;
 assert.equal(uniformPageKind(address,1,0,0),UNIFORM_KIND.bulk);
 assert.equal(uniformBlockStored(address,uniformPageBlock(address,1,0,0)),false);
 const words=uniformBlockNeighbourWords(address.seed.tail,address.seed.blocks);
 assert.equal(words[27*low+14],bulk16PageWord(5));assert.equal(words[27*high+12],bulk16PageWord(5));
 const walls=new Set(words.subarray(27*blocks+1,27*blocks+1+words[27*blocks]!));
 assert.ok(walls.has(6*low+1)&&walls.has(6*high),"both faces toward the BULK16 page are wall faces");
 assert.deepEqual(uniformBlockHeld(words,low,[1,0,0],[false,false,false]),[true,false,false]);
 assert.deepEqual(uniformBlockHeld(words,high,[-1,0,0],[false,false,false]),[true,false,false]);
 assert.deepEqual(uniformNearFrom(words,low,[0,0,0],UNIFORM_NEAR_SPAN.cell,[33,5,6],[0,0,0]),[31,5,6]);
 assert.deepEqual(uniformNearFrom(words,high,[2,0,0],UNIFORM_NEAR_SPAN.tile,[15,2,3],[0,0,0]),[16,2,3]);
 const pool={table:address.seed.tail,records:address.seed.blocks,rows:words};
 assert.equal(uniformExtrapolationClosedFace(pool,low,[7,1,2],0,8),true);
 assert.equal(uniformExtrapolationClosedFace(pool,low,[1,7,2],1,8),true,"the box's own wall");
 assert.equal(uniformExtrapolationClosedFace(pool,high,[7,1,2],0,8),true,"the box's own wall");
 assert.equal(uniformExtrapolationClosedFace(pool,low,[6,1,2],0,8),false);
 assert.deepEqual(narrowBandNear(address.seed.tail,words,blocks,UNIFORM_NEAR_SPAN.cell,[20,5,6],[33,5,6],[0,0,0]),[31,5,6]);
});

test("BULK16 page: GPU readers without the slot store keep its word and hold at it, as the host twins do",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("BULK16 readers",async device=>{
  const {address,low,high}=bulkBox(),tail=address.seed.tail,blocks=address.blocks;
  const dimensions=[96,32,32] as const;
  const layout=createUniformMixedLayoutFromWidths({dimensions,cellSize_m:[1,1,1],origin_m:{x:0,y:0,z:0}},new Uint8Array(address.boxTiles).fill(4),[],address);
  const ownership=new UniformMixedOwnership(device,layout,false,0,true);
  const nl=uniformBlockNeighbourLayout(blocks);
  const rows=device.createBuffer({size:4*nl.end,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:64+4*nl.end,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   device.queue.writeBuffer(rows,0,tail.subarray(0,UNIFORM_PAGE_TABLE_WORDS+blocks));
   const builder=new UniformBlockNeighbours(device,blocks,[{buffer:rows,offset:0}]);await builder.initialize();
   const module=device.createShaderModule({code:uniformMixedTopologyWGSL(ownership.capacity,0)+uniformMixedTopologyNeighboursWGSL()+`
@group(1) @binding(0) var<storage,read_write> out:array<u32>;
@compute @workgroup_size(1) fn main(){
 let page=vec3i(1,0,0);
 let raw=umPageRawWord(page);out[0]=raw;out[1]=umPageKind(page);out[2]=select(0u,raw&0x3fffffffu,umBulkTag(raw));out[3]=u32(umCellPresent(vec3i(40,5,6)));
 let near=umCellNearFrom(${low}u,vec3i(33,5,6));out[4]=u32(near.x);out[5]=u32(near.y);out[6]=u32(near.z);
 let back=umCellNearFrom(${high}u,vec3i(60,5,6));out[7]=u32(back.x);
 out[8]=umBlockNeighbour(${low}u,vec3i(1,0,0));out[9]=u32(umFaceLow(vec3i(63,5,6),0u));out[10]=u32(umFaceLow(vec3i(31,5,6),0u));
 out[11]=umAcrossKind(vec3i(31,5,6),0u,1);out[12]=umPageKind(vec3i(-1,0,0));out[13]=umPageKind(vec3i(2,0,0));out[14]=umPageWord(page);
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
   const resultLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
   // The fixed world specialization, whose SOLID pages are those outside the box
   // (umSolidIsBox), in a solver with the slot store (umBulkSlots: tagged words exist).
   const constants={umFixedWorldGeometry:1,umBulkSlots:1,...Object.fromEntries(Array.from({length:8},(_,i)=>[`umWindowWord${24+i}`,tail[UNIFORM_DIRECTORY_WORDS+24+i]!]))};
   const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,resultLayout]}),compute:{module,entryPoint:"main",constants}});
   const group=device.createBindGroup({layout:resultLayout,entries:[{binding:0,resource:{buffer:output}}]});
   const e=device.createCommandEncoder();builder.encode(e);
   const p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,ownership.bindGroup);p.setBindGroup(1,group);p.dispatchWorkgroups(1);p.end();
   e.copyBufferToBuffer(output,0,read,0,64);e.copyBufferToBuffer(rows,0,read,64,4*nl.end);device.queue.submit([e.finish()]);
   await read.mapAsync(GPUMapMode.READ);const all=new Uint32Array(read.getMappedRange()).slice();read.unmap();
   const u=all.subarray(0,16),gpuRows=all.subarray(16);
   assert.deepEqual(Array.from(u.subarray(0,14)),[bulk16PageWord(5),UNIFORM_KIND.bulk,5,0,31,5,6,64,bulk16PageWord(5),1,0,UNIFORM_KIND.bulk,UNIFORM_KIND.solid,UNIFORM_KIND.resident]);assert.equal(u[14],blocks-1,"the page word folds the tag to blocks-1");
   // bnRows and bnWalls rebuild the host twin's rows and wall list.
   const host=uniformBlockNeighbourWords(tail,address.seed.blocks);
   const at=UNIFORM_PAGE_TABLE_WORDS+blocks;
   for(let b=1;b<blocks-1;b++){if(address.seed.blocks[b]===0xffffffff)continue;for(let i=0;i<27;i++)assert.equal(gpuRows[at+27*b+i],host[27*b+i],`block ${b} row word ${i}`);}
   const count=host[27*blocks]!;assert.equal(gpuRows[at+27*blocks],count);
   assert.deepEqual(Array.from(gpuRows.subarray(at+27*blocks+1,at+27*blocks+1+count)),Array.from(host.subarray(27*blocks+1,27*blocks+1+count)));
  }finally{read.destroy();output.destroy();rows.destroy();ownership.destroy();}
 });
});

// The regression the falling-slab column found (docs/plans/bulk16-plan-2026-10-28.md,
// M3): most of a frame's pipelines are compiled without uniformDetailPipeline's
// constants (the census's classify, seed and dilate, the ring's enter, the h
// slots' mark, the block neighbours' rows, the tile classifier, ...). Each
// reads the runtime header, so a tagged word there must still read as the
// BULK16 page it is, never as SOLID: a liquid page read as a wall by some
// stages and as liquid by others is what sent the column's samples nonfinite.
test("BULK16 page: a pipeline compiled without the fixed world constants reads the tag as BULK16, not SOLID",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("BULK16 generic readers",async device=>{
  const {address}=bulkBox(),dimensions=[96,32,32] as const;
  const layout=createUniformMixedLayoutFromWidths({dimensions,cellSize_m:[1,1,1],origin_m:{x:0,y:0,z:0}},new Uint8Array(address.boxTiles).fill(4),[],address);
  const ownership=new UniformMixedOwnership(device,layout,false,0,false);
  const output=device.createBuffer({size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   const module=device.createShaderModule({code:uniformMixedTopologyWGSL(ownership.capacity,0)+`
@group(1) @binding(0) var<storage,read_write> out:array<u32>;
@compute @workgroup_size(1) fn main(){
 let page=vec3i(1,0,0);let raw=umPageRawWord(page);
 out[0]=umPageKind(page);out[1]=select(0u,raw&0x3fffffffu,umBulkTag(raw));out[2]=u32(umCellPresent(vec3i(40,5,6)));
 out[3]=umAcrossKind(vec3i(31,5,6),0u,1);out[4]=umPageKind(vec3i(0,0,0));out[5]=umPageKind(vec3i(-1,0,0));
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
   const resultLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
   // No constants: the generic consumer's runtime header (no umFixedWorldGeometry, no umBulkSlots).
   const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,resultLayout]}),compute:{module,entryPoint:"main"}});
   const group=device.createBindGroup({layout:resultLayout,entries:[{binding:0,resource:{buffer:output}}]});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,ownership.bindGroup);p.setBindGroup(1,group);p.dispatchWorkgroups(1);p.end();
   e.copyBufferToBuffer(output,0,read,0,32);device.queue.submit([e.finish()]);
   await read.mapAsync(GPUMapMode.READ);const u=Array.from(new Uint32Array(read.getMappedRange()).subarray(0,6));read.unmap();
   assert.deepEqual(u,[UNIFORM_KIND.bulk,5,0,UNIFORM_KIND.bulk,UNIFORM_KIND.resident,UNIFORM_KIND.solid]);
  }finally{read.destroy();output.destroy();ownership.destroy();}
 });
});
