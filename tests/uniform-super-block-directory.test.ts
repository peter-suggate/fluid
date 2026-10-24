import assert from "node:assert/strict";
import test from "node:test";
import {USD_AIR,USD_AXIS_CELLS,USD_FATAL,USD_FREE,USD_HEADER,USD_KIND,USD_SOLID,USD_STATE,USD_TAIL_ROWS,UniformSuperBlockDirectory,uniformSuperBlockBlockWord,uniformSuperBlockCapacity,uniformSuperBlockChild,uniformSuperBlockEntry} from "../lib/methods/uniform/uniform-super-block-directory";
import {UBA_STRIDE} from "../lib/methods/uniform/uniform-brick-arena";
import {withUniformDevice} from "./helpers/uniform-geometric";

test("super-block directory: capacity from tiers or the census, capped at the pressure tail's rows; 8192 h an axis",()=>{
 assert.equal(uniformSuperBlockCapacity(0),8);
 assert.equal(uniformSuperBlockCapacity(100),150);
 assert.equal(uniformSuperBlockCapacity(10_000),USD_TAIL_ROWS);
 assert.equal(uniformSuperBlockCapacity(100,{superBlocks:33}),33);
 assert.throws(()=>uniformSuperBlockCapacity(1,{superBlocks:0}));
 assert.throws(()=>new UniformSuperBlockDirectory({capacity:USD_TAIL_ROWS+1,blocks:10,idleSteps:1}));
 assert.equal(USD_AXIS_CELLS,8192);
 assert.equal(uniformSuperBlockEntry(-1,64,65),63|(0<<6)|(1<<12));
 assert.equal(uniformSuperBlockChild(-1,6,9),3|(2<<2)|(1<<4));
 assert.equal(uniformSuperBlockBlockWord(-1,255,256),255|(255<<8));
 const d=new UniformSuperBlockDirectory({capacity:5,blocks:10,idleSteps:2,channels:[{name:"pressure",layout:"cell"},{name:"velocity",layout:"face",banks:2}]});
 assert.equal(d.base("pressure"),0);assert.equal(d.base("velocity",0,0),UBA_STRIDE*7);assert.equal(d.base("velocity",2,1),UBA_STRIDE*7*6);
 assert.equal(d.poolWords,UBA_STRIDE*7*7);assert.equal(d.allocatedBytes,4*(d.layout.words+d.poolWords));
 assert.throws(()=>d.base("phi"));
});

// Pages: z < 0 is SOLID, the rest AIR, unless seeded.
const other=(_x:number,_y:number,z:number)=>z<0?USD_KIND.solid:USD_KIND.air;
const SEED=[{x:1,y:2,z:3,block:1},{x:-1,y:0,z:0,block:2},{x:2,y:2,z:3,block:4}];

test("super-block directory seed: slots in first appearance, sentinels, page words, block words and the free stack",()=>{
 const d=new UniformSuperBlockDirectory({capacity:4,blocks:10,idleSteps:2});
 const s=d.seed(SEED,other,[{x:0,y:0,z:-1},{x:1,y:0,z:0},{x:0,y:-1,z:-1}]),l=d.layout;
 assert.equal(s[l.top+uniformSuperBlockEntry(0,0,0)],1);assert.equal(s[l.top+uniformSuperBlockEntry(-1,0,0)],2);
 assert.equal(s[l.top+uniformSuperBlockEntry(0,0,-1)],USD_SOLID);assert.equal(s[l.top+uniformSuperBlockEntry(1,0,0)],USD_AIR);
 assert.equal(s[l.top+uniformSuperBlockEntry(5,5,5)],USD_SOLID,"never named: SOLID");
 assert.equal(d.pageWord(s,1,2,3),1);assert.equal(d.pageWord(s,-1,0,0),2);assert.equal(d.pageWord(s,2,2,3),4);
 assert.equal(d.pageWord(s,0,0,0),USD_AIR);assert.equal(d.pageWord(s,0,0,-4),USD_SOLID);assert.equal(d.pageWord(s,4,0,0),USD_AIR);
 assert.equal(d.pageWord(s,20,20,20),USD_SOLID);
 // The mixed super-block (0,-1,-1)'s pages are all SOLID (z < 0): SOLID.
 assert.equal(s[l.top+uniformSuperBlockEntry(0,-1,-1)],USD_SOLID);
 assert.equal(s[l.blockWords+1],uniformSuperBlockBlockWord(1,2,3));assert.equal(s[l.blockWords+2],uniformSuperBlockBlockWord(-1,0,0));
 assert.equal(s[l.blockWords+3],USD_FREE);
 assert.equal(s[USD_HEADER.live],2);assert.deepEqual([...s.subarray(l.live,l.live+2)],[1,2]);
 assert.equal(s[USD_HEADER.free],2);assert.deepEqual([...s.subarray(l.free,l.free+2)],[4,3]);
 assert.equal(s[l.records+8*0+3],USD_STATE.sentinel);assert.equal(s[l.records+8*d.solidSlot+3],USD_STATE.sentinel);
 assert.ok(s.subarray(l.child+64*d.solidSlot,l.child+64*d.slots).every(w=>w===USD_SOLID));
 assert.ok(s.subarray(l.child,l.child+64).every(w=>w===USD_AIR));
 assert.equal(new Int32Array(s.buffer)[l.records+8*2],-1);
 // Mixed AIR/SOLID super-blocks named without a seeded page take a slot.
 const mixed=d.seed([],(_x,_y,z)=>z<1?USD_KIND.solid:USD_KIND.air,[{x:0,y:0,z:0}]);
 assert.equal(mixed[l.top],1);assert.equal(d.pageWord(mixed,0,0,0),USD_SOLID);assert.equal(d.pageWord(mixed,0,0,1),USD_AIR);
 assert.throws(()=>d.seed([{x:0,y:0,z:0,block:1},{x:256,y:0,z:0,block:2}],other),/aliases/);
 assert.throws(()=>d.seed([{x:0,y:0,z:0,block:1},{x:0,y:0,z:0,block:2}],other),/child word/);
 assert.throws(()=>d.seed([{x:0,y:0,z:0,block:1},{x:1,y:0,z:0,block:1}],other),/twice/);
 assert.throws(()=>d.seed([{x:0,y:0,z:0,block:9}],other),/outside/);
 assert.throws(()=>d.seed([0,4,8,12,16].map((x,i)=>({x,y:0,z:0,block:i+1})),other),/slots/);
});

/** One step: requests, the commit, then page writes ([x, y, z, block]; block 0 clears) and lookups. */
interface Step{requests:readonly (readonly number[])[];pages?:readonly (readonly number[])[]}
/** A free block's word as the test kernel repacks usdBlockPage (8 bits an axis). */
const FREE_PAGE=0xffffff;
const LOOKUPS=[[1,2,3],[-1,0,0],[9,1,1],[0,0,-4],[4,0,0],[0,0,0],[264,1,2],[13,0,0],[17,0,0]];

test("super-block directory: claims, keeps, duplicates, page writes, idle retirement, quarantine and fail-fast",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const d=new UniformSuperBlockDirectory({capacity:4,blocks:10,idleSteps:2,requestCapacity:16,channels:[{name:"p",layout:"cell"}]}),l=d.layout;
 const steps:Step[]=[
  // 1: a claim requested twice, a keep, a SOLID request (nothing); a page in the new slot, one in an AIR super-block (refused).
  {requests:[[2,0,0],[2,0,0],[0,0,0],[0,0,-1]],pages:[[9,1,1,3],[5,0,0,5]]},
  // 2-4: the page clears; the slot idles from the next commit and retires at the second.
  {requests:[],pages:[[9,1,1,0]]},{requests:[]},{requests:[]},
  // 5: the quarantined slot is free again: a super-block on the same entry takes it.
  {requests:[[66,0,0]],pages:[[264,1,2,3]]},
  // 6: the old super-block on that entry is an alias.
  {requests:[[2,0,0]]},
  // 7: two claims, one free slot: the exhausted fatal, the loser's entry AIR.
  {requests:[[3,0,0],[4,0,0]]},
 ];
 await withUniformDevice("super-block directory",async device=>{
  const code=`
@group(0) @binding(0) var<storage,read_write> usd:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read> commands:array<i32>;
@group(0) @binding(2) var<storage,read_write> out:array<u32>;
@group(0) @binding(3) var<storage,read_write> pool:array<u32>;
${d.wgsl("usd")}
${d.poolWGSL("pool")}
// commands: request count, page count, then requests (x, y, z, 0) and pages (x, y, z, block), then lookups (x, y, z, 0).
@compute @workgroup_size(64) fn request(@builtin(global_invocation_id) g:vec3u){if(g.x<u32(commands[0])){let c=2u+4u*g.x;usdRequest(vec3i(commands[c],commands[c+1u],commands[c+2u]));}}
@compute @workgroup_size(1) fn pages(){
 let n=u32(commands[0]);let m=u32(commands[1]);
 for(var i=0u;i<m;i++){
  let c=2u+4u*(n+i);let p=vec3i(commands[c],commands[c+1u],commands[c+2u]);let block=u32(commands[c+3u]);
  if(block==0u){usdClearPage(p);out[32u+i]=2u;}else{out[32u+i]=select(0u,1u,usdSetPage(p,block));}
 }
 for(var i=0u;i<${LOOKUPS.length}u;i++){let c=2u+4u*(n+m+i);out[i]=usdPageWord(vec3i(commands[c],commands[c+1u],commands[c+2u]));}
 for(var b=0u;b<10u;b++){let q=usdBlockPage(b);out[16u+b]=q.x|(q.y<<8u)|(q.z<<16u);}
 usdStore(USD_P_0_0,1u,511u,3.5);out[40u]=bitcast<u32>(usdLoad(USD_P_0_0,1u,511u));
}`;
  const module=device.createShaderModule({code});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  assert.deepEqual(errors.map(m=>`${m.lineNum}: ${m.message}`),[]);
  const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
  const state=device.createBuffer({size:4*l.words,usage}),commands=device.createBuffer({size:4*256,usage}),out=device.createBuffer({size:4*64,usage}),pool=device.createBuffer({size:4*d.poolWords,usage});
  // The super-blocks the steps claim start AIR (an entry never named is SOLID).
  device.queue.writeBuffer(state,0,d.seed(SEED,other,[{x:0,y:0,z:-1},{x:1,y:0,z:0},{x:2,y:0,z:0},{x:3,y:0,z:0},{x:4,y:0,z:0}]));
  const bgl=device.createBindGroupLayout({entries:[0,1,2,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:(binding===1?"read-only-storage":"storage") as GPUBufferBindingType}}))});
  const layout=device.createPipelineLayout({bindGroupLayouts:[bgl]});
  const pipelines=new Map<string,GPUComputePipeline>();
  for(const entry of ["request","pages",...d.commitEntries.map(c=>c.entry)])pipelines.set(entry,await device.createComputePipelineAsync({layout,compute:{module,entryPoint:entry}}));
  const bind=device.createBindGroup({layout:bgl,entries:[state,commands,out,pool].map((buffer,binding)=>({binding,resource:{buffer}}))});
  const read=async(buffer:GPUBuffer)=>{
   const staging=device.createBuffer({size:buffer.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   const e=device.createCommandEncoder();e.copyBufferToBuffer(buffer,0,staging,0,buffer.size);device.queue.submit([e.finish()]);
   await staging.mapAsync(GPUMapMode.READ);const data=new Uint32Array(staging.getMappedRange().slice(0));staging.unmap();staging.destroy();return data;
  };
  const results:{s:Uint32Array;o:Uint32Array}[]=[];
  for(const step of steps){
   const pages=step.pages??[];
   device.queue.writeBuffer(commands,0,new Int32Array([step.requests.length,pages.length,...step.requests.flatMap(r=>[...r,0]),...pages.flat(),...LOOKUPS.flatMap(p=>[...p,0])]));
   const e=device.createCommandEncoder();e.clearBuffer(out);const pass=e.beginComputePass();pass.setBindGroup(0,bind);
   pass.setPipeline(pipelines.get("request")!);pass.dispatchWorkgroups(1);
   for(const c of d.commitEntries){pass.setPipeline(pipelines.get(c.entry)!);pass.dispatchWorkgroups(c.workgroups);}
   pass.setPipeline(pipelines.get("pages")!);pass.dispatchWorkgroups(1);
   pass.end();device.queue.submit([e.finish()]);
   results.push({s:await read(state),o:await read(out)});
  }
  for(const b of [state,commands,out,pool])b.destroy();
  const H=USD_HEADER,top=(s:Uint32Array,x:number,y:number,z:number)=>s[l.top+uniformSuperBlockEntry(x,y,z)]!;
  const coord=(s:Uint32Array,slot:number)=>[...new Int32Array(s.buffer,4*(l.records+8*slot),3)];
  const live=(s:Uint32Array)=>[...s.subarray(l.live,l.live+s[H.live]!)].sort();
  // 1
  {const {s,o}=results[0]!;
   assert.equal(s[H.fatal],0);assert.equal(top(s,2,0,0),3);assert.deepEqual(coord(s,3),[2,0,0]);assert.equal(s[l.records+8*3+3],USD_STATE.live);
   assert.equal(s[H.claimed],1);assert.equal(s[l.claimed],3);assert.equal(s[H.free],1);assert.deepEqual(live(s),[1,2,3]);
   assert.equal(top(s,0,0,-1),USD_SOLID);assert.equal(s[H.requests],0);
   assert.deepEqual([...o.subarray(32,34)],[1,0]);
   assert.deepEqual([...o.subarray(0,LOOKUPS.length)],[1,2,3,USD_SOLID,USD_AIR,USD_AIR,USD_AIR,USD_AIR,USD_AIR]);
   assert.equal(o[16+3],uniformSuperBlockBlockWord(9,1,1));assert.equal(o[16+2],uniformSuperBlockBlockWord(-1,0,0));assert.equal(o[16+5],FREE_PAGE);
   assert.equal(new Float32Array(o.buffer)[40],3.5);}
  // 2: the page cleared, its block word free; the slot was stored at this commit: idle 0.
  {const {s,o}=results[1]!;
   assert.equal(o[2],USD_AIR);assert.equal(o[16+3],FREE_PAGE);assert.equal(s[l.records+8*3+4],0);assert.deepEqual(live(s),[1,2,3]);}
  // 3: idle 1. 4: retired, its entry AIR, its slot quarantined (not yet free).
  {assert.equal(results[2]!.s[l.records+8*3+4],1);
   const {s}=results[3]!;
   assert.equal(top(s,2,0,0),USD_AIR);assert.equal(s[l.records+8*3+3],USD_STATE.pending);assert.equal(s[H.pending],1);assert.equal(s[H.free],1);
   assert.equal(s[H.retired],1);assert.deepEqual(live(s),[1,2]);}
  // 5: recycled and claimed again by the super-block that shares the entry.
  {const {s,o}=results[4]!;
   assert.equal(s[H.fatal],0);assert.equal(top(s,66,0,0),3);assert.equal(uniformSuperBlockEntry(66,0,0),uniformSuperBlockEntry(2,0,0));
   assert.deepEqual(coord(s,3),[66,0,0]);assert.equal(s[H.pending],0);assert.equal(s[H.free],1);
   assert.equal(o[32],1);assert.equal(o[6],3);assert.equal(o[16+3],uniformSuperBlockBlockWord(264,1,2));
   assert.ok(s.subarray(l.child+64*3,l.child+64*4).every((w,c)=>w===(c===uniformSuperBlockChild(264,1,2)?3:USD_AIR)));}
  // 6: alias.
  {const {s}=results[5]!;assert.equal(s[H.fatal],USD_FATAL.alias);assert.equal(s[H.fatalEntry],uniformSuperBlockEntry(2,0,0));assert.equal(top(s,66,0,0),3);}
  // 7: exhausted; the winner holds slot 4, the loser's entry is AIR again; the stack is empty, never negative.
  {const {s}=results[6]!;
   assert.equal(s[H.fatal],USD_FATAL.alias|USD_FATAL.exhausted);
   const words=[top(s,3,0,0),top(s,4,0,0)].sort();assert.deepEqual(words,[USD_AIR,4]);
   assert.equal(s[H.free],0);assert.equal(s[H.claimed],1);assert.deepEqual(live(s),[1,2,3,4]);}
 });
});
