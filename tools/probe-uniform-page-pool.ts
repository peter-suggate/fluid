/** Dawn proof of the GPU page pool (lib/methods/uniform/uniform-page-pool.ts).
 * Every commit is read back whole, here only, and audited: the table holds
 * AIR, SOLID or a distinct resident block whose page masks to the entry, the
 * SOLID entries are exactly the seeded ones, every block word is its
 * resident page's window coordinate or FREE, the resident list is the set of
 * table blocks, the ring's free and quarantined runs and the residents
 * partition the slots, the claim words and keeps are clear, each mirror's
 * tail equals the state's with the words around it untouched, and the WGSL
 * readers agree with the buffers.
 * Small pools (table 8, 18 blocks, idle 2) cover host commands and a kernel
 * of 8192 invocations asking for a few pages at once, negative, huge and
 * wrapped coordinates, retirement after two idle commits, the one-commit
 * quarantine, exhaustion, aliases against a resident page and within one
 * step, a request list overflow and an entry turned SOLID between request
 * and claim, after which the audit is shown to catch a changed mirror word;
 * a 2050-block pool runs the strided launches and the many-block lanes, and
 * one of 65536 blocks (the most the layout takes) is seeded full, emptied
 * and partly refilled. Seeded pools of the letters shape (table 64, 208 blocks): the 192
 * box pages in scan order with every other entry SOLID, read back against
 * the host copy and asked toward SOLID; the same pages in permuted blocks
 * with free holes; and a half box with AIR above it under random requests
 * and keeps, checked against a host model of the resident set.
 * Live allocation inside a box, each scenario against a host model of its
 * script: a list of 16-cell pages of the box with a reach, each asked for by
 * its 32-cell page (the probe's own upCertify) on a grid shorter than the list.
 * The letters box kept whole for twenty commits; a block of seeds walking a
 * path through a box four times the pool, claiming ahead, retiring behind
 * after idleSteps (3, then 1), reusing blocks and reclaiming pages, from an
 * identity seed and from a permuted one with holes, which must agree commit
 * for commit; the same path one block short, exhausted at the commit the
 * model names; pinned pages that never age; an exhaustion, a reach over the
 * cap and a seeded page that is not resident, each with its page and its
 * message. With a box the audit also compares the published record with the
 * table after every commit, and is shown to catch a changed flag.
 * --rounds=N repeats the unseeded scenarios and the random one on fresh
 * pools (the requesting invocations race, so the blocks differ from run to
 * run).
 * node --import tsx tools/probe-uniform-page-pool.ts
 */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { UniformPagePool,uniformPagePoolBoxSeed,uniformPagePoolFatalMessage,uniformPagePoolLayout,uniformPagePoolPublished,uniformPagePoolSeed,uniformPagePoolWGSL,
 UNIFORM_PAGE_POOL_BLOCK as BLOCK,UNIFORM_PAGE_POOL_FATAL as FATAL,UNIFORM_PAGE_POOL_FREE as FREE,UNIFORM_PAGE_POOL_KIND as KIND,UNIFORM_PAGE_POOL_NONE as NONE,UNIFORM_PAGE_POOL_PUBLISHED as PUBLISHED,
 UNIFORM_PAGE_POOL_RECORD as RECORD,UNIFORM_PAGE_POOL_SOLID as SOLID,UNIFORM_PAGE_POOL_STATUS as STATUS,UNIFORM_PAGE_POOL_WORK as WORK,
 type UniformPagePoolBoxSeedOptions,type UniformPagePoolOptions,type UniformPagePoolSeedPage } from "../lib/methods/uniform/uniform-page-pool";

type Page=readonly [number,number,number];
/** A residency certificate: resident pages of 16 cells by box coordinate, each seeded (1) or margin (0), and the reach. */
interface Certificate {readonly pages:readonly (readonly [number,number,number,number])[];readonly reach:number}
const ASKS=32,KEEPS=32,LOOKS=64,GROUPS=128,CERTS=2048,CERT_GROUPS=4;
/** Script words: the header, the pages asked for, the blocks kept, the pages looked up, then the certificate (its reach, four words an entry). */
const SCRIPT_KEEPS=4+4*ASKS,SCRIPT_LOOKS=SCRIPT_KEEPS+KEEPS,SCRIPT_CERT=SCRIPT_LOOKS+4*LOOKS,SCRIPT_WORDS=SCRIPT_CERT+1+4*CERTS;
const key=(p:Page)=>p.join(",");
const errors:string[]=[],started=performance.now();
const rounds=+(process.argv.find(a=>a.startsWith("--rounds="))?.slice(9)??1);
/** First index at which two word arrays differ, -1 when they are equal. */
const differ=(a:Uint32Array,b:Uint32Array)=>{if(a.length!==b.length)return Math.min(a.length,b.length);for(let i=0;i<a.length;i++)if(a[i]!==b[i])return i;return -1;};
const random=(seed:number)=>()=>{seed=(seed+0x6d2b79f5)|0;let t=Math.imul(seed^(seed>>>15),1|seed);t=(t+Math.imul(t^(t>>>7),61|t))^t;return ((t^(t>>>14))>>>0)/4294967296;};
const shuffled=<T>(items:readonly T[],rand:()=>number)=>{const a=[...items];for(let i=a.length-1;i>0;i--){const j=Math.floor(rand()*(i+1));[a[i],a[j]]=[a[j]!,a[i]!];}return a;};

// The host seed and the layout reject what they cannot hold, before any GPU work.
{
 const l=uniformPagePoolLayout({table:8,blocks:18,idleSteps:2}),solid=()=>KIND.solid,page=(x:number,y:number,z:number,block:number):UniformPagePoolSeedPage=>({x,y,z,block});
 assert.throws(()=>uniformPagePoolSeed(l,[page(0,0,0,1),page(1,0,0,1)],solid),/block 1 already holds page \(0, 0, 0\)/);
 assert.throws(()=>uniformPagePoolSeed(l,[page(0,0,0,1),page(0,0,0,2)],solid),/listed twice/);
 assert.throws(()=>uniformPagePoolSeed(l,[page(3,0,0,1),page(-5,8,-16,2)],solid),/aliases page \(3, 0, 0\) of block 1/);
 for(const block of [0,17,18,-1,1.5])assert.throws(()=>uniformPagePoolSeed(l,[page(0,0,0,block)],solid),/outside 1\.\.16/);
 assert.throws(()=>uniformPagePoolSeed(l,[page(0.5,0,0,1)],solid),/32-bit integers/);
 assert.throws(()=>uniformPagePoolSeed(l,[],()=>KIND.resident),/must be classified/);
 assert.throws(()=>uniformPagePoolLayout({table:128,blocks:18,idleSteps:1}),/2\.\.64/);assert.throws(()=>uniformPagePoolLayout({blocks:2,idleSteps:1}),/3\.\.65536/);
 assert.deepEqual([3,18,208,2050].map(blocks=>uniformPagePoolLayout({blocks,idleSteps:1}).ringCapacity),[1,16,256,2048]);
}
const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
const adapter=await createProcessRetainedDawnGPU(dawn,["backend=metal"]).requestAdapter();assert.ok(adapter);
const device=managedGPUDevice(await adapter.requestDevice(),{requireWorkerRealm:false});
device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});

interface Snapshot {
 generation:number;free:number[];pending:number[];claimed:number[];fatal:number;fatalPage:Page;fatalGeneration:number;fatalFirst:number;fatalDetail:number;requests:number;
 /** uniformPagePoolFatalMessage of the receipt. */
 message:string|null;
 /** The published record as read back (empty without a box), and the receipt. */
 published:Uint32Array;status:Uint32Array;
 /** Resident page key -> block, and -> idle commits. */
 resident:Map<string,number>;idle:Map<string,number>;
 /** upBlockOf, upBlockWrapped and upKind of every page looked up. */
 looked:Map<string,readonly [number,number,number]>;
 /** The buffers as read back. */
 state:Uint32Array;work:Uint32Array;
}
interface Setup {
 /** The resident set to seed, and the kind of every other entry (AIR when omitted). */
 pages?:readonly UniformPagePoolSeedPage[];other?:(x:number,y:number,z:number)=>number;
 /** A resident set of the layout's box instead (uniformPagePoolBoxSeed). */
 box?:UniformPagePoolBoxSeedOptions;
 /** Pages pinned for the pool's life. */
 pins?:readonly Page[];
 /** Seed and pin after initialize, so seed() writes the mirrors; else before, so initialize does. */
 late?:boolean;
 /** Mirror buffers (default 2). */
 mirrors?:number;
}
async function harness(options:UniformPagePoolOptions,setup:Setup={}){
 const pool=new UniformPagePool(device,options);
 const l=pool.layout,mask=l.table-1,bits=l.tableBits;
 const entryOf=(p:Page)=>((p[0]&mask)|((p[1]&mask)<<bits)|((p[2]&mask)<<(2*bits)))>>>0;
 const blockWordOf=(p:Page)=>((p[0]&mask)|((p[1]&mask)<<6)|((p[2]&mask)<<12))>>>0;
 // Each mirror at its own offset inside a guard pattern.
 const guard=(i:number)=>(0xa5000000|i)>>>0;
 const mirrors=[5,1031,64,333].slice(0,setup.mirrors??2).map(offset=>{
  const words=offset+l.tail.words+9,buffer=device.createBuffer({size:4*words,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(buffer,0,Uint32Array.from({length:words},(_,i)=>guard(i)));return {buffer,offset};
 });
 const seeding=()=>{
  // Pinned ahead of the seed when early, so the seed's upload carries the pins; after it when late, so pin() writes them.
  const pin=()=>{if(setup.pins)pool.pin(setup.pins.map(([x,y,z])=>({x,y,z})));};
  if(!setup.late)pin();
  if(setup.pages)pool.seed(uniformPagePoolSeed(l,setup.pages,setup.other??(()=>KIND.air)));
  if(setup.box)pool.seed(uniformPagePoolBoxSeed(l,setup.box));
  if(setup.late)pin();
 };
 if(!setup.late)seeding();
 await pool.initialize(mirrors);
 await assert.rejects(pool.initialize(mirrors),/already initialized/);
 if(setup.late)seeding();
 const seed=pool.seeded;
 /** The SOLID entries: the seeded ones and those the probe wrote between request and claim. */
 const solid=new Set<number>();seed.table.forEach((word,e)=>{if(word===SOLID)solid.add(e);});
 const module=device.createShaderModule({label:"Page pool probe consumer",code:/* wgsl */`${uniformPagePoolWGSL(l,0,0,1)}
// asks, keeps, looks, certificate entries, then the pages asked for, the blocks kept, the pages looked up and the certificate.
@group(1) @binding(0) var<storage,read> script:array<i32>;
@group(1) @binding(1) var<storage,read_write> seen:array<u32>;
@compute @workgroup_size(64) fn drive(@builtin(global_invocation_id) gid:vec3u){
 let asks=u32(script[0]);let keeps=u32(script[1]);
 if(asks>0u){let a=4u+4u*(gid.x%asks);upRequest(vec3i(script[a],script[a+1u],script[a+2u]));}
 if(keeps>0u){upKeep(u32(script[${SCRIPT_KEEPS}u+gid.x%keeps]));}
}
// A certificate's list: a count, 16-cell pages relative to the box, a seeded flag each, and a reach.
fn certCount()->u32{return u32(script[3]);}
fn certPage(i:u32)->vec3i{let a=${SCRIPT_CERT+1}u+4u*i;return vec3i(script[a],script[a+1u],script[a+2u]);}
fn certSeeded(i:u32)->bool{return script[${SCRIPT_CERT+4}u+4u*i]!=0;}
// Entries lane, lane+width, ... ask for their 32-cell pages. A reach over the cap asks for nothing; a seeded
// entry whose page is not resident yet raises closure and is asked for all the same.
fn upCertify(lane:u32,width:u32){
 let count=certCount();let reach=u32(script[${SCRIPT_CERT}u]);
 for(var i=lane;i<count;i+=width){
  let q=vec3i(${l.box.origin.join(",")})+(certPage(i)>>vec3u(1u));
  if(reach>UP_CLOSURE_CAP){upFailWith(UP_FATAL_CLOSURE_CAP,q,reach);return;}
  if(certSeeded(i)&&upBlockOf(q)==UP_NONE){upFail(UP_FATAL_CLOSURE,q);}
  upRequest(q);
 }
}
// A fixed grid shorter than the list: each lane strides it.
@compute @workgroup_size(64) fn certify(@builtin(global_invocation_id) gid:vec3u){upCertify(gid.x,${64*CERT_GROUPS}u);}
@compute @workgroup_size(1) fn look(){
 seen[0]=upGeneration();seen[1]=upResidentCount();seen[2]=upNewCount();seen[3]=upFatal();
 var bad=0u;
 for(var i=0u;i<upResidentCount();i++){
  let block=upResidentBlock(i);let q=upPageOf(block);
  if(!upStored(block)||upBlockOf(q)!=block||upBlockWrapped(q)!=block||upWord(q)!=block||upKind(q)!=UP_RESIDENT){bad+=1u;}
  if(any(upWindowPage(upBlockWord(block))!=(q&vec3i(i32(UP_TABLE)-1)))){bad+=1u;}
 }
 for(var i=0u;i<upNewCount();i++){let block=upNewBlock(i);if(!upStored(block)||upBlockOf(upPageOf(block))!=block||upBlockOf(upNewPage(i))!=block){bad+=1u;}}
 if(upBlockWord(0u)!=UP_FREE||upBlockWord(UP_SOLID_BLOCK)!=UP_FREE||upStored(0u)||upStored(UP_SOLID_BLOCK)||upStored(UP_NONE)){bad+=1u;}
 seen[4]=bad;seen[5]=UP_EDGE;seen[6]=UP_BLOCKS;seen[7]=UP_TABLE;
 let looks=u32(script[2]);
 for(var i=0u;i<looks;i++){let a=${SCRIPT_LOOKS}u+4u*i;let q=vec3i(script[a],script[a+1u],script[a+2u]);seen[8u+3u*i]=upBlockOf(q);seen[9u+3u*i]=upBlockWrapped(q);seen[10u+3u*i]=upKind(q);}
}`});
 assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
 const poolLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
 const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[poolLayout,poolLayout]});
 const [drive,look,certify]=await Promise.all(["drive","look","certify"].map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
 const words=new Int32Array(SCRIPT_WORDS),seenWords=8+3*LOOKS;
 const script=device.createBuffer({size:4*words.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
 const seen=device.createBuffer({size:4*seenWords,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
 assert.equal(pool.published?.size??0,4*l.published.words);assert.equal(pool.volumePageSource?.records.buffer,pool.published);
 const sources=[pool.state,pool.work,seen,...mirrors.map(m=>m.buffer),...(pool.published?[pool.published]:[])];
 const reads=[...sources.map(b=>b.size),4*STATUS.words].map(size=>device.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}));
 const poolGroup=device.createBindGroup({layout:poolLayout,entries:[{binding:0,resource:{buffer:pool.state}},{binding:1,resource:{buffer:pool.work}}]});
 const scriptGroup=device.createBindGroup({layout:poolLayout,entries:[{binding:0,resource:{buffer:script}},{binding:1,resource:{buffer:seen}}]});
 /** The first pages the steps named: each is looked up through the WGSL readers after every commit. */
 const known=new Map<string,Page>();
 let commits=0;
 /** One step: the kernel asks and keeps, the host commands are queued, the pool commits; then the audit.
  * commit false: no launch but the audit (the pool as seeded). between: run on the host after the
  * kernel's requests are submitted and before the commit is encoded. */
 const step=async(input:{host?:readonly Page[];kernel?:readonly Page[];keep?:readonly number[];look?:readonly Page[];certificate?:Certificate;commit?:boolean;between?:()=>void}={}):Promise<Snapshot>=>{
  const {host=[],kernel=[],keep=[],look:extra=[],certificate,commit=true,between}=input;
  for(const p of [...extra,...kernel,...host])if(known.size<LOOKS)known.set(key(p),p);
  const looks=[...known.values()];
  assert.ok(kernel.length<=ASKS&&keep.length<=KEEPS&&looks.length<=LOOKS&&(certificate?.pages.length??0)<=CERTS&&(commit||!(host.length||kernel.length||keep.length||certificate)));
  words.fill(0);words.set([kernel.length,keep.length,looks.length,certificate?.pages.length??0]);
  kernel.forEach((p,i)=>words.set(p,4+4*i));words.set(keep,SCRIPT_KEEPS);looks.forEach((p,i)=>words.set(p,SCRIPT_LOOKS+4*i));
  if(certificate){words[SCRIPT_CERT]=certificate.reach;certificate.pages.forEach((p,i)=>words.set(p,SCRIPT_CERT+1+4*i));}
  device.queue.writeBuffer(script,0,words);
  for(const p of host)pool.request(...p);
  let encoder=device.createCommandEncoder();
  if(commit){
   const pass=encoder.beginComputePass();pass.setPipeline(drive);pass.setBindGroup(0,poolGroup);pass.setBindGroup(1,scriptGroup);pass.dispatchWorkgroups(GROUPS);
   if(certificate){pass.setPipeline(certify);pass.dispatchWorkgroups(CERT_GROUPS);}
   pass.end();
   if(between){device.queue.submit([encoder.finish()]);between();encoder=device.createCommandEncoder();}
   pool.encodeCommit(encoder);commits++;
  }
  const pass=encoder.beginComputePass();pass.setPipeline(look);pass.setBindGroup(0,poolGroup);pass.setBindGroup(1,scriptGroup);pass.dispatchWorkgroups(1);pass.end();
  sources.forEach((b,i)=>encoder.copyBufferToBuffer(b,0,reads[i]!,0,b.size));pool.encodeReceipt(encoder,reads[sources.length]!,0);
  device.queue.submit([encoder.finish()]);
  const copies=await Promise.all(reads.map(async buffer=>{await buffer.mapAsync(GPUMapMode.READ);const copy=new Uint32Array(buffer.getMappedRange()).slice();buffer.unmap();return copy;}));
  const [state,work,saw]=copies as [Uint32Array,Uint32Array,Uint32Array],receipt=copies[sources.length]!;
  assert.deepEqual(errors,[]);
  const at=commit?`commit ${commits}`:`seed`,signed=new Int32Array(state.buffer);
  const record=(block:number)=>{const r=l.state.records+RECORD.words*block;return {page:[signed[r]!,signed[r+1]!,signed[r+2]!] as Page,state:state[r+RECORD.state]!,generation:state[r+RECORD.generation]!,idle:state[r+RECORD.idle]!};};
  // The receipt is the status; the generation counts commits.
  assert.deepEqual([...receipt],[...state.subarray(0,STATUS.words)],at);
  const generation=state[STATUS.generation]!;assert.equal(generation,commits,at);
  // Table entries: AIR, a seeded SOLID, or a distinct resident block whose page masks to the entry.
  const tabled=new Set<number>();let solids=0;
  for(let e=0;e<l.entries;e++){
   const word=state[l.state.table+e]!;if(!word)continue;
   if(word===SOLID){solids++;if(!solid.has(e))assert.fail(`${at}: entry ${e} became SOLID`);continue;}
   if(solid.has(e)||word>l.blocks-2||tabled.has(word))assert.fail(`${at}: entry ${e} holds block ${word}`);
   tabled.add(word);
   const r=record(word);if(r.state!==BLOCK.resident||entryOf(r.page)!==e)assert.fail(`${at}: entry ${e} holds block ${word} of page ${r.page} in state ${r.state}`);
  }
  assert.equal(solids,solid.size,`${at}: SOLID entries`);
  // Records, resident list and table agree; the list is ascending; the sentinels never change.
  const slots=Array.from({length:l.slots},(_,i)=>i+1);
  const residents=slots.filter(block=>record(block).state===BLOCK.resident);
  assert.deepEqual([record(0).state,record(l.solidBlock).state],[BLOCK.sentinel,BLOCK.sentinel],at);
  assert.deepEqual(residents,[...tabled].sort((a,b)=>a-b),at);
  assert.equal(state[STATUS.resident],residents.length,at);
  assert.deepEqual([...state.subarray(l.state.resident,l.state.resident+residents.length)],residents,at);
  // Block words: a resident block's window page, FREE for every other block and both sentinels.
  for(let block=0;block<l.blocks;block++){
   const r=record(block);assert.equal(state[l.state.blocks+block],r.state===BLOCK.resident?blockWordOf(r.page):FREE,`${at}: block word ${block}`);
  }
  // Each mirror holds the state's tail at its offset and nothing else of it changed.
  const tail=state.subarray(l.state.table,l.state.table+l.tail.words);
  mirrors.forEach((m,i)=>{
   const got=copies[3+i]!;
   assert.equal(differ(got.subarray(m.offset,m.offset+l.tail.words),tail),-1,`${at}: mirror ${i} tail`);
   for(let w=0;w<got.length;w++)if((w<m.offset||w>=m.offset+l.tail.words)&&got[w]!==guard(w))assert.fail(`${at}: mirror ${i} word ${w} was written`);
  });
  // The new list is exactly the blocks claimed by this generation.
  const claimed=[...state.subarray(l.state.claimed,l.state.claimed+state[STATUS.claimed]!)];
  assert.deepEqual([...claimed].sort((a,b)=>a-b),commit?residents.filter(block=>record(block).generation===generation):[],at);
  for(const block of claimed)assert.equal(record(block).idle,0,at);
  // The ring: free blocks from the head, then the quarantined; with the residents, every slot once.
  const head=work[WORK.head]!,ring=(i:number)=>work[l.work.ring+((head+i)&(l.ringCapacity-1))]!;
  const free=Array.from({length:state[STATUS.free]!},(_,i)=>ring(i)),pending=Array.from({length:state[STATUS.pending]!},(_,i)=>ring(free.length+i));
  for(const block of free)assert.equal(record(block).state,BLOCK.free,`${at}: free block ${block}`);
  for(const block of pending)assert.equal(record(block).state,BLOCK.pending,`${at}: quarantined block ${block}`);
  assert.deepEqual([...free,...pending,...residents].sort((a,b)=>a-b),slots,at);
  // The commit left no request, keep or claim behind but the pin list, and the status carries the sticky bits.
  assert.deepEqual([work[WORK.requests],work[WORK.host],work[WORK.pops],work[WORK.pins]],[0,0,0,pool.pinned],at);
  (setup.pins??[]).forEach((p,i)=>assert.deepEqual([...new Int32Array(work.buffer,4*(l.work.pins+4*i),3)],[...p],`${at}: pin ${i}`));
  assert.ok(work.subarray(l.work.keep,l.work.keep+l.blocks).every(w=>w===0),`${at}: keeps`);
  assert.ok(work.subarray(l.work.claims,l.work.claims+3*l.entries).every(w=>w===0),`${at}: claim words`);
  assert.equal(state[STATUS.fatal],work[WORK.flags],at);
  // The WGSL readers.
  assert.deepEqual([...saw.subarray(0,8)],[generation,residents.length,claimed.length,state[STATUS.fatal],0,32,l.blocks,l.table],at);
  const resident=new Map(residents.map(block=>[key(record(block).page),block])),idle=new Map(residents.map(block=>[key(record(block).page),record(block).idle]));
  assert.equal(resident.size,residents.length,at);
  const looked=new Map<string,readonly [number,number,number]>();
  looks.forEach((p,i)=>{
   const word=state[l.state.table+entryOf(p)]!,got=[saw[8+3*i]!,saw[9+3*i]!,saw[10+3*i]!] as const;looked.set(key(p),got);
   assert.deepEqual(got,[resident.get(key(p))??NONE,Math.min(word,l.solidBlock),word===0?KIND.air:word===SOLID?KIND.solid:KIND.resident],`${at}: upBlockOf, upBlockWrapped, upKind of ${p}`);
  });
  // The published record: the header, the resident count and one flag a box page, from this generation's table.
  const published=pool.published?copies[3+mirrors.length]!:new Uint32Array(0);
  if(pool.published){
   const {origin,pages:[px,py,pz]}=l.box,flags:number[]=[];
   for(let z=0;z<pz;z++)for(let y=0;y<py;y++)for(let x=0;x<px;x++)flags.push(+resident.has(key([origin[0]+x,origin[1]+y,origin[2]+z])));
   assert.equal(differ(published,Uint32Array.from([32,px,py,pz,residents.length,0,0,0,...flags])),-1,`${at}: published record`);
   assert.equal(PUBLISHED.flags,8);
  }
  // The fatal message is null exactly while no bit is set.
  const message=uniformPagePoolFatalMessage(l,receipt);assert.equal(message===null,state[STATUS.fatal]===0,at);
  return {generation,free,pending,claimed,fatal:state[STATUS.fatal]!,fatalPage:[signed[STATUS.fatalPage]!,signed[STATUS.fatalPage+1]!,signed[STATUS.fatalPage+2]!],
   fatalGeneration:state[STATUS.fatalGeneration]!,fatalFirst:state[STATUS.fatalFirst]!,fatalDetail:state[STATUS.fatalDetail]!,requests:state[STATUS.requests]!,message,published,status:receipt,resident,idle,looked,state,work};
 };
 /** What a classification launch between request and claim would do: p's entry becomes SOLID, in the state and in every mirror. */
 const classify=(p:Page)=>{
  const e=entryOf(p),word=new Uint32Array([SOLID]);solid.add(e);
  device.queue.writeBuffer(pool.state,4*(l.state.table+e),word);for(const m of mirrors)device.queue.writeBuffer(m.buffer,4*(m.offset+l.tail.table+e),word);
 };
 /** Change one word of a mirror behind the pool's back: the next audit must reject it. */
 const poke=(mirror:number,word:number)=>device.queue.writeBuffer(mirrors[mirror]!.buffer,4*word,new Uint32Array([0x12345678]));
 const distinct=(pages:readonly Page[])=>assert.equal(new Set(pages.map(entryOf)).size,pages.length,"probe pages must not alias");
 /** Change one flag of the published record behind the pool's back: the next audit without a commit must reject it. */
 const smudge=(page:number)=>device.queue.writeBuffer(pool.published!,4*(l.published.flags+page),new Uint32Array([7]));
 return {pool,seed,step,classify,poke,smudge,mirrors,distinct,entryOf,destroy:()=>{pool.destroy();script.destroy();seen.destroy();for(const b of reads)b.destroy();for(const m of mirrors)m.buffer.destroy();}};
}
const keys=(pages:readonly Page[])=>pages.map(key).sort();
const ascending=(blocks:readonly number[])=>[...blocks].sort((a,b)=>a-b);
const blocksOf=(s:Snapshot,pages:readonly Page[])=>pages.map(p=>{const block=s.resident.get(key(p));assert.ok(block!==undefined,`page ${p} must be resident`);return block;});
/** Every page resident in `before` holds the same block in `after`. */
const intact=(before:Snapshot,after:Snapshot,pages:readonly Page[])=>assert.deepEqual(blocksOf(after,pages),blocksOf(before,pages));
const small={table:8,blocks:18,idleSteps:2,hostCapacity:8} as const;

// A mirror must fit its buffer, be writable storage and be bound once.
{
 const pool=new UniformPagePool(device,small),l=pool.layout,usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST;
 const fits=device.createBuffer({size:4*(3+l.tail.words),usage}),plain=device.createBuffer({size:4*l.tail.words,usage:GPUBufferUsage.COPY_DST});
 await assert.rejects(pool.initialize([{buffer:fits,offset:4}]),/do not fit its buffer/);
 await assert.rejects(pool.initialize([{buffer:plain,offset:0}]),/needs STORAGE and COPY_DST/);
 await assert.rejects(pool.initialize([{buffer:fits,offset:0},{buffer:fits,offset:3}]),/bound to the commit already/);
 await assert.rejects(pool.initialize([{buffer:pool.state,offset:0}]),/bound to the commit already/);
 await assert.rejects(pool.initialize(Array.from({length:5},()=>({buffer:fits,offset:0}))),/mirrors at most/);
 assert.throws(()=>pool.encodeCommit(device.createCommandEncoder()),/not initialized/);
 pool.destroy();fits.destroy();plain.destroy();
}
// Lifecycle: claims from both sources, keeps, retirement, quarantine, exhaustion, recovery of the blocks.
for(let round=0;round<rounds;round++){
 const h=await harness(small);
 const A:Page[]=[[0,0,0],[-1,-1,-1],[1<<26,-(1<<26),12345],[2147483647,-2147483648,-5]],B:Page[]=[[3,4,5],[-9,2,1],[100,-200,300]];
 const C:Page[]=Array.from({length:9},(_,i)=>[i%3-8*i,17+Math.floor(i/3),6-8*i] as const),D:Page[]=[A[2]!,[5,-5,8005]],E:Page[]=[[6,6,6],[6,6,-793],[-3,6,86],[6,-2,0]];
 const F:Page[]=Array.from({length:16},(_,i)=>[i%8+8*(i-8),(i>>3)-16*i,-8*i] as const);
 h.distinct([...A,...B,...C]);h.distinct([...A.slice(0,2),...B.slice(0,2),...C,...D,...E]);h.distinct(F);
 // 0: the pool as constructed: every entry AIR, every block free in ascending order.
 const s0=await h.step({commit:false,look:[[8,0,0],[1,0,0]]});
 assert.deepEqual([s0.resident.size,s0.free,s0.pending,s0.fatal],[0,Array.from({length:16},(_,i)=>i+1),[],0]);assert.equal(differ(s0.state,h.seed.state),-1);
 // 1: four host pages, three kernel pages, and two asked for by both.
 const s1=await h.step({host:A,kernel:[...B,A[0]!,A[1]!]});
 assert.deepEqual([...s1.resident.keys()].sort(),keys([...A,...B]));
 assert.deepEqual([s1.claimed.length,s1.free.length,s1.pending.length,s1.fatal,s1.requests],[7,9,0,0,7]);
 assert.deepEqual(ascending(s1.claimed),[1,2,3,4,5,6,7]);
 // 2: two kept by block, two by asking again (kernel and host); the other three age. Keeping a sentinel or nothing does nothing.
 const kept=[A[0]!,B[0]!,A[1]!,B[1]!],aged=[A[2]!,A[3]!,B[2]!],keep=[...blocksOf(s1,[A[0]!,B[0]!]),0,17,-1];
 const s2=await h.step({kernel:[A[1]!],host:[B[1]!],keep});
 intact(s1,s2,[...A,...B]);assert.deepEqual([s2.resident.size,s2.claimed.length,s2.requests,s2.free.length],[7,0,0,9]);
 assert.deepEqual([...kept,...aged].map(p=>s2.idle.get(key(p))),[0,0,0,0,1,1,1]);
 // 3: the three retire while nine pages claim exactly the nine free blocks: none of the retired is reissued.
 const retired=blocksOf(s2,aged);
 const s3=await h.step({kernel:[A[1]!,...C.slice(0,5)],host:[B[1]!,...C.slice(5)],keep});
 intact(s2,s3,kept);assert.deepEqual([...s3.resident.keys()].sort(),keys([...kept,...C]));
 assert.deepEqual(ascending(s3.claimed),ascending(s2.free));
 assert.deepEqual(ascending(s3.pending),ascending(retired));
 assert.deepEqual([s3.free.length,s3.fatal,s3.requests],[0,0,9]);
 // 4: the next commit reissues the quarantined blocks, oldest first; one goes to a page that was retired.
 const s4=await h.step({kernel:D,keep:[...s3.resident.values()]});
 intact(s3,s4,[...kept,...C]);assert.deepEqual(ascending(s4.claimed),ascending(s3.pending.slice(0,2)));
 assert.deepEqual([s4.resident.size,s4.free.length,s4.pending.length,s4.fatal],[15,1,0,0]);assert.deepEqual(s4.free,s3.pending.slice(2));
 // 5: four pages for one block: exhausted, one coordinate named, every resident page intact.
 const s5=await h.step({kernel:E.slice(0,2),host:E.slice(2),keep:[...s4.resident.values()]});
 intact(s4,s5,[...kept,...C,...D]);
 assert.deepEqual([s5.fatal,s5.fatalFirst,s5.fatalGeneration,s5.resident.size,s5.claimed.length,s5.free.length,s5.requests],[FATAL.exhausted,FATAL.exhausted,5,16,1,0,4]);
 assert.deepEqual(s5.claimed,s4.free);
 assert.ok(keys(E).includes(key(s5.fatalPage))&&!s5.resident.has(key(s5.fatalPage)),`exhaustion names ${s5.fatalPage}`);
 assert.equal(E.filter(p=>s5.resident.has(key(p))).length,1);
 // 6, 7: nothing kept: every page ages, then retires; the fatal stays as first raised.
 const s6=await h.step();
 intact(s5,s6,[...kept,...C,...D]);assert.ok([...s6.idle.values()].every(idle=>idle===1));
 const s7=await h.step();
 assert.deepEqual([s7.resident.size,s7.pending.length,s7.free.length],[0,16,0]);
 // 8: sixteen pages take the whole pool back.
 const s8=await h.step({kernel:F.slice(0,8),host:F.slice(8)});
 assert.deepEqual([...s8.resident.keys()].sort(),keys(F));assert.deepEqual([s8.claimed.length,s8.free.length,s8.pending.length],[16,0,0]);
 for(const s of [s6,s7,s8])assert.deepEqual([s.fatal,s.fatalFirst,s.fatalGeneration,key(s.fatalPage)],[FATAL.exhausted,FATAL.exhausted,5,key(s5.fatalPage)]);
 if(!round)console.log(JSON.stringify({scenario:"lifecycle",commits:s8.generation,retiredAt3:retired,reissuedAt4:s4.claimed,exhaustedPage:s5.fatalPage,fatal:s8.fatal}));
 h.destroy();
}
// A page one table away from a resident page; then three pages of one entry in one step.
for(let round=0;round<rounds;round++){
 const h=await harness(small);
 const q:Page=[1,2,3],r:Page=[4,4,4],far:Page=[9,2,3],trio:Page[]=[[5,5,5],[5,13,5],[-3,5,5]];
 const s1=await h.step({host:[q,r]});assert.deepEqual([s1.resident.size,s1.fatal],[2,0]);
 const keep=blocksOf(s1,[q,r]);
 const s2=await h.step({kernel:[far],keep});
 intact(s1,s2,[q,r]);
 assert.deepEqual([s2.fatal,s2.fatalFirst,s2.fatalGeneration,key(s2.fatalPage),s2.resident.size,s2.claimed.length,s2.free.length],[FATAL.alias,FATAL.alias,2,key(far),2,0,14]);
 // The aliasing page reads q's entry: resident by kind and by wrapped block, but not its own.
 assert.deepEqual(s2.looked.get(key(far)),[NONE,keep[0],KIND.resident]);
 const s3=await h.step({kernel:trio.slice(0,2),host:trio.slice(2),keep});
 intact(s1,s3,[q,r]);assert.equal(trio.filter(p=>s3.resident.has(key(p))).length,1);
 assert.deepEqual([s3.fatal,s3.fatalGeneration,key(s3.fatalPage),s3.resident.size],[FATAL.alias,2,key(far),3]);
 if(!round)console.log(JSON.stringify({scenario:"alias against a resident page",fatal:s2.fatal,page:s2.fatalPage}));
 h.destroy();
}
// Aliases within one step, with nothing resident: one page is claimed, one is named. No mirror.
for(let round=0;round<rounds;round++){
 const h=await harness(small,{mirrors:0});
 const trio:Page[]=[[5,5,5],[5,13,5],[5,5,-3]];
 const s1=await h.step({kernel:trio});
 assert.deepEqual([s1.fatal,s1.fatalFirst,s1.fatalGeneration,s1.resident.size,s1.claimed.length],[FATAL.alias,FATAL.alias,1,1,1]);
 assert.ok(keys(trio).includes(key(s1.fatalPage))&&keys(trio).includes([...s1.resident.keys()][0]!));
 if(!round)console.log(JSON.stringify({scenario:"alias within a step",fatal:s1.fatal,page:s1.fatalPage,resident:[...s1.resident.keys()]}));
 h.destroy();
}
// Six distinct requests against a list of four: two dropped and named; asked again, they are claimed. Four mirrors.
for(let round=0;round<rounds;round++){
 const h=await harness({...small,requestCapacity:4},{mirrors:4});
 const G:Page[]=Array.from({length:6},(_,i)=>[i-80,3*i,1-8*i] as const);h.distinct(G);
 const s1=await h.step({kernel:G});
 const got=G.filter(p=>s1.resident.has(key(p))),dropped=G.filter(p=>!s1.resident.has(key(p)));
 assert.deepEqual([s1.fatal,s1.fatalFirst,s1.fatalGeneration,s1.resident.size,s1.claimed.length,s1.requests,dropped.length],[FATAL.requests,FATAL.requests,1,4,4,4,2]);
 assert.ok(keys(dropped).includes(key(s1.fatalPage)));
 const s2=await h.step({kernel:dropped,keep:blocksOf(s1,got)});
 intact(s1,s2,got);assert.deepEqual([...s2.resident.keys()].sort(),keys(G));assert.deepEqual([s2.fatal,s2.claimed.length],[FATAL.requests,2]);
 // The host list is bounded at the call.
 assert.throws(()=>h.pool.request(0.5,0,0),RangeError);
 assert.throws(()=>{for(let i=0;i<9;i++)h.pool.request(i,0,0);},/host commands exceed 8/);
 if(!round)console.log(JSON.stringify({scenario:"request overflow",fatal:s1.fatal,page:s1.fatalPage,dropped}));
 h.destroy();
}
// An entry turned SOLID after its page was listed and before the claim: the request is dropped, no block is popped, nothing is fatal.
for(let round=0;round<rounds;round++){
 const h=await harness(small);
 const p:Page=[2,3,4],q:Page=[5,5,5],again:Page=[10,-5,12];h.distinct([p,q]);assert.equal(h.entryOf(again),h.entryOf(p));
 const s1=await h.step({kernel:[p,q],between:()=>h.classify(p)});
 assert.deepEqual([[...s1.resident.keys()],s1.claimed.length,s1.requests,s1.free.length,s1.fatal],[[key(q)],1,2,15,0]);
 assert.deepEqual(s1.looked.get(key(p)),[NONE,17,KIND.solid]);
 // Asked again from both sources, by its own coordinate and by one a table away: nothing.
 const s2=await h.step({kernel:[p,again],host:[again,p],keep:blocksOf(s1,[q])});
 intact(s1,s2,[q]);assert.deepEqual([s2.resident.size,s2.claimed.length,s2.requests,s2.free.length,s2.fatal],[1,0,0,15,0]);
 // The audit itself: a word changed in a mirror's tail, then one beside it, is caught.
 h.poke(1,h.mirrors[1]!.offset+h.pool.layout.tail.blocks+3);await assert.rejects(h.step(),/mirror 1 tail/);
 h.poke(0,h.mirrors[0]!.offset-1);await assert.rejects(h.step(),/mirror 0 word 4 was written/);
 if(!round)console.log(JSON.stringify({scenario:"solid between request and claim",resident:[...s2.resident.keys()],fatal:s2.fatal}));
 h.destroy();
}
// 2050 blocks: host and request lists longer than their launches (strided), nine blocks a lane, idle 1.
for(let round=0;round<rounds;round++){
 const h=await harness({table:16,blocks:2050,idleSteps:1,hostCapacity:2048,requestCapacity:2048});
 const W:Page[]=Array.from({length:3448},(_,i)=>[i%16-16*(i%5),(i>>4)%16+112,(i>>8)-16*(i%3)] as const);h.distinct(W);
 const s1=await h.step({host:W.slice(0,1500)});
 assert.deepEqual([...s1.resident.keys()].sort(),keys(W.slice(0,1500)));
 assert.deepEqual(ascending(s1.claimed),Array.from({length:1500},(_,i)=>i+1));assert.deepEqual([s1.free.length,s1.pending.length,s1.requests,s1.fatal],[548,0,1500,0]);
 // 2: 32 kept, 1468 retire, 548 pages take exactly the free blocks.
 const kept=W.slice(0,32),keep=blocksOf(s1,kept);
 const s2=await h.step({host:W.slice(1500,2048),keep});
 intact(s1,s2,kept);assert.deepEqual([...s2.resident.keys()].sort(),keys([...kept,...W.slice(1500,2048)]));
 assert.deepEqual(ascending(s2.claimed),ascending(s1.free));assert.deepEqual(ascending(s2.pending),ascending(blocksOf(s1,W.slice(32,1500))));
 assert.deepEqual([s2.free.length,s2.fatal],[0,0]);
 // 3: 1400 pages take the oldest quarantined blocks; the 548 retire.
 const s3=await h.step({host:W.slice(2048),keep});
 intact(s1,s3,kept);assert.deepEqual([...s3.resident.keys()].sort(),keys([...kept,...W.slice(2048)]));
 assert.deepEqual(ascending(s3.claimed),ascending(s2.pending.slice(0,1400)));assert.deepEqual(s3.free,s2.pending.slice(1400));
 assert.deepEqual(ascending(s3.pending),ascending(s2.claimed));assert.equal(s3.fatal,0);
 if(!round)console.log(JSON.stringify({scenario:"2050 blocks",resident:[s1.resident.size,s2.resident.size,s3.resident.size],retired:[s2.pending.length,s3.pending.length]}));
 h.destroy();
}

// The largest pool the layout takes, 65536 blocks under a table of 64, seeded full: 65534 pages in scan order, AIR
// elsewhere. All but 32 retire in one commit (256 blocks a lane), then 2048 host pages take the oldest quarantined blocks.
{
 const pages=Array.from({length:65534},(_,e)=>({x:e&63,y:(e>>6)&63,z:e>>12,block:e+1}));
 const h=await harness({table:64,blocks:65536,idleSteps:1,hostCapacity:2048,requestCapacity:2048},{pages});
 const s0=await h.step({commit:false,look:[[0,0,0],[61,63,15],[62,63,15],[0,0,16]]});
 assert.equal(differ(s0.state,h.seed.state),-1);assert.deepEqual([s0.resident.size,s0.free,s0.pending],[65534,[],[]]);
 assert.deepEqual([[0,0,0],[61,63,15],[62,63,15],[0,0,16]].map(p=>s0.looked.get(p.join(","))),[[1,1,KIND.resident],[65534,65534,KIND.resident],[NONE,0,KIND.air],[NONE,0,KIND.air]]);
 const keep=Array.from({length:32},(_,i)=>2000*i+7);
 const s1=await h.step({keep});
 assert.deepEqual([ascending([...s1.resident.values()]),s1.free,s1.pending.length,s1.fatal],[keep,[],65502,0]);
 const fresh:Page[]=Array.from({length:2048},(_,i)=>[i&63,i>>6,32] as const);
 const s2=await h.step({host:fresh,keep});
 assert.deepEqual([s2.resident.size,s2.claimed.length,s2.requests,s2.free.length,s2.pending.length,s2.fatal],[2080,2048,2048,63454,0,0]);
 assert.deepEqual(ascending(s2.claimed),s1.pending.slice(0,2048));assert.deepEqual(s2.free,s1.pending.slice(2048));
 console.log(JSON.stringify({scenario:"65536 blocks",resident:[s0.resident.size,s1.resident.size,s2.resident.size],retired:s1.pending.length,stateBytes:h.pool.state.size,workBytes:h.pool.work.size}));
 h.destroy();
}

// The letters shape: 208 blocks (206 slots, a ring of 256), table 64, the 8x6x4 box of pages at the window origin.
const letters={table:64,blocks:208} as const;
const box:Page[]=[];for(let z=0;z<4;z++)for(let y=0;y<6;y++)for(let x=0;x<8;x++)box.push([x,y,z]);
/** Pages whose entries lie outside the box, by every face, the far corner and coordinates tables away. */
const outside:Page[]=[[-1,0,0],[8,0,0],[0,-1,0],[0,6,0],[0,0,-1],[0,0,4],[63,63,63],[8,6,4],[32,32,32],[200,5,3],[-55,-57,-123],[2147483647,-2147483648,77],[7,5,68+64],[-64,70,0]];
/** The seed's words, bit for bit: both buffers, and (by the audit) every mirror. */
const asSeeded=(h:Awaited<ReturnType<typeof harness>>,s:Snapshot)=>{
 const l=h.pool.layout;
 assert.equal(differ(s.state,h.seed.state),-1,"state as seeded");
 assert.equal(differ(s.work.subarray(l.work.ring,l.work.ring+l.ringCapacity),h.seed.ring),-1,"ring as seeded");
 assert.ok(s.work.every((w,i)=>w===0||(i>=l.work.ring&&i<l.work.ring+l.ringCapacity)),"work clear outside the ring");
};
/** Table, block words and resident list as the seed wrote them. */
const tailAsSeeded=(h:Awaited<ReturnType<typeof harness>>,s:Snapshot)=>{
 const l=h.pool.layout;
 assert.equal(differ(s.state.subarray(l.state.table,l.state.table+l.tail.words),h.seed.tail),-1,"tail as seeded");
 assert.equal(differ(s.state.subarray(l.state.resident,l.state.resident+h.seed.resident.length),h.seed.resident),-1,"resident list as seeded");
};
// Static set in scan order, every other entry SOLID, seeded before initialize.
{
 const h=await harness({...letters,idleSteps:4},{pages:box.map(([x,y,z])=>({x,y,z,block:1+x+8*(y+6*z)})),other:()=>KIND.solid});
 const l=h.pool.layout,solids=h.seed.table.reduce((n,w)=>n+ +(w===SOLID),0);
 assert.deepEqual([h.seed.resident.length,h.seed.free,solids,l.tail.words,l.ringCapacity],[192,14,262144-192,262352,256]);
 const inside:Page[]=[[0,0,0],[7,5,3],[3,2,1]];
 const s0=await h.step({commit:false,look:[...inside,[64,0,0],...outside]});
 asSeeded(h,s0);
 assert.deepEqual([s0.generation,s0.resident.size,s0.free,s0.pending,s0.fatal],[0,192,Array.from({length:14},(_,i)=>193+i),[],0]);
 for(const [x,y,z] of box)assert.equal(s0.resident.get(key([x,y,z])),1+x+8*(y+6*z));
 assert.deepEqual(inside.map(p=>s0.looked.get(key(p))),[[1,1,KIND.resident],[192,192,KIND.resident],[68,68,KIND.resident]]);
 assert.deepEqual(s0.looked.get("64,0,0"),[NONE,1,KIND.resident]);
 for(const p of outside)assert.deepEqual(s0.looked.get(key(p)),[NONE,207,KIND.solid],`page ${p}`);
 // 1, 2: asked toward SOLID from the kernel and the host, beside residents asked again: nothing listed, nothing claimed, nothing fatal.
 const s1=await h.step({kernel:[...outside,[0,0,0],[7,5,3]],host:[...outside.slice(3),[4,4,2]],keep:Array.from({length:32},(_,i)=>i+1)});
 const s2=await h.step({kernel:outside.slice(5),host:outside});
 for(const s of [s1,s2]){
  tailAsSeeded(h,s);assert.deepEqual([s.resident.size,s.claimed.length,s.requests,s.free,s.pending,s.fatal,s.fatalFirst],[192,0,0,s0.free,[],0,0]);
  for(const p of outside)assert.deepEqual(s.looked.get(key(p)),[NONE,207,KIND.solid],`page ${p}`);
 }
 assert.deepEqual([s1.idle.get("0,0,0"),s1.idle.get("7,5,3"),s1.idle.get("4,4,2"),s1.idle.get("1,3,0"),s1.idle.get("5,5,3"),s2.idle.get("0,0,0")],[0,0,0,0,1,1]);
 console.log(JSON.stringify({scenario:"static seed, requests toward SOLID",resident:s2.resident.size,solid:solids,free:s2.free.length,tailBytes:4*l.tail.words,generation:s2.generation,fatal:s2.fatal}));
 h.destroy();
}
// The same pages in permuted blocks with fourteen free holes among them, seeded after initialize.
{
 const order=shuffled(Array.from({length:206},(_,i)=>i+1),random(7)).slice(0,192);
 const h=await harness({...letters,idleSteps:4},{pages:box.map(([x,y,z],i)=>({x,y,z,block:order[i]!})),other:()=>KIND.solid,late:true});
 const holes=Array.from({length:206},(_,i)=>i+1).filter(block=>!order.includes(block));
 const s0=await h.step({commit:false,look:[...box.slice(0,40),...outside]});
 asSeeded(h,s0);
 assert.deepEqual([s0.generation,s0.resident.size,s0.free,s0.pending,s0.fatal],[0,192,holes,[],0]);
 box.forEach((p,i)=>assert.equal(s0.resident.get(key(p)),order[i]));
 box.slice(0,40).forEach((p,i)=>assert.deepEqual(s0.looked.get(key(p)),[order[i],order[i],KIND.resident]));
 assert.ok(holes.some(block=>block<192)&&order.some((block,i)=>i>0&&block<order[i-1]!),"blocks must be out of scan order with holes among them");
 const s1=await h.step({kernel:outside,host:box.slice(100,160)});
 tailAsSeeded(h,s1);assert.deepEqual([s1.resident.size,s1.claimed.length,s1.requests,s1.free,s1.fatal],[192,0,0,holes,0]);intact(s0,s1,box);
 console.log(JSON.stringify({scenario:"permuted seed with holes",resident:s1.resident.size,holes,firstBlocks:order.slice(0,8)}));
 h.destroy();
}
// A half box seeded in permuted blocks with AIR above it and at six stray entries, every other entry SOLID; then random
// requests (kernel and host, some toward SOLID) and keeps, against a host model of the resident set and its idle counts.
for(let round=0;round<rounds;round++){
 const rand=random(0x51ed+round),far=(p:Page,i:number):Page=>[p[0]+64*(i%5-2),p[1]-64*(i%3),p[2]+64*(i%7-3)];
 const strays:Page[]=[[40,41,42],[63,0,13],[20,63,5],[33,9,60],[8,6,4],[62,62,62]];
 const universe=[...box,...strays].map(far),seeded=universe.filter((_,i)=>i<96);
 const order=shuffled(Array.from({length:206},(_,i)=>i+1),rand);
 const l=uniformPagePoolLayout({...letters,idleSteps:2}),mask=l.table-1,entry=(x:number,y:number,z:number)=>(x&mask)|((y&mask)<<6)|((z&mask)<<12);
 const air=new Set(universe.map(p=>entry(...p)));assert.equal(air.size,universe.length);
 const h=await harness({...letters,idleSteps:2},{pages:seeded.map(([x,y,z],i)=>({x,y,z,block:order[i]!})),other:(x,y,z)=>air.has(entry(x,y,z))?KIND.air:KIND.solid,late:round%2===1});
 let s=await h.step({commit:false});
 asSeeded(h,s);assert.deepEqual([s.resident.size,s.free.length],[96,110]);
 let model=new Map(seeded.map(p=>[key(p),0])),claims=0,retirements=0,most=96,least=96;
 const solidPage=():Page=>{for(;;){const p:Page=[Math.floor(rand()*64),Math.floor(rand()*64),Math.floor(rand()*64)];if(!air.has(entry(...p)))return far(p,Math.floor(rand()*105));}};
 for(let turn=0;turn<24;turn++){
  const asked=shuffled(universe.filter(()=>rand()<0.4),rand),kernel=[...asked.slice(0,30),solidPage(),solidPage()],host=[...asked.slice(30,92),solidPage(),solidPage()];
  const keeping=[...s.resident].filter(()=>rand()<0.15).slice(0,KEEPS),want=new Set([...asked.slice(0,92).map(key),...keeping.map(([k])=>k)]);
  const next=new Map<string,number>();let claimed=0,retired=0;
  for(const [k,idle] of model){const now=want.has(k)?0:idle+1;if(now<2)next.set(k,now);else retired++;}
  for(const p of asked.slice(0,92))if(!model.has(key(p))){next.set(key(p),0);claimed++;}
  s=await h.step({kernel,host,keep:keeping.map(([,block])=>block)});
  assert.deepEqual([...s.idle].sort(),[...next].sort(),`turn ${turn}`);
  assert.deepEqual([s.claimed.length,s.pending.length,s.requests,s.fatal],[claimed,retired,claimed,0],`turn ${turn}`);
  model=next;claims+=claimed;retirements+=retired;most=Math.max(most,s.resident.size);least=Math.min(least,s.resident.size);
 }
 // The ring head went round its 256 words although the pool has 206 slots.
 assert.ok(claims>l.ringCapacity,`only ${claims} claims`);assert.equal(s.work[WORK.head],claims&(l.ringCapacity-1));
 if(!round)console.log(JSON.stringify({scenario:"seeded pool, random requests and keeps",commits:s.generation,claims,retirements,resident:[least,most],ringHead:s.work[WORK.head]}));
 h.destroy();
}

// Live allocation inside a box: a list of 16-cell pages with a reach asks through the probe's upCertify,
// pinned pages are asked for by every commit, and the published record follows the table. Each scenario is checked
// against a host model of the same script: the resident pages and their idle counts after every commit.
type Triple=readonly [number,number,number];
/** The certificate of seed pages of 16 cells in a box: every such page of the box within `reach16` of a seed (Chebyshev), the seeds flagged. */
const certificateOf=(pages:Triple,seeds:readonly Page[],reach16:number,reach:number):Certificate=>{
 const found=new Map<string,[number,number,number,number]>();
 for(const s of seeds)for(let dz=-reach16;dz<=reach16;dz++)for(let dy=-reach16;dy<=reach16;dy++)for(let dx=-reach16;dx<=reach16;dx++){
  const p:Page=[s[0]+dx,s[1]+dy,s[2]+dz];if(p.some((c,k)=>c<0||c>=2*pages[k]!))continue;
  const seeded=+!(dx||dy||dz),had=found.get(key(p));
  if(had)had[3]|=seeded;else found.set(key(p),[...p,seeded]);
 }
 return {pages:[...found.values()],reach};
};
/** The pool pages a certificate asks for. */
const asked=(origin:Triple,c:Certificate)=>new Set(c.pages.map(([x,y,z])=>key([origin[0]+(x>>1),origin[1]+(y>>1),origin[2]+(z>>1)])));
/** One commit of the model: a page asked for is claimed or has its idle count reset, any other ages and retires at idleSteps. */
const commitModel=(model:ReadonlyMap<string,number>,want:ReadonlySet<string>,idleSteps:number)=>{
 const next=new Map<string,number>();let claimed=0,retired=0;
 for(const [k,idle] of model){const now=want.has(k)?0:idle+1;if(now<idleSteps)next.set(k,now);else retired++;}
 for(const k of want)if(!model.has(k)){next.set(k,0);claimed++;}
 return {next,claimed,retired};
};
const pageOf=(k:string)=>k.split(",").map(Number) as unknown as Page;
// Static keep-all: the letters box seeded whole (scan order, then permuted with holes), its whole certificate asked every commit.
{
 const all:[number,number,number,number][]=[];for(let z=0;z<8;z++)for(let y=0;y<12;y++)for(let x=0;x<16;x++)all.push([x,y,z,+((x+y+z)%3===0)]);
 const counts:number[][]=[];
 for(const permute of [undefined,11]){
  const h=await harness({...letters,idleSteps:4,box:{origin:[0,0,0],pages:[8,6,4]}},{box:{permute},mirrors:permute===undefined?4:2,late:permute!==undefined});
  let s=await h.step({commit:false});
  asSeeded(h,s);assert.equal(differ(s.published,uniformPagePoolPublished(h.seed)),-1);
  if(permute===undefined)for(const [x,y,z] of box)assert.equal(s.resident.get(key([x,y,z])),1+x+8*(y+6*z));
  else assert.ok(s.free.some(block=>block<192)&&box.some(([x,y,z])=>s.resident.get(key([x,y,z]))!==1+x+8*(y+6*z)),"blocks must be out of scan order with holes among them");
  const resident:number[]=[];
  for(let commit=1;commit<=20;commit++){
   s=await h.step({certificate:{pages:all,reach:16}});
   tailAsSeeded(h,s);resident.push(s.resident.size);
   assert.deepEqual([s.resident.size,s.claimed.length,s.pending.length,s.requests,s.free.length,s.fatal,s.status[STATUS.generation]],[192,0,0,0,14,0,commit],`commit ${commit}`);
   assert.ok([...s.idle.values()].every(idle=>idle===0),`commit ${commit}: every page kept`);
  }
  counts.push(resident);
  console.log(JSON.stringify({scenario:`static keep-all, ${permute===undefined?"identity":"permuted"} seed`,commits:s.generation,resident:[Math.min(...resident),Math.max(...resident)],certificate:all.length,fatal:s.fatal}));
  h.destroy();
 }
 assert.deepEqual(counts[0],counts[1]);
}
// Growth along a path: a 2x2x2 block of seeded 16-cell pages walks out along x and z, up y and back, one such page a
// step inside a margin of two, in a box of 240 pages and a pool three blocks over the path's need. Pages are claimed ahead,
// retired idleSteps commits after the margin leaves them and claimed again on the way back; two pinned pages stay.
// With idleSteps 1 a page retires in the first commit that does not ask for it, so blocks turn over at once.
for(const idleSteps of [3,1]){
 const origin:Triple=[-3,2,5],pages:Triple=[12,5,4],reach16=2;
 const heads:Page[]=[];
 {let p:[number,number,number]=[1,1,1];heads.push([...p]);const walk=(axis:number,by:number,n:number)=>{for(let i=0;i<n;i++){p[axis]!+=by;if(axis===0&&i%5===2)p[2]+=by;heads.push([...p]);}};
  walk(0,1,20);walk(1,1,6);walk(0,-1,19);walk(1,-1,6);}
 const cube:Page[]=[];for(let z=0;z<2;z++)for(let y=0;y<2;y++)for(let x=0;x<2;x++)cube.push([x,y,z]);
 const certificates=heads.map(h=>certificateOf(pages,cube.map(([x,y,z])=>[h[0]+x,h[1]+y,h[2]+z] as const),reach16,4*reach16));
 const pins:Page[]=[[origin[0]+11,origin[1]+4,origin[2]+3],[origin[0],origin[1]+4,origin[2]+3]];
 // The model's whole run, ahead of the pool: the seed is the first certificate's pages, and commit n asks certificate n-1 and the pins.
 const seeded=asked(origin,certificates[0]!),wants=certificates.map(c=>new Set([...asked(origin,c),...pins.map(key)]));
 const models:{next:Map<string,number>;claimed:number;retired:number}[]=[];let need=0,exhaustedAt=0;
 {let model=new Map([...seeded].map(k=>[k,0]));for(const want of wants){const m=commitModel(model,want,idleSteps);need=Math.max(need,model.size+m.claimed);models.push(m);model=m.next;}}
 assert.ok(need+3<pages[0]*pages[1]*pages[2]&&seeded.size<need,"the pool must be smaller than the box and larger than the seed");
 const inSeed=(x:number,y:number,z:number)=>seeded.has(key([origin[0]+x,origin[1]+y,origin[2]+z]));
 const runs:{counters:number[][];published:Uint32Array[]}[]=[];
 for(const permute of [undefined,5]){
  const h=await harness({table:64,blocks:need+3+2,idleSteps,requestCapacity:64,box:{origin,pages}},{box:{resident:inSeed,permute},pins,mirrors:permute===undefined?4:2,late:permute!==undefined});
  let s=await h.step({commit:false});
  assert.deepEqual([...s.resident.keys()].sort(),[...seeded].sort());
  const counters:number[][]=[],published:Uint32Array[]=[],retiredAt=new Map<number,number>(),left=new Set<string>();let reused=0,reclaimed=0,model:ReadonlyMap<string,number>=new Map([...seeded].map(k=>[k,0]));
  for(let n=0;n<certificates.length;n++){
   const m=models[n]!,before=s,at=`commit ${n+1}`;
   s=await h.step({certificate:certificates[n]});
   assert.deepEqual([...s.idle].sort(),[...m.next].sort(),at);
   assert.deepEqual([s.claimed.length,s.pending.length,s.requests,s.fatal,s.free.length],[m.claimed,m.retired,m.claimed,0,h.pool.layout.slots-m.next.size-m.retired],at);
   assert.deepEqual([STATUS.generation,STATUS.resident,STATUS.claimed,STATUS.pending,STATUS.free,STATUS.requests,STATUS.fatal].map(w=>s.status[w]),[n+1,m.next.size,m.claimed,m.retired,s.free.length,m.claimed,0],at);
   // Pins are resident from the first commit and never age; every surviving page keeps its block.
   for(const p of pins)assert.equal(s.idle.get(key(p)),0,`${at}: pin ${p}`);
   intact(before,s,[...before.resident.keys()].filter(k=>s.resident.has(k)).map(pageOf));
   // Claims take the oldest free blocks, the blocks the last commit retired behind them; none retired by this commit.
   assert.deepEqual(ascending(s.claimed),ascending([...before.free,...before.pending].slice(0,m.claimed)),at);
   for(const block of s.claimed){assert.ok(!s.pending.includes(block),at);if(retiredAt.has(block))reused++;}
   for(const k of s.resident.keys())if(!model.has(k)&&left.has(k))reclaimed++;
   for(const [k,block] of before.resident)if(!s.resident.has(k)){left.add(k);retiredAt.set(block,n+1);assert.ok(s.pending.includes(block),at);}
   counters.push([s.generation,s.resident.size,s.claimed.length,s.pending.length,s.free.length,s.requests,s.fatal]);published.push(s.published);model=m.next;
  }
  assert.ok(reused>0&&reclaimed>0,`blocks reused ${reused}, pages claimed again ${reclaimed}`);
  runs.push({counters,published});
  const resident=counters.map(c=>c[1]!);
  console.log(JSON.stringify({scenario:`growth along a path, ${permute===undefined?"identity":"permuted"} seed`,idleSteps,commits:s.generation,blocks:h.pool.layout.blocks,boxPages:h.pool.layout.box.count,seeded:seeded.size,resident:[Math.min(...resident),Math.max(...resident)],
   claims:counters.reduce((n,c)=>n+c[2]!,0),retirements:counters.reduce((n,c)=>n+c[3]!,0),blocksReused:reused,pagesClaimedAgain:reclaimed,fatal:s.fatal}));
  if(permute===undefined&&idleSteps===3)console.log(`generation resident claimed retired free requests fatal\n${counters.filter((_,i)=>i<6||i%10===9).map(c=>c.join(" ")).join("\n")}`);
  h.destroy();
 }
 // The permuted seed with holes gives the same pages, counters and published records as the identity seed.
 assert.deepEqual(runs[1]!.counters,runs[0]!.counters);
 runs[0]!.published.forEach((record,i)=>assert.equal(differ(runs[1]!.published[i]!,record),-1,`published record of commit ${i+1}`));
 // One block fewer than the path needs: exhausted at the commit the model names, by a page that commit asked for.
 {
  const h=await harness({table:64,blocks:need-1+2,idleSteps,requestCapacity:64,box:{origin,pages}},{box:{resident:inSeed},pins});
  let s=await h.step({commit:false}),model:ReadonlyMap<string,number>=new Map([...seeded].map(k=>[k,0]));
  for(let n=0;n<certificates.length&&!s.fatal;n++){
   const m=models[n]!,residentBefore=model.size;
   s=await h.step({certificate:certificates[n]});
   if(residentBefore+m.claimed<=need-1){assert.equal(s.fatal,0,`commit ${n+1}`);assert.deepEqual([...s.idle].sort(),[...m.next].sort());model=m.next;continue;}
   exhaustedAt=n+1;
   assert.deepEqual([s.fatal,s.fatalFirst,s.fatalGeneration,s.fatalDetail,s.claimed.length,s.requests],[FATAL.exhausted,FATAL.exhausted,n+1,residentBefore,m.claimed-1,m.claimed]);
   assert.ok(wants[n]!.has(key(s.fatalPage))&&!model.has(key(s.fatalPage))&&!s.resident.has(key(s.fatalPage)),`exhaustion names ${s.fatalPage}`);
   assert.equal(s.message,`Page pool exhausted at commit ${n+1}: page (${s.fatalPage.join(", ")}) requested with ${residentBefore} of ${need-1} blocks resident and the other ${need-1-residentBefore} claimed by the same commit. Raise pagePool.blocks (4.9 MiB each) or shrink the scene.`);
  }
  assert.ok(exhaustedAt>0,"the path must exhaust the smaller pool");
  console.log(JSON.stringify({scenario:"growth along a path, one block short",idleSteps,blocks:need+1,exhaustedAt,page:s.fatalPage,message:s.message}));
  h.destroy();
 }
}
// A box of 4x4x2 pages in a pool of sixteen blocks: pins, the certificate's fatals and the published record's audit.
const tight={table:8,blocks:18,idleSteps:2,hostCapacity:8,pinCapacity:4,closureCap:6,box:{origin:[0,0,0],pages:[4,4,2]}} as const;
/** Certificate entries, seeded, over every 16-cell page of the box pages `inside` accepts. */
const over=(inside:(x:number,y:number,z:number)=>boolean)=>{const list:[number,number,number,number][]=[];for(let z=0;z<4;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++)if(inside(x>>1,y>>1,z>>1))list.push([x,y,z,1]);return list;};
// Pins: claimed by the first commit, never aged, while a host page beside them retires; a pin toward SOLID is nothing.
for(const late of [false,true]){
 const pins:Page[]=[[0,0,0],[3,3,1],[5,0,0]],other:Page=[1,1,0];
 const h=await harness(tight,{box:{resident:()=>false},pins,late});
 assert.throws(()=>h.pool.pin(Array.from({length:5},(_,x)=>({x,y:0,z:0}))),/pins exceed 4/);assert.throws(()=>h.pool.pin([{x:0.5,y:0,z:0}]),RangeError);assert.equal(h.pool.pinned,3);
 const s0=await h.step({commit:false,look:pins});assert.deepEqual([s0.resident.size,s0.published[PUBLISHED.resident]],[0,0]);
 const s1=await h.step({host:[other]});
 assert.deepEqual([[...s1.resident.keys()].sort(),s1.claimed.length,s1.requests,s1.fatal],[keys([pins[0]!,pins[1]!,other]),3,3,0]);
 assert.deepEqual(s1.looked.get(key(pins[2]!)),[NONE,17,KIND.solid]);
 let s=s1;const residents:number[]=[];
 for(let commit=2;commit<=9;commit++){
  s=await h.step();residents.push(s.resident.size);
  intact(s1,s,[pins[0]!,pins[1]!]);assert.deepEqual([s.idle.get(key(pins[0]!)),s.idle.get(key(pins[1]!)),s.claimed.length,s.requests,s.fatal],[0,0,0,0,0],`commit ${commit}`);
 }
 assert.deepEqual(residents,[3,2,2,2,2,2,2,2]);
 console.log(JSON.stringify({scenario:`pinned pages, pinned ${late?"after":"before"} the seed`,commits:s.generation,pinned:h.pool.pinned,resident:[...s.resident.keys()],fatal:s.fatal}));
 h.destroy();
}
// Exhaustion with one page named: the pool is full and kept, and the certificate's margin reaches one page more.
{
 const h=await harness(tight,{box:{resident:(_x,_y,z)=>z===0}});
 const certificate:Certificate={pages:[...over((_x,_y,z)=>z===0),[2,4,2,0]],reach:3};
 const s0=await h.step({commit:false});assert.deepEqual([s0.resident.size,s0.free.length],[16,0]);
 const s1=await h.step({certificate});
 intact(s0,s1,[...s0.resident.keys()].map(pageOf));
 assert.deepEqual([s1.fatal,s1.fatalFirst,s1.fatalGeneration,s1.fatalPage,s1.fatalDetail,s1.resident.size,s1.claimed.length,s1.requests],[FATAL.exhausted,FATAL.exhausted,1,[1,2,1],16,16,0,1]);
 assert.equal(s1.message,"Page pool exhausted at commit 1: page (1, 2, 1) requested with 16 of 16 blocks resident and the other 0 claimed by the same commit. Raise pagePool.blocks (4.9 MiB each) or shrink the scene.");
 // The audit of the published record itself: a flag changed behind the pool is caught, and the next commit rewrites it.
 h.smudge(21);await assert.rejects(h.step({commit:false}),/published record/);
 const s2=await h.step({certificate});assert.deepEqual([s2.fatal,s2.fatalGeneration,s2.fatalPage,s2.resident.size],[FATAL.exhausted,1,[1,2,1],16]);
 console.log(JSON.stringify({scenario:"exhaustion names its page",fatal:s1.fatal,page:s1.fatalPage,message:s1.message}));
 h.destroy();
}
// A reach over the cap: closureCap with the reach, nothing asked for, so every page ages; at the cap the same certificate asks.
{
 const h=await harness(tight,{box:{resident:(_x,y,z)=>z===0&&y<2}});
 const pages=over((_x,y,z)=>z===0&&y<3),want=asked([0,0,0],{pages,reach:0});
 const s0=await h.step({commit:false});assert.equal(s0.resident.size,8);
 const s1=await h.step({certificate:{pages,reach:7}});
 intact(s0,s1,[...s0.resident.keys()].map(pageOf));
 assert.deepEqual([s1.fatal,s1.fatalFirst,s1.fatalGeneration,s1.fatalDetail,s1.resident.size,s1.claimed.length,s1.requests],[FATAL.closureCap,FATAL.closureCap,1,7,8,0,0]);
 assert.ok(want.has(key(s1.fatalPage)),`closure cap names ${s1.fatalPage}`);assert.ok([...s1.idle.values()].every(idle=>idle===1));
 assert.equal(s1.message,`Departure travel gives a closure of 7 tiles, over the cap of 6 (commit 1, page (${s1.fatalPage.join(", ")})): a sample is moving faster than the residency certificate covers.`);
 const s2=await h.step({certificate:{pages,reach:6}});
 assert.deepEqual([[...s2.resident.keys()].sort(),s2.claimed.length,s2.requests,s2.fatal,s2.fatalFirst,s2.fatalGeneration,s2.fatalDetail,key(s2.fatalPage)],[[...want].sort(),4,4,FATAL.closureCap|FATAL.closure,FATAL.closureCap,1,7,key(s1.fatalPage)]);
 console.log(JSON.stringify({scenario:"reach over the closure cap",fatal:s1.fatal,page:s1.fatalPage,message:s1.message}));
 h.destroy();
}
// A seeded certificate page in a page that is not resident: closure with that page; a margin page beside it is an ordinary claim.
{
 const h=await harness(tight,{box:{resident:(_x,y,z)=>z===0&&y<2}});
 const pages:[number,number,number,number][]=[...over((_x,y,z)=>z===0&&y<2),[1,5,0,1],[6,4,1,0]];
 const s0=await h.step({commit:false});
 const s1=await h.step({certificate:{pages,reach:6}});
 intact(s0,s1,[...s0.resident.keys()].map(pageOf));
 assert.deepEqual([s1.fatal,s1.fatalFirst,s1.fatalGeneration,s1.fatalPage,s1.fatalDetail,s1.resident.size,s1.claimed.length,s1.requests],[FATAL.closure,FATAL.closure,1,[0,2,0],0,10,2,2]);
 assert.ok(s1.resident.has("0,2,0")&&s1.resident.has("3,2,0"));
 assert.equal(s1.message,"Liquid or band tile found in page (0, 2, 0), which was not resident at the start of commit 1.");
 // Resident now, the same certificate raises nothing more.
 const s2=await h.step({certificate:{pages,reach:6}});assert.deepEqual([s2.fatal,s2.fatalGeneration,s2.claimed.length,s2.resident.size],[FATAL.closure,1,0,10]);
 console.log(JSON.stringify({scenario:"seeded page not resident",fatal:s1.fatal,page:s1.fatalPage,message:s1.message}));
 h.destroy();
 // Margin alone claims without a fatal; a seeded page toward SOLID (past the box) is a closure and stays unstored.
 const g=await harness(tight,{box:{resident:(_x,y,z)=>z===0&&y<2}});
 const t1=await g.step({certificate:{pages:[...over((_x,y,z)=>z===0&&y<2),[1,5,0,0],[6,4,1,0]],reach:6}});
 assert.deepEqual([t1.fatal,t1.resident.size,t1.claimed.length,t1.message],[0,10,2,null]);
 const t2=await g.step({certificate:{pages:[...over((_x,y,z)=>z===0&&y<2),[8,0,0,1]],reach:6}});
 assert.deepEqual([t2.fatal,t2.fatalFirst,t2.fatalPage,t2.fatalGeneration,t2.claimed.length,t2.requests,t2.resident.has("4,0,0")],[FATAL.closure,FATAL.closure,[4,0,0],2,0,0,false]);
 g.destroy();
}
await device.queue.onSubmittedWorkDone();assert.deepEqual(errors,[]);device.destroy();
console.log(`uniform page pool: ok (${rounds} round${rounds===1?"":"s"}, ${Math.round(performance.now()-started)} ms)`);
