/** Page initialise: a block is written in full when a commit claims it, and
 * a block a commit retires hands each vertex it held to the stored block
 * that holds that vertex next (the retire handoff below).
 *
 * UniformPageInitialise writes the AIR value of every field of a detail
 * storage over every texel the storage holds for each block of the pool's
 * new list (upNewCount / upNewBlock, the blocks the last commit claimed): a
 * cell or face field's 32^3 h texels and its base twin (8^3; a face's 8x8x32),
 * a vertex field's 33^3 h texels (skirt included) and its 9^3 base, a
 * per-tile field's 8^3 texels, and the 3072 rows of every low-face-keyed
 * buffer it is given. A block's texels start at umBlockCoord(block) times the
 * class's block extent, in every texture but an h store, which places them at
 * the block's h slot (the storage's slotState; a block with none has no h
 * texels to write).
 *
 * The fields are the storage's own list (fieldTextures) as of construction,
 * every generation of a ping-ponged field being a field of its own. AIR is
 * the field's own empty value (uniformDetailField(handle).empty: the scene's
 * far-air phi for a vertex phi field, 0 for every other). A field of another class or format, or a
 * texture that is not a whole block grid of its class's extent, throws; a
 * field created later throws at the next encode. A vertex block is written
 * whole, its low skirt and high planes included, whichever of them hold a
 * vertex now: a vertex's slot moves between a page's high plane and its
 * neighbours' skirts as pages are claimed and retired. A claim makes no
 * copy: a vertex that enters a claimed block reads its AIR image. The texel
 * it left (a neighbour's skirt or high plane) keeps whatever a kernel last
 * stored there while it held the vertex, so a retire hands every vertex the
 * retired block held to the texel that holds it next, as a BULK16 demote
 * does (uniform-bulk-page-transfer.ts). Without that handoff the vertex
 * would read the stale value: on the 128^3 dam break, a column page's
 * skirt kept the deep phi (-0.117 m) a demote had handed it at t = 0, and
 * when the page below retired at about step 180, that phi surfaced as
 * liquid in far air, outside every resident page (the census's absent-near
 * residency audit).
 *
 * Launch: one compute pass; one pipeline per four fields (two storage
 * textures a field: its h store, or a 1^3 stand-in while it has none, and its
 * base), each dispatched slots*chunks workgroups of 64 lanes whatever the
 * count. Workgroup (slot, chunk) strides the list from `slot` by `slots`; its
 * lanes are the 8x8 columns of a block and `chunk` a z slab. With nothing
 * claimed every lane reads the count and returns. encodeAll runs the same
 * pipelines over every block, the sentinels included (setup only).
 *
 * Retire handoff: a pipeline per batch of vertex fields, in the same pass
 * after the field pipelines, a workgroup per block of the commit's retired
 * list (the quarantine keeps its texels until the next commit). For each
 * texel on the block's faces, the block held the vertex before the commit
 * when no page ahead of it in the vertex's slot walk (umVertexSlot) was
 * stored before the commit: stored now and not claimed by this commit, or
 * retired by it. Its next holder is the first page of the walk stored now.
 * The handoff writes the block's texel there, in the base and, where the
 * holder has an h slot, in the h store (the field's AIR where the retired
 * block had none). It writes nothing to a holder this commit claimed (the
 * AIR image of a claim stands, as above) nor for a block a demote retired
 * (the demote handed its vertices before the commit). Two retired blocks
 * never write one texel: a vertex has one holder before the commit.
 *
 * Per-key buffers of other owners are written by those owners: each
 * implements UniformPageInitBlocks over an entry built with
 * uniformPageInitBlocksWGSL, and encode calls them in the order given.
 * UniformPageTileWords is that hook for the tile words of the topologies. */
import { uniformDetailField, type UniformDetailGroup, type UniformDetailStorage } from "./uniform-detail-fields";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { UNIFORM_PAGE_POOL_RECORD, UNIFORM_PAGE_POOL_STATUS, uniformPagePoolWGSL, type UniformPagePool, type UniformPagePoolLayout } from "./uniform-page-pool";

/** Keys of one block in each dense key space: key = keys*block + local. */
export const UNIFORM_PAGE_INIT_KEYS={tile:512,cell:32768,vertex:35937,lowFace:3072} as const;
/** The tile word of a key with no owner: 4h, the reserved invalid owner. */
export const UNIFORM_PAGE_INIT_TILE_WORD=0x3fffffff;
/** Claim slots of every initialise launch: the list is strided by this. */
export function uniformPageInitSlots(layout:UniformPagePoolLayout):number{return Math.min(32,layout.requests);}
/** Workgroups an entry of uniformPageInitBlocksWGSL is dispatched with (x). */
export function uniformPageInitGroups(layout:UniformPagePoolLayout,chunks=1):number{return uniformPageInitSlots(layout)*chunks;}

/** An owner of per-key buffers: its launches write the AIR image over the
 * keys of the blocks the last commit claimed. Fixed direct dispatches
 * (uniformPageInitGroups) compiled at setup; nothing allocated, nothing read back. */
export interface UniformPageInitBlocks{
 readonly label:string;
 encodeInitBlocks(encoder:GPUCommandEncoder):void;
}
/** The entry an owner adds to a module that includes uniformPagePoolWGSL of
 * the same layout: `body` runs once for every key of every new block with
 * `block`, `local` (the key within its block, below `keys`) and
 * `key` = keys*block+local in scope. It must not return. Dispatch
 * uniformPageInitGroups(layout, chunks) workgroups; a lane takes
 * keys/(64*chunks) keys of each block its slot strides to. */
export function uniformPageInitBlocksWGSL(layout:UniformPagePoolLayout,entry:string,keys:number,body:string,chunks=1):string{
 if(!Number.isSafeInteger(keys)||keys<1||!Number.isSafeInteger(chunks)||chunks<1)throw new RangeError(`Uniform page initialise entry ${entry}: ${keys} keys in ${chunks} chunks`);
 return /* wgsl */`
@compute @workgroup_size(64) fn ${entry}(@builtin(workgroup_id) piGroup:vec3u,@builtin(local_invocation_index) piLane:u32){
 let piCount=upNewCount();
 for(var piAt=piGroup.x/${chunks}u;piAt<piCount;piAt+=${uniformPageInitSlots(layout)}u){
  let block=upNewBlock(piAt);
  for(var local=piLane+64u*(piGroup.x%${chunks}u);local<${keys}u;local+=${64*chunks}u){let key=${keys}u*block+local;${body}}
 }
}`;
}

/** A class's texels of one block: `extent` texels an axis; a lane (an 8x8
 * column index) takes `step` texels of x and of y, a chunk `depth`/chunks of
 * z; with a skirt the last lane of an axis and the last chunk take one more. */
export interface UniformPageInitShape{readonly extent:readonly [number,number,number];readonly step:number;readonly depth:number;readonly skirt:boolean}
export const UNIFORM_PAGE_INIT_SHAPES:Readonly<Record<"cell"|"face"|"vertex",{readonly h:UniformPageInitShape;readonly base:UniformPageInitShape}>>={
 cell:{h:{extent:[32,32,32],step:4,depth:32,skirt:false},base:{extent:[8,8,8],step:1,depth:8,skirt:false}},
 face:{h:{extent:[32,32,32],step:4,depth:32,skirt:false},base:{extent:[8,8,32],step:1,depth:32,skirt:false}},
 vertex:{h:{extent:[33,33,33],step:4,depth:32,skirt:true},base:{extent:[9,9,9],step:1,depth:8,skirt:true}},
};
export interface UniformPageInitialiseField{readonly kind:"cell"|"face"|"vertex";readonly value:number}
/** A low-face-keyed table: `offset` its first word in its buffer, `value` the f32 every row of a new block takes. */
export interface UniformPageInitialiseRows{readonly offset:number;readonly value:number}
const PAGE_INIT_CHUNKS:readonly number[]=[1,2,4,8,16,32];
const F32=new Float32Array(1),U32=new Uint32Array(F32.buffer);
const literal=(value:number)=>{F32[0]=value;if(!Number.isFinite(F32[0]!))throw new RangeError(`Uniform page initialise: AIR value ${value} is not a finite f32`);return U32[0]===0?"0.0":`bitcast<f32>(${U32[0]}u)`;};
const bits=(value:number)=>{literal(value);return U32[0]!;};

/** One field pipeline's module: group 0 the pool state, the mode and the
 * row buffers; group 1 two write-only storage textures a field (2i its h
 * store, 2i+1 its base). */
export function uniformPageInitialiseWGSL(layout:UniformPagePoolLayout,blockBits:number,chunks:number,fields:readonly UniformPageInitialiseField[],rows:readonly UniformPageInitialiseRows[]):string{
 if(!PAGE_INIT_CHUNKS.includes(chunks))throw new RangeError(`Uniform page initialise chunks must be 1, 2, 4, 8, 16 or 32: ${chunks}`);
 const format=(f:UniformPageInitialiseField)=>f.kind==="face"?"rgba32float":"r32float";
 // A shape shallower than the chunks takes one z plane a chunk; the chunks past its depth idle.
 const fill=(name:string,s:UniformPageInitShape,value:string,g="g")=>{
  const z=Math.max(1,s.depth/chunks),slabs=s.depth/z,extra=s.skirt?`+vec3u(vec2u(l.xy==vec2u(7u)),u32(chunk==${slabs-1}u))`:"";
  return `${slabs<chunks?`if(chunk<${slabs}u)`:""}{let o=${g}*vec3u(${s.extent.map(n=>`${n}u`).join(",")});let a=vec3u(${s.step}u*l.xy,${z}u*chunk);let b=a+vec3u(${s.step}u,${s.step}u,${z}u)${extra};
   for(var z=a.z;z<b.z;z++){for(var y=a.y;y<b.y;y++){for(var x=a.x;x<b.x;x++){textureStore(${name},vec3i(o+vec3u(x,y,z)),${value});}}}}`;
 };
 return /* wgsl */`${uniformPagePoolWGSL(layout,0,0)}
// x: 0 the blocks the last commit claimed, 1 every block.
@group(0) @binding(1) var<uniform> piMode:vec4u;
${rows.map((_,j)=>`@group(0) @binding(${2+j}) var<storage,read_write> piRows${j}:array<u32>;`).join("\n")}
// The detail storage's h slots (slotState): an h store places a block at its slot, none at slot 0 but the AIR block's own.
@group(0) @binding(${2+rows.length}) var<storage,read> piSlots:array<u32>;
${fields.map((f,i)=>`@group(1) @binding(${2*i}) var piH${i}:texture_storage_3d<${format(f)},write>;@group(1) @binding(${2*i+1}) var piBase${i}:texture_storage_3d<${format(f)},write>;`).join("\n")}
const PI_SLOTS:u32=${uniformPageInitSlots(layout)}u;const PI_CHUNKS:u32=${chunks}u;const PI_BITS:u32=${blockBits}u;
fn piBlockCoord(b:u32)->vec3u{let m=(1u<<PI_BITS)-1u;return vec3u(b&m,(b>>PI_BITS)&m,b>>(2u*PI_BITS));}
@compute @workgroup_size(8,8,1) fn piFields(@builtin(workgroup_id) w:vec3u,@builtin(local_invocation_id) l:vec3u,@builtin(local_invocation_index) lane:u32){
 let all=piMode.x!=0u;var count=upNewCount();if(all){count=UP_BLOCKS;}
 let chunk=w.x%PI_CHUNKS;
 for(var i=w.x/PI_CHUNKS;i<count;i+=PI_SLOTS){
  var block=i;if(!all){block=upNewBlock(i);}
  let g=piBlockCoord(block);let slot=piSlots[block];let gh=piBlockCoord(slot);let held=slot!=0u||block==0u;
${fields.map((f,i)=>{const s=UNIFORM_PAGE_INIT_SHAPES[f.kind],v=literal(f.value),value=f.kind==="face"?`vec4f(${v})`:`vec4f(${v},0.0,0.0,0.0)`;
  return `  // A field with no h store binds a 1^3 stand-in there.\n  if(textureDimensions(piH${i}).x>1u&&held)${fill(`piH${i}`,s.h,value,"gh")}\n  ${fill(`piBase${i}`,s.base,value)}`;}).join("\n")}
${rows.map((r,j)=>`  for(var r=lane+64u*chunk;r<${UNIFORM_PAGE_INIT_KEYS.lowFace}u;r+=${64*chunks}u){piRows${j}[${r.offset}u+${UNIFORM_PAGE_INIT_KEYS.lowFace}u*block+r]=${bits(r.value)}u;}`).join("\n")}
 }
}`;
}

/** The retire handoff's rule (prNext) on the host: the texel that holds,
 * after the commit, vertex `l` of the block that held page `page` (world
 * pages, `bits` 5 for the h lattice or 3 for the 4h one, l 0..2^bits on
 * each axis), or undefined where that block did not hold the vertex before
 * the commit (`before(q)`: page q was stored before it), no page holds it
 * now (`now(q)`: the block that stores q, or -1) or the holder is a block
 * this commit claimed (`fresh(b)`). */
export function uniformPageRetireNext(page:readonly number[],l:readonly number[],bits:number,before:(q:readonly number[])=>boolean,now:(q:readonly number[])=>number,fresh:(block:number)=>boolean):{readonly block:number;readonly local:readonly number[]}|undefined{
 const mask=(1<<bits)-1,v=[0,1,2].map(a=>(page[a]!<<bits)+l[a]!),own=v.map(x=>(x-1)>>bits),local=v.map(x=>((x-1)&mask)+1),plane=v.map(x=>(x&mask)===0);
 const walk=(k:number)=>[0,1,2].some(a=>((k>>a)&1)!==0&&!plane[a])?undefined:own.map((q,a)=>q+((k>>a)&1));
 for(let k=0;k<8;k++){
  const q=walk(k);if(!q)continue;
  if(q.every((x,a)=>x===page[a]))break;
  if(before(q))return undefined;
 }
 for(let k=0;k<8;k++){
  const q=walk(k);if(!q)continue;
  const block=now(q);if(block<0)continue;
  return fresh(block)?undefined:{block,local:local.map((x,a)=>((k>>a)&1)?0:x)};
 }
 return undefined;
}

/** The retire handoff's module (see the header): group 0 the pool state and
 * the h slots, group 1 two read-write storage textures a vertex field (2i
 * its h store, 2i+1 its base). */
export function uniformPageRetireWGSL(layout:UniformPagePoolLayout,blockBits:number,fields:readonly UniformPageInitialiseField[]):string{
 if(fields.some(f=>f.kind!=="vertex"))throw new Error("Uniform page retire handoff: only vertex fields move with residency");
 return /* wgsl */`${uniformPagePoolWGSL(layout,0,0)}
@group(0) @binding(1) var<storage,read> prSlots:array<u32>;
${fields.map((_,i)=>`@group(1) @binding(${2*i}) var prH${i}:texture_storage_3d<r32float,read_write>;@group(1) @binding(${2*i+1}) var prBase${i}:texture_storage_3d<r32float,read_write>;`).join("\n")}
const PR_SLOTS:u32=${uniformPageInitSlots(layout)}u;const PR_BITS:u32=${blockBits}u;
fn prBlockCoord(b:u32)->vec3i{let m=(1u<<PR_BITS)-1u;return vec3i(vec3u(b&m,(b>>PR_BITS)&m,b>>(2u*PR_BITS)));}
// Claimed by this commit.
fn prFresh(b:u32)->bool{return upState[UP_AT_RECORDS+8u*b+${UNIFORM_PAGE_POOL_RECORD.generation}u]==upGeneration();}
// Bit of page + d (d in -1..1 an axis) in the 3^3 mask around a retired block's page.
fn prBit(d:vec3i)->u32{let e=vec3u(d+vec3i(1));return 1u<<(e.x+3u*e.y+9u*e.z);}
// uniformPageRetireNext: xyz the holder's local texel, w the holder, UP_NONE for none.
fn prNext(page:vec3i,l:vec3u,bits:u32,before:u32)->vec4u{
 let mask=(1<<bits)-1;let v=(page<<vec3u(bits))+vec3i(l);let c=v-vec3i(1);
 let own=c>>vec3u(bits);let local=vec3u(c&vec3i(mask))+vec3u(1u);let plane=(v&vec3i(mask))==vec3i(0);
 for(var k=0u;k<8u;k++){
  let step=vec3<bool>((k&1u)!=0u,(k&2u)!=0u,(k&4u)!=0u);if(any(step&!plane)){continue;}
  let q=own+select(vec3i(0),vec3i(1),step);
  if(all(q==page)){break;}
  if((before&prBit(q-page))!=0u){return vec4u(0u,0u,0u,UP_NONE);}
 }
 for(var k=0u;k<8u;k++){
  let step=vec3<bool>((k&1u)!=0u,(k&2u)!=0u,(k&4u)!=0u);if(any(step&!plane)){continue;}
  let b=upBlockOf(own+select(vec3i(0),vec3i(1),step));if(b==UP_NONE){continue;}
  if(prFresh(b)){break;}
  return vec4u(select(local,vec3u(0u),step),b);
 }
 return vec4u(0u,0u,0u,UP_NONE);
}
var<workgroup> prBefore:atomic<u32>;var<workgroup> prShared:u32;
@compute @workgroup_size(64) fn prRetired(@builtin(workgroup_id) w:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){prShared=upState[${UNIFORM_PAGE_POOL_STATUS.pending}u];}
 let count=workgroupUniformLoad(&prShared);
 for(var i=w.x;i<count;i+=PR_SLOTS){
  if(lane==0u){atomicStore(&prBefore,0u);}
  workgroupBarrier();
  let b=upState[UP_AT_RETIRED+i];let page=upPageOf(b);
  // The 3^3 pages around it that were stored before the commit: stored now
  // and not claimed by it, or retired by it (the list, all lanes).
  if(lane<27u){let n=upBlockOf(page+vec3i(i32(lane%3u),i32((lane/3u)%3u),i32(lane/9u))-vec3i(1));if(n!=UP_NONE&&!prFresh(n)){atomicOr(&prBefore,1u<<lane);}}
  for(var j=lane;j<count;j+=64u){let d=upPageOf(upState[UP_AT_RETIRED+j])-page;if(all(abs(d)<=vec3i(1))){atomicOr(&prBefore,prBit(d));}}
  workgroupBarrier();
  let before=atomicLoad(&prBefore);
  // A demote left the page its BULK16 word, and handed its vertices before the commit.
  if(!upBulkWord(upWord(page))){
   // The base: 9^3 texels, those on the block's faces (an interior vertex has no other holder).
   for(var t=lane;t<729u;t+=64u){
    let l=vec3u(t%9u,(t/9u)%9u,t/81u);if(all(l>=vec3u(1u))&&all(l<=vec3u(7u))){continue;}
    let next=prNext(page,l,3u,before);if(next.w==UP_NONE){continue;}
    let src=prBlockCoord(b)*9+vec3i(l);let to=prBlockCoord(next.w)*9+vec3i(next.xyz);
${fields.map((_,i)=>`    textureStore(prBase${i},to,textureLoad(prBase${i},src));`).join("\n")}
   }
   // The h store (absent: a 1^3 stand-in), 33^3 texels at each block's h slot.
   if(${fields.map((_,i)=>`textureDimensions(prH${i}).x>1u`).join("||")}){
    let held=prSlots[b];
    for(var t=lane;t<35937u;t+=64u){
     let l=vec3u(t%33u,(t/33u)%33u,t/1089u);if(all(l>=vec3u(1u))&&all(l<=vec3u(31u))){continue;}
     let next=prNext(page,l,5u,before);if(next.w==UP_NONE){continue;}
     let slot=prSlots[next.w];if(slot==0u){continue;}
     let at=prBlockCoord(held)*33+vec3i(l);let to=prBlockCoord(slot)*33+vec3i(next.xyz);
${fields.map((f,i)=>`     if(textureDimensions(prH${i}).x>1u){textureStore(prH${i},to,select(vec4f(${literal(f.value)},0.0,0.0,0.0),textureLoad(prH${i},at),held!=0u));}`).join("\n")}
    }
   }
  }
  workgroupBarrier();
 }
}`;
}

async function compile(device:GPUDevice,label:string,code:string):Promise<GPUShaderModule>{
 const module=device.createShaderModule({label,code});
 const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
 if(errors.length)throw new Error(`${label}\n${errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n")}`);
 return module;
}

/** A low-face-keyed buffer the module writes (3072 rows a block). */
export interface UniformPageInitialiseLowFaces{
 readonly buffer:GPUBuffer;
 /** The table's first word in the buffer (default 0). */
 readonly offset?:number;
 /** The f32 AIR value of a row (default 0). */
 readonly value?:number;
}
export interface UniformPageInitialiseOptions{
 /** The census's residency-near bound on a 4h owner's corner phi (16*MAX_H):
  * a vertex field's AIR below it would seed residency from a claimed block. */
 readonly near:number;
 /** Every low-face-keyed buffer, STORAGE usage. */
 readonly lowFaces?:readonly UniformPageInitialiseLowFaces[];
 /** The owners of per-key buffers, in launch order. */
 readonly owners?:readonly UniformPageInitBlocks[];
 /** z slabs of a block, each a workgroup of a claim slot: 1, 2, 4, 8, 16 or 32 (default 32, a z plane of
  * the h store each: a claimed block's texels spread over 32 workgroups, not 4). */
 readonly chunks?:number;
}
interface Field extends UniformPageInitialiseField{readonly handle:GPUTexture}
interface Batch{readonly fields:readonly Field[];readonly rows:readonly Required<UniformPageInitialiseLowFaces>[];readonly held:UniformDetailGroup}

export class UniformPageInitialise{
 /** The storage's fields at construction, in its order. */
 readonly fields:readonly Field[];
 /** Workgroups of each field pipeline's dispatch. */
 readonly groups:number;
 /** Field pipelines: launches of encode before the owners'. */
 get passes():number{return this.batches.length;}
 private readonly batches:Batch[]=[];
 private readonly owners:readonly UniformPageInitBlocks[];
 private readonly chunks:number;
 private readonly mode:GPUBuffer[]=[];
 private pipelines?:GPUComputePipeline[];
 /** Group 0 of each pipeline over the new list, and over every block. */
 private listed:GPUBindGroup[]=[];
 private every:GPUBindGroup[]=[];
 private textureLayouts:GPUBindGroupLayout[]=[];
 private static readonly PASS:GPUComputePassDescriptor={label:"Uniform page initialise"};
 constructor(private readonly device:GPUDevice,private readonly pool:UniformPagePool,private readonly storage:UniformDetailStorage,options:UniformPageInitialiseOptions){
  const a=storage.layout.address,l=pool.layout,K=UNIFORM_PAGE_INIT_KEYS;
  if(a.blocks!==l.blocks)throw new Error(`Uniform page initialise: the detail storage holds ${a.blocks} blocks, the pool ${l.blocks}`);
  this.chunks=options.chunks??32;if(!PAGE_INIT_CHUNKS.includes(this.chunks))throw new RangeError(`Uniform page initialise chunks must be 1, 2, 4, 8, 16 or 32: ${this.chunks}`);
  this.owners=[...(options.owners??[])];this.groups=uniformPageInitSlots(l)*this.chunks;
  this.fields=storage.fieldTextures.map(handle=>{
   const kind=storage.kindOf(handle);
   if(kind!=="cell"&&kind!=="face"&&kind!=="vertex")throw new Error(`Uniform page initialise: ${handle.label} is of class ${kind}, which has no block shape here`);
   if(handle.format!==(kind==="face"?"rgba32float":"r32float"))throw new Error(`Uniform page initialise: ${handle.label} is a ${kind} field of format ${handle.format}`);
   const value=uniformDetailField(handle)!.empty;literal(value);
   // A vertex whose slot moves into a claimed block (off a neighbour's skirt) or out of a retired one reads far air on both sides only if AIR is far.
   if(kind==="vertex"&&!(value>=options.near))throw new RangeError(`Uniform page initialise: ${handle.label} takes AIR ${value}, nearer than the residency bound ${options.near}`);
   return {handle,kind,value};
  });
  const rows=(options.lowFaces??[]).map(({buffer,offset=0,value=0},i)=>{
   if(!Number.isSafeInteger(offset)||offset<0||4*(offset+K.lowFace*l.blocks)>buffer.size)throw new RangeError(`Uniform page initialise: low-face buffer ${i} (${buffer.label}) of ${buffer.size} bytes does not hold ${K.lowFace*l.blocks} rows at word ${offset}`);
   if(!(buffer.usage&GPUBufferUsage.STORAGE))throw new Error(`Uniform page initialise: low-face buffer ${i} (${buffer.label}) needs STORAGE usage`);
   if((options.lowFaces??[]).findIndex(o=>o.buffer===buffer)!==i)throw new Error(`Uniform page initialise: low-face buffer ${i} (${buffer.label}) is listed twice`);
   literal(value);return {buffer,offset,value};
  });
  // Two storage textures a field; the pool state beside the row buffers.
  const per=Math.floor(device.limits.maxStorageTexturesPerShaderStage/2),most=device.limits.maxStorageBuffersPerShaderStage-2;
  if(per<1||most<1)throw new RangeError("Uniform page initialise: the device binds no storage texture pair or no second storage buffer in a stage");
  const count=Math.max(Math.ceil(this.fields.length/per),Math.ceil(rows.length/most));
  for(let i=0;i<count;i++)this.batches.push({fields:this.fields.slice(per*i,per*i+per),rows:rows.slice(most*i,most*i+most),held:{} as UniformDetailGroup});
  const vertices=this.fields.filter(f=>f.kind==="vertex");
  for(let i=0;i<vertices.length;i+=per)this.retires.push({fields:vertices.slice(i,i+per),rows:[],held:{} as UniformDetailGroup});
 }
 /** The retire handoff's batches: the vertex fields, two read-write textures each. */
 private readonly retires:Batch[]=[];
 private retireLayouts:GPUBindGroupLayout[]=[];
 private retireGroups:GPUBindGroup[]=[];
 private retirePipelines:GPUComputePipeline[]=[];
 /** The texture group of a batch over the fields' textures as they are now. */
 private bind(i:number,retire=false):GPUBindGroup{
  const batch=retire?this.retires[i]!:this.batches[i]!,layout=retire?this.retireLayouts[i]!:this.textureLayouts[i]!;
  const s=this.storage,G=s.layout.address.blockGrid;
  // An h store's depth is its slots' (whole blocks; the block grid's while every block is its own slot).
  const shaped=(t:GPUTexture,e:readonly [number,number,number],what:string,slots=false)=>{
   if(t.width!==e[0]*G[0]||t.height!==e[1]*G[1]||(slots?t.depthOrArrayLayers%e[2]!==0||t.depthOrArrayLayers>e[2]*G[2]:t.depthOrArrayLayers!==e[2]*G[2]))throw new Error(`Uniform page initialise: ${t.label} ${what} is ${t.width}x${t.height}x${t.depthOrArrayLayers}, not ${e.join("x")} texels for each of ${G.join("x")} blocks`);
   return t;
  };
  return this.device.createBindGroup({label:`Uniform page ${retire?"retire":"initialise"} fields ${i}`,layout,entries:batch.fields.flatMap((f,k)=>{
   const shape=UNIFORM_PAGE_INIT_SHAPES[f.kind],base=s.baseOf(f.handle),h=s.physical(f.handle);
   if(!base)throw new Error(`Uniform page initialise: ${f.handle.label} has no base block`);
   // A per-tile field, and every field while no h generation is resident, is its base alone.
   return [{binding:2*k,resource:(h===base?s.stub(f.handle):shaped(h,shape.h.extent,"h store",true)).createView()},{binding:2*k+1,resource:shaped(base,shape.base.extent,"base").createView()}];
  })});
 }
 /** Compile every pipeline encode and encodeAll launch, and bind them. */
 async initialize():Promise<void>{
  if(this.started)throw new Error("Uniform page initialise is already initialized");
  this.started=true;
  const device=this.device,l=this.pool.layout,bits=this.storage.layout.address.blockBits;
  for(const all of [0,1]){const m=device.createBuffer({label:`Uniform page initialise mode ${all}`,size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(m,0,new Uint32Array([all,0,0,0]));this.mode.push(m);}
  const pipelines=await Promise.all(this.batches.map(async(b,i)=>{
   const module=await compile(device,`Uniform page initialise ${i}`,uniformPageInitialiseWGSL(l,bits,this.chunks,b.fields,b.rows));
   const buffers=device.createBindGroupLayout({label:`Uniform page initialise buffers ${i}`,entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
    ...b.rows.map((_,j)=>({binding:2+j,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),{binding:2+b.rows.length,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage" as const}}]});
   const textures=this.textureLayouts[i]=device.createBindGroupLayout({label:`Uniform page initialise fields ${i}`,entries:b.fields.flatMap((f,k)=>[0,1].map(t=>({binding:2*k+t,visibility:GPUShaderStage.COMPUTE,
    storageTexture:{access:"write-only" as const,format:(f.kind==="face"?"rgba32float":"r32float") as GPUTextureFormat,viewDimension:"3d" as const}})))});
   const group=(mode:GPUBuffer)=>device.createBindGroup({layout:buffers,entries:[{binding:0,resource:{buffer:this.pool.state}},{binding:1,resource:{buffer:mode}},...b.rows.map((r,j)=>({binding:2+j,resource:{buffer:r.buffer}})),{binding:2+b.rows.length,resource:{buffer:this.storage.slotState,size:4*l.blocks}}]});
   this.listed[i]=group(this.mode[0]!);this.every[i]=group(this.mode[1]!);
   // The storage rebuilds the texture group when it replaces a field's textures (admit, retire, an adopted base).
   let last:GPUBindGroup|undefined;const build=(base?:boolean)=>base&&last?last:last=this.bind(i);
   b.held.group=b.held.base=build();this.storage.track(b.held,build);
   return device.createComputePipelineAsync({label:`Uniform page initialise ${i}`,layout:device.createPipelineLayout({bindGroupLayouts:[buffers,textures]}),compute:{module,entryPoint:"piFields"}});
  }));
  this.pipelines=pipelines;
  // The retire handoff: the pool state and the h slots, then each vertex field's h store and base, read-write.
  const state=device.createBindGroupLayout({label:"Uniform page retire state",entries:[0,1].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage" as const}}))});
  const stateGroup=device.createBindGroup({label:"Uniform page retire state",layout:state,entries:[{binding:0,resource:{buffer:this.pool.state}},{binding:1,resource:{buffer:this.storage.slotState,size:4*l.blocks}}]});
  this.retirePipelines=await Promise.all(this.retires.map(async(b,i)=>{
   const module=await compile(device,`Uniform page retire ${i}`,uniformPageRetireWGSL(l,bits,b.fields));
   const textures=this.retireLayouts[i]=device.createBindGroupLayout({label:`Uniform page retire fields ${i}`,entries:b.fields.flatMap((_,k)=>[0,1].map(t=>({binding:2*k+t,visibility:GPUShaderStage.COMPUTE,
    storageTexture:{access:"read-write" as const,format:"r32float" as GPUTextureFormat,viewDimension:"3d" as const}})))});
   this.retireGroups[i]=stateGroup;
   let last:GPUBindGroup|undefined;const build=(base?:boolean)=>base&&last?last:last=this.bind(i,true);
   b.held.group=b.held.base=build();this.storage.track(b.held,build);
   return device.createComputePipelineAsync({label:`Uniform page retire ${i}`,layout:device.createPipelineLayout({bindGroupLayouts:[state,textures]}),compute:{module,entryPoint:"prRetired"}});
  }));
 }
 private started=false;
 private launch(encoder:GPUCommandEncoder,buffers:readonly GPUBindGroup[]):void{
  const pipelines=this.pipelines;
  if(!pipelines)throw new Error("Uniform page initialise is not initialized");
  if(this.storage.fieldTextures.length!==this.fields.length)throw new Error(`Uniform page initialise: the detail storage holds ${this.storage.fieldTextures.length} fields, ${this.fields.length} when the initialise was built`);
  const pass=encoder.beginComputePass(UniformPageInitialise.PASS);
  for(let i=0;i<pipelines.length;i++){pass.setPipeline(pipelines[i]!);pass.setBindGroup(0,buffers[i]!);pass.setBindGroup(1,this.batches[i]!.held.group);pass.dispatchWorkgroups(this.groups);}
  // After the claims' AIR image: a retired block's vertices pass to their next holders (none at setup).
  if(buffers===this.listed)for(let i=0;i<this.retirePipelines.length;i++){pass.setPipeline(this.retirePipelines[i]!);pass.setBindGroup(0,this.retireGroups[i]!);pass.setBindGroup(1,this.retires[i]!.held.group);pass.dispatchWorkgroups(uniformPageInitSlots(this.pool.layout));}
  pass.end();
 }
 /** After a commit, before any reader of its new blocks: the AIR image over
  * every field texel and low-face row of each, the retire handoff of the
  * blocks it retired, then every owner's hook. */
 encode(encoder:GPUCommandEncoder):void{
  this.launch(encoder,this.listed);
  for(let i=0;i<this.owners.length;i++)this.owners[i]!.encodeInitBlocks(encoder);
 }
 /** Setup, before the scene's uploads: the AIR image over every block the
  * fields and low-face buffers hold, sentinels and free blocks included.
  * The owners' buffers are not touched. */
 encodeAll(encoder:GPUCommandEncoder):void{this.launch(encoder,this.every);}
 destroy():void{for(const m of this.mode)m.destroy();}
}

/** A topology buffer of n tile keys: tile words [0,n), the stencil pair of key t at 2n+2t, its blend word at 4n+t. */
export interface UniformPageTopology{readonly buffer:GPUBuffer;readonly tiles:number}
/** An ownership's held topology (the builder's `current`). */
export function uniformPageTopology(ownership:UniformMixedOwnership):UniformPageTopology{
 const c=ownership.capacity;
 if(c.metadataBytes!==20*c.tiles)throw new Error(`Uniform page initialise: a topology of ${c.tiles} tile keys holds ${c.metadataBytes} metadata bytes, not five words a key`);
 return {buffer:ownership.presentation.buffer,tiles:c.tiles};
}
export function uniformPageTileWordsWGSL(layout:UniformPagePoolLayout,tiles:readonly number[]):string{
 return /* wgsl */`${uniformPagePoolWGSL(layout,0,0)}
${tiles.map((_,i)=>`@group(0) @binding(${1+i}) var<storage,read_write> piTopology${i}:array<u32>;`).join("\n")}
${uniformPageInitBlocksWGSL(layout,"piTileWords",UNIFORM_PAGE_INIT_KEYS.tile,tiles.map((n,i)=>`piTopology${i}[key]=${UNIFORM_PAGE_INIT_TILE_WORD}u;piTopology${i}[${2*n}u+2u*key]=0u;piTopology${i}[${2*n+1}u+2u*key]=0u;piTopology${i}[${4*n}u+key]=0u;`).join(""))}`;
}
/** The tile words of new blocks, in every topology given: each of a block's
 * 512 keys becomes a 4h tile of the invalid owner with zero stencil and
 * blend words (no h neighbour, no valid neighbour, outside the ring), as a
 * key no page holds reads. The builder's next generation then finds a
 * claimed block 4h. */
export class UniformPageTileWords implements UniformPageInitBlocks{
 readonly label="Uniform page tile words";
 private pipeline?:GPUComputePipeline;
 private group?:GPUBindGroup;
 private readonly groups:number;
 private static readonly PASS:GPUComputePassDescriptor={label:"Uniform page tile words"};
 constructor(private readonly device:GPUDevice,private readonly pool:UniformPagePool,private readonly topologies:readonly UniformPageTopology[]){
  const l=pool.layout,keys=UNIFORM_PAGE_INIT_KEYS.tile*l.blocks;
  if(!topologies.length||topologies.length>device.limits.maxStorageBuffersPerShaderStage-1)throw new RangeError(`Uniform page tile words: ${topologies.length} topologies beside the pool state`);
  topologies.forEach((t,i)=>{
   if(t.tiles!==keys)throw new Error(`Uniform page tile words: topology ${i} has ${t.tiles} tile keys, the pool's ${l.blocks} blocks ${keys}`);
   if(t.buffer.size<20*t.tiles||!(t.buffer.usage&GPUBufferUsage.STORAGE))throw new Error(`Uniform page tile words: topology ${i} (${t.buffer.label}) needs five words a key and STORAGE usage`);
   if(topologies.findIndex(o=>o.buffer===t.buffer)!==i)throw new Error(`Uniform page tile words: topology ${i} (${t.buffer.label}) is listed twice`);
  });
  this.groups=uniformPageInitGroups(l);
 }
 async initialize():Promise<void>{
  if(this.pipeline)throw new Error("Uniform page tile words is already initialized");
  const device=this.device,module=await compile(device,this.label,uniformPageTileWordsWGSL(this.pool.layout,this.topologies.map(t=>t.tiles)));
  const layout=device.createBindGroupLayout({label:this.label,entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},...this.topologies.map((_,i)=>({binding:1+i,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}}))]});
  this.group=device.createBindGroup({label:this.label,layout,entries:[{binding:0,resource:{buffer:this.pool.state}},...this.topologies.map((t,i)=>({binding:1+i,resource:{buffer:t.buffer}}))]});
  this.pipeline=await device.createComputePipelineAsync({label:this.label,layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:"piTileWords"}});
 }
 encodeInitBlocks(encoder:GPUCommandEncoder):void{
  if(!this.pipeline)throw new Error("Uniform page tile words is not initialized");
  const pass=encoder.beginComputePass(UniformPageTileWords.PASS);
  pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.group!);pass.dispatchWorkgroups(this.groups);
  pass.end();
 }
}
