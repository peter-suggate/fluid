/** H slots: where the detail storage's h stores place each block
 * (umFineSlot, uniform-lattice-address.ts). Block-granular residency, the
 * first stage of h storage only where h exists
 * (docs/plans/tiered-adaptivity-huge-grids-2026-10-24.md §8): an h store is
 * the block grid's width and height but only as deep as its slots need, and
 * a block keeps h texels only while it holds a slot. Slot 0 is the sentinel:
 * its texels hold each field's empty value, every slotless block reads
 * there, and a store there is dropped (the Store forms).
 *
 * A block needs a slot while it holds a tile of the detail ring, or the
 * texel of a vertex of one (a ring tile on a page's low face has corners in
 * the skirt or the high plane of the page below: umVertexSlot of its eight
 * corners). Slots are claimed from a free stack, fail fast when it is empty
 * (UNIFORM_DETAIL_SLOT_VIOLATION in the detail violation word), and are
 * released once the adopted generation no longer needs them.
 *
 * State (u32 words, UniformFineSlotWords): the slot of every block, its need
 * mark, the free count and stack, the claimed count and list (the blocks a
 * claim gave a slot: init clears their slots to each field's empty value). */

/** The detail violation bit of a claim the free stack cannot meet. */
export const UNIFORM_DETAIL_SLOT_VIOLATION=8192;
/** The default capacity over the t=0 need, and its floor (fluid.tiers.fineBlocks overrides both). */
export const UNIFORM_FINE_SLOT_GROWTH=1.5,UNIFORM_FINE_SLOT_FLOOR=16;
/** Under a coarse start the default also holds the h ceiling's tiles at this
 * many a slot: a ring is sparse in its blocks (letters at 4273 h tiles held
 * 98 blocks, 44 a block; the seep's jet 40 at 500 tiles, 12 a block). */
export const UNIFORM_FINE_SLOT_TILES=32;
/** Workgroups an init launch strides the claimed list with. */
const INIT_GROUPS=32;

export interface UniformFineSlotWords{readonly slot:number;readonly need:number;readonly free:number;readonly stack:number;readonly claimed:number;readonly list:number;readonly words:number}
export function uniformFineSlotWords(blocks:number,capacity:number):UniformFineSlotWords{
 const free=2*blocks,claimed=free+1+capacity;
 return {slot:0,need:blocks,free,stack:free+1,claimed,list:claimed+1,words:claimed+1+capacity};
}
/** Blocks of the slot grid on z: slot s sits at umBlockCoord(s), the block grid's width and height a layer. */
/** The scene's compaction (fluid.tiers.fineBlocks): a coarse start (tiers.start "4h") compacts at "auto" unless told otherwise; undefined keeps the whole-grid stores. */
export function uniformFineBlocks(tiers?:{readonly fineBlocks?:number|"auto";readonly start?:"h"|"4h"}):number|"auto"|undefined{
 return tiers?.fineBlocks??(tiers?.start==="4h"?"auto":undefined);
}
export function uniformFineSlotLayers(blockGrid:readonly number[],capacity:number):number{return Math.ceil((capacity+1)/(blockGrid[0]!*blockGrid[1]!));}
/** Slots for a need: requested (fluid.tiers.fineBlocks) or the need x 1.5 with
 * the floor and at least `least` (a coarse start's ceiling share), rounded up
 * to a whole layer (the texture holds it anyway), and no more than the blocks
 * that can hold a tile (every block but the two sentinels). */
export function uniformFineSlotCapacity(blocks:number,blockGrid:readonly number[],need:number,requested?:number,least=0):number{
 if(requested!==undefined&&!(Number.isSafeInteger(requested)&&requested>=1))throw new RangeError(`fluid.tiers.fineBlocks must be a positive whole number of blocks, not ${requested}`);
 const want=requested??Math.max(UNIFORM_FINE_SLOT_FLOOR,Math.ceil(UNIFORM_FINE_SLOT_GROWTH*need),least);
 if(want<need)throw new RangeError(`fluid.tiers.fineBlocks ${want} holds fewer blocks than the t=0 detail ring needs (${need})`);
 const layer=blockGrid[0]!*blockGrid[1]!;
 return Math.min(layer*uniformFineSlotLayers(blockGrid,want)-1,Math.max(1,blocks-2));
}

/** A field an init launch clears: its format, class (a vertex block is 33³, any other 32³) and empty value. */
export interface UniformFineSlotField{readonly format:"r32float"|"rgba32float";readonly vertex:boolean;readonly empty:number}

/** The init entry over up to four fields of one format (bindings 0..3) and
 * the state (binding 4): every texel of each claimed block's slot takes the
 * field's empty value. */
export function uniformFineSlotInitWGSL(fields:readonly UniformFineSlotField[],words:UniformFineSlotWords,blockBits:number):string{
 const format=fields[0]!.format;
 const value=(f:UniformFineSlotField)=>{const v=Number.isInteger(f.empty)?`${f.empty}.0`:`${f.empty}`;return format==="rgba32float"?`vec4f(${v})`:`vec4f(${v},0.0,0.0,0.0)`;};
 const fill=(edge:number,list:readonly number[])=>!list.length?"":`  for(var t=l;t<${edge**3}u;t+=256u){let p=o*${edge}u+vec3u(t%${edge}u,(t/${edge}u)%${edge}u,t/${edge*edge}u);${list.map(i=>`textureStore(fsDst${i},vec3i(p),${value(fields[i]!)});`).join("")}}\n`;
 const index=fields.map((_,i)=>i);
 return /* wgsl */`${fields.map((_,i)=>`@group(0) @binding(${i}) var fsDst${i}:texture_storage_3d<${format},write>;`).join("\n")}
@group(0) @binding(4) var<storage,read> fsState:array<u32>;
fn fsCoord(s:u32)->vec3u{let m=(1u<<${blockBits}u)-1u;return vec3u(s&m,(s>>${blockBits}u)&m,s>>${2*blockBits}u);}
@compute @workgroup_size(256) fn init(@builtin(workgroup_id) w:vec3u,@builtin(local_invocation_index) l:u32){
 let n=fsState[${words.claimed}u];
 for(var i=w.x;i<n;i+=${INIT_GROUPS}u){
  let o=fsCoord(fsState[${words.slot}u+fsState[${words.list}u+i]]);
${fill(32,index.filter(i=>!fields[i]!.vertex))}${fill(33,index.filter(i=>fields[i]!.vertex))} }
}
`;
}

/** The mark, claim and release entries. seam: the lattice address over a
 * topology buffer by name (the detail ring's: pages from its tail, the
 * dimensions from drDimensions). */
function slotsWGSL(seam:(table:string)=>string,words:UniformFineSlotWords,blocks:number,capacity:number,violation:number):string{
 return /* wgsl */`@group(0) @binding(0) var<storage,read> drHeldWords:array<u32>;
@group(0) @binding(1) var<storage,read> drNextWords:array<u32>;
@group(0) @binding(2) var<uniform> drDimensions:vec4u;
@group(0) @binding(3) var<storage,read_write> fsState:array<atomic<u32>>;
@group(0) @binding(4) var<storage,read_write> fsSupport:array<atomic<u32>>;
fn drTiles()->vec3u{return drDimensions.xyz;}
fn drCount()->u32{return 512u*drDimensions.w;}
${seam("drHeldWords")}
const FS_BLOCKS:u32=${blocks}u;const FS_CAPACITY:u32=${capacity}u;
// A ring tile of the next generation marks its block and the blocks holding its corners.
@compute @workgroup_size(64) fn mark(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x;if(!umTileLive(t)){return;}
 if((drNextWords[2u*drCount()+2u*t+1u]&1u)==0u){return;}
 atomicStore(&fsState[${words.need}u+(t>>9u)],1u);
 let c=vec3i(umTileCoord(t));
 for(var k=0;k<8;k++){
  let s=umVertexSlot(4*(c+vec3i(k&1,(k>>1)&1,k>>2)));
  if(umBlockStored(s.w)){atomicStore(&fsState[${words.need}u+s.w],1u);}
 }
}
// A needed block with no slot pops one; an empty stack is fatal.
@compute @workgroup_size(64) fn claim(@builtin(global_invocation_id) gid:vec3u){
 let b=gid.x;if(b==0u||b>=FS_BLOCKS-1u){return;}
 if(atomicLoad(&fsState[${words.need}u+b])==0u||atomicLoad(&fsState[${words.slot}u+b])!=0u){return;}
 let f=atomicSub(&fsState[${words.free}u],1u);
 if(f==0u||f>FS_CAPACITY){atomicAdd(&fsState[${words.free}u],1u);atomicOr(&fsSupport[${violation}u],${UNIFORM_DETAIL_SLOT_VIOLATION}u);return;}
 atomicStore(&fsState[${words.slot}u+b],atomicLoad(&fsState[${words.stack}u+f-1u]));
 let i=atomicAdd(&fsState[${words.claimed}u],1u);atomicStore(&fsState[${words.list}u+i],b);
}
// A slot the adopted generation does not need returns to the stack.
@compute @workgroup_size(64) fn release(@builtin(global_invocation_id) gid:vec3u){
 let b=gid.x;if(b>=FS_BLOCKS){return;}
 let s=atomicLoad(&fsState[${words.slot}u+b]);
 if(s==0u||atomicLoad(&fsState[${words.need}u+b])!=0u){return;}
 atomicStore(&fsState[${words.slot}u+b],0u);
 let f=atomicAdd(&fsState[${words.free}u],1u);atomicStore(&fsState[${words.stack}u+f],s);
}
`;
}

/** The slot state and its kernels. Built at a capacity (uniformFineSlotCapacity), allocated once. */
export class UniformFineSlots{
 readonly state:GPUBuffer;
 readonly words:UniformFineSlotWords;
 private pipelines?:{mark:GPUComputePipeline;claim:GPUComputePipeline;release:GPUComputePipeline;layout:GPUBindGroupLayout};
 private readonly groups=new Map<GPUBuffer,GPUBindGroup>();
 constructor(private readonly device:GPUDevice,readonly blocks:number,readonly capacity:number){
  this.words=uniformFineSlotWords(blocks,capacity);
  this.state=device.createBuffer({label:"Uniform detail h slots",size:4*this.words.words,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
 }
 get allocatedBytes():number{return this.state.size;}
 /** The state of a host assignment (a slot a block, 0 none) at a capacity
  * (slots 1..capacity, at most the state's): the unused slots stacked so
  * that the lowest pops first, nothing claimed. */
 write(slots:Uint32Array,capacity=this.capacity):void{
  const w=this.words,words=new Uint32Array(w.words),used=new Set<number>();
  if(capacity>this.capacity)throw new Error(`Uniform h slots: capacity ${capacity} past the state's ${this.capacity}`);
  words.set(slots,w.slot);for(const s of slots)if(s)used.add(s);
  if(used.size!==slots.filter(s=>s!==0).length||[...used].some(s=>s>capacity))throw new Error("Uniform h slots: an assignment gives a slot twice or past the capacity");
  const free:number[]=[];for(let s=capacity;s>=1;s--)if(!used.has(s))free.push(s);
  words[w.free]=free.length;words.set(free,w.stack);
  this.device.queue.writeBuffer(this.state,0,words);
 }
 async initialize(seam:(table:string)=>string,dimensions:GPUBuffer,support:GPUBuffer,violationWord:number):Promise<void>{
  const device=this.device,C=GPUShaderStage.COMPUTE;
  const module=device.createShaderModule({label:"Uniform detail h slots",code:slotsWGSL(seam,this.words,this.blocks,this.capacity,violationWord)});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=device.createBindGroupLayout({label:"Uniform detail h slots",entries:[
   {binding:0,visibility:C,buffer:{type:"read-only-storage"}},{binding:1,visibility:C,buffer:{type:"read-only-storage"}},{binding:2,visibility:C,buffer:{type:"uniform"}},
   {binding:3,visibility:C,buffer:{type:"storage"}},{binding:4,visibility:C,buffer:{type:"storage"}}]});
  const pipeline=(entryPoint:string)=>device.createComputePipelineAsync({label:`Uniform detail h slots ${entryPoint}`,layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint}});
  const [mark,claim,release]=await Promise.all([pipeline("mark"),pipeline("claim"),pipeline("release")]);
  this.pipelines={mark,claim,release,layout};this.dimensions=dimensions;this.support=support;
 }
 private dimensions?:GPUBuffer;
 private support?:GPUBuffer;
 private group(held:GPUBuffer,next:GPUBuffer):GPUBindGroup{
  // One group a next topology (the held one, a builder generation's, a host layout's transient).
  let g=this.groups.get(next);
  if(!g){
   if(this.groups.size>=8)this.groups.clear();
   this.groups.set(next,g=this.device.createBindGroup({label:"Uniform detail h slots",layout:this.pipelines!.layout,entries:[{binding:0,resource:{buffer:held}},{binding:1,resource:{buffer:next}},
    {binding:2,resource:{buffer:this.dimensions!}},{binding:3,resource:{buffer:this.state}},{binding:4,resource:{buffer:this.support!}}]}));
  }
  return g;
 }
 /** Slots for every block the next generation's ring needs (held: the
  * topology whose page table places the vertices), the claimed list beside
  * them. tiles: the tile keys. */
 encodeClaim(encoder:GPUCommandEncoder,held:GPUBuffer,next:GPUBuffer,tiles:number):void{
  const p=this.pipelines;if(!p)throw new Error("Uniform h slots are not initialized");
  const w=this.words;
  encoder.clearBuffer(this.state,4*w.need,4*this.blocks);encoder.clearBuffer(this.state,4*w.claimed,4);
  const pass=encoder.beginComputePass({label:"Uniform detail h slot claim"}),g=this.group(held,next);
  pass.setBindGroup(0,g);
  pass.setPipeline(p.mark);pass.dispatchWorkgroups(Math.ceil(tiles/64));
  pass.setPipeline(p.claim);pass.dispatchWorkgroups(Math.ceil(this.blocks/64));
  pass.end();
 }
 /** After the remap: the blocks the last claim's generation does not need give their slots back. */
 encodeRelease(encoder:GPUCommandEncoder,held:GPUBuffer):void{
  const p=this.pipelines;if(!p)throw new Error("Uniform h slots are not initialized");
  const pass=encoder.beginComputePass({label:"Uniform detail h slot release"});
  pass.setBindGroup(0,this.group(held,held));pass.setPipeline(p.release);pass.dispatchWorkgroups(Math.ceil(this.blocks/64));pass.end();
 }
 /** Workgroups of an init launch. */
 static readonly initGroups=INIT_GROUPS;
 destroy():void{this.state.destroy();}
}
