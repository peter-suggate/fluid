/** The frame brick store: the brick-resident step's level-0 slot space
 * (docs/plans/brick-resident-step-2026-11-08.md §3.1). It is a
 * UniformFineBrickStore on the frame's ownership, with the arena's table.
 *
 * A brick (8³ h cells, id 64·block + local) is marked where it holds an h
 * tile of the ownership's h list (class fine, a non-core bit: no dilation;
 * a neighbour brick that is not live stages through ubaStage's seam rule).
 * A brick keeps its slot, record and channel values while it stays live:
 * only enter and leave churn runs (ubaUpdate, ubaLists, ubaLevel1,
 * ubaRecord, ubaRelease). Record word 28 masks the brick's h-owned tiles,
 * so a 4h tile inside a live brick stages through the seam rule too.
 *
 * Capacity is authored, never grown. An h tile lies in one brick, so the
 * ownership's h tile ceiling (capacity.fineTiles, the scene's authored
 * peak) bounds the live bricks and is the default. A pop from an empty free stack sets the fatal
 * word (fatalOffset).
 *
 * Fused passes bind the table at (group, binding) and include
 * tableWGSL(name). The table lists hold the live bricks (ubaCount(UBA_LIVE_COUNT),
 * UBA_LIVE) and the held pages (ubaLevelCount(2u), ubaLevelSlot(2u, e)).
 * Channels move into the store's pool only where a fused pass reads them.
 * Elsewhere a pass stages from the paged textures by lattice coordinate,
 * which the detail rewrite resolves. */
import { UNIFORM_ARENA_CLASS, UNIFORM_ARENA_RELEASE_ENTRY, UNIFORM_ARENA_TABLE_ENTRIES } from "./uniform-brick-arena";
import { UNIFORM_FINE_BRICK_FLOOR, UniformFineBrickStore, type UniformFineBrickChannel } from "./uniform-fine-brick-store";
import { uniformDetailModule, uniformDetailPipeline } from "./uniform-detail-fields";
import { UNIFORM_LIVE_KEY_GROUPS, UNIFORM_LIVE_LANE_GROUPS } from "./uniform-live-lists";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformStageListsWGSL } from "./uniform-stage-lists";

/** Slots by default: the h tile ceiling (a live brick holds at least one h tile), at least FLOOR, at most the lattice's bricks. */
export function uniformFrameBrickCapacity(blocks:number,fineTiles:number,bricks?:number):number{
 if(bricks!==undefined){if(!Number.isInteger(bricks)||bricks<1)throw new Error(`Frame bricks: an authored capacity of ${bricks}`);return Math.min(64*blocks,bricks);}
 return Math.min(64*blocks,Math.max(UNIFORM_FINE_BRICK_FLOOR,fineTiles));
}
const PASS:GPUComputePassDescriptor={label:"Uniform frame bricks"};
const groupsOf=(n:number)=>Math.max(1,Math.ceil(n/64));

export class UniformFrameBricks{
 readonly store:UniformFineBrickStore;
 readonly table:GPUBuffer;
 private readonly layout:GPUBindGroupLayout;
 private readonly group:GPUBindGroup;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 private initialized=false;
 constructor(private readonly device:GPUDevice,private readonly ownership:UniformMixedOwnership,options:{capacity?:number;channels?:readonly UniformFineBrickChannel[]}={}){
  const blocks=ownership.capacity.address.blocks,capacity=uniformFrameBrickCapacity(blocks,ownership.capacity.fineTiles,options.capacity);
  this.store=new UniformFineBrickStore({blocks,capacity,channels:options.channels??[],lists:[{name:"fine",any:UNIFORM_ARENA_CLASS.fine}]});
  this.table=device.createBuffer({label:"Uniform frame brick table",size:4*this.store.arena.tableWords,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  for(const w of this.store.arena.initialWrites(0))device.queue.writeBuffer(this.table,w.offset,w.data as Uint32Array<ArrayBuffer>);
  this.layout=device.createBindGroupLayout({label:"Uniform frame bricks",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
  this.group=device.createBindGroup({layout:this.layout,entries:[{binding:0,resource:{buffer:this.table}}]});
 }
 get allocatedBytes():number{return this.table.size;}
 get capacity():number{return this.store.capacity;}
 /** The table's WGSL for a fused pass that binds it as `name` (array<atomic<u32>>): the arena's accessors and constants, with the h-owned tile mask. Needs the ownership topology and uniformStageListsWGSL. */
 tableWGSL(name:string):string{
  return /* wgsl */`
// A frame brick's h-owned tiles (record word 28): bit x|y<<1|z<<2 of its 2³ tiles.
fn ufbOwnedTiles(id:u32)->u32{
 let b=id>>6u;if(!umBlockStored(b)){return 0u;}
 let l=id&63u;let first=2u*vec3u(l&3u,(l>>2u)&3u,l>>4u);var mask=0u;
 for(var k=0u;k<8u;k++){let t=512u*b+umTileLocal(first+umCorner(k,2u));if(umTileLive(t)&&umTileWidth(t)==1u){mask|=1u<<k;}}
 return mask;
}
`+this.store.arena.tableWGSL(name,"0u","ufbOwnedTiles(id)");
 }
 async initialize():Promise<void>{
  const o=this.ownership;
  const code=uniformMixedTopologyWGSL(o.capacity,0)+uniformStageListsWGSL+/* wgsl */`
@group(1) @binding(0) var<storage,read_write> ufbTable:array<atomic<u32>>;
`+this.tableWGSL("ufbTable")+/* wgsl */`
// A thread an h tile of the ownership's h list: its brick is marked fine.
@compute @workgroup_size(64) fn ufbMark(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var j=gid.x;j<umCounts.x;j+=64u*groups.x){ubaMark(ubaBrickAt(vec3i(umTileCoord(umTopology[UM_TILES+j])*4u)),${UNIFORM_ARENA_CLASS.fine}u);}
}
`;
  const module=uniformDetailModule(this.device,{label:"Uniform frame bricks",code});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(errors.length)throw new Error(`Frame bricks: ${errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n")}`);
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[o.bindLayout,this.layout]});
  await Promise.all(["ufbMark",...UNIFORM_ARENA_TABLE_ENTRIES,UNIFORM_ARENA_RELEASE_ENTRY].map(async entryPoint=>{
   this.pipelines.set(entryPoint,await uniformDetailPipeline(this.device,o,{layout,compute:{module,entryPoint,constants:{umDispatchX:o.dispatchX}}}));
  }));
  this.initialized=true;
 }
 /** The step's table: clear the marks and counts, mark the h list's bricks, then the arena's churn build. After the frame's topology is final. */
 encodeUpdate(encoder:GPUCommandEncoder):void{
  if(!this.initialized)throw new Error("Frame bricks are not initialized");
  const clear=this.store.arena.stepClear(0);encoder.clearBuffer(this.table,clear.offset,clear.size);
  const pass=encoder.beginComputePass(PASS);pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group);
  const cap=this.store.capacity,groups:Record<string,number>={ufbMark:groupsOf(this.ownership.capacity.fineTiles),ubaUpdate:UNIFORM_LIVE_KEY_GROUPS,ubaLists:UNIFORM_LIVE_LANE_GROUPS,ubaLevel1:1,ubaRecord:Math.min(groupsOf(32*(cap+this.ownership.capacity.address.blocks)),4*UNIFORM_LIVE_KEY_GROUPS)};
  for(const entry of ["ufbMark",...UNIFORM_ARENA_TABLE_ENTRIES]){pass.setPipeline(this.pipelines.get(entry)!);pass.dispatchWorkgroups(groups[entry]!);}
  pass.end();
 }
 /** After the step's last reader of leaving bricks: their slots back onto the free stack. */
 encodeRelease(encoder:GPUCommandEncoder):void{
  const pass=encoder.beginComputePass(PASS);pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group);
  pass.setPipeline(this.pipelines.get(UNIFORM_ARENA_RELEASE_ENTRY)!);pass.dispatchWorkgroups(UNIFORM_LIVE_LANE_GROUPS);pass.end();
 }
 /** The byte offset of the fatal word (0, or UBA_SEAM | the brick refused a slot). */
 get fatalOffset():number{return this.store.arena.fatalOffset(0);}
 destroy():void{this.table.destroy();}
}
