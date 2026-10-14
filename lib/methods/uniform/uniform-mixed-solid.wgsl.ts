/** Embedded solids for fine mixed owners: the paged voxel mask
 * (SolidOccupancyMask's image, the host's active scratch, with the bed-slope lane in it), the terrain
 * heightfield and the GPU-resident rigid bodies (WebGPURigidBodySystem's
 * state buffer, params.boundary.z of them), read through the native
 * parameter block. Every helper is a transcription of its native namesake in
 * webgpu-uniform-reference.wgsl.ts (cellOpenFraction, faceOpenFraction,
 * pressureFaceData, insideAnyTraceSolid). Bodies move: the host rebuilds the
 * all-4h record every frame they exist, displaces liquid out of the cells
 * they enter, and promotes the tiles they sweep (uniform-mixed-bodies.ts).
 * The host's mixed frame always compiles it (live voxel edits and bodies
 * reach a scene built without solids); stages built without it get inert stubs.
 * Coarse (4h) simulation owners never reach these helpers: promotion keeps
 * liquid one full tile away from any cut cell a 4h owner holds (a dry cut
 * tile may run 4h, without solid terms). Band pressure's all-4h levels read
 * the static coarse record instead (see UniformMixedSolid.coarse). */
import {uniformMixedPressureStorage} from "./uniform-mixed-pressure-boundary.wgsl";
import {sceneShapeWgsl} from "../../core/scene-shape";
import {SOLID_MASK_DIRECTORY_WORDS,SOLID_MASK_WALL,solidOccupancyMaskWGSL,type SolidOccupancyMaskPages} from "../../core/solid-occupancy-mask";
import type {UniformMixedLayout} from "./uniform-mixed-layout";
import {uniformMixedChangedTilesWGSL} from "./uniform-mixed-layout-builder";
import {uniformDetailPick} from "./uniform-detail-fields";
import {UNIFORM_PAGE_TABLE_WORDS,UNIFORM_PAGE_WINDOW,uniformBlockFaceWGSL,uniformFaceTileSlots,uniformLatticeAddressWGSL,type UniformLatticeAddress} from "./uniform-lattice-address";
import {uniformPagePoolWGSL,type UniformPagePoolLayout} from "./uniform-page-pool";


/** The record's rows, by tile key: a row a key (umSolidTiles), then a wall
 * slot a tile of each of a block's six faces (umFaceTile), the all-4h
 * layout's own index (uniformMixedPressureStorage) up to umSolidCount; after
 * them a flag pair a key, then three words a key. */
function solidRecordRowsWGSL(keys:number):string{return `
fn umSolidTiles()->u32{return ${keys}u;}
fn umSolidCount()->u32{return ${keys+uniformFaceTileSlots(keys/512)}u;}
`;}
/** The page pool a frame claims blocks from: its state buffer (bound
 * read-only, for the list of blocks its last commit claimed) and layout. */
export interface UniformMixedSolidPool {readonly state:GPUBuffer;readonly layout:UniformPagePoolLayout}
/** Words of a topology up to the end of its page tail: five a tile key, the
 * page table, a word a block. */
const pagedTopologyWords=(layout:UniformMixedLayout):number=>5*layout.tiles.length+UNIFORM_PAGE_TABLE_WORDS+layout.tiles.length/512;
/** Tile keys, for a record kernel: the paged
 * lattice address (umTileLive, umTileCoord) over the page table and block
 * records in the tail of the topology the kernel binds as `name`, after its
 * five words a tile key (UM_PAGE_TABLE, UM_BLOCK_RECORD). */
function solidRecordAddressWGSL(layout:UniformMixedLayout,name:string):string{
 const keys=layout.tiles.length,table=5*keys;
 const paged={blocks:`${keys/512}u`};
 return /* wgsl */`
fn umPageWord(page:vec3i)->u32{let e=vec3u(page)&vec3u(${UNIFORM_PAGE_WINDOW-1}u);return ${name}[${table}u+(e.x|(e.y<<6u)|(e.z<<12u))];}
fn umBlockWord(b:u32)->u32{return ${name}[${table+UNIFORM_PAGE_TABLE_WORDS}u+b];}
${uniformLatticeAddressWGSL({constant:true,paged})}`;
}

/** Units of a bed-slope byte: 127 is half a cell of rise in a cell of run. */
export const UNIFORM_SOLID_BED_SLOPE_UNITS=254;
/** The parts of the solid mask image (SolidOccupancyMask) this library
 * places, in u32 words, for a lattice of `cells`:
 *
 *   bedSlope  after the directory, two signed bytes per cell of the lattice
 *             and its one-cell halo, cell (x,y,z) at index
 *             i=(x+1)+sx*((y+1)+sy*(z+1)), a word to two cells: bits
 *             16*(i&1)..+15 of word bedSlopeBaseWord+(i>>1), its low byte the
 *             bed height's gradient along x and its high byte along z, in
 *             cells per cell times UNIFORM_SOLID_BED_SLOPE_UNITS, pointing
 *             uphill. Non-zero only for an open cell that sits on a solid
 *             voxel's top face (umBedSlope).
 *   cutMap    after the lane, a word per tile key
 *
 * The lane's words are the mask's `laneWords`; umBedSlope and umSolidCutMap
 * derive the same offsets. */
export function uniformSolidScratchLayout(cells:readonly [number,number,number]):{bedSlopeBaseWord:number;bedSlopeWords:number;cutMapOffsetWords:number}{
 const bedSlopeWords=Math.ceil((cells[0]+2)*(cells[1]+2)*(cells[2]+2)/2);
 return {bedSlopeBaseWord:SOLID_MASK_DIRECTORY_WORDS,bedSlopeWords,cutMapOffsetWords:SOLID_MASK_DIRECTORY_WORDS+bedSlopeWords};
}
/** What the solver sizes its solid mask by (SolidOccupancyMask): the
 * lattice's page table and tile keys, the bed-slope lane, and a slot for
 * every page of the box and of the first layer of pages above an open top.
 * No edit inside the tank can then run out of slots; one that makes more
 * pages partly solid than that is refused by the mask. */
export function uniformSolidMaskPages(address:UniformLatticeAddress):SolidOccupancyMaskPages{
 const [x,y,z]=address.pages;
 return {table:address.seed.table,tileKeys:address.tileKeys,slots:x*y*z+(address.openTop?x*z:0),laneWords:uniformSolidScratchLayout(address.cells).bedSlopeWords};
}
/** A bed slope as its lane entry: 16 bits, x in the low byte, each rounded
 * half away from zero as the bake rounds (uniform-bed-slope-bake). */
export function uniformSolidBedSlopeEntry(slopeX:number,slopeZ:number):number{
 const byte=(slope:number)=>Math.sign(slope)*Math.min(127,Math.floor(Math.abs(slope)*UNIFORM_SOLID_BED_SLOPE_UNITS+0.5))&0xff;
 return byte(slopeX)|byte(slopeZ)<<8;
}
/** The slope a lane holds for a haloed cell index: umBedSlope's read. */
export function uniformSolidBedSlopeAt(lane:Uint32Array,index:number):[x:number,z:number]{
 const entry=lane[index>>>1]!>>>(16*(index&1));
 return [(entry<<24>>24)/UNIFORM_SOLID_BED_SLOPE_UNITS,(entry<<16>>24)/UNIFORM_SOLID_BED_SLOPE_UNITS];
}

export interface UniformMixedSolidResources {
 params:GPUBuffer;
 /** SolidOccupancyMask's image: the page directory, the tile cut map (a
  * word a tile key, all cut as the host uploads it; the record build
  * writes it) and the page slots. */
 scratch:GPUBuffer;terrain:GPUTexture;
 /** WebGPURigidBodySystem.stateBuffer: twelve 128-byte body records,
  * copied into the library's uniform mirror by encodeBodies (a storage
  * binding would exceed the band's per-stage storage budget). */
 bodies:GPUBuffer;
 /** Tiles the host promotes as solid-coupled (uniformMixedSolidTiles().coupled):
  * the h pressure band's reserve on top of its surface capacity. */
 coupledTiles:number;
 /** Whether the host has written `params` at least once. Required with the
  * all-4h record, which is built from that block: from a zero block every
  * tile reads uncut for the life of the frame. */
 paramsWritten?:()=>boolean;
}

/** The static all-4h solid record of band pressure (one vec4 per 4h owner,
 * owner index = the tile's key): (open, V+x, V+y, V+z), the two-level native
 * mgDownsampleTopology of the h CM11a record (open is the mean of the 64 h
 * cells, a face V the mean of its 16 h faces). Then one vec4 per all-4h wall
 * slot (umBoundaryIndex order: a tile of each block face) whose x is that
 * face's mean h V: together the
 * level-0 topology of band pressure. Then one vec4 per tile whose x is 1 when
 * the tile is cut: an h cell of it or of its one-cell ring has open < 1 (the
 * host's coupled tiles), and whose y is 1 when the simulation holds it at h
 * (encodeSimulation, per relayout).
 * Promotion is liquid-conditional: a dry cut tile may be 4h, and its h texels
 * are then stale, so the all-4h levels treat it as uncut (umSolidCut).
 * Every entry is at a tile key; the tile cut map (a word a tile, in the
 * mask image) is at the tile's key too and holds the cut flag. All of a tile
 * (its cut flag and cut map word, its owner row, the wall slots of a tile on
 * a face of its block, y, its open cells and patch codes) is written only for a tile
 * a page stores: by the build of its block when the pool claims it
 * (encodeNewBlocks) and by every rebuild, which visits the live tile keys
 * (encodeCoarse). Nothing is written when a block retires: its tiles'
 * entries keep what they held and have no reader. */
export interface UniformMixedSolidCoarse {
 readonly bindLayout:GPUBindGroupLayout;readonly bindGroup:GPUBindGroup;readonly record:GPUBuffer;
 /** Owners plus halo slots: the level-0 topology record's vec4 count. A
  * tile's (cut, h) flags follow at count + t, then three vec4 per tile at
  * count + tiles + 3t: its open h cells (16 bits a component, cell x+4y+16z)
  * and its 48 +face patch codes (24 bits a component, four a patch). */
 readonly count:number;
}

const RECORD_PASS:GPUComputePassDescriptor={label:"Uniform mixed all-4h solid record"},WIDTHS_PASS:GPUComputePassDescriptor={label:"Uniform mixed solid simulation widths"};
const NEW_BLOCKS_PASS:GPUComputePassDescriptor={label:"Uniform mixed solid record of claimed blocks"};
/** Blocks the claimed-block build's fixed grid holds a workgroup per tile
 * for; a commit that claims more is strided over the same grid. */
const NEW_BLOCK_GRID=8;
/** ceil(jobs/per) in whole steps: a fractional quotient is a heap number until its caller is optimized. */
const ceilDiv=(jobs:number,per:number):number=>{const up=jobs+per-1;return (up-up%per)/per;};

export class UniformMixedSolid {
 readonly bindLayout:GPUBindGroupLayout;
 readonly bindGroup:GPUBindGroup;
 readonly coupledTiles:number;
 private readonly bodySource:GPUBuffer;
 private readonly bodies:GPUBuffer;
 /** The all-4h record for band pressure (constructed with its all-4h layout). */
 readonly coarse?:UniformMixedSolidCoarse;
 /** build and the cut flags over the tile keys (in full, bodies-only), and
  * both over the pool's new list (claimed, cutsClaimed).
  * group: their output group over the topology last encoded with. */
 private builder?:{pipeline:GPUComputePipeline;bodies?:GPUComputePipeline;claimed?:GPUComputePipeline;cuts:GPUComputePipeline;cutsBodies:GPUComputePipeline;cutsClaimed?:GPUComputePipeline;layout:GPUBindGroupLayout;group?:GPUBindGroup;topology?:GPUBuffer};
 /** The library's group as the record builder binds it: the mask image writable, for the tile cut map. */
 private readonly recordSolids?:{readonly layout:GPUBindGroupLayout;readonly group:GPUBindGroup};
 private readonly pool?:UniformMixedSolidPool;
 /** Workgroups of the claimed-block build's fixed grid. */
 private readonly claimedGroups:number;
 /** The body mirror the record was last built with: a bodies-only rebuild
  * revisits the tiles near it or the current mirror. */
 private builtBodies?:GPUBuffer;
 /** The next build must visit every tile (first build, voxel edit). */
 private full=true;
 private simulation?:{pipeline:GPUComputePipeline;layout:GPUBindGroupLayout;group?:GPUBindGroup;topology?:GPUBuffer;
  listed?:GPUComputePipeline;listLayout:GPUBindGroupLayout;list?:{readonly topology:GPUBuffer;readonly changes:GPUBuffer;readonly group:GPUBindGroup}};
 private readonly params:GPUBuffer;
 private built=false;
 private readonly paramsWritten?:()=>boolean;
 /** The device's workgroups per dimension: read once, the limits object is the device's to rebuild. */
 private readonly maxGroups:number;
 /** pool: the page pool whose commits claim this frame's blocks; the record
  * of each claimed block is then built by encodeNewBlocks. */
 constructor(private readonly device:GPUDevice,resources:UniformMixedSolidResources,private readonly coarseLayout?:UniformMixedLayout,pool?:UniformMixedSolidPool){
  if(resources.params.size<272||resources.terrain.format!=="r32float")throw new Error("Mixed solids require the native parameter block and terrain heightfield");
  if(resources.bodies.size<12*128)throw new Error("Mixed solids require the rigid-body state buffer");
  if(!Number.isSafeInteger(resources.coupledTiles)||resources.coupledTiles<0)throw new Error("Mixed solids require the host's coupled tile count");
  this.coupledTiles=resources.coupledTiles;this.maxGroups=device.limits.maxComputeWorkgroupsPerDimension;
  this.params=resources.params;
  if(coarseLayout&&!resources.paramsWritten)throw new Error("The coarse solid record requires the host's parameter-block write receipt");
  this.paramsWritten=resources.paramsWritten;
  this.bodySource=resources.bodies;
  this.bodies=device.createBuffer({label:"Uniform mixed solid body mirror",size:12*128,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  const bodies=this.bodies;
  const entries:GPUBindGroupLayoutEntry[]=[
   {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:2,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"2d"}},
   {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
  ];
  const resourcesOf=():GPUBindGroupEntry[]=>[
   {binding:0,resource:{buffer:resources.params,size:272}},{binding:1,resource:{buffer:resources.scratch}},
   {binding:2,resource:resources.terrain.createView()},{binding:4,resource:{buffer:bodies}},
  ];
  this.bindLayout=device.createBindGroupLayout({label:"Uniform mixed static solids",entries});
  this.bindGroup=device.createBindGroup({layout:this.bindLayout,entries:resourcesOf()});
  this.claimedGroups=512*Math.min(NEW_BLOCK_GRID,pool?.layout.requests??0);
  if(coarseLayout){
   // One record entry per tile key: the all-4h layout's owner index is the tile's key.
   const tiles=coarseLayout.tiles.length;
   if(coarseLayout.tiles.some(word=>(word&0xc0000000)!==0)||coarseLayout.cellCount!==tiles)throw new Error("The coarse solid record requires the all-4h layout");
   if(coarseLayout.tiles.length%512)throw new Error(`The coarse solid record's layout has ${coarseLayout.tiles.length} tile keys: not whole blocks of 512`);
   if(pool&&512*pool.layout.blocks!==coarseLayout.tiles.length)throw new Error(`The coarse solid record's layout has ${coarseLayout.tiles.length} tile keys; the page pool has ${pool.layout.blocks} blocks of 512`);
   if(resources.scratch.size<4*(uniformSolidScratchLayout(coarseLayout.address.cells).cutMapOffsetWords+coarseLayout.tiles.length))throw new Error(`The solid mask image of ${resources.scratch.size} bytes has no room for the tile cut map of ${coarseLayout.tiles.length} tile keys after its directory and bed-slope lane`);
   this.pool=pool;
   const writable=device.createBindGroupLayout({label:"Uniform mixed static solids of the record build",entries:entries.map(entry=>entry.binding===1?{...entry,buffer:{type:"storage" as const}}:entry)});
   this.recordSolids={layout:writable,group:device.createBindGroup({layout:writable,entries:resourcesOf()})};
   const count=uniformMixedPressureStorage(coarseLayout).count;
   const record=device.createBuffer({label:"Uniform mixed all-4h solid record",size:16*(count+4*tiles),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
   const bindLayout=device.createBindGroupLayout({label:"Uniform mixed static solids with the all-4h record",entries:[...entries,
    {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}}]});
   this.coarse={bindLayout,record,count,bindGroup:device.createBindGroup({layout:bindLayout,entries:[...resourcesOf(),{binding:3,resource:{buffer:record}}]})};
  }
 }
 /** The library's group for a stage that reads tile records where the frame
  * has them (umTileOpen, umTileMask): the record group, else the plain one,
  * whose tile helpers answer uncut. */
 get tileLayout():GPUBindGroupLayout{return this.coarse?.bindLayout??this.bindLayout;}
 get tileGroup():GPUBindGroup{return this.coarse?.bindGroup??this.bindGroup;}
 /** Whether the scene holds any in-domain solid: a voxel, terrain or a
  * rigid body. The host sets it at construction, after every live edit and
  * when bodies are placed; stages then dispatch select()'s variant. */
 present=true;
 /** Select the pipeline built during setup. Solids are a uniform flag. */
 select(pipeline:GPUComputePipeline):GPUComputePipeline{return uniformDetailPick(pipeline);}
 get allocatedBytes():number{return (this.coarse?.record.size??0)+this.bodies.size+(this.builtBodies?.size??0);}
 /** Mirror the rigid state into the library: at a frame's head, and after
  * the rigid integration for the census that follows it. */
 encodeBodies(encoder:GPUCommandEncoder):void{encoder.copyBufferToBuffer(this.bodySource,0,this.bodies,0,12*128);}
 async initialize():Promise<void>{
  const layout=this.coarseLayout,coarse=this.coarse;if(!layout||!coarse||this.builder)return;
  const dispatchX=this.device.limits.maxComputeWorkgroupsPerDimension;
  const pool=this.pool;
  const module=this.device.createShaderModule({label:"Uniform mixed all-4h solid record",code:/* wgsl */`
${uniformMixedSolidWGSL(0,undefined,false,true)}
@group(1) @binding(0) var<storage,read_write> record:array<vec4f>;
override umDispatchX:u32=65535u;
@group(1) @binding(1) var<uniform> umBuiltBodies:array<UMRigidBody,12>;
@group(1) @binding(2) var<storage,read> topology:array<u32>;
${solidRecordAddressWGSL(layout,"topology")}
${pool?uniformPagePoolWGSL(pool.layout,1,3):""}
// Bodies-only rebuild: a tile far from every body, as last built and now,
// keeps its record (its cut ring and face samples reach two cells out).
override umBodyTiles:bool=false;
fn umBodyTileNear(b:UMRigidBody,lo:vec3f,hi:vec3f,margin:f32)->bool{
 if(b.dimensions.w<=0.0){return false;}
 let p=b.positionShape.xyz;return distance(clamp(p,lo,hi),p)<=b.dimensions.w+margin;
}
fn umBodyTileDirty(tc:vec3u)->bool{
 let h=umSolidParams.cellGravity.xyz;let lo=umSolidWorldCell(vec3i(tc*4u))-0.5*h;let hi=lo+4.0*h;
 let margin=2.0*max(h.x,max(h.y,h.z));
 for(var i=0u;i<12u;i++){if(umBodyTileNear(umSolidBodies[i],lo,hi,margin)||umBodyTileNear(umBuiltBodies[i],lo,hi,margin)){return true;}}
 return false;
}
var<workgroup> sums:array<vec4f,64>;
var<workgroup> lows:array<vec4f,64>;
// What depends on the solid alone and a per-frame kernel would recompute:
// the tile's open h cells (umCellOpen > 1e-5), and each +face patch's V where
// it is a whole number of eighths (the eight-sample V always is): code
// 8V + 1 of patch 16 axis + u + 4 v, 0 for any other V.
var<workgroup> openCells:array<atomic<u32>,2>;
var<workgroup> patchCodes:array<atomic<u32>,8>;
${solidRecordRowsWGSL(layout.tiles.length)}
${uniformBlockFaceWGSL}
// umBoundaryIndex of the all-4h layout: the wall slot of tile key t on face 2 axis + side of its block.
fn umSolidHalo(t:u32,axis:u32,side:u32)->u32{return umSolidTiles()+umFaceTile(t>>9u,2u*axis+side,umLocalTile(t));}
// The cut flag of the tile at key t (x of its flags, and its word of the tile
// cut map): an h cell of the tile or of its one-cell ring has open < 1. A
// cell of a page the lattice walls off cuts nothing. Nothing for a key that
// holds no tile.
fn umCutTile(t:u32){
 if(!umTileLive(t)){return;}
 let tc=umTileCoord(t);if(umBodyTiles&&!umBodyTileDirty(tc)){return;}
 var cut=0u;
 for(var i=0u;i<216u&&cut==0u;i++){
  let q=vec3i(tc*4u)-vec3i(1)+vec3i(vec3u(i%6u,(i/6u)%6u,i/36u));
  if(umSolidValid(q)&&umCellOpen(q)<1.0){cut=1u;}
 }
 record[umSolidCount()+t].x=f32(cut);
 umSolidScratch[umSolidCutMap()+t]=cut;
}
// Every stored tile: a lane a tile key.
@compute @workgroup_size(64) fn cuts(@builtin(global_invocation_id) g:vec3u){umCutTile(g.x+umDispatchX*64u*g.y);}
// The record of the tile at key t, one workgroup: nothing for a key that
// holds no tile (a free block's). Every word but the cut flag is written, y
// as 4h until encodeSimulation reads the tile's word. A tile on a face of
// its block writes that face's wall slot too: the V a wall there holds,
// read only where a SOLID page lies across the face.
fn umBuildTile(t:u32,lane:u32){
 if(!umTileLive(t)){return;}
 let tc=umTileCoord(t);let row=t;
 if(umBodyTiles&&!umBodyTileDirty(tc)){return;}
 let l=vec3u(lane%4u,(lane/4u)%4u,lane/16u);let c=vec3i(tc*4u+l);
 if(lane==0u){for(var i=0u;i<2u;i++){atomicStore(&openCells[i],0u);}for(var i=0u;i<8u;i++){atomicStore(&patchCodes[i],0u);}}
 var own=vec4f(umCellOpen(c),0.0,0.0,0.0);var low=vec4f(0.0);
 for(var axis=0u;axis<3u;axis++){
  if(l[axis]==3u){own[axis+1u]=umPressureFaceV(c,axis);}
  if(l[axis]==0u){var q=c;q[axis]-=1;low[axis+1u]=umPressureFaceV(q,axis);}
 }
 workgroupBarrier();
 if(own.x>1e-5){atomicOr(&openCells[lane>>5u],1u<<(lane&31u));}
 for(var axis=0u;axis<3u;axis++){
  let eighths=8.0*own[axis+1u];if(l[axis]!=3u||eighths!=floor(eighths)||eighths<0.0||eighths>8.0){continue;}
  let slot=16u*axis+l[(axis+1u)%3u]+4u*l[(axis+2u)%3u];
  atomicOr(&patchCodes[slot/6u],(u32(eighths)+1u)<<(4u*(slot%6u)));
 }
 sums[lane]=own;lows[lane]=low;
 workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){sums[lane]+=sums[lane+stride];lows[lane]+=lows[lane+stride];}workgroupBarrier();}
 if(lane==0u){
  let s=sums[0];let lo=lows[0];
  record[row]=vec4f(s.x/64.0,s.y/16.0,s.z/16.0,s.w/16.0);
  record[umSolidCount()+row].y=0.0;
  let cells=vec2u(atomicLoad(&openCells[0]),atomicLoad(&openCells[1]));
  record[umSolidCount()+umSolidTiles()+3u*row]=vec4f(vec4u(cells.x&0xffffu,cells.x>>16u,cells.y&0xffffu,cells.y>>16u));
  for(var w=0u;w<2u;w++){record[umSolidCount()+umSolidTiles()+3u*row+1u+w]=vec4f(vec4u(atomicLoad(&patchCodes[4u*w]),atomicLoad(&patchCodes[4u*w+1u]),atomicLoad(&patchCodes[4u*w+2u]),atomicLoad(&patchCodes[4u*w+3u])));}
  for(var axis=0u;axis<3u;axis++){
   let at=tc[axis]&7u;
   if(at==0u){record[umSolidHalo(t,axis,0u)]=vec4f(lo[axis+1u]/16.0,0.0,0.0,0.0);}
   if(at==7u){record[umSolidHalo(t,axis,1u)]=vec4f(s[axis+1u]/16.0,0.0,0.0,0.0);}
  }
 }
}
// Every stored tile: a workgroup per tile key.
@compute @workgroup_size(64) fn build(@builtin(workgroup_id) g:vec3u,@builtin(local_invocation_index) lane:u32){
 umBuildTile(g.x+umDispatchX*g.y,lane);
}
${pool?/* wgsl */`// The tiles of the blocks the last commit claimed (the pool's new list): job
// j is tile j % 512 of new block j / 512, whose key is 512 x the block + that
// index. A fixed grid of workgroups strides the jobs, whatever their count.
@compute @workgroup_size(64) fn buildNew(@builtin(workgroup_id) g:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let jobs=512u*upNewCount();
 for(var job=g.x+groups.x*g.y;job<jobs;job+=groups.x*groups.y){umBuildTile(512u*upNewBlock(job>>9u)+(job&511u),lane);}
}
// Their cut flags: the same jobs, strided by a fixed row of lanes.
@compute @workgroup_size(64) fn cutsNew(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) groups:vec3u){
 let jobs=512u*upNewCount();
 for(var job=g.x;job<jobs;job+=64u*groups.x){umCutTile(512u*upNewBlock(job>>9u)+(job&511u));}
}`:""}`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const output=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   ...(pool?[{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}} as const]:[])]});
  const builderLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.recordSolids!.layout,output]});
  // The bodies-only rebuild is dispatched once a body has moved (needs).
  const build=(entryPoint:string,umBodyTiles:number)=>this.device.createComputePipelineAsync({layout:builderLayout,compute:{module,entryPoint,constants:{umDispatchX:dispatchX,umBodyTiles}}});
  const [pipeline,bodies,cuts,cutsBodies,claimed,cutsClaimed]=await Promise.all([build("build",0),build("build",1),build("cuts",0),build("cuts",1),pool?build("buildNew",0):undefined,pool?build("cutsNew",0):undefined]);
  const built=this.builtBodies=this.device.createBuffer({label:"Uniform mixed solid record body mirror",size:12*128,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.builder={bodies,pipeline,claimed,cuts,cutsBodies,cutsClaimed,layout:output};
  const flagModule=this.device.createShaderModule({label:"Uniform mixed solid simulation widths",code:/* wgsl */`
${solidRecordRowsWGSL(layout.tiles.length)}
override umDispatchX:u32=65535u;
@group(0) @binding(0) var<storage,read_write> record:array<vec4f>;
@group(0) @binding(1) var<storage,read> topology:array<u32>;
${solidRecordAddressWGSL(layout,"topology")}
// y of each stored tile's cut record: the simulation tile word is h.
@compute @workgroup_size(64) fn widths(@builtin(global_invocation_id) g:vec3u){
 let t=g.x+umDispatchX*64u*g.y;if(!umTileLive(t)){return;}
 record[umSolidCount()+t].y=select(0.0,1.0,(topology[t]&0x80000000u)!=0u);
}`});
  // The adopted generation's changed tiles only: every other tile kept its width.
  const listModule=this.device.createShaderModule({label:"Uniform mixed solid changed widths",code:/* wgsl */`
${solidRecordRowsWGSL(layout.tiles.length)}
override umDispatchX:u32=65535u;
@group(0) @binding(0) var<storage,read_write> record:array<vec4f>;
@group(0) @binding(1) var<storage,read> topology:array<u32>;
${solidRecordAddressWGSL(layout,"topology")}
${uniformMixedChangedTilesWGSL(layout.tiles.length,0,2)}
// A key that holds no tile keeps its row.
@compute @workgroup_size(64) fn widthsListed(@builtin(global_invocation_id) g:vec3u){
 let i=g.x+umDispatchX*64u*g.y;if(i>=umChangedCount()){return;}
 let t=umChangedTile(i);if(!umTileLive(t)){return;}
 record[umSolidCount()+t].y=select(0.0,1.0,(topology[t]&0x80000000u)!=0u);
}`});
  const flagErrors=[...(await flagModule.getCompilationInfo()).messages,...(await listModule.getCompilationInfo()).messages].filter(m=>m.type==="error");if(flagErrors.length)throw new Error(flagErrors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const flagLayout=this.device.createBindGroupLayout({label:"Uniform mixed solid simulation widths",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}}]});
  const listLayout=this.device.createBindGroupLayout({label:"Uniform mixed solid changed widths",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}}]});
  // The listed rewrite follows a GPU relayout's changed tiles (Dynamic).
  const simulation:NonNullable<UniformMixedSolid["simulation"]>={layout:flagLayout,listLayout,pipeline:undefined as unknown as GPUComputePipeline};
  await Promise.all([
   this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[flagLayout]}),compute:{module:flagModule,entryPoint:"widths",constants:{umDispatchX:dispatchX}}}).then(p=>{simulation.pipeline=p;}),
   this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[listLayout]}),compute:{module:listModule,entryPoint:"widthsListed",constants:{umDispatchX:dispatchX}}}).then(p=>{simulation.listed=p;})]);
  this.simulation=simulation;
 }
 /** Builds the all-4h record from the solids the native host has published
  * (its parameter block, voxel mask and body mirror): once, in full again
  * after a live edit, and after bodies move only the tiles near them as
  * last built or now. The cut flags, the cut map and the rest of the record
  * are built over the tile keys, for the stored tiles.
  * topology: the ownership's (its tail names the stored tiles). */
 encodeCoarse(encoder:GPUCommandEncoder,topology:GPUBufferBinding):void{
  const layout=this.coarseLayout;if(!layout||this.built)return;
  if(!this.builder)throw new Error("The coarse solid record is not initialized");
  if(!this.paramsWritten!())throw new Error("The coarse solid record was encoded before the host's first write of the native parameter block: built from a zero block it holds every tile uncut");
  const keys=layout.tiles.length,x=Math.min(keys,this.maxGroups),groups=ceilDiv(keys,64),gx=Math.min(groups,this.maxGroups),gy=ceilDiv(groups,gx);
  const pass=encoder.beginComputePass(RECORD_PASS);
  pass.setBindGroup(0,this.recordSolids!.group);pass.setBindGroup(1,this.builderGroup(topology));
  pass.setPipeline(this.full?this.builder.cuts:this.builder.cutsBodies);pass.dispatchWorkgroups(gx,gy);
  pass.setPipeline(this.full?this.builder.pipeline:this.builder.bodies!);pass.dispatchWorkgroups(x,ceilDiv(keys,x));pass.end();
  encoder.copyBufferToBuffer(this.bodies,0,this.builtBodies!,0,12*128);this.full=false;
  this.built=true;
 }
 /** Builds the record of every tile of each block the page pool's commit
  * just claimed, its cut flag and cut map word included, from the solids as
  * published: the same values the full build writes. Encode after the
  * commit and before anything reads a tile of those blocks. Two fixed
  * launches stride the pool's new-block list.
  * topology: the ownership's, its tail as the commit left it. */
 encodeNewBlocks(encoder:GPUCommandEncoder,topology:GPUBufferBinding):void{
  if(!this.coarseLayout)return;
  const claimed=this.builder?.claimed,cuts=this.builder?.cutsClaimed;if(!claimed||!cuts)throw new Error("The coarse solid record builds claimed blocks only with a page pool, once initialized");
  const pass=encoder.beginComputePass(NEW_BLOCKS_PASS);
  pass.setBindGroup(0,this.recordSolids!.group);pass.setBindGroup(1,this.builderGroup(topology));
  pass.setPipeline(cuts);pass.dispatchWorkgroups(this.claimedGroups/64,1);
  pass.setPipeline(claimed);pass.dispatchWorkgroups(this.claimedGroups,1);pass.end();
 }
 /** A record kernel reads tile keys through the page table and block records after the topology's tile words. */
 private paged(topology:GPUBufferBinding):void{
  if(topology.offset||(topology.size??topology.buffer.size)<4*pagedTopologyWords(this.coarseLayout!))throw new Error("The solid record reads the page table and block records after the topology's tile words: the binding does not hold them");
 }
 /** The builds' output group over a topology: made when the buffer changes. */
 private builderGroup(topology:GPUBufferBinding):GPUBindGroup{
  const b=this.builder!;
  if(b.topology!==topology.buffer||!b.group){
   this.paged(topology);b.topology=topology.buffer;
   b.group=this.device.createBindGroup({layout:b.layout,entries:[{binding:0,resource:{buffer:this.coarse!.record}},{binding:1,resource:{buffer:this.builtBodies!}},{binding:2,resource:topology},
    ...(this.pool?[{binding:3,resource:{buffer:this.pool.state}}]:[])]});
  }
  return b.group;
 }
 /** The record does not describe the published solids yet (never built, or
  * invalidated): a reader ahead of the next frame's head builds it first. */
 get recordStale():boolean{return !!this.coarseLayout&&!this.built;}
 /** A live voxel edit changed the mask (every tile), or only bodies moved
  * (the tiles near them): rebuild the all-4h record. */
 invalidate(bodiesOnly=false):void{this.built=false;if(!bodiesOnly)this.full=true;}
 /** Records which tiles the simulation holds at h (the record's y), from
  * its tile words. Encode after encodeCoarse and after every relayout,
  * before the all-4h levels read umSolidCut. changes: the changed tiles
  * (UniformMixedGenerationBuffers.changes) of the generation just adopted,
  * when y still records the ownership it replaced (no record build since the
  * last encode): only those tiles' y can differ, so only they are written.
  * The kernels read tile keys through the page table and block records in
  * the topology's tail. */
 encodeSimulation(encoder:GPUCommandEncoder,topology:GPUBufferBinding,changes?:GPUBuffer):void{
  const layout=this.coarseLayout,coarse=this.coarse,s=this.simulation;if(!layout||!coarse)return;
  if(!s)throw new Error("The coarse solid record is not initialized");
  const groups=ceilDiv(layout.tiles.length,64),x=Math.min(groups,this.maxGroups);
  const pass=encoder.beginComputePass(WIDTHS_PASS);
  if(changes){
   if(s.list?.topology!==topology.buffer||s.list.changes!==changes){this.paged(topology);s.list={topology:topology.buffer,changes,group:this.device.createBindGroup({layout:s.listLayout,entries:[{binding:0,resource:{buffer:coarse.record}},{binding:1,resource:topology},{binding:2,resource:{buffer:changes}}]})};}
   pass.setPipeline(s.listed!);pass.setBindGroup(0,s.list.group);
  }else{
   if(s.topology!==topology.buffer){this.paged(topology);s.topology=topology.buffer;s.group=this.device.createBindGroup({layout:s.layout,entries:[{binding:0,resource:{buffer:coarse.record}},{binding:1,resource:topology}]});}
   pass.setPipeline(s.pipeline);pass.setBindGroup(0,s.group!);
  }
  pass.dispatchWorkgroups(x,ceilDiv(groups,x));pass.end();
 }
 destroy():void{this.coarse?.record.destroy();this.bodies.destroy();this.builtBodies?.destroy();}
}

/** Build a stage pipeline during setup; all solid reads use the scene uniform. */
export function uniformMixedSolidPipeline(_solid:UniformMixedSolid|undefined,create:(constants:Record<string,number>)=>Promise<GPUComputePipeline>,_options?:{solidsOnly?:boolean;entry?:string}):Promise<GPUComputePipeline>{
 return create({});
}

/** Requires the topology and the library (with the record, or every 4h owner
 * answers uncut). A vertex is buried when no incident owner has capacity at
 * its own width: an h cell its open fraction, a 4h owner its tile's mean. phi
 * there is not state. The corners of a cut 4h owner are state even where
 * every h cell around them is closed: its plane is read at them. A box tile
 * no page stores counts as a 4h owner. */
export const uniformMixedVertexBuriedWGSL=/* wgsl */`
fn umVertexBuried(p:vec3i)->bool{
 let base=p-vec3i(1);
 for(var k=0u;k<8u;k++){
  let cell=base+vec3i(umCorner(k,2u));if(!umSolidValid(cell)){continue;}
  let tile=cell>>vec3u(2u);let t=umTileKey(tile);
  if(!umKeyStored(t)){if(umTileOpenAt(tile)>1e-5){return false;}}
  else if(umTileWidth(t)==4u){if(umTileOpen(t)>1e-5){return false;}}
  else if(umCellIsOpen(cell)){return false;}
 }
 return true;
}`;

/** Requires the lattice address (umTileKey, umKeyStored, umTilePresent).
 * `group` undefined emits inert stubs with the same ABI: a cell is valid off a SOLID page.
 * `coarse` (the record's owner/halo count) binds the all-4h record: the
 * group is then UniformMixedSolid.coarse.bindLayout, and umSolidCoarse(i),
 * umSolidCut(tile) (cut and simulated at h), umSolidStaticCut(tile),
 * umTileOpen(tile) and umTileSupported(tile) read it; without it the tile
 * helpers answer uncut (a frame built without the record promotes every cut tile).
 * `gated` (every stage but the record builder, which writes the gate) answers
 * the helpers from the tile cut map where no solid is near: one word instead
 * of the voxel, terrain and body samples. `writable` binds the mask image
 * read_write, for the record builder's own group. */
export function uniformMixedSolidWGSL(group?:number,coarse?:number,gated=true,writable=false):string{
 if(group===undefined)return /* wgsl */ `
fn umSolidEnabled()->bool{return false;}
fn umSolidValid(p:vec3i)->bool{return umCellKind(p)!=UM_SOLID;}
fn umSolidVoxel(p:vec3i)->bool{return false;}
fn umBedSlope(p:vec3i)->vec2f{return vec2f(0);}
fn umCellOpen(p:vec3i)->f32{return select(0.0,1.0,umSolidValid(p));}
fn umCellIsOpen(p:vec3i)->bool{return umSolidValid(p);}
fn umCellInsideSolid(p:vec3i)->bool{return false;}
fn umFaceOpen(id:vec3i,axis:u32)->f32{return 1.0;}
fn umPressureFaceV(id:vec3i,axis:u32)->f32{return 1.0;}
fn umSolidAtWorldCell(p:vec3f)->bool{return false;}
fn umSolidFaceVelocity(id:vec3i,axis:u32)->f32{return 0.0;}
fn umBodyCount()->u32{return 0u;}
fn umTileOpen(t:u32)->f32{return 1.0;}
fn umTileOpenAt(p:vec3i)->f32{return 1.0;}
fn umTileSupported(t:u32)->bool{return false;}
`;
 return /* wgsl */ `
struct UMSolidParams {
 dimsDt:vec4f,cellGravity:vec4f,container:vec4f,physical:vec4f,boundary:vec4f,
 inflowA:vec4f,inflowB:vec4f,inflowC:vec4f,tuning:vec4f,drop:vec4f,dropExtent:vec4f,
 twoLevel:vec4f,agreement:vec4f,lean:vec4f,splash:vec4f,splashB:vec4f,cleanup:vec4f,
}
@group(${group}) @binding(0) var<uniform> umSolidParams:UMSolidParams;
fn umSolidDimensions()->vec3u{return vec3u(umSolidParams.dimsDt.xyz);}
@group(${group}) @binding(1) var<storage,${writable?"read_write":"read"}> umSolidScratch:array<u32>;
@group(${group}) @binding(2) var umSolidTerrain:texture_2d<f32>;
struct UMRigidBody {
 positionShape:vec4f,dimensions:vec4f,orientation:vec4f,linearVelocity:vec4f,
 angularVelocity:vec4f,inverseMassInertia:vec4f,angularMomentumRestitution:vec4f,material:vec4f,
}
@group(${group}) @binding(4) var<uniform> umSolidBodies:array<UMRigidBody,12>;
// lean.x certifies that voxels, terrain and the live body roster are empty.
fn umSolidEnabled()->bool{return ${gated?"umSolidParams.lean.x<0.5":"true"};}
${coarse===undefined?`fn umTileOpen(t:u32)->f32{return 1.0;}
fn umTileOpenAt(p:vec3i)->f32{return 1.0;}
fn umTileSupported(t:u32)->bool{return false;}`:`@group(${group}) @binding(3) var<storage,read> umSolidRecord:array<vec4f>;
${solidRecordRowsWGSL(coarse*4/7)}
// A tile's entries are at its key. Only a stored tile's entries are current
// (built when its block is claimed and by every rebuild), so every helper
// below that takes a key takes a live one, and the three that are asked
// about a tile whatever its page (umCellIsOpen, umTileOpenAt, umTileSupported
// of the tile below) answer for an unstored one from the solids themselves.
// The mean of umCellOpen over the 64 h cells of the tile at p.
fn umTileOpenCells(p:vec3i)->f32{
 var open=0.0;for(var i=0u;i<64u;i++){open+=umCellOpen(4*p+vec3i(vec3u(i%4u,(i/4u)%4u,i/16u)));}
 return open/64.0;
}
// A tile's open fraction (the mean of its 64 h cells' umCellOpen): what a
// 4h owner of the tile holds.
fn umTileOpen(t:u32)->f32{
 if(!umSolidEnabled()){return 1.0;}
 return umSolidRecord[t].x;
}
// The same of the tile at p, whatever its page.
fn umTileOpenAt(p:vec3i)->f32{
 let t=umTileKey(p);if(umKeyStored(t)){return umTileOpen(t);}
 if(!umSolidEnabled()){return 1.0;}
 return umTileOpenCells(p);
}
// An all-4h owner or halo slot of the record (the all-4h layout's own index):
// a stored tile's row, or a halo slot beside one.
fn umSolidCoarse(i:u32)->vec4f{return umSolidRecord[i];}
// A cut tile standing on solid: its -y face (the tile below's V+, or the
// container floor) is not whole. V a 4h owner holds there rests on it.
fn umTileSupported(t:u32)->bool{
 if(!umSolidEnabled()){return false;}
 let tc=umTileCoord(t);if(umSolidRecord[t].x>=0.99999){return false;}
 if(tc.y==0u){return true;}
 let below=umTileKey(vec3i(tc)-vec3i(0,1,0));if(umKeyStored(below)){return umSolidRecord[below].z<0.99999;}
 var v=0.0;for(var i=0u;i<16u;i++){v+=umPressureFaceV(vec3i(4u*tc+vec3u(i%4u,0u,i/4u))-vec3i(0,1,0),1u);}
 return v/16.0<0.99999;
}
fn umSolidStaticCut(t:u32)->bool{return umSolidRecord[umSolidCount()+t].x>0.5;}
// A cut tile whose h texels are live: the simulation holds it at h.
fn umSolidCut(t:u32)->bool{let r=umSolidRecord[umSolidCount()+t];return r.x>0.5&&r.y>0.5;}
// The simulation holds this tile at h (cut or not).
fn umSolidFineTile(t:u32)->bool{return umSolidRecord[umSolidCount()+t].y>0.5;}
// The record's three solid-only words of tile t.
fn umSolidTileWord(t:u32,word:u32)->vec4u{return vec4u(umSolidRecord[umSolidCount()+umSolidTiles()+3u*t+word]);}
// The open h cells (umCellOpen > 1e-5) of tile t: bit x+4y+16z.
fn umSolidOpenCells(t:u32)->vec2u{let r=umSolidTileWord(t,0u);return vec2u(r.x|(r.y<<16u),r.z|(r.w<<16u));}
// umPressureFaceV of a +axis patch of tile t (id a cell on that plane): the
// record's where it is a whole number of eighths, else the full evaluation.
fn umPatchFaceV(t:u32,id:vec3i,axis:u32)->f32{
 let l=vec3u(id)&vec3u(3u);let slot=16u*axis+l[(axis+1u)%3u]+4u*l[(axis+2u)%3u];
 let code=(umSolidTileWord(t,1u+slot/24u)[(slot/6u)%4u]>>(4u*(slot%6u)))&15u;
 if(code==0u){return umPressureFaceV(id,axis);}
 return 0.125*f32(code-1u);
}`}
// The mask image (SolidOccupancyMask): umSolidPage(p), the directory word of
// the page of cell p, whatever page of the window that is and whether or not
// a block stores it, and umSolidVoxel(p), its bit. A page the lattice walls
// off (the page table's SOLID) reads solid throughout.
${solidOccupancyMaskWGSL("umSolidScratch","umSolidPage","umSolidVoxel")}
// The cell is not in a page the lattice walls off: one directory word.
fn umSolidValid(p:vec3i)->bool{return umSolidPage(p)!=${SOLID_MASK_WALL}u;}
// The bed-slope lane, after the directory and by cell of the lattice and its
// one-cell halo (uniformSolidScratchLayout): the gradient of the bed's height along x and
// z, in cells per cell and pointing uphill, for an open cell that sits on a
// solid voxel's top face. Zero for every other cell: a voxel tread is then
// the level floor it looks like.
fn umSolidHaloCells()->u32{let s=umSolidDimensions()+vec3u(2u);return s.x*s.y*s.z;}
fn umBedSlopeBase()->u32{return ${SOLID_MASK_DIRECTORY_WORDS}u;}
// The tile cut map's first word: after the lane.
fn umSolidCutMap()->u32{return umBedSlopeBase()+(umSolidHaloCells()+1u)/2u;}
fn umBedSlope(p:vec3i)->vec2f{
 if(!umSolidEnabled()){return vec2f(0);}
 let shape=vec3i(umSolidDimensions())+vec3i(2);let q=p+vec3i(1);
 if(any(q<vec3i(0))||any(q>=shape)){return vec2f(0);}
 let index=u32(q.x+shape.x*(q.y+shape.y*q.z));
 let entry=umSolidScratch[umBedSlopeBase()+(index>>1u)]>>(16u*(index&1u));
 return vec2f(vec2i(bitcast<i32>(entry<<24u)>>24u,bitcast<i32>(entry<<16u)>>24u))/${UNIFORM_SOLID_BED_SLOPE_UNITS}.0;
}
fn umSolidHasTerrain()->bool{return umSolidParams.container.w>0.5;}
fn umSolidTerrainHeight(x:i32,z:i32)->f32{let d=vec3i(umSolidDimensions());return textureLoad(umSolidTerrain,vec2i(clamp(x,0,d.x-1),clamp(z,0,d.z-1)),0).x;}
fn umCellInsideTerrain(p:vec3i)->bool{if(!umSolidHasTerrain()){return false;}return f32(p.y)+0.5<umSolidTerrainHeight(p.x,p.z);}
fn umCellTerrainFraction(p:vec3i)->f32{if(!umSolidHasTerrain()){return 0.0;}return clamp(umSolidTerrainHeight(p.x,p.z)-f32(p.y),0.0,1.0);}
fn umSolidWorldCell(id:vec3i)->vec3f{let h=umSolidParams.cellGravity.xyz;
 return vec3f(-0.5*umSolidParams.container.x+(f32(id.x)+0.5)*h.x,(f32(id.y)+0.5)*h.y,-0.5*umSolidParams.container.z+(f32(id.z)+0.5)*h.z);}
// The tile cut map (UniformMixedSolid, after the bed-slope lane in the mask
// image): one word per tile key, zero when no h cell of the tile or its
// one-cell ring has open < 1 -- every cell and face sample a helper takes for
// a cell of the tile, but for a wall page across a page face (umSolidFaceClear).
// A tile no page stores has no word and takes the full path. Bodies move
// after the map is built, so a cell near a body always takes the full path.
fn umSolidTileClear(p:vec3i)->bool{
 ${gated?`let t=umTileKey(p>>vec3u(2u));
 if(!umKeyStored(t)||umSolidScratch[umSolidCutMap()+t]!=0u){return false;}
 if(umSolidParams.boundary.z<0.5){return true;}
 let world=umSolidWorldCell(p);let h=umSolidParams.cellGravity.xyz;
 for(var i=0u;i<umBodyCount();i++){if(umBodyNear(i,world,3.0*max(h.x,max(h.y,h.z)))){return false;}}
 return true;`:"return false;"}
}
// The +axis face of id (neighbor across it) meets no solid: a wall page is
// solid in the directory and cuts no tile, so a face on a page's high side
// asks the page across it.
fn umSolidFaceClear(id:vec3i,neighbor:vec3i,axis:u32)->bool{return umSolidTileClear(id)&&((id[axis]&31)!=31||umSolidValid(neighbor));}
// Rigid bodies: the native RigidBody helpers over the shape table. Every
// body query first rejects on the body's bounding sphere (state dimensions.w,
// boundingRadius), so cells away from every body pay one distance per body.
${sceneShapeWgsl()}
fn umBodyCount()->u32{if(!umSolidEnabled()){return 0u;}return min(u32(max(round(umSolidParams.boundary.z),0.0)),12u);}
fn umQuatRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);let uuv=cross(q.yzw,uv);return v+2.0*(q.x*uv+uuv);}
fn umBodyLocal(i:u32,world:vec3f)->vec3f{let b=umSolidBodies[i];return umQuatRotate(vec4f(b.orientation.x,-b.orientation.yzw),world-b.positionShape.xyz);}
fn umBodyNear(i:u32,world:vec3f,margin:f32)->bool{let b=umSolidBodies[i];return distance(world,b.positionShape.xyz)<=b.dimensions.w+margin;}
fn umInsideBody(i:u32,world:vec3f)->bool{let b=umSolidBodies[i];return umBodyNear(i,world,0.0)&&rigidShapeInside(i32(round(b.positionShape.w)),b.dimensions.xyz,umBodyLocal(i,world));}
fn umBodyDistance(i:u32,world:vec3f)->f32{let b=umSolidBodies[i];return rigidShapeDistance(i32(round(b.positionShape.w)),b.dimensions.xyz,umBodyLocal(i,world));}
fn umBodyVelocity(i:u32,world:vec3f)->vec3f{let b=umSolidBodies[i];return b.linearVelocity.xyz+cross(b.angularVelocity.xyz,world-b.positionShape.xyz);}
// rigidBodyIndexAt: the first body containing the point, -1 for none.
fn umBodyAt(world:vec3f)->i32{
 for(var i=0u;i<umBodyCount();i++){if(umInsideBody(i,world)){return i32(i);}}
 return -1;
}
// bodySolidFraction: the CPU voxelizer's eight sub-cell samples.
fn umBodyFraction(i:u32,p:vec3i)->f32{
 let h=umSolidParams.cellGravity.xyz;let centre=umSolidWorldCell(p);
 if(!umBodyNear(i,centre,length(0.4*h))){return 0.0;}
 var inside=0.0;
 for(var corner=0u;corner<8u;corner++){
  let offset=vec3f(select(-0.4,0.4,(corner&1u)!=0u),select(-0.4,0.4,(corner&2u)!=0u),select(-0.4,0.4,(corner&4u)!=0u));
  if(umInsideBody(i,centre+offset*h)){inside+=1.0;}
 }
 return inside/8.0;
}
// bodySolidFractionAt: the largest body coverage of the cell.
fn umCellBodyFraction(p:vec3i)->f32{
 var fraction=0.0;
 for(var i=0u;i<umBodyCount();i++){fraction=max(fraction,umBodyFraction(i,p));}
 return fraction;
}
// cellOpenFraction.
fn umCellOpen(p:vec3i)->f32{
 if(!umSolidEnabled()){return select(0.0,1.0,umSolidValid(p));}
 if(umSolidTileClear(p)){return 1.0;}
 if(umSolidVoxel(p)){return 0.0;}
 return clamp((1.0-umCellBodyFraction(p))*(1.0-umCellTerrainFraction(p)),0.0,1.0);
}
// The exact >1e-5 predicate, cached by the solid record builder. Consumers
// of the coarse record run after its rebuild at the frame head (also after
// voxel/terrain edits and body motion). The builder, stages without a
// record and a cell of a tile with no current row (no page stores it) keep
// the full geometric evaluation;
// fractional queries still use umCellOpen. A cell of a wall page is closed in
// either form.
fn umCellIsOpen(p:vec3i)->bool{
 ${coarse!==undefined&&gated?`if(!umSolidEnabled()){return umSolidValid(p);}
 let t=umTileKey(p>>vec3u(2u));if(!umKeyStored(t)){return umCellOpen(p)>1e-5;}
 let l=vec3u(p)&vec3u(3u);let lane=l.x+4u*(l.y+4u*l.z);
 let mask=umSolidOpenCells(t);
 return (select(mask.x,mask.y,lane>=32u)&(1u<<(lane&31u)))!=0u;`:"return umCellOpen(p)>1e-5;"}
}
// cellInsideSolid: voxel, terrain or a body at the centre -- the CM11a p_min=0 rows.
fn umCellInsideSolid(p:vec3i)->bool{
 if(!umSolidEnabled()){return false;}
 if(umSolidTileClear(p)){return false;}
 if(umSolidVoxel(p)){return true;}
 return umCellInsideTerrain(p)||umBodyAt(umSolidWorldCell(p))>=0;
}
fn umSolidFaceWorld(id:vec3i,axis:u32)->vec3f{var world=umSolidWorldCell(id);world[axis]+=0.5*umSolidParams.cellGravity.xyz[axis];return world;}
fn umSolidTraceWorld(p:vec3f)->vec3f{let h=umSolidParams.cellGravity.xyz;
 return vec3f(-0.5*umSolidParams.container.x+p.x*h.x,p.y*h.y,-0.5*umSolidParams.container.z+p.z*h.z);}
fn umSolidVoxelAtWorld(world:vec3f)->bool{
 let h=umSolidParams.cellGravity.xyz;
 let p=vec3i(floor(vec3f((world.x+0.5*umSolidParams.container.x)/h.x,world.y/h.y,(world.z+0.5*umSolidParams.container.z)/h.z)));
 return umSolidVoxel(p);
}
fn umSolidTerrainAtWorld(world:vec3f)->bool{
 if(!umSolidHasTerrain()){return false;}
 let h=umSolidParams.cellGravity.xyz;let d=vec3i(umSolidDimensions());
 let x=clamp(i32(floor((world.x+0.5*umSolidParams.container.x)/h.x)),0,d.x-1);
 let z=clamp(i32(floor((world.z+0.5*umSolidParams.container.z)/h.z)),0,d.z-1);
 return world.y<umSolidTerrainHeight(x,z)*h.y;
}
fn umSolidAtWorld(world:vec3f)->bool{return umSolidVoxelAtWorld(world)||umSolidTerrainAtWorld(world)||umBodyAt(world)>=0;}
// insideAnyTraceSolid for a lattice-coordinate point: voxels and bodies
// (terrain is not a trace solid natively).
fn umSolidAtWorldCell(p:vec3f)->bool{
 if(!umSolidEnabled()){return false;}
 if(umSolidTileClear(vec3i(floor(p)))){return false;}
 let world=umSolidTraceWorld(p);return umSolidVoxelAtWorld(world)||umBodyAt(world)>=0;}
// pressureFaceData(id,axis)[axis]: the moving-wall velocity u_s of a MAC
// face. Its dual-cell samples classify ownership; each covering body's
// rigid velocity is evaluated at the face centre (static voxels and terrain
// are zero). A face whose samples hold no solid takes the nearest body's
// velocity within one cell (extrapolatedRigidVelocityAtFace), else zero.
// Wall faces and voxel-adjacent faces are static walls: zero.
fn umSolidFaceVelocity(id:vec3i,axis:u32)->f32{
 if(!umSolidEnabled()){return 0.0;}
 if(umBodyCount()==0u){return 0.0;}
 var neighbor=id;neighbor[axis]+=1;
 if(umSolidVoxel(id)||umSolidVoxel(neighbor)){return 0.0;}
 let world=umSolidFaceWorld(id,axis);let h=umSolidParams.cellGravity.xyz;let reach=max(h.x,max(h.y,h.z));
 var near=false;
 for(var i=0u;i<umBodyCount();i++){near=near||umBodyNear(i,world,2.0*reach);}
 if(!near){return 0.0;}
 var solid=0.0;var velocity=0.0;
 for(var sampleIndex=0u;sampleIndex<8u;sampleIndex+=1u){
  let sampleWorld=world+vec3f(select(-0.4,0.4,(sampleIndex&1u)!=0u)*h.x,select(-0.4,0.4,(sampleIndex&2u)!=0u)*h.y,select(-0.4,0.4,(sampleIndex&4u)!=0u)*h.z);
  if(umSolidVoxelAtWorld(sampleWorld)||umSolidTerrainAtWorld(sampleWorld)){solid+=1.0;continue;}
  let body=umBodyAt(sampleWorld);
  if(body>=0){solid+=1.0;velocity+=umBodyVelocity(u32(body),world)[axis];}
 }
 if(solid>0.0){return velocity/solid;}
 var nearest=reach;var result=0.0;
 for(var i=0u;i<umBodyCount();i++){
  let d=abs(umBodyDistance(i,world));
  if(d<=nearest){nearest=d;result=umBodyVelocity(i,world)[axis];}
 }
 return result;
}
// faceOpenFraction: the Sec. 3.6 transverse face aperture.
fn umFaceOpen(id:vec3i,axis:u32)->f32{
 if(!umSolidEnabled()){return 1.0;}
 var neighbor=id;neighbor[axis]+=1;
 if(umSolidFaceClear(id,neighbor,axis)){return 1.0;}
 if(umSolidVoxel(id)||umSolidVoxel(neighbor)){return 0.0;}
 let world=umSolidFaceWorld(id,axis);let h=umSolidParams.cellGravity.xyz;
 let tangentA=(axis+1u)%3u;let tangentB=(axis+2u)%3u;var solid=0.0;
 for(var sampleIndex=0u;sampleIndex<4u;sampleIndex+=1u){
  var sampleWorld=world;
  sampleWorld[tangentA]+=select(-0.35,0.35,(sampleIndex&1u)!=0u)*h[tangentA];
  sampleWorld[tangentB]+=select(-0.35,0.35,(sampleIndex&2u)!=0u)*h[tangentB];
  solid+=select(0.0,1.0,umSolidAtWorld(sampleWorld));
 }
 return 1.0-solid*0.25;
}
// pressureFaceData(id,axis).w, geometric and static: the CM11a dual-cell V.
// A wall page is voxels: its face holds half the open fraction of the cell
// this side of it, and nothing between two wall cells. A depth symmetry
// plane is such a wall, as for solid-free mixed frames and every coarse
// owner (the native CM11a record used V=0). An AIR page is open space.
fn umPressureFaceV(id:vec3i,axis:u32)->f32{
 if(!umSolidEnabled()){return 1.0;}
 var neighbor=id;neighbor[axis]+=1;
 if(umSolidFaceClear(id,neighbor,axis)){return 1.0;}
 if(umSolidVoxel(id)||umSolidVoxel(neighbor)){return 0.5*(umCellOpen(id)+umCellOpen(neighbor));}
 let world=umSolidFaceWorld(id,axis);let h=umSolidParams.cellGravity.xyz;var solid=0.0;
 for(var sampleIndex=0u;sampleIndex<8u;sampleIndex+=1u){
  let sampleWorld=world+vec3f(select(-0.4,0.4,(sampleIndex&1u)!=0u)*h.x,select(-0.4,0.4,(sampleIndex&2u)!=0u)*h.y,select(-0.4,0.4,(sampleIndex&4u)!=0u)*h.z);
  solid+=select(0.0,1.0,umSolidAtWorld(sampleWorld));
 }
 return 1.0-solid/8.0;
}
`;
}
