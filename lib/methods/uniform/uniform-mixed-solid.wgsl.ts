import { GPU_RIGID_BODY_CAPACITY } from "../../core/rigid-capacity";
import {uniformSolidCutMaskWGSL} from "./uniform-solid-cut-mask.wgsl";
import { sculptCollisionWGSL } from "../../core/sculpt-collision.wgsl";
import { fluidRigidBoundaryWGSL } from "../../core/fluid-rigid-boundary.wgsl";
import type { LevelSetSolidBoundarySource } from "../../core/levelset-consumer-abi";
import { uniformRigidCoverageSource } from "./uniform-rigid-coverage.wgsl";
import { uniformTerrainSamplingWGSL } from "./uniform-terrain-sampling.wgsl";
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
import {UNIFORM_OPEN_WORLD_TERRAIN_PARAMS_BYTES,uniformOpenWorldTerrainSampleWGSL} from "./uniform-open-world-terrain";
import { uniformWindowAddressWGSL } from "./uniform-window-address";
import {uniformMixedPressureStorage} from "./uniform-mixed-pressure-boundary.wgsl";
import {sceneShapeWgsl} from "../../core/scene-shape";
import {SOLID_MASK_TABLE_WORDS,SOLID_MASK_FULL,SOLID_MASK_WALL,solidOccupancyMaskWGSL,type SolidOccupancyMaskPages} from "../../core/solid-occupancy-mask";
import {UNIFORM_BED_SLOPE_LANE_DIRECTORY_WORDS,UNIFORM_BED_SLOPE_LAYERS,uniformBedSlopeDirectoryWord} from "./uniform-bed-slope-bake";
import type {UniformMixedLayout} from "./uniform-mixed-layout";
import {uniformMixedChangedTilesWGSL} from "./uniform-mixed-layout-builder";
import {uniformDetailPick} from "./uniform-detail-fields";
import {UNIFORM_PAGE_TABLE_WORDS,uniformBlockFaceWGSL,uniformFaceTileSlots,uniformLatticeAddressWGSL,type UniformLatticeAddress} from "./uniform-lattice-address";
import {uniformPagePoolWGSL,type UniformPagePoolLayout} from "./uniform-page-pool";
import {UNIFORM_LIVE_GROUP_GROUPS,UNIFORM_LIVE_KEY_GROUPS,uniformLiveListsOfWGSL} from "./uniform-live-lists";
import { uniformSolidBoundaryWGSL } from "./uniform-solid-boundary.wgsl";
import {geometricPlaneBoxWGSL} from "../../core/geometric-plane-box.wgsl";


/** The record's rows, by tile key: a row a key (umSolidTiles), then a wall
 * slot a tile of each of a block's six faces (umFaceTile), the all-4h
 * layout's own index (uniformMixedPressureStorage) up to umSolidCount; after
 * them a flag pair a key, three words a key, two boundary rows a key, the
 * field-cut row and the aperture stamp row of each key. */
function solidRecordRowsWGSL(keys:number):string{return `
fn umSolidTiles()->u32{return ${keys}u;}
fn umSolidCount()->u32{return ${keys+uniformFaceTileSlots(keys/512)}u;}
// The boundary-brick pointer and 64 cell validity bits (two rows a key; the
// second's z: a body reached the tile at its build, w: its slot word).
fn umSolidBoundaryRow(t:u32)->u32{return umSolidCount()+4u*umSolidTiles()+2u*t;}
// The cells the procedural field cuts (a row a key).
fn umSolidFieldCutRow(t:u32)->u32{return umSolidCount()+6u*umSolidTiles()+t;}
// The aperture stamp of a key: the tile's world coordinate and the epoch of
// its build. Every stored tile's, slot or none.
fn umSolidApertureStampRow(t:u32)->u32{return umSolidCount()+7u*umSolidTiles()+t;}
// The candidate bodies of each key (bodyMask, every head, at the pose it
// reads): bit i set where slot i < count is inverted or its bounding sphere
// comes within UM_BODY_MASK_REACH hmax (plus 1% of a cell for rounding) of
// the key's tile box (its cells' extent). Four keys a row, one key a lane,
// each an exact f32 of at most GPU_RIGID_BODY_CAPACITY (24) bits. Valid
// only while umSolidBodyPoseCurrent and the stamps match (umBodyTileMask):
// a block stamp row per 512 keys (its page, w 1 when live at the head) and
// one row of the frame origin and the body count.
const UM_BODY_MASK_REACH:u32=3u;
fn umBodyTileMaskRow(t:u32)->u32{return umSolidCount()+8u*umSolidTiles()+(t>>2u);}
fn umBodyBlockStampRow(b:u32)->u32{return umSolidCount()+8u*umSolidTiles()+umSolidTiles()/4u+b;}
fn umBodyMaskFrameRow()->u32{return umSolidCount()+8u*umSolidTiles()+umSolidTiles()/4u+umSolidTiles()/512u;}
// The swept-body region mask's mirror (UniformMixedSolid.bodyRegions): a vec4
// per 16h region after the per-tile rows and the candidate masks, then the
// pending/count header.
fn umBodyRegionBase()->u32{return umBodyMaskFrameRow()+1u;}
// The epoch the host advances with every solid edit, after the mask's header.
// y certifies that the record was built at the currently published body pose.
// Publishing bodies clears it; the last full/body record-build dispatch sets it.
fn umSolidApertureEpochRow()->u32{return umBodyRegionBase()+umSolidTiles()/64u+1u;}
// A tile slot's 77 rows, after the epoch: the exact 64 h-cell open fractions
// (16 rows, four cells a vec4 in x+4y+16z order), then the h face apertures
// (umPressureFaceV's value of each cell's +x, +y and +z face, four cells a
// vec4 in x+4y+16z order, 16 rows an axis), the slot's owner (x: its key + 1,
// 0 free; yzw: exact valid-cell cut mask, packed 24+24+16 bits), then the tile's low faces (4 rows an axis, the face's cell at
// l[axis+1]+4l[axis+2]: a face whose low cell no page stores reads them).
// Only a tile whose fractions or apertures are not all 1 holds a slot
// (UniformMixedSolid's slot capacity: every key, or the tiles solids and the
// box's walls can cut).
fn umSolidSlotRow(s:u32)->u32{return umSolidApertureEpochRow()+1u+77u*s;}
fn umSolidApertureSlotRow(s:u32)->u32{return umSolidSlotRow(s)+16u;}
// A tile's slot word, its decoded slot: none and every fraction and aperture
// 1 (UM_SOLID_WHOLE, a word of 0 or below: a word below 0 is a slot the tile
// holds while whole, freed by the next reclaim), none recorded
// (UM_SOLID_GEOMETRY, the word UM_SOLID_GEOMETRY_WORD: the slots ran out, so
// readers evaluate the solids), else the slot (the word is slot + 1).
const UM_SOLID_WHOLE:u32=0xffffffffu;const UM_SOLID_GEOMETRY:u32=0xfffffffeu;const UM_SOLID_GEOMETRY_WORD:f32=16777216.0;
fn umSolidSlotOfWord(w:f32)->u32{if(!(w>0.0)){return UM_SOLID_WHOLE;}if(w>=UM_SOLID_GEOMETRY_WORD){return UM_SOLID_GEOMETRY;}return u32(w)-1u;}
// A window move preserves a resident page's geometry. Stamp its world tile,
// not its changing position in the window; a reused key still changes identity.
fn umSolidApertureCoord(t:u32)->vec3i{return vec3i(umTileCoord(t))+8*umFramePage();}
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
${uniformWindowAddressWGSL(`${name}`,`${table}u`,`${keys/512}u`)}
${uniformLatticeAddressWGSL({constant:true,paged})}`;
}

/** Units of a bed-slope byte: 127 is half a cell of rise in a cell of run. */
export const UNIFORM_SOLID_BED_SLOPE_UNITS=254;
/** The parts of the solid mask image (SolidOccupancyMask) this library
 * places, in u32 words, whatever the lattice:
 *
 *   bedSlope  after the directory, the bed-slope lane's top table
 *             (uniform-bed-slope-bake): a word a super-block column (4x4
 *             page columns) x + 64 z of the lattice, the first word of its
 *             child table in the mask's buffer, 0 for none; a child word a
 *             page column (x&3) + 4 (z&3), the first word of its slot, 0
 *             for none. A slot holds UNIFORM_BED_SLOPE_LAYERS words a
 *             column (x&31) + 32 (z&31), a word a floor cell, lowest first:
 *             (y + 1) << 16 | its entry, 0 for a level one. An entry's low
 *             byte is the bed height's gradient along x and its high byte
 *             along z, in cells per cell times UNIFORM_SOLID_BED_SLOPE_UNITS,
 *             pointing uphill. Non-zero only for an open cell that sits on a
 *             solid voxel's top face near a partly solid page (umBedSlope).
 *   cutMap    after the lane's directory, a word per tile key
 *
 * The lane's top table is the mask's `laneWords`; its child tables and
 * slots lie past the image, where the owner places them. umBedSlope and umSolidCutMap derive
 * the same offsets. */
export function uniformSolidScratchLayout():{bedSlopeBaseWord:number;bedSlopeWords:number;cutMapOffsetWords:number}{
 const bedSlopeWords=UNIFORM_BED_SLOPE_LANE_DIRECTORY_WORDS;
 return {bedSlopeBaseWord:SOLID_MASK_TABLE_WORDS,bedSlopeWords,cutMapOffsetWords:SOLID_MASK_TABLE_WORDS+bedSlopeWords};
}
/** What the solver sizes its solid mask by (SolidOccupancyMask): the
 * lattice's walls (umWorldSolid's rule) and tile keys, the bed-slope lane's directory, and
 * at most a slot for every page of the box and of the first layer of pages
 * above an open top, and in an open world one for each block the pool can
 * store a page in, as many as a directory word can name. The solver seals
 * the mask at fewer once it holds the scene (seal); an edit that makes more
 * pages partly solid than that is refused by the mask. The mask's world is
 * the box's: its cell (0,0,0) is the box's low cell, at the window's page
 * address.boxPage. */
export function uniformSolidMaskPages(address:UniformLatticeAddress,halfSolid=false):SolidOccupancyMaskPages{
 const [x,y,z]=address.pages;
 const laneWords=uniformSolidScratchLayout().bedSlopeWords,firstSlot=Math.ceil((SOLID_MASK_TABLE_WORDS+laneWords+address.tileKeys)/1024);
 const walls={policy:address.world?3:address.openTop?2:1,origin:address.boxPage,pages:address.pages} as const;
 return {walls,tileKeys:address.tileKeys,slots:Math.max(0,Math.min(SOLID_MASK_FULL-firstSlot,x*y*z+(address.openTop?x*z:0)+(address.world?address.blocks-2:0))),
  laneWords,separateTerrain:true,halfSolid,origin:address.boxPage};
}
/** A bed slope as its lane entry: 16 bits, x in the low byte, each rounded
 * half away from zero as the bake rounds (uniform-bed-slope-bake). */
export function uniformSolidBedSlopeEntry(slopeX:number,slopeZ:number):number{
 const byte=(slope:number)=>Math.sign(slope)*Math.min(127,Math.floor(Math.abs(slope)*UNIFORM_SOLID_BED_SLOPE_UNITS+0.5))&0xff;
 return byte(slopeX)|byte(slopeZ)<<8;
}
/** The slope a lane holds for cell (x,y,z) of the lattice: umBedSlope's
 * read, of `words` the mask buffer's from word 0 (or any copy that holds
 * the lane's directory and slots at their words). */
export function uniformSolidBedSlopeAt(words:ArrayLike<number>,x:number,y:number,z:number):[x:number,z:number]{
 if(x<0||z<0||x>=8192||z>=8192)return [0,0];
 const slot=uniformBedSlopeDirectoryWord(words,uniformSolidScratchLayout().bedSlopeBaseWord,x>>5,z>>5);
 let entry=0;
 if(slot)for(let k=0,at=slot+UNIFORM_BED_SLOPE_LAYERS*((x&31)+32*(z&31));k<UNIFORM_BED_SLOPE_LAYERS;k++){const w=words[at+k]!;if(w>>>16===y+1){entry=w;break;}}
 return [(entry<<24>>24)/UNIFORM_SOLID_BED_SLOPE_UNITS,(entry<<16>>24)/UNIFORM_SOLID_BED_SLOPE_UNITS];
}

/** The swept-body dependency mask (UniformMixedSolid.bodyRegions): one u32
 * per 16h work region (region = umBodyRegionOfTile(key), the work control's
 * umPageOfTile), then UNIFORM_BODY_REGION_HEADER_WORDS header words.
 *
 *   bits 0-7   swept reach: floor(d / hmax), saturating at 255, where d is the
 *              distance from the region's 16h box to the support swept by
 *              every body slot whose pose or presence changed since the last
 *              record build: the capsule from its built centre to its current
 *              centre with the larger bounding radius (the bounding sphere
 *              covers every rotation). 0 intersects; 255 is nothing in 254h.
 *   bits 8-15  pose reach: the same against every present body's current
 *              bounding sphere, moving or not.
 *   bit 16     the region's block was not live when the mask was written: both
 *              reaches are 0, so a claimed or recycled page is never clean.
 *
 * A consumer with a complete dependency footprint of m h cells is dirty where
 * reach <= m (d <= m hmax implies it). Written by every encodeCoarse, at each
 * frame head after encodeBodies: old pose is the mirror the record was last
 * built with (the previous head), new the current mirror. A frame's own tail
 * integration reaches the next head's mask.
 * Header: [present bodies now, present bodies at the last build, moved-slot
 * bitmask, any moved]. After it, two words a region: bit k is the
 * bodies-only rebuild's test of the region's tile k (umSolidRegionTile), set
 * where the region's swept reach is at most 2 and a moved body, as last built
 * or now, comes within 2h of the tile (UNIFORM_BODY_TILE_WORDS). */
export const UNIFORM_BODY_REGION_HEADER_WORDS=4;
export const UNIFORM_BODY_REGION_UNKNOWN=0x10000;
/** The per-tile bits after the header: two words per region. */
export const UNIFORM_BODY_TILE_WORDS=2;
/** Derived geometry reads the record's open fractions and capacities (rebuilt
 * within 2h of a moved body's swept support, tile by tile) and each owner's
 * own cells: a region beyond this swept reach keeps its geometry (the work
 * control's finish and the geometry's body launch). */
export const UNIFORM_BODY_GEOMETRY_REACH=3;
/** Accessors over the mask bound as `var<storage,read> <name>:array<u32>`:
 * bodyRegionSwept, bodyRegionPose (reaches in h) and bodyRegionOfTile. */
export function uniformBodyRegionsWGSL(name:string):string{return /* wgsl */`
fn bodyRegionOfTile(t:u32)->u32{return 8u*(t>>9u)+((t>>2u)&1u)+2u*((t>>5u)&1u)+4u*((t>>8u)&1u);}
fn bodyRegionSwept(region:u32)->u32{return ${name}[region]&0xffu;}
fn bodyRegionPose(region:u32)->u32{return (${name}[region]>>8u)&0xffu;}
`;}

export interface UniformMixedSolidResources {
 analyticTerrain?:GPUBuffer;
 params:GPUBuffer;
 /** SolidOccupancyMask's image: the page directory, the tile cut map (a
  * word a tile key, all cut as the host uploads it; the record build
  * writes it) and the page slots. */
 scratch:GPUBuffer;terrain:GPUTexture;
 /** WebGPURigidBodySystem.stateBuffer: capacity-sized 128-byte body records,
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
 /** Slots of the record's exact fractions and apertures (77 rows each), held
  * only by tiles where some of them are not 1: at most the tiles within a
  * cell of a solid or of the box's walls. Absent (or at least every tile
  * key): each key its own slot. Fewer: a free stack, reclaimed ahead of each
  * build; a tile the stack cannot serve reads the solids themselves
  * (UM_SOLID_GEOMETRY), counted in word 1 of slotState. */
 slots?:number;
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
 * entries keep what they held and have no reader (a slot it held returns to
 * the free stack at the next reclaim). */
export interface UniformMixedSolidCoarse {
 readonly bindLayout:GPUBindGroupLayout;readonly bindGroup:GPUBindGroup;readonly record:GPUBuffer;
 /** Owners plus halo slots: the level-0 topology record's vec4 count. A
  * tile's (cut, h) flags follow at count + t, then three vec4 per tile at
  * count + tiles + 3t: its open h cells (16 bits a component, cell x+4y+16z)
  * and its 48 +face patch codes (24 bits a component, four a patch).
  * Two more vec4 per tile follow at count + 4tiles + 2t: the boundary-brick
  * pointer and 64 cell validity bits, in exact 16-bit float pieces, then the
  * field cut and aperture stamp rows; the tile slots of the exact fractions
  * and apertures follow the body mask (solidRecordRowsWGSL). */
 readonly count:number;
}

/** Workgroups of the bodies-only cut launch's fixed region grid at most. */
const BODY_REGION_GRID=1024;
/** buildRegions' 64-workgroup chunks over a pool's resident regions: each round of a chunk takes 64 regions (8 blocks), so 8 chunks take 64 blocks a round. */
const BODY_REGION_CHUNKS=8;
const RECORD_PASS:GPUComputePassDescriptor={label:"Uniform mixed all-4h solid record"},WIDTHS_PASS:GPUComputePassDescriptor={label:"Uniform mixed solid simulation widths"};
const NEW_BLOCKS_PASS:GPUComputePassDescriptor={label:"Uniform mixed solid record of claimed blocks"};
/** Blocks the claimed-block build's fixed grid holds a workgroup per tile
 * for; a commit that claims more is strided over the same grid. */
const NEW_BLOCK_GRID=8;
/** ceil(jobs/per) in whole steps: a fractional quotient is a heap number until its caller is optimized. */
const ceilDiv=(jobs:number,per:number):number=>{const up=jobs+per-1;return (up-up%per)/per;};

export class UniformMixedSolid {
 readonly surfaceBoundary:LevelSetSolidBoundarySource;
 readonly bindLayout:GPUBindGroupLayout;
 readonly bindGroup:GPUBindGroup;
 readonly coupledTiles:number;
 private readonly bodySource:GPUBuffer;
 private readonly bodies:GPUBuffer;
 private readonly analyticTerrain:GPUBuffer;
 private readonly ownsAnalyticTerrain:boolean;
 /** The all-4h record for band pressure (constructed with its all-4h layout). */
 readonly coarse?:UniformMixedSolidCoarse;
 /** build and the cut flags over the tile keys (in full, bodies-only), and
  * both over the pool's new list (claimed, cutsClaimed).
  * group: their output group over the topology last encoded with. */
 private builder?:{pipeline:GPUComputePipeline;bodies?:GPUComputePipeline;claimed?:GPUComputePipeline;cuts:GPUComputePipeline;cutsBodies:GPUComputePipeline;cutsClaimed?:GPUComputePipeline;surfaceClear:GPUComputePipeline;surfaceClearBodies:GPUComputePipeline;surfaceClearNew?:GPUComputePipeline;bodyMask:GPUComputePipeline;slotsReclaim?:GPUComputePipeline;slotsReclaimNew?:GPUComputePipeline;layout:GPUBindGroupLayout;group?:GPUBindGroup;topology?:GPUBuffer};
 /** The swept-body dependency mask (UNIFORM_BODY_REGION_* layout): a u32 per
  * 16h region and the header, rewritten by every encodeCoarse. */
 private readonly bodyMask?:GPUBuffer;
 /** The mask for stages that bind it directly (uniformBodyRegionsWGSL);
  * stages binding tileGroup read its record mirror (umBodySweptReach). */
 get bodyRegions():{readonly buffer:GPUBuffer;readonly regions:number;readonly headerWord:number}{
  if(!this.bodyMask||!this.coarseLayout)throw new Error("The swept-body region mask exists with the all-4h solid record only");
  const regions=this.coarseLayout.tiles.length/64;return {buffer:this.bodyMask,regions,headerWord:regions};
 }
 /** The record's tile slots (UniformMixedSolidResources.slots), and their
  * free stack: [top, tiles it could not serve, 0, 0, slot ids] when fewer
  * than the tile keys, else the two counters alone. */
 readonly slots:number=0;
 readonly slotState?:GPUBuffer;
 /** Byte offset in coarse.record of the mask's pending word (consumeBodyEvidence). */
 private bodyPendingByte=0;
 /** The aperture rows' epoch (umSolidApertureEpochRow): its byte, and the host's copy. */
 private epochByte=0;private readonly epoch=new Float32Array(4);
 /** The library's group as the record builder binds it: the mask image writable, for the tile cut map. */
 private readonly recordSolids?:{readonly layout:GPUBindGroupLayout;readonly group:GPUBindGroup};
 private readonly pool?:UniformMixedSolidPool;
 /** Workgroups of the claimed-block build's fixed grid. */
 private readonly claimedGroups:number;
 /** The reclaim's fixed grid over the pooled slots. */
 private get slotGroups():number{return Math.min(UNIFORM_LIVE_GROUP_GROUPS,ceilDiv(this.slots,64));}
 /** The body mirror the record was last built with: a bodies-only rebuild
  * revisits the tiles near it or the current mirror. */
 private builtBodies?:GPUBuffer;
 /** The next build must visit every tile (first build, voxel edit). */
 private full=true;
 private simulation?:{pipeline:GPUComputePipeline;layout:GPUBindGroupLayout;group?:GPUBindGroup;topology?:GPUBuffer;
  listed?:GPUComputePipeline;listLayout:GPUBindGroupLayout;list?:{readonly topology:GPUBuffer;readonly changes:GPUBuffer;readonly group:GPUBindGroup}};
 private built=false;
 private readonly paramsWritten?:()=>boolean;
 /** The device's workgroups per dimension: read once, the limits object is the device's to rebuild. */
 private readonly maxGroups:number;
 /** pool: the page pool whose commits claim this frame's blocks; the record
  * of each claimed block is then built by encodeNewBlocks. */
 constructor(private readonly device:GPUDevice,resources:UniformMixedSolidResources,private readonly coarseLayout?:UniformMixedLayout,pool?:UniformMixedSolidPool){
  if(resources.params.size<272||resources.terrain.format!=="r32float")throw new Error("Mixed solids require the native parameter block and terrain heightfield");
  if(resources.bodies.size<GPU_RIGID_BODY_CAPACITY*128)throw new Error("Mixed solids require the rigid-body state buffer");
  if(!Number.isSafeInteger(resources.coupledTiles)||resources.coupledTiles<0)throw new Error("Mixed solids require the host's coupled tile count");
  this.coupledTiles=resources.coupledTiles;this.maxGroups=device.limits.maxComputeWorkgroupsPerDimension;
  this.ownsAnalyticTerrain=!resources.analyticTerrain;
  this.analyticTerrain=resources.analyticTerrain??device.createBuffer({label:"Inactive analytic terrain",size:UNIFORM_OPEN_WORLD_TERRAIN_PARAMS_BYTES,usage:GPUBufferUsage.UNIFORM});
  if(coarseLayout&&!resources.paramsWritten)throw new Error("The coarse solid record requires the host's parameter-block write receipt");
  this.paramsWritten=resources.paramsWritten;
  this.bodySource=resources.bodies;
  this.bodies=device.createBuffer({label:"Uniform mixed solid body mirror",size:GPU_RIGID_BODY_CAPACITY*128,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  const bodies=this.bodies;
  this.surfaceBoundary={params:{buffer:resources.params,size:272},mask:{buffer:resources.scratch},terrain:resources.terrain,
   bodies:{buffer:bodies},analyticTerrain:{buffer:this.analyticTerrain}};
  const entries:GPUBindGroupLayoutEntry[]=[
   {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:2,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"2d"}},
   {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
  ];
  const resourcesOf=():GPUBindGroupEntry[]=>[
   {binding:0,resource:{buffer:resources.params,size:272}},{binding:1,resource:{buffer:resources.scratch}},
   {binding:2,resource:resources.terrain.createView()},{binding:4,resource:{buffer:bodies}},{binding:5,resource:{buffer:this.analyticTerrain}},
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
   if(resources.scratch.size<4*(uniformSolidScratchLayout().cutMapOffsetWords+coarseLayout.tiles.length))throw new Error(`The solid mask image of ${resources.scratch.size} bytes has no room for the tile cut map of ${coarseLayout.tiles.length} tile keys after its directory and bed-slope lane directory`);
   this.pool=pool;
   const writable=device.createBindGroupLayout({label:"Uniform mixed static solids of the record build",entries:entries.map(entry=>entry.binding===1?{...entry,buffer:{type:"storage" as const}}:entry)});
   this.recordSolids={layout:writable,group:device.createBindGroup({layout:writable,entries:resourcesOf()})};
   const count=uniformMixedPressureStorage(coarseLayout).count;
   // 8 vec4 rows per tile, the candidate masks, then the body mask mirror: a vec4 per 16h region
   // and its header, the epoch row and 77 rows a slot (umSolidSlotRow).
   const slots=this.slots=Math.min(tiles,Math.max(1,resources.slots??tiles));
   if(!Number.isSafeInteger(slots))throw new Error(`The solid record's ${resources.slots} slots are not a whole count`);
   // The candidate masks: four keys a row, a stamp row a block, a frame row (umBodyTileMaskRow).
   const perTile=8*tiles+tiles/4+tiles/512+1;
   const recordBytes=16*(count+perTile+tiles/64+2+77*slots);this.bodyPendingByte=16*(count+perTile+tiles/64);this.epochByte=16*(count+perTile+tiles/64+1);
   // The free stack (pooled slots): [top, tiles the stack could not serve, 0, 0, slot ids].
   this.slotState=device.createBuffer({label:"Uniform mixed solid record slots",size:4*(4+(slots<tiles?slots:0)),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
   if(slots<tiles){const stack=new Uint32Array(4+slots);stack[0]=slots;for(let i=0;i<slots;i++)stack[4+i]=i;device.queue.writeBuffer(this.slotState,0,stack);}
   this.bodyMask=device.createBuffer({label:"Uniform mixed swept-body region mask",size:4*((1+UNIFORM_BODY_TILE_WORDS)*tiles/64+UNIFORM_BODY_REGION_HEADER_WORDS),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
   if(recordBytes>device.limits.maxStorageBufferBindingSize)throw new RangeError(`Solid cell-fraction record needs ${recordBytes} bytes, above this device's storage binding limit ${device.limits.maxStorageBufferBindingSize}`);
   const record=device.createBuffer({label:"Uniform mixed all-4h solid record",size:recordBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
   // Epoch 1 (a zeroed stamp never matches): a key's aperture rows hold only from its build.
   this.epoch[0]=1;device.queue.writeBuffer(record,this.epochByte,this.epoch);
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
 get allocatedBytes():number{return (this.coarse?.record.size??0)+(this.slotState?.size??0)+(this.bodyMask?.size??0)+this.bodies.size+(this.builtBodies?.size??0)+(this.ownsAnalyticTerrain?this.analyticTerrain.size:0);}
 /** Mirror the rigid state into the library: at a frame's head, and after
  * the rigid integration for the census that follows it. */
 encodeBodies(encoder:GPUCommandEncoder):void{
  encoder.copyBufferToBuffer(this.bodySource,0,this.bodies,0,GPU_RIGID_BODY_CAPACITY*128);
  // Invalidate in queue order, including publication after tail integration.
  // A host write here could invalidate an earlier frame still in flight.
  if(this.coarse)encoder.clearBuffer(this.coarse.record,this.epochByte+4,4);
 }
 async initialize():Promise<void>{
  const layout=this.coarseLayout,coarse=this.coarse;if(!layout||!coarse||this.builder)return;
  const dispatchX=this.device.limits.maxComputeWorkgroupsPerDimension;
  const pool=this.pool,pooled=this.slots<layout.tiles.length;
  const module=this.device.createShaderModule({label:"Uniform mixed all-4h solid record",code:/* wgsl */`
${uniformMixedSolidWGSL(0,undefined,false,true)}
@group(1) @binding(0) var<storage,read_write> record:array<vec4f>;
override umDispatchX:u32=65535u;
@group(1) @binding(1) var<uniform> umBuiltBodies:array<UMRigidBody,${GPU_RIGID_BODY_CAPACITY}>;
@group(1) @binding(2) var<storage,read> topology:array<u32>;
${solidRecordAddressWGSL(layout,"topology")}
${pool?uniformPagePoolWGSL(pool.layout,1,3)+uniformLiveListsOfWGSL("upState",pool.layout):""}
@group(1) @binding(4) var<storage,read_write> bodyRegions:array<u32>;
const UM_REGIONS:u32=${layout.tiles.length/64}u;
${pooled?/* wgsl */`// Pooled slots: [top, tiles the stack could not serve, 0, 0, free slot ids].
@group(1) @binding(5) var<storage,read_write> solidSlots:array<atomic<u32>>;
const UM_SOLID_SLOTS:u32=${this.slots}u;
fn umSolidSlotOwner(s:u32)->u32{return u32(record[umSolidApertureSlotRow(s)+48u].x);}
// The tile's word for this build (lane 0 of its workgroup): the slot it holds
// (its word names a slot whose owner row names it back) while it needs one,
// else one off the stack. Builds only pop: a tile that no longer needs its
// slot writes 0, and the reclaim ahead of the next build frees it.
fn umSolidClaimSlot(t:u32,odd:bool)->f32{
 let held=umSolidSlotOfWord(record[umSolidBoundaryRow(t)+1u].w);
 if(held<UM_SOLID_SLOTS&&umSolidSlotOwner(held)==t+1u){return select(0.0,f32(held+1u),odd);}
 if(!odd){return 0.0;}
 let top=atomicSub(&solidSlots[0],1u);
 if(top==0u||top>UM_SOLID_SLOTS){atomicAdd(&solidSlots[0],1u);atomicAdd(&solidSlots[1],1u);return UM_SOLID_GEOMETRY_WORD;}
 let s=atomicLoad(&solidSlots[4u+top-1u]);
 record[umSolidApertureSlotRow(s)+48u]=vec4f(f32(t+1u),0.0,0.0,0.0);
 return f32(s+1u);
}
// Ahead of each build, alone in its launch (pushes only): a slot whose owner
// holds no tile now, or whose word no longer names it, returns to the stack.
fn umSolidReclaim(s:u32){
 let owner=umSolidSlotOwner(s);if(owner==0u){return;}
 let t=owner-1u;
 if(umTileLive(t)&&umSolidSlotOfWord(record[umSolidBoundaryRow(t)+1u].w)==s){return;}
 record[umSolidApertureSlotRow(s)+48u]=vec4f(0.0);
 let top=atomicAdd(&solidSlots[0],1u);atomicStore(&solidSlots[4u+top],s);
}
@compute @workgroup_size(64) fn slotsReclaim(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var s=g.x;s<UM_SOLID_SLOTS;s+=64u*groups.x){umSolidReclaim(s);}
}
${pool?/* wgsl */`// Ahead of the claimed blocks' build: only when the commit claimed some.
@compute @workgroup_size(64) fn slotsReclaimNew(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) groups:vec3u){
 if(upNewCount()==0u){return;}
 for(var s=g.x;s<UM_SOLID_SLOTS;s+=64u*groups.x){umSolidReclaim(s);}
}`:""}`:/* wgsl */`// Every key its own slot: the tile's word names it where the tile needs it.
fn umSolidClaimSlot(t:u32,odd:bool)->f32{return select(0.0,f32(t+1u),odd);}`}
// Bodies-only rebuild: a tile far from every body that moved, as last built
// and now, keeps its record (its cut ring and face samples reach two cells
// out). A body at rest was built where it stands: its tiles stay authoritative.
override umBodyTiles:bool=false;
fn umBodyTileNear(b:UMRigidBody,lo:vec3f,hi:vec3f,margin:f32)->bool{
 if(b.dimensions.w<=0.0){return false;}
 let p=b.positionShape.xyz;return distance(clamp(p,lo,hi),p)<=b.dimensions.w+margin;
}
// Slot i holds a body now / held one at the last record build.
fn umBodyPresentNow(i:u32)->bool{return i<umBodyCount()&&umSolidBodies[i].dimensions.w>0.0;}
fn umBodyPresentBuilt(i:u32)->bool{return i<bodyRegions[UM_REGIONS+1u]&&umBuiltBodies[i].dimensions.w>0.0;}
// The slot's solid changed since the build: presence, pose or shape (bitwise).
fn umBodyMoved(i:u32)->bool{
 let now=umBodyPresentNow(i);if(now!=umBodyPresentBuilt(i)){return true;}if(!now){return false;}
 let a=umSolidBodies[i];let b=umBuiltBodies[i];
 return any(bitcast<vec4u>(a.positionShape)!=bitcast<vec4u>(b.positionShape))||any(bitcast<vec4u>(a.dimensions)!=bitcast<vec4u>(b.dimensions))||any(bitcast<vec4u>(a.orientation)!=bitcast<vec4u>(b.orientation));
}
fn umBuilderRegionOfTile(t:u32)->u32{return 8u*(t>>9u)+((t>>2u)&1u)+2u*((t>>5u)&1u)+4u*((t>>8u)&1u);}
// A pose's signed distance (fluidRigidDistance) at a world point.
fn umBodyPoseDistance(b:UMRigidBody,world:vec3f)->f32{
 return fluidRigidDistance(i32(round(b.positionShape.w)),b.dimensions.xyz,umQuatRotate(vec4f(b.orientation.x,-b.orientation.yzw),world-b.positionShape.xyz));
}
// The tile a moved body may change: one moved-slot bitmask, the tile's 4h
// box. Its record reads the body at points within 2 hmax of that box (cut
// ring and face samples), its touched flag (umSolidBodyBuilt) the bounding
// sphere within hmax, and each face's aperture whether the sphere comes
// within |h|/2 of the face centre (umPressureFaceV's body-near form). A slot
// present at both poses whose flag agrees changes nothing where both poses'
// centre distances exceed, with one sign, the reach R from the tile's
// centre to those points (the shape distances are 1-Lipschitz, scene-shape.ts;
// the cup's max of exact distances is), so every point is outside both poses
// or inside both, and where outside, the sphere reaches every face centre of
// the tile at both poses or none at either.
// How a pose's bounding sphere, grown by m, meets a box: 0 misses all of
// it, 2 holds all of it, 1 neither (a slack keeps rounding out of both).
fn umBodySphereClass(b:UMRigidBody,lo:vec3f,hi:vec3f,m:f32,slack:f32)->u32{
 if(b.positionShape.w<0.0){return 2u;}
 let p=b.positionShape.xyz;let reach=b.dimensions.w+m;
 if(distance(clamp(p,lo,hi),p)>reach+slack){return 0u;}
 if(length(max(abs(lo-p),abs(hi-p)))<=reach-slack){return 2u;}
 return 1u;
}
fn umBodyTileSwept(tc:vec3u,moved:u32)->bool{
 let h=umSolidParams.cellGravity.xyz;let hmax=max(h.x,max(h.y,h.z));
 let lo=umSolidWorldCell(vec3i(tc*4u))-0.5*h;let hi=lo+4.0*h;
 let margin=2.0*hmax;let centre=0.5*(lo+hi);let reach=1.01*length(2.0*h+vec3f(margin));
 for(var i=0u;i<${GPU_RIGID_BODY_CAPACITY}u;i++){
  if(((moved>>i)&1u)==0u){continue;}
  let now=umBodyPresentNow(i);let built=umBodyPresentBuilt(i);let a=umSolidBodies[i];let b=umBuiltBodies[i];
  if(!((now&&umBodyTileNear(a,lo,hi,margin))||(built&&umBodyTileNear(b,lo,hi,margin)))){continue;}
  if(now!=built||umBodyTileNear(a,lo,hi,hmax)!=umBodyTileNear(b,lo,hi,hmax)){return true;}
  let d=vec2f(umBodyPoseDistance(a,centre),umBodyPoseDistance(b,centre));
  if(all(d<vec2f(-reach))){continue;}
  if(!all(d>vec2f(reach))){return true;}
  let near=vec2u(umBodySphereClass(a,lo,hi,0.5*length(h),1e-3*hmax),umBodySphereClass(b,lo,hi,0.5*length(h),1e-3*hmax));
  if(near.x!=near.y||near.x==1u){return true;}
 }
 return false;
}
// The bodies-only rebuild's test of tile t, as bodyMask wrote it this head
// (its region's swept reach at most 2, then umBodyTileSwept). Evaluated in
// the builders themselves, inlined ahead of umBuildTile, the test made
// buildRegions' Metal compile take 110-150 s of a cold construction; read as
// a bit it takes under a second.
const UM_BODY_TILE_BITS:u32=UM_REGIONS+${UNIFORM_BODY_REGION_HEADER_WORDS}u;
fn umBodyTileDirty(t:u32)->bool{
 let k=(t&3u)|(((t>>3u)&3u)<<2u)|(((t>>6u)&3u)<<4u);
 return ((bodyRegions[UM_BODY_TILE_BITS+${UNIFORM_BODY_TILE_WORDS}u*umBuilderRegionOfTile(t)+(k>>5u)]>>(k&31u))&1u)!=0u;
}
// Distance from a box to a segment's points: convex along the segment, so a
// ternary search brackets its minimum; the bracket width times the length
// bounds the remaining error (1-Lipschitz), returned as a lower bound.
fn umBoxSegmentDistance(lo:vec3f,hi:vec3f,a:vec3f,b:vec3f)->f32{
 var l=0.0;var r=1.0;
 for(var k=0u;k<24u;k++){
  let m1=l+(r-l)/3.0;let m2=r-(r-l)/3.0;
  let p1=mix(a,b,m1);let p2=mix(a,b,m2);
  if(distance(clamp(p1,lo,hi),p1)<=distance(clamp(p2,lo,hi),p2)){r=m2;}else{l=m1;}
 }
 let p=mix(a,b,0.5*(l+r));
 return distance(clamp(p,lo,hi),p)-(r-l)*distance(a,b)-1e-6*(1.0+length(a)+length(b));
}
fn umBodyReach(d:f32,hmax:f32)->u32{
 // NaN and negative distances intersect.
 if(!(d>0.0)){return 0u;}return u32(min(floor(d/hmax),255.0));
}
// A workgroup per 16h region, a lane per tile: the swept/pose reaches
// (UniformMixedSolid.bodyRegions and its mirror at umBodyRegionBase in the
// record), the region's tile bits (umBodyTileDirty: a region beyond a swept
// reach of 2 has none set), and each tile's candidate bodies
// (umBodyTileMaskRow). Lanes below the capacity take a body each.
var<workgroup> maskMoved:atomic<u32>;
var<workgroup> maskBodies:atomic<u32>;
var<workgroup> maskSwept:atomic<u32>;
var<workgroup> maskPose:atomic<u32>;
var<workgroup> maskTileBits:array<atomic<u32>,2>;
var<workgroup> maskTiles:array<u32,64>;
// Slot i's bounding sphere within the mask reach of a box (umBodyTileMaskRow).
fn umBodyMaskNear(i:u32,lo:vec3f,hi:vec3f)->bool{
 let b=umSolidBodies[i];let h=umSolidParams.cellGravity.xyz;let p=b.positionShape.xyz;
 return b.positionShape.w<0.0||distance(clamp(p,lo,hi),p)<=b.dimensions.w+(f32(UM_BODY_MASK_REACH)+0.01)*max(h.x,max(h.y,h.z));
}
@compute @workgroup_size(64) fn bodyMask(@builtin(workgroup_id) g:vec3u,@builtin(local_invocation_index) lane:u32){
 let region=g.x+umDispatchX*g.y;
 if(region==0u&&lane==0u){
  var moved=0u;for(var i=0u;i<${GPU_RIGID_BODY_CAPACITY}u;i++){if(umBodyMoved(i)){moved|=1u<<i;}}
  bodyRegions[UM_REGIONS]=umBodyCount();bodyRegions[UM_REGIONS+2u]=moved;bodyRegions[UM_REGIONS+3u]=u32(moved!=0u);
  // Pending: the first derived-geometry consumer after this head takes the swept evidence.
  record[umBodyRegionBase()+UM_REGIONS]=vec4f(1.0,f32(moved!=0u),f32(umBodyCount()),0.0);
  record[umBodyMaskFrameRow()]=vec4f(umSolidParams.frame.xyz,f32(umBodyCount()));
 }
 if(region>=UM_REGIONS){return;}
 let first=umSolidRegionTile(region,0u);
 let bits=UM_BODY_TILE_BITS+${UNIFORM_BODY_TILE_WORDS}u*region;
 let live=umTileLive(first);
 if((region&7u)==0u&&lane==0u){record[umBodyBlockStampRow(region>>3u)]=vec4f(vec3f(umBlockPage(region>>3u)),select(0.0,1.0,live));}
 if(!live){if(lane==0u){bodyRegions[region]=${UNIFORM_BODY_REGION_UNKNOWN}u;record[umBodyRegionBase()+region]=vec4f(0.0,0.0,1.0,0.0);bodyRegions[bits]=0u;bodyRegions[bits+1u]=0u;}return;}
 let h=umSolidParams.cellGravity.xyz;let hmax=max(h.x,max(h.y,h.z));
 let lo=umSolidWorldCell(vec3i(umTileCoord(first)*4u))-0.5*h;let hi=lo+16.0*h;
 if(lane==0u){atomicStore(&maskMoved,0u);atomicStore(&maskBodies,0u);atomicStore(&maskSwept,255u);atomicStore(&maskPose,255u);atomicStore(&maskTileBits[0],0u);atomicStore(&maskTileBits[1],0u);}
 workgroupBarrier();
 // umBodyReach is monotone in the distance, so the minimum of the slots'
 // reaches is the reach of their minimum distance.
 if(lane<${GPU_RIGID_BODY_CAPACITY}u){
  let i=lane;let now=umBodyPresentNow(i);let b=umSolidBodies[i];
  if(i<umBodyCount()&&umBodyMaskNear(i,lo,hi)){atomicOr(&maskBodies,1u<<i);}
  if(now){let p=b.positionShape.xyz;atomicMin(&maskPose,umBodyReach(distance(clamp(p,lo,hi),p)-b.dimensions.w,hmax));}
  if(umBodyMoved(i)){
   atomicOr(&maskMoved,1u<<i);
   let built=umBodyPresentBuilt(i);let o=umBuiltBodies[i];var swept=3.402823e38;
   if(now&&built){swept=umBoxSegmentDistance(lo,hi,o.positionShape.xyz,b.positionShape.xyz)-max(o.dimensions.w,b.dimensions.w);}
   else if(now){let p=b.positionShape.xyz;swept=distance(clamp(p,lo,hi),p)-b.dimensions.w;}
   else if(built){let p=o.positionShape.xyz;swept=distance(clamp(p,lo,hi),p)-o.dimensions.w;}
   atomicMin(&maskSwept,umBodyReach(swept,hmax));
  }
 }
 workgroupBarrier();
 let s=atomicLoad(&maskSwept);let q=atomicLoad(&maskPose);let moved=atomicLoad(&maskMoved);let near=atomicLoad(&maskBodies);
 let t=umSolidRegionTile(region,lane);let tc=umTileCoord(t);
 if(s<=2u&&umBodyTileSwept(tc,moved)){atomicOr(&maskTileBits[lane>>5u],1u<<(lane&31u));}
 let tlo=umSolidWorldCell(vec3i(tc*4u))-0.5*h;let thi=tlo+4.0*h;
 var mask=0u;var remaining=near;
 while(remaining!=0u){let i=firstTrailingBit(remaining);remaining&=remaining-1u;if(umBodyMaskNear(i,tlo,thi)){mask|=1u<<i;}}
 maskTiles[lane]=mask;
 workgroupBarrier();
 // Lanes 4k..4k+3 hold keys 4j..4j+3 (umSolidRegionTile): one row.
 if((lane&3u)==0u){record[umBodyTileMaskRow(t)]=vec4f(f32(maskTiles[lane]),f32(maskTiles[lane+1u]),f32(maskTiles[lane+2u]),f32(maskTiles[lane+3u]));}
 if(lane==0u){
  bodyRegions[region]=s|(q<<8u);record[umBodyRegionBase()+region]=vec4f(f32(s),f32(q),0.0,0.0);
  bodyRegions[bits]=atomicLoad(&maskTileBits[0]);bodyRegions[bits+1u]=atomicLoad(&maskTileBits[1]);
 }
}
var<workgroup> sums:array<vec4f,64>;
var<workgroup> lows:array<vec4f,64>;
// What depends on the solid alone and a per-frame kernel would recompute:
// the tile's open h cells (umCellOpen > 1e-5), and each +face patch's V where
// it is a whole number of eighths (the eight-sample V always is): code
// 8V + 1 of patch 16 axis + u + 4 v, 0 for any other V.
var<workgroup> openCells:array<atomic<u32>,2>;
var<workgroup> patchCodes:array<atomic<u32>,8>;
var<workgroup> boundaryCells:array<u32,64>;
// The tile's cells the procedural field cuts: a boundary cell with a node at
// or inside the surface. Elsewhere its trilinear value is positive throughout
// (bounded by the corners), so no contact or crossing can occur there.
var<workgroup> fieldCut:array<atomic<u32>,2>;
${solidRecordRowsWGSL(layout.tiles.length)}
${uniformBlockFaceWGSL}
// umBoundaryIndex of the all-4h layout: the wall slot of tile key t on face 2 axis + side of its block.
fn umSolidHalo(t:u32,axis:u32,side:u32)->u32{return umSolidTiles()+umFaceTile(t>>9u,2u*axis+side,umLocalTile(t));}
// The cut flag of the tile at key t (x of its flags, and its word of the tile
// cut map): an h cell of the tile or of its one-cell ring has open < 1. A
// cell of a page the lattice walls off cuts nothing. Nothing for a key that
// holds no tile.
// Fraction builds precede cut classification. Their exact valid-cell cut
// masks replace 216 repeated geometric tests with 27 subset intersections.
// Missing/stale/overflow rows retain the geometric predicate. The old and
// new body support already rebuild with a two-cell guard, so unchanged
// neighbors retain valid masks.
${uniformSolidCutMaskWGSL}
fn umCutNeighbor(tc:vec3u,i:u32)->bool{
 let d=vec3i(i32(i%3u)-1,i32((i/3u)%3u)-1,i32(i/9u)-1);
 let origin=4*(vec3i(tc)+d);let t=umTileKey(origin>>vec3u(2u));
 if(umTileLive(t)){
  let stamp=record[umSolidApertureStampRow(t)];
  if(all(vec3i(stamp.xyz)==umSolidApertureCoord(t))&&stamp.w==record[umSolidApertureEpochRow()].x){
   let slot=umSolidSlotOfWord(record[umSolidBoundaryRow(t)+1u].w);
   if(slot==UM_SOLID_WHOLE){return false;}
   if(slot!=UM_SOLID_GEOMETRY){
    let mask=umUnpackCutMask(record[umSolidApertureSlotRow(slot)+48u].yzw);
    return any((mask&umCutRingMasks[i])!=vec2u(0u));
   }
  }
 }
 // Only the neighbor's subset belongs to this tile's ring.
 let lo=select(vec3i(0),vec3i(3),d<vec3i(0));
 let hi=select(vec3i(4),vec3i(1),d>vec3i(0));
 for(var z=lo.z;z<hi.z;z++){for(var y=lo.y;y<hi.y;y++){for(var x=lo.x;x<hi.x;x++){
  let q=origin+vec3i(x,y,z);if(umSolidValid(q)&&umCellOpen(q)<1.0){return true;}
 }}}
 return false;
}
fn umCutTile(t:u32){
 if(!umTileLive(t)){return;}
 let tc=umTileCoord(t);if(umBodyTiles&&!umBodyTileDirty(t)){return;}
 var cut=0u;
 for(var i=0u;i<27u&&cut==0u;i++){
  if(umCutNeighbor(tc,i)){cut=1u;}
 }
 record[umSolidCount()+t].x=f32(cut);
 umSolidScratch[umSolidCutMap()+t]=cut;
}
// Every stored tile: a lane a tile key.
@compute @workgroup_size(64) fn cuts(@builtin(global_invocation_id) g:vec3u){umCutTile(g.x+umDispatchX*64u*g.y);}
// Bodies-only (umBodyTiles): a fixed grid strides the 16h regions, a lane a
// tile of the region; a region the swept support does not reach within 2h
// (umBodyTileDirty's margin) exits whole, on a workgroup-uniform decision. A
// free block's region (UNKNOWN) holds no tile to build.
var<workgroup> regionReach:u32;
var<workgroup> regionCuts:atomic<u32>;
fn umBuilderRegionDirty(region:u32)->bool{return region<UM_REGIONS&&(bodyRegions[region]&0x100ffu)<=2u;}
// An item is an eighth of a region: eight of its tiles, eight lanes a tile,
// each lane a strided eighth of the tile's 27 neighbor subsets (umCutTile's
// test, any order). The existing lane assignment and launch sizes stay fixed.
// With a page pool the items are the resident blocks' (ulResidentBlock): a
// free block's regions hold no tile, so the launch follows the pool's
// residents, not its reservation.
fn umBuilderRegionItems()->u32{return ${pool?"64u*ulResidentCount()":"8u*UM_REGIONS"};}
fn umBuilderItemRegion(item:u32)->u32{return ${pool?"8u*ulResidentBlock(item>>6u)+((item>>3u)&7u)":"item>>3u"};}
@compute @workgroup_size(64) fn cutsRegions(@builtin(workgroup_id) g:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var item=g.x;item<umBuilderRegionItems();item+=groups.x){
  let region=umBuilderItemRegion(item);
  if(lane==0u){regionReach=u32(umBuilderRegionDirty(region));atomicStore(&regionCuts,0u);}
  if(workgroupUniformLoad(&regionReach)==0u){continue;}
  let slot=lane>>3u;let t=umSolidRegionTile(region,8u*(item&7u)+slot);
  var live=umTileLive(t);var tc=vec3u(0u);
  if(live){tc=umTileCoord(t);live=!(umBodyTiles&&!umBodyTileDirty(t));}
  if(live){
   for(var i=lane&7u;i<27u;i+=8u){
    if(umCutNeighbor(tc,i)){atomicOr(&regionCuts,1u<<slot);break;}
   }
  }
  workgroupBarrier();
  if(live&&(lane&7u)==0u){
   let cut=(atomicLoad(&regionCuts)>>slot)&1u;
   record[umSolidCount()+t].x=f32(cut);
   umSolidScratch[umSolidCutMap()+t]=cut;
  }
  workgroupBarrier();
 }
}
// The record of the tile at key t, one workgroup: nothing for a key that
// holds no tile (a free block's). Every word but the cut flag is written, y
// as 4h until encodeSimulation reads the tile's word. A tile on a face of
// its block writes that face's wall slot too: the V a wall there holds,
// read only where a SOLID page lies across the face.
// widths: write y from the bound topology's tile word (an in-place rebuild,
// whose topology is the live ownership); a claimed block's tiles read 4h
// until encodeSimulation.
var<workgroup> buildDirty:u32;
var<workgroup> buildBodyCandidates:u32;
var<workgroup> apertures:array<vec3f,64>;
var<workgroup> lowApertures:array<vec3f,64>;
// Some fraction or aperture of the tile is not 1; the tile's slot word.
var<workgroup> slotOdd:atomic<u32>;
var<workgroup> fractionCut:array<atomic<u32>,2>;
var<workgroup> slotWord:f32;
fn umBuildTile(t:u32,lane:u32,widths:bool){
 if(!umTileLive(t)){return;}
 let tc=umTileCoord(t);let row=t;
 // The mask is read_write storage here: decide on one lane, exit uniformly.
 if(umBodyTiles){if(lane==0u){buildDirty=u32(umBodyTileDirty(t));}if(workgroupUniformLoad(&buildDirty)==0u){return;}}
 let l=vec3u(lane%4u,(lane/4u)%4u,lane/16u);let c=vec3i(tc*4u+l);
 if(lane==0u){for(var i=0u;i<2u;i++){atomicStore(&openCells[i],0u);atomicStore(&fieldCut[i],0u);atomicStore(&fractionCut[i],0u);}for(var i=0u;i<8u;i++){atomicStore(&patchCodes[i],0u);}atomicStore(&slotOdd,0u);}
 // Share the roster rejection across all fractions and face queries of
 // this tile. The three-cell radius covers its one-cell neighbor queries
 // and their sampling margins; each query retains its original near test.
 if(lane==0u){buildBodyCandidates=umBodyTileCandidates(tc);}
 umSolidQueryBodies=workgroupUniformLoad(&buildBodyCandidates);
 var own=vec4f(umCellOpen(c),0.0,0.0,0.0);var low=vec4f(0.0);var face=vec3f(0.0);var lowFace=vec3f(0.0);
 // Every cell's +faces (the aperture rows; a tile's own +faces also feed its
 // sums and patch codes), then the tile's low faces (its low aperture rows;
 // on its block's low side they feed the halo too).
 for(var k=0u;k<6u;k++){
  let axis=k%3u;let edge=l[axis]==0u;let below=edge&&(tc[axis]&7u)==0u;
  if(k>=3u&&!edge){continue;}
  var q=c;if(k>=3u){q[axis]-=1;}let v=umPressureFaceV(q,axis);
  if(k>=3u){lowFace[axis]=v;if(below){low[axis+1u]=v;}}else{face[axis]=v;if(l[axis]==3u){own[axis+1u]=v;}}
 }
 apertures[lane]=face;lowApertures[lane]=lowFace;
 workgroupBarrier();
 if(own.x>1e-5){atomicOr(&openCells[lane>>5u],1u<<(lane&31u));}
 if(umSolidValid(c)&&own.x<1.0){atomicOr(&fractionCut[lane>>5u],1u<<(lane&31u));}
 let fieldRecord=umBoundaryRecord(c);
 if(fieldRecord!=0u){
  let n=vec3u(c)&vec3u(7u);var lowest=1e30;
  for(var k=0u;k<8u;k++){let m=n+vec3u(k&1u,(k>>1u)&1u,k>>2u);lowest=min(lowest,bitcast<f32>(umSolidScratch[fieldRecord+m.x+9u*(m.y+9u*m.z)]));}
  if(lowest<=0.0){atomicOr(&fieldCut[lane>>5u],1u<<(lane&31u));}
 }
 for(var axis=0u;axis<3u;axis++){
  let eighths=8.0*own[axis+1u];if(l[axis]!=3u||eighths!=floor(eighths)||eighths<0.0||eighths>8.0){continue;}
  let slot=16u*axis+l[(axis+1u)%3u]+4u*l[(axis+2u)%3u];
  atomicOr(&patchCodes[slot/6u],(u32(eighths)+1u)<<(4u*(slot%6u)));
 }
 sums[lane]=own;lows[lane]=low;boundaryCells[lane]=fieldRecord;
 // The rows a slot holds: this cell's fraction, its +faces and its low faces
 // on the tile's low sides (the low rows' faces).
 var odd=own.x!=1.0||any(face!=vec3f(1.0));
 for(var axis=0u;axis<3u;axis++){if(l[axis]==0u&&lowFace[axis]!=1.0){odd=true;}}
 if(odd){atomicOr(&slotOdd,1u);}
 workgroupBarrier();
 if(lane==0u){slotWord=umSolidClaimSlot(t,atomicLoad(&slotOdd)!=0u);}
 let word=workgroupUniformLoad(&slotWord);let slotted=word>0.0&&word<UM_SOLID_GEOMETRY_WORD;let s=u32(max(word,1.0))-1u;
 // Preserve the exact h-cell fractions before the all-4h reduction consumes
 // the shared rows. This record rebuilds with edits, bodies and page claims.
 if(slotted&&lane<16u){
  let first=4u*lane;
  record[umSolidSlotRow(s)+lane]=vec4f(sums[first].x,sums[first+1u].x,sums[first+2u].x,sums[first+3u].x);
 }
 if(slotted&&lane<48u){
  let axis=lane/16u;let first=4u*(lane%16u);
  record[umSolidApertureSlotRow(s)+lane]=vec4f(apertures[first][axis],apertures[first+1u][axis],apertures[first+2u][axis],apertures[first+3u][axis]);
 }
 if(slotted&&lane==48u){record[umSolidApertureSlotRow(s)+48u]=vec4f(f32(t+1u),umPackCutMask(vec2u(atomicLoad(&fractionCut[0]),atomicLoad(&fractionCut[1]))));}
 if(lane==48u){record[umSolidApertureStampRow(t)]=vec4f(vec3f(umSolidApertureCoord(t)),record[umSolidApertureEpochRow()].x);}
 if(slotted&&lane>48u&&lane<61u){
  let axis=(lane-49u)/4u;var v=vec4f(0.0);
  for(var j=0u;j<4u;j++){let slot=4u*((lane-49u)%4u)+j;var m=vec3u(0u);m[(axis+1u)%3u]=slot%4u;m[(axis+2u)%3u]=slot/4u;v[j]=lowApertures[m.x+4u*(m.y+4u*m.z)][axis];}
  record[umSolidApertureSlotRow(s)+lane]=v;
 }
 workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){sums[lane]+=sums[lane+stride];lows[lane]+=lows[lane+stride];}workgroupBarrier();}
 if(lane==0u){
  // A 4-cell tile never crosses an 8-cell boundary brick. Every valid
  // record is therefore the same pointer; two masks retain edited holes.
  var pointer=0u;var valid=vec2u(0u);
  for(var cell=0u;cell<64u;cell++){
   if(boundaryCells[cell]!=0u){pointer=boundaryCells[cell];valid[cell>>5u]|=1u<<(cell&31u);}
  }
  // Store 16-bit pieces as exact floats, avoiding NaN/denormal payloads.
  let boundaryRow=umSolidBoundaryRow(t);
  record[boundaryRow]=vec4f(vec4u(pointer&65535u,pointer>>16u,valid.x&65535u,valid.x>>16u));
  // z: a body of this build's mirror reaches within one cell of the tile
  // (after pose publication, umCellOpen falls back to geometry there).
  let bh=umSolidParams.cellGravity.xyz;let blo=umSolidWorldCell(vec3i(tc*4u))-0.5*bh;let bhi=blo+4.0*bh;var touched=0u;
  for(var i=0u;i<umBodyCount();i++){if(umBodyTileNear(umSolidBodies[i],blo,bhi,max(bh.x,max(bh.y,bh.z)))){touched=1u;}}
  record[boundaryRow+1u]=vec4f(vec3f(vec3u(valid.y&65535u,valid.y>>16u,touched)),word);
  let cut=vec2u(atomicLoad(&fieldCut[0]),atomicLoad(&fieldCut[1]));
  record[umSolidFieldCutRow(t)]=vec4f(vec4u(cut.x&65535u,cut.x>>16u,cut.y&65535u,cut.y>>16u));
  let s=sums[0];let lo=lows[0];
  record[row]=vec4f(s.x/64.0,s.y/16.0,s.z/16.0,s.w/16.0);
  record[umSolidCount()+row].y=select(0.0,select(0.0,1.0,(topology[t]&0x80000000u)!=0u),widths);
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
 umBuildTile(g.x+umDispatchX*g.y,lane,true);
}
// Bodies-only: a fixed grid of UM_BODY_GRID workgroups (64 per 64 regions).
// Workgroup w builds tile w % 64 of the 64 regions w / 64 + k UM_BODY_GRID / 64,
// so one region's tiles spread over 64 workgroups. Its lanes read those 64
// regions' words at once; it then builds only the tiles of dirty ones (a
// workgroup-uniform bit mask), and exits after one read where none is.
const UM_BODY_GRID:u32=64u*((UM_REGIONS+63u)/64u);
var<workgroup> jobBits:array<atomic<u32>,2>;
var<workgroup> jobMask:vec2u;
${pool?/* wgsl */`// With a page pool, the regions are the resident blocks' (ulResidentBlock):
// a free block's regions hold no tile. The grid is a fixed whole number of
// 64-workgroup chunks; chunk c takes rounds of 64 regions, c + chunks k, a
// lane a region's word, and tile w % 64 of each dirty one.
@compute @workgroup_size(64) fn buildRegions(@builtin(workgroup_id) g:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let w=g.x+groups.x*g.y;let chunks=(groups.x*groups.y)>>6u;let chunk=w>>6u;if(chunk>=chunks){return;}
 let regions=8u*ulResidentCount();
 for(var base=64u*chunk;base<regions;base+=64u*chunks){
  if(lane<2u){atomicStore(&jobBits[lane],0u);}
  workgroupBarrier();
  let v=base+lane;
  if(v<regions&&umBuilderRegionDirty(8u*ulResidentBlock(v>>3u)+(v&7u))){atomicOr(&jobBits[lane>>5u],1u<<(lane&31u));}
  workgroupBarrier();
  if(lane==0u){jobMask=vec2u(atomicLoad(&jobBits[0]),atomicLoad(&jobBits[1]));}
  let mask=workgroupUniformLoad(&jobMask);
  for(var k=0u;k<64u;k++){
   if(((mask[k>>5u]>>(k&31u))&1u)!=0u){let r=base+k;umBuildTile(umSolidRegionTile(8u*ulResidentBlock(r>>3u)+(r&7u),w&63u),lane,true);}
  }
 }
}`:/* wgsl */`@compute @workgroup_size(64) fn buildRegions(@builtin(workgroup_id) g:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let w=g.x+groups.x*g.y;if(w>=UM_BODY_GRID){return;}
 if(lane<2u){atomicStore(&jobBits[lane],0u);}
 workgroupBarrier();
 if(umBuilderRegionDirty((w>>6u)+lane*(UM_BODY_GRID/64u))){atomicOr(&jobBits[lane>>5u],1u<<(lane&31u));}
 workgroupBarrier();
 if(lane==0u){jobMask=vec2u(atomicLoad(&jobBits[0]),atomicLoad(&jobBits[1]));}
 let mask=workgroupUniformLoad(&jobMask);
 for(var k=0u;k<64u;k++){
  if(((mask[k>>5u]>>(k&31u))&1u)!=0u){umBuildTile(umSolidRegionTile((w>>6u)+k*(UM_BODY_GRID/64u),w&63u),lane,true);}
 }
}`}
// Persistent certificate in the cut flag's unused z lane. A true bit proves
// the exact 27-tile predicate surface advection previously rebuilt each step:
// every cell whole (open, and none the procedural field cuts: the open bits
// count a cut cell's fraction open).
fn umBuildSurfaceClear(t:u32){
 if(!umTileLive(t)){return;}
 let p=vec3i(umTileCoord(t));var clear=true;
 for(var k=0u;k<27u;k++){
  let q=p+vec3i(vec3u(k%3u,(k/3u)%3u,k/9u))-vec3i(1);let n=umTileKey(q);
  if(!umKeyStored(n)){continue;}
  let mask=vec4u(record[umSolidCount()+umSolidTiles()+3u*n]);
  let cut=vec4u(record[umSolidFieldCutRow(n)]);
  clear=clear&&all(mask==vec4u(65535u))&&all(cut==vec4u(0u));
 }
 record[umSolidCount()+t].z=select(0.0,1.0,clear);
}
// One workgroup owns a 16h region inside a 32h storage page. Reuse the
// existing flag pass to establish a stronger, genuinely uncut certificate.
// Unlike z (canonical-open), w can eliminate whole solid operations. It is
// valid only until bodies change after this record build.
var<workgroup> regionSolid:atomic<u32>;
var<workgroup> regionRefresh:u32;
fn umSolidRegionTile(region:u32,lane:u32)->u32{
 let r=region&7u;
 return 512u*(region>>3u)+4u*(r&1u)+32u*((r>>1u)&1u)+256u*((r>>2u)&1u)
  +(lane&3u)+8u*((lane>>2u)&3u)+64u*(lane>>4u);
}
fn umBuildRegionClear(t:u32,lane:u32){
 if(lane==0u){
  atomicStore(&regionSolid,0u);
  if(umTileLive(t)){
   let h=umSolidParams.cellGravity.xyz;
   let lo=umSolidWorldCell(vec3i(umTileCoord(t)*4u))-0.5*h;
   let hi=lo+16.0*h;let margin=3.0*max(h.x,max(h.y,h.z));
   // The corner fractions alone can miss a small body at a cell centre.
   for(var i=0u;i<umBodyCount();i++){
    if(umBodyTileNear(umSolidBodies[i],lo,hi,margin)){atomicStore(&regionSolid,1u);break;}
   }
  }
 }
 workgroupBarrier();
 if(!umTileLive(t)){atomicOr(&regionSolid,1u);}
 else if(record[umSolidCount()+t].x>0.5){atomicOr(&regionSolid,1u);}
 workgroupBarrier();
 if(umTileLive(t)){record[umSolidCount()+t].w=select(0.0,1.0,atomicLoad(&regionSolid)==0u);}
}
var<workgroup> regionSwept:u32;
${pool?/* wgsl */`// With a page pool a fixed grid strides the resident blocks' regions
// (ulResidentBlock): a tile no page stores is written by neither helper.
@compute @workgroup_size(64) fn surfaceClear(@builtin(workgroup_id) g:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 // The last launch of a record build: later masks compare against this count.
 if(g.x==0u&&lane==0u){bodyRegions[UM_REGIONS+1u]=umBodyCount();record[umSolidApertureEpochRow()].y=1.0;}
 for(var visit=g.x;visit<8u*ulResidentCount();visit+=groups.x){
  let region=8u*ulResidentBlock(visit>>3u)+(visit&7u);
  // Bodies-only: z reads 27 tiles' open masks (rebuilt within 2h of the swept
  // support, so z changes within 6h of it) and w the cut flags and current
  // bodies within 3h. Further regions keep both bits.
  if(lane==0u){regionSwept=select(0u,bodyRegions[min(region,UM_REGIONS-1u)]&0x100ffu,umBodyTiles);}
  if(workgroupUniformLoad(&regionSwept)>8u){continue;}
  let t=umSolidRegionTile(region,lane);
  umBuildSurfaceClear(t);umBuildRegionClear(t,lane);
 }
}`:/* wgsl */`@compute @workgroup_size(64) fn surfaceClear(@builtin(workgroup_id) g:vec3u,@builtin(local_invocation_index) lane:u32){
 let region=g.x+umDispatchX*g.y;
 // The last launch of a record build: later masks compare against this count.
 if(region==0u&&lane==0u){bodyRegions[UM_REGIONS+1u]=umBodyCount();record[umSolidApertureEpochRow()].y=1.0;}
 // Bodies-only: z reads 27 tiles' open masks (rebuilt within 2h of the swept
 // support, so z changes within 6h of it) and w the cut flags and current
 // bodies within 3h. Further regions keep both bits.
 if(lane==0u){regionSwept=select(0u,bodyRegions[min(region,UM_REGIONS-1u)]&0x100ffu,umBodyTiles);}
 if(workgroupUniformLoad(&regionSwept)>8u){return;}
 let t=umSolidRegionTile(region,lane);
 umBuildSurfaceClear(t);umBuildRegionClear(t,lane);
}`}
${pool?/* wgsl */`// Each tile has exactly one writer, even overlapping new-page halos.
// No claims means no geometric neighbourhood work. A retirement may retain
// false (conservative), while every claim refreshes its entire incident halo.
// A fixed grid of workgroups strides the resident blocks' regions (a tile
// no page stores is written by neither helper).
@compute @workgroup_size(64) fn surfaceClearNew(@builtin(workgroup_id) g:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let moved=any(vec3u(upState[UP_AT_WINDOW+4u],upState[UP_AT_WINDOW+5u],upState[UP_AT_WINDOW+6u])!=vec3u(0u));
 if(lane==0u){regionRefresh=select(0u,1u,upNewCount()!=0u||moved);}
 if(workgroupUniformLoad(&regionRefresh)==0u){return;}
 for(var visit=g.x;visit<8u*ulResidentCount();visit+=groups.x){
 let t=umSolidRegionTile(8u*ulResidentBlock(visit>>3u)+(visit&7u),lane);
 if(umTileLive(t)){
  let p=vec3i(umTileCoord(t));var dirty=moved;
  for(var i=0u;i<upNewCount();i++){
   let low=8*vec3i(umBlockPage(upNewBlock(i)))-vec3i(1);dirty=dirty||(all(p>=low)&&all(p<=low+vec3i(9)));
  }
  if(dirty){umBuildSurfaceClear(t);}
 }
 // Claims/window moves can change incident cut rings. Refresh the cheap
 // region reduction for all live regions in this already-dispatched pass.
 umBuildRegionClear(t,lane);
 }
}
`:""}
${pool?/* wgsl */`// The tiles of the blocks the last commit claimed (the pool's new list): job
// j is tile j % 512 of new block j / 512, whose key is 512 x the block + that
// index. A fixed grid of workgroups strides the jobs, whatever their count.
@compute @workgroup_size(64) fn buildNew(@builtin(workgroup_id) g:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let jobs=512u*upNewCount();
 for(var job=g.x+groups.x*g.y;job<jobs;job+=groups.x*groups.y){umBuildTile(512u*upNewBlock(job>>9u)+(job&511u),lane,false);}
}
// Their cut flags: the same jobs on the same grid, a workgroup a tile, its
// lanes over the 27 neighbor subsets of the tile and its ring (umCutTile's set).
var<workgroup> cutAny:atomic<u32>;
@compute @workgroup_size(64) fn cutsNew(@builtin(workgroup_id) g:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let jobs=512u*upNewCount();
 for(var job=g.x+groups.x*g.y;job<jobs;job+=groups.x*groups.y){
  let t=512u*upNewBlock(job>>9u)+(job&511u);let live=umTileLive(t);
  if(lane==0u){atomicStore(&cutAny,0u);}
  workgroupBarrier();
  if(live){
   let tc=umTileCoord(t);
   for(var i=lane;i<27u;i+=64u){if(umCutNeighbor(tc,i)){atomicOr(&cutAny,1u);break;}}
  }
  workgroupBarrier();
  if(live&&lane==0u){let cut=atomicLoad(&cutAny);record[umSolidCount()+t].x=f32(cut);umSolidScratch[umSolidCutMap()+t]=cut;}
 }
}`:""}`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const output=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   ...(pool?[{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}} as const]:[]),{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   ...(pooled?[{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}} as const]:[])]});
  const builderLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.recordSolids!.layout,output]});
  // The bodies-only rebuild is dispatched once a body has moved (needs).
  const build=(entryPoint:string,umBodyTiles:number)=>this.device.createComputePipelineAsync({layout:builderLayout,compute:{module,entryPoint,constants:{umDispatchX:dispatchX,umBodyTiles}}});
  const [pipeline,bodies,cuts,cutsBodies,claimed,cutsClaimed,surfaceClear,surfaceClearNew,surfaceClearBodies,bodyMask]=await Promise.all([build("build",0),build("buildRegions",1),build("cuts",0),build("cutsRegions",1),pool?build("buildNew",0):undefined,pool?build("cutsNew",0):undefined,build("surfaceClear",0),pool?build("surfaceClearNew",0):undefined,build("surfaceClear",1),build("bodyMask",0)]);
  const [slotsReclaim,slotsReclaimNew]=pooled?await Promise.all([build("slotsReclaim",0),pool?build("slotsReclaimNew",0):undefined]):[];
  this.builtBodies=this.device.createBuffer({label:"Uniform mixed solid record body mirror",size:GPU_RIGID_BODY_CAPACITY*128,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  this.builder={bodies,pipeline,claimed,cuts,cutsBodies,cutsClaimed,surfaceClear,surfaceClearBodies,surfaceClearNew,bodyMask,slotsReclaim,slotsReclaimNew,layout:output};
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
// A key that holds no tile keeps its row. A fixed grid strides the list.
@compute @workgroup_size(64) fn widthsListed(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var i=g.x;i<umChangedCount();i+=64u*groups.x){
  let t=umChangedTile(i);if(!umTileLive(t)){continue;}
  record[umSolidCount()+t].y=select(0.0,1.0,(topology[t]&0x80000000u)!=0u);
 }
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
  const layout=this.coarseLayout;if(!layout)return;
  if(!this.builder)throw new Error("The coarse solid record is not initialized");
  if(!this.paramsWritten!())throw new Error("The coarse solid record was encoded before the host's first write of the native parameter block: built from a zero block it holds every tile uncut");
  const keys=layout.tiles.length,x=Math.min(keys,this.maxGroups),groups=ceilDiv(keys,64),gx=Math.min(groups,this.maxGroups),gy=ceilDiv(groups,gx);
  const pass=encoder.beginComputePass(RECORD_PASS);
  pass.setBindGroup(0,this.recordSolids!.group);pass.setBindGroup(1,this.builderGroup(topology));
  // The swept-body mask every head: from the record's built mirror to the current one.
  const regions=keys/64,rx=Math.min(regions,this.maxGroups);
  pass.setPipeline(this.builder.bodyMask);pass.dispatchWorkgroups(rx,ceilDiv(regions,rx));
  if(!this.built){
   // A full build: a lane (cuts), a workgroup (build) per tile key. Bodies
   // only: fixed grids over the regions (cutsRegions) and their tile jobs
   // (buildRegions), each skipping a region the mask puts beyond reach.
   if(this.builder.slotsReclaim){pass.setPipeline(this.builder.slotsReclaim);pass.dispatchWorkgroups(this.slotGroups);}
   if(this.full){
    pass.setPipeline(this.builder.pipeline);pass.dispatchWorkgroups(x,ceilDiv(keys,x));
    pass.setPipeline(this.builder.cuts);pass.dispatchWorkgroups(gx,gy);
   }else{
    // With a pool, a fixed grid of chunks strides the resident blocks' regions.
    const grid=64*(this.pool?BODY_REGION_CHUNKS:ceilDiv(keys/64,64)),bx=Math.min(grid,this.maxGroups);
    pass.setPipeline(this.builder.bodies!);pass.dispatchWorkgroups(bx,ceilDiv(grid,bx));
    pass.setPipeline(this.builder.cutsBodies);pass.dispatchWorkgroups(Math.min(8*keys/64,BODY_REGION_GRID));
   }
   pass.setPipeline(this.full?this.builder.surfaceClear:this.builder.surfaceClearBodies);
   if(this.pool)pass.dispatchWorkgroups(UNIFORM_LIVE_GROUP_GROUPS);else pass.dispatchWorkgroups(gx,gy);
  }
  pass.end();
  if(this.built)return;
  encoder.copyBufferToBuffer(this.bodies,0,this.builtBodies!,0,GPU_RIGID_BODY_CAPACITY*128);this.full=false;
  this.built=true;
 }
 /** The first derived-geometry consumer after a head takes the mask's swept
  * evidence once (its pending word): its retained results then hold the
  * current pose, so later launches in the frame compare phi alone. */
 consumeBodyEvidence(encoder:GPUCommandEncoder):void{if(this.coarse)encoder.clearBuffer(this.coarse.record,this.bodyPendingByte,16);}
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
  if(this.builder!.slotsReclaimNew){pass.setPipeline(this.builder!.slotsReclaimNew);pass.dispatchWorkgroups(this.slotGroups);}
  pass.setPipeline(claimed);pass.dispatchWorkgroups(this.claimedGroups,1);
  pass.setPipeline(cuts);pass.dispatchWorkgroups(this.claimedGroups,1);
  pass.setPipeline(this.builder!.surfaceClearNew!);pass.dispatchWorkgroups(Math.min(UNIFORM_LIVE_GROUP_GROUPS,this.coarseLayout.tiles.length/64));pass.end();
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
    ...(this.pool?[{binding:3,resource:{buffer:this.pool.state}}]:[]),{binding:4,resource:{buffer:this.bodyMask!}},
    ...(b.slotsReclaim?[{binding:5,resource:{buffer:this.slotState!}}]:[])]});
  }
  return b.group;
 }
 /** The record does not describe the published solids yet (never built, or
  * invalidated): a reader ahead of the next frame's head builds it first. */
 get recordStale():boolean{return !!this.coarseLayout&&!this.built;}
 /** A live voxel edit changed the mask (every tile), or only bodies moved
  * (the tiles near them): rebuild the all-4h record. */
 invalidate(bodiesOnly=false):void{
  this.built=false;if(bodiesOnly)return;
  // An edit: every key's aperture rows are stale until the full build restamps them.
  this.full=true;if(this.coarse){this.epoch[0]++;this.device.queue.writeBuffer(this.coarse.record,this.epochByte,this.epoch);}
 }
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
   pass.setPipeline(s.listed!);pass.setBindGroup(0,s.list.group);pass.dispatchWorkgroups(Math.min(UNIFORM_LIVE_KEY_GROUPS,groups));pass.end();return;
  }else{
   if(s.topology!==topology.buffer){this.paged(topology);s.topology=topology.buffer;s.group=this.device.createBindGroup({layout:s.layout,entries:[{binding:0,resource:{buffer:coarse.record}},{binding:1,resource:topology}]});}
   pass.setPipeline(s.pipeline);pass.setBindGroup(0,s.group!);
  }
  pass.dispatchWorkgroups(x,ceilDiv(groups,x));pass.end();
 }
 destroy():void{this.coarse?.record.destroy();this.slotState?.destroy();this.bodyMask?.destroy();this.bodies.destroy();this.builtBodies?.destroy();if(this.ownsAnalyticTerrain)this.analyticTerrain.destroy();}
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
// The capacity of one incident cell at its owner's width; t: the key of its tile.
fn umVertexCellCapacity(cell:vec3i,t:u32)->bool{
 if(!umSolidValid(cell)){return false;}
 if(!umKeyStored(t)){return umTileOpenAt(cell>>vec3u(2u))>1e-5;}
 if(umTileWidth(t)==4u){return umTileOpen(t)>1e-5;}
 return umStoredCellIsOpen(cell,t);
}
fn umVertexBuried(p:vec3i)->bool{
 let base=p-vec3i(1);
 for(var k=0u;k<8u;k++){
  let cell=base+vec3i(umCorner(k,2u));
  if(umVertexCellCapacity(cell,umTileKey(cell>>vec3u(2u)))){return false;}
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
fn umSolidBodyPoseCurrent()->bool{return false;}
fn collisionSculptNear(world:vec3f)->bool{return false;}
fn umTerrainBoundary(q:vec3f)->vec4f{return vec4f(0,1,0,1e6);}
fn umSolidRegionClearAtBuild(p:vec3i)->bool{return true;}
fn umSolidPresent()->bool{return false;}
fn umSolidSkip()->bool{return false;}
fn umSolidValid(p:vec3i)->bool{return umCellKind(p)!=UM_SOLID;}
fn umSolidVoxel(p:vec3i)->bool{return false;}
fn umBedSlope(p:vec3i)->vec2f{return vec2f(0);}
fn umBoundaryBase()->u32{return 0u;}
fn umBoundaryRecord(p:vec3i)->u32{return 0u;}
fn umBoundarySampleAt(q:vec3f,c:vec3i,r:u32)->vec4f{return vec4f(0,0,0,1e6);}
fn umBoundarySample(p:vec3f)->vec4f{return vec4f(0,0,0,1e6);}
fn umCellOpen(p:vec3i)->f32{return select(0.0,1.0,umSolidValid(p));}
fn umCellIsOpen(p:vec3i)->bool{return umSolidValid(p);}
fn umStoredCellIsOpen(p:vec3i,t:u32)->bool{return umSolidValid(p);}
fn umCellInsideSolid(p:vec3i)->bool{return false;}
fn umFaceOpen(id:vec3i,axis:u32,candidates:u32)->f32{return 1.0;}
fn umPressureFaceV(id:vec3i,axis:u32)->f32{return 1.0;}
fn umPressureFaceVRecorded(id:vec3i,axis:u32)->f32{return 1.0;}
fn umSolidAperturesServe(p:vec3i)->u32{return 0u;}
fn umSolidApertureCell(id:vec3i,axis:u32)->vec3i{return id;}
fn umSolidApertureRows(id:vec3i,axis:u32)->u32{return 0u;}
fn umSolidApertureAt(rows:u32,id:vec3i,axis:u32)->f32{return 1.0;}
fn umSolidApertureRecord(id:vec3i,axis:u32)->f32{return 1.0;}
fn umSolidAtWorldCell(p:vec3f)->bool{return false;}
fn umSolidFaceVelocity(id:vec3i,axis:u32)->f32{return 0.0;}
fn umBodyCount()->u32{return 0u;}
fn umTileOpen(t:u32)->f32{return 1.0;}
fn umTileOpenAt(p:vec3i)->f32{return 1.0;}
fn umTileSupported(t:u32)->bool{return false;}
fn umBodyRegionOfTile(t:u32)->u32{return 8u*(t>>9u)+((t>>2u)&1u)+2u*((t>>5u)&1u)+4u*((t>>8u)&1u);}
fn umBodySweptReach(region:u32)->u32{return 255u;}
fn umBodyPoseReach(region:u32)->u32{return 255u;}
fn umBodySweptNearTile(t:u32,margin:u32)->bool{return false;}
fn umBodyPoseNearTile(t:u32,margin:u32)->bool{return false;}
fn umBodyEvidencePending()->bool{return false;}
`;
 const bodyLoop=writable?"var bodyCandidates=umSolidQueryBodies&((1u<<umBodyCount())-1u);while(bodyCandidates!=0u){let i=firstTrailingBit(bodyCandidates);bodyCandidates&=bodyCandidates-1u;":"for(var i=0u;i<umBodyCount();i++){";
 return /* wgsl */ `
struct UMSolidParams {
 dimsDt:vec4f,cellGravity:vec4f,frame:vec4f,physical:vec4f,boundary:vec4f,
 inflowA:vec4f,inflowB:vec4f,inflowC:vec4f,tuning:vec4f,drop:vec4f,dropExtent:vec4f,
 twoLevel:vec4f,agreement:vec4f,lean:vec4f,splash:vec4f,splashB:vec4f,cleanup:vec4f,
}
@group(${group}) @binding(0) var<uniform> umSolidParams:UMSolidParams;
fn umSolidDimensions()->vec3u{return vec3u(umSolidParams.dimsDt.xyz);}
// The box's low cell in the lattice (tuning.xyz): what the box-shaped solid
// stores (the bed-slope lane) are indexed from. Zero but in an open world.
fn umSolidBoxCell()->vec3i{return vec3i(umSolidParams.tuning.xyz);}
@group(${group}) @binding(1) var<storage,${writable?"read_write":"read"}> umSolidScratch:array<u32>;
@group(${group}) @binding(2) var umSolidTerrain:texture_2d<f32>;
struct UMAnalyticTerrain{frame:vec4f,ground:vec4f,features:array<vec4f,${(UNIFORM_OPEN_WORLD_TERRAIN_PARAMS_BYTES-32)/16}>}
@group(${group}) @binding(5) var<uniform> umAnalyticTerrain:UMAnalyticTerrain;
${uniformOpenWorldTerrainSampleWGSL("umAnalyticTerrain")}
struct UMRigidBody {
 positionShape:vec4f,dimensions:vec4f,orientation:vec4f,linearVelocity:vec4f,
 angularVelocity:vec4f,inverseMassInertia:vec4f,angularMomentumRestitution:vec4f,material:vec4f,
}
@group(${group}) @binding(4) var<uniform> umSolidBodies:array<UMRigidBody,${GPU_RIGID_BODY_CAPACITY}>;
// lean.x certifies that voxels, terrain and the live body roster are empty.
${gated?`// A pipeline compiled with umSolidFree has its solid code folded away: its
// kernel runs only while umSolidPresent() is false (it tests the gate
// itself) and leaves the step to a general pipeline otherwise. A general
// pipeline compiled with umSolidTwinned is launched beside such a twin of
// its entry, and its kernel runs only while a solid is present: an entry
// of the pair returns at once where umSolidSkip(), so exactly one of the
// two launches does the work, chosen on the GPU each step.
override umSolidFree:bool=false;
override umSolidTwinned:bool=false;
fn umSolidPresent()->bool{return umSolidParams.lean.x<0.5;}
fn umSolidEnabled()->bool{return !umSolidFree&&umSolidPresent();}
fn umSolidSkip()->bool{return select(umSolidTwinned&&!umSolidPresent(),umSolidPresent(),umSolidFree);}`:"fn umSolidEnabled()->bool{return true;}"}
${coarse===undefined?`fn umSolidRegionClearAtBuild(p:vec3i)->bool{return false;}
// No record mirror: any present body may reach any region.
fn umBodySweptReach(region:u32)->u32{return select(255u,0u,umBodyCount()>0u);}
fn umBodyPoseReach(region:u32)->u32{return select(255u,0u,umBodyCount()>0u);}
fn umBodySweptNearTile(t:u32,margin:u32)->bool{return umBodyCount()>0u;}
fn umBodyPoseNearTile(t:u32,margin:u32)->bool{return umBodyCount()>0u;}
fn umBodyEvidencePending()->bool{return umBodyCount()>0u;}
fn umBodyTileMask(t:u32)->u32{return (1u<<umBodyCount())-1u;}
fn umBodyCellMask(p:vec3i)->u32{return (1u<<umBodyCount())-1u;}
fn umTileOpen(t:u32)->f32{return 1.0;}
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
// The same of the tile at p, whatever its page. An unstored tile of a page
// the lattice walls off is closed throughout: each of its cells' umCellOpen
// is umCellOpenGeometry's voxel answer, 0 (no tile clearance or boundary
// record off a stored valid page), so their mean is 0 without the 64.
fn umTileOpenAt(p:vec3i)->f32{
 let t=umTileKey(p);if(umKeyStored(t)){return umTileOpen(t);}
 if(!umSolidEnabled()){return 1.0;}
 if(umSolidPage(4*p)==${SOLID_MASK_WALL}u){return 0.0;}
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
// A conservative proof of no closed cell in the tile's 27-tile halo.
// Invalidated with record builds and refreshed around every page claim.
fn umSolidSurfaceClear(t:u32)->bool{return umSolidRecord[umSolidCount()+t].z>0.5;}
// Head-of-frame certificate only: do not use after rigid integration moves
// bodies. Domain-wall cells/faces still require umSolidValid explicitly.
fn umSolidRegionClearAtBuild(p:vec3i)->bool{
 if(!umSolidEnabled()){return true;}
 let t=umTileKey(p>>vec3u(2u));if(!umKeyStored(t)){return false;}
 return umSolidRecord[umSolidCount()+t].w>0.5;
}
// The swept-body region mask's record mirror (UniformMixedSolid.bodyRegions):
// reaches in h from a 16h region to the support swept by bodies that moved
// since the last record build (swept) and to every present body (pose).
// Current from the frame head's record pass for the rest of the frame.
// A region index past the record's (another layout's) reads as reached.
fn umBodySweptReach(region:u32)->u32{if(!umSolidEnabled()){return 255u;}if(region>=umSolidTiles()/64u){return 0u;}return u32(umSolidRecord[umBodyRegionBase()+region].x);}
fn umBodyPoseReach(region:u32)->u32{if(!umSolidEnabled()){return 255u;}if(region>=umSolidTiles()/64u){return 0u;}return u32(umSolidRecord[umBodyRegionBase()+region].y);}
// Tile key t's region is within margin h of the swept/pose support; a key no
// page stores answers true.
fn umBodySweptNearTile(t:u32,margin:u32)->bool{if(!umSolidEnabled()){return false;}if(!umKeyStored(t)){return true;}return umBodySweptReach(umBodyRegionOfTile(t))<=margin;}
// The candidate bodies of key t (umBodyTileMaskRow): every slot below the
// count where the head's masks do not certify the pose, the frame, the
// count and t's block. Every body query of a point of the tile's cells
// with a margin up to UM_BODY_MASK_REACH hmax visits only these: a slot
// outside the mask lies farther than that from all of them.
fn umBodyTileMask(t:u32)->u32{
 let every=(1u<<umBodyCount())-1u;
 if(!umSolidEnabled()||t>=umSolidTiles()||!umTileLive(t)||!umSolidBodyPoseCurrent()){return every;}
 let b=t>>9u;let stamp=umSolidRecord[umBodyBlockStampRow(b)];let frame=umSolidRecord[umBodyMaskFrameRow()];
 if(stamp.w!=1.0||any(stamp.xyz!=vec3f(umBlockPage(b)))||any(frame.xyz!=umSolidParams.frame.xyz)||frame.w!=f32(umBodyCount())){return every;}
 return u32(umSolidRecord[umBodyTileMaskRow(t)][t&3u])&every;
}
// The candidates of h cell p's tile.
fn umBodyCellMask(p:vec3i)->u32{return umBodyTileMask(umTileKey(p>>vec3u(2u)));}
fn umBodyPoseNearTile(t:u32,margin:u32)->bool{
 if(!umSolidEnabled()){return false;}if(!umKeyStored(t)){return true;}
 if(umBodyPoseReach(umBodyRegionOfTile(t))>margin){return false;}
 // Region reach is a broad phase only: a body must reach this tile itself.
 let h=umSolidParams.cellGravity.xyz;let lo=umSolidWorldCell(vec3i(4u*umTileCoord(t)))-.5*h;let hi=lo+4.0*h;
 var remaining=select((1u<<umBodyCount())-1u,umBodyTileMask(t),margin<=UM_BODY_MASK_REACH);
 while(remaining!=0u){let i=firstTrailingBit(remaining);remaining&=remaining-1u;let b=umSolidBodies[i];if(b.dimensions.w<=0.0){continue;}
  if(distance(clamp(b.positionShape.xyz,lo,hi),b.positionShape.xyz)<=b.dimensions.w+f32(margin)*max(h.x,max(h.y,h.z))){return true;}}
 return false;
}
// This head's swept evidence has not been taken by a derived-geometry launch yet.
fn umBodyEvidencePending()->bool{return umSolidEnabled()&&umSolidRecord[umBodyRegionBase()+umSolidTiles()/64u].x>0.5;}
// A body of the record's build mirror reached within a cell of tile t.
fn umSolidBodyBuilt(t:u32)->bool{return umSolidRecord[umSolidBoundaryRow(t)+1u].z>0.5;}
// This certificate survives page claims and derived-geometry consumers, but
// not body publication. It permits exact reuse even beside a moving body:
// between its record rebuild and its next integration the pose is unchanged.
fn umSolidBodyPoseCurrent()->bool{return umSolidRecord[umSolidApertureEpochRow()].y>0.5;}
// Stored tile t's slot (umSolidSlotOfWord).
fn umSolidTileSlot(t:u32)->u32{return umSolidSlotOfWord(umSolidRecord[umSolidBoundaryRow(t)+1u].w);}
// The recorded open fraction of h cell p of stored tile t, or -1 where no
// slot holds it (the solids' own evaluation serves it).
fn umSolidRecordedOpen(t:u32,p:vec3i)->f32{
 let slot=umSolidTileSlot(t);if(slot==UM_SOLID_WHOLE){return 1.0;}if(slot==UM_SOLID_GEOMETRY){return -1.0;}
 let local=vec3u(p)&vec3u(3u);let cell=local.x+4u*(local.y+4u*local.z);
 return umSolidRecord[umSolidSlotRow(slot)+(cell>>2u)][cell&3u];
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
${solidOccupancyMaskWGSL("umSolidScratch","umSolidPage","umSolidVoxel",true)}
// The cell is not in a page the lattice walls off: one directory word.
fn umSolidValid(p:vec3i)->bool{return umSolidPage(p)!=${SOLID_MASK_WALL}u;}
// The bed-slope lane, its directory after the mask's and its slots by page
// column of the box (uniformSolidScratchLayout): the gradient of the bed's height along x and
// z, in cells per cell and pointing uphill, for an open cell that sits on a
// solid voxel's top face. Zero for every other cell: a voxel tread is then
// the level floor it looks like.
fn umBedSlopeBase()->u32{return ${SOLID_MASK_TABLE_WORDS}u;}
// The tile cut map's first word: after the lane's directory.
fn umSolidCutMap()->u32{return ${uniformSolidScratchLayout().cutMapOffsetWords}u;}
fn umBedSlope(p:vec3i)->vec2f{
 if(!umSolidEnabled()||umBoundaryRecord(p)!=0u){return vec2f(0);}
 let q=p-umSolidBoxCell();
 if(any(q<vec3i(0))||any(vec3u(q)>=umSolidDimensions())){return vec2f(0);}
 let top=umSolidScratch[umBedSlopeBase()+u32(q.x>>7u)+64u*u32(q.z>>7u)];
 if(top==0u){return vec2f(0);}
 let slot=umSolidScratch[top+u32((q.x>>5u)&3)+4u*u32((q.z>>5u)&3)];
 if(slot==0u){return vec2f(0);}
 let at=slot+${UNIFORM_BED_SLOPE_LAYERS}u*u32((q.x&31)+32*(q.z&31));var entry=0u;
 for(var k=0u;k<${UNIFORM_BED_SLOPE_LAYERS}u;k++){let w=umSolidScratch[at+k];if(w>>16u==u32(q.y)+1u){entry=w;break;}}
 return vec2f(vec2i(bitcast<i32>(entry<<24u)>>24u,bitcast<i32>(entry<<16u)>>24u))/${UNIFORM_SOLID_BED_SLOPE_UNITS}.0;
}
fn umSolidHasTerrain()->bool{return umSolidParams.frame.w>0.5;}
${uniformTerrainSamplingWGSL}
${sculptCollisionWGSL("umSolidScratch")}
// Positive in fluid space, gradient in lattice coordinates, value in metres.
fn umTerrainBoundary(q:vec3f)->vec4f{
 let h=umSolidParams.cellGravity.xyz;let surface=umTerrainSurface(q);
 let terrain=(q.y-surface.x)*h.y;let world=umSolidParams.frame.xyz+q*h;
 let cut=collisionSculptSample(world);
 if(cut.distance<1e19&&cut.distance<sculptFieldBand()-1e-7&&-cut.distance>terrain){return vec4f(-cut.normal*h,-cut.distance);}
 return vec4f(vec3f(-surface.y,1,-surface.z)*h.y,terrain);
}
${geometricPlaneBoxWGSL.replaceAll("geometric", "umTerrain")}
${coarse!==undefined&&gated?uniformSolidBoundaryWGSL.replace("fn umBoundaryRecord(p:vec3i)->u32{","fn umBoundaryRecordGeometry(p:vec3i)->u32{")+/* wgsl */`
// The record is static boundary geometry, independent of rigid-body pose.
// Every page claim and static edit builds it before a consuming stage.
fn umBoundaryRecord(p:vec3i)->u32{
 if(!umSolidValid(p)||umBoundaryBase()==0u){return 0u;}
 let tile=umTileKey(p>>vec3u(2u));
 if(!umKeyStored(tile)){return umBoundaryRecordGeometry(p);}
 let row=umSolidBoundaryRow(tile);
 let a=vec4u(umSolidRecord[row]);let b=vec3u(umSolidRecord[row+1u].xyz);
 let local=vec3u(p)&vec3u(3u);let cell=local.x+4u*(local.y+4u*local.z);
 let mask=select(a.z|(a.w<<16u),b.x|(b.y<<16u),cell>=32u);
 if((mask&(1u<<(cell&31u)))==0u){return 0u;}
 return a.x|(a.y<<16u);
}
// The cells of stored tile t the procedural field cuts (the record build's
// fieldCut): bit x+4y+16z. Its other cells are whole or the open bits' closed.
fn umSolidFieldCut(t:u32)->vec2u{
 if(umBoundaryBase()==0u){return vec2u(0u);}
 let r=vec4u(umSolidRecord[umSolidFieldCutRow(t)]);
 return vec2u(r.x|(r.y<<16u),r.z|(r.w<<16u));
}
// Cell p's field-cut bit; a cell of a tile no page stores may be cut.
fn umSolidFieldCutAt(p:vec3i)->bool{
 let t=umTileKey(p>>vec3u(2u));if(!umKeyStored(t)){return true;}
 let m=umSolidFieldCut(t);let l=vec3u(p)&vec3u(3u);let cell=l.x+4u*(l.y+4u*l.z);
 return (select(m.x,m.y,cell>=32u)&(1u<<(cell&31u)))!=0u;
}
`:uniformSolidBoundaryWGSL+/* wgsl */`
// No record: every cell of a scene with a boundary may be cut.
fn umSolidFieldCut(t:u32)->vec2u{return select(vec2u(0xffffffffu),vec2u(0u),umBoundaryBase()==0u);}
fn umSolidFieldCutAt(p:vec3i)->bool{return true;}
`}
// Integrate four local planes. Exact on a planar heightfield; subdivision
// resolves the column-centre patch boundaries instead of quantizing a face
// aperture to quarters. Zero normal width evaluates a face's area fraction.
fn umTerrainBoxOpen(q:vec3f,width:vec3f)->f32{
 if(!umSolidHasTerrain()){return 1.0;}
 var open=0.0;let part=width*vec3f(0.5,1,0.5);
 for(var k=0u;k<4u;k++){
  let at=q+vec3f(select(-0.25,0.25,(k&1u)!=0u)*width.x,0,select(-0.25,0.25,(k&2u)!=0u)*width.z);
  let s=umTerrainBoundary(at);
  open+=0.25*umTerrainPlaneBoxFraction(s.xyz,s.w,part);
 }
 return open;
}
fn umCellInsideTerrain(p:vec3i)->bool{if(!umSolidHasTerrain()){return false;}return umTerrainBoundary(vec3f(p)+0.5).w<0.0;}
// The unit box at q: x umTerrainBoxOpen, and y umBoundaryBoxOpen where
// boundary (else 0), through one umTerrainPlaneBoxFraction site (Metal
// inlines every site, and umCellOpenGeometry is inlined at every umCellOpen).
fn umUnitBoxOpen(q:vec3f,boundary:bool)->vec2f{
 let width=vec3f(1);let terrain=umSolidHasTerrain();
 var open=vec2f(select(1.0,0.0,terrain),0.0);
 for(var j=select(4u,0u,terrain);j<select(4u,12u,boundary);j++){
  var normal=vec3f(0);var offset=0.0;var part=width*0.5;var weight=0.125;
  if(j<4u){
   let at=q+vec3f(select(-0.25,0.25,(j&1u)!=0u)*width.x,0,select(-0.25,0.25,(j&2u)!=0u)*width.z);
   let s=umTerrainBoundary(at);normal=s.xyz;offset=s.w;part=width*vec3f(0.5,1,0.5);weight=0.25;
  }else{
   let k=j-4u;let bit=vec3u(k&1u,(k>>1u)&1u,k>>2u);let at=q+select(vec3f(-0.25),vec3f(0.25),bit!=vec3u(0))*width;
   let s=umBoundarySample(at);normal=s.xyz;offset=s.w;
  }
  let fraction=weight*umTerrainPlaneBoxFraction(normal,offset,part);
  if(j<4u){open.x+=fraction;}else{open.y+=fraction;}
 }
 return open;
}
fn umSolidWorldCell(id:vec3i)->vec3f{let h=umSolidParams.cellGravity.xyz;
 return vec3f(umSolidParams.frame.x+(f32(id.x)+0.5)*h.x,umSolidParams.frame.y+(f32(id.y)+0.5)*h.y,umSolidParams.frame.z+(f32(id.z)+0.5)*h.z);}
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
 var remaining=umBodyTileMask(t);
 while(remaining!=0u){let i=firstTrailingBit(remaining);remaining&=remaining-1u;if(umBodyNear(i,world,3.0*max(h.x,max(h.y,h.z)))){return false;}}
 return true;`:"return false;"}
}
// The +axis face of id (neighbor across it) meets no solid: a wall page is
// solid in the directory and cuts no tile, so a face on a page's high side
// asks the page across it.
fn umSolidFaceClear(id:vec3i,neighbor:vec3i,axis:u32)->bool{return umSolidTileClear(id)&&((id[axis]&31)!=31||umSolidValid(neighbor));}
// Rigid bodies: the native RigidBody helpers over the shape table. Every
// obstacle query first rejects on its bounding sphere. A closed fluid
// interior has an unbounded solid exterior and cannot take that rejection.
${sceneShapeWgsl()}
${fluidRigidBoundaryWGSL}
fn umBodyCount()->u32{if(!umSolidEnabled()){return 0u;}return min(u32(max(round(umSolidParams.boundary.z),0.0)),${GPU_RIGID_BODY_CAPACITY}u);}
fn umBodyRegionOfTile(t:u32)->u32{return 8u*(t>>9u)+((t>>2u)&1u)+2u*((t>>5u)&1u)+4u*((t>>8u)&1u);}
// Any present body's bounding sphere within margin of a world point: every
// body sample a helper takes lies within its own margin of that point.
${writable?/* wgsl */`// Only the record builder restricts the roster. Other writable entry
// points start with every bit set and retain the unrestricted behavior.
var<private> umSolidQueryBodies:u32=${(2 ** GPU_RIGID_BODY_CAPACITY - 1)}u;
fn umBodyTileCandidates(tc:vec3u)->u32{
 let h=umSolidParams.cellGravity.xyz;let centre=umSolidWorldCell(vec3i(tc*4u))+1.5*h;var mask=0u;
 for(var i=0u;i<umBodyCount();i++){if(umBodyNear(i,centre,length(3.0*h))){mask|=1u<<i;}}
 return mask;
}`:""}
fn umBodyCandidates(world:vec3f,margin:f32)->u32{
 var mask=0u;${bodyLoop}if(umBodyNear(i,world,margin)){mask|=1u<<i;}}
 return mask;
}
fn umAnyBodyNear(world:vec3f,margin:f32)->bool{
 ${bodyLoop}if(umBodyNear(i,world,margin)){return true;}}
 return false;
}
fn umQuatRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);let uuv=cross(q.yzw,uv);return v+2.0*(q.x*uv+uuv);}
fn umBodyLocal(i:u32,world:vec3f)->vec3f{let b=umSolidBodies[i];return umQuatRotate(vec4f(b.orientation.x,-b.orientation.yzw),world-b.positionShape.xyz);}
fn umBodyNear(i:u32,world:vec3f,margin:f32)->bool{let b=umSolidBodies[i];return b.positionShape.w<0.0||distance(world,b.positionShape.xyz)<=b.dimensions.w+margin;}
fn umInsideBody(i:u32,world:vec3f)->bool{let b=umSolidBodies[i];return umBodyNear(i,world,0.0)&&fluidRigidInside(i32(round(b.positionShape.w)),b.dimensions.xyz,umBodyLocal(i,world));}
fn umBodyDistance(i:u32,world:vec3f)->f32{let b=umSolidBodies[i];return fluidRigidDistance(i32(round(b.positionShape.w)),b.dimensions.xyz,umBodyLocal(i,world));}
fn umBodyVelocity(i:u32,world:vec3f)->vec3f{let b=umSolidBodies[i];return b.linearVelocity.xyz+cross(b.angularVelocity.xyz,world-b.positionShape.xyz);}
// rigidBodyIndexAt: the first body containing the point, -1 for none.
fn umBodyAt(world:vec3f)->i32{
 ${bodyLoop}if(umInsideBody(i,world)){return i32(i);}}
 return -1;
}
// A query's conservative body set is reused by all of its sample points.
// Visit bits in roster order: overlapping solids keep the same first owner.
fn umBodyAtCandidates(world:vec3f,candidates:u32)->i32{
 var remaining=candidates;
 while(remaining!=0u){let i=firstTrailingBit(remaining);remaining&=remaining-1u;
  if(umInsideBody(i,world)){return i32(i);}
 }
 return -1;
}
// umBodyAt of a point of h cell p: its tile's candidates (umBodyTileMask).
fn umBodyAtCell(p:vec3i,world:vec3f)->i32{${writable?"return umBodyAt(world);":"return umBodyAtCandidates(world,umBodyCellMask(p));"}}
${uniformRigidCoverageSource(writable?"umSolidQueryBodies":undefined)}
// cellOpenFraction.
fn umCellOpenGeometry(p:vec3i)->f32{
 if(!umSolidEnabled()){return select(0.0,1.0,umSolidValid(p));}
 if(umSolidTileClear(p)){return 1.0;}
 let boundary=umBoundaryRecord(p);
 if(boundary==0u&&umSolidVoxel(p)){return 0.0;}
 // One body and one terrain site for both forms: Metal inlines each site,
 // and this function is inlined at every umCellOpen site of a kernel.
 let body=umCellBodyFraction(p);let box=umUnitBoxOpen(vec3f(p)+0.5,boundary!=0u);let terrain=1.0-box.x;
 if(boundary!=0u){return min(box.y,min(1.0-body,1.0-terrain));}
 return clamp((1.0-body)*(1.0-terrain),0.0,1.0);
}
fn umCellOpen(p:vec3i)->f32{
 ${coarse!==undefined&&gated?`// Bodies may move after the record rebuild (the tail's integration).
 // At the built pose, the exact record also serves cells beside bodies.
 // After publication, entered and vacated cells need geometry until rebuilt.
 if(umSolidEnabled()){
  let tile=umTileKey(p>>vec3u(2u));
  if(umKeyStored(tile)){
   let h=umSolidParams.cellGravity.xyz;
   if(umSolidBodyPoseCurrent()||umBodyCount()==0u||(!umSolidBodyBuilt(tile)&&!umAnyBodyNear(umSolidWorldCell(p),length(0.5*h)))){
    let open=umSolidRecordedOpen(tile,p);if(open>=0.0){return open;}
   }
  }
 }`:""}
 return umCellOpenGeometry(p);
}
// The exact >1e-5 predicate, cached by the solid record builder. Consumers
// of the coarse record run after its rebuild at the frame head (also after
// voxel/terrain edits and body motion). The builder, stages without a
// record and a cell of a tile with no current row (no page stores it) keep
// the full geometric evaluation;
// fractional queries use the same record's exact h-cell fractions. A cell of a wall page is closed in
// either form.
fn umCellIsOpen(p:vec3i)->bool{
 ${coarse!==undefined&&gated?`if(!umSolidEnabled()){return umSolidValid(p);}
 // An unstored tile's cell of a wall page: umCellOpenGeometry's voxel 0.
 let t=umTileKey(p>>vec3u(2u));if(!umKeyStored(t)){if(!umSolidValid(p)){return false;}return umCellOpen(p)>1e-5;}
 return umStoredCellIsOpen(p,t);`:"return umCellOpen(p)>1e-5;"}
}
// umCellIsOpen(p) where t, the key of p's tile, is stored: the record's bit,
// without the unstored tile's geometry. A caller that knows the tile is
// stored then holds no umCellOpen site (Metal inlines one per site).
fn umStoredCellIsOpen(p:vec3i,t:u32)->bool{
 ${coarse!==undefined&&gated?`if(!umSolidEnabled()){return umSolidValid(p);}
 let l=vec3u(p)&vec3u(3u);let lane=l.x+4u*(l.y+4u*l.z);
 let mask=umSolidOpenCells(t);
 return (select(mask.x,mask.y,lane>=32u)&(1u<<(lane&31u)))!=0u;`:"return umCellIsOpen(p);"}
}
// cellInsideSolid: voxel, terrain or a body at the centre -- the CM11a p_min=0 rows.
fn umCellInsideSolid(p:vec3i)->bool{
 if(!umSolidEnabled()){return false;}
 if(umSolidTileClear(p)){return false;}
 if(umBoundaryRecord(p)!=0u){return umBoundarySample(vec3f(p)+0.5).w<0.0||umCellInsideTerrain(p)||umBodyAtCell(p,umSolidWorldCell(p))>=0;}
 if(umSolidVoxel(p)){return true;}
 return umCellInsideTerrain(p)||umBodyAtCell(p,umSolidWorldCell(p))>=0;
}
fn umSolidFaceWorld(id:vec3i,axis:u32)->vec3f{var world=umSolidWorldCell(id);world[axis]+=0.5*umSolidParams.cellGravity.xyz[axis];return world;}
fn umSolidTraceWorld(p:vec3f)->vec3f{let h=umSolidParams.cellGravity.xyz;
 return vec3f(umSolidParams.frame.x+p.x*h.x,umSolidParams.frame.y+p.y*h.y,umSolidParams.frame.z+p.z*h.z);}
fn umSolidVoxelAtWorld(world:vec3f)->bool{
 let h=umSolidParams.cellGravity.xyz;
 let p=vec3i(floor(vec3f((world.x-umSolidParams.frame.x)/h.x,(world.y-umSolidParams.frame.y)/h.y,(world.z-umSolidParams.frame.z)/h.z)));
 if(umBoundaryRecord(p)!=0u){let q=vec3f((world.x-umSolidParams.frame.x)/h.x,(world.y-umSolidParams.frame.y)/h.y,(world.z-umSolidParams.frame.z)/h.z);return umBoundarySample(q).w<0.0;}
 return umSolidVoxel(p);
}
fn umSolidTerrainAtWorld(world:vec3f)->bool{
 if(!umSolidHasTerrain()){return false;}
 let h=umSolidParams.cellGravity.xyz;let d=vec3i(umSolidDimensions());
 let x=clamp(i32(floor((world.x-umSolidParams.frame.x)/h.x)),0,d.x-1);
 let z=clamp(i32(floor((world.z-umSolidParams.frame.z)/h.z)),0,d.z-1);
 let q=vec3f((world.x-umSolidParams.frame.x)/h.x,(world.y-umSolidParams.frame.y)/h.y,(world.z-umSolidParams.frame.z)/h.z);
 return umTerrainBoundary(q).w<0.0;
}
fn umSolidAtWorld(world:vec3f)->bool{return umSolidVoxelAtWorld(world)||umSolidTerrainAtWorld(world)||umBodyAt(world)>=0;}
// insideAnyTraceSolid for a lattice-coordinate point: voxels and bodies
// (terrain is not a trace solid natively).
fn umSolidAtWorldCell(p:vec3f)->bool{
 if(!umSolidEnabled()){return false;}
 let cell=vec3i(floor(p));if(umSolidTileClear(cell)){return false;}
 let world=umSolidTraceWorld(p);return umSolidVoxelAtWorld(world)||umBodyAtCell(cell,world)>=0;}
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
 if((umSolidVoxel(id)&&umBoundaryRecord(id)==0u)||(umSolidVoxel(neighbor)&&umBoundaryRecord(neighbor)==0u)){return 0.0;}
 let world=umSolidFaceWorld(id,axis);let h=umSolidParams.cellGravity.xyz;let reach=max(h.x,max(h.y,h.z));
 // Keep the existing broad-phase reach, retaining its candidate identities.
 // Every dual-cell sample is within 0.4*sqrt(3)*reach, and the nearest-wall
 // fallback reaches one cell: both are contained in this same 2*reach set.
 var candidates=0u;var near=${writable?"umSolidQueryBodies&((1u<<umBodyCount())-1u)":"umBodyCellMask(id)"};
 while(near!=0u){let i=firstTrailingBit(near);near&=near-1u;if(umBodyNear(i,world,2.0*reach)){candidates|=1u<<i;}}
 if(candidates==0u){return 0.0;}
 var solid=0.0;var velocity=0.0;
 for(var sampleIndex=0u;sampleIndex<8u;sampleIndex+=1u){
  let sampleWorld=world+vec3f(select(-0.4,0.4,(sampleIndex&1u)!=0u)*h.x,select(-0.4,0.4,(sampleIndex&2u)!=0u)*h.y,select(-0.4,0.4,(sampleIndex&4u)!=0u)*h.z);
  if(umSolidVoxelAtWorld(sampleWorld)||umSolidTerrainAtWorld(sampleWorld)){solid+=1.0;continue;}
  let body=umBodyAtCandidates(sampleWorld,candidates);
  if(body>=0){solid+=1.0;velocity+=umBodyVelocity(u32(body),world)[axis];}
 }
 if(solid>0.0){return velocity/solid;}
 var nearest=reach;var result=0.0;
 var remaining=candidates;
 while(remaining!=0u){let i=firstTrailingBit(remaining);remaining&=remaining-1u;
  let d=abs(umBodyDistance(i,world));
  if(d<=nearest){nearest=d;result=umBodyVelocity(i,world)[axis];}
 }
 return result;
}
// faceOpenFraction: the Sec. 3.6 transverse face aperture, at a face a body
// reaches (its samples lie within 0.35 sqrt(2) h of the face centre) that is
// neither clear nor beside a voxel: umPressureFaceV's rigid contact, its one
// caller, has taken those forms.
fn umFaceOpen(id:vec3i,axis:u32,candidates:u32)->f32{
 if(!umSolidEnabled()){return 1.0;}
 let world=umSolidFaceWorld(id,axis);let h=umSolidParams.cellGravity.xyz;
 let tangentA=(axis+1u)%3u;let tangentB=(axis+2u)%3u;var solid=0.0;
 for(var sampleIndex=0u;sampleIndex<4u;sampleIndex+=1u){
  var sampleWorld=world;
  sampleWorld[tangentA]+=select(-0.35,0.35,(sampleIndex&1u)!=0u)*h[tangentA];
  sampleWorld[tangentB]+=select(-0.35,0.35,(sampleIndex&2u)!=0u)*h[tangentB];
  solid+=select(0.0,1.0,umSolidVoxelAtWorld(sampleWorld)||umSolidTerrainAtWorld(sampleWorld)||umBodyAtCandidates(sampleWorld,candidates)>=0);
 }
 return 1.0-solid*0.25;
}
// Pressure flux weight: face aperture at rigid contact, CM11a dual-cell V
// for static voxel/terrain boundaries.
// A wall page is voxels: its face holds half the open fraction of the cell
// this side of it, and nothing between two wall cells. A depth symmetry
// plane is such a wall, as for solid-free mixed frames and every coarse
// owner (the native CM11a record used V=0). An AIR page is open space.
fn umPressureFaceV(id:vec3i,axis:u32)->f32{
 if(!umSolidEnabled()){return 1.0;}
 var q=vec3f(id)+0.5;q[axis]+=0.5;
 // Body-local: a face no body reaches (within length(0.5h) of its centre,
 // past every 0.4h dual-cell sample) takes the body-free evaluation, so a
 // distant body changes no aperture.
 let world=umSolidFaceWorld(id,axis);let h=umSolidParams.cellGravity.xyz;
 let candidates=umBodyCandidates(world,length(0.5*h));let bodyNear=candidates!=0u;
 // One terrain, boundary and umCellOpen site each: Metal inlines every site,
 // and this function is inlined at every face a kernel weighs.
 let boxed=!bodyNear&&umBoundaryTouchesBox(q,vec3f(1));
 if(!boxed){
  var neighbor=id;neighbor[axis]+=1;
  if(umSolidFaceClear(id,neighbor,axis)){return 1.0;}
  if((umSolidVoxel(id)&&umBoundaryRecord(id)==0u)||(umSolidVoxel(neighbor)&&umBoundaryRecord(neighbor)==0u)){
   var open=0.0;
   for(var j=0u;j<2u;j++){let cell=umCellOpen(select(id,neighbor,j==1u));if(j==0u){open=cell;}else{return 0.5*(open+cell);}}
  }
  // Rigid contact uses the face aperture. A dual cell can contain liquid on
  // one side of a completely blocked face; treating that volume as an open
  // face couples the fluid to ghost liquid inside the body at the waterline.
  // Gravity then creates velocity through the body even while it is fixed.
  if(bodyNear){return umFaceOpen(id,axis,candidates);}
 }
 let box=umUnitBoxOpen(q,boxed);
 if(boxed){return min(box.y,box.x);}
 return box.x;
}
// The cell whose tile's aperture rows hold the face of cell id normal to
// axis on its high side: id, or where no page stores id's tile (a domain or
// lattice wall below the face) the cell across the face, whose low rows do.
fn umSolidApertureCell(id:vec3i,axis:u32)->vec3i{
 ${coarse!==undefined&&gated?`if(umKeyStored(umTileKey(id>>vec3u(2u)))){return id;}
 var high=id;high[axis]+=1;return high;`:"return id;"}
}
// Whether the record's aperture rows of cell p's tile are good (call it with
// umSolidApertureCell's cell; umSolidApertureRecord reads them, a lean read
// with no geometry inlined): 0 they are; 1 the geometry must serve their
// faces (no page stores the tile, no record is bound, or a body reached the
// tile at its build or reaches its region after pose publication); 2 the rows are stale (a claim
// or an edit the record missed): the caller latches its stage's fatal.
fn umSolidAperturesServe(p:vec3i)->u32{
 ${coarse!==undefined&&gated?`if(!umSolidEnabled()){return 0u;}
 let t=umTileKey(p>>vec3u(2u));if(!umKeyStored(t)){return 1u;}
 let stamp=umSolidRecord[umSolidApertureStampRow(t)];
 if(any(vec3i(stamp.xyz)!=umSolidApertureCoord(t))||stamp.w!=umSolidRecord[umSolidApertureEpochRow()].x){return 2u;}
 if(umSolidTileSlot(t)==UM_SOLID_GEOMETRY){return 1u;}
 if(!umSolidBodyPoseCurrent()&&umBodyCount()>0u&&(umSolidBodyBuilt(t)||umBodyPoseNearTile(t,2u))){return 1u;}
 return 0u;`:"return select(0u,1u,umSolidEnabled());"}
}
// The aperture rows of the face of cell id normal to axis on its high side
// (umSolidApertureCell's tile's slot; its low rows flagged in the top bit,
// UM_SOLID_WHOLE where the tile holds no slot), for umSolidApertureAt: one
// per tile and axis side, so a kernel that reads a tile's faces looks its
// four up once.
fn umSolidApertureRows(id:vec3i,axis:u32)->u32{
 ${coarse!==undefined&&gated?`let rowCell=umSolidApertureCell(id,axis);
 let slot=umSolidTileSlot(umTileKey(rowCell>>vec3u(2u)));if(slot>=UM_SOLID_GEOMETRY){return UM_SOLID_WHOLE;}
 return umSolidApertureSlotRow(slot)|select(0u,0x80000000u,any(rowCell!=id));`:"return 0u;"}
}
// The aperture of that face from its rows (umSolidApertureRows of any cell
// whose face reads the same tile's same rows).
fn umSolidApertureAt(rows:u32,id:vec3i,axis:u32)->f32{
 ${coarse!==undefined&&gated?`let l=vec3u(id)&vec3u(3u);
 if(rows==UM_SOLID_WHOLE){return 1.0;}
 if(rows<0x80000000u){let cell=l.x+4u*(l.y+4u*l.z);return umSolidRecord[rows+16u*axis+(cell>>2u)][cell&3u];}
 let slot=l[(axis+1u)%3u]+4u*l[(axis+2u)%3u];return umSolidRecord[(rows&0x7fffffffu)+49u+4u*axis+(slot>>2u)][slot&3u];`:"return 1.0;"}
}
// umPressureFaceV of the face of cell id normal to axis on its high side,
// from the aperture rows: only where umSolidAperturesServe of
// umSolidApertureCell(id,axis) is 0.
fn umSolidApertureRecord(id:vec3i,axis:u32)->f32{
 ${coarse!==undefined&&gated?`return umSolidApertureAt(umSolidApertureRows(id,axis),id,axis);`:"return -1.0;"}
}
// umPressureFaceV from the record's aperture rows where they hold it: a
// stored tile no body reached at its build and none reaches now (umCellOpen's
// rule). A stored tile whose stamp is not its world coordinate at the current epoch
// (a claim or an edit the record missed) answers -1: the caller latches its
// stage's fatal. Elsewhere, and without the record, the geometry. Call it at
// one site a kernel: it inlines umPressureFaceV.
fn umPressureFaceVRecorded(id:vec3i,axis:u32)->f32{
 ${coarse!==undefined&&gated?`if(umSolidEnabled()){
  let rowCell=umSolidApertureCell(id,axis);let t=umTileKey(rowCell>>vec3u(2u));
  if(umKeyStored(t)){
   let stamp=umSolidRecord[umSolidApertureStampRow(t)];
   if(any(vec3i(stamp.xyz)!=umSolidApertureCoord(t))||stamp.w!=umSolidRecord[umSolidApertureEpochRow()].x){return -1.0;}
   let slot=umSolidTileSlot(t);
   if(slot!=UM_SOLID_GEOMETRY&&(umSolidBodyPoseCurrent()||umBodyCount()==0u||(!umSolidBodyBuilt(t)&&!umAnyBodyNear(umSolidFaceWorld(id,axis),length(0.5*umSolidParams.cellGravity.xyz))))){
    if(slot==UM_SOLID_WHOLE){return 1.0;}
    return umSolidApertureAt(umSolidApertureSlotRow(slot)|select(0u,0x80000000u,any(rowCell!=id)),id,axis);
   }
  }
 }`:""}
 return umPressureFaceV(id,axis);
}
`;
}
