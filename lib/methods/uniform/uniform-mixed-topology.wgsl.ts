import { uniformCompiledTopologyWGSL } from "./uniform-compiled-topology";
import { uniformMidpointTraceWGSL } from "./uniform-midpoint-trace.wgsl";
import { uniformMixedVelocitySamplingWGSL } from "./uniform-mixed-velocity-sampling.wgsl";
import { uniformMixedFacesWGSL } from "./uniform-mixed-faces.wgsl";
import type { UniformMixedLayout } from "./uniform-mixed-layout";

/** Regular tiers with at most this many tiles join the fused interface launch. */
export const UNIFORM_MIXED_FUSED_REGULAR_TILES=64;
/** A regular 4h tier with at most this many owners (256 groups) is
 * launch-bound: they ride another launch as packed lanes (umRegularCoarseOwner)
 * instead of a width-specialized launch of their own. */
export const UNIFORM_MIXED_PACKED_REGULAR_OWNERS=16384;
/** umCountedJobs: GPU-counted launches (uniformMixedCertifiedEntriesWGSL
 * entries striding a fixed grid, ownership.dispatch*Counted). Each count
 * mirrors the entry's owner lookup, so no host membership sizes the launch.
 * owners: umOwner slots, 64 per job (tier, planned, interface or regular
 * list); tiles: umTileJobOwner, one tile per job; all: umAllOwner slots, 64
 * per job; regularCoarse: umRegularCoarseOwner slots, 64 per job; fused:
 * umFusedOwner jobs (seams, then with umFusedJobs the small regular tiers);
 * fusedQuad: fused with seam 4h tiles packed four per job
 * (uniformMixedFaceTileDispatchWGSL); hanging: one job per hanging slot;
 * fineTiles: one job per h list tile; coarseTiles: 64 4h list tiles per job;
 * residentAll: all's h jobs, then one job per resident page (umResidentAllOwner,
 * its 4h tiles one lane each); residentPages: one job per resident page, a
 * lane per tile (umResidentPageTile). */
export const UNIFORM_MIXED_COUNTED={owners:1,tiles:2,all:3,regularCoarse:4,fused:5,fusedQuad:6,hanging:7,fineTiles:8,coarseTiles:9,residentAll:10,residentPages:11} as const;
/** umFusedRegularGate for owners/tiles/regularCoarse launches: skipFused
 * empties a tier that rides the fused launch (dispatchRegular skipFused);
 * onlyFused empties one that does not. */
export const UNIFORM_MIXED_FUSED_GATE={skipFused:1,onlyFused:2} as const;
/** Relayout fatal bit (UNIFORM_MIXED_RELAYOUT_FATAL.hangingCapacity): a
 * generation needed more hanging slots than the preallocated cache. */
export const UNIFORM_MIXED_OVERFLOW_HANGING=1;
/** The layout builder's sticky fatal bit for a generation with more h tiles
 * than the owner-indexed buffers hold (UniformMixedCapacity.fineTiles). */
export const UNIFORM_MIXED_OVERFLOW_FINE=16;
/** Residency certificate (docs: fig7-256 3x plan, S2): 16³-cell pages of 4³
 * tiles on dense storage. Support words from uniformMixedResidencyWord: the
 * resident page count, sticky closure-violation bits (in-frame readers or
 * them when they would read outside the closure; the next census turns them
 * into a builder fatal), the census's closure radius in tiles (diagnostic),
 * one reserved word, a flag per page (non-zero: resident), then the resident
 * pages in ascending order. update() makes every page resident; the census
 * (UniformMixedDynamicClassifier) rewrites them each frame. */
export const uniformMixedResidencyWord=(tiles:number)=>9*tiles+28;
export const UNIFORM_MIXED_PAGE_TILES=4;
export function uniformMixedPageDimensions(lattice:{readonly dimensions:readonly number[]}):[number,number,number]{
  const p=lattice.dimensions.map(d=>Math.ceil(d/4/UNIFORM_MIXED_PAGE_TILES));return [p[0]!,p[1]!,p[2]!];
}
export const uniformMixedPageCount=(lattice:{readonly dimensions:readonly number[]})=>uniformMixedPageDimensions(lattice).reduce((n,d)=>n*d,1);
/** Sticky detail-storage violation bits (uniform-detail-fields.ts): set by
 * a store a non-resident tile cannot hold; the frame receipt makes it fatal. */
export const uniformMixedDetailViolationWord=(tiles:number,lattice:{readonly dimensions:readonly number[]})=>uniformMixedResidencyWord(tiles)+4+2*uniformMixedPageCount(lattice);
/** Support words an ownership of `tiles` tiles on `lattice` allocates. */
export const uniformMixedSupportWords=(tiles:number,lattice:{readonly dimensions:readonly number[]})=>uniformMixedDetailViolationWord(tiles,lattice)+4;

/** One packed topology buffer: tile records, h/4h worklists, then frozen stencil masks.
 * Ownership and tracing share this ABI. Two tiers: 0 = h (width 1, 64 owners
 * per tile), 1 = 4h (width 4, one owner per tile). umCounts = (h tiles, 4h
 * tiles, 0, 8). A 64-lane group visits one h tile or 64 coarse tiles; coarse dispatch never
 * pays 63 idle lanes.
 */
/** The lattice and tile count a topology shader is specialized for: an
 * ownership's fixed capacity (or a layout, whose tiles are the same lattice). */
export type UniformMixedTopologyShape = { readonly lattice: UniformMixedLayout["lattice"]; readonly tiles: number | { readonly length: number } };
export function uniformMixedTopologyWGSL(shape: UniformMixedTopologyShape, group: number, prefix = ""): string {
  const layout = { lattice: shape.lattice, tiles: { length: typeof shape.tiles === "number" ? shape.tiles : shape.tiles.length } };
  if (prefix && !/^[a-zA-Z][a-zA-Z0-9]*$/.test(prefix)) throw new Error("Invalid mixed topology namespace");
  const source = /* wgsl */ `
@group(${group}) @binding(0) var<storage,read> umTopology:array<u32>;
@group(${group}) @binding(1) var<uniform> umCounts:vec4u;
@group(${group}) @binding(2) var<storage,read_write> umSupport:array<u32>;
override umCellWidth:u32=1u;
override umDispatchX:u32=65535u;
override umRegularFine:bool=false;
override umPlannedFine:u32=0u;
override umInterfaceTiles:bool=false;
override umRegularTiles:bool=false;
override umMergedTiles:bool=false;
// With umMergedTiles, take the fused jobs (every seam tile, then small regular tiers) instead.
override umFusedJobs:bool=false;
// With umMergedTiles, leave out the general h list: its tiles run a launch of
// their own, and umMergedPack seam 4h tiles share each tile job.
override umMergedCoarse:bool=false;
override umMergedPack:u32=1u;
// Certified launches (uniformMixedCertifiedEntriesWGSL): 0 none, 1 the
// umPlannedFine certificate list, 2 the merged tile jobs, 3 the merged jobs
// with seam 4h tiles packed four per job (uniformMixedFaceTileDispatchWGSL).
override umCertifiedJobs:u32=0u;
// GPU-counted launches (UNIFORM_MIXED_COUNTED); 0 leaves umCertifiedJobs in charge.
override umCountedJobs:u32=0u;
override umFusedRegularGate:u32=0u;
const UM_D=vec3u(${layout.lattice.dimensions.map(n => `${n}u`).join(',')});const UM_T=UM_D/4u;
const UM_TILES:u32=${layout.tiles.length}u;
const UM_OVERFLOW_HANGING:u32=${UNIFORM_MIXED_OVERFLOW_HANGING}u;
// Residency (uniformMixedResidencyWord): pages of 4³ tiles, flags, list.
const UM_PD:vec3u=(UM_T+vec3u(3u))/4u;const UM_PAGES:u32=UM_PD.x*UM_PD.y*UM_PD.z;
const UM_RESIDENCY:u32=9u*UM_TILES+28u;
// Detail storage (uniform-detail-fields.ts): its table follows the 4 words
// per tile in umTopology; its violation bits follow the residency region.
const UM_DETAIL:u32=4u*UM_TILES;
const UM_DETAIL_VIOLATION:u32=UM_RESIDENCY+4u+2u*UM_PAGES;
fn umResidentPageCount()->u32{return umSupport[UM_RESIDENCY];}
fn umPageOfTile(t:u32)->u32{let c=umTileCoord(t)/4u;return c.x+UM_PD.x*(c.y+UM_PD.y*c.z);}
fn umPageCoord(page:u32)->vec3u{return vec3u(page%UM_PD.x,(page/UM_PD.x)%UM_PD.y,page/(UM_PD.x*UM_PD.y));}
fn umPageResident(page:u32)->bool{return umSupport[UM_RESIDENCY+4u+page]!=0u;}
fn umTileResident(t:u32)->bool{return umPageResident(umPageOfTile(t));}
fn umResidentPage(job:u32)->u32{return umSupport[UM_RESIDENCY+4u+UM_PAGES+job];}
// Tile lane of page, or UM_TILES past the lattice (a partial edge page).
fn umPageTile(page:u32,lane:u32)->u32{let c=umPageCoord(page)*4u+umCorner(lane,4u);if(any(c>=UM_T)){return UM_TILES;}return umTileAt(c);}
// Tile lane of resident page job (residentPages), or UM_TILES.
fn umResidentPageTile(job:u32,lane:u32)->u32{if(job>=umResidentPageCount()){return UM_TILES;}return umPageTile(umResidentPage(job),lane);}
fn umTileWidth(t:u32)->u32{return select(4u,1u,(umTopology[t]&0x80000000u)!=0u);}
fn umTileSupport(t:u32)->u32{return umSupport[3u*UM_TILES+t];}
// The mixed pressure schedule's slot gate (support 9n+24, 0 open): its gate
// kernel closes a pressure level's launches inside a disabled slot.
fn umSlotClosed()->bool{return umSupport[9u*UM_TILES+24u]!=0u;}
fn umTileStencil(t:u32)->vec2u{return vec2u(umTopology[2u*UM_TILES+2u*t],umTopology[2u*UM_TILES+2u*t+1u]);}
fn umTileMaximumWidth(t:u32)->u32{return umTileStencil(t).x>>27u;}
fn umTileMinimumWidth(t:u32)->u32{return umTileStencil(t).y>>27u;}
// The detail ring (UNIFORM_MIXED_DETAIL_RING): an h tile within three tiles.
fn umTileMirrored(t:u32)->bool{return (umTopology[2u*UM_TILES+2u*t+1u]&1u)!=0u;}
fn umTileAt(p:vec3u)->u32{return p.x+UM_T.x*(p.y+UM_T.y*p.z);}
fn umTileCoord(t:u32)->vec3u{return vec3u(t%UM_T.x,(t/UM_T.x)%UM_T.y,t/(UM_T.x*UM_T.y));}
fn umCorner(k:u32,side:u32)->vec3u{return vec3u(k%side,(k/side)%side,k/(side*side));}
${uniformCompiledTopologyWGSL}
struct UMOwner {tile:u32,lane:u32,width:u32,index:u32}
fn umAllOwner(gid:vec3u)->UMOwner {
 let slot=gid.x+umDispatchX*64u*gid.y;let fine=umCounts.x*64u;
 if(slot<fine){let tile=umTopology[UM_TILES+slot/64u];let lane=slot%64u;return UMOwner(tile,lane,1u,(umTopology[tile]&0x3fffffffu)+lane);}
 let local=slot-fine;if(local>=umCounts.y){return UMOwner();}
 let tile=umTopology[UM_TILES+umCounts.x+local];return UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu);
}
// umAllOwner over the residency certificate: every h owner (the h list's
// jobs), then one job per resident page, a lane per tile, holding its 4h
// owner. A 4h tile of an absent page is far air (V=0, corner phi at least
// 16h), audited by the next census.
fn umResidentAllOwner(gid:vec3u)->UMOwner {
 let slot=gid.x+umDispatchX*64u*gid.y;let fine=umCounts.x*64u;
 if(slot<fine){let tile=umTopology[UM_TILES+slot/64u];let lane=slot%64u;return UMOwner(tile,lane,1u,(umTopology[tile]&0x3fffffffu)+lane);}
 let tile=umResidentPageTile((slot-fine)/64u,slot%64u);
 if(tile>=UM_TILES){return UMOwner();}
 if(umTileWidth(tile)!=4u){return UMOwner();}
 return UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu);
}
// Regular 4h owner s: tiles whose 3x3x3 stencil has one width, listed by the
// ownership (support 8n+20: count, three zero words, then the list).
fn umRegularCoarseOwner(slot:u32)->UMOwner {
 let base=8u*UM_TILES+20u;if(slot>=umSupport[base]){return UMOwner();}
 let tile=umSupport[base+4u+slot];return UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu);
}
fn umOwner(gid:vec3u)->UMOwner {
 let slot=gid.x+umDispatchX*64u*gid.y;
 if(umMergedTiles){
  // 64 slots per merged tile job (see umTileJobOwner); then 64 regular
  // coarse owners per packed job.
  let tiles=umMergedTileJobs();
  if(!umFusedJobs&&slot/64u>=tiles){return umRegularCoarseOwner(slot-64u*tiles);}
  var owner=umMergedTileJob(slot/64u);let lane=slot%64u;
  if(owner.width==0u||lane>=64u/(owner.width*owner.width*owner.width)){return UMOwner();}
  owner.lane=lane;owner.index+=lane;return owner;
 }
 let cells=64u/(umCellWidth*umCellWidth*umCellWidth);
 let job=slot/cells;var offset=0u;var count=umCounts.x;
 if(umInterfaceTiles){
  let header=7u*UM_TILES+16u;var tier=0u;var start=0u;
  if(umCellWidth==4u){tier=1u;start=umSupport[header];}
  if(job>=umSupport[header+tier]){return UMOwner();}
  let tile=umSupport[header+4u+start+job];let lane=slot%cells;
  return UMOwner(tile,lane,umCellWidth,(umTopology[tile]&0x3fffffffu)+lane);
 }
 if(umCellWidth==4u){offset=umCounts.x;count=umCounts.y;}
 if(umCellWidth==1u&&umPlannedFine!=0u){
  count=umSupport[4u*UM_TILES+umPlannedFine];if(job>=count){return UMOwner();}
  let tile=umSupport[(4u+umPlannedFine)*UM_TILES+16u+job];let lane=slot%64u;
  return UMOwner(tile,lane,1u,(umTopology[tile]&0x3fffffffu)+lane);
 }
 if(job>=count){return UMOwner();}
 let tile=umTopology[UM_TILES+offset+job];let lane=slot%cells;
 if(umRegularTiles&&umTileMaximumWidth(tile)!=umTileMinimumWidth(tile)){return UMOwner();}
 return UMOwner(tile,lane,umCellWidth,(umTopology[tile]&0x3fffffffu)+lane);
}
// One workgroup per interface tile of every tier, then every tile of each
// small regular tier (UNIFORM_MIXED_FUSED_REGULAR_TILES). The owner carries
// its runtime width, so a single launch replaces the tiny per-tier launches.
fn umFusedRegularTier(tier:u32)->bool{
 let count=umCounts[tier];return count>umSupport[7u*UM_TILES+16u+tier]&&count<=${UNIFORM_MIXED_FUSED_REGULAR_TILES}u;
}
// Not a small fused tier: see UNIFORM_MIXED_PACKED_REGULAR_OWNERS.
fn umPackedRegularTier(tier:u32)->bool{
 let owners=(umCounts[tier]-umSupport[7u*UM_TILES+16u+tier])*(64u>>(6u*tier));
 return !umFusedRegularTier(tier)&&owners<=${UNIFORM_MIXED_PACKED_REGULAR_OWNERS}u;
}
fn umFusedOwner(group:vec3u,lane:u32,regular:bool)->UMOwner {
 let header=7u*UM_TILES+16u;var job=group.x+umDispatchX*group.y;
 let seams=umSupport[header]+umSupport[header+1u];
 var tile=0u;
 if(job<seams){tile=umSupport[header+4u+job];}
 else{
  if(!regular){return UMOwner();}
  job-=seams;var offset=0u;var found=false;
  for(var tier=0u;tier<2u;tier++){
   let count=umCounts[tier];
   if(!found&&umFusedRegularTier(tier)){if(job<count){tile=umTopology[UM_TILES+offset+job];found=true;}else{job-=count;}}
   offset+=count;
  }
  if(!found||umTileMaximumWidth(tile)!=umTileMinimumWidth(tile)){return UMOwner();}
 }
 let width=umTileWidth(tile);let side=4u/width;if(lane>=side*side*side){return UMOwner();}
 return UMOwner(tile,lane,width,(umTopology[tile]&0x3fffffffu)+lane);
}
// One workgroup per tile. Merged launches take the certificate's general h
// list, then every seam 4h tile, so their serial latencies overlap instead
// of paying dependent launches; otherwise the umCellWidth tier.
// Jobs from umMergedTileJobs() on pack 64 regular coarse owners each
// (umRegularCoarseOwner), one lane per owner as regular fine work runs.
fn umMergedTileJobs()->u32 {
 if(umMergedCoarse){return (umSupport[7u*UM_TILES+17u]+umMergedPack-1u)/umMergedPack;}
 return umSupport[4u*UM_TILES+2u]+umSupport[7u*UM_TILES+17u];
}
fn umMergedTileJob(index:u32)->UMOwner {
 if(umFusedJobs){return umFusedOwner(vec3u(index,0u,0u),0u,true);}
 var job=index;var tile=0u;var width=1u;let general=select(umSupport[4u*UM_TILES+2u],0u,umMergedCoarse);
 // No nested umSupport index: the frame plan rewrites these reads as atomics.
 let header=7u*UM_TILES+16u;let fine=umSupport[header];
 if(job<general){tile=umSupport[6u*UM_TILES+16u+job];}
 else{job-=general;width=4u;if(job>=umSupport[header+1u]){return UMOwner();}tile=umSupport[header+4u+fine+job];}
 return UMOwner(tile,0u,width,umTopology[tile]&0x3fffffffu);
}
fn umTileJobOwner(group:vec3u)->UMOwner {
 let job=group.x+umDispatchX*group.y;
 if(umMergedTiles){return umMergedTileJob(job);}
 let cells=64u/(umCellWidth*umCellWidth*umCellWidth);return umOwner(vec3u(job*cells,0u,0u));
}
// Jobs of this certified launch, from the frame plan's lists: tiles of the
// certificate list, or general h and seam 4h tile jobs then 64 packed regular
// 4h owners per job.
fn umCertifiedJobCount()->u32 {
 if(umCertifiedJobs==1u){return umSupport[4u*UM_TILES+umPlannedFine];}
 let merged=umMergedTileJobs()+(umSupport[8u*UM_TILES+20u]+63u)/64u;
 if(umCertifiedJobs==2u){return merged;}
 let fours=umSupport[7u*UM_TILES+17u];return merged-fours+(fours+3u)/4u;
}
fn umFusedGateOpen(tier:u32)->bool{return umFusedRegularGate==0u||(umFusedRegularGate==${UNIFORM_MIXED_FUSED_GATE.onlyFused}u)==umFusedRegularTier(tier);}
// umOwner's list length for tier: interface, planned certificate or tier tiles.
fn umTierListCount(tier:u32)->u32 {
 if(umInterfaceTiles){return umSupport[7u*UM_TILES+16u+tier];}
 if(tier==0u&&umPlannedFine!=0u){return umSupport[4u*UM_TILES+umPlannedFine];}
 return umCounts[tier];
}
// Jobs of a counted launch (UNIFORM_MIXED_COUNTED), from GPU counts only.
fn umCountedJobCount()->u32 {
 let tier=select(0u,1u,umCellWidth==4u);let header=7u*UM_TILES+16u;
 // The whole h list (a job per tile) or 4h list (64 tiles per job).
 if(umCountedJobs==${UNIFORM_MIXED_COUNTED.fineTiles}u){return umCounts.x;}
 if(umCountedJobs==${UNIFORM_MIXED_COUNTED.coarseTiles}u){return (umCounts.y+63u)/64u;}
 if(umCountedJobs==${UNIFORM_MIXED_COUNTED.owners}u||umCountedJobs==${UNIFORM_MIXED_COUNTED.tiles}u){
  // A regular launch with no regular tile is empty (dispatchRegular skips it).
  if(!umFusedGateOpen(tier)||(umRegularTiles&&umCounts[tier]<=umSupport[header+tier])){return 0u;}
  let tiles=umTierListCount(tier);
  if(umCountedJobs==${UNIFORM_MIXED_COUNTED.tiles}u){return tiles;}
  return (tiles*(64u>>(6u*tier))+63u)/64u;
 }
 if(umCountedJobs==${UNIFORM_MIXED_COUNTED.all}u){return umCounts.x+(umCounts.y+63u)/64u;}
 if(umCountedJobs==${UNIFORM_MIXED_COUNTED.residentAll}u){return umCounts.x+umResidentPageCount();}
 if(umCountedJobs==${UNIFORM_MIXED_COUNTED.residentPages}u){return umResidentPageCount();}
 if(umCountedJobs==${UNIFORM_MIXED_COUNTED.regularCoarse}u){if(!umFusedGateOpen(1u)){return 0u;}return (umSupport[8u*UM_TILES+20u]+63u)/64u;}
 let fine=umSupport[header];let fours=umSupport[header+1u];
 // The cache's own modules bound their slots by umHangingSlots(); the
 // builder leaves slots past it unslotted and flags UM_OVERFLOW_HANGING.
 if(umCountedJobs==${UNIFORM_MIXED_COUNTED.hanging}u){return fine+fours;}
 var jobs=fine+fours;
 if(umFusedJobs){for(var t=0u;t<2u;t++){if(umFusedRegularTier(t)){jobs+=umCounts[t];}}}
 if(umCountedJobs==${UNIFORM_MIXED_COUNTED.fusedQuad}u){jobs-=fours-(fours+3u)/4u;}
 return jobs;
}
fn umLaunchJobCount()->u32 {
 if(umCountedJobs!=0u){return umCountedJobCount();}
 return umCertifiedJobCount();
}
fn umOrigin(o:UMOwner)->vec3u{return umTileCoord(o.tile)*4u+umCorner(o.lane,4u/o.width)*o.width;}
${uniformMixedFacesWGSL}
`;
  // Transfers read two ownership levels in one shader. Namespace their entire
  // ABI, retaining the same lookup implementation and buffer representation.
  return prefix ? source.replace(/\b(?:um[A-Z]\w*|UM_\w+|UMOwner|UMFace)\b/g, name => prefix + name) : source;
}

/** Rewrites each named compute entry of `source` into a fixed grid-stride
 * launch over its jobs (umLaunchJobCount): the entry's body runs once per job
 * as if it were workgroup (job,0,0) of a one-row launch, so every owner lookup
 * keeps its meaning. Compile the entries with umCertifiedJobs
 * (ownership.dispatchCertified) or umCountedJobs (UNIFORM_MIXED_COUNTED,
 * ownership.dispatch*Counted) set; any grid is correct. Owner-slot modes
 * assume 64-lane entries. The job count is workgroup-uniform, so bodies may
 * keep their barriers. count: the WGSL expression of the job count, for a
 * list the ownership does not hold (a GPU-compacted list read by the module
 * itself, e.g. uniformMixedChangedTilesWGSL's umDilatedCount()); size the
 * launch to that list's bound. */
export function uniformMixedCertifiedEntriesWGSL(source: string, entries: readonly string[], count = "umLaunchJobCount()"): string {
  let out = source;
  for (const entry of entries) {
    const pattern = new RegExp(`@compute\\s+@workgroup_size\\((\\d+)\\)\\s+fn\\s+${entry}\\s*\\(((?:[^()]|\\([^()]*\\))*)\\)\\s*\\{`);
    const match = pattern.exec(out);
    if (!match) throw new Error(`Certified entry ${entry} not found`);
    const size = Number(match[1]);
    const args: string[] = [];
    const forwarded:string[]=[];
    const params = match[2]!.split(",").map(p => p.trim()).filter(Boolean).map(p => {
      const m = /^@builtin\((\w+)\)\s*(\w+)\s*:\s*(\w+)$/.exec(p);
      if (!m) throw new Error(`Certified entry ${entry} has an unsupported parameter ${p}`);
      const [, builtin, name, type] = m;
      args.push(builtin === "global_invocation_id" ? `vec3u(umJob*${size}u+umLane,0u,0u)`
        : builtin === "workgroup_id" ? "vec3u(umJob,0u,0u)"
        : builtin === "local_invocation_index" ? "umLane"
        : builtin === "subgroup_invocation_id" || builtin === "subgroup_size" ? (forwarded.push(`@builtin(${builtin}) um_${builtin}:${type}`),`um_${builtin}`)
        : (() => { throw new Error(`Certified entry ${entry} reads ${builtin}`); })());
      return `${name}:${type}`;
    });
    if (pattern.test(out.slice(match.index + 1))) throw new Error(`Certified entry ${entry} is ambiguous`);
    out = out.slice(0, match.index) + `fn ${entry}Job(${params.join(",")}){` + out.slice(match.index + match[0].length) + /* wgsl */ `
var<workgroup> ${entry}Jobs:u32;
@compute @workgroup_size(${size}) fn ${entry}(@builtin(workgroup_id) umGroup:vec3u,@builtin(num_workgroups) umGroups:vec3u,@builtin(local_invocation_index) umLane:u32${forwarded.length?","+forwarded.join(","):""}){
 if(umLane==0u){${entry}Jobs=${count};}
 let jobs=workgroupUniformLoad(&${entry}Jobs);
 for(var umJob=umGroup.x;umJob<jobs;umJob+=umGroups.x){${entry}Job(${args.join(",")});workgroupBarrier();}
}
`;
  }
  return out;
}
/** The same rewrite, named for GPU-counted launches (umCountedJobs). */
export const uniformMixedCountedEntriesWGSL=uniformMixedCertifiedEntriesWGSL;

/** Appended after native storage specialization, for certified boxes
 * without interior solids. Canonical mixed MAC input is bound as transportIn without a halo.
 * RK2 is shared with native; interior-solid collision walking is unnecessary.
 */
export function uniformMixedNativeTraceWGSL(layout: UniformMixedTopologyShape): string {
  return uniformMixedTopologyWGSL(layout, 1) + /* wgsl */ `
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32 {
 if(anchor[axis]<0){return boundaryVelocityIn[boundaryFaceIndex(max(anchor,vec3i(0)),axis)];}
 return textureLoad(transportIn,anchor,0)[axis];
}
${uniformMixedVelocitySamplingWGSL}
fn umTraceMixed(p:vec3f,dt:f32)->vec3f {
${uniformMidpointTraceWGSL("umSampleVelocity")}
 return end;
}
@compute @workgroup_size(64) fn uvMixedTrace(@builtin(global_invocation_id) gid:vec3u){
 let o=umOwner(gid);if(o.width==0u){return;}
 let origin=umOrigin(o);let p=vec3f(origin)+vec3f(0.5*f32(o.width));
 textureStore(velocityOut,vec3i(origin),vec4f(umTraceMixed(p,params.dimsDt.w),f32(o.width)));
}
`;
}
