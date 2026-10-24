import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedSolidWGSL, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";
import type { UniformPagePool } from "./uniform-page-pool";
import { UNIFORM_LIVE_KEY_GROUPS, uniformLiveListsOfWGSL } from "./uniform-live-lists";

/** Clearances are counted to this many tiles at most (a 16h page and more). */
export const UNIFORM_TILE_CONTEXT_CAP = 8;
/** Words ahead of the per-key words: the flags, then three spare. */
export const UNIFORM_TILE_CONTEXT_HEADER = 4;
/** Header flags: the clearance words hold this build's solids; no solid is
 * present at all (every tile reads the cap). Neither: every tile reads 0. */
export const UNIFORM_TILE_CONTEXT_VALID = 1;
export const UNIFORM_TILE_CONTEXT_FREE = 2;

const BUILD_PASS: GPUComputePassDescriptor = { label: "Uniform tile context" };

/** The context's accessors, for a module that binds its buffer read-only at
 * (group, binding) and includes them after uniformMixedTopologyWGSL.
 *
 * utcSolidClearance(t): the Chebyshev distance in tiles, at most
 * UNIFORM_TILE_CONTEXT_CAP, from stored tile key t to the nearest tile that
 * is closed: one no page stores (AIR, SOLID, outside the window), or one whose
 * cells or one-cell ring hold a cell the solids close (umCellOpen < 1, the
 * solid record's tile cut map). Every tile within distance c - 1 of t is then
 * stored and holds no such cell in itself or its ring: a trace or stencil
 * that stays within r tiles (plus one cell) of t meets no solid when
 * utcSolidClear(t, r), and every gated solid helper (umCellOpen,
 * umPressureFaceV, umSolidValid) answers there as it does with no solid. It
 * is 0 while bodies are present (they move after the build), and the cap
 * everywhere while no solid is (umSolidPresent false: the helpers are free
 * already). A key no page stores reads 0. */
export function uniformTileContextWGSL(group: number, binding: number): string {
  return /* wgsl */ `
@group(${group}) @binding(${binding}) var<storage,read> utcContext:array<u32>;
const UTC_CAP:u32=${UNIFORM_TILE_CONTEXT_CAP}u;
const UTC_CLEARANCE:u32=${UNIFORM_TILE_CONTEXT_HEADER}u;
fn utcSolidClearance(t:u32)->u32{
 let flags=utcContext[0];
 if((flags&${UNIFORM_TILE_CONTEXT_FREE}u)!=0u){return UTC_CAP;}
 if((flags&${UNIFORM_TILE_CONTEXT_VALID}u)==0u||!umKeyStored(t)){return 0u;}
 return min(utcContext[UTC_CLEARANCE+t],UTC_CAP);
}
fn utcSolidClear(t:u32,reach:u32)->bool{return utcSolidClearance(t)>reach;}
`;
}

/** Neighbour slots, for a module that includes uniformMixedTopologyNeighboursWGSL:
 * the key of the tile d tiles from key t (each axis within -8..8) from one
 * word of t's block's neighbour row (UniformBlockNeighbours), UM_TILES where
 * no page stores it. A kernel resolving a window resolves its page's 27 row
 * words once (utcNeighbourRow, lanes 0..26) and keys from them. */
export const uniformTileContextNeighboursWGSL = /* wgsl */ `
fn utcNeighbourTile(t:u32,d:vec3i)->u32{
 let l=vec3i(umLocalTile(t))+d;let w=umBlockNeighbour(t>>9u,l>>vec3u(3u));
 if(!umBlockStored(w)){return UM_TILES;}
 return 512u*w+umTileLocal(vec3u(l&vec3i(7)));
}
fn utcNeighbourRow(b:u32,lane:u32)->u32{return umBlockNeighbourWord(b,min(lane,26u));}
`;

/** A workgroup's tile and its six face neighbours, resolved once by its
 * lanes instead of per face: lanes 0..5 call utcLoadSide(t, lane) for tile
 * key t, then the workgroup takes a barrier before any read. Side s =
 * 2 axis + (positive): the neighbour's key (UM_TILES where no page stores
 * it), width (0 there) and page kind (UM_RESIDENT where stored, else the
 * table's AIR or SOLID); utcOrigin is t's first cell. */
export const uniformTileContextSidesWGSL = /* wgsl */ `
struct UTCSide {key:u32,width:u32,kind:u32}
var<workgroup> utcSides:array<UTCSide,6>;
var<workgroup> utcOrigin:vec3i;
fn utcLoadSide(t:u32,lane:u32){
 let c=vec3i(umTileCoord(t));var d=vec3i(0);d[lane/2u]=select(-1,1,(lane&1u)!=0u);
 let k=umTileKey(c+d);var side=UTCSide(UM_TILES,0u,umTileKind(c+d));
 if(umKeyStored(k)){side=UTCSide(k,umTileWidth(k),UM_RESIDENT);}
 utcSides[lane]=side;
 if(lane==0u){utcOrigin=4*c;}
}
`;

/** Per-tile velocity bounds, over the frame plan's certificate buffers as a
 * module already binds them (UniformMixedFramePlan: `speeds` is the
 * ownership's, `updateError` its updateErrors). Current from the frame's
 * certificate (encodeCertificate, after extension) until the next.
 * utcTileSpeed: the largest |face velocity| of the tile's owners, m/s.
 * utcTileGradient: its velocity gradient bound, 1/s (tileError).
 * utcTileReach: the signed per-axis reach in tiles (plus, minus) of every
 * characteristic from the tile this frame, the certificate's +2 margin in;
 * 31 saturated. utcTileSpreadReach: the same over the reach box (certify's).
 * utcTileClass: the certificate class of an h tile (UNIFORM_STAGE_CERTIFICATE). */
export function uniformTileContextBoundsWGSL(speeds: string, updateError: string): string {
  return /* wgsl */ `
fn utcTileSpeed(t:u32)->f32{return ${updateError}[8u*t+4u];}
fn utcTileGradient(t:u32)->f32{return ${updateError}[12u*UM_TILES+4u*t+3u];}
fn utcReachWord(w:u32)->array<vec3u,2>{return array<vec3u,2>(vec3u(w&31u,(w>>5u)&31u,(w>>10u)&31u),vec3u((w>>15u)&31u,(w>>20u)&31u,(w>>25u)&31u));}
fn utcTileReach(t:u32)->array<vec3u,2>{return utcReachWord(${speeds}[t]);}
fn utcTileSpreadReach(t:u32)->array<vec3u,2>{return utcReachWord(${speeds}[UM_TILES+t]);}
fn utcTileClass(t:u32)->u32{return ${speeds}[UM_TILES+t]>>30u;}
`;
}

/** Class bits of a tile, from words every topology module holds (no pass):
 * UTC_FINE an h tile, UTC_STENCIL a fine stencil (h support, every
 * neighbour h: its samples' taps are stored unit faces), UTC_CLEAR its
 * 27-tile neighbourhood solid-free (utcSolidClear(t, 1)). Requires
 * uniformTileContextWGSL. */
export const uniformTileContextClassWGSL = /* wgsl */ `
const UTC_FINE:u32=1u;const UTC_STENCIL:u32=2u;const UTC_CLEAR:u32=4u;
fn utcClassOf(t:u32)->u32{
 if(!umTileLive(t)){return 0u;}
 var bits=select(0u,UTC_FINE,umTileWidth(t)==1u);
 if((umTileSupport(t)&1u)!=0u&&umTileMaximumWidth(t)==1u){bits|=UTC_STENCIL;}
 if(utcSolidClear(t,1u)){bits|=UTC_CLEAR;}
 return bits;
}
`;

/** The frame's tile context: what a tile's neighbourhood holds, derived
 * once and read by any stage instead of probing per sample.
 *
 * Built here: the solid clearance (uniformTileContextWGSL), a distance
 * transform over the tile grid. A tile is closed where no page stores it or
 * its cut-map word is set; the Chebyshev distance to the nearest closed tile,
 * capped, is separable: three passes of one axis each (x from the closed
 * test, y and z over the pass before, an unstored tile reading 0), each a
 * fixed grid striding the pool's resident tile keys (uniform-live-lists.ts:
 * a key no page stores is written by none and read as 0). Static: it
 * is built after each solid record build (encodeBuild: a live edit, bodies)
 * and after each page commit that claimed or retired a block (encodeClaims:
 * the launches read the pool's claimed and retired counts and leave at once
 * when both are 0; a retired page is closed to its neighbours). While bodies are
 * present, or no solid is, the header says so and the passes write nothing.
 *
 * Wrapped (no storage of their own): the neighbour slots (the block
 * neighbour rows), the velocity bounds (the certificate's) and the class
 * bits (topology words): see the WGSL helpers above.
 *
 * Bind: layoutEntry(binding) in a consumer's group layout, groupEntry(binding)
 * in its group, uniformTileContextWGSL(group, binding) in its module. */
export class UniformTileContext {
  /** Header, then a clearance word a tile key, then two passes' scratch. */
  readonly buffer: GPUBuffer;
  readonly allocatedBytes: number;
  private readonly tiles: number;
  private readonly layout?: GPUBindGroupLayout;
  private group?: GPUBindGroup;
  private pipelines?: { readonly header: GPUComputePipeline; readonly passes: readonly GPUComputePipeline[]; readonly gated: readonly GPUComputePipeline[] };
  private readonly grid: number;
  /** solid: the frame's solid library with its record (none: free throughout).
   * pool: the page pool (its resident list is what the passes visit, its
   * claimed and retired counts gate encodeClaims). */
  constructor(private readonly device: GPUDevice, private readonly ownership: UniformMixedOwnership, private readonly solid: UniformMixedSolid | undefined, private readonly pool: UniformPagePool) {
    this.tiles = ownership.capacity.tiles;
    const words = UNIFORM_TILE_CONTEXT_HEADER + 3 * this.tiles;
    this.buffer = device.createBuffer({ label: "Uniform tile context", size: 4 * words, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.allocatedBytes = this.buffer.size;
    this.grid = Math.min(UNIFORM_LIVE_KEY_GROUPS, Math.ceil(this.tiles / 64));
    // No record: nothing to measure, and the helpers answer free.
    if (!solid?.coarse) { device.queue.writeBuffer(this.buffer, 0, new Uint32Array([UNIFORM_TILE_CONTEXT_FREE])); return; }
    this.layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ] });
    this.group = device.createBindGroup({ layout: this.layout, entries: [{ binding: 0, resource: { buffer: this.buffer } }, { binding: 1, resource: { buffer: pool.state } }] });
  }
  layoutEntry(binding: number): GPUBindGroupLayoutEntry { return { binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }; }
  groupEntry(binding: number): GPUBindGroupEntry { return { binding, resource: { buffer: this.buffer } }; }
  async initialize(): Promise<void> {
    const solid = this.solid;
    if (!solid?.coarse || !this.layout) return;
    const T = this.tiles, A = UNIFORM_TILE_CONTEXT_HEADER + T, B = A + T, cap = UNIFORM_TILE_CONTEXT_CAP;
    const code = uniformMixedTopologyWGSL(this.ownership.capacity, 0) + uniformMixedSolidWGSL(1, solid.coarse.count) + /* wgsl */ `
@group(2) @binding(0) var<storage,read_write> utcContext:array<u32>;
@group(2) @binding(1) var<storage,read> utcPool:array<u32>;
${uniformLiveListsOfWGSL("utcPool", this.pool.layout)}
// A commit launch: work only when the pool's last commit claimed or retired a block (status words 2 and 4).
override utcClaims:bool=false;
const UTC_CAP:u32=${cap}u;const UTC_CLEARANCE:u32=${UNIFORM_TILE_CONTEXT_HEADER}u;const UTC_A:u32=${A}u;const UTC_B:u32=${B}u;
// The solids can be measured: present, and no body (bodies move after a build).
fn utcMeasured()->bool{return umSolidPresent()&&umBodyCount()==0u;}
fn utcRun()->bool{return (!utcClaims||utcPool[2]!=0u||utcPool[4]!=0u)&&utcMeasured();}
@compute @workgroup_size(1) fn utcHeader(){
 utcContext[0]=select(select(0u,${UNIFORM_TILE_CONTEXT_VALID}u,utcMeasured()),${UNIFORM_TILE_CONTEXT_FREE}u,!umSolidPresent());
}
fn utcClosed(c:vec3i)->bool{let k=umTileKey(c);if(!umKeyStored(k)){return true;}return umSolidScratch[umSolidCutMap()+k]!=0u;}
// The distance a pass read: 0 where no page stores the tile.
fn utcPassed(base:u32,c:vec3i)->u32{let k=umTileKey(c);if(!umKeyStored(k)){return 0u;}return utcContext[base+k];}
// The k-th resident tile key, UM_TILES where its block's tile holds no page.
fn utcKey(k:u32)->u32{let t=512u*ulResidentBlock(k>>9u)+(k&511u);return select(UM_TILES,t,umTileLive(t));}
fn utcAxisXTile(t:u32){
 let c=vec3i(umTileCoord(t));var d=UTC_CAP;
 for(var s=0;s<i32(UTC_CAP);s++){if(utcClosed(c+vec3i(s,0,0))||utcClosed(c-vec3i(s,0,0))){d=u32(s);break;}}
 utcContext[UTC_A+t]=d;
}
fn utcAxis(t:u32,axis:u32,source:u32)->u32{
 let c=vec3i(umTileCoord(t));var d=utcContext[source+t];
 for(var s=1;s<i32(UTC_CAP);s++){
  if(u32(s)>=d){break;}
  var e=vec3i(0);e[axis]=s;
  d=min(d,max(u32(s),min(utcPassed(source,c+e),utcPassed(source,c-e))));
 }
 return d;
}
@compute @workgroup_size(64) fn utcAxisX(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 if(!utcRun()){return;}
 for(var k=gid.x;k<512u*ulResidentCount();k+=64u*groups.x){let t=utcKey(k);if(t!=UM_TILES){utcAxisXTile(t);}}
}
@compute @workgroup_size(64) fn utcAxisY(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 if(!utcRun()){return;}
 for(var k=gid.x;k<512u*ulResidentCount();k+=64u*groups.x){let t=utcKey(k);if(t!=UM_TILES){utcContext[UTC_B+t]=utcAxis(t,1u,UTC_A);}}
}
@compute @workgroup_size(64) fn utcAxisZ(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 if(!utcRun()){return;}
 for(var k=gid.x;k<512u*ulResidentCount();k+=64u*groups.x){let t=utcKey(k);if(t!=UM_TILES){utcContext[UTC_CLEARANCE+t]=utcAxis(t,2u,UTC_B);}}
}
`;
    const module = this.device.createShaderModule({ label: "Uniform tile context", code });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join("\n"));
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.ownership.bindLayout, solid.tileLayout, this.layout] });
    const compile = (entryPoint: string, claims: boolean) => this.device.createComputePipelineAsync({ layout, compute: { module, entryPoint, constants: entryPoint === "utcHeader" ? {} : { umDispatchX: this.ownership.dispatchX, utcClaims: +claims } } });
    const axes = ["utcAxisX", "utcAxisY", "utcAxisZ"];
    const [header, ...rest] = await Promise.all([compile("utcHeader", false), ...axes.map(a => compile(a, false)), ...axes.map(a => compile(a, true))]);
    this.pipelines = { header: header!, passes: rest.slice(0, 3), gated: rest.slice(3) };
  }
  /** After a solid record build (UniformMixedSolid.encodeCoarse building:
   * recordStale before it): the header and every clearance. */
  encodeBuild(encoder: GPUCommandEncoder): void { this.encode(encoder, false); }
  /** After a page commit (UniformMixedFrame.encodePageCommit), its claimed
   * blocks' record built: every clearance again where the commit claimed a
   * block (read on the GPU), nothing otherwise. */
  encodeClaims(encoder: GPUCommandEncoder): void { this.encode(encoder, true); }
  private encode(encoder: GPUCommandEncoder, claims: boolean): void {
    const p = this.pipelines;
    if (!this.solid?.coarse) return;
    if (!p || !this.group) throw new Error("Uniform tile context is not initialized");
    const pass = encoder.beginComputePass(BUILD_PASS);
    pass.setBindGroup(0, this.ownership.bindGroup); pass.setBindGroup(1, this.solid.tileGroup); pass.setBindGroup(2, this.group);
    if (!claims) { pass.setPipeline(p.header); pass.dispatchWorkgroups(1); }
    const passes = claims ? p.gated : p.passes;
    for (let i = 0; i < passes.length; i++) { pass.setPipeline(passes[i]!); pass.dispatchWorkgroups(this.grid); }
    pass.end();
  }
  destroy(): void { this.buffer.destroy(); }
}
