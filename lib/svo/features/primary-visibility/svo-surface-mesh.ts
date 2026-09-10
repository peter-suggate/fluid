import { svoCellContourWGSL } from "../construction/svo-cell-contour";
import { SPARSE_SCENE_MAINTENANCE_INCOMPLETE_OVERFLOW, SPARSE_SCENE_MAINTENANCE_STATE_WORDS } from "../../../core/webgpu-sparse-scene-proxies";
import type { WorkProgress } from "../../../core/work-progress";
import { SVO_SCREEN_SPACE_TERMINATION_CONTRACT } from "../lighting-visibility/svo-screen-space-termination";

/**
 * Cached opaque voxel boundary quads. All coordinates are on the accepted
 * finest-cell lattice, so adjacent bricks produce identical shared vertices.
 * The GPU publication header is also the indirect draw/dispatch buffer.
 *
 * Every brick owns one contiguous quad range in a bump-allocated arena, and a
 * per-leaf table records that range with the brick's address key. A scene
 * publication re-extracts only the bricks the voxelizer's maintenance pass
 * marked dirty (plus every brick whose slot key changed), freeing each old
 * range in place while the rest of the mesh keeps drawing. A whole-mesh
 * rebuild (first build, an unreliable dirty list, or compaction of the holes
 * those frees leave) extracts into a second arena while the first is still
 * drawn, then flips. While any build is pending, quads inside the dirty
 * boxes are culled and the pixels whose rays cross those boxes are traced,
 * so the edited region is exact and everything else stays rasterized.
 */
export interface SvoSurfaceMeshStatus {
  state: "pending" | "ready" | "blocked";
  quads?: number;
  /** Surface-bearing resident bricks selected at each level, before frustum culling. */
  lodBricks?: readonly number[];
  detail?: string;
  requiredQuads?: number;
  /** Quads currently owned by live bricks in the drawn arena. */
  liveQuads?: number;
  capacityQuads?: number;
  allocatedBytes?: number;
  maximumBytes?: number;
  builds?: number;
  requirementComplete?: boolean;
  completedBricks?: number;
  totalBricks?: number;
  /** The drawn arena holds a complete mesh: a pending build updates it in place or beside it. */
  drawn?: boolean;
  buildPhase?: "extracting" | "capacity" | "complete";
  buildKind?: "incremental" | "initial" | "replacement";
  restartReason?: "initial" | "topology" | "geometry" | "publication" | "compaction";
  fallbackReason?: "budget" | "smooth" | "inside-solid" | "publication" | "extraction";
}

/** Mesh-specific meaning stays beside the producer; the panel only renders facts. */
export function surfaceMeshProgress(status?: SvoSurfaceMeshStatus): WorkProgress {
  if (!status) return { label: "Waiting for mesh publication", state: "waiting", detail: "Mesh counters have not arrived from the GPU." };
  const state = status.state === "ready" ? "complete" : status.state === "blocked" ? "waiting" : "active";
  const capacity = status.buildPhase === "capacity";
  const reason = status.restartReason === "topology" ? "Topology changed; re-extracting the affected bricks."
    : status.restartReason === "geometry" ? "Geometry changed; re-extracting the affected bricks."
    : status.restartReason === "publication" ? "Source publication changed; rebuilding the current scene."
    : status.restartReason === "compaction" ? "Compacting freed quad storage; rebuilding beside the drawn mesh." : undefined;
  return {
    label: status.state === "ready" ? "Mesh ready"
      : status.state === "blocked" ? "Raster unavailable"
      : capacity ? "Expanding mesh storage" : status.drawn ? "Updating voxel surfaces" : "Extracting voxel surfaces",
    state,
    completed: status.completedBricks,
    total: status.totalBricks,
    unit: "bricks",
    generation: status.builds,
    phase: status.state === "ready" ? "complete" : capacity ? "capacity" : "extracting",
    phases: [{ id: "extracting", label: "Extract" }, { id: "capacity", label: "Storage" }, { id: "complete", label: "Ready" }],
    detail: [reason, status.state === "pending"
      ? status.drawn ? "The cached mesh stays drawn; only edited regions are traced until their bricks are re-extracted."
        : "The current voxel scene is traced while its raster mesh is built." : undefined,
      status.state === "blocked" ? status.detail : undefined].filter(Boolean).join(" "),
  };
}

export const SVO_SURFACE_MESH_BYTES = 64 * 1024 * 1024;
export const SVO_SURFACE_MESH_QUAD_BYTES = 32;
export const SVO_SURFACE_MESH_HEADER_BYTES = 64;
/** First 64 bytes retain the public draw/diagnostic ABI; tail is builder state. */
export const SVO_SURFACE_MESH_STATE_BYTES = 224;
/** Smallest arena a replacement build is given; a compaction never shrinks below it. */
export const SVO_SURFACE_MESH_MINIMUM_QUADS = SVO_SURFACE_MESH_BYTES / SVO_SURFACE_MESH_QUAD_BYTES;
/** Dirty box slots; the last always holds the union of the others, so a build masks up to 63 boxes individually and a longer dirty list collapses to the union alone. */
export const SVO_SURFACE_MESH_BOX_CAPACITY = 64;
export const SVO_SURFACE_MESH_BOX_INDIVIDUAL_CAPACITY = SVO_SURFACE_MESH_BOX_CAPACITY - 1;
export const SVO_SURFACE_MESH_BOX_UNION_SLOT = SVO_SURFACE_MESH_BOX_CAPACITY - 1;
/** Words of one dirty box record: minimum xyz, pad, maximum xyz, pad. */
export const SVO_SURFACE_MESH_BOX_WORDS = 8;
/** Words per leaf: base, count | depth << 27, key x/y, selected LOD, history valid, previous threshold, previous cap. */
export const SVO_SURFACE_MESH_TABLE_WORDS = 8;
/** Words per worklist entry scratch: pending count, emitted count, previous base, previous count word. */
export const SVO_SURFACE_MESH_SCRATCH_WORDS = 4;
export const SVO_SURFACE_MESH_COUNT_MASK = 0x07ff_ffff;

/**
 * Word map of the mesh state buffer. Words 0-15 are the public header the
 * Dawn benchmark and the background fragment read; the tail is builder state.
 * Words marked host are written only by the host and read by the GPU, so a
 * host write never races a GPU store of the same word.
 */
export const SVO_SURFACE_MESH_STATE = Object.freeze({
  drawVertexCount: 0,
  drawInstanceCount: 1,
  drawFirstVertex: 2,
  drawFirstInstance: 3,
  /** Quads allocated in the drawn arena, holes included: the cull iterates this many. */
  frontCursor: 4,
  /** Bit 0: the target arena overflowed this batch. Bit 1: extraction exceeded a bound. */
  errorFlags: 5,
  topologyRevision: 6,
  geometryRevision: 7,
  /** Count/emit dispatch during a batch; cull dispatch after publication. */
  extractDispatch: 8,
  building: 11,
  builds: 12,
  /** The drawn arena holds a complete mesh. */
  usable: 13,
  processedBricks: 14,
  /** 1: camera inside a solid voxel. 2: smooth reconstruction selected. */
  withheld: 15,
  checkpoint: 16,
  overflowLength: 17,
  worklistCount: 18,
  restartReason: 19,
  allocateDispatch: 20,
  markDispatch: 23,
  /** Arena slot (0/1) that is drawn. */
  front: 26,
  /** 0 idle, 1 incremental into the front, 2 initial into the front, 3 replacement into the back. */
  mode: 27,
  backCursor: 28,
  liveQuads: 29,
  /** Bit 0 needBack, bit 1 markPending, bit 2 boxesPending, bit 3 started (back acquired). */
  flags: 30,
  wantedBackQuads: 31,
  /** host: bricks one batch extracts. */
  bricksPerBatch: 32,
  consumedMaintenanceRevision: 33,
  boxCount: 34,
  liveTarget: 35,
  batchBegin: 36,
  batchEnd: 37,
  /** host: 1 when a maintenance dirty list is bound. */
  hostMaintenance: 38,
  /** host: leaf slots the range table covers. */
  hostLeafCapacity: 39,
  /** host: generation of the back arena binding, bumped on every allocation. */
  hostBackGeneration: 40,
  backGenerationConsumed: 41,
  batchPending: 42,
  /** host: word offset of the maintenance state block in the bound maintenance buffer. */
  hostMaintenanceStateWords: 43,
  /** host: word offset of the dirty brick records in the bound maintenance buffer. */
  hostMaintenanceRecordsWords: 44,
  /** host: dirty brick records the maintenance buffer can hold. */
  hostMaintenanceCapacity: 45,
  contourMode: 46,
  lodBricks: 48,
  wordCount: 56,
} as const);

export const SVO_SURFACE_MESH_FLAGS = Object.freeze({ needBack: 1, markPending: 2, boxesPending: 4, started: 8 } as const);
export const SVO_SURFACE_MESH_MODE = Object.freeze({ idle: 0, incremental: 1, initial: 2, replacement: 3 } as const);
export const SVO_SURFACE_MESH_RESTART_REASONS = Object.freeze(["publication", "initial", "topology", "geometry", "publication", "compaction"] as const);

/** Words of the work buffer before the per-leaf table: the dirty box records. */
export const SVO_SURFACE_MESH_WORK_TABLE_OFFSET_WORDS = SVO_SURFACE_MESH_BOX_CAPACITY * SVO_SURFACE_MESH_BOX_WORDS;
/** Work buffer size for `leafCapacity` slots: boxes, range table, scratch, worklist. */
export function surfaceMeshWorkBytes(leafCapacity: number): number {
  const slots = Math.max(1, Math.floor(leafCapacity));
  return (SVO_SURFACE_MESH_WORK_TABLE_OFFSET_WORDS + slots * (SVO_SURFACE_MESH_TABLE_WORDS + SVO_SURFACE_MESH_SCRATCH_WORDS + 1)) * 4;
}

/**
 * Bricks one presentation extracts on the first presentation of a build. Each
 * brick is counted and then emitted, about 8 µs of GPU work per brick pair on
 * the hero garden, so this is a few milliseconds beside the frame.
 */
export const SVO_SURFACE_MESH_BUILD_BRICKS_INITIAL = 512;
/**
 * Ceiling while the drawn mesh stays visible: a replacement or incremental
 * build shares the frame with the rasterized scene, so it is paced to keep
 * the frame interactive rather than to finish soonest.
 */
export const SVO_SURFACE_MESH_BUILD_BRICKS_DRAWN_MAXIMUM = 2048;
/**
 * Ceiling while nothing is drawn: every presentation of an initial build also
 * traces the whole scene at full resolution as its fallback, and on a large
 * scene that trace outweighs the extraction, so the build races to finish.
 */
export const SVO_SURFACE_MESH_BUILD_BRICKS_UNDRAWN_MAXIMUM = 16384;

/**
 * Bricks to extract in the presentation after `pendingPresentations`
 * consecutive pending presentations of one build. Zero is the first.
 */
export function surfaceMeshBuildBricks(pendingPresentations: number, drawn: boolean): number {
  const ramp = Math.max(0, Math.floor(pendingPresentations));
  const ceiling = drawn ? SVO_SURFACE_MESH_BUILD_BRICKS_DRAWN_MAXIMUM : SVO_SURFACE_MESH_BUILD_BRICKS_UNDRAWN_MAXIMUM;
  return Math.min(ceiling, SVO_SURFACE_MESH_BUILD_BRICKS_INITIAL * 2 ** Math.min(ramp, 30));
}

/** Presentations a build of `bricks` bricks needs under the ramp, from a cold start. */
export function surfaceMeshBuildPresentations(bricks: number, drawn: boolean): number {
  let remaining = Math.max(0, bricks); let presentations = 0;
  while (remaining > 0) {
    remaining -= surfaceMeshBuildBricks(presentations, drawn);
    presentations += 1;
  }
  return presentations;
}

/** What the host must do after reading one state receipt. */
export interface SvoSurfaceMeshReceipt {
  status: SvoSurfaceMeshStatus;
  /** Arena slot the GPU now draws. */
  front: 0 | 1;
  building: boolean;
  mode: number;
  /** The GPU is waiting for a back arena of at least this many bytes. */
  needBackBytes?: number;
  /** The target arena overflowed at this quad capacity and waits for a larger binding. */
  grow?: { slot: 0 | 1; overflowQuads: number };
}

/**
 * Interpret a state receipt. Pure so the host's decisions are testable
 * without a device: `arenaBytes` are the two arena slots as bound when the
 * receipt's frame was encoded.
 */
export function interpretSurfaceMeshState(words: ArrayLike<number>, context: { arenaBytes: readonly [number, number]; maximumBytes: number }): SvoSurfaceMeshReceipt {
  const W = SVO_SURFACE_MESH_STATE;
  const word = (index: number) => words[index] ?? 0;
  const front = (word(W.front) & 1) as 0 | 1;
  const mode = word(W.mode);
  const building = word(W.building) !== 0;
  const targetSlot = mode === SVO_SURFACE_MESH_MODE.replacement ? ((1 - front) as 0 | 1) : front;
  const flags = word(W.flags);
  const usable = word(W.usable) !== 0;
  const errors = word(W.errorFlags);
  const overflowed = building && (errors & 1) !== 0;
  const extractionFailed = (errors & 2) !== 0;
  const withheld = word(W.withheld);
  const allocatedBytes = context.arenaBytes[front];
  const targetBytes = context.arenaBytes[targetSlot];
  const needBack = building && mode === SVO_SURFACE_MESH_MODE.replacement && (flags & SVO_SURFACE_MESH_FLAGS.needBack) !== 0
    && (flags & SVO_SURFACE_MESH_FLAGS.started) === 0;
  const wantedBackBytes = Math.max(word(W.wantedBackQuads), SVO_SURFACE_MESH_MINIMUM_QUADS) * SVO_SURFACE_MESH_QUAD_BYTES;
  // A grow the host already answered is one the GPU has not yet observed.
  const grow = overflowed && targetBytes / SVO_SURFACE_MESH_QUAD_BYTES <= word(W.overflowLength);
  const canGrow = grow && targetBytes < context.maximumBytes && !extractionFailed;
  const budgetBlocked = grow && targetBytes >= context.maximumBytes;
  const fallback = !usable || withheld !== 0 || extractionFailed || budgetBlocked;
  const reason: NonNullable<SvoSurfaceMeshStatus["fallbackReason"]> = withheld === 2 ? "smooth"
    : withheld === 1 ? "inside-solid" : extractionFailed ? "extraction" : budgetBlocked ? "budget" : "publication";
  const pending = building && !extractionFailed && !budgetBlocked;
  const completedBricks = word(W.processedBricks);
  const totalBricks = word(W.worklistCount);
  const status: SvoSurfaceMeshStatus = {
    state: pending ? "pending" : fallback ? "blocked" : "ready",
    lodBricks: Array.from({ length: 4 }, (_, level) => word(W.lodBricks + level)),
    quads: word(W.drawInstanceCount), requiredQuads: word(W.frontCursor), liveQuads: word(W.liveQuads),
    capacityQuads: allocatedBytes / SVO_SURFACE_MESH_QUAD_BYTES, allocatedBytes, maximumBytes: context.maximumBytes,
    builds: word(W.builds), requirementComplete: !building && !extractionFailed, completedBricks, totalBricks,
    drawn: usable && withheld === 0 && !extractionFailed,
    buildPhase: building ? (overflowed || needBack ? "capacity" : "extracting") : "complete",
    buildKind: mode === SVO_SURFACE_MESH_MODE.incremental ? "incremental" : mode === SVO_SURFACE_MESH_MODE.initial ? "initial"
      : mode === SVO_SURFACE_MESH_MODE.replacement ? "replacement" : undefined,
    restartReason: SVO_SURFACE_MESH_RESTART_REASONS[word(W.restartReason)] ?? "publication",
  };
  if (pending) {
    status.detail = overflowed || needBack ? "Mesh storage growing; completed bricks are retained."
      : usable ? `Updating mesh: ${completedBricks.toLocaleString()} / ${totalBricks.toLocaleString()} bricks re-extracted; the cached mesh stays drawn and edited regions are traced.`
        : `Building mesh: ${completedBricks.toLocaleString()} / ${totalBricks.toLocaleString()} bricks processed; current voxels remain visible through exact traversal.`;
  } else if (fallback) {
    status.fallbackReason = reason;
    status.detail = reason === "smooth" ? "Raster requires voxel-flat surfaces; current SVO traversal remains visible."
      : reason === "inside-solid" ? "Camera is inside a solid voxel; current SVO traversal remains visible."
      : reason === "budget" ? "Surface mesh exceeds the allocation limit; current SVO traversal remains visible."
      : reason === "extraction" ? "Surface extraction exceeded its subdivision limit; current SVO traversal remains visible."
      : "Waiting for a complete voxel publication; exact planes remain visible.";
  }
  const receipt: SvoSurfaceMeshReceipt = { status, front, building, mode };
  if (needBack) receipt.needBackBytes = wantedBackBytes;
  if (canGrow) receipt.grow = { slot: targetSlot, overflowQuads: word(W.overflowLength) };
  return receipt;
}

export function svoSurfaceMeshWGSL(group: number, flatNormalsFlag: number, culling = true, contours = false): string {
  const W = SVO_SURFACE_MESH_STATE;
  const F = SVO_SURFACE_MESH_FLAGS;
  const M = SVO_SURFACE_MESH_MODE;
  return /* wgsl */ `
// \`face\` packs the boundary face in bits 0-2, the detail level in bits 3-5 and
// the brick's octree depth in bits 6-10 and normal agreement in bits 11-18.
// Level k quads bound cells of 2^k
// resident voxels; level 0 is the exact voxel boundary. A freed quad keeps its
// slot with a zero extent until the arena is compacted.
struct SurfaceQuad { origin:vec3u, identity:u32, extent:vec3u, face:u32 }
// Bit 31 selects a triangle: origin is the cell base and extent contains
// three packed 10-bit-per-axis local vertices. Vertex 3 repeats vertex 2.
${contours ? svoCellContourWGSL : ""}
fn meshContoursEnabled()->bool{return ${contours ? "dry.meshFilterNormals.w>0.5" : "false"};}
// Inflation is baked into each triangle record so the old front stays valid
// while replacement geometry is built with a new setting.
fn meshInflationCode()->u32{return ${contours ? "u32(round(clamp(dry.meshFilterNormals.w-1.0,0.0,0.5)*100.0))" : "0u"};}
fn meshRecordInflation(quad:SurfaceQuad)->f32{return f32((quad.face>>11u)&63u)/100.0;}
${contours ? `fn meshTrianglePoint(quad:SurfaceQuad,word:u32)->vec3f{
  let inflation=meshRecordInflation(quad);
  return vec3f(0.5)+(contourUnpackPoint(word)-vec3f(0.5))*(1.0+2.0*inflation);
}` : ""}
fn meshTriangle(quad:SurfaceQuad)->bool{return (quad.face&0x80000000u)!=0u;}
fn meshQuadExtent(quad:SurfaceQuad)->vec3u{
  if(meshTriangle(quad)){return vec3u(1u<<(dry.mapping.maximumDepth-meshQuadDepth(quad.face)));}
  return quad.extent;
}
fn meshContourCode(voxel:u32)->u32{
  ${contours ? "if(meshContoursEnabled()){return drySceneContourOfVoxel(voxel);}" : ""}
  return 0u;
}
@group(${group}) @binding(30) var<storage,read_write> meshState:array<atomic<u32>>;
@group(${group}) @binding(32) var<storage,read_write> meshArena0:array<SurfaceQuad>;
@group(${group}) @binding(38) var<storage,read_write> meshArena1:array<SurfaceQuad>;
@group(${group}) @binding(34) var<storage,read_write> meshVisibleOutput:array<u32>;
// Dirty boxes, per-leaf quad ranges, per-entry scratch and the brick worklist.
@group(${group}) @binding(40) var<storage,read_write> meshWork:array<atomic<u32>>;
// The voxelizer's maintenance arena: its state block (dirty brick count,
// overflow flags, requested and completed revisions) and dirty brick records
// sit at the word offsets the host stamps into the state buffer.
@group(${group}) @binding(41) var<storage,read> meshMaintenance:array<u32>;
@group(${group}) @binding(31) var<storage,read> meshQuads0:array<SurfaceQuad>;
@group(${group}) @binding(37) var<storage,read> meshQuads1:array<SurfaceQuad>;
@group(${group}) @binding(33) var<storage,read> meshHeader:array<u32>;
@group(${group}) @binding(35) var<storage,read> meshVisible:array<u32>;
@group(${group}) @binding(42) var<storage,read> meshWorkRead:array<u32>;

const MESH_BOX_CAPACITY:u32=${SVO_SURFACE_MESH_BOX_CAPACITY}u;
const MESH_BOX_INDIVIDUAL:u32=${SVO_SURFACE_MESH_BOX_INDIVIDUAL_CAPACITY}u;
const MESH_BOX_UNION:u32=${SVO_SURFACE_MESH_BOX_UNION_SLOT}u;
const MESH_BOX_WORDS:u32=${SVO_SURFACE_MESH_BOX_WORDS}u;
const MESH_TABLE_OFFSET:u32=${SVO_SURFACE_MESH_WORK_TABLE_OFFSET_WORDS}u;
const MESH_COUNT_MASK:u32=${SVO_SURFACE_MESH_COUNT_MASK}u;
const MESH_MINIMUM_QUADS:u32=${SVO_SURFACE_MESH_MINIMUM_QUADS}u;
const MESH_MAINTENANCE_INCOMPLETE:u32=${SPARSE_SCENE_MAINTENANCE_INCOMPLETE_OVERFLOW}u;
const MESH_MAINTENANCE_DIRTY_COUNT:u32=${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.dirtyBrickCount}u;
const MESH_MAINTENANCE_OVERFLOW:u32=${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.overflowFlags}u;
const MESH_MAINTENANCE_REQUESTED:u32=${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.requestedRevision}u;
const MESH_MAINTENANCE_COMPLETED:u32=${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.completedRevision}u;
const MESH_NO_ENTRY:u32=0xffffffffu;

fn meshQuadFace(word:u32)->u32{return word&7u;}
fn meshQuadLevel(word:u32)->u32{return (word>>3u)&7u;}
fn meshQuadDepth(word:u32)->u32{return (word>>6u)&31u;}
fn meshPackFace(face:u32,level:u32,depth:u32)->u32{return face|(level<<3u)|(depth<<6u);}
fn meshQuadDead(quad:SurfaceQuad)->bool{return all(quad.extent==vec3u(0u));}
// Levels a brick can offer: the voxel boundary plus one per halving of the
// brick edge down to a single cell.
fn meshLevelCount()->u32{return countTrailingZeros(max(dry.mapping.brickSize,1u))+1u;}
fn meshJobsPerBrick()->u32{return ${contours ? "6u*dry.mapping.brickSize+select(meshLevelCount()-1u,0u,meshContoursEnabled())" : "6u*dry.mapping.brickSize+meshLevelCount()-1u"};}
// Filtered-detail threshold in live pixels; zero is the exact voxel mesh.
// Authored at the screen-space contract's reference height so it stays angular.
fn dryMeshLodPixels()->f32{
  ${contours ? "if(meshContoursEnabled()){return 0.0;}" : ""}
  return select(0.0,max(dry.lod.w,0.0),dry.meshFilter.x>0.5)*uniforms.viewport.y/${SVO_SCREEN_SPACE_TERMINATION_CONTRACT.referenceViewportHeightPixels};
}
// Quads merge on material alone. A level-0 quad reads its voxel's baked normal
// per fragment, so its normal half is left absent; a coarse quad carries the
// mean baked normal of the voxels behind it that face the way it does.
fn meshMergeIdentity(identity:u32)->u32{return sceneIdentityMaterial(identity)|(SCENE_IDENTITY_NO_NORMAL<<16u);}

fn meshLeafCapacity()->u32{return atomicLoad(&meshState[${W.hostLeafCapacity}]);}
fn meshTableAt(leaf:u32)->u32{return MESH_TABLE_OFFSET+leaf*${SVO_SURFACE_MESH_TABLE_WORDS}u;}
fn meshScratchAt(entry:u32)->u32{return MESH_TABLE_OFFSET+meshLeafCapacity()*${SVO_SURFACE_MESH_TABLE_WORDS}u+entry*${SVO_SURFACE_MESH_SCRATCH_WORDS}u;}
fn meshListAt(entry:u32)->u32{return MESH_TABLE_OFFSET+meshLeafCapacity()*${SVO_SURFACE_MESH_TABLE_WORDS + SVO_SURFACE_MESH_SCRATCH_WORDS}u+entry;}
fn meshFront()->u32{return atomicLoad(&meshState[${W.front}])&1u;}
fn meshTargetArena()->u32{
  let front=meshFront();
  return select(front,1u-front,atomicLoad(&meshState[${W.mode}])==${M.replacement}u);
}
fn meshCursorWord(arena:u32)->u32{return select(${W.backCursor}u,${W.frontCursor}u,arena==meshFront());}
fn meshArenaLength(arena:u32)->u32{if(arena==0u){return arrayLength(&meshArena0);}return arrayLength(&meshArena1);}
fn meshArenaStore(arena:u32,index:u32,quad:SurfaceQuad){if(arena==0u){meshArena0[index]=quad;}else{meshArena1[index]=quad;}}
fn meshArenaLoad(arena:u32,index:u32)->SurfaceQuad{if(arena==0u){return meshArena0[index];}return meshArena1[index];}
fn meshDrawnQuad(index:u32)->SurfaceQuad{if((meshHeader[${W.front}]&1u)==0u){return meshQuads0[index];}return meshQuads1[index];}

// A leaf slot holds a drawable brick: a voxel terminal whose node links back
// to it and whose lifecycle is current. Dirty or relocating bricks are skipped
// and picked up again by the publication that completes them.
fn meshLeafCurrent(leaf:u32)->bool{
  if(leaf>=svoControlLoad(1u)){return false;}
  let topology=svoLeafLoad(leaf).topology;
  if(topology.x>=svoControlLoad(0u)||topology.z!=0u){return false;}
  let node=svoNodeLoad(topology.x);
  return node.links.z==leaf&&svoBrickLifecycleCurrent(svoBrickLifecycleDecode(node.links.w));
}
struct MeshBrickBox { minimum:vec3u, maximum:vec3u }
// Closed lattice box of a leaf's brick, so bricks that only share a face still overlap.
fn meshLeafBox(leaf:u32)->MeshBrickBox{
  let node=svoNodeLoad(svoLeafLoad(leaf).topology.x);
  let size=(1u<<(dry.mapping.maximumDepth-node.address.z))*dry.mapping.brickSize;
  let base=svoDecodeMorton(node.address.x,node.address.y,node.address.z)*size;
  return MeshBrickBox(base,base+vec3u(size));
}
fn meshBoxOverlaps(a:MeshBrickBox,bMinimum:vec3u,bMaximum:vec3u)->bool{
  return all(a.minimum<=bMaximum)&&all(a.maximum>=bMinimum);
}
fn meshBoxMinimum(box:u32)->vec3u{
  let base=box*MESH_BOX_WORDS;
  return vec3u(atomicLoad(&meshWork[base]),atomicLoad(&meshWork[base+1u]),atomicLoad(&meshWork[base+2u]));
}
fn meshBoxMaximum(box:u32)->vec3u{
  let base=box*MESH_BOX_WORDS+4u;
  return vec3u(atomicLoad(&meshWork[base]),atomicLoad(&meshWork[base+1u]),atomicLoad(&meshWork[base+2u]));
}
fn meshMaskActive()->bool{
  return atomicLoad(&meshState[${W.building}])!=0u&&atomicLoad(&meshState[${W.boxCount}])!=0u;
}
// Whether a brick touches any dirty box: such a brick's faces against the
// dirty brick may have changed, so it is re-extracted with it.
fn meshBoxesOverlap(box:MeshBrickBox)->bool{
  let unionMinimum=meshBoxMinimum(MESH_BOX_UNION);let unionMaximum=meshBoxMaximum(MESH_BOX_UNION);
  if(any(unionMinimum>unionMaximum)||!meshBoxOverlaps(box,unionMinimum,unionMaximum)){return false;}
  let count=min(atomicLoad(&meshState[${W.boxCount}]),MESH_BOX_INDIVIDUAL);
  for(var i=0u;i<count;i+=1u){
    let minimum=meshBoxMinimum(i);let maximum=meshBoxMaximum(i);
    if(any(minimum>maximum)){continue;}
    if(meshBoxOverlaps(box,minimum,maximum)){return true;}
  }
  return false;
}
// Whether a quad lies wholly inside a dirty box. Only such quads yield to the
// trace: every ray that could hit one touches the box and is traced, while a
// quad that merely touches a box keeps drawing, its untouched part being
// exactly what the edit left alone.
fn meshBoxesContain(minimum:vec3u,maximum:vec3u)->bool{
  let unionMinimum=meshBoxMinimum(MESH_BOX_UNION);let unionMaximum=meshBoxMaximum(MESH_BOX_UNION);
  if(any(unionMinimum>unionMaximum)||any(minimum<unionMinimum)||any(maximum>unionMaximum)){return false;}
  let count=min(atomicLoad(&meshState[${W.boxCount}]),MESH_BOX_INDIVIDUAL);
  for(var i=0u;i<count;i+=1u){
    let boxMinimum=meshBoxMinimum(i);let boxMaximum=meshBoxMaximum(i);
    if(any(boxMinimum>boxMaximum)){continue;}
    if(all(minimum>=boxMinimum)&&all(maximum<=boxMaximum)){return true;}
  }
  return false;
}

struct MeshRegion { origin:vec3f, size:f32, identity:u32, contour:u32 }
// Lookup returns the containing cell OR empty octree region. Its extent lets
// a coarse boundary stop subdividing as soon as its neighbour is uniform.
fn meshRegionAt(p:vec3f)->MeshRegion {
  let rootSize=f32((1u<<dry.mapping.maximumDepth)*dry.mapping.brickSize);
  if(any(p<vec3f(0.0))||any(p>=vec3f(rootSize))){return MeshRegion(vec3f(-rootSize),rootSize*3.0,0u,0u);}
  var origin=vec3f(0.0);var size=rootSize;var index=0u;
  for(var level=0u;level<=dry.mapping.maximumDepth;level+=1u){
    if(index>=svoControlLoad(0u)){return MeshRegion(origin,size,0u,0u);}
    let node=svoNodeLoad(index);
    if(node.links.z!=SVO_INVALID){
      let leaf=svoLeafLoad(node.links.z).topology;
      if(leaf.x!=index||leaf.z!=0u){return MeshRegion(origin,size,0u,0u);}
      if(!svoBrickLifecycleCurrent(svoBrickLifecycleDecode(node.links.w))){return MeshRegion(origin,size,0u,0u);}
      let cellSize=size/f32(dry.mapping.brickSize);
      let local=vec3u(clamp(floor((p-origin)/cellSize),vec3f(0.0),vec3f(f32(dry.mapping.brickSize-1u))));
      let voxel=svoBrickVoxelIndex(leaf.y,local,dry.mapping.brickSize);
      var identity=0u;if(voxel<dryVoxelCapacity()){identity=sceneIdentityAt(voxel);}
      return MeshRegion(origin+vec3f(local)*cellSize,cellSize,identity,select(0u,meshContourCode(voxel),sceneIdentitySolid(identity)));
    }
    size*=0.5;let upper=p>=origin+vec3f(size);
    let octant=select(0u,1u,upper.x)|select(0u,2u,upper.y)|select(0u,4u,upper.z);
    origin+=select(vec3f(0.0),vec3f(size),upper);
    let mask=node.address.w&255u;let bit=1u<<octant;
    if((mask&bit)==0u){return MeshRegion(origin,size,0u,0u);}
    index=node.links.x+countOneBits(mask&(bit-1u));
  }
  return MeshRegion(origin,size,0u,0u);
}
fn meshPointSolid(p:vec3f)->bool{
  let region=meshRegionAt(p);if(!sceneIdentitySolid(region.identity)){return false;}
  ${contours ? `if(region.contour!=0u){
    let clip=cellContour(sceneIdentityNormal(region.identity),dry.mapping.cellSize*region.size,region.contour);
    return dot(clip.normal,(p-region.origin)/region.size-vec3f(0.5))<=clip.high;
  }` : ""}
  return true;
}
// Whether the lattice-aligned cube of \`cell\` finest cells around \`p\` is solid
// at every resident voxel. A coarse boundary face hides behind a neighbour only
// when the neighbour's finest detail covers it completely, so a neighbour drawn
// finer than this brick can never open a gap; a partial neighbour leaves the
// face to the depth test. A neighbour subtree finer than the cube answers
// "uncovered" rather than being walked.
fn meshNeighbourCovered(p:vec3f,cell:u32)->bool {
  let n=dry.mapping.brickSize;
  let rootSize=f32((1u<<dry.mapping.maximumDepth)*n);
  if(any(p<vec3f(0.0))||any(p>=vec3f(rootSize))){return false;}
  var origin=vec3f(0.0);var size=rootSize;var index=0u;
  for(var level=0u;level<=dry.mapping.maximumDepth;level+=1u){
    if(index>=svoControlLoad(0u)){return false;}
    let node=svoNodeLoad(index);
    if(node.links.z!=SVO_INVALID){
      let leaf=svoLeafLoad(node.links.z).topology;
      if(leaf.x!=index||leaf.z!=0u){return false;}
      if(!svoBrickLifecycleCurrent(svoBrickLifecycleDecode(node.links.w))){return false;}
      let voxelSize=size/f32(n);
      if(voxelSize>=f32(cell)){
        let local=vec3u(clamp(floor((p-origin)/voxelSize),vec3f(0.0),vec3f(f32(n-1u))));
        let voxel=svoBrickVoxelIndex(leaf.y,local,n);
        return voxel<dryVoxelCapacity()&&sceneIdentitySolid(sceneIdentityAt(voxel));
      }
      let cubeOrigin=floor(p/f32(cell))*f32(cell);
      let first=vec3u(clamp(floor((cubeOrigin-origin)/voxelSize),vec3f(0.0),vec3f(f32(n-1u))));
      let span=min(u32(round(f32(cell)/voxelSize)),n);
      for(var z=0u;z<span;z+=1u){for(var y=0u;y<span;y+=1u){for(var x=0u;x<span;x+=1u){
        let local=min(first+vec3u(x,y,z),vec3u(n-1u));
        let voxel=svoBrickVoxelIndex(leaf.y,local,n);
        if(voxel>=dryVoxelCapacity()||!sceneIdentitySolid(sceneIdentityAt(voxel))){return false;}
      }}}
      return true;
    }
    if(size<=f32(cell)){return false;}
    size*=0.5;let upper=p>=origin+vec3f(size);
    let octant=select(0u,1u,upper.x)|select(0u,2u,upper.y)|select(0u,4u,upper.z);
    origin+=select(vec3f(0.0),vec3f(size),upper);
    let mask=node.address.w&255u;let bit=1u<<octant;
    if((mask&bit)==0u){return false;}
    index=node.links.x+countOneBits(mask&(bit-1u));
  }
  return false;
}
// Material of one level-k cell of a brick: the first solid voxel's, or zero
// for a cell of pure air. A cell is solid when any voxel in it is, so far
// detail dilates rather than vanishes. Solidity and merging are decided on
// this alone; the shading normal belongs to a face, not to the cell.
fn meshCellMaterial(payload:u32,cell:vec3u,level:u32,n:u32)->u32 {
  let span=1u<<level;
  for(var z=0u;z<span;z+=1u){for(var y=0u;y<span;y+=1u){for(var x=0u;x<span;x+=1u){
    let voxel=svoBrickVoxelIndex(payload,cell*span+vec3u(x,y,z),n);
    if(voxel>=dryVoxelCapacity()){continue;}
    let identity=sceneIdentityAt(voxel);
    if(sceneIdentitySolid(identity)){return sceneIdentityMaterial(identity);}
  }}}
  return 0u;
}
// Identity of one exposed face of a level-k cell: the cell's first solid
// material, with the normalised mean of the baked normals of just those solid
// voxels that face the same way as the quad. A rim cell's top voxels no longer
// tilt its side quad and its side voxels no longer tilt its top quad, and the
// two sides of a thin wall never cancel because each face averages its own
// side. Cache the mean and its agreement separately; the fragment applies
// the live agreement threshold without rebuilding the mesh.
var<private> meshFaceAgreement:u32;
var<private> meshMaskAgreement:array<u32,64>;
fn meshFaceIdentity(payload:u32,cell:vec3u,level:u32,n:u32,face:u32)->u32 {
  let span=1u<<level;let axis=face/2u;let towards=select(-1.0,1.0,(face&1u)!=0u);
  var material=0u;var sum=vec3f(0.0);var count=0u;
  for(var z=0u;z<span;z+=1u){for(var y=0u;y<span;y+=1u){for(var x=0u;x<span;x+=1u){
    let voxel=svoBrickVoxelIndex(payload,cell*span+vec3u(x,y,z),n);
    if(voxel>=dryVoxelCapacity()){continue;}
    let identity=sceneIdentityAt(voxel);
    if(!sceneIdentitySolid(identity)){continue;}
    if(material==0u){material=sceneIdentityMaterial(identity);}
    if(!sceneIdentityHasNormal(identity)){continue;}
    let baked=sceneIdentityNormal(identity);
    if(baked[axis]*towards<=0.0){continue;}
    sum+=baked;count+=1u;
  }}}
  meshFaceAgreement=0u;
  if(count>0u){meshFaceAgreement=u32(round(clamp(length(sum)/f32(count),0.0,1.0)*255.0));}
  if(material==0u){return 0u;}
  var normalWord=SCENE_IDENTITY_NO_NORMAL;
  if(dot(sum,sum)>1e-12){normalWord=svoGBufferPackNormalOct8(normalize(sum));}
  return material|(normalWord<<16u);
}
// Extraction runs twice per brick with identical traversal: the count phase
// sizes the brick's range, the emit phase fills it. These privates carry the
// phase and the brick's worklist entry through the shared emitters.
var<private> meshPhase:u32;
var<private> meshEntry:u32;
var<private> meshEmitArena:u32;
var<private> meshEmitBase:u32;
var<private> meshEmitCount:u32;
fn meshAppend(origin:vec3u,extent:vec3u,packedFace:u32,identity:u32){
  if(meshPhase==0u){atomicAdd(&meshWork[meshScratchAt(meshEntry)],1u);return;}
  let slot=atomicAdd(&meshWork[meshScratchAt(meshEntry)+1u],1u);
  // The two phases traverse the same published voxels; a mismatch is an
  // extraction fault, reported rather than written past the brick's range.
  if(slot>=meshEmitCount){atomicOr(&meshState[${W.errorFlags}],2u);return;}
  meshArenaStore(meshEmitArena,meshEmitBase+slot,SurfaceQuad(origin,identity,extent,packedFace));
}
${contours ? `
fn meshAppendContourPolygon(base:vec3u,depth:u32,identity:u32,polygon:ContourPolygon){
  for(var i=1u;i+1u<polygon.count;i+=1u){
    let a=contourPackPoint(polygon.points[0]);let b=contourPackPoint(polygon.points[i]);let c=contourPackPoint(polygon.points[i+1u]);
    if(a==b||b==c||c==a){continue;}
    let n=cross(contourUnpackPoint(b)-contourUnpackPoint(a),contourUnpackPoint(c)-contourUnpackPoint(a));
    if(dot(n,n)<1e-12){continue;}
    meshAppend(base,vec3u(a,b,c),0x80000000u|(meshInflationCode()<<11u)|meshPackFace(0u,0u,depth),identity);
  }
}
fn meshEmitContour(base:vec3u,scale:u32,depth:u32,identity:u32,code:u32){
  var contour=cellContour(sceneIdentityNormal(identity),dry.mapping.cellSize*f32(scale),code);
  let inflation=f32(meshInflationCode())/100.0;
  // Work in the expanded cube's normalized coordinates. Scaling the support
  // inversely keeps the world-space plane fixed instead of inflating the solid.
  contour.high/=1.0+2.0*inflation;
  if(contour.valid==0u){return;}
  for(var face=0u;face<6u;face+=1u){
    var polygon=contourClipPolygon(contourCubeFace(face),contour);
    if(polygon.count<3u){continue;}
    let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
    var p=vec3f(base)+vec3f(0.5*f32(scale));
    p[axis]=f32(base[axis])+select(-0.25,f32(scale)+0.25,(face&1u)!=0u);
    let neighbour=meshRegionAt(p);
    let fits=f32(base[u])>=neighbour.origin[u]&&f32(base[v])>=neighbour.origin[v]
      &&f32(base[u]+scale)<=neighbour.origin[u]+neighbour.size
      &&f32(base[v]+scale)<=neighbour.origin[v]+neighbour.size;
    if(inflation==0.0&&fits&&sceneIdentitySolid(neighbour.identity)){
      if(neighbour.contour==0u){continue;}
      // Keep only the face outside the neighbour's clipped solid. Translate
      // its support plane into this cell, including coarse/fine scale changes.
      let other=cellContour(sceneIdentityNormal(neighbour.identity),dry.mapping.cellSize*neighbour.size,neighbour.contour);
      let ratio=f32(scale)/neighbour.size;
      let offset=dot(other.normal,(vec3f(base)-neighbour.origin)/neighbour.size+vec3f(0.5*ratio-0.5));
      polygon=contourClipPolygon(polygon,CellContour(-other.normal,(offset-other.high)/ratio,1u));
    }
    // A face spanning several finer neighbours remains conservative; their
    // closed surfaces hide its covered portions in the depth test.
    meshAppendContourPolygon(base,depth,identity,polygon);
  }
  meshAppendContourPolygon(base,depth,identity,contourCap(contour));
}
` : ""}
// Greedy rectangles over one face layer's exposed-cell mask. \`m\` cells per
// side, each \`cell\` lattice units wide; the mask is consumed as it is merged.
fn meshEmitMask(mask:ptr<function,array<u32,64>>,m:u32,base:vec3u,cell:u32,face:u32,layer:u32,packedFace:u32){
  let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  for(var y=0u;y<m;y+=1u){for(var x=0u;x<m;x+=1u){
    let identity=(*mask)[x+y*m];if(identity==0u){continue;}
    let agreement=meshMaskAgreement[x+y*m];
    var width=1u;loop{if(x+width>=m){break;}if((*mask)[x+width+y*m]!=identity||meshMaskAgreement[x+width+y*m]!=agreement){break;}width+=1u;}
    var height=1u;loop{if(y+height>=m){break;}var same=true;for(var k=0u;k<width;k+=1u){if((*mask)[x+k+(y+height)*m]!=identity||meshMaskAgreement[x+k+(y+height)*m]!=agreement){same=false;}}
      if(!same){break;}height+=1u;}
    for(var j=0u;j<height;j+=1u){for(var k=0u;k<width;k+=1u){(*mask)[x+k+(y+j)*m]=0u;}}
    var origin=base;origin[axis]+=(layer+select(0u,1u,(face&1u)!=0u))*cell;origin[u]+=x*cell;origin[v]+=y*cell;
    var extent=vec3u(0u);extent[u]=width*cell;extent[v]=height*cell;meshAppend(origin,extent,packedFace|(agreement<<11u),identity);
  }}
}
// Neighbour coverage at mixed-resolution boundaries is resolved recursively,
// including partially empty fine children. A DFS needs at most 3*depth+1 slots.
fn meshBoundary(origin:vec3u,size:u32,face:u32,identity:u32,packedFace:u32){
  let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  var pending:array<vec4u,64>;var count=1u;pending[0]=vec4u(origin,size);
  loop {
    if(count==0u){break;}
    count-=1u;let item=pending[count];let o=item.xyz;let s=item.w;
    var p=vec3f(o);p[u]+=f32(s)*0.5;p[v]+=f32(s)*0.5;
    p[axis]+=select(-0.25,0.25,(face&1u)!=0u);
    let neighbour=meshRegionAt(p);
    let fits=f32(o[u])>=neighbour.origin[u]&&f32(o[v])>=neighbour.origin[v]
      &&f32(o[u]+s)<=neighbour.origin[u]+neighbour.size
      &&f32(o[v]+s)<=neighbour.origin[v]+neighbour.size;
    if(fits||s==1u){
      if((!sceneIdentitySolid(neighbour.identity)||neighbour.contour!=0u)){var extent=vec3u(0u);extent[u]=s;extent[v]=s;meshAppend(o,extent,packedFace,identity);}
    }else{
      if(count+4u>64u){atomicOr(&meshState[${W.errorFlags}],2u);break;}
      let half=s/2u;
      for(var child=0u;child<4u;child+=1u){var q=o;q[u]+=(child&1u)*half;q[v]+=((child>>1u)&1u)*half;pending[count]=vec4u(q,half);count+=1u;}
    }
  }
}
// One coarse level of one brick in a single invocation: every cell's material
// is derived once from the brick's own voxels, then all six faces' layers are
// masked from that table. Within the brick both sides of a face use the same
// dilated cells; across a brick boundary the face hides only behind complete
// coverage, so mixed levels between neighbours stay watertight. Exposure is a
// property of the cell, so it is decided from the table, but the identity a
// quad carries is the exposed face's own mean normal, read from the cell's
// voxels once per exposed face; an interior cell reads none.
fn meshBuildLevel(payload:u32,base:vec3u,scale:u32,depth:u32,level:u32,n:u32){
  let m=n>>level;let cell=scale<<level;
  var cells:array<u32,64>;
  for(var i=0u;i<m*m*m;i+=1u){cells[i]=meshCellMaterial(payload,vec3u(i%m,(i/m)%m,i/(m*m)),level,n);}
  let packedBase=meshPackFace(0u,level,depth);
  for(var face=0u;face<6u;face+=1u){
    let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;let positive=(face&1u)!=0u;
    for(var layer=0u;layer<m;layer+=1u){
      var mask:array<u32,64>;
      for(var y=0u;y<m;y+=1u){for(var x=0u;x<m;x+=1u){
        let index=x+y*m;mask[index]=0u;var c=vec3u(0u);c[axis]=layer;c[u]=x;c[v]=y;
        if(cells[c.x+c.y*m+c.z*m*m]==0u){continue;}
        let boundary=select(layer==0u,layer==m-1u,positive);
        var exposed=false;
        if(boundary){
          var p=vec3f(base+c*cell);p[u]+=f32(cell)*0.5;p[v]+=f32(cell)*0.5;
          p[axis]+=select(-0.25,f32(cell)+0.25,positive);
          exposed=!meshNeighbourCovered(p,cell);
        }else{
          var adjacent=c;adjacent[axis]=u32(i32(layer)+select(-1,1,positive));
          exposed=cells[adjacent.x+adjacent.y*m+adjacent.z*m*m]==0u;
        }
        if(exposed){mask[index]=meshFaceIdentity(payload,c,level,n,face);meshMaskAgreement[index]=meshFaceAgreement;}
      }}
      meshEmitMask(&mask,m,base,cell,face,layer,packedBase|face);
    }
  }
}
// One face layer or one coarse level of one brick.
fn meshExtractJob(leafIndex:u32,local:u32){
  let n=dry.mapping.brickSize;
  let leaf=svoLeafLoad(leafIndex).topology;
  let node=svoNodeLoad(leaf.x);
  let scale=1u<<(dry.mapping.maximumDepth-node.address.z);
  let base=svoDecodeMorton(node.address.x,node.address.y,node.address.z)*(scale*n);
  if(local>=6u*n){meshBuildLevel(leaf.y,base,scale,node.address.z,local-6u*n+1u,n);return;}
  // Face/layer masks are independent: each invocation owns exactly one.
  // Greedy rectangles and mixed-level boundary subdivision are unchanged;
  // only append order within a brick's range may differ.
  let face=local/n;let layer=local%n;
  let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  let packedFace=meshPackFace(face,0u,node.address.z);
  var mask:array<u32,64>;
  for(var y=0u;y<n;y+=1u){for(var x=0u;x<n;x+=1u){
    let m=x+y*n;mask[m]=0u;var c=vec3u(0u);c[axis]=layer;c[u]=x;c[v]=y;
    let voxel=svoBrickVoxelIndex(leaf.y,c,n);if(voxel>=dryVoxelCapacity()){continue;}
    let identity=meshMergeIdentity(sceneIdentityAt(voxel));if(!sceneIdentitySolid(identity)){continue;}
    ${contours ? `let code=meshContourCode(voxel);
    if(code!=0u){
      // The first face job for this cell emits its entire closed clipped cube.
      if(face==0u){meshEmitContour(base+c*scale,scale,node.address.z,sceneIdentityAt(voxel),code);}
      continue;
    }` : ""}
    var origin=base+c*scale;origin[axis]+=select(0u,scale,(face&1u)!=0u);
    let boundary=select(layer==0u,layer==n-1u,(face&1u)!=0u);
    if(boundary){
      var p=vec3f(origin);p[u]+=f32(scale)*0.5;p[v]+=f32(scale)*0.5;p[axis]+=select(-0.25,0.25,(face&1u)!=0u);
      let neighbour=meshRegionAt(p);
      let fits=f32(origin[u])>=neighbour.origin[u]&&f32(origin[v])>=neighbour.origin[v]
        &&f32(origin[u]+scale)<=neighbour.origin[u]+neighbour.size&&f32(origin[v]+scale)<=neighbour.origin[v]+neighbour.size;
      if(fits){if((!sceneIdentitySolid(neighbour.identity)||neighbour.contour!=0u)){mask[m]=identity;}}
      else{meshBoundary(origin,scale,face,identity,packedFace);}
      continue;
    }
    var adjacent=c;adjacent[axis]=u32(i32(layer)+select(-1,1,(face&1u)!=0u));
    let other=svoBrickVoxelIndex(leaf.y,adjacent,n);
    if(!sceneIdentitySolid(sceneIdentityAt(other))||meshContourCode(other)!=0u){mask[m]=identity;}
  }}
  meshEmitMask(&mask,n,base,scale,face,layer,packedFace);
}
fn meshStoreDispatch(word:u32,groups:u32){
  atomicStore(&meshState[word],min(groups,65535u));
  atomicStore(&meshState[word+1u],max(1u,(groups+65534u)/65535u));
  atomicStore(&meshState[word+2u],1u);
}
fn meshClearDispatch(word:u32){
  atomicStore(&meshState[word],0u);atomicStore(&meshState[word+1u],1u);atomicStore(&meshState[word+2u],1u);
}
fn meshMaintenanceBound()->bool{return atomicLoad(&meshState[${W.hostMaintenance}])!=0u;}
fn meshMaintenanceWord(index:u32)->u32{
  let at=atomicLoad(&meshState[${W.hostMaintenanceStateWords}])+index;
  if(at>=arrayLength(&meshMaintenance)){return 0u;}
  return meshMaintenance[at];
}
fn meshMaintenanceDirtyCount()->u32{
  return min(meshMaintenanceWord(MESH_MAINTENANCE_DIRTY_COUNT),atomicLoad(&meshState[${W.hostMaintenanceCapacity}]));
}
fn meshMaintenanceDirtyLeaf(index:u32)->u32{
  let at=atomicLoad(&meshState[${W.hostMaintenanceRecordsWords}])+index*4u;
  if(at>=arrayLength(&meshMaintenance)){return MESH_NO_ENTRY;}
  return meshMaintenance[at];
}

// Once per batch: notice a new publication, choose how to answer it, and
// wait for storage the host still has to provide. Every decision is GPU
// resident so a stale host receipt can never restart or skip a build.
@compute @workgroup_size(1)
fn surfaceMeshPrepare(){
  for(var level=0u;level<8u;level+=1u){atomicStore(&meshState[${W.lodBricks}u+level],0u);}
  atomicStore(&meshState[${W.drawVertexCount}],4u);
  meshClearDispatch(${W.extractDispatch}u);meshClearDispatch(${W.allocateDispatch}u);meshClearDispatch(${W.markDispatch}u);
  // Smooth reconstruction has a view-dependent face fallback and cannot be
  // represented by these cached boundary quads. Withhold raster geometry for this unsupported representation.
  if((dry.materialPublication.w&${flatNormalsFlag}u)==0u){atomicStore(&meshState[${W.withheld}],2u);atomicStore(&meshState[${W.drawInstanceCount}],0u);return;}
  let valid=dryPublicationWord(0u)!=0u&&(dryPublicationWord(1u)&REQUIRED_FIELDS)==REQUIRED_FIELDS;
  if(!valid){
    atomicStore(&meshState[${W.usable}],0u);atomicStore(&meshState[${W.drawInstanceCount}],0u);
    atomicStore(&meshState[${W.building}],0u);atomicStore(&meshState[${W.errorFlags}],0u);atomicStore(&meshState[${W.mode}],0u);
    return;
  }
  let p=(uniforms.cameraPosition.xyz-dry.mapping.worldOrigin)/dry.mapping.cellSize;
  atomicStore(&meshState[${W.withheld}],select(0u,1u,meshPointSolid(p)));
  var mode=atomicLoad(&meshState[${W.mode}]);
  var building=atomicLoad(&meshState[${W.building}])!=0u;
  var flags=atomicLoad(&meshState[${W.flags}]);
  let topologyChanged=atomicLoad(&meshState[${W.topologyRevision}])!=dryPublicationWord(2u);
  let contourChanged=atomicLoad(&meshState[${W.contourMode}])!=bitcast<u32>(dry.meshFilterNormals.w);
  let geometryChanged=atomicLoad(&meshState[${W.geometryRevision}])!=dryPublicationWord(3u);
  let usable=atomicLoad(&meshState[${W.usable}])!=0u;
  let first=atomicLoad(&meshState[${W.builds}])==0u||(!usable&&!building);
  // Freed ranges leave holes the cull still walks; once they outnumber the
  // live quads, rebuild beside the drawn mesh into a compact arena.
  let live=atomicLoad(&meshState[${W.liveQuads}]);
  let holes=atomicLoad(&meshState[${W.frontCursor}])-min(live,atomicLoad(&meshState[${W.frontCursor}]));
  let compact=!building&&usable&&!first&&!topologyChanged&&!geometryChanged&&holes>live+65536u;
  if(topologyChanged||geometryChanged||first||compact||contourChanged){
    atomicStore(&meshState[${W.contourMode}],bitcast<u32>(dry.meshFilterNormals.w));
    atomicStore(&meshState[${W.topologyRevision}],dryPublicationWord(2u));
    atomicStore(&meshState[${W.geometryRevision}],dryPublicationWord(3u));
    atomicAdd(&meshState[${W.builds}],1u);
    atomicStore(&meshState[${W.errorFlags}],0u);
    var reason=4u;
    if(first){reason=1u;}else if(compact){reason=5u;}else if(topologyChanged){reason=2u;}else{reason=3u;}
    var next=${M.replacement}u;
    var reliable=false;
    if(first){
      next=${M.initial}u;
      atomicStore(&meshState[${W.frontCursor}],0u);atomicStore(&meshState[${W.liveQuads}],0u);atomicStore(&meshState[${W.usable}],0u);
    }else if(!compact&&!contourChanged&&meshMaintenanceBound()){
      // The dirty list describes exactly one completed maintenance revision.
      // It is trusted only when that revision is the next one after the last
      // consumed, is complete, and no newer request has begun rewriting it.
      let requested=meshMaintenanceWord(MESH_MAINTENANCE_REQUESTED);let completed=meshMaintenanceWord(MESH_MAINTENANCE_COMPLETED);
      let overflow=meshMaintenanceWord(MESH_MAINTENANCE_OVERFLOW);
      reliable=requested==completed&&completed==atomicLoad(&meshState[${W.consumedMaintenanceRevision}])+1u
        &&(overflow&MESH_MAINTENANCE_INCOMPLETE)==0u;
      // An incremental answer joins a running incremental build; it never
      // joins a replacement build, whose completed bricks would be stale.
      if(reliable&&mode!=${M.replacement}u){next=${M.incremental}u;}
    }
    // Any publication the mesh answers consumes the revision the list holds,
    // so the next list is judged against it whether or not this one was used.
    if(meshMaintenanceBound()){atomicStore(&meshState[${W.consumedMaintenanceRevision}],meshMaintenanceWord(MESH_MAINTENANCE_COMPLETED));}
    if(reliable){flags|=${F.boxesPending}u;}else{atomicStore(&meshState[${W.boxCount}],0u);}
    // A publication that lands during an incremental build restarts the
    // worklist from a fresh mark, so no slot is listed twice; its dirty boxes
    // stay, so bricks the earlier publication still owed are re-marked.
    atomicStore(&meshState[${W.worklistCount}],0u);atomicStore(&meshState[${W.processedBricks}],0u);
    if(next==${M.replacement}u){
      atomicStore(&meshState[${W.backCursor}],0u);atomicStore(&meshState[${W.liveTarget}],0u);
      // Sized for the live mesh with a quarter's headroom; overflow grows it.
      atomicStore(&meshState[${W.wantedBackQuads}],max(live+live/4u+65536u,MESH_MINIMUM_QUADS));
      flags&=~${F.started}u;
    }
    if(next==${M.initial}u){atomicStore(&meshState[${W.liveTarget}],0u);}
    atomicStore(&meshState[${W.mode}],next);atomicStore(&meshState[${W.building}],1u);
    atomicStore(&meshState[${W.restartReason}],reason);
    flags|=${F.markPending}u;
    mode=next;building=true;
  }
  if(!building){atomicStore(&meshState[${W.flags}],flags);return;}
  // A replacement build needs a back arena the host has bound since the last
  // flip. The generation the host stamps is consumed only by a flip, so a
  // restart before the flip keeps the arena it already has; an arena smaller
  // than wanted is not refused, it overflows and grows like any other.
  if(mode==${M.replacement}u&&(flags&${F.started}u)==0u){
    let generation=atomicLoad(&meshState[${W.hostBackGeneration}]);
    if(generation!=atomicLoad(&meshState[${W.backGenerationConsumed}])){
      flags|=${F.started}u;flags&=~${F.needBack}u;
    }else{
      flags|=${F.needBack}u;atomicStore(&meshState[${W.flags}],flags);return;
    }
  }
  // The host copies the retained prefix into a larger arena. Resume only when
  // that larger binding is actually visible; stale receipts never clear GPU
  // error state or move the cursor.
  if((atomicLoad(&meshState[${W.errorFlags}])&1u)!=0u){
    if(meshArenaLength(meshTargetArena())>atomicLoad(&meshState[${W.overflowLength}])){atomicAnd(&meshState[${W.errorFlags}],~1u);}
    else{atomicStore(&meshState[${W.flags}],flags);return;}
  }
  // The mark stays pending until the schedule pass that follows it has run,
  // so a presentation that encodes no build passes cannot complete a build
  // over an empty worklist.
  if((flags&${F.markPending}u)!=0u){meshStoreDispatch(${W.markDispatch}u,(meshLeafCapacity()+63u)/64u);}
  atomicStore(&meshState[${W.flags}],flags);
}
// Dirty boxes of the maintenance revision being answered: one closed lattice
// box per dirty brick while they fit, else their union. Runs as one workgroup
// so the union can be reduced with workgroup-uniform barriers.
var<workgroup> meshBoxUnionMinimum:array<atomic<u32>,3>;
var<workgroup> meshBoxUnionMaximum:array<atomic<u32>,3>;
@compute @workgroup_size(64)
fn surfaceMeshBoxes(@builtin(local_invocation_index) lane:u32){
  let pending=(atomicLoad(&meshState[${W.flags}])&${F.boxesPending}u)!=0u&&meshMaintenanceBound();
  var count=0u;
  if(pending){count=meshMaintenanceDirtyCount();}
  if(lane<3u){atomicStore(&meshBoxUnionMinimum[lane],0xffffffffu);atomicStore(&meshBoxUnionMaximum[lane],0u);}
  workgroupBarrier();
  var previous=0u;
  if(pending){previous=atomicLoad(&meshState[${W.boxCount}]);}
  // A publication that lands during an incremental build appends its boxes
  // to the ones that build still owes.
  let joining=pending&&previous>0u&&atomicLoad(&meshState[${W.mode}])==${M.incremental}u;
  let base=select(0u,min(previous,MESH_BOX_INDIVIDUAL),joining);
  let individual=pending&&base+count<=MESH_BOX_INDIVIDUAL;
  if(pending){
    for(var i=lane;i<count;i+=64u){
      let leaf=meshMaintenanceDirtyLeaf(i);
      var minimum=vec3u(0xffffffffu);var maximum=vec3u(0u);
      if(leaf!=MESH_NO_ENTRY&&meshLeafCurrent(leaf)){let box=meshLeafBox(leaf);minimum=box.minimum;maximum=box.maximum;}
      if(individual){
        let out=(base+i)*MESH_BOX_WORDS;
        atomicStore(&meshWork[out],minimum.x);atomicStore(&meshWork[out+1u],minimum.y);atomicStore(&meshWork[out+2u],minimum.z);
        atomicStore(&meshWork[out+4u],maximum.x);atomicStore(&meshWork[out+5u],maximum.y);atomicStore(&meshWork[out+6u],maximum.z);
      }
      for(var axis=0u;axis<3u;axis+=1u){atomicMin(&meshBoxUnionMinimum[axis],minimum[axis]);atomicMax(&meshBoxUnionMaximum[axis],maximum[axis]);}
    }
    // The boxes already owed join the union either way.
    if(joining&&lane==0u){
      let minimum=meshBoxMinimum(MESH_BOX_UNION);let maximum=meshBoxMaximum(MESH_BOX_UNION);
      for(var axis=0u;axis<3u;axis+=1u){atomicMin(&meshBoxUnionMinimum[axis],minimum[axis]);atomicMax(&meshBoxUnionMaximum[axis],maximum[axis]);}
    }
  }
  workgroupBarrier();
  if(lane==0u&&pending){
    var unionMinimum=vec3u(0u);var unionMaximum=vec3u(0u);
    for(var axis=0u;axis<3u;axis+=1u){unionMinimum[axis]=atomicLoad(&meshBoxUnionMinimum[axis]);unionMaximum[axis]=atomicLoad(&meshBoxUnionMaximum[axis]);}
    let out=MESH_BOX_UNION*MESH_BOX_WORDS;
    atomicStore(&meshWork[out],unionMinimum.x);atomicStore(&meshWork[out+1u],unionMinimum.y);atomicStore(&meshWork[out+2u],unionMinimum.z);
    atomicStore(&meshWork[out+4u],unionMaximum.x);atomicStore(&meshWork[out+5u],unionMaximum.y);atomicStore(&meshWork[out+6u],unionMaximum.z);
    if(individual){atomicStore(&meshState[${W.boxCount}],base+count);}
    else{
      // Too many boxes to list: the union alone stands for them all.
      atomicStore(&meshWork[0],unionMinimum.x);atomicStore(&meshWork[1],unionMinimum.y);atomicStore(&meshWork[2],unionMinimum.z);
      atomicStore(&meshWork[4],unionMaximum.x);atomicStore(&meshWork[5],unionMaximum.y);atomicStore(&meshWork[6],unionMaximum.z);
      atomicStore(&meshState[${W.boxCount}],1u);
    }
    atomicAnd(&meshState[${W.flags}],~${F.boxesPending}u);
  }
}
fn meshAppendWork(leaf:u32){
  atomicStore(&meshWork[meshTableAt(leaf)+5u],0u);
  let index=atomicAdd(&meshState[${W.worklistCount}],1u);
  if(index>=meshLeafCapacity()){return;}
  atomicStore(&meshWork[meshListAt(index)],leaf);
  let scratch=meshScratchAt(index);
  atomicStore(&meshWork[scratch],0u);atomicStore(&meshWork[scratch+1u],0u);
}
// One thread per leaf slot: decide which bricks this build extracts. A whole
// build takes every current brick and clears the table entries of the rest.
// An incremental build takes the bricks whose slot key changed, the bricks
// that vanished (to free their ranges) and the bricks touching a dirty box.
@compute @workgroup_size(64)
fn surfaceMeshMark(@builtin(global_invocation_id) id:vec3u){
  let slot=id.x+id.y*65535u*64u;
  if(slot>=meshLeafCapacity()){return;}
  let table=meshTableAt(slot);
  let countWord=atomicLoad(&meshWork[table+1u]);
  let keyX=atomicLoad(&meshWork[table+2u]);let keyY=atomicLoad(&meshWork[table+3u]);
  let held=countWord!=0u||keyX!=0u||keyY!=0u;
  let current=meshLeafCurrent(slot);
  if(atomicLoad(&meshState[${W.mode}])!=${M.incremental}u){
    if(current){meshAppendWork(slot);}
    else if(held){
      atomicStore(&meshWork[table],0u);atomicStore(&meshWork[table+1u],0u);atomicStore(&meshWork[table+2u],0u);atomicStore(&meshWork[table+3u],0u);
    }
    return;
  }
  if(!current){if(held){meshAppendWork(slot);}return;}
  let node=svoNodeLoad(svoLeafLoad(slot).topology.x);
  let sameKey=held&&keyX==node.address.x&&keyY==node.address.y&&(countWord>>27u)==node.address.z;
  if(!sameKey||meshBoxesOverlap(meshLeafBox(slot))){meshAppendWork(slot);}
}
// Once per batch, after marking: the batch's brick span and its dispatches.
@compute @workgroup_size(1)
fn surfaceMeshSchedule(){
  meshClearDispatch(${W.extractDispatch}u);meshClearDispatch(${W.allocateDispatch}u);
  atomicStore(&meshState[${W.batchBegin}],0u);atomicStore(&meshState[${W.batchEnd}],0u);
  atomicStore(&meshState[${W.batchPending}],0u);
  if(atomicLoad(&meshState[${W.building}])==0u){return;}
  let flags=atomicLoad(&meshState[${W.flags}]);
  if(atomicLoad(&meshState[${W.mode}])==${M.replacement}u&&(flags&${F.started}u)==0u){return;}
  if((atomicLoad(&meshState[${W.errorFlags}])&1u)!=0u){return;}
  // Past every wait the mark pass has just run over the whole table.
  atomicAnd(&meshState[${W.flags}],~${F.markPending}u);
  let begin=atomicLoad(&meshState[${W.processedBricks}]);
  let total=min(atomicLoad(&meshState[${W.worklistCount}]),meshLeafCapacity());
  let end=min(total,begin+max(atomicLoad(&meshState[${W.bricksPerBatch}]),1u));
  atomicStore(&meshState[${W.batchBegin}],begin);atomicStore(&meshState[${W.batchEnd}],end);
  let arena=meshTargetArena();
  atomicStore(&meshState[${W.checkpoint}],atomicLoad(&meshState[meshCursorWord(arena)]));
  let jobs=(end-begin)*meshJobsPerBrick();
  meshStoreDispatch(${W.extractDispatch}u,(jobs+63u)/64u);
  meshStoreDispatch(${W.allocateDispatch}u,(end-begin+63u)/64u);
}
fn meshBatchJob(job:u32)->vec2u{
  let begin=atomicLoad(&meshState[${W.batchBegin}]);
  let entry=begin+job/meshJobsPerBrick();
  if(entry>=atomicLoad(&meshState[${W.batchEnd}])){return vec2u(MESH_NO_ENTRY,0u);}
  return vec2u(entry,job%meshJobsPerBrick());
}
// Count phase: size every brick of the batch.
@compute @workgroup_size(64)
fn surfaceMeshCount(@builtin(global_invocation_id) id:vec3u){
  let job=meshBatchJob(id.x+id.y*65535u*64u);
  if(job.x==MESH_NO_ENTRY){return;}
  let leaf=atomicLoad(&meshWork[meshListAt(job.x)]);
  if(!meshLeafCurrent(leaf)){return;}
  meshPhase=0u;meshEntry=job.x;
  meshExtractJob(leaf,job.y);
}
// Allocate phase, one thread per batch brick: free the brick's old range in
// the drawn arena, reserve its new range in the target arena and record it.
@compute @workgroup_size(64)
fn surfaceMeshAllocate(@builtin(global_invocation_id) id:vec3u){
  let entry=atomicLoad(&meshState[${W.batchBegin}])+id.x+id.y*65535u*64u;
  if(entry>=atomicLoad(&meshState[${W.batchEnd}])){return;}
  let leaf=atomicLoad(&meshWork[meshListAt(entry)]);
  let scratch=meshScratchAt(entry);
  let pending=atomicExchange(&meshWork[scratch],0u);
  atomicStore(&meshWork[scratch+1u],0u);
  let table=meshTableAt(leaf);
  let mode=atomicLoad(&meshState[${W.mode}]);
  // The previous range is remembered for the emit pass, which frees it, and
  // for an overflow rollback, which gives it back untouched.
  atomicStore(&meshWork[scratch+2u],atomicLoad(&meshWork[table]));
  atomicStore(&meshWork[scratch+3u],atomicLoad(&meshWork[table+1u]));
  if(mode==${M.incremental}u){
    let oldCount=atomicLoad(&meshWork[table+1u])&MESH_COUNT_MASK;
    atomicSub(&meshState[${W.liveQuads}],min(oldCount,atomicLoad(&meshState[${W.liveQuads}])));
  }
  var keyX=0u;var keyY=0u;var depth=0u;
  if(meshLeafCurrent(leaf)){let node=svoNodeLoad(svoLeafLoad(leaf).topology.x);keyX=node.address.x;keyY=node.address.y;depth=node.address.z;}
  let arena=meshTargetArena();
  let base=atomicAdd(&meshState[meshCursorWord(arena)],pending);
  if(base+pending>meshArenaLength(arena)){atomicOr(&meshState[${W.errorFlags}],1u);}
  atomicStore(&meshWork[table],base);atomicStore(&meshWork[table+1u],(pending&MESH_COUNT_MASK)|(depth<<27u));
  atomicStore(&meshWork[table+2u],keyX);atomicStore(&meshWork[table+3u],keyY);
  atomicAdd(&meshState[${W.batchPending}],pending);
  if(mode==${M.incremental}u){atomicAdd(&meshState[${W.liveQuads}],pending);}
  else{atomicAdd(&meshState[${W.liveTarget}],pending);}
}
// Emit phase: the same traversal as the count phase, writing into the
// brick's reserved range. It runs only once the whole batch has its ranges,
// so it is also where a brick's previous range is freed: an overflowing
// batch never reaches it, and the drawn mesh keeps every brick until its
// replacement is written in the same presentation.
@compute @workgroup_size(64)
fn surfaceMeshEmit(@builtin(global_invocation_id) id:vec3u){
  if((atomicLoad(&meshState[${W.errorFlags}])&1u)!=0u){return;}
  let job=meshBatchJob(id.x+id.y*65535u*64u);
  if(job.x==MESH_NO_ENTRY){return;}
  let leaf=atomicLoad(&meshWork[meshListAt(job.x)]);
  if(job.y==0u&&atomicLoad(&meshState[${W.mode}])==${M.incremental}u){
    let scratch=meshScratchAt(job.x);
    let oldBase=atomicLoad(&meshWork[scratch+2u]);
    let oldCount=atomicLoad(&meshWork[scratch+3u])&MESH_COUNT_MASK;
    let front=meshFront();let length=meshArenaLength(front);
    let dead=SurfaceQuad(vec3u(0u),0u,vec3u(0u),0u);
    for(var i=0u;i<oldCount;i+=1u){if(oldBase+i<length){meshArenaStore(front,oldBase+i,dead);}}
  }
  if(!meshLeafCurrent(leaf)){return;}
  let table=meshTableAt(leaf);
  meshPhase=1u;meshEntry=job.x;meshEmitArena=meshTargetArena();
  meshEmitBase=atomicLoad(&meshWork[table]);meshEmitCount=atomicLoad(&meshWork[table+1u])&MESH_COUNT_MASK;
  meshExtractJob(leaf,job.y);
}
@compute @workgroup_size(1)
fn surfaceMeshPublish(){
  let building=atomicLoad(&meshState[${W.building}])!=0u;
  let scheduled=atomicLoad(&meshState[${W.batchEnd}])>atomicLoad(&meshState[${W.batchBegin}]);
  let mode=atomicLoad(&meshState[${W.mode}]);
  let flags=atomicLoad(&meshState[${W.flags}]);
  let waiting=(mode==${M.replacement}u&&(flags&${F.started}u)==0u)||(flags&${F.markPending}u)!=0u
    ||((atomicLoad(&meshState[${W.errorFlags}])&1u)!=0u&&!scheduled);
  if(building&&!waiting&&(scheduled||atomicLoad(&meshState[${W.worklistCount}])==0u)){
    let arena=meshTargetArena();
    if((atomicLoad(&meshState[${W.errorFlags}])&1u)!=0u){
      // An overflowing batch is rolled back whole to its checkpoint; every
      // brick before it keeps its range and every brick in it gets its
      // previous range back. Retry after the arena grows.
      atomicStore(&meshState[meshCursorWord(arena)],atomicLoad(&meshState[${W.checkpoint}]));
      atomicStore(&meshState[${W.overflowLength}],meshArenaLength(arena));
      let batch=atomicLoad(&meshState[${W.batchPending}]);
      if(mode==${M.incremental}u){
        atomicSub(&meshState[${W.liveQuads}],min(batch,atomicLoad(&meshState[${W.liveQuads}])));
        for(var entry=atomicLoad(&meshState[${W.batchBegin}]);entry<atomicLoad(&meshState[${W.batchEnd}]);entry+=1u){
          let leaf=atomicLoad(&meshWork[meshListAt(entry)]);let scratch=meshScratchAt(entry);let table=meshTableAt(leaf);
          let previousCount=atomicLoad(&meshWork[scratch+3u]);
          atomicStore(&meshWork[table],atomicLoad(&meshWork[scratch+2u]));atomicStore(&meshWork[table+1u],previousCount);
          atomicAdd(&meshState[${W.liveQuads}],previousCount&MESH_COUNT_MASK);
        }
      }
      else{atomicSub(&meshState[${W.liveTarget}],min(batch,atomicLoad(&meshState[${W.liveTarget}])));}
    }else{
      let processed=atomicLoad(&meshState[${W.batchEnd}]);
      atomicStore(&meshState[${W.processedBricks}],processed);
      if(processed>=min(atomicLoad(&meshState[${W.worklistCount}]),meshLeafCapacity())){
        if(mode==${M.replacement}u){
          let front=meshFront();
          atomicStore(&meshState[${W.front}],1u-front);
          atomicStore(&meshState[${W.frontCursor}],atomicLoad(&meshState[${W.backCursor}]));
          atomicStore(&meshState[${W.backCursor}],0u);
          atomicStore(&meshState[${W.backGenerationConsumed}],atomicLoad(&meshState[${W.hostBackGeneration}]));
          atomicAnd(&meshState[${W.flags}],~${F.started}u);
        }
        if(mode!=${M.incremental}u){atomicStore(&meshState[${W.liveQuads}],atomicLoad(&meshState[${W.liveTarget}]));}
        atomicStore(&meshState[${W.usable}],1u);
        atomicStore(&meshState[${W.building}],0u);atomicStore(&meshState[${W.mode}],0u);
        atomicStore(&meshState[${W.boxCount}],0u);
      }
    }
  }
  let usable=atomicLoad(&meshState[${W.usable}])!=0u&&atomicLoad(&meshState[${W.withheld}])==0u&&(atomicLoad(&meshState[${W.errorFlags}])&2u)==0u;
  let quads=select(0u,atomicLoad(&meshState[${W.frontCursor}]),usable);
  atomicStore(&meshState[${W.drawInstanceCount}],${culling ? "0u" : "quads"});
  meshStoreDispatch(${W.extractDispatch}u,(quads+63u)/64u);
}
// Which of a brick's levels this camera draws. Selection has one writer per
// leaf, before culling, so all quads use the same hysteresis history.
fn meshCellFootprint(base:vec3u,brickLattice:u32)->f32{
  let camera=dryRasterPrimaryCamera();
  let half=vec3f(f32(brickLattice))*dry.mapping.cellSize*0.5;
  let centre=dry.mapping.worldOrigin+vec3f(base)*dry.mapping.cellSize+half-camera[0];
  let radius=length(half);let distanceSquared=dot(centre,centre);
  if(distanceSquared<=radius*radius){return 1e30;}
  return uniforms.viewport.y/cameraTanHalfFov()*radius/sqrt(distanceSquared-radius*radius)/f32(dry.mapping.brickSize);
}
fn meshSelectLevel(footprint:f32,threshold:f32,maximum:u32,previous:u32,valid:bool,hysteresis:f32)->u32{
  if(threshold<=0.0||footprint>=1e29){return 0u;}
  var selected=min(previous,maximum);
  if(!valid){
    return min(maximum,u32(max(0.0,floor(log2(threshold/max(footprint,1e-6))))));
  }
  // Coarsen below the lower edge; refine above the upper edge. A large jump
  // can cross several levels in one frame, while small camera jitter cannot.
  while(selected<maximum&&footprint*f32(1u<<(selected+1u))<=threshold*(1.0-hysteresis)){selected+=1u;}
  while(selected>0u&&footprint*f32(1u<<selected)>threshold*(1.0+hysteresis)){selected-=1u;}
  return selected;
}
@compute @workgroup_size(64)
fn surfaceMeshSelect(@builtin(global_invocation_id) id:vec3u){
  let leaf=id.x+id.y*65535u*64u;
  if(leaf>=meshLeafCapacity()){return;}
  let table=meshTableAt(leaf);
  if(!meshLeafCurrent(leaf)){atomicStore(&meshWork[table+5u],0u);return;}
  let box=meshLeafBox(leaf);let threshold=dryMeshLodPixels();
  let maximum=min(meshLevelCount()-1u,u32(dry.meshFilter.z));
  let valid=atomicLoad(&meshWork[table+5u])!=0u
    &&atomicLoad(&meshWork[table+6u])==bitcast<u32>(threshold)
    &&atomicLoad(&meshWork[table+7u])==maximum;
  let level=meshSelectLevel(meshCellFootprint(box.minimum,box.maximum.x-box.minimum.x),threshold,maximum,
    atomicLoad(&meshWork[table+4u]),valid,dry.meshFilter.w);
  atomicStore(&meshWork[table+4u],level);atomicStore(&meshWork[table+5u],1u);
  atomicStore(&meshWork[table+6u],bitcast<u32>(threshold));atomicStore(&meshWork[table+7u],maximum);
  if((atomicLoad(&meshWork[table+1u])&MESH_COUNT_MASK)>0u){atomicAdd(&meshState[${W.lodBricks}u+level],1u);}
}
fn meshQuadBrickBase(quad:SurfaceQuad)->vec3u{
  let axis=meshQuadFace(quad.face)/2u;
  let size=dry.mapping.brickSize*(1u<<(dry.mapping.maximumDepth-meshQuadDepth(quad.face)));
  var base=(quad.origin/size)*size;
  if(!meshTriangle(quad)&&(meshQuadFace(quad.face)&1u)!=0u&&quad.origin[axis]%size==0u){base[axis]-=size;}
  return base;
}
fn meshQuadLeaf(quad:SurfaceQuad)->u32{
  let depth=meshQuadDepth(quad.face);let base=meshQuadBrickBase(quad);
  var index=0u;
  for(var level=0u;level<depth;level+=1u){
    let node=svoNodeLoad(index);let shift=dry.mapping.maximumDepth-level-1u;
    let coordinate=(base/dry.mapping.brickSize)>>vec3u(shift);
    let octant=(coordinate.x&1u)|((coordinate.y&1u)<<1u)|((coordinate.z&1u)<<2u);
    let mask=node.address.w&255u;let bit=1u<<octant;
    if((mask&bit)==0u){return MESH_NO_ENTRY;}
    index=node.links.x+countOneBits(mask&(bit-1u));
    if(index>=svoControlLoad(0u)){return MESH_NO_ENTRY;}
  }
  return svoNodeLoad(index).links.z;
}
fn meshQuadSelected(quad:SurfaceQuad)->bool{
  let level=meshQuadLevel(quad.face);
  if(dryMeshLodPixels()<=0.0){return level==0u;}
  let leaf=meshQuadLeaf(quad);
  if(leaf==MESH_NO_ENTRY){return level==0u;}
  let at=MESH_TABLE_OFFSET+leaf*${SVO_SURFACE_MESH_TABLE_WORDS}u+4u;
  return level==atomicLoad(&meshWork[at]);
}
${culling ? "" : `// Vertex-only reader: the compute culler uses the writable table binding.
fn meshQuadSelectedRead(quad:SurfaceQuad)->bool{
  let level=meshQuadLevel(quad.face);
  if(dryMeshLodPixels()<=0.0){return level==0u;}
  let leaf=meshQuadLeaf(quad);
  if(leaf==MESH_NO_ENTRY){return level==0u;}
  let at=MESH_TABLE_OFFSET+leaf*${SVO_SURFACE_MESH_TABLE_WORDS}u+4u;
  return level==meshWorkRead[at];
}`}
// Cull exact cached quad bounds before invoking any vertex shader. No
// occlusion or screen-size approximation: subpixel geometry is retained.
// While a build is pending, quads inside its dirty boxes yield to the traced
// background so the edited region is exact before its bricks are re-extracted.
fn meshQuadVisible(index:u32)->bool{
  if(index>=atomicLoad(&meshState[${W.frontCursor}])){return false;}
  let quad=meshArenaLoad(meshFront(),index);
  if(meshQuadDead(quad)){return false;}
  if(!meshQuadSelected(quad)){return false;}
  if(meshMaskActive()&&meshBoxesContain(quad.origin,quad.origin+meshQuadExtent(quad))){return false;}
  let face=meshQuadFace(quad.face);let axis=face/2u;
  let origin=dry.mapping.worldOrigin+vec3f(quad.origin)*dry.mapping.cellSize;
  let camera=dryRasterPrimaryCamera();
  var facing=(camera[0][axis]-origin[axis])*select(-1.0,1.0,(face&1u)!=0u);
  ${contours ? `if(meshTriangle(quad)){
    let a=meshTrianglePoint(quad,quad.extent.x);let b=meshTrianglePoint(quad,quad.extent.y);let c=meshTrianglePoint(quad,quad.extent.z);
    let n=cross((b-a)*dry.mapping.cellSize,(c-a)*dry.mapping.cellSize);
    let point=origin+a*vec3f(meshQuadExtent(quad))*dry.mapping.cellSize;
    facing=dot(camera[0]-point,n)/max(length(n),1e-20);
  }` : ""}
  // Retain the coplanar tolerance band to avoid rounding-dependent holes.
  let epsilon=max(1e-5,abs(origin[axis])*1e-6);
  if(facing < -epsilon){return false;}
  let baseHalf=vec3f(meshQuadExtent(quad))*dry.mapping.cellSize*0.5;
  let center=origin+baseHalf-camera[0];
  let halfExtent=baseHalf*select(1.0,1.0+2.0*meshRecordInflation(quad),meshTriangle(quad));
  let z=dot(center,camera[1]);
  if(z+dot(halfExtent,abs(camera[1])) < DRY_REVERSED_Z_NEAR_M-epsilon){return false;}
  let ty=cameraTanHalfFov();let tx=ty*uniforms.viewport.x/max(uniforms.viewport.y,1.0);
  let planes=array<vec3f,4>(camera[2]-tx*camera[1],-camera[2]-tx*camera[1],
    camera[3]-ty*camera[1],-camera[3]-ty*camera[1]);
  for(var i=0u;i<4u;i+=1u){
    if(dot(center,planes[i])>dot(halfExtent,abs(planes[i]))+epsilon){return false;}
  }
  return true;
}
var<workgroup> meshCullCount:atomic<u32>;
var<workgroup> meshCullBase:u32;
@compute @workgroup_size(64)
fn surfaceMeshCull(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32){
  if(lane==0u){atomicStore(&meshCullCount,0u);}
  workgroupBarrier();
  let index=id.x+id.y*65535u*64u;
  let visible=meshQuadVisible(index);
  var rank=0u;if(visible){rank=atomicAdd(&meshCullCount,1u);}
  workgroupBarrier();
  if(lane==0u){
    let count=atomicLoad(&meshCullCount);
    meshCullBase=0u;if(count!=0u){meshCullBase=atomicAdd(&meshState[${W.drawInstanceCount}],count);}
  }
  workgroupBarrier();
  if(visible){meshVisibleOutput[meshCullBase+rank]=index;}
}
struct MeshVertexOut {
  @builtin(position) position:vec4f,
  @location(0) world:vec3f,
  @location(1) @interpolate(flat) identity:u32,
  @location(2) @interpolate(flat) normal:vec3f,
  @location(3) @interpolate(flat) level:u32,
  @location(4) @interpolate(flat) agreement:f32,
  @location(5) @interpolate(flat) cellPixels:f32,
}
@vertex fn surfaceMeshVertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->MeshVertexOut{
  let quad=meshDrawnQuad(${culling ? "meshVisible[instance]" : "instance"});
  let face=meshQuadFace(quad.face);let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  let corners=array<vec2u,4>(vec2u(1u,0u),vec2u(1u,1u),vec2u(0u,0u),vec2u(0u,1u));
  // Swapping the two in-plane coordinates reverses winding on negative
  // faces while preserving the 00–11 diagonal and the covered rectangle.
  let corner=select(corners[vertex],corners[vertex].yx,(face&1u)==0u);var lattice=quad.origin;lattice[u]+=corner.x*quad.extent[u];lattice[v]+=corner.y*quad.extent[v];
  var world=dry.mapping.worldOrigin+vec3f(lattice)*dry.mapping.cellSize;
  ${contours ? `if(meshTriangle(quad)){
    let p=meshTrianglePoint(quad,quad.extent[min(vertex,2u)]);
    world=dry.mapping.worldOrigin+(vec3f(quad.origin)+p*vec3f(meshQuadExtent(quad)))*dry.mapping.cellSize;
  }` : ""}
  let camera=dryRasterPrimaryCamera();let relative=world-camera[0];let z=dot(relative,camera[1]);
  var position=vec4f(dot(relative,camera[2])/(cameraTanHalfFov()*uniforms.viewport.x/max(uniforms.viewport.y,1.0)),dot(relative,camera[3])/cameraTanHalfFov(),DRY_REVERSED_Z_NEAR_M,z);
  ${culling ? "" : "// Without the cull pass, an unselected or freed quad collapses to a zero-area strip.\n  if(meshQuadDead(quad)||!meshQuadSelectedRead(quad)){position=vec4f(0.0,0.0,0.0,1.0);}"}
  var normal=vec3f(0.0);normal[axis]=select(-1.0,1.0,(face&1u)!=0u);
  ${contours ? `if(meshTriangle(quad)){
    let a=meshTrianglePoint(quad,quad.extent.x);let b=meshTrianglePoint(quad,quad.extent.y);let c=meshTrianglePoint(quad,quad.extent.z);
    normal=normalize(cross((b-a)*dry.mapping.cellSize,(c-a)*dry.mapping.cellSize));
  }` : ""}
  let brickLattice=dry.mapping.brickSize*(1u<<(dry.mapping.maximumDepth-meshQuadDepth(quad.face)));
  return MeshVertexOut(position,world,quad.identity,normal,meshQuadLevel(quad.face),f32((quad.face>>11u)&255u)/255.0,
    meshCellFootprint(meshQuadBrickBase(quad),brickLattice));
}
struct MeshSurfaceOut {
  @location(0) packedSurface:vec4u,@location(1) identityMedia:vec4u,
  @location(2) geometry:vec4f,@location(3) opaqueIdentity:vec2u,
}
fn meshFilteredNormal(face:vec3f,baked:vec3f,level:u32,agreement:f32,cellPixels:f32)->vec3f{
  if(dry.meshFilter.x<=0.5||dry.meshFilterNormals.x<=0.5||dry.meshFilter.y<=0.0
    ||dot(baked,face)<0.0||(level>0u&&agreement<dry.meshFilterNormals.y)){return face;}
  var strength=dry.meshFilter.y;
  if(level==0u&&dry.meshFilterNormals.z>0.5){
    let threshold=max(dryMeshLodPixels(),0.001);
    strength*=1.0-smoothstep(threshold,threshold*2.0,cellPixels);
  }
  return normalize(mix(face,baked,strength));
}
@fragment fn surfaceMeshFragment(input:MeshVertexOut)->MeshSurfaceOut{
  let camera=dryRasterPrimaryCamera();let rd=normalize(input.world-camera[0]);let t=length(input.world-camera[0]);
  var normal=input.normal;
  if(!meshContoursEnabled()&&dry.meshFilter.x>0.5&&dry.meshFilterNormals.x>0.5&&dry.meshFilter.y>0.0){
    // Filtered detail shades the baked normal: a coarse quad carries its cells'
    // mean, an exact quad reads the voxel just behind its face. The surface
    // itself stays the quad, so depth is unchanged; a baked normal facing away
    // from the face keeps the face.
    var word=input.identity;
    if(input.level==0u){
      let lattice=(input.world-dry.mapping.worldOrigin)/dry.mapping.cellSize-input.normal*0.25;
      word=meshRegionAt(lattice).identity;
    }
    // Air's zero word is not the absent-normal sentinel, so solidity gates the read.
    if(sceneIdentitySolid(word)&&sceneIdentityHasNormal(word)){
      normal=meshFilteredNormal(input.normal,sceneIdentityNormal(word),input.level,input.agreement,input.cellPixels);
    }
  }
  let shading=dryShadingNormal(input.identity,normal);
  let hit=DryHit(t,shading.normal,sceneIdentityMaterial(input.identity),DRY_OWNER_NONE,shading.featureId,DRY_GBUFFER_FIELD_VOXEL,DRY_GBUFFER_MOTION_STATIC,0u,0.0,vec3u(0u));
  // The quad is the surface; the baked normal is only how it is lit. Publishing
  // the face as the geometric normal is what lets the deferred lighting start
  // its shadow and contact rays outside this cube: biased along a baked normal
  // that leans toward the light, a silhouette's side face fires its shadow ray
  // back through its own solid and reads as black.
  var out=dryRasterPrimaryFacedSurface(hit,input.normal,camera[0],rd,camera[1],SVO_GBUFFER_PRODUCER_BRICK);
  // Raster material bits 27..30 carry level+1 for the read-only LOD overlay.
  // Production mesh shading masks these bits in drySplitIdentityAt.
  out.opaqueIdentity.x|=(input.level+1u)<<27u;
  return MeshSurfaceOut(out.packedSurface,out.identityMedia,out.geometry,out.opaqueIdentity);
}
// Whether a camera ray crosses any dirty box of the pending build, in world
// metres. Pixels that do are traced; the rest keep the cached mesh.
fn meshRayTouchesBox(slot:u32,origin:vec3f,inverse:vec3f)->bool{
  let base=slot*MESH_BOX_WORDS;
  let minimum=vec3u(meshWorkRead[base],meshWorkRead[base+1u],meshWorkRead[base+2u]);
  let maximum=vec3u(meshWorkRead[base+4u],meshWorkRead[base+5u],meshWorkRead[base+6u]);
  if(any(minimum>maximum)){return false;}
  let low=dry.mapping.worldOrigin+vec3f(minimum)*dry.mapping.cellSize-origin;
  let high=dry.mapping.worldOrigin+vec3f(maximum)*dry.mapping.cellSize-origin;
  let t0=min(low*inverse,high*inverse);let t1=max(low*inverse,high*inverse);
  let enter=max(max(t0.x,t0.y),t0.z);let exit=min(min(t1.x,t1.y),t1.z);
  return exit>=max(enter,0.0);
}
fn meshRayCrossesDirtyBox(origin:vec3f,direction:vec3f)->bool{
  let inverse=1.0/select(direction,vec3f(1e-9),abs(direction)<vec3f(1e-9));
  if(!meshRayTouchesBox(MESH_BOX_UNION,origin,inverse)){return false;}
  let count=min(meshHeader[${W.boxCount}],MESH_BOX_INDIVIDUAL);
  for(var i=0u;i<count;i+=1u){if(meshRayTouchesBox(i,origin,inverse)){return true;}}
  return false;
}
@fragment fn surfaceMeshBackground(input:VertexOut)->DryRasterPrimaryOut{
  dryRasterPrimaryReset();let camera=dryRasterPrimaryCamera();let rd=dryRasterPrimaryRay(input.position.xy,camera);
  var hit=missHit();
  // Exact planes are independent of the cached voxel mesh. Without a drawable
  // mesh, traverse the current published SVO so edits appear immediately and
  // unaffected room geometry remains visible. While a build updates a drawn
  // mesh, only the pixels whose rays cross its dirty boxes are traced; the
  // cull withholds the cached quads inside those boxes. This uses the same
  // voxel authority, never the preceding mesh or an analytic approximation.
  hit=dryPlanarCatalogHit(camera[0],rd,0.0,DRY_MISS);
  var producer=SVO_GBUFFER_PRODUCER_RASTER_BACKGROUND;
  let undrawn=meshHeader[${W.usable}]==0u||meshHeader[${W.withheld}]!=0u||(meshHeader[${W.errorFlags}]&2u)!=0u;
  let masked=!undrawn&&meshHeader[${W.building}]!=0u&&meshHeader[${W.boxCount}]!=0u&&meshRayCrossesDirtyBox(camera[0],rd);
  if(undrawn||masked){
    let current=traceStatic(camera[0],rd);
    if(current.t<hit.t){hit=current;producer=SVO_GBUFFER_PRODUCER_BRICK;}
  }
  if(hit.t>=DRY_MISS){return dryRasterPrimaryMiss();}
  return dryRasterPrimarySurface(hit,camera[0],rd,camera[1],producer);
}
`;
}
