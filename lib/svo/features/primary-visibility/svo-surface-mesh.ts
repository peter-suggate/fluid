import { SVO_MATERIAL_FLAGS } from "../../contracts/svo-material-flags";
import { svoMeshPlanarRasterWGSL } from "./svo-mesh-planar-raster";
import { svoAuthoredSurfaceNormalsWGSL } from "./svo-authored-surface-normals";
import { proceduralSolidRenderWGSL } from "../../../core/procedural-solid-render.wgsl";
import { svoDualMarchingCubesMeshWGSL, svoVoxelMeshWGSL, svoContourMeshWGSL, svoDualContouringMeshWGSL } from "../meshing/plugins";
import { svoCellContourWGSL } from "../construction/svo-cell-contour";
import { SPARSE_SCENE_DIRTY_JOURNAL_RECORD_WORD, SPARSE_SCENE_MAINTENANCE_STATE_WORDS } from "../../../core/webgpu-sparse-scene-proxies";
import type { WorkProgress } from "../../../core/work-progress";
import { SVO_SCREEN_SPACE_TERMINATION_CONTRACT } from "../lighting-visibility/svo-screen-space-termination";

/**
 * Cached opaque voxel boundary quads. All coordinates are on the accepted
 * finest-cell lattice, so adjacent bricks produce identical shared vertices.
 * The GPU publication header is also the indirect draw buffer.
 *
 * Every brick owns one contiguous quad range in a single arena, and a
 * per-leaf table records that range with the brick's address key. The mesh
 * reads the voxelizer's dirty journal with its own cursor: every leaf the
 * voxelizer finalizes is re-extracted, together with the bricks that touch
 * it, however many revisions or chunk frames produced those entries and
 * whether or not an earlier build is still running. A brick whose new range
 * has the size class of its old one is rewritten in place; otherwise the old
 * range goes to its class's free stack and the new one is popped from its
 * own, so the arena reaches a steady state under edits instead of filling
 * with holes. Only a first build, a meshing style change or a lapped journal
 * extracts the whole scene, in place, with nothing drawn meanwhile. While a
 * build is pending, quads inside the boxes of the finalized bricks it still
 * owes are culled and the pixels whose rays cross those boxes are traced, so
 * an edit is exact the frame after its voxels land.
 */
export interface SvoSurfaceMeshStatus {
  state: "pending" | "ready" | "blocked";
  quads?: number;
  /** Surface-bearing resident bricks selected at each level, before frustum culling. */
  lodBricks?: readonly number[];
  detail?: string;
  requiredQuads?: number;
  /** Quads currently owned by live bricks in the arena. */
  liveQuads?: number;
  capacityQuads?: number;
  allocatedBytes?: number;
  maximumBytes?: number;
  /** Builds started since the builder was reset, of either kind. */
  builds?: number;
  requirementComplete?: boolean;
  completedBricks?: number;
  totalBricks?: number;
  /** The arena holds a complete mesh: a pending build updates it in place. */
  drawn?: boolean;
  buildPhase?: "extracting" | "capacity" | "complete";
  /** Kind of the build now running; absent while idle. */
  buildKind?: "incremental" | "initial";
  restartReason?: "initial" | "topology" | "geometry" | "publication" | "journal";
  /** Kind of the build that last completed. */
  lastBuildKind?: "incremental" | "initial";
  /** Bricks the build that last completed extracted: its dirty bricks and the bricks touching them. */
  lastBuildBricks?: number;
  /** Builds completed by re-extracting journaled bricks into the drawn arena. */
  incrementalBuilds?: number;
  /** Builds completed by extracting every brick with nothing drawn. */
  wholeBuilds?: number;
  /** Bricks extracted by every build so far, modulo 2^32. */
  rebuiltBricks?: number;
  /** Journal entries the mesh has consumed, modulo 2^32. */
  journalCursor?: number;
  /** Arena buffers the host has created for this pipeline, the first included. */
  arenaAllocations?: number;
  /** Times the host has rebuilt the mesh bind groups. */
  arenaRebinds?: number;
  /** The arena is at its final size: an overflow from here is a failure, not a growth. */
  reserveFrozen?: boolean;
  fallbackReason?: "budget" | "smooth" | "inside-solid" | "publication" | "extraction";
}

/**
 * Construction actually failed: the arena cannot hold the mesh, or extraction
 * exceeded its subdivision bound. Smooth reconstruction and an inside-solid
 * camera withhold the mesh by policy; traversal is then the intended primary,
 * and live SVO startup must settle rather than halt the simulation.
 */
export function surfaceMeshConstructionFailed(
  status?: Pick<SvoSurfaceMeshStatus, "state" | "fallbackReason">,
): boolean {
  return status?.state === "blocked"
    && (status.fallbackReason === "budget" || status.fallbackReason === "extraction");
}

/**
 * A live first-frame fence can settle. Smooth surfaces and an inside-solid
 * camera keep SVO traversal visible on purpose; waiting for publication is
 * still pending, not ready.
 */
export function surfaceMeshAllowsLiveStartup(
  status?: Pick<SvoSurfaceMeshStatus, "state" | "fallbackReason">,
): boolean {
  if (!status || status.state === "ready") return true;
  return status.state === "blocked"
    && (status.fallbackReason === "smooth" || status.fallbackReason === "inside-solid");
}

/** Mesh-specific meaning stays beside the producer; the panel only renders facts. */
export function surfaceMeshProgress(status?: SvoSurfaceMeshStatus): WorkProgress {
  if (!status) return { label: "Waiting for mesh publication", state: "waiting", detail: "Mesh counters have not arrived from the GPU." };
  const state = status.state === "ready" ? "complete" : status.state === "blocked" ? "waiting" : "active";
  const capacity = status.buildPhase === "capacity";
  const reason = status.restartReason === "topology" ? "Topology changed; re-extracting the affected bricks."
    : status.restartReason === "geometry" ? "Geometry changed; re-extracting the affected bricks."
    : status.restartReason === "publication" ? "Source publication changed; rebuilding the current scene."
    : status.restartReason === "journal" ? "More bricks changed than the dirty journal holds; rebuilding the current scene." : undefined;
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
/** First build grows on GPU overflow and earns its edit reserve on completion.
 * Small scenes need not allocate the historical 64 MiB arena up front. */
export const SVO_SURFACE_MESH_INITIAL_BYTES = 16 * 1024 * 1024;
export const SVO_SURFACE_MESH_QUAD_BYTES = 32;
export const SVO_SURFACE_MESH_HEADER_BYTES = 64;
/** First 64 bytes retain the public draw/diagnostic ABI; tail is builder state. */
export const SVO_SURFACE_MESH_STATE_BYTES = 320;
/** Historical arena size used to preserve the edit-reserve contract. */
export const SVO_SURFACE_MESH_MINIMUM_QUADS = SVO_SURFACE_MESH_BYTES / SVO_SURFACE_MESH_QUAD_BYTES;
/**
 * Quads kept free past a whole build's cursor for the edits that follow it:
 * half the mesh again, and never less than 16 MiB of quads.
 * The arena is sized to this once, when the whole build completes, and is
 * never resized by an edit.
 */
export function surfaceMeshReserveQuads(cursorQuads: number): number {
  return Math.max(Math.ceil(Math.max(0, cursorQuads) / 2), SVO_SURFACE_MESH_MINIMUM_QUADS / 4);
}
/** Dirty box slots; the last always holds the union of the others, so a build masks up to 63 boxes individually and a longer dirty set collapses to the union alone. */
export const SVO_SURFACE_MESH_BOX_CAPACITY = 64;
export const SVO_SURFACE_MESH_BOX_INDIVIDUAL_CAPACITY = SVO_SURFACE_MESH_BOX_CAPACITY - 1;
export const SVO_SURFACE_MESH_BOX_UNION_SLOT = SVO_SURFACE_MESH_BOX_CAPACITY - 1;
/** Words of one dirty box record: minimum xyz, pad, maximum xyz, pad. */
export const SVO_SURFACE_MESH_BOX_WORDS = 8;
/** Pending worklist entries the dirty box pass sweeps; past it the union grown at claim time stands for them all. */
export const SVO_SURFACE_MESH_BOX_SWEEP_ENTRIES = 4096;
/**
 * Words per leaf: base, count | (depth + 1) << 27 (zero while the slot holds
 * no brick), key x/y, selected LOD, history valid, previous threshold,
 * previous cap, worklist claim (0 unlisted, 1 listed as touching a dirty
 * brick, 2 listed as dirty).
 */
export const SVO_SURFACE_MESH_TABLE_WORDS = 9;
/** Words per worklist entry scratch: pending count, emitted count, previous base, previous count word. */
export const SVO_SURFACE_MESH_SCRATCH_WORDS = 4;
export const SVO_SURFACE_MESH_COUNT_MASK = 0x07ff_ffff;
/**
 * Range size classes with a free stack. Class 0 is the empty range; class
 * `c > 0` holds `(4 + (c - 1) % 4) << ((c - 1) >> 2)` quads, four classes an
 * octave from 4 quads up, so a range is at most a quarter larger than the
 * quads in it. A range's class is always the class of its brick's quad count,
 * which is why the table needs no capacity word.
 */
export const SVO_SURFACE_MESH_CLASS_COUNT = 64;
/** Free ranges one class's stack holds; a range freed past it is left as a hole. */
export const SVO_SURFACE_MESH_CLASS_STACK_CAPACITY = 1024;
/** Journal entries one presentation ingests: twice the voxelizer's default brick budget. */
export const SVO_SURFACE_MESH_INGEST_ENTRIES = 4096;
/** Workgroups of the ingest pass; its threads stride over (entry, direction) pairs. */
export const SVO_SURFACE_MESH_INGEST_GROUPS = 256;

/** Size class of a range holding `count` quads. */
export function surfaceMeshRangeClass(count: number): number {
  if (count <= 0) return 0;
  const quads = Math.max(count, 4);
  const octave = 31 - Math.clz32(quads);
  const step = (1 << octave) >>> 2;
  return (octave - 2) * 4 + Math.ceil((quads - (1 << octave)) / step) + 1;
}

/** Quads a range of `rangeClass` holds. */
export function surfaceMeshClassCapacity(rangeClass: number): number {
  if (rangeClass <= 0) return 0;
  return (4 + ((rangeClass - 1) & 3)) * 2 ** ((rangeClass - 1) >>> 2);
}

/**
 * Extraction jobs per brick, an upper bound over the meshing styles a shader
 * variant can run: dual marching cubes extracts one row of cells a job, the
 * voxel mesh one face plane a job plus one job per coarse level.
 */
export function surfaceMeshJobsPerBrick(brickSize: number, dualMarchingCubes: boolean): number {
  const size = Math.max(1, Math.floor(brickSize));
  return dualMarchingCubes ? size * size : 6 * size + 31 - Math.clz32(size);
}

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
  /** Quads allocated in the arena, free ranges included: the cull iterates this many. */
  frontCursor: 4,
  /** Bit 0: the arena overflowed this batch. Bit 1: extraction exceeded a bound. */
  errorFlags: 5,
  topologyRevision: 6,
  geometryRevision: 7,
  /** Arena quads the cull considers, stored by publication: the cursor while the mesh is drawable, else zero. Words 9 and 10 are spare. */
  cullQuads: 8,
  building: 11,
  builds: 12,
  /** The arena holds a complete mesh. */
  usable: 13,
  processedBricks: 14,
  /** 1: camera inside a solid voxel. 2: smooth reconstruction selected. */
  withheld: 15,
  checkpoint: 16,
  overflowLength: 17,
  /** Bricks appended to the worklist since the build began; the list is a ring over the leaf capacity. */
  worklistCount: 18,
  restartReason: 19,
  /** Journal entries consumed, modulo 2^32. */
  journalCursor: 20,
  /** Journal span the ingest pass of this presentation reads. */
  ingestBegin: 21,
  ingestEnd: 22,
  /** Bricks the build that last completed extracted. */
  lastBuildBricks: 23,
  incrementalBuilds: 24,
  wholeBuilds: 25,
  /** Always zero: there is one arena. Kept for the readers of the draw header. */
  front: 26,
  /** 0 idle, 1 incremental into the drawn arena, 2 whole with nothing drawn. */
  mode: 27,
  /** Mode of the build that last completed. */
  lastBuildMode: 28,
  liveQuads: 29,
  /** Bit 0 markAll, bit 2 scanned, bit 3 scheduled. */
  flags: 30,
  /** host: threads of the count and emit dispatches. */
  hostExtractThreads: 31,
  /** host: bricks one batch extracts. */
  bricksPerBatch: 32,
  /** The voxelizer's completed revision when the mesh last looked. */
  consumedMaintenanceRevision: 33,
  boxCount: 34,
  /** Bricks extracted by every build so far, modulo 2^32. */
  rebuiltBricks: 35,
  batchBegin: 36,
  batchEnd: 37,
  /** host: 1 when a maintenance arena with a dirty journal is bound. */
  hostMaintenance: 38,
  /** host: leaf slots the range table covers. */
  hostLeafCapacity: 39,
  /** host: word offset of the maintenance state block in the bound maintenance buffer. */
  hostMaintenanceStateWords: 43,
  /** host: word offset of the dirty brick records, whose journal words the mesh reads. */
  hostMaintenanceRecordsWords: 44,
  /** host: dirty brick records, and so journal entries, the maintenance buffer holds. */
  hostMaintenanceCapacity: 45,
  contourMode: 46,
  /** Word offset in low 31 bits; high bit tags function-graph (DMC) records. */
  hostSurfaceVertices: 47,
  lodBricks: 48,
  /**
   * Two boxes of six words (the complement of the minimum xyz, then the maximum
   * xyz, in finest cells, so zeroed words are an empty box) over the bricks
   * claimed for extraction, one per parity of the build that extracts them.
   * The shadow maps redraw inside them and empty the one whose build is over
   * (`svo-raster-ao.ts`).
   */
  shadowBoxes: 56,
  /** Boundary image offset followed by its world origin and cell size (f32). */
  hostSolidBoundary: 68,
  /** Renderer heightfield header and its live authored-edit page image. */
  hostRenderTerrain: 75,
  hostTerrainEdits: 76,
  /** Material at primitive zero; each lookup verifies the candidate identity. */
  hostPrimitiveMaterialBase: 77,
  wordCount: 80,
} as const);

/**
 * `markAll`: the next leaf scan lists every current brick and clears the rest
 * of the table. `scanned`: that scan has run. `scheduled`: this
 * presentation's batch was scheduled, so publication may apply it.
 */
export const SVO_SURFACE_MESH_FLAGS = Object.freeze({ markAll: 1, scanned: 4, scheduled: 8 } as const);
export const SVO_SURFACE_MESH_MODE = Object.freeze({ idle: 0, incremental: 1, initial: 2 } as const);
export const SVO_SURFACE_MESH_RESTART_REASONS = Object.freeze(["publication", "initial", "topology", "geometry", "publication", "journal"] as const);

/** Words of the class free stacks: a count and `SVO_SURFACE_MESH_CLASS_STACK_CAPACITY` range bases each. */
export const SVO_SURFACE_MESH_WORK_STACK_OFFSET_WORDS = SVO_SURFACE_MESH_BOX_CAPACITY * SVO_SURFACE_MESH_BOX_WORDS;
export const SVO_SURFACE_MESH_STACK_WORDS = SVO_SURFACE_MESH_CLASS_STACK_CAPACITY + 1;
/** Words of the work buffer before the per-leaf table: the dirty box records and the class free stacks. */
export const SVO_SURFACE_MESH_WORK_TABLE_OFFSET_WORDS = SVO_SURFACE_MESH_WORK_STACK_OFFSET_WORDS
  + SVO_SURFACE_MESH_CLASS_COUNT * SVO_SURFACE_MESH_STACK_WORDS;
/** Work buffer size for `leafCapacity` slots: boxes, free stacks, range table, scratch, worklist. */
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
 * Ceiling while the drawn mesh stays visible: an incremental build shares the
 * frame with the rasterized scene, so it is paced to keep the frame
 * interactive rather than to finish soonest.
 */
export const SVO_SURFACE_MESH_BUILD_BRICKS_DRAWN_MAXIMUM = 2048;
/**
 * Ceiling while nothing is drawn: every presentation of a whole build also
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
  building: boolean;
  mode: number;
  /** The arena holds a complete mesh, whether or not it is withheld from the frame. */
  usable: boolean;
  /** The arena overflowed at this quad capacity and the build waits for a larger binding. */
  grow?: { overflowQuads: number };
}

/**
 * Interpret a state receipt. Pure so the host's decisions are testable
 * without a device: `arenaBytes` is the arena as bound when the receipt's
 * frame was encoded, and `growable` is whether the host may still enlarge it
 * (it may during a whole build and until that build's reserve is in place;
 * an edit never enlarges it).
 */
export function interpretSurfaceMeshState(words: ArrayLike<number>, context: { arenaBytes: number; maximumBytes: number; growable: boolean }): SvoSurfaceMeshReceipt {
  const W = SVO_SURFACE_MESH_STATE;
  const word = (index: number) => words[index] ?? 0;
  const mode = word(W.mode);
  const building = word(W.building) !== 0;
  const usable = word(W.usable) !== 0;
  const errors = word(W.errorFlags);
  const overflowed = building && (errors & 1) !== 0;
  const extractionFailed = (errors & 2) !== 0;
  const withheld = word(W.withheld);
  const allocatedBytes = context.arenaBytes;
  // A grow the host already answered is one the GPU has not yet observed.
  const grow = overflowed && allocatedBytes / SVO_SURFACE_MESH_QUAD_BYTES <= word(W.overflowLength);
  const canGrow = grow && context.growable && allocatedBytes < context.maximumBytes && !extractionFailed;
  const budgetBlocked = grow && !canGrow && !extractionFailed;
  const fallback = !usable || withheld !== 0 || extractionFailed || budgetBlocked;
  const reason: NonNullable<SvoSurfaceMeshStatus["fallbackReason"]> = withheld === 2 ? "smooth"
    : withheld === 1 ? "inside-solid" : extractionFailed ? "extraction" : budgetBlocked ? "budget" : "publication";
  const pending = building && !extractionFailed && !budgetBlocked;
  const completedBricks = word(W.processedBricks);
  const totalBricks = word(W.worklistCount);
  const kind = (value: number) => value === SVO_SURFACE_MESH_MODE.incremental ? "incremental" as const
    : value === SVO_SURFACE_MESH_MODE.initial ? "initial" as const : undefined;
  const status: SvoSurfaceMeshStatus = {
    state: pending ? "pending" : fallback ? "blocked" : "ready",
    lodBricks: Array.from({ length: 4 }, (_, level) => word(W.lodBricks + level)),
    quads: word(W.drawInstanceCount), requiredQuads: word(W.frontCursor), liveQuads: word(W.liveQuads),
    capacityQuads: allocatedBytes / SVO_SURFACE_MESH_QUAD_BYTES, allocatedBytes, maximumBytes: context.maximumBytes,
    builds: word(W.builds), requirementComplete: !building && !extractionFailed, completedBricks, totalBricks,
    drawn: usable && withheld === 0 && !extractionFailed,
    buildPhase: building ? (overflowed ? "capacity" : "extracting") : "complete",
    buildKind: building ? kind(mode) : undefined,
    restartReason: SVO_SURFACE_MESH_RESTART_REASONS[word(W.restartReason)] ?? "publication",
    lastBuildKind: kind(word(W.lastBuildMode)), lastBuildBricks: word(W.lastBuildBricks),
    incrementalBuilds: word(W.incrementalBuilds), wholeBuilds: word(W.wholeBuilds),
    rebuiltBricks: word(W.rebuiltBricks), journalCursor: word(W.journalCursor),
    reserveFrozen: !context.growable,
  };
  if (pending) {
    status.detail = overflowed ? "Mesh storage growing; completed bricks are retained."
      : usable ? `Updating mesh: ${completedBricks.toLocaleString()} / ${totalBricks.toLocaleString()} bricks re-extracted; the cached mesh stays drawn and edited regions are traced.`
        : `Building mesh: ${completedBricks.toLocaleString()} / ${totalBricks.toLocaleString()} bricks processed; current voxels remain visible through exact traversal.`;
  } else if (fallback) {
    status.fallbackReason = reason;
    status.detail = reason === "smooth" ? "Raster requires voxel-flat surfaces; current SVO traversal remains visible."
      : reason === "inside-solid" ? "Camera is inside a solid voxel; current SVO traversal remains visible."
      : reason === "budget" ? (context.growable
        ? "Surface mesh exceeds the allocation limit; current SVO traversal remains visible."
        : "An edit needs more surface mesh storage than was reserved when the scene was built; edited regions remain traced.")
      : reason === "extraction" ? "Surface extraction exceeded its subdivision limit; current SVO traversal remains visible."
      : "Waiting for a complete voxel publication; exact planes remain visible.";
  }
  const receipt: SvoSurfaceMeshReceipt = { status, building, mode, usable };
  if (canGrow) receipt.grow = { overflowQuads: word(W.overflowLength) };
  return receipt;
}

export function svoSurfaceMeshWGSL(group: number, flatNormalsFlag: number, culling = true, contours = false, retainSampledShadowCasters = false): string {
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
// A dual triangle (bit 30) carries two more bits per axis for each vertex in
// face bits 11-28, which a dual cell spanning leaves of two sizes needs.
${contours ? svoCellContourWGSL : ""}
fn meshContoursEnabled()->bool{return ${contours ? "dry.meshFilterNormals.w>0.5&&dry.meshFilterNormals.w<2.0" : "false"};}
// Inflation is baked into each triangle record, so a record is drawn the way
// it was extracted whatever the setting is now.
fn meshInflationCode()->u32{return ${contours ? "u32(round(clamp(dry.meshFilterNormals.w-1.0,0.0,0.5)*100.0))" : "0u"};}
fn meshRecordInflation(quad:SurfaceQuad)->f32{return f32((quad.face>>11u)&63u)/100.0;}
${contours ? `fn meshTrianglePoint(quad:SurfaceQuad,index:u32)->vec3f{
  let word=quad.extent[index];
  if((quad.face&0x40000000u)!=0u){
    let high=(quad.face>>(11u+6u*index))&63u;
    return vec3f(f32((word&1023u)|((high&3u)<<10u)),f32(((word>>10u)&1023u)|(((high>>2u)&3u)<<10u)),f32(((word>>20u)&1023u)|(((high>>4u)&3u)<<10u)))/512.0;
  }
  let inflation=meshRecordInflation(quad);
  return vec3f(0.5)+(contourUnpackPoint(word)-vec3f(0.5))*(1.0+2.0*inflation);
}` : ""}
fn meshDmcEnabled()->bool{return dry.meshFilterNormals.w>=3.0;}
// Shared dual-grid policy (uniform resolution, expanded dirty masks and normals).
fn meshDcEnabled()->bool{return dry.meshFilterNormals.w>=2.0;}
fn meshDualTriangle(quad:SurfaceQuad)->bool{return (quad.face&0x40000000u)!=0u;}
// A dual triangle's origin words carry its three vertex normals above bit 16.
fn meshQuadOrigin(quad:SurfaceQuad)->vec3u{return select(quad.origin,quad.origin&vec3u(0xffffu),meshDualTriangle(quad));}
fn meshTriangle(quad:SurfaceQuad)->bool{return (quad.face&0x80000000u)!=0u;}
fn meshQuadExtent(quad:SurfaceQuad)->vec3u{
  if(meshTriangle(quad)){return vec3u((1u<<(dry.mapping.maximumDepth-meshQuadDepth(quad.face)))*select(1u,2u,meshDualTriangle(quad)));}
  return quad.extent;
}
fn meshContourCode(voxel:u32)->u32{
  ${contours ? "if(meshContoursEnabled()){return drySceneContourOfVoxel(voxel);}" : ""}
  return 0u;
}
@group(${group}) @binding(30) var<storage,read_write> meshState:array<atomic<u32>>;
@group(${group}) @binding(32) var<storage,read_write> meshArena:array<SurfaceQuad>;
@group(${group}) @binding(34) var<storage,read_write> meshVisibleOutput:array<u32>;
// Dirty boxes, class free stacks, per-leaf quad ranges, per-entry scratch and the brick worklist.
@group(${group}) @binding(40) var<storage,read_write> meshWork:array<atomic<u32>>;
// The voxelizer's maintenance arena: its state block (completed revision and
// the dirty journal's write cursor) and the dirty brick records that carry
// the journal entries sit at the word offsets the host stamps into the state.
@group(${group}) @binding(41) var<storage,read> meshMaintenance:array<u32>;
@group(${group}) @binding(31) var<storage,read> meshQuads:array<SurfaceQuad>;
@group(${group}) @binding(33) var<storage,read> meshHeader:array<u32>;
@group(${group}) @binding(35) var<storage,read> meshVisible:array<u32>;
@group(${group}) @binding(42) var<storage,read> meshWorkRead:array<u32>;

const MESH_BOX_CAPACITY:u32=${SVO_SURFACE_MESH_BOX_CAPACITY}u;
const MESH_BOX_INDIVIDUAL:u32=${SVO_SURFACE_MESH_BOX_INDIVIDUAL_CAPACITY}u;
const MESH_BOX_UNION:u32=${SVO_SURFACE_MESH_BOX_UNION_SLOT}u;
const MESH_BOX_WORDS:u32=${SVO_SURFACE_MESH_BOX_WORDS}u;
const MESH_BOX_SWEEP:u32=${SVO_SURFACE_MESH_BOX_SWEEP_ENTRIES}u;
const MESH_STACK_OFFSET:u32=${SVO_SURFACE_MESH_WORK_STACK_OFFSET_WORDS}u;
const MESH_STACK_WORDS:u32=${SVO_SURFACE_MESH_STACK_WORDS}u;
const MESH_STACK_CAPACITY:u32=${SVO_SURFACE_MESH_CLASS_STACK_CAPACITY}u;
const MESH_CLASS_COUNT:u32=${SVO_SURFACE_MESH_CLASS_COUNT}u;
const MESH_TABLE_OFFSET:u32=${SVO_SURFACE_MESH_WORK_TABLE_OFFSET_WORDS}u;
const MESH_COUNT_MASK:u32=${SVO_SURFACE_MESH_COUNT_MASK}u;
const MESH_MAINTENANCE_COMPLETED:u32=${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.completedRevision}u;
const MESH_MAINTENANCE_JOURNAL_WRITE:u32=${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.journalWriteCursor}u;
const MESH_JOURNAL_RECORD_WORD:u32=${SPARSE_SCENE_DIRTY_JOURNAL_RECORD_WORD}u;
const MESH_INGEST_ENTRIES:u32=${SVO_SURFACE_MESH_INGEST_ENTRIES}u;
const MESH_INGEST_THREADS:u32=${SVO_SURFACE_MESH_INGEST_GROUPS * 64}u;
const MESH_CLAIM_WORD:u32=8u;
const MESH_CLAIM_TOUCHING:u32=1u;
const MESH_CLAIM_DIRTY:u32=2u;
const MESH_NO_ENTRY:u32=0xffffffffu;

fn meshQuadFace(word:u32)->u32{return word&7u;}
fn meshQuadLevel(word:u32)->u32{return (word>>3u)&7u;}
fn meshQuadDepth(word:u32)->u32{return (word>>6u)&31u;}
fn meshPackFace(face:u32,level:u32,depth:u32)->u32{return face|(level<<3u)|(depth<<6u);}
fn meshQuadDead(quad:SurfaceQuad)->bool{return all(quad.extent==vec3u(0u));}
// Levels a brick can offer: the voxel boundary plus one per halving of the
// brick edge down to a single cell.
fn meshLevelCount()->u32{return countTrailingZeros(max(dry.mapping.brickSize,1u))+1u;}
fn meshJobsPerBrick()->u32{${contours ? "if(meshDmcEnabled()){return dry.mapping.brickSize*dry.mapping.brickSize;}" : ""}return ${contours ? "6u*dry.mapping.brickSize+select(meshLevelCount()-1u,0u,meshContoursEnabled()||meshDcEnabled())" : "6u*dry.mapping.brickSize+meshLevelCount()-1u"};}
// Filtered-detail threshold in live pixels; zero is the exact voxel mesh.
// Authored at the screen-space contract's reference height so it stays angular.
fn dryMeshLodPixels()->f32{
  ${contours ? "if(meshContoursEnabled()||meshDcEnabled()){return 0.0;}" : ""}
  return select(0.0,max(dry.lod.w,0.0),dry.meshFilter.x>0.5)*uniforms.viewport.y/${SVO_SCREEN_SPACE_TERMINATION_CONTRACT.referenceViewportHeightPixels};
}
// Quads merge on material alone. A level-0 quad reads its voxel's baked normal
// per fragment, so its normal half is left absent; a coarse quad carries the
// mean baked normal of the voxels behind it that face the way it does.
fn meshMergeIdentity(identity:u32)->u32{return sceneIdentityMaterial(identity)|(SCENE_IDENTITY_NO_NORMAL<<16u);}

fn meshLeafCapacity()->u32{return atomicLoad(&meshState[${W.hostLeafCapacity}]);}
fn meshTableAt(leaf:u32)->u32{return MESH_TABLE_OFFSET+leaf*${SVO_SURFACE_MESH_TABLE_WORDS}u;}
// The worklist and its scratch are rings over the leaf capacity: an entry
// number counts appends since the build began.
fn meshEntrySlot(entry:u32)->u32{return entry%max(meshLeafCapacity(),1u);}
fn meshScratchAt(slot:u32)->u32{return MESH_TABLE_OFFSET+meshLeafCapacity()*${SVO_SURFACE_MESH_TABLE_WORDS}u+slot*${SVO_SURFACE_MESH_SCRATCH_WORDS}u;}
fn meshListAt(slot:u32)->u32{return MESH_TABLE_OFFSET+meshLeafCapacity()*${SVO_SURFACE_MESH_TABLE_WORDS + SVO_SURFACE_MESH_SCRATCH_WORDS}u+slot;}
fn meshStackAt(rangeClass:u32)->u32{return MESH_STACK_OFFSET+rangeClass*MESH_STACK_WORDS;}
fn meshDrawnQuad(index:u32)->SurfaceQuad{return meshQuads[index];}

// Size class of a range of \`count\` quads and the quads a class holds: four
// classes an octave from 4 quads up. A brick's range always has the class of
// its quad count, so equal classes mean the brick is rewritten where it is.
fn meshRangeClass(count:u32)->u32{
  if(count==0u){return 0u;}
  let quads=max(count,4u);
  let octave=31u-countLeadingZeros(quads);
  let step=(1u<<octave)>>2u;
  return (octave-2u)*4u+(quads-(1u<<octave)+step-1u)/step+1u;
}
fn meshClassCapacity(rangeClass:u32)->u32{
  if(rangeClass==0u){return 0u;}
  return (4u+((rangeClass-1u)&3u))<<((rangeClass-1u)>>2u);
}
// Free stacks are popped only by the allocate pass and pushed only by the
// emit pass, so each pass sees one kind of concurrent access: a pop that
// finds the stack empty, or a push that finds it full, undoes its own step.
fn meshPopRange(rangeClass:u32)->u32{
  if(rangeClass==0u||rangeClass>=MESH_CLASS_COUNT){return MESH_NO_ENTRY;}
  let stack=meshStackAt(rangeClass);
  let held=atomicSub(&meshWork[stack],1u);
  if(held==0u||held>MESH_STACK_CAPACITY){atomicAdd(&meshWork[stack],1u);return MESH_NO_ENTRY;}
  return atomicLoad(&meshWork[stack+held]);
}
fn meshPushRange(rangeClass:u32,base:u32){
  if(rangeClass==0u||rangeClass>=MESH_CLASS_COUNT){return;}
  let stack=meshStackAt(rangeClass);
  let held=atomicAdd(&meshWork[stack],1u);
  if(held>=MESH_STACK_CAPACITY){atomicSub(&meshWork[stack],1u);return;}
  atomicStore(&meshWork[stack+1u+held],base);
}

// A leaf slot is linked while a voxel terminal's node points back at it, and
// current while that brick's lifecycle is too. A linked brick that is not
// current is being repaired: the voxelizer journals it when it finishes.
fn meshLeafLinked(leaf:u32)->bool{
  if(leaf>=svoControlLoad(1u)){return false;}
  let topology=svoLeafLoad(leaf).topology;
  if(topology.x>=svoControlLoad(0u)||topology.z!=0u){return false;}
  return svoNodeLoad(topology.x).links.z==leaf;
}
fn meshLeafCurrent(leaf:u32)->bool{
  if(!meshLeafLinked(leaf)){return false;}
  return svoBrickLifecycleCurrent(svoBrickLifecycleDecode(svoNodeLoad(svoLeafLoad(leaf).topology.x).links.w));
}
struct MeshBrickBox { minimum:vec3u, maximum:vec3u }
fn meshAddressBox(keyX:u32,keyY:u32,depth:u32)->MeshBrickBox{
  let size=(1u<<(dry.mapping.maximumDepth-min(depth,dry.mapping.maximumDepth)))*dry.mapping.brickSize;
  let base=svoDecodeMorton(keyX,keyY,depth)*size;
  return MeshBrickBox(base,base+vec3u(size));
}
// Closed lattice box of a leaf's brick, so bricks that only share a face still overlap.
fn meshLeafBox(leaf:u32)->MeshBrickBox{
  let node=svoNodeLoad(svoLeafLoad(leaf).topology.x);
  return meshAddressBox(node.address.x,node.address.y,node.address.z);
}
// The box a dirty brick masks. Dual-grid faces depend on cells across both
// face and diagonal neighbours, so their mask reaches one cell past the brick.
fn meshMaskBox(box:MeshBrickBox)->MeshBrickBox{
  if(!meshDcEnabled()){return box;}
  return MeshBrickBox(box.minimum-min(box.minimum,vec3u(1u)),box.maximum+vec3u(1u));
}
fn meshBoxMinimum(box:u32)->vec3u{
  let base=box*MESH_BOX_WORDS;
  return vec3u(atomicLoad(&meshWork[base]),atomicLoad(&meshWork[base+1u]),atomicLoad(&meshWork[base+2u]));
}
fn meshBoxMaximum(box:u32)->vec3u{
  let base=box*MESH_BOX_WORDS+4u;
  return vec3u(atomicLoad(&meshWork[base]),atomicLoad(&meshWork[base+1u]),atomicLoad(&meshWork[base+2u]));
}
fn meshBoxStore(box:u32,minimum:vec3u,maximum:vec3u){
  let base=box*MESH_BOX_WORDS;
  atomicStore(&meshWork[base],minimum.x);atomicStore(&meshWork[base+1u],minimum.y);atomicStore(&meshWork[base+2u],minimum.z);
  atomicStore(&meshWork[base+4u],maximum.x);atomicStore(&meshWork[base+5u],maximum.y);atomicStore(&meshWork[base+6u],maximum.z);
}
fn meshMaskActive()->bool{return atomicLoad(&meshState[${W.boxCount}])!=0u;}
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

struct MeshRegion { origin:vec3f, size:f32, identity:u32, contour:u32, voxel:u32 }
// Lookup returns the containing cell OR empty octree region. Its extent lets
// a coarse boundary stop subdividing as soon as its neighbour is uniform.
fn meshRegionAt(p:vec3f)->MeshRegion {
  let rootSize=f32((1u<<dry.mapping.maximumDepth)*dry.mapping.brickSize);
  if(any(p<vec3f(0.0))||any(p>=vec3f(rootSize))){return MeshRegion(vec3f(-rootSize),rootSize*3.0,0u,0u,SVO_INVALID);}
  var origin=vec3f(0.0);var size=rootSize;var index=0u;
  for(var level=0u;level<=dry.mapping.maximumDepth;level+=1u){
    if(index>=svoControlLoad(0u)){return MeshRegion(origin,size,0u,0u,SVO_INVALID);}
    let node=svoNodeLoad(index);
    if(node.links.z!=SVO_INVALID){
      let leaf=svoLeafLoad(node.links.z).topology;
      if(leaf.x!=index||leaf.z!=0u){return MeshRegion(origin,size,0u,0u,SVO_INVALID);}
      if(!svoBrickLifecycleCurrent(svoBrickLifecycleDecode(node.links.w))){return MeshRegion(origin,size,0u,0u,SVO_INVALID);}
      let cellSize=size/f32(dry.mapping.brickSize);
      let local=vec3u(clamp(floor((p-origin)/cellSize),vec3f(0.0),vec3f(f32(dry.mapping.brickSize-1u))));
      let voxel=svoBrickVoxelIndex(leaf.y,local,dry.mapping.brickSize);
      var identity=0u;if(voxel<dryVoxelCapacity()){identity=sceneIdentityAt(voxel);}
      return MeshRegion(origin+vec3f(local)*cellSize,cellSize,identity,select(0u,meshContourCode(voxel),sceneIdentitySolid(identity)),voxel);
    }
    size*=0.5;let upper=p>=origin+vec3f(size);
    let octant=select(0u,1u,upper.x)|select(0u,2u,upper.y)|select(0u,4u,upper.z);
    origin+=select(vec3f(0.0),vec3f(size),upper);
    let mask=node.address.w&255u;let bit=1u<<octant;
    if((mask&bit)==0u){return MeshRegion(origin,size,0u,0u,SVO_INVALID);}
    index=node.links.x+countOneBits(mask&(bit-1u));
  }
  return MeshRegion(origin,size,0u,0u,SVO_INVALID);
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
var<private> meshEmitBase:u32;
var<private> meshEmitCount:u32;
fn meshAppend(origin:vec3u,extent:vec3u,packedFace:u32,identity:u32){
  // Native sampled surfaces replace these proxy faces at draw time. Reject in
  // BOTH count and emit phases, before reserving arena space or paying culling,
  // vertex and fragment work. Their voxel occupancy remains for lighting.
  ${retainSampledShadowCasters ? "// Raster shadow maps still consume the proxy casters." : `if((dryMaterial(sceneIdentityMaterial(identity)).identity.w&${SVO_MATERIAL_FLAGS.sampledVolume}u)!=0u){return;}`}
  if(meshPhase==0u){atomicAdd(&meshWork[meshScratchAt(meshEntry)],1u);return;}
  let slot=atomicAdd(&meshWork[meshScratchAt(meshEntry)+1u],1u);
  // The two phases traverse the same published voxels; a mismatch is an
  // extraction fault, reported rather than written past the brick's range.
  if(slot>=meshEmitCount){atomicOr(&meshState[${W.errorFlags}],2u);return;}
  meshArena[meshEmitBase+slot]=SurfaceQuad(origin,identity,extent,packedFace);
}
${contours ? svoContourMeshWGSL : ""}
${svoVoxelMeshWGSL(W.errorFlags, contours)}
${contours ? `// A dual cell graded past what a triangle record holds is an extraction fault.
fn meshDmcUnsupported(){atomicOr(&meshState[${W.errorFlags}],2u);}
fn meshDmcPackNormal(normal:vec3f)->u32{return svoGBufferPackNormalOct8(normal);}
${svoDualContouringMeshWGSL}${svoDualMarchingCubesMeshWGSL}` : ""}
fn meshExtractJob(leafIndex:u32,local:u32){
  ${contours ? `if(meshDcEnabled()){
    let leaf=svoLeafLoad(leafIndex).topology;let node=svoNodeLoad(leaf.x);
    let scale=1u<<(dry.mapping.maximumDepth-node.address.z);
    let base=svoDecodeMorton(node.address.x,node.address.y,node.address.z)*(scale*dry.mapping.brickSize);
    if(meshDmcEnabled()){meshDmcExtract(base,scale,node.address.z,local,dry.mapping.brickSize);}
    else{meshDcExtract(base,scale,node.address.z,local,dry.mapping.brickSize);}
    return;
  }` : ""}
  meshVoxelExtractJob(leafIndex,local);
}
fn meshMaintenanceBound()->bool{return atomicLoad(&meshState[${W.hostMaintenance}])!=0u;}
fn meshMaintenanceWord(index:u32)->u32{
  let at=atomicLoad(&meshState[${W.hostMaintenanceStateWords}])+index;
  if(at>=arrayLength(&meshMaintenance)){return 0u;}
  return meshMaintenance[at];
}
fn meshJournalWrite()->u32{
  if(!meshMaintenanceBound()){return 0u;}
  return meshMaintenanceWord(MESH_MAINTENANCE_JOURNAL_WRITE);
}
// Leaf of journal entry \`entry\`: the journal is a ring over the dirty brick records.
fn meshJournalLeaf(entry:u32)->u32{
  let capacity=max(atomicLoad(&meshState[${W.hostMaintenanceCapacity}]),1u);
  let at=atomicLoad(&meshState[${W.hostMaintenanceRecordsWords}])+(entry%capacity)*4u+MESH_JOURNAL_RECORD_WORD;
  if(at>=arrayLength(&meshMaintenance)){return MESH_NO_ENTRY;}
  return meshMaintenance[at];
}
// Where the shadow maps must redraw: every brick claimed for the build that
// extracts it. A claim made between builds belongs to the next one.
fn meshShadowGrow(box:MeshBrickBox){
  let build=atomicLoad(&meshState[${W.builds}])+select(1u,0u,atomicLoad(&meshState[${W.building}])!=0u);
  let base=${W.shadowBoxes}u+(build&1u)*6u;
  for(var axis=0u;axis<3u;axis+=1u){
    atomicMax(&meshState[base+axis],~box.minimum[axis]);atomicMax(&meshState[base+3u+axis],box.maximum[axis]);
  }
}
fn meshUnionReset(){meshBoxStore(MESH_BOX_UNION,vec3u(0xffffffffu),vec3u(0u));}
fn meshUnionGrow(box:MeshBrickBox){
  let base=MESH_BOX_UNION*MESH_BOX_WORDS;
  for(var axis=0u;axis<3u;axis+=1u){
    atomicMin(&meshWork[base+axis],box.minimum[axis]);atomicMax(&meshWork[base+4u+axis],box.maximum[axis]);
  }
}
// The brick a leaf slot's table entry was last extracted for, if any, and
// whether that is the brick the slot addresses now.
struct MeshHeld { held:bool, same:bool, box:MeshBrickBox }
fn meshHeld(leaf:u32)->MeshHeld{
  let table=meshTableAt(leaf);
  let countWord=atomicLoad(&meshWork[table+1u]);
  let keyX=atomicLoad(&meshWork[table+2u]);let keyY=atomicLoad(&meshWork[table+3u]);
  let tag=countWord>>27u;
  let held=tag!=0u;
  var same=false;
  if(held&&meshLeafLinked(leaf)){
    let node=svoNodeLoad(svoLeafLoad(leaf).topology.x);
    same=keyX==node.address.x&&keyY==node.address.y&&tag==node.address.z+1u;
  }
  return MeshHeld(held,same,meshAddressBox(keyX,keyY,tag-min(tag,1u)));
}
fn meshAppendWork(leaf:u32){
  atomicStore(&meshWork[meshTableAt(leaf)+5u],0u);
  let slot=meshEntrySlot(atomicAdd(&meshState[${W.worklistCount}],1u));
  atomicStore(&meshWork[meshListAt(slot)],leaf);
  let scratch=meshScratchAt(slot);
  atomicStore(&meshWork[scratch],0u);atomicStore(&meshWork[scratch+1u],0u);
}
// List a brick for extraction once. The claim is held from here until the
// brick's batch is emitted, so however many journal entries and neighbours
// name a brick, the worklist holds it once and never more bricks than slots.
fn meshClaim(leaf:u32,claim:u32){
  if(atomicMax(&meshWork[meshTableAt(leaf)+MESH_CLAIM_WORD],claim)==0u){meshAppendWork(leaf);}
}
// A brick whose voxels changed, or whose slot was freed or re-keyed: list it
// and grow the mask over where it is and where its held quads are. A brick
// still being repaired where it was is left to its journal entry.
fn meshClaimDirty(leaf:u32){
  let linked=meshLeafLinked(leaf);let current=meshLeafCurrent(leaf);
  let held=meshHeld(leaf);
  let moved=held.held&&!held.same;
  if(linked&&!current&&!moved){return;}
  meshClaim(leaf,MESH_CLAIM_DIRTY);
  if(current){let box=meshMaskBox(meshLeafBox(leaf));meshUnionGrow(box);meshShadowGrow(box);}
  if(moved){let box=meshMaskBox(held.box);meshUnionGrow(box);meshShadowGrow(box);}
}
fn meshClaimTouching(leaf:u32){
  if(meshLeafCurrent(leaf)){meshClaim(leaf,MESH_CLAIM_TOUCHING);}
}
// Whether a child octant of the brick-sized region one step along
// \`direction\` touches the brick it was stepped from: towards +axis the
// region's low half does, towards -axis its high half.
fn meshOctantTouches(octant:u32,direction:vec3i)->bool{
  let bits=vec3i(i32(octant&1u),i32((octant>>1u)&1u),i32((octant>>2u)&1u));
  return all((direction==vec3i(0))|(bits==select(vec3i(0),vec3i(1),direction<vec3i(0))));
}
// List every current brick that touches \`leaf\` across the face, edge or
// corner \`direction\` names. The region beside the brick is either inside a
// brick at least as large, or empty, or subdivided, in which case only the
// descendants along the shared boundary are walked.
fn meshClaimNeighbours(leaf:u32,direction:vec3i){
  let node=svoNodeLoad(svoLeafLoad(leaf).topology.x);
  let depth=min(node.address.z,dry.mapping.maximumDepth);
  let coordinate=vec3i(svoDecodeMorton(node.address.x,node.address.y,depth))+direction;
  if(any(coordinate<vec3i(0))||any(coordinate>=vec3i(i32(1u<<depth)))){return;}
  let beside=vec3u(coordinate);
  var index=0u;
  for(var level=0u;level<depth;level+=1u){
    if(index>=svoControlLoad(0u)){return;}
    let at=svoNodeLoad(index);
    if(at.links.z!=SVO_INVALID){meshClaimTouching(at.links.z);return;}
    let shifted=beside>>vec3u(depth-level-1u);
    let octant=(shifted.x&1u)|((shifted.y&1u)<<1u)|((shifted.z&1u)<<2u);
    let mask=at.address.w&255u;let bit=1u<<octant;
    if((mask&bit)==0u){return;}
    index=at.links.x+countOneBits(mask&(bit-1u));
  }
  var nodes:array<u32,16>;var next:array<u32,16>;
  var top=0i;nodes[0]=index;next[0]=0u;
  for(var step=0u;step<8192u&&top>=0;step+=1u){
    let current=nodes[top];
    if(current>=svoControlLoad(0u)){top-=1;continue;}
    let at=svoNodeLoad(current);
    if(at.links.z!=SVO_INVALID){meshClaimTouching(at.links.z);top-=1;continue;}
    let mask=at.address.w&255u;
    var octant=next[top];
    while(octant<8u&&(!meshOctantTouches(octant,direction)||(mask&(1u<<octant))==0u)){octant+=1u;}
    if(octant>=8u){top-=1;continue;}
    // A subtree deeper than the walk's stack is a fault, never a silent miss.
    if(top>=15){atomicOr(&meshState[${W.errorFlags}],2u);return;}
    next[top]=octant+1u;
    top+=1;nodes[top]=at.links.x+countOneBits(mask&((1u<<octant)-1u));next[top]=0u;
  }
  if(top>=0){atomicOr(&meshState[${W.errorFlags}],2u);}
}

// Once per presentation: notice what the voxelizer has finalized, start or
// join the build that answers it, and hand the ingest pass its journal span.
// Every decision is GPU resident so a stale host receipt can never restart
// or skip a build.
@compute @workgroup_size(1)
fn surfaceMeshPrepare(){
  for(var level=0u;level<8u;level+=1u){atomicStore(&meshState[${W.lodBricks}u+level],0u);}
  atomicStore(&meshState[${W.drawVertexCount}],4u);
  atomicStore(&meshState[${W.cullQuads}],0u);
  atomicStore(&meshState[${W.ingestBegin}],0u);atomicStore(&meshState[${W.ingestEnd}],0u);
  // Smooth raster surfaces use the dual-grid attachment. Publication and
  // attachment checks below keep extraction on the matching geometry generation.
  let attachment=atomicLoad(&meshState[${W.hostSurfaceVertices}]);
  let dualReady=!meshDcEnabled()||((attachment&0x7fffffffu)!=0u&&((attachment&0x80000000u)!=0u)==meshDmcEnabled());
  let valid=dryPublicationWord(0u)!=0u&&(dryPublicationWord(1u)&REQUIRED_FIELDS)==REQUIRED_FIELDS&&dualReady;
  if(!valid){
    atomicStore(&meshState[${W.usable}],0u);atomicStore(&meshState[${W.drawInstanceCount}],0u);
    atomicStore(&meshState[${W.building}],0u);atomicStore(&meshState[${W.errorFlags}],0u);atomicStore(&meshState[${W.mode}],0u);
    atomicStore(&meshState[${W.flags}],0u);
    return;
  }
  let p=(uniforms.cameraPosition.xyz-dry.mapping.worldOrigin)/dry.mapping.cellSize;
  atomicStore(&meshState[${W.withheld}],select(0u,1u,meshPointSolid(p)));
  var building=atomicLoad(&meshState[${W.building}])!=0u;
  var flags=atomicLoad(&meshState[${W.flags}])&~${F.scheduled}u;
  // The leaf scan a flag asked for ran with the last presentation's selection.
  if((flags&${F.scanned}u)!=0u){flags=0u;}
  let bound=meshMaintenanceBound();
  let write=meshJournalWrite();
  var cursor=atomicLoad(&meshState[${W.journalCursor}]);
  // The journal is a ring: a consumer more than its capacity behind, or
  // behind a wrapped write cursor, has lost entries and extracts everything.
  let lapped=bound&&(write<cursor||write-cursor>atomicLoad(&meshState[${W.hostMaintenanceCapacity}]));
  let topologyChanged=atomicLoad(&meshState[${W.topologyRevision}])!=dryPublicationWord(2u);
  let contourChanged=atomicLoad(&meshState[${W.contourMode}])!=bitcast<u32>(dry.meshFilterNormals.w);
  let geometryChanged=atomicLoad(&meshState[${W.geometryRevision}])!=dryPublicationWord(3u);
  let usable=atomicLoad(&meshState[${W.usable}])!=0u;
  let first=atomicLoad(&meshState[${W.builds}])==0u||(!usable&&!building);
  atomicStore(&meshState[${W.contourMode}],bitcast<u32>(dry.meshFilterNormals.w));
  atomicStore(&meshState[${W.topologyRevision}],dryPublicationWord(2u));
  atomicStore(&meshState[${W.geometryRevision}],dryPublicationWord(3u));
  if(bound){atomicStore(&meshState[${W.consumedMaintenanceRevision}],meshMaintenanceWord(MESH_MAINTENANCE_COMPLETED));}
  if(first||contourChanged||lapped||(!bound&&(topologyChanged||geometryChanged))){
    // A whole build, in place: nothing is drawn, every range is void, and the
    // leaf scan lists every current brick. The journal entries up to here
    // describe bricks that scan will list anyway.
    var reason=4u;
    if(first){reason=1u;}else if(lapped){reason=5u;}else if(contourChanged){reason=4u;}else if(topologyChanged){reason=2u;}else{reason=3u;}
    cursor=write;atomicStore(&meshState[${W.journalCursor}],write);
    atomicStore(&meshState[${W.frontCursor}],0u);atomicStore(&meshState[${W.liveQuads}],0u);atomicStore(&meshState[${W.usable}],0u);
    atomicStore(&meshState[${W.worklistCount}],0u);atomicStore(&meshState[${W.processedBricks}],0u);
    atomicStore(&meshState[${W.errorFlags}],0u);atomicStore(&meshState[${W.overflowLength}],0u);
    atomicStore(&meshState[${W.boxCount}],0u);meshUnionReset();
    for(var rangeClass=0u;rangeClass<MESH_CLASS_COUNT;rangeClass+=1u){atomicStore(&meshWork[meshStackAt(rangeClass)],0u);}
    atomicAdd(&meshState[${W.builds}],1u);
    atomicStore(&meshState[${W.mode}],${M.initial}u);atomicStore(&meshState[${W.building}],1u);
    atomicStore(&meshState[${W.restartReason}],reason);
    flags=${F.markAll}u;building=true;
  }else if(!building&&(write!=cursor||atomicLoad(&meshState[${W.worklistCount}])!=atomicLoad(&meshState[${W.processedBricks}]))){
    // Bricks were finalized, or the last selection listed slots a topology
    // change freed or re-keyed. The worklist keeps what selection listed.
    atomicAdd(&meshState[${W.builds}],1u);
    atomicStore(&meshState[${W.mode}],${M.incremental}u);atomicStore(&meshState[${W.building}],1u);
    atomicStore(&meshState[${W.restartReason}],select(3u,2u,write==cursor));
    building=true;
  }
  atomicStore(&meshState[${W.flags}],flags);
  if(!building){return;}
  // The host answers an overflow with a larger binding only while it may
  // still size the arena; the batch resumes when that binding is visible.
  if((atomicLoad(&meshState[${W.errorFlags}])&1u)!=0u&&arrayLength(&meshArena)>atomicLoad(&meshState[${W.overflowLength}])){
    atomicAnd(&meshState[${W.errorFlags}],~1u);
  }
  atomicStore(&meshState[${W.ingestBegin}],cursor);
  atomicStore(&meshState[${W.ingestEnd}],cursor+min(write-cursor,MESH_INGEST_ENTRIES));
}
// Mark from the journal: one item per (entry, direction), the centre
// direction listing the finalized brick itself and the other 26 the bricks
// that touch it. Work is proportional to the bricks finalized, never to the
// leaf capacity. A fixed direct dispatch strides over the items.
@compute @workgroup_size(64)
fn surfaceMeshIngest(@builtin(global_invocation_id) id:vec3u){
  let begin=atomicLoad(&meshState[${W.ingestBegin}]);
  let count=atomicLoad(&meshState[${W.ingestEnd}])-begin;
  // The pass that reads the entries commits the cursor, so a presentation
  // that encodes no ingest leaves them for the next.
  if(id.x==0u&&count!=0u){atomicStore(&meshState[${W.journalCursor}],begin+count);}
  if(id.x>=MESH_INGEST_THREADS){return;}
  let items=count*27u;
  for(var item=id.x;item<items;item+=MESH_INGEST_THREADS){
    let leaf=meshJournalLeaf(begin+item/27u);
    if(leaf>=meshLeafCapacity()){continue;}
    let direction=item%27u;
    if(direction==13u){meshClaimDirty(leaf);continue;}
    if(!meshLeafCurrent(leaf)){continue;}
    meshClaimNeighbours(leaf,vec3i(i32(direction%3u),i32((direction/3u)%3u),i32(direction/9u))-vec3i(1));
  }
}
// Once per presentation, after ingestion: the batch's brick span.
@compute @workgroup_size(1)
fn surfaceMeshSchedule(){
  atomicStore(&meshState[${W.batchBegin}],0u);atomicStore(&meshState[${W.batchEnd}],0u);
  if(atomicLoad(&meshState[${W.building}])==0u){return;}
  if((atomicLoad(&meshState[${W.errorFlags}])&1u)!=0u){return;}
  let begin=atomicLoad(&meshState[${W.processedBricks}]);
  let end=begin+min(atomicLoad(&meshState[${W.worklistCount}])-begin,max(atomicLoad(&meshState[${W.bricksPerBatch}]),1u));
  atomicStore(&meshState[${W.batchBegin}],begin);atomicStore(&meshState[${W.batchEnd}],end);
  atomicStore(&meshState[${W.checkpoint}],atomicLoad(&meshState[${W.frontCursor}]));
  atomicOr(&meshState[${W.flags}],${F.scheduled}u);
}
// Count and emit are fixed direct dispatches of the thread count the host
// stamps; a thread strides over the batch's (brick, job) pairs.
fn meshExtractThreads()->u32{return max(atomicLoad(&meshState[${W.hostExtractThreads}]),64u);}
// Count phase: size every brick of the batch.
@compute @workgroup_size(64)
fn surfaceMeshCount(@builtin(global_invocation_id) id:vec3u){
  let threads=meshExtractThreads();
  if(id.x>=threads){return;}
  let begin=atomicLoad(&meshState[${W.batchBegin}]);
  let jobsPerBrick=meshJobsPerBrick();
  let jobs=(atomicLoad(&meshState[${W.batchEnd}])-begin)*jobsPerBrick;
  for(var job=id.x;job<jobs;job+=threads){
    let slot=meshEntrySlot(begin+job/jobsPerBrick);
    let leaf=atomicLoad(&meshWork[meshListAt(slot)]);
    if(!meshLeafCurrent(leaf)){continue;}
    meshPhase=0u;meshEntry=slot;
    meshExtractJob(leaf,job%jobsPerBrick);
  }
}
// Allocate phase, one thread per batch brick: reserve the brick's new range
// and record it. A range of the old one's size class is the old one, to be
// rewritten in place; otherwise it comes from its class's free stack, or
// from the end of the arena when that stack is empty.
@compute @workgroup_size(64)
fn surfaceMeshAllocate(@builtin(global_invocation_id) id:vec3u){
  let entry=atomicLoad(&meshState[${W.batchBegin}])+id.x;
  if(id.x>=atomicLoad(&meshState[${W.batchEnd}])-atomicLoad(&meshState[${W.batchBegin}])){return;}
  let slot=meshEntrySlot(entry);
  let leaf=atomicLoad(&meshWork[meshListAt(slot)]);
  let scratch=meshScratchAt(slot);
  let pending=atomicExchange(&meshWork[scratch],0u)&MESH_COUNT_MASK;
  atomicStore(&meshWork[scratch+1u],0u);
  let table=meshTableAt(leaf);
  // The previous range is remembered for the emit pass, which frees it, and
  // for an overflow rollback, which gives it back untouched.
  let oldBase=atomicLoad(&meshWork[table]);let oldWord=atomicLoad(&meshWork[table+1u]);
  atomicStore(&meshWork[scratch+2u],oldBase);atomicStore(&meshWork[scratch+3u],oldWord);
  let current=meshLeafCurrent(leaf);
  // Queued for repair again since it was listed: its quads stay as they are
  // and the voxelizer journals it when the repair lands. A slot re-keyed to a
  // brick under repair gives up the quads of the brick it used to address.
  let held=meshHeld(leaf);
  if(!current&&meshLeafLinked(leaf)&&(!held.held||held.same)){atomicStore(&meshWork[scratch+2u],MESH_NO_ENTRY);return;}
  var keyX=0u;var keyY=0u;var tag=0u;
  if(current){let node=svoNodeLoad(svoLeafLoad(leaf).topology.x);keyX=node.address.x;keyY=node.address.y;tag=node.address.z+1u;}
  let oldCount=oldWord&MESH_COUNT_MASK;
  let rangeClass=meshRangeClass(pending);
  var base=oldBase;
  if(rangeClass!=meshRangeClass(oldCount)){
    base=meshPopRange(rangeClass);
    if(rangeClass==0u){base=0u;}
    else if(base==MESH_NO_ENTRY){
      let capacity=meshClassCapacity(rangeClass);
      base=atomicAdd(&meshState[${W.frontCursor}],capacity);
      if(base+capacity>arrayLength(&meshArena)){atomicOr(&meshState[${W.errorFlags}],1u);}
    }
  }
  atomicStore(&meshWork[table],base);atomicStore(&meshWork[table+1u],pending|(tag<<27u));
  atomicStore(&meshWork[table+2u],keyX);atomicStore(&meshWork[table+3u],keyY);
  atomicAdd(&meshState[${W.liveQuads}],pending);atomicSub(&meshState[${W.liveQuads}],oldCount);
}
// Emit phase: the same traversal as the count phase, writing into the
// brick's reserved range. It runs only once the whole batch has its ranges,
// so it is also where a brick's previous range is freed and its claim
// released: an overflowing batch never reaches it, and the drawn mesh keeps
// every brick until its replacement is written in the same presentation.
@compute @workgroup_size(64)
fn surfaceMeshEmit(@builtin(global_invocation_id) id:vec3u){
  if((atomicLoad(&meshState[${W.errorFlags}])&1u)!=0u){return;}
  let threads=meshExtractThreads();
  if(id.x>=threads){return;}
  let begin=atomicLoad(&meshState[${W.batchBegin}]);
  let jobsPerBrick=meshJobsPerBrick();
  let jobs=(atomicLoad(&meshState[${W.batchEnd}])-begin)*jobsPerBrick;
  let dead=SurfaceQuad(vec3u(0u),0u,vec3u(0u),0u);
  let length=arrayLength(&meshArena);
  for(var job=id.x;job<jobs;job+=threads){
    let slot=meshEntrySlot(begin+job/jobsPerBrick);let local=job%jobsPerBrick;
    let leaf=atomicLoad(&meshWork[meshListAt(slot)]);
    let table=meshTableAt(leaf);let scratch=meshScratchAt(slot);
    let oldBase=atomicLoad(&meshWork[scratch+2u]);
    if(local==0u){atomicStore(&meshWork[table+MESH_CLAIM_WORD],0u);}
    if(oldBase==MESH_NO_ENTRY){continue;}
    let base=atomicLoad(&meshWork[table]);let count=atomicLoad(&meshWork[table+1u])&MESH_COUNT_MASK;
    if(local==0u){
      let oldCount=atomicLoad(&meshWork[scratch+3u])&MESH_COUNT_MASK;
      let rangeClass=meshRangeClass(count);let oldClass=meshRangeClass(oldCount);
      if(oldClass!=rangeClass){
        for(var i=0u;i<oldCount;i+=1u){if(oldBase+i<length){meshArena[oldBase+i]=dead;}}
        meshPushRange(oldClass,oldBase);
      }
      // Every slot of the range past the quads written below is dead, whether
      // it held the brick's earlier quads or whatever the arena held before.
      let capacity=max(meshClassCapacity(rangeClass),select(0u,oldCount,oldClass==rangeClass));
      for(var i=count;i<capacity;i+=1u){if(base+i<length){meshArena[base+i]=dead;}}
    }
    if(!meshLeafCurrent(leaf)){continue;}
    meshPhase=1u;meshEntry=slot;
    meshEmitBase=base;meshEmitCount=count;
    meshExtractJob(leaf,local);
  }
}
@compute @workgroup_size(1)
fn surfaceMeshPublish(){
  let flags=atomicLoad(&meshState[${W.flags}]);
  atomicAnd(&meshState[${W.flags}],~${F.scheduled}u);
  if(atomicLoad(&meshState[${W.building}])!=0u){
    let begin=atomicLoad(&meshState[${W.batchBegin}]);let end=atomicLoad(&meshState[${W.batchEnd}]);
    if((flags&${F.scheduled}u)!=0u){
      if((atomicLoad(&meshState[${W.errorFlags}])&1u)!=0u){
        // An overflowing batch is rolled back whole to its checkpoint: every
        // brick in it gets its previous range back and keeps its claim.
        atomicStore(&meshState[${W.frontCursor}],atomicLoad(&meshState[${W.checkpoint}]));
        atomicStore(&meshState[${W.overflowLength}],arrayLength(&meshArena));
        for(var entry=begin;entry!=end;entry+=1u){
          let slot=meshEntrySlot(entry);
          let leaf=atomicLoad(&meshWork[meshListAt(slot)]);let scratch=meshScratchAt(slot);let table=meshTableAt(leaf);
          if(atomicLoad(&meshWork[scratch+2u])==MESH_NO_ENTRY){continue;}
          let previousCount=atomicLoad(&meshWork[scratch+3u]);
          atomicSub(&meshState[${W.liveQuads}],atomicLoad(&meshWork[table+1u])&MESH_COUNT_MASK);
          atomicAdd(&meshState[${W.liveQuads}],previousCount&MESH_COUNT_MASK);
          atomicStore(&meshWork[table],atomicLoad(&meshWork[scratch+2u]));atomicStore(&meshWork[table+1u],previousCount);
        }
      }else{
        atomicStore(&meshState[${W.processedBricks}],end);
        atomicAdd(&meshState[${W.rebuiltBricks}],end-begin);
      }
    }
    // The build is complete when the journal is consumed, no leaf scan is
    // owed and every listed brick has been extracted. A brick finalized
    // while it ran joined it; nothing restarted.
    let waiting=(flags&${F.markAll}u)!=0u||(atomicLoad(&meshState[${W.errorFlags}])&1u)!=0u;
    let processed=atomicLoad(&meshState[${W.processedBricks}]);
    if(!waiting&&meshJournalWrite()==atomicLoad(&meshState[${W.journalCursor}])
      &&processed==atomicLoad(&meshState[${W.worklistCount}])){
      let mode=atomicLoad(&meshState[${W.mode}]);
      atomicStore(&meshState[${W.lastBuildMode}],mode);atomicStore(&meshState[${W.lastBuildBricks}],processed);
      atomicAdd(&meshState[select(${W.wholeBuilds}u,${W.incrementalBuilds}u,mode==${M.incremental}u)],1u);
      atomicStore(&meshState[${W.usable}],1u);
      atomicStore(&meshState[${W.building}],0u);atomicStore(&meshState[${W.mode}],0u);
      atomicStore(&meshState[${W.worklistCount}],0u);atomicStore(&meshState[${W.processedBricks}],0u);
      atomicStore(&meshState[${W.boxCount}],0u);meshUnionReset();
    }
  }
  let usable=atomicLoad(&meshState[${W.usable}])!=0u&&atomicLoad(&meshState[${W.withheld}])==0u&&(atomicLoad(&meshState[${W.errorFlags}])&2u)==0u;
  let quads=select(0u,atomicLoad(&meshState[${W.frontCursor}]),usable);
  atomicStore(&meshState[${W.drawInstanceCount}],${culling ? "0u" : "quads"});
  atomicStore(&meshState[${W.cullQuads}],quads);
}
// Dirty boxes of the build's pending bricks, recomputed every presentation
// after publication: one closed lattice box per finalized brick the mesh has
// not yet re-extracted while they fit, else their union. A brick leaves the
// mask in the presentation that emits it. Runs as one workgroup so the union
// can be reduced with workgroup-uniform barriers.
var<workgroup> meshBoxUnionMinimum:array<atomic<u32>,3>;
var<workgroup> meshBoxUnionMaximum:array<atomic<u32>,3>;
var<workgroup> meshBoxSlots:atomic<u32>;
fn meshBoxRecord(box:MeshBrickBox){
  let slot=atomicAdd(&meshBoxSlots,1u);
  if(slot<MESH_BOX_INDIVIDUAL){meshBoxStore(slot,box.minimum,box.maximum);}
  for(var axis=0u;axis<3u;axis+=1u){atomicMin(&meshBoxUnionMinimum[axis],box.minimum[axis]);atomicMax(&meshBoxUnionMaximum[axis],box.maximum[axis]);}
}
@compute @workgroup_size(64)
fn surfaceMeshBoxes(@builtin(local_invocation_index) lane:u32){
  // Only a drawn mesh is masked; a whole build traces everything. Bricks the
  // selection just listed are masked before the build that extracts them starts.
  let masking=atomicLoad(&meshState[${W.usable}])!=0u;
  let begin=atomicLoad(&meshState[${W.processedBricks}]);
  let pending=atomicLoad(&meshState[${W.worklistCount}])-begin;
  let sweep=masking&&pending<=MESH_BOX_SWEEP;
  if(lane<3u){atomicStore(&meshBoxUnionMinimum[lane],0xffffffffu);atomicStore(&meshBoxUnionMaximum[lane],0u);}
  if(lane==0u){atomicStore(&meshBoxSlots,0u);}
  workgroupBarrier();
  if(sweep){
    for(var i=lane;i<pending;i+=64u){
      let leaf=atomicLoad(&meshWork[meshListAt(meshEntrySlot(begin+i))]);
      if(atomicLoad(&meshWork[meshTableAt(leaf)+MESH_CLAIM_WORD])!=MESH_CLAIM_DIRTY){continue;}
      if(meshLeafCurrent(leaf)){meshBoxRecord(meshMaskBox(meshLeafBox(leaf)));}
      let held=meshHeld(leaf);
      if(held.held&&!held.same){meshBoxRecord(meshMaskBox(held.box));}
    }
  }
  workgroupBarrier();
  if(lane!=0u){return;}
  if(!masking){atomicStore(&meshState[${W.boxCount}],0u);return;}
  // Finalized bricks the ingest pass has not reached are not listed yet, so
  // until it catches up the whole frame is traced.
  if(meshJournalWrite()!=atomicLoad(&meshState[${W.journalCursor}])){
    meshBoxStore(MESH_BOX_UNION,vec3u(0u),vec3u(0xffffffffu));
    meshBoxStore(0u,vec3u(0u),vec3u(0xffffffffu));
    atomicStore(&meshState[${W.boxCount}],1u);return;
  }
  if(sweep){
    var unionMinimum=vec3u(0u);var unionMaximum=vec3u(0u);
    for(var axis=0u;axis<3u;axis+=1u){unionMinimum[axis]=atomicLoad(&meshBoxUnionMinimum[axis]);unionMaximum[axis]=atomicLoad(&meshBoxUnionMaximum[axis]);}
    meshBoxStore(MESH_BOX_UNION,unionMinimum,unionMaximum);
    let slots=atomicLoad(&meshBoxSlots);
    if(slots<=MESH_BOX_INDIVIDUAL){atomicStore(&meshState[${W.boxCount}],slots);return;}
  }
  // Too many boxes to list, or too many pending bricks to sweep: the union
  // alone stands for them all. Unswept, it is the union grown as bricks were
  // claimed, which covers every pending brick.
  meshBoxStore(0u,meshBoxMinimum(MESH_BOX_UNION),meshBoxMaximum(MESH_BOX_UNION));
  atomicStore(&meshState[${W.boxCount}],1u);
}
// Which of a brick's levels this camera draws. Selection has one writer per
// leaf, before culling, so all quads use the same hysteresis history. It is
// also the one pass that visits every leaf slot each presentation, so the
// leaf scans ride on it: a whole build lists every current brick and voids
// the rest of the table, and every presentation lists the slots a topology
// change freed or re-keyed, which no journal entry names.
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
  let flags=atomicLoad(&meshState[${W.flags}]);
  if(leaf==0u&&(flags&${F.markAll}u)!=0u){atomicOr(&meshState[${W.flags}],${F.scanned}u);}
  if(leaf>=meshLeafCapacity()){return;}
  let table=meshTableAt(leaf);
  let current=meshLeafCurrent(leaf);
  if((flags&${F.markAll}u)!=0u){
    atomicStore(&meshWork[table],0u);atomicStore(&meshWork[table+1u],0u);atomicStore(&meshWork[table+2u],0u);atomicStore(&meshWork[table+3u],0u);
    atomicStore(&meshWork[table+MESH_CLAIM_WORD],select(0u,MESH_CLAIM_TOUCHING,current));
    if(current){meshAppendWork(leaf);}
  }else if(atomicLoad(&meshWork[table+MESH_CLAIM_WORD])!=MESH_CLAIM_DIRTY){
    // A slot whose table no longer describes the brick it addresses was freed
    // or re-keyed by a topology change, which no journal entry names.
    let held=meshHeld(leaf);
    if((current&&!held.same)||(!current&&held.held)){meshClaimDirty(leaf);}
  }
  if(!current){atomicStore(&meshWork[table+5u],0u);return;}
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
  let origin=meshQuadOrigin(quad);var base=(origin/size)*size;
  if(!meshTriangle(quad)&&(meshQuadFace(quad.face)&1u)!=0u&&origin[axis]%size==0u){base[axis]-=size;}
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
  let quad=meshArena[index];
  if(meshQuadDead(quad)){return false;}
  if(!meshQuadSelected(quad)){return false;}
  // A dual triangle's vertices may lie up to four cells from its origin, so
  // its bounds are its own vertices rather than the record's nominal extent.
  var boundsLow=vec3f(0.0);var boundsHigh=vec3f(meshQuadExtent(quad));
  ${contours ? `if(meshDualTriangle(quad)){
    let a=meshTrianglePoint(quad,0u);let b=meshTrianglePoint(quad,1u);let c=meshTrianglePoint(quad,2u);
    boundsLow=min(a,min(b,c))*boundsHigh;boundsHigh=max(a,max(b,c))*boundsHigh;
  }` : ""}
  if(meshMaskActive()&&meshBoxesContain(meshQuadOrigin(quad)+vec3u(floor(boundsLow)),meshQuadOrigin(quad)+vec3u(ceil(boundsHigh)))){return false;}
  let face=meshQuadFace(quad.face);let axis=face/2u;
  let origin=dry.mapping.worldOrigin+vec3f(meshQuadOrigin(quad))*dry.mapping.cellSize;
  let camera=dryRasterPrimaryCamera();
  var facing=(camera[0][axis]-origin[axis])*select(-1.0,1.0,(face&1u)!=0u);
  ${contours ? `if(meshTriangle(quad)){
    let a=meshTrianglePoint(quad,0u);let b=meshTrianglePoint(quad,1u);let c=meshTrianglePoint(quad,2u);
    let n=cross((b-a)*dry.mapping.cellSize,(c-a)*dry.mapping.cellSize);
    let point=origin+a*vec3f(meshQuadExtent(quad))*dry.mapping.cellSize;
    facing=dot(camera[0]-point,n)/max(length(n),1e-20);
  }` : ""}
  // Retain the coplanar tolerance band to avoid rounding-dependent holes.
  let epsilon=max(1e-5,abs(origin[axis])*1e-6);
  if(facing < -epsilon){return false;}
  let baseHalf=(boundsHigh-boundsLow)*dry.mapping.cellSize*0.5;
  let center=origin+boundsLow*dry.mapping.cellSize+baseHalf-camera[0];
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
  // A fixed direct dispatch over the arena's capacity: threads past the quads
  // publication made drawable have nothing to test.
  let index=id.x+id.y*65535u*64u;
  let visible=index<atomicLoad(&meshState[${W.cullQuads}])&&meshQuadVisible(index);
  var rank=0u;if(visible){rank=atomicAdd(&meshCullCount,1u);}
  workgroupBarrier();
  if(lane==0u){
    let count=atomicLoad(&meshCullCount);
    meshCullBase=0u;if(count!=0u){meshCullBase=atomicAdd(&meshState[${W.drawInstanceCount}],count);}
  }
  workgroupBarrier();
  if(visible){meshVisibleOutput[meshCullBase+rank]=index;}
}
${proceduralSolidRenderWGSL(`meshHeader[${W.hostSolidBoundary}u]`, "meshMaintenance[index]",
  `bitcast<vec3f>(vec3u(meshHeader[${W.hostSolidBoundary+4}u],meshHeader[${W.hostSolidBoundary+5}u],meshHeader[${W.hostSolidBoundary+6}u]))`,
  `bitcast<vec3f>(vec3u(meshHeader[${W.hostSolidBoundary+1}u],meshHeader[${W.hostSolidBoundary+2}u],meshHeader[${W.hostSolidBoundary+3}u]))`, "meshBoundary")}
${svoAuthoredSurfaceNormalsWGSL(W.hostRenderTerrain, W.hostTerrainEdits, W.hostPrimitiveMaterialBase)}
struct MeshVertexOut {
  @builtin(position) position:vec4f,
  @location(0) world:vec3f,
  @location(1) @interpolate(flat) identity:u32,
  @location(2) normal:vec3f,
  @location(3) @interpolate(flat) level:u32,
  @location(4) @interpolate(flat) agreement:f32,
  @location(5) @interpolate(flat) cellPixels:f32,
  @location(6) @interpolate(flat) geometricNormal:vec3f,
  @location(7) boundaryNormal:vec3f,
}
@vertex fn surfaceMeshVertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->MeshVertexOut{
  let quad=meshDrawnQuad(${culling ? "meshVisible[instance]" : "instance"});
  let face=meshQuadFace(quad.face);let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  let corners=array<vec2u,4>(vec2u(1u,0u),vec2u(1u,1u),vec2u(0u,0u),vec2u(0u,1u));
  // Swapping the two in-plane coordinates reverses winding on negative
  // faces while preserving the 00–11 diagonal and the covered rectangle.
  let corner=select(corners[vertex],corners[vertex].yx,(face&1u)==0u);var lattice=meshQuadOrigin(quad);lattice[u]+=corner.x*quad.extent[u];lattice[v]+=corner.y*quad.extent[v];
  var world=dry.mapping.worldOrigin+vec3f(lattice)*dry.mapping.cellSize;
  ${contours ? `if(meshTriangle(quad)){
    let p=meshTrianglePoint(quad,min(vertex,2u));
    world=dry.mapping.worldOrigin+(vec3f(meshQuadOrigin(quad))+p*vec3f(meshQuadExtent(quad)))*dry.mapping.cellSize;
  }` : ""}
  let camera=dryRasterPrimaryCamera();let relative=world-camera[0];let z=dot(relative,camera[1]);
  var position=vec4f(dot(relative,camera[2])/(cameraTanHalfFov()*uniforms.viewport.x/max(uniforms.viewport.y,1.0)),dot(relative,camera[3])/cameraTanHalfFov(),DRY_REVERSED_Z_NEAR_M,z);
  ${culling ? "" : "// Without the cull pass, an unselected or freed quad collapses to a zero-area strip.\n  if(meshQuadDead(quad)||!meshQuadSelectedRead(quad)){position=vec4f(0.0,0.0,0.0,1.0);}"}
  var normal=vec3f(0.0);normal[axis]=select(-1.0,1.0,(face&1u)!=0u);
  ${contours ? `if(meshTriangle(quad)){
    let a=meshTrianglePoint(quad,0u);let b=meshTrianglePoint(quad,1u);let c=meshTrianglePoint(quad,2u);
    normal=normalize(cross((b-a)*dry.mapping.cellSize,(c-a)*dry.mapping.cellSize));
  }` : ""}
  let geometricNormal=normal;
  ${contours ? `if(meshDualTriangle(quad)&&(quad.face&0x20000000u)==0u){
    // The mesher stored each vertex's fitted normal; a vertex whose solid
    // leaf published none keeps the triangle's own.
    let word=quad.origin[min(vertex,2u)]>>16u;
    if(word!=SCENE_IDENTITY_NO_NORMAL){normal=svoGBufferUnpackNormalOct8(word);}
  }` : ""}
  var boundaryNormal=vec3f(0);
  if(meshDcEnabled()&&meshHeader[${W.hostSolidBoundary}u]!=0u){
    // Evaluate on the shared mesh vertex, not an interior fit's medial crease.
    // Interpolation keeps normals continuous across both triangles and the
    // sampled field's cell boundaries, including around a rounded rim.
    let boundary=meshBoundarySample(world);
    if(abs(boundary.w)<2.0*max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z))){let g=meshBoundaryNormal(world,boundary);if(dot(g,g)>1e-12){boundaryNormal=normalize(g);}}
  }
  if(meshDcEnabled()&&dot(boundaryNormal,boundaryNormal)==0.0){
    boundaryNormal=meshAuthoredSurfaceNormal(world,sceneIdentityMaterial(quad.identity),geometricNormal);
  }
  let brickLattice=dry.mapping.brickSize*(1u<<(dry.mapping.maximumDepth-meshQuadDepth(quad.face)));
  return MeshVertexOut(position,world,quad.identity,normal,meshQuadLevel(quad.face),f32((quad.face>>11u)&255u)/255.0,
    meshCellFootprint(meshQuadBrickBase(quad),brickLattice),geometricNormal,boundaryNormal);
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
  // Imported volumes keep their native lattice. This coarse derivative serves
  // lighting/occlusion; the source field's depth-tested proxy draws its surface.
  if((dryMaterial(sceneIdentityMaterial(input.identity)).identity.w&${SVO_MATERIAL_FLAGS.sampledVolume}u)!=0u){discard;}
  let camera=dryRasterPrimaryCamera();let rd=normalize(input.world-camera[0]);let t=length(input.world-camera[0]);
  var normal=normalize(input.normal);var receiverNormal=input.geometricNormal;
  if(dot(input.boundaryNormal,input.boundaryNormal)>1e-12){normal=normalize(input.boundaryNormal);receiverNormal=normal;}
  if(!meshContoursEnabled()&&!meshDcEnabled()&&dry.meshFilter.x>0.5&&dry.meshFilterNormals.x>0.5&&dry.meshFilter.y>0.0){
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
  var shading=dryShadingNormal(input.identity,normal);
  if(meshDcEnabled()){shading=DryShadingNormal(normal,SVO_FEATURE_SMOOTH);}
  // Tag reconstructed receivers for the shared lighting encoder. Their implicit
  // no-owner slot carries the geometric normal through both deferred passes.
  let hit=DryHit(t,shading.normal,sceneIdentityMaterial(input.identity),DRY_OWNER_NONE,shading.featureId,DRY_GBUFFER_FIELD_VOXEL,DRY_GBUFFER_MOTION_STATIC,0u,0.0,vec3u(0u,0u,select(0u,1u,meshDcEnabled()||meshContoursEnabled())));
  // Voxel quads retain their exact geometric face for ray bias. A sampled
  // implicit surface instead uses its own gradient: a tessellation normal's
  // hemisphere gate would bring flat triangle shading back at the terminator.
  var out=dryRasterPrimaryFacedSurface(hit,receiverNormal,camera[0],rd,camera[1],SVO_GBUFFER_PRODUCER_BRICK);
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
${svoMeshPlanarRasterWGSL}
@fragment fn surfaceMeshBackground(input:VertexOut)->DryRasterPrimaryOut{
  dryRasterPrimaryReset();let camera=dryRasterPrimaryCamera();let rd=dryRasterPrimaryRay(input.position.xy,camera);
  var hit=missHit();
  // Exact planes are rasterized independently of the cached voxel mesh. Without a drawable
  // mesh, traverse the current published SVO so edits appear immediately and
  // unaffected room geometry remains visible. While a build updates a drawn
  // mesh, only the pixels whose rays cross the boxes of the finalized bricks
  // it still owes are traced; the cull withholds the cached quads inside
  // those boxes. This uses the same voxel authority, never the preceding
  // mesh or an analytic approximation.
  // Accepted planes are drawn by meshPlanarFragment in this same pass.
  var producer=SVO_GBUFFER_PRODUCER_RASTER_BACKGROUND;
  let undrawn=meshHeader[${W.usable}]==0u||meshHeader[${W.withheld}]!=0u||(meshHeader[${W.errorFlags}]&2u)!=0u;
  let masked=!undrawn&&meshHeader[${W.boxCount}]!=0u&&meshRayCrossesDirtyBox(camera[0],rd);
  if(undrawn||masked){
    let current=traceStatic(camera[0],rd);
    if(current.t<hit.t){hit=current;producer=SVO_GBUFFER_PRODUCER_BRICK;}
  }else{
    // The backdrop ground is not octree content, so it is never meshed: the
    // tiled terrain walk answers it here with its first voxel, and the depth
    // this writes hides the set's quads that sit under it.
    let backdrop=dryBackdropTrace(camera[0],rd,hit.t);
    if(backdrop.t<hit.t){hit=backdrop;producer=SVO_GBUFFER_PRODUCER_BRICK;}
  }
  if(hit.t>=DRY_MISS){return dryRasterPrimaryMiss();}
  return dryRasterPrimarySurface(hit,camera[0],rd,camera[1],producer);
}
`;
}
