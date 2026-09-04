/**
 * Pixel-trace ABI: the record format the probes write for one pixel, and the
 * host decode that turns it into drawable 3D geometry.
 *
 * There are two shapes of answer here because the renderer has two ways of
 * resolving primary visibility, and they perform genuinely different work.
 *
 * - `traced`: one camera ray is re-marched by a probe that mirrors the
 *   shipping megakernel, appending a record per unit of work — each hierarchy
 *   node it opens, each child box it rejects, each leaf brick it enters, each
 *   fine cell its DDA steps.
 * - `raster`: there is no per-pixel search to mirror. A compute pass emitted one
 *   proxy box per resident brick and the rasterizer's depth test picked the
 *   winner, so the records describe *which proxies covered this pixel*, what
 *   each one's fragment cost, and which of them won.
 *
 * Everything from the surface onward — shadow rays, cone marches, ambient
 * occlusion, global illumination — is shared, because the deferred lighting pass
 * is the same in both.
 *
 * Nothing here knows about WebGPU: the layout is the contract between
 * `webgpu-svo-pixel-trace.ts` and `webgpu-svo-brick-raster-probe.ts` (which
 * write it on the GPU) and the overlay pipeline plus the HUD (which read it),
 * and it is exercised directly by tests.
 */
import { DECORATION_SEGMENT_FLOATS } from "../core/visualization-decorations";
import {
  decorationVisualization,
  hasFields,
  type Visualization,
} from "../core/visualization-registry";

export const SVO_PIXEL_TRACE_ABI_VERSION = 4;
/** "SVT" plus the ABI version, so a stale mapped buffer is never decoded. */
export const SVO_PIXEL_TRACE_MAGIC = 0x53565404;
/**
 * Words 0..39 carry exactly what version 3 carried, so the traced probe's
 * writes did not move; 40.. describe the raster primary, which version 3 had no
 * way to express. Records begin at a multiple of four either way.
 */
export const SVO_PIXEL_TRACE_HEADER_WORDS = 64;
export const SVO_PIXEL_TRACE_RECORD_WORDS = 12;
export const SVO_PIXEL_TRACE_DEFAULT_RECORD_CAPACITY = 4096;
export const SVO_PIXEL_TRACE_MAXIMUM_RECORD_CAPACITY = 65536;

/**
 * Records live in an r32uint storage texture, not a storage buffer.
 *
 * The dry-scene fragment stage already binds ten storage buffers, which is
 * exactly the per-stage ceiling browsers report on Apple silicon, so a diagnostic
 * that wanted an eleventh could never run where it matters. Storage textures are
 * a separate budget the dry pass does not touch. One row is 256 words = 1024
 * bytes, which is both a legal `bytesPerRow` and padding-free, so the copied
 * buffer is the same flat word array the decode below expects.
 */
export const SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS = 256;

export function svoPixelTraceTextureRows(recordCapacity = SVO_PIXEL_TRACE_DEFAULT_RECORD_CAPACITY): number {
  return Math.ceil(svoPixelTraceWordCount(recordCapacity) / SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS);
}

export function svoPixelTraceWordCount(recordCapacity = SVO_PIXEL_TRACE_DEFAULT_RECORD_CAPACITY): number {
  if (!Number.isSafeInteger(recordCapacity) || recordCapacity < 1 || recordCapacity > SVO_PIXEL_TRACE_MAXIMUM_RECORD_CAPACITY) {
    throw new RangeError(`Pixel-trace record capacity must be an integer from 1 to ${SVO_PIXEL_TRACE_MAXIMUM_RECORD_CAPACITY}`);
  }
  return SVO_PIXEL_TRACE_HEADER_WORDS + recordCapacity * SVO_PIXEL_TRACE_RECORD_WORDS;
}

/** Readback size: whole rows, so the copy needs no per-row padding. */
export function svoPixelTraceBufferBytes(recordCapacity = SVO_PIXEL_TRACE_DEFAULT_RECORD_CAPACITY): number {
  return svoPixelTraceTextureRows(recordCapacity) * SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS * 4;
}

/**
 * Record kinds. The numbering is the ABI; the order below is also the natural
 * narrative order of one ray, which the HUD reuses.
 */
export const SVO_PIXEL_TRACE_KINDS = Object.freeze({
  /** An internal hierarchy node the ray entered and expanded. */
  hierarchyNode: 1,
  /** A child box the ray enters; deferred onto the stack or descended into. */
  childAccepted: 2,
  /** A child box tested by the same slab arithmetic and rejected. */
  childRejected: 3,
  /** A terminal leaf: one brick of cells the ray must walk. */
  leafBrick: 4,
  /** One fine cell visited by the in-brick DDA. */
  brickCell: 5,
  /** An exact analytic surface intersection issued from an occupied cell. */
  exactTest: 6,
  /** The resolved primary surface point. */
  primaryHit: 7,
  /** A shadow ray from the surface toward one light sample. */
  shadowRay: 8,
  /** One cone sample of a shadow march against the coverage pyramid. */
  coneSample: 9,
  /** One cone sample of an ambient-occlusion march. */
  occlusionConeSample: 10,
  /** A terrain heightfield bracket/refinement probe along the ray. */
  terrainStep: 11,
  /** A rigid-body analytic test; the renderer keeps at most twelve. */
  rigidTest: 12,
  /** One opacity/radiance tap from a wide diffuse global-illumination cone. */
  globalIlluminationConeSample: 13,
  /**
   * Raster primary: one brick proxy whose box covers this pixel — that is, one
   * fragment the rasterizer produced here. `a`/`b` are the proxy AABB, `level`
   * is the sort bucket, `detail` is the instance index in draw order, and
   * `tEnter`/`tExit` are the ray interval its fragment clamped to.
   */
  brickProxy: 14,
  /**
   * The full leaf AABB of a proxy, drawn around the occupied sub-box the
   * emitter actually published. The gap between them is the occupancy word
   * doing its job, which is otherwise invisible.
   */
  leafBounds: 15,
  /** A rigid impostor proxy covering this pixel, plus its analytic test. */
  rigidProxy: 16,
  /** The thin-glass pane key the discovery pass wrote at this pixel. */
  glassPane: 17,
  /** The reduced-rate cone-lighting texel this pixel's visibility came from. */
  prepassTexel: 18,
} as const);

export type SvoPixelTraceKind = typeof SVO_PIXEL_TRACE_KINDS[keyof typeof SVO_PIXEL_TRACE_KINDS];

export const SVO_PIXEL_TRACE_FLAGS = Object.freeze({
  /** The recorded test produced a hit (child entered, primitive intersected). */
  hit: 1 << 0,
  /** A child that traversal descended into immediately rather than deferring. */
  descended: 1 << 1,
  /** A child pushed onto the near-to-far stack for later. */
  deferred: 1 << 2,
  /** A brick the ray crossed without finding any surface. */
  emptySkip: 1 << 3,
  /** A cell whose owner tag named a candidate primitive. */
  tagged: 1 << 4,
  /** Second phase of the split shadow march: samples anchored at the emitter. */
  emitterAnchored: 1 << 5,
  /** Work stopped here because a shader budget was exhausted. */
  budgetExhausted: 1 << 6,
  /** Raster primary: this proxy's fragment won the depth test. It is the pixel. */
  depthWinner: 1 << 7,
  /**
   * Shaded, found a surface, and lost the depth test to something nearer.
   * Distinct from {@link discarded}: this fragment wrote a depth that was then
   * beaten, which is a different outcome from never writing one.
   */
  depthLoser: 1 << 8,
  /**
   * The in-brick DDA found no surface, so the fragment discarded. It still cost
   * a full invocation, which is why it is never folded in with the instances
   * the cull rejected before the draw.
   */
  discarded: 1 << 9,
  /**
   * A tile-based hidden-surface pass could legally have killed this fragment
   * before it shaded. Whether it did is not observable from inside the shader,
   * so anything carrying this bit is reported as an upper bound, never as work
   * that certainly happened.
   */
  hsrEligible: 1 << 10,
} as const);

/** Probe status. Mirrors the traversal status codes plus probe-only outcomes. */
export const SVO_PIXEL_TRACE_STATUS = Object.freeze({
  /** No probe has run for this request yet. */
  pending: 0,
  /** The probe ran and the ray terminated on a surface. */
  hit: 1,
  /** The probe ran and the ray left the scene. */
  miss: 2,
  /** The probe could not run: publication incomplete or fields invalid. */
  unavailable: 3,
  /** A bounded loop hit its ceiling; the recorded prefix is still exact. */
  exhausted: 4,
  /** Topology or payload data failed a validity check. */
  invalid: 5,
} as const);

export type SvoPixelTraceStatus = typeof SVO_PIXEL_TRACE_STATUS[keyof typeof SVO_PIXEL_TRACE_STATUS];

/**
 * Header word offsets. Words 0..15 identify the request, 16..31 count work, and
 * 32.. describe the grid the work happened on.
 */
export const SVO_PIXEL_TRACE_HEADER = Object.freeze({
  magic: 0,
  status: 1,
  recordCount: 2,
  droppedRecords: 3,
  pixelX: 4,
  pixelY: 5,
  rayOrigin: 6,
  rayDirection: 9,
  hitDistance: 12,
  hitNormal: 13,
  nodeVisits: 16,
  leafVisits: 17,
  emptyBrickSkips: 18,
  voxelWork: 19,
  exactTests: 20,
  maximumDepth: 21,
  shadowNodeVisits: 22,
  shadowLeafVisits: 23,
  shadowWork: 24,
  mipSteps: 25,
  traversalFailure: 26,
  hitOwnerId: 27,
  hitMaterialId: 28,
  hitFeatureId: 29,
  shadedLights: 30,
  requestToken: 31,
  /**
   * Finest voxel width, in metres: the width of a level-0 cell of the coverage
   * pyramid. Every mip level a cone sample reads is this doubled `level` times,
   * so without it the overlay cannot draw the footprint a tap actually covered.
   */
  minimumVoxel: 32,
  giConeTaps: 33,
  giConeCount: 34,
  giVisibility: 35,
  giRadiance: 36,
  /** bit 0: GLOBAL selected; bit 1: radiance publication ready. */
  giState: 39,

  /* Words 40.. describe the raster primary. Zero throughout on a traced frame. */

  /** {@link SVO_PIXEL_TRACE_PRIMARY_MODE}: how primary visibility was resolved. */
  primaryMode: 40,
  /** {@link SVO_PIXEL_TRACE_STAGES}: which passes contributed to this record set. */
  stagesPresent: 41,
  /** Leaves the emitter walked. Frame-wide, not per-pixel. */
  residentLeaves: 42,
  /** Leaves rejected as empty from the published occupancy word. Frame-wide. */
  emptyBricks: 43,
  /** Proxies rejected by the frustum test. Frame-wide. */
  frustumCulled: 44,
  /** Proxies that survived emission and were bucketed. Frame-wide. */
  candidatesEmitted: 45,
  /** Instances the indirect draw submitted. Frame-wide. */
  instancesDrawn: 46,
  /** Proxies whose box the camera ray pierces — the overdraw at this pixel. */
  coveringProxies: 47,
  /** Covering proxies whose DDA found a surface; the rest discarded. */
  proxiesWithSurface: 48,
  /** Covering proxies a hidden-surface pass could have killed before shading. */
  hsrEligibleProxies: 49,
  /** Draw-order index of the proxy that won the depth test. */
  winnerInstanceIndex: 50,
  /** Sort bucket the winner landed in, of {@link SVO_PIXEL_TRACE_SORT_BUCKETS}. */
  winnerSortBucket: 51,
  /** DDA cells stepped across *every* covering proxy, not just the winner. */
  ddaCellsAcrossProxies: 52,
  // Word 53 is reserved.
  /** {@link SVO_PIXEL_TRACE_PREPASS_STATE}. */
  prepassState: 54,
} as const);

export const SVO_PIXEL_TRACE_GI_STATE = Object.freeze({ enabled: 1 << 0, ready: 1 << 1 } as const);

/**
 * How primary visibility was resolved for the frame this trace came from.
 *
 * This is the diagnostic's honesty switch. The two modes perform different work
 * and therefore have different things to say; a picker that showed one mode's
 * vocabulary over the other's frame would be describing a shader that did not
 * run. Carried in the record rather than read from the live renderer so a pinned
 * trace keeps explaining the frame it was actually taken from.
 */
export const SVO_PIXEL_TRACE_PRIMARY_MODE = Object.freeze({ traced: 0, raster: 1 } as const);

export type SvoPixelTracePrimaryMode = keyof typeof SVO_PIXEL_TRACE_PRIMARY_MODE;

/**
 * Passes that contributed to this record set.
 *
 * A pass that did not run is absent, never zero: "no rigid bodies in this scene"
 * and "the rigid pass found nothing at this pixel" are different facts and the
 * HUD renders them differently.
 */
export const SVO_PIXEL_TRACE_STAGES = Object.freeze({
  brickCull: 1 << 0,
  brickRaster: 1 << 1,
  terrain: 1 << 2,
  rigid: 1 << 3,
  glass: 1 << 4,
  conePrepass: 1 << 5,
  deferredLighting: 1 << 6,
} as const);

export type SvoPixelTraceStage = keyof typeof SVO_PIXEL_TRACE_STAGES;

/**
 * Reduced-rate cone lighting.
 *
 * Only "did the frame run one" is recorded, and it is stamped by the host: the
 * probe is composed inline at full rate by construction, so it cannot observe
 * the reduced pass beside it. Whether *this* pixel was a boundary pixel resolved
 * inline is a fact only the reduced shader holds, and it is deliberately not
 * guessed at here.
 */
export const SVO_PIXEL_TRACE_PREPASS_STATE = Object.freeze({
  /** The frame ran reduced-rate cone lighting. */
  active: 1 << 0,
} as const);

/** Buckets the brick emitter's counting sort quantizes min view depth into. */
export const SVO_PIXEL_TRACE_SORT_BUCKETS = 1024;

export type SvoTraceVec3 = readonly [number, number, number];

export interface SvoPixelTraceCounters {
  readonly nodeVisits: number;
  readonly leafVisits: number;
  readonly emptyBrickSkips: number;
  readonly voxelWork: number;
  readonly exactTests: number;
  readonly maximumDepth: number;
  readonly shadowNodeVisits: number;
  readonly shadowLeafVisits: number;
  readonly shadowWork: number;
  readonly mipSteps: number;
  readonly traversalFailure: number;
  readonly shadedLights: number;
}

/** One record, decoded. `a`/`b` carry a box, a segment, or a point plus radii. */
export interface SvoPixelTraceRecord {
  readonly kind: SvoPixelTraceKind;
  readonly order: number;
  readonly level: number;
  readonly detail: number;
  readonly flags: number;
  readonly a: SvoTraceVec3;
  readonly b: SvoTraceVec3;
  readonly tEnter_m: number;
  readonly tExit_m: number;
}

/**
 * What the raster primary did to find this pixel's surface.
 *
 * The cull counters are read verbatim from the sort-state buffer the frame's own
 * emission pass wrote, so they are measured rather than mirrored. They are
 * frame-wide: they describe one camera's worth of work, not this pixel's, and
 * the HUD is careful never to add them to per-pixel figures.
 */
export interface SvoPixelTraceRaster {
  readonly residentLeaves: number;
  readonly emptyBricks: number;
  readonly frustumCulled: number;
  readonly candidatesEmitted: number;
  readonly instancesDrawn: number;
  /** Proxies covering this pixel: the fragments the rasterizer produced here. */
  readonly coveringProxies: number;
  /** Of those, the ones whose in-brick DDA found a surface. */
  readonly proxiesWithSurface: number;
  /**
   * Covering proxies a tile-based hidden-surface pass could have killed before
   * shading. `coveringProxies - hsrEligibleProxies` is the count that certainly
   * shaded; the total is the upper bound.
   */
  readonly hsrEligibleProxies: number;
  /** Absent when nothing won — a sky pixel, or terrain in front of every brick. */
  readonly winner?: { readonly instanceIndex: number; readonly sortBucket: number };
  /** DDA cells stepped across every covering proxy, winner and losers alike. */
  readonly ddaCellsAcrossProxies: number;
}

export interface SvoPixelTrace {
  readonly version: number;
  readonly status: SvoPixelTraceStatus;
  /** How primary visibility was resolved; decides what vocabulary applies. */
  readonly primaryMode: SvoPixelTracePrimaryMode;
  /** Passes that contributed to this record set. */
  readonly stages: readonly SvoPixelTraceStage[];
  readonly pixel: readonly [number, number];
  readonly requestToken: number;
  readonly ray: { readonly origin_m: SvoTraceVec3; readonly direction: SvoTraceVec3 };
  readonly hit?: {
    readonly position_m: SvoTraceVec3;
    readonly geometricNormal: SvoTraceVec3;
    readonly distance_m: number;
    readonly ownerId: number;
    readonly materialId: number;
    readonly featureId: number;
  };
  readonly counters: SvoPixelTraceCounters;
  /** Present exactly when `primaryMode` is `raster`. */
  readonly raster?: SvoPixelTraceRaster;
  /**
   * Whether the two probes agree about where the surface is. Present only on a
   * merged raster trace, where there are two independent answers to compare.
   */
  readonly primaryParity?: SvoPixelTracePrimaryParity;
  /** Present when the frame ran reduced-rate cone lighting. */
  readonly conePrepass?: { readonly active: true };
  readonly globalIllumination?: {
    readonly ready: boolean;
    readonly coneTaps: number;
    readonly coneCount: number;
    readonly visibility: number;
    readonly radiance: SvoTraceVec3;
  };
  readonly records: readonly SvoPixelTraceRecord[];
  /** Records the shader produced but could not store; the prefix stays exact. */
  readonly droppedRecords: number;
  /**
   * Width of a level-0 coverage-pyramid voxel. Absent when the frame reported
   * no usable grid, in which case mip footprints are simply not drawn: a guessed
   * footprint would be a lie about the level a cone sample read.
   */
  readonly minimumVoxel_m?: number;
}

function isKind(value: number): value is SvoPixelTraceKind {
  return Object.values(SVO_PIXEL_TRACE_KINDS).includes(value as SvoPixelTraceKind);
}

function vec3(view: Float32Array, offset: number): SvoTraceVec3 {
  return [view[offset], view[offset + 1], view[offset + 2]];
}

/**
 * Decode one mapped probe buffer. A malformed or superseded buffer returns
 * `undefined` rather than a partly-believable trace: the overlay draws nothing
 * instead of drawing a lie.
 */
export function decodeSvoPixelTrace(words: Uint32Array, floats: Float32Array): SvoPixelTrace | undefined {
  if (words.length < SVO_PIXEL_TRACE_HEADER_WORDS || floats.length < SVO_PIXEL_TRACE_HEADER_WORDS) return undefined;
  const header = SVO_PIXEL_TRACE_HEADER;
  if (words[header.magic] !== SVO_PIXEL_TRACE_MAGIC) return undefined;
  const status = words[header.status];
  if (!Object.values(SVO_PIXEL_TRACE_STATUS).includes(status as SvoPixelTraceStatus)) return undefined;
  // The readback is padded to whole texture rows, so its length is an upper
  // bound on capacity, not the capacity. The shader reports exactly how many
  // records it had to drop, and `produced - dropped` is what it stored.
  const physicalCapacity = Math.floor((words.length - SVO_PIXEL_TRACE_HEADER_WORDS) / SVO_PIXEL_TRACE_RECORD_WORDS);
  const produced = words[header.recordCount];
  const dropped = words[header.droppedRecords];
  const stored = Math.max(0, Math.min(produced - dropped, physicalCapacity));
  const records: SvoPixelTraceRecord[] = [];
  for (let index = 0; index < stored; index += 1) {
    const base = SVO_PIXEL_TRACE_HEADER_WORDS + index * SVO_PIXEL_TRACE_RECORD_WORDS;
    const kind = words[base];
    if (!isKind(kind)) continue;
    records.push({
      kind,
      order: index,
      level: words[base + 1],
      detail: words[base + 2],
      flags: words[base + 3],
      a: vec3(floats, base + 4),
      b: vec3(floats, base + 7),
      tEnter_m: floats[base + 10],
      tExit_m: floats[base + 11],
    });
  }
  const distance_m = floats[header.hitDistance];
  const direction = vec3(floats, header.rayDirection);
  const origin_m = vec3(floats, header.rayOrigin);
  const minimumVoxel = floats[header.minimumVoxel];
  const giState = words[header.giState];
  const raster = words[header.primaryMode] === SVO_PIXEL_TRACE_PRIMARY_MODE.raster;
  const stageBits = words[header.stagesPresent];
  const prepass = words[header.prepassState];
  // A winner is only claimed when a proxy record carries the bit. The header's
  // index alone cannot distinguish "instance 0 won" from "nothing won", and a
  // sky pixel legitimately has no winner at all.
  const winnerRecorded = records.some((record) => record.kind === SVO_PIXEL_TRACE_KINDS.brickProxy
    && (record.flags & SVO_PIXEL_TRACE_FLAGS.depthWinner) !== 0);
  const hit = status === SVO_PIXEL_TRACE_STATUS.hit && Number.isFinite(distance_m) && distance_m > 0
    ? {
      position_m: origin_m.map((value, axis) => value + direction[axis] * distance_m) as unknown as SvoTraceVec3,
      geometricNormal: vec3(floats, header.hitNormal),
      distance_m,
      ownerId: words[header.hitOwnerId],
      materialId: words[header.hitMaterialId],
      featureId: words[header.hitFeatureId],
    }
    : undefined;
  return {
    version: SVO_PIXEL_TRACE_ABI_VERSION,
    status: status as SvoPixelTraceStatus,
    primaryMode: raster ? "raster" : "traced",
    stages: (Object.keys(SVO_PIXEL_TRACE_STAGES) as SvoPixelTraceStage[])
      .filter((stage) => (stageBits & SVO_PIXEL_TRACE_STAGES[stage]) !== 0),
    pixel: [words[header.pixelX], words[header.pixelY]],
    requestToken: words[header.requestToken],
    ray: { origin_m, direction },
    hit,
    counters: {
      nodeVisits: words[header.nodeVisits],
      leafVisits: words[header.leafVisits],
      emptyBrickSkips: words[header.emptyBrickSkips],
      voxelWork: words[header.voxelWork],
      exactTests: words[header.exactTests],
      maximumDepth: words[header.maximumDepth],
      shadowNodeVisits: words[header.shadowNodeVisits],
      shadowLeafVisits: words[header.shadowLeafVisits],
      shadowWork: words[header.shadowWork],
      mipSteps: words[header.mipSteps],
      traversalFailure: words[header.traversalFailure],
      shadedLights: words[header.shadedLights],
    },
    raster: raster ? {
      residentLeaves: words[header.residentLeaves],
      emptyBricks: words[header.emptyBricks],
      frustumCulled: words[header.frustumCulled],
      candidatesEmitted: words[header.candidatesEmitted],
      instancesDrawn: words[header.instancesDrawn],
      coveringProxies: words[header.coveringProxies],
      proxiesWithSurface: words[header.proxiesWithSurface],
      hsrEligibleProxies: words[header.hsrEligibleProxies],
      winner: winnerRecorded
        ? { instanceIndex: words[header.winnerInstanceIndex], sortBucket: words[header.winnerSortBucket] }
        : undefined,
      ddaCellsAcrossProxies: words[header.ddaCellsAcrossProxies],
    } : undefined,
    conePrepass: (prepass & SVO_PIXEL_TRACE_PREPASS_STATE.active) !== 0 ? { active: true } : undefined,
    globalIllumination: (giState & SVO_PIXEL_TRACE_GI_STATE.enabled) !== 0 ? {
      ready: (giState & SVO_PIXEL_TRACE_GI_STATE.ready) !== 0,
      coneTaps: words[header.giConeTaps],
      coneCount: words[header.giConeCount],
      visibility: floats[header.giVisibility],
      radiance: vec3(floats, header.giRadiance),
    } : undefined,
    records,
    droppedRecords: Math.max(0, produced - stored),
    minimumVoxel_m: Number.isFinite(minimumVoxel) && minimumVoxel > 0 ? minimumVoxel : undefined,
  };
}

/**
 * Agreement between the two probes about where this pixel's surface is.
 *
 * The raster probe elects a winner from the frame's own instance list; the
 * lighting probe resolves the same surface the shipping deferred pass does.
 * They are independent routes to one answer, and the renderer's claim is that
 * the raster and traced primaries are bit-identical. So a disagreement here is
 * not noise to be smoothed over — it is that claim failing, and the HUD says so
 * rather than quietly preferring one number.
 */
export interface SvoPixelTracePrimaryParity {
  readonly agrees: boolean;
  readonly rasterDistance_m: number;
  readonly lightingDistance_m: number;
}

/**
 * Record that the frame ran reduced-rate cone lighting.
 *
 * Stamped by the host because only the host knows: the probe is composed inline
 * at full rate, so from inside it the reduced pass beside it is invisible. This
 * matters because it changes what the drawn cones mean — at reduced rate they
 * are the full-rate mirror of visibility this pixel was *given*, not the cones
 * that were marched for it.
 */
export function withSvoPixelTraceConePrepass(
  trace: SvoPixelTrace | undefined,
  active: boolean,
): SvoPixelTrace | undefined {
  if (!trace || !active) return trace;
  return {
    ...trace,
    stages: trace.stages.includes("conePrepass") ? trace.stages : [...trace.stages, "conePrepass"],
    conePrepass: { active: true },
  };
}

/**
 * Fold the raster primary probe's answer into the lighting probe's.
 *
 * Two probes because the renderer has two seams: finding the surface (now the
 * rasterizer's job, explained by `primary`) and lighting it (still a fragment
 * pass, mirrored by `lighting`). Either may be absent — the primary probe does
 * not run in traced mode, and neither has answered on the first frame.
 */
export function mergeSvoPixelTrace(
  lighting: SvoPixelTrace | undefined,
  primary: SvoPixelTrace | undefined,
): SvoPixelTrace | undefined {
  if (!primary) return lighting;
  if (!lighting) return primary;
  // Two probes answering two different pixels cannot be composed into one
  // account of either. This happens while the pointer moves and one readback
  // lands a frame before the other; showing the newer alone beats splicing.
  if (lighting.pixel[0] !== primary.pixel[0] || lighting.pixel[1] !== primary.pixel[1]) {
    return primary.requestToken >= lighting.requestToken ? primary : lighting;
  }
  const records = [...primary.records, ...lighting.records]
    .map((record, order) => ({ ...record, order }));
  const rasterDistance_m = primary.hit?.distance_m ?? Number.POSITIVE_INFINITY;
  const lightingDistance_m = lighting.hit?.distance_m ?? Number.POSITIVE_INFINITY;
  const bothMissed = !primary.hit && !lighting.hit;
  // Compared against the finest voxel, not an absolute epsilon: the two routes
  // solve the same analytic surface but enter its cell from intervals clamped
  // differently, so agreement is "the same surface", not "the same float".
  const tolerance_m = Math.max(1e-4, (lighting.minimumVoxel_m ?? primary.minimumVoxel_m ?? 0.01) * 0.5);
  return {
    ...lighting,
    primaryMode: "raster",
    stages: [...new Set([...primary.stages, ...lighting.stages])],
    raster: primary.raster,
    records,
    droppedRecords: primary.droppedRecords + lighting.droppedRecords,
    minimumVoxel_m: lighting.minimumVoxel_m ?? primary.minimumVoxel_m,
    primaryParity: {
      // A raster winner is only one contributor to the pixel: terrain, a rigid
      // body or a glass pane can legitimately sit in front of every brick, and
      // the lighting probe sees those. Disagreement is only claimed when the
      // raster probe found a surface the lighting probe does not corroborate.
      agrees: bothMissed || !primary.hit || Math.abs(rasterDistance_m - lightingDistance_m) <= tolerance_m
        || lightingDistance_m < rasterDistance_m,
      rasterDistance_m,
      lightingDistance_m,
    },
  };
}

/* ------------------------------------------------------------------------- */
/* Presentation: palette, layers, and the segment geometry the overlay draws. */
/* ------------------------------------------------------------------------- */

/**
 * Layers are what the user toggles. Each owns a colour and a legend line, and
 * every record maps to exactly one layer.
 *
 * This is the union across both primary modes; {@link svoPixelTraceLayersForMode}
 * is the subset a given trace can actually populate. The union is what the
 * visualization registry and the persisted layer selection are keyed on, so
 * switching primary mode never invalidates either.
 */
export const SVO_PIXEL_TRACE_LAYERS = [
  "primary-ray",
  "hierarchy",
  "rejected",
  "bricks",
  "proxies",
  "proxy-losers",
  "winner",
  "cells",
  "exact",
  "terrain",
  "rigid",
  "shadow-rays",
  "cones",
  "gi-cones",
  "prepass",
] as const;

export type SvoPixelTraceLayer = typeof SVO_PIXEL_TRACE_LAYERS[number];

/**
 * Layers each primary mode can populate.
 *
 * `hierarchy`, `rejected` and `bricks` are absent from the raster set rather
 * than renamed into it: nothing in a raster frame corresponds to a per-pixel
 * octree descent, and a layer that draws a plausible fiction is worse than one
 * that is simply not offered. Symmetrically the traced mode has no proxies to
 * show, because it never built any.
 */
const MODE_LAYERS: Readonly<Record<SvoPixelTracePrimaryMode, readonly SvoPixelTraceLayer[]>> = Object.freeze({
  traced: Object.freeze([
    "primary-ray", "hierarchy", "rejected", "bricks", "cells", "exact",
    "terrain", "rigid", "shadow-rays", "cones", "gi-cones", "prepass",
  ]) as readonly SvoPixelTraceLayer[],
  raster: Object.freeze([
    "primary-ray", "proxies", "proxy-losers", "winner", "cells", "exact",
    "terrain", "rigid", "shadow-rays", "cones", "gi-cones", "prepass",
  ]) as readonly SvoPixelTraceLayer[],
});

export function svoPixelTraceLayersForMode(mode: SvoPixelTracePrimaryMode): readonly SvoPixelTraceLayer[] {
  return MODE_LAYERS[mode];
}

export interface SvoPixelTraceLayerDefinition {
  readonly layer: SvoPixelTraceLayer;
  readonly label: string;
  readonly description: string;
  /** Linear-space RGB, matching the overlay's additive compositing. */
  readonly colorLinear: readonly [number, number, number];
  readonly swatch: `#${string}`;
  /** Line width in device pixels before depth ghosting. */
  readonly width_px: number;
}

/** sRGB hex to the linear triple the overlay shader consumes. */
function linearFromHex(hex: string): readonly [number, number, number] {
  const value = Number.parseInt(hex.slice(1), 16);
  const channels = [(value >> 16) & 0xff, (value >> 8) & 0xff, value & 0xff];
  return channels.map((channel) => {
    const normalized = channel / 255;
    return normalized <= 0.04045 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4;
  }) as unknown as readonly [number, number, number];
}

const LAYER_SOURCE: readonly { layer: SvoPixelTraceLayer; label: string; description: string; swatch: `#${string}`; width_px: number }[] = [
  { layer: "primary-ray", label: "Primary ray", description: "The camera ray, solid to the surface that won this pixel and faint beyond it.", swatch: "#f2ab4e", width_px: 2.4 },
  { layer: "hierarchy", label: "Nodes opened", description: "Octree boxes the ray entered and expanded, brightest at the deepest level. Dense traces draw an even sample; the count remains exact.", swatch: "#7fd4ff", width_px: 1.3 },
  { layer: "rejected", label: "Boxes rejected", description: "Children tested by the same slab arithmetic and thrown away. Dense traces draw an even sample; the count remains exact.", swatch: "#e2687f", width_px: 1.0 },
  { layer: "bricks", label: "Leaf bricks", description: "Terminal leaves reached; each holds one brick of cells.", swatch: "#ffe066", width_px: 2.0 },
  { layer: "proxies", label: "Proxies drawn", description: "Every brick proxy whose box covers this pixel — one fragment each, shaded whatever the depth test later decided. Coloured by draw order, cool first to warm last.", swatch: "#7fd4ff", width_px: 1.3 },
  { layer: "proxy-losers", label: "Proxies beaten", description: "Covering proxies that lost: dashed where the brick held no surface and the fragment discarded, dimmed where it found one behind the winner.", swatch: "#e2687f", width_px: 1.0 },
  { layer: "winner", label: "Winning brick", description: "The proxy whose fragment won the depth test, with its full leaf box as a hairline — the gap between them is the published occupancy tightening the draw.", swatch: "#ffe066", width_px: 2.0 },
  { layer: "cells", label: "Brick cells", description: "Fine cells the in-brick DDA stepped through. Dense traces draw an even sample; the count remains exact.", swatch: "#9fb4c0", width_px: 1.0 },
  { layer: "exact", label: "Exact tests", description: "Analytic surface intersections issued from tagged cells.", swatch: "#45c6bc", width_px: 1.8 },
  { layer: "terrain", label: "Terrain march", description: "Heightfield bracket and refinement probes from the background pass every pixel pays.", swatch: "#8fbf6a", width_px: 1.4 },
  { layer: "rigid", label: "Rigid impostors", description: "Analytic body proxies covering this pixel and the intersections they issued.", swatch: "#d98b5f", width_px: 1.6 },
  { layer: "shadow-rays", label: "Shadow rays", description: "Surface-to-light visibility queries, one per light sample.", swatch: "#ff7ad9", width_px: 1.6 },
  { layer: "cones", label: "Cone samples", description: "Coverage-pyramid taps stitched into cones: a ring at the cone's width, the mip voxel it read, coloured by level.", swatch: "#c2a6ff", width_px: 1.2 },
  { layer: "gi-cones", label: "GI hemisphere", description: "Wide diffuse-radiance cones fanned across the upper hemisphere from the shaded surface.", swatch: "#ffb454", width_px: 1.8 },
  { layer: "prepass", label: "Prepass texel", description: "The coarse cone-lighting texel this pixel's visibility was read from, drawn as its footprint on the surface.", swatch: "#6fa8d6", width_px: 1.2 },
];

export const SVO_PIXEL_TRACE_LAYER_DEFINITIONS: Readonly<Record<SvoPixelTraceLayer, SvoPixelTraceLayerDefinition>> = Object.freeze(
  Object.fromEntries(LAYER_SOURCE.map((entry) => [entry.layer, Object.freeze({ ...entry, colorLinear: linearFromHex(entry.swatch) })])) as
    unknown as Record<SvoPixelTraceLayer, SvoPixelTraceLayerDefinition>,
);

/**
 * Default layer for a kind, for the kinds whose layer does not depend on the
 * outcome recorded in their flags. A proxy's layer does, so prefer
 * {@link svoPixelTraceLayerForRecord} wherever a whole record is in hand.
 */
export function svoPixelTraceLayerForKind(kind: SvoPixelTraceKind): SvoPixelTraceLayer {
  switch (kind) {
    case SVO_PIXEL_TRACE_KINDS.hierarchyNode:
    case SVO_PIXEL_TRACE_KINDS.childAccepted: return "hierarchy";
    case SVO_PIXEL_TRACE_KINDS.childRejected: return "rejected";
    case SVO_PIXEL_TRACE_KINDS.leafBrick: return "bricks";
    case SVO_PIXEL_TRACE_KINDS.brickCell: return "cells";
    case SVO_PIXEL_TRACE_KINDS.exactTest:
    case SVO_PIXEL_TRACE_KINDS.primaryHit: return "exact";
    case SVO_PIXEL_TRACE_KINDS.rigidTest:
    case SVO_PIXEL_TRACE_KINDS.rigidProxy: return "rigid";
    case SVO_PIXEL_TRACE_KINDS.terrainStep: return "terrain";
    case SVO_PIXEL_TRACE_KINDS.shadowRay: return "shadow-rays";
    case SVO_PIXEL_TRACE_KINDS.coneSample:
    case SVO_PIXEL_TRACE_KINDS.occlusionConeSample: return "cones";
    case SVO_PIXEL_TRACE_KINDS.globalIlluminationConeSample: return "gi-cones";
    case SVO_PIXEL_TRACE_KINDS.brickProxy: return "proxies";
    case SVO_PIXEL_TRACE_KINDS.leafBounds: return "winner";
    // The pane key is an identity the lighting pass reads, not geometry the
    // primary produced; it is reported in the narrative and drawn by nothing.
    case SVO_PIXEL_TRACE_KINDS.glassPane: return "exact";
    case SVO_PIXEL_TRACE_KINDS.prepassTexel: return "prepass";
  }
}

/**
 * Layer for one record, which for a brick proxy depends on how it ended.
 *
 * Winner, beaten, and discarded are three different costs with three different
 * meanings, and separating them is the whole point of the raster picker: the
 * losers are the overdraw, and the discarded ones are fragments that ran a full
 * DDA for nothing.
 */
export function svoPixelTraceLayerForRecord(record: Pick<SvoPixelTraceRecord, "kind" | "flags">): SvoPixelTraceLayer {
  if (record.kind === SVO_PIXEL_TRACE_KINDS.brickProxy) {
    if ((record.flags & SVO_PIXEL_TRACE_FLAGS.depthWinner) !== 0) return "winner";
    return (record.flags & (SVO_PIXEL_TRACE_FLAGS.depthLoser | SVO_PIXEL_TRACE_FLAGS.discarded)) !== 0
      ? "proxy-losers"
      : "proxies";
  }
  return svoPixelTraceLayerForKind(record.kind);
}

/* ------------------------------------------------------------------------- */
/* Mip levels: the ladder a cone climbs as it widens.                          */
/* ------------------------------------------------------------------------- */

/**
 * One colour per coverage-pyramid level, cool for the fine levels and warm for
 * the coarse ones.
 *
 * A cone tap's level is not a free choice: the march picks the level whose voxel
 * width matches the cone's diameter at that distance, so this ladder is the
 * visual form of "the further the cone travels, the blurrier the thing it is
 * allowed to read". Colouring by level makes that climb legible in one glance,
 * which a single flat cone colour cannot. The last entry covers every level
 * beyond it, so a deep pyramid degrades to "coarsest" instead of wrapping.
 */
export const SVO_PIXEL_TRACE_MIP_SWATCHES: readonly `#${string}`[] = Object.freeze([
  "#8fe8ff", "#a7c4ff", "#c2a6ff", "#dd9df0", "#f0a3cf", "#ffb3a7",
]);

const MIP_COLORS_LINEAR: readonly (readonly [number, number, number])[] =
  Object.freeze(SVO_PIXEL_TRACE_MIP_SWATCHES.map(linearFromHex));

function mipIndex(level: number): number {
  if (!Number.isFinite(level)) return 0;
  return Math.max(0, Math.min(SVO_PIXEL_TRACE_MIP_SWATCHES.length - 1, Math.floor(level)));
}

/** Swatch for one mip level, for the HUD legend. */
export function svoPixelTraceMipSwatch(level: number): `#${string}` {
  return SVO_PIXEL_TRACE_MIP_SWATCHES[mipIndex(level)];
}

/** Overlay colour for one mip level, in the linear space the shader blends in. */
export function svoPixelTraceMipColorLinear(level: number): readonly [number, number, number] {
  return MIP_COLORS_LINEAR[mipIndex(level)];
}

/**
 * Width of the voxel a cone tap at this level read. Level 0 is the finest cell
 * and each level doubles, which is the whole content of "mip level": undefined
 * when the frame did not report a grid width.
 */
export function svoPixelTraceMipFootprint_m(trace: SvoPixelTrace, level: number): number | undefined {
  const minimum = trace.minimumVoxel_m;
  if (!minimum || !Number.isFinite(level)) return undefined;
  return minimum * 2 ** Math.max(0, Math.floor(level));
}

/* ------------------------------------------------------------------------- */
/* Draw order: the ramp a proxy is coloured by.                                */
/* ------------------------------------------------------------------------- */

/**
 * Cool for the first proxy drawn, warm for the last.
 *
 * A proxy's position in the draw order is the only thing the counting sort
 * controls, and the sort is the raster primary's one real performance lever: it
 * exists so tile-based hidden-surface removal gets to reject occluded bricks
 * before shading them. Colouring by order makes the question "did the sort put
 * the winner near the front?" answerable at a glance, which a flat layer colour
 * cannot. Deliberately the same device as the mip ladder above.
 */
export const SVO_PIXEL_TRACE_ORDER_SWATCHES: readonly `#${string}`[] = Object.freeze([
  "#7fd4ff", "#9ec6ff", "#c0b4fb", "#dda6ea", "#f09cc8", "#ff9fa0",
]);

const ORDER_COLORS_LINEAR: readonly (readonly [number, number, number])[] =
  Object.freeze(SVO_PIXEL_TRACE_ORDER_SWATCHES.map(linearFromHex));

/** `position` of `total`, 0-based. A single proxy sits at the cool end. */
function orderRampIndex(position: number, total: number): number {
  if (!Number.isFinite(position) || total <= 1) return 0;
  const span = SVO_PIXEL_TRACE_ORDER_SWATCHES.length - 1;
  return Math.max(0, Math.min(span, Math.round((position / (total - 1)) * span)));
}

export function svoPixelTraceOrderSwatch(position: number, total: number): `#${string}` {
  return SVO_PIXEL_TRACE_ORDER_SWATCHES[orderRampIndex(position, total)];
}

function svoPixelTraceOrderColorLinear(position: number, total: number): readonly [number, number, number] {
  return ORDER_COLORS_LINEAR[orderRampIndex(position, total)];
}

export interface SvoPixelTraceMipRung {
  readonly level: number;
  /** Cone samples that read this level. */
  readonly taps: number;
  /** Voxel width at this level, when the trace reported the grid. */
  readonly footprint_m?: number;
  readonly swatch: `#${string}`;
}

/**
 * The mip levels this trace actually touched, finest first. Empty when the ray
 * took no cone samples, which is the honest answer for an unshaded miss.
 */
export function svoPixelTraceMipLadder(trace: SvoPixelTrace): readonly SvoPixelTraceMipRung[] {
  const taps = new Map<number, number>();
  for (const record of trace.records) {
    if (record.kind !== SVO_PIXEL_TRACE_KINDS.coneSample
      && record.kind !== SVO_PIXEL_TRACE_KINDS.occlusionConeSample
      && record.kind !== SVO_PIXEL_TRACE_KINDS.globalIlluminationConeSample) continue;
    taps.set(record.level, (taps.get(record.level) ?? 0) + 1);
  }
  return [...taps.entries()]
    .sort(([left], [right]) => left - right)
    .map(([level, count]) => ({
      level,
      taps: count,
      footprint_m: svoPixelTraceMipFootprint_m(trace, level),
      swatch: svoPixelTraceMipSwatch(level),
    }));
}

/**
 * One authority for the instance layout, held by the framework this geometry is
 * drawn through. Kept under the old name so the trace's own call sites read in
 * its own vocabulary.
 */
export const SVO_PIXEL_TRACE_SEGMENT_FLOATS = DECORATION_SEGMENT_FLOATS;

export interface SvoPixelTraceGeometryOptions {
  /** Layers to emit. Absent layers cost nothing at draw time. */
  readonly layers?: readonly SvoPixelTraceLayer[];
  /** Ring facets per cone sample. Lower is cheaper; 10 reads as a circle. */
  readonly coneRingFacets?: number;
  /** Uniform width multiplier, so the HUD can thin the overlay on dense rays. */
  readonly widthScale?: number;
  /** Per dense box layer; absent preserves every recorded box. */
  readonly maximumDenseBoxesPerLayer?: number;
}

/**
 * The forensic overlay is explanatory, not the source of its counters. Drawing
 * hundreds of overlapping boxes can cost more than the one-pixel probe itself,
 * so the live visualization samples each dense layer while the HUD continues
 * to report every record written by the shader.
 */
export const SVO_PIXEL_TRACE_VISUALIZATION_MAXIMUM_DENSE_BOXES = 32;

const DENSE_BOX_LAYERS: ReadonlySet<SvoPixelTraceLayer> = new Set([
  "hierarchy", "rejected", "cells",
]);

function sampledDenseBoxOrders(
  trace: SvoPixelTrace,
  enabled: ReadonlySet<SvoPixelTraceLayer>,
  maximumPerLayer: number | undefined,
): ReadonlyMap<SvoPixelTraceLayer, ReadonlySet<number>> {
  if (maximumPerLayer === undefined) return new Map();
  const maximum = Math.max(2, Math.floor(maximumPerLayer));
  const byLayer = new Map<SvoPixelTraceLayer, SvoPixelTraceRecord[]>();
  for (const record of trace.records) {
    const layer = svoPixelTraceLayerForRecord(record);
    if (!enabled.has(layer) || !DENSE_BOX_LAYERS.has(layer)) continue;
    const records = byLayer.get(layer);
    if (records) records.push(record); else byLayer.set(layer, [record]);
  }
  const sampled = new Map<SvoPixelTraceLayer, ReadonlySet<number>>();
  for (const [layer, records] of byLayer) {
    if (records.length <= maximum) continue;
    const orders = new Set<number>();
    // Preserve endpoints and any record whose flags identify a positive result.
    orders.add(records[0].order);
    orders.add(records[records.length - 1].order);
    const important = records.filter((record) =>
      (record.flags & (SVO_PIXEL_TRACE_FLAGS.hit | SVO_PIXEL_TRACE_FLAGS.tagged)) !== 0);
    const importantSlots = Math.max(0, maximum - orders.size);
    for (let slot = 0; slot < importantSlots && important.length > 0; slot += 1) {
      orders.add(important[Math.round((slot / Math.max(1, importantSlots - 1)) * (important.length - 1))].order);
    }
    // Even coverage keeps the nested hierarchy and the DDA path legible without
    // biasing the picture toward whichever records happened to be written first.
    for (let slot = 0; slot < maximum && orders.size < maximum; slot += 1) {
      orders.add(records[Math.round((slot / Math.max(1, maximum - 1)) * (records.length - 1))].order);
    }
    if (orders.size < maximum) {
      for (const record of records) {
        orders.add(record.order);
        if (orders.size >= maximum) break;
      }
    }
    sampled.set(layer, orders);
  }
  return sampled;
}

export interface SvoPixelTraceGeometry {
  /** Interleaved per-instance segment data; see SVO_PIXEL_TRACE_SEGMENT_FLOATS. */
  readonly segments: Float32Array<ArrayBuffer>;
  readonly segmentCount: number;
  /** Highest `order` emitted, so the reveal animation can normalize against it. */
  readonly orderCount: number;
  readonly countsByLayer: Readonly<Record<SvoPixelTraceLayer, number>>;
}

const BOX_EDGES: readonly (readonly [number, number])[] = Object.freeze([
  [0, 1], [1, 3], [3, 2], [2, 0],
  [4, 5], [5, 7], [7, 6], [6, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
]);

/** Diagonal of a box, used to scale dash periods to whatever the box is. */
function distanceBetween(from: SvoTraceVec3, to: SvoTraceVec3): number {
  return Math.hypot(to[0] - from[0], to[1] - from[1], to[2] - from[2]);
}

function boxCorner(minimum: SvoTraceVec3, maximum: SvoTraceVec3, index: number): SvoTraceVec3 {
  return [
    (index & 1) === 0 ? minimum[0] : maximum[0],
    (index & 2) === 0 ? minimum[1] : maximum[1],
    (index & 4) === 0 ? minimum[2] : maximum[2],
  ];
}

function orthonormalBasis(direction: SvoTraceVec3): readonly [SvoTraceVec3, SvoTraceVec3] {
  const length = Math.hypot(...direction) || 1;
  const forward: SvoTraceVec3 = [direction[0] / length, direction[1] / length, direction[2] / length];
  const helper: SvoTraceVec3 = Math.abs(forward[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
  const tangent: SvoTraceVec3 = [
    helper[1] * forward[2] - helper[2] * forward[1],
    helper[2] * forward[0] - helper[0] * forward[2],
    helper[0] * forward[1] - helper[1] * forward[0],
  ];
  const tangentLength = Math.hypot(...tangent) || 1;
  const unitTangent: SvoTraceVec3 = [tangent[0] / tangentLength, tangent[1] / tangentLength, tangent[2] / tangentLength];
  const bitangent: SvoTraceVec3 = [
    forward[1] * unitTangent[2] - forward[2] * unitTangent[1],
    forward[2] * unitTangent[0] - forward[0] * unitTangent[2],
    forward[0] * unitTangent[1] - forward[1] * unitTangent[0],
  ];
  return [unitTangent, bitangent];
}

/** Per-segment styling. Everything the overlay needs beyond the two endpoints. */
interface SegmentStyle {
  readonly layer: SvoPixelTraceLayer;
  readonly order: number;
  readonly intensity: number;
  readonly kind: number;
  readonly dash_m?: number;
  readonly widthBoost?: number;
  /**
   * Draw a screen-space arrowhead at `end`, sized in device pixels. Directional
   * work is drawn with one: a ray that shows where it is going explains the
   * traversal in a way a bare line cannot.
   */
  readonly arrow?: boolean;
  /** Overrides the layer colour. The mip ladder is the only user. */
  readonly colorLinear?: readonly [number, number, number];
}

/**
 * Arrowhead size relative to line width, and the bounds it is clamped into.
 * Constant in device pixels, so a head neither swamps a near ray nor vanishes
 * on a far one.
 */
const ARROW_HEAD_SCALE = 3.4;
const ARROW_HEAD_MINIMUM_PX = 4.5;
const ARROW_HEAD_MAXIMUM_PX = 13;

/** Cone axis azimuths used to stitch consecutive taps into a cone silhouette. */
const CONE_ENVELOPE_AZIMUTHS: readonly number[] = Object.freeze([0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2]);

/**
 * Identity of the cone march a tap belongs to.
 *
 * `detail` names the march, and the anchored bit separates a split shadow
 * march's two phases: the second walks back from the emitter, so stitching it to
 * the first would draw one cone that reverses direction halfway along itself.
 */
function coneKey(record: SvoPixelTraceRecord): string {
  return `${record.kind}:${record.detail}:${record.flags & SVO_PIXEL_TRACE_FLAGS.emitterAnchored}`;
}

/**
 * Flatten a trace into camera-facing segment instances.
 *
 * Every visual is a segment: boxes become their twelve edges, cone samples
 * become a ring in the plane normal to the cone axis, and the hit becomes a
 * small three-axis cross plus a ring on its tangent plane. One instance layout
 * therefore draws the whole diagnostic in a single instanced call. Directional
 * segments emit a second instance for their arrowhead, which the vertex shader
 * expands in screen space rather than in the world, so a head stays the same
 * size and never folds away when the ray points at the camera.
 */
export function buildSvoPixelTraceGeometry(
  trace: SvoPixelTrace,
  options: SvoPixelTraceGeometryOptions = {},
): SvoPixelTraceGeometry {
  const enabled = new Set<SvoPixelTraceLayer>(options.layers ?? SVO_PIXEL_TRACE_LAYERS);
  const facets = Math.max(3, Math.min(24, Math.round(options.coneRingFacets ?? 10)));
  const widthScale = Number.isFinite(options.widthScale) ? Math.max(0.1, Math.min(4, options.widthScale!)) : 1;
  const values: number[] = [];
  const countsByLayer = Object.fromEntries(SVO_PIXEL_TRACE_LAYERS.map((layer) => [layer, 0])) as Record<SvoPixelTraceLayer, number>;
  const orderCount = Math.max(1, trace.records.length);
  const visibleDenseBoxOrders = sampledDenseBoxOrders(
    trace, enabled, options.maximumDenseBoxesPerLayer,
  );

  const push = (start: SvoTraceVec3, end: SvoTraceVec3, style: SegmentStyle) => {
    const definition = SVO_PIXEL_TRACE_LAYER_DEFINITIONS[style.layer];
    const color = style.colorLinear ?? definition.colorLinear;
    const width_px = definition.width_px * widthScale * (style.widthBoost ?? 1);
    const intensity = Math.max(0, Math.min(1, style.intensity));
    const order = style.order / orderCount;
    const emit = (head_px: number) => {
      values.push(
        start[0], start[1], start[2], width_px,
        end[0], end[1], end[2], order,
        color[0], color[1], color[2], intensity,
        style.kind, style.dash_m ?? 0, head_px, 0,
      );
      countsByLayer[style.layer] += 1;
    };
    emit(0);
    // The head is its own instance built from the same two endpoints: the shaft
    // keeps its dash and its thickness, and the shader turns the head instance
    // into a solid screen-space triangle at `end`.
    if (style.arrow) {
      emit(Math.max(ARROW_HEAD_MINIMUM_PX, Math.min(ARROW_HEAD_MAXIMUM_PX, width_px * ARROW_HEAD_SCALE)));
    }
  };

  const pushBox = (minimum: SvoTraceVec3, maximum: SvoTraceVec3, style: SegmentStyle) => {
    for (const [from, to] of BOX_EDGES) {
      push(boxCorner(minimum, maximum, from), boxCorner(minimum, maximum, to), style);
    }
  };

  const pushCube = (center: SvoTraceVec3, width_m: number, style: SegmentStyle) => {
    if (!(width_m > 0)) return;
    const half = width_m / 2;
    pushBox(
      [center[0] - half, center[1] - half, center[2] - half],
      [center[0] + half, center[1] + half, center[2] + half],
      style,
    );
  };

  const ringPoint = (
    center: SvoTraceVec3,
    tangent: SvoTraceVec3,
    bitangent: SvoTraceVec3,
    radius_m: number,
    angle: number,
  ): SvoTraceVec3 => [
    center[0] + (tangent[0] * Math.cos(angle) + bitangent[0] * Math.sin(angle)) * radius_m,
    center[1] + (tangent[1] * Math.cos(angle) + bitangent[1] * Math.sin(angle)) * radius_m,
    center[2] + (tangent[2] * Math.cos(angle) + bitangent[2] * Math.sin(angle)) * radius_m,
  ];

  const pushRing = (center: SvoTraceVec3, axis: SvoTraceVec3, radius_m: number, style: SegmentStyle) => {
    if (!(radius_m > 0)) return;
    const [tangent, bitangent] = orthonormalBasis(axis);
    let previous: SvoTraceVec3 | undefined;
    for (let facet = 0; facet <= facets; facet += 1) {
      const point = ringPoint(center, tangent, bitangent, radius_m, (facet / facets) * Math.PI * 2);
      if (previous) push(previous, point, style);
      previous = point;
    }
  };

  const { origin_m, direction } = trace.ray;
  const along = (t: number): SvoTraceVec3 => [
    origin_m[0] + direction[0] * t,
    origin_m[1] + direction[1] * t,
    origin_m[2] + direction[2] * t,
  ];

  // The primary ray is an arrow chain, not one long line. Each arrow is one
  // stretch the ray actually advanced through — a brick it walked, or the empty
  // space it skipped to reach the next one — so the picture carries the march
  // order and its direction, which is the thing a static line hides.
  //
  // Under the raster primary there is no such march to draw: nothing walked this
  // ray, so the chain would be an invention. The ray is drawn as one arrow to
  // the surface that won, and the depth ladder below carries the ordering story
  // instead — which is the honest form of it, because the proxies covering a
  // pixel were shaded concurrently rather than in sequence.
  const rasterPrimary = trace.primaryMode === "raster";
  if (enabled.has("primary-ray") && rasterPrimary) {
    const terminus_m = trace.hit?.distance_m
      ?? Math.max(1, ...trace.records
        .filter((record) => record.kind === SVO_PIXEL_TRACE_KINDS.brickProxy)
        .map((record) => record.tExit_m));
    push(origin_m, along(terminus_m), {
      layer: "primary-ray", order: 0, intensity: 1, kind: SVO_PIXEL_TRACE_KINDS.primaryHit,
      widthBoost: 1.35, arrow: true,
    });
    if (!trace.hit) {
      push(along(terminus_m), along(terminus_m * 1.35 + 1), {
        layer: "primary-ray", order: orderCount, intensity: 0.4, kind: SVO_PIXEL_TRACE_KINDS.primaryHit,
        dash_m: 0.3, widthBoost: 0.7, arrow: true,
      });
    }
  }
  if (enabled.has("primary-ray") && !rasterPrimary) {
    const rayStyle = (order: number, intensity: number, dash_m: number, widthBoost: number): SegmentStyle => ({
      layer: "primary-ray", order, intensity, kind: SVO_PIXEL_TRACE_KINDS.primaryHit, dash_m, widthBoost, arrow: true,
    });
    const boxes = trace.records.filter((record) => record.kind === SVO_PIXEL_TRACE_KINDS.hierarchyNode
      || record.kind === SVO_PIXEL_TRACE_KINDS.leafBrick);
    const entry_m = Math.max(0, boxes.length > 0 ? Math.min(...boxes.map((record) => record.tEnter_m)) : 0);
    const terminus_m = trace.hit?.distance_m
      ?? Math.max(entry_m, ...boxes.map((record) => record.tExit_m), entry_m + 1);
    push(origin_m, along(entry_m), rayStyle(0, 0.5, 0.22, 0.7));
    // Bricks are the ray's real stepping stones: the traversal descends to find
    // them, then walks them in near-to-far order.
    const stride = Math.max(1e-5, (terminus_m - entry_m) * 1e-3);
    let cursor_m = entry_m;
    for (const record of trace.records) {
      if (record.kind !== SVO_PIXEL_TRACE_KINDS.leafBrick) continue;
      const from_m = Math.max(cursor_m, record.tEnter_m);
      const to_m = Math.min(terminus_m, record.tExit_m);
      if (to_m <= from_m + stride) continue;
      // A gap means the traversal proved the space between two bricks empty and
      // jumped it; dashed, because nothing was examined along it.
      if (from_m > cursor_m + stride) push(along(cursor_m), along(from_m), rayStyle(record.order, 0.45, 0.16, 0.8));
      push(along(from_m), along(to_m), rayStyle(record.order, 1, 0, 1.35));
      cursor_m = to_m;
    }
    if (cursor_m < terminus_m - stride) push(along(cursor_m), along(terminus_m), rayStyle(orderCount, 1, 0, 1.35));
    if (!trace.hit) {
      push(along(terminus_m), along(terminus_m * 1.35 + 1), rayStyle(orderCount, 0.4, 0.3, 0.7));
    }
  }

  // Cone taps are stitched into cones. Grouping first is what lets consecutive
  // taps of the same march be joined rim to rim: on its own a tap is a ring, and
  // a scatter of rings does not read as a cone widening with distance.
  const coneGroups = new Map<string, SvoPixelTraceRecord[]>();
  if (enabled.has("cones") || enabled.has("gi-cones")) {
    for (const record of trace.records) {
      if (record.kind !== SVO_PIXEL_TRACE_KINDS.coneSample
        && record.kind !== SVO_PIXEL_TRACE_KINDS.occlusionConeSample
        && record.kind !== SVO_PIXEL_TRACE_KINDS.globalIlluminationConeSample) continue;
      const group = coneGroups.get(coneKey(record));
      if (group) group.push(record); else coneGroups.set(coneKey(record), [record]);
    }
    for (const group of coneGroups.values()) {
      const coneLayer = svoPixelTraceLayerForKind(group[0].kind);
      if (!enabled.has(coneLayer)) continue;
      const axis = group[0].b;
      const [tangent, bitangent] = orthonormalBasis(axis);
      const first = group[0];
      if (coneLayer === "gi-cones" && trace.hit) {
        const apexStyle: SegmentStyle = {
          layer: coneLayer, order: first.order, intensity: 0.8, kind: first.kind, widthBoost: 1.05,
        };
        for (const azimuth of CONE_ENVELOPE_AZIMUTHS) {
          push(trace.hit.position_m, ringPoint(first.a, tangent, bitangent, first.tEnter_m, azimuth), apexStyle);
        }
      }
      for (let index = 1; index < group.length; index += 1) {
        const previous = group[index - 1], current = group[index];
        const style: SegmentStyle = {
          layer: coneLayer, order: current.order,
          intensity: coneLayer === "gi-cones" ? 0.78 : 0.34, kind: current.kind,
          widthBoost: coneLayer === "gi-cones" ? 1.1 : 0.75,
          colorLinear: coneLayer === "gi-cones" ? undefined : svoPixelTraceMipColorLinear(current.level),
        };
        for (const azimuth of CONE_ENVELOPE_AZIMUTHS) {
          push(
            ringPoint(previous.a, tangent, bitangent, previous.tEnter_m, azimuth),
            ringPoint(current.a, tangent, bitangent, current.tEnter_m, azimuth),
            style,
          );
        }
      }
      // The axis, arrowed. Only for occlusion marches: a shadow march already has
      // its shadow ray drawn along the same line, and two lines on one axis is
      // clutter, but an occlusion fan has nothing else to say where it pointed.
      const last = group[group.length - 1];
      if (last !== first && (last.kind === SVO_PIXEL_TRACE_KINDS.occlusionConeSample
        || last.kind === SVO_PIXEL_TRACE_KINDS.globalIlluminationConeSample)) {
        const axisLength_m = Math.hypot(last.a[0] - first.a[0], last.a[1] - first.a[1], last.a[2] - first.a[2]);
        push(coneLayer === "gi-cones" && trace.hit ? trace.hit.position_m : first.a, last.a, {
          layer: coneLayer, order: last.order, intensity: coneLayer === "gi-cones" ? 0.9 : 0.5, kind: last.kind,
          dash_m: Math.max(1e-4, axisLength_m / 6), widthBoost: coneLayer === "gi-cones" ? 1.2 : 0.9, arrow: true,
          colorLinear: coneLayer === "gi-cones" ? undefined : svoPixelTraceMipColorLinear(last.level),
        });
      }
    }
  }

  // Proxies are coloured by their rank among the proxies covering *this pixel*,
  // not by their absolute instance index: the reader is comparing the handful
  // that competed here, and a global index would put them all in one colour.
  const proxyRank = new Map<number, number>();
  const proxies = trace.records.filter((record) => record.kind === SVO_PIXEL_TRACE_KINDS.brickProxy);
  proxies.forEach((record, index) => proxyRank.set(record.order, index));

  // The depth ladder: one rung per covering proxy, drawn across the ray at the
  // interval that proxy's fragment clamped to. Read along the ray it is the
  // tournament in profile — how many boxes competed, how deep they stacked, and
  // where the winner sat among them.
  if (rasterPrimary && proxies.length > 0) {
    const [tangent, bitangent] = orthonormalBasis(direction);
    const rung_m = Math.max(1e-4, (trace.minimumVoxel_m ?? 0.05) * 1.6);
    for (const record of proxies) {
      const layer = svoPixelTraceLayerForRecord(record);
      if (!enabled.has(layer)) continue;
      const rank = proxyRank.get(record.order) ?? 0;
      const winner = (record.flags & SVO_PIXEL_TRACE_FLAGS.depthWinner) !== 0;
      const discarded = (record.flags & SVO_PIXEL_TRACE_FLAGS.discarded) !== 0;
      const style: SegmentStyle = {
        layer, order: record.order, kind: record.kind,
        intensity: winner ? 1 : discarded ? 0.4 : 0.62,
        widthBoost: winner ? 1.5 : 0.9,
        dash_m: discarded ? rung_m / 4 : 0,
        colorLinear: svoPixelTraceOrderColorLinear(rank, proxies.length),
      };
      // A tick across the ray at entry, and a shaft spanning the interval the
      // fragment was allowed to walk.
      const entry = along(record.tEnter_m);
      push(
        [entry[0] - tangent[0] * rung_m, entry[1] - tangent[1] * rung_m, entry[2] - tangent[2] * rung_m],
        [entry[0] + tangent[0] * rung_m, entry[1] + tangent[1] * rung_m, entry[2] + tangent[2] * rung_m],
        style,
      );
      const offset = rung_m * (0.35 + 0.55 * rank);
      const shaftFrom = along(record.tEnter_m), shaftTo = along(record.tExit_m);
      push(
        [shaftFrom[0] + bitangent[0] * offset, shaftFrom[1] + bitangent[1] * offset, shaftFrom[2] + bitangent[2] * offset],
        [shaftTo[0] + bitangent[0] * offset, shaftTo[1] + bitangent[1] * offset, shaftTo[2] + bitangent[2] * offset],
        { ...style, widthBoost: (style.widthBoost ?? 1) * 0.8 },
      );
    }
  }

  for (const record of trace.records) {
    const layer = svoPixelTraceLayerForRecord(record);
    if (!enabled.has(layer)) continue;
    const visibleOrders = visibleDenseBoxOrders.get(layer);
    if (visibleOrders && !visibleOrders.has(record.order)) continue;
    switch (record.kind) {
      case SVO_PIXEL_TRACE_KINDS.hierarchyNode:
        // Deeper boxes are the interesting ones; fade the shallow ancestors.
        pushBox(record.a, record.b, {
          layer, order: record.order, kind: record.kind,
          intensity: 0.34 + 0.66 * Math.min(1, record.level / Math.max(1, trace.counters.maximumDepth)),
        });
        break;
      case SVO_PIXEL_TRACE_KINDS.childAccepted:
        pushBox(record.a, record.b, {
          layer, order: record.order, kind: record.kind, widthBoost: 0.85,
          intensity: (record.flags & SVO_PIXEL_TRACE_FLAGS.descended) !== 0 ? 0.9 : 0.5,
          dash_m: (record.flags & SVO_PIXEL_TRACE_FLAGS.deferred) !== 0 ? 0.18 : 0,
        });
        break;
      case SVO_PIXEL_TRACE_KINDS.childRejected:
        pushBox(record.a, record.b, {
          layer, order: record.order, kind: record.kind, intensity: 0.55, dash_m: 0.13, widthBoost: 0.8,
        });
        break;
      case SVO_PIXEL_TRACE_KINDS.leafBrick:
        pushBox(record.a, record.b, {
          layer, order: record.order, kind: record.kind, widthBoost: 1.15,
          intensity: (record.flags & SVO_PIXEL_TRACE_FLAGS.emptySkip) !== 0 ? 0.55 : 1,
        });
        break;
      case SVO_PIXEL_TRACE_KINDS.brickCell:
        pushBox(record.a, record.b, {
          layer, order: record.order, kind: record.kind,
          intensity: (record.flags & SVO_PIXEL_TRACE_FLAGS.tagged) !== 0 ? 1 : 0.42,
        });
        break;
      case SVO_PIXEL_TRACE_KINDS.exactTest:
        pushBox(record.a, record.b, {
          layer, order: record.order, kind: record.kind, widthBoost: 1.2,
          intensity: (record.flags & SVO_PIXEL_TRACE_FLAGS.hit) !== 0 ? 1 : 0.6,
        });
        break;
      case SVO_PIXEL_TRACE_KINDS.primaryHit: {
        const normal = record.b;
        const radius_m = Math.max(1e-4, record.tExit_m);
        const style: SegmentStyle = { layer, order: record.order, intensity: 1, kind: record.kind };
        pushRing(record.a, normal, radius_m, style);
        push(record.a, [
          record.a[0] + normal[0] * radius_m * 2.2,
          record.a[1] + normal[1] * radius_m * 2.2,
          record.a[2] + normal[2] * radius_m * 2.2,
        ], { ...style, widthBoost: 1.4, arrow: true });
        break;
      }
      case SVO_PIXEL_TRACE_KINDS.rigidTest:
      case SVO_PIXEL_TRACE_KINDS.terrainStep:
        push(record.a, record.b, {
          layer, order: record.order, intensity: 0.7, kind: record.kind, dash_m: 0.1, widthBoost: 0.8, arrow: true,
        });
        break;
      case SVO_PIXEL_TRACE_KINDS.brickProxy: {
        // The box the rasterizer drew. Its colour is where it sat in the draw
        // order; its treatment is how it ended.
        const winner = (record.flags & SVO_PIXEL_TRACE_FLAGS.depthWinner) !== 0;
        const discarded = (record.flags & SVO_PIXEL_TRACE_FLAGS.discarded) !== 0;
        const uncertain = (record.flags & SVO_PIXEL_TRACE_FLAGS.hsrEligible) !== 0;
        pushBox(record.a, record.b, {
          layer, order: record.order, kind: record.kind,
          colorLinear: svoPixelTraceOrderColorLinear(proxyRank.get(record.order) ?? 0, Math.max(1, proxies.length)),
          widthBoost: winner ? 1.25 : 0.85,
          // A fragment the tiler may have killed is drawn fainter than one that
          // certainly ran: the picture must not assert work it cannot observe.
          intensity: (winner ? 1 : discarded ? 0.45 : 0.62) * (uncertain ? 0.6 : 1),
          dash_m: discarded ? Math.max(1e-4, distanceBetween(record.a, record.b) / 12) : 0,
        });
        break;
      }
      case SVO_PIXEL_TRACE_KINDS.leafBounds:
        // Hairline, so the occupied sub-box nested inside it stays the subject.
        pushBox(record.a, record.b, {
          layer, order: record.order, kind: record.kind, intensity: 0.4, widthBoost: 0.55,
          dash_m: Math.max(1e-4, distanceBetween(record.a, record.b) / 16),
        });
        break;
      case SVO_PIXEL_TRACE_KINDS.rigidProxy:
        pushBox(record.a, record.b, {
          layer, order: record.order, kind: record.kind, widthBoost: 0.9,
          intensity: (record.flags & SVO_PIXEL_TRACE_FLAGS.hit) !== 0 ? 1 : 0.5,
          dash_m: (record.flags & SVO_PIXEL_TRACE_FLAGS.hit) !== 0 ? 0 : 0.12,
        });
        break;
      case SVO_PIXEL_TRACE_KINDS.prepassTexel: {
        // The coarse texel's footprint where it lands on the surface, plus the
        // segment from the shaded point to it — the provenance of the visibility
        // this pixel was given rather than the visibility it computed.
        pushBox(record.a, record.b, {
          layer, order: record.order, kind: record.kind, widthBoost: 0.8,
          intensity: (record.flags & SVO_PIXEL_TRACE_FLAGS.hit) !== 0 ? 0.85 : 0.45,
          dash_m: (record.flags & SVO_PIXEL_TRACE_FLAGS.hit) !== 0
            ? Math.max(1e-4, distanceBetween(record.a, record.b) / 10) : 0,
        });
        break;
      }
      // Identity the lighting pass reads, not geometry the primary produced.
      case SVO_PIXEL_TRACE_KINDS.glassPane: break;
      case SVO_PIXEL_TRACE_KINDS.shadowRay: {
        const occluded = (record.flags & SVO_PIXEL_TRACE_FLAGS.hit) !== 0;
        push(record.a, record.b, {
          layer, order: record.order, kind: record.kind, widthBoost: 1.2, arrow: true,
          intensity: occluded ? 1 : 0.55, dash_m: occluded ? 0 : 0.16,
        });
        break;
      }
      case SVO_PIXEL_TRACE_KINDS.coneSample:
      case SVO_PIXEL_TRACE_KINDS.occlusionConeSample:
      case SVO_PIXEL_TRACE_KINDS.globalIlluminationConeSample: {
        // `b` is the march direction, tEnter the cone half-width at this sample,
        // tExit the coverage it read. Coverage drives brightness, so the taps
        // that actually occlude are the ones that stand out; the colour is the
        // mip level, so the climb from fine to coarse is visible along the cone.
        const coverage = Math.max(0, Math.min(1, record.tExit_m));
        const colorLinear = svoPixelTraceMipColorLinear(record.level);
        pushRing(record.a, record.b, record.tEnter_m, {
          layer, order: record.order, kind: record.kind,
          intensity: record.kind === SVO_PIXEL_TRACE_KINDS.globalIlluminationConeSample ? 0.82 : 0.28 + 0.72 * coverage,
          colorLinear: record.kind === SVO_PIXEL_TRACE_KINDS.globalIlluminationConeSample ? undefined : colorLinear,
        });
        // The voxel the tap actually read, at its true width: the cone picked
        // this level because the level's voxel is about as wide as the cone is
        // here, so the cube sitting just inside the ring *is* the LOD choice.
        // Centred on the tap rather than snapped to the pyramid's grid, which
        // this side of the ABI does not know.
        const footprint_m = svoPixelTraceMipFootprint_m(trace, record.level);
        if (footprint_m !== undefined) {
          pushCube(record.a, footprint_m, {
            layer, order: record.order, kind: record.kind, colorLinear, widthBoost: 0.7,
            intensity: 0.18 + 0.42 * coverage, dash_m: footprint_m / 5,
          });
        }
        break;
      }
    }
  }

  return {
    segments: Float32Array.from(values),
    segmentCount: values.length / SVO_PIXEL_TRACE_SEGMENT_FLOATS,
    orderCount,
    countsByLayer,
  };
}

/* ------------------------------------------------------------------------- */
/* Narrative: the ordered story of one ray, for the HUD.                      */
/* ------------------------------------------------------------------------- */

export interface SvoPixelTraceNarrativeStep {
  readonly id: string;
  readonly label: string;
  readonly detail: string;
  readonly value: string;
  readonly layer?: SvoPixelTraceLayer;
  /**
   * This step's figures describe one camera's worth of work, not this pixel's.
   * The HUD badges them separately and never adds them to per-pixel totals.
   */
  readonly frameWide?: boolean;
}

function metres(value: number): string {
  if (!Number.isFinite(value)) return "—";
  return value >= 100 ? `${value.toFixed(0)} m` : value >= 1 ? `${value.toFixed(2)} m` : `${(value * 100).toFixed(1)} cm`;
}

function integer(value: number): string {
  return Number.isFinite(value) ? Math.max(0, Math.round(value)).toLocaleString() : "—";
}

/**
 * The raster primary's account of itself: a ladder of passes, in encoder order.
 *
 * The first step is deliberately badged as frame-wide. The cull walks every
 * resident leaf once per camera, not once per pixel, and quietly presenting its
 * counters beside per-pixel ones would overstate what this pixel cost by orders
 * of magnitude.
 */
function rasterNarrative(trace: SvoPixelTrace, kinds: Map<number, number>): SvoPixelTraceNarrativeStep[] {
  const raster = trace.raster;
  const cells = kinds.get(SVO_PIXEL_TRACE_KINDS.brickCell) ?? 0;
  const exact = kinds.get(SVO_PIXEL_TRACE_KINDS.exactTest) ?? 0;
  const steps: SvoPixelTraceNarrativeStep[] = [];
  if (raster) {
    steps.push({
      id: "cull", label: "Emit and cull bricks", layer: "proxies", frameWide: true,
      detail: `${integer(raster.residentLeaves)} resident leaves, less ${integer(raster.emptyBricks)} empty`
        + ` and ${integer(raster.frustumCulled)} outside the frustum, sorted into ${SVO_PIXEL_TRACE_SORT_BUCKETS} depth buckets`,
      value: `${integer(raster.instancesDrawn)} drawn`,
    });
  }
  // The terrain record states the interval the secant march bracketed and how it
  // ended. Not a step count: producing one honestly would need a counter inside
  // the shared heightfield sampler, and that string is fingerprint-gated.
  const terrain = trace.records.find((record) => record.kind === SVO_PIXEL_TRACE_KINDS.terrainStep);
  steps.push({
    id: "background", label: "Draw the background", layer: "terrain",
    detail: terrain
      ? `the heightfield march every pixel pays, bracketed over ${metres(terrain.tEnter_m)}–${metres(terrain.tExit_m)};`
        + ((terrain.flags & SVO_PIXEL_TRACE_FLAGS.hit) !== 0
          ? " its depth is what the bricks then had to beat"
          : " it found no ground, so the bricks competed against the far plane")
      : "no terrain in this scene; the background pass wrote the miss encoding and cleared the depth",
    value: terrain ? ((terrain.flags & SVO_PIXEL_TRACE_FLAGS.hit) !== 0 ? metres(terrain.tExit_m) : "miss") : "none",
  });
  if (raster) {
    // The one number worth reading: how many fragments this pixel cost.
    const certain = Math.max(0, raster.coveringProxies - raster.hsrEligibleProxies);
    const bound = raster.hsrEligibleProxies > 0 ? `${integer(certain)}–${integer(raster.coveringProxies)}` : integer(raster.coveringProxies);
    steps.push({
      id: "proxies", label: "Rasterize brick proxies", layer: "proxies",
      detail: `${integer(raster.proxiesWithSurface)} of them found a surface and wrote a depth;`
        + ` the rest ran a full DDA and discarded`
        + (raster.hsrEligibleProxies > 0
          ? `. ${integer(raster.hsrEligibleProxies)} sat behind the winner, so the tiler may have killed them before shading`
          : ""),
      value: `${bound} proxies`,
    });
    steps.push({
      id: "winner", label: "Resolve the depth test", layer: "winner",
      detail: raster.winner
        ? `instance ${integer(raster.winner.instanceIndex)} of ${integer(raster.instancesDrawn)} drawn, from depth bucket ${integer(raster.winner.sortBucket)}`
          + ` — leaves partition space, so nearest wins outright`
        : "no proxy won: this pixel is background, terrain, or a body",
      value: raster.winner ? `#${integer(raster.winner.instanceIndex)}` : "none",
    });
    steps.push({
      id: "cells", label: "Walk the winning brick", layer: "cells",
      detail: raster.ddaCellsAcrossProxies > cells
        ? `${integer(raster.ddaCellsAcrossProxies)} cells were stepped in total across every covering proxy, winner and beaten alike`
        : "a bounded DDA inside one 8³ brick; a tagged cell is a maybe, never a yes",
      value: `${integer(cells)} cells`,
    });
  }
  steps.push({
    id: "exact", label: "Solve the surface", layer: "exact",
    detail: trace.hit
      ? `hit at ${metres(trace.hit.distance_m)}, owner ${trace.hit.ownerId === 0xffff ? "none" : trace.hit.ownerId}, material ${trace.hit.materialId}`
      : "no analytic surface at this pixel",
    value: `${integer(exact)} tests`,
  });
  const rigidProxies = kinds.get(SVO_PIXEL_TRACE_KINDS.rigidProxy) ?? 0;
  if (rigidProxies > 0) {
    steps.push({
      id: "rigid", label: "Rasterize rigid impostors", layer: "rigid",
      detail: "analytic body proxies covering this pixel, depth-tested against the bricks",
      value: `${integer(rigidProxies)} proxies`,
    });
  }
  return steps;
}

/** Human-readable account of what the shader did, in the order it did it. */
export function svoPixelTraceNarrative(trace: SvoPixelTrace): readonly SvoPixelTraceNarrativeStep[] {
  const counters = trace.counters;
  const kinds = new Map<number, number>();
  for (const record of trace.records) kinds.set(record.kind, (kinds.get(record.kind) ?? 0) + 1);
  const rejected = kinds.get(SVO_PIXEL_TRACE_KINDS.childRejected) ?? 0;
  const accepted = kinds.get(SVO_PIXEL_TRACE_KINDS.childAccepted) ?? 0;
  const cells = kinds.get(SVO_PIXEL_TRACE_KINDS.brickCell) ?? 0;
  const visibilityCones = (kinds.get(SVO_PIXEL_TRACE_KINDS.coneSample) ?? 0)
    + (kinds.get(SVO_PIXEL_TRACE_KINDS.occlusionConeSample) ?? 0);
  const giCones = kinds.get(SVO_PIXEL_TRACE_KINDS.globalIlluminationConeSample) ?? 0;
  const shadows = kinds.get(SVO_PIXEL_TRACE_KINDS.shadowRay) ?? 0;
  // Only the primary differs between modes. Everything from the surface onward
  // is one deferred lighting pass either way, so the tail below is shared.
  const steps: SvoPixelTraceNarrativeStep[] = trace.primaryMode === "raster"
    ? rasterNarrative(trace, kinds)
    : [
      {
        id: "descend", label: "Descend the hierarchy", layer: "hierarchy",
        detail: `${accepted} child boxes entered, ${rejected} rejected by the slab test`,
        value: `${counters.nodeVisits} nodes`,
      },
      {
        id: "leaves", label: "Reach leaf bricks", layer: "bricks",
        detail: counters.emptyBrickSkips > 0
          ? `${counters.emptyBrickSkips} crossed without a surface before the hit`
          : "front-to-back order let the first occupied brick finish the ray",
        value: `${counters.leafVisits} bricks`,
      },
      {
        id: "cells", label: "Walk leaf bricks", layer: "cells",
        detail: `${cells} recorded cells across ${counters.leafVisits} reached brick${counters.leafVisits === 1 ? "" : "s"}`
          + (counters.leafVisits > 0
            ? ` (${(counters.voxelWork / counters.leafVisits).toFixed(1)} steps per brick on average); a tagged cell is a maybe, never a yes`
            : "; a tagged cell is a maybe, never a yes"),
        value: `${counters.voxelWork} cells`,
      },
      {
        id: "exact", label: "Solve the surface", layer: "exact",
        detail: trace.hit
          ? `hit at ${metres(trace.hit.distance_m)}, owner ${trace.hit.ownerId === 0xffff ? "none" : trace.hit.ownerId}, material ${trace.hit.materialId}`
          : "no analytic surface along this ray",
        value: `${counters.exactTests} tests`,
      },
    ];
  if (trace.conePrepass) {
    steps.push({
      id: "prepass", label: "Read the cone prepass", layer: "prepass",
      // Deliberately not claiming which texel served this pixel: the reduced
      // shader rejects boundary pixels and resolves them inline, and only that
      // shader knows which happened here. The probe runs at full rate beside it.
      detail: "the frame ran cone lighting at reduced rate, so some of this pixel's visibility"
        + " was read from a coarse texel rather than marched for it; the cones drawn below are the"
        + " full-rate mirror, which is the upper bound on what it cost",
      value: "reduced",
    });
  }
  if (shadows > 0 || counters.shadowNodeVisits > 0) {
    steps.push({
      id: "shadow", label: "Query visibility", layer: "shadow-rays",
      detail: `${counters.shadowNodeVisits} hierarchy nodes and ${counters.shadowLeafVisits} bricks for exact visibility`,
      value: `${shadows} rays`,
    });
  }
  if (trace.globalIllumination) {
    const gi = trace.globalIllumination;
    const radiance = gi.radiance.map((channel) => Math.max(0, channel).toFixed(2)).join(" / ");
    steps.push({
      id: "gi", label: "Gather global illumination", layer: "gi-cones",
      detail: gi.ready
        ? `${gi.coneCount} wide cones across the upper hemisphere; bounced RGB ${radiance}; broad diffuse visibility ${Math.round(Math.max(0, Math.min(1, gi.visibility)) * 100)}%`
        : "GLOBAL was selected, but the generation-matched tetrahedral radiance atlas was unavailable",
      value: gi.ready ? `${gi.coneCount} cones · ${gi.coneTaps} taps` : "fallback",
    });
  }
  if (visibilityCones > 0 || giCones > 0 || counters.mipSteps > 0) {
    const ladder = svoPixelTraceMipLadder(trace);
    const coarsest = ladder[ladder.length - 1];
    steps.push({
      id: "cones", label: "March the coverage pyramid", layer: visibilityCones > 0 ? "cones" : "gi-cones",
      // The level is the whole story of a cone tap: the wider the cone has grown
      // by the time it reaches a sample, the blurrier the level it is allowed to
      // read, and the cheaper that sample is.
      detail: coarsest && ladder[0]
        ? `${ladder.length > 1 ? `levels ${ladder[0].level}–${coarsest.level}` : `level ${coarsest.level}`}: the wider the cone, the coarser the voxel it reads`
        : "soft shadows and ambient occlusion read blur levels, not geometry",
      value: `${counters.mipSteps} taps`,
    });
  }
  if (counters.traversalFailure > 0) {
    steps.push({
      id: "failure", label: "Bounded work exhausted",
      detail: counters.traversalFailure >= 2
        ? "invalid topology or payload data; the renderer fails closed"
        : "a budget ceiling stopped this ray before it terminated",
      value: counters.traversalFailure >= 2 ? "invalid" : "exhausted",
    });
  }
  return steps;
}

/** Total drawable work, used for the HUD's headline number. */
export function svoPixelTraceTotalWork(trace: SvoPixelTrace): number {
  return svoPixelTraceStageCosts(trace).reduce((total, stage) => total + stage.work, 0);
}

/**
 * One entry per pass that spent per-pixel work on this pixel, in encoder order.
 *
 * Deliberately per-pixel only: the brick cull is frame-wide and is excluded, so
 * the bar sums to something a reader can honestly compare between two pixels.
 * Units are work items — cells stepped, rays issued, taps taken — not time; the
 * probe counts what the shader did, and turning that into milliseconds would be
 * a guess this diagnostic has no basis for.
 */
export interface SvoPixelTraceStageCost {
  readonly id: string;
  readonly label: string;
  readonly layer: SvoPixelTraceLayer;
  readonly work: number;
}

export function svoPixelTraceStageCosts(trace: SvoPixelTrace): readonly SvoPixelTraceStageCost[] {
  const counters = trace.counters;
  const stages: SvoPixelTraceStageCost[] = [];
  // Terrain is deliberately absent: its cost is a march this probe brackets but
  // does not count, and inventing a work figure to fill a bar segment would make
  // every other segment a lie by comparison. It is reported as an outcome.
  if (trace.primaryMode === "raster") {
    // Every covering proxy's DDA, not just the winner's: the beaten fragments
    // are the overdraw, and hiding them would make the pass look free.
    const primary = trace.raster?.ddaCellsAcrossProxies ?? counters.voxelWork;
    if (primary > 0) stages.push({ id: "primary", label: "Brick raster", layer: "proxies", work: primary });
  } else {
    const primary = counters.nodeVisits + counters.leafVisits + counters.voxelWork;
    if (primary > 0) stages.push({ id: "primary", label: "Traversal", layer: "hierarchy", work: primary });
  }
  if (counters.exactTests > 0) {
    stages.push({ id: "exact", label: "Exact tests", layer: "exact", work: counters.exactTests });
  }
  const shadow = counters.shadowNodeVisits + counters.shadowLeafVisits;
  if (shadow > 0) stages.push({ id: "shadow", label: "Shadow rays", layer: "shadow-rays", work: shadow });
  if (counters.mipSteps > 0) {
    stages.push({ id: "cones", label: "Cone taps", layer: "cones", work: counters.mipSteps });
  }
  return stages;
}

/* ------------------------------------------------------------------------- */
/* Pinning: turning one viewport click into one frozen ray.                    */
/* ------------------------------------------------------------------------- */

/**
 * Traces the probe may answer with before a click's request is abandoned.
 *
 * The probe traces one pixel per frame and reads it back the frame after, so the
 * clicked pixel comes back within two traces on any working path. A generous
 * ceiling on *traces* rather than on milliseconds is what keeps this honest on a
 * heavy scene, where one frame can take seconds.
 */
export const SVO_PIXEL_TRACE_PIN_PATIENCE = 8;

/** A click waiting for its own pixel to come back before the ray freezes. */
export interface SvoPixelTracePinRequest {
  /** Viewport fractions of the click, 0..1 from the top-left. */
  readonly normalizedX: number;
  readonly normalizedY: number;
  /** The camera the click was aimed from; any move invalidates the aim. */
  readonly cameraKey: string;
  /** Trace revision at the click, so patience is counted in traces. */
  readonly revision: number;
}

/**
 * What one viewport click asks the diagnostic for: this pixel.
 *
 * A click is a re-aim, not a toggle. It used to be a toggle, on the argument
 * that once a ray is pinned the camera is free to orbit away from the pixel that
 * produced it, so the pixel under a second click bears no relation to the frozen
 * ray. True, and it is an argument for re-aiming: the second click names a new
 * ray, from where the camera now stands, and freezing *that* is what was asked.
 * Going live instead spent every other click doing nothing a reader could see,
 * which is unaffordable now that the click is how a pixel is chosen at all — see
 * the probe's claim on the press in `WebGPUViewport`. Releasing without aiming
 * somewhere new is the HUD's Unpin button, and putting the probe down.
 *
 * A request already in flight is simply replaced. The one in flight is the
 * pixel the reader has since moved on from, and the handshake only ever pins a
 * request whose own pixel came back — see `resolveSvoPixelTracePin`.
 */
export function svoPixelTracePinClick(state: {
  readonly normalizedX: number;
  readonly normalizedY: number;
  readonly cameraKey: string;
  readonly revision: number;
}): { readonly request: SvoPixelTracePinRequest } {
  return {
    request: {
      normalizedX: Math.max(0, Math.min(1, state.normalizedX)),
      normalizedY: Math.max(0, Math.min(1, state.normalizedY)),
      cameraKey: state.cameraKey,
      revision: state.revision,
    },
  };
}

export type SvoPixelTracePinResolution =
  /** The requested pixel came back; freeze it. */
  | "pin"
  /** Still in flight. */
  | "wait"
  /** It will not be answered as asked; follow the pointer again. */
  | "abandon";

/**
 * Whether an in-flight pin request has landed, become moot, or is still waiting.
 *
 * `answered` must mean the newest decoded trace *is* the pixel this request
 * asked about. Pinning on any earlier trace would freeze a neighbouring ray,
 * because the readback present at the click was requested a frame or two before
 * it. A moved camera abandons the request for the same reason a pinned ray stops
 * probing: the clicked pixel now names a different ray, so honouring the click
 * would freeze a ray the user never aimed at.
 */
export function resolveSvoPixelTracePin(
  request: SvoPixelTracePinRequest,
  state: {
    readonly answered: boolean;
    readonly cameraKey: string;
    /** False when the probe cannot produce a trace at all on this frame. */
    readonly probeCanAnswer: boolean;
    readonly revision: number;
  },
): SvoPixelTracePinResolution {
  if (state.cameraKey !== request.cameraKey) return "abandon";
  if (state.answered) return "pin";
  if (!state.probeCanAnswer) return "abandon";
  return state.revision - request.revision >= SVO_PIXEL_TRACE_PIN_PATIENCE ? "abandon" : "wait";
}

/**
 * What to do with a pinned ray when the scene it was traced against changes —
 * another light, a republished topology, a new traversal budget, or simply the
 * simulation advancing a step.
 *
 * A pin freezes one recorded ray. Scene motion marks that answer stale, but it
 * never turns the pin into an implicit stream of new probes; the user can unpin
 * and sample again when a new frame is wanted.
 */
export function resolveSvoPixelTracePinnedFrame(state: {
  readonly pinned: boolean;
  readonly sceneChanged: boolean;
}): { readonly stale: boolean } {
  return { stale: state.pinned && state.sceneChanged };
}

/* ------------------------------------------------------------------------- */
/* Visualization: the ray probe, joining the shared overlay by wrapping.      */
/* ------------------------------------------------------------------------- */

/**
 * One registry entry per traced layer.
 *
 * The geometry builder above is not rewritten to join the framework: it already
 * emits the shared instance layout in world metres, so each layer's decoration
 * builds only its own segments and concatenates them into the frame's single
 * buffer. That is the whole migration for a producer that predates the registry.
 *
 * One entry per layer rather than one for the whole trace, because the layers
 * are what a reader turns on and off — and declaring them here is what lets the
 * HUD's buttons and their swatches be a read of the catalog rather than a second
 * hand-maintained list.
 */
/**
 * The pass each layer's work belongs to, matching the labels the frame encoder
 * emits. A layer that named the wrong pass would send a reader to the wrong line
 * of the profile.
 */
const LAYER_PASS: Readonly<Record<SvoPixelTraceLayer, string>> = Object.freeze({
  "primary-ray": "SVO primary visibility",
  hierarchy: "SVO primary visibility",
  rejected: "SVO primary visibility",
  bricks: "SVO primary visibility",
  proxies: "SVO primary brick raster",
  "proxy-losers": "SVO primary brick raster",
  winner: "SVO primary brick raster",
  cells: "SVO primary brick raster",
  exact: "SVO primary visibility",
  terrain: "SVO primary background and terrain",
  rigid: "SVO analytic rigid discovery",
  "shadow-rays": "SVO deferred dry lighting",
  cones: "SVO deferred dry lighting",
  "gi-cones": "SVO deferred dry lighting",
  prepass: "SVO cone-lighting prepass",
});

export const svoPixelTraceVisualizations: readonly Visualization[] = Object.freeze(
  SVO_PIXEL_TRACE_LAYERS.map((layer) => {
    const definition = SVO_PIXEL_TRACE_LAYER_DEFINITIONS[layer];
    return decorationVisualization<SvoPixelTrace>({
      kind: "decoration",
      id: `svo-traversal/${layer}`,
      pass: LAYER_PASS[layer],
      group: layer,
      label: definition.label,
      swatch: definition.swatch,
      description: definition.description,
      source: "One re-traced camera ray, recorded in execution order by the shipping traversal law",
      legend: [{ swatch: definition.swatch, label: definition.label }],
      accepts: (subject): subject is SvoPixelTrace =>
        hasFields(subject, ["records", "ray", "pixel"])
        && Array.isArray((subject as SvoPixelTrace).records),
      // The record list is the picture. Its length plus the trace's identity is
      // enough: a probe answering the same pixel with the same record count has
      // not changed what this layer draws.
      key: (trace, context) =>
        `${trace.pixel.join("_")}.${trace.requestToken}.${trace.records.length}.${context.widthScale}`,
      build(trace, context, into) {
        const geometry = buildSvoPixelTraceGeometry(trace, {
          layers: [layer], widthScale: context.widthScale,
          maximumDenseBoxesPerLayer: SVO_PIXEL_TRACE_VISUALIZATION_MAXIMUM_DENSE_BOXES,
        });
        // World metres already, so the builder's lattice transform is bypassed.
        into.append(geometry.segments, geometry.segmentCount);
        return [];
      },
    });
  }),
);
