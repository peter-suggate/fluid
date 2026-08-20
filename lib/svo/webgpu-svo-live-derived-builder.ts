import {
  SPARSE_BRICK_GPU_LAYOUT,
  SPARSE_BRICK_NO_OWNER,
  SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII,
  sparseBrickBandedLeafCodecWGSL,
  sparseBrickSceneGeometryCodecWGSL,
  sparseBrickSceneIdentityCodecWGSL,
  type SparseBrickLeafPayloadMode,
  type SparseBrickOctreeGPU,
  type SparseBrickPayloadProfileName,
  type SparseBrickPlan,
  type SparseBrickSceneGeometryFormat,
  type SparseBrickSize,
} from "./sparse-brick-octree";
import { SVO_BRICK_LIFECYCLE } from "./svo-brick-occupancy";
import { completeCooperativeBuild } from "../core/cooperative-build";
import { svoEnvironmentLightingWGSL } from "./svo-environment-lighting";
import { svoLightWGSL } from "./svo-light-abi";
import { svoMaterialWGSL } from "./svo-material-abi";
import {
  SVO_NODE_MIP_LAYOUT,
  SVO_NODE_MIP_OPACITY_STORAGE,
  raiseSvoNodeMipSeedToFloor,
  svoNodeMipOpacityChannels,
  svoNodeMipSeedKey,
  type SvoNodeMipCoordinate,
  type SvoNodeMipSeedPage,
} from "./svo-node-mip-pyramid";
import { svoTetrahedralRadianceWGSL } from "./svo-tetrahedral-radiance";
import { VOXEL_MATERIAL_IDS } from "../core/voxel-scene";
import {
  svoBandedReconstructionEnabled,
  svoSolidDirectOcclusionEnabled,
  svoSolidMarchOffsetFixEnabled,
} from "./svo-banded-leaf-payload";
import { liveSvoDerivedPageValidityWGSL } from "./webgpu-svo-live-derived-cache";
import type { WebGpuLiveSvoNodeMipGpuTarget } from "./webgpu-svo-node-mip-pyramid";
import {
  LIVE_SVO_RADIANCE_STORAGE,
  liveSvoRadianceBytesPerTexel,
  type WebGpuLiveSvoTetrahedralRadianceGpuTarget,
} from "./webgpu-svo-tetrahedral-radiance";

export const LIVE_SVO_DERIVED_WORKLIST = Object.freeze({
  countWord: 0,
  generationWord: 1,
  levelWord: 2,
  recordOffsetWord: 3,
  dispatchIndirectOffsetBytes: 16,
  headerWords: 8,
  recordWords: 12,
  destinationSlotWord: 0,
  sourcePageCoordinateWord: 1,
  sourceLeafWord: 4,
  /**
   * One record layout at every level: slot, page coordinate, eight children.
   *
   * The page coordinate used to be carried only by the base level and by the
   * radiance floor, because only those two needed to locate the cells they
   * cover. Every parent needs it now: a child page that is absent is no longer
   * proof of empty space — it can be the interior of a coarse leaf that
   * publishes at its own level — and reading that leaf means turning a texel
   * back into a world cell. Twelve words hold exactly one of each.
   */
  childSlotWord: 4,
  /** Historic alias; the radiance floor's layout is now every level's layout. */
  radianceBaseChildSlotWord: 4,
  invalidIndex: 0xffff_ffff,
  /**
   * A child slot that names a *leaf* rather than a page.
   *
   * `emitPage` resolves an absent child page once, at plan time, into the
   * deepest active leaf covering it — which, because a page is emitted for
   * every leaf at that leaf's own level, is always a leaf whose extent contains
   * the whole child page. The reduction then samples that leaf's voxels
   * directly instead of averaging in a zero it would have no way to distinguish
   * from air. Slots are physical atlas indices and never reach the top bit.
   */
  leafSlotTag: 0x8000_0000,
} as const);

export const LIVE_SVO_RADIANCE_FEEDBACK = Object.freeze({
  /**
   * On, now that the radiance floor makes it affordable.
   *
   * The historic objection was cost and in-place mutation: the base pass ran
   * once per finest voxel, so a garden paid a full-resolution diffuse solve to
   * move a low-frequency field. With radiance capped three levels above the
   * leaf the base pass runs on ~500x fewer texels, and a phase that lands
   * mid-rotation now perturbs a 5 cm irradiance sample rather than a voxel.
   */
  enabledByDefault: true,
  phaseCount: 4,
  /**
   * At the 0.85 transport ceiling, 24 rotations bound the unpropagated fixed-
   * point residual below 2.1% (0.85^24). Static scenes then go exactly idle.
   */
  settleCycleCount: 24,
  settleFrameCount: 96,
  directionCount: 4,
  distanceCells: [2.5, 6, 18, 54] as const,
  /** Pure-white display materials remain contractive in the transport solve. */
  maximumTransportAlbedo: 0.85,
} as const);

/**
 * One fixed GPU-authored worklist per virtual mip level. Records must be unique
 * by destination slot. Level zero supplies its virtual page coordinate and
 * resolves the deepest active leaf per texel; parent records supply eight child
 * slots in xyz-bit order. Dispatch args cover count * 10^3 threads.
 */
export interface LiveSvoDerivedGpuWorklist {
  buffer: GPUBuffer;
  capacity: number;
  bindingOffsetBytes?: number;
  bindingSizeBytes?: number;
  indirectOffsetBytes?: number;
}

export interface WebGpuLiveSvoDerivedBuilderOptions {
  tree: SparseBrickOctreeGPU;
  nodeMips: WebGpuLiveSvoNodeMipGpuTarget;
  radiance: WebGpuLiveSvoTetrahedralRadianceGpuTarget;
  /** vec4f per material ID: linear emissive RGB plus unused alpha. */
  materialEmission: GPUBuffer;
  /** Canonical SVO material table used for diffuse transport albedo. */
  materialPbr?: GPUBuffer;
  /** Selected image-free environment record used as the open-cone source. */
  environmentLighting?: GPUBuffer;
  /** Bounded authored light table; the first four records feed diffuse transport. */
  lights?: GPUBuffer;
  lightCount?: number;
  worklists: readonly LiveSvoDerivedGpuWorklist[];
  /** GPU publication generation used to certify newly allocated empty pages. */
  generationSource: LiveSvoDerivedGenerationSource;
  /** Number of address-plan slots resident in the fixed atlas. */
  plannedPageCount: number;
  /** Finest canonical tree level used for cell-to-leaf traversal. */
  finestLevel: number;
  worldOrigin_m?: readonly [number, number, number];
  cellSize_m?: readonly [number, number, number];
  /** Previous-frame radiance feedback. Enabled when both optional lighting buffers are supplied. */
  radianceFeedback?: boolean;
  /** Optional compact scratch page grid; defaults to the smallest grid that fits the largest level worklist. */
  scratchAtlasPages?: readonly [number, number, number];
  /** Finest level that owns a radiance page. See `SVO_RADIANCE_LEVEL_FLOOR`. */
  radianceFloorLevel?: number;
  /**
   * Worklist depth the deepest *radiance* level can reach.
   *
   * The radiance scratch used to be sized like the opacity scratch, which meant
   * one 16 kB page per level-zero page — as much memory as the whole radiance
   * atlas, spent on a staging buffer. Above the floor a level holds hundreds of
   * pages, not tens of thousands, and this is what says so.
   */
  radianceScratchCapacity?: number;
  label?: string;
}

export interface LiveSvoDirtyLeafSource {
  buffer: GPUBuffer;
  countOffsetBytes: number;
  recordOffsetBytes: number;
  capacity: number;
  /** Source records contain leafIndex in word zero and use this stride. */
  recordStrideWords: number;
}

/** One nonzero, monotonically increasing generation shared by every dirty lane. */
export interface LiveSvoDerivedGenerationSource {
  buffer: GPUBuffer;
  offsetBytes: number;
}

export interface WebGpuLiveSvoDerivedWorklistPlannerOptions {
  tree: SparseBrickOctreeGPU;
  nodeMips: WebGpuLiveSvoNodeMipGpuTarget;
  /** Scene, fluid, and topology dirty lanes are unioned before any page build. */
  dirtyLeafSources: readonly LiveSvoDirtyLeafSource[];
  generationSource: LiveSvoDerivedGenerationSource;
  levelCount: number;
  finestLevel: number;
  /**
   * Worklist depth each level may reach, as a vector or as one figure for all.
   *
   * A vector is what the address plan supplies (`pageCapacityByLevel`), and the
   * arena lays its sections out by prefix sum over it. A scalar is the old
   * uniform shape, kept because it is the natural thing for a fixture to say.
   */
  pageCapacityPerLevel: number | readonly number[];
  /** Finest level that owns a radiance page; its records carry a page coordinate. */
  radianceFloorLevel?: number;
  label?: string;
}

/**
 * Where a coarse leaf's data lives in the pyramid, and the A/B that measures it.
 *
 * `own-level` (the default) gives every leaf exactly one page, at the level
 * whose texels are its voxels. `extent` is the shape that shipped: a coarse
 * leaf publishes a base page for every finest brick it covers, holding the same
 * `brickSize^3` values `8^p` times over. The two differ in what the *base*
 * level holds inside a coarse leaf and in nothing else — every parent level is
 * exact under both, because a reduction whose child page is absent reads the
 * leaf directly (`LIVE_SVO_DERIVED_WORKLIST.leafSlotTag`).
 *
 * Read once at module load, like `FLUID_SVO_NODE_MIP_APRON`, so the WGSL and
 * the CPU plan cannot disagree about which shape this process is building.
 */
export const LIVE_SVO_COARSE_LEAF_PAGES_ENVIRONMENT_VARIABLE = "FLUID_SVO_COARSE_LEAF_PAGES";

export function liveSvoCoarseLeafPagePolicy(
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): "own-level" | "extent" {
  const raw = environment?.[LIVE_SVO_COARSE_LEAF_PAGES_ENVIRONMENT_VARIABLE];
  if (raw === undefined || raw === "") return "own-level";
  if (raw !== "own-level" && raw !== "extent") {
    throw new RangeError(`${LIVE_SVO_COARSE_LEAF_PAGES_ENVIRONMENT_VARIABLE} must be "own-level" or "extent"`);
  }
  return raw;
}

const COARSE_LEAF_PAGES = liveSvoCoarseLeafPagePolicy();

/** CPU mirror of the GPU dirty-leaf coverage used by generic planning tests. */
export function liveSvoLeafBasePages(options: {
  coordinate: readonly [number, number, number];
  leafLevel: number;
  finestLevel: number;
  brickSize: SparseBrickSize;
}): readonly (readonly [number, number, number])[] {
  const { coordinate, leafLevel, finestLevel, brickSize } = options;
  if (!Number.isInteger(leafLevel) || !Number.isInteger(finestLevel) || leafLevel < 0 || leafLevel > finestLevel || finestLevel > 21) {
    throw new RangeError("Live derived leaf levels must be ordered integers in [0, 21]");
  }
  if (coordinate.some((value) => !Number.isSafeInteger(value) || value < 0)) {
    throw new RangeError("Live derived leaf coordinate must be nonnegative safe integers");
  }
  const scale = 2 ** (finestLevel - leafLevel);
  const first = coordinate.map((value) => Math.floor(value * scale * brickSize / SVO_NODE_MIP_LAYOUT.interiorSize));
  const last = coordinate.map((value) => Math.floor(((value + 1) * scale * brickSize - 1) / SVO_NODE_MIP_LAYOUT.interiorSize));
  const pages: [number, number, number][] = [];
  for (let z = first[2]; z <= last[2]; z += 1) for (let y = first[1]; y <= last[1]; y += 1) for (let x = first[0]; x <= last[0]; x += 1) {
    pages.push([x, y, z]);
  }
  return pages;
}

/**
 * The one pyramid page a leaf owns, at the level whose texels are its voxels.
 *
 * A leaf spans `2^(finest - level) * brickSize` cells per axis and holds
 * `brickSize^3` voxels. A virtual page at mip level `p` spans `8 * 2^p` cells
 * over `8^3` texels, so the level where one texel is exactly one voxel is
 * `p = log2(cells / 8)` — and at that level the leaf *is* one page, because a
 * leaf's origin is aligned to its own extent.
 *
 * {@link liveSvoLeafBasePages} instead expanded every leaf to every base page
 * in its extent, which made the page count a function of claim *volume*: a
 * coarse leaf published `8^p` pages holding `brickSize^3` distinct values
 * between them, i.e. the same data `8^p` times. On `hero-garden-hose` at
 * environment refinement depth 3 that was 58 % of all pages from 0.59 % of the
 * leaves, and it is the entire reason the pyramid grew ~7.9x per halving of
 * the leaf while the ground's surface grew ~3.6x.
 *
 * Nothing is lost by publishing at the leaf's own level, because nothing finer
 * exists to lose: the levels beneath hold no page, and a parent whose child
 * page is absent now reads the leaf directly (`childSample`'s tagged fallback
 * in the build shader). What a *cone* loses is a level-0 sample in the interior
 * of a coarse region, which is by construction a region the topology declined
 * to resolve.
 *
 * A brick smaller than a page (`brickSize` 4 at the finest level) has no level
 * of its own and still lands in the base page containing it, exactly as before.
 */
export function liveSvoLeafPage(options: {
  coordinate: readonly [number, number, number];
  leafLevel: number;
  finestLevel: number;
  brickSize: SparseBrickSize;
}): SvoNodeMipSeedPage {
  const { coordinate, leafLevel, finestLevel, brickSize } = options;
  if (!Number.isInteger(leafLevel) || !Number.isInteger(finestLevel) || leafLevel < 0 || leafLevel > finestLevel || finestLevel > 21) {
    throw new RangeError("Live derived leaf levels must be ordered integers in [0, 21]");
  }
  if (coordinate.some((value) => !Number.isSafeInteger(value) || value < 0)) {
    throw new RangeError("Live derived leaf coordinate must be nonnegative safe integers");
  }
  const interior = SVO_NODE_MIP_LAYOUT.interiorSize;
  const cells = 2 ** (finestLevel - leafLevel) * brickSize;
  if (cells < interior) {
    const cellOrigin = coordinate.map((value) => value * cells);
    return { level: 0, coordinate: cellOrigin.map((value) => Math.floor(value / interior)) as unknown as SvoNodeMipCoordinate };
  }
  const level = Math.log2(cells / interior);
  if (!Number.isInteger(level)) throw new RangeError("Live derived leaf extent must be a power-of-two multiple of the page interior");
  return { level, coordinate: [...coordinate] as unknown as SvoNodeMipCoordinate };
}

/**
 * Virtual pages the canonical sparse topology claims, each at its own level.
 *
 * One page per leaf — see {@link liveSvoLeafPage}. Empty space with no leaf
 * still never consumes a physical opacity/radiance slot.
 *
 * `floorLevel` raises every seed below it to the floor
 * ({@link SVO_OPACITY_LEVEL_FLOOR}). Because a raised seed lands on a page the
 * plan's ancestor walk would have produced anyway, the result at every level at
 * or above the floor is *identical* to the unfloored plan — the only difference
 * is that the levels beneath it are gone. Zero is the shipped behaviour.
 *
 * The GPU worklist needs no matching parameter: `emitPage` asks the direct page
 * table for a slot, a floored plan gives levels below the floor a zero-depth
 * slab, and the lookup answers `INVALID`. The page coordinate it carries is
 * halved per level from the seed either way, so the first level that *does*
 * resolve is the floor, holding the same coordinate this function computes.
 */
export function liveSvoPlanBasePages(
  plan: SparseBrickPlan,
  policy: "own-level" | "extent" = COARSE_LEAF_PAGES,
  floorLevel = 0,
): readonly SvoNodeMipSeedPage[] {
  return completeCooperativeBuild(liveSvoPlanBasePagesSteps(plan, policy, floorLevel));
}

/**
 * The same seeds, offered as slices.
 *
 * One pass over every planned leaf, and the curvature refinement now plans
 * 441k of them at environment refinement depth 3 — 45 ms per 112k measured, so
 * roughly 175 ms held whole. The opacity floor collapses many of those leaves
 * onto one seed, which shrinks what the *address* plan then costs but not what
 * this pass does: it still visits every leaf to find that out.
 */
export function* liveSvoPlanBasePagesSteps(
  plan: SparseBrickPlan,
  policy: "own-level" | "extent" = COARSE_LEAF_PAGES,
  floorLevel = 0,
): Generator<unknown, readonly SvoNodeMipSeedPage[], undefined> {
  const unique = new Map<string, SvoNodeMipSeedPage>();
  const add = (seed: SvoNodeMipSeedPage): void => {
    const page = floorLevel > 0 ? raiseSvoNodeMipSeedToFloor(seed, floorLevel) : seed;
    unique.set(svoNodeMipSeedKey(page), page);
  };
  let visited = 0;
  for (const leaf of plan.leaves) {
    if ((visited += 1) % 4096 === 0) yield;
    const node = plan.nodes[leaf.nodeIndex];
    if (!node) throw new RangeError(`Live derived leaf ${leaf.index} references a missing node`);
    const request = {
      coordinate: [node.coordinate.x, node.coordinate.y, node.coordinate.z] as const,
      leafLevel: node.level,
      finestLevel: plan.maximumDepth,
      brickSize: plan.brickSize,
    };
    if (policy === "extent") {
      for (const at of liveSvoLeafBasePages(request)) {
        add({ level: 0, coordinate: at as unknown as SvoNodeMipCoordinate });
      }
      continue;
    }
    add(liveSvoLeafPage(request));
  }
  return [...unique.values()];
}

/**
 * `Params` below: 3 vec4u, two 12-word tables, then 12 per-level capacities.
 * Stated once so the host write and the struct cannot drift apart.
 */
export const LIVE_SVO_DERIVED_PLANNER_PARAMS_BYTES = (4 * 3 + 12 + 12 + 12) * 4;

export const liveSvoDerivedWorklistWGSL = /* wgsl */ `
// \`capacities\` is per level, because the arena's sections are: a level holds at
// most its own grid's pages, so laying every section out at the deepest level's
// depth spent 48 B a record on capacity the coarse levels can never reach.
// vec4u rather than array<u32,12> so the element alignment is unambiguously
// legal in the uniform address space.
struct Params{source:vec4u,domain:vec4u,limits:vec4u,zOffsets:array<u32,12>,sections:array<u32,12>,capacities:array<vec4u,3>}
@group(0) @binding(0) var<storage,read> source:array<u32>;
@group(0) @binding(1) var<storage,read> control:array<u32>;
@group(0) @binding(2) var<storage,read> topology:array<u32>;
@group(0) @binding(3) var directTable:texture_3d<u32>;
@group(0) @binding(4) var<storage,read_write> claims:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> output:array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params:Params;
@group(0) @binding(7) var<storage,read> generationState:array<u32>;
const INVALID:u32=0xffffffffu;const ACTIVE:u32=${SVO_BRICK_LIFECYCLE.activeBit}u;const RECORD_WORDS:u32=${LIVE_SVO_DERIVED_WORKLIST.recordWords}u;
fn levelCapacity(level:u32)->u32{return params.capacities[level/4u][level%4u];}
fn keyBit(lo:u32,hi:u32,bit:u32)->u32{if(bit>=32u){return(hi>>(bit-32u))&1u;}return(lo>>bit)&1u;}
fn morton(lo:u32,hi:u32,level:u32)->vec3u{var p=vec3u(0u);for(var bit=0u;bit<level;bit+=1u){let s=1u<<bit;p+=vec3u(keyBit(lo,hi,3u*bit),keyBit(lo,hi,3u*bit+1u),keyBit(lo,hi,3u*bit+2u))*s;}return p;}
fn tableSlot(level:u32,p:vec3u)->u32{if(level>=params.domain.y||p.x>=params.domain.z||p.y>=params.domain.w){return INVALID;}let dims=textureDimensions(directTable);let levelStart=params.zOffsets[level];var levelEnd=dims.z;if(level+1u<params.domain.y){levelEnd=params.zOffsets[level+1u];}if(p.z>=levelEnd-levelStart){return INVALID;}let z=levelStart+p.z;let value=textureLoad(directTable,vec3i(i32(p.x),i32(p.y),i32(z)),0).x;return select(INVALID,value-1u,value>0u);}
fn deepestLeaf(globalCell:vec3u)->u32{let brickSize=control[11];if(brickSize==0u||control[0]==0u){return INVALID;}let brick=globalCell/brickSize;var node=0u;var selected=INVALID;
 for(var level=0u;level<=params.domain.x&&node<control[0];level+=1u){let nodeBase=node*8u;let leaf=topology[nodeBase+6u];if(leaf<control[1]&&(topology[nodeBase+7u]&ACTIVE)!=0u){selected=leaf;}if(level==params.domain.x){break;}let bit=params.domain.x-level-1u;let octant=((brick.x>>bit)&1u)|(((brick.y>>bit)&1u)<<1u)|(((brick.z>>bit)&1u)<<2u);let mask=topology[nodeBase+3u]&0xffu;if((mask&(1u<<octant))==0u){break;}node=topology[nodeBase+4u]+countOneBits(mask&((1u<<octant)-1u));}
 return selected;}
fn emitPage(level:u32,page:vec3u,generation:u32){let slot=tableSlot(level,page);if(slot==INVALID||slot>=arrayLength(&claims)){return;}if(atomicExchange(&claims[slot],generation)==generation){return;}
 let section=params.sections[level];let recordIndex=atomicAdd(&output[section],1u);if(recordIndex>=levelCapacity(level)){return;}let record=section+${LIVE_SVO_DERIVED_WORKLIST.headerWords}u+recordIndex*RECORD_WORDS;atomicStore(&output[record],slot);
 // One layout at every level now: slot, page coordinate, eight children.
 atomicStore(&output[record+1u],page.x);atomicStore(&output[record+2u],page.y);atomicStore(&output[record+3u],page.z);
 if(level==0u){for(var octant=0u;octant<8u;octant+=1u){let bit=vec3u(octant&1u,(octant>>1u)&1u,(octant>>2u)&1u);atomicStore(&output[record+4u+octant],deepestLeaf(page*${SVO_NODE_MIP_LAYOUT.interiorSize}u+bit*4u));}}
 else{
  // An absent child page is resolved here, once, rather than read as zero in
  // the reduction: a leaf that publishes at its own coarse level leaves every
  // level beneath it unpaged, and those cells are solid ground, not air.
  // Because *every* leaf publishes a page, an absent child page provably has
  // no leaf inside it, so the leaf covering its centre covers all of it.
  let childScale=1u<<(level-1u);
  for(var child=0u;child<8u;child+=1u){let bit=vec3u(child&1u,(child>>1u)&1u,(child>>2u)&1u);
   let childPage=page*2u+bit;var slotOrLeaf=tableSlot(level-1u,childPage);
   if(slotOrLeaf==INVALID){
    let centre=(childPage*${SVO_NODE_MIP_LAYOUT.interiorSize}u+vec3u(${SVO_NODE_MIP_LAYOUT.interiorSize / 2}u))*childScale;
    let leaf=deepestLeaf(centre);
    if(leaf!=INVALID){slotOrLeaf=${LIVE_SVO_DERIVED_WORKLIST.leafSlotTag}u|leaf;}}
   atomicStore(&output[record+${LIVE_SVO_DERIVED_WORKLIST.childSlotWord}u+child],slotOrLeaf);}}
}
@compute @workgroup_size(64) fn populate(@builtin(global_invocation_id) gid:vec3u){let dirty=gid.x;let allLive=params.limits.w!=0u;let partitioned=params.limits.w==2u;let count=select(min(source[params.source.x],params.limits.x),min(control[1],params.limits.x),allLive);if(dirty>=count||(partitioned&&dirty%max(params.source.w,1u)!=params.source.z)){return;}let generation=generationState[params.source.y];if(generation==0u){return;}
 var leaf=dirty;if(!allLive){leaf=source[params.source.z+dirty*params.source.w];}if(leaf>=control[1]){return;}let leafBase=control[16]+leaf*4u;let node=topology[leafBase];let leafLevel=topology[node*8u+2u];if(leafLevel>params.domain.x){return;}let scale=1u<<(params.domain.x-leafLevel);let brickOrigin=morton(topology[leafBase+2u],topology[leafBase+3u],leafLevel)*scale;let cellMinimum=brickOrigin*control[11];
${COARSE_LEAF_PAGES === "extent" ? /* wgsl */ ` let cellMaximum=(brickOrigin+vec3u(scale))*control[11];let firstPage=cellMinimum/${SVO_NODE_MIP_LAYOUT.interiorSize}u;let lastPage=(cellMaximum-vec3u(1u))/${SVO_NODE_MIP_LAYOUT.interiorSize}u;
 for(var z=firstPage.z;z<=lastPage.z;z+=1u){for(var y=firstPage.y;y<=lastPage.y;y+=1u){for(var x=firstPage.x;x<=lastPage.x;x+=1u){var page=vec3u(x,y,z);for(var level=0u;level<params.domain.y;level+=1u){emitPage(level,page,generation);page>>=vec3u(1u);}}}}` : /* wgsl */ ` // One page per leaf, at the level whose texels are its voxels — the CPU mirror
 // is \`liveSvoLeafPage\`. Expanding a coarse leaf across its extent published the
 // same brickSize^3 values once per finest brick it covered, which is what made
 // the pyramid scale with claim volume instead of with surface area.
 let cells=scale*control[11];var seedLevel=0u;
 if(cells>=${SVO_NODE_MIP_LAYOUT.interiorSize}u){seedLevel=firstTrailingBit(cells)-${Math.log2(SVO_NODE_MIP_LAYOUT.interiorSize)}u;}
 var page=cellMinimum/(${SVO_NODE_MIP_LAYOUT.interiorSize}u<<seedLevel);
 for(var level=seedLevel;level<params.domain.y;level+=1u){emitPage(level,page,generation);page>>=vec3u(1u);}`}
}
@compute @workgroup_size(1) fn finalize(@builtin(global_invocation_id) gid:vec3u){let level=gid.x;if(level>=params.domain.y){return;}let section=params.sections[level];let count=min(atomicLoad(&output[section]),levelCapacity(level));atomicStore(&output[section+1u],generationState[params.source.y]);atomicStore(&output[section+2u],level);atomicStore(&output[section+3u],${LIVE_SVO_DERIVED_WORKLIST.headerWords}u);let groups=(count*${SVO_NODE_MIP_LAYOUT.physicalSize ** 3}u+255u)/256u;let x=min(groups,65535u);atomicStore(&output[section+4u],x);atomicStore(&output[section+5u],select(0u,(groups+x-1u)/x,x>0u));atomicStore(&output[section+6u],1u);}
`;

/**
 * Payload reads that differ between a world with a solver and one without.
 *
 * A `dry` world allocates neither the dynamic geometry/velocity/material lanes
 * nor the two dead channels of the scene geometry lane, so every read of them
 * is compiled out here rather than clamped at runtime: an absent lane's offset
 * is the 256-byte zero page, and `offset + voxel * stride` would walk straight
 * out of it into the first present lane. See SPARSE_BRICK_PAYLOAD_PROFILES.
 *
 * Channel *meaning* is preserved across profiles — only the stride and the
 * index move — so the `full` expansion is character-identical to the shader
 * that shipped before profiles existed, which `tests/webgpu-svo-live-derived-
 * builder.test.ts` pins by matching against the exported constants.
 *
 * A dry world may narrow those two channels further (see
 * {@link SparseBrickSceneGeometryFormat}). That moves the unpack, not the meaning:
 * every surviving arm returns `solidDistance` in metres. A removed 2-byte arm
 * returned per-leaf band units instead, which was tolerable only because
 * `safeNormal` central-differences six samples from one leaf and normalises, so the
 * per-leaf scale cancelled. The `min` against the solver's metres on the `full`
 * profile is exactly why narrowing is refused there — and the reason any future
 * scaled arm must declare its units rather than rely on that cancellation.
 */

/**
 * `solidDistance` under the banded leaf payload's reconstruction rule.
 *
 * The B0.5b gate (`docs/svo-banded-leaf-payload-handoff.md`): the banded layout
 * stores a distance only for the band and the one-cell ring `safeNormal` reads,
 * and reconstructs everything else as the saturated value signed by occupancy.
 * That reconstruction is the design's one lossy semantic — 86.4 % of occupied
 * voxels lose their normal by it — and it can be tested *without* the arena, the
 * masks, the palette or the allocator by changing only what this function
 * returns. If the frame hash holds here, the rest of the layout is engineering; if
 * it moves, no amount of allocator work would have discovered that.
 *
 * The record predicate is the encoder's, expressed the same way: a voxel is
 * stored when it or one of its six axis neighbours is band, with the neighbours
 * clamped inside the leaf exactly as `safeNormal` clamps them. That clamped
 * 6-neighbourhood *is* the encoder's dilation — at a face the clamp folds the
 * outward neighbour onto the voxel itself, which is the same set the encoder
 * marks.
 *
 * The saturation is `SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII` cell radii of the
 * leaf's *own* voxel, so it matches `encodeBandedLeaf` rather than approximating
 * it, and the finest cell size is baked as a literal because the build shader's
 * uniform block does not carry one.
 */
/**
 * How far a radiance gather starts from the voxel that emits it, in finest cells.
 *
 * `1.5 * scale` unconditionally is what shipped, and at a coarse level that is
 * tens of finest cells — so a voxel metres inside the ground began its visibility
 * march *above* the terrain and took direct sun. That is the root cause of the
 * buried-radiance leak measured at 22.22/255 mean, and it is why fixing the
 * origin is preferable to omitting the radiance downstream of it.
 *
 * The offset cannot simply shrink. A **band** voxel has the surface passing
 * through it, so it needs an offset on the order of its own extent or its gather
 * self-intersects the voxel it started in — that is the self-shadow acne
 * `1.5 * scale` was buying, and it gets worse, not better, at coarse levels.
 *
 * The distinction is not the offset's size but whether there is anywhere outside
 * to move to. A **fully solid** voxel is interior everywhere within it; no offset
 * along a normal can put its origin in air that is legitimately its own. So the
 * rule is not a smaller offset, it is *no* offset for a fully solid voxel, and the
 * shipped offset for every voxel the surface actually touches. Expressed on the
 * fraction rather than on band membership because that is the exact question —
 * "is any part of this voxel not solid" — and it needs no stencil.
 */
function derivedMarchOffsetWGSL(solidOffsetFix: boolean): string {
  if (!solidOffsetFix) return "(1.5*f32(scale))";
  return "(select(1.5*f32(scale),0.0,bandedFractionAt(leaf,sampleLocal)>=1.0))";
}

/**
 * A fully solid receiver is opaque to direct light, decided before any step.
 *
 * `directVisibility` shares the gather march's step schedule, and that schedule is
 * expressed in *receiver* widths so a step from a floor-level block cannot land
 * inside the block that emitted it. That ratio is what stops a gather
 * self-sampling; for occlusion it is precisely backwards, because it guarantees
 * the *first* sample clears the receiver's own block. At a floor level of 3 and a
 * 6.25 mm cell the first sample is 20 finest cells — 12.5 cm — away, and the
 * hero's terrain is 0.14-0.38 m thick, so every sample from a voxel just under
 * the ground is already in open air above it, `1 - coverage` is 1 at every step,
 * and solid earth takes full sun.
 *
 * Rescaling the occlusion march alone does not fix it, and that is measured: the
 * same four steps taken in *finest* cells — 8x shorter here — recover 1.42/255 of
 * a 12.30/255 leak, and add 0.013/255 on top of this guard. The coarse pyramid
 * holds one average coverage per floor-level block and cannot separate "just below
 * the surface" from "just above": a first step short enough to find the ground
 * under a buried voxel reads the receiver's own straddling block for an exposed
 * voxel too, and dims every lit surface by that block's coverage. That is the
 * self-shadow acne the current scaling buys, arriving as a uniform darkening
 * instead of speckle.
 *
 * So the question is answered where the answer actually exists — in the fine
 * fraction, which knows whether *this* voxel has any air in it at all. A fully
 * solid voxel sees no sun, exactly and independently of any step length. A
 * partially solid voxel is by definition one the surface passes through, is
 * entitled to sun, and keeps the shipped march unchanged. Same predicate as the
 * march-offset rule (`bandedFractionAt >= 1.0`), deliberately: two arms that
 * disagree about which voxels are solid would be two definitions of solid.
 */
function derivedSolidDirectOcclusionWGSL(enabled: boolean) {
  return {
    /** Threaded rather than recomputed: the fraction read is a payload fetch. */
    parameter: enabled ? ",receiverSolid:bool" : "",
    guard: enabled ? "if(receiverSolid){return 0.0;}" : "",
    forward: enabled ? ",receiverSolid" : "",
    receiver: enabled ? ",bandedFractionAt(leaf,sampleLocal)>=1.0" : "",
  };
}

function derivedSolidDistanceWGSL(
  lane: { solidDistance: string; sceneCoverage: string },
  cellSize_m: readonly [number, number, number] | undefined,
  // The band helpers on their own: the march-offset rule needs
  // `bandedFractionAt` without the banded reconstruction being on.
  sharedOnly = false,
): string {
  const stored = /* wgsl */ `fn solidDistance(leafIndex:u32,local:vec3u)->f32{let voxel=leafVoxel(leafIndex,local);return ${lane.solidDistance};}`;
  // The band helpers are shared rather than inlined per caller: the banded
  // reconstruction and the march-offset rule ask the same "is this voxel's normal
  // real" question, and defining it twice is how two callers come to disagree
  // about which voxels have one.
  const shared = /* wgsl */ `
fn bandedFractionAt(leafIndex:u32,local:vec3u)->f32{let voxel=leafVoxel(leafIndex,local);return ${lane.sceneCoverage};}
fn bandedBandAt(leafIndex:u32,local:vec3u)->bool{let f=bandedFractionAt(leafIndex,local);return f>0.0&&f<1.0;}
fn bandedRecordAt(leafIndex:u32,local:vec3u)->bool{
  if(bandedBandAt(leafIndex,local)){return true;}
  let lo=max(local,vec3u(1u))-1u;let hi=min(local+1u,vec3u(control[11]-1u));
  return bandedBandAt(leafIndex,vec3u(lo.x,local.y,local.z))||bandedBandAt(leafIndex,vec3u(hi.x,local.y,local.z))
    ||bandedBandAt(leafIndex,vec3u(local.x,lo.y,local.z))||bandedBandAt(leafIndex,vec3u(local.x,hi.y,local.z))
    ||bandedBandAt(leafIndex,vec3u(local.x,local.y,lo.z))||bandedBandAt(leafIndex,vec3u(local.x,local.y,hi.z));
}
fn bandedLeafScale(leafIndex:u32)->u32{
  let node=topology[control[16]+leafIndex*4u];let level=topology[node*8u+2u];
  let finest=finestLevel();
  return select(1u,1u<<(finest-level),finest>level);
}`;
  const helpers = sharedOnly ? shared : "";
  if (!cellSize_m) return `${stored}${helpers}`;
  const cell = cellSize_m.map((value) => value.toExponential(9)).join(",");
  return /* wgsl */ `${shared}
const BANDED_FINEST_CELL:vec3f=vec3f(${cell});
fn storedSolidDistance(leafIndex:u32,local:vec3u)->f32{let voxel=leafVoxel(leafIndex,local);return ${lane.solidDistance};}
fn bandedSaturation(leafIndex:u32)->f32{
  return ${SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII}.0*0.5*length(BANDED_FINEST_CELL*f32(bandedLeafScale(leafIndex)));
}
fn solidDistance(leafIndex:u32,local:vec3u)->f32{
  if(bandedRecordAt(leafIndex,local)){return storedSolidDistance(leafIndex,local);}
  let saturation=bandedSaturation(leafIndex);
  return select(saturation,-saturation,bandedFractionAt(leafIndex,local)>0.0);
}`;
}

function derivedLaneAccess(
  profile: SparseBrickPayloadProfileName,
  sceneGeometryFormat: SparseBrickSceneGeometryFormat = "f32x2",
  leafPayloadMode: SparseBrickLeafPayloadMode = "dense",
) {
  const dry = profile === "dry";
  const format = dry ? sceneGeometryFormat : "f32x2";
  const mode = dry ? leafPayloadMode : "dense";
  const stride = dry ? "2u" : "4u";
  const scene = (channel: string) => `payload[params.laneOffsets.z+voxel*${stride}+${channel}]`;
  const packedWord = "payload[sceneGeometryWord(params.laneOffsets.z,voxel)]";
  const narrowedDistance = `sceneDistanceOf(${packedWord},voxel)`;
  const narrowedCoverage = `clamp(sceneFractionOf(${packedWord},voxel),0.,1.)`;
  // Scene identity is not a lane under `banded`, so it is not an expression
  // either: it comes from the shared codec, addressed by the same
  // `bandedLanes` block the voxeliser is handed. The two geometry channels
  // deliberately do *not* move with it — this builder still reads the dense
  // `sceneGeometry` lane, which is the B2 cutover and a separate piece of work.
  const identityCodec = mode === "dense" ? "" : /* wgsl */ `${sparseBrickBandedLeafCodecWGSL({
    occupancyBase: "params.bandedLanes.x", recordMaskBase: "params.bandedLanes.y",
    headerBase: "params.bandedLanes.z", blobBase: "params.bandedLanes.w",
    recordsBase: "0u",
    load: (index) => `payload[${index}]`, mode, records: false,
  })}`;
  return {
    /** Helper declarations the narrowed formats and the identity decode need in scope. */
    codec: `${sparseBrickSceneGeometryCodecWGSL(format)}${identityCodec}
${sparseBrickSceneIdentityCodecWGSL({
      mode, materialOwnerBase: "params.laneOffsets.w", load: (index) => `payload[${index}]`,
    })}`,
    /** The gradient of this is the only smooth normal the world can offer. */
    solidDistance: dry
      ? format === "f32x2" ? `bitcast<f32>(${scene("0u")})` : narrowedDistance
      : `min(bitcast<f32>(payload[voxel*4u+1u]),bitcast<f32>(${scene("1u")}))`,
    // `>>> 0` because `0xffff << 16` is -65536 in JS's signed shift, and WGSL has
    // no unary minus on `u32` — the dry expansion failed to compile without it.
    dynamicIdentity: dry ? `${(SPARSE_BRICK_NO_OWNER << 16) >>> 0}u` : "payload[params.laneOffsets.y+voxel]",
    dynamicCoverage: dry ? "0." : "clamp(bitcast<f32>(payload[voxel*4u+2u]),0.,1.)",
    sceneCoverage: format === "f32x2"
      ? `clamp(bitcast<f32>(${scene(dry ? "1u" : "2u")}),0.,1.)`
      : narrowedCoverage,
    fluidFraction: dry ? "0." : "clamp(bitcast<f32>(payload[params.laneOffsets.x+voxel*4u+3u]),0.,1.)",
  };
}

/**
 * Where a page's interior starts, physically.
 *
 * Zero by default (see `SVO_NODE_MIP_LAYOUT`), so this is a rename of the `+1u`
 * that used to be sprinkled through every atlas read. Keeping it named rather
 * than deleting it is what lets `FLUID_SVO_NODE_MIP_APRON=1` put the shell back
 * for an A/B without hunting for literals in six shaders.
 */
const APRON = SVO_NODE_MIP_LAYOUT.apron;
/** Interior-texel clamp that reproduces a replica apron of the declared width. */
const INTERIOR_CLAMP_LOW = (-0.5 * APRON).toFixed(1);
const INTERIOR_CLAMP_HIGH = (1 - 0.5 * APRON).toFixed(1);

/**
 * Opacity lanes this build actually reduces.
 *
 * On a dry world `fluidFraction` is the literal `0.`, so `mean.y` is a mean of
 * zeros and `maximum.y` a maximum of zeros — arithmetic whose only destination
 * is two texture channels the narrow format does not have. Compiling it out is
 * not the saving (that is the format); it is what stops the shader claiming to
 * carry a lane its page cannot hold.
 */
function derivedOpacityLanes(format: GPUTextureFormat) {
  const fluid = svoNodeMipOpacityChannels(format) > 2;
  return {
    fluid,
    /** Base-level store. `solid` and `fluid` must be in scope. */
    storeBase: fluid
      ? "vec4f(solid,select(0.,1.,solid>0.),clamp(fluid,0.,1.),select(0.,1.,fluid>0.))"
      : "vec4f(solid,select(0.,1.,solid>0.),0.,0.)",
    /** Parent-level store. `mean:vec2f` and `maximum:vec2f` must be in scope. */
    storeParent: fluid
      ? "vec4f(mean.x,maximum.x,mean.y,maximum.y)"
      : "vec4f(mean.x,maximum.x,0.,0.)",
  };
}

/**
 * The two uniform lanes that place a radiance page.
 *
 * `radianceAtlas` is `xyz` = the radiance atlas page grid, `w` = the slot the
 * atlas begins at. `radianceScratch` is `xyz` = the compact radiance scratch
 * grid, `w` = the radiance floor level. Both atlases are addressed by the *same*
 * slot number the opacity pyramid uses — page validity, the claims buffer and
 * the black-page certificate all stay in that one index space — so the only
 * thing the floor changes is where a slot lands physically, and whether it
 * lands anywhere at all.
 */
const LIVE_SVO_RADIANCE_PARAMS_FIELDS = "radianceAtlas:vec4u,radianceScratch:vec4u";

/** Requires `params` and `PHYSICAL` to be in scope. */
const liveSvoRadianceAddressingWGSL = /* wgsl */ `
fn radianceFloorLevel()->u32{return params.radianceScratch.w;}
fn radianceSlotResident(slot:u32)->bool{return slot>=params.radianceAtlas.w;}
fn radianceSlotOrigin(slot:u32)->vec3u{
  let pages=params.radianceAtlas.xyz;let index=slot-params.radianceAtlas.w;
  return vec3u(index%pages.x,(index/pages.x)%pages.y,index/(pages.x*pages.y))*PHYSICAL;
}
fn radianceScratchOrigin(recordIndex:u32)->vec3u{
  let pages=params.radianceScratch.xyz;
  return vec3u(recordIndex%pages.x,(recordIndex/pages.x)%pages.y,recordIndex/(pages.x*pages.y))*PHYSICAL;
}
fn radianceLobeDepth()->u32{return params.radianceScratch.z*PHYSICAL;}
`;

/**
 * Deepest active leaf covering one finest-level cell.
 *
 * A copy of the worklist planner's descent, which the radiance floor now needs
 * on the build side too: a floor-level texel spans `2^floor` cells and no longer
 * matches the eight per-octant leaves a base record carries. Requires `control`,
 * `topology`, `ACTIVE`, `INVALID` and a `finestLevel()` accessor in scope.
 */
const liveSvoDeepestLeafWGSL = /* wgsl */ `
fn deepestLeaf(globalCell:vec3u)->u32{
  let brickSize=control[11];if(brickSize==0u||control[0]==0u){return INVALID;}
  let finest=finestLevel();let brick=globalCell/brickSize;var node=0u;var selected=INVALID;
  for(var level=0u;level<=finest&&node<control[0];level+=1u){
    let nodeBase=node*8u;let leaf=topology[nodeBase+6u];
    if(leaf<control[1]&&(topology[nodeBase+7u]&ACTIVE)!=0u){selected=leaf;}
    if(level==finest){break;}
    let bit=finest-level-1u;
    let octant=((brick.x>>bit)&1u)|(((brick.y>>bit)&1u)<<1u)|(((brick.z>>bit)&1u)<<2u);
    let mask=topology[nodeBase+3u]&0xffu;if((mask&(1u<<octant))==0u){break;}
    node=topology[nodeBase+4u]+countOneBits(mask&((1u<<octant)-1u));}
  return selected;}
`;

/**
 * Radiance storage declarations follow the atlas the device actually allocated.
 *
 * The lobe value is `vec3f` everywhere in this shader; the fourth component of
 * the `textureStore` operand is discarded by a three-channel format and was a
 * literal `1.0` in the four-channel one, so the WGSL below is the *same*
 * arithmetic under either declaration — only the stored precision differs.
 */
export function liveSvoDerivedBuildWGSLFor(
  profile: SparseBrickPayloadProfileName = "full",
  radianceFormat: GPUTextureFormat = LIVE_SVO_RADIANCE_STORAGE.fallbackFormat,
  opacityFormat: GPUTextureFormat = SVO_NODE_MIP_OPACITY_STORAGE.wideFormat,
  sceneGeometryFormat: SparseBrickSceneGeometryFormat = "f32x2",
  /**
   * The finest cell size, present only when the banded reconstruction arm is on.
   * `undefined` — the default — emits the shipped `solidDistance` unchanged.
   */
  bandedReconstructionCellSize_m?: readonly [number, number, number],
  /** How the world stores scene identity. Decides which identity decode compiles. */
  leafPayloadMode: SparseBrickLeafPayloadMode = "dense",
): string {
  const lane = derivedLaneAccess(profile, sceneGeometryFormat, leafPayloadMode);
  const solidDistance = derivedSolidDistanceWGSL(lane, bandedReconstructionCellSize_m);

  const opacity = derivedOpacityLanes(opacityFormat);
  return /* wgsl */ `
${liveSvoDerivedPageValidityWGSL}
struct Params{targetAtlasPages:vec4u,scratchAtlasPages:vec4u,limits:vec4u,laneOffsets:vec4u,${LIVE_SVO_RADIANCE_PARAMS_FIELDS},bandedLanes:vec4u}
@group(0) @binding(0) var<storage,read> control:array<u32>;
@group(0) @binding(1) var<storage,read> topology:array<u32>;
@group(0) @binding(2) var<storage,read> payload:array<u32>;
@group(0) @binding(3) var<storage,read> worklist:array<u32>;
@group(0) @binding(4) var<storage,read> emission:array<vec4f>;
@group(0) @binding(5) var opacitySource:texture_3d<f32>;
@group(0) @binding(6) var opacityScratch:texture_storage_3d<${opacityFormat},write>;
@group(0) @binding(7) var radianceSource0:texture_3d<f32>;
@group(0) @binding(8) var radianceSource1:texture_3d<f32>;
@group(0) @binding(9) var radianceSource2:texture_3d<f32>;
@group(0) @binding(10) var radianceSource3:texture_3d<f32>;
@group(0) @binding(11) var radianceScratch:texture_storage_3d<${radianceFormat},write>;
@group(0) @binding(12) var<uniform> params:Params;
@group(0) @binding(13) var opacityValidity:texture_2d<u32>;
@group(0) @binding(14) var radianceValidity:texture_2d<u32>;
@group(0) @binding(15) var<storage,read_write> scratchValidity:array<u32>;

const INVALID:u32=0xffffffffu;
const LEAF_SLOT_TAG:u32=${LIVE_SVO_DERIVED_WORKLIST.leafSlotTag}u;
const PHYSICAL:u32=${SVO_NODE_MIP_LAYOUT.physicalSize}u;
const INTERIOR:u32=${SVO_NODE_MIP_LAYOUT.interiorSize}u;
const RECORD_WORDS:u32=${LIVE_SVO_DERIVED_WORKLIST.recordWords}u;
const ACTIVE:u32=${SVO_BRICK_LIFECYCLE.activeBit}u;
const TETRA0:vec3f=vec3f(.577350269,.577350269,.577350269);
const TETRA1:vec3f=vec3f(.577350269,-.577350269,-.577350269);
const TETRA2:vec3f=vec3f(-.577350269,.577350269,-.577350269);
const TETRA3:vec3f=vec3f(-.577350269,-.577350269,.577350269);

fn linearIndex(gid:vec3u,groups:vec3u)->u32{return gid.x+gid.y*groups.x*256u+gid.z*groups.x*groups.y*256u;}
fn finestLevel()->u32{return params.scratchAtlasPages.w;}
${lane.codec}
${liveSvoRadianceAddressingWGSL}
${liveSvoDeepestLeafWGSL}
fn slotOrigin(slot:u32)->vec3u{
  let pages=params.targetAtlasPages.xyz;
  return vec3u(slot%pages.x,(slot/pages.x)%pages.y,slot/(pages.x*pages.y))*PHYSICAL;
}
fn scratchOrigin(recordIndex:u32)->vec3u{
  let pages=params.scratchAtlasPages.xyz;
  return vec3u(recordIndex%pages.x,(recordIndex/pages.x)%pages.y,recordIndex/(pages.x*pages.y))*PHYSICAL;
}
fn physicalCoordinate(index:u32)->vec3u{return vec3u(index%PHYSICAL,(index/PHYSICAL)%PHYSICAL,index/(PHYSICAL*PHYSICAL));}
// At a zero apron this is the identity, which is the point: the shell it used to
// strip held a clamped copy of the very texel the clamp already selects.
fn interiorCoordinate(physical:vec3u)->vec3u{return clamp(physical,vec3u(${APRON}u),vec3u(INTERIOR-1u+${APRON}u))-${APRON}u;}
fn keyBit(lo:u32,hi:u32,bit:u32)->u32{if(bit>=32u){return(hi>>(bit-32u))&1u;}return(lo>>bit)&1u;}
fn morton(lo:u32,hi:u32,level:u32)->vec3u{var p=vec3u(0u);for(var bit=0u;bit<level;bit+=1u){let s=1u<<bit;p+=vec3u(keyBit(lo,hi,3u*bit),keyBit(lo,hi,3u*bit+1u),keyBit(lo,hi,3u*bit+2u))*s;}return p;}
fn leafLocal(globalCell:vec3u,leaf:u32)->vec3u{let brickSize=control[11];let finest=params.scratchAtlasPages.w;let node=topology[control[16]+leaf*4u];let nodeBase=node*8u;let level=topology[nodeBase+2u];let scale=1u<<(finest-level);let origin=morton(topology[nodeBase],topology[nodeBase+1u],level)*scale*brickSize;return min((globalCell-origin)/scale,vec3u(brickSize-1u));}
fn leafVoxel(leafIndex:u32,local:vec3u)->u32{
  let leafBase=control[16]+leafIndex*4u;
  return topology[leafBase+1u]+local.x+local.y*control[11]+local.z*control[11]*control[11];
}
${solidDistance}
fn safeNormal(leafIndex:u32,local:vec3u)->vec3f{
  let lo=max(local,vec3u(1u))-1u;let hi=min(local+1u,vec3u(control[11]-1u));
  let gradient=vec3f(solidDistance(leafIndex,vec3u(hi.x,local.y,local.z))-solidDistance(leafIndex,vec3u(lo.x,local.y,local.z)),
    solidDistance(leafIndex,vec3u(local.x,hi.y,local.z))-solidDistance(leafIndex,vec3u(local.x,lo.y,local.z)),
    solidDistance(leafIndex,vec3u(local.x,local.y,hi.z))-solidDistance(leafIndex,vec3u(local.x,local.y,lo.z)));
  return select(vec3f(0.,1.,0.),normalize(gradient),dot(gradient,gradient)>1e-12);
}
/**
 * Opacity of one finest-level cell, read straight out of the leaf that owns it.
 *
 * The same arithmetic the base level performs, minus the descent: the caller
 * already knows the leaf. This is what a reduction reads where the level below
 * holds no page — the interior of a coarse leaf, which publishes one page at
 * its own level and nothing beneath it.
 */
fn leafOpacityAt(leaf:u32,cell:vec3u)->vec4f{
  let sampleLocal=leafLocal(cell,leaf);let voxel=leafVoxel(leaf,sampleLocal);
  let dynamicIdentity=${lane.dynamicIdentity};let sceneIdentity=sceneIdentityAt(voxel);
  let dynamicCoverage=${lane.dynamicCoverage};let sceneCoverage=${lane.sceneCoverage};
  let dynamicSolid=select(dynamicCoverage,0.,(dynamicIdentity&0xffffu)==${VOXEL_MATERIAL_IDS.containerGlass}u);
  let sceneSolid=select(sceneCoverage,0.,(sceneIdentity&0xffffu)==${VOXEL_MATERIAL_IDS.containerGlass}u);
  let fluid=${lane.fluidFraction};let solid=1.-(1.-dynamicSolid)*(1.-sceneSolid);
  return ${opacity.storeBase};
}
/** The finest cell at the centre of one child-level texel of this record's page. */
fn childCell(record:u32,fine:vec3u)->vec3u{
  let page=vec3u(worklist[record+1u],worklist[record+2u],worklist[record+3u]);
  let childScale=1u<<(worklist[2]-1u);
  return (page*INTERIOR*2u+fine)*childScale+vec3u(childScale>>1u);
}
fn childSample(record:u32,childBase:u32,local:vec3u,sampleIndex:u32)->vec4f{
  let bit=vec3u(sampleIndex&1u,(sampleIndex>>1u)&1u,(sampleIndex>>2u)&1u);
  let fine=local*2u+bit;let octant=(fine.x/INTERIOR)|((fine.y/INTERIOR)<<1u)|((fine.z/INTERIOR)<<2u);
  let child=worklist[record+childBase+octant];
  if(child==INVALID){return vec4f(0.);}
  if((child&LEAF_SLOT_TAG)!=0u){return leafOpacityAt(child&~LEAF_SLOT_TAG,childCell(record,fine));}
  return textureLoad(opacitySource,vec3i(slotOrigin(child)+(fine%INTERIOR)+${APRON}u),0);
}
fn childRadiance(record:u32,local:vec3u,sampleIndex:u32,lobe:u32)->vec3f{
  let bit=vec3u(sampleIndex&1u,(sampleIndex>>1u)&1u,(sampleIndex>>2u)&1u);
  let fine=local*2u+bit;let octant=(fine.x/INTERIOR)|((fine.y/INTERIOR)<<1u)|((fine.z/INTERIOR)<<2u);
  let child=worklist[record+${LIVE_SVO_DERIVED_WORKLIST.childSlotWord}u+octant];
  if(child==INVALID||(child&LEAF_SLOT_TAG)!=0u||!radianceSlotResident(child)){return vec3f(0.);}
  let coordinate=vec3i(radianceSlotOrigin(child)+(fine%INTERIOR)+${APRON}u);
  if(lobe==0u){return textureLoad(radianceSource0,coordinate,0).rgb;}
  if(lobe==1u){return textureLoad(radianceSource1,coordinate,0).rgb;}
  if(lobe==2u){return textureLoad(radianceSource2,coordinate,0).rgb;}
  return textureLoad(radianceSource3,coordinate,0).rgb;
}
fn writeRadiance(coordinate:vec3u,a:vec3f,b:vec3f,c:vec3f,d:vec3f){
  let depth=radianceLobeDepth();
  textureStore(radianceScratch,coordinate,vec4f(a,1.));textureStore(radianceScratch,coordinate+vec3u(0u,0u,depth),vec4f(b,1.));
  textureStore(radianceScratch,coordinate+vec3u(0u,0u,depth*2u),vec4f(c,1.));textureStore(radianceScratch,coordinate+vec3u(0u,0u,depth*3u),vec4f(d,1.));
}
fn referencedChildrenReady(record:u32,childBase:u32,radianceChildren:bool)->bool{
  for(var childIndex=0u;childIndex<8u;childIndex+=1u){let child=worklist[record+childBase+childIndex];
    // A tagged child names a leaf, not a page: the payload it reads is the
    // publication's own and needs no page to have been built first.
    if(child==INVALID||(child&LEAF_SLOT_TAG)!=0u){continue;}
    if(textureLoad(opacityValidity,svoDerivedPageValidityTexel(textureDimensions(opacityValidity),child),0).x==0u){return false;}
    if(radianceChildren&&textureLoad(radianceValidity,svoDerivedPageValidityTexel(textureDimensions(radianceValidity),child),0).x==0u){return false;}}
  return true;
}
/**
 * Radiance for one texel of the level the radiance chain is *based* at.
 *
 * Below the floor there is no radiance page to reduce, so this level stops
 * being a reduction and becomes a source. Its texel spans \`2^level\` cells, and
 * the three things it needs come from three different places: the exact mean
 * coverage from the opacity reduction that has just run over its children, the
 * emitting material from a tree descent at the texel's centre cell, and the
 * outward normal from the gradient of that same coverage across the eight
 * children — the iso-surface normal *at the texel's own scale*, which is the
 * one a cone this coarse is asking about. A degenerate gradient (a buried or a
 * uniformly thin block) falls back to the leaf's own signed-distance normal.
 */
fn radianceBaseNormal(record:u32,childBase:u32,local:vec3u,leaf:u32,sampleLocal:vec3u)->vec3f{
  var gradient=vec3f(0.);
  for(var child=0u;child<8u;child+=1u){
    let sign=vec3f(vec3u(child&1u,(child>>1u)&1u,(child>>2u)&1u))*2.-1.;
    gradient+=sign*childSample(record,childBase,local,child).x;}
  if(dot(gradient,gradient)>1e-6){return -normalize(gradient);}
  if(leaf<control[1]){return safeNormal(leaf,sampleLocal);}
  return vec3f(0.,1.,0.);
}

@compute @workgroup_size(256)
fn buildPages(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
  let index=linearIndex(gid,groups);let texels=PHYSICAL*PHYSICAL*PHYSICAL;
  let recordIndex=index/texels;let count=min(worklist[0],params.limits.x);if(recordIndex>=count){return;}
  let physical=physicalCoordinate(index-recordIndex*texels);let local=interiorCoordinate(physical);
  let record=worklist[3]+recordIndex*RECORD_WORDS;let destination=scratchOrigin(recordIndex)+physical;
  let radianceDestination=radianceScratchOrigin(recordIndex)+physical;
  let level=worklist[2];let floorLevel=radianceFloorLevel();
  if(level==0u){
    if(all(physical==vec3u(0u))){scratchValidity[recordIndex]=worklist[1];}let page=vec3u(worklist[record+1u],worklist[record+2u],worklist[record+3u]);let octant=(local.x/4u)|((local.y/4u)<<1u)|((local.z/4u)<<2u);let leaf=worklist[record+${LIVE_SVO_DERIVED_WORKLIST.sourceLeafWord}u+octant];if(leaf>=control[1]){textureStore(opacityScratch,destination,vec4f(0.));if(floorLevel==0u){writeRadiance(radianceDestination,vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.));}return;}let sampleLocal=leafLocal(page*INTERIOR+local,leaf);
    let voxel=leafVoxel(leaf,sampleLocal);let dynamicIdentity=${lane.dynamicIdentity};let sceneIdentity=sceneIdentityAt(voxel);
    let dynamicCoverage=${lane.dynamicCoverage};let sceneCoverage=${lane.sceneCoverage};
    // Container glass remains structural geometry, but it is not an opacity
    // source: otherwise the cone hierarchy turns the vessel into a projected
    // cutout even though the exact/composite paths treat it as dielectric.
    let dynamicSolid=select(dynamicCoverage,0.,(dynamicIdentity&0xffffu)==${VOXEL_MATERIAL_IDS.containerGlass}u);let sceneSolid=select(sceneCoverage,0.,(sceneIdentity&0xffffu)==${VOXEL_MATERIAL_IDS.containerGlass}u);
    ${opacity.fluid ? `let fluid=${lane.fluidFraction};` : ""}let solid=1.-(1.-dynamicSolid)*(1.-sceneSolid);
    textureStore(opacityScratch,destination,${opacity.storeBase});
    if(floorLevel!=0u){return;}
    let material=select(dynamicIdentity&0xffffu,sceneIdentity&0xffffu,sceneSolid>=dynamicSolid&&sceneSolid>0.);var emitted=vec3f(0.);
    if(material<arrayLength(&emission)){emitted=max(emission[material].rgb,vec3f(0.));}
    let normal=safeNormal(leaf,sampleLocal);let covered=solid;
    writeRadiance(radianceDestination,emitted*covered*max(0.,dot(normal,TETRA0)),emitted*covered*max(0.,dot(normal,TETRA1)),
      emitted*covered*max(0.,dot(normal,TETRA2)),emitted*covered*max(0.,dot(normal,TETRA3)));
    return;
  }
  let radianceBase=level==floorLevel;
  // One layout at every level: the coordinate every parent now needs to resolve
  // a leaf-tagged child is where the radiance floor always kept it.
  let childBase=${LIVE_SVO_DERIVED_WORKLIST.childSlotWord}u;
  if(all(physical==vec3u(0u))){scratchValidity[recordIndex]=select(0u,worklist[1],referencedChildrenReady(record,childBase,level>floorLevel));}
  var mean=vec2f(0.);var maximum=vec2f(0.);var r0=vec3f(0.);var r1=vec3f(0.);var r2=vec3f(0.);var r3=vec3f(0.);
  for(var child=0u;child<8u;child+=1u){let value=childSample(record,childBase,local,child);mean+=value.xz/8.;maximum=max(maximum,value.yw);}
  if(level>floorLevel){
    for(var child=0u;child<8u;child+=1u){
      r0+=childRadiance(record,local,child,0u)/8.;r1+=childRadiance(record,local,child,1u)/8.;
      r2+=childRadiance(record,local,child,2u)/8.;r3+=childRadiance(record,local,child,3u)/8.;}
  }
  textureStore(opacityScratch,destination,${opacity.storeParent});
  if(level<floorLevel){return;}
  if(radianceBase){
    let page=vec3u(worklist[record+1u],worklist[record+2u],worklist[record+3u]);
    let scale=1u<<level;
    let cell=(page*INTERIOR+local)*scale+vec3u(scale>>1u);
    let leaf=deepestLeaf(cell);
    var emitted=vec3f(0.);var sampleLocal=vec3u(0u);
    if(leaf<control[1]){
      sampleLocal=leafLocal(cell,leaf);
      let material=sceneIdentityAt(leafVoxel(leaf,sampleLocal))&0xffffu;
      if(material<arrayLength(&emission)){emitted=max(emission[material].rgb,vec3f(0.));}}
    let normal=radianceBaseNormal(record,childBase,local,leaf,sampleLocal);
    r0=emitted*mean.x*max(0.,dot(normal,TETRA0));r1=emitted*mean.x*max(0.,dot(normal,TETRA1));
    r2=emitted*mean.x*max(0.,dot(normal,TETRA2));r3=emitted*mean.x*max(0.,dot(normal,TETRA3));
  }
  writeRadiance(radianceDestination,r0,r1,r2,r3);
}`;
}

/** The `full` expansion. Byte-identical to the pre-profile shader. */
export const liveSvoDerivedBuildWGSL = liveSvoDerivedBuildWGSLFor("full");

/**
 * A deliberately low-rate Jacobi-style diffuse solve. Base pages read the
 * previous complete radiance atlas and write emission + reflected incident
 * light into compact scratch; parent pages then reduce the newly published
 * children. The recurrence replaces, rather than adds to, previous radiance,
 * so energy cannot grow merely because more frames have elapsed.
 */
export function liveSvoRadianceFeedbackWGSLFor(
  profile: SparseBrickPayloadProfileName = "full",
  radianceFormat: GPUTextureFormat = LIVE_SVO_RADIANCE_STORAGE.fallbackFormat,
  sceneGeometryFormat: SparseBrickSceneGeometryFormat = "f32x2",
  bandedReconstructionCellSize_m?: readonly [number, number, number],
  solidMarchOffsetFix = false,
  solidDirectOcclusion = false,
  /** How the world stores scene identity. Decides which identity decode compiles. */
  leafPayloadMode: SparseBrickLeafPayloadMode = "dense",
): string {
  const lane = derivedLaneAccess(profile, sceneGeometryFormat, leafPayloadMode);
  // The offset rule and the direct-occlusion rule both read the fraction, so the
  // shared helpers must be in scope whenever either is on, not only when banding
  // is.
  const solidDistance = derivedSolidDistanceWGSL(
    lane, bandedReconstructionCellSize_m, solidMarchOffsetFix || solidDirectOcclusion);
  const marchOffsetCells = derivedMarchOffsetWGSL(solidMarchOffsetFix);
  const directOcclusion = derivedSolidDirectOcclusionWGSL(solidDirectOcclusion);

  return /* wgsl */ `
${liveSvoDerivedPageValidityWGSL}
${svoMaterialWGSL}
${svoEnvironmentLightingWGSL}
${svoLightWGSL}
${svoTetrahedralRadianceWGSL}
struct Params{targetAtlasPages:vec4u,scratchAtlasPages:vec4u,limits:vec4u,laneOffsets:vec4u,direct:vec4u,zOffsets:array<u32,12>,mappingOrigin:vec4f,mappingCellSize:vec4f,${LIVE_SVO_RADIANCE_PARAMS_FIELDS},bandedLanes:vec4u}
@group(0) @binding(0) var<storage,read> control:array<u32>;
@group(0) @binding(1) var<storage,read> topology:array<u32>;
@group(0) @binding(2) var<storage,read> payload:array<u32>;
@group(0) @binding(3) var<storage,read> worklist:array<u32>;
@group(0) @binding(4) var opacitySource:texture_3d<f32>;
@group(0) @binding(5) var radianceSource0:texture_3d<f32>;
@group(0) @binding(6) var radianceSource1:texture_3d<f32>;
@group(0) @binding(7) var radianceSource2:texture_3d<f32>;
@group(0) @binding(8) var radianceSource3:texture_3d<f32>;
@group(0) @binding(9) var radianceScratch:texture_storage_3d<${radianceFormat},write>;
@group(0) @binding(10) var<uniform> params:Params;
@group(0) @binding(11) var directTable:texture_3d<u32>;
@group(0) @binding(12) var<storage,read> materials:array<SvoMaterialRecord>;
@group(0) @binding(13) var<storage,read> environment:array<SvoEnvironmentLightingRecord>;
@group(0) @binding(14) var opacityValidity:texture_2d<u32>;
@group(0) @binding(15) var radianceValidity:texture_2d<u32>;
@group(0) @binding(16) var<storage,read_write> scratchValidity:array<u32>;
@group(0) @binding(17) var<storage,read> lights:array<SvoLightRecord>;
@group(0) @binding(18) var atlasSampler:sampler;

const INVALID:u32=0xffffffffu;const PHYSICAL:u32=${SVO_NODE_MIP_LAYOUT.physicalSize}u;const INTERIOR:u32=${SVO_NODE_MIP_LAYOUT.interiorSize}u;
const RECORD_WORDS:u32=${LIVE_SVO_DERIVED_WORKLIST.recordWords}u;const ACTIVE:u32=${SVO_BRICK_LIFECYCLE.activeBit}u;const PI:f32=3.141592653589793;
fn linearIndex(gid:vec3u,groups:vec3u)->u32{return gid.x+gid.y*groups.x*256u+gid.z*groups.x*groups.y*256u;}
fn finestLevel()->u32{return params.scratchAtlasPages.w;}
${lane.codec}
${liveSvoRadianceAddressingWGSL}
${liveSvoDeepestLeafWGSL}
fn slotOrigin(slot:u32)->vec3u{let p=params.targetAtlasPages.xyz;return vec3u(slot%p.x,(slot/p.x)%p.y,slot/(p.x*p.y))*PHYSICAL;}
fn scratchOrigin(recordIndex:u32)->vec3u{let p=params.scratchAtlasPages.xyz;return vec3u(recordIndex%p.x,(recordIndex/p.x)%p.y,recordIndex/(p.x*p.y))*PHYSICAL;}
fn physicalCoordinate(index:u32)->vec3u{return vec3u(index%PHYSICAL,(index/PHYSICAL)%PHYSICAL,index/(PHYSICAL*PHYSICAL));}
// At a zero apron this is the identity, which is the point: the shell it used to
// strip held a clamped copy of the very texel the clamp already selects.
fn interiorCoordinate(physical:vec3u)->vec3u{return clamp(physical,vec3u(${APRON}u),vec3u(INTERIOR-1u+${APRON}u))-${APRON}u;}
fn keyBit(lo:u32,hi:u32,bit:u32)->u32{if(bit>=32u){return(hi>>(bit-32u))&1u;}return(lo>>bit)&1u;}
fn morton(lo:u32,hi:u32,level:u32)->vec3u{var p=vec3u(0u);for(var bit=0u;bit<level;bit+=1u){let s=1u<<bit;p+=vec3u(keyBit(lo,hi,3u*bit),keyBit(lo,hi,3u*bit+1u),keyBit(lo,hi,3u*bit+2u))*s;}return p;}
fn leafLocal(globalCell:vec3u,leaf:u32)->vec3u{let brickSize=control[11];let finest=params.scratchAtlasPages.w;let node=topology[control[16]+leaf*4u];let nodeBase=node*8u;let level=topology[nodeBase+2u];let scale=1u<<(finest-level);let origin=morton(topology[nodeBase],topology[nodeBase+1u],level)*scale*brickSize;return min((globalCell-origin)/scale,vec3u(brickSize-1u));}
fn leafVoxel(leafIndex:u32,local:vec3u)->u32{let leafBase=control[16]+leafIndex*4u;return topology[leafBase+1u]+local.x+local.y*control[11]+local.z*control[11]*control[11];}
${solidDistance}
fn safeNormal(leafIndex:u32,local:vec3u)->vec3f{let lo=max(local,vec3u(1u))-1u;let hi=min(local+1u,vec3u(control[11]-1u));let gradient=vec3f(
 solidDistance(leafIndex,vec3u(hi.x,local.y,local.z))-solidDistance(leafIndex,vec3u(lo.x,local.y,local.z)),
 solidDistance(leafIndex,vec3u(local.x,hi.y,local.z))-solidDistance(leafIndex,vec3u(local.x,lo.y,local.z)),
 solidDistance(leafIndex,vec3u(local.x,local.y,hi.z))-solidDistance(leafIndex,vec3u(local.x,local.y,lo.z)));
 return select(vec3f(0.,1.,0.),normalize(gradient),dot(gradient,gradient)>1e-12);}
fn writeRadiance(coordinate:vec3u,a:vec3f,b:vec3f,c:vec3f,d:vec3f){let depth=radianceLobeDepth();
 textureStore(radianceScratch,coordinate,vec4f(a,1.));textureStore(radianceScratch,coordinate+vec3u(0u,0u,depth),vec4f(b,1.));
 textureStore(radianceScratch,coordinate+vec3u(0u,0u,depth*2u),vec4f(c,1.));textureStore(radianceScratch,coordinate+vec3u(0u,0u,depth*3u),vec4f(d,1.));}
fn tableSlot(level:u32,page:vec3u)->u32{if(level>=params.direct.x||page.x>=params.direct.y||page.y>=params.direct.z){return INVALID;}let dims=textureDimensions(directTable);let start=params.zOffsets[level];var end=dims.z;if(level+1u<params.direct.x){end=params.zOffsets[level+1u];}if(page.z>=end-start){return INVALID;}let encoded=textureLoad(directTable,vec3i(i32(page.x),i32(page.y),i32(start+page.z)),0).x;return select(INVALID,encoded-1u,encoded>0u);}
struct SceneSample{coverage:f32,radiance:SvoTetraRadiance,valid:u32}
// Callers must pass a level at or above the radiance floor; below it there is no
// radiance page to gather from, and the opacity there is finer than this solve
// resolves anyway.
fn sceneSample(positionIn:vec3f,level:u32)->SceneSample{if(any(positionIn<vec3f(0.0))){return SceneSample(0.,SvoTetraRadiance(vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.)),1u);}let scale=exp2(f32(level));let levelVoxel=positionIn/scale;let pageFloor=floor(levelVoxel/f32(INTERIOR));if(any(pageFloor<vec3f(0.0))||any(pageFloor>=vec3f(2097152.0))){return SceneSample(0.,SvoTetraRadiance(vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.)),1u);}let page=vec3u(pageFloor);let slot=tableSlot(level,page);if(slot==INVALID||!radianceSlotResident(slot)){return SceneSample(0.,SvoTetraRadiance(vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.)),1u);}let opacityDimensions=textureDimensions(opacityValidity);let radianceDimensions=textureDimensions(radianceValidity);if(textureLoad(opacityValidity,svoDerivedPageValidityTexel(opacityDimensions,slot),0).x==0u||textureLoad(radianceValidity,svoDerivedPageValidityTexel(radianceDimensions,slot),0).x==0u){return SceneSample(0.,SvoTetraRadiance(vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.)),0u);}let interiorTexel=clamp(levelVoxel-vec3f(page)*f32(INTERIOR)-vec3f(.5),vec3f(${INTERIOR_CLAMP_LOW}),vec3f(f32(INTERIOR)-${INTERIOR_CLAMP_HIGH}));let opacityUv=(vec3f(slotOrigin(slot))+vec3f(f32(${APRON}u)+.5)+interiorTexel)/vec3f(textureDimensions(opacitySource));let radianceUv=(vec3f(radianceSlotOrigin(slot))+vec3f(f32(${APRON}u)+.5)+interiorTexel)/vec3f(textureDimensions(radianceSource0));return SceneSample(textureSampleLevel(opacitySource,atlasSampler,opacityUv,0.0).x,SvoTetraRadiance(textureSampleLevel(radianceSource0,atlasSampler,radianceUv,0.0).rgb,textureSampleLevel(radianceSource1,atlasSampler,radianceUv,0.0).rgb,textureSampleLevel(radianceSource2,atlasSampler,radianceUv,0.0).rgb,textureSampleLevel(radianceSource3,atlasSampler,radianceUv,0.0).rgb),1u);}
fn hemisphereDirection(normal:vec3f,index:u32)->vec3f{var basis=svoTetraDirection0();if(index==1u){basis=svoTetraDirection1();}else if(index==2u){basis=svoTetraDirection2();}else if(index==3u){basis=svoTetraDirection3();}basis=select(-basis,basis,dot(basis,normal)>=0.0);return normalize(normal*.45+basis);}
// The four gather steps are expressed in *receiver* widths, not in finest
// voxels: the base of the radiance chain is now a floor-level texel spanning
// 2^floor cells, and a 2.5-cell first step from an eight-cell block lands inside
// the block that emitted it. Scaling the distances and shifting the level ladder
// by the same factor keeps every step's ratio of distance to sample width
// exactly what it was when the chain was based at the leaf.
fn incidentAlong(origin:vec3f,normal:vec3f,index:u32)->vec3f{let direction=hemisphereDirection(normal,index);var transmittance=1.0;var incident=vec3f(0.0);
 let floorLevel=radianceFloorLevel();let floorScale=exp2(f32(floorLevel));
 for(var step=0u;step<4u;step+=1u){var distance=2.5;if(step==1u){distance=6.0;}else if(step==2u){distance=18.0;}else if(step==3u){distance=54.0;}let level=min(step+floorLevel,params.direct.x-1u);let sample=sceneSample(origin+direction*distance*floorScale,level);if(sample.valid==0u){return vec3f(0.0);}let coverage=clamp(sample.coverage,0.0,1.0);incident+=transmittance*svoTetraRadianceAlong(sample.radiance,-direction);transmittance*=1.0-coverage;if(transmittance<=0.01){break;}}
 return incident+transmittance*svoEnvironmentDiffuseIrradiance(environment[0],normal)/PI;}
fn directVisibility(originCells:vec3f,towardLight:vec3f,maximumDistance_m:f32${directOcclusion.parameter})->f32{${directOcclusion.guard}let cellScale=max(params.mappingCellSize.x,max(params.mappingCellSize.y,params.mappingCellSize.z))*exp2(f32(radianceFloorLevel()));var transmittance=1.0;for(var step=0u;step<4u;step+=1u){var distance_m=cellScale*2.5;if(step==1u){distance_m=cellScale*6.0;}else if(step==2u){distance_m=cellScale*18.0;}else if(step==3u){distance_m=cellScale*54.0;}if(maximumDistance_m>0.0){distance_m=min(distance_m,maximumDistance_m*.9);}let position=originCells+towardLight*distance_m/params.mappingCellSize.xyz;let sample=sceneSample(position,min(step+radianceFloorLevel(),params.direct.x-1u));if(sample.valid==0u){return 0.0;}transmittance*=1.0-clamp(sample.coverage,0.0,1.0);}return clamp(transmittance,0.0,1.0);}
fn directIncident(worldPosition:vec3f,originCells:vec3f,normal:vec3f${directOcclusion.parameter})->vec3f{var result=vec3f(0.0);let count=min(min(params.direct.w,arrayLength(&lights)),4u);for(var index=0u;index<4u;index+=1u){if(index>=count){break;}let light=lights[index];let base=svoLightRadiance(light);var toward=light.directionCone.xyz;var maximumDistance_m=0.0;var attenuation=1.0;if(light.identity.x!=SVO_LIGHT_DIRECTIONAL){let offset=light.positionRange.xyz-worldPosition;let distanceSquared=dot(offset,offset);if(distanceSquared<=1e-10||(light.positionRange.w>0.0&&distanceSquared>=light.positionRange.w*light.positionRange.w)){continue;}maximumDistance_m=sqrt(distanceSquared);toward=offset/maximumDistance_m;let rangeFade=select(1.0,pow(clamp(1.0-maximumDistance_m/max(light.positionRange.w,1e-6),0.0,1.0),2.0),light.positionRange.w>0.0);attenuation=rangeFade/max(1.0,distanceSquared);if(light.identity.x==SVO_LIGHT_SPHERE_AREA){let area=4.0*PI*light.shape.x*light.shape.x;attenuation=rangeFade*area/max(area,distanceSquared);}else if(light.identity.x==SVO_LIGHT_RECTANGLE_AREA){let area=4.0*light.axisUWidth.w*light.axisVHeight.w;attenuation=rangeFade*max(dot(normalize(light.directionCone.xyz),-toward),0.0)*area/max(area,distanceSquared);}else if(light.identity.x==SVO_LIGHT_SPOT){attenuation*=svoLightConeFalloff(light,toward);}}let cosine=max(dot(normal,toward),0.0);if(cosine>0.0&&directVisibility(originCells,toward,maximumDistance_m${directOcclusion.forward})>0.0){result+=base*(attenuation*cosine/PI);}}return result;}
fn childRadiance(record:u32,local:vec3u,sampleIndex:u32,lobe:u32)->vec3f{let bit=vec3u(sampleIndex&1u,(sampleIndex>>1u)&1u,(sampleIndex>>2u)&1u);let fine=local*2u+bit;let octant=(fine.x/INTERIOR)|((fine.y/INTERIOR)<<1u)|((fine.z/INTERIOR)<<2u);let child=worklist[record+${LIVE_SVO_DERIVED_WORKLIST.childSlotWord}u+octant];if(child==INVALID||!radianceSlotResident(child)){return vec3f(0.);}let coordinate=vec3i(radianceSlotOrigin(child)+(fine%INTERIOR)+${APRON}u);if(lobe==0u){return textureLoad(radianceSource0,coordinate,0).rgb;}if(lobe==1u){return textureLoad(radianceSource1,coordinate,0).rgb;}if(lobe==2u){return textureLoad(radianceSource2,coordinate,0).rgb;}return textureLoad(radianceSource3,coordinate,0).rgb;}
fn childrenReady(record:u32)->bool{for(var childIndex=0u;childIndex<8u;childIndex+=1u){let child=worklist[record+${LIVE_SVO_DERIVED_WORKLIST.childSlotWord}u+childIndex];if(child!=INVALID&&textureLoad(radianceValidity,svoDerivedPageValidityTexel(textureDimensions(radianceValidity),child),0).x==0u){return false;}}return true;}
/**
 * Outward normal at the scale of one floor-level texel, from its own opacity page.
 *
 * The taps are clamped inside the page rather than allowed to walk into an
 * apron. That is not an approximation of the old behaviour, it *is* the old
 * behaviour: an apron texel held a clamped copy of the edge texel, so a tap into
 * it and a tap clamped to the edge read the same byte. Without the clamp a zero
 * apron would read the adjacent atlas slot, which is a different page of the
 * world.
 */
fn pageGradientNormal(pageOrigin:vec3u,center:vec3u)->vec3f{
 let lo=max(center,vec3u(1u))-vec3u(1u);
 let hi=min(center+vec3u(1u),vec3u(PHYSICAL-1u));
 let gradient=vec3f(
  textureLoad(opacitySource,vec3i(pageOrigin+vec3u(hi.x,center.y,center.z)),0).x-textureLoad(opacitySource,vec3i(pageOrigin+vec3u(lo.x,center.y,center.z)),0).x,
  textureLoad(opacitySource,vec3i(pageOrigin+vec3u(center.x,hi.y,center.z)),0).x-textureLoad(opacitySource,vec3i(pageOrigin+vec3u(center.x,lo.y,center.z)),0).x,
  textureLoad(opacitySource,vec3i(pageOrigin+vec3u(center.x,center.y,hi.z)),0).x-textureLoad(opacitySource,vec3i(pageOrigin+vec3u(center.x,center.y,lo.z)),0).x);
 if(dot(gradient,gradient)<=1e-8){return vec3f(0.);}
 return -normalize(gradient);}
@compute @workgroup_size(256) fn feedbackPages(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){let index=linearIndex(gid,groups);let texels=PHYSICAL*PHYSICAL*PHYSICAL;let recordIndex=index/texels;let count=min(worklist[0],params.limits.x);if(recordIndex>=count){return;}let physical=physicalCoordinate(index-recordIndex*texels);let local=interiorCoordinate(physical);let record=worklist[3]+recordIndex*RECORD_WORDS;let level=worklist[2];let floorLevel=radianceFloorLevel();let destination=radianceScratchOrigin(recordIndex)+physical;
 // Below the floor a page carries no radiance at all. It still certifies itself
 // valid so its parents are never blocked waiting for a page that will not come.
 if(level<floorLevel){if(all(physical==vec3u(0u))){scratchValidity[recordIndex]=worklist[1];}return;}
 if(level>floorLevel){if(all(physical==vec3u(0u))){scratchValidity[recordIndex]=select(0u,worklist[1],childrenReady(record));}var r0=vec3f(0.);var r1=vec3f(0.);var r2=vec3f(0.);var r3=vec3f(0.);for(var child=0u;child<8u;child+=1u){r0+=childRadiance(record,local,child,0u)/8.;r1+=childRadiance(record,local,child,1u)/8.;r2+=childRadiance(record,local,child,2u)/8.;r3+=childRadiance(record,local,child,3u)/8.;}writeRadiance(destination,r0,r1,r2,r3);return;}
 if(all(physical==vec3u(0u))){scratchValidity[recordIndex]=worklist[1];}let slot=worklist[record];let pageOrigin=slotOrigin(slot);let center=local+vec3u(${APRON}u);let coverage=textureLoad(opacitySource,vec3i(pageOrigin+center),0).x;let page=vec3u(worklist[record+1u],worklist[record+2u],worklist[record+3u]);
 // At the floor level one texel spans 2^level cells, so the eight per-octant
 // leaves a base record carries no longer address it; the material comes from a
 // descent at the texel's own centre cell instead.
 let scale=1u<<level;let globalCell=(page*INTERIOR+local)*scale+vec3u(scale>>1u);
 var leaf=worklist[record+${LIVE_SVO_DERIVED_WORKLIST.sourceLeafWord}u+((local.x/4u)|((local.y/4u)<<1u)|((local.z/4u)<<2u))];
 if(level>0u){leaf=deepestLeaf(globalCell);}
 if(coverage<=0.0||leaf>=control[1]){writeRadiance(destination,vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.));return;}let sampleLocal=leafLocal(globalCell,leaf);
 let voxel=leafVoxel(leaf,sampleLocal);let dynamicIdentity=${lane.dynamicIdentity};let sceneIdentity=sceneIdentityAt(voxel);let dynamicCoverage=${lane.dynamicCoverage};let sceneCoverage=${lane.sceneCoverage};let materialIndex=select(dynamicIdentity&0xffffu,sceneIdentity&0xffffu,sceneCoverage>=dynamicCoverage&&sceneCoverage>0.);if(materialIndex>=arrayLength(&materials)){writeRadiance(destination,vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.));return;}let material=materials[materialIndex];let fineNormal=safeNormal(leaf,sampleLocal);let coarseNormal=pageGradientNormal(pageOrigin,center);let normal=select(fineNormal,coarseNormal,level>0u&&dot(coarseNormal,coarseNormal)>0.5);let originCells=vec3f(globalCell)+vec3f(.5)+normal*${marchOffsetCells};let transportAlbedo=clamp(material.baseColorOpacity.rgb*(1.0-clamp(material.surface.x,0.0,1.0)),vec3f(0.0),vec3f(${LIVE_SVO_RADIANCE_FEEDBACK.maximumTransportAlbedo}));var incident=vec3f(0.0);for(var direction=0u;direction<${LIVE_SVO_RADIANCE_FEEDBACK.directionCount}u;direction+=1u){incident+=incidentAlong(originCells,normal,direction)/f32(${LIVE_SVO_RADIANCE_FEEDBACK.directionCount});}let worldPosition=params.mappingOrigin.xyz+(vec3f(globalCell)+vec3f(.5))*params.mappingCellSize.xyz;incident+=directIncident(worldPosition,originCells,normal${directOcclusion.receiver});let outgoing=max(material.emissiveRoughness.rgb,vec3f(0.0))+transportAlbedo*incident;writeRadiance(destination,outgoing*coverage*max(0.,dot(normal,svoTetraDirection0())),outgoing*coverage*max(0.,dot(normal,svoTetraDirection1())),outgoing*coverage*max(0.,dot(normal,svoTetraDirection2())),outgoing*coverage*max(0.,dot(normal,svoTetraDirection3())));
}`;
}

/** The `full` expansion. Byte-identical to the pre-profile shader. */
export const liveSvoRadianceFeedbackWGSL = liveSvoRadianceFeedbackWGSLFor("full");

export function liveSvoDerivedCopyWGSLFor(
  radianceFormat: GPUTextureFormat = LIVE_SVO_RADIANCE_STORAGE.fallbackFormat,
  opacityFormat: GPUTextureFormat = SVO_NODE_MIP_OPACITY_STORAGE.wideFormat,
): string {
  return /* wgsl */ `
${liveSvoDerivedPageValidityWGSL}
struct Params{targetAtlasPages:vec4u,scratchAtlasPages:vec4u,limits:vec4u,laneOffsets:vec4u,${LIVE_SVO_RADIANCE_PARAMS_FIELDS},bandedLanes:vec4u}
@group(0) @binding(0) var<storage,read> worklist:array<u32>;
@group(0) @binding(1) var opacityScratch:texture_3d<f32>;
@group(0) @binding(2) var opacityDestination:texture_storage_3d<${opacityFormat},write>;
@group(0) @binding(3) var radianceScratch:texture_3d<f32>;
@group(0) @binding(4) var radianceDestination:texture_storage_3d<${radianceFormat},write>;
@group(0) @binding(5) var opacityValidity:texture_storage_2d<r32uint,write>;
@group(0) @binding(6) var radianceValidity:texture_storage_2d<r32uint,write>;
@group(0) @binding(7) var<uniform> params:Params;
@group(0) @binding(8) var<storage,read> scratchValidity:array<u32>;
const PHYSICAL:u32=${SVO_NODE_MIP_LAYOUT.physicalSize}u;const RECORD_WORDS:u32=${LIVE_SVO_DERIVED_WORKLIST.recordWords}u;
fn linearIndex(gid:vec3u,groups:vec3u)->u32{return gid.x+gid.y*groups.x*256u+gid.z*groups.x*groups.y*256u;}
${liveSvoRadianceAddressingWGSL}
fn targetOrigin(slot:u32)->vec3u{let p=params.targetAtlasPages.xyz;return vec3u(slot%p.x,(slot/p.x)%p.y,slot/(p.x*p.y))*PHYSICAL;}
fn scratchOrigin(slot:u32)->vec3u{let p=params.scratchAtlasPages.xyz;return vec3u(slot%p.x,(slot/p.x)%p.y,slot/(p.x*p.y))*PHYSICAL;}
fn physical(index:u32)->vec3u{return vec3u(index%PHYSICAL,(index/PHYSICAL)%PHYSICAL,index/(PHYSICAL*PHYSICAL));}
@compute @workgroup_size(256) fn invalidatePages(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;let count=min(worklist[0],params.limits.x);if(i>=count){return;}let record=worklist[3]+i*RECORD_WORDS;let slot=worklist[record];textureStore(opacityValidity,svoDerivedPageValidityTexel(textureDimensions(opacityValidity),slot),vec4u(0u));textureStore(radianceValidity,svoDerivedPageValidityTexel(textureDimensions(radianceValidity),slot),vec4u(0u));}
@compute @workgroup_size(256) fn copyOpacity(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let index=linearIndex(gid,groups);let texels=PHYSICAL*PHYSICAL*PHYSICAL;let recordIndex=index/texels;let count=min(worklist[0],params.limits.x);if(recordIndex>=count){return;}
 let local=physical(index-recordIndex*texels);let record=worklist[3]+recordIndex*RECORD_WORDS;let slot=worklist[record];let source=vec3i(scratchOrigin(recordIndex)+local);let destination=vec3i(targetOrigin(slot)+local);
 textureStore(opacityDestination,destination,textureLoad(opacityScratch,source,0));if(all(local==vec3u(0u))&&scratchValidity[recordIndex]!=0u){textureStore(opacityValidity,svoDerivedPageValidityTexel(textureDimensions(opacityValidity),slot),vec4u(scratchValidity[recordIndex]));}}
fn copyRadianceLobe(gid:vec3u,groups:vec3u,lobe:u32){let index=linearIndex(gid,groups);let texels=PHYSICAL*PHYSICAL*PHYSICAL;let recordIndex=index/texels;let count=min(worklist[0],params.limits.x);if(recordIndex>=count){return;}
 let local=physical(index-recordIndex*texels);let record=worklist[3]+recordIndex*RECORD_WORDS;let slot=worklist[record];let scratch=radianceScratchOrigin(recordIndex)+local+vec3u(0u,0u,lobe*radianceLobeDepth());
 // Below the floor there is no radiance page to publish, but the level still
 // certifies itself valid: an absent page and a black page answer every cone
 // identically, and a page that never goes valid stalls its parents forever.
 if(radianceSlotResident(slot)){textureStore(radianceDestination,vec3i(radianceSlotOrigin(slot)+local),textureLoad(radianceScratch,vec3i(scratch),0));}
 if(lobe==3u&&all(local==vec3u(0u))&&scratchValidity[recordIndex]!=0u){textureStore(radianceValidity,svoDerivedPageValidityTexel(textureDimensions(radianceValidity),slot),vec4u(scratchValidity[recordIndex]));}}
@compute @workgroup_size(256) fn copyRadiance0(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){copyRadianceLobe(gid,groups,0u);}
@compute @workgroup_size(256) fn copyRadiance1(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){copyRadianceLobe(gid,groups,1u);}
@compute @workgroup_size(256) fn copyRadiance2(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){copyRadianceLobe(gid,groups,2u);}
@compute @workgroup_size(256) fn copyRadiance3(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){copyRadianceLobe(gid,groups,3u);}`;
}

/** The `rgba16float` expansion. Byte-identical to the pre-format-choice shader. */
export const liveSvoDerivedCopyWGSL = liveSvoDerivedCopyWGSLFor();

/**
 * New WebGPU textures are logically zero-initialized. This runtime pass
 * certifies those zero pages against the current live publication before any
 * occupied-page worklist invalidates and overwrites them. It is deliberately
 * independent of scene contents: every address-resident page starts as empty,
 * and ordinary dirty propagation owns every later edit.
 */
export const liveSvoDerivedEmptyInitializationWGSL = /* wgsl */ `
${liveSvoDerivedPageValidityWGSL}
struct Params{targetAtlasPages:vec4u,scratchAtlasPages:vec4u,limits:vec4u,laneOffsets:vec4u,${LIVE_SVO_RADIANCE_PARAMS_FIELDS},bandedLanes:vec4u}
@group(0) @binding(0) var opacityValidity:texture_storage_2d<r32uint,write>;
@group(0) @binding(1) var radianceValidity:texture_storage_2d<r32uint,write>;
@group(0) @binding(2) var<storage,read> generationState:array<u32>;
@group(0) @binding(3) var<uniform> params:Params;
@compute @workgroup_size(256) fn initializeValidEmptyPages(@builtin(global_invocation_id) gid:vec3u){
  let slot=gid.x;if(slot>=params.limits.w){return;}let generation=generationState[params.limits.z];if(generation==0u){return;}
  textureStore(opacityValidity,svoDerivedPageValidityTexel(textureDimensions(opacityValidity),slot),vec4u(generation));
  textureStore(radianceValidity,svoDerivedPageValidityTexel(textureDimensions(radianceValidity),slot),vec4u(generation));
}`;

interface LevelBindings { worklist: LiveSvoDerivedGpuWorklist; invalidate: GPUBindGroup; build: GPUBindGroup; feedback?: GPUBindGroup; opacityCopy: GPUBindGroup; radianceCopy: readonly GPUBindGroup[] }

interface LiveSvoDerivedBuilderPipelineState {
  emptyInitializationPipeline: GPUComputePipeline;
  emptyInitializationBindGroup: GPUBindGroup;
  buildPipeline: GPUComputePipeline;
  feedbackPipeline?: GPUComputePipeline;
  invalidatePipeline: GPUComputePipeline;
  opacityCopyPipeline: GPUComputePipeline;
  radianceCopyPipelines: readonly GPUComputePipeline[];
  levels: readonly LevelBindings[];
}

function compactScratchAtlasPages(capacity: number, maximumTextureDimension3D: number): readonly [number, number, number] {
  const physical = SVO_NODE_MIP_LAYOUT.physicalSize;
  const maximumXyPages = Math.floor(maximumTextureDimension3D / physical);
  const maximumZPages = Math.floor(maximumTextureDimension3D / (physical * 4));
  if (maximumXyPages < 1 || maximumZPages < 1 || capacity > maximumXyPages * maximumXyPages * maximumZPages) {
    throw new RangeError("Live SVO derived scratch work budget exceeds the device 3D texture limit");
  }
  const x = Math.min(capacity, maximumXyPages);
  const y = Math.min(Math.ceil(capacity / x), maximumXyPages);
  return [x, y, Math.ceil(capacity / (x * y))];
}

/** GPU-only incremental builder; work is exactly bounded by GPU worklist counts. */
export class WebGpuLiveSvoDerivedBuilder {
  readonly allocatedBytes: number;
  readonly scratchAtlasPages: readonly [number, number, number];
  /** Sized by the deepest *radiance* level, which the floor makes far shallower. */
  readonly radianceScratchAtlasPages: readonly [number, number, number];
  readonly radianceFloorLevel: number;
  readonly radianceFeedbackEnabled: boolean;
  /** Mirrors the target atlas; scratch, copy destination and WGSL must agree. */
  readonly radianceFormat: GPUTextureFormat;
  /** Mirrors the opacity atlas; scratch, copy destination and WGSL must agree. */
  readonly opacityFormat: GPUTextureFormat;
  private readonly plannedPageCount: number;
  private readonly opacityScratch: GPUTexture;
  private readonly radianceScratch: GPUTexture;
  private readonly scratchValidity: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly feedbackParams?: GPUBuffer;
  private readonly feedbackSampler?: GPUSampler;
  private pipelineState?: LiveSvoDerivedBuilderPipelineState;
  private pipelineInitialization?: Promise<void>;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly options: WebGpuLiveSvoDerivedBuilderOptions) {
    // The two atlases share a *slot* index space — page validity, the claims
    // buffer and every worklist record name one slot for both — but no longer a
    // shape: the radiance floor gives radiance far fewer physical pages.
    if (options.nodeMips.pageCapacity !== options.radiance.pageCapacity) throw new Error("Live opacity and radiance page capacities must match");
    if (options.nodeMips.atlasPages.reduce((product, value) => product * value, 1) < options.nodeMips.pageCapacity) {
      throw new RangeError("Live derived target atlas does not contain its declared page capacity");
    }
    if (options.worklists.length < 1) throw new RangeError("At least one live derived worklist is required");
    if (!Number.isSafeInteger(options.plannedPageCount) || options.plannedPageCount < 1
      || options.plannedPageCount > options.nodeMips.pageCapacity) {
      throw new RangeError("Live derived planned page count must fit the target page atlas");
    }
    this.plannedPageCount = options.plannedPageCount;
    if (!Number.isInteger(options.generationSource.offsetBytes) || options.generationSource.offsetBytes < 0
      || options.generationSource.offsetBytes % 4 !== 0) {
      throw new RangeError("Live derived generation offset must be a nonnegative u32-aligned byte offset");
    }
    if (!Number.isInteger(options.finestLevel) || options.finestLevel < 0 || options.finestLevel > 21) {
      throw new RangeError("Live derived finest level must be an integer in [0, 21]");
    }
    this.radianceFeedbackEnabled = options.radianceFeedback
      ?? Boolean(options.materialPbr && options.environmentLighting && options.lights);
    if (this.radianceFeedbackEnabled && (!options.materialPbr || !options.environmentLighting || !options.lights)) {
      throw new Error("Live SVO radiance feedback requires material, environment, and light buffers");
    }
    if (!Number.isSafeInteger(options.lightCount ?? 0) || (options.lightCount ?? 0) < 0) {
      throw new RangeError("Live SVO feedback light count must be a nonnegative safe integer");
    }
    if ((options.worldOrigin_m ?? [0, 0, 0]).some((value) => !Number.isFinite(value))
      || (options.cellSize_m ?? [1, 1, 1]).some((value) => !Number.isFinite(value) || value <= 0)) {
      throw new RangeError("Live SVO feedback mapping must have a finite origin and positive cell size");
    }
    const scratchCapacity = Math.max(...options.worklists.map(({ capacity }) => capacity));
    if (!Number.isSafeInteger(scratchCapacity) || scratchCapacity < 1 || scratchCapacity > options.nodeMips.pageCapacity) {
      throw new RangeError("Live derived worklist capacity must fit the target page atlas");
    }
    const maximumTextureDimension3D = Number(device.limits?.maxTextureDimension3D) || 2_048;
    this.scratchAtlasPages = options.scratchAtlasPages ?? compactScratchAtlasPages(scratchCapacity, maximumTextureDimension3D);
    if (this.scratchAtlasPages.some((value) => !Number.isSafeInteger(value) || value < 1)
      || this.scratchAtlasPages.reduce((product, value) => product * value, 1) < scratchCapacity
      || this.scratchAtlasPages[0] * SVO_NODE_MIP_LAYOUT.physicalSize > maximumTextureDimension3D
      || this.scratchAtlasPages[1] * SVO_NODE_MIP_LAYOUT.physicalSize > maximumTextureDimension3D
      || this.scratchAtlasPages[2] * SVO_NODE_MIP_LAYOUT.physicalSize * 4 > maximumTextureDimension3D) {
      throw new RangeError("Live derived scratch atlas cannot contain the bounded worklist on this device");
    }
    const label = options.label ?? "Live SVO derived content";
    this.radianceFormat = options.radiance.format ?? LIVE_SVO_RADIANCE_STORAGE.fallbackFormat;
    this.opacityFormat = options.nodeMips.format ?? SVO_NODE_MIP_OPACITY_STORAGE.wideFormat;
    this.radianceFloorLevel = options.radianceFloorLevel ?? 0;
    if (!Number.isSafeInteger(this.radianceFloorLevel) || this.radianceFloorLevel < 0
      || this.radianceFloorLevel >= options.worklists.length) {
      throw new RangeError("Live derived radiance floor must be a level the worklists cover");
    }
    const radianceScratchCapacity = Math.max(1, Math.min(scratchCapacity, options.radianceScratchCapacity ?? scratchCapacity));
    this.radianceScratchAtlasPages = compactScratchAtlasPages(radianceScratchCapacity, maximumTextureDimension3D);
    const scratchTexels = this.scratchAtlasPages.map((value) => value * SVO_NODE_MIP_LAYOUT.physicalSize) as [number, number, number];
    const radianceScratchTexels = this.radianceScratchAtlasPages.map((value) => value * SVO_NODE_MIP_LAYOUT.physicalSize) as [number, number, number];
    this.opacityScratch = device.createTexture({ label: `${label} opacity scratch`, size: scratchTexels, dimension: "3d", format: this.opacityFormat,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
    this.radianceScratch = device.createTexture({ label: `${label} radiance scratch`, size: [radianceScratchTexels[0], radianceScratchTexels[1], radianceScratchTexels[2] * 4],
      dimension: "3d", format: this.radianceFormat, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
    this.scratchValidity = device.createBuffer({ label: `${label} scratch page validity`, size: scratchCapacity * 4,
      usage: GPUBufferUsage.STORAGE });
    // The banded lane bases ride at the end of both parameter blocks, so
    // `configureRadianceSlotOffset`'s two hard-coded write offsets — and every
    // other field's — stay exactly where they were.
    const bandedLanes = options.tree.bandedLaneWordOffsets.slice(0, 4);
    this.params = device.createBuffer({ label: `${label} parameters`, size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.params, 0, new Uint32Array([
      ...options.nodeMips.atlasPages, 0,
      ...this.scratchAtlasPages, options.finestLevel,
      options.nodeMips.pageCapacity, SVO_NODE_MIP_LAYOUT.physicalSize,
      options.generationSource.offsetBytes / 4, options.plannedPageCount,
      options.tree.velocityOffsetBytes / 4, options.tree.materialOwnerOffsetBytes / 4,
      options.tree.sceneGeometryOffsetBytes / 4, options.tree.scenePayloadLanes.materialOwnerWords,
      ...options.radiance.atlasPages, options.radiance.slotOffset ?? 0,
      ...this.radianceScratchAtlasPages, this.radianceFloorLevel,
      ...bandedLanes,
    ]));
    if (this.radianceFeedbackEnabled) {
      this.feedbackSampler = device.createSampler({ label: `${label} radiance feedback sampler`,
        addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge",
        minFilter: "linear", magFilter: "linear", mipmapFilter: "nearest" });
      this.feedbackParams = device.createBuffer({ label: `${label} radiance feedback parameters`, size: 208,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const feedbackData = new ArrayBuffer(208), feedbackWords = new Uint32Array(feedbackData), feedbackFloats = new Float32Array(feedbackData);
      feedbackWords.set([...options.nodeMips.atlasPages, 0], 0);
      feedbackWords.set([...this.scratchAtlasPages, options.finestLevel], 4);
      feedbackWords.set([options.nodeMips.pageCapacity, SVO_NODE_MIP_LAYOUT.physicalSize,
        options.generationSource.offsetBytes / 4, options.plannedPageCount], 8);
      feedbackWords.set([options.tree.velocityOffsetBytes / 4, options.tree.materialOwnerOffsetBytes / 4,
        options.tree.sceneGeometryOffsetBytes / 4, options.tree.scenePayloadLanes.materialOwnerWords], 12);
      feedbackWords.set([options.worklists.length, options.nodeMips.directPageTableDimensions[0],
        options.nodeMips.directPageTableDimensions[1], options.lightCount ?? 0], 16);
      feedbackWords.set(options.nodeMips.directPageTableLevelZOffsets.slice(0, 12), 20);
      feedbackFloats.set([...(options.worldOrigin_m ?? [0, 0, 0]), 0], 32);
      feedbackFloats.set([...(options.cellSize_m ?? [1, 1, 1]), 0], 36);
      feedbackWords.set([...options.radiance.atlasPages, options.radiance.slotOffset ?? 0], 40);
      feedbackWords.set([...this.radianceScratchAtlasPages, this.radianceFloorLevel], 44);
      feedbackWords.set(bandedLanes, 48);
      device.queue.writeBuffer(this.feedbackParams, 0, feedbackData);
    }
    const texelCount = scratchTexels[0] * scratchTexels[1] * scratchTexels[2];
    const radianceTexelCount = radianceScratchTexels[0] * radianceScratchTexels[1] * radianceScratchTexels[2];
    this.allocatedBytes = texelCount * svoNodeMipOpacityChannels(this.opacityFormat)
      + radianceTexelCount * 4 * liveSvoRadianceBytesPerTexel(this.radianceFormat)
      + scratchCapacity * 4 + this.params.size + (this.feedbackParams?.size ?? 0);
  }

  /**
   * Re-point the radiance atlas after a plan growth renumbered its slots.
   *
   * Growth keeps the atlas shape — it was sized for the domain — but changes how
   * many slots fall below the radiance floor, and therefore where the atlas
   * starts. Nothing else in the parameter block moves.
   */
  configureRadianceSlotOffset(slotOffset: number): void {
    if (this.destroyed) return;
    if (!Number.isSafeInteger(slotOffset) || slotOffset < 0) throw new RangeError("Live radiance slot offset must be a non-negative integer");
    this.device.queue.writeBuffer(this.params, 64 + 12, new Uint32Array([slotOffset]));
    if (this.feedbackParams) this.device.queue.writeBuffer(this.feedbackParams, 160 + 12, new Uint32Array([slotOffset]));
  }

  async initializePipelines(): Promise<void> {
    if (this.pipelineState) return;
    if (this.destroyed) throw new Error("Cannot initialize destroyed live SVO derived builder");
    if (!this.pipelineInitialization) {
      this.pipelineInitialization = this.compilePipelineState().then((state) => {
        if (this.destroyed) throw new Error("Live SVO derived builder was destroyed during pipeline initialization");
        this.pipelineState = state;
      });
    }
    try {
      await this.pipelineInitialization;
    } catch (error) {
      this.pipelineInitialization = undefined;
      throw error;
    }
  }

  private async compilePipelineState(): Promise<LiveSvoDerivedBuilderPipelineState> {
    const { device, options } = this;
    const label = options.label ?? "Live SVO derived content";
    // The lane set is a property of the tree, so the shader follows it rather
    // than taking a second, separately-authored opinion about what is resident.
    const profile = options.tree.payloadProfile;
    const sceneGeometry = options.tree.sceneGeometryFormat;
    // The B0.5b gate. Off by default, because off is what ships; when on, the
    // only thing that changes anywhere in the world is what `solidDistance`
    // returns outside the banded record set.
    const bandedCellSize = svoBandedReconstructionEnabled() ? options.cellSize_m : undefined;
    // Gated separately from banding: the skip changes the product frame by itself,
    // so it has to stand on its own hash before banding rides on top of it.
    // Root cause rather than omission: gated on its own hash, off by default.
    const solidMarchOffsetFix = svoSolidMarchOffsetFixEnabled();
    // The second mechanism of the same leak — the occlusion march inheriting the
    // gather's receiver-width steps — and gated on its own hash for the same
    // reason: it changes the hero's direct sun by itself.
    const solidDirectOcclusion = svoSolidDirectOcclusionEnabled();
    const buildModule = device.createShaderModule({
      label: `${label} build shader`,
      code: liveSvoDerivedBuildWGSLFor(
        profile, this.radianceFormat, this.opacityFormat, sceneGeometry, bandedCellSize,
        options.tree.leafPayloadMode),
    });
    const feedbackModule = this.radianceFeedbackEnabled
      ? device.createShaderModule({ label: `${label} radiance feedback shader`,
        code: liveSvoRadianceFeedbackWGSLFor(
          profile, this.radianceFormat, sceneGeometry, bandedCellSize, solidMarchOffsetFix,
          solidDirectOcclusion, options.tree.leafPayloadMode) })
      : undefined;
    const copyModule = device.createShaderModule({ label: `${label} copy shader`,
      code: liveSvoDerivedCopyWGSLFor(this.radianceFormat, this.opacityFormat) });
    const emptyInitializationModule = device.createShaderModule({
      label: `${label} valid-empty initialization shader`, code: liveSvoDerivedEmptyInitializationWGSL,
    });
    const [emptyInitializationPipeline, buildPipeline, feedbackPipeline, invalidatePipeline, opacityCopyPipeline, radianceCopyPipelines] = await Promise.all([
      device.createComputePipelineAsync({ label: `${label} valid-empty initialization pipeline`, layout: "auto",
        compute: { module: emptyInitializationModule, entryPoint: "initializeValidEmptyPages" } }),
      device.createComputePipelineAsync({ label: `${label} build pipeline`, layout: "auto", compute: { module: buildModule, entryPoint: "buildPages" } }),
      feedbackModule ? device.createComputePipelineAsync({ label: `${label} radiance feedback pipeline`, layout: "auto",
        compute: { module: feedbackModule, entryPoint: "feedbackPages" } }) : Promise.resolve(undefined),
      device.createComputePipelineAsync({ label: `${label} invalidate pipeline`, layout: "auto", compute: { module: copyModule, entryPoint: "invalidatePages" } }),
      device.createComputePipelineAsync({ label: `${label} opacity copy pipeline`, layout: "auto", compute: { module: copyModule, entryPoint: "copyOpacity" } }),
      Promise.all([0, 1, 2, 3].map((lobe) => device.createComputePipelineAsync({ label: `${label} radiance copy ${lobe} pipeline`,
        layout: "auto", compute: { module: copyModule, entryPoint: `copyRadiance${lobe}` } }))),
    ]);
    if (this.destroyed) throw new Error("Live SVO derived builder was destroyed during pipeline initialization");
    const emptyInitializationBindGroup = device.createBindGroup({
      layout: emptyInitializationPipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: options.nodeMips.pageValidity.view },
        { binding: 1, resource: options.radiance.pageValidity.view },
        { binding: 2, resource: { buffer: options.generationSource.buffer } },
        { binding: 3, resource: { buffer: this.params } },
      ],
    });
    const radianceScratchView = this.radianceScratch.createView({ dimension: "3d" });
    const targetRadianceViews = options.radiance.textures.map((texture) => texture.createView({ dimension: "3d" }));
    const materialPbr = options.materialPbr, environmentLighting = options.environmentLighting, lights = options.lights;
    const feedbackParams = this.feedbackParams, feedbackSampler = this.feedbackSampler;
    const levels = options.worklists.map((worklist) => ({ worklist,
      invalidate: device.createBindGroup({ layout: invalidatePipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: worklist.buffer, offset: worklist.bindingOffsetBytes ?? 0, size: worklist.bindingSizeBytes } },
        { binding: 5, resource: options.nodeMips.pageValidity.view }, { binding: 6, resource: options.radiance.pageValidity.view },
        { binding: 7, resource: { buffer: this.params } },
      ] }),
      build: device.createBindGroup({ layout: buildPipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: options.tree.control, size: SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes } }, { binding: 1, resource: { buffer: options.tree.topology, offset: options.tree.topologyOffsetBytes } },
        { binding: 2, resource: { buffer: options.tree.payload } }, { binding: 3, resource: { buffer: worklist.buffer, offset: worklist.bindingOffsetBytes ?? 0, size: worklist.bindingSizeBytes } },
        { binding: 4, resource: { buffer: options.materialEmission } }, { binding: 5, resource: options.nodeMips.texture.createView({ dimension: "3d" }) },
        { binding: 6, resource: this.opacityScratch.createView({ dimension: "3d" }) },
        ...targetRadianceViews.map((resource, index) => ({ binding: 7 + index, resource })),
        { binding: 11, resource: radianceScratchView }, { binding: 12, resource: { buffer: this.params } },
        { binding: 13, resource: options.nodeMips.pageValidity.view }, { binding: 14, resource: options.radiance.pageValidity.view },
        { binding: 15, resource: { buffer: this.scratchValidity } },
      ] }),
      feedback: feedbackPipeline && materialPbr && environmentLighting && lights && feedbackParams && feedbackSampler
        ? device.createBindGroup({ layout: feedbackPipeline.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: options.tree.control, size: SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes } },
          { binding: 1, resource: { buffer: options.tree.topology, offset: options.tree.topologyOffsetBytes } },
          { binding: 2, resource: { buffer: options.tree.payload } },
          { binding: 3, resource: { buffer: worklist.buffer, offset: worklist.bindingOffsetBytes ?? 0, size: worklist.bindingSizeBytes } },
          { binding: 4, resource: options.nodeMips.texture.createView({ dimension: "3d" }) },
          ...targetRadianceViews.map((resource, index) => ({ binding: 5 + index, resource })),
          { binding: 9, resource: radianceScratchView }, { binding: 10, resource: { buffer: feedbackParams } },
          { binding: 11, resource: options.nodeMips.directPageTableTexture.createView({ dimension: "3d" }) },
          { binding: 12, resource: { buffer: materialPbr } }, { binding: 13, resource: { buffer: environmentLighting } },
          { binding: 14, resource: options.nodeMips.pageValidity.view }, { binding: 15, resource: options.radiance.pageValidity.view },
          { binding: 16, resource: { buffer: this.scratchValidity } }, { binding: 17, resource: { buffer: lights } },
          { binding: 18, resource: feedbackSampler },
        ] }) : undefined,
      opacityCopy: device.createBindGroup({ layout: opacityCopyPipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: worklist.buffer, offset: worklist.bindingOffsetBytes ?? 0, size: worklist.bindingSizeBytes } }, { binding: 1, resource: this.opacityScratch.createView({ dimension: "3d" }) },
        { binding: 2, resource: options.nodeMips.texture.createView({ dimension: "3d" }) },
        { binding: 5, resource: options.nodeMips.pageValidity.view }, { binding: 7, resource: { buffer: this.params } },
        { binding: 8, resource: { buffer: this.scratchValidity } },
      ] }),
      radianceCopy: radianceCopyPipelines.map((pipeline, lobe) => device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: worklist.buffer, offset: worklist.bindingOffsetBytes ?? 0, size: worklist.bindingSizeBytes } }, { binding: 3, resource: radianceScratchView },
        { binding: 4, resource: targetRadianceViews[lobe] }, { binding: 6, resource: options.radiance.pageValidity.view },
        { binding: 7, resource: { buffer: this.params } }, { binding: 8, resource: { buffer: this.scratchValidity } },
      ] })),
    }));
    return { emptyInitializationPipeline, emptyInitializationBindGroup, buildPipeline, feedbackPipeline, invalidatePipeline,
      opacityCopyPipeline, radianceCopyPipelines, levels };
  }

  encode(encoder: GPUCommandEncoder, initializeEmpty = false): void {
    if (this.destroyed) return;
    const state = this.pipelineState;
    if (!state) throw new Error("Live SVO derived builder pipelines are not initialized");
    if (initializeEmpty) {
      const pass = encoder.beginComputePass({ label: "Certify live SVO address pages as empty" });
      pass.setPipeline(state.emptyInitializationPipeline); pass.setBindGroup(0, state.emptyInitializationBindGroup);
      pass.dispatchWorkgroups(Math.ceil(this.plannedPageCount / 256)); pass.end();
    }
    // All affected pages fail closed before any level is rebuilt.
    for (const level of state.levels) {
      const pass = encoder.beginComputePass({ label: "Invalidate live SVO derived pages" });
      pass.setPipeline(state.invalidatePipeline); pass.setBindGroup(0, level.invalidate);
      pass.dispatchWorkgroups(Math.ceil(level.worklist.capacity / 256)); pass.end();
    }
    // Finest-to-coarsest order is the caller's worklist order. Each copy pass
    // makes complete children visible before the next parent build begins.
    for (const level of state.levels) {
      const indirect = level.worklist.indirectOffsetBytes ?? (level.worklist.bindingOffsetBytes ?? 0) + LIVE_SVO_DERIVED_WORKLIST.dispatchIndirectOffsetBytes;
      const build = encoder.beginComputePass({ label: "Build live SVO derived pages" });
      build.setPipeline(state.buildPipeline); build.setBindGroup(0, level.build); build.dispatchWorkgroupsIndirect(level.worklist.buffer, indirect); build.end();
      const opacity = encoder.beginComputePass({ label: "Publish live SVO opacity pages" });
      opacity.setPipeline(state.opacityCopyPipeline); opacity.setBindGroup(0, level.opacityCopy); opacity.dispatchWorkgroupsIndirect(level.worklist.buffer, indirect); opacity.end();
      state.radianceCopyPipelines.forEach((pipeline, lobe) => {
        const radiance = encoder.beginComputePass({ label: `Publish live SVO radiance lobe ${lobe}` });
        radiance.setPipeline(pipeline); radiance.setBindGroup(0, level.radianceCopy[lobe]);
        radiance.dispatchWorkgroupsIndirect(level.worklist.buffer, indirect); radiance.end();
      });
    }
  }

  /** Publish one GPU-compacted phase of previous-frame diffuse feedback. */
  encodeRadianceFeedback(encoder: GPUCommandEncoder): void {
    if (this.destroyed || !this.radianceFeedbackEnabled) return;
    const state = this.pipelineState;
    if (!state?.feedbackPipeline) throw new Error("Live SVO radiance feedback pipeline is not initialized");
    for (const level of state.levels) {
      if (!level.feedback) continue;
      const indirect = level.worklist.indirectOffsetBytes
        ?? (level.worklist.bindingOffsetBytes ?? 0) + LIVE_SVO_DERIVED_WORKLIST.dispatchIndirectOffsetBytes;
      const feedback = encoder.beginComputePass({ label: "Feed back live SVO diffuse radiance" });
      feedback.setPipeline(state.feedbackPipeline); feedback.setBindGroup(0, level.feedback);
      feedback.dispatchWorkgroupsIndirect(level.worklist.buffer, indirect); feedback.end();
      state.radianceCopyPipelines.forEach((pipeline, lobe) => {
        const radiance = encoder.beginComputePass({ label: `Publish feedback SVO radiance lobe ${lobe}` });
        radiance.setPipeline(pipeline); radiance.setBindGroup(0, level.radianceCopy[lobe]);
        radiance.dispatchWorkgroupsIndirect(level.worklist.buffer, indirect); radiance.end();
      });
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.opacityScratch.destroy(); this.radianceScratch.destroy(); this.scratchValidity.destroy(); this.params.destroy(); this.feedbackParams?.destroy(); this.destroyed = true;
  }
}

function align256(value: number): number { return Math.ceil(value / 256) * 256; }

interface LiveSvoDerivedPlannerSourceAllocation {
  source: LiveSvoDirtyLeafSource;
  capacity: number;
  params: GPUBuffer;
}

interface LiveSvoDerivedPlannerPipelineState {
  populatePipeline: GPUComputePipeline;
  finalizePipeline: GPUComputePipeline;
  sources: readonly { capacity: number; bindGroup: GPUBindGroup }[];
  initial: { capacity: number; bindGroup: GPUBindGroup };
  feedback: { capacity: number; bindGroup: GPUBindGroup };
  finalizeBindGroup: GPUBindGroup;
}

/** GPU compaction/deduplication from a unified dirty-leaf stream into level worklists. */
export class WebGpuLiveSvoDerivedWorklistPlanner {
  readonly worklists: readonly LiveSvoDerivedGpuWorklist[];
  readonly allocatedBytes: number;
  private readonly arena: GPUBuffer;
  private readonly claims: GPUBuffer;
  private readonly sources: readonly LiveSvoDerivedPlannerSourceAllocation[];
  private readonly initial: LiveSvoDerivedPlannerSourceAllocation;
  private readonly feedback: LiveSvoDerivedPlannerSourceAllocation;
  private readonly sectionOffsetsBytes: readonly number[];
  private readonly pageCapacityByLevel: readonly number[];
  private pipelineState?: LiveSvoDerivedPlannerPipelineState;
  private pipelineInitialization?: Promise<void>;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly options: WebGpuLiveSvoDerivedWorklistPlannerOptions) {
    if (!Number.isInteger(options.levelCount) || options.levelCount < 1 || options.levelCount > 12) throw new RangeError("Live derived level count must be in [1, 12]");
    const requestedCapacity = options.pageCapacityPerLevel;
    const capacities = typeof requestedCapacity === "number"
      ? Array.from({ length: options.levelCount }, () => requestedCapacity)
      : [...requestedCapacity];
    if (capacities.length !== options.levelCount) throw new RangeError("Live derived per-level capacities must name every level");
    if (capacities.some((capacity) => !Number.isInteger(capacity) || capacity < 1)) throw new RangeError("Live derived per-level capacity must be positive");
    this.pageCapacityByLevel = capacities;
    if (options.dirtyLeafSources.length < 1) throw new RangeError("At least one live dirty-leaf source is required");
    for (const source of options.dirtyLeafSources) {
      if (!Number.isInteger(source.recordStrideWords) || source.recordStrideWords < 1) throw new RangeError("Dirty leaf stride must be positive");
      if (!Number.isInteger(source.capacity) || source.capacity < 1) throw new RangeError("Dirty leaf capacity must be positive");
      if ([source.countOffsetBytes, source.recordOffsetBytes].some((offset) => !Number.isInteger(offset) || offset < 0 || offset % 4 !== 0)) {
        throw new RangeError("Dirty leaf offsets must be nonnegative u32-aligned byte offsets");
      }
    }
    if (!Number.isInteger(options.generationSource.offsetBytes) || options.generationSource.offsetBytes < 0 || options.generationSource.offsetBytes % 4 !== 0) {
      throw new RangeError("Derived generation offset must be a nonnegative u32-aligned byte offset");
    }
    // Sections are laid out by prefix sum over the per-level depths, each still
    // 256-aligned so `bindingOffsetBytes` stays a legal storage-buffer offset.
    // Only this line ever assumed a uniform stride; `sectionOffsetsBytes` was
    // already an array and every consumer reads it by index.
    const sectionBytes = capacities.map((capacity) =>
      align256((LIVE_SVO_DERIVED_WORKLIST.headerWords + capacity * LIVE_SVO_DERIVED_WORKLIST.recordWords) * 4));
    const sectionOffsetsBytes: number[] = [];
    let arenaBytes = 0;
    for (const bytes of sectionBytes) { sectionOffsetsBytes.push(arenaBytes); arenaBytes += bytes; }
    this.sectionOffsetsBytes = sectionOffsetsBytes;
    const label = options.label ?? "Live SVO derived worklists";
    this.arena = device.createBuffer({ label: `${label} arena`, size: arenaBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    this.claims = device.createBuffer({ label: `${label} slot generation claims`, size: options.nodeMips.pageCapacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.worklists = this.sectionOffsetsBytes.map((offset, level) => ({ buffer: this.arena, capacity: capacities[level],
      bindingOffsetBytes: offset, bindingSizeBytes: sectionBytes[level], indirectOffsetBytes: offset + LIVE_SVO_DERIVED_WORKLIST.dispatchIndirectOffsetBytes }));
    const createSource = (source: LiveSvoDirtyLeafSource, suffix: string) => {
      const params = device.createBuffer({ label: `${label} ${suffix} parameters`, size: LIVE_SVO_DERIVED_PLANNER_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      return { source, capacity: source.capacity, params };
    };
    this.sources = options.dirtyLeafSources.map((source, index) => createSource(source, `source ${index}`));
    const initialSource: LiveSvoDirtyLeafSource = { buffer: options.tree.control, countOffsetBytes: 0, recordOffsetBytes: 0,
      capacity: options.tree.leafCapacity, recordStrideWords: 1 };
    this.initial = createSource(initialSource, "initial all-leaves");
    this.feedback = createSource(initialSource, "temporal feedback leaves");
    this.configurePlan(options.nodeMips, options.finestLevel, options.levelCount);
    this.allocatedBytes = arenaBytes + options.nodeMips.pageCapacity * 4
      + LIVE_SVO_DERIVED_PLANNER_PARAMS_BYTES * (this.sources.length + 2);
  }

  async initializePipelines(): Promise<void> {
    if (this.pipelineState) return;
    if (this.destroyed) throw new Error("Cannot initialize destroyed live SVO derived worklist planner");
    if (!this.pipelineInitialization) {
      this.pipelineInitialization = this.compilePipelineState().then((state) => {
        if (this.destroyed) throw new Error("Live SVO derived worklist planner was destroyed during pipeline initialization");
        this.pipelineState = state;
      });
    }
    try {
      await this.pipelineInitialization;
    } catch (error) {
      this.pipelineInitialization = undefined;
      throw error;
    }
  }

  private async compilePipelineState(): Promise<LiveSvoDerivedPlannerPipelineState> {
    const { device, options } = this;
    const label = options.label ?? "Live SVO derived worklists";
    const plannerModule = device.createShaderModule({ label: `${label} shader`, code: liveSvoDerivedWorklistWGSL });
    const [populatePipeline, finalizePipeline] = await Promise.all([
      device.createComputePipelineAsync({ label: `${label} populate pipeline`, layout: "auto",
        compute: { module: plannerModule, entryPoint: "populate" } }),
      device.createComputePipelineAsync({ label: `${label} finalize pipeline`, layout: "auto",
        compute: { module: plannerModule, entryPoint: "finalize" } }),
    ]);
    if (this.destroyed) throw new Error("Live SVO derived worklist planner was destroyed during pipeline initialization");
    const bindSource = ({ source, capacity, params }: LiveSvoDerivedPlannerSourceAllocation) => ({ capacity,
      bindGroup: device.createBindGroup({ layout: populatePipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: source.buffer } },
        { binding: 1, resource: { buffer: options.tree.control, size: SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes } },
        { binding: 2, resource: { buffer: options.tree.topology, offset: options.tree.topologyOffsetBytes } },
        { binding: 3, resource: options.nodeMips.directPageTableTexture.createView({ dimension: "3d" }) },
        { binding: 4, resource: { buffer: this.claims } }, { binding: 5, resource: { buffer: this.arena } },
        { binding: 6, resource: { buffer: params } }, { binding: 7, resource: { buffer: options.generationSource.buffer } },
      ] }),
    });
    const sources = this.sources.map(bindSource);
    const initial = bindSource(this.initial);
    const feedback = bindSource(this.feedback);
    const finalizeBindGroup = device.createBindGroup({ layout: finalizePipeline.getBindGroupLayout(0), entries: [
      { binding: 5, resource: { buffer: this.arena } }, { binding: 6, resource: { buffer: this.sources[0].params } },
      { binding: 7, resource: { buffer: options.generationSource.buffer } },
    ] });
    return { populatePipeline, finalizePipeline, sources, initial, feedback, finalizeBindGroup };
  }

  configurePlan(target: WebGpuLiveSvoNodeMipGpuTarget, finestLevel = this.options.finestLevel, levelCount = this.options.levelCount): void {
    if (!Number.isInteger(finestLevel) || finestLevel < 0) throw new RangeError("Live derived finest level must be nonnegative");
    if (!Number.isInteger(levelCount) || levelCount < 1 || levelCount > this.options.levelCount) {
      throw new RangeError("Configured live derived levels must fit the fixed planner allocation");
    }
    const write = (params: GPUBuffer, source: LiveSvoDirtyLeafSource, allLiveMode: 0 | 1 | 2, phase = 0, phaseCount = 1) => {
      const words = new Uint32Array(LIVE_SVO_DERIVED_PLANNER_PARAMS_BYTES / 4);
      words.set([source.countOffsetBytes / 4, this.options.generationSource.offsetBytes / 4,
        allLiveMode === 2 ? phase : source.recordOffsetBytes / 4,
        allLiveMode === 2 ? phaseCount : source.recordStrideWords], 0);
      words.set([finestLevel, levelCount, target.directPageTableDimensions[0], target.directPageTableDimensions[1]], 4);
      // `limits.z` carried `target.pageCapacity`, which no entry point ever read;
      // it now names the radiance floor level, which `emitPage` does read.
      // `limits.y` is the deepest section's depth; the bound both entry points
      // test against is `capacities[level]`, because the sections differ.
      words.set([source.capacity, Math.max(...this.pageCapacityByLevel),
        this.options.radianceFloorLevel ?? 0, allLiveMode], 8);
      words.set(target.directPageTableLevelZOffsets.slice(0, 12), 12);
      words.set(this.sectionOffsetsBytes.map((offset) => offset / 4), 24);
      words.set(this.pageCapacityByLevel, 36);
      this.device.queue.writeBuffer(params, 0, words);
    };
    this.options.dirtyLeafSources.forEach((source, index) => write(this.sources[index].params, source, 0));
    write(this.initial.params, { buffer: this.options.tree.control, countOffsetBytes: 0, recordOffsetBytes: 0,
      capacity: this.options.tree.leafCapacity, recordStrideWords: 1 }, 1);
    write(this.feedback.params, { buffer: this.options.tree.control, countOffsetBytes: 0, recordOffsetBytes: 0,
      capacity: this.options.tree.leafCapacity, recordStrideWords: 1 }, 2);
  }

  /** Compacts all supplied scene/fluid/topology streams into one deduplicated hierarchy. */
  encode(encoder: GPUCommandEncoder): void {
    this.encodeSources(encoder, "dirty");
  }

  /** First publication: unions every currently live tree leaf with all supplied dirty streams entirely on GPU. */
  encodeInitial(encoder: GPUCommandEncoder): void {
    this.encodeSources(encoder, "initial");
  }

  /** Compact one rotating subset of all live leaves for temporal GI feedback. */
  encodeRadianceFeedback(encoder: GPUCommandEncoder, phase: number, phaseCount = LIVE_SVO_RADIANCE_FEEDBACK.phaseCount): void {
    if (!Number.isSafeInteger(phaseCount) || phaseCount < 1 || !Number.isSafeInteger(phase) || phase < 0 || phase >= phaseCount) {
      throw new RangeError("Live SVO feedback phase must fit its positive phase count");
    }
    this.configurePlan(this.options.nodeMips, this.options.finestLevel, this.options.levelCount);
    const words = new Uint32Array([0, this.options.generationSource.offsetBytes / 4, phase, phaseCount]);
    this.device.queue.writeBuffer(this.feedback.params, 0, words);
    this.encodeSources(encoder, "feedback");
  }

  private encodeSources(encoder: GPUCommandEncoder, sourceMode: "dirty" | "initial" | "feedback"): void {
    if (this.destroyed) return;
    const state = this.pipelineState;
    if (!state) throw new Error("Live SVO derived worklist planner pipelines are not initialized");
    encoder.clearBuffer(this.claims);
    for (const offset of this.sectionOffsetsBytes) encoder.clearBuffer(this.arena, offset, LIVE_SVO_DERIVED_WORKLIST.headerWords * 4);
    const populate = encoder.beginComputePass({ label: "Populate live SVO derived worklists" });
    populate.setPipeline(state.populatePipeline);
    if (sourceMode === "initial") {
      populate.setBindGroup(0, state.initial.bindGroup); populate.dispatchWorkgroups(Math.ceil(state.initial.capacity / 64));
    }
    if (sourceMode === "feedback") {
      populate.setBindGroup(0, state.feedback.bindGroup); populate.dispatchWorkgroups(Math.ceil(state.feedback.capacity / 64));
    } else {
      for (const source of state.sources) {
        populate.setBindGroup(0, source.bindGroup); populate.dispatchWorkgroups(Math.ceil(source.capacity / 64));
      }
    }
    populate.end();
    const finalize = encoder.beginComputePass({ label: "Finalize live SVO derived worklists" });
    finalize.setPipeline(state.finalizePipeline); finalize.setBindGroup(0, state.finalizeBindGroup); finalize.dispatchWorkgroups(this.options.levelCount); finalize.end();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.arena.destroy(); this.claims.destroy(); this.sources.forEach(({ params }) => params.destroy()); this.initial.params.destroy(); this.feedback.params.destroy(); this.destroyed = true;
  }
}
