/** Sparse, page-backed mip data derived from a complete unified-octree generation. */

/**
 * Why there is no apron.
 *
 * A page used to be 10³ physical for an 8³ interior so that hardware linear
 * filtering could read one texel past a page boundary in a single fetch — 1.953x
 * of every byte in the atlas, and of every thread the derived builder launches.
 *
 * On the live path that shell never held a neighbour. `buildPages` derives the
 * value it stores from `interiorCoordinate(physical) = clamp(physical,1,8)-1`,
 * so the apron texel is a *replica of the page's own edge texel*, not the
 * adjacent page's. Sampling at interior coordinate `t` with a one-texel replica
 * apron and sampling at `clamp(t, 0, interiorSize-1)` with no apron produce
 * identical results for every `t`, because the two taps the filter blends across
 * the boundary hold the same byte. The apron was paying 1.953x to interpolate a
 * value against itself.
 *
 * (`createSvoNodeMipPageWithApron` *can* fill true neighbour data, and nothing
 * on the production path calls it — only the synthetic cone benchmark and its
 * test, both of which are content with the clamp the live builder already had.)
 *
 * `apron` is retained as a *variable* zero rather than deleted, for two reasons.
 * The arithmetic that used to add it still reads as addressing rather than as a
 * magic constant; and `FLUID_SVO_NODE_MIP_APRON=1` restores the 10³ page exactly
 * — same clamp range, same offsets, same page bytes — so the claim above is an
 * A/B anyone can run rather than an assertion they have to take on trust.
 */
import { completeCooperativeBuild } from "./cooperative-build";

const SVO_NODE_MIP_APRON_ENVIRONMENT_VARIABLE = "FLUID_SVO_NODE_MIP_APRON";

function svoNodeMipApron(): number {
  const raw = typeof process !== "undefined"
    ? process.env?.[SVO_NODE_MIP_APRON_ENVIRONMENT_VARIABLE]
    : undefined;
  if (raw === undefined || raw === "") return 0;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < 0 || value > 1) {
    throw new RangeError(`${SVO_NODE_MIP_APRON_ENVIRONMENT_VARIABLE} must be 0 or 1`);
  }
  return value;
}

const APRON = svoNodeMipApron();
const PHYSICAL_SIZE = 8 + 2 * APRON;

export const SVO_NODE_MIP_LAYOUT = Object.freeze({
  interiorSize: 8,
  apron: APRON,
  physicalSize: PHYSICAL_SIZE,
  /** Widest opacity page. A dry world stores two of these; see `SVO_NODE_MIP_OPACITY_STORAGE`. */
  channelCount: 4,
  bytesPerTexel: 4,
  bytesPerPage: PHYSICAL_SIZE ** 3 * 4,
  keyWords: 4,
  keyBytes: 16,
  directoryBytesPerPage: 32,
  apronEnvironmentVariable: SVO_NODE_MIP_APRON_ENVIRONMENT_VARIABLE,
} as const);

export const SVO_NODE_MIP_LANES = Object.freeze({
  solidMean: 0,
  solidMaximum: 1,
  fluidMean: 2,
  fluidMaximum: 3,
} as const);

/**
 * How wide one opacity texel has to be.
 *
 * The four lanes above are only ever four on a world with a solver. A dry world
 * hard-codes `fluidFraction = "0."` in the derived builder's payload expansion
 * (`webgpu-svo-live-derived-builder.ts`), so `fluidMean` and `fluidMaximum` are
 * provably zero at every level: the base pass writes a literal zero and the
 * parent reduction is a mean and a maximum over zeros. **Half of every opacity
 * page on a dry world was a constant.**
 *
 * Nothing reads them either — the dry renderer's only reads of a node-mip sample
 * are `solidMean` and `solidMaximum` — but the narrow format is still gated on
 * dryness rather than applied everywhere, because a wet world's lanes are live
 * data and the point of this constant is that the *format follows the payload
 * profile*, exactly as the builder's lane expansion already does.
 *
 * `rg8unorm` is only a *storage* format under `texture-formats-tier1`, and the
 * atlas, the builder's scratch and the publish pass all need it as one. A device
 * without the feature keeps the four-channel page; it is a memory optimisation,
 * never a correctness requirement.
 */
export const SVO_NODE_MIP_OPACITY_STORAGE = Object.freeze({
  /** `solidMean`, `solidMaximum`. Dry worlds on a device with the feature. */
  narrowFormat: "rg8unorm" as GPUTextureFormat,
  narrowChannels: 2,
  /** `solidMean`, `solidMaximum`, `fluidMean`, `fluidMaximum`. */
  wideFormat: "rgba8unorm" as GPUTextureFormat,
  wideChannels: 4,
  requiredFeature: "texture-formats-tier1" as GPUFeatureName,
  /** `0` pins the wide page; `1` asks for the narrow one where the device allows it. */
  environmentVariable: "FLUID_SVO_NODE_MIP_NARROW_OPACITY",
} as const);

/** Lanes an opacity page of this format actually stores. */
export function svoNodeMipOpacityChannels(format: GPUTextureFormat): number {
  return format === SVO_NODE_MIP_OPACITY_STORAGE.narrowFormat
    ? SVO_NODE_MIP_OPACITY_STORAGE.narrowChannels
    : SVO_NODE_MIP_OPACITY_STORAGE.wideChannels;
}

/** Physical bytes one opacity page of this format occupies in the atlas. */
export function svoNodeMipPageBytes(format: GPUTextureFormat = SVO_NODE_MIP_OPACITY_STORAGE.wideFormat): number {
  return SVO_NODE_MIP_LAYOUT.physicalSize ** 3 * svoNodeMipOpacityChannels(format);
}

/**
 * The opacity format a world should allocate.
 *
 * Dryness is the payload profile's decision, not this module's; the caller
 * passes it. The environment variable is an A/B lever, so an explicit `0` pins
 * the wide page even on a dry world with the feature present.
 */
export function svoNodeMipOpacityFormat(options: {
  dry: boolean;
  features?: { has(feature: string): boolean };
}): GPUTextureFormat {
  const raw = typeof process !== "undefined"
    ? process.env?.[SVO_NODE_MIP_OPACITY_STORAGE.environmentVariable]
    : undefined;
  if (raw === "0") return SVO_NODE_MIP_OPACITY_STORAGE.wideFormat;
  if (!options.dry) return SVO_NODE_MIP_OPACITY_STORAGE.wideFormat;
  return options.features?.has(SVO_NODE_MIP_OPACITY_STORAGE.requiredFeature)
    ? SVO_NODE_MIP_OPACITY_STORAGE.narrowFormat
    : SVO_NODE_MIP_OPACITY_STORAGE.wideFormat;
}

/**
 * Where the *radiance* atlas stops following the leaf down.
 *
 * Opacity has to live at the finest leaf: it carries the silhouette and the
 * contact shadow, and a cone that reads a coarser coverage loses the arris of
 * the coping and the gap between two florets. Radiance does not. Bounced light
 * is a low-frequency field; a 5 cm irradiance sample reconstructs a garden's
 * indirect term as well as a 6 mm one, and it is 512x cheaper.
 *
 * That matters because radiance is *four* lobes of 10^3 texels — 16 000 B of
 * the 20 000 B a pyramid page costs. Halving the leaf multiplies the page count
 * by eight, so radiance is what makes a fine leaf unaffordable, not opacity.
 *
 * **Why capping is safe.** `planSvoNodeMipPyramid` walks every occupied base
 * page up through every level, inserting an ancestor at each. Residency is
 * therefore *ancestor-closed*: if a page at level `L` is resident, so is its
 * ancestor at every level above it. A radiance lookup at a level finer than the
 * floor can always be redirected to `coordinate >> (floor - level)` at the
 * floor, and that page is guaranteed resident. There is no missing-page case.
 *
 * **Why it is anchored to a world size and not just a level count.** A floor
 * expressed purely as "three levels coarser than the finest" still multiplies
 * radiance by eight every time the leaf halves — from a 512x smaller base, but
 * with the same exponent. Pinning the floor to a *cell size* instead makes the
 * radiance atlas a function of the world's extent alone: refine the leaf and the
 * floor level rises with it, leaving the radiance page count unchanged. At the
 * reference 6.25 mm leaf the two readings coincide, which is where the default
 * "three levels" comes from.
 */
export const SVO_RADIANCE_LEVEL_FLOOR = Object.freeze({
  /** Default depth of the cap at `referenceCellSize_m`. */
  levelsCoarserThanFinest: 3,
  referenceCellSize_m: 0.006_25,
  /** 6.25 mm x 2^3. The world size one radiance texel covers, at any leaf. */
  targetCellSize_m: 0.006_25 * 2 ** 3,
  /** Exact override, in levels above the finest. `0` restores the old behaviour. */
  environmentVariable: "FLUID_SVO_RADIANCE_LEVEL_FLOOR",
} as const);

function radianceLevelFloorOverride(): number | undefined {
  const raw = typeof process !== "undefined"
    ? process.env?.[SVO_RADIANCE_LEVEL_FLOOR.environmentVariable]
    : undefined;
  if (raw === undefined || raw === "") return undefined;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${SVO_RADIANCE_LEVEL_FLOOR.environmentVariable} must be a non-negative integer level`);
  }
  return value;
}

/**
 * The finest virtual mip level that owns a radiance page.
 *
 * Levels below it are opacity-only: no atlas slot, no scratch, no build work.
 * Always clamped into `[0, levelCount - 1]`, so a domain too shallow to hold
 * the cap simply keeps radiance at its own coarsest level rather than losing it.
 */
export function svoRadianceLevelFloor(options: {
  levelCount: number;
  /** Finest leaf edge in metres. Absent keeps the plain level-count reading. */
  cellSize_m?: number;
  /** Explicit level; wins over both the environment and the world anchor. */
  override?: number;
}): number {
  const { levelCount } = options;
  if (!Number.isSafeInteger(levelCount) || levelCount < 1) throw new RangeError("SVO radiance floor needs a positive level count");
  const explicit = options.override ?? radianceLevelFloorOverride();
  if (explicit !== undefined) return Math.max(0, Math.min(levelCount - 1, explicit));
  const cellSize_m = options.cellSize_m;
  const anchored = cellSize_m !== undefined && Number.isFinite(cellSize_m) && cellSize_m > 0
    ? Math.round(Math.log2(SVO_RADIANCE_LEVEL_FLOOR.targetCellSize_m / cellSize_m))
    : SVO_RADIANCE_LEVEL_FLOOR.levelsCoarserThanFinest;
  return Math.max(0, Math.min(levelCount - 1, anchored));
}

/** Slots a plan hands to levels below the floor; the radiance atlas starts after them. */
export function svoNodeMipRadianceSlotOffset(plan: SvoNodeMipPyramidPlan, radianceFloorLevel: number): number {
  // Slots are handed out in level-major order, so every page below the floor
  // occupies a contiguous prefix and the radiance atlas is the suffix shifted
  // down by its length. Counting is exact rather than derived from the domain,
  // because a reserved (non-total) plan holds fewer fine pages than the domain.
  let offset = 0;
  for (const page of plan.pages) if (page.key.level < radianceFloorLevel) offset += 1;
  return offset;
}

/**
 * Where the *opacity* pyramid stops following the leaf down.
 *
 * The base level is the whole cost. Measured on `hero-garden-hose` under the
 * curvature refinement rule, level 0 is 84.9 % of the pyramid's pages at
 * refinement depth 0, 55.9 % at depth 1, 74.1 % at depth 2 and **81.0 % at
 * depth 3** — 415 379 of 512 893 pages, 405.6 MB of a 500.9 MB payload. Every
 * level above it together is a fifth of it, because each level divides the
 * page count by roughly eight.
 *
 * That base level is one texel per finest voxel: a level-0 page is exactly one
 * finest brick (`liveSvoLeafPage`, `brickSize` 8 = `interiorSize` 8). Nothing
 * reads it at that resolution except a cone whose diameter is one voxel, and
 * refining the leaf makes those voxels smaller without making the *cones* any
 * narrower — the same argument that produced `SVO_RADIANCE_LEVEL_FLOOR`, one
 * field over.
 *
 * So the floor is anchored to a **world size**, not to a level count: the
 * finest opacity texel stays at the reference 6.25 mm however fine the tree
 * gets. At the reference leaf the floor is level 0 and every byte, every page
 * and every sample is exactly what shipped; at refinement depth N it rises
 * with N and the atlas stops growing with the leaf.
 *
 * **Why it is exact, and why it stops at one level.** A level-1 page's eight
 * children are level-0 pages, and a level-0 page is one finest brick, so it is
 * one leaf. When the child page is absent the derived worklist resolves the
 * child slot to that leaf (`deepestLeaf` of the child's centre) and the build
 * reads the payload directly — `childScale` is 1 at level 1, so the eight
 * samples the reduction takes per texel are exactly the eight finest cells
 * that texel covers. The level-1 page therefore holds the same mean it always
 * held, at *higher* precision than reading eight quantised level-0 texels.
 *
 * At level 2 that stops being true: a child level-1 page spans eight leaves and
 * the record carries one slot for it, so `deepestLeaf` names one of the eight
 * and `leafLocal` clamps every cell outside it to that leaf's edge voxel. A
 * deeper floor needs the fallback to resolve a leaf *per sample*, which is a
 * change to the build shader rather than to a plan. Until then the maximum is
 * one level, and asking for more is clamped rather than silently wrong.
 */
export const SVO_OPACITY_LEVEL_FLOOR = Object.freeze({
  /** 6.25 mm. The world size one base opacity texel covers, at any leaf. */
  targetCellSize_m: 0.006_25,
  /** Levels the base may rise by before the build fallback stops being exact. */
  maximumLevels: 1,
  /** Exact override, in levels above the finest. `0` restores the old behaviour. */
  environmentVariable: "FLUID_SVO_OPACITY_LEVEL_FLOOR",
} as const);

function opacityLevelFloorOverride(): number | undefined {
  const raw = typeof process !== "undefined"
    ? process.env?.[SVO_OPACITY_LEVEL_FLOOR.environmentVariable]
    : undefined;
  if (raw === undefined || raw === "") return undefined;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${SVO_OPACITY_LEVEL_FLOOR.environmentVariable} must be a non-negative integer level`);
  }
  return value;
}

/**
 * The finest virtual mip level that owns an opacity page.
 *
 * Levels below it hold no page, no atlas slot and no direct-table slab. A
 * sample below the floor reads the floor instead — `dryNodeMipAt` clamps — and
 * that page is guaranteed resident by the same ancestor-closed argument the
 * radiance floor uses.
 *
 * The environment variable is an A/B lever and wins over the anchor, but never
 * over the soundness bound: an override past {@link SVO_OPACITY_LEVEL_FLOOR}'s
 * `maximumLevels` is clamped, because past it the pages would be wrong rather
 * than coarse.
 */
export function svoOpacityLevelFloor(options: {
  levelCount: number;
  /** Finest leaf edge in metres. Absent keeps the base at the finest level. */
  cellSize_m?: number;
  /** Explicit level; wins over both the environment and the world anchor. */
  override?: number;
}): number {
  const { levelCount } = options;
  if (!Number.isSafeInteger(levelCount) || levelCount < 1) throw new RangeError("SVO opacity floor needs a positive level count");
  const ceiling = Math.min(levelCount - 1, SVO_OPACITY_LEVEL_FLOOR.maximumLevels);
  const explicit = options.override ?? opacityLevelFloorOverride();
  if (explicit !== undefined) return Math.max(0, Math.min(ceiling, explicit));
  const cellSize_m = options.cellSize_m;
  const anchored = cellSize_m !== undefined && Number.isFinite(cellSize_m) && cellSize_m > 0
    ? Math.round(Math.log2(SVO_OPACITY_LEVEL_FLOOR.targetCellSize_m / cellSize_m))
    : 0;
  return Math.max(0, Math.min(ceiling, anchored));
}

export type SvoNodeMipCoordinate = readonly [number, number, number];

export interface SvoNodeMipPageKey {
  /** Complete source-octree publication generation. */
  generation: number;
  /** Zero is the finest virtual mip page level. */
  level: number;
  /** Virtual page coordinate at `level`. */
  coordinate: SvoNodeMipCoordinate;
}

export interface SvoNodeMipAtlasShape {
  pages: SvoNodeMipCoordinate;
  texels: SvoNodeMipCoordinate;
  capacity: number;
}

export interface SvoNodeMipPagePlan {
  key: SvoNodeMipPageKey;
  keyString: string;
  slot: number;
  atlasPage: SvoNodeMipCoordinate;
  atlasTexelOrigin: SvoNodeMipCoordinate;
}

export interface SvoNodeMipPyramidPlan {
  generation: number;
  pages: readonly SvoNodeMipPagePlan[];
  atlas: SvoNodeMipAtlasShape;
  requestedPageCount: number;
  residentPageCount: number;
  overflowPageCount: number;
  pagePayloadBytes: number;
  /** Physical atlas allocation, including unused slots introduced by its 3D shape. */
  atlasBytes: number;
  directoryBytes: number;
  allocatedBytes: number;
  complete: boolean;
}

/**
 * One page a source claims, at the level whose texels match its own resolution.
 *
 * A finest-level leaf seeds `level: 0`, which is what a bare coordinate means
 * and what every producer used to emit. A *coarse* leaf holds `brickSize^3`
 * voxels over `2^p` times that extent, so the page whose texels line up with
 * its voxels one-for-one is `p` levels up — and expanding it to `8^p` base
 * pages would materialise the same `brickSize^3` values `8^p` times over. That
 * expansion, not the leaf count, is what made the pyramid scale with volume:
 * see {@link liveSvoLeafPage}.
 */
export interface SvoNodeMipSeedPage {
  level: number;
  coordinate: SvoNodeMipCoordinate;
}

export type SvoNodeMipPageSeed = SvoNodeMipCoordinate | SvoNodeMipSeedPage;

/** A bare coordinate is a finest-level page; anything else names its own level. */
export function svoNodeMipSeedPage(seed: SvoNodeMipPageSeed): SvoNodeMipSeedPage {
  if (Array.isArray(seed)) return { level: 0, coordinate: seed as SvoNodeMipCoordinate };
  const page = seed as SvoNodeMipSeedPage;
  if (!Number.isSafeInteger(page.level) || page.level < 0) {
    throw new RangeError("SVO node-mip seed level must be a non-negative safe integer");
  }
  return page;
}

/** Level-qualified identity of a seed, for plan membership and growth. */
export function svoNodeMipSeedKey(seed: SvoNodeMipPageSeed): string {
  const { level, coordinate: at } = svoNodeMipSeedPage(seed);
  return `${level}:${at[0]},${at[1]},${at[2]}`;
}

/**
 * Raise a seed to the opacity floor, keeping the region it names.
 *
 * A seed below the floor becomes its own ancestor at the floor, which is a page
 * the plan would have inserted anyway — so raising every seed removes levels
 * from the bottom of the pyramid and leaves every level above it with exactly
 * the pages it already had. See {@link SVO_OPACITY_LEVEL_FLOOR}.
 */
export function raiseSvoNodeMipSeedToFloor(seed: SvoNodeMipPageSeed, floorLevel: number): SvoNodeMipSeedPage {
  if (!Number.isSafeInteger(floorLevel) || floorLevel < 0) throw new RangeError("SVO opacity floor level must be a non-negative safe integer");
  const page = svoNodeMipSeedPage(seed);
  if (page.level >= floorLevel) return page;
  const shift = 2 ** (floorLevel - page.level);
  return {
    level: floorLevel,
    coordinate: page.coordinate.map((component) => Math.floor(component / shift)) as unknown as SvoNodeMipCoordinate,
  };
}

export interface SvoNodeMipPlanOptions {
  generation: number;
  /**
   * Pages the topology claims. Ancestor pages are inserted automatically from
   * each seed's own level, so residency stays ancestor-closed whatever level
   * the seeds arrive at.
   */
  occupiedPages: readonly SvoNodeMipPageSeed[];
  levelCount: number;
  capacity?: number;
  atlasPages?: SvoNodeMipCoordinate;
}

export type SvoNodeMipRgba8 = readonly [number, number, number, number];

export interface SvoNodeMipPublication {
  completeGeneration: number;
  plan: SvoNodeMipPyramidPlan;
}

export interface SvoNodeMipPublicationCandidate {
  generation: number;
  plan: SvoNodeMipPyramidPlan;
  directoryComplete: boolean;
  payloadComplete: boolean;
  apronsComplete: boolean;
}

export type SvoNodeMipPublicationDecision =
  | { published: true; reason: "published"; visible: SvoNodeMipPublication }
  | { published: false; reason: "generation-order" | "incomplete-plan" | "incomplete-directory" | "incomplete-payload" | "incomplete-aprons"; visible?: SvoNodeMipPublication };

const UINT32_MAX = 0xffff_ffff;
const MORTON_AXIS_BITS = 21;
const MORTON_AXIS_MAX = (1 << MORTON_AXIS_BITS) - 1;

function uint32(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > UINT32_MAX) throw new RangeError(`${label} must fit uint32`);
  return value >>> 0;
}

function coordinate(value: SvoNodeMipCoordinate, label: string): SvoNodeMipCoordinate {
  if (value.length !== 3 || value.some((component) => !Number.isSafeInteger(component) || component < 0)) {
    throw new RangeError(`${label} must contain three non-negative safe integers`);
  }
  return value;
}

/** 63-bit Morton code; 21 coordinate bits per axis avoids lossy JS bitwise arithmetic. */
export function encodeSvoNodeMipMorton(value: SvoNodeMipCoordinate): bigint {
  coordinate(value, "SVO node-mip coordinate");
  if (value.some((component) => component > MORTON_AXIS_MAX)) {
    throw new RangeError(`SVO node-mip coordinates must fit ${MORTON_AXIS_BITS} bits per axis`);
  }
  let result = 0n;
  for (let bit = 0; bit < MORTON_AXIS_BITS; bit += 1) {
    const shift = BigInt(bit * 3);
    result |= BigInt((value[0] >>> bit) & 1) << shift;
    result |= BigInt((value[1] >>> bit) & 1) << (shift + 1n);
    result |= BigInt((value[2] >>> bit) & 1) << (shift + 2n);
  }
  return result;
}

export function decodeSvoNodeMipMorton(morton: bigint): SvoNodeMipCoordinate {
  if (morton < 0n || morton >= (1n << 63n)) throw new RangeError("SVO node-mip Morton code must fit 63 bits");
  const result = [0, 0, 0];
  for (let bit = 0; bit < MORTON_AXIS_BITS; bit += 1) {
    const shift = BigInt(bit * 3);
    result[0] += Number((morton >> shift) & 1n) * 2 ** bit;
    result[1] += Number((morton >> (shift + 1n)) & 1n) * 2 ** bit;
    result[2] += Number((morton >> (shift + 2n)) & 1n) * 2 ** bit;
  }
  return result as unknown as SvoNodeMipCoordinate;
}

/** Stable CPU/GPU ABI: generation, level, Morton low word, Morton high word. */
export function packSvoNodeMipPageKey(key: SvoNodeMipPageKey): Uint32Array {
  const generation = uint32(key.generation, "SVO node-mip generation");
  const level = uint32(key.level, "SVO node-mip level");
  const morton = encodeSvoNodeMipMorton(key.coordinate);
  return new Uint32Array([generation, level, Number(morton & 0xffff_ffffn), Number(morton >> 32n)]);
}

export function unpackSvoNodeMipPageKey(words: ArrayLike<number>): SvoNodeMipPageKey {
  if (words.length < SVO_NODE_MIP_LAYOUT.keyWords) throw new RangeError("SVO node-mip key requires four words");
  const generation = uint32(words[0], "Packed SVO node-mip generation");
  const level = uint32(words[1], "Packed SVO node-mip level");
  const morton = BigInt(uint32(words[2], "Packed SVO node-mip Morton low word"))
    | BigInt(uint32(words[3], "Packed SVO node-mip Morton high word")) << 32n;
  return { generation, level, coordinate: decodeSvoNodeMipMorton(morton) };
}

export function svoNodeMipPageKey(key: SvoNodeMipPageKey): string {
  const words = packSvoNodeMipPageKey(key);
  return `${words[0]}:${words[1]}:${words[3].toString(16).padStart(8, "0")}${words[2].toString(16).padStart(8, "0")}`;
}

function automaticAtlasShape(capacity: number): SvoNodeMipCoordinate {
  if (capacity === 0) return [0, 0, 0];
  const x = Math.ceil(Math.cbrt(capacity));
  const y = Math.ceil(Math.sqrt(capacity / x));
  const z = Math.ceil(capacity / (x * y));
  return [x, y, z];
}

export function svoNodeMipAtlasPageCoordinate(slot: number, atlasPages: SvoNodeMipCoordinate): SvoNodeMipCoordinate {
  coordinate(atlasPages, "SVO node-mip atlas page dimensions");
  const capacity = atlasPages[0] * atlasPages[1] * atlasPages[2];
  if (!Number.isSafeInteger(slot) || slot < 0 || slot >= capacity) throw new RangeError("SVO node-mip atlas slot is outside the atlas");
  return [slot % atlasPages[0], Math.floor(slot / atlasPages[0]) % atlasPages[1], Math.floor(slot / (atlasPages[0] * atlasPages[1]))];
}

export function planSvoNodeMipPyramid(options: SvoNodeMipPlanOptions): SvoNodeMipPyramidPlan {
  return completeCooperativeBuild(planSvoNodeMipPyramidSteps(options));
}

/** Seeds per yield offer; each walks one chain of `levelCount` map inserts. */
const NODE_MIP_PLAN_YIELD_BATCH = 2048;

/**
 * The same pyramid plan, offered as slices.
 *
 * With the opacity floor in place this is still the longest unyielded block a
 * refined scene build has: 2955 ms at 112k occupied base pages, 567 ms at 24k.
 * It is called twice per address plan — once as the occupancy probe and once
 * for real — and both calls sit in the derived-lighting prologue, before that
 * block creates any device resource at all, so a yield here is as safe as one
 * in the octree plan.
 */
export function* planSvoNodeMipPyramidSteps(
  options: SvoNodeMipPlanOptions,
): Generator<unknown, SvoNodeMipPyramidPlan, undefined> {
  const generation = uint32(options.generation, "SVO node-mip generation");
  if (!Number.isSafeInteger(options.levelCount) || options.levelCount < 1 || options.levelCount > 32) {
    throw new RangeError("SVO node-mip level count must be an integer in [1, 32]");
  }
  const all = new Map<string, SvoNodeMipPageKey>();
  let visited = 0;
  for (const seed of options.occupiedPages) {
    const input = svoNodeMipSeedPage(seed);
    let current = [...coordinate(input.coordinate, "Occupied SVO node-mip page")] as [number, number, number];
    for (let level = input.level; level < options.levelCount; level += 1) {
      const key = { generation, level, coordinate: current as SvoNodeMipCoordinate };
      all.set(svoNodeMipPageKey(key), key);
      current = current.map((component) => Math.floor(component / 2)) as [number, number, number];
    }
    if ((visited += 1) % NODE_MIP_PLAN_YIELD_BATCH === 0) yield;
  }
  // Level-major Morton order permits a binary-search directory lookup without
  // consuming another storage binding in the dry renderer.
  //
  // The Morton code is decorated onto each entry rather than recomputed inside
  // the comparator, for the reason `svo-wide-fanout` gives at its own sort: a
  // sort asks its comparator O(n log n) times and this one encoded two BigInts
  // per call, so a plan of 140k pages paid several million encodes to order a
  // list of n. It matters twice over here — the sort is the one part of this
  // function no yield can interrupt, so its cost *is* the residual stall, and
  // shrinking it is the only way to make that stall smaller.
  const decorated = [...all.entries()].map((entry) => ({
    entry, morton: encodeSvoNodeMipMorton(entry[1].coordinate),
  }));
  yield;
  decorated.sort((a, b) => a.entry[1].level - b.entry[1].level || (a.morton < b.morton ? -1 : 1));
  yield;
  const ordered = decorated.map((value) => value.entry);
  const requestedPageCount = ordered.length;
  const explicitShape = options.atlasPages ? coordinate(options.atlasPages, "SVO node-mip atlas page dimensions") : undefined;
  const explicitCapacity = explicitShape ? explicitShape[0] * explicitShape[1] * explicitShape[2] : Number.MAX_SAFE_INTEGER;
  const requestedCapacity = options.capacity ?? requestedPageCount;
  if (!Number.isSafeInteger(requestedCapacity) || requestedCapacity < 0) throw new RangeError("SVO node-mip capacity must be a non-negative safe integer");
  const capacity = Math.min(requestedCapacity, explicitCapacity);
  const residentPageCount = Math.min(requestedPageCount, capacity);
  const atlasPages = explicitShape ?? automaticAtlasShape(residentPageCount);
  const physical = SVO_NODE_MIP_LAYOUT.physicalSize;
  const pages: SvoNodeMipPagePlan[] = [];
  for (let slot = 0; slot < residentPageCount; slot += 1) {
    const [keyString, key] = ordered[slot];
    const atlasPage = svoNodeMipAtlasPageCoordinate(slot, atlasPages);
    pages.push({ key, keyString, slot, atlasPage, atlasTexelOrigin: atlasPage.map((component) => component * physical) as unknown as SvoNodeMipCoordinate });
    if ((slot + 1) % NODE_MIP_PLAN_YIELD_BATCH === 0) yield;
  }
  const pagePayloadBytes = residentPageCount * SVO_NODE_MIP_LAYOUT.bytesPerPage;
  const atlasBytes = atlasPages[0] * atlasPages[1] * atlasPages[2] * SVO_NODE_MIP_LAYOUT.bytesPerPage;
  const directoryBytes = residentPageCount * SVO_NODE_MIP_LAYOUT.directoryBytesPerPage;
  return {
    generation, pages,
    atlas: {
      pages: atlasPages,
      texels: atlasPages.map((component) => component * physical) as unknown as SvoNodeMipCoordinate,
      capacity: atlasPages[0] * atlasPages[1] * atlasPages[2],
    },
    requestedPageCount, residentPageCount,
    overflowPageCount: requestedPageCount - residentPageCount,
    pagePayloadBytes, atlasBytes, directoryBytes,
    allocatedBytes: atlasBytes + directoryBytes,
    complete: requestedPageCount === residentPageCount,
  };
}

function byte(value: number): number {
  if (!Number.isFinite(value)) throw new RangeError("SVO node-mip lane must be finite");
  return Math.max(0, Math.min(255, Math.round(value)));
}

/** Reduces eight RGBA8 children. Mean lanes are averaged; maximum lanes are conservative maxima. */
export function reduceSvoNodeMipChildren(children: readonly SvoNodeMipRgba8[]): SvoNodeMipRgba8 {
  if (children.length !== 8) throw new RangeError("SVO node-mip reduction requires eight children");
  let solidMean = 0, fluidMean = 0, solidMaximum = 0, fluidMaximum = 0;
  for (const child of children) {
    if (child.length !== 4) throw new RangeError("SVO node-mip child requires four lanes");
    const lanes = child.map(byte);
    solidMean += lanes[0]; fluidMean += lanes[2];
    solidMaximum = Math.max(solidMaximum, lanes[1]); fluidMaximum = Math.max(fluidMaximum, lanes[3]);
  }
  return [Math.round(solidMean / 8), solidMaximum, Math.round(fluidMean / 8), fluidMaximum];
}

export function svoNodeMipTexelOffset(x: number, y: number, z: number): number {
  const size = SVO_NODE_MIP_LAYOUT.physicalSize;
  if (![x, y, z].every((value) => Number.isInteger(value) && value >= 0 && value < size)) throw new RangeError("SVO node-mip physical texel is outside its page");
  return ((z * size + y) * size + x) * SVO_NODE_MIP_LAYOUT.channelCount;
}

export interface SvoNodeMipVirtualTexelAddress {
  page: SvoNodeMipCoordinate;
  texel: SvoNodeMipCoordinate;
}

/** Resolves an apron coordinate (-1 or 8) to the adjacent same-level virtual page. */
export function resolveSvoNodeMipVirtualTexel(
  page: SvoNodeMipCoordinate,
  texel: readonly [number, number, number],
): SvoNodeMipVirtualTexelAddress | undefined {
  coordinate(page, "SVO node-mip virtual page");
  if (texel.some((component) => !Number.isInteger(component))) throw new RangeError("SVO node-mip virtual texel must contain integers");
  const n = SVO_NODE_MIP_LAYOUT.interiorSize;
  const resultPage = [0, 0, 0];
  const resultTexel = [0, 0, 0];
  for (let axis = 0; axis < 3; axis += 1) {
    const pageOffset = Math.floor(texel[axis] / n);
    resultPage[axis] = page[axis] + pageOffset;
    if (resultPage[axis] < 0) return undefined;
    resultTexel[axis] = ((texel[axis] % n) + n) % n;
  }
  return { page: resultPage as unknown as SvoNodeMipCoordinate, texel: resultTexel as unknown as SvoNodeMipCoordinate };
}

export type SvoNodeMipApronSampler = (address: SvoNodeMipVirtualTexelAddress) => SvoNodeMipRgba8 | undefined;

/** Fills aprons from same-level neighbours and clamps only where a neighbour is not resident. */
export function createSvoNodeMipPageWithApron(
  pageCoordinate: SvoNodeMipCoordinate,
  interior: Uint8Array,
  sampleNeighbour: SvoNodeMipApronSampler,
): Uint8Array {
  const result = createSvoNodeMipPage(interior);
  const n = SVO_NODE_MIP_LAYOUT.interiorSize;
  const apron = SVO_NODE_MIP_LAYOUT.apron;
  const size = SVO_NODE_MIP_LAYOUT.physicalSize;
  // At `apron === 0` the page is all interior, every texel is skipped, and the
  // result is the plain interior copy — which is exactly what the live builder
  // has always produced, since its own apron was a clamped replica.
  for (let z = 0; z < size; z += 1) for (let y = 0; y < size; y += 1) for (let x = 0; x < size; x += 1) {
    if (x >= apron && x < n + apron && y >= apron && y < n + apron && z >= apron && z < n + apron) continue;
    const address = resolveSvoNodeMipVirtualTexel(pageCoordinate, [x - apron, y - apron, z - apron]);
    const sampled = address && sampleNeighbour(address);
    if (sampled) result.set(sampled.map(byte), svoNodeMipTexelOffset(x, y, z));
  }
  return result;
}

/** Creates a physical page, clamping the 8^3 interior into any apron the layout declares. */
export function createSvoNodeMipPage(interior: Uint8Array): Uint8Array {
  const n = SVO_NODE_MIP_LAYOUT.interiorSize;
  const apron = SVO_NODE_MIP_LAYOUT.apron;
  const size = SVO_NODE_MIP_LAYOUT.physicalSize;
  const channels = SVO_NODE_MIP_LAYOUT.channelCount;
  if (interior.byteLength !== n * n * n * channels) throw new RangeError("SVO node-mip interior must contain 8^3 RGBA8 texels");
  const result = new Uint8Array(SVO_NODE_MIP_LAYOUT.bytesPerPage);
  for (let z = 0; z < size; z += 1) for (let y = 0; y < size; y += 1) for (let x = 0; x < size; x += 1) {
    const sourceX = Math.max(0, Math.min(n - 1, x - apron));
    const sourceY = Math.max(0, Math.min(n - 1, y - apron));
    const sourceZ = Math.max(0, Math.min(n - 1, z - apron));
    const source = ((sourceZ * n + sourceY) * n + sourceX) * channels;
    result.set(interior.subarray(source, source + channels), svoNodeMipTexelOffset(x, y, z));
  }
  return result;
}

/** Publication is atomic: incomplete candidate state never replaces the last complete generation. */
export function publishSvoNodeMipGeneration(
  visible: SvoNodeMipPublication | undefined,
  candidate: SvoNodeMipPublicationCandidate,
): SvoNodeMipPublicationDecision {
  uint32(candidate.generation, "Candidate SVO node-mip generation");
  if (candidate.plan.generation !== candidate.generation || (visible && candidate.generation <= visible.completeGeneration)) {
    return { published: false, reason: "generation-order", visible };
  }
  if (!candidate.plan.complete) return { published: false, reason: "incomplete-plan", visible };
  if (!candidate.directoryComplete) return { published: false, reason: "incomplete-directory", visible };
  if (!candidate.payloadComplete) return { published: false, reason: "incomplete-payload", visible };
  if (!candidate.apronsComplete) return { published: false, reason: "incomplete-aprons", visible };
  return { published: true, reason: "published", visible: { completeGeneration: candidate.generation, plan: candidate.plan } };
}
