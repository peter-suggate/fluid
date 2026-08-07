import {
  SVO_NODE_MIP_LAYOUT,
  planSvoNodeMipPyramidSteps,
  raiseSvoNodeMipSeedToFloor,
  svoNodeMipRadianceSlotOffset,
  svoNodeMipSeedKey,
  svoNodeMipSeedPage,
  svoRadianceLevelFloor,
  type SvoNodeMipCoordinate,
  type SvoNodeMipPageSeed,
  type SvoNodeMipPyramidPlan,
  type SvoNodeMipSeedPage,
} from "./svo-node-mip-pyramid";
import { completeCooperativeBuild } from "./cooperative-build";

/**
 * Where the live node-mip pyramid's *addresses* are decided, as opposed to its
 * contents.
 *
 * The pyramid used to fix its address plan at the first publication and treat
 * every later activation outside it as fatal: `liveDerivedAddressPlanValid`
 * went false, the opacity pyramid and the radiance atlas were nulled, and cone
 * visibility fell back to exact traversal — a ~15x frame cost whose only report
 * was a status string nobody reads. That was survivable while the only thing
 * that activated a page was the initial publication. Incremental voxelization
 * activates pages continuously, so it trips that path by construction.
 *
 * This module makes the plan cover the *domain* rather than the first frame's
 * occupancy. Two shapes, in preference order:
 *
 *  1. **Total.** Every base page the declared domain can hold gets a slot up
 *     front. No edit can activate a page outside the plan, because there is no
 *     outside. This is what the hero garden takes: 384 base pages, 441 pyramid
 *     pages against 245 occupied ones — 196 extra slots, ~7 MB.
 *  2. **Reserved.** The domain is too large to address whole, so the plan
 *     covers what is occupied plus a bounded reserve, and grows into that
 *     reserve when an edit needs it. Growth re-plans in the same Morton order
 *     the directory's binary-search fallback requires, so page slots renumber
 *     and the caller must rebuild every page — which is why the reserve exists
 *     at all rather than re-planning per activation.
 *
 * The reserve is priced in bytes, not in a fraction, because that is what it
 * actually costs: a slot is one opacity page plus four radiance lobes in the
 * atlas, and — because a full rebuild puts a whole level through the builder's
 * scratch at once — one more of each in scratch.
 */
export const SVO_NODE_MIP_ADDRESS_PLAN = Object.freeze({
  /**
   * Opacity page + four radiance lobes, in the target atlas.
   *
   * Priced at 8 B/texel — the `rgba16float` fallback — deliberately, even though
   * a device with `texture-formats-tier1` allocates the lobes as
   * `rg11b10ufloat` at 4. The reserve is a *byte* budget, so a cheaper page
   * would buy proportionally more reserved slots and move the atlas shape with
   * the device's feature set. Holding the conservative figure keeps the address
   * plan a pure function of the scene: the narrow format then simply spends
   * less than its budget, which is never a fault.
   */
  atlasBytesPerPage: SVO_NODE_MIP_LAYOUT.bytesPerPage + 4 * SVO_NODE_MIP_LAYOUT.physicalSize ** 3 * 8,
  /**
   * What one reserved slot costs in total. Twice the atlas figure: the builder's
   * compact scratch is sized by the largest per-level worklist, and raising the
   * addressable page count raises that worklist by the same amount.
   */
  reservedBytesPerPage: 2 * (SVO_NODE_MIP_LAYOUT.bytesPerPage + 4 * SVO_NODE_MIP_LAYOUT.physicalSize ** 3 * 8),
  /**
   * Physical allocation this module is willing to spend on addresses that hold
   * nothing yet. Sized so an ordinary authored room plans totally — the hero
   * garden needs 196 of these — while a solver-sized domain still gets a usable
   * growth window instead of the whole lattice.
   */
  maximumReserveBytes: 48 * 1024 * 1024,
} as const);

export interface SvoNodeMipAddressPlanOptions {
  /**
   * Virtual pages the current topology covers, each at its own level.
   *
   * A bare coordinate still means a finest-level page, so every producer that
   * only ever had base pages reads unchanged. A coarse leaf seeds the level
   * whose texels are its voxels — see {@link SvoNodeMipSeedPage}.
   */
  occupiedBasePages: readonly SvoNodeMipPageSeed[];
  /** Finest-level page grid of the whole editable domain. */
  basePageDimensions: readonly [number, number, number];
  levelCount: number;
  /** Pages this device can address at all; the directory's area limit. */
  addressCapacity: number;
  /** Stable across re-plans: page-local validity is the live generation authority. */
  generation?: number;
  maximumReserveBytes?: number;
  /** Finest leaf edge in metres; anchors the radiance floor to a world size. */
  cellSize_m?: number;
  /** Explicit radiance floor level. Zero restores a radiance page at every level. */
  radianceFloorLevel?: number;
  /**
   * Finest level that owns an *opacity* page; see `SVO_OPACITY_LEVEL_FLOOR`.
   *
   * The plan carries it rather than deriving it so that growth stays inside the
   * shape the first publication chose: an edit hands this module bare level-0
   * coordinates, and re-seeding one of those under a floored plan would put a
   * base level back under part of the scene and nowhere else. Zero is the
   * shipped behaviour and every path below is then the identity.
   */
  opacityFloorLevel?: number;
}

export interface SvoNodeMipAddressPlan {
  /** The plan the pyramid, the radiance atlas and the planner are configured from. */
  plan: SvoNodeMipPyramidPlan;
  /** Fixed for the lifetime of the world, so no re-plan reallocates an atlas. */
  atlasPages: readonly [number, number, number];
  atlasTexels: readonly [number, number, number];
  pageCapacity: number;
  levelCount: number;
  basePageDimensions: readonly [number, number, number];
  /** `level:x,y,z` keys of every page the plan is seeded from. */
  basePageKeys: ReadonlySet<string>;
  /** The seeds behind those keys, so a growth can re-plan without losing levels. */
  seedPages: readonly SvoNodeMipSeedPage[];
  /** Every base page of the declared domain is addressed; growth can never be needed. */
  total: boolean;
  domainBasePageCount: number;
  domainPyramidPageCount: number;
  /** Slots held back for activations the first publication did not see. */
  reservePages: number;
  /** Worklist depth the deepest level may need, which bounds the builder's scratch. */
  pageCapacityPerLevel: number;
  /**
   * Worklist depth *each* level may need, which sizes the arena's sections.
   *
   * The scalar above is a maximum over levels, and the derived worklist arena
   * used to lay every section out at it — so the coarsest level, which holds one
   * page, got the base level's capacity at 48 B a record. A level's own bound is
   * its own domain grid (`levelDimensions[level]`, which the planner's direct
   * page table cannot address past) capped by what the reserve could still add
   * to it, and both of those are exact rather than conservative: the plan's
   * total headroom is `reservePages`, so a level holding `p` pages today can
   * hold at most `p + reservePages` after any growth this plan admits.
   */
  pageCapacityByLevel: readonly number[];
  maximumReserveBytes: number;
  /**
   * Finest level that owns a radiance page. See `SVO_RADIANCE_LEVEL_FLOOR`.
   *
   * Everything below is opacity-only, and the fields beneath describe a radiance
   * atlas that is *not* the opacity atlas: fewer slots, its own shape, and its
   * own origin in the shared slot numbering.
   */
  radianceFloorLevel: number;
  /** Slots the plan gives to levels below the floor; the radiance atlas starts here. */
  radianceSlotOffset: number;
  /** Physical radiance slots, sized for the domain rather than for today's plan. */
  radiancePageCapacity: number;
  radianceAtlasPages: readonly [number, number, number];
  radianceAtlasTexels: readonly [number, number, number];
  /** Radiance worklist depth one level may need; bounds the radiance scratch. */
  radiancePageCapacityPerLevel: number;
  /** Finest level that owns an opacity page. Zero is the shipped pyramid. */
  opacityFloorLevel: number;
}

function positiveDimensions(value: readonly [number, number, number]): void {
  if (value.length !== 3 || value.some((component) => !Number.isSafeInteger(component) || component <= 0)) {
    throw new RangeError("Node-mip base page dimensions must be three positive safe integers");
  }
}

export function svoNodeMipBasePageKey(page: SvoNodeMipCoordinate): string {
  return `${page[0]},${page[1]},${page[2]}`;
}

/** Page grid at each virtual level, given the finest one. Mirrors the plan's `floor(c/2)` ancestry. */
export function svoNodeMipLevelPageDimensions(
  basePageDimensions: readonly [number, number, number],
  levelCount: number,
): readonly (readonly [number, number, number])[] {
  positiveDimensions(basePageDimensions);
  if (!Number.isSafeInteger(levelCount) || levelCount < 1 || levelCount > 32) {
    throw new RangeError("Node-mip level count must be an integer in [1, 32]");
  }
  const levels: (readonly [number, number, number])[] = [];
  let current = [...basePageDimensions] as [number, number, number];
  for (let level = 0; level < levelCount; level += 1) {
    levels.push([...current] as [number, number, number]);
    current = current.map((component) => Math.max(1, Math.ceil(component / 2))) as [number, number, number];
  }
  return levels;
}

/**
 * Pages a plan over *every* base page of the domain would request, without
 * materialising them. Deciding whether the total shape is affordable must not
 * itself cost the allocation being decided against.
 */
export function svoNodeMipDomainPyramidPageCount(
  basePageDimensions: readonly [number, number, number],
  levelCount: number,
): number {
  return svoNodeMipLevelPageDimensions(basePageDimensions, levelCount)
    .reduce((total, dimensions) => total + dimensions[0] * dimensions[1] * dimensions[2], 0);
}

/** Every base page of the domain, in the order the plan will sort anyway. */
export function svoNodeMipDomainBasePages(
  basePageDimensions: readonly [number, number, number],
): readonly SvoNodeMipCoordinate[] {
  positiveDimensions(basePageDimensions);
  const [width, height, depth] = basePageDimensions;
  const pages: SvoNodeMipCoordinate[] = [];
  for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) pages.push([x, y, z]);
  return pages;
}

/**
 * Atlas page shape for a fixed slot capacity.
 *
 * Deliberately the same near-cubic rule `planSvoNodeMipPyramid` applies when no
 * shape is supplied, so a world that never grows allocates exactly what it
 * always did. It is restated here rather than imported because the point of
 * passing the shape explicitly is that it stops depending on the resident page
 * count — the atlas has to outlive every re-plan.
 */
export function svoNodeMipAtlasPagesForCapacity(capacity: number): readonly [number, number, number] {
  if (!Number.isSafeInteger(capacity) || capacity <= 0) throw new RangeError("Node-mip atlas capacity must be positive");
  const x = Math.ceil(Math.cbrt(capacity));
  const y = Math.ceil(Math.sqrt(capacity / x));
  return [x, y, Math.ceil(capacity / (x * y))];
}

/**
 * Direct page-table volume that addresses every page of the domain.
 *
 * The table is one Z slab per virtual level, so its extent is the domain's page
 * grid in X and Y and the sum of the per-level depths in Z. Allocating it at
 * domain size rather than at the first plan's size is what lets a re-plan
 * publish new coordinates without reallocating a texture the planner and the
 * builder already hold views of. A plan smaller than the allocation writes into
 * the same texture with its own level offsets, so an over-sized table is always
 * safe.
 *
 * `floorLevel` drops the slabs the opacity floor removed. It is worth doing
 * rather than leaving as harmless slack: the table is *dense* over the page
 * grid while the pages in it are sparse, so the base level alone is 95.3 MB of
 * the hero garden's 95.5 MB table at refinement depth 3, against 12.0 MB for
 * every level above it. The shader indexes its slabs by absolute level and
 * reads each level's depth as the gap between consecutive offsets, so a level
 * below the floor is simply a zero-depth slab and every lookup into it answers
 * "not resident" — which is exactly what a floored plan means.
 */
export function svoNodeMipDomainDirectPageTableDimensions(
  basePageDimensions: readonly [number, number, number],
  levelCount: number,
  floorLevel = 0,
): readonly [number, number, number] {
  if (!Number.isSafeInteger(floorLevel) || floorLevel < 0) throw new RangeError("Node-mip opacity floor level must be a non-negative safe integer");
  const levels = svoNodeMipLevelPageDimensions(basePageDimensions, levelCount);
  const base = levels[Math.min(floorLevel, levels.length - 1)];
  return [
    base[0],
    base[1],
    levels.reduce((depth, dimensions, level) => depth + (level < floorLevel ? 0 : dimensions[2]), 0),
  ];
}

function levelPageCounts(plan: SvoNodeMipPyramidPlan, levelCount: number): number[] {
  const counts = new Array<number>(levelCount).fill(0);
  for (const page of plan.pages) if (page.key.level < levelCount) counts[page.key.level] += 1;
  return counts;
}

function* buildPlanSteps(
  basePages: readonly SvoNodeMipPageSeed[],
  levelCount: number,
  pageCapacity: number,
  atlasPages: readonly [number, number, number],
  generation: number,
): Generator<unknown, SvoNodeMipPyramidPlan, undefined> {
  return yield* planSvoNodeMipPyramidSteps({
    generation,
    occupiedPages: basePages,
    levelCount,
    capacity: pageCapacity,
    atlasPages: atlasPages as SvoNodeMipCoordinate,
  });
}

function buildPlan(
  basePages: readonly SvoNodeMipPageSeed[],
  levelCount: number,
  pageCapacity: number,
  atlasPages: readonly [number, number, number],
  generation: number,
): SvoNodeMipPyramidPlan {
  return completeCooperativeBuild(buildPlanSteps(basePages, levelCount, pageCapacity, atlasPages, generation));
}

/** Deduplicated seeds in their canonical `{ level, coordinate }` shape, at or above the floor. */
function canonicalSeeds(seeds: readonly SvoNodeMipPageSeed[], floorLevel = 0): SvoNodeMipSeedPage[] {
  const unique = new Map<string, SvoNodeMipSeedPage>();
  for (const seed of seeds) {
    const page = floorLevel > 0 ? raiseSvoNodeMipSeedToFloor(seed, floorLevel) : svoNodeMipSeedPage(seed);
    unique.set(svoNodeMipSeedKey(page), page);
  }
  return [...unique.values()];
}

/** Every page of the domain at the floor level, in the order the plan sorts anyway. */
function domainSeedPages(
  basePageDimensions: readonly [number, number, number],
  levelDimensions: readonly (readonly [number, number, number])[],
  floorLevel: number,
): readonly SvoNodeMipPageSeed[] {
  if (floorLevel === 0) return svoNodeMipDomainBasePages(basePageDimensions);
  const [width, height, depth] = levelDimensions[Math.min(floorLevel, levelDimensions.length - 1)];
  const pages: SvoNodeMipSeedPage[] = [];
  for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    pages.push({ level: floorLevel, coordinate: [x, y, z] });
  }
  return pages;
}

/**
 * Decide the address plan once, for the lifetime of the world.
 *
 * Throws nothing a caller has to interpret: an unusable domain comes back as a
 * plan whose `complete` is false, which is the same signal the caller already
 * treats as "derived lighting is unavailable".
 */
export function planSvoNodeMipAddresses(options: SvoNodeMipAddressPlanOptions): SvoNodeMipAddressPlan {
  return completeCooperativeBuild(planSvoNodeMipAddressesSteps(options));
}

/**
 * The same decision, offered as slices.
 *
 * Two pyramid plans — the occupancy probe and the real one — and after the
 * opacity floor landed they are together the longest block a refined build has
 * left: 2955 ms at 112k occupied base pages, 567 ms at 24k. Every yield is in
 * the derived-lighting *prologue*, which owns no device resource: the first one
 * that block creates is `new WebGpuLiveSvoNodeMipPyramid`, and there is no
 * yield at or after it.
 */
export function* planSvoNodeMipAddressesSteps(
  options: SvoNodeMipAddressPlanOptions,
): Generator<unknown, SvoNodeMipAddressPlan, undefined> {
  positiveDimensions(options.basePageDimensions);
  if (!Number.isSafeInteger(options.addressCapacity) || options.addressCapacity <= 0) {
    throw new RangeError("Node-mip address capacity must be a positive safe integer");
  }
  const generation = options.generation ?? 1;
  const levelCount = options.levelCount;
  const levelDimensions = svoNodeMipLevelPageDimensions(options.basePageDimensions, levelCount);
  const opacityFloorLevel = Math.max(0, Math.min(levelCount - 1, Math.trunc(options.opacityFloorLevel ?? 0)));
  // Every count below is over the levels that can hold a page. Below the floor
  // there are none, so a floored domain is the floor level's grid and the sum
  // above it — not the base grid, which no longer describes anything resident.
  const domainBasePageCount = levelDimensions[opacityFloorLevel].reduce((product, value) => product * value, 1);
  const domainPyramidPageCount = levelDimensions.reduce(
    (count, dimensions, level) => count + (level < opacityFloorLevel ? 0 : dimensions[0] * dimensions[1] * dimensions[2]), 0);
  const maximumReserveBytes = options.maximumReserveBytes ?? SVO_NODE_MIP_ADDRESS_PLAN.maximumReserveBytes;
  const reserveBudgetPages = Math.max(0, Math.floor(maximumReserveBytes / SVO_NODE_MIP_ADDRESS_PLAN.reservedBytesPerPage));

  // What the old behaviour would have planned, and the floor for everything below.
  const occupiedProbe = yield* buildPlanSteps(canonicalSeeds(options.occupiedBasePages, opacityFloorLevel),
    levelCount, options.addressCapacity,
    svoNodeMipAtlasPagesForCapacity(Math.max(1, Math.min(options.addressCapacity, domainPyramidPageCount))), generation);
  const occupiedPageCount = occupiedProbe.requestedPageCount;

  const reservePages = Math.min(reserveBudgetPages, Math.max(0, domainPyramidPageCount - occupiedPageCount));
  const pageCapacity = Math.max(1, Math.min(options.addressCapacity, occupiedPageCount + reservePages));
  const total = pageCapacity >= domainPyramidPageCount;
  const basePages = canonicalSeeds(
    total ? domainSeedPages(options.basePageDimensions, levelDimensions, opacityFloorLevel) : options.occupiedBasePages,
    opacityFloorLevel);
  const atlasPages = svoNodeMipAtlasPagesForCapacity(pageCapacity);
  const plan = yield* buildPlanSteps(basePages, levelCount, pageCapacity, atlasPages, generation);
  // Pages a level's own grid can hold. Zero below the floor, where no page
  // exists; the direct page table rejects every coordinate outside it, so this
  // is a hard ceiling on that level and not an estimate.
  const denseLevelCounts = levelDimensions.map((dimensions, level) => (level < opacityFloorLevel
    ? 0 : dimensions[0] * dimensions[1] * dimensions[2]));
  const denseLevelMaximum = Math.max(1, ...denseLevelCounts);
  const planLevelCounts = levelPageCounts(plan, levelCount);
  const planLevelMaximum = Math.max(1, ...planLevelCounts);
  // Same policy per level as the scalar's, with each level's own two terms: a
  // total plan is already the whole grid, and a partial one can grow by at most
  // the shared `reservePages` — all of which could land on any single level.
  const pageCapacityByLevel = denseLevelCounts.map((dense, level) => Math.max(1, Math.min(dense,
    total ? dense : planLevelCounts[level] + reservePages)));
  const pageCapacityPerLevel = Math.max(1, Math.min(denseLevelMaximum,
    total ? denseLevelMaximum : planLevelMaximum + reservePages));
  // The radiance atlas is sized from the *domain* above the floor, never from
  // today's plan: a later growth can add pages at any level, and the atlas is
  // fixed for the lifetime of the world. A level can never hold more pages than
  // the domain has at that level, because the planner's direct page table
  // rejects coordinates outside the domain grid, so the domain sum is a bound.
  // Never below the opacity floor: the radiance atlas rides the opacity
  // pyramid's slot numbering, so a radiance base at a level that owns no page
  // would ask the builder to seed a chain from records that do not exist. The
  // anchored radiance floor is three levels up at the reference leaf and rises
  // with refinement, so this clamp only ever binds when both levers are forced.
  const radianceFloorLevel = Math.max(opacityFloorLevel, svoRadianceLevelFloor({
    levelCount, cellSize_m: options.cellSize_m, override: options.radianceFloorLevel,
  }));
  const radianceLevels = levelDimensions.slice(radianceFloorLevel)
    .map((dimensions) => dimensions[0] * dimensions[1] * dimensions[2]);
  const radiancePageCapacity = Math.max(1, Math.min(pageCapacity,
    radianceLevels.reduce((total_, count) => total_ + count, 0)));
  const radianceAtlasPages = svoNodeMipAtlasPagesForCapacity(radiancePageCapacity);
  const radiancePageCapacityPerLevel = Math.max(1, Math.min(pageCapacityPerLevel, Math.max(1, ...radianceLevels)));
  return {
    plan,
    atlasPages,
    atlasTexels: atlasPages.map((value) => value * SVO_NODE_MIP_LAYOUT.physicalSize) as [number, number, number],
    pageCapacity,
    levelCount,
    basePageDimensions: options.basePageDimensions,
    basePageKeys: new Set(basePages.map(svoNodeMipSeedKey)),
    seedPages: basePages,
    total,
    domainBasePageCount,
    domainPyramidPageCount,
    reservePages,
    pageCapacityPerLevel,
    pageCapacityByLevel,
    maximumReserveBytes,
    radianceFloorLevel,
    radianceSlotOffset: svoNodeMipRadianceSlotOffset(plan, radianceFloorLevel),
    radiancePageCapacity,
    radianceAtlasPages,
    radianceAtlasTexels: radianceAtlasPages.map((value) => value * SVO_NODE_MIP_LAYOUT.physicalSize) as [number, number, number],
    radiancePageCapacityPerLevel,
    opacityFloorLevel,
  };
}

export interface SvoNodeMipAddressPlanGrowth {
  plan: SvoNodeMipAddressPlan;
  /** Base pages this growth added. Empty means the request was already covered. */
  addedBasePages: readonly SvoNodeMipCoordinate[];
  /**
   * Every physical slot was reassigned, so the caller owes a full rebuild.
   *
   * Always true today. Slots are handed out in level-major Morton order because
   * the sampled directory is binary-searched in exactly that order whenever the
   * direct page table is unavailable, and there is no way to append a page in
   * the middle of that order without moving the pages after it.
   */
  rebuildRequired: boolean;
}

/**
 * Extend an address plan to cover pages an edit activated.
 *
 * Returns `undefined` when the reserve cannot hold them — the one case where
 * the caller must still withdraw. A total plan never reaches that branch,
 * because `pagesOutsideSvoNodeMipAddressPlan` is empty for it by construction.
 */
export function growSvoNodeMipAddressPlan(
  current: SvoNodeMipAddressPlan,
  requestedBasePages: readonly SvoNodeMipCoordinate[],
): SvoNodeMipAddressPlanGrowth | undefined {
  const added: SvoNodeMipCoordinate[] = [];
  const keys = new Set(current.basePageKeys);
  const seeds = [...current.seedPages];
  const floorLevel = current.opacityFloorLevel ?? 0;
  for (const page of requestedBasePages) {
    // Domain membership is asked of the *finest* coordinate the caller named,
    // before the floor collapses it: a page one step outside the base grid maps
    // inside the floor's grid, and accepting it would address geometry the
    // planner's direct table cannot reach.
    if (page.some((component, axis) => !Number.isSafeInteger(component) || component < 0
      || component >= current.basePageDimensions[axis])) {
      // Outside the declared domain: no plan of any size addresses it, and
      // silently dropping it would leak light exactly where geometry appeared.
      return undefined;
    }
    const seed = floorLevel > 0 ? raiseSvoNodeMipSeedToFloor(page, floorLevel) : { level: 0, coordinate: page };
    const key = svoNodeMipSeedKey(seed);
    if (keys.has(key)) continue;
    keys.add(key);
    seeds.push(seed);
    added.push(page);
  }
  if (added.length === 0) return { plan: current, addedBasePages: [], rebuildRequired: false };
  const plan = buildPlan(seeds, current.levelCount, current.pageCapacity, current.atlasPages, current.plan.generation);
  if (!plan.complete) return undefined;
  // Per level, against the section that level was actually allocated. The
  // arena's sections are sized independently now, so a growth that fits the
  // deepest level says nothing about whether it fits the one it landed on.
  const grownLevelCounts = levelPageCounts(plan, current.levelCount);
  if (grownLevelCounts.some((count, level) => count > current.pageCapacityByLevel[level])) return undefined;
  return {
    plan: {
      ...current,
      plan,
      basePageKeys: keys,
      seedPages: seeds,
      total: keys.size >= current.domainBasePageCount,
      // Renumbered slots move where the radiance atlas starts. The atlas shape
      // itself is domain-sized and therefore survives every growth untouched.
      radianceSlotOffset: svoNodeMipRadianceSlotOffset(plan, current.radianceFloorLevel),
    },
    addedBasePages: added,
    rebuildRequired: true,
  };
}

/** The base pages a request names that the plan does not already address. */
export function pagesOutsideSvoNodeMipAddressPlan(
  plan: SvoNodeMipAddressPlan,
  requestedBasePages: readonly SvoNodeMipCoordinate[],
): readonly SvoNodeMipCoordinate[] {
  const seen = new Set<string>();
  const result: SvoNodeMipCoordinate[] = [];
  const floorLevel = plan.opacityFloorLevel ?? 0;
  for (const page of requestedBasePages) {
    // Membership is asked at the level the plan stores, so eight base pages
    // under one floored page answer "already addressed" — which they are.
    const key = svoNodeMipSeedKey(floorLevel > 0 ? raiseSvoNodeMipSeedToFloor(page, floorLevel) : page);
    if (plan.basePageKeys.has(key) || seen.has(key)) continue;
    seen.add(key);
    result.push(page);
  }
  return result;
}
