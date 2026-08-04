import {
  SVO_NODE_MIP_LAYOUT,
  planSvoNodeMipPyramid,
  type SvoNodeMipCoordinate,
  type SvoNodeMipPyramidPlan,
} from "./svo-node-mip-pyramid";

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
  /** Opacity page + four rgba16float radiance lobes, in the target atlas. */
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
  /** Finest-level virtual pages the current topology actually covers. */
  occupiedBasePages: readonly SvoNodeMipCoordinate[];
  /** Finest-level page grid of the whole editable domain. */
  basePageDimensions: readonly [number, number, number];
  levelCount: number;
  /** Pages this device can address at all; the directory's area limit. */
  addressCapacity: number;
  /** Stable across re-plans: page-local validity is the live generation authority. */
  generation?: number;
  maximumReserveBytes?: number;
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
  /** `x,y,z` keys of every base page the plan currently addresses. */
  basePageKeys: ReadonlySet<string>;
  /** Every base page of the declared domain is addressed; growth can never be needed. */
  total: boolean;
  domainBasePageCount: number;
  domainPyramidPageCount: number;
  /** Slots held back for activations the first publication did not see. */
  reservePages: number;
  /** Worklist depth one level may need, which bounds the builder's scratch. */
  pageCapacityPerLevel: number;
  maximumReserveBytes: number;
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
 */
export function svoNodeMipDomainDirectPageTableDimensions(
  basePageDimensions: readonly [number, number, number],
  levelCount: number,
): readonly [number, number, number] {
  const levels = svoNodeMipLevelPageDimensions(basePageDimensions, levelCount);
  return [
    levels[0][0],
    levels[0][1],
    levels.reduce((depth, dimensions) => depth + dimensions[2], 0),
  ];
}

function levelPageCounts(plan: SvoNodeMipPyramidPlan, levelCount: number): number[] {
  const counts = new Array<number>(levelCount).fill(0);
  for (const page of plan.pages) if (page.key.level < levelCount) counts[page.key.level] += 1;
  return counts;
}

function buildPlan(
  basePages: readonly SvoNodeMipCoordinate[],
  levelCount: number,
  pageCapacity: number,
  atlasPages: readonly [number, number, number],
  generation: number,
): SvoNodeMipPyramidPlan {
  return planSvoNodeMipPyramid({
    generation,
    occupiedPages: basePages,
    levelCount,
    capacity: pageCapacity,
    atlasPages: atlasPages as SvoNodeMipCoordinate,
  });
}

/**
 * Decide the address plan once, for the lifetime of the world.
 *
 * Throws nothing a caller has to interpret: an unusable domain comes back as a
 * plan whose `complete` is false, which is the same signal the caller already
 * treats as "derived lighting is unavailable".
 */
export function planSvoNodeMipAddresses(options: SvoNodeMipAddressPlanOptions): SvoNodeMipAddressPlan {
  positiveDimensions(options.basePageDimensions);
  if (!Number.isSafeInteger(options.addressCapacity) || options.addressCapacity <= 0) {
    throw new RangeError("Node-mip address capacity must be a positive safe integer");
  }
  const generation = options.generation ?? 1;
  const levelCount = options.levelCount;
  const levelDimensions = svoNodeMipLevelPageDimensions(options.basePageDimensions, levelCount);
  const domainBasePageCount = options.basePageDimensions.reduce((product, value) => product * value, 1);
  const domainPyramidPageCount = svoNodeMipDomainPyramidPageCount(options.basePageDimensions, levelCount);
  const maximumReserveBytes = options.maximumReserveBytes ?? SVO_NODE_MIP_ADDRESS_PLAN.maximumReserveBytes;
  const reserveBudgetPages = Math.max(0, Math.floor(maximumReserveBytes / SVO_NODE_MIP_ADDRESS_PLAN.reservedBytesPerPage));

  // What the old behaviour would have planned, and the floor for everything below.
  const occupiedProbe = buildPlan(options.occupiedBasePages, levelCount, options.addressCapacity,
    svoNodeMipAtlasPagesForCapacity(Math.max(1, Math.min(options.addressCapacity, domainPyramidPageCount))), generation);
  const occupiedPageCount = occupiedProbe.requestedPageCount;

  const reservePages = Math.min(reserveBudgetPages, Math.max(0, domainPyramidPageCount - occupiedPageCount));
  const pageCapacity = Math.max(1, Math.min(options.addressCapacity, occupiedPageCount + reservePages));
  const total = pageCapacity >= domainPyramidPageCount;
  const atlasPages = svoNodeMipAtlasPagesForCapacity(pageCapacity);
  const basePages = total ? svoNodeMipDomainBasePages(options.basePageDimensions) : options.occupiedBasePages;
  const plan = buildPlan(basePages, levelCount, pageCapacity, atlasPages, generation);
  const denseLevelMaximum = levelDimensions.reduce(
    (maximum, dimensions) => Math.max(maximum, dimensions[0] * dimensions[1] * dimensions[2]), 1);
  const planLevelMaximum = Math.max(1, ...levelPageCounts(plan, levelCount));
  const pageCapacityPerLevel = Math.max(1, Math.min(denseLevelMaximum,
    total ? denseLevelMaximum : planLevelMaximum + reservePages));
  return {
    plan,
    atlasPages,
    atlasTexels: atlasPages.map((value) => value * SVO_NODE_MIP_LAYOUT.physicalSize) as [number, number, number],
    pageCapacity,
    levelCount,
    basePageDimensions: options.basePageDimensions,
    basePageKeys: new Set(basePages.map(svoNodeMipBasePageKey)),
    total,
    domainBasePageCount,
    domainPyramidPageCount,
    reservePages,
    pageCapacityPerLevel,
    maximumReserveBytes,
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
  for (const page of requestedBasePages) {
    const key = svoNodeMipBasePageKey(page);
    if (keys.has(key)) continue;
    if (page.some((component, axis) => !Number.isSafeInteger(component) || component < 0
      || component >= current.basePageDimensions[axis])) {
      // Outside the declared domain: no plan of any size addresses it, and
      // silently dropping it would leak light exactly where geometry appeared.
      return undefined;
    }
    keys.add(key);
    added.push(page);
  }
  if (added.length === 0) return { plan: current, addedBasePages: [], rebuildRequired: false };
  const basePages = [...keys].map((key) => key.split(",").map(Number) as unknown as SvoNodeMipCoordinate);
  const plan = buildPlan(basePages, current.levelCount, current.pageCapacity, current.atlasPages, current.plan.generation);
  if (!plan.complete) return undefined;
  const levelMaximum = Math.max(1, ...levelPageCounts(plan, current.levelCount));
  if (levelMaximum > current.pageCapacityPerLevel) return undefined;
  return {
    plan: {
      ...current,
      plan,
      basePageKeys: keys,
      total: keys.size >= current.domainBasePageCount,
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
  for (const page of requestedBasePages) {
    const key = svoNodeMipBasePageKey(page);
    if (plan.basePageKeys.has(key) || seen.has(key)) continue;
    seen.add(key);
    result.push(page);
  }
  return result;
}
