/**
 * Host-side census of a published owner-page arena.
 *
 * Diagnostic only -- no simulation branch reads any of this. It exists because
 * "did this scene coarsen?" cannot be answered by a leaf count: a domain can
 * carry the expected number of leaves and still be unit-fine along every wall,
 * which is exactly the failure the per-face boundary-strip histograms catch.
 * The per-Y profile answers the companion question of whether the deep
 * interior coarsened or only the air above the free surface.
 *
 * Leaves are deduplicated by `(origin, size)` identity because the arena is
 * probed once per finest cell, so every cell a leaf covers reports that leaf.
 */
import {
  lookupOctreeOwnerPage,
  OCTREE_OWNER_PAGE_LOOKUP_STATUS,
  type OctreeOwnerLeafSize,
  type OctreeOwnerPagePlan,
} from "./webgpu-octree-owner-pages";
import { OCTREE_POWER_BOUNDARY_STRIP_MIN_CELLS } from "./octree-pressure-capacity";

export interface OctreeTopologyLeafCensus {
  readonly generation: number;
  readonly residentOwnerPages: number;
  readonly topologyLeaves: number;
  readonly topologyNodes: number;
  readonly representedCells: number;
  readonly leafCountsBySize: Readonly<Record<string, number>>;
  /** Coarse leaf origins per finest-grid Y layer; diagnostic spatial profile. */
  readonly coarseLeafCountsByOriginY: readonly number[];
  /** Leaf-size histograms intersecting each three-cell world-boundary strip. */
  readonly boundaryStripLeafCountsBySize: Readonly<Record<
    "xLow" | "xHigh" | "yLow" | "yHigh" | "zLow" | "zHigh",
    Readonly<Record<string, number>>
  >>;
}

export function censusOctreeTopologyLeaves(
  ownerWords: ArrayLike<number>,
  plan: OctreeOwnerPagePlan,
  maximumLeafSize: OctreeOwnerLeafSize,
): OctreeTopologyLeafCensus {
  if (ownerWords.length < plan.allocatedWords) {
    throw new RangeError("Octree topology census owner arena is truncated");
  }
  const counts = new Map<number, number>();
  const coarseLeafCountsByOriginY = new Array<number>(plan.dimensions[1]).fill(0);
  const boundaryCounts = {
    xLow: new Map<number, number>(), xHigh: new Map<number, number>(),
    yLow: new Map<number, number>(), yHigh: new Map<number, number>(),
    zLow: new Map<number, number>(), zHigh: new Map<number, number>(),
  };
  const addBoundary = (face: keyof typeof boundaryCounts, size: number) => {
    const faceCounts = boundaryCounts[face];
    faceCounts.set(size, (faceCounts.get(size) ?? 0) + 1);
  };
  const identities = new Set<string>();
  const nodes = new Set<string>();
  let representedCells = 0;
  for (let z = 0; z < plan.dimensions[2]; z += 1) {
    for (let y = 0; y < plan.dimensions[1]; y += 1) {
      for (let x = 0; x < plan.dimensions[0]; x += 1) {
        const owner = lookupOctreeOwnerPage(
          ownerWords, plan, [x, y, z], maximumLeafSize,
        );
        if ((owner.status & (OCTREE_OWNER_PAGE_LOOKUP_STATUS.missing
          | OCTREE_OWNER_PAGE_LOOKUP_STATUS.invalid)) !== 0) continue;
        const identity = `${owner.origin[0]},${owner.origin[1]},${owner.origin[2]},${owner.size}`;
        if (identities.has(identity)) continue;
        identities.add(identity);
        for (let corner = 0; corner < 8; corner += 1) {
          nodes.add(`${owner.origin[0] + ((corner & 1) ? owner.size : 0)},`
            + `${owner.origin[1] + ((corner & 2) ? owner.size : 0)},`
            + `${owner.origin[2] + ((corner & 4) ? owner.size : 0)}`);
        }
        counts.set(owner.size, (counts.get(owner.size) ?? 0) + 1);
        if (owner.size > 1) coarseLeafCountsByOriginY[owner.origin[1]]! += 1;
        const high = owner.origin.map((coordinate) => coordinate + owner.size);
        const strip = OCTREE_POWER_BOUNDARY_STRIP_MIN_CELLS;
        if (owner.origin[0] < strip) addBoundary("xLow", owner.size);
        if (high[0]! > plan.dimensions[0] - strip) addBoundary("xHigh", owner.size);
        if (owner.origin[1] < strip) addBoundary("yLow", owner.size);
        if (high[1]! > plan.dimensions[1] - strip) addBoundary("yHigh", owner.size);
        if (owner.origin[2] < strip) addBoundary("zLow", owner.size);
        if (high[2]! > plan.dimensions[2] - strip) addBoundary("zHigh", owner.size);
        representedCells += owner.size ** 3;
      }
    }
  }
  const leafCountsBySize = Object.fromEntries(
    [...counts.entries()]
      .sort(([left], [right]) => left - right)
      .map(([size, count]) => [String(size), count]),
  );
  const boundaryStripLeafCountsBySize = Object.fromEntries(
    Object.entries(boundaryCounts).map(([face, faceCounts]) => [face,
      Object.freeze(Object.fromEntries([...faceCounts.entries()]
        .sort(([left], [right]) => left - right)
        .map(([size, count]) => [String(size), count]))),
    ]),
  ) as OctreeTopologyLeafCensus["boundaryStripLeafCountsBySize"];
  return Object.freeze({
    generation: Number(ownerWords[7] ?? 0) >>> 0,
    residentOwnerPages: Number(ownerWords[1] ?? 0) >>> 0,
    topologyLeaves: identities.size,
    topologyNodes: nodes.size,
    representedCells,
    leafCountsBySize: Object.freeze(leafCountsBySize),
    coarseLeafCountsByOriginY: Object.freeze(coarseLeafCountsByOriginY),
    boundaryStripLeafCountsBySize: Object.freeze(boundaryStripLeafCountsBySize),
  });
}
