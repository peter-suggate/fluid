import { SPARSE_BRICK_TOPOLOGY_MUTATION, sparseBrickTopologyMutationNodeReserve } from "../../../core/webgpu-sparse-brick-topology-mutation";
import { SVO_SURFACE_MESH_SCRATCH_WORDS, SVO_SURFACE_MESH_TABLE_WORDS } from "../primary-visibility/svo-surface-mesh";
import { SVO_BRICK_RASTER_CONTRACT, svoBrickRasterAddressableNodes } from "../primary-visibility/webgpu-svo-brick-raster";
import { SPARSE_BRICK_GPU_LAYOUT } from "./sparse-brick-octree";

/**
 * The most a display world's live reserve may cost: the leaves an edit session
 * can add to the tree, with the nodes, payload, maintenance slots and fitted
 * dual-grid samples each one needs, and the lighting pages those leaves bring
 * with them. The arenas are sized once, when the scene is built, and never
 * grow, so this is every finest brick a session can ever bring into being.
 *
 * Sized to hold the whole declared domain of the editor's default scene, so
 * that scene never refuses a brick. Built smooth, its 9,792-brick domain holds
 * 3,194 leaves and can gain 6,598, at 15,788 bytes a leaf (see
 * `liveTopologyReserveShares`) and 7,651 lighting pages at 36,864: 368.3 MiB.
 * Built as cubes it can charge 11,880 leaves at 7,598 bytes for the same
 * pages, 355.1 MiB. A larger scene reserves what this buys and no more, and
 * the lighting pages are three quarters of the price: 6,183 smooth bricks at
 * depth 6 and 6,101 at depth 9.
 */
export const OCTREE_LIVE_SCENE_TOPOLOGY_RESERVE_BUDGET_BYTES = 384 * 1024 * 1024;

/**
 * Leaves every edit a session can make inside the declared brick domain can
 * charge the reserve between them, whatever the scene was built with.
 *
 * A finest brick under no analytic terminal costs one leaf. One under a
 * terminal costs the splits down to it, seven preserved siblings a level, each
 * split charged once however many bricks share it. So a parent of finest
 * bricks is charged at most eight, and every node above it at most seven.
 */
export function liveTopologyDomainLeaves(brickDimensions: readonly [number, number, number],
  maximumDepth: number): number {
  const nodesAt = (level: number) => brickDimensions.reduce((product, bricks) =>
    product * Math.ceil(bricks / 2 ** (maximumDepth - level)), 1);
  if (maximumDepth === 0) return nodesAt(0);
  let leaves = 8 * nodesAt(maximumDepth - 1);
  for (let level = 0; level < maximumDepth - 1; level += 1) leaves += 7 * nodesAt(level);
  return leaves;
}

/** What a display world stores a leaf in, as far as the reserve's price goes. */
export interface LiveTopologyReserveLeafOptions {
  readonly maximumDepth: number;
  readonly payloadBytesPerLeaf: number;
  readonly candidatesPerBrick: number;
  /** Voxels of a leaf the dual grid keeps fitted samples for; none where the display draws cubes. */
  readonly dualGridVoxelsPerLeaf: number;
  /** Whether the scene was built with terminals an edit can split. */
  readonly splittableTerminals: boolean;
}

/**
 * Bytes one reserved leaf adds to each buffer sized by leaf capacity. The tree:
 * its leaf record, the nodes `sparseBrickTopologyMutationNodeReserve` sets
 * aside for it, and its payload. The maintenance arena: its dirty record, its
 * candidate slots and its claim word, and under a dual grid four words of
 * fitted sample a voxel. Then the residency's one word of leaf state, and the
 * mutation records `liveTopologyMutationRecords` sets aside for it. Last, what
 * the renderer keeps a leaf: the mesh's range table, and the brick raster's
 * candidate and published instances.
 */
export function liveTopologyReserveShares(options: LiveTopologyReserveLeafOptions) {
  const layout = SPARSE_BRICK_GPU_LAYOUT;
  return {
    leafRecord: layout.leafStrideBytes,
    nodes: 8 * options.maximumDepth * layout.nodeStrideBytes,
    payload: options.payloadBytesPerLeaf,
    maintenance: (4 + options.candidatesPerBrick + 1) * 4,
    dualGrid: options.dualGridVoxelsPerLeaf * 4 * 4,
    leafState: 4,
    mutationRequest: liveTopologyMutationRecordsPerLeaf(options.maximumDepth)
      * SPARSE_BRICK_TOPOLOGY_MUTATION.recordWords * 4,
    presentation: (SVO_SURFACE_MESH_TABLE_WORDS + SVO_SURFACE_MESH_SCRATCH_WORDS + 1) * 4
      + 2 * SVO_BRICK_RASTER_CONTRACT.instanceStrideBytes,
  } as const;
}

/**
 * Lighting pages a reserve of new finest pages is given, over the levels of
 * the node-mip pyramid that hold pages, finest first.
 *
 * A level cannot gain more than its grid has free, and what the planned leaves
 * hold of it is at least an eighth of the level below. Within that, built
 * surfaces are sheets, so each coarser level is given a quarter of the one
 * below: four thirds of a page a brick. Bricks scattered one to a parent need
 * more, and an edit that would run past these pages is refused when it is
 * validated, as one past the leaves is.
 */
export function liveTopologyDerivedReservePages(levelPages: readonly number[], occupiedBasePages: number,
  newBasePages: number): number {
  let pages = 0;
  for (let level = 0; level < levelPages.length; level += 1) {
    const free = Math.max(0, levelPages[level]! - Math.ceil(occupiedBasePages / 8 ** level));
    pages += Math.min(free, Math.ceil(newBasePages / 4 ** level));
  }
  return pages;
}

/**
 * Records one reserved leaf can put in a mutation batch. Activated, it writes
 * the nodes made on the way down to it, one a level, and the node they hang
 * from with the seven children that node can already have. Released in the
 * same batch as another is activated, it writes its own retirement and the
 * node left standing above it with that node's seven others. A split writes
 * nine records for seven leaves, which is fewer.
 */
function liveTopologyMutationRecordsPerLeaf(maximumDepth: number): number { return maximumDepth + 17; }

/**
 * Records the mutation worklist holds, so that one batch can activate every
 * reserved leaf and release as many: the host stages a whole edit as one
 * scatter, however many bricks it names. A terminal that is already a finest
 * brick is activated for one record and no leaf, and the root of a tree
 * planned empty is one more.
 */
export function liveTopologyMutationRecords(leaves: number, maximumDepth: number,
  finestSplittableTerminals: number): number {
  return leaves * liveTopologyMutationRecordsPerLeaf(maximumDepth) + finestSplittableTerminals + 16;
}

export interface LiveTopologyReservePlan {
  /** Leaves to reserve past the ones the scene is built with. */
  readonly leaves: number;
  /** What every in-domain edit could charge between them. */
  readonly domainLeaves: number;
  readonly bytesPerLeaf: number;
  /** Records one mutation batch may carry. */
  readonly mutationRecords: number;
  /**
   * Lighting pages to reserve with the leaves. A reserve that covers the
   * domain asks for the whole pyramid, which the address plan cuts to what is
   * unoccupied, so that every page of the domain has its slot.
   */
  readonly derivedPages: number;
  /** The reserve's price: the leaves, and the lighting pages priced for them. */
  readonly bytes: number;
}

/**
 * Leaves a display world reserves for live edits, decided before anything is
 * allocated: what its whole declared domain can charge, up to what the budget
 * buys of them and of the lighting pages they bring, and never fewer than the
 * floor.
 *
 * The device and the brick raster's node index bound it too. A reserve they
 * cannot hold is cut back to the largest they can, and no further than the
 * floor: a scene too large for even that is refused by the arenas' own checks,
 * by name, as it always was.
 */
export function planLiveTopologyLeafReserve(options: {
  /** Finest bricks of the declared scene domain, per axis. */
  readonly brickDimensions: readonly [number, number, number];
  readonly maximumDepth: number;
  /** The reserve of a scene the budget buys less for; the domain bounds it too. */
  readonly minimumLeaves: number;
  readonly plannedNodes: number;
  readonly plannedLeaves: number;
  /** Planned leaves that are finest bricks, which no edit activates again. */
  readonly plannedFinestLeaves: number;
  /** Planned terminals an edit can split, analytic or sampled, at any level. */
  readonly splittableTerminals: number;
  /** Those of them at the finest level, which an edit activates for no leaf. */
  readonly finestSplittableTerminals: number;
  /** Payload arena bytes of this world at a leaf capacity. */
  readonly payloadBytes: (leafCapacity: number) => number;
  /** The largest buffer, and binding, the device takes. */
  readonly storageLimitBytes: number;
  readonly candidatesPerBrick: number;
  /** Voxels of a leaf the dual grid keeps fitted samples for; none where the display draws cubes. */
  readonly dualGridVoxelsPerLeaf: number;
  /** Maintenance arena bytes that are no leaf's: at least the solid page image. */
  readonly maintenanceFixedBytes: number;
  /** The node-mip pyramid the leaves are lit from; absent where none is built. */
  readonly derived?: {
    /** Pages of the declared domain at each level that holds pages, finest first. */
    readonly levelPages: readonly number[];
    /** Finest of those pages the planned leaves hold, at least. */
    readonly occupiedBasePages: number;
    readonly bytesPerPage: number;
  };
  readonly budgetBytes?: number;
}): LiveTopologyReservePlan {
  const { maximumDepth, minimumLeaves, plannedNodes, plannedLeaves, derived } = options;
  for (const [name, value] of [["minimum", minimumLeaves], ["planned node count", plannedNodes],
    ["planned leaf count", plannedLeaves], ["planned finest leaf count", options.plannedFinestLeaves],
    ["splittable terminal count", options.splittableTerminals],
    ["finest splittable terminal count", options.finestSplittableTerminals]] as const) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(`Live topology reserve ${name} must be a non-negative safe integer`);
    }
  }
  const budgetBytes = options.budgetBytes ?? OCTREE_LIVE_SCENE_TOPOLOGY_RESERVE_BUDGET_BYTES;
  const built = Math.max(1, plannedLeaves);
  // The payload arena pads between lanes, so one leaf's share is read off a span of them.
  const span = 1024;
  const payloadBytesPerLeaf = (options.payloadBytes(built + span) - options.payloadBytes(built)) / span;
  const splittable = options.splittableTerminals > 0;
  const shares = liveTopologyReserveShares({ maximumDepth, payloadBytesPerLeaf,
    candidatesPerBrick: options.candidatesPerBrick, dualGridVoxelsPerLeaf: options.dualGridVoxelsPerLeaf,
    splittableTerminals: splittable });
  const bytesPerLeaf = Object.values(shares).reduce((sum, bytes) => sum + bytes, 0);
  const domainBricks = options.brickDimensions.reduce((product, bricks) => product * bricks, 1);
  // With nothing to split, a new finest brick is one leaf, and the domain has
  // only so many bricks that are not leaves already.
  const domainLeaves = splittable ? liveTopologyDomainLeaves(options.brickDimensions, maximumDepth)
    : Math.max(0, domainBricks - options.plannedFinestLeaves);
  // A finest terminal's page is lit already, so only the leaves' bricks are new pages.
  const derivedPages = (leaves: number) => derived ? liveTopologyDerivedReservePages(derived.levelPages,
    derived.occupiedBasePages, splittable ? leaves + Math.ceil(leaves / 7) : leaves) : 0;
  const price = (leaves: number) => leaves * bytesPerLeaf + derivedPages(leaves) * (derived?.bytesPerPage ?? 0);
  const layout = SPARSE_BRICK_GPU_LAYOUT;
  const holds = (leaves: number): boolean => {
    const nodeCapacity = plannedNodes + sparseBrickTopologyMutationNodeReserve(maximumDepth, leaves);
    const leafCapacity = Math.max(1, plannedLeaves + leaves);
    return nodeCapacity <= svoBrickRasterAddressableNodes()
      && options.payloadBytes(leafCapacity) <= options.storageLimitBytes
      // One alignment gap separates the node records from the leaf records.
      && layout.topologyOffsetBytes + nodeCapacity * layout.nodeStrideBytes + 256
        + leafCapacity * layout.leafStrideBytes <= options.storageLimitBytes
      && options.maintenanceFixedBytes + leafCapacity * (shares.maintenance + shares.dualGrid)
        <= options.storageLimitBytes;
  };
  let low = Math.min(minimumLeaves, domainLeaves), high = domainLeaves;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (price(middle) <= budgetBytes && holds(middle)) low = middle;
    else high = middle - 1;
  }
  return {
    leaves: low, domainLeaves, bytesPerLeaf,
    mutationRecords: liveTopologyMutationRecords(low, maximumDepth, options.finestSplittableTerminals),
    derivedPages: !derived ? 0 : low === domainLeaves
      ? derived.levelPages.reduce((sum, pages) => sum + pages, 0) : derivedPages(low),
    bytes: price(low),
  };
}
