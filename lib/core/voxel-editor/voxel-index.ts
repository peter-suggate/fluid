import type { SceneDescription } from "../model";
import { composeSolidPatch, validateSolidPatch, type SolidPatchPageStore } from "../solid-patch/compose";
import { solidFractionOccupies } from "../solid-patch/kind";
import { solidPatchListExtends } from "../solid-patch-list";
import { SOLID_WORLD_BRICK_CELLS, SOLID_WORLD_VOXELS_PER_PAGE, solidGroundForScene, type SolidGround,
  type SolidWorldCoordinate, type SolidWorldPage, type SolidWorldVoxelPatch } from "../solid-world";
import { EDITOR_HISTORY_LIMIT } from "../stores/history-store";

const B = SOLID_WORLD_BRICK_CELLS;
/** Pages along the edge of a node, so a node is 64 cells across. */
const NODE_PAGES = 8;
/** Nodes either side of the origin whose coordinates pack exactly into one number. */
const NEAR_NODES = 1 << 16;

/** A page of the index, tagged with the build that may still write to it. */
export interface VoxelIndexPage extends SolidWorldPage {
  readonly epoch: number;
}

/** 8 x 8 x 8 pages, x fastest. A hole is a page no patch has touched. */
export interface VoxelIndexNode {
  readonly epoch: number;
  readonly pages: (VoxelIndexPage | undefined)[];
}

/**
 * The page that stands for "authored, and nothing left".
 *
 * Over terrain an emptied page still has to be remembered — the ground under it
 * is analytic and would otherwise show through a hole someone dug — but its
 * cells are all the same and not worth 2.5 kB each time. One shared page of
 * empty cells says so. It is never handed to the compositor and never written.
 */
export const VOXEL_INDEX_CLEARED_PAGE: VoxelIndexPage = Object.freeze({
  coordinate: [0, 0, 0] as SolidWorldCoordinate,
  solidFraction: new Uint8Array(SOLID_WORLD_VOXELS_PER_PAGE),
  signedDistanceQ8: new Int16Array(SOLID_WORLD_VOXELS_PER_PAGE).fill(0x7fff),
  materialId: new Uint16Array(SOLID_WORLD_VOXELS_PER_PAGE),
  revision: 0, epoch: 0,
});

const nodeKey = (x: number, y: number, z: number): number | string =>
  x >= -NEAR_NODES && x < NEAR_NODES && y >= -NEAR_NODES && y < NEAR_NODES && z >= -NEAR_NODES && z < NEAR_NODES
    ? ((z + NEAR_NODES) * (2 * NEAR_NODES) + (y + NEAR_NODES)) * (2 * NEAR_NODES) + (x + NEAR_NODES)
    // A patch may stand anywhere in i32; past 4 million cells out the key is spelled instead.
    : `${x},${y},${z}`;

const pageSlot = (px: number, py: number, pz: number): number =>
  (px & 7) + NODE_PAGES * ((py & 7) + NODE_PAGES * (pz & 7));

const holdsSolid = (fractions: Uint8Array): boolean => {
  for (let cell = 0; cell < fractions.length; cell += 1) if (solidFractionOccupies(fractions[cell]!)) return true;
  return false;
};

let lastEpoch = 0;

/**
 * What the authored patches of a document compose to, as sparse 8-cell pages.
 *
 * This is the editor's copy of the solid world, built for asking questions
 * rather than for uploading: only the pages a patch touches exist, the ground
 * stays analytic everywhere else, and a document that grows by a stroke grows
 * its index by the stroke. Pages are written by `composeSolidPatch` over the
 * same `SolidGround` the solid world composes over, so a cell here is the cell
 * the world holds.
 *
 * The rule for which cells a patch has decided is per page: **a page exists
 * once any patch has touched it, and from then on it is the whole answer for
 * its 512 cells.** Over terrain it is seeded with the ground before the first
 * patch lands, so a fractional fill, a paint or a clear composes against what
 * was really there, and a cell the patch left alone still reads as ground.
 * Where no page exists, no patch has an opinion and the ground answers.
 *
 * An index is immutable once built. A later index shares every node and page
 * its patches did not touch, and copies the ones they did.
 */
export class VoxelIndex {
  /** Patches of `patches` this index composes; the list must not grow in place. */
  readonly length: number;

  constructor(
    readonly patches: readonly SolidWorldVoxelPatch[],
    readonly ground: SolidGround | undefined,
    private readonly nodes: ReadonlyMap<number | string, VoxelIndexNode>,
    /** The union of every patch box, or `undefined` for an empty list. */
    readonly minimum: SolidWorldCoordinate | undefined,
    readonly maximumExclusive: SolidWorldCoordinate | undefined,
  ) {
    this.length = patches.length;
  }

  /** The node holding node coordinate (cell / 64, floored), if any patch reached it. */
  node(nx: number, ny: number, nz: number): VoxelIndexNode | undefined {
    return this.nodes.get(nodeKey(nx, ny, nz));
  }

  /** The page at a page coordinate (cell / 8, floored): cells, the cleared page, or nothing authored. */
  page(px: number, py: number, pz: number): VoxelIndexPage | undefined {
    return this.nodes.get(nodeKey(px >> 3, py >> 3, pz >> 3))?.pages[pageSlot(px, py, pz)];
  }

  /** Pages holding cells, pages remembered as cleared, and nodes: what the index costs. */
  get statistics(): { readonly pages: number; readonly clearedPages: number; readonly nodes: number } {
    let pages = 0, clearedPages = 0;
    for (const node of this.nodes.values()) for (const page of node.pages) {
      if (page === VOXEL_INDEX_CLEARED_PAGE) clearedPages += 1;
      else if (page) pages += 1;
    }
    return { pages, clearedPages, nodes: this.nodes.size };
  }

  /** This index with `patches[from..]` composed over it. */
  static extend(base: VoxelIndex | undefined, patches: readonly SolidWorldVoxelPatch[],
    ground: SolidGround | undefined): VoxelIndex {
    return VoxelIndex.build(base, patches, ground).finish();
  }

  /**
   * `extend`, able to stop between any two patches.
   *
   * A document that arrives whole — a load, a preset — is every patch at once,
   * and composing them is the one cost here that is the size of the document
   * rather than of an edit. Built in turns it can be paid before anyone points
   * at the scene instead of by the first pointer move over it.
   */
  static build(base: VoxelIndex | undefined, patches: readonly SolidWorldVoxelPatch[],
    ground: SolidGround | undefined): VoxelIndexBuild {
    const epoch = ++lastEpoch;
    const nodes = new Map(base?.nodes);
    const written: VoxelIndexPage[] = [];
    const minimum = base?.minimum ? [...base.minimum] : [Infinity, Infinity, Infinity];
    const maximum = base?.maximumExclusive ? [...base.maximumExclusive] : [-Infinity, -Infinity, -Infinity];
    const store: SolidPatchPageStore = {
      page(coordinate, create) {
        const key = nodeKey(coordinate[0] >> 3, coordinate[1] >> 3, coordinate[2] >> 3);
        const slot = pageSlot(coordinate[0], coordinate[1], coordinate[2]);
        let node = nodes.get(key);
        const existing = node?.pages[slot];
        if (existing?.epoch === epoch) return existing;
        let page: VoxelIndexPage;
        if (existing && existing !== VOXEL_INDEX_CLEARED_PAGE) {
          page = { coordinate: existing.coordinate, solidFraction: existing.solidFraction.slice(),
            signedDistanceQ8: existing.signedDistanceQ8.slice(), materialId: existing.materialId.slice(),
            revision: existing.revision + 1, epoch };
        } else {
          // A cleared page has already had its ground taken away; only a page
          // nobody has touched starts from the terrain.
          const grounded = !existing && ground?.stands(coordinate[0], coordinate[1], coordinate[2]);
          if (!create && !grounded) return undefined;
          page = { coordinate: [coordinate[0], coordinate[1], coordinate[2]],
            solidFraction: new Uint8Array(SOLID_WORLD_VOXELS_PER_PAGE),
            signedDistanceQ8: new Int16Array(SOLID_WORLD_VOXELS_PER_PAGE).fill(0x7fff),
            materialId: new Uint16Array(SOLID_WORLD_VOXELS_PER_PAGE), revision: 1, epoch };
          if (grounded && !ground!.seed(page) && !create) return undefined;
        }
        if (!node) nodes.set(key, node = { epoch, pages: new Array(NODE_PAGES ** 3) });
        else if (node.epoch !== epoch) nodes.set(key, node = { epoch, pages: node.pages.slice() });
        node.pages[slot] = page;
        written.push(page);
        return page;
      },
    };
    const total = patches.length;
    let next = base?.length ?? 0;
    const advance = (expired?: () => boolean): boolean => {
      while (next < total) {
        const patch = patches[next]!;
        validateSolidPatch(patch);
        // A patch composes whole, so a turn can only stop inside one before it
        // starts: the ground under it is sampled first, a tile at a time.
        if (expired && ground && !ground.sample(patch.minimum, patch.maximumExclusive, expired)) break;
        for (let axis = 0; axis < 3; axis += 1) {
          minimum[axis] = Math.min(minimum[axis]!, patch.minimum[axis]!);
          maximum[axis] = Math.max(maximum[axis]!, patch.maximumExclusive[axis]!);
        }
        composeSolidPatch(patch, store);
        next += 1;
        if (expired?.()) break;
      }
      return next >= total;
    };
    const finish = (): VoxelIndex => {
      advance();
      // Only a page this build wrote can have become empty. Over the ground it
      // stays as the cleared page; anywhere else an empty page is no page at
      // all, exactly as the solid world drops it.
      for (const page of written) {
        if (holdsSolid(page.solidFraction)) continue;
        const q = page.coordinate;
        const key = nodeKey(q[0] >> 3, q[1] >> 3, q[2] >> 3);
        const node = nodes.get(key)!;
        node.pages[pageSlot(q[0], q[1], q[2])] = ground?.stands(q[0], q[1], q[2]) ? VOXEL_INDEX_CLEARED_PAGE : undefined;
        if (node.pages.every((entry) => entry === undefined)) nodes.delete(key);
      }
      const bounded = total > 0;
      return new VoxelIndex(patches, ground, nodes,
        bounded ? minimum as unknown as SolidWorldCoordinate : undefined,
        bounded ? maximum as unknown as SolidWorldCoordinate : undefined);
    };
    return { base, ground, total, advance, finish };
  }
}

/** An index part-way through being composed. Nothing may be asked of it until `finish`. */
export interface VoxelIndexBuild {
  readonly base: VoxelIndex | undefined;
  readonly ground: SolidGround | undefined;
  /** Patches the finished index will hold. */
  readonly total: number;
  /** Compose patches until `expired` says to stop. True once every patch is in. */
  advance(expired?: () => boolean): boolean;
  finish(): VoxelIndex;
}

const indexes = new WeakMap<readonly SolidWorldVoxelPatch[], VoxelIndex>();
/**
 * Indexes still alive, newest first: the bases a new list may extend.
 *
 * Weak, because an index lives exactly as long as a patch list it answers for.
 * As deep as the undo history and a little more, so every document undo can
 * walk back to is found by its content — each arrives as a clone nobody has
 * seen — along with a stroke's base and the proposals made over it.
 */
const recentIndexes: WeakRef<VoxelIndex>[] = [];
const RECENT_INDEXES = EDITOR_HISTORY_LIMIT + 8;

const sameBox = (left: SolidWorldVoxelPatch, right: SolidWorldVoxelPatch): boolean =>
  left === right || (left.operation === right.operation
    && left.minimum.every((value, axis) => value === right.minimum[axis])
    && left.maximumExclusive.every((value, axis) => value === right.maximumExclusive[axis]));

/**
 * The index of an ordered patch list standing on `ground`.
 *
 * Memoized per list, and incremental: a list that is an indexed list plus a
 * suffix copies only the pages the suffix touches, and a list that equals an
 * indexed one (a clone, an undo back to a remembered document) takes that
 * index whole. Anything else composes every patch from nothing.
 */
export function voxelIndexForPatches(patches: readonly SolidWorldVoxelPatch[],
  ground: SolidGround | undefined): VoxelIndex {
  const cached = indexes.get(patches);
  if (cached && cached.ground === ground && cached.length === patches.length) return cached;
  // A question cannot wait for the turns a preparation had left.
  const build = buildFor(patches, ground);
  builds.delete(patches);
  const base = build ? build.base : baseFor(patches, ground);
  const index = build ? build.finish()
    : base && base.length === patches.length ? base : VoxelIndex.extend(base, patches, ground);
  adopt(patches, index, base);
  return index;
}

const builds = new WeakMap<readonly SolidWorldVoxelPatch[], VoxelIndexBuild>();

function buildFor(patches: readonly SolidWorldVoxelPatch[], ground: SolidGround | undefined): VoxelIndexBuild | undefined {
  const build = builds.get(patches);
  return build && build.ground === ground && build.total === patches.length ? build : undefined;
}

/**
 * Build the index of a list ahead of the first question asked of it.
 *
 * Composes until `expired` says to stop and picks up where it left off on the
 * next call; true once the index stands. `voxelIndexForPatches` finishes
 * whatever is left if it is asked first, so preparing never changes an answer —
 * only who waits for it.
 */
export function prepareVoxelIndex(patches: readonly SolidWorldVoxelPatch[], ground: SolidGround | undefined,
  expired: () => boolean): boolean {
  const cached = indexes.get(patches);
  if (cached && cached.ground === ground && cached.length === patches.length) return true;
  let build = buildFor(patches, ground);
  if (!build) {
    const base = baseFor(patches, ground);
    if (base && base.length === patches.length) { adopt(patches, base, base); return true; }
    builds.set(patches, build = VoxelIndex.build(base, patches, ground));
  }
  if (!build.advance(expired)) return false;
  builds.delete(patches);
  adopt(patches, build.finish(), build.base);
  return true;
}

/** The longest remembered index `patches` begins with, to extend rather than compose from nothing. */
function baseFor(patches: readonly SolidWorldVoxelPatch[], ground: SolidGround | undefined): VoxelIndex | undefined {
  const candidates: VoxelIndex[] = [];
  for (let entry = recentIndexes.length - 1; entry >= 0; entry -= 1) {
    const candidate = recentIndexes[entry]!.deref();
    if (!candidate) recentIndexes.splice(entry, 1);
    // A list that grew in place after it was indexed no longer says what the index holds.
    else if (candidate.ground === ground && candidate.patches.length === candidate.length
      && candidate.length <= patches.length) candidates.push(candidate);
  }
  // Longest first: the base worth having is the one that leaves the shortest suffix.
  return candidates.sort((left, right) => right.length - left.length).find((candidate) =>
    // Nearly every list that is not a prefix differs at its own last patch.
    (candidate.length === 0 || sameBox(candidate.patches[candidate.length - 1]!, patches[candidate.length - 1]!))
    && solidPatchListExtends(candidate.patches, patches));
}

function adopt(patches: readonly SolidWorldVoxelPatch[], index: VoxelIndex, base: VoxelIndex | undefined): void {
  indexes.set(patches, index);
  // The base goes back to the front with what it produced: a document that
  // every proposal of a long stroke extends must not age out behind them.
  if (base && base !== index) remember(base);
  remember(index);
}

function remember(index: VoxelIndex): void {
  const held = recentIndexes.findIndex((entry) => entry.deref() === index);
  if (held === 0) return;
  if (held > 0) recentIndexes.splice(held, 1);
  recentIndexes.unshift(new WeakRef(index));
  recentIndexes.length = Math.min(recentIndexes.length, RECENT_INDEXES);
}

/** The index of a scene's authored solids over its own terrain. */
export function voxelIndexForScene(
  scene: Pick<SceneDescription, "solidVoxels" | "terrain" | "container" | "voxelDomain">,
): VoxelIndex {
  return voxelIndexForPatches(scene.solidVoxels, solidGroundForScene(scene));
}

const PREPARATION_TURN_MS = 1.5;

/**
 * Keep the index of a pane's document built, in turns short enough to sit
 * between two frames. Returns the stop function.
 *
 * Every question the editor asks of the solids goes through the index, and the
 * first one after a document arrives used to be the one that built it: the
 * first pointer move over a loaded scene. This builds it as soon as the
 * document is in the store instead. After an edit there is one stroke to add,
 * which is one short turn; after a load there is the whole list, which is many.
 */
export function startVoxelIndexPreparation(
  scenes: { getState(): { scene: Parameters<typeof voxelIndexForScene>[0] }; subscribe(listener: () => void): () => void },
): () => void {
  let turn: ReturnType<typeof setTimeout> | undefined;
  const prepare = () => {
    turn = undefined;
    const { scene } = scenes.getState();
    const start = performance.now();
    const ground = solidGroundForScene(scene);
    const expired = () => performance.now() - start >= PREPARATION_TURN_MS;
    // Then the rest of the ground: a pick crosses columns no patch stands on.
    if (!prepareVoxelIndex(scene.solidVoxels, ground, expired)
      || (ground && !ground.sample(ground.minimum, ground.maximumExclusive, expired))) turn = setTimeout(prepare, 0);
  };
  const schedule = () => { turn ??= setTimeout(prepare, 0); };
  const stop = scenes.subscribe(schedule);
  schedule();
  return () => {
    stop();
    if (turn !== undefined) clearTimeout(turn);
  };
}
