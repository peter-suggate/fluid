import type { SparseBrickCoordinate, SparseBrickOctreeGPU, SparseBrickPlan } from "../svo/features/construction/sparse-brick-octree";
import { SPARSE_BRICK_GPU_LAYOUT, SPARSE_BRICK_INVALID_INDEX, SPARSE_BRICK_LEAF_TERMINAL } from "../svo/features/construction/sparse-brick-octree";
import { SVO_BRICK_LIFECYCLE, SVO_BRICK_OCCUPANCY } from "../svo/features/construction/svo-brick-occupancy";

/**
 * The live tree's allocator is the host's. `SparseBrickTopologyMirror` holds
 * every node's links and every leaf's backlink, hands out node and leaf slots
 * from its own free lists, and writes each record an edit changes in its final
 * form. The GPU pass is a scatter of those records: it allocates nothing and
 * decides nothing, so the host knows the tree after an edit without reading it
 * back, and an edit taken back returns exactly the slots it took.
 */
export const SPARSE_BRICK_TOPOLOGY_MUTATION = Object.freeze({
  headerWords: 8,
  recordWords: 8,
  /** Records in the batch; nonzero is what tells a publication its topology moved. */
  countWord: 0,
  generationWord: 1,
  capacityWord: 2,
  /** Written by the GPU: nonzero means the whole batch was refused. */
  overflowFlagsWord: 3,
  /** Node and leaf high-water marks and resident leaves after the batch. */
  nodeCountWord: 4,
  leafCountWord: 5,
  activeLeavesWord: 6,
  /** Node slot the record writes; invalid for a record that only retires a leaf. */
  recordDestinationWord: 0,
  /** Node slot whose flags the record carries over; the gather pass replaces it with the flags. */
  recordSourceWord: 1,
  recordKeyLowWord: 2,
  recordKeyHighWord: 3,
  recordFirstChildWord: 4,
  recordLeafWord: 5,
  /** Level, child mask, terminal kind and flag mode. */
  recordPackedWord: 6,
  recordTerminalWord: 7,
  levelMask: 0x1f,
  childMaskShift: 8,
  terminalKindShift: 16,
  modeShift: 20,
  /** Carry the source's occupancy summary. */
  modeKeepOccupancy: 1,
  /** Carry the source's lifecycle bits. */
  modeKeepLifecycle: 2,
  /** A leaf to voxelize: active, dirty and queued. */
  modeSetLive: 4,
  overflowMalformedRecord: 8,
  overflowRequestBudget: 16,
});

export interface SparseBrickTopologyMutationWorklist {
  buffer: GPUBuffer;
  /** Maximum records allocated after the eight-word header. */
  capacity: number;
}

export interface SparseBrickTopologyMutationOptions {
  generation: number;
  /** Records the staged batch holds, which sizes both dispatches. */
  recordCount: number;
}

/**
 * Node slots a lifetime budget of reserved leaves can hold at once. A node
 * whose children an edit changed keeps them in one eight-slot block, and every
 * such node has a reserved leaf beneath it, so a leaf accounts for at most one
 * block for each of its ancestors. The last slot is the root of an empty tree.
 */
export function sparseBrickTopologyMutationNodeReserve(
  maximumDepth: number,
  uniqueActivationBudget: number,
): number {
  if (!Number.isSafeInteger(maximumDepth) || maximumDepth < 0 || maximumDepth > 21) {
    throw new RangeError("Topology mutation depth must be 0..21");
  }
  if (!Number.isSafeInteger(uniqueActivationBudget) || uniqueActivationBudget < 0) {
    throw new RangeError("Unique topology activation budget must be a non-negative safe integer");
  }
  const reserve = (maximumDepth === 0 ? 0 : 8 * maximumDepth * uniqueActivationBudget) + 1;
  if (!Number.isSafeInteger(reserve)) throw new RangeError("Topology mutation reserve exceeds safe integer range");
  return reserve;
}

/**
 * Morton order of two finest bricks, x in the low bit as the tree's octants
 * are. Bricks activated ascending and released descending take and return
 * slots as a stack, which is what makes an edit's slots its own to give back.
 */
export function compareSparseBrickMorton(a: SparseBrickCoordinate, b: SparseBrickCoordinate): number {
  const dx = a.x ^ b.x, dy = a.y ^ b.y, dz = a.z ^ b.z;
  // The axis whose highest differing bit is highest decides, z before y before x.
  const below = (p: number, q: number): boolean => p < q && p < (p ^ q);
  if (below(dz, dy) ? below(dy, dx) : below(dz, dx)) return a.x - b.x;
  return below(dz, dy) ? a.y - b.y : a.z - b.z;
}

const INVALID = SPARSE_BRICK_INVALID_INDEX;
const M = SPARSE_BRICK_TOPOLOGY_MUTATION;
const KEEP_ALL = M.modeKeepOccupancy | M.modeKeepLifecycle;
/** A terminal an edit may split, or turn into a voxel leaf where it is already a finest brick. */
const LEAF_SPLITTABLE = 1;
/** A finest voxel leaf an edit brought into being, which an edit may take back. */
const LEAF_RELEASABLE = 2;

function popcount8(value: number): number {
  let bits = value & 0xff;
  bits -= (bits >>> 1) & 0x55;
  bits = (bits & 0x33) + ((bits >>> 2) & 0x33);
  return (bits + (bits >>> 4)) & 0x0f;
}

/** A node record the GPU has not been sent yet, and where its flags come from. */
interface PendingNode {
  /** Slot holding the flags on the GPU now; invalid for a node it has never held. */
  source: number;
  mode: number;
  keyLow: number;
  keyHigh: number;
  level: number;
}

/** What a set of unique finest bricks would take from the arenas. */
export interface SparseBrickTopologyDemand {
  /** Leaves: one a brick, and seven for each terminal split on the way to one. */
  readonly leaves: number;
  /** Eight-slot child blocks, at most. */
  readonly blocks: number;
  /** An upper bound on the records the batch would write. */
  readonly records: number;
}

export interface SparseBrickTopologyMirrorOptions {
  readonly maximumDepth: number;
  readonly nodeCapacity: number;
  readonly leafCapacity: number;
  /** Records one batch may carry. */
  readonly recordCapacity: number;
  /** The planned tree, which is the one the GPU is published. */
  readonly plan: Pick<SparseBrickPlan, "nodes" | "leaves" | "maximumDepth">;
  /**
   * The planned terminals an edit may split: analytic ones, and sampled ones
   * the caller resamples.
   */
  readonly splittable: (leafIndex: number) => boolean;
}

/**
 * Fixed slots given out from a LIFO free list, and from the end once that is
 * empty. Freeing the end slot lowers the end past every free slot beneath it,
 * so the list never holds the end slot, and slots freed in the reverse of the
 * order they were taken leave the arena exactly as it was.
 */
class SlotArena {
  private freeCount = 0;
  private head = INVALID;
  private readonly next: Uint32Array;
  private readonly previous: Uint32Array;
  private readonly free: Uint8Array;

  constructor(readonly capacity: number, public high: number) {
    this.next = new Uint32Array(capacity);
    this.previous = new Uint32Array(capacity);
    this.free = new Uint8Array(capacity);
  }

  get available(): number { return this.capacity - this.high + this.freeCount; }
  get used(): number { return this.high - this.freeCount; }

  private unlink(slot: number): void {
    const next = this.next[slot]!, previous = this.previous[slot]!;
    if (previous === INVALID) this.head = next;
    else this.next[previous] = next;
    if (next !== INVALID) this.previous[next] = previous;
    this.free[slot] = 0;
    this.freeCount -= 1;
  }

  /** A slot, or invalid when none is left. */
  take(): number {
    const slot = this.head;
    if (slot !== INVALID) { this.unlink(slot); return slot; }
    return this.high < this.capacity ? this.high++ : INVALID;
  }

  give(slot: number): void {
    if (slot !== this.high - 1) {
      this.next[slot] = this.head;
      this.previous[slot] = INVALID;
      if (this.head !== INVALID) this.previous[this.head] = slot;
      this.head = slot;
      this.free[slot] = 1;
      this.freeCount += 1;
      return;
    }
    this.high -= 1;
    while (this.high > 0 && this.free[this.high - 1] === 1) {
      this.unlink(this.high - 1);
      this.high -= 1;
    }
  }
}

/**
 * The host's exact copy of the live tree's links, and its allocator.
 *
 * Children stay a compact run in octant order, as every traversal reads them.
 * A parent whose child set is the planned one keeps the run the plan gave it;
 * any other keeps its children at the front of one eight-slot block, rewritten
 * in place as children come and go. Blocks and leaves come off LIFO free lists
 * or the arena's end, and a slot freed at the end lowers the end again, so
 * bricks released in the reverse of the order they were activated leave both
 * arenas exactly as they were.
 *
 * Every change is recorded against the tree the GPU last received, so several
 * edits staged before one frame collapse into the single batch that takes the
 * GPU from that tree to the current one.
 */
export class SparseBrickTopologyMirror {
  readonly maximumDepth: number;
  readonly nodeCapacity: number;
  readonly leafCapacity: number;
  readonly recordCapacity: number;
  private readonly nodeMask: Uint8Array;
  private readonly nodeFirst: Uint32Array;
  private readonly nodeLeaf: Uint32Array;
  private readonly nodePlanMask: Uint8Array;
  private readonly nodePlanFirst: Uint32Array;
  private readonly leafNode: Uint32Array;
  private readonly leafKind: Uint8Array;
  private readonly leafTerminal: Uint32Array;
  private readonly leafFlags: Uint8Array;
  private readonly plannedNodes: number;
  /** The first slot of block zero: past the planned nodes, or past the root of a tree planned empty. */
  private readonly blockBase: number;
  private readonly blocks: SlotArena;
  private readonly leaves: SlotArena;
  /** Whether slot zero holds a root; false only for a tree planned empty that holds no brick. */
  private rootLive: boolean;
  private readonly pending = new Map<number, PendingNode>();
  /** Leaves freed since the GPU's tree, which may need a retiring record. */
  private readonly freedLeaves = new Set<number>();
  private topologyChanged = false;
  private readonly pathSlot: Uint32Array;
  private readonly pathKeyLow: Uint32Array;
  private readonly pathKeyHigh: Uint32Array;
  private readonly childOctant = new Uint8Array(8);
  private readonly childSlot = new Uint32Array(8);
  private readonly childMask = new Uint8Array(8);
  private readonly childFirst = new Uint32Array(8);
  private readonly childLeaf = new Uint32Array(8);
  private readonly childPlanMask = new Uint8Array(8);
  private readonly childPlanFirst = new Uint32Array(8);
  private readonly childPending: (PendingNode | undefined)[] = new Array(8).fill(undefined);

  constructor(options: SparseBrickTopologyMirrorOptions) {
    const { maximumDepth, nodeCapacity, leafCapacity, recordCapacity, plan, splittable } = options;
    if (!Number.isSafeInteger(maximumDepth) || maximumDepth < 0 || maximumDepth > 21) {
      throw new RangeError("Topology mutation depth must be 0..21");
    }
    for (const [name, value] of [["node capacity", nodeCapacity], ["leaf capacity", leafCapacity],
      ["record capacity", recordCapacity]] as const) {
      if (!Number.isSafeInteger(value) || value < 1 || value >= INVALID) {
        throw new RangeError(`Topology mirror ${name} must be a positive integer below 2^32 - 1`);
      }
    }
    this.maximumDepth = maximumDepth;
    this.nodeCapacity = nodeCapacity;
    this.leafCapacity = leafCapacity;
    this.recordCapacity = recordCapacity;
    this.nodeMask = new Uint8Array(nodeCapacity);
    this.nodeFirst = new Uint32Array(nodeCapacity).fill(INVALID);
    this.nodeLeaf = new Uint32Array(nodeCapacity).fill(INVALID);
    this.nodePlanMask = new Uint8Array(nodeCapacity);
    this.nodePlanFirst = new Uint32Array(nodeCapacity).fill(INVALID);
    this.leafNode = new Uint32Array(leafCapacity).fill(INVALID);
    this.leafKind = new Uint8Array(leafCapacity);
    this.leafTerminal = new Uint32Array(leafCapacity).fill(INVALID);
    this.leafFlags = new Uint8Array(leafCapacity);
    this.pathSlot = new Uint32Array(maximumDepth + 1);
    this.pathKeyLow = new Uint32Array(maximumDepth + 1);
    this.pathKeyHigh = new Uint32Array(maximumDepth + 1);
    if (plan.maximumDepth !== maximumDepth) throw new RangeError("Planned tree depth differs from the mirror's");
    if (plan.nodes.length > nodeCapacity || plan.leaves.length > leafCapacity) {
      throw new RangeError("Planned tree exceeds the topology mirror's capacity");
    }
    for (const node of plan.nodes) {
      const first = node.childMask === 0 ? INVALID : node.firstChild;
      this.nodeMask[node.index] = node.childMask;
      this.nodeFirst[node.index] = first;
      this.nodeLeaf[node.index] = node.leafIndex;
      this.nodePlanMask[node.index] = node.childMask;
      this.nodePlanFirst[node.index] = first;
    }
    for (const leaf of plan.leaves) {
      this.leafNode[leaf.index] = leaf.nodeIndex;
      this.leafKind[leaf.index] = leaf.terminalKind;
      this.leafTerminal[leaf.index] = leaf.terminalIndex;
      this.leafFlags[leaf.index] = splittable(leaf.index) ? LEAF_SPLITTABLE : 0;
    }
    this.plannedNodes = plan.nodes.length;
    this.rootLive = plan.nodes.length > 0;
    this.blockBase = Math.max(1, plan.nodes.length);
    this.blocks = new SlotArena(Math.floor((nodeCapacity - this.blockBase) / 8), 0);
    this.leaves = new SlotArena(leafCapacity, plan.leaves.length);
  }

  /** Leaves the arena can still give out. */
  get freeLeaves(): number { return this.leaves.available; }
  get residentLeaves(): number { return this.leaves.used; }
  /** Node slots below which every live node sits, which is what the tree publishes as its node count. */
  get nodeCount(): number {
    return this.blocks.high > 0 ? this.blockBase + 8 * this.blocks.high : this.rootLive ? Math.max(1, this.plannedNodes) : 0;
  }
  get leafCount(): number { return this.leaves.high; }
  /** Eight-slot blocks the node arena can still give out. */
  get freeBlocks(): number { return this.blocks.available; }
  /** An upper bound on the records `pack` would write now. */
  get pendingRecords(): number { return this.pending.size + this.freedLeaves.size; }
  /** Whether the tree differs from the one the GPU was last sent. */
  get hasPendingRecords(): boolean { return this.topologyChanged; }

  private octant(coordinate: SparseBrickCoordinate, level: number): number {
    const bit = this.maximumDepth - level;
    return ((coordinate.x >>> bit) & 1) | (((coordinate.y >>> bit) & 1) << 1) | (((coordinate.z >>> bit) & 1) << 2);
  }

  /**
   * The slot of a brick's finest node, or invalid where the tree stops short
   * of it. With `path`, the slots and keys on the way are kept for a caller
   * that walks back up.
   */
  private finestNode(coordinate: SparseBrickCoordinate, path: boolean): number {
    if (!this.rootLive) return INVALID;
    let node = 0, keyLow = 0, keyHigh = 0;
    for (let level = 0; ; level += 1) {
      if (path) { this.pathSlot[level] = node; this.pathKeyLow[level] = keyLow; this.pathKeyHigh[level] = keyHigh; }
      if (level === this.maximumDepth) return node;
      const octant = this.octant(coordinate, level + 1), mask = this.nodeMask[node]!;
      if ((mask & (1 << octant)) === 0) return INVALID;
      node = this.nodeFirst[node]! + popcount8(mask & ((1 << octant) - 1));
      keyHigh = ((keyHigh << 3) | (keyLow >>> 29)) >>> 0;
      keyLow = ((keyLow << 3) | octant) >>> 0;
    }
  }

  /**
   * Whether some leaf already holds this finest brick's voxels. A terminal an
   * edit can split does not: it stores no authored detail until it is split.
   */
  covered(coordinate: SparseBrickCoordinate): boolean {
    if (!this.rootLive) return false;
    let node = 0;
    for (let level = 0; ; level += 1) {
      const leaf = this.nodeLeaf[node]!;
      if (leaf !== INVALID) return (this.leafFlags[leaf]! & LEAF_SPLITTABLE) === 0;
      if (level === this.maximumDepth) return false;
      const octant = this.octant(coordinate, level + 1), mask = this.nodeMask[node]!;
      if ((mask & (1 << octant)) === 0) return false;
      node = this.nodeFirst[node]! + popcount8(mask & ((1 << octant) - 1));
    }
  }

  /** Whether this finest brick is a leaf an edit activated and `release` would take back. */
  releasable(coordinate: SparseBrickCoordinate): boolean {
    const node = this.finestNode(coordinate, false);
    const leaf = node === INVALID ? INVALID : this.nodeLeaf[node]!;
    return leaf !== INVALID && (this.leafFlags[leaf]! & LEAF_RELEASABLE) !== 0;
  }

  private prefixKey(coordinate: SparseBrickCoordinate, level: number): number | string {
    const shift = this.maximumDepth - level;
    const x = coordinate.x >>> shift, y = coordinate.y >>> shift, z = coordinate.z >>> shift;
    // Sixteen bits an axis and five of level stay exact in a double.
    return this.maximumDepth <= 16 ? ((level * 65536 + x) * 65536 + y) * 65536 + z : `${level}:${x},${y},${z}`;
  }

  /**
   * What activating these bricks would take, without changing anything.
   * `coordinates` are unique finest bricks no leaf covers.
   */
  demand(coordinates: readonly SparseBrickCoordinate[]): SparseBrickTopologyDemand {
    // Nodes the batch creates, and existing ones whose children it changes.
    const seen = new Set<number | string>();
    let leaves = 0, blocks = 0, records = 0;
    const first = (coordinate: SparseBrickCoordinate, level: number): boolean => {
      const key = this.prefixKey(coordinate, level);
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    };
    for (const coordinate of coordinates) {
      let node = this.rootLive ? 0 : INVALID;
      let level = 0, terminal = INVALID;
      while (node !== INVALID) {
        terminal = this.nodeLeaf[node]!;
        if (terminal !== INVALID || level === this.maximumDepth) break;
        const octant = this.octant(coordinate, level + 1), mask = this.nodeMask[node]!;
        if ((mask & (1 << octant)) === 0) break;
        node = this.nodeFirst[node]! + popcount8(mask & ((1 << octant) - 1));
        level += 1;
      }
      if (terminal !== INVALID) {
        if ((this.leafFlags[terminal]! & LEAF_SPLITTABLE) === 0) continue;
        // A finest terminal becomes a voxel leaf in the leaf it has.
        if (level === this.maximumDepth) { if (first(coordinate, level)) records += 1; continue; }
        // A split writes its node and eight children, and keeps the node's leaf.
        for (let below = level; below < this.maximumDepth; below += 1) {
          if (first(coordinate, below)) { records += 9; blocks += 1; leaves += 7; }
        }
        continue;
      }
      leaves += 1;
      let created = 0;
      if (node !== INVALID) {
        if (level === this.maximumDepth) { if (first(coordinate, level)) records += 1; continue; }
        // The deepest node there is gains a child, and all it has may move.
        const mask = this.nodeMask[node]!, planMask = this.nodePlanMask[node]!;
        if (first(coordinate, level)) {
          records += 1 + popcount8(mask);
          if (mask === 0 || (planMask !== 0 && mask === planMask)) blocks += 1;
        }
        created = level + 1;
      }
      for (let below = created; below <= this.maximumDepth; below += 1) {
        if (!first(coordinate, below)) continue;
        records += 1;
        if (below < this.maximumDepth) blocks += 1;
      }
    }
    return { leaves, blocks, records };
  }

  private allocateLeaf(): number {
    const leaf = this.leaves.take();
    if (leaf === INVALID) throw new Error("Topology mirror ran out of leaves past its preflight");
    return leaf;
  }

  private freeLeaf(leaf: number): void {
    this.leafNode[leaf] = INVALID;
    this.leafFlags[leaf] = 0;
    this.freedLeaves.add(leaf);
    this.leaves.give(leaf);
  }

  private allocateBlock(): number {
    const block = this.blocks.take();
    if (block === INVALID) throw new Error("Topology mirror ran out of nodes past its preflight");
    return this.blockBase + 8 * block;
  }

  private freeBlock(block: number): void {
    this.blocks.give((block - this.blockBase) / 8);
  }

  /** The record a node will be sent, made on first touch from the flags it holds now. */
  private touch(node: number, keyLow: number, keyHigh: number, level: number): PendingNode {
    let entry = this.pending.get(node);
    if (!entry) this.pending.set(node, entry = { source: node, mode: KEEP_ALL, keyLow, keyHigh, level });
    this.topologyChanged = true;
    return entry;
  }

  private placeNode(slot: number, mask: number, first: number, leaf: number, planMask: number, planFirst: number): void {
    this.nodeMask[slot] = mask;
    this.nodeFirst[slot] = first;
    this.nodeLeaf[slot] = leaf;
    this.nodePlanMask[slot] = planMask;
    this.nodePlanFirst[slot] = planFirst;
    if (leaf !== INVALID) this.leafNode[leaf] = slot;
  }

  /**
   * Rewrite a parent's children with one octant removed or one added, and
   * return the added child's slot. The run goes back to the planned slots when
   * the child set is the planned one, and otherwise sits in one block.
   */
  private relayout(parent: number, keyLow: number, keyHigh: number, level: number,
    removed: number, added: number): number {
    const oldMask = this.nodeMask[parent]!, oldFirst = this.nodeFirst[parent]!;
    const planMask = this.nodePlanMask[parent]!;
    const inBlock = oldMask !== 0 && !(planMask !== 0 && oldMask === planMask);
    let kept = 0;
    for (let octant = 0, ordinal = 0; octant < 8; octant += 1) {
      if ((oldMask & (1 << octant)) === 0) continue;
      const slot = oldFirst + ordinal;
      ordinal += 1;
      const entry = this.pending.get(slot);
      this.pending.delete(slot);
      if (octant === removed) continue;
      this.childOctant[kept] = octant;
      this.childSlot[kept] = slot;
      this.childMask[kept] = this.nodeMask[slot]!;
      this.childFirst[kept] = this.nodeFirst[slot]!;
      this.childLeaf[kept] = this.nodeLeaf[slot]!;
      this.childPlanMask[kept] = this.nodePlanMask[slot]!;
      this.childPlanFirst[kept] = this.nodePlanFirst[slot]!;
      this.childPending[kept] = entry;
      kept += 1;
    }
    let newMask = oldMask;
    if (removed >= 0) newMask &= ~(1 << removed);
    if (added >= 0) newMask |= 1 << added;
    let first: number;
    if (newMask === 0) {
      first = INVALID;
      if (inBlock) this.freeBlock(oldFirst);
    } else if (planMask !== 0 && newMask === planMask) {
      first = this.nodePlanFirst[parent]!;
      if (inBlock) this.freeBlock(oldFirst);
    } else {
      first = inBlock ? oldFirst : this.allocateBlock();
    }
    const childKeyHigh = ((keyHigh << 3) | (keyLow >>> 29)) >>> 0;
    let inserted = INVALID;
    for (let octant = 0, ordinal = 0, child = 0; octant < 8; octant += 1) {
      if ((newMask & (1 << octant)) === 0) continue;
      const slot = first + ordinal;
      ordinal += 1;
      const childKeyLow = ((keyLow << 3) | octant) >>> 0;
      if (octant === added) {
        this.placeNode(slot, 0, INVALID, INVALID, 0, INVALID);
        this.pending.set(slot, { source: INVALID, mode: 0, keyLow: childKeyLow, keyHigh: childKeyHigh, level: level + 1 });
        inserted = slot;
        continue;
      }
      const from = this.childSlot[child]!;
      const entry = this.childPending[child];
      this.placeNode(slot, this.childMask[child]!, this.childFirst[child]!, this.childLeaf[child]!,
        this.childPlanMask[child]!, this.childPlanFirst[child]!);
      this.childPending[child] = undefined;
      child += 1;
      // A child that stays where it was keeps whatever record it already had.
      if (entry) this.pending.set(slot, entry);
      else if (slot !== from) {
        this.pending.set(slot, { source: from, mode: KEEP_ALL, keyLow: childKeyLow, keyHigh: childKeyHigh, level: level + 1 });
      }
    }
    this.nodeMask[parent] = newMask;
    this.nodeFirst[parent] = first;
    this.touch(parent, keyLow, keyHigh, level);
    return inserted;
  }

  /**
   * Split a terminal into eight that each keep its representation. Child zero
   * keeps the terminal's own leaf, so a split takes seven.
   */
  private split(node: number, keyLow: number, keyHigh: number, level: number, leaf: number): void {
    const kind = this.leafKind[leaf]!, terminal = this.leafTerminal[leaf]!;
    const parent = this.pending.get(node);
    const source = parent ? parent.source : node;
    // An analytic terminal's occupancy summary still describes each of its octants.
    const occupancy = kind === SPARSE_BRICK_LEAF_TERMINAL.planarBoundary && source !== INVALID
      && (!parent || (parent.mode & M.modeKeepOccupancy) !== 0);
    const block = this.allocateBlock();
    const childKeyHigh = ((keyHigh << 3) | (keyLow >>> 29)) >>> 0;
    for (let octant = 0; octant < 8; octant += 1) {
      const slot = block + octant, childLeaf = octant === 0 ? leaf : this.allocateLeaf();
      this.placeNode(slot, 0, INVALID, childLeaf, 0, INVALID);
      this.leafKind[childLeaf] = kind;
      this.leafTerminal[childLeaf] = terminal;
      this.leafFlags[childLeaf] = LEAF_SPLITTABLE;
      this.pending.set(slot, { source: occupancy ? source : INVALID,
        mode: M.modeSetLive | (occupancy ? M.modeKeepOccupancy : 0),
        keyLow: ((keyLow << 3) | octant) >>> 0, keyHigh: childKeyHigh, level: level + 1 });
    }
    this.nodeLeaf[node] = INVALID;
    this.nodeMask[node] = 0xff;
    this.nodeFirst[node] = block;
    this.pending.set(node, { source: INVALID, mode: 0, keyLow, keyHigh, level });
    this.topologyChanged = true;
  }

  /**
   * Bring a finest brick's voxel leaf into being: the nodes down to it, a split
   * of every terminal on the way, and a leaf to voxelize. False where a leaf
   * already covers the brick.
   */
  activate(coordinate: SparseBrickCoordinate): boolean {
    if (!this.rootLive) {
      this.rootLive = true;
      this.placeNode(0, 0, INVALID, INVALID, 0, INVALID);
      this.pending.set(0, { source: INVALID, mode: 0, keyLow: 0, keyHigh: 0, level: 0 });
    }
    let node = 0, keyLow = 0, keyHigh = 0;
    for (let level = 0; ; level += 1) {
      const leaf = this.nodeLeaf[node]!;
      if (leaf !== INVALID) {
        if ((this.leafFlags[leaf]! & LEAF_SPLITTABLE) === 0) return false;
        if (level === this.maximumDepth) {
          // Runtime topology growth is voxel residency. Analytic terminals are
          // authored by the scene planner and never inferred from an edit.
          this.leafKind[leaf] = SPARSE_BRICK_LEAF_TERMINAL.voxels;
          this.leafTerminal[leaf] = INVALID;
          this.leafFlags[leaf] = 0;
          this.touch(node, keyLow, keyHigh, level).mode |= M.modeSetLive;
          return true;
        }
        this.split(node, keyLow, keyHigh, level, leaf);
      } else if (level === this.maximumDepth) {
        const created = this.allocateLeaf();
        this.nodeLeaf[node] = created;
        this.leafNode[created] = node;
        this.leafKind[created] = SPARSE_BRICK_LEAF_TERMINAL.voxels;
        this.leafTerminal[created] = INVALID;
        this.leafFlags[created] = LEAF_RELEASABLE;
        this.touch(node, keyLow, keyHigh, level).mode |= M.modeSetLive;
        return true;
      }
      const octant = this.octant(coordinate, level + 1), mask = this.nodeMask[node]!;
      node = (mask & (1 << octant)) !== 0 ? this.nodeFirst[node]! + popcount8(mask & ((1 << octant) - 1))
        : this.relayout(node, keyLow, keyHigh, level, -1, octant);
      keyHigh = ((keyHigh << 3) | (keyLow >>> 29)) >>> 0;
      keyLow = ((keyLow << 3) | octant) >>> 0;
    }
  }

  /**
   * Take back a finest leaf an edit activated: its slot returns to the free
   * list, its node leaves its parent, and every node left with no child and no
   * leaf follows it. False where the brick is not such a leaf.
   */
  release(coordinate: SparseBrickCoordinate): boolean {
    const node = this.finestNode(coordinate, true);
    const leaf = node === INVALID ? INVALID : this.nodeLeaf[node]!;
    if (leaf === INVALID || (this.leafFlags[leaf]! & LEAF_RELEASABLE) === 0) return false;
    this.freeLeaf(leaf);
    this.nodeLeaf[node] = INVALID;
    this.topologyChanged = true;
    let level = this.maximumDepth;
    for (; level > 0; level -= 1) {
      const parent = this.pathSlot[level - 1]!;
      this.relayout(parent, this.pathKeyLow[level - 1]!, this.pathKeyHigh[level - 1]!, level - 1,
        this.octant(coordinate, level), -1);
      if (this.nodeMask[parent] !== 0 || this.nodePlanMask[parent] !== 0 || this.nodeLeaf[parent] !== INVALID) break;
    }
    // The root of a tree planned empty goes with its last brick: its record
    // still goes out, and the node count it goes out with publishes no node.
    if (level === 0 && this.plannedNodes === 0 && this.nodeMask[0] === 0) this.rootLive = false;
    return true;
  }

  /**
   * The batch that takes the GPU's tree to this one: a header, then one record
   * for each node whose words changed and one for each leaf slot left empty
   * below the leaf count.
   */
  pack(generation: number): Uint32Array<ArrayBuffer> {
    if (!Number.isInteger(generation) || generation < 0 || generation > 0xffff_ffff) {
      throw new RangeError("Topology mutation generation must fit uint32");
    }
    const retired: number[] = [];
    for (const leaf of this.freedLeaves) if (leaf < this.leaves.high && this.leafNode[leaf] === INVALID) retired.push(leaf);
    const count = this.pending.size + retired.length;
    if (count > this.recordCapacity) {
      throw new RangeError(`Topology mutation needs ${count} records and one batch carries ${this.recordCapacity}`);
    }
    const words = new Uint32Array(M.headerWords + count * M.recordWords);
    words[M.countWord] = count;
    words[M.generationWord] = generation >>> 0;
    words[M.capacityWord] = this.recordCapacity;
    words[M.nodeCountWord] = this.nodeCount;
    words[M.leafCountWord] = this.leaves.high;
    words[M.activeLeavesWord] = this.residentLeaves;
    let base = M.headerWords;
    for (const [node, entry] of this.pending) {
      const leaf = this.nodeLeaf[node]!;
      words[base + M.recordDestinationWord] = node;
      words[base + M.recordSourceWord] = entry.source;
      words[base + M.recordKeyLowWord] = entry.keyLow;
      words[base + M.recordKeyHighWord] = entry.keyHigh;
      words[base + M.recordFirstChildWord] = this.nodeFirst[node]!;
      words[base + M.recordLeafWord] = leaf;
      words[base + M.recordPackedWord] = entry.level | (this.nodeMask[node]! << M.childMaskShift)
        | ((leaf === INVALID ? 0 : this.leafKind[leaf]!) << M.terminalKindShift) | (entry.mode << M.modeShift);
      words[base + M.recordTerminalWord] = leaf === INVALID ? INVALID : this.leafTerminal[leaf]!;
      base += M.recordWords;
    }
    for (const leaf of retired) {
      words[base + M.recordDestinationWord] = INVALID;
      words[base + M.recordSourceWord] = INVALID;
      words[base + M.recordFirstChildWord] = INVALID;
      words[base + M.recordLeafWord] = leaf;
      words[base + M.recordTerminalWord] = INVALID;
      base += M.recordWords;
    }
    return words;
  }

  /** The packed batch was encoded: the GPU's tree is this one. */
  commit(): void {
    this.pending.clear();
    this.freedLeaves.clear();
    this.topologyChanged = false;
  }
}

export const webgpuSparseBrickTopologyMutationWGSL = /* wgsl */ `
// Control and topology are one physical allocation and one shader resource.
// Topology indices remain relative to the legacy topology slice; the accessors
// translate them into structural-arena words at the ownership boundary.
@group(0) @binding(0) var<storage,read_write> structure:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> requests:array<atomic<u32>>;

struct Params{limits:vec4u}
@group(0) @binding(2) var<uniform> params:Params;

const INVALID:u32=${SPARSE_BRICK_INVALID_INDEX}u;
const ACTIVE:u32=${SVO_BRICK_LIFECYCLE.activeBit}u;
const DIRTY:u32=${SVO_BRICK_LIFECYCLE.dirtyBit}u;
const QUEUED:u32=${SVO_BRICK_LIFECYCLE.queuedBit}u;
const LIFECYCLE_KEPT:u32=${SVO_BRICK_LIFECYCLE.mask & ~SVO_BRICK_LIFECYCLE.relocatingBit}u;
const OCCUPANCY_MASK:u32=${SVO_BRICK_OCCUPANCY.metadataMask}u;
// Bits of a node's child-mask word above the mask: a leaf's fitted contour.
const CONTOUR_MASK:u32=0xffffff00u;
const HEADER:u32=${M.headerWords}u;
const RECORD_WORDS:u32=${M.recordWords}u;
const KEEP_OCCUPANCY:u32=${M.modeKeepOccupancy}u;
const KEEP_LIFECYCLE:u32=${M.modeKeepLifecycle}u;
const SET_LIVE:u32=${M.modeSetLive}u;
const MALFORMED:u32=${M.overflowMalformedRecord}u;
const BUDGET:u32=${M.overflowRequestBudget}u;
const TOPOLOGY_BASE:u32=${SPARSE_BRICK_GPU_LAYOUT.topologyOffsetBytes / Uint32Array.BYTES_PER_ELEMENT}u;

fn loadControl(word:u32)->u32{return atomicLoad(&structure[word]);}
fn loadRequest(word:u32)->u32{return atomicLoad(&requests[word]);}
fn storeNode(node:u32,word:u32,value:u32){atomicStore(&structure[TOPOLOGY_BASE+node*8u+word],value);}
// A batch past the worklist is refused whole: the host's tree already holds
// every record of it, so applying a prefix would leave the two apart.
fn recordBudget()->u32{
  let available=select(0u,(arrayLength(&requests)-HEADER)/RECORD_WORDS,arrayLength(&requests)>=HEADER);
  return min(loadRequest(${M.capacityWord}u),min(available,params.limits.x));
}

// Every record reads the flags it carries before any record is written, so a
// record may write a slot another record reads: children shift within their
// block, and a freed slot is reused, inside one batch.
@compute @workgroup_size(64)
fn gatherTopologyFlags(@builtin(global_invocation_id) gid:vec3u){
  let count=loadRequest(${M.countWord}u);
  if(count>recordBudget()){
    if(gid.x==0u){atomicOr(&requests[${M.overflowFlagsWord}u],BUDGET);}
    return;
  }
  if(gid.x>=count){return;}
  let base=HEADER+gid.x*RECORD_WORDS;
  let destination=loadRequest(base+${M.recordDestinationWord}u);
  let source=loadRequest(base+${M.recordSourceWord}u);
  let leaf=loadRequest(base+${M.recordLeafWord}u);
  let nodeCapacity=loadControl(8u);
  if((destination!=INVALID&&destination>=nodeCapacity)||(source!=INVALID&&source>=nodeCapacity)
      ||(leaf!=INVALID&&leaf>=loadControl(9u))||(destination==INVALID&&leaf==INVALID)){
    atomicOr(&requests[${M.overflowFlagsWord}u],MALFORMED);
    return;
  }
  let mode=loadRequest(base+${M.recordPackedWord}u)>>${M.modeShift}u;
  var flags=0u;
  if(source!=INVALID){
    let carried=atomicLoad(&structure[TOPOLOGY_BASE+source*8u+7u]);
    if((mode&KEEP_OCCUPANCY)!=0u){flags|=carried&OCCUPANCY_MASK;}
    if((mode&KEEP_LIFECYCLE)!=0u){flags|=carried&LIFECYCLE_KEPT;}
  }
  if((mode&SET_LIVE)!=0u){flags|=ACTIVE|DIRTY|QUEUED;}
  atomicStore(&requests[base+${M.recordSourceWord}u],flags);
  // A leaf that keeps its voxels keeps the contour fitted to them, which rides
  // above its node's child mask. A node with a leaf has no children, so the
  // contour travels where the record's first child would.
  if(destination!=INVALID&&leaf!=INVALID){
    var contour=0u;
    if(source!=INVALID&&(mode&KEEP_LIFECYCLE)!=0u&&(mode&SET_LIVE)==0u){
      contour=atomicLoad(&structure[TOPOLOGY_BASE+source*8u+3u])&CONTOUR_MASK;
    }
    atomicStore(&requests[base+${M.recordFirstChildWord}u],contour);
  }
}

// One thread a record, and no two records name the same node or the same leaf.
@compute @workgroup_size(64)
fn scatterTopology(@builtin(global_invocation_id) gid:vec3u){
  if(loadRequest(${M.overflowFlagsWord}u)!=0u){return;}
  let count=loadRequest(${M.countWord}u);
  let voxelsPerBrick=loadControl(11u)*loadControl(11u)*loadControl(11u);
  if(gid.x==0u){
    let nodes=loadRequest(${M.nodeCountWord}u);let leaves=loadRequest(${M.leafCountWord}u);
    atomicStore(&structure[19],nodes);atomicStore(&structure[0],nodes);
    atomicStore(&structure[23],leaves);atomicStore(&structure[1],leaves);
    atomicStore(&structure[2],leaves*voxelsPerBrick);
    atomicStore(&structure[28],loadRequest(${M.activeLeavesWord}u));
    atomicStore(&structure[31],params.limits.y);atomicStore(&structure[3],params.limits.y);
  }
  if(gid.x>=count){return;}
  let base=HEADER+gid.x*RECORD_WORDS;
  let destination=loadRequest(base+${M.recordDestinationWord}u);
  let leaf=loadRequest(base+${M.recordLeafWord}u);
  let packed=loadRequest(base+${M.recordPackedWord}u);
  if(destination!=INVALID){
    let mask=(packed>>${M.childMaskShift}u)&0xffu;
    storeNode(destination,0u,loadRequest(base+${M.recordKeyLowWord}u));
    storeNode(destination,1u,loadRequest(base+${M.recordKeyHighWord}u));
    storeNode(destination,2u,packed&${M.levelMask}u);
    let first=loadRequest(base+${M.recordFirstChildWord}u);
    storeNode(destination,3u,select(mask,mask|first,leaf!=INVALID));
    storeNode(destination,4u,select(first,INVALID,leaf!=INVALID));
    storeNode(destination,5u,countOneBits(mask));
    storeNode(destination,6u,leaf);
    storeNode(destination,7u,loadRequest(base+${M.recordSourceWord}u));
  }
  if(leaf!=INVALID){
    // A record with no node retires the slot: no node is below the node count.
    let leafBase=TOPOLOGY_BASE+loadControl(16u)+leaf*4u;
    atomicStore(&structure[leafBase],destination);
    atomicStore(&structure[leafBase+1u],leaf*voxelsPerBrick);
    atomicStore(&structure[leafBase+2u],(packed>>${M.terminalKindShift}u)&3u);
    atomicStore(&structure[leafBase+3u],loadRequest(base+${M.recordTerminalWord}u));
  }
}
`;

export class WebGpuSparseBrickTopologyMutator {
  readonly allocatedBytes = 16;
  private gather!: GPUComputePipeline;
  private scatter!: GPUComputePipeline;
  private readonly layout: GPUBindGroupLayout;
  private readonly params: GPUBuffer;
  private readonly paramWords = new Uint32Array(4);
  private readonly label: string;
  private bindGroup?: GPUBindGroup;
  private boundStructure?: GPUBuffer;
  private boundWorklist?: GPUBuffer;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, label = "Sparse brick topology mutation") {
    this.label = label;
    this.params = device.createBuffer({
      label: `${label} parameters`,
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.layout = device.createBindGroupLayout({
      label: `${label} layout`,
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
  }

  async initializePipelines(): Promise<void> {
    if (this.gather) return;
    const module = this.device.createShaderModule({
      label: `${this.label} shader`, code: webgpuSparseBrickTopologyMutationWGSL,
    });
    const layout = this.device.createPipelineLayout({ label: `${this.label} pipeline layout`, bindGroupLayouts: [this.layout] });
    [this.gather, this.scatter] = await Promise.all(["gatherTopologyFlags", "scatterTopology"].map(entryPoint =>
      this.device.createComputePipelineAsync({
        label: `${this.label} ${entryPoint} pipeline`, layout, compute: { module, entryPoint },
      })));
  }

  /**
   * Bind the tree and the worklist every batch is scattered between. Once, at
   * build: neither buffer is replaced while the scene lives, and an edit
   * creates nothing.
   */
  bind(tree: SparseBrickOctreeGPU, worklist: SparseBrickTopologyMutationWorklist): void {
    if (this.destroyed) throw new Error("Sparse brick topology mutator is destroyed");
    this.bindGroup = this.device.createBindGroup({
      label: `${this.label} bindings`,
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: tree.structure } },
        { binding: 1, resource: { buffer: worklist.buffer } },
        { binding: 2, resource: { buffer: this.params } },
      ],
    });
    this.boundStructure = tree.structure;
    this.boundWorklist = worklist.buffer;
  }

  /** Two fixed dispatches over the staged records, whatever the tree's depth. */
  encode(
    encoder: GPUCommandEncoder,
    tree: SparseBrickOctreeGPU,
    worklist: SparseBrickTopologyMutationWorklist,
    options: SparseBrickTopologyMutationOptions,
  ): void {
    if (this.destroyed) throw new Error("Sparse brick topology mutator is destroyed");
    if (!Number.isSafeInteger(worklist.capacity) || worklist.capacity < 0 || worklist.capacity > 0xffff_ffff) {
      throw new RangeError("Topology mutation worklist capacity must fit uint32");
    }
    if (!Number.isInteger(options.generation) || options.generation < 0 || options.generation > 0xffff_ffff) {
      throw new RangeError("Topology mutation generation must fit uint32");
    }
    if (!Number.isSafeInteger(options.recordCount) || options.recordCount < 1 || options.recordCount > worklist.capacity) {
      throw new RangeError("Topology mutation record count must be within the worklist");
    }
    const groups = Math.ceil(options.recordCount / 64);
    if (groups > this.device.limits.maxComputeWorkgroupsPerDimension) {
      throw new RangeError("Topology mutation batch exceeds one dispatch");
    }
    this.paramWords[0] = worklist.capacity;
    this.paramWords[1] = options.generation;
    this.device.queue.writeBuffer(this.params, 0, this.paramWords);
    if (!this.bindGroup || this.boundStructure !== tree.structure || this.boundWorklist !== worklist.buffer) {
      throw new Error("Sparse brick topology mutator is not bound to this tree and worklist");
    }
    for (const [pipeline, label] of [[this.gather, "Gather sparse brick topology flags"],
      [this.scatter, "Scatter sparse brick topology"]] as const) {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.dispatchWorkgroups(groups);
      pass.end();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.params.destroy();
  }
}
