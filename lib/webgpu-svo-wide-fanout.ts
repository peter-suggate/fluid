import {
  SVO_WIDE_INVALID_INDEX,
  SVO_WIDE_MICRO_MIP_WORDS,
  type SvoWideFanoutPlan,
  type SvoWideOpacity,
} from "./svo-wide-fanout";

export const SVO_WIDE_PUBLICATION_STAGES = Object.freeze({
  directory: 1 << 0,
  descriptors: 1 << 1,
  microMips: 1 << 2,
} as const);

export const SVO_WIDE_GPU_LAYOUT = Object.freeze({
  pageStrideBytes: 32,
  descriptorStrideBytes: 16,
  controlStrideBytes: 64,
  microMipStrideBytes: SVO_WIDE_MICRO_MIP_WORDS * 4,
  pageWords: 8,
  descriptorWords: 4,
  microMipWords: SVO_WIDE_MICRO_MIP_WORDS,
  descriptorKinds: { terminal: 1, page: 2 } as const,
  controlWords: {
    publishedPages: 0,
    publishedDescriptors: 1,
    generation: 2,
    sourceGeneration: 3,
    overflowFlags: 4,
    requiredStages: 5,
    completedStages: 6,
    payloadWritesComplete: 7,
    maximumDepth: 8,
    microMipWords: 9,
  } as const,
} as const);

export interface PackedSvoWideFanout {
  pages: Uint32Array<ArrayBuffer>;
  descriptors: Uint32Array<ArrayBuffer>;
  microMips: Uint32Array<ArrayBuffer>;
  control: Uint32Array<ArrayBuffer>;
}

export interface SvoWidePackedPublicationView extends PackedSvoWideFanout {
  expectedSourceGeneration?: number;
}

export interface WebGPUSvoWideFanoutCapacity {
  maximumPages: number;
  maximumDescriptors: number;
}

export interface WebGPUSvoWideFanoutSource {
  readonly control: GPUBufferBinding;
  readonly pages: GPUBufferBinding;
  readonly descriptors: GPUBufferBinding;
  readonly microMips: GPUBufferBinding;
  readonly generation: number;
  readonly sourceGeneration: number;
  readonly pageCount: number;
  readonly descriptorCount: number;
  readonly maximumDepth: number;
}

export interface SvoWideTraversalPublication {
  readonly generation: number;
  readonly sourceGeneration: number;
  readonly pageCount: number;
  readonly descriptorCount: number;
}

export type SvoWideTraversalCapabilityResolution =
  | { status: "ready"; publication: SvoWideTraversalPublication; source: WebGPUSvoWideFanoutSource }
  | { status: "missing" | "source-stale" | "invalid"; reason: string };

export interface WebGPUSvoWideFanoutAllocation extends WebGPUSvoWideFanoutCapacity {
  controlBytes: number;
  pageBytes: number;
  descriptorBytes: number;
  microMipBytes: number;
  allocatedBytes: number;
}

export type SvoWidePublicationValidation =
  | { status: "ready"; generation: number; sourceGeneration: number; pageCount: number; descriptorCount: number }
  | { status: "source-stale" | "incomplete" | "overflow" | "invalid"; reason: string };

export function planWebgpuSvoWideFanoutAllocation(capacity: WebGPUSvoWideFanoutCapacity): WebGPUSvoWideFanoutAllocation {
  if (!Number.isSafeInteger(capacity.maximumPages) || capacity.maximumPages < 1
      || !Number.isSafeInteger(capacity.maximumDescriptors) || capacity.maximumDescriptors < 1) {
    throw new RangeError("Wide-fanout capacities must be positive safe integers");
  }
  const controlBytes = SVO_WIDE_GPU_LAYOUT.controlStrideBytes;
  const pageBytes = capacity.maximumPages * SVO_WIDE_GPU_LAYOUT.pageStrideBytes;
  const descriptorBytes = capacity.maximumDescriptors * SVO_WIDE_GPU_LAYOUT.descriptorStrideBytes;
  const microMipBytes = capacity.maximumPages * SVO_WIDE_GPU_LAYOUT.microMipStrideBytes;
  if (![pageBytes, descriptorBytes, microMipBytes].every(Number.isSafeInteger)) throw new RangeError("Wide-fanout allocation exceeds safe byte accounting");
  return { ...capacity, controlBytes, pageBytes, descriptorBytes, microMipBytes,
    allocatedBytes: controlBytes + pageBytes + descriptorBytes + microMipBytes };
}

export type WebGPUSvoWideFanoutEncodeStatus = "encoded" | "unchanged" | "capacity-exhausted" | "invalid" | "destroyed";

function unorm8(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value * 255)));
}

export function packSvoWideOpacity(value: SvoWideOpacity): number {
  return (unorm8(value.solidMean) | (unorm8(value.solidMaximum) << 8)
    | (unorm8(value.fluidMean) << 16) | (unorm8(value.fluidMaximum) << 24)) >>> 0;
}

export function unpackSvoWideOpacity(word: number): SvoWideOpacity {
  return {
    solidMean: (word & 0xff) / 255,
    solidMaximum: ((word >>> 8) & 0xff) / 255,
    fluidMean: ((word >>> 16) & 0xff) / 255,
    fluidMaximum: ((word >>> 24) & 0xff) / 255,
  };
}

function splitMorton(value: bigint): readonly [number, number] {
  return [Number(value & 0xffff_ffffn) >>> 0, Number((value >> 32n) & 0xffff_ffffn) >>> 0];
}

/** Pack a complete CPU plan into fixed-stride GPU ABI arrays. */
export function packSvoWideFanout(plan: SvoWideFanoutPlan): PackedSvoWideFanout {
  const pages = new Uint32Array(plan.pages.length * SVO_WIDE_GPU_LAYOUT.pageWords);
  const descriptors = new Uint32Array(plan.descriptorCount * SVO_WIDE_GPU_LAYOUT.descriptorWords);
  const microMips = new Uint32Array(plan.pages.length * SVO_WIDE_GPU_LAYOUT.microMipWords);
  let descriptorIndex = 0;
  for (const page of plan.pages) {
    const [mortonLow, mortonHigh] = splitMorton(page.morton);
    pages.set([mortonLow, mortonHigh, page.level, page.occupancyLow, page.occupancyHigh,
      descriptorIndex, page.descriptors.length, 0], page.index * SVO_WIDE_GPU_LAYOUT.pageWords);
    for (const descriptor of page.descriptors) {
      const kind = SVO_WIDE_GPU_LAYOUT.descriptorKinds[descriptor.kind];
      const meta = (kind | (descriptor.slot << 2) | (descriptor.sourceLevel << 8)) >>> 0;
      descriptors.set([meta, descriptor.kind === "page" ? descriptor.pageIndex : descriptor.sourceNodeIndex,
        descriptor.kind === "page" ? SVO_WIDE_INVALID_INDEX : descriptor.sourceLeafIndex,
        packSvoWideOpacity(descriptor.opacity)], descriptorIndex * SVO_WIDE_GPU_LAYOUT.descriptorWords);
      descriptorIndex += 1;
    }
    for (let word = 0; word < page.microMips.length; word += 1) {
      microMips[page.index * SVO_WIDE_GPU_LAYOUT.microMipWords + word] = packSvoWideOpacity(page.microMips[word]);
    }
  }
  const required = Object.values(SVO_WIDE_PUBLICATION_STAGES).reduce((mask, stage) => mask | stage, 0);
  const control = new Uint32Array(SVO_WIDE_GPU_LAYOUT.controlStrideBytes / 4);
  control.set([plan.pages.length, plan.descriptorCount, plan.generation, plan.sourceGeneration, 0,
    required, required, 1, plan.maximumDepth, SVO_WIDE_GPU_LAYOUT.microMipWords]);
  return { pages, descriptors, microMips, control };
}

/**
 * Single-publication GPU owner for a host-planned static structural topology.
 * `encode` deliberately accepts an encoder for stage-chain symmetry, but the
 * immutable upload uses queue writes and publishes the control record last.
 */
export class WebGPUSvoWideFanout {
  readonly allocation: WebGPUSvoWideFanoutAllocation;
  readonly allocatedBytes: number;
  private readonly controlBuffer: GPUBuffer;
  private readonly pageBuffer: GPUBuffer;
  private readonly descriptorBuffer: GPUBuffer;
  private readonly microMipBuffer: GPUBuffer;
  private published?: WebGPUSvoWideFanoutSource;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, capacity: WebGPUSvoWideFanoutCapacity) {
    this.allocation = planWebgpuSvoWideFanoutAllocation(capacity);
    this.allocatedBytes = this.allocation.allocatedBytes;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.controlBuffer = device.createBuffer({ label: "SVO wide-fanout control", size: this.allocation.controlBytes, usage: storage });
    this.pageBuffer = device.createBuffer({ label: "SVO wide-fanout pages", size: this.allocation.pageBytes, usage: storage });
    this.descriptorBuffer = device.createBuffer({ label: "SVO wide-fanout descriptors", size: this.allocation.descriptorBytes, usage: storage });
    this.microMipBuffer = device.createBuffer({ label: "SVO wide-fanout opacity micro-mips", size: this.allocation.microMipBytes, usage: storage });
  }

  encode(
    _encoder: GPUCommandEncoder,
    plan: SvoWideFanoutPlan,
    canonical?: SvoWideCanonicalTopologyView,
  ): WebGPUSvoWideFanoutEncodeStatus {
    if (this.destroyed) return "destroyed";
    if (this.published) {
      if (this.published.generation !== plan.generation || this.published.sourceGeneration !== plan.sourceGeneration) {
        throw new RangeError("Static wide-fanout owner cannot be republished for a different generation");
      }
      return "unchanged";
    }
    if (plan.pages.length > this.allocation.maximumPages || plan.descriptorCount > this.allocation.maximumDescriptors) {
      const control = new Uint32Array(SVO_WIDE_GPU_LAYOUT.controlStrideBytes / 4);
      control[SVO_WIDE_GPU_LAYOUT.controlWords.generation] = plan.generation;
      control[SVO_WIDE_GPU_LAYOUT.controlWords.sourceGeneration] = plan.sourceGeneration;
      control[SVO_WIDE_GPU_LAYOUT.controlWords.overflowFlags] = 1;
      this.device.queue.writeBuffer(this.controlBuffer, 0, control);
      return "capacity-exhausted";
    }
    const packed = packSvoWideFanout(plan);
    // The publication is immutable, so every semantic invariant the traversal
    // relies on is proven exactly once here. Failure publishes nothing: the
    // renderer sees no capability and stays on canonical traversal.
    if (validateSvoWidePackedPlan(packed, plan, canonical).status !== "ready") return "invalid";
    if (packed.pages.byteLength > 0) this.device.queue.writeBuffer(this.pageBuffer, 0, packed.pages);
    if (packed.descriptors.byteLength > 0) this.device.queue.writeBuffer(this.descriptorBuffer, 0, packed.descriptors);
    if (packed.microMips.byteLength > 0) this.device.queue.writeBuffer(this.microMipBuffer, 0, packed.microMips);
    // Queue order is the publication fence: payloads become visible before this complete control record.
    this.device.queue.writeBuffer(this.controlBuffer, 0, packed.control);
    this.published = {
      control: { buffer: this.controlBuffer, offset: 0, size: this.allocation.controlBytes },
      pages: { buffer: this.pageBuffer, offset: 0, size: this.allocation.pageBytes },
      descriptors: { buffer: this.descriptorBuffer, offset: 0, size: this.allocation.descriptorBytes },
      microMips: { buffer: this.microMipBuffer, offset: 0, size: this.allocation.microMipBytes },
      generation: plan.generation, sourceGeneration: plan.sourceGeneration,
      pageCount: plan.pages.length, descriptorCount: plan.descriptorCount, maximumDepth: plan.maximumDepth,
    };
    return "encoded";
  }

  capability(): WebGPUSvoWideFanoutSource | undefined {
    return this.destroyed ? undefined : this.published;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.published = undefined;
    this.controlBuffer.destroy();
    this.pageBuffer.destroy();
    this.descriptorBuffer.destroy();
    this.microMipBuffer.destroy();
  }
}

function bindingByteLength(binding: GPUBufferBinding): number {
  const offset = binding.offset ?? 0;
  const size = binding.size ?? binding.buffer.size - offset;
  if (!Number.isSafeInteger(offset) || offset < 0 || (offset & 3) !== 0
      || !Number.isSafeInteger(size) || size < 0 || (size & 3) !== 0
      || offset + size > binding.buffer.size) return -1;
  return size;
}

/**
 * Fail-closed renderer gate for the optional hierarchy. The canonical source
 * remains authoritative; a missing, stale, or structurally impossible view is
 * represented by no capability and must select canonical traversal.
 */
export function resolveSvoWideTraversalCapability(
  source: WebGPUSvoWideFanoutSource | undefined,
  expectedSourceGeneration: number,
  expectedMaximumDepth: number,
): SvoWideTraversalCapabilityResolution {
  if (!source) return { status: "missing", reason: "Wide-fanout capability is absent" };
  if (!Number.isInteger(expectedSourceGeneration) || expectedSourceGeneration < 1 || expectedSourceGeneration > 0xffff_ffff
      || !Number.isInteger(expectedMaximumDepth) || expectedMaximumDepth < 0 || expectedMaximumDepth > 21) {
    return { status: "invalid", reason: "Canonical wide-fanout expectations are invalid" };
  }
  if (source.sourceGeneration !== (expectedSourceGeneration >>> 0)) {
    return { status: "source-stale", reason: "Wide-fanout source generation is stale" };
  }
  const uint32 = (value: number) => Number.isInteger(value) && value >= 0 && value <= 0xffff_ffff;
  if (!uint32(source.generation) || source.generation === 0 || !uint32(source.sourceGeneration)
      || !uint32(source.pageCount) || !uint32(source.descriptorCount)
      || !Number.isInteger(source.maximumDepth) || source.maximumDepth !== expectedMaximumDepth) {
    return { status: "invalid", reason: "Wide-fanout capability metadata is invalid" };
  }
  const pagesBytes = source.pageCount * SVO_WIDE_GPU_LAYOUT.pageStrideBytes;
  const descriptorBytes = source.descriptorCount * SVO_WIDE_GPU_LAYOUT.descriptorStrideBytes;
  const microMipBytes = source.pageCount * SVO_WIDE_GPU_LAYOUT.microMipStrideBytes;
  if (source.pageCount === 0 && source.descriptorCount !== 0
      || bindingByteLength(source.control) < SVO_WIDE_GPU_LAYOUT.controlStrideBytes
      || bindingByteLength(source.pages) < pagesBytes
      || bindingByteLength(source.descriptors) < descriptorBytes
      || bindingByteLength(source.microMips) < microMipBytes) {
    return { status: "invalid", reason: "Wide-fanout capability exceeds its published binding bounds" };
  }
  return {
    status: "ready",
    publication: {
      generation: source.generation,
      sourceGeneration: source.sourceGeneration,
      pageCount: source.pageCount,
      descriptorCount: source.descriptorCount,
    },
    source,
  };
}

function popcount(value: number): number {
  let count = 0;
  for (let bits = value >>> 0; bits !== 0; bits >>>= 1) count += bits & 1;
  return count;
}

/** Validate every publication bound and relationship before the generation can be sampled. */
export function validateSvoWideFanoutPublication(view: SvoWidePackedPublicationView): SvoWidePublicationValidation {
  const words = SVO_WIDE_GPU_LAYOUT.controlWords;
  if (view.control.length < SVO_WIDE_GPU_LAYOUT.controlStrideBytes / 4) return { status: "invalid", reason: "Control record is truncated" };
  const pageCount = view.control[words.publishedPages];
  const descriptorCount = view.control[words.publishedDescriptors];
  const generation = view.control[words.generation];
  const sourceGeneration = view.control[words.sourceGeneration];
  if (view.control[words.overflowFlags] !== 0) return { status: "overflow", reason: "Wide publication overflowed capacity" };
  if (view.expectedSourceGeneration !== undefined && sourceGeneration !== (view.expectedSourceGeneration >>> 0)) {
    return { status: "source-stale", reason: "Wide publication does not match the visible source generation" };
  }
  const required = view.control[words.requiredStages];
  const completed = view.control[words.completedStages];
  if ((completed & required) !== required || view.control[words.payloadWritesComplete] === 0) {
    return { status: "incomplete", reason: "Wide publication stages or payload writes are incomplete" };
  }
  if (view.control[words.microMipWords] !== SVO_WIDE_GPU_LAYOUT.microMipWords
      || view.control[words.maximumDepth] > 21
      || view.pages.length % SVO_WIDE_GPU_LAYOUT.pageWords !== 0
      || view.descriptors.length % SVO_WIDE_GPU_LAYOUT.descriptorWords !== 0
      || pageCount > view.pages.length / SVO_WIDE_GPU_LAYOUT.pageWords
      || descriptorCount > view.descriptors.length / SVO_WIDE_GPU_LAYOUT.descriptorWords
      || view.microMips.length < pageCount * SVO_WIDE_GPU_LAYOUT.microMipWords) {
    return { status: "invalid", reason: "Published wide counts exceed ABI storage bounds" };
  }
  if (pageCount === 0 && descriptorCount !== 0) return { status: "invalid", reason: "Empty directory publishes descriptors" };
  for (let pageIndex = 0; pageIndex < pageCount; pageIndex += 1) {
    const base = pageIndex * SVO_WIDE_GPU_LAYOUT.pageWords;
    const level = view.pages[base + 2];
    const maskLow = view.pages[base + 3];
    const maskHigh = view.pages[base + 4];
    const first = view.pages[base + 5];
    const count = view.pages[base + 6];
    if ((level & 1) !== 0 || level > view.control[words.maximumDepth] || first + count > descriptorCount
        || count !== popcount(maskLow) + popcount(maskHigh)) {
      return { status: "invalid", reason: "Wide page header is inconsistent" };
    }
    for (let local = 0; local < count; local += 1) {
      const descriptorBase = (first + local) * SVO_WIDE_GPU_LAYOUT.descriptorWords;
      const meta = view.descriptors[descriptorBase];
      const kind = meta & 3;
      const slot = (meta >>> 2) & 63;
      const sourceLevel = (meta >>> 8) & 0xff;
      const occupied = slot < 32 ? ((maskLow >>> slot) & 1) : ((maskHigh >>> (slot - 32)) & 1);
      if (occupied === 0 || (kind !== 1 && kind !== 2)) return { status: "invalid", reason: "Wide descriptor is absent from its page mask" };
      if (kind === 2) {
        const childIndex = view.descriptors[descriptorBase + 1];
        if (childIndex >= pageCount || childIndex <= pageIndex || sourceLevel !== level + 2) {
          return { status: "invalid", reason: "Wide child descriptor is outside the directory hierarchy" };
        }
      } else if (sourceLevel < level || sourceLevel > level + 2 || sourceLevel > view.control[words.maximumDepth]) {
        return { status: "invalid", reason: "Wide terminal level is outside its page" };
      }
    }
  }
  return { status: "ready", generation, sourceGeneration, pageCount, descriptorCount };
}

/** Canonical packed-topology arrays used for publish-time terminal cross-checks. */
export interface SvoWideCanonicalTopologyView {
  /** 8 words per node: mortonLow, mortonHigh, level, childMask, firstChild, childCount, leafIndex, reserved. */
  nodes: Uint32Array;
  /** 4 words per leaf: nodeIndex, voxelOffset, localIndex, reserved. */
  leaves: Uint32Array;
  nodeCount?: number;
  leafCount?: number;
}

const CANONICAL_NODE_WORDS = 8;
const CANONICAL_LEAF_WORDS = 4;

function mortonWordsAtLevel(coordinate: readonly [number, number, number], level: number): readonly [number, number] {
  let morton = 0n;
  for (let bit = 0; bit < level; bit += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      morton |= BigInt((coordinate[axis] >> bit) & 1) << BigInt(3 * bit + axis);
    }
  }
  return splitMorton(morton);
}

/**
 * Prove at publish time everything the traversal loop previously re-proved per
 * ray: the packed arrays are a faithful encoding of the plan, page/descriptor
 * hierarchy links agree with the plan geometry, and (when a canonical view is
 * supplied) every terminal descriptor agrees with the canonical node/leaf
 * records it references. A publication that fails here must never be sampled;
 * the GPU hot loop keeps only memory-safety clamps.
 */
export function validateSvoWidePackedPlan(
  packed: PackedSvoWideFanout,
  plan: SvoWideFanoutPlan,
  canonical?: SvoWideCanonicalTopologyView,
): SvoWidePublicationValidation {
  const invalid = (reason: string): SvoWidePublicationValidation => ({ status: "invalid", reason });
  const base = validateSvoWideFanoutPublication({ ...packed, expectedSourceGeneration: plan.sourceGeneration });
  if (base.status !== "ready") return base;
  if (base.generation !== plan.generation || base.pageCount !== plan.pages.length
      || base.descriptorCount !== plan.descriptorCount
      || packed.control[SVO_WIDE_GPU_LAYOUT.controlWords.maximumDepth] !== plan.maximumDepth) {
    return invalid("Packed wide control record does not match its plan");
  }
  if (plan.pages.length > 0 && (plan.pages[0].level !== 0 || plan.pages[0].morton !== 0n)) {
    return invalid("Wide directory root must anchor the canonical origin");
  }
  let nodeCount = 0;
  let leafCount = 0;
  if (canonical) {
    nodeCount = canonical.nodeCount ?? Math.floor(canonical.nodes.length / CANONICAL_NODE_WORDS);
    leafCount = canonical.leafCount ?? Math.floor(canonical.leaves.length / CANONICAL_LEAF_WORDS);
    if (!Number.isInteger(nodeCount) || nodeCount < 0 || nodeCount * CANONICAL_NODE_WORDS > canonical.nodes.length
        || !Number.isInteger(leafCount) || leafCount < 0 || leafCount * CANONICAL_LEAF_WORDS > canonical.leaves.length) {
      return invalid("Canonical cross-check view bounds are inconsistent");
    }
  }
  let descriptorIndex = 0;
  for (const page of plan.pages) {
    const pageBase = page.index * SVO_WIDE_GPU_LAYOUT.pageWords;
    const [mortonLow, mortonHigh] = splitMorton(page.morton);
    const [expectedLow, expectedHigh] = mortonWordsAtLevel(
      [page.coordinate[0], page.coordinate[1], page.coordinate[2]], page.level);
    if (mortonLow !== expectedLow || mortonHigh !== expectedHigh) {
      return invalid("Wide page Morton key does not match its coordinate");
    }
    if (packed.pages[pageBase] !== mortonLow || packed.pages[pageBase + 1] !== mortonHigh
        || packed.pages[pageBase + 2] !== page.level
        || packed.pages[pageBase + 3] !== page.occupancyLow || packed.pages[pageBase + 4] !== page.occupancyHigh
        || packed.pages[pageBase + 5] !== descriptorIndex || packed.pages[pageBase + 6] !== page.descriptors.length) {
      return invalid("Packed wide page header does not match its plan");
    }
    for (const descriptor of page.descriptors) {
      const words = packed.descriptors.subarray(
        descriptorIndex * SVO_WIDE_GPU_LAYOUT.descriptorWords, (descriptorIndex + 1) * SVO_WIDE_GPU_LAYOUT.descriptorWords);
      descriptorIndex += 1;
      const kind = SVO_WIDE_GPU_LAYOUT.descriptorKinds[descriptor.kind];
      const meta = (kind | (descriptor.slot << 2) | (descriptor.sourceLevel << 8)) >>> 0;
      const reference = descriptor.kind === "page" ? descriptor.pageIndex : descriptor.sourceNodeIndex;
      const sourceLeaf = descriptor.kind === "page" ? SVO_WIDE_INVALID_INDEX : descriptor.sourceLeafIndex;
      if (words[0] !== meta || words[1] !== reference || words[2] !== sourceLeaf
          || words[3] !== packSvoWideOpacity(descriptor.opacity)) {
        return invalid("Packed wide descriptor does not match its plan");
      }
      const slotCoordinate = [descriptor.slot & 3, (descriptor.slot >> 2) & 3, (descriptor.slot >> 4) & 3] as const;
      const globalCoordinate = [
        page.coordinate[0] * 4 + slotCoordinate[0],
        page.coordinate[1] * 4 + slotCoordinate[1],
        page.coordinate[2] * 4 + slotCoordinate[2],
      ] as const;
      if (descriptor.kind === "page") {
        const child = plan.pages[descriptor.pageIndex];
        if (!child || child.index !== descriptor.pageIndex || child.level !== page.level + 2
            || child.level !== descriptor.sourceLevel
            || child.coordinate[0] !== globalCoordinate[0] || child.coordinate[1] !== globalCoordinate[1]
            || child.coordinate[2] !== globalCoordinate[2]) {
          return invalid("Wide child page geometry does not match its parent slot");
        }
        continue;
      }
      const divisor = 2 ** (page.level + 2 - descriptor.sourceLevel);
      const terminalCoordinate = [
        Math.floor(globalCoordinate[0] / divisor),
        Math.floor(globalCoordinate[1] / divisor),
        Math.floor(globalCoordinate[2] / divisor),
      ] as const;
      if (!canonical) continue;
      if (descriptor.sourceNodeIndex >= nodeCount || descriptor.sourceLeafIndex >= leafCount) {
        return invalid("Wide terminal references are outside the canonical topology");
      }
      const nodeBase = descriptor.sourceNodeIndex * CANONICAL_NODE_WORDS;
      const leafBase = descriptor.sourceLeafIndex * CANONICAL_LEAF_WORDS;
      const [terminalLow, terminalHigh] = mortonWordsAtLevel(terminalCoordinate, descriptor.sourceLevel);
      if (canonical.nodes[nodeBase + 2] !== descriptor.sourceLevel
          || canonical.nodes[nodeBase + 6] !== descriptor.sourceLeafIndex
          || canonical.leaves[leafBase] !== descriptor.sourceNodeIndex
          || canonical.nodes[nodeBase] !== terminalLow || canonical.nodes[nodeBase + 1] !== terminalHigh) {
        return invalid("Wide terminal disagrees with its canonical node and leaf records");
      }
    }
  }
  if (descriptorIndex !== plan.descriptorCount) return invalid("Wide descriptor stream does not cover its plan");
  return base;
}

/**
 * Binding-free WGSL helpers. The consumer owns bind groups and buffer names;
 * this fragment only defines the shared ABI, rank/addressing, and mip decode.
 */
export const webgpuSvoWideFanoutHelpersWGSL = /* wgsl */`
const SVO_WIDE_KIND_TERMINAL: u32 = 1u;
const SVO_WIDE_KIND_PAGE: u32 = 2u;
const SVO_WIDE_MICRO_MIP_WORDS: u32 = 73u;

struct SvoWidePage {
  mortonLow: u32,
  mortonHigh: u32,
  level: u32,
  occupancyLow: u32,
  occupancyHigh: u32,
  firstDescriptor: u32,
  descriptorCount: u32,
  flags: u32,
};

struct SvoWideDescriptor {
  metadata: u32,
  reference: u32,
  sourceLeaf: u32,
  opacity: u32,
};

fn svoWideDescriptorKind(descriptor: SvoWideDescriptor) -> u32 { return descriptor.metadata & 3u; }
fn svoWideDescriptorSlot(descriptor: SvoWideDescriptor) -> u32 { return (descriptor.metadata >> 2u) & 63u; }
fn svoWideDescriptorSourceLevel(descriptor: SvoWideDescriptor) -> u32 { return (descriptor.metadata >> 8u) & 255u; }

fn svoWideSlotCoordinate(slot: u32) -> vec3u {
  return vec3u(slot & 3u, (slot >> 2u) & 3u, (slot >> 4u) & 3u);
}

fn svoWideOccupied(page: SvoWidePage, slot: u32) -> bool {
  if (slot < 32u) { return ((page.occupancyLow >> slot) & 1u) != 0u; }
  return ((page.occupancyHigh >> (slot - 32u)) & 1u) != 0u;
}

fn svoWideDescriptorRank(page: SvoWidePage, slot: u32) -> u32 {
  if (slot < 32u) {
    let before = select((1u << slot) - 1u, 0xffffffffu, slot == 32u);
    return countOneBits(page.occupancyLow & before);
  }
  let local = slot - 32u;
  let before = select((1u << local) - 1u, 0xffffffffu, local == 32u);
  return countOneBits(page.occupancyLow) + countOneBits(page.occupancyHigh & before);
}

fn svoWideOpacityChannels(packed: u32) -> vec4f {
  return vec4f(f32(packed & 255u), f32((packed >> 8u) & 255u),
    f32((packed >> 16u) & 255u), f32((packed >> 24u) & 255u)) / 255.0;
}

fn svoWideMicroMipOffset(lod: u32) -> u32 {
  return select(select(0u, 64u, lod == 1u), 72u, lod >= 2u);
}
`;

/**
 * Resumable near-to-far traversal over the optional 4^3 hierarchy.
 *
 * This fragment deliberately depends on the canonical traversal library for
 * SvoMapping, SvoRay, SvoTraversalHit, svoLeaves, status constants, Morton
 * decode, and ray/AABB intersection. A page-local DDA visits at most ten
 * cells for a non-degenerate ray; parent DDA frames survive both tail descent
 * and returned terminal leaves. Only the two traversal payloads are bound.
 *
 * The publication is immutable and semantically validated once at publish
 * (validateSvoWidePackedPlan inside WebGPUSvoWideFanout.encode), so the hot
 * loop keeps only memory-safety clamps: index bounds, stack capacity, work
 * budgets, and the degenerate-ray boundary-tie detection needed for the
 * exactness fallback. Page headers are re-checked once per first entry.
 */
export const webgpuSvoWideFanoutTraversalWGSL = /* wgsl */`
${webgpuSvoWideFanoutHelpersWGSL}
@group(0) @binding(0) var<storage, read> svoWidePages: array<SvoWidePage>;
@group(0) @binding(1) var<storage, read> svoWideDescriptors: array<SvoWideDescriptor>;

struct SvoWidePublication {
  generation: u32,
  sourceGeneration: u32,
  pageCount: u32,
  descriptorCount: u32,
};

struct SvoWideCursorFrame {
  pageIndex: u32,
  nextT: f32,
  exitT: f32,
  entered: u32,
  cellSteps: u32,
};

struct SvoWideTraversalCursor {
  frames: array<SvoWideCursorFrame, 12>,
  depth: u32,
  state: u32,
  pageVisits: u32,
  _padding: u32,
};

const SVO_WIDE_CURSOR_UNAVAILABLE: u32 = 0u;
const SVO_WIDE_CURSOR_ACTIVE: u32 = 1u;
const SVO_WIDE_CURSOR_COMPLETE: u32 = 2u;
const SVO_WIDE_CURSOR_INVALID: u32 = 3u;
const SVO_WIDE_CURSOR_EXHAUSTED: u32 = 4u;
const SVO_WIDE_CURSOR_STACK_CAPACITY: u32 = 12u;
const SVO_WIDE_MAXIMUM_PAGE_VISITS: u32 = 128u;
const SVO_WIDE_MAXIMUM_CELL_STEPS: u32 = 12u;

fn svoWidePublicationReady(publication: SvoWidePublication, canonicalSourceGeneration: u32) -> bool {
  return publication.generation != 0u
    && publication.sourceGeneration != 0u
    && publication.sourceGeneration == canonicalSourceGeneration
    && publication.pageCount <= arrayLength(&svoWidePages)
    && publication.descriptorCount <= arrayLength(&svoWideDescriptors)
    && (publication.pageCount != 0u || publication.descriptorCount == 0u);
}

fn svoWideCanonicalBounds(level: u32, coordinate: vec3u, mapping: SvoMapping) -> mat2x3f {
  let scale = f32((1u << (mapping.maximumDepth - level)) * mapping.brickSize);
  let minimum = mapping.worldOrigin + vec3f(coordinate) * scale * mapping.cellSize;
  return mat2x3f(minimum, minimum + scale * mapping.cellSize);
}

fn svoWidePageCoordinate(page: SvoWidePage) -> vec3u {
  return svoDecodeMorton(page.mortonLow, page.mortonHigh, page.level);
}

fn svoWideDdaSlotCoordinate(localPoint: vec3f, direction: vec3f) -> vec3u {
  let scaled = clamp(localPoint * 4.0, vec3f(0.0), vec3f(3.9999998));
  var coordinate = vec3u(floor(scaled));
  // A ray parallel to a shared face intersects both closed cells in the
  // canonical traversal. Select the lower coordinate to preserve its stable
  // ascending-octant tie break instead of depending on floating-point sign.
  for (var axis = 0u; axis < 3u; axis += 1u) {
    let boundary = round(scaled[axis]);
    if (direction[axis] == 0.0 && boundary > 0.0
        && abs(scaled[axis] - boundary) <= 1e-6) {
      coordinate[axis] = u32(boundary) - 1u;
    }
  }
  return min(coordinate, vec3u(3u));
}

fn svoWidePageHeaderValid(page: SvoWidePage, pageIndex: u32, publication: SvoWidePublication, mapping: SvoMapping) -> bool {
  return pageIndex < publication.pageCount
    && (page.level & 1u) == 0u
    && page.level <= mapping.maximumDepth
    && page.firstDescriptor <= publication.descriptorCount
    && page.descriptorCount <= publication.descriptorCount - page.firstDescriptor
    && page.descriptorCount == countOneBits(page.occupancyLow) + countOneBits(page.occupancyHigh);
}

fn svoWideCursorInitialize(
  cursor: ptr<function, SvoWideTraversalCursor>,
  ray: SvoRay,
  mapping: SvoMapping,
  publication: SvoWidePublication,
  canonicalSourceGeneration: u32,
) -> bool {
  (*cursor).depth = 0u;
  (*cursor).pageVisits = 0u;
  (*cursor).state = SVO_WIDE_CURSOR_UNAVAILABLE;
  if (!svoWidePublicationReady(publication, canonicalSourceGeneration)) { return false; }
  if (svoControl[12] != 0u) { return false; }
  // Closed-AABB canonical traversal can visit both sides of a shared face for
  // an exactly parallel ray. Defer that rare tie case to the canonical cursor
  // before yielding anything; ordinary camera/light rays remain accelerated.
  if (any(ray.direction == vec3f(0.0))) { return false; }
  if (mapping.nodeCount == 0u) {
    (*cursor).state = SVO_WIDE_CURSOR_COMPLETE;
    return true;
  }
  if (publication.pageCount == 0u) { return false; }
  let root = svoWidePages[0];
  if (!svoWidePageHeaderValid(root, 0u, publication, mapping)
      || root.level != 0u || root.mortonLow != 0u || root.mortonHigh != 0u) {
    (*cursor).state = SVO_WIDE_CURSOR_INVALID;
    return false;
  }
  let interval = svoRayAabbWithInverse(ray, 1.0 / ray.direction, svoRootBounds(mapping));
  if (interval.x == 0.0) {
    (*cursor).state = SVO_WIDE_CURSOR_COMPLETE;
    return true;
  }
  (*cursor).frames[0] = SvoWideCursorFrame(0u, interval.y, interval.z, 0u, 0u);
  (*cursor).depth = 1u;
  (*cursor).state = SVO_WIDE_CURSOR_ACTIVE;
  return true;
}

fn svoWideCursorMiss(status: u32, visits: u32) -> SvoTraversalHit {
  return SvoTraversalHit(status, visits, 0xffffffffu, 0xffffffffu, 0u, 0u, 0.0, 0.0);
}

fn svoWideCursorNext(
  cursor: ptr<function, SvoWideTraversalCursor>,
  ray: SvoRay,
  mapping: SvoMapping,
  maximumTraversalDepth: u32,
  publication: SvoWidePublication,
  canonicalSourceGeneration: u32,
) -> SvoTraversalHit {
  var callVisits = 0u;
  if ((*cursor).state == SVO_WIDE_CURSOR_COMPLETE) { return svoWideCursorMiss(SVO_STATUS_MISS, 0u); }
  if ((*cursor).state != SVO_WIDE_CURSOR_ACTIVE
      || !svoWidePublicationReady(publication, canonicalSourceGeneration)) {
    (*cursor).state = SVO_WIDE_CURSOR_INVALID;
    return svoWideCursorMiss(SVO_STATUS_INVALID_TOPOLOGY, 0u);
  }
  let inverseDirection = 1.0 / ray.direction;
  let visitLimit = min(max(mapping.maxVisits, 1u), SVO_WIDE_MAXIMUM_PAGE_VISITS);
  // The page record and its derived Morton coordinate and bounds are loop
  // invariants of every cell step inside one page; refresh them only when the
  // active frame changes so steps stay load- and decode-free.
  var cachedPageIndex = 0xffffffffu;
  var page: SvoWidePage;
  var pageCoordinate = vec3u(0u);
  var pageBounds = mat2x3f();
  for (var guard = 0u; guard < SVO_WIDE_MAXIMUM_PAGE_VISITS * SVO_WIDE_MAXIMUM_CELL_STEPS; guard += 1u) {
    if ((*cursor).depth == 0u) {
      (*cursor).state = SVO_WIDE_CURSOR_COMPLETE;
      return svoWideCursorMiss(SVO_STATUS_MISS, callVisits);
    }
    let frameIndex = (*cursor).depth - 1u;
    var frame = (*cursor).frames[frameIndex];
    if (frame.pageIndex >= publication.pageCount) {
      (*cursor).state = SVO_WIDE_CURSOR_INVALID;
      return svoWideCursorMiss(SVO_STATUS_INVALID_TOPOLOGY, callVisits);
    }
    if (frame.pageIndex != cachedPageIndex) {
      cachedPageIndex = frame.pageIndex;
      page = svoWidePages[frame.pageIndex];
      pageCoordinate = svoWidePageCoordinate(page);
      pageBounds = svoWideCanonicalBounds(page.level, pageCoordinate, mapping);
    }
    if (frame.entered == 0u) {
      if (callVisits >= visitLimit) {
        (*cursor).state = SVO_WIDE_CURSOR_EXHAUSTED;
        return svoWideCursorMiss(SVO_STATUS_WORK_EXHAUSTED, callVisits);
      }
      frame.entered = 1u;
      (*cursor).frames[frameIndex] = frame;
      (*cursor).pageVisits += 1u;
      callVisits += 1u;
      // Semantic header consistency is proven once at publish time; re-check
      // only on first entry so a corrupted upload still fails closed without
      // per-step validation traffic.
      if (!svoWidePageHeaderValid(page, frame.pageIndex, publication, mapping)) {
        (*cursor).state = SVO_WIDE_CURSOR_INVALID;
        return svoWideCursorMiss(SVO_STATUS_INVALID_TOPOLOGY, callVisits);
      }
    }
    if (page.level > maximumTraversalDepth) {
      (*cursor).state = SVO_WIDE_CURSOR_EXHAUSTED;
      return svoWideCursorMiss(SVO_STATUS_WORK_EXHAUSTED, callVisits);
    }
    if (frame.nextT > frame.exitT || (frame.cellSteps != 0u && frame.nextT == frame.exitT)) {
      (*cursor).depth -= 1u;
      continue;
    }
    if (frame.cellSteps >= SVO_WIDE_MAXIMUM_CELL_STEPS) {
      (*cursor).state = SVO_WIDE_CURSOR_INVALID;
      return svoWideCursorMiss(SVO_STATUS_INVALID_TOPOLOGY, callVisits);
    }
    (*cursor).frames[frameIndex].cellSteps += 1u;
    let pageExtent = pageBounds[1] - pageBounds[0];
    let probeT = frame.nextT;
    let localPointRaw = (ray.origin + ray.direction * probeT - pageBounds[0]) / pageExtent;
    let entryGrid = localPointRaw * 4.0;
    var entryBoundaryAxes = 0u;
    for (var axis = 0u; axis < 3u; axis += 1u) {
      let boundary = round(entryGrid[axis]);
      if (boundary > 0.0 && boundary < 4.0 && abs(entryGrid[axis] - boundary) <= 1e-5) { entryBoundaryAxes += 1u; }
    }
    if ((frame.cellSteps == 0u && entryBoundaryAxes != 0u) || entryBoundaryAxes >= 2u) {
      (*cursor).state = SVO_WIDE_CURSOR_INVALID;
      return svoWideCursorMiss(SVO_STATUS_INVALID_TOPOLOGY, callVisits);
    }
    let localBias = sign(ray.direction) * 1e-6;
    let localPoint = clamp(localPointRaw + localBias, vec3f(0.0), vec3f(0.99999994));
    let slotCoordinate = svoWideDdaSlotCoordinate(localPoint, ray.direction);
    let slot = slotCoordinate.x | (slotCoordinate.y << 2u) | (slotCoordinate.z << 4u);
    let slotMinimum = pageBounds[0] + pageExtent * (vec3f(slotCoordinate) * 0.25);
    let slotBounds = mat2x3f(slotMinimum, slotMinimum + pageExtent * 0.25);
    let slotInterval = svoRayAabbWithInverse(ray, inverseDirection, slotBounds);
    if (slotInterval.x == 0.0) {
      (*cursor).state = SVO_WIDE_CURSOR_INVALID;
      return svoWideCursorMiss(SVO_STATUS_INVALID_TOPOLOGY, callVisits);
    }
    let slotExit = min(frame.exitT, slotInterval.z);
    let exitGrid = (ray.origin + ray.direction * slotExit - pageBounds[0]) / pageExtent * 4.0;
    var internalBoundaryAxes = 0u;
    for (var axis = 0u; axis < 3u; axis += 1u) {
      let boundary = round(exitGrid[axis]);
      if (boundary > 0.0 && boundary < 4.0 && abs(exitGrid[axis] - boundary) <= 1e-5) {
        internalBoundaryAxes += 1u;
      }
    }
    if (internalBoundaryAxes >= 2u) {
      (*cursor).state = SVO_WIDE_CURSOR_INVALID;
      return svoWideCursorMiss(SVO_STATUS_INVALID_TOPOLOGY, callVisits);
    }
    (*cursor).frames[frameIndex].nextT = slotExit;
    if (!svoWideOccupied(page, slot)) { continue; }
    let rank = svoWideDescriptorRank(page, slot);
    if (rank >= page.descriptorCount) {
      (*cursor).state = SVO_WIDE_CURSOR_INVALID;
      return svoWideCursorMiss(SVO_STATUS_INVALID_TOPOLOGY, callVisits);
    }
    let descriptorIndex = page.firstDescriptor + rank;
    if (descriptorIndex >= publication.descriptorCount) {
      (*cursor).state = SVO_WIDE_CURSOR_INVALID;
      return svoWideCursorMiss(SVO_STATUS_INVALID_TOPOLOGY, callVisits);
    }
    let descriptor = svoWideDescriptors[descriptorIndex];
    let kind = svoWideDescriptorKind(descriptor);
    let sourceLevel = svoWideDescriptorSourceLevel(descriptor);
    if (svoWideDescriptorSlot(descriptor) != slot) {
      (*cursor).state = SVO_WIDE_CURSOR_INVALID;
      return svoWideCursorMiss(SVO_STATUS_INVALID_TOPOLOGY, callVisits);
    }
    let slotGlobalCoordinate = pageCoordinate * 4u + slotCoordinate;
    if (kind == SVO_WIDE_KIND_PAGE) {
      if (sourceLevel != page.level + 2u || sourceLevel > mapping.maximumDepth
          || descriptor.reference <= frame.pageIndex || descriptor.reference >= publication.pageCount
          || (*cursor).depth >= SVO_WIDE_CURSOR_STACK_CAPACITY) {
        (*cursor).state = SVO_WIDE_CURSOR_INVALID;
        return svoWideCursorMiss(SVO_STATUS_INVALID_TOPOLOGY, callVisits);
      }
      // The child page's header, level, and Morton coordinate were proven to
      // match this slot at publish time; the child is validated on entry.
      (*cursor).frames[(*cursor).depth] = SvoWideCursorFrame(
        descriptor.reference, max(frame.nextT, slotInterval.y), slotExit, 0u, 0u);
      (*cursor).depth += 1u;
      continue;
    }
    if (kind != SVO_WIDE_KIND_TERMINAL || sourceLevel < page.level
        || sourceLevel > page.level + 2u || sourceLevel > mapping.maximumDepth
        || descriptor.reference >= mapping.nodeCount
        || descriptor.sourceLeaf >= mapping.leafCount) {
      (*cursor).state = SVO_WIDE_CURSOR_INVALID;
      return svoWideCursorMiss(SVO_STATUS_INVALID_TOPOLOGY, callVisits);
    }
    if (sourceLevel > maximumTraversalDepth) {
      (*cursor).state = SVO_WIDE_CURSOR_EXHAUSTED;
      return svoWideCursorMiss(SVO_STATUS_WORK_EXHAUSTED, callVisits);
    }
    let terminalDivisor = 1u << (page.level + 2u - sourceLevel);
    let terminalCoordinate = slotGlobalCoordinate / terminalDivisor;
    // The canonical node record was cross-validated against this terminal at
    // publish time (level, leaf back-pointer, Morton coordinate). The leaf is
    // still loaded because the hit needs its voxel offset; its back-pointer is
    // a free consistency compare on already-loaded data.
    let leaf = svoLeaves[descriptor.sourceLeaf];
    if (leaf.topology.x != descriptor.reference) {
      (*cursor).state = SVO_WIDE_CURSOR_INVALID;
      return svoWideCursorMiss(SVO_STATUS_INVALID_TOPOLOGY, callVisits);
    }
    let terminalInterval = svoRayAabbWithInverse(ray, inverseDirection,
      svoWideCanonicalBounds(sourceLevel, terminalCoordinate, mapping));
    if (terminalInterval.x == 0.0) {
      (*cursor).state = SVO_WIDE_CURSOR_INVALID;
      return svoWideCursorMiss(SVO_STATUS_INVALID_TOPOLOGY, callVisits);
    }
    (*cursor).frames[frameIndex].nextT = min(frame.exitT, terminalInterval.z);
    return SvoTraversalHit(SVO_STATUS_HIT, callVisits, descriptor.reference, descriptor.sourceLeaf,
      leaf.topology.y, sourceLevel, max(ray.tMin, terminalInterval.y), min(ray.tMax, terminalInterval.z));
  }
  (*cursor).state = SVO_WIDE_CURSOR_EXHAUSTED;
  return svoWideCursorMiss(SVO_STATUS_WORK_EXHAUSTED, callVisits);
}
`;

export interface WebgpuSvoWideFanoutTraversalBindings {
  group?: number;
  pages?: number;
  descriptors?: number;
}

/** Bind the resumable wide traversal without consuming control or micro-mip slots. */
export function createWebgpuSvoWideFanoutTraversalWGSL(bindings: WebgpuSvoWideFanoutTraversalBindings = {}): string {
  const group = bindings.group ?? 0;
  const pages = bindings.pages ?? 0;
  const descriptors = bindings.descriptors ?? 1;
  for (const [label, value] of Object.entries({ group, pages, descriptors })) {
    if (!Number.isInteger(value) || value < 0) throw new RangeError(`Wide SVO WGSL ${label} must be a non-negative integer`);
  }
  if (pages === descriptors) throw new RangeError("Wide SVO WGSL bindings must be distinct");
  return webgpuSvoWideFanoutTraversalWGSL
    .replace("@group(0) @binding(0) var<storage, read> svoWidePages", `@group(${group}) @binding(${pages}) var<storage, read> svoWidePages`)
    .replace("@group(0) @binding(1) var<storage, read> svoWideDescriptors", `@group(${group}) @binding(${descriptors}) var<storage, read> svoWideDescriptors`);
}
