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
}

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

export type WebGPUSvoWideFanoutEncodeStatus = "encoded" | "unchanged" | "capacity-exhausted" | "destroyed";

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

  encode(_encoder: GPUCommandEncoder, plan: SvoWideFanoutPlan): WebGPUSvoWideFanoutEncodeStatus {
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
      pageCount: plan.pages.length, descriptorCount: plan.descriptorCount,
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
  meta: u32,
  reference: u32,
  sourceLeaf: u32,
  opacity: u32,
};

fn svoWideDescriptorKind(descriptor: SvoWideDescriptor) -> u32 { return descriptor.meta & 3u; }
fn svoWideDescriptorSlot(descriptor: SvoWideDescriptor) -> u32 { return (descriptor.meta >> 2u) & 63u; }
fn svoWideDescriptorSourceLevel(descriptor: SvoWideDescriptor) -> u32 { return (descriptor.meta >> 8u) & 255u; }

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
