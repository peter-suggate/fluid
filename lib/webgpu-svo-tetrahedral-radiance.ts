import {
  SVO_NODE_MIP_LAYOUT,
  publishSvoNodeMipGeneration,
  resolveSvoNodeMipVirtualTexel,
  type SvoNodeMipPageKey,
  type SvoNodeMipPublication,
  type SvoNodeMipPublicationDecision,
  type SvoNodeMipPyramidPlan,
  type SvoNodeMipVirtualTexelAddress,
} from "./svo-node-mip-pyramid";
import {
  SVO_TETRAHEDRAL_RADIANCE_LAYOUT,
  svoTetrahedralRadianceAtlasBytes,
} from "./svo-tetrahedral-radiance";

export const WEBGPU_SVO_TETRAHEDRAL_RADIANCE_LAYOUT = Object.freeze({
  format: SVO_TETRAHEDRAL_RADIANCE_LAYOUT.textureFormat,
  directionCount: SVO_TETRAHEDRAL_RADIANCE_LAYOUT.directionCount,
  dimension: "3d" as GPUTextureDimension,
  usage: Object.freeze(["texture-binding", "copy-dst"] as const),
} as const);

/** Four packed RGB9E5 words, in the tetrahedral direction order defined by the CPU/GPU ABI. */
export type SvoPackedTetrahedralRadiance = readonly [number, number, number, number];

export type SvoTetrahedralRadianceApronSampler = (
  address: SvoNodeMipVirtualTexelAddress,
) => SvoPackedTetrahedralRadiance | undefined;

const INTERIOR_TEXELS = SVO_NODE_MIP_LAYOUT.interiorSize ** 3;
const PHYSICAL_TEXELS = SVO_NODE_MIP_LAYOUT.physicalSize ** 3;
const INTERIOR_WORDS = INTERIOR_TEXELS * SVO_TETRAHEDRAL_RADIANCE_LAYOUT.directionCount;
const PHYSICAL_WORDS = PHYSICAL_TEXELS * SVO_TETRAHEDRAL_RADIANCE_LAYOUT.directionCount;

function spatialTexelOffset(x: number, y: number, z: number, size: number): number {
  return ((z * size + y) * size + x) * SVO_TETRAHEDRAL_RADIANCE_LAYOUT.directionCount;
}

function packedSample(input: ArrayLike<number>, offset: number): SvoPackedTetrahedralRadiance {
  return [input[offset] >>> 0, input[offset + 1] >>> 0, input[offset + 2] >>> 0, input[offset + 3] >>> 0];
}

function assertInterior(interior: Uint32Array): void {
  if (interior.length !== INTERIOR_WORDS) {
    throw new RangeError("SVO tetrahedral-radiance interior must contain 8^3 four-word texels");
  }
}

/** Creates a physical 10^3 packed page whose apron clamps the 8^3 interior. */
export function createSvoTetrahedralRadiancePage(interior: Uint32Array): Uint32Array<ArrayBuffer> {
  assertInterior(interior);
  const n = SVO_NODE_MIP_LAYOUT.interiorSize;
  const physical = SVO_NODE_MIP_LAYOUT.physicalSize;
  const result = new Uint32Array(PHYSICAL_WORDS);
  for (let z = 0; z < physical; z += 1) for (let y = 0; y < physical; y += 1) for (let x = 0; x < physical; x += 1) {
    const sourceX = Math.max(0, Math.min(n - 1, x - 1));
    const sourceY = Math.max(0, Math.min(n - 1, y - 1));
    const sourceZ = Math.max(0, Math.min(n - 1, z - 1));
    result.set(packedSample(interior, spatialTexelOffset(sourceX, sourceY, sourceZ, n)), spatialTexelOffset(x, y, z, physical));
  }
  return result;
}

/** Fills the apron from resident same-level neighbours, clamping only missing neighbours. */
export function createSvoTetrahedralRadiancePageWithApron(
  pageCoordinate: SvoNodeMipPageKey["coordinate"],
  interior: Uint32Array,
  sampleNeighbour: SvoTetrahedralRadianceApronSampler,
): Uint32Array<ArrayBuffer> {
  const result = createSvoTetrahedralRadiancePage(interior);
  const n = SVO_NODE_MIP_LAYOUT.interiorSize;
  const physical = SVO_NODE_MIP_LAYOUT.physicalSize;
  for (let z = 0; z < physical; z += 1) for (let y = 0; y < physical; y += 1) for (let x = 0; x < physical; x += 1) {
    if (x > 0 && x < n + 1 && y > 0 && y < n + 1 && z > 0 && z < n + 1) continue;
    const address = resolveSvoNodeMipVirtualTexel(pageCoordinate, [x - 1, y - 1, z - 1]);
    const sample = address && sampleNeighbour(address);
    if (sample) result.set(packedSample(sample, 0), spatialTexelOffset(x, y, z, physical));
  }
  return result;
}

export interface WebGpuSvoTetrahedralRadianceVisibleGeneration {
  generation: number;
  /** The exact opacity node-mip topology; page slots and atlas origins are shared. */
  plan: SvoNodeMipPyramidPlan;
  /** Exact zero-radiance slots; safe to consume without sampling any radiance lobe. */
  blackSlots: ReadonlySet<number>;
  textures: readonly [GPUTexture, GPUTexture, GPUTexture, GPUTexture];
  views: readonly [GPUTextureView, GPUTextureView, GPUTextureView, GPUTextureView];
}

interface OwnedGeneration extends WebGpuSvoTetrahedralRadianceVisibleGeneration {
  blackSlots: Set<number>;
  uploadedSlots: Set<number>;
  payloadComplete: boolean;
  apronsComplete: boolean;
}

export interface WebGpuSvoTetrahedralRadianceTelemetry {
  visibleGeneration: number;
  candidateGeneration: number;
  residentPages: number;
  uploadedPages: number;
  blackPages: number;
  allocatedBytes: number;
  fallback: "none" | "previous-complete-generation" | "unavailable";
}

function createGeneration(device: GPUDevice, plan: SvoNodeMipPyramidPlan): OwnedGeneration {
  const size = plan.atlas.texels.map((component) => Math.max(1, component)) as [number, number, number];
  const textures = [0, 1, 2, 3].map((direction) => device.createTexture({
    label: `SVO tetrahedral radiance lobe ${direction} generation ${plan.generation}`,
    size,
    dimension: WEBGPU_SVO_TETRAHEDRAL_RADIANCE_LAYOUT.dimension,
    format: WEBGPU_SVO_TETRAHEDRAL_RADIANCE_LAYOUT.format,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  })) as [GPUTexture, GPUTexture, GPUTexture, GPUTexture];
  const views = textures.map((texture) => texture.createView({ dimension: "3d" })) as [
    GPUTextureView, GPUTextureView, GPUTextureView, GPUTextureView,
  ];
  return {
    generation: plan.generation,
    plan,
    textures,
    views,
    blackSlots: new Set<number>(),
    uploadedSlots: new Set<number>(),
    payloadComplete: plan.pages.length === 0,
    apronsComplete: plan.pages.length === 0,
  };
}

function matchingPage(candidate: OwnedGeneration, key: SvoNodeMipPageKey) {
  return candidate.plan.pages.find((entry) => entry.key.level === key.level
    && entry.key.coordinate.every((component, axis) => component === key.coordinate[axis]));
}

function markPageComplete(candidate: OwnedGeneration, slot: number): void {
  candidate.uploadedSlots.add(slot);
  candidate.payloadComplete = candidate.uploadedSlots.size === candidate.plan.pages.length;
  candidate.apronsComplete = candidate.payloadComplete;
}

function deinterleavePhysicalPage(data: Uint32Array): [Uint32Array<ArrayBuffer>, Uint32Array<ArrayBuffer>, Uint32Array<ArrayBuffer>, Uint32Array<ArrayBuffer>] {
  const lobes = [new Uint32Array(PHYSICAL_TEXELS), new Uint32Array(PHYSICAL_TEXELS),
    new Uint32Array(PHYSICAL_TEXELS), new Uint32Array(PHYSICAL_TEXELS)] as [
      Uint32Array<ArrayBuffer>, Uint32Array<ArrayBuffer>, Uint32Array<ArrayBuffer>, Uint32Array<ArrayBuffer>,
    ];
  for (let texel = 0; texel < PHYSICAL_TEXELS; texel += 1) {
    const source = texel * SVO_TETRAHEDRAL_RADIANCE_LAYOUT.directionCount;
    for (let direction = 0; direction < SVO_TETRAHEDRAL_RADIANCE_LAYOUT.directionCount; direction += 1) {
      lobes[direction][texel] = data[source + direction];
    }
  }
  return lobes;
}

/** Owns the optional four-texture radiance atlas without duplicating node-mip lookup resources. */
export class WebGpuSvoTetrahedralRadiance {
  private visible?: OwnedGeneration;
  private candidate?: OwnedGeneration;
  private destroyed = false;

  constructor(private readonly device: GPUDevice) {}

  /** Allocates a candidate using the node-mip plan's exact atlas shape and physical slots. */
  beginGeneration(plan: SvoNodeMipPyramidPlan): void {
    this.assertAlive();
    if (this.candidate) this.destroyGeneration(this.candidate);
    this.candidate = createGeneration(this.device, plan);
  }

  /** Uploads one already apron-complete 10^3 page, interleaved as four RGB9E5 words per texel. */
  uploadPhysicalPage(key: SvoNodeMipPageKey, data: Uint32Array): void {
    this.assertAlive();
    const candidate = this.requireCandidate(key.generation);
    if (data.length !== PHYSICAL_WORDS) {
      throw new RangeError("SVO tetrahedral-radiance physical upload must contain 10^3 four-word texels");
    }
    const page = matchingPage(candidate, key);
    if (!page) throw new Error("SVO tetrahedral-radiance upload key is not resident in the candidate plan");
    const size = SVO_NODE_MIP_LAYOUT.physicalSize;
    const lobes = deinterleavePhysicalPage(data);
    for (let direction = 0; direction < SVO_TETRAHEDRAL_RADIANCE_LAYOUT.directionCount; direction += 1) {
      this.device.queue.writeTexture(
        { texture: candidate.textures[direction], origin: page.atlasTexelOrigin },
        lobes[direction],
        { bytesPerRow: size * Uint32Array.BYTES_PER_ELEMENT, rowsPerImage: size },
        [size, size, size],
      );
    }
    markPageComplete(candidate, page.slot);
  }

  /** Pads an 8^3 packed interior, optionally sourcing apron texels from same-level neighbours. */
  uploadInteriorPage(
    key: SvoNodeMipPageKey,
    interior: Uint32Array,
    sampleNeighbour?: SvoTetrahedralRadianceApronSampler,
  ): void {
    const page = sampleNeighbour
      ? createSvoTetrahedralRadiancePageWithApron(key.coordinate, interior, sampleNeighbour)
      : createSvoTetrahedralRadiancePage(interior);
    this.uploadPhysicalPage(key, page);
  }

  /**
   * Certifies a physical page and its apron as black without an upload. WebGPU
   * textures are zero-initialized, and packed RGB9E5 zero is black.
   */
  certifyBlackPage(key: SvoNodeMipPageKey): void {
    this.assertAlive();
    const candidate = this.requireCandidate(key.generation);
    const page = matchingPage(candidate, key);
    if (!page) throw new Error("SVO tetrahedral-radiance black-page key is not resident in the candidate plan");
    candidate.blackSlots.add(page.slot);
    markPageComplete(candidate, page.slot);
  }

  /** Atomically swaps only after the whole candidate atlas, including aprons, is complete. */
  publish(): SvoNodeMipPublicationDecision {
    this.assertAlive();
    if (!this.candidate) throw new Error("No SVO tetrahedral-radiance candidate generation exists");
    const oldPublication: SvoNodeMipPublication | undefined = this.visible
      ? { completeGeneration: this.visible.generation, plan: this.visible.plan }
      : undefined;
    const decision = publishSvoNodeMipGeneration(oldPublication, {
      generation: this.candidate.generation,
      plan: this.candidate.plan,
      directoryComplete: true,
      payloadComplete: this.candidate.payloadComplete,
      apronsComplete: this.candidate.apronsComplete,
    });
    if (decision.published) {
      if (this.visible) this.destroyGeneration(this.visible);
      this.visible = this.candidate;
      this.candidate = undefined;
    }
    return decision;
  }

  visibleGeneration(): WebGpuSvoTetrahedralRadianceVisibleGeneration | undefined {
    if (!this.visible) return undefined;
    const { generation, plan, textures, views, blackSlots } = this.visible;
    return { generation, plan, textures, views, blackSlots };
  }

  telemetry(): WebGpuSvoTetrahedralRadianceTelemetry {
    const source = this.candidate ?? this.visible;
    return {
      visibleGeneration: this.visible?.generation ?? 0,
      candidateGeneration: this.candidate?.generation ?? 0,
      residentPages: source?.plan.residentPageCount ?? 0,
      uploadedPages: this.candidate?.uploadedSlots.size ?? this.visible?.uploadedSlots.size ?? 0,
      blackPages: source?.blackSlots.size ?? 0,
      allocatedBytes: (this.visible ? svoTetrahedralRadianceAtlasBytes(this.visible.plan) : 0)
        + (this.candidate ? svoTetrahedralRadianceAtlasBytes(this.candidate.plan) : 0),
      fallback: this.candidate
        ? (this.visible ? "previous-complete-generation" : "unavailable")
        : this.visible ? "none" : "unavailable",
    };
  }

  destroy(): void {
    if (this.destroyed) return;
    if (this.visible) this.destroyGeneration(this.visible);
    if (this.candidate) this.destroyGeneration(this.candidate);
    this.visible = undefined;
    this.candidate = undefined;
    this.destroyed = true;
  }

  private requireCandidate(generation: number): OwnedGeneration {
    if (!this.candidate || this.candidate.generation !== generation) {
      throw new Error("SVO tetrahedral-radiance upload does not match the candidate generation");
    }
    return this.candidate;
  }

  private assertAlive(): void {
    if (this.destroyed) throw new Error("WebGpuSvoTetrahedralRadiance is destroyed");
  }

  private destroyGeneration(generation: OwnedGeneration): void {
    generation.textures.forEach((texture) => texture.destroy());
  }
}
