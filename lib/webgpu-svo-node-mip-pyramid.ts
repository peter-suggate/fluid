import {
  SVO_NODE_MIP_LAYOUT,
  createSvoNodeMipPage,
  packSvoNodeMipPageKey,
  publishSvoNodeMipGeneration,
  type SvoNodeMipPageKey,
  type SvoNodeMipPagePlan,
  type SvoNodeMipPublication,
  type SvoNodeMipPublicationDecision,
  type SvoNodeMipPyramidPlan,
} from "./svo-node-mip-pyramid";
import { svoNodeMipSamplingWGSL } from "./svo-node-mip-sampling";

export const WEBGPU_SVO_NODE_MIP_LAYOUT = Object.freeze({
  format: "rgba8unorm" as GPUTextureFormat,
  directoryTextureFormat: "rgba32uint" as GPUTextureFormat,
  directoryTexelsPerPage: 2,
  dimension: "3d" as GPUTextureDimension,
  directoryStrideBytes: SVO_NODE_MIP_LAYOUT.directoryBytesPerPage,
  directoryWordsPerPage: SVO_NODE_MIP_LAYOUT.directoryBytesPerPage / 4,
  sampler: Object.freeze({ magFilter: "linear", minFilter: "linear", mipmapFilter: "nearest" } as const),
} as const);

export const WEBGPU_SVO_NODE_MIP_DIRECTORY_ABI = Object.freeze({
  keyTexelX: 0,
  locationTexelX: 1,
  keyLanes: Object.freeze({ generation: 0, level: 1, mortonLow: 2, mortonHigh: 3 }),
  locationLanes: Object.freeze({ atlasOriginX: 0, atlasOriginY: 1, atlasOriginZ: 2, slot: 3 }),
} as const);

/**
 * Bindings intentionally live here, outside `svoNodeMipSamplingWGSL`; consumers can
 * embed the sampling library into an existing renderer bind group without collisions.
 */
export const webgpuSvoNodeMipSamplingValidationWGSL = /* wgsl */ `
${svoNodeMipSamplingWGSL}
@group(0) @binding(0) var nodeMipAtlas:texture_3d<f32>;
@group(0) @binding(1) var nodeMipSampler:sampler;
@group(0) @binding(2) var nodeMipDirectory:texture_2d<u32>;
@compute @workgroup_size(1) fn validateNodeMipSampling(){
  _=svoNodeMipSamplePage(nodeMipAtlas,nodeMipSampler,vec3u(0u),vec3f(0.0));
  _=svoNodeMipDirectoryEntry(nodeMipDirectory,0u);
}
`;

export interface WebGpuSvoNodeMipVisibleGeneration {
  generation: number;
  plan: SvoNodeMipPyramidPlan;
  texture: GPUTexture;
  view: GPUTextureView;
  sampler: GPUSampler;
  directory: GPUBuffer;
  /** Sampled uint directory avoids consuming an additional renderer storage binding. */
  directoryTexture: GPUTexture;
  directoryView: GPUTextureView;
}

interface OwnedGeneration extends WebGpuSvoNodeMipVisibleGeneration {
  uploadedSlots: Set<number>;
  directoryComplete: boolean;
  payloadComplete: boolean;
  apronsComplete: boolean;
}

export interface WebGpuSvoNodeMipTelemetry {
  visibleGeneration: number;
  candidateGeneration: number;
  residentPages: number;
  uploadedPages: number;
  allocatedBytes: number;
  fallback: "none" | "previous-complete-generation" | "unavailable";
}

function createGeneration(device: GPUDevice, plan: SvoNodeMipPyramidPlan, sampler: GPUSampler): OwnedGeneration {
  const dimensions = plan.atlas.texels.map((component) => Math.max(1, component)) as [number, number, number];
  const texture = device.createTexture({
    label: `SVO node mip atlas generation ${plan.generation}`,
    size: dimensions,
    dimension: WEBGPU_SVO_NODE_MIP_LAYOUT.dimension,
    format: WEBGPU_SVO_NODE_MIP_LAYOUT.format,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const directory = device.createBuffer({
    label: `SVO node mip directory generation ${plan.generation}`,
    size: Math.max(SVO_NODE_MIP_LAYOUT.directoryBytesPerPage, plan.pages.length * SVO_NODE_MIP_LAYOUT.directoryBytesPerPage),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const directoryTexture = device.createTexture({
    label: `SVO node mip sampled directory generation ${plan.generation}`,
    size: [WEBGPU_SVO_NODE_MIP_LAYOUT.directoryTexelsPerPage, Math.max(1, plan.pages.length)],
    format: WEBGPU_SVO_NODE_MIP_LAYOUT.directoryTextureFormat,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  return {
    generation: plan.generation,
    plan,
    texture,
    view: texture.createView({ dimension: "3d" }),
    sampler,
    directory,
    directoryTexture,
    directoryView: directoryTexture.createView(),
    uploadedSlots: new Set<number>(),
    directoryComplete: false,
    payloadComplete: false,
    apronsComplete: false,
  };
}

function packDirectoryEntry(page: SvoNodeMipPagePlan): Uint32Array {
  const key = packSvoNodeMipPageKey(page.key);
  return new Uint32Array([
    key[0], key[1], key[2], key[3],
    page.atlasTexelOrigin[0], page.atlasTexelOrigin[1], page.atlasTexelOrigin[2], page.slot,
  ]);
}

/** GPU resource owner with explicit two-generation publication and old-generation fallback. */
export class WebGpuSvoNodeMipPyramid {
  private readonly sampler: GPUSampler;
  private visible?: OwnedGeneration;
  private candidate?: OwnedGeneration;
  private destroyed = false;

  constructor(private readonly device: GPUDevice) {
    this.sampler = device.createSampler({
      label: "SVO node mip atlas sampler",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
      addressModeW: "clamp-to-edge",
      ...WEBGPU_SVO_NODE_MIP_LAYOUT.sampler,
    });
  }

  /** Begins an upload without changing the generation currently visible to rendering. */
  beginGeneration(plan: SvoNodeMipPyramidPlan): void {
    this.assertAlive();
    if (this.candidate) this.destroyGeneration(this.candidate);
    this.candidate = createGeneration(this.device, plan, this.sampler);
    const words = new Uint32Array(Math.max(1, plan.pages.length) * WEBGPU_SVO_NODE_MIP_LAYOUT.directoryWordsPerPage);
    for (const page of plan.pages) words.set(packDirectoryEntry(page), page.slot * WEBGPU_SVO_NODE_MIP_LAYOUT.directoryWordsPerPage);
    if (plan.pages.length) this.device.queue.writeBuffer(this.candidate.directory, 0, words, 0, plan.pages.length * WEBGPU_SVO_NODE_MIP_LAYOUT.directoryWordsPerPage);
    if (plan.pages.length) this.device.queue.writeTexture(
      { texture: this.candidate.directoryTexture },
      words,
      { bytesPerRow: SVO_NODE_MIP_LAYOUT.directoryBytesPerPage, rowsPerImage: plan.pages.length },
      [WEBGPU_SVO_NODE_MIP_LAYOUT.directoryTexelsPerPage, plan.pages.length],
    );
    this.candidate.directoryComplete = true;
    if (plan.pages.length === 0) {
      this.candidate.payloadComplete = true;
      this.candidate.apronsComplete = true;
    }
  }

  /** Uploads an already apron-padded 10^3 RGBA8 physical page. */
  uploadPhysicalPage(key: SvoNodeMipPageKey, data: Uint8Array): void {
    this.assertAlive();
    const candidate = this.requireCandidate(key.generation);
    if (data.byteLength !== SVO_NODE_MIP_LAYOUT.bytesPerPage) throw new RangeError("SVO node-mip physical upload must contain 10^3 RGBA8 texels");
    const page = candidate.plan.pages.find((entry) => entry.key.level === key.level
      && entry.key.coordinate.every((component, axis) => component === key.coordinate[axis]));
    if (!page) throw new Error("SVO node-mip upload key is not resident in the candidate plan");
    const size = SVO_NODE_MIP_LAYOUT.physicalSize;
    this.device.queue.writeTexture(
      { texture: candidate.texture, origin: page.atlasTexelOrigin },
      new Uint8Array(data),
      { bytesPerRow: size * SVO_NODE_MIP_LAYOUT.bytesPerTexel, rowsPerImage: size },
      [size, size, size],
    );
    candidate.uploadedSlots.add(page.slot);
    candidate.payloadComplete = candidate.uploadedSlots.size === candidate.plan.pages.length;
    candidate.apronsComplete = candidate.payloadComplete;
  }

  /** Pads an 8^3 interior on the CPU and uploads the resulting physical page. */
  uploadInteriorPage(key: SvoNodeMipPageKey, interior: Uint8Array): void {
    this.uploadPhysicalPage(key, createSvoNodeMipPage(interior));
  }

  /**
   * Attempts atomic publication. On every rejection the previous complete texture,
   * directory and sampler remain visible and may be used as a coarse/exact fallback.
   */
  publish(): SvoNodeMipPublicationDecision {
    this.assertAlive();
    if (!this.candidate) throw new Error("No SVO node-mip candidate generation exists");
    const oldPublication: SvoNodeMipPublication | undefined = this.visible
      ? { completeGeneration: this.visible.generation, plan: this.visible.plan }
      : undefined;
    const decision = publishSvoNodeMipGeneration(oldPublication, {
      generation: this.candidate.generation,
      plan: this.candidate.plan,
      directoryComplete: this.candidate.directoryComplete,
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

  visibleGeneration(): WebGpuSvoNodeMipVisibleGeneration | undefined {
    if (!this.visible) return undefined;
    const { generation, plan, texture, view, sampler, directory, directoryTexture, directoryView } = this.visible;
    return { generation, plan, texture, view, sampler, directory, directoryTexture, directoryView };
  }

  telemetry(): WebGpuSvoNodeMipTelemetry {
    const source = this.candidate ?? this.visible;
    return {
      visibleGeneration: this.visible?.generation ?? 0,
      candidateGeneration: this.candidate?.generation ?? 0,
      residentPages: source?.plan.residentPageCount ?? 0,
      uploadedPages: this.candidate?.uploadedSlots.size ?? this.visible?.uploadedSlots.size ?? 0,
      allocatedBytes: (this.visible ? this.visible.plan.allocatedBytes + this.visible.plan.directoryBytes : 0)
        + (this.candidate ? this.candidate.plan.allocatedBytes + this.candidate.plan.directoryBytes : 0),
      fallback: this.candidate ? (this.visible ? "previous-complete-generation" : "unavailable") : this.visible ? "none" : "unavailable",
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
    if (!this.candidate || this.candidate.generation !== generation) throw new Error("SVO node-mip upload does not match the candidate generation");
    return this.candidate;
  }

  private assertAlive(): void {
    if (this.destroyed) throw new Error("WebGpuSvoNodeMipPyramid is destroyed");
  }

  private destroyGeneration(generation: OwnedGeneration): void {
    generation.texture.destroy();
    generation.directory.destroy();
    generation.directoryTexture.destroy();
  }
}
