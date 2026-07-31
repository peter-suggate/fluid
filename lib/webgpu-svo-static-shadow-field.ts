import { svoNodeMipPageKey, type SvoNodeMipPageKey } from "./svo-node-mip-pyramid";
import {
  SVO_STATIC_SHADOW_FIELD_LAYOUT,
  type SvoStaticShadowCertificate,
  type SvoStaticShadowFieldPlan,
} from "./svo-static-shadow-field";

export const WEBGPU_SVO_STATIC_SHADOW_FIELD_LAYOUT = Object.freeze({
  parameterBytes: 32,
  proofJobWords: 5,
  proofJobBytes: 5 * Uint32Array.BYTES_PER_ELEMENT,
  workgroupSize: 64,
  bindings: Object.freeze({
    parameters: 0,
    certificates: 1,
    proofJobs: 2,
    dirtySlots: 3,
  }),
} as const);

export const WEBGPU_SVO_STATIC_SHADOW_CERTIFICATE_CODES = Object.freeze({
  unknown: 0,
  visible: 1,
  occluded: 2,
  mixed: 3,
} satisfies Readonly<Record<SvoStaticShadowCertificate, number>>);

export interface WebGpuSvoStaticShadowProof {
  page: SvoNodeMipPageKey | string;
  lightId: number;
  certificate: Exclude<SvoStaticShadowCertificate, "unknown">;
}

export interface WebGpuSvoStaticShadowFieldOptions {
  maximumAtlasCapacity: number;
  maximumProofJobs?: number;
  maximumDirtyPages?: number;
}

export interface WebGpuSvoStaticShadowFieldUpdate {
  plan: SvoStaticShadowFieldPlan;
  /**
   * Required on the first update and whenever the exact cache key changes.
   * It is encoded before invalidations and proofs in the same command stream.
   */
  fullInvalidate?: boolean;
  dirtyPages?: readonly (SvoNodeMipPageKey | string)[];
  proofs?: readonly WebGpuSvoStaticShadowProof[];
}

export interface WebGpuSvoStaticShadowFieldView {
  cacheKey: string;
  sourceGeneration: number;
  lightRevision: number;
  lightCount: number;
  atlasCapacity: number;
  buffer: GPUBuffer;
  binding: GPUBufferBinding;
}

export function webGpuSvoStaticShadowBuildBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  return [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
  ];
}

/** Renderer-side binding; consumers load two uint masks by the shared node-mip slot. */
export function webGpuSvoStaticShadowRenderBindGroupLayoutEntry(
  binding: number,
  visibility: GPUShaderStageFlags = GPUShaderStage.FRAGMENT,
): GPUBindGroupLayoutEntry {
  if (!Number.isInteger(binding) || binding < 0) throw new RangeError("Static-shadow render binding must be non-negative");
  return { binding, visibility, buffer: { type: "read-only-storage" } };
}

function positiveCapacity(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive safe integer`);
  return value;
}

function keyString(page: SvoNodeMipPageKey | string): string {
  return typeof page === "string" ? page : svoNodeMipPageKey(page);
}

/**
 * Packs exact-revision proof jobs for the GPU publisher. Duplicate page/light
 * jobs are rejected so result ordering cannot affect the two atomic bitplanes.
 */
export function packWebGpuSvoStaticShadowProofs(
  plan: SvoStaticShadowFieldPlan,
  proofs: readonly WebGpuSvoStaticShadowProof[],
): Uint32Array<ArrayBuffer> {
  const pageByKey = new Map(plan.pages.map((page) => [page.keyString, page]));
  const channelByLightId = new Map(plan.lightChannels.map((light) => [light.lightId, light]));
  const seen = new Set<string>();
  const words = new Uint32Array(proofs.length * WEBGPU_SVO_STATIC_SHADOW_FIELD_LAYOUT.proofJobWords);
  proofs.forEach((proof, index) => {
    const page = pageByKey.get(keyString(proof.page));
    if (!page) throw new Error("Static-shadow proof page is not resident in the exact topology");
    const light = channelByLightId.get(proof.lightId);
    if (!light) throw new Error(`Static-shadow proof light ${proof.lightId} is not in the exact light table`);
    const identity = `${page.slot}:${light.channel}`;
    if (seen.has(identity)) throw new Error(`Duplicate static-shadow proof job ${identity}`);
    seen.add(identity);
    const certificate = WEBGPU_SVO_STATIC_SHADOW_CERTIFICATE_CODES[proof.certificate];
    const offset = index * WEBGPU_SVO_STATIC_SHADOW_FIELD_LAYOUT.proofJobWords;
    words.set([page.slot, light.channel, certificate, plan.sourceGeneration, plan.lightRevision], offset);
  });
  return words;
}

export function packWebGpuSvoStaticShadowDirtySlots(
  plan: SvoStaticShadowFieldPlan,
  pages: readonly (SvoNodeMipPageKey | string)[],
): Uint32Array<ArrayBuffer> {
  const pageByKey = new Map(plan.pages.map((page) => [page.keyString, page]));
  const seen = new Set<number>();
  return Uint32Array.from(pages.map((input) => {
    const page = pageByKey.get(keyString(input));
    if (!page) throw new Error("Static-shadow dirty page is not resident in the exact topology");
    if (seen.has(page.slot)) throw new Error(`Duplicate static-shadow dirty slot ${page.slot}`);
    seen.add(page.slot);
    return page.slot;
  }));
}

/**
 * The proof pass only publishes conservative results produced against the
 * exact static generation and light revision. Stale work is discarded as 00,
 * which the renderer resolves by tracing the current frame.
 */
export const webGpuSvoStaticShadowFieldShader = /* wgsl */ `
struct StaticShadowParams {
  sourceGeneration:u32,
  lightRevision:u32,
  atlasCapacity:u32,
  lightCount:u32,
  proofCount:u32,
  dirtyCount:u32,
  reserved0:u32,
  reserved1:u32,
}
struct StaticShadowPage {
  visibleMask:atomic<u32>,
  occludedMask:atomic<u32>,
}
struct StaticShadowProof {
  slot:u32,
  lightChannel:u32,
  certificate:u32,
  sourceGeneration:u32,
  lightRevision:u32,
}
@group(0) @binding(0) var<uniform> params:StaticShadowParams;
@group(0) @binding(1) var<storage,read_write> shadowPages:array<StaticShadowPage>;
@group(0) @binding(2) var<storage,read> proofJobs:array<StaticShadowProof>;
@group(0) @binding(3) var<storage,read> dirtySlots:array<u32>;

const STATIC_SHADOW_VISIBLE:u32=1u;
const STATIC_SHADOW_OCCLUDED:u32=2u;
const STATIC_SHADOW_MIXED:u32=3u;

@compute @workgroup_size(${WEBGPU_SVO_STATIC_SHADOW_FIELD_LAYOUT.workgroupSize})
fn clearStaticShadowField(@builtin(global_invocation_id) gid:vec3u) {
  let slot=gid.x;
  if(slot>=params.atlasCapacity){return;}
  atomicStore(&shadowPages[slot].visibleMask,0u);
  atomicStore(&shadowPages[slot].occludedMask,0u);
}

@compute @workgroup_size(${WEBGPU_SVO_STATIC_SHADOW_FIELD_LAYOUT.workgroupSize})
fn invalidateStaticShadowPages(@builtin(global_invocation_id) gid:vec3u) {
  let index=gid.x;
  if(index>=params.dirtyCount){return;}
  let slot=dirtySlots[index];
  if(slot>=params.atlasCapacity){return;}
  atomicStore(&shadowPages[slot].visibleMask,0u);
  atomicStore(&shadowPages[slot].occludedMask,0u);
}

@compute @workgroup_size(${WEBGPU_SVO_STATIC_SHADOW_FIELD_LAYOUT.workgroupSize})
fn publishStaticShadowProofs(@builtin(global_invocation_id) gid:vec3u) {
  let index=gid.x;
  if(index>=params.proofCount){return;}
  let proof=proofJobs[index];
  if(proof.sourceGeneration!=params.sourceGeneration||proof.lightRevision!=params.lightRevision
    ||proof.slot>=params.atlasCapacity||proof.lightChannel>=params.lightCount||proof.lightChannel>=32u){return;}
  let lightBit=1u<<proof.lightChannel;
  if(proof.certificate==STATIC_SHADOW_VISIBLE||proof.certificate==STATIC_SHADOW_MIXED){
    atomicOr(&shadowPages[proof.slot].visibleMask,lightBit);
  }
  if(proof.certificate==STATIC_SHADOW_OCCLUDED||proof.certificate==STATIC_SHADOW_MIXED){
    atomicOr(&shadowPages[proof.slot].occludedMask,lightBit);
  }
}
`;

/**
 * Binding-free render helper. A revision mismatch and 00 both return unknown,
 * so renderer integration must take the exact current-frame static path.
 */
export const webGpuSvoStaticShadowFieldConsumerWGSL = /* wgsl */ `
const STATIC_SHADOW_UNKNOWN:u32=0u;
const STATIC_SHADOW_VISIBLE:u32=1u;
const STATIC_SHADOW_OCCLUDED:u32=2u;
const STATIC_SHADOW_MIXED:u32=3u;
fn resolveStaticShadowCertificate(
  masks:vec2u,
  lightChannel:u32,
  fieldSourceGeneration:u32,
  fieldLightRevision:u32,
  activeSourceGeneration:u32,
  activeLightRevision:u32,
)->u32{
  if(fieldSourceGeneration!=activeSourceGeneration||fieldLightRevision!=activeLightRevision||lightChannel>=32u){
    return STATIC_SHADOW_UNKNOWN;
  }
  let lightBit=1u<<lightChannel;
  return select(0u,1u,(masks.x&lightBit)!=0u)|select(0u,2u,(masks.y&lightBit)!=0u);
}
`;

function dispatchCount(workItems: number): number {
  return Math.ceil(workItems / WEBGPU_SVO_STATIC_SHADOW_FIELD_LAYOUT.workgroupSize);
}

/** Owns the fixed GPU publication buffers and deterministic update pipelines. */
export class WebGpuSvoStaticShadowField {
  readonly buffer: GPUBuffer;
  private readonly paramsBuffer: GPUBuffer;
  private readonly proofBuffer: GPUBuffer;
  private readonly dirtyBuffer: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;
  private readonly clearPipeline: GPUComputePipeline;
  private readonly invalidatePipeline: GPUComputePipeline;
  private readonly publishPipeline: GPUComputePipeline;
  private readonly maximumAtlasCapacity: number;
  private readonly maximumProofJobs: number;
  private readonly maximumDirtyPages: number;
  private active?: WebGpuSvoStaticShadowFieldView;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, options: WebGpuSvoStaticShadowFieldOptions) {
    this.maximumAtlasCapacity = positiveCapacity(options.maximumAtlasCapacity, "Static-shadow atlas capacity");
    this.maximumProofJobs = positiveCapacity(
      options.maximumProofJobs ?? Math.min(this.maximumAtlasCapacity * SVO_STATIC_SHADOW_FIELD_LAYOUT.maximumLights, 65_536),
      "Static-shadow proof capacity",
    );
    this.maximumDirtyPages = positiveCapacity(
      options.maximumDirtyPages ?? this.maximumAtlasCapacity,
      "Static-shadow dirty-page capacity",
    );
    const usage = GPUBufferUsage;
    this.buffer = device.createBuffer({
      label: "SVO static shadow certificates",
      size: this.maximumAtlasCapacity * SVO_STATIC_SHADOW_FIELD_LAYOUT.bytesPerPage,
      usage: usage.STORAGE | usage.COPY_SRC | usage.COPY_DST,
    });
    this.paramsBuffer = device.createBuffer({
      label: "SVO static shadow parameters",
      size: WEBGPU_SVO_STATIC_SHADOW_FIELD_LAYOUT.parameterBytes,
      usage: usage.UNIFORM | usage.COPY_DST,
    });
    this.proofBuffer = device.createBuffer({
      label: "SVO static shadow proof jobs",
      size: this.maximumProofJobs * WEBGPU_SVO_STATIC_SHADOW_FIELD_LAYOUT.proofJobBytes,
      usage: usage.STORAGE | usage.COPY_DST,
    });
    this.dirtyBuffer = device.createBuffer({
      label: "SVO static shadow dirty slots",
      size: this.maximumDirtyPages * Uint32Array.BYTES_PER_ELEMENT,
      usage: usage.STORAGE | usage.COPY_DST,
    });
    const layout = device.createBindGroupLayout({
      label: "SVO static shadow build layout",
      entries: webGpuSvoStaticShadowBuildBindGroupLayoutEntries(),
    });
    const pipelineLayout = device.createPipelineLayout({ label: "SVO static shadow pipeline layout", bindGroupLayouts: [layout] });
    const shaderModule = device.createShaderModule({ label: "SVO static shadow publication", code: webGpuSvoStaticShadowFieldShader });
    const pipeline = (entryPoint: string) => device.createComputePipeline({
      label: `SVO static shadow ${entryPoint}`,
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint },
    });
    this.clearPipeline = pipeline("clearStaticShadowField");
    this.invalidatePipeline = pipeline("invalidateStaticShadowPages");
    this.publishPipeline = pipeline("publishStaticShadowProofs");
    this.bindGroup = device.createBindGroup({
      label: "SVO static shadow build bindings",
      layout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.buffer } },
        { binding: 2, resource: { buffer: this.proofBuffer } },
        { binding: 3, resource: { buffer: this.dirtyBuffer } },
      ],
    });
  }

  encodeUpdate(encoder: GPUCommandEncoder, update: WebGpuSvoStaticShadowFieldUpdate): WebGpuSvoStaticShadowFieldView {
    this.assertAlive();
    const { plan } = update;
    if (plan.atlasCapacity > this.maximumAtlasCapacity) throw new RangeError("Static-shadow plan exceeds GPU atlas capacity");
    if (plan.lightChannels.length > SVO_STATIC_SHADOW_FIELD_LAYOUT.maximumLights) throw new RangeError("Static-shadow plan exceeds GPU light capacity");
    const keyChanged = this.active?.cacheKey !== plan.cacheKey;
    if (keyChanged && !update.fullInvalidate) {
      throw new Error("A changed static-shadow revision requires full invalidation before same-frame use");
    }
    const proofs = update.proofs ?? [];
    const dirtyPages = update.dirtyPages ?? [];
    if (proofs.length > this.maximumProofJobs) throw new RangeError("Static-shadow proof update exceeds GPU job capacity");
    if (dirtyPages.length > this.maximumDirtyPages) throw new RangeError("Static-shadow dirty update exceeds GPU page capacity");
    const proofWords = packWebGpuSvoStaticShadowProofs(plan, proofs);
    const dirtyWords = packWebGpuSvoStaticShadowDirtySlots(plan, dirtyPages);
    const params = new Uint32Array([
      plan.sourceGeneration,
      plan.lightRevision,
      plan.atlasCapacity,
      plan.lightChannels.length,
      proofs.length,
      dirtyPages.length,
      0,
      0,
    ]);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, params);
    if (proofWords.length) this.device.queue.writeBuffer(this.proofBuffer, 0, proofWords);
    if (dirtyWords.length) this.device.queue.writeBuffer(this.dirtyBuffer, 0, dirtyWords);
    const pass = encoder.beginComputePass({ label: "SVO static shadow exact-revision update" });
    pass.setBindGroup(0, this.bindGroup);
    if (update.fullInvalidate) {
      pass.setPipeline(this.clearPipeline);
      pass.dispatchWorkgroups(dispatchCount(plan.atlasCapacity));
    }
    if (dirtyPages.length) {
      pass.setPipeline(this.invalidatePipeline);
      pass.dispatchWorkgroups(dispatchCount(dirtyPages.length));
    }
    if (proofs.length) {
      pass.setPipeline(this.publishPipeline);
      pass.dispatchWorkgroups(dispatchCount(proofs.length));
    }
    pass.end();
    this.active = {
      cacheKey: plan.cacheKey,
      sourceGeneration: plan.sourceGeneration,
      lightRevision: plan.lightRevision,
      lightCount: plan.lightChannels.length,
      atlasCapacity: plan.atlasCapacity,
      buffer: this.buffer,
      binding: { buffer: this.buffer, size: plan.atlasCapacity * SVO_STATIC_SHADOW_FIELD_LAYOUT.bytesPerPage },
    };
    return this.active;
  }

  visibleView(): WebGpuSvoStaticShadowFieldView | undefined {
    return this.active;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.buffer.destroy();
    this.paramsBuffer.destroy();
    this.proofBuffer.destroy();
    this.dirtyBuffer.destroy();
    this.active = undefined;
    this.destroyed = true;
  }

  private assertAlive(): void {
    if (this.destroyed) throw new Error("WebGpuSvoStaticShadowField is destroyed");
  }
}
