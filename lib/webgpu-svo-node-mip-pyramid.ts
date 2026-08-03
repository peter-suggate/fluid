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
import {
  WebGpuLiveSvoDerivedPageState,
  type LiveSvoDerivedPageValidityBinding,
} from "./webgpu-svo-live-derived-cache";

export const WEBGPU_SVO_NODE_MIP_LAYOUT = Object.freeze({
  format: "rgba8unorm" as GPUTextureFormat,
  directoryTextureFormat: "rgba32uint" as GPUTextureFormat,
  directPageTableTextureFormat: "r32uint" as GPUTextureFormat,
  directPageTableMaximumLevels: 12,
  directPageTableMaximumBytes: 64 * 1024 * 1024,
  directoryTexelsPerPage: 2,
  dimension: "3d" as GPUTextureDimension,
  directoryStrideBytes: SVO_NODE_MIP_LAYOUT.directoryBytesPerPage,
  directoryWordsPerPage: SVO_NODE_MIP_LAYOUT.directoryBytesPerPage / 4,
  sampler: Object.freeze({ magFilter: "linear", minFilter: "linear", mipmapFilter: "nearest" } as const),
} as const);

/**
 * How the sampled directory's pages are laid out in a 2D texture.
 *
 * The directory used to be two texels wide with one *row* per page, which made
 * the 2D height limit the entire page ceiling — 16 384 on an M1 Max, which the
 * hero pond reaches at a 5 mm lattice and every dense domain reaches eventually.
 * Wrapping pages across columns instead costs one integer divide in the lookup
 * and squares the ceiling. Pages stay in slot order along the row and then wrap,
 * so the CPU-side buffer needs no repacking and the directory remains sorted by
 * (level, morton) in page-index order, which is what the marcher's binary search
 * over a level's run depends on.
 *
 * One column reproduces the old layout byte for byte, so every scene below the
 * wrap threshold is unchanged.
 */
export function webGpuSvoNodeMipDirectoryShape(
  pages: number,
  maximumDimension: number,
): { columns: number; rows: number; texels: readonly [number, number] } {
  if (!Number.isSafeInteger(pages) || pages < 0) throw new RangeError("SVO node-mip directory page count must be a non-negative safe integer");
  const width = Math.max(1, Math.floor(maximumDimension / WEBGPU_SVO_NODE_MIP_LAYOUT.directoryTexelsPerPage));
  const height = Math.max(1, maximumDimension);
  const columns = Math.min(width, Math.max(1, Math.ceil(Math.max(1, pages) / height)));
  const rows = Math.max(1, Math.ceil(Math.max(1, pages) / columns));
  if (rows > height) throw new RangeError(`SVO node-mip directory needs ${rows} rows, exceeding the ${height} texture limit`);
  return { columns, rows, texels: [columns * WEBGPU_SVO_NODE_MIP_LAYOUT.directoryTexelsPerPage, rows] };
}

/**
 * Largest page count this device can address.
 *
 * The binding constraint is the sampled directory's area under the layout above.
 * Capping below it does not save memory — an atlas is only allocated for the
 * pages that are actually resident — it just silently discards scene geometry,
 * which the marcher then reads as empty air because a non-resident page samples
 * as zero. The atlas volume and the direct page table are checked separately by
 * the caller; this number is only about addressability.
 *
 * The floor keeps a device that reports no limits usable rather than empty.
 */
export function webGpuSvoNodeMipMaximumPages(device: Pick<GPUDevice, "limits">): number {
  const raw = Number(device.limits?.maxTextureDimension2D);
  const limit = Number.isSafeInteger(raw) && raw >= 2_048 ? raw : 2_048;
  return limit * Math.floor(limit / WEBGPU_SVO_NODE_MIP_LAYOUT.directoryTexelsPerPage);
}

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
  /** Level slabs of slot+1 values. Zero is non-resident; the directory remains a fallback. */
  directPageTableTexture: GPUTexture;
  directPageTableView: GPUTextureView;
  directPageTableDimensions: readonly [number, number, number];
  directPageTableLevelZOffsets: Uint32Array<ArrayBuffer>;
  directPageTableReady: boolean;
  directPageTableBytes: number;
  /** Optional world-space coordinate frame when it differs from the structural tree. */
  worldOrigin_m?: readonly [number, number, number];
  /** Full sparse-lighting lattice extent, including authored scenery around the solver tank. */
  worldExtent_m?: readonly [number, number, number];
}

interface OwnedGeneration extends WebGpuSvoNodeMipVisibleGeneration {
  /** Column/row wrap of the sampled directory; the shader rederives it from the width. */
  directoryShape: { columns: number; rows: number; texels: readonly [number, number] };
  directPageTableWords: Uint32Array<ArrayBuffer>;
  uploadedSlots: Set<number>;
  directoryComplete: boolean;
  payloadComplete: boolean;
  apronsComplete: boolean;
}

export interface WebGpuSvoNodeMipDirectPageTable {
  dimensions: readonly [number, number, number];
  levelZOffsets: Uint32Array<ArrayBuffer>;
  words: Uint32Array<ArrayBuffer>;
  ready: boolean;
}

/**
 * Packs each virtual mip level into a Z slab. The texel stores atlas slot+1, so
 * shader lookup needs one textureLoad and can derive the physical atlas origin
 * from the regular row-major slot mapping. Pathological sparse extents retain
 * the compact sorted-directory fallback instead of allocating a huge volume.
 */
export function createWebGpuSvoNodeMipDirectPageTable(
  plan: SvoNodeMipPyramidPlan,
  maximumDimension = 2_048,
): WebGpuSvoNodeMipDirectPageTable {
  const levelZOffsets = new Uint32Array(WEBGPU_SVO_NODE_MIP_LAYOUT.directPageTableMaximumLevels);
  if (plan.pages.length === 0) return { dimensions: [1, 1, 1], levelZOffsets, words: new Uint32Array(1), ready: false };
  // Folded rather than spread: `Math.max(...pages)` is an argument list, and a
  // domain fine enough to need six figures of pages overflows the call stack
  // before it ever reaches a device limit.
  const levelCount = plan.pages.reduce((maximum, page) => Math.max(maximum, page.key.level + 1), 1);
  if (levelCount > WEBGPU_SVO_NODE_MIP_LAYOUT.directPageTableMaximumLevels) {
    return { dimensions: [1, 1, 1], levelZOffsets, words: new Uint32Array(1), ready: false };
  }
  let width = 1, height = 1, depth = 0;
  for (let level = 0; level < levelCount; level += 1) {
    levelZOffsets[level] = depth;
    let levelDepth = 0;
    for (const page of plan.pages) if (page.key.level === level) {
      width = Math.max(width, page.key.coordinate[0] + 1);
      height = Math.max(height, page.key.coordinate[1] + 1);
      levelDepth = Math.max(levelDepth, page.key.coordinate[2] + 1);
    }
    depth += levelDepth;
  }
  const wordCount = width * height * depth;
  const ready = width <= maximumDimension && height <= maximumDimension && depth <= maximumDimension
    && Number.isSafeInteger(wordCount)
    && wordCount * Uint32Array.BYTES_PER_ELEMENT <= WEBGPU_SVO_NODE_MIP_LAYOUT.directPageTableMaximumBytes;
  if (!ready) return { dimensions: [1, 1, 1], levelZOffsets: new Uint32Array(levelZOffsets.length), words: new Uint32Array(1), ready: false };
  const words = new Uint32Array(wordCount);
  for (const page of plan.pages) {
    const [x, y, z] = page.key.coordinate;
    words[((levelZOffsets[page.key.level] + z) * height + y) * width + x] = page.slot + 1;
  }
  return { dimensions: [width, height, depth], levelZOffsets, words, ready: true };
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
  const directoryShape = webGpuSvoNodeMipDirectoryShape(plan.pages.length, device.limits?.maxTextureDimension2D ?? 2_048);
  const directoryTexture = device.createTexture({
    label: `SVO node mip sampled directory generation ${plan.generation}`,
    size: [...directoryShape.texels],
    format: WEBGPU_SVO_NODE_MIP_LAYOUT.directoryTextureFormat,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const directPageTable = createWebGpuSvoNodeMipDirectPageTable(plan, device.limits?.maxTextureDimension3D ?? 2_048);
  const directPageTableTexture = device.createTexture({
    label: `SVO node mip direct page table generation ${plan.generation}`,
    size: directPageTable.dimensions,
    dimension: "3d",
    format: WEBGPU_SVO_NODE_MIP_LAYOUT.directPageTableTextureFormat,
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
    directoryShape,
    directPageTableTexture,
    directPageTableView: directPageTableTexture.createView({ dimension: "3d" }),
    directPageTableDimensions: directPageTable.dimensions,
    directPageTableLevelZOffsets: directPageTable.levelZOffsets,
    directPageTableReady: directPageTable.ready,
    directPageTableBytes: directPageTable.ready ? directPageTable.words.byteLength : 0,
    directPageTableWords: directPageTable.words,
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
    const { columns, rows, texels } = this.candidate.directoryShape;
    // Slot order along the row, so the same linear buffer serves the storage
    // directory and the wrapped texture; the tail of the last row stays zero,
    // which reads back as generation 0 and therefore as a miss.
    const words = new Uint32Array(columns * rows * WEBGPU_SVO_NODE_MIP_LAYOUT.directoryWordsPerPage);
    for (const page of plan.pages) words.set(packDirectoryEntry(page), page.slot * WEBGPU_SVO_NODE_MIP_LAYOUT.directoryWordsPerPage);
    if (plan.pages.length) this.device.queue.writeBuffer(this.candidate.directory, 0, words, 0, plan.pages.length * WEBGPU_SVO_NODE_MIP_LAYOUT.directoryWordsPerPage);
    if (plan.pages.length) this.device.queue.writeTexture(
      { texture: this.candidate.directoryTexture },
      words,
      { bytesPerRow: columns * SVO_NODE_MIP_LAYOUT.directoryBytesPerPage, rowsPerImage: rows },
      [...texels],
    );
    if (this.candidate.directPageTableReady) {
      this.device.queue.writeTexture(
        { texture: this.candidate.directPageTableTexture },
        this.candidate.directPageTableWords,
        { bytesPerRow: this.candidate.directPageTableDimensions[0] * Uint32Array.BYTES_PER_ELEMENT, rowsPerImage: this.candidate.directPageTableDimensions[1] },
        this.candidate.directPageTableDimensions,
      );
    }
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
    const { generation, plan, texture, view, sampler, directory, directoryTexture, directoryView,
      directPageTableTexture, directPageTableView, directPageTableDimensions, directPageTableLevelZOffsets,
      directPageTableReady, directPageTableBytes } = this.visible;
    return { generation, plan, texture, view, sampler, directory, directoryTexture, directoryView,
      directPageTableTexture, directPageTableView, directPageTableDimensions, directPageTableLevelZOffsets,
      directPageTableReady, directPageTableBytes };
  }

  telemetry(): WebGpuSvoNodeMipTelemetry {
    const source = this.candidate ?? this.visible;
    return {
      visibleGeneration: this.visible?.generation ?? 0,
      candidateGeneration: this.candidate?.generation ?? 0,
      residentPages: source?.plan.residentPageCount ?? 0,
      uploadedPages: this.candidate?.uploadedSlots.size ?? this.visible?.uploadedSlots.size ?? 0,
      allocatedBytes: (this.visible ? this.visible.plan.allocatedBytes + this.visible.directPageTableBytes : 0)
        + (this.candidate ? this.candidate.plan.allocatedBytes + this.candidate.directPageTableBytes : 0),
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
    generation.directPageTableTexture.destroy();
  }
}

export interface WebGpuLiveSvoNodeMipOptions {
  pageCapacity: number;
  atlasTexels: readonly [number, number, number];
  /** Fixed direct-table extent. Omit to use only the compact sampled directory. */
  directPageTableDimensions?: readonly [number, number, number];
  label?: string;
}

export interface WebGpuLiveSvoNodeMipVisibleGeneration extends WebGpuSvoNodeMipVisibleGeneration {
  pageValidity: LiveSvoDerivedPageValidityBinding;
}

export interface WebGpuLiveSvoNodeMipGpuTarget {
  texture: GPUTexture;
  pageValidity: LiveSvoDerivedPageValidityBinding;
  atlasPages: readonly [number, number, number];
  pageCapacity: number;
  directPageTableTexture: GPUTexture;
  directPageTableDimensions: readonly [number, number, number];
  directPageTableLevelZOffsets: Uint32Array<ArrayBuffer>;
}

/**
 * Live fixed-capacity node-mip atlas. Construction allocates empty capacity,
 * while every scene revision invalidates and repairs only its dirty slot
 * worklist. Unlike whole-atlas candidate generations, no update allocates or
 * exposes old opacity inside a dirty region.
 */
export class WebGpuLiveSvoNodeMipPyramid {
  readonly allocatedBytes: number;
  readonly pageState: WebGpuLiveSvoDerivedPageState;

  private readonly texture: GPUTexture;
  private readonly view: GPUTextureView;
  private readonly sampler: GPUSampler;
  private readonly directory: GPUBuffer;
  private readonly directoryTexture: GPUTexture;
  private readonly directoryView: GPUTextureView;
  private readonly directoryShape: { columns: number; rows: number; texels: readonly [number, number] };
  private readonly directPageTableTexture: GPUTexture;
  private readonly directPageTableView: GPUTextureView;
  private readonly directPageTableDimensions: readonly [number, number, number];
  private readonly directPageTableLevelZOffsets = new Uint32Array(WEBGPU_SVO_NODE_MIP_LAYOUT.directPageTableMaximumLevels);
  private readonly directPageTableWords: Uint32Array<ArrayBuffer>;
  private directoryLayoutKey = "";
  private directPageTableReady = false;
  private plan?: SvoNodeMipPyramidPlan;
  private pendingPlan?: SvoNodeMipPyramidPlan;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly options: WebGpuLiveSvoNodeMipOptions) {
    const { pageCapacity, atlasTexels } = options;
    if (!Number.isSafeInteger(pageCapacity) || pageCapacity <= 0) throw new RangeError("Live node-mip page capacity must be positive");
    if (atlasTexels.some((value) => !Number.isSafeInteger(value) || value <= 0)) throw new RangeError("Live node-mip atlas dimensions must be positive integers");
    if (atlasTexels.some((value) => value % SVO_NODE_MIP_LAYOUT.physicalSize !== 0)
      || atlasTexels.reduce((product, value) => product * (value / SVO_NODE_MIP_LAYOUT.physicalSize), 1) < pageCapacity) {
      throw new RangeError("Live node-mip atlas must be page-aligned and contain its declared capacity");
    }
    const direct = options.directPageTableDimensions ?? [1, 1, 1];
    if (direct.some((value) => !Number.isSafeInteger(value) || value <= 0)) throw new RangeError("Live node-mip direct-table dimensions must be positive integers");
    const label = options.label ?? "Live SVO node mips";
    this.texture = device.createTexture({ label: `${label} atlas`, size: atlasTexels, dimension: "3d", format: WEBGPU_SVO_NODE_MIP_LAYOUT.format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST });
    this.view = this.texture.createView({ dimension: "3d" });
    this.sampler = device.createSampler({ label: `${label} sampler`, addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge",
      addressModeW: "clamp-to-edge", ...WEBGPU_SVO_NODE_MIP_LAYOUT.sampler });
    this.directory = device.createBuffer({ label: `${label} directory`, size: pageCapacity * SVO_NODE_MIP_LAYOUT.directoryBytesPerPage,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.directoryShape = webGpuSvoNodeMipDirectoryShape(pageCapacity, device.limits?.maxTextureDimension2D ?? 2_048);
    this.directoryTexture = device.createTexture({ label: `${label} sampled directory`,
      size: [...this.directoryShape.texels], format: WEBGPU_SVO_NODE_MIP_LAYOUT.directoryTextureFormat,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.directoryView = this.directoryTexture.createView();
    this.directPageTableDimensions = direct;
    this.directPageTableWords = new Uint32Array(direct[0] * direct[1] * direct[2]);
    this.directPageTableTexture = device.createTexture({ label: `${label} direct page table`, size: direct, dimension: "3d",
      format: WEBGPU_SVO_NODE_MIP_LAYOUT.directPageTableTextureFormat, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.directPageTableView = this.directPageTableTexture.createView({ dimension: "3d" });
    this.pageState = new WebGpuLiveSvoDerivedPageState(device, pageCapacity, label);
    this.allocatedBytes = atlasTexels[0] * atlasTexels[1] * atlasTexels[2] * SVO_NODE_MIP_LAYOUT.bytesPerTexel
      + pageCapacity * SVO_NODE_MIP_LAYOUT.directoryBytesPerPage
      + this.directoryShape.columns * this.directoryShape.rows * SVO_NODE_MIP_LAYOUT.directoryBytesPerPage
      + direct[0] * direct[1] * direct[2] * 4 + this.pageState.allocatedBytes;
  }

  private uploadPlanLayout(plan: SvoNodeMipPyramidPlan): void {
    const layoutKey = plan.pages.map((page) => `${page.key.level}:${page.key.coordinate.join(",")}:${page.slot}:${page.atlasTexelOrigin.join(",")}`).join(";");
    if (layoutKey === this.directoryLayoutKey) return;
    const directoryWords = new Uint32Array(this.directoryShape.columns * this.directoryShape.rows * WEBGPU_SVO_NODE_MIP_LAYOUT.directoryWordsPerPage);
    // Directory generation is intentionally stable. Page-local validity is the
    // live generation authority, so transform/material edits never rewrite the
    // complete directory or direct table.
    for (const page of plan.pages) directoryWords.set(packDirectoryEntry({ ...page, key: { ...page.key, generation: 1 } }), page.slot * WEBGPU_SVO_NODE_MIP_LAYOUT.directoryWordsPerPage);
    this.device.queue.writeBuffer(this.directory, 0, directoryWords, 0, this.options.pageCapacity * WEBGPU_SVO_NODE_MIP_LAYOUT.directoryWordsPerPage);
    this.device.queue.writeTexture({ texture: this.directoryTexture }, directoryWords,
      { bytesPerRow: this.directoryShape.columns * SVO_NODE_MIP_LAYOUT.directoryBytesPerPage, rowsPerImage: this.directoryShape.rows },
      [...this.directoryShape.texels]);
    const direct = createWebGpuSvoNodeMipDirectPageTable(plan, Math.max(...this.directPageTableDimensions));
    this.directPageTableLevelZOffsets.fill(0);
    this.directPageTableLevelZOffsets.set(direct.levelZOffsets);
    this.directPageTableReady = direct.ready && direct.dimensions.every((value, axis) => value <= this.directPageTableDimensions[axis]);
    if (this.directPageTableReady) {
      this.directPageTableWords.fill(0);
      const [sourceWidth, sourceHeight, sourceDepth] = direct.dimensions;
      const [targetWidth, targetHeight] = this.directPageTableDimensions;
      for (let z = 0; z < sourceDepth; z += 1) for (let y = 0; y < sourceHeight; y += 1) {
        const source = (z * sourceHeight + y) * sourceWidth;
        const target = (z * targetHeight + y) * targetWidth;
        this.directPageTableWords.set(direct.words.subarray(source, source + sourceWidth), target);
      }
      this.device.queue.writeTexture({ texture: this.directPageTableTexture }, this.directPageTableWords,
        { bytesPerRow: targetWidth * 4, rowsPerImage: targetHeight }, this.directPageTableDimensions);
    }
    this.directoryLayoutKey = layoutKey;
  }

  /** Stage only page topology; a GPU builder owns payload and validity publication. */
  prepareGpuUpdate(plan: SvoNodeMipPyramidPlan): WebGpuLiveSvoNodeMipGpuTarget {
    this.assertAlive();
    if (plan.pages.length > this.options.pageCapacity) throw new RangeError("Live node-mip page capacity exceeded");
    if (plan.atlas.texels.some((value, axis) => value > this.options.atlasTexels[axis])) throw new RangeError("Live node-mip atlas capacity exceeded");
    this.pendingPlan = plan;
    this.uploadPlanLayout(plan);
    return this.gpuTarget();
  }

  /** Called after GPU build commands are encoded; page-validity writes remain the publication authority. */
  acceptGpuUpdate(generation: number): void {
    if (!this.pendingPlan || this.pendingPlan.generation !== generation) throw new Error("GPU node-mip completion does not match the staged plan");
    this.plan = this.pendingPlan; this.pendingPlan = undefined;
  }

  gpuTarget(): WebGpuLiveSvoNodeMipGpuTarget {
    const physical = SVO_NODE_MIP_LAYOUT.physicalSize;
    return { texture: this.texture, pageValidity: this.pageState.validity,
      atlasPages: this.options.atlasTexels.map((value) => Math.floor(value / physical)) as [number, number, number],
      pageCapacity: this.options.pageCapacity, directPageTableTexture: this.directPageTableTexture,
      directPageTableDimensions: this.directPageTableDimensions, directPageTableLevelZOffsets: this.directPageTableLevelZOffsets };
  }

  visibleGeneration(): WebGpuLiveSvoNodeMipVisibleGeneration | undefined {
    if (!this.plan) return undefined;
    return { generation: this.plan.generation, plan: this.plan, texture: this.texture, view: this.view, sampler: this.sampler,
      directory: this.directory, directoryTexture: this.directoryTexture, directoryView: this.directoryView,
      directPageTableTexture: this.directPageTableTexture, directPageTableView: this.directPageTableView,
      directPageTableDimensions: this.directPageTableDimensions, directPageTableLevelZOffsets: this.directPageTableLevelZOffsets,
      directPageTableReady: this.directPageTableReady, directPageTableBytes: this.directPageTableReady ? this.directPageTableDimensions.reduce((a, b) => a * b, 4) : 0,
      pageValidity: this.pageState.validity };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.texture.destroy(); this.directory.destroy(); this.directoryTexture.destroy(); this.directPageTableTexture.destroy(); this.pageState.destroy();
    this.destroyed = true;
  }

  private assertAlive(): void { if (this.destroyed) throw new Error("Live SVO node-mip pyramid is destroyed"); }
}
