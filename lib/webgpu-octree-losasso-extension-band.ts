import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import type { PassBroker } from "./webgpu-pass-broker";
import {
  WebGPUOctreeLosassoVelocityExtension,
  OCTREE_LOSASSO_EXTENSION_WIDTH,
  type WebGPUOctreeLosassoVelocityExtensionSource,
} from "./webgpu-octree-losasso-velocity-extension";
import type { WebGPUOctreeLosassoVelocitySamplerSource } from
  "./webgpu-octree-losasso-velocity-sampler";
import { octreeLosassoExtensionBandWGSL } from
  "./webgpu-octree-losasso-extension-band.wgsl";

export const OCTREE_LOSASSO_EXTENSION_ADJACENCIES = 6;

export interface WebGPUOctreeLosassoWetFaceSource {
  readonly control: GPUBuffer;
  readonly faceGeometry: GPUBuffer;
  readonly axisFaceDirectory: GPUBuffer;
  readonly directoryCapacity: number;
  readonly projectedVelocity: GPUBuffer;
  readonly extendedVelocity: GPUBuffer;
}

export interface WebGPUOctreeLosassoExtensionBandOptions {
  readonly dimensions: readonly [number, number, number];
  readonly maximumLeafSize: number;
  readonly cellSize: number;
  readonly domainOrigin?: readonly [number, number, number];
  readonly wetFaceCapacity: number;
  readonly maximumResidentFineBricks: number;
  readonly wet: WebGPUOctreeLosassoWetFaceSource;
}

export interface OctreeLosassoExtensionBandPlan {
  readonly dimensions: readonly [number, number, number];
  readonly faceCapacity: number;
  readonly directoryCapacity: number;
  readonly adjacencyCapacity: number;
  readonly allocatedBytes: number;
}

export function planOctreeLosassoExtensionBand(
  dimensions: readonly [number, number, number],
  wetFaceCapacity: number,
  maximumResidentFineBricks: number,
): OctreeLosassoExtensionBandPlan {
  for (const [axis, value] of dimensions.entries()) {
    if (!Number.isSafeInteger(value) || value < 1) {
      throw new RangeError(`Losasso extension-band dimension ${axis} must be positive`);
    }
  }
  for (const [label, value] of [["wet face", wetFaceCapacity],
    ["fine brick", maximumResidentFineBricks]] as const) {
    if (!Number.isSafeInteger(value) || value < 1) {
      throw new RangeError(`Losasso extension-band ${label} capacity must be positive`);
    }
  }
  // Air records are finest-lattice MAC faces and geometric claims deduplicate
  // them globally.  Bound the compact arena by the complete finest MAC
  // lattice, not by a per-page proposal estimate: a B4 page spans two coarse
  // cells per axis, so its trilinear footprint alone is 144 faces rather than
  // the 54-face footprint of a single coarse cell.  The full lattice is both
  // tighter and an actual hard bound after the W7 closure dispatches.
  const [nx, ny, nz] = dimensions;
  const finestMacFaces = (nx + 1) * ny * nz
    + nx * (ny + 1) * nz
    + nx * ny * (nz + 1);
  const faceCapacity = wetFaceCapacity + finestMacFaces;
  if (!Number.isSafeInteger(faceCapacity) || faceCapacity > 0xffff_ffff) {
    throw new RangeError("Losasso compact extension-band face count exceeds u32");
  }
  let directoryCapacity = 1;
  while (directoryCapacity < 2 * faceCapacity) directoryCapacity *= 2;
  const adjacencyCapacity = OCTREE_LOSASSO_EXTENSION_ADJACENCIES * faceCapacity;
  const allocatedBytes = 32 + 112 + faceCapacity * (16 + 16 + 4 + 4 + 4)
    + (faceCapacity + 1) * 4 + adjacencyCapacity * 4 + directoryCapacity * 12;
  return Object.freeze({ dimensions: [dimensions[0], dimensions[1], dimensions[2]] as const,
    faceCapacity, directoryCapacity,
    adjacencyCapacity, allocatedBytes });
}

const ENTRY_POINTS = [
  "beginLosassoExtensionBand",
  "publishLosassoWetSeedFaces",
  "publishLosassoAirBandFaces",
  "dilateLosassoAirBand5",
  "dilateLosassoAirBand6",
  "dilateLosassoAirBand7",
  "buildLosassoExtensionAdjacency",
  "finishLosassoExtensionBand",
  "gatherLosassoProjectedSeeds",
  "publishLosassoWetExtended",
] as const;
type EntryPoint = typeof ENTRY_POINTS[number];

const BINDINGS: Readonly<Record<EntryPoint, readonly number[]>> = Object.freeze({
  beginLosassoExtensionBand: [0, 4, 8, 9, 13, 14, 16, 17],
  publishLosassoWetSeedFaces: [0, 4, 5, 8, 9, 10, 13, 16, 17],
  publishLosassoAirBandFaces: [0, 1, 2, 3, 8, 9, 10, 13, 17],
  dilateLosassoAirBand5: [0, 8, 9, 10, 13, 17],
  dilateLosassoAirBand6: [0, 8, 9, 10, 13, 17],
  dilateLosassoAirBand7: [0, 8, 9, 10, 13, 17],
  buildLosassoExtensionAdjacency: [0, 8, 10, 11, 12, 13],
  finishLosassoExtensionBand: [0, 4, 8],
  gatherLosassoProjectedSeeds: [0, 7, 8, 14, 16],
  publishLosassoWetExtended: [4, 7, 15],
});

/**
 * Owns the finest-lattice W=7 air-face graph. It is separate from the pressure
 * face authority: air support can therefore never become a matrix incidence.
 */
export class WebGPUOctreeLosassoExtensionBand {
  readonly plan: OctreeLosassoExtensionBandPlan;
  readonly source: WebGPUOctreeLosassoVelocityExtensionSource;
  readonly samplerSource: WebGPUOctreeLosassoVelocitySamplerSource;
  readonly allocatedBytes: number;
  readonly width = OCTREE_LOSASSO_EXTENSION_WIDTH;
  readonly publication = "compact-fine-brick-W7-same-axis-band" as const;

  private readonly params: GPUBuffer;
  private readonly control: GPUBuffer;
  private readonly metrics: GPUBuffer;
  private readonly geometry: GPUBuffer;
  private readonly adjacencyOffsets: GPUBuffer;
  private readonly adjacency: GPUBuffer;
  private readonly directory: GPUBuffer;
  private readonly projectedSeeds: GPUBuffer;
  private readonly extended: GPUBuffer;
  private readonly seedWetFace: GPUBuffer;
  private readonly geometricClaims: GPUBuffer;
  private readonly extension: WebGPUOctreeLosassoVelocityExtension;
  private readonly bindGroupCache = new Map<GPUComputePipeline,
    { entries: readonly { binding: number; buffer: GPUBuffer }[]; group: GPUBindGroup }[]>();
  private pipelines?: Readonly<Record<EntryPoint, GPUComputePipeline>>;
  private publishedFineGeneration = 0;
  private destroyed = false;

  constructor(private readonly device: GPUDevice,
    private readonly options: WebGPUOctreeLosassoExtensionBandOptions) {
    this.plan = planOctreeLosassoExtensionBand(options.dimensions,
      options.wetFaceCapacity, options.maximumResidentFineBricks);
    if (!Number.isSafeInteger(options.maximumLeafSize) || options.maximumLeafSize < 1
      || (options.maximumLeafSize & (options.maximumLeafSize - 1)) !== 0) {
      throw new RangeError("Losasso extension-band maximum leaf size must be dyadic");
    }
    const [nx, ny, nz] = options.dimensions;
    const perLevel = (nx + 1) * ny * nz + nx * (ny + 1) * nz + nx * ny * (nz + 1);
    if (perLevel * (1 + Math.log2(options.maximumLeafSize)) > 0xffff_fffe) {
      throw new RangeError("Losasso extension-band geometric keys exceed collision-free u32 range");
    }
    if (!Number.isFinite(options.cellSize) || options.cellSize <= 0) {
      throw new RangeError("Losasso extension-band cell size must be positive and finite");
    }
    if (!Number.isSafeInteger(options.wet.directoryCapacity)
      || options.wet.directoryCapacity < 1
      || (options.wet.directoryCapacity & (options.wet.directoryCapacity - 1)) !== 0) {
      throw new RangeError("Losasso wet-face directory capacity must be a power of two");
    }
    const maxStorage = device.limits.maxStorageBufferBindingSize;
    const sizes = [this.plan.faceCapacity * 16, this.plan.adjacencyCapacity * 4,
      this.plan.directoryCapacity * 8, this.plan.directoryCapacity * 4];
    if (sizes.some((size) => size > maxStorage)) {
      throw new RangeError("Losasso extension band exceeds a WebGPU storage binding limit");
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const make = (label: string, size: number) => device.createBuffer({ label, size: Math.max(16, size), usage: storage });
    this.params = device.createBuffer({ label: "Losasso extension-band parameters", size: 112,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.control = make("Losasso extension-band authority", 32);
    this.metrics = make("Losasso extension-band metrics", this.plan.faceCapacity * 16);
    this.geometry = make("Losasso extension-band finest face geometry", this.plan.faceCapacity * 16);
    this.adjacencyOffsets = make("Losasso extension-band adjacency offsets", (this.plan.faceCapacity + 1) * 4);
    this.adjacency = make("Losasso extension-band deterministic adjacency", this.plan.adjacencyCapacity * 4);
    this.directory = make("Losasso extension-band sampler directory", this.plan.directoryCapacity * 8);
    this.projectedSeeds = make("Losasso extension-band projected seeds", this.plan.faceCapacity * 4);
    this.extended = make("Losasso extension-band extended velocity", this.plan.faceCapacity * 4);
    this.seedWetFace = make("Losasso extension-band wet seed mapping", this.plan.faceCapacity * 4);
    this.geometricClaims = make("Losasso extension-band geometric key claims",
      this.plan.directoryCapacity * 4);
    this.source = Object.freeze({ faceCapacity: this.plan.faceCapacity, control: this.control,
      faceMetrics: this.metrics, adjacencyOffsets: this.adjacencyOffsets,
      adjacencyFaces: this.adjacency, projectedVelocity: this.projectedSeeds,
      extendedVelocity: this.extended });
    this.samplerSource = Object.freeze({ control: this.control, faceGeometry: this.geometry,
      extendedVelocity: this.extended, axisFaceDirectory: this.directory,
      directoryCapacity: this.plan.directoryCapacity, dimensions: options.dimensions,
      maximumLeafSize: options.maximumLeafSize, fineCellSize: options.cellSize });
    this.extension = new WebGPUOctreeLosassoVelocityExtension(device, this.source);
    this.allocatedBytes = this.plan.allocatedBytes + this.extension.allocatedBytes;
  }

  get initializationTasks(): readonly { readonly label: string; readonly run: () => Promise<void> }[] {
    return [{ label: "Compile Losasso W=7 extension-band publisher", run: () => this.initialize() },
      ...this.extension.initializationTasks];
  }

  async initialize(): Promise<void> {
    this.assertLive();
    if (!this.pipelines) {
      const shaderModule = this.device.createShaderModule({ label: "Losasso W=7 extension-band shader",
        code: octreeLosassoExtensionBandWGSL });
      const pairs: [EntryPoint, GPUComputePipeline][] = [];
      for (const entryPoint of ENTRY_POINTS) {
        pairs.push([entryPoint, await this.device.createComputePipelineAsync({
          label: `Losasso extension band - ${entryPoint}`, layout: "auto",
          compute: { module: shaderModule, entryPoint },
        })]);
      }
      this.pipelines = Object.freeze(Object.fromEntries(pairs)) as Readonly<Record<EntryPoint, GPUComputePipeline>>;
    }
    await this.extension.initialize();
  }

  private cachedBindGroup(pipeline: GPUComputePipeline,
    entries: readonly { binding: number; buffer: GPUBuffer }[]): GPUBindGroup {
    const variants = this.bindGroupCache.get(pipeline) ?? [];
    const cached = variants.find((variant) => variant.entries.length === entries.length
      && variant.entries.every((entry, index) => entry.binding === entries[index]!.binding
        && entry.buffer === entries[index]!.buffer));
    if (cached) return cached.group;
    const stableEntries = entries.map((entry) => ({ ...entry }));
    const group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: stableEntries.map(({ binding, buffer }) =>
        ({ binding, resource: { buffer } })) });
    variants.push({ entries: stableEntries, group }); this.bindGroupCache.set(pipeline, variants);
    return group;
  }

  /** Publish geometry, air classification, graph layers, seeds, and sampler lookup. */
  encodePublication(broker: PassBroker, fine: WebGPUFineLevelSetBrickSource): void {
    this.assertReady();
    const finePlan = fine.plan;
    if (finePlan.finestCellDimensions.some((value, axis) => value !== this.plan.dimensions[axis])) {
      throw new RangeError("Losasso extension band and fine phi have different velocity domains");
    }
    if (finePlan.maximumResidentBricks > this.options.maximumResidentFineBricks) {
      throw new RangeError("Losasso fine brick publication exceeds extension-band headroom");
    }
    const bytes = new ArrayBuffer(112), words = new Uint32Array(bytes), floats = new Float32Array(bytes);
    words.set([...this.plan.dimensions, this.options.maximumLeafSize], 0);
    words.set([this.plan.faceCapacity, this.plan.directoryCapacity,
      this.options.wet.directoryCapacity, OCTREE_LOSASSO_EXTENSION_WIDTH], 4);
    floats.set([...(this.options.domainOrigin ?? [0, 0, 0]), this.options.cellSize], 8);
    words.set([...finePlan.brickDimensions, finePlan.brickResolution], 12);
    words.set([...finePlan.sampleDimensions, finePlan.samplesPerBrick], 16);
    floats.set([...finePlan.domainOrigin, finePlan.fineCellWidth], 20);
    words.set([finePlan.maximumResidentBricks, finePlan.maximumResidentBricks, 7, fine.generation], 24);
    this.device.queue.writeBuffer(this.params, 0, bytes);
    const buffers = new Map<number, GPUBuffer>([
      [0, this.params], [1, fine.metadata], [2, fine.worklist], [3, fine.samples],
      [4, this.options.wet.control], [5, this.options.wet.faceGeometry],
      [6, this.options.wet.axisFaceDirectory], [7, this.options.wet.projectedVelocity],
      [8, this.control], [9, this.metrics], [10, this.geometry],
      [11, this.adjacencyOffsets], [12, this.adjacency], [13, this.directory],
      [14, this.projectedSeeds], [15, this.options.wet.extendedVelocity],
      [16, this.seedWetFace],
      [17, this.geometricClaims],
    ]);
    const run = (entryPoint: EntryPoint, groups: number) => {
      const pipeline = this.pipelines![entryPoint];
      const pass = broker.compute({ label: `Losasso extension band - ${entryPoint}` });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.cachedBindGroup(pipeline,
        BINDINGS[entryPoint].map((binding) => ({ binding, buffer: buffers.get(binding)! }))));
      const groupsX = Math.min(groups, 65_535), groupsY = Math.ceil(groups / groupsX);
      if (groupsY > 65_535) throw new RangeError("Losasso compact band exceeds 2D dispatch limits");
      pass.dispatchWorkgroups(groupsX, groupsY);
    };
    run("beginLosassoExtensionBand", Math.ceil(Math.max(this.plan.faceCapacity,
      this.plan.directoryCapacity) / 64));
    run("publishLosassoWetSeedFaces", Math.ceil(this.options.wetFaceCapacity / 64));
    const coarseCellsPerBrick = Math.ceil(finePlan.brickResolution / finePlan.fineFactor);
    const macFacesPerBrick = 3 * (coarseCellsPerBrick + 1) * (coarseCellsPerBrick + 2) ** 2;
    run("publishLosassoAirBandFaces",
      Math.ceil(macFacesPerBrick * finePlan.maximumResidentBricks / 64));
    run("dilateLosassoAirBand5", Math.ceil(this.plan.faceCapacity / 64));
    run("dilateLosassoAirBand6", Math.ceil(this.plan.faceCapacity / 64));
    // A finest MAC trilinear corner can be three face-adjacency (L1) steps
    // from the signed W4 transport core. The old two closures advertised W7
    // but only guaranteed W6, leaving diagonal/corner traces without velocity
    // coverage and biasing an otherwise D4-exact front toward the grid axes.
    run("dilateLosassoAirBand7", Math.ceil(this.plan.faceCapacity / 64));
    run("buildLosassoExtensionAdjacency", Math.ceil(this.plan.faceCapacity / 64));
    run("finishLosassoExtensionBand", 1);
    this.publishedFineGeneration = fine.generation;
  }

  /** Gather current projected wet seeds, run exactly K=8, then publish the wet view. */
  encodeOncePerAdvance(broker: PassBroker, advance: number, topologyEpoch: number): boolean {
    this.assertReady();
    if (this.publishedFineGeneration < 1) {
      throw new Error("Losasso extension band must be published before extension");
    }
    const gather = this.pipelines!.gatherLosassoProjectedSeeds;
    let pass = broker.compute({ label: "Losasso extension band - gather projected seeds" });
    pass.setPipeline(gather); pass.setBindGroup(0, this.cachedBindGroup(gather, [
      { binding: 0, buffer: this.params },
      { binding: 7, buffer: this.options.wet.projectedVelocity },
      { binding: 8, buffer: this.control },
      { binding: 14, buffer: this.projectedSeeds },
      { binding: 16, buffer: this.seedWetFace },
    ]));
    const groups = Math.ceil(this.plan.faceCapacity / 64);
    pass.dispatchWorkgroups(Math.min(groups, 65_535), Math.ceil(groups / Math.min(groups, 65_535)));
    const encoded = this.extension.encodeOncePerAdvance(broker, advance, topologyEpoch);
    if (!encoded) return false;
    const publish = this.pipelines!.publishLosassoWetExtended;
    pass = broker.compute({ label: "Losasso extension band - publish wet velocity view" });
    pass.setPipeline(publish); pass.setBindGroup(0, this.cachedBindGroup(publish, [
      { binding: 4, buffer: this.options.wet.control },
      { binding: 7, buffer: this.options.wet.projectedVelocity },
      { binding: 15, buffer: this.options.wet.extendedVelocity },
    ]));
    pass.dispatchWorkgroups(Math.ceil(this.options.wet.projectedVelocity.size / 4 / 64));
    return true;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.extension.destroy();
    for (const buffer of [this.params, this.control, this.metrics, this.geometry,
      this.adjacencyOffsets, this.adjacency, this.directory, this.projectedSeeds,
      this.extended, this.seedWetFace, this.geometricClaims]) buffer.destroy();
    this.bindGroupCache.clear();
    this.pipelines = undefined;
  }
  private assertReady(): void { this.assertLive(); if (!this.pipelines) throw new Error("Losasso extension band is not initialized"); }
  private assertLive(): void { if (this.destroyed) throw new Error("Losasso extension band is destroyed"); }
}
