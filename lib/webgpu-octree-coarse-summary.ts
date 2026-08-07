import type { GPUInitializationTask } from "./gpu-initialization";
import type { OctreePowerCoarseLevelSetSampleSource } from "./webgpu-octree-power-coarse-levelset";
import type { WebGPUOctreeLosassoVelocitySamplerSource } from
  "./webgpu-octree-losasso-velocity-sampler";
import { octreeLosassoStagedVelocitySamplingWGSL } from
  "./webgpu-octree-losasso-velocity-sampler.wgsl";
import { OCTREE_AIR_SUPPORT_LAYOUT_VERSION, OCTREE_AIR_SUPPORT_OWNER_HASH,
  OCTREE_AIR_SUPPORT_TAG, OCTREE_AIR_SUPPORT_VALID, octreeAirSupportOwnerHashStartWGSL,
  type OctreeAirVelocitySupportLayout } from "./webgpu-octree-air-velocity-support";
import { PassBroker } from "./webgpu-pass-broker";

const PAGE_SIZE = 32;
const ENTRY_WORDS = 12;
const coarseSummaryLosassoVelocitySamplingWGSL = octreeLosassoStagedVelocitySamplingWGSL
  .replace(/\bcoarse\b/g, "losassoControl")
  .replace(/\bfaceGeometry\b/g, "losassoFaceGeometry")
  .replace(/\bextendedVelocity\b/g, "losassoExtendedVelocity")
  .replace(/\bfaceDirectory\b/g, "losassoFaceDirectory")
  .replace(/\bstagedVelocity\b/g, "losassoStagedVelocity")
  .replace(/p\.velocityDimensions/g, "p.dims")
  .replace(/p\.directoryCapacity/g, "u32(arrayLength(&losassoFaceDirectory))")
  .replace(/p\.closed/g, "1u")
  .replace(/p\.openTop/g, "u32(p.time.w)")
  .replace(/p\.domainOrigin/g, "vec3f(0)")
  .replace(/p\.velocityCellSize/g, "p.physicalCellSize");

/**
 * Fraction of the coarse tracker's trilinear velocity stencil that must
 * resolve to a published air-support owner before the sample is usable.
 *
 * One means all eight corners, which is the historical rule and the only
 * value that is bit-identical to it. Anything lower renormalizes by the
 * covered weight, which lets the tracker keep advecting where the
 * proven-reach corridor is thinner than the stencil. Construction-stable and
 * authored, never inferred from the corridor.
 */
export function coarseSummaryVelocityStencilCoverage(
  environment: Readonly<Record<string, string | undefined>> | undefined =
    typeof process !== "undefined" ? process.env : undefined,
): number {
  const authored = Number(environment?.FLUID_COARSE_VELOCITY_STENCIL_COVERAGE ?? 1);
  if (!Number.isFinite(authored) || authored <= 0 || authored > 1) return 0.999;
  // The historical literal, kept exactly: eight canonical weights summing to
  // one do not reproduce 1.0 bitwise.
  return authored >= 1 ? 0.999 : authored;
}

/** GPU-authored launch shapes for the recurring coarse-only summary build.
 *
 * A control singleton publishes the dense-lattice, live-row, and live-entry
 * dispatch records before the build runs, so an unpublished air support zeroes
 * the twelve dense sweeps on-GPU and the row/entry stages launch from
 * `coarse.rowCount`/`state[1]` instead of allocation capacities. Launch shape
 * only: an executed invocation computes exactly what the direct launch did.
 * Default OFF until the Gate A A/B lands. */
export function octreeCoarseSummaryIndirectDispatchEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_COARSE_SUMMARY_INDIRECT_DISPATCH === "1";
}

function coarseSummaryDiagnosticsEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_COARSE_SUMMARY_DIAGNOSTICS === "1"
    || resolved?.FLUID_FIELD_STATS === "1";
}

/** Legacy whole-domain phi inflation, retained only as an attribution arm. */
export function coarseSummaryGlobalVolumeCorrectionEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_COARSE_GLOBAL_VOLUME_CORRECTION === "1";
}

export interface OctreeCoarseSummaryPlan {
  readonly baseDimensions: readonly [number, number, number];
  readonly levelDimensions: readonly (readonly [number, number, number])[];
  readonly levelOffsets: readonly number[];
  readonly hierarchyKeyCapacity: number;
  readonly topLevelPageCount: number;
  readonly directoryPageCapacity: number;
  readonly entryCapacity: number;
  readonly entryOffsetWords: number;
  readonly phiOffsetWords: number;
  readonly directoryWords: number;
  readonly allocatedBytes: number;
}

/**
 * Eikonal sweeps per advance, from the reach the refinement ladder needs.
 *
 * `seedDenseRedistance` writes one domain extent everywhere the interface does
 * not cut, and `sweepDenseRedistance` only ever takes a `min`. That is textbook
 * fast marching from +infinity and it is monotone: after k Jacobi sweeps every
 * cell within k finest cells of the zero set carries its true distance, and
 * every cell still reading the seed is provably farther than k. So the seed is
 * an honest "deeper than k", NOT a fallback -- provided k covers every distance
 * a consumer will ask about.
 *
 * It did not. Five sweeps were hard-coded while
 * `pressureRefinementEvidence` reads |phi| as a distance out to
 * `bandCells + gradingLayers * maximumLeafSize` cells. Everything past five
 * cells reported the same constant, so the ladder coarsened on a number that
 * carried no depth at all.
 *
 * Seeding from the ADVECTED magnitude instead is the tempting cheap fix and it
 * is wrong: a min-only sweep can lower a value but never raise one, so an
 * advected field that has been compressed can only decay further, and the
 * interior coarsening bleeds away over the run rather than holding.
 *
 * The count is clamped to the longest axis -- no sweep can carry information
 * farther than the lattice -- and kept odd so the last one lands in the output
 * bank. The floor of five preserves the historical reach for any caller that
 * declares no requirement.
 */
export function planOctreeRedistanceSweeps(
  reachCells: number | undefined,
  dimensions: readonly [number, number, number],
): number {
  const longestAxis = Math.max(dimensions[0], dimensions[1], dimensions[2]);
  const requested = Number.isFinite(reachCells) ? Math.ceil(reachCells!) : 5;
  const bounded = Math.max(5, Math.min(requested, longestAxis));
  return bounded % 2 === 0 ? bounded + 1 : bounded;
}

export function planOctreeCoarseSummary(
  dimensions: readonly [number, number, number],
  rowCapacity: number,
): OctreeCoarseSummaryPlan {
  const baseDimensions = dimensions.map((value) => Math.ceil(value / 4)) as [number, number, number];
  const levelDimensions: Array<readonly [number, number, number]> = [];
  const levelOffsets: number[] = [];
  let current = baseDimensions, hierarchyKeyCapacity = 0;
  while (true) {
    levelOffsets.push(hierarchyKeyCapacity); levelDimensions.push(current);
    hierarchyKeyCapacity += current[0] * current[1] * current[2];
    if (current.every((value) => value === 1)) break;
    current = current.map((value) => Math.ceil(value / 2)) as [number, number, number];
  }
  const topLevelPageCount = Math.ceil(hierarchyKeyCapacity / PAGE_SIZE);
  const entryCapacity = Math.min(hierarchyKeyCapacity, rowCapacity * levelDimensions.length);
  const directoryPageCapacity = Math.min(topLevelPageCount, Math.max(1, entryCapacity));
  const entryOffsetWords = 16 + topLevelPageCount + directoryPageCapacity * PAGE_SIZE;
  const phiOffsetWords = entryOffsetWords + Math.max(1, entryCapacity) * ENTRY_WORDS;
  const domainVolume = dimensions[0] * dimensions[1] * dimensions[2];
  const directoryWords = phiOffsetWords + 3 * domainVolume;
  return {
    baseDimensions, levelDimensions, levelOffsets, hierarchyKeyCapacity,
    topLevelPageCount, directoryPageCapacity, entryCapacity, entryOffsetWords,
    phiOffsetWords, directoryWords,
    // Directory + state + params + indirect dispatch + optional-binding sentinel.
    allocatedBytes: directoryWords * 4 + 128 + 112 + 36 + 128,
  };
}

/** Compact coarse-lattice tracker and B4-and-parent hierarchy for factor-one.
 * It owns one value per coarse cell plus redistance scratch, but allocates no
 * fine samples, fine-band pages, fine topology, or fine transport worklists. */
export class WebGPUOctreeCoarseSummary {
  readonly plan: OctreeCoarseSummaryPlan;
  readonly directory: GPUBuffer;
  private readonly domainVolume: number;
  /** Eikonal sweeps per advance; odd, so the last one lands in the output bank. */
  private readonly redistanceSweeps: number;
  private readonly state: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly dispatchArgs: GPUBuffer;
  private readonly bindingSentinel: GPUBuffer;
  private readonly indirectDispatch = octreeCoarseSummaryIndirectDispatchEnabled();
  private readonly pipelines: Record<string, GPUComputePipeline> = {};
  private readonly bindGroups = new Map<GPUComputePipeline, GPUBindGroup>();
  private module?: GPUShaderModule;
  private hierarchyInitialized = false;
  private destroyed = false;
  private readonly entries = ["resetSummary", "resetSummaryValues", "ensureSummaryPages", "ensureSupportSummaryPages",
    "ensureSummaryRanks", "ensureSupportSummaryRanks", "predictSummaryCells",
    "seedDenseRedistance", "redistanceScratchToOutput", "redistanceOutputToScratch",
    "summarizeDenseVolume", "prepareVolumeCorrection", "correctAndAggregateSummaryCells",
    "finalizeSummaryEntries", "publishSummary",
    "publishDenseComplement", "correctCoarseDirectory",
    "prepareSummaryDispatch", "prepareSummaryFinalizeDispatch"] as const;

  constructor(private readonly device: GPUDevice,
    private readonly coarse: Pick<OctreePowerCoarseLevelSetSampleSource, "directory" | "rowCapacity">,
    dimensions: readonly [number, number, number],
    private readonly air: Readonly<{ arena: GPUBuffer; layout: OctreeAirVelocitySupportLayout;
      rowVelocities: GPUBuffer; initialPhi: Float32Array; physicalCellSize: number; timestep_s: number;
      losassoVelocity?: WebGPUOctreeLosassoVelocitySamplerSource;
      openTopBoundary?: boolean;
      /** Largest authored octree leaf. The owner lookup probes dyadic
       * identities from here down to one, so a value below the real maximum
       * silently misses every coarser leaf. */
      maximumLeafSize: number;
      /**
       * How far, in finest cells, the published phi must be a real distance.
       *
       * The sweep below is min-only fast marching from a seed one domain extent
       * wide, so its reach IS the sweep count: after k sweeps a cell at true
       * distance <= k has converged and every other cell still reports the
       * seed. That is a correct algorithm and a correct sentinel -- as long as
       * k covers every distance a consumer will ask about.
       *
       * `pressureRefinementEvidence` asks about distances out to its widest
       * protection width, `bandCells + gradingLayers * maximumLeafSize`, so
       * that is what the caller passes. Five sweeps against a width of twenty
       * meant the refinement ladder read a CONSTANT wherever the surface was
       * more than five cells away, and coarsened the dam-break front to the
       * ceiling three cells from the free surface.
       */
      redistanceReachCells?: number }>,
    _deferPipelineCompilation = true) {
    if (!Number.isSafeInteger(air.maximumLeafSize) || air.maximumLeafSize < 1
      || (air.maximumLeafSize & (air.maximumLeafSize - 1)) !== 0) {
      throw new RangeError("Coarse-only tracker requires a power-of-two maximum leaf size");
    }
    this.plan = planOctreeCoarseSummary(dimensions, coarse.rowCapacity);
    this.domainVolume = dimensions[0] * dimensions[1] * dimensions[2];
    this.redistanceSweeps = planOctreeRedistanceSweeps(
      air.redistanceReachCells, dimensions);
    const maximumBinding = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    if (this.plan.directoryWords * 4 > maximumBinding) {
      throw new RangeError("Coarse-only summary hierarchy exceeds the device storage binding limit");
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.directory = device.createBuffer({ label: "coarse-only direct summary hierarchy",
      size: this.plan.directoryWords * 4, usage: storage });
    // 32 words. The receipt tallies live at 19/20, and a runtime-sized
    // `array<atomic<u32>>` silently drops out-of-bounds atomics rather than
    // faulting, so an 80-byte buffer made `state[20]` a write into nowhere.
    this.state = device.createBuffer({ label: "coarse-only summary build state", size: 128, usage: storage });
    this.params = device.createBuffer({ label: "coarse-only summary parameters", size: 112,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // Records: 0 dense lattice, 1 live coarse rows, 2 live summary entries.
    this.dispatchArgs = device.createBuffer({ label: "coarse-only summary dispatch arguments",
      size: 36, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    this.bindingSentinel = device.createBuffer({ label: "coarse-only summary optional binding sentinel",
      size: 128, usage: GPUBufferUsage.STORAGE });
    const words = new Uint32Array(16);
    words.set(this.plan.baseDimensions, 0);
    words.set(dimensions, 4);
    words[7] = air.maximumLeafSize;
    words.set([this.plan.hierarchyKeyCapacity, this.plan.topLevelPageCount,
      this.plan.directoryPageCapacity, this.plan.entryCapacity,
      this.plan.entryOffsetWords, this.plan.levelDimensions.length - 1,
      coarse.rowCapacity, this.plan.directoryWords], 8);
    const data = new ArrayBuffer(112); new Uint32Array(data, 0, 16).set(words);
    const tail = new Uint32Array(data, 64, 4);
    // The last word is the dense coarse lattice's own cardinality. It used to
    // be spelled `ownerDirectoryCellCapacity` because the owner directory was
    // dense and the two were the same number; they are not related now.
    tail.set([air.layout.controlOffsetWords, air.layout.supportVectorOffsetWords,
      air.layout.ownerDirectoryOffsetWords, dimensions[0] * dimensions[1] * dimensions[2]]);
    const time = new Float32Array(data, 80, 4);
    time[0] = air.timestep_s;
    time[1] = coarseSummaryVelocityStencilCoverage();
    time[2] = air.losassoVelocity ? 1 : 0;
    time[3] = air.openTopBoundary ? 1 : 0;
    // The prepare singleton reproduces the host's 2D dispatch split exactly.
    new Uint32Array(data, 96, 1)[0] = device.limits.maxComputeWorkgroupsPerDimension;
    new Float32Array(data, 100, 1)[0] = air.physicalCellSize;
    new Uint32Array(data, 104, 2).set([
      coarseSummaryDiagnosticsEnabled() ? 1 : 0,
      coarseSummaryGlobalVolumeCorrectionEnabled() ? 1 : 0,
    ]);
    device.queue.writeBuffer(this.params, 0, data);
    if (air.initialPhi.length !== dimensions[0] * dimensions[1] * dimensions[2]) {
      throw new RangeError("Coarse-only tracker bootstrap must cover the complete coarse lattice");
    }
    // `Float32Array` inputs may be typed over `ArrayBufferLike` (and therefore
    // potentially shared), while WebGPU uploads require an `ArrayBuffer`
    // backed view.  Take an owned snapshot here; bootstrap is construction-only
    // and the directory must not observe a concurrently mutable source anyway.
    const initialPhi = new Float32Array(air.initialPhi);
    device.queue.writeBuffer(this.directory, this.plan.phiOffsetWords * 4, initialPhi);
    const initialState = new Uint32Array(20);
    const referenceCells = air.initialPhi.reduce((sum, phi) =>
      sum + Math.max(0, Math.min(1, 0.5 - phi / air.physicalCellSize)), 0);
    initialState[14] = Math.round(4096 * referenceCells);
    initialState[16] = 1;
    device.queue.writeBuffer(this.state, 0, initialState);
  }

  private descriptor(entryPoint: string): GPUComputePipelineDescriptor {
    this.module ??= this.device.createShaderModule({ label: "coarse-only summary hierarchy",
      code: coarseSummaryWGSL });
    return { label: `coarse-only ${entryPoint}`, layout: "auto",
      compute: { module: this.module, entryPoint } };
  }
  initializationTasks(): GPUInitializationTask[] {
    if (Object.keys(this.pipelines).length !== 0) return [];
    return this.entries.map((entryPoint) => ({
      id: `octree.coarse-summary.pipeline.${entryPoint}`,
      phase: "adaptive-topology" as const,
      label: `Compile coarse-only summary · ${entryPoint}`,
      run: async () => { this.pipelines[entryPoint] =
        await this.device.createComputePipelineAsync(this.descriptor(entryPoint)); },
    }));
  }

  encode(broker: PassBroker): void {
    if (this.destroyed) throw new Error("Coarse-only summary hierarchy is destroyed");
    const dispatch = (entry: typeof this.entries[number], items: number, record?: 0 | 1 | 2) => {
      const groups = Math.ceil(Math.max(1, items) / 256);
      const width = Math.min(groups, this.device.limits.maxComputeWorkgroupsPerDimension);
      const prepare = entry === "prepareSummaryDispatch" || entry === "prepareSummaryFinalizeDispatch";
      const usesCoarse = entry === "ensureSummaryPages" || entry === "ensureSummaryRanks"
        || entry === "predictSummaryCells"
        || entry === "publishSummary"
        || entry === "publishDenseComplement"
        || entry === "correctCoarseDirectory"
        || entry === "prepareSummaryDispatch";
      const usesAir = entry === "predictSummaryCells" || entry === "prepareSummaryDispatch";
      const usesVelocity = entry === "predictSummaryCells";
      const usesState = true;
      const usesLosassoControl = entry === "prepareSummaryDispatch";
      const pipeline = this.pipelines[entry]!;
      const group = this.bindGroups.get(pipeline) ?? this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          ...(prepare ? [] : [{ binding: 0, resource: { buffer: this.directory } }]),
          ...(usesCoarse ? [
            { binding: 1, resource: { buffer: this.coarse.directory } },
          ] : []),
          ...(usesState ? [{ binding: 2, resource: { buffer: this.state } }] : []),
          { binding: 3, resource: { buffer: this.params } },
          ...(usesAir ? [
            { binding: 4, resource: { buffer: this.air.arena } },
          ] : []),
          ...(usesVelocity ? [
            { binding: 5, resource: { buffer: this.air.rowVelocities } },
            { binding: 7, resource: { buffer: this.air.losassoVelocity?.control
              ?? this.bindingSentinel } },
            { binding: 11, resource: { buffer: this.air.losassoVelocity?.stagedVelocity
              ?? this.bindingSentinel } },
          ] : []),
          ...(usesLosassoControl ? [{ binding: 7, resource: { buffer: this.air.losassoVelocity?.control
            ?? this.bindingSentinel } }] : []),
          ...(prepare ? [{ binding: 6, resource: { buffer: this.dispatchArgs } }] : []),
        ],
      });
      if (!this.bindGroups.has(pipeline)) this.bindGroups.set(pipeline, group);
      const pass = broker.compute({ label: `Build coarse-only summary · ${entry}` });
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      if (this.indirectDispatch && record !== undefined) {
        pass.dispatchWorkgroupsIndirect(this.dispatchArgs, record * 12);
      } else {
        pass.dispatchWorkgroups(width, Math.ceil(groups / width));
      }
    };
    if (this.indirectDispatch) {
      dispatch("prepareSummaryDispatch", 1);
      broker.fence("coarse-only summary indirect arguments published");
    }
    if (!this.hierarchyInitialized) {
      dispatch("resetSummary", Math.max(this.plan.phiOffsetWords, this.plan.entryCapacity));
      // The dense tracker covers every B4 base block, so its hierarchy is the
      // complete [0,keyCapacity) key set. Build that set directly instead of
      // making all 64 cells in every block race to claim the same key.
      dispatch("ensureSupportSummaryPages", this.plan.hierarchyKeyCapacity);
      dispatch("ensureSupportSummaryRanks", this.plan.hierarchyKeyCapacity);
      this.hierarchyInitialized = true;
    } else {
      // Keys, pages and ranks are a construction-time function of the fixed
      // dense lattice. Retain that directory and reset only per-advance entry
      // payloads and counters.
      dispatch("resetSummaryValues", Math.max(this.plan.entryCapacity, 32));
    }
    if (this.indirectDispatch) {
      // state[1] is final only after both rank stages; the entry record cannot
      // be authored by the head singleton. Both fences keep the argument
      // buffer's storage writes out of the passes that consume it as INDIRECT.
      broker.fence("coarse-only summary rank stages retired");
      dispatch("prepareSummaryFinalizeDispatch", 1);
      broker.fence("coarse-only summary entry arguments published");
    }
    dispatch("predictSummaryCells", this.air.layout.ownerDirectoryCellCapacity, 0);
    dispatch("seedDenseRedistance", this.air.layout.ownerDirectoryCellCapacity, 0);
    // The seed lands in bank 2, so an ODD count finishes in the output bank.
    // A Jacobi sweep advances the front exactly one cell, which is why the
    // count is the reach and the reach is planned rather than literal.
    for (let sweep = 0; sweep < this.redistanceSweeps; sweep += 1) {
      dispatch(sweep % 2 === 0 ? "redistanceScratchToOutput" : "redistanceOutputToScratch",
        this.air.layout.ownerDirectoryCellCapacity, 0);
    }
    dispatch("summarizeDenseVolume", this.air.layout.ownerDirectoryCellCapacity, 0);
    dispatch("prepareVolumeCorrection", 1);
    dispatch("correctAndAggregateSummaryCells", this.air.layout.ownerDirectoryCellCapacity, 0);
    dispatch("finalizeSummaryEntries", this.plan.entryCapacity, 2);
    dispatch("publishSummary", 1);
    dispatch("publishDenseComplement", this.air.layout.ownerDirectoryCellCapacity, 0);
    dispatch("correctCoarseDirectory", this.coarse.rowCapacity, 1);
  }

  /** Diagnostic-only receipt. `completions` counts advances whose prediction
   * covered the whole lattice; `advances` counts every encoded publication.
   * A ratio below one means consumers saw a held surface on the difference. */
  async readReceipt(): Promise<Readonly<{ advances: number; completions: number;
    predictedCells: number; domainVolume: number; published: boolean; error: number;
    predictedVolume: number; targetVolume: number; interfaceCells: number;
    correction: number; bank: number; frozenCells: number;
    interfaceVelocityQueries: number; interfaceVelocityValid: number;
    interfacePhiMoved: number; maximumInterfaceSpeed: number;
    airUnpublishedAdvances: number; airErrorWord: number; airValidWord: number;
    airLayoutWord: number; airArenaShort: number;
    airFirstErrorStage: number; airFirstErrorItem: number;
    denseBanks: readonly Readonly<{ minimum: number; maximum: number; negative: number; zero: number }>[] }>> {
    const denseBytes = 3 * this.domainVolume * 4;
    const readback = this.device.createBuffer({ label: "coarse-only tracker receipt readback",
      size: 192 + denseBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Read coarse-only tracker receipt" });
    encoder.copyBufferToBuffer(this.state, 0, readback, 0, 128);
    encoder.copyBufferToBuffer(this.directory, 0, readback, 128, 64);
    encoder.copyBufferToBuffer(this.directory, this.plan.phiOffsetWords * 4,
      readback, 192, denseBytes);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange().slice(0));
      const header = 32;
      const dense = new Float32Array(words.buffer, 192, 3 * this.domainVolume);
      const denseBanks = [0, 1, 2].map((bank) => {
        let minimum = Number.POSITIVE_INFINITY, maximum = Number.NEGATIVE_INFINITY;
        let negative = 0, zero = 0;
        for (let item = 0; item < this.domainVolume; item += 1) {
          const value = dense[bank * this.domainVolume + item]!;
          minimum = Math.min(minimum, value); maximum = Math.max(maximum, value);
          if (value < 0) negative += 1; else if (value === 0) zero += 1;
        }
        return { minimum, maximum, negative, zero };
      });
      return { advances: words[19], completions: words[20],
        predictedCells: words[18], domainVolume: this.domainVolume,
        published: words[header + 9] === 0x8000_0000, error: words[header + 0],
        predictedVolume: words[12] / 4096, targetVolume: words[14] / 4096,
        interfaceCells: words[13],
        correction: new Float32Array(new Uint32Array([words[15]]).buffer)[0],
        bank: words[17] & 1, frozenCells: words[31],
        interfaceVelocityQueries: words[27], interfaceVelocityValid: words[28],
        interfacePhiMoved: words[29],
        maximumInterfaceSpeed: new Float32Array(new Uint32Array([words[30]]).buffer)[0],
        airUnpublishedAdvances: words[21], airErrorWord: words[22],
        airValidWord: words[23], airLayoutWord: words[24], airArenaShort: words[25],
        airFirstErrorStage: words[26] >>> 24, airFirstErrorItem: words[26] & 0x00ff_ffff,
        denseBanks };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.bindGroups.clear();
    this.directory.destroy(); this.state.destroy(); this.params.destroy();
    this.dispatchArgs.destroy(); this.bindingSentinel.destroy();
  }
}

export const coarseSummaryWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;const PUBLISHED:u32=0x80000000u;
const COARSE_AUTHORITY:u32=0x80000000u;const PHI_INTERFACE:u32=4u;
const PAGE_SIZE:u32=${PAGE_SIZE}u;const ENTRY_WORDS:u32=${ENTRY_WORDS}u;
struct P{baseDims:vec3u,pad0:u32,dims:vec3u,maximumLeafSize:u32,keyCapacity:u32,topPages:u32,pageCapacity:u32,
 entryCapacity:u32,entryOffset:u32,maximumLevel:u32,rowCapacity:u32,directoryWords:u32,
 airControl:u32,airVectors:u32,airOwners:u32,domainVolume:u32,time:vec4f,maxWorkgroups:u32,physicalCellSize:f32,
 diagnostics:u32,globalVolumeCorrection:u32}
struct CoarseEntry{cellPlusOne:u32,size:u32,phi:f32,minimumPhi:f32,maximumPhi:f32,flags:u32,row:u32,volume:f32}
struct CoarseDirectory{state:u32,generation:u32,rowCount:u32,maximumLeafSize:u32,dimensions:vec3u,
 physicalCellSize:f32,entries:array<CoarseEntry>}
@group(0)@binding(0)var<storage,read_write>directory:array<atomic<u32>>;
@group(0)@binding(1)var<storage,read_write>coarse:CoarseDirectory;
@group(0)@binding(2)var<storage,read_write>state:array<atomic<u32>>;
@group(0)@binding(3)var<uniform>p:P;
@group(0)@binding(4)var<storage,read>air:array<u32>;
@group(0)@binding(5)var<storage,read>rowVelocities:array<vec4f>;
@group(0)@binding(6)var<storage,read_write>dispatchArgs:array<u32>;
@group(0)@binding(7)var<storage,read>losassoControl:array<u32>;
@group(0)@binding(8)var<storage,read>losassoFaceGeometry:array<vec4u>;
@group(0)@binding(9)var<storage,read>losassoExtendedVelocity:array<f32>;
@group(0)@binding(10)var<storage,read>losassoFaceDirectory:array<vec2u>;
@group(0)@binding(11)var<storage,read>losassoStagedVelocity:array<u32>;
fn linear(w:vec3u,n:vec3u,l:u32)->u32{return (w.x+w.y*n.x)*256u+l;}
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}
fn finite3(v:vec3f)->bool{return all(v==v)&&all(abs(v)<vec3f(3.402823e38));}
fn canonicalSum8(values:ptr<function,array<f32,8>>)->f32{for(var i=1u;i<8u;i+=1u){let value=(*values)[i];var j=i;loop{if(j==0u||abs((*values)[j-1u])<=abs(value)){break;}(*values)[j]=(*values)[j-1u];j-=1u;}(*values)[j]=value;}var sum=0.;var i=0u;loop{if(i>=8u){break;}let magnitude=abs((*values)[i]);var balance=0;var j=i;loop{if(j>=8u||abs((*values)[j])!=magnitude){break;}if((*values)[j]>0.){balance+=1;}else if((*values)[j]<0.){balance-=1;}j+=1u;}sum+=f32(balance)*magnitude;i=j;}return sum;}
fn canonicalWeight(a:f32,b:f32,c:f32)->f32{var low=a;var middle=b;var high=c;if(low>middle){let swap=low;low=middle;middle=swap;}if(middle>high){let swap=middle;middle=high;high=swap;}if(low>middle){let swap=low;low=middle;middle=swap;}return low*middle*high;}
fn ordered(v:f32)->u32{let bits=bitcast<u32>(v);return select(bits^0x80000000u,~bits,(bits&0x80000000u)!=0u);}
fn topWord(key:u32)->u32{return 16u+key/PAGE_SIZE;}
fn poolOffset()->u32{return 16u+p.topPages;}
fn pageWord(page:u32,key:u32)->u32{return poolOffset()+page*PAGE_SIZE+(key&(PAGE_SIZE-1u));}
fn entryBase(rank:u32)->u32{return p.entryOffset+rank*ENTRY_WORDS;}
fn phiOffset()->u32{return p.entryOffset+p.entryCapacity*ENTRY_WORDS;}
fn phiWord(bank:u32,item:u32)->u32{return phiOffset()+bank*p.domainVolume+item;}
fn mortonPart(v:u32)->u32{var x=v&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;
 x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;}
fn morton(cell:u32)->u32{let q=vec3u(cell%p.dims.x,(cell/p.dims.x)%p.dims.y,cell/(p.dims.x*p.dims.y));
 return mortonPart(q.x)|(mortonPart(q.y)<<1u)|(mortonPart(q.z)<<2u);}
fn less(aSize:u32,aCell:u32,bSize:u32,bCell:u32)->bool{let al=31u-countLeadingZeros(aSize);
 let bl=31u-countLeadingZeros(bSize);return al<bl||(al==bl&&morton(aCell)<morton(bCell));}
fn coarseSlot(cell:u32,size:u32)->u32{let count=min(coarse.rowCount,arrayLength(&coarse.entries));var lo=0u;var hi=count;
 while(lo<hi){let mid=lo+(hi-lo)/2u;let e=coarse.entries[mid];if(less(e.size,e.cellPlusOne-1u,size,cell)){lo=mid+1u;}else{hi=mid;}}
 if(lo<count){let e=coarse.entries[lo];if(e.cellPlusOne==cell+1u&&e.size==size){return lo;}}return INVALID;}
fn coarseAt(point:vec3f)->vec2f{if(coarse.state!=PUBLISHED||any(point<vec3f(0.0))||any(point>=vec3f(p.dims))){return vec2f(0.0);}
 let q=vec3u(floor(point));var size=1u;loop{let origin=(q/vec3u(size))*vec3u(size);
  let cell=origin.x+p.dims.x*(origin.y+p.dims.y*origin.z);let slot=coarseSlot(cell,size);
  if(slot!=INVALID){let e=coarse.entries[slot];if((e.flags&9u)==9u&&finite(e.phi)){return vec2f(e.phi,1.0);}}
  if(size>=coarse.maximumLeafSize){break;}size*=2u;}return vec2f(0.0);}
// The sparse coarse directory deliberately stores no positive-air rows. Build
// the missing side of the signed distance only at summary-query time from the
// nearest represented interface samples. This is computation over the sole
// coarse phi authority, not a second transported field or a fine-band cache.
fn extrapolatedCoarseAt(point:vec3f)->vec2f{let direct=coarseAt(point);if(direct.y!=0.0){return direct;}
 var best=3.402823e38;var found=false;let centre=vec3i(floor(point));
 for(var z=-1;z<=1;z+=1){for(var y=-1;y<=1;y+=1){for(var x=-1;x<=1;x+=1){let q=centre+vec3i(x,y,z);
  if(any(q<vec3i(0))||any(q>=vec3i(p.dims))){continue;}let samplePoint=vec3f(q)+vec3f(0.5);let sample=coarseAt(samplePoint);
  if(sample.y!=0.0){let candidate=sample.x+p.physicalCellSize*length(point-samplePoint);if(finite(candidate)){best=min(best,candidate);found=true;}}}}}
 return select(vec2f(0.0),vec2f(best,1.0),found);}
fn interpolatedCoarseAt(point:vec3f)->vec2f{let upper=max(vec3f(p.dims)-vec3f(0.5),vec3f(0.5));
 let x=clamp(point,vec3f(0.5),upper);let low=vec3i(floor(x-vec3f(0.5)));let t=x-(vec3f(low)+vec3f(0.5));
 var terms:array<f32,8>;var weights:array<f32,8>;
 for(var corner=0u;corner<8u;corner+=1u){let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
  let q=clamp(low+offset,vec3i(0),vec3i(p.dims)-vec3i(1));let sample=extrapolatedCoarseAt(vec3f(q)+vec3f(0.5));
  let weight=canonicalWeight(select(1.0-t.x,t.x,(corner&1u)!=0u),select(1.0-t.y,t.y,(corner&2u)!=0u),select(1.0-t.z,t.z,(corner&4u)!=0u));if(weight>0.0&&sample.y!=0.0){terms[corner]=weight*sample.x;weights[corner]=weight;}}
 let value=canonicalSum8(&terms);let total=canonicalSum8(&weights);
 return select(vec2f(0.0),vec2f(value/max(total,1e-12),1.0),total>0.999);}
struct AxisSample{low:i32,lowWeight:f32,highWeight:f32}
fn centeredAxisSample(value:f32,dimension:u32)->AxisSample{let half=0.5*f32(dimension);let bounded=clamp(value,-half+0.5,half-0.5);let positiveLattice=(half-0.5)+abs(bounded);let positiveLow=i32(floor(positiveLattice));let fraction=fract(positiveLattice);if(bounded>=0.0){return AxisSample(positiveLow,1.0-fraction,fraction);}if(fraction==0.0){return AxisSample(i32(dimension)-1-positiveLow,1.0,0.0);}return AxisSample(i32(dimension)-2-positiveLow,fraction,1.0-fraction);}
fn densePhiAtCentered(bank:u32,point:vec3f)->vec2f{let ax=centeredAxisSample(point.x,p.dims.x);let ay=centeredAxisSample(point.y,p.dims.y);let az=centeredAxisSample(point.z,p.dims.z);
 var terms:array<f32,8>;
 for(var corner=0u;corner<8u;corner+=1u){let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
  let q=clamp(vec3i(ax.low,ay.low,az.low)+offset,vec3i(0),vec3i(p.dims)-vec3i(1));let item=u32(q.x)+p.dims.x*(u32(q.y)+p.dims.y*u32(q.z));
  let at=phiWord(bank,item);if(at>=arrayLength(&directory)){return vec2f(0.0);}let sample=bitcast<f32>(atomicLoad(&directory[at]));
  if(!finite(sample)){return vec2f(0.0);}let weight=canonicalWeight(select(ax.lowWeight,ax.highWeight,(corner&1u)!=0u),select(ay.lowWeight,ay.highWeight,(corner&2u)!=0u),select(az.lowWeight,az.highWeight,(corner&4u)!=0u));terms[corner]=weight*sample;}
 return vec2f(canonicalSum8(&terms),1.0);}
fn densePhiAt(bank:u32,point:vec3f)->vec2f{return densePhiAtCentered(bank,point-0.5*vec3f(p.dims));}
fn quantizePhi(value:f32)->f32{return round(value*65536.0)/65536.0;}
// The version word is interpolated, never spelled. It was frozen at a literal
// 2 while the air-support layout moved to 3, so this predicate answered false
// on every step: predictSummaryCells returned before writing a single cell,
// state[18] never reached the domain cardinality, and the redistance sweeps,
// volume correction, and publication all bailed on that same counter. The
// coarse-only tracker -- the whole surface authority at fine factor one --
// silently stopped advecting.
fn airPublished()->bool{return p.airControl+15u<arrayLength(&air)&&air[p.airControl]==0u
 &&air[p.airControl+13u]==${OCTREE_AIR_SUPPORT_VALID}u
 &&air[p.airControl+14u]==${OCTREE_AIR_SUPPORT_LAYOUT_VERSION}u;}
fn losassoMode()->bool{return p.time.z==1.0;}
fn losassoPublished()->bool{return arrayLength(&losassoControl)>=4u&&losassoControl[3]==1u;}
fn transportPublished()->bool{return select(airPublished(),losassoPublished(),losassoMode());}
// The analytic dense phi is already a complete surface authority at t=0.
// Losasso's staggered velocity publication follows topology construction, so
// exactly the cold publication may publish that held field; otherwise the
// compact rows and dense complement wait on one another.  state[16] means a
// phi bank exists and stays set forever, so it must not be used as this gate:
// doing so silently relabelled every later missing-velocity step as complete.
fn trackerRunnable()->bool{return transportPublished()
 ||(losassoMode()&&atomicLoad(&state[19])==0u);}
fn ownerHashCapacity()->u32{if(p.airOwners>=arrayLength(&air)){return 0u;}
 return (arrayLength(&air)-p.airOwners)/${OCTREE_AIR_SUPPORT_OWNER_HASH.recordWords}u;}
${octreeAirSupportOwnerHashStartWGSL("ownerHashStart")}
// The owner directory is the adaptive (origin,size) identity hash, NOT the
// retired dense finest-cell map: word zero is a key, not a tag. Reading
// \`owners[4 * cell]\` here resolved almost every coarse cell to an empty slot
// and then interpreted the zero as row zero, so the whole dense phi advected
// on one row's velocity. Probe the containing dyadic leaf exactly as fine
// transport does, largest authored leaf first, and stop the chain at the
// first zero key.
fn supportIdentity(item:u32)->vec4u{if(!airPublished()||item>=p.domainVolume){return vec4u(INVALID);}
 let capacity=ownerHashCapacity();if(capacity==0u){return vec4u(INVALID);}
 let q=vec3u(item%p.dims.x,(item/p.dims.x)%p.dims.y,item/(p.dims.x*p.dims.y));
 var size=p.maximumLeafSize;
 loop{let origin=(q/vec3u(size))*vec3u(size);
  let originCell=origin.x+p.dims.x*(origin.y+p.dims.y*origin.z);
  let start=ownerHashStart(originCell,size,capacity);
  for(var probe=0u;probe<min(capacity,${OCTREE_AIR_SUPPORT_OWNER_HASH.maximumProbes}u);probe+=1u){
   let at=p.airOwners+${OCTREE_AIR_SUPPORT_OWNER_HASH.recordWords}u*((start+probe)%capacity);
   if(at+3u>=arrayLength(&air)){return vec4u(INVALID);}
   let key=air[at];if(key==0u){break;}
   if(key-1u==originCell&&air[at+1u]==size){return vec4u(air[at+2u],originCell,size,air[at+3u]);}}
  if(size<=1u){break;}size>>=1u;}
 return vec4u(INVALID);}
fn supportVelocity(tag:u32)->vec4f{if(tag==INVALID){return vec4f(0.0);}var v=vec4f(0.0);
 if((tag&${OCTREE_AIR_SUPPORT_TAG}u)==0u){let bank=air[p.airControl+3u]&1u;let at=bank*p.rowCapacity+tag;
  if(tag>=p.rowCapacity||at>=arrayLength(&rowVelocities)){return vec4f(0.0);}v=rowVelocities[at];}
 else{let support=tag&${(~OCTREE_AIR_SUPPORT_TAG) >>> 0}u;if(support>=air[p.airControl+6u]){return vec4f(0.0);}let at=p.airVectors+4u*support;
  if(at+3u>=arrayLength(&air)){return vec4f(0.0);}v=vec4f(bitcast<f32>(air[at]),bitcast<f32>(air[at+1u]),bitcast<f32>(air[at+2u]),bitcast<f32>(air[at+3u]));}
 return select(vec4f(0.0),v,finite(v.x)&&finite(v.y)&&finite(v.z)&&v.w>0.0);}
fn velocityAtCentered(point:vec3f)->vec4f{let ax=centeredAxisSample(point.x,p.dims.x);let ay=centeredAxisSample(point.y,p.dims.y);let az=centeredAxisSample(point.z,p.dims.z);
 var termsX:array<f32,8>;var termsY:array<f32,8>;var termsZ:array<f32,8>;var weights:array<f32,8>;
 for(var corner=0u;corner<8u;corner+=1u){let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
  let q=clamp(vec3i(ax.low,ay.low,az.low)+offset,vec3i(0),vec3i(p.dims)-vec3i(1));let item=u32(q.x)+p.dims.x*(u32(q.y)+p.dims.y*u32(q.z));
  let v=supportVelocity(supportIdentity(item).x);let weight=canonicalWeight(select(ax.lowWeight,ax.highWeight,(corner&1u)!=0u),select(ay.lowWeight,ay.highWeight,(corner&2u)!=0u),select(az.lowWeight,az.highWeight,(corner&4u)!=0u));
  if(weight>0.0&&v.w>0.0){termsX[corner]=weight*v.x;termsY[corner]=weight*v.y;termsZ[corner]=weight*v.z;weights[corner]=weight;}}
 let result=vec3f(canonicalSum8(&termsX),canonicalSum8(&termsY),canonicalSum8(&termsZ));let total=canonicalSum8(&weights);
 // Coverage rule for the trilinear stencil, authored in p.time.y.
 //
 // At 1.0 every one of the eight corners must resolve to a published
 // air-support owner, and a single uncovered corner makes the whole sample
 // invalid -- which makes predictSummaryCells leave that cell's phi exactly
 // where it was. That was harmless while the corridor was approximately the
 // entire air partition. Against Bet 1.3's proven-reach corridor the leading
 // edge of a spreading front is routinely half-covered, so the front advances
 // at the speed the corridor grows rather than the speed of the fluid.
 //
 // Lowering the rule renormalizes by the covered weight instead. That is a
 // physics change, not a launch shape, so it stays an authored A/B.
 return select(vec4f(0.0),vec4f(result/max(total,1e-12),1.0),total>=p.time.y);}
fn velocityAtPoint(point:vec3f)->vec4f{return velocityAtCentered(point-0.5*vec3f(p.dims));}
${coarseSummaryLosassoVelocitySamplingWGSL}
fn transportVelocityAt(point:vec3f)->vec4f{if(losassoMode()){let sample=losassoVelocityAtGrid(point);
  return vec4f(sample.value,f32(sample.valid));}return velocityAtPoint(point);}
fn supportBase(item:u32)->u32{if(item>=p.domainVolume){return INVALID;}
 let q=vec3u(item%p.dims.x,(item/p.dims.x)%p.dims.y,item/(p.dims.x*p.dims.y));
 let b=q/4u;return b.x+p.baseDims.x*(b.y+p.baseDims.y*b.z);}
fn hierarchyKey(base:u32,targetLevel:u32)->u32{let xy=p.baseDims.x*p.baseDims.y;let z=base/xy;let rem=base-z*xy;
 let y=rem/p.baseDims.x;var q=vec3u(rem-y*p.baseDims.x,y,z);var d=p.baseDims;var offset=0u;
 for(var level=0u;level<targetLevel;level+=1u){offset+=d.x*d.y*d.z;d=(d+vec3u(1u))/2u;q/=2u;}
 return offset+q.x+d.x*(q.y+d.y*q.z);}
fn rowBaseAndLevel(e:CoarseEntry)->vec2u{if(e.cellPlusOne==0u||e.size==0u||(e.size&(e.size-1u))!=0u){return vec2u(INVALID);}
 let cell=e.cellPlusOne-1u;if(cell>=p.dims.x*p.dims.y*p.dims.z){return vec2u(INVALID);}
 let origin=vec3u(cell%p.dims.x,(cell/p.dims.x)%p.dims.y,cell/(p.dims.x*p.dims.y));
 let side=max(1u,e.size/4u);let b=origin/4u;if(e.size>=4u&&any(origin%vec3u(e.size)!=vec3u(0u))){return vec2u(INVALID);}
 if(any(b%vec3u(side)!=vec3u(0u))){return vec2u(INVALID);}let base=b.x+p.baseDims.x*(b.y+p.baseDims.y*b.z);
 return vec2u(base,31u-countLeadingZeros(side));}
fn rankForKey(key:u32)->u32{let pagePlusOne=atomicLoad(&directory[topWord(key)]);
 if(pagePlusOne==0u||pagePlusOne==INVALID||pagePlusOne>p.pageCapacity){return INVALID;}
 let rankPlusOne=atomicLoad(&directory[pageWord(pagePlusOne-1u,key)]);
 return select(INVALID,rankPlusOne-1u,rankPlusOne>0u&&rankPlusOne<=p.entryCapacity);}
@compute @workgroup_size(256)fn resetSummary(@builtin(workgroup_id)w:vec3u,@builtin(num_workgroups)n:vec3u,
 @builtin(local_invocation_index)l:u32){let i=linear(w,n,l);if(i<phiOffset()){atomicStore(&directory[i],0u);}
 // Target volume, transport initialization and active bank survive rebuilds.
 // So do the receipt tallies at 19/20: they count advances encoded against
 // advances that actually completed, so a single readback at the end of a run
 // reports how intermittent the tracker was. Only 18, the per-advance
 // prediction counter the completeness test reads, is cleared here.
 if(i<arrayLength(&state)&&(i<14u||i==15u||i==18u||(i>=27u&&i<=31u))){atomicStore(&state[i],0u);}}
@compute @workgroup_size(256)fn resetSummaryValues(@builtin(workgroup_id)w:vec3u,@builtin(num_workgroups)n:vec3u,
 @builtin(local_invocation_index)l:u32){let i=linear(w,n,l);
 if(i<16u){atomicStore(&directory[i],0u);}
 if(i<p.entryCapacity){let base=entryBase(i);atomicStore(&directory[base+1u],ordered(3.402823e38));
  atomicStore(&directory[base+2u],ordered(-3.402823e38));atomicStore(&directory[base+3u],bitcast<u32>(3.402823e38));
  for(var j=4u;j<ENTRY_WORDS;j+=1u){atomicStore(&directory[base+j],0u);}}
 if(i<arrayLength(&state)&&((i>=2u&&i<14u)||i==15u||i==18u||(i>=27u&&i<=31u))){atomicStore(&state[i],0u);}}
@compute @workgroup_size(256)fn ensureSummaryPages(@builtin(workgroup_id)w:vec3u,@builtin(num_workgroups)n:vec3u,
 @builtin(local_invocation_index)l:u32){let row=linear(w,n,l);if(coarse.state!=PUBLISHED||row>=coarse.rowCount||row>=p.rowCapacity){return;}
 let e=coarse.entries[row];let bl=rowBaseAndLevel(e);if(bl.x==INVALID||(e.flags&9u)!=9u){atomicOr(&state[2],1u);return;}
 for(var level=bl.y;level<=p.maximumLevel;level+=1u){let key=hierarchyKey(bl.x,level);let top=topWord(key);
  var claim=atomicCompareExchangeWeak(&directory[top],0u,INVALID);for(var retry=0u;retry<4u&&!claim.exchanged&&claim.old_value==0u;retry+=1u){claim=atomicCompareExchangeWeak(&directory[top],0u,INVALID);}if(claim.exchanged){let page=atomicAdd(&state[0],1u);
   if(page>=p.pageCapacity){atomicStore(&directory[top],0u);atomicOr(&state[2],1u);continue;}
   for(var j=0u;j<PAGE_SIZE;j+=1u){atomicStore(&directory[poolOffset()+page*PAGE_SIZE+j],0u);}
   atomicStore(&directory[top],page+1u);}}}
@compute @workgroup_size(256)fn ensureSupportSummaryPages(@builtin(workgroup_id)w:vec3u,@builtin(num_workgroups)n:vec3u,
 @builtin(local_invocation_index)l:u32){let key=linear(w,n,l);if(key>=p.keyCapacity){return;}let top=topWord(key);
  var claim=atomicCompareExchangeWeak(&directory[top],0u,INVALID);for(var retry=0u;retry<4u&&!claim.exchanged&&claim.old_value==0u;retry+=1u){claim=atomicCompareExchangeWeak(&directory[top],0u,INVALID);}if(claim.exchanged){let page=atomicAdd(&state[0],1u);
   if(page>=p.pageCapacity){atomicStore(&directory[top],0u);atomicOr(&state[2],1u);return;}for(var j=0u;j<PAGE_SIZE;j+=1u){atomicStore(&directory[poolOffset()+page*PAGE_SIZE+j],0u);}atomicStore(&directory[top],page+1u);}}
@compute @workgroup_size(256)fn ensureSummaryRanks(@builtin(workgroup_id)w:vec3u,@builtin(num_workgroups)n:vec3u,
 @builtin(local_invocation_index)l:u32){let row=linear(w,n,l);if(coarse.state!=PUBLISHED||row>=coarse.rowCount||row>=p.rowCapacity){return;}
 let bl=rowBaseAndLevel(coarse.entries[row]);if(bl.x==INVALID){return;}for(var level=bl.y;level<=p.maximumLevel;level+=1u){
  let key=hierarchyKey(bl.x,level);let pagePlusOne=atomicLoad(&directory[topWord(key)]);if(pagePlusOne==0u||pagePlusOne==INVALID){atomicOr(&state[2],2u);continue;}
  let word=pageWord(pagePlusOne-1u,key);var claim=atomicCompareExchangeWeak(&directory[word],0u,INVALID);for(var retry=0u;retry<4u&&!claim.exchanged&&claim.old_value==0u;retry+=1u){claim=atomicCompareExchangeWeak(&directory[word],0u,INVALID);}if(claim.exchanged){
   let rank=atomicAdd(&state[1],1u);if(rank>=p.entryCapacity){atomicStore(&directory[word],0u);atomicOr(&state[2],1u);continue;}
   let base=entryBase(rank);atomicStore(&directory[base],key);atomicStore(&directory[base+1u],ordered(3.402823e38));
   atomicStore(&directory[base+2u],ordered(-3.402823e38));atomicStore(&directory[base+3u],bitcast<u32>(3.402823e38));
   for(var j=4u;j<ENTRY_WORDS;j+=1u){atomicStore(&directory[base+j],0u);}atomicStore(&directory[word],rank+1u);}}}
@compute @workgroup_size(256)fn ensureSupportSummaryRanks(@builtin(workgroup_id)w:vec3u,@builtin(num_workgroups)n:vec3u,
 @builtin(local_invocation_index)l:u32){let key=linear(w,n,l);if(key>=p.keyCapacity){return;}let pagePlusOne=atomicLoad(&directory[topWord(key)]);
  if(pagePlusOne==0u||pagePlusOne==INVALID){atomicOr(&state[2],2u);return;}let word=pageWord(pagePlusOne-1u,key);
  var claim=atomicCompareExchangeWeak(&directory[word],0u,INVALID);for(var retry=0u;retry<4u&&!claim.exchanged&&claim.old_value==0u;retry+=1u){claim=atomicCompareExchangeWeak(&directory[word],0u,INVALID);}if(claim.exchanged){let rank=atomicAdd(&state[1],1u);
   if(rank>=p.entryCapacity){atomicStore(&directory[word],0u);atomicOr(&state[2],1u);return;}let base=entryBase(rank);
   atomicStore(&directory[base],key);atomicStore(&directory[base+1u],ordered(3.402823e38));atomicStore(&directory[base+2u],ordered(-3.402823e38));
   atomicStore(&directory[base+3u],bitcast<u32>(3.402823e38));for(var j=4u;j<ENTRY_WORDS;j+=1u){atomicStore(&directory[base+j],0u);}atomicStore(&directory[word],rank+1u);}}
@compute @workgroup_size(256)fn aggregateSummaryRows(@builtin(workgroup_id)w:vec3u,@builtin(num_workgroups)n:vec3u,
 @builtin(local_invocation_index)l:u32){let row=linear(w,n,l);if(coarse.state!=PUBLISHED||row>=coarse.rowCount||row>=p.rowCapacity){return;}
 let e=coarse.entries[row];let bl=rowBaseAndLevel(e);if(bl.x==INVALID||(e.flags&9u)!=9u||!finite(e.phi)||!finite(e.minimumPhi)||!finite(e.maximumPhi)){atomicOr(&state[2],4u);return;}
 let ma=select(min(abs(e.minimumPhi),abs(e.maximumPhi)),0.0,e.minimumPhi<=0.0&&e.maximumPhi>=0.0);
 for(var level=bl.y;level<=p.maximumLevel;level+=1u){let key=hierarchyKey(bl.x,level);let rank=rankForKey(key);
  if(rank==INVALID){atomicOr(&state[2],2u);continue;}let base=entryBase(rank);atomicMin(&directory[base+1u],ordered(e.minimumPhi));
  atomicMax(&directory[base+2u],ordered(e.maximumPhi));atomicMin(&directory[base+3u],bitcast<u32>(ma));}}
@compute @workgroup_size(256)fn predictSummaryCells(@builtin(workgroup_id)w:vec3u,@builtin(num_workgroups)n:vec3u,
 @builtin(local_invocation_index)l:u32){let item=linear(w,n,l);
 // Diagnostic-only, one lane per advance. An unpublished air support is the
 // single reason this pass can decline, and the three control words say which
 // clause of the receipt failed, so record them rather than inferring later.
 if(item==0u&&!transportPublished()){atomicAdd(&state[21],1u);
  if(p.airControl+15u<arrayLength(&air)){atomicStore(&state[22],air[p.airControl]);
   atomicStore(&state[23],air[p.airControl+13u]);atomicStore(&state[24],air[p.airControl+14u]);
   atomicStore(&state[26],air[p.airControl+1u]);}
  else{atomicStore(&state[25],1u);}}
 if(item>=p.domainVolume||!trackerRunnable()){return;}
 let q=vec3u(item%p.dims.x,(item/p.dims.x)%p.dims.y,item/(p.dims.x*p.dims.y));let point=vec3f(q)+vec3f(0.5);
 let initialized=atomicLoad(&state[16])!=0u;let readBank=atomicLoad(&state[17])&1u;let writeBank=select(0u,1u-readBank,initialized);
 var sample=vec2f(0.0);if(!initialized){sample=interpolatedCoarseAt(point);}else{let currentAt=phiWord(readBank,item);
  if(currentAt<arrayLength(&directory)){let current=bitcast<f32>(atomicLoad(&directory[currentAt]));
   sample=vec2f(current,select(0.0,1.0,finite(current)));}}
 let h=p.physicalCellSize;
 let nearInterface=p.diagnostics!=0u&&initialized&&abs(sample.x)<=2.0*h;if(nearInterface){atomicAdd(&state[27],1u);}
 let velocity=transportVelocityAt(point);if(nearInterface&&velocity.w>0.0){atomicAdd(&state[28],1u);
  atomicMax(&state[30],bitcast<u32>(length(velocity.xyz)));}
 if(initialized&&velocity.w==0.0){atomicAdd(&state[31],1u);}
 if(initialized&&velocity.w>0.0){let midpoint=point-(0.5*p.time.x/h)*velocity.xyz;
  let middleVelocity=transportVelocityAt(midpoint);let traced=select(velocity,middleVelocity,middleVelocity.w>0.0);
  let departure=point-(p.time.x/h)*traced.xyz;let transported=densePhiAt(readBank,departure);if(transported.y!=0.0){
   if(nearInterface&&abs(transported.x-sample.x)>1e-6*h){atomicAdd(&state[29],1u);}sample=transported;}}
 let predicted=quantizePhi(select(p.physicalCellSize,sample.x,sample.y!=0.0));let at=phiWord(writeBank,item);
 if(at>=arrayLength(&directory)||!finite(predicted)){atomicOr(&state[2],4u);return;}atomicStore(&directory[at],bitcast<u32>(predicted));
 atomicAdd(&state[18],1u);}
fn outputBank()->u32{let initialized=atomicLoad(&state[16])!=0u;let readBank=atomicLoad(&state[17])&1u;return select(0u,1u-readBank,initialized);}
fn denseRaw(bank:u32,item:u32)->f32{let at=phiWord(bank,item);if(at>=arrayLength(&directory)){return 3.402823e38;}
 return bitcast<f32>(atomicLoad(&directory[at]));}
fn neighborItem(q:vec3i)->u32{return u32(q.x)+p.dims.x*(u32(q.y)+p.dims.y*u32(q.z));}
const AXIS_DIRECTIONS=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
@compute @workgroup_size(256)fn seedDenseRedistance(@builtin(workgroup_id)w:vec3u,@builtin(num_workgroups)n:vec3u,
 @builtin(local_invocation_index)l:u32){let item=linear(w,n,l);if(item>=p.domainVolume||atomicLoad(&state[18])!=p.domainVolume){return;}
 let source=denseRaw(outputBank(),item);if(!finite(source)){atomicOr(&state[2],4u);return;}let q=vec3i(i32(item%p.dims.x),i32((item/p.dims.x)%p.dims.y),i32(item/(p.dims.x*p.dims.y)));
 var magnitude=p.physicalCellSize*f32(max(p.dims.x,max(p.dims.y,p.dims.z)));if(source==0.0){magnitude=0.0;}
 for(var axis=0u;axis<6u;axis+=1u){let otherQ=q+AXIS_DIRECTIONS[axis];if(any(otherQ<vec3i(0))||any(otherQ>=vec3i(p.dims))){continue;}
  let other=denseRaw(outputBank(),neighborItem(otherQ));if(!finite(other)||!((source<0.0&&other>=0.0)||(source>=0.0&&other<0.0))){continue;}
  magnitude=min(magnitude,p.physicalCellSize*abs(source)/max(abs(source)+abs(other),1e-12));}
 let value=select(magnitude,-magnitude,source<0.0);atomicStore(&directory[phiWord(2u,item)],bitcast<u32>(value));}
fn sweepDenseRedistance(sourceBank:u32,destinationBank:u32,item:u32){if(atomicLoad(&state[18])!=p.domainVolume){return;}
 let source=denseRaw(sourceBank,item);if(!finite(source)){atomicOr(&state[2],4u);return;}
 let q=vec3i(i32(item%p.dims.x),i32((item/p.dims.x)%p.dims.y),i32(item/(p.dims.x*p.dims.y)));var magnitude=abs(source);
 var nearest=vec3f(3.402823e38);for(var axis=0u;axis<3u;axis+=1u){for(var side=0u;side<2u;side+=1u){
  let otherQ=q+AXIS_DIRECTIONS[2u*axis+side];if(any(otherQ<vec3i(0))||any(otherQ>=vec3i(p.dims))){continue;}
  let other=denseRaw(sourceBank,neighborItem(otherQ));if(finite(other)){nearest[axis]=min(nearest[axis],abs(other));}}}
 // Sethian's first-order upwind Eikonal update: solve
 // sum_i(max(u-a_i,0)^2)=h^2 over the sorted axis minima. The old
 // min(a_i+h) recurrence is an L1 distance and systematically advances a
 // dam-break diagonal faster than its axes.
 let a0=min(nearest.x,min(nearest.y,nearest.z));let a2=max(nearest.x,max(nearest.y,nearest.z));
 let a1=max(min(nearest.x,nearest.y),max(min(nearest.x,nearest.z),min(nearest.y,nearest.z)));
 let h=p.physicalCellSize;var updated=a0+h;
 if(finite(a1)&&updated>a1){let discriminant=max(0.0,2.0*h*h-(a0-a1)*(a0-a1));updated=0.5*(a0+a1+sqrt(discriminant));}
 if(finite(a2)&&updated>a2){let sum=a0+a1+a2;let squares=a0*a0+a1*a1+a2*a2;
  let discriminant=max(0.0,sum*sum-3.0*(squares-h*h));updated=(sum+sqrt(discriminant))/3.0;}
 magnitude=min(magnitude,updated);
 let value=select(magnitude,-magnitude,source<0.0);atomicStore(&directory[phiWord(destinationBank,item)],bitcast<u32>(value));}
@compute @workgroup_size(256)fn redistanceScratchToOutput(@builtin(workgroup_id)w:vec3u,@builtin(num_workgroups)n:vec3u,
 @builtin(local_invocation_index)l:u32){let item=linear(w,n,l);if(item<p.domainVolume){sweepDenseRedistance(2u,outputBank(),item);}}
@compute @workgroup_size(256)fn redistanceOutputToScratch(@builtin(workgroup_id)w:vec3u,@builtin(num_workgroups)n:vec3u,
 @builtin(local_invocation_index)l:u32){let item=linear(w,n,l);if(item<p.domainVolume){sweepDenseRedistance(outputBank(),2u,item);}}
@compute @workgroup_size(256)fn summarizeDenseVolume(@builtin(workgroup_id)w:vec3u,@builtin(num_workgroups)n:vec3u,
 @builtin(local_invocation_index)l:u32){let item=linear(w,n,l);if(item>=p.domainVolume){return;}let value=denseRaw(outputBank(),item);
 if(atomicLoad(&state[18])!=p.domainVolume){return;}
 if(!finite(value)){atomicOr(&state[2],4u);return;}let occupancy=clamp(0.5-value/p.physicalCellSize,0.0,1.0);
 atomicAdd(&state[12],u32(round(4096.0*occupancy)));if(abs(value)<p.physicalCellSize){atomicAdd(&state[13],1u);}}
@compute @workgroup_size(1)fn prepareVolumeCorrection(){if(p.directoryWords>arrayLength(&directory)){atomicOr(&state[2],4u);return;}
 if(atomicLoad(&state[18])!=p.domainVolume){atomicStore(&state[15],bitcast<u32>(0.0));atomicOr(&state[2],4u);return;}
 if(p.globalVolumeCorrection==0u){atomicStore(&state[15],bitcast<u32>(0.0));return;}
 let predictedVolume=atomicLoad(&state[12]);let targetVolume=atomicLoad(&state[14]);
 if(targetVolume==0u){atomicStore(&state[14],predictedVolume);atomicStore(&state[15],bitcast<u32>(0.0));}
 else{let interfaceCount=max(1u,atomicLoad(&state[13]));let error=f32(i32(predictedVolume)-i32(targetVolume))/4096.0;
  let correction=clamp(p.physicalCellSize*error/f32(interfaceCount),-0.5*p.physicalCellSize,0.5*p.physicalCellSize);
  atomicStore(&state[15],bitcast<u32>(correction));}}
@compute @workgroup_size(256)fn correctAndAggregateSummaryCells(@builtin(workgroup_id)w:vec3u,@builtin(num_workgroups)n:vec3u,
 @builtin(local_invocation_index)l:u32){let item=linear(w,n,l);if(item>=p.domainVolume){return;}
 if(atomicLoad(&state[18])!=p.domainVolume){return;}
 let initialized=atomicLoad(&state[16])!=0u;let readBank=atomicLoad(&state[17])&1u;let writeBank=select(0u,1u-readBank,initialized);
 let at=phiWord(writeBank,item);if(at>=arrayLength(&directory)){atomicOr(&state[2],4u);return;}
 let value=quantizePhi(bitcast<f32>(atomicLoad(&directory[at]))+bitcast<f32>(atomicLoad(&state[15])));if(!finite(value)){atomicOr(&state[2],4u);return;}
 atomicStore(&directory[at],bitcast<u32>(value));let magnitude=abs(value);
 let q=vec3u(item%p.dims.x,(item/p.dims.x)%p.dims.y,item/(p.dims.x*p.dims.y));let b=q/4u;
 let baseKey=b.x+p.baseDims.x*(b.y+p.baseDims.y*b.z);for(var level=0u;level<=p.maximumLevel;level+=1u){
  let rank=rankForKey(hierarchyKey(baseKey,level));if(rank==INVALID){atomicOr(&state[2],2u);continue;}let base=entryBase(rank);
  atomicMin(&directory[base+1u],ordered(value));atomicMax(&directory[base+2u],ordered(value));atomicMin(&directory[base+3u],bitcast<u32>(magnitude));
  if(level==0u){let bit=(q.x&3u)+4u*((q.y&3u)+4u*(q.z&3u));let word=bit>>5u;let mask=1u<<(bit&31u);
   atomicOr(&directory[base+8u+word],mask);if(value<0.0){atomicOr(&directory[base+10u+word],mask);}atomicStore(&directory[base+4u],64u);atomicStore(&directory[base+5u],1u);}}}
@compute @workgroup_size(256)fn finalizeSummaryEntries(@builtin(workgroup_id)w:vec3u,@builtin(num_workgroups)n:vec3u,
 @builtin(local_invocation_index)l:u32){let rank=linear(w,n,l);let count=min(atomicLoad(&state[1]),p.entryCapacity);
 if(rank<count){let base=entryBase(rank);let lo=atomicLoad(&directory[base+1u]);let hi=atomicLoad(&directory[base+2u]);
  let ma=bitcast<f32>(atomicLoad(&directory[base+3u]));if(lo==ordered(3.402823e38)||hi==ordered(-3.402823e38)||!finite(ma)){atomicOr(&state[2],4u);}
  atomicStore(&directory[base+6u],COARSE_AUTHORITY);}}
@compute @workgroup_size(1)fn publishSummary(){let count=min(atomicLoad(&state[1]),p.entryCapacity);
 let initialized=atomicLoad(&state[16])!=0u;let readBank=atomicLoad(&state[17])&1u;let writeBank=select(0u,1u-readBank,initialized);
 // state[18] counts the cells predictSummaryCells actually wrote. Anything
 // short of the whole lattice means the prediction bailed -- almost always
 // because the air-support publication this advance consumes was not VALID --
 // and every stage after it (the redistance sweeps, the volume correction,
 // the dense complement) already declined to run on that same counter.
 //
 // The bank flip was already conditioned on it. The publication receipt was
 // NOT, so a stalled advance kept advertising PUBLISHED over the previous
 // advance's phi and consumers could not tell a fresh surface from a held
 // one. Make completeness part of the receipt: a tracker that did not run
 // says so, and the consumer retains its own last good surface instead of
 // being handed stale bytes labelled fresh.
 let complete=atomicLoad(&state[18])==p.domainVolume;
 atomicAdd(&state[19],1u);
 if(complete){atomicStore(&state[16],1u);atomicStore(&state[17],writeBank);atomicAdd(&state[20],1u);}
 let error=atomicLoad(&state[2]);atomicStore(&directory[0],error);atomicStore(&directory[1],coarse.generation);
  atomicStore(&directory[2],count);atomicStore(&directory[3],p.entryCapacity);atomicStore(&directory[4],p.baseDims.x);
  atomicStore(&directory[5],p.baseDims.y);atomicStore(&directory[6],p.baseDims.z);atomicStore(&directory[7],p.maximumLevel);
  atomicStore(&directory[8],p.entryOffset);atomicStore(&directory[9],select(0u,PUBLISHED,error==0u&&complete));
  atomicStore(&directory[10],p.keyCapacity);atomicStore(&directory[11],0u);atomicStore(&directory[12],16u);
  atomicStore(&directory[13],count);atomicStore(&directory[14],PAGE_SIZE);atomicStore(&directory[15],p.topPages);}
@compute @workgroup_size(256)fn publishDenseComplement(@builtin(workgroup_id)w:vec3u,@builtin(num_workgroups)n:vec3u,
 @builtin(local_invocation_index)l:u32){let item=linear(w,n,l);if(item>=p.domainVolume
 ||atomicLoad(&state[18])!=p.domainVolume){return;}let slot=p.rowCapacity+item;
 if(slot>=arrayLength(&coarse.entries)){atomicOr(&state[2],4u);return;}
 let value=bitcast<f32>(atomicLoad(&directory[phiWord(atomicLoad(&state[17])&1u,item)]));
 if(!finite(value)){atomicOr(&state[2],4u);return;}
 coarse.entries[slot]=CoarseEntry(item+1u,1u,value,value,value,9u,INVALID,0.0);
 // On the cold Losasso step this dense lattice is the source from which the
 // first compact row directory is restricted, so waiting for that directory
 // to publish creates a cycle.  Publish the backend-neutral header here; the
 // row bridge later fills rowCount and the lower generation bits.
 if(item==0u){coarse.maximumLeafSize=p.maximumLeafSize;coarse.dimensions=p.dims;coarse.physicalCellSize=p.physicalCellSize;
  coarse.generation|=0x40000000u;coarse.state=PUBLISHED;}}
@compute @workgroup_size(256)fn correctCoarseDirectory(@builtin(workgroup_id)w:vec3u,@builtin(num_workgroups)n:vec3u,
 @builtin(local_invocation_index)l:u32){let row=linear(w,n,l);if(atomicLoad(&directory[9])!=PUBLISHED||coarse.state!=PUBLISHED
 ||row>=min(coarse.rowCount,p.rowCapacity)){return;}let e=coarse.entries[row];if(e.cellPlusOne==0u||e.size==0u){return;}
 let cell=e.cellPlusOne-1u;let origin=vec3u(cell%p.dims.x,(cell/p.dims.x)%p.dims.y,cell/(p.dims.x*p.dims.y));
 let point=vec3f(origin)+vec3f(0.5*f32(e.size));let sample=densePhiAt(atomicLoad(&state[17])&1u,point);if(sample.y==0.0){return;}
 coarse.entries[row].phi=sample.x;coarse.entries[row].minimumPhi=sample.x;coarse.entries[row].maximumPhi=sample.x;
 coarse.entries[row].flags&=~PHI_INTERFACE;}
// Same 2D split as the host launch: width=min(groups,limit), height=ceil.
// Kernels index through num_workgroups, so equal triples mean equal items.
fn publishDispatchRecord(slot:u32,groups:u32){let width=min(groups,p.maxWorkgroups);
 var height=1u;if(width!=0u){height=(groups+width-1u)/width;}
 dispatchArgs[3u*slot]=width;dispatchArgs[3u*slot+1u]=height;dispatchArgs[3u*slot+2u]=1u;}
@compute @workgroup_size(1)fn prepareSummaryDispatch(){
 // The dense chain launches zero workgroups on an unpublished air support, so
 // this singleton carries predictSummaryCells's lane-zero receipt diagnostics:
 // one record per declined advance, naming the failing control words.
 if(!trackerRunnable()){atomicAdd(&state[21],1u);
  if(p.airControl+15u<arrayLength(&air)){atomicStore(&state[22],air[p.airControl]);
   atomicStore(&state[23],air[p.airControl+13u]);atomicStore(&state[24],air[p.airControl+14u]);
   atomicStore(&state[26],air[p.airControl+1u]);}
  else{atomicStore(&state[25],1u);}}
 publishDispatchRecord(0u,select(0u,(p.domainVolume+255u)/256u,trackerRunnable()));
 let rows=min(coarse.rowCount,p.rowCapacity);
 publishDispatchRecord(1u,select(0u,(rows+255u)/256u,coarse.state==PUBLISHED));
 publishDispatchRecord(2u,0u);}
@compute @workgroup_size(1)fn prepareSummaryFinalizeDispatch(){
 publishDispatchRecord(2u,(min(atomicLoad(&state[1]),p.entryCapacity)+255u)/256u);}
`;
