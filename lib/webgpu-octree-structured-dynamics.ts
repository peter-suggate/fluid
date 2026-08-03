/** Class-specialized timestep kernels over the packed six-family authority. */

import type { OctreePowerTopologySource } from "./webgpu-octree-power-topology";
import {
  OCTREE_STRUCTURED_GPU_ERROR,
  OCTREE_STRUCTURED_GPU_TRANSFER_LIST_OFFSET_WORDS,
  OCTREE_STRUCTURED_TOPOLOGY_TRANSFER_DISPATCH_OFFSET_BYTES,
  type DirectStructuredVelocitySource,
} from "./webgpu-octree-structured-velocity-gpu";
import {
  OCTREE_AIR_SUPPORT_LAYOUT_VERSION,
  OCTREE_AIR_SUPPORT_OWNER_HASH,
  OCTREE_AIR_SUPPORT_VALID,
  octreeAirSupportOwnerHashStartWGSL,
  type OctreeAirVelocitySupportLayout,
} from "./webgpu-octree-air-velocity-support";
import type { PassBroker } from "./webgpu-pass-broker";
import { octreeAlgorithmDiagnosticsEnabled } from "./octree-algorithm-diagnostics";
import { GPU_RIGID_BODY_CAPACITY } from "./webgpu-rigid-body";
import type { SurfaceInflowState } from "./webgpu-quadtree-builder";

const STRUCTURED_ROW_UNION_DISPATCH_OFFSET_BYTES = 9 * 12;
const STRUCTURED_FAMILY_UNION_DISPATCH_OFFSET_BYTES = 10 * 12;
// Record 4 is repurposed for the flattened class-7/8 face grid, so the
// dry-identity RHS zero dispatches from this dedicated tail record.
const STRUCTURED_IDENTITY_RHS_DISPATCH_OFFSET_BYTES = 11 * 12;
type StructuredDynamicsPipelineBundle = Readonly<Record<string, GPUComputePipeline>>;
type StructuredDynamicsPipelineProgress =
  (entryPoint: string, completed: number, total: number) => void;
interface StructuredDynamicsPipelineCompilation {
  readonly promise: Promise<StructuredDynamicsPipelineBundle>;
  readonly listeners: Set<StructuredDynamicsPipelineProgress>;
}
const structuredDynamicsPipelineCache = new WeakMap<GPUDevice,
  Map<string, StructuredDynamicsPipelineBundle>>();
const structuredDynamicsPipelineCompilations = new WeakMap<GPUDevice,
  Map<string, StructuredDynamicsPipelineCompilation>>();
/** Class four is unused by recurring dynamics, so prepare reuses its indirect
 * record for one workgroup per class-7/8 boundary face. */
export const STRUCTURED_BOUNDARY_DRY_PROBE_DISPATCH_OFFSET_BYTES = 4 * 12;

/**
 * Default-on A/B switch for flattening the order-free class-7/8 carry test.
 *
 * A zero restores the former per-face serial six-neighbour walks exactly.
 * Selection is construction-stable so one dynamics instance never mixes
 * cached and uncached carry decisions.
 */
export function structuredBoundaryAdvectionFlatteningEnabled(
  environment: Readonly<Record<string, string | undefined>> | undefined =
    typeof process !== "undefined" ? process.env : undefined,
): boolean {
  return environment?.FLUID_STRUCTURED_BOUNDARY_ADVECT_FLAT !== "0";
}

/** Diagnostic A/B for the exact-identity deep-interior carry optimization. */
export function structuredDeepIdentityCarryEnabled(
  environment: Readonly<Record<string, string | undefined>> | undefined =
    typeof process !== "undefined" ? process.env : undefined,
): boolean {
  return environment?.FLUID_STRUCTURED_DEEP_IDENTITY_CARRY !== "0";
}

/**
 * Default-on collapse of dispatch handoffs that exchange only ordinary
 * storage. A zero restores the historical pass splits for exact A/B.
 */
export function structuredDynamicsPlainStoragePassCompactionEnabled(
  environment: Readonly<Record<string, string | undefined>> | undefined =
    typeof process !== "undefined" ? process.env : undefined,
): boolean {
  return environment?.FLUID_STRUCTURED_DYNAMICS_COMPACT_PASS !== "0";
}

/** CPU oracle for the shader's independent six-direction Boolean reduction. */
export function structuredRowTouchesDryProbeOracle(
  liquid: boolean,
  directionTouchesDry: readonly boolean[],
): boolean {
  if (directionTouchesDry.length !== 6) {
    throw new RangeError("Structured dry-row oracle requires six directional probes");
  }
  return !liquid || directionTouchesDry.some(Boolean);
}
/**
 * Bounded GPU relaxation budget for the Section 5 ordinary-face march.
 *
 * Aanjaneya et al. prescribe copying from the face geometrically closest to
 * the free surface, not widening the extrapolation domain until a particular
 * scene happens to pass. The final encoded wave remains a fixed-point proof:
 * support that cannot converge within this production budget is rejected by
 * the publication rather than admitted through a wider numerical stencil.
 */
export const STRUCTURED_VELOCITY_EXTENSION_LAYERS = 6;
export const STRUCTURED_PROJECTION_ENERGY_WORDS = 32;

/** Workgroup size of every slot-shaped structured kernel. */
export const STRUCTURED_SLOT_WORKGROUP_SIZE = 64;
/** `maxComputeWorkgroupsPerDimension` floor required of every WebGPU device. */
const MAXIMUM_WORKGROUPS_PER_DIMENSION = 65_535;

/**
 * Two-dimensional dispatch for a slot-capacity-shaped kernel.
 *
 * `slotCapacity` is `pressureRowCapacity * maximumFaceIncidence`, so it passes
 * the one-dimensional 65,535-workgroup ceiling at roughly 140,000 pressure
 * rows -- well inside the domains the adaptive octree is meant to carry. The X
 * extent is pinned at exactly 65,535 whenever it saturates, which is what lets
 * the kernel recover a linear slot handle from a constant stride instead of a
 * uniform. Below saturation Y is 1 and the stride is never applied.
 */
export function structuredSlotDispatch(slotCapacity: number): readonly [number, number, number] {
  if (!Number.isSafeInteger(slotCapacity) || slotCapacity < 1) {
    throw new RangeError("Structured slot dispatch capacity must be a positive integer");
  }
  const blocks = Math.ceil(slotCapacity / STRUCTURED_SLOT_WORKGROUP_SIZE);
  const x = Math.min(MAXIMUM_WORKGROUPS_PER_DIMENSION, blocks);
  return [x, Math.ceil(blocks / x), 1];
}

/**
 * Whether the four-stage kinetic-energy probe is encoded this run.
 *
 * The probe is a diagnostic but it is NOT write-only, so it cannot simply be
 * deleted or gated on `FLUID_ALGORITHM_DIAGNOSTICS` alone. `run-webgpu-smoke`
 * promotes an invalid or generation-incoherent energy record into a hard
 * `generationFailures` entry ("structured projection energy is invalid or
 * generation-incoherent") for every captured audit snapshot, and it captures
 * those snapshots whenever `FLUID_POWER_GENERATION_AUDIT=1` or
 * `FLUID_STABILITY_ENVELOPE=1`. The 500-step acceptance lane
 * (`test:webgpu:minimal-power-dam-break`) sets both, with
 * `FLUID_POWER_AUDIT_EVERY_STEPS=1`, so the probe must stay on for every step
 * there. `FLUID_ENERGY_EVERY_STEPS>0` additionally logs the four stage
 * energies through `readStats`.
 *
 * The throughput lanes read none of it: `tools/benchmark-power-dam.ts` sets
 * `FLUID_ALGORITHM_DIAGNOSTICS=0` and `FLUID_STABILITY_ENVELOPE=0` unless
 * `--algorithm-diagnostics` / `--artifact` is given, and never sets
 * `FLUID_POWER_GENERATION_AUDIT`. There the four `@workgroup_size(128)`
 * summarizers -- each dispatched as a fixed one-workgroup grid that sweeps the
 * entire class 5-8 face workset, the stage-1 one also running a full
 * `staggeredSample` per wet face -- are pure cost.
 *
 * When the probe is off, `projectionEnergyStats` keeps its zero-initialized
 * (or stale) contents and `decodeStructuredProjectionEnergy` fails closed on
 * `epoch === 0`, so no host ever observes a fabricated zero-energy sample.
 * `FLUID_STRUCTURED_ENERGY_PROBE=1|0` forces the decision either way.
 */
export function structuredProjectionEnergyProbeEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  const forced = resolved?.FLUID_STRUCTURED_ENERGY_PROBE;
  if (forced === "1") return true;
  if (forced === "0") return false;
  return octreeAlgorithmDiagnosticsEnabled(resolved)
    || resolved?.FLUID_STABILITY_ENVELOPE === "1"
    || resolved?.FLUID_POWER_GENERATION_AUDIT === "1"
    || Number(resolved?.FLUID_ENERGY_EVERY_STEPS ?? 0) > 0;
}

export interface StructuredProjectionEnergySample {
  readonly epoch: number;
  readonly activeBank: 0 | 1;
  readonly familySampleCount: number;
  readonly startKineticEnergyProxy: number;
  readonly postAdvectionKineticEnergyProxy: number;
  readonly preProjectionKineticEnergyProxy: number;
  readonly postProjectionKineticEnergyProxy: number;
  readonly wetStartKineticEnergyProxy: number;
  readonly wetPostAdvectionKineticEnergyProxy: number;
  readonly wetPreProjectionKineticEnergyProxy: number;
  readonly wetPostProjectionKineticEnergyProxy: number;
  readonly wetFaceCount: number;
  /** Wet energies re-weighted by the face liquid fraction theta (1/pressure
   * scale): the variational face mass, the closest face-based analogue of the
   * physical rho/2 integral |u|^2 over liquid. The unweighted proxies above
   * charge a barely-wet surface face its full dual volume. */
  readonly wetStartThetaEnergyProxy: number;
  readonly wetPostAdvectionThetaEnergyProxy: number;
  readonly wetPreProjectionThetaEnergyProxy: number;
  readonly wetPostProjectionThetaEnergyProxy: number;
  readonly staggeredPathCount: number;
  readonly projectionEnergyRatio: number;
}

export interface StructuredProjectionEnergyDecode {
  readonly sample: StructuredProjectionEnergySample | null;
  readonly blocker: string | null;
}

/** Decode the one-generation four-stage reduction (start-of-step, post-
 * advection, post-force/pre-projection, post-projection). An initialized or
 * partial buffer never becomes a zero-energy observation. */
export function decodeStructuredProjectionEnergy(
  words: ArrayLike<number>,
): StructuredProjectionEnergyDecode {
  if (words.length < STRUCTURED_PROJECTION_ENERGY_WORDS) {
    return Object.freeze({ sample: null, blocker: "structured projection-energy report is truncated" });
  }
  const stages = [0, 1, 2, 3].map((stage) => ({
    flags: Number(words[8 * stage]) >>> 0,
    epochAndBank: Number(words[8 * stage + 1]) >>> 0,
    count: Number(words[8 * stage + 2]) >>> 0,
    energyBits: Number(words[8 * stage + 3]) >>> 0,
    wetCount: Number(words[8 * stage + 4]) >>> 0,
    wetEnergyBits: Number(words[8 * stage + 5]) >>> 0,
    wetThetaEnergyBits: Number(words[8 * stage + 6]) >>> 0,
    staggeredPathCount: Number(words[8 * stage + 7]) >>> 0,
  }));
  const failed = stages.find((stage) => stage.flags !== 0);
  if (failed) return Object.freeze({ sample: null,
    blocker: `structured projection-energy reduction failed with flags ${failed.flags}` });
  const epoch = stages[0]!.epochAndBank >>> 1;
  const activeBank = stages[0]!.epochAndBank & 1;
  if (epoch === 0 || stages.some((stage) => stage.epochAndBank !== stages[0]!.epochAndBank)) {
    return Object.freeze({ sample: null,
      blocker: "structured projection-energy stages are unpublished or generation-incoherent" });
  }
  if (stages[0]!.count === 0 || stages.some((stage) => stage.count !== stages[0]!.count)) {
    return Object.freeze({ sample: null,
      blocker: "structured projection-energy stages have incomplete family coverage" });
  }
  const bits = new Uint32Array(stages.flatMap((stage) =>
    [stage.energyBits, stage.wetEnergyBits, stage.wetThetaEnergyBits]));
  const energy = new Float32Array(bits.buffer);
  if (Array.from(energy).some((value) => !Number.isFinite(value) || value < 0)) {
    return Object.freeze({ sample: null,
      blocker: "structured projection-energy stages contain invalid energy" });
  }
  const pre = energy[6]!;
  const post = energy[9]!;
  return Object.freeze({ sample: Object.freeze({
    epoch, activeBank: activeBank as 0 | 1, familySampleCount: stages[0]!.count,
    startKineticEnergyProxy: energy[0]!,
    postAdvectionKineticEnergyProxy: energy[3]!,
    preProjectionKineticEnergyProxy: pre,
    postProjectionKineticEnergyProxy: post,
    wetStartKineticEnergyProxy: energy[1]!,
    wetPostAdvectionKineticEnergyProxy: energy[4]!,
    wetPreProjectionKineticEnergyProxy: energy[7]!,
    wetPostProjectionKineticEnergyProxy: energy[10]!,
    wetStartThetaEnergyProxy: energy[2]!,
    wetPostAdvectionThetaEnergyProxy: energy[5]!,
    wetPreProjectionThetaEnergyProxy: energy[8]!,
    wetPostProjectionThetaEnergyProxy: energy[11]!,
    wetFaceCount: stages[1]!.wetCount,
    staggeredPathCount: stages[1]!.staggeredPathCount,
    projectionEnergyRatio: pre === 0 ? 1 : post / pre,
  }), blocker: null });
}

/**
 * Diagnostic-only: read the GPU-authored per-class indirect dispatch widths back
 * to the host.
 *
 * Union dispatches run one workgroup of 64 lanes per 64 workset entries,
 * so the recorded X dimension IS the occupancy ceiling of every family kernel:
 * a class holding 430 faces can never put more than seven workgroups on a
 * 32-core GPU no matter how expensive each face is. Nothing on the host knows
 * that number otherwise -- the worksets are published entirely GPU-side by
 * `publishStructuredBoundaryWorksets` -- and profiling the family kernels
 * without it invites the exact misreading this census was added to settle
 * (see the class 5/7/8 split in the 2026-07-28 mini-dam measurement).
 *
 * Off by default: it stages a 108-byte copy and forces a pass boundary.
 */
export function structuredWorksetCensusEnabled(
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): boolean {
  return environment?.FLUID_WORKSET_CENSUS === "1";
}

export interface StructuredDynamicsResources {
  readonly structured: DirectStructuredVelocitySource;
  readonly topology: OctreePowerTopologySource;
  readonly pressure: GPUBuffer;
  readonly divergenceRhs: GPUBuffer;
  readonly separationMask: GPUBuffer;
  /** Accepted dynamic-boundary classification, one u32 liquid bit per banked row. */
  readonly liquidMask: GPUBuffer;
  /** Required normal velocity of the solid boundary, one f32 per family handle
   * in each of the two structured banks. The boundary producer writes the bank
   * named by the accepted structured publication, so every read here must apply
   * the same `bank()*slotCapacity` base. */
  readonly solidNormalVelocities: GPUBuffer;
  /** Authoritative GPU rigid state; the same 32-float-per-body records the
   * boundary producer samples for apertures and solid normal velocities. */
  readonly rigidBodies: GPUBuffer;
  /** Construction-stable active roster. Zero omits the four body-impulse
   * pipelines; the scene rebuilds the solver when this roster changes. */
  readonly bodyCount: number;
  /** Resident twelve-i32-per-body fixed-point exchange the GPU rigid
   * integrator consumes. Slots 0..2 carry the linear impulse and 3..5 the
   * angular impulse, both scaled by 1e6; see `gpuRigidBodyShader`. */
  readonly rigidExchange: GPUBuffer;
  /** Current-step boundary-class worksets and their fail-closed publication. */
  readonly boundaryWorksets: GPUBuffer;
  readonly boundaryControl: GPUBuffer;
  /** Current compact-row adjacency for every byte-packed Delaunay selector. */
  readonly selectorRows: GPUBuffer;
  readonly selectorStride: number;
  readonly selectorOffsetWords: number;
  readonly airSupportLayout: OctreeAirVelocitySupportLayout;
  readonly dimensions: readonly [number, number, number];
  readonly physicalCellSize: number;
  readonly closedBoundaryMask: number;
}

/**
 * No stage consumes a general face or incidence record. The only recurring
 * writes are destination-owned family values, one RHS per row, and one
 * projected row vector. Section 5 face extrapolation is published by the
 * dedicated air-support producer after projection.
 */
export class WebGPUStructuredVelocityDynamics {
  readonly encodedAdvectionDispatchCount: 3 | 4;
  readonly encodedForceDivergenceDispatchCount = 4;
  readonly encodedProjectionDispatchCount = 4;
  readonly allocatedBytes: number;
  /** Per-slot vec4f: transported momentum xyz and kinetic dissipation w. */
  readonly transportMetrics: GPUBuffer;
  /** Failure-only topology readback metadata for the shared selector-row buffer. */
  readonly selectorStride: number;
  readonly selectorOffsetWords: number;
  readonly dimensions: readonly [number, number, number];
  /** Eight-word coherent pre/post projection kinetic-energy report. */
  readonly projectionEnergyStats: GPUBuffer;
  /** Immutable packed volume/slot/tetra catalog shared by direct consumers. */
  readonly catalog: GPUBuffer;
  readonly catalogOffsetsWords: readonly [number, number, number, number, number];
  /** Nine accepted class dispatch records followed by row/family union records. */
  readonly dispatch: GPUBuffer;
  /** Diagnostic-only A/B face values captured immediately after candidate
   * topology transfer, followed by accepted and candidate control headers. */
  readonly topologyTransferAudit?: GPUBuffer;
  /** Diagnostic-only accepted geometry/value and inactive-bank characteristic
   * records captured immediately after advection commit. */
  readonly advectionSymmetryAudit?: GPUBuffer;
  private readonly params: readonly [GPUBuffer, GPUBuffer, GPUBuffer];
  private shaderModule!: GPUShaderModule;
  private readonly pipelineCacheKey: string;
  private prepare!: GPUComputePipeline;
  private topologyTransfer!: GPUComputePipeline;
  private boundaryDryProbe?: GPUComputePipeline;
  private advection!: GPUComputePipeline;
  private advectionCommit!: GPUComputePipeline;
  private force!: GPUComputePipeline;
  private divergence!: GPUComputePipeline;
  private zeroIdentityRhs!: GPUComputePipeline;
  private separation!: GPUComputePipeline;
  private bodyImpulse?: GPUComputePipeline;
  private projection!: GPUComputePipeline;
  private reconstruct!: GPUComputePipeline;
  private summarizePreProjectionEnergy!: GPUComputePipeline;
  private summarizePostProjectionEnergy!: GPUComputePipeline;
  private summarizeStartEnergy!: GPUComputePipeline;
  private summarizePostAdvectionEnergy!: GPUComputePipeline;
  private pipelinesInitialized = false;
  private pipelineInitialization?: Promise<void>;
  private readonly groups = new WeakMap<GPUComputePipeline,
    WeakMap<GPUBuffer, WeakMap<GPUBuffer, GPUBindGroup>>>();
  /** Encode the four energy summarizers only when a host actually reads them. */
  private readonly projectionEnergyProbe = structuredProjectionEnergyProbeEnabled();
  /** Construction-stable exact A/B for the class-7/8 carry gate. */
  private readonly flattenedBoundaryAdvection = structuredBoundaryAdvectionFlatteningEnabled();
  /** Construction-stable diagnostic gate for the deep-interior identity carry. */
  private readonly deepIdentityCarry = structuredDeepIdentityCarryEnabled();
  /** Storage-only dispatch handoffs share a pass unless the legacy A/B asks. */
  private readonly compactPlainStoragePass =
    structuredDynamicsPlainStoragePassCompactionEnabled();
  /** Diagnostic-only per-class dispatch-width census; see `censusTick`. */
  private readonly censusEnabled = structuredWorksetCensusEnabled();
  private censusStaging?: GPUBuffer;
  private censusPhase: "idle" | "copied" | "mapping" = "idle";
  private censusStep = 0;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly resources: StructuredDynamicsResources,
    _deferPipelineCompilation = true) {
    const { structured, topology } = resources;
    if (!(resources.physicalCellSize > 0) || !Number.isFinite(resources.physicalCellSize)
      || resources.dimensions.some((value) => !Number.isSafeInteger(value) || value < 1)
      || resources.pressure.size < structured.plan.rowCapacity * 4
      || resources.divergenceRhs.size < structured.plan.rowCapacity * 4
      || resources.separationMask.size < resources.dimensions[0] * resources.dimensions[1] * resources.dimensions[2] * 4
      || resources.liquidMask.size < structured.plan.rowCapacity * 2 * 4
      || resources.solidNormalVelocities.size < structured.plan.slotCapacity * 2 * 4
      || resources.rigidBodies.size < GPU_RIGID_BODY_CAPACITY * 8 * 16
      || resources.rigidExchange.size < GPU_RIGID_BODY_CAPACITY * 12 * 4
      || !Number.isSafeInteger(resources.bodyCount) || resources.bodyCount < 0
      || resources.bodyCount > GPU_RIGID_BODY_CAPACITY
      || !Number.isSafeInteger(resources.selectorStride) || resources.selectorStride < 1
      || !Number.isSafeInteger(resources.selectorOffsetWords) || resources.selectorOffsetWords < 0
      || resources.selectorRows.size < (resources.selectorOffsetWords
        + structured.plan.rowCapacity * resources.selectorStride) * 4
      || resources.airSupportLayout.rowCapacity !== structured.plan.rowCapacity
      || resources.airSupportLayout.slotCapacity !== structured.plan.slotCapacity
      || resources.airSupportLayout.selectorTagOffsetWords !== resources.selectorOffsetWords
      || resources.airSupportLayout.selectorStride !== resources.selectorStride
      || resources.selectorRows.size < resources.airSupportLayout.totalBytes
      || !topology.catalogTetrahedronHeaders || !topology.catalogTetrahedra
      || !topology.catalogTetrahedronVertices) {
      throw new RangeError("Structured dynamics resources are invalid or undersized");
    }
    this.selectorStride = resources.selectorStride;
    this.selectorOffsetWords = resources.selectorOffsetWords;
    this.dimensions = resources.dimensions;
    const pieces = [topology.catalogVolumes, topology.catalogFaces,
      topology.catalogTetrahedronHeaders, topology.catalogTetrahedronVertices,
      topology.catalogTetrahedra] as const;
    const offsets: number[] = []; let catalogBytes = 0;
    for (const piece of pieces) { offsets.push(catalogBytes); catalogBytes += piece.size; }
    this.catalogOffsetsWords = offsets.map((offset) => offset / 4) as unknown as
      readonly [number, number, number, number, number];
    const maximumBinding = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    if (catalogBytes > maximumBinding) throw new RangeError("Structured dynamics catalog exceeds binding limits");
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.catalog = device.createBuffer({ label: "Structured dynamics immutable catalog", size: catalogBytes, usage: storage });
    const copy = device.createCommandEncoder({ label: "Install structured dynamics catalog" });
    pieces.forEach((piece, index) => copy.copyBufferToBuffer(piece, 0, this.catalog, offsets[index]!, piece.size));
    device.queue.submit([copy.finish()]);
    this.dispatch = device.createBuffer({ label: "Structured dynamics accepted indirect arguments",
      size: 12 * 12, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT
        | GPUBufferUsage.COPY_SRC });
    this.transportMetrics = resources.selectorRows;
    this.projectionEnergyStats = device.createBuffer({
      label: "Structured paired projection kinetic energy",
      size: STRUCTURED_PROJECTION_ENERGY_WORDS * 4,
      usage: storage,
    });
    if (typeof process !== "undefined" && process.env.FLUID_SYMMETRY_STAGE_AUDIT === "1") {
      this.topologyTransferAudit = device.createBuffer({
        label: "Structured topology-transfer symmetry audit",
        size: 2 * structured.plan.slotCapacity * 4 + 2 * 128
          + structured.plan.slotCapacity * 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      this.advectionSymmetryAudit = device.createBuffer({
        label: "Structured advection symmetry audit",
        // Header, then fifteen slot-shaped words for each authority bank:
        // value, owner, neighbor, metadata, area, inverse distance, fraction,
        // normal vec4, and centroid vec4.
        size: 128 + 2 * 15 * structured.plan.slotCapacity * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
    }
    this.params = [0, 1, 2].map((stage) => device.createBuffer({
      label: `Structured dynamics parameters ${stage}`, size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })) as unknown as readonly [GPUBuffer, GPUBuffer, GPUBuffer];
    const words = new Uint32Array(64), floats = new Float32Array(words.buffer);
    words.set([structured.plan.rowCapacity, structured.plan.slotCapacity,
      structured.plan.maximumCaseSlots, structured.plan.authorityWords,
      structured.plan.worksetStrideWords, structured.worksetBankStrideWords,
      ...resources.dimensions, resources.closedBoundaryMask >>> 0,
      offsets[0]! / 4, offsets[1]! / 4, offsets[2]! / 4, offsets[3]! / 4,
      offsets[4]! / 4, topology.catalogTetrahedronVertexCount ?? 0,
      topology.rowTemplateHeaderOffsetBytes / 4,
      topology.reconstructionDataOffsetBytes / 4,
    ], 0);
    words.set(Object.values(structured.plan.offsets), 18);
    // Diagnostic-only A/B: bit 0 captures topology-transfer intermediates;
    // bits 1..5 capture inactive authority owner/neighbor, metadata, scalar
    // geometry, normals, and centroids. This retains the production command
    // graph while isolating which supposedly-dead scratch changes it.
    const debugWriteMask = typeof process !== "undefined"
      ? Number(process.env.FLUID_STRUCTURED_DEBUG_WRITE_MASK ?? 0) : 0;
    words[39] = this.topologyTransferAudit ? 63
      : Number.isInteger(debugWriteMask) ? debugWriteMask & 63 : 0;
    floats[40] = resources.physicalCellSize;
    words[48] = resources.selectorOffsetWords;
    words[49] = resources.selectorStride;
    words[50] = resources.airSupportLayout.regularTagOffsetWords;
    words[51] = resources.airSupportLayout.controlOffsetWords;
    words[52] = resources.airSupportLayout.supportVectorOffsetWords;
    words[53] = resources.airSupportLayout.supportCapacity;
    words[54] = resources.airSupportLayout.ownerDirectoryOffsetWords;
    words[55] = resources.airSupportLayout.ownerDirectorySlotCapacity;
    for (const buffer of this.params) device.queue.writeBuffer(buffer, 0, words.buffer);
    // The roster captures every construction-stable reachability decision
    // (flattening, body exchange, and energy diagnostics). The audit bit names
    // the only source variant that can retain the same entry-point roster.
    this.pipelineCacheKey = `${this.advectionSymmetryAudit ? "advection-audit" : "production"}\0${
      this.deepIdentityCarry ? "deep-carry" : "trace-all"}\0${
      this.pipelineEntryPoints().join("\0")}`;
    this.encodedAdvectionDispatchCount = this.flattenedBoundaryAdvection ? 4 : 3;
    this.allocatedBytes = catalogBytes + this.dispatch.size
      + this.projectionEnergyStats.size
      + (this.topologyTransferAudit?.size ?? 0)
      + (this.advectionSymmetryAudit?.size ?? 0)
      + 3 * 256;
  }

  private pipelineEntryPoints(): readonly string[] {
    return [
      "prepareStructuredDynamics", "transferStructuredTopologyCandidate",
      ...(this.flattenedBoundaryAdvection ? ["classifyStructuredBoundaryDryProbes"] : []),
      this.flattenedBoundaryAdvection
        ? "advectStructuredFamiliesFlattenedBoundary" : "advectStructuredFamilies",
      "commitAdvectedStructuredFamilies", "forceStructuredFamilies",
      "divergenceStructuredRows", "zeroStructuredIdentityRhsRows", "separateStructuredRows",
      ...(this.resources.bodyCount === 0 ? [] : ["exchangeStructuredBodyImpulseRows"]),
      "projectStructuredFamilies", "reconstructStructuredRows",
      ...(this.projectionEnergyProbe ? [
        "summarizeStructuredPreProjectionEnergy", "summarizeStructuredPostProjectionEnergy",
        "summarizeStructuredStartEnergy", "summarizeStructuredPostAdvectionEnergy",
      ] : []),
    ];
  }

  private pipelineDescriptor(entryPoint: string): GPUComputePipelineDescriptor {
    return { label: entryPoint, layout: "auto",
      compute: { module: this.shaderModule, entryPoint, constants: {
        deepIdentityCarryEnabled: this.deepIdentityCarry ? 1 : 0,
      } } };
  }

  private assignPipelines(pipelines: Readonly<Record<string, GPUComputePipeline>>): void {
    this.prepare = pipelines.prepareStructuredDynamics!;
    this.topologyTransfer = pipelines.transferStructuredTopologyCandidate!;
    this.boundaryDryProbe = this.flattenedBoundaryAdvection
      ? pipelines.classifyStructuredBoundaryDryProbes : undefined;
    this.advection = pipelines[this.flattenedBoundaryAdvection
      ? "advectStructuredFamiliesFlattenedBoundary" : "advectStructuredFamilies"]!;
    this.advectionCommit = pipelines.commitAdvectedStructuredFamilies!;
    this.force = pipelines.forceStructuredFamilies!;
    this.divergence = pipelines.divergenceStructuredRows!;
    this.zeroIdentityRhs = pipelines.zeroStructuredIdentityRhsRows!;
    this.separation = pipelines.separateStructuredRows!;
    this.bodyImpulse = this.resources.bodyCount === 0
      ? undefined : pipelines.exchangeStructuredBodyImpulseRows!;
    this.projection = pipelines.projectStructuredFamilies!;
    this.reconstruct = pipelines.reconstructStructuredRows!;
    if (this.projectionEnergyProbe) {
      this.summarizePreProjectionEnergy = pipelines.summarizeStructuredPreProjectionEnergy!;
      this.summarizePostProjectionEnergy = pipelines.summarizeStructuredPostProjectionEnergy!;
      this.summarizeStartEnergy = pipelines.summarizeStructuredStartEnergy!;
      this.summarizePostAdvectionEnergy = pipelines.summarizeStructuredPostAdvectionEnergy!;
    }
    this.pipelinesInitialized = true;
  }

  async initializePipelines(
    onProgress: StructuredDynamicsPipelineProgress = () => {},
  ): Promise<void> {
    if (this.destroyed) throw new Error("Structured dynamics is destroyed");
    if (this.pipelinesInitialized) return;
    if (this.pipelineInitialization) return this.pipelineInitialization;
    this.pipelineInitialization = (async () => {
      const entryPoints = this.pipelineEntryPoints();
      let deviceCache = structuredDynamicsPipelineCache.get(this.device);
      if (!deviceCache) {
        deviceCache = new Map();
        structuredDynamicsPipelineCache.set(this.device, deviceCache);
      }
      let pipelines = deviceCache.get(this.pipelineCacheKey);
      if (!pipelines) {
        this.shaderModule = this.device.createShaderModule({
          label: "Structured velocity dynamics", code: structuredVelocityDynamicsWGSL,
        });
        let compilations = structuredDynamicsPipelineCompilations.get(this.device);
        if (!compilations) {
          compilations = new Map();
          structuredDynamicsPipelineCompilations.set(this.device, compilations);
        }
        let compilation = compilations.get(this.pipelineCacheKey);
        if (!compilation) {
          const listeners = new Set<StructuredDynamicsPipelineProgress>();
          // Start on the next microtask so two instances initialized together
          // both subscribe before the first per-entry progress notification.
          const promise = Promise.resolve().then(async () => {
            const compiled: Record<string, GPUComputePipeline> = {};
            for (let index = 0; index < entryPoints.length; index += 1) {
              const entryPoint = entryPoints[index]!;
              onProgress(entryPoint, index, entryPoints.length);
              for (const listener of listeners) listener(entryPoint, index, entryPoints.length);
              compiled[entryPoint] = await this.device.createComputePipelineAsync(
                this.pipelineDescriptor(entryPoint));
              onProgress(entryPoint, index + 1, entryPoints.length);
              for (const listener of listeners) listener(entryPoint, index + 1, entryPoints.length);
            }
            return Object.freeze(compiled);
          }).then((compiled) => {
            const published = deviceCache!.get(this.pipelineCacheKey) ?? compiled;
            deviceCache!.set(this.pipelineCacheKey, published);
            return published;
          }).finally(() => { compilations!.delete(this.pipelineCacheKey); });
          compilation = { promise, listeners };
          compilations.set(this.pipelineCacheKey, compilation);
        } else {
          compilation.listeners.add(onProgress);
        }
        try {
          pipelines = await compilation.promise;
        } finally {
          compilation.listeners.delete(onProgress);
        }
      }
      this.assignPipelines(pipelines);
    })();
    return this.pipelineInitialization;
  }

  private requirePipelines(): void {
    if (!this.pipelinesInitialized) throw new Error("Structured dynamics pipelines are not initialized");
  }

  private update(stage: 0 | 1 | 2, dt: number, density: number,
    gravity: readonly [number, number, number], inflow?: SurfaceInflowState): GPUBuffer {
    if (!(dt >= 0) || !Number.isFinite(dt) || !(density > 0) || !Number.isFinite(density)
      || gravity.some((value) => !Number.isFinite(value))) throw new RangeError("Invalid structured dynamics parameters");
    const bytes = new ArrayBuffer(28), floats = new Float32Array(bytes);
    // physical.w is otherwise padding. Carry the authored dynamics stage so
    // the already-required prepare singleton can reset the corridor-miss
    // tripwire exactly once per substep, without another dispatch or host
    // scheduling decision.
    floats[0] = dt; floats[1] = density; floats[2] = stage; floats.set(gravity, 3);
    this.device.queue.writeBuffer(this.params[stage], 164, bytes);
    const inflowBytes = new ArrayBuffer(32), inflowFloats = new Float32Array(inflowBytes);
    if (inflow) {
      inflowFloats.set([inflow.outletCenter_m.x, inflow.outletCenter_m.y,
        inflow.outletCenter_m.z, inflow.radius_m], 0);
      const desired = [inflow.velocity_m_s.x * inflow.strength,
        inflow.velocity_m_s.y * inflow.strength,
        inflow.velocity_m_s.z * inflow.strength];
      // The aperture normalization is scalar flux calibration. Applying it to
      // one selected Cartesian component rotates non-axis-aligned nozzles.
      // Scale the authored vector as a vector and keep w as an enabled flag.
      for (let component = 0; component < 3; component += 1) {
        desired[component] *= inflow.apertureScale;
      }
      inflowFloats.set([...desired, inflow.strength > 0 ? 1 : 0], 4);
    }
    this.device.queue.writeBuffer(this.params[stage], 224, inflowBytes);
    return this.params[stage];
  }

  /**
   * Publish the coupling roster into the projection-stage parameter word the
   * body-impulse adjoint reads. Only stage 2 runs that kernel; advection and
   * the force/divergence stage never sample a body.
   */
  private updateCouplingBodies(count: number): void {
    if (!Number.isSafeInteger(count) || count < 0 || count > this.resources.bodyCount) {
      throw new RangeError("Structured dynamics coupling body count is out of range");
    }
    this.device.queue.writeBuffer(this.params[2], 152, new Uint32Array([count]));
  }

  private entries(params: GPUBuffer, pressure: GPUBuffer): Readonly<Record<number, GPUBuffer>> {
    const { structured, topology, divergenceRhs, liquidMask } = this.resources;
    return { 0: params, 1: structured.control, 2: structured.authority,
      3: structured.rowGeometry, 4: structured.rowVelocities, 5: topology.metrics,
      6: this.catalog, 11: this.resources.boundaryWorksets,
      12: this.dispatch, 13: pressure, 14: divergenceRhs, 15: this.resources.separationMask, 16: liquidMask,
      17: this.resources.boundaryControl, 18: this.transportMetrics,
      22: this.resources.solidNormalVelocities, 23: this.projectionEnergyStats,
      24: structured.candidateControl, 25: this.resources.rigidBodies,
      26: this.resources.rigidExchange };
  }

  private group(pipeline: GPUComputePipeline, entries: readonly number[], params: GPUBuffer,
    pressure = this.resources.pressure): GPUBindGroup {
    let byParams = this.groups.get(pipeline);
    if (!byParams) { byParams = new WeakMap(); this.groups.set(pipeline, byParams); }
    let byPressure = byParams.get(params);
    if (!byPressure) { byPressure = new WeakMap(); byParams.set(params, byPressure); }
    const cached = byPressure.get(pressure); if (cached) return cached;
    const resources = this.entries(params, pressure);
    const group = this.device.createBindGroup({ label: `${pipeline.label} bindings ${entries.join(",")}`,
      layout: pipeline.getBindGroupLayout(0), entries: entries.map((binding) => ({
      binding, resource: { buffer: resources[binding]! },
    })) });
    byPressure.set(pressure, group); return group;
  }

  /**
   * Do not hoist this to run once per substep. The three call sites look
   * redundant because the kernel reads no per-stage parameter word, but the
   * argument it publishes is gated on `acc(0u)==0u`, and `accepted` is the
   * read_write structured control that every class kernel ORs a rejection code
   * into. Re-preparing is what fail-closes the rest of the substep: an
   * advection that rejects a sample zeroes the force, divergence, projection
   * and reconstruction dispatch arguments. One shared prepare would let those
   * stages keep running over a field already known to be invalid.
   *
   * The fence is required for the usual reason: this writes `this.dispatch` as
   * storage and the union kernels read the same buffer as an indirect argument.
   */
  private encodePrepare(broker: PassBroker, params: GPUBuffer): void {
    const pass = broker.compute({ label: "Prepare accepted structured dynamics worksets" });
    pass.setPipeline(this.prepare); pass.setBindGroup(0, this.group(this.prepare, [0, 1, 11, 12, 17], params));
    pass.dispatchWorkgroups(1); broker.fence("structured indirect arguments published");
  }

  private encodeProjectionEnergy(broker: PassBroker, params: GPUBuffer,
    phase: "start" | "advected" | "pre" | "post"): void {
    // Diagnostic-only work, but consumed by a hard smoke gate whenever the
    // audit/stability-envelope lanes ask for it -- see
    // `structuredProjectionEnergyProbeEnabled`. The four call sites keep their
    // exact substep positions so the stage ordering stays authored here.
    if (!this.projectionEnergyProbe) return;
    const pipeline = { start: this.summarizeStartEnergy,
      advected: this.summarizePostAdvectionEnergy,
      pre: this.summarizePreProjectionEnergy,
      post: this.summarizePostProjectionEnergy }[phase];
    const pass = broker.compute({ label: `Structured dynamics report ${phase} kinetic energy` });
    pass.setPipeline(pipeline);
    // The energy entry points read the accepted value bank through binding 3;
    // binding 4 (rowVelocity) is only used by reconstruction kernels. Dawn's
    // auto layout therefore omits it, and supplying it makes bind-group
    // creation fail validation during the t=0 authority warmup.
    pass.setBindGroup(0, this.group(pipeline, [0, 1, 2, 3, 5, 11, 16, 17, 18, 23], params));
    pass.dispatchWorkgroups(1);
  }

  private encodeUnion(broker: PassBroker, pipeline: GPUComputePipeline,
    union: "rows" | "families", bindings: readonly number[], params: GPUBuffer,
    label: string, pressure?: GPUBuffer): void {
    const pass = broker.compute({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.group(pipeline, bindings, params, pressure));
    pass.dispatchWorkgroupsIndirect(this.dispatch, union === "rows"
      ? STRUCTURED_ROW_UNION_DISPATCH_OFFSET_BYTES
      : STRUCTURED_FAMILY_UNION_DISPATCH_OFFSET_BYTES);
  }

  /**
   * Stage one 108-byte copy of the accepted class dispatch record, then decode
   * the PREVIOUS frame's copy. Splitting the copy and the map across two frames
   * is what makes the reading trustworthy: `mapAsync` issued in the same frame
   * resolves against the queue serial that existed before this encoder was
   * submitted, so it would report the record from two steps earlier without
   * saying so.
   */
  private censusTick(broker: PassBroker): void {
    if (!this.censusEnabled || this.destroyed) return;
    this.censusStep += 1;
    const staging = this.censusStaging ??= this.device.createBuffer({
      label: "Structured dynamics workset census staging",
      size: 9 * 12, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    if (this.censusPhase === "copied") {
      this.censusPhase = "mapping";
      const step = this.censusStep;
      void staging.mapAsync(GPUMapMode.READ).then(() => {
        const words = [...new Uint32Array(staging.getMappedRange())];
        staging.unmap();
        this.censusPhase = "idle";
        console.log(JSON.stringify({ phase: "structured-workset-census", step,
          // Class 4 is unused; 0-3 are row classes and 5-8 the face families.
          workgroupsByClass: [0, 1, 2, 3, 4, 5, 6, 7, 8].map((cls) => words[3 * cls] ?? 0),
        }));
      }).catch(() => { this.censusPhase = "idle"; });
      return;
    }
    if (this.censusPhase !== "idle") return;
    // The broker's own copy, never its raw-encoder escape hatch: structured
    // dynamics must not reach the underlying encoder beneath the pressure
    // spine, and a diagnostic is not a reason to breach that. The invariant is
    // pinned textually by tests/webgpu-pass-broker.test.ts, so naming the
    // forbidden accessor even in a comment fails that test.
    broker.copyBufferToBuffer(this.dispatch, 0, staging, 0, 9 * 12);
    this.censusPhase = "copied";
  }

  encodeAdvection(broker: PassBroker, dt: number, inflow?: SurfaceInflowState): void {
    if (this.destroyed) throw new Error("Structured dynamics is destroyed");
    this.requirePipelines();
    const params = this.update(0, dt, 1, [0, 0, 0], inflow); this.encodePrepare(broker, params);
    this.censusTick(broker);
    this.encodeProjectionEnergy(broker, params, "start");
    if (this.boundaryDryProbe) {
      const pass = broker.compute({ label: "Flatten structured boundary carry probes" });
      pass.setPipeline(this.boundaryDryProbe);
      pass.setBindGroup(0, this.group(this.boundaryDryProbe,
        [0, 1, 2, 3, 11, 16, 17], params));
      pass.dispatchWorkgroupsIndirect(this.dispatch,
        STRUCTURED_BOUNDARY_DRY_PROBE_DISPATCH_OFFSET_BYTES);
    }
    this.encodeUnion(broker, this.advection, "families",
      [0, 1, 2, 3, 4, 5, 6, 11, 16, 17, 18], params, "Advect structured families");
    // The sampler writes only the inactive bank; commit runs as a later
    // dispatch and reads that bank as ordinary storage. In-pass dispatch
    // ordering closes the race without a pass boundary.
    if (!this.compactPlainStoragePass) {
      broker.fence("structured advected destinations staged");
    }
    this.encodeUnion(broker, this.advectionCommit, "families",
      [0, 1, 2, 11, 17, 18], params, "Commit advected structured families");
    // Keep this boundary: the next dynamics prepare rewrites `this.dispatch`
    // as storage after advection and commit consumed it as INDIRECT.
    broker.fence("structured indirect arguments retired after advection");
    const advectionSymmetryAudit = this.advectionSymmetryAudit;
    if (advectionSymmetryAudit) {
      const { structured } = this.resources;
      const plan = structured.plan, slotBytes = plan.slotCapacity * 4;
      broker.copyBufferToBuffer(structured.control, 0, advectionSymmetryAudit, 0, 128);
      const scalarFields = [plan.offsets.values, plan.offsets.ownerRows,
        plan.offsets.neighborRows, plan.offsets.metadata, plan.offsets.areas,
        plan.offsets.inverseDistances, plan.offsets.fractions] as const;
      for (let bank = 0; bank < 2; bank += 1) {
        const destinationBase = 128 + bank * 15 * slotBytes;
        scalarFields.forEach((field, index) => broker.copyBufferToBuffer(structured.authority,
          (bank * plan.authorityWords + field) * 4, advectionSymmetryAudit,
          destinationBase + index * slotBytes, slotBytes));
        broker.copyBufferToBuffer(structured.authority,
          (bank * plan.authorityWords + plan.offsets.normals) * 4,
          advectionSymmetryAudit, destinationBase + 7 * slotBytes, 4 * slotBytes);
        broker.copyBufferToBuffer(structured.authority,
          (bank * plan.authorityWords + plan.offsets.centroids) * 4,
          advectionSymmetryAudit, destinationBase + 11 * slotBytes, 4 * slotBytes);
      }
    }
    this.encodeProjectionEnergy(broker, params, "advected");
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic after structured advection commit");
    }
  }

  /** Transfer the accepted projected field onto faces created by a split,
   * merge, or changed interface stencil. Exact face identities are preserved
   * bit-for-bit by the publisher and bypass this interpolation. */
  encodeTopologyTransferCandidate(broker: PassBroker): void {
    if (this.destroyed) throw new Error("Structured dynamics is destroyed");
    this.requirePipelines();
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic before topology velocity transfer");
    }
    const params = this.params[0];
    const pass = broker.compute({ label: "Transfer accepted velocity to changed topology faces" });
    pass.setPipeline(this.topologyTransfer);
    pass.setBindGroup(0, this.group(this.topologyTransfer,
      [0, 1, 2, 3, 4, 5, 6, 16, 17, 18, 24], params));
    pass.dispatchWorkgroupsIndirect(this.resources.structured.liveRowDispatch,
      OCTREE_STRUCTURED_TOPOLOGY_TRANSFER_DISPATCH_OFFSET_BYTES);
    if (this.topologyTransferAudit) {
      broker.fence("capture topology-transfer symmetry audit");
      const { structured } = this.resources;
      const valueBytes = structured.plan.slotCapacity * 4;
      broker.copyBufferToBuffer(structured.authority,
        structured.plan.offsets.values * 4, this.topologyTransferAudit, 0, valueBytes);
      broker.copyBufferToBuffer(structured.authority,
        (structured.plan.authorityWords + structured.plan.offsets.values) * 4,
        this.topologyTransferAudit, valueBytes, valueBytes);
      broker.copyBufferToBuffer(structured.control, 0,
        this.topologyTransferAudit, 2 * valueBytes, 128);
      broker.copyBufferToBuffer(structured.candidateControl, 0,
        this.topologyTransferAudit, 2 * valueBytes + 128, 128);
      broker.copyBufferToBuffer(this.transportMetrics, 0,
        this.topologyTransferAudit, 2 * valueBytes + 256,
        structured.plan.slotCapacity * 16);
    }
    // The following reconstruction consumes only the transferred ordinary
    // storage and reuses the same already-published indirect buffer.
    if (!this.compactPlainStoragePass) {
      broker.fence("changed topology face velocities transferred");
    }
  }

  // The divergence RHS is dimensional: rhs = rho*flux/dt. `encodeProjection`
  // undoes it as v -= dt*grad(p)/rho, so both halves must be given the SAME
  // density. Passing 1 here while the projection received the scene density
  // scaled every pressure gradient by 1/rho, so the projection removed ~0.1%
  // of the divergence: the column free-fell under gravity but no horizontal
  // momentum was ever generated and the dam front did not advance.
  encodeForcesAndDivergence(broker: PassBroker, dt: number, density: number,
    gravity: readonly [number, number, number], inflow?: SurfaceInflowState): void {
    if (this.destroyed) throw new Error("Structured dynamics is destroyed");
    this.requirePipelines();
    const params = this.update(1, dt, density, gravity, inflow); this.encodePrepare(broker, params);
    this.encodeUnion(broker, this.force, "families",
      [0, 1, 2, 11, 16, 17, 22], params, "Force and constrain structured families");
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic before pre-projection energy summary");
    }
    this.encodeProjectionEnergy(broker, params, "pre");
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic after pre-projection energy summary");
    }
    this.encodeUnion(broker, this.divergence, "rows",
      [0, 1, 2, 5, 6, 11, 14, 16, 17, 22], params, "Fuse structured divergence RHS rows");
    // Disjoint row set from the union above, so pass order cannot matter.
    const zeroIdentity = broker.compute({ label: "Zero dry-identity divergence RHS rows" });
    zeroIdentity.setPipeline(this.zeroIdentityRhs);
    zeroIdentity.setBindGroup(0, this.group(this.zeroIdentityRhs, [0, 1, 11, 14, 17], params));
    zeroIdentity.dispatchWorkgroupsIndirect(this.dispatch,
      STRUCTURED_IDENTITY_RHS_DISPATCH_OFFSET_BYTES);
  }

  encodeProjection(broker: PassBroker, dt: number, density: number,
    gravity: readonly [number, number, number] = [0, 0, 0],
    pressure = this.resources.pressure,
    couplingBodyCount = 0,
    inflow?: SurfaceInflowState): void {
    if (this.destroyed) throw new Error("Structured dynamics is destroyed");
    this.requirePipelines();
    const params = this.update(2, dt, density, gravity, inflow); this.encodePrepare(broker, params);
    this.updateCouplingBodies(couplingBodyCount);
    // Publish the lagged unilateral-contact active set from the solved
    // pressure: rows holding tension against gravity-opposed closed world
    // faces are marked, and the NEXT step's boundary rebuild opens exactly
    // those faces. The solved pressure itself is never mutated, so this
    // step's projection and divergence bookkeeping stay exactly the
    // variational solution.
    this.encodeUnion(broker, this.separation, "rows",
      [0, 1, 2, 3, 5, 6, 11, 13, 15, 16, 17], params,
      "Mark structured overhead separation rows", pressure);
    // Read the solved pressure before the projection stage consumes it. Both
    // stages are read-only in `pressure`, so no fence separates them; the
    // exchange only writes the resident rigid buffer, which the integrator
    // reads once, after the whole advance.
    if (couplingBodyCount > 0) {
      this.encodeUnion(broker, this.bodyImpulse!, "rows",
        [0, 1, 2, 5, 6, 11, 13, 16, 17, 25, 26], params,
        "Exchange structured body impulse rows", pressure);
    }
    this.encodeUnion(broker, this.projection, "families",
      [0, 1, 2, 11, 13, 16, 17, 22], params,
      "Project structured families", pressure);
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic before post-projection energy summary");
    }
    this.encodeProjectionEnergy(broker, params, "post");
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic after post-projection energy summary");
    }
    this.encodeUnion(broker, this.reconstruct, "rows",
      [0, 1, 2, 4, 5, 6, 11, 17], params,
      "Reconstruct projected structured rows");
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.catalog.destroy(); this.dispatch.destroy();
    this.projectionEnergyStats.destroy();
    this.topologyTransferAudit?.destroy();
    this.advectionSymmetryAudit?.destroy();
    this.params.forEach((buffer) => buffer.destroy());
  }
}

export const structuredVelocityDynamicsWGSL = /* wgsl */ `
override deepIdentityCarryEnabled:bool=true;
struct P{rowCapacity:u32,slotCapacity:u32,maxSlots:u32,authorityWords:u32,worksetStride:u32,worksetBankStride:u32,dimensionX:u32,dimensionY:u32,dimensionZ:u32,closedMask:u32,denseOffset:u32,slotGeometryOffset:u32,tetraHeaderOffset:u32,tetraVertexOffset:u32,tetraOffset:u32,tetraVertexCount:u32,templateHeaderOffset:u32,reconstructionOffset:u32,valuesOffset:u32,ownerOffset:u32,neighborOffset:u32,metadataOffset:u32,areaOffset:u32,inverseOffset:u32,fractionOffset:u32,pressureScaleOffset:u32,normalOffset:u32,centroidOffset:u32,rowNeighborOffset:u32,rowReciprocalOffset:u32,rowOwnerMetadataOffset:u32,rowHandleOffset:u32,rowSignOffset:u32,rowCatalogOffset:u32,rowAxisOffset:u32,rowFamilyPrefixOffset:u32,rowFamilyHandleOffset:u32,rowFamilySlotOffset:u32,bodyCount:u32,padB:u32,physical:vec4f,gravity:vec4f,selectorOffsetWords:u32,selectorStride:u32,regularTagOffsetWords:u32,supportControlOffsetWords:u32,supportVectorOffsetWords:u32,supportCapacity:u32,ownerDirectoryOffsetWords:u32,ownerDirectorySlotCapacity:u32,inflowPositionRadius:vec4f,inflowVelocity:vec4f}
struct Metric{caseId:u32,transformAndFlags:u32,volume:f32,error:u32}
struct SlotGeometry{neighborOffsetSize:vec4f,areaCentroid:vec4f,normalInverseDistance:vec4f}

@group(0)@binding(0)var<uniform>p:P;
@group(0)@binding(1)var<storage,read_write>accepted:array<atomic<u32>>;
@group(0)@binding(2)var<storage,read_write>a:array<u32>;
@group(0)@binding(3)var<storage,read>rowGeometry:array<vec4u>;
@group(0)@binding(4)var<storage,read_write>rowVelocity:array<vec4f>;
@group(0)@binding(5)var<storage,read>metrics:array<Metric>;
@group(0)@binding(6)var<storage,read>catalog:array<u32>;
@group(0)@binding(11)var<storage,read>worksets:array<u32>;
@group(0)@binding(12)var<storage,read_write>indirect:array<u32>;
@group(0)@binding(13)var<storage,read>pressure:array<f32>;
@group(0)@binding(14)var<storage,read_write>rhs:array<f32>;
@group(0)@binding(15)var<storage,read_write>separationMask:array<u32>;
@group(0)@binding(16)var<storage,read>liquid:array<u32>;
@group(0)@binding(17)var<storage,read>boundaryControl:array<u32>;
@group(0)@binding(18)var<storage,read_write>transportMetrics:array<vec4f>;
@group(0)@binding(22)var<storage,read>solidNormalVelocities:array<f32>;
@group(0)@binding(23)var<storage,read_write>projectionEnergyStats:array<u32>;
@group(0)@binding(24)var<storage,read_write>candidate:array<atomic<u32>>;
struct RigidBody{positionShape:vec4f,dimensions:vec4f,orientation:vec4f,linearVelocity:vec4f,angularVelocity:vec4f,inverseMassInertia:vec4f,angularMomentumRestitution:vec4f,material:vec4f}
@group(0)@binding(25)var<storage,read>rigidBodies:array<RigidBody>;
@group(0)@binding(26)var<storage,read_write>rigidExchange:array<atomic<i32>>;

const INVALID:u32=0xffffffffu;
const ERROR_SAMPLE:u32=${OCTREE_STRUCTURED_GPU_ERROR.carry}u;
const SUPPORT_TAG:u32=0x80000000u;
const SUPPORT_VALID:u32=${OCTREE_AIR_SUPPORT_VALID}u;
const SUPPORT_LAYOUT_VERSION:u32=${OCTREE_AIR_SUPPORT_LAYOUT_VERSION}u;
const CANDIDATE_VALID:u32=${0x5356_454c}u;

fn acc(index:u32)->u32{return atomicLoad(&accepted[index]);}
fn rejectSample(stage:u32,index:u32){
  atomicOr(&accepted[0],ERROR_SAMPLE);
  atomicMin(&accepted[1],(stage<<24u)|(index&0x00ffffffu));
}
fn rejectVector(stage:u32,index:u32,detail:vec4f,cls:u32){
  atomicOr(&accepted[0],ERROR_SAMPLE);
  let packed=(stage<<24u)|(index&0x00ffffffu);
  let previous=atomicMin(&accepted[1],packed);
  if(packed<previous&&arrayLength(&accepted)>=11u){
    atomicStore(&accepted[6],bitcast<u32>(detail.x));
    atomicStore(&accepted[7],bitcast<u32>(detail.y));
    atomicStore(&accepted[8],bitcast<u32>(detail.z));
    atomicStore(&accepted[9],bitcast<u32>(detail.w));
    atomicStore(&accepted[10],cls);
  }
}
// Coverage census for the sparse Section-5 corridor. Words 11/12 hold the
// count of owner probes that found no published identity and the first such
// finest cell; the existing stage-0 prepare singleton resets them, so this
// adds no scan, pass, dispatch, or readback.
//
// This deliberately does NOT reject the generation. extendedOwnerVelocity is
// a speculative probe at all five of its call sites -- each one tests
// vectorValid and falls through to an incident-row or pinned-element sample
// -- so a cell outside the corridor is an ordinary "no published extension
// here" answer, not a coverage failure. Poisoning accepted[0] on that
// answer invalidated every later workset in the same advance. Under-coverage
// that would actually corrupt is caught where the value is consumed, by the
// existing publication receipts and the supportPublicationValid gate; this
// census is what makes a shrinking corridor visible before then.
fn countOutOfCorridorRead(q:vec3u){
  let d=dimensions();let cell=q.x+d.x*(q.y+d.y*q.z);
  if(arrayLength(&accepted)>=13u){
    atomicAdd(&accepted[11],1u);atomicMin(&accepted[12],cell);
  }
}
// A directory that cannot address its own arena is a layout fault. Unlike a
// corridor miss there is no correct answer to fall back to, so this keeps the
// fail-closed rejection.
fn rejectOwnerDirectoryBounds(q:vec3u){
  let d=dimensions();let cell=q.x+d.x*(q.y+d.y*q.z);
  countOutOfCorridorRead(q);
  atomicOr(&accepted[0],ERROR_SAMPLE);
  atomicMin(&accepted[1],(9u<<24u)|(cell&0x00ffffffu));
}
// The active authority bank, resolved once per invocation.
//
// The only writes to the accepted authority header in this module are the
// rejection helpers above (indices 0, 1, and 6..12) plus the stage-0 tripwire
// reset below. Indices 2 (pressure-row count), 3 (generation),
// 4 (active bank) and 5 (family handle count) are republished between
// dispatches, behind a fence, and are read-only for the whole lifetime of an
// invocation. The compiler cannot hoist this itself because accepted is an
// atomic read_write binding, so every acc() had to be re-issued as a
// device-scope atomic load: bank() alone was recomputed inside every
// abase()/rbase()/lbase()/sbase()/wbase() -- about seven per slot in
// divergenceRow and one per binary-search step in acceptedDirectoryFind.
var<private> bankWord:u32;
var<private> bankResolved:bool;
fn bank()->u32{if(!bankResolved){bankWord=acc(4u)&1u;bankResolved=true;}return bankWord;}
fn abase()->u32{return bank()*p.authorityWords;}
fn rbase()->u32{return bank()*p.rowCapacity;}
fn lbase()->u32{return bank()*p.rowCapacity;}
fn sbase()->u32{return bank()*p.slotCapacity;}
fn wbase(cls:u32)->u32{return bank()*p.worksetBankStride+cls*p.worksetStride;}
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}
fn finite3(v:vec3f)->bool{return all(v==v)&&all(abs(v)<=vec3f(3.402823e38));}
fn invalidVector()->vec4f{return vec4f(0.,0.,0.,-1.);}
fn vectorValid(v:vec4f)->bool{return v.w>0.&&finite3(v.xyz);}
fn canonicalVelocityDot(a:vec3f,b:vec3f)->f32{
  return canonicalInterpolation4(array<f32,4>(a.x*b.x,a.y*b.y,a.z*b.z,0.));}
fn bitsf(at:u32)->f32{return bitcast<f32>(catalog[at]);}
fn dimensions()->vec3u{return vec3u(p.dimensionX,p.dimensionY,p.dimensionZ);}
// Build row centres about the domain centre from an integer doubled-cell
// offset.  Direct (q + size/2) * h construction gives reflected rows
// unrelated last bits because the large positive coordinate is rounded before
// the symmetry is applied.  The centred offset is exactly odd under every
// horizontal D4 transform; adding the shared domain centre is postponed until
// the final world-space address is required.
fn domainWorldCenter()->vec3f{return .5*vec3f(dimensions())*p.physical.x;}
fn rowCenteredGridOffset(rg:vec4u)->vec3f{
  let d=dimensions();let q=vec3u(rg.x%d.x,(rg.x/d.x)%d.y,rg.x/(d.x*d.y));
  return .5*vec3f(vec3i(2u*q+vec3u(rg.y))-vec3i(d));
}
fn signs(code:u32)->vec3f{let b=code&7u;return vec3f(select(1.,-1.,(b&1u)!=0u),select(1.,-1.,(b&2u)!=0u),select(1.,-1.,(b&4u)!=0u));}
fn inverseTransform(v:vec3f,code:u32)->vec3f{let q=v*signs(code);let k=(code/8u)%6u;if(k==0u){return q;}if(k==1u){return q.xzy;}if(k==2u){return q.yxz;}if(k==3u){return q.zxy;}if(k==4u){return q.yzx;}return q.zyx;}
fn powerTransform(v:vec3f,code:u32)->vec3f{let k=(code/8u)%6u;var q=v;if(k==1u){q=v.xzy;}else if(k==2u){q=v.yxz;}else if(k==3u){q=v.yzx;}else if(k==4u){q=v.zxy;}else if(k==5u){q=v.zyx;}return q*signs(code);}
fn caseHeader(caseId:u32)->vec2u{let at=p.denseOffset+p.templateHeaderOffset+4u*caseId;return vec2u(catalog[at],catalog[at+1u]);}
fn geom(global:u32)->SlotGeometry{let at=p.slotGeometryOffset+12u*global;return SlotGeometry(vec4f(bitsf(at),bitsf(at+1u),bitsf(at+2u),bitsf(at+3u)),vec4f(bitsf(at+4u),bitsf(at+5u),bitsf(at+6u),bitsf(at+7u)),vec4f(bitsf(at+8u),bitsf(at+9u),bitsf(at+10u),bitsf(at+11u)));}
// boundaryControl is a read-only binding and the accepted words this reads
// (2, 3, 4) are never written here, so the publication header is settled for
// the whole invocation. It was being re-derived from five storage loads plus
// four device atomics on every sample tap, because supportPublicationValid
// calls it and taggedVelocity calls that first -- 8x per cellSample and up to
// 48x per staggeredSample.
var<private> boundaryValidWord:bool;
var<private> boundaryValidResolved:bool;
fn boundaryValid()->bool{
  if(!boundaryValidResolved){
    boundaryValidWord=arrayLength(&boundaryControl)>=7u&&boundaryControl[0]==0u&&boundaryControl[2]==acc(2u)&&boundaryControl[4]==acc(3u)&&boundaryControl[5]==bank()&&boundaryControl[6]==acc(3u);
    boundaryValidResolved=true;
  }
  return boundaryValidWord;
}
// Class dispatch records pin X at exactly 65,535 workgroups whenever a class
// saturates one dimension (see publishStructuredClassDispatch in
// webgpu-octree-structured-velocity-gpu.ts and its boundary twin), so a class
// item folds back with that constant stride. A class that fits one dimension
// dispatches Y=1 and never reaches the second term.
fn classItem(g:vec3u)->u32{return g.x+g.y*65535u*64u;}
fn workItem(cls:u32,index:u32)->u32{let base=wbase(cls);if(!boundaryValid()||worksets[base]!=acc(3u)||index>=worksets[base+1u]){return INVALID;}return worksets[base+7u+index];}
// Map one dense union invocation back to its original class-local index. The
// four published payloads remain untouched and retain their deterministic
// per-class ordering; only their dispatch records are collapsed.
fn unionClassItem(first:u32,index:u32)->vec2u{
  var local=index;
  for(var offset=0u;offset<4u;offset+=1u){
    let cls=first+offset;let count=worksets[wbase(cls)+1u];
    if(local<count){return vec2u(cls,local);}
    local-=count;
  }
  return vec2u(INVALID);
}
fn publishUnionDispatch(out:u32,first:u32){
  var valid=acc(0u)==0u&&acc(3u)!=0u&&boundaryValid();
  var count=0u;
  for(var offset=0u;offset<4u;offset+=1u){
    let base=wbase(first+offset);
    valid=valid&&worksets[base]==acc(3u);
    count+=worksets[base+1u];
  }
  let blocks=select(0u,(count+63u)/64u,valid);
  let x=min(65535u,blocks);
  var y=1u;if(x!=0u){y=(blocks+x-1u)/x;}
  indirect[out]=x;indirect[out+1u]=y;indirect[out+2u]=1u;
}
fn candidateTransferItem(index:u32)->u32{
 if(arrayLength(&candidate)<7u||atomicLoad(&candidate[0])!=CANDIDATE_VALID){return INVALID;}
 let at=${OCTREE_STRUCTURED_GPU_TRANSFER_LIST_OFFSET_WORDS}u+index;
 if(atomicLoad(&candidate[4])==0u||at>=arrayLength(&candidate)){return INVALID;}
 return atomicLoad(&candidate[at]);
}

@compute @workgroup_size(1)
fn prepareStructuredDynamics(){
  if(p.physical.w<.5&&arrayLength(&accepted)>=13u){
    atomicStore(&accepted[11],0u);atomicStore(&accepted[12],INVALID);
  }
  for(var cls=0u;cls<9u;cls+=1u){
    let base=wbase(cls);
    let valid=acc(0u)==0u&&acc(3u)!=0u&&boundaryValid()&&worksets[base]==acc(3u);
    let out=3u*cls;
    indirect[out]=select(0u,worksets[base+4u],valid);
    // Y and Z are published words, not constants: a slot-shaped class whose
    // block count passes 65,535 is recorded as (65535, ceil(blocks/65535), 1)
    // and classItem folds the two dimensions back.
    indirect[out+1u]=select(1u,worksets[base+5u],valid);
    indirect[out+2u]=select(1u,worksets[base+6u],valid);
  }
  // Dynamics has no class-4 recurring kernel. Reuse that otherwise-dead
  // record for one workgroup per class-7/8 face, concatenating the two
  // accepted worklists without materializing another list.
  let base7=wbase(7u);let base8=wbase(8u);
  let flatValid=acc(0u)==0u&&acc(3u)!=0u&&boundaryValid()
    &&worksets[base7]==acc(3u)&&worksets[base8]==acc(3u);
  let flatCount=select(0u,worksets[base7+1u]+worksets[base8+1u],flatValid);
  let flatX=min(65535u,flatCount);
  var flatY=1u;if(flatX!=0u){flatY=(flatCount+flatX-1u)/flatX;}
  indirect[12u]=flatX;
  indirect[13u]=flatY;
  indirect[14u]=1u;
  publishUnionDispatch(27u,0u);
  publishUnionDispatch(30u,5u);
  // Class 4's own record was repurposed above, so the dry-identity RHS zero
  // dispatches from a dedicated tail record instead.
  let base4=wbase(4u);
  let valid4=acc(0u)==0u&&acc(3u)!=0u&&boundaryValid()&&worksets[base4]==acc(3u);
  indirect[33u]=select(0u,worksets[base4+4u],valid4);
  indirect[34u]=select(1u,worksets[base4+5u],valid4);
  indirect[35u]=select(1u,worksets[base4+6u],valid4);
}

fn value(handle:u32)->f32{return bitcast<f32>(a[abase()+p.valuesOffset+handle]);}
fn setValue(handle:u32,v:f32){a[abase()+p.valuesOffset+handle]=bitcast<u32>(v);}
fn owner(handle:u32)->u32{return a[abase()+p.ownerOffset+handle];}
fn neighbor(handle:u32)->u32{return a[abase()+p.neighborOffset+handle];}
fn normal(handle:u32)->vec3f{let at=abase()+p.normalOffset+4u*handle;return vec3f(bitcast<f32>(a[at]),bitcast<f32>(a[at+1u]),bitcast<f32>(a[at+2u]));}
fn centroid(handle:u32)->vec3f{let at=abase()+p.centroidOffset+4u*handle;return vec3f(bitcast<f32>(a[at]),bitcast<f32>(a[at+1u]),bitcast<f32>(a[at+2u]));}
// Aanjaneya et al. 2017, equations (3)-(4), store u.n on power faces and form
// divergence from A_face(u.n). A source is therefore a prescribed face-normal
// boundary value: the RHS must see it, and pressure projection must not turn
// that authored boundary degree of freedom back into an unconstrained face.
fn inflowNormalVelocity(handle:u32)->vec2f{
  if(p.inflowVelocity.w<1.||p.inflowPositionRadius.w<=0.){return vec2f(0.);}
  // A velocity source is a short oriented Dirichlet plug. Since this power
  // discretization stores only u.n, every incident face normal in that plug
  // must receive dot(authoredVelocity,n). Prescribing only the cap fixes flux
  // but leaves tangential reconstruction underdetermined; prescribing only a
  // dominant-axis subset is both underdetermined and grid biased.
  let h=p.physical.x;
  let world=centroid(handle)+vec3f(-.5*f32(p.dimensionX)*h,0.,-.5*f32(p.dimensionZ)*h);
  let delta=world-p.inflowPositionRadius.xyz;
  let speed=length(p.inflowVelocity.xyz);if(speed<=1e-6){return vec2f(0.);}
  let direction=p.inflowVelocity.xyz/speed;let axial=dot(delta,direction);
  let radial=length(delta-axial*direction);let edge=max(.70710678*h,1e-6);
  var coverage=clamp(.5+.5*(p.inflowPositionRadius.w-radial)/edge,0.,1.);
  coverage=coverage*coverage*(3.-2.*coverage);
  let n=normal(handle);
  if(axial<-.55*h||axial>2.55*h||coverage<=0.){return vec2f(0.);}
  return vec2f(dot(p.inflowVelocity.xyz,n)*coverage,1.);
}
fn rowSlotCount(row:u32)->u32{return caseHeader(metrics[row].caseId).y;}
fn liquidAt(row:u32)->u32{return liquid[lbase()+row];}
fn solidVelocityAt(handle:u32)->f32{return solidNormalVelocities[sbase()+handle];}
fn velocitySample(row:u32)->vec4f{
  if(row>=acc(2u)){return invalidVector();}
  // Section 5 advection consumes the previous projected and extended field.
  // The final destination-gather layer publishes air rows back into this
  // canonical full-vector field. Keeping the choice out of this hot sampler
  // preserves the device's ten-storage-buffer advection contract.
  let sample=rowVelocity[rbase()+row];
  if(sample.w<=0.||!finite3(sample.xyz)){return invalidVector();}
  return vec4f(sample.xyz,1.);
}
fn supportWord(word:u32)->u32{
  if(word>=4u*arrayLength(&transportMetrics)){return INVALID;}
  return bitcast<vec4u>(transportMetrics[word/4u])[word&3u];
}
fn supportCount()->u32{return supportWord(p.supportControlOffsetWords+6u);}
// Fifteen supportWord loads plus boundaryValid(), previously re-evaluated on
// every single sample tap. The sixteen control words it reads live at
// p.supportControlOffsetWords, which the air-support arena places AFTER the
// whole slotCapacity*16-byte transport-metric region (planOctreeAirVelocity-
// Support sets selectorTagOffsetBytes = transportMetricBytes and puts the
// control block after the selector and regular tag regions). The only writes
// this module makes to transportMetrics are transportMetrics[handle] for
// handle < acc(5u) <= slotCapacity, strictly inside that first region -- the
// same disjointness the existing code already relies on, or advection would
// corrupt the support header it validates. So the publication header is
// invariant for the whole invocation.
var<private> supportValidWord:bool;
var<private> supportValidResolved:bool;
var<private> supportFailureWord:u32;
fn supportPublicationValid()->bool{
  if(supportValidResolved){return supportValidWord;}
  let base=p.supportControlOffsetWords;
  var valid=false;
  supportFailureWord=0u;
  let inBounds=base+16u<=4u*arrayLength(&transportMetrics);
  let boundaryReady=boundaryValid();
  supportFailureWord|=select(1u,0u,inBounds);
  supportFailureWord|=select(2u,0u,boundaryReady);
  if(inBounds&&boundaryReady){
    let count=supportWord(base+6u);let capacity=supportWord(base+7u);
    let faces=supportWord(base+10u);let seeds=supportWord(base+11u);
    supportFailureWord|=select(4u,0u,supportWord(base)==0u);
    supportFailureWord|=select(8u,0u,supportWord(base+1u)==INVALID);
    supportFailureWord|=select(16u,0u,supportWord(base+2u)==acc(3u));
    supportFailureWord|=select(32u,0u,supportWord(base+3u)==bank());
    supportFailureWord|=select(64u,0u,supportWord(base+4u)==boundaryControl[4u]);
    supportFailureWord|=select(128u,0u,supportWord(base+5u)==acc(2u));
    supportFailureWord|=select(256u,0u,capacity==p.supportCapacity&&count<=capacity);
    supportFailureWord|=select(512u,0u,supportWord(base+13u)==SUPPORT_VALID);
    supportFailureWord|=select(1024u,0u,supportWord(base+14u)==SUPPORT_LAYOUT_VERSION);
    // Words 8 and 9 count the two independent demand families: transition
    // selector vertices and regular 3x3x3 stencil sites. Uniform scenes can
    // legitimately publish support entirely through the regular family, so a
    // non-empty transaction requires either family rather than selectors
    // specifically.
    let demandCount=supportWord(base+8u)+supportWord(base+9u);
    supportFailureWord|=select(2048u,0u,seeds<=faces
      &&(count==0u||(demandCount>0u&&faces>0u&&seeds>0u)));
    valid=supportFailureWord==0u;
  }
  supportValidWord=valid;supportValidResolved=true;
  return valid;
}
fn taggedVelocity(tag:u32)->vec4f{
  if(tag==INVALID||!supportPublicationValid()){return invalidVector();}
  if((tag&SUPPORT_TAG)==0u){return velocitySample(tag);}
  let support=tag&0x7fffffffu;let count=supportCount();
  if(support>=count||support>=p.supportCapacity){return invalidVector();}
  let at=p.supportVectorOffsetWords/4u+support;
  if(at>=arrayLength(&transportMetrics)){return invalidVector();}
  let sample=transportMetrics[at];
  return select(invalidVector(),vec4f(sample.xyz,1.),vectorValid(sample));
}
${octreeAirSupportOwnerHashStartWGSL("ownerHashStart")}
fn extendedOwnerTag(q:vec3u)->u32{
  let d=dimensions();let capacity=p.ownerDirectorySlotCapacity;
  if(capacity==0u||any(q>=d)){return INVALID;}
  // Probe the containing dyadic identity at each authored leaf size; one hash
  // record represents an entire coarse leaf.
  var size=${OCTREE_AIR_SUPPORT_OWNER_HASH.maximumLeafSize}u;
  loop{let origin=(q/vec3u(size))*vec3u(size);
    let originCell=origin.x+d.x*(origin.y+d.y*origin.z);
    let start=ownerHashStart(originCell,size,capacity);
    for(var probe=0u;probe<min(capacity,${OCTREE_AIR_SUPPORT_OWNER_HASH.maximumProbes}u);probe+=1u){
      let at=p.ownerDirectoryOffsetWords
        +${OCTREE_AIR_SUPPORT_OWNER_HASH.recordWords}u*((start+probe)%capacity);
      // A directory that does not fit its own arena is an ABI fault, not a
      // corridor miss: reject the generation rather than answer from it.
      if(at>4u*arrayLength(&transportMetrics)||4u*arrayLength(&transportMetrics)-at<4u){
        rejectOwnerDirectoryBounds(q);return INVALID;}
      let storedKey=supportWord(at);if(storedKey==0u){break;}let storedOrigin=storedKey-1u;
      let storedSize=supportWord(at+1u);if(storedOrigin==originCell&&storedSize==size){
        let tag=supportWord(at+2u);return select(INVALID,tag,tag!=INVALID);}}
    if(size==1u){break;}size>>=1u;}
  countOutOfCorridorRead(q);return INVALID;
}
// Section 5 extrapolates the projected velocity outside the liquid before
// level-set transport. A newly wet pressure row therefore initializes from
// that accepted extended field when no old liquid row contains its face. The
// adaptive identity hash names the exact octree owner and its published
// row/support vector; this is the same authority consumed by fine transport.
//
// A power-face centroid may lie exactly on a regular-grid face. Selecting only
// floor(point/h) then makes the result depend on the final float32 rounding:
// reflection exchanges the low and high incident cells, but 14.000001 and
// 18.000000 both select their high cell. Section 5 defines one interpolated
// field, not an owner-sided trace. Preserve the single-owner fast path away
// from seams; on a face/edge/corner canonically fold each DISTINCT incident
// owner limit. Deduplication prevents a coarse owner from receiving extra
// weight merely because several finest directory cells name it.
fn extendedOwnerVelocity(point:vec3f)->vec4f{
  if(!finite3(point)||!supportPublicationValid()){return invalidVector();}
  let d=dimensions();let volume=d.x*d.y*d.z;
  if(volume==0u||p.ownerDirectorySlotCapacity==0u){return invalidVector();}
  let upper=max(vec3f(d)-vec3f(1e-4),vec3f(0.));
  let grid=clamp(point/p.physical.x,vec3f(0.),upper);let rounded=round(grid);
  var seamMask=0u;for(var axis=0u;axis<3u;axis+=1u){
    if(abs(grid[axis]-rounded[axis])<=1e-5){seamMask|=1u<<axis;}}
  if(seamMask==0u){let tag=extendedOwnerTag(vec3u(floor(grid)));return taggedVelocity(tag);}
  var tags:array<u32,8>;var xTerms:array<f32,8>;var yTerms:array<f32,8>;var zTerms:array<f32,8>;var count=0u;
  for(var mask=0u;mask<8u;mask+=1u){
    if((mask&~seamMask)!=0u){continue;}var q=vec3i(floor(grid));var inside=true;
    for(var axis=0u;axis<3u;axis+=1u){if((seamMask&(1u<<axis))!=0u){
      q[axis]=i32(rounded[axis])+select(-1,0,(mask&(1u<<axis))!=0u);}}
    if(any(q<vec3i(0))||any(q>=vec3i(d))){inside=false;}if(!inside){continue;}
    let tag=extendedOwnerTag(vec3u(q));if(tag==INVALID){continue;}
    var duplicate=false;for(var prior=0u;prior<count;prior+=1u){if(tags[prior]==tag){duplicate=true;}}
    if(duplicate){continue;}let sample=taggedVelocity(tag);if(!vectorValid(sample)){continue;}
    tags[count]=tag;xTerms[count]=sample.x;yTerms[count]=sample.y;zTerms[count]=sample.z;count+=1u;
  }
  if(count==0u){return invalidVector();}if(count==1u){return taggedVelocity(tags[0]);}
  let inverseCount=1./f32(count);let result=inverseCount*vec3f(canonicalInterpolation8(xTerms,count),
    canonicalInterpolation8(yTerms,count),canonicalInterpolation8(zTerms,count));
  return select(invalidVector(),vec4f(result,1.),finite3(result));
}
fn axisNeighbor(row:u32,axis:u32,positive:bool)->u32{
  if(row>=acc(2u)||axis>=3u){return INVALID;}
  let direction=2u*axis+select(0u,1u,positive);
  let other=a[abase()+p.rowAxisOffset+6u*row+direction];
  if(other>=acc(2u)){return INVALID;}
  return other;
}
fn regularTag(row:u32,offset:vec3i)->u32{
  if(row>=acc(2u)||any(offset<vec3i(-1))||any(offset>vec3i(1))){return INVALID;}
  let local=u32(offset.x+1)+3u*u32(offset.y+1)+9u*u32(offset.z+1);
  return supportWord(p.regularTagOffsetWords+27u*row+local);
}
// All catalog geometry is authored on an exact canonical lattice, while its
// resolved world coordinates acquire different final ulps on opposite sides
// of the domain (center + offset versus center - offset). Snap interpolation
// coordinates, not stored state, to a dyadic subcell lattice before choosing
// weights. Reflections and x/z rotations then reach bit-identical canonical
// coordinates without materially changing the Section 5 characteristic.
fn snapInterpolationCoordinate(value:f32)->f32{
  return round(value*65536.)/65536.;
}
fn snapInterpolationCoordinates(value:vec3f)->vec3f{
  return vec3f(snapInterpolationCoordinate(value.x),
    snapInterpolationCoordinate(value.y),snapInterpolationCoordinate(value.z));
}
// Convert through the shared domain centre before snapping.  Unlike
// point/h, this coordinate is exactly odd under horizontal reflection and is
// merely permuted by x/z exchange.  The dyadic snap is an intentional
// arithmetic barrier: it prevents backend fast-math from reassociating the
// centred expression back into the asymmetric absolute-world construction.
fn centeredGridPoint(point:vec3f)->vec3f{
  return snapInterpolationCoordinates((point-domainWorldCenter())/p.physical.x);
}
// Construct a semi-Lagrangian characteristic in the same row-local canonical
// frame as Section 5's cube/tetrahedron interpolant. Absolute world-space
// subtraction rounds opposite sides of the domain differently, and a global
// frame still leaves reflected transition cases with different catalog
// transforms. In canonical cell coordinates a reflection is only a sign flip
// and x/z rotation is only a permutation before the shared dyadic snap.
fn characteristicPoint(row:u32,x:vec3f,dt:f32,velocity:vec3f)->vec3f{
  if(row>=acc(2u)){return x-dt*velocity;}
  let rg=rowGeometry[rbase()+row];
  let rowSize=f32(rg.y);let extent=rowSize*p.physical.x;
  let transform=metrics[row].transformAndFlags&63u;
  let local=snapInterpolationCoordinates(powerTransform(
    (centeredGridPoint(x)-rowCenteredGridOffset(rg))/rowSize,transform));
  let displacement=snapInterpolationCoordinates(dt*powerTransform(velocity,transform)/extent);
  let traced=snapInterpolationCoordinates(local-displacement);
  let centeredGrid=snapInterpolationCoordinates(rowCenteredGridOffset(rg)
    +rowSize*inverseTransform(traced,transform));
  return domainWorldCenter()+centeredGrid*p.physical.x;
}
// A shared face's stored owner is always the low-coordinate row. Reflection
// exchanges that arbitrary storage side, but it does not exchange the
// physical side into which a backward characteristic departs. Select the
// incident row on the upstream side of the face (opposite the velocity's
// normal direction) before constructing the row-local snapped point. This is
// also the local dual element that contains the characteristic for an
// infinitesimal step, matching Section 5's containing-element interpolation.
fn characteristicIncidentRow(handle:u32,velocity:vec3f)->u32{
  let low=owner(handle);let high=neighbor(handle);
  if(high==INVALID||high==low){return low;}
  let normalVelocity=canonicalVelocityDot(velocity,normal(handle));
  return select(high,low,normalVelocity>=0.);
}
// Aanjaneya et al. 2017, Section 5: "For improved efficiency and accuracy in
// regular regions away from level transitions, we revert to standard per-axis
// face-based velocity interpolation." The cell-vector trilinear basis below
// (cellSample) averages a row's same-axis faces before it blends, so sampling
// at a face centre returned a [1,2,1]/4 filter of the neighbouring face
// values instead of the face's own degree of freedom, low-passing the whole
// velocity field on every substep even as CFL approaches zero. The staggered
// basis reproduces each face's own value exactly (weight one) at its centre.
fn regularFaceHandle(row:u32,axis:u32,positive:bool)->u32{
  if(row>=acc(2u)||axis>=3u){return INVALID;}
  var world=vec3f(0.);world[axis]=select(-1.,1.,positive);
  let canonical=powerTransform(world,metrics[row].transformAndFlags&63u);
  let magnitude=abs(canonical);
  let family=select(select(2u,1u,magnitude.y>magnitude.z),0u,magnitude.x>max(magnitude.y,magnitude.z));
  let orientation=select(0u,1u,canonical[family]>0.);
  // Publisher-resolved O(1) family-slot base: classify wrote
  // rowFamilyHandles[6*row+family] and the Section 6.3 publication filled
  // rowFamilySlots[base+orientation] for every incident side of the face.
  let slotBase=a[abase()+p.rowFamilyHandleOffset+6u*row+family];
  if(slotBase+orientation>=48u*p.rowCapacity){return INVALID;}
  return a[abase()+p.rowFamilySlotOffset+slotBase+orientation];
}
fn faceAxisValue(handle:u32,axis:u32)->vec2f{
  if(handle==INVALID||handle>=acc(5u)){return vec2f(0.,0.);}
  let n=normal(handle);
  let magnitude=abs(n);
  if(magnitude[axis]<=.9999||dot(magnitude,vec3f(1.))-magnitude[axis]>1e-4){return vec2f(0.,0.);}
  let sample=value(handle);
  if(!finite(sample)){return vec2f(0.,0.);}
  // The stored degree of freedom is u dot n for the face's own world normal;
  // recover the signed world-axis component exactly, never an average.
  return vec2f(select(-sample,sample,n[axis]>0.),1.);
}
// One axis-normal face plane of the staggered stencil. Section 5 permits this
// optimization only in regular regions. Every incident cell must therefore be
// a live, wet, same-size regular row; a support/dry cell is the extrapolated
// free-surface band, not a regular liquid region. Reject that entire staggered
// sample so the caller uses the paper's cube/tetra interpolant over the
// already-published extended vectors. Mixing exact wet face DOFs with copied
// cell vectors made the result depend on which side of a reflected free face
// supplied the plane.
fn staggeredPlaneValue(anchor:u32,rg:vec4u,origin:vec3f,h:f32,axis:u32,plane:i32,transverse:vec3i)->vec2f{
  var faceTerms=array<f32,4>(0.,0.,0.,0.);var faceCount=0u;
  let d=dimensions();
  for(var candidate=0u;candidate<2u;candidate+=1u){
    let cellAt=plane-i32(candidate);
    if(cellAt<-1||cellAt>1){continue;}
    let centerAt=origin[axis]+(f32(cellAt)+.5)*h;
    if(centerAt<0.||centerAt>f32(d[axis])*p.physical.x){continue;}
    var offset=transverse;offset[axis]=cellAt;
    let tag=regularTag(anchor,offset);
    if(tag==INVALID){return vec2f(0.,0.);}
    if((tag&SUPPORT_TAG)!=0u){return vec2f(0.,0.);}
    // A same-size live neighbour qualifies regardless of caseId — its face
    // handle is validated axis-normal and finite below, which is the real
    // requirement. A genuinely irregular neighbour fails that check instead.
    if(tag>=acc(2u)||rowGeometry[rbase()+tag].y!=rg.y){return vec2f(0.,0.);}
    if(liquidAt(tag)==0u){return vec2f(0.,0.);}
    let resolved=faceAxisValue(regularFaceHandle(tag,axis,candidate==1u),axis);
    if(resolved.y==0.){return vec2f(0.,0.);}
    faceTerms[faceCount]=resolved.x;faceCount+=1u;
  }
  if(faceCount>0u){return vec2f(canonicalInterpolation4(faceTerms)/f32(faceCount),1.);}
  return vec2f(0.,0.);
}
fn staggeredComponent(anchor:u32,rg:vec4u,origin:vec3f,h:f32,x:vec3f,axis:u32)->vec2f{
  let d=dimensions();
  var sample=x;
  // The axis-normal face lattice reaches the domain walls exactly (wall
  // faces are published boundary faces of interior rows); transverse axes
  // keep the cell-centred .5*h constant physical-boundary extension.
  sample[axis]=clamp(sample[axis],0.,f32(d[axis])*p.physical.x);
  for(var other=0u;other<3u;other+=1u){
    if(other==axis){continue;}
    sample[other]=clamp(sample[other],.5*h,f32(d[other])*p.physical.x-.5*h);
  }
  let center=origin+vec3f(.5*h);
  // Form all weights in the anchor-centred frame. Subtracting the absolute
  // origin independently on opposite sides of the domain leaves different
  // final ulps before the dyadic snap; the centred coordinate is the actual
  // Section 5 interpolation coordinate and transforms exactly under D4.
  let local=snapInterpolationCoordinates((sample-center)/h);
  let along=local[axis]+.5;
  let plane=clamp(i32(floor(along)),-1,1);
  var tAlong=snapInterpolationCoordinate(clamp(along-f32(plane),0.,1.));
  // Measure-zero snap: a face centre must keep weight one on its own value
  // under floating-point dust, mirroring the old-mesh epsilon discipline at
  // interpolation-element boundaries.
  if(tAlong<1e-5){tAlong=0.;}else if(tAlong>1.-1e-5){tAlong=1.;}
  var low=vec3i(0);
  var tTransverse=vec3f(0.);
  for(var other=0u;other<3u;other+=1u){
    if(other==axis){continue;}
    if(local[other]<0.){low[other]=-1;}
    var t=clamp(local[other]-f32(low[other]),0.,1.);
    if(t<1e-5){t=0.;}else if(t>1.-1e-5){t=1.;}
    tTransverse[other]=t;
  }
  var terms:array<f32,8>;var termCount=0u;
  for(var corner=0u;corner<8u;corner+=1u){
    let alongWeight=select(1.-tAlong,tAlong,(corner&1u)!=0u);
    var transverseWeights=vec2f(1.);var transverseCount=0u;
    var offset=vec3i(0);
    var bit=1u;
    for(var other=0u;other<3u;other+=1u){
      if(other==axis){continue;}
      let high=(corner&(1u<<bit))!=0u;
      transverseWeights[transverseCount]=select(1.-tTransverse[other],tTransverse[other],high);
      transverseCount+=1u;
      offset[other]=low[other]+select(0,1,high);
      bit+=1u;
    }
    let weight=canonicalProduct3(alongWeight,transverseWeights.x,transverseWeights.y);
    if(weight<=0.){continue;}
    for(var other=0u;other<3u;other+=1u){
      if(other==axis){continue;}
      let requested=center[other]+f32(offset[other])*h;
      if(requested<.5*h||requested>f32(d[other])*p.physical.x-.5*h){offset[other]=0;}
    }
    let resolved=staggeredPlaneValue(anchor,rg,origin,h,axis,plane+i32(corner&1u),offset);
    if(resolved.y==0.){return vec2f(0.,0.);}
    terms[termCount]=weight*resolved.x;termCount+=1u;
  }
  let result=canonicalInterpolation8(terms,termCount);
  if(!finite(result)){return vec2f(0.,0.);}
  return vec2f(result,1.);
}
fn staggeredSample(anchor:u32,x:vec3f)->vec4f{
  // Eligibility is the producer's published regular closure (centre cube tag
  // = the row id), not caseId==0: a wall-touching row keeps axis-normal cube
  // faces and must stay on the exact per-axis face basis.
  if(anchor>=acc(2u)||regularTag(anchor,vec3i(0))!=anchor||!finite3(x)){return invalidVector();}
  let rg=rowGeometry[rbase()+anchor];
  let h=f32(rg.y)*p.physical.x;
  if(!finite(h)||h<=0.){return invalidVector();}
  let d=dimensions();
  let q=vec3u(rg.x%d.x,(rg.x/d.x)%d.y,rg.x/(d.x*d.y));
  let origin=vec3f(q)*p.physical.x;
  var result=vec3f(0.);
  for(var axis=0u;axis<3u;axis+=1u){
    let component=staggeredComponent(anchor,rg,origin,h,x,axis);
    if(component.y==0.){return invalidVector();}
    result[axis]=component.x;
  }
  if(!finite3(result)){return invalidVector();}
  return vec4f(result,1.);
}
fn cellSample(anchor:u32,x:vec3f)->vec4f{
  if(anchor>=acc(2u)||!finite3(x)){return invalidVector();}
  let rg=rowGeometry[rbase()+anchor];
  let d=dimensions();
  let h=f32(rg.y)*p.physical.x;
  if(!finite(h)||h<=0.){return invalidVector();}
  // Cell-centred velocity has a constant physical-boundary extension. Keep
  // both the cube weights and its eight requested centres in centred-grid
  // units. This is the second symmetry-sensitive round trip after
  // characteristicPoint: returning the corrected midpoint to absolute world
  // space and independently subtracting a large positive cube centre changed
  // the chosen/interpolated support vector on reflected free-surface rows.
  // A missing live interior row still rejects, while the basis never requests
  // an exterior row.
  let rowSize=f32(rg.y);let halfDomain=.5*vec3f(d);
  let sampleGrid=clamp(centeredGridPoint(x),-halfDomain+vec3f(.5*rowSize),
    halfDomain-vec3f(.5*rowSize));
  let centerGrid=rowCenteredGridOffset(rg);
  var lowOffset=vec3i(0);
  for(var axis=0u;axis<3u;axis+=1u){
    if(sampleGrid[axis]<centerGrid[axis]){lowOffset[axis]=-1;}
  }
  let lowCenterGrid=centerGrid+vec3f(lowOffset)*rowSize;
  let t=snapInterpolationCoordinates(clamp((sampleGrid-lowCenterGrid)/rowSize,
    vec3f(0),vec3f(1)));
  var termsX:array<f32,8>;var termsY:array<f32,8>;var termsZ:array<f32,8>;
  var termCount=0u;
  for(var corner=0u;corner<8u;corner+=1u){
    let weight=canonicalProduct3(select(1.-t.x,t.x,(corner&1u)!=0u),
      select(1.-t.y,t.y,(corner&2u)!=0u),select(1.-t.z,t.z,(corner&4u)!=0u));
    if(weight<=0.){continue;}
    var offset=lowOffset+vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    var requestedGrid=centerGrid+vec3f(offset)*rowSize;
    for(var axis=0u;axis<3u;axis+=1u){
      if(requestedGrid[axis]<-halfDomain[axis]+.5*rowSize
        ||requestedGrid[axis]>halfDomain[axis]-.5*rowSize){offset[axis]=0;}
    }
    requestedGrid=centerGrid+vec3f(offset)*rowSize;
    let requestedCenter=domainWorldCenter()+requestedGrid*p.physical.x;
    // Consume the same dense, geometric Section 5 owner publication as
    // arbitrary-point extension. The row-relative regular-tag cache is built
    // through floating world centres and can name opposite half-open owners
    // after reflection at the advancing domain boundary.
    let sample=extendedOwnerVelocity(requestedCenter);
    if(!vectorValid(sample)){return invalidVector();}
    let term=weight*sample.xyz;
    termsX[termCount]=term.x;termsY[termCount]=term.y;termsZ[termCount]=term.z;
    termCount+=1u;
  }
  let result=vec3f(canonicalInterpolation8(termsX,termCount),
    canonicalInterpolation8(termsY,termCount),
    canonicalInterpolation8(termsZ,termCount));
  if(!finite3(result)){return invalidVector();}
  return vec4f(result,1.);
}
fn regularSample(anchor:u32,x:vec3f)->vec4f{
  // Per-axis face-based interpolation first; the cell-vector cube basis
  // remains the mandated fallback wherever the staggered stencil touches a
  // level transition or an unextended region, and its reject paths remain
  // the fail-closed terminus when both bases are unavailable.
  let staggered=staggeredSample(anchor,x);
  if(vectorValid(staggered)){return staggered;}
  return cellSample(anchor,x);
}
fn selectorVelocity(row:u32,selectorIndex:u32,selector:vec4f)->vec4f{
  if(row>=acc(2u)||selectorIndex>=p.tetraVertexCount||selectorIndex>=p.selectorStride
    ||!finite3(selector.xyz)||!finite(selector.w)){return vec4f(f32(selectorIndex),10.,f32(row),-1.);}
  if(length(selector.xyz)<1e-7){return velocitySample(row);}
  let selectorAt=p.selectorOffsetWords+row*p.selectorStride+selectorIndex;
  if(selectorAt>=4u*arrayLength(&transportMetrics)){return vec4f(f32(selectorIndex),1.,0.,-1.);}
  let other=supportWord(selectorAt);
  if(other!=INVALID){
    let sample=taggedVelocity(other);
    if(!vectorValid(sample)){return vec4f(f32(selectorIndex),2.,f32(other),-1.);}
    return sample;
  }
  // The paper's local Delaunay mesh includes face and edge neighbors, so a
  // missing selector inside the domain is incomplete live topology. Only a
  // catalog vertex proven exterior receives the physical-boundary extension.
  let rg=rowGeometry[rbase()+row];
  let d=dimensions();let q=vec3u(rg.x%d.x,(rg.x/d.x)%d.y,rg.x/(d.x*d.y));
  let extent=f32(rg.y)*p.physical.x;
  let neighborExtent=selector.w*extent;
  if(!finite(extent)||extent<=0.||!finite(neighborExtent)||neighborExtent<=0.){return invalidVector();}
  let center=(vec3f(q)+.5*f32(rg.y))*p.physical.x;
  let selectorCenter=center+inverseTransform(selector.xyz,metrics[row].transformAndFlags&63u)*extent;
  let lower=vec3f(.5*neighborExtent);
  let upper=vec3f(d)*p.physical.x-lower;
  let tolerance=max(1e-5,neighborExtent*2e-5);
  if(all(selectorCenter>=lower-vec3f(tolerance))&&all(selectorCenter<=upper+vec3f(tolerance))){
    return vec4f(f32(selectorIndex),3.,f32(other),-1.);
  }
  return velocitySample(row);
}
fn canonicalProduct3(a:f32,b:f32,c:f32)->f32{
  var factors=array<f32,3>(a,b,c);
  for(var i=1u;i<3u;i+=1u){let value=factors[i];var j=i;
    loop{if(j==0u||abs(factors[j-1u])<=abs(value)){break;}factors[j]=factors[j-1u];j-=1u;}
    factors[j]=value;}
  return factors[0]*factors[1]*factors[2];
}
fn canonicalDeterminant(a:vec3f,b:vec3f,c:vec3f)->f32{
  var terms:array<f32,8>;
  terms[0]=canonicalProduct3(a.x,b.y,c.z);
  terms[1]=canonicalProduct3(a.y,b.z,c.x);
  terms[2]=canonicalProduct3(a.z,b.x,c.y);
  terms[3]=-canonicalProduct3(a.z,b.y,c.x);
  terms[4]=-canonicalProduct3(a.y,b.x,c.z);
  terms[5]=-canonicalProduct3(a.x,b.z,c.y);
  return canonicalInterpolation8(terms,6u);
}
fn tetraWeights(point:vec3f,x:vec3f,y:vec3f,z:vec3f)->vec4f{
  let d=canonicalDeterminant(x,y,z);
  if(!finite(d)||abs(d)<1e-10){return vec4f(-2.);}
  let a0=canonicalDeterminant(point,y,z)/d;
  let a1=canonicalDeterminant(x,point,z)/d;
  let a2=canonicalDeterminant(x,y,point)/d;
  var complementTerms=array<f32,4>(1.,-a0,-a1,-a2);
  return vec4f(canonicalInterpolation4(complementTerms),a0,a1,a2);
}
fn tetraSelectorGeometry(selector:u32)->vec4f{let at=p.tetraVertexOffset+4u*selector;
  return vec4f(bitsf(at),bitsf(at+1u),bitsf(at+2u),bitsf(at+3u));}
fn transformedD4Selector(selector:u32,transform:u32)->u32{if(selector>=p.tetraVertexCount){return INVALID;}
  let source=tetraSelectorGeometry(selector);let wanted=powerTransform(source.xyz,transform);
  for(var candidate=0u;candidate<p.tetraVertexCount;candidate+=1u){let value=tetraSelectorGeometry(candidate);
    if(all(value.xyz==wanted)&&value.w==source.w){return candidate;}}
  return INVALID;}
fn transitionFanSample(row:u32,localPoint:vec3f,first:u32,count:u32,transform:u32)->vec4f{
  for(var ti=0u;ti<count;ti+=1u){let packed=catalog[p.tetraOffset+first+ti];
    var selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);
    if(transform!=0u){selectors=vec3u(transformedD4Selector(selectors.x,transform),
      transformedD4Selector(selectors.y,transform),transformedD4Selector(selectors.z,transform));}
    if(any(selectors>=vec3u(p.tetraVertexCount))){return vec4f(f32(ti),22.,f32(p.tetraVertexCount),-1.);}
    let sa=tetraSelectorGeometry(selectors.x);let sb=tetraSelectorGeometry(selectors.y);let sc=tetraSelectorGeometry(selectors.z);
    let weights=tetraWeights(localPoint,sa.xyz,sb.xyz,sc.xyz);
    if(all(weights>=vec4f(-2e-6))&&all(weights<=vec4f(1.000002))){var positive=max(weights,vec4f(0.));
      let positiveSum=canonicalInterpolation4(array<f32,4>(positive.x,positive.y,positive.z,positive.w));
      if(!finite(positiveSum)||positiveSum<=0.){return vec4f(f32(ti),25.,positiveSum,-1.);}positive/=positiveSum;
      var termsX=array<f32,4>(0.,0.,0.,0.);var termsY=array<f32,4>(0.,0.,0.,0.);
      var termsZ=array<f32,4>(0.,0.,0.,0.);
      if(positive.x>0.){let v0=velocitySample(row);if(!vectorValid(v0)){return vec4f(f32(ti),23.,f32(row),-1.);}let term=positive.x*v0.xyz;termsX[0]=term.x;termsY[0]=term.y;termsZ[0]=term.z;}
      if(positive.y>0.){let v1=selectorVelocity(row,selectors.x,sa);if(!vectorValid(v1)){return v1;}let term=positive.y*v1.xyz;termsX[1]=term.x;termsY[1]=term.y;termsZ[1]=term.z;}
      if(positive.z>0.){let v2=selectorVelocity(row,selectors.y,sb);if(!vectorValid(v2)){return v2;}let term=positive.z*v2.xyz;termsX[2]=term.x;termsY[2]=term.y;termsZ[2]=term.z;}
      if(positive.w>0.){let v3=selectorVelocity(row,selectors.z,sc);if(!vectorValid(v3)){return v3;}let term=positive.w*v3.xyz;termsX[3]=term.x;termsY[3]=term.y;termsZ[3]=term.z;}
      let result=vec3f(canonicalInterpolation4(termsX),canonicalInterpolation4(termsY),canonicalInterpolation4(termsZ));
      if(!finite3(result)){return invalidVector();}return vec4f(result,1.);}}
  return vec4f(255.,24.,f32(count),-1.);}
fn transitionSample(row:u32,x:vec3f)->vec4f{
  if(row>=acc(2u)||!finite3(x)){return vec4f(255.,20.,f32(row),-1.);}let rg=rowGeometry[rbase()+row];
  let extent=f32(rg.y)*p.physical.x;
  if(!finite(extent)||extent<=0.){return vec4f(255.,21.,extent,-1.);}
  let local=snapInterpolationCoordinates(powerTransform(
    (centeredGridPoint(x)-rowCenteredGridOffset(rg))/f32(rg.y),
    metrics[row].transformAndFlags&63u));
  let thAt=p.tetraHeaderOffset+3u*metrics[row].caseId;let first=catalog[thAt];let count=catalog[thAt+1u];
  // Fan closure depends only on immutable catalog topology. The generator
  // publishes its exact D4 mask in the header; reproving it here nested a
  // selector-catalog search inside every characteristic sample.
  // The row transform can permute world vertical onto any canonical axis.
  // Conjugating horizontal D4 by that transform therefore selects the full
  // stabilizer of canonical x, y, or z. The catalog packs the exact fan
  // closure mask for all three groups; selecting the conjugate group is what
  // makes a physical reflection independent of case canonicalization.
  let rowTransform=metrics[row].transformAndFlags&63u;
  let canonicalVertical=abs(powerTransform(vec3f(0.,1.,0.),rowTransform));
  let fixedAxis=select(select(2u,1u,canonicalVertical.y>.5),0u,canonicalVertical.x>.5);
  let symmetryMask=(catalog[thAt+2u]>>(8u+8u*fixedAxis))&255u;
  let d4=array<u32,24>(0u,2u,4u,6u,8u,10u,12u,14u,
    0u,1u,4u,5u,40u,41u,44u,45u,
    0u,1u,2u,3u,16u,17u,18u,19u);var xTerms:array<f32,8>;
  var yTerms:array<f32,8>;var zTerms:array<f32,8>;var sampleCount=0u;
  for(var symmetry=0u;symmetry<8u;symmetry+=1u){let transform=d4[8u*fixedAxis+symmetry];
    if((symmetryMask&(1u<<symmetry))==0u){continue;}
    let sample=transitionFanSample(row,local,first,count,transform);if(!vectorValid(sample)){return sample;}
    xTerms[sampleCount]=sample.x;yTerms[sampleCount]=sample.y;zTerms[sampleCount]=sample.z;sampleCount+=1u;}
  if(sampleCount==0u){return vec4f(255.,27.,f32(count),-1.);}let inverseCount=1./f32(sampleCount);
  return vec4f(inverseCount*vec3f(canonicalInterpolation8(xTerms,sampleCount),
    canonicalInterpolation8(yTerms,sampleCount),canonicalInterpolation8(zTerms,sampleCount)),1.);
}

// Aanjaneya et al. 2017 Section 5
// (docs/papers/aanjaneya-2017-power-liquids.txt) defines the interpolation
// domain as the dual mesh of cubes and locally Delaunay tetrahedra whose
// vertices are octree cell centres. An octree leaf containing a point is only
// a search seed: near a T-junction, the point's dual element can be incident
// on a neighbouring cell centre instead. Keep both element evaluators exact;
// this helper never turns an incomplete interpolant into a constant field.
fn interpolationElementSample(anchor:u32,point:vec3f)->vec4f{
  if(anchor==INVALID||anchor>=acc(2u)){return invalidVector();}
  if(regularTag(anchor,vec3i(0))!=anchor){return transitionSample(anchor,point);}
  return regularSample(anchor,point);
}
// Search the published Delaunay adjacency, not arbitrary spatial rows. A
// tetrahedron containing point is accepted only by transitionSample's
// barycentric bounds; a regular neighbour stays on the paper's cube basis.
// This path runs only after the seed star rejects, so the ordinary hot path is
// unchanged while a characteristic crossing a T-junction can enter the exact
// adjacent dual element.
fn adjacentInterpolationElementSample(anchor:u32,point:vec3f)->vec4f{
  if(anchor==INVALID||anchor>=acc(2u)||p.selectorStride==0u){return invalidVector();}
  for(var selectorIndex=0u;selectorIndex<p.selectorStride;selectorIndex+=1u){
    let selectorAt=p.selectorOffsetWords+anchor*p.selectorStride+selectorIndex;
    if(selectorAt>=4u*arrayLength(&transportMetrics)){return invalidVector();}
    let tag=supportWord(selectorAt);
    if(tag==INVALID||(tag&SUPPORT_TAG)!=0u||tag==anchor||tag>=acc(2u)){continue;}
    let sample=interpolationElementSample(tag,point);
    if(vectorValid(sample)){return sample;}
  }
  return invalidVector();
}
fn incidentInterpolationElementSample(anchor:u32,point:vec3f)->vec4f{
  var sample=interpolationElementSample(anchor,point);
  if(vectorValid(sample)){return sample;}
  return adjacentInterpolationElementSample(anchor,point);
}

// Resolve the accepted leaf containing a physical point. The row directory is
// ordered by (level, Morton), so split children can find their old parent and
// merged parents can find the appropriate old child without a row identity.
fn mortonPart10(v:u32)->u32{var q=v&1023u;q=(q|(q<<16u))&0x030000ffu;q=(q|(q<<8u))&0x0300f00fu;q=(q|(q<<4u))&0x030c30c3u;q=(q|(q<<2u))&0x09249249u;return q;}
fn mortonCell(cell:u32)->u32{let d=dimensions();let q=vec3u(cell%d.x,(cell/d.x)%d.y,cell/(d.x*d.y));return mortonPart10(q.x)|(mortonPart10(q.y)<<1u)|(mortonPart10(q.z)<<2u);}
fn directoryLevel(size:u32)->u32{return 31u-countLeadingZeros(size);}
fn directoryLess(entry:vec4u,level:u32,morton:u32)->bool{let oldLevel=directoryLevel(entry.y);let oldMorton=mortonCell(entry.x);return oldLevel<level||(oldLevel==level&&oldMorton<morton);}
fn acceptedDirectoryFind(cell:u32,size:u32)->u32{
  let wantedLevel=directoryLevel(size);let wantedMorton=mortonCell(cell);var low=0u;var high=min(acc(2u),p.rowCapacity);
  while(low<high){let middle=low+(high-low)/2u;let entry=rowGeometry[rbase()+middle];if(directoryLess(entry,wantedLevel,wantedMorton)){low=middle+1u;}else{high=middle;}}
  if(low<acc(2u)){let entry=rowGeometry[rbase()+low];if(entry.x==cell&&entry.y==size){return low;}}
  return INVALID;
}
fn acceptedRowContainingFinestCell(finest:vec3u)->u32{
  let d=dimensions();if(any(finest>=d)){return INVALID;}let maximum=max(d.x,max(d.y,d.z));var size=1u;
  for(var level=0u;level<31u;level+=1u){if(size>maximum){break;}let origin=(finest/vec3u(size))*size;
    let cell=origin.x+d.x*(origin.y+d.y*origin.z);let row=acceptedDirectoryFind(cell,size);if(row!=INVALID){return row;}size<<=1u;}
  return INVALID;
}
// Single-entry memoization of the leaf walk. This is a PURE memoization: it
// must return the identical row id, never a cheaper predicate.
//
// The walk is a function of the finest cell containing the clamped point and
// of nothing else -- every level keys on floor(grid/size)*size, which is
// constant across a finest cell for every size, so two points in the same
// finest cell issue the identical (cell,size) probe sequence and reach the
// identical answer. Caching the FOUND row's own box therefore also answers
// every point inside it: a hit at (origin,size) can only differ from a fresh
// walk if some accepted row lay strictly inside that one, and accepted rows are
// octree leaves, hence disjoint -- this function's contract is "the accepted
// leaf containing a physical point". A miss caches only the single finest cell
// it was proven for.
//
// Nothing can invalidate the entry mid-invocation: rowGeometry is a read-only
// binding and the accepted row count that bounds the directory is never
// written here.
//
// The caller that pays for this is rowTouchesDry -- advect runs it for both
// incident rows, up to twelve walks per face, purely to decide the carry gate
// -- followed by characteristicSample's midpoint/departure pair and oldAnchor's
// bounded 5^3 neighbourhood sweep. The carry, carriedMarker and wall-row
// eligibility SEMANTICS are untouched; only the cost of reaching the same
// answer changes.
var<private> containedResolved:bool;
var<private> containedOrigin:vec3u;
var<private> containedSize:u32;
var<private> containedRow:u32;
fn acceptedRowContaining(point:vec3f)->u32{
  if(!finite3(point)){return INVALID;}let d=dimensions();let upper=max(vec3f(d)-vec3f(1e-4),vec3f(0.));let grid=clamp(point/p.physical.x,vec3f(0.),upper);
  let finest=vec3u(floor(grid));
  if(containedResolved&&all(finest>=containedOrigin)&&all(finest<containedOrigin+vec3u(containedSize))){return containedRow;}
  let maximum=max(d.x,max(d.y,d.z));var size=1u;
  for(var level=0u;level<31u;level+=1u){if(size>maximum){break;}let origin=vec3u(floor(grid/f32(size)))*size;let cell=origin.x+d.x*(origin.y+d.y*origin.z);let row=acceptedDirectoryFind(cell,size);if(row!=INVALID){containedResolved=true;containedOrigin=origin;containedSize=size;containedRow=row;return row;}size<<=1u;}
  containedResolved=true;containedOrigin=finest;containedSize=1u;containedRow=INVALID;
  return INVALID;
}
fn candidateRowCenter(candidateBank:u32,row:u32)->vec3f{
  if(row>=p.rowCapacity){return vec3f(3.402823e38);}
  let rg=rowGeometry[candidateBank*p.rowCapacity+row];let d=dimensions();let q=vec3u(rg.x%d.x,(rg.x/d.x)%d.y,rg.x/(d.x*d.y));
  return (vec3f(q)+.5*f32(rg.y))*p.physical.x;
}
// A candidate centre on an old leaf/free-surface seam has no unique half-open
// owner. It is interior only when EVERY finest cell incident on that exact
// half-grid point names the same accepted leaf; otherwise Section 5's
// extrapolated frontier, not an arbitrary low/high old-row trace, is the old
// field authority. Integer doubled coordinates keep the predicate exactly D4.
fn candidateRowCenterInsideOld(candidateBank:u32,row:u32)->bool{
  if(row>=p.rowCapacity){return false;}let rg=rowGeometry[candidateBank*p.rowCapacity+row];let d=dimensions();
  let origin=vec3u(rg.x%d.x,(rg.x/d.x)%d.y,rg.x/(d.x*d.y));let doubled=2u*origin+vec3u(rg.y);
  var seamMask=0u;for(var axis=0u;axis<3u;axis+=1u){if((doubled[axis]&1u)==0u){seamMask|=1u<<axis;}}
  var resolved=INVALID;for(var mask=0u;mask<8u;mask+=1u){if((mask&~seamMask)!=0u){continue;}var q=vec3i(doubled/2u);
    for(var axis=0u;axis<3u;axis+=1u){if((seamMask&(1u<<axis))!=0u&&(mask&(1u<<axis))==0u){q[axis]-=1;}}
    if(any(q<vec3i(0))||any(q>=vec3i(d))){continue;}let incident=acceptedRowContainingFinestCell(vec3u(q));
    if(incident==INVALID||(resolved!=INVALID&&incident!=resolved)){return false;}resolved=incident;}
  return resolved!=INVALID;
}
fn oldFieldAt(anchor:u32,point:vec3f)->vec4f{
  if(anchor==INVALID){return invalidVector();}var sample=invalidVector();if(regularTag(anchor,vec3i(0))!=anchor){sample=transitionSample(anchor,point);}else{sample=regularSample(anchor,point);}
  // Interpolation can leave its local closure at a newly exposed face. The
  // already projected row vector is the bounded piecewise-constant fallback;
  // zero is never presented as a successful topology transfer.
  if(!vectorValid(sample)){sample=velocitySample(anchor);}return sample;
}
// One changed face owns one 128-lane workgroup. The first five lanes evaluate
// the scalar search's ordered centre/backtrace probes; all 125 neighbourhood
// offsets are then evaluated in parallel only when those probes miss. Integer
// pair reduction preserves the scalar choice exactly: first probe rank for the
// backtrace, then (squared distance, accepted row) for the 5^3 search.
var<workgroup> transferProbeRows:array<u32,128>;
var<workgroup> transferProbeRanks:array<u32,128>;
var<workgroup> transferOld:vec4f;
var<workgroup> transferDirect:u32;
var<workgroup> transferNeighbor:u32;
var<workgroup> transferHandle:u32;
var<workgroup> transferDisposition:u32;
var<workgroup> transferCandidateBank:u32;
var<workgroup> transferOwner:u32;
var<workgroup> transferInterior:vec2u;
var<workgroup> transferPoint:vec4f;
var<workgroup> transferNormal:vec4f;
fn reduceTransferProbe(lid:u32)->u32{
  workgroupBarrier();
  for(var width=64u;width>0u;width>>=1u){
    if(lid<width){let other=lid+width;let rank=transferProbeRanks[lid];let otherRank=transferProbeRanks[other];let row=transferProbeRows[lid];let otherRow=transferProbeRows[other];if(otherRank<rank||(otherRank==rank&&otherRow<row)){transferProbeRanks[lid]=otherRank;transferProbeRows[lid]=otherRow;}}
    workgroupBarrier();
  }
  return transferProbeRows[0];
}
fn oldAnchor(candidateBank:u32,candidateRow:u32,n:vec3f,lid:u32)->u32{
  let center=candidateRowCenter(candidateBank,candidateRow);
  var row=INVALID;var rank=INVALID;
  if(lid<5u){let probe=select(center,center-f32(lid)*.5*p.physical.x*n,lid>0u);row=acceptedRowContaining(probe);rank=select(INVALID,lid,row!=INVALID);}
  transferProbeRows[lid]=row;transferProbeRanks[lid]=rank;
  let reducedDirect=reduceTransferProbe(lid);if(lid==0u){transferDirect=reducedDirect;}let direct=workgroupUniformLoad(&transferDirect);if(direct!=INVALID){return direct;}
  row=INVALID;rank=INVALID;
  if(lid<125u){let z=lid/25u;let rem=lid-z*25u;let y=rem/5u;let x=rem-y*5u;let offset=vec3i(i32(x)-2i,i32(y)-2i,i32(z)-2i);if(any(offset!=vec3i(0))){row=acceptedRowContaining(center+vec3f(offset)*p.physical.x);if(row!=INVALID){rank=u32(offset.x*offset.x+offset.y*offset.y+offset.z*offset.z);}}}
  transferProbeRows[lid]=row;transferProbeRanks[lid]=rank;
  return reduceTransferProbe(lid);
}
fn candidateClosedWorld(point:vec3f,n:vec3f)->bool{
  let grid=point/p.physical.x;let d=dimensions();
  for(var axis=0u;axis<3u;axis+=1u){
    if(n[axis]<-.5&&grid[axis]<=1e-4&&(p.closedMask&(1u<<(2u*axis)))!=0u){return true;}
    if(n[axis]>.5&&grid[axis]>=f32(d[axis])-1e-4&&(p.closedMask&(1u<<(2u*axis+1u)))!=0u){return true;}
  }
  return false;
}
fn rejectCandidateTransfer(handle:u32)->bool{atomicStore(&candidate[0],ERROR_SAMPLE);return handle<atomicMin(&candidate[1],handle);}
// Candidate class 4 is the compact set of faces whose exact identity could not
// be carried. Unlike ordinary 64-lane class work, its indirect record publishes
// one workgroup per item so the neighbourhood search is wide rather than one
// dependent page walk per changed face.
@compute @workgroup_size(128)fn transferStructuredTopologyCandidate(@builtin(workgroup_id)g:vec3u,@builtin(local_invocation_index)lid:u32){
  if(lid==0u){
    let index=g.x+g.y*65535u;let handle=candidateTransferItem(index);transferHandle=handle;transferDisposition=3u;
    transferInterior=vec2u(0u);
    if(handle!=INVALID&&atomicLoad(&candidate[6])!=0u&&handle<atomicLoad(&candidate[3])){
      let candidateBank=atomicLoad(&candidate[5])&1u;let base=candidateBank*p.authorityWords;transferCandidateBank=candidateBank;
      let marker=bitcast<f32>(a[base+p.centroidOffset+4u*handle+3u]);
      if(marker<=0.5){let at=base+p.centroidOffset+4u*handle;let point=vec3f(bitcast<f32>(a[at]),bitcast<f32>(a[at+1u]),bitcast<f32>(a[at+2u]));let nt=base+p.normalOffset+4u*handle;let n=vec3f(bitcast<f32>(a[nt]),bitcast<f32>(a[nt+1u]),bitcast<f32>(a[nt+2u]));transferPoint=vec4f(point,0.);transferNormal=vec4f(n,0.);transferOwner=a[base+p.ownerOffset+handle];transferNeighbor=a[base+p.neighborOffset+handle];transferDisposition=select(select(0u,2u,candidateClosedWorld(point,n)),1u,!finite3(point)||!finite3(n));}
    }
  }
  let disposition=workgroupUniformLoad(&transferDisposition);let handle=workgroupUniformLoad(&transferHandle);if(disposition==3u){return;}if(disposition==1u){if(lid==0u){_ = rejectCandidateTransfer(handle);}return;}
  let candidateBank=workgroupUniformLoad(&transferCandidateBank);let base=candidateBank*p.authorityWords;let point=workgroupUniformLoad(&transferPoint).xyz;let n=workgroupUniformLoad(&transferNormal).xyz;
  if(disposition==2u){if(lid==0u){a[base+p.valuesOffset+handle]=bitcast<u32>(0.);}return;}
  let candidateOwner=workgroupUniformLoad(&transferOwner);let candidateNeighbor=workgroupUniformLoad(&transferNeighbor);
  // Paper Section 5 frontier discipline: a face whose owner centre lies
  // beyond the old liquid is newly wetted and takes the extrapolated field —
  // a COPY of the closest projected face value — never an interpolant
  // evaluated outside its element (which extends the frontier velocity
  // gradient one cell and ratchets the corner jet, measured at +45% ME).
  let uniformNeighbor=workgroupUniformLoad(&transferNeighbor);
  if(lid==0u){let ownerInsideOld=candidateRowCenterInsideOld(candidateBank,candidateOwner);
    let neighborInsideOld=uniformNeighbor!=INVALID&&candidateRowCenterInsideOld(candidateBank,uniformNeighbor);
    transferInterior=vec2u(select(0u,1u,ownerInsideOld),select(0u,1u,neighborInsideOld));
    transferOld=invalidVector();if(!ownerInsideOld&&!neighborInsideOld){transferOld=extendedOwnerVelocity(point);}}
  let initialOld=workgroupUniformLoad(&transferOld);
  let interior=workgroupUniformLoad(&transferInterior);
  var ownerAnchor=INVALID;var neighborAnchor=INVALID;
  if(!vectorValid(initialOld)){
    if(interior.x!=0u){ownerAnchor=oldAnchor(candidateBank,candidateOwner,n,lid);}
    if(interior.y!=0u){neighborAnchor=oldAnchor(candidateBank,uniformNeighbor,-n,lid);}
    if(lid==0u){let ownerField=oldFieldAt(ownerAnchor,point);var neighborField=invalidVector();if(uniformNeighbor!=INVALID&&neighborAnchor!=ownerAnchor){neighborField=oldFieldAt(neighborAnchor,point);}
    // Reflection may exchange the arbitrary stored owner with its neighbour.
    // A one-sided old frontier has one geometric authority regardless of that
    // ordering. Only a genuinely two-sided old face has two valid limits; fold
    // that unavoidable ambiguity canonically so the transfer commutes with D4.
    transferOld=ownerField;if(!vectorValid(ownerField)){transferOld=neighborField;}
    if(vectorValid(ownerField)&&vectorValid(neighborField)){
      transferOld=vec4f(.5*vec3f(canonicalInterpolation4(array<f32,4>(ownerField.x,neighborField.x,0.,0.)),
        canonicalInterpolation4(array<f32,4>(ownerField.y,neighborField.y,0.,0.)),
        canonicalInterpolation4(array<f32,4>(ownerField.z,neighborField.z,0.,0.))),1.);}
    if(!vectorValid(transferOld)){transferOld=extendedOwnerVelocity(point);}}
    _ = workgroupUniformLoad(&transferOld);
  }
  if(lid==0u){if(!vectorValid(transferOld)){if(rejectCandidateTransfer(handle)&&arrayLength(&candidate)>=16u){atomicStore(&candidate[9],candidateOwner);atomicStore(&candidate[10],candidateNeighbor);atomicStore(&candidate[11],ownerAnchor);atomicStore(&candidate[12],neighborAnchor);atomicStore(&candidate[13],bitcast<u32>(point.x));atomicStore(&candidate[14],bitcast<u32>(point.y));atomicStore(&candidate[15],bitcast<u32>(point.z));}return;}let projected=canonicalVelocityDot(transferOld.xyz,n);if(!finite(projected)){_ = rejectCandidateTransfer(handle);return;}a[base+p.valuesOffset+handle]=bitcast<u32>(projected);if((p.padB&1u)!=0u&&handle<arrayLength(&transportMetrics)){let debugFlags=select(0u,1u,vectorValid(initialOld))|select(0u,2u,ownerAnchor!=INVALID)|select(0u,4u,neighborAnchor!=INVALID);transportMetrics[handle]=vec4f(transferOld.xyz,bitcast<f32>(debugFlags));}}
}

// Characteristic sources must be this substep's prior face values: the
// staggered sampler reads neighbouring face degrees of freedom, so writing a
// destination into the accepted bank mid-dispatch would race those reads and
// advect some faces through a partially updated field. Every destination
// therefore stages into the inactive authority value bank (rewritten
// wholesale by each future candidate publication) and the fenced commit
// kernels copy the staged words back bit-exactly.
fn nextValueAt(handle:u32)->u32{return (1u-bank())*p.authorityWords+p.valuesOffset+handle;}
fn setNextValue(handle:u32,v:f32){a[nextValueAt(handle)]=bitcast<u32>(v);}
fn debugAdvectionWord(offset:u32,handle:u32,value:f32){
  var mask=0u;
  if(offset==p.ownerOffset||offset==p.neighborOffset){mask=2u;}
  else if(offset==p.metadataOffset){mask=4u;}
  else if(offset==p.areaOffset||offset==p.inverseOffset||offset==p.fractionOffset){mask=8u;}
  if((p.padB&mask)!=0u){a[(1u-bank())*p.authorityWords+offset+handle]=bitcast<u32>(value);}
}
fn debugAdvection3(offset:u32,handle:u32,value:vec3f){
  let at=(1u-bank())*p.authorityWords+offset+4u*handle;
  let mask=select(32u,16u,offset==p.normalOffset);
  if((p.padB&mask)!=0u){a[at]=bitcast<u32>(value.x);a[at+1u]=bitcast<u32>(value.y);a[at+2u]=bitcast<u32>(value.z);}
}
// Local Delaunay meshes are authored per cell (Section 5's "slight
// subtlety"). A characteristic exactly on an octree seam is therefore
// incident to multiple equally valid local stars. Resolving only floor(x)
// makes the interpolant depend on the positive-coordinate side. Evaluate the
// original point through every distinct incident leaf and canonically fold the
// valid limits; off-seam points retain the ordinary single-anchor hot path.
// MEASURED: the per-thread scratch here is NOT the cost of this pass.
//
// This function declares four dynamically-indexed array<*,8> locals and calls
// canonicalInterpolation8 three times, which takes array<f32,8> BY VALUE and
// copies it again -- roughly 80 words that Metal allocates as per-thread
// scratch STATICALLY, for all 64 lanes, whether or not the seamMask early-out
// is taken. That is the same shape as E5's -3.6% named-scalar rewrite, so it
// looked like the next one.
//
// It is not. Compiling the whole seam fold out at source -- removing every
// array AND all of its work -- moved the Advect structured families pass from
// 40.34 to 37.28 ms/advance on symmetric-expansion: -3.06 ms against a floor
// of 3.65 ms (A/A) on that lane. A bit-exact named-scalar rewrite recovers only
// the scratch part of that, so its ceiling is well under the floor. The 40 ms
// is the interpolation itself -- three characteristicSample calls per face --
// not the scratch. Do not spend a delicate canonical-fold rewrite here.
fn seamInterpolationSample(point:vec3f)->vec4f{
  let grid=point/p.physical.x;var seamMask=0u;
  for(var axis=0u;axis<3u;axis+=1u){
    if(abs(grid[axis]-round(grid[axis]))<=1e-5){seamMask|=1u<<axis;}}
  if(seamMask==0u){return invalidVector();}
  var rows:array<u32,8>;var rowCount=0u;var xTerms:array<f32,8>;
  var yTerms:array<f32,8>;var zTerms:array<f32,8>;var sampleCount=0u;
  for(var corner=0u;corner<8u;corner+=1u){
    // Only seam axes contribute distinct probes. Keeping the surviving corner
    // order preserves the canonical fold bit-for-bit while reducing a common
    // one-axis seam from eight directory walks to two.
    if((corner&(~seamMask))!=0u){continue;}var probe=point;
    for(var axis=0u;axis<3u;axis+=1u){if((seamMask&(1u<<axis))!=0u){
      probe[axis]+=select(-1e-4,1e-4,(corner&(1u<<axis))!=0u)*p.physical.x;}}
    let candidate=acceptedRowContaining(probe);if(candidate==INVALID||candidate>=acc(2u)){continue;}
    var duplicate=false;for(var prior=0u;prior<rowCount;prior+=1u){duplicate=duplicate||rows[prior]==candidate;}
    if(duplicate){continue;}rows[rowCount]=candidate;rowCount+=1u;
    let sample=incidentInterpolationElementSample(candidate,point);if(!vectorValid(sample)){continue;}
    xTerms[sampleCount]=sample.x;yTerms[sampleCount]=sample.y;zTerms[sampleCount]=sample.z;sampleCount+=1u;
  }
  if(sampleCount==0u){return invalidVector();}let inverseCount=1./f32(sampleCount);
  return vec4f(inverseCount*vec3f(canonicalInterpolation8(xTerms,sampleCount),
    canonicalInterpolation8(yTerms,sampleCount),canonicalInterpolation8(zTerms,sampleCount)),1.);
}
// Sample the accepted field at an arbitrary characteristic point. Main's
// old-mesh advection re-resolved the containing leaf for every sample point
// (owner(x)); pinning one face's owner closure for the whole midpoint trace
// evaluates the interpolant outside its own element once the departure leaves
// the incident cell, and the clamped/renormalized basis then acts as a
// nearest-hull projection — a speed-proportional energy sink measured at
// -0.8%/step (slow) to -4.6%/step (post-impact) on the mini dam. Resolve the
// element containing the point first; the pinned incident row remains the
// fallback so a point on a directory seam never rejects a face that the
// owner-local closure could serve.
fn characteristicSample(row:u32,point:vec3f)->vec4f{
  let seam=seamInterpolationSample(point);if(vectorValid(seam)){return seam;}
  var anchor=acceptedRowContaining(point);
  if(anchor!=INVALID&&anchor<acc(2u)){
    var sample=interpolationElementSample(anchor,point);
    if(vectorValid(sample)){return sample;}
    sample=adjacentInterpolationElementSample(anchor,point);
    if(vectorValid(sample)){return sample;}
  }
  // Outside the liquid dual mesh, Section 5 prescribes the face field copied
  // from the closest free-surface face. Consume that publication before an
  // incident-row fallback: regularSample clamps its cube coordinates, so
  // evaluating it first silently extends the liquid-side gradient and
  // repeatedly damps a free jet's tangential momentum.
  if(anchor==INVALID){
    let extended=extendedOwnerVelocity(point);
    if(vectorValid(extended)){return extended;}
  }
  var pinned=interpolationElementSample(row,point);
  if(vectorValid(pinned)){return pinned;}
  let adjacent=adjacentInterpolationElementSample(row,point);
  if(vectorValid(adjacent)){return adjacent;}
  // A directory seam can have an accepted anchor while neither local dual
  // closure accepts the point. Retain the same published extension as the
  // final fail-closed fallback for that exceptional case.
  let extended=extendedOwnerVelocity(point);
  if(vectorValid(extended)){return extended;}
  return pinned;
}
// A stored face has an arbitrary low-coordinate owner. Reflections exchange
// that owner with the other incident row, so choosing only owner(handle) at
// the measure-zero face centre selects different dual stars on mirrored
// faces. Section 5 defines one field on the dual mesh, not an owner-sided
// field. Resolve the seam by canonically averaging both valid incident
// element limits; at a free surface there is only one liquid-side limit.
fn faceCharacteristicSample(handle:u32,point:vec3f)->vec4f{
  let low=owner(handle);let high=neighbor(handle);
  // At a one-sided free-surface face, Section 5's published extension is the
  // unique outside-field authority. Use it for tangential components while
  // imposing this face's exact stored u dot n. The former owner-local
  // tetrahedral sample did not reproduce the face DOF and a reflected local
  // fan could therefore launch different midpoint characteristics.
  if(high==INVALID){
    let extended=extendedOwnerVelocity(point);
    let n=normal(handle);let normalValue=value(handle);
    if(vectorValid(extended)&&finite3(n)&&finite(normalValue)){
      let corrected=extended.xyz+(normalValue-canonicalVelocityDot(extended.xyz,n))*n;
      if(finite3(corrected)){return vec4f(corrected,1.);}
    }
  }
  let lowSample=incidentInterpolationElementSample(low,point);
  var highSample=invalidVector();
  if(high!=INVALID&&high!=low){highSample=incidentInterpolationElementSample(high,point);}
  if(vectorValid(lowSample)&&vectorValid(highSample)){
    let result=.5*vec3f(
      canonicalInterpolation4(array<f32,4>(lowSample.x,highSample.x,0.,0.)),
      canonicalInterpolation4(array<f32,4>(lowSample.y,highSample.y,0.,0.)),
      canonicalInterpolation4(array<f32,4>(lowSample.z,highSample.z,0.,0.)));
    if(finite3(result)){return vec4f(result,1.);}
  }
  if(vectorValid(lowSample)){return lowSample;}
  if(vectorValid(highSample)){return highSample;}
  return characteristicSample(low,point);
}
fn rowTouchesDryDirection(row:u32,direction:u32)->bool{
  if(row>=acc(2u)||liquidAt(row)==0u){return true;}
  let rg=rowGeometry[rbase()+row];let d=dimensions();
  let q=vec3u(rg.x%d.x,(rg.x/d.x)%d.y,rg.x/(d.x*d.y));
  let dimension=direction/2u;let positive=(direction&1u)!=0u;var probe=q;
  if(positive){if(q[dimension]+rg.y>=d[dimension]){return false;}probe[dimension]+=rg.y;}
  else{if(q[dimension]==0u){return false;}probe[dimension]-=1u;}
  let other=acceptedRowContainingFinestCell(probe);
  return other==INVALID||other>=acc(2u)||liquidAt(other)==0u;
}
// A row is "dynamic" when it is dry or borders air: exactly the band whose
// power-cell geometry main's identity key would have invalidated, forcing a
// re-trace. The accepted row set holds ONLY liquid rows, so "borders air"
// means an in-domain six-neighbour probe that resolves to no accepted row
// (or, defensively, to a non-liquid one). A probe outside the closed domain
// is a wall, not air; a deep column against a wall still carries.
fn rowTouchesDry(row:u32)->bool{
  if(row>=acc(2u)||liquidAt(row)==0u){return true;}
  for(var direction=0u;direction<6u;direction+=1u){
    if(rowTouchesDryDirection(row,direction)){return true;}
  }
  return false;
}
// The inactive authority bank is already the advection transaction's scratch
// (its value channel holds nextValue). Its owner/neighbor channels are not
// read until a future topology candidate rewrites the entire bank, so they
// provide two collision-free u32 cache words per accepted face.
fn dryOwnerCacheAt(handle:u32)->u32{return (1u-bank())*p.authorityWords+p.ownerOffset+handle;}
fn dryNeighborCacheAt(handle:u32)->u32{return (1u-bank())*p.authorityWords+p.neighborOffset+handle;}
fn rowTouchesDryCached(handle:u32,neighborSide:bool)->bool{
  let at=select(dryOwnerCacheAt(handle),dryNeighborCacheAt(handle),neighborSide);
  return a[at]!=0u;
}

var<workgroup> dryProbePartial:array<u32,64>;
var<workgroup> dryProbeHandle:u32;
var<workgroup> dryProbeOwner:u32;
var<workgroup> dryProbeNeighbor:u32;
var<workgroup> dryProbeEligible:u32;
// A changed-topology publication marks exact identities per face in
// centroid.w. The exact-topology fast path deliberately skips that scatter,
// so its accepted receipt carries the equivalent all-live-faces proof in word
// 13. Treating the receipt as the marker preserves HEAD's advection semantics
// without reintroducing any row, slot, or compiled-image work.
fn faceIdentityCarried(handle:u32)->bool{return acc(13u)!=0u
  ||bitcast<f32>(a[abase()+p.centroidOffset+4u*handle+3u])>.5;}
// One workgroup owns one class-7/8 face. Lanes 0..5 perform the owner's six
// independent neighbour probes; lanes 8..13 do the neighbour row. Two
// eight-lane OR trees reproduce rowTouchesDry without changing the
// directory search, liquid test, physical-wall rule, or carry predicate.
@compute @workgroup_size(64)fn classifyStructuredBoundaryDryProbes(
  @builtin(workgroup_id)g:vec3u,@builtin(local_invocation_index)lid:u32
){
  if(lid==0u){
    let flatIndex=g.x+g.y*65535u;
    let count7=worksets[wbase(7u)+1u];
    var cls=7u;var index=flatIndex;
    if(flatIndex>=count7){cls=8u;index=flatIndex-count7;}
    let handle=workItem(cls,index);
    dryProbeHandle=handle;dryProbeOwner=INVALID;dryProbeNeighbor=INVALID;dryProbeEligible=0u;
    if(handle!=INVALID&&handle<acc(5u)){
      let lo=owner(handle);let hi=neighbor(handle);
      dryProbeOwner=lo;dryProbeNeighbor=hi;
      dryProbeEligible=select(0u,1u,faceIdentityCarried(handle)&&hi!=INVALID);
    }
  }
  workgroupBarrier();
  let handle=dryProbeHandle;let eligible=dryProbeEligible!=0u;
  var dry=0u;
  if(eligible&&lid<16u){
    let local=lid&7u;
    if(local<6u){
      let row=select(dryProbeOwner,dryProbeNeighbor,lid>=8u);
      dry=select(0u,1u,rowTouchesDryDirection(row,local));
    }
  }
  dryProbePartial[lid]=dry;
  workgroupBarrier();
  for(var width=4u;width>0u;width>>=1u){
    if(lid<16u&&(lid&7u)<width){dryProbePartial[lid]|=dryProbePartial[lid+width];}
    workgroupBarrier();
  }
  if(handle!=INVALID&&handle<acc(5u)){
    if(lid==0u){a[dryOwnerCacheAt(handle)]=select(1u,dryProbePartial[0],eligible);}
    if(lid==8u){a[dryNeighborCacheAt(handle)]=select(1u,dryProbePartial[8],eligible);}
  }
}

fn advect(cls:u32,index:u32,flattenedDryRows:bool){
  let handle=workItem(cls,index);
  if(handle==INVALID||handle>=acc(5u)){return;}
  if(!supportPublicationValid()){
    transportMetrics[handle]=invalidVector();
    rejectVector(1u,handle,bitcast<vec4f>(vec4u(supportFailureWord,
      supportWord(p.supportControlOffsetWords+2u),acc(3u),
      supportWord(p.supportControlOffsetWords+3u))),cls);
    return;
  }
  let row=owner(handle);
  let x=centroid(handle);
  let prior=value(handle);
  setNextValue(handle,prior);
  debugAdvectionWord(p.ownerOffset,handle,3.402823e38);
  debugAdvectionWord(p.neighborOffset,handle,3.402823e38);
  debugAdvectionWord(p.metadataOffset,handle,3.402823e38);
  debugAdvectionWord(p.areaOffset,handle,3.402823e38);
  debugAdvectionWord(p.inverseOffset,handle,3.402823e38);
  debugAdvectionWord(p.fractionOffset,handle,3.402823e38);
  debugAdvection3(p.normalOffset,handle,vec3f(3.402823e38));
  debugAdvection3(p.centroidOffset,handle,vec3f(3.402823e38));
  let aperture=bitcast<f32>(a[abase()+p.fractionOffset+handle]);
  if(!finite(aperture)||aperture<0.||aperture>1.){transportMetrics[handle]=invalidVector();rejectSample(3u,handle);return;}
  // The source is an application boundary condition around the paper's
  // Section 5 transport, not another sample of the advected liquid field. A
  // prescribed nozzle face therefore has no characteristic to trace: publish
  // its authored normal velocity directly. Ordinary interface faces still use
  // the exact cube/tetrahedral interpolation below and still fail closed when
  // their extended-field support is incomplete.
  let prescribed=inflowNormalVelocity(handle);
  if(prescribed.y>0.){
    let n=normal(handle);
    let area=bitcast<f32>(a[abase()+p.areaOffset+handle]);
    if(!finite(prescribed.x)||!finite3(n)||!finite(area)||area<=0.){transportMetrics[handle]=invalidVector();rejectSample(3u,handle);return;}
    setNextValue(handle,prescribed.x);
    transportMetrics[handle]=vec4f(prescribed.x*n*area,0.);
    return;
  }
  // A fully prescribed solid face has no transported degree of freedom. Keep
  // the staged prior value: forceFamily re-imposes the exact solid boundary
  // value immediately after this commit and before any divergence consumer,
  // and dropping the solid-velocity read here keeps this kernel inside
  // WebGPU's ten-storage-buffer per-stage limit.
  if(aperture==0.){
    let n=normal(handle);
    let area=bitcast<f32>(a[abase()+p.areaOffset+handle]);
    if(!finite3(n)||!finite(area)||area<=0.||!finite(prior)){transportMetrics[handle]=invalidVector();rejectSample(3u,handle);return;}
    transportMetrics[handle]=vec4f(prior*n*area,0.);
    return;
  }
  // Main's carry semantics, translated to this lane's identity model. Main's
  // DELTA_CARRIED key includes the power-cell geometryCode, which changes
  // whenever the liquid configuration around a face changes — so main
  // re-traces the whole dynamic interface band every step and carries ONLY
  // faces whose local liquid neighbourhood is static. This lane's carry
  // marker keys on structural octree identity alone, which survives interface
  // motion; using it unconditionally froze the momentum field (a corner
  // reservoir stopped sliding down its wall). Gate the carry on the face
  // being deep interior: both incident rows liquid and no six-neighbour of
  // either incident row dry. Interface-band faces always trace, paying the
  // semi-Lagrangian cost exactly where main pays it; deep-interior faces
  // carry exactly, removing the dt-independent whole-field resampling loss
  // measured at up to ~3.5%/step of kinetic energy at the dam-break impact.
  // The accepted row set is liquid-only, so a face with no neighbour row is a
  // free-surface (or wall) face: never carried. Closed-wall faces already
  // returned through the aperture==0 branch above.
  let hiRow=neighbor(handle);
  if(deepIdentityCarryEnabled&&faceIdentityCarried(handle)&&hiRow!=INVALID){
    var deepInterior=false;
    if(flattenedDryRows){
      deepInterior=!rowTouchesDryCached(handle,false)&&!rowTouchesDryCached(handle,true);
    }else{
      deepInterior=!rowTouchesDry(row)&&!rowTouchesDry(hiRow);
    }
    if(deepInterior){
      let n=normal(handle);
      let area=bitcast<f32>(a[abase()+p.areaOffset+handle]);
      if(!finite3(n)||!finite(area)||area<=0.||!finite(prior)){transportMetrics[handle]=invalidVector();rejectSample(3u,handle);return;}
      transportMetrics[handle]=vec4f(prior*n*area,0.);
      return;
    }
  }
  var adv=faceCharacteristicSample(handle,x);
  if(!vectorValid(adv)){transportMetrics[handle]=adv;rejectVector(1u,handle,adv,cls);return;}
  debugAdvectionWord(p.ownerOffset,handle,adv.x);
  debugAdvectionWord(p.neighborOffset,handle,adv.y);
  debugAdvectionWord(p.metadataOffset,handle,adv.z);
  // Second-order midpoint backtrace, as main's old-mesh advection performed
  // (v0 at the face, vm at the half-step, departure from vm). A first-order
  // Euler trace dissipates O(dt^2*|v|*|grad v|) per step, which measured as
  // mechanical-energy retention decaying to 0.83 by t=0.24 s on the mini dam
  // while the midpoint scheme holds ~0.99.
  let midpointRow=characteristicIncidentRow(handle,adv.xyz);
  let midpoint=characteristicPoint(midpointRow,x,.5*p.physical.y,adv.xyz);
  debugAdvectionWord(p.areaOffset,handle,midpoint.x);
  debugAdvectionWord(p.inverseOffset,handle,midpoint.y);
  debugAdvectionWord(p.fractionOffset,handle,midpoint.z);
  let middle=characteristicSample(midpointRow,midpoint);
  if(!vectorValid(middle)){transportMetrics[handle]=middle;rejectVector(1u,handle,middle,cls);return;}
  debugAdvection3(p.normalOffset,handle,middle.xyz);
  let departureRow=characteristicIncidentRow(handle,middle.xyz);
  let departure=characteristicPoint(departureRow,x,p.physical.y,middle.xyz);
  debugAdvection3(p.centroidOffset,handle,departure);
  let transported=characteristicSample(departureRow,departure);
  if(!vectorValid(transported)){transportMetrics[handle]=transported;rejectVector(2u,handle,transported,cls);return;}
  let n=normal(handle);
  let projected=canonicalVelocityDot(transported.xyz,n);
  let area=bitcast<f32>(a[abase()+p.areaOffset+handle]);
  if(!finite(projected)||!finite(area)||area<=0.||!finite(prior)){transportMetrics[handle]=invalidVector();rejectSample(3u,handle);return;}
  setNextValue(handle,projected);
  transportMetrics[handle]=vec4f(projected*n*area,.5*area*max(0.,prior*prior-projected*projected));
}
@compute @workgroup_size(64)fn advectStructuredFamilies(@builtin(global_invocation_id)g:vec3u){let item=unionClassItem(5u,classItem(g));if(item.x!=INVALID){advect(item.x,item.y,false);}}
@compute @workgroup_size(64)fn advectStructuredFamiliesFlattenedBoundary(@builtin(global_invocation_id)g:vec3u){let item=unionClassItem(5u,classItem(g));if(item.x!=INVALID){advect(item.x,item.y,item.x>=7u);}}

fn commitAdvected(cls:u32,index:u32){
  let handle=workItem(cls,index);
  if(handle==INVALID||handle>=acc(5u)){return;}
  // Mirror the advect gate exactly: a lane that staged nothing must not copy
  // stale inactive-bank words over the accepted values.
  if(!supportPublicationValid()){return;}
  a[abase()+p.valuesOffset+handle]=a[nextValueAt(handle)];
}
@compute @workgroup_size(64)fn commitAdvectedStructuredFamilies(@builtin(global_invocation_id)g:vec3u){let item=unionClassItem(5u,classItem(g));if(item.x!=INVALID){commitAdvected(item.x,item.y);}}

var<workgroup> projectionEnergyPartial:array<f32,128>;
var<workgroup> projectionEnergyCount:array<u32,128>;
var<workgroup> projectionEnergyInvalid:array<u32,128>;
var<workgroup> projectionEnergyWetPartial:array<f32,128>;
var<workgroup> projectionEnergyWetCount:array<u32,128>;
var<workgroup> projectionEnergyWetTheta:array<f32,128>;
var<workgroup> projectionEnergyStaggeredPath:array<u32,128>;
fn summarizeProjectionEnergy(lane:u32,stage:u32){
  var energy=0.;
  var wetEnergy=0.;
  var wetThetaEnergy=0.;
  var count=0u;
  var wetCount=0u;
  var invalid=0u;
  var staggeredPathCount=0u;
  if(acc(0u)!=0u||acc(3u)==0u||!boundaryValid()){invalid=1u;}
  for(var cls=5u;cls<9u;cls+=1u){
    let base=wbase(cls);
    if(worksets[base]!=acc(3u)){invalid=1u;continue;}
    let size=worksets[base+1u];
    for(var index=lane;index<size;index+=128u){
      let handle=worksets[base+7u+index];
      if(handle>=acc(5u)){invalid=1u;continue;}
      let sample=value(handle);
      let area=bitcast<f32>(a[abase()+p.areaOffset+handle]);
      let aperture=bitcast<f32>(a[abase()+p.fractionOffset+handle]);
      let inverseDistance=bitcast<f32>(a[abase()+p.inverseOffset+handle]);
      if(!finite(sample)||!finite(area)||area<=0.||!finite(aperture)||aperture<0.||aperture>1.
        ||!finite(inverseDistance)||inverseDistance<=0.){invalid=1u;continue;}
      let dualVolume=area/inverseDistance;
      let contribution=.5*aperture*dualVolume*sample*sample;
      if(!finite(contribution)||contribution<0.){invalid=1u;continue;}
      energy+=contribution;
      count+=1u;
      // Attribution needs the liquid's energy separately: dry faces carry
      // Section 5 extension copies whose decay is not a physical loss.
      let lo=owner(handle);
      let hi=neighbor(handle);
      if(lo<acc(2u)&&(liquidAt(lo)!=0u||(hi!=INVALID&&hi<acc(2u)&&liquidAt(hi)!=0u))){
        wetEnergy+=contribution;
        wetCount+=1u;
        // Physical-norm attribution: weight the face by its liquid fraction
        // theta (the GFM pressure scale is 1/theta; interior faces carry
        // scale 1). The unweighted proxy charges a barely-wet surface face
        // its full dual volume, which hid where mechanical energy actually
        // enters during the dam wall impact.
        let scale=bitcast<f32>(a[abase()+p.pressureScaleOffset+handle]);
        if(finite(scale)&&scale>=1.){wetThetaEnergy+=contribution/scale;}
        else{wetThetaEnergy+=contribution;}
        // Stage 1 also takes a sampler-path census at the face centroid:
        // which basis would this face's advection have used?
        if(stage==1u&&lo<acc(2u)&&regularTag(lo,vec3i(0))==lo
          &&vectorValid(staggeredSample(lo,centroid(handle)))){staggeredPathCount+=1u;}
      }
    }
  }
  projectionEnergyPartial[lane]=energy;
  projectionEnergyCount[lane]=count;
  projectionEnergyInvalid[lane]=invalid;
  projectionEnergyWetPartial[lane]=wetEnergy;
  projectionEnergyWetCount[lane]=wetCount;
  projectionEnergyWetTheta[lane]=wetThetaEnergy;
  projectionEnergyStaggeredPath[lane]=staggeredPathCount;
  workgroupBarrier();
  for(var width=64u;width>0u;width>>=1u){
    if(lane<width){
      projectionEnergyPartial[lane]+=projectionEnergyPartial[lane+width];
      projectionEnergyCount[lane]+=projectionEnergyCount[lane+width];
      projectionEnergyInvalid[lane]|=projectionEnergyInvalid[lane+width];
      projectionEnergyWetPartial[lane]+=projectionEnergyWetPartial[lane+width];
      projectionEnergyWetCount[lane]+=projectionEnergyWetCount[lane+width];
      projectionEnergyWetTheta[lane]+=projectionEnergyWetTheta[lane+width];
      projectionEnergyStaggeredPath[lane]+=projectionEnergyStaggeredPath[lane+width];
    }
    workgroupBarrier();
  }
  if(lane==0u){
    // Four independent per-stage records; the host cross-checks generation,
    // bank, and family coverage across stages. Stage order in the substep:
    // 0 start-of-step (post-remap), 1 post-advection, 2 post-force
    // (pre-projection), 3 post-projection.
    let generation=acc(3u);
    let failed=projectionEnergyInvalid[0]!=0u||projectionEnergyCount[0]==0u
      ||!finite(projectionEnergyPartial[0]);
    let base=8u*stage;
    projectionEnergyStats[base]=select(0u,1u+stage,failed);
    projectionEnergyStats[base+1u]=(generation<<1u)|bank();
    projectionEnergyStats[base+2u]=projectionEnergyCount[0];
    projectionEnergyStats[base+3u]=bitcast<u32>(projectionEnergyPartial[0]);
    projectionEnergyStats[base+4u]=projectionEnergyWetCount[0];
    projectionEnergyStats[base+5u]=bitcast<u32>(projectionEnergyWetPartial[0]);
    projectionEnergyStats[base+6u]=bitcast<u32>(projectionEnergyWetTheta[0]);
    projectionEnergyStats[base+7u]=projectionEnergyStaggeredPath[0];
  }
}
@compute @workgroup_size(128)
fn summarizeStructuredPreProjectionEnergy(@builtin(local_invocation_index)lane:u32){
  summarizeProjectionEnergy(lane,2u);
}
@compute @workgroup_size(128)
fn summarizeStructuredPostProjectionEnergy(@builtin(local_invocation_index)lane:u32){
  summarizeProjectionEnergy(lane,3u);
}
@compute @workgroup_size(128)
fn summarizeStructuredStartEnergy(@builtin(local_invocation_index)lane:u32){
  summarizeProjectionEnergy(lane,0u);
}
@compute @workgroup_size(128)
fn summarizeStructuredPostAdvectionEnergy(@builtin(local_invocation_index)lane:u32){
  summarizeProjectionEnergy(lane,1u);
}

fn closedWorld(handle:u32)->bool{
  if(neighbor(handle)!=INVALID){return false;}
  let x=centroid(handle)/p.physical.x;
  let n=normal(handle);
  for(var axis=0u;axis<3u;axis+=1u){
    if(n[axis]<-.5&&x[axis]<=1e-4&&(p.closedMask&(1u<<(2u*axis)))!=0u){return true;}
    if(n[axis]>.5&&x[axis]>=f32(dimensions()[axis])-1e-4&&(p.closedMask&(1u<<(2u*axis+1u)))!=0u){return true;}
  }
  return false;
}
fn forceFamily(cls:u32,index:u32){
  let handle=workItem(cls,index);
  if(handle==INVALID||handle>=acc(5u)){return;}
  let aperture=bitcast<f32>(a[abase()+p.fractionOffset+handle]);
  let solid=solidVelocityAt(handle);
  if(!finite(aperture)||aperture<0.||aperture>1.||!finite(solid)){rejectSample(10u,handle);return;}
  let prescribed=inflowNormalVelocity(handle);
  if(prescribed.y>0.){if(!finite(prescribed.x)){rejectSample(13u,handle);return;}setValue(handle,prescribed.x);return;}
  if(aperture==0.){setValue(handle,solid);return;}
  // Gravity is a body force on the liquid. Dry faces only carry Section 5
  // extended and advected air values; integrating gravity into them built a
  // field that grew by g*dt every substep in air, which the projection never
  // sees and the extension march never resets, and which the staggered
  // regular sampler would ingest at the free surface. Every face the
  // divergence RHS integrates is wet by construction (its owner is liquid).
  let lo=owner(handle);
  let hi=neighbor(handle);
  if(lo>=acc(2u)||(hi!=INVALID&&hi>=acc(2u))){rejectSample(12u,handle);return;}
  let wet=liquidAt(lo)!=0u||(hi!=INVALID&&liquidAt(hi)!=0u);
  if(!wet){return;}
  let forced=value(handle)+p.physical.y*canonicalVelocityDot(p.gravity.xyz,normal(handle));
  if(!finite(forced)){rejectSample(11u,handle);return;}
  setValue(handle,forced);
}
@compute @workgroup_size(64)fn forceStructuredFamilies(@builtin(global_invocation_id)g:vec3u){let item=unionClassItem(5u,classItem(g));if(item.x!=INVALID){forceFamily(item.x,item.y);}}

fn divergenceRow(cls:u32,index:u32){
  let row=workItem(cls,index);
  if(row==INVALID||row>=acc(2u)){return;}
  if(liquidAt(row)==0u){rhs[row]=0.;return;}
  var fluxTerms:array<f32,31>;
  let count=rowSlotCount(row);
  if(count>p.maxSlots){rhs[row]=0.;rejectSample(20u,row);return;}
  for(var local=0u;local<count;local+=1u){
    let at=row*p.maxSlots+local;
    let handle=a[abase()+p.rowHandleOffset+at];
    if(handle==INVALID||handle>=acc(5u)){rhs[row]=0.;rejectSample(21u,row);return;}
    let sign=f32(bitcast<i32>(a[abase()+p.rowSignOffset+at]));
    let area=bitcast<f32>(a[abase()+p.areaOffset+handle]);
    let aperture=bitcast<f32>(a[abase()+p.fractionOffset+handle]);
    let solid=solidVelocityAt(handle);
    let sample=value(handle);
    if(!finite(sign)||!finite(area)||!finite(aperture)||aperture<0.||aperture>1.||!finite(solid)||!finite(sample)){rhs[row]=0.;rejectSample(22u,row);return;}
    let prescribed=inflowNormalVelocity(handle);
    let boundaryVelocity=select(aperture*sample+(1.-aperture)*solid,prescribed.x,prescribed.y>0.);
    fluxTerms[local]=sign*area*boundaryVelocity;
  }
  let flux=canonicalReconstructionSum(fluxTerms,count);
  if(!finite(flux)||!finite(p.physical.y)||p.physical.y<0.
    ||!finite(p.physical.z)||p.physical.z<=0.){rhs[row]=0.;rejectSample(23u,row);return;}
  // The fenced t=0 publication is the exact zero-length member of the same
  // projection operator family.  It validates every geometric/velocity input
  // above, but its integrated divergence RHS is identically zero; dividing by
  // dt would be undefined.  Positive-time substeps retain the physical
  // integrated Eq. (3)/(4) equation below, and invalid inputs still fail closed.
  if(p.physical.y==0.){rhs[row]=0.;return;}
  // A stores area/distance and projection applies dt/rho times the pressure
  // gradient. Therefore A p = -(rho/dt) integral(u.n dA), with this solver's
  // residual convention. Dividing by cell volume would make the operator,
  // RHS, and projection describe different equations at different leaf sizes.
  rhs[row]=p.physical.z*flux/p.physical.y;
}
@compute @workgroup_size(64)fn divergenceStructuredRows(@builtin(global_invocation_id)g:vec3u){let item=unionClassItem(0u,classItem(g));if(item.x!=INVALID){divergenceRow(item.x,item.y);}}
// Class 4 sits outside the liquid pressure system, so the row union above
// never visits it and nothing else defines its divergence RHS: a reused row
// index would otherwise carry the previous generation's flux into the solve.
// The identity row's RHS is definitionally zero (applyIdentityRow re-proves
// that fail-closed), so author it from the class's own accepted worklist.
@compute @workgroup_size(64)fn zeroStructuredIdentityRhsRows(@builtin(global_invocation_id)g:vec3u){let row=workItem(4u,classItem(g));if(row!=INVALID&&row<acc(2u)){rhs[row]=0.;}}

// The paper's cut-cell solid coupling (Batty-style apertures) constrains
// u.n on solid faces bilaterally, so the solve balances a liquid sheet on
// the tank ceiling with sustained suction: the sheet hangs instead of
// falling away. A free-surface liquid in contact with a solid it may leave
// is a unilateral constraint -- contact pressure obeys p >= 0, with p < 0
// replaced by separation. Post-solve pressure clamping enforces neither
// side: clamped rows stop conserving mass and measured as sinks that pulled
// MORE liquid onto the ceiling. Instead this stage only MARKS liquid rows
// that hold tension against the closed ceiling (one-step-lagged active set
// of the contact LCP); the next step's boundary rebuild opens the marked
// rows' world faces so the solve itself resolves the separation with a
// p = 0 ghost and fully consistent divergence. The mask stores the exact
// world-face bits in tension-against-gravity, so a marked row stays
// released while its tension persists and re-closes the step after the
// tension clears. Authored terrain/rigid overhangs are still bilateral;
// extending the mask to SDF-cut faces is the known follow-up.
fn worldContactBit(handle:u32)->u32{
  if(neighbor(handle)!=INVALID){return 0u;}
  let x=centroid(handle)/p.physical.x;
  let n=normal(handle);
  for(var axis=0u;axis<3u;axis+=1u){
    if(n[axis]<-.5&&x[axis]<=1e-4){return 1u<<(2u*axis);}
    if(n[axis]>.5&&x[axis]>=f32(dimensions()[axis])-1e-4){return 1u<<(2u*axis+1u);}
  }
  return 0u;
}
fn markSeparationRow(cls:u32,index:u32){
  let row=workItem(cls,index);
  if(row==INVALID||row>=acc(2u)){return;}
  let rg=rowGeometry[rbase()+row];
  let cell=rg.x;
  if(cell>=arrayLength(&separationMask)){return;}
  var faceBits=0u;
  let weight=length(p.gravity.xyz);
  if(liquidAt(row)!=0u&&weight>1e-6){
    let solved=pressure[row];
    let count=rowSlotCount(row);
    // Complementarity hysteresis. Pressure below a quarter of the one-cell
    // hydrostatic scale rho g h is numerically indistinguishable from release,
    // so it opens or renews the face; only genuine contact pressure re-closes
    // it. Requiring the first solve to cross zero held the initially stationary
    // top layer for one substep even though gravity had already released it.
    // Without renewal this
    // mark was an unconditional per-epoch overwrite: a separated film solves
    // p ~ 0, loses its mark the very next epoch, re-welds to the lid through
    // the closed face's divergence row, and falls at an open/closed duty
    // cycle instead of g (the free-fall drop oracles measure the top liquid
    // layer saturating near 0.9 m/s under a 2.8 m/s parabola).
    let previous=separationMask[cell];
    let previousBits=previous&63u;
    let age=(acc(3u)-(previous>>6u))&0x3ffffffu;
    let contact=.25*p.physical.z*weight*p.physical.x;
    let opening=finite(solved)&&solved<contact;
    let renewing=finite(solved)&&previousBits!=0u&&age<=2u&&solved<contact;
    if((opening||renewing)&&count<=p.maxSlots){
      // Gravity-relative, axis-free: a closed world face separates when its
      // outward normal opposes gravity within 60 degrees (overhead
      // contact), whichever axis and sign the scene's gravity and closed
      // boundaries use. Wider active sets measured strictly worse: freeing
      // vertical walls during the impact transient turns every released
      // face into a p = 0 ghost that compressed liquid accelerates
      // through (2x the late ceiling residue), and freeing floor-like
      // contact let the whole pool trampoline off the tank floor.
      let up=-p.gravity.xyz/weight;
      for(var local=0u;local<count;local+=1u){
        let handle=a[abase()+p.rowHandleOffset+row*p.maxSlots+local];
        if(handle==INVALID||handle>=acc(5u)){continue;}
        let bit=worldContactBit(handle);
        if(bit==0u||(p.closedMask&bit)==0u){continue;}
        if(dot(normal(handle),up)>.5){faceBits|=bit;}
      }
      // Renewal keeps faces that tension opened; it must not open new ones.
      if(!opening){faceBits&=previousBits;}
    }
  }
  separationMask[cell]=(acc(3u)<<6u)|faceBits;
}
@compute @workgroup_size(64)fn separateStructuredRows(@builtin(global_invocation_id)g:vec3u){let item=unionClassItem(0u,classItem(g));if(item.x!=INVALID){markSeparationRow(item.x,item.y);}}

// Fluid -> rigid, the exact adjoint of the solid term the divergence RHS
// already integrates.
//
// divergenceRow constrains each liquid row with
//   sum_h sign*area*(aperture*u_f + (1-aperture)*u_s) = 0,
// so the solved pressure is the multiplier of a constraint the SOLID normal
// velocity enters. Differentiating that constraint with respect to u_s gives
// the force the fluid transmits through the blocked share of each cut face:
//   F_h = p_row * area_h * (1 - aperture_h) * (sign_h * n_h),
// i.e. pressure pushing outward from the row onto whatever occupies the
// closed fraction. Summed over a submerged body's whole cut boundary against
// a hydrostatic field this is exactly -rho*V*g, so buoyancy, form drag and
// the impact response all fall out of the one term; nothing here re-derives
// Archimedes on the side. The impulse lands in the same fixed-point exchange
// the tall-cell and quadtree lanes write, so the resident GPU integrator is
// unchanged and the response is frame-lagged by exactly one substep.
//
// Attribution matches resolveStructuredSolidSlots face for face: the
// nearest body whose SDF is within half the face's own row spacing owns the
// blocked share. A face cut by terrain and by no body contributes nothing.
fn quaternionRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);let uuv=cross(q.yzw,uv);return v+2.*(q.x*uv+uuv);}
fn quaternionInverseRotate(q:vec4f,v:vec3f)->vec3f{return quaternionRotate(vec4f(q.x,-q.yzw),v);}
fn rigidSdf(body:RigidBody,world:vec3f)->f32{
  let q=quaternionInverseRotate(body.orientation,world-body.positionShape.xyz);
  let d=body.dimensions.xyz;let shape=i32(round(body.positionShape.w));
  if(shape==0){return length(q)-d.x;}
  if(shape==1){let b=abs(q)-.5*d;return length(max(b,vec3f(0)))+min(max(b.x,max(b.y,b.z)),0.);}
  if(shape==2){let cy=clamp(q.y,-.5*d.y,.5*d.y);return length(vec3f(q.x,q.y-cy,q.z))-d.x;}
  let radial=length(q.xz)-d.x;let axial=abs(q.y)-.5*d.y;
  return length(max(vec2f(radial,axial),vec2f(0)))+min(max(radial,axial),0.);
}
// Structured centroids are lattice-origin; authored rigid poses centre x and
// z about the container, exactly as the solid vertex-SDF producer does.
fn solidWorld(x:vec3f)->vec3f{return x-vec3f(.5*f32(p.dimensionX)*p.physical.x,0.,.5*f32(p.dimensionZ)*p.physical.x);}
fn accumulateBodyImpulse(body:u32,impulse:vec3f,torque:vec3f){
  let base=body*12u;
  if(base+5u>=arrayLength(&rigidExchange)){return;}
  atomicAdd(&rigidExchange[base],i32(round(impulse.x*1e6)));
  atomicAdd(&rigidExchange[base+1u],i32(round(impulse.y*1e6)));
  atomicAdd(&rigidExchange[base+2u],i32(round(impulse.z*1e6)));
  atomicAdd(&rigidExchange[base+3u],i32(round(torque.x*1e6)));
  atomicAdd(&rigidExchange[base+4u],i32(round(torque.y*1e6)));
  atomicAdd(&rigidExchange[base+5u],i32(round(torque.z*1e6)));
}
fn bodyImpulseRow(cls:u32,index:u32){
  let row=workItem(cls,index);
  if(row==INVALID||row>=acc(2u)){return;}
  if(p.bodyCount==0u||liquidAt(row)==0u){return;}
  let solved=pressure[row];
  if(!finite(solved)){rejectSample(40u,row);return;}
  // t=0 publishes the same operator with a zero-length step; there is no
  // impulse to transfer and dividing the exchange by it would be undefined.
  if(p.physical.y<=0.){return;}
  let count=rowSlotCount(row);
  if(count>p.maxSlots){rejectSample(41u,row);return;}
  for(var local=0u;local<count;local+=1u){
    let at=row*p.maxSlots+local;
    let handle=a[abase()+p.rowHandleOffset+at];
    if(handle==INVALID||handle>=acc(5u)){rejectSample(42u,row);return;}
    let aperture=bitcast<f32>(a[abase()+p.fractionOffset+handle]);
    if(!finite(aperture)||aperture<0.||aperture>1.){rejectSample(43u,handle);return;}
    if(aperture>=1.){continue;}
    let area=bitcast<f32>(a[abase()+p.areaOffset+handle]);
    let sign=f32(bitcast<i32>(a[abase()+p.rowSignOffset+at]));
    let n=normal(handle);
    if(!finite(area)||!finite(sign)||!finite3(n)){rejectSample(44u,handle);return;}
    let world=solidWorld(centroid(handle));
    if(!finite3(world)){rejectSample(45u,handle);return;}
    let inv=bitcast<f32>(a[abase()+p.inverseOffset+handle]);
    let half=.5/max(inv,1e-20);
    var best=1e20;var chosen=INVALID;
    for(var body=0u;body<p.bodyCount;body+=1u){
      if(body>=arrayLength(&rigidBodies)){rejectSample(46u,handle);return;}
      let sdf=rigidSdf(rigidBodies[body],world);
      if(sdf<half&&sdf<best){best=sdf;chosen=body;}
    }
    if(chosen==INVALID){continue;}
    let impulse=p.physical.y*solved*area*(1.-aperture)*sign*n;
    if(!finite3(impulse)){rejectSample(47u,handle);return;}
    let torque=cross(world-rigidBodies[chosen].positionShape.xyz,impulse);
    if(!finite3(torque)){rejectSample(48u,handle);return;}
    accumulateBodyImpulse(chosen,impulse,torque);
  }
}
@compute @workgroup_size(64)fn exchangeStructuredBodyImpulseRows(@builtin(global_invocation_id)g:vec3u){let item=unionClassItem(0u,classItem(g));if(item.x!=INVALID){bodyImpulseRow(item.x,item.y);}}

fn projectFamily(cls:u32,index:u32){
  let handle=workItem(cls,index);
  if(handle==INVALID||handle>=acc(5u)){return;}
  let lo=owner(handle);
  let hi=neighbor(handle);
  if(lo>=acc(2u)||(hi!=INVALID&&hi>=acc(2u))){rejectSample(30u,handle);return;}
  let n=normal(handle);
  if(!finite3(n)){rejectSample(31u,handle);return;}
  let aperture=bitcast<f32>(a[abase()+p.fractionOffset+handle]);
  let solid=solidVelocityAt(handle);
  if(!finite(aperture)||aperture<0.||aperture>1.||!finite(solid)){rejectSample(32u,handle);return;}
  var projected=solid;
  if(aperture>0.){
    var pressureLo=0.;
    var pressureHi=0.;
    if(liquidAt(lo)!=0u){pressureLo=pressure[lo];}
    if(hi!=INVALID&&liquidAt(hi)!=0u){pressureHi=pressure[hi];}
    let inv=bitcast<f32>(a[abase()+p.inverseOffset+handle]);
    let scale=bitcast<f32>(a[abase()+p.pressureScaleOffset+handle]);
    projected=value(handle)-p.physical.y*(pressureHi-pressureLo)*inv*scale/p.physical.z;
  }
  let prescribed=inflowNormalVelocity(handle);if(prescribed.y>0.){projected=prescribed.x;}
  if(!finite(projected)){rejectSample(33u,handle);return;}
  setValue(handle,projected);
}
@compute @workgroup_size(64)fn projectStructuredFamilies(@builtin(global_invocation_id)g:vec3u){let item=unionClassItem(5u,classItem(g));if(item.x!=INVALID){projectFamily(item.x,item.y);}}

fn canonicalInterpolation8(values:array<f32,8>,count:u32)->f32{
  var sorted=values;
  for(var i=1u;i<count;i+=1u){let value=sorted[i];var j=i;
    loop{if(j==0u||abs(sorted[j-1u])<=abs(value)){break;}sorted[j]=sorted[j-1u];j-=1u;}
    sorted[j]=value;
  }
  var sum=0.;var i=0u;
  loop{if(i>=count){break;}let magnitude=abs(sorted[i]);var balance=0;var j=i;
    loop{if(j>=count||abs(sorted[j])!=magnitude){break;}
      if(sorted[j]>0.){balance+=1;}else if(sorted[j]<0.){balance-=1;}j+=1u;}
    sum+=f32(balance)*magnitude;i=j;
  }
  return sum;
}
fn canonicalInterpolation4(values:array<f32,4>)->f32{
  var widened:array<f32,8>;
  for(var i=0u;i<4u;i+=1u){widened[i]=values[i];}
  return canonicalInterpolation8(widened,4u);
}
fn canonicalReconstructionSum(values:array<f32,31>,count:u32)->f32{
  var sorted=values;
  for(var i=1u;i<count;i+=1u){let value=sorted[i];var j=i;
    loop{if(j==0u||abs(sorted[j-1u])<=abs(value)){break;}sorted[j]=sorted[j-1u];j-=1u;}
    sorted[j]=value;
  }
  var sum=0.;var i=0u;
  loop{
    if(i>=count){break;}
    let magnitude=abs(sorted[i]);var balance=0;var j=i;
    loop{if(j>=count||abs(sorted[j])!=magnitude){break;}
      if(sorted[j]>0.){balance+=1;}else if(sorted[j]<0.){balance-=1;}j+=1u;}
    sum+=f32(balance)*magnitude;i=j;
  }
  return sum;
}
fn reconstructRow(cls:u32,index:u32){
  let row=workItem(cls,index);
  if(row==INVALID||row>=acc(2u)){return;}
  let header=caseHeader(metrics[row].caseId);
  if(header.y>p.maxSlots){rejectSample(50u,row);return;}
  var termsX:array<f32,31>;var termsY:array<f32,31>;var termsZ:array<f32,31>;
  for(var local=0u;local<header.y;local+=1u){
    let at=row*p.maxSlots+local;
    let handle=a[abase()+p.rowHandleOffset+at];
    if(handle==INVALID||handle>=acc(5u)){rejectSample(51u,row);return;}
    let other=a[abase()+p.rowNeighborOffset+at];
    if(other!=INVALID&&other>=acc(2u)){rejectSample(52u,row);return;}
    let sample=f32(bitcast<i32>(a[abase()+p.rowSignOffset+at]))*value(handle);
    let global=a[abase()+p.rowCatalogOffset+at];
    let reconstruction=p.denseOffset+p.reconstructionOffset+3u*global;
    let coefficient=vec3f(bitsf(reconstruction),bitsf(reconstruction+1u),bitsf(reconstruction+2u));
    if(!finite(sample)||!finite3(coefficient)){rejectSample(53u,row);return;}
    let term=coefficient*sample;termsX[local]=term.x;termsY[local]=term.y;termsZ[local]=term.z;
  }
  let canonical=vec3f(canonicalReconstructionSum(termsX,header.y),
    canonicalReconstructionSum(termsY,header.y),canonicalReconstructionSum(termsZ,header.y));
  let projected=inverseTransform(canonical,metrics[row].transformAndFlags&63u);
  if(!finite3(projected)){rejectSample(54u,row);return;}
  rowVelocity[rbase()+row]=vec4f(projected,1.);
}
@compute @workgroup_size(64)fn reconstructStructuredRows(@builtin(global_invocation_id)g:vec3u){let item=unionClassItem(0u,classItem(g));if(item.x!=INVALID){reconstructRow(item.x,item.y);}}
`;
