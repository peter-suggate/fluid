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
  OCTREE_AIR_SUPPORT_VALID,
  type OctreeAirVelocitySupportLayout,
} from "./webgpu-octree-air-velocity-support";
import type { PassBroker } from "./webgpu-pass-broker";
import { octreeAlgorithmDiagnosticsEnabled } from "./octree-algorithm-diagnostics";
import { GPU_RIGID_BODY_CAPACITY } from "./webgpu-rigid-body";
import type { SurfaceInflowState } from "./webgpu-quadtree-builder";

const ROW_CLASSES = [0, 1, 2, 3] as const;
const FAMILY_CLASSES = [5, 6, 7, 8] as const;
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
 * `encodeClasses` dispatches one workgroup of 64 lanes per 64 workset entries,
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
  readonly encodedAdvectionDispatchCount = 9;
  readonly encodedForceDivergenceDispatchCount = 10;
  readonly encodedProjectionDispatchCount = 13;
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
  /** Nine accepted class dispatch records, one vec3u per workset class. */
  readonly dispatch: GPUBuffer;
  private readonly params: readonly [GPUBuffer, GPUBuffer, GPUBuffer];
  private readonly prepare: GPUComputePipeline;
  private readonly topologyTransfer: GPUComputePipeline;
  private readonly advection: readonly GPUComputePipeline[];
  private readonly advectionCommit: readonly GPUComputePipeline[];
  private readonly force: readonly GPUComputePipeline[];
  private readonly divergence: readonly GPUComputePipeline[];
  private readonly separation: readonly GPUComputePipeline[];
  private readonly bodyImpulse: readonly GPUComputePipeline[];
  private readonly projection: readonly GPUComputePipeline[];
  private readonly reconstruct: readonly GPUComputePipeline[];
  private readonly summarizePreProjectionEnergy: GPUComputePipeline;
  private readonly summarizePostProjectionEnergy: GPUComputePipeline;
  private readonly summarizeStartEnergy: GPUComputePipeline;
  private readonly summarizePostAdvectionEnergy: GPUComputePipeline;
  private readonly groups = new WeakMap<GPUComputePipeline,
    WeakMap<GPUBuffer, WeakMap<GPUBuffer, GPUBindGroup>>>();
  /** Encode the four energy summarizers only when a host actually reads them. */
  private readonly projectionEnergyProbe = structuredProjectionEnergyProbeEnabled();
  /** Diagnostic-only per-class dispatch-width census; see `censusTick`. */
  private readonly censusEnabled = structuredWorksetCensusEnabled();
  private censusStaging?: GPUBuffer;
  private censusPhase: "idle" | "copied" | "mapping" = "idle";
  private censusStep = 0;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly resources: StructuredDynamicsResources) {
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
      size: 9 * 12, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT
        | GPUBufferUsage.COPY_SRC });
    this.transportMetrics = resources.selectorRows;
    this.projectionEnergyStats = device.createBuffer({
      label: "Structured paired projection kinetic energy",
      size: STRUCTURED_PROJECTION_ENERGY_WORDS * 4,
      usage: storage,
    });
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
    floats[40] = resources.physicalCellSize;
    words[48] = resources.selectorOffsetWords;
    words[49] = resources.selectorStride;
    words[50] = resources.airSupportLayout.regularTagOffsetWords;
    words[51] = resources.airSupportLayout.controlOffsetWords;
    words[52] = resources.airSupportLayout.supportVectorOffsetWords;
    words[53] = resources.airSupportLayout.supportCapacity;
    words[54] = resources.airSupportLayout.ownerDirectoryOffsetWords;
    words[55] = resources.airSupportLayout.ownerDirectoryCellCapacity;
    for (const buffer of this.params) device.queue.writeBuffer(buffer, 0, words.buffer);
    const shaderModule = device.createShaderModule({ label: "Structured velocity dynamics", code: structuredVelocityDynamicsWGSL });
    const make = (name: string) => device.createComputePipeline({ label: name,
      layout: "auto", compute: { module: shaderModule, entryPoint: name } });
    this.prepare = make("prepareStructuredDynamics");
    this.topologyTransfer = make("transferStructuredTopologyCandidate");
    this.advection = FAMILY_CLASSES.map((value) => make(`advectStructuredClass${value}`));
    this.advectionCommit = FAMILY_CLASSES.map((value) => make(`commitAdvectedStructuredClass${value}`));
    this.force = FAMILY_CLASSES.map((value) => make(`forceStructuredClass${value}`));
    this.divergence = ROW_CLASSES.map((value) => make(`divergenceStructuredClass${value}`));
    this.separation = ROW_CLASSES.map((value) => make(`separateStructuredClass${value}`));
    this.bodyImpulse = ROW_CLASSES.map((value) => make(`exchangeStructuredBodyImpulseClass${value}`));
    this.projection = FAMILY_CLASSES.map((value) => make(`projectStructuredClass${value}`));
    this.reconstruct = ROW_CLASSES.map((value) => make(`reconstructStructuredClass${value}`));
    this.summarizePreProjectionEnergy = make("summarizeStructuredPreProjectionEnergy");
    this.summarizePostProjectionEnergy = make("summarizeStructuredPostProjectionEnergy");
    this.summarizeStartEnergy = make("summarizeStructuredStartEnergy");
    this.summarizePostAdvectionEnergy = make("summarizeStructuredPostAdvectionEnergy");
    this.allocatedBytes = catalogBytes + this.dispatch.size
      + this.projectionEnergyStats.size
      + 3 * 256;
  }

  private update(stage: 0 | 1 | 2, dt: number, density: number,
    gravity: readonly [number, number, number], inflow?: SurfaceInflowState): GPUBuffer {
    if (!(dt >= 0) || !Number.isFinite(dt) || !(density > 0) || !Number.isFinite(density)
      || gravity.some((value) => !Number.isFinite(value))) throw new RangeError("Invalid structured dynamics parameters");
    const bytes = new ArrayBuffer(28), floats = new Float32Array(bytes);
    floats[0] = dt; floats[1] = density; floats.set(gravity, 3);
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
    if (!Number.isSafeInteger(count) || count < 0 || count > GPU_RIGID_BODY_CAPACITY) {
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
   * storage and `encodeClasses` reads the same buffer as an indirect argument.
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
    pass.setBindGroup(0, this.group(pipeline, [0, 1, 2, 3, 4, 5, 11, 16, 17, 18, 23], params));
    pass.dispatchWorkgroups(1);
  }

  private encodeClasses(broker: PassBroker, pipelines: readonly GPUComputePipeline[], classes: readonly number[],
    bindings: readonly number[], params: GPUBuffer, label: string, pressure?: GPUBuffer): void {
    pipelines.forEach((pipeline, index) => {
      const pass = broker.compute({ label: `${label} ${classes[index]}` }); pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.group(pipeline, bindings, params, pressure));
      pass.dispatchWorkgroupsIndirect(this.dispatch, classes[index]! * 12);
    });
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
    const params = this.update(0, dt, 1, [0, 0, 0], inflow); this.encodePrepare(broker, params);
    this.censusTick(broker);
    this.encodeProjectionEnergy(broker, params, "start");
    this.encodeClasses(broker, this.advection, FAMILY_CLASSES, [0, 1, 2, 3, 4, 5, 6, 11, 16, 17, 18], params,
      "Advect structured family class");
    // The staggered regular sampler reads neighbouring face degrees of
    // freedom while every lane produces its own destination, so destinations
    // stage into the inactive authority bank and become the accepted values
    // only after this fence closes the intra-dispatch race window.
    broker.fence("structured advected destinations staged");
    this.encodeClasses(broker, this.advectionCommit, FAMILY_CLASSES, [0, 1, 2, 11, 17, 18], params,
      "Commit advected structured family class");
    broker.fence("structured advected destinations committed");
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
    broker.fence("changed topology face velocities transferred");
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
    const params = this.update(1, dt, density, gravity, inflow); this.encodePrepare(broker, params);
    this.encodeClasses(broker, this.force, FAMILY_CLASSES, [0, 1, 2, 11, 16, 17, 22], params,
      "Force and constrain structured family class");
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic before pre-projection energy summary");
    }
    this.encodeProjectionEnergy(broker, params, "pre");
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic after pre-projection energy summary");
    }
    this.encodeClasses(broker, this.divergence, ROW_CLASSES, [0, 1, 2, 5, 6, 11, 14, 16, 17, 22], params,
      "Fuse structured divergence RHS class");
  }

  encodeProjection(broker: PassBroker, dt: number, density: number,
    gravity: readonly [number, number, number] = [0, 0, 0],
    pressure = this.resources.pressure,
    couplingBodyCount = 0,
    inflow?: SurfaceInflowState): void {
    if (this.destroyed) throw new Error("Structured dynamics is destroyed");
    const params = this.update(2, dt, density, gravity, inflow); this.encodePrepare(broker, params);
    this.updateCouplingBodies(couplingBodyCount);
    // Publish the lagged unilateral-contact active set from the solved
    // pressure: rows holding tension against gravity-opposed closed world
    // faces are marked, and the NEXT step's boundary rebuild opens exactly
    // those faces. The solved pressure itself is never mutated, so this
    // step's projection and divergence bookkeeping stay exactly the
    // variational solution.
    this.encodeClasses(broker, this.separation, ROW_CLASSES,
      [0, 1, 2, 3, 5, 6, 11, 13, 15, 16, 17], params,
      "Mark structured overhead separation row class", pressure);
    // Read the solved pressure before the projection stage consumes it. Both
    // stages are read-only in `pressure`, so no fence separates them; the
    // exchange only writes the resident rigid buffer, which the integrator
    // reads once, after the whole advance.
    if (couplingBodyCount > 0) {
      this.encodeClasses(broker, this.bodyImpulse, ROW_CLASSES,
        [0, 1, 2, 5, 6, 11, 13, 16, 17, 25, 26], params,
        "Exchange structured body impulse row class", pressure);
    }
    this.encodeClasses(broker, this.projection, FAMILY_CLASSES,
      [0, 1, 2, 11, 13, 16, 17, 22], params,
      "Project structured family class", pressure);
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic before post-projection energy summary");
    }
    this.encodeProjectionEnergy(broker, params, "post");
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic after post-projection energy summary");
    }
    this.encodeClasses(broker, this.reconstruct, ROW_CLASSES,
      [0, 1, 2, 4, 5, 6, 11, 17], params,
      "Reconstruct projected structured rows");
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.catalog.destroy(); this.dispatch.destroy();
    this.projectionEnergyStats.destroy();
    this.params.forEach((buffer) => buffer.destroy());
  }
}

export const structuredVelocityDynamicsWGSL = /* wgsl */ `
struct P{rowCapacity:u32,slotCapacity:u32,maxSlots:u32,authorityWords:u32,worksetStride:u32,worksetBankStride:u32,dimensionX:u32,dimensionY:u32,dimensionZ:u32,closedMask:u32,denseOffset:u32,slotGeometryOffset:u32,tetraHeaderOffset:u32,tetraVertexOffset:u32,tetraOffset:u32,tetraVertexCount:u32,templateHeaderOffset:u32,reconstructionOffset:u32,valuesOffset:u32,ownerOffset:u32,neighborOffset:u32,metadataOffset:u32,areaOffset:u32,inverseOffset:u32,fractionOffset:u32,pressureScaleOffset:u32,normalOffset:u32,centroidOffset:u32,rowNeighborOffset:u32,rowReciprocalOffset:u32,rowOwnerMetadataOffset:u32,rowHandleOffset:u32,rowSignOffset:u32,rowCatalogOffset:u32,rowAxisOffset:u32,rowFamilyPrefixOffset:u32,rowFamilyHandleOffset:u32,rowFamilySlotOffset:u32,bodyCount:u32,padB:u32,physical:vec4f,gravity:vec4f,selectorOffsetWords:u32,selectorStride:u32,regularTagOffsetWords:u32,supportControlOffsetWords:u32,supportVectorOffsetWords:u32,supportCapacity:u32,ownerDirectoryOffsetWords:u32,ownerDirectoryCellCapacity:u32,inflowPositionRadius:vec4f,inflowVelocity:vec4f}
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
// The active authority bank, resolved once per invocation.
//
// The ONLY writes to the accepted control anywhere in this module are
// rejectSample and rejectVector, which target indices 0, 1 and 6..10 -- see the
// two functions directly above. Indices 2 (pressure-row count), 3 (generation),
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
fn bitsf(at:u32)->f32{return bitcast<f32>(catalog[at]);}
fn dimensions()->vec3u{return vec3u(p.dimensionX,p.dimensionY,p.dimensionZ);}
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
fn candidateTransferItem(index:u32)->u32{
 if(arrayLength(&candidate)<7u||atomicLoad(&candidate[0])!=CANDIDATE_VALID){return INVALID;}
 let at=${OCTREE_STRUCTURED_GPU_TRANSFER_LIST_OFFSET_WORDS}u+index;
 if(atomicLoad(&candidate[4])==0u||at>=arrayLength(&candidate)){return INVALID;}
 return atomicLoad(&candidate[at]);
}

@compute @workgroup_size(1)
fn prepareStructuredDynamics(){
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
// Section 5 extrapolates the projected velocity outside the liquid before
// level-set transport. A newly wet pressure row therefore initializes from
// that accepted extended field when no old liquid row contains its face. The
// dense finest-cell directory names the exact octree owner and its published
// row/support vector; this is the same authority consumed by fine transport.
fn extendedOwnerVelocity(point:vec3f)->vec4f{
  if(!finite3(point)||!supportPublicationValid()){return invalidVector();}
  let d=dimensions();let volume=d.x*d.y*d.z;
  if(volume==0u||p.ownerDirectoryCellCapacity<volume){return invalidVector();}
  let upper=max(vec3f(d)-vec3f(1e-4),vec3f(0.));
  let q=vec3u(floor(clamp(point/p.physical.x,vec3f(0.),upper)));
  let cell=q.x+d.x*(q.y+d.y*q.z);let at=p.ownerDirectoryOffsetWords+4u*cell;
  if(at>4u*arrayLength(&transportMetrics)||4u*arrayLength(&transportMetrics)-at<4u){return invalidVector();}
  let tag=supportWord(at);let origin=supportWord(at+1u);let size=supportWord(at+2u);
  if(tag==INVALID||origin==INVALID||size==0u){return invalidVector();}
  let oq=vec3u(origin%d.x,(origin/d.x)%d.y,origin/(d.x*d.y));
  if(any(q<oq)||any(q>=oq+vec3u(size))){return invalidVector();}
  return taggedVelocity(tag);
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
// One axis-normal face plane of the staggered stencil. The face's published
// degree of freedom is authoritative whenever an incident cell is a live
// same-size regular row; an incident Section 5 support cell contributes its
// published extended vector component instead. An in-domain cell that
// resolves to a transition row, a different size, or no publication makes
// the whole sample ineligible (status 0) so the caller falls back to the
// paper's cube/tetra interpolant rather than substituting zero or an average.
fn staggeredPlaneValue(anchor:u32,rg:vec4u,origin:vec3f,h:f32,axis:u32,plane:i32,transverse:vec3i)->vec2f{
  var faceFound=false;var face=0.;
  var supportFound=false;var support=0.;
  let d=dimensions();
  for(var candidate=0u;candidate<2u;candidate+=1u){
    let cellAt=plane-i32(candidate);
    if(cellAt<-1||cellAt>1){continue;}
    let centerAt=origin[axis]+(f32(cellAt)+.5)*h;
    if(centerAt<0.||centerAt>f32(d[axis])*p.physical.x){continue;}
    var offset=transverse;offset[axis]=cellAt;
    let tag=regularTag(anchor,offset);
    if(tag==INVALID){return vec2f(0.,0.);}
    if((tag&SUPPORT_TAG)!=0u){
      let extended=taggedVelocity(tag);
      if(!vectorValid(extended)){return vec2f(0.,0.);}
      if(!supportFound){support=extended[axis];supportFound=true;}
      continue;
    }
    // A same-size live neighbour qualifies regardless of caseId — its face
    // handle is validated axis-normal and finite below, which is the real
    // requirement. A genuinely irregular neighbour fails that check instead.
    if(tag>=acc(2u)||rowGeometry[rbase()+tag].y!=rg.y){return vec2f(0.,0.);}
    // A dry live row's face degrees of freedom receive no gravity, no
    // projection update, and are never rewritten by the Section 5 march, so
    // they lag the liquid by a full substep at exactly the free surface.
    // Contribute the row's committed Section 5 extended vector instead, the
    // same closest-interface copy a support cell supplies; a wet incident
    // cell on the same plane still wins with its exact face value below.
    if(liquidAt(tag)==0u){
      let extended=taggedVelocity(tag);
      if(!vectorValid(extended)){return vec2f(0.,0.);}
      if(!supportFound){support=extended[axis];supportFound=true;}
      continue;
    }
    let resolved=faceAxisValue(regularFaceHandle(tag,axis,candidate==1u),axis);
    if(resolved.y==0.){return vec2f(0.,0.);}
    if(!faceFound){face=resolved.x;faceFound=true;}
  }
  if(faceFound){return vec2f(face,1.);}
  if(supportFound){return vec2f(support,1.);}
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
  let along=(sample[axis]-origin[axis])/h;
  let plane=clamp(i32(floor(along)),-1,1);
  var tAlong=clamp(along-f32(plane),0.,1.);
  // Measure-zero snap: a face centre must keep weight one on its own value
  // under floating-point dust, mirroring the old-mesh epsilon discipline at
  // interpolation-element boundaries.
  if(tAlong<1e-5){tAlong=0.;}else if(tAlong>1.-1e-5){tAlong=1.;}
  let center=origin+vec3f(.5*h);
  var low=vec3i(0);
  var tTransverse=vec3f(0.);
  for(var other=0u;other<3u;other+=1u){
    if(other==axis){continue;}
    if(sample[other]<center[other]){low[other]=-1;}
    let lowCenter=center[other]+f32(low[other])*h;
    var t=clamp((sample[other]-lowCenter)/h,0.,1.);
    if(t<1e-5){t=0.;}else if(t>1.-1e-5){t=1.;}
    tTransverse[other]=t;
  }
  var result=0.;
  for(var corner=0u;corner<8u;corner+=1u){
    var weight=select(1.-tAlong,tAlong,(corner&1u)!=0u);
    var offset=vec3i(0);
    var bit=1u;
    for(var other=0u;other<3u;other+=1u){
      if(other==axis){continue;}
      let high=(corner&(1u<<bit))!=0u;
      weight*=select(1.-tTransverse[other],tTransverse[other],high);
      offset[other]=low[other]+select(0,1,high);
      bit+=1u;
    }
    if(weight<=0.){continue;}
    for(var other=0u;other<3u;other+=1u){
      if(other==axis){continue;}
      let requested=center[other]+f32(offset[other])*h;
      if(requested<.5*h||requested>f32(d[other])*p.physical.x-.5*h){offset[other]=0;}
    }
    let resolved=staggeredPlaneValue(anchor,rg,origin,h,axis,plane+i32(corner&1u),offset);
    if(resolved.y==0.){return vec2f(0.,0.);}
    result+=weight*resolved.x;
  }
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
  let d=dimensions();let q=vec3u(rg.x%d.x,(rg.x/d.x)%d.y,rg.x/(d.x*d.y));
  let h=f32(rg.y)*p.physical.x;
  if(!finite(h)||h<=0.){return invalidVector();}
  // Cell-centred velocity has a constant physical-boundary extension. This
  // is the production basis at a domain wall: a missing live interior row
  // still rejects, while the basis never requests an exterior row.
  let sampleX=clamp(x,vec3f(.5*h),vec3f(d)*p.physical.x-vec3f(.5*h));
  let center=(vec3f(q)+.5*f32(rg.y))*p.physical.x;
  var lowOffset=vec3i(0);
  for(var axis=0u;axis<3u;axis+=1u){
    if(sampleX[axis]<center[axis]){lowOffset[axis]=-1;}
  }
  let lowCenter=center+vec3f(lowOffset)*h;
  let t=clamp((sampleX-lowCenter)/h,vec3f(0),vec3f(1));
  var result=vec3f(0.);
  for(var corner=0u;corner<8u;corner+=1u){
    let weight=select(1.-t.x,t.x,(corner&1u)!=0u)*select(1.-t.y,t.y,(corner&2u)!=0u)*select(1.-t.z,t.z,(corner&4u)!=0u);
    if(weight<=0.){continue;}
    var offset=lowOffset+vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let requestedCenter=center+vec3f(offset)*h;
    for(var axis=0u;axis<3u;axis+=1u){
      if(requestedCenter[axis]<.5*h||requestedCenter[axis]>f32(d[axis])*p.physical.x-.5*h){offset[axis]=0;}
    }
    let sample=taggedVelocity(regularTag(anchor,offset));
    if(!vectorValid(sample)){return invalidVector();}
    result+=weight*sample.xyz;
  }
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
fn tetraWeights(point:vec3f,x:vec3f,y:vec3f,z:vec3f)->vec4f{
  let d=dot(x,cross(y,z));
  if(!finite(d)||abs(d)<1e-10){return vec4f(-2.);}
  let a0=dot(point,cross(y,z))/d;
  let a1=dot(x,cross(point,z))/d;
  let a2=dot(x,cross(y,point))/d;
  return vec4f(1.-a0-a1-a2,a0,a1,a2);
}
fn transitionSample(row:u32,x:vec3f)->vec4f{
  if(row>=acc(2u)||!finite3(x)){return vec4f(255.,20.,f32(row),-1.);}
  let rg=rowGeometry[rbase()+row];
  let d=dimensions();let q=vec3u(rg.x%d.x,(rg.x/d.x)%d.y,rg.x/(d.x*d.y));
  let extent=f32(rg.y)*p.physical.x;
  if(!finite(extent)||extent<=0.){return vec4f(255.,21.,extent,-1.);}
  let center=(vec3f(q)+.5*f32(rg.y))*p.physical.x;
  let local=powerTransform((x-center)/extent,metrics[row].transformAndFlags&63u);
  let thAt=p.tetraHeaderOffset+3u*metrics[row].caseId;
  let first=catalog[thAt];
  let count=catalog[thAt+1u];
  for(var ti=0u;ti<count;ti+=1u){
    let packed=catalog[p.tetraOffset+first+ti];
    let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);
    if(any(selectors>=vec3u(p.tetraVertexCount))){return vec4f(f32(ti),22.,f32(p.tetraVertexCount),-1.);}
    let va=p.tetraVertexOffset+4u*selectors.x;
    let vb=p.tetraVertexOffset+4u*selectors.y;
    let vc=p.tetraVertexOffset+4u*selectors.z;
    let sa=vec4f(bitsf(va),bitsf(va+1u),bitsf(va+2u),bitsf(va+3u));
    let sb=vec4f(bitsf(vb),bitsf(vb+1u),bitsf(vb+2u),bitsf(vb+3u));
    let sc=vec4f(bitsf(vc),bitsf(vc+1u),bitsf(vc+2u),bitsf(vc+3u));
    let weights=tetraWeights(local,sa.xyz,sb.xyz,sc.xyz);
    if(all(weights>=vec4f(-2e-6))&&all(weights<=vec4f(1.000002))){
      // Barycentric vertices with zero weight do not contribute to the
      // paper's tetrahedral interpolant. Avoid requiring an extrapolated air
      // row for those vertices, but keep every positive contributor strict.
      var positive=max(weights,vec4f(0.));
      let positiveSum=dot(positive,vec4f(1.));
      if(!finite(positiveSum)||positiveSum<=0.){return vec4f(f32(ti),25.,positiveSum,-1.);}
      positive/=positiveSum;
      var result=vec3f(0.);
      if(positive.x>0.){let v0=velocitySample(row);if(!vectorValid(v0)){return vec4f(f32(ti),23.,f32(row),-1.);}result+=positive.x*v0.xyz;}
      if(positive.y>0.){let v1=selectorVelocity(row,selectors.x,sa);if(!vectorValid(v1)){return v1;}result+=positive.y*v1.xyz;}
      if(positive.z>0.){let v2=selectorVelocity(row,selectors.y,sb);if(!vectorValid(v2)){return v2;}result+=positive.z*v2.xyz;}
      if(positive.w>0.){let v3=selectorVelocity(row,selectors.z,sc);if(!vectorValid(v3)){return v3;}result+=positive.w*v3.xyz;}
      if(!finite3(result)){return invalidVector();}
      return vec4f(result,1.);
    }
  }
  return vec4f(255.,24.,f32(count),-1.);
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
fn oldFieldAt(anchor:u32,point:vec3f)->vec4f{
  if(anchor==INVALID){return invalidVector();}var sample=invalidVector();if(regularTag(anchor,vec3i(0))!=anchor){sample=transitionSample(anchor,point);}else{sample=regularSample(anchor,point);}
  // Interpolation can leave its local closure at a newly exposed face. The
  // already projected row vector is the bounded piecewise-constant fallback;
  // zero is never presented as a successful topology transfer.
  if(!vectorValid(sample)){sample=velocitySample(anchor);}return sample;
}
fn oldAnchor(candidateBank:u32,candidateRow:u32,n:vec3f)->u32{
  let center=candidateRowCenter(candidateBank,candidateRow);var row=acceptedRowContaining(center);if(row!=INVALID){return row;}
  // A newly wet candidate centre can lie just beyond the old liquid frontier.
  // Search back into its owner side by a bounded CFL support distance; the
  // accepted Section 5 closure then evaluates the new face point itself.
  for(var step=1u;step<=4u;step+=1u){row=acceptedRowContaining(center-f32(step)*.5*p.physical.x*n);if(row!=INVALID){return row;}}
  // The interface can enter a cell transverse to this particular face normal.
  // Section 5 guarantees a one-ring extended old field, so choose the nearest
  // accepted carrier in the bounded 5^3 neighbourhood and evaluate the new
  // centroid through that carrier's regular/tetrahedral closure.
  var nearest=INVALID;var nearestDistance=0x7fffffffi;
  for(var dz=-2i;dz<=2i;dz+=1i){for(var dy=-2i;dy<=2i;dy+=1i){for(var dx=-2i;dx<=2i;dx+=1i){
    let offset=vec3i(dx,dy,dz);if(all(offset==vec3i(0))){continue;}let candidateRow=acceptedRowContaining(center+vec3f(offset)*p.physical.x);if(candidateRow==INVALID){continue;}
    let distance=dx*dx+dy*dy+dz*dz;if(distance<nearestDistance||(distance==nearestDistance&&candidateRow<nearest)){nearest=candidateRow;nearestDistance=distance;}
  }}}
  return nearest;
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
// be carried. Its producer publishes the same two-dimensional indirect shape
// as every accepted class, so the folded item uses the common pinned stride.
@compute @workgroup_size(64)fn transferStructuredTopologyCandidate(@builtin(global_invocation_id)g:vec3u){
  let index=g.x+g.y*65535u*64u;let handle=candidateTransferItem(index);if(handle==INVALID||atomicLoad(&candidate[6])==0u||handle>=atomicLoad(&candidate[3])){return;}
  let candidateBank=atomicLoad(&candidate[5])&1u;let base=candidateBank*p.authorityWords;let marker=bitcast<f32>(a[base+p.centroidOffset+4u*handle+3u]);if(marker>0.5){return;}
  let at=base+p.centroidOffset+4u*handle;let point=vec3f(bitcast<f32>(a[at]),bitcast<f32>(a[at+1u]),bitcast<f32>(a[at+2u]));let nt=base+p.normalOffset+4u*handle;let n=vec3f(bitcast<f32>(a[nt]),bitcast<f32>(a[nt+1u]),bitcast<f32>(a[nt+2u]));
  if(!finite3(point)||!finite3(n)){_ = rejectCandidateTransfer(handle);return;}
  if(candidateClosedWorld(point,n)){a[base+p.valuesOffset+handle]=bitcast<u32>(0.);return;}
  let candidateOwner=a[base+p.ownerOffset+handle];let candidateNeighbor=a[base+p.neighborOffset+handle];
  // Paper Section 5 frontier discipline: a face whose owner centre lies
  // beyond the old liquid is newly wetted and takes the extrapolated field —
  // a COPY of the closest projected face value — never an interpolant
  // evaluated outside its element (which extends the frontier velocity
  // gradient one cell and ratchets the corner jet, measured at +45% ME).
  let ownerInsideOld=acceptedRowContaining(candidateRowCenter(candidateBank,candidateOwner))!=INVALID;
  var old=invalidVector();
  if(!ownerInsideOld){old=extendedOwnerVelocity(point);}
  var ownerAnchor=INVALID;var neighborAnchor=INVALID;
  if(!vectorValid(old)){
    ownerAnchor=oldAnchor(candidateBank,candidateOwner,n);let ownerField=oldFieldAt(ownerAnchor,point);var neighborField=invalidVector();
    if(candidateNeighbor!=INVALID){neighborAnchor=oldAnchor(candidateBank,candidateNeighbor,-n);if(neighborAnchor!=ownerAnchor){neighborField=oldFieldAt(neighborAnchor,point);}}
    // Resolve the two-sided ambiguity the way main's sampleOldIncident did:
    // one interpolated sample through the incident owner-side element, never a
    // mean of two independent closures — averaging both sides low-passes every
    // newly created interface face on every topology change.
    old=ownerField;if(!vectorValid(ownerField)){old=neighborField;}
    if(!vectorValid(old)){old=extendedOwnerVelocity(point);}
  }
  if(!vectorValid(old)){if(rejectCandidateTransfer(handle)&&arrayLength(&candidate)>=16u){atomicStore(&candidate[9],candidateOwner);atomicStore(&candidate[10],candidateNeighbor);atomicStore(&candidate[11],ownerAnchor);atomicStore(&candidate[12],neighborAnchor);atomicStore(&candidate[13],bitcast<u32>(point.x));atomicStore(&candidate[14],bitcast<u32>(point.y));atomicStore(&candidate[15],bitcast<u32>(point.z));}return;}let projected=dot(old.xyz,n);if(!finite(projected)){_ = rejectCandidateTransfer(handle);return;}a[base+p.valuesOffset+handle]=bitcast<u32>(projected);
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
// A row is "dynamic" when it is dry or borders air: exactly the band whose
// power-cell geometry main's identity key would have invalidated, forcing a
// re-trace. The accepted row set holds ONLY liquid rows, so "borders air"
// means an in-domain six-neighbour probe that resolves to no accepted row
// (or, defensively, to a non-liquid one). A probe outside the closed domain
// is a wall, not air; a deep column against a wall still carries.
fn rowTouchesDry(row:u32)->bool{
  if(row>=acc(2u)||liquidAt(row)==0u){return true;}
  let rg=rowGeometry[rbase()+row];
  let h=f32(rg.y)*p.physical.x;
  let d=dimensions();
  let q=vec3u(rg.x%d.x,(rg.x/d.x)%d.y,rg.x/(d.x*d.y));
  let center=(vec3f(q)+.5*f32(rg.y))*p.physical.x;
  let extent=vec3f(d)*p.physical.x;
  for(var direction=0u;direction<6u;direction+=1u){
    var probe=center;
    probe[direction/2u]+=select(-h,h,(direction&1u)==1u);
    if(any(probe<vec3f(0.))||any(probe>=extent)){continue;}
    let other=acceptedRowContaining(probe);
    if(other==INVALID||other>=acc(2u)||liquidAt(other)==0u){return true;}
  }
  return false;
}
fn advect(cls:u32,index:u32){
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
  // Follow the exact owner-local closure committed by the Section 5 producer.
  // A face class is transition when either incident row is nonuniform, so it
  // cannot choose this owner's basis. Conversely, a nominal case-zero owner
  // with a body-diagonal coarse neighbour has no cube tags and uses the
  // retained case-zero Delaunay fan. The centre cube tag is the producer's
  // unambiguous marker: every complete regular closure publishes the row id.
  let centerTag=regularTag(row,vec3i(0));
  let useTransition=centerTag!=row;
  let x=centroid(handle);
  let prior=value(handle);
  setNextValue(handle,prior);
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
  let carriedMarker=bitcast<f32>(a[abase()+p.centroidOffset+4u*handle+3u]);
  // The accepted row set is liquid-only, so a face with no neighbour row is a
  // free-surface (or wall) face: never carried. Closed-wall faces already
  // returned through the aperture==0 branch above.
  let hiRow=neighbor(handle);
  if(carriedMarker>.5&&hiRow!=INVALID&&!rowTouchesDry(row)&&!rowTouchesDry(hiRow)){
    let n=normal(handle);
    let area=bitcast<f32>(a[abase()+p.areaOffset+handle]);
    if(!finite3(n)||!finite(area)||area<=0.||!finite(prior)){transportMetrics[handle]=invalidVector();rejectSample(3u,handle);return;}
    transportMetrics[handle]=vec4f(prior*n*area,0.);
    return;
  }
  var adv=invalidVector();
  if(useTransition){adv=transitionSample(row,x);}else{adv=regularSample(row,x);}
  if(!vectorValid(adv)){transportMetrics[handle]=adv;rejectVector(1u,handle,adv,cls);return;}
  // Second-order midpoint backtrace, as main's old-mesh advection performed
  // (v0 at the face, vm at the half-step, departure from vm). A first-order
  // Euler trace dissipates O(dt^2*|v|*|grad v|) per step, which measured as
  // mechanical-energy retention decaying to 0.83 by t=0.24 s on the mini dam
  // while the midpoint scheme holds ~0.99.
  let midpoint=x-.5*p.physical.y*adv.xyz;
  let middle=characteristicSample(row,midpoint);
  if(!vectorValid(middle)){transportMetrics[handle]=middle;rejectVector(1u,handle,middle,cls);return;}
  let departure=x-p.physical.y*middle.xyz;
  let transported=characteristicSample(row,departure);
  if(!vectorValid(transported)){transportMetrics[handle]=transported;rejectVector(2u,handle,transported,cls);return;}
  let n=normal(handle);
  let projected=dot(transported.xyz,n);
  let area=bitcast<f32>(a[abase()+p.areaOffset+handle]);
  if(!finite(projected)||!finite(area)||area<=0.||!finite(prior)){transportMetrics[handle]=invalidVector();rejectSample(3u,handle);return;}
  setNextValue(handle,projected);
  transportMetrics[handle]=vec4f(projected*n*area,.5*area*max(0.,prior*prior-projected*projected));
}
@compute @workgroup_size(64)fn advectStructuredClass5(@builtin(global_invocation_id)g:vec3u){advect(5u,classItem(g));}
@compute @workgroup_size(64)fn advectStructuredClass6(@builtin(global_invocation_id)g:vec3u){advect(6u,classItem(g));}
@compute @workgroup_size(64)fn advectStructuredClass7(@builtin(global_invocation_id)g:vec3u){advect(7u,classItem(g));}
@compute @workgroup_size(64)fn advectStructuredClass8(@builtin(global_invocation_id)g:vec3u){advect(8u,classItem(g));}

fn commitAdvected(cls:u32,index:u32){
  let handle=workItem(cls,index);
  if(handle==INVALID||handle>=acc(5u)){return;}
  // Mirror the advect gate exactly: a lane that staged nothing must not copy
  // stale inactive-bank words over the accepted values.
  if(!supportPublicationValid()){return;}
  a[abase()+p.valuesOffset+handle]=a[nextValueAt(handle)];
}
@compute @workgroup_size(64)fn commitAdvectedStructuredClass5(@builtin(global_invocation_id)g:vec3u){commitAdvected(5u,classItem(g));}
@compute @workgroup_size(64)fn commitAdvectedStructuredClass6(@builtin(global_invocation_id)g:vec3u){commitAdvected(6u,classItem(g));}
@compute @workgroup_size(64)fn commitAdvectedStructuredClass7(@builtin(global_invocation_id)g:vec3u){commitAdvected(7u,classItem(g));}
@compute @workgroup_size(64)fn commitAdvectedStructuredClass8(@builtin(global_invocation_id)g:vec3u){commitAdvected(8u,classItem(g));}

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
  let forced=value(handle)+p.physical.y*dot(p.gravity.xyz,normal(handle));
  if(!finite(forced)){rejectSample(11u,handle);return;}
  setValue(handle,forced);
}
@compute @workgroup_size(64)fn forceStructuredClass5(@builtin(global_invocation_id)g:vec3u){forceFamily(5u,classItem(g));}
@compute @workgroup_size(64)fn forceStructuredClass6(@builtin(global_invocation_id)g:vec3u){forceFamily(6u,classItem(g));}
@compute @workgroup_size(64)fn forceStructuredClass7(@builtin(global_invocation_id)g:vec3u){forceFamily(7u,classItem(g));}
@compute @workgroup_size(64)fn forceStructuredClass8(@builtin(global_invocation_id)g:vec3u){forceFamily(8u,classItem(g));}

fn divergenceRow(cls:u32,index:u32){
  let row=workItem(cls,index);
  if(row==INVALID||row>=acc(2u)){return;}
  if(liquidAt(row)==0u){rhs[row]=0.;return;}
  var flux=0.;
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
    flux+=sign*area*boundaryVelocity;
  }
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
@compute @workgroup_size(64)fn divergenceStructuredClass0(@builtin(global_invocation_id)g:vec3u){divergenceRow(0u,classItem(g));}
@compute @workgroup_size(64)fn divergenceStructuredClass1(@builtin(global_invocation_id)g:vec3u){divergenceRow(1u,classItem(g));}
@compute @workgroup_size(64)fn divergenceStructuredClass2(@builtin(global_invocation_id)g:vec3u){divergenceRow(2u,classItem(g));}
@compute @workgroup_size(64)fn divergenceStructuredClass3(@builtin(global_invocation_id)g:vec3u){divergenceRow(3u,classItem(g));}

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
    // Complementarity hysteresis. Strict tension (p < 0) opens a face; an
    // already open face renews while the solved contact pressure stays below
    // a quarter of the one-cell hydrostatic contact scale rho g h, and only
    // genuine positive contact pressure re-closes it. Without renewal this
    // mark was an unconditional per-epoch overwrite: a separated film solves
    // p ~ 0, loses its mark the very next epoch, re-welds to the lid through
    // the closed face's divergence row, and falls at an open/closed duty
    // cycle instead of g (the free-fall drop oracles measure the top liquid
    // layer saturating near 0.9 m/s under a 2.8 m/s parabola).
    let previous=separationMask[cell];
    let previousBits=previous&63u;
    let age=(acc(3u)-(previous>>6u))&0x3ffffffu;
    let contact=.25*p.physical.z*weight*p.physical.x;
    let opening=finite(solved)&&solved<0.;
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
@compute @workgroup_size(64)fn separateStructuredClass0(@builtin(global_invocation_id)g:vec3u){markSeparationRow(0u,classItem(g));}
@compute @workgroup_size(64)fn separateStructuredClass1(@builtin(global_invocation_id)g:vec3u){markSeparationRow(1u,classItem(g));}
@compute @workgroup_size(64)fn separateStructuredClass2(@builtin(global_invocation_id)g:vec3u){markSeparationRow(2u,classItem(g));}
@compute @workgroup_size(64)fn separateStructuredClass3(@builtin(global_invocation_id)g:vec3u){markSeparationRow(3u,classItem(g));}

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
@compute @workgroup_size(64)fn exchangeStructuredBodyImpulseClass0(@builtin(global_invocation_id)g:vec3u){bodyImpulseRow(0u,classItem(g));}
@compute @workgroup_size(64)fn exchangeStructuredBodyImpulseClass1(@builtin(global_invocation_id)g:vec3u){bodyImpulseRow(1u,classItem(g));}
@compute @workgroup_size(64)fn exchangeStructuredBodyImpulseClass2(@builtin(global_invocation_id)g:vec3u){bodyImpulseRow(2u,classItem(g));}
@compute @workgroup_size(64)fn exchangeStructuredBodyImpulseClass3(@builtin(global_invocation_id)g:vec3u){bodyImpulseRow(3u,classItem(g));}

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
@compute @workgroup_size(64)fn projectStructuredClass5(@builtin(global_invocation_id)g:vec3u){projectFamily(5u,classItem(g));}
@compute @workgroup_size(64)fn projectStructuredClass6(@builtin(global_invocation_id)g:vec3u){projectFamily(6u,classItem(g));}
@compute @workgroup_size(64)fn projectStructuredClass7(@builtin(global_invocation_id)g:vec3u){projectFamily(7u,classItem(g));}
@compute @workgroup_size(64)fn projectStructuredClass8(@builtin(global_invocation_id)g:vec3u){projectFamily(8u,classItem(g));}

fn reconstructRow(cls:u32,index:u32){
  let row=workItem(cls,index);
  if(row==INVALID||row>=acc(2u)){return;}
  let header=caseHeader(metrics[row].caseId);
  if(header.y>p.maxSlots){rejectSample(50u,row);return;}
  var canonical=vec3f(0.);
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
    canonical+=coefficient*sample;
  }
  let projected=inverseTransform(canonical,metrics[row].transformAndFlags&63u);
  if(!finite3(projected)){rejectSample(54u,row);return;}
  rowVelocity[rbase()+row]=vec4f(projected,1.);
}
@compute @workgroup_size(64)fn reconstructStructuredClass0(@builtin(global_invocation_id)g:vec3u){reconstructRow(0u,classItem(g));}
@compute @workgroup_size(64)fn reconstructStructuredClass1(@builtin(global_invocation_id)g:vec3u){reconstructRow(1u,classItem(g));}
@compute @workgroup_size(64)fn reconstructStructuredClass2(@builtin(global_invocation_id)g:vec3u){reconstructRow(2u,classItem(g));}
@compute @workgroup_size(64)fn reconstructStructuredClass3(@builtin(global_invocation_id)g:vec3u){reconstructRow(3u,classItem(g));}
`;
