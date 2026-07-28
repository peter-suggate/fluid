/** Transactional dynamic free-surface/solid coefficients for structured families. */

import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import {
  makeOctreePowerCoarseLevelSetSampleWGSL,
  type OctreePowerCoarseLevelSetSampleSource,
} from "./webgpu-octree-power-coarse-levelset";
import type { OctreeSolidVertexSdfSource } from "./webgpu-octree-solid-vertex-sdf";
import {
  OCTREE_STRUCTURED_GPU_VALID,
  type DirectStructuredVelocitySource,
} from "./webgpu-octree-structured-velocity-gpu";
import type { PassBroker } from "./webgpu-pass-broker";

type StructuredControlAuthority = "candidate-ready" | "accepted";
interface StructuredBoundaryGroupCache {
  readonly candidateReady?: readonly GPUBindGroup[];
  readonly accepted?: readonly GPUBindGroup[];
}

export interface StructuredBoundaryResources {
  readonly structured: DirectStructuredVelocitySource;
  /** One u32 per finest cell: (generation << 1) | ceiling-tension flag,
   * written by the dynamics projection stage. A fresh flag opens the row's
   * closed world faces this rebuild (unilateral ceiling contact). */
  readonly separationMask: GPUBuffer;
  /** Identity-keyed coarse phi from the accepted pre-adaptation topology.
   * Candidate rows may be inserted or reordered, so row-indexed values are
   * not a valid source for the inactive structured generation. */
  readonly coarse: Pick<OctreePowerCoarseLevelSetSampleSource, "directory" | "rowCapacity">;
  readonly solid?: OctreeSolidVertexSdfSource;
  readonly rigidBodies: GPUBuffer;
  readonly bodyCount: number;
  readonly dimensions: readonly [number, number, number];
  readonly physicalCellSize: number;
  readonly closedBoundaryMask: number;
  /** Exact authored SDF used only until compact coarse phi publishes. */
  readonly analyticBootstrap?: Readonly<{
    initialCondition: "dam-break" | "tank-fill";
    fillFraction: number;
  }>;
  /**
   * The imported dense t=0 level set, used by scenes whose surface has no
   * closed form (terrain, rigid bodies, explicitly seeded bricks). It plays
   * exactly the same one-shot role as `analyticBootstrap` and retires through
   * the same hook; without either, coarse phi has no authority at t=0 and the
   * boundary coefficients fail closed on positivity.
   */
  readonly bootstrapLevelSet?: GPUTexture;
}

export class WebGPUStructuredBoundaryCoefficients {
  /** Accepted row liquid mask; one u32 per row. */
  readonly liquidMask: GPUBuffer;
  /** A/B family-handle normal velocity of the closest intersecting rigid. */
  readonly solidNormalVelocities: GPUBuffer;
  readonly control: GPUBuffer;
  readonly candidateControl: GPUBuffer;
  /** Packed A/B current-boundary row/family worksets, matching the structured bank. */
  readonly worksets: GPUBuffer;
  readonly worksetStrideWords: number;
  readonly worksetBankStrideWords: number;
  readonly encodedDispatchCount = 11;
  readonly allocatedBytes: number;
  private readonly worksetClasses: GPUBuffer;
  private readonly worksetBlocks: GPUBuffer;
  private readonly candidateMask: GPUBuffer;
  private readonly candidates: GPUBuffer;
  private readonly candidateSolidNormalVelocities: GPUBuffer;
  private readonly dispatch: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly inertSolidStorage?: GPUBuffer;
  private readonly prepareCandidate: GPUComputePipeline;
  private readonly prepareAccepted: GPUComputePipeline;
  private readonly classify: GPUComputePipeline;
  private readonly resolve: GPUComputePipeline;
  private readonly resolveSolid: GPUComputePipeline;
  private readonly commit: GPUComputePipeline;
  private readonly rebuild: GPUComputePipeline;
  private readonly countRowClasses: GPUComputePipeline;
  private readonly countFamilyClasses: GPUComputePipeline;
  private readonly scanWorksetBlocks: GPUComputePipeline;
  private readonly scatterRowWorksets: GPUComputePipeline;
  private readonly scatterFamilyWorksets: GPUComputePipeline;
  private readonly accept: GPUComputePipeline;
  private readonly acceptGroup: GPUBindGroup;
  private readonly groupsByFineParams = new WeakMap<GPUBuffer,
    WeakMap<GPUBuffer, StructuredBoundaryGroupCache>>();
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly resources: StructuredBoundaryResources) {
    const { structured } = resources;
    if (!(resources.physicalCellSize > 0) || !Number.isFinite(resources.physicalCellSize)
      || resources.dimensions.some((value) => !Number.isSafeInteger(value) || value < 1)
      || resources.separationMask.size < resources.dimensions[0]! * resources.dimensions[1]! * resources.dimensions[2]! * 4
      || resources.coarse.rowCapacity !== structured.plan.rowCapacity
      || resources.coarse.directory.size < 32 + structured.plan.rowCapacity * 32
      || !Number.isSafeInteger(resources.bodyCount) || resources.bodyCount < 0 || resources.bodyCount > 12
      || resources.rigidBodies.size < 12 * 8 * 16
      || resources.analyticBootstrap
        && (!Number.isFinite(resources.analyticBootstrap.fillFraction)
          || resources.analyticBootstrap.fillFraction < 0
          || resources.analyticBootstrap.fillFraction > 1)
      || resources.solid && resources.solid.plan.rowCapacity !== structured.plan.rowCapacity) {
      throw new RangeError("Structured boundary resources are invalid or undersized");
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.liquidMask = device.createBuffer({ label: "Accepted structured liquid row mask",
      size: 2 * structured.plan.rowCapacity * 4, usage: storage });
    this.candidateMask = device.createBuffer({ label: "Candidate structured liquid row mask",
      size: structured.plan.rowCapacity * 4, usage: storage });
    this.candidates = device.createBuffer({ label: "Candidate structured aperture and pressure scale",
      size: structured.plan.slotCapacity * 8, usage: storage });
    this.solidNormalVelocities = device.createBuffer({ label: "A/B structured solid normal velocities",
      size: 2 * structured.plan.slotCapacity * 4, usage: storage });
    this.candidateSolidNormalVelocities = device.createBuffer({ label: "Candidate structured solid normal velocities",
      size: structured.plan.slotCapacity * 4, usage: storage });
    this.control = device.createBuffer({ label: "Accepted structured boundary coefficient publication", size: 64, usage: storage });
    this.candidateControl = device.createBuffer({ label: "Candidate structured boundary coefficient publication", size: 64, usage: storage });
    this.dispatch = device.createBuffer({ label: "Structured boundary live dispatch", size: 24,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    this.worksetStrideWords = structured.plan.worksetStrideWords;
    this.worksetBankStrideWords = structured.worksetBankStrideWords;
    this.worksets = device.createBuffer({ label: "Packed A/B dynamic structured boundary worksets",
      size: 2 * this.worksetBankStrideWords * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC });
    // Workset compaction scratch. Both are pure per-advance intermediates:
    // the class of every row then every slot, and 9 classes of per-block
    // prefix bases over whichever of the two ranges that class draws from.
    this.worksetClasses = device.createBuffer({ label: "Structured boundary workset element classes",
      size: (structured.plan.rowCapacity + structured.plan.slotCapacity) * 4, usage: storage });
    this.worksetBlocks = device.createBuffer({ label: "Structured boundary workset block prefixes",
      size: 9 * Math.ceil(Math.max(structured.plan.rowCapacity, structured.plan.slotCapacity) / 64) * 4,
      usage: storage });
    this.params = device.createBuffer({ label: "Structured boundary parameters", size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    if (!resources.solid) {
      this.inertSolidStorage = device.createBuffer({ label: "Inert structured solid SDF binding", size: 80,
        usage: storage });
    }
    const words = new Uint32Array(32), floats = new Float32Array(words.buffer);
    words.set([structured.plan.rowCapacity, structured.plan.slotCapacity,
      structured.plan.maximumCaseSlots, structured.plan.authorityWords,
      19, structured.section63.coefficientBankStrideWords,
      resources.closedBoundaryMask >>> 0, resources.solid ? 1 : 0,
      ...resources.dimensions, resources.bodyCount], 0);
    floats[12] = resources.physicalCellSize;
    const o = structured.plan.offsets;
    words.set([o.ownerRows, o.neighborRows, o.areas, o.inverseDistances,
      o.fractions, o.pressureScales, o.normals, o.centroids,
      o.rowNeighbors, o.rowSlotHandles, o.rowSlotSigns, o.rowCatalogSlots], 16);
    words.set([this.worksetStrideWords, this.worksetBankStrideWords,
      resources.analyticBootstrap?.initialCondition === "tank-fill" ? 1
        : resources.analyticBootstrap?.initialCondition === "dam-break" ? 2
          : resources.bootstrapLevelSet ? 3 : 0], 28);
    floats[31] = resources.analyticBootstrap?.fillFraction ?? 0;
    device.queue.writeBuffer(this.params, 0, words.buffer);
    const module = device.createShaderModule({ label: "Structured boundary coefficients",
      code: structuredBoundaryCoefficientWGSL });
    const make = (entryPoint: string) => device.createComputePipeline({ label: entryPoint,
      layout: "auto", compute: { module, entryPoint } });
    this.prepareCandidate = make("prepareStructuredBoundaryCandidate");
    this.prepareAccepted = make("prepareStructuredBoundaryAccepted");
    this.classify = make("classifyStructuredLiquidRows");
    this.resolve = make("resolveStructuredBoundarySlots");
    this.resolveSolid = make("resolveStructuredSolidSlots");
    this.commit = make("commitStructuredBoundarySlots");
    this.rebuild = make("rebuildStructuredBoundaryRows");
    this.countRowClasses = make("countStructuredRowClasses");
    this.countFamilyClasses = make("countStructuredFamilyClasses");
    this.scanWorksetBlocks = make("scanStructuredWorksetBlocks");
    this.scatterRowWorksets = make("scatterStructuredRowWorksets");
    this.scatterFamilyWorksets = make("scatterStructuredFamilyWorksets");
    this.accept = make("acceptStructuredBoundary");
    this.acceptGroup = device.createBindGroup({ layout: this.accept.getBindGroupLayout(0), entries: [
      { binding: 16, resource: { buffer: this.candidateControl } },
      { binding: 22, resource: { buffer: this.control } },
    ] });
    this.allocatedBytes = this.liquidMask.size + this.solidNormalVelocities.size
      + this.candidateMask.size + this.candidates.size + this.candidateSolidNormalVelocities.size
      + this.control.size + this.candidateControl.size + this.dispatch.size + this.params.size + this.worksets.size
      + this.worksetClasses.size + this.worksetBlocks.size
      + (this.inertSolidStorage?.size ?? 0);
  }

  private groups(fine: Pick<WebGPUFineLevelSetBrickSource,
    "params" | "metadata" | "worklist" | "flags" | "phi">,
    structuredControl: GPUBuffer, authority: StructuredControlAuthority): readonly GPUBindGroup[] {
    let byControl = this.groupsByFineParams.get(fine.params);
    if (!byControl) { byControl = new WeakMap(); this.groupsByFineParams.set(fine.params, byControl); }
    const cached = byControl.get(structuredControl);
    const cachedGroups = authority === "candidate-ready" ? cached?.candidateReady : cached?.accepted;
    if (cachedGroups) return cachedGroups;
    const { structured, solid } = this.resources;
    const section63 = structured.section63;
    const fparams = fine.params;
    const inertSolid = this.inertSolidStorage;
    const buffers: Record<number, GPUBuffer> = { 0: this.params, 1: structuredControl,
      2: structured.authority, 3: structured.rowGeometry, 4: this.resources.coarse.directory,
      5: fparams, 6: fine.worklist, 7: fine.metadata,
      8: fine.flags, 9: fine.phi, 10: solid?.arena ?? inertSolid!,
      11: this.candidates, 12: this.candidateMask, 13: this.liquidMask,
      14: section63.coefficients, 16: this.candidateControl, 17: this.dispatch,
      18: this.worksets, 19: this.resources.rigidBodies, 20: this.solidNormalVelocities,
      21: this.candidateSolidNormalVelocities, 22: this.control,
      23: section63.topologyMetrics, 24: section63.catalogFaces,
      25: this.resources.separationMask,
      27: this.worksetClasses, 28: this.worksetBlocks };
    // Binding 26 is the only texture in this layout; every other entry is a
    // buffer from the map above.
    // `coarseValue` references the texture unconditionally, so the auto layout
    // always contains binding 26 regardless of which bootstrap mode is live.
    if (!this.resources.bootstrapLevelSet) {
      throw new Error("Structured boundary coefficients require a bootstrap level-set binding");
    }
    const bootstrapView = this.resources.bootstrapLevelSet.createView({ dimension: "3d" });
    const group = (pipeline: GPUComputePipeline, bindings: readonly number[]) => this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries: bindings.map((binding) => ({ binding,
        resource: binding === 26 ? bootstrapView! : { buffer: buffers[binding]! } })),
    });
    const prepare = authority === "candidate-ready" ? this.prepareCandidate : this.prepareAccepted;
    const groups = Object.freeze([
      group(prepare, [0, 1, 16, 17]),
      group(this.classify, [0, 3, 4, 5, 6, 7, 8, 9, 12, 16, 26]),
      group(this.resolve, [0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16, 25, 26]),
      group(this.resolveSolid, [0, 2, 3, 10, 11, 16, 19, 21]),
      group(this.commit, [0, 2, 11, 16, 20, 21]),
      group(this.rebuild, [0, 2, 12, 13, 14, 16, 24]),
      group(this.countRowClasses, [0, 3, 14, 16, 27, 28]),
      group(this.countFamilyClasses, [0, 2, 3, 13, 14, 16, 27, 28]),
      group(this.scanWorksetBlocks, [0, 16, 18, 28]),
      group(this.scatterRowWorksets, [0, 16, 18, 27, 28]),
      group(this.scatterFamilyWorksets, [0, 16, 18, 27, 28]),
    ]);
    byControl.set(structuredControl, authority === "candidate-ready"
      ? { ...cached, candidateReady: groups }
      : { ...cached, accepted: groups });
    return groups;
  }

  private encodeFromStructuredControl(broker: PassBroker, fine: Pick<WebGPUFineLevelSetBrickSource,
    "params" | "metadata" | "worklist" | "flags" | "phi">,
    structuredControl: GPUBuffer, authority: StructuredControlAuthority): void {
    if (this.destroyed) throw new Error("Structured boundary coefficients are destroyed");
    const groups = this.groups(fine, structuredControl, authority);
    const prepare = authority === "candidate-ready" ? this.prepareCandidate : this.prepareAccepted;
    const stages = [prepare, this.classify, this.resolve, this.resolveSolid, this.commit,
      this.rebuild] as const;
    const prepareLabel = authority === "candidate-ready"
      ? "Prepare structured boundary candidate-ready transaction"
      : "Prepare structured boundary accepted recurring transaction";
    const labels = [prepareLabel, "Classify fine-over-coarse liquid rows",
      "Resolve canonical free-surface boundary slots", "Resolve canonical solid boundary slots",
      "Commit canonical structured boundary slots",
      "Rebuild symmetric structured boundary rows"] as const;
    stages.forEach((pipeline, index) => {
      const pass = broker.compute({ label: labels[index] }); pass.setPipeline(pipeline); pass.setBindGroup(0, groups[index]!);
      if (index === 0) pass.dispatchWorkgroups(1);
      else pass.dispatchWorkgroupsIndirect(this.dispatch, index >= 2 && index <= 4 ? 12 : 0);
      // NOTE: stages 1..5 exchange only plain storage, which WebGPU already
      // orders inside a pass, so in principle only the prepare (which writes
      // `this.dispatch` as storage for later indirect reads) needs a boundary.
      // Collapsing them measured under 1 ms on a 96 ms frame and coincided with
      // an authoritative resolved-row publication being rejected in-app, so the
      // fences stay until that is understood. Launch structure is not this
      // frame's cost -- see docs/OCTREE_PHASE0_BASELINE.md.
      broker.fence(labels[index]);
    });
    // Workset publication: count -> block scan -> scatter, all five dispatches
    // in the one pass the serial kernel used to occupy. No fence is needed
    // between them -- they hand off only plain storage (`worksetClasses`,
    // `worksetBlocks`, `dynamicWorksets`), which WebGPU orders within a pass,
    // and nothing here reads a buffer this pass writes as an indirect
    // argument. `this.dispatch` is bound INDIRECT-only for the whole pass.
    // `FLUID_STRUCTURED_WORKSET_SPLIT=1` fences each stage into its own pass.
    // SUPERSEDED by `FLUID_GPU_ISOLATE_PASS_LABELS=1` (`lib/webgpu-pass-broker.ts`),
    // which does this for every labelled stage in the frame; keep this only if
    // you want the split without the global flag.
    //
    // This group is where the instrument's two bugs were caught, so the history
    // is worth keeping: these five dispatches once reported 3.604 ms on
    // whichever one was LAST, and swapping the final two moved that identical
    // number with the position rather than the kernel. The cause was NOT Metal
    // encoder merging (that theory was tested and refuted -- bracket coverage
    // never exceeds 1, so passes do not overlap). It was `PassBroker.compute()`
    // silently dropping the label when a pass is already open, leaving one
    // bracket running across every downstream stage, compounded by Dawn's
    // `timestamp_quantization` toggle rounding every reading to 65536 ns so
    // anything genuinely cheap read as exactly zero. With both fixed these five
    // dispatches total 0.19 ms, matching the ~0.3 ms the wall clock measured.
    const splitPublication = process.env.FLUID_STRUCTURED_WORKSET_SPLIT === "1";
    let publication = broker.compute({ label: "Publish dynamic structured boundary worksets" });
    const publish = (label: string, pipeline: GPUComputePipeline, group: GPUBindGroup,
      indirectOffset?: number) => {
      if (splitPublication) { broker.fence(label); publication = broker.compute({ label }); }
      publication.setPipeline(pipeline); publication.setBindGroup(0, group);
      if (indirectOffset === undefined) publication.dispatchWorkgroups(1);
      else publication.dispatchWorkgroupsIndirect(this.dispatch, indirectOffset);
    };
    publish("Workset count row classes", this.countRowClasses, groups[6]!, 0);
    publish("Workset count family classes", this.countFamilyClasses, groups[7]!, 12);
    publish("Workset scan blocks", this.scanWorksetBlocks, groups[8]!);
    publish("Workset scatter rows", this.scatterRowWorksets, groups[9]!, 0);
    publish("Workset scatter families", this.scatterFamilyWorksets, groups[10]!, 12);
  }

  /** Build an inactive boundary publication from a structured candidate that
   * has reached its candidate-ready magic. This path never accepts committed
   * structured-control semantics. */
  encodeCandidate(broker: PassBroker, fine: Pick<WebGPUFineLevelSetBrickSource,
    "params" | "metadata" | "worklist" | "flags" | "phi">,
    structuredCandidateControl: GPUBuffer): void {
    this.encodeFromStructuredControl(broker, fine, structuredCandidateControl, "candidate-ready");
  }

  private encodeAcceptedCandidate(broker: PassBroker, fine: Pick<WebGPUFineLevelSetBrickSource,
    "params" | "metadata" | "worklist" | "flags" | "phi">): void {
    this.encodeFromStructuredControl(broker, fine, this.resources.structured.control, "accepted");
  }

  encodeReadyCommit(broker: PassBroker): void {
    if (this.destroyed) throw new Error("Structured boundary coefficients are destroyed");
    const pass = broker.compute({ label: "Accept structured boundary publication" });
    pass.setPipeline(this.accept); pass.setBindGroup(0, this.acceptGroup);
    pass.dispatchWorkgroups(1);
  }

  encode(broker: PassBroker, fine: Pick<WebGPUFineLevelSetBrickSource,
    "params" | "metadata" | "worklist" | "flags" | "phi">): void {
    this.encodeAcceptedCandidate(broker, fine);
    this.encodeReadyCommit(broker);
  }

  /** End the one-shot authored t=0 authority. Subsequent invalid coarse-phi
   * samples are errors; the analytic surface is never a recovery fallback. */
  retireAnalyticBootstrap(): void {
    this.device.queue.writeBuffer(this.params, 120, new Uint32Array([0]));
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    for (const buffer of [this.liquidMask, this.solidNormalVelocities, this.candidateMask, this.candidates,
      this.candidateSolidNormalVelocities, this.worksetClasses, this.worksetBlocks,
      this.control, this.candidateControl, this.dispatch, this.params, this.inertSolidStorage]) buffer?.destroy();
    this.worksets.destroy();
  }
}

export const structuredBoundaryCoefficientWGSL = /* wgsl */ `
struct P{counts:vec4u,resolved:vec4u,dimensions:vec4u,physical:vec4f,offset0:vec4u,offset1:vec4u,offset2:vec4u,pad:vec4u}
struct FineP{brickDimensions:vec3u,brickResolution:u32,sampleDimensions:vec3u,samplesPerBrick:u32,origin:vec3f,width:f32,worklistCapacity:u32,worklistHeaderWords:u32,pageCapacity:u32,generation:u32,activeCount:u32,invalid:u32,fineFactor:u32,timestep:f32}
// The 64-byte publication has always carried eight words past the pad. They now
// hold a ghost-fluid theta histogram, which is the only production evidence of
// how stiff a scene's free surface actually is: theta multiplies every surface
// face coefficient by 1/theta, so a surface pinned at the 1e-2 floor raises the
// pressure system's condition number by two orders of magnitude and can exhaust
// the encoded solve budget. Diagnostic only; no kernel reads it back.
struct C{flags:atomic<u32>,firstError:atomic<u32>,rows:u32,slots:u32,epoch:u32,bank:u32,published:u32,pad:u32,
 theta:array<atomic<u32>,8>}
struct VertexArena{header:array<u32,16>,values:array<u32>}
struct RigidBody{positionShape:vec4f,dimensions:vec4f,orientation:vec4f,linearVelocity:vec4f,angularVelocity:vec4f,inverseMassInertia:vec4f,angularMomentumRestitution:vec4f,material:vec4f}
struct Metric{caseId:u32,transformAndFlags:u32,volume:f32,error:u32}
struct CatalogSlotGeometry{neighborOffsetSize:vec4f,areaCentroid:vec4f,normalInverseDistance:vec4f}
@group(0)@binding(0)var<uniform>p:P;@group(0)@binding(1)var<storage,read>accepted:array<u32>;@group(0)@binding(2)var<storage,read_write>a:array<u32>;@group(0)@binding(3)var<storage,read>geometry:array<vec4u>;@group(0)@binding(5)var<uniform>fp:FineP;@group(0)@binding(6)var<storage,read>fineWork:array<u32>;@group(0)@binding(7)var<storage,read>fineMeta:array<u32>;@group(0)@binding(8)var<storage,read>fineFlags:array<u32>;@group(0)@binding(9)var<storage,read>finePhi:array<f32>;@group(0)@binding(10)var<storage,read>solid:VertexArena;@group(0)@binding(11)var<storage,read_write>candidates:array<vec2f>;@group(0)@binding(12)var<storage,read_write>candidateLiquid:array<u32>;@group(0)@binding(13)var<storage,read_write>liquid:array<u32>;@group(0)@binding(14)var<storage,read_write>rows:array<u32>;@group(0)@binding(15)var<storage,read_write>diagonal:array<f32>;@group(0)@binding(16)var<storage,read_write>control:C;@group(0)@binding(17)var<storage,read_write>dispatch:array<u32>;@group(0)@binding(18)var<storage,read_write>dynamicWorksets:array<u32>;@group(0)@binding(19)var<storage,read>rigidBodies:array<RigidBody>;@group(0)@binding(20)var<storage,read_write>solidVelocity:array<f32>;@group(0)@binding(21)var<storage,read_write>candidateSolidVelocity:array<f32>;
@group(0)@binding(22)var<storage,read_write>acceptedBoundary:C;@group(0)@binding(23)var<storage,read>metrics:array<Metric>;@group(0)@binding(24)var<storage,read>catalogSlots:array<CatalogSlotGeometry>;
@group(0)@binding(25)var<storage,read>separationMask:array<u32>;
// Workset publication scratch: one class per row then one per slot (offset by
// the row capacity), and a per-class table of per-block prefix bases.
@group(0)@binding(27)var<storage,read_write>worksetClasses:array<u32>;
@group(0)@binding(28)var<storage,read_write>worksetBlocks:array<u32>;
// Host-rasterized t=0 level set, authoritative only while pad.z==3. Analytic
// and post-bootstrap lanes bind a placeholder here and never sample it.
@group(0)@binding(26)var bootstrapLevelSetIn:texture_3d<f32>;
${makeOctreePowerCoarseLevelSetSampleWGSL(4)}
const INVALID:u32=0xffffffffu;const SOLID_VALID:u32=0x80000000u;fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}fn fail(i:u32,f:u32){atomicOr(&control.flags,f);atomicMin(&control.firstError,i);}
fn abase()->u32{return control.bank*p.counts.w;}fn rbase()->u32{return control.bank*p.counts.x;}fn section63Base()->u32{return control.bank*p.resolved.y;}
fn rowCenter(row:u32)->vec3f{let g=geometry[rbase()+row];let q=vec3u(g.x%p.dimensions.x,(g.x/p.dimensions.x)%p.dimensions.y,g.x/(p.dimensions.x*p.dimensions.y));return(vec3f(q)+.5*f32(g.y))*p.physical.x;}
fn finePage(key:u32)->u32{if(fp.worklistHeaderWords!=7u||arrayLength(&fineWork)<7u||fineWork[0]!=fp.generation||fineWork[2]!=fp.pageCapacity||(fineWork[3]&3u)!=3u){return INVALID;}let base=7u+fp.worklistCapacity;if(base+key>=arrayLength(&fineWork)){return INVALID;}let id=fineWork[base+key];let m=id*10u;return select(INVALID,id,id<fp.pageCapacity&&m+2u<arrayLength(&fineMeta)&&fineMeta[m]==id&&fineMeta[m+1u]==key&&fineMeta[m+2u]==fp.generation);}
fn loadFine(q:vec3i)->vec2f{if(any(q<vec3i(0))||any(q>=vec3i(fp.sampleDimensions))){return vec2f(0.);}let u=vec3u(q);let brick=u/fp.brickResolution;let local=u-brick*fp.brickResolution;let key=brick.x+fp.brickDimensions.x*(brick.y+fp.brickDimensions.y*brick.z);let id=finePage(key);if(id==INVALID){return vec2f(0.);}let li=local.x+fp.brickResolution*(local.y+fp.brickResolution*local.z);let at=id*fp.samplesPerBrick+li;if(at>=arrayLength(&finePhi)||at>=arrayLength(&fineFlags)||(fineFlags[at]&1u)==0u||!finite(finePhi[at])){return vec2f(0.);}return vec2f(finePhi[at],1.);}
fn fineSample(x:vec3f,coarsePhi:f32)->vec2f{if(!(fp.width>0.)||fp.brickResolution==0u){return vec2f(coarsePhi,0.);}let lattice=(x-fp.origin)/fp.width-vec3f(.5);let maximum=vec3f(fp.sampleDimensions)-vec3f(1.);if(any(lattice<vec3f(0.))||any(lattice>maximum)){return vec2f(coarsePhi,0.);}let base=vec3i(floor(lattice));let fraction=fract(lattice);var value=0.;for(var corner=0u;corner<8u;corner+=1u){let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let weight=select(1.-fraction.x,fraction.x,o.x==1)*select(1.-fraction.y,fraction.y,o.y==1)*select(1.-fraction.z,fraction.z,o.z==1);if(weight==0.){continue;}let sample=loadFine(base+o);if(sample.y==0.){return vec2f(coarsePhi,0.);}value+=weight*sample.x;}return vec2f(value,1.);}
fn analyticInitialPhi(point:vec3f)->f32{let extent=vec3f(p.dimensions.xyz)*p.physical.x;let world=point-vec3f(.5*extent.x,0.,.5*extent.z);let fill=bitcast<f32>(p.pad.w);if(p.pad.z==1u){return world.y-fill*extent.y;}let heightFraction=max(.92,fill);let footprintFraction=sqrt(fill/max(heightFraction,1e-9));let exposedMaximum=vec3f(-.5*extent.x+footprintFraction*extent.x,heightFraction*extent.y,-.5*extent.z+footprintFraction*extent.z);let q=world-exposedMaximum;return length(max(q,vec3f(0.)))+min(max(q.x,max(q.y,q.z)),0.);}
fn bootstrapTexturePhi(point:vec3f)->f32{let c=vec3i(floor(point/p.physical.x));
 return textureLoad(bootstrapLevelSetIn,clamp(c,vec3i(0),vec3i(p.dimensions.xyz)-vec3i(1)),0).x;}
fn coarseValue(point:vec3f)->vec2f{if(p.pad.z==3u){return vec2f(bootstrapTexturePhi(point),1.);}
 if(p.pad.z!=0u){return vec2f(analyticInitialPhi(point),1.);}let generation=powerCoarseSamples.generation&0x3fffffffu;let expected=control.epoch&0x3fffffffu;if(powerCoarseSamples.state!=0x80000000u||generation!=expected||any(powerCoarseSamples.dimensions!=p.dimensions.xyz)||abs(powerCoarseSamples.physicalCellSize-p.physical.x)>1e-5*max(powerCoarseSamples.physicalCellSize,p.physical.x)){return vec2f(1.,0.);}let value=sampleCoarseOctreePhi(point);let valid=finite(value)&&value<3.402823e38;return vec2f(value,select(0.,1.,valid));}
fn publishStructuredBoundarySetup(rowCount:u32,slotCount:u32,generation:u32,bank:u32,valid:bool){atomicStore(&control.flags,select(1u,0u,valid));atomicStore(&control.firstError,select(0u,INVALID,valid));control.rows=select(0u,rowCount,valid);control.slots=select(0u,slotCount,valid);control.epoch=select(0u,generation,valid);control.bank=select(0u,bank,valid);control.published=0u;for(var b=0u;b<8u;b+=1u){atomicStore(&control.theta[b],0u);}dispatch[0]=(control.rows+63u)/64u;dispatch[1]=1u;dispatch[2]=1u;dispatch[3]=(control.slots+63u)/64u;dispatch[4]=1u;dispatch[5]=1u;}
// Bucket boundaries are chosen around the 1e-2 floor and the values at which a
// 1/theta face coefficient starts to dominate the row: [floor, .05, .1, .25,
// .5, <1, ==1] with bucket 0 reserved for a decoupled face.
fn recordTheta(scale:f32){
 if(!(scale>0.)){atomicAdd(&control.theta[0],1u);return;}
 let theta=1./scale;
 var bucket=7u;
 if(theta<=1.05e-2){bucket=1u;}
 else if(theta<=5e-2){bucket=2u;}
 else if(theta<=1e-1){bucket=3u;}
 else if(theta<=2.5e-1){bucket=4u;}
 else if(theta<=5e-1){bucket=5u;}
 else if(theta<0.999){bucket=6u;}
 atomicAdd(&control.theta[bucket],1u);}
@compute @workgroup_size(1)fn prepareStructuredBoundaryCandidate(){if(arrayLength(&accepted)<6u){publishStructuredBoundarySetup(0u,0u,0u,0u,false);return;}let rowCount=accepted[2];let slotCount=accepted[3];let generation=accepted[4];let bank=accepted[5];let valid=accepted[0]==${OCTREE_STRUCTURED_GPU_VALID}u&&rowCount<=p.counts.x&&slotCount<=p.counts.y&&generation!=0u&&bank<=1u;publishStructuredBoundarySetup(rowCount,slotCount,generation,bank,valid);}
@compute @workgroup_size(1)fn prepareStructuredBoundaryAccepted(){if(arrayLength(&accepted)<6u){publishStructuredBoundarySetup(0u,0u,0u,0u,false);return;}let rowCount=accepted[2];let generation=accepted[3];let bank=accepted[4];let slotCount=accepted[5];let valid=accepted[0]==0u&&rowCount<=p.counts.x&&slotCount<=p.counts.y&&generation!=0u&&bank<=1u;publishStructuredBoundarySetup(rowCount,slotCount,generation,bank,valid);}
@compute @workgroup_size(64)fn classifyStructuredLiquidRows(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=control.rows){return;}let point=rowCenter(row);let coarseSample=coarseValue(point);let sample=fineSample(point,coarseSample.x);if(coarseSample.y==0.&&sample.y==0.){candidateLiquid[row]=0u;fail(row,2u);return;}candidateLiquid[row]=select(0u,1u,sample.x<0.);}
fn handleOwner(h:u32)->u32{return a[abase()+p.offset0.x+h];}fn handleNeighbor(h:u32)->u32{return a[abase()+p.offset0.y+h];}fn handleNormal(h:u32)->vec3f{let at=abase()+p.offset1.z+4u*h;return vec3f(bitcast<f32>(a[at]),bitcast<f32>(a[at+1u]),bitcast<f32>(a[at+2u]));}fn handleCenter(h:u32)->vec3f{let at=abase()+p.offset1.w+4u*h;return vec3f(bitcast<f32>(a[at]),bitcast<f32>(a[at+1u]),bitcast<f32>(a[at+2u]));}fn sbase()->u32{return control.bank*p.counts.y;}fn lbase()->u32{return control.bank*p.counts.x;}fn liquidAt(row:u32)->u32{return liquid[lbase()+row];}
fn quaternionRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);let uuv=cross(q.yzw,uv);return v+2.*(q.x*uv+uuv);}fn quaternionInverseRotate(q:vec4f,v:vec3f)->vec3f{return quaternionRotate(vec4f(q.x,-q.yzw),v);}fn rigidSdf(body:RigidBody,world:vec3f)->f32{let q=quaternionInverseRotate(body.orientation,world-body.positionShape.xyz);let d=body.dimensions.xyz;let shape=i32(round(body.positionShape.w));if(shape==0){return length(q)-d.x;}if(shape==1){let b=abs(q)-.5*d;return length(max(b,vec3f(0)))+min(max(b.x,max(b.y,b.z)),0.);}if(shape==2){let cy=clamp(q.y,-.5*d.y,.5*d.y);return length(vec3f(q.x,q.y-cy,q.z))-d.x;}let radial=length(q.xz)-d.x;let axial=abs(q.y)-.5*d.y;return length(max(vec2f(radial,axial),vec2f(0)))+min(max(radial,axial),0.);}
fn worldBoundaryBit(h:u32)->u32{if(handleNeighbor(h)!=INVALID){return 0u;}let x=handleCenter(h)/p.physical.x;let n=handleNormal(h);for(var axis=0u;axis<3u;axis+=1u){if(n[axis]<-.5&&x[axis]<=1e-4){return 1u<<(2u*axis);}if(n[axis]>.5&&x[axis]>=f32(p.dimensions[axis])-1e-4){return 1u<<(2u*axis+1u);}}return 0u;}fn closedWorld(h:u32)->bool{let bit=worldBoundaryBit(h);return bit!=0u&&(p.resolved.z&bit)!=0u;}
// Unilateral overhead contact: the previous projection marked rows holding
// tension against gravity-opposed closed world faces, storing the exact
// face bits. A fresh mark opens those faces for this epoch, so the solve
// resolves the separation itself (p = 0 ghost past the face) instead of
// pinning the liquid with suction.
fn separationFresh(row:u32,faceBit:u32)->bool{
  let at=rbase()+row;
  if(at>=arrayLength(&geometry)){return false;}
  let cell=geometry[at].x;
  if(cell>=arrayLength(&separationMask)){return false;}
  let word=separationMask[cell];
  if((word&faceBit)==0u){return false;}
  let marked=word>>6u;
  let epoch=(control.epoch&0x3fffffffu)&0x3ffffffu;
  let age=(epoch-marked)&0x3ffffffu;
  return age<=2u;
}
fn solidAt(row:u32,x:vec3f)->f32{if(p.resolved.w==0u||solid.header[5]!=SOLID_VALID||row>=solid.header[2]){return 1e20;}let rg=geometry[rbase()+row];let q=vec3u(rg.x%p.dimensions.x,(rg.x/p.dimensions.x)%p.dimensions.y,rg.x/(p.dimensions.x*p.dimensions.y));let t=clamp((x/p.physical.x-vec3f(q))/f32(rg.y),vec3f(0),vec3f(1));var v=0.;for(var corner=0u;corner<8u;corner+=1u){let w=select(1.-t.x,t.x,(corner&1u)!=0u)*select(1.-t.y,t.y,(corner&2u)!=0u)*select(1.-t.z,t.z,(corner&4u)!=0u);v+=w*bitcast<f32>(solid.values[row*8u+corner]);}return v;}
@compute @workgroup_size(64)fn resolveStructuredBoundarySlots(@builtin(global_invocation_id)g:vec3u){let h=g.x;if(h>=control.slots||atomicLoad(&control.flags)!=0u){return;}let lo=handleOwner(h);let hi=handleNeighbor(h);if(lo>=control.rows){fail(h,4u);return;}let loPoint=rowCenter(lo);let world=worldBoundaryBit(h)!=0u;var hiPoint=handleCenter(h);if(hi!=INVALID){hiPoint=rowCenter(hi);}else if(!world){let inv=bitcast<f32>(a[abase()+p.offset0.w+h]);hiPoint=loPoint+handleNormal(h)/max(inv,1e-20);}let clo=coarseValue(loPoint);let chi=coarseValue(hiPoint);let plo=fineSample(loPoint,clo.x);let phi=fineSample(hiPoint,chi.x);if((clo.y==0.&&plo.y==0.)||(!world&&chi.y==0.&&phi.y==0.)){fail(h,8u|128u);return;}var scale=1.;
// Ghost-fluid crossing handling covers every orientation the moving front
// produces. A band row's centre may legitimately sit on the dry side of its
// own surface face for a step (the frontier admits rows the interface has
// not reached yet); hard-failing that geometry poisoned the whole topology
// epoch, permanently freezing the accepted velocity authority the first time
// an off-cadence step produced it. theta is measured from whichever side is
// wet, with the reference lane's floor so a sliver crossing cannot blow up
// the gradient; a topologically open face with no wet side simply decouples.
if(!world&&(hi==INVALID||((plo.x<0.)!=(phi.x<0.)))){
  if(plo.x<0.&&!(phi.x<0.)){
    let theta=max(-plo.x/(phi.x-plo.x),1e-2);
    if(!finite(theta)||theta>1.){fail(h,8u|1024u);return;}
    scale=1./theta;
  }else if(hi!=INVALID&&phi.x<0.&&!(plo.x<0.)){
    let theta=max(-phi.x/(plo.x-phi.x),1e-2);
    if(!finite(theta)||theta>1.){fail(h,8u|1024u);return;}
    scale=1./theta;
  }else if(plo.x<0.&&phi.x<0.){
    // Open face whose ghost point is still wet: plain liquid gradient.
    scale=1.;
  }else{
    // No wet side: the face carries extended air velocity, no coupling.
    scale=0.;
  }
}else if(hi!=INVALID&&plo.x>=0.&&phi.x>=0.){scale=0.;}else if(world&&plo.x>=0.){scale=0.;}
let boundaryBit=worldBoundaryBit(h);
var aperture=1.;
if(boundaryBit!=0u&&(p.resolved.z&boundaryBit)!=0u){
  // Closed world face. A face carrying a fresh separation mark for exactly
  // this face bit opens for this epoch; every other closed face stays a
  // prescribed solid.
  aperture=select(0.,1.,separationFresh(lo,boundaryBit));
}
recordTheta(scale);candidates[h]=vec2f(aperture,scale);}
@compute @workgroup_size(64)fn resolveStructuredSolidSlots(@builtin(global_invocation_id)g:vec3u){let h=g.x;if(h>=control.slots||atomicLoad(&control.flags)!=0u){return;}let lo=handleOwner(h);let hi=handleNeighbor(h);if(lo>=control.rows){fail(h,4u);return;}let x=handleCenter(h);let n=handleNormal(h);let inv=bitcast<f32>(a[abase()+p.offset0.w+h]);let half=.5/max(inv,1e-20);var aperture=candidates[h].x;if(aperture>0.&&p.resolved.w!=0u){let sdf=min(solidAt(lo,x),select(1e20,solidAt(hi,x),hi!=INVALID));if(!finite(sdf)){fail(h,16u);return;}aperture=clamp(.5+sdf/max(p.physical.x,1e-20),0.,1.);}var best=1e20;var normalVelocity=0.;for(var body=0u;body<p.dimensions.w;body+=1u){if(body>=arrayLength(&rigidBodies)){fail(h,32u);return;}let rb=rigidBodies[body];let sdf=rigidSdf(rb,x);if(sdf<half&&sdf<best){best=sdf;normalVelocity=dot(rb.linearVelocity.xyz+cross(rb.angularVelocity.xyz,x-rb.positionShape.xyz),n);}}candidateSolidVelocity[h]=normalVelocity;candidates[h]=vec2f(aperture,candidates[h].y);}
@compute @workgroup_size(64)fn commitStructuredBoundarySlots(@builtin(global_invocation_id)g:vec3u){let h=g.x;if(h>=control.slots||atomicLoad(&control.flags)!=0u){return;}a[abase()+p.offset1.x+h]=bitcast<u32>(candidates[h].x);a[abase()+p.offset1.y+h]=bitcast<u32>(candidates[h].y);solidVelocity[sbase()+h]=candidateSolidVelocity[h];}
fn canonicalDirection(channel:u32)->vec3i{let d=array<vec3i,18>(vec3i(1,0,0),vec3i(-1,0,0),vec3i(0,1,0),vec3i(0,-1,0),vec3i(0,0,1),vec3i(0,0,-1),vec3i(1,1,0),vec3i(1,-1,0),vec3i(-1,1,0),vec3i(-1,-1,0),vec3i(1,0,1),vec3i(1,0,-1),vec3i(-1,0,1),vec3i(-1,0,-1),vec3i(0,1,1),vec3i(0,1,-1),vec3i(0,-1,1),vec3i(0,-1,-1));return d[channel];}
fn channelForCatalogSlot(global:u32)->u32{if(global>=arrayLength(&catalogSlots)){return INVALID;}let slot=catalogSlots[global];let direction=vec3i(round(slot.areaCentroid.yzw+.5*slot.normalInverseDistance.xyz));for(var channel=0u;channel<18u;channel+=1u){if(all(direction==canonicalDirection(channel))){return channel+1u;}}return INVALID;}
@compute @workgroup_size(64)fn rebuildStructuredBoundaryRows(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=control.rows||atomicLoad(&control.flags)!=0u){return;}liquid[lbase()+row]=candidateLiquid[row];let base=section63Base()+row*19u;if(base+19u>arrayLength(&rows)){fail(row,64u);return;}for(var channel=0u;channel<19u;channel+=1u){rows[base+channel]=0u;}var d=select(1.,0.,candidateLiquid[row]!=0u);for(var local=0u;local<p.counts.z;local+=1u){let at=row*p.counts.z+local;let h=a[abase()+p.offset2.y+at];if(h==INVALID||h>=control.slots){continue;}let lo=handleOwner(h);let hi=handleNeighbor(h);let other=select(lo,hi,lo==row);let aperture=bitcast<f32>(a[abase()+p.offset1.x+h]);let scale=bitcast<f32>(a[abase()+p.offset1.y+h]);let coefficient=bitcast<f32>(a[abase()+p.offset0.z+h])*bitcast<f32>(a[abase()+p.offset0.w+h])*aperture*scale;if(candidateLiquid[row]==0u||coefficient<=0.){continue;}d+=coefficient;if(other!=INVALID&&other<control.rows&&other!=row&&candidateLiquid[other]!=0u){let global=a[abase()+p.offset2.w+at];let channel=channelForCatalogSlot(global);if(channel==INVALID){fail(row,64u);continue;}rows[base+channel]=bitcast<u32>(bitcast<f32>(rows[base+channel])+coefficient);}}rows[base]=bitcast<u32>(d);if(row+1u==control.rows){control.published=control.epoch;}}
fn rowTransition(row:u32)->bool{let at=rbase()+row;return row<control.rows&&at<arrayLength(&geometry)&&geometry[at].z!=0u;}
fn dynamicRowClass(row:u32)->u32{let base=section63Base()+row*19u;var off=0.;for(var channel=1u;channel<19u;channel+=1u){off+=bitcast<f32>(rows[base+channel]);}let boundary=bitcast<f32>(rows[base])>off+max(1e-6,1e-5*abs(off));return select(select(0u,2u,boundary),select(1u,3u,boundary),rowTransition(row));}
// A row's class is read once per row and up to twice per incident slot. The
// serial publication recomputed it every time, so the 19-channel offdiagonal
// sum ran up to four times per slot; caching it is the bulk of the win.
fn cachedRowClass(row:u32)->u32{return select(dynamicRowClass(row),worksetClasses[row],row<control.rows);}
fn dynamicFamilyClass(h:u32)->u32{let lo=handleOwner(h);let hi=handleNeighbor(h);if(liquidAt(lo)==0u&&(hi==INVALID||hi>=control.rows||liquidAt(hi)==0u)){return 9u;}let transition=rowTransition(lo)||(hi!=INVALID&&hi<control.rows&&rowTransition(hi));let aperture=bitcast<f32>(a[abase()+p.offset1.x+h]);let scale=bitcast<f32>(a[abase()+p.offset1.y+h]);let boundary=hi==INVALID||cachedRowClass(lo)>=2u||(hi<control.rows&&cachedRowClass(hi)>=2u)||aperture<0.999999||abs(scale-1.)>1e-6||(hi<control.rows&&liquidAt(lo)!=liquidAt(hi));return select(select(5u,7u,boundary),select(6u,8u,boundary),transition);}
fn worksetBase(cls:u32)->u32{return control.bank*p.pad.y+cls*p.pad.x;}
// Wide deterministic workset publication.
//
// This replaces a single 64-lane workgroup that walked every row and every
// slot, ran a 9-class Hillis-Steele scan across its own lanes (108 barriers),
// and then walked both ranges a second time to scatter. At ~12k slots that is
// a latency-bound serial crawl: 64 resident threads of 98,304, with nothing
// to hide a dependent global load behind. It measured 4.00 ms of a 48.7 ms
// advance.
//
// The replacement is the standard count -> block scan -> scatter compaction,
// one lane per element. Destinations are IDENTICAL to the serial form, not
// merely equivalent: a block's base is the exclusive prefix of the per-block
// counts in ascending block order, and a lane's offset within its block is
// the number of lower lanes carrying the same class, so every class's workset
// is still exactly its members in ascending element order. That determinism
// is why the scan is an ordered integer prefix and not an atomic bump; an
// atomic counter is also banned outright in this shader.
//
// Row classes are 0..3 and family classes 5..8, which never overlap, so the
// row and slot phases share the block table without a shared cursor: the row
// count writes classes 0..4 at row-block indices and the slot count writes
// classes 5..8 at slot-block indices.
const WORKSET_SKIP:u32=0xffffffffu;
var<workgroup> worksetBlockClass:array<u32,64>;
var<workgroup> worksetScan:array<u32,256>;
var<workgroup> worksetGate:u32;
fn worksetPublicationEnabled()->bool{return atomicLoad(&control.flags)==0u&&control.rows>0u&&control.published==control.epoch;}
fn worksetBlockStride()->u32{return (max(p.counts.x,p.counts.y)+63u)/64u;}
fn worksetSlotClassBase()->u32{return p.counts.x;}
fn worksetBlockTotal(cls:u32)->u32{var n=0u;for(var j=0u;j<64u;j+=1u){if(worksetBlockClass[j]==cls){n+=1u;}}return n;}
fn worksetBlockOffset(lane:u32)->u32{let cls=worksetBlockClass[lane];var n=0u;for(var j=0u;j<lane;j+=1u){if(worksetBlockClass[j]==cls){n+=1u;}}return n;}
@compute @workgroup_size(64)fn countStructuredRowClasses(@builtin(global_invocation_id)g:vec3u,@builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wg:vec3u){
 if(lane==0u){worksetGate=select(0u,1u,worksetPublicationEnabled());}
 if(workgroupUniformLoad(&worksetGate)==0u){return;}
 let row=g.x;
 var cls=WORKSET_SKIP;
 if(row<control.rows){cls=dynamicRowClass(row);worksetClasses[row]=cls;}
 worksetBlockClass[lane]=cls;
 workgroupBarrier();
 if(lane<5u){worksetBlocks[lane*worksetBlockStride()+wg.x]=worksetBlockTotal(lane);}
}
@compute @workgroup_size(64)fn countStructuredFamilyClasses(@builtin(global_invocation_id)g:vec3u,@builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wg:vec3u){
 if(lane==0u){worksetGate=select(0u,1u,worksetPublicationEnabled());}
 if(workgroupUniformLoad(&worksetGate)==0u){return;}
 let h=g.x;
 var cls=WORKSET_SKIP;
 if(h<control.slots){let value=dynamicFamilyClass(h);worksetClasses[worksetSlotClassBase()+h]=value;if(value<9u){cls=value;}}
 worksetBlockClass[lane]=cls;
 workgroupBarrier();
 if(lane>=5u&&lane<9u){worksetBlocks[lane*worksetBlockStride()+wg.x]=worksetBlockTotal(lane);}
}
@compute @workgroup_size(256)fn scanStructuredWorksetBlocks(@builtin(local_invocation_index)lane:u32){
 if(lane==0u){worksetGate=select(0u,1u,worksetPublicationEnabled());}
 if(workgroupUniformLoad(&worksetGate)==0u){return;}
 // Every barrier below is reached under bounds derived from the uniform p, so
 // the loop trip counts are uniform even though the live-element predicate is
 // not; out-of-range blocks contribute a zero and never gate a barrier.
 let stride=worksetBlockStride();
 let rowBlocks=(control.rows+63u)/64u;
 let slotBlocks=(control.slots+63u)/64u;
 for(var cls=0u;cls<9u;cls+=1u){
  let blocks=select(rowBlocks,slotBlocks,cls>=5u);
  var carry=0u;
  for(var start=0u;start<stride;start+=256u){
   let at=start+lane;
   let live=at<blocks;
   var value=0u;if(live){value=worksetBlocks[cls*stride+at];}
   worksetScan[lane]=value;workgroupBarrier();
   for(var width=1u;width<256u;width<<=1u){
    var add=0u;if(lane>=width){add=worksetScan[lane-width];}
    workgroupBarrier();worksetScan[lane]+=add;workgroupBarrier();
   }
   if(live){worksetBlocks[cls*stride+at]=carry+worksetScan[lane]-value;}
   let total=worksetScan[255];workgroupBarrier();
   carry+=total;
  }
  if(lane==0u){let base=worksetBase(cls);dynamicWorksets[base]=control.epoch;dynamicWorksets[base+1u]=carry;dynamicWorksets[base+2u]=select(p.counts.x,p.counts.y,cls>=5u);dynamicWorksets[base+3u]=3u;dynamicWorksets[base+4u]=(carry+63u)/64u;dynamicWorksets[base+5u]=1u;dynamicWorksets[base+6u]=1u;}
 }
}
@compute @workgroup_size(64)fn scatterStructuredRowWorksets(@builtin(global_invocation_id)g:vec3u,@builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wg:vec3u){
 if(lane==0u){worksetGate=select(0u,1u,worksetPublicationEnabled());}
 if(workgroupUniformLoad(&worksetGate)==0u){return;}
 let row=g.x;
 var cls=WORKSET_SKIP;
 if(row<control.rows){cls=worksetClasses[row];}
 worksetBlockClass[lane]=cls;
 workgroupBarrier();
 if(cls>=9u){return;}
 dynamicWorksets[worksetBase(cls)+7u+worksetBlocks[cls*worksetBlockStride()+wg.x]+worksetBlockOffset(lane)]=row;
}
@compute @workgroup_size(64)fn scatterStructuredFamilyWorksets(@builtin(global_invocation_id)g:vec3u,@builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wg:vec3u){
 if(lane==0u){worksetGate=select(0u,1u,worksetPublicationEnabled());}
 if(workgroupUniformLoad(&worksetGate)==0u){return;}
 let h=g.x;
 var cls=WORKSET_SKIP;
 if(h<control.slots){cls=worksetClasses[worksetSlotClassBase()+h];}
 worksetBlockClass[lane]=cls;
 workgroupBarrier();
 if(cls>=9u){return;}
 dynamicWorksets[worksetBase(cls)+7u+worksetBlocks[cls*worksetBlockStride()+wg.x]+worksetBlockOffset(lane)]=h;
}
@compute @workgroup_size(1)fn acceptStructuredBoundary(){if(atomicLoad(&control.flags)!=0u||control.epoch==0u||control.published!=control.epoch){return;}atomicStore(&acceptedBoundary.flags,0u);atomicStore(&acceptedBoundary.firstError,INVALID);acceptedBoundary.rows=control.rows;acceptedBoundary.slots=control.slots;acceptedBoundary.epoch=control.epoch;acceptedBoundary.bank=control.bank;acceptedBoundary.published=control.published;acceptedBoundary.pad=0u;for(var b=0u;b<8u;b+=1u){atomicStore(&acceptedBoundary.theta[b],atomicLoad(&control.theta[b]));}}
`;
