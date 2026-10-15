import { proceduralSolidField, proceduralSolidKey, PROCEDURAL_SOLID_MAGIC } from "../../core/procedural-solid";
import { gpuInitializationDevice } from "../../core/gpu-initialization-device";
import {UniformMixedDiagnostics} from "./uniform-mixed-diagnostics";
import { UNIFORM_MIXED_RECEIPT_RING, UniformMixedFrame, type UniformMixedFrameBodies, type UniformMixedFrameParameters, type UniformMixedFrameReceipt, type UniformMixedFrameRelayout, type UniformMixedFrameTrace } from "./uniform-mixed-frame";
import { uniformDetailCapacityRule } from "./uniform-detail-policy";
import { uniformMixedPressureRootBytes } from "./uniform-mixed-pressure-memory";
import { assertUniformMixedOptions, type UniformMixedOptionalFeatures } from "./uniform-mixed-options";
import { assertUniformMixedSolidPromotion, createUniformMixedLayout, createUniformMixedLayoutFromWidths, uniformMixedSolidEdit, uniformMixedSolidTiles, uniformMixedSolidTilesAfterEdit } from "./uniform-mixed-layout";
import { sameUniformDetailSettings, UNIFORM_DETAIL_PARAM_KEYS, uniformDetailCoarseSolids, uniformDetailImportance, uniformDetailSettings, uniformDetailValues, type UniformDetailSettings } from "./uniform-detail-policy";
import { buildUniformDetailRequests, uniformDetailDomain, uniformDetailRequestKey, uniformRegionDetailTier, type DetailRequest } from "./uniform-detail-requests";
import { UniformDetailPlanner, UniformDetailRequestTiles, uniformDetailTileBounds, type UniformDetailPlan } from "./uniform-detail-planner";
import type { UniformWorkEdit } from "./uniform-buffered-work";
import type { SolverDetailFocus, SolverDetailInput } from "../../core/solver-detail";
import { UniformMixedLayoutBuilder, UNIFORM_MIXED_RELAYOUT_RECEIPT, uniformMixedRegionTable, type UniformMixedRegion } from "./uniform-mixed-layout-builder";
import { UniformMixedDynamicClassifier, UNIFORM_MIXED_DYNAMIC_FULL_TOLERANCE, uniformMixedTileWord, type UniformMixedDynamicPolicy } from "./uniform-mixed-dynamic";
import { fluidRegionLattice, refinementRegionLattice } from "../../core/refinement-regions";
import { UniformScratchArena } from "./uniform-scratch-arena";
import { uniformBoxTileCoord, uniformBoxTileIndex, uniformLatticeAddress, uniformTileInBox, uniformTileNear, type UniformLatticeAddress } from "./uniform-lattice-address";
import { UniformTexturePages } from "./uniform-texture-pages";
import { UNIFORM_PAGE_POOL_EDGE, UNIFORM_PAGE_POOL_PIN_MARGIN, uniformPagePoolDropAhead, uniformPagePoolResidency, uniformPagePoolSourcePages, type UniformPagePoolCellBox, type UniformPagePoolResidency } from "./uniform-page-pool";
import { uniformDensityPostProcessingEnabled } from "./uniform-options";
export { uniformDensityPostProcessingEnabled } from "./uniform-options";
import type { DenseLevelSetVolumeConsumerSource } from "../../core/levelset-consumer-abi";
import {
  UNIFORM_VOLUME_EDGE_BYTES,
  UNIFORM_VOLUME_TWO_LEVEL_COUNTER_WORDS,
  UNIFORM_VOLUME_TWO_LEVEL_WORDS_PER_TILE,
} from "./uniform-volume.wgsl";
import { createUniformReferenceComputeShader } from "./webgpu-uniform-reference.wgsl";
import { uniformLatticeFarPhi, uniformLatticeInitialPhi, uniformLatticeInitialVolume } from "./uniform-volume-initial";
import { averageInflowStrength, createInflowGridBoundary, type InflowGridBoundary } from "../../core/inflow-boundary";
import type { InitialLiquidSphere, SceneDescription, ScheduledLiquidDrop } from "../../core/model";
import { liquidExtrusionBounds, liquidExtrusionEdges, liquidExtrusionVolume_m3, LIQUID_EXTRUSION_MAX_EDGES, type LiquidExtrusion } from "../../core/liquid-extrusion";
import { UNIFORM_PARAMS_BYTES, UNIFORM_PARAMS_HEAD_BYTES } from "./uniform-source.wgsl";
import { planUniformHostAllocation } from "./uniform-host-allocation";
import { boundingRadius, initializeRigidBodies, type RigidBodyState } from "../../core/rigid-body";
import { UniformMixedBodies } from "./uniform-mixed-bodies";
import { UniformBedSlopeBake } from "./uniform-bed-slope-bake";
import { UNIFORM_OPEN_WORLD_TERRAIN_COLUMNS, UniformOpenWorldTerrain } from "./uniform-open-world-terrain";
import { UniformPrescribedSolidMotion } from "./uniform-prescribed-solid-motion";
import { fluidFrameCell, sceneFluidBoxPage, sceneFluidFrame, sceneFluidLatticeDimensions, sceneLatticeDimensions, type SceneFluidFrame } from "../../core/scene-lattice";
import { planGPUAdvanceInto, type GPUAdvancePlan } from "../../core/tall-cell-diagnostics";
import type { GPUQuality } from "../../core/gpu-quality";
import { sceneHasTerrain } from "../../core/terrain";
import {
  GPU_RIGID_EXCHANGE_BYTES,
  type GPUEulerianInfo,
  type GPURigidLoad,
  type GPUVelocityTransport,
} from "../../core/webgpu-eulerian";
import { GPUInitializationTaskRunner, settleGPUInitialization, type GPUInitializationTask } from "../../core/gpu-initialization";
import type {
  GPUSolverInstance,
  GPUInitializationReporter,
  InjectedLiquidBall,
  MethodParamValues,
} from "../../core/method-contract";
import { WebGPURigidBodySystem } from "../../core/webgpu-rigid-body";
import { WebGPUUniformVelocityExtrapolator } from "./webgpu-uniform-velocity-extrapolation";
import {
  UNIFORM_CM11A_FULL_CYCLES,
  UNIFORM_CM11A_POST_SWEEPS,
  UNIFORM_CM11A_PRE_SWEEPS,
  UNIFORM_PRESSURE_RESIDUAL_TOLERANCE,
  UNIFORM_CM11A_V_CYCLES,
  WebGPUUniformPressureMultigrid,
  type UniformCM11aSchedule,
} from "./webgpu-uniform-pressure-multigrid";
import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import {
  GPUStageTimestampRecorder,
} from "../../core/performance-trace";
import { usePerformanceInstrumentationStore } from "../../core/stores/performance-instrumentation-store";
import { gpuPhysicsPerformanceActivityFrameId } from "../../core/gpu-performance-activity";
export { UNIFORM_ADVANCE_PHASE } from "./uniform-stages";
export { UNIFORM_FLUID_PIPELINE } from "./uniform-pipeline";
import { UNIFORM_GAMMA_DIFFUSION_DEFAULT_ITERATIONS, UNIFORM_GAMMA_DIFFUSION_MAX_ITERATIONS } from "./parameters";
export { UNIFORM_GAMMA_DIFFUSION_DEFAULT_ITERATIONS, UNIFORM_GAMMA_DIFFUSION_MAX_ITERATIONS } from "./parameters";
import { uniformFixedAdvanceReady, uniformFixedStep_s } from "./uniform-paper";
import { liveFluidEditRefusal, type LiveFluidEdit, type LiveFluidEditResult } from "../../core/live-fluid-edit";
import { SolidOccupancyMask } from "../../core/solid-occupancy-mask";
import { solidWorldForFluidLattice } from "../../core/solid-world";
import { uniformSolidMaskPages, uniformSolidScratchLayout } from "./uniform-mixed-solid.wgsl";
import { UniformDetailStorage, uniformDetailDenseShader, uniformDetailField, type UniformDetailClass } from "./uniform-detail-fields";


export { UNIFORM_PAPER_DT_S } from "./uniform-paper";

export interface WebGPUUniformReferenceOptions {
  narrowBandAdaptiveBudgetPercent?: number;
  narrowBandAdaptiveFadeSeconds?: number;
  narrowBandSparseParticleLifetime?: number;
  /** Seconds over which NB-FLIP closes its gap to the volume budget; zero for no volume control. */
  narrowBandVolumeControlSeconds?: number;
  narrowBandFinePadding?: number;
  narrowBandTransfer?: "flip" | "apic";
  /** NB-FLIP: the advected phi of h vertices is trilinear (4h vertices keep phiCubicAdvection). */
  phiFineTrilinear?: boolean;
  /** Independent dense vertex level set and conservative cell volume, advanced
   * only by the mixed-ownership frame. */
  geometricVolume?: boolean;
  /** Retain stage fields, including inactive cells, for full-field diagnostics. */
  retainStageDiagnosticsForQA?: boolean;
  /** The mixed extension's sweep budget; clamped to extrapolationActiveFrontPassCeiling. */
  extensionFrontSweeps?: number;
  /** Simulation detail controls (uniform-detail-policy.ts). Interim mapping
   * onto this solver's ownership: Requested and Full build the CPU region
   * layout from the detail planner's admitted h tiles on a 4h background;
   * Dynamic runs the GPU surface census (docs/plans/uniform-dynamic-coarsening.md)
   * with the planner's h tiles as statics. */
  detail?: UniformDetailSettings;
  /**
   * Geometric only: the splash-survival controls
   * (docs/uniform-geometric-splash-dissipation-plan.md), both on by default.
   *  - phiCubicAdvection: clamped Catmull-Rom instead of trilinear for the
   *    advected phi of band vertices.
   *  - phiDrain: raise phi-liquid vertices with no V anywhere around them.
   *  - phiPreserveSurface (off by default): redistancing keeps h vertices
   *    beside the surface at their advected value (CM11b sec. 3.4).
   */
  phiCubicAdvection?: boolean;
  phiDrain?: boolean;
  phiPreserveSurface?: boolean;
  /** GPU-resident sparse work boxes; false retains the original dense control. */
  activeRegion?: boolean;
  /** Velocity transport used by Algorithm 1 step 3. */
  velocityTransport?: GPUVelocityTransport;
  /** Reject hierarchy-only air samples when updating liquid momentum. */
  liquidOnlyVelocityAdvection?: boolean;
  /** Ordered six-pass Sec. 3.4 iterations; zero cleanly bypasses the stage. */
  gammaDiffusionIterations?: number;
  /** Sec. 3.6 cut-cell excess redistribution. */
  solidExcessCorrection?: boolean;
  /** Two-way fluid/body exchange and rigid integration. */
  rigidCoupling?: boolean;
  /** CM11a cycle and smoothing schedule. */
  pressureSchedule?: UniformCM11aSchedule;
  /**
   * "lagged" encodes only as many cycles as the last observed step needed;
   * "fixed" encodes the whole configured schedule, which is what the solver
   * always did. Runtime-switchable.
   */
  pressureCycleBudget?: "lagged" | "fixed";
  /** Cycles the lagged budget adds above the last observed demand. */
  pressureBudgetHeadroom?: number;
  /** Paper Sec. 3.8 render reconstruction; Results states it is normally off. */
  densityPostProcessing?: boolean;
  /**
   * "paper" advances at the paper's simulation step (1/30 s, the value used
   * by every example in Sec. 4); "scene" honors the scene-authored maxDt.
   * The method's advection/sharpening balance is only calibrated at the
   * paper's step regime: far below it, per-resample transport diffusion
   * outruns Sec. 3.5 sharpening and the interface dilutes below the 0.5
   * isovalue, leaving dynamically inert mass hanging in mid-air.
   */
  timeStep?: "paper" | "sixtieth" | "scene";
  deferPipelineCompilation?: boolean;
}

// Sec. 3.4 permits one through seven gamma-diffusion repetitions per time
// step. One repetition is the reference schedule: each dimensional sweep is
// one Jacobi update from a single snapshot, as specified by LAF11/CM12. More
// repetitions remain available explicitly, but are not a neutral robustness
// setting: they apply more physical/numerical diffusion per simulation step.


/** Pipelines built per round of the background warm-up. Metal compiles
 * serially (1, 4 and all at once take the same 44-45 s on dam64 and none
 * moves the frame), so one a round: a change that waits queues behind one. */
/**
 * One step's liquid source as the parameter block carries it: the ball or disk
 * a caller injected, or a scheduled extrusion behind its bounding ball.
 */
interface UniformLiquidDrop extends InjectedLiquidBall {
  /** The liquid the source asks for, which is what joins the volume reference. */
  readonly volume_m3: number;
  /** An extrusion's outline edges as uploaded; the three lengths shape it. */
  readonly outline?: Float32Array<ArrayBuffer>;
  readonly offset_m?: number;
  readonly halfDepth_m?: number;
  readonly edgeRadius_m?: number;
}

function uniformInjectedDrop(ball: InjectedLiquidBall): UniformLiquidDrop {
  return { ...ball, volume_m3: ball.halfHeight_m !== undefined
    ? Math.PI * ball.radius_m ** 2 * 2 * ball.halfHeight_m
    : (4 / 3) * Math.PI * ball.radius_m ** 3 };
}

/** The cell boxes the scene's sources write liquid in every step: the
 * nozzle's plug over a step of `step_s`. A live page pool pins their pages.
 * A drop, scheduled or injected, is not one: its pages are asked for around
 * its own step (requestDropPages). */
export function uniformPageSources(scene: SceneDescription, plug: InflowGridBoundary | undefined, step_s: number): UniformPagePoolCellBox[] {
  const frame = sceneFluidFrame(scene);
  const box = (centre: readonly number[], end: readonly number[], radius_m: number): UniformPagePoolCellBox => ({
    low: [0, 1, 2].map(k => Math.floor(fluidFrameCell(frame, k, Math.min(centre[k]!, end[k]!) - radius_m))) as [number, number, number],
    high: [0, 1, 2].map(k => Math.ceil(fluidFrameCell(frame, k, Math.max(centre[k]!, end[k]!) + radius_m))) as [number, number, number] });
  const sources: UniformPagePoolCellBox[] = [], inflow = scene.fluid.inflow;
  if (inflow && plug) {
    const o = plug.outletCenter_m, d = plug.direction, reach = Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z) * step_s;
    sources.push(box([o.x, o.y, o.z], [o.x + reach * d.x, o.y + reach * d.y, o.z + reach * d.z], inflow.radius_m));
  }
  return sources;
}

/** The window page, on one axis, of the low (side < 0) or high face of a
 * drop's bounding box of half-width `r_m` grown by UNIFORM_PAGE_POOL_PIN_MARGIN
 * cells, kept inside the box: its `pages` from page `first`. */
function uniformDropPage(centre_m: number, r_m: number, low_m: number, size_m: number, cells: number, first: number, pages: number, side: number, open=false): number {
  const cell = side < 0 ? Math.floor((centre_m - r_m - low_m) * cells / size_m) - UNIFORM_PAGE_POOL_PIN_MARGIN : Math.ceil((centre_m + r_m - low_m) * cells / size_m) + UNIFORM_PAGE_POOL_PIN_MARGIN - 1;
  const page=Math.floor(cell / UNIFORM_PAGE_POOL_EDGE);return open?page:Math.min(Math.max(page, first), first + pages - 1);
}

/** A scheduled volume in source form; `cell_m` sets how finely an outline's volume is measured. */
function uniformScheduledDrop(volume: InitialLiquidSphere | LiquidExtrusion, cell_m: number): UniformLiquidDrop {
  if (volume.shape === "sphere") return uniformInjectedDrop({ centre_m: volume.center_m, radius_m: volume.radius_m });
  const outline = liquidExtrusionEdges(volume);
  if (outline.length > 4 * LIQUID_EXTRUSION_MAX_EDGES) throw new Error(`Uniform scheduled drops: an extrusion has ${outline.length / 4} outline edges, over the ${LIQUID_EXTRUSION_MAX_EDGES} the parameter block holds`);
  // The bounding ball sits on the mid-plane at the outline's centre: the
  // shader reads the mid-plane from its z and skips the outline outside it.
  const { min_m, max_m } = liquidExtrusionBounds(volume);
  const centre_m = { x: 0.5 * (min_m.x + max_m.x), y: 0.5 * (min_m.y + max_m.y), z: volume.centerZ_m };
  let reach = 0;
  for (let index = 0; index < outline.length; index += 4) reach = Math.max(reach, Math.hypot(outline[index]! - centre_m.x, outline[index + 1]! - centre_m.y));
  return {
    centre_m, radius_m: Math.hypot(reach + volume.offset_m, volume.halfDepth_m),
    volume_m3: liquidExtrusionVolume_m3(volume, 0.25 * cell_m),
    outline, offset_m: volume.offset_m, halfDepth_m: volume.halfDepth_m, edgeRadius_m: volume.edgeRadius_m,
  };
}

/**
 * The two documents hold the very same object in every field but their solid
 * patch list: a voxel stroke's document over the one it extends.
 *
 * Identity, not content. A stroke is composed as `{ ...base, solidVoxels }`,
 * so it answers true there for the price of a pointer per field, and every
 * memo the solver keys by a field's identity (refinement regions, the drop
 * schedule) still names its field afterwards. A document that crossed a
 * structured clone, or had any other field replaced, shares nothing and
 * answers false, which costs it only the full adoption it took before.
 */
function uniformSceneDiffersInSolidsAlone(scene: SceneDescription, running: SceneDescription): boolean {
  const fields = Object.keys(scene) as (keyof SceneDescription)[];
  if (fields.length !== Object.keys(running).length) return false;
  for (const field of fields) {
    if (field !== "solidVoxels" && (!Object.hasOwn(running, field) || scene[field] !== running[field])) return false;
  }
  return true;
}

interface UniformReferencePipelines {
  extrapolationAuthorityDense: GPUComputePipeline;
}



/**
 * Boundaries are free (`GPUStageTimestampRecorder` splices them into passes the
 * frame already encodes) but the trace's query set, resolve and map are not.
 */
/** The native kernels' h fields by binding name (uniformDetailDenseShader):
 * what the t=0 authority and publication groups bind there. The pressure
 * and transport slots are 1^3 placeholders under Geometric. */
const UNIFORM_REFERENCE_DETAIL_CLASSES: Readonly<Record<string, UniformDetailClass>> = {
  velocityOut: "face", volumeIn: "cell", volumeOut: "cell", gammaOut: "cell",
  uvPhiIn: "vertex",
};
const UNIFORM_PHYSICS_TRACE_CADENCE_MS = 100;
/** Params lanes the mixed frame's shader still reads as constants: the fine
 * reach and transport margin in 4h tiles, and the phi agreement clamp. */
const UNIFORM_TWO_LEVEL_FINE_REACH = 2;
const UNIFORM_TRANSPORT_REACH = 1;
const UNIFORM_PHI_AGREEMENT_CLAMP = 0.02;

/**
 * Uniform numerical implementation with page-backed geometric fields and a
 * separate dense paper/reference path.
 *
 * This class deliberately has no octree option or adaptive compatibility
 * branch. Its allocations, pipelines, step graph, and diagnostics are owned
 * entirely by the `uniform` method plugin.
 */
/** Relayout receipts (UNIFORM_MIXED_RELAYOUT_RECEIPT) in flight to the host
 * for diagnostics; a frame that finds none free reports nothing. */
const UNIFORM_MIXED_RELAYOUT_READS = 3;

/** A stack over a store it keeps. An array popped or shifted to empty drops
 * its store and the next push allocates another: once a step, for a pool
 * that holds one record. */
class KeptStack<T>{
  private readonly items:T[]=[];
  size=0;
  take():T|undefined{return this.size>0?this.items[--this.size]:undefined;}
  give(item:T):void{this.items[this.size++]=item;}
  at(index:number):T{return this.items[index]!;}
}
/** executionInfo with the counters this solver publishes beside GPUEulerianInfo's own. */
type UniformPublishedInfo=GPUEulerianInfo&Record<string,unknown>;
/** Parameter records the mixed frame may still hold: the receipts in flight, the last step's and this one's, with room. */
const UNIFORM_MIXED_PARAMETER_RING=UNIFORM_MIXED_RECEIPT_RING+6;
/** Advances after an allocation event whose `info` reads sum the mixed frame's bytes again. */
const UNIFORM_MIXED_ALLOCATION_WATCH=8;
export class WebGPUUniformReferenceSolver implements GPUSolverInstance {
  private readonly executionInfo: GPUEulerianInfo;
  get info(): GPUEulerianInfo { if(this.mixedAllocationDue())this.refreshMixedAllocation();return this.executionInfo; }
  get simulationCellScale(): 1 { return 1; }


  private narrowBandAdaptiveBudgetPercent: number;
  private narrowBandAdaptiveFadeSeconds: number;
  private narrowBandSparseParticleLifetime: number;
  private narrowBandVolumeControlSeconds: number;

  private narrowBandFinePadding: number;
  private readonly narrowBandTransfer: "flip" | "apic";
  private phiFineTrilinear: boolean;
  get particleSource(){return this.mixedFrame?.narrowBandFlip?.particleSource;}
  get pagePoolInfo(){return this.mixedFrame?.pagePoolInfo;}
  get workDomainInfo(){return this.mixedFrame?.workDomainInfo;}
  get narrowBandFlipInfo(){
    const stage=this.mixedFrame?.narrowBandFlip;if(!stage)return undefined;
    // One record, rewritten by each read: the values are the stage's at the read.
    const info=this.narrowBandFlipRecord??=this.narrowBandFlipInfoOf(stage),d=stage.diagnostics;
    info.adaptiveBudgetPercent=this.narrowBandAdaptiveBudgetPercent;info.adaptiveFadeSeconds=this.narrowBandAdaptiveFadeSeconds;info.sparseParticleLifetime=this.narrowBandSparseParticleLifetime;
    info.particles=stage.count;info.capacity=stage.capacity;info.reseedClipped=stage.reseedClipped;info.fineGridPadding=this.narrowBandFinePadding;
    let key:keyof typeof d;for(key in d)info[key]=d[key];
    return info;
  }
  private narrowBandFlipInfoOf(stage:UniformMixedFrame["narrowBandFlip"]){return {adaptiveBudgetPercent:this.narrowBandAdaptiveBudgetPercent,adaptiveFadeSeconds:this.narrowBandAdaptiveFadeSeconds,sparseParticleLifetime:this.narrowBandSparseParticleLifetime,particles:stage.count,capacity:stage.capacity,reseedClipped:stage.reseedClipped,bandWidth:4,fineGridPadding:this.narrowBandFinePadding,transfer:stage.transferMode,flipRatio:stage.transferMode==="apic"?0:0.95,...stage.diagnostics};}
  private narrowBandFlipRecord?:ReturnType<WebGPUUniformReferenceSolver["narrowBandFlipInfoOf"]>;
  private appliedRuntimeValues?: MethodParamValues;
  /** Two-level velocity sampling; off when the lattice has no 4h table. */
  private twoLevelVelocity: boolean;
  /** How far past the fine set the extension must still be exact, in 4h tiles. */
  private readonly twoLevelShellReach: number;
  /** Derived from detail.policy: "dynamic" is the census with its importance criteria, "regions" requests only. */
  private detail: UniformDetailSettings;
  private detailPlanner?: UniformDetailPlanner;
  /** Latest transient focus, and the one the accepted plan used. */
  private detailFocusPending?: SolverDetailFocus;
  private detailFocus?: SolverDetailFocus;
  /** Accepted mixed frames: the planner's step clock (hysteresis, expiry). */
  private detailAcceptedStep = 0;
  /** The last committed plan, its Coarse-suppressed tiles and unconstrained-region count. */
  private detailPlan?: UniformDetailPlan;
  private detailSuppressed?: Uint8Array;
  private detailUnconstrained = 0;
  /** Split pressure's own surface target and centre phi (UniformMixedFrameFields.pressureGeometry). */
  private mixedPressureGeometry?: { target: GPUTexture; centerPhi: GPUTexture };
  /** Splash-survival controls; see the option docs. */
  private phiCubicAdvection: boolean;
  /** Coarse cells whose E1 tables fit the conditioning plane; 0 disables E1. */
  private twoLevelTileCount: number;
  private readonly vertexPhiField?: GPUTexture;
  get vertexPhiTexture(): GPUTexture | undefined {
    if (!this.vertexPhiField) return undefined;
    // A base the detail storage adopted (the published 4h vertex base) is a raw texture to every reader: the field is then its handle.
    const field = this.vertexPhiField, texture = this.live(field);
    return texture === field || uniformDetailField(texture) ? texture : field;
  }
  private readonly fieldPages: UniformTexturePages;
  /** Geometric h fields: base block + patch atlas (uniform-detail-fields.ts). */
  private readonly detailStorage?: UniformDetailStorage;
  private readonly scratchArena: UniformScratchArena;
  private readonly address: UniformLatticeAddress;
  /** The texture a consumer binds: a detail field's current physical texture
   * (replaced when the atlas grows, so resolved at each read). */
  private live(texture: GPUTexture): GPUTexture { return uniformDetailField(texture)?.storage.physical(texture) ?? texture; }
  private readonly executionDenseLevelSetVolumeSource?: DenseLevelSetVolumeConsumerSource;
  private liveSource?: { from: DenseLevelSetVolumeConsumerSource; vertexPhi: GPUTexture; source: DenseLevelSetVolumeConsumerSource };
  get denseLevelSetVolumeSource(): DenseLevelSetVolumeConsumerSource | undefined {
    // The frame re-creates its presented phi with the h-tile capacity; consumers compare the source object.
    // So does the detail vertex field: named only while the frame holds h-tile capacity.
    if (this.mixedSource && this.mixedFrame) {
      const phi = this.mixedFrame.presentation.phi, detail = this.mixedFrame.ownership.capacity.fineTiles > 0 ? this.mixedSource.vertexPhi : undefined;
      if (this.mixedSource.mixedPressurePhi !== phi || this.mixedSource.detailVertexPhi !== detail)
        this.mixedSource = { ...this.mixedSource, mixedPressurePhi: phi, detailVertexPhi: detail };
    }
    const from = this.mixedSource ?? this.executionDenseLevelSetVolumeSource;
    if (!from || !this.detailStorage) return from;
    const vertexPhi = this.live(from.vertexPhi);
    if (this.liveSource?.from !== from || this.liveSource.vertexPhi !== vertexPhi)
      this.liveSource = { from, vertexPhi, source: vertexPhi === from.vertexPhi ? from : { ...from, vertexPhi, detailVertexPhi: from.detailVertexPhi && vertexPhi, openFraction: this.live(from.openFraction) } };
    return this.liveSource.source;
  }
  private readonly vertexPhiScratch?: GPUTexture;
  /** Most recent transported phi, before closest-point redistancing (diagnostics). */
  get advectedVertexPhiTexture(): GPUTexture | undefined { return this.vertexPhiScratch && this.live(this.vertexPhiScratch); }
  /** The geometric t=0 surface publication; see encodeInitialPresentationSurface. */
  private initialPublishPipeline?: GPUComputePipeline;
  private readonly initialDetailGroups: () => { authority: GPUBindGroup; publish: GPUBindGroup };
  private readonly shaderSource: string;
  private readonly groupDescriptors = new WeakMap<GPUBindGroup, GPUBindGroupDescriptor>();
  private readonly executionVolumeTexture: GPUTexture;
  get volumeTexture(): GPUTexture { return this.live(this.executionVolumeTexture); }
  get surfaceFieldTexture(): GPUTexture {
    return this.live(this.surfaceB);
  }
  private readonly executionColumnBaseTexture: GPUTexture;
  get columnBaseTexture(): GPUTexture { return this.executionColumnBaseTexture; }
  private readonly executionVelocityTexture: GPUTexture;
  get velocityTexture(): GPUTexture { return this.live(this.executionVelocityTexture); }
  /** Velocity after advection/forces and before the pressure projection. */
  get preProjectionVelocityTexture(): GPUTexture { return this.live(this.velocityB); }
  /** Padded float32 velocity extension retained for opt-in comparison diagnostics. */
  private readonly executionExtrapolatedVelocityTexture?: GPUTexture;
  /** Undefined on Geometric: the mixed extension publishes no padded field. */
  get extrapolatedVelocityTexture(): GPUTexture | undefined { return this.executionExtrapolatedVelocityTexture; }
  private stageDiagnosticsRequested = false;
  private readonly retainStageDiagnosticsForQA: boolean;
  /** Full FIM diagnostics must be retained when constructing the solver. */
  enableStageDiagnosticsForQA(): void {
   
    if(this.stageDiagnosticsRequested)return;
    if(!this.retainStageDiagnosticsForQA)throw new Error("Construct the solver with retainStageDiagnosticsForQA to inspect intermediate fields");
    this.stageDiagnosticsRequested=true;
  }
  readonly extrapolationActiveFrontPassCeiling: number;
  /** The mixed extension's sweeps per encode, within the ceiling. */
  private extensionSweeps: number;
  private setExtensionSweeps(passes: number): void {
    this.extensionSweeps = Number.isFinite(passes)
      ? Math.min(this.extrapolationActiveFrontPassCeiling, Math.max(1, Math.round(passes))) : this.extrapolationActiveFrontPassCeiling;
  }
  private readonly symmetryStageAuditFields?: WebGPUUniformReferenceSolver["symmetryStageAuditTextures"];
  readonly symmetryStageAuditTextures?: Readonly<{
    preExtrapolationVelocity: GPUTexture;
    previousRawDensity: GPUTexture;
    extrapolationDensityAuthority: GPUTexture;
    densityAdvection: GPUTexture;
    densityDiffusion: GPUTexture;
    gammaPostAdvection: GPUTexture;
    gammaPostDiffusion: GPUTexture;
    velocityPrediction: GPUTexture;
    predictedExtrapolation: GPUTexture;
    reverseAdvection: GPUTexture;
    velocityAdvection: GPUTexture;
    pressureProjection: GPUTexture;
  }>;
  /** Negative domain MAC faces paired with preExtrapolationVelocity. */
  readonly symmetryStageAuditNegativeBoundaryVelocity?: GPUBuffer;
  /** Scalar-packed x/y/z negative-face plane byte length. */
  private readonly executionNegativeBoundaryVelocityBytes: number;
  get negativeBoundaryVelocityBytes(): number { return this.executionNegativeBoundaryVelocityBytes; }
  /** Negative x/y/z domain MAC faces paired with velocityTexture. */
  get negativeBoundaryVelocityBuffer(): GPUBuffer { return this.boundaryVelocityA; }
  /**
   * Read-only accepted pressure/gamma for matched-lattice Dawn comparisons.
   *
   * With the pressure lattice planned on the window the texture is
   * window-local: it has `latticeDimensions` haloed cells and its cell
   * (i,j,k) is simulation cell (i,j,k) - 1 + `latticeOrigin`. A caller that
   * assumes the domain plus a one-cell halo must apply that or keep the
   * lattice off.
   */
  get gridVelocityBoundary(): GPUBufferBinding { return { buffer: this.boundaryVelocityA }; }
  get gridPressureTexture(): GPUTexture { return this.pressureMultigrid.pressureTexture; }
  get gridPressureOrigin(): readonly [number, number, number] { return [0, 0, 0]; }
  /** Stable pool publication plus the live GPU frame; no per-frame allocation. */
  private mixedPageSource?: import("../../core/method-view-records").GPUFluidVolumePageSource;
  get volumePageSource() { return this.mixedPageSource; }
  /** The bed-slope lane of the solid scratch (uniformSolidScratchLayout): `words` u32 from word `baseWord` of `buffer`. The bake writes it; a probe reads it back or zeroes it. */
  get solidBedSlopeLane(): { buffer: GPUBuffer; baseWord: number; words: number } {
    return { buffer: this.activeScratch, baseWord: this.solidBedSlopeBaseWord, words: this.solidBedSlopeWords };
  }
  get physicsFieldsForQA(): {pressure:GPUTexture;gamma:GPUTexture;latticeOrigin:[number,number,number];latticeDimensions:[number,number,number]} {
    return { pressure: this.pressureMultigrid.pressureTexture, gamma: this.live(this.gammaA),
      latticeOrigin: [0, 0, 0],
      latticeDimensions: [this.executionInfo.nx + 2, this.executionInfo.ny + 2, this.executionInfo.nz + 2] };
  }
  /** Eight vec4 decision records for every stored MAC face/component. */
  readonly symmetryStageAuditMacCormackBuffer?: GPUBuffer;
  /** Fixed-point beta produced by Sec. 3.4 before deficit scattering. */
  readonly symmetryStageAuditBetaBuffer?: GPUBuffer;

  private readonly velocityA: GPUTexture;
  private readonly velocityB: GPUTexture;
  private readonly velocityC: GPUTexture;
  private readonly velocityD: GPUTexture;
  private readonly boundaryVelocityA: GPUBuffer;
  private readonly boundaryVelocityB: GPUBuffer;
  private readonly boundaryVelocityC: GPUBuffer;
  private readonly boundaryVelocityD: GPUBuffer;
  private readonly pressureA: GPUTexture;
  private readonly pressureB: GPUTexture;
  private readonly volumeA: GPUTexture;
  private readonly volumeB: GPUTexture;
  private readonly surfaceA: GPUTexture;
  private readonly surfaceB: GPUTexture;
  private readonly gammaA: GPUTexture;
  private readonly gammaB: GPUTexture;
  private readonly heightA: GPUTexture;
  private readonly heightB: GPUTexture;
  private readonly terrainTexture: GPUTexture;
  private readonly transportA: GPUTexture;
  private readonly transportB: GPUTexture;
  private readonly velocityExtrapolator: WebGPUUniformVelocityExtrapolator;
  private readonly pressureMultigrid: WebGPUUniformPressureMultigrid;
  // Origin(3), capacity(3), mode(1).
  private readonly params: GPUBuffer;
  private readonly solidBedSlopeBaseWord: number;
  private readonly solidBedSlopeWords: number;
  private readonly reductions: GPUBuffer;
  private readonly conditioningScratch: GPUBuffer;
  /** The page table and block words the native kernels address through: a
   * mirror of the mixed frame's page pool, the seed's until then. */
  private readonly pages: GPUBuffer;
  private readonly activeScratch: GPUBuffer;
  private readonly solidBoundaryCapacity: number;
  /** Writes the scratch's bed-slope lane from its voxel mask: whole at build, around an edit after it. */
  private bedSlopeBake?: UniformBedSlopeBake;
  /** An open world's ground: the window's terrain columns and the census's solid tiles past the box (UniformOpenWorldTerrain). */
  private readonly openWorldTerrain?: UniformOpenWorldTerrain;
  /** Distinct storage binding for the compile-time-disabled MacCormack audit.
   * WebGPU still validates writable binding aliasing even when the shader's
   * constant-false branch cannot execute. */
  private readonly macCormackAuditBinding: GPUBuffer;
  private readonly rigidExchange: GPUBuffer;
  private readonly rigidSystem: WebGPURigidBodySystem;
  private readonly mainLayout: GPUBindGroupLayout;
  private readonly mainPipelineLayout: GPUPipelineLayout;
  private readonly extrapolationAuthorityGroup: GPUBindGroup;
  private readonly wallFilmResolveGroup: GPUBindGroup;
  private pipelines?: UniformReferencePipelines;
  private inflowBoundary?: InflowGridBoundary;
  private statsReadback?: GPUBuffer;
  private readbackPending = false;
  private lastTime = 0;
  private referenceVolumeCells = 0;
  /** A ball waiting for a step to wet its cells. See `injectLiquidBall`. */
  private pendingDrop?: UniformLiquidDrop;
  /** A step has asked for the pending ball's pages: it lands on the next one. */
  private pendingDropPaged = false;
  /** The drop this step adds: each census of the step asks the page pool for its pages. */
  private pageDrop?: UniformLiquidDrop;
  /** The scheduled drop the next step adds (uniformPagePoolDropAhead): asked for with this step's, so its pages are stored when its own step's census runs. */
  private pageDropAhead?: UniformLiquidDrop;
  /** The injected ball still waiting to land, asked for the same way. */
  private pageInjectedAhead?: UniformLiquidDrop;
  /** The scene's stored set at build: which pages the address seeded and the pool pins. */
  private readonly pageResidency: UniformPagePoolResidency;
  /** The initial liquid the residency was computed from, held for the upload that follows it. */
  private initialVolume?: ReturnType<typeof uniformLatticeInitialVolume>;
  /** The container's cells: what its extents are divided by for a cell's size. The lattice (executionInfo.nx, ny, nz) is these rounded up to whole pages. */
  private readonly containerCells: readonly [number, number, number];
  /** The scene's fluid frame (sceneFluidFrame), memoised by scene document. */
  private frameOf?: { readonly scene: SceneDescription; readonly frame: SceneFluidFrame };
  private get frame(): SceneFluidFrame {
    if (this.frameOf?.scene !== this.scene) this.frameOf = { scene: this.scene, frame: sceneFluidFrame(this.scene) };
    return this.frameOf.frame;
  }
  /** The scene's scheduled drops in source form, built once per schedule. */
  private scheduledDrops?: { readonly schedule: readonly ScheduledLiquidDrop[]; readonly drops: readonly UniformLiquidDrop[] };
  private densityPostProcessing: boolean;
  private gammaDiffusionIterations: number;
  private solidExcessCorrection: boolean;
  private rigidCoupling: boolean;
  private readonly pressureSchedule: UniformCM11aSchedule;
  /** Mixed frames submitted whose receipt is unhandled. */
  private mixedFramesInFlight = 0;
  /** The seeded velocity took its half-step kick (UniformMixedFrame.kick), ahead of the first frame. */
  private mixedKicked = false;
  /** Settles once every mixed frame submitted so far is checked and adopted. */
  private mixedFrameChain: Promise<void> = Promise.resolve();
  /** The GPU relayout is attached (UniformMixedFrame.setRelayout): a frame
   * head runs the census, builder and adopt. Dynamic: every frame, over
   * every tile. Requested and Full: requests only (syncMixedLayout), while
   * they request an h tile. No host layout runs until it is detached. */
  private mixedGpuLayout = false;
  /** Free relayout receipt reads, and the one this frame's head copied into. */
  private readonly mixedRelayoutReads = new KeptStack<GPUBuffer>();
  private mixedRelayoutRead?: GPUBuffer;
  /** Diagnostics sequence: an older read never overwrites a newer one. */
  private mixedRelayoutSequence = 0;
  private mixedRelayoutShown = 0;
  /** Rigid bodies on the mixed frame (coupling and census body tiles). */
  private mixedBodies?: UniformMixedBodies;
  /** The last advance passed bodies: the next one rebuilds the solid record without them. */
  private mixedHadBodies = false;
  private pressureFrameFailure?: Error;
  private deferredFrameScene?: SceneDescription;
  private deferredFrameValues?: MethodParamValues;
  private deferredFrameBodies?: RigidBodyState[];
  /** Host-side cycle budgeting; "fixed" reproduces the pre-P1 command stream. */
  private pressureCycleBudgetLagged: boolean;
  /** No lattice cell (halo excluded) set in the static solid voxel mask; see uvSolidFree. */
  private solidVoxelsEmpty = true;
  private readonly prescribedSolidMotion = new UniformPrescribedSolidMotion();
  /** The pinned advance of a fixed-step mode; undefined runs the scene's maxDt. */
  private fixedStep_s: number | undefined;
  private velocityTransport: GPUVelocityTransport;
  private liquidOnlyVelocityAdvection: boolean;
  private disposed = false;
  private physicsTraceSampleId = 0;
  private physicsTracePending = false;
  private lastPhysicsTraceAt_ms = 0;
  /** CPU mirror of the GPU solid bitmask, so a scene edit uploads only the words its pages changed. */
  private readonly solidMask: SolidOccupancyMask;

  private constructor(
    private readonly device: GPUDevice,
    public scene: SceneDescription,
    quality: GPUQuality,
    _onRigidLoads?: (loads: GPURigidLoad[]) => void,
    options: WebGPUUniformReferenceOptions = {},
  ) {


    this.narrowBandAdaptiveBudgetPercent = Math.max(0,Math.min(100,options.narrowBandAdaptiveBudgetPercent??50));
    this.narrowBandAdaptiveFadeSeconds = Math.max(0.05,Math.min(2,options.narrowBandAdaptiveFadeSeconds??0.5));
    this.narrowBandSparseParticleLifetime = Math.max(0,Math.min(10,options.narrowBandSparseParticleLifetime??2));
    this.narrowBandVolumeControlSeconds = Math.max(0,Math.min(2,options.narrowBandVolumeControlSeconds??0.1));

    this.narrowBandFinePadding = options.narrowBandFinePadding ?? 1;
    this.narrowBandTransfer = options.narrowBandTransfer ?? "flip";
    this.phiFineTrilinear = options.phiFineTrilinear === true;
    this.twoLevelVelocity = true;
    this.phiCubicAdvection = options.phiCubicAdvection !== false;
    this.retainStageDiagnosticsForQA=options.retainStageDiagnosticsForQA===true;

    this.densityPostProcessing = options.densityPostProcessing === true;
    this.gammaDiffusionIterations = Math.round(Math.min(UNIFORM_GAMMA_DIFFUSION_MAX_ITERATIONS,
      Math.max(0, options.gammaDiffusionIterations ?? UNIFORM_GAMMA_DIFFUSION_DEFAULT_ITERATIONS)));
    this.solidExcessCorrection = options.solidExcessCorrection !== false;
    this.rigidCoupling = options.rigidCoupling !== false;
    this.pressureSchedule = options.pressureSchedule ?? {
      fullCycles: UNIFORM_CM11A_FULL_CYCLES,
      vCycles: UNIFORM_CM11A_V_CYCLES,
      preSweeps: UNIFORM_CM11A_PRE_SWEEPS,
      postSweeps: UNIFORM_CM11A_POST_SWEEPS,
      residualTolerance: UNIFORM_PRESSURE_RESIDUAL_TOLERANCE,
    };
    this.pressureCycleBudgetLagged = options.pressureCycleBudget !== "fixed";
    this.fixedStep_s = uniformFixedStep_s(options.timeStep);
    this.velocityTransport = options.velocityTransport === "maccormack"
      ? "maccormack" : "semi-lagrangian";
    this.liquidOnlyVelocityAdvection = options.liquidOnlyVelocityAdvection === true;
    // The stage trace's closing marker pass must dispatch observable work, or
    // Metal skips its end-of-pass timestamp and the first sample retires
    // hardware tracing for this solver. Compile it long before the panel asks.
    void GPUStageTimestampRecorder.prepare(device);
    const cells = this.containerCells = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
    // The lattice is the container's cells rounded up to whole pages; what it adds past the container's high faces is solid or air (solidWorldForFluidLattice).
    const [nx, ny, nz] = sceneFluidLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
    // Every stage addresses whole 4h tiles: a lattice that is not a multiple of four on each axis is refused here.
    // The one lattice address of the scene: its page pool's seed names what is stored, and every layout and stage is handed this object.
    this.inflowBoundary = scene.fluid.inflow
      ? createInflowGridBoundary(scene.fluid.inflow, scene.container, cells) : undefined;
    this.detail = options.detail ?? uniformDetailSettings({});
    // The stored set at build (uniformPagePoolResidency): under the census of the Dynamic policy only the sources' pages are pinned.
    // The box's place in the window: its origin, but in an open world (container.walls "none").
    const boxPage = sceneFluidBoxPage(scene);
    const residency = this.pageResidency = uniformPagePoolResidency([nx, ny, nz], scene.fluid.pagePool,
      uniformPageSources(scene, this.inflowBoundary, Math.max(this.fixedStep_s ?? 0, scene.numerics.maxDt_s)), this.detail.policy === "dynamic",
      () => (this.initialVolume = uniformLatticeInitialVolume(scene)).volume, boxPage);
    // The frame is the page table's window: outside the box it is SOLID, but
    // for the column above an open top, which is AIR and claimable, and but in
    // an open world, where every page above the floor is (uniformLatticeAddress).
    const address = this.address = uniformLatticeAddress([nx, ny, nz], scene.container.top === "open", { ...scene.fluid.pagePool, air: residency.air },
      scene.container.walls === "none" ? { boxPage } : undefined);
    // The mixed-ownership frame owns every geometric advance, on native fields over the shared scratch arena.
    this.scratchArena = new UniformScratchArena(device, address, address.boxCells*UNIFORM_VOLUME_EDGE_BYTES,options.retainStageDiagnosticsForQA,undefined,uniformMixedPressureRootBytes(address));
    this.fieldPages = new UniformTexturePages(device, this.scratchArena);
    {
      if (Math.max(nx,ny,nz)+2 > device.limits.maxTextureDimension3D)
        throw new Error("Uniform Geometric finest lattice exceeds the device texture limit");
      if (address.boxCells*24 > Math.min(device.limits.maxStorageBufferBindingSize,device.limits.maxBufferSize))
        throw new Error("Uniform Geometric donor sums exceed the device storage limit");
    }
    const tileRecords = address.tileKeys;
    // The 4h face table and class map sit in the second conditioning plane,
    // whose per-step clears are ranged away from it. Same size test, plus the
    // shell counter above the two ping-pong planes. The sampler's convention
    // is "fine face 4t+3 is coarse face t", which is what the extension
    // hierarchy's own transfer computes because the coarse level tiles the
    // lattice exactly.
    this.twoLevelTileCount = UNIFORM_VOLUME_TWO_LEVEL_WORDS_PER_TILE * tileRecords
        + UNIFORM_VOLUME_TWO_LEVEL_COUNTER_WORDS <= address.cellKeys ? tileRecords : 0;
    // One tile covers the finest trilinear tap, which reaches one cell below a
    // fine tile. The FIM's accurate band is two of the LARGEST cells, so on an
    // anisotropic lattice it spans 2*max(h)/min(h) of the smallest; the shell
    // must contain that too, measured from a seed tile's own boundary.
    const spacing = [scene.container.width_m / cells[0], scene.container.height_m / cells[1],
      scene.container.depth_m / cells[2]];
    this.twoLevelShellReach = Math.max(1, Math.ceil(0.5 * Math.max(...spacing) / Math.min(...spacing)));
    const allocation = planUniformHostAllocation(address, "maccormack");
    this.executionNegativeBoundaryVelocityBytes = allocation.boundaryVelocityBytes;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
      | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    const texture3d = (label: string, format: GPUTextureFormat, size: GPUExtent3D) =>
      this.fieldPages.createTexture({ label, size, dimension: "3d", format, usage });
    const velocity = (label: string) => texture3d(label, "rgba32float", allocation.velocityExtent);
    // Geometric h fields come from the detail storage, registered with the
    // page layer by their logical dims; their bytes are the storage's (refreshMixedAllocation).
    const detailStorage = this.detailStorage = new UniformDetailStorage(device, address);
    // Vertex phi where nothing is stored: the initial level set's value far from liquid (uniformLatticeInitialPhi).
    const emptyPhi = uniformLatticeFarPhi(scene);
    const field = (label: string, kind: UniformDetailClass) =>
      this.fieldPages.adopt(detailStorage.createField(label, kind, kind === "face" ? "rgba32float" : "r32float", false, kind === "vertex" ? emptyPhi : 0), kind === "vertex" ? address.vertexBox : address.cells);
    const hVelocity = (label: string) => field(label, "face");
    const hScalar = (label: string) => field(label, "cell");
    this.velocityA = hVelocity("Uniform reference velocity A");
    this.velocityB = hVelocity("Uniform reference velocity B");
    const lazyMac = this.velocityTransport !== "maccormack";
    this.velocityC = lazyMac ? texture3d("Uniform reference unused velocity C", "rgba32float", [1,1,1]) : velocity("Uniform reference velocity C");
    this.velocityD = hVelocity("Uniform reference velocity D");
    const boundaryVelocity = (label: string, size = allocation.boundaryVelocityBytes) => device.createBuffer({ label, size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.boundaryVelocityA = boundaryVelocity("Uniform reference negative boundary velocity A");
    this.boundaryVelocityB = boundaryVelocity("Uniform reference negative boundary velocity B");
    // Read only by legacy (non-mixed) groups: Geometric binds a placeholder.
    this.boundaryVelocityC = boundaryVelocity("Uniform reference negative boundary velocity C", 16);
    this.boundaryVelocityD = boundaryVelocity("Uniform reference negative boundary velocity D");

    // Geometric uses CM11a's own pressure hierarchy. Legacy density diffusion
    // is disabled, so the compatibility pressure slots need only valid views.
    const pressure=(label:string)=>texture3d(label,"r32float",[1,1,1]);
    this.pressureA = pressure("Uniform reference pressure A");
    this.pressureB = pressure("Uniform reference pressure B");
    this.volumeA = hScalar("Uniform reference volume A");
    this.volumeB = hScalar("Uniform reference volume B");
    this.surfaceA = hScalar("Uniform reference smoothed surface A");
    this.surfaceB = hScalar("Uniform reference smoothed surface B");
    this.gammaA = hScalar("Uniform reference transport gamma A");
    this.gammaB = hScalar("Uniform reference transport gamma B");
    {
      this.vertexPhiField = field("Uniform Geometric vertex phi", "vertex");
      this.vertexPhiScratch = field("Uniform Geometric vertex phi scratch", "vertex");
      const edgeBytes = address.boxCells * UNIFORM_VOLUME_EDGE_BYTES;
      if (edgeBytes > device.limits.maxStorageBufferBindingSize || edgeBytes > device.limits.maxBufferSize)
        throw new Error(`Uniform Geometric receiver stencils require ${edgeBytes} bytes, exceeding the device limit`);
      this.executionDenseLevelSetVolumeSource = { vertexPhi: this.vertexPhiField, openFraction: this.gammaB,
        cellSize_m: [scene.container.width_m/cells[0], scene.container.height_m/cells[1], scene.container.depth_m/cells[2]] };
    }
    this.heightA = device.createTexture({ label: "Uniform reference column base", size: [nx, nz], format: "rg32float", usage });
    this.heightB = device.createTexture({ label: "Uniform reference column occupancy", size: [nx, nz], format: "rg32float", usage });
    // An open world's ground runs on past the box: its columns are the window's, baked on the GPU (UniformOpenWorldTerrain).
    const openWorldTerrain = scene.container.walls === "none" && sceneHasTerrain(scene);
    this.terrainTexture = device.createTexture({ label: "Uniform reference terrain",
      size: openWorldTerrain ? [UNIFORM_OPEN_WORLD_TERRAIN_COLUMNS, UNIFORM_OPEN_WORLD_TERRAIN_COLUMNS] : [nx, nz], format: "r32float", usage });
    const transportUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
      | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    // The mixed frame reads neither padded transport field: Geometric binds placeholders.
    this.transportA = this.fieldPages.createTexture({ label: "Uniform reference transport A", size: [1,1,1], dimension: "3d", format: "rgba32float", usage: transportUsage });
    this.transportB = this.fieldPages.createTexture({ label: "Uniform reference transport B", size: lazyMac ? [1,1,1] : allocation.transportExtent, dimension: "3d", format: "rgba32float", usage: transportUsage });

    this.params = device.createBuffer({ label: "Uniform reference parameters", size: UNIFORM_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.solidMask = new SolidOccupancyMask(uniformSolidMaskPages(address));
    this.solidMask.update(solidWorldForFluidLattice(scene));
    // No cell off the lattice's walls is solid: a tank's shell lies in wall pages and is not counted.
    this.solidVoxelsEmpty = this.solidMask.empty;
    this.pages = device.createBuffer({ label: "Uniform reference page table", size: address.seed.tail.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.pages, 0, address.seed.tail);
    // The solid mask's image whole (SolidOccupancyMask): the page directory,
    // the bed-slope lane (two bytes a haloed cell, zero until the frame's bake
    // writes it at its build: every bed level; see bedSlopeBake), the mixed
    // solid library's tile cut map, one word per tile key and all cut until
    // built, and the page slots. uniformSolidScratchLayout has the lane's and
    // the cut map's offsets, as umBedSlope and umSolidCutMap derive them. Sized
    // once; a voxel edit uploads the directory words and the slots it rewrote,
    // so the lane keeps its place and its size.
    const solidScratch = uniformSolidScratchLayout([nx, ny, nz]);
    this.solidBedSlopeBaseWord = solidScratch.bedSlopeBaseWord;
    this.solidBedSlopeWords = solidScratch.bedSlopeWords;
    const boundaryField = proceduralSolidField(scene), boundaryImage = boundaryField?.pack();
    if (boundaryField && (boundaryField.source.dimensions.some((v,a)=>v!==this.containerCells[a])
      || boundaryField.source.cellSize_m.some((v,a)=>Math.abs(v-[scene.container.width_m/this.containerCells[0],scene.container.height_m/this.containerCells[1],scene.container.depth_m/this.containerCells[2]][a]!)>1e-9)))
      throw new Error("Procedural boundary lattice changed; regenerate the procedural solid at this resolution");
    this.solidBoundaryCapacity = boundaryImage?.length ?? 0;
    this.activeScratch = device.createBuffer({
      label: "Uniform reference solid page mask, bed-slope lane and tile cut map",
      size: this.solidMask.words.byteLength + 4 * (this.solidBoundaryCapacity + 2),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    // Around the lane: the buffer's zeros are the lane's until the bake, and
    // the host's copy of those words is never touched.
    const solidImage = this.solidMask.layout;
    device.queue.writeBuffer(this.activeScratch, 0, this.solidMask.words, 0, solidImage.lane);
    device.queue.writeBuffer(this.activeScratch, 4 * solidImage.cutMap, this.solidMask.words, solidImage.cutMap);
    if (openWorldTerrain) this.openWorldTerrain = new UniformOpenWorldTerrain(device,
      { columns: this.terrainTexture, scratch: this.activeScratch, cutMapWord: solidScratch.cutMapOffsetWords, tileKeys: address.tileKeys });
    if (boundaryImage) {
      this.uploadSolidBoundary(boundaryImage);
      device.queue.writeBuffer(this.activeScratch, this.activeScratch.size - 8, new Uint32Array([this.solidMask.words.length, PROCEDURAL_SOLID_MAGIC]));
    }
    const source = createUniformReferenceComputeShader();
    const fixedFields = new Map([
      [0,this.velocityA],[1,this.velocityB],[3,this.pressureB],[4,this.volumeA],[5,this.volumeB],
      [12,this.velocityC],[13,this.velocityD],[14,this.transportA],[16,this.surfaceA],
      [20,this.surfaceA],[24,this.gammaA],[25,this.gammaB],[31,this.vertexPhiField!],[32,this.vertexPhiScratch!],
    ]);
    // These bindings always hold retained native textures, even when ping-pong
    // groups change. Keep their interpolation free of scratch address branches.
    const nativeBindings = new Set([0,1,2,3,4,5,12,13,14,16,20,21,24,25,31,32]);
    this.shaderSource = this.fieldPages.shader(source,fixedFields,true,nativeBindings);
    this.conditioningScratch = device.createBuffer({ label: "Uniform reference compatibility scratch", size: this.scratchArena.conditioningBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.velocityExtrapolator = new WebGPUUniformVelocityExtrapolator(device, this.address, this.params);
    // Without a ceil(n/4) hierarchy level there is no 4h field to sample.
    if (!this.velocityExtrapolator.coarseVelocityTableAvailable) this.twoLevelTileCount = 0;
    if (this.twoLevelTileCount === 0) this.twoLevelVelocity = false;
    // Sixteen wavefronts cover the complete 26-neighbour dependency diameter of the two-cell accurate band.
    this.extrapolationActiveFrontPassCeiling = Math.min(Math.max(nx, ny, nz), 16);
    this.extensionSweeps = this.extrapolationActiveFrontPassCeiling;
    if (options.extensionFrontSweeps !== undefined) this.setExtensionSweeps(options.extensionFrontSweeps);
    this.reductions = device.createBuffer({ label: "Uniform reference diagnostics and volume control", size: 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });


    this.macCormackAuditBinding = this.symmetryStageAuditMacCormackBuffer ?? device.createBuffer({
      label: "Uniform disabled MacCormack audit binding",
      // One vec4 is the minimum valid runtime-array storage resource. It is
      // never accessed in this shader variant and must not alias binding 19.
      size: 16,
      usage: GPUBufferUsage.STORAGE,
    });
    this.rigidExchange = device.createBuffer({ label: "Uniform reference rigid exchange", size: GPU_RIGID_EXCHANGE_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.rigidSystem = new WebGPURigidBodySystem(device, scene, this.rigidExchange,
      this.terrainTexture, [cells[0], cells[2]]);
    const initialRigidBodies = initializeRigidBodies(scene.rigidBodies);
    this.prescribedSolidMotion.sample(initialRigidBodies, 0);
    this.rigidSystem.syncBodies(initialRigidBodies);

    const geometricLayout: GPUBindGroupLayoutEntry[] = [
      { binding: 31, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 32, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
    ];
    const mainEntries: GPUBindGroupLayoutEntry[] = [
      ...geometricLayout,
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rg32float", viewDimension: "2d" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 12, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 13, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 14, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 15, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
      { binding: 16, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 19, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 20, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 21, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      { binding: 24, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 25, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 26, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 27, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 28, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 29, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 30, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ];
    mainEntries.push(...this.fieldPages.layout([]));
    this.mainLayout = device.createBindGroupLayout({ entries: mainEntries });
    this.pressureMultigrid = new WebGPUUniformPressureMultigrid(this.pressureSchedule,
      scene.container.depthBoundary !== "symmetry", this.fieldPages);
    this.mainPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.mainLayout] });
    const sampler = device.createSampler({ minFilter: "linear", magFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
    let physicalViews = false;
    const view = (texture: GPUTexture) => this.fieldPages.view(texture, physicalViews);
    const group = (velocityIn: GPUTexture, velocityOut: GPUTexture, pressureIn: GPUTexture,
      pressureOut: GPUTexture, volumeIn: GPUTexture, volumeOut: GPUTexture,
      heightIn: GPUTexture, heightOut: GPUTexture, predicted = velocityIn,
      reversed = velocityIn, transport = this.transportA, surface = volumeIn,
      gammaRead = this.gammaA, gammaWrite = this.gammaB,
      boundaryRead = this.boundaryVelocityA, boundaryWrite = this.boundaryVelocityB,
      velocityPhase = volumeIn, reversePhi = false) => this.createPageAwareGroup({
        layout: this.mainLayout, entries: ([
          ...([
            { binding: 31, resource: view((reversePhi ? this.vertexPhiScratch! : this.vertexPhiField!)) },
            { binding: 32, resource: view((reversePhi ? this.vertexPhiField! : this.vertexPhiScratch!)) },
          ]),
          { binding: 0, resource: view(velocityIn) }, { binding: 1, resource: view(velocityOut) },
          { binding: 2, resource: view(pressureIn) }, { binding: 3, resource: view(pressureOut) },
          { binding: 4, resource: view(volumeIn) }, { binding: 5, resource: view(volumeOut) },
          { binding: 6, resource: { buffer: this.params } }, { binding: 7, resource: view(heightIn) },
          { binding: 8, resource: view(heightOut) }, { binding: 9, resource: { buffer: this.reductions } },
          { binding: 10, resource: { buffer: this.rigidSystem.stateBuffer } }, { binding: 11, resource: { buffer: this.rigidExchange } },
          { binding: 12, resource: view(predicted) }, { binding: 13, resource: view(reversed) },
          { binding: 14, resource: view(transport) }, { binding: 15, resource: sampler },
          { binding: 16, resource: view(velocityPhase) },
          { binding: 19, resource: { buffer: this.conditioningScratch } },
          { binding: 20, resource: view(surface) }, { binding: 21, resource: view(this.terrainTexture) },
          { binding: 24, resource: view(gammaRead) }, { binding: 25, resource: view(gammaWrite) },
          { binding: 26, resource: { buffer: boundaryRead } }, { binding: 27, resource: { buffer: boundaryWrite } },
          { binding: 28, resource: { buffer: this.macCormackAuditBinding } },
          { binding: 29, resource: { buffer: this.pages } },
          { binding: 30, resource: { buffer: this.activeScratch } },
        ] as GPUBindGroupEntry[]),
      });
    this.extrapolationAuthorityGroup = group(
      this.velocityA, this.velocityD, this.pressureA, this.pressureB,
      this.volumeA, this.surfaceA, this.heightB, this.heightA,
    );
    this.wallFilmResolveGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.surfaceB, this.heightB, this.heightA);
    // Packed detail storage: the two t=0 kernels' groups over the fields'
    // physical textures, built at each use (the atlas may have grown).
    this.initialDetailGroups = () => {
      physicalViews = true;
      try {
        return {
          authority: group(this.velocityA, this.velocityD, this.pressureA, this.pressureB, this.volumeA, this.surfaceA, this.heightB, this.heightA),
          publish: group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.surfaceB, this.heightB, this.heightA),
        };
      } finally { physicalViews = false; }
    };

    const count = address.boxCells;
    // The legacy host plan reserves nine scalars; this owner has eight. Add
    // the extension hierarchy/origins and the two RG columns plus terrain,
    // which that plan does not include. Scratch backing is reconciled below.
    const hostAuxiliaryBytes = this.velocityExtrapolator.scratchBytes - count*4 + nx*nz*20;
    this.executionInfo = {
      nx, ny, nz, tankCells: cells, storedNy: ny, cellCount: count, equivalentUniformCells: count,
      compressionRatio: 1, activeCompressionRatio: 1, activeSampleCount: count,
      regularLayers: ny, maximumNeighborDelta: 0, gridKind: "uniform",
      cellSize_m: Math.min(scene.container.width_m / cells[0], scene.container.height_m / cells[1], scene.container.depth_m / cells[2]),
      pressureIterations: 0, pressureSolver: `CM11a LCP multigrid (${this.pressureSchedule.fullCycles} Full-Cycles + ${this.pressureSchedule.vCycles} V-Cycles, ${this.pressureSchedule.preSweeps}/${this.pressureSchedule.postSweeps} pre/post ${"projected Jacobi; V-first adaptive"})`,
      allocatedBytes: allocation.allocatedBytes - (count*8-8) - (allocation.conditioningBytes-this.scratchArena.conditioningBytes) - (lazyMac ? count*16+(nx+2)*(ny+2)*(nz+2)*16-32 : 0) - ((nx+2)*(ny+2)*(nz+2)*16-16+allocation.boundaryVelocityBytes-16) + 8 + this.pressureMultigrid.allocatedBytes
        + (8 * address.vertexKeys + 12)
        + this.pages.size + this.activeScratch.size
        + hostAuxiliaryBytes + (this.symmetryStageAuditMacCormackBuffer ? 0 : 16), quality,
      submittedTime_s: 0, simulatedTime_s: 0, completedTime_s: 0,
      simulationLag_s: 0, encodedSteps: 0, maximumTallCellHeight: 0,
      volumeControl: false,
      hostFluidAuthority: "gpu-resident", hostSimulationSizedWorkItems: 0,
      uniformPressureCycleBudget: "adaptive",
      hostSchedulingUsesReadback: this.pressureCycleBudgetLagged && this.pressureMultigrid.residualTolerance > 0,
    };
    this.executionVolumeTexture = this.volumeA;
    this.executionColumnBaseTexture = this.heightA;
    this.executionVelocityTexture = this.velocityA;
    // What the overlays and the particles bind (volumeTexture, velocityTexture, preProjectionVelocityTexture, surfaceFieldTexture, the open fraction).
    if(this.symmetryStageAuditFields)this.symmetryStageAuditTextures=Object.freeze(Object.fromEntries(
      Object.entries(this.symmetryStageAuditFields).map(([name,field])=>[name,field]),
    )) as NonNullable<WebGPUUniformReferenceSolver["symmetryStageAuditTextures"]>;
    this.initializeVolumeAndTerrain();
  }

  /** Bound staging just as field-page initialization does. Native Dawn aborts
   * on a single boundary upload larger than 8 MiB (the long slide exceeds it). */
  private uploadSolidBoundary(image: Uint32Array): void {
    const chunkWords = 2 * 1024 * 1024;
    for (let begin = 0; begin < image.length; begin += chunkWords)
      this.device.queue.writeBuffer(this.activeScratch, this.solidMask.words.byteLength + begin * 4,
        image.subarray(begin, begin + chunkWords) as Uint32Array<ArrayBuffer>);
  }

  static async createAsync(
    device: GPUDevice,
    scene: SceneDescription,
    quality: GPUQuality,
    onRigidLoads: ((loads: GPURigidLoad[]) => void) | undefined,
    options: WebGPUUniformReferenceOptions,
    onProgress: GPUInitializationReporter,
    signal: AbortSignal = new AbortController().signal,
  ): Promise<WebGPUUniformReferenceSolver> {
    const scope = gpuInitializationDevice(device, signal);
    device = scope.device;
    const runner = new GPUInitializationTaskRunner(onProgress, signal);
    let solver: WebGPUUniformReferenceSolver | undefined;
    try {
      await runner.run([{ id: "uniform.allocate", phase: "allocation", label: "Allocate uniform reference resources", run: () => {
        solver = new WebGPUUniformReferenceSolver(device, scene, quality, onRigidLoads, { ...options, deferPipelineCompilation: true });
      } }]);
      await runner.run(solver!.initializationTasks(signal));
      await runner.run([{id:"uniform.mixed",phase:"solver-pipelines",label:"Prepare shared mixed ownership",run:async(_signal,report)=>{
        const unsubscribe=gpuCompilationManagerFor(device).subscribe(snapshot=>{
          if(snapshot.progress&&!signal.aborted)report?.(`${snapshot.progress.label}: ${snapshot.progress.current ?? "compiling"} (${snapshot.queued} queued)`);
        });
        try { await solver!.initializeMixedFrame(); } finally { unsubscribe(); }
      }}]);
      scope.commit();
      return solver!;
    } catch (error) {
      scope.rollback();
      solver?.destroy();
      throw error;
    }
  }

  private mixedFrame?: UniformMixedFrame;
  private mixedFrameTrace(): UniformMixedFrameTrace | undefined {
    const instrumentation=usePerformanceInstrumentationStore.getState();
    // The clock is read only for a trace that may start (its reading is a heap number).
    if(!instrumentation.enabled||this.physicsTracePending)return undefined;
    return this.openMixedFrameTrace(performance.now());
  }
  /** mixedFrameTrace for a trace that may start (apart from it: these closures would cost every advance a context). */
  private openMixedFrameTrace(requested:number): UniformMixedFrameTrace | undefined {
    if(requested-this.lastPhysicsTraceAt_ms<UNIFORM_PHYSICS_TRACE_CADENCE_MS)return undefined;
    this.executionInfo.physicsCPUTrace=undefined;
    if(!GPUStageTimestampRecorder.supported(this.device)||!GPUStageTimestampRecorder.markersReady(this.device)){
      this.executionInfo.physicsTrace=undefined;
      this.executionInfo.physicsTraceUnavailable="GPU hardware timing is unavailable. No substitute timing is reported.";
      return undefined;
    }
    const sampleId=++this.physicsTraceSampleId,context=`uniform:sim-${this.lastTime.toFixed(6)}`;
    // Close each submitted command buffer before awaiting its receipt. A
    // timestamp chain spanning submissions would charge CPU waits to pressure.
    const segments:GPUStageTimestampRecorder[]=[];
    let active:GPUStageTimestampRecorder|undefined;
    this.physicsTracePending=true;this.lastPhysicsTraceAt_ms=requested;
    return {
      instrument:encoder=>{
        if(active)throw new Error("Mixed timing segment already open");
        active=new GPUStageTimestampRecorder(this.device,sampleId,"physics",context);active.begin();
        return active.instrument(encoder);
      },
      phase:(encoder,phase)=>active!.completePhase(encoder,phase),
      submit:(encoder,anchor)=>{
        active!.anchorFinalBoundary(anchor);active!.resolve(encoder);segments.push(active!);active=undefined;
      },
      submitted:()=>{
        this.executionInfo.physicsCaptureIdentity={sampleId,context,frameId:gpuPhysicsPerformanceActivityFrameId({sampleId,context})};
        void Promise.all(segments.map(segment=>segment.read())).then(traces=>{
          const current=usePerformanceInstrumentationStore.getState();
          if(this.disposed||!current.enabled||current.enabledAt_ms>requested)return;
          if(traces.some(trace=>!trace)){
            this.executionInfo.physicsTrace=undefined;
            this.executionInfo.physicsTraceUnavailable="GPU hardware timing was invalid. No substitute timing is reported.";
            return;
          }
          const complete=traces.map(trace=>trace!);
          this.executionInfo.physicsTrace={...complete.at(-1)!,total_ms:complete.reduce((sum,trace)=>sum+trace.total_ms,0),phases:complete.flatMap(trace=>trace.phases)};
          this.executionInfo.physicsTraceUnavailable=undefined;
        }).catch(error=>{
          if(!this.disposed){this.executionInfo.physicsTrace=undefined;this.executionInfo.physicsTraceUnavailable=`GPU hardware timing failed: ${String(error)}`;}
        }).finally(()=>{this.physicsTracePending=false;});
      },
      abort:()=>{active?.destroy();segments.forEach(segment=>segment.destroy());this.physicsTracePending=false;},
    };
  }
  private mixedDiagnostics?: UniformMixedDiagnostics;
  private mixedAccountedBytes?:number;
  /** Detail storage bytes in executionInfo: its fields' physical textures, which grow with residency. */
  private detailAccountedBytes=0;
  private refreshMixedAllocation():void{
    if(!this.mixedFrame||this.mixedAccountedBytes===undefined)return;
    const bytes=this.mixedFrame.allocatedBytes,detail=this.detailStorage?.allocatedBytes??0;
    // Bytes that moved may move again (a stage's first launch, a reservation's follow-up): the next advances read too.
    if(bytes!==this.mixedAccountedBytes||detail!==this.detailAccountedBytes)this.mixedAllocationWatch=UNIFORM_MIXED_ALLOCATION_WATCH;
    this.executionInfo.allocatedBytes+=bytes-this.mixedAccountedBytes+detail-this.detailAccountedBytes;this.mixedAccountedBytes=bytes;this.detailAccountedBytes=detail;
    this.mixedAllocationStale=false;this.mixedAllocationRevision=this.mixedFrame.ownership.capacityRevision;this.detailAllocationRevision=this.detailStorage?this.detailStorage.allocationRevision:-1;
  }
  /** `info` sums the frame's and the detail storage's bytes only when they
   * may have moved: an event that reserves, attaches, lays out or fails
   * (watchMixedAllocation), the advances and receipts of the watch after one,
   * another h-tile capacity (capacityRevision) or another set of detail
   * textures (allocationRevision). A read with none of these builds nothing. */
  private mixedAllocationStale=true;
  private mixedAllocationWatch=UNIFORM_MIXED_ALLOCATION_WATCH;
  private mixedAllocationRevision=-1;
  private detailAllocationRevision=-1;
  private watchMixedAllocation():void{this.mixedAllocationStale=true;this.mixedAllocationWatch=UNIFORM_MIXED_ALLOCATION_WATCH;}
  private mixedAllocationDue():boolean{
    const frame=this.mixedFrame;if(!frame||this.mixedAccountedBytes===undefined)return false;
    return this.mixedAllocationStale||frame.ownership.capacityRevision!==this.mixedAllocationRevision||(this.detailStorage?this.detailStorage.allocationRevision:-1)!==this.detailAllocationRevision;
  }
  /** Layout identity of the applied host layout. */
  private mixedRegionKey="";
  /** Identity of the last request the frame refused (its h tiles exceed what
   * the device holds): the accepted generation above keeps running. */
  private mixedRefusedKey="";
  private mixedGeneration=0;
  /** Terrain heights in cells, retained for solid promotion. */
  private mixedTerrainCells?: Float32Array;
  private mixedSolidMaskStamp=0;
  /** The promotion of the mask at this stamp: a scan of the lattice, made again only for another mask. */
  private mixedSolidTiles?:{stamp:number;terrain?:Float32Array;tiles:ReturnType<typeof uniformMixedSolidTiles>&{any:boolean}};
  private mixedSolidPromotion(){
    const terrain=sceneHasTerrain(this.scene)?this.mixedTerrainCells:undefined,held=this.mixedSolidTiles;
    if(held&&held.stamp===this.mixedSolidMaskStamp&&held.terrain===terrain)return held.tiles;
    const solid=uniformMixedSolidTiles(fluidRegionLattice(this.scene).dimensions as [number,number,number],this.solidMask,terrain);
    const tiles={...solid,any:solid.coupled.some(c=>c!==0)};
    this.mixedSolidTiles={stamp:this.mixedSolidMaskStamp,terrain,tiles};
    return tiles;
  }
  private mixedSource?: DenseLevelSetVolumeConsumerSource;
  private async initializeMixedFrame():Promise<void>{
    this.assertMixedOptions();
    if(!this.vertexPhiField||!this.vertexPhiScratch)throw new Error("Mixed Uniform requires the shared native field arena");
    const fine=createUniformMixedLayout(fluidRegionLattice(this.scene),[],1,undefined,this.address);
    const promotion=this.mixedSolidPromotion();
    const detail=this.detailStorage!;
    const pressureGeometry={target:detail.createField("Uniform mixed pressure surface target","cell","r32float",true),centerPhi:detail.createField("Uniform mixed pressure centre phi","cell","r32float",true)};
    this.mixedPressureGeometry=pressureGeometry;
    // Host layouts (Requested, Full) reserve their own h tiles, starting from
    // none; Dynamic holds every tile its page pool can store
    // (address.poolTiles), paid here for the run: no generation has more h
    // tiles than that, so its capacity never moves. No byte budget is
    // passed: WebGPU reports no device memory, so only the device's buffer
    // limits refuse a reservation (UniformMixedFrameCapacity).
    const tiles=this.address.poolTiles,gpuLayout=this.detail.policy==="dynamic";
    const solids=!this.solidVoxelsEmpty||sceneHasTerrain(this.scene),bodies=this.scene.rigidBodies.length;
    this.mixedFrame=new UniformMixedFrame(this.device,fine,{fineTiles:gpuLayout?tiles:0,liquidBand:gpuLayout,particleTransfer:this.narrowBandTransfer,fixedWorldGeometry:true},{
      arena:this.scratchArena,volume:this.volumeA,volumeScratch:this.volumeB,
      velocity:this.velocityA,velocityScratch:this.velocityB,departure:this.velocityD,
      negative:this.boundaryVelocityA,negativeScratch:this.boundaryVelocityB,negativeDeparture:this.boundaryVelocityD,
      phi:this.vertexPhiField,phiScratch:this.vertexPhiScratch,phase:this.surfaceA,centerPhi:this.surfaceB,target:this.gammaB,correction:this.gammaA,
      pressure:this.pressureMultigrid.prepareMixedContinuation(),extension:this.velocityExtrapolator.prepareMixedContinuation(),sourceParams:this.params,
      // Live edits and rigid bodies share the pipelines built at setup.
      solid:{params:this.params,scratch:this.activeScratch,terrain:this.terrainTexture,analyticTerrain:this.openWorldTerrain?.sampleParams,bodies:this.rigidSystem.stateBuffer,coupledTiles:promotion.coupled.reduce((n,c)=>n+c,0),paramsWritten:()=>this.paramsWritten},
      pressureGeometry,detail,pages:{buffer:this.pages,offset:0},residency:this.pageResidency,
    },this.address.openTop,this.pressureSchedule);
    this.mixedFrame.solid!.present=solids||bodies>0;
    this.mixedDiagnostics=new UniformMixedDiagnostics(this.device,this.mixedFrame.ownership,this.volumeA,this.velocityA,this.vertexPhiField,this.reductions);
    // The scene uniform: the census marks this frame's drop and inflow plug band on the GPU.
    this.mixedDynamic=new UniformMixedDynamicClassifier(this.device,this.mixedFrame.ownership,this.volumeA,this.vertexPhiField,this.velocityB,this.params,this.mixedFrame.pagePool,this.mixedFrame.narrowBandFlip.particleTiles,this.address.pages[0]*this.address.pages[1]*this.address.pages[2]);
    this.mixedFrame.registerPageOwner(this.mixedDynamic);this.mixedFrame.registerCertificate(this.mixedDynamic);
    this.mixedBuilder=new UniformMixedLayoutBuilder(this.device,this.mixedDynamic.bandBits,this.mixedFrame.ownership);
    this.mixedBodies=new UniformMixedBodies(this.device,this.mixedFrame.ownership,this.mixedFrame.solid!,{velocity:this.velocityA,phi:this.vertexPhiField,exchange:this.rigidExchange});
    // The frame's forces and its samples read the bed-slope lane; the bake is
    // its only writer. Built with the frame whether or not the scene has a
    // voxel yet: an edit may add the first.
    const bedSlopeBake=new UniformBedSlopeBake(this.device,this.activeScratch,{size:[this.executionInfo.nx,this.executionInfo.ny,this.executionInfo.nz],laneBaseWord:this.solidBedSlopeBaseWord,origin:this.frame.boxCell});
    // One concurrent compile: serial stages left a first load waiting on
    // each uncached pipeline in turn.
    const dynamic=this.mixedDynamic,builder=this.mixedBuilder,mixedBodies=this.mixedBodies;
    await settleGPUInitialization([this.mixedFrame.initialize(),this.mixedDiagnostics.initialize(),
      dynamic.initialize(),builder.initialize(),mixedBodies.initialize(),bedSlopeBake.initialize(),this.openWorldTerrain?.initialize()]);
    // The lane of the mask the scratch holds, before the first frame reads
    // it. An edit taken up while the pipelines compiled is in that mask; one
    // from here on re-bakes around itself (applySceneUniforms).
    const bedSlope=this.device.createCommandEncoder({label:"Uniform bed-slope bake"});
    bedSlopeBake.encode(bedSlope);
    // An open world's ground, before the first layout builds the solid record and cut map from it.
    if(this.openWorldTerrain){const f=this.frame;this.openWorldTerrain.encodeBake(bedSlope,this.scene.terrain!,[f.origin_m[0],f.origin_m[2]],[f.cellSize_m[0],f.cellSize_m[2]],f.cellSize_m[1]);
      this.executionInfo.allocatedBytes+=this.openWorldTerrain.allocatedBytes;}
    this.device.queue.submit([bedSlope.finish()]);
    this.bedSlopeBake=bedSlopeBake;
    this.executionInfo.allocatedBytes+=bedSlopeBake.scratchBytes;
    this.mixedFrame.setLayoutViews(this.layoutViews);
    this.mixedAccountedBytes=this.mixedFrame.allocatedBytes;this.watchMixedAllocation();
    this.executionInfo.allocatedBytes+=this.mixedAccountedBytes+this.mixedDynamic.allocatedBytes+this.mixedBuilder.allocatedBytes+(this.mixedBodies?.allocatedBytes??0);
    this.mixedPageSource={...this.mixedFrame.pagePool.volumePageSource!,frame:{buffer:this.params,size:48}};
    this.mixedSource={vertexPhi:this.vertexPhiField,coarseVertexPhi:this.mixedFrame.coarseVertexPhi,detailVertexPhi:this.mixedFrame.ownership.capacity.fineTiles>0?this.vertexPhiField:undefined,openFraction:this.gammaB,cellSize_m:fine.lattice.cellSize_m,mixedOwnership:this.mixedFrame.ownership.presentation,mixedBlocks:this.address.blocks,mixedFrame:{buffer:this.params,size:48},mixedPressure:this.mixedFrame.presentation.pressure,mixedPressurePhi:this.mixedFrame.presentation.phi,mixedSupport:{buffer:this.mixedFrame.ownership.support}};
    for(let i=0;i<UNIFORM_MIXED_RELAYOUT_READS;i++)this.mixedRelayoutReads.give(this.device.createBuffer({label:`Uniform mixed relayout receipt read ${i}`,size:UNIFORM_MIXED_RELAYOUT_RECEIPT.words*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}));
    this.executionInfo.allocatedBytes+=UNIFORM_MIXED_RELAYOUT_READS*UNIFORM_MIXED_RELAYOUT_RECEIPT.words*4;
    // The initial layout is the host's (authored regions, solids and bodies
    // at h); dynamic coarsening then relayouts it on the GPU every frame.
    // Accessor detail storage holds the t=0 uploads until its first layout is
    // installed, so the t=0 authority and publication follow that layout.
    // They also make the first write of the parameter block, which the first
    // layout already reads: UniformMixedFrame.updateLayout builds the all-4h
    // solid record and the tile cut map from it. Written here, as identity's
    // t=0 publication has by now; built from a zero block the record holds
    // every tile uncut for the life of the frame.
    const deferred=this.detailStorage?.layout.placement==="packed";
    if(deferred)this.writeParams(0,Math.min(this.scene.rigidBodies.length,12),0);
    this.syncMixedLayout();
    // t=0: the uploaded phi on the initial layout (a refused or unchanged
    // layout encodes no relayout publish).
    this.mixedFrame.publishCoarseVertexPhi();
    if(deferred)this.encodeInitialPresentationSurface();
  }
  private mixedDynamic?: UniformMixedDynamicClassifier;
  /** GPU ownership builder for dynamic relayouts; region edits stay on the CPU path. */
  private mixedBuilder?: UniformMixedLayoutBuilder;
  private mixedBuilderStaticKey?: string;
  /** A requests-only layout's request (Requested, Full): the detail plan, the
   * host's bound on its h tiles (the plan's and, with solid contact
   * requested, every tile a solid can promote: the census's
   * liquid-conditional promotion is a subset) and the census's solid mask. */
  private mixedRequest?:{
    /** The requests' tiles (the plan of a manual-only list), moved by difference. */
    tiles:UniformDetailRequestTiles;
    /** Per tile, bit 0: a request holds it; bit 1: a solid can promote it. */
    mask:Uint8Array;count:number;coupled?:Uint8Array;
    /** What it answers: set again, a candidate that was not accepted returns to it. */
    input:{requests:readonly DetailRequest[];solid?:ReturnType<WebGPUUniformReferenceSolver["mixedSolidPromotion"]>};
    /** Tiles whose request bit moved since the builder's statics were written from `tiles`. */
    moved:number[];
    /** The last set moved request tiles only: their count, and with their
     * 26 neighbours, each once (UniformWorkEdit). Absent: the solid mask moved too. */
    edit?:UniformWorkEdit};
  /** The request tiles the builder's static masks mirror (editStatic follows them by `moved`). */
  private mixedStaticsOf?:UniformDetailRequestTiles;
  /** The coupled mask the census holds (setSolid). */
  private mixedSolidSet?:Uint8Array;
  private mixedNoSolid?:Uint8Array;
  /** The build that is due answers request edits only (mixedRequest.edit,
   * summed): the frame keeps its launch budgets and adds this bound.
   * Absent: any other cause, and the budgets start again at the ceilings. */
  private mixedEdit?:UniformWorkEdit;
  /** The h tiles of the last host layout (hostMixedLayout): zero is all-4h, the layout a request's attach edits. */
  private mixedHostFine=-1;
  private mixedEditSeen?:{stamp:number;tiles:Uint32Array};
  /** The layout is due a build. edit: a request edit's bound; any cause without one makes the build an unknown one. */
  private staleMixed(edit?:UniformWorkEdit):void{
    const held=this.mixedEdit;
    this.mixedEdit=!edit||(this.mixedStale&&!held)?undefined:{tiles:(held?.tiles??0)+edit.tiles,reach:(held?.reach??0)+edit.reach};
    this.mixedStale=true;
  }
  /** Tiles within Chebyshev distance one of the listed tiles (box indices), each once. */
  private mixedEditReach(...lists:readonly number[][]):number{
    const a=this.address,n=a.boxTiles,p=this.mixedEditTile,total=lists.reduce((sum,l)=>sum+l.length,0);
    if(27*total>=n)return n;
    let seen=this.mixedEditSeen;
    if(!seen||seen.tiles.length!==n||seen.stamp===0xffffffff)seen=this.mixedEditSeen={stamp:0,tiles:new Uint32Array(n)};
    const stamp=++seen.stamp,tiles=seen.tiles;let count=0;
    for(const list of lists)for(const t of list){
      const [x,y,z]=uniformBoxTileCoord(a,t,p);
      for(let k=z-1;k<=z+1;k++)for(let j=y-1;j<=y+1;j++)for(let i=x-1;i<=x+1;i++){
        if(!uniformTileInBox(a,i,j,k))continue;
        const q=uniformBoxTileIndex(a,i,j,k);if(tiles[q]!==stamp){tiles[q]=stamp;count++;}
      }
    }
    return count;
  }
  private readonly mixedEditTile:[number,number,number]=[0,0,0];
  /** The attached relayout is Dynamic's census: the importance criteria on,
   * the pressure band at its liquid bound. */
  private mixedCensus=false;
  /** The builder's admission capacity: the h tiles the storage holds. undefined: detached. */
  private mixedReserved?:number;
  /** The largest need a deferred build reported that the storage does not hold yet. */
  private mixedDeferredNeed=0;
  /** The capacity rule's view of this request (followMixedCapacity): the h
   * tiles its last builds asked for, oldest first, the rule's window at most. */
  private mixedNeeds:number[]=[];
  /** Builds encoded; the receipts read back from them, for the next sync. */
  private mixedBuilds=0;
  private readonly mixedReceipts:{build:number;epoch:number;sync:number;need:number;admit:number;fine:number}[]=[];
  /** The receipts waiting at the head of mixedReceipts, which keeps its store (KeptStack). */
  private mixedReceiptCount=0;
  private mixedSyncs=0;
  /** Static h tiles the builder was given that no receipt counts yet (Dynamic: a region, a moved focus). */
  private mixedStaticsAdded?:{tiles:number;build:number};
  private readonly mixedCapacityRule=uniformDetailCapacityRule();
  private mixedDeferrals={builds:0,tiles:0,changes:0};
  /** Bumped when the request changes: an older receipt's need is another request's. */
  private mixedRequestEpoch=0;
  /** A requests-only census must run every frame: bodies, or solid contact over solids. */
  private mixedLive=false;
  /** The next head must build: the request, the capacity or the roster changed since the last build. */
  private mixedStale=false;
  private mixedRosterKey="";
  /** The last advance's roster: a sync between frames bounds the same bodies. */
  private mixedRoster:readonly RigidBodyState[]=[];
  private mixedRosterDt=0;
  /** What the last accepted sync answered: the scene, the settings and the solid mask. */
  private mixedSynced?:{scene:SceneDescription;detail:UniformDetailSettings;stamp:number};
  /** A live solid edit's tiles under Requested or Full, joined to the next
   * census: at h for the build whose head displaces liquid out of the new
   * solid (the displacement moves h owners only), then whatever the requests
   * or the census say. build: the build that carried it. Its receipt lets
   * the tiles go when it was admitted; deferred, the join and the
   * displacement are made again for the next build (takeMixedReceipts), so
   * no edit's displacement is retired before its tiles exist. */
  private mixedEditJoin?:{tiles:Uint8Array;build?:number};
  /** A host (CPU) layout, between frames. Three uses remain: the t = 0
   * layout (the uploads are h data, deposited once on it), zero detail (no
   * relayout is attached), and the return of h-tile capacity (a GPU-built
   * generation's h tiles reach the host two frames late, so it cannot
   * shrink under one). mask: the tiles at h on a 4h background; absent,
   * Dynamic's start (Coarse regions at 4h, solid contact at h). Returns the
   * frame's refusal, with nothing changed. */
  private hostMixedLayout(mask?:Uint8Array):string|undefined{
    const started=performance.now(),lattice=fluidRegionLattice(this.scene),regions=this.scene.fluid.refinementRegions??[];
    let layout;
    if(mask){
      const snapped=regions.flatMap(r=>{const box=uniformDetailTileBounds(lattice,{min:r.min_m,max:r.max_m});return box?[{id:r.id,...box}]:[];});
      layout=createUniformMixedLayoutFromWidths(lattice,Uint8Array.from(mask,m=>m?1:4),snapped,this.address);
    }else{
      const forced=this.mixedSolidPromotion().forced;
      layout=createUniformMixedLayout(lattice,regions,1,forced,this.address);assertUniformMixedSolidPromotion(layout,forced);
    }
    const built=performance.now(),refusal=this.mixedFrame!.updateLayout(layout);
    if(refusal)return refusal;
    this.mixedGeneration++;this.mixedHostFine=layout.fineTiles.length;
    Object.assign(this.executionInfo,{uniformMixedGeneration:this.mixedGeneration,uniformMixedFineTiles:layout.fineTiles.length,
      uniformMixedCoarseTiles:layout.coarseTiles.length,
      uniformMixedOwners:layout.cellCount,uniformSimulationCellScale:undefined,
      uniformMixedLayoutBuild_ms:built-started,uniformMixedLayoutApply_ms:performance.now()-built});
    this.publishContact(layout.fineTiles.length);
    return undefined;
  }
  /** Requested with no solid-contact request: a static solid requests
   * nothing (its tiles are cut 4h owners). Bodies request through the census. */
  private mixedRequestOf():boolean{
    const lattice=refinementRegionLattice(this.scene);
    const set=buildUniformDetailRequests({settings:this.detail,regions:this.scene.fluid.refinementRegions??[],domain:uniformDetailDomain(lattice),acceptedStep:this.detailAcceptedStep});
    this.detailUnconstrained=set.unconstrainedRegions.length;
    // No automatic source outside Dynamic: its planner state goes with it.
    this.detailPlanner=undefined;this.detailPlan=undefined;
    return this.setMixedRequest({requests:set.requests,solid:this.detail.solidContact?this.mixedSolidPromotion():undefined});
  }
  /** Bring the kept request to `input` by what differs: the tiles that
   * entered and left the request boxes, and every tile's promotion bit only
   * under another solid mask or contact setting. True: a tile or the solid
   * mask moved, so the layout a build would give did. */
  private setMixedRequest(input:NonNullable<WebGPUUniformReferenceSolver["mixedRequest"]>["input"]):boolean{
    let request=this.mixedRequest;const lattice=fluidRegionLattice(this.scene),held=request?.tiles.lattice;
    // Another lattice names other tiles: its request starts empty.
    if(!request||!held||JSON.stringify(held)!==JSON.stringify(lattice)){const tiles=new UniformDetailRequestTiles(lattice);request=this.mixedRequest={tiles,mask:new Uint8Array(tiles.tiles),count:0,input:{requests:[]},moved:[]};}
    const {mask,tiles,moved}=request,{entered,left}=tiles.set(input.requests),forced=input.solid?.forced,solids=forced!==request.input.solid?.forced;
    if(solids){
      const fine=tiles.fine;let count=0;
      for(let t=0;t<mask.length;t++){const m=fine[t]!|(forced?forced[t]!<<1:0);mask[t]=m;if(m)count++;}
      request.count=count;
    }else{
      for(const t of entered){if(!mask[t])request.count++;mask[t]!|=1;}
      for(const t of left)if(!(mask[t]!&=2))request.count--;
    }
    if(this.mixedStaticsOf===tiles){for(const t of entered)moved.push(t);for(const t of left)moved.push(t);}
    request.coupled=input.solid?.any?input.solid.coupled:undefined;request.input=input;
    request.edit=solids?undefined:{tiles:entered.length+left.length,reach:this.mixedEditReach(entered,left)};
    return solids||entered.length>0||left.length>0;
  }
  /** The layout's owner, between frames. One path: the GPU relayout (census,
   * builder, adopt in the frame head). Dynamic attaches it with the
   * importance criteria on. Requested and Full run the same relayout with
   * them off (uniformDetailImportance), so it builds from requests only: the
   * plan's tiles as builder statics, the solid-contact request as the
   * census's liquid-conditional promotion, bodies as the tiles their GPU
   * poses sweep. It is attached while the host's bound on those requests is
   * not zero. With no request nothing is attached: the all-4h host layout,
   * C = 0, base blocks only, no census. Capacity is one rule for every
   * policy (followMixedCapacity). hostMixedLayout says what the host still
   * builds. bodies: the roster the next frame advances (absent: the last
   * one's). advancing: the frame is next (Dynamic's statics are planned
   * once per accepted step). False: the device refused the request's h
   * tiles; the accepted generation keeps running and uniformDetail.rejected
   * names why. */
  private syncMixedLayout(bodies:readonly RigidBodyState[]=this.mixedRoster,dt=this.mixedRosterDt,advancing=false):boolean{
    const frame=this.mixedFrame;if(!frame)return true;
    const first=this.mixedGeneration===0,tiles=this.address.boxTiles;
    this.mixedRoster=bodies;this.mixedRosterDt=dt;
    const taken=this.takeMixedReceipts();
    if(this.detail.policy==="dynamic"){
      if(!this.mixedGpuLayout||!this.mixedCensus){
        if(first){
          const refusal=this.hostMixedLayout();if(refusal)throw new Error(`Uniform simulation detail: ${refusal}`);
          this.publishDetail(this.planDetail());
        }
        // Every tile the pool can store at the attach: frame 1's head runs
        // on the all-h start, and a census's build may ask for any stored
        // tile.
        frame.setRelayout(this.mixedRelayout());this.watchMixedAllocation();
        this.mixedGpuLayout=true;this.mixedCensus=true;this.mixedRequest=undefined;this.resetMixedCapacity();
        this.mixedBuilderStaticKey=undefined;this.mixedBuilderStaticOf=undefined;this.mixedRegionKey="";this.mixedRefusedKey="";this.mixedLive=false;this.mixedSynced=undefined;this.mixedEditJoin=undefined;
      }
      // Static h tiles reach the builder here, ahead of the build that reads
      // them, so the capacity holds the new ones before any receipt counts them.
      if(advancing&&!this.detailStaticCurrent()){
        const before=this.detailPlan?.fine;this.refreshMixedBuilderStatics(this.mixedSolidPromotion());
        const after=this.detailPlan!.fine;let added=0;
        for(let t=0;t<after.length;t++)if(after[t]&&!before?.[t])added++;
        if(added)this.mixedStaticsAdded={tiles:added+(this.mixedStaticsAdded?.tiles??0),build:this.mixedBuilds};
      }
      const swept=bodies.length?this.mixedBodyTiles(bodies,dt):undefined,joined=this.mixedEditJoin?.tiles;
      let extra=this.mixedStaticsAdded?.tiles??0;
      if(swept||joined)for(let t=0;t<tiles;t++)if(swept?.[t]||joined?.[t])extra++;
      // The bound is the capacity the attach reserved, every tile the pool
      // can store: nothing grows here. A census has no generation to fall
      // back to.
      const refusal=this.followMixedCapacity(this.address.poolTiles,extra,true,true);
      if(refusal)throw new Error(`Uniform simulation detail: ${refusal}`);
      return true;
    }
    // Nothing asked since the last sync and no body to bound: only the capacity follows its receipts.
    const synced=this.mixedSynced;
    if(!first&&!bodies.length&&!this.mixedRosterKey&&synced?.scene===this.scene&&synced.detail===this.detail&&synced.stamp===this.mixedSolidMaskStamp
      &&!this.mixedEditJoin&&!taken.released){
      const request=this.mixedRequest;
      if(this.mixedGpuLayout&&request&&taken.any){
        const refusal=this.followMixedCapacity(request.count,0,this.mixedLive,false);
        if(refusal){this.executionInfo.uniformDetail={...this.executionInfo.uniformDetail!,rejected:refusal};return false;}
      }
      return true;
    }
    return this.requestMixedLayout(frame,first,tiles,bodies,dt);
  }
  /** syncMixedLayout for another request, roster or owner (apart from it: these closures would cost every sync a context). */
  private requestMixedLayout(frame:UniformMixedFrame,first:boolean,tiles:number,bodies:readonly RigidBodyState[],dt:number):boolean{
    // Whatever it reserves or attaches moves the frame's bytes.
    this.watchMixedAllocation();
    const requestKey=`${JSON.stringify(this.scene.fluid.refinementRegions??[])}:${this.detail.policy}:${this.detail.solidContact}`,key=`${this.mixedSolidMaskStamp}:${requestKey}`;
    // The roster, not its poses: the census follows those on the GPU.
    const roster=bodies.length?JSON.stringify(bodies.map(b=>[b.description.id,b.description.shape,b.description.dimensions_m,b.description.motion])):"";
    // A generation Dynamic built: the requests take its relayout over where it stands.
    const census=this.mixedGpuLayout&&this.mixedCensus;
    let fresh=false,moved=false;
    // A candidate that is not accepted leaves the kept request as it was.
    const kept=this.mixedRequest,prior=kept&&{input:kept.input,moved:kept.moved.length};
    const keep=()=>{
      if(!fresh)return;
      if(this.mixedRequest!==kept)this.mixedRequest=kept;
      else if(prior){this.setMixedRequest(prior.input);kept!.moved.length=prior.moved;}
    };
    // Refused before: the same request is not asked again.
    if(!prior||(key!==this.mixedRegionKey&&key!==this.mixedRefusedKey)){moved=this.mixedRequestOf();fresh=true;}
    // A refused scene request was taken back: the layout answers the request again.
    else if(key===this.mixedRegionKey&&this.mixedRefusedKey){this.mixedRefusedKey="";this.executionInfo.uniformDetail={...this.executionInfo.uniformDetail!,rejected:undefined};}
    // Another request or owner: no receipt counts it, so the host's bound
    // holds the capacity again. A solid edit or a roster change alone keeps
    // the receipts: its tiles are the join's and the bodies' reach, counted below.
    const edited=fresh&&!census&&this.mixedRegionKey.slice(this.mixedRegionKey.indexOf(":")+1)===requestKey;
    if(fresh&&!edited)this.resetMixedCapacity();
    // A request whose tiles and solid mask stand (a region moved inside its tiles) keeps the layout it has.
    // One that moved request tiles only, on a layout nothing but the requests hold
    // (no solid contact, body or join: an empty census band), is a counted edit.
    const request=this.mixedRequest!;
    if((fresh&&(moved||census))||roster!==this.mixedRosterKey)
      this.staleMixed(fresh&&!census&&roster===this.mixedRosterKey&&this.mixedGpuLayout&&!this.mixedLive&&!bodies.length&&!request.coupled&&!this.mixedEditJoin?request.edit:undefined);
    this.mixedRosterKey=roster;
    // The bound: the request's tiles, the tiles the roster's poses can reach
    // this step and a live edit's join. extra: those the host adds in one
    // frame with no liquid moving, over whatever the receipts count.
    const mask=request.mask,swept=bodies.length&&request.count<tiles?this.mixedBodyTiles(bodies,dt):undefined,joined=this.mixedEditJoin?.tiles;
    let need=request.count,extra=0;
    if(swept||joined)for(let t=0;t<tiles;t++){if(!mask[t]&&(swept?.[t]||joined?.[t])){need++;extra++;}else if(joined?.[t])extra++;}
    need=Math.min(tiles,need);
    // The liquid moves the need: solid contact over solids, bodies.
    const live=request.count<tiles&&(bodies.length>0||!!request.coupled);
    const refuse=(refusal:string)=>{
      // Nothing accepted yet: there is no generation to keep running.
      if(first)throw new Error(`Uniform simulation detail: ${refusal}`);
      keep();
      if(fresh)this.mixedRefusedKey=key;
      this.executionInfo.uniformDetail={...this.executionInfo.uniformDetail!,rejected:refusal};
      return false;
    };
    // Haloed patches (QA) hold their first layout: no relayout of either kind.
    const patch=false;
    if(this.mixedGpuLayout&&need===0){
      // Nothing is requested: zero detail is the host's all-4h layout, C = 0.
      frame.setRelayout(undefined);this.mixedGpuLayout=false;this.mixedCensus=false;this.mixedReserved=undefined;
      const refusal=this.hostMixedLayout(mask);
      if(refusal)throw new Error(`Uniform simulation detail: ${refusal}`);
      this.publishMixedCapacity();
    }else if(!this.mixedGpuLayout&&(first||(fresh&&(need===0||patch)))){
      const refusal=this.hostMixedLayout(mask);if(refusal)return refuse(refusal);
      this.publishMixedCapacity();
    }
    if(need>0&&!patch){
      // Admission before the request changes anything: the capacity its bound needs.
      let refusal=this.followMixedCapacity(need,extra,live,false);
      // Dynamic's generation stays; its band was the liquid's, a request's is every h tile.
      if(!refusal&&census){refusal=frame.reserveFine(frame.ownership.capacity.fineTiles,false,true);this.refreshMixedAllocation();}
      if(refusal)return refuse(refusal);
      if(!this.mixedGpuLayout){
        this.refreshRequestStatics(request);
        // On the storage the rule just reserved. The attach builds. From
        // all-4h, a request nothing but the host moves (no body, solid
        // contact or join) is a counted edit of that layout: its tiles and
        // their reach bound the build, and the frame keeps its budgets.
        let counted:UniformWorkEdit|undefined;
        if(!first&&this.mixedHostFine===0&&!bodies.length&&!request.coupled&&!this.mixedEditJoin){
          const listed:number[]=[];for(let t=0;t<mask.length;t++)if(mask[t])listed.push(t);
          counted={tiles:listed.length,reach:this.mixedEditReach(listed)};
        }
        frame.setRelayout(this.mixedRelayout(),frame.ownership.capacity.fineTiles,!!counted);this.mixedGpuLayout=true;this.staleMixed();this.mixedEdit=counted;
      }else if(fresh||census)this.refreshRequestStatics(request);
      this.mixedCensus=false;this.publishMixedCapacity();
    }
    if(fresh){this.mixedRegionKey=key;this.mixedRefusedKey="";this.publishDetail({diagnostics:request.tiles.diagnostics(this.detail.budgetPercent)});}
    else if(!this.mixedRefusedKey&&this.executionInfo.uniformDetail?.rejected)this.executionInfo.uniformDetail={...this.executionInfo.uniformDetail,rejected:undefined};
    // Every tile already h (Full): nothing for a census to follow.
    this.mixedLive=this.mixedGpuLayout&&live;
    this.mixedSynced={scene:this.scene,detail:this.detail,stamp:this.mixedSolidMaskStamp};
    return true;
  }
  /** A new request or owner: its receipts start again, and a return staged for the last one is taken back. */
  private resetMixedCapacity():void{
    this.mixedRequestEpoch++;this.mixedNeeds=[];this.mixedDeferredNeed=0;this.mixedStaticsAdded=undefined;
  }
  /** The receipts that arrived since the last sync, in the rule's window.
   * released: a live edit's join was admitted and its tiles follow the
   * requests again. A build under the staged cap that fit it confirms the cap. */
  private takeMixedReceipts():{any:boolean;released:boolean}{
    const rule=this.mixedCapacityRule,sync=++this.mixedSyncs;let any=false,released=false;
    const waiting=this.mixedReceipts;
    while(this.mixedReceiptCount&&waiting[0]!.sync<sync){
      // Read here and nowhere later: the record is the next receipt's to fill (readMixedRelayout).
      const r=waiting[0]!,over=r.need>r.admit,join=this.mixedEditJoin,left=--this.mixedReceiptCount;any=true;this.mixedReceiptPool.give(r);
      for(let i=0;i<left;i++)waiting[i]=waiting[i+1]!;
      if(join?.build!==undefined&&r.build>=join.build){
        if(over){
          // Deferred with the edit: its displacement ran on the tiles that were
          // already h. The join and the displacement are made again for the
          // build the grown capacity admits.
          join.build=undefined;this.mixedDynamic!.join(this.mixedTileWords(join.tiles));this.mixedFrame!.editSolids();
        }else{this.mixedEditJoin=undefined;this.staleMixed();released=true;}
      }
      if(this.mixedStaticsAdded&&r.build>=this.mixedStaticsAdded.build)this.mixedStaticsAdded=undefined;
      if(r.epoch!==this.mixedRequestEpoch)continue;
      this.mixedNeeds.push(r.need);if(this.mixedNeeds.length>rule.windowBuilds)this.mixedNeeds.shift();
      if(over){
        // The build kept its generation: the next sync holds its need and the head builds again.
        this.mixedDeferredNeed=Math.max(this.mixedDeferredNeed,r.need);this.staleMixed();
        this.mixedDeferrals.builds++;this.mixedDeferrals.tiles+=Math.max(0,r.need-r.fine);
      }
    }
    const taken=this.mixedTaken;taken.any=any;taken.released=released;
    return taken;
  }
  /** takeMixedReceipts' answer, rewritten by each call, and the receipt records it has read. */
  private readonly mixedTaken={any:false,released:false};
  private readonly mixedReceiptPool=new KeptStack<{build:number;epoch:number;sync:number;need:number;admit:number;fine:number}>();
  /** One h-tile capacity rule for Requested, Dynamic and Full. The storage
   * holds C h tiles and the builder admits A <= C: a build that asks for
   * more keeps its generation and its receipt says what it needed.
   * bound: the host's count of what the next build may ask. It holds the
   * capacity until a receipt of this request arrives (the first-frame hint,
   * which covers solid contact), and for a request only the host changes it
   * is exact and stays the rule. extra: tiles the host adds in one frame
   * with no liquid moving (the bodies' reach, a live edit's join, static
   * tiles newly planned), held above what the receipts count. live: the
   * liquid moves the need (a census), so the capacity keeps the rule's
   * headroom over the last receipt (UniformDetailCapacityRule). Growth is
   * between frames (reserveFine) and stops at address.poolTiles: a build
   * marks h only where a tile is stored, so none asks for more. Dynamic's
   * attach reserves that much, so under it nothing here grows. The rule
   * holds what it reserved: the capacity never returns.
   * Returns the device's refusal of a growth, with nothing changed. */
  private followMixedCapacity(bound:number,extra:number,live:boolean,liquidBand:boolean):string|undefined{
    const frame=this.mixedFrame!,rule=this.mixedCapacityRule,T=this.address.poolTiles,held=frame.ownership.capacity.fineTiles;
    const needs=this.mixedNeeds,last=Math.max(needs.length?needs[needs.length-1]!:0,this.mixedDeferredNeed);
    // What must fit now, and what a growth reserves.
    const must=Math.max(Math.min(T,bound),this.mixedDeferredNeed,live?Math.min(T,this.grownMixedTiles(last)+extra):0);
    if(must>held){
      const want=Math.max(must,live?Math.min(T,Math.min(T,this.roomedMixedTiles(last)+extra)+rule.floorTiles):0);
      this.watchMixedAllocation();
      let reservation=frame.fineReservation(want,liquidBand,live);
      if(reservation.refusal&&want>must)reservation=frame.fineReservation(must,liquidBand,true);
      if(reservation.refusal)return reservation.refusal;
      const refusal=frame.reserveFine(reservation.fineTiles,liquidBand,true);if(refusal)return refusal;
      this.mixedDeferrals.changes++;
    }
    if(this.mixedDeferredNeed<=frame.ownership.capacity.fineTiles)this.mixedDeferredNeed=0;
    const admission=frame.ownership.capacity.fineTiles;
    this.admitMixed(admission,!live&&!liquidBand&&bound<=admission);
    return undefined;
  }
  /** The builder's admission capacity. A capacity of every tile the pool can store admits anything: no admit launch.
   * fits: it holds everything a host-counted request asks, so the build under it moves no tile (a zero edit). */
  /** ceil((1 + grow) need) and ceil((1 + headroom) need) of the capacity rule,
   * each kept for the need it last answered: the product is a heap number
   * until the call is optimized, and the need is the same from frame to frame. */
  private grownMixedTiles(need:number):number{
    if(need!==this.mixedGrownNeed){this.mixedGrownNeed=need;this.mixedGrown=Math.ceil((1+this.mixedCapacityRule.grow)*need);}
    return this.mixedGrown;
  }
  private roomedMixedTiles(need:number):number{
    if(need!==this.mixedRoomedNeed){this.mixedRoomedNeed=need;this.mixedRoomed=Math.ceil((1+this.mixedCapacityRule.headroom)*need);}
    return this.mixedRoomed;
  }
  private mixedGrownNeed=-1;private mixedGrown=0;private mixedRoomedNeed=-1;private mixedRoomed=0;
  private admitMixed(tiles?:number,fits=false):void{
    const frame=this.mixedFrame!,capacity=frame.ownership.capacity;
    if(tiles!==this.mixedReserved){this.mixedBuilder!.setAdmission(tiles===undefined||tiles>=capacity.address.poolTiles?undefined:tiles);this.mixedReserved=tiles;this.staleMixed(fits?{tiles:0,reach:0}:undefined);}
    this.publishMixedCapacity();
  }
  private publishMixedCapacity():void{
    const capacity=this.mixedFrame!.ownership.capacity.fineTiles,d=this.mixedDeferrals;
    const info=this.executionInfo;
    if(capacity!==info.uniformMixedFineCapacity)this.refreshMixedAllocation();
    info.uniformMixedFineCapacity=capacity;info.uniformMixedAdmission=this.mixedGpuLayout?this.mixedReserved??capacity:capacity;
    info.uniformMixedDeferredBuilds=d.builds;info.uniformMixedDeferredTiles=d.tiles;info.uniformMixedCapacityChanges=d.changes;
  }
  /** Plan detail for the current policy, regions and latest focus at the
   * accepted step, and commit it (the planner's automatic state advances). */
  private planDetail():UniformDetailPlan{
    const lattice=fluidRegionLattice(this.scene),planner=this.detailPlanner??=new UniformDetailPlanner(lattice);
    const focus=this.detailFocusPending,step=this.detailAcceptedStep;
    const set=buildUniformDetailRequests({settings:this.detail,regions:this.scene.fluid.refinementRegions??[],domain:uniformDetailDomain(refinementRegionLattice(this.scene)),acceptedStep:step,focus});
    const plan=planner.plan({requests:set.requests,suppressions:set.suppressions,automaticSources:set.automaticSources,acceptedStep:step,budgetPercent:this.detail.budgetPercent});
    planner.commit(plan);
    const suppressed=new Uint8Array(planner.tiles);
    for(const box of set.suppressions)for(const t of planner.tilesOf(box.bounds_m))suppressed[t]=1;
    this.detailPlan=plan;this.detailSuppressed=suppressed;this.detailFocus=focus;this.detailUnconstrained=set.unconstrainedRegions.length;
    return plan;
  }
  /** The accepted plan's diagnostics, beside what this solver adds outside it. */
  private publishDetail(plan:Pick<UniformDetailPlan,"diagnostics">):void{
    const d=plan.diagnostics,policy=this.detail.policy,contactTiles=this.contactTilesOf(this.executionInfo.uniformMixedFineTiles??0,d.admittedTiles);
    this.executionInfo.uniformDetail={policy,requestKey:uniformDetailRequestKey(policy,this.scene.fluid.refinementRegions),
      focusRevision:policy==="dynamic"&&this.detail.nearFocus?this.detailFocus?.revision:undefined,
      tiles:d.tiles,requestedTiles:d.requestedTiles,admittedTiles:d.admittedTiles,supportTiles:d.supportTiles,deferredTiles:d.deferredTiles,
      budgetClippedTiles:d.budgetClippedTiles,budgetTiles:d.budgetTiles,automaticCostTiles:d.automaticCostTiles,contactTiles,
      patchCells:d.patchCells,residentPatches:d.residentPatches,allocatedCells:d.allocatedCells,activeCells:d.activeCells,
      supportCells:d.supportCells,wastedCells:d.wastedCells,reasons:d.reasons,unconstrainedRegions:this.detailUnconstrained,
      mapping:policy==="full"?"Every tile at h (GPU layout, requests only)."
        :policy==="requested"?uniformDetailCoarseSolids(this.detail)?"4h background, static solids on cut 4h owners; Fine regions at h, bodies at h where they meet liquid (GPU layout, requests only)."
        :"4h background; Fine regions at h, solids and bodies at h where they meet liquid (GPU layout, requests only)."
        :"Interim: the GPU surface census picks h or 4h each frame; Fine regions and focus tiles are held h, Coarse regions 4h except solid contact.",
      };
  }
  /** h tiles beyond the plan's under Requested and Full: solid and body
   * contact (for one build, a live edit's tiles). Dynamic's census picks
   * its own tiles, so it reports none. */
  private contactTilesOf(fine:number,admitted:number):number{return this.detail.policy==="dynamic"?0:Math.max(0,fine-admitted);}
  /** The running layout's h tiles changed (a host layout, a build's receipt). */
  private publishContact(fine:number):void{
    const d=this.executionInfo.uniformDetail;if(!d)return;
    const contactTiles=this.contactTilesOf(fine,d.admittedTiles);
    if(contactTiles!==d.contactTiles)this.executionInfo.uniformDetail={...d,contactTiles};
  }
  /** Transient focus: the latest revision wins, consumed at the next frame head. */
  applyDetailInput(input:SolverDetailInput):void{
    const focus=input.focus;
    if(focus&&(!this.detailFocusPending||focus.revision!==this.detailFocusPending.revision))this.detailFocusPending=focus;
  }
  /** Dynamic statics are current while this key holds: replanned on a focus
   * revision, a setting, or each accepted step while hysteresis or churn is pending. */
  private detailStaticKey():string{
    const pending=this.detailPlanner&&(!this.detailPlanner.settled||(this.detailPlan?.diagnostics.churnDeferredTiles??0)>0);
    const focus=this.detail.nearFocus?this.detailFocusPending?.revision??0:0;
    return `${this.mixedSolidMaskStamp}:${JSON.stringify(this.scene.fluid.refinementRegions??[])}:${JSON.stringify(this.detail)}:${focus}${pending?`:${this.detailAcceptedStep}`:""}`;
  }
  /** What mixedBuilderStaticKey was built from (holdDetailStaticKey): the
   * region list and the settings by identity (each is replaced, never
   * rewritten), the rest by value. */
  private mixedBuilderStaticOf?:{regions:unknown;detail:UniformDetailSettings;stamp:number;focus:number;pending:boolean;step:number};
  private holdDetailStaticKey():void{
    const pending=!!(this.detailPlanner&&(!this.detailPlanner.settled||(this.detailPlan?.diagnostics.churnDeferredTiles??0)>0));
    this.mixedBuilderStaticKey=this.detailStaticKey();
    this.mixedBuilderStaticOf={regions:this.scene.fluid.refinementRegions,detail:this.detail,stamp:this.mixedSolidMaskStamp,
      focus:this.detail.nearFocus?this.detailFocusPending?.revision??0:0,pending,step:this.detailAcceptedStep};
  }
  /** detailStaticKey() === mixedBuilderStaticKey, asked every frame: under
   * the held region list and settings the key's other terms decide it and no
   * text is built. Another list or settings object compares the keys (equal
   * content under another object holds it from then on). */
  private detailStaticCurrent():boolean{
    const held=this.mixedBuilderStaticOf;
    if(held&&this.mixedBuilderStaticKey!==undefined&&held.regions===this.scene.fluid.refinementRegions&&held.detail===this.detail){
      const pending=!!(this.detailPlanner&&(!this.detailPlanner.settled||(this.detailPlan?.diagnostics.churnDeferredTiles??0)>0));
      const focus=this.detail.nearFocus?this.detailFocusPending?.revision??0:0;
      return held.stamp===this.mixedSolidMaskStamp&&held.focus===focus&&held.pending===pending&&(!pending||held.step===this.detailAcceptedStep);
    }
    const current=this.detailStaticKey()===this.mixedBuilderStaticKey;
    if(current&&held){held.regions=this.scene.fluid.refinementRegions;held.detail=this.detail;}
    return current;
  }
  /** Horizon one: the frame head classifies the state the frame starts from
   * for this frame's dt and advects on the generation it builds. Smooth
   * surface tiles may use 4h; detail and impact protection retain h. This
   * also selects surface pressure resolution, so broad motion and sheets
   * are assessed by properties rather than an identical fine trajectory. */
  private mixedRelayout():UniformMixedFrameRelayout{
    const dynamic=this.mixedDynamic!,builder=this.mixedBuilder!;
    return {generation:builder.generation,receipt:builder.receipt,reasons:dynamic.reasons,importance:dynamic.importance,
    // Requests only: a layout only the host changes is built once per change, and the frames between run the head without the census.
    due:()=>this.detail.policy==="dynamic"||this.mixedLive||this.mixedStale,
    edit:()=>this.detail.policy==="dynamic"||this.mixedLive||!this.mixedStale?undefined:this.mixedEdit,encode:(encoder,dt,views)=>{
      // Requests-only statics are the host's, refreshed with their admission (syncMixedLayout).
      if(this.detail.policy==="dynamic"&&!this.detailStaticCurrent())this.refreshMixedBuilderStatics(this.mixedSolidPromotion());
      // One more build after the last live frame returns what its bodies held.
      this.mixedStale=this.mixedLive;this.mixedEdit=undefined;
      const build=this.mixedBuilds++,join=this.mixedEditJoin;if(join&&join.build===undefined)join.build=build;
      const g=this.scene.fluid.gravity_m_s2;
      // The detail controls' importance criteria: what the census requires at h.
      const held=this.mixedImportance(this.address.boxTiles),importance=held.importance;
      // The build's counters clear in the census's blit run.
      builder.encodeClear(encoder);
      if(this.detail.policy==="dynamic")this.mixedFrame?.narrowBandFlip?.refine(encoder,dynamic.joinTarget(),dt,this.narrowBandFinePadding);
      // One policy record, rewritten for each census (the census keeps none of it).
      // (The scene's gravity when it is applied, the settings' numbers when they are replaced: read each step, each is a heap number.)
      const gravity=this.mixedCensusGravity,applied=this.mixedCensusSceneRevision!==this.sceneRevision;
      if(applied){this.mixedCensusSceneRevision=this.sceneRevision;gravity[0]=g.x;gravity[1]=g.y;gravity[2]=g.z;}
      const policy=this.mixedCensusPolicy??={dt,steps:1,gravity,reach:this.detail.marginTiles,hysteresis:0,
        surfaceTolerance:importance.shapeTolerance,fastTravel:0,importance,
        boundaryTravel:importance.impactTravel,up:Math.sign(-g.y),
        fullTolerance:UNIFORM_MIXED_DYNAMIC_FULL_TOLERANCE,emptyTolerance:Math.max(0,1e-6),reasons:views};
      policy.dt=dt;policy.reach=this.detail.marginTiles;policy.reasons=views;
      if(applied)policy.up=Math.sign(-g.y);
      if(policy.importance!==importance){policy.importance=importance;policy.surfaceTolerance=importance.shapeTolerance;policy.boundaryTravel=importance.impactTravel;}
      dynamic.encode(encoder,policy,false);
      // The census listed its certificate's pages: the commit stores them, this step's drop's and the next step's scheduled drop's, before any stage of the generation reads a block.
      if(this.pageDrop)this.requestDropPages(this.pageDrop);
      if(this.pageDropAhead)this.requestDropPages(this.pageDropAhead);
      if(this.pageInjectedAhead)this.requestDropPages(this.pageInjectedAhead);
      this.mixedFrame!.encodePageCommit(encoder);
      this.mixedFrame?.narrowBandFlip?.activity(encoder,dynamic.importance,dt,this.narrowBandAdaptiveFadeSeconds,this.narrowBandSparseParticleLifetime);
      builder.encode(encoder);
    },adopted:encoder=>{
      // Diagnostics only: a free read takes this build's receipt; mapped
      // once the frame submits (readMixedRelayout), never awaited.
      const read=this.mixedRelayoutReads.take();
      if(read){const r=builder.receipt;encoder.copyBufferToBuffer(r.buffer,r.offset,read,0,r.words*4);this.mixedRelayoutRead=read;}
    }};
  }
  /** The census's importance policy for these settings and this lattice
   * (uniformDetailImportance, with the census's own additions), kept while
   * both stand: the settings are replaced, never rewritten. criteria: one is on. */
  private mixedImportance(tiles:number){
    let held=this.mixedImportanceOf;
    if(!held||held.detail!==this.detail||held.tiles!==tiles)held=this.mixedImportanceOf=this.mixedImportanceFor(this.detail,tiles);
    held.budget.percent=this.narrowBandAdaptiveBudgetPercent;
    return held;
  }
  private mixedImportanceFor(detail:UniformDetailSettings,tiles:number){
    const budget={percent:this.narrowBandAdaptiveBudgetPercent};
    const importance={...uniformDetailImportance(detail,tiles),surfaceAuthority:true,
      ...(detail.policy==="dynamic"?{budgetTiles:undefined,activityBudget:budget}:{})};
    return {detail,tiles,importance,budget,criteria:Object.values(importance.criteria).some(Boolean)};
  }
  private mixedImportanceOf?:ReturnType<WebGPUUniformReferenceSolver["mixedImportanceFor"]>;
  private readonly mixedCensusGravity:[number,number,number]=[0,0,0];
  private mixedCensusSceneRevision=0;
  private mixedCensusPolicy?:UniformMixedDynamicPolicy;
  /** Map the relayout receipt the frame just submitted into executionInfo. */
  private readMixedRelayout():void{
    const read=this.mixedRelayoutRead;this.mixedRelayoutRead=undefined;if(!read)return;
    // The read's own record carries what its receipt answers to: a read is mapped once at a time.
    const slot=this.mixedRelayoutSlots.get(read)??this.mixedRelayoutSlot(read);
    slot.sequence=++this.mixedRelayoutSequence;slot.epoch=this.mixedRequestEpoch;slot.host=this.mixedGeneration;slot.build=this.mixedBuilds-1;slot.sync=this.mixedSyncs;
    read.mapAsync(GPUMapMode.READ).then(slot.mapped,this.failMixedRelayout);
  }
  /** A read's record, made once (apart from readMixedRelayout: this closure would cost every read a context). */
  private mixedRelayoutSlot(read:GPUBuffer){const made={read,sequence:0,epoch:0,host:0,build:0,sync:0,mapped:():void=>this.takeMixedRelayout(made)};this.mixedRelayoutSlots.set(read,made);return made;}
  private readonly mixedRelayoutSlots=new Map<GPUBuffer,{read:GPUBuffer;sequence:number;epoch:number;host:number;build:number;sync:number;mapped:()=>void}>();
  private readonly failMixedRelayout=(error:unknown):void=>{if(!this.disposed)this.failMixedFrame(error);};
  /** A mapped receipt's words, copied out of the mapping (float lanes over the same words). */
  private mixedRelayoutWords?:Uint32Array<ArrayBuffer>;
  private mixedRelayoutFloats?:Float32Array<ArrayBuffer>;
  private takeMixedRelayout(slot:{read:GPUBuffer;sequence:number;epoch:number;host:number;build:number;sync:number}):void{
    const read=slot.read,mapped=new Uint32Array(read.getMappedRange()),r=this.mixedRelayoutWords?.length===mapped.length?this.mixedRelayoutWords:this.mixedRelayoutWords=new Uint32Array(mapped.length);
    r.set(mapped);read.unmap();this.mixedRelayoutReads.give(read);
    const R=UNIFORM_MIXED_RELAYOUT_RECEIPT,C=R.census,f=this.mixedRelayoutFloats?.buffer===r.buffer?this.mixedRelayoutFloats:this.mixedRelayoutFloats=new Float32Array(r.buffer);
    if(this.disposed)return;
    // The h tiles the build asked for, the admission it ran under and the
    // generation it left: the next sync's capacity (takeMixedReceipts).
    const receipt=this.mixedReceiptPool.take()??{build:0,epoch:0,sync:0,need:0,admit:0,fine:0};
    receipt.build=slot.build;receipt.epoch=slot.epoch;receipt.sync=slot.sync;receipt.need=r[R.need]!;receipt.admit=r[R.admit]!;receipt.fine=r[R.tiers]!;
    this.mixedReceipts[this.mixedReceiptCount++]=receipt;
    if(slot.sequence<this.mixedRelayoutShown)return;
    this.mixedRelayoutShown=slot.sequence;
    // A fatal build latches frame failure 6/7 on the GPU (the remap); the
    // frame's receipt names it. Nothing here decides anything.
    // A host layout replaced the generation this build belongs to: its counts are not the running layout's.
    if(slot.host!==this.mixedGeneration||!this.mixedGpuLayout)return;
    const generation=r[R.generation]!,fine=r[R.tiers]!,coarse=r[R.tiers+1]!,info=this.executionInfo as UniformPublishedInfo;
    this.publishContact(fine);
    info.uniformMixedGeneration=this.mixedGeneration+generation;
    info.uniformMixedFineTiles=fine;info.uniformMixedCoarseTiles=coarse;info.uniformMixedOwners=64*fine+coarse;info.uniformSimulationCellScale=undefined;
    info.uniformMixedLayoutBuild_ms=0;info.uniformMixedLayoutApply_ms=0;
    info.uniformMixedDynamicRelayouts=generation;info.uniformMixedDynamicChangedTiles=r[R.changed]!;
    info.uniformMixedDynamicInterfaceTiles=r[C]!;info.uniformMixedDynamicBandTiles=r[C+2]!;
    info.uniformMixedDynamicRefined=r[C+4]!;info.uniformMixedDynamicCoarsened=r[C+5]!;info.uniformMixedDynamicRequiredTiles=r[C+12]!;info.uniformMixedDynamicBoundaryTiles=r[C+16]!;info.uniformMixedDynamicSolidTiles=r[C+19]!;info.uniformMixedDynamicUnresolvedCoarse=r[C+1]!;
    info.uniformMixedDynamicCoarsePartialVolume=r[C+6]!;info.uniformMixedDynamicCoarsePhiCrossing=r[C+7]!;info.uniformMixedDynamicCoarseDryLiquidPhi=r[C+3]!;
    // The two pairs are this solver's, rewritten in place like the record that holds them.
    const deficit=this.mixedInteriorDeficit,air=this.mixedAirVolume;deficit[0]=f[C+8]!;deficit[1]=f[C+9]!;air[0]=f[C+10]!;air[1]=f[C+11]!;
    info.uniformMixedDynamicInteriorDeficit=deficit;info.uniformMixedDynamicAirVolume=air;
  }
  private readonly mixedInteriorDeficit:[number,number]=[0,0];
  private readonly mixedAirVolume:[number,number]=[0,0];
  /** Tiles a roster's bodies can touch over `dt` (bounding sphere dilated by
   * its travel and one cell), plus one tile around them: the host's count of
   * what the census promotes around the GPU poses (UniformMixedBodies
   * .encodeTiles via setBodies, liquid-conditional). The roster's poses and
   * speeds may be older than the GPU's: a census that needs more defers its
   * build and its receipt grows the capacity (readMixedRelayout). */
  private mixedBodyScratch?:Uint8Array;
  private mixedBodyTiles(bodies:readonly RigidBodyState[],dt:number):Uint8Array{
    const {cellSize_m,origin_m}=fluidRegionLattice(this.scene);
    const a=this.address,count=a.boxTiles,wx=4*cellSize_m[0],wy=4*cellSize_m[1],wz=4*cellSize_m[2],cell=Math.max(cellSize_m[0],cellSize_m[1],cellSize_m[2]);
    // One scratch mask per solver: the callers count it and keep nothing.
    const tiles=this.mixedBodyScratch?.length===count?this.mixedBodyScratch.fill(0):(this.mixedBodyScratch=new Uint8Array(count));
    for(const body of bodies){
      const v=body.linearVelocity_m_s,reach=boundingRadius(body)+Math.hypot(v.x,v.y,v.z)*dt+cell;
      const cx=body.position_m.x-origin_m.x,cy=body.position_m.y-origin_m.y,cz=body.position_m.z-origin_m.z;
      // The tiles the reach touches and one beyond on each side, held to the stored tiles.
      const x0=uniformTileNear(a,0,Math.floor((cx-reach)/wx)-1),x1=uniformTileNear(a,0,Math.floor((cx+reach)/wx)+1);
      const y0=uniformTileNear(a,1,Math.floor((cy-reach)/wy)-1),y1=uniformTileNear(a,1,Math.floor((cy+reach)/wy)+1);
      const z0=uniformTileNear(a,2,Math.floor((cz-reach)/wz)-1),z1=uniformTileNear(a,2,Math.floor((cz+reach)/wz)+1);
      for(let z=z0;z<=z1;z++)for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++)tiles[uniformBoxTileIndex(a,x,y,z)]=1;
    }
    return tiles;
  }
  /** A live voxel edit on the mixed path, as the tiles the mask reported
   * while it folded the edit (uniformMixedSolidEdit). The frame's solid kernels read the
   * host mask directly; here every tile within one cell of a changed voxel
   * (and the solid-coupled tiles around it, liquid-conditional promotion's
   * rule) joins the next census's band (a GPU join), which the frame head
   * adopts before it displaces liquid out of the new solid. A requests-only
   * layout takes the edit through its request instead (the caller's
   * syncMixedLayout): the solid mask of the next census, when contact is
   * requested, and cut 4h owners otherwise. */
  private editMixedSolids(edit:ReturnType<typeof uniformMixedSolidEdit>):void{
    const frame=this.mixedFrame!,touched=this.mixedSolidTouched!;
    const dimensions=fluidRegionLattice(this.scene).dimensions as [number,number,number];
    try{
      // The promotion held for the mask before this edit (the caller has
      // stamped the edited one) follows it by the touched tiles. A first
      // build or another terrain is the lattice scan (mixedSolidPromotion).
      const terrain=sceneHasTerrain(this.scene)?this.mixedTerrainCells:undefined,promotion=this.mixedSolidTiles;
      if(promotion&&promotion.stamp===this.mixedSolidMaskStamp-1&&promotion.terrain===terrain){
        const solid=uniformMixedSolidTilesAfterEdit(promotion.tiles,dimensions,this.solidMask,edit,terrain);
        this.mixedSolidTiles={stamp:this.mixedSolidMaskStamp,terrain,tiles:{...solid,any:solid.coupledTiles>0}};
      }
      if(this.mixedGpuLayout&&this.detail.policy==="dynamic"){
        const solid=this.mixedSolidPromotion();
        // Refresh the builder statics and census solid mask now: the next census reads them.
        this.refreshMixedBuilderStatics(solid);
        this.joinMixedEdit(solid.coupled,edit.tiles);
      }else if(this.detail.policy!=="dynamic"&&true){
        // Requests only: nothing else asks for a new solid's tiles when solid
        // contact is off, and a dry promotion skips them when it is on. The
        // join holds them at h for the build that displaces (syncMixedLayout
        // counts them into the capacity before it).
        this.joinMixedEdit(this.mixedSolidPromotion().coupled,edit.tiles);
      }
    }finally{for(const t of edit.tiles)touched[t]=0;}
    frame.editSolids();
  }
  /** One byte per tile, zero between edits: the tiles of the edit being applied (editMixedSolids). */
  private mixedSolidTouched?:Uint8Array;
  /** The edit's join: its touched tiles and, liquid-conditional promotion's
   * rule with those as the liquid (the census's solidActive and
   * solidPromote), the 26
   * neighbours of each that is solid-coupled, over the tiles a join already
   * holds. Walked from the touched tiles: the edit's cost, not the lattice's. */
  private joinMixedEdit(coupled:Uint8Array,tiles:readonly number[]):void{
    const a=this.address,p=this.mixedEditTile,held=this.mixedEditJoin,join=held?held.tiles.slice():new Uint8Array(a.boxTiles);
    for(const t of tiles){
      join[t]=1;
      if(!coupled[t])continue;
      const [tx,ty,tz]=uniformBoxTileCoord(a,t,p);
      for(let z=tz-1;z<=tz+1;z++)for(let y=ty-1;y<=ty+1;y++)
        for(let x=tx-1;x<=tx+1;x++)if(uniformTileInBox(a,x,y,z))join[uniformBoxTileIndex(a,x,y,z)]=1;
    }
    this.mixedDynamic!.join(this.mixedTileWords(join));this.mixedEditJoin={tiles:join};
  }
  /** The set tiles of a box-shaped tile mask as tile words of the window (uniformMixedTileWord). */
  private mixedTileWords(mask:Uint8Array):number[]{
    const a=this.address,p=this.mixedEditTile,words:number[]=[];
    for(let t=0;t<mask.length;t++)if(mask[t]){uniformBoxTileCoord(a,t,p);words.push(uniformMixedTileWord(p[0],p[1],p[2]));}
    return words;
  }
  /** The scene's Fine and Coarse regions as the builder's and the census's region table. */
  private mixedRegionTable():Uint32Array<ArrayBuffer>{
    const lattice=fluidRegionLattice(this.scene),regions:UniformMixedRegion[]=[];
    for(const r of this.scene.fluid.refinementRegions??[]){
      const tier=uniformRegionDetailTier(r),box=uniformDetailTileBounds(lattice,{min:r.min_m,max:r.max_m});
      if(box&&(tier==="fine"||tier==="coarse"))regions.push({min:box.min,max:box.max,kind:tier});
    }
    return uniformMixedRegionTable(regions);
  }
  /** Static h tiles: the Fine regions' (the region table). Solid promotion
   * is liquid-conditional, in the census band (solidActive/solidPromote). A
   * Coarse region outside every Fine one holds no band tile but those a
   * solid promotes (the census's decide). */
  private refreshMixedBuilderStatics(solid:ReturnType<WebGPUUniformReferenceSolver["mixedSolidPromotion"]>):void{
    const plan=this.planDetail(),table=this.mixedRegionTable();
    this.mixedBuilder!.setRegions(table);this.mixedDynamic!.setRegions(table);this.mixedDynamic!.setSolid(this.mixedTileWords(solid.coupled));
    this.mixedStaticsOf=undefined;this.mixedSolidSet=solid.coupled;
    this.holdDetailStaticKey();
    this.publishDetail(plan);
  }
  /** A requests-only relayout's statics: the scene's region table (a
   * Requested or Full policy's request tiles are no longer static h tiles:
   * the table holds boxes, not a mask), and the census's solid mask only when
   * solid contact is requested; a solid mask the census holds is not written
   * again. */
  private refreshRequestStatics(request:NonNullable<WebGPUUniformReferenceSolver["mixedRequest"]>):void{
    const builder=this.mixedBuilder!,dynamic=this.mixedDynamic!,fine=request.tiles.fine;
    if(this.mixedStaticsOf!==request.tiles){const table=this.mixedRegionTable();builder.setRegions(table);dynamic.setRegions(table);this.mixedStaticsOf=request.tiles;}
    request.moved.length=0;
    const coupled=request.coupled??(this.mixedNoSolid?.length===fine.length?this.mixedNoSolid:this.mixedNoSolid=new Uint8Array(fine.length));
    if(coupled!==this.mixedSolidSet){dynamic.setSolid(this.mixedTileWords(coupled));this.mixedSolidSet=coupled;}
    this.holdDetailStaticKey();
  }
  /** A frame's encode or receipt failed: fatal for the solver. */
  private failMixedFrame(error:unknown):void{
    if(this.disposed||this.pressureFrameFailure)return;
    this.pressureFrameFailure=error instanceof Error?error:new Error(String(error));
    this.executionInfo.simulationPipelineError=this.pressureFrameFailure.message;this.watchMixedAllocation();
  }
  private assertMixedOptions(): void {
    const options = this.mixedOptions;
    options.velocityTransport = this.velocityTransport; options.liquidOnlyVelocityAdvection = this.liquidOnlyVelocityAdvection;
    assertUniformMixedOptions(options);
  }
  /** assertMixedOptions' record, rewritten by each check. */
  private readonly mixedOptions: UniformMixedOptionalFeatures = { velocityTransport: "semi-lagrangian", liquidOnlyVelocityAdvection: false };

  private initializationTasks(signal?: AbortSignal): GPUInitializationTask[] {
    const tasks = [...this.rigidSystem.initializationTasks()];
    const compiler = gpuCompilationManagerFor(this.device);
    // Packed detail storage: the two native t=0 kernels address the fields
    // through the storage's accessors.
    const packed = this.detailStorage?.layout.placement === "packed" ? this.detailStorage : undefined;
    const shaderModule = compiler.createShaderModule({ label: "Uniform reference kernels",
      code: packed ? uniformDetailDenseShader(this.shaderSource, UNIFORM_REFERENCE_DETAIL_CLASSES) : this.shaderSource });
    const layout = this.mainPipelineLayout;
    const compiled: Partial<UniformReferencePipelines> = {};
    // Uniform Geometric advances only through the mixed frame: of the native
    // kernels it dispatches just the t=0 dense authority seed (plus uvPublish).
    const programs = [["extrapolationAuthorityDense", "Seed dense interface authority", "buildDenseExtrapolationAuthority"]] as const;
    const ids = programs.map(([key]) => `uniform.pipeline.${key}`);
    programs.forEach(([key, label, entryPoint], index) => tasks.push({
      id: ids[index], phase: "solver-pipelines", label,
      run: async () => {
        compiled[key] = await compiler.compileComputePipeline({
          label: `Uniform reference - ${entryPoint}`,
          layout,
          compute: { module: shaderModule, entryPoint },
        }, { priority: "visible", signal });
      },
    }));
    // The t=0 presentation is the only native geometric pass left: every
    // advance, and its publication, belongs to the mixed-ownership frame.
    {
      const id = "uniform.volume.uvPublish"; ids.push(id);
      tasks.push({ id, phase: "solver-pipelines", label: "uvPublish", run: async () => {
        this.initialPublishPipeline = await compiler.compileComputePipeline({
          label: "Uniform Geometric - uvPublish", layout,
          compute: { module: shaderModule, entryPoint: "uvPublish" },
        }, { priority: "visible", signal });
      } });
    }
    { const id="uniform.fields.scratch-copy"; ids.push(id);
      tasks.push({id,phase:"solver-pipelines",label:"Uniform scratch field copy",run:()=>this.fieldPages.initialize(signal)}); }
    const pipelineReadyId = "uniform.pipeline.publish";
    tasks.push({ id: pipelineReadyId, phase: "solver-pipelines", label: "Publish uniform reference programs", dependencies: ids, run: () => {
      this.pipelines = compiled as UniformReferencePipelines;
    } });
    const extrapolationReadyId = "uniform.extrapolation.pipelines";
    tasks.push({
      id: extrapolationReadyId,
      phase: "solver-pipelines",
      label: "Compile paper Sec. 3.3 extrapolation programs",
      run: () => this.velocityExtrapolator.initialize(signal),
    });
    const multigridReadyId = "uniform.pressure.multigrid";
    tasks.push({
      id: multigridReadyId,
      phase: "solver-pipelines",
      label: "Publish the uniform pipeline's facts",
      run: () => { this.publishUniformPipelineFacts(); },
    });
    tasks.push({ id: "uniform.surface.initial", phase: "upload", label: "Reconstruct smooth t=0 surface", dependencies: [pipelineReadyId, extrapolationReadyId, multigridReadyId], run: () => { this.executionInfo.allocatedBytes += this.fieldPages.allocationOverheadBytes; if (!packed) this.encodeInitialPresentationSurface(); } });
    tasks.push({ id: "uniform.warmup", phase: "warmup", label: "Fence uniform t=0 uploads", dependencies: ["uniform.surface.initial"], run: async () => { await this.device.queue.onSubmittedWorkDone(); } });
    return tasks;
  }

  private encodeInitialPresentationSurface(): void {
    if (!this.pipelines) throw new Error("Uniform reference pipelines are not initialized");
    this.writeParams(0, Math.min(this.scene.rigidBodies.length, 12), 0);
    const encoder = this.device.createCommandEncoder({ label: "Uniform reference t=0 surface reconstruction" });
    // Seed static rho'/face-open authority once across the lattice. Runtime
    // updates only need the rolling active box; untouched air keeps this exact
    // static boundary state instead of paying the same full pass every frame.
    // Packed detail storage: the same two kernels through the storage's
    // accessors (initializationTasks), which deposit the canonical texels.
    const packed = this.detailStorage?.layout.placement === "packed" ? this.detailStorage : undefined;
    const groups = packed ? this.initialDetailGroups() : { authority: this.extrapolationAuthorityGroup, publish: this.wallFilmResolveGroup };
    this.runDirect(encoder, "Uniform initial Sec. 3.3 interface authority",
      this.pipelines.extrapolationAuthorityDense, groups.authority,
      [Math.ceil(this.executionInfo.nx / 4), Math.ceil(this.executionInfo.ny / 4), Math.ceil(this.executionInfo.nz / 4)]);
    {
      // Use the same page traversal as runtime publication. A dense-shaped
      // launch cannot cover a page-shaped entry point on a large domain.
      this.run(encoder, "Uniform Geometric initial page surface publication",
        this.initialPublishPipeline!, groups.publish);
    }
    // The two kernels store through no twin: beside an h generation the
    // fields' bases take their canonical texels from it.
    packed?.encodeBaseRefresh(encoder, [this.velocityD, this.velocityB, this.surfaceA, this.surfaceB, this.gammaB]);
    this.device.queue.submit([encoder.finish()]);
  }

  /** The parameter block has been written: the all-4h solid record may be built from it. */
  private paramsWritten = false;
  private frameParamsSeeded = false;
  private movingWindowStep = 0;
  /** Probe-only, off by default: force a page shift at a numerical step. */
  probeWindowShift(step:number,delta:readonly [number,number,number]):void{
    if(!Number.isInteger(step)||step<1||!delta.every(v=>(v|0)===v))throw new RangeError("Invalid window probe shift");
    const pool=this.mixedFrame!.pagePool;this.device.queue.writeBuffer(pool.state,4*(pool.layout.state.window+20),new Int32Array([step,...delta]));
  }
  /** Counts applySceneUniforms: what is derived from the scene alone is derived once per count, not once per step. */
  private sceneRevision = 1;
  private paramsSceneRevision = 0;
  /** writeParams' head words (the Float32Array the queue copies). */
  private readonly paramsWords = new Float32Array(72);
  private writeParams(dt: number, activeBodyCount: number, inflowStrength: number, drop?: UniformLiquidDrop, volumeCells = this.referenceVolumeCells): void {
    const c = this.scene.container;
    this.paramsWritten = true;
    // The head's words in a scratch the queue copies: one lane per line.
    const f = this.paramsWords;
    // The lanes that follow the scene alone (its lattice, container, liquid
    // and nozzle) are written when it is applied and stand in the scratch
    // between: each is a number read from the scene, which is a heap number a
    // step until this function is optimised.
    if (this.paramsSceneRevision !== this.sceneRevision) {
      this.paramsSceneRevision = this.sceneRevision;
      const width = c.width_m, height = c.height_m, depth = c.depth_m;
      const inflow = this.scene.fluid.inflow, outlet = this.inflowBoundary?.outletCenter_m;
      f[0] = this.executionInfo.nx;
      f[1] = this.executionInfo.ny;
      f[2] = this.executionInfo.nz;
      f[4] = width / this.containerCells[0];
      f[5] = height / this.containerCells[1];
      f[6] = depth / this.containerCells[2];
      f[7] = this.scene.fluid.gravity_m_s2.y;
      // The frame's origin: the world position of lattice cell (0,0,0)'s low corner.
      const origin_m = this.frame.origin_m;
      f[8] = origin_m[0];
      f[9] = origin_m[1];
      f[10] = origin_m[2];
      f[12] = this.scene.fluid.density_kg_m3;
      f[13] = this.scene.fluid.dynamicViscosity_Pa_s;
      f[16] = this.scene.fluid.surfaceTension_N_m;
      f[20] = outlet?.x ?? 0;
      f[21] = outlet?.y ?? 0;
      f[22] = outlet?.z ?? 0;
      f[23] = inflow?.radius_m ?? 0;
      f[24] = inflow?.velocity_m_s.x ?? 0;
      f[25] = inflow?.velocity_m_s.y ?? 0;
      f[26] = inflow?.velocity_m_s.z ?? 0;
      f[27] = this.inflowBoundary?.apertureScale ?? 0;
      f[30] = c.fillFraction * this.containerCells[1];
    }
    f[3] = dt;
    f[11] = sceneHasTerrain(this.scene) ? 1 : 0;
    // The mixed frame's fine reach in 4h tiles, or -1 with two-level off: the
    // two-level branch in sampleVelocityComponent is then never taken.
    // w: the retired V-claims-pressure-rows lane, always zero.
    f[14] = this.twoLevelEnabled ? UNIFORM_TWO_LEVEL_FINE_REACH : -1;
    f[15] = 0;
    f[17] = c.fluidWallMode === "no-slip" ? 1 : 0;
    f[18] = activeBodyCount;
    f[19] = this.address.openTop ? 1 : 0;
    f[28] = inflowStrength;
    f[29] = volumeCells;
    f[31] = 4;
    // tuning.xyz: the box's low cell in the lattice (umSolidBoxCell), zero but in an open world.
    f[32] = this.frame.boxCell[0];
    f[33] = this.frame.boxCell[1];
    f[34] = this.frame.boxCell[2];
    f[35] = c.depthBoundary === "symmetry" ? 1 : 0;
    f[36] = drop?.centre_m.x ?? 0;
    f[37] = drop?.centre_m.y ?? 0;
    f[38] = drop?.centre_m.z ?? 0;
    f[39] = drop?.radius_m ?? 0;
    f[40] = drop?.halfHeight_m ?? 0;
    f[41] = this.liquidOnlyVelocityAdvection ? 1 : 0;
    // w: velocityD still holds this advance's V_face authority when pressure
    // runs. MacCormack and the stage audit both overwrite it after the store.
    f[42] = 0;
    f[43] = this.sharedFaceOpenValid() ? 1 : 0;
    // The shell reach in 4h tiles, whether the extension's finest passes and
    // the far-air advection arm run on those tiles, and the transport lane
    // (reach + 8, or -1 with the live set off). All inert while two-level is off.
    f[44] = this.twoLevelShellReach;
    f[45] = this.twoLevelEnabled ? 1 : 0;
    f[46] = this.twoLevelEnabled ? 1 : 0;
    f[47] = this.twoLevelEnabled && 0 > 0 ? UNIFORM_TRANSPORT_REACH + 8 : -1;
    // agreement: the retired compaction, phi seed and shift lanes; clamp.
    f[48] = 0;
    f[49] = 0;
    f[50] = 0;
    f[51] = UNIFORM_PHI_AGREEMENT_CLAMP;
    // lean.x: the no-cut-cell certificate. All three sources of a cut cell
    // are host state -- the packed voxel mask, the terrain heightfield and
    // the live body list -- so this is decided here once a step rather than
    // probed per sample. See uvSolidFree.
    f[52] = this.solidVoxelsEmpty && !sceneHasTerrain(this.scene) && activeBodyCount === 0 ? 1 : 0;
    // lean.yzw and splash: retired census window, phi far-air arm, per-tile
    // reach and splash-survival lanes, always zero.
    f[53] = this.movingWindowStep;
    f[54] = 0;
    f[55] = 0;
    f[56] = 0;
    f[57] = 0;
    f[58] = 0;
    f[59] = 0;
    // splashB: cubic phi advection, retired seed cells, retired ghost-phi
    // drain, retired airborne momentum.
    f[60] = this.phiCubicAdvection ? 1 : 0;
    f[61] = 0;
    f[62] = 0;
    f[63] = 0;
    f[64] = 0;
    f[65] = 0;
    f[66] = 0;
    f[67] = 0;
    // extrusion: the edge count is what selects the outline over the ball,
    // so it is rewritten every step and zero on all but an extrusion's own.
    f[68] = (drop?.outline?.length ?? 0) / 4;
    f[69] = drop?.offset_m ?? 0;
    f[70] = drop?.halfDepth_m ?? 0;
    f[71] = drop?.edgeRadius_m ?? 0;
    if(!this.frameParamsSeeded){this.device.queue.writeBuffer(this.params,0,f);this.frameParamsSeeded=true;}
    else{
      // frame.xyz and tuning.xyz belong to the GPU after the initial seed.
      this.device.queue.writeBuffer(this.params,0,f,0,8);
      this.device.queue.writeBuffer(this.params,44,f,11,21);
      this.device.queue.writeBuffer(this.params,140,f,35,37);
    }
    if (drop?.outline) this.device.queue.writeBuffer(this.params, UNIFORM_PARAMS_HEAD_BYTES + 16, drop.outline);
  }

  /** Queue the pool pages a drop's bounding box touches (grown as a pinned source's is) for the next page commit. Allocates nothing. */
  private requestDropPages(drop: UniformLiquidDrop): void {
    const pool = this.mixedFrame!.pagePool, f = this.frame, o = f.origin_m, s = f.size_m, cells = f.cells;
    const r = Math.max(drop.radius_m, drop.halfHeight_m ?? 0), last = pool.layout.box.pages, b = pool.layout.box.origin, open=this.scene.container.walls==="none";
    const x0 = uniformDropPage(drop.centre_m.x, r, o[0], s[0], cells[0], b[0], last[0], -1, open), x1 = uniformDropPage(drop.centre_m.x, r, o[0], s[0], cells[0], b[0], last[0], 1, open);
    const y0 = uniformDropPage(drop.centre_m.y, r, o[1], s[1], cells[1], b[1], last[1], -1, open), y1 = uniformDropPage(drop.centre_m.y, r, o[1], s[1], cells[1], b[1], last[1], 1, open);
    const z0 = uniformDropPage(drop.centre_m.z, r, o[2], s[2], cells[2], b[2], last[2], -1, open), z1 = uniformDropPage(drop.centre_m.z, r, o[2], s[2], cells[2], b[2], last[2], 1, open);
    for (let z = z0; z <= z1; z++) for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) pool.request(x, y, z);
  }

  /**
   * The scheduled drop this step adds: the one whose time lies in
   * (start, end]. Steps tile the clock, so each drop is added exactly once.
   */
  private scheduledDrop(start_s: number, end_s: number): UniformLiquidDrop | undefined {
    const schedule = this.scene.fluid.scheduledDrops;
    if (!schedule) return undefined;
    if (this.scheduledDrops?.schedule !== schedule) {
      const c = this.scene.container, [nx, ny, nz] = this.containerCells;
      const cell_m = Math.min(c.width_m / nx, c.height_m / ny, c.depth_m / nz);
      this.scheduledDrops = { schedule, drops: schedule.map(({ volume }) => uniformScheduledDrop(volume, cell_m)) };
    }
    // Accumulated step ends miss a time written as a fraction by an ulp either
    // way; the same slack on both ends keeps the intervals a tiling.
    const slack = 1e-6 * (end_s - start_s);
    let due = 0, first = -1;
    for (let index = 0; index < schedule.length; index++) {
      const time_s = schedule[index]!.time_s;
      if (time_s > start_s + slack && time_s <= end_s + slack) { if (!due++) first = index; }
    }
    if (due > 1) throw new Error(`Uniform scheduled drops: ${due} drops fall in the step ending at ${end_s} s (the first at ${schedule[first]!.time_s} s); a step adds one drop, so space them at least a step apart`);
    return due ? this.scheduledDrops.drops[first] : undefined;
  }

  /**
   * Add a ball of liquid to the solve that is already running.
   *
   * The alternative is re-seeding from the edited document, which throws away
   * the run the user is watching in order to add water to it — so authoring a
   * ball at t > 0 would mean losing everything that had happened. This is the
   * same mass source the nozzle uses, on the same guard, applied on exactly one
   * step: the ball appears in the field where it was dropped and is transported
   * from there like any other liquid.
   *
   * The next step asks for its pages and the one after consumes it, which is
   * also what happens if the clock is paused when it is called — the ball
   * lands when the clock next runs.
   */
  injectLiquidBall(ball: InjectedLiquidBall): void {
    this.mixedFrame?.invalidateExtension();
   
    if (!(ball.radius_m > 0)) return;
    this.pendingDrop = uniformInjectedDrop(ball);
    this.pendingDropPaged = false;
  }

  /** The editor's water shapes, as far as the one-step drop source reaches: a ball, added. */
  async editFluid(edit: LiveFluidEdit): Promise<LiveFluidEditResult> {
    this.mixedFrame?.invalidateExtension();
   
    if (edit.operation !== "add" || edit.shape !== "ball") {
      return { accepted: false, reason: "This fluid method adds water as balls only; choose Sparse Geometric to remove water or drop other shapes." };
    }
    const frame = this.frame;
    const refusal = liveFluidEditRefusal(edit, { origin_m: frame.container_m, cellSize_m: frame.cellSize_m, dimensions: frame.cells });
    if (refusal) return { accepted: false, reason: refusal };
    if (this.pendingDrop) return { accepted: false, reason: "The previous drop lands when the clock next runs; run the simulation first." };
    this.injectLiquidBall({ centre_m: edit.center_m, radius_m: edit.radius_m });
    return { accepted: true };
  }

  /**
   * Adopt controls whose pipelines and storage are already resident.
   *
   * The renderer calls this once per resolved frame. Keep it allocation-free
   * and idempotent: stage gates alter host-side command encoding, while the
   * sharpening scalars reach WGSL through the existing per-advance params
   * buffer. Turning Sec. 3.8 on while paused is the one transition that needs
   * work immediately, so reconstruct the current density once for display.
   */
  applyRuntimeValues(values: MethodParamValues): void {
    // The renderer calls this every frame. Identical controls must not retire
    // geometry/support caches, the census extension, or rewrite the pressure
    // hierarchy's uniforms, nor drain the frame pipeline to apply nothing.
    const previous=this.deferredFrameValues??this.appliedRuntimeValues, keys=Object.keys(values);
    if(previous && keys.length===Object.keys(previous).length
      && keys.every(key=>Object.hasOwn(previous,key)&&Object.is(previous[key],values[key]))) {
      return;
    }
    if (this.mixedFrameInFlight) { this.deferredFrameValues = {...values}; return; }
    this.mixedFrame?.invalidateExtension();
    // The renderer reapplies the latest values before the next advance.
    const finite = (key: string, fallback: number, minimum: number, maximum: number) => {
      const value = Number(values[key]);
      return Number.isFinite(value) ? Math.min(maximum, Math.max(minimum, value)) : fallback;
    };
    this.narrowBandFinePadding=finite("fineGridPadding",this.narrowBandFinePadding,0,4);
    if(values.phiFineAdvection!==undefined)this.phiFineTrilinear=values.phiFineAdvection==="trilinear";
    this.narrowBandAdaptiveBudgetPercent=finite("adaptiveBudgetPercent",this.narrowBandAdaptiveBudgetPercent,0,100);
    this.narrowBandAdaptiveFadeSeconds=finite("adaptiveFadeSeconds",this.narrowBandAdaptiveFadeSeconds,0.05,2);
    this.narrowBandSparseParticleLifetime=finite("sparseParticleLifetime",this.narrowBandSparseParticleLifetime,0,10);
    this.narrowBandVolumeControlSeconds=finite("volumeControlSeconds",this.narrowBandVolumeControlSeconds,0,2);
    this.pressureMultigrid.setResidualTolerance(finite("pressureResidualTolerance", UNIFORM_PRESSURE_RESIDUAL_TOLERANCE, 0, 100));
    // Switching to "fixed" mid-run restores the full encoded schedule on the
    // next step; switching back drops the previous demand sample and uses
    // the startup budget.
    const lagged = values.pressureCycleBudget !== "fixed";
    this.pressureCycleBudgetLagged = lagged;
    const postProcessing = uniformDensityPostProcessingEnabled(
      values.densityPostProcessing,
      this.scene.sceneId,
    );
    const refreshPresentation = postProcessing !== this.densityPostProcessing;
    this.densityPostProcessing = postProcessing;
    this.gammaDiffusionIterations = values.gammaDiffusion === "off" ? 0 : Math.round(finite(
      "gammaDiffusionIterations", UNIFORM_GAMMA_DIFFUSION_DEFAULT_ITERATIONS, 1,
      UNIFORM_GAMMA_DIFFUSION_MAX_ITERATIONS,
    ));
    this.solidExcessCorrection = values.solidExcessCorrection !== "off";
    this.rigidCoupling = values.rigidCoupling !== "off";
    this.fixedStep_s = uniformFixedStep_s(values.timeStep);
    if(values.velocityTransport === "maccormack" && this.velocityC.width===1 && this.executionInfo.nx>1)
      throw new Error("Changing velocity transport requires rebuilding the Uniform Geometric solver");
    this.velocityTransport = values.velocityTransport === "maccormack"
      ? "maccormack" : "semi-lagrangian";
    this.liquidOnlyVelocityAdvection = values.liquidOnlyVelocityAdvection === "on";
    // Partial-values callers must not reset the sweep budget.
    if (values.extensionFrontSweeps !== undefined && Number(values.extensionFrontSweeps) !== this.extensionSweeps) {
      this.setExtensionSweeps(Number(values.extensionFrontSweeps));
      this.publishUniformPipelineFacts();
    }
    {
      // Absent detail keys keep their current settings.
      const detail = uniformDetailSettings({...uniformDetailValues(this.detail), ...Object.fromEntries(UNIFORM_DETAIL_PARAM_KEYS.filter(k => values[k] !== undefined).map(k => [k, values[k]]))});
      if (!sameUniformDetailSettings(detail, this.detail)) {
        // Solid contact changes on a running state: the remap hands a cut
        // tile to a 4h owner with its plane at the corners the h rule buried,
        // and splits one back by its cells' open fractions.
        const accepted = this.detail;
        this.detail = detail;
        // The relayout follows from the next frame's head (time is kept):
        // Dynamic's static key sees the change, a requests-only layout
        // takes its request here.
        // Refused: the accepted generation keeps running, under the settings
        // its frames were built for. The request is withdrawn (asked again
        // only when it is made again); its rejection stays reported.
        if (!this.syncMixedLayout()) { this.detail = accepted; this.mixedRefusedKey = ""; }
      } else if (this.executionInfo.uniformDetail?.rejected && UNIFORM_DETAIL_PARAM_KEYS.some(k => values[k] !== undefined)) {
        // The request is the running generation again: nothing is refused.
        this.executionInfo.uniformDetail = { ...this.executionInfo.uniformDetail, rejected: undefined };
      }
      if (values.phiCubicAdvection !== undefined) this.phiCubicAdvection = values.phiCubicAdvection === "on";
      this.solidExcessCorrection = false;
      this.densityPostProcessing = false;
      this.gammaDiffusionIterations = 0;
    }
    if (refreshPresentation && this.pipelines) {
      this.encodeInitialPresentationSurface();
    }
    this.appliedRuntimeValues={...values};
  }

  /** Pays the reduction the last step skipped, from the fields it left behind. */


  private initializeVolumeAndTerrain(): void {
    const { nx, ny, nz } = this.executionInfo;
    const c = this.scene.container;
    const {volume,terrain,initial,dam} = this.initialVolume ?? uniformLatticeInitialVolume(this.scene);
    this.initialVolume = undefined;
    const cellHeight = c.height_m / this.containerCells[1];
    if (this.vertexPhiField && this.vertexPhiScratch) {
      // Cut 4h solid owners read their plane at corners the h rule buries.
      const phi = uniformLatticeInitialPhi(this.scene, uniformDetailCoarseSolids(this.detail));
      this.upload3DF32(this.vertexPhiField, phi);
      this.upload3DF32(this.vertexPhiScratch, phi);
    }
    this.upload3DF32(this.volumeA, volume);
    this.upload3DF32(this.volumeB, volume);
    this.upload3DF32(this.surfaceA, volume);
    this.upload3DF32(this.surfaceB, volume);
    const gamma = new Float32Array(this.address.boxCells);
    this.upload3DF32(this.gammaA, gamma);
    this.upload3DF32(this.gammaB, gamma);
    const zeroBoundaryVelocity = new Float32Array(this.executionNegativeBoundaryVelocityBytes / 4);
    this.device.queue.writeBuffer(this.boundaryVelocityA, 0, zeroBoundaryVelocity);
    this.device.queue.writeBuffer(this.boundaryVelocityB, 0, zeroBoundaryVelocity);
    this.device.queue.writeBuffer(this.boundaryVelocityC, 0, zeroBoundaryVelocity, 0, this.boundaryVelocityC.size / 4);
    this.device.queue.writeBuffer(this.boundaryVelocityD, 0, zeroBoundaryVelocity);
    this.referenceVolumeCells = initial;
    const terrainCells = Float32Array.from(terrain, (height) => height / cellHeight);
    // An open world's columns are the window's, baked with the first layout (initializeMixedFrame).
    if (!this.openWorldTerrain) this.upload2DF32(this.terrainTexture, terrainCells, nx, nz);
    this.mixedTerrainCells = terrainCells;
    Object.assign(this.executionInfo, {
      initialVolumeCellSum: initial, volumeCellSum: initial,
      representedVolumeCellSum: initial, volumeDrift: 0, representedVolumeDrift: 0,
      rawVolumeDrift: 0, volumeTelemetrySource: "initial-condition",
      maxSpeed_m_s: 0,
      front_m: this.scene.fluid.initialCondition === "dam-break"
        ? -c.width_m / 2 + dam.max.x * c.width_m : c.width_m / 2,
      frontTelemetrySource: "initial-condition",
    });
  }

  /** Initialization hook for manufactured GPU boundary fixtures. Public textures
   * are read-only presentation adapters when fields use page storage. */
  initializeVelocityForQA(values: Float32Array): void {
    this.mixedFrame?.invalidateExtension();
    if(this.lastTime!==0 || values.length!==4*this.executionInfo.cellCount)throw new Error("Velocity fixture must initialize the complete field at t=0");
    this.upload3DF32(this.velocityA,values);
  }

  private upload3DF32(texture: GPUTexture, values: Float32Array): void {
    const field=uniformDetailField(texture);
    if(field) { field.storage.upload(texture,values); return; }
    this.fieldPages.upload(texture,values);
  }

  private upload2DF32(texture: GPUTexture, values: Float32Array, nx: number, nz: number): void {
    const rowBytes = nx * 4, padded = Math.ceil(rowBytes / 256) * 256;
    const packed = new Uint8Array(padded * nz), source = new Uint8Array(values.buffer);
    for (let z = 0; z < nz; z += 1) packed.set(source.subarray(rowBytes * z, rowBytes * (z + 1)), padded * z);
    this.device.queue.writeTexture({ texture }, packed, { bytesPerRow: padded, rowsPerImage: nz }, { width: nx, height: nz });
  }

  private dispatch(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, group: GPUBindGroup): void {
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(
      Math.ceil(this.executionInfo.nx / 4), Math.ceil(this.executionInfo.ny / 4), Math.ceil(this.executionInfo.nz / 4));
  }

  private sharedFaceOpenValid(): boolean {
    return this.velocityTransport !== "maccormack" && !this.symmetryStageAuditFields;
  }

  private run(encoder: GPUCommandEncoder, label: string, pipeline: GPUComputePipeline, group: GPUBindGroup): void {
    const pass = encoder.beginComputePass({ label });
    this.dispatch(pass, pipeline, group);
    pass.end();
  }

  private runDirect(encoder: GPUCommandEncoder, label: string, pipeline: GPUComputePipeline,
    group: GPUBindGroup, workgroups: readonly [number, number, number]): void {
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(...workgroups); pass.end();
  }

  /**
   * Instance-accurate pass counts for the pipeline panel's chips. Plain data on
   * the info record so it survives the worker structured-clone boundary; the
   * counts depend on grid dimensions, so they publish once the multigrid plan
   * and extrapolation hierarchy exist.
   */
  private publishUniformPipelineFacts(): void {
    // The mixed pressure root runs the traversal itself: the hierarchy encodes no passes of its own.
    const multigridPasses = { setup: 0, "full-cycle": 0, "v-cycle": 0, finish: 0 };
    this.executionInfo.uniformPipelineFacts = {
      extrapolationFrontSweeps: this.extensionSweeps,
      extrapolationHierarchyLevels: this.velocityExtrapolator.hierarchyLevelCount,
      // The hierarchy encodes into the mixed extension's pass.
      extrapolationPassesPerInvocation: 0,
      multigridLevels: this.pressureMultigrid.levelCount,
      multigridPasses,
      multigridPassesTotal: 0,
      pressureSchedule: this.pressureSchedule,
    };
  }

  /**
   * Cycles the next pressure solve encodes, and the telemetry that explains it.
   *
   * Fixed mode returns the configured schedule, so its command stream is the
   * one the solver encoded before P1 existed. Lagged mode sizes the encoded
   * prefix from the latest asynchronous diagnostics sample; the GPU-side
   * residual gate still stops a converged solve inside that prefix.
   */


  /**
   * Ask the queue for the cycle counters this step just wrote.
   *
   * This is the same asynchronous, post-submit readback shape as `readStats`,
   * on its own twelve-byte buffer so the two never contend: nothing in the
   * frame path waits on it, and a step simply skips the copy while an earlier
   * map is still outstanding. It never gates correctness — only how many
   * cycles the *next* step bothers to encode.
   */


  /** Mixed layout views (band reasons, sampler certificate, the frame's
   * starting tiles) are recorded only while a layer draws them. */
  private layoutViews = false;
  setLayoutViewsEnabled(enabled: boolean): void {
    if (enabled === this.layoutViews) return;
    this.layoutViews = enabled;
    this.mixedFrame?.setLayoutViews(enabled);
  }

  /** Two-level sampling is geometric and dimensionally possible. */
  private get twoLevelEnabled(): boolean {
    return this.twoLevelVelocity && this.twoLevelTileCount > 0;
  }

  private createPageAwareGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup {
    const group = this.fieldPages.createBindGroup(descriptor);
    this.groupDescriptors.set(group,descriptor);
    return group;
  }

  /** Mixed frames run ahead of their receipts: the pipeline admits another
   * frame unless UNIFORM_MIXED_RECEIPT_RING receipts are unchecked or a
   * deferred edit is draining the pipeline. Each frame adopts its own
   * layout at its head, so no census gates admission. */
  get framePending(): boolean {
    return this.mixedFramesInFlight >= UNIFORM_MIXED_RECEIPT_RING
      || (this.mixedFrameInFlight && this.deferredFrameEdit);
  }
  /** Scene, value and body edits wait for every submitted frame. */
  private get mixedFrameInFlight(): boolean { return this.mixedFramesInFlight > 0; }
  private get deferredFrameEdit(): boolean {
    return this.deferredFrameScene !== undefined || this.deferredFrameValues !== undefined || this.deferredFrameBodies !== undefined;
  }
  get deferredFramePublication(): boolean {
    return !!this.mixedFrame;
  }
  get presentationPending(): boolean { return this.framePending || this.pressureFrameFailure !== undefined; }
  async awaitFrameCompletion(): Promise<void> {
   
    await this.mixedFrameChain;
    if (this.pressureFrameFailure) throw this.pressureFrameFailure;
    await this.device.queue.onSubmittedWorkDone();
  }
  async assertSimulationHealthy(completion?: Promise<void>): Promise<void> {
    // Presentation already awaited this frame. A second queue-wide fence here
    // includes newer presentations and holds their predecessor's throughput slot.
    await (completion ?? this.awaitFrameCompletion());
  }

  advanceTo(time_s: number, bodies: RigidBodyState[] = []): boolean {
    if(this.disposed)return false;
    if (this.framePending || this.pressureFrameFailure) return false;
    if(this.mixedFrame){
      this.assertMixedOptions();
      // Reject unsupported geometry before advancing the clock or consuming a
      // queued liquid edit. The last accepted frame remains usable for edits.
      // Static voxels, terrain and non-box vessels are fine-owner solids;
      // live voxel edits relayout (editMixedSolids); rigid bodies are solid
      // geometry of the frame's solid library, promoted and coupled per step
      // (syncMixedLayout, UniformMixedBodies).
    }
    // The paper's method is calibrated for its own large-step regime (dt=1/30
    // in every Sec. 4 example): sharpening opposes per-resample transport
    // blur, so far smaller scene steps structurally out-diffuse it.
    // A paper advance is exactly 1/30 s. Treating that value only as maxDt
    // let browser callers feed 4 ms targets, producing a different 250 Hz
    // method than the Dawn/paper lane and overwhelming sharpening with many
    // extra resamples. Accumulate target time until one complete paper step
    // is available; never encode a fractional paper step.
    // Each clock number is read once below: a read of one is a heap number until this is optimised.
    const stepStart_s = this.lastTime, fixedStep_s = this.fixedStep_s;
    if (fixedStep_s !== undefined && !uniformFixedAdvanceReady(time_s, stepStart_s, fixedStep_s)) return false;
    const advance = this.advancePlan;
    if (!planGPUAdvanceInto(advance, time_s, stepStart_s, fixedStep_s ?? this.scene.numerics.maxDt_s)) return false;
    if(!this.mixedFrame) throw new Error("Narrow-band FLIP advances only through the mixed-ownership frame");
    if (!this.pipelines) throw new Error("Uniform reference pipelines are not initialized");
    const dt = advance.dt_s, stepEnd_s = advance.nextTime_s;
    this.lastTime = stepEnd_s;
    this.executionInfo.submittedTime_s = stepEnd_s;
    if (!this.deferredFramePublication) this.executionInfo.simulatedTime_s = stepEnd_s;
    this.executionInfo.simulationLag_s = advance.lag_s;
    this.executionInfo.lastDt_s = dt;
    this.executionInfo.lastSubsteps = 1;
    this.executionInfo.encodedSteps = (this.executionInfo.encodedSteps ?? 0) + 1;
    // The first twelve bodies, in a roster this solver keeps (the sample reads it and keeps nothing).
    const roster = this.advanceRoster, rostered = Math.min(12, bodies.length);
    for (let i = 0; i < rostered; i++) roster[i] = bodies[i]!;
    if (roster.length !== rostered) roster.length = rostered;
    const activeBodies = this.prescribedSolidMotion.sample(roster, dt);
    this.rigidSystem.syncBodies(activeBodies);
    const c = this.scene.container;
    const volumeBeforeSources=this.referenceVolumeCells;let volumeCells=volumeBeforeSources;
    const inflow = this.scene.fluid.inflow;
    const strength = inflow ? averageInflowStrength(inflow, stepEnd_s - dt, stepEnd_s) : 0;
    if (this.inflowBoundary && strength > 0) {
      const cellVolume = c.width_m * c.height_m * c.depth_m
        / (this.containerCells[0] * this.containerCells[1] * this.containerCells[2]);
      volumeCells += this.inflowBoundary.flowRate_m3_s * strength * dt / cellVolume; this.referenceVolumeCells = volumeCells;
    }
    // The drop is consumed by this step and only this step, so the lane is
    // cleared before anything can encode a second one. Its mass joins the
    // reference the same way the nozzle's does — analytically, from the volume
    // asked for rather than from the volume that fitted — because the guard
    // that caps a cell at full is the same guard, and the drift telemetry is
    // the instrument that says how much the two disagreed.
    // A scheduled drop is the scene's own and takes its step; a ball injected
    // for the same one waits for the next, exactly as it waits out a pause.
    const scheduled = this.scheduledDrop(stepStart_s, stepEnd_s);
    // An injected ball lands a step after one asked for its pages, as a
    // scheduled drop does: the census marks a page resident only from the
    // tiles of blocks stored when it runs, so pages first claimed by the
    // ball's own commit stay unmarked and the next census audits its liquid
    // there as absent.
    const injected = !scheduled && this.pendingDropPaged ? this.pendingDrop : undefined;
    const drop = scheduled ?? injected;
    if (injected) { this.pendingDrop = undefined; this.pendingDropPaged = false; }
    this.pageDrop = drop;
    // The scheduled drop the next step adds: its pages are asked for a step
    // early, so the census of its own step finds them stored.
    const schedule = this.scene.fluid.scheduledDrops, ahead = schedule ? uniformPagePoolDropAhead(schedule, stepEnd_s, dt, fixedStep_s ?? this.scene.numerics.maxDt_s) : -1;
    this.pageDropAhead = ahead < 0 ? undefined : this.scheduledDrops!.drops[ahead];
    this.pageInjectedAhead = this.pendingDrop; this.pendingDropPaged = this.pendingDrop !== undefined;
    if (drop) {
      const cellVolume = c.width_m * c.height_m * c.depth_m
        / (this.containerCells[0] * this.containerCells[1] * this.containerCells[2]);
      volumeCells += drop.volume_m3 / cellVolume; this.referenceVolumeCells = volumeCells;
    }
    this.executionInfo.referenceLiquidVolume_cells = volumeCells;
    if(this.mixedFrame){
      this.mixedFrame.solid!.present=!this.solidVoxelsEmpty||sceneHasTerrain(this.scene)||activeBodies.length>0;
      // The layout's owner for this roster: the census follows the bodies' GPU poses.
      try{this.syncMixedLayout(activeBodies,dt,true);}catch(error){this.failMixedFrame(error);return true;}
    }
    this.movingWindowStep++;
    this.writeParams(dt, activeBodies.length, strength, drop, volumeCells);
    if(this.mixedFrame){
      const frame=this.mixedFrame;
      // Dynamic coarsening: the frame's head classifies the state it starts
      // from for this one step and adopts the generation it builds before
      // advecting (horizon one); moving bodies sweep this step's tiles into
      // that census. The tail extends the final velocity for the next head.
      const bodyCount=activeBodies.length;
      // The census's body mark and the coupling read this step's values when
      // the frame encodes them, inside advance below: each is one function.
      // The cell's volume and height follow the scene alone.
      if(this.mixedStepSceneRevision!==this.sceneRevision){
        this.mixedStepSceneRevision=this.sceneRevision;this.mixedStepCellHeight=c.height_m/this.containerCells[1];
        this.mixedStepCellVolume=c.width_m*c.height_m*c.depth_m/(this.containerCells[0]*this.containerCells[1]*this.containerCells[2]);
      }
      this.mixedStepDt=dt;
      this.mixedDynamic?.setBodies(bodyCount?this.markMixedBodies:this.openWorldTerrain?this.markOpenWorldTerrain:undefined);
      // Bodies: the frame rebuilds its solid record and displaces liquid out
      // of the cells they entered; once more after the last one leaves. Native
      // coupleRigid then the rigid integration run after the projection.
      const frameBodies=bodyCount||this.mixedHadBodies?bodyCount&&this.rigidCoupling?this.mixedCoupledBodies:this.mixedUncoupledBodies:undefined;
      this.mixedHadBodies=bodyCount>0;
      let receipt:Promise<UniformMixedFrameReceipt>,kick:Promise<unknown>|undefined;
      try{
        // The frame keeps a step's parameters until its receipt is checked and
        // compares the last step's with this one's: a ring longer than the
        // receipts in flight, each record rewritten when its turn returns.
        const ring=this.mixedParameters,slot=this.mixedParameter=(this.mixedParameter+1)%UNIFORM_MIXED_PARAMETER_RING,p=ring[slot]??={} as UniformMixedFrameParameters;
        p.dt=dt;p.addedVolumeCells=volumeCells-volumeBeforeSources;p.volumeControlSeconds=this.narrowBandVolumeControlSeconds;
        // The scene's own numbers stand in a record until the scene is applied again.
        if(this.mixedParameterScenes[slot]!==this.sceneRevision){
          this.mixedParameterScenes[slot]=this.sceneRevision;
          p.gravity=this.scene.fluid.gravity_m_s2.y;p.density=this.scene.fluid.density_kg_m3;p.viscosity=this.scene.fluid.dynamicViscosity_Pa_s;p.surfaceTension=this.scene.fluid.surfaceTension_N_m;
        }
        p.openTop=this.address.openTop;p.noSlip=this.scene.container.fluidWallMode==="no-slip";p.cubic=this.phiCubicAdvection;p.fineTrilinear=this.phiFineTrilinear;p.pressureTolerance=this.pressureMultigrid.residualTolerance;p.redistance=true;p.extensionSweeps=this.extensionSweeps;
        // Impacts raise pressure demand faster than a plan two frames old
        // follows. One additional reserve covers long-dam frame 79's third
        // cycle under a 4h surface, and the dam 128^3 all-fine far-wall
        // impact's fourth (8 of 8 perturbed runs through 60 frames with it,
        // 7 of 8 without), without encoding the seven-slot envelope.
        p.pressureReserve=1;
        const support=p.supportPolicy??={fineReach:UNIFORM_TWO_LEVEL_FINE_REACH,shellReach:this.twoLevelShellReach,twoLevel:this.twoLevelEnabled,shellOnly:this.twoLevelEnabled};
        support.fineReach=UNIFORM_TWO_LEVEL_FINE_REACH;support.shellReach=this.twoLevelShellReach;support.twoLevel=this.twoLevelEnabled;support.shellOnly=this.twoLevelEnabled;
        // The seed is t=0's velocity; the drift reads it half a step on. The
        // kick is encoded ahead of the first frame with a frame's head (its
        // census, where one is due), so that frame's census bounds the
        // kicked field. Its receipt holds a ring slot until it is checked.
        if(!this.mixedKicked)kick=this.kickMixedFrame(frame,p,dt,!!frameBodies);
        receipt=frame.advance(p,this.mixedFrameTrace(),this.mixedGpuLayout&&(this.detail.policy==="dynamic"||this.mixedLive),frameBodies);
        if(kick)receipt=Promise.all([receipt,kick]).then(([frame])=>frame);
      }catch(error){kick?.catch(()=>undefined);this.mixedRelayoutRead=undefined;this.failMixedFrame(error);return true;}
      this.readMixedRelayout();
      this.mixedFramesInFlight++;
      if(this.mixedAllocationWatch>0){this.mixedAllocationWatch--;this.mixedAllocationStale=true;}
      // The step's completion record (its time, with handlers built once) is
      // its receipt's alone until one of them has run.
      const completion=this.mixedCompletions.take()??this.mixedCompletion();completion.time_s=stepEnd_s;
      const handled=receipt.then(completion.accepted,completion.failed);
      // awaitFrameCompletion's fence: this frame's handling while it is the
      // only one unhandled, else one promise for every frame until none is.
      if(this.mixedFrameChainFailure===undefined){
        if(this.mixedFramesUnhandled===0)this.mixedFrameChain=handled;
        else if(!this.mixedFramesIdle)this.mixedFrameChain=new Promise<void>(this.holdMixedFramesIdle);
      }
      this.mixedFramesUnhandled++;
      return true;
    }

    throw new Error("Narrow-band FLIP requires its mixed ownership frame");
  }

  /** The first frame's half kick (apart from advanceTo: this closure would cost every advance a context). */
  private kickMixedFrame(frame:UniformMixedFrame,p:UniformMixedFrameParameters,dt:number,bodies:boolean):Promise<unknown>{
    this.mixedKicked=true;const k=frame.kick(p,dt/2,bodies);this.readMixedRelayout();this.mixedFramesInFlight++;
    return k.finally(()=>{this.mixedFramesInFlight--;});
  }
  /** One advance's plan and roster, rewritten by each advance. */
  private readonly advancePlan: GPUAdvancePlan = { dt_s: 0, nextTime_s: 0, lag_s: 0 };
  private readonly advanceRoster: RigidBodyState[] = [];
  /** This step's values for the body mark and the coupling the frame encodes inside its advance. */
  private mixedStepDt=0;private mixedStepCellVolume=0;private mixedStepCellHeight=0;private mixedStepSceneRevision=0;
  private readonly markMixedBodies=(encoder:GPUCommandEncoder,tiles:GPUBuffer):void=>{this.mixedBodies!.encodeTiles(encoder,tiles,this.mixedStepDt);this.openWorldTerrain?.encodeTiles(encoder,tiles);};
  private readonly markOpenWorldTerrain=(encoder:GPUCommandEncoder,tiles:GPUBuffer):void=>this.openWorldTerrain!.encodeTiles(encoder,tiles);
  private readonly mixedCoupledBodies:UniformMixedFrameBodies={couple:(encoder:GPUCommandEncoder)=>{
    this.mixedBodies!.encodeCoupling(encoder);this.rigidSystem.encode(encoder,this.mixedStepDt,this.mixedStepCellVolume,1,this.mixedStepCellHeight);
    this.mixedFrame!.solid!.encodeBodies(encoder);
  }};
  private readonly mixedUncoupledBodies:UniformMixedFrameBodies={couple:undefined};
  private readonly mixedParameters:UniformMixedFrameParameters[]=[];
  /** The scene revision each record's scene numbers were read at. */
  private readonly mixedParameterScenes:number[]=[];
  private mixedParameter=0;
  /** Frames whose receipt no handler has finished with, and the fence a second one opens. */
  private mixedFramesUnhandled=0;
  private mixedFramesIdle?:{resolve:()=>void;reject:(error:unknown)=>void};
  private readonly holdMixedFramesIdle=(resolve:()=>void,reject:(error:unknown)=>void):void=>{this.mixedFramesIdle={resolve,reject};};
  /** A handler threw (a deferred edit's): every later fence rejects with it, as the chain of handlers did. */
  private mixedFrameChainFailure?:{error:unknown};
  private readonly mixedCompletions=new KeptStack<{time_s:number;accepted:(receipt:UniformMixedFrameReceipt)=>void;failed:(error:unknown)=>void}>();
  private mixedCompletion(){
    const completion={time_s:0,
      accepted:(receipt:UniformMixedFrameReceipt):void=>{
        try{this.acceptMixedFrame(completion.time_s,receipt);}catch(error){this.failMixedFrame(error);}
        this.mixedCompletions.give(completion);this.settleMixedFrame();
      },
      failed:(error:unknown):void=>{this.failMixedFrame(error);this.mixedCompletions.give(completion);this.settleMixedFrame();}};
    return completion;
  }
  /** A frame's receipt was accepted: its counters, and the clock it completed. */
  private acceptMixedFrame(time_s:number,receipt:UniformMixedFrameReceipt):void{
    if(this.disposed)return;
    const info=this.executionInfo as UniformPublishedInfo,flip=this.mixedFrame?.narrowBandFlip;
    info.narrowBandFlipParticles=flip?.count;info.narrowBandFlipReseedClipped=flip?.reseedClipped;info.narrowBandFlipUnsupportedParticles=flip?.diagnostics.unsupported;info.narrowBandFlipMaxSurfaceDistance=flip?.diagnostics.afterMaxOutside;
    info.simulatedTime_s=time_s;info.completedTime_s=time_s;
    info.uniformPressureAcceptedResidual=receipt.residual;info.uniformPressureCyclesExecuted=receipt.cycles;info.uniformPressureCyclesConverged=true;
    info.uniformPressureCyclesEncoded=receipt.encoded;info.uniformPressureCyclesConfigured=this.pressureSchedule.fullCycles+this.pressureSchedule.vCycles;
    info.uniformVolumeDustCells=receipt.dustOwners;info.uniformVolumeDustMass_cells=receipt.dustMass_cells;
    info.uniformVolumeOrphanDustCells=receipt.orphanDustOwners;info.uniformVolumeOrphanDustMass_cells=receipt.orphanDustMass_cells;
    info.uniformPressureBandTiles=receipt.bandTiles;info.uniformPressureBandCycles=receipt.bandCycles;info.uniformPressureBandResidual=receipt.bandResidual;
    // The detail planner's clock: hysteresis and expiry count accepted frames.
    this.detailAcceptedStep++;
  }
  /** After a frame's receipt was handled either way: the edits that waited for
   * every submitted frame, then the fence once no frame is unhandled. */
  private settleMixedFrame():void{
    try{
      this.mixedFramesInFlight--;
      if(this.mixedAllocationWatch>0)this.mixedAllocationStale=true;
      if(!(this.disposed||this.pressureFrameFailure||this.mixedFrameInFlight)){
        const scene=this.deferredFrameScene,values=this.deferredFrameValues,bodies=this.deferredFrameBodies;
        this.deferredFrameScene=undefined;this.deferredFrameValues=undefined;this.deferredFrameBodies=undefined;
        if(scene)this.applySceneUniforms(scene);if(values)this.applyRuntimeValues(values);if(bodies)this.syncRigidBodies(bodies);
      }
    }catch(error){
      this.mixedFrameChainFailure??={error};
      const idle=this.mixedFramesIdle;this.mixedFramesIdle=undefined;this.mixedFramesUnhandled--;
      // The fence that covers this frame rejects: the shared one, else this handler's own promise.
      if(idle){idle.reject(error);return;}
      throw error;
    }
    if(--this.mixedFramesUnhandled===0){const idle=this.mixedFramesIdle;this.mixedFramesIdle=undefined;idle?.resolve();}
  }
  async readStats(): Promise<GPUEulerianInfo> {
   
    await this.awaitFrameCompletion();
    this.refreshMixedAllocation();
    if(this.mixedFrame){
      if(this.disposed||this.readbackPending)return this.executionInfo;
      this.readbackPending=true;
      this.statsReadback??=this.device.createBuffer({label:"Uniform diagnostics readback",size:264,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      try{
        const encoder=this.device.createCommandEncoder();this.mixedDiagnostics!.encode(encoder);
        encoder.copyBufferToBuffer(this.reductions,0,this.statsReadback,0,24);
        encoder.copyBufferToBuffer(this.params,32,this.statsReadback,80,12);
        encoder.copyBufferToBuffer(this.mixedFrame.ownership.support,this.mixedFrame.ownership.capacity.tiles*16,this.statsReadback,24,16);
        const budget=this.mixedFrame.narrowBandVolumeBudget;
        if(budget){encoder.copyBufferToBuffer(budget,0,this.statsReadback,40,36);this.mixedFrame.requestNarrowBandVolumeProbe();}
        this.device.queue.submit([encoder.finish()]);
        await this.statsReadback.mapAsync(GPUMapMode.READ);const words=new Uint32Array(this.statsReadback.getMappedRange(),0,23).slice();
        // The liquid's volume is the budget's measurement of phi, from the
        // last step that took it (this read requests the next). The V
        // texture is the pressure fill: an uncut, non-planar owner's is an
        // eight-probe vote, an eighth of the owner per probe, so on a 4h
        // surface its sum strays from phi's volume by percents.
        let measured:number|undefined;
        if(budget){const b=new Float32Array(words.buffer,40,9);if(b[1]!>0){if(b[4]!>0)measured=b[4]!;Object.assign(this.executionInfo,{
          narrowBandVolumeControlRate_per_s:this.narrowBandVolumeControlSeconds>0?b[8]:0,
          narrowBandTargetVolume_cells:b[0],narrowBandOutflowVolume_cells:b[2],
          narrowBandVolumeShift_cells:b[3]!/Math.min(...this.mixedFrame.ownership.capacity.lattice.cellSize_m),
          narrowBandVolumeBeforeCorrection_cells:b[4],narrowBandInitialVolume_cells:b[5],
          narrowBandVolumeBudgetDrift:((measured??words[3]!/2048)-b[0]!)/Math.max(1,b[0]!),
        });}}
        const represented=words[0]!/2048,fill=words[3]!/2048,volume=measured??fill,reference=Math.max(1,this.referenceVolumeCells);
        Object.assign(this.executionInfo,{representedVolumeCellSum:represented,volumeCellSum:volume,
          representedVolumeDrift:(represented-reference)/reference,rawVolumeDrift:(fill-reference)/reference,volumeDrift:(volume-reference)/reference,
          volumeTelemetrySource:measured===undefined?"dense-volume":"narrow-band-phi",frontTelemetrySource:"dense-volume",
          front_m:new Float32Array(words.buffer)[20]!+words[1]!*this.frame.cellSize_m[0],
          maxSpeed_m_s:new Float32Array(words.buffer)[2],uniformTwoLevelVelocity:this.twoLevelEnabled,
          uniformTwoLevelFineReach:UNIFORM_TWO_LEVEL_FINE_REACH,uniformTwoLevelShellReach:this.twoLevelShellReach,
          uniformTwoLevelFineTiles:words[4],uniformTwoLevelShellTiles:words[5],uniformTwoLevelTilesTotal:this.address.boxTiles,
          uniformTwoLevelExtensionTiles:this.twoLevelEnabled,uniformMixedRegularTiles:words[7],uniformMixedGeneralTiles:words[8]});
        return this.executionInfo;
      }finally{if(this.statsReadback.mapState==="mapped")this.statsReadback.unmap();this.readbackPending=false;}
    }
    return this.executionInfo;
  }


  /** A voxel stroke reaches this solver as solid-mask bits, so only a different lattice is refused. */
  validateLiveSolidEdit(scene: SceneDescription): void {
    if (proceduralSolidKey(scene.solidBoundary) !== proceduralSolidKey(this.scene.solidBoundary)) throw new Error("Procedural surface changed; reset the scene to rebuild its boundary");
    const [nx, ny] = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
    if (nx !== this.containerCells[0] || ny !== this.containerCells[1]) throw new Error("This voxel edit changes the fluid lattice; reset the scene to apply it.");
  }

  /** Fence paused remapping so the renderer invalidates its retained surface
   * and publishes resolution status without advancing simulation time. */
  async refreshSceneTopology():Promise<void>{
    this.mixedFrame?.invalidateExtension();
    if(!this.mixedFrame)return;
    await this.awaitFrameCompletion();await this.device.queue.onSubmittedWorkDone();
  }

  applySceneUniforms(scene: SceneDescription): void {
    // A stroke that moves no solid cell (a paint, a fill over what is already
    // solid) hands over a document that differs from the running one in its
    // patch list alone, and nothing here is derived from that list but the
    // occupancy mask. The mask answers for itself without being written, and
    // the document is then taken up with the frame, the layout, the boundary
    // and every cache left exactly as they are. A scene already waiting on the
    // frame in flight is applied first, so nothing overtakes it here.
    if (scene !== this.scene && !this.deferredFrameScene && uniformSceneDiffersInSolidsAlone(scene, this.scene)
      && !scene.solidBoundary && !this.scene.solidBoundary
      && this.solidMask.adoptIfUnchanged(solidWorldForFluidLattice(scene))) {
      // The layout stays synced to the document it was synced to.
      if (this.mixedSynced?.scene === this.scene) this.mixedSynced.scene = scene;
      this.scene = scene;
      return;
    }
    this.mixedFrame?.invalidateExtension();
    this.appliedRuntimeValues = undefined;
    if (this.mixedFrameInFlight) { this.deferredFrameScene = scene; return; }
    const boundaryImage = proceduralSolidField(scene)?.pack();
    if ((boundaryImage?.length ?? 0) > this.solidBoundaryCapacity) throw new Error("Procedural boundary capacity changed; reset the scene to apply it");
    const boundaryChanged = scene.solidBoundary !== undefined || this.scene.solidBoundary !== undefined;
    if (boundaryChanged) {
      if (boundaryImage) this.uploadSolidBoundary(boundaryImage);
      this.device.queue.writeBuffer(this.activeScratch, this.activeScratch.size - 8,
        new Uint32Array(boundaryImage ? [this.solidMask.words.length, PROCEDURAL_SOLID_MAGIC] : [0,0]));
      this.mixedFrame?.solid?.invalidate();
      this.mixedSolidMaskStamp++;
    }
    this.scene = scene; this.sceneRevision++;
    // The mask tells the edit every cell it moves as it folds the scene in;
    // the columns of those cells are the bed-slope lane's to bake again.
    const edit = this.mixedFrame ? uniformMixedSolidEdit(refinementRegionLattice(scene).dimensions as [number, number, number],
      this.mixedSolidTouched ??= new Uint8Array(this.address.boxTiles)) : undefined;
    let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
    const dirty = this.solidMask.update(solidWorldForFluidLattice(scene), (x, y, z, solid) => {
      edit?.cell(x, y, z, solid);
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (z < z0) z0 = z; if (z > z1) z1 = z;
    });
    if (dirty) { this.solidVoxelsEmpty = this.solidMask.empty; this.mixedSolidMaskStamp++; }
    // Dynamic coarsening joins the edit's tiles on the GPU (the next head
    // adopts them); a requests-only layout takes the scene's request.
    if (dirty && edit) this.editMixedSolids(edit);
    this.syncMixedLayout();
    // The directory words and the page slots the edit rewrote; the buffer is never resized.
    if (dirty) for (const range of dirty) this.device.queue.writeBuffer(this.activeScratch, range.firstWord * 4, this.solidMask.words, range.firstWord, range.wordCount);
    // The bed-slope lane again around the edit, from the mask just uploaded:
    // one region bake to a submit.
    if (dirty && this.bedSlopeBake && x1 >= x0) {
      const encoder = this.device.createCommandEncoder({ label: "Uniform bed-slope re-bake" });
      this.bedSlopeBake.encodeRegion(encoder, { minimum: [x0, -1, z0], maximumExclusive: [x1 + 1, this.executionInfo.ny + 1, z1 + 1] });
      this.device.queue.submit([encoder.finish()]); this.bedSlopeBake.submitted();
    }
    // The plug is a walk over a whole face of the lattice and a function of
    // the nozzle and the container alone. A voxel edit carries both objects
    // forward, and then the plug it would rebuild is the one already held.
    const inflow = scene.fluid.inflow, plug = this.inflowBoundaryOf;
    if (!plug || plug.inflow !== inflow || plug.container !== scene.container) {
      this.inflowBoundary = inflow
        ? createInflowGridBoundary(inflow, scene.container, this.containerCells)
        : undefined;
      this.inflowBoundaryOf = { inflow, container: scene.container };
      // A live page pool pins the sources' pages: they follow a nozzle that moved.
      if (this.pageResidency.live) this.mixedFrame?.pagePool.pin(uniformPagePoolSourcePages([this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz],
        uniformPageSources(scene, this.inflowBoundary, Math.max(this.fixedStep_s ?? 0, scene.numerics.maxDt_s)), this.address.boxPage));
    }
    this.pressureMultigrid.setDepthSymmetry(scene.container.depthBoundary === "symmetry");
  }
  /** The nozzle and container objects `inflowBoundary` was last built from here. */
  private inflowBoundaryOf?: { inflow: SceneDescription["fluid"]["inflow"]; container: SceneDescription["container"] };

  syncRigidBodies(bodies: readonly RigidBodyState[]): void {
    this.mixedFrame?.invalidateExtension();
    if (this.mixedFrameInFlight) { this.deferredFrameBodies = structuredClone([...bodies]); return; }
    this.rigidSystem.syncBodies(bodies);
  }
  get rigidRenderBuffer(): GPUBuffer { return this.rigidSystem.renderBuffer; }
  get rigidMotionBuffer(): GPUBuffer { return this.rigidSystem.motionBuffer; }
  setSelectedRigidBody(index: number): void { this.rigidSystem.setSelectedIndex(index); }
  pickRigidBody(origin: RigidBodyState["position_m"], direction: RigidBodyState["position_m"]) { return this.rigidSystem.pick(origin, direction); }
  readRigidBodyPoses() { return this.rigidSystem.readPoses(); }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.mixedFrame?.destroy();this.mixedPressureGeometry?.target.destroy();this.mixedPressureGeometry?.centerPhi.destroy();
    this.detailStorage?.destroy();
    this.mixedDynamic?.destroy();this.mixedBuilder?.destroy();for(let i=0;i<this.mixedRelayoutReads.size;i++)this.mixedRelayoutReads.at(i).destroy();this.mixedRelayoutRead?.destroy();this.mixedBodies?.destroy();
    for (const texture of new Set([
      this.velocityA, this.velocityB, this.velocityC, this.velocityD,
      this.pressureA, this.pressureB, this.volumeA, this.volumeB,
      this.surfaceA, this.surfaceB, this.gammaA, this.gammaB,
      ...Object.values(this.symmetryStageAuditFields ?? {}),
      this.heightA, this.heightB, this.terrainTexture,
      this.transportA, this.transportB,
    ])) texture.destroy();
    this.vertexPhiField?.destroy(); this.vertexPhiScratch?.destroy();
    this.scratchArena.destroy();
    this.boundaryVelocityA.destroy(); this.boundaryVelocityB.destroy();
    this.boundaryVelocityC.destroy(); this.boundaryVelocityD.destroy();
    this.symmetryStageAuditNegativeBoundaryVelocity?.destroy();
    this.symmetryStageAuditBetaBuffer?.destroy();
    this.macCormackAuditBinding.destroy();
    this.velocityExtrapolator.destroy();
    this.pressureMultigrid.destroy();
    this.params.destroy();
    this.reductions.destroy();
    this.conditioningScratch.destroy();
    this.pages.destroy();
    this.activeScratch.destroy(); this.bedSlopeBake?.destroy(); this.openWorldTerrain?.destroy();
    this.fieldPages.destroy();
    this.rigidSystem.destroy(); this.rigidExchange.destroy();
    this.statsReadback?.destroy();
    // An outstanding map rejects on destroy; `readPressureCycleDemand` catches
    // it and the `disposed` guard keeps it from touching a dead solver.
  }
}
