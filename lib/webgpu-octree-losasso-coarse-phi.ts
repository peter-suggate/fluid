import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import type { PassBroker } from "./webgpu-pass-broker";
import type { FineLevelSetVolumeCoarseSource } from "./webgpu-octree-fine-levelset-volume";
import type { FineLevelSetSummaryCoarseSource } from "./webgpu-octree-fine-levelset-summary-direct";
import type { OctreeFineSeedAdapterCoarsePhiSource } from "./webgpu-octree-fine-seed-adapter";
import type { LosassoFreeSurfacePressureMode } from "./octree-coarse-backend";
import type { SurfaceInflowState } from "./webgpu-quadtree-builder";
import type { WebGPUOctreeLosassoVelocitySamplerSource } from
  "./webgpu-octree-losasso-velocity-sampler";
import {
  OCTREE_LOSASSO_COARSE_PHI_MAGIC,
  octreeLosassoCoarseOnlyPhiWGSL,
  octreeLosassoCoarsePhiWGSL,
  octreeLosassoWarmPhiWGSL,
  octreeLosassoVolumeBridgeWGSL,
} from "./webgpu-octree-losasso-coarse-phi.wgsl";

export const OCTREE_LOSASSO_COARSE_PHI_ENTRY_WORDS = 8;
export const OCTREE_LOSASSO_COARSE_PHI_DIRECTORY_WORDS = 4;
export const OCTREE_LOSASSO_GHOST_DISTANCE_WORDS = 4;

export interface WebGPUOctreeLosassoCoarsePhiInput {
  readonly leafHeaders: GPUBuffer;
  /** Accepted Losasso authority `[epoch,rowCount,faceCount,valid,...]`. */
  readonly coarseControl: GPUBuffer;
  readonly faces: GPUBuffer;
  readonly faceGeometry: GPUBuffer;
  /** Finest-cell solid fraction/owner pairs for solid-aware ghosting. */
  readonly solidCells: GPUBuffer;
  readonly dimensions: readonly [number, number, number];
  readonly maximumLeafSize: number;
  readonly cellSize: number;
}

export interface WebGPUOctreeLosassoCoarsePhiSource {
  /** One-binding compact directory consumed by fine topology. */
  readonly arena: GPUBuffer;
  readonly rowPhi: GPUBuffer;
  /** vec4u(bitcast(distance), bitcast(theta), bitcast(airPhi), flags). */
  readonly ghostDistances: GPUBuffer;
  readonly rowCapacity: number;
  readonly faceCapacity: number;
  readonly directoryCapacity: number;
  /** Generic compact-directory bridge consumed by shared factor-4 volume correction. */
  readonly volumeDirectory: GPUBuffer;
  readonly volumePublication: GPUBuffer;
  readonly physicalVolumes: GPUBuffer;
  /** Full retired+current coarse-row delta for the direct fine-summary hierarchy. */
  readonly summaryDelta: GPUBuffer;
  /** Affine gradient carried by the factor-one compact surface tracker. */
  readonly rowGradient: GPUBuffer;
}

export interface OctreeLosassoCoarsePhiPlan extends Omit<WebGPUOctreeLosassoCoarsePhiSource,
  "arena" | "rowPhi" | "ghostDistances" | "volumeDirectory" | "volumePublication" | "physicalVolumes" | "summaryDelta" | "rowGradient"> {
  readonly arenaBytes: number;
  readonly allocatedBytes: number;
  readonly encodedDispatchCount: 6;
}

export function planOctreeLosassoCoarsePhi(
  rowCapacity: number,
  faceCapacity: number,
): OctreeLosassoCoarsePhiPlan {
  for (const [label, value] of [["row", rowCapacity], ["face", faceCapacity]] as const) {
    if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`Losasso coarse-phi ${label} capacity must be positive`);
  }
  let directoryCapacity = 1;
  while (directoryCapacity < 2 * rowCapacity) directoryCapacity *= 2;
  const arenaBytes = (20 + OCTREE_LOSASSO_COARSE_PHI_ENTRY_WORDS * rowCapacity
    + OCTREE_LOSASSO_COARSE_PHI_DIRECTORY_WORDS * directoryCapacity) * 4;
  return Object.freeze({ rowCapacity, faceCapacity, directoryCapacity, arenaBytes,
    encodedDispatchCount: 6 as const,
    allocatedBytes: 232 + 2 * arenaBytes + 48 * rowCapacity
      + OCTREE_LOSASSO_GHOST_DISTANCE_WORDS * 4 * faceCapacity
      + 32 + rowCapacity * 32 + 64 + rowCapacity * 4 + (16 + 8 * rowCapacity) * 4 });
}

/** WGSL interface injected into `WebGPUFineLevelSetTopology`; one binding. */
export function makeOctreeLosassoCoarsePhiSampleWGSL(binding = 9): string {
  if (!Number.isSafeInteger(binding) || binding < 0) throw new RangeError("Losasso coarse-phi binding must be non-negative");
  return /* wgsl */ `
@group(0)@binding(${binding})var<storage,read>losassoCoarsePhi:array<u32>;
fn losassoCoarseHash(cell:u32,size:u32)->u32{return ((cell*0x9e3779b1u)^size)*0x85ebca6bu;}
fn losassoCoarseLookup(cell:u32,size:u32)->u32{let capacity=losassoCoarsePhi[11];if(capacity==0u||(capacity&(capacity-1u))!=0u){return 0xffffffffu;}
 let hash=losassoCoarseHash(cell,size);let base=losassoCoarsePhi[10];let mask=capacity-1u;
 for(var probe=0u;probe<32u;probe+=1u){let at=base+4u*((hash+probe)&mask);let key=losassoCoarsePhi[at];if(key==0u){return 0xffffffffu;}
  if(key==cell+1u&&losassoCoarsePhi[at+1u]==size&&losassoCoarsePhi[at+3u]==hash){return losassoCoarsePhi[at+2u]-1u;}}
 return 0xffffffffu;}
fn sampleCoarseOctreePhi(position:vec3f)->f32{let invalidPhi=3.402823e38;
 if(arrayLength(&losassoCoarsePhi)<20u||losassoCoarsePhi[0]!=${OCTREE_LOSASSO_COARSE_PHI_MAGIC}u){return invalidPhi;}
 let dimensions=vec3u(losassoCoarsePhi[5],losassoCoarsePhi[6],losassoCoarsePhi[7]);let width=bitcast<f32>(losassoCoarsePhi[8]);
 let origin=vec3f(bitcast<f32>(losassoCoarsePhi[15]),bitcast<f32>(losassoCoarsePhi[16]),bitcast<f32>(losassoCoarsePhi[17]));
 if(!(width>0.)){return invalidPhi;}let grid=(position-origin)/width;if(any(grid<vec3f(0))||any(grid>=vec3f(dimensions))){return invalidPhi;}
 let q=vec3u(floor(grid));var size=1u;loop{let leafOrigin=(q/vec3u(size))*vec3u(size);
  let cell=leafOrigin.x+dimensions.x*(leafOrigin.y+dimensions.y*leafOrigin.z);let row=losassoCoarseLookup(cell,size);
  if(row!=0xffffffffu&&row<losassoCoarsePhi[2]){let entry=losassoCoarsePhi[9]+8u*row;let flags=losassoCoarsePhi[entry+5u];
   let value=bitcast<f32>(losassoCoarsePhi[entry+2u]);if((flags&3u)==3u&&value==value){return value;}return invalidPhi;}
  if(size>=losassoCoarsePhi[4]){break;}size*=2u;}
 return .5*width;}
`;
}

/** Compact fine↔coarse exchange and free-surface ghost-distance schedule. */
export class WebGPUOctreeLosassoCoarsePhiExchange {
  readonly plan: OctreeLosassoCoarsePhiPlan;
  readonly source: WebGPUOctreeLosassoCoarsePhiSource;
  private readonly arenas: readonly [GPUBuffer, GPUBuffer];
  private readonly params: GPUBuffer;
  private readonly readyDispatch: GPUBuffer;
  private readonly coarseOnlyParams: GPUBuffer;
  private readonly coarseOnlyNextGradient: GPUBuffer;
  private readonly bindGroupCache = new Map<GPUComputePipeline,
    { buffers: readonly GPUBuffer[]; group: GPUBindGroup }[]>();
  private pipelines?: Readonly<Record<"prepare" | "prepareRefresh" | "warm" | "restrict"
    | "refresh" | "finalize" | "volume" | "volumeRefresh" | "ghost" | "publish",
    GPUComputePipeline>>;
  private coarseOnlyPipelines?: Readonly<Record<"advect" | "publish" | "ghost",
    GPUComputePipeline>>;
  private coarseOnlyGeneration = 0;
  private destroyed = false;

  constructor(private readonly device: GPUDevice,
    rowCapacity: number, faceCapacity: number,
    private readonly freeSurfacePressureMode: LosassoFreeSurfacePressureMode = "subcell-contact",
    denseComplementCells = 0) {
    this.plan = planOctreeLosassoCoarsePhi(rowCapacity, faceCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.params = device.createBuffer({ label: "Losasso coarse-phi exchange constants", size: 112,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.readyDispatch = device.createBuffer({ label: "Losasso coarse-phi ready reuse dispatch", size: 24,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC });
    this.coarseOnlyParams = device.createBuffer({ label: "Losasso coarse-only phi constants", size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.arenas = [0, 1].map((slot) => device.createBuffer({
      label: `Losasso compact coarse-phi directory generation ${slot}`, size: this.plan.arenaBytes, usage: storage,
    })) as unknown as readonly [GPUBuffer, GPUBuffer];
    this.source = {
      arena: this.arenas[0],
      rowPhi: device.createBuffer({ label: "Losasso compact row phi", size: rowCapacity * 16, usage: storage }),
      ghostDistances: device.createBuffer({ label: "Losasso free-surface ghost distances", size: faceCapacity * 16, usage: storage }),
      volumeDirectory: device.createBuffer({ label: "Losasso generic coarse-volume directory",
        size: 32 + (rowCapacity + denseComplementCells) * 32, usage: storage }),
      volumePublication: device.createBuffer({ label: "Losasso generic coarse-volume publication",
        size: 64, usage: storage }),
      physicalVolumes: device.createBuffer({ label: "Losasso coarse physical volumes",
        size: rowCapacity * 4, usage: storage }),
      summaryDelta: device.createBuffer({ label: "Losasso coarse summary full-row delta",
        size: (16 + 8 * rowCapacity) * 4, usage: storage }),
      rowGradient: device.createBuffer({ label: "Losasso compact coarse-phi gradients",
        size: rowCapacity * 16, usage: storage }),
      rowCapacity, faceCapacity, directoryCapacity: this.plan.directoryCapacity,
    };
    this.coarseOnlyNextGradient = device.createBuffer({
      label: "Losasso next compact coarse-phi gradients", size: rowCapacity * 16, usage: storage,
    });
  }

  get initializationTasks(): readonly { readonly label: string; readonly run: () => Promise<void> }[] {
    return [{ label: "Compile Losasso coarse-phi exchange", run: () => this.initialize() }];
  }
  initializePipelines(): Promise<void> { return this.initialize(); }
  async initialize(): Promise<void> {
    this.assertLive(); if (this.pipelines) return;
    const shaderModule = this.device.createShaderModule({ label: "Losasso compact coarse-phi exchange shader",
      code: octreeLosassoCoarsePhiWGSL });
    const volumeModule = this.device.createShaderModule({ label: "Losasso compact volume bridge shader",
      code: octreeLosassoVolumeBridgeWGSL });
    const warmModule = this.device.createShaderModule({ label: "Losasso coarse-phi warm remap shader",
      code: octreeLosassoWarmPhiWGSL });
    const coarseOnlyModule = this.device.createShaderModule({
      label: "Losasso compact coarse-only phi transport shader",
      code: octreeLosassoCoarseOnlyPhiWGSL,
    });
    const make = (entryPoint: string) => this.device.createComputePipelineAsync({
      label: entryPoint, layout: "auto", compute: { module: shaderModule, entryPoint },
    });
    // Dawn's Metal compiler serializes this module internally; compiling its
    // four auto-layout entry points concurrently can race the native frontend.
    const prepare = await make("prepareLosassoCoarsePhi");
    const prepareRefresh = await make("prepareLosassoCoarsePhiRefresh");
    const warm = await this.device.createComputePipelineAsync({ label: "warmLosassoCoarsePhi", layout: "auto",
      compute: { module: warmModule, entryPoint: "warmLosassoCoarsePhi" } });
    const restrict = await make("restrictLosassoCoarsePhi");
    const refresh = await make("refreshLosassoCoarsePhi");
    const finalize = await make("finalizeLosassoCoarsePhi");
    const ghost = await make("publishLosassoGhostDistances");
    const publish = await make("publishLosassoCoarsePhiArena");
    const volume = await this.device.createComputePipelineAsync({ label: "publishLosassoVolumeBridge", layout: "auto",
      compute: { module: volumeModule, entryPoint: "publishLosassoVolumeBridge" } });
    const volumeRefresh = await this.device.createComputePipelineAsync({ label: "refreshLosassoVolumeBridge", layout: "auto",
      compute: { module: volumeModule, entryPoint: "refreshLosassoVolumeBridge" } });
    const advect = await this.device.createComputePipelineAsync({ label: "advectLosassoCoarsePhi", layout: "auto",
      compute: { module: coarseOnlyModule, entryPoint: "advectLosassoCoarsePhi" } });
    const coarsePublish = await this.device.createComputePipelineAsync({ label: "publishLosassoCoarseOnlyRows", layout: "auto",
      compute: { module: coarseOnlyModule, entryPoint: "publishLosassoCoarseOnlyRows" } });
    const coarseGhost = await this.device.createComputePipelineAsync({ label: "publishLosassoCoarseOnlyGhosts", layout: "auto",
      compute: { module: coarseOnlyModule, entryPoint: "publishLosassoCoarseOnlyGhosts" } });
    this.pipelines = Object.freeze({ prepare, prepareRefresh, warm, restrict, refresh,
      finalize, volume, volumeRefresh, ghost, publish });
    this.coarseOnlyPipelines = Object.freeze({ advect, publish: coarsePublish, ghost: coarseGhost });
  }

  /** Advance and publish the factor-one compact surface without allocating a
   * Section-5 fine page pool. The adapter leaves supply the cold affine SDF;
   * later generations sample the retained compact directory. */
  encodeCoarseOnly(
    broker: PassBroker,
    input: WebGPUOctreeLosassoCoarsePhiInput,
    seedLeaves: GPUBuffer,
    velocity: WebGPUOctreeLosassoVelocitySamplerSource,
    dt_s: number,
    inflow?: SurfaceInflowState,
    openTopBoundary = false,
  ): WebGPUOctreeLosassoCoarsePhiSource {
    this.assertLive();
    if (!this.pipelines || !this.coarseOnlyPipelines) {
      throw new Error("Losasso coarse-only phi pipelines are not initialized");
    }
    if (!Number.isFinite(dt_s) || dt_s < 0) {
      throw new RangeError("Losasso coarse-only phi timestep must be non-negative and finite");
    }
    if (velocity.dimensions.some((value, axis) => value !== input.dimensions[axis])
      || Math.abs(velocity.fineCellSize - input.cellSize) > 1e-6 * input.cellSize) {
      throw new RangeError("Losasso coarse-only phi sampler does not match the pressure lattice");
    }
    if (dt_s > 0 || this.coarseOnlyGeneration === 0) this.coarseOnlyGeneration += 1;
    this.updateCoarseOnlySharedParams(input, this.coarseOnlyGeneration);
    const bytes = new ArrayBuffer(96), words = new Uint32Array(bytes), floats = new Float32Array(bytes);
    words.set([...velocity.dimensions, velocity.directoryCapacity], 0);
    floats.set([0, 0, 0, dt_s], 4);
    floats[8] = input.cellSize; words[9] = input.maximumLeafSize;
    words[10] = 1 | (this.freeSurfacePressureMode === "cell-centered-air" ? 2 : 0);
    words[11] = openTopBoundary ? 1 : 0;
    words.set([this.plan.rowCapacity, this.plan.faceCapacity,
      this.plan.directoryCapacity, this.coarseOnlyGeneration], 12);
    if (inflow) {
      floats.set([inflow.outletCenter_m.x, inflow.outletCenter_m.y,
        inflow.outletCenter_m.z, inflow.radius_m], 16);
      floats.set([inflow.velocity_m_s.x, inflow.velocity_m_s.y,
        inflow.velocity_m_s.z, inflow.strength], 20);
    }
    this.device.queue.writeBuffer(this.coarseOnlyParams, 0, bytes);

    const nextArena = this.arenas[1];
    const mainBuffers = new Map<number, GPUBuffer>([
      [0, this.params], [4, input.leafHeaders], [5, input.coarseControl], [6, input.faces],
      [8, nextArena], [11, this.source.volumeDirectory], [12, this.source.volumePublication],
      [13, this.source.physicalVolumes], [14, this.source.arena], [15, this.source.summaryDelta],
      [16, this.readyDispatch],
    ]);
    const runMain = this.runner(broker, mainBuffers);
    runMain("prepare", [0, 4, 5, 6, 8, 14, 16], 1);

    const customBuffers = [this.coarseOnlyParams, velocity.control, input.leafHeaders,
      seedLeaves, this.source.arena, this.source.rowPhi, this.source.rowGradient,
      // Binding 8 is consumed by publishLosassoCoarseOnlyGhosts and indexed by
      // pressure face id.  The W7 sampler geometry has a different cardinality
      // and ordering; binding it here made symmetric pressure faces inherit
      // unrelated axes/spans and therefore different ghost dual distances.
      this.coarseOnlyNextGradient, input.faceGeometry, velocity.extendedVelocity,
      velocity.axisFaceDirectory, nextArena, input.faces, this.source.ghostDistances,
      input.solidCells, input.coarseControl, this.source.volumeDirectory];
    const runCustom = (pipeline: GPUComputePipeline, bindings: readonly number[], groups: number) => {
      const pass = broker.compute({ label: pipeline.label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bindings.map((binding) => ({ binding,
          resource: { buffer: customBuffers[binding]! } })),
      }));
      pass.dispatchWorkgroups(groups);
    };
    const rowGroups = Math.ceil(this.plan.rowCapacity / 64);
    const faceGroups = Math.ceil(this.plan.faceCapacity / 64);
    runCustom(this.coarseOnlyPipelines.advect, [0, 2, 5, 7, 16], rowGroups);
    runCustom(this.coarseOnlyPipelines.publish, [0, 2, 5, 11, 15], rowGroups);
    runMain("finalize", [0, 5, 8, 14], 1);
    runMain("volume", [0, 5, 8, 11, 12, 13, 14, 15], Math.ceil(2 * this.plan.rowCapacity / 64));
    runCustom(this.coarseOnlyPipelines.ghost,
      [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16], faceGroups);
    runMain("publish", [0, 5, 8, 14], [this.readyDispatch, 12]);
    broker.copyBufferToBuffer(this.coarseOnlyNextGradient, 0,
      this.source.rowGradient, 0, this.source.rowGradient.size);
    return this.source;
  }

  get coarseOnlyPublishedGeneration(): number { return this.coarseOnlyGeneration; }

  encode(broker: PassBroker, fine: WebGPUFineLevelSetBrickSource,
    input: WebGPUOctreeLosassoCoarsePhiInput): WebGPUOctreeLosassoCoarsePhiSource {
    this.assertLive(); if (!this.pipelines) throw new Error("Losasso coarse-phi pipelines are not initialized");
    this.updateParams(fine, input, 0);
    // `source.arena` is part of several long-lived projection/topology bind
    // groups. Keep that buffer identity stable for the lifetime of the
    // exchange. Build the next generation in the private staging arena while
    // the published arena remains available to the warm remap and full-row
    // summary delta, then order its in-pass compute publication after every
    // reader of the old generation has finished.
    const previousArena = this.source.arena;
    const nextArena = this.arenas[1];
    const buffers = new Map<number, GPUBuffer>([
      [0, this.params], [1, fine.metadata], [2, fine.worklist], [3, fine.samples],
      [4, input.leafHeaders], [5, input.coarseControl], [6, input.faces],
      [7, input.faceGeometry], [8, nextArena], [9, this.source.rowPhi],
      [10, this.source.ghostDistances],
      [11, this.source.volumeDirectory], [12, this.source.volumePublication],
      [13, this.source.physicalVolumes], [14, previousArena], [15, this.source.summaryDelta],
      [16, this.readyDispatch], [17, input.solidCells],
    ]);
    const run = this.runner(broker, buffers);
    run("prepare", [0, 4, 5, 6, 8, 14, 16], 1);
    run("warm", [0, 4, 5, 9, 14], [this.readyDispatch, 0]);
    run("restrict", [0, 1, 2, 3, 4, 5, 8, 9, 14], Math.ceil(this.plan.rowCapacity / 64));
    run("finalize", [0, 5, 8, 14], 1);
    run("volume", [0, 5, 8, 11, 12, 13, 14, 15], Math.ceil(2 * this.plan.rowCapacity / 64));
    run("ghost", [0, 1, 2, 3, 5, 6, 7, 8, 10, 14, 17], Math.ceil(this.plan.faceCapacity / 64));
    run("publish", [0, 5, 8, 14], [this.readyDispatch, 12]);
    return this.source;
  }

  /** Refresh phi-dependent fields against an unchanged accepted row/face epoch.
   * The retained directory and row identities are validated on the GPU; a
   * mismatch fails the refreshed publication closed instead of rebuilding it. */
  encodeFieldRefresh(broker: PassBroker, fine: WebGPUFineLevelSetBrickSource,
    input: WebGPUOctreeLosassoCoarsePhiInput): WebGPUOctreeLosassoCoarsePhiSource {
    this.assertLive(); if (!this.pipelines) throw new Error("Losasso coarse-phi pipelines are not initialized");
    this.updateParams(fine, input, 1);
    const arena = this.source.arena;
    const buffers = new Map<number, GPUBuffer>([
      [0, this.params], [1, fine.metadata], [2, fine.worklist], [3, fine.samples],
      [4, input.leafHeaders], [5, input.coarseControl], [6, input.faces],
      [7, input.faceGeometry], [8, arena], [9, this.source.rowPhi],
      [10, this.source.ghostDistances], [11, this.source.volumeDirectory],
      [12, this.source.volumePublication], [13, this.source.physicalVolumes],
      [14, this.arenas[1]], [15, this.source.summaryDelta], [17, input.solidCells],
    ]);
    const run = this.runner(broker, buffers);
    run("prepareRefresh", [0, 4, 5, 6, 8], 1);
    run("refresh", [0, 1, 2, 3, 4, 5, 8, 9, 14], Math.ceil(this.plan.rowCapacity / 64));
    run("finalize", [0, 5, 8, 14], 1);
    run("volumeRefresh", [0, 8, 11, 12, 13, 15], Math.ceil(this.plan.rowCapacity / 64));
    run("ghost", [0, 1, 2, 3, 5, 6, 7, 8, 10, 14, 17], Math.ceil(this.plan.faceCapacity / 64));
    return this.source;
  }

  /** Rebuild only aperture/free-surface ghost state after an analytic rigid
   * boundary refresh. Row identities and coarse phi remain unchanged. */
  encodeGhostRefresh(broker: PassBroker, fine: WebGPUFineLevelSetBrickSource,
    input: WebGPUOctreeLosassoCoarsePhiInput): void {
    this.assertLive();
    if (!this.pipelines) throw new Error("Losasso coarse-phi pipelines are not initialized");
    this.updateParams(fine, input, 1);
    const buffers = new Map<number, GPUBuffer>([
      [0, this.params], [1, fine.metadata], [2, fine.worklist], [3, fine.samples],
      [4, input.leafHeaders], [5, input.coarseControl], [6, input.faces],
      [7, input.faceGeometry], [8, this.source.arena], [10, this.source.ghostDistances],
      [14, this.arenas[1]], [17, input.solidCells],
    ]);
    this.runner(broker, buffers)("ghost", [0, 1, 2, 3, 5, 6, 7, 8, 10, 14, 17],
      Math.ceil(this.plan.faceCapacity / 64));
  }

  private updateParams(fine: WebGPUFineLevelSetBrickSource,
    input: WebGPUOctreeLosassoCoarsePhiInput, scheduleMode: 0 | 1): void {
    if (input.dimensions.some((value) => !Number.isSafeInteger(value) || value < 1)
      || !Number.isSafeInteger(input.maximumLeafSize) || input.maximumLeafSize < 1
      || (input.maximumLeafSize & (input.maximumLeafSize - 1)) !== 0
      || !Number.isFinite(input.cellSize) || input.cellSize <= 0) {
      throw new RangeError("Losasso coarse-phi geometry is invalid");
    }
    const bytes = new ArrayBuffer(112), words = new Uint32Array(bytes), floats = new Float32Array(bytes);
    words.set(fine.plan.brickDimensions, 0); words[3] = fine.plan.brickResolution;
    words.set(fine.plan.sampleDimensions, 4); words[7] = fine.plan.samplesPerBrick;
    floats.set(fine.plan.domainOrigin, 8); floats[11] = fine.plan.fineCellWidth;
    words.set([fine.plan.maximumResidentBricks, fine.generation,
      fine.plan.maximumResidentBricks, 7], 12);
    words.set(input.dimensions, 16); words[19] = input.maximumLeafSize;
    floats[20] = input.cellSize; words[21] = this.plan.directoryCapacity;
    words[22] = this.plan.rowCapacity; words[23] = this.plan.faceCapacity;
    words[24] = scheduleMode; words[25] = Math.ceil(this.plan.rowCapacity / 64);
    words[26] = Math.ceil(this.plan.arenaBytes / 4 / 64);
    words[27] = this.freeSurfacePressureMode === "cell-centered-air" ? 1 : 0;
    this.device.queue.writeBuffer(this.params, 0, bytes);
  }

  private updateCoarseOnlySharedParams(
    input: WebGPUOctreeLosassoCoarsePhiInput,
    generation: number,
  ): void {
    const bytes = new ArrayBuffer(112), words = new Uint32Array(bytes), floats = new Float32Array(bytes);
    words.set([1, 1, 1, 1, ...input.dimensions, 1], 0);
    floats.set([0, 0, 0, input.cellSize], 8);
    // `schedule.z` is consumed by prepareLosassoCoarsePhi to author the
    // indirect stable-arena publication.  A single workgroup copies only the
    // 20-word header and the first handful of row-entry words, leaving the
    // directory at the end of the arena stale.  Factor-one consumers (the W7
    // band and recurring frontier) resolve rows through that directory, so a
    // truncated publication looks valid while silently holding the surface.
    words.set([1, generation, Math.ceil(this.plan.arenaBytes / 4 / 64), 0], 12);
    words.set(input.dimensions, 16); words[19] = input.maximumLeafSize;
    floats[20] = input.cellSize; words[21] = this.plan.directoryCapacity;
    words[22] = this.plan.rowCapacity; words[23] = this.plan.faceCapacity;
    words[24] = 2; words[25] = Math.ceil(this.plan.rowCapacity / 64);
    words[26] = Math.ceil(this.plan.arenaBytes / 4 / 64);
    words[27] = this.freeSurfacePressureMode === "cell-centered-air" ? 1 : 0;
    this.device.queue.writeBuffer(this.params, 0, bytes);
  }

  private runner(broker: PassBroker, buffers: ReadonlyMap<number, GPUBuffer>) {
    return (name: keyof NonNullable<typeof this.pipelines>,
      bindings: readonly number[], groups: number | readonly [GPUBuffer, number]) => {
      const pipeline = this.pipelines![name]; const pass = broker.compute({ label: `Losasso coarse phi - ${name}` });
      const bound = bindings.map((binding) => buffers.get(binding)!);
      const variants = this.bindGroupCache.get(pipeline) ?? [];
      const cached = variants.find((variant) => variant.buffers.length === bound.length
        && variant.buffers.every((buffer, index) => buffer === bound[index]));
      const group = cached?.group ?? this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0), entries: bindings.map((binding, index) =>
          ({ binding, resource: { buffer: bound[index]! } })),
      });
      if (!cached) { variants.push({ buffers: bound, group }); this.bindGroupCache.set(pipeline, variants); }
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      if (typeof groups === "number") pass.dispatchWorkgroups(groups);
      else pass.dispatchWorkgroupsIndirect(groups[0], groups[1]);
    };
  }

  fineTopologyEntry(binding = 9): GPUBindGroupEntry {
    return { binding, resource: { buffer: this.source.arena } };
  }
  volumeCoarseSource(input: WebGPUOctreeLosassoCoarsePhiInput): FineLevelSetVolumeCoarseSource {
    return Object.freeze({ headers: input.leafHeaders, records: this.source.rowPhi,
      physicalVolumes: this.source.physicalVolumes, sampleDirectory: this.source.volumeDirectory,
      publicationControl: this.source.volumePublication, rowCount: input.coarseControl,
      dimensions: input.dimensions, physicalCellSize: input.cellSize,
      maximumLeafSize: input.maximumLeafSize, sampleRowCapacity: this.plan.rowCapacity });
  }
  summaryCoarseSource(): FineLevelSetSummaryCoarseSource {
    return Object.freeze({ directory: this.source.volumeDirectory, control: this.source.volumePublication,
      delta: this.source.summaryDelta, deltaHeaderWords: 16 as const, deltaRecordWords: 4 as const });
  }
  fineSeedCoarsePhiSource(): OctreeFineSeedAdapterCoarsePhiSource {
    return Object.freeze({ values: this.source.rowPhi, gradients: this.source.rowGradient,
      control: this.source.volumePublication });
  }
  destroy(): void { if (this.destroyed) return; this.destroyed = true;
    this.params.destroy(); this.readyDispatch.destroy(); this.coarseOnlyParams.destroy();
    this.coarseOnlyNextGradient.destroy(); for (const arena of this.arenas) arena.destroy();
    this.source.rowPhi.destroy(); this.source.rowGradient.destroy(); this.source.ghostDistances.destroy();
    this.source.volumeDirectory.destroy(); this.source.volumePublication.destroy(); this.source.physicalVolumes.destroy();
    this.source.summaryDelta.destroy();
    this.bindGroupCache.clear();
    this.pipelines = undefined; this.coarseOnlyPipelines = undefined; }
  private assertLive(): void { if (this.destroyed) throw new Error("Losasso coarse-phi exchange is destroyed"); }
}
