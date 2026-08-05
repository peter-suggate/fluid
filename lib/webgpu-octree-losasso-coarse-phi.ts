import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import type { PassBroker } from "./webgpu-pass-broker";
import type { FineLevelSetVolumeCoarseSource } from "./webgpu-octree-fine-levelset-volume";
import type { FineLevelSetSummaryCoarseSource } from "./webgpu-octree-fine-levelset-summary-direct";
import type { OctreeFineSeedAdapterCoarsePhiSource } from "./webgpu-octree-fine-seed-adapter";
import type { LosassoFreeSurfacePressureMode } from "./octree-coarse-backend";
import {
  OCTREE_LOSASSO_COARSE_PHI_MAGIC,
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
}

export interface OctreeLosassoCoarsePhiPlan extends Omit<WebGPUOctreeLosassoCoarsePhiSource,
  "arena" | "rowPhi" | "ghostDistances" | "volumeDirectory" | "volumePublication" | "physicalVolumes" | "summaryDelta"> {
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
    allocatedBytes: 136 + 2 * arenaBytes + 16 * rowCapacity
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
  private readonly bindGroupCache = new Map<GPUComputePipeline,
    { buffers: readonly GPUBuffer[]; group: GPUBindGroup }[]>();
  private pipelines?: Readonly<Record<"prepare" | "prepareRefresh" | "warm" | "restrict"
    | "refresh" | "finalize" | "volume" | "volumeRefresh" | "ghost" | "publish",
    GPUComputePipeline>>;
  private destroyed = false;

  constructor(private readonly device: GPUDevice,
    rowCapacity: number, faceCapacity: number,
    private readonly freeSurfacePressureMode: LosassoFreeSurfacePressureMode = "subcell-contact") {
    this.plan = planOctreeLosassoCoarsePhi(rowCapacity, faceCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.params = device.createBuffer({ label: "Losasso coarse-phi exchange constants", size: 112,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.readyDispatch = device.createBuffer({ label: "Losasso coarse-phi ready reuse dispatch", size: 24,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC });
    this.arenas = [0, 1].map((slot) => device.createBuffer({
      label: `Losasso compact coarse-phi directory generation ${slot}`, size: this.plan.arenaBytes, usage: storage,
    })) as unknown as readonly [GPUBuffer, GPUBuffer];
    this.source = {
      arena: this.arenas[0],
      rowPhi: device.createBuffer({ label: "Losasso compact row phi", size: rowCapacity * 16, usage: storage }),
      ghostDistances: device.createBuffer({ label: "Losasso free-surface ghost distances", size: faceCapacity * 16, usage: storage }),
      volumeDirectory: device.createBuffer({ label: "Losasso generic coarse-volume directory",
        size: 32 + rowCapacity * 32, usage: storage }),
      volumePublication: device.createBuffer({ label: "Losasso generic coarse-volume publication",
        size: 64, usage: storage }),
      physicalVolumes: device.createBuffer({ label: "Losasso coarse physical volumes",
        size: rowCapacity * 4, usage: storage }),
      summaryDelta: device.createBuffer({ label: "Losasso coarse summary full-row delta",
        size: (16 + 8 * rowCapacity) * 4, usage: storage }),
      rowCapacity, faceCapacity, directoryCapacity: this.plan.directoryCapacity,
    };
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
    this.pipelines = Object.freeze({ prepare, prepareRefresh, warm, restrict, refresh,
      finalize, volume, volumeRefresh, ghost, publish });
  }

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
      [16, this.readyDispatch],
    ]);
    const run = this.runner(broker, buffers);
    run("prepare", [0, 4, 5, 6, 8, 14, 16], 1);
    run("warm", [0, 4, 5, 9, 14], [this.readyDispatch, 0]);
    run("restrict", [0, 1, 2, 3, 4, 5, 8, 9, 14], Math.ceil(this.plan.rowCapacity / 64));
    run("finalize", [0, 5, 8, 14], 1);
    run("volume", [0, 5, 8, 11, 12, 13, 14, 15], Math.ceil(2 * this.plan.rowCapacity / 64));
    run("ghost", [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 14], Math.ceil(this.plan.faceCapacity / 64));
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
      [14, this.arenas[1]], [15, this.source.summaryDelta],
    ]);
    const run = this.runner(broker, buffers);
    run("prepareRefresh", [0, 4, 5, 6, 8], 1);
    run("refresh", [0, 1, 2, 3, 4, 5, 8, 9, 14], Math.ceil(this.plan.rowCapacity / 64));
    run("finalize", [0, 5, 8, 14], 1);
    run("volumeRefresh", [0, 8, 11, 12, 13, 15], Math.ceil(this.plan.rowCapacity / 64));
    run("ghost", [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 14], Math.ceil(this.plan.faceCapacity / 64));
    return this.source;
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
    return Object.freeze({ values: this.source.rowPhi, control: this.source.volumePublication });
  }
  destroy(): void { if (this.destroyed) return; this.destroyed = true;
    this.params.destroy(); this.readyDispatch.destroy(); for (const arena of this.arenas) arena.destroy(); this.source.rowPhi.destroy(); this.source.ghostDistances.destroy();
    this.source.volumeDirectory.destroy(); this.source.volumePublication.destroy(); this.source.physicalVolumes.destroy();
    this.source.summaryDelta.destroy();
    this.bindGroupCache.clear();
    this.pipelines = undefined; }
  private assertLive(): void { if (this.destroyed) throw new Error("Losasso coarse-phi exchange is destroyed"); }
}
