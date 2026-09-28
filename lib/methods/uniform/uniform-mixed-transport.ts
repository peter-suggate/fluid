import { UniformMixedOwnership, compileMixedTiers } from "./uniform-mixed-ownership";
import type { UniformMixedLayout } from "./uniform-mixed-layout";
import type { UniformScratchArena } from "./uniform-scratch-arena";
import { UNIFORM_MIXED_TRANSPORT_LIVE_HEADER, uniformMixedTransportWGSL } from "./uniform-mixed-transport.wgsl";
import type { UniformMixedSolid } from "./uniform-mixed-solid.wgsl";

/** Workgroups per live-list transport launch at most. */
const TRANSPORT_GRID = 2048;
const entries = ["clear", "build", "decodeSampled", "decode", "rowsFallback", "rowsDivide", "gather", "restrictVolume", "copyVolume"] as const;
type Entry = typeof entries[number];
/** Owner lists of the transport live set: rows are built for receivers, limbs
 * cleared and decoded for their donors; copy and restriction stay dense. */
const liveList: Partial<Record<Entry, 1 | 2>> = { build: 1, rowsFallback: 1, rowsDivide: 1, gather: 1, clear: 2, decodeSampled: 2, decode: 2 };

/** Conservative volume stage, not a separate simulation. Persistent volume and
 * trace fields and the large transient arena belong to the native Uniform host.
 * Only ownership/worklists, counts and two-bit sampling flags are allocated here.
 */
export class UniformMixedTransportStage {
  readonly ownership: UniformMixedOwnership;
  get topologyLayout(): GPUBindGroupLayout { return this.ownership.bindLayout; }
  get topologyGroup(): GPUBindGroup { return this.ownership.bindGroup; }
  private readonly workBytes: number;
  get allocatedBytes():number{return this.workBytes+this.ownership.allocatedBytes;}
  get dispatchX(): number { return this.ownership.dispatchX; }
  private readonly pipelines = new Map<Entry, GPUComputePipeline[]>();
  private readonly resourcesLayout: GPUBindGroupLayout;
  private readonly restrictGroup: GPUBindGroup;
  private readonly transportGroup: GPUBindGroup;
  private readonly sampling: GPUBuffer;
  private readonly live: GPUBuffer;
  private readonly livePipelines: GPUComputePipeline[] = [];
  private readonly coarseRows=new Map<Entry,GPUComputePipeline>();
  /** Workgroups of the h and 4h live-list launches, and of the coarse rows. */
  private readonly liveGrid: readonly [number, number];
  private readonly coarseRowGrid: number;
  private readonly cells: number;

  get layout(): UniformMixedLayout { return this.ownership.layout; }

  constructor(private readonly device: GPUDevice, layout: UniformMixedLayout,
    arena: UniformScratchArena, fineVolume: GPUTexture,
    restrictedVolume: GPUTexture, departures: GPUTexture,private readonly sourceParams?:GPUBuffer,private readonly solid?:UniformMixedSolid,private readonly parallelCoarseRows=false) {
    this.cells = layout.tiles.length * 64;
    // Fine indexing keeps the borrowed slice ABI stable across ownership changes.
    if (arena.edgeBytes < this.cells * 40 || arena.donorOffset + this.cells * 28 > arena.byteLength)
      throw new Error("Native Uniform scratch cannot hold mixed transport slices");
    this.ownership = new UniformMixedOwnership(device, layout);
    this.sampling = device.createBuffer({ label: "Uniform mixed departure sampling widths", size: layout.tiles.byteLength,
      usage: GPUBufferUsage.STORAGE });
    const tiles = layout.tiles.length;
    this.live = device.createBuffer({ label: "Uniform mixed transport live set", size: (UNIFORM_MIXED_TRANSPORT_LIVE_HEADER + 7 * tiles) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.workBytes = this.sampling.size + this.live.size;
    // Live-list launches are fixed grid-stride grids: the layout's bound
    // (every tile listed), capped where the GPU is saturated.
    this.liveGrid = [Math.min(TRANSPORT_GRID, tiles), Math.max(1, Math.min(TRANSPORT_GRID, Math.ceil(tiles / 64)))];
    this.coarseRowGrid = Math.min(TRANSPORT_GRID, tiles);
    this.resourcesLayout = device.createBindGroupLayout({ entries: [
      ...[0, 1, 2, 3].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })),
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      ...(sourceParams?[{binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform" as const}}]:[]),
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    const group = (input: GPUTexture, output: GPUTexture) => device.createBindGroup({ layout: this.resourcesLayout, entries: [
      { binding: 0, resource: { buffer: this.sampling } },
      { binding: 1, resource: { buffer: arena.buffer, offset: 0, size: this.cells * 40 } },
      { binding: 2, resource: { buffer: arena.buffer, offset: arena.donorOffset, size: this.cells * 24 } },
      { binding: 3, resource: { buffer: arena.buffer, offset: arena.donorOffset + this.cells * 24, size: this.cells * 4 } },
      { binding: 4, resource: input.createView() }, { binding: 5, resource: output.createView() },
      { binding: 6, resource: departures.createView() },
      ...(sourceParams?[{binding:7,resource:{buffer:sourceParams,size:176}}]:[]),
      { binding: 8, resource: { buffer: this.live } },
    ] });
    this.restrictGroup = group(fineVolume, restrictedVolume);
    this.transportGroup = group(restrictedVolume, fineVolume);
  }

  async initialize(): Promise<void> {
    const module = this.device.createShaderModule({ label: "Uniform mixed conservative transport", code: uniformMixedTransportWGSL(this.layout,!!this.sourceParams,!!this.solid) });
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.topologyLayout, this.resourcesLayout, ...(this.solid?[this.solid.bindLayout]:[])] });
    for (const entryPoint of entries) this.pipelines.set(entryPoint, await compileMixedTiers(umCellWidth =>
      this.device.createComputePipelineAsync({ layout, compute: { module, entryPoint,
        constants: { umCellWidth, umDispatchX: this.dispatchX, umTransportList: liveList[entryPoint] ?? 0 } } })));
    if(this.parallelCoarseRows)for(const entry of ["rowsFallback","rowsDivide"] as const){
      this.coarseRows.set(entry,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:entry+"Coarse",constants:{umCellWidth:4,umDispatchX:this.dispatchX,umTransportList:1}}}));
    }
    const tile = (entryPoint: string, constants: Record<string, number> = {}) => this.device.createComputePipelineAsync({ layout,
      compute: { module, entryPoint, constants: { umCellWidth: 1, umDispatchX: this.dispatchX, ...constants } } });
    // The dependency chain of the live set, in order: see uniformMixedTransportLiveWGSL.
    const S = 1, R1 = 2, D2 = 4, Q2 = 8, D1 = 16, Q1 = 32, Q0 = 64, DONOR = 128;
    const chain: [string, number, number][] = [["liveGather", S, R1], ["liveScatter", R1, D2], ["liveGather", D2, Q2], ["liveScatter", Q2, D1],
      ["liveGather", D1, Q1], ["liveGather", Q1, Q0], ["liveScatter", S | Q1 | Q0, DONOR]];
    this.livePipelines.push(...await Promise.all([tile("liveSeed"), ...chain.map(([entry, tpFrom, tpInto]) => tile(entry, { tpFrom, tpInto })),
      tile("liveCompact")]));
  }

  /** Reused by the native trace pipeline so every stage visits the same owners. */
  dispatch(pass: GPUComputePassEncoder, pipelines: readonly GPUComputePipeline[]): void {
    this.ownership.dispatch(pass, pipelines);
  }

  private begin(encoder: GPUCommandEncoder, label: string, group: GPUBindGroup): GPUComputePassEncoder {
    const pass = encoder.beginComputePass({ label: `Uniform mixed ${label}` });
    pass.setBindGroup(0, this.topologyGroup); pass.setBindGroup(1, group);
    if (this.solid) pass.setBindGroup(2, this.solid.bindGroup);
    return pass;
  }

  /** shared: a pass already bound to the transport group (encodeTransport). */
  private run(encoder: GPUCommandEncoder, entry: Entry, shared?: GPUComputePassEncoder): void {
    const pipelines = this.pipelines.get(entry);
    if (!pipelines) throw new Error("Mixed transport has not been initialized");
    const restrict = entry === "restrictVolume" || entry === "copyVolume";
    if (shared && restrict) throw new Error("Mixed transport restriction binds its own group");
    const pass = shared ?? this.begin(encoder, entry, restrict ? this.restrictGroup : this.transportGroup);
    const list = liveList[entry];
    // One direct grid-stride launch per tier (0 = h, 1 = 4h) of the live list.
    if (list) for (const tier of [0, 1] as const) {
      const coarse=tier===1?this.coarseRows.get(entry):undefined;
      pass.setPipeline(coarse??pipelines[tier]!);pass.dispatchWorkgroups(coarse?this.coarseRowGrid:this.liveGrid[tier]);
    }
    else this.dispatch(pass, pipelines);
    if (!shared) pass.end();
  }

  /** Rows and donors whose restricted transport equals the dense one, from
   * this frame's departures and volume (per-tile reach dependency chain). */
  private encodeLiveSet(encoder: GPUCommandEncoder): void {
    const seed = this.livePipelines[0];
    if (!seed) throw new Error("Mixed transport has not been initialized");
    const tiles = this.layout.tiles.length, groups = Math.ceil(tiles / 64);
    encoder.clearBuffer(this.live, 0, UNIFORM_MIXED_TRANSPORT_LIVE_HEADER * 4);
    const pass = this.begin(encoder, "transport live set", this.transportGroup);
    pass.setPipeline(seed!); pass.dispatchWorkgroups(Math.min(tiles, this.dispatchX), Math.ceil(tiles / this.dispatchX));
    for (const pipeline of this.livePipelines.slice(1)) { pass.setPipeline(pipeline); pass.dispatchWorkgroups(Math.min(groups, this.dispatchX), Math.ceil(groups / this.dispatchX)); }
    pass.end();
  }

  encodeCopy(encoder: GPUCommandEncoder): void { this.run(encoder, "copyVolume"); }

  encodeRestriction(encoder: GPUCommandEncoder): void { this.run(encoder, "restrictVolume"); }

  /** Call only after tracing: extension shares the edge/donor backing. */
  encodeTransport(encoder: GPUCommandEncoder): void {
    this.encodeLiveSet(encoder);
    // One pass: a dispatch is its own usage scope, and every entry here binds the transport group.
    const pass = this.begin(encoder, "transport", this.transportGroup);
    this.run(encoder, "clear", pass); this.run(encoder, "build", pass); this.run(encoder, "decodeSampled", pass);
    for (let round = 0; round < 3; round++) {
      this.run(encoder, round === 0 ? "rowsFallback" : "rowsDivide", pass); this.run(encoder, "decode", pass);
    }
    this.run(encoder, "gather", pass);
    pass.end();
  }

  destroy(): void { this.ownership.destroy(); this.sampling.destroy(); this.live.destroy(); }
}
