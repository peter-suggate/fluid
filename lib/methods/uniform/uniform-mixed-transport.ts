import { uniformDetailBindLayout, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import { UniformMixedOwnership, compileMixedTiers } from "./uniform-mixed-ownership";
import { UNIFORM_MIXED_COUNTED, uniformMixedCountedEntriesWGSL } from "./uniform-mixed-topology.wgsl";
import type { UniformMixedLayout } from "./uniform-mixed-layout";
import { UNIFORM_MIXED_TRANSPORT_LIVE_HEADER, uniformMixedTransportWGSL } from "./uniform-mixed-transport.wgsl";
import { UNIFORM_PARAMS_BYTES } from "./uniform-mixed-source.wgsl";
import { uniformMixedSolidPipeline, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";

/** Workgroups per live-list transport launch at most. */
const TRANSPORT_GRID = 2048;
const entries = ["build", "decode", "rowsFallback", "rowsDivide", "gather", "copyVolume"] as const;
type Entry = typeof entries[number];
/** Live-set lists (uniformMixedTransportLiveWGSL): 1 rows Q0, 2 every donor,
 * 3 rows Q1, 4 rows Q2, 5 rows R1, 6 donors D1, 7 donors D2. */
type LiveList = 1 | 2 | 3 | 4 | 5 | 6 | 7;
/** The list each live launch strides, in encode order: each round's rows are
 * those a later round reads, and each decode covers the donors its round
 * added to. */
const liveLaunches: readonly (readonly [Entry, LiveList])[] = [["build", 1], ["rowsFallback", 3], ["decode", 2],
  ["rowsDivide", 4], ["decode", 6], ["rowsDivide", 5], ["decode", 7], ["gather", 5]];
/** The copy back: gather's rows (R1) from output to volume. */
const copyLaunch = ["copyVolume", 5] as const;
/** Tier entry points that replace the generic one: h rows sum a tile in
 * workgroup memory, 4h rows normalize four rows per workgroup. */
const tierEntries: Partial<Record<Entry, readonly [string, string]>> = { build: ["buildFine", "build"],
  rowsFallback: ["rowsFallbackFine", "rowsFallbackCoarse"], rowsDivide: ["rowsDivideFine", "rowsDivideCoarse"] };

/** Conservative volume stage, not a separate simulation. Persistent volume and
 * trace fields belong to the native Uniform host and the transient rows to
 * the caller's stage scratch (bindScratch). Only ownership/worklists, counts
 * and two-bit sampling flags are allocated here.
 */
export class UniformMixedTransportStage {
  readonly ownership: UniformMixedOwnership;
  get topologyLayout(): GPUBindGroupLayout { return this.ownership.bindLayout; }
  get topologyGroup(): GPUBindGroup { return this.ownership.bindGroup; }
  private readonly workBytes: number;
  get allocatedBytes():number{return this.workBytes+this.ownership.allocatedBytes;}
  get dispatchX(): number { return this.ownership.dispatchX; }
  /** Tier pipelines by `${entry}:${list}`. */
  private readonly pipelines = new Map<string, GPUComputePipeline[]>();
  private readonly resourcesLayout: GPUBindGroupLayout;
  private copyGroup?: UniformDetailGroup;
  private transportGroup?: UniformDetailGroup;
  private readonly sampling: GPUBuffer;
  private readonly live: GPUBuffer;
  private readonly livePipelines: GPUComputePipeline[] = [];
  /** Workgroups of the h and 4h live-list launches. */
  private readonly liveGrid: readonly [number, number];
  private readonly resolved: boolean;

  /** The scratch ranges at `fineTiles` h tiles of `tiles`: the edge rows by
   * owner rank (640 words an h tile's 64 rows, 128 a 4h row), then three
   * words an owner of donor sums and flags, then one of decoded sums. */
  static scratchRanges(tiles: number, fineTiles = tiles) {
    const owners = 63 * fineTiles + tiles, align = (bytes: number) => Math.ceil(bytes / 256) * 256;
    const edges = { offset: 0, size: 4 * (640 * fineTiles + 128 * (tiles - fineTiles)) };
    const donors = { offset: align(edges.size), size: 12 * owners };
    const sums = { offset: align(donors.offset + donors.size), size: 4 * owners };
    return { edges, donors, sums, bytes: sums.offset + sums.size };
  }

  /** Transport reads fineVolume and writes scratchVolume: every owner's V,
   * gather's rows with the regular dust floor applied (dust: the cleanup's
   * vertex field, tuning and accounting; resolved as UniformMixedCleanup).
   * The orphan census or encodeCopy returns it to fineVolume. Its rows and
   * donor sums borrow a scratch buffer: bindScratch before the first encode. */
  constructor(private readonly device: GPUDevice, layout: UniformMixedLayout,
    private readonly fineVolume: GPUTexture,
    private readonly scratchVolume: GPUTexture, private readonly departures: GPUTexture,
    private readonly dust: { phi: GPUTexture, params: GPUBuffer, reductions: GPUBuffer, resolved: boolean },
    private readonly sourceParams?:GPUBuffer,private readonly solid?:UniformMixedSolid,fineTiles?:number) {
    // fineTiles: the simulation ownership's h-tile capacity (every tile unless given).
    this.ownership = new UniformMixedOwnership(device, layout, true, fineTiles);
    this.resolved = dust.resolved;
    if (dust.reductions.size < 48) throw new Error("Mixed transport requires twelve dust accounting words");
    const tiles = this.ownership.capacity.tiles;
    this.sampling = device.createBuffer({ label: "Uniform mixed departure sampling widths", size: tiles * 4,
      usage: GPUBufferUsage.STORAGE });
    this.live = device.createBuffer({ label: "Uniform mixed transport live set", size: (UNIFORM_MIXED_TRANSPORT_LIVE_HEADER + 31 * tiles) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.workBytes = this.sampling.size + this.live.size;
    // Live-list launches are fixed grid-stride grids: the layout's bound
    // (every tile listed), capped where the GPU is saturated.
    this.liveGrid = [Math.min(TRANSPORT_GRID, tiles), Math.max(1, Math.min(TRANSPORT_GRID, Math.ceil(tiles / 64)))];
    this.resourcesLayout = uniformDetailBindLayout(device,{ entries: [
      ...[0, 1, 2, 3].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })),
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      ...(sourceParams?[{binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform" as const}}]:[]),
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
  }

  /** Bind the rows and donor sums in `scratch` (scratchRanges at the
   * ownership's h-tile capacity, from offset 0). Again between frames after
   * that capacity changed: every encode rebuilds them, so nothing carries. */
  bindScratch(scratch: GPUBuffer): void {
    const c = this.ownership.capacity, r = UniformMixedTransportStage.scratchRanges(c.tiles, c.fineTiles), dust = this.dust;
    if (scratch.size < r.bytes) throw new Error("Mixed transport scratch cannot hold its rows and donor sums");
    const group = (input: GPUTexture, output: GPUTexture) => uniformDetailGroup(this.device,{ layout: this.resourcesLayout, entries: [
      { binding: 0, resource: { buffer: this.sampling } },
      { binding: 1, resource: { buffer: scratch, ...r.edges } },
      { binding: 2, resource: { buffer: scratch, ...r.donors } },
      { binding: 3, resource: { buffer: scratch, ...r.sums } },
      { binding: 4, resource: input }, { binding: 5, resource: output },
      { binding: 6, resource: this.departures },
      ...(this.sourceParams?[{binding:7,resource:{buffer:this.sourceParams,size:UNIFORM_PARAMS_BYTES}}]:[]),
      { binding: 8, resource: { buffer: this.live } },
      { binding: 9, resource: dust.phi }, { binding: 10, resource: { buffer: dust.params, size: 32 } },
      { binding: 11, resource: { buffer: dust.reductions, size: 48 } },
    ] });
    this.transportGroup = group(this.fineVolume, this.scratchVolume);
    this.copyGroup = group(this.scratchVolume, this.fineVolume);
  }

  async initialize(): Promise<void> {
    // The generator reads the lattice and tile count only (fixed by capacity).
    const module = uniformDetailModule(this.device,{ label: "Uniform mixed conservative transport",
      code: uniformMixedCountedEntriesWGSL(uniformMixedTransportWGSL(this.ownership.layout,!!this.sourceParams,!!this.solid,this.resolved,this.solid?.coarse?.count), ["liveSeed", "liveSeedCoarse"]) });
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
    if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.topologyLayout, this.resourcesLayout, ...(this.solid?[this.solid.tileLayout]:[])] });
    const twin = (create: (solid: Record<string, number>) => Promise<GPUComputePipeline>) => uniformMixedSolidPipeline(this.solid, create);
    const keys = [...new Set([...liveLaunches, copyLaunch].map(([entry, list]) => `${entry}:${list}`))].map(key => key.split(":") as [Entry, string]);
    const tiers = Promise.all(keys.map(async ([entry, list]) => { this.pipelines.set(`${entry}:${list}`, await compileMixedTiers(umCellWidth => twin(solid =>
      uniformDetailPipeline(this.device,this.ownership,{ layout, compute: { module, entryPoint: tierEntries[entry]?.[umCellWidth===1?0:1] ?? entry,
        constants: { umCellWidth, umDispatchX: this.dispatchX, umTransportList: Number(list), ...solid } } })))); }));
    const tile = (entryPoint: string, constants: Record<string, number> = {}) => twin(solid => uniformDetailPipeline(this.device,this.ownership,{ layout,
      compute: { module, entryPoint, constants: { umCellWidth: 1, umDispatchX: this.dispatchX, ...constants, ...solid } } }));
    // The dependency chain of the live set, in order: see uniformMixedTransportLiveWGSL.
    const S = 1, R1 = 2, D2 = 4, Q2 = 8, D1 = 16, Q1 = 32, Q0 = 64, DONOR = 128;
    const chain: [string, number, number][] = [["liveGather", S, R1], ["liveScatter", R1, D2], ["liveGather", D2, Q2], ["liveScatter", Q2, D1],
      ["liveGather", D1, Q1], ["liveGather", Q1, Q0], ["liveScatter", S | Q1 | Q0, DONOR]];
    const live = Promise.all([tile("liveSeed", { umCountedJobs: UNIFORM_MIXED_COUNTED.fineTiles }), tile("liveSeedCoarse", { umCountedJobs: UNIFORM_MIXED_COUNTED.coarseTiles }), ...chain.map(([entry, tpFrom, tpInto]) => tile(entry, { tpFrom, tpInto })),
      tile("liveCompact")]).then(pipelines => { this.livePipelines.push(...pipelines); });
    await Promise.all([tiers, live]);
  }

  /** The variant of a pipeline for the scene's current solids. */
  private variant(pipeline: GPUComputePipeline): GPUComputePipeline { return uniformDetailPick(this.solid?.select(pipeline) ?? pipeline); }

  private begin(encoder: GPUCommandEncoder, label: string, group?: UniformDetailGroup): GPUComputePassEncoder {
    if (!group) throw new Error("Mixed transport scratch has not been bound");
    const pass = encoder.beginComputePass({ label: `Uniform mixed ${label}` });
    pass.setBindGroup(0, this.topologyGroup); pass.setBindGroup(1, group.group);
    if (this.solid) pass.setBindGroup(2, this.solid.tileGroup);
    return pass;
  }

  /** One direct grid-stride launch per tier (0 = h, 1 = 4h) of a live list,
   * in a pass already bound to the entry's group. */
  private run(pass: GPUComputePassEncoder, entry: Entry, list: LiveList): void {
    const pipelines = this.pipelines.get(`${entry}:${list}`);
    if (!pipelines) throw new Error("Mixed transport has not been initialized");
    for (const tier of [0, 1] as const) {
      const pipeline=this.variant(pipelines[tier]!);
      if(tier===1&&(entry==="rowsFallback"||entry==="rowsDivide"))
        this.ownership.dispatchBuffered(pass,pipeline,"coarse",Math.min(TRANSPORT_GRID,Math.ceil(this.ownership.capacity.tiles/4)),4);
      else this.ownership.dispatchTierCounted(pass,pipeline,tier,false,false,this.liveGrid[tier]);
    }
  }

  /** Rows and donors whose restricted transport equals the dense one, from
   * this frame's departures and volume (per-tile reach dependency chain). */
  private encodeLiveSet(encoder: GPUCommandEncoder): void {
    const [seed, seedCoarse] = this.livePipelines;
    if (!seed || !seedCoarse) throw new Error("Mixed transport has not been initialized");
    // Every tile of the lattice (UM_TILES): no membership sizes these.
    const tiles = this.ownership.capacity.tiles, groups = Math.ceil(tiles / 64);
    encoder.clearBuffer(this.live, 0, UNIFORM_MIXED_TRANSPORT_LIVE_HEADER * 4);
    const pass = this.begin(encoder, "transport live set", this.transportGroup);
    // Seeds cover the h/4h partition: a workgroup per h tile, a lane per 4h tile.
    this.ownership.dispatchTierCounted(pass, this.variant(seed), 0);
    this.ownership.dispatchTierCounted(pass, this.variant(seedCoarse), 1);
    for (const pipeline of this.livePipelines.slice(2)) { pass.setPipeline(this.variant(pipeline)); pass.dispatchWorkgroups(Math.min(groups, this.dispatchX), Math.ceil(groups / this.dispatchX)); }
    pass.end();
  }

  /** Returns transported V to the fine volume when no cleanup does: only
   * gather's rows differ there. Encode after encodeTransport. */
  encodeCopy(encoder: GPUCommandEncoder): void {
    const pass = this.begin(encoder, "transport copy", this.copyGroup);
    this.run(pass, ...copyLaunch); pass.end();
  }

  /** Call only after tracing: extension shares the edge/donor backing.
   * Leaves V in the scratch volume (constructor). */
  encodeTransport(encoder: GPUCommandEncoder): void {
    this.encodeLiveSet(encoder);
    // One pass: a dispatch is its own usage scope, and every entry here binds the transport group.
    const pass = this.begin(encoder, "transport", this.transportGroup);
    for (const [entry, list] of liveLaunches) this.run(pass, entry, list);
    pass.end();
  }

  destroy(): void { this.ownership.destroy(); this.sampling.destroy(); this.live.destroy(); }
}
