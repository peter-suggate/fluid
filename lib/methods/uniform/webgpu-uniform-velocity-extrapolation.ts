import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import type { UniformLatticeAddress } from "./uniform-lattice-address";
import { UNIFORM_EXTRAPOLATION_BLOCK_SWEEPS, uniformVelocityExtrapolationShader } from "./webgpu-uniform-velocity-extrapolation.wgsl";

type Dims3 = readonly [number, number, number];
/** The 4h, 8h, 16h and 32h levels: 8, 4, 2 and 1 cells a block axis. */
const LEVEL_EDGES = [8, 4, 2, 1] as const;
/** Cells a kernel workgroup covers (three component lanes each). */
const GROUP_CELLS = 64;

interface HierarchyLevel {
  readonly dims: Dims3;
  readonly down: GPUTexture;
  readonly up: GPUTexture;
  readonly originsDown: GPUTexture;
  readonly originsUp: GPUTexture;
}

/** Bytes of one level cell: `down` and `up` rgba32float and two rgba32uint
 * source-bound pairs. */
const CELL_BYTES = 6 * 16;

/**
 * Sec. 3.3 nearest-source velocity hierarchy from the 4h root down, by pool
 * block: each level's textures hold every block's cells at the block's place
 * in the block grid, so their size follows the pool and not the scene's box.
 *
 * The mixed owners write the root's known velocities and source bounds and
 * read the filled root back. This module owns the levels, their kernels, bind
 * groups and dispatch schedule; the parent supplies its parameters and the
 * page pool's tail (page table, block records, neighbour rows).
 */
export class WebGPUUniformVelocityExtrapolator {
  /** What each kernel reads and writes: restrict and sweep share one layout. */
  private readonly layouts: { readonly restrict: GPUBindGroupLayout; readonly prolong: GPUBindGroupLayout; readonly replay: GPUBindGroupLayout };
  /** levels[0] is the 4h root; the last is one cell a block. */
  private readonly levels: HierarchyLevel[] = [];
  /** Workgroups of a launch over every block's cells of each level. */
  private readonly groupCounts: readonly number[];
  /** restrictGroups[level-1] fills `level` from the one above it. */
  private restrictGroups: GPUBindGroup[] = [];
  /** The 32h level's sweeps, alternating its `down` and `up` textures. */
  private sweepGroups: GPUBindGroup[] = [];
  private prolongGroups: GPUBindGroup[] = [];
  private replayGroups: GPUBindGroup[] = [];
  /** True where 4h cells tile the lattice exactly. */
  readonly coarseVelocityTableAvailable: boolean;
  private mixedPipelines?: { readonly restrict: GPUComputePipeline; readonly sweep: GPUComputePipeline; readonly prolong: GPUComputePipeline; readonly replay: GPUComputePipeline };

  constructor(private readonly device: GPUDevice, private readonly address: UniformLatticeAddress, private readonly params: GPUBuffer) {
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
      | GPUTextureUsage.COPY_SRC;
    const float = { sampleType: "unfilterable-float", viewDimension: "3d" } as const;
    const uint = { sampleType: "uint", viewDimension: "3d" } as const;
    const entries: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: float },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: float },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: uint },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: uint },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32uint", viewDimension: "3d" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ];
    const layout = (kernel: string, bindings: readonly number[]) => device.createBindGroupLayout({
      label: `Uniform Sec. 3.3 extrapolation layout - ${kernel}`, entries: entries.filter(entry => bindings.includes(entry.binding)),
    });
    this.layouts = { restrict: layout("restrict", [0, 1, 3, 4, 6, 7]), prolong: layout("prolong", [0, 1, 2, 3, 4, 5, 6, 7]), replay: layout("replay", [1, 2, 3, 5, 7]) };

    // CM11b Sec. 3.3.1 requires the reverse pass to leave every finest-grid
    // value known that a source reaches. A block's levels stop at its one
    // 32h cell; mixedSweepBlocks carries values between blocks from there.
    // Storage starts at the 4h root: nothing is held for the lattice or n/2.
    const grid = address.blockGrid;
    for (const edge of LEVEL_EDGES) {
      const size: Dims3 = [edge * grid[0], edge * grid[1], edge * grid[2]];
      const velocity = (direction: "down" | "up") => device.createTexture({
        label: `Uniform Sec. 3.3 hierarchy ${direction} ${size.join("x")}`,
        size, dimension: "3d", format: "rgba32float", usage,
      });
      // Lower source bounds in the first nz layers, upper in the second.
      const origins = () => device.createTexture({
        label: `Uniform nearest-source origins ${size.join("x")}`,
        size: [size[0], size[1], 2 * size[2]], dimension: "3d", format: "rgba32uint", usage,
      });
      this.levels.push({ dims: size, down: velocity("down"), up: velocity("up"), originsDown: origins(), originsUp: origins() });
    }
    this.groupCounts = LEVEL_EDGES.map(edge => Math.ceil(address.blocks * edge ** 3 / GROUP_CELLS));
    if (this.groupCounts[0]! > device.limits.maxComputeWorkgroupsPerDimension)
      throw new Error(`Mixed extension hierarchy cannot launch over ${address.blocks} blocks`);
    this.coarseVelocityTableAvailable = address.cells.every(value => value % 4 === 0 || value === 1);
  }

  /** Bind the levels to the page pool's tail: `topology` holds the page
   * table from word `pageTable` on, then the block records and the neighbour
   * rows. Called once the owner of that buffer exists. */
  private attach(topology: GPUBuffer, pageTable: number): void {
    const device = this.device, levels = this.levels, count = levels.length, params = this.params;
    const offset = 4 * pageTable;
    if (offset % device.limits.minStorageBufferOffsetAlignment !== 0) throw new Error("Mixed extension hierarchy needs an aligned page table");
    const pool = { binding: 7, resource: { buffer: topology, offset } };
    const view = (texture: GPUTexture) => texture.createView();
    const restrict = (source: GPUTexture, sourceOrigins: GPUTexture, target: GPUTexture, targetOrigins: GPUTexture) => device.createBindGroup({ layout: this.layouts.restrict, entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: view(source) },
      { binding: 3, resource: view(target) },
      { binding: 4, resource: view(sourceOrigins) },
      { binding: 6, resource: view(targetOrigins) },
      pool,
    ] });
    this.restrictGroups = []; this.sweepGroups = []; this.prolongGroups = []; this.replayGroups = [];
    for (let level = 1; level < count; level += 1) {
      const finer = levels[level - 1]!, target = levels[level]!;
      this.restrictGroups.push(restrict(finer.down, finer.originsDown, target.down, target.originsDown));
    }
    const top = levels[count - 1]!;
    for (let sweep = 0; sweep < UNIFORM_EXTRAPOLATION_BLOCK_SWEEPS; sweep += 1) this.sweepGroups.push(sweep % 2 === 0
      ? restrict(top.down, top.originsDown, top.up, top.originsUp)
      : restrict(top.up, top.originsUp, top.down, top.originsDown));
    // Prolong fills a level's unknown values from the complete level below
    // it, recording which coarse taps it took; replay re-reads those taps.
    for (let level = 0; level < count - 1; level += 1) {
      const fine = levels[level]!, below = levels[level + 1]!;
      this.prolongGroups.push(device.createBindGroup({ layout: this.layouts.prolong, entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: view(below.up) },
        { binding: 2, resource: view(fine.down) },
        { binding: 3, resource: view(fine.up) },
        { binding: 4, resource: view(below.originsUp) },
        { binding: 5, resource: view(fine.originsDown) },
        { binding: 6, resource: view(fine.originsUp) },
        pool,
      ] }));
      this.replayGroups.push(device.createBindGroup({ layout: this.layouts.replay, entries: [
        { binding: 1, resource: view(below.up) },
        { binding: 2, resource: view(fine.down) },
        { binding: 3, resource: view(fine.up) },
        { binding: 5, resource: view(fine.originsUp) },
        pool,
      ] }));
    }
  }

  /** Continue the one extension hierarchy from canonical 4h support. Mixed
   * owners provide the accurate band and its source bounds; this never visits
   * or reconstructs the finest lattice. */
  prepareMixedContinuation() {
    const root=this.levels[0]!,mixed=this.mixedPipelines,counts=this.groupCounts,top=this.levels.length-1;
    if(!mixed)throw new Error("Mixed extension requires the resident 4h nearest-source hierarchy");
    return {input:root.down,inputOrigins:root.originsDown,output:root.up,outputOrigins:root.originsUp,
      attach:(topology:GPUBuffer,pageTable:number)=>this.attach(topology,pageTable),
      /** Encodes into the caller's compute pass; its group 0 is left rebound. */
      encode:(pass:GPUComputePassEncoder,replay=false)=>{
        const up=replay?this.replayGroups:this.prolongGroups;
        if(up.length!==top)throw new Error("Mixed extension hierarchy is not attached to a page pool");
        pass.setPipeline(mixed.restrict);
        for(let level=1;level<=top;level++){pass.setBindGroup(0,this.restrictGroups[level-1]!);pass.dispatchWorkgroups(counts[level]!);}
        pass.setPipeline(mixed.sweep);
        for(let sweep=0;sweep<UNIFORM_EXTRAPOLATION_BLOCK_SWEEPS;sweep++){pass.setBindGroup(0,this.sweepGroups[sweep]!);pass.dispatchWorkgroups(counts[top]!);}
        pass.setPipeline(replay?mixed.replay:mixed.prolong);
        for(let level=top-1;level>=0;level--){pass.setBindGroup(0,up[level]!);pass.dispatchWorkgroups(counts[level]!);}
      }};
  }

  /** The down/up velocity levels and their source-bound pairs. */
  get scratchBytes(): number {
    return this.levels.reduce((sum, level) => sum + level.dims[0] * level.dims[1] * level.dims[2] * CELL_BYTES, 0);
  }

  get hierarchyLevelCount(): number { return this.levels.length; }

  async initialize(signal?: AbortSignal): Promise<void> {
    if (this.mixedPipelines) return;
    const compiler = gpuCompilationManagerFor(this.device);
    const shaderModule = compiler.createShaderModule({
      label: "Uniform Sec. 3.3 extrapolation kernels",
      code: uniformVelocityExtrapolationShader(this.address.blocks, this.address.blockBits),
    });
    const compile = (label: string, entryPoint: string, layout: GPUBindGroupLayout) => compiler.compileComputePipeline({
      label, layout: this.device.createPipelineLayout({ label: `${label} layout`, bindGroupLayouts: [layout] }),
      compute: { module: shaderModule, entryPoint },
    }, { priority: "critical", signal });
    const [restrict, sweep, prolong, replay] = await Promise.all([
      compile("Uniform mixed extension hierarchy restrict", "mixedRestrictKnownVelocity", this.layouts.restrict),
      compile("Uniform mixed extension hierarchy block sweep", "mixedSweepBlocks", this.layouts.restrict),
      compile("Uniform mixed extension hierarchy prolong", "mixedProlongUnknownVelocity", this.layouts.prolong),
      compile("Uniform mixed extension hierarchy replay", "mixedReplayUnknownVelocity", this.layouts.replay),
    ]);
    this.mixedPipelines = { restrict, sweep, prolong, replay };
  }

  destroy(): void {
    this.levels.forEach((level) => { level.down.destroy(); level.up.destroy(); level.originsDown.destroy(); level.originsUp.destroy(); });
  }
}
