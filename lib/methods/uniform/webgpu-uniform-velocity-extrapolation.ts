import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import { uniformVelocityExtrapolationShader } from "./webgpu-uniform-velocity-extrapolation.wgsl";

type Dims3 = readonly [number, number, number];
/** mixedHierarchyTail's workgroup storage (MIXED_TAIL_CELLS/LEVELS in the WGSL). */
const MIXED_TAIL_CELLS = 256;
const MIXED_TAIL_LEVELS = 16;

interface HierarchyLevel {
  readonly dims: Dims3;
  readonly down: GPUTexture;
  readonly up: GPUTexture;
  readonly originsDown: GPUTexture;
  readonly originsUp: GPUTexture;
}

/**
 * Sec. 3.3 nearest-source velocity hierarchy from the 4h root down.
 *
 * The mixed owners write the root's known velocities and source bounds and
 * read the filled root back. This module owns the levels, their kernels, bind
 * groups and dispatch schedule; the parent supplies only its parameters.
 */
export class WebGPUUniformVelocityExtrapolator {
  /** What each kernel reads and writes: restrict and tail share one layout. */
  private readonly layouts: { readonly restrict: GPUBindGroupLayout; readonly prolong: GPUBindGroupLayout; readonly replay: GPUBindGroupLayout };
  /** levels[0] is the 4h root; the last is one cell. */
  private readonly levels: HierarchyLevel[] = [];
  /** restrictGroups[level-1] fills `level` from the one above it. */
  private readonly restrictGroups: GPUBindGroup[] = [];
  private readonly prolongGroups: GPUBindGroup[] = [];
  private readonly replayGroups: GPUBindGroup[] = [];
  /** True once the ceil(n/4) level exists and tiles the lattice exactly. */
  readonly coarseVelocityTableAvailable: boolean;
  /** Levels [1, tailLevel) dispatch one lane per (cell, component); levels
   * [tailLevel, end) run in one workgroup (mixedHierarchyTail). */
  private readonly tailLevel: number;
  private readonly tailGroup?: GPUBindGroup;
  private mixedPipelines?: { readonly restrict: GPUComputePipeline; readonly prolong: GPUComputePipeline; readonly replay: GPUComputePipeline; readonly tail: GPUComputePipeline };

  constructor(private readonly device: GPUDevice, private readonly dims: Dims3, params: GPUBuffer) {
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
    ];
    const layout = (kernel: string, bindings: readonly number[]) => device.createBindGroupLayout({
      label: `Uniform Sec. 3.3 extrapolation layout - ${kernel}`, entries: entries.filter(entry => bindings.includes(entry.binding)),
    });
    this.layouts = { restrict: layout("restrict", [0, 1, 3, 4, 6]), prolong: layout("prolong", [0, 1, 2, 3, 4, 5, 6]), replay: layout("replay", [1, 2, 3, 5]) };

    // CM11b Sec. 3.3.1 requires the reverse pass to leave every finest-grid
    // value known, which takes a coarsest level that sees liquid from every
    // cell: coarsen until EVERY axis is one cell, a collapsed axis staying at
    // one (ceil(1/2) = 1). Prolongation fills only unknown values, so levels
    // above one that is already fully known change nothing. Ceil division
    // preserves coverage for scene sizes that are not exact powers of two.
    // Storage starts at the 4h root: nothing is held for the lattice or n/2.
    const half = (d: Dims3): Dims3 => [Math.ceil(d[0] / 2), Math.ceil(d[1] / 2), Math.ceil(d[2] / 2)];
    for (let levelDims = half(dims); Math.max(...levelDims) > 1;) {
      levelDims = half(levelDims);
      const size = levelDims;
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
    const levels = this.levels, count = levels.length;
    const coarse = levels[0];
    // The 2D reference's single cell layer is a collapsed axis at every level;
    // its sampler clamps to that one coarse layer, so the table still applies.
    this.coarseVelocityTableAvailable = !!coarse
      && dims.every((value, axis) => value === 4 * coarse.dims[axis]! || (value === 1 && coarse.dims[axis] === 1));

    const view = (texture: GPUTexture) => texture.createView();
    const restrict = (source: GPUTexture, sourceOrigins: GPUTexture, target: GPUTexture, targetOrigins: GPUTexture) => device.createBindGroup({ layout: this.layouts.restrict, entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: view(source) },
      { binding: 3, resource: view(target) },
      { binding: 4, resource: view(sourceOrigins) },
      { binding: 6, resource: view(targetOrigins) },
    ] });
    // The tail: the coarsest levels whose cells fit one workgroup's storage.
    let tail = count, cells = 0;
    while (tail > 1 && count - (tail - 1) <= MIXED_TAIL_LEVELS) {
      const n = levels[tail - 1]!.dims.reduce((a, b) => a * b, 1);
      if (cells + n > MIXED_TAIL_CELLS) break;
      cells += n; tail -= 1;
    }
    this.tailLevel = tail;
    for (let level = 1; level < tail; level += 1) {
      const finer = levels[level - 1]!, target = levels[level]!;
      this.restrictGroups.push(restrict(finer.down, finer.originsDown, target.down, target.originsDown));
    }
    // The tail leaves its finest level complete: restricted when nothing lies
    // below it, filled otherwise.
    const tailSource = levels[tail - 1], tailTarget = levels[tail];
    if (tailSource && tailTarget) this.tailGroup = tail === count - 1
      ? restrict(tailSource.down, tailSource.originsDown, tailTarget.down, tailTarget.originsDown)
      : restrict(tailSource.down, tailSource.originsDown, tailTarget.up, tailTarget.originsUp);
    // Prolong fills a level's unknown values from the complete level below
    // it, recording which coarse taps it took; replay re-reads those taps.
    for (let level = 0; level < Math.min(tail, count - 1); level += 1) {
      const fine = levels[level]!, below = levels[level + 1]!, last = level === count - 2;
      const coarser = last ? below.down : below.up, coarserOrigins = last ? below.originsDown : below.originsUp;
      this.prolongGroups.push(device.createBindGroup({ layout: this.layouts.prolong, entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: view(coarser) },
        { binding: 2, resource: view(fine.down) },
        { binding: 3, resource: view(fine.up) },
        { binding: 4, resource: view(coarserOrigins) },
        { binding: 5, resource: view(fine.originsDown) },
        { binding: 6, resource: view(fine.originsUp) },
      ] }));
      this.replayGroups.push(device.createBindGroup({ layout: this.layouts.replay, entries: [
        { binding: 1, resource: view(coarser) },
        { binding: 2, resource: view(fine.down) },
        { binding: 3, resource: view(fine.up) },
        { binding: 5, resource: view(fine.originsUp) },
      ] }));
    }
  }

  /** Continue the one extension hierarchy from canonical 4h support. Mixed
   * owners provide the accurate band and its source bounds; this never visits
   * or reconstructs the finest lattice. */
  prepareMixedContinuation() {
    const root=this.levels[0],mixed=this.mixedPipelines,tail=this.tailLevel;
    if(!mixed || !root || root.dims.some((n,a)=>n*4!==this.dims[a]))
      throw new Error("Mixed extension requires the resident 4h nearest-source hierarchy");
    const hasLower=this.levels.length>1;
    // Each level's workgroups: fixed with the hierarchy, so the per-frame encode builds nothing.
    const groups=this.levels.map(level=>level.dims.map(n=>Math.ceil(n/4)) as [number,number,number]);
    return {input:root.down,inputOrigins:root.originsDown,
      output:hasLower?root.up:root.down,outputOrigins:hasLower?root.originsUp:root.originsDown,
      /** Encodes into the caller's compute pass; its group 0 is left rebound. */
      encode:(pass:GPUComputePassEncoder,replay=false)=>{
        const count=this.levels.length,up=replay?this.replayGroups:this.prolongGroups;
        pass.setPipeline(mixed.restrict);
        for(let level=1;level<tail;level++){const g=groups[level]!;pass.setBindGroup(0,this.restrictGroups[level-1]!);pass.dispatchWorkgroups(g[0],g[1],g[2]);}
        if(this.tailGroup){pass.setPipeline(mixed.tail);pass.setBindGroup(0,this.tailGroup);pass.dispatchWorkgroups(1);}
        pass.setPipeline(replay?mixed.replay:mixed.prolong);
        for(let level=Math.min(tail,count-1)-1;level>=0;level--){const g=groups[level]!;pass.setBindGroup(0,up[level]!);pass.dispatchWorkgroups(g[0],g[1],g[2]);}
      }};
  }

  /** The down/up velocity levels and their source-bound pairs. */
  get scratchBytes(): number {
    return this.levels.reduce((sum, level) => sum + level.dims[0] * level.dims[1] * level.dims[2] * 6 * 16, 0);
  }

  get hierarchyLevelCount(): number { return this.levels.length; }

  async initialize(signal?: AbortSignal): Promise<void> {
    if (this.mixedPipelines) return;
    const compiler = gpuCompilationManagerFor(this.device);
    const shaderModule = compiler.createShaderModule({
      label: "Uniform Sec. 3.3 extrapolation kernels",
      code: uniformVelocityExtrapolationShader,
    });
    const compile = (label: string, entryPoint: string, layout: GPUBindGroupLayout) => compiler.compileComputePipeline({
      label, layout: this.device.createPipelineLayout({ label: `${label} layout`, bindGroupLayouts: [layout] }),
      compute: { module: shaderModule, entryPoint },
    }, { priority: "critical", signal });
    const [restrict, prolong, replay, tail] = await Promise.all([
      compile("Uniform mixed extension hierarchy restrict", "mixedRestrictKnownVelocity", this.layouts.restrict),
      compile("Uniform mixed extension hierarchy prolong", "mixedProlongUnknownVelocity", this.layouts.prolong),
      compile("Uniform mixed extension hierarchy replay", "mixedReplayUnknownVelocity", this.layouts.replay),
      compile("Uniform mixed extension hierarchy tail", "mixedHierarchyTail", this.layouts.restrict),
    ]);
    this.mixedPipelines = { restrict, prolong, replay, tail };
  }

  destroy(): void {
    this.levels.forEach((level) => { level.down.destroy(); level.up.destroy(); level.originsDown.destroy(); level.originsUp.destroy(); });
  }
}
