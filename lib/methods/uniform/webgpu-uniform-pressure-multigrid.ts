import { UNIFORM_MIXED_CONTINUATION_LEVEL as MIXED_CONTINUATION_LEVEL, uniformPressureScratchShader } from "./uniform-scratch-arena";
import type { UniformTexturePages } from "./uniform-texture-pages";
import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import { uniformPressureInPlaceSmootherWGSL, uniformPressureMultigridWGSL } from "./webgpu-uniform-pressure-multigrid.wgsl";

import {
  UNIFORM_CM11A_PHI_PRESERVATION_LEVELS,
  UNIFORM_PRESSURE_RESIDUAL_TOLERANCE,
  type UniformCM11aSchedule,
} from "./pressure-policy";
export * from "./pressure-policy";
export * from "./pressure-plan";
import { planUniformCM11aHierarchy, type UniformCM11aLevelSize } from "./pressure-plan";

/**
 * Largest level (halo included) whose smoothing visit runs as one workgroup.
 * Measured on an M1 Max: 10^3 and below win (-2.6% mini64, -1.4% fig7 128^3);
 * at 18^3 a lane is ~35 serial updates per visit even 1024 wide and the fused
 * visit loses to twelve ~6 us launches on the GPU, though it still wins on CPU.
 */
const FUSED_VISIT_MAX_CELLS = Math.max(0, Number((typeof process !== "undefined"
  ? process.env?.FLUID_UNIFORM_FUSED_VISIT_CELLS : undefined) ?? 1000));

const ENTRY_POINTS = [
  "mgDownsampleTopology", "mgExtrapolatePhiOneCell", "mgBakeCoefficients", "mgBuildSmoothTiles", "mgBuildCycleTiles",
] as const;
type EntryPoint = typeof ENTRY_POINTS[number];

const ENTRY_BINDINGS: Readonly<Record<EntryPoint, readonly number[]>> = Object.freeze({
  mgDownsampleTopology: [0, 5, 6, 7, 8], mgExtrapolatePhiOneCell: [0, 5, 6, 7],
  mgBakeCoefficients: [0, 5, 7, 15],
  mgBuildSmoothTiles: [0, 11, 14, 18],
  mgBuildCycleTiles: [0, 18, 19],
});

/**
 * The same split, per entry point and computed once. The plan walk runs this
 * test thousands of times and rebuilding the sets and filtered lists inside it
 * was a measurable share of a re-plan.
 */
const SAMPLED_BINDINGS: readonly number[] = [5, 7, 11, 14];
const WRITABLE_BINDINGS: readonly number[] = [6, 8, 15];
const entryBindingsWhere = (keep: readonly number[]): Readonly<Record<EntryPoint, readonly number[]>> =>
  Object.freeze(Object.fromEntries((Object.entries(ENTRY_BINDINGS) as [EntryPoint, readonly number[]][])
    .map(([entry, bindings]) => [entry, Object.freeze(bindings.filter((b) => keep.includes(b)))]))) as
    Readonly<Record<EntryPoint, readonly number[]>>;
const ENTRY_SAMPLED = entryBindingsWhere(SAMPLED_BINDINGS);
const ENTRY_WRITABLE = entryBindingsWhere(WRITABLE_BINDINGS);

type TexturePair = readonly [GPUTexture, GPUTexture];
export interface UniformPressureMultigridLevel {
  /** Texture dimensions include one persistent solid/domain halo cell. */
  readonly dimensions: readonly [number, number, number];
  readonly pressure: TexturePair;
  readonly rhs: TexturePair;
  readonly phi: TexturePair;
  readonly volume: TexturePair;
  readonly residual: TexturePair;
  readonly minimum: TexturePair;
  /** Immutable (+x,+y,+z) pressure coefficients and liquid flag for this solve. */
  readonly coefficients: GPUTexture;
}

interface PlannedDispatch {
  readonly pipeline: GPUComputePipeline;
  readonly group: GPUBindGroup;
  readonly entryPoint: EntryPoint;
  readonly workgroups: readonly [number, number, number];
}

interface GroupResources {
  pressureIn: GPUTexture; pressureOut: GPUTexture;
  rhsIn: GPUTexture; rhsOut: GPUTexture;
  phiIn: GPUTexture; phiOut: GPUTexture;
  volumeIn: GPUTexture; volumeOut: GPUTexture;
  residualIn: GPUTexture; residualOut: GPUTexture;
  minimumIn: GPUTexture; minimumOut: GPUTexture;
  coefficientsIn: GPUTexture; coefficientsOut: GPUTexture;
}

/**
 * The 4h-and-below CM11a hierarchy the mixed pressure solve borrows: its levels,
 * the per-solve setup kernels (topology pyramid, phi continuation, baked
 * coefficients, tile lists), their module and their bind groups.
 */
export class WebGPUUniformPressureMultigrid {
  /** The continuation's levels, its 4h root first. */
  private readonly levels: readonly UniformPressureMultigridLevel[];
  /** Levels of the whole hierarchy: the continuation's and the two finer ones it never visits. */
  readonly levelCount: number;
  /** The published grid pressure: one texel no kernel writes. */
  readonly pressureTexture: GPUTexture;
  /** The root's liquid tiles, then one classification word per tile. */
  private readonly smoothTiles?: GPUBuffer;
  /** The root's liquid (+) one tile, plus every constrained tile: the caller's cycles run from this. */
  private readonly cycleTiles?: GPUBuffer;
  private readonly rootTiles: number;
  private readonly smoothTileInputLayout?: GPUBindGroupLayout;
  private readonly smoothTileInputGroup?: GPUBindGroup;

  private tileEntry(entry: EntryPoint): boolean { return entry === "mgBuildSmoothTiles" || entry === "mgBuildCycleTiles"; }
  private get shaderSource():string {
    return uniformPressureScratchShader(uniformPressureMultigridWGSL+(this.inPlaceCapable?uniformPressureInPlaceSmootherWGSL:""));
  }
  private readonly logicalDimensions = new Map<GPUTexture,readonly [number,number,number]>();
  private readonly arena: GPUBuffer;
  readonly allocatedBytes: number;
  private readonly spacing: readonly [number, number, number];
  /** Finest physical lattice, the reference every level's spacing scales from. */
  private readonly finestSize!: UniformCM11aLevelSize;
  private readonly groupLayouts: Readonly<Record<EntryPoint, GPUBindGroupLayout>>;
  private readonly ownedParams: GPUBuffer[] = [];
  private readonly ownedGroups: GPUBindGroup[] = [];
  /**
   * The plan is thousands of dispatches over a few dozen textures, and the
   * same (entry point, resources, parameters) triple recurs on every cycle of
   * every level. Building each of the three afresh per dispatch is what made a
   * re-plan a visible hitch: on the whole 64x512x64 domain, 59k texture views,
   * 4.2k parameter buffers and 4.2k bind groups cost 143 ms of the 178.
   * Caching them is pure de-duplication -- a bind group is immutable, and two
   * dispatches naming the same resources need only one -- so the encoded
   * command stream is unchanged.
   */
  private readonly viewCache = new Map<GPUTexture, GPUTextureView>();
  private readonly paramCache = new Map<string, GPUBuffer>();
  private readonly groupCache = new Map<string, GPUBindGroup>();
  /** Stable per-texture ids, so a bind group's resources have a cache key. */
  private readonly textureIds = new Map<GPUTexture, number>();
  private pipelines?: Readonly<Record<EntryPoint, GPUComputePipeline>>;
  private activeResidualTolerance = 0;
  private isDestroyed = false;
  private readonly inPlaceCapable: boolean;
  private inPlaceSmoothing: boolean;

  constructor(private readonly device: GPUDevice,
    dimensions: readonly [number, number, number],
    spacing: readonly [number, number, number],
    schedule: UniformCM11aSchedule,
    /** List the root's tiles for the caller's cycles. The caller vouches
     * that the scene has no depth symmetry. */
    inPlaceSmoothing: boolean, private readonly scratchFields: UniformTexturePages,
    /** The main shader's params uniform, for the tile kernels' minimal group 0. */
    mainParams: GPUBuffer) {
    this.inPlaceCapable = inPlaceSmoothing;
    this.inPlaceSmoothing = this.inPlaceCapable;
    if (!scratchFields.scratch) throw new Error("The CM11a hierarchy needs the shared scratch arena");
    this.arena = scratchFields.scratch.buffer;
    const hierarchy = planUniformCM11aHierarchy(
      dimensions as readonly [number, number, number], 3);
    if (hierarchy.rejection) throw new RangeError(hierarchy.rejection);
    if (!spacing.every((value) => Number.isFinite(value) && value > 0)) {
      throw new RangeError("CM11a grid spacing must be positive and finite");
    }
    this.spacing = spacing;
    this.levelCount = hierarchy.levelCount;
    this.finestSize = hierarchy.levelDimensions[0]!;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
      | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    let allocatedBytes = 0;
    const texture = (label: string, format: GPUTextureFormat,
      size: readonly [number, number, number]) => {
      const result = scratchFields.createTexture({ label, size: [...size], dimension: "3d", format, usage });
      this.logicalDimensions.set(result,size);
      allocatedBytes += size[0] * size[1] * size[2] * (format === "rgba32float" ? 16 : 4);
      return result;
    };
    this.pressureTexture = texture("Uniform CM11a L0 pressure A", "r32float", [1, 1, 1]);
    const levels: UniformPressureMultigridLevel[] = [];
    for (let index = MIXED_CONTINUATION_LEVEL; index < this.levelCount; index += 1) {
      const physicalSize = hierarchy.levelDimensions[index]!;
      const size: readonly [number, number, number] = [physicalSize[0] + 2, physicalSize[1] + 2, physicalSize[2] + 2];
      const pair = (field: string, format: GPUTextureFormat = "r32float"): TexturePair => [
        texture(`Uniform CM11a L${index} ${field} A`, format, size),
        texture(`Uniform CM11a L${index} ${field} B`, format, size),
      ];
      // These fields never ping-pong: the plan always addresses slot zero.
      // Keep the pair-shaped resource interface without allocating an unused
      // second volume or residual field at every level.
      const single = (field: string, format: GPUTextureFormat = "r32float"): TexturePair => {
        const value = texture(`Uniform CM11a L${index} ${field} A`, format, size);
        return [value, value];
      };
      levels.push(Object.freeze({ dimensions: size, pressure: pair("pressure"), rhs: pair("rhs"),
        phi: pair("phi"), volume: single("V", "rgba32float"), residual: single("residual"),
        minimum: pair("p-min"), coefficients: texture(`Uniform CM11a L${index} coefficients`, "rgba32float", size) }));
    }
    this.levels = Object.freeze(levels);

    const root = levels[0];
    this.rootTiles = root ? root.dimensions.reduce((n,d)=>n*Math.ceil(d/4),1) : 0;
    if(this.inPlaceCapable) {
      // Only a root past the fused visit is listed.
      if(root&&root.dimensions.reduce((n,d)=>n*d,1)>FUSED_VISIT_MAX_CELLS){
        const listUsage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;
        // The liquid list is followed by one classification word per tile, so
        // the dilation pass can read a tile's neighbours without rescanning
        // their cells.
        this.smoothTiles=device.createBuffer({label:"Uniform pressure liquid tile list",size:16+8*this.rootTiles,usage:listUsage});
        this.cycleTiles=device.createBuffer({label:"Uniform pressure cycle tile list",size:16+4*this.rootTiles,usage:listUsage});
        allocatedBytes+=this.smoothTiles.size+this.cycleTiles.size;
      }
      // The tile kernels read only the arena and the main params uniform: a
      // minimal group 0 keeps their work lists within device limits.
      this.smoothTileInputLayout=device.createBindGroupLayout({entries:[
        {binding:35,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
        {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]});
      this.smoothTileInputGroup=device.createBindGroup({layout:this.smoothTileInputLayout,entries:[
        {binding:35,resource:{buffer:this.arena}},{binding:6,resource:{buffer:mainParams}}]});
    }
    this.setResidualTolerance(schedule.residualTolerance ?? UNIFORM_PRESSURE_RESIDUAL_TOLERANCE);
    this.allocatedBytes = allocatedBytes;
    const textureBinding = { sampleType: "unfilterable-float", viewDimension: "3d" } as const;
    const vectorStorage = { access: "write-only", format: "rgba32float", viewDimension: "3d" } as const;
    const allEntries: GPUBindGroupLayoutEntry[] = [
      { binding: 18, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 19, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ...[5, 7, 11].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, texture: textureBinding })),
      { binding: 6, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, storageTexture: vectorStorage },
      { binding: 14, visibility: GPUShaderStage.COMPUTE, texture: textureBinding },
      { binding: 15, visibility: GPUShaderStage.COMPUTE, storageTexture: vectorStorage },
    ];
    this.groupLayouts = Object.freeze(Object.fromEntries(ENTRY_POINTS.map((entryPoint) => [entryPoint,
      device.createBindGroupLayout({ label: `Uniform CM11a hierarchy layout - ${entryPoint}`,
        entries: allEntries.filter(({ binding }) => ENTRY_BINDINGS[entryPoint].includes(binding)) }),
    ])) as Record<EntryPoint, GPUBindGroupLayout>);
  }

  setResidualTolerance(value: number): void { this.activeResidualTolerance = Number.isFinite(value) ? Math.max(0, value) : 0; }

  /** The pressure residual tolerance, which the solver hands the mixed pressure solve. */
  get residualTolerance(): number { return this.activeResidualTolerance; }

  /** The root's tile lists are built only without depth symmetry: a
   * continuation prepared after a live edit introduces it has no cycle list. */
  setDepthSymmetry(symmetric: boolean): void {
    const next = this.inPlaceCapable && !symmetric;
    if (next === this.inPlaceSmoothing) return;
    this.inPlaceSmoothing = next;
  }

  /** The setup kernels' group 0 is the caller's, bar the tile kernels': of it
   * they read the params uniform (binding 6) and the arena (binding 35). */
  async initialize(input: { readonly uniformBindGroupLayout: GPUBindGroupLayout;
    readonly signal?: AbortSignal }): Promise<void> {
    this.assertLive(); if (this.pipelines) return;
    const compiler = gpuCompilationManagerFor(this.device);
    const shaderModule = compiler.createShaderModule({ label: "Uniform CM11a pressure hierarchy",
      code: this.shaderSource });
    const entries = await Promise.all(ENTRY_POINTS.filter((entryPoint) =>
      !this.tileEntry(entryPoint) || this.inPlaceSmoothing).map(async (entryPoint) => [entryPoint,
      await compiler.compileComputePipeline({ label: `Uniform CM11a - ${entryPoint}`,
        layout: this.device.createPipelineLayout({ label: `Uniform CM11a layout - ${entryPoint}`,
          bindGroupLayouts: [this.tileEntry(entryPoint) ? this.smoothTileInputLayout! : input.uniformBindGroupLayout, this.groupLayouts[entryPoint]] }),
        compute: { module: shaderModule, entryPoint } },
        { priority: "visible", signal: input.signal })] as const));
    this.pipelines = Object.freeze(Object.fromEntries(entries) as Record<EntryPoint, GPUComputePipeline>);
  }

  /** Borrow the existing 4h-and-below hierarchy for a mixed correction.
   * Setup/dispatch parameters are prepared once. The caller must serialize
   * it with native solves and populate every input, including the halo. */
  prepareMixedContinuation() {
    this.assertLive();const root=this.levels[0];
    if (!this.pipelines || !root || root.dimensions.some((d,a)=>d-2!==this.finestSize[a]!/4))
      throw new Error("Mixed continuation requires the full-domain native 4h hierarchy");
    // The setup alone: the caller runs the traversal itself (see levels).
    const setup=this.buildSetup();
    // Per dispatch, built once: its own pass's descriptor (no shared pass)
    // and whether its group 0 is the tile lists' (see encodeSetup).
    const setupPasses:GPUComputePassDescriptor[]=setup.map(d=>({label:`Uniform mixed continuation ${d.entryPoint}`}));
    const setupTiled=setup.map(d=>this.tileEntry(d.entryPoint));
    const listed=this.listed;
    const field=(texture:GPUTexture)=>{
      const metadata=this.scratchFields.scratchMetadata(texture);
      const dimensions=this.logicalDimensions.get(texture)!;
      const buffer:GPUBufferBinding|undefined=(metadata&0x80000000)!==0?{
        buffer:this.arena,offset:(metadata&0x7ffffffc)*4,
        size:dimensions.reduce((n,d)=>n*d,1)*(texture.format==="rgba32float"?16:4),
      }:undefined;
      return {texture,buffer,dimensions};
    };
    return {
      pressure:field(root.pressure[0]),rhs:field(root.rhs[0]),minimum:field(root.minimum[0]),
      phi:field(root.phi[0]),topology:field(root.volume[0]),
      /** Every level from the root down, for a caller that runs the
       * traversal itself: both parities of the pressure, the V-cycle and
       * Full-Cycle RHS and bound, and the baked coefficients (+x, +y, +z,
       * then the liquid mask of the row and its six neighbours). */
      levels:this.levels.map(l=>({dimensions:l.dimensions,
        pressure:[field(l.pressure[0]),field(l.pressure[1])] as const,rhs:[field(l.rhs[0]),field(l.rhs[1])] as const,
        minimum:[field(l.minimum[0]),field(l.minimum[1])] as const,coefficients:field(l.coefficients)})),
      /** With a cycle list: the far-field seeds the caller writes every solve
       * (both pressure parities at zero, the RHS at zero, the bound at the
       * root's), and the list itself -- its count at
       * word 0 and tile indices of the (dims+3)/4 lattice from word 4, built
       * by the setup from the seeded bound and the baked liquid. */
      cycle:listed?{pressure:field(root.pressure[1]),
        list:this.cycleTiles!,tiles:this.rootTiles}:undefined,
      /** The per-solve setup: the topology pyramid, the baked coefficients
       * and the root's lists (it clears those lists alone: the outer
       * mixed solve owns diagnostics and acceptance). `shared` batches every
       * dispatch into its open pass, ending it only for the list clears. */
      encodeSetup:(encoder:GPUCommandEncoder,uniformGroup:GPUBindGroup,
        shared?:{readonly pass:GPUComputePassEncoder;readonly commands:GPUCommandEncoder})=>{
        this.assertLive();
        if(listed){
          const commands=shared?shared.commands:encoder;
          commands.clearBuffer(this.smoothTiles!,0,4);commands.clearBuffer(this.cycleTiles!,0,4);
        }
        // A shared pass keeps its pipeline and groups between dispatches:
        // rebind only what changes (each call is host encode and submit work).
        let bound:GPUComputePassEncoder|undefined,pipeline:GPUComputePipeline|undefined,bound0:GPUBindGroup|undefined,bound1:GPUBindGroup|undefined;
        for(let i=0;i<setup.length;i++){
          const dispatch=setup[i]!,pass=shared?shared.pass:encoder.beginComputePass(setupPasses[i]!);
          if(bound!==pass){bound=pass;pipeline=undefined;bound0=undefined;bound1=undefined;}
          const group0=setupTiled[i]?this.smoothTileInputGroup!:uniformGroup;
          if(pipeline!==dispatch.pipeline){pipeline=dispatch.pipeline;pass.setPipeline(dispatch.pipeline);}
          if(bound1!==dispatch.group){bound1=dispatch.group;pass.setBindGroup(1,dispatch.group);}
          if(bound0!==group0){bound0=group0;pass.setBindGroup(0,group0);}
          pass.dispatchWorkgroups(dispatch.workgroups[0],dispatch.workgroups[1],dispatch.workgroups[2]);
          if(!shared)pass.end();
        }
      },
    };
  }

  /** One view per texture; the plan asks for the same handful thousands of times. */
  private viewOf(texture: GPUTexture): GPUTextureView {
    let view = this.viewCache.get(texture);
    if (!view) { view = texture.createView(); this.viewCache.set(texture, view); }
    return view;
  }

  private textureId(texture: GPUTexture): number {
    let id = this.textureIds.get(texture);
    if (id === undefined) { id = this.textureIds.size; this.textureIds.set(texture, id); }
    return id;
  }

  /** Whether the continuation's root runs its per-cycle operators from its
   * cycle list: the caller seeds the root's far field every solve and the
   * setup builds the list from those seeds. */
  private get listed(): boolean { return this.inPlaceSmoothing && this.cycleTiles !== undefined; }
  private buildSetup(): PlannedDispatch[] {
    const result: PlannedDispatch[] = [];
    const p = new Array(this.levels.length).fill(0);
    const phi = new Array(this.levels.length).fill(0); const min = new Array(this.levels.length).fill(0);
    const emit = (entryPoint: EntryPoint, sourceIndex: number, destinationIndex = sourceIndex,
      overrides: Partial<GroupResources> = {}, control: readonly [number, number, number, number] = [0, 0, 0, 0],
      dispatchDimensions = this.levels[destinationIndex]!.dimensions) => {
      const source = this.levels[sourceIndex]!, destination = this.levels[destinationIndex]!;
      const defaults: GroupResources = {
        pressureIn: source.pressure[p[sourceIndex]!]!, pressureOut: destination.pressure[p[destinationIndex]! ^ 1]!,
        rhsIn: source.rhs[0], rhsOut: destination.rhs[0], phiIn: source.phi[phi[sourceIndex]!]!,
        phiOut: destination.phi[phi[destinationIndex]! ^ 1]!, volumeIn: source.volume[0], volumeOut: destination.volume[0],
        residualIn: source.residual[0], residualOut: destination.residual[0],
        minimumIn: source.minimum[min[sourceIndex]!]!, minimumOut: destination.minimum[min[destinationIndex]! ^ 1]!,
        coefficientsIn: source.coefficients, coefficientsOut: destination.coefficients,
      };
      const resources = { ...defaults, ...overrides };
      // Indexed by binding, so neither this nor the alias test below allocates
      // per dispatch: the walk runs them thousands of times per re-plan.
      const texturesByBinding: (GPUTexture | undefined)[] = [];
      texturesByBinding[1] = resources.pressureIn; texturesByBinding[2] = resources.pressureOut;
      texturesByBinding[3] = resources.rhsIn; texturesByBinding[4] = resources.rhsOut;
      texturesByBinding[5] = resources.phiIn; texturesByBinding[6] = resources.phiOut;
      texturesByBinding[7] = resources.volumeIn; texturesByBinding[8] = resources.volumeOut;
      texturesByBinding[9] = resources.residualIn; texturesByBinding[10] = resources.residualOut;
      texturesByBinding[11] = resources.minimumIn; texturesByBinding[12] = resources.minimumOut;
      texturesByBinding[14] = resources.coefficientsIn; texturesByBinding[15] = resources.coefficientsOut;
      for (const sampledBinding of ENTRY_SAMPLED[entryPoint]) {
        const texture = texturesByBinding[sampledBinding];
        for (const writableBinding of ENTRY_WRITABLE[entryPoint]) {
          if (texture !== undefined && texture === texturesByBinding[writableBinding]) {
            throw new Error(`Uniform CM11a ${entryPoint} aliases a sampled and writable texture`);
          }
        }
      }
      const paperM = this.levelCount;
      const paperDestination = paperM - MIXED_CONTINUATION_LEVEL - destinationIndex;
      const [params, paramsKey] = this.parameterBuffer(source.dimensions, destination.dimensions,
        MIXED_CONTINUATION_LEVEL + destinationIndex,
        [control[0] || paperDestination, control[1] || paperM - UNIFORM_CM11A_PHI_PRESERVATION_LEVELS,
          control[2], control[3]], texturesByBinding);
      // Only the bindings this entry point declares reach the group, so the
      // key is that filtered list -- two entry points that read the same
      // texture through different bindings must not share a group.
      const bindings = ENTRY_BINDINGS[entryPoint];
      let groupKey = `${entryPoint}|${paramsKey}`;
      for (const binding of bindings) {
        const texture = texturesByBinding[binding];
        if (texture !== undefined) groupKey += `|${binding}:${this.textureId(texture)}`;
      }
      let group = this.groupCache.get(groupKey);
      if (!group) {
        const allEntries: GPUBindGroupEntry[] = [
          ...(this.tileEntry(entryPoint) ? [{ binding: 18, resource: { buffer: this.smoothTiles! } },
            { binding: 19, resource: { buffer: this.cycleTiles! } }] : []),
          { binding: 0, resource: { buffer: params } },
          { binding: 5, resource: this.viewOf(resources.phiIn) }, { binding: 6, resource: this.viewOf(resources.phiOut) },
          { binding: 7, resource: this.viewOf(resources.volumeIn) }, { binding: 8, resource: this.viewOf(resources.volumeOut) },
          { binding: 11, resource: this.viewOf(resources.minimumIn) },
          { binding: 14, resource: this.viewOf(resources.coefficientsIn) },
          { binding: 15, resource: this.viewOf(resources.coefficientsOut) },
        ];
        group = this.device.createBindGroup({ label: `Uniform CM11a bindings - ${entryPoint}`,
          layout: this.groupLayouts[entryPoint],
          entries: allEntries.filter(({ binding }) => bindings.includes(binding)) });
        this.groupCache.set(groupKey, group);
        this.ownedGroups.push(group);
      }
      const pipeline = this.pipelines![entryPoint];
      if (!pipeline) throw new Error(`CM11a ${entryPoint} is not compiled for this hierarchy`);
      result.push({ pipeline, group, entryPoint,
        workgroups: [Math.ceil(dispatchDimensions[0] / 4), Math.ceil(dispatchDimensions[1] / 4), Math.ceil(dispatchDimensions[2] / 4)] });
    };
    const emitSetup = (entryPoint: "mgExtrapolatePhiOneCell" | "mgBakeCoefficients",
      level: number, overrides: Partial<GroupResources>) => emit(entryPoint, level, level, overrides);
    // CM11a Algorithm 1 builds the complete raw phi/V pyramid first. Phi
    // continuation is a separate per-level operation and must never feed the
    // next coarsening step.
    // The sign-aware levels (control.y) are counted from the hierarchy's
    // level 0: every coarsening of the root averages. Its first coarsening
    // (control.z, that level's paper number) keeps the air that averaging
    // would erase (mgAirVanishes).
    const keepAir: readonly [number, number, number, number] = [0, 0, this.levels.length - 1, 0];
    for (let level = 0; level + 1 < this.levels.length; level += 1) {
      emit("mgDownsampleTopology", level, level + 1, {
        phiIn: this.levels[level]!.phi[0], phiOut: this.levels[level + 1]!.phi[0],
        volumeOut: this.levels[level + 1]!.volume[0] }, keepAir);
    }
    for (let level = 0; level < this.levels.length; level += 1) {
      emitSetup("mgExtrapolatePhiOneCell", level,
        { phiIn: this.levels[level]!.phi[0], phiOut: this.levels[level]!.phi[1] });
      phi[level] = 1;
      emitSetup("mgBakeCoefficients", level, {
        phiIn: this.levels[level]!.phi[1], coefficientsOut: this.levels[level]!.coefficients,
      });
      if(level===0&&this.listed){
        emit("mgBuildSmoothTiles",0);
        // One thread per tile, reading the 27 classification words its
        // neighbourhood wrote. Coefficients are immutable for the solve, so
        // both lists are built once and shared by every cycle.
        emit("mgBuildCycleTiles",0,0,{},[0,0,0,0],[4*Math.ceil(this.rootTiles/64),1,1]);
      }
    }
    return result;
  }

  /**
   * The uniform bytes a dispatch reads, and a key for them. Dimensions,
   * level and control repeat across the setup, so it needs only a few
   * distinct buffers.
   */
  private parameterBuffer(level: readonly [number, number, number], coarse: readonly [number, number, number],
    activeLevel: number, control: readonly [number, number, number, number],
    texturesByBinding: readonly (GPUTexture|undefined)[]): [GPUBuffer, string] {
    const bytes = new ArrayBuffer(352); const u = new Uint32Array(bytes); const f = new Float32Array(bytes);
    u.set(this.finestSize.map((n) => n + 2), 0); u.set(level, 4); u.set(coarse, 8);
    // Each axis has coarsened by however many times *it* was halved, which is
    // no longer one shared 2**levelIndex once a hierarchy is semi-coarsened.
    // Reading the factor back off the lattice keeps the two in step by
    // construction, and reproduces 2**levelIndex exactly when they are equal.
    f.set(this.spacing.map((value, axis) =>
      value * (this.finestSize[axis]! / (level[axis]! - 2))), 12);
    u.set(control, 16);
    u[3] = activeLevel;
    texturesByBinding.forEach((texture,binding)=>{
      if(texture){u.set(this.logicalDimensions.get(texture)!,20+4*binding);u[23+4*binding]=this.scratchFields.scratchMetadata(texture);}
    });
    const key = `${Array.from(u.subarray(20)).join(",")}|${level.join(",")}|${coarse.join(",")}|${activeLevel}|${control.join(",")}`;
    const cached = this.paramCache.get(key);
    if (cached) return [cached, key];
    const buffer = this.device.createBuffer({ label: "Uniform CM11a dispatch parameters", size: bytes.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buffer, 0, bytes); this.ownedParams.push(buffer);
    this.paramCache.set(key, buffer);
    return [buffer, key];
  }

  destroy(): void { if (this.isDestroyed) return; this.isDestroyed = true;
    this.pressureTexture.destroy();
    for (const level of this.levels) for (const pair of [level.pressure, level.rhs, level.phi,
      level.volume, level.residual, level.minimum]) {
      for (const texture of new Set(pair)) texture.destroy();
    }
    for (const level of this.levels) level.coefficients.destroy();
    this.smoothTiles?.destroy(); this.cycleTiles?.destroy();
    for (const buffer of this.ownedParams) buffer.destroy();
    this.ownedParams.length = 0; this.ownedGroups.length = 0; this.pipelines = undefined;
    this.viewCache.clear(); this.paramCache.clear(); this.groupCache.clear(); this.textureIds.clear();
  }
  private assertLive(): void { if (this.isDestroyed) throw new Error("Uniform CM11a hierarchy is destroyed"); }
}
