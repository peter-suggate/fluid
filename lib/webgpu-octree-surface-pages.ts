/**
 * Leaf-attached narrow-band level-set pages for the unified octree solver.
 *
 * This module is intentionally independent of the current dense octree
 * projection.  Its allocation is O(compact leaves + resident surface pages),
 * never O(the bounding-box volume).  An upstream topology pass publishes a
 * GPU candidate list containing finest interface leaves and its velocity-swept
 * halo; this owner allocates pages, transports phi and redistances over GPU
 * indirect worklists without a CPU count readback.
 */

export const OCTREE_SURFACE_INVALID = 0xffff_ffff;
/** Canonical page ABI shared by every topology, transport, and render consumer. */
export const OCTREE_SURFACE_PAGE_RESOLUTION = 2 as const;
/**
 * Unified octree page reserve relative to the compact pressure-row arena.
 * Direct paged topology publishes a wider core/halo set than the generic
 * standalone transported band; dam-break cold start currently uses ~37%.
 */
export const OCTREE_UNIFIED_SURFACE_RESIDENT_FRACTION = 0.40;
export const OCTREE_SURFACE_LEAF_RECORD_BYTES = 64;
export const OCTREE_SURFACE_CANDIDATE_BYTES = 8;
export const OCTREE_SURFACE_CONTROL_WORDS = 16;
export const OCTREE_SURFACE_MAX_BACKTRACE_SEGMENTS = 16;
/** Authority bit raised when any traced sample leaves resident sparse support. */
export const OCTREE_SURFACE_DEPARTURE_OUTSIDE_BAND = 1 << 6;

export const OCTREE_SURFACE_STATE = Object.freeze({
  resident: 1 << 0,
  core: 1 << 1,
  halo: 1 << 2,
  desired: 1 << 3,
  activated: 1 << 4,
  /** Leaf record is live even when it has no resident fine page. */
  live: 1 << 5,
} as const);

export interface OctreeSurfacePageOptions {
  pageResolution?: 2 | 4;
  maximumResidentFraction?: number;
  maximumPages?: number;
  maximumArenaBytes?: number;
  /** Finest-cell Manhattan radius indexed on the air side of core pages. */
  airHaloCells?: number;
  /** Maximum adaptive-velocity samples in one page-sample backtrace. */
  maximumSegments?: number;
}

export interface OctreeSurfaceDensePublication {
  /** Transitional topology input. Only resident page owners are updated. */
  texture: GPUTexture;
  dimensions: readonly [number, number, number];
}

export interface OctreeSurfacePagePlan {
  leafCapacity: number;
  requestedPageCapacity: number;
  pageCapacity: number;
  degraded: boolean;
  pageResolution: 2 | 4;
  samplesPerPage: number;
  hashCapacity: number;
  controlOffsetWords: number;
  pageTableOffsetWords: number;
  statesOffsetWords: number;
  freeListOffsetWords: number;
  pageOwnersOffsetWords: number;
  activeOffsetWords: number;
  hashOffsetWords: number;
  airHashOffsetWords: number;
  airHashCapacity: number;
  airHaloCells: number;
  maximumSegments: number;
  phiAOffsetWords: number;
  phiBOffsetWords: number;
  seedOffsetWords: number;
  allocatedWords: number;
  arenaBytes: number;
  /** Arena plus the 128-byte uniform and 12-byte indirect dispatch. */
  allocatedBytes: number;
  denseEquivalentBytes: number;
}

export interface OctreeSurfaceCandidateSource {
  /** Array of { leafRow, flags }; flags must contain core or halo. */
  candidates: GPUBuffer;
  /**
   * word 0=count, words 1..3=dispatch xyz. Adapter-backed sources additionally
   * publish [generation, valid, error, capacity] in words 4..7 so recurring
   * topology consumers can distinguish valid-empty from failed-empty.
   */
  countAndDispatch: GPUBuffer;
  indirectOffsetBytes?: number;
}

export interface OctreeSurfacePageResources {
  /** Array of 64-byte SurfaceLeaf records matching the full-width WGSL ABI below. */
  leaves: GPUBuffer;
  /** Previous compact-row generation, including the published spatial page slot in `pad`. */
  previousLeaves?: GPUBuffer;
  candidates: OctreeSurfaceCandidateSource;
}

export interface OctreeSurfacePageSource {
  plan: OctreeSurfacePagePlan;
  arena: GPUBufferBinding;
  leaves: GPUBufferBinding;
  params: GPUBufferBinding;
  /** word 6 is active count; word 3 is overflow (zero means authoritative). */
  control: { buffer: GPUBuffer; wordOffset: 0; overflowWord: 3; generationWord: 7 };
  /** Header is count, indirect x/y/z, followed by compact leaf rows. */
  activePages: { buffer: GPUBuffer; offsetBytes: number; entriesOffsetBytes: number; indirectBuffer: GPUBuffer; indirectOffsetBytes: 0 };
  phiAOffsetBytes: number;
  pageTableOffsetBytes: number;
}

export interface OctreeSurfacePageDiagnostics {
  pageCapacity: number;
  activePages: number;
  candidatePages: number;
  allocatedPages: number;
  freePages: number;
  overflow: boolean;
  overflowCode: number;
  referenceVolumeCells: number;
  transportedVolumeCells: number;
  interfacePages: number;
  correctionShiftCells: number;
  /** Detected trajectories that left resident support; any nonzero value is non-authoritative. */
  departureOutsideResidentBand: number;
  finestResidentPages: number;
  coarseResidentPages: number;
  maximumResidentLeafSize: number;
  /** Raw adapter publication, staged separately from page-arena lifecycle words. */
  adapterCandidateRows: number;
  adapterDispatchX: number;
}

/**
 * Validates the host-visible portion of the sparse page ABI.  Keep this at
 * every binding seam: a valid GPUBuffer can otherwise hide a 2^3/4^3 shape
 * mismatch until a shader quietly takes its non-authoritative fallback.
 */
export function validateOctreeSurfacePagePlan(plan: OctreeSurfacePagePlan): void {
  if (plan.pageResolution !== 2 && plan.pageResolution !== 4) {
    throw new RangeError("Octree surface page resolution must be 2 or 4");
  }
  if (plan.samplesPerPage !== plan.pageResolution ** 3) {
    throw new RangeError("Octree surface samplesPerPage must equal pageResolution cubed");
  }
  if (!Number.isSafeInteger(plan.maximumSegments)
    || plan.maximumSegments < 1
    || plan.maximumSegments > OCTREE_SURFACE_MAX_BACKTRACE_SEGMENTS) {
    throw new RangeError(`Octree surface maximum segments must be an integer in [1, ${OCTREE_SURFACE_MAX_BACKTRACE_SEGMENTS}]`);
  }
  positiveInteger(plan.leafCapacity, "Octree surface leaf capacity");
  positiveInteger(plan.pageCapacity, "Octree surface page capacity");
  if (plan.pageCapacity > plan.leafCapacity) {
    throw new RangeError("Octree surface page capacity cannot exceed leaf capacity");
  }
  if (plan.phiBOffsetWords - plan.phiAOffsetWords !== plan.pageCapacity * plan.samplesPerPage
    || plan.seedOffsetWords - plan.phiBOffsetWords !== plan.pageCapacity * plan.samplesPerPage) {
    throw new RangeError("Octree surface phi arena offsets do not match the page shape");
  }
}

export function validateOctreeSurfacePageSource(source: OctreeSurfacePageSource): void {
  validateOctreeSurfacePagePlan(source.plan);
  const requireBuffer = (binding: GPUBufferBinding, label: string) => {
    if (!binding || !("buffer" in binding) || !binding.buffer) {
      throw new RangeError(`Octree surface ${label} binding is missing`);
    }
  };
  requireBuffer(source.arena, "arena");
  requireBuffer(source.leaves, "leaf");
  requireBuffer(source.params, "parameter");
  if (source.phiAOffsetBytes !== source.plan.phiAOffsetWords * 4
    || source.pageTableOffsetBytes !== source.plan.pageTableOffsetWords * 4) {
    throw new RangeError("Octree surface source byte offsets do not match its plan");
  }
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

function nextPowerOfTwo(value: number): number {
  let result = 1;
  while (result < value) result *= 2;
  return result;
}

function wordsForCapacity(leafCapacity: number, pageCapacity: number, samplesPerPage: number) {
  // The hash is also the direct topology sampler's all-leaf lookup. Keeping it
  // at <= 50% load gives every finest-cell query a bounded dyadic probe even
  // when the leaf has no resident fine page and uses its affine fallback.
  const hashCapacity = nextPowerOfTwo(Math.max(2, leafCapacity * 2));
  // The physical band has heavy overlap along a surface, so eight exact-key
  // slots per possible resident row is conservative at scene scale. Keep a
  // 64-slot floor so tiny arenas can still represent one radius-four stencil.
  const airHashCapacity = nextPowerOfTwo(Math.max(64, pageCapacity * 8));
  const pageTableOffsetWords = OCTREE_SURFACE_CONTROL_WORDS;
  const statesOffsetWords = pageTableOffsetWords + leafCapacity;
  const freeListOffsetWords = statesOffsetWords + leafCapacity;
  const pageOwnersOffsetWords = freeListOffsetWords + pageCapacity;
  const activeOffsetWords = pageOwnersOffsetWords + pageCapacity;
  const hashOffsetWords = activeOffsetWords + 4 + pageCapacity;
  const airHashOffsetWords = hashOffsetWords + hashCapacity;
  const phiAOffsetWords = airHashOffsetWords + 2 * airHashCapacity;
  const phiBOffsetWords = phiAOffsetWords + pageCapacity * samplesPerPage;
  const seedOffsetWords = phiBOffsetWords + pageCapacity * samplesPerPage;
  const allocatedWords = seedOffsetWords + pageCapacity * samplesPerPage;
  return {
    hashCapacity, airHashCapacity, pageTableOffsetWords, statesOffsetWords, freeListOffsetWords,
    pageOwnersOffsetWords, activeOffsetWords, hashOffsetWords, phiAOffsetWords,
    airHashOffsetWords, phiBOffsetWords, seedOffsetWords, allocatedWords,
  };
}

/** Exact persistent allocation plan. `denseEquivalentBytes` is phi A/B/seed. */
export function planOctreeSurfacePages(
  leafCapacityValue: number,
  denseDimensions: readonly [number, number, number],
  options: OctreeSurfacePageOptions = {},
): OctreeSurfacePagePlan {
  const leafCapacity = positiveInteger(leafCapacityValue, "Octree surface leaf capacity");
  denseDimensions.forEach((value, axis) => positiveInteger(value, `Dense comparison dimension ${axis}`));
  const pageResolution = options.pageResolution ?? OCTREE_SURFACE_PAGE_RESOLUTION;
  if (pageResolution !== 2 && pageResolution !== 4) throw new RangeError("Octree surface page resolution must be 2 or 4");
  const samplesPerPage = pageResolution ** 3;
  const airHaloCells = options.airHaloCells ?? 4;
  if (!Number.isSafeInteger(airHaloCells) || airHaloCells < 1 || airHaloCells > 8) {
    throw new RangeError("Octree surface air halo must be an integer in [1, 8]");
  }
  const maximumSegments = options.maximumSegments ?? 8;
  if (!Number.isSafeInteger(maximumSegments)
    || maximumSegments < 1
    || maximumSegments > OCTREE_SURFACE_MAX_BACKTRACE_SEGMENTS) {
    throw new RangeError(`Octree surface maximum segments must be an integer in [1, ${OCTREE_SURFACE_MAX_BACKTRACE_SEGMENTS}]`);
  }
  const fraction = options.maximumResidentFraction ?? 0.25;
  if (!Number.isFinite(fraction) || fraction <= 0 || fraction > 1) throw new RangeError("Octree surface resident fraction must be in (0, 1]");
  const fractionCapacity = Math.max(1, Math.ceil(leafCapacity * fraction));
  const hardCapacity = options.maximumPages === undefined
    ? leafCapacity
    : Math.max(1, Math.min(leafCapacity, Math.floor(options.maximumPages)));
  if (!Number.isFinite(hardCapacity)) throw new RangeError("Octree surface maximum pages must be finite");
  const requestedPageCapacity = Math.min(fractionCapacity, hardCapacity);
  let pageCapacity = requestedPageCapacity;
  if (options.maximumArenaBytes !== undefined) {
    if (!Number.isFinite(options.maximumArenaBytes) || options.maximumArenaBytes < 4) {
      throw new RangeError("Octree surface arena byte ceiling must be finite and positive");
    }
    while (pageCapacity > 0 && wordsForCapacity(leafCapacity, pageCapacity, samplesPerPage).allocatedWords * 4 > options.maximumArenaBytes) pageCapacity -= 1;
    if (pageCapacity < 1) throw new RangeError("Octree surface arena byte ceiling cannot hold one page");
  }
  const offsets = wordsForCapacity(leafCapacity, pageCapacity, samplesPerPage);
  const plan = {
    leafCapacity, requestedPageCapacity, pageCapacity,
    degraded: pageCapacity < requestedPageCapacity,
    pageResolution, airHaloCells, maximumSegments,
    samplesPerPage, ...offsets,
    controlOffsetWords: 0,
    arenaBytes: offsets.allocatedWords * 4,
    allocatedBytes: offsets.allocatedWords * 4 + 128 + 12,
    denseEquivalentBytes: denseDimensions[0] * denseDimensions[1] * denseDimensions[2] * 3 * 4,
  };
  validateOctreeSurfacePagePlan(plan);
  return plan;
}

/** Bounded segment planner shared by CPU oracles and the WGSL transport. */
export function planOctreeSurfaceBacktraceSegments(
  speed: number,
  dt: number,
  pageCellSize: number,
  maximumSegments = 8,
): number {
  if (!Number.isSafeInteger(maximumSegments)
    || maximumSegments < 1
    || maximumSegments > OCTREE_SURFACE_MAX_BACKTRACE_SEGMENTS) {
    throw new RangeError(`Octree surface maximum segments must be an integer in [1, ${OCTREE_SURFACE_MAX_BACKTRACE_SEGMENTS}]`);
  }
  const boundedSpeed = Number.isFinite(speed) ? Math.max(0, speed) : 0;
  const boundedDt = Number.isFinite(dt) ? Math.max(0, dt) : 0;
  const h = Number.isFinite(pageCellSize) ? Math.max(1e-9, pageCellSize) : 1e-9;
  return Math.max(1, Math.min(maximumSegments, Math.ceil(boundedSpeed * boundedDt / h)));
}

export interface OctreeSurfaceBacktraceResult {
  departure: readonly [number, number, number];
  segments: number;
  velocitySamples: number;
}

/**
 * CPU oracle for the shader's piecewise-Euler sparse trajectory integration.
 * Segment count is fixed from the local initial speed, while velocity is
 * resampled at every segment endpoint.
 */
export function traceOctreeSurfaceBacktrace(
  position: readonly [number, number, number],
  sampleVelocity: (point: readonly [number, number, number]) => readonly [number, number, number],
  dt: number,
  pageCellSize: number,
  maximumSegments = 8,
): OctreeSurfaceBacktraceResult {
  if (!position.every(Number.isFinite)) throw new RangeError("Octree surface backtrace position must be finite");
  const step = Number.isFinite(dt) ? Math.max(0, dt) : 0;
  let velocity = sampleVelocity(position);
  if (!velocity.every(Number.isFinite)) throw new RangeError("Octree surface backtrace velocity must be finite");
  const segments = planOctreeSurfaceBacktraceSegments(Math.hypot(...velocity), step, pageCellSize, maximumSegments);
  const segmentDt = step / segments;
  const departure: [number, number, number] = [...position];
  for (let segment = 0; segment < segments; segment += 1) {
    departure[0] -= velocity[0] * segmentDt;
    departure[1] -= velocity[1] * segmentDt;
    departure[2] -= velocity[2] * segmentDt;
    if (segment + 1 < segments) {
      velocity = sampleVelocity(departure);
      if (!velocity.every(Number.isFinite)) throw new RangeError("Octree surface backtrace velocity must be finite");
    }
  }
  return { departure, segments, velocitySamples: segments };
}

/** Conservative cell halo for a backtrace plus interpolation/redistance. */
export function requiredOctreeSurfaceHaloCells(
  maximumSpeed: number,
  dt: number,
  finestCellSize: number,
  interpolationRadius = 2,
  redistanceRadius = 2,
): number {
  const speed = Number.isFinite(maximumSpeed) ? Math.max(0, maximumSpeed) : 0;
  const step = Number.isFinite(dt) ? Math.max(0, dt) : 0;
  const h = Number.isFinite(finestCellSize) ? Math.max(1e-9, finestCellSize) : 1e-9;
  return Math.ceil(speed * step / h) + Math.max(0, Math.ceil(interpolationRadius)) + Math.max(0, Math.ceil(redistanceRadius));
}

export interface OctreeSurfaceLeafOracle {
  row: number;
  origin: readonly [number, number, number];
  size: number;
  finest: boolean;
  phiMin: number;
  phiMax: number;
}

/** CPU oracle for the GPU topology producer's core/swept-halo candidate list. */
export function selectOctreeSurfaceCandidates(
  leaves: readonly OctreeSurfaceLeafOracle[],
  haloCells: number,
): ReadonlyArray<{ row: number; flags: number }> {
  const halo = Math.max(0, Math.ceil(haloCells));
  const core = leaves.filter((leaf) => leaf.finest && leaf.phiMin <= 0 && leaf.phiMax >= 0);
  const distance = (a: OctreeSurfaceLeafOracle, b: OctreeSurfaceLeafOracle) => Math.max(...[0, 1, 2].map((axis) => {
    const a1 = a.origin[axis] + a.size, b1 = b.origin[axis] + b.size;
    return Math.max(0, b.origin[axis] - a1, a.origin[axis] - b1);
  }));
  return leaves.filter((leaf) => leaf.finest).flatMap((leaf) => {
    if (core.includes(leaf)) return [{ row: leaf.row, flags: OCTREE_SURFACE_STATE.core }];
    return core.some((interfaceLeaf) => distance(leaf, interfaceLeaf) <= halo)
      ? [{ row: leaf.row, flags: OCTREE_SURFACE_STATE.halo }]
      : [];
  });
}

function createBuffer(device: GPUDevice, label: string, size: number, usage: GPUBufferUsageFlags, data?: ArrayBufferView<ArrayBuffer>) {
  const result = device.createBuffer({ label, size: Math.max(4, size), usage });
  if (data?.byteLength) device.queue.writeBuffer(result, 0, data);
  return result;
}

export class WebGPUOctreeSurfacePages {
  readonly plan: OctreeSurfacePagePlan;
  readonly allocatedBytes: number;
  readonly arena: GPUBuffer;
  private readonly dispatch: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly leaves: GPUBuffer;
  private readonly candidateSource: OctreeSurfaceCandidateSource;
  private readonly bindGroup: GPUBindGroup;
  private readonly previousLeavesFallback?: GPUBuffer;
  private densePublicationBindGroup?: GPUBindGroup;
  private readonly densePublicationPipeline?: GPUComputePipeline;
  private readonly pipelines: Record<"reset" | "mark" | "retire" | "allocate" | "publish" | "prepare" | "initialize" | "reduceReference" | "transport" | "copy" | "seed" | "redistanceAB" | "redistanceBA" | "reduceCurrent" | "correctVolume", GPUComputePipeline>;
  private generation = 0;
  private destroyed = false;
  private lastDiagnostics: OctreeSurfacePageDiagnostics = { pageCapacity: 0, activePages: 0, candidatePages: 0, allocatedPages: 0, freePages: 0, overflow: false, overflowCode: 0, referenceVolumeCells: 0, transportedVolumeCells: 0, interfacePages: 0, correctionShiftCells: 0, departureOutsideResidentBand: 0, finestResidentPages: 0, coarseResidentPages: 0, maximumResidentLeafSize: 0, adapterCandidateRows: 0, adapterDispatchX: 0 };

  constructor(
    private readonly device: GPUDevice,
    resources: OctreeSurfacePageResources,
    leafCapacity: number,
    private readonly denseDimensions: readonly [number, number, number],
    private readonly cellSize: readonly [number, number, number],
    options: OctreeSurfacePageOptions = {},
    densePublication?: OctreeSurfaceDensePublication,
  ) {
    this.leaves = resources.leaves;
    this.candidateSource = resources.candidates;
    const maximumArenaBytes = Math.min(
      options.maximumArenaBytes ?? Number.POSITIVE_INFINITY,
      Number(device.limits.maxStorageBufferBindingSize), Number(device.limits.maxBufferSize),
    );
    this.plan = planOctreeSurfacePages(leafCapacity, denseDimensions, { ...options, maximumArenaBytes });
    this.allocatedBytes = this.plan.allocatedBytes;
    const initial = new Uint32Array(this.plan.allocatedWords);
    initial.fill(OCTREE_SURFACE_INVALID, this.plan.pageTableOffsetWords, this.plan.pageTableOffsetWords + this.plan.leafCapacity);
    initial.fill(OCTREE_SURFACE_INVALID, this.plan.pageOwnersOffsetWords, this.plan.pageOwnersOffsetWords + this.plan.pageCapacity);
    initial.fill(OCTREE_SURFACE_INVALID, this.plan.hashOffsetWords, this.plan.hashOffsetWords + this.plan.hashCapacity);
    for (let slot = 0; slot < this.plan.pageCapacity; slot += 1) initial[this.plan.freeListOffsetWords + slot] = slot;
    initial[0] = this.plan.pageCapacity;
    // Words 8-10 are live hierarchy diagnostics. Leave them zero until the
    // first publication pass so an initializing UI never reports capacities
    // as page counts or a hash size as a leaf scale.
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.arena = createBuffer(device, "Octree surface page arena", initial.byteLength, storage, initial);
    this.dispatch = createBuffer(device, "Octree surface page indirect dispatch", 12, GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT);
    this.params = createBuffer(device, "Octree surface page parameters", 128, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const layout = device.createBindGroupLayout({ label: "Octree surface page layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ] });
    const shaderModule = device.createShaderModule({ label: "Octree surface pages", code: octreeSurfacePageShader });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipeline = (entryPoint: string) => device.createComputePipeline({ label: `Octree surface ${entryPoint}`, layout: pipelineLayout, compute: { module: shaderModule, entryPoint } });
    this.pipelines = {
      reset: pipeline("resetLeaves"), mark: pipeline("markCandidates"), retire: pipeline("retirePages"),
      allocate: pipeline("allocatePages"), publish: pipeline("publishPages"), prepare: pipeline("prepareDispatch"),
      initialize: pipeline("initializeActivatedPages"), reduceReference: pipeline("reduceReferenceVolume"),
      transport: pipeline("transportPhi"), copy: pipeline("copyPhiBToA"), seed: pipeline("seedRedistance"),
      redistanceAB: pipeline("redistanceAToB"), redistanceBA: pipeline("redistanceBToA"),
      reduceCurrent: pipeline("reduceCurrentVolume"), correctVolume: pipeline("correctPageVolume"),
    };
    this.previousLeavesFallback = resources.previousLeaves ? undefined : createBuffer(device,
      "Empty previous octree surface leaf", OCTREE_SURFACE_LEAF_RECORD_BYTES, GPUBufferUsage.STORAGE);
    this.bindGroup = device.createBindGroup({ label: "Octree surface page bindings", layout, entries: [
      { binding: 0, resource: { buffer: resources.leaves } },
      { binding: 1, resource: { buffer: resources.candidates.candidates } },
      { binding: 2, resource: { buffer: resources.candidates.countAndDispatch } },
      { binding: 3, resource: { buffer: this.arena } },
      { binding: 4, resource: { buffer: this.params } },
      // Without a preserved generation, bind the already-read-only candidate
      // arena as a robust-zero fallback. Aliasing `leaves` here would make the
      // same buffer read-write and read-only in one synchronization scope.
      { binding: 5, resource: { buffer: resources.previousLeaves ?? this.previousLeavesFallback! } },
    ] });
    if (densePublication) {
      if (densePublication.dimensions.some((value) => !Number.isSafeInteger(value) || value < 1)) {
        throw new RangeError("Octree surface publication dimensions must be positive integers");
      }
      const publicationLayout = device.createBindGroupLayout({ label: "Octree surface dense publication layout", entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      ] });
      const publicationModule = device.createShaderModule({ label: "Octree surface dense publication", code: octreeSurfaceDensePublicationShader });
      this.densePublicationPipeline = device.createComputePipeline({
        label: "Publish octree surface pages to topology phi",
        layout: device.createPipelineLayout({ bindGroupLayouts: [publicationLayout] }),
        compute: { module: publicationModule, entryPoint: "publishDensePhi" },
      });
      this.densePublicationBindGroup = device.createBindGroup({ label: "Octree surface dense publication bindings", layout: publicationLayout, entries: [
        { binding: 0, resource: { buffer: this.leaves } },
        { binding: 1, resource: { buffer: this.arena } },
        { binding: 2, resource: { buffer: this.params } },
        { binding: 3, resource: densePublication.texture.createView() },
      ] });
    }
    this.writeParams(0);
  }

  private writeParams(dt: number) {
    const data = new ArrayBuffer(128), u = new Uint32Array(data), f = new Float32Array(data), p = this.plan;
    u.set([p.leafCapacity, p.pageCapacity, p.pageResolution, p.samplesPerPage], 0);
    u.set([p.pageTableOffsetWords, p.statesOffsetWords, p.freeListOffsetWords, p.pageOwnersOffsetWords], 4);
    u.set([p.activeOffsetWords, p.hashOffsetWords, p.phiAOffsetWords, p.phiBOffsetWords], 8);
    u.set([p.seedOffsetWords, p.hashCapacity, this.generation, 0], 12);
    u.set([p.airHashOffsetWords,p.airHashCapacity,p.airHaloCells,p.maximumSegments],20);
    u.set(this.denseDimensions,24);
    f.set([this.cellSize[0], this.cellSize[1], this.cellSize[2], Number.isFinite(dt) ? Math.max(0, dt) : 0], 16);
    this.device.queue.writeBuffer(this.params, 0, data);
  }

  encodeLifecycle(encoder: GPUCommandEncoder): void {
    if (this.destroyed) return;
    this.generation += 1; this.writeParams(0);
    const stage = (label: string, pipeline: GPUComputePipeline, encodeDispatch: (pass: GPUComputePassEncoder) => void) => {
      const pass = encoder.beginComputePass({ label });
      pass.setBindGroup(0, this.bindGroup); pass.setPipeline(pipeline); encodeDispatch(pass); pass.end();
    };
    // Separate passes are required here: each stage consumes globally visible
    // atomics from the prior stage, and a workgroup barrier cannot synchronize
    // independent dispatches.
    stage("Reset octree surface pages", this.pipelines.reset, (pass) => pass.dispatchWorkgroups(Math.ceil(this.plan.leafCapacity / 64)));
    // Reset consumes the previous generation's spatial hash to migrate page
    // slots across compact-row reordering. Reuse the same tables for the new
    // generation only after that pass has finished.
    encoder.clearBuffer(this.arena, this.plan.hashOffsetWords * 4, this.plan.hashCapacity * 4);
    encoder.clearBuffer(this.arena, this.plan.airHashOffsetWords * 4, this.plan.airHashCapacity * 8);
    // candidateControl is also storage binding 2. WebGPU forbids using one
    // buffer as STORAGE and INDIRECT in the same pass, which rejects the whole
    // encoder on Dawn. Launch the capacity-bounded domain directly; both
    // kernels already guard every invocation with candidateControl[0].
    const candidateWorkgroups = Math.ceil(this.plan.leafCapacity / 64);
    stage("Mark octree surface candidates", this.pipelines.mark, (pass) => pass.dispatchWorkgroups(candidateWorkgroups));
    stage("Retire octree surface pages", this.pipelines.retire, (pass) => pass.dispatchWorkgroups(Math.ceil(Math.max(this.plan.leafCapacity, this.plan.pageCapacity) / 64)));
    stage("Allocate octree surface pages", this.pipelines.allocate, (pass) => pass.dispatchWorkgroups(candidateWorkgroups));
    stage("Publish octree surface pages", this.pipelines.publish, (pass) => pass.dispatchWorkgroups(Math.ceil(this.plan.leafCapacity / 64)));
    stage("Prepare octree surface dispatch", this.pipelines.prepare, (pass) => pass.dispatchWorkgroups(1));
    encoder.copyBufferToBuffer(this.arena, (this.plan.activeOffsetWords + 1) * 4, this.dispatch, 0, 12);
  }

  encodeTransport(encoder: GPUCommandEncoder, dt: number): void {
    if (this.destroyed) return;
    this.writeParams(dt);
    encoder.clearBuffer(this.arena, 13 * 4, 3 * 4);
    const initialize = encoder.beginComputePass({ label: "Initialize octree surface pages" });
    initialize.setBindGroup(0, this.bindGroup);
    initialize.setPipeline(this.pipelines.initialize); initialize.dispatchWorkgroupsIndirect(this.dispatch, 0);
    initialize.end();
    const reference = encoder.beginComputePass({ label: "Measure octree surface reference volume" });
    reference.setBindGroup(0, this.bindGroup); reference.setPipeline(this.pipelines.reduceReference); reference.dispatchWorkgroupsIndirect(this.dispatch, 0); reference.end();
    const pass = encoder.beginComputePass({ label: "Transport octree surface pages" });
    pass.setBindGroup(0, this.bindGroup);
    pass.setPipeline(this.pipelines.transport); pass.dispatchWorkgroupsIndirect(this.dispatch, 0);
    pass.end();
    const copy = encoder.beginComputePass({ label: "Commit transported octree surface pages" });
    copy.setBindGroup(0, this.bindGroup); copy.setPipeline(this.pipelines.copy); copy.dispatchWorkgroupsIndirect(this.dispatch, 0); copy.end();
  }

  encodeRedistance(encoder: GPUCommandEncoder, iterations = 4): void {
    if (this.destroyed) return;
    const count = Math.max(2, Math.min(16, Math.ceil(iterations / 2) * 2));
    const dispatch = (label: string, pipeline: GPUComputePipeline) => {
      const pass = encoder.beginComputePass({ label }); pass.setBindGroup(0, this.bindGroup);
      pass.setPipeline(pipeline); pass.dispatchWorkgroupsIndirect(this.dispatch, 0); pass.end();
    };
    dispatch("Seed octree surface redistance", this.pipelines.seed);
    for (let iteration = 0; iteration < count; iteration += 1) dispatch(`Redistance octree surface pages ${iteration + 1}`, iteration % 2 === 0 ? this.pipelines.redistanceAB : this.pipelines.redistanceBA);
  }

  /** Preserve the active-page smooth volume measured immediately before transport. */
  encodeVolumeCorrection(encoder: GPUCommandEncoder): void {
    if (this.destroyed) return;
    // Re-reduce after each bounded shift. Three fixed GPU rounds converge
    // cells that cross the smooth-Heaviside shoulder without a CPU readback.
    for (let iteration = 0; iteration < 3; iteration += 1) {
      encoder.clearBuffer(this.arena, 14 * 4, 2 * 4);
      const reduce = encoder.beginComputePass({ label: `Conserve octree surface page volume ${iteration + 1}` });
      reduce.setBindGroup(0, this.bindGroup);
      reduce.setPipeline(this.pipelines.reduceCurrent); reduce.dispatchWorkgroupsIndirect(this.dispatch, 0);
      reduce.end();
      const correct = encoder.beginComputePass({ label: `Correct octree surface page volume ${iteration + 1}` });
      correct.setBindGroup(0, this.bindGroup);
      correct.setPipeline(this.pipelines.correctVolume); correct.dispatchWorkgroupsIndirect(this.dispatch, 0);
      correct.end();
    }
  }

  /**
   * Transitional publication for the still-dense topology classifier. It
   * touches one finest cell per active page and never reads a velocity texture.
   */
  encodeDensePublication(encoder: GPUCommandEncoder): boolean {
    if (this.destroyed || !this.densePublicationPipeline || !this.densePublicationBindGroup) return false;
    const pass = encoder.beginComputePass({ label: "Publish octree surface pages to topology phi" });
    pass.setPipeline(this.densePublicationPipeline);
    pass.setBindGroup(0, this.densePublicationBindGroup);
    pass.dispatchWorkgroupsIndirect(this.dispatch, 0);
    pass.end();
    return true;
  }

  /** Drop the one-shot dense target after its bootstrap submission. */
  releaseDensePublicationBinding(): boolean {
    if (!this.densePublicationBindGroup) return false;
    this.densePublicationBindGroup = undefined;
    return true;
  }

  get source(): OctreeSurfacePageSource {
    const p = this.plan;
    return {
      plan: p, arena: { buffer: this.arena }, leaves: { buffer: this.leaves },
      params: { buffer: this.params }, control: { buffer: this.arena, wordOffset: 0, overflowWord: 3, generationWord: 7 },
      activePages: { buffer: this.arena, offsetBytes: p.activeOffsetWords * 4, entriesOffsetBytes: (p.activeOffsetWords + 4) * 4, indirectBuffer: this.dispatch, indirectOffsetBytes: 0 },
      phiAOffsetBytes: p.phiAOffsetWords * 4, pageTableOffsetBytes: p.pageTableOffsetWords * 4,
    };
  }

  async readDiagnostics(): Promise<OctreeSurfacePageDiagnostics> {
    if (this.destroyed) return this.lastDiagnostics;
    const readback = this.device.createBuffer({ label: "Octree surface page diagnostics", size: 72, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Read octree surface page diagnostics" });
    encoder.copyBufferToBuffer(this.arena, 0, readback, 0, 64);
    // Preserve all arena words and stage the adapter header in dedicated QA
    // words. Diagnostics never feed simulation publication decisions.
    encoder.copyBufferToBuffer(this.candidateSource.countAndDispatch, 0, readback, 64, 8);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const reference = words[13] / 4096, transported = words[14] / 4096, interfacePages = words[15];
      const band = 4 * this.cellSize[1];
      const shift = interfacePages > 0
        ? Math.max(-1.5 * Math.min(...this.cellSize), Math.min(1.5 * Math.min(...this.cellSize), (transported - reference) * band / interfacePages))
        : 0;
      this.lastDiagnostics = {
        pageCapacity: this.plan.pageCapacity, activePages: words[6], candidatePages: words[12], allocatedPages: words[4], freePages: words[0], overflow: words[3] !== 0, overflowCode: words[3],
        referenceVolumeCells: reference, transportedVolumeCells: transported,
        interfacePages, correctionShiftCells: shift / Math.max(1e-9, Math.min(...this.cellSize)),
        departureOutsideResidentBand: words[11],
        finestResidentPages: words[8], coarseResidentPages: words[9], maximumResidentLeafSize: words[10],
        adapterCandidateRows: words[16], adapterDispatchX: words[17],
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
    return this.lastDiagnostics;
  }

  get volumeDiagnostics() { return this.lastDiagnostics; }

  destroy(): void { if (!this.destroyed) { this.destroyed = true; this.arena.destroy(); this.dispatch.destroy(); this.params.destroy(); this.previousLeavesFallback?.destroy(); } }
}

export const octreeSurfacePageShader = /* wgsl */ `
struct SurfaceLeaf { originX:u32,originY:u32,originZ:u32,size:u32,flags:u32,pad0:u32,pad1:u32,pad2:u32,phiGradient:vec4f,motion:vec4f }
struct Candidate { row:u32,flags:u32 }
struct Params { shape:vec4u, offsets0:vec4u, offsets1:vec4u, offsets2:vec4u, cellDt:vec4f, spare0:vec4u, spare1:vec4u, spare2:vec4u }
@group(0) @binding(0) var<storage,read_write> leaves:array<SurfaceLeaf>;
@group(0) @binding(1) var<storage,read> candidates:array<Candidate>;
@group(0) @binding(2) var<storage,read> candidateControl:array<u32>;
@group(0) @binding(3) var<storage,read_write> arena:array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params:Params;
@group(0) @binding(5) var<storage,read> previousLeaves:array<SurfaceLeaf>;
const INVALID=0xffffffffu; const CLAIMED=0x80000000u; const RESIDENT=1u; const CORE=2u; const HALO=4u; const DESIRED=8u; const ACTIVATED=16u; const LIVE=32u; const DEPARTURE_OUTSIDE_BAND=64u;
fn leafOrigin(leaf:SurfaceLeaf)->vec3u{return vec3u(leaf.originX,leaf.originY,leaf.originZ);}
// Exact identity in the existing u32-linear topology domain.  The coordinate
// hash chooses a probe chain; this key distinguishes collisions without
// truncating any axis to ten bits.  +1 reserves zero for an empty slot.
fn airCellKey(p:vec3u)->u32{return p.x+params.spare1.x*(p.y+params.spare1.y*p.z)+1u;}
fn pageTable(row:u32)->u32{return atomicLoad(&arena[params.offsets0.x+row]);}
fn state(row:u32)->u32{return atomicLoad(&arena[params.offsets0.y+row]);}
fn pageWord(offset:u32,slot:u32,local:u32)->u32{return offset+slot*params.shape.w+local;}
fn loadPhi(offset:u32,slot:u32,local:u32)->f32{return bitcast<f32>(atomicLoad(&arena[pageWord(offset,slot,local)]));}
fn storePhi(offset:u32,slot:u32,local:u32,value:f32){atomicStore(&arena[pageWord(offset,slot,local)],bitcast<u32>(value));}
fn hashCoord(q:vec3u)->u32{var h=(q.x*73856093u)^(q.y*19349663u)^(q.z*83492791u);h^=h>>16u;return h;}
fn leafContains(leaf:SurfaceLeaf,p:vec3f)->bool{let o=vec3f(leafOrigin(leaf));return all(p>=o)&&all(p<o+vec3f(f32(leaf.size)));}
fn leafFallback(leaf:SurfaceLeaf,p:vec3f)->f32{let c=vec3f(leafOrigin(leaf))+vec3f(0.5*f32(leaf.size));let physicalGradient=leaf.phiGradient.yzw/max(params.cellDt.xyz,vec3f(1e-9));let boundedPhysical=physicalGradient/max(1.0,length(physicalGradient));let cellGradient=boundedPhysical*params.cellDt.xyz;return leaf.phiGradient.x+dot(cellGradient,p-c);}
fn localIndex(q:vec3u)->u32{return q.x+params.shape.z*(q.y+params.shape.z*q.z);}
fn findLeafRow(p:vec3f,hintRow:u32)->u32{
  if(any(p<vec3f(0.0))||any(p>=vec3f(params.spare1.xyz))){return INVALID;}
  if(hintRow<params.shape.x&&leafContains(leaves[hintRow],p)){return hintRow;}
  let mask=params.offsets2.y-1u;
  // The hash key is origin/leaf-size, so query every representable dyadic
  // level. This keeps coarse/fine crossings sparse and bounded (11*32 probes).
  for(var size=1u;size<=1024u;size<<=1u){
    let q=vec3u(floor(p/f32(size)));var slot=hashCoord(q)&mask;
    for(var probe=0u;probe<32u;probe+=1u){let encoded=atomicLoad(&arena[params.offsets1.y+slot]);if(encoded==0u){break;}let row=encoded-1u;if(row<params.shape.x&&leafContains(leaves[row],p)){return row;}slot=(slot+1u)&mask;}
  }
  return INVALID;
}
fn findResidentRow(p:vec3f,fallbackRow:u32)->u32{let row=findLeafRow(p,fallbackRow);return select(row,fallbackRow,row==INVALID);}
fn residentBandRow(p:vec3f,hintRow:u32)->u32{let row=findLeafRow(p,hintRow);if(row==INVALID||(state(row)&RESIDENT)==0u){return INVALID;}let slot=pageTable(row);return select(INVALID,row,slot<params.shape.y);}
fn samplePage(row:u32,p:vec3f,offset:u32)->f32{
  let leaf=leaves[row];let slot=pageTable(row);if(slot==INVALID||slot>=params.shape.y||!leafContains(leaf,p)){return leafFallback(leaf,p);}
  let origin=vec3f(leafOrigin(leaf));let grid=clamp((p-origin)/f32(leaf.size)*f32(params.shape.z)-vec3f(0.5),vec3f(0.0),vec3f(f32(params.shape.z-1u)));
  let a=vec3u(floor(grid));let b=min(a+vec3u(1u),vec3u(params.shape.z-1u));let t=fract(grid);
  let c000=loadPhi(offset,slot,localIndex(a));let c100=loadPhi(offset,slot,localIndex(vec3u(b.x,a.y,a.z)));let c010=loadPhi(offset,slot,localIndex(vec3u(a.x,b.y,a.z)));let c110=loadPhi(offset,slot,localIndex(vec3u(b.x,b.y,a.z)));
  let c001=loadPhi(offset,slot,localIndex(vec3u(a.x,a.y,b.z)));let c101=loadPhi(offset,slot,localIndex(vec3u(b.x,a.y,b.z)));let c011=loadPhi(offset,slot,localIndex(vec3u(a.x,b.y,b.z)));let c111=loadPhi(offset,slot,localIndex(b));
  return mix(mix(mix(c000,c100,t.x),mix(c010,c110,t.x),t.y),mix(mix(c001,c101,t.x),mix(c011,c111,t.x),t.y),t.z);
}
fn hierarchicalPhi(ownerRow:u32,p:vec3f,offset:u32)->f32{let row=findResidentRow(p,ownerRow);return samplePage(row,p,offset);}
fn invocation(gid:vec3u)->vec3u{let stream=gid.x+gid.y*65535u*256u;let count=atomicLoad(&arena[params.offsets1.x]);if(stream>=count*params.shape.w){return vec3u(INVALID);}return vec3u(atomicLoad(&arena[params.offsets1.x+4u+stream/params.shape.w]),stream%params.shape.w,stream);}
fn localCoord(local:u32)->vec3u{return vec3u(local%params.shape.z,(local/params.shape.z)%params.shape.z,local/(params.shape.z*params.shape.z));}
fn samplePosition(row:u32,local:u32)->vec3f{let leaf=leaves[row];return vec3f(leafOrigin(leaf))+(vec3f(localCoord(local))+vec3f(0.5))*f32(leaf.size)/f32(params.shape.z);}
fn activeRow(item:u32)->u32{let count=atomicLoad(&arena[params.offsets1.x]);if(item>=count){return INVALID;}return atomicLoad(&arena[params.offsets1.x+4u+item]);}
fn centrePhi(row:u32)->f32{let slot=pageTable(row);if(slot==INVALID||slot>=params.shape.y){return 0.0;}let lo=params.shape.z/2u-1u;let hi=params.shape.z/2u;var value=0.0;for(var z=lo;z<=hi;z+=1u){for(var y=lo;y<=hi;y+=1u){for(var x=lo;x<=hi;x+=1u){value+=loadPhi(params.offsets1.z,slot,localIndex(vec3u(x,y,z)));}}}return value*0.125;}
fn smoothVolume(value:f32)->f32{return clamp(0.5-value/max(1e-9,4.0*params.cellDt.y),0.0,1.0);}
fn conservativeAdd(word:u32,value:u32){let old=atomicAdd(&arena[word],value);if(old>0xffffffffu-value){atomicOr(&arena[3],32u);}}
fn previousPageRow(leaf:SurfaceLeaf)->u32{if(leaf.size==0u){return INVALID;}let q=leafOrigin(leaf)/leaf.size;let mask=params.offsets2.y-1u;var at=hashCoord(q)&mask;for(var probe=0u;probe<32u;probe+=1u){let encoded=atomicLoad(&arena[params.offsets1.y+at]);if(encoded==0u){break;}if(encoded!=INVALID){let oldRow=encoded-1u;if(oldRow<params.shape.x&&oldRow<arrayLength(&previousLeaves)){let oldLeaf=previousLeaves[oldRow];if(all(leafOrigin(oldLeaf)==leafOrigin(leaf))&&oldLeaf.size==leaf.size){return oldRow;}}}at=(at+1u)&mask;}return INVALID;}
fn claimPreviousPage(oldRow:u32,newRow:u32,slot:u32)->bool{for(var attempt=0u;attempt<16u;attempt+=1u){let result=atomicCompareExchangeWeak(&arena[params.offsets0.w+slot],oldRow,newRow|CLAIMED);if(result.exchanged){return true;}if(result.old_value!=oldRow){return false;}}return false;}
@compute @workgroup_size(64) fn resetLeaves(@builtin(global_invocation_id) gid:vec3u){let row=gid.x;if(row>=params.shape.x){return;}let current=leaves[row];atomicStore(&arena[params.offsets0.x+row],INVALID);atomicStore(&arena[params.offsets0.y+row],0u);if(current.size!=0u&&(current.flags&LIVE)!=0u){let oldRow=previousPageRow(current);if(oldRow!=INVALID){let slot=previousLeaves[oldRow].pad0;if(slot<params.shape.y&&claimPreviousPage(oldRow,row,slot)){atomicStore(&arena[params.offsets0.x+row],slot);atomicStore(&arena[params.offsets0.y+row],RESIDENT);}}}if(row==0u){atomicStore(&arena[3],0u);atomicStore(&arena[4],0u);atomicStore(&arena[5],0u);atomicStore(&arena[6],0u);atomicStore(&arena[7],params.offsets2.z);atomicStore(&arena[8],0u);atomicStore(&arena[9],0u);atomicStore(&arena[10],0u);atomicStore(&arena[11],0u);atomicStore(&arena[12],candidateControl[0]);atomicStore(&arena[params.offsets1.x],0u);}}
@compute @workgroup_size(64) fn markCandidates(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;if(i>=candidateControl[0]){return;}if(i>=arrayLength(&candidates)){atomicOr(&arena[3],1u);return;}let c=candidates[i];if(c.row>=params.shape.x){atomicOr(&arena[3],1u);return;}atomicOr(&arena[params.offsets0.y+c.row],DESIRED|(c.flags&(CORE|HALO)));}
fn pushFree(slot:u32){let free=atomicAdd(&arena[0],1u);if(free<params.shape.y){atomicStore(&arena[params.offsets0.z+free],slot);atomicAdd(&arena[5],1u);}else{atomicOr(&arena[3],2u);}}
fn releaseClaim(slot:u32,claimed:u32){for(var attempt=0u;attempt<16u;attempt+=1u){let result=atomicCompareExchangeWeak(&arena[params.offsets0.w+slot],claimed,claimed&~CLAIMED);if(result.exchanged||result.old_value!=claimed){return;}}atomicOr(&arena[3],2u);}
@compute @workgroup_size(64) fn retirePages(@builtin(global_invocation_id) gid:vec3u){let item=gid.x;if(item<params.shape.y){let owner=atomicLoad(&arena[params.offsets0.w+item]);if(owner!=INVALID){if((owner&CLAIMED)!=0u){releaseClaim(item,owner);}else{let stale=atomicExchange(&arena[params.offsets0.w+item],INVALID);if(stale!=INVALID&&(stale&CLAIMED)==0u){pushFree(item);}}}}if(item>=params.shape.x){return;}let s=state(item);if((s&RESIDENT)==0u||(s&DESIRED)!=0u){return;}let slot=atomicExchange(&arena[params.offsets0.x+item],INVALID);if(slot<params.shape.y){let owner=atomicExchange(&arena[params.offsets0.w+slot],INVALID);if(owner!=INVALID){pushFree(slot);}}atomicStore(&arena[params.offsets0.y+item],0u);}
fn popFree()->u32{var slot=INVALID;loop{let count=atomicLoad(&arena[0]);if(count==0u){break;}let result=atomicCompareExchangeWeak(&arena[0],count,count-1u);if(result.exchanged){slot=atomicLoad(&arena[params.offsets0.z+count-1u]);break;}}return slot;}
fn claimPage(row:u32)->bool{for(var attempt=0u;attempt<16u;attempt+=1u){let claimed=atomicCompareExchangeWeak(&arena[params.offsets0.x+row],INVALID,INVALID-1u);if(claimed.exchanged){return true;}if(claimed.old_value!=INVALID){return false;}}return false;}
@compute @workgroup_size(64) fn allocatePages(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;if(i>=candidateControl[0]){return;}if(i>=arrayLength(&candidates)){atomicOr(&arena[3],1u);return;}let row=candidates[i].row;if(row>=params.shape.x||(state(row)&DESIRED)==0u||pageTable(row)!=INVALID||!claimPage(row)){return;}let slot=popFree();if(slot==INVALID){atomicStore(&arena[params.offsets0.x+row],INVALID);atomicOr(&arena[3],4u);return;}atomicStore(&arena[params.offsets0.x+row],slot);atomicStore(&arena[params.offsets0.w+slot],row);atomicOr(&arena[params.offsets0.y+row],RESIDENT|ACTIVATED);atomicAdd(&arena[4],1u);}
fn insertHash(row:u32){let leaf=leaves[row];let q=leafOrigin(leaf)/max(1u,leaf.size);let mask=params.offsets2.y-1u;var slot=hashCoord(q)&mask;for(var probe=0u;probe<32u;probe+=1u){let result=atomicCompareExchangeWeak(&arena[params.offsets1.y+slot],0u,row+1u);if(result.exchanged||result.old_value==row+1u){return;}slot=(slot+1u)&mask;}atomicOr(&arena[3],16u);}
fn insertAirAlias(p:vec3u,row:u32){
  let mask=params.spare0.y-1u;var slot=hashCoord(p)&mask;let key=airCellKey(p);
  // The exact cell key and complemented row form one compact cell -> incident
  // leaf record. atomicMax makes duplicate dilation deterministic: the
  // smallest incident row wins independently of invocation order.
  let encodedRow=0xffffffffu-row;
  for(var probe=0u;probe<32u;probe+=1u){let at=params.spare0.x+2u*slot;let result=atomicCompareExchangeWeak(&arena[at],0u,key);if(result.exchanged||result.old_value==key){atomicMax(&arena[at+1u],encodedRow);return;}slot=(slot+1u)&mask;}
  // Alias coverage is best-effort ghost support; the exact leaf hash and
  // signed owner background remain authoritative when this bounded table is full.
}
fn publishAirAliases(row:u32){
  let leaf=leaves[row];let o=leafOrigin(leaf);let d=vec3i(params.spare1.xyz);let radius=i32(params.spare0.z);
  // An axis-connected dilation covers every bounded finite-difference and
  // backtrace neighbor without materializing a finest-resolution tile.
  for(var z=-radius;z<=radius;z+=1){for(var y=-radius;y<=radius;y+=1){for(var x=-radius;x<=radius;x+=1){
    if((x==0&&y==0&&z==0)||(abs(x)+abs(y)+abs(z)>radius)){continue;}
    let q=vec3i(o)+vec3i(x,y,z);if(any(q<vec3i(0))||any(q>=d)){continue;}
    let point=vec3f(q)+vec3f(0.5);if(leafFallback(leaf,point)>=0.0){insertAirAlias(vec3u(q),row);}
  }}}
}
@compute @workgroup_size(64) fn publishPages(@builtin(global_invocation_id) gid:vec3u){let row=gid.x;if(row>=params.shape.x){return;}let leaf=leaves[row];let s=state(row);leaves[row].pad0=INVALID;if(((leaf.flags&LIVE)==0u&&(s&DESIRED)==0u)||leaf.size==0u){return;}insertHash(row);if((s&RESIDENT)==0u){return;}let slot=pageTable(row);if(slot>=params.shape.y){return;}leaves[row].pad0=slot;if(leaf.size==1u){atomicAdd(&arena[8],1u);}else{atomicAdd(&arena[9],1u);}atomicMax(&arena[10],leaf.size);if(leaf.size==1u&&(s&CORE)!=0u){publishAirAliases(row);}let out=atomicAdd(&arena[params.offsets1.x],1u);if(out>=params.shape.y){atomicOr(&arena[3],8u);return;}atomicStore(&arena[params.offsets1.x+4u+out],row);}
@compute @workgroup_size(1) fn prepareDispatch(){let count=min(atomicLoad(&arena[params.offsets1.x]),params.shape.y);let groups=max(1u,(count*params.shape.w+255u)/256u);atomicStore(&arena[1],count);atomicStore(&arena[6],count);atomicMax(&arena[2],count);atomicStore(&arena[params.offsets1.x+1u],min(65535u,groups));atomicStore(&arena[params.offsets1.x+2u],max(1u,(groups+65534u)/65535u));atomicStore(&arena[params.offsets1.x+3u],1u);}
@compute @workgroup_size(256) fn initializeActivatedPages(@builtin(global_invocation_id) gid:vec3u){let item=invocation(gid);if(item.x==INVALID||atomicLoad(&arena[3])!=0u||(state(item.x)&ACTIVATED)==0u){return;}let p=samplePosition(item.x,item.y);storePhi(params.offsets1.z,pageTable(item.x),item.y,leafFallback(leaves[item.x],p));}
@compute @workgroup_size(64) fn reduceReferenceVolume(@builtin(global_invocation_id) gid:vec3u){let row=activeRow(gid.x);if(row==INVALID||atomicLoad(&arena[3])!=0u){return;}conservativeAdd(13u,u32(round(smoothVolume(centrePhi(row))*4096.0)));}
struct Backtrace { departure:vec3f,segments:u32,resident:bool }
fn traceSparseDeparture(ownerRow:u32,p:vec3f)->Backtrace{
  var row=residentBandRow(p,ownerRow);if(row==INVALID){return Backtrace(p,1u,false);}
  var velocity=leaves[row].motion.xyz;
  if(!all(abs(velocity)<=vec3f(3.402823e38))){return Backtrace(p,1u,false);}
  let pageH=max(1e-9,min(params.cellDt.x,min(params.cellDt.y,params.cellDt.z))*f32(leaves[ownerRow].size)/f32(params.shape.z));
  let maximum=max(1u,params.spare0.w);let segments=u32(clamp(ceil(length(velocity)*params.cellDt.w/pageH),1.0,f32(maximum)));let segmentDt=params.cellDt.w/f32(segments);var departure=p;
  for(var segment=0u;segment<segments;segment+=1u){
    departure-=velocity*segmentDt/max(params.cellDt.xyz,vec3f(1e-9));
    row=residentBandRow(departure,row);if(row==INVALID){return Backtrace(departure,segments,false);}
    if(segment+1u<segments){velocity=leaves[row].motion.xyz;if(!all(abs(velocity)<=vec3f(3.402823e38))){return Backtrace(departure,segments,false);}}
  }
  return Backtrace(departure,segments,true);
}
@compute @workgroup_size(256) fn transportPhi(@builtin(global_invocation_id) gid:vec3u){
  let item=invocation(gid);let failure=atomicLoad(&arena[3]);if(item.x==INVALID||(failure&0xffffffbfu)!=0u){return;}let row=item.x;let p=samplePosition(row,item.y);let leaf=leaves[row];
  if((state(row)&ACTIVATED)!=0u){storePhi(params.offsets1.w,pageTable(row),item.y,leafFallback(leaf,p));return;}
  let trace=traceSparseDeparture(row,p);if(!trace.resident){atomicAdd(&arena[11],1u);}
  // A moving interface can legitimately backtrace just beyond the currently
  // resident halo. Preserve the sample with the owning leaf's affine
  // continuation instead of poisoning the entire arena: hierarchicalPhi
  // already resolves a resident neighbour when one exists and otherwise
  // falls back to this row without another pass or dense publication.
  storePhi(params.offsets1.w,pageTable(row),item.y,hierarchicalPhi(row,trace.departure,params.offsets1.z));
}
@compute @workgroup_size(256) fn copyPhiBToA(@builtin(global_invocation_id) gid:vec3u){let item=invocation(gid);if(item.x==INVALID||atomicLoad(&arena[3])!=0u){return;}let slot=pageTable(item.x);storePhi(params.offsets1.z,slot,item.y,loadPhi(params.offsets1.w,slot,item.y));atomicAnd(&arena[params.offsets0.y+item.x],~ACTIVATED);}
@compute @workgroup_size(256) fn seedRedistance(@builtin(global_invocation_id) gid:vec3u){let item=invocation(gid);if(item.x==INVALID||atomicLoad(&arena[3])!=0u){return;}let slot=pageTable(item.x);storePhi(params.offsets2.x,slot,item.y,loadPhi(params.offsets1.z,slot,item.y));}
fn redistance(row:u32,local:u32,inputOffset:u32)->f32{let p=samplePosition(row,local);let leaf=leaves[row];let gridStep=f32(leaf.size)/f32(params.shape.z);let physicalStep=max(vec3f(1e-6),gridStep*params.cellDt.xyz);let dx=vec3f(gridStep,0,0);let dy=vec3f(0,gridStep,0);let dz=vec3f(0,0,gridStep);let gx=(hierarchicalPhi(row,p+dx,inputOffset)-hierarchicalPhi(row,p-dx,inputOffset))/(2.0*physicalStep.x);let gy=(hierarchicalPhi(row,p+dy,inputOffset)-hierarchicalPhi(row,p-dy,inputOffset))/(2.0*physicalStep.y);let gz=(hierarchicalPhi(row,p+dz,inputOffset)-hierarchicalPhi(row,p-dz,inputOffset))/(2.0*physicalStep.z);let slot=pageTable(row);let original=loadPhi(params.offsets2.x,slot,local);let current=hierarchicalPhi(row,p,inputOffset);let h=min(physicalStep.x,min(physicalStep.y,physicalStep.z));let sign=original/sqrt(original*original+h*h);return current-0.3*h*sign*(length(vec3f(gx,gy,gz))-1.0);}
@compute @workgroup_size(256) fn redistanceAToB(@builtin(global_invocation_id) gid:vec3u){let item=invocation(gid);if(item.x==INVALID||atomicLoad(&arena[3])!=0u){return;}storePhi(params.offsets1.w,pageTable(item.x),item.y,redistance(item.x,item.y,params.offsets1.z));}
@compute @workgroup_size(256) fn redistanceBToA(@builtin(global_invocation_id) gid:vec3u){let item=invocation(gid);if(item.x==INVALID||atomicLoad(&arena[3])!=0u){return;}storePhi(params.offsets1.z,pageTable(item.x),item.y,redistance(item.x,item.y,params.offsets1.w));}
@compute @workgroup_size(64) fn reduceCurrentVolume(@builtin(global_invocation_id) gid:vec3u){let row=activeRow(gid.x);if(row==INVALID||atomicLoad(&arena[3])!=0u){return;}let value=centrePhi(row);conservativeAdd(14u,u32(round(smoothVolume(value)*4096.0)));if(abs(value)<2.0*params.cellDt.y){conservativeAdd(15u,1u);}}
@compute @workgroup_size(256) fn correctPageVolume(@builtin(global_invocation_id) gid:vec3u){let item=invocation(gid);if(item.x==INVALID||atomicLoad(&arena[3])!=0u){return;}let interfaceCount=atomicLoad(&arena[15]);if(interfaceCount==0u){return;}let delta=(f32(atomicLoad(&arena[14]))-f32(atomicLoad(&arena[13])))/4096.0;let shift=clamp(delta*(4.0*params.cellDt.y)/f32(interfaceCount),-1.5*min(params.cellDt.x,min(params.cellDt.y,params.cellDt.z)),1.5*min(params.cellDt.x,min(params.cellDt.y,params.cellDt.z)));let slot=pageTable(item.x);storePhi(params.offsets1.z,slot,item.y,loadPhi(params.offsets1.z,slot,item.y)+shift);}
`;

export const octreeSurfaceDensePublicationShader = /* wgsl */ `
struct SurfaceLeaf { originX:u32,originY:u32,originZ:u32,size:u32,flags:u32,pad0:u32,pad1:u32,pad2:u32,phiGradient:vec4f,motion:vec4f }
struct Params { shape:vec4u, offsets0:vec4u, offsets1:vec4u, offsets2:vec4u, cellDt:vec4f, spare0:vec4u, spare1:vec4u, spare2:vec4u }
@group(0) @binding(0) var<storage,read> leaves:array<SurfaceLeaf>;
@group(0) @binding(1) var<storage,read> arena:array<u32>;
@group(0) @binding(2) var<uniform> params:Params;
@group(0) @binding(3) var densePhi:texture_storage_3d<r32float,write>;
const INVALID=0xffffffffu;
fn pageWord(slot:u32,local:u32)->u32{return params.offsets1.z+slot*params.shape.w+local;}
fn phi(slot:u32,q:vec3u)->f32{let local=q.x+params.shape.z*(q.y+params.shape.z*q.z);return bitcast<f32>(arena[pageWord(slot,local)]);}
@compute @workgroup_size(64) fn publishDensePhi(@builtin(global_invocation_id) gid:vec3u){
  if(arena[3]!=0u){return;}let activeCount=arena[params.offsets1.x];let item=gid.x;if(item>=activeCount){return;}
  let row=arena[params.offsets1.x+4u+item];if(row>=params.shape.x){return;}let slot=arena[params.offsets0.x+row];if(slot==INVALID||slot>=params.shape.y){return;}
  // An even-resolution cell-centred page has its leaf-centre value at the
  // average of the central 2^3 samples. Runtime pages select finest (size-one)
  // interface leaves, so publication updates exactly one topology cell.
  let lo=params.shape.z/2u-1u;let hi=params.shape.z/2u;var value=0.0;for(var z=lo;z<=hi;z+=1u){for(var y=lo;y<=hi;y+=1u){for(var x=lo;x<=hi;x+=1u){value+=phi(slot,vec3u(x,y,z));}}}
  let leaf=leaves[row];let origin=vec3u(leaf.originX,leaf.originY,leaf.originZ);textureStore(densePhi,origin,vec4f(value*0.125,0,0,0));
}
`;
