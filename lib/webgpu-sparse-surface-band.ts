/**
 * Dynamically paged, two-sided narrow band for the octree liquid surface.
 *
 * The pressure octree and the dense compatibility level set remain separate
 * concerns. This object owns detail-selected interface pages plus the bounded
 * support halo needed to transport them, with a fixed slot pool and an
 * explicit coarse-field fallback for every missing lookup. No shader is
 * allowed to derive a payload address from a logical page before validating
 * the page-table entry.
 */

export const SPARSE_SURFACE_INVALID_PAGE = 0xffff_ffff;
export const SPARSE_SURFACE_RESIDENT = 1;
export const SPARSE_SURFACE_CORE = 2;
export const SPARSE_SURFACE_HALO = 4;
export const SPARSE_SURFACE_ACTIVATED = 8;
export const SPARSE_SURFACE_DESIRED = 16;
export const SPARSE_SURFACE_ACTIVE_DISPATCH_OFFSET_BYTES = 4;
/** Sixteen-word allocator publication consumed by simulation and rendering. */
export const SPARSE_SURFACE_CONTROL_BYTES = 64;

export type SparseSurfaceBandMode = "mirror" | "authoritative";

export interface SparseSurfaceBandOptions {
  /** Fine samples per coarse transport-cell edge. */
  refinementFactor?: 1 | 2 | 4;
  brickSize?: 4 | 8;
  /** Minimum two-sided phi support, measured in fine cells. */
  bandCells?: number;
  /** Extra fine cells reserved for interpolation and transport stencils. */
  stencilCells?: number;
  /** Consecutive undesired updates retained before releasing a slot. */
  retireAfterFrames?: number;
  /** Fraction of the logical page lattice backed by physical slots. */
  maximumResidentFraction?: number;
  /** Optional hard physical-page ceiling. */
  maximumPages?: number;
  density_kg_m3?: number;
  surfaceTension_N_m?: number;
  pressureIterations?: number;
  /** Experimental local velocity correction. Off keeps the 32-pass global Chebyshev solve authoritative. */
  fineDynamics?: boolean;
  mode?: SparseSurfaceBandMode;
}

export interface SparseSurfaceBandPlan {
  coarseDimensions: readonly [number, number, number];
  fineDimensions: readonly [number, number, number];
  brickDimensions: readonly [number, number, number];
  logicalPageCount: number;
  physicalPageCapacity: number;
  brickSize: 4 | 8;
  refinementFactor: 1 | 2 | 4;
  voxelsPerPage: number;
  bytesPerPage: number;
  allocatedPayloadBytes: number;
}

export interface SparseSurfaceBandStats {
  resident: number;
  core: number;
  halo: number;
  activated: number;
  retired: number;
  overflow: number;
  free: number;
  peakResident: number;
  generation: number;
  logicalPageCount: number;
  physicalPageCapacity: number;
}

export interface SparseSurfaceBandGPUSource {
  /** Renderer staging accepts only the solver-authoritative fine field. */
  mode: SparseSurfaceBandMode;
  pageTable: GPUBufferBinding;
  states: GPUBufferBinding;
  activePages: GPUBufferBinding;
  phi: GPUBufferBinding;
  velocity: GPUBufferBinding;
  params: GPUBufferBinding;
  control: GPUBufferBinding;
  coarseLevelSet: GPUTexture;
  coarseVelocity: GPUTexture;
  fineDimensions: readonly [number, number, number];
  brickDimensions: readonly [number, number, number];
  brickSize: 4 | 8;
  refinementFactor: 1 | 2 | 4;
  pageCapacity: number;
  revision: number;
}

function positiveInteger(value: number, label: string) {
  if (!Number.isInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

export function requiredSparseSurfaceBandCells(
  maximumSpeed_m_s: number,
  dt_s: number,
  fineCellSize_m: number,
  interpolationRadius = 2,
  redistanceRadius = 2,
  couplingRadius = 1,
) {
  const speed = Number.isFinite(maximumSpeed_m_s) ? Math.max(0, maximumSpeed_m_s) : 0;
  const dt = Number.isFinite(dt_s) ? Math.max(0, dt_s) : 0;
  const h = Number.isFinite(fineCellSize_m) ? Math.max(1e-9, fineCellSize_m) : 1e-9;
  return Math.max(1, Math.ceil(speed * dt / h)
    + Math.max(0, Math.ceil(interpolationRadius))
    + Math.max(0, Math.ceil(redistanceRadius))
    + Math.max(0, Math.ceil(couplingRadius)));
}

/** CPU mirror of Ando--Batty Eq. 39, used by deterministic sizing invariants. */
export function sparseSurfaceSizingSignal(
  surfaceCurvature_m_inv: number,
  diagonalVelocityGradient_s_inv: readonly [number, number, number],
  curvatureWeight = 4,
  velocityWeight_s_m = 3,
) {
  const curvature = Number.isFinite(surfaceCurvature_m_inv) ? Math.abs(surfaceCurvature_m_inv) : 0;
  const strain2 = diagonalVelocityGradient_s_inv.reduce((sum, value) => sum + (Number.isFinite(value) ? value * value : 0), 0);
  return Math.max(0, curvatureWeight) * curvature + Math.max(0, velocityWeight_s_m) * Math.sqrt(strain2);
}

export function sparseSurfaceNeedsRefinement(sizing_m_inv: number, parentCellSize_m: number) {
  return Number.isFinite(sizing_m_inv) && Number.isFinite(parentCellSize_m)
    && sizing_m_inv > 0 && parentCellSize_m > 0 && sizing_m_inv * parentCellSize_m > 1;
}

export function planSparseSurfaceBand(
  coarseDimensions: readonly [number, number, number],
  options: SparseSurfaceBandOptions = {},
): SparseSurfaceBandPlan {
  coarseDimensions.forEach((value, axis) => positiveInteger(value, `Coarse surface dimension ${axis}`));
  const refinementFactor = options.refinementFactor ?? 2;
  if (refinementFactor !== 1 && refinementFactor !== 2 && refinementFactor !== 4) throw new RangeError("Surface refinement factor must be 1, 2, or 4");
  const brickSize = options.brickSize ?? 8;
  if (brickSize !== 4 && brickSize !== 8) throw new RangeError("Surface brick size must be 4 or 8");
  const fineDimensions = coarseDimensions.map((value) => value * refinementFactor) as [number, number, number];
  const brickDimensions = fineDimensions.map((value) => Math.ceil(value / brickSize)) as [number, number, number];
  const logicalPageCount = brickDimensions[0] * brickDimensions[1] * brickDimensions[2];
  const fraction = Number.isFinite(options.maximumResidentFraction)
    ? Math.max(1 / logicalPageCount, Math.min(1, options.maximumResidentFraction!))
    : 0.75;
  const fractionCapacity = Math.max(1, Math.ceil(logicalPageCount * fraction));
  const hardCapacity = options.maximumPages === undefined
    ? logicalPageCount
    : Math.max(1, Math.min(logicalPageCount, Math.floor(options.maximumPages)));
  const physicalPageCapacity = Math.min(hardCapacity, fractionCapacity);
  const voxelsPerPage = brickSize ** 3;
  // The default geometric-detail path owns phi A/B only. Experimental local
  // dynamics additionally owns velocity-residual A/B and pressure A/B.
  // Allocator metadata is accounted separately by the owning class.
  const bytesPerPage = voxelsPerPage * (options.fineDynamics ? 48 : 8);
  return {
    coarseDimensions: [...coarseDimensions], fineDimensions, brickDimensions,
    logicalPageCount, physicalPageCapacity, brickSize, refinementFactor,
    voxelsPerPage, bytesPerPage, allocatedPayloadBytes: physicalPageCapacity * bytesPerPage,
  };
}

// Control words: free, generation, overflow, activated, retired, resident,
// core, halo, peak resident.  The active list owns its indirect dispatch.
export const sparseSurfaceResidencyShader = /* wgsl */ `
struct Params {
  coarseDims: vec4u,
  fineDims: vec4u,
  brickDims: vec4u,
  settings: vec4f,
  cellAndDt: vec4f,
  sizing: vec4f,
  physical: vec4f,
}
@group(0) @binding(0) var coarsePhi: texture_3d<f32>;
@group(0) @binding(1) var coarseVelocity: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> pageTable: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> states: array<u32>;
@group(0) @binding(4) var<storage, read_write> freeList: array<u32>;
@group(0) @binding(5) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> activePages: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: Params;
@group(0) @binding(8) var<storage, read_write> sizingA: array<f32>;
@group(0) @binding(9) var<storage, read_write> sizingB: array<f32>;
struct SolidCell { fraction: f32, owner: i32 }
@group(0) @binding(10) var<storage, read> solidCells: array<SolidCell>;

const INVALID: u32 = 0xffffffffu;
const RESIDENT: u32 = 1u;
const CORE: u32 = 2u;
const HALO: u32 = 4u;
const ACTIVATED: u32 = 8u;
const DESIRED: u32 = 16u;
const PRIORITY_SHIFT: u32 = 8u;
const PRIORITY_MASK: u32 = 3u << PRIORITY_SHIFT;

fn pageCoordinate(index: u32) -> vec3u {
  return vec3u(index % params.brickDims.x, (index / params.brickDims.x) % params.brickDims.y,
    index / (params.brickDims.x * params.brickDims.y));
}
fn validCoarse(q: vec3i) -> bool { return all(q >= vec3i(0)) && all(q < vec3i(params.coarseDims.xyz)); }
fn clampCoarse(q: vec3i) -> vec3i { return clamp(q, vec3i(0), vec3i(params.coarseDims.xyz) - vec3i(1)); }
fn coarseIndex(q: vec3i) -> u32 {
  let p=vec3u(clampCoarse(q));return p.x+params.coarseDims.x*(p.y+params.coarseDims.y*p.z);
}
fn phi(q: vec3i) -> f32 { return textureLoad(coarsePhi, clampCoarse(q), 0).x; }
fn speed(q: vec3i) -> f32 { return length(textureLoad(coarseVelocity, clampCoarse(q), 0).xyz); }
fn velocity(q: vec3i) -> vec3f { return textureLoad(coarseVelocity, clampCoarse(q), 0).xyz; }
fn solidPhi(q: vec3i) -> f32 {
  let i=coarseIndex(q);let h=min(params.cellAndDt.x,min(params.cellAndDt.y,params.cellAndDt.z));
  if(i>=arrayLength(&solidCells)){return 0.5*h;}
  return (0.5-clamp(solidCells[i].fraction,0.0,1.0))*h;
}
fn combinedPhi(q: vec3i) -> f32 { return phi(q)+solidPhi(q); }
fn surfaceNormal(q: vec3i) -> vec3f {
  let gradient=vec3f(
    (combinedPhi(q+vec3i(1,0,0))-combinedPhi(q-vec3i(1,0,0)))/(2.0*params.cellAndDt.x),
    (combinedPhi(q+vec3i(0,1,0))-combinedPhi(q-vec3i(0,1,0)))/(2.0*params.cellAndDt.y),
    (combinedPhi(q+vec3i(0,0,1))-combinedPhi(q-vec3i(0,0,1)))/(2.0*params.cellAndDt.z));
  return gradient/max(length(gradient),1e-6);
}
fn sizingCellA(q: vec3i) -> f32 {
  if(!validCoarse(q)){return 0.0;}let i=coarseIndex(q);if(i>=arrayLength(&sizingA)){return 0.0;}return sizingA[i];
}
fn sizingCellB(q: vec3i) -> f32 {
  if(!validCoarse(q)){return 0.0;}let i=coarseIndex(q);if(i>=arrayLength(&sizingB)){return 0.0;}return sizingB[i];
}
fn sizingAtA(position: vec3f) -> f32 {
  let p=clamp(position,vec3f(0.0),vec3f(params.coarseDims.xyz-vec3u(1)));
  let a=vec3i(floor(p));let b=min(a+vec3i(1),vec3i(params.coarseDims.xyz)-vec3i(1));let t=fract(p);
  let s000=sizingCellA(vec3i(a.x,a.y,a.z));let s100=sizingCellA(vec3i(b.x,a.y,a.z));
  let s010=sizingCellA(vec3i(a.x,b.y,a.z));let s110=sizingCellA(vec3i(b.x,b.y,a.z));
  let s001=sizingCellA(vec3i(a.x,a.y,b.z));let s101=sizingCellA(vec3i(b.x,a.y,b.z));
  let s011=sizingCellA(vec3i(a.x,b.y,b.z));let s111=sizingCellA(vec3i(b.x,b.y,b.z));
  return mix(mix(mix(s000,s100,t.x),mix(s010,s110,t.x),t.y),mix(mix(s001,s101,t.x),mix(s011,s111,t.x),t.y),t.z);
}
fn interfaceAdjacent(q: vec3i) -> bool {
  let p=phi(q);let h=min(params.cellAndDt.x,min(params.cellAndDt.y,params.cellAndDt.z));
  if(abs(p)<=0.75*h){return true;}
  return p*phi(q+vec3i(1,0,0))<=0.0||p*phi(q-vec3i(1,0,0))<=0.0
    ||p*phi(q+vec3i(0,1,0))<=0.0||p*phi(q-vec3i(0,1,0))<=0.0
    ||p*phi(q+vec3i(0,0,1))<=0.0||p*phi(q-vec3i(0,0,1))<=0.0;
}

// Ando--Batty 2020, Eq. 39.  The signal is evaluated only next to the
// interface, so unrelated bulk strain cannot activate a nearby flat surface.
// The previous scalar sizing field is semi-Lagrangian advected and decayed as
// in Eq. 36, retaining a moving splash briefly without frame-count flicker.
@compute @workgroup_size(64)
fn evaluateSizing(@builtin(global_invocation_id) gid: vec3u) {
  let i=gid.x;let count=params.coarseDims.x*params.coarseDims.y*params.coarseDims.z;
  if(i>=count||i>=arrayLength(&sizingB)){return;}
  let q=vec3i(vec3u(i%params.coarseDims.x,(i/params.coarseDims.x)%params.coarseDims.y,i/(params.coarseDims.x*params.coarseDims.y)));
  let hx=params.cellAndDt.x;let hy=params.cellAndDt.y;let hz=params.cellAndDt.z;
  var proposed=0.0;
  if(interfaceAdjacent(q)){
    // Eq. 39 writes Laplacian(phi), equal to div(normalized grad(phi)) for a
    // signed-distance field. Our mass controller deliberately perturbs the
    // distance profile, so use the normalized form: a monotone planar phi then
    // remains exactly zero-curvature instead of spuriously refining the plane.
    let curvature=(surfaceNormal(q+vec3i(1,0,0)).x-surfaceNormal(q-vec3i(1,0,0)).x)/(2.0*hx)
      +(surfaceNormal(q+vec3i(0,1,0)).y-surfaceNormal(q-vec3i(0,1,0)).y)/(2.0*hy)
      +(surfaceNormal(q+vec3i(0,0,1)).z-surfaceNormal(q-vec3i(0,0,1)).z)/(2.0*hz);
    let dux=(velocity(q+vec3i(1,0,0)).x-velocity(q-vec3i(1,0,0)).x)/(2.0*hx);
    let duy=(velocity(q+vec3i(0,1,0)).y-velocity(q-vec3i(0,1,0)).y)/(2.0*hy);
    let duz=(velocity(q+vec3i(0,0,1)).z-velocity(q-vec3i(0,0,1)).z)/(2.0*hz);
    proposed=4.0*abs(curvature)+3.0*sqrt(dux*dux+duy*duy+duz*duz);
  }
  let departure=vec3f(q)-velocity(q)*params.cellAndDt.w/params.cellAndDt.xyz;
  let decay=pow(0.9,params.cellAndDt.w/0.01);
  sizingB[i]=max(proposed,decay*sizingAtA(departure));
}

fn propagatedA(q: vec3i) -> f32 {
  let s=sizingCellA(q);
  return (max(s,sizingCellA(q+vec3i(1,0,0)))+max(s,sizingCellA(q-vec3i(1,0,0)))
    +max(s,sizingCellA(q+vec3i(0,1,0)))+max(s,sizingCellA(q-vec3i(0,1,0)))
    +max(s,sizingCellA(q+vec3i(0,0,1)))+max(s,sizingCellA(q-vec3i(0,0,1))))/6.0;
}
fn propagatedB(q: vec3i) -> f32 {
  let s=sizingCellB(q);
  return (max(s,sizingCellB(q+vec3i(1,0,0)))+max(s,sizingCellB(q-vec3i(1,0,0)))
    +max(s,sizingCellB(q+vec3i(0,1,0)))+max(s,sizingCellB(q-vec3i(0,1,0)))
    +max(s,sizingCellB(q+vec3i(0,0,1)))+max(s,sizingCellB(q-vec3i(0,0,1))))/6.0;
}
@compute @workgroup_size(64)
fn propagateBToA(@builtin(global_invocation_id) gid: vec3u) {
  let i=gid.x;let count=params.coarseDims.x*params.coarseDims.y*params.coarseDims.z;if(i>=count||i>=arrayLength(&sizingA)){return;}
  let q=vec3i(vec3u(i%params.coarseDims.x,(i/params.coarseDims.x)%params.coarseDims.y,i/(params.coarseDims.x*params.coarseDims.y)));sizingA[i]=propagatedB(q);
}
@compute @workgroup_size(64)
fn propagateAToB(@builtin(global_invocation_id) gid: vec3u) {
  let i=gid.x;let count=params.coarseDims.x*params.coarseDims.y*params.coarseDims.z;if(i>=count||i>=arrayLength(&sizingB)){return;}
  let q=vec3i(vec3u(i%params.coarseDims.x,(i/params.coarseDims.x)%params.coarseDims.y,i/(params.coarseDims.x*params.coarseDims.y)));sizingB[i]=propagatedA(q);
}

@compute @workgroup_size(1)
fn resetCounters() {
  atomicAdd(&control[1], 1u);
  for (var i = 2u; i <= 7u; i += 1u) { atomicStore(&control[i], 0u); }
  atomicStore(&activePages[0], 0u);
  atomicStore(&activePages[1], 0u);
  atomicStore(&activePages[2], 1u);
  atomicStore(&activePages[3], 1u);
}

@compute @workgroup_size(64)
fn classifyDesired(@builtin(global_invocation_id) gid: vec3u) {
  let pageIndex = gid.x;
  if (pageIndex >= params.brickDims.w || pageIndex >= arrayLength(&states)) { return; }
  let brickSize = params.fineDims.w;
  let factor = params.coarseDims.w;
  let fineBegin = pageCoordinate(pageIndex) * brickSize;
  let fineEnd = min(fineBegin + vec3u(brickSize), params.fineDims.xyz);
  let coarseBegin=vec3i(fineBegin/factor);
  let coarseEnd=clamp(vec3i((fineEnd+vec3u(factor-1u))/factor),vec3i(0),vec3i(params.coarseDims.xyz));
  let coarseH=min(params.cellAndDt.x,min(params.cellAndDt.y,params.cellAndDt.z));
  var maximumSpeed = 0.0;
  var core=false;
  var coreScore=0.0;
  for (var z = coarseBegin.z; z < coarseEnd.z; z += 1) {
    for (var y = coarseBegin.y; y < coarseEnd.y; y += 1) {
      for (var x = coarseBegin.x; x < coarseEnd.x; x += 1) {
        let q = vec3i(x, y, z);
        maximumSpeed = max(maximumSpeed, speed(q));
        let sampleSizing=sizingCellA(q);
        if(interfaceAdjacent(q)){
          let score=sampleSizing*coarseH;core=core||score>1.0;coreScore=max(coreScore,score);
        }
      }
    }
  }
  let fineH = min(params.cellAndDt.x, min(params.cellAndDt.y, params.cellAndDt.z)) / f32(params.coarseDims.w);
  let supportFine=max(params.settings.x,params.settings.y)+ceil(maximumSpeed*params.cellAndDt.w/max(fineH,1e-8));
  let supportCoarse=i32(ceil(supportFine/f32(factor)));
  var desired=core;
  if(!desired){
    let haloBegin=max(vec3i(0),coarseBegin-vec3i(supportCoarse));
    let haloEnd=min(vec3i(params.coarseDims.xyz),coarseEnd+vec3i(supportCoarse));
    for(var z=haloBegin.z;z<haloEnd.z&&!desired;z+=1){for(var y=haloBegin.y;y<haloEnd.y&&!desired;y+=1){for(var x=haloBegin.x;x<haloEnd.x;x+=1){
      let q=vec3i(x,y,z);let sampleSizing=sizingCellA(q);
      desired=desired||(interfaceAdjacent(q)&&sampleSizing*coarseH>1.0);
    }}}
  }
  let previous = states[pageIndex];
  let wasResident = (previous & RESIDENT) != 0u;
  var dry = select(min(0xffffu, (previous >> 16u) + 1u), 0u, desired);
  let resident = desired || (wasResident && dry <= u32(params.settings.z));
  var flags = select(0u, RESIDENT, resident)
    | select(0u, CORE, core)
    | select(0u, HALO, resident && !core)
    | select(0u, DESIRED, desired);
  let priority=select(0u,select(1u,select(2u,3u,coreScore>=8.0),coreScore>=4.0),coreScore>=2.0);
  flags|=(priority<<PRIORITY_SHIFT)&PRIORITY_MASK;
  states[pageIndex] = flags | (dry << 16u);
}

@compute @workgroup_size(64)
fn retirePages(@builtin(global_invocation_id) gid: vec3u) {
  let pageIndex = gid.x;
  if (pageIndex >= params.brickDims.w || pageIndex >= arrayLength(&states)) { return; }
  if ((states[pageIndex] & RESIDENT) != 0u) { return; }
  let slot = atomicExchange(&pageTable[pageIndex], INVALID);
  if (slot == INVALID || slot >= u32(params.sizing.w)) { return; }
  let freeSlot = atomicAdd(&control[0], 1u);
  if (freeSlot < arrayLength(&freeList)) { freeList[freeSlot] = slot; }
  else { atomicStore(&control[2], 1u); }
  atomicAdd(&control[4], 1u);
}

fn allocateAndListFor(pageIndex: u32, wantCore: bool, wantedPriority: u32) {
  if (pageIndex >= params.brickDims.w || pageIndex >= arrayLength(&states)) { return; }
  var state = states[pageIndex];
  if ((state & RESIDENT) == 0u) { return; }
  let isCore=(state&CORE)!=0u;let priority=(state&PRIORITY_MASK)>>PRIORITY_SHIFT;
  if(isCore!=wantCore||(wantCore&&priority!=wantedPriority)){return;}
  var slot = atomicLoad(&pageTable[pageIndex]);
  if (slot == INVALID) {
    let oldFree = atomicSub(&control[0], 1u);
    if (oldFree == 0u) {
      atomicAdd(&control[0], 1u);
      atomicStore(&control[2], 1u);
      states[pageIndex] = state & ~RESIDENT;
      return;
    }
    let freeIndex = oldFree - 1u;
    if (freeIndex >= arrayLength(&freeList)) {
      atomicAdd(&control[0], 1u);
      atomicStore(&control[2], 1u);
      states[pageIndex] = state & ~RESIDENT;
      return;
    }
    slot = freeList[freeIndex];
    if (slot >= u32(params.sizing.w)) {
      atomicAdd(&control[0], 1u);
      atomicStore(&control[2], 1u);
      states[pageIndex] = state & ~RESIDENT;
      return;
    }
    atomicStore(&pageTable[pageIndex], slot);
    state |= ACTIVATED;
    states[pageIndex] = state;
    atomicAdd(&control[3], 1u);
  }
  let activeIndex = atomicAdd(&activePages[0], 1u);
  if (4u + activeIndex < arrayLength(&activePages)) {
    atomicStore(&activePages[4u + activeIndex], pageIndex);
  } else {
    atomicStore(&control[2], 1u);
    return;
  }
  let resident = atomicAdd(&control[5], 1u) + 1u;
  atomicMax(&control[8], resident);
  if ((state & CORE) != 0u) { atomicAdd(&control[6], 1u); }
  else { atomicAdd(&control[7], 1u); }
}

// Strongest paper sizing wins physical slots first. Halo pages are allocated
// only after every detail seed, so a tight budget can never evict the feature
// that justified its interpolation support.
@compute @workgroup_size(64)
fn allocateCore3(@builtin(global_invocation_id) gid: vec3u){allocateAndListFor(gid.x,true,3u);}
@compute @workgroup_size(64)
fn allocateCore2(@builtin(global_invocation_id) gid: vec3u){allocateAndListFor(gid.x,true,2u);}
@compute @workgroup_size(64)
fn allocateCore1(@builtin(global_invocation_id) gid: vec3u){allocateAndListFor(gid.x,true,1u);}
@compute @workgroup_size(64)
fn allocateCore0(@builtin(global_invocation_id) gid: vec3u){allocateAndListFor(gid.x,true,0u);}
@compute @workgroup_size(64)
fn allocateHalo(@builtin(global_invocation_id) gid: vec3u){allocateAndListFor(gid.x,false,0u);}

@compute @workgroup_size(1)
fn finalizeDispatch() {
  let resident = min(atomicLoad(&activePages[0]), u32(params.sizing.w));
  let voxelsPerPage = params.fineDims.w * params.fineDims.w * params.fineDims.w;
  let blocks = (resident * voxelsPerPage + 255u) / 256u;
  let x = min(blocks, 65535u);
  var y = 1u;
  if (x > 0u) { y = (blocks + x - 1u) / x; }
  atomicStore(&activePages[1], x);
  atomicStore(&activePages[2], y);
}
`;

export const sparseSurfaceFieldShader = /* wgsl */ `
struct Params {
  coarseDims: vec4u,
  fineDims: vec4u,
  brickDims: vec4u,
  settings: vec4f,
  cellAndDt: vec4f,
  sizing: vec4f,
  physical: vec4f,
}
@group(0) @binding(0) var coarsePhi: texture_3d<f32>;
@group(0) @binding(1) var coarseVelocity: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> pageTable: array<u32>;
@group(0) @binding(3) var<storage, read> states: array<u32>;
@group(0) @binding(4) var<storage, read> activePages: array<u32>;
@group(0) @binding(5) var<storage, read_write> phiA: array<f32>;
@group(0) @binding(6) var<storage, read_write> phiB: array<f32>;
@group(0) @binding(7) var<storage, read_write> fineVelocity: array<vec4f>;
@group(0) @binding(8) var<uniform> params: Params;

const INVALID: u32 = 0xffffffffu;
const ACTIVATED: u32 = 8u;

fn pageCoordinate(index: u32) -> vec3u {
  return vec3u(index % params.brickDims.x, (index / params.brickDims.x) % params.brickDims.y,
    index / (params.brickDims.x * params.brickDims.y));
}
fn logicalPage(q: vec3u) -> u32 {
  let page = q / params.fineDims.w;
  return page.x + params.brickDims.x * (page.y + params.brickDims.y * page.z);
}
fn localVoxel(q: vec3u) -> u32 {
  let local = q % params.fineDims.w;
  return local.x + params.fineDims.w * (local.y + params.fineDims.w * local.z);
}
fn payloadIndex(q: vec3u) -> u32 {
  if (any(q >= params.fineDims.xyz)) { return INVALID; }
  let page = logicalPage(q);
  if (page >= arrayLength(&pageTable)) { return INVALID; }
  let slot = pageTable[page];
  if (slot == INVALID || slot >= u32(params.sizing.w)) { return INVALID; }
  return slot * params.fineDims.w * params.fineDims.w * params.fineDims.w + localVoxel(q);
}
fn clampCoarse(q: vec3i) -> vec3i { return clamp(q, vec3i(0), vec3i(params.coarseDims.xyz) - vec3i(1)); }
fn densePhiAtFine(position: vec3f) -> f32 {
  let factor = f32(params.coarseDims.w);
  let coarse = clamp((position + vec3f(0.5)) / factor - vec3f(0.5), vec3f(0.0), vec3f(params.coarseDims.xyz - vec3u(1)));
  let a = vec3i(floor(coarse));
  let b = min(a + vec3i(1), vec3i(params.coarseDims.xyz) - vec3i(1));
  let t = fract(coarse);
  let p000 = textureLoad(coarsePhi, vec3i(a.x,a.y,a.z), 0).x;
  let p100 = textureLoad(coarsePhi, vec3i(b.x,a.y,a.z), 0).x;
  let p010 = textureLoad(coarsePhi, vec3i(a.x,b.y,a.z), 0).x;
  let p110 = textureLoad(coarsePhi, vec3i(b.x,b.y,a.z), 0).x;
  let p001 = textureLoad(coarsePhi, vec3i(a.x,a.y,b.z), 0).x;
  let p101 = textureLoad(coarsePhi, vec3i(b.x,a.y,b.z), 0).x;
  let p011 = textureLoad(coarsePhi, vec3i(a.x,b.y,b.z), 0).x;
  let p111 = textureLoad(coarsePhi, vec3i(b.x,b.y,b.z), 0).x;
  return mix(mix(mix(p000,p100,t.x),mix(p010,p110,t.x),t.y), mix(mix(p001,p101,t.x),mix(p011,p111,t.x),t.y), t.z);
}
fn denseVelocityAtFine(position: vec3f) -> vec3f {
  let factor = f32(params.coarseDims.w);
  let coarse = clamp((position + vec3f(0.5)) / factor - vec3f(0.5), vec3f(0.0), vec3f(params.coarseDims.xyz - vec3u(1)));
  let a = vec3i(floor(coarse));
  let b = min(a + vec3i(1), vec3i(params.coarseDims.xyz) - vec3i(1));
  let t = fract(coarse);
  let v000 = textureLoad(coarseVelocity, vec3i(a.x,a.y,a.z), 0).xyz;
  let v100 = textureLoad(coarseVelocity, vec3i(b.x,a.y,a.z), 0).xyz;
  let v010 = textureLoad(coarseVelocity, vec3i(a.x,b.y,a.z), 0).xyz;
  let v110 = textureLoad(coarseVelocity, vec3i(b.x,b.y,a.z), 0).xyz;
  let v001 = textureLoad(coarseVelocity, vec3i(a.x,a.y,b.z), 0).xyz;
  let v101 = textureLoad(coarseVelocity, vec3i(b.x,a.y,b.z), 0).xyz;
  let v011 = textureLoad(coarseVelocity, vec3i(a.x,b.y,b.z), 0).xyz;
  let v111 = textureLoad(coarseVelocity, vec3i(b.x,b.y,b.z), 0).xyz;
  return mix(mix(mix(v000,v100,t.x),mix(v010,v110,t.x),t.y), mix(mix(v001,v101,t.x),mix(v011,v111,t.x),t.y), t.z);
}
fn phiCell(q: vec3i, useB: bool) -> f32 {
  if (any(q < vec3i(0)) || any(q >= vec3i(params.fineDims.xyz))) { return densePhiAtFine(vec3f(q)); }
  let index = payloadIndex(vec3u(q));
  if (index == INVALID || index >= arrayLength(&phiA) || index >= arrayLength(&phiB)) { return densePhiAtFine(vec3f(q)); }
  return select(phiA[index], phiB[index], useB);
}
fn trilinearPhi(position: vec3f, useB: bool) -> f32 {
  let p = clamp(position, vec3f(0.0), vec3f(params.fineDims.xyz - vec3u(1)));
  let a = vec3i(floor(p));
  let b = min(a + vec3i(1), vec3i(params.fineDims.xyz) - vec3i(1));
  let t = fract(p);
  let p000 = phiCell(vec3i(a.x,a.y,a.z), useB); let p100 = phiCell(vec3i(b.x,a.y,a.z), useB);
  let p010 = phiCell(vec3i(a.x,b.y,a.z), useB); let p110 = phiCell(vec3i(b.x,b.y,a.z), useB);
  let p001 = phiCell(vec3i(a.x,a.y,b.z), useB); let p101 = phiCell(vec3i(b.x,a.y,b.z), useB);
  let p011 = phiCell(vec3i(a.x,b.y,b.z), useB); let p111 = phiCell(vec3i(b.x,b.y,b.z), useB);
  return mix(mix(mix(p000,p100,t.x),mix(p010,p110,t.x),t.y), mix(mix(p001,p101,t.x),mix(p011,p111,t.x),t.y), t.z);
}
fn invocation(gid: vec3u) -> vec4u {
  let stream = gid.x + gid.y * activePages[1] * 256u;
  let voxels = params.fineDims.w * params.fineDims.w * params.fineDims.w;
  let activeIndex = stream / voxels;
  if (activeIndex >= activePages[0] || 4u + activeIndex >= arrayLength(&activePages)) { return vec4u(INVALID); }
  let pageIndex = activePages[4u + activeIndex];
  if (pageIndex >= params.brickDims.w) { return vec4u(INVALID); }
  let localIndex = stream - activeIndex * voxels;
  let local = vec3u(localIndex % params.fineDims.w, (localIndex / params.fineDims.w) % params.fineDims.w, localIndex / (params.fineDims.w * params.fineDims.w));
  let q = pageCoordinate(pageIndex) * params.fineDims.w + local;
  if (any(q >= params.fineDims.xyz)) { return vec4u(INVALID); }
  let payload = payloadIndex(q);
  return vec4u(q, payload);
}

@compute @workgroup_size(256)
fn mirrorDense(@builtin(global_invocation_id) gid: vec3u) {
  let item = invocation(gid);
  if (item.w == INVALID) { return; }
  let p = densePhiAtFine(vec3f(item.xyz));
  phiA[item.w] = p; phiB[item.w] = p;
  if (params.physical.w > 0.5 && item.w < arrayLength(&fineVelocity)) { fineVelocity[item.w] = vec4f(0.0); }
}

@compute @workgroup_size(256)
fn initializeActivated(@builtin(global_invocation_id) gid: vec3u) {
  let item = invocation(gid);
  if (item.w == INVALID) { return; }
  let page = logicalPage(item.xyz);
  if ((states[page] & ACTIVATED) == 0u) { return; }
  let p = densePhiAtFine(vec3f(item.xyz));
  phiA[item.w] = p; phiB[item.w] = p;
  if (params.physical.w > 0.5 && item.w < arrayLength(&fineVelocity)) { fineVelocity[item.w] = vec4f(0.0); }
}

@compute @workgroup_size(256)
fn advectAToB(@builtin(global_invocation_id) gid: vec3u) {
  let item = invocation(gid);
  if (item.w == INVALID) { return; }
  let position = vec3f(item.xyz);
  var residual = vec3f(0.0);
  if (params.physical.w > 0.5 && item.w < arrayLength(&fineVelocity)) { residual = fineVelocity[item.w].xyz; }
  let velocity = denseVelocityAtFine(position) + residual;
  let fineCell = params.cellAndDt.xyz / f32(params.coarseDims.w);
  let midpoint = position - 0.5 * velocity * params.cellAndDt.w / fineCell;
  let midpointVelocity = denseVelocityAtFine(midpoint) + residual;
  let departure = position - midpointVelocity * params.cellAndDt.w / fineCell;
  let advected = trilinearPhi(departure, false);
  let coarseTarget = densePhiAtFine(position);
  // This is a hierarchical correction, not an unrelated second liquid. Fine
  // phi may carry crests and troughs within two fine cells of the transported
  // coarse interface, but cannot create detached zero sets deep in coarse air
  // or liquid. The bound also makes a newly allocated coarse-fallback page
  // continuous with an already transported neighbor.
  let maximumDetail = 2.0 * min(fineCell.x, min(fineCell.y, fineCell.z));
  phiB[item.w] = clamp(advected, coarseTarget - maximumDetail, coarseTarget + maximumDetail);
}

@compute @workgroup_size(256)
fn copyBToA(@builtin(global_invocation_id) gid: vec3u) {
  let item = invocation(gid); if (item.w == INVALID) { return; }
  phiA[item.w] = phiB[item.w];
}
`;

/** Local composite correction carried as a fine velocity residual.
 *
 * Coarse velocity supplies the Dirichlet boundary condition at a missing page;
 * the resident residual is advected, receives the fine capillary force, and is
 * projected by a page-crossing pressure correction.  Restricting the residual
 * to zero at the band boundary makes total velocity converge continuously to
 * the coarse solve instead of running an unrelated second fluid solver.
 */
export const sparseSurfaceDynamicsShader = /* wgsl */ `
struct Params {
  coarseDims: vec4u,
  fineDims: vec4u,
  brickDims: vec4u,
  settings: vec4f,
  cellAndDt: vec4f,
  sizing: vec4f,
  physical: vec4f,
}
@group(0) @binding(0) var coarseVelocity: texture_3d<f32>;
@group(0) @binding(1) var<storage, read> pageTable: array<u32>;
@group(0) @binding(2) var<storage, read> activePages: array<u32>;
@group(0) @binding(3) var<storage, read> phi: array<f32>;
@group(0) @binding(4) var<storage, read_write> residualA: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> residualB: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> pressureA: array<f32>;
@group(0) @binding(7) var<storage, read_write> pressureB: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

const INVALID: u32 = 0xffffffffu;
fn pageCoordinate(index: u32) -> vec3u {
  return vec3u(index % params.brickDims.x, (index / params.brickDims.x) % params.brickDims.y,
    index / (params.brickDims.x * params.brickDims.y));
}
fn payloadIndex(q: vec3i) -> u32 {
  if (any(q < vec3i(0)) || any(q >= vec3i(params.fineDims.xyz))) { return INVALID; }
  let uq = vec3u(q); let brickSize = params.fineDims.w; let page = uq / brickSize;
  let logical = page.x + params.brickDims.x * (page.y + params.brickDims.y * page.z);
  if (logical >= arrayLength(&pageTable)) { return INVALID; }
  let slot = pageTable[logical];
  if (slot == INVALID || slot >= u32(params.sizing.w)) { return INVALID; }
  let local = uq % brickSize;
  return slot * brickSize * brickSize * brickSize + local.x + brickSize * (local.y + brickSize * local.z);
}
fn invocation(gid: vec3u) -> vec4u {
  let stream = gid.x + gid.y * activePages[1] * 256u;
  let voxels = params.fineDims.w * params.fineDims.w * params.fineDims.w;
  let activeIndex = stream / voxels;
  if (activeIndex >= activePages[0] || 4u + activeIndex >= arrayLength(&activePages)) { return vec4u(INVALID); }
  let pageIndex = activePages[4u + activeIndex];
  if (pageIndex >= params.brickDims.w) { return vec4u(INVALID); }
  let localIndex = stream - activeIndex * voxels;
  let local = vec3u(localIndex % params.fineDims.w, (localIndex / params.fineDims.w) % params.fineDims.w,
    localIndex / (params.fineDims.w * params.fineDims.w));
  let q = pageCoordinate(pageIndex) * params.fineDims.w + local;
  if (any(q >= params.fineDims.xyz)) { return vec4u(INVALID); }
  return vec4u(q, payloadIndex(vec3i(q)));
}
fn denseVelocity(position: vec3f) -> vec3f {
  let factor = f32(params.coarseDims.w);
  let p = clamp((position + vec3f(0.5)) / factor - vec3f(0.5), vec3f(0.0), vec3f(params.coarseDims.xyz - vec3u(1)));
  let a=vec3i(floor(p));let b=min(a+vec3i(1),vec3i(params.coarseDims.xyz)-vec3i(1));let t=fract(p);
  let v000=textureLoad(coarseVelocity,vec3i(a.x,a.y,a.z),0).xyz;let v100=textureLoad(coarseVelocity,vec3i(b.x,a.y,a.z),0).xyz;
  let v010=textureLoad(coarseVelocity,vec3i(a.x,b.y,a.z),0).xyz;let v110=textureLoad(coarseVelocity,vec3i(b.x,b.y,a.z),0).xyz;
  let v001=textureLoad(coarseVelocity,vec3i(a.x,a.y,b.z),0).xyz;let v101=textureLoad(coarseVelocity,vec3i(b.x,a.y,b.z),0).xyz;
  let v011=textureLoad(coarseVelocity,vec3i(a.x,b.y,b.z),0).xyz;let v111=textureLoad(coarseVelocity,vec3i(b.x,b.y,b.z),0).xyz;
  return mix(mix(mix(v000,v100,t.x),mix(v010,v110,t.x),t.y),mix(mix(v001,v101,t.x),mix(v011,v111,t.x),t.y),t.z);
}
fn residualCell(q: vec3i, useB: bool) -> vec3f {
  let index=payloadIndex(q);if(index==INVALID||index>=arrayLength(&residualA)||index>=arrayLength(&residualB)){return vec3f(0.0);}
  return select(residualA[index].xyz,residualB[index].xyz,useB);
}
fn residualAt(position: vec3f, useB: bool) -> vec3f {
  let p=clamp(position,vec3f(0.0),vec3f(params.fineDims.xyz-vec3u(1)));let a=vec3i(floor(p));let b=min(a+vec3i(1),vec3i(params.fineDims.xyz)-vec3i(1));let t=fract(p);
  let v000=residualCell(vec3i(a.x,a.y,a.z),useB);let v100=residualCell(vec3i(b.x,a.y,a.z),useB);
  let v010=residualCell(vec3i(a.x,b.y,a.z),useB);let v110=residualCell(vec3i(b.x,b.y,a.z),useB);
  let v001=residualCell(vec3i(a.x,a.y,b.z),useB);let v101=residualCell(vec3i(b.x,a.y,b.z),useB);
  let v011=residualCell(vec3i(a.x,b.y,b.z),useB);let v111=residualCell(vec3i(b.x,b.y,b.z),useB);
  return mix(mix(mix(v000,v100,t.x),mix(v010,v110,t.x),t.y),mix(mix(v001,v101,t.x),mix(v011,v111,t.x),t.y),t.z);
}
fn totalVelocity(q: vec3i) -> vec3f { return denseVelocity(vec3f(q)) + residualCell(q,true); }
fn phiCell(q: vec3i) -> f32 {
  let index=payloadIndex(q);if(index==INVALID||index>=arrayLength(&phi)){return 4.0*min(params.cellAndDt.x,min(params.cellAndDt.y,params.cellAndDt.z));}
  return phi[index];
}
fn normalAt(q: vec3i) -> vec3f {
  let h=params.cellAndDt.xyz/f32(params.coarseDims.w);
  let gradient=vec3f((phiCell(q+vec3i(1,0,0))-phiCell(q-vec3i(1,0,0)))/(2.0*h.x),
    (phiCell(q+vec3i(0,1,0))-phiCell(q-vec3i(0,1,0)))/(2.0*h.y),
    (phiCell(q+vec3i(0,0,1))-phiCell(q-vec3i(0,0,1)))/(2.0*h.z));
  return gradient/max(length(gradient),1e-6);
}
fn pressureCell(q: vec3i, useB: bool) -> f32 {
  let index=payloadIndex(q);if(index==INVALID||index>=arrayLength(&pressureA)||index>=arrayLength(&pressureB)){return 0.0;}
  return select(pressureA[index],pressureB[index],useB);
}
fn divergence(q: vec3i) -> f32 {
  let h=params.cellAndDt.xyz/f32(params.coarseDims.w);
  return (totalVelocity(q+vec3i(1,0,0)).x-totalVelocity(q-vec3i(1,0,0)).x)/(2.0*h.x)
    +(totalVelocity(q+vec3i(0,1,0)).y-totalVelocity(q-vec3i(0,1,0)).y)/(2.0*h.y)
    +(totalVelocity(q+vec3i(0,0,1)).z-totalVelocity(q-vec3i(0,0,1)).z)/(2.0*h.z);
}

@compute @workgroup_size(256)
fn advectResidualAToB(@builtin(global_invocation_id) gid: vec3u){
  let item=invocation(gid);if(item.w==INVALID){return;}let p=vec3f(item.xyz);
  let u0=denseVelocity(p)+residualAt(p,false);let h=params.cellAndDt.xyz/f32(params.coarseDims.w);
  let midpoint=p-0.5*u0*params.cellAndDt.w/h;let um=denseVelocity(midpoint)+residualAt(midpoint,false);
  residualB[item.w]=vec4f(residualAt(p-um*params.cellAndDt.w/h,false),0.0);
}

@compute @workgroup_size(256)
fn applySurfaceForce(@builtin(global_invocation_id) gid: vec3u){
  let item=invocation(gid);if(item.w==INVALID){return;}let q=vec3i(item.xyz);let h=params.cellAndDt.xyz/f32(params.coarseDims.w);let hMin=min(h.x,min(h.y,h.z));
  let value=phiCell(q);let band=1.5*hMin;if(abs(value)>=band||params.physical.y<=0.0){return;}
  let n=normalAt(q);let kappa=(normalAt(q+vec3i(1,0,0)).x-normalAt(q-vec3i(1,0,0)).x)/(2.0*h.x)
    +(normalAt(q+vec3i(0,1,0)).y-normalAt(q-vec3i(0,1,0)).y)/(2.0*h.y)
    +(normalAt(q+vec3i(0,0,1)).z-normalAt(q-vec3i(0,0,1)).z)/(2.0*h.z);
  let delta=0.5*(1.0+cos(3.14159265*value/band))/band;
  var dv=-params.cellAndDt.w*(params.physical.y/max(params.physical.x,1e-6))*kappa*delta*n;
  let maximumDv=0.5*hMin/max(params.cellAndDt.w,1e-6);let magnitude=length(dv);if(magnitude>maximumDv){dv*=maximumDv/magnitude;}
  residualB[item.w]=vec4f(residualB[item.w].xyz+dv,0.0);
}

fn jacobi(q: vec3i,useB:bool)->f32{
  let h=params.cellAndDt.xyz/f32(params.coarseDims.w);let wx=1.0/(h.x*h.x);let wy=1.0/(h.y*h.y);let wz=1.0/(h.z*h.z);
  let rhs=params.physical.x/max(params.cellAndDt.w,1e-6)*divergence(q);
  let sum=wx*(pressureCell(q+vec3i(1,0,0),useB)+pressureCell(q-vec3i(1,0,0),useB))
    +wy*(pressureCell(q+vec3i(0,1,0),useB)+pressureCell(q-vec3i(0,1,0),useB))
    +wz*(pressureCell(q+vec3i(0,0,1),useB)+pressureCell(q-vec3i(0,0,1),useB));
  return (sum-rhs)/(2.0*(wx+wy+wz));
}
@compute @workgroup_size(256)
fn jacobiAToB(@builtin(global_invocation_id) gid:vec3u){let item=invocation(gid);if(item.w!=INVALID){pressureB[item.w]=jacobi(vec3i(item.xyz),false);}}
@compute @workgroup_size(256)
fn jacobiBToA(@builtin(global_invocation_id) gid:vec3u){let item=invocation(gid);if(item.w!=INVALID){pressureA[item.w]=jacobi(vec3i(item.xyz),true);}}

@compute @workgroup_size(256)
fn projectResidual(@builtin(global_invocation_id) gid:vec3u){
  let item=invocation(gid);if(item.w==INVALID){return;}let q=vec3i(item.xyz);let h=params.cellAndDt.xyz/f32(params.coarseDims.w);
  let gradient=vec3f((pressureCell(q+vec3i(1,0,0),false)-pressureCell(q-vec3i(1,0,0),false))/(2.0*h.x),
    (pressureCell(q+vec3i(0,1,0),false)-pressureCell(q-vec3i(0,1,0),false))/(2.0*h.y),
    (pressureCell(q+vec3i(0,0,1),false)-pressureCell(q-vec3i(0,0,1),false))/(2.0*h.z));
  residualB[item.w]=vec4f(residualB[item.w].xyz-params.cellAndDt.w/max(params.physical.x,1e-6)*gradient,0.0);
}
@compute @workgroup_size(256)
fn commitResidual(@builtin(global_invocation_id) gid:vec3u){
  let item=invocation(gid);if(item.w==INVALID){return;}let q=vec3i(item.xyz);
  let h=params.cellAndDt.xyz/f32(params.coarseDims.w);let hMin=min(h.x,min(h.y,h.z));
  var correction=residualB[item.w].xyz;
  // The residual represents unresolved capillary-scale motion, not a second
  // bulk velocity solve. Limit it by twice the physical capillary-wave speed
  // and taper it smoothly to the coarse Dirichlet state across the support
  // halo. This prevents a truncated local pressure solve from accumulating a
  // page-scale discontinuity while retaining fine ripple velocities at phi=0.
  let capillarySpeed=sqrt(max(params.physical.y,0.0)/max(params.physical.x*hMin,1e-8));
  let speedLimit=max(0.02,2.0*capillarySpeed);
  let magnitude=length(correction);
  if(magnitude>speedLimit){correction*=speedLimit/magnitude;}
  let inner=1.5*hMin;let outer=max(inner+0.5*hMin,max(params.settings.x,params.settings.y)*hMin);
  correction*=1.0-smoothstep(inner,outer,abs(phiCell(q)));
  residualA[item.w]=vec4f(correction,0.0);
}
`;

function buffer(
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBufferView<ArrayBuffer>,
) {
  const result = device.createBuffer({ label, size: Math.max(4, size), usage });
  if (data?.byteLength) device.queue.writeBuffer(result, 0, data);
  return result;
}

export class WebGPUSparseSurfaceBand {
  readonly plan: SparseSurfaceBandPlan;
  readonly allocatedBytes: number;
  readonly mode: SparseSurfaceBandMode;

  private readonly pageTable: GPUBuffer;
  private readonly states: GPUBuffer;
  private readonly freeList: GPUBuffer;
  private readonly control: GPUBuffer;
  private readonly activePages: GPUBuffer;
  private readonly phiA: GPUBuffer;
  private readonly phiB: GPUBuffer;
  private readonly sizingA: GPUBuffer;
  private readonly sizingB: GPUBuffer;
  private readonly velocity: GPUBuffer;
  private readonly velocityB: GPUBuffer;
  private readonly pressureA: GPUBuffer;
  private readonly pressureB: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly residencyGroup: GPUBindGroup;
  private readonly fieldGroup: GPUBindGroup;
  private readonly dynamicsGroup?: GPUBindGroup;
  private readonly residencyPipelines: Record<"reset" | "evaluateSizing" | "propagateBA" | "propagateAB" | "classify" | "retire" | "allocateCore3" | "allocateCore2" | "allocateCore1" | "allocateCore0" | "allocateHalo" | "finalize", GPUComputePipeline>;
  private readonly fieldPipelines: Record<"mirror" | "initialize" | "advect" | "copyBA", GPUComputePipeline>;
  private readonly dynamicsPipelines?: Record<"advect" | "force" | "jacobiAB" | "jacobiBA" | "project" | "commit", GPUComputePipeline>;
  private currentPhi: "a" | "b" = "a";
  private generation = 0;
  private destroyed = false;
  private statsReadback?: GPUBuffer;
  private statsReadbackBusy = false;
  private readonly options: SparseSurfaceBandOptions;

  constructor(
    private readonly device: GPUDevice,
    coarseDimensions: readonly [number, number, number],
    private readonly coarseCellSize: readonly [number, number, number],
    private readonly coarseLevelSet: GPUTexture,
    private readonly coarseVelocity: GPUTexture,
    private readonly coarseSolids: GPUBuffer,
    options: SparseSurfaceBandOptions = {},
  ) {
    // The vector residual is the largest single payload (vec4f per voxel).
    // Clamp the page pool to the adapter's binding and buffer limits before
    // allocating so factor-4 requests remain bounded instead of causing a
    // device loss on GPUs with the common 128 MiB storage-binding ceiling.
    const configuredBrickSize = options.brickSize ?? 8;
    const largestBytesPerPage = configuredBrickSize ** 3 * (options.fineDynamics ? 16 : 4);
    const maximumPagesByDevice = Math.max(1, Math.floor(Math.min(
      Number(device.limits.maxStorageBufferBindingSize),
      Number(device.limits.maxBufferSize),
    ) / largestBytesPerPage));
    const requestedMaximum = options.maximumPages ?? Number.POSITIVE_INFINITY;
    const effectiveOptions = { ...options, maximumPages: Math.min(requestedMaximum, maximumPagesByDevice) };
    this.plan = planSparseSurfaceBand(coarseDimensions, effectiveOptions);
    this.mode = options.mode ?? "mirror";
    this.options = effectiveOptions;
    coarseCellSize.forEach((value) => { if (!(value > 0) || !Number.isFinite(value)) throw new RangeError("Surface cell size must be positive and finite"); });
    const pageTableData = new Uint32Array(this.plan.logicalPageCount); pageTableData.fill(SPARSE_SURFACE_INVALID_PAGE);
    const freeData = Uint32Array.from({ length: this.plan.physicalPageCapacity }, (_, index) => index);
    const controlData = new Uint32Array(SPARSE_SURFACE_CONTROL_BYTES / 4);
    controlData[0] = this.plan.physicalPageCapacity;
    const storageCopy = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.pageTable = buffer(device, "Sparse surface logical page table", pageTableData.byteLength, storageCopy, pageTableData);
    this.states = buffer(device, "Sparse surface logical page states", this.plan.logicalPageCount * 4, storageCopy);
    this.freeList = buffer(device, "Sparse surface free page slots", freeData.byteLength, storageCopy, freeData);
    this.control = buffer(device, "Sparse surface allocator control", SPARSE_SURFACE_CONTROL_BYTES, storageCopy, controlData);
    this.activePages = buffer(device, "Sparse surface active pages and dispatch", (4 + this.plan.physicalPageCapacity) * 4,
      storageCopy | GPUBufferUsage.INDIRECT);
    const scalarBytes = this.plan.physicalPageCapacity * this.plan.voxelsPerPage * 4;
    const dynamicsBytes = this.options.fineDynamics ? scalarBytes : 4;
    this.phiA = buffer(device, "Sparse surface phi A", scalarBytes, storageCopy);
    this.phiB = buffer(device, "Sparse surface phi B", scalarBytes, storageCopy);
    const coarseSizingBytes = coarseDimensions[0] * coarseDimensions[1] * coarseDimensions[2] * 4;
    this.sizingA = buffer(device, "Sparse surface advected sizing A", coarseSizingBytes, storageCopy);
    this.sizingB = buffer(device, "Sparse surface advected sizing B", coarseSizingBytes, storageCopy);
    this.velocity = buffer(device, "Sparse surface velocity residual A", dynamicsBytes * 4, storageCopy);
    this.velocityB = buffer(device, "Sparse surface velocity residual B", dynamicsBytes * 4, storageCopy);
    this.pressureA = buffer(device, "Sparse surface pressure correction A", dynamicsBytes, storageCopy);
    this.pressureB = buffer(device, "Sparse surface pressure correction B", dynamicsBytes, storageCopy);
    this.params = buffer(device, "Sparse surface parameters", 112, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

    const residencyLayout = device.createBindGroupLayout({ label: "Sparse surface residency layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      ...[2,3,4,5,6].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })),
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ] });
    const fieldLayout = device.createBindGroupLayout({ label: "Sparse surface field layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ] });
    const residencyModule = device.createShaderModule({ label: "Sparse surface residency", code: sparseSurfaceResidencyShader });
    const fieldModule = device.createShaderModule({ label: "Sparse surface field", code: sparseSurfaceFieldShader });
    const residencyPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [residencyLayout] });
    const fieldPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [fieldLayout] });
    const residency = (label: string, entryPoint: string) => device.createComputePipeline({ label, layout: residencyPipelineLayout, compute: { module: residencyModule, entryPoint } });
    const field = (label: string, entryPoint: string) => device.createComputePipeline({ label, layout: fieldPipelineLayout, compute: { module: fieldModule, entryPoint } });
    this.residencyPipelines = {
      reset: residency("Reset sparse surface residency", "resetCounters"),
      evaluateSizing: residency("Evaluate advected sparse surface sizing", "evaluateSizing"),
      propagateBA: residency("Propagate sparse surface sizing B to A", "propagateBToA"),
      propagateAB: residency("Propagate sparse surface sizing A to B", "propagateAToB"),
      classify: residency("Classify sparse surface pages", "classifyDesired"),
      retire: residency("Retire sparse surface pages", "retirePages"),
      allocateCore3: residency("Allocate highest-priority sparse surface cores", "allocateCore3"),
      allocateCore2: residency("Allocate high-priority sparse surface cores", "allocateCore2"),
      allocateCore1: residency("Allocate medium-priority sparse surface cores", "allocateCore1"),
      allocateCore0: residency("Allocate low-priority sparse surface cores", "allocateCore0"),
      allocateHalo: residency("Allocate sparse surface stencil halos", "allocateHalo"),
      finalize: residency("Finalize sparse surface dispatch", "finalizeDispatch"),
    };
    this.fieldPipelines = {
      mirror: field("Mirror dense field into sparse surface", "mirrorDense"),
      initialize: field("Initialize activated sparse surface pages", "initializeActivated"),
      advect: field("Advect sparse surface", "advectAToB"),
      copyBA: field("Commit sparse surface B to A", "copyBToA"),
    };
    this.residencyGroup = device.createBindGroup({ label: "Sparse surface residency bindings", layout: residencyLayout, entries: [
      { binding: 0, resource: coarseLevelSet.createView() }, { binding: 1, resource: coarseVelocity.createView() },
      { binding: 2, resource: { buffer: this.pageTable } }, { binding: 3, resource: { buffer: this.states } },
      { binding: 4, resource: { buffer: this.freeList } }, { binding: 5, resource: { buffer: this.control } },
      { binding: 6, resource: { buffer: this.activePages } }, { binding: 7, resource: { buffer: this.params } },
      { binding: 8, resource: { buffer: this.sizingA } }, { binding: 9, resource: { buffer: this.sizingB } },
      { binding: 10, resource: { buffer: this.coarseSolids } },
    ] });
    this.fieldGroup = device.createBindGroup({ label: "Sparse surface field bindings", layout: fieldLayout, entries: [
      { binding: 0, resource: coarseLevelSet.createView() }, { binding: 1, resource: coarseVelocity.createView() },
      { binding: 2, resource: { buffer: this.pageTable } }, { binding: 3, resource: { buffer: this.states } },
      { binding: 4, resource: { buffer: this.activePages } }, { binding: 5, resource: { buffer: this.phiA } },
      { binding: 6, resource: { buffer: this.phiB } }, { binding: 7, resource: { buffer: this.velocity } },
      { binding: 8, resource: { buffer: this.params } },
    ] });
    // The production path intentionally has no page-local pressure solver: it
    // consumes the existing global octree projection, preserving its 32-pass
    // Chebyshev dispatch. Compile and bind the research dynamics only when an
    // explicit caller opts into it, so the default pays neither its pipelines
    // nor its per-page velocity/pressure payload.
    if (this.options.fineDynamics) {
      const dynamicsLayout = device.createBindGroupLayout({ label: "Sparse surface dynamics layout", entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        ...[4,5,6,7].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })),
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ] });
      const dynamicsModule = device.createShaderModule({ label: "Sparse surface dynamics", code: sparseSurfaceDynamicsShader });
      const dynamicsPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [dynamicsLayout] });
      const dynamics = (label: string, entryPoint: string) => device.createComputePipeline({ label, layout: dynamicsPipelineLayout, compute: { module: dynamicsModule, entryPoint } });
      this.dynamicsPipelines = {
        advect: dynamics("Advect fine velocity residual", "advectResidualAToB"),
        force: dynamics("Apply fine capillary surface force", "applySurfaceForce"),
        jacobiAB: dynamics("Fine pressure correction A to B", "jacobiAToB"),
        jacobiBA: dynamics("Fine pressure correction B to A", "jacobiBToA"),
        project: dynamics("Project fine velocity residual", "projectResidual"),
        commit: dynamics("Commit fine velocity residual", "commitResidual"),
      };
      this.dynamicsGroup = device.createBindGroup({ label: "Sparse surface dynamics bindings", layout: dynamicsLayout, entries: [
        { binding: 0, resource: coarseVelocity.createView() }, { binding: 1, resource: { buffer: this.pageTable } },
        { binding: 2, resource: { buffer: this.activePages } }, { binding: 3, resource: { buffer: this.phiA } },
        { binding: 4, resource: { buffer: this.velocity } }, { binding: 5, resource: { buffer: this.velocityB } },
        { binding: 6, resource: { buffer: this.pressureA } }, { binding: 7, resource: { buffer: this.pressureB } },
        { binding: 8, resource: { buffer: this.params } },
      ] });
    }
    this.allocatedBytes = pageTableData.byteLength + this.plan.logicalPageCount * 4 + freeData.byteLength + 64
      + (4 + this.plan.physicalPageCapacity) * 4 + this.plan.allocatedPayloadBytes + 2 * coarseSizingBytes + 112;
    this.writeParams(0, this.options);
  }

  private writeParams(dt_s: number, options: SparseSurfaceBandOptions) {
    const data = new ArrayBuffer(112), u = new Uint32Array(data), f = new Float32Array(data);
    const p = this.plan;
    u.set([p.coarseDimensions[0], p.coarseDimensions[1], p.coarseDimensions[2], p.refinementFactor], 0);
    u.set([p.fineDimensions[0], p.fineDimensions[1], p.fineDimensions[2], p.brickSize], 4);
    u.set([p.brickDimensions[0], p.brickDimensions[1], p.brickDimensions[2], p.logicalPageCount], 8);
    f.set([
      Math.max(1, options.bandCells ?? 4), Math.max(0, options.stencilCells ?? 5),
      Math.max(0, Math.min(0xffff, Math.round(options.retireAfterFrames ?? 3))), this.mode === "authoritative" ? 1 : 0,
    ], 12);
    f.set([this.coarseCellSize[0], this.coarseCellSize[1], this.coarseCellSize[2], Math.max(0, dt_s)], 16);
    // Paper sizing weights are mirrored here for diagnostics/shader ABI; the
    // production residency shader uses gamma_phi=4 and gamma_u=3 uniformly.
    f.set([4, 3, 0, p.physicalPageCapacity], 20);
    f.set([
      Math.max(1e-6, options.density_kg_m3 ?? 1_000), Math.max(0, options.surfaceTension_N_m ?? 0),
      Math.max(2, Math.min(32, Math.round(options.pressureIterations ?? 12))), options.fineDynamics ? 1 : 0,
    ], 24);
    this.device.queue.writeBuffer(this.params, 0, data);
  }

  encode(encoder: GPUCommandEncoder, dt_s: number, options: Pick<SparseSurfaceBandOptions, "bandCells" | "stencilCells" | "retireAfterFrames"> = {}) {
    if (this.destroyed) return;
    this.writeParams(dt_s, { ...this.options, ...options });
    const pages = Math.ceil(this.plan.logicalPageCount / 64);
    const coarseCells = Math.ceil((this.plan.coarseDimensions[0] * this.plan.coarseDimensions[1] * this.plan.coarseDimensions[2]) / 64);
    const residency = encoder.beginComputePass({ label: "Sparse surface page lifecycle" });
    residency.setBindGroup(0, this.residencyGroup);
    residency.setPipeline(this.residencyPipelines.reset); residency.dispatchWorkgroups(1);
    residency.setPipeline(this.residencyPipelines.evaluateSizing); residency.dispatchWorkgroups(coarseCells);
    // The paper limits propagation to five updates per cell to avoid excessive
    // diffusion. Five ping-pong passes finish in A, which becomes next frame's
    // advected history and this frame's page-classification authority.
    residency.setPipeline(this.residencyPipelines.propagateBA); residency.dispatchWorkgroups(coarseCells);
    residency.setPipeline(this.residencyPipelines.propagateAB); residency.dispatchWorkgroups(coarseCells);
    residency.setPipeline(this.residencyPipelines.propagateBA); residency.dispatchWorkgroups(coarseCells);
    residency.setPipeline(this.residencyPipelines.propagateAB); residency.dispatchWorkgroups(coarseCells);
    residency.setPipeline(this.residencyPipelines.propagateBA); residency.dispatchWorkgroups(coarseCells);
    residency.setPipeline(this.residencyPipelines.classify); residency.dispatchWorkgroups(pages);
    residency.setPipeline(this.residencyPipelines.retire); residency.dispatchWorkgroups(pages);
    residency.setPipeline(this.residencyPipelines.allocateCore3); residency.dispatchWorkgroups(pages);
    residency.setPipeline(this.residencyPipelines.allocateCore2); residency.dispatchWorkgroups(pages);
    residency.setPipeline(this.residencyPipelines.allocateCore1); residency.dispatchWorkgroups(pages);
    residency.setPipeline(this.residencyPipelines.allocateCore0); residency.dispatchWorkgroups(pages);
    residency.setPipeline(this.residencyPipelines.allocateHalo); residency.dispatchWorkgroups(pages);
    residency.setPipeline(this.residencyPipelines.finalize); residency.dispatchWorkgroups(1);
    residency.end();
    const field = encoder.beginComputePass({ label: this.mode === "mirror" ? "Mirror sparse surface band" : "Advance sparse surface band" });
    field.setBindGroup(0, this.fieldGroup);
    if (this.mode === "mirror") {
      field.setPipeline(this.fieldPipelines.mirror);
      field.dispatchWorkgroupsIndirect(this.activePages, SPARSE_SURFACE_ACTIVE_DISPATCH_OFFSET_BYTES);
      this.currentPhi = "a";
    } else {
      field.setPipeline(this.fieldPipelines.initialize);
      field.dispatchWorkgroupsIndirect(this.activePages, SPARSE_SURFACE_ACTIVE_DISPATCH_OFFSET_BYTES);
      field.setPipeline(this.fieldPipelines.advect);
      field.dispatchWorkgroupsIndirect(this.activePages, SPARSE_SURFACE_ACTIVE_DISPATCH_OFFSET_BYTES);
      // Commit the bounded hierarchical transport result. Dense coarse phi is
      // the signed-distance anchor; repeatedly redistancing a tiny resident
      // patch here erodes the sub-cell crests and troughs it is meant to keep.
      field.setPipeline(this.fieldPipelines.copyBA);
      field.dispatchWorkgroupsIndirect(this.activePages, SPARSE_SURFACE_ACTIVE_DISPATCH_OFFSET_BYTES);
      this.currentPhi = "a";
    }
    field.end();
    const dynamicsGroup = this.dynamicsGroup;
    const dynamicsPipelines = this.dynamicsPipelines;
    if (this.mode === "authoritative" && dynamicsGroup && dynamicsPipelines) {
      // The correction pressure has a deterministic zero initial condition on
      // every substep. Missing pages also read zero, so the residual converges
      // to the coarse Dirichlet velocity at the narrow-band boundary.
      encoder.clearBuffer(this.pressureA);
      encoder.clearBuffer(this.pressureB);
      const dynamics = encoder.beginComputePass({ label: "Sparse fine-band velocity and pressure correction" });
      dynamics.setBindGroup(0, dynamicsGroup);
      const dispatch = (pipeline: GPUComputePipeline) => {
        dynamics.setPipeline(pipeline);
        dynamics.dispatchWorkgroupsIndirect(this.activePages, SPARSE_SURFACE_ACTIVE_DISPATCH_OFFSET_BYTES);
      };
      dispatch(dynamicsPipelines.advect);
      dispatch(dynamicsPipelines.force);
      const requestedIterations = Math.max(2, Math.min(32, Math.round(this.options.pressureIterations ?? 12)));
      const iterations = requestedIterations + requestedIterations % 2;
      for (let iteration = 0; iteration < iterations; iteration += 1) {
        dispatch(iteration % 2 === 0 ? dynamicsPipelines.jacobiAB : dynamicsPipelines.jacobiBA);
      }
      dispatch(dynamicsPipelines.project);
      dispatch(dynamicsPipelines.commit);
      dynamics.end();
    }
    this.generation += 1;
  }

  get source(): SparseSurfaceBandGPUSource {
    return {
      mode: this.mode,
      pageTable: { buffer: this.pageTable }, activePages: { buffer: this.activePages },
      states: { buffer: this.states },
      phi: { buffer: this.currentPhi === "a" ? this.phiA : this.phiB }, velocity: { buffer: this.velocity },
      params: { buffer: this.params }, coarseLevelSet: this.coarseLevelSet, coarseVelocity: this.coarseVelocity,
      control: { buffer: this.control },
      fineDimensions: this.plan.fineDimensions, brickDimensions: this.plan.brickDimensions,
      brickSize: this.plan.brickSize, refinementFactor: this.plan.refinementFactor,
      pageCapacity: this.plan.physicalPageCapacity, revision: this.generation,
    };
  }

  /** Geometry-only semi-Lagrangian phi transport is not a CFL constraint.
   * Only the opt-in explicit fine dynamics requires the fine-cell capillary
   * and velocity timestep; the production path retains the coarse solver dt. */
  get requiresFineTimestep() { return Boolean(this.options.fineDynamics); }

  async readStats(): Promise<SparseSurfaceBandStats> {
    if (this.destroyed) return { resident: 0, core: 0, halo: 0, activated: 0, retired: 0, overflow: 0, free: 0, peakResident: 0, generation: 0, logicalPageCount: this.plan.logicalPageCount, physicalPageCapacity: this.plan.physicalPageCapacity };
    // The pooled staging buffer cannot be copied into again while mapped or
    // pending; overlapping callers fall back to a transient buffer.
    const pooled = !this.statsReadbackBusy;
    const readback = pooled
      ? (this.statsReadback ??= this.device.createBuffer({ label: "Sparse surface stats readback", size: SPARSE_SURFACE_CONTROL_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }))
      : this.device.createBuffer({ label: "Sparse surface stats readback (transient)", size: SPARSE_SURFACE_CONTROL_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    if (pooled) this.statsReadbackBusy = true;
    const encoder = this.device.createCommandEncoder({ label: "Read sparse surface stats" });
    encoder.copyBufferToBuffer(this.control, 0, readback, 0, SPARSE_SURFACE_CONTROL_BYTES);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange(0, SPARSE_SURFACE_CONTROL_BYTES));
      return {
        free: words[0], generation: words[1], overflow: words[2], activated: words[3], retired: words[4],
        resident: words[5], core: words[6], halo: words[7], peakResident: words[8],
        logicalPageCount: this.plan.logicalPageCount, physicalPageCapacity: this.plan.physicalPageCapacity,
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      if (pooled) this.statsReadbackBusy = false;
      else readback.destroy();
    }
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.statsReadback?.destroy();
    for (const resource of [this.pageTable, this.states, this.freeList, this.control, this.activePages, this.phiA, this.phiB, this.sizingA, this.sizingB, this.velocity, this.velocityB, this.pressureA, this.pressureB, this.params]) resource.destroy();
  }
}
