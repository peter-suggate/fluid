import {
  SPARSE_BRICK_GPU_LAYOUT,
  SPARSE_BRICK_INVALID_INDEX,
} from "./sparse-brick-octree";
import { FLUID_BRICK_RESIDENT } from "./webgpu-fluid-brick-residency";
import type { SvoVec3 } from "./webgpu-svo-traversal";
import {
  SPARSE_VOXEL_PUBLICATION_STATE,
  SPARSE_VOXEL_VALID_FIELDS,
  type SparseVoxelStructuralRenderSource,
} from "./webgpu-voxel-debug";

export const SVO_STRUCTURAL_POINT_LOOKUP_MAX_VISITS = 24;
export const SVO_STRUCTURAL_CLEARED_PHI_MIN = 3e38;

export interface SvoStructuralFluidPackedFixture {
  /** Existing 32-u32 sparse control block. */
  control: Uint32Array;
  /** Eight-u32 node records, with the structural binding offset already applied. */
  nodes: Uint32Array;
  /** Four-u32 leaf records, with the structural binding offset already applied. */
  leaves: Uint32Array;
  /** Existing vec4f geometry records; channel x is coarse fluid phi in metres. */
  geometry: Float32Array;
  /** Per-leaf flags from GPUFluidBrickResidency. */
  fluidLeafStates: Uint32Array;
  /** Existing eight-u32 structural publication state. */
  publicationState: Uint32Array;
  domain: SparseVoxelStructuralRenderSource["domain"];
  /** Optional cached consumer fence. A mismatch is invalid, never a miss. */
  expectedCompleteGeneration?: number;
}

export type SvoStructuralFluidInvalidReason =
  | "publication-state-too-small"
  | "unpublished-generation"
  | "generation-mismatch"
  | "missing-valid-fields"
  | "source-overflow"
  | "invalid-layout"
  | "invalid-topology"
  | "work-exhausted"
  | "nonresident-leaf"
  | "invalid-payload";

export type SvoStructuralFluidMissReason = "outside-domain" | "missing-branch";

export type SvoStructuralFluidCellLookup =
  | {
    status: "valid";
    phi_m: number;
    cell: readonly [number, number, number];
    nodeIndex: number;
    leafIndex: number;
    voxelIndex: number;
    nodeVisits: number;
  }
  | { status: "miss"; reason: SvoStructuralFluidMissReason; nodeVisits: number }
  | { status: "invalid"; reason: SvoStructuralFluidInvalidReason; nodeVisits: number };

export interface SvoStructuralFineFluidSample {
  phi_m: number;
  valid: boolean;
}

export type SvoStructuralFineFluidSampler = (worldPosition_m: SvoVec3) => SvoStructuralFineFluidSample | undefined;

export type SvoStructuralExclusiveFluidSample =
  | { status: "valid"; owner: "fine" | "coarse"; phi_m: number; coarse?: Extract<SvoStructuralFluidCellLookup, { status: "valid" }> }
  | { status: "miss"; owner: "none"; reason: SvoStructuralFluidMissReason }
  | { status: "invalid"; owner: "none"; reason: SvoStructuralFluidInvalidReason | "invalid-fine-sample" };

export type SvoStructuralFluidGradient =
  | {
    status: "valid";
    gradient: SvoVec3;
    normal: SvoVec3 | null;
    schemes: readonly ["central" | "forward" | "backward", "central" | "forward" | "backward", "central" | "forward" | "backward"];
    center: Extract<SvoStructuralFluidCellLookup, { status: "valid" }>;
  }
  | { status: "miss"; reason: SvoStructuralFluidMissReason }
  | { status: "invalid"; reason: SvoStructuralFluidInvalidReason | "insufficient-neighbors" };

const NODE_WORDS = SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes / Uint32Array.BYTES_PER_ELEMENT;
const LEAF_WORDS = SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes / Uint32Array.BYTES_PER_ELEMENT;
const GEOMETRY_FLOATS = SPARSE_BRICK_GPU_LAYOUT.geometryStrideBytes / Float32Array.BYTES_PER_ELEMENT;
const MAXIMUM_DEPTH = 21;

function finiteVec3(value: readonly number[], label: string): void {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${label} must contain three finite components`);
  }
}

function validateDomain(domain: SparseVoxelStructuralRenderSource["domain"]): void {
  finiteVec3(domain.worldOrigin_m, "Structural world origin");
  finiteVec3(domain.cellSize_m, "Structural cell size");
  if (domain.cellSize_m.some((component) => component <= 0)) throw new RangeError("Structural cell size must be positive");
  if (domain.dimensionsCells.some((component) => !Number.isSafeInteger(component) || component < 1)) {
    throw new RangeError("Structural cell dimensions must be positive safe integers");
  }
  if (domain.brickSize !== 4 && domain.brickSize !== 8) throw new RangeError("Structural brick size must be 4 or 8");
  if (!Number.isInteger(domain.maximumDepth) || domain.maximumDepth < 0 || domain.maximumDepth > MAXIMUM_DEPTH) {
    throw new RangeError(`Structural maximum depth must be 0..${MAXIMUM_DEPTH}`);
  }
}

function invalid(reason: SvoStructuralFluidInvalidReason, nodeVisits = 0): SvoStructuralFluidCellLookup {
  return { status: "invalid", reason, nodeVisits };
}

function validatePublication(source: SvoStructuralFluidPackedFixture): SvoStructuralFluidCellLookup | null {
  validateDomain(source.domain);
  if (source.publicationState.length * Uint32Array.BYTES_PER_ELEMENT < SPARSE_VOXEL_PUBLICATION_STATE.strideBytes) {
    return invalid("publication-state-too-small");
  }
  const generation = source.publicationState[SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration];
  if (generation === 0) return invalid("unpublished-generation");
  if (source.expectedCompleteGeneration !== undefined && generation !== (source.expectedCompleteGeneration >>> 0)) {
    return invalid("generation-mismatch");
  }
  const required = SPARSE_VOXEL_VALID_FIELDS.topology | SPARSE_VOXEL_VALID_FIELDS.coarseFluid;
  const validFields = source.publicationState[SPARSE_VOXEL_PUBLICATION_STATE.validFields];
  if ((validFields & required) !== required) return invalid("missing-valid-fields");
  if (source.control.length <= SPARSE_BRICK_GPU_LAYOUT.controlWords.overflowFlags) return invalid("invalid-layout");
  if (source.control[SPARSE_BRICK_GPU_LAYOUT.controlWords.overflowFlags] !== 0) return invalid("source-overflow");
  if (source.nodes.length % NODE_WORDS !== 0 || source.leaves.length % LEAF_WORDS !== 0 || source.geometry.length % GEOMETRY_FLOATS !== 0) {
    return invalid("invalid-layout");
  }
  if (source.control[SPARSE_BRICK_GPU_LAYOUT.controlWords.brickSize] !== source.domain.brickSize) return invalid("invalid-layout");
  const publishedNodes = source.control[SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedNodes];
  const publishedLeaves = source.control[SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedLeaves];
  const publishedVoxels = source.control[SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedVoxels];
  if (publishedNodes > source.nodes.length / NODE_WORDS || publishedLeaves > source.leaves.length / LEAF_WORDS
      || publishedVoxels > source.geometry.length / GEOMETRY_FLOATS || publishedLeaves > source.fluidLeafStates.length) {
    return invalid("invalid-layout");
  }
  if (publishedNodes === 0) return invalid("invalid-topology");
  return null;
}

function popcountBefore(mask: number, octant: number): number {
  let bits = octant === 0 ? 0 : mask & ((1 << octant) - 1);
  let count = 0;
  while (bits !== 0) { count += bits & 1; bits >>>= 1; }
  return count;
}

function mortonBit(low: number, high: number, bit: number): number {
  return bit < 32 ? (low >>> bit) & 1 : (high >>> (bit - 32)) & 1;
}

function decodeMorton(low: number, high: number, level: number): readonly [number, number, number] {
  const coordinate = [0, 0, 0];
  for (let bit = 0; bit < level; bit += 1) {
    const scale = 2 ** bit;
    coordinate[0] += mortonBit(low, high, bit * 3) * scale;
    coordinate[1] += mortonBit(low, high, bit * 3 + 1) * scale;
    coordinate[2] += mortonBit(low, high, bit * 3 + 2) * scale;
  }
  return coordinate as [number, number, number];
}

function pointToCell(source: SvoStructuralFluidPackedFixture, position_m: SvoVec3): readonly [number, number, number] | null {
  finiteVec3(position_m, "Structural sample position");
  const cell = position_m.map((component, axis) => Math.floor(
    (component - source.domain.worldOrigin_m[axis]) / source.domain.cellSize_m[axis],
  )) as [number, number, number];
  return cell.some((component, axis) => component < 0 || component >= source.domain.dimensionsCells[axis]) ? null : cell;
}

/** Bounded packed-topology lookup for one finest-grid cell coordinate. */
export function lookupSvoStructuralCoarseFluidCell(
  source: SvoStructuralFluidPackedFixture,
  cell: readonly [number, number, number],
): SvoStructuralFluidCellLookup {
  const publicationFailure = validatePublication(source);
  if (publicationFailure) return publicationFailure;
  if (cell.some((component) => !Number.isSafeInteger(component))) throw new RangeError("Structural cell coordinate must contain integers");
  if (cell.some((component, axis) => component < 0 || component >= source.domain.dimensionsCells[axis])) {
    return { status: "miss", reason: "outside-domain", nodeVisits: 0 };
  }
  const publishedNodes = source.control[SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedNodes];
  const publishedLeaves = source.control[SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedLeaves];
  const publishedVoxels = source.control[SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedVoxels];
  const finestBrick = cell.map((component) => Math.floor(component / source.domain.brickSize)) as [number, number, number];
  let nodeIndex = 0;
  for (let visits = 1; visits <= SVO_STRUCTURAL_POINT_LOOKUP_MAX_VISITS; visits += 1) {
    if (nodeIndex >= publishedNodes) return invalid("invalid-topology", visits - 1);
    const nodeBase = nodeIndex * NODE_WORDS;
    const level = source.nodes[nodeBase + 2];
    if (level > source.domain.maximumDepth) return invalid("invalid-topology", visits);
    const leafIndex = source.nodes[nodeBase + 6];
    if (leafIndex !== SPARSE_BRICK_INVALID_INDEX) {
      if (leafIndex >= publishedLeaves) return invalid("invalid-topology", visits);
      const leafBase = leafIndex * LEAF_WORDS;
      if (source.leaves[leafBase] !== nodeIndex) return invalid("invalid-topology", visits);
      if ((source.fluidLeafStates[leafIndex] & FLUID_BRICK_RESIDENT) === 0) return invalid("nonresident-leaf", visits);
      const nodeCoordinate = decodeMorton(source.nodes[nodeBase], source.nodes[nodeBase + 1], level);
      const scale = 2 ** (source.domain.maximumDepth - level);
      const nodeOrigin = nodeCoordinate.map((component) => component * source.domain.brickSize * scale);
      const local = cell.map((component, axis) => Math.floor((component - nodeOrigin[axis]) / scale)) as [number, number, number];
      if (local.some((component) => component < 0 || component >= source.domain.brickSize)) return invalid("invalid-topology", visits);
      const voxelIndex = source.leaves[leafBase + 1] + local[0]
        + local[1] * source.domain.brickSize + local[2] * source.domain.brickSize ** 2;
      if (voxelIndex >= publishedVoxels || voxelIndex >= source.geometry.length / GEOMETRY_FLOATS) return invalid("invalid-payload", visits);
      const phi_m = source.geometry[voxelIndex * GEOMETRY_FLOATS];
      if (!Number.isFinite(phi_m) || Math.abs(phi_m) >= SVO_STRUCTURAL_CLEARED_PHI_MIN) return invalid("invalid-payload", visits);
      return { status: "valid", phi_m, cell: [...cell] as [number, number, number], nodeIndex, leafIndex, voxelIndex, nodeVisits: visits };
    }
    if (level >= source.domain.maximumDepth) return invalid("invalid-topology", visits);
    const shift = source.domain.maximumDepth - level - 1;
    const octant = ((Math.floor(finestBrick[0] / 2 ** shift) & 1)
      | ((Math.floor(finestBrick[1] / 2 ** shift) & 1) << 1)
      | ((Math.floor(finestBrick[2] / 2 ** shift) & 1) << 2));
    const childMask = source.nodes[nodeBase + 3] & 0xff;
    if ((childMask & (1 << octant)) === 0) return { status: "miss", reason: "missing-branch", nodeVisits: visits };
    const firstChild = source.nodes[nodeBase + 4];
    const childCount = source.nodes[nodeBase + 5];
    if (firstChild === SPARSE_BRICK_INVALID_INDEX || childCount !== popcountBefore(childMask, 8)) {
      return invalid("invalid-topology", visits);
    }
    const childIndex = firstChild + popcountBefore(childMask, octant);
    if (childIndex >= publishedNodes || source.nodes[childIndex * NODE_WORDS + 2] !== level + 1) {
      return invalid("invalid-topology", visits);
    }
    nodeIndex = childIndex;
  }
  return invalid("work-exhausted", SVO_STRUCTURAL_POINT_LOOKUP_MAX_VISITS);
}

/** Locate the containing finest-grid cell for an arbitrary world position. */
export function sampleSvoStructuralCoarseFluidAtWorld(
  source: SvoStructuralFluidPackedFixture,
  position_m: SvoVec3,
): SvoStructuralFluidCellLookup {
  const cell = pointToCell(source, position_m);
  return cell ? lookupSvoStructuralCoarseFluidCell(source, cell) : { status: "miss", reason: "outside-domain", nodeVisits: 0 };
}

/** Fine owns the position only when it explicitly returns a finite valid sample; coarse otherwise owns it. */
export function sampleSvoStructuralFluidExclusive(
  source: SvoStructuralFluidPackedFixture,
  position_m: SvoVec3,
  fineSampler?: SvoStructuralFineFluidSampler,
): SvoStructuralExclusiveFluidSample {
  const fine = fineSampler?.(position_m);
  if (fine?.valid) {
    if (!Number.isFinite(fine.phi_m)) return { status: "invalid", owner: "none", reason: "invalid-fine-sample" };
    return { status: "valid", owner: "fine", phi_m: fine.phi_m };
  }
  const coarse = sampleSvoStructuralCoarseFluidAtWorld(source, position_m);
  if (coarse.status === "valid") return { status: "valid", owner: "coarse", phi_m: coarse.phi_m, coarse };
  if (coarse.status === "miss") return { status: "miss", owner: "none", reason: coarse.reason };
  return { status: "invalid", owner: "none", reason: coarse.reason };
}

/** Anisotropic central/one-sided gradient; each neighbor independently traverses across brick boundaries. */
export function gradientSvoStructuralCoarseFluid(
  source: SvoStructuralFluidPackedFixture,
  position_m: SvoVec3,
): SvoStructuralFluidGradient {
  const cell = pointToCell(source, position_m);
  if (!cell) return { status: "miss", reason: "outside-domain" };
  const center = lookupSvoStructuralCoarseFluidCell(source, cell);
  if (center.status === "miss") return { status: "miss", reason: center.reason };
  if (center.status === "invalid") return { status: "invalid", reason: center.reason };
  const gradient = [0, 0, 0];
  const schemes: Array<"central" | "forward" | "backward"> = [];
  for (let axis = 0; axis < 3; axis += 1) {
    const backwardCell = [...cell] as [number, number, number];
    const forwardCell = [...cell] as [number, number, number];
    backwardCell[axis] -= 1;
    forwardCell[axis] += 1;
    const backward = lookupSvoStructuralCoarseFluidCell(source, backwardCell);
    const forward = lookupSvoStructuralCoarseFluidCell(source, forwardCell);
    if (backward.status === "invalid") return { status: "invalid", reason: backward.reason };
    if (forward.status === "invalid") return { status: "invalid", reason: forward.reason };
    const h = source.domain.cellSize_m[axis];
    if (backward.status === "valid" && forward.status === "valid") {
      gradient[axis] = (forward.phi_m - backward.phi_m) / (2 * h);
      schemes.push("central");
    } else if (forward.status === "valid") {
      gradient[axis] = (forward.phi_m - center.phi_m) / h;
      schemes.push("forward");
    } else if (backward.status === "valid") {
      gradient[axis] = (center.phi_m - backward.phi_m) / h;
      schemes.push("backward");
    } else return { status: "invalid", reason: "insufficient-neighbors" };
  }
  const magnitude = Math.hypot(...gradient);
  const normal: SvoVec3 | null = magnitude > 1e-12
    ? [gradient[0] / magnitude, gradient[1] / magnitude, gradient[2] / magnitude]
    : null;
  return {
    status: "valid",
    gradient: gradient as unknown as SvoVec3,
    normal,
    schemes: schemes as ["central" | "forward" | "backward", "central" | "forward" | "backward", "central" | "forward" | "backward"],
    center,
  };
}

/**
 * Binding-free helper. The composing shader declares the six structural
 * arrays using the `svoStructural*` names below; bind-group numbers remain the
 * consumer's choice. No debug-record expansion or solver-private resource is
 * assumed.
 */
export const svoStructuralCoarseFluidSamplingWGSL = /* wgsl */ `
const SVO_STRUCTURAL_SAMPLE_MISS:u32=0u;
const SVO_STRUCTURAL_SAMPLE_VALID:u32=1u;
const SVO_STRUCTURAL_SAMPLE_INVALID:u32=2u;
const SVO_STRUCTURAL_SAMPLE_EXHAUSTED:u32=3u;
const SVO_STRUCTURAL_MAX_VISITS:u32=${SVO_STRUCTURAL_POINT_LOOKUP_MAX_VISITS}u;
const SVO_STRUCTURAL_INVALID_INDEX:u32=0xffffffffu;
const SVO_STRUCTURAL_RESIDENT:u32=${FLUID_BRICK_RESIDENT}u;
const SVO_STRUCTURAL_REQUIRED_FIELDS:u32=${SPARSE_VOXEL_VALID_FIELDS.topology | SPARSE_VOXEL_VALID_FIELDS.coarseFluid}u;
const SVO_STRUCTURAL_CLEARED_PHI:f32=3.0e38;

struct SvoStructuralSamplingDomain {
  worldOrigin_m:vec4f,
  cellSize_m:vec4f,
  dimensionsBrick:vec4u,
  // x maximum depth, y expected complete generation (0 disables cached-fence check)
  settings:vec4u,
}
struct SvoStructuralCellSample {
  // x status, y node, z leaf, w voxel
  identity:vec4u,
  // x phi metres, y node visits
  value:vec4f,
}
struct SvoStructuralGradientSample {
  // x status, y packs 2-bit x/y/z finite-difference schemes
  metadata:vec4u,
  gradient:vec4f,
  normal:vec4f,
}

fn svoStructuralInvalid(status:u32,visits:u32)->SvoStructuralCellSample{
  return SvoStructuralCellSample(vec4u(status,0xffffffffu,0xffffffffu,0xffffffffu),vec4f(0.0,f32(visits),0.0,0.0));
}
fn svoStructuralPopcountBefore(mask:u32,octant:u32)->u32{
  return countOneBits(mask&select(0u,(1u<<octant)-1u,octant>0u));
}
fn svoStructuralMortonBit(low:u32,high:u32,bit:u32)->u32{
  if(bit>=32u){return (high>>(bit-32u))&1u;}
  return (low>>bit)&1u;
}
fn svoStructuralDecodeMorton(low:u32,high:u32,level:u32)->vec3u{
  var coordinate=vec3u(0u);
  for(var bit=0u;bit<21u;bit+=1u){if(bit>=level){break;}let scale=1u<<bit;coordinate.x+=svoStructuralMortonBit(low,high,3u*bit)*scale;coordinate.y+=svoStructuralMortonBit(low,high,3u*bit+1u)*scale;coordinate.z+=svoStructuralMortonBit(low,high,3u*bit+2u)*scale;}
  return coordinate;
}
fn svoStructuralCoarseFluidCell(
  domain:SvoStructuralSamplingDomain,cell:vec3i
)->SvoStructuralCellSample{
  if(any(cell<vec3i(0))||any(cell>=vec3i(domain.dimensionsBrick.xyz))){return svoStructuralInvalid(SVO_STRUCTURAL_SAMPLE_MISS,0u);}
  let generation=svoStructuralPublication[${SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration}u];
  if(generation==0u||(domain.settings.y!=0u&&generation!=domain.settings.y)){return svoStructuralInvalid(SVO_STRUCTURAL_SAMPLE_INVALID,0u);}
  if((svoStructuralPublication[${SPARSE_VOXEL_PUBLICATION_STATE.validFields}u]&SVO_STRUCTURAL_REQUIRED_FIELDS)!=SVO_STRUCTURAL_REQUIRED_FIELDS||svoStructuralControl[${SPARSE_BRICK_GPU_LAYOUT.controlWords.overflowFlags}u]!=0u){return svoStructuralInvalid(SVO_STRUCTURAL_SAMPLE_INVALID,0u);}
  let publishedNodes=svoStructuralControl[${SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedNodes}u];let publishedLeaves=svoStructuralControl[${SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedLeaves}u];let publishedVoxels=svoStructuralControl[${SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedVoxels}u];
  let brickSize=domain.dimensionsBrick.w;if(brickSize!=svoStructuralControl[${SPARSE_BRICK_GPU_LAYOUT.controlWords.brickSize}u]){return svoStructuralInvalid(SVO_STRUCTURAL_SAMPLE_INVALID,0u);}
  let finestBrick=vec3u(cell)/brickSize;var nodeIndex=0u;
  for(var visits=1u;visits<=SVO_STRUCTURAL_MAX_VISITS;visits+=1u){
    if(nodeIndex>=publishedNodes||nodeIndex*8u+7u>=arrayLength(&svoStructuralNodes)){return svoStructuralInvalid(SVO_STRUCTURAL_SAMPLE_INVALID,visits-1u);}
    let nodeBase=nodeIndex*8u;let level=svoStructuralNodes[nodeBase+2u];if(level>domain.settings.x){return svoStructuralInvalid(SVO_STRUCTURAL_SAMPLE_INVALID,visits);}
    let leafIndex=svoStructuralNodes[nodeBase+6u];
    if(leafIndex!=SVO_STRUCTURAL_INVALID_INDEX){
      if(leafIndex>=publishedLeaves||leafIndex>=arrayLength(&svoStructuralLeafStates)||leafIndex*4u+3u>=arrayLength(&svoStructuralLeaves)){return svoStructuralInvalid(SVO_STRUCTURAL_SAMPLE_INVALID,visits);}
      let leafBase=leafIndex*4u;if(svoStructuralLeaves[leafBase]!=nodeIndex){return svoStructuralInvalid(SVO_STRUCTURAL_SAMPLE_INVALID,visits);}
      if((svoStructuralLeafStates[leafIndex]&SVO_STRUCTURAL_RESIDENT)==0u){return svoStructuralInvalid(SVO_STRUCTURAL_SAMPLE_INVALID,visits);}
      let coordinate=svoStructuralDecodeMorton(svoStructuralNodes[nodeBase],svoStructuralNodes[nodeBase+1u],level);let scale=1u<<(domain.settings.x-level);
      let origin=coordinate*brickSize*scale;let local=vec3u(cell-vec3i(origin))/scale;if(any(local>=vec3u(brickSize))){return svoStructuralInvalid(SVO_STRUCTURAL_SAMPLE_INVALID,visits);}
      let voxelIndex=svoStructuralLeaves[leafBase+1u]+local.x+local.y*brickSize+local.z*brickSize*brickSize;
      if(voxelIndex>=publishedVoxels||voxelIndex>=arrayLength(&svoStructuralGeometry)){return svoStructuralInvalid(SVO_STRUCTURAL_SAMPLE_INVALID,visits);}
      let phi=svoStructuralGeometry[voxelIndex].x;if(phi!=phi||abs(phi)>=SVO_STRUCTURAL_CLEARED_PHI){return svoStructuralInvalid(SVO_STRUCTURAL_SAMPLE_INVALID,visits);}
      return SvoStructuralCellSample(vec4u(SVO_STRUCTURAL_SAMPLE_VALID,nodeIndex,leafIndex,voxelIndex),vec4f(phi,f32(visits),0.0,0.0));
    }
    if(level>=domain.settings.x){return svoStructuralInvalid(SVO_STRUCTURAL_SAMPLE_INVALID,visits);}
    let shift=domain.settings.x-level-1u;let octant=((finestBrick.x>>shift)&1u)|(((finestBrick.y>>shift)&1u)<<1u)|(((finestBrick.z>>shift)&1u)<<2u);
    let childMask=svoStructuralNodes[nodeBase+3u]&0xffu;if((childMask&(1u<<octant))==0u){return svoStructuralInvalid(SVO_STRUCTURAL_SAMPLE_MISS,visits);}
    let firstChild=svoStructuralNodes[nodeBase+4u];let childCount=svoStructuralNodes[nodeBase+5u];if(firstChild==SVO_STRUCTURAL_INVALID_INDEX||childCount!=countOneBits(childMask)){return svoStructuralInvalid(SVO_STRUCTURAL_SAMPLE_INVALID,visits);}
    nodeIndex=firstChild+svoStructuralPopcountBefore(childMask,octant);
  }
  return svoStructuralInvalid(SVO_STRUCTURAL_SAMPLE_EXHAUSTED,SVO_STRUCTURAL_MAX_VISITS);
}
fn svoStructuralCoarseFluidWorld(
  domain:SvoStructuralSamplingDomain,position_m:vec3f
)->SvoStructuralCellSample{
  let cell=vec3i(floor((position_m-domain.worldOrigin_m.xyz)/domain.cellSize_m.xyz));
  return svoStructuralCoarseFluidCell(domain,cell);
}
fn svoStructuralCoarseFluidGradientWorld(
  domain:SvoStructuralSamplingDomain,position_m:vec3f
)->SvoStructuralGradientSample{
  let cell=vec3i(floor((position_m-domain.worldOrigin_m.xyz)/domain.cellSize_m.xyz));
  let center=svoStructuralCoarseFluidCell(domain,cell);
  if(center.identity.x!=SVO_STRUCTURAL_SAMPLE_VALID){return SvoStructuralGradientSample(vec4u(center.identity.x,0u,0u,0u),vec4f(0.0),vec4f(0.0));}
  var gradient=vec3f(0.0);var schemes=0u;
  for(var axis=0u;axis<3u;axis+=1u){
    var backwardCell=cell;var forwardCell=cell;backwardCell[axis]-=1;forwardCell[axis]+=1;
    let backward=svoStructuralCoarseFluidCell(domain,backwardCell);let forward=svoStructuralCoarseFluidCell(domain,forwardCell);
    if(backward.identity.x>SVO_STRUCTURAL_SAMPLE_VALID||forward.identity.x>SVO_STRUCTURAL_SAMPLE_VALID){return SvoStructuralGradientSample(vec4u(SVO_STRUCTURAL_SAMPLE_INVALID,0u,0u,0u),vec4f(0.0),vec4f(0.0));}
    let h=domain.cellSize_m[axis];
    if(backward.identity.x==SVO_STRUCTURAL_SAMPLE_VALID&&forward.identity.x==SVO_STRUCTURAL_SAMPLE_VALID){gradient[axis]=(forward.value.x-backward.value.x)/(2.0*h);}
    else if(forward.identity.x==SVO_STRUCTURAL_SAMPLE_VALID){gradient[axis]=(forward.value.x-center.value.x)/h;schemes|=1u<<(2u*axis);}
    else if(backward.identity.x==SVO_STRUCTURAL_SAMPLE_VALID){gradient[axis]=(center.value.x-backward.value.x)/h;schemes|=2u<<(2u*axis);}
    else{return SvoStructuralGradientSample(vec4u(SVO_STRUCTURAL_SAMPLE_INVALID,0u,0u,0u),vec4f(0.0),vec4f(0.0));}
  }
  let magnitude=length(gradient);var normal=vec3f(0.0);if(magnitude>1e-12){normal=gradient/magnitude;}
  return SvoStructuralGradientSample(vec4u(SVO_STRUCTURAL_SAMPLE_VALID,schemes,0u,0u),vec4f(gradient,0.0),vec4f(normal,select(0.0,1.0,magnitude>1e-12)));
}
`;
