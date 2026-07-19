import { unpackMaterialOwner, type SparseBrickSize } from "./sparse-brick-octree";
import type { SvoLeafHit, SvoRay, SvoVec3, SvoWorldMapping } from "./webgpu-svo-traversal";

export const SVO_PAYLOAD_DDA_MAX_VISITS = 512;

export const SVO_PAYLOAD_DDA_STATUS = Object.freeze({
  miss: 0,
  hit: 1,
  invalid: 2,
  workExhausted: 3,
});

export type SvoPayloadDdaResult =
  | { status: "miss"; visits: number }
  | { status: "invalid"; visits: number; reason: string }
  | { status: "work-exhausted"; visits: number }
  | {
    status: "hit";
    visits: number;
    payloadIndex: number;
    local: readonly [number, number, number];
    materialId: number;
    ownerId: number;
    tEnter: number;
    tExit: number;
  };

export interface SvoPayloadDdaOptions {
  maxVoxelVisits?: number;
  /** Published payload count from sparse control word 2. */
  publishedVoxelCount?: number;
}

const finite3 = (value: SvoVec3, label: string): void => {
  if (value.some((component) => !Number.isFinite(component))) throw new RangeError(`${label} must be finite`);
};

function brickVoxelIndex(offset: number, local: readonly [number, number, number], brickSize: SparseBrickSize): number {
  return offset + local[0] + local[1] * brickSize + local[2] * brickSize * brickSize;
}

/**
 * CPU mirror of the binding-free WGSL below. This resolves the first occupied
 * x-major material/owner voxel inside a terminal leaf; it never expands debug
 * cubes and treats every publication/budget failure as a typed fail-closed
 * result.
 */
export function traverseSvoLeafPayload(
  ray: SvoRay,
  leaf: SvoLeafHit,
  materialOwners: Uint32Array,
  mapping: SvoWorldMapping,
  options: SvoPayloadDdaOptions = {},
): SvoPayloadDdaResult {
  finite3(ray.origin, "Ray origin");
  finite3(ray.direction, "Ray direction");
  if (ray.direction.every((component) => component === 0)) throw new RangeError("Ray direction must be non-zero");
  const budget = options.maxVoxelVisits ?? SVO_PAYLOAD_DDA_MAX_VISITS;
  if (!Number.isInteger(budget) || budget < 1 || budget > SVO_PAYLOAD_DDA_MAX_VISITS) {
    throw new RangeError(`Payload DDA budget must be 1..${SVO_PAYLOAD_DDA_MAX_VISITS}`);
  }
  const published = options.publishedVoxelCount ?? materialOwners.length;
  if (!Number.isInteger(published) || published < 0 || published > materialOwners.length) {
    return { status: "invalid", visits: 0, reason: "Published payload count exceeds the material/owner buffer" };
  }
  const rayMinimum = Math.max(ray.tMin ?? 0, 0, leaf.tEnter);
  const rayMaximum = Math.min(ray.tMax ?? Number.POSITIVE_INFINITY, leaf.tExit);
  if (rayMaximum < rayMinimum) return { status: "miss", visits: 0 };
  const brickSize = mapping.brickSize;
  const extent = leaf.bounds.minimum.map((minimum, axis) => (leaf.bounds.maximum[axis] - minimum) / brickSize) as [number, number, number];
  if (extent.some((component) => !Number.isFinite(component) || component <= 0)) {
    return { status: "invalid", visits: 0, reason: "Leaf extent is invalid" };
  }

  let entry = rayMinimum;
  const probeT = Math.min(rayMaximum, entry + 1e-5);
  const point = ray.origin.map((component, axis) => component + ray.direction[axis] * probeT);
  const cell = point.map((component, axis) => Math.max(0, Math.min(brickSize - 1,
    Math.floor((component - leaf.bounds.minimum[axis]) / extent[axis])))) as [number, number, number];
  const step = ray.direction.map((component) => component >= 0 ? 1 : -1) as [number, number, number];
  const nextT = [0, 0, 0];
  const deltaT = [0, 0, 0];
  for (let axis = 0; axis < 3; axis += 1) {
    if (Math.abs(ray.direction[axis]) <= 1e-9) {
      nextT[axis] = Number.POSITIVE_INFINITY;
      deltaT[axis] = Number.POSITIVE_INFINITY;
      continue;
    }
    const boundary = leaf.bounds.minimum[axis] + (cell[axis] + (step[axis] > 0 ? 1 : 0)) * extent[axis];
    nextT[axis] = (boundary - ray.origin[axis]) / ray.direction[axis];
    deltaT[axis] = Math.abs(extent[axis] / ray.direction[axis]);
  }

  for (let visits = 1; visits <= budget; visits += 1) {
    if (cell.some((component) => component < 0 || component >= brickSize) || entry > rayMaximum) {
      return { status: "miss", visits: visits - 1 };
    }
    const payloadIndex = brickVoxelIndex(leaf.voxelOffset, cell, brickSize);
    if (payloadIndex >= published || payloadIndex >= materialOwners.length) {
      return { status: "invalid", visits, reason: "Leaf payload index is outside the published material/owner range" };
    }
    const { materialId, ownerId } = unpackMaterialOwner(materialOwners[payloadIndex]);
    const cellExit = Math.min(nextT[0], nextT[1], nextT[2], rayMaximum);
    if (materialId !== 0) {
      return { status: "hit", visits, payloadIndex, local: [...cell] as [number, number, number], materialId, ownerId, tEnter: entry, tExit: cellExit };
    }
    const advance = Math.min(nextT[0], nextT[1], nextT[2]);
    if (!Number.isFinite(advance)) return { status: "miss", visits };
    for (let axis = 0; axis < 3; axis += 1) {
      if (nextT[axis] <= advance + 1e-6) {
        cell[axis] += step[axis];
        nextT[axis] += deltaT[axis];
      }
    }
    entry = advance;
  }
  return { status: "work-exhausted", visits: budget };
}

/**
 * Binding-free payload DDA. The composing shader declares
 * `svoMaterialOwners: array<u32>` and includes `webgpuSvoTraversalWGSL` first.
 */
export const svoStructuralPayloadDdaWGSL = /* wgsl */ `
const SVO_PAYLOAD_STATUS_MISS:u32=0u;
const SVO_PAYLOAD_STATUS_HIT:u32=1u;
const SVO_PAYLOAD_STATUS_INVALID:u32=2u;
const SVO_PAYLOAD_STATUS_EXHAUSTED:u32=3u;
const SVO_PAYLOAD_MAX_VISITS:u32=${SVO_PAYLOAD_DDA_MAX_VISITS}u;
struct SvoPayloadDdaHit {
  // x status, y visits, z payload index, w leaf index
  metadata:vec4u,
  // x material, y owner, z/y brick-local coordinate
  identityLocal:vec4u,
  // x local z, y entry t, z exit t
  localDistance:vec4f,
}
fn svoPayloadDdaMiss(status:u32,visits:u32,leafIndex:u32)->SvoPayloadDdaHit{
  return SvoPayloadDdaHit(vec4u(status,visits,SVO_INVALID,leafIndex),vec4u(SVO_INVALID),vec4f(0.0));
}
fn svoTraverseLeafPayload(
  ray:SvoRay,mapping:SvoMapping,leaf:SvoTraversalHit,publishedVoxelCount:u32,workLimit:u32
)->SvoPayloadDdaHit{
  if(leaf.status!=SVO_STATUS_HIT||leaf.nodeIndex>=mapping.nodeCount||(mapping.brickSize!=4u&&mapping.brickSize!=8u)){return svoPayloadDdaMiss(SVO_PAYLOAD_STATUS_INVALID,0u,leaf.leafIndex);}
  let bounds=svoNodeBounds(svoNodes[leaf.nodeIndex],mapping);let extent=(bounds[1]-bounds[0])/f32(mapping.brickSize);
  if(any(extent<=vec3f(0.0))){return svoPayloadDdaMiss(SVO_PAYLOAD_STATUS_INVALID,0u,leaf.leafIndex);}
  var entry=max(max(leaf.tEnter,ray.tMin),0.0);let rayExit=min(leaf.tExit,ray.tMax);
  if(rayExit<entry){return svoPayloadDdaMiss(SVO_PAYLOAD_STATUS_MISS,0u,leaf.leafIndex);}
  let probeT=min(rayExit,entry+1e-5);let point=ray.origin+ray.direction*probeT;
  var cell=vec3i(clamp(floor((point-bounds[0])/extent),vec3f(0.0),vec3f(f32(mapping.brickSize-1u))));
  let step=select(vec3i(-1),vec3i(1),ray.direction>=vec3f(0.0));
  let nextBoundary=bounds[0]+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;
  var nextT=select(vec3f(3.4e38),(nextBoundary-ray.origin)/ray.direction,abs(ray.direction)>vec3f(1e-9));
  let deltaT=select(vec3f(3.4e38),abs(extent/ray.direction),abs(ray.direction)>vec3f(1e-9));
  let limit=min(max(workLimit,1u),SVO_PAYLOAD_MAX_VISITS);
  for(var visits=1u;visits<=SVO_PAYLOAD_MAX_VISITS;visits+=1u){
    if(visits>limit){return svoPayloadDdaMiss(SVO_PAYLOAD_STATUS_EXHAUSTED,limit,leaf.leafIndex);}
    if(any(cell<vec3i(0))||any(cell>=vec3i(i32(mapping.brickSize)))||entry>rayExit){return svoPayloadDdaMiss(SVO_PAYLOAD_STATUS_MISS,visits-1u,leaf.leafIndex);}
    let payloadIndex=svoBrickVoxelIndex(leaf.voxelOffset,vec3u(cell),mapping.brickSize);
    if(payloadIndex>=publishedVoxelCount||payloadIndex>=arrayLength(&svoMaterialOwners)){return svoPayloadDdaMiss(SVO_PAYLOAD_STATUS_INVALID,visits,leaf.leafIndex);}
    let packed=svoMaterialOwners[payloadIndex];let material=packed&0xffffu;let owner=packed>>16u;let cellExit=min(min(nextT.x,nextT.y),min(nextT.z,rayExit));
    if(material!=0u){return SvoPayloadDdaHit(vec4u(SVO_PAYLOAD_STATUS_HIT,visits,payloadIndex,leaf.leafIndex),vec4u(material,owner,u32(cell.x),u32(cell.y)),vec4f(f32(cell.z),entry,cellExit,0.0));}
    let advance=min(nextT.x,min(nextT.y,nextT.z));if(advance>=3.0e38){return svoPayloadDdaMiss(SVO_PAYLOAD_STATUS_MISS,visits,leaf.leafIndex);}
    if(nextT.x<=advance+1e-6){cell.x+=step.x;nextT.x+=deltaT.x;}if(nextT.y<=advance+1e-6){cell.y+=step.y;nextT.y+=deltaT.y;}if(nextT.z<=advance+1e-6){cell.z+=step.z;nextT.z+=deltaT.z;}entry=advance;
  }
  return svoPayloadDdaMiss(SVO_PAYLOAD_STATUS_EXHAUSTED,limit,leaf.leafIndex);
}
`;
