import { OCTREE_LOSASSO_FACE_WORDS } from "./octree-losasso-operator";
import { octreeCompensatedF32WGSL } from "./octree-compensated-f32.wgsl";
import { octreeOwnerPageLookupWgsl } from "./webgpu-octree-owner-pages";

/**
 * Compact topology-to-face publisher for the Losasso backend.
 *
 * The shader deliberately consumes only LeafHeader rows, their finalized frontier,
 * and the owner-page arena.  In particular it has no declaration for a Power
 * descriptor, catalogue, topology LUT, edge channel, or Delaunay workset.
 */
export const webgpuOctreeLosassoBackendWGSL = /* wgsl */ `
const INVALID: u32 = 0xffffffffu;
const FACE_WORDS: u32 = ${OCTREE_LOSASSO_FACE_WORDS}u;
const ERROR_CAPACITY: u32 = 1u;
const ERROR_HEADER: u32 = 2u;
const ERROR_OWNER: u32 = 4u;
const ERROR_NEIGHBOR_ROW: u32 = 8u;
const FACE_CLOSED_BOUNDARY: u32 = 0x40000000u;
const FACE_ROW_ON_POSITIVE_SIDE: u32 = 0x80000000u;

struct Params {
  dimensionsMaximumLeaf: vec4u,
  brickDimensionsLogicalCount: vec4u,
  arenaOffsetsCapacity: vec4u,
  capacities: vec4u, // rows, faces, incidences, reserved
  cellSize: vec4f,
  domainOrigin: vec4f,
  boundary: vec4u,
};
struct LeafHeader {
  cell: u32, entryStart: u32, entryCount: u32, size: u32,
  diagonal: f32, rhs: f32, pad0: u32, pad1: u32, gradient: vec4f,
};
struct Face {
  negativeRow: u32, positiveRow: u32, axis: u32, reserved: u32,
  area: f32, inverseDistance: f32, openFraction: f32, normalVelocity: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> leafHeaders: array<LeafHeader>;
@group(0) @binding(2) var<storage, read> sourceFrontier: array<u32>;
@group(0) @binding(3) var<storage, read> ownerPageArena: array<u32>;
@group(0) @binding(4) var<storage, read_write> authority: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> rowFaceOffsets: array<u32>;
@group(0) @binding(6) var<storage, read_write> rowFaces: array<u32>;
@group(0) @binding(7) var<storage, read_write> faces: array<Face>;
@group(0) @binding(8) var<storage, read_write> rowDispatch: array<u32>;
@group(0) @binding(9) var<storage, read_write> faceDispatch: array<u32>;
@group(0) @binding(10) var<storage, read_write> rhs: array<f32>;
@group(0) @binding(11) var<storage, read_write> faceMetrics: array<vec4u>;
@group(0) @binding(12) var<storage, read_write> adjacencyOffsets: array<u32>;
@group(0) @binding(13) var<storage, read_write> scratch: array<atomic<u32>>;
@group(0) @binding(14) var<storage, read_write> faceGeometry: array<vec4u>;
@group(0) @binding(15) var<storage, read_write> faceDirectory: array<vec2u>;
@group(0) @binding(16) var<storage, read_write> diagonal: array<f32>;
@group(0) @binding(17) var<storage, read_write> solverAuthority: array<u32>;
@group(0) @binding(18) var<storage, read> solidCells: array<u32>;
@group(0) @binding(19) var<storage, read> rigidBodies: array<RigidBody>;
@group(0) @binding(21) var<storage, read> ownerCandidate: array<u32>;
@group(0) @binding(22) var<storage, read> acceptedAuthority: array<u32>;

var<workgroup> prefixTotals: array<u32, 256>;

struct RigidBody {
  positionShape:vec4f, dimensions:vec4f, orientation:vec4f,
  linearVelocity:vec4f, angularVelocity:vec4f, inverseMassInertia:vec4f,
  angularMomentumRestitution:vec4f, material:vec4f,
};

${octreeOwnerPageLookupWgsl.replaceAll("ownerPageLookupParams", "params").replace(
  `let table = ownerPageArena[${10}u] >> 31u;`,
  `let acceptedTable = ownerPageArena[${10}u] >> 31u;
  let table = 1u - acceptedTable;`,
)}

fn topologyReused()->bool{return atomicLoad(&authority[5])==1u;}
fn rows() -> u32 { return select(min(atomicLoad(&authority[1]), params.capacities.x),0u,topologyReused()); }
fn faceCountBase() -> u32 { return 0u; }
fn faceBaseBase() -> u32 { return params.capacities.x; }
fn incidenceCountBase() -> u32 { return 2u * params.capacities.x; }
fn incidenceCursorBase() -> u32 { return 3u * params.capacities.x; }
fn fail(flag: u32) { atomicOr(&authority[4], flag); }
fn losassoFinite(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
fn losassoAddExact(limbs:ptr<function,array<i32,36>>,value:f32){let bits=bitcast<u32>(value);let magnitude=bits&0x7fffffffu;
 if(magnitude==0u){return;}let rawExponent=(magnitude>>23u)&0xffu;let fraction=magnitude&0x7fffffu;
 let significand=select(fraction,0x800000u|fraction,rawExponent!=0u);let shift=select(3u,rawExponent+2u,rawExponent!=0u);
 let firstLimb=shift>>3u;let shifted=significand<<(shift&7u);let sign=select(1,-1,(bits&0x80000000u)!=0u);
 for(var digit=0u;digit<4u;digit+=1u){let limb=firstLimb+digit;let byte=i32((shifted>>(digit*8u))&0xffu);
  if(byte!=0&&limb<36u){(*limbs)[limb]+=sign*byte;}}}
fn losassoFloorDiv256(value:i32)->vec2i{var carry=value/256;var digit=value-carry*256;if(digit<0){digit+=256;carry-=1;}return vec2i(carry,digit);}
fn losassoExactValue(source:ptr<function,array<i32,36>>)->f32{var limbs=(*source);
 for(var limb=0u;limb+1u<36u;limb+=1u){let normalized=losassoFloorDiv256(limbs[limb]);limbs[limb]=normalized.y;limbs[limb+1u]+=normalized.x;}
 let negative=limbs[35]<0;if(negative){for(var limb=0u;limb<36u;limb+=1u){limbs[limb]=-limbs[limb];}
  for(var limb=0u;limb+1u<36u;limb+=1u){let normalized=losassoFloorDiv256(limbs[limb]);limbs[limb]=normalized.y;limbs[limb+1u]+=normalized.x;}}
 var magnitude=0.;for(var limb=0u;limb<36u;limb+=1u){magnitude+=ldexp(f32(limbs[limb]),-152+i32(limb*8u));}
 return select(magnitude,-magnitude,negative);}
${octreeCompensatedF32WGSL}

fn originOf(header: LeafHeader) -> vec3u {
  let dimensions = params.dimensionsMaximumLeaf.xyz;
  return vec3u(header.cell % dimensions.x,
    (header.cell / dimensions.x) % dimensions.y,
    header.cell / (dimensions.x * dimensions.y));
}
fn headerValid(header: LeafHeader) -> bool {
  let dimensions = params.dimensionsMaximumLeaf.xyz;
  if (header.size == 0u || header.size > params.dimensionsMaximumLeaf.w
      || (header.size & (header.size - 1u)) != 0u) { return false; }
  let origin = originOf(header);
  return all(origin % vec3u(header.size) == vec3u(0u))
    && all(vec3u(header.size) <= dimensions)
    && all(origin <= dimensions - vec3u(header.size));
}
fn mortonPart(value: u32) -> u32 {
  var x = value & 1023u;
  x = (x | (x << 16u)) & 0x030000ffu;
  x = (x | (x << 8u)) & 0x0300f00fu;
  x = (x | (x << 4u)) & 0x030c30c3u;
  x = (x | (x << 2u)) & 0x09249249u;
  return x;
}
fn morton(origin: vec3u) -> u32 {
  return mortonPart(origin.x) | (mortonPart(origin.y) << 1u)
    | (mortonPart(origin.z) << 2u);
}
fn findRow(origin: vec3u, size: u32) -> u32 {
  let count = rows();
  let wantedLevel = 31u - countLeadingZeros(size);
  let wantedMorton = morton(origin);
  var low = 0u;
  var high = count;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    let header = leafHeaders[middle];
    let level = 31u - countLeadingZeros(header.size);
    let key = morton(originOf(header));
    if (level < wantedLevel || (level == wantedLevel && key < wantedMorton)) {
      low = middle + 1u;
    } else { high = middle; }
  }
  if (low < count) {
    let header = leafHeaders[low];
    if (header.size == size && all(originOf(header) == origin)) { return low; }
  }
  return INVALID;
}
fn axisVector(axis: u32) -> vec3u {
  return select(select(vec3u(0u, 0u, 1u), vec3u(0u, 1u, 0u), axis == 1u),
    vec3u(1u, 0u, 0u), axis == 0u);
}
fn tangentA(axis: u32) -> vec3u {
  return select(select(vec3u(1u, 0u, 0u), vec3u(1u, 0u, 0u), axis == 1u),
    vec3u(0u, 1u, 0u), axis == 0u);
}
fn tangentB(axis: u32) -> vec3u {
  return select(select(vec3u(0u, 1u, 0u), vec3u(0u, 0u, 1u), axis == 1u),
    vec3u(0u, 0u, 1u), axis == 0u);
}
fn positiveProbe(origin: vec3u, size: u32, axis: u32, a: u32, b: u32,
    subdivision: u32) -> vec3i {
  let axisStep = axisVector(axis) * vec3u(size);
  let span = size / subdivision;
  let offset = tangentA(axis) * vec3u(a * span + span / 2u)
    + tangentB(axis) * vec3u(b * span + span / 2u);
  return vec3i(origin + axisStep + offset);
}
fn negativeProbe(origin: vec3u, size: u32, axis: u32, a: u32, b: u32,
    subdivision: u32) -> vec3i {
  let span = size / subdivision;
  let offset = tangentA(axis) * vec3u(a * span + span / 2u)
    + tangentB(axis) * vec3u(b * span + span / 2u);
  return vec3i(origin + offset) - vec3i(axisVector(axis));
}
struct FacePatchPlan { mask: u32, subdivision: u32 }

fn validFaceNeighbor(header: LeafHeader, neighbor: OctreeOwnerPageLookupResult) -> bool {
  if ((neighbor.status & OWNER_PAGE_LOOKUP_INVALID) != 0u) {
    fail(ERROR_OWNER); return false;
  }
  if ((neighbor.status & OWNER_PAGE_LOOKUP_MISSING) == 0u
      && (neighbor.size * 2u < header.size || neighbor.size > 2u * header.size)) {
    fail(ERROR_OWNER); return false;
  }
  return true;
}

fn positiveFacePlan(header: LeafHeader, axis: u32) -> FacePatchPlan {
  let origin = originOf(header);
  let dimensions = params.dimensionsMaximumLeaf.xyz;
  if (origin[axis] + header.size >= dimensions[axis]) {
    // Closed wall faces remain in the compact graph. Fine phi conditions them
    // to zero-aperture Neumann while wet and opens them as ghost-fluid
    // Dirichlet faces only after the surface has separated.
    return FacePatchPlan(1u, 1u);
  }
  // A centre probe observes only one quadrant of a 2:1 seam. Inspect all four
  // quadrants before deciding whether this row owns one full face or four fine
  // patches; otherwise a partially wet seam can be counted at the wrong area.
  var split = false;
  for (var b = 0u; b < 2u; b += 1u) {
    for (var a = 0u; a < 2u; a += 1u) {
      let neighbor = octreeOwnerPageLookup(positiveProbe(origin, header.size, axis, a, b, 2u));
      if (!validFaceNeighbor(header, neighbor)) { return FacePatchPlan(0u, 1u); }
      if ((neighbor.status & OWNER_PAGE_LOOKUP_MISSING) == 0u
          && neighbor.size < header.size) { split = true; }
    }
  }
  if (split) { return FacePatchPlan(0xfu, 2u); }
  return FacePatchPlan(1u, 1u);
}

fn negativeBoundaryPlan(header: LeafHeader, axis: u32) -> FacePatchPlan {
  let origin = originOf(header);
  if (origin[axis] == 0u) {
    return FacePatchPlan(1u, 1u);
  }
  var split = false;
  for (var b = 0u; b < 2u; b += 1u) {
    for (var a = 0u; a < 2u; a += 1u) {
      let neighbor = octreeOwnerPageLookup(negativeProbe(origin, header.size, axis, a, b, 2u));
      if (!validFaceNeighbor(header, neighbor)) { return FacePatchPlan(0u, 1u); }
      if ((neighbor.status & OWNER_PAGE_LOOKUP_MISSING) == 0u
          && neighbor.size < header.size) { split = true; }
    }
  }
  if (!split) {
    let neighbor = octreeOwnerPageLookup(negativeProbe(origin, header.size, axis, 0u, 0u, 1u));
    if (!validFaceNeighbor(header, neighbor)) { return FacePatchPlan(0u, 1u); }
    var missing = (neighbor.status & OWNER_PAGE_LOOKUP_MISSING) != 0u;
    if (!missing) { missing = findRow(neighbor.origin, neighbor.size) == INVALID; }
    return FacePatchPlan(select(0u, 1u, missing), 1u);
  }
  var mask = 0u;
  for (var b = 0u; b < 2u; b += 1u) {
    for (var a = 0u; a < 2u; a += 1u) {
      let neighbor = octreeOwnerPageLookup(negativeProbe(origin, header.size, axis, a, b, 2u));
      var missing = (neighbor.status & OWNER_PAGE_LOOKUP_MISSING) != 0u;
      if (!missing) { missing = findRow(neighbor.origin, neighbor.size) == INVALID; }
      if (missing) { mask |= 1u << (a + 2u * b); }
    }
  }
  return FacePatchPlan(mask, 2u);
}

fn exactRowTopologyReuse(generation:u32,count:u32)->bool{
 let base=10u+3u*params.capacities.x;
 return base+15u<arrayLength(&sourceFrontier)&&sourceFrontier[base+8u]==0x52444c54u
  &&sourceFrontier[base]==count&&sourceFrontier[base+1u]==count
  &&sourceFrontier[base+2u]==count&&sourceFrontier[base+3u]==0u
  &&sourceFrontier[base+4u]==0u&&sourceFrontier[base+5u]==0u
  &&sourceFrontier[base+6u]==0u&&sourceFrontier[base+7u]==generation
  &&sourceFrontier[base+15u]==1u&&arrayLength(&acceptedAuthority)>=8u
  &&acceptedAuthority[0]!=0u&&acceptedAuthority[1]==count
  &&acceptedAuthority[2]<=params.capacities.y&&acceptedAuthority[3]==1u
  &&acceptedAuthority[4]==0u&&acceptedAuthority[6]>=2u*acceptedAuthority[2]
  &&acceptedAuthority[6]<=params.capacities.w
  &&(acceptedAuthority[6]&(acceptedAuthority[6]-1u))==0u;}

@compute @workgroup_size(1)
fn beginLosassoPublication() {
  var rawRows = 0u;
  var generation = 0u;
  var sourceValid = false;
  if (arrayLength(&sourceFrontier) >= 10u && arrayLength(&ownerCandidate) >= 29u
      && arrayLength(&ownerPageArena) > 3u) {
    let selector = sourceFrontier[7u];
    generation = ownerCandidate[24u];
    if (selector <= 1u) {
      rawRows = sourceFrontier[selector];
      sourceValid = sourceFrontier[6u] == 1u && sourceFrontier[9u] == 0u
        && sourceFrontier[8u] == generation && generation != 0u
        && ownerCandidate[28u] == 0x80000000u && ownerCandidate[23u] == 0u
        && ownerCandidate[27u] <= ownerPageArena[3u];
    }
  }
  let count = min(rawRows, params.capacities.x);
  let reuse=sourceValid&&rawRows<=params.capacities.x&&exactRowTopologyReuse(generation,count);
  atomicStore(&authority[0], select(0u, generation, sourceValid));
  atomicStore(&authority[1], count);
  atomicStore(&authority[2], select(0u,acceptedAuthority[2],reuse));
  atomicStore(&authority[3], select(0u,1u,reuse));
  atomicStore(&authority[4], select(ERROR_OWNER,
    select(0u, ERROR_CAPACITY, rawRows > params.capacities.x), sourceValid));
  atomicStore(&authority[5],select(0u,1u,reuse));
  atomicStore(&authority[6],select(0u,acceptedAuthority[6],reuse));
  atomicStore(&authority[7],0u);
  rowDispatch[0] = 0u; rowDispatch[1] = 1u; rowDispatch[2] = 1u;
  faceDispatch[0] = 0u; faceDispatch[1] = 1u; faceDispatch[2] = 1u;
  faceDispatch[3] = 0u; faceDispatch[4] = 1u; faceDispatch[5] = 1u;
  solverAuthority[0] = ERROR_CAPACITY;
  solverAuthority[1] = 0u;
  for (var word = 2u; word < 7u; word += 1u) { solverAuthority[word] = 0u; }
  rowDispatch[0] = select(0u, (count + 63u) / 64u,
    sourceValid && rawRows <= params.capacities.x && !reuse);
  if(reuse){rowDispatch[0]=(count+63u)/64u;faceDispatch[0]=(acceptedAuthority[2]+63u)/64u;
   solverAuthority[0]=0u;solverAuthority[1]=INVALID;solverAuthority[2]=count;
   solverAuthority[3]=acceptedAuthority[2];solverAuthority[4]=generation;
   solverAuthority[5]=0u;solverAuthority[6]=generation;}
}

@compute @workgroup_size(64)
fn countLosassoFaces(@builtin(global_invocation_id) invocation: vec3u) {
  let row = invocation.x;
  if (row >= rows()) { return; }
  let header = leafHeaders[row];
  var count = 0u;
  if (!headerValid(header)) { fail(ERROR_HEADER); }
  else {
    let owner = octreeOwnerPageLookup(vec3i(originOf(header)));
    if ((owner.status & (OWNER_PAGE_LOOKUP_INVALID | OWNER_PAGE_LOOKUP_MISSING)) != 0u
        || owner.size != header.size || any(owner.origin != originOf(header))) {
      fail(ERROR_OWNER);
    } else {
      for (var axis = 0u; axis < 3u; axis += 1u) {
        count += countOneBits(positiveFacePlan(header, axis).mask)
          + countOneBits(negativeBoundaryPlan(header, axis).mask);
      }
    }
  }
  atomicStore(&scratch[faceCountBase() + row], count);
  atomicStore(&scratch[incidenceCountBase() + row], 0u);
  atomicStore(&scratch[incidenceCursorBase() + row], 0u);
  rhs[row] = 0.0;
}

@compute @workgroup_size(256)
fn prefixLosassoFaces(@builtin(local_invocation_index) lane: u32) {
  let count = rows();
  let chunk = (count + 255u) / 256u;
  let begin = min(lane * chunk, count);
  let end = min(begin + chunk, count);
  var laneTotal = 0u;
  for (var row = begin; row < end; row += 1u) {
    laneTotal += atomicLoad(&scratch[faceCountBase() + row]);
  }
  prefixTotals[lane] = laneTotal;
  workgroupBarrier();
  for (var offset = 1u; offset < 256u; offset *= 2u) {
    var addend = 0u;
    if (lane >= offset) { addend = prefixTotals[lane - offset]; }
    workgroupBarrier();
    prefixTotals[lane] += addend;
    workgroupBarrier();
  }
  var total = prefixTotals[lane] - laneTotal;
  for (var row = begin; row < end; row += 1u) {
    atomicStore(&scratch[faceBaseBase() + row], total);
    total += atomicLoad(&scratch[faceCountBase() + row]);
  }
  if (lane == 255u && !topologyReused()) {
    total = prefixTotals[lane];
    if (total > params.capacities.y) { fail(ERROR_CAPACITY); total = 0u; }
    atomicStore(&authority[2], total);
    faceDispatch[0] = (total + 63u) / 64u;
    var directoryCapacity=2u;
    while(directoryCapacity<2u*total&&directoryCapacity<params.capacities.w){directoryCapacity*=2u;}
    if(directoryCapacity<2u*total||directoryCapacity>params.capacities.w){fail(ERROR_CAPACITY);directoryCapacity=0u;}
    atomicStore(&authority[6],directoryCapacity);
    faceDispatch[3]=(directoryCapacity+63u)/64u;
  }
}

fn writeFace(faceId: u32, row: u32, neighbor: OctreeOwnerPageLookupResult, axis: u32,
    subSize: u32, rowSize: u32, geometry: vec3u) {
  let neighborRow = findRow(neighbor.origin, neighbor.size);
  let physicalArea = select(select(params.cellSize.x * params.cellSize.y,
      params.cellSize.x * params.cellSize.z, axis == 1u),
      params.cellSize.y * params.cellSize.z, axis == 0u) * f32(subSize * subSize);
  let distance = 0.5 * (f32(rowSize) + f32(neighbor.size)) * params.cellSize[axis];
  // No compact positive row denotes the free-surface Dirichlet side. The
  // geometric centre distance is a conservative staged value; the dynamics
  // producer replaces it with its phi-clamped ghost-fluid distance.
  let closedBoundary = geometry[axis] == params.dimensionsMaximumLeaf[axis]
    && (params.boundary.x & (1u << (2u * axis + 1u))) != 0u;
  faces[faceId] = Face(row, neighborRow, axis,
    subSize | select(0u, FACE_CLOSED_BOUNDARY, closedBoundary),
    physicalArea, 1.0 / distance, select(1.0, 0.0, closedBoundary), 0.0);
  let packedAxisSpan = axis | ((31u - countLeadingZeros(subSize)) << 2u);
  faceGeometry[faceId] = vec4u(packedAxisSpan, geometry);
  // Staged extension publication: every geometric face remains a projection
  // seed until compact coarse-phi layers are added to this reduced seam.
  faceMetrics[faceId] = vec4u(axis, 1u, bitcast<u32>(0.0), 0u);
  atomicAdd(&scratch[incidenceCountBase() + row], 1u);
  if (neighborRow != INVALID) {
    atomicAdd(&scratch[incidenceCountBase() + neighborRow], 1u);
  }
}

fn writeNegativeBoundaryFace(faceId: u32, row: u32, neighborSize: u32, axis: u32,
    subSize: u32, rowSize: u32, geometry: vec3u) {
  let physicalArea = select(select(params.cellSize.x * params.cellSize.y,
      params.cellSize.x * params.cellSize.z, axis == 1u),
      params.cellSize.y * params.cellSize.z, axis == 0u) * f32(subSize * subSize);
  let distance = 0.5 * (f32(rowSize) + f32(neighborSize)) * params.cellSize[axis];
  // High bit records that the compact row is geometrically on the positive
  // side. Projection uses it to preserve the +axis pressure-gradient sign.
  let closedBoundary = geometry[axis] == 0u
    && (params.boundary.x & (1u << (2u * axis))) != 0u;
  faces[faceId] = Face(row, INVALID, axis,
    FACE_ROW_ON_POSITIVE_SIDE | subSize
      | select(0u, FACE_CLOSED_BOUNDARY, closedBoundary),
    physicalArea, 1.0 / distance, select(1.0, 0.0, closedBoundary), 0.0);
  let packedAxisSpan = axis | ((31u - countLeadingZeros(subSize)) << 2u);
  faceGeometry[faceId] = vec4u(packedAxisSpan, geometry);
  faceMetrics[faceId] = vec4u(axis, 1u, bitcast<u32>(0.0), 0u);
  atomicAdd(&scratch[incidenceCountBase() + row], 1u);
}

fn denseSolid(cell:vec3i)->vec2f{
 let dimensions=params.dimensionsMaximumLeaf.xyz;
 if(any(cell<vec3i(0))||any(vec3u(cell)>=dimensions)){return vec2f(0.,-1.);}
 let q=vec3u(cell);let index=q.x+dimensions.x*(q.y+dimensions.y*q.z);let word=2u*index;
 if(word+1u>=arrayLength(&solidCells)){return vec2f(0.,-1.);}
 let fraction=bitcast<f32>(solidCells[word]);let owner=bitcast<i32>(solidCells[word+1u]);
 return vec2f(clamp(select(0.,fraction,fraction==fraction),0.,1.),f32(owner));
}
fn solidNormal(owner:i32,axis:u32,world:vec3f)->f32{
 if(owner<0||u32(owner)>=arrayLength(&rigidBodies)){return 0.;}
 let body=rigidBodies[u32(owner)];
 let velocity=body.linearVelocity.xyz+cross(body.angularVelocity.xyz,world-body.positionShape.xyz);
 return velocity[axis];
}
fn quaternionRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);let uuv=cross(q.yzw,uv);return v+2.*(q.x*uv+uuv);}
fn quaternionInverseRotate(q:vec4f,v:vec3f)->vec3f{return quaternionRotate(vec4f(q.x,-q.yzw),v);}
fn rigidSdf(body:RigidBody,world:vec3f)->f32{
 let q=quaternionInverseRotate(body.orientation,world-body.positionShape.xyz);
 let d=body.dimensions.xyz;let shape=i32(round(body.positionShape.w));
 if(shape==0){return length(q)-d.x;}
 if(shape==1){let b=abs(q)-.5*d;return length(max(b,vec3f(0)))+min(max(b.x,max(b.y,b.z)),0.);}
 if(shape==2){let cy=clamp(q.y,-.5*d.y,.5*d.y);return length(vec3f(q.x,q.y-cy,q.z))-d.x;}
 let radial=length(q.xz)-d.x;let axial=abs(q.y)-.5*d.y;
 return length(max(vec2f(radial,axial),vec2f(0)))+min(max(radial,axial),0.);
}
@compute @workgroup_size(64)
fn conditionLosassoFaces(@builtin(global_invocation_id) invocation:vec3u){
 let faceId=invocation.x;if(faceId>=atomicLoad(&authority[2])||atomicLoad(&authority[4])!=0u){return;}
 var face=faces[faceId];let geometry=faceGeometry[faceId];let axis=face.axis;
 let span=face.reserved&0xffffu;let origin=geometry.yzw;
 if(span==0u||span>2896u){fail(ERROR_CAPACITY);return;}
 var exactOpen:array<i32,36>;var exactSolidVelocity:array<i32,36>;var exactSolidWeight:array<i32,36>;var valid=true;
 for(var b=0u;b<span;b+=1u){for(var a=0u;a<span;a+=1u){
  var q=origin+tangentA(axis)*vec3u(a)+tangentB(axis)*vec3u(b);
  let positive=denseSolid(vec3i(q));let negative=denseSolid(vec3i(q)-vec3i(axisVector(axis)));
  let selected=select(negative,positive,positive.x>negative.x);var solid=max(negative.x,positive.x);
  var centre=vec3f(q);for(var component=0u;component<3u;component+=1u){if(component!=axis){centre[component]+=.5;}}
  let owner=i32(selected.y);
  // Cell occupancy is only a broad-phase owner attribution.  Resolve the
  // actual blocked area on the face plane from four analytic SDF samples,
  // matching the structured lane's clamp(.5+sdf/h) aperture model.  Terrain
  // has no analytic rigid owner and retains its dense fractional fallback.
  var solidVelocity=0.;
  if(owner>=0&&u32(owner)<arrayLength(&rigidBodies)){
   solid=0.;let body=rigidBodies[u32(owner)];
   for(var sample=0u;sample<4u;sample+=1u){
    var point=centre;
    let ta=tangentA(axis);let tb=tangentB(axis);
    point+=(select(.25,.75,(sample&1u)!=0u)-.5)*vec3f(ta);
    point+=(select(.25,.75,(sample&2u)!=0u)-.5)*vec3f(tb);
    let world=params.domainOrigin.xyz+point*params.cellSize.xyz;
    let aperture=clamp(.5+rigidSdf(body,world)/max(params.cellSize[axis],1e-20),0.,1.);
    let blocked=.25*(1.-aperture);solid+=blocked;
    solidVelocity+=blocked*solidNormal(owner,axis,world);
   }
  }else if(solid>0.){
   let world=params.domainOrigin.xyz+centre*params.cellSize.xyz;
   solidVelocity=solid*solidNormal(owner,axis,world);
  }
  losassoAddExact(&exactOpen,1.-solid);
  if(solid>0.){
   let velocityTerm=solidVelocity;if(!losassoFinite(velocityTerm)){valid=false;}
   else{losassoAddExact(&exactSolidVelocity,velocityTerm);}losassoAddExact(&exactSolidWeight,solid);
  }
 }}
 let openSum=losassoExactValue(&exactOpen);let solidVelocitySum=losassoExactValue(&exactSolidVelocity);
 let solidWeight=losassoExactValue(&exactSolidWeight);if(!valid||!losassoFinite(openSum)||!losassoFinite(solidVelocitySum)
   ||!losassoFinite(solidWeight)){fail(ERROR_HEADER);return;}
 let closedBoundary=(face.reserved&FACE_CLOSED_BOUNDARY)!=0u;
 face.openFraction=select(openSum/f32(span*span),0.,closedBoundary);
 face.normalVelocity=select(select(0.,solidVelocitySum/solidWeight,solidWeight>0.),0.,closedBoundary);
 // Candidate rows have no accepted-index phi. Keep the geometric dual
 // distance here; the accepted fine-to-coarse exchange conditions free-
 // surface faces and republishes the matching diagonal after ready commit.
 faces[faceId]=face;
}

@compute @workgroup_size(64)
fn emitLosassoFaces(@builtin(global_invocation_id) invocation: vec3u) {
  let row = invocation.x;
  if (row >= rows() || atomicLoad(&authority[4]) != 0u) { return; }
  let header = leafHeaders[row];
  let origin = originOf(header);
  var faceId = atomicLoad(&scratch[faceBaseBase() + row]);
  for (var axis = 0u; axis < 3u; axis += 1u) {
    let negativePlan = negativeBoundaryPlan(header, axis);
    let negativeSubdivision = negativePlan.subdivision;
    let negativeSubSize = header.size / negativeSubdivision;
    for (var b = 0u; b < negativeSubdivision; b += 1u) {
      for (var a = 0u; a < negativeSubdivision; a += 1u) {
        let patchBit = 1u << (a + negativeSubdivision * b);
        if ((negativePlan.mask & patchBit) != 0u) {
          var neighborSize=header.size;
          if(origin[axis]>0u){let owner=octreeOwnerPageLookup(negativeProbe(origin,header.size,axis,
            a,b,negativeSubdivision));neighborSize=select(header.size,owner.size,
            (owner.status&OWNER_PAGE_LOOKUP_MISSING)==0u);}
          let geometry = origin + tangentA(axis) * vec3u(a * negativeSubSize)
            + tangentB(axis) * vec3u(b * negativeSubSize);
          writeNegativeBoundaryFace(faceId, row, neighborSize, axis,
            negativeSubSize, header.size, geometry);
          faceId += 1u;
        }
      }
    }
    let plan = positiveFacePlan(header, axis);
    let subdivision = plan.subdivision;
    let subSize = header.size / subdivision;
    for (var b = 0u; b < subdivision; b += 1u) {
      for (var a = 0u; a < subdivision; a += 1u) {
        let patchBit = 1u << (a + subdivision * b);
        if ((plan.mask & patchBit) != 0u) {
          var owner:OctreeOwnerPageLookupResult;
          if(origin[axis]+header.size>=params.dimensionsMaximumLeaf[axis]){
            owner.origin=origin+axisVector(axis)*vec3u(header.size);
            owner.size=header.size;owner.status=0u;
          }else{owner=octreeOwnerPageLookup(positiveProbe(origin, header.size, axis,
            a, b, subdivision));}
          if ((owner.status & OWNER_PAGE_LOOKUP_INVALID) != 0u) {
            fail(ERROR_OWNER);
          } else {
            if ((owner.status & OWNER_PAGE_LOOKUP_MISSING) != 0u) {
              owner.origin=vec3u(positiveProbe(origin,header.size,axis,a,b,subdivision));
              owner.size=header.size;owner.status=0u;
            }
            let geometry = origin + axisVector(axis) * vec3u(header.size)
              + tangentA(axis) * vec3u(a * subSize)
              + tangentB(axis) * vec3u(b * subSize);
            writeFace(faceId, row, owner, axis, subSize, header.size, geometry);
          }
          faceId += 1u;
        }
      }
    }
  }
}

@compute @workgroup_size(256)
fn prefixLosassoIncidences(@builtin(local_invocation_index) lane: u32) {
  let count = rows();
  let chunk = (count + 255u) / 256u;
  let begin = min(lane * chunk, count);
  let end = min(begin + chunk, count);
  var laneTotal = 0u;
  for (var row = begin; row < end; row += 1u) {
    laneTotal += atomicLoad(&scratch[incidenceCountBase() + row]);
  }
  prefixTotals[lane] = laneTotal;
  workgroupBarrier();
  for (var offset = 1u; offset < 256u; offset *= 2u) {
    var addend = 0u;
    if (lane >= offset) { addend = prefixTotals[lane - offset]; }
    workgroupBarrier();
    prefixTotals[lane] += addend;
    workgroupBarrier();
  }
  var total = prefixTotals[lane] - laneTotal;
  for (var row = begin; row < end; row += 1u) {
    rowFaceOffsets[row] = total;
    total += atomicLoad(&scratch[incidenceCountBase() + row]);
    atomicStore(&scratch[incidenceCursorBase() + row], 0u);
  }
  if (lane == 255u && !topologyReused()) {
    total = prefixTotals[lane];
    rowFaceOffsets[count] = total;
    if (total > params.capacities.z) { fail(ERROR_CAPACITY); }
  }
}

@compute @workgroup_size(64)
fn scatterLosassoIncidences(@builtin(global_invocation_id) invocation: vec3u) {
  let faceId = invocation.x;
  if (faceId >= atomicLoad(&authority[2]) || atomicLoad(&authority[4]) != 0u) { return; }
  let face = faces[faceId];
  let negativeSlot = atomicAdd(&scratch[incidenceCursorBase() + face.negativeRow], 1u);
  rowFaces[rowFaceOffsets[face.negativeRow] + negativeSlot] = faceId;
  if (face.positiveRow != INVALID) {
    let positiveSlot = atomicAdd(&scratch[incidenceCursorBase() + face.positiveRow], 1u);
    rowFaces[rowFaceOffsets[face.positiveRow] + positiveSlot] = faceId;
  }
}

@compute @workgroup_size(64)
fn sortLosassoIncidences(@builtin(global_invocation_id) invocation: vec3u) {
  let row = invocation.x;
  if (row >= rows() || atomicLoad(&authority[4]) != 0u) { return; }
  let begin = rowFaceOffsets[row];
  let end = rowFaceOffsets[row + 1u];
  if (begin > end || end > arrayLength(&rowFaces) || end - begin > 24u) {
    fail(ERROR_CAPACITY); diagonal[row] = 0.0; rhs[row] = 0.0; return;
  }
  for (var cursor = begin + 1u; cursor < end; cursor += 1u) {
    let value = rowFaces[cursor];
    var insertion = cursor;
    while (insertion > begin && rowFaces[insertion - 1u] > value) {
      rowFaces[insertion] = rowFaces[insertion - 1u];
      insertion -= 1u;
    }
    rowFaces[insertion] = value;
  }
  var diagonalSum=CompensatedF32(0.,0.);
  var fluxSum=CompensatedF32(0.,0.);
  var valid=true;var allSolidBlocked=end>begin;
  for (var cursor = begin; cursor < end; cursor += 1u) {
    let faceId=rowFaces[cursor];if(faceId>=atomicLoad(&authority[2])||faceId>=arrayLength(&faces)){valid=false;continue;}
    let face = faces[faceId];
    let coefficient=(face.openFraction * face.area) * face.inverseDistance;
    let sign = select(-1.0, 1.0, face.negativeRow == row);
    let flux=sign * face.openFraction * face.area * face.normalVelocity;
    if(!losassoFinite(coefficient)||coefficient<0.||!losassoFinite(flux)){valid=false;continue;}
    allSolidBlocked=allSolidBlocked&&face.openFraction<=1e-7;
    diagonalSum=addCompensatedF32(diagonalSum,coefficient);
    fluxSum=addCompensatedF32(fluxSum,flux);
  }
  let diagonalValue=compensatedValue(diagonalSum);
  let integratedFlux=compensatedValue(fluxSum);
  if(valid&&allSolidBlocked&&losassoFinite(integratedFlux)){
    // Degenerate all-solid wet rows are harmless identities, not malformed
    // topology.  The wet classifier normally removes them; this fallback
    // keeps a raster/publication race from poisoning the complete candidate.
    diagonal[row]=1.;rhs[row]=0.;return;
  }
  if(!valid||!losassoFinite(diagonalValue)||!losassoFinite(integratedFlux)||!(diagonalValue>0.)){
    fail(ERROR_HEADER);diagonal[row]=0.;rhs[row]=0.;return;}
  diagonal[row] = diagonalValue;
  rhs[row] = integratedFlux;
}

@compute @workgroup_size(64)
fn clearLosassoFaceDirectory(@builtin(global_invocation_id) invocation: vec3u) {
  if(topologyReused()){return;}
  let slot = invocation.x;
  if (slot < atomicLoad(&authority[6])) { faceDirectory[slot] = vec2u(0u); }
}

@compute @workgroup_size(64)
fn clearLosassoAdjacencyOffsets(@builtin(global_invocation_id) invocation: vec3u) {
  let face = invocation.x;
  let count = atomicLoad(&authority[2]);
  if (face < count) { adjacencyOffsets[face] = 0u; }
  if (face == 0u) { adjacencyOffsets[count] = 0u; }
}

@compute @workgroup_size(1)
fn finishLosassoPublication() {
  if(topologyReused()){return;}
  let directoryCapacity = atomicLoad(&authority[6]);
  let mask = directoryCapacity - 1u;
  let count = atomicLoad(&authority[2]);
  for (var face = 0u; face < count; face += 1u) {
    let geometry = faceGeometry[face];
    var hash = (geometry.x + 1u) * 0x9e3779b1u;
    hash = (hash ^ geometry.y) * 0x85ebca6bu;
    hash = (hash ^ geometry.z) * 0xc2b2ae35u;
    hash = (hash ^ geometry.w) * 0x27d4eb2du;
    var installed = false;
    for (var probe = 0u; probe < 32u; probe += 1u) {
      let slot = (hash + probe) & mask;
      if (faceDirectory[slot].x == 0u) {
        faceDirectory[slot] = vec2u(face + 1u, hash);
        installed = true;
        break;
      }
    }
    if (!installed) { fail(ERROR_CAPACITY); }
  }
  let valid = atomicLoad(&authority[4]) == 0u && atomicLoad(&authority[0]) != 0u;
  if (valid) {
    atomicStore(&authority[3], 1u);
    rowDispatch[0] = (rows() + 63u) / 64u;
    faceDispatch[0] = (atomicLoad(&authority[2]) + 63u) / 64u;
    solverAuthority[0] = 0u;
    solverAuthority[1] = INVALID;
    solverAuthority[2] = rows();
    solverAuthority[3] = count;
    solverAuthority[4] = atomicLoad(&authority[0]);
    solverAuthority[5] = 0u;
    solverAuthority[6] = atomicLoad(&authority[0]);
  } else {
    atomicStore(&authority[3], 0u);
    rowDispatch[0] = 0u;
    faceDispatch[0] = 0u;
    solverAuthority[0] = atomicLoad(&authority[4]);
  }
}
`;
