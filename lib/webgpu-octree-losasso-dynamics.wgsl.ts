/** Reduced first-order axis-face dynamics for the Losasso coarse backend. */
export const octreeLosassoDynamicsWGSL = /* wgsl */ `
const INVALID_FACE: u32 = 0xffffffffu;
const INVALID_ROW: u32 = 0xffffffffu;

struct Params {
  dimensionsMaximumLeaf: vec4u,
  capacities: vec4u, // faces, rows, directory, closed-boundary bit mask
  domainOriginDt: vec4f,
  cellWidthDensity: vec4f,
  gravity: vec4f,
  inflowPositionRadius: vec4f,
  inflowVelocityEnabled: vec4f,
  band: vec4u,
};
struct Face {
  negativeRow: u32,
  positiveRow: u32,
  axis: u32,
  span: u32,
  area: f32,
  inverseDistance: f32,
  openFraction: f32,
  solidNormalVelocity: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> authority: array<u32>;
@group(0) @binding(2) var<storage, read> faces: array<Face>;
@group(0) @binding(3) var<storage, read> faceGeometry: array<vec4u>;
@group(0) @binding(4) var<storage, read> faceDirectory: array<vec2u>;
@group(0) @binding(5) var<storage, read> extendedVelocity: array<f32>;
@group(0) @binding(6) var<storage, read_write> advectedVelocity: array<f32>;
@group(0) @binding(7) var<storage, read_write> predictedVelocity: array<f32>;
@group(0) @binding(8) var<storage, read> rowFaceOffsets: array<u32>;
@group(0) @binding(9) var<storage, read> rowFaces: array<u32>;
@group(0) @binding(10) var<storage, read_write> rightHandSide: array<f32>;
@group(0) @binding(12) var<storage, read> bandControl: array<u32>;
@group(0) @binding(13) var<storage, read> bandGeometry: array<vec4u>;
@group(0) @binding(14) var<storage, read> bandDirectory: array<vec2u>;
@group(0) @binding(15) var<storage, read> bandVelocity: array<f32>;

fn finite(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823e38;
}
fn addExact(limbs:ptr<function,array<i32,36>>,value:f32){let bits=bitcast<u32>(value);let magnitude=bits&0x7fffffffu;
  if(magnitude==0u){return;}let rawExponent=(magnitude>>23u)&0xffu;let fraction=magnitude&0x7fffffu;
  let significand=select(fraction,0x800000u|fraction,rawExponent!=0u);let shift=select(3u,rawExponent+2u,rawExponent!=0u);
  let firstLimb=shift>>3u;let shifted=significand<<(shift&7u);let sign=select(1,-1,(bits&0x80000000u)!=0u);
  for(var digit=0u;digit<4u;digit+=1u){let limb=firstLimb+digit;let byte=i32((shifted>>(digit*8u))&0xffu);
    if(byte!=0&&limb<36u){(*limbs)[limb]+=sign*byte;}}}
fn floorDiv256(value:i32)->vec2i{var carry=value/256;var digit=value-carry*256;if(digit<0){digit+=256;carry-=1;}return vec2i(carry,digit);}
fn exactValue(source:ptr<function,array<i32,36>>)->f32{var limbs=(*source);
  for(var limb=0u;limb+1u<36u;limb+=1u){let normalized=floorDiv256(limbs[limb]);limbs[limb]=normalized.y;limbs[limb+1u]+=normalized.x;}
  let negative=limbs[35]<0;if(negative){for(var limb=0u;limb<36u;limb+=1u){limbs[limb]=-limbs[limb];}
    for(var limb=0u;limb+1u<36u;limb+=1u){let normalized=floorDiv256(limbs[limb]);limbs[limb]=normalized.y;limbs[limb+1u]+=normalized.x;}}
  var magnitude=0.;for(var limb=0u;limb<36u;limb+=1u){magnitude+=ldexp(f32(limbs[limb]),-152+i32(limb*8u));}
  return select(magnitude,-magnitude,negative);}
fn liveFaces() -> u32 {
  if (arrayLength(&authority) < 4u || authority[3] != 1u) { return 0u; }
  return min(authority[2], params.capacities.x);
}
fn liveRows() -> u32 {
  if (arrayLength(&authority) < 4u || authority[3] != 1u) { return 0u; }
  return min(authority[1], params.capacities.y);
}
fn faceHash(packedAxisSpan: u32, coordinate: vec3u) -> u32 {
  var value = (packedAxisSpan + 1u) * 0x9e3779b1u;
  value = (value ^ coordinate.x) * 0x85ebca6bu;
  value = (value ^ coordinate.y) * 0xc2b2ae35u;
  return (value ^ coordinate.z) * 0x27d4eb2du;
}
fn exactFace(packedAxisSpan: u32, coordinate: vec3u) -> u32 {
  let capacity = params.band.y;
  if (capacity == 0u || (capacity & (capacity - 1u)) != 0u) {
    return INVALID_FACE;
  }
  let hash = faceHash(packedAxisSpan, coordinate);
  let mask = capacity - 1u;
  for (var probe = 0u; probe < 32u; probe += 1u) {
    let record = bandDirectory[(hash + probe) & mask];
    if (record.x == 0u) { return INVALID_FACE; }
    let face = record.x - 1u;
    if (record.y == hash && arrayLength(&bandControl) >= 4u && bandControl[3] == 1u
        && face < min(bandControl[2], params.band.x)
        && face < arrayLength(&bandGeometry)
        && all(bandGeometry[face] == vec4u(packedAxisSpan, coordinate))) {
      return face;
    }
  }
  return INVALID_FACE;
}

// A stored record covers a dyadic square in its two tangential directions.
// Search fine-to-coarse so a transition subface wins over the coarse record.
fn containingFace(axis: u32, coordinate: vec3u) -> u32 {
  var span = 1u;
  var logSpan = 0u;
  while (span <= params.dimensionsMaximumLeaf.w) {
    var origin = coordinate;
    for (var component = 0u; component < 3u; component += 1u) {
      if (component != axis) { origin[component] = (origin[component] / span) * span; }
    }
    let face = exactFace(axis | (logSpan << 2u), origin);
    if (face != INVALID_FACE) { return face; }
    if (span > params.dimensionsMaximumLeaf.w / 2u) { break; }
    span *= 2u;
    logSpan += 1u;
  }
  return INVALID_FACE;
}

struct VelocitySample { value: vec3f, valid: bool };

fn closedDomainFace(axis: u32, coordinate: vec3u) -> bool {
  let side = select(2u * axis + 1u, 2u * axis, coordinate[axis] == 0u);
  let onBoundary = coordinate[axis] == 0u
    || coordinate[axis] == params.dimensionsMaximumLeaf[axis];
  return onBoundary && (params.capacities.w & (1u << side)) != 0u;
}

// First-order staggered sampling. Piecewise-constant tangential sampling is
// intentional: it is conservative at a 2:1 face and requires no virtual fine
// velocities or Power edge channels.
fn velocityAtGrid(gridValue: vec3f) -> VelocitySample {
  let grid = clamp(gridValue, vec3f(0.0), vec3f(params.dimensionsMaximumLeaf.xyz));
  var result = vec3f(0.0);
  for (var axis = 0u; axis < 3u; axis += 1u) {
    var coordinate = vec3i(floor(grid));
    coordinate[axis] = i32(floor(grid[axis] + 0.5));
    var upper = vec3i(params.dimensionsMaximumLeaf.xyz) - vec3i(1);
    upper[axis] = i32(params.dimensionsMaximumLeaf[axis]);
    coordinate = clamp(coordinate, vec3i(0), upper);
    // An even-span face centre can lie exactly between two first-order
    // tangential samples. A one-sided floor chooses the upper cell on both
    // reflected sides and is therefore not D4-equivariant. Average only the
    // exact tie (two samples, or four at a tangential corner); all other
    // positions retain the piecewise-constant first-order sample.
    var splitMask = 0u;
    var tangent = 0u;
    for (var component = 0u; component < 3u; component += 1u) {
      if (component == axis) { continue; }
      if (grid[component] == floor(grid[component]) && coordinate[component] > 0) {
        splitMask |= 1u << tangent;
      }
      tangent += 1u;
    }
    var exact: array<i32,36>;
    var sampleCount = 0u;
    for (var candidate = 0u; candidate < 4u; candidate += 1u) {
      if ((candidate & ~splitMask) != 0u) { continue; }
      var sampleCoordinate = coordinate;
      tangent = 0u;
      for (var component = 0u; component < 3u; component += 1u) {
        if (component == axis) { continue; }
        if ((candidate & (1u << tangent)) != 0u) { sampleCoordinate[component] -= 1; }
        tangent += 1u;
      }
      let unsignedCoordinate = vec3u(sampleCoordinate);
      let face = containingFace(axis, unsignedCoordinate);
      var componentValue = 0.0;
      if (face == INVALID_FACE || face >= arrayLength(&bandVelocity)) {
        if (!closedDomainFace(axis, unsignedCoordinate)) {
          return VelocitySample(vec3f(0.0), false);
        }
      } else {
        componentValue = bandVelocity[face];
        if (!finite(componentValue)) { return VelocitySample(vec3f(0.0), false); }
      }
      addExact(&exact, componentValue);
      sampleCount += 1u;
    }
    if (sampleCount == 0u) { return VelocitySample(vec3f(0.0), false); }
    result[axis] = exactValue(&exact) / f32(sampleCount);
  }
  return VelocitySample(result, true);
}

fn faceCenterGrid(faceId: u32) -> vec3f {
  let record = faceGeometry[faceId];
  let axis = record.x & 3u;
  let span = 1u << (record.x >> 2u);
  var coordinate = vec3f(record.yzw);
  for (var component = 0u; component < 3u; component += 1u) {
    if (component != axis) { coordinate[component] += 0.5 * f32(span); }
  }
  return coordinate;
}

fn faceCenter(faceId: u32) -> vec3f {
  return params.domainOriginDt.xyz + faceCenterGrid(faceId) * params.cellWidthDensity.xyz;
}

fn inflowAxis() -> u32 {
  let magnitude = abs(params.inflowVelocityEnabled.xyz);
  if (magnitude.y > magnitude.x && magnitude.y >= magnitude.z) { return 1u; }
  if (magnitude.z > magnitude.x) { return 2u; }
  return 0u;
}

// Return {prescribed normal velocity, matched}. The authored aperture scale
// and time-ramped strength are already applied to inflowVelocityEnabled.xyz.
// Coverage is evaluated at the actual dyadic face centre, with an edge width
// derived from that face's tangential span, so D4 transition subfaces neither
// double-source nor inherit a fine-grid-only footprint.
fn inflowNormalVelocity(faceId: u32) -> vec2f {
  if (params.inflowVelocityEnabled.w < 0.5
      || params.inflowPositionRadius.w <= 0.0) { return vec2f(0.0); }
  let record = faceGeometry[faceId];
  let axis = record.x & 3u;
  if (axis != inflowAxis() || params.dimensionsMaximumLeaf[axis] < 2u) {
    return vec2f(0.0);
  }
  let plane = u32(clamp(i32(round((params.inflowPositionRadius[axis]
    - params.domainOriginDt[axis]) / params.cellWidthDensity[axis])),
    1, i32(params.dimensionsMaximumLeaf[axis]) - 1));
  if (record[axis + 1u] != plane) { return vec2f(0.0); }
  let velocity = params.inflowVelocityEnabled.xyz;
  let speed = length(velocity);
  if (speed <= 1e-6 || !finite(speed)) { return vec2f(0.0); }
  let direction = velocity / speed;
  let relative = faceCenter(faceId) - params.inflowPositionRadius.xyz;
  let axial = dot(relative, direction);
  let radial = length(relative - axial * direction);
  let span = f32(1u << (record.x >> 2u));
  var tangentWidth = vec2f(0.0);
  var tangent = 0u;
  for (var component = 0u; component < 3u; component += 1u) {
    if (component != axis) {
      tangentWidth[tangent] = span * params.cellWidthDensity[component];
      tangent += 1u;
    }
  }
  let edge = max(0.5 * length(tangentWidth), 1e-6);
  var coverage = clamp(0.5 + 0.5
    * (params.inflowPositionRadius.w - radial) / edge, 0.0, 1.0);
  coverage = coverage * coverage * (3.0 - 2.0 * coverage);
  if (coverage <= 0.0) { return vec2f(0.0); }
  let prescribed = velocity[axis] * coverage;
  return vec2f(select(0.0, prescribed, finite(prescribed)), 1.0);
}

@compute @workgroup_size(64)
fn advectLosassoFaces(@builtin(global_invocation_id) invocation: vec3u) {
  let faceId = invocation.x;
  if (faceId >= liveFaces() || faceId >= arrayLength(&advectedVelocity)) { return; }
  let centre = faceCenterGrid(faceId);
  let prior = velocityAtGrid(centre);
  let previous = select(0.0, prior.value[faces[faceId].axis], prior.valid);
  if (params.domainOriginDt.w == 0.0) {
    advectedVelocity[faceId] = select(0.0, previous, finite(previous));
    return;
  }
  let carrier = velocityAtGrid(centre);
  var value = previous;
  if (carrier.valid) {
    let departure = centre
      - (params.domainOriginDt.w / params.cellWidthDensity.xyz) * carrier.value;
    let sample = velocityAtGrid(departure);
    if (sample.valid) { value = sample.value[faces[faceId].axis]; }
  }
  advectedVelocity[faceId] = select(0.0, value, finite(value));
}

@compute @workgroup_size(64)
fn forceLosassoFaces(@builtin(global_invocation_id) invocation: vec3u) {
  let faceId = invocation.x;
  if (faceId >= liveFaces() || faceId >= arrayLength(&advectedVelocity)
      || faceId >= arrayLength(&predictedVelocity)) { return; }
  if (params.domainOriginDt.w == 0.0) {
    let previous = advectedVelocity[faceId];
    predictedVelocity[faceId] = select(0.0, previous, finite(previous));
    return;
  }
  let face = faces[faceId];
  let fluid = advectedVelocity[faceId]
    + params.domainOriginDt.w * params.gravity[face.axis];
  let inflow = inflowNormalVelocity(faceId);
  let drivenFluid = select(fluid, inflow.x, inflow.y > 0.5);
  // Aperture blends the resolved fluid face toward the prescribed moving-solid
  // normal velocity. Closed faces therefore contribute exactly the solid flux.
  let value = face.solidNormalVelocity
    + face.openFraction * (drivenFluid - face.solidNormalVelocity);
  predictedVelocity[faceId] = select(face.solidNormalVelocity, value, finite(value));
}

@group(0) @binding(11) var<storage, read_write> projectedVelocity: array<f32>;

@compute @workgroup_size(64)
fn constrainLosassoInflowFaces(@builtin(global_invocation_id) invocation: vec3u) {
  let faceId = invocation.x;
  if (params.inflowVelocityEnabled.w < 0.5 || faceId >= liveFaces()
      || faceId >= arrayLength(&projectedVelocity)) { return; }
  let inflow = inflowNormalVelocity(faceId);
  if (inflow.y <= 0.5) { return; }
  let face = faces[faceId];
  let value = face.solidNormalVelocity
    + face.openFraction * (inflow.x - face.solidNormalVelocity);
  projectedVelocity[faceId] = select(face.solidNormalVelocity, value, finite(value));
}

@compute @workgroup_size(64)
fn divergenceLosassoRows(@builtin(global_invocation_id) invocation: vec3u) {
  let row = invocation.x;
  if (row >= liveRows() || row + 1u >= arrayLength(&rowFaceOffsets)
      || row >= arrayLength(&rightHandSide)) { return; }
  if (params.domainOriginDt.w == 0.0) {
    rightHandSide[row] = 0.0;
    return;
  }
  let begin = rowFaceOffsets[row];
  let end = rowFaceOffsets[row + 1u];
  if (begin > end || end > arrayLength(&rowFaces) || end - begin > 24u) {
    rightHandSide[row] = 3.402823e38;
    return;
  }
  var exactFlux: array<i32, 36>;
  var validTerms = true;
  for (var cursor = begin; cursor < end; cursor += 1u) {
    let faceId = rowFaces[cursor];
    if (faceId >= liveFaces() || faceId >= arrayLength(&predictedVelocity)
        || faceId >= arrayLength(&faces)) { validTerms = false; continue; }
    let face = faces[faceId];
    let rowOnPositiveSide = (face.span & 0x80000000u) != 0u;
    let orientation = select(-1.0, 1.0,
      face.negativeRow == row && !rowOnPositiveSide);
    let term = orientation * face.area * predictedVelocity[faceId];
    if (!finite(term)) { validTerms = false; } else { addExact(&exactFlux, term); }
  }
  // A uses integrated area/distance coefficients and projection applies dt/rho,
  // so the matching pressure RHS is rho/dt times integrated outward flux.
  let scale = params.cellWidthDensity.w / max(params.domainOriginDt.w, 1e-12);
  let value = exactValue(&exactFlux) * scale;
  rightHandSide[row] = select(3.402823e38, value, validTerms && finite(value));
}
`;
