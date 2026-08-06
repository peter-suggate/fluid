/** Axis-face dynamics for the Losasso coarse backend. */
export const octreeLosassoDynamicsWGSL = /* wgsl */ `
const INVALID_FACE: u32 = 0xffffffffu;
const INVALID_ROW: u32 = 0xffffffffu;
const INVALID_VELOCITY: f32 = 3.402823e38;
const BAND_FIELD: u32 = 0u;
const ADVECTED_FIELD: u32 = 1u;
const FACE_INTERFACE_NEARBY: u32 = 0x10000000u;

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
@group(0) @binding(15) var<storage, read> bandVelocityArena: array<vec4u>;

fn finite(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823e38;
}
fn addExact(limbs:ptr<function,array<i32,36>>,value:f32){let bits=bitcast<u32>(value);let magnitude=bits&0x7fffffffu;
  if(magnitude==0u){return;}let rawExponent=(magnitude>>23u)&0xffu;let fraction=magnitude&0x7fffffu;
  let significand=select(fraction,0x800000u|fraction,rawExponent!=0u);let shift=select(3u,rawExponent+2u,rawExponent!=0u);
  let firstLimb=shift>>3u;let shifted=significand<<(shift&7u);let sign=select(1,-1,(bits&0x80000000u)!=0u);
  for(var digit=0u;digit<4u;digit+=1u){let limb=firstLimb+digit;let byte=i32((shifted>>(digit*8u))&0xffu);
    if(byte!=0&&limb<36u){(*limbs)[limb]+=sign*byte;}}}
fn floorDiv256(value:i32)->vec2i{let carry=value>>8;return vec2i(carry,value-carry*256);}
fn exactValue(source:ptr<function,array<i32,36>>)->f32{
  for(var limb=0u;limb+1u<36u;limb+=1u){let normalized=floorDiv256((*source)[limb]);(*source)[limb]=normalized.y;(*source)[limb+1u]+=normalized.x;}
  let negative=(*source)[35]<0;if(negative){for(var limb=0u;limb<36u;limb+=1u){(*source)[limb]=-(*source)[limb];}
    for(var limb=0u;limb+1u<36u;limb+=1u){let normalized=floorDiv256((*source)[limb]);(*source)[limb]=normalized.y;(*source)[limb+1u]+=normalized.x;}}
  var magnitude=0.;for(var limb=0u;limb<36u;limb+=1u){magnitude+=ldexp(f32((*source)[limb]),-152+i32(limb*8u));}
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

struct VelocitySample {
  value: vec3f,
  lower: vec3f,
  upper: vec3f,
  valid: bool,
  boundary: bool,
};

fn invalidVelocitySample() -> VelocitySample {
  return VelocitySample(vec3f(0.0), vec3f(0.0), vec3f(0.0), false, false);
}

fn symmetricProduct3(value: vec3f) -> f32 {
  let lowPair = min(value.x, value.y);
  let highPair = max(value.x, value.y);
  let low = min(lowPair, value.z);
  let high = max(highPair, value.z);
  let middle = max(lowPair, min(highPair, value.z));
  return (low * middle) * high;
}

fn quantizeTraceOffset(value: vec3f) -> vec3f {
  return round(value * 65536.0) / 65536.0;
}

fn closedDomainFace(axis: u32, coordinate: vec3u) -> bool {
  let side = select(2u * axis + 1u, 2u * axis, coordinate[axis] == 0u);
  let onBoundary = coordinate[axis] == 0u
    || coordinate[axis] == params.dimensionsMaximumLeaf[axis];
  return onBoundary && (params.capacities.w & (1u << side)) != 0u;
}

fn stagedNodeIndex(node: vec3u) -> u32 {
  let dimensions = params.dimensionsMaximumLeaf.xyz + vec3u(1u);
  return node.x + dimensions.x * (node.y + dimensions.y * node.z);
}

// Losasso et al. first reconstruct all three velocity components at octree
// nodes by averaging the incident MAC faces.  At a 2:1 transition,
// containingFace maps each fine quadrant to its owning coarse face; repeated
// ownership therefore gives the coarse value its covered-area weight without
// inventing virtual fine velocities.
fn velocityAtNode(node: vec3u, field: u32) -> VelocitySample {
  if (arrayLength(&bandControl) < 4u || bandControl[3] != 1u) {
    return invalidVelocitySample();
  }
  let nodeCount = (params.dimensionsMaximumLeaf.x + 1u)
    * (params.dimensionsMaximumLeaf.y + 1u) * (params.dimensionsMaximumLeaf.z + 1u);
  let at = stagedNodeIndex(node) + select(nodeCount, 0u, field == BAND_FIELD);
  if (at >= arrayLength(&bandVelocityArena)) { return invalidVelocitySample(); }
  let staged = bandVelocityArena[at];
  if ((staged.w & 1u) == 0u) { return invalidVelocitySample(); }
  let result = vec3f(bitcast<f32>(staged.x), bitcast<f32>(staged.y), bitcast<f32>(staged.z));
  if (!all(result == result) || any(abs(result) > vec3f(3.402823e38))) {
    return invalidVelocitySample();
  }
  return VelocitySample(result, result, result, true, (staged.w & 2u) != 0u);
}

// Trilinear interpolation of the reconstructed nodal field is the Section 5
// sampling rule.  Zero-weight corners are not fetched: sparse extension only
// has to cover the stencil that actually contributes to this sample.
fn velocityAtGrid(gridValue: vec3f, field: u32) -> VelocitySample {
  let grid = clamp(gridValue, vec3f(0.0), vec3f(params.dimensionsMaximumLeaf.xyz));
  let lower = vec3u(floor(grid));
  let upper = min(lower + vec3u(1u), params.dimensionsMaximumLeaf.xyz);
  let fraction = grid - vec3f(lower);
  var exactX: array<i32,36>;
  var exactY: array<i32,36>;
  var exactZ: array<i32,36>;
  var stencilLower = vec3f(3.402823e38);
  var stencilUpper = vec3f(-3.402823e38);
  var boundary = any(grid != gridValue);
  for (var corner = 0u; corner < 8u; corner += 1u) {
    let high = vec3u(corner & 1u, (corner >> 1u) & 1u, (corner >> 2u) & 1u);
    let weightVector = select(vec3f(1.0) - fraction, fraction, high != vec3u(0u));
    let weight = symmetricProduct3(weightVector);
    if (weight == 0.0) { continue; }
    let node = select(lower, upper, high != vec3u(0u));
    let sample = velocityAtNode(node, field);
    if (!sample.valid) { return invalidVelocitySample(); }
    boundary = boundary || sample.boundary;
    addExact(&exactX, weight * sample.value.x);
    addExact(&exactY, weight * sample.value.y);
    addExact(&exactZ, weight * sample.value.z);
    stencilLower = min(stencilLower, sample.value);
    stencilUpper = max(stencilUpper, sample.value);
  }
  let result = vec3f(exactValue(&exactX), exactValue(&exactY), exactValue(&exactZ));
  return VelocitySample(result, stencilLower, stencilUpper, true, boundary);
}

struct VelocityTrace {
  point: vec3f,
  valid: bool,
  boundary: bool,
};

// A signed midpoint characteristic. Positive dt backtraces the forward
// predictor; negative dt traces the predictor forward for MacCormack reversal.
fn traceVelocityFromFirst(
  origin: vec3f, dt: f32, field: u32, first: VelocitySample,
) -> VelocityTrace {
  if (!first.valid) { return VelocityTrace(origin, false, false); }
  let gridDt = dt / params.cellWidthDensity.xyz;
  let midpointOffset = quantizeTraceOffset(-0.5 * gridDt * first.value);
  let midpoint = velocityAtGrid(origin + midpointOffset, field);
  let carrier = select(first.value, midpoint.value, midpoint.valid);
  return VelocityTrace(origin + quantizeTraceOffset(-gridDt * carrier), true,
    first.boundary || (midpoint.valid && midpoint.boundary));
}

fn traceVelocity(origin: vec3f, dt: f32, field: u32) -> VelocityTrace {
  return traceVelocityFromFirst(origin, dt, field, velocityAtGrid(origin, field));
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
  let prior = velocityAtGrid(centre, BAND_FIELD);
  let previous = select(0.0, prior.value[faces[faceId].axis], prior.valid);
  if (params.domainOriginDt.w == 0.0) {
    advectedVelocity[faceId] = select(0.0, previous, finite(previous));
    return;
  }
  var value = previous;
  let departure = traceVelocityFromFirst(
    centre, params.domainOriginDt.w, BAND_FIELD, prior);
  if (departure.valid) {
    let sample = velocityAtGrid(departure.point, BAND_FIELD);
    if (sample.valid) { value = sample.value[faces[faceId].axis]; }
  }
  advectedVelocity[faceId] = select(0.0, value, finite(value));
}

// predictedVelocity is scratch here; the force pass overwrites it after the
// correction. An invalid reversal explicitly disables correction if the
// published W7 predictor support cannot provide a complete nodal stencil.
@compute @workgroup_size(64)
fn reverseLosassoFaces(@builtin(global_invocation_id) invocation: vec3u) {
  let faceId = invocation.x;
  if (faceId >= liveFaces() || faceId >= arrayLength(&predictedVelocity)) { return; }
  let centre = faceCenterGrid(faceId);
  let arrival = traceVelocity(centre, -params.domainOriginDt.w, ADVECTED_FIELD);
  if (arrival.valid && !arrival.boundary
      && (faces[faceId].span & FACE_INTERFACE_NEARBY) == 0u) {
    let reversed = velocityAtGrid(arrival.point, ADVECTED_FIELD);
    if (reversed.valid && !reversed.boundary) {
      predictedVelocity[faceId] = reversed.value[faces[faceId].axis];
      return;
    }
  }
  predictedVelocity[faceId] = INVALID_VELOCITY;
}

@compute @workgroup_size(64)
fn correctLosassoFaces(@builtin(global_invocation_id) invocation: vec3u) {
  let faceId = invocation.x;
  if (faceId >= liveFaces() || faceId >= arrayLength(&advectedVelocity)
      || faceId >= arrayLength(&predictedVelocity)
      || params.domainOriginDt.w == 0.0) { return; }
  let prediction = advectedVelocity[faceId];
  let reversed = predictedVelocity[faceId];
  if (!finite(prediction) || !finite(reversed) || reversed == INVALID_VELOCITY) { return; }
  let centre = faceCenterGrid(faceId);
  let original = velocityAtGrid(centre, BAND_FIELD);
  let departure = traceVelocityFromFirst(
    centre, params.domainOriginDt.w, BAND_FIELD, original);
  if (!original.valid || !departure.valid) { return; }
  let sourceStencil = velocityAtGrid(departure.point, BAND_FIELD);
  if (!sourceStencil.valid) { return; }
  // Selle-style fallback: keep the first-order S1a prediction anywhere the
  // forward or reverse characteristic touched a solid/domain clamp or the
  // one-cell interface seed layer. MacCormack remains enabled in the bulk.
  if (original.boundary || departure.boundary || sourceStencil.boundary
      || (faces[faceId].span & FACE_INTERFACE_NEARBY) != 0u) { return; }
  let axis = faces[faceId].axis;
  let corrected = prediction + 0.5 * (original.value[axis] - reversed);
  if (!finite(corrected)) { return; }
  // Bounded MacCormack: the correction may restore transported extrema but
  // cannot create values outside the eight reconstructed source nodes that
  // contributed to this face's forward prediction.
  advectedVelocity[faceId] = clamp(corrected,
    sourceStencil.lower[axis], sourceStencil.upper[axis]);
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
  // Aperture blends the resolved fluid face toward the prescribed moving-solid
  // normal velocity. The authored inflow is a boundary condition, however,
  // and must override the visual nozzle's closed solid cap just as it does in
  // the Power path; masking it by openFraction leaves a stationary hose mouth.
  let resolved = face.solidNormalVelocity
    + face.openFraction * (fluid - face.solidNormalVelocity);
  let value = select(resolved, inflow.x, inflow.y > 0.5);
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
  // Projection must not turn the prescribed source back into the nozzle's
  // solid velocity, including where the visual cap has zero open fraction.
  projectedVelocity[faceId] = select(0.0, inflow.x, finite(inflow.x));
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
