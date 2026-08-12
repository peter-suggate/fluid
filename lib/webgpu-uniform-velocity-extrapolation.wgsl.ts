/**
 * Uniform MAC velocity extrapolation from CM12 Sec. 3.3, CM11b Secs. 3.3 and
 * 3.3.1, and Jeong/Ross/Whitaker 2007.
 *
 * The accurate JRW07/extension-PDE solve is limited to CM11b's two-cell narrow
 * band. Farther velocities are filled by CM11b Sec. 3.3.1's known-value,
 * renormalized trilinear restriction followed by reverse-order prolongation.
 * Each staggered component is interpolated on its own positive-face lattice.
 */
export const uniformVelocityExtrapolationShader = /* wgsl */ `
struct Params {
  dimsDt: vec4f,
  cellGravity: vec4f,
  container: vec4f,
  physical: vec4f,
  boundary: vec4f,
  inflowPositionRadius: vec4f,
  inflowVelocityLength: vec4f,
  inflowTiming: vec4f,
}
struct FrontParams {
  sourceParity: u32,
  targetParity: u32,
  hierarchySourceUsesBaseDims: u32,
  hierarchyTargetUsesBaseDims: u32,
}
struct ConvergenceState {
  activeA: atomic<u32>,
  activeB: atomic<u32>,
  latestParity: atomic<u32>,
  executedPasses: atomic<u32>,
}
struct DispatchArgs {
  x: atomic<u32>,
  y: atomic<u32>,
  z: atomic<u32>,
}
@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var densityIn: texture_3d<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var primaryIn: texture_3d<f32>;
@group(0) @binding(4) var secondaryIn: texture_3d<f32>;
@group(0) @binding(5) var primaryOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var secondaryOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var<uniform> frontParams: FrontParams;
// xyz contains the exact uniform solver faceOpenFraction for positive MAC
// faces, prepared beside rho'=rho/V by the parent authority pass.
@group(0) @binding(8) var faceOpenIn: texture_3d<f32>;
@group(0) @binding(9) var<storage, read_write> convergence: ConvergenceState;
@group(0) @binding(10) var<storage, read_write> dispatchArgs: DispatchArgs;

const DISTANCE_INFINITY: f32 = 65504.0;
const LIQUID_ISOVALUE: f32 = 0.5;
const ACCURATE_BAND_CELLS: f32 = 2.0;

fn baseDims() -> vec3i { return vec3i(textureDimensions(densityIn)); }
fn cellSize() -> vec3f { return params.cellGravity.xyz; }
fn accurateBandDistance() -> f32 {
  let h = cellSize();
  return ACCURATE_BAND_CELLS * max(h.x, max(h.y, h.z));
}
fn inBounds(p: vec3i, d: vec3i) -> bool {
  return all(p >= vec3i(0)) && all(p < d);
}
fn d4Sum3(value:array<f32,3>)->f32{return (value[0]+value[2])+value[1];}
fn d4Sum8Vec2(value:array<vec2f,8>)->vec2f{
  let y0=(value[0]+value[5])+(value[1]+value[4]);
  let y1=(value[2]+value[7])+(value[3]+value[6]);
  return y0+y1;
}
fn componentAxis(component: u32) -> vec3i {
  var axis = vec3i(0);
  axis[component] = 1;
  return axis;
}
fn validFace(p: vec3i, component: u32, d: vec3i) -> bool {
  // The dense texture stores the positive face of every cell. The final
  // component index is normally a closed outer wall, but +Y can be the
  // authored atmospheric face; faceOpenIn decides that at the finest level.
  return inBounds(p, d);
}
fn openBaseFace(p: vec3i, component: u32) -> bool {
  return validFace(p, component, baseDims()) && textureLoad(faceOpenIn, p, 0)[component] > 1e-5;
}
fn bitFor(component: u32) -> u32 { return 1u << component; }
fn componentKnown(state: vec4f, component: u32) -> bool {
  return (u32(round(state.w)) & bitFor(component)) != 0u;
}
fn withKnown(mask: u32, component: u32, known: bool) -> u32 {
  return select(mask & ~bitFor(component), mask | bitFor(component), known);
}
fn density(p: vec3i) -> f32 {
  let d = baseDims();
  return select(0.0, textureLoad(densityIn, clamp(p, vec3i(0), d - vec3i(1)), 0).x,
    inBounds(p, d));
}
fn sourceFace(p: vec3i, component: u32) -> bool {
  let d = baseDims();
  if (!openBaseFace(p, component)) { return false; }
  return density(p) > LIQUID_ISOVALUE || density(p + componentAxis(component)) > LIQUID_ISOVALUE;
}
fn faceValue(state: vec4f, component: u32) -> f32 { return state[component]; }
fn faceDistance(state: vec4f, component: u32) -> f32 { return state[component]; }
// FIM's convergence threshold is not numerically fixed by JRW07. Use a local
// relative tolerance clamped to the physical domain scale; unlike the former
// half-precision storage this does not quantize velocity before MacCormack.
fn fimConvergenceEpsilon(referenceDistance: f32) -> f32 {
  let h = cellSize();
  let minimumScale = 2.0 * min(h.x, min(h.y, h.z));
  let domainDiagonal = length(vec3f(baseDims()) * h);
  return clamp(abs(referenceDistance), minimumScale, domainDiagonal) * 1.1920929e-7;
}

fn oneNeighborTouchesSource(p: vec3i, component: u32) -> bool {
  let d = baseDims();
  for (var axis = 0u; axis < 3u; axis += 1u) {
    var step = vec3i(0); step[axis] = 1;
    if (validFace(p - step, component, d) && sourceFace(p - step, component)) { return true; }
    if (validFace(p + step, component, d) && sourceFace(p + step, component)) { return true; }
  }
  return false;
}

@compute @workgroup_size(4, 4, 4)
fn seedActiveFront(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec3i(gid); let d = baseDims();
  if (!inBounds(p, d)) { return; }
  let inputVelocity = textureLoad(velocityIn, p, 0).xyz;
  var values = vec3f(0.0);
  var distances = vec3f(DISTANCE_INFINITY);
  var knownMask = 0u;
  var activeMask = 0u;
  for (var component = 0u; component < 3u; component += 1u) {
    if (!openBaseFace(p, component)) { continue; }
    if (sourceFace(p, component)) {
      values[component] = inputVelocity[component];
      distances[component] = 0.0;
      knownMask |= bitFor(component);
    } else if (oneNeighborTouchesSource(p, component)) {
      activeMask |= bitFor(component);
    }
  }
  textureStore(primaryOut, p, vec4f(values, f32(knownMask)));
  textureStore(secondaryOut, p, vec4f(distances, f32(activeMask)));
  if (activeMask != 0u) { atomicAdd(&convergence.activeA, countOneBits(activeMask & 7u)); }
}

fn neighborDistance(p: vec3i, component: u32) -> f32 {
  let d = baseDims();
  if (!openBaseFace(p, component)) { return DISTANCE_INFINITY; }
  return faceDistance(textureLoad(secondaryIn, p, 0), component);
}
fn neighborValue(p: vec3i, component: u32) -> f32 {
  let d = baseDims();
  if (!openBaseFace(p, component)) { return 0.0; }
  let state = textureLoad(primaryIn, p, 0);
  return select(0.0, faceValue(state, component), componentKnown(state, component));
}

// JRW07 Eq. 2.1 with f=1, solved from the smallest neighbor on each axis.
// The spacing travels with each sorted neighbor, so rectangular cells retain
// the stated anisotropic Godunov equation rather than assuming cubic cells.
fn godunovDistance(p: vec3i, component: u32) -> f32 {
  var minima = array<f32, 3>();
  var spacing = array<f32, 3>();
  let h = cellSize();
  for (var axis = 0u; axis < 3u; axis += 1u) {
    var step = vec3i(0); step[axis] = 1;
    minima[axis] = min(neighborDistance(p - step, component), neighborDistance(p + step, component));
    spacing[axis] = h[axis];
  }
  for (var i = 0u; i < 2u; i += 1u) {
    for (var j = i + 1u; j < 3u; j += 1u) {
      if (minima[j] < minima[i]) {
        let a = minima[i]; minima[i] = minima[j]; minima[j] = a;
        let s = spacing[i]; spacing[i] = spacing[j]; spacing[j] = s;
      }
    }
  }
  if (minima[0] >= 0.5 * DISTANCE_INFINITY) { return DISTANCE_INFINITY; }
  var root = minima[0] + spacing[0];
  for (var count = 2u; count <= 3u; count += 1u) {
    if (minima[count - 1u] >= 0.5 * DISTANCE_INFINITY || root <= minima[count - 1u]) { break; }
    var a = 0.0; var b = 0.0; var c = -1.0;
    for (var index = 0u; index < count; index += 1u) {
      let inverseH2 = 1.0 / (spacing[index] * spacing[index]);
      a += inverseH2;
      b += minima[index] * inverseH2;
      c += minima[index] * minima[index] * inverseH2;
    }
    root = (b + sqrt(max(b * b - a * c, 0.0))) / a;
  }
  return root;
}

// First-order upwind discretization of CM11b Eq. 7 at one MAC component.
// For a solved Eikonal distance U, n·grad(u)=0 gives weights
// (U-U_upwind)/h^2. Equal Godunov minimizers are averaged, avoiding a
// direction-dependent tie break on symmetric interfaces.
fn upwindExtensionValue(p: vec3i, component: u32, solvedDistance: f32) -> f32 {
  let h = cellSize();
  var weightedTerms:array<f32,3>;var weightTerms:array<f32,3>;
  let epsilon = fimConvergenceEpsilon(solvedDistance);
  for (var axis = 0u; axis < 3u; axis += 1u) {
    weightedTerms[axis]=0.0;weightTerms[axis]=0.0;
    var step = vec3i(0); step[axis] = 1;
    let lowDistance = neighborDistance(p - step, component);
    let highDistance = neighborDistance(p + step, component);
    let minimumDistance = min(lowDistance, highDistance);
    if (minimumDistance >= solvedDistance - epsilon) { continue; }
    var minimizerValue = 0.0; var minimizerCount = 0.0;
    if (abs(lowDistance - minimumDistance) <= epsilon) {
      minimizerValue += neighborValue(p - step, component); minimizerCount += 1.0;
    }
    if (abs(highDistance - minimumDistance) <= epsilon) {
      minimizerValue += neighborValue(p + step, component); minimizerCount += 1.0;
    }
    if (minimizerCount <= 0.0) { continue; }
    let weight = (solvedDistance - minimumDistance) / (h[axis] * h[axis]);
    weightedTerms[axis]=weight*minimizerValue/minimizerCount;
    weightTerms[axis]=weight;
  }
  let weightedValue=d4Sum3(weightedTerms);let weightSum=d4Sum3(weightTerms);
  return select(0.0, weightedValue / weightSum, weightSum > 0.0);
}

fn isActive(state: vec4f, component: u32) -> bool {
  return (u32(round(state.w)) & bitFor(component)) != 0u;
}
fn activeNodeConverges(p: vec3i, component: u32) -> bool {
  let d = baseDims();
  if (!openBaseFace(p, component) || sourceFace(p, component)) { return false; }
  let distanceState = textureLoad(secondaryIn, p, 0);
  if (!isActive(distanceState, component)) { return false; }
  let oldDistance = faceDistance(distanceState, component);
  if (oldDistance >= 0.5 * DISTANCE_INFINITY) { return false; }
  let updatedDistance = min(oldDistance, godunovDistance(p, component));
  if (updatedDistance > accurateBandDistance()) { return false; }
  let epsilon = fimConvergenceEpsilon(oldDistance);
  return abs(updatedDistance - oldDistance) <= epsilon;
}
fn activatedByConvergedUpwindNeighbor(p: vec3i, component: u32) -> bool {
  let d = baseDims();
  let candidate = godunovDistance(p, component);
  if (!openBaseFace(p, component) || candidate > accurateBandDistance()) { return false; }
  let ownDistance = faceDistance(textureLoad(secondaryIn, p, 0), component);
  let epsilon = fimConvergenceEpsilon(ownDistance);
  for (var axis = 0u; axis < 3u; axis += 1u) {
    var step = vec3i(0); step[axis] = 1;
    let low = p - step; let high = p + step;
    if (activeNodeConverges(low, component)) {
      let updatedNeighborDistance = min(neighborDistance(low, component), godunovDistance(low, component));
      if (ownDistance > updatedNeighborDistance + epsilon) { return true; }
    }
    if (activeNodeConverges(high, component)) {
      let updatedNeighborDistance = min(neighborDistance(high, component), godunovDistance(high, component));
      if (ownDistance > updatedNeighborDistance + epsilon) { return true; }
    }
  }
  return false;
}

@compute @workgroup_size(4, 4, 4)
fn updateActiveFront(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec3i(gid); let d = baseDims();
  if (!inBounds(p, d)) { return; }
  let oldValues = textureLoad(primaryIn, p, 0);
  let oldDistances = textureLoad(secondaryIn, p, 0);
  var values = oldValues.xyz;
  var distances = oldDistances.xyz;
  var knownMask = u32(round(oldValues.w));
  var activeMask = u32(round(oldDistances.w));
  for (var component = 0u; component < 3u; component += 1u) {
    if (!openBaseFace(p, component) || sourceFace(p, component)) { continue; }
    if (isActive(oldDistances, component)) {
      let oldDistance = distances[component];
      let epsilon = fimConvergenceEpsilon(oldDistance);
      let candidate = godunovDistance(p, component);
      let updatedDistance = min(oldDistance, candidate);
      if (updatedDistance <= accurateBandDistance()) {
        distances[component] = updatedDistance;
        values[component] = upwindExtensionValue(p, component, updatedDistance);
        knownMask = withKnown(knownMask, component, true);
      }
      let converged = oldDistance < 0.5 * DISTANCE_INFINITY
        && abs(updatedDistance - oldDistance) <= epsilon;
      activeMask = withKnown(activeMask, component, !converged && updatedDistance <= accurateBandDistance());
    } else if (activatedByConvergedUpwindNeighbor(p, component)) {
      // JRW07: a newly added point is not updated until the next iteration.
      activeMask = withKnown(activeMask, component, true);
    }
  }
  textureStore(primaryOut, p, vec4f(values, f32(knownMask)));
  textureStore(secondaryOut, p, vec4f(distances, f32(activeMask)));
  if (activeMask != 0u) {
    if (frontParams.targetParity == 0u) {
      atomicAdd(&convergence.activeA, countOneBits(activeMask & 7u));
    } else {
      atomicAdd(&convergence.activeB, countOneBits(activeMask & 7u));
    }
  }
}

fn loadActiveCounter(index: u32) -> u32 {
  return select(atomicLoad(&convergence.activeA), atomicLoad(&convergence.activeB), index == 1u);
}
fn clearActiveCounter(index: u32) {
  if (index == 0u) { atomicStore(&convergence.activeA, 0u); }
  else { atomicStore(&convergence.activeB, 0u); }
}

// FIM's source algorithm terminates when the active list is empty. The host
// encodes a domain-bounded chain of indirect updates; this one-thread stage
// turns every remaining dispatch into a zero-work dispatch immediately after
// that source-stated condition is reached.
@compute @workgroup_size(1)
fn prepareActiveDispatch(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x != 0u) { return; }
  let source = frontParams.sourceParity;
  let targetIndex = frontParams.targetParity;
  let sourceCount = loadActiveCounter(source);
  if (source == targetIndex) {
    atomicStore(&convergence.latestParity, source);
  } else {
    if (sourceCount > 0u) {
      atomicStore(&convergence.latestParity, targetIndex);
      atomicAdd(&convergence.executedPasses, 1u);
    }
    clearActiveCounter(source);
  }
  let activeCount = loadActiveCounter(targetIndex);
  let d = baseDims();
  atomicStore(&dispatchArgs.x, select(0u, u32((d.x + 3) / 4), activeCount > 0u));
  atomicStore(&dispatchArgs.y, select(0u, u32((d.y + 3) / 4), activeCount > 0u));
  atomicStore(&dispatchArgs.z, select(0u, u32((d.z + 3) / 4), activeCount > 0u));
}

// Indirect termination leaves the last written ping-pong side dynamic. Resolve
// the completed two-cell band into canonical textures before hierarchy transfer.
@compute @workgroup_size(4, 4, 4)
fn resolveConvergedFront(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec3i(gid); let d = baseDims();
  if (!inBounds(p, d)) { return; }
  let parity = atomicLoad(&convergence.latestParity);
  let values = select(textureLoad(primaryIn, p, 0), textureLoad(velocityIn, p, 0), parity == 1u);
  let distances = select(textureLoad(secondaryIn, p, 0), textureLoad(densityIn, p, 0), parity == 1u);
  textureStore(primaryOut, p, values);
  textureStore(secondaryOut, p, distances);
}

fn hierarchySourceDims() -> vec3i {
  return select(vec3i(textureDimensions(primaryIn)), baseDims(),
    frontParams.hierarchySourceUsesBaseDims != 0u);
}
fn hierarchyTargetDims() -> vec3i {
  return select(vec3i(textureDimensions(primaryOut)), baseDims(),
    frontParams.hierarchyTargetUsesBaseDims != 0u);
}

// CM11b Sec. 3.3.1 evaluates each hierarchy transfer by trilinear
// interpolation using known velocities only and renormalizes the weights.
// The component-axis coordinate is face centered; the other two coordinates
// are cell centered, preserving the uniform solver's staggered MAC geometry.
fn hierarchyComponentSample(
  targetPosition: vec3i, sourceDims: vec3i, targetDims: vec3i, component: u32,
) -> vec2f {
  var sourcePosition = (vec3f(targetPosition) + vec3f(0.5))
    * vec3f(sourceDims) / vec3f(targetDims) - vec3f(0.5);
  sourcePosition[component] = f32(targetPosition[component] + 1)
    * f32(sourceDims[component]) / f32(targetDims[component]) - 1.0;
  let lower = vec3i(floor(sourcePosition));
  let fraction = fract(sourcePosition);
  var contributions:array<vec2f,8>;
  for (var oz = 0; oz <= 1; oz += 1) {
    for (var oy = 0; oy <= 1; oy += 1) {
      for (var ox = 0; ox <= 1; ox += 1) {
        let offset = vec3i(ox, oy, oz);
        let q = clamp(lower + offset, vec3i(0), sourceDims - vec3i(1));
        let selector = vec3f(offset);
        let weights = select(vec3f(1.0) - fraction, fraction, selector > vec3f(0.5));
        let weight = weights.x * weights.y * weights.z;
        let state = textureLoad(primaryIn, q, 0);
        let corner=u32(ox+2*oy+4*oz);contributions[corner]=vec2f(0.0);
        if (weight > 0.0 && componentKnown(state, component)) {
          contributions[corner]=vec2f(weight*state[component],weight);
        }
      }
    }
  }
  let contribution=d4Sum8Vec2(contributions);let weightedValue=contribution.x;let weightSum=contribution.y;
  return vec2f(select(0.0, weightedValue / weightSum, weightSum > 0.0), weightSum);
}

fn hierarchyKnownContribution(p: vec3i, sourceDims: vec3i, component: u32) -> vec2f {
  if (!inBounds(p, sourceDims)) { return vec2f(0.0); }
  let state = textureLoad(primaryIn, p, 0);
  return select(vec2f(0.0), vec2f(state[component], 1.0), componentKnown(state, component));
}

// CM11b Sec. 3.3.1 also defines a coarse velocity as known when at least one
// of its corresponding finer-cell velocities is known. On a regular MAC grid,
// collapsing the component axis can put the face-centred point stencil wholly
// on an unknown outer face even though the coarse cell's dual 2x2x2 footprint
// contains known values. This restriction-only footprint is the literal
// regular-grid counterpart of the paper's tall-cell declaration. It is used
// only when the geometric face stencil above has no support; prolongation
// remains the stated staggered trilinear interpolation.
fn hierarchyCorrespondingCellSample(
  targetPosition: vec3i, sourceDims: vec3i, component: u32,
) -> vec2f {
  let lower = 2 * targetPosition;
  var contribution = vec2f(0.0);
  for (var oy = 0; oy <= 1; oy += 1) {
    // Group the X/Z-transposed pair before accumulating each plane so the
    // fallback does not introduce an arbitrary X-before-Z reduction order.
    let c00 = hierarchyKnownContribution(lower + vec3i(0, oy, 0), sourceDims, component);
    let c11 = hierarchyKnownContribution(lower + vec3i(1, oy, 1), sourceDims, component);
    let c10 = hierarchyKnownContribution(lower + vec3i(1, oy, 0), sourceDims, component);
    let c01 = hierarchyKnownContribution(lower + vec3i(0, oy, 1), sourceDims, component);
    contribution += (c00 + c11) + (c10 + c01);
  }
  return vec2f(
    select(0.0, contribution.x / contribution.y, contribution.y > 0.0),
    contribution.y,
  );
}

@compute @workgroup_size(4, 4, 4)
fn restrictKnownVelocity(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec3i(gid);
  let sourceDims = hierarchySourceDims();
  let targetDims = hierarchyTargetDims();
  if (!inBounds(p, targetDims)) { return; }
  var values = vec3f(0.0);
  var knownMask = 0u;
  for (var component = 0u; component < 3u; component += 1u) {
    var result = hierarchyComponentSample(p, sourceDims, targetDims, component);
    // The corresponding-cell fallback is cell-centred. It is valid for the
    // vertical component used by the paper's tall-cell construction, but its
    // 2x2x2 footprint is not centred on horizontal MAC faces: applying it to
    // x/z chooses a different normal-axis stencil after reflection. Preserve
    // the staggered trilinear interpolation for those components.
    if (result.y <= 0.0 && component == 1u) {
      result = hierarchyCorrespondingCellSample(p, sourceDims, component);
    }
    if (result.y > 0.0) {
      values[component] = result.x;
      knownMask |= bitFor(component);
    }
  }
  textureStore(primaryOut, p, vec4f(values, f32(knownMask)));
}

@compute @workgroup_size(4, 4, 4)
fn prolongUnknownVelocity(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec3i(gid);
  let sourceDims = hierarchySourceDims();
  let targetDims = hierarchyTargetDims();
  if (!inBounds(p, targetDims)) { return; }
  let existing = textureLoad(secondaryIn, p, 0);
  var values = existing.xyz;
  var knownMask = u32(round(existing.w));
  for (var component = 0u; component < 3u; component += 1u) {
    if (componentKnown(existing, component)) { continue; }
    let result = hierarchyComponentSample(p, sourceDims, targetDims, component);
    if (result.y > 0.0) {
      values[component] = result.x;
      knownMask |= bitFor(component);
    }
  }
  textureStore(primaryOut, p, vec4f(values, f32(knownMask)));
}

@compute @workgroup_size(4, 4, 4)
fn packTransportShell(@builtin(global_invocation_id) gid: vec3u) {
  let padded = vec3i(gid); let d = baseDims();
  if (any(padded >= d + vec3i(2))) { return; }
  let p = padded - vec3i(1);
  if (!inBounds(p, d)) { textureStore(primaryOut, padded, vec4f(0.0)); return; }
  let state = textureLoad(primaryIn, p, 0);
  var values = vec3f(0.0); var knownMask = 0u; var openMask = 0u;
  for (var component = 0u; component < 3u; component += 1u) {
    if (openBaseFace(p, component)) {
      openMask |= bitFor(component);
      if (componentKnown(state, component)) {
        values[component] = state[component]; knownMask |= bitFor(component);
      }
    }
  }
  // w retains known bits 0..2 and authoritative open-face bits 3..5 for the
  // opt-in Dawn conformance readback. All transport consumers sample xyz only.
  textureStore(primaryOut, padded, vec4f(values, f32(knownMask | (openMask << 3u))));
}
`;
