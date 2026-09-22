/**
 * Uniform MAC velocity extrapolation from CM12 Sec. 3.3, CM11b Secs. 3.3 and
 * 3.3.1, and Jeong/Ross/Whitaker 2007.
 *
 * The accurate JRW07/extension-PDE solve is limited to CM11b's two-cell narrow
 * band. Farther velocities are filled by CM11b Sec. 3.3.1's known-value,
 * renormalized trilinear restriction followed by reverse-order prolongation.
 * Each staggered component is interpolated on its own positive-face lattice.
 * The geometric method instead carries nearest original-source provenance to
 * keep separate fluid bodies from diluting each other's air extension.
 */
import { createCm12NumericsWGSL } from "../../core/cm12-numerics";

export const uniformVelocityExtrapolationShader = /* wgsl */ `
${createCm12NumericsWGSL(true)}
struct Params {
  dimsDt: vec4f,
  cellGravity: vec4f,
  container: vec4f,
  physical: vec4f,
  boundary: vec4f,
  inflowPositionRadius: vec4f,
  inflowVelocityLength: vec4f,
  inflowTiming: vec4f,
  tuning: vec4f,
  drop: vec4f,
  dropExtent: vec4f,
  // Shared with the parent solver's own Params. x: the shell reach in 4h tiles
  // (unused here; the class map already carries the dilated set). y: 1 when this
  // module's finest passes run on the shell tiles instead of densely.
  twoLevel: vec4f,
}
struct FrontParams {
  sourceParity: u32,
  targetParity: u32,
  hierarchySourceUsesBaseDims: u32,
  hierarchyTargetUsesBaseDims: u32,
  activeLevel: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
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
@group(0) @binding(11) var<storage, read> activeRegion: array<u32>;
// The parent solver's conditioning scratch. Words [N, N+4C) are the 4h table
// the two-level sampler reads: three face components and a class word whose
// bit 1 is SHELL. This module writes only the three face words, in one pass of
// its own, and reads only the class word -- never both in one dispatch.
@group(0) @binding(12) var<storage, read_write> tileScratch: array<u32>;

@group(0) @binding(13) var sourceOrigins: texture_3d<u32>;
@group(0) @binding(14) var existingOrigins: texture_3d<u32>;
@group(0) @binding(15) var outputOrigins: texture_storage_3d<rgba32uint, write>;
// Separate storage from the indirect buffer to avoid read/write usage aliasing.
@group(0) @binding(16) var<storage, read_write> shellTiles: array<atomic<u32>>;
override COMPACT_SHELL: bool = false;
override REUSE_CONVERGED_DISTANCE: bool = false;
@compute @workgroup_size(64)
fn buildShellList(@builtin(global_invocation_id) gid:vec3u) {
  let c=vec3u(coarseDims()); let i=gid.x;
  if(i>=c.x*c.y*c.z){return;}
  if((tileScratch[tileTableBase()+4u*i+3u]&2u)==0u){return;}
  let slot=atomicAdd(&shellTiles[0],1u);atomicStore(&shellTiles[4u+slot],i);
}
@compute @workgroup_size(1)
fn publishShellList() {
  // Two dispatch axes cover the full tile capacity without a 65535-group limit.
  let count=atomicLoad(&shellTiles[0]);let width=min(count,256u);
  atomicStore(&shellTiles[1],width);
  atomicStore(&shellTiles[2],(count+max(width,1u)-1u)/max(width,1u));
  atomicStore(&shellTiles[3],1u);
}
override SOURCE_AWARE_HIERARCHY: bool = false;
// Pipeline constants make provenance index decoding shifts/multiplies instead
// of per-candidate dynamic integer divisions, especially on power-of-two grids.
override ROOT_NX: u32 = 1u;
override ROOT_NY: u32 = 1u;
override ROOT_NZ: u32 = 1u;

const DISTANCE_INFINITY: f32 = 65504.0;
const ACCURATE_BAND_CELLS: f32 = 2.0;

fn baseDims() -> vec3i { return vec3i(textureDimensions(densityIn)); }
// The tile table is addressed in the PARENT's lattice, so it must not be sized
// from densityIn: the resolve pass rebinds that slot to an (n+2)^3 FIM scratch
// texture, and a table base computed from it lands outside the table entirely.
// faceOpenIn is the one binding every group points at the parent's own lattice.
fn tileDims() -> vec3i { return vec3i(textureDimensions(faceOpenIn)); }
fn coarseDims() -> vec3i { return (tileDims() + vec3i(3)) / 4; }
fn coarseIndex(t: vec3i) -> u32 { let c = coarseDims(); return u32(t.x + c.x * (t.y + c.y * t.z)); }
fn tileTableBase() -> u32 { let d = tileDims(); return u32(d.x * d.y * d.z); }
/**
 * Shrunk-extension gate. With it off every predicate below is true and the
 * module is bit-identical to the dense schedule.
 *
 * SHELL is the fine set dilated by one more tile than the sampler's, so it
 * covers (a) the finest trilinear tap, which reaches one cell below a fine
 * tile, (b) the FIM's own two-cell accurate band around any liquid cell, and
 * (c) the +-1 and +-2 neighbour reads of the Godunov update. Everything outside
 * it is stale, and every reader consults this predicate rather than being
 * cleared: an out-of-shell face reads as unknown and infinitely far, which is
 * exactly what a dense schedule writes wherever the band does not reach.
 */
fn tiledExtension() -> bool { return params.twoLevel.y > 0.5; }
fn shellAt(p: vec3i) -> bool {
  if (!tiledExtension()) { return true; }
  let cell = clamp(p, vec3i(0), tileDims() - vec3i(1));
  return (tileScratch[tileTableBase() + 4u * coarseIndex(cell / 4) + 3u] & 2u) != 0u;
}
// Group counts come from the host, sized off a box a couple of steps old, so
// the last workgroups overrun the exact window. Those threads exit here, at
// the window's exact extent for the finest passes and at a level's own packed
// extent for the hierarchy, rather than extending velocity into air nobody
// reads. A level whose extent did not fit the packed word reports all ones and
// clips nothing, which is the safe direction.
fn activeUnpackExtent(packed:u32)->vec3u{
  if((packed&0x40000000u)==0u){return vec3u(0xffffffffu);}
  return vec3u(packed&1023u,(packed>>10u)&1023u,(packed>>20u)&1023u);
}
fn activeBaseId(gid:vec3u)->vec3i{
  if(COMPACT_SHELL && tiledExtension()) {
    let slot=gid.x/4u+(gid.y/4u)*atomicLoad(&shellTiles[1]);
    if(slot>=atomicLoad(&shellTiles[0])){return vec3i(-1);}
    let tile=atomicLoad(&shellTiles[4u+slot]);let c=vec3u(coarseDims());
    return vec3i(vec3u(tile%c.x,(tile/c.x)%c.y,tile/(c.x*c.y))*4u+gid%vec3u(4));
  }
  let origin=vec3u(activeRegion[7],activeRegion[8],activeRegion[9]);
  if(any(gid>=vec3u(activeRegion[10],activeRegion[11],activeRegion[12])-origin)){return vec3i(-1);}
  return vec3i(gid)+vec3i(origin);
}
fn hierarchyActiveId(gid:vec3u)->vec3i{
  if(frontParams.hierarchyTargetUsesBaseDims!=0u){return activeBaseId(gid);}
  if(frontParams.activeLevel==0xffffffffu){return vec3i(gid);}
  let base=16u+10u*frontParams.activeLevel;
  if(any(gid>=activeUnpackExtent(activeRegion[base+9u]))){return vec3i(-1);}
  // Pressure hierarchy coordinates carry a one-cell domain halo; velocity
  // hierarchy textures do not. Its conservative 2/3-cell active halo remains
  // after translating the origin back by one.
  return vec3i(gid)+vec3i(vec3u(activeRegion[base],activeRegion[base+1u],activeRegion[base+2u]))-vec3i(1);
}
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
  return density(p) > CM12_LIQUID_ISOVALUE || density(p + componentAxis(component)) > CM12_LIQUID_ISOVALUE;
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

@compute @workgroup_size(4,4,4)
fn clearExtrapolationState(@builtin(global_invocation_id) gid:vec3u){
  let p=vec3i(gid);if(!inBounds(p,baseDims())){return;}
  textureStore(primaryOut,p,vec4f(0.0));
  textureStore(secondaryOut,p,vec4f(vec3f(DISTANCE_INFINITY),0.0));
}

@compute @workgroup_size(4, 4, 4)
fn seedActiveFront(@builtin(global_invocation_id) gid: vec3u) {
  let p = activeBaseId(gid); let d = baseDims();
  if (!inBounds(p, d) || !shellAt(p)) { return; }
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
  if (!openBaseFace(p, component) || !shellAt(p)) { return DISTANCE_INFINITY; }
  return faceDistance(textureLoad(secondaryIn, p, 0), component);
}
fn neighborValue(p: vec3i, component: u32) -> f32 {
  let d = baseDims();
  if (!openBaseFace(p, component) || !shellAt(p)) { return 0.0; }
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
  if (!openBaseFace(p, component) || sourceFace(p, component) || !shellAt(p)) { return false; }
  let distanceState = textureLoad(secondaryIn, p, 0);
  if (!isActive(distanceState, component)) { return false; }
  let oldDistance = faceDistance(distanceState, component);
  if (oldDistance >= 0.5 * DISTANCE_INFINITY) { return false; }
  let updatedDistance = min(oldDistance, godunovDistance(p, component));
  if (updatedDistance > accurateBandDistance()) { return false; }
  let epsilon = fimConvergenceEpsilon(oldDistance);
  return abs(updatedDistance - oldDistance) <= epsilon;
}
// Return convergence and its solved distance together; callers need both.
fn convergedDistance(p:vec3i,component:u32)->f32 {
  if(!openBaseFace(p,component)||sourceFace(p,component)||!shellAt(p)){return DISTANCE_INFINITY;}
  let state=textureLoad(secondaryIn,p,0);let old=state[component];
  if(!isActive(state,component)||old>=0.5*DISTANCE_INFINITY){return DISTANCE_INFINITY;}
  let updated=min(old,godunovDistance(p,component));
  if(updated>accurateBandDistance()||abs(updated-old)>fimConvergenceEpsilon(old)){return DISTANCE_INFINITY;}
  return updated;
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
    if(REUSE_CONVERGED_DISTANCE) {
      if(ownDistance>convergedDistance(low,component)+epsilon
        ||ownDistance>convergedDistance(high,component)+epsilon){return true;}
      continue;
    }
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
  let p = activeBaseId(gid); let d = baseDims();
  if (!inBounds(p, d) || !shellAt(p)) { return; }
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
  if(COMPACT_SHELL && tiledExtension()) {
    atomicStore(&dispatchArgs.x,select(0u,atomicLoad(&shellTiles[1]),activeCount>0u));
    atomicStore(&dispatchArgs.y,select(0u,atomicLoad(&shellTiles[2]),activeCount>0u));
    atomicStore(&dispatchArgs.z,select(0u,1u,activeCount>0u));return;
  }
  atomicStore(&dispatchArgs.x, select(0u, activeRegion[13], activeCount > 0u));
  atomicStore(&dispatchArgs.y, select(0u, activeRegion[14], activeCount > 0u));
  atomicStore(&dispatchArgs.z, select(0u, activeRegion[15], activeCount > 0u));
}

// Indirect termination leaves the last written ping-pong side dynamic. Resolve
// the completed two-cell band into canonical textures before hierarchy transfer.
@compute @workgroup_size(4, 4, 4)
fn resolveConvergedFront(@builtin(global_invocation_id) gid: vec3u) {
  let p = activeBaseId(gid); let d = baseDims();
  if (!inBounds(p, d) || !shellAt(p)) { return; }
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

/**
 * The one stale-state leak the shrunk extension can create, closed at the
 * reader. Only the finest restrict has a base-dimension source, and that source
 * is resolvedValues, which the shrunk resolve writes inside SHELL only. A
 * tile that has just left SHELL still carries last step's real band values with
 * their known bits set; unchecked, the restrict would carry them up and
 * corrupt the very ceil(n/4) level the sampler now depends on. Treating an
 * out-of-shell fine face as unknown is exactly what the dense schedule writes
 * there -- resolveConvergedFront copies a seed side whose far-air entry is
 * (0,0,0, known=0) -- so no clearing pass and no hysteresis is needed.
 */
fn sourceKnownAt(q: vec3i) -> bool {
  return frontParams.hierarchySourceUsesBaseDims == 0u || shellAt(q);
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
        if (weight > 0.0 && componentKnown(state, component) && sourceKnownAt(q)) {
          contributions[corner]=vec2f(weight*state[component],weight);
        }
      }
    }
  }
  let contribution=d4Sum8Vec2(contributions);let weightedValue=contribution.x;let weightSum=contribution.y;
  return vec2f(select(0.0, weightedValue / weightSum, weightSum > 0.0), weightSum);
}

fn hierarchyKnownContribution(p: vec3i, sourceDims: vec3i, component: u32) -> vec2f {
  if (!inBounds(p, sourceDims) || !sourceKnownAt(p)) { return vec2f(0.0); }
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

// Carry original fine-face provenance rather than promoting filled coarse air
// to a new source. Otherwise a distant stationary pool repeatedly averages
// into the extension under a falling drop. Fine FIM values remain untouched.
struct NearestSample { value: f32, weight: f32, origin: u32 }
fn faceLocation(p: vec3i, dims: vec3i, component: u32) -> vec3f {
  var point = (vec3f(p) + vec3f(0.5)) * vec3f(baseDims()) / vec3f(dims);
  point[component] = f32(p[component]+1) * f32(baseDims()[component]) / f32(dims[component]);
  return point;
}
fn originalFace(origin: u32) -> vec3i {
  let d = vec3u(ROOT_NX,ROOT_NY,ROOT_NZ); let index = origin - 1u;
  return vec3i(vec3u(index % d.x, (index / d.x) % d.y, index / (d.x*d.y)));
}
fn nearestHierarchySample(p: vec3i, sd: vec3i, td: vec3i, component: u32, footprint: bool) -> NearestSample {
  let location = faceLocation(p, td, component);
  var sourcePosition = (vec3f(p)+vec3f(0.5))*vec3f(sd)/vec3f(td)-vec3f(0.5);
  sourcePosition[component] = f32(p[component]+1)*f32(sd[component])/f32(td[component])-1.0;
  let lower = select(vec3i(floor(sourcePosition)), 2*p, footprint);
  let h = params.cellGravity.xyz;
  let epsilon = 1e-6 * min(h.x,min(h.y,h.z)) * min(h.x,min(h.y,h.z));
  var best = 1e30;
  var distances: array<f32,8>;
  var origins: array<u32,8>;
  var values: array<f32,8>;
  for (var k=0u; k<8u; k++) {
    let q = clamp(lower + vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)),vec3i(0),sd-vec3i(1));
    let state = textureLoad(primaryIn,q,0);
    distances[k] = 1e30;
    if (!componentKnown(state,component) || !sourceKnownAt(q)) { continue; }
    var origin: u32;
    if (frontParams.hierarchySourceUsesBaseDims != 0u) {
      let d = baseDims(); origin = u32(q.x+d.x*(q.y+d.y*q.z))+1u;
    } else { origin = textureLoad(sourceOrigins,q,0)[component]; }
    if (origin == 0u) { continue; }
    let delta = (faceLocation(originalFace(origin),baseDims(),component)-location)*h;
    let distance = dot(delta,delta);
    distances[k] = distance; origins[k] = origin; values[k] = state[component];
    best = min(best,distance);
  }
  var contributions: array<vec2f,8>; var origin = 0u;
  for (var k=0u; k<8u; k++) {
    contributions[k] = vec2f(0.0);
    if (origins[k] != 0u && abs(distances[k]-best) <= epsilon) {
      contributions[k] = vec2f(values[k],1.0);
      if (origin == 0u) { origin = origins[k]; }
    }
  }
  let sum = d4Sum8Vec2(contributions);
  return NearestSample(select(0.0,sum.x/sum.y,sum.y>0.0),sum.y,origin);
}

@compute @workgroup_size(4, 4, 4)
fn restrictKnownVelocity(@builtin(global_invocation_id) gid: vec3u) {
  let p = hierarchyActiveId(gid);
  let sourceDims = hierarchySourceDims();
  let targetDims = hierarchyTargetDims();
  if (!inBounds(p, targetDims)) { return; }
  var values = vec3f(0.0);
  var knownMask = 0u;
  var origins = vec3u(0u);
  for (var component = 0u; component < 3u; component += 1u) {
    if (SOURCE_AWARE_HIERARCHY) {
      var result = nearestHierarchySample(p,sourceDims,targetDims,component,false);
      if (result.weight <= 0.0) { result = nearestHierarchySample(p,sourceDims,targetDims,component,true); }
      if (result.weight > 0.0) {
        values[component] = result.value; origins[component] = result.origin;
        knownMask |= bitFor(component);
      }
      continue;
    }
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
  if (SOURCE_AWARE_HIERARCHY) { textureStore(outputOrigins,p,vec4u(origins,0u)); }
  textureStore(primaryOut, p, vec4f(values, f32(knownMask)));
}

fn prolongValue(p: vec3i) -> vec4f {
  let existing = textureLoad(secondaryIn, p, 0);
  var values = existing.xyz;
  var knownMask = u32(round(existing.w));
  var origins = vec3u(0u);
  if (SOURCE_AWARE_HIERARCHY && frontParams.hierarchyTargetUsesBaseDims == 0u) {
    origins = textureLoad(existingOrigins,p,0).xyz;
  }
  for (var component = 0u; component < 3u; component += 1u) {
    if (componentKnown(existing, component)) { continue; }
    if (SOURCE_AWARE_HIERARCHY) {
      let result = nearestHierarchySample(p,hierarchySourceDims(),hierarchyTargetDims(),component,false);
      if (result.weight > 0.0) {
        values[component] = result.value; origins[component] = result.origin;
        knownMask |= bitFor(component);
      }
    } else {
      let result = hierarchyComponentSample(p, hierarchySourceDims(), hierarchyTargetDims(), component);
      if (result.y > 0.0) { values[component] = result.x; knownMask |= bitFor(component); }
    }
  }
  if (SOURCE_AWARE_HIERARCHY && frontParams.hierarchyTargetUsesBaseDims == 0u) {
    textureStore(outputOrigins,p,vec4u(origins,0u));
  }
  return vec4f(values,f32(knownMask));
}
@compute @workgroup_size(4,4,4)
fn prolongUnknownVelocity(@builtin(global_invocation_id) gid: vec3u) {
  let p = hierarchyActiveId(gid);
  if (!inBounds(p,hierarchyTargetDims())) { return; }
  if (frontParams.hierarchyTargetUsesBaseDims != 0u && !shellAt(p)) { return; }
  textureStore(primaryOut,p,prolongValue(p));
}
fn packValue(p: vec3i, state: vec4f) {
  var values = vec3f(0.0); var knownMask = 0u; var openMask = 0u;
  for (var component = 0u; component < 3u; component += 1u) {
    if (openBaseFace(p, component)) {
      openMask |= bitFor(component);
      if (componentKnown(state, component)) {
        values[component] = state[component]; knownMask |= bitFor(component);
      }
    }
  }
  textureStore(primaryOut,p+vec3i(1),vec4f(values,f32(knownMask | (openMask << 3u))));
}
@compute @workgroup_size(4,4,4)
fn prolongAndPack(@builtin(global_invocation_id) gid: vec3u) {
  let p = hierarchyActiveId(gid);
  if (!inBounds(p,baseDims()) || !shellAt(p)) { return; }
  packValue(p,prolongValue(p));
}

@compute @workgroup_size(4, 4, 4)
fn packTransportShell(@builtin(global_invocation_id) gid: vec3u) {
  let p=activeBaseId(gid);let d=baseDims();
  if(!inBounds(p,d)||!shellAt(p)){return;}
  let state = textureLoad(primaryIn, p, 0);
  packValue(p,state);
}

/**
 * Publish the ceil(n/4) hierarchy level as the parent solver's 4h face table.
 *
 * primaryIn is that level's prolong-filled up texture, or its down when
 * it is the coarsest level and nothing prolongs into it. The transfer's
 * component-axis map is (t+1)*S/T - 1, which at S/T = 4 is fine face 4t+3, the
 * upper face of coarse cell t -- the same convention the parent's coarse
 * sampler reads, and the reason the host requires every axis to be a multiple
 * of four before it offers this path.
 *
 * Two conventions are applied on top of the level's own values. An unknown
 * component publishes zero, which is what the fine pack writes for an unknown
 * face. And the last coarse layer on the component axis maps onto the outer
 * domain wall, where the fine pack writes zero unless the wall is open (an
 * authored atmospheric +Y face); the hierarchy would otherwise publish the
 * prolonged interior value there. Interior closed faces need no such test:
 * every tile holding a partially open cell seeds, so solids are always inside
 * the fine set and their faces are never read from this table.
 */
@compute @workgroup_size(4,4,4)
fn publishCoarseVelocityTable(@builtin(global_invocation_id) gid: vec3u) {
  let t = vec3i(gid); let c = coarseDims();
  if (!inBounds(t, c) || any(vec3i(textureDimensions(primaryIn)) != c)) { return; }
  let state = textureLoad(primaryIn, t, 0);
  let slot = tileTableBase() + 4u * coarseIndex(t);
  for (var component = 0u; component < 3u; component += 1u) {
    var value = select(0.0, state[component], componentKnown(state, component));
    if (t[component] + 1 == c[component]) {
      var open = false;
      for (var a = 0; a < 4; a += 1) { for (var b = 0; b < 4; b += 1) {
        var p = 4 * t; p[(component + 1u) % 3u] += a; p[(component + 2u) % 3u] += b; p[component] += 3;
        if (openBaseFace(p, component)) { open = true; }
      } }
      if (!open) { value = 0.0; }
    }
    tileScratch[slot + component] = bitcast<u32>(value);
  }
}
`;
