/**
 * Narita et al. Sec. 4.1 on WebGPU.
 *
 * The expensive update path is deliberately dense and pointer-free on the
 * GPU: one thread first scans each vertical column into a 2D sizing field,
 * then a ping-pong map stores the owning dyadic leaf for every finest x/z
 * cell. The resident level set is advanced independently every simulation
 * step; construction only evaluates sizing and rebuilds the horizontal tree.
 */

import { inflowBoundaryWGSL } from "./inflow-boundary";
import { gpuCompilationManagerFor } from "./gpu-compilation-manager";
import { FLUID_BRICK_ACTIVE_SURFACE_DISPATCH_OFFSET_BYTES } from "./webgpu-fluid-brick-residency";

function largestPowerOfTwoAtMost(value: number) {
  let result = 1;
  while (result * 2 <= value) result *= 2;
  return result;
}

/** Jump-flood schedule for the resident surface's five-cell distance band. */
export function quadtreeSurfaceJumpSequence(nx: number, ny: number, nz: number) {
  const domainStart = largestPowerOfTwoAtMost(Math.max(1, nx, ny, nz));
  if (domainStart <= 4) {
    const full: number[] = [];
    for (let jump = domainStart; jump >= 1; jump /= 2) full.push(jump);
    return full;
  }
  return [4, 2, 1, 1];
}

export interface SparseSurfaceExecutionSource {
  /** Header + active (brick, leaf) pairs emitted by GPU fluid residency. */
  worklist: GPUBuffer;
  /** Per-logical-brick resident flags used to reject stale jump-flood seeds. */
  states: GPUBuffer;
  brickSize: 4 | 8;
}

/** Per-step nozzle state for sourcing fluid into the resident level set. */
export interface SurfaceInflowState {
  outletCenter_m: { x: number; y: number; z: number };
  radius_m: number;
  velocity_m_s: { x: number; y: number; z: number };
  apertureScale: number;
  strength: number;
}

/** Catastrophic level-set loss circuit breaker with wide hysteresis. */
export function nextQuadtreeVofReconciliationActive(active: boolean, representedVolumeDrift: number) {
  if (!Number.isFinite(representedVolumeDrift)) return active;
  return active ? representedVolumeDrift < -0.02 : representedVolumeDrift < -0.10;
}

export function quadtreeVofReconciliationFraction(missingVolumeCells: number, mismatchCells: number) {
  if (!(missingVolumeCells > 0) || !(mismatchCells > 0)) return 0;
  return Math.max(1 / 512, Math.min(1 / 32, missingVolumeCells / (8 * mismatchCells)));
}

export interface WebGPUQuadtreeSurfaceCache {
  layout: GPUBindGroupLayout;
  pipelineLayout: GPUPipelineLayout;
  shaderModule: GPUShaderModule;
  pipelines: {
    advectLevelSet: GPUComputePipeline;
    advectPredict: GPUComputePipeline;
    advectReverse: GPUComputePipeline;
    advectCorrect: GPUComputePipeline;
    reduceVolume: GPUComputePipeline;
    seedDistance: GPUComputePipeline;
    jumpFlood: GPUComputePipeline;
    finalizeDistance: GPUComputePipeline;
    correctLevelSetVolume: GPUComputePipeline;
    commitLevelSetVolumeCorrection: GPUComputePipeline;
    copyLevelSet: GPUComputePipeline;
    cullDebris: GPUComputePipeline;
  };
}

/**
 * Per-step level-set transport used by the adaptive pressure path.
 *
 * Narita Alg. 1 advects phi once per physics step. Keeping a canonical GPU
 * texture plus a scratch texture makes that cadence independent of the much
 * slower CPU sparse-topology rebuild. The canonical texture never changes
 * identity, so every projection rebuilt from the shared state sees the newest
 * surface without a CPU upload.
 */
export const quadtreeSurfaceShader = /* wgsl */ `
struct SurfaceParams { dims: vec4u, cellAndDt: vec4f, control: vec4f, cellGravity: vec4f, container: vec4f, inflowPositionRadius: vec4f, inflowVelocityLength: vec4f, inflowTiming: vec4f }
struct PassParams { jump: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var phiIn: texture_3d<f32>;
@group(0) @binding(2) var phiOut: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var<storage, read> distanceSeedsIn: array<vec2u>;
@group(0) @binding(4) var<storage, read_write> distanceSeedsOut: array<vec2u>;
@group(0) @binding(5) var<uniform> params: SurfaceParams;
@group(0) @binding(6) var<uniform> passParams: PassParams;
@group(0) @binding(7) var<storage, read_write> reductions: array<atomic<u32>>;
@group(0) @binding(8) var predictedPhiIn: texture_3d<f32>;
@group(0) @binding(9) var reversedPhiIn: texture_3d<f32>;
@group(0) @binding(10) var reconcileVolumeIn: texture_3d<f32>;
struct SurfaceSolidCell { fraction: f32, owner: i32 }
@group(0) @binding(11) var<storage, read> surfaceSolids: array<SurfaceSolidCell>;
// Optional sparse execution source. params.container.w is the brick size when
// enabled and zero for the dense reference path.
@group(0) @binding(12) var<storage, read> surfaceBrickWorklist: array<u32>;
@group(0) @binding(13) var<storage, read> surfaceBrickStates: array<u32>;

fn sparseSurfaceEnabled() -> bool { return params.container.w >= 4.0; }
fn surfaceLocalCell(localIndex: u32, brickSize: u32) -> vec3u {
  return vec3u(localIndex % brickSize, (localIndex / brickSize) % brickSize,
    localIndex / (brickSize * brickSize));
}
fn surfaceBrickCoordinate(brickIndex: u32, brickSize: u32) -> vec3u {
  let brickDims = (params.dims.xyz + vec3u(brickSize - 1u)) / brickSize;
  return vec3u(brickIndex % brickDims.x, (brickIndex / brickDims.x) % brickDims.y,
    brickIndex / (brickDims.x * brickDims.y));
}
// Dense launches use ordinary 4x4x4 workgroups. Sparse launches use the
// residency worklist's 64-thread indirect stream (header words 12..14).
fn surfaceDispatchCell(wid: vec3u, localIndex: u32) -> vec3u {
  if (!sparseSurfaceEnabled()) {
    return wid * vec3u(4u) + surfaceLocalCell(localIndex, 4u);
  }
  let stream = (wid.x + wid.y * surfaceBrickWorklist[12]) * 64u + localIndex;
  let brickSize = u32(params.container.w);
  let voxelsPerBrick = brickSize * brickSize * brickSize;
  let brickSlot = stream / voxelsPerBrick;
  if (brickSlot >= surfaceBrickWorklist[0]) { return params.dims.xyz; }
  let brickIndex = surfaceBrickWorklist[16u + 2u * brickSlot];
  return surfaceBrickCoordinate(brickIndex, brickSize) * brickSize
    + surfaceLocalCell(stream % voxelsPerBrick, brickSize);
}
fn sparseSurfaceCellResident(q: vec3i) -> bool {
  if (!sparseSurfaceEnabled()) { return true; }
  if (any(q < vec3i(0)) || any(q >= vec3i(params.dims.xyz))) { return false; }
  let brickSize = u32(params.container.w);
  let brickDims = (params.dims.xyz + vec3u(brickSize - 1u)) / brickSize;
  let brick = vec3u(q) / brickSize;
  let brickIndex = brick.x + brickDims.x * (brick.y + brickDims.y * brick.z);
  return brickIndex < arrayLength(&surfaceBrickStates) && (surfaceBrickStates[brickIndex] & 1u) != 0u;
}

fn index3(q: vec3u) -> u32 { return q.x + params.dims.x * (q.y + params.dims.y * q.z); }
fn clamp3(q: vec3i) -> vec3i { return clamp(q, vec3i(0), vec3i(params.dims.xyz) - vec3i(1)); }
fn loadPhi(q: vec3i) -> f32 { return textureLoad(phiIn, clamp3(q), 0).x; }
fn loadVelocity(q: vec3i) -> vec3f { return textureLoad(velocityIn, clamp3(q), 0).xyz; }
fn loadReconcileAlpha(q: vec3i) -> f32 { return clamp(textureLoad(reconcileVolumeIn, clamp3(q), 0).x, 0.0, 1.0); }
fn centredMacVelocity(q: vec3i) -> vec3f {
  let own = loadVelocity(q);
  // velocityIn stores the negative-face MAC sample for each component. The
  // level-set trace needs the cell-centre velocity (Narita Alg. 1 line 4).
  return 0.5 * vec3f(
    own.x + loadVelocity(q - vec3i(1, 0, 0)).x,
    own.y + loadVelocity(q - vec3i(0, 1, 0)).y,
    own.z + loadVelocity(q - vec3i(0, 0, 1)).z
  );
}
fn trilinearPhi(position: vec3f) -> f32 {
  let hi = vec3f(params.dims.xyz - vec3u(1));
  let p = clamp(position, vec3f(0.0), hi);
  let a = vec3i(floor(p)); let b = min(a + vec3i(1), vec3i(params.dims.xyz) - vec3i(1)); let t = fract(p);
  let x00 = mix(loadPhi(vec3i(a.x, a.y, a.z)), loadPhi(vec3i(b.x, a.y, a.z)), t.x);
  let x10 = mix(loadPhi(vec3i(a.x, b.y, a.z)), loadPhi(vec3i(b.x, b.y, a.z)), t.x);
  let x01 = mix(loadPhi(vec3i(a.x, a.y, b.z)), loadPhi(vec3i(b.x, a.y, b.z)), t.x);
  let x11 = mix(loadPhi(vec3i(a.x, b.y, b.z)), loadPhi(vec3i(b.x, b.y, b.z)), t.x);
  return mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
}
fn hMin() -> f32 { return min(params.cellAndDt.x, min(params.cellAndDt.y, params.cellAndDt.z)); }
fn volumeCorrectedPhi(value: f32, q: vec3i) -> f32 {
  let h = hMin();
  // Paper-aligned normal displacement: phi's own represented-volume error
  // supplies the speed and only the signed-distance interface band moves.
  if (!(abs(value) < 1.5 * h)) { return value; }
  let corrected = value - params.control.x * h * params.cellAndDt.w;
  // Localized controller: when phi and the conservative VOF field decisively
  // disagree somewhere, the push concentrates on the cells whose disagreement
  // it reduces instead of uniformly sanding down agreeing thin sheets and
  // droplets. cellGravity.w blends back to the pure global controller as the
  // measured mismatch approaches zero (and stays exactly 1 when no VOF field
  // is bound), so healthy transport keeps the legacy behavior unchanged.
  var weight = params.cellGravity.w;
  if (weight < 1.0) {
    let wet = loadReconcileAlpha(q) >= 0.5;
    let shrinking = params.control.x < 0.0;
    let helped = select((value >= 0.0) && wet, (value < 0.0) && !wet, shrinking);
    if (helped) { weight = 1.0; }
  }
  return mix(value, corrected, weight);
}
fn inflowGridDims() -> vec3i { return vec3i(params.dims.xyz); }
${inflowBoundaryWGSL}
fn centredMacVelocityAt(position: vec3f) -> vec3f {
  let hi = vec3f(params.dims.xyz - vec3u(1));
  let p = clamp(position, vec3f(0.0), hi);
  let a = vec3i(floor(p)); let b = min(a + vec3i(1), vec3i(params.dims.xyz) - vec3i(1)); let t = fract(p);
  let x00 = mix(centredMacVelocity(vec3i(a.x, a.y, a.z)), centredMacVelocity(vec3i(b.x, a.y, a.z)), t.x);
  let x10 = mix(centredMacVelocity(vec3i(a.x, b.y, a.z)), centredMacVelocity(vec3i(b.x, b.y, a.z)), t.x);
  let x01 = mix(centredMacVelocity(vec3i(a.x, a.y, b.z)), centredMacVelocity(vec3i(b.x, a.y, b.z)), t.x);
  let x11 = mix(centredMacVelocity(vec3i(a.x, b.y, b.z)), centredMacVelocity(vec3i(b.x, b.y, b.z)), t.x);
  return mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
}
// RK2 midpoint backtrace, mirroring the restricted method's traceDeparture.
fn departurePoint(p: vec3f, dt: f32) -> vec3f {
  let cellsPerMetre = vec3f(1.0) / params.cellAndDt.xyz;
  let first = centredMacVelocityAt(p);
  let midpoint = p - 0.5 * first * dt * cellsPerMetre;
  return p - centredMacVelocityAt(midpoint) * dt * cellsPerMetre;
}
@compute @workgroup_size(4, 4, 4)
fn advectLevelSet(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) localIndex: u32) {
  let gid = surfaceDispatchCell(wid, localIndex);
  if (any(gid >= params.dims.xyz)) { return; }
  // Narita et al. Sec. 4.5: interpolate velocity from the saved previous
  // staggered grid, backtrace, then interpolate the previous level set.
  let q = vec3i(gid);
  let departure = departurePoint(vec3f(gid), params.cellAndDt.w);
  var phi = volumeCorrectedPhi(trilinearPhi(departure), q);
  // The nozzle sources fluid directly into the resident surface, exactly as
  // the restricted method's finishAdvection clamps phi at inflow cells.
  if (isInflowVelocityCell(q)) { phi = min(phi, -0.5 * hMin() * inflowApertureFraction(q) * inflowStrength()); }
  if (sparseSurfaceEnabled()) { accumulateSparseVolumeDelta(loadPhi(q), phi, gid); }
  textureStore(phiOut, q, vec4f(phi, 0.0, 0.0, 0.0));
}
@compute @workgroup_size(4, 4, 4)
fn advectPredict(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) localIndex: u32) {
  let gid = surfaceDispatchCell(wid, localIndex);
  if (any(gid >= params.dims.xyz)) { return; }
  textureStore(phiOut, vec3i(gid), vec4f(trilinearPhi(departurePoint(vec3f(gid), params.cellAndDt.w)), 0.0, 0.0, 0.0));
}
@compute @workgroup_size(4, 4, 4)
fn advectReverse(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) localIndex: u32) {
  let gid = surfaceDispatchCell(wid, localIndex);
  // phiIn is the predicted field; tracing it forward (negative dt backtrace)
  // recovers the BFECC error estimate.
  if (any(gid >= params.dims.xyz)) { return; }
  textureStore(phiOut, vec3i(gid), vec4f(trilinearPhi(departurePoint(vec3f(gid), -params.cellAndDt.w)), 0.0, 0.0, 0.0));
}
@compute @workgroup_size(4, 4, 4)
fn advectCorrect(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) localIndex: u32) {
  let gid = surfaceDispatchCell(wid, localIndex);
  if (any(gid >= params.dims.xyz)) { return; }
  let q = vec3i(gid);
  let original = loadPhi(q);
  let predicted = textureLoad(predictedPhiIn, q, 0).x;
  let reversed = textureLoad(reversedPhiIn, q, 0).x;
  var corrected = predicted + 0.5 * (original - reversed);
  // Bounded MacCormack: clamp to the eight previous-phi corners around the
  // forward departure point, falling back to the monotone prediction
  // (mirrors the restricted method's boundedMacCormack).
  let hi = vec3f(params.dims.xyz - vec3u(1));
  let a = vec3i(floor(clamp(departurePoint(vec3f(gid), params.cellAndDt.w), vec3f(0.0), hi)));
  var lower = loadPhi(a); var upper = lower;
  for (var corner = 1u; corner < 8u; corner += 1u) {
    let offset = vec3i(i32(corner & 1u), i32((corner >> 1u) & 1u), i32((corner >> 2u) & 1u));
    let value = loadPhi(a + offset); lower = min(lower, value); upper = max(upper, value);
  }
  if (corrected < lower || corrected > upper) { corrected = predicted; }
  var phi = volumeCorrectedPhi(corrected, q);
  if (isInflowVelocityCell(q)) { phi = min(phi, -0.5 * hMin() * inflowApertureFraction(q) * inflowStrength()); }
  if (sparseSurfaceEnabled()) { accumulateSparseVolumeDelta(loadPhi(q), phi, gid); }
  textureStore(phiOut, q, vec4f(phi, 0.0, 0.0, 0.0));
}
fn surfaceOpenFraction(gid: vec3u) -> f32 {
  if (params.inflowTiming.z < 0.5) { return 1.0; }
  return 1.0 - clamp(surfaceSolids[index3(gid)].fraction, 0.0, 1.0);
}
fn accumulateVolume(value: f32, gid: vec3u) {
  // Match the renderer and smoke oracle: adaptive phi is converted to a
  // smooth Heaviside over four vertical cell widths. Round the fixed-point
  // contribution instead of truncating it. Truncation introduces a negative
  // error for every mixed cell; as a dam break stretches its interface that
  // bias grows with surface area and makes the controller expand phi even
  // when the exact represented volume is already correct.
  let occupied = clamp(0.5 - value / (4.0 * params.cellAndDt.y), 0.0, 1.0);
  // A moving solid changes the volume available to liquid. Conserving raw
  // phi occupancy would count liquid hidden inside the body and cancel the
  // free-surface rise that should accompany immersion. Instead conserve the
  // physical open-fluid volume alpha*(1-s); alpha*s is independently reported
  // as displaced volume by the rigid coupling pass.
  let open = surfaceOpenFraction(gid);
  atomicAdd(&reductions[0], u32(occupied * open * 256.0 + 0.5));
  // The derivative of that Heaviside is non-zero for |phi| < 2*hy. Counting
  // the same support makes the global Newton displacement consistent with
  // the represented-volume functional it is correcting.
  if (abs(value) < 2.0 * params.cellAndDt.y) { atomicAdd(&reductions[1], u32(open * 256.0 + 0.5)); }
}
fn representedOccupancy(value: f32, gid: vec3u) -> f32 {
  return clamp(0.5 - value / (4.0 * params.cellAndDt.y), 0.0, 1.0) * surfaceOpenFraction(gid);
}
// Sparse execution keeps the global volume total persistent and applies only
// each active cell's signed change. Two's-complement addition through u32
// atomics is exact modulo 2^32 for the bounded per-step deltas here.
fn accumulateSparseVolumeDelta(before: f32, after: f32, gid: vec3u) {
  let delta = i32(round((representedOccupancy(after, gid) - representedOccupancy(before, gid)) * 256.0));
  atomicAdd(&reductions[0], bitcast<u32>(delta));
}
fn accumulateSparseInterface(value: f32, gid: vec3u) {
  if (abs(value) < 2.0 * params.cellAndDt.y) {
    atomicAdd(&reductions[1], u32(surfaceOpenFraction(gid) * 256.0 + 0.5));
  }
}
@compute @workgroup_size(4, 4, 4)
fn reduceVolume(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) localIndex: u32) {
  let gid = surfaceDispatchCell(wid, localIndex);
  if (any(gid >= params.dims.xyz)) { return; }
  accumulateVolume(loadPhi(vec3i(gid)), gid);
}
// Seeds carry the interface point itself (the cell's phi projected along the
// gradient) in 16.6 fixed point per axis, not the seed cell's coordinates.
// Measuring the jump flood against projected points keeps the rebuilt far
// field second-order: measuring against cell centres and adding the seed's
// own offset at finalize overestimates by up to a cell for tangential seeds.
fn packSeedPoint(p: vec3f) -> vec2u {
  let q = vec3u(clamp(p * 64.0, vec3f(0.0), vec3f(65535.0)));
  return vec2u(q.x | (q.y << 16u), q.z);
}
fn unpackSeedPoint(word: vec2u) -> vec3f {
  return vec3f(f32(word.x & 0xffffu), f32(word.x >> 16u), f32(word.y & 0xffffu)) / 64.0;
}
fn reconciliationSelected(gid: vec3u) -> bool {
  var hash = index3(gid) ^ (params.dims.w * 747796405u + 2891336453u);
  hash = (hash ^ (hash >> 16u)) * 2246822519u; hash = (hash ^ (hash >> 13u)) * 3266489917u; hash ^= hash >> 16u;
  return f32(hash & 0xffffu) < params.control.w * 65536.0;
}
@compute @workgroup_size(4, 4, 4)
fn seedDistance(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) localIndex: u32) {
  let gid = surfaceDispatchCell(wid, localIndex);
  if (any(gid >= params.dims.xyz)) { return; }
  let p = vec3i(gid); let wet = loadPhi(p) < 0.0;
  var crosses = (loadPhi(p + vec3i(1, 0, 0)) < 0.0) != wet || (loadPhi(p - vec3i(1, 0, 0)) < 0.0) != wet
    || (loadPhi(p + vec3i(0, 1, 0)) < 0.0) != wet || (loadPhi(p - vec3i(0, 1, 0)) < 0.0) != wet
    || (loadPhi(p + vec3i(0, 0, 1)) < 0.0) != wet || (loadPhi(p - vec3i(0, 0, 1)) < 0.0) != wet;
  var word = vec2u(0xffffffffu, 0xffffffffu);
  if (crosses) {
    let h = params.cellAndDt.xyz;
    let gradient = vec3f(
        (loadPhi(p + vec3i(1, 0, 0)) - loadPhi(p - vec3i(1, 0, 0))) / (2.0 * h.x),
        (loadPhi(p + vec3i(0, 1, 0)) - loadPhi(p - vec3i(0, 1, 0))) / (2.0 * h.y),
        (loadPhi(p + vec3i(0, 0, 1)) - loadPhi(p - vec3i(0, 0, 1))) / (2.0 * h.z));
    let magnitude = max(length(gradient), 1e-6);
    // Chopp-style sub-cell distance, clamped inside the cell so a degenerate
    // gradient cannot eject the interface point.
    let distance = clamp(loadPhi(p) / magnitude, -0.87 * hMin(), 0.87 * hMin());
    let point = vec3f(gid) - distance * (gradient / magnitude) / h;
    word = packSeedPoint(clamp(point, vec3f(0.0), vec3f(params.dims.xyz - vec3u(1))));
  }
  distanceSeedsOut[index3(gid)] = word;
}
fn seedDistanceSquared(cell: vec3u, word: vec2u) -> f32 {
  if (word.y > 0xffffu) { return 3.402823e38; }
  let delta = (vec3f(cell) - unpackSeedPoint(word)) * params.cellAndDt.xyz;
  return dot(delta, delta);
}
@compute @workgroup_size(4, 4, 4)
fn jumpFlood(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) localIndex: u32) {
  let gid = surfaceDispatchCell(wid, localIndex);
  if (any(gid >= params.dims.xyz)) { return; }
  var best = distanceSeedsIn[index3(gid)]; var bestDistance = seedDistanceSquared(gid, best); let jump = i32(passParams.jump);
  for (var dz = -1; dz <= 1; dz += 1) { for (var dy = -1; dy <= 1; dy += 1) { for (var dx = -1; dx <= 1; dx += 1) {
    let q = clamp(vec3i(gid) + vec3i(dx, dy, dz) * jump, vec3i(0), vec3i(params.dims.xyz) - vec3i(1));
    if (!sparseSurfaceCellResident(q)) { continue; }
    let candidate = distanceSeedsIn[index3(vec3u(q))]; let candidateDistance = seedDistanceSquared(gid, candidate);
    if (candidateDistance < bestDistance) { best = candidate; bestDistance = candidateDistance; }
  } } }
  distanceSeedsOut[index3(gid)] = best;
}
@compute @workgroup_size(4, 4, 4)
fn finalizeDistance(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) localIndex: u32) {
  let gid = surfaceDispatchCell(wid, localIndex);
  if (any(gid >= params.dims.xyz)) { return; }
  let advected = loadPhi(vec3i(gid));
  let h = hMin();
  var result = advected;
  // The narrow band keeps the advected phi verbatim, so redistancing never
  // moves the interface. 2.5h keeps every cell a CFL-bounded backtrace can
  // sample from inside the smoothly advected field; jump flood supplies only
  // the far-field signed distance and never becomes the surface authority.
  //
  // Band membership follows the jump-flood distance to the CURRENT interface,
  // not the advected magnitude. A cell swept by a moving interface keeps a
  // fossil near-zero value even when it is now deep inside one phase; judged
  // by |advected| alone it stays "narrow band" forever, its gradient decays
  // to ~0 (the contour can no longer be transported at the flow speed), and
  // the volume controller can eventually walk its sign across zero.
  let word = distanceSeedsIn[index3(gid)];
  let hasSeed = word.y <= 0xffffu;
  let interfaceDistance = select(3.402823e38, sqrt(seedDistanceSquared(gid, word)), hasSeed);
  if (abs(advected) >= 2.5 * h || interfaceDistance >= 2.5 * h) {
    var distance = 5.0 * h;
    if (hasSeed) { distance = min(5.0 * h, max(2.5 * h, interfaceDistance)); }
    result = select(distance, -distance, advected < 0.0);
  }
  // VOF is not part of ordinary surface evolution. It is consulted only when
  // the wide-hysteresis circuit breaker has detected catastrophic represented
  // liquid loss, and then only to restore liquid phi failed to carry.
  let alpha = loadReconcileAlpha(vec3i(gid));
  let wet = alpha >= 0.5;
  let signMismatch = (result < 0.0) != wet;
  let decisiveMismatch = signMismatch && abs(result) > 0.5 * h;
  if (decisiveMismatch) {
    atomicAdd(&reductions[3], 1u);
    if (params.control.y > 0.5 && wet && reconciliationSelected(gid)) {
      // Invert the same four-cell smooth Heaviside used by reduceVolume and
      // the renderer. A fixed +/-0.5h reseed assigns every repaired dry cell
      // 37.5% liquid (and every wet cell 62.5%), creating phantom represented
      // volume even though conservative VOF mass is exact.
      let volumePhi = (0.5 - alpha) * (4.0 * params.cellAndDt.y);
      result = select(max(0.02 * h, volumePhi), min(-0.02 * h, volumePhi), wet);
    }
  }
  // In the normal path finalization already owns the exact value that the
  // following reduction used to reload from the texture. Debris culling can
  // still change that value, so its opt-in path reduces after the cull/copy.
  if (params.control.z <= 0.5) {
    if (sparseSurfaceEnabled()) { accumulateSparseInterface(result, gid); }
    else { accumulateVolume(result, gid); }
  }
  textureStore(phiOut, vec3i(gid), vec4f(result, 0.0, 0.0, 0.0));
}
@compute @workgroup_size(4, 4, 4)
fn correctLevelSetVolume(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) localIndex: u32) {
  let gid = surfaceDispatchCell(wid, localIndex);
  if (any(gid >= params.dims.xyz)) { return; }
  let represented = f32(atomicLoad(&reductions[0])) / 256.0;
  let desiredVolume = params.inflowTiming.y;
  let interfaceCells = max(1.0, f32(atomicLoad(&reductions[1])) / 256.0);
  // H(phi)=clamp(0.5-phi/(4*hy),0,1), so a uniform distance shift changes
  // represented volume by approximately -interfaceCells*shift/(4*hy).
  // Clamp to a narrow-band-sized correction so one noisy frame cannot move
  // the surface discontinuously.
  let shift = clamp((represented - desiredVolume) * (4.0 * params.cellAndDt.y) / interfaceCells,
                    -1.5 * hMin(), 1.5 * hMin());
  textureStore(phiOut, vec3i(gid), vec4f(loadPhi(vec3i(gid)) + shift, 0.0, 0.0, 0.0));
}
@compute @workgroup_size(1)
fn commitLevelSetVolumeCorrection() {
  let represented = f32(atomicLoad(&reductions[0])) / 256.0;
  let interfaceCells = max(1.0, f32(atomicLoad(&reductions[1])) / 256.0);
  let shift = clamp((represented - params.inflowTiming.y) * (4.0 * params.cellAndDt.y) / interfaceCells,
                    -1.5 * hMin(), 1.5 * hMin());
  let corrected = max(0.0, represented - interfaceCells * shift / (4.0 * params.cellAndDt.y));
  atomicStore(&reductions[0], u32(corrected * 256.0 + 0.5));
}
@compute @workgroup_size(4, 4, 4)
fn copyLevelSet(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) localIndex: u32) {
  let gid = surfaceDispatchCell(wid, localIndex);
  if (any(gid >= params.dims.xyz)) { return; }
  textureStore(phiOut, vec3i(gid), vec4f(loadPhi(vec3i(gid)), 0.0, 0.0, 0.0));
}
@compute @workgroup_size(4, 4, 4)
fn cullDebris(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) localIndex: u32) {
  let gid = surfaceDispatchCell(wid, localIndex);
  if (any(gid >= params.dims.xyz)) { return; }
  let q = vec3i(gid); let value = loadPhi(q); var result = value;
  if (params.control.z > 0.5 && value < 0.0 && textureLoad(reconcileVolumeIn, q, 0).x < 0.5) {
    let threshold = 0.25 * hMin();
    let isolated = loadPhi(q + vec3i(1, 0, 0)) > threshold && loadPhi(q - vec3i(1, 0, 0)) > threshold
      && loadPhi(q + vec3i(0, 1, 0)) > threshold && loadPhi(q - vec3i(0, 1, 0)) > threshold
      && loadPhi(q + vec3i(0, 0, 1)) > threshold && loadPhi(q - vec3i(0, 0, 1)) > threshold;
    if (isolated) { result = 0.5 * hMin(); atomicAdd(&reductions[2], 1u); }
  }
  textureStore(phiOut, q, vec4f(result, 0.0, 0.0, 0.0));
}
`;

function uploadLevelSetTexture(device: GPUDevice, texture: GPUTexture, phi: Float32Array, nx: number, ny: number, nz: number) {
  const rowBytes = nx * 4, pitch = Math.ceil(rowBytes / 256) * 256;
  const upload = new Uint8Array(pitch * ny * nz), source = new Uint8Array(phi.buffer, phi.byteOffset, phi.byteLength);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) upload.set(source.subarray(rowBytes * (y + ny * z), rowBytes * (y + ny * z + 1)), pitch * (y + ny * z));
  device.queue.writeTexture({ texture }, upload, { bytesPerRow: pitch, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: nz });
}

const quadtreeSurfaceCacheByDevice = new WeakMap<GPUDevice, WebGPUQuadtreeSurfaceCache>();
const quadtreeSurfaceCacheCompilationByDevice = new WeakMap<GPUDevice, Promise<WebGPUQuadtreeSurfaceCache>>();

/**
 * Compile the transported-surface pipeline family without blocking the
 * constructor. Concurrent callers share one manager-owned compilation and
 * all later callers reuse the published cache for the device generation.
 */
export function createWebGPUQuadtreeSurfaceCache(
  device: GPUDevice,
  cache?: WebGPUQuadtreeSurfaceCache,
): Promise<WebGPUQuadtreeSurfaceCache> {
  if (cache) return Promise.resolve(cache);
  const published = quadtreeSurfaceCacheByDevice.get(device);
  if (published) return Promise.resolve(published);
  const pending = quadtreeSurfaceCacheCompilationByDevice.get(device);
  if (pending) return pending;

  const compilation = compileWebGPUQuadtreeSurfaceCache(device).then((compiled) => {
    const result = quadtreeSurfaceCacheByDevice.get(device) ?? compiled;
    quadtreeSurfaceCacheByDevice.set(device, result);
    return result;
  }).finally(() => {
    quadtreeSurfaceCacheCompilationByDevice.delete(device);
  });
  quadtreeSurfaceCacheCompilationByDevice.set(device, compilation);
  return compilation;
}

async function compileWebGPUQuadtreeSurfaceCache(device: GPUDevice): Promise<WebGPUQuadtreeSurfaceCache> {
  const layout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
    { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 8, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 9, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 10, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
  ] });
  const compiler = gpuCompilationManagerFor(device);
  const shaderModule = compiler.createShaderModule({ label: "Resident quadtree level set", code: quadtreeSurfaceShader });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipeline = (entryPoint: string) => compiler.compileComputePipeline({
    label: `Quadtree surface ${entryPoint}`,
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint },
  });
  const [advectLevelSet, advectPredict, advectReverse, advectCorrect, reduceVolume,
    seedDistance, jumpFlood, finalizeDistance, correctLevelSetVolume,
    commitLevelSetVolumeCorrection, copyLevelSet, cullDebris] = await Promise.all([
    pipeline("advectLevelSet"), pipeline("advectPredict"), pipeline("advectReverse"),
    pipeline("advectCorrect"), pipeline("reduceVolume"), pipeline("seedDistance"),
    pipeline("jumpFlood"), pipeline("finalizeDistance"), pipeline("correctLevelSetVolume"),
    pipeline("commitLevelSetVolumeCorrection"), pipeline("copyLevelSet"), pipeline("cullDebris"),
  ]);
  return { layout, pipelineLayout, shaderModule, pipelines: {
    advectLevelSet, advectPredict, advectReverse, advectCorrect, reduceVolume,
    seedDistance, jumpFlood, finalizeDistance, correctLevelSetVolume,
    commitLevelSetVolumeCorrection, copyLevelSet, cullDebris,
  } };
}

export class WebGPUQuadtreeSurfaceState {
  readonly cache?: WebGPUQuadtreeSurfaceCache;
  readonly texture: GPUTexture;
  /** Retained so a warm re-seed can rewrite phi without rebuilding the state. */
  private uploadedDimensions?: { nx: number; ny: number; nz: number };
  private volumeBandForReseed = 0;
  private cellForReseed?: { x: number; y: number; z: number };

  /**
   * Overwrite the resident level set in place for a warm re-seed, restoring the
   * volume references to the t=0 values the fresh-build path would have set.
   * The texture, its bindings, and every dependent allocation are untouched —
   * only its contents change, which is what makes the re-seed cheap.
   */
  reseedLevelSet(device: GPUDevice, phi: Float32Array): boolean {
    const dims = this.uploadedDimensions, cell = this.cellForReseed;
    if (!dims || !cell || phi.length !== dims.nx * dims.ny * dims.nz) return false;
    uploadLevelSetTexture(device, this.texture, phi, dims.nx, dims.ny, dims.nz);
    this.referenceVolumeCells = phi.reduce((sum, value) => sum + Math.max(0, Math.min(1, 0.5 - value / this.volumeBandForReseed)), 0);
    this.volumeCells = this.referenceVolumeCells;
    this.interfaceCells = phi.reduce((sum, value) => sum + (Math.abs(value) < 1.5 * Math.min(cell.x, cell.y, cell.z) ? 1 : 0), 0);
    return true;
  }
  private readonly scratch?: GPUTexture;
  private readonly predicted?: GPUTexture;
  private readonly reversed?: GPUTexture;
  private readonly seedsA?: GPUBuffer;
  private readonly seedsB?: GPUBuffer;
  private readonly params?: GPUBuffer;
  private readonly passBuffer?: GPUBuffer;
  private readonly reductions?: GPUBuffer;
  private passStride = 0;
  private jumps: number[] = [];
  private readonly groups?: { advect: GPUBindGroup; predict: GPUBindGroup; reverse: GPUBindGroup; correct: GPUBindGroup; reduce: GPUBindGroup; seed: GPUBindGroup; jumpAB: GPUBindGroup; jumpBA: GPUBindGroup; finalizeA: GPUBindGroup; finalizeB: GPUBindGroup; cull: GPUBindGroup };
  private readbackPending = false;
  private referenceVolumeCells: number;
  private volumeCells: number;
  private interfaceCells = 0;
  private culledDebrisCells = 0;
  private mismatchCells = 0;
  private correctionSpeed = 0;
  /** 1 = legacy global volume controller; <1 concentrates it on phi/VOF disagreement. */
  private volumeControlAgreeWeight = 1;
  private readonly hasReconcileVolume: boolean;
  private readonly reconcileEnabled: boolean;
  private reconcileActive = false;
  private reconcileFraction = 0;
  private surfaceSequence = 0;
  private readonly ownedReconcileFallback?: GPUTexture;
  private readonly ownedSolidFallback?: GPUBuffer;
  private readonly ownedSparseFallback?: GPUBuffer;

  constructor(private readonly device: GPUDevice, private readonly dims: { nx: number; ny: number; nz: number }, private readonly cell: { x: number; y: number; z: number }, velocity: GPUTexture | undefined, initialPhi: Float32Array, cache?: WebGPUQuadtreeSurfaceCache, reconcileVolume?: GPUTexture, private readonly debrisCulling = false, reconcileEnabled = reconcileVolume !== undefined, private readonly gpuVolumeCorrection = false, private readonly monotoneLevelSetTransport = false, private readonly solidFractions?: GPUBuffer, private readonly sparseExecution?: SparseSurfaceExecutionSource, private readonly presentationOnly = false, private readonly placeholderOnly = false) {
    this.cache = cache;
    if (!presentationOnly && !cache) {
      throw new Error("A transported quadtree surface requires an initialized pipeline cache; await createWebGPUQuadtreeSurfaceCache(device) before construction");
    }
    this.hasReconcileVolume = reconcileVolume !== undefined;
    this.reconcileEnabled = reconcileEnabled && reconcileVolume !== undefined;
    const textureUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC;
    if (placeholderOnly && (!presentationOnly || initialPhi.length !== 1)) {
      throw new RangeError("A placeholder surface publication requires presentation-only mode and one phi sample");
    }
    const textureDimensions = placeholderOnly ? { nx: 1, ny: 1, nz: 1 } : dims;
    this.texture = device.createTexture({ label: placeholderOnly ? "Quadtree level-set format placeholder" : "Resident quadtree level set", size: [textureDimensions.nx, textureDimensions.ny, textureDimensions.nz], dimension: "3d", format: "r32float", usage: textureUsage });
    uploadLevelSetTexture(device, this.texture, initialPhi, textureDimensions.nx, textureDimensions.ny, textureDimensions.nz);
    this.uploadedDimensions = textureDimensions;
    const volumeBand = 4 * cell.y;
    this.referenceVolumeCells = initialPhi.reduce((sum, value) => sum + Math.max(0, Math.min(1, 0.5 - value / volumeBand)), 0);
    this.volumeCells = this.referenceVolumeCells;
    this.interfaceCells = initialPhi.reduce((sum, value) => sum + (Math.abs(value) < 1.5 * Math.min(cell.x, cell.y, cell.z) ? 1 : 0), 0);
    this.volumeBandForReseed = volumeBand;
    this.cellForReseed = cell;
    // Global-fine storage owns transport, redistance, and volume control.
    // Retain only the topology/render publication texture; in particular do
    // not hide box-sized scratch allocations behind unused bind groups.
    if (presentationOnly) return;
    if (!velocity) throw new Error("A transported surface state requires a dense velocity texture");
    this.scratch = device.createTexture({ label: "Resident quadtree level-set advection scratch", size: [dims.nx, dims.ny, dims.nz], dimension: "3d", format: "r32float", usage: textureUsage });
    this.predicted = device.createTexture({ label: "Resident quadtree level-set MacCormack prediction", size: [dims.nx, dims.ny, dims.nz], dimension: "3d", format: "r32float", usage: textureUsage });
    this.reversed = device.createTexture({ label: "Resident quadtree level-set MacCormack reversal", size: [dims.nx, dims.ny, dims.nz], dimension: "3d", format: "r32float", usage: textureUsage });
    const bytes = Math.max(8, dims.nx * dims.ny * dims.nz * 8), seedUsage = GPUBufferUsage.STORAGE;
    this.seedsA = device.createBuffer({ label: "Quadtree surface seeds A", size: bytes, usage: seedUsage });
    this.seedsB = device.createBuffer({ label: "Quadtree surface seeds B", size: bytes, usage: seedUsage });
    this.params = device.createBuffer({ label: "Quadtree surface parameters", size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.reductions = device.createBuffer({ label: "Quadtree level-set volume diagnostics", size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.reductions, 0, new Uint32Array([Math.round(this.volumeCells * 256), Math.round(this.interfaceCells * 256), 0, 0]));
    this.jumps = quadtreeSurfaceJumpSequence(dims.nx, dims.ny, dims.nz);
    const alignment = device.limits.minUniformBufferOffsetAlignment;
    this.passStride = Math.ceil(16 / alignment) * alignment;
    const passData = new Uint8Array(Math.max(this.passStride, this.passStride * this.jumps.length));
    this.jumps.forEach((jump, index) => new Uint32Array(passData.buffer, index * this.passStride, 4).set([jump, 0, 0, 0]));
    this.passBuffer = device.createBuffer({ label: "Quadtree surface pass parameters", size: passData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.passBuffer, 0, passData);
    // Bindings 8/9 (MacCormack predicted/reversed inputs) must never alias a
    // group's storage output: WebGPU rejects sampled+writable usage of one
    // texture in the same dispatch scope. Only `correct` reads them for real;
    // every other group binds textures it does not write.
    // The reconcile binding needs a texture even when reconciliation is off
    // (pure level-set transport, as the redistance/transport tests exercise);
    // a one-texel fallback keeps the layout uniform and control.y gates reads.
    const reconcileTexture = reconcileVolume ?? (this.ownedReconcileFallback = device.createTexture({ label: "Quadtree surface reconcile fallback", size: [1, 1, 1], dimension: "3d", format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING }));
    const surfaceSolidBuffer = solidFractions ?? (this.ownedSolidFallback = device.createBuffer({ label: "Quadtree surface solid fallback", size: 8, usage: GPUBufferUsage.STORAGE }));
    const sparseFallback = this.sparseExecution ? undefined : (this.ownedSparseFallback = device.createBuffer({ label: "Quadtree surface sparse fallback", size: 64, usage: GPUBufferUsage.STORAGE }));
    const sparseWorklist = this.sparseExecution?.worklist ?? sparseFallback!;
    const sparseStates = this.sparseExecution?.states ?? sparseFallback!;
    const scratch = this.scratch!, predictedTexture = this.predicted!, reversedTexture = this.reversed!;
    const seedsA = this.seedsA!, seedsB = this.seedsB!, params = this.params!, passBuffer = this.passBuffer!, reductions = this.reductions!;
    const surfaceCache = this.cache!;
    const group = (phiIn: GPUTexture, phiOut: GPUTexture, seedIn: GPUBuffer, seedOut: GPUBuffer, predicted: GPUTexture = predictedTexture, reversed: GPUTexture = reversedTexture) => device.createBindGroup({ layout: surfaceCache.layout, entries: [
      { binding: 0, resource: velocity.createView() }, { binding: 1, resource: phiIn.createView() }, { binding: 2, resource: phiOut.createView() },
      { binding: 3, resource: { buffer: seedIn } }, { binding: 4, resource: { buffer: seedOut } }, { binding: 5, resource: { buffer: params } },
      { binding: 6, resource: { buffer: passBuffer, size: 16 } }, { binding: 7, resource: { buffer: reductions } },
      { binding: 8, resource: predicted.createView() }, { binding: 9, resource: reversed.createView() },
      { binding: 10, resource: reconcileTexture.createView() }, { binding: 11, resource: { buffer: surfaceSolidBuffer } },
      { binding: 12, resource: { buffer: sparseWorklist } }, { binding: 13, resource: { buffer: sparseStates } }
    ] });
    this.groups = {
      advect: group(this.texture, scratch, seedsA, seedsB),
      predict: group(this.texture, predictedTexture, seedsA, seedsB, this.texture, this.texture),
      reverse: group(predictedTexture, reversedTexture, seedsA, seedsB, predictedTexture, predictedTexture),
      correct: group(this.texture, scratch, seedsA, seedsB),
      reduce: group(this.texture, scratch, seedsA, seedsB),
      seed: group(scratch, this.texture, seedsB, seedsA),
      jumpAB: group(scratch, this.texture, seedsA, seedsB),
      jumpBA: group(scratch, this.texture, seedsB, seedsA),
      finalizeA: group(scratch, this.texture, seedsA, seedsB),
      finalizeB: group(scratch, this.texture, seedsB, seedsA),
      cull: group(this.texture, scratch, seedsA, seedsB)
    };
  }

  encode(encoder: GPUCommandEncoder, dt_s: number, inflow?: SurfaceInflowState) {
    if (this.presentationOnly) return;
    const cache = this.cache!, params = this.params!, groups = this.groups!, reductions = this.reductions!;
    const { nx, ny, nz } = this.dims;
    const parameterData = new ArrayBuffer(128);
    new Uint32Array(parameterData, 0, 4).set([nx, ny, nz, this.surfaceSequence++]);
    new Float32Array(parameterData, 16, 4).set([this.cell.x, this.cell.y, this.cell.z, dt_s]);
    // GPU correction consumes the resident reduction and commits its own
    // shift below. Never feed asynchronously mapped diagnostic state back
    // into that authoritative path; host correction/reconciliation remains
    // available to the CPU-packed quadtree reference variants.
    new Float32Array(parameterData, 32, 4).set([
      this.gpuVolumeCorrection ? 0 : this.correctionSpeed,
      !this.gpuVolumeCorrection && this.reconcileActive ? 1 : 0,
      this.debrisCulling ? 1 : 0,
      this.gpuVolumeCorrection ? 0 : this.reconcileFraction,
    ]);
    new Float32Array(parameterData, 48, 4).set([this.cell.x, this.cell.y, this.cell.z, this.volumeControlAgreeWeight]);
    new Float32Array(parameterData, 64, 4).set([this.cell.x * nx, this.cell.y * ny, this.cell.z * nz, this.sparseExecution?.brickSize ?? 0]);
    if (inflow) {
      new Float32Array(parameterData, 80, 4).set([inflow.outletCenter_m.x, inflow.outletCenter_m.y, inflow.outletCenter_m.z, inflow.radius_m]);
      new Float32Array(parameterData, 96, 4).set([inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z, inflow.apertureScale]);
    }
    new Float32Array(parameterData, 112, 4).set([inflow?.strength ?? 0, this.referenceVolumeCells, this.solidFractions ? 1 : 0, 0]);
    this.device.queue.writeBuffer(params, 0, parameterData);
    const dispatch = (pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, group: GPUBindGroup, offset = 0) => {
      pass.setPipeline(pipeline); pass.setBindGroup(0, group, [offset]);
      if (this.sparseExecution) pass.dispatchWorkgroupsIndirect(this.sparseExecution.worklist, FLUID_BRICK_ACTIVE_SURFACE_DISPATCH_OFFSET_BYTES);
      else pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4));
    };
    // Octree selects monotone RK2 semi-Lagrangian transport because applying a
    // BFECC correction directly to signed phi creates new zero crossings when
    // an impact folds a steep band. Other users retain bounded MacCormack for
    // smooth-shape accuracy. Both paths then rebuild the consumed far band.
    // Word 0 is a persistent global volume total in sparse mode. Active
    // bricks contribute signed deltas; per-frame interface/mismatch counters
    // are rebuilt from the resident band.
    if (this.sparseExecution) encoder.clearBuffer(reductions, 4, 12);
    else encoder.clearBuffer(reductions);
    // Dependent texture stages use separate passes and explicit command
    // ordering for both dense and sparse execution.
    const surfaceDispatch = (label: string, pipeline: GPUComputePipeline, group: GPUBindGroup, offset = 0) => {
      const pass = encoder.beginComputePass({ label });
      dispatch(pass, pipeline, group, offset);
      pass.end();
    };
    if (this.monotoneLevelSetTransport) {
      surfaceDispatch("Quadtree surface level-set advection", cache.pipelines.advectLevelSet, groups.advect);
    } else {
      surfaceDispatch("Quadtree surface advection predictor", cache.pipelines.advectPredict, groups.predict);
      surfaceDispatch("Quadtree surface advection reverse", cache.pipelines.advectReverse, groups.reverse);
      surfaceDispatch("Quadtree surface advection correction", cache.pipelines.advectCorrect, groups.correct);
    }
    surfaceDispatch("Quadtree surface distance seeds", cache.pipelines.seedDistance, groups.seed);
    this.jumps.forEach((_, index) => {
      surfaceDispatch(
        `Quadtree surface jump flood ${index}`,
        cache.pipelines.jumpFlood,
        index % 2 === 0 ? groups.jumpAB : groups.jumpBA,
        index * this.passStride,
      );
    });
    surfaceDispatch(
      "Quadtree surface finalize distance",
      cache.pipelines.finalizeDistance,
      this.jumps.length % 2 === 0 ? groups.finalizeA : groups.finalizeB,
    );
    if (this.debrisCulling) {
      const cullPass = encoder.beginComputePass({ label: "Quadtree surface debris cull" });
      dispatch(cullPass, cache.pipelines.cullDebris, groups.cull); cullPass.end();
      encoder.copyTextureToTexture({ texture: this.scratch! }, { texture: this.texture }, [nx, ny, nz]);
      const reductionPass = encoder.beginComputePass({ label: "Quadtree surface post-cull reduction" });
      dispatch(reductionPass, cache.pipelines.reduceVolume, groups.reduce); reductionPass.end();
    }
    // Consume the most recent phi-volume reduction entirely on the GPU. When
    // optional debris culling is enabled by another method this is its
    // post-cull volume; octree uses the ordinary redistanced phi reduction.
    if (this.gpuVolumeCorrection) {
      const correctionPass = encoder.beginComputePass({ label: "GPU level-set volume correction" });
      dispatch(correctionPass, cache.pipelines.correctLevelSetVolume, groups.advect);
      correctionPass.setPipeline(cache.pipelines.commitLevelSetVolumeCorrection);
      correctionPass.setBindGroup(0, groups.advect, [0]);
      correctionPass.dispatchWorkgroups(1);
      correctionPass.end();
      if (this.sparseExecution) {
        const copyPass = encoder.beginComputePass({ label: "Commit sparse level-set correction" });
        dispatch(copyPass, cache.pipelines.copyLevelSet, groups.seed);
        copyPass.end();
      } else encoder.copyTextureToTexture({ texture: this.scratch! }, { texture: this.texture }, [nx, ny, nz]);
    }
  }

  async readVolumeDiagnostics() {
    if (!this.reductions) return this.volumeDiagnostics;
    if (this.readbackPending) return this.volumeDiagnostics;
    this.readbackPending = true;
    const readback = this.device.createBuffer({ label: "Quadtree level-set volume readback", size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder(); encoder.copyBufferToBuffer(this.reductions, 0, readback, 0, 16); this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange()); this.volumeCells = words[0] / 256; this.interfaceCells = words[1] / 256; this.culledDebrisCells = words[2]; this.mismatchCells = words[3];
      if (this.reconcileEnabled && !this.gpuVolumeCorrection) {
        this.reconcileActive = nextQuadtreeVofReconciliationActive(this.reconcileActive, (this.volumeCells - this.referenceVolumeCells) / Math.max(1, this.referenceVolumeCells));
        this.reconcileFraction = this.reconcileActive ? quadtreeVofReconciliationFraction(this.referenceVolumeCells - this.volumeCells, this.mismatchCells) : 0;
      }
      // The smooth Heaviside is four cells wide, so its derivative converts a
      // one-cell normal shift into roughly one quarter cell of measured
      // volume. The 30 Hz readback/control loop adds roughly one sample of
      // delay, so a 1.5x lead factor prevents the dam-break transient from
      // crossing the 2% envelope while the unchanged +/-30 cells/s clamp
      // remains the hard safety bound.
      if (!this.gpuVolumeCorrection) this.correctionSpeed = Math.max(-30, Math.min(30,
        6 * (this.referenceVolumeCells - this.volumeCells) / Math.max(this.interfaceCells, 1) / (1 / 30)));
      // Volume-controller localization: once decisive phi/VOF disagreement is
      // a meaningful fraction of the interface band, agreeing cells stop
      // receiving the global normal push (weight -> 0) and the correction
      // targets the disagreement instead. Near-zero mismatch keeps weight 1,
      // i.e. exactly the legacy uniform controller.
      if (!this.gpuVolumeCorrection) this.volumeControlAgreeWeight = this.hasReconcileVolume
        ? Math.max(0, Math.min(1, 1 - 4 * this.mismatchCells / Math.max(1, this.interfaceCells)))
        : 1;
    } finally {
      if (readback.mapState === "mapped") readback.unmap(); readback.destroy(); this.readbackPending = false;
    }
    return this.volumeDiagnostics;
  }

  get volumeDiagnostics() { return { referenceVolumeCells: this.referenceVolumeCells, volumeCells: this.volumeCells, interfaceCells: this.interfaceCells, correctionSpeed: this.correctionSpeed, culledDebrisCells: this.culledDebrisCells, mismatchFraction: this.mismatchCells / Math.max(1, this.dims.nx * this.dims.ny * this.dims.nz), reconciliationActive: this.reconcileActive, volumeControlAgreeWeight: this.volumeControlAgreeWeight }; }
  addReferenceVolumeCells(cells: number) { if (Number.isFinite(cells) && cells > 0) this.referenceVolumeCells += cells; }

  /**
   * The dense field stops being an authority after global-fine publication,
   * but it cannot yet be destroyed: recurring projection, boundary, and fine
   * seed bind groups retain views of this texture even after their shaders
   * select the Section-5 fine-SPGrid/background-octree authorities. WebGPU
   * validates every bound resource, including dynamically unused bindings.
   *
   * Keep this compatibility resource alive until `destroy()` unless every
   * recurring group is atomically rebound to a replacement texture first.
   * See Aanjaneya et al. 2017 Section 5
   * (`docs/papers/aanjaneya-2017-power-liquids.txt`): the two meshes are
   * independent authorities; that assumption does not waive bind-group
   * resource lifetime.
   */
  releasePresentationTexture() {
    return 0;
  }

  destroy() {
    this.texture.destroy(); this.scratch?.destroy(); this.predicted?.destroy(); this.reversed?.destroy(); this.seedsA?.destroy(); this.seedsB?.destroy(); this.params?.destroy(); this.passBuffer?.destroy(); this.reductions?.destroy(); this.ownedReconcileFallback?.destroy(); this.ownedSolidFallback?.destroy(); this.ownedSparseFallback?.destroy();
  }
}
