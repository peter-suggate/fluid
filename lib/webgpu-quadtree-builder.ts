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
import { quadtreeSizingWeights } from "./quadtree-tall-cell-grid";

/** Per-step nozzle state for sourcing fluid into the resident level set. */
export interface SurfaceInflowState {
  outletCenter_m: { x: number; y: number; z: number };
  radius_m: number;
  velocity_m_s: { x: number; y: number; z: number };
  apertureScale: number;
  strength: number;
}

/** W0 loss-recovery circuit breaker with hysteresis. */
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

fn index3(q: vec3u) -> u32 { return q.x + params.dims.x * (q.y + params.dims.y * q.z); }
fn clamp3(q: vec3i) -> vec3i { return clamp(q, vec3i(0), vec3i(params.dims.xyz) - vec3i(1)); }
fn loadPhi(q: vec3i) -> f32 { return textureLoad(phiIn, clamp3(q), 0).x; }
fn loadVelocity(q: vec3i) -> vec3f { return textureLoad(velocityIn, clamp3(q), 0).xyz; }
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
fn volumeCorrectedPhi(value: f32) -> f32 {
  let h = hMin();
  return select(value, value - params.control.x * h * params.cellAndDt.w, abs(value) < 1.5 * h);
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
fn advectLevelSet(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  // Narita et al. Sec. 4.5: interpolate velocity from the saved previous
  // staggered grid, backtrace, then interpolate the previous level set.
  let departure = vec3f(gid) - centredMacVelocity(vec3i(gid)) * params.cellAndDt.w / params.cellAndDt.xyz;
  var phi = volumeCorrectedPhi(trilinearPhi(departure));
  // The nozzle sources fluid directly into the resident surface, exactly as
  // the restricted method's finishAdvection clamps phi at inflow cells.
  let q = vec3i(gid);
  if (isInflowVelocityCell(q)) { phi = min(phi, -0.5 * hMin() * inflowApertureFraction(q) * inflowStrength()); }
  textureStore(phiOut, q, vec4f(phi, 0.0, 0.0, 0.0));
}
@compute @workgroup_size(4, 4, 4)
fn advectPredict(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  textureStore(phiOut, vec3i(gid), vec4f(trilinearPhi(departurePoint(vec3f(gid), params.cellAndDt.w)), 0.0, 0.0, 0.0));
}
@compute @workgroup_size(4, 4, 4)
fn advectReverse(@builtin(global_invocation_id) gid: vec3u) {
  // phiIn is the predicted field; tracing it forward (negative dt backtrace)
  // recovers the BFECC error estimate.
  if (any(gid >= params.dims.xyz)) { return; }
  textureStore(phiOut, vec3i(gid), vec4f(trilinearPhi(departurePoint(vec3f(gid), -params.cellAndDt.w)), 0.0, 0.0, 0.0));
}
@compute @workgroup_size(4, 4, 4)
fn advectCorrect(@builtin(global_invocation_id) gid: vec3u) {
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
  var phi = volumeCorrectedPhi(corrected);
  if (isInflowVelocityCell(q)) { phi = min(phi, -0.5 * hMin() * inflowApertureFraction(q) * inflowStrength()); }
  textureStore(phiOut, q, vec4f(phi, 0.0, 0.0, 0.0));
}
@compute @workgroup_size(4, 4, 4)
fn reduceVolume(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  let value = loadPhi(vec3i(gid)); let cell = hMin();
  // Match the renderer and smoke oracle: adaptive phi is converted to a
  // smooth Heaviside over four vertical cell widths.
  let occupied = clamp(0.5 - value / (4.0 * params.cellAndDt.y), 0.0, 1.0);
  atomicAdd(&reductions[0], u32(occupied * 256.0));
  if (abs(value) < 1.5 * cell) { atomicAdd(&reductions[1], 256u); }
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
fn loadReconcileAlpha(q: vec3i) -> f32 { return clamp(textureLoad(reconcileVolumeIn, clamp3(q), 0).x, 0.0, 1.0); }
fn loadReconcileWet(q: vec3i) -> bool { return loadReconcileAlpha(q) >= 0.5; }
fn reconciliationSelected(gid: vec3u) -> bool {
  var hash = index3(gid) ^ (params.dims.w * 747796405u + 2891336453u);
  hash = (hash ^ (hash >> 16u)) * 2246822519u; hash = (hash ^ (hash >> 13u)) * 3266489917u; hash ^= hash >> 16u;
  return f32(hash & 0xffffu) < params.control.w * 65536.0;
}
@compute @workgroup_size(4, 4, 4)
fn seedDistance(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  let p = vec3i(gid); let wet = loadPhi(p) < 0.0;
  var crosses = (loadPhi(p + vec3i(1, 0, 0)) < 0.0) != wet || (loadPhi(p - vec3i(1, 0, 0)) < 0.0) != wet
    || (loadPhi(p + vec3i(0, 1, 0)) < 0.0) != wet || (loadPhi(p - vec3i(0, 1, 0)) < 0.0) != wet
    || (loadPhi(p + vec3i(0, 0, 1)) < 0.0) != wet || (loadPhi(p - vec3i(0, 0, 1)) < 0.0) != wet;
  // With VOF reconciliation active, a phi sign change deep inside a
  // VOF-saturated region is a fossil artifact, not a surface: seeding it
  // would anchor the narrow band around fake interfaces and shield the whole
  // neighbourhood from far-field repair. Only phi crossings with a VOF
  // wet/dry transition in the immediate neighbourhood are real interfaces.
  if (crosses && params.control.y > 0.5) {
    let wetVof = loadReconcileWet(p);
    let vofTransition = loadReconcileWet(p + vec3i(1, 0, 0)) != wetVof || loadReconcileWet(p - vec3i(1, 0, 0)) != wetVof
      || loadReconcileWet(p + vec3i(0, 1, 0)) != wetVof || loadReconcileWet(p - vec3i(0, 1, 0)) != wetVof
      || loadReconcileWet(p + vec3i(0, 0, 1)) != wetVof || loadReconcileWet(p - vec3i(0, 0, 1)) != wetVof;
    crosses = vofTransition;
  }
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
fn jumpFlood(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  var best = distanceSeedsIn[index3(gid)]; var bestDistance = seedDistanceSquared(gid, best); let jump = i32(passParams.jump);
  for (var dz = -1; dz <= 1; dz += 1) { for (var dy = -1; dy <= 1; dy += 1) { for (var dx = -1; dx <= 1; dx += 1) {
    let q = clamp(vec3i(gid) + vec3i(dx, dy, dz) * jump, vec3i(0), vec3i(params.dims.xyz) - vec3i(1));
    let candidate = distanceSeedsIn[index3(vec3u(q))]; let candidateDistance = seedDistanceSquared(gid, candidate);
    if (candidateDistance < bestDistance) { best = candidate; bestDistance = candidateDistance; }
  } } }
  distanceSeedsOut[index3(gid)] = best;
}
@compute @workgroup_size(4, 4, 4)
fn finalizeDistance(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  let advected = loadPhi(vec3i(gid));
  let h = hMin();
  var result = advected;
  // The narrow band keeps the advected phi verbatim, so redistancing never
  // moves the interface (the old +0.5h floor snapped it to the cell
  // lattice). Only the far field is rebuilt from the jump-flood distances.
  // 2.5h keeps every cell a CFL-bounded backtrace can sample from inside the
  // smoothly advected field; the jump-flood reconstruction (a point-cloud
  // distance with sub-cell tangential ripple) only ever feeds the far field.
  //
  // Band membership follows the jump-flood distance to the CURRENT interface,
  // not the advected magnitude. A cell swept by a moving interface keeps a
  // fossil near-zero value even when it is now deep inside one phase; judged
  // by |advected| alone it stays "narrow band" forever, its gradient decays
  // to ~0 (the contour can no longer be transported at the flow speed), and
  // the volume controller (|phi| < 1.5h) eventually walks its sign across
  // zero, creating fake air pockets inside the liquid that drop pressure
  // DOFs and zero velocities (the 2026-07 dam-break lateral freeze).
  let word = distanceSeedsIn[index3(gid)];
  let hasSeed = word.y <= 0xffffu;
  let interfaceDistance = select(3.402823e38, sqrt(seedDistanceSquared(gid, word)), hasSeed);
  if (abs(advected) >= 2.5 * h || interfaceDistance >= 2.5 * h) {
    var distance = 5.0 * h;
    if (hasSeed) {
      distance = min(5.0 * h, max(2.5 * h, interfaceDistance));
    }
    result = select(distance, -distance, advected < 0.0);
  }
  // GPU port of reconcileLevelSetWithVolume: the conservative VOF is the
  // transported mass field, and decisive wet/dry disagreement — liquid the
  // advected level set never saw, or dry regions it still thinks are wet by
  // more than half a cell — is overruled toward the VOF. The half-cell
  // reseed becomes a jump-flood seed on the next step, which rebuilds the
  // surrounding distances. Without this the projection's phi-based air
  // classification zeroes velocities inside VOF-wet water the level set
  // lost, which locks the two surfaces apart permanently.
  let alpha = loadReconcileAlpha(vec3i(gid));
  let wet = alpha >= 0.5;
  let signMismatch = (result < 0.0) != wet;
  let decisiveMismatch = signMismatch && abs(result) > 0.5 * h;
  if (decisiveMismatch) {
    atomicAdd(&reductions[3], 1u);
    // W0 protects against the diagnosed catastrophic path: conservative VOF
    // still contains liquid but phi has opened an interior air pocket. Do not
    // symmetrically erase phi-wet cells from the diffused VOF threshold; that
    // turned a benign interface-representation mismatch into multi-percent
    // surface-volume error. W7 still observes both directions above.
    if (params.control.y > 0.5 && wet && reconciliationSelected(gid)) {
      // Invert the same four-cell smooth Heaviside used by reduceVolume and
      // the renderer. A fixed +/-0.5h reseed assigns every repaired dry cell
      // 37.5% liquid (and every wet cell 62.5%), creating phantom represented
      // volume even though conservative VOF mass is exact.
      let volumePhi = (0.5 - alpha) * (4.0 * params.cellAndDt.y);
      result = select(max(0.02 * h, volumePhi), min(-0.02 * h, volumePhi), wet);
    }
  }
  textureStore(phiOut, vec3i(gid), vec4f(result, 0.0, 0.0, 0.0));
}
@compute @workgroup_size(4, 4, 4)
fn cullDebris(@builtin(global_invocation_id) gid: vec3u) {
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

function ensureSurfaceCache(device: GPUDevice, cache?: WebGPUQuadtreeSurfaceCache) {
  if (cache) return cache;
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
    { binding: 10, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } }
  ] });
  const shaderModule = device.createShaderModule({ label: "Resident quadtree level set", code: quadtreeSurfaceShader });
  void shaderModule.getCompilationInfo().then((info) => {
    for (const message of info.messages) if (message.type === "error") console.error(`Resident quadtree level-set WGSL ${message.lineNum}:${message.linePos} ${message.message}`);
  }).catch(() => { /* Device loss is reported by the owning solver. */ });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipeline = (entryPoint: string) => device.createComputePipeline({ label: `Quadtree surface ${entryPoint}`, layout: pipelineLayout, compute: { module: shaderModule, entryPoint } });
  return { layout, pipelineLayout, shaderModule, pipelines: {
    advectLevelSet: pipeline("advectLevelSet"), advectPredict: pipeline("advectPredict"),
    advectReverse: pipeline("advectReverse"), advectCorrect: pipeline("advectCorrect"),
    reduceVolume: pipeline("reduceVolume"),
    seedDistance: pipeline("seedDistance"), jumpFlood: pipeline("jumpFlood"), finalizeDistance: pipeline("finalizeDistance"), cullDebris: pipeline("cullDebris")
  } };
}

export class WebGPUQuadtreeSurfaceState {
  readonly cache: WebGPUQuadtreeSurfaceCache;
  readonly texture: GPUTexture;
  private readonly scratch: GPUTexture;
  private readonly predicted: GPUTexture;
  private readonly reversed: GPUTexture;
  private readonly seedsA: GPUBuffer;
  private readonly seedsB: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly passBuffer: GPUBuffer;
  private readonly reductions: GPUBuffer;
  private readonly passStride: number;
  private readonly jumps: number[];
  private readonly groups: { advect: GPUBindGroup; predict: GPUBindGroup; reverse: GPUBindGroup; correct: GPUBindGroup; reduce: GPUBindGroup; seed: GPUBindGroup; jumpAB: GPUBindGroup; jumpBA: GPUBindGroup; finalizeA: GPUBindGroup; finalizeB: GPUBindGroup; cull: GPUBindGroup };
  private readbackPending = false;
  private referenceVolumeCells: number;
  private volumeCells: number;
  private interfaceCells = 0;
  private culledDebrisCells = 0;
  private mismatchCells = 0;
  private correctionSpeed = 0;
  private readonly reconcileEnabled: boolean;
  private reconcileActive = false;
  private reconcileFraction = 0;
  private surfaceSequence = 0;
  private readonly ownedReconcileFallback?: GPUTexture;

  constructor(private readonly device: GPUDevice, private readonly dims: { nx: number; ny: number; nz: number }, private readonly cell: { x: number; y: number; z: number }, velocity: GPUTexture, initialPhi: Float32Array, cache?: WebGPUQuadtreeSurfaceCache, reconcileVolume?: GPUTexture, private readonly debrisCulling = false, reconcileEnabled = reconcileVolume !== undefined) {
    this.cache = ensureSurfaceCache(device, cache);
    this.reconcileEnabled = reconcileEnabled && reconcileVolume !== undefined;
    const textureUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC;
    this.texture = device.createTexture({ label: "Resident quadtree level set", size: [dims.nx, dims.ny, dims.nz], dimension: "3d", format: "r32float", usage: textureUsage });
    this.scratch = device.createTexture({ label: "Resident quadtree level-set advection scratch", size: [dims.nx, dims.ny, dims.nz], dimension: "3d", format: "r32float", usage: textureUsage });
    this.predicted = device.createTexture({ label: "Resident quadtree level-set MacCormack prediction", size: [dims.nx, dims.ny, dims.nz], dimension: "3d", format: "r32float", usage: textureUsage });
    this.reversed = device.createTexture({ label: "Resident quadtree level-set MacCormack reversal", size: [dims.nx, dims.ny, dims.nz], dimension: "3d", format: "r32float", usage: textureUsage });
    uploadLevelSetTexture(device, this.texture, initialPhi, dims.nx, dims.ny, dims.nz);
    const bytes = Math.max(8, dims.nx * dims.ny * dims.nz * 8), seedUsage = GPUBufferUsage.STORAGE;
    this.seedsA = device.createBuffer({ label: "Quadtree surface seeds A", size: bytes, usage: seedUsage });
    this.seedsB = device.createBuffer({ label: "Quadtree surface seeds B", size: bytes, usage: seedUsage });
    this.params = device.createBuffer({ label: "Quadtree surface parameters", size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.reductions = device.createBuffer({ label: "Quadtree level-set volume diagnostics", size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const volumeBand = 4 * cell.y;
    this.referenceVolumeCells = initialPhi.reduce((sum, value) => sum + Math.max(0, Math.min(1, 0.5 - value / volumeBand)), 0);
    this.volumeCells = this.referenceVolumeCells;
    this.interfaceCells = initialPhi.reduce((sum, value) => sum + (Math.abs(value) < 1.5 * Math.min(cell.x, cell.y, cell.z) ? 1 : 0), 0);
    device.queue.writeBuffer(this.reductions, 0, new Uint32Array([Math.round(this.volumeCells * 256), Math.round(this.interfaceCells * 256), 0, 0]));
    this.jumps = [];
    for (let jump = largestPowerOfTwoAtMost(Math.max(dims.nx, dims.ny, dims.nz)); jump >= 1; jump /= 2) this.jumps.push(jump);
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
    const group = (phiIn: GPUTexture, phiOut: GPUTexture, seedIn: GPUBuffer, seedOut: GPUBuffer, predicted: GPUTexture = this.predicted, reversed: GPUTexture = this.reversed) => device.createBindGroup({ layout: this.cache.layout, entries: [
      { binding: 0, resource: velocity.createView() }, { binding: 1, resource: phiIn.createView() }, { binding: 2, resource: phiOut.createView() },
      { binding: 3, resource: { buffer: seedIn } }, { binding: 4, resource: { buffer: seedOut } }, { binding: 5, resource: { buffer: this.params } },
      { binding: 6, resource: { buffer: this.passBuffer, size: 16 } }, { binding: 7, resource: { buffer: this.reductions } },
      { binding: 8, resource: predicted.createView() }, { binding: 9, resource: reversed.createView() },
      { binding: 10, resource: reconcileTexture.createView() }
    ] });
    this.groups = {
      advect: group(this.texture, this.scratch, this.seedsA, this.seedsB),
      predict: group(this.texture, this.predicted, this.seedsA, this.seedsB, this.texture, this.texture),
      reverse: group(this.predicted, this.reversed, this.seedsA, this.seedsB, this.predicted, this.predicted),
      correct: group(this.texture, this.scratch, this.seedsA, this.seedsB),
      reduce: group(this.texture, this.scratch, this.seedsA, this.seedsB),
      seed: group(this.scratch, this.texture, this.seedsB, this.seedsA),
      jumpAB: group(this.scratch, this.texture, this.seedsA, this.seedsB),
      jumpBA: group(this.scratch, this.texture, this.seedsB, this.seedsA),
      finalizeA: group(this.scratch, this.texture, this.seedsA, this.seedsB),
      finalizeB: group(this.scratch, this.texture, this.seedsB, this.seedsA),
      cull: group(this.texture, this.scratch, this.seedsA, this.seedsB)
    };
  }

  encode(encoder: GPUCommandEncoder, dt_s: number, inflow?: SurfaceInflowState) {
    const { nx, ny, nz } = this.dims;
    const parameterData = new ArrayBuffer(128);
    new Uint32Array(parameterData, 0, 4).set([nx, ny, nz, this.surfaceSequence++]);
    new Float32Array(parameterData, 16, 4).set([this.cell.x, this.cell.y, this.cell.z, dt_s]);
    new Float32Array(parameterData, 32, 4).set([this.correctionSpeed, this.reconcileActive ? 1 : 0, this.debrisCulling ? 1 : 0, this.reconcileFraction]);
    new Float32Array(parameterData, 48, 4).set([this.cell.x, this.cell.y, this.cell.z, 0]);
    new Float32Array(parameterData, 64, 4).set([this.cell.x * nx, this.cell.y * ny, this.cell.z * nz, 0]);
    if (inflow) {
      new Float32Array(parameterData, 80, 4).set([inflow.outletCenter_m.x, inflow.outletCenter_m.y, inflow.outletCenter_m.z, inflow.radius_m]);
      new Float32Array(parameterData, 96, 4).set([inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z, inflow.apertureScale]);
      new Float32Array(parameterData, 112, 4).set([inflow.strength, 0, 0, 0]);
    }
    this.device.queue.writeBuffer(this.params, 0, parameterData);
    const dispatch = (pipeline: GPUComputePipeline, group: GPUBindGroup, offset = 0) => {
      const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group, [offset]);
      pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4)); pass.end();
    };
    // Bounded-MacCormack transport (Narita Sec. 4.5 with the restricted
    // method's BFECC correction), then a full sub-cell-preserving jump-flood
    // redistance every step so downstream consumers (sizing, segmentation,
    // Eq. 25 free-surface scale) always see a true signed-distance field.
    encoder.clearBuffer(this.reductions);
    dispatch(this.cache.pipelines.advectPredict, this.groups.predict);
    dispatch(this.cache.pipelines.advectReverse, this.groups.reverse);
    dispatch(this.cache.pipelines.advectCorrect, this.groups.correct);
    dispatch(this.cache.pipelines.seedDistance, this.groups.seed);
    this.jumps.forEach((_, index) => {
      dispatch(this.cache.pipelines.jumpFlood, index % 2 === 0 ? this.groups.jumpAB : this.groups.jumpBA, index * this.passStride);
    });
    dispatch(this.cache.pipelines.finalizeDistance, this.jumps.length % 2 === 0 ? this.groups.finalizeA : this.groups.finalizeB);
    if (this.debrisCulling) {
      dispatch(this.cache.pipelines.cullDebris, this.groups.cull);
      encoder.copyTextureToTexture({ texture: this.scratch }, { texture: this.texture }, [nx, ny, nz]);
    }
    dispatch(this.cache.pipelines.reduceVolume, this.groups.reduce);
  }

  async readVolumeDiagnostics() {
    if (this.readbackPending) return this.volumeDiagnostics;
    this.readbackPending = true;
    const readback = this.device.createBuffer({ label: "Quadtree level-set volume readback", size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder(); encoder.copyBufferToBuffer(this.reductions, 0, readback, 0, 16); this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange()); this.volumeCells = words[0] / 256; this.interfaceCells = words[1] / 256; this.culledDebrisCells = words[2]; this.mismatchCells = words[3];
      if (this.reconcileEnabled) {
        this.reconcileActive = nextQuadtreeVofReconciliationActive(this.reconcileActive, (this.volumeCells - this.referenceVolumeCells) / Math.max(1, this.referenceVolumeCells));
        this.reconcileFraction = this.reconcileActive ? quadtreeVofReconciliationFraction(this.referenceVolumeCells - this.volumeCells, this.mismatchCells) : 0;
      }
      // The smooth Heaviside is four cells wide, so its derivative converts a
      // one-cell normal shift into roughly one quarter cell of measured
      // volume. The 30 Hz readback/control loop adds roughly one sample of
      // delay, so a 1.5x lead factor prevents the dam-break transient from
      // crossing the 2% envelope while the unchanged +/-30 cells/s clamp
      // remains the hard safety bound.
      this.correctionSpeed = Math.max(-30, Math.min(30, 6 * (this.referenceVolumeCells - this.volumeCells) / Math.max(this.interfaceCells, 1) / (1 / 30)));
    } finally {
      if (readback.mapState === "mapped") readback.unmap(); readback.destroy(); this.readbackPending = false;
    }
    return this.volumeDiagnostics;
  }

  get volumeDiagnostics() { return { referenceVolumeCells: this.referenceVolumeCells, volumeCells: this.volumeCells, interfaceCells: this.interfaceCells, correctionSpeed: this.correctionSpeed, culledDebrisCells: this.culledDebrisCells, mismatchFraction: this.mismatchCells / Math.max(1, this.dims.nx * this.dims.ny * this.dims.nz), reconciliationActive: this.reconcileActive }; }
  addReferenceVolumeCells(cells: number) { if (Number.isFinite(cells) && cells > 0) this.referenceVolumeCells += cells; }

  destroy() {
    this.texture.destroy(); this.scratch.destroy(); this.predicted.destroy(); this.reversed.destroy(); this.seedsA.destroy(); this.seedsB.destroy(); this.params.destroy(); this.passBuffer.destroy(); this.reductions.destroy(); this.ownedReconcileFallback?.destroy();
  }
}

export interface WebGPUQuadtreeConstructionCache {
  layout: GPUBindGroupLayout;
  pipelineLayout: GPUPipelineLayout;
  shaderModule: GPUShaderModule;
  pipelines: {
    advectLevelSet: GPUComputePipeline;
    seedDistance: GPUComputePipeline;
    jumpFlood: GPUComputePipeline;
    finalizeDistance: GPUComputePipeline;
    evaluateSizing: GPUComputePipeline;
    refine: GPUComputePipeline;
    smoothTopology: GPUComputePipeline;
    sampleLeafProfiles: GPUComputePipeline;
  };
  workspace?: {
    key: string;
    topologyA: GPUBuffer; topologyB: GPUBuffer; sizing: GPUBuffer; explicit: GPUBuffer;
    staticParams: GPUBuffer; levelSetOut: GPUBuffer; distanceSeedA: GPUBuffer; distanceSeedB: GPUBuffer;
    passBuffer: GPUBuffer; readback: GPUBuffer;
    columnProfiles: GPUBuffer; profileReadback: GPUBuffer;
    querySet?: GPUQuerySet; queryResolve?: GPUBuffer;
  };
}

export interface GPUQuadtreeBuildResult {
  /** Leaf-major [centre, footprint min, footprint max] phi profiles; empty for the resident GPU sparse-pack path. */
  columnProfiles: Float32Array;
  packedCells: Uint32Array;
  diagnostics: Float32Array;
  mismatchFraction: number;
  /** Timestamp-query duration of only the adaptive construction kernels. */
  gpuKernel_ms?: number;
  /** Submit-to-map wall time, including older work already queued on the device. */
  gpuWall_ms: number;
}

export const quadtreeConstructionShader = /* wgsl */ `
struct StaticParams { dims: vec4u, cellAndDt: vec4f, sizing: vec4f }
struct PassParams { values: vec4u }
@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var volumeIn: texture_3d<f32>;
@group(0) @binding(2) var phiIn: texture_3d<f32>;
@group(0) @binding(3) var<storage, read_write> phiAdvanced: array<f32>;
@group(0) @binding(4) var<storage, read> distanceSeedsIn: array<u32>;
@group(0) @binding(5) var<storage, read> topologyIn: array<u32>;
@group(0) @binding(6) var<storage, read_write> topologyOut: array<u32>;
@group(0) @binding(7) var<uniform> params: StaticParams;
@group(0) @binding(8) var<storage, read> explicitSizing: array<f32>;
@group(0) @binding(9) var<uniform> passParams: PassParams;
@group(0) @binding(10) var<storage, read_write> sizingField: array<f32>;
@group(0) @binding(11) var<storage, read_write> distanceSeedsOut: array<u32>;
@group(0) @binding(13) var<storage, read_write> columnProfiles: array<f32>;

fn index2(q: vec2u) -> u32 { return q.x + params.dims.x * q.y; }
fn index3(q: vec3u) -> u32 { return q.x + params.dims.x * (q.y + params.dims.y * q.z); }
fn clamp3(q: vec3i) -> vec3i { return clamp(q, vec3i(0), vec3i(params.dims.xyz) - vec3i(1)); }
fn loadInputPhi(q: vec3i) -> f32 { return textureLoad(phiIn, clamp3(q), 0).x; }
fn loadAdvancedPhi(q: vec3i) -> f32 { return phiAdvanced[index3(vec3u(clamp3(q)))]; }
fn loadVelocity(q: vec3i) -> vec3f { return textureLoad(velocityIn, clamp3(q), 0).xyz; }
fn loadVolume(q: vec3i) -> f32 { return textureLoad(volumeIn, clamp3(q), 0).x; }
fn effectiveWet(q: vec3i) -> bool {
  return loadAdvancedPhi(clamp3(q)) < 0.0;
}
fn trilinearInput(position: vec3f) -> f32 {
  let hi = vec3f(params.dims.xyz - vec3u(1));
  let p = clamp(position, vec3f(0.0), hi);
  let a = vec3i(floor(p)); let b = min(a + vec3i(1), vec3i(params.dims.xyz) - vec3i(1)); let t = fract(p);
  let x00 = mix(loadInputPhi(vec3i(a.x, a.y, a.z)), loadInputPhi(vec3i(b.x, a.y, a.z)), t.x);
  let x10 = mix(loadInputPhi(vec3i(a.x, b.y, a.z)), loadInputPhi(vec3i(b.x, b.y, a.z)), t.x);
  let x01 = mix(loadInputPhi(vec3i(a.x, a.y, b.z)), loadInputPhi(vec3i(b.x, a.y, b.z)), t.x);
  let x11 = mix(loadInputPhi(vec3i(a.x, b.y, b.z)), loadInputPhi(vec3i(b.x, b.y, b.z)), t.x);
  return mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
}

@compute @workgroup_size(4, 4, 4)
fn advectLevelSet(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  let q = vec3i(gid); let own = loadVelocity(q);
  let velocity = 0.5 * vec3f(
    own.x + loadVelocity(q - vec3i(1, 0, 0)).x,
    own.y + loadVelocity(q - vec3i(0, 1, 0)).y,
    own.z + loadVelocity(q - vec3i(0, 0, 1)).z
  );
  let cellsPerMetre = vec3f(1.0) / params.cellAndDt.xyz;
  let departure = vec3f(gid) - velocity * params.cellAndDt.w * cellsPerMetre;
  phiAdvanced[index3(gid)] = trilinearInput(departure);
}

fn packSeed(q: vec3u) -> u32 { return q.x | (q.y << 10u) | (q.z << 20u); }
fn unpackSeed(word: u32) -> vec3u { return vec3u(word & 1023u, (word >> 10u) & 1023u, (word >> 20u) & 1023u); }
@compute @workgroup_size(4, 4, 4)
fn seedDistance(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  let p = vec3i(gid); let wet = effectiveWet(p); var isInterface = effectiveWet(p + vec3i(1, 0, 0)) != wet || effectiveWet(p - vec3i(1, 0, 0)) != wet
    || effectiveWet(p + vec3i(0, 1, 0)) != wet || effectiveWet(p - vec3i(0, 1, 0)) != wet
    || effectiveWet(p + vec3i(0, 0, 1)) != wet || effectiveWet(p - vec3i(0, 0, 1)) != wet;
  distanceSeedsOut[index3(gid)] = select(0xffffffffu, packSeed(gid), isInterface);
}
fn seedDistanceSquared(cell: vec3u, word: u32) -> f32 {
  if (word == 0xffffffffu) { return 3.402823e38; }
  let delta = (vec3f(cell) - vec3f(unpackSeed(word))) * params.cellAndDt.xyz;
  return dot(delta, delta);
}
@compute @workgroup_size(4, 4, 4)
fn jumpFlood(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  var best = distanceSeedsIn[index3(gid)]; var bestDistance = seedDistanceSquared(gid, best); let jump = i32(passParams.values.w);
  for (var dz = -1; dz <= 1; dz += 1) { for (var dy = -1; dy <= 1; dy += 1) { for (var dx = -1; dx <= 1; dx += 1) {
    let q = clamp(vec3i(gid) + vec3i(dx, dy, dz) * jump, vec3i(0), vec3i(params.dims.xyz) - vec3i(1));
    let candidate = distanceSeedsIn[index3(vec3u(q))]; let candidateDistance = seedDistanceSquared(gid, candidate);
    if (candidateDistance < bestDistance) { best = candidate; bestDistance = candidateDistance; }
  } } }
  distanceSeedsOut[index3(gid)] = best;
}
@compute @workgroup_size(4, 4, 4)
fn finalizeDistance(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  let word = distanceSeedsIn[index3(gid)]; let hMin = min(params.cellAndDt.x, min(params.cellAndDt.y, params.cellAndDt.z)); var distance = 5.0 * hMin;
  if (word != 0xffffffffu) {
    distance = min(5.0 * hMin, sqrt(seedDistanceSquared(gid, word)) + 0.5 * hMin);
  }
  phiAdvanced[index3(gid)] = select(distance, -distance, effectiveWet(vec3i(gid)));
}

fn strainMagnitude(q: vec3i) -> f32 {
  let h = params.cellAndDt.xyz;
  let vx = (loadVelocity(q + vec3i(1, 0, 0)).x - loadVelocity(q - vec3i(1, 0, 0)).x) / (2.0 * h.x);
  let vy = (loadVelocity(q + vec3i(0, 1, 0)).y - loadVelocity(q - vec3i(0, 1, 0)).y) / (2.0 * h.y);
  let vz = (loadVelocity(q + vec3i(0, 0, 1)).z - loadVelocity(q - vec3i(0, 0, 1)).z) / (2.0 * h.z);
  return length(vec3f(vx, vy, vz));
}

fn speedGradientMagnitude(q: vec3i) -> f32 {
  let h = params.cellAndDt.xyz;
  let dx = (length(loadVelocity(q + vec3i(1, 0, 0))) - length(loadVelocity(q - vec3i(1, 0, 0)))) / (2.0 * h.x);
  let dy = (length(loadVelocity(q + vec3i(0, 1, 0))) - length(loadVelocity(q - vec3i(0, 1, 0)))) / (2.0 * h.y);
  let dz = (length(loadVelocity(q + vec3i(0, 0, 1))) - length(loadVelocity(q - vec3i(0, 0, 1)))) / (2.0 * h.z);
  return length(vec3f(dx, dy, dz));
}

@compute @workgroup_size(8, 8)
fn evaluateSizing(@builtin(global_invocation_id) gid: vec3u) {
  let q = gid.xy; if (any(q >= params.dims.xz)) { return; }
  let hx = params.cellAndDt.x; let hy = params.cellAndDt.y; let hz = params.cellAndDt.z;
  let band = 2.0 * max(hx, max(hy, hz)); var maximum = explicitSizing[index2(q)];
  for (var y = 0u; y < params.dims.y; y += 1u) {
    let p = vec3i(i32(q.x), i32(y), i32(q.y)); let phi = loadAdvancedPhi(p); let wet = phi < 0.0;
    let crosses = effectiveWet(p + vec3i(1, 0, 0)) != wet || effectiveWet(p - vec3i(1, 0, 0)) != wet
      || effectiveWet(p + vec3i(0, 1, 0)) != wet || effectiveWet(p - vec3i(0, 1, 0)) != wet
      || effectiveWet(p + vec3i(0, 0, 1)) != wet || effectiveWet(p - vec3i(0, 0, 1)) != wet;
    if (abs(phi) > band && !crosses) { continue; }
    let laplacian = (loadAdvancedPhi(p + vec3i(1, 0, 0)) - 2.0 * phi + loadAdvancedPhi(p - vec3i(1, 0, 0))) / (hx * hx)
      + (loadAdvancedPhi(p + vec3i(0, 1, 0)) - 2.0 * phi + loadAdvancedPhi(p - vec3i(0, 1, 0))) / (hy * hy)
      + (loadAdvancedPhi(p + vec3i(0, 0, 1)) - 2.0 * phi + loadAdvancedPhi(p - vec3i(0, 0, 1))) / (hz * hz);
    let frontSpeedDemand = ${quadtreeSizingWeights.frontSpeed.toFixed(1)} * length(loadVelocity(p)) / min(hx, min(hy, hz));
    maximum = max(maximum, params.sizing.x * abs(laplacian) + params.sizing.y * strainMagnitude(p) + ${quadtreeSizingWeights.speedGradient.toFixed(1)} * speedGradientMagnitude(p) + frontSpeedDemand);
  }
  sizingField[index2(q)] = maximum;
}

fn leafOrigin(word: u32) -> vec2u { return vec2u(word & 1023u, (word >> 10u) & 1023u); }
fn leafSize(word: u32) -> u32 { return (word >> 20u) & 1023u; }
fn packLeaf(origin: vec2u, size: u32) -> u32 { return origin.x | (origin.y << 10u) | (size << 20u); }
fn childForCell(origin: vec2u, size: u32, cell: vec2u) -> u32 {
  let child = size / 2u; var childOrigin = origin;
  if (cell.x >= origin.x + child) { childOrigin.x += child; }
  if (cell.y >= origin.y + child) { childOrigin.y += child; }
  return packLeaf(childOrigin, child);
}

@compute @workgroup_size(8, 8)
fn refine(@builtin(global_invocation_id) gid: vec3u) {
  let q = gid.xy; if (any(q >= params.dims.xz)) { return; }
  let index = index2(q); let word = topologyIn[index]; let size = leafSize(word);
  if (size != passParams.values.x || size <= 1u) { topologyOut[index] = word; return; }
  let origin = leafOrigin(word); var demand = 0.0;
  for (var z = origin.y; z < origin.y + size; z += 1u) { for (var x = origin.x; x < origin.x + size; x += 1u) { demand = max(demand, sizingField[index2(vec2u(x, z))]); } }
  let h = params.sizing.z; let alpha = params.sizing.w; let width = f32(size) * h; var testedWidth = width;
  if (alpha <= 0.0) { testedWidth = h; }
  else if (alpha < 1.0) { testedWidth = exp2(log2(width / h) / log2(1.0 + alpha)) * h; }
  topologyOut[index] = select(word, childForCell(origin, size, q), demand > 1.0 / testedWidth);
}

fn neighborTooFine(origin: vec2u, size: u32, dilation: bool) -> bool {
  for (var offset = 0u; offset < size; offset += 1u) {
    if (origin.x > 0u) { let other = leafSize(topologyIn[index2(vec2u(origin.x - 1u, origin.y + offset))]); if (select(other * 2u < size, other < size, dilation)) { return true; } }
    if (origin.x + size < params.dims.x) { let other = leafSize(topologyIn[index2(vec2u(origin.x + size, origin.y + offset))]); if (select(other * 2u < size, other < size, dilation)) { return true; } }
    if (origin.y > 0u) { let other = leafSize(topologyIn[index2(vec2u(origin.x + offset, origin.y - 1u))]); if (select(other * 2u < size, other < size, dilation)) { return true; } }
    if (origin.y + size < params.dims.z) { let other = leafSize(topologyIn[index2(vec2u(origin.x + offset, origin.y + size))]); if (select(other * 2u < size, other < size, dilation)) { return true; } }
  }
  return false;
}

@compute @workgroup_size(8, 8)
fn smoothTopology(@builtin(global_invocation_id) gid: vec3u) {
  let q = gid.xy; if (any(q >= params.dims.xz)) { return; }
  let index = index2(q); let word = topologyIn[index]; let size = leafSize(word);
  if (size <= 1u) { topologyOut[index] = word; return; }
  let origin = leafOrigin(word); let split = neighborTooFine(origin, size, passParams.values.z != 0u);
  topologyOut[index] = select(word, childForCell(origin, size, q), split);
}

@compute @workgroup_size(64)
fn sampleLeafProfiles(@builtin(global_invocation_id) gid: vec3u) {
  let leaf = gid.x; if (leaf >= passParams.values.x) { return; }
  let word = bitcast<u32>(explicitSizing[leaf]);
  let origin = leafOrigin(word); let size = leafSize(word);
  let position = vec2f(origin) + vec2f(f32(size) * 0.5 - 0.5);
  let a = vec2u(floor(position)); let b = min(a + vec2u(1), params.dims.xz - vec2u(1)); let t = fract(position);
  for (var y = 0u; y < params.dims.y; y += 1u) {
    let p00 = loadInputPhi(vec3i(i32(a.x), i32(y), i32(a.y)));
    let p10 = loadInputPhi(vec3i(i32(b.x), i32(y), i32(a.y)));
    let p01 = loadInputPhi(vec3i(i32(a.x), i32(y), i32(b.y)));
    let p11 = loadInputPhi(vec3i(i32(b.x), i32(y), i32(b.y)));
    let profile = 3u * (leaf * params.dims.y + y);
    columnProfiles[profile] = mix(mix(p00, p10, t.x), mix(p01, p11, t.x), t.y);
    var minimum = 3.402823e38; var maximum = -3.402823e38;
    for (var z = origin.y; z < origin.y + size; z += 1u) { for (var x = origin.x; x < origin.x + size; x += 1u) {
      let value = loadInputPhi(vec3i(i32(x), i32(y), i32(z)));
      minimum = min(minimum, value); maximum = max(maximum, value);
    } }
    columnProfiles[profile + 1u] = minimum; columnProfiles[profile + 2u] = maximum;
  }
}
`;

function largestPowerOfTwoAtMost(value: number) {
  let result = 1;
  while (result * 2 <= value) result *= 2;
  return result;
}

/** Same dyadic forest tiling used by the CPU reference builder. */
export function packedQuadtreeRootMap(nx: number, nz: number, maximumLeafSize: number) {
  if (nx <= 0 || nz <= 0 || nx > 1023 || nz > 1023) throw new Error("Unsupported GPU quadtree dimensions");
  const output = new Uint32Array(nx * nz);
  const maximumRootSize = largestPowerOfTwoAtMost(Math.max(1, Math.round(maximumLeafSize)));
  const tile = (x: number, z: number, width: number, depth: number) => {
    if (width <= 0 || depth <= 0) return;
    const size = Math.min(maximumRootSize, largestPowerOfTwoAtMost(Math.min(width, depth)));
    const word = x | (z << 10) | (size << 20);
    for (let zz = z; zz < z + size; zz += 1) for (let xx = x; xx < x + size; xx += 1) output[xx + nx * zz] = word;
    tile(x + size, z, width - size, size);
    tile(x, z + size, width, depth - size);
  };
  tile(0, 0, nx, nz);
  return output;
}

function ensureCache(device: GPUDevice, cache?: WebGPUQuadtreeConstructionCache) {
  if (cache) return cache;
  const layout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
    { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
  ] });
  const shaderModule = device.createShaderModule({ label: "GPU quadtree construction", code: quadtreeConstructionShader });
  void shaderModule.getCompilationInfo().then((info) => {
    for (const message of info.messages) if (message.type === "error") console.error(`GPU quadtree WGSL ${message.lineNum}:${message.linePos} ${message.message}`);
  }).catch(() => { /* Device loss is reported by the owning solver. */ });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipeline = (entryPoint: string) => device.createComputePipeline({ label: `GPU quadtree ${entryPoint}`, layout: pipelineLayout, compute: { module: shaderModule, entryPoint } });
  return { layout, shaderModule, pipelineLayout, pipelines: { advectLevelSet: pipeline("advectLevelSet"), seedDistance: pipeline("seedDistance"), jumpFlood: pipeline("jumpFlood"), finalizeDistance: pipeline("finalizeDistance"), evaluateSizing: pipeline("evaluateSizing"), refine: pipeline("refine"), smoothTopology: pipeline("smoothTopology"), sampleLeafProfiles: pipeline("sampleLeafProfiles") } };
}

export class WebGPUQuadtreeBuilder {
  readonly cache: WebGPUQuadtreeConstructionCache;
  private readonly rootMap: Uint32Array;

  constructor(
    private readonly device: GPUDevice,
    private readonly dims: { nx: number; ny: number; nz: number },
    private readonly cell: { x: number; y: number; z: number },
    private readonly maximumLeafSize: number,
    private readonly adaptivityStrength: number,
    private readonly smoothingDilations = 3,
    cache?: WebGPUQuadtreeConstructionCache
  ) {
    this.cache = ensureCache(device, cache);
    this.rootMap = packedQuadtreeRootMap(dims.nx, dims.nz, maximumLeafSize);
  }

  async build(inputs: {
    velocity: GPUTexture;
    volume: GPUTexture;
    levelSet: GPUTexture;
    explicitSizing: Float32Array;
    diagnosticBuffer: GPUBuffer;
    diagnosticBytes: number;
    /** Skip the CPU-reference leaf profiles when the GPU sparse pack consumes the owner map directly. */
    readLeafProfiles?: boolean;
  }): Promise<GPUQuadtreeBuildResult> {
    const { nx, ny, nz } = this.dims, cellCount2 = nx * nz;
    if (inputs.explicitSizing.length !== cellCount2) throw new Error("Invalid explicit GPU sizing field");
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const maxRoot = largestPowerOfTwoAtMost(Math.min(Math.max(1, Math.round(this.maximumLeafSize)), nx, nz));
    const balancePasses = Math.max(2, Math.ceil(Math.log2(maxRoot)) + 1);
    const operations: Array<{ entry: "refine" | "smoothTopology"; size: number; mode: number }> = [];
    for (let size = maxRoot; size > 1; size /= 2) operations.push({ entry: "refine", size, mode: 0 });
    for (let pass = 0; pass < balancePasses; pass += 1) operations.push({ entry: "smoothTopology", size: 0, mode: 0 });
    for (let dilation = 0; dilation < this.smoothingDilations; dilation += 1) {
      operations.push({ entry: "smoothTopology", size: 0, mode: 1 });
      for (let pass = 0; pass < balancePasses; pass += 1) operations.push({ entry: "smoothTopology", size: 0, mode: 0 });
    }
    const jumps: number[] = [];
    for (let jump = largestPowerOfTwoAtMost(Math.max(nx, ny, nz)); jump >= 1; jump /= 2) jumps.push(jump);
    const alignment = this.device.limits.minUniformBufferOffsetAlignment;
    const passStride = Math.ceil(16 / alignment) * alignment;
    const passData = new Uint8Array(Math.max(passStride, passStride * (operations.length + jumps.length)));
    operations.forEach((operation, index) => new Uint32Array(passData.buffer, index * passStride, 4).set([operation.size, maxRoot, operation.mode, 0]));
    jumps.forEach((jump, index) => new Uint32Array(passData.buffer, (operations.length + index) * passStride, 4).set([0, 0, 0, jump]));
    const align = (value: number, alignment: number) => Math.ceil(value / alignment) * alignment;
    const phiBytes = nx * ny * nz * 4, profileCapacityBytes = 3 * phiBytes, topologyBytes = cellCount2 * 4;
    const topologyOffset = 0;
    const diagnosticOffset = align(topologyOffset + topologyBytes, 8);
    const queryOffset = align(diagnosticOffset + inputs.diagnosticBytes, 8);
    const hasTimestamps = this.device.features.has("timestamp-query");
    const readbackBytes = queryOffset + (hasTimestamps ? 16 : 0);
    const workspaceKey = `${nx}x${ny}x${nz}:${passData.byteLength}:${readbackBytes}`;
    if (this.cache.workspace?.key !== workspaceKey) {
      WebGPUQuadtreeBuilder.destroyWorkspace(this.cache.workspace);
      this.cache.workspace = {
        key: workspaceKey,
        topologyA: this.device.createBuffer({ label: "GPU quadtree topology A", size: Math.max(4, topologyBytes), usage }),
        topologyB: this.device.createBuffer({ label: "GPU quadtree topology B", size: Math.max(4, topologyBytes), usage }),
        sizing: this.device.createBuffer({ label: "GPU quadtree sizing", size: Math.max(4, topologyBytes), usage }),
        explicit: this.device.createBuffer({ label: "GPU quadtree explicit sizing", size: Math.max(4, topologyBytes), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
        staticParams: this.device.createBuffer({ label: "GPU quadtree parameters", size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
        levelSetOut: this.device.createBuffer({ label: "GPU quadtree level-set readback source", size: Math.max(4, phiBytes), usage }),
        distanceSeedA: this.device.createBuffer({ label: "GPU quadtree binding scratch A", size: 4, usage }),
        distanceSeedB: this.device.createBuffer({ label: "GPU quadtree binding scratch B", size: 4, usage }),
        passBuffer: this.device.createBuffer({ label: "GPU quadtree pass parameters", size: passData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
        readback: this.device.createBuffer({ label: "GPU quadtree compact readback", size: Math.max(8, readbackBytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
        columnProfiles: this.device.createBuffer({ label: "GPU quadtree conservative phi profiles", size: Math.max(4, profileCapacityBytes), usage }),
        profileReadback: this.device.createBuffer({ label: "GPU quadtree phi-profile readback", size: Math.max(4, profileCapacityBytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
        querySet: hasTimestamps ? this.device.createQuerySet({ type: "timestamp", count: 2 }) : undefined,
        queryResolve: hasTimestamps ? this.device.createBuffer({ label: "GPU quadtree timestamp resolve", size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : undefined
      };
    }
    const { topologyA, topologyB, sizing, explicit, staticParams, levelSetOut, distanceSeedA, distanceSeedB, passBuffer, readback, columnProfiles, profileReadback, querySet, queryResolve } = this.cache.workspace!;
    this.device.queue.writeBuffer(topologyA, 0, this.rootMap.buffer as ArrayBuffer, this.rootMap.byteOffset, this.rootMap.byteLength);
    this.device.queue.writeBuffer(explicit, 0, inputs.explicitSizing.buffer as ArrayBuffer, inputs.explicitSizing.byteOffset, inputs.explicitSizing.byteLength);
    const staticData = new ArrayBuffer(48);
    new Uint32Array(staticData, 0, 4).set([nx, ny, nz, 0]);
    // Construction consumes the already-advanced resident level set. A zero-dt
    // trace only supplies the dense storage buffer used by sizing; no
    // accumulated-time advection occurs during rebuilds.
    new Float32Array(staticData, 16, 4).set([this.cell.x, this.cell.y, this.cell.z, 0]);
    new Float32Array(staticData, 32, 4).set([quadtreeSizingWeights.curvature, quadtreeSizingWeights.strain, Math.min(this.cell.x, this.cell.z), Math.max(0, Math.min(1, this.adaptivityStrength))]);
    this.device.queue.writeBuffer(staticParams, 0, staticData);
    this.device.queue.writeBuffer(passBuffer, 0, passData);

    const group = (input: GPUBuffer, output: GPUBuffer, seedInput: GPUBuffer, seedOutput: GPUBuffer) => this.device.createBindGroup({ layout: this.cache.layout, entries: [
      { binding: 0, resource: inputs.velocity.createView() }, { binding: 1, resource: inputs.volume.createView() },
      { binding: 2, resource: inputs.levelSet.createView() }, { binding: 3, resource: { buffer: levelSetOut } },
      { binding: 4, resource: { buffer: seedInput } },
      { binding: 5, resource: { buffer: input } }, { binding: 6, resource: { buffer: output } },
      { binding: 7, resource: { buffer: staticParams } }, { binding: 8, resource: { buffer: explicit } },
      { binding: 9, resource: { buffer: passBuffer, size: 16 } }, { binding: 10, resource: { buffer: sizing } },
      { binding: 11, resource: { buffer: seedOutput } },
      { binding: 13, resource: { buffer: columnProfiles } }
    ] });
    const groupAB = group(topologyA, topologyB, distanceSeedA, distanceSeedB), groupBA = group(topologyB, topologyA, distanceSeedA, distanceSeedB);
    // First read back the compact leaf-owner map. Once decoded, a second tiny
    // GPU pass samples only one vertical phi profile per unique leaf instead
    // of transferring the complete dense 3D level set.
    const encoder = this.device.createCommandEncoder({ label: "GPU quadtree construction" });
    {
      const pass = encoder.beginComputePass(querySet ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0 } } : undefined); pass.setPipeline(this.cache.pipelines.advectLevelSet); pass.setBindGroup(0, groupAB, [0]);
      pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4)); pass.end();
    }
    {
      const pass = encoder.beginComputePass(); pass.setPipeline(this.cache.pipelines.evaluateSizing); pass.setBindGroup(0, groupAB, [0]);
      pass.dispatchWorkgroups(Math.ceil(nx / 8), Math.ceil(nz / 8)); pass.end();
    }
    let currentIsA = true;
    operations.forEach((operation, index) => {
      const last = index === operations.length - 1;
      const pass = encoder.beginComputePass(last && querySet ? { timestampWrites: { querySet, endOfPassWriteIndex: 1 } } : undefined); pass.setPipeline(this.cache.pipelines[operation.entry]); pass.setBindGroup(0, currentIsA ? groupAB : groupBA, [index * passStride]);
      pass.dispatchWorkgroups(Math.ceil(nx / 8), Math.ceil(nz / 8)); pass.end(); currentIsA = !currentIsA;
    });
    const finalTopology = currentIsA ? topologyA : topologyB;
    encoder.copyBufferToBuffer(finalTopology, 0, readback, topologyOffset, topologyBytes);
    encoder.copyBufferToBuffer(inputs.diagnosticBuffer, 0, readback, diagnosticOffset, inputs.diagnosticBytes);
    if (querySet && queryResolve) {
      encoder.resolveQuerySet(querySet, 0, 2, queryResolve, 0);
      encoder.copyBufferToBuffer(queryResolve, 0, readback, queryOffset, 16);
    }
    const submittedAt = performance.now(); this.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    let packedCells: Uint32Array, diagnostics: Float32Array, gpuKernel_ms: number | undefined;
    try {
      const mapped = readback.getMappedRange();
      packedCells = new Uint32Array(mapped, topologyOffset, topologyBytes / 4).slice();
      diagnostics = new Float32Array(mapped, diagnosticOffset, inputs.diagnosticBytes / 4).slice();
      const timestamps = querySet ? new BigUint64Array(mapped, queryOffset, 2) : undefined;
      gpuKernel_ms = timestamps ? Number(timestamps[1] - timestamps[0]) / 1e6 : undefined;
    } finally {
      readback.unmap();
    }
    if (inputs.readLeafProfiles === false) return { columnProfiles: new Float32Array(0), packedCells, diagnostics, mismatchFraction: 0, gpuKernel_ms, gpuWall_ms: performance.now() - submittedAt };
    const leafWords = Uint32Array.from(new Set(packedCells));
    const profileBytes = leafWords.length * ny * 3 * 4;
    this.device.queue.writeBuffer(explicit, 0, leafWords);
    this.device.queue.writeBuffer(passBuffer, 0, new Uint32Array([leafWords.length, 0, 0, 0]));
    const profileEncoder = this.device.createCommandEncoder({ label: "GPU quadtree leaf profiles" });
    {
      const pass = profileEncoder.beginComputePass(); pass.setPipeline(this.cache.pipelines.sampleLeafProfiles); pass.setBindGroup(0, groupAB, [0]);
      pass.dispatchWorkgroups(Math.ceil(leafWords.length / 64)); pass.end();
    }
    profileEncoder.copyBufferToBuffer(columnProfiles, 0, profileReadback, 0, profileBytes);
    this.device.queue.submit([profileEncoder.finish()]);
    await profileReadback.mapAsync(GPUMapMode.READ, 0, profileBytes);
    let profiles: Float32Array;
    try { profiles = new Float32Array(profileReadback.getMappedRange(0, profileBytes)).slice(); }
    finally { profileReadback.unmap(); }
    return { columnProfiles: profiles, packedCells, diagnostics, mismatchFraction: 0, gpuKernel_ms, gpuWall_ms: performance.now() - submittedAt };
  }

  private static destroyWorkspace(workspace?: WebGPUQuadtreeConstructionCache["workspace"]) {
    if (!workspace) return;
    for (const buffer of [workspace.topologyA, workspace.topologyB, workspace.sizing, workspace.explicit, workspace.staticParams, workspace.levelSetOut, workspace.distanceSeedA, workspace.distanceSeedB, workspace.passBuffer, workspace.readback, workspace.columnProfiles, workspace.profileReadback, workspace.queryResolve]) buffer?.destroy();
    workspace.querySet?.destroy();
  }

  static destroyCache(cache?: WebGPUQuadtreeConstructionCache) {
    WebGPUQuadtreeBuilder.destroyWorkspace(cache?.workspace);
    if (cache) cache.workspace = undefined;
  }
}
