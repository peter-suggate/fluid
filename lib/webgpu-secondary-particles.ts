/**
 * GPU secondary-liquid particles shared by the Eulerian solvers.
 *
 * Particles are one-way by default. An explicitly enabled, bounded correction
 * can union only near-interface particles back into the resident level set;
 * it never injects particle momentum into the pressure solve. A fixed ring
 * keeps both paths allocation- and readback-free.
 */

import { CAMERA_TAN_HALF_FOV } from "./webgpu-camera";
import type { GPUInitializationTask } from "./gpu-initialization";

const secondaryParticlePipelineCache = new WeakMap<GPUDevice, Map<string, GPUComputePipeline[]>>();

export const SECONDARY_PARTICLE_STRIDE_BYTES = 64;
export const DEFAULT_SECONDARY_PARTICLE_CAPACITY = 16_384;

export type SecondaryParticleFieldLayout = "uniform" | "restricted-tall-cell";
export type SecondaryParticleSurfaceEncoding = "level-set" | "occupancy";

export interface SecondaryParticleSamplingSource {
  surfaceTexture: GPUTexture;
  velocityTexture: GPUTexture;
  columnBaseTexture: GPUTexture;
  fieldLayout: SecondaryParticleFieldLayout;
  surfaceEncoding: SecondaryParticleSurfaceEncoding;
}

export interface GPUSecondaryParticleSource {
  readonly buffer: GPUBuffer;
  readonly capacity: number;
  readonly strideBytes: typeof SECONDARY_PARTICLE_STRIDE_BYTES;
  /** Draws only the ring prefix that has ever received a particle. */
  readonly indirectBuffer?: GPUBuffer;
  readonly indirectOffset?: number;
}

export interface SecondaryParticleGrid {
  nx: number;
  ny: number;
  nz: number;
}

export interface SecondaryParticleDomain {
  width_m: number;
  height_m: number;
  depth_m: number;
  topOpen: boolean;
  gravity_m_s2: { x: number; y: number; z: number };
  density_kg_m3: number;
  surfaceTension_N_m: number;
  randomSeed: number;
}

export function secondaryParticleCapacity(value: number | undefined) {
  if (!Number.isFinite(value)) return DEFAULT_SECONDARY_PARTICLE_CAPACITY;
  return Math.max(1_024, Math.min(65_536, Math.round(value! / 1_024) * 1_024));
}

export const secondaryParticleComputeShader = /* wgsl */ `
struct Particle {
  positionRadius: vec4f,
  velocityAge: vec4f,
  birthNormalLifetime: vec4f,
  shape: vec4f,
}

struct ParticleState {
  drawVertexCount: u32,
  drawInstanceCount: atomic<u32>,
  drawFirstVertex: u32,
  drawFirstInstance: u32,
  cursor: atomic<u32>,
  spawned: atomic<u32>,
  reserved0: atomic<u32>,
  reserved1: atomic<u32>,
}

struct Params {
  gridAndDt: vec4f,
  cellAndMinimum: vec4f,
  containerAndTop: vec4f,
  gravityAndSeed: vec4f,
  controls: vec4f,
  fieldModes: vec4f,
  material: vec4f,
}

@group(0) @binding(0) var surfaceField: texture_3d<f32>;
@group(0) @binding(1) var velocityField: texture_3d<f32>;
@group(0) @binding(2) var columnBases: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(4) var<storage, read_write> particleState: ParticleState;
@group(0) @binding(5) var<uniform> params: Params;

fn dims() -> vec3i { return vec3i(params.gridAndDt.xyz); }
fn dt() -> f32 { return params.gridAndDt.w; }
fn cellSize() -> vec3f { return params.cellAndMinimum.xyz; }
fn minimumCell() -> f32 { return params.cellAndMinimum.w; }
fn capacity() -> u32 { return u32(params.controls.x); }
fn restrictedLayout() -> bool { return params.fieldModes.x > 0.5; }
fn occupancySurface() -> bool { return params.fieldModes.y > 0.5; }

fn hash(value: u32) -> u32 {
  var x = value;
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

fn random01(value: u32) -> f32 {
  return f32(hash(value) & 0x00ffffffu) / 16777216.0;
}

fn validCell(p: vec3i) -> bool {
  return all(p >= vec3i(0)) && all(p < dims());
}

fn packedY(cell: vec3i) -> vec2i {
  let base = i32(round(textureLoad(columnBases, cell.xz, 0).x));
  if (cell.y < base && base > 0) {
    let denominator = f32(max(base - 1, 1));
    return vec2i(-1, bitcast<i32>(clamp(f32(cell.y) / denominator, 0.0, 1.0)));
  }
  return vec2i(2 + cell.y - base, 0);
}

fn surfaceRaw(cell: vec3i) -> f32 {
  let q = clamp(cell, vec3i(0), dims() - vec3i(1));
  if (!restrictedLayout()) { return textureLoad(surfaceField, q, 0).x; }
  let mapped = packedY(q);
  if (mapped.x < 0) {
    let t = bitcast<f32>(mapped.y);
    return mix(textureLoad(surfaceField, vec3i(q.x, 0, q.z), 0).x, textureLoad(surfaceField, vec3i(q.x, 1, q.z), 0).x, t);
  }
  let stored = vec3i(textureDimensions(surfaceField));
  if (mapped.x >= stored.y) { return 4.0 * minimumCell(); }
  return textureLoad(surfaceField, vec3i(q.x, mapped.x, q.z), 0).x;
}

fn phiCell(cell: vec3i) -> f32 {
  let raw = surfaceRaw(cell);
  return select(raw, (0.5 - clamp(raw, 0.0, 1.0)) * 4.0 * minimumCell(), occupancySurface());
}

fn velocityRaw(cell: vec3i) -> vec3f {
  let q = clamp(cell, vec3i(0), dims() - vec3i(1));
  if (!restrictedLayout()) { return textureLoad(velocityField, q, 0).xyz; }
  let mapped = packedY(q);
  if (mapped.x < 0) {
    let t = bitcast<f32>(mapped.y);
    return mix(textureLoad(velocityField, vec3i(q.x, 0, q.z), 0).xyz, textureLoad(velocityField, vec3i(q.x, 1, q.z), 0).xyz, t);
  }
  let stored = vec3i(textureDimensions(velocityField));
  if (mapped.x >= stored.y) { return vec3f(0.0); }
  return textureLoad(velocityField, vec3i(q.x, mapped.x, q.z), 0).xyz;
}

// The shared velocity texture uses MAC samples: component a at a cell and its
// negative-axis neighbor bracket the cell center.
fn centeredVelocity(cell: vec3i) -> vec3f {
  let q = clamp(cell, vec3i(0), dims() - vec3i(1));
  let here = velocityRaw(q);
  return 0.5 * vec3f(
    here.x + velocityRaw(q - vec3i(1, 0, 0)).x,
    here.y + velocityRaw(q - vec3i(0, 1, 0)).y,
    here.z + velocityRaw(q - vec3i(0, 0, 1)).z
  );
}

fn gridCoordinate(world: vec3f) -> vec3f {
  let local = world - vec3f(-0.5 * params.containerAndTop.x, 0.0, -0.5 * params.containerAndTop.z);
  return local / params.containerAndTop.xyz * vec3f(dims()) - vec3f(0.5);
}

fn insideDomain(world: vec3f) -> bool {
  let half = 0.5 * params.containerAndTop.xz;
  return world.x >= -half.x && world.x <= half.x && world.z >= -half.y && world.z <= half.y
    && world.y >= 0.0 && world.y <= params.containerAndTop.y;
}

fn samplePhi(world: vec3f) -> f32 {
  if (!insideDomain(world)) { return 4.0 * minimumCell(); }
  let q = clamp(gridCoordinate(world), vec3f(0.0), vec3f(dims() - vec3i(1)));
  let base = vec3i(floor(q));
  let f = fract(q);
  let z0 = mix(mix(phiCell(base), phiCell(base + vec3i(1, 0, 0)), f.x), mix(phiCell(base + vec3i(0, 1, 0)), phiCell(base + vec3i(1, 1, 0)), f.x), f.y);
  let z1 = mix(mix(phiCell(base + vec3i(0, 0, 1)), phiCell(base + vec3i(1, 0, 1)), f.x), mix(phiCell(base + vec3i(0, 1, 1)), phiCell(base + vec3i(1, 1, 1)), f.x), f.y);
  return mix(z0, z1, f.z);
}

fn sampleVelocity(world: vec3f) -> vec3f {
  if (!insideDomain(world)) { return vec3f(0.0); }
  let q = clamp(gridCoordinate(world), vec3f(0.0), vec3f(dims() - vec3i(1)));
  let base = vec3i(floor(q));
  let f = fract(q);
  let z0 = mix(mix(centeredVelocity(base), centeredVelocity(base + vec3i(1, 0, 0)), f.x), mix(centeredVelocity(base + vec3i(0, 1, 0)), centeredVelocity(base + vec3i(1, 1, 0)), f.x), f.y);
  let z1 = mix(mix(centeredVelocity(base + vec3i(0, 0, 1)), centeredVelocity(base + vec3i(1, 0, 1)), f.x), mix(centeredVelocity(base + vec3i(0, 1, 1)), centeredVelocity(base + vec3i(1, 1, 1)), f.x), f.y);
  return mix(z0, z1, f.z);
}

fn surfaceNormal(world: vec3f) -> vec3f {
  let h = cellSize();
  let gradient = vec3f(
    samplePhi(world + vec3f(h.x, 0.0, 0.0)) - samplePhi(world - vec3f(h.x, 0.0, 0.0)),
    samplePhi(world + vec3f(0.0, h.y, 0.0)) - samplePhi(world - vec3f(0.0, h.y, 0.0)),
    samplePhi(world + vec3f(0.0, 0.0, h.z)) - samplePhi(world - vec3f(0.0, 0.0, h.z))
  );
  if (dot(gradient, gradient) > 1e-10) { return normalize(gradient); }
  return vec3f(0.0, 1.0, 0.0);
}

fn deactivate(index: u32, particle: Particle) {
  var dead = particle;
  dead.positionRadius.w = 0.0;
  dead.shape.z = 0.0;
  particles[index] = dead;
}

fn capillaryTime(radius: f32) -> f32 {
  let density = max(params.material.x, 1.0);
  let sigma = params.material.y;
  // Paper-comparison scenes intentionally disable capillarity in the grid
  // solver. Keep their detached fragments stretched for longer rather than
  // inventing a hidden surface-tension coefficient or dividing by zero.
  if (sigma <= 1e-6) { return 0.32; }
  return clamp(sqrt(density * radius * radius * radius / sigma), 0.035, 0.45);
}

@compute @workgroup_size(64)
fn updateParticles(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  if (index >= capacity()) { return; }
  var particle = particles[index];
  if (particle.shape.z < 0.5 || particle.positionRadius.w <= 0.0) { return; }

  var position = particle.positionRadius.xyz;
  var velocity = particle.velocityAge.xyz;
  var age = particle.velocityAge.w + dt();
  let lifetime = particle.birthNormalLifetime.w;
  if (age >= lifetime) { deactivate(index, particle); return; }

  velocity = (velocity + params.gravityAndSeed.xyz * dt()) * exp(-0.18 * dt());
  position += velocity * dt();
  let half = 0.5 * params.containerAndTop.xz;
  if (position.y < 0.0 || position.y > params.containerAndTop.y + 2.0 * minimumCell()) { deactivate(index, particle); return; }
  if (abs(position.x) > half.x || abs(position.z) > half.y) { deactivate(index, particle); return; }
  // Re-entered droplets belong to the resolved body again; no long-lived
  // surface marker is retained.
  if (age > 2.0 * dt() && samplePhi(position) < -0.2 * minimumCell()) { deactivate(index, particle); return; }

  particle.positionRadius = vec4f(position, particle.positionRadius.w);
  particle.velocityAge = vec4f(velocity, age);
  particles[index] = particle;
}

@compute @workgroup_size(4, 4, 4)
fn spawnParticles(@builtin(global_invocation_id) gid: vec3u) {
  let cell = vec3i(gid);
  if (!validCell(cell)) { return; }
  let phi = phiCell(cell);
  let h = minimumCell();
  if (phi < -1.75 * h || phi > 0.0) { return; }

  let cellCenter = vec3f(-0.5 * params.containerAndTop.x, 0.0, -0.5 * params.containerAndTop.z)
    + (vec3f(cell) + vec3f(0.5)) * cellSize();
  let linear = u32(cell.x) + u32(dims().x) * (u32(cell.y) + u32(dims().y) * u32(cell.z));
  let step = u32(params.fieldModes.w);
  let macroCell = cell / vec3i(3);
  let macroDims = (dims() + vec3i(2)) / vec3i(3);
  let macroLinear = u32(macroCell.x) + u32(macroDims.x) * (u32(macroCell.y) + u32(macroDims.y) * u32(macroCell.z));
  let eventSeed = macroLinear ^ u32(params.gravityAndSeed.w) ^ hash(step / 8u + 0x68bc21ebu);
  let seed = linear ^ u32(params.gravityAndSeed.w) ^ hash(step);
  let jitter = vec3f(
    random01(seed ^ 0x68bc21ebu) - 0.5,
    random01(seed ^ 0x02e5be93u) - 0.5,
    random01(seed ^ 0x967a889bu) - 0.5
  );
  let trial = cellCenter + jitter * cellSize() * 0.78;
  let trialPhi = samplePhi(trial);
  if (trialPhi < -1.75 * h || trialPhi > 0.0) { return; }

  // Section 3.9.3 assigns each trial particle the local signed-distance
  // radius. Keeping that variation is important visually: a fixed radius
  // turns a breaking sheet into a regular string of identical dots.
  let radius = clamp(-trialPhi, 0.08 * h, 0.85 * h);
  let normal = surfaceNormal(trial);
  let velocity = sampleVelocity(trial);
  let outwardSpeed = dot(velocity, normal);
  if (outwardSpeed <= 0.0) { return; }

  // Chentanez-Mueller Sec. 3.9.3: promote a near-surface trial particle only
  // if its advected position has escaped farther than twice its radius. Their
  // examples use a 1/30 s simulation step. Our solver may split that interval
  // into 4 ms stability substeps, so using dt() here would nearly eliminate
  // successful trials; only the generation probe uses the paper horizon.
  let generationDt = max(dt(), 1.0 / 30.0);
  let escaped = trial + velocity * generationDt;
  if (!insideDomain(escaped) || samplePhi(escaped) <= 2.0 * radius) { return; }
  // A coarse, slowly-changing event gain makes neighboring cells break up in
  // coherent bursts while the fine-cell hash retains irregular boundaries.
  // This avoids a neighbor grid and does not synchronize whole macrocells.
  let eventNoise = random01(eventSeed);
  let eventGain = 0.18 + 1.72 * smoothstep(0.18, 0.92, eventNoise);
  let probability = clamp(dt() * params.fieldModes.z * (0.5 + outwardSpeed) * eventGain, 0.0, 0.72);
  if (random01(seed) > probability) { return; }

  // Chentanez-Mueller Sec. 3.9.2 identifies thin regions by finding air on
  // both sides of the interface normal. We retain that information only as a
  // render shape: the one-way particles still never modify phi or pressure.
  let sheetProbe = 2.0 * h;
  let thinSheet = samplePhi(trial + normal * sheetProbe) > 0.0 && samplePhi(trial - normal * sheetProbe) > 0.0;

  // The paper emits a number of escape particles at the successful trial,
  // rather than one probabilistic marker. Small coherent clusters preserve
  // the shape of the unresolved sheet while the fixed ring bounds their cost.
  let clusterCount = 1u + min(3u, u32(floor(outwardSpeed / max(params.controls.y, 0.1))));
  for (var emitted = 0u; emitted < clusterCount; emitted += 1u) {
    let particleSeed = seed ^ hash(emitted + 0x9e3779b9u);
    let scatter = vec3f(
      random01(particleSeed ^ 0x85ebca6bu) - 0.5,
      random01(particleSeed ^ 0xc2b2ae35u) - 0.5,
      random01(particleSeed ^ 0x27d4eb2fu) - 0.5
    );
    // Outward speed is our available proxy for breakup energy. Bias energetic
    // events toward smaller fragments, as a high Weber number would, but keep
    // a minority of coarse fragments so the result is not another uniform
    // size class. The exponent and floor deliberately bound this heuristic
    // until a sampled air-relative velocity is available.
    let speedRatio = outwardSpeed / max(params.controls.y, 0.1);
    let energyRatio = max(1.0, speedRatio * speedRatio);
    let breakupScale = clamp(pow(energyRatio, -0.32), 0.38, 1.0);
    let sizeBias = 0.55 + 0.4 * random01(particleSeed ^ 0xd3a2646cu);
    var energyRadiusScale = mix(1.0, breakupScale, sizeBias);
    if (random01(particleSeed ^ 0xfd7046c5u) < 0.16) {
      energyRadiusScale = 0.88 + 0.24 * random01(particleSeed ^ 0xb55a4f09u);
    }
    let particleRadius = radius * (0.72 + 0.5 * random01(particleSeed ^ 0x165667b1u)) * energyRadiusScale;
    let particlePosition = escaped + (scatter - normal * min(dot(scatter, normal), 0.0)) * radius * 0.7;
    let particleVelocity = velocity + scatter * (0.12 * length(velocity) + 0.08);
    let lifetime = 1.1 + 1.2 * random01(particleSeed ^ 0x3c6ef372u);
    let tangentialSpeed = length(velocity - normal * outwardSpeed);
    let ligament = !thinSheet && (speedRatio > 1.15 || tangentialSpeed > params.controls.y);
    let shapeKind = select(select(0.0, 1.0, ligament), 2.0, thinSheet);
    var initialAspect = 1.0;
    if (shapeKind > 1.5) {
      initialAspect = clamp(1.55 + 0.55 * speedRatio + 0.35 * random01(particleSeed ^ 0xa24baed5u), 1.55, 3.2);
    } else if (shapeKind > 0.5) {
      initialAspect = clamp(1.35 + 0.7 * speedRatio + 0.45 * random01(particleSeed ^ 0x9fb21c65u), 1.35, 3.8);
    }
    let absoluteSlot = atomicAdd(&particleState.cursor, 1u);
    let slot = absoluteSlot % capacity();
    atomicAdd(&particleState.spawned, 1u);
    atomicMax(&particleState.drawInstanceCount, min(absoluteSlot + 1u, capacity()));
    particles[slot] = Particle(
      vec4f(particlePosition, particleRadius),
      vec4f(particleVelocity, 0.0),
      vec4f(normal, lifetime),
      vec4f(initialAspect, shapeKind, 1.0, capillaryTime(particleRadius))
    );
  }
}
`;

/**
 * Optional particle-level-set correction. The atomic field stores the nearest
 * particle sphere SDF at each touched cell. Correction is deliberately narrow:
 * detached spray is ignored, the interface can move by at most 0.2h per
 * substep, and no corrected sample is pushed deeper than -0.5h. This preserves
 * thin protrusions without turning the spray ring into an unbounded mass source.
 */
export const secondaryParticleCorrectionShader = /* wgsl */ `
struct Particle {
  positionRadius: vec4f,
  velocityAge: vec4f,
  birthNormalLifetime: vec4f,
  shape: vec4f,
}
struct Params {
  gridAndDt: vec4f,
  cellAndMinimum: vec4f,
  containerAndTop: vec4f,
  gravityAndSeed: vec4f,
  controls: vec4f,
  fieldModes: vec4f,
  material: vec4f,
}
@group(0) @binding(0) var surfaceField: texture_3d<f32>;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> nearestParticlePhi: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var correctedSurface: texture_storage_3d<r32float, write>;

fn dims() -> vec3u { return vec3u(params.gridAndDt.xyz); }
fn cellSize() -> vec3f { return params.cellAndMinimum.xyz; }
fn hMin() -> f32 { return params.cellAndMinimum.w; }
fn capacity() -> u32 { return u32(params.controls.x); }
fn correctionStrength() -> f32 { return clamp(params.controls.w, 0.0, 1.0); }
fn index3(q: vec3u) -> u32 { return q.x + dims().x * (q.y + dims().y * q.z); }
fn domainOrigin() -> vec3f { return vec3f(-0.5 * params.containerAndTop.x, 0.0, -0.5 * params.containerAndTop.z); }
fn cellCentre(q: vec3i) -> vec3f { return domainOrigin() + (vec3f(q) + vec3f(0.5)) * cellSize(); }
fn cellAt(world: vec3f) -> vec3i { return vec3i(floor((world - domainOrigin()) / cellSize())); }

@compute @workgroup_size(4, 4, 4)
fn resetParticleCorrection(@builtin(global_invocation_id) gid: vec3u) {
  if (all(gid < dims())) { atomicStore(&nearestParticlePhi[index3(gid)], 0x7f800000u); }
}

@compute @workgroup_size(64)
fn splatParticleCorrection(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= capacity() || correctionStrength() <= 0.0) { return; }
  let particle = particles[gid.x];
  if (particle.shape.z < 0.5 || particle.positionRadius.w <= 0.0) { return; }
  let centreCell = clamp(cellAt(particle.positionRadius.xyz), vec3i(0), vec3i(dims()) - vec3i(1));
  let residentPhi = textureLoad(surfaceField, centreCell, 0).x;
  // A genuinely detached droplet remains render-only. Feedback is restricted
  // to particles still close enough to represent an under-resolved sheet.
  if (abs(residentPhi) > 2.0 * hMin()) { return; }
  for (var dz = -2; dz <= 2; dz += 1) { for (var dy = -2; dy <= 2; dy += 1) { for (var dx = -2; dx <= 2; dx += 1) {
    let q = centreCell + vec3i(dx, dy, dz);
    if (any(q < vec3i(0)) || any(q >= vec3i(dims()))) { continue; }
    let particlePhi = length(cellCentre(q) - particle.positionRadius.xyz) - particle.positionRadius.w;
    if (particlePhi > 2.0 * hMin()) { continue; }
    // Shift the bounded signed value into the non-negative float range, where
    // IEEE-754 bit order permits an atomic integer minimum.
    let encoded = bitcast<u32>(clamp(particlePhi, -hMin(), 2.0 * hMin()) + 2.0 * hMin());
    atomicMin(&nearestParticlePhi[index3(vec3u(q))], encoded);
  } } }
}

@compute @workgroup_size(4, 4, 4)
fn applyParticleCorrection(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let residentPhi = textureLoad(surfaceField, vec3i(gid), 0).x;
  let encoded = atomicLoad(&nearestParticlePhi[index3(gid)]);
  var result = residentPhi;
  if (encoded != 0x7f800000u && residentPhi > -0.5 * hMin() && residentPhi < 2.0 * hMin()) {
    let particlePhi = bitcast<f32>(encoded) - 2.0 * hMin();
    let maximumShift = 0.2 * hMin() * correctionStrength();
    result = max(-0.5 * hMin(), residentPhi - min(max(residentPhi - particlePhi, 0.0), maximumShift));
  }
  textureStore(correctedSurface, vec3i(gid), vec4f(result, 0.0, 0.0, 0.0));
}
`;

/**
 * Exact ellipsoid interfaces for the raster water buffers. New sheet and
 * ligament fragments retain their birth deformation, then relax toward a
 * sphere on their capillary time scale. The optical composite cannot
 * distinguish these interfaces from the Eulerian surface, so they inherit
 * the resolved water's refraction, absorption, reflection, and occlusion.
 */
export const secondaryParticleOpticalShader = /* wgsl */ `
struct ViewUniforms {
  viewport: vec4f,
  cameraPosition: vec4f,
  cameraTarget: vec4f,
  container: vec4f,
}

struct Particle {
  positionRadius: vec4f,
  velocityAge: vec4f,
  birthNormalLifetime: vec4f,
  shape: vec4f,
}

@group(0) @binding(0) var<uniform> view: ViewUniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct EllipsoidVertex {
  @builtin(position) clip: vec4f,
  @location(0) rayPoint: vec3f,
  @location(1) @interpolate(flat) center: vec3f,
  @location(2) @interpolate(flat) inverseAxis0: vec3f,
  @location(3) @interpolate(flat) inverseAxis1: vec3f,
  @location(4) @interpolate(flat) inverseAxis2: vec3f,
  @location(5) @interpolate(flat) enabled: f32,
}

struct InterfaceFragment {
  @location(0) position: vec4f,
  @location(1) normal: vec4f,
  @builtin(frag_depth) depth: f32,
}

fn cameraForward() -> vec3f { return normalize(view.cameraTarget.xyz - view.cameraPosition.xyz); }
fn cameraRight() -> vec3f { return normalize(cross(cameraForward(), vec3f(0.0, 1.0, 0.0))); }
fn cameraUp() -> vec3f { return normalize(cross(cameraRight(), cameraForward())); }

fn safeNormalize(value: vec3f, fallback: vec3f) -> vec3f {
  let magnitude2 = dot(value, value);
  return select(fallback, value * inverseSqrt(magnitude2), magnitude2 > 1e-10);
}

fn perpendicular(axis: vec3f) -> vec3f {
  let reference = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(axis.y) < 0.82);
  return safeNormalize(cross(axis, reference), vec3f(0.0, 0.0, 1.0));
}

fn project(world: vec3f) -> vec4f {
  let forward = cameraForward();
  let relative = world - view.cameraPosition.xyz;
  let eyeDepth = dot(relative, forward);
  let aspect = view.viewport.x / max(view.viewport.y, 1.0);
  let ndc = vec2f(dot(relative, cameraRight()) / (max(eyeDepth, 0.001) * aspect * ${CAMERA_TAN_HALF_FOV}), dot(relative, cameraUp()) / (max(eyeDepth, 0.001) * ${CAMERA_TAN_HALF_FOV}));
  return vec4f(ndc * eyeDepth, clamp(eyeDepth / 50.0, 0.0, 1.0) * eyeDepth, eyeDepth);
}

@vertex
fn ellipsoidVertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> EllipsoidVertex {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  let particle = particles[instance];
  var output: EllipsoidVertex;
  output.center = particle.positionRadius.xyz;
  output.inverseAxis0 = vec3f(0.0);
  output.inverseAxis1 = vec3f(0.0);
  output.inverseAxis2 = vec3f(0.0);
  output.rayPoint = output.center;
  output.enabled = particle.shape.z;
  let radius = particle.positionRadius.w;
  if (output.enabled < 0.5 || radius <= 0.0) {
    output.clip = vec4f(2.0, 2.0, 2.0, 1.0);
    output.enabled = 0.0;
    return output;
  }

  let normal = safeNormalize(particle.birthNormalLifetime.xyz, vec3f(0.0, 1.0, 0.0));
  let velocity = particle.velocityAge.xyz;
  let flowAxis = safeNormalize(velocity, perpendicular(normal));
  let tangent = safeNormalize(velocity - normal * dot(velocity, normal), perpendicular(normal));
  let sheetSide = safeNormalize(cross(normal, tangent), perpendicular(normal));
  let ligamentSide = perpendicular(flowAxis);
  let ligamentOther = safeNormalize(cross(flowAxis, ligamentSide), normal);
  let capillaryTime = max(particle.shape.w, 0.02);
  let aspect = 1.0 + (max(particle.shape.x, 1.0) - 1.0) * exp(-particle.velocityAge.w / capillaryTime);
  let shapeKind = particle.shape.y;
  var axis0: vec3f;
  var axis1: vec3f;
  var axis2: vec3f;
  if (shapeKind > 1.5) {
    // Sheet particles are oblate: two expanded in-plane axes and a thin
    // normal axis. sqrt(s) * sqrt(s) / s preserves the source volume.
    let inPlane = sqrt(aspect);
    axis0 = tangent * radius * inPlane;
    axis1 = sheetSide * radius * inPlane;
    axis2 = normal * radius / aspect;
  } else {
    // Ligaments are prolate along their motion. Drop-class particles have an
    // aspect of one and therefore use the same exact path as a sphere.
    let transverse = inverseSqrt(aspect);
    axis0 = flowAxis * radius * aspect;
    axis1 = ligamentSide * radius * transverse;
    axis2 = ligamentOther * radius * transverse;
  }
  output.inverseAxis0 = axis0 / dot(axis0, axis0);
  output.inverseAxis1 = axis1 / dot(axis1, axis1);
  output.inverseAxis2 = axis2 / dot(axis2, axis2);

  let right = cameraRight();
  let up = cameraUp();
  let forward = cameraForward();
  let extentRight = sqrt(dot(axis0, right) * dot(axis0, right) + dot(axis1, right) * dot(axis1, right) + dot(axis2, right) * dot(axis2, right));
  let extentUp = sqrt(dot(axis0, up) * dot(axis0, up) + dot(axis1, up) * dot(axis1, up) + dot(axis2, up) * dot(axis2, up));
  let extentDepth = sqrt(dot(axis0, forward) * dot(axis0, forward) + dot(axis1, forward) * dot(axis1, forward) + dot(axis2, forward) * dot(axis2, forward));
  let centerDepth = dot(output.center - view.cameraPosition.xyz, forward);
  let perspectivePad = 1.035 + extentDepth / max(centerDepth, 0.02);
  output.rayPoint = output.center + right * corners[vertex].x * extentRight * perspectivePad + up * corners[vertex].y * extentUp * perspectivePad;
  output.clip = project(output.rayPoint);
  if (output.clip.w <= 0.001) { output.clip = vec4f(2.0, 2.0, 2.0, 1.0); output.enabled = 0.0; }
  return output;
}

fn ellipsoidInterface(input: EllipsoidVertex, back: bool) -> InterfaceFragment {
  if (input.enabled < 0.5) { discard; }
  let rayOrigin = view.cameraPosition.xyz;
  let rayDirection = safeNormalize(input.rayPoint - rayOrigin, cameraForward());
  let offset = rayOrigin - input.center;
  let qOrigin = vec3f(dot(offset, input.inverseAxis0), dot(offset, input.inverseAxis1), dot(offset, input.inverseAxis2));
  let qDirection = vec3f(dot(rayDirection, input.inverseAxis0), dot(rayDirection, input.inverseAxis1), dot(rayDirection, input.inverseAxis2));
  let a = dot(qDirection, qDirection);
  let b = dot(qOrigin, qDirection);
  let discriminant = b * b - a * (dot(qOrigin, qOrigin) - 1.0);
  if (discriminant < 0.0 || a <= 1e-12) { discard; }
  let root = sqrt(max(discriminant, 0.0));
  let nearDistance = (-b - root) / a;
  let farDistance = (-b + root) / a;
  let distance = select(nearDistance, farDistance, back);
  if (distance <= 1e-4) { discard; }
  let world = rayOrigin + rayDirection * distance;
  let relative = world - input.center;
  let q = vec3f(dot(relative, input.inverseAxis0), dot(relative, input.inverseAxis1), dot(relative, input.inverseAxis2));
  let normal = safeNormalize(input.inverseAxis0 * q.x + input.inverseAxis1 * q.y + input.inverseAxis2 * q.z, -rayDirection);
  let clip = project(world);
  return InterfaceFragment(vec4f(world, 1.0), vec4f(normal, 1.0), clamp(clip.z / max(clip.w, 0.001), 0.0, 1.0));
}

@fragment fn ellipsoidFront(input: EllipsoidVertex) -> InterfaceFragment { return ellipsoidInterface(input, false); }
@fragment fn ellipsoidBack(input: EllipsoidVertex) -> InterfaceFragment { return ellipsoidInterface(input, true); }
`;

export class WebGPUSecondaryParticleSystem {
  readonly renderSource: GPUSecondaryParticleSource;
  readonly allocatedBytes: number;
  private readonly particles: GPUBuffer;
  private readonly state: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;
  private readonly shaderModule: GPUShaderModule;
  private readonly pipelineLayout: GPUPipelineLayout;
  private readonly correction?: {
    candidates: GPUBuffer;
    texture: GPUTexture;
    shaderModule: GPUShaderModule;
    pipelineLayout: GPUPipelineLayout;
    bindGroup: GPUBindGroup;
    surfaceTexture: GPUTexture;
  };
  private updatePipeline?: GPUComputePipeline;
  private spawnPipeline?: GPUComputePipeline;
  private resetCorrectionPipeline?: GPUComputePipeline;
  private splatCorrectionPipeline?: GPUComputePipeline;
  private applyCorrectionPipeline?: GPUComputePipeline;
  private step = 0;

  constructor(
    private readonly device: GPUDevice,
    private readonly grid: SecondaryParticleGrid,
    private readonly domain: SecondaryParticleDomain,
    source: SecondaryParticleSamplingSource,
    capacityValue = DEFAULT_SECONDARY_PARTICLE_CAPACITY,
    deferPipelineCompilation = false,
    private readonly surfaceCorrectionStrength = 0
  ) {
    const capacity = secondaryParticleCapacity(capacityValue);
    this.particles = device.createBuffer({
      label: `Secondary liquid particles (${capacity})`,
      size: capacity * SECONDARY_PARTICLE_STRIDE_BYTES,
      usage: GPUBufferUsage.STORAGE
    });
    this.state = device.createBuffer({ label: "Secondary particle ring state and draw arguments", size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
    this.params = device.createBuffer({ label: "Secondary particle parameters", size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.state, 0, new Uint32Array([6, 0, 0, 0, 0, 0, 0, 0]));
    this.renderSource = { buffer: this.particles, capacity, strideBytes: SECONDARY_PARTICLE_STRIDE_BYTES, indirectBuffer: this.state, indirectOffset: 0 };
    const correctionEnabled = source.surfaceEncoding === "level-set" && surfaceCorrectionStrength > 0;
    const correctionBytes = correctionEnabled ? grid.nx * grid.ny * grid.nz * 8 : 0;
    this.allocatedBytes = capacity * SECONDARY_PARTICLE_STRIDE_BYTES + 144 + correctionBytes;

    this.shaderModule = device.createShaderModule({ label: "Secondary liquid particle kernels", code: secondaryParticleComputeShader });
    const layout = device.createBindGroupLayout({ label: "Secondary particle simulation bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
    ] });
    this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    this.bindGroup = device.createBindGroup({ layout, entries: [
      { binding: 0, resource: source.surfaceTexture.createView({ dimension: "3d" }) },
      { binding: 1, resource: source.velocityTexture.createView({ dimension: "3d" }) },
      { binding: 2, resource: source.columnBaseTexture.createView() },
      { binding: 3, resource: { buffer: this.particles } },
      { binding: 4, resource: { buffer: this.state } },
      { binding: 5, resource: { buffer: this.params } }
    ] });
    if (correctionEnabled) {
      const correctionLayout = device.createBindGroupLayout({ label: "Secondary particle surface-correction bindings", entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } }
      ] });
      const candidates = device.createBuffer({
        label: "Secondary particle nearest-surface candidates",
        size: Math.max(4, grid.nx * grid.ny * grid.nz * 4),
        usage: GPUBufferUsage.STORAGE
      });
      const texture = device.createTexture({
        label: "Secondary particle corrected level set",
        size: [grid.nx, grid.ny, grid.nz],
        dimension: "3d",
        format: "r32float",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC
      });
      const shaderModule = device.createShaderModule({ label: "Secondary particle surface-correction kernels", code: secondaryParticleCorrectionShader });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [correctionLayout] });
      const bindGroup = device.createBindGroup({ label: "Secondary particle surface-correction resources", layout: correctionLayout, entries: [
        { binding: 0, resource: source.surfaceTexture.createView({ dimension: "3d" }) },
        { binding: 1, resource: { buffer: this.particles } },
        { binding: 2, resource: { buffer: candidates } },
        { binding: 3, resource: { buffer: this.params } },
        { binding: 4, resource: texture.createView({ dimension: "3d" }) }
      ] });
      this.correction = { candidates, texture, shaderModule, pipelineLayout, bindGroup, surfaceTexture: source.surfaceTexture };
    }
    this.writeParameters(1 / 60, source);
    if (!deferPipelineCompilation) this.createPipelinesSync();
  }

  private descriptor(entryPoint: "updateParticles" | "spawnParticles"): GPUComputePipelineDescriptor {
    return { layout: this.pipelineLayout, compute: { module: this.shaderModule, entryPoint } };
  }

  private correctionDescriptor(entryPoint: "resetParticleCorrection" | "splatParticleCorrection" | "applyParticleCorrection"): GPUComputePipelineDescriptor {
    if (!this.correction) throw new Error("Secondary particle surface correction is disabled");
    return { layout: this.correction.pipelineLayout, compute: { module: this.correction.shaderModule, entryPoint } };
  }

  private createPipelinesSync() {
    this.updatePipeline = this.device.createComputePipeline(this.descriptor("updateParticles"));
    this.spawnPipeline = this.device.createComputePipeline(this.descriptor("spawnParticles"));
    if (this.correction) {
      this.resetCorrectionPipeline = this.device.createComputePipeline(this.correctionDescriptor("resetParticleCorrection"));
      this.splatCorrectionPipeline = this.device.createComputePipeline(this.correctionDescriptor("splatParticleCorrection"));
      this.applyCorrectionPipeline = this.device.createComputePipeline(this.correctionDescriptor("applyParticleCorrection"));
    }
    let cache = secondaryParticlePipelineCache.get(this.device);
    if (!cache) { cache = new Map(); secondaryParticlePipelineCache.set(this.device, cache); }
    cache.set(this.correction ? "corrected" : "one-way", [this.updatePipeline, this.spawnPipeline, ...(this.correction ? [this.resetCorrectionPipeline!, this.splatCorrectionPipeline!, this.applyCorrectionPipeline!] : [])]);
  }

  get pipelineCount() { return this.correction ? 5 : 2; }

  initializationTasks(): GPUInitializationTask[] {
    const cacheKey = this.correction ? "corrected" : "one-way";
    const cached = secondaryParticlePipelineCache.get(this.device)?.get(cacheKey);
    if (cached) return [{
      id: "secondary-particles.pipeline-cache", phase: "secondary-particles", label: "Reuse compiled secondary-liquid programs", run: () => {
        this.updatePipeline = cached[0]; this.spawnPipeline = cached[1];
        if (this.correction) { this.resetCorrectionPipeline = cached[2]; this.splatCorrectionPipeline = cached[3]; this.applyCorrectionPipeline = cached[4]; }
      },
    }];
    const definitions = [["Advecting secondary particles", "updateParticles"], ["Seeding escaped spray", "spawnParticles"]] as const;
    const tasks: GPUInitializationTask[] = definitions.map(([label, entryPoint]) => ({
      id: `secondary-particles.pipeline.${entryPoint}`,
      phase: "secondary-particles",
      label,
      run: async () => {
        const pipeline = await this.device.createComputePipelineAsync(this.descriptor(entryPoint));
        if (entryPoint === "updateParticles") this.updatePipeline = pipeline;
        else this.spawnPipeline = pipeline;
      },
    }));
    if (this.correction) {
      const correctionDefinitions = [
        ["Clearing particle surface correction", "resetParticleCorrection"],
        ["Splatting near-surface particles", "splatParticleCorrection"],
        ["Applying bounded particle surface correction", "applyParticleCorrection"]
      ] as const;
      tasks.push(...correctionDefinitions.map(([label, entryPoint]) => ({
        id: `secondary-particles.pipeline.${entryPoint}`,
        phase: "secondary-particles" as const,
        label,
        run: async () => {
          const pipeline = await this.device.createComputePipelineAsync(this.correctionDescriptor(entryPoint));
          if (entryPoint === "resetParticleCorrection") this.resetCorrectionPipeline = pipeline;
          else if (entryPoint === "splatParticleCorrection") this.splatCorrectionPipeline = pipeline;
          else { this.applyCorrectionPipeline = pipeline; let cache = secondaryParticlePipelineCache.get(this.device); if (!cache) { cache = new Map(); secondaryParticlePipelineCache.set(this.device, cache); } cache.set(cacheKey, [this.updatePipeline!, this.spawnPipeline!, this.resetCorrectionPipeline!, this.splatCorrectionPipeline!, this.applyCorrectionPipeline]); }
        },
      })));
    } else {
      const last = tasks[tasks.length - 1];
      const run = last.run;
      last.run = async (signal) => { await run(signal); let cache = secondaryParticlePipelineCache.get(this.device); if (!cache) { cache = new Map(); secondaryParticlePipelineCache.set(this.device, cache); } cache.set(cacheKey, [this.updatePipeline!, this.spawnPipeline!]); };
    }
    return tasks;
  }

  async initializePipelines(onProgress: (label: string, completed: number, total: number) => void) {
    const tasks = this.initializationTasks();
    const signal = new AbortController().signal;
    for (let index = 0; index < tasks.length; index += 1) {
      onProgress(tasks[index].label, index, tasks.length);
      await tasks[index].run(signal);
      onProgress(tasks[index].label, index + 1, tasks.length);
    }
  }

  private writeParameters(dt: number, source: SecondaryParticleSamplingSource) {
    const hx = this.domain.width_m / this.grid.nx;
    const hy = this.domain.height_m / this.grid.ny;
    const hz = this.domain.depth_m / this.grid.nz;
    const h = Math.min(hx, hy, hz);
    const outwardThreshold = Math.max(0.35, 0.45 * Math.sqrt(Math.abs(this.domain.gravity_m_s2.y) * h));
    this.device.queue.writeBuffer(this.params, 0, new Float32Array([
      this.grid.nx, this.grid.ny, this.grid.nz, dt,
      hx, hy, hz, h,
      this.domain.width_m, this.domain.height_m, this.domain.depth_m, this.domain.topOpen ? 1 : 0,
      this.domain.gravity_m_s2.x, this.domain.gravity_m_s2.y, this.domain.gravity_m_s2.z, this.domain.randomSeed,
      this.renderSource.capacity, outwardThreshold, 0.22, Math.max(0, Math.min(1, this.surfaceCorrectionStrength)),
      source.fieldLayout === "restricted-tall-cell" ? 1 : 0, source.surfaceEncoding === "occupancy" ? 1 : 0, 7, this.step,
      this.domain.density_kg_m3, this.domain.surfaceTension_N_m, 0, 0
    ]));
  }

  prepareStep(dt: number, source: SecondaryParticleSamplingSource) {
    this.step += 1;
    this.writeParameters(dt, source);
  }

  encode(encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites) {
    if (!this.updatePipeline || !this.spawnPipeline) return;
    const update = encoder.beginComputePass({
      label: "Advect spray droplets",
      ...(timestampWrites?.beginningOfPassWriteIndex !== undefined ? { timestampWrites: { querySet: timestampWrites.querySet, beginningOfPassWriteIndex: timestampWrites.beginningOfPassWriteIndex } } : {})
    });
    update.setPipeline(this.updatePipeline);
    update.setBindGroup(0, this.bindGroup);
    update.dispatchWorkgroups(Math.ceil(this.renderSource.capacity / 64));
    update.end();
    const spawn = encoder.beginComputePass({
      label: "Seed escaped spray droplets",
      ...(timestampWrites?.endOfPassWriteIndex !== undefined ? { timestampWrites: { querySet: timestampWrites.querySet, endOfPassWriteIndex: timestampWrites.endOfPassWriteIndex } } : {})
    });
    spawn.setPipeline(this.spawnPipeline);
    spawn.setBindGroup(0, this.bindGroup);
    spawn.dispatchWorkgroups(Math.ceil(this.grid.nx / 4), Math.ceil(this.grid.ny / 4), Math.ceil(this.grid.nz / 4));
    spawn.end();
    if (this.correction && this.resetCorrectionPipeline && this.splatCorrectionPipeline && this.applyCorrectionPipeline) {
      const correction = encoder.beginComputePass({ label: "Bounded particle-to-level-set correction" });
      correction.setBindGroup(0, this.correction.bindGroup);
      correction.setPipeline(this.resetCorrectionPipeline);
      correction.dispatchWorkgroups(Math.ceil(this.grid.nx / 4), Math.ceil(this.grid.ny / 4), Math.ceil(this.grid.nz / 4));
      correction.setPipeline(this.splatCorrectionPipeline);
      correction.dispatchWorkgroups(Math.ceil(this.renderSource.capacity / 64));
      correction.setPipeline(this.applyCorrectionPipeline);
      correction.dispatchWorkgroups(Math.ceil(this.grid.nx / 4), Math.ceil(this.grid.ny / 4), Math.ceil(this.grid.nz / 4));
      correction.end();
      encoder.copyTextureToTexture(
        { texture: this.correction.texture },
        { texture: this.correction.surfaceTexture },
        [this.grid.nx, this.grid.ny, this.grid.nz]
      );
    }
  }

  destroy() {
    this.particles.destroy();
    this.state.destroy();
    this.params.destroy();
    this.correction?.candidates.destroy();
    this.correction?.texture.destroy();
  }
}

export class SecondaryParticleRenderPipeline {
  private opticalFrontPipeline?: GPURenderPipeline;
  private opticalBackPipeline?: GPURenderPipeline;
  private opticalBindGroup?: GPUBindGroup;
  private layout?: GPUBindGroupLayout;
  private source?: GPUSecondaryParticleSource;

  constructor(private readonly device: GPUDevice, private readonly uniformBuffer: GPUBuffer) {}

  async initialize() {
    const opticalModule = this.device.createShaderModule({ label: "Secondary liquid optical interfaces", code: secondaryParticleOpticalShader });
    this.layout = this.device.createBindGroupLayout({ label: "Secondary particle render bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } }
    ] });
    const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    const opticalDescriptor = (label: string, entryPoint: "ellipsoidFront" | "ellipsoidBack"): GPURenderPipelineDescriptor => ({
      label, layout: pipelineLayout,
      vertex: { module: opticalModule, entryPoint: "ellipsoidVertex" },
      fragment: { module: opticalModule, entryPoint, targets: [{ format: "rgba16float" }, { format: "rgba16float" }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }
    });
    [this.opticalFrontPipeline, this.opticalBackPipeline] = await Promise.all([
      this.device.createRenderPipelineAsync(opticalDescriptor("Secondary spray front interfaces", "ellipsoidFront")),
      this.device.createRenderPipelineAsync(opticalDescriptor("Secondary spray back interfaces", "ellipsoidBack"))
    ]);
  }

  setSource(source: GPUSecondaryParticleSource | undefined) {
    if (source === this.source) return;
    this.source = source;
    const bindGroup = source && this.layout ? this.device.createBindGroup({ layout: this.layout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: source.buffer } }
    ] }) : undefined;
    this.opticalBindGroup = bindGroup;
  }

  get active() {
    return Boolean(this.source && this.opticalBindGroup);
  }

  encodeOpticalInterface(pass: GPURenderPassEncoder, side: "front" | "back") {
    const pipeline = side === "front" ? this.opticalFrontPipeline : this.opticalBackPipeline;
    if (!pipeline || !this.opticalBindGroup || !this.source) return false;
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.opticalBindGroup);
    if (this.source.indirectBuffer) pass.drawIndirect(this.source.indirectBuffer, this.source.indirectOffset ?? 0);
    else pass.draw(6, this.source.capacity);
    return true;
  }
}
