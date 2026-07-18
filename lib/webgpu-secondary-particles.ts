/**
 * GPU secondary-liquid particles shared by the Eulerian solvers.
 *
 * The simulation intentionally remains one-way: particles sample the liquid
 * surface and projected velocity, but never enter the pressure solve. This is
 * the escaped-droplet part of the Chentanez and Mueller extension, not
 * particle-based thickening. A fixed ring keeps the path allocation- and
 * readback-free.
 */

export const SECONDARY_PARTICLE_STRIDE_BYTES = 48;
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
  attributes: vec4f,
}

struct ParticleState {
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
  dead.attributes.z = 0.0;
  particles[index] = dead;
}

@compute @workgroup_size(64)
fn updateParticles(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  if (index >= capacity()) { return; }
  var particle = particles[index];
  if (particle.attributes.z < 0.5 || particle.positionRadius.w <= 0.0) { return; }

  var position = particle.positionRadius.xyz;
  var velocity = particle.velocityAge.xyz;
  var age = particle.velocityAge.w + dt();
  let lifetime = particle.attributes.y;
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
  let seed = linear ^ u32(params.gravityAndSeed.w) ^ u32(params.fieldModes.w);
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
  let probability = clamp(dt() * params.fieldModes.z * (0.5 + outwardSpeed), 0.0, 0.72);
  if (random01(seed) > probability) { return; }

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
    let slot = atomicAdd(&particleState.cursor, 1u) % capacity();
    atomicAdd(&particleState.spawned, 1u);
    particles[slot] = Particle(
      vec4f(particlePosition, particleRadius),
      vec4f(particleVelocity, 0.0),
      vec4f(0.0, lifetime, 1.0, random01(particleSeed ^ 0x1b873593u))
    );
  }
}
`;

export const secondaryParticleRenderShader = /* wgsl */ `
struct ViewUniforms {
  viewport: vec4f,
  cameraPosition: vec4f,
  cameraTarget: vec4f,
  container: vec4f,
}

struct Particle {
  positionRadius: vec4f,
  velocityAge: vec4f,
  attributes: vec4f,
}

@group(0) @binding(0) var<uniform> view: ViewUniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
struct ParticleVertex {
  @builtin(position) clip: vec4f,
  @location(0) local: vec2f,
  @location(1) opacity: f32,
}

fn project(world: vec3f) -> vec4f {
  let forward = normalize(view.cameraTarget.xyz - view.cameraPosition.xyz);
  let right = normalize(cross(forward, vec3f(0.0, 1.0, 0.0)));
  let up = normalize(cross(right, forward));
  let relative = world - view.cameraPosition.xyz;
  let depth = dot(relative, forward);
  let aspect = view.viewport.x / max(view.viewport.y, 1.0);
  let ndc = vec2f(dot(relative, right) / (max(depth, 0.001) * aspect * 0.72), dot(relative, up) / (max(depth, 0.001) * 0.72));
  return vec4f(ndc * depth, clamp(depth / 50.0, 0.0, 1.0) * depth, depth);
}

@vertex
fn particleVertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> ParticleVertex {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  let particle = particles[instance];
  var output: ParticleVertex;
  output.local = corners[vertex];
  output.opacity = particle.attributes.z;
  if (particle.attributes.z < 0.5 || particle.positionRadius.w <= 0.0) {
    output.clip = vec4f(2.0, 2.0, 2.0, 1.0);
    output.opacity = 0.0;
    return output;
  }
  let forward = normalize(view.cameraTarget.xyz - view.cameraPosition.xyz);
  let right = normalize(cross(forward, vec3f(0.0, 1.0, 0.0)));
  let up = normalize(cross(right, forward));
  let world = particle.positionRadius.xyz + (right * output.local.x + up * output.local.y) * particle.positionRadius.w * 1.35;
  output.clip = project(world);
  if (output.clip.w <= 0.001) { output.clip = vec4f(2.0, 2.0, 2.0, 1.0); output.opacity = 0.0; }
  return output;
}

@fragment
fn particleFragment(input: ParticleVertex) -> @location(0) vec4f {
  let radius2 = dot(input.local, input.local);
  if (radius2 > 1.0 || input.opacity <= 0.0) { discard; }
  let edge = 1.0 - smoothstep(0.34, 1.0, radius2);
  let color = vec3f(0.62, 0.80, 0.90);
  let alpha = 0.46 * edge;
  let highlight = pow(max(0.0, 1.0 - radius2), 3.0);
  return vec4f(color + vec3f(0.08) * highlight, alpha * input.opacity);
}
`;

/**
 * Spray-only sphere impostors for the raster water interface buffers. The
 * optical composite cannot distinguish these interfaces from the extracted
 * Eulerian surface, so droplets inherit the same refraction, absorption,
 * environment reflection, Fresnel response, and rigid/glass occlusion.
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
  attributes: vec4f,
}

@group(0) @binding(0) var<uniform> view: ViewUniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct SphereVertex {
  @builtin(position) clip: vec4f,
  @location(0) local: vec2f,
  @location(1) @interpolate(flat) center: vec3f,
  @location(2) @interpolate(flat) radius: f32,
  @location(3) @interpolate(flat) enabled: f32,
}

struct InterfaceFragment {
  @location(0) position: vec4f,
  @location(1) normal: vec4f,
  @builtin(frag_depth) depth: f32,
}

fn cameraForward() -> vec3f { return normalize(view.cameraTarget.xyz - view.cameraPosition.xyz); }
fn cameraRight() -> vec3f { return normalize(cross(cameraForward(), vec3f(0.0, 1.0, 0.0))); }
fn cameraUp() -> vec3f { return normalize(cross(cameraRight(), cameraForward())); }

fn project(world: vec3f) -> vec4f {
  let forward = cameraForward();
  let relative = world - view.cameraPosition.xyz;
  let eyeDepth = dot(relative, forward);
  let aspect = view.viewport.x / max(view.viewport.y, 1.0);
  let ndc = vec2f(dot(relative, cameraRight()) / (max(eyeDepth, 0.001) * aspect * 0.72), dot(relative, cameraUp()) / (max(eyeDepth, 0.001) * 0.72));
  return vec4f(ndc * eyeDepth, clamp(eyeDepth / 50.0, 0.0, 1.0) * eyeDepth, eyeDepth);
}

@vertex
fn sphereVertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> SphereVertex {
  var corners = array<vec2f, 6>(
    vec2f(-1.05, -1.05), vec2f(1.05, -1.05), vec2f(-1.05, 1.05),
    vec2f(-1.05, 1.05), vec2f(1.05, -1.05), vec2f(1.05, 1.05)
  );
  let particle = particles[instance];
  var output: SphereVertex;
  output.local = corners[vertex];
  output.center = particle.positionRadius.xyz;
  output.radius = particle.positionRadius.w;
  output.enabled = particle.attributes.z;
  if (output.enabled < 0.5 || output.radius <= 0.0) {
    output.clip = vec4f(2.0, 2.0, 2.0, 1.0);
    output.enabled = 0.0;
    return output;
  }
  let world = output.center + (cameraRight() * output.local.x + cameraUp() * output.local.y) * output.radius;
  output.clip = project(world);
  if (output.clip.w <= 0.001) { output.clip = vec4f(2.0, 2.0, 2.0, 1.0); output.enabled = 0.0; }
  return output;
}

fn sphereInterface(input: SphereVertex, back: bool) -> InterfaceFragment {
  let radius2 = dot(input.local, input.local);
  if (input.enabled < 0.5 || radius2 > 1.0) { discard; }
  let z = sqrt(max(0.0, 1.0 - radius2));
  let facing = select(-1.0, 1.0, back);
  let localNormal = cameraRight() * input.local.x + cameraUp() * input.local.y + cameraForward() * (facing * z);
  let normal = normalize(localNormal);
  let world = input.center + normal * input.radius;
  let clip = project(world);
  return InterfaceFragment(vec4f(world, 1.0), vec4f(normal, 1.0), clamp(clip.z / max(clip.w, 0.001), 0.0, 1.0));
}

@fragment fn sphereFront(input: SphereVertex) -> InterfaceFragment { return sphereInterface(input, false); }
@fragment fn sphereBack(input: SphereVertex) -> InterfaceFragment { return sphereInterface(input, true); }
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
  private updatePipeline?: GPUComputePipeline;
  private spawnPipeline?: GPUComputePipeline;
  private step = 0;

  constructor(
    private readonly device: GPUDevice,
    private readonly grid: SecondaryParticleGrid,
    private readonly domain: SecondaryParticleDomain,
    source: SecondaryParticleSamplingSource,
    capacityValue = DEFAULT_SECONDARY_PARTICLE_CAPACITY,
    deferPipelineCompilation = false
  ) {
    const capacity = secondaryParticleCapacity(capacityValue);
    this.particles = device.createBuffer({
      label: `Secondary liquid particles (${capacity})`,
      size: capacity * SECONDARY_PARTICLE_STRIDE_BYTES,
      usage: GPUBufferUsage.STORAGE
    });
    this.state = device.createBuffer({ label: "Secondary particle ring state", size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.params = device.createBuffer({ label: "Secondary particle parameters", size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.state, 0, new Uint32Array(4));
    this.renderSource = { buffer: this.particles, capacity, strideBytes: SECONDARY_PARTICLE_STRIDE_BYTES };
    this.allocatedBytes = capacity * SECONDARY_PARTICLE_STRIDE_BYTES + 112;

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
    this.writeParameters(1 / 60, source);
    if (!deferPipelineCompilation) this.createPipelinesSync();
  }

  private descriptor(entryPoint: "updateParticles" | "spawnParticles"): GPUComputePipelineDescriptor {
    return { layout: this.pipelineLayout, compute: { module: this.shaderModule, entryPoint } };
  }

  private createPipelinesSync() {
    this.updatePipeline = this.device.createComputePipeline(this.descriptor("updateParticles"));
    this.spawnPipeline = this.device.createComputePipeline(this.descriptor("spawnParticles"));
  }

  async initializePipelines(onProgress: (label: string, completed: number, total: number) => void) {
    const definitions = [["Advecting secondary particles", "updateParticles"], ["Seeding escaped spray", "spawnParticles"]] as const;
    for (let index = 0; index < definitions.length; index += 1) {
      const [label, entryPoint] = definitions[index];
      onProgress(label, index, definitions.length);
      const pipeline = await this.device.createComputePipelineAsync(this.descriptor(entryPoint));
      if (entryPoint === "updateParticles") this.updatePipeline = pipeline;
      else this.spawnPipeline = pipeline;
      onProgress(label, index + 1, definitions.length);
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
      this.renderSource.capacity, outwardThreshold, 0.22, 0,
      source.fieldLayout === "restricted-tall-cell" ? 1 : 0, source.surfaceEncoding === "occupancy" ? 1 : 0, 7, this.step
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
  }

  destroy() {
    this.particles.destroy();
    this.state.destroy();
    this.params.destroy();
  }
}

export class SecondaryParticleRenderPipeline {
  private fallbackPipeline?: GPURenderPipeline;
  private opticalFrontPipeline?: GPURenderPipeline;
  private opticalBackPipeline?: GPURenderPipeline;
  private overlayBindGroup?: GPUBindGroup;
  private opticalBindGroup?: GPUBindGroup;
  private layout?: GPUBindGroupLayout;
  private source?: GPUSecondaryParticleSource;

  constructor(private readonly device: GPUDevice, private readonly format: GPUTextureFormat, private readonly uniformBuffer: GPUBuffer) {}

  async initialize() {
    const overlayModule = this.device.createShaderModule({ label: "Secondary liquid particle overlay", code: secondaryParticleRenderShader });
    const opticalModule = this.device.createShaderModule({ label: "Secondary liquid optical interfaces", code: secondaryParticleOpticalShader });
    this.layout = this.device.createBindGroupLayout({ label: "Secondary particle render bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } }
    ] });
    const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    const fallbackDescriptor = (label: string): GPURenderPipelineDescriptor => ({
      label, layout: pipelineLayout,
      vertex: { module: overlayModule, entryPoint: "particleVertex" },
      fragment: { module: overlayModule, entryPoint: "particleFragment", targets: [{
        format: this.format,
        blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" }
        }
      }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
    const opticalDescriptor = (label: string, entryPoint: "sphereFront" | "sphereBack"): GPURenderPipelineDescriptor => ({
      label, layout: pipelineLayout,
      vertex: { module: opticalModule, entryPoint: "sphereVertex" },
      fragment: { module: opticalModule, entryPoint, targets: [{ format: "rgba16float" }, { format: "rgba16float" }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }
    });
    [this.fallbackPipeline, this.opticalFrontPipeline, this.opticalBackPipeline] = await Promise.all([
      this.device.createRenderPipelineAsync(fallbackDescriptor("Spray droplet fallback overlay")),
      this.device.createRenderPipelineAsync(opticalDescriptor("Secondary spray front interfaces", "sphereFront")),
      this.device.createRenderPipelineAsync(opticalDescriptor("Secondary spray back interfaces", "sphereBack"))
    ]);
  }

  setSource(source: GPUSecondaryParticleSource | undefined) {
    if (source === this.source) return;
    this.source = source;
    const bindGroup = source && this.layout ? this.device.createBindGroup({ layout: this.layout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: source.buffer } }
    ] }) : undefined;
    this.overlayBindGroup = bindGroup;
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
    pass.draw(6, this.source.capacity);
    return true;
  }

  encode(encoder: GPUCommandEncoder, target: GPUTextureView, timestampWrites?: GPURenderPassTimestampWrites) {
    if (!this.fallbackPipeline || !this.overlayBindGroup || !this.source) return false;
    const pass = encoder.beginRenderPass({ label: "Render fallback spray droplets", colorAttachments: [{ view: target, loadOp: "load", storeOp: "store" }], ...(timestampWrites ? { timestampWrites } : {}) });
    pass.setPipeline(this.fallbackPipeline);
    pass.setBindGroup(0, this.overlayBindGroup);
    pass.draw(6, this.source.capacity);
    pass.end();
    return true;
  }
}
