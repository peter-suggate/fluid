import { WATER_INTERFACE_DEPTH, waterInterfaceDepthWGSL } from "./water-interface-depth";
import { cameraApertureShaderLibrary } from "./webgpu-camera";
import type { GPUFluidParticleSource } from "./webgpu-particle-overlay";

/**
 * The particle display's Water view: a particle method's own records drawn as
 * water by the water renderer itself, as a screen-space fluid front.
 *
 * The records are drawn as the eye depth of their nearest sphere surface,
 * that depth is smoothed by a depth-aware Gaussian a few sphere radii wide (a
 * crowd of samples fuses into a sheet, one drop stays a drop, and samples at
 * different depths stay apart), and the smoothed surface is written into the
 * water's front interface attachments beside the level set's triangles, under
 * the same depth rule (the nearest front wins). Whatever surface mode is
 * chosen then shades it as it shades the liquid: under Shade, a sheet or drop
 * reflects the scene and refracts what is behind it.
 *
 * Raw spheres are not drawn as interfaces: where they overlap, a bed of beads
 * shades as lace, and the refracted ray of one sphere leaves through another's
 * normal. Nor is a particle back face drawn. Screen space holds no back
 * surface; samples near the level set give fronts but have none, so a back
 * would exist only in dashes across a splash, and the shader's exit search,
 * hopping between them, scatters its refraction into the sky. Without one, a
 * particle front ends at the body's own back where it lies over the body, and
 * at the shader's one-cell exit where it is a drop in the air.
 *
 * Every live record that could show takes part. A sample wholly inside the
 * level set (its centre more than a radius in) gives no front, since the
 * body's own front is nearer; skipping it is what keeps the cost to the
 * surface's samples. A ballistic sample has left every grid cell, so its
 * depth is not kept, and it always draws.
 */

/** Sphere radius over the solver's display radius: neighbours overlap, so a sheet of samples reads as a sheet. */
export const PARTICLE_WATER_RADIUS_SCALE = 1.5;
/** Gaussian width of the surface smoothing, in sphere radii. */
export const PARTICLE_WATER_SMOOTHING_RADII = 1.5;
/** Depth difference, in sphere radii, over which two samples stop being one surface. */
export const PARTICLE_WATER_RANGE_RADII = 3;
/**
 * Longest smoothing reach in pixels. The kernel takes at most six taps a side
 * over a disc, so this keeps it dense (taps at most a pixel and a half apart):
 * a sparser kernel aliases at every silhouette into speckle.
 */
export const PARTICLE_WATER_MAX_REACH_PX = 9;

const viewUniformsWGSL = /* wgsl */ `
struct ViewUniforms {
  viewport: vec4f,
  cameraPosition: vec4f,
  cameraTarget: vec4f,
  container: vec4f, cameraBasisPadding:array<vec4f,22>, cameraBasisRight:vec4f }
@group(0) @binding(0) var<uniform> view: ViewUniforms;
${cameraApertureShaderLibrary("view")}
fn cameraForward() -> vec3f { return normalize(view.cameraTarget.xyz - view.cameraPosition.xyz); }
fn cameraRight() -> vec3f { return cameraRightDirection(cameraForward()); }
fn cameraUp() -> vec3f { return normalize(cross(cameraRight(), cameraForward())); }
fn cameraAspect() -> f32 { return view.viewport.x / max(view.viewport.y, 1.0); }
`;

/** Records to eye depth: one camera-facing quad a record, ray-traced to its sphere. */
export const particleWaterSphereShader = /* wgsl */ `
${waterInterfaceDepthWGSL}
${viewUniformsWGSL}
struct Records {
  // x stride in floats, y capacity, z the signed-depth float (position units,
  // negative inside the liquid), w one when records keep that depth
  words: vec4u,
  // xyz metres per position unit, w sphere radius in metres
  scale: vec4f,
  // xyz world position of the tank's minimum corner, w one to take it from the solver's frame
  origin: vec4f,
  // x the ballistic flag's float, its top bit set when the flag is that float's sign bit
  flags: vec4u,
}
@group(0) @binding(1) var<uniform> records: Records;
@group(0) @binding(2) var<storage, read> particles: array<f32>;
struct ParticleFrame { dims: vec4f, h: vec4f, origin: vec4f }
@group(0) @binding(3) var<uniform> particleFrame: ParticleFrame;

struct SphereVertex {
  @builtin(position) clip: vec4f,
  @location(0) rayPoint: vec3f,
  @location(1) @interpolate(flat) center: vec3f,
  @location(2) @interpolate(flat) radius: f32,
}
struct DepthFragment {
  @location(0) eyeDepth: vec4f,
  @builtin(frag_depth) depth: f32,
}

fn project(world: vec3f) -> vec4f {
  let relative = world - view.cameraPosition.xyz;
  let eyeDepth = dot(relative, cameraForward());
  let ndc = vec2f(dot(relative, cameraRight()) / (max(eyeDepth, 0.001) * cameraAspect() * cameraTanHalfFov()),
    dot(relative, cameraUp()) / (max(eyeDepth, 0.001) * cameraTanHalfFov()));
  return vec4f(ndc * eyeDepth, WATER_INTERFACE_NEAR_M, eyeDepth);
}

@vertex fn sphereVertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> SphereVertex {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  var output: SphereVertex;
  output.clip = vec4f(2.0, 2.0, 2.0, 1.0);
  output.rayPoint = vec3f(0.0);
  output.center = vec3f(0.0);
  output.radius = 0.0;
  if (instance >= records.words.y) { return output; }
  let at = records.words.x * instance;
  if (!(particles[at + 3u] > 0.0)) { return output; }
  let radius = records.scale.w;
  if (records.words.w != 0u) {
    let flag = particles[at + (records.flags.x & 0x7fffffffu)];
    let ballistic = select(flag == 1.0, (bitcast<u32>(flag) >> 31u) != 0u, (records.flags.x & 0x80000000u) != 0u);
    let reach = radius / max(min(records.scale.x, min(records.scale.y, records.scale.z)), 1e-9);
    if (!ballistic && particles[at + records.words.z] < -reach) { return output; }
  }
  let local = vec3f(particles[at], particles[at + 1u], particles[at + 2u]);
  let origin = select(records.origin.xyz, particleFrame.origin.xyz, records.origin.w > 0.5);
  let center = local * records.scale.xyz + origin;
  let centerDepth = dot(center - view.cameraPosition.xyz, cameraForward());
  if (centerDepth - radius <= WATER_INTERFACE_NEAR_M) { return output; }
  // A sphere's silhouette is wider than its radius off-axis.
  let pad = 1.035 + radius / max(centerDepth, 0.02);
  output.rayPoint = center + (cameraRight() * corners[vertex].x + cameraUp() * corners[vertex].y) * radius * pad;
  output.clip = project(output.rayPoint);
  output.center = center;
  output.radius = radius;
  return output;
}

@fragment fn sphereDepth(input: SphereVertex) -> DepthFragment {
  if (input.radius <= 0.0) { discard; }
  let origin = view.cameraPosition.xyz;
  let direction = normalize(input.rayPoint - origin);
  let offset = origin - input.center;
  let b = dot(offset, direction);
  let discriminant = b * b - (dot(offset, offset) - input.radius * input.radius);
  if (discriminant < 0.0) { discard; }
  let near = -b - sqrt(discriminant);
  if (near <= 1e-4) { discard; }
  let eyeDepth = near * dot(direction, cameraForward());
  return DepthFragment(vec4f(eyeDepth, 0.0, 0.0, 0.0), clamp(WATER_INTERFACE_NEAR_M / max(eyeDepth, 1e-4), 0.0, 1.0));
}
`;

/** Full-screen passes: the depth-aware smoothing, then the front it implies. */
export const particleWaterScreenShader = /* wgsl */ `
${waterInterfaceDepthWGSL}
${viewUniformsWGSL}
// x sphere radius (m), y smoothing width (radii), z depth range (radii), w longest reach (px)
@group(0) @binding(1) var<uniform> smoothing: vec4f;
@group(0) @binding(2) var source: texture_2d<f32>;

@vertex fn screenVertex(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4f {
  let corner = vec2f(f32((vertex << 1u) & 2u), f32(vertex & 2u));
  return vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
}

fn eyeDepthAt(pixel: vec2i) -> f32 {
  let size = vec2i(textureDimensions(source));
  if (any(pixel < vec2i(0)) || any(pixel >= size)) { return 0.0; }
  return textureLoad(source, pixel, 0).x;
}

// One two-dimensional pass: a range weight is not separable, and split across
// two passes it leaves streaks along the second pass's axis wherever samples
// overlap. Empty pixels add nothing, so silhouettes neither grow nor bleed.
@fragment fn smoothDepth(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let pixel = vec2i(position.xy);
  let center = eyeDepthAt(pixel);
  if (center <= 0.0) { return vec4f(0.0); }
  let focal = 0.5 * f32(textureDimensions(source).y) / max(cameraTanHalfFov(), 1e-3);
  let sigma = smoothing.y * smoothing.x * focal / center;
  let reach = min(ceil(2.0 * sigma), smoothing.w);
  if (reach < 1.0) { return vec4f(center, 0.0, 0.0, 0.0); }
  let taps = i32(min(reach, 6.0));
  let stride = reach / f32(taps);
  let range = smoothing.z * smoothing.x;
  var sum = 0.0;
  var weights = 0.0;
  for (var y = -taps; y <= taps; y += 1) {
    for (var x = -taps; x <= taps; x += 1) {
      if (x * x + y * y > taps * taps) { continue; }
      let offset = vec2f(f32(x), f32(y)) * stride;
      let depth = eyeDepthAt(pixel + vec2i(round(offset)));
      if (depth <= 0.0) { continue; }
      let gap = (depth - center) / range;
      let weight = exp(-0.5 * dot(offset, offset) / (sigma * sigma) - gap * gap);
      sum += depth * weight;
      weights += weight;
    }
  }
  return vec4f(sum / max(weights, 1e-12), 0.0, 0.0, 0.0);
}

/** The unnormalised ray through a pixel, one unit deep along the view axis. */
fn rayThrough(pixel: vec2i) -> vec3f {
  let size = vec2f(textureDimensions(source));
  let uv = (vec2f(pixel) + 0.5) / size;
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  return cameraForward() + (cameraRight() * ndc.x * cameraAspect() + cameraUp() * ndc.y) * cameraTanHalfFov();
}
fn worldAt(pixel: vec2i) -> vec4f {
  let depth = eyeDepthAt(pixel);
  if (depth <= 0.0) { return vec4f(0.0); }
  return vec4f(view.cameraPosition.xyz + rayThrough(pixel) * depth, 1.0);
}
/** The surface tangent along one screen axis, from whichever neighbour lies nearer in depth. */
fn tangentAlong(pixel: vec2i, axis: vec2i, center: vec3f) -> vec3f {
  let ahead = worldAt(pixel + axis);
  let behind = worldAt(pixel - axis);
  let forward = cameraForward();
  if (ahead.w > 0.5 && (behind.w < 0.5 || abs(dot(ahead.xyz - center, forward)) <= abs(dot(center - behind.xyz, forward)))) { return ahead.xyz - center; }
  if (behind.w > 0.5) { return center - behind.xyz; }
  return vec3f(0.0);
}

struct InterfaceFragment {
  @location(0) position: vec4f,
  @location(1) normal: vec4f,
  @builtin(frag_depth) depth: f32,
}
@fragment fn frontInterface(@builtin(position) position: vec4f) -> InterfaceFragment {
  let pixel = vec2i(position.xy);
  let depth = eyeDepthAt(pixel);
  if (depth <= 0.0) { discard; }
  let ray = rayThrough(pixel);
  let center = view.cameraPosition.xyz + ray * depth;
  var normal = cross(tangentAlong(pixel, vec2i(1, 0), center), tangentAlong(pixel, vec2i(0, 1), center));
  if (dot(normal, normal) < 1e-20) { normal = -ray; }
  normal = normalize(normal);
  if (dot(normal, ray) > 0.0) { normal = -normal; }
  // Surface samples lie on the level set's own front, so the two fronts are
  // coplanar there and the depth test would flip between them pixel by pixel.
  // The particle front competes a sphere radius behind where it is: it wins
  // only where it stands clear of the body (spray, sheets, crests), and leaves
  // the body's own smooth front wherever it merely traces it.
  return InterfaceFragment(vec4f(center, 1.0), vec4f(normal, 1.0), clamp(WATER_INTERFACE_NEAR_M / max(depth + smoothing.x, 1e-4), 0.0, 1.0));
}
`;

const RECORDS_BYTES = 64;

async function checkedModule(device: GPUDevice, label: string, code: string) {
  const module = device.createShaderModule({ label, code });
  const errors = (await module.getCompilationInfo()).messages.filter((message) => message.type === "error");
  if (errors.length > 0) throw new Error(`${label}:\n${errors.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join("\n")}`);
  return module;
}

/** Per frame: a sphere-depth pass and a smoothing pass, then one full-screen draw in the water's front interface pass. */
export class ParticleWaterInterface {
  private sphereLayout?: GPUBindGroupLayout;
  private screenLayout?: GPUBindGroupLayout;
  private spherePipeline?: GPURenderPipeline;
  private smoothPipeline?: GPURenderPipeline;
  private interfacePipeline?: GPURenderPipeline;
  private readonly records: GPUBuffer;
  private readonly smoothing: GPUBuffer;
  private readonly indirect: GPUBuffer;
  private readonly fallbackFrame: GPUBuffer;
  private readonly data = new ArrayBuffer(RECORDS_BYTES);
  private source?: GPUFluidParticleSource;
  private sphereGroup?: GPUBindGroup;
  private sphereGroupKey?: readonly [GPUBuffer, GPUBuffer | undefined];
  private container: readonly [number, number, number] = [1, 1, 1];
  private targets?: { width: number; height: number; depth: GPUTexture; raw: GPUTexture; smooth: GPUTexture; smoothGroup: GPUBindGroup; interfaceGroup: GPUBindGroup };
  private drawable = false;

  constructor(private readonly device: GPUDevice, private readonly viewUniforms: GPUBuffer) {
    this.records = device.createBuffer({ label: "Particle water records", size: RECORDS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.smoothing = device.createBuffer({ label: "Particle water smoothing", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.indirect = device.createBuffer({ label: "Particle water draw arguments", size: 16, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.indirect, 0, new Uint32Array([6, 0, 0, 0]));
    this.fallbackFrame = device.createBuffer({ label: "Particle water frame fallback", size: 48, usage: GPUBufferUsage.UNIFORM });
  }

  async initialize(): Promise<void> {
    const [sphereModule, screenModule] = await Promise.all([
      checkedModule(this.device, "Particle water spheres", particleWaterSphereShader),
      checkedModule(this.device, "Particle water surface", particleWaterScreenShader),
    ]);
    this.sphereLayout = this.device.createBindGroupLayout({ label: "Particle water sphere bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
    ] });
    this.screenLayout = this.device.createBindGroupLayout({ label: "Particle water surface bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
    ] });
    const screenLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.screenLayout] });
    const screenVertex = { module: screenModule, entryPoint: "screenVertex" };
    const nearestWins = { format: WATER_INTERFACE_DEPTH.format, depthWriteEnabled: true, depthCompare: WATER_INTERFACE_DEPTH.compare };
    [this.spherePipeline, this.smoothPipeline, this.interfacePipeline] = await Promise.all([
      this.device.createRenderPipelineAsync({
        label: "Particle water depth", layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.sphereLayout] }),
        vertex: { module: sphereModule, entryPoint: "sphereVertex" },
        fragment: { module: sphereModule, entryPoint: "sphereDepth", targets: [{ format: "r32float" }] },
        primitive: { topology: "triangle-list", cullMode: "none" }, depthStencil: nearestWins,
      }),
      this.device.createRenderPipelineAsync({
        label: "Particle water smoothing", layout: screenLayout, vertex: screenVertex,
        fragment: { module: screenModule, entryPoint: "smoothDepth", targets: [{ format: "r32float" }] },
        primitive: { topology: "triangle-list" },
      }),
      // The water interface attachments: world position, then the normal with the film in alpha.
      this.device.createRenderPipelineAsync({
        label: "Particle water front interfaces", layout: screenLayout, vertex: screenVertex,
        fragment: { module: screenModule, entryPoint: "frontInterface", targets: [{ format: "rgba32float" }, { format: "rgba16float" }] },
        primitive: { topology: "triangle-list" }, depthStencil: nearestWins,
      }),
    ]);
  }

  /** The records to draw this frame; absent draws none. */
  setSource(source: GPUFluidParticleSource | undefined, container_m: readonly [number, number, number]): void {
    this.source = source;
    this.container = container_m;
  }

  private ensureTargets(width: number, height: number) {
    if (this.targets?.width === width && this.targets.height === height) return this.targets;
    this.destroyTargets();
    const image = (label: string) => this.device.createTexture({ label, size: [width, height], format: "r32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    const raw = image("Particle water depth");
    const smooth = image("Particle water smoothed depth");
    const depth = this.device.createTexture({ label: "Particle water depth test", size: [width, height], format: WATER_INTERFACE_DEPTH.format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const group = (input: GPUTexture) => this.device.createBindGroup({ layout: this.screenLayout!, entries: [
      { binding: 0, resource: { buffer: this.viewUniforms } }, { binding: 1, resource: { buffer: this.smoothing } }, { binding: 2, resource: input.createView() },
    ] });
    this.targets = { width, height, depth, raw, smooth, smoothGroup: group(raw), interfaceGroup: group(smooth) };
    return this.targets;
  }

  private destroyTargets() {
    const targets = this.targets;
    if (!targets) return;
    for (const texture of [targets.depth, targets.raw, targets.smooth]) texture.destroy();
    this.targets = undefined;
  }

  /**
   * Draws and smooths this frame's particle front at the interface
   * attachments' size; call before the interface passes. False draws none.
   */
  prepare(encoder: GPUCommandEncoder, width: number, height: number): boolean {
    this.drawable = false;
    const source = this.source;
    if (!source || !this.sphereLayout || !this.spherePipeline || !this.smoothPipeline || source.capacity === 0 || width <= 0 || height <= 0) return false;
    const radius = source.radius_m * PARTICLE_WATER_RADIUS_SCALE;
    const u = new Uint32Array(this.data), f = new Float32Array(this.data);
    u.set([source.strideFloats, source.capacity, source.grid?.depthFloat ?? 0, source.grid ? 1 : 0], 0);
    f.set([...source.positionScale_m, radius], 4);
    f.set([-0.5 * this.container[0], 0, -0.5 * this.container[2], source.frame ? 1 : 0], 8);
    u.set([source.grid ? (source.grid.ballisticFloat | (source.grid.ballisticSignBit ? 0x80000000 : 0)) >>> 0 : 0, 0, 0, 0], 12);
    this.device.queue.writeBuffer(this.records, 0, this.data);
    this.device.queue.writeBuffer(this.smoothing, 0, new Float32Array([radius, PARTICLE_WATER_SMOOTHING_RADII, PARTICLE_WATER_RANGE_RADII, PARTICLE_WATER_MAX_REACH_PX]));
    if (source.liveCount) encoder.copyBufferToBuffer(source.liveCount.buffer, source.liveCount.byteOffset, this.indirect, 4, 4);
    else this.device.queue.writeBuffer(this.indirect, 4, new Uint32Array([source.capacity]));
    const frame = source.frame?.buffer;
    if (!this.sphereGroup || this.sphereGroupKey?.[0] !== source.buffer || this.sphereGroupKey?.[1] !== frame) {
      this.sphereGroup = this.device.createBindGroup({ label: "Particle water sphere bind group", layout: this.sphereLayout, entries: [
        { binding: 0, resource: { buffer: this.viewUniforms } },
        { binding: 1, resource: { buffer: this.records } },
        { binding: 2, resource: { buffer: source.buffer } },
        { binding: 3, resource: source.frame ?? { buffer: this.fallbackFrame } },
      ] });
      this.sphereGroupKey = [source.buffer, frame];
    }
    const targets = this.ensureTargets(width, height);
    const spheres = encoder.beginRenderPass({ label: "Particle water depth",
      colorAttachments: [{ view: targets.raw.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }],
      depthStencilAttachment: { view: targets.depth.createView(), depthClearValue: WATER_INTERFACE_DEPTH.clearValue, depthLoadOp: "clear", depthStoreOp: "discard" } });
    spheres.setPipeline(this.spherePipeline);
    spheres.setBindGroup(0, this.sphereGroup);
    spheres.drawIndirect(this.indirect, 0);
    spheres.end();
    const smooth = encoder.beginRenderPass({ label: "Particle water smoothing",
      colorAttachments: [{ view: targets.smooth.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }] });
    smooth.setPipeline(this.smoothPipeline);
    smooth.setBindGroup(0, targets.smoothGroup);
    smooth.draw(3);
    smooth.end();
    this.drawable = true;
    return true;
  }

  /** One full-screen draw inside the water's front interface pass that `prepare` readied this frame; a back pass takes none. */
  encode(pass: GPURenderPassEncoder, side: "front" | "back"): void {
    if (side !== "front" || !this.drawable || !this.interfacePipeline || !this.targets) return;
    pass.setPipeline(this.interfacePipeline);
    pass.setBindGroup(0, this.targets.interfaceGroup);
    pass.draw(3);
  }

  destroy(): void {
    this.destroyTargets();
    this.records.destroy(); this.smoothing.destroy(); this.indirect.destroy(); this.fallbackFrame.destroy();
    this.sphereGroup = undefined; this.source = undefined;
  }
}
