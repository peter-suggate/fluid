/** Optical rendering for the retained secondary-particle source ABI. */
import { cameraApertureShaderLibrary } from "./webgpu-camera";

export const SECONDARY_PARTICLE_STRIDE_BYTES = 64;

export interface GPUSecondaryParticleSource {
  readonly buffer: GPUBuffer;
  readonly capacity: number;
  readonly strideBytes: typeof SECONDARY_PARTICLE_STRIDE_BYTES;
  /** Draws only the ring prefix that has ever received a particle. */
  readonly indirectBuffer?: GPUBuffer;
  readonly indirectOffset?: number;
}

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
${cameraApertureShaderLibrary("view")}
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
  let ndc = vec2f(dot(relative, cameraRight()) / (max(eyeDepth, 0.001) * aspect * cameraTanHalfFov()), dot(relative, cameraUp()) / (max(eyeDepth, 0.001) * cameraTanHalfFov()));
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
      fragment: { module: opticalModule, entryPoint, targets: [{ format: "rgba32float" }, { format: "rgba16float" }] },
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
