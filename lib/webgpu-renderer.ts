import { cameraBasis, cameraPosition, dot } from "./math";
import type { CameraState, SceneDescription, ViewMode } from "./model";
import { boundingRadius, type RigidBodyState } from "./rigid-body";
import type { EulerianRenderState } from "./eulerian-solver";
import type { GPUEulerianInfo, GPURigidLoad, GPUQuality } from "./webgpu-eulerian";
import { getMethod, type GPUSolverInstance, type MethodParamValues } from "./methods";
import { GridOverlayPipeline } from "./webgpu-grid-overlay";
import { requiredFluidDeviceLimits } from "./webgpu-device-limits";
import { RasterWaterPipeline } from "./webgpu-water-pipeline";
import { environmentIndex, type EnvironmentId, defaultEnvironmentId } from "./environments";
import { environmentShaderLibrary } from "./webgpu-environments";
import { MAX_TERRAIN_FEATURES, TERRAIN_DEFAULT_FLAT, TERRAIN_UNION_EXPONENT, sceneHasTerrain } from "./terrain";
import { gpuBatchDepth } from "./simulation/gpu-clock";
import { SecondaryParticleRenderPipeline } from "./webgpu-secondary-particles";
import { SparseVoxelDebugRenderer, type VoxelRenderMode } from "./webgpu-voxel-debug";
import { unifiedLightingShaderLibrary } from "./webgpu-lighting";
import { CAMERA_TAN_HALF_FOV } from "./webgpu-camera";

export type SimulationBackend = "webgpu" | "cpu-reference";
export type WaterRenderMode = "rasterized" | "ray-marched";

/** Column-major right-handed world-to-WebGPU-clip transform for voxel raster passes. */
export function voxelViewProjectionMatrix(camera: CameraState, aspect: number, near = 0.01, far = 100): Float32Array {
  const basis = cameraBasis(camera), position = basis.position;
  const view = new Float32Array([
    basis.right.x, basis.up.x, -basis.forward.x, 0,
    basis.right.y, basis.up.y, -basis.forward.y, 0,
    basis.right.z, basis.up.z, -basis.forward.z, 0,
    -dot(basis.right, position), -dot(basis.up, position), dot(basis.forward, position), 1
  ]);
  const safeNear = Math.max(1e-4, near), safeFar = Math.max(safeNear + 1, far);
  const focal = 1 / CAMERA_TAN_HALF_FOV;
  const projection = new Float32Array([
    focal / Math.max(1e-4, aspect), 0, 0, 0,
    0, focal, 0, 0,
    0, 0, safeFar / (safeNear - safeFar), -1,
    0, 0, safeNear * safeFar / (safeNear - safeFar), 0
  ]);
  const result = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) for (let row = 0; row < 4; row += 1) {
    let value = 0;
    for (let index = 0; index < 4; index += 1) value += projection[index * 4 + row] * view[column * 4 + index];
    result[column * 4 + row] = value;
  }
  return result;
}

/**
 * Debug cross-section of the solver grid, after Chentanez & Mueller Fig. 2:
 * teal tall cells, outlined regular cells with centre sample dots, and the
 * tall cell's top/bottom subcell samples. Uniform grids render as an
 * all-regular band. `position` selects the slice layer as a 0..1 fraction.
 *
 * `mode` recolors represented cells from GPU-resident fields: "cfl" shows the
 * per-cell component CFL at the solver's substep dt (the quantity whose
 * maximum picks the adaptive substep count), "speed" the velocity magnitude
 * normalized by the last reported liquid maximum. Both sample live solver
 * textures in the overlay shader — no readback is involved.
 */
export type GridOverlayMode = "structure" | "resolution" | "optical" | "cfl" | "speed" | "phi" | "divergence" | "pressure" | "projection" | "representation";

export interface GridOverlayConfig {
  axis: "off" | "z" | "x" | "y";
  position: number;
  mode?: GridOverlayMode;
}

/** Everything the renderer needs to know about the selected method. */
export interface SimulationRunConfig {
  methodId: string;
  quality: GPUQuality;
  values: MethodParamValues;
}

export type GPUStatus =
  | { state: "initializing"; label: string; phase?: string; completed?: number; total?: number; startedAt_ms?: number }
  | { state: "ready"; label: string; adapter: string }
  | { state: "unavailable"; label: string }
  | { state: "lost"; label: string };

export interface RendererFrameMetrics {
  cpuFrame_ms: number;
  cpuPhysicsSubmit_ms: number;
  cpuDataUpload_ms: number;
  cpuRenderEncode_ms: number;
  gpuRender_ms?: number;
  gpuSurfaceExtraction_ms?: number;
  gpuDryScene_ms?: number;
  gpuInterfaces_ms?: number;
  gpuSprayFront_ms?: number;
  gpuSprayBack_ms?: number;
  gpuSprayRender_ms?: number;
  gpuOpticalComposite_ms?: number;
  gpuUpscale_ms?: number;
  methodId?: string;
  waterRenderMode?: WaterRenderMode;
  gpuRenderTimestampAvailable?: boolean;
}

export interface RenderStageTimings {
  total_ms: number;
  surfaceExtraction_ms?: number;
  dryScene_ms?: number;
  interfaces_ms?: number;
  sprayFront_ms: number;
  sprayBack_ms: number;
  sprayRender_ms: number;
  opticalComposite_ms: number;
  upscale_ms: number;
}

export function decodeRenderStageTimestamps(times: ArrayLike<bigint>, rasterized: boolean, surfaceUpdated: boolean, sprayRendered = true): RenderStageTimings {
  if (times.length < 16) throw new Error("Render timestamp sample must contain 16 query values");
  const duration = (start: number, end: number) => {
    const milliseconds = Number(times[end] - times[start]) / 1e6;
    return Number.isFinite(milliseconds) && milliseconds >= 0 && milliseconds < 10_000 ? milliseconds : 0;
  };
  const upscale_ms = duration(10, 11);
  const sprayFront_ms = sprayRendered ? duration(12, 13) : 0;
  const sprayBack_ms = sprayRendered && rasterized ? duration(14, 15) : 0;
  const sprayRender_ms = sprayFront_ms + sprayBack_ms;
  if (!rasterized) {
    const opticalComposite_ms = duration(0, 1);
    return { total_ms: opticalComposite_ms + sprayRender_ms + upscale_ms, sprayFront_ms, sprayBack_ms, sprayRender_ms, opticalComposite_ms, upscale_ms };
  }
  const surfaceExtraction_ms = surfaceUpdated ? duration(0, 1) : undefined;
  const dryScene_ms = duration(2, 3);
  const interfaces_ms = duration(4, 5) + duration(6, 7);
  const opticalComposite_ms = duration(8, 9);
  return {
    total_ms: (surfaceExtraction_ms ?? 0) + dryScene_ms + interfaces_ms + sprayRender_ms + opticalComposite_ms + upscale_ms,
    surfaceExtraction_ms, dryScene_ms, interfaces_ms, sprayFront_ms, sprayBack_ms, sprayRender_ms, opticalComposite_ms, upscale_ms
  };
}

const shader = /* wgsl */ `
struct Uniforms {
  viewport: vec4f,
  cameraPosition: vec4f,
  cameraTarget: vec4f,
  container: vec4f,
  options: vec4f,
  gridInfo: vec4f,
  debug: vec4f,
  environment: vec4f,
  terrainMeta: vec4f,
  terrainFeatures: array<vec4f, 16>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct BodyGPU {
  positionRadius: vec4f,
  halfSizeShape: vec4f,
  orientation: vec4f,
  colorSelected: vec4f,
}

@group(0) @binding(1) var<storage, read> bodies: array<BodyGPU, 12>;
@group(0) @binding(2) var fluidField: texture_3d<f32>;
@group(0) @binding(3) var tallCellBases: texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vertexMain(@builtin(vertex_index) index: u32) -> VertexOutput {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0)
  );
  var result: VertexOutput;
  result.position = vec4f(positions[index], 0.0, 1.0);
  result.uv = positions[index] * 0.5 + 0.5;
  return result;
}

fn boxIntersection(ro: vec3f, rd: vec3f, boundsMin: vec3f, boundsMax: vec3f) -> vec2f {
  let inverse = 1.0 / rd;
  let t0 = (boundsMin - ro) * inverse;
  let t1 = (boundsMax - ro) * inverse;
  let near3 = min(t0, t1);
  let far3 = max(t0, t1);
  return vec2f(max(max(near3.x, near3.y), near3.z), min(min(far3.x, far3.y), far3.z));
}

${environmentShaderLibrary}
${unifiedLightingShaderLibrary}

fn gridLine(value: vec2f, scale: f32) -> f32 {
  let g = abs(sin(value * 3.14159265 / max(scale, 0.0001)));
  return 1.0 - smoothstep(0.0, 0.085, min(g.x, g.y));
}

struct BodyHit {
  t: f32,
  normal: vec3f,
  color: vec3f,
  selected: f32,
}

fn quatRotate(q: vec4f, v: vec3f) -> vec3f {
  let uv = cross(q.yzw, v);
  let uuv = cross(q.yzw, uv);
  return v + 2.0 * (q.x * uv + uuv);
}

fn quatInverseRotate(q: vec4f, v: vec3f) -> vec3f {
  return quatRotate(vec4f(q.x, -q.yzw), v);
}

fn sphereLocalHit(ro: vec3f, rd: vec3f, center: vec3f, radius: f32) -> vec4f {
  let oc = ro - center;
  let b = dot(oc, rd);
  let discriminant = b * b - dot(oc, oc) + radius * radius;
  if (discriminant < 0.0) { return vec4f(1e20, 0.0, 1.0, 0.0); }
  let root = sqrt(discriminant);
  var t = -b - root;
  if (t <= 0.0001) { t = -b + root; }
  if (t <= 0.0001) { return vec4f(1e20, 0.0, 1.0, 0.0); }
  return vec4f(t, normalize(ro + rd * t - center));
}

fn cylinderLocalHit(ro: vec3f, rd: vec3f, radius: f32, halfHeight: f32, capped: bool) -> vec4f {
  var best = vec4f(1e20, 0.0, 1.0, 0.0);
  let a = rd.x * rd.x + rd.z * rd.z;
  if (a > 1e-8) {
    let b = ro.x * rd.x + ro.z * rd.z;
    let c = ro.x * ro.x + ro.z * ro.z - radius * radius;
    let discriminant = b * b - a * c;
    if (discriminant >= 0.0) {
      var t = (-b - sqrt(discriminant)) / a;
      if (t <= 0.0001) { t = (-b + sqrt(discriminant)) / a; }
      let y = ro.y + rd.y * t;
      if (t > 0.0001 && abs(y) <= halfHeight) {
        let p = ro + rd * t;
        best = vec4f(t, normalize(vec3f(p.x, 0.0, p.z)));
      }
    }
  }
  if (capped && abs(rd.y) > 1e-8) {
    let tTop = (halfHeight - ro.y) / rd.y;
    let pTop = ro + rd * tTop;
    if (tTop > 0.0001 && tTop < best.x && dot(pTop.xz, pTop.xz) <= radius * radius) { best = vec4f(tTop, 0.0, 1.0, 0.0); }
    let tBottom = (-halfHeight - ro.y) / rd.y;
    let pBottom = ro + rd * tBottom;
    if (tBottom > 0.0001 && tBottom < best.x && dot(pBottom.xz, pBottom.xz) <= radius * radius) { best = vec4f(tBottom, 0.0, -1.0, 0.0); }
  }
  return best;
}

fn intersectBody(ro: vec3f, rd: vec3f, body: BodyGPU) -> vec4f {
  let localOrigin = quatInverseRotate(body.orientation, ro - body.positionRadius.xyz);
  let localDirection = quatInverseRotate(body.orientation, rd);
  let shape = i32(round(body.halfSizeShape.w));
  var localHit = vec4f(1e20, 0.0, 1.0, 0.0);
  if (shape == 0) {
    localHit = sphereLocalHit(localOrigin, localDirection, vec3f(0.0), body.halfSizeShape.x);
  } else if (shape == 1) {
    let boxHit = boxIntersection(localOrigin, localDirection, -body.halfSizeShape.xyz, body.halfSizeShape.xyz);
    var t = boxHit.x;
    if (t <= 0.0001) { t = boxHit.y; }
    if (t > 0.0001 && boxHit.x <= boxHit.y) {
      let p = localOrigin + localDirection * t;
      let q = abs(p / max(body.halfSizeShape.xyz, vec3f(1e-6)));
      var n = vec3f(0.0, 0.0, sign(p.z));
      if (q.x >= q.y && q.x >= q.z) { n = vec3f(sign(p.x), 0.0, 0.0); }
      else if (q.y >= q.z) { n = vec3f(0.0, sign(p.y), 0.0); }
      localHit = vec4f(t, n);
    }
  } else if (shape == 2) {
    let side = cylinderLocalHit(localOrigin, localDirection, body.halfSizeShape.x, body.halfSizeShape.y, false);
    let upper = sphereLocalHit(localOrigin, localDirection, vec3f(0.0, body.halfSizeShape.y, 0.0), body.halfSizeShape.x);
    let lower = sphereLocalHit(localOrigin, localDirection, vec3f(0.0, -body.halfSizeShape.y, 0.0), body.halfSizeShape.x);
    localHit = side;
    if (upper.x < localHit.x) { localHit = upper; }
    if (lower.x < localHit.x) { localHit = lower; }
  } else {
    localHit = cylinderLocalHit(localOrigin, localDirection, body.halfSizeShape.x, body.halfSizeShape.y, true);
  }
  return vec4f(localHit.x, quatRotate(body.orientation, localHit.yzw));
}

fn nearestBody(ro: vec3f, rd: vec3f) -> BodyHit {
  var result = BodyHit(1e20, vec3f(0.0, 1.0, 0.0), vec3f(0.7), 0.0);
  let bodyCount = u32(round(u.options.z));
  for (var index: u32 = 0u; index < 12u; index += 1u) {
    if (index >= bodyCount) { break; }
    let hit = intersectBody(ro, rd, bodies[index]);
    if (hit.x < result.t) {
      result = BodyHit(hit.x, normalize(hit.yzw), bodies[index].colorSelected.xyz, bodies[index].colorSelected.w);
    }
  }
  return result;
}

// Level-set fields become a smooth occupancy whose 0.5 contour is phi = 0.
// The band spans four cells so no corner of a surface-crossing cube saturates
// (the cube diagonal is under two cells); a saturated corner biases the linear
// crossing estimate and renders as cell-pitch lattice artifacts.
fn occupancyFromPhi(phi: f32) -> f32 {
  let band = 4.0 * u.container.y / max(u.gridInfo.y, 1.0);
  return clamp(0.5 - phi / band, 0.0, 1.0);
}

fn fluidSample(cell: vec3i) -> f32 {
  let dims = vec3i(u.gridInfo.xyz);
  let q = clamp(cell, vec3i(0), dims - vec3i(1));
  let mode = u.gridInfo.w;
  if (mode < 1.5) { return textureLoad(fluidField, q, 0).x; }
  if (mode > 2.5) { return occupancyFromPhi(textureLoad(fluidField, q, 0).x); }
  let base = i32(round(textureLoad(tallCellBases, q.xz, 0).x));
  if (q.y < base && base > 0) {
    let t = clamp(f32(q.y) / f32(max(base - 1, 1)), 0.0, 1.0);
    return occupancyFromPhi(mix(textureLoad(fluidField, vec3i(q.x, 0, q.z), 0).x, textureLoad(fluidField, vec3i(q.x, 1, q.z), 0).x, t));
  }
  let packedY = 2 + q.y - base;
  let stored = vec3i(textureDimensions(fluidField));
  if (packedY < 2 || packedY >= stored.y) { return 0.0; }
  return occupancyFromPhi(textureLoad(fluidField, vec3i(q.x, packedY, q.z), 0).x);
}

fn fluidValue(uvw: vec3f) -> f32 {
  let dims = vec3i(u.gridInfo.xyz);
  let q = clamp(uvw * vec3f(dims) - vec3f(0.5), vec3f(0.0), vec3f(dims - vec3i(1)));
  let base = vec3i(floor(q));
  let f = fract(q);
  let lower = vec4f(fluidSample(base), fluidSample(base + vec3i(1, 0, 0)), fluidSample(base + vec3i(0, 1, 0)), fluidSample(base + vec3i(1, 1, 0)));
  let upper = vec4f(fluidSample(base + vec3i(0, 0, 1)), fluidSample(base + vec3i(1, 0, 1)), fluidSample(base + vec3i(0, 1, 1)), fluidSample(base + vec3i(1, 1, 1)));
  let z0 = mix(mix(lower.x, lower.y, f.x), mix(lower.z, lower.w, f.x), f.y);
  let z1 = mix(mix(upper.x, upper.y, f.x), mix(upper.z, upper.w, f.x), f.y);
  return mix(z0, z1, f.z);
}

struct InterfaceCell {
  base: vec3f,
  lower: vec4f,
  upper: vec4f,
}

fn loadInterfaceCell(uvw: vec3f) -> InterfaceCell {
  let dims = vec3i(u.gridInfo.xyz);
  let q = clamp(uvw * vec3f(dims) - vec3f(0.5), vec3f(0.0), vec3f(dims - vec3i(1)));
  let base = clamp(vec3i(floor(q)), vec3i(0), dims - vec3i(2));
  return InterfaceCell(
    vec3f(base),
    vec4f(fluidSample(base), fluidSample(base + vec3i(1, 0, 0)), fluidSample(base + vec3i(0, 1, 0)), fluidSample(base + vec3i(1, 1, 0))),
    vec4f(fluidSample(base + vec3i(0, 0, 1)), fluidSample(base + vec3i(1, 0, 1)), fluidSample(base + vec3i(0, 1, 1)), fluidSample(base + vec3i(1, 1, 1)))
  );
}

fn interfaceValue(cell: InterfaceCell, q: vec3f) -> f32 {
  let f = clamp(q - cell.base, vec3f(0.0), vec3f(1.0));
  let z0 = mix(mix(cell.lower.x, cell.lower.y, f.x), mix(cell.lower.z, cell.lower.w, f.x), f.y);
  let z1 = mix(mix(cell.upper.x, cell.upper.y, f.x), mix(cell.upper.z, cell.upper.w, f.x), f.y);
  return mix(z0, z1, f.z);
}

fn interfaceGradient(cell: InterfaceCell, q: vec3f, size: vec3f) -> vec3f {
  let dims = vec3f(u.gridInfo.xyz);
  let f = clamp(q - cell.base, vec3f(0.0), vec3f(1.0));
  let dx0 = mix(cell.lower.y - cell.lower.x, cell.lower.w - cell.lower.z, f.y);
  let dx1 = mix(cell.upper.y - cell.upper.x, cell.upper.w - cell.upper.z, f.y);
  let dy0 = mix(cell.lower.z - cell.lower.x, cell.lower.w - cell.lower.y, f.x);
  let dy1 = mix(cell.upper.z - cell.upper.x, cell.upper.w - cell.upper.y, f.x);
  let z0 = mix(mix(cell.lower.x, cell.lower.y, f.x), mix(cell.lower.z, cell.lower.w, f.x), f.y);
  let z1 = mix(mix(cell.upper.x, cell.upper.y, f.x), mix(cell.upper.z, cell.upper.w, f.x), f.y);
  return vec3f(mix(dx0, dx1, f.z), mix(dy0, dy1, f.z), z1 - z0) * dims / size;
}

fn refineFluidHit(ro: vec3f, rd: vec3f, a: f32, b: f32, valueA: f32, valueB: f32, boundsMin: vec3f, size: vec3f) -> vec4f {
  let dims = vec3f(u.gridInfo.xyz);
  let denominator = valueB - valueA;
  let fraction = select(0.5, clamp((0.5 - valueA) / denominator, 0.05, 0.95), abs(denominator) > 1e-6);
  var t = mix(a, b, fraction);
  let cell = loadInterfaceCell((ro + rd * t - boundsMin) / size);
  for (var iteration = 0u; iteration < 4u; iteration += 1u) {
    let q = (ro + rd * t - boundsMin) * dims / size - vec3f(0.5);
    let gradient = interfaceGradient(cell, q, size);
    let derivative = dot(gradient, rd);
    if (abs(derivative) < 1e-6) { break; }
    t = clamp(t - (interfaceValue(cell, q) - 0.5) / derivative, a, b);
  }
  let hitQ = (ro + rd * t - boundsMin) * dims / size - vec3f(0.5);
  let gradient = interfaceGradient(cell, hitQ, size);
  let normal = select(-rd, -normalize(gradient), length(gradient) > 1e-5);
  return vec4f(t, normal);
}

struct FluidHit{entry:f32,exit:f32,normal:vec3f}
fn fluidRayHit(ro: vec3f, rd: vec3f, nearT: f32, farT: f32, boundsMin: vec3f, size: vec3f) -> FluidHit {
  let dims = vec3i(u.gridInfo.xyz);
  let span = max(farT - nearT, 0.0);
  let cellSize = min(min(size.x / f32(dims.x), size.y / f32(dims.y)), size.z / f32(dims.z));
  let stepSize = max(span / 144.0, cellSize * 0.65);
  var previousT = nearT;
  var previous = fluidValue((ro + rd * previousT - boundsMin) / size);
  var entry = select(1e20, nearT, previous > 0.5);
  var surfaceNormal = -rd;
  var t = min(nearT + stepSize, farT);
  for (var sampleIndex = 0u; sampleIndex < 176u; sampleIndex += 1u) {
    let occupied = fluidValue((ro + rd * t - boundsMin) / size);
    if (entry > 1e19 && occupied > 0.5 && previous <= 0.5) {
      let refined = refineFluidHit(ro, rd, previousT, t, previous, occupied, boundsMin, size);
      entry = refined.x;
      surfaceNormal = refined.yzw;
    } else if (entry < 1e19 && occupied <= 0.5 && previous > 0.5) {
      let refined = refineFluidHit(ro, rd, previousT, t, previous, occupied, boundsMin, size);
      return FluidHit(entry, refined.x, surfaceNormal);
    }
    previous = occupied;
    previousT = t;
    if (t >= farT) { break; }
    t = min(t + stepSize, farT);
  }
  if(entry<1e19){return FluidHit(entry,farT,surfaceNormal);}return FluidHit(1e20,1e20,vec3f(0.0,1.0,0.0));
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let resolution = max(u.viewport.xy, vec2f(1.0));
  let time = u.viewport.z;
  let scientific = u.options.x > 0.5;
  let ndc = input.uv * 2.0 - 1.0;
  let aspect = resolution.x / resolution.y;

  let ro = u.cameraPosition.xyz;
  let forward = normalize(u.cameraTarget.xyz - ro);
  let right = normalize(cross(forward, vec3f(0.0, 1.0, 0.0)));
  let up = normalize(cross(right, forward));
  let rd = normalize(forward + right * ndc.x * aspect * ${CAMERA_TAN_HALF_FOV} + up * ndc.y * ${CAMERA_TAN_HALF_FOV});

  let room = sampleEnvironment(ro, rd);
  var color = room.color;

  let size = u.container.xyz;
  let center = vec3f(0.0, size.y * 0.5, 0.0);
  let halfSize = size * 0.5;
  let boundsMin = center - halfSize;
  let boundsMax = center + halfSize;
  let hit = boxIntersection(ro, rd, boundsMin, boundsMax);
  let rigidHit = nearestBody(ro, rd);

  let floorT = room.depth;

  if (hit.x <= hit.y && hit.y > 0.0) {
    let nearT = max(hit.x, 0.0);
    let entry = ro + rd * nearT;
    let waterBase = u.container.w;
    let wave = 0.011 * sin(entry.x * 8.0 + time * 0.72) * cos(entry.z * 7.0 - time * 0.47)
             + 0.006 * sin(entry.x * 17.0 - entry.z * 11.0 + time * 0.31);
    var waterT = (waterBase + wave - ro.y) / rd.y;
    var waterPoint = ro + rd * waterT;
    let secondWave = 0.011 * sin(waterPoint.x * 8.0 + time * 0.72) * cos(waterPoint.z * 7.0 - time * 0.47)
                   + 0.006 * sin(waterPoint.x * 17.0 - waterPoint.z * 11.0 + time * 0.31);
    waterT = (waterBase + secondWave - ro.y) / rd.y;
    waterPoint = ro + rd * waterT;
    var solverHit = FluidHit(waterT, hit.y, vec3f(0.0, 1.0, 0.0));
    let shouldMarch = u.gridInfo.w > 0.5;
    if (shouldMarch) {
      solverHit = fluidRayHit(ro, rd, nearT, hit.y, boundsMin, size);
      waterT = solverHit.entry;
      waterPoint = ro + rd * waterT;
    }

    var insideWater = waterT >= nearT && waterT <= hit.y
      && abs(waterPoint.x) <= halfSize.x && abs(waterPoint.z) <= halfSize.z;
    if (shouldMarch) { insideWater = solverHit.entry < 1e19; }
    if (insideWater) {
      let dx = 0.088 * cos(waterPoint.x * 8.0 + time * 0.72) * cos(waterPoint.z * 7.0 - time * 0.47)
             + 0.102 * cos(waterPoint.x * 17.0 - waterPoint.z * 11.0 + time * 0.31);
      let dz = -0.077 * sin(waterPoint.x * 8.0 + time * 0.72) * sin(waterPoint.z * 7.0 - time * 0.47)
             - 0.066 * cos(waterPoint.x * 17.0 - waterPoint.z * 11.0 + time * 0.31);
      var normal = normalize(vec3f(-dx, 1.0, -dz));
      if (shouldMarch) { normal = normalize(solverHit.normal); }
      let fresnel = 0.0204 + 0.9796 * pow(1.0 - max(dot(normal, -rd), 0.0), 5.0);
      let depth = clamp((hit.y - waterT) / max(size.y, 0.001), 0.0, 1.0);
      let thickness=max(0.0,solverHit.exit-solverHit.entry);let transmission=exp(-vec3f(0.95,0.28,0.16)*thickness);let scatter=vec3f(0.018,0.34,0.29)*(vec3f(1.0)-transmission);
      let refracted=color*transmission+scatter;let reflected=environmentLight(reflect(rd,normal));
      var waterColor = mix(refracted,reflected,fresnel);
      waterColor+=vec3f(0.025,0.12,0.105)*(1.0-exp(-thickness*7.0));
      waterColor += environmentLightColor() * pow(max(dot(reflect(rd, normal), environmentLightDirection()), 0.0), 64.0);

      if (scientific) {
        let grid = gridLine(waterPoint.xz, max(u.options.y, 0.01));
        waterColor = mix(waterColor, vec3f(0.42, 0.96, 0.82), grid * 0.58);
      }
      color = mix(color, waterColor, 0.82 + depth * 0.1);
    }

    // The garden pond is set into the ground — no glass vessel to tint.
    if (environmentIndex() != 7) {
      let q = abs((entry - center) / max(halfSize, vec3f(0.001)));
      let edge = max(max(min(q.x, q.y), min(q.x, q.z)), min(q.y, q.z));
      let edgeAlpha = smoothstep(0.91, 0.995, edge);
      let glassFresnel = pow(1.0 - abs(dot(rd, normalize(entry - center))), 3.0);
      let glass = vec3f(0.42, 0.78, 0.72);
      color = mix(color, glass, 0.035 + glassFresnel * 0.035 + edgeAlpha * 0.54);
    }
  }

  if (rigidHit.t < 1e19 && (floorT <= 0.0 || rigidHit.t < floorT)) {
    let rigidPoint = ro + rd * rigidHit.t;
    let light = environmentLightDirection();
    let material = unifiedMaterial(rigidHit.color, 1.0, vec3f(0.0), 0.22, vec3f(0.04), 0.0, vec3f(0.18, 0.42, 0.37), 1.0);
    let lighting = unifiedLightingInput(rigidHit.normal, -rd, light, environmentLightColor());
    var rigidColor = shadeUnifiedSurface(material, lighting);
    let rim = pow(1.0 - max(dot(-rd, rigidHit.normal), 0.0), 3.0);
    if (rigidPoint.y < u.container.w) {
      let submergence = clamp((u.container.w - rigidPoint.y) / max(size.y, 0.001), 0.0, 1.0);
      rigidColor = mix(rigidColor, rigidColor * vec3f(0.35, 0.72, 0.7) + vec3f(0.01, 0.11, 0.1), 0.4 + submergence * 0.32);
    }
    rigidColor += rigidHit.selected * vec3f(0.18, 0.55, 0.43) * (0.28 + rim);
    color = rigidColor;
  }

  color = environmentForeground(color, ndc);
  let vignette = 1.0 - 0.22 * dot(ndc * 0.58, ndc * 0.58);
  color *= vignette;
  color = color / (color + vec3f(1.0));
  color = pow(color, vec3f(1.0 / 2.2));
  return vec4f(color, 1.0);
}
`;

const upscaleShader = /* wgsl */ `
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var sourceSampler: sampler;
struct Out { @builtin(position) position: vec4f, @location(0) uv: vec2f }
@vertex fn vertexMain(@builtin(vertex_index) index: u32) -> Out {
  var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: Out; out.position = vec4f(positions[index], 0.0, 1.0); out.uv = positions[index] * 0.5 + 0.5; return out;
}
@fragment fn fragmentMain(input: Out) -> @location(0) vec4f { return textureSample(source, sourceSampler, vec2f(input.uv.x, 1.0-input.uv.y)); }
`;

export class FluidLabRenderer {
  private device?: GPUDevice;
  private disposed = false;
  private deviceLost = false;
  private context?: GPUCanvasContext;
  private pipeline?: GPURenderPipeline;
  private upscalePipeline?: GPURenderPipeline;
  private upscaleSampler?: GPUSampler;
  private upscaleBindGroup?: GPUBindGroup;
  private waterPipeline?: RasterWaterPipeline;
  private secondaryParticlePipeline?: SecondaryParticleRenderPipeline;
  private voxelDebugPipeline?: SparseVoxelDebugRenderer;
  private gridOverlayPipeline?: GridOverlayPipeline;
  private presentationTexture?: GPUTexture;
  private voxelDebugDepth?: GPUTexture;
  private presentationTextureKey = "";
  private activeRenderScale = 1;
  private readonly rasterRenderScale = 0.72;
  private uniformBuffer?: GPUBuffer;
  private bodyBuffer?: GPUBuffer;
  private fluidTexture?: GPUTexture;
  private columnBaseTexture?: GPUTexture;
  private gridCellTexture?: GPUTexture;
  private velocityFallbackTexture?: GPUTexture;
  private pressureSamplesFallbackTexture?: GPUTexture;
  private scalarFallbackTexture?: GPUTexture;
  private fluidTextureKey = "";
  private fluidRevision = -1;
  private gpuFluid?: GPUSolverInstance;
  private readonly retiredGPUFluids = new Set<GPUSolverInstance>();
  private gpuFluidKey = "";
  private gpuFluidPendingKey = "";
  private gpuFluidPending?: Promise<void>;
  private gpuFluidRequestGeneration = 0;
  private adapterName = "WebGPU adapter";
  private gpuInfoCallback?: (info: GPUEulerianInfo) => void;
  private gpuRigidLoadCallback?: (loads: GPURigidLoad[]) => void;
  private gpuAdvanceCompletedCallback?: (time_s: number) => void;
  private gpuPendingBatches = 0;
  private lastGPUCompletionAt_ms = -Infinity;
  private lastGPUCompletedTime_s = 0;
  private gpuFluidGeneration = 0;
  private lastGPUReadbackSecond = -1;
  private bindGroup?: GPUBindGroup;
  private format?: GPUTextureFormat;
  private renderQuerySet?: GPUQuerySet;
  private renderQueryResolve?: GPUBuffer;
  private renderReadbackPending = false;
  private lastRenderQueryAt = -Infinity;
  private gpuRender_ms?: number;
  private gpuSurfaceExtraction_ms?: number;
  private gpuDryScene_ms?: number;
  private gpuInterfaces_ms?: number;
  private gpuSprayFront_ms?: number;
  private gpuSprayBack_ms?: number;
  private gpuSprayRender_ms?: number;
  private gpuOpticalComposite_ms?: number;
  private gpuUpscale_ms?: number;
  private renderTimingContext = "";
  private deviceRecoveryAttempts = 0;
  private lastDeviceRecoveryAt_ms = -Infinity;

  constructor(private readonly canvas: HTMLCanvasElement, private readonly onStatus: (status: GPUStatus) => void, onGPUInfo?: (info: GPUEulerianInfo) => void, onGPURigidLoads?: (loads: GPURigidLoad[]) => void, onGPUAdvanceCompleted?: (time_s: number) => void) { this.gpuInfoCallback = onGPUInfo; this.gpuRigidLoadCallback = onGPURigidLoads; this.gpuAdvanceCompletedCallback = onGPUAdvanceCompleted; }

  async initialize(): Promise<void> {
    const startedAt_ms=performance.now();
    const progress=(label:string,completed:number,total=7,phase="renderer")=>this.onStatus({state:"initializing",label,phase,completed,total,startedAt_ms});
    progress("Requesting WebGPU adapter",0);
    if (!("gpu" in navigator)) {
      this.onStatus({ state: "unavailable", label: "WebGPU is not available in this browser" });
      return;
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
      this.onStatus({ state: "unavailable", label: "No compatible GPU adapter was found" });
      return;
    }
    progress("Requesting GPU device",1);
    const requiredFeatures: GPUFeatureName[] = adapter.features.has("timestamp-query") ? ["timestamp-query"] : [];
    const requiredLimits = requiredFluidDeviceLimits(adapter.limits);
    const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
    if (this.disposed) { device.destroy(); return; }
    const context = this.canvas.getContext("webgpu");
    if (!context) {
      this.onStatus({ state: "unavailable", label: "WebGPU canvas context could not be created" });
      return;
    }
    this.device = device;
    this.context = context;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    device.addEventListener("uncapturederror", (event) => console.error(`WebGPU validation: ${event.error.message}`));
    void device.lost.then((info) => {
      if (this.disposed || this.device !== device || this.deviceLost) return;
      this.deviceLost = true;
      const fluid = this.gpuFluid;
      this.gpuFluid = undefined;
      this.gpuFluidKey = "";
      this.gpuFluidPendingKey = "";
      this.gpuFluidPending = undefined;
      this.resetGPUQueueTracking();
      this.gpuFluidGeneration += 1;
      // A solver initialization pending on the lost device must never attach
      // after recovery: its resources belong to the dead device and any bind
      // group mixing them with the replacement device fails validation.
      this.gpuFluidRequestGeneration += 1;
      try { fluid?.destroy(); } catch { /* Resources may already be invalid after device loss. */ }
      // Breadcrumbs for hang diagnosis: the last known solver state narrows a
      // watchdog reset down to a stage without needing a reproduction.
      if (fluid) console.error("GPU device lost mid-simulation", { reason: info.reason, message: info.message, submittedTime_s: fluid.info.submittedTime_s, completedTime_s: fluid.info.completedTime_s, pendingBatches: this.gpuPendingBatches, encodedSteps: fluid.info.encodedSteps, gpuTimings: fluid.info.gpuTimings });
      this.onStatus({ state: "lost", label: `GPU device lost: ${info.message || info.reason}` });
      this.scheduleDeviceRecovery(info.reason);
    }).catch((error: unknown) => {
      if (!this.disposed) console.error("Unable to observe WebGPU device loss", error);
    });
    if(device.features.has("timestamp-query")){this.renderQuerySet=device.createQuerySet({type:"timestamp",count:16});this.renderQueryResolve=device.createBuffer({size:16*8,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});}
    context.configure({ device, format: this.format, alphaMode: "opaque" });

    progress("Checking presentation shader",2);
    const shaderModule = device.createShaderModule({ label: "Fluid Lab presentation shader", code: shader });
    const compilation = await shaderModule.getCompilationInfo();
    if (this.disposed || this.deviceLost) return;
    const errors = compilation.messages.filter((message) => message.type === "error");
    if (errors.length > 0) throw new Error(errors.map((error) => `${error.lineNum}:${error.linePos} ${error.message}`).join("\n"));

    progress("Compiling presentation pipeline",3);
    this.pipeline = await device.createRenderPipelineAsync({
      label: "Fluid Lab ray presentation",
      layout: "auto",
      vertex: { module: shaderModule, entryPoint: "vertexMain" },
      fragment: { module: shaderModule, entryPoint: "fragmentMain", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" }
    });
    const upscaleModule=device.createShaderModule({label:"Presentation upscale shader",code:upscaleShader});
    this.upscalePipeline=await device.createRenderPipelineAsync({label:"Presentation upscale",layout:"auto",vertex:{module:upscaleModule,entryPoint:"vertexMain"},fragment:{module:upscaleModule,entryPoint:"fragmentMain",targets:[{format:this.format}]},primitive:{topology:"triangle-list"}});
    this.upscaleSampler=device.createSampler({magFilter:"linear",minFilter:"linear"});
    this.uniformBuffer = device.createBuffer({ label: "Fluid Lab view uniforms", size: 400, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bodyBuffer = device.createBuffer({ label: "Fluid Lab rigid bodies", size: 12 * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.fluidTexture = device.createTexture({ size: [1, 1, 1], dimension: "3d", format: "r8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.columnBaseTexture = device.createTexture({ label: "Uniform-grid tall-cell fallback", size: [1, 1], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.gridCellTexture = device.createTexture({ label: "Uniform-grid adaptive-cell fallback", size: [1, 1, 1], dimension: "3d", format: "rg32uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.velocityFallbackTexture = device.createTexture({ label: "Overlay velocity fallback", size: [1, 1, 1], dimension: "3d", format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.pressureSamplesFallbackTexture = device.createTexture({ label: "Overlay pressure-sample fallback", size: [1, 1, 1], dimension: "3d", format: "rgba32uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.scalarFallbackTexture = device.createTexture({ label: "Overlay scalar fallback", size: [1, 1, 1], dimension: "3d", format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const gridOverlayPipeline = new GridOverlayPipeline(device, this.format, this.uniformBuffer, this.bodyBuffer);
    try {
      progress("Compiling grid overlay",4);
      await gridOverlayPipeline.initialize();
      this.gridOverlayPipeline = gridOverlayPipeline;
    } catch (error) {
      console.warn("Grid overlay pipeline unavailable", error);
    }
    const voxelDebugPipeline = new SparseVoxelDebugRenderer(device, { colorFormat: this.format });
    try {
      progress("Compiling sparse voxel inspection",5);
      await voxelDebugPipeline.initialize();
      this.voxelDebugPipeline = voxelDebugPipeline;
    } catch (error) {
      voxelDebugPipeline.destroy();
      console.warn("Sparse voxel inspection unavailable", error);
    }
    const waterPipeline = new RasterWaterPipeline(device, this.format, this.uniformBuffer, this.bodyBuffer);
    try {
      progress("Compiling raster water pipelines",5);
      await waterPipeline.initialize((label,completed,total)=>progress(label,completed,total,"water-renderer"));
      this.waterPipeline = waterPipeline;
    } catch (error) {
      // The legacy path has deliberately independent shaders/resources.  An
      // adapter-specific failure in the optional optical pipeline must not make
      // the viewport unusable or compromise the requested comparison mode.
      waterPipeline.destroy();
      console.warn("Raster water pipeline unavailable; using the intact ray marcher", error);
    }
    const secondaryParticlePipeline = new SecondaryParticleRenderPipeline(device, this.format, this.uniformBuffer);
    try {
      progress("Compiling secondary liquid particles",6);
      await secondaryParticlePipeline.initialize();
      this.secondaryParticlePipeline = secondaryParticlePipeline;
      this.waterPipeline?.setSecondaryParticles(secondaryParticlePipeline);
    } catch (error) {
      console.warn("Secondary liquid particle renderer unavailable", error);
    }
    if (this.disposed || this.deviceLost) return;
    this.rebuildBindGroup();

    const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
    this.adapterName = info ? [info.vendor, info.architecture].filter(Boolean).join(" · ") || "WebGPU adapter" : "WebGPU adapter";
    progress("Renderer ready; preparing solver",7);
    this.onStatus({ state: "ready", label: "WebGPU renderer ready", adapter: this.adapterName });
  }

  /**
   * A lost device leaves the app permanently dead without intervention: every
   * frame-loop entry point guards on deviceLost, so a transient TDR would
   * otherwise present as a hard crash until reload. Recover by re-running
   * initialize() on a fresh device; the solver rebuilds automatically from the
   * scene on the next frame (simulation state does not survive the loss).
   * Attempts are bounded so a deterministic fault (a shader that kills the
   * device on every submit) cannot loop device creation forever.
   */
  private scheduleDeviceRecovery(reason: string) {
    if (this.disposed || reason === "destroyed") return;
    if (performance.now() - this.lastDeviceRecoveryAt_ms > 60_000) this.deviceRecoveryAttempts = 0;
    if (this.deviceRecoveryAttempts >= 3) return;
    this.deviceRecoveryAttempts += 1;
    this.lastDeviceRecoveryAt_ms = performance.now();
    setTimeout(() => { void this.recoverDevice(); }, 500 * this.deviceRecoveryAttempts);
  }

  private async recoverDevice() {
    if (this.disposed || !this.deviceLost) return;
    // Resources on a lost device are already invalid and need no destroy;
    // drop every device-scoped reference so the frame loop's !this.device
    // guards hold until initialize() completes on the replacement device.
    this.device = undefined; this.context = undefined;
    this.pipeline = undefined; this.upscalePipeline = undefined; this.upscaleSampler = undefined; this.upscaleBindGroup = undefined;
    this.waterPipeline = undefined; this.gridOverlayPipeline = undefined; this.voxelDebugPipeline = undefined; this.secondaryParticlePipeline = undefined;
    this.bindGroup = undefined; this.uniformBuffer = undefined; this.bodyBuffer = undefined;
    this.presentationTexture = undefined; this.voxelDebugDepth = undefined; this.presentationTextureKey = "";
    this.fluidTexture = undefined; this.columnBaseTexture = undefined; this.gridCellTexture = undefined;
    this.velocityFallbackTexture = undefined; this.pressureSamplesFallbackTexture = undefined; this.scalarFallbackTexture = undefined;
    this.fluidTextureKey = ""; this.fluidRevision = -1;
    this.renderQuerySet = undefined; this.renderQueryResolve = undefined; this.renderReadbackPending = false;
    this.retiredGPUFluids.clear();
    this.deviceLost = false;
    try {
      await this.initialize();
    } catch (error) {
      this.onStatus({ state: "unavailable", label: error instanceof Error ? `GPU recovery failed: ${error.message}` : "GPU recovery failed" });
    }
  }

  private rebuildBindGroup(texture = this.fluidTexture, columnBases = this.columnBaseTexture, gridCells = this.gridCellTexture, velocity = this.velocityFallbackTexture, pressureSamples = this.pressureSamplesFallbackTexture, divergence = this.scalarFallbackTexture, pressure = this.scalarFallbackTexture) {
    if (!this.device || this.disposed || this.deviceLost || !this.pipeline || !this.uniformBuffer || !this.bodyBuffer || !texture || !columnBases || !gridCells || !velocity || !pressureSamples || !divergence || !pressure) return;
    this.bindGroup = this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.bodyBuffer } },
      { binding: 2, resource: texture.createView({ dimension: "3d" }) },
      { binding: 3, resource: columnBases.createView() }
    ] });
    this.waterPipeline?.setVolume(texture, columnBases);
    this.waterPipeline?.setSparseSurface(this.gpuFluid?.sparseSurfaceBand);
    this.gridOverlayPipeline?.setSparseSurface(this.gpuFluid?.sparseSurfaceBand);
    this.gridOverlayPipeline?.setVolume(texture, columnBases, gridCells, velocity, pressureSamples, divergence, pressure);
  }

  private solverKey(scene:SceneDescription,config:SimulationRunConfig){return`${config.methodId}:${config.quality}:${JSON.stringify(config.values)}:${scene.environment??"default"}:${scene.container.width_m}:${scene.container.height_m}:${scene.container.depth_m}:${scene.container.fillFraction}:${scene.fluid.initialCondition}:${JSON.stringify(scene.fluid.initialBrickSeeds_m??null)}:${scene.fluid.density_kg_m3}:${scene.fluid.dynamicViscosity_Pa_s}:${scene.fluid.surfaceTension_N_m}:${scene.fluid.gravity_m_s2.y}:${scene.container.fluidWallMode}:${JSON.stringify(scene.fluid.inflow??null)}`;}

  private resetGPUQueueTracking() {
    this.gpuPendingBatches = 0;
    this.lastGPUCompletionAt_ms = -Infinity;
    this.lastGPUCompletedTime_s = 0;
  }

  private retireGPUFluid(fluid: GPUSolverInstance) {
    const device = this.device;
    if (!device || this.deviceLost) { fluid.destroy(); return; }
    this.retiredGPUFluids.add(fluid);
    // A method switch can occur after a frame encoded the old solver's
    // textures but before that frame submits. Defer the queue fence to the
    // next animation frame so it covers that final submission.
    requestAnimationFrame(() => {
      void device.queue.onSubmittedWorkDone().catch(() => { /* Device loss invalidates the resources. */ }).finally(() => {
        if (this.retiredGPUFluids.delete(fluid)) fluid.destroy();
      });
    });
  }

  private beginGPUFluidInitialization(scene:SceneDescription,config:SimulationRunConfig,key:string){
    if(!this.device||this.disposed||this.deviceLost)return;
    const method=getMethod(config.methodId);if(!method.createSolver)return;
    const device=this.device,generation=++this.gpuFluidRequestGeneration,startedAt_ms=performance.now();
    const previous=this.gpuFluid;
    // Clear the live-solver reference before the detach rebind below:
    // rebuildBindGroup re-attaches this.gpuFluid's sparse surface band to the
    // water and grid-overlay pipelines, so an old reference here would rebind
    // the very buffers the retirement fence is about to destroy.
    this.gpuFluid=undefined;
    if(previous){
      // Detach presentation bind groups from the live solver textures before
      // retiring them. Option A/B switches rebuild asynchronously; without
      // this fallback rebind, the grid overlay can keep submitting the old
      // topology texture after its queue fence has completed and destroyed it.
      this.rebuildBindGroup();
      this.secondaryParticlePipeline?.setSource(undefined);
      this.voxelDebugPipeline?.setSource(undefined);
      this.retireGPUFluid(previous);
    }
    this.gpuFluidKey="";this.gpuFluidPendingKey=key;this.resetGPUQueueTracking();this.gpuFluidGeneration+=1;this.lastGPUReadbackSecond=-1;
    const report=(progress:{phase:string;label:string;completed:number;total:number})=>{if(this.disposed||this.deviceLost||generation!==this.gpuFluidRequestGeneration)return;this.onStatus({state:"initializing",...progress,startedAt_ms});};
    report({phase:"solver",label:`Preparing ${method.shortLabel} solver`,completed:0,total:1});
    const create=method.createSolverAsync
      ? method.createSolverAsync(device,scene,config.quality,config.values,this.gpuRigidLoadCallback,report)
      : new Promise<GPUSolverInstance>((resolve,reject)=>setTimeout(()=>{try{resolve(method.createSolver!(device,scene,config.quality,config.values,this.gpuRigidLoadCallback));}catch(error){reject(error);}},0));
    this.gpuFluidPending=create.then((solver)=>{
      if(this.disposed||this.deviceLost||generation!==this.gpuFluidRequestGeneration){solver.destroy();return;}
      this.gpuFluid=solver;this.gpuFluidKey=key;this.gpuFluidPendingKey="";this.rebuildBindGroup(solver.surfaceFieldTexture??solver.volumeTexture,solver.columnBaseTexture,solver.gridCellTexture??this.gridCellTexture,solver.velocityTexture??this.velocityFallbackTexture,solver.gridPressureSamplesTexture??this.pressureSamplesFallbackTexture,solver.gridDivergenceTexture??this.scalarFallbackTexture,solver.gridPressureTexture??this.scalarFallbackTexture);this.secondaryParticlePipeline?.setSource(solver.secondaryParticles);this.voxelDebugPipeline?.setSource(solver.sparseVoxelRenderSource);this.gpuInfoCallback?.(solver.info);this.onStatus({state:"ready",label:"WebGPU solver ready",adapter:this.adapterName});
    }).catch((error:unknown)=>{if(this.disposed||generation!==this.gpuFluidRequestGeneration)return;this.gpuFluidPendingKey="";this.onStatus({state:"unavailable",label:error instanceof Error?`GPU initialization failed: ${error.message}`:"GPU initialization failed"});}).finally(()=>{if(generation===this.gpuFluidRequestGeneration)this.gpuFluidPending=undefined;});
  }

  private ensureGPUFluid(scene: SceneDescription, config: SimulationRunConfig, time_s: number, bodies: RigidBodyState[], targetFps: number) {
    if (!this.device || this.disposed || this.deviceLost) return undefined;
    const method = getMethod(config.methodId);
    if (!method.createSolver) return undefined;
    const key=this.solverKey(scene,config);
    if(!this.gpuFluid||key!==this.gpuFluidKey){if(this.gpuFluidPendingKey!==key)this.beginGPUFluidInitialization(scene,config,key);return undefined;}
    if (time_s < (this.gpuFluid.info.submittedTime_s ?? 0)) {this.beginGPUFluidInitialization(scene,config,key);return undefined;}
    const previousSubmittedTime = this.gpuFluid.info.submittedTime_s ?? 0;
    // Submit one presentation-sized batch whenever prepared transport exists,
    // even when an earlier batch fence is unresolved. The controller bounds
    // tall-cell and octree preparation to two batches, keeping the queue fed
    // while bounding partitioned rigid-feedback latency.
    const batchLimit = gpuBatchDepth(config.methodId, scene.numerics.fixedDt_s, bodies.length > 0, targetFps);
    let submittedTime = previousSubmittedTime;
    for (let batch = 0; batch < batchLimit && submittedTime + 1e-9 < time_s; batch += 1) {
      if (!this.gpuFluid.advanceTo(time_s, bodies)) break;
      const nextSubmittedTime = this.gpuFluid.info.submittedTime_s ?? submittedTime;
      if (nextSubmittedTime <= submittedTime) break;
      submittedTime = nextSubmittedTime;
    }
    if (submittedTime > previousSubmittedTime) {
      const fluid = this.gpuFluid, generation = this.gpuFluidGeneration;
      const submittedAt_ms = performance.now();
      const batchSimulation_s = submittedTime - previousSubmittedTime;
      if (this.gpuPendingBatches === 0 && Number.isFinite(this.lastGPUCompletionAt_ms)) {
        fluid.info.gpuQueueStarved_ms = Math.max(0, submittedAt_ms - this.lastGPUCompletionAt_ms);
      }
      this.gpuPendingBatches += 1;
      fluid.info.gpuPendingBatches = this.gpuPendingBatches;
      fluid.info.gpuInFlightSimulation_s = Math.max(0, submittedTime - (fluid.info.completedTime_s ?? 0));
      void this.device.queue.onSubmittedWorkDone().then(() => {
        if (this.disposed || this.deviceLost || this.gpuFluid !== fluid || this.gpuFluidGeneration !== generation) return;
        const completedAt_ms = performance.now();
        this.gpuPendingBatches = Math.max(0, this.gpuPendingBatches - 1);
        fluid.info.completedTime_s = Math.max(fluid.info.completedTime_s ?? 0, submittedTime);
        fluid.info.gpuPendingBatches = this.gpuPendingBatches;
        fluid.info.gpuInFlightSimulation_s = Math.max(0, (fluid.info.submittedTime_s ?? submittedTime) - fluid.info.completedTime_s);
        fluid.info.gpuBatchWall_ms = completedAt_ms - submittedAt_ms;
        fluid.info.gpuBatchSimulation_s = batchSimulation_s;
        if (Number.isFinite(this.lastGPUCompletionAt_ms) && submittedTime > this.lastGPUCompletedTime_s) {
          fluid.info.gpuCompletionWall_ms = completedAt_ms - this.lastGPUCompletionAt_ms;
          fluid.info.gpuCompletionSimulation_s = submittedTime - this.lastGPUCompletedTime_s;
        }
        this.lastGPUCompletionAt_ms = completedAt_ms;
        this.lastGPUCompletedTime_s = submittedTime;
        this.gpuInfoCallback?.({ ...fluid.info });
        this.gpuAdvanceCompletedCallback?.(submittedTime);
      }).catch(() => { /* Device loss is reported by device.lost. */ });
    }
    // Sample at 30 Hz of simulation time so a paper-sized 1/30 s step cannot
    // cross an instability threshold between diagnostic readbacks. The solver
    // coalesces reads while a previous map is pending.
    const diagnosticTick=Math.round(time_s*30);if(diagnosticTick!==this.lastGPUReadbackSecond){this.lastGPUReadbackSecond=diagnosticTick;void this.gpuFluid.readStats().then(info=>this.gpuInfoCallback?.({...info})).catch(()=>{ /* Device loss is reported by device.lost. */ });}
    return this.gpuFluid.info;
  }

  private uploadFluid(fluid?: EulerianRenderState) {
    if (!this.device || this.disposed || this.deviceLost || !fluid) return;
    const key = `${fluid.nx}x${fluid.ny}x${fluid.nz}`;
    if (key !== this.fluidTextureKey) {
      this.fluidTexture?.destroy();
      this.fluidTexture = this.device.createTexture({ label: "Eulerian occupied cells", size: [fluid.nx, fluid.ny, fluid.nz], dimension: "3d", format: "r8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      this.fluidTextureKey = key; this.fluidRevision = -1; this.rebuildBindGroup();
    }
    if (fluid.revision === this.fluidRevision || !this.fluidTexture) return;
    const bytesPerRow = Math.ceil(fluid.nx / 256) * 256;
    const packed = new Uint8Array(bytesPerRow * fluid.ny * fluid.nz);
    for (let k = 0; k < fluid.nz; k += 1) for (let j = 0; j < fluid.ny; j += 1) {
      const source = fluid.nx * (j + fluid.ny * k);
      packed.set(fluid.occupancy.subarray(source, source + fluid.nx), bytesPerRow * (j + fluid.ny * k));
    }
    this.device.queue.writeTexture({ texture: this.fluidTexture }, packed, { bytesPerRow, rowsPerImage: fluid.ny }, { width: fluid.nx, height: fluid.ny, depthOrArrayLayers: fluid.nz });
    this.fluidRevision = fluid.revision;
  }

  resize(renderScale = 1): void {
    if (this.disposed || this.deviceLost) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(this.canvas.clientWidth * ratio));
    const height = Math.max(1, Math.floor(this.canvas.clientHeight * ratio));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    if (!this.device || !this.format || !this.upscalePipeline || !this.upscaleSampler) return;
    this.activeRenderScale = renderScale;
    const renderWidth = Math.max(1, Math.floor(width * renderScale));
    const renderHeight = Math.max(1, Math.floor(height * renderScale));
    const key = `${renderWidth}x${renderHeight}`;
    if (key === this.presentationTextureKey) return;
    this.presentationTexture?.destroy();
    this.voxelDebugDepth?.destroy();
    this.presentationTexture = this.device.createTexture({label:"Water presentation target",size:[renderWidth,renderHeight],format:this.format,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING});
    this.voxelDebugDepth = this.device.createTexture({label:"Sparse voxel inspection depth",size:[renderWidth,renderHeight],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT});
    this.upscaleBindGroup=this.device.createBindGroup({layout:this.upscalePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:this.presentationTexture.createView()},{binding:1,resource:this.upscaleSampler}]});
    this.presentationTextureKey=key;
    this.waterPipeline?.ensureSize(renderWidth, renderHeight);
  }

  get presentationResolution(): string {
    if (!this.presentationTexture) return `${this.canvas.width} × ${this.canvas.height}`;
    return `${this.presentationTexture.width} × ${this.presentationTexture.height} (${Math.round(this.activeRenderScale * 100)}%)`;
  }

  draw(time_s: number, scene: SceneDescription, camera: CameraState, view: ViewMode, bodies: RigidBodyState[], selectedBodyId: string | undefined, fluid: EulerianRenderState | undefined, backend: SimulationBackend, config: SimulationRunConfig, gridOverlay?: GridOverlayConfig, waterRenderMode: WaterRenderMode = "rasterized", environmentId: EnvironmentId = defaultEnvironmentId, targetFps = 60, voxelRenderMode: VoxelRenderMode = "smooth"): RendererFrameMetrics {
    if (!this.device || this.disposed || this.deviceLost || !this.context || !this.pipeline || !this.uniformBuffer || !this.bodyBuffer || !this.bindGroup) return {cpuFrame_ms:0,cpuPhysicsSubmit_ms:0,cpuDataUpload_ms:0,cpuRenderEncode_ms:0};
    this.resize(waterRenderMode === "rasterized" ? this.rasterRenderScale : 1);
    if (!this.presentationTexture || !this.upscalePipeline || !this.upscaleBindGroup) return {cpuFrame_ms:0,cpuPhysicsSubmit_ms:0,cpuDataUpload_ms:0,cpuRenderEncode_ms:0};
    const start = performance.now();
    const timingContext = `${config.methodId}:${config.quality}:${waterRenderMode}:${voxelRenderMode}`;
    if (timingContext !== this.renderTimingContext) { this.renderTimingContext = timingContext; this.gpuRender_ms = undefined; this.gpuSurfaceExtraction_ms=undefined;this.gpuDryScene_ms=undefined;this.gpuInterfaces_ms=undefined;this.gpuSprayFront_ms=undefined;this.gpuSprayBack_ms=undefined;this.gpuSprayRender_ms=undefined;this.gpuOpticalComposite_ms=undefined;this.gpuUpscale_ms=undefined; }
    const position = cameraPosition(camera);
    const physicsStart=performance.now();
    if (backend === "webgpu" && gridOverlay?.axis !== "off") this.gpuFluid?.ensureGridDiagnosticTextures?.();
    const gpuInfo = backend === "webgpu" ? this.ensureGPUFluid(scene, config, time_s, bodies, targetFps) : undefined;
    if (gpuInfo && this.gpuFluid && this.gridCellTexture && this.velocityFallbackTexture && this.pressureSamplesFallbackTexture && this.scalarFallbackTexture) this.gridOverlayPipeline?.setVolume(this.gpuFluid.surfaceFieldTexture ?? this.gpuFluid.volumeTexture, this.gpuFluid.columnBaseTexture, this.gpuFluid.gridCellTexture ?? this.gridCellTexture, this.gpuFluid.velocityTexture ?? this.velocityFallbackTexture, this.gpuFluid.gridPressureSamplesTexture ?? this.pressureSamplesFallbackTexture, this.gpuFluid.gridDivergenceTexture ?? this.scalarFallbackTexture, this.gpuFluid.gridPressureTexture ?? this.scalarFallbackTexture);
    const cpuPhysicsSubmit_ms=performance.now()-physicsStart,uploadStart=performance.now();
    if (backend === "cpu-reference") this.uploadFluid(fluid);
    const uniform = new Float32Array([
      this.presentationTexture.width, this.presentationTexture.height, time_s, 0,
      position.x, position.y, position.z, 0,
      camera.target_m.x, camera.target_m.y, camera.target_m.z, 0,
      scene.container.width_m, scene.container.height_m, scene.container.depth_m, scene.container.height_m * scene.container.fillFraction,
      // options.w carries the largest represented adaptive pressure-cell
      // width. The grid overlay uses it to normalize its categorical scale
      // palette to the hierarchy that can actually exist in this solver.
      view === "scientific" ? 1 : 0, scene.nominalResolution.length_m, Math.min(bodies.length, 12), gpuInfo?.quadtreeMaximumFluidScale ?? 1,
      // Field mode: 1 = raw occupancy, 2 = packed tall-cell level set,
      // 3 = uniform-layout level set (quadtree resident phi).
      gpuInfo?.nx ?? fluid?.nx ?? 1, gpuInfo?.ny ?? fluid?.ny ?? 1, gpuInfo?.nz ?? fluid?.nz ?? 1, gpuInfo ? (gpuInfo.gridKind === "restricted-tall-cell" ? 2 : gpuInfo.gridKind === "quadtree-tall-cell" || gpuInfo.gridKind === "octree" ? 3 : 1) : fluid ? 1 : 0,
      gridOverlay?.axis === "z" ? 1 : gridOverlay?.axis === "x" ? 2 : gridOverlay?.axis === "y" ? 3 : 0, gridOverlay?.position ?? 0.5, gpuInfo?.gridKind === "quadtree-tall-cell" || gpuInfo?.gridKind === "octree" ? 1 : 0,
      gridOverlay?.mode === "cfl" ? 1 : gridOverlay?.mode === "speed" ? 2 : gridOverlay?.mode === "phi" ? 3 : gridOverlay?.mode === "divergence" ? 4 : gridOverlay?.mode === "pressure" ? 5 : gridOverlay?.mode === "representation" ? 6 : gridOverlay?.mode === "optical" ? 7 : gridOverlay?.mode === "projection" && gpuInfo?.gridKind === "octree" ? 8 : gridOverlay?.mode === "resolution" && (gpuInfo?.gridKind === "quadtree-tall-cell" || gpuInfo?.gridKind === "octree") ? 9 : 0,
      environmentIndex(environmentId), gpuInfo?.lastDt_s ?? 0, gpuInfo?.maxSpeed_m_s ?? 0,
      gpuInfo?.gridKind === "quadtree-tall-cell" ? (gpuInfo.quadtreeOpticalLayerMode === "adaptive-motion" ? 2 : 1) : 0
    ]);
    // Terrain heightfield mirror for the environment shaders: meta lane plus
    // two vec4 lanes per feature, matching lib/terrain.ts semantics exactly.
    const packed = new Float32Array(100);
    packed.set(uniform, 0);
    if (sceneHasTerrain(scene) && scene.terrain) {
      const terrain = scene.terrain;
      const features = terrain.features.slice(0, MAX_TERRAIN_FEATURES);
      packed.set([1, terrain.baseHeight_m, features.length, TERRAIN_UNION_EXPONENT], 32);
      features.forEach((feature, index) => {
        packed.set([feature.center_m.x, feature.center_m.z, feature.radius_m.x, feature.radius_m.z], 36 + index * 8);
        packed.set([(feature.kind === "mound" ? 1 : -1) * feature.amount_m, feature.rotation_rad ?? 0, feature.flat ?? TERRAIN_DEFAULT_FLAT, 0], 40 + index * 8);
      });
    }
    this.device.queue.writeBuffer(this.uniformBuffer, 0, packed);
    const bodyData = new Float32Array(12 * 16);
    const shapeIndex = { sphere: 0, box: 1, capsule: 2, cylinder: 3 } as const;
    const palette = [[0.95, 0.63, 0.29], [0.48, 0.66, 0.96], [0.84, 0.42, 0.48], [0.66, 0.52, 0.92]];
    bodies.slice(0, 12).forEach((body, index) => {
      const offset = index * 16;
      const d = body.description.dimensions_m;
      const half = body.description.shape === "box" ? [d.x / 2, d.y / 2, d.z / 2] : body.description.shape === "sphere" ? [d.x, d.x, d.x] : [d.x, d.y / 2, d.x];
      const color = palette[shapeIndex[body.description.shape]];
      bodyData.set([body.position_m.x, body.position_m.y, body.position_m.z, boundingRadius(body)], offset);
      bodyData.set([half[0], half[1], half[2], shapeIndex[body.description.shape]], offset + 4);
      bodyData.set([body.orientation.w, body.orientation.x, body.orientation.y, body.orientation.z], offset + 8);
      bodyData.set([color[0], color[1], color[2], body.description.id === selectedBodyId ? 1 : 0], offset + 12);
    });
    this.device.queue.writeBuffer(this.bodyBuffer, 0, bodyData);
    const cpuDataUpload_ms=performance.now()-uploadStart,renderStart=performance.now();
    const encoder = this.device.createCommandEncoder({ label: "Fluid Lab frame" });
    this.secondaryParticlePipeline?.setSource(backend === "webgpu" ? this.gpuFluid?.secondaryParticles : undefined);
    // One interval surrounds the complete active presentation path. For raster
    // optics it starts at isosurface extraction and ends after final upscale;
    // for the legacy path it starts at its full-screen pass and ends likewise.
    const sampleRenderGPU=Boolean(this.renderQuerySet&&this.renderQueryResolve&&!this.renderReadbackPending&&renderStart-this.lastRenderQueryAt>=250);
    const rasterResult = waterRenderMode === "rasterized" && this.waterPipeline?.encode(
      encoder, this.presentationTexture.createView(),
      gpuInfo?.nx ?? fluid?.nx ?? 1, gpuInfo?.ny ?? fluid?.ny ?? 1, gpuInfo?.nz ?? fluid?.nz ?? 1,
      gpuInfo?.gridKind === "restricted-tall-cell", gpuInfo?.maximumNeighborDelta ?? 0,
      gpuInfo?.encodedSteps ?? fluid?.revision ?? 0,
      targetFps,
      sampleRenderGPU&&this.renderQuerySet?{
        extraction:{querySet:this.renderQuerySet,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1},
        scene:{querySet:this.renderQuerySet,beginningOfPassWriteIndex:2,endOfPassWriteIndex:3},
        frontInterfaces:{querySet:this.renderQuerySet,beginningOfPassWriteIndex:4,endOfPassWriteIndex:5},
        backInterfaces:{querySet:this.renderQuerySet,beginningOfPassWriteIndex:6,endOfPassWriteIndex:7},
        sprayFront:{querySet:this.renderQuerySet,beginningOfPassWriteIndex:12,endOfPassWriteIndex:13},
        sprayBack:{querySet:this.renderQuerySet,beginningOfPassWriteIndex:14,endOfPassWriteIndex:15},
        composite:{querySet:this.renderQuerySet,beginningOfPassWriteIndex:8,endOfPassWriteIndex:9}
      }:undefined
    );
    const rasterized = Boolean(rasterResult);
    if (!rasterized) {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: this.presentationTexture.createView(), clearValue: { r: 0.01, g: 0.025, b: 0.024, a: 1 }, loadOp: "clear", storeOp: "store" }],
        ...(sampleRenderGPU&&this.renderQuerySet?{timestampWrites:{querySet:this.renderQuerySet,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:{})
      });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.draw(3);
      pass.end();
    }
    if (voxelRenderMode !== "smooth" && this.voxelDebugDepth) {
      const sceneExtent = Math.hypot(scene.container.width_m, scene.container.height_m, scene.container.depth_m);
      this.voxelDebugPipeline?.encode(encoder, {
        mode: voxelRenderMode,
        colorTarget: this.presentationTexture.createView(),
        depthTarget: this.voxelDebugDepth.createView(),
        depthLoadOp: "clear",
        // Inspection is a representation switch, not a subtle overlay. Clear
        // the smooth hybrid frame so contiguous voxels and brick bounds remain
        // unmistakable even for a still, fully filled region.
        colorLoadOp: "clear",
        viewProjection: voxelViewProjectionMatrix(camera, this.presentationTexture.width / Math.max(1, this.presentationTexture.height), 0.01, camera.distance_m + sceneExtent * 3),
        cameraPosition: [position.x, position.y, position.z],
        containerBounds: {
          min: [-scene.container.width_m / 2, 0, -scene.container.depth_m / 2],
          max: [scene.container.width_m / 2, scene.container.height_m, scene.container.depth_m / 2]
        },
        containerClosedTop: scene.container.top === "closed",
        exposure: 1,
        gridOpacity: 0.88
      });
    }
    const fallbackSprayRendered = !rasterized && Boolean(this.secondaryParticlePipeline?.encode(encoder, this.presentationTexture.createView(), sampleRenderGPU&&this.renderQuerySet?{querySet:this.renderQuerySet,beginningOfPassWriteIndex:12,endOfPassWriteIndex:13}:undefined));
    if (gridOverlay?.axis !== "off") this.gridOverlayPipeline?.encode(encoder, this.presentationTexture.createView());
    const upscalePass=encoder.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:{r:0.01,g:0.025,b:0.024,a:1},loadOp:"clear",storeOp:"store"}],...(sampleRenderGPU&&this.renderQuerySet?{timestampWrites:{querySet:this.renderQuerySet,beginningOfPassWriteIndex:10,endOfPassWriteIndex:11}}:{})});
    upscalePass.setPipeline(this.upscalePipeline);upscalePass.setBindGroup(0,this.upscaleBindGroup);upscalePass.draw(3);upscalePass.end();
    let renderReadback:GPUBuffer|undefined;if(sampleRenderGPU&&this.renderQuerySet&&this.renderQueryResolve){this.lastRenderQueryAt=renderStart;this.renderReadbackPending=true;encoder.resolveQuerySet(this.renderQuerySet,0,16,this.renderQueryResolve,0);renderReadback=this.device.createBuffer({size:16*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});encoder.copyBufferToBuffer(this.renderQueryResolve,0,renderReadback,0,16*8);}
    this.device.queue.submit([encoder.finish()]);
    if(renderReadback){const readback=renderReadback,sampledContext=timingContext,sampledRasterized=rasterized,surfaceUpdated=Boolean(rasterResult&&rasterResult.surfaceUpdated),sampledSprayRendered=rasterized?Boolean(rasterResult&&rasterResult.sprayRendered):fallbackSprayRendered;void readback.mapAsync(GPUMapMode.READ).then(()=>{const stage=decodeRenderStageTimestamps(new BigUint64Array(readback.getMappedRange()),sampledRasterized,surfaceUpdated,sampledSprayRendered);if(this.renderTimingContext===sampledContext){this.gpuSurfaceExtraction_ms=stage.surfaceExtraction_ms;this.gpuDryScene_ms=stage.dryScene_ms;this.gpuInterfaces_ms=stage.interfaces_ms;this.gpuSprayFront_ms=stage.sprayFront_ms;this.gpuSprayBack_ms=stage.sprayBack_ms;this.gpuSprayRender_ms=stage.sprayRender_ms;this.gpuOpticalComposite_ms=stage.opticalComposite_ms;this.gpuUpscale_ms=stage.upscale_ms;this.gpuRender_ms=stage.total_ms;}readback.unmap();readback.destroy();}).catch(()=>readback.destroy()).finally(()=>{this.renderReadbackPending=false;});}
    return {cpuFrame_ms:performance.now()-start,cpuPhysicsSubmit_ms,cpuDataUpload_ms,cpuRenderEncode_ms:performance.now()-renderStart,gpuRender_ms:this.gpuRender_ms,gpuSurfaceExtraction_ms:this.gpuSurfaceExtraction_ms,gpuDryScene_ms:this.gpuDryScene_ms,gpuInterfaces_ms:this.gpuInterfaces_ms,gpuSprayFront_ms:this.gpuSprayFront_ms,gpuSprayBack_ms:this.gpuSprayBack_ms,gpuSprayRender_ms:this.gpuSprayRender_ms,gpuOpticalComposite_ms:this.gpuOpticalComposite_ms,gpuUpscale_ms:this.gpuUpscale_ms,methodId:config.methodId,waterRenderMode,gpuRenderTimestampAvailable:Boolean(this.renderQuerySet)};
  }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    const fluid = this.gpuFluid;
    this.gpuFluid = undefined;
    this.gpuFluidRequestGeneration += 1;
    this.gpuFluidPendingKey = "";
    this.resetGPUQueueTracking();
    this.gpuFluidGeneration += 1;
    try { fluid?.destroy(); } catch { /* Device loss can invalidate solver resources first. */ }
    for (const retired of this.retiredGPUFluids) { try { retired.destroy(); } catch { /* Best-effort cleanup after device loss. */ } }
    this.retiredGPUFluids.clear();
    try { this.waterPipeline?.destroy(); } catch { /* Best-effort cleanup after device loss. */ }
    try { this.gridOverlayPipeline?.destroy(); } catch { /* Best-effort cleanup after device loss. */ }
    try { this.voxelDebugPipeline?.destroy(); } catch { /* Best-effort cleanup after device loss. */ }
    for (const resource of [this.presentationTexture, this.voxelDebugDepth, this.fluidTexture, this.columnBaseTexture, this.gridCellTexture, this.velocityFallbackTexture, this.pressureSamplesFallbackTexture, this.scalarFallbackTexture, this.uniformBuffer, this.bodyBuffer, this.renderQuerySet, this.renderQueryResolve]) {
      try { resource?.destroy(); } catch { /* Best-effort cleanup during hot reload. */ }
    }
    try { this.device?.destroy(); } catch { /* The device may already be lost. */ }
  }
}
