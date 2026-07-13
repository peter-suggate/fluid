import { cameraPosition } from "./math";
import type { CameraState, SceneDescription, ViewMode } from "./model";
import type { RigidBodyState } from "./rigid-body";
import type { EulerianRenderState } from "./eulerian-solver";
import type { GPUEulerianInfo, GPURigidLoad, GPUQuality } from "./webgpu-eulerian";
import { WebGPUHierarchicalSolver } from "./webgpu-hierarchical-solver";

export type SimulationBackend = "webgpu" | "cpu-reference";

export type GPUStatus =
  | { state: "initializing"; label: string }
  | { state: "ready"; label: string; adapter: string; computeAvailable: boolean }
  | { state: "unavailable"; label: string }
  | { state: "lost"; label: string };

export interface RendererFrameMetrics {
  cpuFrame_ms: number;
  cpuPhysicsSubmit_ms: number;
  cpuDataUpload_ms: number;
  cpuRenderEncode_ms: number;
  gpuRender_ms?: number;
}

async function verifyComputeExecution(device:GPUDevice):Promise<{available:boolean;detail:string}>{
  const uploadExpected=0x13572468,expected=0x5a17c0de,output=device.createBuffer({label:"WebGPU compute capability output",size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),readback=device.createBuffer({label:"WebGPU compute capability readback",size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
    device.queue.writeBuffer(output,0,new Uint32Array([uploadExpected]));
    const copyEncoder=device.createCommandEncoder({label:"WebGPU buffer-copy capability commands"});copyEncoder.copyBufferToBuffer(output,0,readback,0,16);device.queue.submit([copyEncoder.finish()]);await readback.mapAsync(GPUMapMode.READ);const uploaded=new Uint32Array(readback.getMappedRange())[0];readback.unmap();if(uploaded!==uploadExpected)return{available:false,detail:`buffer copy mismatch (${uploaded.toString(16)})`};
    device.pushErrorScope("validation");
    const shaderModule=device.createShaderModule({label:"WebGPU compute capability shader",code:`@group(0) @binding(0) var<storage,read_write> result:array<vec4u>;@compute @workgroup_size(1) fn main(){result[0]=vec4u(${expected}u,1u,2u,3u);}`});const compilation=await shaderModule.getCompilationInfo();const shaderError=compilation.messages.find(message=>message.type==="error");if(shaderError){await device.popErrorScope();return{available:false,detail:`shader error: ${shaderError.message}`};}const pipeline=await device.createComputePipelineAsync({label:"WebGPU compute capability pipeline",layout:"auto",compute:{module:shaderModule,entryPoint:"main"}}),group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]}),encoder=device.createCommandEncoder({label:"WebGPU compute capability commands"}),pass=encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();encoder.copyBufferToBuffer(output,0,readback,0,16);device.queue.submit([encoder.finish()]);const validation=await device.popErrorScope();if(validation)return{available:false,detail:validation.message};await device.queue.onSubmittedWorkDone();await readback.mapAsync(GPUMapMode.READ);const actual=new Uint32Array(readback.getMappedRange())[0];readback.unmap();return{available:actual===expected,detail:actual===expected?"sentinel matched":`sentinel mismatch (${actual.toString(16)})`};
  }catch(error){return{available:false,detail:error instanceof Error?error.message:String(error)};}finally{output.destroy();readback.destroy();}
}

const shader = /* wgsl */ `
struct Uniforms {
  viewport: vec4f,
  cameraPosition: vec4f,
  cameraTarget: vec4f,
  container: vec4f,
  options: vec4f,
  gridInfo: vec4f,
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

fn fluidSample(cell: vec3i) -> f32 {
  let dims = vec3i(u.gridInfo.xyz);
  return textureLoad(fluidField, clamp(cell, vec3i(0), dims - vec3i(1)), 0).x;
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
  let rd = normalize(forward + right * ndc.x * aspect * 0.72 + up * ndc.y * 0.72);

  let skyT = clamp(input.uv.y, 0.0, 1.0);
  var color = mix(vec3f(0.018, 0.042, 0.041), vec3f(0.055, 0.098, 0.092), skyT);
  color += 0.025 * pow(max(dot(rd, normalize(vec3f(-0.4, 0.8, 0.3))), 0.0), 18.0);

  let size = u.container.xyz;
  let center = vec3f(0.0, size.y * 0.5, 0.0);
  let halfSize = size * 0.5;
  let boundsMin = center - halfSize;
  let boundsMax = center + halfSize;
  let hit = boxIntersection(ro, rd, boundsMin, boundsMax);
  let rigidHit = nearestBody(ro, rd);

  let floorT = (-0.025 - ro.y) / rd.y;
  if (floorT > 0.0) {
    let floorPoint = ro + rd * floorT;
    let radial = length(floorPoint.xz);
    let floorGrid = gridLine(floorPoint.xz, 0.1);
    let floorFade = exp(-radial * 0.7);
    color = mix(color, vec3f(0.045, 0.085, 0.079) + floorGrid * vec3f(0.05, 0.16, 0.135), 0.28 * floorFade);
  }

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
      let refracted=color*transmission+scatter;let reflected=mix(vec3f(0.025,0.07,0.07),vec3f(0.19,0.38,0.34),clamp(reflect(rd,normal).y*0.5+0.5,0.0,1.0));
      var waterColor = mix(refracted,reflected,fresnel);
      waterColor+=vec3f(0.025,0.12,0.105)*(1.0-exp(-thickness*7.0));
      waterColor += vec3f(0.16, 0.72, 0.64) * pow(max(dot(reflect(rd, normal), normalize(vec3f(-0.5, 0.8, 0.25))), 0.0), 64.0);

      if (scientific) {
        let grid = gridLine(waterPoint.xz, max(u.options.y, 0.01));
        waterColor = mix(waterColor, vec3f(0.42, 0.96, 0.82), grid * 0.58);
      }
      color = mix(color, waterColor, 0.82 + depth * 0.1);
    }

    let q = abs((entry - center) / max(halfSize, vec3f(0.001)));
    let edge = max(max(min(q.x, q.y), min(q.x, q.z)), min(q.y, q.z));
    let edgeAlpha = smoothstep(0.91, 0.995, edge);
    let glassFresnel = pow(1.0 - abs(dot(rd, normalize(entry - center))), 3.0);
    let glass = vec3f(0.42, 0.78, 0.72);
    color = mix(color, glass, 0.035 + glassFresnel * 0.035 + edgeAlpha * 0.54);
  }

  if (rigidHit.t < 1e19 && (floorT <= 0.0 || rigidHit.t < floorT)) {
    let rigidPoint = ro + rd * rigidHit.t;
    let light = normalize(vec3f(-0.45, 0.8, 0.3));
    let diffuse = 0.22 + 0.78 * max(dot(rigidHit.normal, light), 0.0);
    let rim = pow(1.0 - max(dot(-rd, rigidHit.normal), 0.0), 3.0);
    var rigidColor = rigidHit.color * diffuse + vec3f(0.18, 0.42, 0.37) * rim;
    if (rigidPoint.y < u.container.w) {
      let submergence = clamp((u.container.w - rigidPoint.y) / max(size.y, 0.001), 0.0, 1.0);
      rigidColor = mix(rigidColor, rigidColor * vec3f(0.35, 0.72, 0.7) + vec3f(0.01, 0.11, 0.1), 0.4 + submergence * 0.32);
    }
    rigidColor += rigidHit.selected * vec3f(0.18, 0.55, 0.43) * (0.28 + rim);
    color = rigidColor;
  }

  let vignette = 1.0 - 0.22 * dot(ndc * 0.58, ndc * 0.58);
  color *= vignette;
  color = color / (color + vec3f(1.0));
  color = pow(color, vec3f(1.0 / 2.2));
  return vec4f(color, 1.0);
}
`;

const hierarchyShader = shader
  .replace("@group(0) @binding(2) var fluidField: texture_3d<f32>;", `
struct HierarchyCell { negAlpha:vec4f, posPressure:vec4f }
struct HierarchyParams { dimsDt:vec4f, pageBrick:vec4f, originH:vec4f, physical:vec4f, containerBodies:vec4f, topology:vec4f, boundary:vec4f }
@group(0) @binding(2) var<storage,read> hierarchyCells:array<HierarchyCell>;
@group(0) @binding(3) var<storage,read> hierarchyMeta:array<vec4u>;
@group(0) @binding(4) var<storage,read> hierarchyPages:array<u32>;
@group(0) @binding(5) var<storage,read> hierarchyParams:HierarchyParams;`)
  .replace(`fn fluidSample(cell: vec3i) -> f32 {
  let dims = vec3i(u.gridInfo.xyz);
  return textureLoad(fluidField, clamp(cell, vec3i(0), dims - vec3i(1)), 0).x;
}`, `fn fluidSample(cell:vec3i)->f32 {
  let dims=vec3i(u.gridInfo.xyz);let q=vec3u(clamp(cell,vec3i(0),dims-vec3i(1)));let page=q/4u;let pd=vec3u(hierarchyParams.pageBrick.xyz);let slot=hierarchyPages[page.x+pd.x*(page.y+pd.y*page.z)];let brick=hierarchyMeta[slot*2u];let local=vec3u(clamp(floor((vec3f(q)-vec3f(brick.xyz))/f32(brick.w)),vec3f(0.0),vec3f(3.0)));let index=slot*64u+local.x+4u*(local.y+4u*local.z);return hierarchyCells[index].negAlpha.w;
}`);

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
  private context?: GPUCanvasContext;
  private pipeline?: GPURenderPipeline;
  private hierarchyPipeline?: GPURenderPipeline;
  private upscalePipeline?: GPURenderPipeline;
  private upscaleSampler?: GPUSampler;
  private upscaleBindGroup?: GPUBindGroup;
  private presentationTexture?: GPUTexture;
  private presentationTextureKey = "";
  private readonly renderScale = 1;
  private uniformBuffer?: GPUBuffer;
  private bodyBuffer?: GPUBuffer;
  private fluidTexture?: GPUTexture;
  private fluidTextureKey = "";
  private fluidRevision = -1;
  private gpuFluid?: WebGPUHierarchicalSolver;
  private gpuFluidKey = "";
  private gpuInfoCallback?: (info: GPUEulerianInfo) => void;
  private gpuRigidLoadCallback?: (loads: GPURigidLoad[]) => void;
  private lastGPUReadbackSecond = -1;
  private bindGroup?: GPUBindGroup;
  private hierarchyBindGroup?: GPUBindGroup;
  private hierarchyBindRevision = -1;
  private format?: GPUTextureFormat;
  private renderQuerySet?: GPUQuerySet;
  private renderQueryResolve?: GPUBuffer;
  private renderReadbackPending = false;
  private lastRenderQueryAt = -Infinity;
  private gpuRender_ms?: number;
  private computeAvailable = true;

  constructor(private readonly canvas: HTMLCanvasElement, private readonly onStatus: (status: GPUStatus) => void, onGPUInfo?: (info: GPUEulerianInfo) => void, onGPURigidLoads?: (loads: GPURigidLoad[]) => void) { this.gpuInfoCallback = onGPUInfo; this.gpuRigidLoadCallback = onGPURigidLoads; }

  async initialize(): Promise<void> {
    this.onStatus({ state: "initializing", label: "Requesting WebGPU adapter" });
    if (!("gpu" in navigator)) {
      this.onStatus({ state: "unavailable", label: "WebGPU is not available in this browser" });
      return;
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
      this.onStatus({ state: "unavailable", label: "No compatible GPU adapter was found" });
      return;
    }
    const requiredFeatures: GPUFeatureName[] = adapter.features.has("timestamp-query") ? ["timestamp-query"] : [];
    if(adapter.limits.maxStorageBuffersPerShaderStage<9){this.onStatus({state:"unavailable",label:"Hierarchical WebGPU requires 9 storage buffers per shader stage"});return;}
    const device = await adapter.requestDevice({ requiredFeatures,requiredLimits:{maxStorageBuffersPerShaderStage:9} });
    const computeProbe=await verifyComputeExecution(device);this.computeAvailable=computeProbe.available;
    const context = this.canvas.getContext("webgpu");
    if (!context) {
      this.onStatus({ state: "unavailable", label: "WebGPU canvas context could not be created" });
      return;
    }
    this.device = device;
    this.context = context;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    if(device.features.has("timestamp-query")){this.renderQuerySet=device.createQuerySet({type:"timestamp",count:2});this.renderQueryResolve=device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});}
    context.configure({ device, format: this.format, alphaMode: "opaque" });

    const shaderModule = device.createShaderModule({ label: "Fluid Lab presentation shader", code: shader });
    const compilation = await shaderModule.getCompilationInfo();
    const errors = compilation.messages.filter((message) => message.type === "error");
    if (errors.length > 0) throw new Error(errors.map((error) => `${error.lineNum}:${error.linePos} ${error.message}`).join("\n"));

    this.pipeline = device.createRenderPipeline({
      label: "Fluid Lab ray presentation",
      layout: "auto",
      vertex: { module: shaderModule, entryPoint: "vertexMain" },
      fragment: { module: shaderModule, entryPoint: "fragmentMain", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" }
    });
    const hierarchyModule=device.createShaderModule({label:"Fluid Lab hierarchical presentation shader",code:hierarchyShader});
    const hierarchyCompilation=await hierarchyModule.getCompilationInfo();const hierarchyErrors=hierarchyCompilation.messages.filter((message)=>message.type==="error");if(hierarchyErrors.length>0)throw new Error(hierarchyErrors.map((error)=>`${error.lineNum}:${error.linePos} ${error.message}`).join("\n"));
    this.hierarchyPipeline=device.createRenderPipeline({label:"Fluid Lab hierarchical ray presentation",layout:"auto",vertex:{module:hierarchyModule,entryPoint:"vertexMain"},fragment:{module:hierarchyModule,entryPoint:"fragmentMain",targets:[{format:this.format}]},primitive:{topology:"triangle-list"}});
    const upscaleModule=device.createShaderModule({label:"Presentation upscale shader",code:upscaleShader});
    this.upscalePipeline=device.createRenderPipeline({label:"Presentation upscale",layout:"auto",vertex:{module:upscaleModule,entryPoint:"vertexMain"},fragment:{module:upscaleModule,entryPoint:"fragmentMain",targets:[{format:this.format}]},primitive:{topology:"triangle-list"}});
    this.upscaleSampler=device.createSampler({magFilter:"linear",minFilter:"linear"});
    this.uniformBuffer = device.createBuffer({ label: "Fluid Lab view uniforms", size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bodyBuffer = device.createBuffer({ label: "Fluid Lab rigid bodies", size: 12 * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.fluidTexture = device.createTexture({ size: [1, 1, 1], dimension: "3d", format: "r8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.rebuildBindGroup();

    device.addEventListener("uncapturederror", (event) => console.error(`WebGPU validation: ${event.error.message}`));
    device.lost.then((info) => this.onStatus({ state: "lost", label: `GPU device lost: ${info.message || info.reason}` }));
    const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
    const adapterName = info ? [info.vendor, info.architecture].filter(Boolean).join(" · ") || "WebGPU adapter" : "WebGPU adapter";
    this.onStatus({state:"ready",label:this.computeAvailable?"WebGPU compute and renderer ready":`WebGPU compute unavailable (${computeProbe.detail}) · simulation paused`,adapter:adapterName,computeAvailable:this.computeAvailable});
  }

  private rebuildBindGroup(texture = this.fluidTexture) {
    if (!this.device || !this.pipeline || !this.uniformBuffer || !this.bodyBuffer || !texture) return;
    this.bindGroup = this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.bodyBuffer } },
      { binding: 2, resource: texture.createView({ dimension: "3d" }) }
    ] });
  }

  private rebuildHierarchyBindGroup(): void {
    if(!this.device||!this.hierarchyPipeline||!this.uniformBuffer||!this.bodyBuffer||!this.gpuFluid)return;const resources=this.gpuFluid.hierarchyRenderResources;if(resources.revision===this.hierarchyBindRevision)return;
    this.hierarchyBindGroup=this.device.createBindGroup({layout:this.hierarchyPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:{buffer:this.bodyBuffer}},{binding:2,resource:{buffer:resources.cells}},{binding:3,resource:{buffer:resources.metadata}},{binding:4,resource:{buffer:resources.pageTable}},{binding:5,resource:{buffer:resources.params}}]});this.hierarchyBindRevision=resources.revision;
  }

  private ensureGPUFluid(scene: SceneDescription, quality: GPUQuality, time_s: number, bodies: RigidBodyState[]) {
    if (!this.device) return undefined;
    const key = `${quality}:${scene.nominalResolution.length_m}:${JSON.stringify(scene.hierarchy)}:${scene.container.width_m}:${scene.container.height_m}:${scene.container.depth_m}:${scene.container.fillFraction}:${scene.fluid.initialCondition}:${scene.fluid.density_kg_m3}:${scene.fluid.dynamicViscosity_Pa_s}:${scene.fluid.surfaceTension_N_m}:${scene.fluid.gravity_m_s2.y}:${scene.container.fluidWallMode}`;
    if (!this.gpuFluid || key !== this.gpuFluidKey) {
      this.gpuFluid?.destroy(); this.gpuFluid = new WebGPUHierarchicalSolver(this.device, scene, quality, this.gpuRigidLoadCallback); this.gpuFluidKey = key;this.hierarchyBindRevision=-1;this.hierarchyBindGroup=undefined;
      this.rebuildBindGroup(this.gpuFluid.volumeTexture); this.gpuInfoCallback?.(this.gpuFluid.info);
    }
    if (!this.gpuFluid.advanceTo(time_s, bodies)) {
      this.gpuFluid.destroy(); this.gpuFluid = new WebGPUHierarchicalSolver(this.device, scene, quality, this.gpuRigidLoadCallback);this.hierarchyBindRevision=-1;this.hierarchyBindGroup=undefined; this.rebuildBindGroup(this.gpuFluid.volumeTexture); this.gpuInfoCallback?.(this.gpuFluid.info);
    }
    const diagnosticTick=Math.floor(time_s*10);if(diagnosticTick!==this.lastGPUReadbackSecond){this.lastGPUReadbackSecond=diagnosticTick;void this.gpuFluid.readStats().then(info=>this.gpuInfoCallback?.({...info}));}
    return this.gpuFluid.info;
  }

  private uploadFluid(fluid?: EulerianRenderState) {
    if (!this.device || !fluid) return;
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

  private clearCPUFluid():void {
    if(!this.device||!this.pipeline||this.fluidTextureKey==="")return;this.fluidTexture?.destroy();this.fluidTexture=this.device.createTexture({size:[1,1,1],dimension:"3d",format:"r8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});this.fluidTextureKey="";this.fluidRevision=-1;this.rebuildBindGroup();
  }

  resize(): void {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(this.canvas.clientWidth * ratio));
    const height = Math.max(1, Math.floor(this.canvas.clientHeight * ratio));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    if (!this.device || !this.format || !this.upscalePipeline || !this.upscaleSampler) return;
    const renderWidth = Math.max(1, Math.floor(width * this.renderScale));
    const renderHeight = Math.max(1, Math.floor(height * this.renderScale));
    const key = `${renderWidth}x${renderHeight}`;
    if (key === this.presentationTextureKey) return;
    this.presentationTexture?.destroy();
    this.presentationTexture = this.device.createTexture({label:"Raymarch presentation target",size:[renderWidth,renderHeight],format:this.format,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING});
    this.upscaleBindGroup=this.device.createBindGroup({layout:this.upscalePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:this.presentationTexture.createView()},{binding:1,resource:this.upscaleSampler}]});
    this.presentationTextureKey=key;
  }

  get presentationResolution(): string {
    if (!this.presentationTexture) return `${this.canvas.width} × ${this.canvas.height}`;
    return `${this.presentationTexture.width} × ${this.presentationTexture.height} (${Math.round(this.renderScale * 100)}%)`;
  }

  draw(time_s: number, scene: SceneDescription, camera: CameraState, view: ViewMode, bodies: RigidBodyState[], selectedBodyId?: string, fluid?: EulerianRenderState, backend: SimulationBackend = "webgpu", quality: GPUQuality = "balanced"): RendererFrameMetrics {
    if (!this.device || !this.context || !this.pipeline || !this.uniformBuffer || !this.bodyBuffer || !this.bindGroup) return {cpuFrame_ms:0,cpuPhysicsSubmit_ms:0,cpuDataUpload_ms:0,cpuRenderEncode_ms:0};
    this.resize();
    if (!this.presentationTexture || !this.upscalePipeline || !this.upscaleBindGroup) return {cpuFrame_ms:0,cpuPhysicsSubmit_ms:0,cpuDataUpload_ms:0,cpuRenderEncode_ms:0};
    const start = performance.now();
    const position = cameraPosition(camera);
    const physicsStart=performance.now();
    const useGPUFluid=backend==="webgpu"&&this.computeAvailable,gpuInfo=useGPUFluid?this.ensureGPUFluid(scene,quality,time_s,bodies):undefined;
    if(useGPUFluid)this.rebuildHierarchyBindGroup();
    const cpuPhysicsSubmit_ms=performance.now()-physicsStart,uploadStart=performance.now();
    if(backend==="cpu-reference")this.uploadFluid(fluid);else if(!useGPUFluid)this.clearCPUFluid();
    const uniform = new Float32Array([
      this.presentationTexture.width, this.presentationTexture.height, time_s, 0,
      position.x, position.y, position.z, 0,
      camera.target_m.x, camera.target_m.y, camera.target_m.z, 0,
      scene.container.width_m, scene.container.height_m, scene.container.depth_m, scene.container.height_m * scene.container.fillFraction,
      view === "scientific" ? 1 : 0, scene.nominalResolution.length_m, Math.min(bodies.length, 12), 0,
      gpuInfo?.nx ?? fluid?.nx ?? 1, gpuInfo?.ny ?? fluid?.ny ?? 1, gpuInfo?.nz ?? fluid?.nz ?? 1, gpuInfo || fluid ? 1 : 0
    ]);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniform);
    const bodyData = new Float32Array(12 * 16);
    const shapeIndex = { sphere: 0, box: 1, capsule: 2, cylinder: 3 } as const;
    const palette = [[0.95, 0.63, 0.29], [0.48, 0.66, 0.96], [0.84, 0.42, 0.48], [0.66, 0.52, 0.92]];
    bodies.slice(0, 12).forEach((body, index) => {
      const offset = index * 16;
      const d = body.description.dimensions_m;
      const half = body.description.shape === "box" ? [d.x / 2, d.y / 2, d.z / 2] : body.description.shape === "sphere" ? [d.x, d.x, d.x] : [d.x, d.y / 2, d.x];
      const color = palette[shapeIndex[body.description.shape]];
      bodyData.set([body.position_m.x, body.position_m.y, body.position_m.z, 0], offset);
      bodyData.set([half[0], half[1], half[2], shapeIndex[body.description.shape]], offset + 4);
      bodyData.set([body.orientation.w, body.orientation.x, body.orientation.y, body.orientation.z], offset + 8);
      bodyData.set([color[0], color[1], color[2], body.description.id === selectedBodyId ? 1 : 0], offset + 12);
    });
    this.device.queue.writeBuffer(this.bodyBuffer, 0, bodyData);
    const cpuDataUpload_ms=performance.now()-uploadStart,renderStart=performance.now();
    const encoder = this.device.createCommandEncoder({ label: "Fluid Lab frame" });
    const sampleRenderGPU=Boolean(this.renderQuerySet&&this.renderQueryResolve&&!this.renderReadbackPending&&renderStart-this.lastRenderQueryAt>=250);
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.presentationTexture.createView(), clearValue: { r: 0.01, g: 0.025, b: 0.024, a: 1 }, loadOp: "clear", storeOp: "store" }],
      ...(sampleRenderGPU&&this.renderQuerySet?{timestampWrites:{querySet:this.renderQuerySet,beginningOfPassWriteIndex:0}}:{})
    });
    pass.setPipeline(useGPUFluid&&this.hierarchyPipeline?this.hierarchyPipeline:this.pipeline);
    pass.setBindGroup(0,useGPUFluid&&this.hierarchyBindGroup?this.hierarchyBindGroup:this.bindGroup);
    pass.draw(3);
    pass.end();
    const upscalePass=encoder.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:{r:0.01,g:0.025,b:0.024,a:1},loadOp:"clear",storeOp:"store"}],...(sampleRenderGPU&&this.renderQuerySet?{timestampWrites:{querySet:this.renderQuerySet,endOfPassWriteIndex:1}}:{})});
    upscalePass.setPipeline(this.upscalePipeline);upscalePass.setBindGroup(0,this.upscaleBindGroup);upscalePass.draw(3);upscalePass.end();
    let renderReadback:GPUBuffer|undefined;if(sampleRenderGPU&&this.renderQuerySet&&this.renderQueryResolve){this.lastRenderQueryAt=renderStart;this.renderReadbackPending=true;encoder.resolveQuerySet(this.renderQuerySet,0,2,this.renderQueryResolve,0);renderReadback=this.device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});encoder.copyBufferToBuffer(this.renderQueryResolve,0,renderReadback,0,16);}
    this.device.queue.submit([encoder.finish()]);
    if(renderReadback){const readback=renderReadback;void readback.mapAsync(GPUMapMode.READ).then(()=>{const times=new BigUint64Array(readback.getMappedRange());this.gpuRender_ms=Number(times[1]-times[0])/1e6;readback.unmap();readback.destroy();}).catch(()=>readback.destroy()).finally(()=>{this.renderReadbackPending=false;});}
    return {cpuFrame_ms:performance.now()-start,cpuPhysicsSubmit_ms,cpuDataUpload_ms,cpuRenderEncode_ms:performance.now()-renderStart,gpuRender_ms:this.gpuRender_ms};
  }
}
