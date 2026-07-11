import { cameraPosition } from "./math";
import type { CameraState, SceneDescription, SolverMode, ViewMode } from "./model";

export type GPUStatus =
  | { state: "initializing"; label: string }
  | { state: "ready"; label: string; adapter: string }
  | { state: "unavailable"; label: string }
  | { state: "lost"; label: string };

const shader = /* wgsl */ `
struct Uniforms {
  viewport: vec4f,
  cameraPosition: vec4f,
  cameraTarget: vec4f,
  container: vec4f,
  options: vec4f,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

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

fn hash21(p: vec2f) -> f32 {
  let q = fract(p * vec2f(123.34, 456.21));
  return fract((q.x + 45.32) * (q.y + 45.32));
}

fn gridLine(value: vec2f, scale: f32) -> f32 {
  let g = abs(sin(value * 3.14159265 / max(scale, 0.0001)));
  return 1.0 - smoothstep(0.0, 0.085, min(g.x, g.y));
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let resolution = max(u.viewport.xy, vec2f(1.0));
  let time = u.viewport.z;
  let mode = i32(round(u.viewport.w));
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

    var localMode = mode;
    if (mode == 2) {
      localMode = select(1, 0, input.uv.x < 0.5);
    }

    let insideWater = waterT >= nearT && waterT <= hit.y
      && abs(waterPoint.x) <= halfSize.x && abs(waterPoint.z) <= halfSize.z;
    if (insideWater) {
      let dx = 0.088 * cos(waterPoint.x * 8.0 + time * 0.72) * cos(waterPoint.z * 7.0 - time * 0.47)
             + 0.102 * cos(waterPoint.x * 17.0 - waterPoint.z * 11.0 + time * 0.31);
      let dz = -0.077 * sin(waterPoint.x * 8.0 + time * 0.72) * sin(waterPoint.z * 7.0 - time * 0.47)
             - 0.066 * cos(waterPoint.x * 17.0 - waterPoint.z * 11.0 + time * 0.31);
      let normal = normalize(vec3f(-dx, 1.0, -dz));
      let fresnel = 0.035 + 0.62 * pow(1.0 - max(dot(normal, -rd), 0.0), 5.0);
      let depth = clamp((hit.y - waterT) / max(size.y, 0.001), 0.0, 1.0);
      var waterColor = mix(vec3f(0.018, 0.19, 0.19), vec3f(0.02, 0.42, 0.39), max(dot(normal, normalize(vec3f(-0.3, 0.8, 0.2))), 0.0));
      waterColor = mix(waterColor, color, fresnel);
      waterColor += vec3f(0.16, 0.72, 0.64) * pow(max(dot(reflect(rd, normal), normalize(vec3f(-0.5, 0.8, 0.25))), 0.0), 64.0);

      if (scientific && localMode == 0) {
        let grid = gridLine(waterPoint.xz, max(u.options.y, 0.01));
        waterColor = mix(waterColor, vec3f(0.42, 0.96, 0.82), grid * 0.58);
      }
      if (localMode == 1) {
        let particleUV = waterPoint.xz / max(u.options.y, 0.01);
        let cell = floor(particleUV);
        let local = fract(particleUV) - 0.5;
        let jitter = vec2f(hash21(cell), hash21(cell + 19.7)) - 0.5;
        let particle = 1.0 - smoothstep(0.11, 0.22, length(local - jitter * 0.38));
        waterColor = mix(waterColor * 0.58, vec3f(0.42, 0.98, 0.86), particle * 0.92);
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

  if (mode == 2) {
    let divider = 1.0 - smoothstep(0.0, 1.8 / resolution.x, abs(input.uv.x - 0.5));
    color = mix(color, vec3f(0.45, 0.96, 0.84), divider * 0.55);
  }

  let vignette = 1.0 - 0.22 * dot(ndc * 0.58, ndc * 0.58);
  color *= vignette;
  color = color / (color + vec3f(1.0));
  color = pow(color, vec3f(1.0 / 2.2));
  return vec4f(color, 1.0);
}
`;

export class FluidLabRenderer {
  private device?: GPUDevice;
  private context?: GPUCanvasContext;
  private pipeline?: GPURenderPipeline;
  private uniformBuffer?: GPUBuffer;
  private bindGroup?: GPUBindGroup;
  private format?: GPUTextureFormat;

  constructor(private readonly canvas: HTMLCanvasElement, private readonly onStatus: (status: GPUStatus) => void) {}

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
    const device = await adapter.requestDevice();
    const context = this.canvas.getContext("webgpu");
    if (!context) {
      this.onStatus({ state: "unavailable", label: "WebGPU canvas context could not be created" });
      return;
    }
    this.device = device;
    this.context = context;
    this.format = navigator.gpu.getPreferredCanvasFormat();
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
    this.uniformBuffer = device.createBuffer({ label: "Fluid Lab view uniforms", size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }]
    });

    device.lost.then((info) => this.onStatus({ state: "lost", label: `GPU device lost: ${info.message || info.reason}` }));
    const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
    const adapterName = info ? [info.vendor, info.architecture].filter(Boolean).join(" · ") || "WebGPU adapter" : "WebGPU adapter";
    this.onStatus({ state: "ready", label: "WebGPU renderer ready", adapter: adapterName });
  }

  resize(): void {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(this.canvas.clientWidth * ratio));
    const height = Math.max(1, Math.floor(this.canvas.clientHeight * ratio));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
  }

  draw(time_s: number, scene: SceneDescription, camera: CameraState, mode: SolverMode, view: ViewMode): number {
    if (!this.device || !this.context || !this.pipeline || !this.uniformBuffer || !this.bindGroup) return 0;
    this.resize();
    const start = performance.now();
    const position = cameraPosition(camera);
    const modeValue = mode === "eulerian" ? 0 : mode === "particle" ? 1 : 2;
    const uniform = new Float32Array([
      this.canvas.width, this.canvas.height, time_s, modeValue,
      position.x, position.y, position.z, 0,
      camera.target_m.x, camera.target_m.y, camera.target_m.z, 0,
      scene.container.width_m, scene.container.height_m, scene.container.depth_m, scene.container.height_m * scene.container.fillFraction,
      view === "scientific" ? 1 : 0, scene.nominalResolution.length_m, 0, 0
    ]);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniform);
    const encoder = this.device.createCommandEncoder({ label: "Fluid Lab frame" });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.context.getCurrentTexture().createView(), clearValue: { r: 0.01, g: 0.025, b: 0.024, a: 1 }, loadOp: "clear", storeOp: "store" }]
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return performance.now() - start;
  }
}
