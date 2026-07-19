export type GPUStageCaptureLane = "physics" | "presentation";
export type GPUStageCapturePhase = "idle" | "armed" | "encoding" | "submitted" | "reading" | "ready" | "failed";
export type GPUStageCaptureVisualization = "scalar" | "diverging" | "magnitude" | "color" | "normal" | "categorical";

export interface GPUStageCaptureBaseline {
  methodId: string;
  renderTimingContext?: string;
  stage_ms?: number;
  gpuTotal_ms?: number;
  sampleCount: number;
}

export interface GPUStageCaptureRequest {
  lane: GPUStageCaptureLane;
  stageKey: string;
  resourceId: string;
  label: string;
  selector: 0 | 1 | 2 | 3 | 4 | 5;
  selectorLabel: string;
  visualization: GPUStageCaptureVisualization;
  units?: string;
  axis?: 0 | 1 | 2;
  slice?: number;
  nearZero?: number;
  baseline: GPUStageCaptureBaseline;
}

export interface GPUStageCaptureIdentity {
  methodId: string;
  sceneId?: string;
  simulationTime_s?: number;
  step?: number;
  rendererContext?: string;
  generation?: number;
}

export interface GPUStageCaptureArtifact {
  captureId: number;
  lane: GPUStageCaptureLane;
  stageKey: string;
  resourceId: string;
  label: string;
  selectorLabel: string;
  visualization: GPUStageCaptureVisualization;
  units?: string;
  dimensions: [number, number, number];
  previewWidth: number;
  previewHeight: number;
  previewRgba: Uint8ClampedArray;
  totalValues: number;
  invalidValues: number;
  nearZeroValues: number;
  negativeValues: number;
  positiveValues: number;
  minimum: number;
  maximum: number;
  histogram: number[];
  interpretation: string;
  identity: GPUStageCaptureIdentity;
  baseline: GPUStageCaptureBaseline;
  readbackWall_ms: number;
  stagingBytes: number;
  instrumented: true;
}

export interface GPUStageCaptureState {
  revision: number;
  phase: GPUStageCapturePhase;
  captureId: number;
  request?: GPUStageCaptureRequest;
  artifact?: GPUStageCaptureArtifact;
  message?: string;
}

type CaptureListener = () => void;

const initialState: GPUStageCaptureState = { revision: 0, phase: "idle", captureId: 0 };

class GPUStageCaptureCoordinator {
  private state = initialState;
  private listeners = new Set<CaptureListener>();

  getSnapshot = () => this.state;
  getServerSnapshot = () => initialState;
  subscribe = (listener: CaptureListener) => { this.listeners.add(listener); return () => this.listeners.delete(listener); };

  private publish(patch: Omit<Partial<GPUStageCaptureState>, "revision">) {
    this.state = { ...this.state, ...patch, revision: this.state.revision + 1 };
    for (const listener of this.listeners) listener();
  }

  arm(request: GPUStageCaptureRequest) {
    const captureId = this.state.captureId + 1;
    this.state = { revision: this.state.revision + 1, phase: "armed", captureId, request, message: "Waiting for the next matching GPU stage" };
    for (const listener of this.listeners) listener();
    return captureId;
  }

  cancel() {
    // Advance the token so an already-submitted map cannot republish after the
    // user cancels while its readback is still in flight.
    this.state = { revision: this.state.revision + 1, phase: "idle", captureId: this.state.captureId + 1 };
    for (const listener of this.listeners) listener();
  }

  matches(lane: GPUStageCaptureLane, stageKey: string) {
    return this.state.phase === "armed" && this.state.request?.lane === lane && this.state.request.stageKey === stageKey;
  }

  claim(lane: GPUStageCaptureLane, stageKey: string) {
    if (!this.matches(lane, stageKey) || !this.state.request) return undefined;
    const claim = { captureId: this.state.captureId, request: this.state.request };
    this.publish({ phase: "encoding", message: `Encoding ${this.state.request.label}` });
    return claim;
  }

  submitted(captureId: number) {
    if (this.state.captureId !== captureId || this.state.phase !== "encoding") return false;
    this.publish({ phase: "submitted", message: "Diagnostic work submitted" });
    return true;
  }

  reading(captureId: number) {
    if (this.state.captureId !== captureId || (this.state.phase !== "submitted" && this.state.phase !== "encoding")) return false;
    this.publish({ phase: "reading", message: "Reading bounded diagnostic products" });
    return true;
  }

  complete(captureId: number, artifact: GPUStageCaptureArtifact) {
    if (this.state.captureId !== captureId) return;
    this.publish({ phase: "ready", artifact, message: "Capture ready" });
  }

  fail(captureId: number, error: unknown) {
    if (this.state.captureId !== captureId) return;
    const message = error instanceof Error ? error.message : String(error);
    this.publish({ phase: "failed", message });
  }
}

export const gpuStageCapture = new GPUStageCaptureCoordinator();

export interface GPUStageCaptureTarget {
  lane: GPUStageCaptureLane;
  stageKey: string;
  resourceId: string;
  label: string;
  selector: GPUStageCaptureRequest["selector"];
  selectorLabel: string;
  visualization: GPUStageCaptureVisualization;
  units?: string;
  methods?: readonly string[];
  nearZero?: number;
}

export const GPU_STAGE_CAPTURE_TARGETS: readonly GPUStageCaptureTarget[] = [
  { lane: "physics", stageKey: "advection", resourceId: "predicted-velocity", label: "Predicted velocity", selector: 4, selectorLabel: "speed", visualization: "magnitude", units: "m/s", nearZero: 1e-4 },
  { lane: "physics", stageKey: "pressure", resourceId: "pressure", label: "Pressure field", selector: 0, selectorLabel: "pressure", visualization: "diverging", units: "Pa", nearZero: 1e-5 },
  { lane: "physics", stageKey: "projection", resourceId: "projected-velocity", label: "Projected velocity", selector: 4, selectorLabel: "speed", visualization: "magnitude", units: "m/s", nearZero: 1e-4 },
  { lane: "physics", stageKey: "surface-update", resourceId: "surface-phi", label: "Updated signed distance", selector: 0, selectorLabel: "phi", visualization: "diverging", units: "m", nearZero: 1e-4, methods: ["octree", "quadtree-tall-cell"] },
  { lane: "physics", stageKey: "topology", resourceId: "topology-owner", label: "Adaptive cell ownership", selector: 0, selectorLabel: "owner", visualization: "categorical", methods: ["octree", "quadtree-tall-cell"] },
  { lane: "presentation", stageKey: "dry-scene", resourceId: "dry-scene-hdr", label: "Dry-scene HDR output", selector: 5, selectorLabel: "luminance", visualization: "color" },
  { lane: "presentation", stageKey: "svo-temporal", resourceId: "svo-temporal-output", label: "SVO temporal output", selector: 5, selectorLabel: "luminance", visualization: "color" },
  { lane: "presentation", stageKey: "interfaces", resourceId: "front-normal", label: "Front-interface normal", selector: 4, selectorLabel: "normal length", visualization: "normal" },
  { lane: "presentation", stageKey: "composite", resourceId: "optical-composite", label: "Optical composite", selector: 5, selectorLabel: "luminance", visualization: "color" },
] as const;

export function captureTargetForStage(methodId: string, lane: GPUStageCaptureLane, stageKey: string) {
  return GPU_STAGE_CAPTURE_TARGETS.find((target) => target.lane === lane && target.stageKey === stageKey && (!target.methods || target.methods.includes(methodId)));
}

const SUMMARY_BYTES = 256;
const SUMMARY_WORDS = SUMMARY_BYTES / 4;
const HISTOGRAM_OFFSET = 8;
const HISTOGRAM_BINS = 32;
const MAX_PREVIEW_EDGE = 256;

type CaptureDimension = "2d" | "3d";

interface CapturePipelines {
  layout: GPUBindGroupLayout;
  analyze: GPUComputePipeline;
  preview: GPUComputePipeline;
}

type CaptureSampleType = "float" | "uint";
type DeviceCapturePipelines = Map<string, CapturePipelines>;

const devicePipelines = new WeakMap<GPUDevice, DeviceCapturePipelines>();

const shaderFor = (dimension: CaptureDimension, sampleType: CaptureSampleType) => /* wgsl */`
struct Params {
  size: vec4u,
  view: vec4u,
  threshold: vec4u,
}
@group(0) @binding(0) var sourceTexture: texture_${dimension}<${sampleType === "float" ? "f32" : "u32"}>;
@group(0) @binding(1) var<storage, read_write> summary: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> previewValues: array<vec4f>;

fn finiteValue(value: f32) -> bool { return value == value && abs(value) <= 3.402823e38; }
fn orderedFloat(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  if ((bits & 0x80000000u) != 0u) { return ~bits; }
  return bits ^ 0x80000000u;
}
fn floatFromOrdered(value: u32) -> f32 {
  let bits = select(~value, value ^ 0x80000000u, (value & 0x80000000u) != 0u);
  return bitcast<f32>(bits);
}
fn metric(value: vec4f) -> f32 {
  switch params.view.w {
    case 0u: { return value.x; }
    case 1u: { return value.y; }
    case 2u: { return value.z; }
    case 3u: { return value.w; }
    case 4u: { return length(value.xyz); }
    default: { return dot(value.rgb, vec3f(0.2126, 0.7152, 0.0722)); }
  }
}
fn loadSource(id: vec3u) -> vec4f {
  ${dimension === "3d" ? `return ${sampleType === "float" ? "" : "vec4f("}textureLoad(sourceTexture, vec3i(id), 0)${sampleType === "float" ? "" : ")"};` : `return ${sampleType === "float" ? "" : "vec4f("}textureLoad(sourceTexture, vec2i(id.xy), 0)${sampleType === "float" ? "" : ")"};`}
}

@compute @workgroup_size(${dimension === "3d" ? "4, 4, 4" : "8, 8, 1"})
fn analyze(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.size.x || id.y >= params.size.y || id.z >= params.size.z) { return; }
  let value = metric(loadSource(id));
  atomicAdd(&summary[0], 1u);
  if (!finiteValue(value)) { atomicAdd(&summary[1], 1u); return; }
  atomicMin(&summary[2], orderedFloat(value));
  atomicMax(&summary[3], orderedFloat(value));
  if (abs(value) <= bitcast<f32>(params.threshold.x)) { atomicAdd(&summary[4], 1u); }
  if (value < 0.0) { atomicAdd(&summary[5], 1u); }
  if (value > 0.0) { atomicAdd(&summary[6], 1u); }
}

fn mapIndex(pixel: u32, outputSize: u32, sourceSize: u32) -> u32 {
  if (outputSize <= 1u || sourceSize <= 1u) { return 0u; }
  return min(sourceSize - 1u, (pixel * (sourceSize - 1u) + (outputSize - 1u) / 2u) / (outputSize - 1u));
}
fn previewSource(pixel: vec2u) -> vec3u {
  if (${dimension === "2d" ? "true" : "false"}) {
    return vec3u(mapIndex(pixel.x, params.size.w, params.size.x), mapIndex(pixel.y, params.view.x, params.size.y), 0u);
  }
  if (params.view.y == 1u) {
    return vec3u(params.view.z, mapIndex(pixel.y, params.view.x, params.size.y), mapIndex(pixel.x, params.size.w, params.size.z));
  }
  if (params.view.y == 2u) {
    return vec3u(mapIndex(pixel.x, params.size.w, params.size.x), params.view.z, mapIndex(pixel.y, params.view.x, params.size.z));
  }
  return vec3u(mapIndex(pixel.x, params.size.w, params.size.x), mapIndex(pixel.y, params.view.x, params.size.y), params.view.z);
}

@compute @workgroup_size(8, 8, 1)
fn preview(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.size.w || id.y >= params.view.x) { return; }
  let raw = loadSource(previewSource(id.xy));
  let value = metric(raw);
  previewValues[id.y * params.size.w + id.x] = raw;
  if (!finiteValue(value)) { return; }
  let minimum = floatFromOrdered(atomicLoad(&summary[2]));
  let maximum = floatFromOrdered(atomicLoad(&summary[3]));
  let normalized = clamp((value - minimum) / max(maximum - minimum, 1e-20), 0.0, 0.999999);
  let bin = u32(normalized * ${HISTOGRAM_BINS}.0);
  atomicAdd(&summary[${HISTOGRAM_OFFSET}u + bin], 1u);
}
`;

function createPipelines(device: GPUDevice, dimension: CaptureDimension, sampleType: CaptureSampleType): CapturePipelines {
  const layout = device.createBindGroupLayout({
    label: `GPU stage capture ${dimension} bindings`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: sampleType === "float" ? "unfilterable-float" : "uint", viewDimension: dimension } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const shaderModule = device.createShaderModule({ label: `GPU stage capture ${dimension} ${sampleType} shader`, code: shaderFor(dimension, sampleType) });
  void shaderModule.getCompilationInfo().then((info) => {
    for (const message of info.messages) if (message.type === "error") console.error(`GPU stage capture WGSL ${message.lineNum}:${message.linePos} ${message.message}`);
  }).catch(() => { /* Device loss is reported by the renderer. */ });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  return {
    layout,
    analyze: device.createComputePipeline({ label: `Analyze ${dimension} diagnostic resource`, layout: pipelineLayout, compute: { module: shaderModule, entryPoint: "analyze" } }),
    preview: device.createComputePipeline({ label: `Preview ${dimension} diagnostic resource`, layout: pipelineLayout, compute: { module: shaderModule, entryPoint: "preview" } }),
  };
}

function pipelinesFor(device: GPUDevice, dimension: CaptureDimension, sampleType: CaptureSampleType) {
  const cached = devicePipelines.get(device) ?? new Map<string, CapturePipelines>();
  devicePipelines.set(device, cached);
  const key = `${dimension}:${sampleType}`;
  const existing = cached.get(key);
  if (existing) return existing;
  const created = createPipelines(device, dimension, sampleType);
  cached.set(key, created);
  return created;
}

function previewDimensions(dimensions: [number, number, number], dimension: CaptureDimension, axis: 0 | 1 | 2) {
  const [width, height] = dimension === "2d" || axis === 0
    ? [dimensions[0], dimensions[1]]
    : axis === 1 ? [dimensions[2], dimensions[1]] : [dimensions[0], dimensions[2]];
  const scale = Math.min(1, MAX_PREVIEW_EDGE / Math.max(width, height, 1));
  return [Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale))] as const;
}

function orderedFloat(value: number) {
  const float = new Float32Array([value]);
  const bits = new Uint32Array(float.buffer)[0];
  return (bits & 0x8000_0000) !== 0 ? (~bits >>> 0) : (bits ^ 0x8000_0000) >>> 0;
}

function floatFromOrdered(value: number) {
  const bits = (value & 0x8000_0000) !== 0 ? (value ^ 0x8000_0000) >>> 0 : (~value >>> 0);
  return new Float32Array(new Uint32Array([bits]).buffer)[0];
}

function selectedMetric(values: Float32Array, offset: number, selector: GPUStageCaptureRequest["selector"]) {
  if (selector <= 3) return values[offset + selector];
  if (selector === 4) return Math.hypot(values[offset], values[offset + 1], values[offset + 2]);
  return values[offset] * 0.2126 + values[offset + 1] * 0.7152 + values[offset + 2] * 0.0722;
}

const clampByte = (value: number) => Math.max(0, Math.min(255, Math.round(value * 255)));

function viridis(value: number) {
  const t = Math.max(0, Math.min(1, value));
  return [
    0.267 + t * (0.993 - 0.267),
    0.005 + Math.sin(t * Math.PI) * 0.75 + t * 0.15,
    0.329 + (1 - t) * 0.2 - t * 0.185,
  ];
}

function categorical(value: number) {
  const id = Math.max(0, Math.round(value)) >>> 0;
  return [((id * 97) % 251) / 250, ((id * 57 + 43) % 241) / 240, ((id * 17 + 101) % 239) / 238];
}

function previewRgba(values: Float32Array, request: GPUStageCaptureRequest, minimum: number, maximum: number) {
  const pixels = new Uint8ClampedArray(values.length / 4 * 4);
  const absoluteRange = Math.max(Math.abs(minimum), Math.abs(maximum), 1e-20);
  const range = Math.max(maximum - minimum, 1e-20);
  for (let offset = 0; offset < values.length; offset += 4) {
    const metric = selectedMetric(values, offset, request.selector);
    let color: number[];
    if (![values[offset], values[offset + 1], values[offset + 2], metric].every(Number.isFinite)) color = [1, 0, 1];
    else if (request.visualization === "color") color = [0, 1, 2].map((channel) => Math.pow(Math.max(0, values[offset + channel]) / (1 + Math.max(0, values[offset + channel])), 1 / 2.2));
    else if (request.visualization === "normal") color = [values[offset] * 0.5 + 0.5, values[offset + 1] * 0.5 + 0.5, values[offset + 2] * 0.5 + 0.5];
    else if (request.visualization === "categorical") color = categorical(metric);
    else if (request.visualization === "diverging") {
      const signed = Math.max(-1, Math.min(1, metric / absoluteRange));
      color = signed < 0 ? [0.12, 0.35 + 0.35 * (1 + signed), 0.95] : [0.95, 0.35 + 0.35 * (1 - signed), 0.12];
    } else color = viridis((metric - minimum) / range);
    pixels[offset] = clampByte(color[0]); pixels[offset + 1] = clampByte(color[1]); pixels[offset + 2] = clampByte(color[2]); pixels[offset + 3] = 255;
  }
  return pixels;
}

function interpretation(request: GPUStageCaptureRequest, total: number, invalid: number, nearZero: number, minimum: number, maximum: number) {
  if (total === 0) return `No shader invocations reached the diagnostic pass. Verify that this stage dispatched work and that its captured extent is current.`;
  if (invalid > 0) return `${invalid.toLocaleString()} of ${total.toLocaleString()} values are non-finite; inspect this stage before performance tuning.`;
  const nearZeroFraction = total > 0 ? nearZero / total : 0;
  if (nearZeroFraction >= 0.7) return `${(nearZeroFraction * 100).toFixed(1)}% of the full ${request.selectorLabel} domain is at or below ${request.nearZero ?? 1e-6}${request.units ? ` ${request.units}` : ""}; domain or activity gating is a strong candidate.`;
  return `${request.selectorLabel} spans ${minimum.toPrecision(4)} to ${maximum.toPrecision(4)}${request.units ? ` ${request.units}` : ""}; ${(nearZeroFraction * 100).toFixed(1)}% of the domain is near zero.`;
}

export interface EncodeGPUStageTextureCaptureOptions {
  device: GPUDevice;
  encoder: GPUCommandEncoder;
  lane: GPUStageCaptureLane;
  stageKey: string;
  texture: GPUTexture;
  dimension: CaptureDimension;
  sampleType?: CaptureSampleType;
  dimensions: [number, number, number];
  identity: GPUStageCaptureIdentity;
}

export interface PendingGPUStageCapture {
  captureId: number;
  afterSubmit(): void;
}

export function encodeGPUStageTextureCapture(options: EncodeGPUStageTextureCaptureOptions): PendingGPUStageCapture | undefined {
  const claim = gpuStageCapture.claim(options.lane, options.stageKey);
  if (!claim) return undefined;
  const startedAt = performance.now();
  try {
    const request = claim.request;
    const axis = options.dimension === "2d" ? 0 : request.axis ?? 0;
    const [previewWidth, previewHeight] = previewDimensions(options.dimensions, options.dimension, axis);
    const previewBytes = previewWidth * previewHeight * 16;
    const params = options.device.createBuffer({ label: "GPU stage capture parameters", size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const summary = options.device.createBuffer({ label: "GPU stage capture summary", size: SUMMARY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const preview = options.device.createBuffer({ label: "GPU stage capture preview", size: previewBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = options.device.createBuffer({ label: "GPU stage capture readback", size: SUMMARY_BYTES + previewBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const sliceSize = axis === 0 ? options.dimensions[2] : axis === 1 ? options.dimensions[0] : options.dimensions[1];
    const slice = Math.min(sliceSize - 1, Math.max(0, Math.round((request.slice ?? 0.5) * Math.max(0, sliceSize - 1))));
    const parameterWords = new Uint32Array(12);
    parameterWords.set([options.dimensions[0], options.dimensions[1], options.dimensions[2], previewWidth, previewHeight, axis, slice, request.selector]);
    parameterWords[8] = new Uint32Array(new Float32Array([request.nearZero ?? 1e-6]).buffer)[0];
    const initialSummary = new Uint32Array(SUMMARY_WORDS);
    initialSummary[2] = orderedFloat(Number.POSITIVE_INFINITY);
    initialSummary[3] = orderedFloat(Number.NEGATIVE_INFINITY);
    options.device.queue.writeBuffer(params, 0, parameterWords);
    options.device.queue.writeBuffer(summary, 0, initialSummary);
    const pipelines = pipelinesFor(options.device, options.dimension, options.sampleType ?? "float");
    const group = options.device.createBindGroup({
      label: `GPU stage capture ${request.resourceId}`,
      layout: pipelines.layout,
      entries: [
        { binding: 0, resource: options.texture.createView({ dimension: options.dimension }) },
        { binding: 1, resource: { buffer: summary } },
        { binding: 2, resource: { buffer: params } },
        { binding: 3, resource: { buffer: preview } },
      ],
    });
    const analyze = options.encoder.beginComputePass({ label: `Analyze ${request.label}` });
    analyze.setPipeline(pipelines.analyze); analyze.setBindGroup(0, group);
    analyze.dispatchWorkgroups(Math.ceil(options.dimensions[0] / (options.dimension === "3d" ? 4 : 8)), Math.ceil(options.dimensions[1] / (options.dimension === "3d" ? 4 : 8)), options.dimension === "3d" ? Math.ceil(options.dimensions[2] / 4) : 1);
    analyze.end();
    const previewPass = options.encoder.beginComputePass({ label: `Preview ${request.label}` });
    previewPass.setPipeline(pipelines.preview); previewPass.setBindGroup(0, group);
    previewPass.dispatchWorkgroups(Math.ceil(previewWidth / 8), Math.ceil(previewHeight / 8)); previewPass.end();
    options.encoder.copyBufferToBuffer(summary, 0, readback, 0, SUMMARY_BYTES);
    options.encoder.copyBufferToBuffer(preview, 0, readback, SUMMARY_BYTES, previewBytes);

    return {
      captureId: claim.captureId,
      afterSubmit() {
        if (!gpuStageCapture.submitted(claim.captureId)) { params.destroy(); summary.destroy(); preview.destroy(); readback.destroy(); return; }
        gpuStageCapture.reading(claim.captureId);
        void readback.mapAsync(GPUMapMode.READ).then(() => {
          const mapped = readback.getMappedRange();
          const words = new Uint32Array(mapped, 0, SUMMARY_WORDS);
          const copiedPreview = new Float32Array(previewWidth * previewHeight * 4);
          copiedPreview.set(new Float32Array(mapped, SUMMARY_BYTES, copiedPreview.length));
          const total = words[0], invalid = words[1];
          const minimum = total > invalid ? floatFromOrdered(words[2]) : Number.NaN;
          const maximum = total > invalid ? floatFromOrdered(words[3]) : Number.NaN;
          const artifact: GPUStageCaptureArtifact = {
            captureId: claim.captureId,
            lane: request.lane,
            stageKey: request.stageKey,
            resourceId: request.resourceId,
            label: request.label,
            selectorLabel: request.selectorLabel,
            visualization: request.visualization,
            units: request.units,
            dimensions: options.dimensions,
            previewWidth,
            previewHeight,
            previewRgba: previewRgba(copiedPreview, request, minimum, maximum),
            totalValues: total,
            invalidValues: invalid,
            nearZeroValues: words[4],
            negativeValues: words[5],
            positiveValues: words[6],
            minimum,
            maximum,
            histogram: Array.from(words.slice(HISTOGRAM_OFFSET, HISTOGRAM_OFFSET + HISTOGRAM_BINS)),
            interpretation: interpretation(request, total, invalid, words[4], minimum, maximum),
            identity: options.identity,
            baseline: request.baseline,
            readbackWall_ms: performance.now() - startedAt,
            stagingBytes: SUMMARY_BYTES + previewBytes,
            instrumented: true,
          };
          readback.unmap();
          gpuStageCapture.complete(claim.captureId, artifact);
        }).catch((error) => gpuStageCapture.fail(claim.captureId, error)).finally(() => {
          params.destroy(); summary.destroy(); preview.destroy(); readback.destroy();
        });
      },
    };
  } catch (error) {
    gpuStageCapture.fail(claim.captureId, error);
    return undefined;
  }
}
