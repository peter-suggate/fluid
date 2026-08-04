import { OCTREE_LOSASSO_INVALID_ROW } from "./octree-losasso-operator";
import type { PassBroker } from "./webgpu-pass-broker";

export const octreeLosassoProjectionWGSL = /* wgsl */ `
const INVALID_ROW: u32 = ${OCTREE_LOSASSO_INVALID_ROW}u;
struct Face { negativeRow: u32, positiveRow: u32, axis: u32, reserved: u32,
  area: f32, inverseDistance: f32, openFraction: f32, normalVelocity: f32 };
struct Params { pressureScale: f32, reserved0: u32, reserved1: u32, reserved2: u32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> authority: array<u32>;
@group(0) @binding(2) var<storage, read> faces: array<Face>;
@group(0) @binding(3) var<storage, read> pressure: array<f32>;
@group(0) @binding(4) var<storage, read> predictedVelocity: array<f32>;
@group(0) @binding(5) var<storage, read_write> projectedVelocity: array<f32>;
@compute @workgroup_size(64)
fn projectLosassoFaces(@builtin(global_invocation_id) invocation: vec3u) {
  let faceId = invocation.x;
  if (authority[3] != 1u || faceId >= authority[2]) { return; }
  let face = faces[faceId];
  let rowOnPositiveSide = (face.reserved & 0x80000000u) != 0u;
  var negativePressure = pressure[face.negativeRow];
  var positivePressure = 0.0;
  if (face.positiveRow != INVALID_ROW) { positivePressure = pressure[face.positiveRow]; }
  if (rowOnPositiveSide) {
    positivePressure = pressure[face.negativeRow];
    negativePressure = 0.0;
  }
  projectedVelocity[faceId] = predictedVelocity[faceId] - params.pressureScale
    * (positivePressure - negativePressure)
    * face.inverseDistance * face.openFraction;
}
`;

export interface WebGPUOctreeLosassoProjectionSource {
  readonly control: GPUBuffer;
  readonly faces: GPUBuffer;
  readonly faceDispatch: GPUBuffer;
  readonly faceDispatchOffsetBytes?: number;
  /** Axis-face velocity after advection, body force, and solid aperture. */
  readonly predictedVelocity: GPUBuffer;
  readonly projectedVelocity: GPUBuffer;
}

/** Six-binding axis-face projection; output is the extension seed field. */
export class WebGPUOctreeLosassoProjection {
  readonly bindingCount = 6;
  private readonly params: GPUBuffer;
  private readonly layout: GPUBindGroupLayout;
  private readonly pipelineLayout: GPUPipelineLayout;
  private pipeline?: GPUComputePipeline;
  private readonly groups = new WeakMap<GPUBuffer, GPUBindGroup>();
  private pressureScale: number;
  private destroyed = false;

  constructor(private readonly device: GPUDevice,
    readonly source: WebGPUOctreeLosassoProjectionSource,
    pressureScale = 1) {
    if (!Number.isFinite(pressureScale) || pressureScale < 0) {
      throw new RangeError("Losasso projection scale must be non-negative and finite");
    }
    this.pressureScale = pressureScale;
    this.params = device.createBuffer({ label: "Losasso projection constants", size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.params, 0, Float32Array.of(pressureScale, 0, 0, 0));
    const readOnly = { type: "read-only-storage" as const };
    this.layout = device.createBindGroupLayout({ label: "Losasso axis-face projection reduced layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ] });
    this.pipelineLayout = device.createPipelineLayout({ label: "Losasso axis-face projection pipeline layout",
      bindGroupLayouts: [this.layout] });
  }

  get initializationTasks(): readonly { readonly label: string; readonly run: () => Promise<void> }[] {
    return [{ label: "Compile Losasso axis-face projection", run: () => this.initialize() }];
  }
  async initialize(): Promise<void> {
    this.assertLive(); if (this.pipeline) return;
    const shaderModule = this.device.createShaderModule({ label: "Losasso axis-face projection shader", code: octreeLosassoProjectionWGSL });
    this.pipeline = await this.device.createComputePipelineAsync({ label: "Project Losasso axis faces",
      layout: this.pipelineLayout, compute: { module: shaderModule, entryPoint: "projectLosassoFaces" } });
  }
  encode(broker: PassBroker, pressure: GPUBuffer, pressureScale = this.pressureScale): void {
    this.assertLive(); if (!this.pipeline) throw new Error("Losasso projection is not initialized");
    if (!Number.isFinite(pressureScale) || pressureScale < 0) {
      throw new RangeError("Losasso projection scale must be non-negative and finite");
    }
    if (pressureScale !== this.pressureScale) {
      this.pressureScale = pressureScale;
      this.device.queue.writeBuffer(this.params, 0, Float32Array.of(pressureScale));
    }
    let group = this.groups.get(pressure);
    if (!group) {
      group = this.device.createBindGroup({ label: "Losasso axis-face projection bindings", layout: this.layout,
        entries: [this.params, this.source.control, this.source.faces, pressure,
          this.source.predictedVelocity, this.source.projectedVelocity]
          .map((buffer, binding) => ({ binding, resource: { buffer } })) });
      this.groups.set(pressure, group);
    }
    const pass = broker.compute({ label: "Losasso pressure - axis-face projection" });
    pass.setPipeline(this.pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroupsIndirect(this.source.faceDispatch, this.source.faceDispatchOffsetBytes ?? 0);
  }
  destroy(): void { if (this.destroyed) return; this.destroyed = true; this.params.destroy(); this.pipeline = undefined; }
  private assertLive(): void { if (this.destroyed) throw new Error("Losasso projection is destroyed"); }
}
