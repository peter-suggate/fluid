import { OCTREE_LOSASSO_INVALID_ROW } from "./octree-losasso-operator";
import type { PassBroker } from "./webgpu-pass-broker";

export const octreeLosassoProjectionWGSL = /* wgsl */ `
const INVALID_ROW: u32 = ${OCTREE_LOSASSO_INVALID_ROW}u;
const FACE_SOLID_NEUMANN: u32 = 0x08000000u;
const FACE_SEPARATED: u32 = 0x20000000u;
const FACE_CLOSED_BOUNDARY: u32 = 0x40000000u;
const FACE_ROW_ON_POSITIVE_SIDE: u32 = 0x80000000u;
struct Face { negativeRow: u32, positiveRow: u32, axis: u32, reserved: u32,
  area: f32, inverseDistance: f32, openFraction: f32, normalVelocity: f32 };
struct Params { pressureScale: f32, contactPressure: f32, gravity: vec4f };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> authority: array<u32>;
@group(0) @binding(2) var<storage, read_write> faces: array<Face>;
@group(0) @binding(3) var<storage, read> pressure: array<f32>;
@group(0) @binding(4) var<storage, read> predictedVelocity: array<f32>;
@group(0) @binding(5) var<storage, read_write> projectedVelocity: array<f32>;
@group(0) @binding(6) var<storage, read> faceMetrics: array<vec4u>;
@compute @workgroup_size(64)
fn projectLosassoFaces(@builtin(global_invocation_id) invocation: vec3u) {
  let faceId = invocation.x;
  if (authority[3] != 1u || faceId >= authority[2]) { return; }
  var face = faces[faceId];
  let rowOnPositiveSide = (face.reserved & 0x80000000u) != 0u;
  var negativePressure = pressure[face.negativeRow];
  var positivePressure = 0.0;
  if (face.positiveRow != INVALID_ROW) { positivePressure = pressure[face.positiveRow]; }
  if (rowOnPositiveSide) {
    positivePressure = pressure[face.negativeRow];
    negativePressure = 0.0;
  }
  // A missing solid-interior row is Neumann, not atmospheric Dirichlet.
  let solidNeumann = (face.reserved & FACE_SOLID_NEUMANN) != 0u;
  var projected = predictedVelocity[faceId];
  if (!solidNeumann) {
    // Octree T-junction row centres can be vertically offset across an x/z
    // face. Their hydrostatic pressure difference is tangential, not a normal
    // gradient; removing it keeps gravity exactly balanced on graded rows.
    let deltaYCells = bitcast<f32>(faceMetrics[faceId].z);
    let hydrostaticTangential = params.gravity.w * params.gravity.y * deltaYCells;
    projected -= params.pressureScale
      * ((positivePressure - negativePressure) - hydrostaticTangential)
      * face.inverseDistance * face.openFraction;
  } else {
    // This face borders carved rigid interior, not atmospheric air. Enforce
    // impermeability in both directions; allowing a separating component here
    // lets gravity peel fully submerged liquid away from the lower hemisphere
    // and creates a persistent recirculation around a stationary body.
    projected = face.normalVelocity;
  }
  // Closed contact is unilateral. The fine band transports the separating
  // interface; this lagged active-set bit makes the next coarse operator use
  // the matching p=0 air ghost without post-solve pressure clamping. Every
  // wall may release under tension. A gravity-opposed ceiling gets the small
  // hydrostatic bias needed to release a stationary film before strict
  // negative pressure appears; other walls open only below ambient pressure.
  if ((face.reserved & FACE_CLOSED_BOUNDARY) != 0u) {
    let weight = length(params.gravity.xyz);
    let outward = select(1.0, -1.0, rowOnPositiveSide);
    let overhead = weight > 1e-6 && outward * (-params.gravity[face.axis] / weight) > 0.5;
    let solved = pressure[face.negativeRow];
    let finitePressure = solved == solved && abs(solved) <= 3.402823e38;
    let wasSeparated = (face.reserved & FACE_SEPARATED) != 0u;
    // A separated face is an air-pressure contact, not an open boundary. The
    // active-set bit is one solve behind, so close it immediately on approach;
    // otherwise retain only motion away from the wall and back into the tank.
    let approaching = overhead && outward * projected > 1e-6;
    if (wasSeparated) {
      projected = outward * select(max(0.0, outward * projected),
        min(0.0, outward * projected), overhead);
    }
    let releasePressure = select(0.0, params.contactPressure, overhead);
    let renewalPressure = select(max(releasePressure, 1e-4 * params.contactPressure),
      0.25 * params.contactPressure, overhead);
    let opening = finitePressure && !approaching && select(solved < releasePressure,
      solved < renewalPressure, wasSeparated);
    face.reserved = select(face.reserved & ~FACE_SEPARATED,
      face.reserved | FACE_SEPARATED, opening);
    faces[faceId] = face;
  }
  projectedVelocity[faceId] = select(0.0, projected,
    projected == projected && abs(projected) <= 3.402823e38);
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
  /** z is bitcast<f32>(positive-row centre y - negative-row centre y), in cells. */
  readonly faceMetrics: GPUBuffer;
}

/** Seven-binding axis-face projection; output is the extension seed field. */
export class WebGPUOctreeLosassoProjection {
  readonly bindingCount = 7;
  private readonly params: GPUBuffer;
  private readonly layout: GPUBindGroupLayout;
  private readonly pipelineLayout: GPUPipelineLayout;
  private pipeline?: GPUComputePipeline;
  private readonly groups = new WeakMap<GPUBuffer, GPUBindGroup>();
  private pressureScale: number;
  private gravity: readonly [number, number, number] = [0, 0, 0];
  private destroyed = false;

  constructor(private readonly device: GPUDevice,
    readonly source: WebGPUOctreeLosassoProjectionSource,
    private readonly physical: Readonly<{ density: number; physicalCellSize: number }>,
    pressureScale = 1) {
    if (!Number.isFinite(pressureScale) || pressureScale < 0) {
      throw new RangeError("Losasso projection scale must be non-negative and finite");
    }
    this.pressureScale = pressureScale;
    if (!Number.isFinite(physical.density) || physical.density <= 0
      || !Number.isFinite(physical.physicalCellSize) || physical.physicalCellSize <= 0) {
      throw new RangeError("Losasso projection physical scale must be positive and finite");
    }
    this.params = device.createBuffer({ label: "Losasso projection constants", size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.uploadParams(pressureScale, 0, [0, 0, 0]);
    const readOnly = { type: "read-only-storage" as const };
    this.layout = device.createBindGroupLayout({ label: "Losasso axis-face projection reduced layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
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
  encode(broker: PassBroker, pressure: GPUBuffer, pressureScale = this.pressureScale,
    gravity: readonly [number, number, number] = this.gravity): void {
    this.assertLive(); if (!this.pipeline) throw new Error("Losasso projection is not initialized");
    if (!Number.isFinite(pressureScale) || pressureScale < 0) {
      throw new RangeError("Losasso projection scale must be non-negative and finite");
    }
    if (gravity.some((value) => !Number.isFinite(value))) {
      throw new RangeError("Losasso projection gravity must be finite");
    }
    if (pressureScale !== this.pressureScale || gravity.some((value, axis) => value !== this.gravity[axis])) {
      this.pressureScale = pressureScale;
      this.gravity = [...gravity];
      const weight = Math.hypot(...gravity);
      const contactPressure = 0.25 * this.physical.density * weight
        * this.physical.physicalCellSize;
      this.uploadParams(pressureScale, contactPressure, gravity);
    }
    let group = this.groups.get(pressure);
    if (!group) {
      group = this.device.createBindGroup({ label: "Losasso axis-face projection bindings", layout: this.layout,
        entries: [this.params, this.source.control, this.source.faces, pressure,
          this.source.predictedVelocity, this.source.projectedVelocity, this.source.faceMetrics]
          .map((buffer, binding) => ({ binding, resource: { buffer } })) });
      this.groups.set(pressure, group);
    }
    const pass = broker.compute({ label: "Losasso pressure - axis-face projection" });
    pass.setPipeline(this.pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroupsIndirect(this.source.faceDispatch, this.source.faceDispatchOffsetBytes ?? 0);
  }
  private uploadParams(pressureScale: number, contactPressure: number,
    gravity: readonly [number, number, number]): void {
    const bytes = new ArrayBuffer(32);
    const floats = new Float32Array(bytes);
    floats[0] = pressureScale; floats[1] = contactPressure;
    floats.set(gravity, 4);
    floats[7] = this.physical.density * this.physical.physicalCellSize;
    this.device.queue.writeBuffer(this.params, 0, bytes);
  }
  destroy(): void { if (this.destroyed) return; this.destroyed = true; this.params.destroy(); this.pipeline = undefined; }
  private assertLive(): void { if (this.destroyed) throw new Error("Losasso projection is destroyed"); }
}
