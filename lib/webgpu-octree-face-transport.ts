import { OCTREE_CONSUMER_MAX_FACE_CANDIDATES } from "./octree-consumer-sampling";
import type { OctreeFaceMirrorSource } from "./webgpu-octree-face-mirror";

export const OCTREE_FACE_TRANSPORT_VELOCITY_BYTES = 4;
export const OCTREE_FACE_TRANSPORT_CFL_BYTES = 16;
export const OCTREE_FACE_TRANSPORT_PARAMETER_BYTES = 48;
export const OCTREE_FACE_TRANSFER_RECORD_BYTES = 24;
export const OCTREE_FACE_TRANSFER_MAX_SOURCES = 4;
export const OCTREE_FACE_TRANSFER_INVALID = 0xffffffff;
export const OCTREE_FACE_KEY_WORDS = 4;
export const OCTREE_FACE_KEY_BYTES = OCTREE_FACE_KEY_WORDS * 4;

export type OctreeFaceOrigin = readonly [number, number, number];

export interface OctreeFaceTransferDescriptor {
  /** Full-width canonical origin. New publications must use this field. */
  readonly origin?: OctreeFaceOrigin;
  /** Legacy 10:10:10 origin accepted only for migration compatibility. */
  readonly packedOrigin?: number;
  readonly axisSpan: number;
  readonly normalVelocity: number;
}

export interface OctreeFaceTransferRecord {
  readonly newFace: number;
  /** 0 = initialize, 1 = exact/prolongation, 4 = equal-area restriction. */
  readonly sourceCount: 0 | 1 | 4;
  readonly oldFaces: readonly [number, number, number, number];
}

export interface OctreeFaceTopologyTransfer {
  readonly records: readonly OctreeFaceTransferRecord[];
  readonly velocities: Float32Array;
  readonly packedRecords: Uint32Array;
}

function validU32(value: number): boolean {
  return Number.isSafeInteger(value) && value >= 0 && value <= 0xffffffff;
}

function unpackLegacyTransferOrigin(word: number): [number, number, number] {
  return [word & 1023, (word >>> 10) & 1023, (word >>> 20) & 1023];
}

export function octreeFaceDescriptorOrigin(face: OctreeFaceTransferDescriptor): [number, number, number] {
  const explicit = face.origin;
  if (explicit !== undefined) {
    if (explicit.length !== 3 || explicit.some((value) => !validU32(value))) {
      throw new RangeError("Face origin coordinates must be unsigned 32-bit integers");
    }
    if (face.packedOrigin !== undefined) {
      const legacy = unpackLegacyTransferOrigin(face.packedOrigin >>> 0);
      if (legacy.some((value, axis) => value !== explicit[axis])) {
        throw new RangeError("Explicit and legacy face origins disagree");
      }
    }
    return [explicit[0], explicit[1], explicit[2]];
  }
  if (!validU32(face.packedOrigin ?? -1)) throw new RangeError("Face descriptor requires a valid origin");
  return unpackLegacyTransferOrigin(face.packedOrigin! >>> 0);
}

/** Exact, endian-independent four-word canonical face key ABI. */
export function encodeOctreeFaceKey(origin: OctreeFaceOrigin, axisSpan: number): Uint32Array {
  if (origin.length !== 3 || origin.some((value) => !validU32(value)) || !validU32(axisSpan)) {
    throw new RangeError("Face keys require unsigned 32-bit origin coordinates and axis/span");
  }
  return new Uint32Array([origin[0], origin[1], origin[2], axisSpan]);
}

export function decodeOctreeFaceKey(words: ArrayLike<number>): { origin: [number, number, number]; axisSpan: number } {
  if (words.length < OCTREE_FACE_KEY_WORDS) throw new RangeError("Face key requires four words");
  const values = [words[0], words[1], words[2], words[3]];
  if (values.some((value) => !validU32(value))) throw new RangeError("Face key words must be unsigned 32-bit integers");
  return { origin: [values[0], values[1], values[2]], axisSpan: values[3] };
}

function transferFaceKey(origin: OctreeFaceOrigin, axisSpan: number): string {
  return `${origin[0]}:${origin[1]}:${origin[2]}:${axisSpan >>> 0}`;
}

/**
 * Conservative CPU oracle for the topology builder's compact transfer map.
 * Under 2:1 balance every new face is either exact, one coarse-face injection,
 * or the equal-area restriction of four fine faces. There is no global hash in
 * the GPU ABI: the builder emits these transient records while it owns both
 * old and new sorted/local topology.
 */
export function buildOctreeFaceTopologyTransfer(
  previous: readonly OctreeFaceTransferDescriptor[],
  next: readonly OctreeFaceTransferDescriptor[],
): OctreeFaceTopologyTransfer {
  const oldByKey = new Map<string, number>();
  previous.forEach((face, index) => {
    const origin = octreeFaceDescriptorOrigin(face);
    const key = transferFaceKey(origin, face.axisSpan);
    if (oldByKey.has(key)) throw new RangeError("Face transfer source contains a duplicate canonical key");
    oldByKey.set(key, index);
  });
  const records: OctreeFaceTransferRecord[] = [];
  const velocities = new Float32Array(next.length);
  const invalidSources: [number, number, number, number] = [
    OCTREE_FACE_TRANSFER_INVALID, OCTREE_FACE_TRANSFER_INVALID,
    OCTREE_FACE_TRANSFER_INVALID, OCTREE_FACE_TRANSFER_INVALID,
  ];
  for (let newFace = 0; newFace < next.length; newFace += 1) {
    const face = next[newFace];
    const origin = octreeFaceDescriptorOrigin(face);
    const axis = face.axisSpan & 3;
    const span = face.axisSpan >>> 2;
    if (axis > 2 || span < 1 || (span & (span - 1)) !== 0) throw new RangeError("Face transfer requires a dyadic span and valid axis");
    const exact = oldByKey.get(transferFaceKey(origin, face.axisSpan));
    if (exact !== undefined) {
      const oldFaces: [number, number, number, number] = [exact, ...invalidSources.slice(1)] as [number, number, number, number];
      records.push({ newFace, sourceCount: 1, oldFaces }); velocities[newFace] = previous[exact].normalVelocity; continue;
    }
    const tangentialA = (axis + 1) % 3; const tangentialB = (axis + 2) % 3;
    if (span > 1) {
      const half = span >>> 1; const fine: number[] = [];
      for (let quadrant = 0; quadrant < 4; quadrant += 1) {
        const candidate: [number, number, number] = [...origin];
        candidate[tangentialA] += (quadrant & 1) === 0 ? 0 : half;
        candidate[tangentialB] += (quadrant & 2) === 0 ? 0 : half;
        const index = oldByKey.get(transferFaceKey(candidate, axis | (half << 2)));
        if (index !== undefined) fine.push(index);
      }
      if (fine.length === 4) {
        const oldFaces = fine as [number, number, number, number];
        records.push({ newFace, sourceCount: 4, oldFaces });
        velocities[newFace] = oldFaces.reduce((sum, index) => sum + previous[index].normalVelocity, 0) * 0.25;
        continue;
      }
    }
    const coarseSpan = span * 2; const coarseOrigin: [number, number, number] = [...origin];
    coarseOrigin[tangentialA] = Math.floor(coarseOrigin[tangentialA] / coarseSpan) * coarseSpan;
    coarseOrigin[tangentialB] = Math.floor(coarseOrigin[tangentialB] / coarseSpan) * coarseSpan;
    const coarse = oldByKey.get(transferFaceKey(coarseOrigin, axis | (coarseSpan << 2)));
    if (coarse !== undefined) {
      const oldFaces: [number, number, number, number] = [coarse, ...invalidSources.slice(1)] as [number, number, number, number];
      records.push({ newFace, sourceCount: 1, oldFaces }); velocities[newFace] = previous[coarse].normalVelocity; continue;
    }
    records.push({ newFace, sourceCount: 0, oldFaces: [...invalidSources] });
    velocities[newFace] = 0;
  }
  const packedRecords = new Uint32Array(records.length * (OCTREE_FACE_TRANSFER_RECORD_BYTES / 4));
  records.forEach((record, index) => packedRecords.set([record.newFace, record.sourceCount, ...record.oldFaces], index * 6));
  return { records, velocities, packedRecords };
}

export interface OctreeFaceTransportPlan {
  faceCapacity: number;
  velocityBytes: number;
  allocatedBytes: number;
}

export function planOctreeFaceTransport(faceCapacity: number): OctreeFaceTransportPlan {
  if (!Number.isInteger(faceCapacity) || faceCapacity < 1) {
    throw new RangeError("Octree face transport capacity must be positive");
  }
  const velocityBytes = faceCapacity * OCTREE_FACE_TRANSPORT_VELOCITY_BYTES;
  return {
    faceCapacity,
    velocityBytes,
    allocatedBytes: 2 * velocityBytes + OCTREE_FACE_TRANSPORT_CFL_BYTES + OCTREE_FACE_TRANSPORT_PARAMETER_BYTES,
  };
}

export interface OctreeFaceTransportOptions {
  dt: number;
  acceleration?: readonly [number, number, number];
  /**
   * Reindex velocities from transferred FaceRecord.normalVelocity after a
   * topology rebuild. This is a compact-face to compact-buffer copy; it never
   * samples the dense compatibility velocity. Leave false between rebuilds.
   */
  reseedFromMirror?: boolean;
  publishToMirror?: boolean;
}

export interface OctreeFaceVelocitySource {
  readonly buffer: GPUBuffer;
  readonly cfl: GPUBuffer;
  readonly faceCapacity: number;
  readonly allocatedBytes: number;
}

/** Four-word telemetry produced alongside compact face transport. */
export interface OctreeFaceVelocityDiagnostics {
  readonly maxComponentCfl: number;
  readonly maxSpeed_m_s: number;
  readonly nonFiniteCount: number;
  readonly transportedFaceCount: number;
}

export function decodeOctreeFaceVelocityDiagnostics(bytes: ArrayBufferLike): OctreeFaceVelocityDiagnostics {
  if (bytes.byteLength < OCTREE_FACE_TRANSPORT_CFL_BYTES) {
    throw new RangeError(`Octree face diagnostics require ${OCTREE_FACE_TRANSPORT_CFL_BYTES} bytes`);
  }
  const words = new Uint32Array(bytes, 0, OCTREE_FACE_TRANSPORT_CFL_BYTES / 4);
  const floats = new Float32Array(bytes, 0, OCTREE_FACE_TRANSPORT_CFL_BYTES / 4);
  return {
    maxComponentCfl: Number.isFinite(floats[0]) ? Math.max(0, floats[0]) : Number.POSITIVE_INFINITY,
    maxSpeed_m_s: Number.isFinite(floats[1]) ? Math.max(0, floats[1]) : Number.POSITIVE_INFINITY,
    nonFiniteCount: words[2],
    transportedFaceCount: words[3],
  };
}

/**
 * U3 adaptive MAC transport kernel.
 *
 * Sampling is bounded by the 2:1 topology invariant: a face gathers the two
 * adjacent leaves' signed incidence slabs (at most 2 * 24 candidates), then
 * performs resolution-aware interpolation over those canonical faces. A CFL
 * bounded backtrace cannot escape this one-ring stencil. Velocities live in
 * two compact f32 face buffers; no box-sized texture is allocated here.
 *
 * Integration order:
 *   publish/rebuild faces -> encode transport -> assemble face RHS -> project
 *   -> apply pressure gradient to the active compact face buffer.
 */
export class WebGPUOctreeFaceTransport {
  readonly plan: OctreeFaceTransportPlan;
  readonly cfl: GPUBuffer;
  private readonly device: GPUDevice;
  private readonly source: OctreeFaceMirrorSource;
  private readonly params: GPUBuffer;
  private readonly velocities: readonly [GPUBuffer, GPUBuffer];
  private readonly seedPipeline: GPUComputePipeline;
  private readonly transportPipeline: GPUComputePipeline;
  private readonly bindGroups: readonly [GPUBindGroup, GPUBindGroup];
  private diagnosticsReadback?: GPUBuffer;
  private diagnosticsPending?: Promise<OctreeFaceVelocityDiagnostics>;
  private active = 0;
  private initialized = false;

  constructor(
    device: GPUDevice,
    source: OctreeFaceMirrorSource,
    cellSize: readonly [number, number, number],
    dimensions: readonly [number, number, number] = [1023, 1023, 1023],
  ) {
    if (cellSize.some((value) => !Number.isFinite(value) || value <= 0)) {
      throw new RangeError("Octree face transport cell sizes must be finite and positive");
    }
    if (dimensions.some((value) => !Number.isSafeInteger(value) || value < 1 || value > 0xffffffff)) {
      throw new RangeError("Octree face transport dimensions must be positive unsigned 32-bit integers");
    }
    this.device = device;
    this.source = source;
    this.plan = planOctreeFaceTransport(source.plan.faceCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.velocities = [
      device.createBuffer({ label: "Octree face velocity A", size: this.plan.velocityBytes, usage: storage }),
      device.createBuffer({ label: "Octree face velocity B", size: this.plan.velocityBytes, usage: storage }),
    ];
    this.cfl = device.createBuffer({ label: "Octree face transport CFL", size: OCTREE_FACE_TRANSPORT_CFL_BYTES, usage: storage });
    this.params = device.createBuffer({
      label: "Octree face transport parameters",
      size: OCTREE_FACE_TRANSPORT_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.params, 16, new Float32Array([cellSize[0], cellSize[1], cellSize[2], 0]));
    device.queue.writeBuffer(this.params, 32, new Uint32Array([dimensions[0], dimensions[1], dimensions[2], 0]));

    const layout = device.createBindGroupLayout({ label: "Octree face transport layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    const shaderModule = device.createShaderModule({ label: "Octree adaptive face transport", code: octreeFaceTransportShader });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    this.seedPipeline = device.createComputePipeline({
      label: "Seed compact octree face velocities",
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "seedFaceVelocities" },
    });
    this.transportPipeline = device.createComputePipeline({
      label: "Advect compact octree face velocities",
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "transportFaceVelocities" },
    });
    const createBindings = (input: GPUBuffer, output: GPUBuffer): GPUBindGroup => device.createBindGroup({
      label: "Octree face transport bindings",
      layout,
      entries: [
        { binding: 0, resource: { buffer: source.control } },
        { binding: 1, resource: { buffer: source.faces } },
        { binding: 2, resource: { buffer: source.incidence } },
        { binding: 3, resource: { buffer: this.params } },
        { binding: 4, resource: { buffer: input } },
        { binding: 5, resource: { buffer: output } },
        { binding: 6, resource: { buffer: this.cfl } },
      ],
    });
    this.bindGroups = [createBindings(this.velocities[0], this.velocities[1]), createBindings(this.velocities[1], this.velocities[0])];
  }

  encode(encoder: GPUCommandEncoder, options: OctreeFaceTransportOptions): void {
    if (!Number.isFinite(options.dt) || options.dt < 0) throw new RangeError("Octree face transport dt must be finite and non-negative");
    if (options.publishToMirror === false) throw new Error("Detached octree face transport publication is not implemented");
    const acceleration = options.acceleration ?? [0, -9.81, 0];
    if (acceleration.some((value) => !Number.isFinite(value))) throw new RangeError("Octree face acceleration must be finite");
    this.device.queue.writeBuffer(this.params, 0, new Float32Array([options.dt, acceleration[0], acceleration[1], acceleration[2]]));
    const groups = Math.ceil(this.plan.faceCapacity / 256);
    if (!this.initialized || options.reseedFromMirror) {
      const seed = encoder.beginComputePass({ label: "Seed compact octree face velocities" });
      seed.setPipeline(this.seedPipeline);
      seed.setBindGroup(0, this.bindGroups[0]);
      seed.dispatchWorkgroups(groups);
      seed.end();
      this.active = 1;
      this.initialized = true;
    }
    encoder.clearBuffer(this.cfl);
    const transport = encoder.beginComputePass({ label: "Advect compact octree face velocities" });
    transport.setPipeline(this.transportPipeline);
    transport.setBindGroup(0, this.bindGroups[this.active]);
    transport.dispatchWorkgroups(groups);
    transport.end();
    this.active = 1 - this.active;
    // The shader publishes into the face records so the existing face
    // divergence/RHS consumer sees the transported value.
  }

  get velocitySource(): OctreeFaceVelocitySource {
    return {
      buffer: this.velocities[this.active],
      cfl: this.cfl,
      faceCapacity: this.plan.faceCapacity,
      allocatedBytes: this.plan.allocatedBytes,
    };
  }

  /**
   * Read the final compact transport reduction without touching a dense
   * velocity texture. Concurrent HUD/smoke polls share one pooled map.
   */
  readDiagnostics(): Promise<OctreeFaceVelocityDiagnostics> {
    if (this.diagnosticsPending) return this.diagnosticsPending;
    const readback = this.diagnosticsReadback ??= this.device.createBuffer({
      label: "Octree pooled compact-face diagnostics readback",
      size: OCTREE_FACE_TRANSPORT_CFL_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: "Read compact octree face diagnostics" });
    encoder.copyBufferToBuffer(this.cfl, 0, readback, 0, OCTREE_FACE_TRANSPORT_CFL_BYTES);
    this.device.queue.submit([encoder.finish()]);
    const pending = (async () => {
      try {
        await readback.mapAsync(GPUMapMode.READ);
        const snapshot = readback.getMappedRange(0, OCTREE_FACE_TRANSPORT_CFL_BYTES).slice(0);
        return decodeOctreeFaceVelocityDiagnostics(snapshot);
      } finally {
        if (readback.mapState === "mapped") readback.unmap();
      }
    })();
    this.diagnosticsPending = pending;
    void pending.finally(() => {
      if (this.diagnosticsPending === pending) this.diagnosticsPending = undefined;
    }).catch(() => { /* The caller receives device-loss/readback failures. */ });
    return pending;
  }

  invalidateTopology(): void { this.initialized = false; }

  destroy(): void {
    this.velocities[0].destroy();
    this.velocities[1].destroy();
    this.cfl.destroy();
    this.params.destroy();
    this.diagnosticsReadback?.destroy();
  }
}

export const octreeFaceTransportShader = /* wgsl */ `
struct FaceRecord { negativeRow: u32, positiveRow: u32, originX: u32, originY: u32, originZ: u32, axisSpan: u32, normalVelocity: f32, area: f32 }
struct TransportParams { dtAcceleration: vec4f, cellSize: vec4f, dimensions: vec4u }
@group(0) @binding(0) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> faces: array<FaceRecord>;
@group(0) @binding(2) var<storage, read_write> incidence: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: TransportParams;
@group(0) @binding(4) var<storage, read> velocityIn: array<f32>;
@group(0) @binding(5) var<storage, read_write> velocityOut: array<f32>;
// maximum component CFL, maximum speed, non-finite count, transported count
@group(0) @binding(6) var<storage, read_write> cfl: array<atomic<u32>>;
const INVALID = 0xffffffffu;
const INCIDENCE_PER_ROW = ${OCTREE_CONSUMER_MAX_FACE_CANDIDATES}u;

fn faceOrigin(face: FaceRecord) -> vec3u { return vec3u(face.originX, face.originY, face.originZ); }
fn faceAxis(face: FaceRecord) -> u32 { return face.axisSpan & 3u; }
fn faceSpan(face: FaceRecord) -> u32 { return face.axisSpan >> 2u; }
fn component(value: vec3f, axis: u32) -> f32 { return select(select(value.z, value.y, axis == 1u), value.x, axis == 0u); }
fn faceCentre(face: FaceRecord) -> vec3f {
  let axis = faceAxis(face); let span = f32(faceSpan(face)); var p = vec3f(faceOrigin(face));
  p[(axis + 1u) % 3u] += 0.5 * span; p[(axis + 2u) % 3u] += 0.5 * span;
  return p * params.cellSize.xyz;
}
fn domainBoundary(face: FaceRecord) -> bool {
  let axis=faceAxis(face);let coordinate=faceOrigin(face)[axis];
  return (face.negativeRow==INVALID||face.positiveRow==INVALID)
    && (coordinate==0u||coordinate==params.dimensions[axis]);
}
fn validRow(row: u32) -> bool { return row != INVALID && row < atomicLoad(&control[3]); }

// Resolution-aware one-ring interpolation. With 2:1 balance and CFL <= 1,
// both departure and arrival lie inside the adjacent leaves represented by
// these two bounded incidence slabs. Face span controls the support radius.
fn sampleComponent(point: vec3f, axis: u32, rowA: u32, rowB: u32, fallback: f32) -> f32 {
  var weighted = 0.0; var weights = 0.0; var nearest = fallback; var nearestD2 = 3.402823e38; var visited=0u;
  for (var side = 0u; side < 2u; side += 1u) {
    if(visited>=INCIDENCE_PER_ROW){break;}let row = select(rowA, rowB, side == 1u); if (!validRow(row)) { continue; }
    let count = min(atomicLoad(&incidence[row]), INCIDENCE_PER_ROW);
    for (var local = 0u; local < count; local += 1u) {
      if(visited>=INCIDENCE_PER_ROW){break;}visited+=1u;
      let index = atomicLoad(&incidence[atomicLoad(&control[3]) + row * INCIDENCE_PER_ROW + local]);
      if (index >= atomicLoad(&control[0]) || index >= atomicLoad(&control[2])) { continue; }
      let candidate = faces[index]; if (faceAxis(candidate) != axis) { continue; }
      let delta = (point - faceCentre(candidate)) / params.cellSize.xyz; let d2 = dot(delta, delta);
      if (d2 < nearestD2) { nearestD2 = d2; nearest = velocityIn[index]; }
      let support = max(1.0, f32(faceSpan(candidate))); let weight = 1.0 / max(0.0625 * support * support, d2);
      weighted += weight * velocityIn[index]; weights += weight;
    }
  }
  if (weights > 0.0) { return weighted / weights; } return nearest;
}
fn sampleVelocity(point: vec3f, face: FaceRecord, fallback: f32) -> vec3f {
  return vec3f(
    sampleComponent(point, 0u, face.negativeRow, face.positiveRow, select(0.0, fallback, faceAxis(face) == 0u)),
    sampleComponent(point, 1u, face.negativeRow, face.positiveRow, select(0.0, fallback, faceAxis(face) == 1u)),
    sampleComponent(point, 2u, face.negativeRow, face.positiveRow, select(0.0, fallback, faceAxis(face) == 2u))
  );
}
fn finite(value: f32) -> bool { return value <= 3.402823e38 && value >= -3.402823e38; }

@compute @workgroup_size(256)
fn seedFaceVelocities(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x; if (index >= atomicLoad(&control[0]) || index >= atomicLoad(&control[2])) { return; }
  velocityOut[index] = faces[index].normalVelocity;
}

@compute @workgroup_size(256)
fn transportFaceVelocities(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x; if (index >= atomicLoad(&control[0]) || index >= atomicLoad(&control[2]) || atomicLoad(&control[1]) != 0u) { return; }
  let face = faces[index]; let centre = faceCentre(face); let advecting = sampleVelocity(centre, face, velocityIn[index]);
  if(domainBoundary(face)){velocityOut[index]=0.0;faces[index].normalVelocity=0.0;atomicAdd(&cfl[3],1u);return;}
  let departure = centre - params.dtAcceleration.x * advecting;
  let transported = sampleComponent(departure, faceAxis(face), face.negativeRow, face.positiveRow, velocityIn[index]);
  let value = transported + params.dtAcceleration.x * component(params.dtAcceleration.yzw, faceAxis(face));
  if (!finite(value)) { atomicAdd(&cfl[2], 1u); return; }
  velocityOut[index] = value; faces[index].normalVelocity = value;
  let h = component(params.cellSize.xyz, faceAxis(face)); let componentCfl = abs(value) * params.dtAcceleration.x / h;
  atomicMax(&cfl[0], bitcast<u32>(componentCfl)); atomicMax(&cfl[1], bitcast<u32>(length(advecting))); atomicAdd(&cfl[3], 1u);
}
`;
