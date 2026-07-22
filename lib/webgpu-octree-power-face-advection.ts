/**
 * Aanjaneya et al. (2017), Section 5 velocity advection across an octree
 * topology change.
 *
 * The projected old generalized-face field is first reconstructed to full
 * vectors at old power-cell centres by `WebGPUOctreePowerVelocity`.  This
 * class snapshots that complete old interpolation mesh.  After the next
 * topology has been built, every new generalized-face centroid is traced
 * backward through the old full-vector interpolant and the sampled vector is
 * projected on the new face normal.  Regular cube regions additionally retain
 * the old staggered Cartesian faces so the paper's per-axis interpolant is an
 * identity at zero timestep; transition regions use the full-vector mesh.
 */

import type { OctreePowerFaceSource } from "./webgpu-octree-power-faces";
import type { OctreePowerTopologySource } from "./webgpu-octree-power-topology";
import type { OctreeFaceMirrorSource } from "./webgpu-octree-face-mirror";

export const OCTREE_POWER_OLD_MESH_ADVECTION_CONTROL_BYTES = 64;
/** WGSL `P` is 80 bytes because `vec3u dims` begins at a 16-byte boundary. */
export const OCTREE_POWER_OLD_MESH_ADVECTION_PARAMETER_BYTES = 80;
export const OCTREE_POWER_OLD_MESH_ADVECTION_VALID = 0x8000_0000;
export const OCTREE_POWER_OLD_MESH_ADVECTION_ERROR = Object.freeze({
  source: 1 << 0,
  generation: 1 << 1,
  capacity: 1 << 2,
  interpolation: 1 << 3,
  nonfinite: 1 << 4,
} as const);
export const OCTREE_POWER_OLD_MESH_PREPARE_BINDINGS = Object.freeze([0, 1, 4, 5, 6, 7, 8] as const);
export const OCTREE_POWER_OLD_MESH_ADVECT_BINDINGS = Object.freeze([0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12] as const);
export const OCTREE_POWER_OLD_MESH_FINALIZE_BINDINGS = Object.freeze([1, 8] as const);

export const OCTREE_POWER_OLD_MESH_AXIS_CONTROL_BYTES = 32;

function nextPowerOfTwo(value: number): number {
  let result = 1;
  while (result < value) result *= 2;
  return result;
}

export interface OctreePowerOldMeshAdvectionPlan {
  readonly rowCapacity: number;
  readonly faceCapacity: number;
  readonly siteHashCapacity: number;
  readonly headerBytes: number;
  readonly metricOffsetBytes: number;
  readonly velocityOffsetBytes: number;
  readonly axisControlOffsetBytes: number;
  readonly axisFaceOffsetBytes: number;
  readonly axisHashOffsetBytes: number;
  readonly axisFaceCapacity: number;
  readonly axisHashCapacity: number;
  readonly arenaBytes: number;
  readonly siteBytes: number;
  readonly allocatedBytes: number;
}

export function planOctreePowerOldMeshAdvection(
  rowCapacity: number,
  faceCapacity: number,
  siteHashCapacity: number,
  axisFaceCapacity = faceCapacity,
): OctreePowerOldMeshAdvectionPlan {
  for (const [value, label] of [[rowCapacity, "row"], [faceCapacity, "face"],
    [siteHashCapacity, "site hash"], [axisFaceCapacity, "axis face"]] as const) {
    if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`Old-mesh ${label} capacity must be positive`);
  }
  const headerBytes = rowCapacity * 48;
  const metricOffsetBytes = headerBytes;
  const velocityOffsetBytes = metricOffsetBytes + rowCapacity * 16;
  const axisControlOffsetBytes = velocityOffsetBytes + rowCapacity * 16;
  const axisFaceOffsetBytes = axisControlOffsetBytes + OCTREE_POWER_OLD_MESH_AXIS_CONTROL_BYTES;
  const axisHashCapacity = nextPowerOfTwo(axisFaceCapacity * 2);
  const axisHashOffsetBytes = axisFaceOffsetBytes + axisFaceCapacity * 32;
  const arenaBytes = axisHashOffsetBytes + axisHashCapacity * 4;
  const siteBytes = siteHashCapacity * 16;
  return { rowCapacity, faceCapacity, siteHashCapacity, headerBytes, metricOffsetBytes,
    velocityOffsetBytes, axisControlOffsetBytes, axisFaceOffsetBytes, axisHashOffsetBytes,
    axisFaceCapacity, axisHashCapacity, arenaBytes, siteBytes,
    allocatedBytes: arenaBytes + siteBytes + OCTREE_POWER_OLD_MESH_ADVECTION_CONTROL_BYTES
      + OCTREE_POWER_OLD_MESH_ADVECTION_PARAMETER_BYTES };
}

export function packOctreePowerOldMeshAdvectionParameters(input: {
  rowCapacity: number; faceCapacity: number; siteHashCapacity: number;
  metricOffsetWords: number; velocityOffsetWords: number;
  dimensions: readonly [number, number, number]; maximumLeafSize: number; generation: number;
  physicalCellSize: number; timestep: number; deferInterpolationFailures?: boolean;
  axisControlOffsetWords?: number; axisFaceOffsetWords?: number; axisHashOffsetWords?: number;
}): ArrayBuffer {
  const data = new ArrayBuffer(OCTREE_POWER_OLD_MESH_ADVECTION_PARAMETER_BYTES);
  const words = new Uint32Array(data), floats = new Float32Array(data);
  words.set([input.rowCapacity, input.faceCapacity, input.siteHashCapacity,
    input.metricOffsetWords, input.velocityOffsetWords]);
  // WGSL uniform layout: vec3u has 16-byte alignment, so words 5--7 are
  // explicit padding and `dims` starts at word 8.
  words.set(input.dimensions, 8);
  words[11] = input.maximumLeafSize; words[12] = input.generation >>> 0;
  floats[13] = input.physicalCellSize; floats[14] = input.timestep;
  words[15] = input.deferInterpolationFailures ? 1 : 0;
  words[16] = input.axisControlOffsetWords ?? 0;
  words[17] = input.axisFaceOffsetWords ?? 0;
  words[18] = input.axisHashOffsetWords ?? 0;
  return data;
}

/** CPU oracle for the characteristic and final normal projection. */
export function advectPowerFaceFromOldVector(
  centroid: readonly [number, number, number],
  normal: readonly [number, number, number],
  dt: number,
  sampleOld: (point: readonly [number, number, number]) => readonly [number, number, number] | undefined,
): number | undefined {
  if (![...centroid, ...normal, dt].every(Number.isFinite) || dt < 0) return undefined;
  const first = sampleOld(centroid); if (!first?.every(Number.isFinite)) return undefined;
  const midpoint = centroid.map((value, axis) => value - 0.5 * dt * first[axis]) as [number, number, number];
  const middle = sampleOld(midpoint); if (!middle?.every(Number.isFinite)) return undefined;
  const departure = centroid.map((value, axis) => value - dt * middle[axis]) as [number, number, number];
  const advected = sampleOld(departure); if (!advected?.every(Number.isFinite)) return undefined;
  const projected = advected[0] * normal[0] + advected[1] * normal[1] + advected[2] * normal[2];
  return Number.isFinite(projected) ? projected : undefined;
}

/**
 * CPU oracle for the regular-region staggered component interpolation.
 *
 * The paper uses trilinear interpolation and calls out its first-order
 * semi-Lagrangian dissipation as a limitation.  We retain that bounded
 * trilinear value as a fallback and limiter, but use a Catmull-Rom tensor
 * reconstruction when the complete four-sample support is resident.  This
 * reproduces affine/quadratic profiles without the per-step low-pass filter
 * while preventing new component extrema.
 */
export function interpolateRegularAxisFaceComponent(
  pointGrid: readonly [number, number, number],
  ownerOrigin: readonly [number, number, number],
  ownerSize: number,
  axis: 0 | 1 | 2,
  sample: (origin: readonly [number, number, number], axis: 0 | 1 | 2, size: number) => number | undefined,
): number | undefined {
  if (![...pointGrid, ...ownerOrigin, ownerSize].every(Number.isFinite)
    || !Number.isSafeInteger(ownerSize) || ownerSize < 1
    || !ownerOrigin.every(value => Number.isSafeInteger(value) && value >= 0)) return undefined;
  const low = [0, 0, 0], fraction = [0, 0, 0];
  for (let dimension = 0; dimension < 3; dimension += 1) {
    if (dimension === axis) {
      low[dimension] = ownerOrigin[dimension];
      fraction[dimension] = (pointGrid[dimension] - low[dimension]) / ownerSize;
    } else {
      const center = ownerOrigin[dimension] + 0.5 * ownerSize;
      low[dimension] = pointGrid[dimension] < center
        ? ownerOrigin[dimension] - ownerSize : ownerOrigin[dimension];
      fraction[dimension] = (pointGrid[dimension] - (low[dimension] + 0.5 * ownerSize)) / ownerSize;
    }
    if (fraction[dimension] < -2e-5 || fraction[dimension] > 1.00002) return undefined;
    fraction[dimension] = Math.max(0, Math.min(1, fraction[dimension]));
  }
  if (fraction.every(value => value <= 1e-7)) {
    const direct = sample(low as [number, number, number], axis, ownerSize);
    return Number.isFinite(direct) ? direct : undefined;
  }
  let linear = 0, minimum = Infinity, maximum = -Infinity;
  for (let corner = 0; corner < 8; corner += 1) {
    const bits = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
    const weight = bits.reduce((product, bit, dimension) => product
      * (bit === 0 ? 1 - fraction[dimension] : fraction[dimension]), 1);
    if (weight === 0) continue;
    const origin = low.map((value, dimension) => value + bits[dimension] * ownerSize) as [number, number, number];
    if (origin.some(value => value < 0)) return undefined;
    const value = sample(origin, axis, ownerSize);
    if (!Number.isFinite(value)) return undefined;
    linear += weight * value!;
    minimum = Math.min(minimum, value!); maximum = Math.max(maximum, value!);
  }
  const cubicWeight = (offset: number, t: number) => {
    const t2 = t * t, t3 = t2 * t;
    if (offset === 0) return -0.5 * t + t2 - 0.5 * t3;
    if (offset === 1) return 1 - 2.5 * t2 + 1.5 * t3;
    if (offset === 2) return 0.5 * t + 2 * t2 - 1.5 * t3;
    return -0.5 * t2 + 0.5 * t3;
  };
  let cubic = 0;
  for (let stencil = 0; stencil < 64; stencil += 1) {
    const offsets = [stencil & 3, (stencil >> 2) & 3, (stencil >> 4) & 3];
    const origin = low.map((value, dimension) => value + (offsets[dimension] - 1) * ownerSize) as
      [number, number, number];
    if (origin.some(value => value < 0)) return linear;
    const value = sample(origin, axis, ownerSize);
    if (!Number.isFinite(value)) return linear;
    const weight = offsets.reduce((product, offset, dimension) => product
      * cubicWeight(offset, fraction[dimension]), 1);
    cubic += weight * value!;
  }
  return Number.isFinite(cubic) ? Math.max(minimum, Math.min(maximum, cubic)) : linear;
}

/**
 * Owner-independent form of the paper's regular staggered interpolant.
 *
 * Section 5 extrapolates velocity on regular octree faces outside the liquid
 * before velocity advection.  Consequently, a valid regular-grid sample must
 * not depend on locating a liquid pressure cell at the query point.  The
 * staggered lattice itself determines the interpolation cell.
 */
export function interpolateRegularAxisFaceComponentAtSize(
  pointGrid: readonly [number, number, number],
  size: number,
  axis: 0 | 1 | 2,
  sample: (origin: readonly [number, number, number], axis: 0 | 1 | 2, size: number) => number | undefined,
): number | undefined {
  if (![...pointGrid, size].every(Number.isFinite) || !Number.isSafeInteger(size) || size < 1) return undefined;
  const ownerOrigin = pointGrid.map((value) => Math.floor(value / size) * size) as [number, number, number];
  return interpolateRegularAxisFaceComponent(pointGrid, ownerOrigin, size, axis, sample);
}

export class WebGPUOctreePowerFaceAdvection {
  readonly plan: OctreePowerOldMeshAdvectionPlan;
  readonly control: GPUBuffer;
  private readonly oldArena: GPUBuffer;
  private readonly oldSites: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly captureParams: GPUBuffer;
  private readonly capturePipeline: GPUComputePipeline;
  private readonly captureAxisPipeline: GPUComputePipeline;
  private readonly preparePipeline: GPUComputePipeline;
  private readonly advectPipeline: GPUComputePipeline;
  private readonly finalizePipeline: GPUComputePipeline;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly topology: OctreePowerTopologySource,
    private readonly faces: OctreePowerFaceSource,
    private readonly axis: OctreeFaceMirrorSource,
  ) {
    if (!topology.catalogTetrahedronHeaders || !topology.catalogTetrahedra || !topology.catalogTetrahedronVertices) {
      throw new RangeError("Old-mesh velocity advection requires the Section 5 Delaunay catalog");
    }
    this.plan = planOctreePowerOldMeshAdvection(topology.plan.rowCapacity, faces.plan.faceCapacity,
      faces.plan.hashCapacity, axis.plan.faceCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.oldArena = device.createBuffer({ label: "Old power interpolation mesh", size: this.plan.arenaBytes,
      usage: storage });
    this.oldSites = device.createBuffer({ label: "Old power interpolation site hash", size: this.plan.siteBytes,
      usage: storage });
    this.control = device.createBuffer({ label: "Old-mesh power-face advection control",
      size: OCTREE_POWER_OLD_MESH_ADVECTION_CONTROL_BYTES, usage: storage });
    this.params = device.createBuffer({ label: "Old-mesh power-face advection parameters",
      size: OCTREE_POWER_OLD_MESH_ADVECTION_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.captureParams = device.createBuffer({ label: "Old-mesh staggered-face snapshot parameters",
      size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.captureParams, 0, new Uint32Array([
      this.plan.axisControlOffsetBytes / 4, this.plan.axisFaceOffsetBytes / 4,
      this.plan.axisHashOffsetBytes / 4, this.plan.axisHashCapacity,
    ]));
    const captureModule = device.createShaderModule({ label: "Capture 2017 old-mesh authority",
      code: octreePowerOldMeshCaptureWGSL });
    const module = device.createShaderModule({ label: "2017 old-mesh full-vector face advection",
      code: octreePowerOldMeshAdvectionWGSL });
    this.capturePipeline = device.createComputePipeline({ label: "Capture old power interpolation authority",
      layout: "auto", compute: { module: captureModule, entryPoint: "captureOldMeshAuthority" } });
    this.captureAxisPipeline = device.createComputePipeline({ label: "Index old staggered Cartesian faces",
      layout: "auto", compute: { module: captureModule, entryPoint: "indexOldAxisFaces" } });
    this.preparePipeline = device.createComputePipeline({ label: "Prepare old-mesh power-face advection",
      layout: "auto", compute: { module, entryPoint: "prepareNewPowerFaceAdvection" } });
    this.advectPipeline = device.createComputePipeline({ label: "Advect new power faces through old mesh",
      layout: "auto", compute: { module, entryPoint: "advectNewPowerFaces" } });
    this.finalizePipeline = device.createComputePipeline({ label: "Publish old-mesh power-face advection",
      layout: "auto", compute: { module, entryPoint: "publishNewPowerFaceAdvection" } });
  }

  /** Capture generation N only after its projected full cell vectors publish. */
  encodeCapture(encoder: GPUCommandEncoder, input: {
    leafHeaders: GPUBuffer;
    rowVelocities: GPUBuffer;
    velocityControl: GPUBuffer;
  }): void {
    this.assertLive();
    encoder.copyBufferToBuffer(input.leafHeaders, 0, this.oldArena, 0, this.plan.headerBytes);
    encoder.copyBufferToBuffer(this.topology.metrics, 0, this.oldArena, this.plan.metricOffsetBytes,
      this.plan.rowCapacity * 16);
    encoder.copyBufferToBuffer(input.rowVelocities, 0, this.oldArena, this.plan.velocityOffsetBytes,
      this.plan.rowCapacity * 16);
    encoder.copyBufferToBuffer(this.axis.control, 0, this.oldArena, this.plan.axisControlOffsetBytes, 24);
    encoder.copyBufferToBuffer(this.axis.faces, 0, this.oldArena, this.plan.axisFaceOffsetBytes,
      this.axis.plan.faceBytes);
    encoder.clearBuffer(this.oldArena, this.plan.axisHashOffsetBytes, this.plan.axisHashCapacity * 4);
    encoder.copyBufferToBuffer(this.faces.siteIndex, 0, this.oldSites, 0, this.plan.siteBytes);
    const pass = encoder.beginComputePass({ label: "Validate old power interpolation snapshot" });
    pass.setPipeline(this.capturePipeline);
    pass.setBindGroup(0, this.device.createBindGroup({ layout: this.capturePipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.faces.control } },
      { binding: 1, resource: { buffer: input.velocityControl } },
      { binding: 2, resource: { buffer: this.topology.control } },
      { binding: 3, resource: { buffer: this.control } },
      { binding: 4, resource: { buffer: this.axis.control } },
      { binding: 5, resource: { buffer: this.oldArena } },
      { binding: 6, resource: { buffer: this.captureParams } },
    ] }));
    pass.dispatchWorkgroups(1); pass.end();
    const axisPass = encoder.beginComputePass({ label: "Index old staggered Cartesian faces" });
    axisPass.setPipeline(this.captureAxisPipeline);
    axisPass.setBindGroup(0, this.device.createBindGroup({ layout: this.captureAxisPipeline.getBindGroupLayout(0), entries: [
      { binding: 3, resource: { buffer: this.control } },
      { binding: 5, resource: { buffer: this.oldArena } },
      { binding: 6, resource: { buffer: this.captureParams } },
    ] }));
    axisPass.dispatchWorkgroups(Math.ceil(this.axis.plan.faceCapacity / 64)); axisPass.end();
  }

  /** Overwrite generation N+1 face velocities from the captured generation N mesh. */
  encodeAdvect(encoder: GPUCommandEncoder, input: {
    seedControl: GPUBuffer;
    dimensions: readonly [number, number, number];
    physicalCellSize: number;
    maximumLeafSize: number;
    generation: number;
    timestep: number;
    deferInterpolationFailures?: boolean;
  }): void {
    this.assertLive();
    if (!input.dimensions.every((value) => Number.isSafeInteger(value) && value > 0)
      || !Number.isSafeInteger(input.maximumLeafSize) || input.maximumLeafSize < 1
      || !Number.isSafeInteger(input.generation) || input.generation < 1
      || !Number.isFinite(input.physicalCellSize) || input.physicalCellSize <= 0
      || !Number.isFinite(input.timestep) || input.timestep < 0) {
      throw new RangeError("Old-mesh face-advection parameters are invalid");
    }
    const data = packOctreePowerOldMeshAdvectionParameters({
      rowCapacity: this.plan.rowCapacity, faceCapacity: this.plan.faceCapacity,
      siteHashCapacity: this.plan.siteHashCapacity,
      metricOffsetWords: this.plan.metricOffsetBytes / 4,
      velocityOffsetWords: this.plan.velocityOffsetBytes / 4,
      dimensions: input.dimensions, maximumLeafSize: input.maximumLeafSize,
      generation: input.generation, physicalCellSize: input.physicalCellSize, timestep: input.timestep,
      deferInterpolationFailures: input.deferInterpolationFailures,
      axisControlOffsetWords: this.plan.axisControlOffsetBytes / 4,
      axisFaceOffsetWords: this.plan.axisFaceOffsetBytes / 4,
      axisHashOffsetWords: this.plan.axisHashOffsetBytes / 4,
    });
    this.device.queue.writeBuffer(this.params, 0, data);
    const t = this.topology;
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: { buffer: this.control } },
      { binding: 2, resource: { buffer: this.oldArena } },
      { binding: 3, resource: { buffer: this.oldSites } },
      { binding: 4, resource: { buffer: this.faces.control } },
      { binding: 5, resource: { buffer: this.faces.faces } },
      { binding: 6, resource: { buffer: this.faces.faceNormals } },
      { binding: 7, resource: { buffer: this.faces.faceCentroids } },
      { binding: 8, resource: { buffer: input.seedControl } },
      { binding: 10, resource: { buffer: t.catalogTetrahedronHeaders! } },
      { binding: 11, resource: { buffer: t.catalogTetrahedra! } },
      { binding: 12, resource: { buffer: t.catalogTetrahedronVertices! } },
    ];
    const group = (pipeline: GPUComputePipeline, bindings: readonly number[]) => this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries: entries.filter((entry) => bindings.includes(entry.binding)),
    });
    // Dispatch boundaries inside one compute pass do not establish a storage
    // dependency for independent workgroups on every backend. Fence the live
    // count publication before its consumers, then fence all face statuses
    // before the all-or-nothing authority gate.
    const prepare = encoder.beginComputePass({ label: "Prepare Section 5 old-mesh face advection" });
    prepare.setPipeline(this.preparePipeline);
    prepare.setBindGroup(0, group(this.preparePipeline, OCTREE_POWER_OLD_MESH_PREPARE_BINDINGS));
    prepare.dispatchWorkgroups(1);
    prepare.end();
    const advect = encoder.beginComputePass({ label: "Section 5 advect new power faces through old mesh" });
    advect.setPipeline(this.advectPipeline);
    advect.setBindGroup(0, group(this.advectPipeline, OCTREE_POWER_OLD_MESH_ADVECT_BINDINGS));
    advect.dispatchWorkgroups(Math.ceil(this.plan.faceCapacity / 64));
    advect.end();
    if (!input.deferInterpolationFailures) {
      const finalize = encoder.beginComputePass({ label: "Publish Section 5 old-mesh face advection" });
      finalize.setPipeline(this.finalizePipeline);
      finalize.setBindGroup(0, group(this.finalizePipeline, OCTREE_POWER_OLD_MESH_FINALIZE_BINDINGS));
      finalize.dispatchWorkgroups(1);
      finalize.end();
    }
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.oldArena.destroy(); this.oldSites.destroy(); this.control.destroy(); this.params.destroy();
    this.captureParams.destroy();
  }
  private assertLive(): void { if (this.destroyed) throw new Error("Old-mesh power-face advection is destroyed"); }
}

export const octreePowerOldMeshCaptureWGSL = /* wgsl */ `
struct AxisFace{negativeRow:u32,positiveRow:u32,originX:u32,originY:u32,originZ:u32,axisSpan:u32,normalVelocity:f32,area:f32}
struct CaptureP{axisControlBase:u32,axisFaceBase:u32,axisHashBase:u32,axisHashCapacity:u32}
@group(0)@binding(0)var<storage,read>faceControl:array<u32>;
@group(0)@binding(1)var<storage,read>velocityControl:array<u32>;
@group(0)@binding(2)var<storage,read>topologyControl:array<u32>;
@group(0)@binding(3)var<storage,read_write>snapshot:array<atomic<u32>>;
@group(0)@binding(4)var<storage,read>axisControl:array<u32>;
@group(0)@binding(5)var<storage,read_write>arena:array<atomic<u32>>;
@group(0)@binding(6)var<uniform>cp:CaptureP;
const VALID=0x80000000u;const SOURCE=1u;const CAPACITY=4u;
@compute @workgroup_size(1)fn captureOldMeshAuthority(){
 var flags=0u;var rows=0u;var generation=0u;
 if(arrayLength(&faceControl)<9u||arrayLength(&velocityControl)<8u||arrayLength(&topologyControl)<5u||arrayLength(&snapshot)<16u||arrayLength(&axisControl)<6u||cp.axisControlBase+8u>arrayLength(&arena)||cp.axisHashBase+cp.axisHashCapacity>arrayLength(&arena)){flags|=CAPACITY;}
 else{rows=faceControl[0];generation=faceControl[7];if(faceControl[8]!=VALID||faceControl[3]!=0u||velocityControl[0]!=VALID||velocityControl[2]!=rows||velocityControl[5]!=rows||velocityControl[7]!=generation||topologyControl[0]!=0u||topologyControl[2]!=0u||topologyControl[3]!=rows||axisControl[1]!=0u||axisControl[0]>axisControl[2]||cp.axisFaceBase+axisControl[2]*8u>arrayLength(&arena)){flags|=SOURCE;}}
 if(cp.axisControlBase+8u<=arrayLength(&arena)){atomicStore(&arena[cp.axisControlBase+6u],cp.axisHashCapacity);atomicStore(&arena[cp.axisControlBase+7u],select(0u,VALID,flags==0u));}
 atomicStore(&snapshot[0],flags);atomicStore(&snapshot[1],0xffffffffu);atomicStore(&snapshot[2],rows);atomicStore(&snapshot[3],0u);atomicStore(&snapshot[4],generation);atomicStore(&snapshot[5],generation);atomicStore(&snapshot[6],select(0u,VALID,flags==0u&&rows>0u));atomicStore(&snapshot[7],0u);
}
fn axisFace(index:u32)->AxisFace{let b=cp.axisFaceBase+index*8u;return AxisFace(atomicLoad(&arena[b]),atomicLoad(&arena[b+1u]),atomicLoad(&arena[b+2u]),atomicLoad(&arena[b+3u]),atomicLoad(&arena[b+4u]),atomicLoad(&arena[b+5u]),bitcast<f32>(atomicLoad(&arena[b+6u])),bitcast<f32>(atomicLoad(&arena[b+7u])));}
fn axisHash(face:AxisFace)->u32{var h=face.originX*73856093u;h^=face.originY*19349663u;h^=face.originZ*83492791u;h^=face.axisSpan*2654435761u;h^=h>>16u;return h;}
fn sameAxisFace(a:AxisFace,b:AxisFace)->bool{return a.originX==b.originX&&a.originY==b.originY&&a.originZ==b.originZ&&a.axisSpan==b.axisSpan;}
fn rejectAxis(index:u32){atomicOr(&snapshot[0],SOURCE);atomicMin(&snapshot[1],index);atomicStore(&snapshot[6],0u);atomicStore(&arena[cp.axisControlBase+7u],0u);}
@compute @workgroup_size(64)fn indexOldAxisFaces(@builtin(global_invocation_id)id:vec3u){let i=id.x;if(atomicLoad(&snapshot[6])!=VALID){return;}let count=atomicLoad(&arena[cp.axisControlBase]);if(i>=count){return;}let face=axisFace(i);let axis=face.axisSpan&3u;let span=face.axisSpan>>2u;if(axis>=3u||span==0u||face.axisSpan!=(axis|(span<<2u))||face.normalVelocity!=face.normalVelocity||abs(face.normalVelocity)>3.402823e38||face.area<=0.||face.area!=face.area){rejectAxis(i);return;}let mask=cp.axisHashCapacity-1u;let start=axisHash(face)&mask;for(var probe=0u;probe<min(64u,cp.axisHashCapacity);probe+=1u){let slot=cp.axisHashBase+((start+probe)&mask);let found=atomicCompareExchangeWeak(&arena[slot],0u,i+1u);if(found.exchanged){return;}let prior=found.old_value;if(prior>0u&&prior<=count&&sameAxisFace(axisFace(prior-1u),face)){rejectAxis(i);return;}}rejectAxis(i);}
`;

export const octreePowerOldMeshAdvectionWGSL = /* wgsl */ `
struct P{rowCapacity:u32,faceCapacity:u32,hashCapacity:u32,metricBase:u32,velocityBase:u32,dims:vec3u,maximumLeafSize:u32,generation:u32,cellSize:f32,dt:f32,p0:u32,p1:u32,p2:u32,p3:u32}
struct Face{negativeRow:u32,positiveRow:u32,geometryCode:u32,flags:u32,normalVelocity:f32,area:f32,inverseDistance:f32,openFraction:f32}
struct AxisFace{negativeRow:u32,positiveRow:u32,originX:u32,originY:u32,originZ:u32,axisSpan:u32,normalVelocity:f32,area:f32}
struct Site{cellPlusOne:atomic<u32>,size:u32,row:u32,pad:u32}struct TH{first:u32,count:u32,flags:u32}struct TV{v:vec4f}
struct C{flags:atomic<u32>,firstError:atomic<u32>,faceCount:atomic<u32>,advected:atomic<u32>,generation:u32,oldGeneration:u32,valid:atomic<u32>,coldFallback:atomic<u32>,p0:u32,p1:u32,p2:u32,p3:u32,p4:u32,firstInterpolationStage:atomic<u32>,firstInterpolationReason:atomic<u32>,p7:u32}
struct Seed{flags:atomic<u32>,firstError:atomic<u32>,rowCount:atomic<u32>,faceCount:atomic<u32>,seededCount:atomic<u32>,generation:atomic<u32>,valid:atomic<u32>}
@group(0)@binding(0)var<uniform>p:P;@group(0)@binding(1)var<storage,read_write>control:C;@group(0)@binding(2)var<storage,read>arena:array<u32>;@group(0)@binding(3)var<storage,read_write>sites:array<Site>;@group(0)@binding(4)var<storage,read>faceControl:array<u32>;@group(0)@binding(5)var<storage,read_write>faces:array<Face>;@group(0)@binding(6)var<storage,read>normals:array<vec4f>;@group(0)@binding(7)var<storage,read_write>centroids:array<vec4f>;@group(0)@binding(8)var<storage,read_write>seed:Seed;@group(0)@binding(10)var<storage,read>tetraHeaders:array<TH>;@group(0)@binding(11)var<storage,read>tetrahedra:array<u32>;@group(0)@binding(12)var<storage,read>vertices:array<TV>;
const INVALID=0xffffffffu;const VALID=0x80000000u;const STATUS_VALID=0x3f800000u;const BOUNDARY=1u;const OPEN=2u;const SOURCE=1u;const GENERATION=2u;const CAPACITY=4u;const INTERPOLATION=8u;const NONFINITE=16u;
fn finite(x:f32)->bool{return x==x&&abs(x)<=3.402823e38;}fn fail(code:u32,index:u32){atomicOr(&control.flags,code);atomicMin(&control.firstError,index);atomicOr(&seed.flags,code);atomicMin(&seed.firstError,index);atomicStore(&seed.valid,0u);}
fn hash(c:u32,s:u32)->u32{var v=c^(s*0x9e3779b9u);v=(v^(v>>16u))*0x7feb352du;v=(v^(v>>15u))*0x846ca68bu;return v^(v>>16u);}fn find(c:u32,s:u32)->u32{let cap=min(p.hashCapacity,arrayLength(&sites));let start=hash(c,s)&(cap-1u);for(var q=0u;q<min(32u,cap);q+=1u){let slot=(start+q)&(cap-1u);let key=atomicLoad(&sites[slot].cellPlusOne);if(key==0u){return INVALID;}if(key==c+1u&&sites[slot].size==s){return sites[slot].row;}}return INVALID;}
fn owner(x:vec3f)->u32{let g=x/p.cellSize;if(any(g<vec3f(0))||any(g>=vec3f(p.dims))){return INVALID;}let q=vec3u(floor(g));var s=1u;loop{let o=(q/s)*s;let c=o.x+p.dims.x*(o.y+p.dims.y*o.z);let r=find(c,s);if(r!=INVALID){return r;}if(s>=p.maximumLeafSize){break;}s*=2u;}return INVALID;}
fn header(row:u32)->vec4u{let b=row*12u;return vec4u(arena[b],arena[b+1u],arena[b+2u],arena[b+3u]);}fn metric(row:u32)->vec2u{let b=p.metricBase+row*4u;return vec2u(arena[b],arena[b+1u]);}fn velocity(row:u32)->vec4f{let b=p.velocityBase+row*4u;return vec4f(bitcast<f32>(arena[b]),bitcast<f32>(arena[b+1u]),bitcast<f32>(arena[b+2u]),bitcast<f32>(arena[b+3u]));}
fn axisFace(index:u32)->AxisFace{let b=p.p2+index*8u;return AxisFace(arena[b],arena[b+1u],arena[b+2u],arena[b+3u],arena[b+4u],arena[b+5u],bitcast<f32>(arena[b+6u]),bitcast<f32>(arena[b+7u]));}
fn axisHash(o:vec3u,axisSpan:u32)->u32{var h=o.x*73856093u;h^=o.y*19349663u;h^=o.z*83492791u;h^=axisSpan*2654435761u;h^=h>>16u;return h;}
fn findAxis(o:vec3u,axis:u32,size:u32)->vec2f{if(p.p1+8u>arrayLength(&arena)){return vec2f(0.);}let count=arena[p.p1];let cap=arena[p.p1+6u];if(arena[p.p1+7u]!=VALID||cap==0u||(cap&(cap-1u))!=0u||p.p3+cap>arrayLength(&arena)){return vec2f(0.);}let axisSpan=axis|(size<<2u);let start=axisHash(o,axisSpan)&(cap-1u);for(var probe=0u;probe<min(64u,cap);probe+=1u){let encoded=arena[p.p3+((start+probe)&(cap-1u))];if(encoded==0u){return vec2f(0.);}let index=encoded-1u;if(index>=count){return vec2f(0.);}let face=axisFace(index);if(face.originX==o.x&&face.originY==o.y&&face.originZ==o.z&&face.axisSpan==axisSpan&&finite(face.normalVelocity)){return vec2f(face.normalVelocity,1.);}}return vec2f(0.);}
// Aanjaneya et al. (2017), Section 5 explicitly reverts to staggered
// per-axis face interpolation in regular regions. At a regular face centroid
// and dt=0 this samples that exact face DOF, rather than applying a
// face->cell->face [1/4,1/2,1/4] smoothing pass.
fn axisOriginInDomain(raw:vec3i,axis:u32,size:u32)->bool{
 if(any(raw<vec3i(0))){return false;}let o=vec3u(raw);if(o[axis]>p.dims[axis]){return false;}
 for(var d=0u;d<3u;d+=1u){if(d!=axis&&o[d]+size>p.dims[d]){return false;}}return true;
}
fn cubicWeight(offset:u32,t:f32)->f32{let t2=t*t;let t3=t2*t;if(offset==0u){return -.5*t+t2-.5*t3;}if(offset==1u){return 1.-2.5*t2+1.5*t3;}if(offset==2u){return .5*t+2.*t2-1.5*t3;}return -.5*t2+.5*t3;}
fn regularAxisComponentAtSize(x:vec3f,size:u32,axis:u32)->vec2f{
 if(size==0u){return vec2f(0.);}let s=f32(size);let g=x/p.cellSize;var low=vec3i(0);var t=vec3f(0.);
 // Section 5 extrapolates the regular octree-face field outside the liquid.
 // Locate the staggered interpolation cell from that field itself; requiring
 // a liquid pressure-row owner here would make the extrapolated air band
 // unreachable during the characteristic trace.
 for(var d=0u;d<3u;d+=1u){if(d==axis){low[d]=i32(floor(g[d]/s))*i32(size);t[d]=(g[d]-f32(low[d]))/s;}else{low[d]=i32(floor((g[d]-.5*s)/s))*i32(size);t[d]=(g[d]-(f32(low[d])+.5*s))/s;}}
 if(any(t<vec3f(-2e-5))||any(t>vec3f(1.00002))){return vec2f(0.);}t=clamp(t,vec3f(0.),vec3f(1.));
 // Preserve the exact staggered DOF without touching irrelevant neighbors.
 if(all(t<=vec3f(1e-7))){if(!axisOriginInDomain(low,axis,size)){return vec2f(0.);}return findAxis(vec3u(low),axis,size);}
 // The paper's trilinear value remains the fail-safe and supplies the local
 // extremum bounds for the higher-order reconstruction below.
 var linear=0.;var minimum=3.402823e38;var maximum=-3.402823e38;
 for(var corner=0u;corner<8u;corner+=1u){let bits=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let w=select(1.-t.x,t.x,bits.x!=0)*select(1.-t.y,t.y,bits.y!=0)*select(1.-t.z,t.z,bits.z!=0);if(w==0.){continue;}let raw=low+bits*i32(size);if(!axisOriginInDomain(raw,axis,size)){return vec2f(0.);}let sample=findAxis(vec3u(raw),axis,size);if(sample.y==0.){return vec2f(0.);}linear+=w*sample.x;minimum=min(minimum,sample.x);maximum=max(maximum,sample.x);}
 if(!finite(linear)){return vec2f(0.);}
 // A bounded Catmull-Rom tensor stencil removes the repeated first-order
 // low-pass filter. Missing support near walls/transitions falls back to the
 // complete trilinear value; clamping prevents new component extrema.
 var cubic=0.;
 for(var stencil=0u;stencil<64u;stencil+=1u){let offsets=vec3u(stencil&3u,(stencil>>2u)&3u,(stencil>>4u)&3u);let w=cubicWeight(offsets.x,t.x)*cubicWeight(offsets.y,t.y)*cubicWeight(offsets.z,t.z);if(abs(w)<=1e-10){continue;}let raw=low+(vec3i(offsets)-vec3i(1))*i32(size);if(!axisOriginInDomain(raw,axis,size)){return vec2f(linear,1.);}let sample=findAxis(vec3u(raw),axis,size);if(sample.y==0.){return vec2f(linear,1.);}cubic+=w*sample.x;}
 if(!finite(cubic)){return vec2f(linear,1.);}return vec2f(clamp(cubic,minimum,maximum),1.);
}
fn regularVector(x:vec3f)->vec4f{
 var size=1u;loop{
  let vx=regularAxisComponentAtSize(x,size,0u);let vy=regularAxisComponentAtSize(x,size,1u);let vz=regularAxisComponentAtSize(x,size,2u);
  if(vx.y>0.&&vy.y>0.&&vz.y>0.){return vec4f(vx.x,vy.x,vz.x,1.);}
  if(size>=p.maximumLeafSize){break;}size*=2u;
 }
 return vec4f(0.,0.,0.,-13.);
}
fn inv(x:vec3f,c:u32)->vec3f{let bits=c&7u;let q=x*vec3f(select(1.,-1.,(bits&1u)!=0u),select(1.,-1.,(bits&2u)!=0u),select(1.,-1.,(bits&4u)!=0u));let k=(c/8u)%6u;if(k==0u){return q;}if(k==1u){return q.xzy;}if(k==2u){return q.yxz;}if(k==3u){return q.zxy;}if(k==4u){return q.yzx;}return q.zyx;}
fn weights(point:vec3f,a:vec3f,b:vec3f,c:vec3f)->vec4f{let d=dot(a,cross(b,c));if(abs(d)<=1e-9){return vec4f(-1.);}let q=vec3f(dot(point,cross(b,c)),dot(a,cross(point,c)),dot(a,cross(b,point)))/d;return vec4f(1.-q.x-q.y-q.z,q);}
fn contained(w:vec4f)->bool{return all(w>=vec4f(-2e-5))&&all(w<=vec4f(1.00002));}
fn bad(code:f32)->vec4f{return vec4f(0.,0.,0.,-code);}fn sampleOld(x:vec3f)->vec4f{let oldRows=atomicLoad(&control.faceCount);let row=owner(x);if(row==INVALID||row>=oldRows||row>=p.rowCapacity){let regular=regularVector(x);return select(bad(1.),regular,regular.w>0.);}let h=header(row);let m=metric(row);let anchor=velocity(row);if(anchor.w<=0.||m.x>=arrayLength(&tetraHeaders)||(m.y&VALID)==0u){return bad(2.);}let o=vec3u(h.x%p.dims.x,(h.x/p.dims.x)%p.dims.y,h.x/(p.dims.x*p.dims.y));let size=h.w;let center=(vec3f(o)+.5*f32(size))*p.cellSize;let th=tetraHeaders[m.x];if((th.flags&1u)!=0u){let vx=regularAxisComponentAtSize(x,size,0u);let vy=regularAxisComponentAtSize(x,size,1u);let vz=regularAxisComponentAtSize(x,size,2u);if(vx.y==0.||vy.y==0.||vz.y==0.){return bad(13.);}return vec4f(vx.x,vy.x,vz.x,1.);}
 let point=inv((x-center)/(f32(size)*p.cellSize),m.y&63u);if(th.first>arrayLength(&tetrahedra)||th.count>arrayLength(&tetrahedra)-th.first){return bad(7.);}for(var local=0u;local<th.count;local+=1u){let packed=tetrahedra[th.first+local];let s=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(s>=vec3u(arrayLength(&vertices)))){return bad(8.);}let w=weights(point,vertices[s.x].v.xyz,vertices[s.y].v.xyz,vertices[s.z].v.xyz);if(!contained(w)){continue;}var vv:array<vec4f,3>;for(var k=0u;k<3u;k+=1u){let q=vertices[s[k]].v;let ns=u32(round(f32(size)*q.w));let nc=center+f32(size)*p.cellSize*inv(q.xyz,m.y&63u);let no=round(nc/p.cellSize-.5*f32(ns));if(any(no<vec3f(0))){return bad(9.);}let n=vec3u(no);let r=find(n.x+p.dims.x*(n.y+p.dims.y*n.z),ns);if(r==INVALID||r>=oldRows){return bad(10.);}vv[k]=velocity(r);if(vv[k].w<=0.){return bad(11.);}}let result=w.x*anchor.xyz+w.y*vv[0].xyz+w.z*vv[1].xyz+w.w*vv[2].xyz;return vec4f(result,1.);}return bad(12.);}
// A power-face centroid lies on the boundary of two dual interpolation
// elements.  The paper's dual cube/tetrahedral interpolant includes that
// boundary, whereas the half-open integer owner map assigns an exact grid
// plane to only one side.  At a liquid-air face that selected side need not
// be a pressure row.  Resolve only that measure-zero ambiguity through the
// incident liquid-side element; a genuinely air-side point still fails.
fn sampleOldIncident(x:vec3f,n:vec3f)->vec4f{let direct=sampleOld(x);if(direct.w>0.||direct.w!=-1.){return direct;}let epsilon=p.cellSize*1e-4;let incident=sampleOld(x-epsilon*n);if(incident.w>0.){return incident;}return direct;}
@compute @workgroup_size(1)fn prepareNewPowerFaceAdvection(){let oldGeneration=control.generation;let oldValid=atomicLoad(&control.valid);let oldRows=atomicLoad(&control.faceCount);atomicStore(&control.flags,0u);atomicStore(&control.firstError,INVALID);atomicStore(&control.advected,0u);atomicStore(&control.firstInterpolationStage,INVALID);atomicStore(&control.firstInterpolationReason,INVALID);control.p7=INVALID;control.oldGeneration=oldGeneration;control.generation=p.generation;atomicStore(&control.valid,0u);atomicStore(&control.coldFallback,0u);let count=select(0u,faceControl[1],arrayLength(&faceControl)>=9u);if(p.generation==1u&&oldValid!=VALID){atomicStore(&control.coldFallback,1u);atomicStore(&control.valid,VALID);}else if(oldValid!=VALID||oldRows==0u){fail(SOURCE,0u);}else if(oldGeneration+1u!=p.generation){fail(GENERATION,0u);}else if(faceControl[8]!=VALID||faceControl[7]!=p.generation||count>p.faceCapacity||count>arrayLength(&faces)||count>arrayLength(&normals)||count>arrayLength(&centroids)){fail(CAPACITY,0u);}atomicStore(&control.faceCount,oldRows);control.p0=count;}
fn storeStatus(i:u32,status:u32){centroids[i].w=bitcast<f32>(status);}fn pending(i:u32,stage:u32,value:vec4f){let reason=u32(max(1.,-value.w));storeStatus(i,(stage<<8u)|reason);atomicMin(&control.firstInterpolationStage,i*4u+stage);atomicMin(&control.firstInterpolationReason,i*16u+min(reason,15u));if(p.p0==0u){fail(INTERPOLATION,i);}}
@compute @workgroup_size(64)fn advectNewPowerFaces(@builtin(global_invocation_id)id:vec3u){let i=id.x;let count=control.p0;if(i>=p.faceCapacity||i>=arrayLength(&centroids)){return;}if(i>=count){storeStatus(i,0u);return;}if(atomicLoad(&control.flags)!=0u||atomicLoad(&control.coldFallback)!=0u){return;}storeStatus(i,0u);let f=faces[i];let n=normals[i].xyz;let x=centroids[i].xyz;if((f.flags&VALID)==0u||!finite(x.x)||!finite(x.y)||!finite(x.z)||!finite(n.x)||!finite(n.y)||!finite(n.z)){fail(NONFINITE,i);return;}if(f.positiveRow==INVALID&&(f.flags&BOUNDARY)!=0u&&(f.flags&OPEN)==0u){faces[i].normalVelocity=0.;storeStatus(i,STATUS_VALID);atomicAdd(&control.advected,1u);}else{let v0=sampleOldIncident(x,n);if(v0.w<=0.){pending(i,1u,v0);return;}let vm=sampleOldIncident(x-.5*p.dt*v0.xyz,n);if(vm.w<=0.){pending(i,2u,vm);return;}let va=sampleOldIncident(x-p.dt*vm.xyz,n);if(va.w<=0.){pending(i,3u,va);return;}let value=dot(va.xyz,n);if(!finite(value)){fail(NONFINITE,i);return;}faces[i].normalVelocity=value;storeStatus(i,STATUS_VALID);atomicAdd(&control.advected,1u);}}
@compute @workgroup_size(1)fn publishNewPowerFaceAdvection(){if(atomicLoad(&control.coldFallback)!=0u){control.p1=0u;control.p2=INVALID;control.p3=0u;control.p4=VALID;return;}if(atomicLoad(&control.flags)==0u&&atomicLoad(&control.advected)==control.p0){atomicStore(&control.valid,VALID);}else{atomicStore(&control.valid,0u);atomicStore(&seed.valid,0u);}control.p1=atomicLoad(&control.flags);control.p2=atomicLoad(&control.firstError);control.p3=atomicLoad(&control.advected);control.p4=atomicLoad(&control.valid);}
`;
