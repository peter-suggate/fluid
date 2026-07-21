/** Fail-closed GPU catalog lookup and PowerRowMetric publication. */

import {
  OCTREE_GENERATED_POWER_CATALOG_MANIFEST,
  type GeneratedOctreePowerCatalogViews,
} from "./generated/octree-power-catalog";
import { OCTREE_POWER_CATALOG_FACE_FLOATS } from "./octree-power-catalog";
import { OCTREE_POWER_ROW_METRIC_BYTES } from "./octree-power-operator";
import { octreePowerCoarseMaskNeedsAcuteRepair } from "./octree-power-topology";

export const OCTREE_POWER_TOPOLOGY_CONTROL_BYTES = 32;
export const OCTREE_POWER_TOPOLOGY_VALID = 0x8000_0000;
export const OCTREE_POWER_TOPOLOGY_BOUNDARY_SHIFT = 8;
export const OCTREE_POWER_TOPOLOGY_BOUNDARY_MASK = 0x0000_3f00;
export const OCTREE_POWER_TOPOLOGY_ERROR = Object.freeze({
  anisotropicCell: 1 << 0,
  lookupMiss: 1 << 1,
  catalogVersion: 1 << 2,
  capacity: 1 << 3,
} as const);

export interface OctreePowerTopologyPlan {
  readonly rowCapacity: number;
  readonly entryCount: number;
  readonly lookupCount: number;
  readonly metricBytes: number;
  readonly catalogBytes: number;
  readonly allocatedBytes: number;
}

export interface OctreePowerTopologySource {
  readonly plan: OctreePowerTopologyPlan;
  readonly metrics: GPUBuffer;
  readonly control: GPUBuffer;
  readonly catalogEntryHeaders: GPUBuffer;
  readonly catalogVolumes: GPUBuffer;
  readonly catalogFaces: GPUBuffer;
  readonly catalogTetrahedronHeaders?: GPUBuffer;
  readonly catalogTetrahedra?: GPUBuffer;
  readonly catalogTetrahedronVertices?: GPUBuffer;
  readonly catalogLookup: GPUBuffer;
  readonly sameOrFinerDirect: GPUBuffer;
  readonly sameOrCoarserDirect: GPUBuffer;
}

export function powerCellSpacingIsotropic(spacing: readonly [number, number, number], tolerance = 1e-5): boolean {
  if (spacing.some((value) => !Number.isFinite(value) || value <= 0) || !Number.isFinite(tolerance) || tolerance < 0) return false;
  return Math.max(...spacing) / Math.min(...spacing) <= 1 + tolerance;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

export function planOctreePowerTopology(rowCapacityValue: number, catalog: GeneratedOctreePowerCatalogViews): OctreePowerTopologyPlan {
  const rowCapacity = positiveInteger(rowCapacityValue, "Power topology row capacity");
  if (catalog.lookup.length % 3 !== 0 || catalog.entryHeaders.length % 2 !== 0
    || catalog.faceData.length % OCTREE_POWER_CATALOG_FACE_FLOATS !== 0 || catalog.entryHeaders.length !== catalog.entryVolumes.length * 2
    || catalog.tetrahedronHeaders.length !== catalog.entryVolumes.length * 3
    || catalog.tetrahedronVertexData.length % 4 !== 0 || catalog.tetrahedronVertexData.length / 4 > 256
    || catalog.sameOrFinerDirect.length !== 1 << 18 || catalog.sameOrCoarserDirect.length !== 1 << 9) {
    throw new RangeError("Power catalog typed-array shape is invalid");
  }
  const entryCount = catalog.entryVolumes.length;
  const lookupCount = catalog.lookup.length / 3;
  const faceCount = catalog.faceData.length / OCTREE_POWER_CATALOG_FACE_FLOATS;
  if (entryCount === 0 || lookupCount === 0 || faceCount === 0) throw new RangeError("Power catalog must not be empty");
  for (let entry = 0; entry < entryCount; entry += 1) {
    const firstFace = catalog.entryHeaders[entry * 2];
    const count = catalog.entryHeaders[entry * 2 + 1];
    if (firstFace > faceCount || count > faceCount - firstFace
      || !(catalog.entryVolumes[entry] > 0) || !Number.isFinite(catalog.entryVolumes[entry])) {
      throw new RangeError("Power catalog entry is invalid");
    }
    const firstTetrahedron = catalog.tetrahedronHeaders[entry * 3];
    const tetrahedronCount = catalog.tetrahedronHeaders[entry * 3 + 1];
    if (firstTetrahedron > catalog.tetrahedronData.length
      || tetrahedronCount > catalog.tetrahedronData.length - firstTetrahedron) {
      throw new RangeError("Power catalog tetrahedron range is invalid");
    }
  }
  for (let index = 0; index < lookupCount; index += 1) {
    const offset = index * 3;
    if ((index > 0 && catalog.lookup[offset] <= catalog.lookup[offset - 3])
      || catalog.lookup[offset + 1] >= entryCount || catalog.lookup[offset + 2] >= 48) {
      throw new RangeError("Power catalog lookup is invalid");
    }
  }
  for (const packed of catalog.sameOrFinerDirect) if (packed === 0xffff_ffff
    || (packed & 0xffff) >= entryCount || (packed >>> 16) >= 48) {
    throw new RangeError("Power catalog same/finer direct lookup is invalid");
  }
  for (let descriptor = 0; descriptor < catalog.sameOrCoarserDirect.length; descriptor += 1) {
    const packed = catalog.sameOrCoarserDirect[descriptor];
    const gradingExclusion = octreePowerCoarseMaskNeedsAcuteRepair(descriptor >>> 3);
    if (packed === 0xffff_ffff ? !gradingExclusion
      : gradingExclusion || (packed & 0xffff) >= entryCount || (packed >>> 16) >= 48) {
      throw new RangeError("Power catalog same/coarser acute-grading lookup is invalid");
    }
  }
  const metricBytes = rowCapacity * OCTREE_POWER_ROW_METRIC_BYTES;
  const catalogBytes = catalog.entryHeaders.byteLength + catalog.entryVolumes.byteLength + catalog.faceData.byteLength
    + catalog.lookup.byteLength + catalog.sameOrFinerDirect.byteLength + catalog.sameOrCoarserDirect.byteLength
    + catalog.tetrahedronHeaders.byteLength + catalog.tetrahedronData.byteLength + catalog.tetrahedronVertexData.byteLength;
  return { rowCapacity, entryCount, lookupCount, metricBytes, catalogBytes,
    allocatedBytes: metricBytes + OCTREE_POWER_TOPOLOGY_CONTROL_BYTES + 16 + 4 + catalogBytes };
}

export class WebGPUOctreePowerTopology {
  readonly plan: OctreePowerTopologyPlan;
  readonly metrics: GPUBuffer;
  readonly control: GPUBuffer;
  readonly catalogEntryHeaders: GPUBuffer;
  readonly catalogVolumes: GPUBuffer;
  readonly catalogFaces: GPUBuffer;
  readonly catalogTetrahedronHeaders: GPUBuffer;
  readonly catalogTetrahedra: GPUBuffer;
  readonly catalogTetrahedronVertices: GPUBuffer;
  readonly catalogLookup: GPUBuffer;
  readonly sameOrFinerDirect: GPUBuffer;
  readonly sameOrCoarserDirect: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly hostRowCount: GPUBuffer;
  private readonly resolvePipeline: GPUComputePipeline;
  private readonly publishPipeline: GPUComputePipeline;
  private readonly layout: GPUBindGroupLayout;
  private readonly device: GPUDevice;
  private destroyed = false;

  constructor(device: GPUDevice, rowCapacity: number, catalog: GeneratedOctreePowerCatalogViews) {
    this.device = device;
    this.plan = planOctreePowerTopology(rowCapacity, catalog);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const upload = (label: string, data: ArrayBufferView) => {
      const buffer = device.createBuffer({
        label,
        size: Math.max(4, data.byteLength),
        // Failure-only Dawn diagnostics read exact selector payloads back to
        // distinguish catalog asymmetry from live topology corruption.
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
      });
      new Uint8Array(buffer.getMappedRange(), 0, data.byteLength)
        .set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      buffer.unmap();
      return buffer;
    };
    this.metrics = device.createBuffer({ label: "Octree power row metrics", size: this.plan.metricBytes, usage: storage });
    this.control = device.createBuffer({ label: "Octree power topology control", size: OCTREE_POWER_TOPOLOGY_CONTROL_BYTES, usage: storage });
    this.params = device.createBuffer({ label: "Octree power topology params", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.hostRowCount = device.createBuffer({ label: "Octree power topology host row count", size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.catalogEntryHeaders = upload("Octree power catalog entry headers", catalog.entryHeaders);
    this.catalogVolumes = upload("Octree power catalog volumes", catalog.entryVolumes);
    this.catalogFaces = upload("Octree power catalog faces", catalog.faceData);
    this.catalogTetrahedronHeaders = upload("Octree power catalog tetrahedron headers", catalog.tetrahedronHeaders);
    this.catalogTetrahedra = upload("Octree power catalog tetrahedra", catalog.tetrahedronData);
    this.catalogTetrahedronVertices = upload("Octree power catalog tetrahedron vertices", catalog.tetrahedronVertexData);
    this.catalogLookup = upload("Octree power catalog lookup", catalog.lookup);
    this.sameOrFinerDirect = upload("Octree power same/finer direct lookup", catalog.sameOrFinerDirect);
    this.sameOrCoarserDirect = upload("Octree power same/coarser direct lookup", catalog.sameOrCoarserDirect);
    this.layout = device.createBindGroupLayout({ label: "Octree power topology layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ] });
    const shaderModule = device.createShaderModule({ label: "Octree power topology", code: octreePowerTopologyShader });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.resolvePipeline = device.createComputePipeline({ label: "Resolve octree power topology", layout: pipelineLayout, compute: { module: shaderModule, entryPoint: "resolvePowerTopology" } });
    this.publishPipeline = device.createComputePipeline({ label: "Publish octree power topology", layout: pipelineLayout, compute: { module: shaderModule, entryPoint: "publishPowerTopology" } });
  }

  private createGroup(descriptors: GPUBuffer, rowCountSource: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({ label: "Octree power topology bindings", layout: this.layout, entries: [
      { binding: 0, resource: { buffer: this.params } }, { binding: 1, resource: { buffer: descriptors } },
      { binding: 2, resource: { buffer: this.metrics } }, { binding: 3, resource: { buffer: this.control } },
      { binding: 4, resource: { buffer: this.catalogLookup } }, { binding: 5, resource: { buffer: this.catalogVolumes } },
      { binding: 6, resource: { buffer: this.sameOrFinerDirect } }, { binding: 7, resource: { buffer: this.sameOrCoarserDirect } },
      { binding: 8, resource: { buffer: rowCountSource } },
    ] });
  }

  encode(encoder: GPUCommandEncoder, descriptors: GPUBuffer, rowCount: number | GPUBuffer, physicalSpacing: readonly [number, number, number]): void {
    if (this.destroyed) throw new Error("Octree power topology is destroyed");
    const hostRowCount = typeof rowCount === "number" ? rowCount : this.plan.rowCapacity;
    if (!Number.isSafeInteger(hostRowCount) || hostRowCount < 0 || hostRowCount > this.plan.rowCapacity) throw new RangeError("Power topology row count exceeds capacity");
    if (typeof rowCount === "number") this.device.queue.writeBuffer(this.hostRowCount, 0, new Uint32Array([rowCount]));
    const anisotropic = powerCellSpacingIsotropic(physicalSpacing) ? 0 : 1;
    this.device.queue.writeBuffer(this.params, 0, new Uint32Array([
      hostRowCount, this.plan.lookupCount,
      anisotropic, OCTREE_GENERATED_POWER_CATALOG_MANIFEST.version,
    ]));
    const group = this.createGroup(descriptors, typeof rowCount === "number" ? this.hostRowCount : rowCount);
    if (hostRowCount > 0) {
      const resolve = encoder.beginComputePass({ label: "Resolve power topology descriptors" });
      resolve.setPipeline(this.resolvePipeline); resolve.setBindGroup(0, group); resolve.dispatchWorkgroups(Math.ceil(hostRowCount / 64)); resolve.end();
    }
    const publish = encoder.beginComputePass({ label: "Publish power topology control" });
    publish.setPipeline(this.publishPipeline); publish.setBindGroup(0, group); publish.dispatchWorkgroups(1); publish.end();
  }

  get source(): OctreePowerTopologySource {
    return { plan: this.plan, metrics: this.metrics, control: this.control, catalogEntryHeaders: this.catalogEntryHeaders,
      catalogVolumes: this.catalogVolumes, catalogFaces: this.catalogFaces, catalogLookup: this.catalogLookup,
      catalogTetrahedronHeaders: this.catalogTetrahedronHeaders, catalogTetrahedra: this.catalogTetrahedra,
      catalogTetrahedronVertices: this.catalogTetrahedronVertices,
      sameOrFinerDirect: this.sameOrFinerDirect, sameOrCoarserDirect: this.sameOrCoarserDirect };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.metrics.destroy(); this.control.destroy(); this.params.destroy(); this.hostRowCount.destroy();
    this.catalogEntryHeaders.destroy(); this.catalogVolumes.destroy(); this.catalogFaces.destroy(); this.catalogLookup.destroy();
    this.catalogTetrahedronHeaders.destroy(); this.catalogTetrahedra.destroy();
    this.catalogTetrahedronVertices.destroy();
    this.sameOrFinerDirect.destroy(); this.sameOrCoarserDirect.destroy();
  }
}

export const octreePowerTopologyShader = /* wgsl */ `
struct Params { rowCount:u32, lookupCount:u32, anisotropic:u32, catalogVersion:u32 }
struct PowerRowMetric { topologyCode:u32, transformAndFlags:u32, volume:f32, reserved:u32 }
struct CatalogLookup { descriptor:u32, entry:u32, transform:u32 }
struct Control { invalidCount:u32, firstInvalid:u32, flags:u32, resolvedCount:u32, version:u32, pad0:u32, pad1:u32, pad2:u32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> descriptors:array<u32>;
@group(0) @binding(2) var<storage,read_write> metrics:array<PowerRowMetric>;
@group(0) @binding(3) var<storage,read_write> control:Control;
@group(0) @binding(4) var<storage,read> lookup:array<CatalogLookup>;
@group(0) @binding(5) var<storage,read> volumes:array<f32>;
@group(0) @binding(6) var<storage,read> sameOrFinerDirect:array<u32>;
@group(0) @binding(7) var<storage,read> sameOrCoarserDirect:array<u32>;
@group(0) @binding(8) var<storage,read> rowCountSource:array<u32>;
const INVALID:u32=0xffffffffu;
const VALID:u32=0x80000000u;
const ANISOTROPIC:u32=1u;
const LOOKUP_MISS:u32=2u;
const CAPACITY:u32=8u;
fn powerTransformVector(value:vec3i,code:u32)->vec3i{
  let signs=vec3i(select(1,-1,(code&1u)!=0u),select(1,-1,(code&2u)!=0u),select(1,-1,(code&4u)!=0u));let permutation=(code/8u)%6u;var q=value;
  if(permutation==1u){q=value.xzy;}else if(permutation==2u){q=value.yxz;}else if(permutation==3u){q=value.yzx;}else if(permutation==4u){q=value.zxy;}else if(permutation==5u){q=value.zyx;}return q*signs;
}
fn boundaryDirectionBit(direction:vec3i)->u32{if(direction.x<0){return 0u;}if(direction.y<0){return 1u;}if(direction.z<0){return 2u;}if(direction.z>0){return 3u;}if(direction.y>0){return 4u;}return 5u;}
fn transformBoundaryMask(mask:u32,transform:u32)->u32{let directions=array<vec3i,6>(vec3i(-1,0,0),vec3i(0,-1,0),vec3i(0,0,-1),vec3i(0,0,1),vec3i(0,1,0),vec3i(1,0,0));var result=0u;for(var bit=0u;bit<6u;bit+=1u){if((mask&(1u<<bit))!=0u){result|=1u<<boundaryDirectionBit(powerTransformVector(directions[bit],transform));}}return result;}
fn resolveBoundaryEntry(interiorEntry:u32,canonicalMask:u32)->u32{let key=interiorEntry*64u+canonicalMask;var low=0u;var high=min(params.lookupCount,arrayLength(&lookup));while(low<high){let middle=low+(high-low)/2u;let candidate=lookup[middle].descriptor;if(candidate<key){low=middle+1u;}else{high=middle;}}if(low>=min(params.lookupCount,arrayLength(&lookup))||lookup[low].descriptor!=key){return INVALID;}return lookup[low].entry;}
fn resolveDescriptor(descriptor:u32)->vec2u{let boundary=(descriptor>>24u)&63u;let geometry=descriptor&0xc0ffffffu;var packed=INVALID;if((geometry&0x80000000u)!=0u){let index=geometry&0x1ffu;if((geometry&0x40fffe00u)==0u&&index<arrayLength(&sameOrCoarserDirect)){packed=sameOrCoarserDirect[index];}}else{let index=geometry&0x3ffffu;if((geometry&0x40fc0000u)==0u&&index<arrayLength(&sameOrFinerDirect)){packed=sameOrFinerDirect[index];}}if(packed==INVALID){return vec2u(INVALID);}let transform=packed>>16u;var entry=packed&0xffffu;if(boundary!=0u){entry=resolveBoundaryEntry(entry,transformBoundaryMask(boundary,transform));if(entry==INVALID){return vec2u(INVALID);}}return vec2u(entry,transform);}
fn requestedRows()->u32{return select(0u,rowCountSource[0],arrayLength(&rowCountSource)>0u);}
@compute @workgroup_size(64) fn resolvePowerTopology(@builtin(global_invocation_id) gid:vec3u){let row=gid.x;let requested=requestedRows();if(row>=requested||row>=arrayLength(&descriptors)||row>=arrayLength(&metrics)){return;}if(params.anisotropic!=0u){metrics[row]=PowerRowMetric(INVALID,0u,0.0,ANISOTROPIC);return;}let descriptor=descriptors[row];let found=resolveDescriptor(descriptor);if(found.x==INVALID||found.x>=arrayLength(&volumes)){metrics[row]=PowerRowMetric(INVALID,0u,0.0,LOOKUP_MISS);return;}let boundary=(descriptor>>16u)&0x3f00u;metrics[row]=PowerRowMetric(found.x,found.y|boundary|VALID,volumes[found.x],0u);}
@compute @workgroup_size(1) fn publishPowerTopology(){let requested=requestedRows();let available=min(requested,min(arrayLength(&descriptors),arrayLength(&metrics)));var invalidCount=requested-available;var resolvedCount=0u;var firstInvalid=select(INVALID,available,available<requested);var flags=select(0u,ANISOTROPIC,params.anisotropic!=0u);if(available<requested){flags|=CAPACITY;}for(var row=0u;row<available;row+=1u){let metric=metrics[row];if((metric.transformAndFlags&VALID)!=0u){resolvedCount+=1u;}else{invalidCount+=1u;firstInvalid=min(firstInvalid,row);flags|=metric.reserved;}}control.invalidCount=invalidCount;control.firstInvalid=firstInvalid;control.flags=flags;control.resolvedCount=resolvedCount;control.version=params.catalogVersion;}
`;
