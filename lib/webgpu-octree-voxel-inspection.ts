import type { SceneDescription } from "./model";
import {
  SPARSE_VOXEL_DEBUG_RECORD_STRIDE,
  createSparseVoxelInspectionPublicationController,
  type SparseVoxelInspectionPublicationProducerController,
  type SparseVoxelRenderSource,
} from "./webgpu-voxel-debug";
import { VOXEL_MATERIAL_IDS, packVoxelDebugMaterialTable } from "./voxel-scene";

/**
 * Expanded inspection records for the page-native octree path.
 *
 * This is deliberately lazy and debug-only. The simulation remains compact:
 * one 48-byte record is allocated per compact leaf only after Raw voxels or
 * Brick grid is selected, and every update stays on the GPU.
 */
export const compactOctreeVoxelInspectionShader = /* wgsl */ `
struct LeafHeader { cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f }
struct DebugRecord { origin:vec4f,extent:vec4f,materialAndFlags:vec4u }
struct Params {
  shape:vec4u,
  dims:vec4u,
  worldOrigin:vec4f,
  cellSize:vec4f,
}
@group(0) @binding(0) var<storage,read> headers:array<LeafHeader>;
@group(0) @binding(1) var<storage,read> rowCount:array<u32>;
@group(0) @binding(2) var<uniform> params:Params;
@group(0) @binding(3) var<storage,read_write> voxelRecords:array<DebugRecord>;
@group(0) @binding(4) var<storage,read_write> brickRecords:array<DebugRecord>;

const INVALID:u32=0xffffffffu;
const ACTIVE:u32=1u;

fn inactiveRecord()->DebugRecord {
  return DebugRecord(vec4f(0.0),vec4f(0.0),vec4u(0u,0u,0u,INVALID));
}
@compute @workgroup_size(64)
fn materialize(@builtin(global_invocation_id) gid:vec3u) {
  let row=gid.x;
  if(row>=params.shape.x||row>=arrayLength(&headers)||row>=arrayLength(&voxelRecords)||row>=arrayLength(&brickRecords)){return;}
  let header=headers[row];
  let live=row<min(rowCount[0],params.shape.x)&&header.size>0u;
  if(!live){
    // Publication buffers persist across topology rebuilds. Clear retired
    // rows so a shrinking pressure frontier cannot leave ghost voxels.
    voxelRecords[row]=inactiveRecord();
    brickRecords[row]=inactiveRecord();
    return;
  }
  let nx=params.dims.x;
  let nxy=nx*params.dims.y;
  let cellOrigin=vec3u(header.cell%nx,(header.cell/nx)%params.dims.y,header.cell/nxy);
  let world=params.worldOrigin.xyz+vec3f(cellOrigin)*params.cellSize.xyz;
  let extent=f32(header.size)*params.cellSize.xyz;
  let level=u32(round(log2(max(1.0,f32(header.size)))));
  let record=DebugRecord(vec4f(world,0.0),vec4f(extent,0.0),vec4u(${VOXEL_MATERIAL_IDS.fluid}u,ACTIVE,level,INVALID));
  voxelRecords[row]=record;
  brickRecords[row]=record;
}
`;

export interface CompactOctreeVoxelInspectionSource {
  leafHeaders: GPUBufferBinding;
  rowCount: GPUBufferBinding;
  rowCapacity: number;
}

function createStorageBuffer(device: GPUDevice, label: string, size: number, data?: ArrayBufferView<ArrayBuffer>) {
  const buffer = device.createBuffer({
    label,
    size: Math.max(4, size),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

export class CompactOctreeVoxelInspection {
  readonly source: SparseVoxelRenderSource;
  readonly allocatedBytes: number;

  private readonly device: GPUDevice;
  private readonly rowCapacity: number;
  private readonly records: GPUBuffer[];
  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly publication: SparseVoxelInspectionPublicationProducerController;
  private destroyed = false;

  constructor(
    device: GPUDevice,
    scene: SceneDescription,
    dimensions: readonly [number, number, number],
    compactSource: CompactOctreeVoxelInspectionSource,
  ) {
    this.device = device;
    this.rowCapacity = compactSource.rowCapacity;
    const capacity = compactSource.rowCapacity;
    const recordBytes = capacity * SPARSE_VOXEL_DEBUG_RECORD_STRIDE;
    const voxelRecords = createStorageBuffer(device, "Compact octree raw voxel records", recordBytes);
    const brickRecords = createStorageBuffer(device, "Compact octree leaf-grid records", recordBytes);
    const voxelCount = createStorageBuffer(device, "Compact octree raw voxel count", 4, new Uint32Array([capacity]));
    const brickCount = createStorageBuffer(device, "Compact octree leaf-grid count", 4, new Uint32Array([capacity]));
    const materialData = new Float32Array(packVoxelDebugMaterialTable());
    const materials = createStorageBuffer(device, "Compact octree inspection materials", materialData.byteLength, materialData);
    const params = device.createBuffer({
      label: "Compact octree inspection parameters",
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const parameterData = new ArrayBuffer(64);
    const uints = new Uint32Array(parameterData);
    const floats = new Float32Array(parameterData);
    uints.set([capacity, 0, 0, 0], 0);
    uints.set([dimensions[0], dimensions[1], dimensions[2], 0], 4);
    floats.set([-scene.container.width_m / 2, 0, -scene.container.depth_m / 2, 0], 8);
    floats.set([
      scene.container.width_m / dimensions[0],
      scene.container.height_m / dimensions[1],
      scene.container.depth_m / dimensions[2],
      0,
    ], 12);
    device.queue.writeBuffer(params, 0, parameterData);

    const shaderModule = device.createShaderModule({ label: "Compact octree voxel inspection", code: compactOctreeVoxelInspectionShader });
    this.pipeline = device.createComputePipeline({
      label: "Materialize compact octree voxel inspection",
      layout: "auto",
      compute: { module: shaderModule, entryPoint: "materialize" },
    });
    this.bindGroup = device.createBindGroup({
      label: "Compact octree voxel inspection bindings",
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: compactSource.leafHeaders },
        { binding: 1, resource: compactSource.rowCount },
        { binding: 2, resource: { buffer: params } },
        { binding: 3, resource: { buffer: voxelRecords } },
        { binding: 4, resource: { buffer: brickRecords } },
      ],
    });
    this.publication = createSparseVoxelInspectionPublicationController(false, (encoder) => this.encode(encoder));
    this.source = {
      materialCount: materialData.length / 8,
      revision: 1,
      voxelRecords: { buffer: voxelRecords },
      voxelCount: { buffer: voxelCount },
      brickRecords: { buffer: brickRecords },
      brickCount: { buffer: brickCount },
      materials: { buffer: materials },
      voxelCapacity: capacity,
      brickCapacity: capacity,
      drawContainerGlass: false,
      inspectionPublication: this.publication,
    };
    this.records = [voxelRecords, brickRecords, voxelCount, brickCount, materials, params];
    this.allocatedBytes = this.records.reduce((sum, buffer) => sum + buffer.size, 0);
  }

  encode(encoder: GPUCommandEncoder): void {
    if (this.destroyed || !this.publication.enabled) return;
    const pass = encoder.beginComputePass({ label: "Publish compact octree voxel inspection" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.rowCapacity / 64));
    pass.end();
    this.publication.markEncoded();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const buffer of this.records) buffer.destroy();
  }
}
