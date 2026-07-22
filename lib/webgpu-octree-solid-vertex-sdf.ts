/**
 * Sparse solid signed-distance samples for the power embedded-boundary solve.
 *
 * Aanjaneya et al. Section 4.1 requires solid signed-distance values at cell
 * vertices.  The simulation's terrain input is a cell-centred height lattice,
 * so this stage materializes eight samples at the actual vertices of every
 * live compact octree owner.  It never allocates the finest logical box.
 */

export const OCTREE_SOLID_VERTEX_SDF_CONTROL_BYTES = 64;
export const OCTREE_SOLID_VERTEX_SDF_VALID = 0x8000_0000;

export const OCTREE_SOLID_VERTEX_SDF_ERROR = Object.freeze({
  source: 1 << 0,
  capacity: 1 << 1,
  header: 1 << 2,
  nonfinite: 1 << 3,
} as const);

export interface OctreeSolidVertexSdfPlan {
  readonly rowCapacity: number;
  readonly sampleCapacity: number;
  readonly sdfBytes: number;
  readonly arenaBytes: number;
  readonly allocatedBytes: number;
}

export interface OctreeSolidVertexSdfSource {
  readonly plan: OctreeSolidVertexSdfPlan;
  /** Eight f32 values per compact owner row, corner bit order x | y<<1 | z<<2. */
  /** 64-byte publication control followed by eight f32 words per owner row. */
  readonly arena: GPUBuffer;
}

export interface OctreeSolidVertexSdfEncodeOptions {
  readonly dimensions: readonly [number, number, number];
  readonly physicalSpacing: readonly [number, number, number];
  readonly generation: number;
  readonly terrainEnabled: boolean;
}

export interface OctreeSolidVertexSdfCpuRow {
  readonly cell: number;
  readonly size: number;
}

export function planOctreeSolidVertexSdf(rowCapacity: number): OctreeSolidVertexSdfPlan {
  if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1) {
    throw new RangeError("Solid vertex-SDF row capacity must be a positive integer");
  }
  const sampleCapacity = rowCapacity * 8;
  const sdfBytes = sampleCapacity * 4;
  return { rowCapacity, sampleCapacity, sdfBytes, arenaBytes: OCTREE_SOLID_VERTEX_SDF_CONTROL_BYTES + sdfBytes,
    allocatedBytes: sdfBytes + OCTREE_SOLID_VERTEX_SDF_CONTROL_BYTES + 64 };
}

function heightAtVertex(
  heightsInCells: ArrayLike<number>,
  dimensions: readonly [number, number, number],
  gridX: number,
  gridZ: number,
): number {
  const [nx, , nz] = dimensions;
  const sample = (x: number, z: number) => Number(heightsInCells[Math.max(0, Math.min(nz - 1, z)) * nx
    + Math.max(0, Math.min(nx - 1, x))]);
  const sx = gridX - 0.5, sz = gridZ - 0.5;
  const x0 = Math.floor(sx), z0 = Math.floor(sz);
  const tx = Math.max(0, Math.min(1, sx - x0)), tz = Math.max(0, Math.min(1, sz - z0));
  const a = sample(x0, z0) * (1 - tx) + sample(x0 + 1, z0) * tx;
  const b = sample(x0, z0 + 1) * (1 - tx) + sample(x0 + 1, z0 + 1) * tx;
  return a * (1 - tz) + b * tz;
}

/** CPU oracle for sparse-owner placement and the cell-centre-to-vertex interpolation. */
export function materializeOctreeTerrainVertexSdf(
  rows: readonly OctreeSolidVertexSdfCpuRow[],
  heightsInCells: ArrayLike<number>,
  dimensions: readonly [number, number, number],
  physicalSpacing: readonly [number, number, number],
): Float32Array {
  const [nx, ny, nz] = dimensions;
  if (![nx, ny, nz].every((value) => Number.isSafeInteger(value) && value > 0)
    || heightsInCells.length !== nx * nz
    || physicalSpacing.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new RangeError("Solid vertex-SDF lattice inputs are invalid");
  }
  const output = new Float32Array(rows.length * 8);
  const volume = nx * ny * nz;
  rows.forEach((row, rowIndex) => {
    const x = row.cell % nx;
    const y = Math.floor(row.cell / nx) % ny;
    const z = Math.floor(row.cell / (nx * ny));
    if (!Number.isSafeInteger(row.cell) || row.cell < 0 || row.cell >= volume
      || !Number.isSafeInteger(row.size) || row.size < 1 || (row.size & (row.size - 1)) !== 0
      || [x, y, z].some((value, axis) => value % row.size !== 0
        || value + row.size > dimensions[axis])) {
      throw new RangeError(`Solid vertex-SDF row ${rowIndex} is not a canonical octree owner`);
    }
    for (let corner = 0; corner < 8; corner += 1) {
      const gx = x + ((corner & 1) !== 0 ? row.size : 0);
      const gy = y + ((corner & 2) !== 0 ? row.size : 0);
      const gz = z + ((corner & 4) !== 0 ? row.size : 0);
      const sdf = (gy - heightAtVertex(heightsInCells, dimensions, gx, gz)) * physicalSpacing[1];
      if (!Number.isFinite(sdf)) throw new RangeError(`Solid vertex-SDF row ${rowIndex} contains non-finite data`);
      output[rowIndex * 8 + corner] = sdf;
    }
  });
  return output;
}

export class WebGPUOctreeSolidVertexSdf {
  readonly plan: OctreeSolidVertexSdfPlan;
  readonly arena: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly publishPipeline: GPUComputePipeline;
  private readonly finishPipeline: GPUComputePipeline;
  private readonly group: GPUBindGroup;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    rowCapacity: number,
    leafHeaders: GPUBuffer,
    rowCount: GPUBuffer,
    terrain: GPUTexture,
    rollbackSeedControl: GPUBuffer,
  ) {
    this.plan = planOctreeSolidVertexSdf(rowCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.arena = device.createBuffer({ label: "Sparse octree solid vertex SDF publication", size: this.plan.arenaBytes, usage: storage });
    this.params = device.createBuffer({ label: "Sparse octree solid vertex SDF parameters", size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const module = device.createShaderModule({ label: "Sparse octree solid vertex SDF", code: octreeSolidVertexSdfShader });
    const layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    this.publishPipeline = device.createComputePipeline({ label: "Materialize sparse solid vertex SDF", layout: pipelineLayout,
      compute: { module, entryPoint: "publishSolidVertexSdf" } });
    this.finishPipeline = device.createComputePipeline({ label: "Validate sparse solid vertex SDF", layout: pipelineLayout,
      compute: { module, entryPoint: "finishSolidVertexSdf" } });
    this.group = device.createBindGroup({ layout, entries: [
      { binding: 0, resource: { buffer: this.params } }, { binding: 1, resource: { buffer: leafHeaders } },
      { binding: 2, resource: { buffer: rowCount } }, { binding: 3, resource: terrain.createView() },
      { binding: 4, resource: { buffer: this.arena } }, { binding: 5, resource: { buffer: rollbackSeedControl } },
    ] });
  }

  encode(encoder: GPUCommandEncoder, options: OctreeSolidVertexSdfEncodeOptions): void {
    if (this.destroyed) throw new Error("Solid vertex-SDF stage is destroyed");
    if (options.dimensions.some((value) => !Number.isSafeInteger(value) || value < 1)
      || options.physicalSpacing.some((value) => !Number.isFinite(value) || value <= 0)
      || !Number.isSafeInteger(options.generation) || options.generation < 0 || options.generation > 0xffff_ffff) {
      throw new RangeError("Solid vertex-SDF encode inputs are invalid");
    }
    const bytes = new ArrayBuffer(64); const words = new Uint32Array(bytes); const floats = new Float32Array(bytes);
    words.set([...options.dimensions, this.plan.rowCapacity], 0);
    floats.set([...options.physicalSpacing, 0], 4);
    words.set([options.generation >>> 0, options.terrainEnabled ? 1 : 0, 0, 0], 8);
    this.device.queue.writeBuffer(this.params, 0, bytes);
    encoder.clearBuffer(this.arena, 0, OCTREE_SOLID_VERTEX_SDF_CONTROL_BYTES);
    const publish = encoder.beginComputePass({ label: "Materialize sparse owner-vertex solid SDF" });
    publish.setPipeline(this.publishPipeline); publish.setBindGroup(0, this.group);
    publish.dispatchWorkgroups(Math.ceil(this.plan.rowCapacity / 64)); publish.end();
    const finish = encoder.beginComputePass({ label: "Publish sparse owner-vertex solid SDF" });
    finish.setPipeline(this.finishPipeline); finish.setBindGroup(0, this.group); finish.dispatchWorkgroups(1); finish.end();
  }

  get source(): OctreeSolidVertexSdfSource { return { plan: this.plan, arena: this.arena }; }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.arena.destroy(); this.params.destroy();
  }
}

export const octreeSolidVertexSdfShader = /* wgsl */ `
struct Header{cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,p0:u32,p1:u32,gradient:vec4f}
struct Params{dims:vec4u,spacing:vec4f,publication:vec4u,padding:vec4u}
struct VertexArena{control:array<atomic<u32>,16>,values:array<u32>}
@group(0)@binding(0)var<uniform>params:Params;
@group(0)@binding(1)var<storage,read>headers:array<Header>;
@group(0)@binding(2)var<storage,read>rowCountSource:array<u32>;
@group(0)@binding(3)var terrain:texture_2d<f32>;
@group(0)@binding(4)var<storage,read_write>arena:VertexArena;
@group(0)@binding(5)var<storage,read>rollbackSeedControl:array<u32>;
const VALID:u32=${OCTREE_SOLID_VERTEX_SDF_VALID}u;const INVALID:u32=0xffffffffu;
const SOURCE:u32=${OCTREE_SOLID_VERTEX_SDF_ERROR.source}u;const CAPACITY:u32=${OCTREE_SOLID_VERTEX_SDF_ERROR.capacity}u;
const HEADER:u32=${OCTREE_SOLID_VERTEX_SDF_ERROR.header}u;const NONFINITE:u32=${OCTREE_SOLID_VERTEX_SDF_ERROR.nonfinite}u;
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}
fn fail(code:u32,index:u32){atomicOr(&arena.control[0],code);atomicMin(&arena.control[6],index);}
fn heightAtVertex(g:vec2f)->f32{let extent=vec2i(textureDimensions(terrain));if(any(extent<=vec2i(0))){return 0.0;}
  let p=g-vec2f(0.5);let base=vec2i(floor(p));let t=clamp(p-vec2f(base),vec2f(0),vec2f(1));
  let hi=extent-vec2i(1);let a=textureLoad(terrain,clamp(base,vec2i(0),hi),0).x;let b=textureLoad(terrain,clamp(base+vec2i(1,0),vec2i(0),hi),0).x;
  let c=textureLoad(terrain,clamp(base+vec2i(0,1),vec2i(0),hi),0).x;let d=textureLoad(terrain,clamp(base+vec2i(1),vec2i(0),hi),0).x;
  return mix(mix(a,b,t.x),mix(c,d,t.x),t.y);}
@compute @workgroup_size(64)fn publishSolidVertexSdf(@builtin(global_invocation_id)id:vec3u){let row=id.x;
  if(row==0u){atomicStore(&arena.control[1],select(0u,min(rowCountSource[0],params.dims.w),arrayLength(&rowCountSource)>0u));atomicStore(&arena.control[4],params.publication.x);atomicStore(&arena.control[6],INVALID);atomicStore(&arena.control[7],params.publication.y);}
  let count=select(0u,min(rowCountSource[0],params.dims.w),arrayLength(&rowCountSource)>0u);if(row>=count){return;}
  if(params.publication.y==0u||row>=arrayLength(&headers)||row*8u+7u>=arrayLength(&arena.values)){fail(select(SOURCE,CAPACITY,params.publication.y!=0u),row);return;}
  let h=headers[row];let volume=params.dims.x*params.dims.y*params.dims.z;let origin=vec3u(h.cell%params.dims.x,(h.cell/params.dims.x)%params.dims.y,h.cell/(params.dims.x*params.dims.y));
  if(h.cell>=volume||h.size==0u||(h.size&(h.size-1u))!=0u||any(origin%h.size!=vec3u(0))||any(origin+vec3u(h.size)>params.dims.xyz)){fail(HEADER,row);return;}
  for(var corner=0u;corner<8u;corner+=1u){let vertex=origin+vec3u(select(0u,h.size,(corner&1u)!=0u),select(0u,h.size,(corner&2u)!=0u),select(0u,h.size,(corner&4u)!=0u));
    let sdf=(f32(vertex.y)-heightAtVertex(vec2f(vertex.xz)))*params.spacing.y;if(!finite(sdf)){fail(NONFINITE,row);return;}arena.values[row*8u+corner]=bitcast<u32>(sdf);atomicAdd(&arena.control[3],1u);}
  atomicAdd(&arena.control[2],1u);}
@compute @workgroup_size(1)fn finishSolidVertexSdf(){let count=atomicLoad(&arena.control[1]);if(arrayLength(&rowCountSource)==0u||rowCountSource[0]>params.dims.w){fail(CAPACITY,0u);}
  let rollbackValid=arrayLength(&rollbackSeedControl)>=7u&&rollbackSeedControl[2]==count&&rollbackSeedControl[5]==params.publication.x&&rollbackSeedControl[6]==VALID;
  atomicStore(&arena.control[8],select(0u,rollbackSeedControl[5],arrayLength(&rollbackSeedControl)>=6u));if(!rollbackValid){fail(SOURCE,0u);}
  if(atomicLoad(&arena.control[0])==0u&&atomicLoad(&arena.control[2])==count&&atomicLoad(&arena.control[3])==count*8u&&atomicLoad(&arena.control[7])!=0u){atomicStore(&arena.control[5],VALID);}else{atomicStore(&arena.control[5],0u);}}
`;
