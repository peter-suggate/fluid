/**
 * Sparse solid signed-distance samples for the power embedded-boundary solve.
 *
 * Aanjaneya et al. Section 4.1 requires solid signed-distance values at cell
 * vertices.  The simulation's terrain input is a cell-centred height lattice,
 * so this stage materializes eight samples at the actual vertices of every
 * live compact octree owner.  It never allocates the finest logical box.
 */

import { PassBroker } from "./webgpu-pass-broker";

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
  readonly bodyCount: number;
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
    allocatedBytes: sdfBytes + OCTREE_SOLID_VERTEX_SDF_CONTROL_BYTES + 64 + 12 };
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
  private readonly activeDispatch: GPUBuffer;
  private preparePipeline!: GPUComputePipeline;
  private publishPipeline!: GPUComputePipeline;
  private finishPipeline!: GPUComputePipeline;
  private shaderModule!: GPUShaderModule;
  private readonly pipelineLayout: GPUPipelineLayout;
  private readonly preparePipelineLayout: GPUPipelineLayout;
  private readonly prepareGroup: GPUBindGroup;
  private readonly group: GPUBindGroup;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    rowCapacity: number,
    leafHeaders: GPUBuffer,
    rowCount: GPUBuffer,
    terrain: GPUTexture,
    rigidBodies: GPUBuffer,
    private readonly rowDeltaControlOffsetWords: number,
  ) {
    this.plan = planOctreeSolidVertexSdf(rowCapacity);
    if (!Number.isSafeInteger(rowDeltaControlOffsetWords) || rowDeltaControlOffsetWords < 0) {
      throw new RangeError("Solid vertex-SDF row-delta control offset must be a non-negative integer");
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.arena = device.createBuffer({
      label: "Sparse octree solid vertex SDF publication",
      size: this.plan.arenaBytes,
      usage: storage,
    });
    this.activeDispatch = device.createBuffer({
      label: "Sparse octree solid vertex SDF active dispatch",
      size: 12,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
    });
    this.params = device.createBuffer({ label: "Sparse octree solid vertex SDF parameters", size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ] });
    this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    // The prepare entry point is the only one that touches the indirect
    // dispatch (binding 6) and the only one that omits the headers, terrain and
    // body bindings, so it needs a layout of its own. Declare that layout
    // explicitly rather than deriving it with "auto": an auto layout silently
    // tracks whatever the shader happens to reference, so adding a binding to
    // prepareSolidVertexSdf without adding it here produced an invalid bind
    // group with no label and no cause, surfacing only as a validation failure
    // in whichever pass first bound it. Declared explicitly, the same drift is
    // a pipeline-creation error naming the binding.
    const prepareLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    this.preparePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [prepareLayout] });
    this.prepareGroup = device.createBindGroup({
      label: "Sparse octree solid vertex SDF prepare",
      layout: prepareLayout,
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 2, resource: { buffer: rowCount } },
        { binding: 4, resource: { buffer: this.arena } },
        { binding: 6, resource: { buffer: this.activeDispatch } },
      ],
    });
    this.group = device.createBindGroup({ label: "Sparse octree solid vertex SDF", layout, entries: [
      { binding: 0, resource: { buffer: this.params } }, { binding: 1, resource: { buffer: leafHeaders } },
      { binding: 2, resource: { buffer: rowCount } }, { binding: 3, resource: terrain.createView() },
      { binding: 4, resource: { buffer: this.arena } },
      { binding: 7, resource: { buffer: rigidBodies } },
    ] });
  }

  async initializePipelines(): Promise<void> {
    if (this.preparePipeline) return;
    this.shaderModule = this.device.createShaderModule({
      label: "Sparse octree solid vertex SDF", code: octreeSolidVertexSdfShader,
    });
    this.preparePipeline = await this.device.createComputePipelineAsync({
      label: "Prepare sparse solid vertex SDF workset", layout: this.preparePipelineLayout,
      compute: { module: this.shaderModule, entryPoint: "prepareSolidVertexSdf" },
    });
    this.publishPipeline = await this.device.createComputePipelineAsync({
      label: "Materialize sparse solid vertex SDF", layout: this.pipelineLayout,
      compute: { module: this.shaderModule, entryPoint: "publishSolidVertexSdf" },
    });
    this.finishPipeline = await this.device.createComputePipelineAsync({
      label: "Validate sparse solid vertex SDF", layout: this.pipelineLayout,
      compute: { module: this.shaderModule, entryPoint: "finishSolidVertexSdf" },
    });
  }

  encode(broker: PassBroker, options: OctreeSolidVertexSdfEncodeOptions): void {
    if (this.destroyed) throw new Error("Solid vertex-SDF stage is destroyed");
    if (options.dimensions.some((value) => !Number.isSafeInteger(value) || value < 1)
      || options.physicalSpacing.some((value) => !Number.isFinite(value) || value <= 0)
      || !Number.isSafeInteger(options.generation) || options.generation < 0 || options.generation > 0xffff_ffff
      || !Number.isSafeInteger(options.bodyCount) || options.bodyCount < 0 || options.bodyCount > 12) {
      throw new RangeError("Solid vertex-SDF encode inputs are invalid");
    }
    const bytes = new ArrayBuffer(64); const words = new Uint32Array(bytes); const floats = new Float32Array(bytes);
    words.set([...options.dimensions, this.plan.rowCapacity], 0);
    floats.set([...options.physicalSpacing, 0], 4);
    words.set([options.generation >>> 0, options.terrainEnabled ? 1 : 0, options.bodyCount, 0], 8);
    words.set([this.rowDeltaControlOffsetWords, 0, 0, 0], 12);
    this.device.queue.writeBuffer(this.params, 0, bytes);
    const prepare = broker.compute({ label: "Prepare sparse owner-vertex solid SDF workset" });
    prepare.setBindGroup(0, this.prepareGroup);
    prepare.setPipeline(this.preparePipeline); prepare.dispatchWorkgroups(1);
    broker.fence("solid vertex SDF active-row dispatch published");
    const publish = broker.compute({ label: "Materialize sparse owner-vertex solid SDF" });
    publish.setBindGroup(0, this.group);
    publish.setPipeline(this.publishPipeline); publish.dispatchWorkgroupsIndirect(this.activeDispatch, 0);
    const finish = broker.compute({ label: "Publish sparse owner-vertex solid SDF" });
    finish.setPipeline(this.finishPipeline); finish.setBindGroup(0, this.group); finish.dispatchWorkgroups(1);
  }

  get source(): OctreeSolidVertexSdfSource { return { plan: this.plan, arena: this.arena }; }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.arena.destroy(); this.params.destroy(); this.activeDispatch.destroy();
  }
}

export const octreeSolidVertexSdfShader = /* wgsl */ `
struct Header{cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,p0:u32,p1:u32,gradient:vec4f}
struct Params{dims:vec4u,spacing:vec4f,publication:vec4u,rowDelta:vec4u}
struct VertexArena{control:array<u32,16>,values:array<u32>}
struct RigidBody{positionShape:vec4f,dimensions:vec4f,orientation:vec4f,linearVelocity:vec4f,angularVelocity:vec4f,inverseMassInertia:vec4f,angularMomentumRestitution:vec4f,material:vec4f}
@group(0)@binding(0)var<uniform>params:Params;
@group(0)@binding(1)var<storage,read>headers:array<Header>;
@group(0)@binding(2)var<storage,read>rowCountSource:array<u32>;
@group(0)@binding(3)var terrain:texture_2d<f32>;
@group(0)@binding(4)var<storage,read_write>arena:VertexArena;
@group(0)@binding(6)var<storage,read_write>activeDispatch:array<u32>;
@group(0)@binding(7)var<storage,read>rigidBodies:array<RigidBody>;
const VALID:u32=${OCTREE_SOLID_VERTEX_SDF_VALID}u;const INVALID:u32=0xffffffffu;
const SOURCE:u32=${OCTREE_SOLID_VERTEX_SDF_ERROR.source}u;const CAPACITY:u32=${OCTREE_SOLID_VERTEX_SDF_ERROR.capacity}u;
const HEADER:u32=${OCTREE_SOLID_VERTEX_SDF_ERROR.header}u;const NONFINITE:u32=${OCTREE_SOLID_VERTEX_SDF_ERROR.nonfinite}u;
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}
fn hasSolidSource()->bool{return params.publication.y!=0u||params.publication.z!=0u;}
fn fail(code:u32,index:u32){let base=index*8u;if(base+1u<arrayLength(&arena.values)){arena.values[base]=0x7fc00000u;arena.values[base+1u]=code;}}
fn heightAtVertex(g:vec2f)->f32{let extent=vec2i(textureDimensions(terrain));if(any(extent<=vec2i(0))){return 0.0;}
  let p=g-vec2f(0.5);let base=vec2i(floor(p));let t=clamp(p-vec2f(base),vec2f(0),vec2f(1));
  let hi=extent-vec2i(1);let a=textureLoad(terrain,clamp(base,vec2i(0),hi),0).x;let b=textureLoad(terrain,clamp(base+vec2i(1,0),vec2i(0),hi),0).x;
  let c=textureLoad(terrain,clamp(base+vec2i(0,1),vec2i(0),hi),0).x;let d=textureLoad(terrain,clamp(base+vec2i(1),vec2i(0),hi),0).x;
  return mix(mix(a,b,t.x),mix(c,d,t.x),t.y);}
fn quaternionRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);let uuv=cross(q.yzw,uv);return v+2.0*(q.x*uv+uuv);}
fn quaternionInverseRotate(q:vec4f,v:vec3f)->vec3f{return quaternionRotate(vec4f(q.x,-q.yzw),v);}
fn rigidSdf(body:RigidBody,world:vec3f)->f32{
  let q=quaternionInverseRotate(body.orientation,world-body.positionShape.xyz);let d=body.dimensions.xyz;
  let shape=i32(round(body.positionShape.w));
  if(shape==0){return length(q)-d.x;}
  if(shape==1){let b=abs(q)-.5*d;return length(max(b,vec3f(0)))+min(max(b.x,max(b.y,b.z)),0.);}
  if(shape==2){let cy=clamp(q.y,-.5*d.y,.5*d.y);return length(vec3f(q.x,q.y-cy,q.z))-d.x;}
  let radial=length(q.xz)-d.x;let axial=abs(q.y)-.5*d.y;let outside=length(max(vec2f(radial,axial),vec2f(0)));
  return outside+min(max(radial,axial),0.);
}
fn worldVertex(vertex:vec3u)->vec3f{let spacing=params.spacing.xyz;return vec3f(
  -.5*f32(params.dims.x)*spacing.x+f32(vertex.x)*spacing.x,
  f32(vertex.y)*spacing.y,
  -.5*f32(params.dims.z)*spacing.z+f32(vertex.z)*spacing.z);}
// The candidate row count is the row-delta control header, exactly the word
// beginStructuredPublication takes as its own row count. The general
// compaction arena is concurrently reused for dirty-tile work and is cleared
// to zero there, so reading its word 0 sampled a tile count -- and, whenever
// the topology rebuild happened to be mid-flight, a zero that failed this
// publication closed for the rest of the run.
fn candidateRowCount()->u32{let at=params.rowDelta.x;
  return select(0u,min(rowCountSource[at],params.dims.w),at<arrayLength(&rowCountSource));}
// Header word 7 of the same control block: the GPU-published candidate epoch,
// exactly what the descriptor's own generation() reads. Deriving the stamp here
// keeps the retry generation GPU-authored -- a host attempt stamp cannot
// witness a rollback that happened after the host wrote it.
fn candidateGeneration()->u32{let at=params.rowDelta.x+7u;
  return select(0u,rowCountSource[at],at<arrayLength(&rowCountSource));}
@compute @workgroup_size(1)fn prepareSolidVertexSdf(){let count=candidateRowCount();
  let generation=candidateGeneration();
  arena.control[0]=0u;arena.control[1]=count;arena.control[2]=0u;arena.control[3]=0u;arena.control[4]=generation;arena.control[5]=0u;arena.control[6]=INVALID;arena.control[7]=select(0u,1u,hasSolidSource());arena.control[8]=0u;
  activeDispatch[0]=(count+63u)/64u;activeDispatch[1]=1u;activeDispatch[2]=1u;}
@compute @workgroup_size(64)fn publishSolidVertexSdf(@builtin(global_invocation_id)id:vec3u){let row=id.x;
  let count=arena.control[1];if(row>=count){return;}
  if(!hasSolidSource()||row>=arrayLength(&headers)||row*8u+7u>=arrayLength(&arena.values)){fail(select(SOURCE,CAPACITY,hasSolidSource()),row);return;}
  let h=headers[row];let volume=params.dims.x*params.dims.y*params.dims.z;let origin=vec3u(h.cell%params.dims.x,(h.cell/params.dims.x)%params.dims.y,h.cell/(params.dims.x*params.dims.y));
  if(h.cell>=volume||h.size==0u||(h.size&(h.size-1u))!=0u||any(origin%h.size!=vec3u(0))||any(origin+vec3u(h.size)>params.dims.xyz)){fail(HEADER,row);return;}
  for(var corner=0u;corner<8u;corner+=1u){let vertex=origin+vec3u(select(0u,h.size,(corner&1u)!=0u),select(0u,h.size,(corner&2u)!=0u),select(0u,h.size,(corner&4u)!=0u));
    var sdf=1e20;if(params.publication.y!=0u){sdf=(f32(vertex.y)-heightAtVertex(vec2f(vertex.xz)))*params.spacing.y;}
    let world=worldVertex(vertex);for(var body=0u;body<params.publication.z;body+=1u){if(body>=arrayLength(&rigidBodies)){fail(CAPACITY,row);return;}sdf=min(sdf,rigidSdf(rigidBodies[body],world));}
    if(!finite(sdf)){fail(NONFINITE,row);return;}arena.values[row*8u+corner]=bitcast<u32>(sdf);}}
@compute @workgroup_size(1)fn finishSolidVertexSdf(){let count=arena.control[1];let generation=candidateGeneration();var flags=0u;var first=INVALID;let at=params.rowDelta.x;
  if(at>=arrayLength(&rowCountSource)||rowCountSource[at]>params.dims.w){flags|=CAPACITY;first=0u;}
  if(!hasSolidSource()){flags|=SOURCE;first=0u;}for(var row=0u;row<count;row+=1u){let base=row*8u;if(base+7u>=arrayLength(&arena.values)){flags|=CAPACITY;first=min(first,row);continue;}if(!finite(bitcast<f32>(arena.values[base]))){let code=arena.values[base+1u];flags|=select(NONFINITE,code,code==SOURCE||code==CAPACITY||code==HEADER||code==NONFINITE);first=min(first,row);continue;}for(var corner=1u;corner<8u;corner+=1u){if(!finite(bitcast<f32>(arena.values[base+corner]))){flags|=NONFINITE;first=min(first,row);}}}
  // The former predicate read the ACCEPTED structured control: it compared that
  // publication's row count (which lags by an epoch, by design) against this
  // candidate's, and required word 6 of that struct to equal this module's own
  // VALID magic. Word 6 is the structured projected flag, so the term was
  // constant false: the arena published invalid on every step of every scene
  // with terrain or a rigid body, solidAt() returned 1e20, and no aperture was
  // ever cut by a solid.
  let rollbackValid=generation!=0u&&count!=0u;
  arena.control[8]=generation;if(!rollbackValid){flags|=SOURCE;first=0u;}
  arena.control[0]=flags;arena.control[6]=first;if(flags==0u&&arena.control[7]!=0u){arena.control[2]=count;arena.control[3]=count*8u;arena.control[5]=VALID;}else{arena.control[2]=0u;arena.control[3]=0u;arena.control[5]=0u;}}
`;
