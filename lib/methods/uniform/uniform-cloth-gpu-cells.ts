import { UniformGPURadixSort, UniformGPUScan, type UniformGPUOrderingKernels } from './uniform-gpu-radix-sort';
import type { UniformClothGPUStencil } from './uniform-cloth-gpu-stencil';

/**
 * The cloth's per-cell material tables built on the GPU from the same
 * clothCellWeights basis the CPU oracles use (prepareGridDragMaterial and
 * the pore-pressure table): grid drag constitutive cells with each vertex's
 * conservative shares, and the sorted pore-pressure material texture.
 *
 * Each vertex owns eight corner entries, 8v + corner, in the oracle's
 * corner order. Open entries are compacted in entry (insertion) order and
 * stably sorted by local cell; a cell's members are then contiguous in
 * vertex-major order, the order the oracles sum them in. Cells come out in
 * z, y, x order, the pore texture's binary-search order.
 */

/** counts[2] bits. */
export const CLOTH_CELLS_STATUS = { noOpenSupport: 1, fiberFraction: 2, permeability: 4, poreCapacity: 8 } as const;
const ALIGN = 64;
const pad = (words: number) => Math.ceil(words / ALIGN) * ALIGN;
/** Work sections (words): entry weight, porosity weight, hard aperture, key, compaction position, cell start, entry cell. */
export function clothCellSections(vertices: number) {
  const E = 8 * vertices, s = pad(E);
  return { entries: E, weight: 0, porosity: s, hard: 2 * s, key: 3 * s, positions: 4 * s, starts: 5 * s, cell: 6 * s, words: 7 * s };
}

const code = /* wgsl */`
struct Params {origin:vec4f,inverseCell:vec4f,dims:vec4u,misc:vec4f};
struct Material {p:vec4i,volume:vec4f,linear0:vec4f,linear1:vec4f,linear2:vec4f,nonlinear0:vec4f,nonlinear1:vec4f,nonlinear2:vec4f};
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> positions:array<vec4f>;
@group(0) @binding(2) var<storage,read> scalars:array<vec4f>;
@group(0) @binding(3) var<storage,read> cornerApertures:array<f32>;
@group(0) @binding(4) var<storage,read> vertexMaterial:array<vec4f>;
@group(0) @binding(5) var<storage,read_write> work:array<u32>;
@group(0) @binding(6) var<storage,read_write> sortKeys:array<u32>;
@group(0) @binding(7) var<storage,read_write> sortValues:array<u32>;
@group(0) @binding(8) var<storage,read_write> counts:array<atomic<u32>>;
@group(0) @binding(9) var<storage,read_write> dragCells:array<Material>;
@group(0) @binding(10) var<storage,read_write> dragLinks:array<vec2u>;
@group(0) @binding(11) var poreTexture:texture_storage_2d<rgba32float,write>;
// The stencil's key frame: origin, dims.
@group(0) @binding(12) var<storage,read> frame:array<vec4i>;
const ALIGN=${ALIGN}u;
fn vertexCount()->u32{return params.dims.w;}
fn entryCount()->u32{return 8u*vertexCount();}
fn stride()->u32{return (entryCount()+ALIGN-1u)/ALIGN*ALIGN;}
fn weightOf(e:u32)->f32{return bitcast<f32>(work[e]);}
fn porosityOf(e:u32)->f32{return bitcast<f32>(work[stride()+e]);}
fn hardOf(e:u32)->f32{return bitcast<f32>(work[2u*stride()+e]);}
fn open(e:u32)->bool{return hardOf(e)>0.0;}
fn fail(bit:u32){atomicOr(&counts[2],bit);}
fn cellVolume()->f32{return params.misc.w;}

// clothCellWeights at every vertex: the hard-aperture weight, the porosity
// weight (relative to the open volume) and the aperture, per corner.
@compute @workgroup_size(64) fn cornerEntries(@builtin(global_invocation_id) gid:vec3u){
 let v=gid.x;if(v>=vertexCount()){return;}let S=stride();
 let q=(positions[v].xyz-params.origin.xyz)*params.inverseCell.xyz-vec3f(0.5);let base=vec3i(floor(q));let f=q-floor(q);
 var w:array<f32,8>;var normalization=0.0;
 for(var c=0u;c<8u;c++){
  let o=vec3u(c&1u,(c>>1u)&1u,(c>>2u)&1u);w[c]=select(1.0-f.x,f.x,o.x==1u)*select(1.0-f.y,f.y,o.y==1u)*select(1.0-f.z,f.z,o.z==1u);
  if(w[c]>0.0){normalization+=w[c]*cornerApertures[8u*v+c];}
 }
 if(normalization<=1e-12){fail(${CLOTH_CELLS_STATUS.noOpenSupport}u);}
 let origin=frame[0].xyz;let dims=vec3u(frame[1].xyz);
 for(var c=0u;c<8u;c++){
  let e=8u*v+c;let hard=cornerApertures[e];let live=w[c]>0.0&&hard>0.0&&normalization>1e-12;
  let local=base+vec3i(i32(c&1u),i32((c>>1u)&1u),i32((c>>2u)&1u))-origin;
  let inside=all(local>=vec3i(0))&&all(vec3u(local)<dims);
  if(live&&!inside){fail(${CLOTH_CELLS_STATUS.noOpenSupport}u);}
  let keep=live&&inside;
  work[e]=bitcast<u32>(select(0.0,w[c]*hard/normalization,keep));work[S+e]=bitcast<u32>(select(0.0,w[c]/normalization,keep));
  work[2u*S+e]=bitcast<u32>(select(0.0,hard,keep));
  work[3u*S+e]=select(0u,u32(local.x)+dims.x*(u32(local.y)+dims.y*u32(local.z)),keep);work[4u*S+e]=select(0u,1u,keep);
 }
}
@compute @workgroup_size(64) fn compactEntries(@builtin(global_invocation_id) gid:vec3u){
 let e=gid.x;let n=entryCount();if(e>=n){return;}let S=stride();let at=work[4u*S+e];
 if(open(e)){sortKeys[at]=work[3u*S+e];sortValues[at]=e;}
 if(e==n-1u){atomicStore(&counts[0],at+select(0u,1u,open(e)));}
}
fn cellStart(j:u32)->bool{return j==0u||sortKeys[j]!=sortKeys[j-1u];}
fn cellOf(j:u32)->u32{return work[5u*stride()+j]+select(0u,1u,cellStart(j))-1u;}
@compute @workgroup_size(64) fn markCells(@builtin(global_invocation_id) gid:vec3u){
 let j=gid.x;if(j>=entryCount()){return;}work[5u*stride()+j]=select(0u,1u,j<atomicLoad(&counts[0])&&cellStart(j));
}
fn poreStore(i:u32,value:vec4f){let width=textureDimensions(poreTexture).x;textureStore(poreTexture,vec2i(i32(i%width),i32(i/width)),value);}
fn permeability(f:f32,d:f32)->vec2f{
 let logTerm=-log(f)-1.476+2.0*f;let d2=d*d;
 return vec2f((logTerm-0.5*f*f)*d2/(16.0*f),(logTerm-1.774*f*f+4.078*f*f*f)*d2/(32.0*f));
}
// One cell's members, summed in order: the grid drag constitutive record and
// the pore-pressure material record.
@compute @workgroup_size(64) fn writeCells(@builtin(global_invocation_id) gid:vec3u){
 let j=gid.x;let n=atomicLoad(&counts[0]);if(j==0u&&n==0u){poreStore(0u,vec4f(0.0));}if(j>=n){return;}let S=stride();let cell=cellOf(j);
 work[6u*S+sortValues[j]]=cell;
 if(j==n-1u){atomicStore(&counts[1],cell+1u);poreStore(0u,bitcast<vec4f>(vec4u(cell+1u,0u,0u,0u)));}
 if(!cellStart(j)){return;}
 let key=sortKeys[j];let V=cellVolume();
 var fiber=0.0;var bound=0.0;var weight=0.0;var micro=vec2f(0.0);var alpha=mat3x3f(vec3f(0.0),vec3f(0.0),vec3f(0.0));
 var poreSolid=0.0;var poreBound=0.0;var poreWeight=0.0;var coefficient=0.0;var tensor=mat3x3f(vec3f(0.0),vec3f(0.0),vec3f(0.0));
 let identity2=mat3x3f(vec3f(2.0,0.0,0.0),vec3f(0.0,2.0,0.0),vec3f(0.0,0.0,2.0));
 var k=j;
 loop{
  if(k>=n||sortKeys[k]!=key){break;}
  let e=sortValues[k];let v=e/8u;let w=weightOf(e);let porosity=porosityOf(e);let s=scalars[v];
  let m=vertexMaterial[4u*v];let pore=mat3x3f(vertexMaterial[4u*v+1u].xyz,vertexMaterial[4u*v+2u].xyz,vertexMaterial[4u*v+3u].xyz);
  fiber+=w*s.z;bound+=w*s.y;weight+=w;micro+=w*m.zw;alpha+=w*(identity2-pore);
  let control=w*positions[v].w;poreSolid+=porosity*s.z/V;poreBound+=porosity*s.y/V;poreWeight+=control;coefficient+=control*m.y;tensor+=control*pore;
  k++;
 }
 let dims=vec3u(frame[1].xyz);let local=vec3u(key%dims.x,(key/dims.x)%dims.y,key/(dims.x*dims.y));
 let p=vec3i(local)+frame[0].xyz;let hard=hardOf(sortValues[j]);
 if(poreSolid+poreBound>=1.0){fail(${CLOTH_CELLS_STATUS.poreCapacity}u);}
 poreStore(1u+5u*cell,bitcast<vec4f>(vec4i(p,0)));poreStore(2u+5u*cell,vec4f(coefficient/poreWeight,poreSolid,poreBound,hard));
 poreStore(3u+5u*cell,vec4f(tensor[0]/poreWeight,0.0));poreStore(4u+5u*cell,vec4f(tensor[1]/poreWeight,0.0));poreStore(5u+5u*cell,vec4f(tensor[2]/poreWeight,0.0));
 // Fiber-free cells carry no drag (the oracle filters them out).
 var record=Material(vec4i(p,0),vec4f(fiber,bound,0.0,0.0),vec4f(0.0),vec4f(0.0),vec4f(0.0),vec4f(0.0),vec4f(0.0),vec4f(0.0));
 if(fiber>0.0){
  let psi=fiber/V;if(!(psi>0.0&&psi<1.0)){fail(${CLOTH_CELLS_STATUS.fiberFraction}u);dragCells[cell]=record;return;}
  let d=micro.x/weight;let mu=micro.y/weight;let rho=params.misc.x;let exponent=params.misc.z;
  let K=permeability(psi,d);if(!(K.x>0.0&&K.y>0.0)){fail(${CLOTH_CELLS_STATUS.permeability}u);dragCells[cell]=record;return;}
  // porousDragCoefficients at zero (linear) and unit relative speed.
  let scale=select(0.0,1.75/sqrt(150.0)*pow(d,exponent-1.0)*pow(mu,1.0-exponent)*pow(rho,exponent)/pow(1.0-psi,1.5),params.misc.y!=0.0);
  let linear=mu/K;let nonlinear=scale/sqrt(K);let shape=alpha*(1.0/weight);
  let L=shape*(linear.x-linear.y)+mat3x3f(vec3f(linear.y,0.0,0.0),vec3f(0.0,linear.y,0.0),vec3f(0.0,0.0,linear.y));
  let N=shape*(nonlinear.x-nonlinear.y)+mat3x3f(vec3f(nonlinear.y,0.0,0.0),vec3f(0.0,nonlinear.y,0.0),vec3f(0.0,0.0,nonlinear.y));
  record.p.w=1;record.linear0=vec4f(L[0],0.0);record.linear1=vec4f(L[1],0.0);record.linear2=vec4f(L[2],0.0);
  record.nonlinear0=vec4f(N[0],0.0);record.nonlinear1=vec4f(N[1],0.0);record.nonlinear2=vec4f(N[2],0.0);
 }
 dragCells[cell]=record;
}
// Records past the cell count carry no drag.
@compute @workgroup_size(64) fn clearCells(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=entryCount()||i<atomicLoad(&counts[1])){return;}dragCells[i].p.w=0;
}
// Each vertex's conservative share of its corner cells' drag: deposited
// fiber over the cell's fiber, as prepareGridDragMaterial's members.
@compute @workgroup_size(64) fn vertexShares(@builtin(global_invocation_id) gid:vec3u){
 let v=gid.x;if(v>=vertexCount()){return;}let S=stride();
 for(var c=0u;c<8u;c++){
  let e=8u*v+c;var link=vec2u(0xffffffffu,0u);
  if(open(e)){let cell=work[6u*S+e];let fiber=dragCells[cell].volume.x;if(dragCells[cell].p.w!=0){link=vec2u(cell,bitcast<u32>(weightOf(e)*scalars[v].z/fiber));}}
  dragLinks[e]=link;
 }
}`;

type Kernel = 'cornerEntries' | 'compactEntries' | 'markCells' | 'writeCells' | 'clearCells' | 'vertexShares';
const USED = new Map<Kernel, readonly number[]>([
  ['cornerEntries', [0, 1, 3, 5, 8, 12]],
  ['compactEntries', [0, 5, 6, 7, 8]],
  ['markCells', [0, 5, 6, 8]],
  ['writeCells', [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12]],
  ['clearCells', [0, 8, 9]],
  ['vertexShares', [0, 2, 5, 9, 10]],
]);

export interface ClothCellStep {
  readonly origin_m: readonly number[]; readonly cellSize_m: readonly number[];
  readonly density_kg_m3: number; readonly nonlinearDrag: boolean; readonly exponent: number;
}

/** One cloth's GPU cell tables. Vertex inputs are the stencil's buffers;
 * positions carry each vertex's control volume in w. */
export class UniformClothGPUCells {
  readonly counts: GPUBuffer; readonly dragCells: GPUBuffer; readonly dragLinks: GPUBuffer; readonly poreTexture: GPUTexture;
  readonly sections: ReturnType<typeof clothCellSections>;
  private readonly params: GPUBuffer; private readonly work: GPUBuffer;
  private readonly owned: GPUBuffer[] = [];
  private readonly groups = new Map<Kernel, GPUBindGroup>();
  private readonly scans: UniformGPUScan[];
  private readonly sort: UniformGPURadixSort;
  private readonly device: GPUDevice; private readonly pipelines: ReadonlyMap<Kernel, GPUComputePipeline>;
  constructor(cellKernels: UniformClothCellsKernels, kernels: UniformGPUOrderingKernels, readonly stencil: UniformClothGPUStencil) {
    const device = this.device = cellKernels.device; this.pipelines = cellKernels.pipelines;
    const V = stencil.topology.vertices, sections = this.sections = clothCellSections(V), E = sections.entries;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const make = (label: string, bytes: number) => { const b = device.createBuffer({ label: `Cloth GPU cells ${label}`, size: Math.max(16, bytes), usage }); this.owned.push(b); return b; };
    this.params = device.createBuffer({ label: 'Cloth GPU cells parameters', size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); this.owned.push(this.params);
    this.work = make('work', 4 * sections.words); this.counts = make('counts', 16);
    this.dragCells = make('drag cells', 128 * E); this.dragLinks = make('drag shares', 8 * E);
    const width = Math.min(1024, device.limits.maxTextureDimension2D), height = Math.ceil((1 + 5 * E) / width);
    if (height > device.limits.maxTextureDimension2D) throw new Error('Cloth pore-pressure cells exceed the device texture limit');
    this.poreTexture = device.createTexture({ label: 'Cloth GPU pore-pressure material', size: [width, height], format: 'rgba32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.scans = [new UniformGPUScan(kernels, this.work, E, 4 * sections.positions), new UniformGPUScan(kernels, this.work, E, 4 * sections.starts)];
    this.sort = new UniformGPURadixSort(kernels, E, 30, this.counts, 'Cloth cell sort');
    const view = this.poreTexture.createView();
    const bindings: (GPUBuffer | GPUTextureView)[] = [this.params, stencil.positions, stencil.scalars, stencil.cornerApertures, stencil.vertexMaterial,
      this.work, this.sort.keys, this.sort.values, this.counts, this.dragCells, this.dragLinks, view, stencil.frame];
    for (const [kernel, used] of USED) {
      const layout = this.pipelines.get(kernel)!.getBindGroupLayout(0);
      this.groups.set(kernel, device.createBindGroup({ label: `Cloth GPU cells ${kernel}`, layout, entries: used.map(binding => {
        const resource = bindings[binding]!; return { binding, resource: binding === 11 ? resource as GPUTextureView : { buffer: resource as GPUBuffer } };
      }) }));
    }
  }
  static async create(device: GPUDevice, kernels: UniformGPUOrderingKernels, stencil: UniformClothGPUStencil): Promise<UniformClothGPUCells> {
    return new UniformClothGPUCells(await UniformClothCellsKernels.create(device), kernels, stencil);
  }
  writeParameters(step: ClothCellStep): void {
    const data = new ArrayBuffer(64), f = new Float32Array(data), u = new Uint32Array(data);
    f.set([step.origin_m[0]!, step.origin_m[1]!, step.origin_m[2]!, 0], 0);
    f.set([1/step.cellSize_m[0]!, 1/step.cellSize_m[1]!, 1/step.cellSize_m[2]!, 0], 4);
    u.set([0, 0, 0, this.stencil.topology.vertices], 8);
    f.set([step.density_kg_m3, step.nonlinearDrag ? 1 : 0, step.exponent, step.cellSize_m[0]! * step.cellSize_m[1]! * step.cellSize_m[2]!], 12);
    this.device.queue.writeBuffer(this.params, 0, data);
  }
  /** Encode after the stencil: vertex moments come from its vertexMaterial. */
  encode(encoder: GPUCommandEncoder): void {
    const V = this.stencil.topology.vertices, E = this.sections.entries, [entries, cells] = this.scans as [UniformGPUScan, UniformGPUScan];
    encoder.clearBuffer(this.counts);
    const run = (pass: GPUComputePassEncoder, kernel: Kernel, n: number) => { pass.setPipeline(this.pipelines.get(kernel)!); pass.setBindGroup(0, this.groups.get(kernel)!); pass.dispatchWorkgroups(Math.max(1, Math.ceil(n / 64))); };
    let pass = encoder.beginComputePass({ label: 'Cloth GPU cell entries' });
    run(pass, 'cornerEntries', V); entries.encode(pass); run(pass, 'compactEntries', E); pass.end();
    this.sort.encode(encoder);
    pass = encoder.beginComputePass({ label: 'Cloth GPU cell tables' });
    run(pass, 'markCells', E); cells.encode(pass); run(pass, 'writeCells', E); run(pass, 'clearCells', E); run(pass, 'vertexShares', V); pass.end();
  }
  destroy(): void { this.sort.destroy(); for (const scan of this.scans) scan.destroy(); for (const b of this.owned) b.destroy(); this.owned.length = 0; this.poreTexture.destroy(); }
}

/** The GPU cells pipelines: compiled once a device, shared by every cloth. */
export class UniformClothCellsKernels {
  private constructor(readonly device: GPUDevice, readonly pipelines: ReadonlyMap<Kernel, GPUComputePipeline>) {}
  static async create(device: GPUDevice): Promise<UniformClothCellsKernels> {
    const most = Math.max(...[...USED.values()].map(used => used.filter(b => b !== 0 && b !== 11).length));
    if (device.limits.maxStorageBuffersPerShaderStage < most) throw new Error(`Cloth GPU cells need ${most} storage buffers a stage; the device allows ${device.limits.maxStorageBuffersPerShaderStage}`);
    const module = device.createShaderModule({ label: 'Cloth GPU cells', code });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
    if (errors.length) throw new Error(errors.map(m => `Cloth GPU cells ${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
    const pipelines = new Map<Kernel, GPUComputePipeline>();
    await Promise.all([...USED.keys()].map(async entryPoint => pipelines.set(entryPoint, await device.createComputePipelineAsync({ label: `Cloth GPU cells ${entryPoint}`, layout: 'auto', compute: { module, entryPoint } }))));
    return new UniformClothCellsKernels(device, pipelines);
  }
}
