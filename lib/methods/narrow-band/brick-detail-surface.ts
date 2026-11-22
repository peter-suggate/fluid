/**
 * A detail engine's liquid surface (brick-detail.ts), meshed from its presentation for the water's interface passes.
 *
 * The parent's mesh is not drawn inside a detail box (the water's surface clip, webgpu-water-pipeline.ts
 * setDetailSurfaces): the box shows this mesh, the fine engine's own φ = 0 contour at h / factor, so the detail is
 * seen, not only simulated. It is the classic marching-cubes mesh of the presentation's vertex φ (brick-present.ts
 * vertexPhi, metres): a cube is a fine cell's eight vertices, and each emitted vertex carries the water's
 * SurfaceVertex layout (world position, w 1: a level-set vertex; the normal, the central-difference gradient of φ
 * interpolated along the edge, outward; film 0), so Shade, Simple and Wire all apply.
 *
 * The box's faces:
 * - a coupled face is open: the fine liquid continues into the parent's, whose mesh meets it there (the relaxation
 *   zone holds the fine φ to the parent's on that face, brick-detail.wgsl.ts);
 * - a face on the container's wall is closed as the parent's mesh is (a vertex past the presented cells is air there),
 *   so the liquid against the wall shows its wall face.
 *
 * One workgroup meshes 8³ cubes, offset by one (cube c spans vertices c .. c + 1, c from -1): its vertices lie in the
 * presentation bricks b - 1 and b on each axis, and a brick the presentation did not write this frame holds far air
 * (BrickPresentation's `shown`), so a workgroup whose eight bricks are all unshown exits at once (empty workgroups are
 * cheap): the work follows the fine engine's liquid, not the box. A workgroup counts its triangles, reserves its
 * vertices once (a reservation past the pool is dropped whole, so the drawn vertices stay a prefix) and then writes.
 */
import { marchingCubesLookupWGSL } from "../../core/marching-cubes-lookup.wgsl";
import type { BrickPresentation } from "./brick-present";

/** SurfaceVertex: position and normal, vec4f each. */
const VERTEX_BYTES = 32;
/** Draw arguments (vertexCount, instanceCount, firstVertex, firstInstance) then the reservation counter. */
const ARGS_BYTES = 32;

export interface BrickDetailSurfaceOptions {
  /** The presented cells (fine). */
  readonly cells: readonly [number, number, number];
  /** The fine lattice's bricks an axis (the presentation's `shown` words). */
  readonly bricks: readonly [number, number, number];
  /** Fine h (m). */
  readonly h_m: number;
  /** Fine vertex 0 in the parent's world (m). */
  readonly origin_m: readonly [number, number, number];
  /** The coupled (open) faces, bit 2a + side. */
  readonly coupled: number;
  /** The air value past the presented cells (m). */
  readonly far_m: number;
}

/** The vertex pool: six vertices (two triangles) a fine cell of twice the box's three face areas. */
export function brickDetailSurfaceCapacity(cells: readonly number[], maxBytes: number): number {
  const [x, y, z] = cells as [number, number, number];
  return Math.max(3, Math.min(Math.floor(maxBytes / VERTEX_BYTES / 3) * 3, 6 * 2 * (x * y + y * z + x * z)));
}

export function brickDetailSurfaceWGSL(o: BrickDetailSurfaceOptions): string {
  const closed = (a: number, side: number) => ((o.coupled >> (2 * a + side)) & 1) === 0;
  const b3 = (side: number) => `vec3<bool>(${[0, 1, 2].map(a => closed(a, side)).join(",")})`;
  const groups = o.cells.map(n => Math.ceil((n + 2) / 8));
  return /* wgsl */`
${marchingCubesLookupWGSL}
struct SurfaceVertex { position:vec4f, normal:vec4f }
const DS_CELLS:vec3i=vec3i(${o.cells.join(",")});
const DS_BRICKS:vec3i=vec3i(${o.bricks.join(",")});
const DS_GROUPS:vec3i=vec3i(${groups.join(",")});
const DS_CLOSED_LO:vec3<bool>=${b3(0)};const DS_CLOSED_HI:vec3<bool>=${b3(1)};
const DS_H:f32=${o.h_m};const DS_FAR:f32=${o.far_m};
const DS_ORIGIN:vec3f=vec3f(${o.origin_m.join(",")});
override capacity:u32=3u;
@group(0) @binding(0) var phi:texture_3d<f32>;
@group(0) @binding(1) var<storage,read> shown:array<u32>;
@group(0) @binding(2) var<storage,read_write> args:array<atomic<u32>,8>;
@group(0) @binding(3) var<storage,read_write> vertices:array<SurfaceVertex>;
var<workgroup> wgCount:atomic<u32>;
var<workgroup> wgBase:u32;
var<workgroup> wgLive:u32;
// Vertex v's φ (m): past the presented cells, air on a closed face and the edge's value on a coupled one.
fn dsPhi(v:vec3i)->f32{
  if(any((v<vec3i(0))&DS_CLOSED_LO)||any((v>DS_CELLS)&DS_CLOSED_HI)){return DS_FAR;}
  return textureLoad(phi,clamp(v,vec3i(0),DS_CELLS),0).x;
}
fn dsGrad(v:vec3i)->vec3f{
  return vec3f(dsPhi(v+vec3i(1,0,0))-dsPhi(v-vec3i(1,0,0)),dsPhi(v+vec3i(0,1,0))-dsPhi(v-vec3i(0,1,0)),dsPhi(v+vec3i(0,0,1))-dsPhi(v-vec3i(0,0,1)));
}
// The cube's corners in marching-cubes order (the water's windowCell order).
const DS_CORNER=array<vec3i,8>(vec3i(0,0,0),vec3i(1,0,0),vec3i(1,1,0),vec3i(0,1,0),vec3i(0,0,1),vec3i(1,0,1),vec3i(1,1,1),vec3i(0,1,1));
// Cube c is meshed: inside the presented cells, or one past them on a closed face (its wall face).
fn dsCube(c:vec3i)->bool{
  let lo=select(vec3i(0),vec3i(-1),DS_CLOSED_LO);let hi=DS_CELLS+select(vec3i(0),vec3i(1),DS_CLOSED_HI);
  return all(c>=lo)&&all(c<hi);
}
fn dsCase(c:vec3i,value:ptr<function,array<f32,8>>)->u32{
  for(var k=0u;k<8u;k++){(*value)[k]=0.5-dsPhi(c+DS_CORNER[k])/DS_H;}
  return mcCase(value);
}
fn dsVertex(c:vec3i,edge:u32,value:ptr<function,array<f32,8>>)->SurfaceVertex{
  let a=MC_EDGE_A[edge];let b=MC_EDGE_B[edge];let pa=c+DS_CORNER[a];let pb=c+DS_CORNER[b];
  let fa=(*value)[a]-0.5;let fb=(*value)[b]-0.5;
  let t=clamp(fa/select(fa-fb,1e-12,abs(fa-fb)<1e-12),0.0,1.0);
  let p=mix(vec3f(pa),vec3f(pb),t);
  var n=mix(dsGrad(pa),dsGrad(pb),t);
  if(dot(n,n)<1e-20){n=vec3f(0.0,1.0,0.0);}
  return SurfaceVertex(vec4f(DS_ORIGIN+p*DS_H,1.0),vec4f(normalize(n),0.0));
}
@compute @workgroup_size(64) fn detailSurface(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let g=vec3i(wid);
  if(lid==0u){
    atomicStore(&wgCount,0u);
    var live=0u;
    for(var k=0;k<8;k++){
      let b=clamp(g-vec3i(1)+vec3i(k&1,(k>>1)&1,k>>2),vec3i(0),DS_BRICKS-vec3i(1));
      live|=shown[u32(b.x+DS_BRICKS.x*(b.y+DS_BRICKS.y*b.z))];
    }
    wgLive=live;
  }
  if(workgroupUniformLoad(&wgLive)==0u){return;}
  var count=0u;
  for(var j=0u;j<8u;j++){
    let i=lid+64u*j;let c=8*g-vec3i(1)+vec3i(i32(i&7u),i32((i>>3u)&7u),i32(i>>6u));
    if(!dsCube(c)){continue;}
    var value:array<f32,8>;count+=mcIndexCount(dsCase(c,&value));
  }
  let offset=atomicAdd(&wgCount,count);
  workgroupBarrier();
  if(lid==0u){
    let total=atomicLoad(&wgCount);var base=0xffffffffu;
    if(total>0u){
      // The reservations past the pool are dropped whole: the drawn count takes the ones that fit, a prefix.
      let r=atomicAdd(&args[4],total);
      if(r+total<=capacity){base=r;atomicAdd(&args[0],total);}else{atomicOr(&args[5],1u);}
    }
    wgBase=base;
  }
  let base=workgroupUniformLoad(&wgBase);
  if(base==0xffffffffu){return;}
  var at=base+offset;
  for(var j=0u;j<8u;j++){
    let i=lid+64u*j;let c=8*g-vec3i(1)+vec3i(i32(i&7u),i32((i>>3u)&7u),i32(i>>6u));
    if(!dsCube(c)){continue;}
    var value:array<f32,8>;let cubeCase=dsCase(c,&value);let n=mcIndexCount(cubeCase);
    for(var k=0u;k<n;k+=3u){
      let va=dsVertex(c,mcEdge(cubeCase,k),&value);let vb=dsVertex(c,mcEdge(cubeCase,k+1u),&value);let vc=dsVertex(c,mcEdge(cubeCase,k+2u),&value);
      // Outward: the triangle turns to the normals' side (φ grows out of the liquid).
      let geometric=cross(vb.position.xyz-va.position.xyz,vc.position.xyz-va.position.xyz);
      vertices[at]=va;
      if(dot(geometric,va.normal.xyz+vb.normal.xyz+vc.normal.xyz)>=0.0){vertices[at+1u]=vb;vertices[at+2u]=vc;}else{vertices[at+1u]=vc;vertices[at+2u]=vb;}
      at+=3u;
    }
  }
}`;
}

/** The fine surface of one detail engine: its pool, draw arguments and the meshing pass. */
export class BrickDetailSurface {
  private constructor(
    readonly vertices: GPUBuffer,
    readonly args: GPUBuffer,
    readonly capacity: number,
    private readonly pipeline: GPUComputePipeline,
    private readonly group: GPUBindGroup,
    private readonly groups: readonly [number, number, number],
    /** The box in the parent's world (m). */
    readonly box_m: readonly [readonly [number, number, number], readonly [number, number, number]],
  ) {}

  static async create(device: GPUDevice, presentation: BrickPresentation, o: BrickDetailSurfaceOptions, box_m: BrickDetailSurface["box_m"]): Promise<BrickDetailSurface> {
    const capacity = brickDetailSurfaceCapacity(o.cells, Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize));
    const module = device.createShaderModule({ label: "nb-brick.detailSurface", code: brickDetailSurfaceWGSL(o) });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(`Detail surface WGSL: ${errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("; ")}`);
    const B = GPUShaderStage.COMPUTE;
    const layout = device.createBindGroupLayout({ label: "nb-brick.detailSurface", entries: [
      { binding: 0, visibility: B, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: B, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: B, buffer: { type: "storage" } },
      { binding: 3, visibility: B, buffer: { type: "storage" } }] });
    const pipeline = await device.createComputePipelineAsync({ label: "nb-brick.detailSurface", layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint: "detailSurface", constants: { capacity } } });
    const vertices = device.createBuffer({ label: "nb-brick.detailSurface.vertices", size: VERTEX_BYTES * capacity, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const args = device.createBuffer({ label: "nb-brick.detailSurface.args", size: ARGS_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(args, 0, new Uint32Array([0, 1, 0, 0, 0, 0, 0, 0]));
    const group = device.createBindGroup({ label: "nb-brick.detailSurface", layout, entries: [
      { binding: 0, resource: presentation.vertexPhi.createView() }, { binding: 1, resource: { buffer: presentation.shown } },
      { binding: 2, resource: { buffer: args } }, { binding: 3, resource: { buffer: vertices } }] });
    const groups = o.cells.map(n => Math.ceil((n + 2) / 8)) as unknown as [number, number, number];
    return new BrickDetailSurface(vertices, args, capacity, pipeline, group, groups, box_m);
  }

  /** The water's draw: the pool and its indirect arguments. */
  get geometry() { return { vertices: this.vertices, indirect: this.args, indirectOffset: 0 }; }

  /** Mesh the presentation's current φ (encode after the presentation's copy). */
  encode(encoder: GPUCommandEncoder) {
    encoder.clearBuffer(this.args, 0, 4);
    encoder.clearBuffer(this.args, 16, 8);
    const pass = encoder.beginComputePass({ label: "nb-brick.detailSurface" });
    pass.setPipeline(this.pipeline); pass.setBindGroup(0, this.group);
    pass.dispatchWorkgroups(...this.groups);
    pass.end();
  }

  destroy() { this.vertices.destroy(); this.args.destroy(); }
}
