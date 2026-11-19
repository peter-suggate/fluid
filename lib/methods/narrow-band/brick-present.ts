/**
 * The brick engine's interim presentation (design §2.10, coordinator ruling
 * 7.1(3)): one copy a frame from the pools into the dense textures the
 * renderer contours today, until its brick path lands (M6).
 *
 * - `vertexPhi` (r32float, lattice + 1 an axis, metres): phi at vertices,
 *   read through the tier authority (brick-owner.wgsl.ts nbVertexPhi): F where
 *   no incident tile is 4h-owned, else the 4h tile's interpolant of its C
 *   corners. Where no page holds a vertex, or the vertex holds air at or past PHI_AIR_C (an absent slot's value,
 *   brick-owner.wgsl.ts, and an entering slot's), it is the scene's far value.
 * - `volume` (r32float, lattice): each cell's liquid fraction, the
 *   six-tetrahedron fill of its eight corner phi (level-set-fill.wgsl.ts); 0 in a
 *   closed F cell (FLAGS.OPEN clear), as the gauge.
 * - `openFraction` (r32float, lattice): 1, as M2 has no cut cells; on a
 *   padded lattice (M3), 0 past the box, whose cells are wall.
 * The textures span the presented cells (`cells`, BrickSolverSetup.presentCells:
 * the container's), not the padded lattice. The renderer's dense surface maps its
 * grid (tankCells) onto the scene's container and reads container / tankCells as
 * the cell size, as the old engine reports (tankCells: its container's cells).
 * Until 2026-11-19 they spanned the lattice, so a padded lattice drew the water
 * squashed toward the floor by container / lattice (reservoir-seep: 256 / 288 in y,
 * the jet 7 cells under its particles). An open top's headroom past the container
 * is not presented (the dense renderer draws the container alone).
 * On a padded lattice a cell past the box holds no liquid (`volume` 0 there),
 * as the old engine's cells behind its shell hold none; its vertices keep
 * their stored air (the vertex row on the box's planes is the wall's).
 *
 * Two passes, one workgroup a lattice brick (8^3 cells):
 * - `presentVertices` writes the brick's vertices, plus the vertex row past the
 *   lattice's upper faces (the extra brick layer);
 * - `presentVolume` fills the brick's cells from those vertices.
 * A brick is held when an F slot holds it or a C page holds its page, and it
 * is written when a brick within one brick is held (shared-face vertices take
 * a neighbour page's authority). A brick neither written now nor shown last
 * frame exits at once (empty workgroups cost ~1 ns), so the work follows the
 * pools; a brick shown last frame and no longer written goes back to far once.
 *
 * Until 2026-11-16 the copy read F bricks only, one workgroup a pool slot. On
 * the tiered dam a 4h-owned region with no F brick then showed as far air
 * (grey holes along the floor), and a 4h-owned tile whose F brick stayed
 * resident (reach, margins) showed its stale F phi (stair banding on the side
 * faces). The textures are allocated once, at the lattice's size: that is the
 * interim cost the ruling allows, and `shown` (a word a lattice brick) is part
 * of it. The copy is encoded outside the step's submissions.
 */
import { brickLowerOn, brickModuleCode } from "./brick-lower";
import { BrickLayout, PHI_AIR, boxPadded, boxWGSL, brickAbiWGSL } from "./brick-abi";
import type { BrickEngine } from "./brick-engine";
import { brickOwnerWGSL } from "./brick-owner.wgsl";
import { levelSetFillWGSL } from "../../core/level-set-fill.wgsl";

const bricksPerAxis = (L: BrickLayout) => L.lattice.map(n => n >> 3) as unknown as readonly [number, number, number];

/** Workgroups an axis: presentVertices covers vertices 0..cells, presentVolume cells 0..cells-1 (one a brick). */
const coverVertices = (cells: readonly number[]) => cells.map(n => Math.ceil((n + 1) / 8));
const coverCells = (cells: readonly number[]) => cells.map(n => Math.ceil(n / 8));

const COMMON = (L: BrickLayout, cells: readonly number[]) => /* wgsl */`
${brickAbiWGSL(L, { set: { name: "nbSet", atomic: false }, dir: { name: "dir", atomic: false } })}
struct PresentParams { far_m:f32, h_m:f32, pad0:f32, pad1:f32 }
const NB_PRESENT_BRICKS:vec3i=vec3i(${bricksPerAxis(L).join(",")});
// The presented cells (the textures' extent: cells, and cells + 1 vertices).
const NB_PRESENT_CELLS:vec3i=vec3i(${cells.join(",")});
@group(0) @binding(0) var<uniform> params:PresentParams;
@group(0) @binding(1) var<storage,read> nbSet:array<u32>;
@group(0) @binding(2) var<storage,read> dir:array<u32>;
@group(0) @binding(3) var<storage,read> phi:array<u32>;
@group(0) @binding(4) var<storage,read_write> shown:array<u32>;
@group(0) @binding(5) var<storage,read> vel:array<f32>;
fn presentIndex(b:vec3i)->u32{let n=NB_PRESENT_BRICKS;return u32(b.x+n.x*(b.y+n.y*b.z));}
// An F slot holds the brick, or a C page holds its page (a 4h-owned region with no F brick reads C).
fn presentHeld(b:vec3i)->bool{if(any(b<vec3i(0))){return false;}return nbFSlotOfBrick(b)<NB_SEAM||nbCSlotOfPage(b>>vec3u(2u))!=NB_NONE;}
// A brick within one brick of a held one: its vertices on shared faces take a neighbour page's authority (a
// vertex's incident tiles lie within one cell), and its cells' upper corners are the next brick's vertices.
fn presentNear(b:vec3i)->bool{
  for(var k=0;k<27;k++){if(presentHeld(b+vec3i(k%3,(k/3)%3,k/9)-vec3i(1))){return true;}}
  return false;
}
`;

const WGSL_VERTICES = (L: BrickLayout, cells: readonly number[]) => COMMON(L, cells) + brickOwnerWGSL({ phi: "phi", vel: "vel", box: boxWGSL(L) }) + /* wgsl */`
@group(0) @binding(6) var vertexPhi:texture_storage_3d<r32float,write>;
var<workgroup> presentState:u32;
@compute @workgroup_size(64) fn presentVertices(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let b=vec3i(wid);let edge=any(b>=NB_PRESENT_BRICKS);
  // The extra layer past the upper faces holds only the vertex row N: always written (a 2D shell of workgroups).
  if(lid==0u){presentState=select(select(0u,1u,presentNear(b))|select(0u,2u,shown[presentIndex(b)]!=0u),1u,edge);}
  let state=workgroupUniformLoad(&presentState);let held=(state&1u)!=0u;
  if(state==0u){return;}
  for(var j=0u;j<8u;j++){
    let v=8*b+nbLocalCoord(lid+64u*j);
    if(any(v>NB_PRESENT_CELLS)){continue;}
    var p=params.far_m;
    if(held){let q=nbVertexPhi(v);if(q<NB_PHI_AIR_C){p=q*params.h_m;}}
    textureStore(vertexPhi,v,vec4f(p,0.0,0.0,0.0));
  }
}`;

/** presentVolume's WGSL (exported for its text test). */
export const WGSL_VOLUME = (L: BrickLayout, cells: readonly number[] = L.lattice) => COMMON(L, cells) + levelSetFillWGSL + /* wgsl */`
@group(0) @binding(6) var vertexPhi:texture_3d<f32>;
@group(0) @binding(7) var volume:texture_storage_3d<r32float,write>;
var<workgroup> presentState:u32;
@compute @workgroup_size(64) fn presentVolume(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let b=vec3i(wid);let i=presentIndex(b);
  // Lane 0 decides for the brick (27 directory reads once) and alone reads and writes shown[i].
  if(lid==0u){presentState=select(0u,1u,presentNear(b))|select(0u,2u,shown[i]!=0u);}
  let state=workgroupUniformLoad(&presentState);let held=(state&1u)!=0u;
  if(state==0u){return;}
  for(var j=0u;j<8u;j++){
    let v=8*b+nbLocalCoord(lid+64u*j);
    if(any(v>=NB_PRESENT_CELLS)){continue;}
    var f=0.0;
    if(${boxPadded(L) ? "held&&nbInBox(v)" : "held"}){
      var c:array<f32,8>;var lo=1e30;var hi=-1e30;
      for(var k=0u;k<8u;k++){c[k]=textureLoad(vertexPhi,v+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)),0).x;lo=min(lo,c[k]);hi=max(hi,c[k]);}
      if(hi<0.0){f=1.0;}else if(lo<0.0){f=fill(c);}
      // A closed F cell (FLAGS.OPEN clear: a static mask voxel, a body) holds no liquid, as the gauge (vGauge).
      let fs=nbFSlotOfBrick(b);
      if(f>0.0&&fs<NB_SEAM&&(phi[NB_PHI_F_FLAGS+NB_STRIDE*fs+nbLocal(v)]&NB_FLAG_OPEN)==0u){f=0.0;}
    }
    textureStore(volume,v,vec4f(f,0.0,0.0,0.0));
  }
  if(lid==0u){shown[i]=select(0u,1u,held);}
}`;

export class BrickPresentation {
  readonly vertexPhi: GPUTexture;
  readonly volume: GPUTexture;
  readonly openFraction: GPUTexture;
  private readonly shown: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly groups: readonly [GPUBindGroup, GPUBindGroup];
  private readonly pipelines: readonly [GPUComputePipeline, GPUComputePipeline];
  /** Bytes of the textures and buffers (the interim cost). */
  readonly bytes: number;

  private constructor(private readonly engine: BrickEngine, far_m: number, private readonly cells: readonly number[], layouts: readonly [GPUBindGroupLayout, GPUBindGroupLayout], pipelines: readonly [GPUComputePipeline, GPUComputePipeline]) {
    const d = engine.device, n = cells, U = GPUTextureUsage;
    const tex = (label: string, size: readonly number[]) => d.createTexture({ label: `nb-brick.present.${label}`, dimension: "3d", format: "r32float", size: [...size], usage: U.TEXTURE_BINDING | U.STORAGE_BINDING | U.COPY_DST | U.COPY_SRC });
    this.vertexPhi = tex("vertexPhi", n.map(v => v + 1)); this.volume = tex("volume", n); this.openFraction = tex("openFraction", n);
    // Once: far phi everywhere, the open fraction 1 (slab by slab, one slab's staging).
    const fillSlabs = (t: GPUTexture, size: readonly number[], value: number) => {
      const slab = new Float32Array(size[0]! * size[1]!).fill(value);
      for (let z = 0; z < size[2]!; z++) d.queue.writeTexture({ texture: t, origin: [0, 0, z] }, slab, { bytesPerRow: 4 * size[0]!, rowsPerImage: size[1]! }, [size[0]!, size[1]!, 1]);
    };
    fillSlabs(this.vertexPhi, n.map(v => v + 1), far_m); fillSlabs(this.openFraction, n, 1);
    // Padded (M3): the cells past the box are wall, closed.
    const box = engine.layout.box;
    if (boxPadded(engine.layout)) for (let z = 0; z < n[2]!; z++) {
      const slab = new Float32Array(n[0]! * n[1]!);
      if (z < box[2]!) for (let y = 0; y < Math.min(box[1]!, n[1]!); y++) slab.fill(1, y * n[0]!, y * n[0]! + Math.min(box[0]!, n[0]!));
      d.queue.writeTexture({ texture: this.openFraction, origin: [0, 0, z] }, slab, { bytesPerRow: 4 * n[0]!, rowsPerImage: n[1]! }, [n[0]!, n[1]!, 1]);
    }
    const bricks = bricksPerAxis(engine.layout).reduce((a, b) => a * b, 1);
    this.shown = d.createBuffer({ label: "nb-brick.present.shown", size: 4 * bricks, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.params = d.createBuffer({ label: "nb-brick.present.params", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.params, 0, new Float32Array([far_m, engine.options.h_m, 0, 0]));
    this.pipelines = pipelines;
    const shared = [
      { binding: 0, resource: { buffer: this.params } }, { binding: 1, resource: { buffer: engine.buffer("set") } },
      { binding: 2, resource: { buffer: engine.buffer("dir") } }, { binding: 3, resource: { buffer: engine.buffer("phi") } },
      { binding: 4, resource: { buffer: this.shown } }, { binding: 5, resource: { buffer: engine.buffer("vel") } }];
    this.groups = [
      d.createBindGroup({ label: "nb-brick.presentVertices", layout: layouts[0], entries: [...shared, { binding: 6, resource: this.vertexPhi.createView() }] }),
      d.createBindGroup({ label: "nb-brick.presentVolume", layout: layouts[1], entries: [...shared, { binding: 6, resource: this.vertexPhi.createView() }, { binding: 7, resource: this.volume.createView() }] })];
    const texels = (s: readonly number[]) => s.reduce((a, b) => a * b, 1);
    this.bytes = 4 * (texels(n.map(v => v + 1)) + 2 * texels(n) + bricks) + 16;
  }

  /** `cells`: the presented cells from the lattice's corner (default the lattice), at most the lattice an axis. */
  static async create(engine: BrickEngine, far_m: number, cells: readonly number[] = engine.layout.lattice): Promise<BrickPresentation> {
    if (!(far_m >= PHI_AIR * engine.options.h_m)) throw new Error(`Brick presentation: far phi ${far_m} m must be at least the band's air value`);
    if (engine.layout.lattice.some(v => v % 8 !== 0)) throw new Error(`Brick presentation: the lattice ${engine.layout.lattice.join("x")} is not whole bricks`);
    if (cells.length !== 3 || cells.some((n, a) => !Number.isInteger(n) || n < 1 || n > engine.layout.lattice[a]!)) throw new Error(`Brick presentation: the presented cells ${cells.join("x")} are not within the lattice ${engine.layout.lattice.join("x")}`);
    const d = engine.device, B = GPUShaderStage.COMPUTE;
    const compile = async (label: string, code: string) => {
      const module = d.createShaderModule({ label, code: brickModuleCode(label, code, brickLowerOn(engine.options.lowerDynamicWrites)) });
      const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
      if (errors.length) throw new Error(`Brick presentation WGSL (${label}): ${errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("; ")}`);
      return module;
    };
    const shared: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: B, buffer: { type: "uniform" } },
      ...[1, 2, 3].map(binding => ({ binding, visibility: B, buffer: { type: "read-only-storage" as const } })),
      { binding: 4, visibility: B, buffer: { type: "storage" } },
      { binding: 5, visibility: B, buffer: { type: "read-only-storage" } }];
    const storage = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: B, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } });
    const layouts = [
      d.createBindGroupLayout({ label: "nb-brick.presentVertices", entries: [...shared, storage(6)] }),
      d.createBindGroupLayout({ label: "nb-brick.presentVolume", entries: [...shared, { binding: 6, visibility: B, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } }, storage(7)] })] as const;
    const [vertices, volume] = await Promise.all([compile("nb-brick.presentVertices", WGSL_VERTICES(engine.layout, cells)), compile("nb-brick.presentVolume", WGSL_VOLUME(engine.layout, cells))]);
    const pipelines = await Promise.all([
      d.createComputePipelineAsync({ label: "nb-brick.presentVertices", layout: d.createPipelineLayout({ bindGroupLayouts: [layouts[0]] }), compute: { module: vertices, entryPoint: "presentVertices" } }),
      d.createComputePipelineAsync({ label: "nb-brick.presentVolume", layout: d.createPipelineLayout({ bindGroupLayouts: [layouts[1]] }), compute: { module: volume, entryPoint: "presentVolume" } })]);
    return new BrickPresentation(engine, far_m, cells, layouts, [pipelines[0]!, pipelines[1]!]);
  }

  /** Encode the copy: the vertices (on the lattice, one extra brick layer for the upper vertex row), then the cells,
   * a workgroup a brick over the presented cells. */
  encode(encoder: GPUCommandEncoder) {
    const [vx, vy, vz] = coverVertices(this.cells), [x, y, z] = coverCells(this.cells), pass = encoder.beginComputePass({ label: "nb-brick.present" });
    pass.setPipeline(this.pipelines[0]); pass.setBindGroup(0, this.groups[0]); pass.dispatchWorkgroups(vx!, vy!, vz!);
    pass.setPipeline(this.pipelines[1]); pass.setBindGroup(0, this.groups[1]); pass.dispatchWorkgroups(x!, y!, z!);
    pass.end();
  }

  destroy() { this.vertexPhi.destroy(); this.volume.destroy(); this.openFraction.destroy(); this.shown.destroy(); this.params.destroy(); }
}
