/**
 * The brick engine's interim presentation (design §2.10, coordinator ruling
 * 7.1(3)): one copy a frame from the pool into the dense textures the
 * renderer contours today, until its brick path lands (M6).
 *
 * - `vertexPhi` (r32float, lattice + 1 an axis, metres): phi at vertices.
 *   Away from live bricks it holds the scene's far value.
 * - `volume` (r32float, lattice): each cell's liquid fraction, the
 *   six-tetrahedron fill of its eight corner phi (level-set-fill.wgsl.ts).
 * - `openFraction` (r32float, lattice): 1, as M2 has no cut cells.
 *
 * Its work follows the pool, never the lattice. Each pass is one workgroup a
 * pool slot:
 * - `presentClear` writes the far value over every brick a slot showed last
 *   time that it no longer holds;
 * - `presentBricks` writes every live brick and records what each slot shows.
 * The textures are allocated once, at the lattice's size: that is the
 * interim cost the ruling allows. The copy is encoded outside the step's
 * submissions, and the adapter reports its time apart from the step's.
 *
 * Presentation reads F bricks only. When 4h tiles land, a cell of a
 * 4h-owned tile reads the C brick (the tier transfer's interpolant).
 */
import { BrickLayout, NONE, PHI_AIR, brickAbiWGSL } from "./brick-abi";
import type { BrickEngine } from "./brick-engine";
import { levelSetFillWGSL } from "../../core/level-set-fill.wgsl";

const WGSL = (L: BrickLayout) => /* wgsl */`
${brickAbiWGSL(L, { set: { name: "nbSet", atomic: false }, dir: { name: "dir", atomic: false } })}
struct PresentParams { far_m:f32, h_m:f32, pad0:f32, pad1:f32 }
@group(0) @binding(0) var<uniform> params:PresentParams;
@group(0) @binding(1) var<storage,read> nbSet:array<u32>;
@group(0) @binding(2) var<storage,read> dir:array<u32>;
@group(0) @binding(3) var<storage,read> phi:array<u32>;
@group(0) @binding(4) var<storage,read_write> shown:array<u32>;
@group(0) @binding(5) var vertexPhi:texture_storage_3d<r32float,write>;
@group(0) @binding(6) var volume:texture_storage_3d<r32float,write>;
${levelSetFillWGSL}
fn liveKey(slot:u32)->u32{return nbFRec(slot,NB_REC_KEY);}
// Vertex phi (h) through the directory: PHI_AIR where no F brick holds it.
fn vertexAt(v:vec3i)->f32{
  if(any(v<vec3i(0))||any(v>NB_LATTICE)){return NB_PHI_AIR;}
  let s=nbFSlotOfBrick(v>>vec3u(3u));
  if(s>=NB_SEAM){return NB_PHI_AIR;}
  return bitcast<f32>(phi[NB_PHI_F_PHI+NB_STRIDE*s+nbLocal(v)]);
}
fn writeBrick(o:vec3i,lid:u32,live:bool){
  for(var j=0u;j<8u;j++){
    let l=lid+64u*j;let v=o+nbLocalCoord(l);
    if(all(v<=NB_LATTICE)){textureStore(vertexPhi,v,vec4f(select(params.far_m,vertexAt(v)*params.h_m,live),0.0,0.0,0.0));}
    if(nbInLattice(v)){
      var f=0.0;
      if(live){
        var c:array<f32,8>;var lo=1e30;var hi=-1e30;
        for(var k=0u;k<8u;k++){c[k]=vertexAt(v+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));lo=min(lo,c[k]);hi=max(hi,c[k]);}
        if(hi<0.0){f=1.0;}else if(lo<0.0){f=fill(c);}
      }
      textureStore(volume,v,vec4f(f,0.0,0.0,0.0));
    }
  }
}
@compute @workgroup_size(64) fn presentClear(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let slot=wid.x;let was=shown[slot];
  if(was==NB_NONE||was==liveKey(slot)){return;}
  writeBrick(8*nbKeyCoord(was),lid,false);
}
@compute @workgroup_size(64) fn presentBricks(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let slot=wid.x;let key=liveKey(slot);
  if(key!=NB_NONE){writeBrick(8*nbKeyCoord(key),lid,true);}
  workgroupBarrier();
  if(lid==0u){shown[slot]=key;}
}`;

export class BrickPresentation {
  readonly vertexPhi: GPUTexture;
  readonly volume: GPUTexture;
  readonly openFraction: GPUTexture;
  private readonly shown: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly group: GPUBindGroup;
  private readonly clear: GPUComputePipeline;
  private readonly bricks: GPUComputePipeline;
  /** Bytes of the textures and buffers (the interim cost). */
  readonly bytes: number;

  private constructor(private readonly engine: BrickEngine, far_m: number, pipelines: { layout: GPUBindGroupLayout; clear: GPUComputePipeline; bricks: GPUComputePipeline }) {
    const d = engine.device, n = engine.layout.lattice, U = GPUTextureUsage;
    const tex = (label: string, size: readonly number[]) => d.createTexture({ label: `nb-brick.present.${label}`, dimension: "3d", format: "r32float", size: [...size], usage: U.TEXTURE_BINDING | U.STORAGE_BINDING | U.COPY_DST | U.COPY_SRC });
    this.vertexPhi = tex("vertexPhi", n.map(v => v + 1)); this.volume = tex("volume", n); this.openFraction = tex("openFraction", n);
    // Once: far phi everywhere, the open fraction 1 (slab by slab, one slab's staging).
    const fillSlabs = (t: GPUTexture, size: readonly number[], value: number) => {
      const slab = new Float32Array(size[0]! * size[1]!).fill(value);
      for (let z = 0; z < size[2]!; z++) d.queue.writeTexture({ texture: t, origin: [0, 0, z] }, slab, { bytesPerRow: 4 * size[0]!, rowsPerImage: size[1]! }, [size[0]!, size[1]!, 1]);
    };
    fillSlabs(this.vertexPhi, n.map(v => v + 1), far_m); fillSlabs(this.openFraction, n, 1);
    const f = engine.layout.cap.f;
    this.shown = d.createBuffer({ label: "nb-brick.present.shown", size: 4 * f, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.shown, 0, new Uint32Array(f).fill(NONE));
    this.params = d.createBuffer({ label: "nb-brick.present.params", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.params, 0, new Float32Array([far_m, engine.options.h_m, 0, 0]));
    this.clear = pipelines.clear; this.bricks = pipelines.bricks;
    this.group = d.createBindGroup({ label: "nb-brick.present", layout: pipelines.layout, entries: [
      { binding: 0, resource: { buffer: this.params } }, { binding: 1, resource: { buffer: engine.buffer("set") } },
      { binding: 2, resource: { buffer: engine.buffer("dir") } }, { binding: 3, resource: { buffer: engine.buffer("phi") } },
      { binding: 4, resource: { buffer: this.shown } }, { binding: 5, resource: this.vertexPhi.createView() }, { binding: 6, resource: this.volume.createView() }] });
    const texels = (s: readonly number[]) => s.reduce((a, b) => a * b, 1);
    this.bytes = 4 * (texels(n.map(v => v + 1)) + 2 * texels(n) + f) + 16;
  }

  static async create(engine: BrickEngine, far_m: number): Promise<BrickPresentation> {
    if (!(far_m >= PHI_AIR * engine.options.h_m)) throw new Error(`Brick presentation: far phi ${far_m} m must be at least the band's air value`);
    const d = engine.device, module = d.createShaderModule({ label: "nb-brick.present", code: WGSL(engine.layout) });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(`Brick presentation WGSL: ${errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("; ")}`);
    const B = GPUShaderStage.COMPUTE;
    const layout = d.createBindGroupLayout({ label: "nb-brick.present", entries: [
      { binding: 0, visibility: B, buffer: { type: "uniform" } },
      ...[1, 2, 3].map(binding => ({ binding, visibility: B, buffer: { type: "read-only-storage" as const } })),
      { binding: 4, visibility: B, buffer: { type: "storage" } },
      ...[5, 6].map(binding => ({ binding, visibility: B, storageTexture: { access: "write-only" as const, format: "r32float" as const, viewDimension: "3d" as const } }))] });
    const pl = d.createPipelineLayout({ bindGroupLayouts: [layout] });
    const [clear, bricks] = await Promise.all(["presentClear", "presentBricks"].map(entryPoint => d.createComputePipelineAsync({ label: `nb-brick.${entryPoint}`, layout: pl, compute: { module, entryPoint } })));
    return new BrickPresentation(engine, far_m, { layout, clear: clear!, bricks: bricks! });
  }

  /** Encode the copy: a workgroup a pool slot, clear then write. */
  encode(encoder: GPUCommandEncoder) {
    const f = this.engine.layout.cap.f, pass = encoder.beginComputePass({ label: "nb-brick.present" });
    pass.setBindGroup(0, this.group);
    pass.setPipeline(this.clear); pass.dispatchWorkgroups(f);
    pass.setPipeline(this.bricks); pass.dispatchWorkgroups(f);
    pass.end();
  }

  destroy() { this.vertexPhi.destroy(); this.volume.destroy(); this.openFraction.destroy(); this.shown.destroy(); this.params.destroy(); }
}
