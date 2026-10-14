/** An open world's ground on the GPU: the terrain columns of the whole window,
 * and the census's solid tiles past the box.
 *
 * A closed scene's terrain is the box's columns, baked on the host with the
 * initial liquid. An open world's ground runs on past the box, wherever its
 * liquid goes, so its column texture is the window's (2048 x 2048 columns,
 * indexed by lattice column like the box's: umSolidTerrainHeight), baked
 * here once at scene build from the analytic description (`baseHeight_m` and
 * up to MAX_TERRAIN_FEATURES features; terrain.ts's `describedTerrainHeightAt`
 * in f32). Sculpted and generated grounds are refused in an open world.
 *
 * The host's solid promotion (uniformMixedSolidTiles) scans the box alone. A
 * tile past it that the ground cuts is found on the GPU: the solid library
 * builds the cut map word of every tile it stores, a claimed block's
 * included, and `encodeTiles` ors the stored tiles whose word is set into the
 * census's solid tiles (UniformMixedDynamicClassifier.setBodies), so liquid
 * reaching one holds it at h as the host's plane does inside the box.
 *
 * Nothing here is compiled or allocated after initialize, and nothing is read
 * back. */
import type { TerrainDescription } from "../../core/terrain";
import { MAX_TERRAIN_FEATURES, TERRAIN_DEFAULT_FLAT, TERRAIN_UNION_EXPONENT } from "../../core/terrain";

/** The window's columns on each axis: FLUID_WINDOW_PAGES pages of LATTICE_PAGE_CELLS. */
export const UNIFORM_OPEN_WORLD_TERRAIN_COLUMNS = 2048;
export const UNIFORM_OPEN_WORLD_TERRAIN_PARAMS_BYTES = 32 + 32 * MAX_TERRAIN_FEATURES;
const PARAMS_BYTES = UNIFORM_OPEN_WORLD_TERRAIN_PARAMS_BYTES;
export function uniformOpenWorldTerrainSampleWGSL(params:string):string{return /* wgsl */ `// featureWeight: a smoothstep from the flat plateau to the elliptical rim.
fn owFeatureWeight(i:u32,x:f32,z:f32)->f32{
 let shape=${params}.features[2u*i];let kind=${params}.features[2u*i+1u];
 let c=cos(kind.y);let s=sin(kind.y);let dx=x-shape.x;let dz=z-shape.y;
 let d=length(vec2f((c*dx+s*dz)/shape.z,(-s*dx+c*dz)/shape.w));
 if(d<=kind.z){return 1.0;}
 if(d>=1.0){return 0.0;}
 let t=1.0-(d-kind.z)/(1.0-kind.z);return t*t*(3.0-2.0*t);
}
fn owHeight(x:f32,z:f32)->f32{
 var mounds=0.0;var carve=0.0;
 for(var i=0u;i<u32(${params}.ground.z);i++){
  let w=owFeatureWeight(i,x,z);let kind=${params}.features[2u*i+1u];
  if(kind.w>0.5){mounds+=kind.x*w;}else{let a=kind.x*w;let a2=a*a;let a4=a2*a2;carve+=a4*a4;}
 }
 let cut=select(0.0,pow(carve,1.0/8.0),carve>0.0);
 return max(0.0,${params}.ground.y+mounds-cut)/${params}.ground.x;
}
`;}

function bakeWGSL(): string {
  if (TERRAIN_UNION_EXPONENT !== 8) throw new Error("The open world's terrain bake takes the basin union's eighth power as three squarings");
  return /* wgsl */ `
struct Params{frame:vec4f,ground:vec4f,features:array<vec4f,${2 * MAX_TERRAIN_FEATURES}>}
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var columns:texture_storage_2d<r32float,write>;
${uniformOpenWorldTerrainSampleWGSL("params")}
// One column a lane: its height in cells at its centre, mounds summed and
// basins merged by their p-norm.
@compute @workgroup_size(8,8) fn bake(@builtin(global_invocation_id) g:vec3u){
 if(any(g.xy>=textureDimensions(columns))){return;}
 let x=params.frame.x+(f32(g.x)+0.5)*params.frame.z;let z=params.frame.y+(f32(g.y)+0.5)*params.frame.w;
 textureStore(columns,vec2i(g.xy),vec4f(owHeight(x,z),0.0,0.0,0.0));
}`;
}

function markWGSL(cutMapWord: number, tileKeys: number): string {
  return /* wgsl */ `
@group(0) @binding(0) var<storage,read> scratch:array<u32>;
@group(0) @binding(1) var<storage,read_write> tiles:array<u32>;
// One word of the census's solid tiles a lane: the keys whose cut map word is
// set, ored into what the bodies marked before it.
@compute @workgroup_size(64) fn mark(@builtin(global_invocation_id) g:vec3u){
 let word=g.x;if(word*32u>=${tileKeys}u){return;}
 var bits=0u;
 for(var i=0u;i<32u;i++){let t=word*32u+i;if(t<${tileKeys}u&&scratch[${cutMapWord}u+t]!=0u){bits|=1u<<i;}}
 if(bits!=0u){tiles[word]|=bits;}
}`;
}

async function compile(device: GPUDevice, label: string, code: string): Promise<GPUShaderModule> {
  const module = device.createShaderModule({ label, code });
  const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
  if (errors.length) throw new Error(`${label}\n${errors.map(m => `${m.lineNum}: ${m.message}`).join("\n")}`);
  return module;
}

export interface UniformOpenWorldTerrainOptions {
  /** The window's column texture (r32float, STORAGE_BINDING, UNIFORM_OPEN_WORLD_TERRAIN_COLUMNS square). */
  readonly columns: GPUTexture;
  /** The solid mask image whose tile cut map `encodeTiles` reads, and the map's first word and keys. */
  readonly scratch: GPUBuffer;
  readonly cutMapWord: number;
  readonly tileKeys: number;
}

export class UniformOpenWorldTerrain {
  private readonly params: GPUBuffer;
  /** Runtime analytic sampling outside the initial cached window. */
  get sampleParams():GPUBuffer{return this.params;}
  private bakePipeline?: GPUComputePipeline;
  private markPipeline?: GPUComputePipeline;
  private bakeGroup?: GPUBindGroup;
  private markGroup?: GPUBindGroup;
  private markTarget?: GPUBuffer;
  private readonly words: number;
  /** Bytes this owns: the column texture and its params. */
  readonly allocatedBytes: number;
  private static readonly PASS: GPUComputePassDescriptor = { label: "Uniform open-world terrain" };

  constructor(private readonly device: GPUDevice, private readonly options: UniformOpenWorldTerrainOptions) {
    const n = UNIFORM_OPEN_WORLD_TERRAIN_COLUMNS, c = options.columns;
    if (c.width !== n || c.height !== n || c.format !== "r32float" || !(c.usage & GPUTextureUsage.STORAGE_BINDING))
      throw new Error(`Uniform open-world terrain: ${c.label} must be a ${n}x${n} r32float storage texture`);
    if (!Number.isSafeInteger(options.tileKeys) || options.tileKeys < 1 || 4 * (options.cutMapWord + options.tileKeys) > options.scratch.size)
      throw new RangeError(`Uniform open-world terrain: ${options.scratch.label} holds no cut map of ${options.tileKeys} keys at word ${options.cutMapWord}`);
    this.words = Math.ceil(options.tileKeys / 32);
    this.params = device.createBuffer({ label: "Uniform open-world terrain params", size: PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.allocatedBytes = 4 * n * n + PARAMS_BYTES;
  }

  /** Compile both pipelines: at scene build, never during simulation. */
  async initialize(): Promise<void> {
    const device = this.device, o = this.options;
    const [bakeModule, markModule] = await Promise.all([
      compile(device, "Uniform open-world terrain bake", bakeWGSL()),
      compile(device, "Uniform open-world terrain solid tiles", markWGSL(o.cutMapWord, o.tileKeys))]);
    [this.bakePipeline, this.markPipeline] = await Promise.all([
      device.createComputePipelineAsync({ label: "Uniform open-world terrain bake", layout: "auto", compute: { module: bakeModule, entryPoint: "bake" } }),
      device.createComputePipelineAsync({ label: "Uniform open-world terrain solid tiles", layout: "auto", compute: { module: markModule, entryPoint: "mark" } })]);
    this.bakeGroup = device.createBindGroup({ label: "Uniform open-world terrain bake", layout: this.bakePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.params } }, { binding: 1, resource: o.columns.createView() }] });
  }

  /** Bake the window's columns, in cells of `cellHeight_m`, at the centres of
   * columns from `origin_m` (the world x and z of lattice column 0's low
   * corner) `cell_m` apart. */
  encodeBake(encoder: GPUCommandEncoder, terrain: TerrainDescription, origin_m: readonly [number, number], cell_m: readonly [number, number], cellHeight_m: number): void {
    if (!this.bakePipeline || !this.bakeGroup) throw new Error("Uniform open-world terrain bakes once initialized");
    if (terrain.grid || terrain.procedural) throw new Error("An open world's terrain is analytic: baseHeight_m and features");
    if (terrain.features.length > MAX_TERRAIN_FEATURES) throw new Error(`An open world's terrain holds at most ${MAX_TERRAIN_FEATURES} features`);
    const p = new Float32Array(PARAMS_BYTES / 4);
    p.set([origin_m[0], origin_m[1], cell_m[0], cell_m[1], cellHeight_m, terrain.baseHeight_m, terrain.features.length, 1]);
    terrain.features.forEach((f, i) => p.set([f.center_m.x, f.center_m.z, f.radius_m.x, f.radius_m.z,
      f.amount_m, f.rotation_rad ?? 0, f.flat ?? TERRAIN_DEFAULT_FLAT, f.kind === "mound" ? 1 : 0], 8 + 8 * i));
    this.device.queue.writeBuffer(this.params, 0, p);
    const pass = encoder.beginComputePass(UniformOpenWorldTerrain.PASS);
    pass.setPipeline(this.bakePipeline); pass.setBindGroup(0, this.bakeGroup);
    pass.dispatchWorkgroups(UNIFORM_OPEN_WORLD_TERRAIN_COLUMNS / 8, UNIFORM_OPEN_WORLD_TERRAIN_COLUMNS / 8); pass.end();
  }

  /** Or the stored tiles the ground cuts into the census's solid tiles: a
   * setBodies marker, after the bodies' own. */
  encodeTiles(encoder: GPUCommandEncoder, tiles: GPUBuffer): void {
    if (!this.markPipeline) throw new Error("Uniform open-world terrain marks tiles once initialized");
    if (this.markTarget !== tiles) {
      if (this.markTarget) throw new Error("Uniform open-world terrain marks one census's solid tiles");
      this.markTarget = tiles;
      this.markGroup = this.device.createBindGroup({ label: "Uniform open-world terrain solid tiles", layout: this.markPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: this.options.scratch } }, { binding: 1, resource: { buffer: tiles, offset: 0, size: 4 * this.words } }] });
    }
    const pass = encoder.beginComputePass(UniformOpenWorldTerrain.PASS);
    pass.setPipeline(this.markPipeline); pass.setBindGroup(0, this.markGroup!);
    pass.dispatchWorkgroups(Math.ceil(this.words / 64)); pass.end();
  }

  destroy(): void { this.params.destroy(); }
}
