import { uniformDetailBindLayout, uniformDetailExtent, uniformDetailField, uniformDetailGroup, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, type UniformDetailGroup } from "./uniform-detail-fields";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";

/** Texel extent of the 4h vertex base of a lattice: one texel per tile corner. */
export function uniformCoarseVertexPhiSize(dimensions: readonly number[]): [number, number, number] {
  return [dimensions[0]! / 4 + 1, dimensions[1]! / 4 + 1, dimensions[2]! / 4 + 1];
}

/** The 4h vertex base of the level set: texel g is phi at lattice vertex 4g,
 * a (t+1)^3 r32float on its own (never folded into an atlas), so a consumer
 * outside the solver loads a tile corner with one textureLoad under every
 * field placement and binds no h-sized texture while no tile is at h.
 * Base-currency rule of the vertex class: the base is current for every tile
 * corner at each published revision. A tile corner is a stored vertex under
 * every layout (the solver's own 4h state), so the publish is a selection:
 * one fixed launch over the tile corners, encoded after the last phi write a
 * presentation can follow (the frame's surface-volume apply, a host relayout's
 * remap, t=0). The solver never reads the base, except under the domain
 * placement of the detail storage, which adopts this texture as the vertex
 * field's base block (UniformDetailStorage.adoptBase): while no tile is at h
 * the base IS the solver's phi, written in place by its kernels, and there
 * is nothing to publish. */
export class UniformCoarseVertexPhi {
  readonly texture: GPUTexture;
  readonly allocatedBytes: number;
  private readonly resources: GPUBindGroupLayout;
  private readonly group: UniformDetailGroup;
  private readonly size: [number, number, number];
  private pipeline?: GPUComputePipeline;

  constructor(private readonly device: GPUDevice, private readonly ownership: UniformMixedOwnership, private readonly phi: GPUTexture) {
    const d = ownership.capacity.lattice.dimensions;
    if (d.some(n => n % 4 !== 0)) throw new Error("The 4h vertex base needs a lattice of whole tiles");
    if (phi.format !== "r32float" || uniformDetailExtent(phi).some((n, a) => n !== d[a]! + 1)) throw new Error("The 4h vertex base publishes the h vertex phi field");
    this.size = uniformCoarseVertexPhiSize(d);
    this.texture = device.createTexture({ label: "Uniform Geometric 4h vertex phi base", size: this.size, dimension: "3d", format: "r32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST });
    this.allocatedBytes = 4 * this.size[0] * this.size[1] * this.size[2];
    this.resources = uniformDetailBindLayout(device,{ label: "Uniform 4h vertex phi base publish", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
    ] });
    this.group = uniformDetailGroup(device, { layout: this.resources, entries: [{ binding: 0, resource: phi }, { binding: 1, resource: this.texture }] });
  }

  /** The publish kernel; `coarseOut` is a raw (tile-resolution) name of the
   * detail rewrite, `vertexPhi` a vertex field: packed, the load is the
   * storage's accessor (a tile corner is a stored vertex). */
  static shader(ownership: UniformMixedOwnership): string {
    return uniformMixedTopologyWGSL(ownership.capacity, 0) + /* wgsl */ `
@group(1) @binding(0) var vertexPhi:texture_3d<f32>;
@group(1) @binding(1) var coarseOut:texture_storage_3d<r32float,write>;
// One lane per tile corner g in [0, t]^3.
@compute @workgroup_size(4,4,4) fn publishCoarseVertexPhi(@builtin(global_invocation_id) g:vec3u){
 if(any(g>UM_T)){return;}
 textureStore(coarseOut,vec3i(g),vec4f(textureLoad(vertexPhi,vec3i(4u*g),0).x,0.0,0.0,0.0));
}`;
  }

  async initialize(): Promise<void> {
    const shader = uniformDetailModule(this.device, { label: "Uniform 4h vertex phi base publish", code: UniformCoarseVertexPhi.shader(this.ownership) });
    const errors = (await shader.getCompilationInfo()).messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join("\n"));
    this.pipeline = await uniformDetailPipeline(this.device, this.ownership,{ label: "Uniform 4h vertex phi base publish",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.ownership.bindLayout, this.resources] }),
      compute: { module: shader, entryPoint: "publishCoarseVertexPhi" } });
  }

  /** The texture is the phi field's base block (the domain placement):
   * the field itself while no tile is at h, and beside the h generation the
   * storage keeps every tile corner current in it (each store of a corner
   * writes both), so nothing is published. */
  get adopted(): boolean {
    const storage = uniformDetailField(this.phi)?.storage;
    return !!storage && (storage.baseOf(this.phi) ?? storage.physical(this.phi)) === this.texture;
  }

  /** One launch: ceil((t+1)/4)^3 workgroups; none while the base is the field's own. */
  encode(encoder: GPUCommandEncoder): void {
    if (!this.pipeline) throw new Error("The 4h vertex phi base is not initialized");
    if (this.adopted) return;
    const pass = encoder.beginComputePass({ label: "Uniform publish 4h vertex phi base" });
    pass.setPipeline(uniformDetailPick(this.pipeline)); pass.setBindGroup(0, this.ownership.bindGroup); pass.setBindGroup(1, this.group.group);
    pass.dispatchWorkgroups(Math.ceil(this.size[0] / 4), Math.ceil(this.size[1] / 4), Math.ceil(this.size[2] / 4));
    pass.end();
  }

  destroy(): void { this.texture.destroy(); }
}
