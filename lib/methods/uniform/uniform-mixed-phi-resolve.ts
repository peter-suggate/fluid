import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { UNIFORM_MIXED_COUNTED, uniformMixedCountedEntriesWGSL, uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedVertexSamplingWGSL } from "./uniform-mixed-vertex-sampling.wgsl";
import { uniformMixedChangedTilesWGSL } from "./uniform-mixed-layout-builder";

/** Writes umVertexValue into every unstored texel of the tiles with a mixed
 * stencil, in place. Readers compiled with the resolved vertex sampler then
 * interpolate eight fine texels at seams instead of searching authorities per
 * tap; uniform tiles keep their direct owner-strided loads. Reconstruction reads only stored texels and the pass writes only
 * unstored ones, so one field serves as input and output. Run after every
 * write a resolved reader follows: canonical writers leave hanging texels stale.
 * One workgroup per seam tile (the ownership's mixed-stencil lists, support
 * 7n+16): on the ungraded h/4h layout a hanging texel's authority is the
 * lowest-index incident 4h tile, so the eight incident tile words and the 27
 * 4h lattice values around the tile are staged once per group and each texel
 * is umVertexFrom4 over workgroup memory (same weights and D4 summation).
 * A GPU-counted launch (UNIFORM_MIXED_COUNTED.fused): the seam count sizes
 * nothing on the host, so a pass without seam tiles still pays one launch.
 * Changed (after a GPU adoption, on a field resolved for the ownership it
 * replaced): a hanging texel's value and whether it hangs depend only on the
 * tile words of its tile's 3x3x3 neighbourhood and on 4h lattice vertices,
 * which are stored under every layout; so only the seam tiles of the adopted
 * generation's dilated changed list can need a write, and the listed launch
 * resolves exactly those. */
export class UniformMixedPhiResolve {
  readonly allocatedBytes = 0;
  private pipeline?: GPUComputePipeline;
  private listed?: GPUComputePipeline;
  private readonly resources: GPUBindGroupLayout;
  private readonly listResources: GPUBindGroupLayout;
  private listGroup?: { readonly changes: GPUBuffer; readonly group: GPUBindGroup };

  constructor(private readonly device: GPUDevice, readonly ownership: UniformMixedOwnership) {
    this.resources = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "read-write", format: "r32float", viewDimension: "3d" } },
    ] });
    this.listResources = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }] });
  }

  bind(field: GPUTexture): GPUBindGroup {
    const size = this.ownership.capacity.lattice.dimensions.map(n => n + 1);
    if (field.format !== "r32float" || [field.width, field.height, field.depthOrArrayLayers].some((n, a) => n !== size[a]))
      throw new Error("Mixed phi resolve requires a native vertex phi field");
    return this.device.createBindGroup({ layout: this.resources, entries: [{ binding: 0, resource: field.createView() }] });
  }

  async initialize(): Promise<void> {
    const header = /* wgsl */ `
 let header=7u*UM_TILES+16u;let valid=job<umSupport[header]+umSupport[header+1u];
 let tile=select(0u,umSupport[header+4u+select(0u,job,valid)],valid);let base=vec3i(umTileCoord(tile));`;
    // A listed tile is resolved only if it is a seam tile of the adopted layout.
    const listed = /* wgsl */ `
 let listed=job<umDilatedCount();let candidate=select(0u,umDilatedTile(select(0u,job,listed)),listed);
 let valid=listed&&umTileMaximumWidth(candidate)!=umTileMinimumWidth(candidate);
 let tile=select(0u,candidate,valid);let base=vec3i(umTileCoord(tile));`;
    const source = (entry: string, job: string, extra = "") => uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity, 0) + extra + /* wgsl */ `
@group(1) @binding(0) var field:texture_storage_3d<r32float,read_write>;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(field,vec3i(p)).x;}
${uniformMixedVertexSamplingWGSL}
var<workgroup> umResolveWords:array<u32,8>;var<workgroup> umResolveLattice:array<f32,27>;
// One lane per tile-local vertex; local 4 belongs to this tile only on the
// upper domain face. Incident tile T+c-1 (c in {0,1}^3) may take c=0 only on
// axes where local is 0. Only 4h tiles own hanging vertices; a missing tile stages as a unit word. Lattice slot m holds the 4h
// vertex (T+m-1)*4, read only for existing owners; aligned vertices are never
// written here, so staging them races no store.
@compute @workgroup_size(125) fn ${entry}(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 // umSupport is read_write storage, so the job test is not uniform to Tint:
 // every lane reaches the barrier and a spare job leaves after it.
 let job=group.x+umDispatchX*group.y;
${job}
 // c=0 (word 0) would serve only local 0 on every axis, a 4-aligned texel
 // this pass never writes: words 1..7 are staged and scanned, and lattice
 // slot 0 (reached only from c=0) is not staged.
 if(valid&&lane>=1u&&lane<8u){let t=base+vec3i(umCorner(lane,2u))-vec3i(1);umResolveWords[lane]=select(0x80000000u,umTopology[umTileAt(vec3u(max(t,vec3i(0))))],all(t>=vec3i(0)));}
 else if(valid&&lane>=9u&&lane<35u){let v=(base+vec3i(umCorner(lane-8u,3u))-vec3i(1))*4;if(all(v>=vec3i(0))){umResolveLattice[lane-8u]=umLoadVertex(vec3u(v));}}
 workgroupBarrier();
 if(!valid){return;}
 let local=umCorner(lane,5u);let p=vec3u(base)*4u+local;
 if(any((local==vec3u(4u))&(p!=UM_D))||all(p%4u==vec3u(0))){return;}
 var best=0xffffffffu;var at=vec3u(0);
 for(var k=1u;k<8u;k++){
  let c=umCorner(k,2u);let word=umResolveWords[k];
  if(any((c==vec3u(0))&(local!=vec3u(0)))||(word&0xc0000000u)!=0u){continue;}
  if((word&0x3fffffffu)<best){best=word&0x3fffffffu;at=c;}
 }
 if(best==0xffffffffu){return;}
 let t=vec3f(local+(vec3u(1)-at)*4u)/4.0;var values:array<f32,8>;
 for(var k=0u;k<8u;k++){
  let corner=umCorner(k,2u);let weight=umVertexWeight(t,corner);let m=at+corner;
  if(weight>0.0){values[k]=weight*umResolveLattice[m.x+3u*(m.y+3u*m.z)];}
 }
 let value=umVertexSum8(values);
 if(bitcast<u32>(value)!=bitcast<u32>(umLoadVertex(p))){textureStore(field,vec3i(p),vec4f(value));}
}`, [entry], entry === "resolve" ? undefined : "umDilatedCount()");
    const module = this.device.createShaderModule({ code: source("resolve", header) });
    const listModule = this.device.createShaderModule({ code: source("resolveListed", listed, uniformMixedChangedTilesWGSL(this.ownership.capacity.tiles, 2, 0)) });
    const errors = [...(await module.getCompilationInfo()).messages, ...(await listModule.getCompilationInfo()).messages].filter(m => m.type === "error");
    if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join("\n"));
    [this.pipeline, this.listed] = await Promise.all([
      this.device.createComputePipelineAsync({
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.ownership.bindLayout, this.resources] }),
        compute: { module, entryPoint: "resolve", constants: { umDispatchX: this.ownership.dispatchX, umCountedJobs: UNIFORM_MIXED_COUNTED.fused } },
      }),
      this.device.createComputePipelineAsync({
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.ownership.bindLayout, this.resources, this.listResources] }),
        compute: { module: listModule, entryPoint: "resolveListed", constants: { umDispatchX: this.ownership.dispatchX } },
      }),
    ]);
  }

  /** changes: the changed tiles (UniformMixedGenerationBuffers.changes) of
   * the generation the ownership just adopted, when the field was resolved
   * for the ownership it replaced (see the class comment). */
  encode(encoder: GPUCommandEncoder, group: GPUBindGroup, changes?: GPUBuffer): void {
    if (!this.pipeline || !this.listed) throw new Error("Mixed phi resolve is not initialized");
    const pass = encoder.beginComputePass({ label: "Uniform mixed phi resolve" });
    pass.setBindGroup(0, this.ownership.bindGroup); pass.setBindGroup(1, group);
    if (changes) {
      if (this.listGroup?.changes !== changes) this.listGroup = { changes, group: this.device.createBindGroup({ layout: this.listResources, entries: [{ binding: 0, resource: { buffer: changes } }] }) };
      pass.setBindGroup(2, this.listGroup.group);
      this.ownership.dispatchCounted(pass, this.listed);
    } else this.ownership.dispatchFusedCounted(pass, this.pipeline);
    pass.end();
  }
}
