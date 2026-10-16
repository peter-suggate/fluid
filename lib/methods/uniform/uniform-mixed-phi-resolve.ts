import { uniformCompiledVertexResolveWGSL } from "./uniform-compiled-topology";
import { uniformDetailBindLayout, uniformDetailClass, uniformDetailModule, uniformDetailPipeline, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { UNIFORM_DETAIL_GUARD_LOAD } from "../../core/uniform-detail-abi";
import { UNIFORM_MIXED_COUNTED, uniformMixedCountedEntriesWGSL, uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedVertexSamplingWGSL } from "./uniform-mixed-vertex-sampling.wgsl";
import { uniformMixedChangedTilesWGSL } from "./uniform-mixed-layout-builder";

const RESOLVE_PASS: GPUComputePassDescriptor = { label: "Uniform mixed phi resolve" };

/** Writes umVertexValue into every unstored texel of the tiles with a mixed
 * stencil, in place. Readers compiled with the resolved vertex sampler then
 * interpolate eight fine texels at seams instead of searching authorities per
 * tap; uniform tiles keep their direct owner-strided loads. Reconstruction reads only stored texels and the pass writes only
 * unstored ones, so one field serves as input and output. Run after every
 * write a resolved reader follows: canonical writers leave hanging texels stale.
 * One workgroup per seam tile (the ownership's mixed-stencil lists, support
 * 7n+16): on the ungraded h/4h layout a hanging texel's authority is the
 * lowest-index incident 4h tile, so the compiled incident mask selects its authority and the 27
 * 4h lattice values around the tile are staged once per group; each texel
 * is umVertexFrom4 over workgroup memory (same weights and D4 summation).
 * A GPU-counted launch (UNIFORM_MIXED_COUNTED.fused): the seam count sizes
 * nothing on the host. With no h tiles, the uniform tier count returns
 * before the job-count barrier, avoiding idle barrier work at the reserved
 * relayout grid width. A nonempty h tier still uses the exact live list.
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
  /** resolve behind a gate word (encodeGated): every group returns while it is zero. */
  private gated?: GPUComputePipeline;
  private readonly gateResources: GPUBindGroupLayout;
  private gateGroup?: { readonly source: GPUBuffer; readonly group: UniformDetailGroup };
  private readonly resources: GPUBindGroupLayout;
  private readonly listResources: GPUBindGroupLayout;
  private listGroup?: { readonly changes: GPUBuffer; readonly group: UniformDetailGroup };

  /** gateWord: the u32 index in the gate source (encodeGated) of its word. */
  constructor(private readonly device: GPUDevice, readonly ownership: UniformMixedOwnership, private readonly gateWord = 0) {
    this.resources = uniformDetailBindLayout(device,{ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "read-write", format: "r32float", viewDimension: "3d" } },
    ] });
    this.listResources = uniformDetailBindLayout(device,{ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }] });
    this.gateResources = uniformDetailBindLayout(device,{ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }] });
  }

  bind(field: GPUTexture): UniformDetailGroup {
    if (field.format !== "r32float" || uniformDetailClass(field) !== "vertex")
      throw new Error("Mixed phi resolve requires a native vertex phi field");
    return uniformDetailGroup(this.device,{ layout: this.resources, entries: [{ binding: 0, resource: field }] });
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
    const source = (entry: string, job: string, extra = "", gate = "") => uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity, 0) + extra + /* wgsl */ `
@group(1) @binding(0) var field:texture_storage_3d<r32float,read_write>;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(field,vec3i(p)).x;}
${uniformMixedVertexSamplingWGSL}
var<workgroup> umResolveLattice:array<f32,27>;
// One lane per tile-local vertex; local 4 belongs to this tile only on the
// upper domain face. Incident tile T+c-1 (c in {0,1}^3) may take c=0 only on
// axes where local is 0. Only 4h tiles own hanging vertices. The compiled
// recipe clips missing tiles. Lattice slot m holds the 4h
// vertex (T+m-1)*4, read only for existing owners; aligned vertices are never
// written here, so staging them races no store.
@compute @workgroup_size(125) fn ${entry}(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){${gate}
 // umSupport is read_write storage, so the job test is not uniform to Tint:
 // every lane reaches the barrier and a spare job leaves after it.
 let job=group.x+umDispatchX*group.y;
${job}
 // Incident c=0 serves only the aligned local-zero vertex, never written
 // here, so its lattice slot 0 is not needed.
 if(valid&&lane>=9u&&lane<35u){let v=(base+vec3i(umCorner(lane-8u,3u))-vec3i(1))*4;if(umVertexPresent(v)){umResolveLattice[lane-8u]=umLoadVertex(vec3u(v));}}
 workgroupBarrier();
 if(!valid){return;}
${uniformCompiledVertexResolveWGSL("umResolveLattice","umVertexWeight","umVertexSum8")}
 if(bitcast<u32>(value)!=bitcast<u32>(${UNIFORM_DETAIL_GUARD_LOAD}textureLoad(field,vec3i(p)).x)){textureStore(field,vec3i(p),vec4f(value));}
}`, [entry], entry === "resolve" ? undefined : "umDilatedCount()", "umCounts.x==0u");
    const module = uniformDetailModule(this.device,{ label: "Uniform mixed phi resolve", code: source("resolve", header) });
    const listModule = uniformDetailModule(this.device,{ label: "Uniform mixed phi resolve (listed)", code: source("resolveListed", listed, uniformMixedChangedTilesWGSL(this.ownership.capacity.tiles, 2, 0)) });
    // The gate is a uniform: every group of a launch takes the same branch.
    // Lane 0 reads the gate word; workgroupUniformLoad makes the test uniform.
    const gateModule = uniformDetailModule(this.device,{ label: "Uniform mixed phi resolve (gated)", code: source("resolve", header, `
@group(2) @binding(0) var<storage,read> umResolveGate:array<u32>;
var<workgroup> umResolveOpen:u32;`, `
 if(lane==0u){umResolveOpen=umResolveGate[${this.gateWord}u];}
 if(workgroupUniformLoad(&umResolveOpen)==0u){return;}`) });
    const errors = [...(await module.getCompilationInfo()).messages, ...(await listModule.getCompilationInfo()).messages, ...(await gateModule.getCompilationInfo()).messages].filter(m => m.type === "error");
    if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join("\n"));
    [this.pipeline, this.listed, this.gated] = await Promise.all([
      uniformDetailPipeline(this.device,this.ownership,{
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.ownership.bindLayout, this.resources] }),
        compute: { module, entryPoint: "resolve", constants: { umDispatchX: this.ownership.dispatchX, umCountedJobs: UNIFORM_MIXED_COUNTED.fused } },
      }),
      uniformDetailPipeline(this.device,this.ownership,{
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.ownership.bindLayout, this.resources, this.listResources] }),
        compute: { module: listModule, entryPoint: "resolveListed", constants: { umDispatchX: this.ownership.dispatchX } },
      }),
      uniformDetailPipeline(this.device,this.ownership,{
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.ownership.bindLayout, this.resources, this.gateResources] }),
        compute: { module: gateModule, entryPoint: "resolve", constants: { umDispatchX: this.ownership.dispatchX, umCountedJobs: UNIFORM_MIXED_COUNTED.fused } },
      }),
    ]);
  }

  /** changes: the changed tiles (UniformMixedGenerationBuffers.changes) of
   * the generation the ownership just adopted, when the field was resolved
   * for the ownership it replaced (see the class comment). */
  encode(encoder: GPUCommandEncoder, group: UniformDetailGroup, changes?: GPUBuffer): void {
    // With no h capacity, every sampled vertex comes from the coarse corners;
    // there are no hanging h values to materialize.
    if(this.ownership.capacity.fineTiles===0)return;
    if (!this.pipeline || !this.listed) throw new Error("Mixed phi resolve is not initialized");
    const pass = encoder.beginComputePass(RESOLVE_PASS);
    pass.setBindGroup(0, this.ownership.bindGroup); pass.setBindGroup(1,group.group);
    if (changes) {
      if (this.listGroup?.changes !== changes) this.listGroup = { changes, group: uniformDetailGroup(this.device,{ layout: this.listResources, entries: [{ binding: 0, resource: { buffer: changes } }] }) };
      pass.setBindGroup(2, this.listGroup.group.group);
      this.ownership.dispatchCounted(pass, this.listed);
    } else this.ownership.dispatchFusedCounted(pass, this.pipeline);
    pass.end();
  }

  /** The full resolve, launched only in effect: its groups return while
   * u32 gateWord (constructor) of source is zero, which its writer sets
   * where it changed a 4h-lattice vertex. Hanging values read only those
   * (see the class comment), so a zero word means this resolve would write
   * nothing. The word is read on the GPU: no readback. */
  encodeGated(encoder: GPUCommandEncoder, group: UniformDetailGroup, source: GPUBuffer): void {
    if(this.ownership.capacity.fineTiles===0)return;
    if (!this.gated) throw new Error("Mixed phi resolve is not initialized");
    if (this.gateGroup?.source !== source) this.gateGroup = { source, group: uniformDetailGroup(this.device,{ layout: this.gateResources, entries: [{ binding: 0, resource: { buffer: source } }] }) };
    const pass = encoder.beginComputePass(RESOLVE_PASS);
    pass.setBindGroup(0, this.ownership.bindGroup); pass.setBindGroup(1,group.group); pass.setBindGroup(2,this.gateGroup.group.group);
    this.ownership.dispatchFusedCounted(pass, this.gated);
    pass.end();
  }
}
