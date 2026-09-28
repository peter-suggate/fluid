import { compileMixedTiers, type UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedVertexSamplingWGSL } from "./uniform-mixed-vertex-sampling.wgsl";

/** Resolution-switch vertex transfer. Restriction visits compact owners and
 * copies only authoritative coincident vertices. Prolongation reconstructs the
 * fine lattice only on leaving mixed mode. Both directions borrow native phi
 * fields; neither creates a field or updates inactive vertices per mixed step. */
export class UniformMixedVertexTransfer {
  readonly allocatedBytes = 0;
  private pipelines?: GPUComputePipeline[];
  private readonly resources: GPUBindGroupLayout;
  private readonly group: GPUBindGroup;

  constructor(private readonly device: GPUDevice, readonly ownership: UniformMixedOwnership,
    input: GPUTexture, output: GPUTexture, private readonly direction: "restrict" | "prolong") {
    const size = ownership.layout.lattice.dimensions.map(n => n + 1);
    for (const texture of [input, output]) {
      if (texture.format !== "r32float" || [texture.width, texture.height, texture.depthOrArrayLayers].some((n, a) => n !== size[a]))
        throw new Error("Mixed vertex transfer requires native vertex phi fields");
    }
    if (input === output) throw new Error("Mixed vertex transfer fields must be disjoint");
    this.resources = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
    ] });
    this.group = device.createBindGroup({ layout: this.resources, entries: [
      { binding: 0, resource: input.createView() }, { binding: 1, resource: output.createView() },
    ] });
  }

  async initialize(): Promise<void> {
    const module = this.device.createShaderModule({ code: uniformMixedTopologyWGSL(this.ownership.layout, 0) + /* wgsl */ `
@group(1) @binding(0) var inputPhi:texture_3d<f32>;
@group(1) @binding(1) var outputPhi:texture_storage_3d<r32float,write>;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(inputPhi,vec3i(p),0).x;}
${uniformMixedVertexSamplingWGSL}
@compute @workgroup_size(64) fn restrictVertices(@builtin(global_invocation_id) gid:vec3u){
 let owner=umOwner(gid);if(owner.width==0u){return;}
 let origin=umOrigin(owner);
 for(var k=0u;k<8u;k++){
  let p=origin+umCorner(k,2u)*owner.width;
  let authority=umVertexAuthority(p);
  // Exactly one writer, including domain walls and shared edges/corners.
  if(authority.index==owner.index){textureStore(outputPhi,vec3i(p),vec4f(umLoadVertex(p)));}
 }
}
@compute @workgroup_size(4,4,4) fn prolongVertices(@builtin(global_invocation_id) id:vec3u){
 if(any(id>UM_D)){return;}
 textureStore(outputPhi,vec3i(id),vec4f(umSampleVertex(vec3f(id))));
}` });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join("\n"));
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.ownership.bindLayout, this.resources] });
    const compile = (umCellWidth: number) => this.device.createComputePipelineAsync({ layout, compute: { module,
        entryPoint: this.direction === "restrict" ? "restrictVertices" : "prolongVertices",
        constants: { umCellWidth, umDispatchX: this.ownership.dispatchX },
      } });
    this.pipelines = this.direction === "restrict" ? await compileMixedTiers(compile) : [await compile(1)];
  }

  encode(encoder: GPUCommandEncoder): void {
    if (!this.pipelines) throw new Error("Mixed vertex transfer is not initialized");
    const pass = encoder.beginComputePass({ label: `Uniform mixed vertex ${this.direction}` });
    pass.setBindGroup(0, this.ownership.bindGroup); pass.setBindGroup(1, this.group);
    if (this.direction === "restrict") this.ownership.dispatch(pass, this.pipelines);
    else {
      pass.setPipeline(this.pipelines[0]!);
      const d = this.ownership.layout.lattice.dimensions;
      pass.dispatchWorkgroups(...d.map(n => Math.ceil((n + 1) / 4)) as [number, number, number]);
    }
    pass.end();
  }
}
