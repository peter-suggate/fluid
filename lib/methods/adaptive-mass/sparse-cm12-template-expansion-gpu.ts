import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import type { SparseCM12TemplateExpansion } from "./sparse-cm12-template-archetypes";

/** Expand native cell/face/term records on the device. The compact recipe owns
 * exact coefficients; placement changes only IDs and dyadic translations. */
export async function expandSparseCM12TemplateArchetypesGPU(device: GPUDevice,
  topology: GPUBuffer, template: Uint32Array, recipe: SparseCM12TemplateExpansion): Promise<void> {
  const [cellBase, rowBase, termBase] = [template[6]!, template[7]!, template[8]!];
  const rowCount = template[3]!;
  const data = device.createBuffer({ label: "CM12 interned native topology placement recipe",
    size: Math.max(4, recipe.words.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  try {
    device.queue.writeBuffer(data, 0, recipe.words.buffer as ArrayBuffer,
      recipe.words.byteOffset, recipe.words.byteLength);
    const code = `
@group(0) @binding(0) var<storage,read> m:array<u32>;
@group(0) @binding(1) var<storage,read_write> t:array<u32>;
fn f(at:u32)->f32{return bitcast<f32>(m[at]);}
fn put(at:u32,value:f32){t[at]=bitcast<u32>(value);}
@compute @workgroup_size(64) fn cells(@builtin(workgroup_id) group:vec3u,
 @builtin(local_invocation_index) lane:u32){
 let range=group.x+group.y*${device.limits.maxComputeWorkgroupsPerDimension}u;
 if(range>=${recipe.cellRangeCount}u){return;}
 let at=${recipe.rangeBase}u+12u*range;
 let origin=vec3f(f(at),f(at+1u),f(at+2u));let scale=f(at+3u);
 let n=vec3u(m[at+4u],m[at+5u],m[at+6u]);let first=m[at+7u];
 let maximum=vec3f(f(at+8u),f(at+9u),f(at+10u));let metadata=m[at+11u];
 for(var ordinal=lane;ordinal<n.x*n.y*n.z;ordinal+=64u){
  let q=vec3u(ordinal%n.x,(ordinal/n.x)%n.y,ordinal/(n.x*n.y));
  let lower=origin+vec3f(q)*scale;let upper=min(lower+vec3f(scale),maximum);
  let width=upper-lower;let center=.5*(lower+upper);
  let cell=${cellBase}u+8u*(first+ordinal);
  put(cell,center.x);put(cell+1u,center.y);put(cell+2u,center.z);
  put(cell+3u,width.x*width.y*width.z);
  put(cell+4u,width.x);put(cell+5u,width.y);put(cell+6u,width.z);t[cell+7u]=metadata;
 }
}
@compute @workgroup_size(64) fn rows(@builtin(global_invocation_id) invocation:vec3u){
 let row=invocation.x+invocation.y*${device.limits.maxComputeWorkgroupsPerDimension * 64}u;
 if(row>=${rowCount}u){return;}
 let instance=${recipe.instanceBase}u+5u*row;let pattern=m[instance];
 let sourceRange=${recipe.rangeBase}u+12u*m[instance+1u];
 let sourceBase=m[sourceRange+7u];var targetBase=0u;
 if(m[instance+2u]!=0xffffffffu){targetBase=m[${recipe.rangeBase}u+12u*m[instance+2u]+7u];}
 let first=m[instance+3u];let requirements=m[instance+4u];let count=m[pattern];
 t[${rowBase}u+row]=first|(count<<23u);
 t[${rowBase + rowCount}u+row]=requirements|m[pattern+1u];
 for(var plane=2u;plane<6u;plane++){t[${rowBase}u+plane*${rowCount}u+row]=m[pattern+plane];}
 for(var axis=0u;axis<3u;axis++){
  put(${rowBase}u+(6u+axis)*${rowCount}u+row,f(pattern+6u+axis)+f(sourceRange+axis));
 }
 for(var term=0u;term<count;term++){
  let local=m[pattern+10u+2u*term];let output=${termBase}u+2u*(first+term);
  t[output]=select(sourceBase,targetBase,(local&0x80000000u)!=0u)+(local&0x7fffffffu);
  t[output+1u]=m[pattern+11u+2u*term];
 }
}
`;
    const layout = device.createBindGroupLayout({ entries: [0, 1].map(binding => ({ binding,
      visibility: GPUShaderStage.COMPUTE, buffer: { type: binding === 0 ? "read-only-storage" as const : "storage" as const } })) });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const compiler = gpuCompilationManagerFor(device);
    const module = compiler.createShaderModule({ label: "CM12 interned topology GPU placement", code });
    const pipelines = await Promise.all(["cells", "rows"].map(entryPoint => compiler.compileComputePipeline({
      label: `CM12 native archetype ${entryPoint}`, layout: pipelineLayout, compute: { module, entryPoint },
    }, { priority: "critical" })));
    const bindings = device.createBindGroup({ layout, entries: [data, topology].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const encoder = device.createCommandEncoder({ label: "CM12 native topology archetype placement" });
    const counts = [recipe.cellRangeCount, Math.ceil(rowCount / 64)];
    for (let index = 0; index < counts.length; index++) {
      const groups = counts[index]!; if (groups === 0) continue;
      const pass = encoder.beginComputePass(); pass.setPipeline(pipelines[index]!); pass.setBindGroup(0, bindings);
      const width = Math.min(groups, device.limits.maxComputeWorkgroupsPerDimension);
      pass.dispatchWorkgroups(width, Math.ceil(groups / width)); pass.end();
    }
    device.queue.submit([encoder.finish()]);
  } finally { data.destroy(); }
}
