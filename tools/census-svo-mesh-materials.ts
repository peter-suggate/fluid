import assert from "node:assert/strict";
import type { SparseVoxelDrySceneRenderer } from "../lib/svo/pipeline/webgpu-svo-dry-scene";

/** Diagnostic-only inspection of the retained draw list. No scene or renderer
 * policy changes; count records actually submitted after camera culling. */
export async function censusSvoMeshMaterials(device: GPUDevice, renderer: SparseVoxelDrySceneRenderer) {
  const buffers = renderer as unknown as { surfaceMeshArena?: GPUBuffer; surfaceMeshVisible?: GPUBuffer; surfaceMeshState?: GPUBuffer };
  const { surfaceMeshArena: arena, surfaceMeshVisible: visible, surfaceMeshState: state } = buffers;
  assert.ok(arena && visible && state && renderer.surfaceMeshStatus?.state === "ready");
  const histogram = device.createBuffer({ size: 65536*4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: histogram.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const module = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read> arena:array<u32>;
@group(0) @binding(1) var<storage,read> visible:array<u32>;
@group(0) @binding(2) var<storage,read> state:array<u32>;
@group(0) @binding(3) var<storage,read_write> histogram:array<atomic<u32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 let i=id.x+id.y*65535u*64u;if(i>=state[1]){return;}
 atomicAdd(&histogram[arena[visible[i]*8u+3u]&65535u],1u);
}` });
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [arena,visible,state,histogram].map((buffer,binding)=>({binding,resource:{buffer}})) });
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,bind);
    const groups=Math.ceil(arena.size/32/64);
    pass.dispatchWorkgroups(Math.min(groups,65535),Math.ceil(groups/65535));pass.end();
    encoder.copyBufferToBuffer(histogram,0,readback,0,histogram.size);device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const counts=Array.from(new Uint32Array(readback.getMappedRange()));readback.unmap();
    return counts.flatMap((records,materialId)=>records?[{materialId,records}]:[]).sort((a,b)=>b.records-a.records);
  } finally { histogram.destroy();readback.destroy(); }
}
