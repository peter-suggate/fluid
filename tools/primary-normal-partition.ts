import assert from "node:assert/strict";
import { SVO_SURFACE_MESH_STATE } from "../lib/svo/features/primary-visibility/svo-surface-mesh";

/** Benchmark-only near/far partition fused into production visibility culling.
 * Preserve the original visibility output for every downstream consumer. */
export function primaryNormalPartition(device: GPUDevice, thresholds: number[]) {
  const shaderCode = new WeakMap<GPUShaderModule, string>();
  const layouts = new WeakMap<GPUPipelineLayout, GPUBindGroupLayout[]>();
  const groupLayouts = new WeakMap<GPUBindGroupLayout, GPUBindGroupLayoutEntry[]>();
  const extended = new WeakMap<GPUBindGroupLayout, GPUBindGroupLayout>();
  const pipelines = new WeakMap<GPUComputePipeline, Map<number, GPUComputePipeline>>();
  const descriptors = new WeakMap<GPUBindGroup, GPUBindGroupDescriptor>();
  const groups = new WeakMap<GPUBindGroup, Map<string, GPUBindGroup>>();
  const resources = new WeakMap<GPUBuffer, Map<number, { buffer: GPUBuffer; paneBytes: number }>>();
  let current: Map<number, { buffer: GPUBuffer; paneBytes: number }> | undefined;
  let currentState: GPUBuffer | undefined;

  function resource(visible: GPUBuffer) {
    let held = resources.get(visible);
    if (held) return held;
    held = new Map();
    const paneBytes = Math.ceil(visible.size / 2 / 256) * 256;
    for (const threshold of thresholds) {
      const buffer = device.createBuffer({ label: `Partitioned voxel indices ${threshold}px`,
        size: 256 + 2 * paneBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true });
      const words = new Uint32Array(buffer.getMappedRange());words[1] = words[5] = 1;words[8] = paneBytes / 4;
      buffer.unmap();held.set(threshold, { buffer, paneBytes });
    }
    resources.set(visible, held);
    const destroy = visible.destroy.bind(visible);
    visible.destroy = () => { for (const entry of held.values()) entry.buffer.destroy();destroy(); };
    return held;
  }

  function group(original: GPUBindGroup, threshold: number, kind: "compute" | "near" | "far") {
    const descriptor = descriptors.get(original);
    if (!descriptor) return original;
    const entries = Array.from(descriptor.entries), binding = kind === "compute" ? 34 : 35;
    const visible = entries.find(entry => entry.binding === binding)?.resource as GPUBufferBinding | undefined;
    if (!visible?.buffer) return original;
    const held = groups.get(original) ?? new Map<string, GPUBindGroup>();groups.set(original, held);
    const key = `${threshold}:${kind}`;
    if (held.has(key)) return held.get(key)!;
    const { buffer, paneBytes } = resource(visible.buffer).get(threshold)!;
    const result = device.createBindGroup(kind === "compute" ? {
      ...descriptor, layout: extended.get(descriptor.layout)!, entries: [...entries, { binding: 43, resource: { buffer } }],
    } : { ...descriptor, entries: entries.map(entry => entry.binding !== 35 ? entry : {
      binding: 35, resource: { buffer, offset: 256 + (kind === "far" ? paneBytes : 0), size: paneBytes },
    }) });
    held.set(key, result);return result;
  }

  return {
    shader(module: GPUShaderModule, code: string) { shaderCode.set(module, code); },
    layout(layout: GPUPipelineLayout, descriptor: GPUPipelineLayoutDescriptor) {
      layouts.set(layout, Array.from(descriptor.bindGroupLayouts) as GPUBindGroupLayout[]);
    },
    groupLayout(layout: GPUBindGroupLayout, descriptor: GPUBindGroupLayoutDescriptor) { groupLayouts.set(layout, Array.from(descriptor.entries)); },
    bindGroup(original: GPUBindGroup, descriptor: GPUBindGroupDescriptor) {
      descriptors.set(original, descriptor);
      const entries = Array.from(descriptor.entries);
      if (entries.some(entry => entry.binding === 30) && entries.some(entry => entry.binding === 34)) {
        current = resource((entries.find(entry => entry.binding === 34)!.resource as GPUBufferBinding).buffer);
        currentState = (entries.find(entry => entry.binding === 30)!.resource as GPUBufferBinding).buffer;
      }
    },
    async compile(original: GPUComputePipeline, descriptor: GPUComputePipelineDescriptor) {
      if (!thresholds.length || descriptor.label !== "surfaceMeshCull") return;
      const originalLayouts = layouts.get(descriptor.layout as GPUPipelineLayout)!;
      assert.equal(originalLayouts.length, 2, "Partition study expects full-resolution split shading");
      const oldGroup = originalLayouts[1]!;
      let extendedGroup = extended.get(oldGroup);
      if (!extendedGroup) {
        extendedGroup = device.createBindGroupLayout({ entries: [...groupLayouts.get(oldGroup)!,
          { binding: 43, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] });
        extended.set(oldGroup, extendedGroup);
      }
      const layout = device.createPipelineLayout({ bindGroupLayouts: [originalLayouts[0]!, extendedGroup] });
      const alternatives = new Map<number, GPUComputePipeline>();
      for (const threshold of thresholds) {
        const code = shaderCode.get(descriptor.compute.module)!;
        const start = code.indexOf("fn surfaceMeshCull(@"), end = code.indexOf("\n}\n", start) + 3;
        assert.ok(start >= 0 && end > start);
        let cull = code.slice(start, end);
        const classification = `
  var tier=0u;
  if(visible){
    let quad=meshArena[index];let camera=dryRasterPrimaryCamera();
    let halfExtent=vec3f(meshQuadExtent(quad))*dry.mapping.cellSize*.5;
    let centre=dry.mapping.worldOrigin+vec3f(meshQuadOrigin(quad))*dry.mapping.cellSize+halfExtent;
    let depth=max(dot(centre-camera[0],camera[1])-dot(halfExtent,abs(camera[1])),DRY_REVERSED_Z_NEAR_M);
    let h=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z))*f32(1u<<(dry.mapping.maximumDepth-meshQuadDepth(quad.face)));
    tier=select(0u,1u,h*uniforms.viewport.y/(2.0*cameraTanHalfFov()*depth)<=${threshold.toFixed(1)});
  }`;
        if(cull.includes("meshCompact(index,visible,false);")){
          cull=cull.replace("meshCompact(index,visible,false);",`meshCompact(index,visible,false);
${classification}
  let near=visible&&tier==0u;let far=visible&&tier==1u;
  let nearCount=subgroupAdd(select(0u,1u,near));let farCount=subgroupAdd(select(0u,1u,far));
  let nearRank=subgroupExclusiveAdd(select(0u,1u,near));let farRank=subgroupExclusiveAdd(select(0u,1u,far));
  var nearBase=0u;var farBase=0u;let vertices=select(6u,3u,meshDmcEnabled());
  if(subgroupElect()){
    if(nearCount>0u){nearBase=atomicAdd(&partitionOutput.args[0],nearCount*vertices)/vertices;}
    if(farCount>0u){farBase=atomicAdd(&partitionOutput.args[4],farCount*vertices)/vertices;}
  }
  nearBase=subgroupBroadcastFirst(nearBase);farBase=subgroupBroadcastFirst(farBase);
  if(near){partitionOutput.indices[nearBase+nearRank]=index;}
  if(far){partitionOutput.indices[partitionOutput.capacity+farBase+farRank]=index;}`);
        }else{
          assert.ok(cull.includes("meshVisibleOutput[meshCullBase+rank]=index"), "Partition study must recognize production compaction");
          cull = cull.replace("  if(lane==0u){atomicStore(&meshCullCount,0u);}", `${classification}
  if(lane==0u){atomicStore(&partitionCounts[0],0u);atomicStore(&partitionCounts[1],0u);atomicStore(&meshCullCount,0u);}`)
          .replace("var rank=0u;if(visible){rank=atomicAdd(&meshCullCount,1u);}", "var rank=0u;var partitionRank=0u;if(visible){rank=atomicAdd(&meshCullCount,1u);partitionRank=atomicAdd(&partitionCounts[tier],1u);}")
          .replace("    let count=atomicLoad(&meshCullCount);", `
    let vertices=select(6u,3u,meshDmcEnabled());
    partitionBases[0]=atomicAdd(&partitionOutput.args[0],atomicLoad(&partitionCounts[0])*vertices)/vertices;
    partitionBases[1]=atomicAdd(&partitionOutput.args[4],atomicLoad(&partitionCounts[1])*vertices)/vertices;
    let count=atomicLoad(&meshCullCount);`)
          .replace("meshVisibleOutput[meshCullBase+rank]=index;", "meshVisibleOutput[meshCullBase+rank]=index;partitionOutput.indices[tier*partitionOutput.capacity+partitionBases[tier]+partitionRank]=index;");
        }
        const module = device.createShaderModule({ label: `Partitioned primary culling ${threshold}px`, code: code.slice(0, start) + cull + code.slice(end) + `
struct PartitionOutput {args:array<atomic<u32>,8>,capacity:u32,padding:array<u32,55>,indices:array<u32>}
@group(1) @binding(43) var<storage,read_write> partitionOutput:PartitionOutput;
var<workgroup> partitionCounts:array<atomic<u32>,2>;
var<workgroup> partitionBases:array<u32,2>;
` });
        alternatives.set(threshold, await device.createComputePipelineAsync({ ...descriptor, layout, compute: { ...descriptor.compute, module } }));
      }
      pipelines.set(original, alternatives);
    },
    pipeline(original: GPUComputePipeline, threshold: number) { return pipelines.get(original)?.get(threshold) ?? original; },
    group,
    reset(encoder: GPUCommandEncoder, threshold: number) {
      const buffer = current!.get(threshold)!.buffer;encoder.clearBuffer(buffer, 0, 4);encoder.clearBuffer(buffer, 16, 4);
    },
    args(originalDrawGroup: GPUBindGroup, threshold: number) {
      const entry = Array.from(descriptors.get(originalDrawGroup)!.entries).find(entry => entry.binding === 35)!;
      return resource((entry.resource as GPUBufferBinding).buffer).get(threshold)!.buffer;
    },
    async totalCensus() {
      assert.ok(currentState, "Primary mesh state must be published");
      const read = device.createBuffer({size:4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      try {
        const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(currentState,SVO_SURFACE_MESH_STATE.packedDraw*4,read,0,4);
        device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
        return {voxelVertices:new Uint32Array(read.getMappedRange())[0]!};
      } finally {read.destroy();}
    },
    async census(threshold?: number) {
      const read = device.createBuffer({size:48,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      try {
        const encoder=device.createCommandEncoder();
        encoder.copyBufferToBuffer(threshold === undefined ? currentState! : current!.get(threshold)!.buffer,threshold === undefined ? SVO_SURFACE_MESH_STATE.nearNormalDraw*4 : 0,read,0,32);
        encoder.copyBufferToBuffer(currentState!,SVO_SURFACE_MESH_STATE.packedDraw*4,read,32,16);
        device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
        const words=new Uint32Array(read.getMappedRange());
        assert.equal(words[0]!+words[4]!,words[8],"Partitioning must preserve every submitted voxel vertex");
        return {nearVoxelVertices:words[0]!,farVoxelVertices:words[4]!};
      } finally {read.destroy();}
    },
  };
}
