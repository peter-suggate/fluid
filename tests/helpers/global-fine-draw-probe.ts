import type { GlobalFineLevelSetConsumerSource } from "../../lib/core/octree-consumer-sampling";
import { globalFineSurfaceClassificationShader } from "../../lib/core/webgpu-water-global-fine-classify";
import { globalFineClassifiedScanShader, GLOBAL_FINE_SURFACE_EMIT_LANES } from "../../lib/core/webgpu-water-global-fine-tetra";

/** Real production classification + triangle scan, with retained draw-count
 * semantics. No camera, raster approximation or CPU surface reconstruction. */
export async function createGlobalFineDrawProbe(device: GPUDevice, source: GlobalFineLevelSetConsumerSource) {
  const owned: GPUBuffer[] = [];
  const make = (size: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) => {
    const buffer = device.createBuffer({ size, usage }); owned.push(buffer); return buffer;
  };
  const uniforms = make(112, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const uniformWords = new Float32Array(28);
  uniformWords.set(source.sampleDimensions.map(value => value * source.fineCellWidth), 12);
  device.queue.writeBuffer(uniforms, 0, uniformWords);
  const params = make(112, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const bytes = new ArrayBuffer(112), u32 = new Uint32Array(bytes), f32 = new Float32Array(bytes);
  u32.set([...source.sampleDimensions, source.brickResolution], 0);
  u32.set([...source.brickDimensions, source.samplesPerBrick], 4);
  u32.set([source.pageCapacity, 7, source.pageCapacity, source.generation], 8);
  f32.set([...source.domainOrigin, source.fineCellWidth], 12); f32[16] = source.fineFactor;
  f32.set([source.fineCellWidth, source.fineCellWidth, source.fineCellWidth], 20);
  f32[27] = source.surfaceMeshRefinement ?? 0;
  device.queue.writeBuffer(params, 0, bytes);
  const draw = make(32), cubes = make(8192 * 8), values = make(8192 * 32);
  const offsets = make(8192 * GLOBAL_FINE_SURFACE_EMIT_LANES * 4), vertices = make(65536 * 32);
  const absentCoarse = make(256), absentTopology = make(32);
  const classification = await device.createComputePipelineAsync({ layout: "auto",
    compute: { module: device.createShaderModule({ code: globalFineSurfaceClassificationShader }), entryPoint: "extractGlobalFineMain" } });
  const scan = await device.createComputePipelineAsync({ layout: "auto",
    compute: { module: device.createShaderModule({ code: globalFineClassifiedScanShader }), entryPoint: "scanGlobalFineTriangles" } });
  const classificationGroup = device.createBindGroup({ layout: classification.getBindGroupLayout(0), entries: [
    [0, uniforms], [4, draw], [5, cubes], [6, values], [8, source.worklist.buffer],
    [9, source.samples.buffer], [10, params], [12, source.metadata.buffer],
    [16, source.coarsePhiDirectory?.buffer ?? absentCoarse], [17, source.topologyControl?.buffer ?? absentTopology],
  ].map(([binding, buffer]) => ({ binding: binding as number, resource: { buffer: buffer as GPUBuffer } })) });
  const scanGroup = device.createBindGroup({ layout: scan.getBindGroupLayout(0), entries: [
    [3, vertices], [4, draw], [5, cubes], [6, values], [7, offsets], [10, params],
  ].map(([binding, buffer]) => ({ binding: binding as number, resource: { buffer: buffer as GPUBuffer } })) });
  return {
    async read(previousVertexCount: number, generation = source.generation) {
      device.queue.writeBuffer(params, 44, new Uint32Array([generation]));
      device.queue.writeBuffer(draw, 0, new Uint32Array([previousVertexCount, 1, 0, 0, 0, 0xffffffff, 0, 0xffffffff]));
      const receipt = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        const encoder = device.createCommandEncoder();
        const classify = encoder.beginComputePass(); classify.setPipeline(classification); classify.setBindGroup(0, classificationGroup);
        classify.dispatchWorkgroups(Math.max(1, Math.ceil(source.pageCapacity * source.samplesPerBrick / 256))); classify.end();
        const prepare = encoder.beginComputePass(); prepare.setPipeline(scan); prepare.setBindGroup(0, scanGroup); prepare.dispatchWorkgroups(1); prepare.end();
        encoder.copyBufferToBuffer(draw, 0, receipt, 0, 32); device.queue.submit([encoder.finish()]);
        await receipt.mapAsync(GPUMapMode.READ); const words = new Uint32Array(receipt.getMappedRange().slice(0)); receipt.unmap();
        return { vertexCount: words[0]!, cubes: words[4]!, allocator: words[5]!, authority: words[6]!, generation: words[7]! };
      } finally { receipt.destroy(); }
    },
    destroy() { for (const buffer of owned) buffer.destroy(); },
  };
}
