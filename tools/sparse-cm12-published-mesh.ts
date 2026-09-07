/** Capture a freshly extracted production surface from the accepted GPU field.
 * This runs the shipping classification/scan/emission shaders. It cannot reuse
 * a renderer's retained mesh and never creates a substitute analytic mesh.
 */
import assert from "node:assert/strict";
import type { WebGPUFineLevelSetBrickSource } from "../lib/core/levelset-consumer-abi";
import { globalFineSurfaceClassificationShader } from "../lib/core/webgpu-water-global-fine-classify";
import { GLOBAL_FINE_SURFACE_EMIT_LANES, globalFineClassifiedEmitShader,
  globalFineClassifiedIndirectScanShader } from "../lib/core/webgpu-water-global-fine-tetra";

async function read(device: GPUDevice, source: GPUBuffer, bytes: number): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size: Math.max(4, bytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, staging, 0, Math.max(4, bytes));
    device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
    return staging.getMappedRange().slice(0);
  } finally { if (staging.mapState === "mapped") staging.unmap(); staging.destroy(); }
}
export async function readPublishedCM12Mesh(device: GPUDevice, source: WebGPUFineLevelSetBrickSource,
  origin: readonly [number, number, number]): Promise<{ mesh: Float32Array; generation: number; activeCubes: number }> {
  const owned: GPUBuffer[] = [];
  const buffer = (label: string, size: number, usage: GPUBufferUsageFlags, contents?: ArrayBufferView) => {
    const result = device.createBuffer({ label, size: Math.max(4, size), usage, mappedAtCreation: !!contents });
    owned.push(result);
    if (contents) { new Uint8Array(result.getMappedRange()).set(new Uint8Array(contents.buffer, contents.byteOffset, contents.byteLength)); result.unmap(); }
    return result;
  };
  try {
    const { plan } = source;
    const dimensions = plan.sampleDimensions;
    const count = dimensions.reduce((a, b) => a * b, 1);
    const cubeCapacity = count * 6;
    // Large enough for these two real scenes, with an explicit overflow check.
    // Allocation is tied to test capacity, never used as a quality criterion.
    const vertexCapacity = Math.max(100_000, dimensions[0] * dimensions[2] * 100);
    const worklistHeader = new Uint32Array(await read(device, source.worklist, 28));
    const uniformWords = new Uint32Array(28); new Float32Array(uniformWords.buffer).set([...dimensions.map(n => n * plan.fineCellWidth), 1], 12);
    const paramsWords = new Uint32Array(28), pf = new Float32Array(paramsWords.buffer);
    paramsWords.set([...dimensions, 8], 0);
    paramsWords.set([...dimensions.map(n => Math.ceil(n / 8)), 512], 4);
    paramsWords.set([plan.maximumResidentBricks, 7, plan.maximumResidentBricks, worklistHeader[0]!], 8);
    pf.set([0, 0, 0, plan.fineCellWidth], 12);
    pf[16] = 1; pf.set([plan.fineCellWidth, plan.fineCellWidth, plan.fineCellWidth, 0], 20);
    pf.set([...origin, source.surfaceMeshRefinement ?? 2], 24);
    const uniforms = buffer("acceptance mesh uniforms", 112, GPUBufferUsage.UNIFORM, uniformWords);
    const params = buffer("acceptance mesh params", 112, GPUBufferUsage.UNIFORM, paramsWords);
    const args = buffer("acceptance mesh receipt", 32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      Uint32Array.from([0, 1, 0, 0, 0, 0xffffffff, 0, 0xffffffff]));
    const cubes = buffer("acceptance mesh cubes", cubeCapacity * 8, GPUBufferUsage.STORAGE);
    const values = buffer("acceptance mesh values", cubeCapacity * 32, GPUBufferUsage.STORAGE);
    const offsets = buffer("acceptance mesh offsets", cubeCapacity * GLOBAL_FINE_SURFACE_EMIT_LANES * 4, GPUBufferUsage.STORAGE);
    const vertices = buffer("acceptance mesh vertices", vertexCapacity * 32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const empty = buffer("acceptance mesh empty", 256, GPUBufferUsage.STORAGE, new Uint32Array(64));
    const directory = source.coarsePhiDirectory ?? buffer("acceptance mesh no coarse directory", 64, GPUBufferUsage.STORAGE, new Uint32Array(16));
    const createPipeline = async (label: string, code: string, entryPoint: string) => device.createComputePipelineAsync({
      label, layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint } });
    const [classify, scan, emit] = await Promise.all([
      createPipeline("acceptance production classify", globalFineSurfaceClassificationShader, "extractGlobalFineMain"),
      createPipeline("acceptance production scan", globalFineClassifiedIndirectScanShader, "scanGlobalFineTriangles"),
      createPipeline("acceptance production emit", globalFineClassifiedEmitShader, "emitGlobalFineTetrahedra"),
    ]);
    const group = (pipeline: GPUComputePipeline, pairs: readonly (readonly [number, GPUBuffer])[]) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries: pairs.map(([binding, resource]) => ({ binding, resource: { buffer: resource } })) });
    const classifyGroup = group(classify, [[0, uniforms], [4, args], [5, cubes], [6, values], [8, source.worklist],
      [9, source.samples], [10, params], [12, source.metadata], [16, directory], [17, empty]]);
    const scanGroup = group(scan, [[3, vertices], [4, args], [5, cubes], [6, values], [7, offsets], [10, params], [11, empty]]);
    const emitGroup = group(emit, [[0, uniforms], [3, vertices], [4, args], [5, cubes], [6, values], [7, offsets],
      [8, source.worklist], [9, source.samples], [10, params], [12, source.metadata], [16, directory]]);
    const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
    pass.setPipeline(classify); pass.setBindGroup(0, classifyGroup); pass.dispatchWorkgroups(Math.ceil(worklistHeader[1]! * 512 / 256));
    pass.setPipeline(scan); pass.setBindGroup(0, scanGroup); pass.dispatchWorkgroups(1);
    pass.setPipeline(emit); pass.setBindGroup(0, emitGroup);
    pass.dispatchWorkgroups(Math.ceil(cubeCapacity / 64), GLOBAL_FINE_SURFACE_EMIT_LANES, 1);
    pass.end(); device.queue.submit([encoder.finish()]);
    const receipt = new Uint32Array(await read(device, args, 32));
    assert.ok(receipt[0]! > 0 && receipt[0]! < vertexCapacity, `fresh mesh empty or capacity exhausted: ${receipt}`);
    assert.ok(receipt[4]! < cubeCapacity, `fresh mesh cube capacity exhausted: ${receipt}`);
    assert.equal(receipt[7], worklistHeader[0], "mesh must consume the current publication generation");
    const mesh = new Float32Array(await read(device, vertices, receipt[0]! * 32));
    return { mesh, generation: receipt[7]!, activeCubes: receipt[4]! };
  } finally { for (const resource of owned) resource.destroy(); }
}
