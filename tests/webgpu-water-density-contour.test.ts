import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { surfaceExtractionShader } from "../lib/webgpu-water-pipeline";

const modulePath = process.env.WEBGPU_NODE_MODULE;

function initializedBuffer(
  device: GPUDevice,
  data: ArrayBufferView,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  const buffer = device.createBuffer({
    size: Math.max(4, Math.ceil(data.byteLength / 4) * 4),
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  );
  buffer.unmap();
  return buffer;
}

test("volume-field marching cubes preserves every published liquid sample", () => {
  const functionBody = /fn presentationFieldCell\(cell:vec3i\)->f32\{([\s\S]*?)\n\}/
    .exec(surfaceExtractionShader)?.[1];
  assert.ok(functionBody, "the volume-field contour adapter must remain explicit");
  assert.equal(functionBody.replace(/\s+/g, ""), "returnfieldCell(cell);",
    "presentation filtering must not demote a published rho > 0.5 sample below the liquid isovalue");
});

test("a one-cell 0.6-density feature emits a 0.5 isosurface", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU density-contour validation",
}, async () => {
  const dawn = await import(pathToFileURL(modulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create(["backend=metal"]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => {
    errors.push((event as { error: { message: string } }).error.message);
  });

  const layout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ...[7, 8, 9, 11, 12].map((binding) => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" as const },
    })),
    { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
  ] });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: {
      module: device.createShaderModule({ code: surfaceExtractionShader }),
      entryPoint: "extractMain",
      constants: { countOnly: 1, sparseField: 0, thinWallFilmsEnabled: 0 },
    },
  });

  const dimensions = 4;
  const uniformData = new Float32Array(28);
  uniformData.set([64, 64, 0, 0], 0);
  uniformData.set([4, 4, 4, 0], 12);
  uniformData.set([dimensions, dimensions, dimensions, 0], 20);
  const uniform = initializedBuffer(device, uniformData, GPUBufferUsage.UNIFORM);
  const volume = device.createTexture({
    size: [dimensions, dimensions, dimensions],
    dimension: "3d",
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const floatsPerRow = 256 / 4;
  const density = new Float32Array(floatsPerRow * dimensions * dimensions);
  density[2 * dimensions * floatsPerRow + 2 * floatsPerRow + 2] = 0.6;
  device.queue.writeTexture(
    { texture: volume },
    density,
    { bytesPerRow: 256, rowsPerImage: dimensions },
    [dimensions, dimensions, dimensions],
  );
  const columns = device.createTexture({
    size: [dimensions, dimensions],
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING,
  });
  const vertices = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE });
  const drawArgs = initializedBuffer(
    device,
    new Uint32Array([0, 1, 0, 0, 0, 0, 0, 0]),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const cubes = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE });
  const fallbackStorage = device.createBuffer({ size: 128, usage: GPUBufferUsage.STORAGE });
  const fallbackParams = device.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM });
  const group = device.createBindGroup({ layout, entries: [
    { binding: 0, resource: { buffer: uniform } },
    { binding: 1, resource: volume.createView() },
    { binding: 2, resource: columns.createView() },
    { binding: 3, resource: { buffer: vertices } },
    { binding: 4, resource: { buffer: drawArgs } },
    { binding: 5, resource: { buffer: cubes } },
    ...[7, 8, 9, 11, 12].map((binding) => ({ binding, resource: { buffer: fallbackStorage } })),
    { binding: 10, resource: { buffer: fallbackParams } },
  ] });
  const readback = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(2, 2, 2);
  pass.end();
  encoder.copyBufferToBuffer(drawArgs, 0, readback, 0, 32);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const vertexCount = new Uint32Array(readback.getMappedRange())[0]!;
  assert.deepEqual(errors, []);
  assert.ok(vertexCount > 0,
    "rho=0.6 is liquid and must produce triangles even when every adjacent cell is empty");
  readback.unmap();
  await device.queue.onSubmittedWorkDone();
  for (const resource of [
    readback, fallbackParams, fallbackStorage, cubes, drawArgs, vertices,
    columns, volume, uniform,
  ]) resource.destroy();
  device.destroy();
});
