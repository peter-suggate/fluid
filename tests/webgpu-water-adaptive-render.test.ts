import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  extractionPrepareShader,
  surfaceExtractionShader,
  surfaceRasterShader,
  WATER_INTERFACE_CULL_MODES,
} from "../lib/webgpu-water-pipeline";

const modulePath = process.env.WEBGPU_NODE_MODULE;

function initializedBuffer(device: GPUDevice, data: ArrayBufferView, usage: GPUBufferUsageFlags): GPUBuffer {
  const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage, mappedAtCreation: true });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buffer.unmap();
  return buffer;
}

test("pageResolution=2 adaptive worklist preserves a nonzero leaf owner through polygonisation", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU adaptive-water rendering checks",
}, async () => {
  const dawn = await import(pathToFileURL(modulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create(["backend=metal"]).requestAdapter();
  assert.ok(adapter);
  assert.ok(adapter.limits.maxStorageBuffersPerShaderStage >= 10);
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => {
    errors.push((event as { error: { message: string } }).error.message);
  });

  const extractLayout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ...[7, 8, 9, 11, 12, 13, 14].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } })),
    { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 15, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
  ] });
  const prepareLayout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
  ] });
  const module = device.createShaderModule({ code: surfaceExtractionShader });
  const extractPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [extractLayout] }),
    compute: { module, entryPoint: "extractAdaptiveLeafMain", constants: { adaptiveField: 1 } },
  });
  const residentExtractPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [extractLayout] }),
    compute: { module, entryPoint: "extractAdaptiveMain", constants: { adaptiveField: 1 } },
  });
  const polygonisePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [extractLayout] }),
    compute: { module, entryPoint: "polygoniseMain", constants: { adaptiveField: 1 } },
  });
  const preparePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [prepareLayout] }),
    compute: { module: device.createShaderModule({ code: extractionPrepareShader }), entryPoint: "prepareMain" },
  });
  const surfaceLayout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
    { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
  ] });
  const surfaceModule = device.createShaderModule({ code: surfaceRasterShader });
  const frontInterfacePipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [surfaceLayout] }),
    vertex: { module: surfaceModule, entryPoint: "surfaceVertex" },
    fragment: { module: surfaceModule, entryPoint: "surfaceFragment", targets: [{ format: "rgba16float" }, { format: "rgba16float" }] },
    primitive: { topology: "triangle-list", frontFace: "ccw", cullMode: WATER_INTERFACE_CULL_MODES.front },
    depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
  });

  const uniformData = new Float32Array(28);
  uniformData.set([128, 128, 0, 0], 0);
  uniformData.set([3, 6, 3, 0], 4);
  uniformData.set([-0.25, 1.75, -0.25, 0], 8);
  uniformData.set([4, 4, 4, 0], 12); // container
  uniformData.set([4, 4, 4, 3], 20); // coarse dimensions and level-set mode
  const uniform = initializedBuffer(device, uniformData, GPUBufferUsage.UNIFORM);
  const leafData = new ArrayBuffer(96), leafU32 = new Uint32Array(leafData), leafF32 = new Float32Array(leafData);
  // A production-like coarse interface leaf. It has an affine phi plane but
  // deliberately has no resident detail page. The renderer must therefore
  // append one scale-aware cube directly from the compact leaf hierarchy.
  leafU32.set([1 | (1 << 10) | (1 << 20), 2, 0x22, 0], 12);
  leafF32.set([0, 0, 2, 0, 0, 0, 0, 0], 16);
  const leaves = initializedBuffer(device, new Uint8Array(leafData), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const arenaWords = new Uint32Array(256);
  arenaWords[16] = 0xffff_ffff;
  arenaWords[17] = 0xffff_ffff; // row 1 has no resident page
  arenaWords[22] = 0; // no active fine-page rows
  const arena = initializedBuffer(device, arenaWords, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const paramsData = new ArrayBuffer(128), paramsU32 = new Uint32Array(paramsData), paramsF32 = new Float32Array(paramsData);
  paramsU32.set([2, 1, 2, 8], 0);
  paramsU32.set([16, 18, 20, 21], 4);
  paramsU32.set([22, 32, 64, 128], 8);
  paramsU32.set([192, 2, 0, 0], 12);
  paramsF32.set([1, 1, 1, 0], 16);
  const params = initializedBuffer(device, new Uint8Array(paramsData), GPUBufferUsage.UNIFORM);
  const fallbackStorage = device.createBuffer({ size: 128, usage: GPUBufferUsage.STORAGE });
  const fallbackUniform = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM });
  const volume = device.createTexture({ size: { width: 4, height: 4, depthOrArrayLayers: 4 }, dimension: "3d", format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING });
  const columns = device.createTexture({ size: [4, 4], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING });
  const vertices = device.createBuffer({ size: 256 * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const drawArgs = initializedBuffer(device, new Uint32Array([0, 1, 0, 0, 0, 0, 0]), GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
  const cubes = device.createBuffer({ size: 256 * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const polygoniseDispatch = device.createBuffer({ size: 12, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
  const extractGroup = device.createBindGroup({ layout: extractLayout, entries: [
    { binding: 0, resource: { buffer: uniform } },
    { binding: 1, resource: volume.createView({ dimension: "3d" }) }, { binding: 2, resource: columns.createView() },
    { binding: 3, resource: { buffer: vertices } }, { binding: 4, resource: { buffer: drawArgs } }, { binding: 5, resource: { buffer: cubes } },
    ...[7, 8, 9, 11, 12].map((binding) => ({ binding, resource: { buffer: fallbackStorage } })),
    { binding: 10, resource: { buffer: fallbackUniform } },
    { binding: 13, resource: { buffer: leaves } }, { binding: 14, resource: { buffer: arena } }, { binding: 15, resource: { buffer: params } },
  ] });
  const prepareGroup = device.createBindGroup({ layout: prepareLayout, entries: [
    { binding: 0, resource: { buffer: drawArgs } }, { binding: 1, resource: { buffer: cubes } }, { binding: 2, resource: { buffer: polygoniseDispatch } },
  ] });
  const surfaceGroup = device.createBindGroup({ layout: surfaceLayout, entries: [
    { binding: 0, resource: { buffer: uniform } }, { binding: 1, resource: { buffer: vertices } },
  ] });
  const countersReadback = device.createBuffer({ size: 24, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const cubeReadback = device.createBuffer({ size: 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const vertexReadback = device.createBuffer({ size: 96, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const interfacePosition = device.createTexture({ size: [128, 128], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const interfaceNormal = device.createTexture({ size: [128, 128], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const interfaceDepth = device.createTexture({ size: [128, 128], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const interfaceBytesPerRow = 128 * 8;
  const interfaceReadback = device.createBuffer({ size: interfaceBytesPerRow * 128, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setBindGroup(0, extractGroup); pass.setPipeline(extractPipeline); pass.dispatchWorkgroups(1);
  pass.setBindGroup(0, prepareGroup); pass.setPipeline(preparePipeline); pass.dispatchWorkgroups(1);
  pass.setBindGroup(0, extractGroup); pass.setPipeline(polygonisePipeline); pass.dispatchWorkgroupsIndirect(polygoniseDispatch, 0);
  pass.end();
  const interfacePass = encoder.beginRenderPass({
    colorAttachments: [
      { view: interfacePosition.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
      { view: interfaceNormal.createView(), clearValue: { r: 0, g: 1, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
    ],
    depthStencilAttachment: { view: interfaceDepth.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
  });
  interfacePass.setPipeline(frontInterfacePipeline); interfacePass.setBindGroup(0, surfaceGroup); interfacePass.drawIndirect(drawArgs, 0); interfacePass.end();
  encoder.copyBufferToBuffer(drawArgs, 0, countersReadback, 0, 24);
  encoder.copyBufferToBuffer(cubes, 0, cubeReadback, 0, 8);
  encoder.copyBufferToBuffer(vertices, 0, vertexReadback, 0, 96);
  encoder.copyTextureToBuffer({ texture: interfaceNormal }, { buffer: interfaceReadback, bytesPerRow: interfaceBytesPerRow, rowsPerImage: 128 }, [128, 128]);
  device.queue.submit([encoder.finish()]);
  await Promise.all([countersReadback.mapAsync(GPUMapMode.READ), cubeReadback.mapAsync(GPUMapMode.READ), vertexReadback.mapAsync(GPUMapMode.READ), interfaceReadback.mapAsync(GPUMapMode.READ)]);
  const counters = new Uint32Array(countersReadback.getMappedRange());
  const firstCube = new Uint32Array(cubeReadback.getMappedRange());
  const firstTriangle = new Float32Array(vertexReadback.getMappedRange());
  const interfacePixels = new Uint16Array(interfaceReadback.getMappedRange());
  let frontCoverage = 0;
  for (let pixel = 0; pixel < 128 * 128; pixel += 1) if (interfacePixels[4 * pixel + 3] !== 0) frontCoverage += 1;
  assert.deepEqual(errors, [], "the production shader and dispatch must pass WebGPU validation");
  assert.equal(counters[4], 1, "the nonresident coarse leaf must append one surface cube");
  assert.equal(firstCube[0] & 0x1fff, 2, "the cube x origin is leaf origin times page resolution");
  assert.equal((firstCube[0] >>> 13) & 0x1fff, 2, "the cube z origin is leaf origin times page resolution");
  assert.equal(firstCube[1] & 0x1fff, 2, "the cube y origin is leaf origin times page resolution");
  assert.equal((firstCube[0] >>> 26) | (((firstCube[1] >>> 13) & 0x3ffff) << 6), 1, "the compact cube record carries the nonzero owner row");
  assert.ok(counters[0] >= 3, "the production polygoniser must emit visible triangle geometry");
  assert.ok([...firstTriangle.slice(0, 24)].every(Number.isFinite), "emitted positions and normals must be finite");
  assert.ok(firstTriangle[3] === 1 && firstTriangle[11] === 1 && firstTriangle[19] === 1, "the first three vertex positions are live");
  const position = (vertex: number) => firstTriangle.slice(vertex * 8, vertex * 8 + 3);
  const normal = (vertex: number) => firstTriangle.slice(vertex * 8 + 4, vertex * 8 + 7);
  const a = position(0), b = position(1), c = position(2);
  const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const geometric = [ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]];
  const averageNormal = [0, 1, 2].map((axis) => normal(0)[axis] + normal(1)[axis] + normal(2)[axis]);
  assert.ok(averageNormal[1] > 0, "negative-inside adaptive phi must produce an upward outward normal");
  assert.ok(geometric.reduce((sum, value, axis) => sum + value * averageNormal[axis], 0) > 0, "adaptive triangle winding must follow its outward normal");
  assert.ok(frontCoverage > 0, "outward adaptive triangles must survive the front-entry cull pass");
  countersReadback.unmap(); cubeReadback.unmap(); vertexReadback.unmap(); interfaceReadback.unmap();

  // A resident page owns only samples inside its leaf. Its affine plane must
  // supply an adjacent air sample when the interface lies exactly on the leaf
  // boundary; clamping the page outside its owner makes this case disappear.
  const residentLeafData = new ArrayBuffer(96), residentLeafU32 = new Uint32Array(residentLeafData), residentLeafF32 = new Float32Array(residentLeafData);
  residentLeafU32.set([1 | (1 << 10) | (1 << 20), 1, 0x22, 0], 12);
  residentLeafF32.set([-0.5, 0, 1, 0, 0, 0, 0, 0], 16);
  const residentArena = new Uint32Array(256);
  residentArena[16] = 0xffff_ffff; residentArena[17] = 0;
  residentArena[19] = 3;
  residentArena[22] = 1; residentArena[23] = 1; residentArena[24] = 1; residentArena[25] = 1; residentArena[26] = 1;
  const residentPhi = new Float32Array(residentArena.buffer, 64 * 4, 8);
  residentPhi.set([-0.75, -0.75, -0.25, -0.25, -0.75, -0.75, -0.25, -0.25]);
  device.queue.writeBuffer(leaves, 0, residentLeafData);
  device.queue.writeBuffer(arena, 0, residentArena);
  device.queue.writeBuffer(drawArgs, 0, new Uint32Array([0, 1, 0, 0, 0, 0, 0]));
  const residentEncoder = device.createCommandEncoder();
  residentEncoder.clearBuffer(vertices); residentEncoder.clearBuffer(cubes);
  const residentCompute = residentEncoder.beginComputePass();
  residentCompute.setBindGroup(0, extractGroup); residentCompute.setPipeline(residentExtractPipeline); residentCompute.dispatchWorkgroups(1);
  residentCompute.setBindGroup(0, prepareGroup); residentCompute.setPipeline(preparePipeline); residentCompute.dispatchWorkgroups(1);
  residentCompute.setBindGroup(0, extractGroup); residentCompute.setPipeline(polygonisePipeline); residentCompute.dispatchWorkgroupsIndirect(polygoniseDispatch, 0);
  residentCompute.end();
  const residentInterface = residentEncoder.beginRenderPass({
    colorAttachments: [
      { view: interfacePosition.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
      { view: interfaceNormal.createView(), clearValue: { r: 0, g: 1, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
    ],
    depthStencilAttachment: { view: interfaceDepth.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
  });
  residentInterface.setPipeline(frontInterfacePipeline); residentInterface.setBindGroup(0, surfaceGroup); residentInterface.drawIndirect(drawArgs, 0); residentInterface.end();
  residentEncoder.copyBufferToBuffer(drawArgs, 0, countersReadback, 0, 24);
  residentEncoder.copyBufferToBuffer(cubes, 0, cubeReadback, 0, 8);
  residentEncoder.copyBufferToBuffer(vertices, 0, vertexReadback, 0, 96);
  residentEncoder.copyTextureToBuffer({ texture: interfaceNormal }, { buffer: interfaceReadback, bytesPerRow: interfaceBytesPerRow, rowsPerImage: 128 }, [128, 128]);
  device.queue.submit([residentEncoder.finish()]);
  await Promise.all([countersReadback.mapAsync(GPUMapMode.READ), cubeReadback.mapAsync(GPUMapMode.READ), vertexReadback.mapAsync(GPUMapMode.READ), interfaceReadback.mapAsync(GPUMapMode.READ)]);
  const residentCounters = new Uint32Array(countersReadback.getMappedRange());
  const residentCube = new Uint32Array(cubeReadback.getMappedRange());
  const residentPixels = new Uint16Array(interfaceReadback.getMappedRange());
  let residentCoverage = 0;
  for (let pixel = 0; pixel < 128 * 128; pixel += 1) if (residentPixels[4 * pixel + 3] !== 0) residentCoverage += 1;
  assert.ok(residentCounters[4] > 0, "a resident page touching its affine air fallback must append a surface cube");
  assert.ok(residentCounters[0] >= 3, "the resident page boundary must polygonise into real geometry");
  assert.equal((residentCube[0] >>> 26) | (((residentCube[1] >>> 13) & 0x3ffff) << 6), 1, "resident cubes retain their nonzero page owner");
  assert.ok((residentCube[1] & 0x80000000) !== 0, "resident page cubes retain unit fine-lattice scale");
  assert.ok(residentCoverage > 0, "the resident page boundary must survive the front-entry cull pass");
  countersReadback.unmap(); cubeReadback.unmap(); vertexReadback.unmap(); interfaceReadback.unmap();
  for (const buffer of [uniform, leaves, arena, params, fallbackStorage, fallbackUniform, vertices, drawArgs, cubes, polygoniseDispatch, countersReadback, cubeReadback, vertexReadback, interfaceReadback]) buffer.destroy();
  for (const texture of [volume, columns, interfacePosition, interfaceNormal, interfaceDepth]) texture.destroy();
  device.destroy();
});
