import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sparseCM12PresentationPageAllocatorWGSL } from "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident";
import { createSparseCM12FramePlanPresentationLayout, createSparseCM12FramePlanPresentationInitialWords } from "../lib/methods/adaptive-volume/sparse-cm12-frame-plan-presentation";
import { createSparseCM12WorldDirectoryLayout } from "../lib/methods/adaptive-volume/sparse-cm12-world-directory";

const modulePath = process.env.WEBGPU_NODE_MODULE;
const nativeInstances = new Set<GPU>();
(modulePath ? test : test.skip)("an initially empty authored atlas publishes signed runtime pages without reading absent atlas records", { timeout: 30_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "tests/sparse-cm12-empty-presentation-dawn.test.ts");
  let device: GPUDevice | undefined;
  const buffers: GPUBuffer[] = [];
  let gpu: GPU | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); nativeInstances.add(gpu);
    const adapter = await gpu.requestAdapter();
    assert.ok(adapter); device = await adapter.requestDevice();
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    const capacity = 4;
    const presentation = createSparseCM12FramePlanPresentationLayout({ pageCapacity: capacity, brickCapacity: capacity, baseWords: 256 });
    const world = createSparseCM12WorldDirectoryLayout({ initialLeaves: 0, growthLeaves: capacity, maximumSpanLog: 0 });
    const activity = new Uint32Array(presentation.totalWords);
    activity.set(createSparseCM12FramePlanPresentationInitialWords(presentation), presentation.baseWords);
    activity[28 + 10] = 1;
    activity[28 + 48 + 10] = 1;
    const coordinates = [[-2, 1, 3], [2, 0, -1]];
    const worldWords = new Uint32Array(world.totalWords);
    for (const [leaf, coordinate] of coordinates.entries()) worldWords.set(coordinate.map(value => value >>> 0), world.leafBaseWords + 5 * leaf);
    const upload = (label: string, words: Uint32Array) => {
      const buffer = device!.createBuffer({ label, size: words.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      device!.queue.writeBuffer(buffer, 0, words as Uint32Array<ArrayBuffer>); buffers.push(buffer); return buffer;
    };
    const topology = upload("Empty authored atlas sentinel", new Uint32Array(1));
    const activityBuffer = upload("Active runtime leaves", activity);
    const metadata = upload("Presentation metadata", new Uint32Array(capacity * 4).fill(0xffffffff));
    const worklist = upload("Presentation worklist", new Uint32Array([1, 0, capacity, 0xc0080003, 0, 1, 1, 0, 0, 0, 0]));
    const directory = upload("Runtime signed coordinates", worldWords);
    const shader = device.createShaderModule({ code: sparseCM12PresentationPageAllocatorWGSL(capacity, 0, 0, presentation, world) });
    // WDR-only specialization legitimately removes the unused atlas binding.
    const compilation = await shader.getCompilationInfo();
    assert.deepEqual(compilation.messages.filter(message => message.type === "error"), []);
    const layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ...[1, 2, 3, 4].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })),
    ] });
    const explicit = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: shader, entryPoint: "allocateSparseCM12PresentationPages" } });
    const group = device.createBindGroup({ layout, entries: [topology, activityBuffer, metadata, worklist, directory].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const readback = device.createBuffer({ size: metadata.size + worklist.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); buffers.push(readback);
    const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass(); pass.setPipeline(explicit); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(metadata, 0, readback, 0, metadata.size);
    encoder.copyBufferToBuffer(worklist, 0, readback, metadata.size, worklist.size);
    device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(readback.getMappedRange());
    assert.equal(words[capacity * 4 + 1], 2, "both newly wet runtime leaves enter renderer worklist");
    const keys = new Set([words[1], words[5]]);
    for (const [x, y, z] of coordinates) assert.ok(keys.has(((x! + 1024) | ((y! + 512) << 11) | ((z! + 1024) << 21)) >>> 0));
    readback.unmap(); assert.deepEqual(errors, []);
  } finally { for (const buffer of buffers) buffer.destroy(); device?.destroy(); if (gpu) nativeInstances.delete(gpu); await releaseWebGPUExclusiveLock(); }
});

(modulePath ? test : test.skip)("a drop into an empty scene publishes a signed wet surface before time advances", { timeout: 180_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "empty fluid surface publication");
  let device: GPUDevice | undefined;
  let solver: import("../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver").WebGPUAdaptiveMassSolver | undefined;
  let drawProbe: Awaited<ReturnType<typeof import("./helpers/global-fine-draw-probe").createGlobalFineDrawProbe>> | undefined;
  let gpu: GPU | undefined;
  try {
    const { createEmptyScene } = await import("../lib/core/empty-scene");
    const { requiredFluidDeviceLimits } = await import("../lib/core/webgpu-device-limits");
    const { WebGPUAdaptiveMassSolver } = await import("../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver");
    const { unpackFineLevelSetPackedPhi } = await import("../lib/core/fine-levelset-packed-sample");
    const dawn = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
    Object.assign(globalThis, dawn.globals); gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); nativeInstances.add(gpu);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    const scene = createEmptyScene(); scene.systems = { ...scene.systems, fluid: true };
    solver = await WebGPUAdaptiveMassSolver.createAsync(device, scene, "balanced", undefined, {
      resolutionMode: "adaptive", brickFineResolution: 8, surfaceFineRings: 1, timeStep: "paper", pressureIterations: 32,
    }, () => {});
    await solver.waitForSimulationReady();
    const owner = solver.sparseWorld;
    const before = await solver.readPresentationPageAllocatorReceiptQA();
    const edit = await solver.editFluid({ operation: "add", shape: "ball", center_m: { x: .3, y: .55, z: .5 }, radius_m: .15 });
    assert.equal(edit.accepted, true, edit.reason);
    const allocator = await solver.readPresentationPageAllocatorReceiptQA();
    const source = solver.globalFineLevelSetSource;
    const read = async (buffer: GPUBuffer) => {
      const target = device!.createBuffer({ size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const encoder = device!.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, 0, target, 0, buffer.size);
      device!.queue.submit([encoder.finish()]); await target.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(target.getMappedRange().slice(0)); target.unmap(); target.destroy(); return words;
    };
    const metadata = await read(source.metadata), worklist = await read(source.worklist), samples = await read(source.samples);
    let activePages = 0, wetSamples = 0, airSamples = 0;
    for (let page = 0; page < metadata.length / 4; page++) {
      if (metadata[4 * page + 2] !== 1) continue;
      activePages++;
      for (let sample = 0; sample < source.plan.samplesPerBrick; sample++) {
        const phi = unpackFineLevelSetPackedPhi(samples[page * source.plan.samplesPerBrick + sample]!);
        wetSamples += Number(phi < 0); airSamples += Number(phi > 0);
      }
    }
    const frame = await solver.readFramePlanPresentationHeaderQA();
    await solver.assertSimulationHealthy();
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ before, allocator, worklist: [...worklist.slice(0, 7)], activePages, wetSamples, airSamples, frame }));
    assert.ok(allocator.residentPages > before.residentPages, "new water receives renderer pages");
    assert.ok(worklist[1]! > 0, "renderer worklist names live runtime pages");
    assert.ok(activePages > 0, "FPP publication accepts runtime pages");
    assert.ok(wetSamples > 0 && airSamples > 0, "published level set crosses the injected water interface");
    const { createGlobalFineDrawProbe } = await import("./helpers/global-fine-draw-probe");
    const { createGlobalFineLevelSetConsumerSource } = await import("../lib/core/octree-consumer-sampling");
    drawProbe = await createGlobalFineDrawProbe(device, createGlobalFineLevelSetConsumerSource(source));
    const wetDraw = await drawProbe.read(0);
    assert.ok(wetDraw.vertexCount > 0, "the exact published ball produces drawable triangles");
    const invalidDraw = await drawProbe.read(wetDraw.vertexCount, source.generation + 1);
    assert.equal(invalidDraw.vertexCount, wetDraw.vertexCount,
      "an invalid publication must retain the preceding mesh");
    const removed = await solver.editFluid({ operation: "remove", shape: "ball",
      center_m: { x: .3, y: .55, z: .5 }, radius_m: .15 });
    assert.equal(removed.accepted, true, removed.reason);
    const removedFields = await solver.readDiagnosticFields(true);
    assert.equal(removedFields.density.reduce((sum, value) => sum + value, 0), 0,
      "the exact inverse stamp empties the paused fluid field");
    const removedMetadata = await read(source.metadata), removedSamples = await read(source.samples);
    let removedActivePages = 0, removedWetSamples = 0;
    for (let page = 0; page < removedMetadata.length / 4; page++) {
      if (removedMetadata[4 * page + 2] !== 1) continue;
      removedActivePages++;
      for (let sample = 0; sample < source.plan.samplesPerBrick; sample++) {
        removedWetSamples += Number(unpackFineLevelSetPackedPhi(
          removedSamples[page * source.plan.samplesPerBrick + sample]!) < 0);
      }
    }
    console.log(JSON.stringify({ removedActivePages, removedWetSamples,
      removedFrame: await solver.readFramePlanPresentationHeaderQA() }));
    assert.equal(removedWetSamples, 0,
      "removal must retire the old surface or republish air before time advances");
    const removedDraw = await drawProbe.read(wetDraw.vertexCount);
    console.log(JSON.stringify({ wetDraw, invalidDraw, removedDraw }));
    assert.equal(removedDraw.vertexCount, 0,
      "a valid empty field must withdraw the preceding water mesh immediately");
    await solver.assertSimulationHealthy();
    assert.deepEqual(errors, []);
    assert.equal(solver.sparseWorld, owner); assert.equal(solver.info.submittedTime_s, 0);
  } finally { drawProbe?.destroy(); solver?.destroy(); if (device) { await device.queue.onSubmittedWorkDone(); device.destroy(); } if (gpu) nativeInstances.delete(gpu); await releaseWebGPUExclusiveLock(); }
});
