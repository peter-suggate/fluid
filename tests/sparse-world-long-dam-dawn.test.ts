import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { FINE_LEVELSET_SIGNED_SPARSE_ADDRESS_FLAG } from
  "../lib/core/fine-levelset-brick-abi";
import { resolveMethodValues } from "../lib/core/method-contract";
import {
  createSparseCM12LongDamBreakScene,
  SPARSE_CM12_LONG_DAM_METHOD_PROFILE,
} from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { packCompactFineLevelSetParams } from
  "../lib/core/compact-fine-levelset-phi";
import { globalFineSurfaceClassificationShader } from
  "../lib/core/webgpu-water-global-fine-classify";
import { globalFineSurfaceDispatch } from "../lib/core/webgpu-water-pipeline";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { decodeSparseCM12SignedPresentationKey } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import type { SparseWorld, SparseWorldPresentation } from "../lib/sparse-world";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

const INITIAL_LONG_DAM_WET_TILE_COUNT = 80;
const INITIAL_LONG_DAM_DRY_SUPPORT_TILE_COUNT = 196;
const INITIAL_LONG_DAM_TILE_COUNT = INITIAL_LONG_DAM_WET_TILE_COUNT
  + INITIAL_LONG_DAM_DRY_SUPPORT_TILE_COUNT;
const LONG_DAM_FAR_WALL_PAGE_X = 23;
// The Sparse CM12 profile advances at the paper's 1/30 s, so the authored
// four-second scene is 120 steps. The previous 1,200 retained the old 4 ms
// scene-step count after this lane moved to the paper timestep and simulated
// forty seconds instead of four.
const LONG_DAM_GATE_STEPS = Number(process.env.FLUID_LONG_DAM_GATE_STEPS ?? 120);
const LONG_DAM_CHECKPOINT_INTERVAL = 50;

const pageRangeSamples = (receipt: Awaited<ReturnType<
  typeof publishedNegativeFront>>, begin: number, end: number): number =>
  receipt.negativeSamplesByPageX.slice(begin, end)
    .reduce((sum, count) => sum + count, 0);

async function readGPUWords(device: GPUDevice, source: GPUBuffer,
  wordCount: number, sourceOffset = 0): Promise<Uint32Array> {
  const readback = device.createBuffer({
    size: 4 * wordCount,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, sourceOffset, readback, 0, 4 * wordCount);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    return new Uint32Array(readback.getMappedRange()).slice();
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
}

async function classifyPublishedSurface(device: GPUDevice,
  presentation: SparseWorldPresentation): Promise<{
    readonly receipt: Uint32Array;
    readonly minimumCubeX: number;
    readonly maximumCubeX: number;
  }> {
  const source = presentation.fineLevelSet;
  const shader = device.createShaderModule({
    label: "Sparse world public-presentation classifier regression",
    code: globalFineSurfaceClassificationShader,
  });
  const layout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 8, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" } },
    { binding: 9, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" } },
    { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 12, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" } },
    { binding: 16, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" } },
    { binding: 17, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" } },
  ] });
  const pipeline = await device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module: shader, entryPoint: "extractGlobalFineMain" },
  });
  const allocate = (label: string, size: number, usage: GPUBufferUsageFlags) =>
    device.createBuffer({ label, size, usage });
  const uniforms = allocate("Sparse world classifier uniforms", 112,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const uniformFloats = new Float32Array(28);
  uniformFloats[13] = source.plan.domainOrigin[1]
    + source.plan.sampleDimensions[1] * source.plan.fineCellWidth;
  device.queue.writeBuffer(uniforms, 0, uniformFloats);
  const params = allocate("Sparse world classifier compact params", 112,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(params, 0, packCompactFineLevelSetParams({
    sampleDimensions: source.plan.sampleDimensions,
    brickDimensions: source.plan.brickDimensions,
    brickResolution: source.plan.brickResolution,
    samplesPerBrick: source.plan.samplesPerBrick,
    pageCapacity: source.plan.maximumResidentBricks,
    fineFactor: source.plan.fineFactor,
    fineCellWidth: source.plan.fineCellWidth,
    domainOrigin: source.plan.domainOrigin,
    // The bridge's buffer ABI retains a publication-local generation until
    // Phase B moves the accepted generation into the device library header.
    generation: source.generation,
  }));
  const drawArgs = allocate("Sparse world classifier receipt", 32,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(drawArgs, 0,
    new Uint32Array([0, 1, 0, 0, 0, 0xffff_ffff, 0, 0xffff_ffff]));
  const recordCapacity = source.plan.maximumResidentBricks
    * source.plan.samplesPerBrick;
  const activeCubes = allocate("Sparse world classifier active cubes",
    8 * recordCapacity, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const cubeValues = allocate("Sparse world classifier values",
    32 * recordCapacity, GPUBufferUsage.STORAGE);
  const coarse = allocate("Sparse world classifier disabled coarse source", 64,
    GPUBufferUsage.STORAGE);
  const topology = allocate("Sparse world classifier disabled topology source", 32,
    GPUBufferUsage.STORAGE);
  const group = device.createBindGroup({ layout, entries: [
    { binding: 0, resource: { buffer: uniforms } },
    { binding: 4, resource: { buffer: drawArgs } },
    { binding: 5, resource: { buffer: activeCubes } },
    { binding: 6, resource: { buffer: cubeValues } },
    { binding: 8, resource: { buffer: source.worklist } },
    { binding: 9, resource: { buffer: source.samples } },
    { binding: 10, resource: { buffer: params } },
    { binding: 12, resource: { buffer: source.metadata } },
    { binding: 16, resource: { buffer: coarse } },
    { binding: 17, resource: { buffer: topology } },
  ] });
  try {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(...globalFineSurfaceDispatch(
      source.plan.maximumResidentBricks, source.plan.samplesPerBrick));
    pass.end();
    device.queue.submit([encoder.finish()]);
    const receipt = await readGPUWords(device, drawArgs, 8);
    const activeCount = Math.min(receipt[4]!, recordCapacity);
    const activeWords = activeCount > 0
      ? await readGPUWords(device, activeCubes, 2 * activeCount)
      : new Uint32Array();
    let minimumCubeX = Number.POSITIVE_INFINITY;
    let maximumCubeX = Number.NEGATIVE_INFINITY;
    for (let record = 0; record < activeCount; record += 1) {
      const x = activeWords[2 * record]! & 0xffff;
      minimumCubeX = Math.min(minimumCubeX, x);
      maximumCubeX = Math.max(maximumCubeX, x);
    }
    return { receipt, minimumCubeX, maximumCubeX };
  } finally {
    for (const buffer of [uniforms, params, drawArgs, activeCubes, cubeValues,
      coarse, topology]) buffer.destroy();
  }
}

async function publishedNegativeFront(device: GPUDevice,
  presentation: SparseWorldPresentation): Promise<{
    readonly pageX: number;
    readonly materialPageX: number;
    readonly residentPages: number;
    readonly worklistGeneration: number;
    readonly metadataGenerations: readonly number[];
    readonly negativeSamplesByPageX: readonly number[];
    readonly negativePagesByPageX: readonly number[];
  }> {
  const source = presentation.fineLevelSet;
  const capacity = source.plan.maximumResidentBricks;
  const directory = await readGPUWords(device, source.worklist, 7 + capacity);
  const residentPages = directory[1]!;
  assert.ok(residentPages <= capacity,
    `published resident pages ${residentPages} exceed capacity ${capacity}`);
  const pages = Array.from(directory.slice(7, 7 + residentPages));
  assert.equal(new Set(pages).size, pages.length,
    "the public presentation must publish each physical page once");

  const [metadata, samples] = await Promise.all([
    readGPUWords(device, source.metadata, 4 * capacity),
    readGPUWords(device, source.samples, source.plan.payloadCapacityBytes / 4),
  ]);
  let pageX = -1;
  const negativeSamplesByPageX = Array<number>(
    source.plan.brickDimensions[0]).fill(0);
  const negativePagesByPageX = Array<number>(
    source.plan.brickDimensions[0]).fill(0);
  const metadataGenerations = new Set<number>();
  for (const page of pages) {
    const pageGeneration = metadata[4 * page + 2]!;
    metadataGenerations.add(pageGeneration);
    assert.ok(pageGeneration > 0
      && pageGeneration <= presentation.acceptedGeneration,
    `page ${page} has invalid publication generation ${pageGeneration} for accepted ${
      presentation.acceptedGeneration}`);
    const first = page * source.plan.samplesPerBrick;
    const negativeCount = samples.subarray(first,
      first + source.plan.samplesPerBrick).reduce((count, sample) =>
      count + (((sample >>> 16) & 16) !== 0 ? 1 : 0), 0);
    if (negativeCount === 0) continue;
    const key = metadata[4 * page + 1]!;
    const x = (directory[3]! & FINE_LEVELSET_SIGNED_SPARSE_ADDRESS_FLAG) !== 0
      ? decodeSparseCM12SignedPresentationKey(key)[0]
      : key % source.plan.brickDimensions[0]!;
    pageX = Math.max(pageX, x);
    negativeSamplesByPageX[x]! += negativeCount;
    negativePagesByPageX[x]! += 1;
  }
  let materialPageX = -1;
  for (let x = 0; x < negativeSamplesByPageX.length; x += 1) {
    // One eighth of a B8 tile is intentionally permissive of a thin leading
    // sheet, but rejects the single stray negative sample that made the old
    // max-page gate report a visually stationary dam as having reached the wall.
    if (negativeSamplesByPageX[x]! >= 64) materialPageX = x;
  }
  return { pageX, materialPageX, residentPages,
    worklistGeneration: directory[0]! & 0x3fff_ffff,
    metadataGenerations: [...metadataGenerations].sort((left, right) => left - right),
    negativeSamplesByPageX, negativePagesByPageX };
}

dawnTest("public sparse world carries Long Dam's material front to the far wall",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-world-long-dam-dawn.test.ts");
    let device: GPUDevice | undefined;
    let solver: Awaited<ReturnType<NonNullable<
      typeof adaptiveMassMethod.createSolverAsync>>> | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU;
        globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([
        `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
        "enable-dawn-features=disable_blob_cache",
      ]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter, "Dawn must expose a WebGPU adapter");
      const rawDevice = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      device = managedGPUDevice(rawDevice, {
        requireWorkerRealm: false,
        maximumConcurrentBundles: 1,
      });
      const validationErrors: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        validationErrors.push(event.error.message);
      });

      const scene = createSparseCM12LongDamBreakScene();
      const values = resolveMethodValues(adaptiveMassMethod, "balanced",
        SPARSE_CM12_LONG_DAM_METHOD_PROFILE.overrides ?? {});
      solver = await adaptiveMassMethod.createSolverAsync!(
        device, scene, "balanced", values, undefined, () => {},
      );
      await (solver as typeof solver & {
        waitForSimulationReady(): Promise<void>;
      }).waitForSimulationReady();
      assert.ok("sparseWorld" in solver,
        "Sparse CM12 solver must expose the public sparse-world lifecycle");
      const world = (solver as typeof solver & { readonly sparseWorld: SparseWorld })
        .sparseWorld;

      const initialStatus = world.status();
      const initialPresentation = world.presentation();
      assert.equal(initialStatus.state, "ready");
      assert.equal(initialStatus.residentTiles, INITIAL_LONG_DAM_TILE_COUNT,
        "generation zero must contain exactly 80 wet tiles and their 196-tile "
          + "conservative velocity-extension support band");
      assert.equal(initialStatus.acceptedGeneration,
        initialPresentation.acceptedGeneration,
        "status and presentation must share one accepted-generation boundary");
      assert.ok(initialStatus.residentTiles < 24 * 12 * 4,
        "generation-zero physical residency must remain below the logical-domain volume");
      assert.ok(initialStatus.capacityTiles < 2 * 24 * 12 * 4,
        "the signed-world growth reserve must remain bounded near the logical domain");
      assert.equal(initialStatus.fault, undefined);
      const initialFront = await publishedNegativeFront(device, initialPresentation);
      assert.equal(initialFront.residentPages, INITIAL_LONG_DAM_TILE_COUNT,
        "generation-zero rendering must consume the complete wet/support topology");
      assert.equal(initialFront.pageX, 3,
        "the authored Long Dam reservoir must end at brick column three");
      assert.equal(initialFront.materialPageX, 3,
        "the authored reservoir must materially occupy brick column three");
      const initialClassification = await classifyPublishedSurface(
        device, initialPresentation);
      assert.equal(initialClassification.receipt[6], 1,
        "the renderer classifier must accept generation-zero presentation");
      assert.equal(initialClassification.receipt[5], 0,
        "a visible crossing must clear the retained-mesh sentinel");
      assert.ok(initialClassification.receipt[4]! > 0,
        "generation-zero presentation must contain visible liquid-air crossings");

      const gateSteps = Number(process.env.FLUID_SPARSE_WORLD_GATE_STEPS
        ?? LONG_DAM_GATE_STEPS);
      const frontTrajectory = [{ step: 0, ...initialFront }];
      let finalResidentPages = initialFront.residentPages;
      for (let step = 1; step <= gateSteps; step += 1) {
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        // Match the browser's two-deep presentation queue: encode and submit
        // the second step before fencing the pair. Serial per-step fences hide
        // overlapping-generation hazards present only in the UI-style path.
        if (step % 2 === 0 || step === gateSteps) {
          await device.queue.onSubmittedWorkDone();
        }
        if (step % LONG_DAM_CHECKPOINT_INTERVAL !== 0
          && step !== gateSteps) continue;

        const status = world.status();
        const presentation = world.presentation();
        assert.notEqual(status.state, "fault");
        assert.equal(status.fault, undefined);
        assert.equal(status.acceptedGeneration, presentation.acceptedGeneration,
          `step ${step} exposed mismatched simulation and presentation generations`);
        const published = await publishedNegativeFront(device, presentation);
        finalResidentPages = published.residentPages;
        assert.ok(published.materialPageX >= frontTrajectory.at(-1)!.materialPageX,
          `material Long Dam front retreated at step ${step}: ${
            frontTrajectory.at(-1)!.materialPageX} -> ${published.materialPageX}`);
        frontTrajectory.push({ step, ...published });
      }

      const finalStatus = world.status();
      const finalPresentation = world.presentation();
      const finalFront = frontTrajectory.at(-1)!;
      if (LONG_DAM_GATE_STEPS >= 180) {
        const reflectedNearSide = frontTrajectory.find(({ step }) => step === 150);
        assert.ok(reflectedNearSide,
          "the extended long-dam gate must sample the reflected crest at step 150");
        assert.ok(pageRangeSamples(reflectedNearSide, 0, 10)
            > 1.5 * pageRangeSamples(reflectedNearSide, 16, 24),
          "after far-wall impact the reflected crest must return to the near half");
        const finalFarInteriorMean = pageRangeSamples(finalFront, 16, 23) / 7;
        assert.ok(finalFront.negativeSamplesByPageX[LONG_DAM_FAR_WALL_PAGE_X]!
            > 1.15 * finalFarInteriorMean,
          "well after reflection the returning wave must retain its far-wall run-up");
      }
      if (process.env.FLUID_SPARSE_WORLD_GATE_TRACE === "1") {
        process.stderr.write(`[sparse-world-long-dam] ${JSON.stringify({
          frontTrajectory,
          finalStatus,
          fineSourceGeneration: finalPresentation.fineLevelSet.generation,
        })}\n`);
      }
      assert.equal(finalFront.materialPageX, LONG_DAM_FAR_WALL_PAGE_X,
        `material Long Dam front stopped before far-wall brick column ${
          LONG_DAM_FAR_WALL_PAGE_X}: ${JSON.stringify(frontTrajectory)}`);
      // Generation zero now carries the conservative dry support band. The
      // moving front reuses and retires those pages, so monotonic page-count
      // growth is no longer evidence of traversal. Require the far-wall
      // cross-section itself to be represented by several complete pages.
      assert.ok(finalFront.negativePagesByPageX[LONG_DAM_FAR_WALL_PAGE_X]! >= 4,
        "front motion must publish a complete far-wall cross-section");
      assert.ok(finalResidentPages > INITIAL_LONG_DAM_WET_TILE_COUNT,
        "the moving front must retain wet pages plus sparse transport support");
      assert.ok(finalStatus.residentTiles <= finalStatus.capacityTiles);
      assert.equal(finalStatus.acceptedGeneration,
        finalPresentation.acceptedGeneration);
      const finalClassification = await classifyPublishedSurface(
        device, finalPresentation);
      if (process.env.FLUID_SPARSE_WORLD_GATE_TRACE === "1") {
        process.stderr.write(`[sparse-world-long-dam-classification] ${JSON.stringify({
          activeCubes: finalClassification.receipt[4],
          minimumCubeX: finalClassification.minimumCubeX,
          maximumCubeX: finalClassification.maximumCubeX,
        })}\n`);
      }
      assert.equal(finalClassification.receipt[6], 1,
        "the renderer classifier must accept the final sparse presentation");
      assert.equal(finalClassification.receipt[5], 0,
        "the advanced front must publish a visible crossing");
      assert.ok(finalClassification.receipt[4]! > 0,
        "the advanced dam must retain a visible liquid-air surface");
      assert.ok(finalClassification.maximumCubeX >= LONG_DAM_FAR_WALL_PAGE_X * 8,
        `classified visible surface stopped at fine-cell x=${
          finalClassification.maximumCubeX}; expected a crossing in far-wall page ${
          LONG_DAM_FAR_WALL_PAGE_X}`);
      assert.deepEqual(validationErrors, []);
    } finally {
      solver?.destroy();
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
