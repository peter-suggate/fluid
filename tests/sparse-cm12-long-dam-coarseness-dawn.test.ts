import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { resolveMethodValues } from "../lib/core/method-contract";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
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
import type { WebGPUFineLevelSetBrickSource } from
  "../lib/core/levelset-consumer-abi";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { initializeSparseBrickAtlasFromScene } from
  "../lib/methods/adaptive-mass/sparse-brick-atlas";
import {
  cloneSparseCM12TilePoolReceiver,
  createSparseCM12TileClonePool,
  sparseCM12TileClonePoolLookup,
  sparseCM12TileCloneSeedsFromBricks,
  SPARSE_CM12_TILE_CLONE_POOL_HEADER,
} from "../lib/methods/adaptive-mass/sparse-cm12-tile-clone-pool";
import { SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER as FPP_HEADER } from
  "../lib/methods/adaptive-mass/sparse-cm12-frame-plan-presentation";
import {
  adaptiveMassPresentationDimensionsForScene,
  WebGPUAdaptiveMassSolver,
} from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

async function readGPUWords(device: GPUDevice, source: GPUBuffer,
  wordCount: number, sourceOffset = 0): Promise<Uint32Array> {
  const readback = device.createBuffer({ size: 4 * wordCount,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
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

function negativePresentationPageFront(metadata: Uint32Array, samples: Uint32Array,
  physicalPages: readonly number[], brickDimensions: readonly number[],
  samplesPerPage: number): number {
  let front = -1;
  for (const page of physicalPages) {
    const first = page * samplesPerPage;
    const negative = samples.subarray(first, first + samplesPerPage).some((sample) =>
      ((sample >>> 16) & 16) !== 0);
    if (!negative) continue;
    front = Math.max(front, metadata[4 * page + 1]! % brickDimensions[0]!);
  }
  return front;
}

async function classifySparsePresentationSurface(
  device: GPUDevice,
  source: WebGPUFineLevelSetBrickSource,
  containerHeight_m: number,
): Promise<Uint32Array> {
  const shader = device.createShaderModule({
    label: "Long Dam sparse presentation classifier regression",
    code: globalFineSurfaceClassificationShader,
  });
  const layout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 16, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 17, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
  ] });
  const pipeline = await device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module: shader, entryPoint: "extractGlobalFineMain" },
  });
  const allocate = (label: string, size: number, usage: GPUBufferUsageFlags) =>
    device.createBuffer({ label, size, usage });
  const uniforms = allocate("Long Dam classifier uniforms", 112,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const uniformFloats = new Float32Array(28);
  uniformFloats[13] = containerHeight_m;
  device.queue.writeBuffer(uniforms, 0, uniformFloats);
  const params = allocate("Long Dam classifier compact params", 112,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const plan = source.plan;
  device.queue.writeBuffer(params, 0, packCompactFineLevelSetParams({
    sampleDimensions: plan.sampleDimensions,
    brickDimensions: plan.brickDimensions,
    brickResolution: plan.brickResolution,
    samplesPerBrick: plan.samplesPerBrick,
    pageCapacity: plan.maximumResidentBricks,
    fineFactor: plan.fineFactor,
    fineCellWidth: plan.fineCellWidth,
    domainOrigin: plan.domainOrigin,
    generation: source.generation,
  }));
  const drawArgs = allocate("Long Dam classifier receipt", 32,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(drawArgs, 0,
    new Uint32Array([0, 1, 0, 0, 0, 0xffff_ffff, 0, 0xffff_ffff]));
  // Counts are incremented before capacity checks, so a one-record payload is
  // sufficient to prove classifier admission without reproducing the renderer's
  // much larger mesh scratch allocation in this architecture regression.
  const activeCubes = allocate("Long Dam classifier first cube", 8,
    GPUBufferUsage.STORAGE);
  const cubeValues = allocate("Long Dam classifier first values", 32,
    GPUBufferUsage.STORAGE);
  const coarse = allocate("Long Dam classifier disabled coarse source", 64,
    GPUBufferUsage.STORAGE);
  const topology = allocate("Long Dam classifier disabled topology source", 32,
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
      plan.maximumResidentBricks, plan.samplesPerBrick));
    pass.end();
    device.queue.submit([encoder.finish()]);
    return await readGPUWords(device, drawArgs, 8);
  } finally {
    for (const buffer of [uniforms, params, drawArgs, activeCubes, cubeValues,
      coarse, topology]) buffer.destroy();
  }
}

dawnTest("long dam keeps deep work coarse and publishes new front receivers",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-long-dam-coarseness-dawn.test.ts");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU;
        globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([
        `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
        // First-load regression: a persistent pipeline cache would turn this
        // into a warm-start test and hide the UI's cold Metal compile seams.
        "enable-dawn-features=disable_blob_cache",
      ]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter, "Dawn must expose a WebGPU adapter");
      const rawDevice = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      // Match the browser exactly: resource owners receive the managed device
      // and every pipeline flows through one serial async compilation lane.
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
      // C0/C1 tile-clone cutover gate. Generation zero is authored directly
      // from wet bricks; capacity headroom is a physical slab, not pre-created
      // logical residency. The shipping solver assertions below remain the
      // front-motion gate while consumers move onto this ABI.
      const initialAtlas = initializeSparseBrickAtlasFromScene(scene, {
        finestDimensions: adaptiveMassPresentationDimensionsForScene(scene),
        brickFineResolution: 8,
        surfaceFineRings: 1,
      });
      const cloneSeeds = sparseCM12TileCloneSeedsFromBricks(initialAtlas.bricks);
      const clonePool = createSparseCM12TileClonePool(cloneSeeds);
      const cloneHeader = SPARSE_CM12_TILE_CLONE_POOL_HEADER;
      const logicalBrickCount = initialAtlas.brickDimensions.reduce(
        (product, value) => product * value, 1,
      );
      assert.equal(cloneSeeds.length, 80,
        "Long Dam generation zero must contain exactly its authored-fluid tiles");
      assert.equal(clonePool.words[cloneHeader.residentCount], cloneSeeds.length);
      assert.ok(clonePool.layout.capacity < logicalBrickCount,
        "tile-pool headroom must not expand to logical-domain volume");
      const seedCoordinates = new Set(cloneSeeds.map((seed) => seed.coordinate.join("/")));
      const firstReceiver = cloneSeeds.map((seed) => [seed.coordinate[0] + 1,
        seed.coordinate[1], seed.coordinate[2]] as [number, number, number])
        .find((coordinate) => coordinate[0] < initialAtlas.brickDimensions[0]
          && !seedCoordinates.has(coordinate.join("/")));
      assert.ok(firstReceiver, "Long Dam authored set must expose a dry receiver face");
      assert.equal(sparseCM12TileClonePoolLookup(clonePool, firstReceiver), undefined);
      const clonedPool = cloneSparseCM12TilePoolReceiver(clonePool, firstReceiver, 8);
      assert.notEqual(sparseCM12TileClonePoolLookup(clonedPool, firstReceiver), undefined);
      assert.equal(clonedPool.words[cloneHeader.residentCount], cloneSeeds.length + 1);
      assert.equal(clonedPool.words[cloneHeader.cloneCount], 1);

      const values = resolveMethodValues(adaptiveMassMethod, "balanced",
        SPARSE_CM12_LONG_DAM_METHOD_PROFILE.overrides ?? {});
      solver = await adaptiveMassMethod.createSolverAsync!(
        device, scene, "balanced", values, undefined, () => {},
      ) as WebGPUAdaptiveMassSolver;
      assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], [192, 96, 32]);
      assert.equal(solver.globalFineLevelSetSource.plan.maximumResidentBricks,
        clonePool.layout.capacity,
        "presentation allocation must follow physical pool capacity, not 1,152 dry keys");
      await solver.waitForSimulationReady();
      const initialPresentationPages =
        await solver.readPresentationPageAllocatorReceiptQA();
      assert.deepEqual(initialPresentationPages, {
        residentPages: 80, faultCode: 0, highWaterMark: 80,
        cloneCount: 0, capacity: clonePool.layout.capacity,
      });
      const fineSource = solver.globalFineLevelSetSource;
      const initialPresentationHeader = await readGPUWords(device,
        fineSource.worklist, 7);
      assert.deepEqual(Array.from(initialPresentationHeader.slice(0, 3)),
        [1, 80, clonePool.layout.capacity],
        "renderer publication must distinguish resident count from slab capacity");
      const initialPresentationMetadata = await readGPUWords(device,
        fineSource.metadata, 4 * clonePool.layout.capacity);
      const presentationControl = fineSource.presentationControl!;
      const initialFPP = await readGPUWords(device, presentationControl.buffer,
        32, presentationControl.offset ?? 0);
      const initialValidMetadata = Array.from({ length: 80 }, (_, page) =>
        initialPresentationMetadata[4 * page + 2]).filter((generation) =>
        generation === 1).length;
      assert.equal(initialValidMetadata, 80,
        `every initially resident renderer page must publish generation one: ${
          JSON.stringify({ fault: initialFPP[FPP_HEADER.faultCode],
            omitted: initialFPP[FPP_HEADER.omittedPageCount],
            dirty: initialFPP[FPP_HEADER.dirtyPageCount],
            executed: initialFPP[FPP_HEADER.executedPageCount],
            coverage: initialFPP[FPP_HEADER.coverageFaultCount],
            firstBrick: initialFPP[FPP_HEADER.firstFaultBrick],
            firstTile: initialFPP[FPP_HEADER.firstFaultTile],
            firstCause: initialFPP[FPP_HEADER.firstFaultCause] })}`);
      const initialPresentationSamples = await readGPUWords(device,
        fineSource.samples, fineSource.plan.payloadCapacityBytes / 4);
      assert.ok(initialPresentationSamples.some((sample) =>
        ((sample >>> 16) & 16) !== 0),
      "generation-zero presentation must contain negative liquid samples");
      const initialPresentationFront = negativePresentationPageFront(
        initialPresentationMetadata, initialPresentationSamples,
        Array.from({ length: 80 }, (_, page) => page),
        fineSource.plan.brickDimensions, fineSource.plan.samplesPerBrick);
      const initialSurfaceClassification = await classifySparsePresentationSurface(
        device, fineSource, scene.container.height_m);
      assert.equal(initialSurfaceClassification[6], 1,
        "renderer classifier must accept the compact generation-zero publication");
      assert.equal(initialSurfaceClassification[5], 0,
        "renderer classifier must clear the retained-mesh sentinel on a crossing");
      assert.ok(initialSurfaceClassification[4]! > 0,
        "renderer classifier must find generation-zero liquid-air crossings");

      const initial = await solver.readGPUActivityPolicy();
      const initialActive = initial.bricks.filter((brick) => brick.active);
      assert.equal(initialActive.length, cloneSeeds.length,
        "production generation zero must accept only authored-fluid tiles");
      assert.deepEqual(new Set(initialActive.map((brick) => brick.coordinate.join("/"))),
        new Set(cloneSeeds.map((seed) => seed.coordinate.join("/"))),
        "production generation-zero keys must equal the authored-fluid set");
      assert.ok(initialActive.some((brick) => brick.supportMask !== 0),
        "authored wet faces must seed receiver intent without allocating receivers");
      assert.ok(initialActive.some((brick) => brick.acceptedResolution < 8));
      assert.ok(initialActive.some((brick) => brick.acceptedResolution === 8));

      let firstStepActiveKeys = new Set<number>();
      let firstStepReceipt: unknown;
      for (let step = 1; step <= 2; step += 1) {
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        await device.queue.onSubmittedWorkDone();
        assert.deepEqual(validationErrors, [],
          `first advances must encode valid dynamic velocity-extension bindings`);
        if (step === 1) {
          const firstStep = await solver.readGPUActivityPolicy();
          firstStepActiveKeys = new Set(firstStep.bricks.filter((brick) => brick.active)
            .map((brick) => brick.key));
          firstStepReceipt = {
            steps: firstStep.acceptedSteps,
            topology: firstStep.acceptedTopologyGeneration,
            faults: firstStep.faultFlags,
            activated: firstStep.newlyActivatedBrickCount,
            prepared: firstStep.preparedBrickCount,
            committed: firstStep.committedBrickCount,
            commitFailed: firstStep.commitFailed,
            pending: firstStep.bricks.filter((brick) => !brick.active
              && (brick.candidateStatus !== 0 || brick.topologyPreparationScheduled))
              .slice(0, 12).map((brick) => ({ q: brick.coordinate,
                plan: brick.plannedResolution, candidate: brick.candidateResolution,
                status: brick.candidateStatus, page: brick.topologyPage })),
          };
        }
      }
      assert.ok(initialActive.every((brick) => firstStepActiveKeys.has(brick.key)),
        "the first activity census must retain every authored-fluid tile");
      const evolved = await solver.readGPUActivityPolicy();
      const active = evolved.bricks.filter((brick) => brick.active);
      const coarse = active.filter((brick) => brick.acceptedResolution < 8);
      const fine = active.filter((brick) => brick.acceptedResolution === 8);
      const deepBottomLeft = active.find((brick) =>
        brick.coordinate[0] === 0
          && brick.coordinate[1] === 0
          && brick.coordinate[2] === 1);
      const acceptedCells = active.reduce((sum, brick) =>
        sum + brick.acceptedResolution ** 3, 0);
      const allFineCells = active.length * 8 ** 3;
      const census = Object.fromEntries([1, 2, 4, 8].map((resolution) => [resolution,
        active.filter((brick) => brick.acceptedResolution === resolution).length]));
      const wetCensus = Object.fromEntries([1, 2, 4, 8].map((resolution) => [resolution,
        active.filter((brick) => brick.acceptedResolution === resolution
          && (brick.reasons & 64) !== 0).length]));

      assert.ok(coarse.length > 0,
        "surface activity must not erase every coarse rung in two frames");
      assert.ok(fine.length < active.length,
        "the active set must not collapse to blanket 8-cubed resolution");
      assert.ok(deepBottomLeft && deepBottomLeft.acceptedResolution < 8,
        `deep bottom-left bulk must stay coarse; got ${
          deepBottomLeft?.acceptedResolution ?? "no active brick"}`);
      assert.ok(acceptedCells <= 0.8 * allFineCells,
        `expected at least 20% active-cell reduction; got ${acceptedCells}/${allFineCells} ${
          JSON.stringify(census)} wet=${JSON.stringify(wetCensus)}`);

      const initialActiveKeys = new Set(initialActive.map((brick) => brick.key));
      const protectedNewKeys = new Set<number>();
      let finalSnapshot = evolved;
      let materialCenterAt24 = Number.NaN;
      const coordinateKey = (coordinate: readonly number[]) => coordinate.join("/");
      // Mirror brickRequestedAsReceiver: a target consumes the bit pointing
      // back from each active neighbour's immutable support snapshot.
      const requestedAsReceiver = (target: (typeof evolved.bricks)[number],
        bricks: typeof evolved.bricks): boolean => {
        const byCoordinate = new Map(bricks.map((brick) =>
          [coordinateKey(brick.coordinate), brick]));
        for (let dz = -1; dz <= 1; dz += 1) for (let dy = -1; dy <= 1; dy += 1) {
          for (let dx = -1; dx <= 1; dx += 1) {
            if (dx === 0 && dy === 0 && dz === 0) continue;
            const neighbour = byCoordinate.get(coordinateKey([
              target.coordinate[0] + dx,
              target.coordinate[1] + dy,
              target.coordinate[2] + dz,
            ]));
            if (!neighbour?.active) continue;
            const bit = 1 - dx + 3 * (1 - dy) + 9 * (1 - dz);
            if ((neighbour.supportMask & 2 ** bit) !== 0) return true;
          }
        }
        return false;
      };

      // Run far enough for the fast leading face to clone dry receivers. A
      // protected receiver must publish at B8 before transport targets it;
      // accepting its coarse construction rung changes the material flow.
      for (let step = 3; step <= 96; step += 1) {
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        await device.queue.onSubmittedWorkDone();
        const snapshot = await solver.readGPUActivityPolicy();
        finalSnapshot = snapshot;
        if (step === 24) {
          const density = (await solver.readDiagnosticFields()).density;
          let mass = 0, firstMoment = 0;
          for (let z = 0; z < 32; z += 1) for (let y = 0; y < 96; y += 1)
            for (let x = 0; x < 192; x += 1) {
              const rho = density[x + 192 * (y + 96 * z)]!;
              mass += rho;firstMoment += rho * (x + 0.5);
            }
          materialCenterAt24 = firstMoment / mass;
        }
        for (const brick of snapshot.bricks) {
          if (!brick.active || initialActiveKeys.has(brick.key)) continue;
          const movingSurface = (brick.reasons & (1 | 128)) === (1 | 128);
          const urgentReceiver = requestedAsReceiver(brick, snapshot.bricks)
            && ((brick.reasons & 64) === 0 || (brick.reasons & 128) !== 0);
          if (!movingSurface && !urgentReceiver) continue;
          protectedNewKeys.add(brick.key);
          assert.equal(brick.acceptedResolution, 8,
            `new front brick ${brick.coordinate.join(",")} was accepted at ${
              brick.acceptedResolution}^3 instead of 8^3 on step ${step}; acceptedSteps=${
              snapshot.acceptedSteps} activated=${brick.activatedStep} reasons=${
              brick.reasons} plan=${brick.plannedResolution}/${brick.planReasons} candidate=${
              brick.candidateResolution}/${brick.candidateStatus}`);
        }
      }
      const evolvedPresentationPages =
        await solver.readPresentationPageAllocatorReceiptQA();
      const evolvedPresentationWorklist = await readGPUWords(device,
        fineSource.worklist, 7 + clonePool.layout.capacity);
      const evolvedPresentationMetadata = await readGPUWords(device,
        fineSource.metadata, 4 * clonePool.layout.capacity);
      const evolvedDirectoryCount = evolvedPresentationWorklist[1]!;
      const evolvedDirectoryPages = Array.from(evolvedPresentationWorklist.slice(
        7, 7 + evolvedDirectoryCount));
      const evolvedDirectoryKeys = evolvedDirectoryPages.map((page) =>
        evolvedPresentationMetadata[4 * page + 1]!);
      assert.equal(evolvedPresentationWorklist[2], clonePool.layout.capacity);
      assert.equal(evolvedPresentationWorklist[4],
        Math.ceil(evolvedDirectoryCount / 64));
      assert.deepEqual(evolvedDirectoryKeys,
        [...evolvedDirectoryKeys].sort((left, right) => left - right),
        "runtime clone pages must publish through a key-sorted sparse directory");
      assert.equal(new Set(evolvedDirectoryPages).size, evolvedDirectoryPages.length,
        "the presentation directory must contain each physical page once");
      const evolvedPresentationSamples = await readGPUWords(device,
        fineSource.samples, fineSource.plan.payloadCapacityBytes / 4);
      const evolvedPresentationFront = negativePresentationPageFront(
        evolvedPresentationMetadata, evolvedPresentationSamples,
        evolvedDirectoryPages, fineSource.plan.brickDimensions,
        fineSource.plan.samplesPerBrick);
      assert.ok(evolvedPresentationFront > initialPresentationFront,
        `Long Dam rendered negative-phi front must advance; initial page x ${
          initialPresentationFront}, evolved page x ${evolvedPresentationFront}`);
      const evolvedSurfaceClassification = await classifySparsePresentationSurface(
        device, fineSource, scene.container.height_m);
      assert.equal(evolvedSurfaceClassification[6], 1,
        "renderer classifier must accept the runtime-cloned sparse directory");
      assert.equal(evolvedSurfaceClassification[5], 0,
        "runtime-cloned crossings must clear the retained-mesh sentinel");
      assert.ok(evolvedSurfaceClassification[4]! > 0,
        "the advanced dam must still classify a visible sparse surface");
      const finalActive = finalSnapshot.bricks.filter((brick) => brick.active);
      const finalNew = finalActive.filter((brick) => !initialActiveKeys.has(brick.key));
      assert.ok(protectedNewKeys.size > 0,
        `the long-dam front must enter a protected receiver; finalActive=${
          finalActive.length} new=${finalNew.length} pages=${
          JSON.stringify(evolvedPresentationPages)} epoch=${JSON.stringify({
            steps: finalSnapshot.acceptedSteps,
            topology: finalSnapshot.acceptedTopologyGeneration,
            faults: finalSnapshot.faultFlags,
            activated: finalSnapshot.newlyActivatedBrickCount,
            prepared: finalSnapshot.preparedBrickCount,
            committed: finalSnapshot.committedBrickCount,
            commitFailed: finalSnapshot.commitFailed,
          })} first=${JSON.stringify(firstStepReceipt)} newRungs=${JSON.stringify(
          Object.fromEntries([1, 2, 4, 8].map((rung) => [rung,
            finalNew.filter((brick) => brick.acceptedResolution === rung).length])))} `
          + `newReasons=${JSON.stringify(finalNew.slice(0, 12).map((brick) => ({
            q: brick.coordinate, rung: brick.acceptedResolution, reasons: brick.reasons,
            plan: brick.plannedResolution, status: brick.candidateStatus,
          })))}`);
      assert.ok(materialCenterAt24 >= 100,
        `Long Dam material flow regressed at paper step 24: center x=${
          materialCenterAt24}; HEAD receipt is x=104.405 and the coarse-receiver `
          + "regression was x=92.747");
      assert.equal(evolvedPresentationPages.faultCode, 0,
        `presentation allocator must remain within its physical working set: ${
          JSON.stringify(evolvedPresentationPages)} active=${finalActive.length}`);
      assert.ok(evolvedPresentationPages.cloneCount > 0,
        "the advancing front must allocate presentation pages on the GPU");
      assert.equal(evolvedPresentationPages.residentPages,
        80 + evolvedPresentationPages.cloneCount);
      assert.equal(evolvedPresentationPages.highWaterMark,
        evolvedPresentationPages.residentPages);
      assert.ok(evolvedPresentationPages.highWaterMark
        < initialAtlas.brickDimensions.reduce((product, value) => product * value, 1),
      "presentation high-water must remain below logical-domain volume");
      assert.deepEqual(validationErrors, []);
    } finally {
      solver?.destroy();
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
