import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod, adaptiveMassSolverOptions } from
  "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { cloneScene, defaultScene } from "../lib/core/model";
import { SPARSE_CM12_RESIDENT_STAGES, SPARSE_CM12_RESIDENT_STAGE_SUBSTAGES, type SparseCM12ResidentStageId } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import { unpackFineLevelSetPackedPhi } from "../lib/core/fine-levelset-packed-sample";

type Activity = Awaited<ReturnType<WebGPUAdaptiveMassSolver["readGPUActivityPolicy"]>>;

const modulePath = process.env.WEBGPU_NODE_MODULE;
const dawnTest = modulePath ? test : test.skip;

dawnTest("frozen mini32 admits dry support, retains every accepted rung, and resumes adaptation", {
  timeout: 180_000,
}, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "sparse-cm12-dynamic-freeze");
  let device: GPUDevice | undefined;
  let solver: WebGPUAdaptiveMassSolver | undefined;
  const live = new Set<GPU>();
  Object.assign(globalThis, { dynamicFreezeGPU: live });
  try {
    const dawn = await import(pathToFileURL(modulePath!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu: GPU = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    live.add(gpu);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => {
      event.preventDefault(); errors.push(event.error.message);
    });
    const scene = sceneDocument(getSceneDefinition("minimal-power-dam-break-32"));
    const dt = 1 / 30;
    scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt;
    const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
      selectorMode: "coarse-first", timeStep: "scene",
    });
    solver = await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(
      device, scene, "balanced", undefined, adaptiveMassSolverOptions(values), () => {});
    await solver.waitForSimulationReady();
    solver.setTopologyFrozen(true);
    const initial = await solver.readGPUActivityPolicy();
    const resident = new Map(initial.bricks.filter(b => b.active).map(b =>
      [b.coordinate.join("/"), { span: b.spanBricks, resolution: b.acceptedResolution }]));
    assert.equal(resident.size, 60);
    const initialCount = resident.size;
    let enteredDryCorner = false;
    for (let step = 1; step <= 12; step++) {
      while (!solver.advanceTo(step * dt, [])) await new Promise(setImmediate);
      await solver.waitForTopologyReady();
      await solver.assertSimulationHealthy();
      const activity: Activity = await solver.readGPUActivityPolicy();
      assert.equal(activity.faultFlags, 0);
      assert.equal(activity.commitFailed, false);
      for (const [coordinate, rung] of resident) {
        const current = activity.bricks.find(b => b.coordinate.join("/") === coordinate);
        assert.ok(current?.active, `${coordinate} must remain resident`);
        assert.deepEqual({ span: current.spanBricks, resolution: current.acceptedResolution }, rung,
          `step ${step}: ${coordinate} must retain its accepted cell sizes`);
      }
      for (const brick of activity.bricks.filter(b => b.active)) {
        const coordinate = brick.coordinate.join("/");
        if (!resident.has(coordinate)) resident.set(coordinate, {
          span: brick.spanBricks, resolution: brick.acceptedResolution,
        });
      }
      const fields = await solver.readDiagnosticFields();
      assert.ok(fields.density.every(Number.isFinite));
      for (let z = 24; z < 32; z++) for (let y = 0; y < 32; y++) {
        for (let x = 24; x < 32; x++) {
          enteredDryCorner ||= fields.density[x + 32 * (y + 32 * z)]! > 0.001;
        }
      }
    }
    assert.ok(resident.size > initialCount, "the advancing front must activate new support");
    assert.ok(enteredDryCorner, "nonzero liquid must use the newly activated corner");
    solver.setTopologyFrozen(false);
    const edited = structuredClone(scene);
    edited.fluid.refinementRegions = [{ id: "resume-refinement", rule: "minimum-cell-size",
      minimumCellSize_cells: 1, maximumCellSize_cells: 1,
      min_m: { x: -1, y: -1, z: -1 }, max_m: { x: 1, y: 1, z: 1 } }];
    solver.applySceneUniforms(edited);
    while (!solver.advanceTo(13 * dt, [])) await new Promise(setImmediate);
    await solver.waitForTopologyReady();
    await solver.assertSimulationHealthy();
    const unfrozen = await solver.readGPUActivityPolicy();
    assert.ok(unfrozen.bricks.some(b => b.active
      && resident.get(b.coordinate.join("/"))?.resolution !== b.acceptedResolution),
    "releasing the toggle must allow existing cells to refine again");
    assert.deepEqual(errors, []);
  } finally {
    solver?.destroy(); device?.destroy(); live.clear();
    await releaseWebGPUExclusiveLock();
  }
});

dawnTest("a paused frozen coarse host accepts superseding signed-world drops through compiled mixed seams", {
  timeout: 180_000,
}, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "sparse-cm12-frozen-world-growth");
  let device: GPUDevice | undefined;
  let solver: WebGPUAdaptiveMassSolver | undefined;
  const live = new Set<GPU>();
  Object.assign(globalThis, { frozenWorldGrowthGPU: live });
  try {
    const dawn = await import(pathToFileURL(modulePath!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu: GPU = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    live.add(gpu);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => {
      event.preventDefault(); errors.push(event.error.message);
    });
    const scene = cloneScene(defaultScene);
    scene.sceneId = "frozen-coarse-host-world-growth";
    scene.rigidBodies = [];
    scene.solidVoxels = [];
    scene.scenery = undefined;
    scene.environment = "stage";
    scene.container = { ...scene.container, width_m: 0.8, height_m: 0.8,
      depth_m: 0.8, fillFraction: 0.25, vessel: "none" };
    scene.voxelDomain.finestCellSize_m = 0.05;
    scene.fluid.initialCondition = "tank-fill";
    scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
    scene.fluid.refinementRegions = [];
    const dt = 1 / 30;
    scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt;
    const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
      selectorMode: "coarse-first", timeStep: "scene",
    });
    if (process.env.FREEZE_DIAGNOSTICS) console.log("freeze solver construction begins");
    solver = await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(device, scene,
      "balanced", undefined, { ...adaptiveMassSolverOptions(values), initialResolutionForQA: 2,
        initialAtlasResidentForQA: true }, () => {});
    if (process.env.FREEZE_DIAGNOSTICS) console.log("freeze solver construction complete");
    await solver.waitForSimulationReady();
    if (process.env.FREEZE_DIAGNOSTICS) console.log("freeze pipelines ready");
    solver.setTopologyFrozen(true);
    const original = (await solver.readGPUActivityPolicy()).bricks.filter(b => b.active);
    assert.equal(original.length, 8);
    assert.ok(original.every(b => b.acceptedResolution === 2));
    const readMass = async () => {
      const source = solver!.fieldSnapshotSourceForQA;
      const activity = await solver!.readGPUActivityPolicy();
      const copy = device!.createBuffer({ size: source.state.size + 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        const encoder = device!.createCommandEncoder();
        encoder.copyBufferToBuffer(source.state, 0, copy, 0, source.state.size);
        encoder.copyBufferToBuffer(source.topologyArena,
          4 * (source.frameControlBaseWords + source.scalarParityWord), copy, source.state.size, 4);
        device!.queue.submit([encoder.finish()]);
        await copy.mapAsync(GPUMapMode.READ);
        const words = new Uint32Array(copy.getMappedRange());
        const floats = new Float32Array(words.buffer);
        const density = words[source.state.size / 4]! & 1
          ? source.layout.densityB : source.layout.densityA;
        const template = source.templateWords;
        const geometry = new Float32Array(template.buffer, template.byteOffset, template.length);
        let total = 0, outside = 0, positiveOutside = 0, maxDensity = 0, maxSpeed = 0, fastestCell = -1;
        let fastestPosition: number[] | undefined;
        const velocity = words[source.state.size / 4]! & 1
          ? source.layout.cellVelocityB : source.layout.cellVelocityA;
        for (const brick of activity.bricks.filter(b => b.active)) {
          const dynamic = brick.topologyPage !== undefined;
          const range = template[11]! + 2 * (4 * brick.leafId + Math.log2(brick.acceptedResolution));
          const first = dynamic ? template[2]! + 512 * brick.topologyPage! : template[range]!;
          const count = dynamic ? 512 : template[range + 1]!;
          for (let at = first; at < first + count; at++) {
            const mass = floats[density + at]! * (dynamic ? 1 : geometry[template[6]! + 8 * at + 3]!);
            total += mass;
            if (brick.coordinate[0] < 0) outside += mass;
            if (brick.coordinate[0] >= 2) positiveOutside += mass;
            maxDensity = Math.max(maxDensity, floats[density + at]!);
            const speed = Math.hypot(...floats.subarray(velocity + 4 * at, velocity + 4 * at + 3));
            if (speed > maxSpeed) {
              maxSpeed = speed; fastestCell = at;
              fastestPosition = dynamic
                ? brick.coordinate.map((q, axis) => 8 * q + ((at - first) >> (3 * axis) & 7) + 0.5)
                : Array.from(geometry.subarray(template[6]! + 8 * at, template[6]! + 8 * at + 3));
            }
          }
        }
        return { total, outside, positiveOutside, maxDensity, maxSpeed, fastestCell, fastestPosition };
      } finally { copy.unmap(); copy.destroy(); }
    };
    const before = await readMass();
    if (process.env.FREEZE_DIAGNOSTICS) console.log("freeze injection begins", solver.info.topologyGenerationCount);
    solver.injectLiquidBall({ centre_m: { x: -0.4, y: 0.6, z: 0 }, radius_m: 0.12 });
    // This edit arrives while the first readiness check is in flight. Its
    // separate signed support must be included before either queued dose runs.
    solver.injectLiquidBall({ centre_m: { x: 0.4, y: 0.6, z: 0 }, radius_m: 0.12 });
    await solver.waitForTopologyReady();
    if (process.env.FREEZE_DIAGNOSTICS) console.log("freeze injection ready", solver.info.topologyGenerationCount);
    await solver.assertSimulationHealthy();
    assert.equal(solver.info.encodedSteps, 0, "a paused edit must not advance physics");
    assert.ok((solver.info.topologyGenerationCount ?? 0) > 0,
      "the coarse/fine world connection must use a compiled replacement graph");
    assert.equal(solver.info.topologyGenerationPending, false);
    const grown = await solver.readGPUActivityPolicy();
    for (const brick of original) {
      const retained = grown.bricks.find(b => b.coordinate.join("/") === brick.coordinate.join("/"));
      assert.ok(retained?.active);
      assert.equal(retained.acceptedResolution, brick.acceptedResolution);
      assert.equal(retained.spanBricks, brick.spanBricks);
    }
    assert.ok(grown.bricks.some(b => b.active && b.coordinate[0] < 0));
    assert.ok(grown.bricks.some(b => b.active && b.coordinate[0] >= 2));
    const after = await readMass();
    assert.ok(after.outside > 1, `deferred support lost its drop dose: ${JSON.stringify(after)}`);
    assert.ok(after.positiveOutside > 1, `the superseding edit lost its drop dose: ${JSON.stringify(after)}`);
    assert.ok(after.total > before.total + 1);
    const signedPresentationLine: Record<number, number> = {};
    {
      const source = solver.globalFineLevelSetSource;
      const buffers = [source.metadata, source.samples];
      const copies = buffers.map(buffer => device!.createBuffer({ size: buffer.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
      try {
        const encoder = device.createCommandEncoder();
        buffers.forEach((buffer, index) => encoder.copyBufferToBuffer(buffer, 0, copies[index]!, 0, buffer.size));
        device.queue.submit([encoder.finish()]);
        await Promise.all(copies.map(copy => copy.mapAsync(GPUMapMode.READ)));
        const metadata = new Uint32Array(copies[0]!.getMappedRange());
        const samples = new Uint32Array(copies[1]!.getMappedRange());
        const r = source.plan.brickResolution;
        for (let page = 0; page < source.plan.maximumResidentBricks; page++) {
          const key = metadata[4 * page + 1]!;
          const bx = (key & 0x7ff) - 1024;
          const by = ((key >>> 11) & 0x3ff) - 512;
          const bz = ((key >>> 21) & 0x7ff) - 1024;
          const y = 11 - by * r, z = 7 - bz * r;
          if (y < 0 || y >= r || z < 0 || z >= r || bx * r >= 0 || bx * r < -8) continue;
          for (let x = 0; x < r; x++) signedPresentationLine[bx * r + x] = unpackFineLevelSetPackedPhi(
            samples[page * source.plan.samplesPerBrick + x + r * (y + r * z)]!);
        }
        if (process.env.FREEZE_SAMPLE_DIAGNOSTICS) console.log("freeze signed presentation line", signedPresentationLine);
      } finally { copies.forEach(copy => { copy.unmap(); copy.destroy(); }); }
    }
    assert.ok(signedPresentationLine[-8]! > 0.15 && signedPresentationLine[-7]! > 0.15,
      "the signed world apron must publish air away from the drop");
    assert.ok(signedPresentationLine[-3]! > 0 && signedPresentationLine[-2]! < 0
      && signedPresentationLine[-1]! < 0,
    `the signed field must contour its own liquid body: ${JSON.stringify(signedPresentationLine)}`);
    await solver.waitForTopologyReady();
    assert.deepEqual(await readMass(), after, "waiting again must not replay the accepted dose");
    const steps = process.env.FREEZE_DIAGNOSTICS ? Number(process.env.FREEZE_STEPS ?? 8) : 8;
    for (let step = 1; step <= steps; step++) {
      if (process.env.FREEZE_DIAGNOSTICS && step === steps && process.env.FREEZE_STAGE_LIMIT) {
        const limit = process.env.FREEZE_STAGE_LIMIT as SparseCM12ResidentStageId;
        assert.ok(SPARSE_CM12_RESIDENT_STAGES.includes(limit));
        solver.sparseWorldTrace.setStageLimitForQA(limit);
        if (process.env.FREEZE_CANDIDATE_LIMIT) {
          assert.ok((SPARSE_CM12_RESIDENT_STAGE_SUBSTAGES["candidate-transfer"] as readonly string[])
            .includes(process.env.FREEZE_CANDIDATE_LIMIT));
          solver.sparseWorldTrace.setCandidatePhaseLimitForQA(process.env.FREEZE_CANDIDATE_LIMIT as
            Parameters<typeof solver.sparseWorldTrace.setCandidatePhaseLimitForQA>[0]);
        }
      }
      const deadline = performance.now() + 30_000;
      while (!solver.advanceTo(step * dt, [])) {
        assert.ok(performance.now() < deadline, `step ${step} never became ready: ${JSON.stringify(solver.info)}`);
        await new Promise(setImmediate);
      }
      if (process.env.FREEZE_DIAGNOSTICS) console.log("freeze step submitted", step, solver.info.topologyGenerationCount);
      await solver.waitForTopologyReady();
      if (process.env.FREEZE_DIAGNOSTICS) console.log("freeze step ready", step, solver.info.topologyGenerationCount);
      await solver.assertSimulationHealthy();
      if (process.env.FREEZE_DIAGNOSTICS) console.log("freeze native fields", step, await readMass());
      const activity: Activity = await solver.readGPUActivityPolicy();
      if (process.env.FREEZE_DIAGNOSTICS && process.env.FREEZE_CANDIDATE_LIMIT) {
        console.log("freeze candidate activity", {
          prepared: activity.preparedBrickCount, committed: activity.committedBrickCount,
          pending: activity.bricks.filter(b => b.planReasons !== 32 && (!b.active || b.topologyPage !== undefined)),
        });
        console.log("freeze world receipt", await solver.readWorldGrowthReceiptQA());
        const source = solver.fieldSnapshotSourceForQA;
        const size = source.acceptedIndirectArguments.size + 4 * (32 + 24);
        const copy = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        try {
          const encoder = device.createCommandEncoder();
          encoder.copyBufferToBuffer(source.topologyArena, 4 * source.topologyWorklistBaseWords, copy, 0, 4 * 32);
          encoder.copyBufferToBuffer(source.topologyArena, 4 * source.acceptedLeafManifestBaseWords, copy, 4 * 32, 4 * 24);
          encoder.copyBufferToBuffer(source.acceptedIndirectArguments, 0, copy, 4 * 56, source.acceptedIndirectArguments.size);
          device.queue.submit([encoder.finish()]); await copy.mapAsync(GPUMapMode.READ);
          const words = new Uint32Array(copy.getMappedRange());
          console.log("freeze topology headers", { worklist: Array.from(words.subarray(0, 32)),
            leafManifest: Array.from(words.subarray(32, 56)), indirect: Array.from(words.subarray(56)),
            cellCapacity: source.cellCapacity, rowCapacity: source.rowCapacity });
        } finally { copy.unmap(); copy.destroy(); }
      }
      assert.equal(activity.faultFlags, 0);
      assert.equal(activity.commitFailed, false);
      for (const brick of grown.bricks.filter(b => b.active)) {
        const retained = activity.bricks.find(b => b.coordinate.join("/") === brick.coordinate.join("/"));
        assert.ok(retained?.active);
        assert.equal(retained.acceptedResolution, brick.acceptedResolution);
        assert.equal(retained.spanBricks, brick.spanBricks);
      }
    }
    const final = await readMass();
    assert.ok(Number.isFinite(final.total) && Number.isFinite(final.maxDensity)
      && Number.isFinite(final.maxSpeed), `stationary fields must remain finite: ${JSON.stringify(final)}`);
    assert.ok(Math.abs(final.total - after.total) <= 1e-5 * after.total,
      `zero-gravity transport must conserve mass: ${JSON.stringify({ after, final })}`);
    assert.ok(final.maxDensity <= 1.0001,
      `stationary liquid must retain its bounded density: ${JSON.stringify(final)}`);
    assert.ok(final.maxSpeed < 1e-5,
      `an unforced liquid must remain stationary: ${JSON.stringify(final)}`);
    assert.deepEqual(errors, []);
  } finally {
    solver?.destroy(); device?.destroy(); live.clear();
    await releaseWebGPUExclusiveLock();
  }
});
