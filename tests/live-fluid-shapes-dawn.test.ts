import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneAtContainerExtents } from "../lib/core/scene-scale";
import { sceneWithSolidStroke } from "../lib/core/solid-world";
import { parseScene } from "../lib/core/model";
import type { LiveFluidEdit } from "../lib/core/live-fluid-edit";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const modulePath = process.env.WEBGPU_NODE_MODULE;
const mass = (values: Float32Array) => values.reduce((sum, value) => sum + value, 0);

(modulePath ? test : test.skip)("live fluid shapes preserve the resident timeline, shape holes, solids and rejected state", { timeout: 240_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "tests/live-fluid-shapes-dawn.test.ts");
  let device: GPUDevice | undefined;
  let solver: WebGPUAdaptiveMassSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    let scene = sceneAtContainerExtents(sceneDocument(getSceneDefinition("water-box-tank-fill")),
      { width_m: 1.6, height_m: 2.4, depth_m: 1.6 });
    scene.rigidBodies = [];
    scene.systems = { ...scene.systems, fluid: true };
    scene.container.fillFraction = 0;
    delete scene.fluid.initialLiquidVolumes;
    scene.fluid.surfaceTension_N_m = 0;
    scene.voxelDomain.finestCellSize_m = .05;
    // Fully occupied interior cells at x=0, y=1.9, z=0 test collider exclusion.
    scene = sceneWithSolidStroke(scene, [{ operation: "fill", minimum: [12, 34, 12], maximumExclusive: [20, 42, 20], materialId: 2 }]);
    assert.doesNotThrow(() => parseScene(JSON.stringify(scene)), "the empty enabled-fluid fixture must be a valid scene document");
    solver = await WebGPUAdaptiveMassSolver.createAsync(device, scene, "balanced", undefined, {
      resolutionMode: "adaptive", brickFineResolution: 8, surfaceFineRings: 1,
      timeStep: "paper", pressureIterations: 32, pressureRelativeTolerance: 0,
    }, () => {});
    console.log("fluid-shape phase: constructor ready");
    await solver.waitForSimulationReady();
    console.log("fluid-shape phase: simulation pipelines ready");
    const world = solver.sparseWorld;
    // Read exactly the GPU-accepted scalar bank, including runtime WDR leaves.
    // The growth receipt uses max(A, B) for occupancy diagnostics and is not
    // a conserved-mass oracle once transport moves water between cells.
    const representedMass = async () => mass((await solver!.readDiagnosticFields(true)).density);
    const initialTime = solver.info.submittedTime_s;
    const initial = await solver.readDiagnosticFields(true);
    assert.equal(mass(initial.density), 0, "shape tests start with enabled fluid but no water");
    assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], [32, 48, 32],
      "the fixture's authored solid indices and shape positions require this lattice");
    const [hx, hy, hz] = [scene.container.width_m / solver.info.nx,
      scene.container.height_m / solver.info.ny, scene.container.depth_m / solver.info.nz];
    assert.ok([hx, hy, hz, solver.info.cellSize_m].every(value => Math.abs(value - .05) < 1e-12));
    const sample = (density: Float32Array, x: number, y: number, z: number) => {
      const ix = Math.floor((x + scene.container.width_m / 2) / hx);
      const iy = Math.floor(y / hy);
      const iz = Math.floor((z + scene.container.depth_m / 2) / hz);
      return density[ix + solver!.info.nx * (iy + solver!.info.ny * iz)]!;
    };
    const receipts: { shape: string; before: number; added: number; removed: number }[] = [];
    for (const [shape, x] of [["ball", -.5], ["cube", 0], ["torus", .5]] as const) {
      const edit: LiveFluidEdit = { operation: "add", shape, center_m: { x, y: 1.2, z: 0 }, radius_m: .2,
        // A two-cell inner radius leaves the sampled center voxel wholly dry.
        ...(shape === "torus" ? { tubeRadius_m: .05 } : {}) };
      const before = await solver.readDiagnosticFields(true);
      const beforeMass = mass(before.density);
      const addedResult = await solver.editFluid(edit);
      assert.equal(addedResult.accepted, true, `${shape}: ${addedResult.reason ?? "rejected"}`);
      assert.equal(solver.sparseWorld, world);
      assert.equal(solver.info.submittedTime_s, initialTime, "a stamp must not step or restart time");
      const added = await solver.readDiagnosticFields(true);
      const addedMass = mass(added.density);
      if (!(addedMass > beforeMass + 1)) {
        const activity = await solver.readGPUActivityPolicy();
        console.log(JSON.stringify({ failedShape: shape, addedResult, beforeMass, addedMass,
          info: solver.info, activity }, (_key, value) => value instanceof Float32Array ? { length: value.length } : value));
      }
      assert.ok(addedMass > beforeMass + 1, `${shape} must increase current liquid mass`);
      assert.ok(added.density.every(value => Number.isFinite(value) && value >= 0));
      if (shape === "torus") {
        assert.equal(sample(added.density, x, 1.2, 0), 0, "the torus center stays empty before any motion");
        assert.ok(sample(added.density, x + .125, 1.2, 0) > 0, "the torus tube contains water");
      } else {
        assert.ok(sample(added.density, x, 1.2, 0) > 0, `${shape} center is wet`);
        const corner = sample(added.density, x + .16, 1.36, .16);
        if (shape === "cube") assert.ok(corner > 0, "the cube retains its corners");
        else assert.equal(corner, 0, "the ball does not fill its bounding-box corners");
      }
      const removedResult = await solver.editFluid({ ...edit, operation: "remove" });
      assert.equal(removedResult.accepted, true, `${shape} remove: ${removedResult.reason ?? "rejected"}`);
      const removed = await solver.readDiagnosticFields(true);
      const removedMass = mass(removed.density);
      assert.ok(removedMass < addedMass, `${shape} removal reduces current liquid`);
      assert.ok(removed.density.every(value => Number.isFinite(value) && value >= 0));
      assert.equal(solver.sparseWorld, world);
      assert.equal(solver.info.submittedTime_s, initialTime);
      receipts.push({ shape, before: beforeMass, added: addedMass, removed: removedMass });
    }
    const solidBefore = await solver.readDiagnosticFields(true);
    const solidResult = await solver.editFluid({ operation: "add", shape: "cube", center_m: { x: 0, y: 1.9, z: 0 }, radius_m: .1 });
    assert.equal(solidResult.accepted, true, solidResult.reason);
    const solidAfter = await solver.readDiagnosticFields(true);
    assert.equal(sample(solidAfter.density, 0, 1.9, 0), 0, "solid cells cannot receive injected liquid");
    assert.ok(Math.abs(mass(solidAfter.density) - mass(solidBefore.density)) < 1e-6,
      "an edit wholly inside the solid must not manufacture mass elsewhere");

    for (const [edit, reason] of [
      [{ operation: "add", shape: "ball", center_m: { x: 100, y: 1, z: 0 }, radius_m: .1 }, /domain|inside|bounds/i],
      [{ operation: "add", shape: "cube", center_m: { x: 0, y: 1, z: 0 }, radius_m: .85 }, /budget|32768|capacity/i],
      [{ operation: "add", shape: "ball", center_m: { x: NaN, y: 1, z: 0 }, radius_m: .1 }, /finite|invalid/i],
      [{ operation: "add", shape: "torus", center_m: { x: 0, y: 1, z: 0 }, radius_m: .2, tubeRadius_m: .1 }, /torus|tube/i],
    ] as const) {
      const before = await solver.readDiagnosticFields(true);
      const generation = world.status().acceptedGeneration;
      const result = await solver.editFluid(edit);
      assert.equal(result.accepted, false);
      assert.match(result.reason ?? "", reason);
      const after = await solver.readDiagnosticFields(true);
      assert.deepEqual(after.density, before.density, "rejection must preserve the entire density field");
      assert.equal(world.status().acceptedGeneration, generation);
      assert.equal(solver.info.submittedTime_s, initialTime);
    }
    const advance = async (time: number) => {
      while (!solver!.advanceTo(time, [])) await new Promise(setImmediate);
      await device!.queue.onSubmittedWorkDone();
    };
    await advance(1 / 30);
    const movingTime = solver.info.submittedTime_s;
    assert.ok(movingTime! > initialTime!);
    const movingBefore = mass((await solver.readDiagnosticFields(true)).density);
    const live = await solver.editFluid({ operation: "add", shape: "ball", center_m: { x: 0, y: .6, z: 0 }, radius_m: .15 });
    assert.equal(live.accepted, true, live.reason);
    assert.equal(solver.info.submittedTime_s, movingTime);
    assert.equal(solver.sparseWorld, world);
    assert.ok(mass((await solver.readDiagnosticFields(true)).density) > movingBefore);
    // A dry initial scene has no wet template owners here: injection allocated
    // runtime pages. Later solid edits must refresh those page-local apertures.
    console.log("fluid-shape phase: three shapes and moving injection complete");
    const wet = await solver.readDiagnosticFields(true);
    assert.ok(sample(wet.density, 0, .6, 0) > 0);
    assert.equal(sample(wet.solidOpenFraction, 0, .6, 0), 1);
    const blocked = sceneWithSolidStroke(scene, [{ operation: "fill", minimum: [15, 11, 15], maximumExclusive: [18, 14, 18], materialId: 2 }]);
    const massBeforeSolid = await representedMass();
    const generationBeforeSolid = world.status().acceptedGeneration;
    console.log("fluid-shape phase: wet overlap query");
    await assert.rejects(solver.prepareLiveSolidEdit(blocked), /overlaps moving water/);
    assert.equal(world.status().acceptedGeneration, generationBeforeSolid,
      "wet rejection must not advance the authored topology generation");
    assert.equal(await representedMass(), massBeforeSolid);
    const rejectedFields = await solver.readDiagnosticFields(true);
    assert.deepEqual(rejectedFields.density, wet.density);
    assert.deepEqual(rejectedFields.solidOpenFraction, wet.solidOpenFraction);
    assert.equal(solver.sparseWorld, world);
    assert.equal(solver.info.submittedTime_s, movingTime);

    // Empty space in the same runtime page is editable without removing or
    // rewinding injected water. Its apertures publish before the next step.
    const dry = sceneWithSolidStroke(scene, [{ operation: "fill", minimum: [20, 11, 15], maximumExclusive: [21, 14, 18], materialId: 2 }]);
    const editStarted = performance.now();
    console.log("fluid-shape phase: dry atomic commit with concurrent frame");
    const pendingSolid = solver.prepareLiveSolidEdit(dry);
    // Submit the next ordinary step while the acceptance receipt is still
    // pending. GPU ordering must make it consume the atomic solid result.
    solver.advanceTo(2 / 30, []);
    const timeBeforeReceipt = solver.info.submittedTime_s;
    assert.ok(timeBeforeReceipt! > movingTime!, "solid acceptance must not block ordinary physics admission");
    await pendingSolid;
    console.log("fluid-shape phase: atomic receipt accepted");
    solver.applySceneUniforms(dry);
    const editElapsed_ms = performance.now() - editStarted;
    const massAfterSolidPublication = await representedMass();
    console.log("fluid-shape phase: post-publication mass read", massAfterSolidPublication);
    const blockedFields = await solver.readDiagnosticFields(true);
    assert.equal(sample(blockedFields.solidOpenFraction, .225, .6, 0), 0,
      "live solids must close dry injected-runtime cells before the next update");
    await advance(3 / 30);
    const massAfterSolidStep = await representedMass();
    console.log(JSON.stringify({ massBeforeSolid, massAfterSolidPublication, massAfterSolidStep,
      solidEditMassDelta: massAfterSolidStep - massBeforeSolid, editElapsed_ms, timeBeforeReceipt }));
    assert.ok(Math.abs(massAfterSolidPublication - massBeforeSolid) <= 1e-4,
      "publishing solid geometry must preserve represented fluid mass");
    assert.ok(Math.abs(massAfterSolidStep - massBeforeSolid) <= 1e-4,
      `dry solid insertion must conserve liquid: ${massBeforeSolid} -> ${massAfterSolidStep}`);
    assert.ok(solver.info.submittedTime_s! > movingTime!);
    const final = await solver.readDiagnosticFields(true);
    assert.ok(final.density.every(Number.isFinite));
    assert.ok(final.pressure.every(Number.isFinite));
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ fluidShapeReceipts: receipts, movingTime, finalTime: solver.info.submittedTime_s }));
  } finally {
    solver?.destroy();
    if (device) { await device.queue.onSubmittedWorkDone(); device.destroy(); }
    await releaseWebGPUExclusiveLock();
  }
});
