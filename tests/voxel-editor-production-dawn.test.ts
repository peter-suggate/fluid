import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createEmptyScene } from "../lib/core/empty-scene";
import { sceneWithSolidStroke } from "../lib/core/solid-world";
import { FluidLabRenderer } from "../lib/core/webgpu-renderer";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { WebGPULiveSvoScene } from "../lib/svo/features/scene-publication/webgpu-live-svo-scene";
import { simulation } from "../lib/core/simulation/controller";
import { voxelTools } from "../lib/core/voxel-editor/registry";
import { beginToolTransaction } from "../lib/core/voxel-editor/transaction";
import { createWorkerSolidEditAcceptance } from "../lib/core/voxel-editor/worker-solid-edit-acceptance";
import type { EditorRay } from "../lib/core/editor-entity";
import type { SceneDescription } from "../lib/core/model";
import { readGpuSolidFractions } from "./helpers/solid-world-gpu-probe";
import type { ToolValues } from "../lib/core/voxel-editor/plugin";

const modulePath = process.env.WEBGPU_NODE_MODULE;
const expectedSolids = ["build", "carve", "box", "cut", "sphere", "drill", "wall", "channel"];
const settle = () => new Promise<void>(resolve => setImmediate(resolve));

test("production editor plugins, worker acceptance and controller history preserve live GPU authority", {
  skip: !modulePath, timeout: 240_000,
}, async () => {
  assert.deepEqual(voxelTools.tools.filter(tool => tool.execution !== "release").map(tool => tool.id), expectedSolids,
    "new solid plugins must be added to the production-path acceptance matrix");
  await acquireWebGPUExclusiveLock("dawn-test", "voxel-editor-production");
  let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined, display: WebGPULiveSvoScene | undefined;
  let detach: (() => void) | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" }); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    let initial = createEmptyScene({ extents_m: { x: 1.6, y: 2.4, z: 1.6 }, finestCellSize_m: .05 });
    initial.systems = { ...initial.systems, fluid: true };
    initial = sceneWithSolidStroke(initial, [{ operation: "fill", minimum: [10, 10, 10], maximumExclusive: [17, 17, 17] }]);
    solver = await WebGPUAdaptiveMassSolver.createAsync(device, initial, "balanced", undefined, {
      resolutionMode: "adaptive", brickFineResolution: 8, surfaceFineRings: 1,
      timeStep: "paper", pressureIterations: 32, pressureRelativeTolerance: 0,
    }, () => {});
    await solver.waitForSimulationReady();
    display = await WebGPULiveSvoScene.create(device, initial, "balanced", () => {});
    const world = solver.sparseWorld, source = display.sparseVoxelSceneSource;
    const renderer = new FluidLabRenderer({} as HTMLCanvasElement, () => {});
    // Real acceptance and both real GPU consumers; canvas presentation is not
    // needed to exercise the exact worker publication and history protocol.
    Object.assign(renderer, { gpuFluid: solver, svoSceneSidecar: display });
    let workerScene = { document: initial, revision: 1 };
    const worker = createWorkerSolidEditAcceptance({
      readScene: () => workerScene,
      writeScene: next => { workerScene = next; },
      accept: (next, current) => renderer.acceptLiveSolidEdit(next, current),
    });
    const session = simulation.session();
    session.method.getState().setMethodId("adaptive-mass");
    session.scene.getState().setScene(initial, "production-native");
    session.history.getState().clear(); session.ui.setState({ voxelStrokePending: false });
    let advanceDuringAcceptance = false;
    const accept = async (next: SceneDescription, base: SceneDescription) => {
      const receipt = worker.accept(structuredClone({ scene: next, base }));
      if (advanceDuringAcceptance) {
        advanceDuringAcceptance = false;
        assert.equal(session.ui.getState().voxelStrokePending, true);
        await advance();
      }
      await receipt;
      assert.deepEqual(workerScene.document.solidVoxels, next.solidVoxels, "worker publishes before receipt");
    };
    detach = simulation.registerLiveSolidEditAcceptance(session.id, accept);
    const ray = (x = -.125, z = -.125): EditorRay => ({ origin: { x, y: 4, z }, direction: { x: 0, y: -1, z: 0 } });
    const maintain = async () => {
      const encoder = device!.createCommandEncoder(); display!.encodeSceneMaintenance(encoder);
      device!.queue.submit([encoder.finish()]); await device!.queue.onSubmittedWorkDone();
      assert.equal(solver!.sparseWorld, world); assert.equal(display!.sparseVoxelSceneSource, source);
      assert.deepEqual(errors, []);
    };
    const start = (id: string, input = ray(), values: ToolValues = { size: 3, depth: 3, shell: 1 }) => {
      const plugin = voxelTools.get(id)!; assert.ok(plugin);
      const base = { scene: session.scene.getState().scene, presetId: "production-native", label: plugin.ui.label };
      const transaction = beginToolTransaction(plugin, {
        scene: () => session.scene.getState().scene,
        publish: async (next, original) => {
          const before = session.scene.getState().scene;
          await accept(next, original);
          if (session.scene.getState().scene !== before) throw new Error("Scene changed during the stroke.");
          session.scene.getState().setScene(next);
        },
        execute: async action => { const result = await renderer.editFluid(action.edit); assert.equal(result.accepted, true, result.reason); },
        begin: () => session.ui.setState({ voxelStrokePending: true }),
        finish: () => { if (session.scene.getState().scene !== base.scene) session.history.getState().record(base); session.ui.setState({ voxelStrokePending: false }); },
        cancel: () => session.ui.setState({ voxelStrokePending: false }),
      }, input, values);
      assert.ok(transaction, `${id} must hit the seeded top face`);
      return transaction;
    };
    const history = async (direction: "undo" | "redo") => {
      assert.equal(simulation[direction](), true);
      const deadline = performance.now() + 15000;
      while (session.ui.getState().voxelStrokePending) { assert.ok(performance.now() < deadline, `${direction} timed out`); await settle(); }
      await maintain();
    };
    const advance = async () => {
      const target = (solver!.info.submittedTime_s ?? 0) + 1 / 30;
      const deadline = performance.now() + 15000;
      while (!solver!.advanceTo(target, [])) { assert.ok(performance.now() < deadline, "simulation failed to advance"); await settle(); }
      await device!.queue.onSubmittedWorkDone();
    };
    const solidFractions = () => readGpuSolidFractions(device!, world, [solver!.info.nx, solver!.info.ny, solver!.info.nz]);
    await advance();
    const concurrentTime = solver.info.submittedTime_s ?? 0;
    advanceDuringAcceptance = true;
    const concurrentEdit = start("box"); await concurrentEdit.update(ray()); await concurrentEdit.finish();
    assert.ok((solver.info.submittedTime_s ?? 0) > concurrentTime, "ordinary physics advances while acceptance is pending");
    await history("undo");
    for (const mode of ["empty-running", "moving-water"]) {
      if (mode === "moving-water") {
        const water = start("fluid-ball", ray(-.5, -.5), { size: 4, height: 2 });
        await water.update(ray(-.5, -.5)); await water.finish(); await advance();
      }
      for (const id of expectedSolids) {
        const before = await solver.readDiagnosticFields(true);
        const solidBefore = await solidFractions();
        const beforeScene = session.scene.getState().scene;
        const beforeTime: number | undefined = solver.info.submittedTime_s;
        const transaction = start(id);
        const update = await transaction.update(ray(-.025, -.075)); await transaction.finish(); await maintain();
        assert.ok(update);
        const solidEdited = await solidFractions();
        const affected = new Set<number>();
        for (const patch of update.patches) {
          for (let z = patch.minimum[2]; z < patch.maximumExclusive[2]; z++)
            for (let y = patch.minimum[1]; y < patch.maximumExclusive[1]; y++)
              for (let x = patch.minimum[0]; x < patch.maximumExclusive[0]; x++) {
                const index: number = x + solver.info.nx * (y + solver.info.ny * z);
                affected.add(index);
                assert.equal(solidEdited[index], patch.operation === "fill" ? 1 : 0, `${mode}/${id}: exact GPU cell ${x},${y},${z}`);
              }
        }
        assert.ok(solidEdited.every((value, index) => affected.has(index) || value === solidBefore[index]), `${id}: untouched GPU cells stay unchanged`);
        const edited = await solver.readDiagnosticFields(true);
        assert.ok(solidEdited.some((value, index) => value !== solidBefore[index]), `${mode}/${id}: immediate GPU boundary change`);
        assert.ok(edited.density.every((value, index) => value === before.density[index]), `${mode}/${id}: solid editing preserves liquid`);
        assert.equal(solver.info.submittedTime_s, beforeTime);
        await history("undo");
        assert.deepEqual(session.scene.getState().scene.solidVoxels, beforeScene.solidVoxels, `${id}: Undo document`);
        assert.ok((await solidFractions()).every((value, index) => value === solidBefore[index]), `${id}: Undo GPU`);
        await history("redo");
        assert.ok((await solidFractions()).every((value, index) => value === solidEdited[index]), `${id}: Redo GPU`);
        await history("undo");
      }
      await advance();
    }
    const fluidIds = ["fluid-ball", "fluid-cube", "fluid-torus"];
    assert.deepEqual(voxelTools.tools.filter(tool => tool.execution === "release").map(tool => tool.id), fluidIds);
    for (const id of fluidIds) {
      const input = ray(.45, .45);
      const values = { size: id === "fluid-torus" ? 8 : 6, thickness: 2, height: 22 };
      const mass = async () => (await solver!.readDiagnosticFields(true)).density.reduce((a, b) => a + b, 0);
      const before = await mass(), historyBefore = session.history.getState();
      const add = start(id, input, values); const addedPreview = await add.update(input); await add.finish();
      const added = await mass(); assert.ok(added > before + 1, `${id}: release adds live water`);
      const remove = start(id, input, { ...values, remove: 1 }); const removedPreview = await remove.update(input); await remove.finish();
      assert.deepEqual({ ...removedPreview!.action!.edit, operation: "add" }, addedPreview!.action!.edit, `${id}: identical add/remove shape geometry`);
      assert.ok(Math.abs(await mass() - before) < 1e-6, `${id}: removal restores baseline mass in the previously empty region`);
      assert.equal(session.history.getState().past, historyBefore.past, "fluid events are not authored history");
      assert.equal(session.history.getState().future, historyBefore.future);
    }
    // Cancellation arrives while a real acceptance is pending, then rolls back
    // through the same worker handler rather than synthesizing inverse patches.
    const beforeCancel = session.scene.getState().scene;
    const cancelSolids = await solidFractions(), cancelHistory = session.history.getState();
    const cancelled = start("box");
    const updating = cancelled.update(ray());
    const finishing = cancelled.finish(true);
    await updating; await finishing; await maintain();
    assert.deepEqual(session.scene.getState().scene.solidVoxels, beforeCancel.solidVoxels);
    assert.ok((await solidFractions()).every((value, index) => value === cancelSolids[index]), "cancel restores exact GPU occupancy");
    assert.equal(session.history.getState().past, cancelHistory.past);
    assert.equal(session.history.getState().future, cancelHistory.future);
    // A former build becomes wet after Undo. Redo must reject before changing
    // either history stack or the accepted document; this caught the UI halt.
    const build = start("box"); await build.update(ray()); await build.finish();
    await history("undo");
    const water = start("fluid-cube", ray(), { size: 3, height: 17 });
    await water.update(ray()); await water.finish();
    const wetScene = session.scene.getState().scene, stacks = session.history.getState();
    const wetFields = await solver.readDiagnosticFields(true), wetSolids = await solidFractions();
    await history("redo");
    assert.equal(session.scene.getState().scene, wetScene);
    assert.ok((await solidFractions()).every((value, index) => value === wetSolids[index]), "wet Redo rejection preserves GPU geometry");
    assert.equal(session.history.getState().past, stacks.past);
    assert.equal(session.history.getState().future, stacks.future);
    assert.ok((await solver.readDiagnosticFields(true)).density.every((value, index) => value === wetFields.density[index]));
    await advance();
    await solver.assertSimulationHealthy();
    assert.equal(world.status().fault, undefined);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ solidTools: expectedSolids, modes: ["empty-running", "moving-water"], finalTime_s: solver.info.submittedTime_s }));
  } finally {
    detach?.(); solver?.destroy(); display?.destroy();
    if (device) { await device.queue.onSubmittedWorkDone(); device.destroy(); }
    await releaseWebGPUExclusiveLock();
  }
});
