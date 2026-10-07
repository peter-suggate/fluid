import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { UniformMacSolver } from "../lib/methods/uniform/mac/solver";
import { readFloatTexture3D, readRgbaTexture3D, smokeRenderHybridPresentation } from "../lib/harness/webgpu-smoke-readbacks";
import { usePerformanceInstrumentationStore } from "../lib/core/stores/performance-instrumentation-store";
import { solidVoxelShellForScene } from "../lib/core/scene-lattice";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("uniform MAC numerical and publication contracts", { timeout: 180_000 }, async t => {
  await acquireWebGPUExclusiveLock("dawn-test", "Uniform MAC baseline");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn as NodeDawnProvider, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits), requiredFeatures: adapter.features.has("timestamp-query") ? ["timestamp-query"] : [] }), { requireWorkerRealm: false });
    const errors: string[] = []; device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    const base = structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
    Object.assign(base.container, { width_m: 1, height_m: 1, depth_m: 1, fillFraction: 0.5, top: "open" });
    base.voxelDomain.finestCellSize_m = 1 / 16; base.rigidBodies = []; base.solidVoxels = [];
    Object.assign(base.fluid, { initialCondition: "tank-fill", initialLiquidVolumes: [], initialBrickSeeds_m: [], initialVelocity_m_s: undefined,
      inflow: undefined, initialHeightField: undefined, surfaceTension_N_m: 0, dynamicViscosity_Pa_s: 0, gravity_m_s2: { x: 0, y: -9.81, z: 0 } });

    await t.test("submission and presentation health do not wait for diagnostic readbacks", async () => {
      let hold = true, mapCalls = 0, release!: () => void;
      const gate = new Promise<void>(resolve => { release = resolve; });
      const observed = new Proxy(device!, { get(target, key) {
        if (key === "createBuffer") return (descriptor: GPUBufferDescriptor) => {
          const buffer = target.createBuffer(descriptor);
          if (descriptor.label?.includes("diagnostic readback")) {
            const map = buffer.mapAsync.bind(buffer);
            buffer.mapAsync = async (...args: Parameters<GPUBuffer["mapAsync"]>) => { mapCalls++; await map(...args); if (hold) await gate; };
          }
          return buffer;
        };
        const value = Reflect.get(target, key, target); return typeof value === "function" ? value.bind(target) : value;
      } });
      const scene = structuredClone(base); scene.fluid.initialCondition = "dam-break";
      // A watchdog releases a regressed blocking initializer so cleanup remains
      // possible; healthy creation must finish while the receipt is still held.
      const watchdog = setTimeout(() => { hold = false; release(); }, 15_000);
      const solver = await UniformMacSolver.createAsync(observed, scene, "balanced");
      clearTimeout(watchdog);
      try {
        assert.equal(hold, true, "initial publication must not wait for diagnostics");
        const before = mapCalls;
        assert.equal(solver.advanceTo(1 / 120, []), true);
        assert.equal(solver.advanceTo(2 / 120, []), true);
        assert.equal(solver.framePending, false); assert.equal(solver.presentationPending, false);
        await solver.assertSimulationHealthy(Promise.resolve());
        assert.equal(solver.advanceTo(3 / 120, []), true);
        assert.equal(mapCalls - before, 2, "the busy three-slot ring skips copies without waiting; no per-iteration/substep maps");
        // The published velocity is readable while the diagnostic promise is held.
        const velocity = await readRgbaTexture3D(device!, solver.velocityTexture, 16, 16, 16);
        assert.ok(velocity.some(value => Math.abs(value) > 0.01));
        solver.applyRuntimeValues({ pressureTolerance: 0.00001 });
        hold = false; release(); await solver.awaitFrameCompletion();
        assert.equal(solver.info.completedTime_s, 3 / 120);
        assert.equal(solver.info.pressureSolveConverged, true, "a later tolerance edit cannot relabel an accepted GPU frame");
      } finally { release(); solver.destroy(); }
    });
    await t.test("a far-ahead UI clock submits one bounded advance", async () => {
      const solver = await UniformMacSolver.createAsync(device!, base, "balanced");
      try {
        assert.equal(solver.advanceTo(1, []), true);
        assert.equal(solver.info.submittedTime_s, 1 / 120);
        assert.ok(solver.info.simulationLag_s! > 0.9);
        await solver.awaitFrameCompletion();
        assert.equal(solver.info.completedTime_s, 1 / 120);
        assert.equal(solver.advanceTo(1, []), true);
        await solver.awaitFrameCompletion();
        assert.equal(solver.info.completedTime_s, 2 / 120);
        assert.equal(solver.info.macSubsteps, 2);
      } finally { solver.destroy(); }
    });
    await t.test("authored dam playback survives splash acceleration with asynchronous diagnostics", async () => {
      const scene = sceneDocument(getSceneDefinition("minimal-power-dam-break-32"));
      const solver = await UniformMacSolver.createAsync(device!, scene, "balanced");
      try {
        let frames = 0;
        while ((solver.info.submittedTime_s ?? 0) < 1 - 1e-10) {
          assert.ok(++frames <= 2000, "the authored one-second run must make bounded progress");
          assert.equal(solver.advanceTo(1, []), true);
          // Emulate the renderer's existing queue fence, without requesting a
          // solver receipt or waiting on a diagnostic map to schedule the step.
          await solver.assertSimulationHealthy(device!.queue.onSubmittedWorkDone());
        }
        await solver.awaitFrameCompletion();
        assert.ok(Math.abs(solver.info.completedTime_s! - 1) < 1e-8);
        assert.ok(solver.info.pressureSolveConverged);
        assert.ok(solver.info.maxDivergence_s! <= 0.0011);
      } finally { solver.destroy(); }
    });
    await t.test("GPU CFL subdivides a fast moving drop without host decisions", async () => {
      const scene = structuredClone(base); scene.container.fillFraction = 0;
      scene.fluid.initialLiquidVolumes = [{ shape: "sphere", center_m: { x: 0, y: 0.65, z: 0 }, radius_m: 0.18 }];
      scene.fluid.initialVelocity_m_s = { x: 20, y: 0, z: 0 };
      const solver = await UniformMacSolver.createAsync(device!, scene, "balanced");
      try {
        await solver.awaitFrameCompletion();
        // A stale speed hint may understate the current velocity, but the GPU
        // must still split the requested interval using the actual field.
        (solver as unknown as { capacitySpeed: number }).capacitySpeed = 5;
        solver.advanceTo(1 / 120, []); await solver.awaitFrameCompletion();
        assert.ok(solver.info.lastSubsteps! > 1, "current GPU velocity must subdivide a stale host request");
        assert.ok(solver.info.lastDt_s! < 1 / 120);
        assert.ok(solver.info.pressureSolveConverged);
      } finally { solver.destroy(); }
    });
    await t.test("an underestimated capacity hint cannot publish a partial frame", async () => {
      const scene = structuredClone(base); scene.container.fillFraction = 0;
      scene.fluid.initialLiquidVolumes = [{ shape: "sphere", center_m: { x: 0, y: 0.65, z: 0 }, radius_m: 0.18 }];
      scene.fluid.initialVelocity_m_s = { x: 20, y: 0, z: 0 };
      const solver = await UniformMacSolver.createAsync(device!, scene, "balanced");
      try {
        const before = await readFloatTexture3D(device!, solver.denseLevelSetVolumeSource.vertexPhi, 17, 17, 17);
        // Simulate a badly lagged sizing hint; actual GPU velocity is unchanged.
        (solver as unknown as { capacitySpeed: number }).capacitySpeed = 0;
        solver.advanceTo(1 / 120, []);
        await assert.rejects(solver.awaitFrameCompletion(), /substep capacity exhausted/);
        const after = await readFloatTexture3D(device!, solver.denseLevelSetVolumeSource.vertexPhi, 17, 17, 17);
        assert.deepEqual(after, before); assert.equal(solver.info.completedTime_s, 0);
      } finally { solver.destroy(); }
    });
    await t.test("hydrostatic pool remains still with a converged free-surface pressure", async () => {
      const solver = await UniformMacSolver.createAsync(device!, base, "balanced");
      try {
        const initial = (await solver.readStats()).volumeCellSum!;
        for (let step = 1; step <= 12; step++) { assert.equal(solver.advanceTo(step / 120, []), true); await solver.awaitFrameCompletion(); }
        const info = await solver.readStats();
        assert.ok(info.pressureSolveConverged); assert.ok(info.pressureResidual! <= 0.001);
        assert.ok(info.maxDivergenceAfter_s! < 0.002, `divergence ${info.maxDivergenceAfter_s}`);
        assert.ok(info.maxSpeed_m_s! < 0.0001, `rest speed ${info.maxSpeed_m_s}`);
        assert.ok(Math.abs(info.volumeCellSum! / initial - 1) < 1e-5, `volume drift ${info.volumeDrift}`);
        assert.equal(solver.denseLevelSetVolumeSource.vertexPhi.width, 17);
        assert.equal(solver.volumeTexture.width, 16); assert.equal(solver.velocityTexture.format, "rgba32float");
        assert.ok(Math.abs(info.completedTime_s! - 0.1) < 1e-12);
      } finally { solver.destroy(); }
    });
    await t.test("dam break moves while retaining the pressure and finite-state contracts", async () => {
      const scene = structuredClone(base); scene.fluid.initialCondition = "dam-break";
      const solver = await UniformMacSolver.createAsync(device!, scene, "balanced");
      try {
        const before = await solver.readStats();
        for (let step = 1; step <= 8; step++) { solver.advanceTo(step / 120, []); await solver.awaitFrameCompletion(); }
        const after = await solver.readStats();
        assert.ok(after.maxSpeed_m_s! > 0.05); assert.ok(after.pressureSolveConverged);
        assert.ok(after.maxDivergenceAfter_s! < 0.003, `divergence ${after.maxDivergenceAfter_s}`);
        assert.ok(Math.abs(after.volumeCellSum! / before.volumeCellSum! - 1) < 0.05, `volume drift ${after.volumeDrift}`);
      } finally { solver.destroy(); }
    });
    await t.test("signed pressure balances upward gravity instead of clamping tension", async () => {
      const scene = structuredClone(base); scene.fluid.gravity_m_s2.y = 9.81;
      const solver = await UniformMacSolver.createAsync(device!, scene, "balanced");
      try {
        solver.advanceTo(1 / 120, []); await solver.awaitFrameCompletion();
        const pressure = await readFloatTexture3D(device!, solver.gridPressureTexture, 16, 16, 16);
        const expected = -scene.fluid.density_kg_m3 * 9.81 * (0.5 - 0.5 / 16);
        assert.ok(Math.abs(pressure[8 + 16 * (0 + 16 * 8)] / expected - 1) < 0.001);
        assert.ok((await solver.readStats()).maxSpeed_m_s! < 0.0001);
      } finally { solver.destroy(); }
    });
    await t.test("a detached drop accelerates at gravity and reports hardware pipeline phases", async () => {
      const scene = structuredClone(base); scene.container.fillFraction = 0;
      scene.fluid.initialLiquidVolumes = [{ shape: "sphere", center_m: { x: 0, y: 0.65, z: 0 }, radius_m: 0.18 }];
      const solver = await UniformMacSolver.createAsync(device!, scene, "balanced");
      const wasEnabled = usePerformanceInstrumentationStore.getState().enabled;
      usePerformanceInstrumentationStore.setState({ enabled: true });
      try {
        for (let step = 1; step <= 6; step++) { solver.advanceTo(step / 120, []); await solver.awaitFrameCompletion(); }
        const velocity = await readRgbaTexture3D(device!, solver.velocityTexture, 16, 16, 16);
        const centre = 4 * (8 + 16 * (10 + 16 * 8));
        assert.ok(Math.abs(velocity[centre + 1] - -9.81 * 0.05) < 0.01, `drop speed ${velocity[centre + 1]}`);
        const info = await solver.readStats(); assert.ok(Math.abs(info.volumeDrift!) < 0.04);
        assert.ok(info.physicsTrace);
        if (device!.features.has("timestamp-query")) {
          assert.equal(info.physicsTrace!.measurementSource, "gpu-hardware-timestamp");
          assert.ok(info.physicsTrace!.phases.some(p => p.label === "MAC ghost-fluid pressure solve" && p.duration_ms > 0));
        }
      } finally { usePerformanceInstrumentationStore.setState({ enabled: wasEnabled }); solver.destroy(); }
    });
    await t.test("a stationary spherical drop has the Laplace pressure jump", async () => {
      const scene = structuredClone(base); scene.container.fillFraction = 0; scene.voxelDomain.finestCellSize_m = 1 / 32;
      scene.fluid.gravity_m_s2.y = 0; scene.fluid.surfaceTension_N_m = 0.072;
      scene.fluid.initialLiquidVolumes = [{ shape: "sphere", center_m: { x: 0, y: 0.5, z: 0 }, radius_m: 0.2 }];
      const solver = await UniformMacSolver.createAsync(device!, scene, "balanced");
      try {
        solver.advanceTo(1 / 120, []); await solver.awaitFrameCompletion();
        const pressure = await readFloatTexture3D(device!, solver.gridPressureTexture, 32, 32, 32);
        const expected = 2 * 0.072 / 0.2;
        assert.ok(Math.abs(pressure[16 + 32 * (16 + 32 * 16)] / expected - 1) < 0.15);
        assert.ok((await solver.readStats()).maxDivergenceAfter_s! < 0.003);
      } finally { solver.destroy(); }
    });
    await t.test("fixed voxel walls do not generate capillary motion in a flat pool", async () => {
      const scene = structuredClone(base); scene.solidVoxels = [...solidVoxelShellForScene(scene)]; scene.fluid.surfaceTension_N_m = 0.072;
      const solver = await UniformMacSolver.createAsync(device!, scene, "balanced");
      try {
        for (let step = 1; step <= 4; step++) { solver.advanceTo(step / 120, []); await solver.awaitFrameCompletion(); }
        assert.ok((await solver.readStats()).maxSpeed_m_s! < 0.0001);
      } finally { solver.destroy(); }
    });
    await t.test("64-cubed dam break converges within the production pressure limit", async () => {
      const scene = structuredClone(base); scene.fluid.initialCondition = "dam-break"; scene.voxelDomain.finestCellSize_m = 1 / 64;
      const solver = await UniformMacSolver.createAsync(device!, scene, "balanced");
      try {
        const start = performance.now(); let encodeMs = 0;
        for (let step = 1; step <= 3; step++) { const encodeStart = performance.now(); solver.advanceTo(step / 120, []); encodeMs += performance.now() - encodeStart; await solver.awaitFrameCompletion(); }
        const info = await solver.readStats();
        assert.ok(info.pressureSolveConverged); assert.ok(info.maxDivergenceAfter_s! < 0.003);
        t.diagnostic(`MAC 64³: ${((performance.now() - start) / 3).toFixed(2)} ms/substep (includes explicit test fence; ${(encodeMs / 3).toFixed(2)} ms CPU encoding), ${info.pressureIterationsExecuted} final-step iterations, ${info.allocatedBytes} bytes`);
      } finally { solver.destroy(); }
    });
    await t.test("under-solving pressure stops before publishing a physical step", async () => {
      const scene = structuredClone(base); scene.voxelDomain.finestCellSize_m = 1 / 64;
      scene.fluid.initialVelocity_m_s = { x: 1, y: 0, z: 0 };
      const solver = await UniformMacSolver.createAsync(device!, scene, "balanced", { pressureLimit: 32, pressureTolerance: 0.00001 });
      try {
        const before = await readFloatTexture3D(device!, solver.denseLevelSetVolumeSource.vertexPhi, 65, 65, 65);
        solver.advanceTo(1 / 120, []);
        solver.applyRuntimeValues({ pressureLimit: 2048, pressureTolerance: 0.1 });
        await assert.rejects(solver.awaitFrameCompletion(), /pressure did not converge:.*limit 32/);
        const after = await readFloatTexture3D(device!, solver.denseLevelSetVolumeSource.vertexPhi, 65, 65, 65);
        assert.deepEqual(after, before, "a rejected GPU frame must retain the accepted presentation");
        assert.equal(solver.info.completedTime_s, 0);
        assert.equal(solver.info.pressureSolveConverged, false);
      } finally { solver.destroy(); }
    });
    await t.test("a small standing wave follows linear gravity-wave phase", async () => {
      const scene = structuredClone(base); scene.container.depth_m = 0.25; scene.voxelDomain.finestCellSize_m = 1 / 32;
      scene.fluid.initialHeightField = { kind: "cosine", baseHeight_m: 0.5, amplitude_m: 0.01, wavelength_m: 1, originX_m: -0.5 };
      const solver = await UniformMacSolver.createAsync(device!, scene, "balanced", { maxStep: 0.004 });
      try {
        const k = 2 * Math.PI, omega = Math.sqrt(9.81 * k * Math.tanh(k * 0.5));
        const target = Math.PI / (3 * omega); // amplitude should be half its starting value
        while ((solver.info.submittedTime_s ?? 0) < target - 1e-10) {
          assert.equal(solver.advanceTo(target, []), true); await solver.awaitFrameCompletion();
        }
        const phi = await readFloatTexture3D(device!, solver.denseLevelSetVolumeSource.vertexPhi, 33, 33, 9);
        let cosine = 0, norm = 0;
        for (let x = 1; x < 32; x++) {
          let height = NaN;
          for (let y = 1; y < 32; y++) {
            const a = phi[x + 33 * (y + 33 * 4)], b = phi[x + 33 * (y + 1 + 33 * 4)];
            if (a <= 0 && b >= 0) { height = (y + -a / (b - a)) / 32; break; }
          }
          assert.ok(Number.isFinite(height)); const mode = Math.cos(k * x / 32);
          cosine += (height - 0.5) * mode; norm += mode * mode;
        }
        const amplitude = cosine / norm;
        assert.ok(Math.abs(amplitude - 0.005) < 0.001, `wave amplitude ${amplitude}, linear reference 0.005`);
      } finally { solver.destroy(); }
    });
    await t.test("the shared water renderer consumes the native MAC publication", async () => {
      const solver = await UniformMacSolver.createAsync(device!, base, "balanced");
      try {
        solver.advanceTo(1 / 120, []); await solver.awaitFrameCompletion();
        const rendered = await smokeRenderHybridPresentation(device!, solver, base, []);
        assert.ok(rendered.frontInterfacePixels > 0); assert.ok(rendered.pairedInterfacePixels > 0);
        assert.equal(rendered.rendererValidationErrorCount, 0);
        assert.equal(rendered.rendererUncapturedErrorCount, 0);
      } finally { solver.destroy(); }
    });
    assert.deepEqual(errors, []);
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
