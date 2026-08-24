import assert from "node:assert/strict";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import { resolveMethodValues } from "../lib/core/method-contract";
import { createMinimalPowerDamBreak32Scene } from "../lib/core/scenes";
import { VISUALIZATION_FIELDS } from "../lib/core/visualization-catalog";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  isPressureJournalOverlayMode,
  PRESSURE_JOURNAL_CELL_BUDGET,
  PRESSURE_JOURNAL_DECADES,
  PRESSURE_JOURNAL_PRESSURE_DECADES,
  PressureJournalOverlay,
  pressureJournalOverlayChannel,
  pressureJournalOverlayVisualizations,
  type PressureJournalOverlayMode,
} from "../lib/core/webgpu-pressure-journal-overlay";
import { isFieldVisualization } from "../lib/core/visualization-registry";
import { optionalRendererPipelineRequests } from "../lib/core/webgpu-renderer";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { SPARSE_CM12_RESIDENT_STAGES } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

/**
 * The view that draws a captured pressure solve.
 *
 * The journal's own tests prove the capture is faithful. These prove the film
 * that reads it is: that its colour scale comes from the capture rather than
 * from the frame, that the four modes are reachable from the catalog, and —
 * the one only a device can answer — that the shader compiles and draws
 * against a real capture without the validator objecting.
 */

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

const MODES = pressureJournalOverlayVisualizations
  .filter(isFieldVisualization)
  .map((visualization) => visualization.mode as PressureJournalOverlayMode);

test("the cell budget decimates the captured cells instead of truncating them", () => {
  for (const cellCount of [0, 1, 37_440, 1_495_524, 40_000_000]) {
    const { stride, cells } = PressureJournalOverlay.plan(cellCount);
    assert.ok(stride >= 1, `${cellCount} cells planned a stride below one`);
    assert.ok(cells <= PRESSURE_JOURNAL_CELL_BUDGET,
      `${cellCount} cells planned ${cells} squares over budget`);
    // A film missing its last third would read as a domain that ends there,
    // which is a claim about the solve rather than about the plot.
    assert.ok(stride * cells >= cellCount,
      `${cellCount} cells leave ${cellCount - stride * cells} unreachable`);
    assert.ok(stride * (cells - 1) < Math.max(cellCount, 1),
      `${cellCount} cells plan ${cells} squares, past the end of the buffer`);
  }
  assert.deepEqual(PressureJournalOverlay.plan(0), { stride: 1, cells: 0 });
});

/**
 * Every mode the catalog advertises must be one the renderer can honour.
 *
 * These two lists are edited in different files and nothing but this holds them
 * together: a catalog entry whose mode the renderer does not recognise appears
 * in the picker, is selectable, and draws the generic slice raymarch instead —
 * a wrong picture rather than a missing one.
 */
test("the catalog's journal modes are the modes the renderer draws", () => {
  assert.equal(MODES.length, 4);
  for (const mode of MODES) {
    assert.ok(isPressureJournalOverlayMode(mode), `${mode} is not a journal mode`);
    const requested = optionalRendererPipelineRequests(
      { axis: "volume", position: 0.5, mode }, true, false);
    assert.ok(requested.includes("pressure-journal-overlay"),
      `${mode} must compile the journal overlay`);
    assert.ok(!requested.includes("grid-overlay"),
      `${mode} must not fall through to the generic field raymarch`);
    assert.deepEqual(
      optionalRendererPipelineRequests({ axis: "off", position: 0.5, mode }, true, false)
        .includes("pressure-journal-overlay"),
      false, `${mode} must compile nothing while hidden`);
  }
  // And the method has to offer them, or the picker never shows the entries.
  const offered = new Set(adaptiveMassMethod.supportedFieldModes ?? []);
  for (const mode of MODES) {
    assert.ok(offered.has(mode), `Sparse CM12 does not offer ${mode}`);
  }
  assert.ok(!isPressureJournalOverlayMode("face-velocity"));
  assert.ok(!isPressureJournalOverlayMode(undefined));
  // Every entry must actually be in the shipped catalog, not merely exported.
  const catalog = new Set(VISUALIZATION_FIELDS.map((entry) => entry.mode));
  for (const mode of MODES) assert.ok(catalog.has(mode), `${mode} is not in the catalog`);
});

/**
 * The calibration, as measured rather than as assumed.
 *
 * Both of these were wrong in the first draft and a real capture corrected
 * them, so they are pinned: pressure came back non-negative on every iteration
 * past the first few, which makes a diverging ramp half-wasted and puts the
 * bulk of the field on the colour that reads as no-data; and the residual
 * spans ten decades across one film, which no single ramp shows.
 */
test("the channels are calibrated to what a capture actually holds", () => {
  const pressure = pressureJournalOverlayChannel("pressure-journal-pressure", 45);
  assert.equal(pressure.field, "pressure");
  assert.equal(pressure.ramp, "sequential-log",
    "measured pressure is non-negative; a diverging ramp wastes half of itself");
  assert.equal(pressure.decades, PRESSURE_JOURNAL_PRESSURE_DECADES);
  assert.ok(!pressure.subtractHydrostatic,
    "the measured field is bulk-near-zero with a heavy tail, not a hydrostatic ramp");
  // The residual reference is the capture's, so the caller's number must reach
  // the ramp untouched — a channel that quietly substituted a constant would
  // renormalise every scene to the same scale.
  for (const mode of MODES.filter((entry) => entry !== "pressure-journal-pressure")) {
    const channel = pressureJournalOverlayChannel(mode, 45);
    assert.equal(channel.ramp, "diverging-log", `${mode} is a signed field`);
    assert.equal(channel.reference, 45, `${mode} lost the capture's reference`);
    assert.equal(channel.decades, PRESSURE_JOURNAL_DECADES);
  }
  // z and d are residual-scaled by construction (z = M⁻¹r, d₀ = z₀), so they
  // must share the residual's scale exactly; drawn on scales of their own the
  // three views could not be compared, which is the main reason to have them.
  const residual = pressureJournalOverlayChannel("pressure-journal-residual", 45);
  for (const mode of ["pressure-journal-preconditioned",
    "pressure-journal-direction"] as const) {
    assert.deepEqual(pressureJournalOverlayChannel(mode, 45).reference, residual.reference);
    assert.deepEqual(pressureJournalOverlayChannel(mode, 45).decades, residual.decades);
  }
});

/**
 * The film is a draw, and the capture behind it is gated.
 *
 * The stage partition is the ledger of what an advance costs, and the snapshot
 * dispatches are deliberately not on it — they exist only on an armed frame.
 * The failure this guards against is the tempting one: adding a permanent
 * capture stage so the film is always available, which would charge every
 * unwatched frame for a picture nobody asked for.
 */
test("the film adds no stage to the advance partition", () => {
  assert.ok(!SPARSE_CM12_RESIDENT_STAGES.some((stage) => /journal|film|snapshot/.test(stage)),
    "journal capture must not appear on the advance stage partition");
});

dawnTest("Dawn draws a captured film without the validator objecting",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-pressure-journal-overlay.test.ts");
    let device: GPUDevice | undefined;
    try {
      const modulePath = dawnModule
        ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
      const dawn = await import(pathToFileURL(modulePath).href) as {
        create(options: string[]): GPU; globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      Object.defineProperty(globalThis, "navigator",
        { configurable: true, value: { gpu } });
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter, "Dawn must expose a WebGPU adapter");
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const uncaptured: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        uncaptured.push(event.error.message);
      });

      const scene = createMinimalPowerDamBreak32Scene();
      const values = resolveMethodValues(adaptiveMassMethod, "balanced",
        { timeStep: "scene", pressureIterations: 32 });
      const solver = await adaptiveMassMethod.createSolverAsync!(
        device, scene, "balanced", { ...values, pressureJournal: "on" },
        undefined, () => {}) as WebGPUAdaptiveMassSolver;
      const format: GPUTextureFormat = "rgba8unorm";
      const overlay = new PressureJournalOverlay(device, format);
      const target = device.createTexture({
        size: { width: 64, height: 64 }, format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      try {
        await overlay.initialize();
        const dt_s = scene.numerics.fixedDt_s ?? scene.numerics.maxDt_s;
        const advance = async (time_s: number) => {
          while (!solver.advanceTo(time_s, [])) await new Promise(setImmediate);
        };
        await advance(dt_s);
        assert.equal(
          solver.sparseWorldUI.control.pressureFilm!.setCaptureEnabled(true), true,
          "the 'on' option must reserve the film");
        await advance(2 * dt_s);
        await device.queue.onSubmittedWorkDone();

        const source = solver.sparseWorldUI.overlays.pressureFilm;
        assert.ok(source, "an armed solver must publish a journal source");
        // The scrub's upper bound. Without it the slider runs to the reserved
        // capacity and its tail replays whatever a previous capture left.
        assert.ok(source!.snapshotCount > 1,
          `the capture filled ${source!.snapshotCount} snapshot slots`);
        assert.ok(source!.snapshotCount <= source!.layout.snapshotCapacity);
        assert.ok(source!.cellCount > 0 && source!.finestCell_m > 0);
        overlay.setSource(source);

        // Draw every mode at every captured iteration. The shader indexes the
        // journal with arithmetic the host performs independently, so an
        // off-by-one in either lands out of bounds — which on this device is an
        // uncaptured validation error rather than a wrong picture.
        const encoder = device.createCommandEncoder();
        const view = target.createView();
        let drawn = 0;
        for (const mode of MODES) {
          for (let snapshot = 0; snapshot < source!.snapshotCount; snapshot += 1) {
            const encoded = overlay.encode(encoder, view, undefined, {
              camera: {
                position_m: [0, 0.5, 2], forward: [0, 0, -1],
                right: [1, 0, 0], up: [0, 1, 0], tanHalfFov: 0.5, aspect: 1,
              },
              viewportWidth: 64, viewportHeight: 64,
              container_m: [scene.container.width_m, scene.container.height_m,
                scene.container.depth_m],
              depthNear_m: 0.1,
              snapshot,
              channel: pressureJournalOverlayChannel(mode, 1),
            });
            assert.equal(encoded, true, `${mode} at ${snapshot} refused to draw`);
            drawn += 1;
          }
        }
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        assert.ok(drawn >= 4, "every mode must have drawn at least once");
        assert.deepEqual(uncaptured, [],
          `the film's draws raised validation errors: ${uncaptured.join("; ")}`);

        // A disarmed solver keeps its source: the previous capture is still
        // there to scrub, which is what makes the film usable while paused.
        solver.sparseWorldUI.control.pressureFilm!.setCaptureEnabled(false);
        assert.ok(solver.sparseWorldUI.overlays.pressureFilm,
          "disarming must not withdraw the captured film");
      } finally {
        target.destroy();
        overlay.destroy();
        solver.destroy();
      }
    } finally {
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
