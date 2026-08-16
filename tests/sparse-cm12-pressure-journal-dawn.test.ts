import assert from "node:assert/strict";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import { resolveMethodValues } from "../lib/core/method-contract";
import { createMinimalPowerDamBreak32Scene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import {
  assertSparseCM12PressureJournal,
  sparseCM12PressureJournalSchedule,
} from "../lib/methods/adaptive-mass/sparse-cm12-pressure-journal";

/**
 * The journal against the solve it filmed.
 *
 * The unit tests pin the layout arithmetic and the decode; only a real device
 * can answer the questions that matter here. Does the ungated kernel actually
 * run once per encoded iteration? Does the cursor land records in order? And
 * critically — does the film agree with the receipt the same frame published?
 * A journal that disagreed with `readDiagnostics` would be a plausible,
 * confident picture of a solve that did not happen.
 */

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

const PRESSURE_ITERATIONS = 32;

async function withSolver<T>(
  run: (solver: WebGPUAdaptiveMassSolver, dt_s: number) => Promise<T>,
): Promise<T> {
  await acquireWebGPUExclusiveLock("dawn-test", "sparse-cm12-pressure-journal");
  try {
    const modulePath = dawnModule
      ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
    const { create, globals } = await import(pathToFileURL(modulePath).href) as {
      create(options: string[]): GPU; globals: Record<string, unknown>;
    };
    Object.assign(globalThis, globals);
    const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    assert.ok(adapter, "WebGPU did not expose an adapter");
    const device = await adapter.requestDevice({
      requiredLimits: requiredFluidDeviceLimits(adapter.limits),
    });
    const scene = createMinimalPowerDamBreak32Scene();
    const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
      timeStep: "scene",
      pressureIterations: PRESSURE_ITERATIONS,
    });
    const solver = await adaptiveMassMethod.createSolverAsync!(
      device, scene, "balanced",
      { ...values, pressureJournal: true },
      undefined, () => {},
    ) as WebGPUAdaptiveMassSolver;
    try {
      const dt_s = scene.numerics.fixedDt_s ?? scene.numerics.maxDt_s;
      return await run(solver, dt_s);
    } finally {
      solver.destroy();
    }
  } finally {
    await releaseWebGPUExclusiveLock();
  }
}

const advance = async (solver: WebGPUAdaptiveMassSolver, time_s: number) => {
  while (!solver.advanceTo(time_s, [])) await new Promise(setImmediate);
};

dawnTest("an unarmed advance leaves no capture behind", async () => {
  await withSolver(async (solver, dt_s) => {
    // Reserved but never armed: the region exists and every dispatch that would
    // fill it was skipped on the host, so the header must still read unarmed.
    assert.ok(solver.pressureJournalLayout.floatCount > 0,
      "the solver was built with the journal capability");
    assert.equal(solver.pressureJournalArmed, false);
    await advance(solver, dt_s);
    assert.equal(await solver.readPressureJournal(), undefined);
  });
});

dawnTest("a captured solve records every encoded iteration, in order", async () => {
  await withSolver(async (solver, dt_s) => {
    await advance(solver, dt_s);
    assert.equal(solver.armPressureJournal(true), true);
    await advance(solver, 2 * dt_s);
    const journal = await solver.readPressureJournal();
    assert.ok(journal, "an armed advance must leave a capture");
    assertSparseCM12PressureJournal(journal!);

    // The seed plus one record per encoded iteration. This is the property the
    // whole cursor mechanism exists to provide, and the only one that cannot be
    // checked without a device.
    assert.equal(journal!.records.length, PRESSURE_ITERATIONS + 1);
    assert.equal(journal!.encodedIterations, PRESSURE_ITERATIONS);
    journal!.records.forEach((record, index) => {
      assert.equal(record.iteration, index, "records must land in encode order");
    });

    // Encoded is not executed: the gate closes the tail, and the film must show
    // that rather than drawing a converged tail as though it had been computed.
    assert.ok(journal!.executedIterations <= PRESSURE_ITERATIONS);
    const active = journal!.records.filter((record) => record.active).length;
    assert.equal(active, journal!.executedIterations,
      "an active record and an executed iteration are the same thing");
  });
});

dawnTest("the film agrees with the receipt of the frame it filmed", async () => {
  await withSolver(async (solver, dt_s) => {
    await advance(solver, dt_s);
    solver.armPressureJournal(true);
    await advance(solver, 2 * dt_s);
    const journal = await solver.readPressureJournal();
    // `info` carries the receipt only once a readback has filled it; reading
    // the field off a solver that was never asked for stats compares the film
    // against an empty object and passes for the wrong reason.
    const info = await solver.readStats();
    assert.ok(journal);

    // The receipt and the journal are written by different kernels reading the
    // same scalars. If they disagree, one of them is describing another frame.
    assert.equal(journal!.executedIterations, info.pressureIterationsExecuted);
    assert.equal(journal!.encodedIterations, info.pressureIterationsEncoded);
    if (info.pressureFirstToleranceCrossingIteration !== undefined) {
      assert.equal(journal!.firstCrossingIteration,
        info.pressureFirstToleranceCrossingIteration);
    }

    // The residual has to fall. A film whose curve is flat or rising is either
    // reading a stale record or filming a solve that is not converging, and
    // both are worth failing on.
    const first = journal!.records[1]!;
    const last = journal!.records[journal!.executedIterations]!;
    assert.ok(Number.isFinite(first.recursiveRelativeL2));
    assert.ok(last.recursiveRelativeL2 <= first.recursiveRelativeL2,
      `residual rose: ${first.recursiveRelativeL2} -> ${last.recursiveRelativeL2}`);
  });
});

dawnTest("snapshots land on the scheduled iterations", async () => {
  await withSolver(async (solver, dt_s) => {
    await advance(solver, dt_s);
    solver.armPressureJournal(true);
    await advance(solver, 2 * dt_s);
    const journal = await solver.readPressureJournal();
    assert.ok(journal);
    // The host chooses the schedule at encode time and the device chooses the
    // slot with its own cursor. Neither tells the other, so this is where that
    // agreement is proven.
    const expected = sparseCM12PressureJournalSchedule(PRESSURE_ITERATIONS,
      solver.pressureJournalLayout.snapshotCapacity);
    assert.deepEqual([...journal!.snapshotIterations], [...expected]);
    assert.ok(expected.length > 0 && expected[0] === 0,
      "the seed must be filmed: the first correction is the largest one");
  });
});

dawnTest("disarming stops the capture without disturbing the solve", async () => {
  await withSolver(async (solver, dt_s) => {
    solver.armPressureJournal(true);
    await advance(solver, dt_s);
    const captured = await solver.readPressureJournal();
    assert.ok(captured);

    solver.armPressureJournal(false);
    await advance(solver, 2 * dt_s);
    // The header is only cleared by an armed frame, so the previous capture
    // survives verbatim: a disarmed advance encodes nothing that touches it.
    const after = await solver.readPressureJournal();
    assert.ok(after);
    assert.equal(after!.records.length, captured!.records.length);
    assert.equal(after!.executedIterations, captured!.executedIterations);
  });
});
