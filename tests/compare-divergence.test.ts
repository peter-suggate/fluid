import assert from "node:assert/strict";
import test from "node:test";

// No `../lib/methods` import and no `session/session.ts`: the divergence oracle
// is arithmetic over figures the panes already publish, and a readout that
// needed the method registry to decide whether two numbers agree would be the
// wrong shape.
import {
  divergenceRows,
  paneClockReading,
  paneStatsFrom,
  panesInStep,
  VOLUME_AGREEMENT_TOLERANCE,
  type DivergenceClocks,
  type DivergenceRow,
  type PaneStats,
} from "../lib/core/compare/divergence";

const DT = 1 / 30;

function clocks(tA: number, tB: number, overrides: Partial<DivergenceClocks> = {}): DivergenceClocks {
  return {
    a: { completedTime_s: tA, step_s: DT },
    b: { completedTime_s: tB, step_s: DT },
    dtDiffers: false,
    identical: true,
    ...overrides,
  };
}

function stats(overrides: Partial<PaneStats> = {}): PaneStats {
  return { volumeCells: 4096, pressureIterations: 12, msPerStep: 9.5, encodedSteps: 30, ...overrides };
}

function row(rows: readonly DivergenceRow[], key: string): DivergenceRow {
  const found = rows.find((candidate) => candidate.key === key);
  assert.ok(found, `no ${key} row`);
  return found;
}

test("the readout is one row per figure, per pane", () => {
  const rows = divergenceRows(stats(), stats(), clocks(1, 1));
  assert.deepEqual(rows.map((entry) => entry.key),
    ["t", "volume", "pressure-iterations", "ms-per-step", "lockstep"]);
  assert.equal(row(rows, "t").a, "1.000");
  assert.equal(row(rows, "volume").b, "4096.00");
  assert.equal(row(rows, "pressure-iterations").a, "12");
  assert.equal(row(rows, "ms-per-step").b, "9.5");
});

test("identical panes at the same step are quiet", () => {
  const rows = divergenceRows(stats(), stats(), clocks(1, 1));
  assert.deepEqual(rows.filter((entry) => entry.tone === "warn"), []);
  assert.equal(row(rows, "lockstep").delta, "in step");
});

test("a lag inside one step is the barrier working, not a divergence", () => {
  const rows = divergenceRows(stats(), stats({ encodedSteps: 29 }), clocks(1, 1 - DT));
  assert.equal(row(rows, "t").tone, "neutral");
  assert.equal(row(rows, "lockstep").delta, "in step");
  assert.equal(panesInStep(clocks(1, 1 - DT)), true);
});

test("a lag beyond one step warns on t and names the pane behind", () => {
  const behind = clocks(1, 1 - 3 * DT);
  const rows = divergenceRows(stats(), stats({ encodedSteps: 27 }), behind);
  assert.equal(row(rows, "t").tone, "warn");
  assert.equal(row(rows, "lockstep").tone, "warn");
  assert.equal(row(rows, "lockstep").delta, "B behind 0.100 s");
  assert.equal(panesInStep(behind), false);
  // And the other way round, so the verdict is not hard-coded to one pane.
  assert.equal(row(divergenceRows(stats(), stats(), clocks(1 - 3 * DT, 1)), "lockstep").delta,
    "A behind 0.100 s");
});

test("volume disagreeing at the same step is the non-determinism tell", () => {
  const rows = divergenceRows(stats(), stats({ volumeCells: 4097 }), clocks(1, 1));
  assert.equal(row(rows, "volume").tone, "warn");
  assert.equal(row(rows, "volume").delta, "+1.00");
  assert.match(row(rows, "volume").note ?? "", /step 30/);
});

test("volume is compared in physical units across different finest resolutions", () => {
  const fine = stats({ volumeCells: 94_208, volumeCellSize_m: 0.0125 });
  const coarse = stats({ volumeCells: 11_776, volumeCellSize_m: 0.025 });
  const volume = row(divergenceRows(fine, coarse, clocks(1, 1)), "volume");
  assert.equal(volume.unit, "m³");
  assert.equal(volume.a, "0.184000");
  assert.equal(volume.b, "0.184000");
  assert.equal(volume.delta, "0.000000");
  assert.equal(volume.tone, "neutral");
  assert.match(volume.note ?? "", /finest-cell equivalents/);
});

test("an unmeasured frame wall says which switch measures it", () => {
  const off = divergenceRows(stats({ msPerStep: undefined }), stats({ msPerStep: undefined }), clocks(1, 1));
  assert.equal(row(off, "ms-per-step").a, "—");
  assert.match(row(off, "ms-per-step").note ?? "", /performance instrument/);
  assert.equal(row(divergenceRows(stats(), stats(), clocks(1, 1)), "ms-per-step").note, "frame wall, per pane");
});

test("volume within tolerance agrees", () => {
  const drifted = 4096 * (1 + VOLUME_AGREEMENT_TOLERANCE / 2);
  assert.equal(row(divergenceRows(stats(), stats({ volumeCells: drifted }), clocks(1, 1)), "volume").tone,
    "neutral");
});

test("two samples from different steps are not compared", () => {
  // B is one advance behind; its mass *should* differ, and calling that a
  // divergence would make the readout warn through every normal frame.
  const rows = divergenceRows(stats(), stats({ volumeCells: 4200, encodedSteps: 29 }), clocks(1, 1 - DT));
  assert.equal(row(rows, "volume").tone, "neutral");
  assert.match(row(rows, "volume").note ?? "", /different steps/);
});

test("a deliberate diff is a comparison, not a divergence", () => {
  const rows = divergenceRows(stats(), stats({ volumeCells: 9000 }),
    clocks(1, 1, { identical: false }));
  assert.equal(row(rows, "volume").tone, "neutral");
});

test("a differing dt warns on the lockstep row whatever the clock says", () => {
  const mixed: DivergenceClocks = {
    a: { completedTime_s: 1, step_s: DT },
    b: { completedTime_s: 1, step_s: 2 * DT },
    dtDiffers: true,
    identical: true,
  };
  const rows = divergenceRows(stats(), stats(), mixed);
  assert.equal(row(rows, "lockstep").tone, "warn");
  assert.equal(row(rows, "lockstep").a, "0.0333");
  assert.equal(row(rows, "lockstep").b, "0.0667");
  assert.match(row(rows, "lockstep").note ?? "", /dt differs/);
  // The coarser pane owns the tolerance: one of *its* steps of lag is legal.
  assert.equal(panesInStep({ ...mixed, b: { completedTime_s: 1 - 2 * DT, step_s: 2 * DT } }), true);
  assert.equal(panesInStep({ ...mixed, b: { completedTime_s: 1 - 3 * DT, step_s: 2 * DT } }), false);
});

test("a pane the host has not registered reads as absent rather than as zero", () => {
  const rows = divergenceRows(stats(), {}, { a: { completedTime_s: 1, step_s: DT }, dtDiffers: false, identical: true });
  assert.equal(row(rows, "t").b, "—");
  assert.equal(row(rows, "volume").b, "—");
  assert.equal(row(rows, "lockstep").delta, "—");
  assert.equal(panesInStep({ dtDiffers: false, identical: true }), undefined);
  assert.deepEqual(rows.filter((entry) => entry.tone === "warn"), []);
});

test("pane clocks are picked out of the host's report by id", () => {
  const reports = [
    { id: "a", completedTime_s: 2, step_s: DT },
    { id: "b", completedTime_s: 1.9, step_s: DT },
  ];
  assert.deepEqual(paneClockReading(reports, "b"), { completedTime_s: 1.9, step_s: DT });
  assert.equal(paneClockReading(reports, "c"), undefined);
});

test("stats fall back to the channel that published", () => {
  assert.deepEqual(paneStatsFrom({ representedVolumeCellSum: 512, pressureIterations: 64, encodedSteps: 3 }, 7.25), {
    volumeCells: 512,
    volumeCellSize_m: undefined,
    volumeDrift: undefined,
    pressureIterations: 64,
    msPerStep: 7.25,
    encodedSteps: 3,
  });
  // The executed count outranks the configured ceiling, and the conservative
  // mass sum outranks the represented one.
  assert.deepEqual(paneStatsFrom(
    { volumeCellSum: 8, representedVolumeCellSum: 9, pressureIterations: 64, pressureIterationsExecuted: 11 }, 0),
    {
      volumeCells: 8,
      volumeCellSize_m: undefined,
      volumeDrift: undefined,
      pressureIterations: 11,
      msPerStep: undefined,
      encodedSteps: undefined,
    });
  assert.deepEqual(paneStatsFrom(null, undefined), {
    volumeCells: undefined,
    volumeCellSize_m: undefined,
    volumeDrift: undefined,
    pressureIterations: undefined,
    msPerStep: undefined,
    encodedSteps: undefined,
  });
});
