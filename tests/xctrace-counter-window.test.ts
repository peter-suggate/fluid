import assert from "node:assert/strict";
import test from "node:test";
import { planCounterWindow } from "../tools/profile-mini-dam-xctrace";

/**
 * An attached GPU-counter capture is only meaningful while the solver is
 * stepping, and every way of getting that wrong is silent: the recording
 * succeeds, the trace exports cleanly, and it contains nothing but other
 * processes' GPU work. The shipped failure was a 125-advance run whose window
 * opened 0.8 s after the last advance, because the schedule counted the delay
 * before spawning `xctrace` but not the ~2.7 s Instruments spends attaching.
 */

/** The lanes the profiler actually runs, plus the pace each was measured at. */
const LANES = [
  { name: "mini 125 (the regression)", steps: 125, perAdvanceMs: 47.2 },
  { name: "mini 500 (lane default)", steps: 500, perAdvanceMs: 49 },
  { name: "moving-interface 62", steps: 62, perAdvanceMs: 47.2 },
  { name: "ui 62", steps: 62, perAdvanceMs: 131 },
  { name: "degenerate 1", steps: 1, perAdvanceMs: 47.2 },
];

test("every lane's counter window opens and closes while the solver is stepping", () => {
  for (const lane of LANES) {
    for (const counterSeconds of [1, 2, 5]) {
      const plan = planCounterWindow({
        requestedSteps: lane.steps, perAdvanceMs: lane.perAdvanceMs, counterSeconds,
      });
      const stepping = plan.steps * lane.perAdvanceMs / 1000;
      const where = `${lane.name} @ ${counterSeconds}s`;
      assert.ok(plan.windowStartSeconds > 0, `${where}: window must open after construction`);
      assert.ok(plan.windowStartSeconds + counterSeconds < stepping,
        `${where}: window closes at ${(plan.windowStartSeconds + counterSeconds).toFixed(2)} s`
        + ` but stepping ends at ${stepping.toFixed(2)} s`);
      // The spawn has to precede the window by the attach latency, never equal
      // it -- conflating the two is the original bug.
      assert.ok(plan.spawnDelaySeconds < plan.windowStartSeconds,
        `${where}: xctrace must be spawned before the window is due`);
    }
  }
});

test("a run too short to hold a window is lengthened rather than mis-scheduled", () => {
  const plan = planCounterWindow({ requestedSteps: 125, perAdvanceMs: 50, counterSeconds: 2 });
  assert.equal(plan.extended, true);
  assert.ok(plan.steps > 125, "the step count must rise to fit the window");
  const longer = planCounterWindow({ requestedSteps: 500, perAdvanceMs: 50, counterSeconds: 2 });
  assert.equal(longer.extended, false, "a lane with room must keep the requested step count");
  assert.equal(longer.steps, 500);
});

test("the window is placed late in the run but never later than it can close", () => {
  const plan = planCounterWindow({ requestedSteps: 500, perAdvanceMs: 50, counterSeconds: 2 });
  // 60 % of a 25 s stepping phase, which has ample room on both sides.
  assert.ok(Math.abs(plan.windowStartSeconds - 15) < 0.01);
  // A short run cannot honour 60 %; it gets clamped to the last placement that
  // still leaves stepping after the window.
  const short = planCounterWindow({ requestedSteps: 149, perAdvanceMs: 50, counterSeconds: 2 });
  assert.ok(short.windowStartSeconds + 2 < 149 * 0.05);
});

test("an explicit warmup cannot ask for a window sooner than xctrace can attach", () => {
  const plan = planCounterWindow({
    requestedSteps: 500, perAdvanceMs: 50, counterSeconds: 2, requestedWarmupSeconds: 0,
  });
  assert.ok(plan.windowStartSeconds >= 2.7, "the window cannot open before the attach completes");
  assert.equal(plan.spawnDelaySeconds, 0, "and the attach is then issued immediately");
  const late = planCounterWindow({
    requestedSteps: 500, perAdvanceMs: 50, counterSeconds: 2, requestedWarmupSeconds: 8,
  });
  assert.ok(Math.abs(late.windowStartSeconds - 8) < 0.01, "a feasible warmup is honoured");
});

test("the labelled part of a window is shorter than the window", () => {
  // The encoder metadata stream -- the join key every interval is attributed
  // through -- takes about a second to start in an attached recording.
  const plan = planCounterWindow({ requestedSteps: 500, perAdvanceMs: 50, counterSeconds: 2 });
  assert.ok(plan.labelledSeconds > 0 && plan.labelledSeconds < 2);
  const brief = planCounterWindow({ requestedSteps: 500, perAdvanceMs: 50, counterSeconds: 1 });
  assert.equal(brief.labelledSeconds, 0, "a 1 s window is all warm-up and labels nothing");
});
