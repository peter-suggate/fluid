import test from "node:test";
import assert from "node:assert/strict";

import { collapseGPUFixedSteps } from "../lib/core/simulation/gpu-clock";
import {
  LOCKSTEP_IN_FLIGHT_DEPTH,
  PaneClockHost,
  PRIMARY_PANE_ID,
} from "../lib/core/simulation/pane-clock";
import { panesInStep } from "../lib/core/compare/divergence";

const DT = 1 / 30;
const EPSILON = 1e-9;

/** The single-pane transport arithmetic as it stood before the host existed. */
function referenceTargets(elapsed: readonly number[], dt: number): number[] {
  let accumulator = 0;
  let target = 0;
  return elapsed.map((step) => {
    accumulator += step;
    const collapsed = collapseGPUFixedSteps(accumulator, dt);
    if (collapsed.steps > 0) {
      accumulator = collapsed.remainder_s;
      target += collapsed.steps * dt;
    }
    return target;
  });
}

/** A pane that reports completion of whatever the host asked for, `lag` ticks late. */
class FakePane {
  private readonly queue: number[] = [];
  constructor(readonly id: string, private readonly lag: number) {}

  /** Admit the host's target into this pane's transport queue. */
  submit(target: number) {
    if (this.queue.length === 0 || this.queue[this.queue.length - 1] < target) this.queue.push(target);
  }

  /** Report every advance whose lag has expired. */
  drain(host: PaneClockHost, tick: number) {
    while (this.queue.length > 0 && tick >= this.lag) {
      const time = this.queue.shift();
      if (time === undefined) break;
      host.completeAdvance(time, this.id);
      if (this.lag > 0) break;
    }
  }
}

test("one pane ticks exactly as the single-pane transport did", () => {
  const host = new PaneClockHost();
  // Jittered frame times, including a stall long enough to owe several steps.
  const elapsed = [0.016, 0.017, 0.0, 0.4, 0.016, 0.0332, 0.0001, 0.0166, 0.0166, 0.25];
  const expected = referenceTargets(elapsed, DT);
  const observed = elapsed.map((step) => {
    host.advance(step, DT);
    return host.targetTime();
  });
  assert.deepEqual(observed, expected);
  assert.equal(host.lockstep(), false);
  assert.equal(host.paneCount(), 1);
  // A stall owes many steps and the single-pane host collapses them at once,
  // which is exactly what the two-pane barrier must refuse to do.
  assert.ok(expected[3] - expected[2] > DT * 2);
});

test("one pane keeps the depth-2 admission window", () => {
  const host = new PaneClockHost();
  host.advance(DT, DT);
  // Target is one step ahead of a pane that has completed nothing: today's
  // transport prepares one step beyond the last completed GPU state.
  assert.equal(host.completedTime(), 0);
  assert.equal(host.canAcceptNextStep(), false);
  host.completeAdvance(DT);
  assert.equal(host.canAcceptNextStep(), true);
  assert.equal(host.completedTime(), DT);
});

test("a lagging pane holds the target to one step past the slowest", () => {
  const host = new PaneClockHost();
  host.registerPane("b");
  assert.equal(host.lockstep(), true);
  assert.equal(host.paneCount(), 2);

  const a = new FakePane(PRIMARY_PANE_ID, 0);
  const b = new FakePane("b", 2);
  let maximumLead = 0;
  let steps = 0;
  for (let tick = 0; tick < 300; tick += 1) {
    steps += host.advance(0.0167, DT);
    a.submit(host.targetTime());
    b.submit(host.targetTime());
    a.drain(host, tick);
    b.drain(host, tick);
    const lead = host.targetTime() - host.completedTime();
    maximumLead = Math.max(maximumLead, lead);
    // The gate: the target never exceeds the slowest pane's completion by
    // more than one dt, at any observed instant.
    assert.ok(lead <= DT + EPSILON, `target ran ${lead} s past the slowest pane at tick ${tick}`);
    const paneA = host.paneCompletedTime(PRIMARY_PANE_ID) ?? 0;
    const paneB = host.paneCompletedTime("b") ?? 0;
    assert.ok(Math.abs(paneA - paneB) <= DT + EPSILON, `panes diverged by ${Math.abs(paneA - paneB)} s`);
  }
  assert.ok(steps > 0, "the barrier stalled the clock entirely");
  assert.ok(maximumLead > 0, "no step was ever in flight");
  // Wall-clock debt beyond one step is dropped rather than banked, so a pane
  // that recovers cannot sprint through a backlog.
  assert.ok(host.targetTime() <= 300 * DT + EPSILON);
});

test("the barrier releases exactly one step per opening", () => {
  const host = new PaneClockHost();
  host.registerPane("b");
  // Ten frames' worth of wall time owed at once.
  assert.equal(host.advance(DT * 10, DT), 1);
  assert.equal(host.targetTime(), DT);
  assert.equal(host.advance(DT * 10, DT), 0, "a shut barrier admits nothing");
  host.completeAdvance(DT, PRIMARY_PANE_ID);
  assert.equal(host.advance(DT, DT), 0, "one pane reaching the target is not the barrier");
  host.completeAdvance(DT, "b");
  assert.equal(host.completedTime(), DT);
  assert.equal(host.advance(DT, DT), 1);
  assert.ok(Math.abs(host.targetTime() - 2 * DT) < EPSILON);
});

test("completion is per pane and the host reports the minimum", () => {
  const host = new PaneClockHost();
  host.registerPane("b");
  host.advance(DT, DT);
  // The return is "the host minimum moved", not "this pane moved": pane A
  // alone cannot move a minimum that pane B still holds at zero, so nothing
  // republishes the product's time.
  assert.equal(host.completeAdvance(DT, PRIMARY_PANE_ID), false);
  assert.equal(host.completedTime(), 0);
  assert.equal(host.paneCompletedTime(PRIMARY_PANE_ID), DT);
  assert.equal(host.completeAdvance(DT, "b"), true);
  assert.equal(host.completedTime(), DT);
  // Completions past the target are clamped, and an unregistered pane is ignored.
  host.completeAdvance(DT * 9, PRIMARY_PANE_ID);
  assert.equal(host.paneCompletedTime(PRIMARY_PANE_ID), DT);
  assert.equal(host.completeAdvance(DT, "c"), false);
});

test("the minimum ignores a pane once it leaves, and pane A never leaves", () => {
  const host = new PaneClockHost();
  host.registerPane("b");
  host.advance(DT, DT);
  host.completeAdvance(DT, PRIMARY_PANE_ID);
  assert.equal(host.completedTime(), 0);
  assert.equal(host.unregisterPane("b"), true);
  assert.equal(host.completedTime(), DT, "the departed pane still held the clock");
  assert.equal(host.lockstep(), false);
  assert.equal(host.unregisterPane(PRIMARY_PANE_ID), false);
  assert.deepEqual(host.paneIds(), [PRIMARY_PANE_ID]);
  // A pane joining mid-run inherits the floor rather than stalling the barrier.
  host.registerPane("b");
  assert.equal(host.paneCompletedTime("b"), DT);
  assert.equal(host.registerPane("b"), false);
});

test("in-flight depth is pinned to one advance while two panes are registered", () => {
  const host = new PaneClockHost();
  assert.equal(host.lockstep(), false);
  host.registerPane("b");
  assert.equal(host.lockstep(), true);
  assert.equal(LOCKSTEP_IN_FLIGHT_DEPTH, 1);
});

test("STEP is refused until the slowest pane lands the previous step", () => {
  const host = new PaneClockHost();
  host.registerPane("b");
  assert.equal(host.step(DT), true);
  assert.equal(host.targetTime(), DT);
  assert.equal(host.step(DT), false);
  host.completeAdvance(DT, PRIMARY_PANE_ID);
  assert.equal(host.step(DT), false);
  host.completeAdvance(DT, "b");
  assert.equal(host.step(DT), true);
  assert.ok(Math.abs(host.targetTime() - 2 * DT) < EPSILON);
});

test("differing pane steps run the host at the smaller one", () => {
  const host = new PaneClockHost();
  host.registerPane("b");
  host.advance(0, DT);
  assert.equal(host.panesDtDiffer(), false);
  assert.equal(host.stepSize_s(), DT);

  host.setPaneDt("b", 0.004);
  assert.equal(host.panesDtDiffer(), true);
  assert.equal(host.stepSize_s(), 0.004);
  assert.equal(host.advance(DT, DT), 1);
  assert.equal(host.targetTime(), 0.004, "the host stepped at pane A's dt, not the smaller");

  host.setPaneDt("b", undefined);
  assert.equal(host.panesDtDiffer(), false);
  assert.equal(host.stepSize_s(), DT);
  // A larger declared step is still a difference the diff strip must name.
  host.setPaneDt("b", 0.05);
  assert.equal(host.panesDtDiffer(), true);
  assert.equal(host.stepSize_s(), DT);
});

test("reset returns every pane to its own t = 0", () => {
  const host = new PaneClockHost();
  host.registerPane("b");
  host.setPaneDt("b", 0.004);
  host.advance(DT, DT);
  host.completeAdvance(host.targetTime(), PRIMARY_PANE_ID);
  host.reset();
  assert.equal(host.targetTime(), 0);
  assert.equal(host.completedTime(), 0);
  assert.equal(host.paneCompletedTime(PRIMARY_PANE_ID), 0);
  assert.equal(host.paneCompletedTime("b"), 0);
  assert.equal(host.pendingTime_s(), 0);
  assert.deepEqual(host.paneIds(), [PRIMARY_PANE_ID, "b"], "reset dropped a pane");
  assert.equal(host.panesDtDiffer(), true, "reset dropped a declared step");
});

test("a pause rewinds the target to what was actually submitted", () => {
  const host = new PaneClockHost();
  host.advance(DT * 3, DT);
  assert.ok(Math.abs(host.targetTime() - 3 * DT) < EPSILON);
  host.completeAdvance(DT, PRIMARY_PANE_ID);
  host.schedulingPaused(2 * DT);
  assert.ok(Math.abs(host.targetTime() - 2 * DT) < EPSILON);
  assert.equal(host.pendingTime_s(), 0);
  // A reset can pause while the renderer still owns the previous solver: a
  // floor beyond the live target is a stale report and changes nothing.
  host.schedulingPaused(9 * DT);
  assert.ok(Math.abs(host.targetTime() - 2 * DT) < EPSILON);
  // With no submitted time the target falls back to what completed.
  const fresh = new PaneClockHost();
  fresh.advance(DT * 3, DT);
  fresh.completeAdvance(DT);
  fresh.schedulingPaused();
  assert.ok(Math.abs(fresh.targetTime() - DT) < EPSILON);
});

test("two panes draining independently cannot rewind each other's queue", () => {
  const host = new PaneClockHost();
  host.registerPane("b");
  host.step(DT);
  host.completeAdvance(DT, PRIMARY_PANE_ID);
  host.completeAdvance(DT, "b");
  host.step(DT);
  // Pane A drained one step further than pane B did.
  host.schedulingPaused(2 * DT, PRIMARY_PANE_ID);
  host.schedulingPaused(DT, "b");
  assert.ok(Math.abs(host.targetTime() - 2 * DT) < EPSILON,
    "pane B's shorter drain orphaned pane A's submitted advance");
});

/**
 * A pane as its renderer actually behaves.
 *
 * Two facts from `webgpu-renderer.ts` decide every pause case below:
 * `submitNextPreparedGPUAdvance` encodes toward the host target only while the
 * solver's own submitted time is *behind* it, and a solver never re-encodes a
 * time it has already submitted. So an advance the host forgets about is an
 * advance nobody will ever issue again.
 */
class QueuedPane {
  submitted = 0;
  private readonly inFlight: number[] = [];
  constructor(readonly id: string) {}

  /** One draw: admit the host's target to this pane's GPU queue, or refuse. */
  draw(host: PaneClockHost) {
    const target = host.targetTime();
    if (this.submitted + EPSILON >= target) return false;
    this.submitted = target;
    this.inFlight.push(target);
    return true;
  }

  /** The oldest admitted advance completes. */
  complete(host: PaneClockHost) {
    const time = this.inFlight.shift();
    if (time === undefined) return false;
    host.completeAdvance(time, this.id);
    return true;
  }

  /** The transport stopped this pane; it reports what it had already admitted. */
  reportDrain(host: PaneClockHost) { host.schedulingPaused(this.submitted, this.id); }
}

/** Run both panes for `frames` draws, completing everything they admit. */
function runPaired(host: PaneClockHost, a: QueuedPane, b: QueuedPane, frames: number) {
  for (let frame = 0; frame < frames; frame += 1) {
    host.advance(DT, DT);
    a.draw(host); b.draw(host);
    a.complete(host); b.complete(host);
  }
}

test("pausing both panes and resuming leaves them advancing and in step", () => {
  const host = new PaneClockHost();
  host.registerPane("b");
  const a = new QueuedPane(PRIMARY_PANE_ID);
  const b = new QueuedPane("b");
  runPaired(host, a, b, 6);
  const pausedAt = host.targetTime();
  assert.ok(pausedAt > 0, "the pair never started");

  // The transport is the host's: every attached pane drains.
  a.reportDrain(host); b.reportDrain(host);
  assert.equal(host.pendingTime_s(), 0, "wall-clock debt survived the pause");
  const clocksWhilePaused = {
    a: { completedTime_s: host.paneCompletedTime(PRIMARY_PANE_ID)!, step_s: DT },
    b: { completedTime_s: host.paneCompletedTime("b")!, step_s: DT },
    dtDiffers: false, identical: true,
  };
  assert.equal(panesInStep(clocksWhilePaused), true);

  // Nothing moves while paused: the controller's tick returns before `advance`.
  assert.equal(host.targetTime(), pausedAt);

  runPaired(host, a, b, 6);
  assert.ok(host.targetTime() > pausedAt, "the clock never resumed");
  const resumed = {
    a: { completedTime_s: host.paneCompletedTime(PRIMARY_PANE_ID)!, step_s: DT },
    b: { completedTime_s: host.paneCompletedTime("b")!, step_s: DT },
    dtDiffers: false, identical: true,
  };
  assert.equal(panesInStep(resumed), true);
  assert.ok(Math.abs(host.paneCompletedTime(PRIMARY_PANE_ID)! - host.paneCompletedTime("b")!) < EPSILON);
});

test("a pause never rewinds past an advance the other pane already encoded", () => {
  const host = new PaneClockHost();
  host.registerPane("b");
  const a = new QueuedPane(PRIMARY_PANE_ID);
  const b = new QueuedPane("b");
  runPaired(host, a, b, 4);
  const landed = host.targetTime();

  // The barrier opens and pane B's draw lands first; pane A has not drawn yet.
  host.advance(DT, DT);
  assert.ok(Math.abs(host.targetTime() - (landed + DT)) < EPSILON);
  assert.equal(b.draw(host), true);
  assert.equal(a.submitted, landed, "pane A drew when it should not have");

  // Pause. Pane A reports first, with a floor a whole step below pane B's
  // queue; the rewind has to wait for pane B rather than orphan that advance.
  a.reportDrain(host);
  assert.ok(Math.abs(host.targetTime() - (landed + DT)) < EPSILON,
    "the target rewound below pane B's encoded advance on pane A's report alone");
  b.reportDrain(host);
  assert.ok(Math.abs(host.targetTime() - (landed + DT)) < EPSILON);

  // Pane B's in-flight advance completes; pane A draws and completes the same
  // step. The barrier opens again, which it never does if the step was lost.
  b.complete(host);
  a.draw(host); a.complete(host);
  assert.ok(Math.abs(host.completedTime() - (landed + DT)) < EPSILON);
  assert.equal(host.canAcceptNextStep(), true);
  runPaired(host, a, b, 4);
  assert.ok(host.targetTime() > landed + DT, "the clock died at the pause");
});

test("a pane that never reports a drain cannot rewind the target on its own", () => {
  // The shipped bug: the transport flipped only pane A's run state, so pane B's
  // renderer never learned the clock had stopped and never drained. Deferring
  // the rewind is what keeps that harmless instead of fatal.
  const host = new PaneClockHost();
  host.registerPane("b");
  const a = new QueuedPane(PRIMARY_PANE_ID);
  const b = new QueuedPane("b");
  runPaired(host, a, b, 3);
  host.advance(DT, DT);
  b.draw(host);
  const target = host.targetTime();
  a.reportDrain(host);
  assert.ok(Math.abs(host.targetTime() - target) < EPSILON);
  // Debt is still dropped: owing wall time across a pause is never right.
  assert.equal(host.pendingTime_s(), 0);
});
