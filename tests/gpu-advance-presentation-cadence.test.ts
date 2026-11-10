import assert from "node:assert/strict";
import test from "node:test";

import {
  canQueuePreparedGPUAdvance,
  FluidLabRenderer,
  presentationHeldByPendingFrame,
  submitNextPreparedGPUAdvance,
} from "../lib/core/webgpu-renderer";
import type { GPUSolverInstance } from "../lib/core/method-contract";
import { PaneClockHost } from "../lib/core/simulation/pane-clock";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { UNIFORM_MIXED_RECEIPT_RING } from "../lib/methods/uniform/uniform-mixed-frame";

const DT = 1 / 30;
const RAF_S = 1 / 60;
const DEPTH = 2;
/** Animation frames a solver's mandatory frame receipt takes to land. */
const RECEIPT_FRAMES = 3;
/** Animation frames a submitted presentation takes to complete. */
const PRESENTATION_FRAMES = 2;

/**
 * A split-submission solver, exercising the shared receipt contract.
 *
 * One advance per call and never more: the paper step is pinned, so a target
 * clock that owes three steps still buys one. The state of that advance is
 * published as `advanceTo` returns — it is the receipt, not the state, that
 * stays outstanding — and no second advance may be encoded until it lands.
 */
class SplitSubmissionSolver {
  readonly info: { submittedTime_s?: number } = { submittedTime_s: 0 };
  private lastTime_s = 0;
  private receiptFrames = 0;

  get framePending() { return this.receiptFrames > 0; }
  awaitFrameCompletion() { return Promise.resolve(); }

  advanceTo(time_s: number) {
    if (this.framePending) return false;
    if (time_s < this.lastTime_s + DT - 1e-9) return false;
    this.lastTime_s += DT;
    this.info.submittedTime_s = this.lastTime_s;
    this.receiptFrames = RECEIPT_FRAMES;
    return true;
  }

  /** GPU work retired between animation frames. */
  settle() { if (this.receiptFrames > 0) this.receiptFrames -= 1; }
}

interface PresentedFrame {
  /** Host clock movement the reader sees when this presentation retires. */
  readonly publishedStep_s: number;
  readonly advancesRetired: number;
}

/**
 * The browser draw loop over one pane, reduced to its admission arithmetic.
 *
 * Structurally `FluidLabRenderer.draw`: the entry gate, the fixed throughput
 * window, one advance per presentation, the post-advance gate, and a
 * presentation that retires the advances queued behind it when its own GPU work
 * completes. `heldRule` is the only variable — it is the one decision this file
 * exists to pin down.
 */
function runDrawLoop(
  frames: number,
  heldRule: (framePending: boolean, advanceSubmitted: boolean) => boolean,
): PresentedFrame[] {
  const host = new PaneClockHost();
  const solver = new SplitSubmissionSolver();
  const presented: PresentedFrame[] = [];
  let pendingBatches = 0;
  let accountedSubmittedTime_s = 0;
  let presentationsInFlight = 0;
  let queuedAdvances: number[] = [];
  let presentationsCompleting: { at: number; advances: number[] }[] = [];

  for (let frame = 0; frame < frames; frame += 1) {
    const due = presentationsCompleting.filter((entry) => entry.at <= frame);
    presentationsCompleting = presentationsCompleting.filter((entry) => entry.at > frame);
    for (const completion of due) {
      presentationsInFlight -= 1;
      pendingBatches -= completion.advances.length;
      const before = host.completedTime();
      for (const time of completion.advances) host.completeAdvance(time);
      presented.push({
        publishedStep_s: host.completedTime() - before,
        advancesRetired: completion.advances.length,
      });
    }
    solver.settle();
    host.advance(RAF_S, DT);
    const target = host.targetTime();
    if (heldRule(solver.framePending, false)) continue;
    if (presentationsInFlight >= DEPTH) continue;
    let advanceSubmitted = false;
    if (canQueuePreparedGPUAdvance(pendingBatches, DEPTH)) {
      const previousSubmittedTime = accountedSubmittedTime_s;
      if ((solver.info.submittedTime_s ?? 0) <= previousSubmittedTime) {
        submitNextPreparedGPUAdvance(solver as unknown as GPUSolverInstance, target, []);
      }
      const submittedTime = solver.info.submittedTime_s ?? previousSubmittedTime;
      if (submittedTime > previousSubmittedTime) {
        accountedSubmittedTime_s = submittedTime;
        pendingBatches += 1;
        queuedAdvances.push(submittedTime);
        advanceSubmitted = true;
      }
    }
    if (heldRule(solver.framePending, advanceSubmitted)) continue;
    presentationsInFlight += 1;
    presentationsCompleting.push({ at: frame + PRESENTATION_FRAMES, advances: queuedAdvances });
    queuedAdvances = [];
  }
  return presented;
}

test("a split-submission advance is presented by the draw that submitted it", () => {
  const presented = runDrawLoop(120, presentationHeldByPendingFrame)
    .filter((frame) => frame.advancesRetired > 0);
  assert.ok(presented.length > 0, "the loop must present the advances it admits");
  for (const frame of presented) {
    assert.equal(frame.advancesRetired, 1,
      "one presented frame must carry exactly one solver advance");
    assert.ok(Math.abs(frame.publishedStep_s - DT) < 1e-9,
      `the host clock must publish one step per presented frame, not ${(frame.publishedStep_s / DT).toFixed(2)}`);
  }
});

test("holding the presentation for the advance it just made shows every second step", () => {
  // The regression this rule replaced: `framePending` alone dropped the picture
  // of each advance as it was submitted, so the renderer presented only once the
  // pending-batch ceiling refused the next one — two steps per presented frame,
  // in the image and in the published clock alike.
  const presented = runDrawLoop(120, (framePending) => framePending)
    .filter((frame) => frame.advancesRetired > 0);
  assert.ok(presented.length > 0, "the regression still presents, just half as often");
  assert.ok(presented.every((frame) => frame.advancesRetired === DEPTH),
    "the regression batches the whole throughput window into one presentation");
  assert.ok(presented.every((frame) => Math.abs(frame.publishedStep_s - DEPTH * DT) < 1e-9),
    "and publishes the host clock in two-step jumps");
});

test("NB-FLIP receipt backpressure does not defer an already-submitted image or drain the worker", () => {
  // Use the production admission/publication getters without constructing GPU
  // resources. A full ring has submitted every frame, including its final
  // fields; only admission of a further advance must wait for a receipt.
  const solver = Object.assign(Object.create(WebGPUUniformReferenceSolver.prototype), {
    mixedFrame: {}, mixedFramesInFlight: UNIFORM_MIXED_RECEIPT_RING,
    executionInfo: { submittedTime_s: 2 * DT, completedTime_s: 0 },
    awaitFrameCompletion: () => { throw new Error("must not drain submitted frames for presentation"); },
  }) as WebGPUUniformReferenceSolver;
  assert.equal(solver.framePending, true, "the next simulation advance remains blocked");
  assert.equal(solver.presentationPending, false, "its image can follow the final submission on the same queue");
  assert.equal(presentationHeldByPendingFrame(solver.framePending, true), false);
  Object.assign(solver, { mixedFramesInFlight: UNIFORM_MIXED_RECEIPT_RING - 1 });
  assert.equal(solver.framePending, false, "one receipt is enough to admit the next advance");
  Object.assign(solver, { deferredFrameValues: {} });
  assert.equal(solver.framePending, true, "an edit still drains outstanding advances before being applied");
  Object.assign(solver, { pressureFrameFailure: new Error("failed frame") });
  assert.equal(solver.presentationPending, true, "failed frames remain withheld");
});

test("motor-assembly contacts resolve on the GPU: a frame in flight neither holds admission nor presentation", () => {
  // Paddleboats used to step its wheel contacts on the CPU after a pose
  // readback, which held every frame until the readback landed. Contacts are
  // now part of the rigid kernel of the same submission, so a boat scene with
  // one frame in flight admits and presents like every other scene.
  const solver = Object.assign(Object.create(WebGPUUniformReferenceSolver.prototype), {
    mixedFrame: {}, mixedFramesInFlight: 1,
    executionInfo: { submittedTime_s: DT, completedTime_s: 0 },
  }) as WebGPUUniformReferenceSolver;
  assert.equal(solver.framePending, false, "a frame in flight below the receipt ring admits the next advance");
  assert.equal(solver.presentationPending, false, "its image follows the submission on the same queue");
  assert.equal((solver as unknown as { deferredFramePublication?: boolean }).deferredFramePublication, undefined,
    "no solver publishes after a readback any more");
});
