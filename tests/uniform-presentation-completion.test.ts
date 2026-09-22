import assert from "node:assert/strict";
import test from "node:test";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { planGPUAdvance } from "../lib/core/tall-cell-diagnostics";

test("paused uniform presentations never submit zero-duration physics or change the surface revision", () => {
  for (const lastTime of [0, 0.017, 1]) {
    const info = { encodedSteps: 7, submittedTime_s: lastTime };
    const solver = Object.assign(Object.create(WebGPUUniformReferenceSolver.prototype), {
      disposed: false, paperTimeStep: false, lastTime, info,
      scene: { numerics: { maxDt_s: 0.017 } },
      // There is intentionally no GPU device: a paused advance must return
      // before touching pipelines, queue submissions, or field resources.
    }) as WebGPUUniformReferenceSolver;
    for (let repaint = 0; repaint < 120; repaint++) {
      assert.equal(solver.advanceTo(lastTime), false);
    }
    assert.deepEqual(info, { encodedSteps: 7, submittedTime_s: lastTime });
    assert.equal(planGPUAdvance(lastTime - 0.01, lastTime, 0.017), undefined);
  }
  const step = planGPUAdvance(0.017, 0, 0.017)!;
  assert.equal(step.dt_s, 0.017);
  assert.equal(step.nextTime_s, 0.017);
  assert.equal(planGPUAdvance(0.017, step.nextTime_s, 0.017), undefined);
});

test("a completed uniform presentation retires while later queue work is still pending", async () => {
  let completeLater!: () => void;
  const later = new Promise<void>(resolve => { completeLater = resolve; });
  const solver = Object.assign(Object.create(WebGPUUniformReferenceSolver.prototype), {
    device: { queue: { onSubmittedWorkDone: () => later } },
  }) as WebGPUUniformReferenceSolver;
  let retired = false;
  const retirement = solver.assertSimulationHealthy(Promise.resolve()).then(() => { retired = true; });
  // A later GPU submission is deliberately left incomplete. Its completion
  // must not be needed to release the already finished presentation slot.
  await new Promise<void>(resolve => setImmediate(resolve));
  assert.equal(retired, true);
  completeLater(); await retirement;
});

test("uniform health retains its standalone fence and propagates failed supplied fences", async () => {
  let finish!: () => void;
  const fence = new Promise<void>(resolve => { finish = resolve; });
  const solver = Object.assign(Object.create(WebGPUUniformReferenceSolver.prototype), {
    device: { queue: { onSubmittedWorkDone: () => fence } },
  }) as WebGPUUniformReferenceSolver;
  let retired = false;
  const standalone = solver.assertSimulationHealthy().then(() => { retired = true; });
  await new Promise<void>(resolve => setImmediate(resolve)); assert.equal(retired, false);
  finish(); await standalone; assert.equal(retired, true);
  const failure = new Error("device lost before frame completed");
  await assert.rejects(solver.assertSimulationHealthy(Promise.reject(failure)), error => error === failure);
});
