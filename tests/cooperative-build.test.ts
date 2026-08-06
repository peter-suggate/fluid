import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  liveSceneBrickCoordinatesForRegionsSteps,
  liveSceneReachableBrickCoordinatesSteps,
} from "../lib/webgpu-octree-sparse-bricks";
import { encodeSvoNodeMipMorton, planSvoNodeMipPyramid } from "../lib/svo-node-mip-pyramid";
import {
  completeCooperativeBuild,
  driveCooperativeBuild,
  isCooperativeBuildAbort,
} from "../lib/cooperative-build";
import {
  planAdaptiveSparseBrickOctree,
  planAdaptiveSparseBrickOctreeSteps,
  type AdaptiveSparseBrickPlanOptions,
} from "../lib/adaptive-sparse-brick-plan";
import type { SparseBrickCoordinate } from "../lib/sparse-brick-octree";

function solidBlock(size: number): SparseBrickCoordinate[] {
  const result: SparseBrickCoordinate[] = [];
  for (let z = 0; z < size; z += 1) for (let y = 0; y < size; y += 1) for (let x = 0; x < size; x += 1) {
    result.push({ x, y, z });
  }
  return result;
}

test("a build that is never aborted runs to completion and returns its value", async () => {
  let steps = 0;
  function* build(): Generator<unknown, string, undefined> {
    for (let index = 0; index < 1000; index += 1) { steps += 1; yield; }
    return "done";
  }
  assert.equal(await driveCooperativeBuild(build()), "done");
  assert.equal(steps, 1000);
});

test("an aborted build stops at its next yield and never reaches the end", async () => {
  let steps = 0;
  let finalized = false;
  const abort = new AbortController();
  function* build(): Generator<unknown, string, undefined> {
    try {
      for (;;) { steps += 1; yield; }
    } finally { finalized = true; }
  }
  // Abort while the first slice is still in flight. The driver only reads the
  // signal at a slice boundary, so this measures the real bound: a build stops
  // one slice after it is told to, not at the end of its work.
  const driven = driveCooperativeBuild(build(), { signal: abort.signal, sliceBudget_ms: 0 });
  abort.abort();
  await assert.rejects(driven, (error: unknown) => isCooperativeBuildAbort(error));
  const stepsAtAbort = steps;
  // The generator was closed, so its `finally` ran and it cannot be resumed —
  // which is what makes "abandoned" mean "released" for a producer that owns
  // resources on the suspended stack.
  assert.equal(finalized, true);
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(steps, stepsAtAbort, "an abandoned build kept running after it was abandoned");
});

test("a build yields the thread often enough for a queued message to be serviced", async () => {
  // The property the render worker actually needs: while a build is running,
  // something posted to the event loop still gets to run, and not after the
  // build has finished.
  let buildSteps = 0;
  const spin_ms = 3;
  function* build(): Generator<unknown, void, undefined> {
    for (let index = 0; index < 60; index += 1) {
      const until = performance.now() + spin_ms;
      while (performance.now() < until) { /* a slice of unyielding work */ }
      buildSteps += 1;
      yield;
    }
  }
  let servicedAfterSteps = -1;
  const channel = new MessageChannel();
  channel.port1.onmessage = () => { servicedAfterSteps = buildSteps; channel.port1.close(); channel.port2.close(); };
  channel.port2.postMessage(0);
  await driveCooperativeBuild(build(), { sliceBudget_ms: 8 });
  assert.ok(servicedAfterSteps >= 0, "the queued message was never delivered");
  assert.ok(servicedAfterSteps < 60,
    `the queued message waited for the whole build (delivered after ${servicedAfterSteps} of 60 steps)`);
});

test("the interruptible octree plan is the plan the one-shot call already produced", () => {
  const options: AdaptiveSparseBrickPlanOptions = {
    brickSize: 8,
    solverBricks: [],
    proxyBricks: solidBlock(8),
    maximumDepth: 3,
    solverLevel: 3,
    maximumEnvironmentCoarseningPower: 2,
  };
  // `planAdaptiveSparseBrickOctree` is the generator driven straight through,
  // so this is really asserting that yielding cannot change a plan — the thing
  // that would otherwise make the interruptible build a second definition of
  // what a scene is.
  const oneShot = planAdaptiveSparseBrickOctree(options);
  const stepped = completeCooperativeBuild(planAdaptiveSparseBrickOctreeSteps(options));
  assert.deepEqual(stepped, oneShot);
  assert.ok(oneShot.leaves.length > 0);
});

test("brick selection offers yields proportional to the bricks it visits", () => {
  // Both halves of the claim, which between them were 20 s of the 36 s build at
  // environment refinement depth 3 and offered nothing.
  const regions = Array.from({ length: 8 }, (_, index) => ({
    minimum: [index, 0, 0] as const,
    maximum: [index + 12, 12, 12] as const,
  }));
  let offers = 0;
  const regionSteps = liveSceneBrickCoordinatesForRegionsSteps(
    regions, [0, 0, 0], [1, 1, 1], 1, [64, 64, 64]);
  let coordinates: SparseBrickCoordinate[] = [];
  for (;;) { const step = regionSteps.next(); if (step.done) { coordinates = step.value; break; } offers += 1; }
  assert.ok(coordinates.length > 1000);
  assert.ok(offers >= 1, `region enumeration offered ${offers} yields for ${coordinates.length} bricks`);

  // A solid that reaches everywhere, so every candidate pays the distance call
  // this pass exists to bound.
  const solids = [{
    minimum: [-1e6, -1e6, -1e6] as const,
    maximum: [1e6, 1e6, 1e6] as const,
    distance_m: () => -1,
  }];
  let reachOffers = 0;
  const reachSteps = liveSceneReachableBrickCoordinatesSteps(coordinates, solids, [0, 0, 0], [1, 1, 1], 1);
  for (;;) { const step = reachSteps.next(); if (step.done) break; reachOffers += 1; }
  assert.ok(reachOffers >= coordinates.length / 128,
    `reach narrowing offered ${reachOffers} yields for ${coordinates.length} candidates`);
});

test("the world build evaluates the claim before the plan call, not inside its arguments", () => {
  // The bug this pins cost 17.5 s of frozen worker and was invisible to every
  // structural check: the narrowing was an argument-literal expression, so it
  // ran after the stage mark and before the plan generator's first offer —
  // between two yield points, where nothing can interrupt. An expression's
  // position is load-bearing here, so it is asserted.
  const world = readFileSync(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url), "utf8");
  const build = world.slice(world.indexOf("  private *buildSteps("));
  assert.match(build, /const reachablePrimitiveBricks = yield\* liveSceneReachableBrickCoordinatesSteps\(/,
    "the reach narrowing must be its own interruptible statement");
  assert.match(build, /proxyBricks: \[\s*\.\.\.reachablePrimitiveBricks,/,
    "the plan call must consume the already-computed claim rather than compute one");
  const planCall = build.slice(build.indexOf("yield* planAdaptiveSparseBrickOctreeSteps({"));
  const planArguments = planCall.slice(0, planCall.indexOf("\n    });"));
  assert.doesNotMatch(planArguments, /liveSceneReachableBrickCoordinates|liveSceneBrickCoordinatesForRegions|sparseSceneTerrainClaimCoordinates/,
    "no claim may be computed inside the plan call's arguments; that is a block no yield can reach");
});

test("the derived-lighting block yields in its prologue and nowhere past its first arena", () => {
  // The block's arenas are locals until its four closing assignments, so a
  // build abandoned inside it would leak an atlas nothing can reach. What makes
  // the prologue safe is that it is pure CPU: the opacity floor, the base-page
  // seeds, the address plan and the direct page table own nothing. Both halves
  // of that are asserted, because the boundary is the whole argument.
  const world = readFileSync(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url), "utf8");
  const build = world.slice(world.indexOf("  private *buildSteps("));
  const block = build.slice(build.indexOf('reportStage("Plan the node-mip pyramid")'));
  const firstArena = block.indexOf("new WebGpuLiveSvoNodeMipPyramid(");
  assert.ok(firstArena > 0, "the derived-lighting block no longer builds the pyramid where this test looks");
  const prologue = block.slice(0, firstArena);
  assert.match(prologue, /yield\* liveSvoPlanBasePagesSteps\(/);
  assert.match(prologue, /yield\* planSvoNodeMipAddressesSteps\(/);

  const closing = block.indexOf("    this.nodeMipPyramid = nodeMipPyramid;");
  assert.ok(closing > firstArena);
  const arenas = block.slice(firstArena, closing);
  assert.doesNotMatch(arenas, /\byield\b/,
    "no yield may sit between the block's first device resource and the assignments that make it reachable");
});

test("the pyramid plan stays level-major Morton ordered", () => {
  // The comparator was rewritten to decorate the Morton code rather than encode
  // two BigInts per call — the sort is the one part of the plan no yield can
  // interrupt, so its cost is the residual stall. This pins the order that
  // rewrite has to preserve: the directory is binary-searched in exactly it.
  const occupied: { level: number; coordinate: readonly [number, number, number] }[] = [];
  for (let z = 0; z < 6; z += 1) for (let y = 0; y < 6; y += 1) for (let x = 0; x < 6; x += 1) {
    occupied.push({ level: 0, coordinate: [x, y, z] });
  }
  const plan = planSvoNodeMipPyramid({ generation: 1, occupiedPages: occupied, levelCount: 4 });
  assert.ok(plan.pages.length > occupied.length);
  for (let index = 1; index < plan.pages.length; index += 1) {
    const previous = plan.pages[index - 1].key;
    const current = plan.pages[index].key;
    assert.ok(previous.level <= current.level, "pages must be level-major");
    if (previous.level !== current.level) continue;
    assert.ok(encodeSvoNodeMipMorton(previous.coordinate) < encodeSvoNodeMipMorton(current.coordinate),
      "pages within a level must be strictly Morton ascending");
  }
  // Slot numbering is the array order; the atlas addressing depends on it.
  plan.pages.forEach((page, slot) => assert.equal(page.slot, slot));
});

test("the octree plan offers yields inside its own passes, not only around them", () => {
  // Interrupt granularity has to be a property of the plan, not of the scene:
  // one offer per call would make a large scene exactly as unbreakable as it
  // was before. Scaled against the node count so this cannot pass by accident
  // on a plan that happens to be tiny.
  const options: AdaptiveSparseBrickPlanOptions = {
    brickSize: 8,
    solverBricks: [],
    proxyBricks: solidBlock(16),
    maximumDepth: 4,
    solverLevel: 4,
    maximumEnvironmentCoarseningPower: 0,
  };
  let offers = 0;
  const steps = planAdaptiveSparseBrickOctreeSteps(options);
  for (;;) { const step = steps.next(); if (step.done) break; offers += 1; }
  assert.ok(offers >= 8, `the plan offered only ${offers} yields for a 16^3 claim`);
});
