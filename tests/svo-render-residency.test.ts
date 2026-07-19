import assert from "node:assert/strict";
import test from "node:test";
import {
  SVO_RENDER_PUBLICATION_STAGES,
  SVO_RENDER_RESIDENCY_LIMITS,
  buildSvoRenderDirtyBrickRequests,
  gateSvoRenderGenerationPublication,
  planSvoRenderResidency,
  svoRenderGenerationGateWGSL,
  sweptSvoRenderBounds,
  velocitySweptSvoRenderBounds,
  type SvoRenderBounds,
  type SvoRenderBrickRequest,
  type SvoRenderLayerRevisions,
  type SvoRenderPublishedGeneration,
  type SvoRenderResidentBrick,
  type SvoRenderResidencyState,
} from "../lib/svo-render-residency";

const domain = {
  origin_m: [0, 0, 0] as const,
  brickSize_m: [1, 1, 1] as const,
  dimensionsBricks: [6, 2, 1] as const,
};

const revision = (staticRevision: number, dynamic: number, fluid: number): SvoRenderLayerRevisions => ({
  static: staticRevision, dynamic, fluid,
});

const bounds = (minimum: readonly [number, number, number], maximum: readonly [number, number, number]): SvoRenderBounds => ({
  minimum_m: minimum,
  maximum_m: maximum,
});

function request(
  coordinate: readonly [number, number, number],
  state: Exclude<SvoRenderResidencyState, "retired">,
  layer: "static" | "dynamic" | "fluid" = "fluid",
): SvoRenderBrickRequest {
  return { coordinate, key: coordinate.join(","), state, layers: [layer], causes: ["test"] };
}

function resident(
  coordinate: readonly [number, number, number],
  state: SvoRenderResidencyState,
  retiredFrames = 0,
): SvoRenderResidentBrick {
  return { coordinate, key: coordinate.join(","), state, layers: ["fluid"], retiredFrames };
}

test("unchanged renderer layer revisions produce no dirty payload work", () => {
  const unchanged = revision(4, 5, 6);
  const plan = buildSvoRenderDirtyBrickRequests({
    domain,
    previousRevisions: unchanged,
    currentRevisions: unchanged,
    changedBounds_m: { static: [bounds([0, 0, 0], [6, 2, 1])] },
    rigidChanges: [{ previousBounds_m: bounds([0, 0, 0], [1, 1, 1]), currentBounds_m: bounds([2, 0, 0], [3, 1, 1]) }],
    fluidPreactivation: [{ currentBounds_m: bounds([1, 0, 0], [2, 1, 1]), velocity_m_s: [10, 0, 0], deltaTime_s: 1 }],
  });
  assert.deepEqual(plan, { requests: [], changedLayers: [], completeDomainLayers: [] });
});

test("revision bounds dirty only intersecting bricks while missing bounds fail safe to the complete domain", () => {
  const localized = buildSvoRenderDirtyBrickRequests({
    domain,
    previousRevisions: revision(1, 2, 3),
    currentRevisions: revision(2, 2, 3),
    changedBounds_m: { static: [bounds([1.2, 0.2, 0], [2.8, 0.8, 1])] },
  });
  assert.deepEqual(localized.requests.map((entry) => entry.key), ["1,0,0", "2,0,0"]);
  assert.ok(localized.requests.every((entry) => entry.state === "active" && entry.layers[0] === "static"));
  assert.deepEqual(localized.changedLayers, ["static"]);
  assert.deepEqual(localized.completeDomainLayers, []);

  const conservative = buildSvoRenderDirtyBrickRequests({
    domain,
    previousRevisions: revision(1, 2, 3),
    currentRevisions: revision(1, 3, 3),
  });
  assert.equal(conservative.requests.length, 12);
  assert.deepEqual(conservative.completeDomainLayers, ["dynamic"]);
  assert.ok(conservative.requests.every((entry) => entry.state === "active"));
});

test("dynamic rigid work covers the complete swept old/new support", () => {
  const previousBounds = bounds([0.2, 0.2, 0], [0.8, 0.8, 1]);
  const currentBounds = bounds([3.2, 0.2, 0], [3.8, 0.8, 1]);
  assert.deepEqual(sweptSvoRenderBounds(previousBounds, currentBounds, 0.1), bounds([0.1, 0.1, -0.1], [3.9, 0.9, 1.1]));
  const plan = buildSvoRenderDirtyBrickRequests({
    domain,
    previousRevisions: revision(1, 7, 1),
    currentRevisions: revision(1, 8, 1),
    changedBounds_m: { dynamic: [] },
    rigidChanges: [{ previousBounds_m: previousBounds, currentBounds_m: currentBounds }],
  });
  assert.deepEqual(plan.requests.map((entry) => entry.key), ["0,0,0", "1,0,0", "2,0,0", "3,0,0"]);
  assert.ok(plan.requests.every((entry) => entry.causes.includes("rigid-swept-bounds")));
});

test("fluid requests prioritize current cores, velocity preactivation, then support halos", () => {
  const fluid = {
    currentBounds_m: bounds([1.1, 0.1, 0], [1.9, 0.9, 1]),
    velocity_m_s: [2, 0, 0] as const,
    deltaTime_s: 1,
  };
  assert.deepEqual(velocitySweptSvoRenderBounds(fluid), bounds([1.1, 0.1, 0], [3.9, 0.9, 1]));
  const plan = buildSvoRenderDirtyBrickRequests({
    domain,
    previousRevisions: revision(1, 1, 4),
    currentRevisions: revision(1, 1, 5),
    changedBounds_m: { fluid: [] },
    fluidPreactivation: [fluid],
    haloBricks: 1,
  });
  const states = Object.fromEntries(plan.requests.map((entry) => [entry.key, entry.state]));
  assert.equal(states["1,0,0"], "core");
  assert.equal(states["2,0,0"], "active");
  assert.equal(states["3,0,0"], "active");
  assert.equal(states["0,0,0"], "halo");
  assert.equal(states["4,0,0"], "halo");
  assert.equal(states["1,1,0"], "halo");
  assert.ok(plan.requests.find((entry) => entry.key === "1,0,0")?.causes.includes("fluid-surface-core"));
});

test("surface cores and active preactivation allocate before halos regardless of input order", () => {
  const plan = planSvoRenderResidency({
    desiredRequests: [request([0, 0, 0], "halo"), request([3, 0, 0], "core"), request([2, 0, 0], "halo"), request([1, 0, 0], "active")],
    capacity: 2,
    coarseCoverageComplete: true,
  });
  assert.deepEqual(plan.residents.map((entry) => [entry.key, entry.state]), [["3,0,0", "core"], ["1,0,0", "active"]]);
  assert.deepEqual(plan.unallocatedRequests.map((entry) => entry.state), ["halo", "halo"]);
  assert.equal(plan.coreCount, 1);
  assert.equal(plan.activeCount, 1);
  assert.equal(plan.haloCount, 0);
});

test("absent residents pass through explicit retired hysteresis before release", () => {
  const first = planSvoRenderResidency({
    desiredRequests: [], previousResidents: [resident([1, 0, 0], "core")], capacity: 2, coarseCoverageComplete: true,
  });
  assert.deepEqual(first.residents.map((entry) => [entry.state, entry.retiredFrames]), [["retired", 1]]);
  const second = planSvoRenderResidency({
    desiredRequests: [], previousResidents: first.residents, capacity: 2, coarseCoverageComplete: true,
  });
  assert.deepEqual(second.residents.map((entry) => [entry.state, entry.retiredFrames]), [["retired", 2]]);
  const third = planSvoRenderResidency({
    desiredRequests: [], previousResidents: second.residents, capacity: 2, coarseCoverageComplete: true,
  });
  assert.deepEqual(third.residents, []);

  const reactivated = planSvoRenderResidency({
    desiredRequests: [request([1, 0, 0], "active")], previousResidents: second.residents, capacity: 2, coarseCoverageComplete: true,
  });
  assert.deepEqual(reactivated.residents.map((entry) => [entry.state, entry.retiredFrames]), [["active", 0]]);
});

test("overflow remains publishable only through complete coarse coverage", () => {
  const desired = [request([0, 0, 0], "core"), request([1, 0, 0], "core"), request([2, 0, 0], "halo")];
  const safe = planSvoRenderResidency({ desiredRequests: desired, capacity: 1, coarseCoverageComplete: true });
  assert.equal(safe.overflowCount, 2);
  assert.equal(safe.coverage, "coarse-fallback");
  assert.equal(safe.publishable, true);
  assert.equal(safe.residents[0].state, "core");

  const unsafe = planSvoRenderResidency({ desiredRequests: desired, capacity: 1, coarseCoverageComplete: false });
  assert.equal(unsafe.coverage, "incomplete");
  assert.equal(unsafe.publishable, false);
});

test("capacity pressure evicts stale retired bricks before requested surface detail", () => {
  const plan = planSvoRenderResidency({
    desiredRequests: [request([4, 0, 0], "core")],
    previousResidents: [resident([0, 0, 0], "retired", 1), resident([1, 0, 0], "halo")],
    capacity: 1,
    coarseCoverageComplete: true,
  });
  assert.deepEqual(plan.residents.map((entry) => entry.key), ["4,0,0"]);
  assert.equal(plan.retiredCount, 0);
  assert.equal(plan.overflowCount, 0);
});

test("complete generation gating publishes all revisions together or retains the old snapshot", () => {
  const visible: SvoRenderPublishedGeneration = { completeGeneration: 10, revisions: revision(2, 4, 8) };
  const requiredStages = Object.values(SVO_RENDER_PUBLICATION_STAGES).reduce((mask, stage) => mask | stage, 0);
  const baseCandidate = {
    targetGeneration: 11,
    revisions: revision(3, 5, 9),
    requiredStages,
    completedStages: requiredStages,
    payloadWritesComplete: true,
    residency: { publishable: true, coverage: "coarse-fallback" as const },
  };
  const incompleteStages = gateSvoRenderGenerationPublication(visible, {
    ...baseCandidate, completedStages: requiredStages & ~SVO_RENDER_PUBLICATION_STAGES.fluidPayload,
  });
  assert.equal(incompleteStages.reason, "incomplete-stages");
  assert.deepEqual(incompleteStages.visible, visible);
  assert.deepEqual(gateSvoRenderGenerationPublication(visible, { ...baseCandidate, payloadWritesComplete: false }).visible, visible);
  assert.equal(gateSvoRenderGenerationPublication(visible, {
    ...baseCandidate, residency: { publishable: false, coverage: "incomplete" },
  }).reason, "incomplete-coverage");
  assert.equal(gateSvoRenderGenerationPublication(visible, { ...baseCandidate, targetGeneration: 12 }).reason, "generation-order");

  const published = gateSvoRenderGenerationPublication(visible, baseCandidate);
  assert.equal(published.reason, "published");
  assert.deepEqual(published.visible, { completeGeneration: 11, revisions: revision(3, 5, 9) });
});

test("fixed planning caps and binding-free publication mirror are explicit", () => {
  assert.deepEqual(SVO_RENDER_RESIDENCY_LIMITS, {
    maximumChangedRegions: 4_096,
    maximumBrickRequests: 1_048_576,
    defaultHaloBricks: 1,
    defaultRetireAfterFrames: 3,
  });
  assert.throws(() => buildSvoRenderDirtyBrickRequests({
    domain,
    previousRevisions: revision(0, 0, 0),
    currentRevisions: revision(1, 0, 0),
    maximumRequests: 1,
  }), /request cap/);
  assert.doesNotMatch(svoRenderGenerationGateWGSL, /@group|@binding/);
  assert.match(svoRenderGenerationGateWGSL, /targetGeneration==gate\.visibleGeneration\+1u/);
  assert.match(svoRenderGenerationGateWGSL, /completedStages&gate\.requiredStages/);
  assert.match(svoRenderGenerationGateWGSL, /payloadWritesComplete!=0u/);
  assert.match(svoRenderGenerationGateWGSL, /coarseCoverageComplete!=0u/);
});
