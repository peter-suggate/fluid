import assert from "node:assert/strict";
import test from "node:test";

import {
  SVO_VISIBILITY_LIMITS,
  createBiasedSvoVisibilityRay,
  svoVisibilityRaysWGSL,
  traceSvoVisibilityRay,
  type SvoVisibilityHit,
  type SvoVisibilityStepSource,
} from "../lib/svo-visibility-rays";

const counts = (nodeVisits = 1, leafVisits = 1, workItems = 1) => ({ nodeVisits, leafVisits, workItems });

function ray(maximumLightDistance_m = 10) {
  return createBiasedSvoVisibilityRay({
    surfacePosition_m: [1, 2, 3],
    geometricNormal: [1, 0, 0],
    directionToLight: [1, 0, 0],
    maximumLightDistance_m,
    cellSize_m: [0.25, 1, 2],
  }, { originBiasCells: 0.1 });
}

test("origin bias follows the geometric normal and anisotropic cell scale", () => {
  const result = createBiasedSvoVisibilityRay({
    surfacePosition_m: [1, 2, 3],
    geometricNormal: [1, 1, 0],
    directionToLight: [0, -4, 0],
    maximumLightDistance_m: 10,
    cellSize_m: [0.25, 1, 2],
  }, { originBiasCells: 0.2 });
  const projected = (0.25 + 1) / Math.sqrt(2);
  assert.ok(Math.abs(result.originBias_m - 0.2 * projected) < 1e-12);
  assert.ok(result.origin_m[0] < 1 && result.origin_m[1] < 2, "bias chooses the ray-facing normal hemisphere");
  assert.deepEqual(result.direction, [0, -1, 0]);
  assert.ok(result.tMax_m < 10, "the biased ray still stops at the original light plane");
});

test("opaque candidates conservatively early-out after one nearest-hit query", () => {
  let calls = 0;
  const blocker: SvoVisibilityHit = { t_m: 2, opaque: true, materialId: 17, ownerId: 4 };
  const result = traceSvoVisibilityRay(ray(), () => {
    calls += 1;
    return { status: "hit", counts: counts(3, 1, 7), hit: blocker };
  });
  assert.equal(calls, 1);
  assert.deepEqual(result, {
    status: "occluded",
    transmittance: [0, 0, 0],
    counts: { nodeVisits: 3, leafVisits: 1, workItems: 7, intersections: 1 },
    blocker,
  });
});

test("visibility never accepts blockers beyond the finite light distance", () => {
  const result = traceSvoVisibilityRay(ray(3), () => ({
    status: "hit", counts: counts(), hit: { t_m: 5, opaque: true },
  }));
  assert.equal(result.status, "visible");
  assert.deepEqual(result.transmittance, [1, 1, 1]);
});

test("bounded transmissive visibility multiplies RGB attenuation and advances monotonically", () => {
  const hits: SvoVisibilityHit[] = [
    { t_m: 1, opaque: false, transmittance: [0.8, 0.6, 0.4] },
    { t_m: 2, opaque: false, transmittance: [0.5, 0.5, 0.25] },
  ];
  const cursors: number[] = [];
  const source: SvoVisibilityStepSource = ({ tMin_m }) => {
    cursors.push(tMin_m);
    const hit = hits.shift();
    return hit ? { status: "hit", counts: counts(), hit } : { status: "miss", counts: counts(1, 0, 1) };
  };
  const result = traceSvoVisibilityRay(ray(), source, { allowTransmission: true, continuationBias_m: 0.01 });
  assert.equal(result.status, "visible");
  assert.deepEqual(result.transmittance, [0.4, 0.3, 0.1]);
  assert.deepEqual(cursors, [0, 1.01, 2.01]);
  assert.equal(result.counts.intersections, 2);

  const conservative = traceSvoVisibilityRay(ray(), () => ({
    status: "hit", counts: counts(), hit: { t_m: 1, opaque: false, transmittance: [1, 1, 1] },
  }));
  assert.equal(conservative.status, "occluded", "transmission is explicitly opt-in");
});

test("node, leaf, work, source, and intersection exhaustion remain explicit and fail closed", () => {
  const cases = [
    { option: { maximumNodeVisits: 2 }, delta: counts(3, 0, 0), expected: "nodes" },
    { option: { maximumLeafVisits: 2 }, delta: counts(0, 3, 0), expected: "leaves" },
    { option: { maximumWorkItems: 2 }, delta: counts(0, 0, 3), expected: "work" },
  ] as const;
  for (const fixture of cases) {
    const result = traceSvoVisibilityRay(ray(), () => ({ status: "miss", counts: fixture.delta }), fixture.option);
    assert.equal(result.status, "exhausted");
    if (result.status === "exhausted") assert.equal(result.exhaustedBy, fixture.expected);
    assert.deepEqual(result.transmittance, [0, 0, 0]);
  }

  const source = traceSvoVisibilityRay(ray(), () => ({ status: "exhausted", counts: counts(), reason: "traversal stack" }));
  assert.equal(source.status, "exhausted");
  if (source.status === "exhausted") assert.equal(source.exhaustedBy, "source");

  const repeated = traceSvoVisibilityRay(ray(), ({ tMin_m }) => ({
    status: "hit", counts: counts(), hit: { t_m: tMin_m + 0.1, opaque: false, transmittance: [1, 1, 1] },
  }), { allowTransmission: true, maximumIntersections: 2, continuationBias_m: 0.001 });
  assert.equal(repeated.status, "exhausted");
  if (repeated.status === "exhausted") assert.equal(repeated.exhaustedBy, "intersections");
  assert.equal(repeated.counts.intersections, 2);
});

test("malformed source data returns an explicit invalid fail-closed result", () => {
  const invalidCount = traceSvoVisibilityRay(ray(), () => ({ status: "miss", counts: counts(-1, 0, 0) }));
  assert.equal(invalidCount.status, "invalid");
  assert.deepEqual(invalidCount.transmittance, [0, 0, 0]);

  const invalidHit = traceSvoVisibilityRay(ray(), () => ({
    status: "hit", counts: counts(), hit: { t_m: -1, opaque: true },
  }));
  assert.equal(invalidHit.status, "invalid");

  const invalidTransmission = traceSvoVisibilityRay(ray(), () => ({
    status: "hit", counts: counts(), hit: { t_m: 1, opaque: false, transmittance: [1.1, 1, 1] },
  }), { allowTransmission: true });
  assert.equal(invalidTransmission.status, "invalid");
});

test("WGSL oracle is binding-free, bounded, biased, and adapter-composable", () => {
  assert.equal(SVO_VISIBILITY_LIMITS.intersections, 8);
  assert.match(svoVisibilityRaysWGSL, /fn svoBiasedVisibilityRay/);
  assert.match(svoVisibilityRaysWGSL, /dot\(abs\(geometricNormal\),cellSize_m\)/);
  assert.match(svoVisibilityRaysWGSL, /maximumLightDistance_m-dot\(offset,directionToLight\)/);
  assert.match(svoVisibilityRaysWGSL, /SVO_VIS_MAX_NODES:u32 = 256u/);
  assert.match(svoVisibilityRaysWGSL, /SVO_VIS_MAX_LEAVES:u32 = 64u/);
  assert.match(svoVisibilityRaysWGSL, /SVO_VIS_MAX_WORK:u32 = 2048u/);
  assert.match(svoVisibilityRaysWGSL, /for\(var interaction=0u;interaction<SVO_VIS_MAX_INTERSECTIONS;interaction\+=1u\)/);
  assert.match(svoVisibilityRaysWGSL, /let step=svoVisibilityNext\(ray,cursor,remaining\)/);
  assert.match(svoVisibilityRaysWGSL, /step\.opaque!=0u\|\|!allowTransmission/);
  assert.match(svoVisibilityRaysWGSL, /step\.t_m>ray\.tMax_m/);
  assert.doesNotMatch(svoVisibilityRaysWGSL, /@group|@binding/);
  assert.doesNotMatch(svoVisibilityRaysWGSL, /fn svoTraceVisibility[\s\S]*svoTraceVisibility\(/,
    "visibility must use a fixed loop rather than recursion");
});
