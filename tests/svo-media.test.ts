import assert from "node:assert/strict";
import test from "node:test";

import {
  attenuateSvoMedium,
  evaluateSvoDielectricTransition,
  resolveSvoMediumBoundaryGroup,
  SVO_MEDIA_LIMITS,
  svoMediaWGSL,
  traceSvoMediaRay,
  type SvoMediaBoundary,
  type SvoMediaBoundarySource,
} from "../lib/svo-media";
import { WATER_OPTICS } from "../lib/webgpu-lighting";

test("Snell refraction and Fresnel are physical at normal and oblique incidence", () => {
  const normal = evaluateSvoDielectricTransition([0, -1, 0], [0, 1, 0], "air", "water");
  assert.equal(normal.totalInternalReflection, false);
  assert.deepEqual(normal.refractedDirection, [0, -1, 0]);
  assert.ok(Math.abs(normal.fresnel - WATER_OPTICS.fresnelF0) < 5e-5);

  const sineIncident = 0.6;
  const oblique = evaluateSvoDielectricTransition(
    [sineIncident, -Math.sqrt(1 - sineIncident ** 2), 0], [0, 1, 0], "air", "water",
  );
  assert.ok(oblique.refractedDirection);
  assert.ok(Math.abs((oblique.refractedDirection?.[0] ?? 0) - sineIncident / WATER_OPTICS.indexOfRefraction) < 1e-12,
    "Snell preserves n1 sin(theta1) = n2 sin(theta2)");
  assert.ok(oblique.fresnel > normal.fresnel);
});

test("water-to-air above the critical angle yields total internal reflection", () => {
  const result = evaluateSvoDielectricTransition([0.9, Math.sqrt(1 - 0.9 ** 2), 0], [0, 1, 0], "water", "air");
  assert.equal(result.totalInternalReflection, true);
  assert.equal(result.refractedDirection, undefined);
  assert.equal(result.fresnel, 1);
  assert.ok(result.reflectedDirection[1] < 0);
});

test("Beer-Lambert water absorption reuses the canonical spectral coefficients", () => {
  assert.deepEqual(attenuateSvoMedium([1, 1, 1], "air", 20), [1, 1, 1]);
  const water = attenuateSvoMedium([1, 1, 1], "water", 2);
  assert.ok(water[0] < water[1] && water[1] < water[2]);
  assert.ok(Math.abs(water[0] - Math.exp(-WATER_OPTICS.absorption[0] * 2)) < 1e-12);
});

test("coincident exits precede entries and never create a fake air layer", () => {
  const waterExit: SvoMediaBoundary = { t_m: 2, medium: "water", geometricNormal: [0, 1, 0], boundaryId: 1 };
  const glassEntry: SvoMediaBoundary = { t_m: 2 + 5e-6, medium: "glass", geometricNormal: [0, -1, 0], boundaryId: 2 };
  const resolved = resolveSvoMediumBoundaryGroup(["air", "water"], [glassEntry, waterExit], [0, 1, 0], 1e-5);
  assert.equal(resolved.from, "water");
  assert.equal(resolved.to, "glass");
  assert.deepEqual(resolved.nextStack, ["air", "glass"]);
  assert.throws(
    () => resolveSvoMediumBoundaryGroup(["air", "water"], [waterExit, { ...glassEntry, t_m: 2.1 }], [0, 1, 0], 1e-5),
    /coincidence epsilon/,
  );
});

test("submerged opaque contact terminates after water absorption", () => {
  const result = traceSvoMediaRay(
    { origin_m: [0, -1, 0], direction: [0, 1, 0], maximumDistance_m: 5 },
    () => ({ status: "hit", boundaries: [{ t_m: 2, medium: "opaque", geometricNormal: [0, -1, 0] }] }),
    { initialMediaStack: ["air", "water"] },
  );
  assert.equal(result.status, "opaque");
  if (result.status === "opaque") assert.equal(result.mediumBefore, "water");
  assert.deepEqual(result.throughput, [0, 0, 0]);
  assert.equal(result.distance_m, 2);
});

test("thin-wall glass attenuates without changing medium or ray direction", () => {
  let calls = 0;
  const source: SvoMediaBoundarySource = () => {
    calls += 1;
    if (calls === 1) return { status: "hit", boundaries: [{
      t_m: 1, medium: "glass", geometricNormal: [0, 1, 0], thinWall: true, thinWallTint: [0.8, 0.9, 1],
    }] };
    return { status: "miss" };
  };
  const result = traceSvoMediaRay(
    { origin_m: [0, 1, 0], direction: [0, -1, 0], maximumDistance_m: 3 }, source,
    { continuationEpsilon_m: 0.001 },
  );
  assert.equal(result.status, "escaped");
  assert.deepEqual(result.mediaStack, ["air"]);
  assert.deepEqual(result.direction, [0, -1, 0]);
  assert.equal(result.counts.transitions, 1);
  assert.equal(result.counts.transmissions, 1);
  assert.ok(result.throughput[0] < result.throughput[1] && result.throughput[1] < result.throughput[2]);
});

test("entry and exit update the stack in order under fixed transition accounting", () => {
  let call = 0;
  const source: SvoMediaBoundarySource = () => {
    call += 1;
    if (call === 1) return { status: "hit", boundaries: [{ t_m: 1, medium: "water", geometricNormal: [0, 1, 0] }] };
    if (call === 2) return { status: "hit", boundaries: [{ t_m: 2, medium: "water", geometricNormal: [0, -1, 0] }] };
    return { status: "miss" };
  };
  const result = traceSvoMediaRay({ origin_m: [0, 2, 0], direction: [0, -1, 0], maximumDistance_m: 6 }, source);
  assert.equal(result.status, "escaped");
  assert.deepEqual(result.mediaStack, ["air"]);
  assert.equal(result.counts.transitions, 2);
  assert.equal(result.counts.transmissions, 2);
  assert.ok(result.throughput[0] < result.throughput[2], "the segment between entry and exit absorbs in water");
});

test("transition, reflection, transmission, query, and source exhaustion are explicit", () => {
  const entry = { t_m: 0.5, medium: "water", geometricNormal: [0, 1, 0] } as const;
  const transition = traceSvoMediaRay(
    { origin_m: [0, 1, 0], direction: [0, -1, 0], maximumDistance_m: 4 },
    () => ({ status: "hit", boundaries: [{ ...entry, medium: "glass", thinWall: true }] }),
    { maximumTransitions: 1, maximumTransmissions: 8 },
  );
  assert.equal(transition.status, "exhausted");
  if (transition.status === "exhausted") assert.equal(transition.exhaustedBy, "transitions");

  const transmission = traceSvoMediaRay(
    { origin_m: [0, 1, 0], direction: [0, -1, 0], maximumDistance_m: 4 },
    () => ({ status: "hit", boundaries: [{ ...entry, medium: "glass", thinWall: true }] }),
    { maximumTransmissions: 1, maximumTransitions: 8 },
  );
  assert.equal(transmission.status, "exhausted");
  if (transmission.status === "exhausted") assert.equal(transmission.exhaustedBy, "transmissions");

  const reflection = traceSvoMediaRay(
    { origin_m: [0, -1, 0], direction: [0.9, Math.sqrt(1 - 0.9 ** 2), 0], maximumDistance_m: 4 },
    ({ ray }) => ({ status: "hit", boundaries: [{
      t_m: 0.5,
      medium: "water",
      geometricNormal: [0, Math.sign(ray.direction[1]), 0],
    }] }),
    { initialMediaStack: ["air", "water"], maximumReflections: 1, maximumTransitions: 8 },
  );
  assert.equal(reflection.status, "exhausted");
  if (reflection.status === "exhausted") assert.equal(reflection.exhaustedBy, "reflections");

  const queries = traceSvoMediaRay(
    { origin_m: [0, 1, 0], direction: [0, -1, 0], maximumDistance_m: 4 },
    ({ queryIndex }) => ({ status: "hit", boundaries: [{ t_m: 0.1, medium: "glass", geometricNormal: [0, 1, 0], thinWall: true, boundaryId: queryIndex }] }),
    { maximumBoundaryQueries: 2, maximumTransitions: 8, maximumTransmissions: 8 },
  );
  assert.equal(queries.status, "exhausted");
  if (queries.status === "exhausted") assert.equal(queries.exhaustedBy, "queries");

  const source = traceSvoMediaRay(
    { origin_m: [0, 1, 0], direction: [0, -1, 0], maximumDistance_m: 4 },
    () => ({ status: "exhausted", reason: "leaf budget" }),
  );
  assert.equal(source.status, "exhausted");
  if (source.status === "exhausted") assert.equal(source.exhaustedBy, "source");
});

test("invalid stack transitions fail closed instead of leaking through", () => {
  const result = traceSvoMediaRay(
    { origin_m: [0, 1, 0], direction: [0, 1, 0], maximumDistance_m: 2 },
    () => ({ status: "hit", boundaries: [{ t_m: 0.5, medium: "water", geometricNormal: [0, 1, 0] }] }),
  );
  assert.equal(result.status, "invalid");
  assert.deepEqual(result.throughput, [0, 0, 0]);
});

test("WGSL media oracle is binding-free and uses only fixed secondary-ray loops", () => {
  assert.equal(SVO_MEDIA_LIMITS.transitions, 8);
  assert.match(svoMediaWGSL, /fn svoMediaRefract/);
  assert.match(svoMediaWGSL, /let k=1\.0-eta\*eta/);
  assert.match(svoMediaWGSL, /fn svoMediaBeer/);
  assert.match(svoMediaWGSL, /Exits precede entries/);
  assert.match(svoMediaWGSL, /SVO_MEDIA_THIN_WALL/);
  assert.match(svoMediaWGSL, /SVO_MEDIA_MAX_TRANSITIONS:u32=8u/);
  assert.match(svoMediaWGSL, /SVO_MEDIA_MAX_REFLECTIONS:u32=4u/);
  assert.match(svoMediaWGSL, /SVO_MEDIA_MAX_TRANSMISSIONS:u32=8u/);
  assert.match(svoMediaWGSL, /for\(var query=0u;query<SVO_MEDIA_MAX_QUERIES;query\+=1u\)/);
  assert.match(svoMediaWGSL, /let step=svoMediaNext\(ray,stack,stackSize\)/);
  assert.doesNotMatch(svoMediaWGSL, /@group|@binding/);
  assert.doesNotMatch(svoMediaWGSL, /fn svoTraceMedia[\s\S]*svoTraceMedia\(/,
    "media traversal must not recurse");
});
