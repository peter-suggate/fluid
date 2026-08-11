import assert from "node:assert/strict";
import test from "node:test";

import { WebGPUOctreeLosassoCoarseBackend } from "../lib/webgpu-octree-losasso-backend";
import {
  octreeLosassoAdaptivePhiRedistanceInitializeWGSL,
  octreeLosassoAdaptivePhiWorklistReachWGSL,
} from "../lib/webgpu-octree-losasso-adaptive-phi.wgsl";
import { octreeLosassoAdaptiveVelocityWGSL }
  from "../lib/webgpu-octree-losasso-adaptive-velocity.wgsl";
import { WebGPUOctreeProjection } from "../lib/webgpu-octree";

const compact = (value: { toString(): string }): string =>
  value.toString().replace(/\s+/g, " ");

test("Losasso cadence reuse is an explicit candidate lifecycle state", () => {
  const due = compact(WebGPUOctreeProjection.prototype.encodeInactiveTopologyCandidateIfDue);
  const build = compact(WebGPUOctreeProjection.prototype.encodeInactiveTopologyCandidate);
  const flip = compact(WebGPUOctreeProjection.prototype.encodeReadyTopologyFlip);

  assert.match(due,
    /candidatePowerGeneration!==0[^]*cannot reuse while an inactive candidate is pending/,
    "a cadence skip must not silently overwrite an already pending transaction");
  assert.match(due, /topologyReusePending=true[^]*return false/,
    "a skipped tail must publish an explicit host reuse receipt");
  assert.match(due, /topologyCadenceSkipCount\+=1/,
    "host cadence skips need their own counter rather than impersonating GPU identity reuse");
  assert.match(build, /topologyReusePending=false[^]*powerAttemptGeneration/,
    "a real candidate must retire any stale reuse receipt before stamping its attempt");
  assert.match(flip,
    /candidatePowerGeneration===0&&this\.topologyReusePending[^]*topologyReusePending=false[^]*return/,
    "only an explicit reuse receipt may bypass the ready candidate transaction");
  assert.doesNotMatch(flip,
    /candidatePowerGeneration===0[^]*coarseDynamics\.topology\.advancesPerEpoch>1/,
    "ready reuse must not depend on construction cadence after a live dial change");
});

test("GPU exact-row identity receipts are counted once per accepted epoch", () => {
  const diagnostics = compact(WebGPUOctreeProjection.prototype.applyLosassoStepDiagnostics);
  assert.match(diagnostics, /authority\[5\]===1&&epoch!==0/,
    "the reduced authority's exact-identity receipt must remain observable");
  assert.match(diagnostics,
    /epoch!==this\.lastObservedExactTopologyReuseEpoch[^]*topologyExactIdentityCount\+=1/,
    "repeated diagnostics for one accepted epoch must not inflate the identity count");
});

test("same-topology surface advance transports phi and publishes a coherent two-bank tuple", () => {
  const advance = compact(WebGPUOctreeLosassoCoarseBackend.prototype.encodeAdaptiveSurfaceAdvance);
  const advection = compact(WebGPUOctreeLosassoCoarseBackend.prototype.encodeAdvection);
  assert.match(advance,
    /adaptiveVelocity\.encodeAcceptedRetainedFields\(broker\)[^]*adaptivePhi\.encodeAcceptedFieldClockSync\(broker\)/,
    "every transported-phi publication must retain complete velocity receipts and a coherent graph clock");
  assert.match(advance,
    /adaptivePhi\.encodeAcceptedAdvance\([^]*adaptiveVelocity\.transportSource\)[^]*adaptivePhi\.encodeAcceptedFinalize\(broker\)/,
    "accepted phi must follow the paper's characteristic transport and redistance path");
  assert.match(advance,
    /adaptiveMass\.encodeDerivedOutputs\([^]*"preserve-and-validate"\)/,
    "conservative mass outputs must not reconstruct sub-cell phi from one coarse-leaf scalar");
  assert.doesNotMatch(advance, /encodeAcceptedExternalPhiPublication/,
    "mass-derived pseudo-phi must not replace transported accepted phi");
  assert.match(advance, /dt_s===0[^]*encodeAcceptedDerivations\(broker\)/,
    "the paused construction frame must retain its already-redistanced bootstrap tuple");
  assert.doesNotMatch(advance, /adaptiveVelocity\.encodeAcceptedField\(broker\)/,
    "positive advances must not leave the accepted predictor bank temporarily stale");
  assert.doesNotMatch(advance, /encodePredictorField/,
    "the positive path must defer its predictor rebuild to S1a");
  assert.match(advection,
    /adaptiveVelocity\.encodePredictorField\(broker\)[^]*adaptivePhi[^]*encodeAcceptedFieldClockSync\(broker\)/,
    "S1a predictor completion must publish the coherent two-bank scalar clock");
  assert.doesNotMatch(advance, /surfaceGraph\.encodeReadyCommit|encodeTopologyRemap/,
    "a same-topology advance must not copy the graph or remap topology worksets");
});

test("candidate closure compiles topology-only velocity stencils once", () => {
  const candidate = compact(WebGPUOctreeLosassoCoarseBackend.prototype.encodeCandidatePublication);
  assert.match(candidate,
    /encodeCandidateStencils\(broker\)[^]*encodeCandidateFieldRound\(broker\)[^]*encodeCandidateVelocityNodalCompletion[^]*encodeCandidateFieldRound\(broker\)[^]*encodeCandidateVelocityNodalCompletion[^]*for\(let round=0;round<2;round\+=1\)[^]*encodeCandidateFieldRound\(broker\)/,
    "four value-closure rounds must share the immutable topology/geometry lookup");
  assert.doesNotMatch(candidate, /encodeCandidateFields\(broker\)/,
    "the combined standalone encoder would recompile stencils inside every closure round");
});

test("adaptive phi accepts geometric-only AMR incidences but requires a real corner membership", () => {
  for (const shader of [octreeLosassoAdaptivePhiWorklistReachWGSL,
    octreeLosassoAdaptivePhiRedistanceInitializeWGSL]) {
    assert.match(shader, /if\(localCorner==INVALID\)\{continue;\}/,
      "a nonconforming coarse leaf may geometrically contain a fine or hanging node without using it as a corner");
    assert.match(shader, /if\(memberships==0u\)\{atomicOr\(&control\[12\],ERR_GRAPH\);\}/,
      "a node absent from every incident leaf corner remains a hard graph error");
  }
});

test("Losasso velocity extrapolation keeps adjacency in graph-slot space", () => {
  assert.match(octreeLosassoAdaptiveVelocityWGSL,
    /if\(neighbor<nodes\)\{direct=neighbor;\}/,
    "compiled adjacency must remain a graph slot for phi lookup and constrained-node resolution");
  assert.match(octreeLosassoAdaptiveVelocityWGSL,
    /avSeedAppendGraph\(avAdjacency\(packet,d\)\)/,
    "frontier seeding must map each graph-slot neighbour exactly once");
  assert.doesNotMatch(octreeLosassoAdaptiveVelocityWGSL,
    /direct=avTopologyLoad\(map\+neighbor\)/,
    "storing packet ids would make propagation map neighbours twice and violate Losasso section 6 extrapolation");
});
