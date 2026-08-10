import assert from "node:assert/strict";
import test from "node:test";

import { WebGPUOctreeLosassoCoarseBackend } from "../lib/webgpu-octree-losasso-backend";
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

test("same-topology surface advance pairs velocity and scalar clocks without graph work", () => {
  const advance = compact(WebGPUOctreeLosassoCoarseBackend.prototype.encodeAdaptiveSurfaceAdvance);
  assert.match(advance,
    /adaptiveVelocity\.encodeAcceptedFields\(broker\);this\.adaptivePhi\.encodeAcceptedFieldClockSync\(broker\)/,
    "phi control must adopt the completed nodal-velocity generation immediately");
  assert.doesNotMatch(advance, /surfaceGraph\.encodeReadyCommit|encodeTopologyRemap/,
    "a same-topology advance must not copy the graph or remap topology worksets");
});
