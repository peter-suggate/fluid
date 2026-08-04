import assert from "node:assert/strict";
import test from "node:test";
import {
  initialGlobalFineAuthorityReadiness,
  initialPowerPressureReadiness,
  type InitialGlobalFineAuthorityDiagnostics,
} from "../lib/webgpu-uniform-eulerian";
import { OCTREE_POWER_COARSE_LEVELSET_VALID } from "../lib/webgpu-octree-power-coarse-levelset";

const VALID = 0x8000_0000;

function fineAuthority(): InitialGlobalFineAuthorityDiagnostics {
  return {
    seedControl: [8, 0],
    topologyControl: [0, 4, 20, 20, 1, 0, 1, 0, 4],
    fineVolumeControl: new Array(16).fill(0),
    worklistHeader: [2, 20, 32, 3, 1, 1, 1],
    coarseControl: [0, 0xffff_ffff, 12, 12, 0, 0, 8, 0, 4, 0, 2,
      OCTREE_POWER_COARSE_LEVELSET_VALID, 0, 0, 0, 0],
    fineRestrictionControl: [42, 8, 0, 0, 12, OCTREE_POWER_COARSE_LEVELSET_VALID],
    structuredVelocityControl: [0, 0xffff_ffff, 12, 2, 0, 12],
    structuredBoundaryControl: [0, 0xffff_ffff, 12, 12, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    configuredFineGeneration: 2,
    fineGenerationSlot: 0,
    scheduledFineGeneration: 2,
    currentFineIsA: true,
  };
}

test("t=0 fine authority requires one coherent fine/coarse/structured publication", () => {
  assert.equal(initialGlobalFineAuthorityReadiness(fineAuthority()).ready, true);
  const noInterface = { ...fineAuthority(), topologyControl: [...fineAuthority().topologyControl] };
  noInterface.topologyControl[1] = 0;
  assert.equal(initialGlobalFineAuthorityReadiness(noInterface).ready, false,
    "a recurring generation cannot replace transported-interface discovery with old external seeds");
  assert.equal(initialGlobalFineAuthorityReadiness(noInterface,
    { externallySeededColdBootstrap: true }).ready, true,
    "the empty predecessor may use explicit seeds only for the cold bootstrap");
  const noActivation = { ...fineAuthority(), topologyControl: [...fineAuthority().topologyControl] };
  noActivation.topologyControl[1] = 0; noActivation.topologyControl[3] = 0;
  assert.equal(initialGlobalFineAuthorityReadiness(noActivation,
    { externallySeededColdBootstrap: true }).ready, false);
  const unpublished = { ...fineAuthority(), topologyControl: [...fineAuthority().topologyControl] };
  unpublished.topologyControl[4] = 0;
  assert.match(initialGlobalFineAuthorityReadiness(unpublished).label, /global-fine topology rejected/);
  const staleCoarse = { ...fineAuthority(), coarseControl: [...fineAuthority().coarseControl] };
  staleCoarse.coarseControl[10] = 1;
  assert.match(initialGlobalFineAuthorityReadiness(staleCoarse).label, /coarse level set/);
  const staleVelocity = { ...fineAuthority(), structuredVelocityControl: [...fineAuthority().structuredVelocityControl] };
  staleVelocity.structuredVelocityControl[3] = 1;
  assert.match(initialGlobalFineAuthorityReadiness(staleVelocity).label,
    /structured velocity\/boundary authority/);
  const staleBoundary = { ...fineAuthority(), structuredBoundaryControl: [...fineAuthority().structuredBoundaryControl] };
  staleBoundary.structuredBoundaryControl[6] = 1;
  assert.match(initialGlobalFineAuthorityReadiness(staleBoundary).label,
    /structured velocity\/boundary authority/);
});

test("t=0 rejection preserves named downstream evidence after device disposal", () => {
  const failed = { ...fineAuthority(), topologyControl: [...fineAuthority().topologyControl] };
  failed.topologyControl[0] = 16;
  failed.topologyControl[4] = 0;
  failed.topologyControl[5] = 1;
  failed.topologyControl[7] = 2 | 4;
  const outcome = initialGlobalFineAuthorityReadiness(failed);
  assert.equal(outcome.ready, false);
  assert.match(outcome.label, /"errors":\["downstreamPublication"\]/);
  assert.match(outcome.label, /"downstream":\["redistance","volume"\]/);
  assert.match(outcome.label, /"structuredVelocity"/);
  assert.match(outcome.label, /"structuredBoundary"/);
});

test("t=0 power pressure requires nonempty resolved rows and Section 4.3 convergence", () => {
  const control = new Uint32Array(16);
  control.set([0, 1, 6, 7, 12]);
  const floats = new Float32Array(control.buffer);
  floats[10] = 1e-10; floats[11] = 1e-18;
  floats[8] = 1; floats[9] = 1e-9;
  const accepted = { authoritative: true, solverLabel: "Octree power MGPCG · persistent executor",
    pressureRows: 12, capacityOverflow: false, mgpcgControl: control };
  assert.equal(initialPowerPressureReadiness(accepted).ready, true);
  assert.equal(initialPowerPressureReadiness({ ...accepted,
    solverLabel: "Octree power MGPCG · row-parallel exact-reduction executor",
  }).ready, true);
  assert.match(initialPowerPressureReadiness({ ...accepted, pressureRows: 0 }).label, /resolved power rows/);
  control[1] = 0;
  assert.match(initialPowerPressureReadiness(accepted).label, /did not converge/);
});
