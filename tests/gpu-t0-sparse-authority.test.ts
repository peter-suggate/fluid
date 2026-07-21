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
  const faceBand = new Uint32Array(16);
  faceBand.set([0, 0xffff_ffff, 12, 21, 42, 2, VALID, 4, 7, 21, 0, 0, 0]);
  const transition = new Uint32Array(16);
  transition.set([0, 0xffff_ffff, 12, 3, 9, VALID, VALID, 0]);
  const point = new Uint32Array([0, 0xffff_ffff, 12, 2, 12, VALID, 4, 7]);
  const transient = new Uint32Array(16);
  transient.set([0, 0xffff_ffff, 12, 360, 90, 90, 12, 2, VALID]);
  const publication = new Uint32Array(16);
  publication.set([0, 0xffff_ffff, 21, 7, 7, 7, 2, 3, VALID]);
  return {
    seedCount: 8, seedError: 0, topologyFlags: 0,
    interfaceBricks: 4, desiredBricks: 20, activatedBricks: 20, activeBricks: 20,
    published: true, rolledBack: false, downstreamFinalizeReason: 0,
    generation: 2, configuredFineGeneration: 2, scheduledFineGeneration: 2,
    coarseDirectoryState: OCTREE_POWER_COARSE_LEVELSET_VALID, coarseDirectoryGeneration: 2,
    coarseControlFlags: 0, coarseControlGeneration: 2,
    coarseControlValid: OCTREE_POWER_COARSE_LEVELSET_VALID,
    fineRestrictionFlags: 0, fineRestrictionUnowned: 0, fineRestrictionRows: 12,
    fineRestrictionValid: OCTREE_POWER_COARSE_LEVELSET_VALID,
    transportControl: Array(8).fill(0), redistanceControl: [0, 0, 8, 1],
    volumeControl: Array(16).fill(0), faceBandControl: Array.from(faceBand),
    faceBandTransitionControl: Array.from(transition),
    faceBandPointFieldControl: Array.from(point),
    faceBandTransientPowerControl: Array.from(transient),
    faceBandPowerPublicationControl: Array.from(publication),
  };
}

test("t=0 fine authority requires the complete paper Section 5 publication", () => {
  assert.equal(initialGlobalFineAuthorityReadiness(fineAuthority()).ready, true);
  assert.equal(initialGlobalFineAuthorityReadiness({ ...fineAuthority(), interfaceBricks: 0 }).ready, false,
    "a recurring generation cannot replace transported-interface discovery with old external seeds");
  assert.equal(initialGlobalFineAuthorityReadiness({ ...fineAuthority(), interfaceBricks: 0 },
    { externallySeededColdBootstrap: true }).ready, true,
    "the empty predecessor discovers no fine interface; external affine seeds author the first fresh SPGrid");
  assert.equal(initialGlobalFineAuthorityReadiness({ ...fineAuthority(), interfaceBricks: 0,
    activatedBricks: 0 }, { externallySeededColdBootstrap: true }).ready, false,
    "external seeds must still produce a nonempty activated and published transaction");
  assert.match(initialGlobalFineAuthorityReadiness({ ...fineAuthority(), published: false }).label,
    /global-fine topology rejected/);
  assert.match(initialGlobalFineAuthorityReadiness({ ...fineAuthority(), coarseDirectoryGeneration: 1 }).label,
    /coarse level set/);
  const failedBand = fineAuthority();
  const failedPublication = [...failedBand.faceBandPowerPublicationControl];
  failedPublication[5] = 6;
  assert.match(initialGlobalFineAuthorityReadiness({ ...failedBand,
    faceBandPowerPublicationControl: failedPublication }).label, /Section 5/);
});

test("t=0 rejection preserves named downstream evidence after device disposal", () => {
  const failed = fineAuthority();
  const topologyControl = [16, 4, 20, 20, 0, 1, 0, 2 | 4];
  const redistanceControlDetailed = [1, 900_000, 8, 0, 16, 123, 2, 40, 1, 20, 22, 9];
  const outcome = initialGlobalFineAuthorityReadiness({
    ...failed,
    topologyFlags: 16,
    topologyControl,
    downstreamFinalizeReason: 2 | 4,
    redistanceControlDetailed,
  });
  assert.equal(outcome.ready, false);
  assert.match(outcome.label, /"errors":\["downstreamPublication"\]/);
  assert.match(outcome.label, /"downstream":\["redistance","volume"\]/);
  assert.match(outcome.label, /"conflictingRequest"/);
  assert.match(outcome.label, /"firstError":123/);
});

test("t=0 power pressure requires nonempty CSR and Section 4.3 convergence", () => {
  const control = new Uint32Array(16);
  control.set([0, 1, 6, 12]);
  const floats = new Float32Array(control.buffer);
  floats[4] = 1e-10; floats[5] = 1;
  const accepted = { authoritative: true, solverLabel: "Octree power PCG · Section 4.3 hybrid",
    pressureRows: 12, pressureEntries: 48, capacityOverflow: false, mgpcgControl: control };
  assert.equal(initialPowerPressureReadiness(accepted).ready, true);
  assert.match(initialPowerPressureReadiness({ ...accepted, pressureRows: 0 }).label, /CSR/);
  control[1] = 0;
  assert.match(initialPowerPressureReadiness(accepted).label, /did not converge/);
});
