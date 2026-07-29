import assert from "node:assert/strict";
import test from "node:test";
import {
  STRUCTURED_GENERATION_AUDIT_SNAPSHOT,
  exactStructuredGenerationAuditFailures,
  finalPerformanceAuthorityFailures,
  unpackStructuredBoundaryControl,
  unpackStructuredGenerationAuditSnapshot,
  unpackStructuredVelocityControl,
  type ExactStructuredGenerationAudit,
} from "../tools/webgpu-smoke-structured-audit";

function coherentAudit(): ExactStructuredGenerationAudit {
  return {
    publishedFineGeneration: 8,
    expectedStructuredEpoch: 7,
    previousFineGeneration: 7,
    previousStructuredEpoch: 6,
    structured: { flags: 0, firstError: 0xffff_ffff, rowCount: 21,
      epoch: 7, activeBank: 1, slotCount: 73 },
    boundary: { flags: 0, firstError: 0xffff_ffff, rowCount: 21, slotCount: 73,
      epoch: 7, activeBank: 1, publishedEpoch: 7 },
  };
}

test("structured control decoders expose the packed accepted transaction", () => {
  assert.deepEqual(unpackStructuredVelocityControl([0, 0xffff_ffff, 21, 7, 1, 73]),
    coherentAudit().structured);
  assert.deepEqual(unpackStructuredBoundaryControl([0, 0xffff_ffff, 21, 73, 7, 1, 7]),
    coherentAudit().boundary);
  assert.throws(() => unpackStructuredVelocityControl([0, 1]), /six words/);
  assert.throws(() => unpackStructuredBoundaryControl([0, 1]), /seven words/);
});

test("packed per-step audit snapshots retain every authority control", () => {
  const layout = STRUCTURED_GENERATION_AUDIT_SNAPSHOT;
  // 24 structured + 28 boundary + 28 fine worklist + 64 mgpcg + 64 fine volume
  // + 128 projection energy (32-word per-stage layout), then the four P0.4
  // verdict segments: 64 topology epoch + 384 SPGrid level delta + 8
  // air-support failure + 16 fine-transport governor.
  assert.equal(layout.strideBytes, 808);
  const bytes = new Uint8Array(2 * layout.strideBytes);
  const write = (record: number, offsetBytes: number, values: readonly number[]) => {
    new Uint32Array(bytes.buffer, record * layout.strideBytes + offsetBytes, values.length).set(values);
  };
  write(1, layout.structuredOffsetBytes, [0, 0xffff_ffff, 21, 8, 0, 73]);
  write(1, layout.boundaryOffsetBytes, [0, 0xffff_ffff, 21, 73, 8, 0, 8]);
  write(1, layout.fineOffsetBytes, [9, 65, 128, 3, 2, 1, 1]);
  const mgpcg = new Uint32Array(16); mgpcg[2] = 7;
  write(1, layout.mgpcgOffsetBytes, Array.from(mgpcg));
  const volume = new Uint32Array(16); volume[13] = 9;
  write(1, layout.fineVolumeOffsetBytes, Array.from(volume));
  const projectionEnergy = new Uint32Array(32);
  projectionEnergy.set([0, 8, 0, 21, 21, 1, 1, 8]);
  write(1, layout.projectionEnergyOffsetBytes, Array.from(projectionEnergy));
  const decoded = unpackStructuredGenerationAuditSnapshot(bytes, 1);
  assert.deepEqual(decoded.structured,
    { flags: 0, firstError: 0xffff_ffff, rowCount: 21, epoch: 8, activeBank: 0, slotCount: 73 });
  assert.deepEqual(decoded.boundary,
    { flags: 0, firstError: 0xffff_ffff, rowCount: 21, slotCount: 73,
      epoch: 8, activeBank: 0, publishedEpoch: 8 });
  assert.deepEqual(Array.from(decoded.fineHeader), [9, 65, 128, 3, 2, 1, 1]);
  assert.equal(decoded.mgpcgControl[2], 7);
  assert.equal(decoded.fineVolumeControl[13], 9);
  assert.deepEqual(Array.from(decoded.projectionEnergyControl), Array.from(projectionEnergy));
  assert.throws(() => unpackStructuredGenerationAuditSnapshot(bytes, 2), /truncated/);
});

test("paper-separated fine SPGrid may advance while a coherent pressure octree is retained", () => {
  // Aanjaneya et al. 2017, Section 5
  // (`docs/papers/aanjaneya-2017-power-liquids.txt`) defines the fine SPGrid
  // and background octree as separate meshes. Fine topology is updated after
  // every advection; no paper rule couples its numeric generation to a
  // pressure-topology epoch that did not need adaptation.
  assert.deepEqual(exactStructuredGenerationAuditFailures(coherentAudit()), []);
  const retained = coherentAudit();
  assert.deepEqual(exactStructuredGenerationAuditFailures({
    ...retained,
    publishedFineGeneration: 9,
    expectedStructuredEpoch: 7,
    previousFineGeneration: 8,
    previousStructuredEpoch: 7,
  }), []);
  const stale = coherentAudit();
  const failures = exactStructuredGenerationAuditFailures({ ...stale,
    structured: { ...stale.structured, epoch: 6 },
    boundary: { ...stale.boundary, activeBank: 0 },
  });
  assert.ok(failures.includes("structured velocity publication is invalid"));
  assert.ok(failures.includes("structured boundary publication is invalid or incoherent"));
  assert.ok(exactStructuredGenerationAuditFailures({
    ...retained,
    expectedStructuredEpoch: 6,
    previousStructuredEpoch: 7,
    structured: { ...retained.structured, epoch: 6 },
    boundary: { ...retained.boundary, epoch: 6, publishedEpoch: 6 },
  }).includes("structured epoch regressed"));
});

test("final performance authority validates seven-word fine and structured publications", () => {
  const coherent = coherentAudit();
  assert.deepEqual(finalPerformanceAuthorityFailures({
    expectedSteps: 6, observedSteps: 6, expectedTime_s: 2, targetTime_s: 2,
    submittedTime_s: 2, fineSourceGeneration: 8,
    fineWorklistHeader: [8, 65, 128, 3, 2, 1, 1], finePageCapacity: 128,
    structured: coherent.structured, boundary: coherent.boundary,
  }), []);
  const failures = finalPerformanceAuthorityFailures({
    expectedSteps: 6, observedSteps: 5, expectedTime_s: 2, targetTime_s: 2,
    submittedTime_s: 1.9, fineSourceGeneration: 7,
    fineWorklistHeader: [6, 129, 128, 0, 1, 0, 1], finePageCapacity: 128,
    structured: { ...coherent.structured, flags: 1 }, boundary: coherent.boundary,
  });
  assert.ok(failures.length >= 6);
});
