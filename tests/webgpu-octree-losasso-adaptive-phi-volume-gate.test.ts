import assert from "node:assert/strict";
import test from "node:test";
import { octreeLosassoAdaptivePhiVolumeEvidenceWGSL,
  octreeLosassoAdaptivePhiWGSL }
  from "../lib/webgpu-octree-losasso-adaptive-phi.wgsl";

test("volume evidence is the sole non-mutating pre/post-redistance path", () => {
  assert.match(octreeLosassoAdaptivePhiWGSL,
    /capturePreRedistanceVolumes[\s\S]*preRedistanceVolumes\[leaf\]=measured\.value/);
  assert.match(octreeLosassoAdaptivePhiWGSL,
    /derivePostRedistanceVolumes[\s\S]*physicalVolumes\[leaf\]=measured\.value[\s\S]*receipts\[select\(40u,48u,candidateDiagnostics\(\)\)\]/);
  assert.doesNotMatch(octreeLosassoAdaptivePhiWGSL,
    /localVolume|LocalVolume|Debt|debt|correctionOffset|applyLocalVolumeOffsets|solveLocalVolumeOffsets/);
});

test("physical drift is telemetry and integrity remains fail closed", () => {
  assert.match(octreeLosassoAdaptivePhiVolumeEvidenceWGSL,
    /let drift=pre-post/);
  assert.match(octreeLosassoAdaptivePhiVolumeEvidenceWGSL,
    /signedDrift\+=drift[\s\S]*totalAbsoluteDrift\+=abs\(drift\)[\s\S]*maximumLeafDrift=max/);
  assert.match(octreeLosassoAdaptivePhiVolumeEvidenceWGSL,
    /receipts\[base\+7u\],select\(0u,1u,valid\)/);
  assert.doesNotMatch(octreeLosassoAdaptivePhiVolumeEvidenceWGSL,
    /abs\(drift\)>|localDebtTolerance|phi\[[^\]]+\]\s*=/,
    "measured drift must neither reject publication nor mutate phi");
  assert.match(octreeLosassoAdaptivePhiVolumeEvidenceWGSL,
    /receipts\[base\+4u\]\)\!=leafCount\(\)[\s\S]*ERR_VOLUME/);
  assert.match(octreeLosassoAdaptivePhiVolumeEvidenceWGSL,
    /expected\.valid==0u\|\|abs\(value-expected\.value\)>p\.volume\.y/);
  assert.match(octreeLosassoAdaptivePhiVolumeEvidenceWGSL,
    /!finite\(post\)\|\|post<0\.\|\|!finite\(pre\)\|\|pre<0\./);
});

test("accepted and candidate volume transactions use disjoint receipt banks", () => {
  assert.match(octreeLosassoAdaptivePhiVolumeEvidenceWGSL,
    /fn volumeReceiptBase\(\)->u32\{return select\(36u,44u,candidate\(\)\);\}/);
  assert.match(octreeLosassoAdaptivePhiVolumeEvidenceWGSL,
    /for\(var i=0u;i<8u;i\+=1u\)\{atomicStore\(&receipts\[base\+i\],0u\);\}/);
  assert.match(octreeLosassoAdaptivePhiWGSL,
    /receipts\[select\(40u,48u,candidateDiagnostics\(\)\)\]/);
  assert.match(octreeLosassoAdaptivePhiWGSL,
    /finalizeCandidateRepair[\s\S]*receipts\[51\]/);
});

test("deleted patch-offset repair has no dormant shader or option", () => {
  const source = octreeLosassoAdaptivePhiWGSL + octreeLosassoAdaptivePhiVolumeEvidenceWGSL;
  for (const forbidden of ["measureOnly", "LocalPatchState", "migrateLocalVolumeDebt",
    "buildLocalVolumeSupport", "assignLocalVolumeReceivers", "localVolumeDebt"] as const) {
    assert.doesNotMatch(source, new RegExp(forbidden));
  }
});
