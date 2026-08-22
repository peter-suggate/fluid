import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const wgslSource = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
  import.meta.url,
), "utf8");
const hostSource = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
  import.meta.url,
), "utf8");

test("mass dirt imports the exact SRR1 tile worklist", () => {
  const begin = wgslSource.indexOf("fn importSparseCM12MassWorkToFramePlanNext");
  const end = wgslSource.indexOf("fn markSparseCM12GlobalFramePlanReceipts", begin);
  assert.ok(begin >= 0 && end > begin);
  const importer = wgslSource.slice(begin, end);

  assert.match(importer, /let rank=gid\.x;let workTile=sirResidentWorkTile\(rank\)/);
  assert.match(importer, /let fineOrigin=sirMassTileOrigin\(rank\)/);
  assert.match(importer, /let cell=ownerCellAt\(vec3i\(fineOrigin\)\)/);
  assert.match(importer, /let brick=cellBrick\(cell\)/);
  assert.match(importer,
    /sirResidentEventCause\(workTile,sirResidentCandidateGeneration\(\)\)/);
  assert.match(importer,
    /cm12FramePlanMarkOwnedNextTile\(brick,tile,[\s\S]*select\(0u,1u<<1u,direct\)/);
  assert.doesNotMatch(importer, /generationMismatch/);
  assert.doesNotMatch(importer,
    /acceptedTemplate(Cell|Row)Invocation|templateBrick(Cell|Row)Range|for\s*\(/,
    "the FPL import must remain one constant-work mapping per SRR1 rank");
});

test("the broad FPL compatibility blast retains accepted gamma, surface, and pressure", () => {
  const begin = wgslSource.indexOf("fn populateSparseCM12PresentationFramePlan");
  const end = wgslSource.indexOf("fn importSparseCM12MassWorkToFramePlanNext", begin);
  assert.ok(begin >= 0 && end > begin);
  const population = wgslSource.slice(begin, end);
  assert.match(population, /cm12FramePlanMarkOwnedNextTile\(brick,lane,0x1cu,0u/,
    "accepted-cell gamma and surface plus pressure retain the compatibility blast");
  assert.match(population, /SPARSE_CM12_DIRTY_CAUSE_BIT\.coefficientChanged/);
  assert.doesNotMatch(population, /SPARSE_CM12_DIRTY_CAUSE_BIT\.generationMismatch/);
  assert.doesNotMatch(population, /cm12FramePlanMarkOwnedNextTile\(brick,lane,0x14u,0u/);
  assert.doesNotMatch(population, /cm12FramePlanMarkOwnedNextTile\(brick,lane,0x1eu,0u/);
});

test("SRR1 and VEX bootstrap causes are topology work, not magenta faults", () => {
  const scalarBegin = wgslSource.indexOf("fn cm12ScalarResultDirtyCauses");
  const scalarEnd = wgslSource.indexOf("const TRANSPORT_CHARACTERISTIC_CLEARANCE", scalarBegin);
  const scalarCauses = wgslSource.slice(scalarBegin, scalarEnd);
  assert.match(scalarCauses, /SCALAR_RESULT_CAUSE\.scalarWrite/);
  assert.match(scalarCauses, /DIRTY_CAUSE_BIT\.densityChanged/);
  assert.match(scalarCauses, /SCALAR_RESULT_CAUSE\.dependencyClosure/);
  assert.match(scalarCauses, /DIRTY_CAUSE_BIT\.dependencyClosure/);
  assert.doesNotMatch(scalarCauses, /generationMismatch|capacityOrProvenance/);

  const vexBegin = wgslSource.indexOf("fn cm12ResidentVelocityExtensionDirtyCause");
  const vexEnd = wgslSource.indexOf("@compute", vexBegin);
  const vexCauses = wgslSource.slice(vexBegin, vexEnd);
  assert.match(vexCauses,
    /VELOCITY_EXTENSION_CAUSE\.bootstrap[\s\S]*DIRTY_CAUSE_BIT\.topologyCreated/);
  assert.doesNotMatch(vexCauses, /DIRTY_CAUSE_BIT\.generationMismatch/);
});

test("FPL launches mass import from SRR1's GPU-authored indirect count", () => {
  const begin = hostSource.indexOf("private encodeFramePlanPresentation(");
  const end = hostSource.indexOf("private encodeVelocityExtensionPlan(", begin);
  assert.ok(begin >= 0 && end > begin);
  const encode = hostSource.slice(begin, end);
  assert.match(encode, /setPipeline\(this\.pipelines\.importSparseCM12MassWorkToFramePlanNext/);
  assert.match(encode,
    /dispatchWorkgroupsIndirect\(this\.scalarResultAuthority\.indirectBuffer,[\s\S]*SPARSE_CM12_SRR1_INDIRECT_FAMILY\.traceGammaAndBeta/);
  assert.doesNotMatch(encode, /read|mapAsync/);
});

test("gamma and surface planning follow their restored accepted baseline", () => {
  const encodeBegin = hostSource.indexOf("private encodeFramePlanPresentation(");
  const encodeEnd = hostSource.indexOf("private encodeVelocityExtensionPlan(", encodeBegin);
  const encode = hostSource.slice(encodeBegin, encodeEnd);
  assert.doesNotMatch(wgslSource, /fn importSparseCM12SurfaceWorkToFramePlanNext/);
  assert.doesNotMatch(encode, /importSparseCM12SurfaceWorkToFramePlanNext/);
  assert.doesNotMatch(encode, /this\.sharpeningCellAuthorityBaseWords \+ 7/);
});

test("production gamma gathers the complete accepted-cell domain", () => {
  const begin = hostSource.indexOf('stage("gamma-diffusion"');
  const end = hostSource.indexOf('stage("surface-sharpening"', begin);
  assert.ok(begin >= 0 && end > begin);
  const gamma = hostSource.slice(begin, end);
  assert.match(gamma, /dispatchAccepted\("finalizeGammaSnapshot", "cell"\)/);
  assert.match(gamma, /dispatchAccepted\("finalizeGammaRefinement", "cell"\)/);
  assert.doesNotMatch(gamma, /dispatchSharpeningBrickAuthority/);
  assert.doesNotMatch(wgslSource, /fn finalizeGamma(?:Snapshot|Refinement)Bricks/);
});

test("SCA1 source publication has no indirect source-tile halo walk", () => {
  assert.doesNotMatch(wgslSource, /SCA1_HEADER\+11u|fn sca1MarkSourceTileHalo/);
  const begin = hostSource.indexOf('stage("surface-sharpening"');
  const end = hostSource.indexOf('stage("symmetry-authority"', begin);
  const surface = hostSource.slice(begin, end);
  assert.doesNotMatch(surface,
    /dispatchSharpeningCellAuthority\("scatterSharpeningMass"\)[\s\S]*dispatch\("finalizeSharpeningCellAuthority", 1\)/);
});

test("compact stages certify only FPL-scheduled tiles", () => {
  const begin = wgslSource.indexOf("fn markSparseCM12GlobalFramePlanReceipts");
  const end = wgslSource.indexOf("var<workgroup>cm12PresentationBrick", begin);
  assert.ok(begin >= 0 && end > begin);
  const receipts = wgslSource.slice(begin, end);
  assert.match(receipts,
    /if\(cm12FramePlanCurrentTileScheduled\(brick,lane,stage\)\)[\s\S]*cm12FramePlanMarkCurrentTileExecuted\(brick,lane,stage\)/);
  assert.doesNotMatch(receipts,
    /for\(var stage=1u;stage<PRESENTATION_FRAME_PLAN_STAGE;stage\+=1u\)\{\s*cm12FramePlanMarkCurrentTileExecuted/);
});
