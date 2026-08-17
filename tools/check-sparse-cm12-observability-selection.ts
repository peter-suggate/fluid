#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { strict as assert } from "node:assert";
import "../lib/methods";
import { resolveMethodValues } from "../lib/core/method-contract";
import { parseQueryState, serializeQueryState } from "../lib/core/url-state";
import {
  adaptiveMassMethod,
  adaptiveMassSolverOptions,
} from "../lib/methods/adaptive-mass/method";

const residentPath = "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts";
const shaderPath = "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts";
const methodPath = "lib/methods/adaptive-mass/method.ts";
const overlayPath = "lib/core/webgpu-grid-overlay.ts";
const resident = readFileSync(residentPath, "utf8");
const shader = readFileSync(shaderPath, "utf8");
const method = readFileSync(methodPath, "utf8");
const overlay = readFileSync(overlayPath, "utf8");
const encodeStart = resident.indexOf("\n  encode(\n    encoder:");
const encodeEnd = resident.indexOf("\n  /**\n   * Turn the marker view", encodeStart);
assert(encodeStart >= 0 && encodeEnd > encodeStart, "resident encode body");
const encodeBody = resident.slice(encodeStart, encodeEnd);
const dispatcherNames = ["dispatch", "dispatchAccepted", "dispatchScalarResult",
  "dispatchPressureCell", "dispatchPressureRow", "dispatchPressureBootstrap",
  "dispatchPressureTopologyRepair", "dispatchPersistentPressureCache",
  "dispatchPressureSolveAuthority", "dispatchPressureBrickSolve",
  "dispatchPressureHierarchySolve", "dispatchFaceProjectionAuthority",
  "dispatchTemporalCell", "dispatchTemporalRow", "dispatchActivity",
  "dispatchVexActivity", "dispatchFrameControl"] as const;
const wholeFrameLexicalCallsites = Object.fromEntries(dispatcherNames.map((name) => [name,
  Math.max(0, (encodeBody.match(new RegExp(`\\b${name}\\(`, "g")) ?? []).length - 1)]));

for (const token of ["sealDirtySchedulerPreEpoch", "finalizeDirtySchedulerEpoch"]) {
  assert.match(resident, new RegExp(`dispatch\\(\"${token}\"`), `${token} is no longer live`);
}
assert.doesNotMatch(resident, /publishGlobalDirtySchedulerPackets|finalizeDirtySchedulerPackets/,
  "global CMD1/PKT1 compatibility projection returned");
assert.match(resident, /sealDirtySchedulerPreEpoch[\s\S]{0,800}copyBufferToBuffer\(this\.activity[\s\S]{0,800}finalizeQueuedDirtySchedulerPackets/,
  "compact dirty scheduler finalization seam moved");
assert.match(resident, /dispatchWorkgroupsIndirect\(this\.presentationIndirectArguments, 0\)/,
  "queued dirty scheduler finalization is no longer GPU-counted");
assert.doesNotMatch(resident, /set(?:Dirty|Observability).*Enabled/,
  "resident unexpectedly gained an observability selection setter");
assert.match(overlay, /let dirtyDisplay=sparseFramePlanColor\(cell,fieldMode\)/,
  "FPL1 is not the sole dirty visualization source");
assert.doesNotMatch(overlay, /sparseDirtyProvenance|sparseDirtyRecordAddress|SPARSE_DIRTY_/,
  "legacy CMD1/PKT1 renderer fallback returned");

// FPL/FPP remains a production presentation authority and is globally encoded.
const framePlanBody = resident.slice(resident.indexOf("private encodeFramePlanPresentation("),
  resident.indexOf("private encodeVelocityExtensionPlan(",
    resident.indexOf("private encodeFramePlanPresentation(")));
assert.equal((framePlanBody.match(/dispatch\("/g) ?? []).length, 10, "FPL plan dispatch count");
assert.equal((framePlanBody.match(/plan\.dispatchWorkgroupsIndirect/g) ?? []).length, 1,
  "FPL VEX blast import dispatch count");
assert.equal((framePlanBody.match(/execute\.dispatchWorkgroups/g) ?? []).length, 5,
  "FPP execute dispatch count");
assert.equal((framePlanBody.match(/encoder\.copyBufferToBuffer/g) ?? []).length, 2,
  "FPL/FPP copy count");
assert.equal((framePlanBody.match(/const (?:plan|execute) = encoder\.beginComputePass/g) ?? []).length,
  2, "production FPL/FPP pass count");

// Current SIR address arithmetic is correct only at B16 (4^3 = 64 tiles/brick).
assert.match(shader, /return key\*64u\+local\.x\+4u\*\(local\.y\+4u\*local\.z\)/,
  "SIR hard-coded 64-tile address changed; refresh audit");
assert.match(shader, /let logical=tile\/64u;let localTile=tile%64u/,
  "SIR hard-coded inverse address changed; refresh audit");
const sirIndexing = [4, 8, 16].map((brickFineResolution) => {
  const tilesPerAxis = brickFineResolution / 4; const tilesPerBrick = tilesPerAxis ** 3;
  return { brickFineResolution, tilesPerAxis, tilesPerBrick,
    currentHardcodedTilesPerBrick: 64, exact: tilesPerBrick === 64 };
});
assert.deepEqual(sirIndexing.map(({ exact }) => exact), [false, false, true]);

// FPP requires one B-sized page. The independent page selector is therefore
// absent and normalization pins every legacy/injected value to B.
assert.doesNotMatch(method, /key: "presentationPageResolution"/,
  "unsupported independent presentation page selector returned");
assert.match(method, /const presentationPageResolution = \([\s\S]{0,160}\): SparseBrickFineResolution => maximum;/,
  "P==B normalization changed");
assert.match(resident, /if \(presentationPageResolution !== atlas\.brickFineResolution\) \{\s*throw new Error\("FPL1\/FPP1 resident cutover requires one B-sized presentation page"\)/,
  "resident P==B constraint changed");
const uiMatrix = [4, 8, 16].map((brick) => ({ brick, page: brick,
  uiNormalizedValid: brick === 16, residentValid: brick === 16 }));
const brickSpec = adaptiveMassMethod.params.find(
  (spec) => spec.key === "brickFineResolution");
assert(brickSpec?.kind === "select");
assert.deepEqual(brickSpec.options.map(({ value }) => value), ["16"]);
assert.equal(adaptiveMassMethod.params.some(
  (spec) => spec.key === "presentationPageResolution"), false);
for (const fixture of [
  { brick: "4", page: "16" },
  { brick: "8", page: "4" },
  { brick: "16", page: "4" },
  { brick: "16", page: "8" },
  { brick: "16", page: "16" },
] as const) {
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
    brickFineResolution: fixture.brick,
    presentationPageResolution: fixture.page,
  });
  assert.equal(values.brickFineResolution, "16");
  assert.equal(values.presentationPageResolution, "16");
  const options = adaptiveMassSolverOptions(values);
  assert.equal(options.brickFineResolution, 16);
  assert.equal(options.presentationPageResolution, 16);
}
for (const brick of ["4", "8"] as const) {
  const state = parseQueryState(`?method=adaptive-mass&param.adaptive-mass.brickFineResolution=${brick}`
    + "&param.adaptive-mass.presentationPageResolution=4");
  assert.equal(state.overrides["adaptive-mass"]?.brickFineResolution, undefined);
  assert.equal(state.overrides["adaptive-mass"]?.presentationPageResolution, undefined);
  const values = resolveMethodValues(adaptiveMassMethod, "balanced",
    state.overrides["adaptive-mass"] ?? {});
  assert.equal(values.brickFineResolution, "16");
  assert.equal(values.presentationPageResolution, "16");
  const canonical = serializeQueryState("?param.adaptive-mass.presentationPageResolution=4",
    { presetId: state.presetId, scene: state.scene },
    { methodId: state.methodId, quality: state.quality, overrides: state.overrides }, state.ui);
  assert.equal(new URLSearchParams(canonical)
    .has("param.adaptive-mass.presentationPageResolution"), false);
  assert.equal(new URLSearchParams(canonical)
    .has("param.adaptive-mass.brickFineResolution"), false);
}
const defaultValues = resolveMethodValues(adaptiveMassMethod, "balanced", {});
assert.equal(defaultValues.brickFineResolution, "16");
assert.equal(defaultValues.presentationPageResolution, "16");

console.log(JSON.stringify({ authority: "FPL1+compact-dirty-scheduler",
  currentFixedTail: {
    fplFpp: { dispatches: 16, passes: 2, copies: 2 },
    dirtySchedulerLifecycle: { dispatches: 3, passes: 2, copies: 1,
      packetDomain: "GPU-authored exact pre-event queue" },
    mandatoryFrameControlCommit: { dispatches: 1, passesSharedWithPrior: true },
    finalAcceptedIndirectPublication: { dispatches: 0, passes: 0, copies: 1 },
    total: { dispatches: 20, passes: 4, copies: 4 },
  },
  wholeFrameLexicalAudit: { dispatcherCallsites: wholeFrameLexicalCallsites,
    copyCommandSites: (encodeBody.match(/encoder\.copyBufferToBuffer\(/g) ?? []).length,
    explicitClosePassSites: (encodeBody.match(/\bclosePass\(\)/g) ?? []).length - 1,
    logicalStages: (encodeBody.match(/\bstage\("/g) ?? []).length,
    note: "lexical sites include mutually exclusive QA branches; runtime totals also expand pressure, VEX, tree-level, grading, and journal loops" },
  globalCMD1PKT1ProjectionDeleted: true,
  framePlanReadsLiveGenerationStampedPackets: true,
  sirIndexing, uiMatrix,
  b16P4P8Conflict: "fenced: production UI/URL normalizes to B16/P16; direct B4/B8 QA remains construction-only",
  selectedFPLCost: { observabilityDispatches: 0,
    failClosedWithoutReceipt: true } }, null, 2));
