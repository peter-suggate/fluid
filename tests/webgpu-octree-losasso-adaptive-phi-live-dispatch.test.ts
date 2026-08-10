import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const implementation = readFileSync(new URL(
  "../lib/webgpu-octree-losasso-adaptive-phi.ts", import.meta.url), "utf8");
const shader = readFileSync(new URL(
  "../lib/webgpu-octree-losasso-adaptive-phi.wgsl.ts", import.meta.url), "utf8");

const acceptedMethods = implementation.slice(
  implementation.indexOf("encodeAcceptedAdvance("),
  implementation.indexOf("encodeJointCommitGate("),
);

test("recurring accepted adaptive phi work is live-indirect, never capacity-launched", () => {
  assert.doesNotMatch(acceptedMethods,
    /this\.plan\.(?:node|leaf|pressureRow|face)Dispatch\[0\]/);
  assert.match(acceptedMethods, /encodeAcceptedLiveSchedule\(broker\)/);
  assert.match(acceptedMethods, /runBufferIndirect\([\s\S]*this\.source\.nodeDispatch/);
  assert.match(acceptedMethods, /runMainBufferIndirect\([\s\S]*this\.source\.rowDispatch/);
  assert.match(acceptedMethods, /runMainBufferIndirect\([\s\S]*this\.source\.leafDispatch/);
  assert.match(acceptedMethods, /runBufferIndirect\([\s\S]*this\.source\.faceDispatch/);
});

test("accepted live scheduler validates one coherent graph/phi/face tuple", () => {
  const schedule = shader.slice(
    shader.indexOf("octreeLosassoAdaptivePhiAcceptedScheduleWGSL"),
    shader.indexOf("octreeLosassoAdaptivePhiScheduleWGSL"),
  );
  assert.match(schedule, /graph\[3\]!=epoch\|\|graph\[4\]!=0u/);
  assert.match(schedule, /atomicLoad\(&state\[1\]\)!=epoch/);
  assert.match(schedule, /graph\[5\]!=atomicLoad\(&state\[2\]\)/);
  assert.match(schedule, /graph\[17\]!=expectedLeaves/);
  assert.match(schedule, /graph\[20\]!=expectedNodes/);
  assert.match(schedule, /faceControl\[0\]!=epoch\|\|faceControl\[1\]!=rows/);
});

test("invalid warm candidates do not poison accepted phi errors", () => {
  const candidateSchedule = shader.slice(shader.indexOf("octreeLosassoAdaptivePhiScheduleWGSL"));
  assert.match(candidateSchedule,
    /if\(!candidateValid&&!warm\)\{atomicOr\(&state\[12\],ERR_GRAPH\);\}/);
  assert.doesNotMatch(candidateSchedule,
    /if\(!candidateValid\)\{atomicOr\(&state\[12\],ERR_GRAPH\);\}/);
});

test("exact volume evidence precedes accepted and candidate publication", () => {
  assert.match(acceptedMethods,
    /capturePreRedistanceVolumes[\s\S]*finishAcceptedRedistanceIndependent[\s\S]*finishAcceptedRedistanceConstrained[\s\S]*projectTransported[\s\S]*prepareVolumeEvidence[\s\S]*derivePostRedistanceVolumes[\s\S]*validateVolumeEvidence[\s\S]*finalizeVolumeEvidence[\s\S]*finalizeAccepted/,
    "redistance must restore hanging-node closure before evidence validation");
  const candidate = implementation.slice(
    implementation.indexOf("encodeCandidateFinalize("),
    implementation.indexOf("encodeAcceptedAdvance("),
  );
  assert.match(candidate,
    /capturePreRedistanceVolumes[\s\S]*finishRedistance[\s\S]*projectTransported[\s\S]*prepareVolumeEvidence[\s\S]*derivePostRedistanceVolumes[\s\S]*validateVolumeEvidence[\s\S]*finalizeVolumeEvidence[\s\S]*finalizeCandidateRepair[\s\S]*stampRepair/);
  assert.doesNotMatch(implementation,
    /localVolume|LocalVolume|correctionSupport|debtMigration|localCorrectionSweeps/);
});

test("transport canonicalization is stable across the measured D4 half-quantum orbit", () => {
  const transport = shader.slice(
    shader.indexOf("octreeLosassoAdaptivePhiTransportWGSL"),
    shader.indexOf("octreeLosassoAdaptivePhiHandoffWGSL"),
  );
  assert.match(transport,
    /fn quantizePhi\(v:f32\)->f32\{return trunc\(v\*65536\.\)\/65536\.;\}/);
  const sourceRaw = 0.18834759294986725;
  const mirrorRaw = 0.18834683299064636;
  assert.equal(Math.trunc(sourceRaw * 65536), Math.trunc(mirrorRaw * 65536));
  assert.notEqual(Math.round(sourceRaw * 65536), Math.round(mirrorRaw * 65536));
});
