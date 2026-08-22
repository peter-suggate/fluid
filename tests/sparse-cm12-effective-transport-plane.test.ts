import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const wgsl = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url,
), "utf8");
const resident = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts", import.meta.url,
), "utf8");
const solver = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver.ts", import.meta.url,
), "utf8");
const effectivePlane = readFileSync(new URL(
  "../lib/methods/adaptive-mass/sparse-cm12-effective-transport-velocity.wgsl.ts",
  import.meta.url,
), "utf8");
const image = readFileSync(new URL(
  "../lib/methods/adaptive-mass/sparse-cm12-transport-execution-image.ts", import.meta.url,
), "utf8");
const imageWGSL = readFileSync(new URL(
  "../lib/methods/adaptive-mass/sparse-cm12-transport-execution-image.wgsl.ts",
  import.meta.url,
), "utf8");
const packetAuthorityWGSL = readFileSync(new URL(
  "../lib/methods/adaptive-mass/sparse-cm12-transport-packet-authority.wgsl.ts",
  import.meta.url,
), "utf8");
const conservativeStage = resident.slice(resident.indexOf('stage("conservative-transport"'),
  resident.indexOf('stage("gamma-diffusion"'));
const functionSource = (source: string, name: string, next: string): string => {
  const begin = source.indexOf(`fn ${name}`);
  const end = source.indexOf(next, begin);
  assert.ok(begin >= 0 && end > begin, `${name} source range must remain identifiable`);
  return source.slice(begin, end);
};

test("the effective vec4 plane is a VEX product, never transport materialization", () => {
  assert.doesNotMatch(conservativeStage, /materializeSparseCM12EffectiveTransportVelocity/);
  assert.match(wgsl,
    /@group\(0\)@binding\(3\)var<storage,read_write>partials:array<vec4f>/);
  assert.match(resident,
    /binding:\s*3,\s*resource:\s*\{\s*buffer:\s*effectiveTransportVelocity/);
  const sample = wgsl.slice(wgsl.indexOf("fn sampleEffectiveTransportVelocity"),
    wgsl.indexOf("fn traceEffectiveTransportDeparture"));
  assert.match(sample, /cm12EffectiveTransportVelocity\(cell\)/);
  assert.match(effectivePlane, /return\s+\$\{plane\}\[cell\]/);
  assert.doesNotMatch(sample, /state\[/);
});

test("the AEI arm enables committed packets rather than the checkpoint tile resolver", () => {
  assert.doesNotMatch(wgsl, /EXP_ACCEPTED_EXECUTION_IMAGE_PACKETS:bool=false/);
  assert.doesNotMatch(wgsl, /packet publication is\s*\n?\/\/ disabled/i);
  assert.match(wgsl, /fn stageSparseCM12TransportExecutionImage[\s\S]*cm12TransportStagedExecutionCell/);
});

test("transport packets are leaf/rung-major 64-cell workgroups", () => {
  assert.match(image, /64/);
  assert.doesNotMatch(image,
    /packetCapacity\s*=\s*checked\(logicalCount\s*\*\s*tilesPerLogicalBrick/);
  assert.doesNotMatch(image,
    /tilesPerLogicalBrickAxis\s*=\s*options\.brickFineResolution\s*\/\s*4/);
});

test("the converted packet identity path performs no per-lane owner reconstruction", () => {
  const packet = functionSource(packetAuthorityWGSL, "cm12StageTransportPacket",
    "fn cm12TransportStagedExecutionCell");
  assert.match(packet, /workgroupBarrier\(\)/,
    "one uniform barrier publishes the staged packet to all 64 lanes");
  assert.doesNotMatch(packet, /compactOwnerCellAt|ownerCellAt|cellOpenVolume/);
});

test("the topology tail compiles shadow TEI, flips once, then replays only the retired slot", () => {
  const tail = resident.slice(resident.indexOf('stage("resolution-planning"'),
    resident.indexOf('stage("presentation-publication"'));
  const retire = tail.indexOf('dispatch("retireUnsupportedEmptyBricks"');
  const transferredPlane = tail.indexOf(
    'dispatchTopologyDelta("publishCandidateTopologyDeltaFromWorklist"');
  const compile = tail.indexOf(
    'dispatchTopologyDelta("compileSparseCM12TransportExecutionImageShadow"');
  const authorize = tail.indexOf('dispatch("validateAndAuthorizeShadowTopology"');
  const flip = tail.indexOf('dispatch("finalizeAuthorizedShadowTopology"');
  const replay = tail.indexOf(
    'dispatchTopologyDelta("replaySparseCM12TransportExecutionImageRetired"');
  assert.ok(retire >= 0 && compile > retire && authorize > compile
    && transferredPlane > authorize && flip > transferredPlane && replay > flip,
  "lifecycle, candidate image, authorization, transferred plane, selector flip, and retired replay must be ordered");
  assert.doesNotMatch(resident, /refreshSparseCM12TransportExecutionImage/);
  assert.doesNotMatch(imageWGSL, /refreshSparseCM12TransportExecutionImage/);
  const deltaCompiler = imageWGSL.slice(imageWGSL.indexOf("fn cm12TeiCompileTopologyDelta"),
    imageWGSL.indexOf("fn compileSparseCM12TransportExecutionImageShadow"));
  assert.match(deltaCompiler, /topologyDeltaLeafInvocation\(rank\)/);
  assert.match(deltaCompiler, /let slot=shadowTopologySlot\(\)/);
  assert.doesNotMatch(deltaCompiler, /cm12TeiWriteLeaf\(0u|cm12TeiWriteLeaf\(1u/);
});

test("raw Phase-1 receipts are reachable only through a construction specialization", () => {
  assert.match(resident, /static createPhase1TransportReceiptOracleForQA\(/);
  const factory = resident.slice(
    resident.indexOf("static createPhase1TransportReceiptOracleForQA("),
    resident.indexOf("private static async createConfigured("));
  assert.match(factory,
    /false, false, false, false, undefined, undefined, true\)/);
  const productionFactory = resident.slice(resident.indexOf("static create("),
    resident.indexOf("static createPressureRefreshOracleForQA("));
  assert.doesNotMatch(productionFactory, /undefined, undefined, true/);
  assert.match(solver, /static createPhase1TransportReceiptOracleForQA\(/);
  assert.match(solver,
    /PHASE1_TRANSPORT_RECEIPT_QA_TOKEN[\s\S]*createPhase1TransportReceiptOracleForQA/);
});
