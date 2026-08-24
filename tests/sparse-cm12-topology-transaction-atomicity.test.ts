import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const wgsl = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
  import.meta.url,
), "utf8");
const host = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
  import.meta.url,
), "utf8");
const vexWgsl = readFileSync(new URL(
  "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension.wgsl.ts",
  import.meta.url,
), "utf8");
const pressureRepairWgsl = readFileSync(new URL(
  "../lib/methods/adaptive-mass/sparse-cm12-pressure-topology-repair.wgsl.ts",
  import.meta.url,
), "utf8");

const functionSource = (source: string, name: string, next: string): string => {
  const begin = source.indexOf(`fn ${name}`);
  const end = source.indexOf(next, begin);
  assert.ok(begin >= 0 && end > begin, `${name} source range must remain identifiable`);
  return source.slice(begin, end);
};

const count = (source: string, pattern: RegExp): number => [...source.matchAll(pattern)].length;

test("activation and retirement stage lifecycle intent without mutating accepted authority", () => {
  const activation = functionSource(wgsl, "stageDormantReceiverActivation",
    "fn activateSweptReceivers");
  const injectionActivation = functionSource(wgsl, "activateLiquidInjectionReceivers",
    "fn retireUnsupportedEmptyBricks");
  const retirement = functionSource(wgsl, "retireUnsupportedEmptyBricks",
    "const PRESENTATION_FRAME_PLAN_STAGE");

  assert.match(activation, /setCandidateBrickActiveAt\(output,true\)/,
    "activation must author candidate membership only");
  assert.match(activation,
    /select\(acceptedBrickResolution\(brick\),BRICK_FINE_RESOLUTION,\s*brickCandidatePlanningEnabled\(brick\)\)/,
    "a packed swept receiver must enter transport at the fine receiver floor");
  assert.match(activation,
    /select\(requested,applySparseCM12RefinementRegionBounds\(brick,requested\),\s*brickCandidatePlanningEnabled\(brick\)\)/,
    "an unpacked receiver must remain allocatable at its complete construction rung");
  assert.match(injectionActivation, /injectionReachesBrick\(brick\)/,
    "liquid injection must populate source tiles without consuming swept-frame requests");
  assert.doesNotMatch(injectionActivation, /brickRequestedAsReceiver\(brick\)/);
  assert.match(retirement, /setCandidateBrickActiveAt\(output,false\)/,
    "retirement must author candidate membership only");
  for (const [label, source] of [["activation", activation],
    ["retirement", retirement]] as const) {
    assert.doesNotMatch(source, /state\[[^\]]+\]\s*=/,
      `${label} must not change accepted scalar, velocity, pressure, or face fields`);
    assert.doesNotMatch(source,
      /scaInvalidateBrickTopologyClosure|cm12Extension(?:RecordRoot|InvalidateRetiredCell)/,
      `${label} must not publish SCA, PTR, or VEX effects before acceptance`);
    assert.doesNotMatch(source,
      /atomic(?:Add|Sub|Store)\(&activity\[(?:8u|9u|10u|11u|output\+10u|output\+34u)/,
      `${label} must not change accepted membership, lifecycle counters, or residue`);
  }
});

test("liquid injection opens and composes its tile-population journal", () => {
  const begin = host.indexOf("Sparse CM12 resident liquid injection topology");
  const end = host.indexOf("Sparse CM12 resident liquid injection\"", begin + 1);
  assert.ok(begin >= 0 && end > begin, "injection source range must remain identifiable");
  const injection = host.slice(begin, end);
  const openPressureTopology = injection.indexOf(
    'dispatchTopology("beginSparseCM12PressureTopologyRepair", 1)');
  const plan = injection.indexOf('dispatchTopology("planBrickResolution", bricks)');
  assert.ok(openPressureTopology >= 0 && plan > openPressureTopology,
    "a first-frame injection must open its tile-population journal before planning receivers");
  assert.match(pressureRepairWgsl,
    /\(acceptedOld==oldState&&pendingNew==newState\)\|\|pendingNew==oldState/,
    "an injection after a frame topology flip must fold its B-to-C tile changes into A-to-B");
});

test("IBO compiles scheduled membership rather than eagerly-mutated accepted membership", () => {
  const compile = functionSource(wgsl, "cm12IBOCompileScheduledLeaf",
    "fn cm12IBOForEachGeometryCompile");
  const validate = functionSource(wgsl, "cm12ISAValidateScheduledLeafPacket",
    "fn cm12ISABeginAuthority");
  assert.match(compile, /scheduledBrickActive\(leaf\)/);
  assert.match(validate, /cm12IBOScheduledCanonical\(leaf\)/);
  for (const source of [compile, validate]) {
    assert.doesNotMatch(source, /scheduledActive=brickActive\(leaf\)/);
  }
});

test("rejection returns without publishing staged lifecycle authority", () => {
  const validate = functionSource(wgsl, "validateAndAuthorizeShadowTopology",
    "fn finalizeAuthorizedShadowTopology");
  const reject = validate.indexOf("if(!valid)");
  const rejectionReturn = validate.indexOf("return;", reject);
  assert.ok(reject >= 0 && rejectionReturn > reject);
  const rejection = validate.slice(reject, rejectionReturn);
  assert.doesNotMatch(rejection,
    /atomicStore\(&topologyArena\[base(?:\+1u|\+2u)?\]/,
    "rejection must leave the accepted generation and selector byte-identical");
  assert.doesNotMatch(rejection, /state\[[^\]]+\]\s*=|effectiveTransportVelocity\[/,
    "rejection must not touch accepted physical fields or the effective plane");
});

test("validation authorizes only; a distinct singleton flips after every stable publication", () => {
  const validate = functionSource(wgsl, "validateAndAuthorizeShadowTopology",
    "fn finalizeAuthorizedShadowTopology");
  assert.doesNotMatch(validate, /atomicStore\(&topologyArena\[base\+2u\],/,
    "validation may publish an authorization receipt, but not the topology selector");
  assert.doesNotMatch(validate, /state\[[^\]]+\]\s*=|effectiveTransportVelocity\[/,
    "validation must not publish stable physical state");

  const selectorStore = "atomicStore(&topologyArena[base+2u],slot)";
  const selectorAt = wgsl.indexOf(selectorStore);
  assert.ok(selectorAt >= 0, "one topology selector publication must remain present");
  assert.equal(count(wgsl, /atomicStore\(&topologyArena\[base\+2u\],slot\)/g), 1,
    "the combined topology/IBO/lifecycle transaction has one selector publication");
  const functionMatches = [...wgsl.slice(0, selectorAt).matchAll(/\bfn\s+(\w+)\s*\(/g)];
  const finalizeName = functionMatches.at(-1)?.[1];
  assert.equal(finalizeName, "finalizeAuthorizedShadowTopology",
    "the sole selector store must live in a distinct finalizer");
  const finalize = functionSource(wgsl, "finalizeAuthorizedShadowTopology",
    "fn publishSparseCM12TopologyVelocityRootsWork");
  const finalizePrefix = wgsl.slice(Math.max(0, wgsl.lastIndexOf("@compute", selectorAt)),
    selectorAt);
  assert.match(finalizePrefix, /@workgroup_size\(1\)/,
    "selector finalization must be a singleton dispatch");
  assert.match(finalize, /topologyArena\[base\+3u\]\)!=2u/,
    "selector finalization must consume the authorization receipt");
  assert.doesNotMatch(finalize, /atomicStore\(&topologyArena\[base\+3u\],3u\)/,
    "a finalizer entered after stable publication may not reject or request rollback");

  const frameBegin = host.indexOf('stage("candidate-transfer"');
  const frameEnd = host.indexOf('stage("brick-retirement"', frameBegin);
  const frame = host.slice(frameBegin, frameEnd);
  const validation = frame.indexOf('dispatch("validateAndAuthorizeShadowTopology", 1)');
  const beginEffects = frame.indexOf(
    'dispatch("beginSparseCM12AuthorizedCandidateEffects", 1)');
  const ptrEffects = frame.indexOf("this.pipelines.publishSparseCM12TopologyPTREffects!");
  const topologyVex = frame.indexOf(
    'dispatchTopologyDelta("publishSparseCM12CandidateTopologyVexEffects")');
  const injectionVex = frame.indexOf(
    'dispatchTopologyDelta("publishSparseCM12CandidateInjectionVexEffects")');
  const sealEffects = frame.indexOf(
    'dispatch("finalizeSparseCM12AuthorizedCandidateEffects", 1)');
  const finishEffects = frame.indexOf(
    'dispatch("finishSparseCM12TopologyEffectsPublication", 1)');
  const fields = frame.indexOf(
    'dispatchTopologyDelta("publishCandidateTopologyDeltaFromWorklist")');
  const faces = frame.indexOf('dispatchShadow("publishCandidateShadowFaces", "row")');
  const finalizeDispatch = frame.indexOf('dispatch("finalizeAuthorizedShadowTopology", 1)');
  assert.ok(validation >= 0 && beginEffects > validation && ptrEffects > beginEffects
    && topologyVex > ptrEffects && injectionVex > topologyVex
    && sealEffects > injectionVex && finishEffects > sealEffects && fields > finishEffects
    && faces > fields && finalizeDispatch > faces,
  "candidate authorization must precede every stable effect/field publication and the sole flip must follow them");
});

test("post-authorization delta publication cannot fault after its first stable write", () => {
  const publish = functionSource(wgsl, "publishCandidateTopologyDeltaWork",
    "@compute @workgroup_size(64)\nfn publishCandidateTopologyDelta(");
  assert.match(publish, /topologyWorklistBase\(\)\+3u\]\)\s*==\s*2u/,
    "an unconditional delta dispatch must no-op unless validation authorized publication");
  const firstStableWrite = publish.search(/state\[[^\]]+\]\s*=/);
  assert.ok(firstStableWrite >= 0, "delta publication must still author stable fields");
  assert.doesNotMatch(publish.slice(firstStableWrite),
    /scaInvalidateBrickTopologyClosure|cm12Extension(?:RecordRoot|InvalidateRetiredCell)/,
    "candidate publication may not call a fallible journal/cache helper after a stable write");

  const vexPublish = functionSource(wgsl, "publishSparseCM12TopologyVelocityRootsWork",
    "@compute @workgroup_size(64)\nfn publishSparseCM12TopologyVelocityRoots(");
  assert.match(vexPublish, /vda1TransactionAuthorized\(\)/,
    "candidate VEX publication must consume the VDA authorization receipt");
  assert.doesNotMatch(vexPublish,
    /cm12ResidentRecordExtensionIncidence|cm12ExtensionInvalidateRetiredCell/,
    "candidate VEX publication must use an infallible preflighted publication seam");

  const frameBegin = host.indexOf('stage("candidate-transfer"');
  const frameEnd = host.indexOf('stage("brick-retirement"', frameBegin);
  const frame = host.slice(frameBegin, frameEnd);
  const seal = frame.indexOf('dispatch("finalizeSparseCM12AuthorizedCandidateEffects", 1)');
  const fields = frame.indexOf(
    'dispatchTopologyDelta("publishCandidateTopologyDeltaFromWorklist")');
  assert.ok(seal >= 0 && seal < fields,
    "the preflighted no-fail PTR/VDA seal must complete before stable delta publication");
});

test("same-active rerung uses the same staged transaction without lifecycle field resets", () => {
  const membership = functionSource(wgsl, "candidateBrickActive", "fn cellActive");
  assert.match(membership, /ACTIVITY_CANDIDATE_ACTIVE/);
  assert.match(membership,
    /fn scheduledBrickActive\(brick:u32\)->bool\{return candidateBrickActive\(brick\);\}/);
  const schedule = functionSource(wgsl, "scheduleTopologyPreparation",
    "fn acquireTopologyPage");
  assert.match(schedule, /candidateBrickActive\(brick\)!=brickActive\(brick\)/,
    "membership changes and same-active rerungs must share one scheduled delta");
  assert.doesNotMatch(schedule, /state\[[^\]]+\]\s*=/,
    "scheduling must not overwrite accepted fields");
});

test("candidate effective velocity stays unpublished until the transaction accepts", () => {
  const frameBegin = host.indexOf('stage("candidate-transfer"');
  const frameEnd = host.indexOf('stage("brick-retirement"', frameBegin);
  assert.ok(frameBegin >= 0 && frameEnd > frameBegin);
  const frame = host.slice(frameBegin, frameEnd);
  const validate = frame.indexOf('dispatch("validateAndAuthorizeShadowTopology", 1)');
  const cells = frame.indexOf(
    'dispatchTopologyDelta("publishCandidateTopologyDeltaFromWorklist")');
  const faces = frame.indexOf('dispatchShadow("publishCandidateShadowFaces", "row")');
  assert.ok(validate >= 0 && cells > validate && faces > validate,
    "stable scalar/velocity/pressure/face fields and the effective plane must remain accepted until the decision");

  const cellKernel = functionSource(wgsl, "publishCandidateTopologyDeltaWork",
    "@compute @workgroup_size(64)\nfn publishCandidateTopologyDelta(");
  const faceKernel = functionSource(wgsl, "publishCandidateShadowFaces",
    "fn validateAndAuthorizeShadowTopology");
  assert.match(cellKernel, /topologyWorklistBase\(\)\+3u\]\)\s*==\s*2u/,
    "the unconditional post-validator dispatch must no-op after rejection");
  assert.match(faceKernel, /topologyWorklistBase\(\)\+3u\]\)\s*==\s*2u/,
    "stable face publication must no-op after rejection");
});

test("aggregate authorization consumes ISA, TFX, and VDA receipts after every fallible check", () => {
  const validate = functionSource(wgsl, "validateAndAuthorizeShadowTopology",
    "fn finalizeAuthorizedShadowTopology");
  const rejection = validate.indexOf("if(!valid)");
  const mainAuthorization = validate.indexOf(
    "atomicStore(&topologyArena[base+3u],2u)");
  assert.ok(rejection >= 0 && mainAuthorization > rejection,
    "main authorization must remain after the single rejection branch");
  assert.match(validate, /residentTopologyEffectsPreflightReady\(\)/,
    "the aggregate decision must consume the exact preflighted SCA/PTR effect receipt");
  assert.match(validate, /\$\{internedBoundaryCommitReceipt\}/,
    "the aggregate decision must consume the IBO/ISA receipt expression");
  const iboReceiptBegin = wgsl.indexOf("const internedBoundaryCommitReceipt");
  const iboReceiptEnd = wgsl.indexOf("return /* wgsl */", iboReceiptBegin);
  assert.ok(iboReceiptBegin >= 0 && iboReceiptEnd > iboReceiptBegin);
  assert.match(wgsl.slice(iboReceiptBegin, iboReceiptEnd), /cm12ISAAuthorityReady\(\)/,
    "the integrated IBO receipt must include ISA generation, closure, and semantic coverage");
  assert.match(validate, /vda1PreflightReady\(/,
    "the aggregate decision must consume the already sealed VDA receipt");
  const tfxAuthorize = validate.indexOf("tfxAuthorize()", rejection);
  const vdaAuthorize = validate.indexOf("vda1Authorize()", rejection);
  assert.ok(tfxAuthorize > rejection && vdaAuthorize > rejection
    && tfxAuthorize < mainAuthorization && vdaAuthorize < mainAuthorization,
  "candidate authorities may authorize only after the sole rejection branch");

  const frameBegin = host.indexOf('stage("candidate-transfer"');
  const frameEnd = host.indexOf('stage("brick-retirement"', frameBegin);
  const frame = host.slice(frameBegin, frameEnd);
  const tfxBegin = frame.indexOf('dispatch("beginSparseCM12TopologyEffectsPreflight", 1)');
  const tfxRecord = frame.indexOf(
    'dispatchTopologyDelta("recordCandidateTopologyEffectsFromTopologyDelta")');
  const tfxSeal = frame.indexOf('dispatch("finalizeSparseCM12TopologyEffectsPreflight", 1)');
  const vdaSeal = frame.indexOf('dispatch("finalizeSparseCM12CandidateEffectsPreflight", 1)');
  const validateDispatch = frame.indexOf('dispatch("validateAndAuthorizeShadowTopology", 1)');
  assert.ok(tfxBegin >= 0 && tfxRecord > tfxBegin && tfxSeal > tfxRecord
    && vdaSeal > tfxRecord && validateDispatch > tfxSeal && validateDispatch > vdaSeal,
  "TFX/VDA candidate recording and every fallible seal must complete before aggregate authorization");
});

test("ISA proves the exact changed set as well as closure uniqueness and generation", () => {
  const finalize = functionSource(wgsl, "cm12ISAFinalizeAuthority",
    "fn cm12ISAAuthorityReady");
  const worklist = functionSource(wgsl, "buildShadowLeafWorklist",
    "fn shadowRowScheduled");
  const changedSet = functionSource(wgsl, "finalizeSparseCM12ISAChangedSetReceipt",
    "fn validateSparseCM12InternedBoundaryDelta");
  assert.match(worklist, /cm12ISASetExpectedChanged\(deltaCompacted,/,
    "the main worklist compiler must independently author ISA changed-set receipts");
  assert.match(worklist, /mainChangedXor/);
  assert.match(worklist, /mainChangedSum/);
  assert.match(changedSet, /ISA1_AUTHORITY_BASE\+13u[\s\S]*ISA1_AUTHORITY_BASE\+8u/,
    "ISA must compare independent and observed changed-set xor receipts");
  assert.match(changedSet, /ISA1_AUTHORITY_BASE\+14u[\s\S]*ISA1_AUTHORITY_BASE\+9u/,
    "ISA must compare independent and observed changed-set sum receipts");
  assert.match(changedSet, /cm12ISARecordFault\(6u/,
    "a changed-set mismatch must reject before semantic validation and authorization");
  assert.match(finalize, /cm12IBOCandidateGeneration\(\)/,
    "ISA receipts must be bound to the candidate generation");
  assert.match(finalize, /ISA1_AUTHORITY_BASE\+4u[\s\S]*ISA1_AUTHORITY_BASE\+8u/,
    "deduped closure count must equal the uniquely claimed validation count");
});

test("the success latch is immediately before the literal final selector store", () => {
  const finalize = functionSource(wgsl, "finalizeAuthorizedShadowTopology",
    "fn publishSparseCM12TopologyVelocityRootsWork");
  const selector = "atomicStore(&topologyArena[base+2u],slot);";
  const selectorAt = finalize.indexOf(selector);
  assert.ok(selectorAt >= 0);
  const prefix = finalize.slice(0, selectorAt);
  assert.match(prefix, /vda1CommitSuccessNoFail\(/,
    "the finalizer must arm the post-selector success latch");
  const between = prefix.slice(prefix.lastIndexOf("vda1CommitSuccessNoFail("));
  const executableBetween = between.replace(/\/\/[^\n]*/g, "")
    .replace(/\/\*[\s\S]*?\*\//g, "").replace(/\s/g, "");
  assert.equal(executableBetween,
    "vda1CommitSuccessNoFail(atomicLoad(&topologyArena[base]));",
    "the success latch call must be immediately before the selector store");
  const suffix = finalize.slice(selectorAt + selector.length)
    .replace(/\/\/[^\n]*/g, "").replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/[}\s]/g, "");
  assert.equal(suffix, "", "the literal selector store must be the final executable instruction");

  const effects = functionSource(wgsl, "finalizeSparseCM12AuthorizedCandidateEffects",
    "fn scaInvalidateCellClosure");
  assert.match(effects, /vda1SealPublicationNoFail\(\)/,
    "authorized candidate effects must seal VDA publication before stable topology fields");
  assert.doesNotMatch(effects,
    /\b(?!\w*NoFail)\w*(?:Fail|Fault|Reject|Abort)\w*\s*\(|return false|atomicCompareExchange|for\s*\(|loop\s*\{/,
    "the authorized effects seal must remain non-failing and constant work");

  const lifecycle = functionSource(wgsl, "sparseCM12TopologyLifecycleAccepted",
    "fn residencyDensityThreshold");
  assert.match(lifecycle, /vda1TransactionSucceeded\(/,
    "injection must consume the post-selector VDA success latch, not phase-2 authorization");
});

test("injection success publishes before flip while rejection leaves authority unopened", () => {
  const begin = host.indexOf("Sparse CM12 resident liquid injection topology");
  const end = host.indexOf("Sparse CM12 resident liquid injection\"", begin + 1);
  assert.ok(begin >= 0 && end > begin, "injection source range must remain identifiable");
  const injection = host.slice(begin, end);
  const validation = injection.indexOf('dispatchTopology("validateAndAuthorizeShadowTopology", 1)');
  const reopen = injection.indexOf('dispatchTopology("reopenVelocityExtensionPlanForInjection", 1)');
  const candidateReopen = injection.indexOf(
    'dispatchTopology("beginSparseCM12AuthorizedCandidateEffects", 1)');
  assert.ok(validation >= 0 && candidateReopen > validation,
    "the candidate VEX generation must not reopen before topology authorization");
  assert.ok(reopen < 0 || reopen > validation,
    "the legacy VEX generation must not reopen before topology authorization");
  const candidateTopologyVex = injection.indexOf(
    'dispatchTopologyDelta("publishSparseCM12CandidateTopologyVexEffects")');
  const candidateInjectionVex = injection.indexOf(
    'dispatchTopologyDelta("publishSparseCM12CandidateInjectionVexEffects")');
  const candidateSeal = injection.indexOf(
    'dispatchTopology("finalizeSparseCM12AuthorizedCandidateEffects", 1)');
  const stableFields = injection.indexOf(
    'dispatchTopologyDelta("publishCandidateTopologyDeltaFromWorklist")');
  const selector = injection.indexOf(
    'dispatchTopology("finalizeAuthorizedShadowTopology", 1)');
  assert.ok(candidateTopologyVex > candidateReopen
    && candidateInjectionVex > candidateTopologyVex && candidateSeal > candidateInjectionVex
    && stableFields > candidateSeal && selector > stableFields,
  "an accepted injection must publish its preflighted candidate effects and fields before the selector flip");

  const reopenKernel = functionSource(vexWgsl, "reopenVelocityExtensionPlanForInjection",
    "fn cm12ExtensionRecordRoot");
  const injectKernel = functionSource(wgsl, "injectLiquid", "fn traceGammaAndBeta");
  const d4Kernel = functionSource(wgsl, "invalidateSparseCM12FrameD4ForInjection",
    "fn sparseCM12FrameControlNoop");
  assert.match(vexWgsl, /if\(!\(\$\{injectionReopenReady\}\)/,
    "the reusable VEX reopen kernel must consume its topology authorization expression");
  assert.match(wgsl,
    /injectionReopenReadyExpression:\s*\n?\s*"atomicLoad\(&topologyArena\[topologyWorklistBase\(\)\+3u\]\)==2u"/,
    "resident injection must bind VEX reopen to this transaction's authorization receipt");
  assert.match(reopenKernel, /injectionReopenReady/);
  const candidateReopenKernel = functionSource(wgsl,
    "reopenCandidateVexPlanAuthorizedNoFail", "fn cm12ResidentCellTile");
  assert.doesNotMatch(candidateReopenKernel,
    /cm12ExtensionFail|atomicCompareExchange|for\s*\(|loop\s*\{/,
    "the candidate authorized reopen seam may not fault, retry, or contend after authorization");
  for (const [label, source] of [["density injection", injectKernel],
    ["D4 invalidation", d4Kernel]] as const) {
    assert.match(source, /sparseCM12TopologyLifecycleAccepted\(\)/,
      `${label} must no-op when the topology transaction rejects`);
  }
  const firstDensityWrite = injectKernel.search(/state\[[^\]]+\]\s*=/);
  assert.ok(firstDensityWrite >= 0);
  assert.doesNotMatch(injectKernel.slice(firstDensityWrite),
    /cm12ResidentRecordExtensionIncidence|scaInvalidateCellClosure/,
    "candidate injection may not enter a fallible VEX/SCA append after its first density write");
  assert.doesNotMatch(d4Kernel, /cm12FCInvalidateD4/,
    "candidate post-injection D4 publication must use its preflighted infallible seam");
  assert.match(d4Kernel, /vda1MarkInjectionPublishedNoFail\(\)/);
});

test("injection copies the GPU delta triplet and never launches world-sized IBO work", () => {
  const begin = host.indexOf("Sparse CM12 resident liquid injection topology");
  const end = host.indexOf("Sparse CM12 resident liquid injection\"", begin + 1);
  assert.ok(begin >= 0 && end > begin, "injection topology source range must remain identifiable");
  const injection = host.slice(begin, end);
  const finalize = injection.indexOf('dispatchTopology("finalizeShadowWorklists", 1)');
  const firstEnd = injection.indexOf("closeTopologyPass();", finalize);
  const copy = injection.indexOf("this.acceptedLeafManifestBaseBytes + 4 * 12", firstEnd);
  const indirect = injection.indexOf(
    'dispatchTopologyIndirect("transferCandidateCellsFromTopologyDelta", 72)', copy);
  assert.ok(finalize >= 0 && firstEnd > finalize && copy > firstEnd && indirect > copy,
    "injection must close, copy manifest words 12..14, reopen, then consume delta indirect work");
  assert.match(injection,
    /copyBufferToBuffer\(this\.topologyArena,[\s\S]*this\.acceptedLeafManifestBaseBytes \+ 4 \* 12,[\s\S]*this\.acceptedIndirectArguments, 72, 12\)/);
  for (const kernel of ["compileSparseCM12InternedBoundaryDelta",
    "validateSparseCM12InternedBoundaryDeltaPackets",
    "compileSparseCM12TransportExecutionImageShadow",
    "replaySparseCM12TransportExecutionImageRetired",
    "replaySparseCM12InternedBoundaryDelta"] as const) {
    assert.match(injection, new RegExp(
      `dispatchTopologyDelta\\("${kernel}"\\)`),
    `${kernel} must consume the exact GPU topology-delta indirect triplet`);
  }
  assert.doesNotMatch(injection,
    /dispatchTopology\("(?:compile|validate|replay)SparseCM12InternedBoundaryDelta",/,
    "IBO construction and replay must remain O(changed-surface), including injection");
});
