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

test("transport holds a scale-invariant source lattice while sharpening stays continuous", () => {
  const stencil = functionSource(wgsl, "effectiveTransportStencilAtSpans",
    "${topologyEffectsEntries}");
  assert.match(stencil, /let spans=max\(vec3f\(1\.0\),inputSpans\)/);
  assert.doesNotMatch(stencil, /cm12TeiOwnerAtFine[\s\S]*widths/);

  const velocity = functionSource(wgsl, "sampleEffectiveTransportVelocityAtSpans",
    "fn traceEffectiveTransportCharacteristic");
  assert.match(velocity, /let spans=max\(vec3f\(1\.0\),spansInput\)/);
  const characteristic = functionSource(wgsl, "traceEffectiveTransportCharacteristic",
    "fn traceEffectiveTransportDeparture");
  assert.match(characteristic,
    /sampleEffectiveTransportVelocityAtSpans\(midpoint,spans\)/);

  const sharpeningDensity = functionSource(wgsl, "sampleSharpeningDensity",
    "fn sampleSharpeningField");
  assert.match(sharpeningDensity, /let spans=vec3f\(1\.0\)/);
  assert.match(sharpeningDensity,
    /let atUpper=select\(vec3(?:b|<bool>)\(false\),clamped>=upper,hasInteriorInterval\)/);
  assert.match(sharpeningDensity, /lower=select\(lower,lower-vec3i\(1\),atUpper\)/);
  assert.match(sharpeningDensity, /fraction=select\(fraction,vec3f\(1\.0\),atUpper\)/);
  const sharpeningField = functionSource(wgsl, "sampleSharpeningField",
    "fn traceSharpeningMass");
  assert.match(sharpeningField,
    /sampleSharpeningDensity\(position\+dx\)-sampleSharpeningDensity\(position-dx\)/);
});

test("fixed-point remainders remain at their source, independent of traversal order", () => {
  const physicalMassScale = functionSource(wgsl, "cm12PhysicalMassFixedScale",
    "fn sharpeningReceipt");
  assert.match(physicalMassScale,
    /bitcast<f32>\(p\.refinementRegionControl\.y\)/,
    "integrated-mass receipts must consume the precomputed physical quantum");
  const gammaScatter = functionSource(wgsl, "scatterGammaRow",
    "@compute @workgroup_size(64)\nfn scatterGammaSnapshotRows");
  assert.equal(gammaScatter.match(/cm12PhysicalMassFixedScale\(\)/g)?.length, 2,
    "paired gamma diffusion receipts must share the physical mass quantum");
  const sharpeningScatter = functionSource(wgsl, "scatterSharpeningCell",
    "@compute @workgroup_size(64)\nfn prepareSharpeningField");
  assert.match(sharpeningScatter,
    /removed\*cm12PhysicalMassFixedScale\(\)/);
  assert.match(sharpeningScatter,
    /addSharpeningReceipt\(cell,removedFixed-distributedFixed\)/);
  assert.doesNotMatch(sharpeningScatter, /lastCorner/);

  const capacityScatter = functionSource(wgsl, "scatterDensityCapacityRepair",
    "fn finalizeDensityCapacityRepair");
  assert.match(capacityScatter,
    /excessMass\*cm12PhysicalMassFixedScale\(\)/);
  assert.match(capacityScatter, /let distributed=share\*neighborCount/);
  assert.match(capacityScatter,
    /atomicAdd\(&conditioning\[6u\*p\.counts\.x\+cell\],-distributed\)/);
  assert.doesNotMatch(capacityScatter, /lastNeighbor/);
});

test("region-equivalent face transport scales the shared cache without taxing defaults",
  () => {
  const cachePublication = functionSource(wgsl,
    "publishSparseCM12FaceVelocitySupport",
    "@compute @workgroup_size(256)\nfn clearSparseCM12RetiredFaceVelocitySupport");
  assert.match(cachePublication, /let value=cm12EffectiveTransportVelocity\(cell\)/);
  assert.doesNotMatch(cachePublication, /incidenceBegin\(cell\)/);

  const trace = functionSource(wgsl, "traceFaceDeparture",
    "fn traceFaceDepartureAtSpans");
  assert.match(trace, /sampleFaceVelocitySupport/);
  assert.doesNotMatch(trace, /\/spans/,
    "ordinary scenes must retain the original face-trace arithmetic");
  assert.doesNotMatch(trace, /sampleFaceVelocityComponentSupport/);
  const scaledTrace = functionSource(wgsl, "traceFaceDepartureAtSpans",
    "fn presentationPhiAt");
  assert.match(scaledTrace, /length\(initial\/spans\)/,
    "face RK2 substeps must measure travel in accepted-cell spans");
  assert.match(wgsl,
    /characteristic=sampleFaceVelocitySupportAtSpans\(departure,spans\)\[axis\]/);

  const policyCache = functionSource(wgsl, "refreshSparseCM12RefinementPolicyCache",
    "fn policyTileMembershipRequired");
  assert.match(policyCache,
    /let bounds=sparseCM12RefinementRegionResolutionBounds\(brick\)/);
  assert.match(policyCache, /refinementPolicyResolutionBits\(bounds\)/);
  const applyBounds = functionSource(wgsl,
    "applySparseCM12RefinementRegionBounds", "fn templateBrickCellRange");
  assert.match(applyBounds, /cachedRefinementPolicyResolutionBounds\(brick\)/);
  assert.doesNotMatch(applyBounds,
    /sparseCM12RefinementRegionResolutionBounds\(brick\)/,
    "per-frame planning must consume the cached constraint, not rescan boxes");
  assert.match(cachePublication,
    /lane==0u&&p\.refinementRegionControl\.z!=0u[\s\S]*refreshSparseCM12RefinementPolicyCache\(brick\)/,
    "policy edits must reuse face publication instead of adding a pipeline");
  assert.doesNotMatch(resident, /dispatch\("cacheSparseCM12RefinementPolicy"/);

  const membership = functionSource(wgsl, "policyTileMembershipRequired",
    "fn brickDeeplyEnclosed");
  assert.match(membership, /ACTIVITY_REFINEMENT_POLICY_MEMBERSHIP/);
  assert.doesNotMatch(membership, /for\(var [xyz]=0u/,
    "hot membership queries must consume the tile receipt instead of scanning siblings");
  const classifyTiles = functionSource(wgsl, "classifyRefinementPolicyTiles",
    "var<workgroup> refinementPolicyRequiredResolution");
  assert.match(classifyTiles, /@builtin\(local_invocation_index\)lane:u32/);
  assert.match(classifyTiles, /linear=lane;linear<count;linear\+=64u/,
    "one leader workgroup must reduce a policy tile cooperatively");
  const closeTiles = functionSource(wgsl, "closeRefinementPolicyTileResolution",
    "fn brickTouchesDemandedMissingWorldPage");
  assert.match(closeTiles, /atomicMax\(&refinementPolicyRequiredResolution/);
  const closeFaces = functionSource(wgsl, "closePlannedResolution",
    "fn validateCandidateResolution");
  assert.doesNotMatch(closeFaces, /for\(var [xyz]=0u;[xyz]<policyScale/,
    "ordinary grading workgroups must not serially rescan a min-8 tile");

  const planning = resident.slice(resident.indexOf('stage("resolution-planning"'),
    resident.indexOf('stage("presentation-publication"'));
  assert.match(planning,
    /classifyRefinementPolicyTiles[\s\S]*planBrickResolution/);
  assert.match(planning,
    /closeRefinementPolicyTileResolution[\s\S]*closePlannedResolution/);
  const frontierSynthesis = functionSource(wgsl,
    "synthesizeSparseWorldFrontierPages", "fn restoreHostExteriorIncidence");
  assert.match(frontierSynthesis,
    /refinementPolicyTileLeader\(leaf,policyScale\)/,
    "a dynamically activated region tile must publish a cooperative leader");
});

test("dry transport retains cumulative gamma instead of injecting a new value", () => {
  const gather = functionSource(wgsl, "gatherConservativeDensity",
    "fn seedTracers");
  assert.match(gather,
    /if\(rhoNext<CM12_DRY_CELL_THRESHOLD\)\{gammaNext=state\[sourceGamma\(\)\+id\];\}/);
  assert.doesNotMatch(gather,
    /if\(rhoNext<CM12_DRY_CELL_THRESHOLD\)\{gammaNext=1\.0;\}/);
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
    /false, true, report\)/);
  const productionFactory = resident.slice(resident.indexOf("static create("),
    resident.indexOf("static createPresentationPublisherOracleForQA("));
  assert.doesNotMatch(productionFactory, /false, true, report/);
  assert.match(solver, /static createPhase1TransportReceiptOracleForQA\(/);
  assert.match(solver,
    /PHASE1_TRANSPORT_RECEIPT_QA_TOKEN[\s\S]*createPhase1TransportReceiptOracleForQA/);
});
