import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { auditWGSLComputeBindingReachability } from "../lib/wgsl-binding-reachability";
import { PassBroker } from "../lib/webgpu-pass-broker";

import {
  dilateFactorOneAirSupportDemand,
  decodeOctreeAirSupportGPUFirstError,
  factorOneAirSupportFrontierIndirectRecords,
  encodeOctreeAirSupportReconstructionHandoff,
  octreeAirSupportChangedFrontierEnabled,
  octreeAirSupportDirectIndirectArgsEnabled,
  octreeAirSupportIndirectFrontierGateEnabled,
  octreeAirSupportReconstructionCompactPassEnabled,
  octreeAirSupportTopologyReuseEnabled,
  OCTREE_AIR_SUPPORT_DIRECT_INDIRECT_ARGS_ENVIRONMENT,
  OCTREE_AIR_SUPPORT_GPU_CANDIDATE_WORDS,
  OCTREE_AIR_SUPPORT_GPU_CANDIDATE_STRIDE,
  OCTREE_AIR_SUPPORT_GPU_DISPATCH_BINDING,
  OCTREE_AIR_SUPPORT_GPU_DISPATCH_PHASES,
  OCTREE_AIR_SUPPORT_GPU_DISPATCH_PHASE_NAMES,
  OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS,
  OCTREE_AIR_SUPPORT_GPU_ERROR,
  OCTREE_AIR_SUPPORT_GPU_FACE_ADJACENCY_STRIDE,
  OCTREE_AIR_SUPPORT_GPU_PARALLEL_FRONTIER_WAVES,
  OCTREE_AIR_SUPPORT_GPU_PARALLEL_MARCH_PREFIX,
  OCTREE_AIR_SUPPORT_GPU_SCRATCH_CONTROL_WORDS,
  OCTREE_AIR_SUPPORT_GPU_SELECTOR_SLOTS,
  OCTREE_AIR_SUPPORT_GPU_TOPOLOGY_STAGE,
  OCTREE_AIR_SUPPORT_GPU_WIDE_MARCH_WAVES,
  OCTREE_AIR_SUPPORT_GPU_WORKGROUP_SIZE,
  WebGPUOctreeAirVelocitySupportProducer,
  octreeAirVelocitySupportPublicationWGSL,
  planOctreeAirSupportCompactCellAuthority,
  planOctreeAirVelocitySupportGPU,
} from "../lib/webgpu-octree-air-velocity-support-gpu";
import {
  OCTREE_AIR_SUPPORT_MAXIMUM_CASE_SELECTORS,
  OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE,
  OCTREE_AIR_SUPPORT_SELECTOR_STRIDE,
} from "../lib/webgpu-octree-air-velocity-support";

const compact = (source: string) => source.replace(/\s+/g, "");

test("bounded topology errors retain stage and item in the existing word", () => {
  assert.deepEqual(decodeOctreeAirSupportGPUFirstError(
    (OCTREE_AIR_SUPPORT_GPU_TOPOLOGY_STAGE.faceReconstruction << 24) | 1820), {
    stage: OCTREE_AIR_SUPPORT_GPU_TOPOLOGY_STAGE.faceReconstruction,
    item: 1820,
  });
  assert.match(compact(octreeAirVelocitySupportPublicationWGSL),
    /fnfailTopology\(stage:u32,item:u32\).*stage<<24u.*item&0x00ffffffu/);
});

test("same-epoch Section 5 topology reuse defaults on and retains a full-rebuild A/B", () => {
  assert.equal(octreeAirSupportTopologyReuseEnabled({}), true);
  assert.equal(octreeAirSupportTopologyReuseEnabled({ FLUID_OCTREE_AIR_SUPPORT_TOPOLOGY_REUSE: "1" }), true);
  assert.equal(octreeAirSupportTopologyReuseEnabled({ FLUID_OCTREE_AIR_SUPPORT_TOPOLOGY_REUSE: "0" }), false);
});

test("Section 5 changed-frontier defaults on and retains a construction-stable dense A/B", () => {
  assert.equal(octreeAirSupportChangedFrontierEnabled({}), true);
  assert.equal(octreeAirSupportChangedFrontierEnabled({ FLUID_OCTREE_AIR_SUPPORT_CHANGED_FRONTIER: "1" }), true);
  assert.equal(octreeAirSupportChangedFrontierEnabled({ FLUID_OCTREE_AIR_SUPPORT_CHANGED_FRONTIER: "0" }), false);
});

test("factor-1 frontier indirect convergence gate defaults on for work minimization", () => {
  assert.equal(octreeAirSupportIndirectFrontierGateEnabled({}), true);
  assert.equal(octreeAirSupportIndirectFrontierGateEnabled({
    FLUID_OCTREE_AIR_SUPPORT_INDIRECT_FRONTIER_GATE: "1",
  }), true);
  assert.equal(octreeAirSupportIndirectFrontierGateEnabled({
    FLUID_OCTREE_AIR_SUPPORT_INDIRECT_FRONTIER_GATE: "0",
  }), false);
});

test("Section 5 fixed-point reconstruction shares one pass by default", () => {
  assert.equal(octreeAirSupportReconstructionCompactPassEnabled({}), true);
  assert.equal(octreeAirSupportReconstructionCompactPassEnabled({
    FLUID_AIR_SUPPORT_RECONSTRUCT_COMPACT_PASS: "0",
  }), false);
  const encode = (environment: Readonly<Record<string, string | undefined>>) => {
    const events: string[] = [];
    let index = 0;
    const encoder = {
      beginComputePass(descriptor?: GPUComputePassDescriptor) {
        const pass = ++index;
        events.push(`begin:${descriptor?.label}`);
        return { end() { events.push(`end:${pass}`); } };
      },
      finish() { events.push("finish"); return {}; },
    } as unknown as GPUCommandEncoder;
    const broker = new PassBroker(encoder, { isolateLabels: false });
    broker.compute({ label: "March Section 5 fixed point" });
    encodeOctreeAirSupportReconstructionHandoff(broker, environment);
    broker.compute({ label: "Reconstruct Section 5 vectors" });
    broker.finish();
    return { passes: broker.computePassCount, events };
  };
  assert.deepEqual(encode({}), {
    passes: 1,
    events: ["begin:March Section 5 fixed point", "end:1", "finish"],
  });
  assert.deepEqual(encode({ FLUID_AIR_SUPPORT_RECONSTRUCT_COMPACT_PASS: "0" }), {
    passes: 2,
    events: [
      "begin:March Section 5 fixed point", "end:1",
      "begin:Reconstruct Section 5 vectors", "end:2", "finish",
    ],
  });
});

test("factor-1 frontier schedules zero every post-convergence dispatch", () => {
  assert.deepEqual(factorOneAirSupportFrontierIndirectRecords(35, 7), [
    2, 1, 1,
    1, 1, 1,
    3, 1, 1,
  ]);
  assert.deepEqual(factorOneAirSupportFrontierIndirectRecords(35, 0), [
    0, 1, 1,
    0, 1, 1,
    0, 1, 1,
  ]);
  assert.deepEqual(factorOneAirSupportFrontierIndirectRecords(35, 7, false), [
    0, 1, 1,
    0, 1, 1,
    0, 1, 1,
  ]);
  assert.throws(() => factorOneAirSupportFrontierIndirectRecords(-1, 0), /non-negative/);
  const shader = compact(octreeAirVelocitySupportPublicationWGSL);
  assert.match(shader,
    /fnfinalizeRetainedAirSupportMarchSchedule.*p\.fineFactor==1u&&\(p\.capturePreceding&8u\)!=0u.*writeDispatch\(10u.*writeDispatch\(13u.*writeDispatch\(16u/s);
  assert.match(shader,
    /fnadvanceAirSupportChangedFrontier.*letwaveActive=s\(0u\)==0u&&changed!=0u.*writeDispatch\(10u.*writeDispatch\(13u.*writeDispatch\(16u/s,
    "the singleton must publish zero records on convergence or failure");
});

test("fine demand and compact fine-cell listing are unconditional in the encoder", () => {
  const encodeSource = compact(
    WebGPUOctreeAirVelocitySupportProducer.prototype.encode.toString());
  // Fine demand is always GPU-page shaped; there is no capacity-launch arm and
  // no toggle for one, so nothing but the fine slot gates the schedule.
  // (`undefined` reads as `void0` once the test runner transpiles the source.)
  const defined = String.raw`!==(?:void0|undefined)`;
  assert.doesNotMatch(encodeSource, /compactFineDemand/);
  assert.match(encodeSource,
    new RegExp(`fineSlot${defined}&&this\\.fineDemandScheduleGroups`));
  // Compact fine cells follow the fine slot, never an environment read.
  assert.match(encodeSource, new RegExp(`constcompactFineCells=fineSlot${defined};`));
  assert.doesNotMatch(encodeSource, /COMPACT_FINE_DEMAND|COMPACT_FINE_CELLS/);
});

test("GPU plan composes both support layouts and bounded candidate schedules", () => {
  const plan = planOctreeAirVelocitySupportGPU(4_096, 4_096 * 30, [16, 16, 16]);
  assert.equal(OCTREE_AIR_SUPPORT_GPU_SELECTOR_SLOTS, OCTREE_AIR_SUPPORT_MAXIMUM_CASE_SELECTORS);
  assert.equal(OCTREE_AIR_SUPPORT_GPU_CANDIDATE_STRIDE,
    OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE + OCTREE_AIR_SUPPORT_GPU_SELECTOR_SLOTS);
  assert.equal(plan.fineCandidateOffset, 4_096 * OCTREE_AIR_SUPPORT_GPU_CANDIDATE_STRIDE);
  assert.equal(plan.candidateCapacity, plan.fineCandidateOffset + plan.domainVolume);
  assert.equal(plan.domainVolume, 4_096);
  assert.equal(plan.records.capacity, plan.support.supportCapacity);
  assert.equal(plan.support.selectorStride, OCTREE_AIR_SUPPORT_SELECTOR_STRIDE);
  assert.equal(plan.support.ownerDirectoryCellCapacity, plan.domainVolume);
  assert.equal(plan.support.ownerDirectorySlotCapacity,
    2 * (plan.rowCapacity + plan.support.supportCapacity));
  assert.equal(plan.support.ownerDirectoryBytes,
    plan.support.ownerDirectorySlotCapacity * 16);
  assert.ok(plan.offsets.candidates < plan.offsets.ranks);
  assert.ok(plan.offsets.ranks < plan.offsets.hashRecords);
  assert.ok(plan.offsets.hashRecords < plan.offsets.touchedSlots);
  assert.ok(plan.offsets.touchedSlots < plan.offsets.ownerTouchedSlots);
  assert.ok(plan.offsets.ownerTouchedSlots < plan.offsets.radixScratch);
  assert.ok(plan.offsets.radixScratch < plan.offsets.radixHistograms);
  assert.ok(plan.offsets.radixHistograms < plan.offsets.blockCounts);
  assert.ok(plan.offsets.blockCounts < plan.offsets.blockOffsets);
  assert.equal(plan.scratchBytes, plan.scratchWords * 4);
  assert.equal(plan.indirectBytes, 9 * 12);
  assert.equal(plan.faceCellCapacity, plan.rowCapacity + plan.support.supportCapacity);
  assert.equal(plan.faceCapacity, plan.faceCellCapacity * 12);
  assert.equal(plan.faceBytes, plan.faceCapacity * 16);
  // 1 incidence count + 30 incident rows + 2x12 signed patch neighbours, then
  // the 4-word resolved cell geometry (origin coordinate and extent) the march
  // reads instead of re-running faceCellIn and coord() per candidate. See
  // docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md Part E3 -- "make one encode cheap".
  assert.equal(OCTREE_AIR_SUPPORT_GPU_FACE_ADJACENCY_STRIDE, 1 + 30 + 24 + 4);
  assert.equal(plan.faceAdjacencyStride, OCTREE_AIR_SUPPORT_GPU_FACE_ADJACENCY_STRIDE);
  assert.equal(plan.faceAdjacencyBytes, plan.faceCellCapacity * plan.faceAdjacencyStride * 4);
  assert.equal(plan.faceFrontierBytes, (16 + 3 * plan.faceCapacity) * 4,
    "two compact queues and generation marks must cover every face exactly");
  assert.equal(plan.directAirVectorBytes, plan.rowCapacity * 16);
  assert.equal(plan.faceArenaBytes, plan.faceAdjacencyBytes + plan.directAirVectorBytes,
    "one face arena must carry the adjacency records and the direct-air staging suffix");
  assert.equal(OCTREE_AIR_SUPPORT_GPU_SCRATCH_CONTROL_WORDS, 96,
    "reuse controls, sparse schedules, and durable latches must remain outside the candidate arena");
});

test("ocean candidate scratch is more than halved by deterministic catalog-incidence ranks", () => {
  const rowCapacity = 65_536;
  const ocean = planOctreeAirVelocitySupportGPU(rowCapacity, rowCapacity * 30, [320, 96, 80]);
  const occurrenceStride = OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE + 3 * 68;
  const occurrenceCapacity = rowCapacity * occurrenceStride + ocean.domainVolume;
  const occurrenceBlocks = Math.ceil(occurrenceCapacity / OCTREE_AIR_SUPPORT_GPU_WORKGROUP_SIZE);
  const occurrenceScratchWords = OCTREE_AIR_SUPPORT_GPU_SCRATCH_CONTROL_WORDS
    + (OCTREE_AIR_SUPPORT_GPU_CANDIDATE_WORDS + 1) * occurrenceCapacity
    + 5 * ocean.domainVolume + 2 * occurrenceBlocks;
  const occurrenceScratchBytes = occurrenceScratchWords * 4;

  assert.equal(ocean.candidateStride, 27 + 36,
    "the row arena must contain real distinct-selector incidences, not tetra-vertex occurrences");
  assert.equal(occurrenceCapacity, 17_596_416);
  assert.equal(ocean.candidateCapacity, 4_227_072);
  assert.equal(occurrenceScratchBytes, 331_244_928);
  assert.equal(ocean.scratchBytes, 76_940_672);
  assert.ok(occurrenceCapacity > 2 * ocean.candidateCapacity,
    `candidate slots reduced only ${(occurrenceCapacity / ocean.candidateCapacity).toFixed(3)}x`);
  assert.ok(occurrenceScratchBytes > 2 * ocean.scratchBytes,
    `candidate/scratch bytes reduced only ${(occurrenceScratchBytes / ocean.scratchBytes).toFixed(3)}x`);
});

test("compact cell-authority prototype removes every domain-volume scratch term", () => {
  // Production footprint headroom for 65,536 rows is 98,304 support owners.
  // Deliberately no dimensions/domain argument: every allocation is proved by
  // the row + support footprint and every recurring scan is driven by the
  // first-claim/touched live count.
  const compact = planOctreeAirSupportCompactCellAuthority(65_536, 98_304);
  assert.equal(compact.identityCapacity, 163_840);
  assert.equal(compact.hashCapacity, 327_680);
  assert.equal(compact.hashRecordWords, 5);
  assert.equal(compact.rowCandidateCapacity, 65_536 * 63);
  assert.equal(compact.fineCandidateCapacity, 98_304);
  assert.equal(compact.candidateCapacity, 4_227_072);
  assert.equal(compact.scratchBytes, 76_940_672);
  assert.ok(154_739_996 > 2 * compact.scratchBytes,
    "the compact authority must more than halve the remaining ocean scratch, not merely remove one mask");
  assert.ok(compact.offsets.hashRecords < compact.offsets.touchedSlots);
  assert.ok(compact.offsets.touchedSlots < compact.offsets.ownerTouchedSlots);
  assert.ok(compact.offsets.ownerTouchedSlots < compact.offsets.radixScratch);
  assert.ok(compact.offsets.radixScratch < compact.offsets.radixHistograms);
  assert.ok(compact.offsets.radixHistograms < compact.offsets.blockCounts);
});

test("WGSL performs deterministic mark, parallel scan, scatter, and tag resolution", () => {
  const shader = compact(octreeAirVelocitySupportPublicationWGSL);
  assert.match(shader, /letresolvedCell=cellOf\(owner\.origin\);letresolvedSize=owner\.size/);
  assert.match(shader, /atomicCompareExchangeWeak\(&scratch\[at\],0u,wanted\)/,
    "a first claim must publish its final key without a cross-lane reservation spin");
  assert.match(shader, /atomicMax\(&scratch\[at\+3u\],INVALID-winner\)/,
    "cold-zero inverted winners must preserve authored atomic-min ordering");
  assert.match(shader, /fncountCompactAuthorityRadix0.*fnscatterCompactAuthorityRadix3/s,
    "fine-only identities must be stably ordered independently of hash claim order");
  assert.doesNotMatch(shader, /directoryWinnerOffset|directoryFlagOffset|waitHashKey/);
  assert.match(shader,
    /writeDispatch\(66u,8u,select\(vec3u\(0u,1u,1u\),dispatchFor\(s\(72u\),256u\),clean&&!reuseTopology\)\)/,
    "a cold owner hash must publish zero clear work and recurring clears must use its prior live list");
  assert.match(shader, /if\(!reuseTopology\)\{sw\(72u,s\(71u\)\);sw\(71u,0u\);\}/,
    "topology reuse must retain the owner hash's touched-slot ledger for the next fresh clear");
  assert.match(shader,
    /fnclearAdaptiveAirSupportOwnerDirectory.*slot=s\(ownerTouchedSlotOffset\(\)\+item\).*fninsertAdaptiveOwner.*atomicCompareExchangeWeak\(&supportArena\[at\],0u,wanted\).*atomicAdd\(&scratch\[71u\],1u\)/s,
    "the consumer hash must publish and consume an exact owner-slot worklist");
  assert.match(shader,
    /claimed\.old_value==wanted\).*atomicLoad\(&supportArena\[at\+1u\]\)!=size\)\{fail\(item,ERROR_TOPOLOGY\);return;\}/s,
    "one owner key may never admit two different dyadic sizes");
  assert.doesNotMatch(shader, /dispatchFor\(ownerHashCapacity\(\),256u\)/,
    "no owner-directory dispatch may be shaped by allocation capacity");
  assert.match(shader, /fnmarkAndScanAirSupportCandidates[\s\S]*marks\[lane\]=mark[\s\S]*marks\[lane\]\+=add/);
  assert.match(shader, /fnprefixAirSupportBlocks[\s\S]*blockScan\[lane\]\+=add[\s\S]*p\.blockOffsetOffset/);
  assert.match(shader, /fnscatterAirSupportRecords[\s\S]*p\.recordOffset\+output\*8u/);
  assert.match(shader, /fnresolveAirSupportTags[\s\S]*atomicStore\(&supportArena\[c\.tagWord\],tag\)/);
  assert.doesNotMatch(shader, /workgroup_size\(1\)fn(?:clear|emit|mark|scatter|resolve)/,
    "all O(N) stages must be parallel");
});

test("producer proves generations and exact owner identities before admitting air support", () => {
  const shader = compact(octreeAirVelocitySupportPublicationWGSL);
  // The producer proves internal coherence against the ACCEPTED epoch, not
  // the host's attempt stamp: when the newest candidate is rejected but a
  // clean older accepted authority survives, support deliberately rebuilds
  // against that epoch so Section 5 continues on one coherent (temporarily
  // reused) topology. Downstream consumers (supportPublicationValid) then
  // require the support epoch to equal the accepted epoch they run under.
  assert.match(shader, /if\(atomicLoad\(&accepted\.flags\)!=0u&&existingReady\)/,
    "a rejected candidate over a clean prior receipt must early-out, preserving that receipt");
  assert.match(shader, /accepted\.bank>1u/);
  assert.match(shader, /ownerPageArena\[7u\]!=accepted\.epoch/);
  assert.match(shader, /boundary!=accepted\.epoch/);
  assert.match(shader, /sw\(3u,accepted\.epoch\)/,
    "the publication must be stamped with the accepted epoch it was built from");
  assert.match(shader, /octreeOwnerPageLookup\(vec3i\(origin\)\)/);
  assert.match(shader, /if\(resolvedCell!=u32\(cell\)\|\|resolvedSize!=size\)/,
    "cube and tetrahedron vertices must both retain their exact octree-cell identity");
  assert.match(shader, new RegExp(`ERROR_TOPOLOGY:u32=${OCTREE_AIR_SUPPORT_GPU_ERROR.topology}u`));
  assert.match(shader, /publishedRow\(resolvedCell,resolvedSize\)/,
    "accepted owner identities must retain direct compact-row tags");
  assert.match(shader,
    /fnmarkFineBandAirSupportDemand.*fineWorklist\[0\]==p\.expectedFineGeneration.*fineMetadata\[id\*10u\+2u\]!=p\.expectedFineGeneration/s,
    "fine-band demand must be generated only from the exact selected fine publication");
  assert.match(shader,
    /letidentityReady=receiptReady&&accepted\.identity!=0u&&boundaryIdentity/,
    "cross-epoch reuse must prove exact structured and liquid-mask identity on the GPU");
  assert.match(shader,
    /letreuseTopology=p\.reuseTopology!=0u&&\(existingReady\|\|identityReady\)&&precedingSupportRows<=p\.supportCapacity;sw\(47u,select\(0u,1u,reuseTopology\)\)/,
    "topology reuse must be GPU-gated by either the same-epoch or exact-identity receipt");
});

test("the second same-epoch Section 5 publication reuses immutable topology and refreshes fine demand", () => {
  const shader = compact(octreeAirVelocitySupportPublicationWGSL);
  const begin = shader.slice(shader.indexOf("fnbeginAirSupportPublication"),
    shader.indexOf("fnclearAirSupportDirectory"));
  assert.match(begin,
    /receiptReady=.*airControlOffset\+3u.*accepted\.bank.*airControlOffset\+15u.*expectedFineGeneration.*existingReady=receiptReady.*airControlOffset\+2u.*accepted\.epoch.*airControlOffset\+4u.*boundaryNow/s,
    "a topology, bank, or liquid-mask/boundary epoch change must force a fresh solution");
  assert.match(begin,
    /boundaryIdentity=.*boundaryEpochOffset\+3u.*boundaryEpoch.*!=0u.*identityReady=receiptReady&&accepted\.identity!=0u&&boundaryIdentity/s,
    "cross-epoch reuse must require exact structured and boundary-liquid identities");
  for (const [word, record] of [[13, 1], [16, 2], [43, 5], [66, 8]] as const) {
    assert.match(begin, new RegExp(
      `writeDispatch\\(${word}u,${record}u,select\\(vec3u\\(0u,1u,1u\\),dispatchFor\\([^;]+\\),clean&&!reuseTopology\\)\\)`),
    `immutable air graph record ${record} must publish zero work under exact identity reuse`);
  }
  assert.match(begin,
    /writeDispatch\(19u,3u,select\(vec3u\(0u,1u,1u\),dispatchFor\(initializeItems,256u\),clean\)\)/,
    "reuse must rehydrate exactly the retained row/support footprint after clearing its live hash slots");
  const prefix = shader.slice(shader.indexOf("fnprefixAirSupportBlocks"),
    shader.indexOf("fnscatterAirSupportRecords"));
  assert.match(prefix, /if\(lane==255u&&!reuse\).*writeDispatch\(43u.*writeDispatch\(66u/s,
    "the prefix must not revive topology or owner-hash work after identity reuse zeroed it");
  assert.match(prefix, /if\(lane==255u&&reuse\)\{republishDispatch\(43u,5u\);republishDispatch\(66u,8u\);\}/,
    "a reuse dispatch must still mirror the zero-work records the staged copy carried");
  assert.match(begin,
    /letretainedGraph=existingReady&&\(p\.capturePreceding&16u\)!=0u&&atomicLoad\(&faceFrontier\[11u\]\)==RETAINED_GRAPH_VALID&&precedingSeeds<=3u\*p\.candidateCapacity&&precedingSeeds<=p\.faceCapacity/s,
    "retained values require the same settled receipt because live seed magnitude participates in winner ordering");
  const clear = shader.slice(shader.indexOf("fnclearAirSupportDirectory"),
    shader.indexOf("fnclearAirSupportCandidates"));
  assert.match(clear, /s\(47u\)!=0u.*~\(QUERY_FINE\|RECORD_FINE\)/s,
    "reuse keeps stable row demand, winners and dense row indices while clearing dynamic fine demand");
  const fineCandidates = shader.slice(shader.indexOf("fnemitFineBandAirSupportCandidates"),
    shader.indexOf("var<workgroup>emitRowActive"));
  assert.match(fineCandidates,
    /letdirect=publishedRow\(resolvedCell,owner\.size\);if\(direct!=INVALID\)\{return;\}if\(demanded==0u\)\{return;\}/,
    "the compact live-identity walk must retire direct and undemanded owners before support claims");
  const topology = shader.slice(shader.indexOf("fnresolveAirSupportTopology"),
    shader.indexOf("fnpublishAirSupportOwnerDirectory"));
  const descriptor = shader.slice(shader.indexOf("fndescriptorForIdentity"),
    shader.indexOf("fnresolveAirSupportTopology"));
  assert.match(descriptor,
    /if\(any\(probe<vec3i\(0\)\)\|\|any\(probe>=vec3i\(p\.dimensions\)\)\)\{sizes\[bit\]=size;continue;\}/,
    "positive-air topology must use the interior fan; ordinary faces apply closed/open boundaries later");
  assert.doesNotMatch(descriptor, /boundary\|=/,
    "air-only transition identities must not request liquid-only clipped catalog cases");
  assert.match(topology,
    /s\(47u\)!=0u.*stable=r\(at\+5u\)&~\(RECORD_FINE<<6u\).*dynamic=authorityFlags\(recordCell\(item\)\)&RECORD_FINE/s,
    "reused records must refresh generation-dependent fine flags without re-resolving catalog topology");
  assert.match(shader, /fnpublishAirSupportOwnerDirectory.*s\(47u\)!=0u.*return/s,
    "the identity-keyed owner directory is immutable within the accepted epoch");
  assert.match(shader, /fnresolveAirSupportFaceAdjacency.*s\(47u\)!=0u.*return/s,
    "face adjacency is immutable within the accepted epoch");
  assert.match(shader, /fnvalidateAirSupportFrontierReciprocity.*s\(50u\)!=0u.*return/s,
    "only a previously validated sparse graph may skip reciprocity validation");
  assert.match(shader,
    /fnfinalizeRetainedAirSupportMarchSchedule.*atomicStore\(&faceFrontier\[11u\],select\(0u,RETAINED_GRAPH_VALID,clean\)\)/s,
    "a failed refresh or construction must revoke retained-graph admission");
});

// A brick is brickResolution^3 samples over a lattice refined by
// fineFactor >= brickResolution, so every sample in a brick shares one
// `q / fineFactor` owner cell. Emitting the demand neighbourhood per SAMPLE
// therefore re-issued the same 27 atomics 64 times, all contending on the same
// 27 words: measured at 2.46 ms/advance of a 33.6 ms frame, and effectively
// free once reduced. atomicOr of one constant bit pattern is commutative and
// idempotent, so the reduced emit publishes an identical flag set -- the
// 500-step minimal-power-dam-break gate reproduces byte-identically.
test("fine-band demand reduces a brick to its distinct owner cells before emitting", () => {
  const shader = compact(octreeAirVelocitySupportPublicationWGSL);
  const demand = shader.slice(shader.indexOf("fnmarkFineBandAirSupportDemand"),
    shader.indexOf("fnmarkFineResolvedOwner"));
  assert.match(demand, /atomicMin\(&markFineBaseCell,cellOf\(base\)\)/,
    "in-band samples must reduce to one base owner cell per brick");
  assert.match(demand, /workgroupBarrier\(\)[\s\S]*letsharedCell=atomicLoad\(&markFineBaseCell\)/,
    "the reduction must be observed behind a workgroup barrier");
  assert.match(demand, /for\(varn=lane;n<count;n\+=64u\)/,
    "the single neighbourhood must be spread over the workgroup, not re-issued per sample");
  // Correctness rests on uniformity being PROVEN, never assumed: a brick that
  // ever spans two base cells must fall back to the per-sample emit.
  assert.match(demand, /atomicStore\(&markFineBaseSplit,1u\)/);
  assert.match(demand, /if\(atomicLoad\(&markFineBaseSplit\)!=0u\)\{if\(inBand\)\{markFineBandDemandNeighborhood\(base\)/,
    "a split brick must retain the exact per-sample neighbourhood emit");
});

test("factor-1 separable demand dilation equals the brute-force cubic neighbourhood", () => {
  const dimensions = [7, 5, 6] as const;
  const [nx, ny, nz] = dimensions;
  const source = new Uint8Array(nx * ny * nz);
  // Exercise boundaries, separated components, and overlapping cubes.
  for (const [x, y, z] of [[0, 0, 0], [6, 4, 5], [3, 2, 3], [4, 1, 2], [1, 4, 1]]) {
    source[x + nx * (y + ny * z)] = 1;
  }
  const brute = (radius: number) => {
    const result = new Uint8Array(source.length);
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        for (let dz = -radius; dz <= radius; dz += 1) {
          for (let dy = -radius; dy <= radius; dy += 1) {
            for (let dx = -radius; dx <= radius; dx += 1) {
              const qx = x + dx, qy = y + dy, qz = z + dz;
              if (qx >= 0 && qx < nx && qy >= 0 && qy < ny && qz >= 0 && qz < nz
                && source[qx + nx * (qy + ny * qz)] !== 0) {
                result[x + nx * (y + ny * z)] = 1;
              }
            }
          }
        }
      }
    }
    return result;
  };
  for (const radius of [0, 1, 3]) {
    assert.deepEqual(dilateFactorOneAirSupportDemand(dimensions, source, radius), brute(radius));
  }
  const shader = compact(octreeAirVelocitySupportPublicationWGSL);
  assert.match(shader, /fncompactFineDemandActive\(\)->bool\{return\(p\.capturePreceding&4u\)!=0u;\}/);
  assert.match(shader,
    /if\(p\.fineFactor==1u\)\{if\(inBand\)\{markFineBandDemandNeighborhood\(base\);\}return;\}/,
    "factor one must expand only live in-band samples, not sweep the air arena");
  assert.doesNotMatch(WebGPUOctreeAirVelocitySupportProducer.prototype.encode.toString(),
    /dilateFineBandAirSupportDemand/);
});

// The candidate arena is provisioned for rowCapacity rows; only the accepted
// rows emit. Sweeping the provisioned extent walked (rowCapacity-rows)*stride
// dead 3-word slots per encode, twice per advance.
test("candidate sweeps are bounded by the live row extent, not the provisioned capacity", () => {
  const shader = compact(octreeAirVelocitySupportPublicationWGSL);
  assert.match(shader,
    /fncandidateBoundFor\(rows:u32\)->u32\{returnmin\(rows\*p\.candidateStride\+s\(30u\),p\.candidateCapacity\)/,
    "the swept extent must be the live rows plus the fine block, clamped to capacity");
  // Relocating the fine block to follow the LAST ACCEPTED row keeps the live
  // index space contiguous. It cannot reorder anything: every row candidate
  // index stays below rows*candidateStride and so still precedes every fine
  // candidate, preserving ranks, directory winners and support-record slots.
  assert.match(shader, /fnfineCandidateBase\(\)->u32\{returnmin\(s\(2u\)\*p\.candidateStride,p\.fineCandidateOffset\)/);
  assert.match(shader,
    /fnscatterCompactFineCandidates[\s\S]*letoutput=fineCandidateBase\(\)\+s\(p\.blockOffsetOffset\+block\)\+localRank/,
    "fine-only candidates must follow live rows in stable compact cell order");
  assert.match(shader, /fnclearAirSupportCandidates[\s\S]*if\(item<s\(6u\)\)\{setCandidate/,
    "the clear must cover exactly the swept extent");
  assert.doesNotMatch(shader, /if\(item<p\.candidateCapacity\)/,
    "no sweep may still be bounded by the provisioned candidate capacity");
});

test("air-support membership is limited to the proven consumer reach corridor", () => {
  const shader = compact(octreeAirVelocitySupportPublicationWGSL);
  const emit = shader.slice(shader.indexOf("fnemitFineBandAirSupportCandidates"),
    shader.indexOf("var<workgroup>emitRowActive"));
  assert.doesNotMatch(emit, /OWNER_PAGE_LOOKUP_TOPOLOGY/);
  assert.match(emit, /letdemanded=s\(at\+2u\).*if\(demanded==0u\)\{return;\}/s,
    "only identities in the transport/interpolation demand cone may become march destinations");
  assert.match(shader,
    /fnfinalizeAirSupportMetadata\(\).*if\(s\(41u\)!=0u\)\{publishCarrierFreeForensic\(s\(42u\)\);fail\(s\(42u\),ERROR_TOPOLOGY\);\}/s,
    "an air face with no carrier ANYWHERE on its axis must reject, not publish provisional air");
  // The ledger word must name the failing face without aliasing. Packing
  // (flags<<16)|(cell<<3)|axis overlapped the flags byte with the cell index on
  // any domain above 8192 cells, so a decoded identity was not trustworthy; the
  // owned-face item index is unique and the producer expands it in finalize.
  assert.match(shader,
    /atomicMin\(&scratch\[42u\],faceRow\*12u\+4u\*axis\+quadrant\)/,
    "the carrier-free ledger must record a lossless owned-face item index");
  assert.doesNotMatch(shader, /atomicMin\(&scratch\[42u\],\(\(\(cell\.w>>6u\)&0xffu\)<<16u\)/,
    "the aliasing (flags<<16)|(cell<<3)|axis ledger packing must not return");
  // Section 5's extension is a GLOBAL closest-point transform. The march is an
  // acceleration of it that is exact only inside one connected component of the
  // demand corridor, and the corridor is not connected in general: an
  // unresolved sub-grid film owns no liquid row, so its band demands a corridor
  // island with no seed in it. Patches the march cannot reach must therefore be
  // resolved by the same transform evaluated exhaustively -- never by a
  // substituted value, and never by relaxing the terminal ledger.
  const completion = shader.slice(shader.indexOf("fncompleteAirSupportIncidentFaces"),
    shader.indexOf("fnextendFace"));
  assert.match(completion,
    /letmarched=faceA\[item\];if\(marched\.w==0u\)\{atomicAdd\(&scratch\[86u\+axis\],1u\);letrecovered=closestSeedFaceAt\(faceRow,axis,quadrant,true\);if\(recovered\.w==0u\)\{atomicAdd\(&scratch\[89u\+axis\],1u\);\}else\{faceA\[item\]=recovered;\}\}/,
    "a march-unreachable patch must take the exact global closest seed face, and must still be counted when no seed exists at all");
  assert.match(shader,
    /receiptReady=.*airControlOffset\+15u.*expectedFineGeneration/s,
    "topology reuse must prove the exact fine dependency generation that defines corridor membership");
});

test("fine-band support closes over the exact regular or transition interpolation dependencies", () => {
  const shader = compact(octreeAirVelocitySupportPublicationWGSL);
  const closure = shader.slice(shader.indexOf("fncloseFineBandAirSupportInterpolationDemand"),
    shader.indexOf("fnemitFineBandAirSupportCandidates"));
  assert.match(closure,
    /if\(caseId==0u&&markExactRegularNeighborhood\(owner\.origin,owner\.size,item\)\)\{return;\}/,
    "case zero may use trilinear closure only after proving its exact regular neighborhood");
  assert.match(closure, /markExactRegularNeighborhood.*for\(vartetra=0u/s,
    "a body-diagonal transition retains exact regular octants and the case-zero Delaunay fan");
  assert.match(closure,
    /letpacked=tetrahedra\[first\+tetra\].*letselector=.*letv=tetraVertices\[selector\].*letselectorCenter=center\+f32\(owner\.size\)\*inverseTransform.*markFineResolvedOwner\(selectorCenter,selectorSize,item\)/s,
    "transition demand must include the actual catalog tetrahedron selector owners");
  // Section 6.2's fan is scene independent, and everything below the selector
  // byte is a function of the selector alone. The generated catalog carries
  // 136.4 vertex occurrences but only 24.7 DISTINCT selectors per fanned entry
  // (5.51x; 1,102,236 occurrences over 199,872 distinct across all 8,083), so
  // the occurrence walk resolved every owner page ~5.5 times over. The closure
  // must publish the DISTINCT set: both terminal operations are idempotent and
  // receive identical arguments on every occurrence, so this is bit-exact, and
  // it is what makes this pass 1.86 rather than 2.65 ms/advance on the mini lane.
  assert.match(closure,
    /varseen=array<u32,8>\(0u,0u,0u,0u,0u,0u,0u,0u\);for\(vartetra=0u/,
    "the transition fan must carry a distinct-selector set rather than re-resolving occurrences");
  assert.match(closure,
    /letselector=\(packed>>\(8u\*vertex\)\)&255u;letselectorBit=1u<<\(selector&31u\);if\(\(seen\[selector>>5u\]&selectorBit\)!=0u\)\{continue;\}seen\[selector>>5u\]\|=selectorBit;/,
    "the distinct-selector guard must precede the vertex load, not merely the owner-page lookup");
  assert.match(shader,
    /fnmarkExactRegularNeighborhood.*letexpectedCenter=clamp\(center\+.*vec3f\(half\),vec3f\(p\.dimensions\)-vec3f\(half\)\).*letresolved=octreeOwnerPageLookup.*letmatches=resolved\.size==size.*atomicOr/s,
    "regular closure must require the exact same-size center after the consumer's boundary clamp");
  assert.match(shader,
    /fnmarkFineResolvedOwner\(expectedCenter:vec3f,expectedSize:u32,item:u32\).*owner\.size!=expectedSize\|\|any\(abs\(ownerCenter-expectedCenter\)>vec3f\(tolerance\)\).*failTopology\(2u,item\)/s,
    "fine closure publication must reject a containing owner with the wrong identity");
  assert.match(closure,
    /letsizef=f32\(owner\.size\)\*v\.w;letselectorSize=u32\(round\(sizef\)\).*selectorSize==0u\|\|abs\(sizef-f32\(selectorSize\)\)>2e-4.*letoriginf=selectorCenter-vec3f\(\.5\*sizef\);letorigin=vec3i\(round\(originf\)\).*abs\(originf-vec3f\(origin\)\)>vec3f\(2e-4\).*origin<vec3i\(0\).*origin\+vec3i\(i32\(selectorSize\)\)>vec3i\(p\.dimensions\).*markFineResolvedOwner/s,
    "transition closure may use constant boundary extension only for a geometrically proven exterior selector");
  assert.match(closure,
    /v\.w<=0\.\|\|!finiteValue\(v\.x\)\|\|!finiteValue\(v\.y\)\|\|!finiteValue\(v\.z\)\|\|!finiteValue\(v\.w\).*fail\(item,ERROR_CATALOG\)/s,
    "non-finite selector geometry must fail as a catalog error rather than masquerade as physical exterior");
  assert.doesNotMatch(closure, /maxLeaf|conservative|approximate/,
    "adaptive closure must be topology exact rather than a guessed spatial radius");
  assert.match(shader, /constQUERY_FINE:u32=0x40000000u/);
  assert.match(closure, /&QUERY_FINE\)==0u/);
  assert.doesNotMatch(closure, /markFineResolvedOwner.*QUERY_FINE/s,
    "VALUE_ONLY closure writes must not recursively become new queries in the same dispatch");
});

test("transition and regular demand slots are fixed and physical exterior stays invalid", () => {
  const shader = compact(octreeAirVelocitySupportPublicationWGSL);
  assert.match(shader, /var<workgroup>emitRowRegular:atomic<u32>/);
  // The 27-site proof runs for EVERY enabled row: eligibility is geometric
  // (boundary-clamped uniformity), not caseId==0, so wall-touching rows keep
  // the exact per-axis staggered face basis. Nonzero-case regular rows also
  // publish their retained tet fan for transition consumers.
  assert.match(shader,
    /if\(lane<27u\).*fineResolvedOwnerMatches\(expectedCenter,g\.y,itemBase\+lane\).*workgroupBarrier\(\);letregular=workgroupUniformLoad\(&emitRowRegular\)!=0u/s,
    "one row-owned workgroup must cooperatively prove the 27-site cube exactly once");
  assert.match(shader, /letneedsSelectors=!regular\|\|g\.z!=0u;/,
    "a regular wall row publishes both closures so transition sampling keeps its selectors");
  assert.match(shader, /if\(!valid&&needsSelectors\)\{fail\(itemBase,ERROR_CATALOG\);\}/,
    "a regular row whose transition fan is corrupt must reject instead of publishing only half its closure");
  assert.match(shader, /if\(regular&&lane<27u\)\{letlocal=lane/);
  assert.match(shader, /var<workgroup>emitRowSelectorFirst:array<atomic<u32>,256>/);
  assert.match(shader, /letoccurrenceActive=.*occurrence<3u\*count/);
  assert.match(shader, /selector=\(packed>>\(8u\*\(occurrence%3u\)\)\)&255u/);
  assert.match(shader,
    /atomicMin\(&emitRowSelectorFirst\[selector\],occurrence\).*atomicLoad\(&emitRowSelectorFirst\[selector\]\)==occurrence/s,
    "only the deterministic first occurrence of a selector may acquire an incidence rank");
  assert.match(shader,
    /for\(varoffset=1u;offset<256u;offset<<=1u\).*letselectorCount=marks\[255u\];if\(selectorCount>36u\).*letselectorRank=marks\[lane\]-1u;letitem=itemBase\+27u\+selectorRank/s,
    "first occurrences must prefix into the proven 36-slot catalog incidence cap");
  assert.doesNotMatch(shader, /letitem=itemBase\+27u\+occurrence/,
    "tetra-vertex occurrences must not provision candidate slots");
  // No early return: a regular wall row's lane may still owe a selector
  // occurrence for the dual-closure publication after stamping its
  // out-of-domain cube tag INVALID.
  assert.match(shader, /if\(!inDomain\)\{atomicStore\(&supportArena\[tag\],INVALID\);\}else\{/);
  assert.match(shader, /any\(origin<vec3i\(0\)\).*atomicStore\(&supportArena\[tag\],INVALID\)/);
  assert.match(shader, /writeDispatch\(16u,2u,select\(vec3u\(0u,1u,1u\),dispatchFor\(rows,1u\),clean&&!reuseTopology\)\)/,
    "the indirect schedule must dispatch one 2-D-safe workgroup per row");
  assert.doesNotMatch(shader, /regularNeighborhoodExact/,
    "candidate lanes must not each repeat the complete 27-owner proof");
});

test("Section 5 chain is power-face seeded, persistently marched, and reconstructed at power centroids", () => {
  const shader = compact(octreeAirVelocitySupportPublicationWGSL);
  const faceValue = shader.slice(shader.indexOf("fnprojectedAxisFaceValue"), shader.indexOf("fnneighborIdentity"));
  assert.match(faceValue, /structuredAuthority\[handleAt\][\s\S]*p\.valuesOffset\+handle[\s\S]*catalogFaces\[global\]/,
    "the seed source must be the accepted projected power-face samples with their catalog geometry");
  assert.match(faceValue, /abs\(aligned\)>=0\.999/, "only axis-normal power faces may seed an ordinary axis face");
  assert.match(faceValue, /abs\(centroid\[axis\]-patchCenter\[axis\]\)<=tolerance/,
    "the seeding power face must be coplanar with the seeded patch");
  assert.match(faceValue, /bestLocal==INVALID\|\|separation<bestDistance/,
    "nearest-face selection is strict-less so the lowest local slot index wins exact ties");
  assert.doesNotMatch(shader, /fnprojectedPowerVector/,
    "no cell-centred reconstructed vector may masquerade as a face seed");
  const seed = shader.slice(shader.indexOf("fnairSupportSeedCarrier"), shader.indexOf("fnbetterFace"));
  assert.match(seed, /projectedAxisFaceValue\(faceRow,axis,patchCenter\)/);
  assert.match(seed, /adjacencyPositive\(faceRow,axis,local%4u\)/);
  assert.match(seed, /projectedAxisFaceValue\(otherRow,axis,patchCenter\)/,
    "an air-owned patch seeds from its positive liquid neighbour's coincident face");
  assert.doesNotMatch(seed, /sum\/f32\(weight\)|sum\+=|weight\+=/,
    "the seed copies one exact face value and never averages adjacent liquid cells");
  assert.match(seed, /vec4u\(bitcast<u32>\(seed\.x\),0u,item,1u\)/,
    "the seed distance origin remains the seeding face patch centre");
  const publication = shader.slice(shader.indexOf("fnresolveAirSupportFaceAdjacency"), shader.indexOf("var<workgroup>seedCounts"));
  assert.match(publication, /catalogNeighbor\(cell,header\.x\+localFace\)[\s\S]*faceRowForIdentity\(identity\)/,
    "T-junction catalog incidence must be resolved during topology publication");
  assert.match(publication, /signedFaceNeighborIdentity\(cell,axis,quadrant,false\)[\s\S]*signedFaceNeighborIdentity\(cell,axis,quadrant,true\)/,
    "all twelve signed regular-face patches must be published once");
  const march = shader.slice(shader.indexOf("fnextendFace"), shader.indexOf("fnquadrantAt"));
  assert.match(march, /adjacencyIncidentCount\(faceRow\)[\s\S]*adjacencyIncident\(faceRow,localFace\)/);
  assert.doesNotMatch(march, /tagForIdentity|faceRowForTag|octreeOwnerPageLookup|catalogNeighbor|caseHeader/,
    "the face march must be a pure indexed gather over published adjacency");
  assert.match(shader,
    /fnbetterFace\(item:u32,candidate:vec4u,best:vec4u\).*candidateDistanceSquared=bitcast<f32>\(candidate\.y\).*candidateMagnitude=abs\(bitcast<f32>\(candidate\.x\)\).*candidateDistanceSquared==bestDistanceSquared.*candidateMagnitude<bestMagnitude/s,
    "equidistant closest faces must first use reflection-invariant normal-speed magnitude");
  // The carrier metric is the true Euclidean distance to the ORIGINAL seed
  // patch (candidate.z, preserved verbatim through every copy) — the
  // closest-point-transform ranking. Accumulating per-hop path length made
  // the metric an axis-graph geodesic (Manhattan-like), which under-drives
  // diagonal spreading and squares off the dam front.
  // Renegotiated by name for the resolved face geometry -- see
  // docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md Part E3. The metric is unchanged:
  // still the Euclidean distance from this patch's centre to the centre of the
  // ORIGINAL seed patch. Only the way the seed patch's centre is obtained
  // changed, from re-resolving the row's cell and running coord() to reading
  // the origin/extent resolveAirSupportFaceAdjacency published for that row.
  // Renegotiated by name a second time for the loop-invariant patch centre.
  // The metric is still the Euclidean distance from this patch's centre to the
  // centre of the ORIGINAL seed patch; the marching patch's own centre is now
  // resolved once per invocation instead of once per candidate, which is why
  // it appears as a value rather than as a re-derivation from `item`.
  assert.match(march,
    /letitemCenter=faceCenterQuarter\(item\);/,
    "the marching patch's own centre must be resolved once, outside the candidate scan");
  assert.match(march,
    /distanceSquared=faceDistanceSquaredFrom\(itemCenter,candidate\.z\)/,
    "the relaxation must rank sources by squared Euclidean distance to the original seed, not accumulated hop length");
  assert.match(shader, /fnrowSeedDistance.*returnsqrt\(bestSquared\)/s,
    "physical distance is recovered once per row for the detached-air gravity ramp");
  assert.match(shader,
    /fnrowSeedDistance\(faceRow:u32\)->f32[\s\S]*ownedFaceQuadrant\(faceRow,axis,quadrant\)[\s\S]*adjacencyNegative\(faceRow,axis,quadrant\)[\s\S]*ownedFaceQuadrant\(negativeRow,axis,negativeQuadrant\)/,
    "the scalar seed distance must include both incident sides because reflection reverses face ownership");
  assert.doesNotMatch(march, /length\(/,
    "the inner candidate scan must not pay a square root for an ordering-only metric");
  assert.match(shader,
    /fnfaceCenter\(item:u32\)->vec3f\{.*letg=adjacencyGeometry\(faceRow\);letorigin=vec3f\(g\.xyz\);letextent=f32\(g\.w\)/s,
    "the patch centre must come from the resolved per-face-row geometry, not a per-candidate cell resolution");
  assert.match(shader,
    /fnsetAdjacencyGeometry\(faceRow:u32,cell:vec4u\)\{.*letq=coord\(cell\.x\).*faceAdjacency\[at\+3u\]=cell\.y/s,
    "the resolved geometry must be exactly coord(cell.x) and cell.y so the centre stays bit-identical");
  assert.match(shader,
    /letcell=faceCell\(faceRow\);if\(cell\.x==INVALID\)\{failTopology\(7u,faceRow\);return;\}setAdjacencyGeometry\(faceRow,cell\)/,
    "face-row geometry must be resolved by the adjacency publication, from the same faceCell the reader used to call");
  // The old floating-point centre is not recomputed in the candidate scan;
  // exact quarter-grid distance is delegated to faceDistanceSquared.
  assert.doesNotMatch(march, /letcenter=faceCenter\(item\);/);
  const candidateScan = march.slice(march.indexOf("for(varlocalFace"),
    march.indexOf("letchanged=any(best!=current)"));
  assert.ok(candidateScan.length > 0);
  assert.doesNotMatch(candidateScan, /s\(29u\)|s\(2u\)|s\(4u\)|s\(8u\)|faceCenter\(item/,
    "no dispatch-uniform scratch word or invariant centre may be re-read inside the candidate scan");
  assert.doesNotMatch(candidateScan, /faceCellIn|coord\(/,
    "the candidate scan must not re-resolve a face row's cell or re-run coord()'s emulated integer divisions");
  assert.match(shader, /clean=errors==0u&&\(s\(35u\)\|s\(36u\)\|s\(37u\)\)==0u/,
    "publication requires a deliberate GPU-observed no-change wave");
  assert.match(march,
    /fnmarchAirSupportFacesChangedFrontier.*frontierCurrentCount=atomicLoad\(&faceFrontier\[axis\]\).*letsource=atomicLoad\(&faceFrontier\[frontierQueueBase\(0u,axis\)\+local\]\).*appendFrontierDestination.*letactiveCount=workgroupUniformLoad\(&frontierActiveCount\).*extendFace\(item,true\).*if\(any\(proposal!=previous\)\).*frontierCurrentCount=frontierChangedCount.*wave>=max\(1u,axisCapacity\)/s,
    "one persistent workgroup per axis must expand, deduplicate, relax, and compact only the changed frontier under the |V_axis| proof bound");
  assert.match(shader,
    /fnvalidateAirSupportFrontierReciprocity.*adjacencyIncident\(faceRow,local\).*adjacencyIncident\(other,back\)==faceRow.*failTopology\(9u,faceRow\)/s,
    "reverse frontier expansion must be admitted only for a proven reciprocal incidence graph");
  assert.match(shader,
    /fncompactAirSupportSeedFrontier.*faceA\[item\]\.w!=0u.*appendSeedFrontier/s,
    "the first frontier must contain only actual projected-face seeds");
  assert.match(shader,
    /fnseedRetainedAirSupportFaces.*invocation<s\(29u\).*s\(50u\)!=0u.*letcarrier=airSupportSeedCarrier\(invocation\).*faceB\[invocation\]=carrier/s,
    "a retained publication must recompute seeds over only the exact GPU-published live-face schedule");
  assert.match(shader,
    /fnrefreshRetainedAirSupportFaceValues.*letseedItem=carrier\.z.*letseed=faceB\[seedItem\].*carrier\.x!=seed\.x.*faceA\[item\]=carrier/s,
    "retained closest-source identities and distances must update only from their exact winning seed value");
  assert.match(shader,
    /fnfinalizeRetainedAirSupportMarchSchedule.*s\(25u\)!=s\(49u\).*atomicStore\(&faceFrontier\[10u\],1u\).*writeDispatch\(32u,4u,select\(vec3u\(0u,1u,1u\),dispatchFor\(s\(29u\),256u\),clean&&!retained\)\)/s,
    "equal seed membership must admit a zero second-publication march schedule and mismatches must fail closed");
  assert.match(shader,
    /fnpackedFrontierLane\(packed:u32\)->vec2u\{returnvec2u\(packed%3u,packed\/3u\)/,
    "occupancy-wide waves must pack live axis queues without provisioned-capacity holes");
  assert.match(shader,
    /fnexpandAirSupportChangedFrontier.*appendFrontierDestination.*fnrelaxAirSupportChangedFrontier.*extendFace\(item,true\).*fncommitAirSupportChangedFrontier.*proposal=faceB\[item\].*faceA\[item\]=proposal.*fnadvanceAirSupportChangedFrontier/s,
    "parallel sparse waves must retain dispatch barriers between expand, relax, commit, and count publication");
  assert.doesNotMatch(march, /STRUCTURED_VELOCITY_EXTENSION_LAYERS|maximumRelaxationWaves|dimensions\.reduce/,
    "the paper march must not be replaced by a widened scene or domain wave count");
  assert.doesNotMatch(march, /sum\+=|weight\+=/,
    "marching copies one deterministic closest source and never averages");
  assert.match(shader, /letcentroid=anchorCenter\+f32\(cell\.y\)\*inverseTransform\(slot\.areaCentroid\.yzw,transform\)/);
  assert.match(shader, /regularVectorAt\(faceRow,centroid\)[\s\S]*canonicalAirSupportDot\(interpolated\.xyz,normal\)/);
  assert.match(shader,
    /fncanonicalAirSupportSum\(values:array<f32,31>,count:u32\)[\s\S]*termsX\[local\]=term\.x[\s\S]*canonicalAirSupportSum\(termsX,header\.y\)/,
    "reflected catalog incidences must reconstruct one exact support vector");
  assert.match(shader,
    /lett=round\(clamp\(\(point\[axis\]-origin\[axis\]\)\/f32\(cell\.y\),0\.,1\.\)\*65536\.\)\/65536\.;terms\[termCount\]=canonicalAirSupportPair/,
    "opposite cell sides must use complementary dyadic interpolation weights");
  assert.match(shader,
    /if\(coordinate<\.5\)\{fixedMask\|=bit;\}elseif\(coordinate>\.5\)[\s\S]*for\(varquadrant=0u;quadrant<4u;quadrant\+=1u\)[\s\S]*canonicalAirSupportSum\(terms,termCount\)\/f32\(termCount\)/,
    "a power centroid on a transverse patch seam must average every incident quadrant");
  assert.match(shader,
    /fncanonicalSeedOffset\(item:u32,seed:u32\)->vec3i[\s\S]*powerTransformVector\(faceCenterQuarter\(seed\)-faceCenterQuarter\(item\),faceCell\(faceRow\)\.w&63u\)[\s\S]*fnbetterFace\(item:u32,candidate:vec4u,best:vec4u\)/,
    "equal-distance closest faces must be ordered in the destination's canonical frame, never by absolute row id");
  assert.match(shader,
    /fnfaceDistanceSquaredFrom\(itemCenter:vec3i,seed:u32\)->f32[\s\S]*itemCenter-faceCenterQuarter\(seed\)[\s\S]*return\.0625\*f32\(squared\)[\s\S]*distanceSquared=faceDistanceSquaredFrom\(itemCenter,candidate\.z\)/,
    "reflected closest-face distances must be exact quarter-grid integers before conversion to f32");
  assert.match(shader,
    /positive\.w==0u&&u32\(origin\[axis\]\)\+cell\.y==p\.dimensions\[axis\][\s\S]*p\.closedBoundaryMask/,
    "a positive closed wall supplies stationary solid only when the air march has no carrier");
  assert.match(shader,
    /else\{negative=incidentFaces\[[\s\S]*if\(negative\.w==0u&&u32\(origin\[axis\]\)==0u&&\(p\.closedBoundaryMask/,
    "the negative closed wall must consume its cached carrier before the same stationary fallback");
  assert.match(shader,
    /fnclosestSeedFaceAt\(faceRow:u32,axis:u32,quadrant:u32,positive:bool\)[\s\S]*signedFaceCenterQuarter[\s\S]*bitcast<f32>\(candidate\.y\)!=0\.[\s\S]*fncompleteAirSupportIncidentFaces[\s\S]*s\(50u\)!=0u[\s\S]*completion=faceA\[retained\.z\][\s\S]*completion=closestSeedFaceAt\(faceRow,axis,quadrant,false\)[\s\S]*else\{negative=incidentFaces\[/,
    "a missing negative owner must cache its closest seeded ordinary face once per publication");
});

test("closed-wall air reconstruction is exactly reflection odd after marched carriers arrive", () => {
  const shader = compact(octreeAirVelocitySupportPublicationWGSL);
  const regular = shader.slice(shader.indexOf("fnregularVectorAt"),
    shader.indexOf("fnrowSeedDistance"));
  assert.match(regular,
    /if\(positive\.w==0u&&u32\(origin\[axis\]\)\+cell\.y==p\.dimensions\[axis\].*positive=vec4u\(bitcast<u32>\(0\.\),0u,INVALID,1u\)/s);
  assert.match(regular,
    /negative=incidentFaces\[.*if\(negative\.w==0u&&u32\(origin\[axis\]\)==0u/s);
  const completion = shader.slice(shader.indexOf("fncompleteAirSupportIncidentFaces"),
    shader.indexOf("fnextendFace"));
  assert.match(completion, /if\(adjacencyNegative\(faceRow,axis,quadrant\)==INVALID\)\{/);
  assert.doesNotMatch(completion, /closedBoundaryMask/,
    "closed-wall air needs the same missing-side carrier cache as interior air");

  // Case 0 reconstructs the two axis faces with coefficients -1/2 and +1/2.
  // Both domain signs consume the reflected marched value; zero remains only
  // the symmetric no-carrier fallback.
  const interior = Math.fround(2.7745232582092285);
  const negativeWallCell = Math.fround(0.5 * Math.fround(interior + interior));
  const positiveWallCell = Math.fround(0.5 * Math.fround(-interior - interior));
  assert.equal(positiveWallCell, Math.fround(-negativeWallCell));
});

test("no liquid film fallback seeds a carrier-free face", () => {
  // A face with no carrier on either side is a genuine air-support rejection
  // and must surface as one. The former film fallback substituted the row's
  // cell-centre velocity there, which masked the rejection instead of
  // resolving it -- on the large dam it carried a dead fine band for 165
  // further steps before anything failed loudly. Rejections are fatal, so the
  // fallback is deleted rather than gated.
  const shader = compact(octreeAirVelocitySupportPublicationWGSL);
  for (const symbol of ["seedAirSupportFilmFaces", "seedRetainedAirSupportFilmFaces",
    "airSupportFilmSeedCarrier", "filmSeedForLiquidRow", "fineInterfaceInCell",
    "fineSeedSample", "fineBandSeedSourceValid"]) {
    assert.doesNotMatch(shader, new RegExp(symbol),
      `${symbol} is part of the deleted film fallback and must not return`);
  }
  assert.doesNotMatch(shader, /scratch\[63u\]/,
    "the film seed tally is gone; the retained membership identity compares scratch[25] alone");
  assert.match(shader, /if\(retained&&s\(25u\)!=s\(49u\)\)\{failTopology\(10u,s\(25u\)\)/,
    "retained membership must reconcile against the projection seed count alone");
});

test("host uses only GPU-published live schedules for fine demand and changed frontiers", () => {
  const encode = compact(WebGPUOctreeAirVelocitySupportProducer.prototype.encode.toString());
  assert.match(encode, /dispatchWorkgroups\(1\).*publish\(/);
  // Ten `publish` sites plus the three-copy identity block: the same thirteen
  // schedule publications the staged encoder always had. `publish` is the one
  // funnel, so a fourteenth cannot reach the command stream unaudited.
  assert.equal((encode.match(/publish\(/g) ?? []).length, 10,
    "identity, radix, sparse-cell, face, retained-march, and factor-1 convergence schedules must remain GPU authored");
  assert.equal((encode.match(/updateIndirectBuffer/g) ?? []).length, 4,
    "staging survives only inside `publish` and the one non-contiguous identity block");
  for (const [name, offset] of [["clearAirSupportDirectory", 0], ["clearAirSupportTags", 12],
    ["emitAirSupportCandidates", 24], ["markAndScanAirSupportCandidates", 36]] as const) {
    assert.match(encode, new RegExp(`run\\("${name}",${offset}\\)`));
  }
  assert.match(encode, /publish\(32,\[4,5\],"face",/);
  assert.match(encode, /broker\.updateIndirectBuffer\(this\.scratch,source\*4,this\.indirect,records\[0\]\*12,records\.length\*12\)/,
    "the staged arm must still copy exactly the record family it publishes");
  assert.match(encode, /dispatchWorkgroupsIndirect\(indirectFor\(48\),48\)/);
  assert.match(encode,
    /prepareFineBandAirSupportDemand[\s\S]*owners\[4\]=this\.dispatchArenas\.fineDemand[\s\S]*markFineBandAirSupportDemand.*dispatchWorkgroupsIndirect\(indirectFor\(48\),48\)/s,
    "fine demand must launch one workgroup per live page from a GPU-authored schedule");
  assert.match(encode, /markFineBandAirSupportDemand.*dispatchWorkgroupsIndirect\(indirectFor\(48\),48\)/s,
    "fine demand must have no provisioned-capacity launch branch");
  assert.doesNotMatch(encode, /maximumResidentBricks|domainVolume\/OCTREE_AIR_SUPPORT_GPU_WORKGROUP_SIZE|dilateFineBandAirSupportDemand/,
    "production air support must not encode capacity/domain-shaped fine work");
  assert.match(encode, /dispatchWorkgroupsIndirect\(indirectFor\(60\),60\)/);
  assert.match(encode,
    /publish\(43,\[5\],"support",[^)]*\);[\s\S]*resolveAirSupportTopology.*dispatchWorkgroupsIndirect\(indirectFor\(60\),60\)/s,
    "support topology work must use the exact GPU-authored support-row count, not capacity");
  assert.match(encode, /run\("finalizeAirSupportMetadata"\)[\s\S]*commitAirSupportDirectRows[\s\S]*run\("commitAirSupportPublication"\)/);
  assert.doesNotMatch(encode, /dispatchWorkgroups\(this\.plan\.faceCapacity/);
  assert.match(encode, /clearBuffer\(this\.scratch,32\*4,6\*4\)/,
    "published indirect words must be recycled once as six storage-visible wave flags");
  assert.match(encode, /validateAirSupportFrontierReciprocity.*compactAirSupportSeedFrontier.*refreshRetainedAirSupportFaceValues.*finalizeRetainedAirSupportMarchSchedule.*broker\.fence\("Section5ordinary-faceseedspublished"\)/s);
  assert.match(encode, /broker\.fence\("Section5ordinary-faceseedspublished"\).*publish\(32,\[4\],"march","",false\).*clearBuffer\(this\.scratch,32\*4,6\*4\)/s,
    "the GPU reuse decision must replace the march schedule only across the seed-publication boundary it already rides");
  assert.match(encode,
    /indirectFrontierGate=changedFrontier&&octreeAirSupportIndirectFrontierGateEnabled\(\)&&fineSlot!==.*this\.inputs\.fineSources\[fineSlot\]\.plan\.fineFactor===1/,
    "only factor 1 may replace the factor-4/8 march graph with convergence-gated records");
  assert.match(encode,
    /if\(indirectFrontierGate\)publish\(10,\[0,1,2\],"march","",false\)/,
    "the initial factor-1 records need a storage-to-indirect visibility boundary");
  assert.match(encode,
    /advanceAirSupportChangedFrontier.*dispatchWorkgroupsIndirect\(indirectFor\(12\),12\).*publish\(10,\[0,1,2\],alternate\?"frontierB":"frontierA",[^)]*\).*marchAirSupportFacesChangedFrontier.*dispatchWorkgroupsIndirect\(indirectFor\(24\),24\)/s,
    "each proven singleton must gate all later wave phases, singletons, and the residual tail");
  assert.match(encode, /constalternate=directArgs&&\(wave&1\)===1/,
    "a wave must publish into the arena it is not itself dispatching from");
  assert.match(encode,
    /marchAirSupportFacesChangedFrontier.*dispatchWorkgroups\(3\).*encodeOctreeAirSupportReconstructionHandoff\(broker\)/s);
  assert.match(encode, /changedFrontier=octreeAirSupportChangedFrontierEnabled\(\)/,
    "the A/B decision must not alter construction or resource layout");
  assert.match(encode, /if\(changedFrontier\).*compactAirSupportSeedFrontier.*else\{for\(letwave=0;wave<OCTREE_AIR_SUPPORT_GPU_PARALLEL_MARCH_PREFIX/s);
  assert.match(encode,
    /constfrontierWaves=octreeAirSupportParallelFrontierWaves\(\);for\(letwave=0;wave<frontierWaves.*expandAirSupportChangedFrontier.*relaxAirSupportChangedFrontier.*commitAirSupportChangedFrontier.*advanceAirSupportChangedFrontier.*marchAirSupportFacesChangedFrontier/s,
    "the sparse default must batch useful frontier work across occupancy-wide dispatches before its exact residual tail");
  assert.equal(OCTREE_AIR_SUPPORT_GPU_PARALLEL_FRONTIER_WAVES, 12);
  assert.match(encode, /for\(letwave=0;wave<OCTREE_AIR_SUPPORT_GPU_WIDE_MARCH_WAVES.*advanceAirSupportMarchWave.*marchAirSupportFacesToFixedPoint/s);
  assert.equal(OCTREE_AIR_SUPPORT_GPU_PARALLEL_MARCH_PREFIX, 12);
  assert.equal(OCTREE_AIR_SUPPORT_GPU_WIDE_MARCH_WAVES, 12);
  assert.doesNotMatch(encode, /maximumRelaxationWaves|prepareAirSupportContinuation|for\(letlayer/,
    "the host must not encode a scene/domain propagation bound");
  const implementation = compact(readFileSync(
    new URL("../lib/webgpu-octree-air-velocity-support-gpu.ts", import.meta.url), "utf8"));
  assert.match(implementation, /WGSLhasnodevice-widebarrier.*last-workgrouplatch/,
    "the unsafe singleton-folding no-go must remain documented beside the retained boundary");
});

test("air-support pipelines can be deferred without parallel driver pressure", () => {
  const host = readFileSync(new URL("../lib/webgpu-octree-air-velocity-support-gpu.ts", import.meta.url), "utf8");
  const hostClass = host.slice(host.indexOf("export class WebGPUOctreeAirVelocitySupportProducer"),
    host.indexOf("export const octreeAirVelocitySupportPublicationWGSL"));
  assert.match(hostClass,
    /constructor\([\s\S]*_deferPipelineCompilation = true\)/,
    "direct construction must remain compilation-free while legacy callers migrate");
  assert.doesNotMatch(hostClass, /createPipelinesSync|createComputePipeline\s*\(/,
    "air-support must expose no synchronous pipeline fallback");
  const initialize = hostClass.slice(hostClass.indexOf("async initializePipelines()"),
    hostClass.indexOf("private parameterData("));
  assert.match(initialize,
    /for \(const entryPoint of this\.pipelineEntryPoints\(\)\) \{[\s\S]*await this\.device\.createComputePipelineAsync/,
    "air-support compilation must await one auto-layout pipeline at a time");
  assert.doesNotMatch(initialize, /Promise\.all/,
    "air-support compilation must not fan out driver work");
  assert.match(initialize, /this\.assignPipelineState\(pipelines\)/,
    "pipeline-dependent bind groups must be built only after the complete compile set");
  assert.match(hostClass,
    /this\.pipelines = Object\.freeze\(pipelines\);[\s\S]*this\.groups = groups;[\s\S]*this\.pipelinesInitialized = true/,
    "encode-visible pipeline and binding state must publish atomically at initialization completion");
});

// Every pipeline in this module is created with `layout: "auto"`, so the bind
// group layout is derived from each entry point's ACTUAL static usage, and
// `createBindGroup` rejects an entry set that does not match it exactly. The
// table above is therefore a DESCRIPTION of the shader, not a request: a
// binding declared but no longer referenced is a hard device error, not slack.
//
// It surfaces only once a GPU builds the group, which is why it reached the UI
// as "[Invalid BindGroup (unlabeled)] is invalid due to a previous error" —
// named after the pass that used the group, not the entry point that declared
// it — when `faceCenter` stopped reaching `faceCellIn` and left rowGeometry (2)
// and recordArena (8) dead on the three march entry points.
//
// `tests/webgpu-octree-spgrid-vcycle.test.ts` has held this same invariant over
// its own auto-layout module for some time; this producer simply never got it.
test("every auto-layout entry point declares exactly its reachable bindings", () => {
  for (const [entryPoint, declared] of Object.entries(OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS)) {
    const reachable = auditWGSLComputeBindingReachability(
      octreeAirVelocitySupportPublicationWGSL, entryPoint).bindings.map(({ binding }) => binding);
    assert.deepEqual([...declared].sort((left, right) => left - right),
      [...reachable].sort((left, right) => left - right),
      `${entryPoint} bind group must equal its exact WGSL auto-layout reachability`);
  }
});

test("producer exposes canonical banked row vectors and avoids double-counting a shared suffix arena", () => {
  const source = compact(Object.getOwnPropertyDescriptor(
    WebGPUOctreeAirVelocitySupportProducer.prototype, "source")!.get!.toString());
  assert.match(source, /canonicalRowVelocities:this\.inputs\.structured\.rowVelocities/);
  const implementation = compact(readFileSync(
    new URL("../lib/webgpu-octree-air-velocity-support-gpu.ts", import.meta.url), "utf8"));
  assert.match(implementation,
    /this\.allocatedBytes=this\.plan\.allocatedBytes\+256\+OCTREE_AIR_SUPPORT_GPU_DISPATCH_PHASE_NAMES\.length\*this\.plan\.indirectBytes-\(inputs\.sharedArena\?this\.plan\.support\.totalBytes:0\)/);
  assert.match(implementation,
    /this\.params=Object\.freeze\(\[0,1\]\.map.*constparameterSlot=this\.parameterSlot;this\.parameterSlot=parameterSlot===0\?1:0.*queue\.writeBuffer\(params,0,this\.parameterData/s,
    "two support transactions encoded into one command buffer must retain invocation-stable uniforms");
  assert.match(implementation,
    /expectedVectors=s\(30u\).*s\(28u\)==expectedVectors/s,
    "the face march must prove every exact demanded vector was reconstructed");
  assert.doesNotMatch(implementation,
    /maximumDisplacementFineCells\s*>\s*STRUCTURED_VELOCITY_EXTENSION_LAYERS/,
    "fine-cell backtrace distance and adaptive face-graph hops are unlike units");
});

// The portable per-compute-stage storage-buffer floor every WebGPU adapter must
// support, and the limit this producer's device is created with.
const M1_STORAGE_BUFFERS_PER_STAGE = 10;

// The Dawn check below is the authority, but it only runs when WEBGPU_NODE_MODULE
// is set -- which is precisely how an eleven-buffer entry point once shipped and
// only surfaced as a GPUPipelineError in the browser. `layout: "auto"` derives
// each pipeline's layout from static reachability, so the CPU auditor computes
// the same set the device would, and this assertion fails in the default suite
// before any GPU is involved. It deliberately budgets ONE spare buffer as well,
// so the next binding added to a nine-buffer entry point is a visible decision
// rather than a silent move onto the ceiling.
test("no Section 5 producer entry point reaches the portable storage-buffer ceiling", () => {
  const over: string[] = [];
  const atCeiling: string[] = [];
  for (const [entryPoint, declared] of Object.entries(OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS)) {
    const audit = auditWGSLComputeBindingReachability(
      octreeAirVelocitySupportPublicationWGSL, entryPoint);
    // The declared table is what createBindGroup is handed; prove it agrees on
    // the storage count too, not just on the binding numbers.
    const declaredStorage = declared.filter((binding) =>
      audit.storage.some((entry) => entry.binding === binding));
    assert.equal(declaredStorage.length, audit.storageCount,
      `${entryPoint} declares ${declaredStorage.length} storage buffers but reaches ${audit.storageCount}`);
    if (audit.storageCount > M1_STORAGE_BUFFERS_PER_STAGE) {
      over.push(`${entryPoint}=${audit.storageCount}`);
    } else if (audit.storageCount === M1_STORAGE_BUFFERS_PER_STAGE) {
      atCeiling.push(entryPoint);
    }
  }
  assert.deepEqual(over, [],
    `entry points exceed maxStorageBuffersPerShaderStage=${M1_STORAGE_BUFFERS_PER_STAGE}`);
  assert.deepEqual(atCeiling, [],
    "these entry points have no storage-buffer headroom left; consolidate before adding another binding");
});

test("Dawn compiles every Section 5 producer entry point at the M1 storage-buffer ceiling", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for WGSL validation",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  assert.ok(adapter.limits.maxStorageBuffersPerShaderStage >= 10);
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  const shaderModule = device.createShaderModule({ code: octreeAirVelocitySupportPublicationWGSL });
  const errors = (await shaderModule.getCompilationInfo()).messages.filter((message) => message.type === "error");
  assert.deepEqual(errors, []);
  device.pushErrorScope("validation");
  for (const entryPoint of Object.keys(OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS)) {
    device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint } });
  }
  const error = await device.popErrorScope(); assert.equal(error, null, error?.message);
  device.destroy();
});

test("GPU plan rejects malformed capacities and domain dimensions", () => {
  assert.throws(() => planOctreeAirVelocitySupportGPU(0, 30, [16, 16, 16]), /positive/);
  assert.throws(() => planOctreeAirVelocitySupportGPU(1, 30, [16, 0, 16]), /positive/);
});

// ---------------------------------------------------------------------------
// Direct STORAGE|INDIRECT dispatch arenas
// ---------------------------------------------------------------------------

test("direct indirect-args authorship defaults off and is a pure host-encoding decision", () => {
  assert.equal(OCTREE_AIR_SUPPORT_DIRECT_INDIRECT_ARGS_ENVIRONMENT,
    "FLUID_OCTREE_AIR_SUPPORT_DIRECT_INDIRECT_ARGS");
  assert.equal(octreeAirSupportDirectIndirectArgsEnabled({}), false);
  assert.equal(octreeAirSupportDirectIndirectArgsEnabled({
    FLUID_OCTREE_AIR_SUPPORT_DIRECT_INDIRECT_ARGS: "0",
  }), false);
  assert.equal(octreeAirSupportDirectIndirectArgsEnabled({
    FLUID_OCTREE_AIR_SUPPORT_DIRECT_INDIRECT_ARGS: "1",
  }), true);
  const implementation = compact(readFileSync(
    new URL("../lib/webgpu-octree-air-velocity-support-gpu.ts", import.meta.url), "utf8"));
  // Construction, layout and the shader must be identical in both arms, exactly
  // as `changedFrontier` is: only which buffer a dispatch reads may change.
  assert.match(implementation, /constdirectArgs=octreeAirSupportDirectIndirectArgsEnabled\(\);/,
    "the arm must be sampled once per encode, not read inside the WGSL or the constructor");
  const constructor = implementation.slice(
    implementation.indexOf("constructor(privatereadonlydevice"),
    implementation.indexOf("privatepipelineEntryPoints()"));
  assert.doesNotMatch(constructor, /octreeAirSupportDirectIndirectArgsEnabled/,
    "arenas must be allocated unconditionally so an A/B cannot move allocation");
  assert.doesNotMatch(compact(octreeAirVelocitySupportPublicationWGSL),
    /DIRECT_INDIRECT_ARGS|directArgs/,
    "the shader must publish both destinations unconditionally");
});

test("dispatch arenas carry STORAGE and INDIRECT, and the staged buffer stays INDIRECT-only", () => {
  const implementation = compact(readFileSync(
    new URL("../lib/webgpu-octree-air-velocity-support-gpu.ts", import.meta.url), "utf8"));
  // The whole point: a buffer with INDIRECT but not STORAGE cannot be written
  // by a compute shader, so its args have to be staged through a copy, and
  // every such copy is a forced compute-pass boundary.
  assert.match(implementation,
    /label:"Structuredair-supportindirectschedules",size:this\.plan\.indirectBytes,usage:GPUBufferUsage\.COPY_DST\|GPUBufferUsage\.INDIRECT/,
    "the staged arena must remain exactly as it was so the default arm is unchanged");
  assert.match(implementation,
    /usage:GPUBufferUsage\.STORAGE\|GPUBufferUsage\.INDIRECT\|GPUBufferUsage\.COPY_SRC,/,
    "every direct arena must be writable by its producing kernel and readable as INDIRECT");
  assert.match(implementation,
    /size:this\.plan\.indirectBytes,\s*usage:GPUBufferUsage\.STORAGE/,
    "an arena must keep the canonical nine-record layout so record offsets never move");
});

test("every schedule producer owns exactly one arena and no pass writes one it dispatches from", () => {
  const shader = compact(octreeAirVelocitySupportPublicationWGSL);
  // The set that calls writeDispatch, the set that declares binding 31, and the
  // phase table's key set are one set. Anything else is either an unpublished
  // schedule or a bind group the device would reject.
  const declares = Object.entries(OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS)
    .filter(([, bindings]) => (bindings as readonly number[])
      .includes(OCTREE_AIR_SUPPORT_GPU_DISPATCH_BINDING))
    .map(([entry]) => entry).sort();
  const authors = Object.keys(OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS).filter((entry) => {
    const body = shader.slice(shader.indexOf(`fn${entry}(`));
    return /^[\s\S]*?writeDispatch\(/.test(body.slice(0, body.indexOf("@compute") + 1 || undefined));
  }).sort();
  assert.deepEqual(Object.keys(OCTREE_AIR_SUPPORT_GPU_DISPATCH_PHASES).sort(), declares);
  assert.deepEqual(declares, authors,
    "binding 31 must be declared by exactly the entry points that publish a schedule");
  for (const entry of declares) {
    const audit = auditWGSLComputeBindingReachability(octreeAirVelocitySupportPublicationWGSL, entry);
    assert.ok(audit.bindings.some(({ binding }) =>
      binding === OCTREE_AIR_SUPPORT_GPU_DISPATCH_BINDING),
    `${entry} must reach the published dispatch arena`);
  }
  // Injective: two producers sharing an arena could collide across the pass
  // boundary that separates them, and a producer sharing an arena with a
  // consumer in its own pass is the STORAGE/INDIRECT conflict this splits.
  const phases = Object.values(OCTREE_AIR_SUPPORT_GPU_DISPATCH_PHASES);
  assert.equal(new Set(phases).size, phases.length,
    "each producing kernel must own its own arena");
  assert.deepEqual([...OCTREE_AIR_SUPPORT_GPU_DISPATCH_PHASE_NAMES].sort(),
    [...new Set([...phases, "frontierB"])].sort(),
    "the allocated arena set must be the phase set plus the frontier ping-pong partner");
  // One operand, two destinations, nothing recomputed: the direct record is the
  // same bits the copy would have moved.
  assert.match(shader,
    /fnwriteDispatch\(at:u32,record:u32,value:vec3u\)\{sw\(at,value\.x\);sw\(at\+1u,value\.y\);sw\(at\+2u,value\.z\);letbase=3u\*record;publishedDispatch\[base\]=value\.x;publishedDispatch\[base\+1u\]=value\.y;publishedDispatch\[base\+2u\]=value\.z;\}/,
    "the scratch words and the arena record must be written from one operand");
  // Two writers, both funnelled. A third -- or a conditional write inside one
  // of these producers with no republication arm -- is how an arena record goes
  // stale where the staged copy carried a retained value: the copy moved the
  // scratch word whether or not that dispatch had rewritten it.
  assert.equal((shader.match(/publishedDispatch\[/g) ?? []).length, 6,
    "publishedDispatch must be written only by writeDispatch and republishDispatch");
  assert.match(shader,
    /fnrepublishDispatch\(at:u32,record:u32\)\{letbase=3u\*record;publishedDispatch\[base\]=s\(at\);publishedDispatch\[base\+1u\]=s\(at\+1u\);publishedDispatch\[base\+2u\]=s\(at\+2u\);\}/,
    "a republication must read exactly the scratch record the staged copy read");
});

/**
 * Encode both arms against a recording command encoder.
 *
 * The producer's own resources are irrelevant here -- only which buffer each
 * indirect dispatch reads and which commands sit between the passes -- so the
 * instance is stubbed down to the fields `encode` actually touches. The broker
 * is the real one, so the boundary census is the production census.
 */
function encodeAirSupportArm(directArgs: boolean) {
  const previous = process.env[OCTREE_AIR_SUPPORT_DIRECT_INDIRECT_ARGS_ENVIRONMENT];
  process.env[OCTREE_AIR_SUPPORT_DIRECT_INDIRECT_ARGS_ENVIRONMENT] = directArgs ? "1" : "0";
  try {
    const named = (name: string) => ({ label: name } as unknown as GPUBuffer);
    const indirect = named("staged");
    const arenas = Object.fromEntries(OCTREE_AIR_SUPPORT_GPU_DISPATCH_PHASE_NAMES
      .map((phase) => [phase, named(phase)]));
    const indirectReads: string[] = [];
    const copies: string[] = [];
    const pass = {
      setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {},
      dispatchWorkgroupsIndirect(buffer: GPUBuffer, offset: number) {
        indirectReads.push(`${(buffer as unknown as { label: string }).label}@${offset}`);
      },
      end() {},
    };
    const encoder = {
      beginComputePass: () => pass,
      copyBufferToBuffer(_s: GPUBuffer, _so: number, destination: GPUBuffer,
        offset: number, size: number) {
        copies.push(`${(destination as unknown as { label: string }).label}@${offset}+${size}`);
      },
      clearBuffer() {},
      finish: () => ({}),
    } as unknown as GPUCommandEncoder;
    const broker = new PassBroker(encoder, { isolateLabels: false });
    const proxy = new Proxy({}, { get: (_t, key) => key }) as Record<string, never>;
    const producer = Object.assign(
      Object.create(WebGPUOctreeAirVelocitySupportProducer.prototype), {
        device: { queue: { writeBuffer() {} } },
        destroyed: false, pipelinesInitialized: true, publicationCount: 0, parameterSlot: 0,
        inputs: { fineSources: [{ plan: { fineFactor: 1 } }, { plan: { fineFactor: 1 } }] },
        params: [{}, {}], groups: [proxy, proxy], pipelines: proxy,
        fineDemandGroups: [[{}, {}], [{}, {}]], fineDemandScheduleGroups: [[{}, {}], [{}, {}]],
        frontierAlternateGroups: [{}, {}],
        scratch: named("scratch"), indirect, dispatchArenas: arenas,
        parameterData: () => new ArrayBuffer(256),
      }) as WebGPUOctreeAirVelocitySupportProducer;
    producer.encode(broker, 1, 0);
    broker.finish();
    const bucket = (reason: string) => broker.boundaryAudit.get(reason);
    return {
      indirectReads, copies,
      passes: broker.computePassCount,
      staged: bucket("stage indirect args")?.passClosures ?? 0,
      stagedCopies: bucket("stage indirect args")?.copyCommands ?? 0,
      stagedBytes: bucket("stage indirect args")?.commandBytes ?? 0,
      closures: [...broker.boundaryAudit.values()]
        .reduce((total, entry) => total + entry.passClosures, 0),
    };
  } finally {
    if (previous === undefined) delete process.env[OCTREE_AIR_SUPPORT_DIRECT_INDIRECT_ARGS_ENVIRONMENT];
    else process.env[OCTREE_AIR_SUPPORT_DIRECT_INDIRECT_ARGS_ENVIRONMENT] = previous;
  }
}

test("direct arenas delete every staging copy and change no pass boundary", () => {
  const staged = encodeAirSupportArm(false);
  const direct = encodeAirSupportArm(true);
  // What the conversion buys: the copies.
  assert.ok(staged.stagedCopies > 0);
  assert.equal(direct.stagedCopies, 0, "no schedule may still travel by copy");
  assert.equal(direct.copies.length, 0, "the direct arm must encode no buffer copy at all");
  assert.equal(direct.staged, 0);
  // What it does NOT buy, and the reason it is worth saying out loud: every one
  // of those copies sat exactly where a producing kernel hands a schedule to a
  // consumer that dispatches from it. That hand-off needs the boundary whether
  // the args travel by copy or are authored in place, so the closure count is
  // identical and only the reason labels move.
  assert.equal(direct.closures, staged.closures,
    "converting authorship must not silently add or remove a barrier");
  assert.equal(direct.passes, staged.passes);
  // Every record is read from the arena its producer authored it into, at the
  // canonical offset the staged arm copied it to.
  assert.deepEqual([...new Set(staged.indirectReads.map((read) => read.split("@")[0]))], ["staged"]);
  const byOffset = new Map<string, Set<string>>();
  for (const read of direct.indirectReads) {
    const [arena, offset] = read.split("@") as [string, string];
    byOffset.set(offset, (byOffset.get(offset) ?? new Set()).add(arena));
  }
  assert.deepEqual(Object.fromEntries([...byOffset]
    .sort((left, right) => Number(left[0]) - Number(right[0]))
    .map(([offset, arenas]) => [offset, [...arenas]])), {
    0: ["identity", "march", "frontierA", "frontierB"],
    12: ["identity", "march", "frontierA", "frontierB"],
    24: ["identity", "frontierB"],
    36: ["identity", "candidates"],
    48: ["fineDemand", "face", "march"],
    60: ["support", "face"],
    72: ["fineClosure", "radix"],
    84: ["fineEmission"],
    96: ["identity", "support"],
  });
  assert.ok(!direct.indirectReads.includes("staged@0"),
    "no dispatch may fall back to the staged arena");
});
